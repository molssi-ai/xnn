"""Self-contained TorchScript export: a trained model as a standalone ``.pt``.

The artifact produced here is driven purely by tensors and carries its own
neighbor list, so a consumer needs only ``libtorch`` / ``torch.jit.load`` --
no ``xnn`` import, no Python model code, no config file. That is what makes a
checkpoint usable from LAMMPS, i-PI, OpenMM, a C++ driver or any other MD
package.

Two entry points are exported on the same module:

``forward(pos, atomic_numbers, cell, pbc)``
    The *whole-system* ABI. The module builds its own neighbor list from the
    cutoff baked into it at export time. This is the general-purpose entry
    point and the only correct one for models carrying a long-range term.

``forward_lammps(pos, edge_index, cell_shifts, atomic_numbers, cell)``
    The *pair-style* ABI, matching :class:`~xnn.common.deploy.LAMMPSWrapper`
    (and hence pair_nequip / pair_mace / pair_allegro): the caller supplies the
    neighbor list. Cheaper, because the MD engine already has a neighbor list,
    but see the long-range caveat below.

Calling conventions
-------------------
One structure per call (inputs are ``(N, 3)``, with no batch dimension).
Positions may be float32 or float64 whatever the weights' dtype: the module
computes in its own dtype and answers in the caller's. Wrapping the call in
``torch.no_grad()`` is fine -- forces come from autograd, so the module
re-enables grad internally and restores the caller's mode afterwards --  but
``torch.inference_mode()`` raises, because tensors created under it can never
participate in autograd.

Long-range (LES) models
-----------------------
:class:`~xnn.common.models.les.LatentEwald` adds an Ewald energy over latent
charges, which is a **global** sum: every atom's latent charge enters, with no
cutoff. It therefore does not decompose into a local, per-domain neighbor list.
A model exported with a long-range term must be driven with the whole system on
one rank (``forward``, or ``forward_lammps`` with a full-system neighbor list) --
an MPI-decomposed pair style that only ever sees its own subdomain plus ghosts
cannot reproduce the trained energy. :func:`export_torchscript_potential` records
``long_range`` in the archive metadata so a consumer can check.
"""
# NOTE: deliberately no ``from __future__ import annotations`` -- it turns the
# class-level attribute annotations below into strings, which TorchScript's
# annotation resolver rejects ("Unknown type annotation: 'bool'").
import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn


def _image_shifts(cell: Tensor, cutoff: float, pbc: Tensor) -> Tensor:
    """Integer lattice-image offsets covering ``cutoff`` along periodic axes.

    Scriptable counterpart of
    :func:`~xnn.common.data.neighborlist._n_repeats` plus the
    ``itertools.product`` enumeration that follows it: the number of repeats
    along an axis comes from the interplanar spacing ``1 / |reciprocal_i|``,
    and non-periodic axes get zero repeats.

    Parameters
    ----------
    cell : Tensor
        Lattice vectors as rows, shape ``(3, 3)``.
    cutoff : float
        Neighbor cutoff radius.
    pbc : Tensor
        Boolean periodicity flags, shape ``(3,)``.

    Returns
    -------
    Tensor
        Integer shift triples, shape ``(S, 3)``, last axis varying fastest
        (the same ordering ``itertools.product`` produces).
    """
    device = cell.device
    recip = torch.linalg.inv(cell).t()          # rows are reciprocal vectors
    spacing = 1.0 / torch.linalg.norm(recip, dim=1)
    reps: List[int] = []
    for i in range(3):
        if bool(pbc[i]):
            reps.append(int(math.ceil(cutoff / float(spacing[i]))))
        else:
            reps.append(0)
    ra = torch.arange(-reps[0], reps[0] + 1, device=device)
    rb = torch.arange(-reps[1], reps[1] + 1, device=device)
    rc = torch.arange(-reps[2], reps[2] + 1, device=device)
    return torch.cartesian_prod(ra, rb, rc)


def build_neighbor_list_ts(pos: Tensor, cutoff: float, cell: Tensor,
                           pbc: Tensor) -> Tuple[Tensor, Tensor]:
    """``torch.jit``-able neighbor list, matching the reference implementation.

    Semantically identical to
    :func:`~xnn.common.data.build_neighbor_list` (same ``dst = i`` /
    ``src = j`` convention, same negated ``cell_shifts``, same wrapping of
    out-of-cell positions with the removed image offsets folded back into the
    shifts), rewritten without ``itertools`` and without ``.tolist()`` so it
    compiles under TorchScript and can live inside the exported artifact.

    A structure is treated as molecular when no periodic flag is set or the
    cell is all zeros, in which case image enumeration is skipped.

    Parameters
    ----------
    pos : Tensor
        Cartesian positions, shape ``(N, 3)``.
    cutoff : float
        Neighbor cutoff radius.
    cell : Tensor
        Lattice vectors as rows, shape ``(3, 3)``; all-zero for a molecule.
    pbc : Tensor
        Boolean periodicity flags, shape ``(3,)``.

    Returns
    -------
    edge_index : Tensor
        Edge list ``[src, dst]``, shape ``(2, E)``.
    cell_shifts : Tensor
        Integer image shift per edge, shape ``(E, 3)``.

    Notes
    -----
    Brute force over ``(S, N, N)`` pairs, like the reference implementation it
    mirrors; memory grows as ``O(S N^2)``. Fine for the molecular and modest
    periodic systems this export targets, but for large cells supply the
    neighbor list from the MD engine via ``forward_lammps`` instead.
    """
    device = pos.device
    dtype = pos.dtype
    n = pos.shape[0]

    offsets = torch.zeros((n, 3), dtype=torch.long, device=device)
    shift_idx = torch.zeros((1, 3), dtype=torch.long, device=device)
    shifts = torch.zeros((1, 3), dtype=dtype, device=device)
    work = pos

    periodic = (bool(pbc.any())
                and bool(torch.linalg.norm(cell, dim=1).sum() > 1e-8))
    if periodic:
        # wrap into the cell along periodic axes; the integer offsets removed
        # here are added back to the shifts below, so the result stays
        # consistent with the original (possibly unwrapped) positions
        frac = pos @ torch.linalg.inv(cell)
        raw = torch.floor(frac).to(torch.long)
        offsets = torch.where(pbc.unsqueeze(0), raw, torch.zeros_like(raw))
        work = pos - offsets.to(dtype) @ cell
        shift_idx = _image_shifts(cell, cutoff, pbc)
        shifts = shift_idx.to(dtype) @ cell

    # (S, N, N, 3): entry [s, i, j] is pos[j] - pos[i] + shift[s]
    rij = (work.unsqueeze(0).unsqueeze(0) - work.unsqueeze(1).unsqueeze(0)
           + shifts.unsqueeze(1).unsqueeze(1))
    dist = torch.linalg.norm(rij, dim=-1)

    within = dist < cutoff
    eye = torch.eye(n, dtype=torch.bool, device=device)
    zero_shift = (shift_idx == 0).all(dim=1)
    within = within & ~(eye.unsqueeze(0)
                        & zero_shift.unsqueeze(1).unsqueeze(1))

    nz = torch.nonzero(within)
    s_idx = nz[:, 0]
    i_idx = nz[:, 1]
    j_idx = nz[:, 2]
    edge_index = torch.stack([j_idx, i_idx], dim=0)
    # negated shift so that pos[dst] - pos[src] + cell_shift @ cell reproduces
    # the selecting displacement; offsets undo the wrapping done above
    cell_shifts = -shift_idx[s_idx] + offsets[j_idx] - offsets[i_idx]
    return edge_index, cell_shifts


class _NoLongRange(nn.Module):
    """Null long-range head for models exported without LES.

    Present so :class:`TorchScriptPotential` has one code path; returns a zero
    energy and an empty latent-charge tensor.
    """

    def forward(self, features: Tensor, pos: Tensor, cell: Tensor,
                pbc: Tensor) -> Tuple[Tensor, Tensor]:
        """Return ``(0, empty)`` with the dtype and device of ``pos``."""
        energy = torch.zeros((), dtype=pos.dtype, device=pos.device)
        charges = torch.zeros((pos.shape[0], 0), dtype=pos.dtype,
                              device=pos.device)
        return energy, charges


class _LatentEwaldHead(nn.Module):
    """Scriptable single-structure view of a trained :class:`LatentEwald`.

    Holds the *trained* submodules directly (``q_net``, the optional parallel
    ``q_linear`` and the :class:`~xnn.common.models.les.EwaldSummation`), so
    the deployed long-range energy is computed by the same code as training --
    only the batched, ``AtomicGraph``-shaped entry point is replaced by a
    single-structure one.

    Parameters
    ----------
    les : xnn.common.models.les.LatentEwald
        The trained wrapper to take the long-range head from.

    Attributes
    ----------
    q_net : torch.nn.Module
        The latent-charge MLP.
    q_linear : torch.nn.Module
        The parallel bias-free linear layer; a dummy when the trained model had
        none (see ``use_linear``).
    use_linear : bool
        Whether ``q_linear`` participates.
    ewald : xnn.common.models.les.EwaldSummation
        The long-range energy module.
    """

    use_linear: bool

    def __init__(self, les):
        super().__init__()
        self.q_net = les.q_net
        self.use_linear = les.q_linear is not None
        if les.q_linear is not None:
            self.q_linear = les.q_linear
        else:
            # placeholder: TorchScript needs the attribute to have a concrete
            # module type, but use_linear gates every call to it
            self.q_linear = nn.Linear(1, 1, bias=False)
        self.ewald = les.ewald

    def forward(self, features: Tensor, pos: Tensor, cell: Tensor,
                pbc: Tensor) -> Tuple[Tensor, Tensor]:
        """Long-range energy and latent charges for one structure.

        Selects the reciprocal-space sum for a periodic structure and the
        real-space direct sum otherwise, using the same test as
        :meth:`~xnn.common.models.les.EwaldSummation.forward`.

        Parameters
        ----------
        features : Tensor
            Invariant per-atom features, shape ``(N, feature_dim)``.
        pos : Tensor
            Cartesian positions, shape ``(N, 3)``.
        cell : Tensor
            Lattice vectors as rows, shape ``(3, 3)``; all-zero for a molecule.
        pbc : Tensor
            Boolean periodicity flags, shape ``(3,)``.

        Returns
        -------
        energy : Tensor
            Scalar long-range energy.
        charges : Tensor
            Latent charges ``q``, shape ``(N, n_channels)``.
        """
        q = self.q_net(features)
        if self.use_linear:
            q = q + self.q_linear(features)
        periodic = (bool(pbc.any())
                    and bool(cell.diagonal().abs().sum() > 1e-6))
        if periodic:
            energy = self.ewald.reciprocal(pos, q, cell)
        else:
            energy = self.ewald.realspace(pos, q)
        return energy, q


class TorchScriptPotential(nn.Module):
    """Tensor-only, scriptable potential: positions in, energy/forces/stress out.

    Wraps a trained model (optionally
    :class:`~xnn.common.models.les.LatentEwald`-wrapped) behind a fixed tensor
    ABI with no ``AtomicGraph`` and no Python-only constructs, so
    ``torch.jit.script`` produces a portable artifact. Forces come from
    ``-dE/dr`` and the stress from the symmetric-strain trick, matching
    :class:`~xnn.common.models.outputs.ForceStressOutput`.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model. Must expose the scriptable tensor core
        ``node_features_energy(atomic_numbers, edge_index, edge_vec)``; a
        :class:`LatentEwald` wrapper is unwrapped automatically and its
        long-range head carried over.
    cutoff : float
        Neighbor-list cutoff, baked in so :meth:`forward` is self-contained.

    Attributes
    ----------
    model : torch.nn.Module
        The short-range core.
    lr : torch.nn.Module
        The long-range head (:class:`_LatentEwaldHead` or :class:`_NoLongRange`).
    cutoff : float
        The neighbor-list cutoff radius.
    has_long_range : bool
        Whether a long-range term is present.
    """

    cutoff: float
    has_long_range: bool

    def __init__(self, model: nn.Module, cutoff: float):
        super().__init__()
        from ..models.les import LatentEwald
        if isinstance(model, LatentEwald):
            core: nn.Module = model.model
            head: nn.Module = _LatentEwaldHead(model)
            self.has_long_range = True
        else:
            core = model
            head = _NoLongRange()
            self.has_long_range = False
        if not hasattr(core, "node_features_energy"):
            raise TypeError(
                f"{type(core).__name__} has no scriptable "
                "'node_features_energy(atomic_numbers, edge_index, edge_vec)' "
                "core, which the TorchScript export requires")
        self.model = core
        self.lr = head
        self.cutoff = float(cutoff)
        # zero-element buffer used only to read back the module's working
        # dtype; seeded from a real parameter (not the ambient default dtype,
        # which need not match the trained weights) and thereafter follows
        # .float() / .double() / .to() like any other buffer
        param_dtype = torch.get_default_dtype()
        for param in self.parameters():
            if param.is_floating_point():
                param_dtype = param.dtype
                break
        self.register_buffer("_dtype_probe",
                             torch.zeros(0, dtype=param_dtype))

    def _evaluate(self, pos: Tensor, atomic_numbers: Tensor,
                  edge_index: Tensor, cell_shifts: Tensor, cell: Tensor,
                  pbc: Tensor) -> Dict[str, Tensor]:
        """Energy, forces and stress for one structure and its neighbor list.

        Positions and cell are displaced by a zero-valued symmetric strain so
        that a single backward pass yields both ``-dE/dr`` and ``dE/deps``.

        Parameters
        ----------
        pos : Tensor
            Cartesian positions, shape ``(N, 3)``.
        atomic_numbers : Tensor
            Per-atom atomic numbers, shape ``(N,)``.
        edge_index : Tensor
            Edge list ``[src, dst]``, shape ``(2, E)``.
        cell_shifts : Tensor
            Integer image shift per edge, shape ``(E, 3)``.
        cell : Tensor
            Lattice vectors as rows, shape ``(3, 3)``.
        pbc : Tensor
            Boolean periodicity flags, shape ``(3,)``.

        Returns
        -------
        dict of str to Tensor
            See :meth:`forward`.
        """
        # compute in the module's own dtype but answer in the caller's, so a
        # driver holding float64 positions (LAMMPS does) can call this without
        # converting anything by hand
        out_dtype = pos.dtype
        dtype = self._dtype_probe.dtype
        pos = pos.to(dtype)
        cell = cell.to(dtype)
        # Forces and stress come from autograd, so the physics must run with
        # grad enabled even when the caller wraps the call in torch.no_grad()
        # -- which MD drivers and ordinary inference loops routinely do. The
        # caller's grad mode is restored as soon as the gradients are in hand.
        # (torch.enable_grad() as a context manager is not scriptable; this
        # save/set/restore is the supported equivalent.)
        prev_grad = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        device = pos.device
        p = pos.detach().requires_grad_(True)
        # (requires_grad set after construction: TorchScript's torch.zeros
        # has no requires_grad keyword)
        strain = torch.zeros((3, 3), dtype=dtype, device=device)
        strain.requires_grad_(True)
        sym = 0.5 * (strain + strain.t())
        p_s = p + torch.mm(p, sym)
        cell_s = cell + torch.mm(cell, sym)

        src = edge_index[0]
        dst = edge_index[1]
        edge_vec = (p_s.index_select(0, dst) - p_s.index_select(0, src)
                    + torch.mm(cell_shifts.to(dtype), cell_s))

        out = self.model.node_features_energy(atomic_numbers, edge_index,
                                              edge_vec)
        features = out[0]
        node_energy = out[1]
        energy_sr = node_energy.sum()
        energy_lr, charges = self.lr(features, p_s, cell_s, pbc)
        energy = energy_sr + energy_lr

        # allow_unused: a purely reference-energy model has no position or
        # strain dependence, and a molecular system has no strain dependence
        grads = torch.autograd.grad([energy], [p, strain], create_graph=False,
                                    allow_unused=True)
        torch.set_grad_enabled(prev_grad)
        g_pos = grads[0]
        g_strain = grads[1]
        forces = torch.zeros_like(pos)
        if g_pos is not None:
            forces = -g_pos

        volume = torch.det(cell).abs()
        stress = torch.zeros((3, 3), dtype=dtype, device=device)
        if g_strain is not None and bool(volume > 1e-8):
            stress = g_strain / volume

        # spread the long-range energy over the atoms, as LatentEwald does, so
        # the per-atom energies still sum to the total
        n_atoms = float(pos.shape[0])
        per_atom = node_energy + energy_lr / n_atoms
        result = {
            "energy": energy.reshape(1),
            "total_energy": energy.reshape(1),
            "energy_sr": energy_sr.reshape(1),
            "energy_lr": energy_lr.reshape(1),
            "node_energy": per_atom,
            "forces": forces,
            "stress": stress,
            "virial": -stress * volume,
            "latent_charges": charges,
        }
        if out_dtype != dtype:
            converted: Dict[str, Tensor] = {}
            for key, value in result.items():
                converted[key] = value.to(out_dtype)
            return converted
        return result

    def forward(self, pos: Tensor, atomic_numbers: Tensor,
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Whole-system entry point; builds its own neighbor list.

        The general-purpose ABI, and the only correct one when the model
        carries a long-range term (see the module docstring).

        Parameters
        ----------
        pos : Tensor
            Cartesian positions, shape ``(N, 3)``.
        atomic_numbers : Tensor
            Per-atom atomic numbers, shape ``(N,)``.
        cell : Tensor or None, optional
            Lattice vectors as rows, shape ``(3, 3)``. ``None`` (the default)
            or all-zero means a molecular system.
        pbc : Tensor or None, optional
            Boolean periodicity flags, shape ``(3,)``. ``None`` (the default)
            means non-periodic.

        Returns
        -------
        dict of str to Tensor
            ``energy`` / ``total_energy`` ``(1,)``, ``energy_sr`` ``(1,)``,
            ``energy_lr`` ``(1,)``, ``node_energy`` ``(N,)``, ``forces``
            ``(N, 3)``, ``stress`` ``(3, 3)``, ``virial`` ``(3, 3)`` and
            ``latent_charges`` ``(N, n_channels)`` (empty without LES).
        """
        dtype = pos.dtype
        device = pos.device
        if cell is None:
            cell_t = torch.zeros((3, 3), dtype=dtype, device=device)
        else:
            cell_t = cell.to(dtype)
        if pbc is None:
            pbc_t = torch.zeros(3, dtype=torch.bool, device=device)
        else:
            pbc_t = pbc.to(torch.bool)

        with torch.no_grad():
            nl = build_neighbor_list_ts(pos.detach(), self.cutoff, cell_t,
                                        pbc_t)
        return self._evaluate(pos, atomic_numbers, nl[0], nl[1], cell_t, pbc_t)

    @torch.jit.export
    def forward_lammps(self, pos: Tensor, edge_index: Tensor,
                       cell_shifts: Tensor, atomic_numbers: Tensor,
                       cell: Tensor) -> Dict[str, Tensor]:
        """Pair-style entry point; the caller supplies the neighbor list.

        Argument order matches :class:`~xnn.common.deploy.LAMMPSWrapper`, so
        an existing pair style needs no change. Periodicity is inferred from
        ``cell`` being nonzero.

        Parameters
        ----------
        pos : Tensor
            Cartesian positions, shape ``(N, 3)``.
        edge_index : Tensor
            Edge list ``[src, dst]``, shape ``(2, E)``.
        cell_shifts : Tensor
            Integer image shift per edge, shape ``(E, 3)``.
        atomic_numbers : Tensor
            Per-atom atomic numbers, shape ``(N,)``.
        cell : Tensor
            Lattice vectors as rows, shape ``(3, 3)``.

        Returns
        -------
        dict of str to Tensor
            The same keys as :meth:`forward`.

        Notes
        -----
        For a model with a long-range term the supplied neighbor list must
        cover the whole system on a single rank; a subdomain-local list does
        not reproduce the trained energy.
        """
        periodic = bool(torch.linalg.norm(cell, dim=1).sum() > 1e-8)
        pbc_t = torch.full((3,), periodic, dtype=torch.bool,
                           device=pos.device)
        return self._evaluate(pos, atomic_numbers, edge_index, cell_shifts,
                              cell.to(pos.dtype), pbc_t)


def export_torchscript_potential(model: nn.Module, cutoff: float, path: str,
                                 metadata: Optional[dict] = None) -> str:
    """Script a trained model to a standalone ``.pt`` and save it.

    Wraps ``model`` in :class:`TorchScriptPotential`, compiles it with
    ``torch.jit.script`` and writes the archive. The cutoff and a
    ``long_range`` flag are embedded as extra files, alongside any caller
    metadata, so a consumer can introspect the artifact without ``xnn``.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model (optionally LES-wrapped).
    cutoff : float
        Neighbor-list cutoff radius to bake in.
    path : str
        Destination ``.pt`` path.
    metadata : dict or None, optional
        Extra key/value metadata; values are stringified.

    Returns
    -------
    str
        The ``path`` written.
    """
    wrapper = TorchScriptPotential(model, cutoff).eval()
    scripted = torch.jit.script(wrapper)
    extra = {"cutoff": str(cutoff),
             "long_range": str(wrapper.has_long_range)}
    if metadata:
        extra.update({k: str(v) for k, v in metadata.items()})
    scripted.save(path, _extra_files=extra)
    return path
