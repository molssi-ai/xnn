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

Add-on terms: LES long-range and D4 dispersion
----------------------------------------------
Two wrappers can sit on top of a core model, in either order or together:
:class:`~xnn.common.models.les.LatentEwald` (an Ewald energy over latent
charges) and :class:`~xnn.common.models.dispersion.DispersionCorrection` (the DFT-D3 /
DFT-D4 corrections; D4's EEQ charges couple every atom of the structure
through a dense linear system). Both are **global**: they do not decompose into a
local, per-domain neighbor list. A model exported with either must be driven
with the whole system on one rank (``forward``, or ``forward_lammps`` with a
full-system neighbor list) -- an MPI-decomposed pair style that only ever sees
its own subdomain plus ghosts cannot reproduce the trained energy.
:func:`export_torchscript_potential` records ``long_range`` and ``dispersion``
in the archive metadata so a consumer can check.

The dispersion wrapper also widens the neighbor list (its cutoffs default to the
upstream 60 / 40 / 30 / 25 bohr): the export bakes in the wrapper's cutoff and
hands the core model only the edges within the core's own radius, exactly as
the eager wrapper does. The total charge of the system is fixed at export
time (``total_charge``, default neutral), since neither tensor ABI carries it.
"""
# NOTE: deliberately no ``from __future__ import annotations`` -- it turns the
# class-level attribute annotations below into strings, which TorchScript's
# annotation resolver rejects ("Unknown type annotation: 'bool'").
import copy
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
    per-atom energy and an empty latent-charge tensor.
    """

    def forward(self, features: Tensor, atomic_numbers: Tensor, pos: Tensor,
                cell: Tensor, pbc: Tensor, edge_index: Tensor,
                edge_vec: Tensor) -> Tuple[Tensor, Tensor]:
        """Return ``(zeros(N), empty(N, 0))`` with the dtype and device of ``pos``."""
        n = pos.shape[0]
        return (torch.zeros(n, dtype=pos.dtype, device=pos.device),
                torch.zeros((n, 0), dtype=pos.dtype, device=pos.device))


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

    def forward(self, features: Tensor, atomic_numbers: Tensor, pos: Tensor,
                cell: Tensor, pbc: Tensor, edge_index: Tensor,
                edge_vec: Tensor) -> Tuple[Tensor, Tensor]:
        """Long-range energy (spread over the atoms) and latent charges.

        Selects the reciprocal-space sum for a periodic structure and the
        real-space direct sum otherwise, using the same test as
        :meth:`~xnn.common.models.les.EwaldSummation.forward`. The energy is
        divided evenly over the atoms, as :class:`LatentEwald` does, so the
        per-atom energies still sum to the total.

        Parameters
        ----------
        features : Tensor
            Invariant per-atom features, shape ``(N, feature_dim)``.
        atomic_numbers : Tensor
            Per-atom atomic numbers, shape ``(N,)`` (unused).
        pos : Tensor
            Cartesian positions, shape ``(N, 3)``.
        cell : Tensor
            Lattice vectors as rows, shape ``(3, 3)``; all-zero for a molecule.
        pbc : Tensor
            Boolean periodicity flags, shape ``(3,)``.
        edge_index, edge_vec : Tensor
            The neighbor list (unused; LES needs no cutoff).

        Returns
        -------
        node_energy : Tensor
            Per-atom share of the long-range energy, shape ``(N,)``.
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
        n_atoms = float(pos.shape[0])
        return energy / n_atoms + torch.zeros(pos.shape[0], dtype=pos.dtype,
                                              device=pos.device), q


class _NoDispersion(nn.Module):
    """Null dispersion head for models exported without D4."""

    def forward(self, features: Tensor, atomic_numbers: Tensor, pos: Tensor,
                cell: Tensor, pbc: Tensor, edge_index: Tensor,
                edge_vec: Tensor) -> Tuple[Tensor, Tensor]:
        """Return ``(zeros(N), empty(N, 0))`` with the dtype and device of ``pos``."""
        n = pos.shape[0]
        return (torch.zeros(n, dtype=pos.dtype, device=pos.device),
                torch.zeros((n, 0), dtype=pos.dtype, device=pos.device))


class _DispersionHead(nn.Module):
    """Scriptable single-structure view of a :class:`DispersionCorrection` wrapper.

    Holds a copy of the wrapper's evaluator (:class:`~xnn.common.models.d4.DFTD4`
    or :class:`~xnn.common.models.d3.DFTD3`; same parameters, so a trained
    damping parameter deploys as trained) pinned to its scriptable paths --
    the dense EEQ regime and the plain three-body loop -- and calls its
    batched core for one structure.

    Parameters
    ----------
    term : torch.nn.Module
        The dispersion evaluator.
    total_charge : float
        Net charge of the deployed system (the tensor ABIs carry none; D4
        reads it, D3 ignores it).

    Attributes
    ----------
    term : torch.nn.Module
        The dispersion evaluator.
    total_charge : float
        The fixed net charge.
    """

    total_charge: float

    def __init__(self, term, total_charge: float = 0.0):
        super().__init__()
        # a copy: the export pins the scriptable paths (dense EEQ, plain
        # triplet loop) without touching the eager model's settings
        term = copy.deepcopy(term)
        if hasattr(term, "regime"):
            term.regime = "dense"
        if hasattr(term, "checkpoint_triplets"):
            term.checkpoint_triplets = False
        self.term = term
        self.total_charge = float(total_charge)

    def forward(self, features: Tensor, atomic_numbers: Tensor, pos: Tensor,
                cell: Tensor, pbc: Tensor, edge_index: Tensor,
                edge_vec: Tensor) -> Tuple[Tensor, Tensor]:
        """Per-atom dispersion energy (eV) and the term's charges ``(N, 1)`` (D4) or ``(N, 0)``.

        The neighbor list must reach the dispersion cutoff (the export bakes
        it in). A nonzero ``cell`` with any ``pbc`` flag selects the periodic
        (Ewald-summed) EEQ charges of D4.
        """
        n = pos.shape[0]
        batch = torch.zeros(n, dtype=torch.long, device=pos.device)
        charge = torch.full((1,), self.total_charge, dtype=pos.dtype, device=pos.device)
        out = self.term.evaluate(atomic_numbers, pos, edge_index, edge_vec, batch, 1,
                                 cell.unsqueeze(0), pbc.unsqueeze(0), charge)
        if "charges" in out:
            aux = out["charges"].unsqueeze(1)
        else:
            aux = torch.zeros((n, 0), dtype=pos.dtype, device=pos.device)
        return out["node_energy"], aux


class _ZeroCore(nn.Module):
    """Core stand-in for a standalone (model-free) :class:`D4Dispersion`.

    Provides the scriptable ``node_features_energy`` ABI with zero energies
    and zero-width features, so pure D4 deploys through the same wrapper.
    """

    cutoff: float

    def __init__(self, cutoff: float):
        super().__init__()
        self.cutoff = float(cutoff)

    def node_features_energy(self, atomic_numbers: Tensor, edge_index: Tensor,
                             edge_vec: Tensor) -> Tuple[Tensor, Tensor]:
        """Return ``(empty(N, 0), zeros(N))``."""
        n = atomic_numbers.shape[0]
        return (torch.zeros((n, 0), dtype=edge_vec.dtype, device=edge_vec.device),
                torch.zeros(n, dtype=edge_vec.dtype, device=edge_vec.device))

    def node_energy(self, atomic_numbers: Tensor, edge_index: Tensor,
                    edge_vec: Tensor) -> Tensor:
        """Return ``zeros(N)``."""
        return torch.zeros(atomic_numbers.shape[0], dtype=edge_vec.dtype,
                           device=edge_vec.device)


def split_wrappers(model: nn.Module, total_charge: float = 0.0):
    """Peel the LES / D4 wrappers off a model.

    Parameters
    ----------
    model : torch.nn.Module
        A core model, optionally wrapped in
        :class:`~xnn.common.models.les.LatentEwald` and/or
        :class:`~xnn.common.models.d4.D4Dispersion` (any nesting order).
    total_charge : float, optional
        Net charge handed to the D4 head, by default 0.

    Returns
    -------
    core : torch.nn.Module
        The innermost model (a :class:`_ZeroCore` for a standalone D4).
    long_range : torch.nn.Module
        :class:`_LatentEwaldHead` or :class:`_NoLongRange`.
    dispersion : torch.nn.Module
        :class:`_DispersionHead` or :class:`_NoDispersion`.
    core_cutoff : float
        The core model's neighbor-list radius (edges beyond it are filtered
        out before the core runs).

    Raises
    ------
    TypeError
        If LES sits directly on a standalone D4 model (its features are the
        D4 per-atom quantities, which the tensor ABI does not reproduce).
    """
    from ..models.dispersion import DispersionCorrection
    from ..models.les import LatentEwald
    core: nn.Module = model
    long_range: nn.Module = _NoLongRange()
    dispersion: nn.Module = _NoDispersion()
    while True:
        if isinstance(core, LatentEwald):
            long_range = _LatentEwaldHead(core)
            core = core.model
        elif isinstance(core, DispersionCorrection):
            dispersion = _DispersionHead(core.term, total_charge)
            if core.model is None:
                if isinstance(long_range, _LatentEwaldHead):
                    raise TypeError("LES on top of a standalone D4 model is not "
                                    "exportable: its features are the D4 "
                                    "per-atom quantities")
                core = _ZeroCore(core.cutoff)
                break
            core = core.model
        else:
            break
    return core, long_range, dispersion, float(getattr(core, "cutoff", 0.0))


class TorchScriptPotential(nn.Module):
    """Tensor-only, scriptable potential: positions in, energy/forces/stress out.

    Wraps a trained model (optionally
    :class:`~xnn.common.models.les.LatentEwald`- and/or
    :class:`~xnn.common.models.d4.D4Dispersion`-wrapped) behind a fixed tensor
    ABI with no ``AtomicGraph`` and no Python-only constructs, so
    ``torch.jit.script`` produces a portable artifact. Forces come from
    ``-dE/dr`` and the stress from the symmetric-strain trick, matching
    :class:`~xnn.common.models.outputs.ForceStressOutput`.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model. Its core must expose the scriptable tensor core
        ``node_features_energy(atomic_numbers, edge_index, edge_vec)``; LES
        and D4 wrappers are unwrapped automatically and carried over as heads.
    cutoff : float
        Neighbor-list cutoff, baked in so :meth:`forward` is self-contained.
        For a D4-wrapped model this is the wrapper's (widened) cutoff; the core
        receives only the edges within its own radius.
    total_charge : float, optional
        Net charge of the deployed system, used by the D4 EEQ charges; by
        default 0 (neutral).

    Attributes
    ----------
    model : torch.nn.Module
        The short-range core.
    lr : torch.nn.Module
        The long-range head (:class:`_LatentEwaldHead` or :class:`_NoLongRange`).
    disp : torch.nn.Module
        The dispersion head (:class:`_DispersionHead` or :class:`_NoDispersion`).
    cutoff : float
        The neighbor-list cutoff radius.
    core_cutoff : float
        The core model's own radius (``<= cutoff``).
    has_long_range : bool
        Whether a long-range term is present.
    has_dispersion : bool
        Whether a D4 dispersion term is present.
    """

    cutoff: float
    core_cutoff: float
    has_long_range: bool
    has_dispersion: bool

    def __init__(self, model: nn.Module, cutoff: float, total_charge: float = 0.0):
        super().__init__()
        core, head, disp, core_cutoff = split_wrappers(model, total_charge)
        self.has_long_range = isinstance(head, _LatentEwaldHead)
        self.has_dispersion = isinstance(disp, _DispersionHead)
        if not hasattr(core, "node_features_energy"):
            raise TypeError(
                f"{type(core).__name__} has no scriptable "
                "'node_features_energy(atomic_numbers, edge_index, edge_vec)' "
                "core, which the TorchScript export requires")
        self.model = core
        self.lr = head
        self.disp = disp
        self.cutoff = float(cutoff)
        self.core_cutoff = min(core_cutoff, self.cutoff) if core_cutoff > 0 else self.cutoff
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

        # the core model sees only the edges within its own cutoff (a D4
        # wrapper widens the neighbor list beyond it)
        core_index = edge_index
        core_vec = edge_vec
        if self.core_cutoff < self.cutoff:
            keep = torch.linalg.norm(edge_vec.detach(), dim=-1) < self.core_cutoff
            core_index = edge_index[:, keep]
            core_vec = edge_vec[keep]
        out = self.model.node_features_energy(atomic_numbers, core_index, core_vec)
        features = out[0]
        node_energy = out[1]
        energy_sr = node_energy.sum()
        node_lr, charges = self.lr(features, atomic_numbers, p_s, cell_s, pbc,
                                   edge_index, edge_vec)
        node_disp, eeq = self.disp(features, atomic_numbers, p_s, cell_s, pbc,
                                   edge_index, edge_vec)
        energy_lr = node_lr.sum()
        energy_disp = node_disp.sum()
        energy = energy_sr + energy_lr + energy_disp

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

        # the add-on terms are already per atom (LES spreads its energy
        # evenly, as LatentEwald does), so the per-atom energies still sum to
        # the total
        per_atom = node_energy + node_lr + node_disp
        result = {
            "energy": energy.reshape(1),
            "total_energy": energy.reshape(1),
            "energy_sr": energy_sr.reshape(1),
            "energy_lr": energy_lr.reshape(1),
            "energy_disp": energy_disp.reshape(1),
            "node_energy": per_atom,
            "forces": forces,
            "stress": stress,
            "virial": -stress * volume,
            "latent_charges": charges,
            "eeq_charges": eeq,
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
            ``energy_lr`` ``(1,)``, ``energy_disp`` ``(1,)``, ``node_energy``
            ``(N,)``, ``forces`` ``(N, 3)``, ``stress`` ``(3, 3)``, ``virial``
            ``(3, 3)``, ``latent_charges`` ``(N, n_channels)`` (empty without
            LES) and ``eeq_charges`` ``(N, 1)`` (empty without D4).
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
        For a model with a long-range or dispersion term the supplied
        neighbor list must cover the whole system on a single rank (and reach
        the exported ``cutoff``); a subdomain-local list does not reproduce
        the trained energy.
        """
        periodic = bool(torch.linalg.norm(cell, dim=1).sum() > 1e-8)
        pbc_t = torch.full((3,), periodic, dtype=torch.bool,
                           device=pos.device)
        return self._evaluate(pos, atomic_numbers, edge_index, cell_shifts,
                              cell.to(pos.dtype), pbc_t)


def export_torchscript_potential(model: nn.Module, cutoff: float, path: str,
                                 metadata: Optional[dict] = None,
                                 total_charge: float = 0.0) -> str:
    """Script a trained model to a standalone ``.pt`` and save it.

    Wraps ``model`` in :class:`TorchScriptPotential`, compiles it with
    ``torch.jit.script`` and writes the archive. The cutoff and the
    ``long_range`` / ``dispersion`` flags are embedded as extra files,
    alongside any caller metadata, so a consumer can introspect the artifact
    without ``xnn``.

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
    total_charge : float, optional
        Net charge of the deployed system (D4 EEQ charges), by default 0.

    Returns
    -------
    str
        The ``path`` written.
    """
    wrapper = TorchScriptPotential(model, cutoff, total_charge).eval()
    scripted = torch.jit.script(wrapper)
    extra = {"cutoff": str(cutoff),
             "long_range": str(wrapper.has_long_range),
             "dispersion": str(wrapper.has_dispersion),
             "total_charge": str(total_charge)}
    if metadata:
        extra.update({k: str(v) for k, v in metadata.items()})
    scripted.save(path, _extra_files=extra)
    return path
