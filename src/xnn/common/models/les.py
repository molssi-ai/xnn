"""Latent Ewald Summation (LES): long-range interactions for any xnn model.

Implements Cheng, *npj Comput Mater* **11**, 80 (2025): short-range MLIPs miss
long-range physics (electrostatics, dispersion) beyond their receptive field.
LES fixes this generically -- a small MLP maps each atom's **invariant
features** to a low-dimensional hidden variable ``q`` (paper eq 2, analogous to
environment-dependent partial charges, but unconstrained), and an Ewald
summation over the structure factor of ``q`` (eqs 3-4) supplies the long-range
energy

    E_lr = (1/V) sum_{0<k<k_c} exp(-sigma^2 k^2 / 2) / k^2 * abs(S(k))^2 .

Faithful to the reference implementation (``cace.modules.EwaldPotential`` of
https://github.com/BingqingCheng/cace and the training scripts of
https://github.com/BingqingCheng/cace-lr-fit): :class:`EwaldSummation` follows
the same algorithm (the triclinic-capable reciprocal-space sum with half-space
symmetry weights, the ``k = 0`` and self-interaction conventions, the
``1/r^6`` dispersion variant of paper eq 5, and the real-space
``erf``-converged direct sum used for non-periodic structures), independently
implemented and verified against the reference to machine precision in
``tests/test_les.py``. One upstream wart is fixed rather than reproduced: the
reference always builds its k-vector grid in float32, which crashes float64
runs; here the grid follows the input dtype.

Because :class:`LatentEwald` only needs *invariant per-atom features*, it wraps
**any** registered xnn model -- every model exposes its features through the
``"node_features"`` output key and a ``node_feature_dim`` attribute (CACE's
symmetrized B features, the scalar channels of MACE / NequIP node features,
Allegro's environment-aggregated edge latents, SchNet / PhysNet feature vectors,
HDNNP/ANI descriptors). Enable it from a config with
``model.extra["long_range"]`` (see
:func:`~xnn.common.models.registry.build_model`) or wrap directly::

    model = LatentEwald(build_model(cfg.model), n_channels=4, sigma=1.0)
    out = ForceStressOutput(model)(graph)   # forces/stress include E_lr

The wrapped energy cost is roughly twice the short-range cost.
"""
from __future__ import annotations

import math
from typing import List

import torch
from torch import Tensor, nn

from ..data import AtomicGraph
from .base import InteratomicPotential


class EwaldSummation(nn.Module):
    """Ewald energy of a (latent) per-atom variable ``q`` (paper eqs 3-5).

    For periodic structures, the reciprocal-space sum over a k-grid limited
    by ``|k| <= 2*pi/dl`` (upstream convention: the paper's ``k_c`` equals
    ``2*pi/dl``); for non-periodic structures (no cell), the equivalent
    ``erf``-converged real-space direct sum. Both handle a multi-dimensional
    ``q``, summing the energies of the channels.

    Parameters
    ----------
    dl : float, optional
        Reciprocal grid resolution; the k-space cutoff is ``2*pi/dl``. By
        default 2.0 (``k_c = pi``, the paper's bulk-water setting; its
        dimer/NaCl runs used ``dl = 3``).
    sigma : float, optional
        Gaussian smearing width in Angstrom, by default 1.0 (paper Methods:
        values between ~0.5 and 2 are reasonable; 1 was best for water).
    exponent : int, optional
        Interaction exponent ``p`` of ``1/r^p``: 1 (electrostatics, default)
        or 6 (London dispersion, paper eq 5).
    remove_self_interaction : bool, optional
        Subtract the Gaussian self energy ``sum q^2 / (sigma (2 pi)^{3/2})``
        from the reciprocal sum, by default ``False`` (the reference training
        scripts keep it; the term is short-ranged and can be absorbed by the
        short-range model either way). Note: for multi-channel ``q`` upstream
        subtracts the total ``sum q^2`` once *per channel* (an
        ``n_channels``-fold over-subtraction); xnn subtracts it once. The
        two agree for 1-dimensional ``q`` and always when the flag is off.

    Notes
    -----
    Charges are in scaled units (upstream ``norm_factor = 1``): a physical
    charge ``Q`` in e corresponds to ``q = Q * sqrt(90.0474)`` for energies
    in eV and distances in Angstrom.
    """

    def __init__(self, dl: float = 2.0, sigma: float = 1.0, exponent: int = 1,
                 remove_self_interaction: bool = False):
        super().__init__()
        if exponent not in (1, 6):
            raise ValueError(f"exponent must be 1 or 6, got {exponent}")
        self.dl = dl
        self.sigma = sigma
        self.exponent = exponent
        self.remove_self_interaction = remove_self_interaction
        self.k_cut_sq = (2.0 * math.pi / dl) ** 2

    def _kfac(self, k2: Tensor) -> Tensor:
        """Interaction kernel in reciprocal space (paper eqs 4 and 5).

        Parameters
        ----------
        k2 : Tensor
            Squared magnitudes ``|k|^2`` of the wave vectors, shape ``(M,)``.
        """
        half_sigma_sq = 0.5 * self.sigma ** 2
        if self.exponent == 1:
            return torch.exp(-half_sigma_sq * k2) / k2
        # dispersion kernel, written in the reduced variable u = sigma|k|/sqrt(2)
        u2 = half_sigma_sq * k2
        u = torch.sqrt(u2)
        tail = math.sqrt(math.pi) * torch.special.erfc(u)
        tail = tail + (0.5 / u ** 3 - 1.0 / u) * torch.exp(-u2)
        return -(k2 ** 1.5 * tail)

    def _self_energy(self, q: Tensor) -> Tensor:
        gaussian_norm = self.sigma * (2.0 * math.pi) ** 1.5
        return q.square().sum() / gaussian_norm

    @torch.jit.export
    def reciprocal(self, pos: Tensor, q: Tensor, cell: Tensor) -> Tensor:
        """Reciprocal-space Ewald energy of one periodic structure.

        Parameters
        ----------
        pos : Tensor
            Cartesian positions, shape ``(n, 3)``.
        q : Tensor
            Hidden variable, shape ``(n, n_channels)``.
        cell : Tensor
            Row-vector cell matrix, shape ``(3, 3)`` (triclinic allowed).

        Returns
        -------
        Tensor
            Scalar long-range energy (summed over channels).
        """
        device, dtype = pos.device, pos.dtype
        # rows of the reciprocal cell, satisfying b_i . a_j = 2 pi delta_ij
        recip = 2.0 * math.pi * torch.linalg.inv(cell).T
        # per-axis integer extent of the candidate grid; the tiny nudge keeps
        # the extent stable when a cell edge length sits within an ulp of a
        # multiple of dl (a documented xnn deviation: upstream truncates the
        # exact quotient, so a rigid rotation of the cell can change the grid)
        # (written as an explicit loop rather than a comprehension over
        # ``.tolist()`` so the method stays ``torch.jit.script``-able for the
        # deploy wrappers; the arithmetic is unchanged)
        lengths = torch.linalg.norm(cell, dim=1)
        n_max: List[int] = []
        for i in range(3):
            n_max.append(max(int(float(lengths[i]) / self.dl + 1e-9), 1))

        # Enumerate one half-space of integer lattice points directly: a
        # triple is generated iff its leading nonzero index is positive, so
        # exactly one of each {+n, -n} pair appears and n = 0 never does.
        # Every surviving k thus stands for its mirror image as well and
        # enters the energy with weight 2.
        na, nb, nc = n_max[0], n_max[1], n_max[2]
        all_b = torch.arange(-nb, nb + 1, device=device)
        all_c = torch.arange(-nc, nc + 1, device=device)
        zero = torch.zeros(1, dtype=torch.long, device=device)
        half_grid = torch.cat([
            torch.cartesian_prod(
                torch.arange(1, na + 1, device=device), all_b, all_c),
            torch.cartesian_prod(
                zero, torch.arange(1, nb + 1, device=device), all_c),
            torch.cartesian_prod(
                zero, zero, torch.arange(1, nc + 1, device=device)),
        ])

        kpts = half_grid.to(dtype) @ recip
        k2 = kpts.square().sum(dim=1)
        # spherical cutoff |k| <= k_c with a relative slack on both bounds;
        # the slack resolves floating-point ties on the boundary shell
        # consistently, keeping the energy exactly rotation-invariant (a
        # documented xnn deviation: upstream compares exactly, so a whole
        # shell can drop out when its |k|^2 lands an ulp above the cutoff)
        in_shell = ((k2 > self.k_cut_sq * 1e-12)
                    & (k2 <= self.k_cut_sq * (1.0 + 1e-9)))
        kpts, k2 = kpts[in_shell], k2[in_shell]

        # |S(k)|^2 per channel via the real and imaginary parts of the
        # structure factor S(k) = sum_i q_i exp(i k . r_i)
        angles = pos @ kpts.T                       # (n, M)
        re_sk = torch.cos(angles).T @ q             # (M, n_channels)
        im_sk = torch.sin(angles).T @ q
        sk_sq = re_sk.square() + im_sk.square()
        energy = (2.0 * (self._kfac(k2).unsqueeze(1) * sk_sq).sum()
                  / torch.det(cell))
        if self.remove_self_interaction and self.exponent == 1:
            energy = energy - self._self_energy(q)
        return energy

    @torch.jit.export
    def realspace(self, pos: Tensor, q: Tensor) -> Tensor:
        """Direct-sum equivalent for a non-periodic structure.

        The pair interaction is ``erf(r / (sqrt(2) sigma)) / r`` -- the
        potential of the Gaussian-smeared charges -- normalized by
        ``1/(4 pi)`` exactly as upstream, so periodic and molecular
        structures share the same energy scale. Only ``exponent = 1`` is
        supported (as upstream).
        """
        if self.exponent != 1:
            raise ValueError("realspace fallback supports exponent=1 only")
        dist = torch.linalg.norm(pos[:, None, :] - pos[None, :, :], dim=-1)
        # two behavioral conventions of the reference are kept on purpose:
        # erf(r / (sqrt(2) sigma)) vanishes at r = 0, silencing the i == j
        # terms, and the 1e-6 offset in the denominator keeps the diagonal
        # finite and differentiable without any masking
        screen = torch.special.erf(dist / (self.sigma * math.sqrt(2.0)))
        pair_kernel = screen / (dist + 1e-6)
        coupling = q[:, None, :] * q[None, :, :]        # (n, n, n_channels)
        energy = (coupling * pair_kernel[:, :, None]).sum() / (4.0 * math.pi)
        if not self.remove_self_interaction:
            energy = energy + self._self_energy(q)
        return energy

    @torch.jit.ignore
    def forward(self, q: Tensor, pos: Tensor, batch: Tensor, num_graphs: int,
                cell: Tensor | None, pbc: Tensor | None = None) -> Tensor:
        """Long-range energy per structure for a batched graph.

        Parameters
        ----------
        q : Tensor
            Hidden variable, shape ``(N,)`` or ``(N, n_channels)``.
        pos : Tensor
            Cartesian positions, shape ``(N, 3)``.
        batch : Tensor
            Structure index of each atom, shape ``(N,)``.
        num_graphs : int
            Number of structures ``B`` in the batch.
        cell : Tensor or None
            Cells of shape ``(B, 3, 3)``, or ``None`` for molecular batches.
        pbc : Tensor or None, optional
            Per-structure periodic flags ``(B, 3)``; a structure is treated
            as periodic when its cell is nonzero and any flag is set.

        Returns
        -------
        Tensor
            Long-range energies, shape ``(B,)``.
        """
        if q.dim() == 1:
            q = q.unsqueeze(1)
        out = q.new_zeros(num_graphs)
        for i in range(num_graphs):
            mask = batch == i
            periodic = (cell is not None
                        and bool(cell[i].diagonal().abs().sum() > 1e-6)
                        and (pbc is None or bool(pbc[i].any())))
            if periodic:
                out[i] = self.reciprocal(pos[mask], q[mask], cell[i])
            else:
                out[i] = self.realspace(pos[mask], q[mask])
        return out


class LatentEwald(InteratomicPotential):
    """Wrap any xnn model with a Latent-Ewald long-range energy (CACE-LR).

    The wrapped model must expose invariant per-atom features through the
    ``"node_features"`` key of its output dict and a ``node_feature_dim``
    attribute -- every built-in xnn model does. A bias-free MLP plus a
    parallel bias-free linear layer (the head used by the reference
    ``cace-lr-fit`` scripts) maps the features to the hidden variable ``q``
    (paper eq 2), and :class:`EwaldSummation` turns ``q`` into the long-range
    energy added to the model's short-range prediction.

    Enable from a config with ``model.extra["long_range"]``, e.g.::

        extra: {..., long_range: {n_channels: 4, sigma: 1.0, dl: 2.0}}

    Parameters
    ----------
    model : InteratomicPotential
        The short-range model to wrap.
    n_channels : int, optional
        Dimension of the hidden variable ``q``, by default 4 (the paper's
        bulk-water/NaCl setting; the reference charged-dimer script uses 1).
    hidden : list of int, optional
        Hidden widths of the ``q`` MLP, by default ``[24, 12]`` (upstream).
    q_bias : bool, optional
        Use a bias in the ``q`` MLP, by default ``False`` (the water script's
        head). The reference charged-dimer script uses ``True`` -- the bias
        lets the latent charge carry a per-structure offset, which matters for
        net-charged systems like ionic dimers.
    q_add_linear : bool, optional
        Add a parallel bias-free linear layer to the ``q`` head, by default
        ``True`` (the water script). The charged-dimer script uses ``False``
        (MLP only).
    dl, sigma, exponent, remove_self_interaction
        Passed to :class:`EwaldSummation`.

    Attributes
    ----------
    model : InteratomicPotential
        The wrapped short-range model.
    q_net : torch.nn.Module
        The latent-charge MLP.
    q_linear : torch.nn.Module or None
        The optional parallel linear layer; ``q = q_net(B) [+ q_linear(B)]``.
    ewald : EwaldSummation
        The long-range energy module.
    cutoff : float
        The wrapped model's neighbor-list cutoff (proxied).

    Notes
    -----
    ``forward`` returns the combined ``"energy"``; the long-range energy is
    spread uniformly over the atoms of each structure in ``"node_energy"``
    so it still sums to the total. The additional keys ``"energy_sr"``,
    ``"energy_lr"``, and ``"latent_charges"`` expose the decomposition.
    """

    def __init__(self, model: InteratomicPotential, n_channels: int = 4,
                 hidden=None, q_bias: bool = False, q_add_linear: bool = True,
                 dl: float = 2.0, sigma: float = 1.0,
                 exponent: int = 1, remove_self_interaction: bool = False):
        super().__init__()
        feature_dim = getattr(model, "node_feature_dim", None)
        if feature_dim is None:
            raise TypeError(
                f"{type(model).__name__} does not expose node_feature_dim / "
                "'node_features'; LatentEwald needs invariant per-atom "
                "features from the wrapped model")
        self.model = model
        self.cutoff = model.cutoff
        # q head: an MLP (as the reference fit scripts) optionally plus a
        # parallel linear layer -- the water script uses bias-free + linear,
        # the charged-dimer script an MLP with bias and no parallel linear
        hidden = list(hidden or [24, 12])
        layers: list[nn.Module] = []
        widths = [feature_dim] + hidden
        for w_in, w_out in zip(widths[:-1], widths[1:]):
            layers += [nn.Linear(w_in, w_out, bias=q_bias), nn.SiLU()]
        layers.append(nn.Linear(widths[-1], n_channels, bias=q_bias))
        self.q_net = nn.Sequential(*layers)
        self.q_linear = (nn.Linear(feature_dim, n_channels, bias=False)
                         if q_add_linear else None)
        self.ewald = EwaldSummation(dl=dl, sigma=sigma, exponent=exponent,
                                    remove_self_interaction=remove_self_interaction)

    def forward(self, data: AtomicGraph) -> dict[str, Tensor]:
        """Short-range prediction plus the latent-Ewald long-range energy.

        Parameters
        ----------
        data : AtomicGraph
            The batched atomic graph.

        Returns
        -------
        dict of str to torch.Tensor
            The wrapped model's outputs with ``"energy"`` and
            ``"node_energy"`` including the long-range term, plus
            ``"energy_sr"`` ``(B,)``, ``"energy_lr"`` ``(B,)`` and
            ``"latent_charges"`` ``(N, n_channels)``.
        """
        out = self.model(data)
        features = out["node_features"]
        q = self.q_net(features)
        if self.q_linear is not None:
            q = q + self.q_linear(features)
        energy_lr = self.ewald(q, data.pos, data.batch, data.num_graphs,
                               data.cell, data.pbc)
        n_atoms = torch.bincount(data.batch, minlength=data.num_graphs)
        out["energy_sr"] = out["energy"]
        out["energy_lr"] = energy_lr
        out["latent_charges"] = q
        out["energy"] = out["energy"] + energy_lr
        out["node_energy"] = out["node_energy"] + (
            energy_lr / n_atoms.to(energy_lr.dtype))[data.batch]
        return out

    @classmethod
    def from_config(cls, cfg) -> "LatentEwald":
        """Not registered directly; built by ``build_model`` via
        ``model.extra['long_range']``."""
        raise NotImplementedError(
            "enable LES via model.extra['long_range'] in the base model's "
            "config, or wrap explicitly: LatentEwald(build_model(cfg), ...)")
