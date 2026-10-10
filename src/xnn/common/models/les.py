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
from typing import List, Optional, Tuple

import torch

from torch import Tensor, nn
from torch.nn import functional as F

from .charge_solve import (bonded_fragments, charge_energy, coulomb_matrix, covalent_radii,
                           fragment_targets, ion_charge_table, solve_charges)
from .dispersion import BOHR, HARTREE
from .ops import cell_volume, scatter_sum

from ..data import AtomicGraph
from .base import InteratomicPotential
from .fast import AutoPolicy, FastPathModule

#: complex entries of one column block of the factorized reciprocal sum (fast path)
FAST_BLOCK_ENTRIES = 5 * 10 ** 7
#: ``use_fast="auto"``: minimum atoms of a periodic structure (LES's cost does not
#: depend on the neighbor graph, so this policy counts atoms, per structure). The
#: factorized sum has a fixed cost of about 2 ms per structure; measured on A100,
#: A30 and V100 in both precisions it breaks even between 1500 and 3000 atoms and
#: wins 2-9x from 5000 atoms on (with 10-30x less memory)
AUTO_POLICY = AutoPolicy(default_min_edges=3000, default_min_edges_float64=3000)
#: latent charge of one elementary charge: two latent charges interact as
#: ``q1 q2 / (2 pi r)``, so ``q = Q * sqrt(2 pi k_e)`` with the Coulomb
#: constant ``k_e = HARTREE * BOHR`` in eV Angstrom; 9.5118 per e
LATENT_CHARGE_PER_E = math.sqrt(2.0 * math.pi * HARTREE * BOHR)


def intra_structure_pairs(batch: Tensor, num_graphs: int) -> Tuple[Tensor, Tensor]:
    """Every ordered pair ``(i, j)``, ``i != j``, of atoms in the same structure.

    Parameters
    ----------
    batch : Tensor
        Structure index of each atom ``(N,)``, in any order.
    num_graphs : int
        Number of structures.

    Returns
    -------
    tuple of Tensor
        Atom indices ``i`` and ``j``, each ``(sum_b n_b (n_b - 1),)``.
    """
    order = torch.argsort(batch, stable=True)
    counts = torch.bincount(batch, minlength=num_graphs)
    start = torch.cumsum(counts, 0) - counts
    graph = batch[order]
    per_atom = counts[graph]                       # size of each sorted atom's structure
    i = torch.repeat_interleave(torch.arange(batch.numel(), device=batch.device), per_atom)
    first = torch.cumsum(per_atom, 0) - per_atom   # first pair of each sorted atom
    local = torch.arange(i.numel(), device=batch.device) - torch.repeat_interleave(first, per_atom)
    j = start[graph[i]] + local
    keep = i != j
    return order[i[keep]], order[j[keep]]


class EwaldSummation(nn.Module, FastPathModule):
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
        dimer/NaCl runs used ``dl = 3``). It applies to **periodic
        structures only**: a structure without a cell takes the real-space
        branch, which is exact and reads ``sigma`` but never ``dl``, so on an
        all-molecular dataset this knob is inert (and is therefore untested
        by the fit, whatever it is set to). The cutoff is recomputed from
        ``dl`` on every call, so it can be retuned on a loaded model.
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
    charge ``Q`` in e corresponds to ``q = Q * LATENT_CHARGE_PER_E``
    (``sqrt(2 pi * 14.3996) = 9.5118``) for energies in eV and distances in
    Angstrom; see :data:`LATENT_CHARGE_PER_E`.
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
        # the factorized reciprocal sum (reciprocal_fast), chosen per evaluation
        # by the owning LatentEwald for the structures of at least
        # fast_min_atoms atoms; off under TorchScript
        self.fast_active = False
        self.fast_min_atoms = 0

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

    def k_set(self, cell: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """The wave vectors of the reciprocal sum of one periodic structure.

        Parameters
        ----------
        cell : Tensor
            Row-vector cell matrix, shape ``(3, 3)`` (triclinic allowed).

        Returns
        -------
        grid : Tensor
            Integer triples ``(M, 3)`` (long) of one half space within the
            spherical cutoff ``|k| <= 2 pi / dl``.
        kpts : Tensor
            The wave vectors ``k = grid @ B``, ``(M, 3)``.
        k2 : Tensor
            ``|k|^2``, ``(M,)``.
        """
        device, dtype = cell.device, cell.dtype
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
        # k_c is derived from dl on every call rather than cached at
        # construction, so retuning the cutoff on a loaded model
        # (``ewald.dl = ...``, as in a convergence study) takes effect
        k_cut_sq = (2.0 * math.pi / self.dl) ** 2
        # spherical cutoff |k| <= k_c with a relative slack on both bounds;
        # the slack resolves floating-point ties on the boundary shell
        # consistently, keeping the energy exactly rotation-invariant (a
        # documented xnn deviation: upstream compares exactly, so a whole
        # shell can drop out when its |k|^2 lands an ulp above the cutoff)
        in_shell = ((k2 > k_cut_sq * 1e-12)
                    & (k2 <= k_cut_sq * (1.0 + 1e-9)))
        return half_grid[in_shell], kpts[in_shell], k2[in_shell]

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
        _, kpts, k2 = self.k_set(cell)
        # |S(k)|^2 per channel via the real and imaginary parts of the
        # structure factor S(k) = sum_i q_i exp(i k . r_i)
        angles = pos @ kpts.T                       # (n, M)
        re_sk = torch.cos(angles).T @ q             # (M, n_channels)
        im_sk = torch.sin(angles).T @ q
        sk_sq = re_sk.square() + im_sk.square()
        energy = (2.0 * (self._kfac(k2).unsqueeze(1) * sk_sq).sum()
                  / cell_volume(cell))
        if self.remove_self_interaction and self.exponent == 1:
            energy = energy - self._self_energy(q)
        return energy

    @torch.jit.unused
    def fast_supported(self, device: torch.device, dtype: torch.dtype) -> bool:
        """The factorized reciprocal sum runs on CUDA devices."""
        return device.type == "cuda" and dtype in (torch.float32, torch.float64)

    @torch.jit.unused
    def reciprocal_fast(self, pos: Tensor, q: Tensor, cell: Tensor) -> Tensor:
        """:meth:`reciprocal` through factorized phases (the fast path).

        The same wave vectors and weights; the structure factors of each
        column of fixed ``(m_1, m_2)`` come from one complex matrix product
        over tabulated phase factors (:mod:`~xnn.common.models.reciprocal`)
        instead of ``(n, M)`` sines and cosines, in column blocks that are
        recomputed in the backward pass. Differentiable to every order that
        :meth:`reciprocal` is.
        """
        from .recompute import recompute
        from .reciprocal import PhaseColumns, complex_dtype, structure_factors

        grid, _, k2 = self.k_set(cell)
        columns = PhaseColumns(grid)
        cdtype = complex_dtype(q.dtype)
        weights = self._kfac(k2)

        def block(p: Tensor, v: Tensor, lat: Tensor, w: Tensor, c0: int, c1: int) -> Tensor:
            theta = PhaseColumns.phases(p, lat)
            sk = structure_factors(columns, columns.column_table(theta, c0, c1, cdtype),
                                   columns.axis_table(theta, cdtype), v)       # (c, C, n_m3)
            sk_sq = (sk.real.square() + sk.imag.square()).sum(1).to(v.dtype)
            return (columns.scatter(w, c0, c1) * sk_sq).sum()

        ranges = list(columns.blocks(pos.shape[0], FAST_BLOCK_ENTRIES))
        if len(ranges) == 1:
            total = block(pos, q, cell, weights, *ranges[0])
        else:
            total = q.new_zeros(())
            for c0, c1 in ranges:
                total = total + recompute(
                    lambda p, v, lat, w, c0=c0, c1=c1: block(p, v, lat, w, c0, c1),
                    pos, q, cell, weights)
        energy = 2.0 * total / cell_volume(cell)
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

    def realspace_batch(self, pos: Tensor, q: Tensor, batch: Tensor, num_graphs: int) -> Tensor:
        """:meth:`realspace` of every structure of a molecular batch at once.

        The same sum over the pairs of each structure, taken over all of them
        in one pass instead of a structure at a time: a training batch of
        molecules then costs a few kernels instead of a loop with a host
        synchronization per structure. The ``i == j`` terms, which vanish
        identically in :meth:`realspace` (``erf(0) = 0``), are left out.

        Parameters
        ----------
        pos : Tensor
            Positions ``(N, 3)``.
        q : Tensor
            Hidden variable ``(N, n_channels)``.
        batch : Tensor
            Structure index of each atom ``(N,)``, in any order.
        num_graphs : int
            Number of structures ``B`` (those without atoms get zero).

        Returns
        -------
        Tensor
            Energies ``(B,)``.
        """
        if self.exponent != 1:
            raise ValueError("realspace fallback supports exponent=1 only")
        i, j = intra_structure_pairs(batch, num_graphs)
        dist = torch.linalg.norm(pos[i] - pos[j], dim=-1)
        screen = torch.special.erf(dist / (self.sigma * math.sqrt(2.0)))
        pair = (q[i] * q[j]).sum(-1) * screen / (dist + 1e-6)
        energy = scatter_sum(pair, batch[i], num_graphs) / (4.0 * math.pi)
        if not self.remove_self_interaction:
            gaussian_norm = self.sigma * (2.0 * math.pi) ** 1.5
            energy = energy + scatter_sum(q.square().sum(-1), batch, num_graphs) / gaussian_norm
        return energy

    def realspace_single(self, pos: Tensor, q: Tensor) -> Tensor:
        """:meth:`realspace` of one structure, with finite second derivatives.

        The same pair sum as :meth:`realspace`, contracted as ``q^T K q`` per
        channel instead of through an ``(N, N, n_channels)`` coupling tensor.
        The ``i == j`` distances, whose terms vanish in :meth:`realspace`, are
        replaced before the square root, whose second derivative is not finite
        at zero. :meth:`forward` takes it for a batch of one molecule
        (molecular dynamics, optimization), where the pair lists of
        :meth:`realspace_batch` would cost more memory.

        Parameters
        ----------
        pos : Tensor
            Positions ``(N, 3)``.
        q : Tensor
            Hidden variable ``(N, n_channels)``.

        Returns
        -------
        Tensor
            The energy, a scalar.
        """
        if self.exponent != 1:
            raise ValueError("realspace fallback supports exponent=1 only")
        diag = torch.eye(pos.shape[0], dtype=torch.bool, device=pos.device)
        d2 = (pos[:, None, :] - pos[None, :, :]).square().sum(-1)
        dist = d2.masked_fill(diag, 1.0).sqrt()
        kernel = torch.special.erf(dist / (self.sigma * math.sqrt(2.0))) / (dist + 1e-6)
        kernel = kernel.masked_fill(diag, 0.0)
        energy = (q * (kernel @ q)).sum() / (4.0 * math.pi)
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
        if cell is None:
            if num_graphs == 1:
                return self.realspace_single(pos, q).reshape(1)
            return self.realspace_batch(pos, q, batch, num_graphs)
        periodic = cell.diagonal(dim1=-2, dim2=-1).abs().sum(-1) > 1e-6
        if pbc is not None:
            periodic = periodic & pbc.any(-1)
        if num_graphs == 1 and not bool(periodic[0]):
            return self.realspace_single(pos, q).reshape(1)
        # the molecular structures all at once, the periodic ones one by one
        molecular = ~periodic[batch]
        out = self.realspace_batch(pos[molecular], q[molecular], batch[molecular], num_graphs)
        for i in periodic.nonzero().flatten().tolist():
            mask = batch == i
            use_fast = self.fast_active and int(mask.sum()) >= self.fast_min_atoms
            reciprocal = self.reciprocal_fast if use_fast else self.reciprocal
            out = out.index_put((torch.tensor([i], device=out.device),),
                                reciprocal(pos[mask], q[mask], cell[i]).reshape(1))
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
    constrain_charge : bool, optional
        Pin each structure's latent charges to its net charge after the q
        head: the channel ``charge_channel`` sums to ``total_charge *
        LATENT_CHARGE_PER_E`` (the batch's ``total_charge``, neutral when
        absent) and every other channel to zero, by a per-structure shift
        ``q_i <- q_i - w_i (sum q - Q') / sum w`` (the correction of
        PhysNet). The sum is then exact for any system size, where a bias in
        the q head (``q_bias``) only learns an offset per atom, and in the
        molecular path the monopole energy of charged structures follows.
        Off by default (the reference behavior). Charged training structures
        must carry their ``total_charge`` label, or they are forced neutral.
    charge_channel : int, optional
        The channel that carries the physical net charge, by default 0.
    charge_weights : str, optional
        ``"uniform"`` (default): the same shift for every atom;
        ``"learned"``: ``w_i = softplus(linear(features_i))``, a learned
        per-atom capacity for the residual.
    charge_solve : bool, optional
        Global charge solve for channel ``charge_channel``
        (:mod:`~xnn.common.models.charge_solve`): the head's output of that
        channel is an electronegativity ``chi_i`` and the charges minimise
        ``chi.q + 1/2 sum J q^2 + E_lr(q)`` with ``sum q = total_charge``
        (in latent units), so every charge responds to every other one
        through the Ewald kernel. Adds ``chi.q + 1/2 J q^2`` to the energy
        (``E_lr`` is the usual one). Off by default. With the coupling it
        contains, ``constrain_charge`` is its coupling-free limit.
    hardness : str, optional
        Where the hardness ``J_i > 0`` of the charge solve comes from:
        ``"element"`` (default), one learned value per element, or
        ``"features"``, ``softplus(linear(features_i))``.
    hardness_init : float, optional
        Initial hardness in eV per e^2 (converted to latent units), by
        default 10.0, about the atomic values of charge equilibration.
    fragments : bool, optional
        One constraint row per fragment instead of one per structure, so no
        charge flows between separate molecules (plain charge equilibration
        is metallic at long range). Fragments are the covalently connected
        components (:func:`~xnn.common.models.charge_solve.bonded_fragments`),
        with the ions of ``ion_charges`` always on their own; each fragment's
        charge comes from the structure's ``fragment_charges`` label (the
        charge of the fragment each atom belongs to) when present, else from
        ``ion_charges`` (neutral otherwise), and must add up to
        ``total_charge``. Off by default.
    bond_factor : float, optional
        Bond criterion: distance below ``bond_factor`` times the sum of the
        covalent radii, by default 1.2.
    ion_charges : dict, optional
        Formal charges by element symbol or atomic number, by default
        :data:`~xnn.common.models.charge_solve.ION_CHARGES` (the alkali,
        alkaline-earth and halide ions).
    dl, sigma, exponent, remove_self_interaction
        Passed to :class:`EwaldSummation`. Note that ``dl`` acts on periodic
        structures only and ``exponent = 6`` is periodic-only as well (the
        real-space branch that molecular structures take implements the
        ``1/r`` kernel), so a dataset without cells trains neither.

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
                 exponent: int = 1, remove_self_interaction: bool = False,
                 constrain_charge: bool = False, charge_channel: int = 0,
                 charge_weights: str = "uniform", charge_solve: bool = False,
                 hardness: str = "element", hardness_init: float = 10.0,
                 fragments: bool = False, bond_factor: float = 1.2,
                 ion_charges: Optional[dict] = None):
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
        if not 0 <= charge_channel < n_channels:
            raise ValueError(f"charge_channel must be in [0, {n_channels}), got {charge_channel}")
        if charge_weights not in ("uniform", "learned"):
            raise ValueError(f"charge_weights must be 'uniform' or 'learned', got {charge_weights!r}")
        self.constrain_charge = bool(constrain_charge)
        self.charge_channel = int(charge_channel)
        self.charge_weight = (nn.Linear(feature_dim, 1)
                              if constrain_charge and charge_weights == "learned" else None)
        if hardness not in ("element", "features"):
            raise ValueError(f"hardness must be 'element' or 'features', got {hardness!r}")
        if charge_solve and exponent != 1:
            raise ValueError("the charge solve needs the Coulomb kernel (exponent=1)")
        self.charge_solve = bool(charge_solve)
        self.hardness_mode = hardness
        # J = softplus(raw), raw initialised so that J starts at hardness_init
        raw0 = math.log(math.expm1(float(hardness_init) / LATENT_CHARGE_PER_E ** 2))
        self.hardness_table = self.hardness_net = None
        if charge_solve and hardness == "element":
            self.hardness_table = nn.Embedding(119, 1)
            nn.init.constant_(self.hardness_table.weight, raw0)
        elif charge_solve:
            self.hardness_net = nn.Linear(feature_dim, 1)
            nn.init.zeros_(self.hardness_net.weight)
            nn.init.constant_(self.hardness_net.bias, raw0)
        self.fragments = bool(fragments) and bool(charge_solve)
        self.bond_factor = float(bond_factor)
        if self.fragments:
            self.register_buffer("covalent_radii", covalent_radii(), persistent=False)
            self.register_buffer("ion_charges", ion_charge_table(ion_charges), persistent=False)
        self.ewald = EwaldSummation(dl=dl, sigma=sigma, exponent=exponent,
                                    remove_self_interaction=remove_self_interaction)
        self.use_fast = "auto"

    @torch.jit.unused
    def set_use_fast(self, use_fast) -> "LatentEwald":
        """Choose the implementation of the reciprocal sum.

        Parameters
        ----------
        use_fast : bool or str
            ``"auto"`` (default): the factorized reciprocal sum
            (:meth:`EwaldSummation.reciprocal_fast`) on a CUDA device for the
            periodic structures of at least :data:`AUTO_POLICY` atoms (3000);
            ``True``: wherever it runs; ``False``: always the reference. The
            wrapped model's own fast paths are set by
            :func:`~xnn.common.models.fast.set_use_fast`, which reaches both.

        Returns
        -------
        LatentEwald
            ``self``.
        """
        from .fast import resolve_use_fast
        self.use_fast = resolve_use_fast(use_fast)
        return self

    @torch.jit.unused
    def _select_fast(self, data: AtomicGraph) -> None:
        from .fast import select
        device, dtype = data.pos.device, data.model_dtype
        # "auto" decides per structure (by its atoms) in EwaldSummation.forward
        select([self.ewald], self.use_fast, None, device, dtype, 0)
        self.ewald.fast_min_atoms = (AUTO_POLICY.threshold(device, dtype)
                                     if isinstance(self.use_fast, str) and self.ewald.fast_active
                                     else 0)

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
        if not torch.jit.is_scripting():
            self._select_fast(data)
        out = self.model(data)
        features = out["node_features"]
        q = self.q_net(features)
        if self.q_linear is not None:
            q = q + self.q_linear(features)
        # with the charge solve, the head's charge channel is the electronegativity
        chi = q[:, self.charge_channel] if self.charge_solve else None
        if self.constrain_charge:
            q = self.constrain(q, features, data.batch, data.num_graphs, data.total_charge)
        # the Ewald sum computes in the model's dtype (a float64 geometry is cast)
        cell = data.cell.to(q.dtype) if data.cell is not None else None
        pos = data.pos.to(q.dtype)
        if chi is not None:
            q, node_charge_energy, mu, hardness, frag = self.solve_charges(
                chi, q, features, data, pos, cell)
        energy_lr = self.ewald(q, pos, data.batch, data.num_graphs, cell, data.pbc)
        n_atoms = torch.bincount(data.batch, minlength=data.num_graphs)
        out["energy_sr"] = out["energy"]
        out["energy_lr"] = energy_lr
        out["latent_charges"] = q
        out["energy"] = out["energy"] + energy_lr
        out["node_energy"] = out["node_energy"] + (
            energy_lr / n_atoms.to(energy_lr.dtype))[data.batch]
        if chi is not None:
            energy_charge = scatter_sum(node_charge_energy, data.batch, data.num_graphs)
            out["energy_charge"] = energy_charge
            out["hardness"] = hardness
            out["chemical_potential"] = mu
            if frag is not None:
                out["fragments"] = frag
            out["energy"] = out["energy"] + energy_charge
            out["node_energy"] = out["node_energy"] + node_charge_energy
        return out

    def hardness_of(self, features: Tensor, atomic_numbers: Tensor) -> Tensor:
        """The hardness ``J_i > 0`` of every atom, ``(N,)``, in latent units."""
        if self.hardness_table is not None:
            raw = self.hardness_table(atomic_numbers).squeeze(1)
        else:
            raw = self.hardness_net(features).squeeze(1)
        return F.softplus(raw)

    def solve_charges(self, chi: Tensor, q: Tensor, features: Tensor, data: AtomicGraph,
                      pos: Tensor, cell: Optional[Tensor]):
        """Replace channel ``charge_channel`` of ``q`` by the solved charges.

        One augmented solve per structure (:func:`~xnn.common.models.charge_solve.solve_charges`
        with the Coulomb matrix of its own kernel: the direct sum for a
        cluster, the reciprocal sum for a cell), with one constraint row per
        structure or, with ``fragments``, per fragment.

        Returns
        -------
        (Tensor, Tensor, Tensor, Tensor, Tensor or None)
            The charges ``(N, n_channels)``, the per-atom energy
            ``chi_i q_i + 1/2 J_i q_i^2`` ``(N,)``, the chemical potentials
            (one per constraint row), the hardness ``(N,)`` and the fragment
            index of every atom ``(N,)`` (``None`` without fragments).
        """
        batch, num_graphs, z = data.batch, data.num_graphs, data.atomic_numbers
        hardness = self.hardness_of(features, z)
        frag_all = None
        if self.fragments:
            frag_all = bonded_fragments(z, data.edge_index, data.edge_vectors().to(pos.dtype),
                                        self.covalent_radii.to(pos.dtype),
                                        self.ion_charges.to(pos.dtype), self.bond_factor)
        solved = torch.zeros_like(chi)
        mus = []
        for i in range(num_graphs):
            idx = (batch == i).nonzero().squeeze(1)
            periodic = (cell is not None
                        and bool(cell[i].diagonal().abs().sum() > 1e-6)
                        and (data.pbc is None or bool(data.pbc[i].any())))
            gamma = coulomb_matrix(self.ewald, pos[idx], cell[i] if periodic else None)
            total = data.total_charge[i].to(chi.dtype) if data.total_charge is not None else None
            if frag_all is None:
                frag_i = None
                target = total * LATENT_CHARGE_PER_E if total is not None else chi.new_zeros(())
            else:
                frag_i = torch.unique(frag_all[idx], return_inverse=True)[1]
                labeled = (data.fragment_charges is not None
                           and (data.fragment_charges_mask is None
                                or bool(data.fragment_charges_mask[i])))
                label = data.fragment_charges[idx].to(chi.dtype) if labeled else None
                target = fragment_targets(frag_i, z[idx], self.ion_charges.to(chi.dtype),
                                          label, total) * LATENT_CHARGE_PER_E
            q_i, mu_i = solve_charges(chi[idx], hardness[idx], gamma, target, frag_i)
            solved = solved.index_copy(0, idx, q_i)
            mus.append(mu_i)
        q = q.clone()
        q[:, self.charge_channel] = solved
        return q, charge_energy(chi, hardness, solved), torch.cat(mus), hardness, frag_all

    def constrain(self, q: Tensor, features: Tensor, batch: Tensor, num_graphs: int,
                  total_charge: Optional[Tensor]) -> Tensor:
        """Shift the latent charges so each structure sums to its target.

        Parameters
        ----------
        q : Tensor
            Latent charges, ``(N, n_channels)``.
        features : Tensor
            Per-atom features, ``(N, F)`` (the learned weights read them).
        batch : Tensor
            Structure index of every atom, ``(N,)``.
        num_graphs : int
            Number of structures.
        total_charge : Tensor or None
            Net charge of every structure in e, ``(B,)``; ``None`` is neutral.

        Returns
        -------
        Tensor
            ``q`` with channel ``charge_channel`` summing to
            ``total_charge * LATENT_CHARGE_PER_E`` and the others to zero.
        """
        target = torch.zeros((num_graphs, q.shape[1]), dtype=q.dtype, device=q.device)
        if total_charge is not None:
            target[:, self.charge_channel] = total_charge.to(q.dtype) * LATENT_CHARGE_PER_E
        residual = scatter_sum(q, batch, num_graphs) - target            # (B, n_channels)
        if self.charge_weight is not None:
            w = F.softplus(self.charge_weight(features))                 # (N, 1)
            return q - w * (residual / scatter_sum(w, batch, num_graphs))[batch]
        n_atoms = torch.bincount(batch, minlength=num_graphs).to(q.dtype)
        return q - (residual / n_atoms.unsqueeze(1))[batch]

    @classmethod
    def from_config(cls, cfg) -> "LatentEwald":
        """Not registered directly; built by ``build_model`` via
        ``model.extra['long_range']``."""
        raise NotImplementedError(
            "enable LES via model.extra['long_range'] in the base model's "
            "config, or wrap explicitly: LatentEwald(build_model(cfg), ...)")
