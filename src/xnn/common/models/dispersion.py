"""Shared machinery of the dispersion add-ons (DFT-D3, DFT-D4).

The two Grimme dispersion models share most of their structure: a real-space
cutoff with an optional quintic switching window, Gaussian coordination-number
weights over tabulated reference systems (D3 paper eq 16, D4 paper eq 8), an
Axilrod-Teller-Muto three-body term with zero damping over the atom triples
of a neighbor list (D3 paper eqs 11-14, D4 paper eqs 22-27), and the same way
of attaching themselves to a short-range model. Those pieces live here, once;
:mod:`~xnn.common.models.d3` and :mod:`~xnn.common.models.d4` hold what is
specific to each model (the coordination-number counting function, the
reference data, the damping functions, D4's charge scaling and EEQ charges).

:class:`DispersionCorrection` is the model-agnostic wrapper both add-ons
derive from: standalone it *is* the dispersion energy, given a short-range
``model`` it adds the dispersion energy to that model's prediction while
handing the model only the edges within its own cutoff. The deploy channels
(:mod:`~xnn.common.deploy.torchscript`, :mod:`~xnn.common.deploy.lammps`)
peel the wrapper off through the same interface.
"""
# NOTE: no ``from __future__ import annotations`` -- TorchScript resolves the
# annotations of the scripted functions at compile time.
import math
from dataclasses import replace
from typing import Dict, Optional

import torch
from torch import Tensor, nn

from ..data import AtomicGraph
from .base import InteratomicPotential
from .ops import build_triplets, scatter_sum, segment_sum

# CODATA 2018, derived from h, m_e, c, alpha and e exactly as the reference
# codes (mctc-lib) do: a_0 = hbar / (m_e c alpha), E_h = m_e c^2 alpha^2. The
# tabulated 0.529177210903 / 27.211386245988 differ in the 12th digit, which
# the coordination numbers would show at the 1e-11 level.
BOHR = 0.5291772109044924      # Angstrom per bohr
HARTREE = 27.21138624593551    # eV per hartree



def switching_function(r: Tensor, cutoff: float, width: float) -> Tensor:
    """Quintic switching window: 1 below ``cutoff - width``, 0 above ``cutoff``.

    ``width <= 0`` gives the sharp cutoff (the upstream default). Inside the
    window ``x = (cutoff - r) / width`` runs from 0 to 1 and the switch is
    ``x^3 (10 - 15 x + 6 x^2)``, with zero slope at both ends.

    Parameters
    ----------
    r : Tensor
        Distances, any shape.
    cutoff : float
        End of the window (same unit as ``r``).
    width : float
        Width of the window; ``<= 0`` disables it.

    Returns
    -------
    Tensor
        The switch value in ``[0, 1]``, same shape as ``r``.
    """
    if width <= 0.0:
        return torch.ones_like(r)
    w = min(width, cutoff)
    x = torch.clamp((cutoff - r) / w, 0.0, 1.0)
    return x ** 3 * (10.0 + x * (6.0 * x - 15.0))


def gaussian_reference_weights(cn: Tensor, refcn: Tensor, valid: Tensor, wf: float,
                               ngw: Optional[Tensor] = None,
                               eps_norm: float = 0.0) -> Tensor:
    """Normalized Gaussian weights of the reference systems in the coordination number.

    D3 paper eq 16 / D4 paper eq 8: ``W_ref = sum_j^{N_ref} exp(-wf j (CN -
    CN_ref)^2) / norm``. When every Gaussian underflows (a coordination number
    far from all references) the reference(s) with the largest CN get weight
    one, as the reference codes do.

    Parameters
    ----------
    cn : Tensor
        Coordination numbers, shape ``(N,)``.
    refcn : Tensor
        Reference coordination numbers per atom, shape ``(N, R)``.
    valid : Tensor
        Boolean mask of the used reference slots, shape ``(N, R)``.
    wf : float
        Gaussian exponent (D3: 4, D4: 6).
    ngw : Tensor or None, optional
        Number of Gaussians per reference (D4's ``N^s``), shape ``(N, R)``;
        ``None`` (D3) uses a single Gaussian per reference.
    eps_norm : float, optional
        Norm below which the fallback applies (D4 uses ``sqrt(tiny)``, D3
        exactly zero).

    Returns
    -------
    Tensor
        Weights, shape ``(N, R)``, zero in unused slots.
    """
    dcn2 = (cn[:, None] - refcn) ** 2
    if ngw is None:
        gw = torch.exp(-wf * dcn2)
    else:
        n_max = int(ngw.max())
        j = torch.arange(1, n_max + 1, device=cn.device)
        gauss = torch.exp(-wf * j.to(cn.dtype)[None, None, :] * dcn2[:, :, None])
        use = j[None, None, :] <= ngw[:, :, None]
        gw = torch.where(use, gauss, torch.zeros_like(gauss)).sum(-1)
    gw = torch.where(valid, gw, torch.zeros_like(gw))
    norm = gw.sum(dim=1)
    ok = norm > eps_norm
    weights = gw / torch.where(ok, norm, torch.ones_like(norm))[:, None]
    neg_inf = torch.full_like(refcn, -math.inf)
    max_cn = torch.where(valid, refcn, neg_inf).max(dim=1).values
    fallback = (valid & ((refcn - max_cn[:, None]).abs() < 1e-12)).to(cn.dtype)
    return torch.where(ok[:, None], weights, fallback)


def three_body_energy(z: Tensor, edge_index: Tensor, edge_vec: Tensor, r: Tensor,
                      c6_mat: Tensor, r0_table: Tensor, s9: Tensor, alp3: float,
                      cutoff: float, width: float, n_atoms: int) -> Tensor:
    """Per-atom Axilrod-Teller-Muto three-body energy with zero damping.

    ``E = s9 sum_ABC C9 (3 cos cos cos + 1) / (R_AB R_BC R_CA)^3 / (1 + 6
    (R_0 / R)^alp3)`` with ``C9 = sqrt(C6_AB C6_BC C6_CA)``, ``R_0`` the
    geometric product of the three pair critical radii and ``R`` that of the
    three distances. Triples are enumerated as pairs of edges sharing a center;
    each geometric triangle is visited once per corner (three times for
    distinct atoms, and correspondingly for the self-image triangles of a
    periodic cell), so a third of the triangle energy is assigned to the
    center at every visit -- which reproduces the atom-resolved bookkeeping
    of the reference codes exactly. The triplet arithmetic runs in chunks so
    memory stays bounded for long cutoffs.

    Parameters
    ----------
    z : Tensor
        Atomic numbers, shape ``(N,)``.
    edge_index : Tensor
        Edges ``[src, dst]`` within the three-body cutoff, shape ``(2, E)``.
    edge_vec : Tensor
        Edge vectors in bohr, shape ``(E, 3)``.
    r : Tensor
        Edge lengths in bohr, shape ``(E,)``.
    c6_mat : Tensor
        Dense pair ``C6`` matrix, shape ``(N, N)``.
    r0_table : Tensor
        Pair critical radii by element, shape ``(Z_max + 1, Z_max + 1)``, bohr.
    s9 : Tensor
        Three-body scaling (scalar tensor, may be learnable).
    alp3 : float
        Damping exponent (``alp / 3`` in D4, ``(alp + 2) / 3`` in D3).
    cutoff, width : float
        Three-body cutoff and switching width, bohr.
    n_atoms : int
        Number of atoms.

    Returns
    -------
    Tensor
        Per-atom three-body energies in hartree, shape ``(N,)``.
    """
    e1, e2, center = build_triplets(edge_index, n_atoms)
    energy = torch.zeros(n_atoms, dtype=r.dtype, device=r.device)
    n_trip = e1.shape[0]
    chunk = 1048576                      # triplets per block (a literal: TorchScript)
    start = 0
    while start < n_trip:
        stop = min(start + chunk, n_trip)
        a, b, c = e1[start:stop], e2[start:stop], center[start:stop]
        j, k = edge_index[0][a], edge_index[0][b]
        e_tri = triplet_energy(a, b, c, j, k, edge_vec, r, z, c6_mat[c, j], c6_mat[c, k],
                               c6_mat[j, k], r0_table, s9, alp3, cutoff, width)
        energy = energy + scatter_sum(e_tri, c, n_atoms)
        start = stop
    return energy

def triplet_energy(a: Tensor, b: Tensor, c: Tensor, j: Tensor, k: Tensor, edge_vec: Tensor,
                   r: Tensor, z: Tensor, c6_cj: Tensor, c6_ck: Tensor, c6_jk: Tensor,
                   r0_table: Tensor, s9: Tensor, alp3: float, cutoff: float,
                   width: float, r0_prod: Optional[Tensor] = None) -> Tensor:
    """ATM energy of one block of triplets, a third per center visit, ``(T,)``.

    ``a, b`` index the two edges ``j -> c`` and ``k -> c`` of every triplet,
    ``c`` the shared center, ``j, k`` the outer atoms; ``c6_*`` are the three
    pair coefficients. The product of the three pair critical radii is read
    from ``r0_table`` by element, or taken from ``r0_prod`` when given. Shared
    by the scripted loop (:func:`three_body_energy`) and the recomputed one
    (:func:`three_body_energy_chunked`), so the formula lives once.
    """
    # index_select rather than x[idx]: its backward is an atomic index_add,
    # where the sort-based backward of advanced indexing serializes the many
    # repeats of every edge over the triplets
    v_ij, v_ik = edge_vec.index_select(0, a), edge_vec.index_select(0, b)
    r_a, r_b = r.index_select(0, a), r.index_select(0, b)
    r2_ij, r2_ik = r_a * r_a, r_b * r_b
    v_jk = v_ij - v_ik
    r2_jk = (v_jk * v_jk).sum(-1)
    keep = (r2_jk <= cutoff * cutoff) & (r2_jk > 2.220446049250313e-16)
    r2_jk = torch.where(keep, r2_jk, torch.ones_like(r2_jk))
    r_jk = torch.sqrt(r2_jk)
    c9 = s9 * torch.sqrt((c6_cj * c6_ck * c6_jk).abs())
    if r0_prod is None:
        zc, zj, zk = z.index_select(0, c), z.index_select(0, j), z.index_select(0, k)
        r0 = r0_table[zc, zj] * r0_table[zc, zk] * r0_table[zj, zk]
    else:
        r0 = r0_prod
    prod2 = r2_ij * r2_ik * r2_jk
    prod1 = torch.sqrt(prod2)
    prod3 = prod2 * prod1
    prod5 = prod3 * prod2
    damp = 1.0 / (1.0 + 6.0 * (r0 / prod1) ** alp3)
    angular = (0.375 * (r2_ij + r2_jk - r2_ik) * (r2_ij - r2_jk + r2_ik)
               * (-r2_ij + r2_jk + r2_ik) / prod5 + 1.0 / prod3)
    sw = (switching_function(r_a, cutoff, width) * switching_function(r_b, cutoff, width)
          * switching_function(r_jk, cutoff, width))
    return torch.where(keep, c9 * angular * damp * sw / 3.0, torch.zeros_like(prod1))


def edge_cell_shifts(pos: Tensor, edge_index: Tensor, edge_vec: Tensor, cell: Tensor,
                     batch: Tensor) -> Tensor:
    """Integer lattice shifts ``S`` of directed edges, ``(E, 3)`` int64.

    Recovered from ``edge_vec = pos[dst] - pos[src] + S @ cell`` (the
    :meth:`~xnn.common.data.AtomicGraph.edge_vectors` convention) with the cell
    of each edge's structure (``cell (B, 3, 3)``, an all-zero cell marks a
    molecule and gives zero shifts), so no neighbor-list metadata is needed.
    """
    src, dst = edge_index[0], edge_index[1]
    b = batch[dst]
    periodic = cell.abs().sum(dim=(1, 2)) > 0
    inv = torch.where(periodic[:, None, None], torch.linalg.inv(torch.where(
        periodic[:, None, None], cell, torch.eye(3, dtype=cell.dtype, device=cell.device))),
        torch.zeros_like(cell))
    frac = torch.einsum("ei,eij->ej", edge_vec - (pos[dst] - pos[src]), inv[b])
    return torch.round(frac).to(torch.long)


def pair_edge_keys(edge_index: Tensor, shifts: Tensor, n_atoms: int) -> Tensor:
    """One int64 key per directed edge from ``(dst, src, shift)``, unique per image."""
    if n_atoms > (1 << 22) or int(shifts.abs().max()) > 31 if shifts.numel() else False:
        raise ValueError("pair keys support up to 4M atoms and |cell shift| <= 31")
    sh = (shifts + 32).to(torch.long)
    code = (sh[:, 0] << 12) | (sh[:, 1] << 6) | sh[:, 2]
    return ((edge_index[1] * n_atoms + edge_index[0]) << 18) | code


def _block_triplets(e0: int, e1: int, edge_index: Tensor, n_atoms: int, once: bool):
    """Triplets of the centers whose (center-sorted) edges are ``e0 <= e < e1``.

    Returns the edge ids ``a, b`` (``j -> c``, ``k -> c``), the atoms
    ``c, j, k`` and the ``distinct`` mask. With ``once`` a triangle of three
    distinct atoms is kept only from its smallest-index corner.
    """
    eids = torch.arange(e0, e1, device=edge_index.device)
    e1_, e2_, c = build_triplets(edge_index[:, e0:e1], n_atoms)
    a, b = eids.index_select(0, e1_), eids.index_select(0, e2_)
    src = edge_index[0]
    j, k = src.index_select(0, a), src.index_select(0, b)
    distinct = (j != c) & (k != c) & (j != k)
    if once and a.numel() > 0:
        canonical = torch.nonzero(~distinct | ((c < j) & (c < k))).squeeze(1)
        a, b, c = a.index_select(0, canonical), b.index_select(0, canonical), c.index_select(0, canonical)
        j, k, distinct = j.index_select(0, canonical), k.index_select(0, canonical), distinct.index_select(0, canonical)
    return a, b, c, j, k, distinct


def _third_side(a: Tensor, b: Tensor, j: Tensor, k: Tensor, n_atoms: int, edge_shift: Tensor,
                sorted_keys: Tensor, key_perm: Tensor):
    """Edge id of the third side ``(j, k)`` of every triplet and whether it exists.

    The third side is the edge ``k -> j`` with shift ``S_b - S_a``, found by its
    packed key; a triplet whose third side lies beyond the cutoff has no edge
    (it is masked by ``keep`` in :func:`triplet_energy` anyway).
    """
    shift = edge_shift.index_select(0, b) - edge_shift.index_select(0, a) + 32
    code = (shift[:, 0] << 12) | (shift[:, 1] << 6) | shift[:, 2]
    key = ((j * n_atoms + k) << 18) | code
    idx = torch.searchsorted(sorted_keys, key).clamp_(max=sorted_keys.shape[0] - 1)
    found = sorted_keys.index_select(0, idx) == key
    return key_perm.index_select(0, idx), found


class _TripletCache:
    """A step's ATM block triples, kept from the forward to the backward pass.

    The forward block enumerates its triples anyway; storing them saves the
    backward pass (the closed-form block gradient) from enumerating them a
    second time, which is 25-35% of the three-body time. Per triple the cache
    holds the two edges ``a, b`` and the third side ``e_jk`` as int32 plus the
    ``distinct`` / ``found`` flags, 14 bytes; the centers and outer atoms are
    recovered from the edges (``c = dst[a]``, ``j = src[a]``, ``k = src[b]``).
    Blocks that do not fit in ``budget`` bytes are re-enumerated as before, and
    an entry is freed as soon as the backward pass has taken it. The cache
    lives in the closures of one :func:`three_body_energy_chunked` call, so
    nothing outlives the autograd graph of a step.
    """

    BYTES_PER_TRIPLE = 14

    def __init__(self, budget: int):
        self.budget = int(budget)
        self.used = 0
        self.store: Dict[tuple, tuple] = {}

    def put(self, key: tuple, a: Tensor, b: Tensor, distinct: Tensor, e_jk: Tensor,
            found: Tensor) -> None:
        nbytes = int(a.numel()) * self.BYTES_PER_TRIPLE
        if self.used + nbytes > self.budget:
            return
        self.store[key] = (a.to(torch.int32), b.to(torch.int32), distinct,
                           e_jk.to(torch.int32), found, nbytes)
        self.used += nbytes

    def take(self, key: tuple, edge_index: Tensor):
        entry = self.store.pop(key, None)
        if entry is None:
            return None
        a32, b32, distinct, e32, found, nbytes = entry
        self.used -= nbytes
        a, b = a32.long(), b32.long()
        src, dst = edge_index[0], edge_index[1]
        return (a, b, dst.index_select(0, a), src.index_select(0, a), src.index_select(0, b),
                distinct, e32.long(), found)


def triplet_cache_budget(like: Tensor, budget: Optional[float]) -> int:
    """Bytes the triple cache may hold: ``budget`` in GB, or ``None`` for a
    quarter of the free device memory (CPU: 1 GB); 0 disables the cache."""
    if budget is not None:
        return int(float(budget) * 2 ** 30)
    if like.device.type != "cuda":
        return 1 << 30
    free, _ = torch.cuda.mem_get_info(like.device)
    return int(0.25 * free)


def _triplet_block(e0: int, e1: int, z: Tensor, edge_index: Tensor, edge_vec: Tensor,
                   r: Tensor, r0_table: Optional[Tensor], s9: Tensor, alp3: float, cutoff: float,
                   width: float, n_atoms: int, c6_mat: Optional[Tensor],
                   alpha_a: Optional[Tensor], alpha_b: Optional[Tensor],
                   r0_atom: Optional[Tensor], a1: Optional[Tensor], a2: Optional[Tensor],
                   c6_edge: Optional[Tensor] = None, edge_shift: Optional[Tensor] = None,
                   sorted_keys: Optional[Tensor] = None, key_perm: Optional[Tensor] = None,
                   once: bool = True, cache: Optional[_TripletCache] = None) -> Tensor:
    """Per-atom ATM energy of the centers whose edges are ``e0 <= e < e1`` in the
    center-sorted edge list (enumerated here, so a recompute block retains
    nothing of the triplets).

    With ``once`` every triangle of three distinct atoms is evaluated from one
    corner only, the one with the smallest index, and a third of its energy is
    assigned to each corner; triangles with a repeated atom (periodic
    self-images) keep the reference bookkeeping of one third per center
    visit. Per-atom energies are identical either way; the work drops by
    three. With a ``cache`` the block's triples are kept for the backward pass
    (:class:`_TripletCache`).
    """
    a, b, c, j, k, distinct = _block_triplets(e0, e1, edge_index, n_atoms, once)
    if a.numel() == 0:
        return torch.zeros(n_atoms, dtype=r.dtype, device=r.device)
    if c6_edge is not None:
        c6_cj, c6_ck = c6_edge.index_select(0, a), c6_edge.index_select(0, b)
        e_jk, found = _third_side(a, b, j, k, n_atoms, edge_shift, sorted_keys, key_perm)
        if cache is not None:
            cache.put((e0, e1), a, b, distinct, e_jk, found)
        # a positive placeholder keeps sqrt's backward finite on masked triplets
        c6_jk = torch.where(found, c6_edge.index_select(0, e_jk), torch.ones_like(c6_cj))
    elif c6_mat is not None:
        c6_cj, c6_ck, c6_jk = c6_mat[c, j], c6_mat[c, k], c6_mat[j, k]
    else:
        # C6 from the dynamic polarizabilities: (alpha_a[i] * alpha_b[j]).sum()
        ac, aj = alpha_a.index_select(0, c), alpha_a.index_select(0, j)
        bj, bk = alpha_b.index_select(0, j), alpha_b.index_select(0, k)
        c6_cj, c6_ck, c6_jk = (ac * bj).sum(-1), (ac * bk).sum(-1), (aj * bk).sum(-1)
    if r0_atom is not None:
        # BJ radii ``a1 sqrt(3 Q_A Q_B) + a2`` from per-atom factors
        # ``rho_A = 3^(1/4) sqrt(Q_A)``: the trainable scalars enter through
        # broadcasts (cheap reductions in the backward pass) instead of a
        # gathered pair table whose gradient is a contended index-accumulate
        rc, rj, rk = r0_atom.index_select(0, c), r0_atom.index_select(0, j), r0_atom.index_select(0, k)
        table = torch.zeros((1, 1), dtype=r.dtype, device=r.device)
        r0_prod = (a1 * rc * rj + a2) * (a1 * rc * rk + a2) * (a1 * rj * rk + a2)
    else:
        table = r0_table
        r0_prod = None
    e_tri = triplet_energy(a, b, c, j, k, edge_vec, r, z, c6_cj, c6_ck, c6_jk,
                           table, s9, alp3, cutoff, width, r0_prod)
    energy = _center_sum(e_tri, c, n_atoms)
    if once:
        # the other two corners of a triangle visited once get their thirds
        e_out = torch.where(distinct, e_tri, torch.zeros_like(e_tri))
        energy = energy + scatter_sum(e_out, j, n_atoms) + scatter_sum(e_out, k, n_atoms)
    return energy


def _switch_and_slope(r: Tensor, cutoff: float, width: float):
    """The quintic switch and its derivative with respect to ``r``."""
    if width <= 0.0:
        return torch.ones_like(r), torch.zeros_like(r)
    w = min(width, cutoff)
    u = torch.clamp((cutoff - r) / w, 0.0, 1.0)
    sw = u ** 3 * (10.0 + u * (6.0 * u - 15.0))
    inside = (u > 0.0) & (u < 1.0)
    slope = torch.where(inside, -30.0 * u * u * (1.0 - u) ** 2 / w, torch.zeros_like(u))
    return sw, slope


def _triplet_block_grad(e0: int, e1: int, grad_out: Tensor, z: Tensor, edge_index: Tensor,
                        edge_vec: Tensor, r: Tensor, table: Tensor, s9: Tensor, c6_edge: Tensor,
                        r0_atom: Tensor, a1: Tensor, a2: Tensor, alp3: float, cutoff: float,
                        width: float, n_atoms: int, edge_shift: Tensor, sorted_keys: Tensor,
                        key_perm: Tensor, once: bool, cache: Optional[_TripletCache] = None):
    """Closed-form gradient of :func:`_triplet_block` (per-edge C6, per-atom radii).

    Given the cotangent ``grad_out`` ``(N,)`` of the per-atom energies, returns
    the gradients with respect to every block input in order (``None`` for
    the integer ones): the two edge vectors of each triplet through the three
    squared distances, the three pair ``C6`` through ``C9``, the radius
    factors and ``a1, a2`` through the damping, and ``s9``. One pass, no
    autograd graph; written in differentiable ops so that autograd over it
    provides the second derivative for force training. The triples come from
    ``cache`` when the forward block stored them, else they are enumerated.
    """
    cached = cache.take((e0, e1), edge_index) if cache is not None else None
    if cached is not None:
        a, b, c, j, k, distinct, e_jk, found = cached
    else:
        a, b, c, j, k, distinct = _block_triplets(e0, e1, edge_index, n_atoms, once)
    zeros_like = lambda t: torch.zeros_like(t)
    if a.numel() == 0:
        return (None, None, zeros_like(edge_vec), zeros_like(r), None, zeros_like(s9),
                zeros_like(c6_edge), zeros_like(r0_atom), zeros_like(a1), zeros_like(a2))
    # weight of each triplet: the cotangent of the corners that receive its thirds
    weight = grad_out.index_select(0, c)
    if once:
        weight = weight + torch.where(distinct, grad_out.index_select(0, j) + grad_out.index_select(0, k),
                                      torch.zeros_like(weight))
    # geometry
    v_ij, v_ik = edge_vec.index_select(0, a), edge_vec.index_select(0, b)
    x, y = (v_ij * v_ij).sum(-1), (v_ik * v_ik).sum(-1)
    v_jk = v_ij - v_ik
    w_ = (v_jk * v_jk).sum(-1)
    keep = (w_ <= cutoff * cutoff) & (w_ > 2.220446049250313e-16)
    w_ = torch.where(keep, w_, torch.ones_like(w_))
    ra, rb, rjk = torch.sqrt(x), torch.sqrt(y), torch.sqrt(w_)
    # C6 and C9
    c6_cj, c6_ck = c6_edge.index_select(0, a), c6_edge.index_select(0, b)
    if cached is None:
        e_jk, found = _third_side(a, b, j, k, n_atoms, edge_shift, sorted_keys, key_perm)
    c6_jk = torch.where(found, c6_edge.index_select(0, e_jk), torch.ones_like(c6_cj))
    c9 = torch.sqrt((c6_cj * c6_ck * c6_jk).abs())
    # radii
    rc, rj, rk = r0_atom.index_select(0, c), r0_atom.index_select(0, j), r0_atom.index_select(0, k)
    p_cj, p_ck, p_jk = a1 * rc * rj + a2, a1 * rc * rk + a2, a1 * rj * rk + a2
    r0 = p_cj * p_ck * p_jk
    # the summand and its pieces
    prod2 = x * y * w_
    prod1 = torch.sqrt(prod2)
    prod3 = prod2 * prod1
    prod5 = prod3 * prod2
    t = (r0 / prod1) ** alp3
    damp = 1.0 / (1.0 + 6.0 * t)
    f1, f2, f3 = x + w_ - y, x - w_ + y, -x + w_ + y
    nn = f1 * f2 * f3
    angular = 0.375 * nn / prod5 + 1.0 / prod3
    sa, sa_r = _switch_and_slope(ra, cutoff, width)
    sb, sb_r = _switch_and_slope(rb, cutoff, width)
    sjk, sjk_r = _switch_and_slope(rjk, cutoff, width)
    sw = sa * sb * sjk
    base = torch.where(keep, weight * s9 * c9 / 3.0, torch.zeros_like(weight))   # E = base * A D S
    e_tri = base * angular * damp * sw
    # d/dx, d/dy, d/dw of A, D, S (x, y, w the squared distances)
    n_x = f2 * f3 + f1 * f3 - f1 * f2
    n_y = -f2 * f3 + f1 * f3 + f1 * f2
    n_w = f2 * f3 - f1 * f3 + f1 * f2
    ang_x = 0.375 * (n_x / prod5 - 2.5 * nn / (prod5 * x)) - 1.5 / (prod3 * x)
    ang_y = 0.375 * (n_y / prod5 - 2.5 * nn / (prod5 * y)) - 1.5 / (prod3 * y)
    ang_w = 0.375 * (n_w / prod5 - 2.5 * nn / (prod5 * w_)) - 1.5 / (prod3 * w_)
    d_common = 3.0 * alp3 * t * damp * damp
    damp_x, damp_y, damp_w = d_common / x, d_common / y, d_common / w_
    sw_x = sa_r / (2.0 * ra) * sb * sjk
    sw_y = sb_r / (2.0 * rb) * sa * sjk
    sw_w = sjk_r / (2.0 * rjk) * sa * sb
    g_x = base * (ang_x * damp * sw + angular * damp_x * sw + angular * damp * sw_x)
    g_y = base * (ang_y * damp * sw + angular * damp_y * sw + angular * damp * sw_y)
    g_w = base * (ang_w * damp * sw + angular * damp_w * sw + angular * damp * sw_w)
    g_vij = 2.0 * (g_x[:, None] * v_ij + g_w[:, None] * v_jk)
    g_vik = 2.0 * (g_y[:, None] * v_ik - g_w[:, None] * v_jk)
    grad_edge_vec = torch.zeros_like(edge_vec).index_add(0, a, g_vij).index_add(0, b, g_vik)
    # C6 (through C9 = sqrt(c6 c6 c6)): dE/dc6 = E / (2 c6); the placeholder rows carry E = 0
    grad_c6 = torch.zeros_like(c6_edge).index_add(0, a, 0.5 * e_tri / c6_cj)
    grad_c6 = grad_c6.index_add(0, b, 0.5 * e_tri / c6_ck)
    grad_c6 = grad_c6.index_add(0, e_jk, torch.where(found, 0.5 * e_tri / c6_jk, torch.zeros_like(e_tri)))
    # radii (through the damping): dE/dr0 = -alp3 E 6 t D / r0, then the three pair factors
    g_r0 = -alp3 * e_tri * 6.0 * t * damp / r0
    g_pcj, g_pck, g_pjk = g_r0 * p_ck * p_jk, g_r0 * p_cj * p_jk, g_r0 * p_cj * p_ck
    grad_rho = torch.zeros_like(r0_atom).index_add(0, c, a1 * (g_pcj * rj + g_pck * rk))
    grad_rho = grad_rho.index_add(0, j, a1 * (g_pcj * rc + g_pjk * rk))
    grad_rho = grad_rho.index_add(0, k, a1 * (g_pck * rc + g_pjk * rj))
    grad_a1 = (g_pcj * rc * rj + g_pck * rc * rk + g_pjk * rj * rk).sum().reshape(a1.shape)
    grad_a2 = (g_pcj + g_pck + g_pjk).sum().reshape(a2.shape)
    grad_s9 = (torch.where(keep, weight * c9 / 3.0, torch.zeros_like(weight))
               * angular * damp * sw).sum().reshape(s9.shape)
    return (None, None, grad_edge_vec, torch.zeros_like(r), None, grad_s9, grad_c6, grad_rho,
            grad_a1, grad_a2)


def _center_sum(e_tri: Tensor, c: Tensor, n_atoms: int) -> Tensor:
    """Per-atom sums of triplet energies whose centers ``c`` arrive sorted.

    A segmented reduction (:func:`~xnn.common.models.ops.segment_sum`) instead
    of the atomic scatter: the triplets of a center are consecutive, so the
    reduction needs no atomics; the result is placed into the ``(N,)`` vector.
    """
    if c.numel() == 0:
        return e_tri.new_zeros(n_atoms)
    c0, c1 = int(c[0]), int(c[-1]) + 1
    lengths = torch.bincount(c - c0, minlength=c1 - c0)
    sums = segment_sum(e_tri, lengths)
    zeros = e_tri.new_zeros
    return torch.cat([zeros(c0), sums, zeros(n_atoms - c1)])


def three_body_energy_chunked(z: Tensor, edge_index: Tensor, edge_vec: Tensor, r: Tensor,
                              r0_table: Optional[Tensor], s9: Tensor, alp3: float, cutoff: float,
                              width: float, n_atoms: int, c6_mat: Optional[Tensor] = None,
                              alpha_a: Optional[Tensor] = None, alpha_b: Optional[Tensor] = None,
                              chunk: Optional[int] = None, r0_atom: Optional[Tensor] = None,
                              a1: Optional[Tensor] = None, a2: Optional[Tensor] = None,
                              c6_edge: Optional[Tensor] = None,
                              edge_shift: Optional[Tensor] = None,
                              once: bool = True, analytic: bool = True,
                              triplet_cache: Optional[float] = None) -> Tensor:
    """:func:`three_body_energy` with memory bounded by one block (eager only).

    The edges are sorted by center once, the centers are cut into blocks
    holding about ``chunk`` triplets each (read off the cumulative pair counts;
    ``None`` sizes the block from the free device memory, between 2^20 and
    2^24, since every block pays a fixed set-up in both passes), and every
    block -- triplet enumeration included -- runs as a
    :func:`~xnn.common.models.recompute.recompute` block, so the retained
    state is the neighbor list and the per-atom sums at every derivative
    order (force training included); the block is recomputed in the backward
    passes. Pair ``C6`` values come from a dense matrix ``c6_mat``
    ``(N, N)`` or, without one, from the polarizability factors ``alpha_a``,
    ``alpha_b`` ``(N, K)`` as ``sum_k alpha_a[i, k] alpha_b[j, k]`` (D4: the
    Casimir-Polder weights folded into ``alpha_a``), which removes the
    ``(N, N)`` table altogether. Faster still, ``c6_edge`` ``(E,)`` gives the
    pair C6 per directed edge (computed once, differentiable in whatever it
    depends on) together with the edges' integer cell shifts ``edge_shift``
    ``(E, 3)``: the two sides through the center are then scalar gathers and
    the third side is found by a binary search over packed pair keys, which
    removes the ``(T, 23)`` gathers and, above all, their contended
    index-accumulate backward (energy + forces 53 -> 15 ns per triplet on an
    A100). The pair critical radii come from the element
    table ``r0_table`` or, for BJ radii ``a1 sqrt(3 Q_A Q_B) + a2``, from the
    per-atom factors ``r0_atom = 3^(1/4) sqrt(Q)`` with the scalars ``a1, a2``
    (cheap gradients when those are trainable). With ``once`` (default) each
    triangle of distinct atoms is evaluated from its smallest-index corner
    only. With ``analytic`` (default) and the per-edge C6 / per-atom radius
    inputs, the block's first derivative is evaluated in closed form
    (:func:`_triplet_block_grad`) in one pass instead of re-running the block
    under autograd. On that path ``triplet_cache`` keeps each block's triples
    from the forward to the backward pass (:class:`_TripletCache`; GB, ``None``
    for a quarter of the free device memory, 0 to disable), so they are
    enumerated once per step. Results equal the scripted loop to rounding.
    """
    from .recompute import recompute
    energy = torch.zeros(n_atoms, dtype=r.dtype, device=r.device)
    if edge_index.shape[1] == 0:
        return energy
    # edges sorted by center once: a block is then an edge range, not a scan
    order = torch.argsort(edge_index[1], stable=True)
    edge_index = edge_index.index_select(1, order)
    edge_vec, r = edge_vec.index_select(0, order), r.index_select(0, order)
    if c6_edge is not None:
        c6_edge, edge_shift = c6_edge.index_select(0, order), edge_shift.index_select(0, order)
    counts = torch.bincount(edge_index[1], minlength=n_atoms)
    offsets = torch.cumsum(counts, 0)
    cum = torch.cumsum(counts * (counts - 1) // 2, 0)
    total = int(cum[-1])
    if total == 0:
        return energy
    if chunk is None:
        chunk = auto_triplet_chunk(r)
    bounds = [0, n_atoms]
    if total > chunk:
        marks = torch.arange(chunk, total, chunk, device=cum.device)
        cuts = torch.unique(torch.searchsorted(cum, marks, right=False) + 1).tolist()
        bounds = [0] + [c for c in cuts if 0 < c < n_atoms] + [n_atoms]
    edge_bounds = [0] + offsets.index_select(0, torch.tensor(bounds[1:], device=offsets.device) - 1).tolist()
    # the block sees tensors only; the C6 source (dense matrix or the two
    # polarizability factors) is passed through so its gradients flow
    use_edge = c6_edge is not None
    use_alpha = c6_mat is None and not use_edge
    use_atom = r0_atom is not None
    if use_atom:
        r0_args = (r0_atom, a1, a2)
        table = torch.zeros((1, 1), dtype=r.dtype, device=r.device)
    else:
        r0_args = ()
        table = r0_table
    if use_edge:
        keys = pair_edge_keys(edge_index, edge_shift, n_atoms)
        sorted_keys, key_perm = torch.sort(keys)
        c6_args = (c6_edge,)
        lookup = (edge_shift, sorted_keys, key_perm)     # constants of the block
    else:
        c6_args = (alpha_a, alpha_b) if use_alpha else (c6_mat,)
        lookup = (None, None, None)

    def block(e0: int, e1: int, z_, ei, ev, r_, r0, s9_, *rest):
        c6 = rest[:len(c6_args)]
        r0a = rest[len(c6_args):]
        return _triplet_block(e0, e1, z_, ei, ev, r_, r0, s9_, alp3, cutoff, width, n_atoms,
                              c6[0] if (c6_mat is not None and not use_edge) else None,
                              c6[0] if use_alpha else None, c6[1] if use_alpha else None,
                              r0a[0] if use_atom else None, r0a[1] if use_atom else None,
                              r0a[2] if use_atom else None,
                              c6[0] if use_edge else None, *lookup, once,
                              cache if use_analytic else None)

    use_analytic = analytic and use_edge and use_atom
    # the triple cache only pays when a backward pass will take the triples
    cache = None
    if use_analytic and torch.is_grad_enabled() and (edge_vec.requires_grad
                                                     or c6_edge.requires_grad):
        budget = triplet_cache_budget(r, triplet_cache)
        if budget > 0:
            cache = _TripletCache(budget)

    def grad_block(e0: int, e1: int, grad_out, z_, ei, ev, r_, tab, s9_, c6e, rho, a1_, a2_):
        return _triplet_block_grad(e0, e1, grad_out, z_, ei, ev, r_, tab, s9_, c6e, rho, a1_, a2_,
                                   alp3, cutoff, width, n_atoms, *lookup, once, cache)

    for e0, e1 in zip(edge_bounds[:-1], edge_bounds[1:]):
        if e1 > e0:
            energy = energy + recompute(
                lambda *t, e0=e0, e1=e1: block(e0, e1, *t),
                z, edge_index, edge_vec, r, table, s9, *c6_args, *r0_args,
                grad_fn=(lambda g, *t, e0=e0, e1=e1: grad_block(e0, e1, g, *t)) if use_analytic else None)
    return energy


def auto_triplet_chunk(like: Tensor) -> int:
    """Triplets per recompute block from the free device memory.

    A block's transient state is about 100 values of ``like.dtype`` plus 40
    bytes of indices per triplet, twice over in the backward pass; the block
    takes a quarter of the free memory within ``[2^20, 2^24]`` (CPU: ``2^22``).
    Fewer, larger blocks amortize the per-block set-up, which is paid in both
    passes.
    """
    if like.device.type != "cuda":
        return 1 << 22
    free, _ = torch.cuda.mem_get_info(like.device)
    per_triplet = 2 * (100 * like.element_size() + 40)
    return int(min(1 << 24, max(1 << 20, free // (4 * per_triplet))))



class DispersionCorrection(InteratomicPotential):
    """A dispersion term as an xnn potential, standalone or wrapped around a model.

    Standalone (``model=None``) the potential is the dispersion energy alone;
    given a short-range ``model`` it adds the dispersion energy to that
    model's prediction: the wrapper's ``cutoff`` is the larger of the model's
    and the term's cutoffs (the neighbor-list radius the data pipeline uses),
    and the wrapped model only ever sees the edges within its own cutoff.
    :class:`~xnn.common.models.d4.D4Dispersion` and
    :class:`~xnn.common.models.d3.D3Dispersion` are the two instances; enable
    either from a config with ``model.extra["dispersion"]`` (see
    :func:`~xnn.common.models.registry.build_model`).

    Parameters
    ----------
    term : torch.nn.Module
        The dispersion evaluator (:class:`~xnn.common.models.d4.DFTD4` or
        :class:`~xnn.common.models.d3.DFTD3`): exposes ``cutoff`` (Angstrom)
        and ``evaluate(atomic_numbers, pos, edge_index, edge_vec, batch,
        num_graphs, cell, pbc, total_charge)`` returning at least
        ``"node_energy"`` (eV), ``"energy_2body"``, ``"energy_3body"``,
        ``"coordination_numbers"`` and ``"node_features"``.
    model : InteratomicPotential or None, optional
        The short-range model to correct; ``None`` for pure dispersion.

    Attributes
    ----------
    term : torch.nn.Module
        The dispersion evaluator.
    model : InteratomicPotential or None
        The wrapped model.
    cutoff : float
        Neighbor-list radius (Angstrom): ``max(model.cutoff, term.cutoff)``.
    node_feature_dim : int
        The wrapped model's feature width, or the term's own per-atom
        descriptors standalone (so LES can wrap a pure dispersion model too).

    Notes
    -----
    ``forward`` returns the combined ``"energy"`` / ``"node_energy"`` and adds
    ``"energy_sr"`` (the wrapped model's energy), ``"energy_disp"``,
    ``"energy_2body"``, ``"energy_3body"`` ``(B,)`` and the term's per-atom
    quantities (``"coordination_numbers"``, and for D4 ``"eeq_charges"``,
    ``"polarizabilities"``, ``"dynamic_polarizabilities"``; for D3
    ``"c6_matrix"``). The total charge of each structure is read from
    ``data.total_charge`` (``(B,)``; zero when absent).
    """

    # output keys of ``term.evaluate`` that are renamed in the wrapper's output
    _RENAME = {"charges": "eeq_charges"}

    def __init__(self, term: nn.Module, model: Optional[InteratomicPotential] = None):
        super().__init__()
        self.term = term
        self.model = model
        inner_cutoff = float(getattr(model, "cutoff", 0.0)) if model is not None else 0.0
        self.cutoff = max(inner_cutoff, float(term.cutoff))
        self.node_feature_dim = (int(getattr(model, "node_feature_dim", 0))
                                 if model is not None else int(term.n_features))

    def inner_graph(self, data: AtomicGraph) -> AtomicGraph:
        """``data`` restricted to the edges within the wrapped model's cutoff."""
        inner_cutoff = float(getattr(self.model, "cutoff", self.cutoff))
        if inner_cutoff >= self.cutoff:
            return data
        with torch.no_grad():
            keep = torch.linalg.norm(data.edge_vectors(), dim=-1) < inner_cutoff
        return replace(data, edge_index=data.edge_index[:, keep],
                       cell_shifts=data.cell_shifts[keep])

    def dispersion(self, data: AtomicGraph) -> Dict[str, Tensor]:
        """Evaluate the dispersion term on a (batched) graph; see :func:`evaluate_on_graph`."""
        return evaluate_on_graph(self.term, data)

    def forward(self, data: AtomicGraph) -> Dict[str, Tensor]:
        """Wrapped-model prediction plus the dispersion energy.

        Parameters
        ----------
        data : AtomicGraph
            The batched graph, built with this wrapper's ``cutoff``.

        Returns
        -------
        dict of str to Tensor
            See the class notes.
        """
        disp = self.dispersion(data)
        node_disp = disp.pop("node_energy")
        energy_disp = disp.pop("energy")
        features = disp.pop("node_features")
        if self.model is None:
            out: Dict[str, Tensor] = {
                "node_energy": node_disp, "energy": energy_disp,
                "node_features": features,
                "energy_sr": torch.zeros_like(energy_disp)}
        else:
            out = dict(self.model(self.inner_graph(data)))
            out["energy_sr"] = out["energy"]
            out["energy"] = out["energy"] + energy_disp
            out["node_energy"] = out["node_energy"] + node_disp
        out["energy_disp"] = energy_disp
        for key, value in disp.items():
            out[self._RENAME.get(key, key)] = value
        return out


def evaluate_on_graph(term: nn.Module, data: AtomicGraph) -> Dict[str, Tensor]:
    """Run a dispersion term's ``evaluate`` on a (batched) :class:`AtomicGraph`.

    Fills in an all-zero cell / periodicity for molecular batches, reads
    ``data.total_charge`` when present (zeros otherwise) and adds
    ``"energy"`` ``(B,)`` to the keys the term returns.

    Parameters
    ----------
    term : torch.nn.Module
        :class:`~xnn.common.models.d3.DFTD3` or :class:`~xnn.common.models.d4.DFTD4`.
    data : AtomicGraph
        The batched graph (built with the term's cutoff).

    Returns
    -------
    dict of str to Tensor
        The term's outputs plus ``"energy"``.
    """
    b = data.num_graphs
    dtype, device = data.pos.dtype, data.pos.device
    cell = (data.cell if data.cell is not None
            else torch.zeros((b, 3, 3), dtype=dtype, device=device))
    pbc = (data.pbc if data.pbc is not None
           else torch.zeros((b, 3), dtype=torch.bool, device=device))
    charge = getattr(data, "total_charge", None)
    if charge is None:
        charge = torch.zeros(b, dtype=dtype, device=device)
    out = term.evaluate(data.atomic_numbers, data.pos, data.edge_index,
                        data.edge_vectors(), data.batch, b, cell, pbc, charge.to(dtype))
    out["energy"] = scatter_sum(out["node_energy"], data.batch, b)
    return out


def options_from_extra(extra: dict, keys) -> dict:
    """Pick the recognized keyword arguments out of a config ``extra`` dict.

    Parameters
    ----------
    extra : dict
        A ``ModelConfig.extra`` mapping (or its ``dispersion`` sub-mapping).
    keys : iterable of str
        The accepted option names.

    Returns
    -------
    dict
        The recognized options; unknown keys (such as ``name``) are ignored.
    """
    return {k: extra[k] for k in keys if k in extra}
