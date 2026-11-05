"""The Axilrod-Teller-Muto three-body term on CPUs: dense per-center tiles, compiled.

The reference (:func:`~xnn.common.models.dispersion.three_body_energy`) lists every
triangle three times, once from each corner, as pairs of edges sharing a center, and
evaluates them as separate tensor operations on gathered copies. On a CPU that is about
twenty nanoseconds per triangle, an order of magnitude above the fused loops of the
Fortran codes. Here the edges of every center are sorted by neighbor index and cut
into tiles; a job is a pair of tiles of one center, and every pair of neighbors in it
forms a triangle with the center. A triangle of three distinct atoms is evaluated once,
from its smallest-index corner (a center pairs only neighbors of larger index), and a
third of its energy goes to each of its atoms, which is the reference's bookkeeping. A
triangle with a repeated atom (two periodic images of one atom, or an image of the
center) is evaluated from every corner with a third of its energy each, in a flat pass
over those pairs. The tile and pair functions run compiled (``torch.compile``, the C++
and OpenMP code of inductor) on fixed batch shapes; their sources are marked dynamic, so
a new atom or edge count does not compile again. The gradients (forces, stress, the
dependence of C6 on the coordination numbers) are those of the compiled functions.
"""
import functools
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

#: Neighbors per tile: the largest; a structure whose centers have fewer neighbors takes the
#: smallest power of two (at least :data:`MIN_TILE`) that holds them.
TILE = 64
MIN_TILE = 16
#: Tile pairs per compiled call.
BATCH = 256
#: Repeated-atom pairs per compiled call.
PAIRS = 1 << 15
#: Squared distances at or below this are coincident images (the reference's guard).
EPS = 2.220446049250313e-16


def _atm(r2ij, r2ik, r2jk, c6ij, c6ik, c6jk, r0, s9, alp3: float):
    """ATM energy of triangles from the squared sides, the three C6 and the radius product."""
    c9 = s9 * torch.sqrt((c6ij * c6ik * c6jk).abs())
    prod2 = r2ij * r2ik * r2jk
    prod1 = torch.sqrt(prod2)
    prod3 = prod2 * prod1
    prod5 = prod3 * prod2
    damp = 1.0 / (1.0 + 6.0 * (r0 / prod1) ** alp3)
    angular = (0.375 * (r2ij + r2jk - r2ik) * (r2ij - r2jk + r2ik) * (-r2ij + r2jk + r2ik) / prod5
               + 1.0 / prod3)
    return c9 * angular * damp


def _switch(r, cutoff: float, width: float):
    """The quintic switch of :func:`~xnn.common.models.dispersion.switching_function`."""
    w = min(width, cutoff)
    x = torch.clamp((cutoff - r) / w, 0.0, 1.0)
    return x ** 3 * (10.0 + x * (6.0 * x - 15.0))


def _tile_energy(v, sw, src, z, c6_mat, r0_table, center, a0, b0, end, live, s9, alp3: float,
                 cut2: float, cutoff: float, width: float, T: int):
    """Tile-pair jobs: the energy of every triangle of distinct atoms (a third of it), summed
    per job over its pairs ``(B,)``, per first neighbor ``(B, T)`` and per second ``(B, T)``."""
    ar = torch.arange(T, device=v.device)
    ra, rb = a0[:, None] + ar, b0[:, None] + ar
    ok_a, ok_b = ra < end[:, None], rb < end[:, None]
    ra, rb = torch.minimum(ra, end[:, None] - 1), torch.minimum(rb, end[:, None] - 1)
    va, vb = v[ra], v[rb]                                          # (B, T, 3)
    atom_a, atom_b = src[ra], src[rb]
    same = (a0 == b0)[:, None, None]
    mask = (ok_a[:, :, None] & ok_b[:, None, :] & live[:, None, None]
            & (~same | (ar[:, None] < ar[None, :])) & (atom_a[:, :, None] != atom_b[:, None, :]))
    vjk = va[:, :, None, :] - vb[:, None, :, :]
    r2jk = (vjk * vjk).sum(-1)
    keep = mask & (r2jk <= cut2) & (r2jk > EPS)
    r2jk = torch.where(keep, r2jk, torch.ones_like(r2jk))
    r2a, r2b = (va * va).sum(-1), (vb * vb).sum(-1)
    c = center[:, None]
    zc, za, zb = z[c], z[atom_a], z[atom_b]
    r0 = (r0_table[zc, za][:, :, None] * r0_table[zc, zb][:, None, :]
          * r0_table[za[:, :, None], zb[:, None, :]])
    e = _atm(r2a[:, :, None], r2b[:, None, :], r2jk, c6_mat[c, atom_a][:, :, None],
             c6_mat[c, atom_b][:, None, :], c6_mat[atom_a[:, :, None], atom_b[:, None, :]], r0, s9, alp3)
    if width > 0.0:
        e = e * sw[ra][:, :, None] * sw[rb][:, None, :] * _switch(torch.sqrt(r2jk), cutoff, width)
    e = torch.where(keep, e, torch.zeros_like(e)) / 3.0
    return e.sum((1, 2)), e.sum(2), e.sum(1)


def _pair_energy(v, sw, src, z, c6_mat, r0_table, center, pj, pk, live, s9, alp3: float, cut2: float,
                 cutoff: float, width: float):
    """Repeated-atom pairs ``(j, k)`` of a center: a third of the triangle energy per visit ``(P,)``."""
    vj, vk = v[pj], v[pk]
    vjk = vj - vk
    r2jk = (vjk * vjk).sum(-1)
    keep = live & (r2jk <= cut2) & (r2jk > EPS)
    r2jk = torch.where(keep, r2jk, torch.ones_like(r2jk))
    aj, ak = src[pj], src[pk]
    zc, zj, zk = z[center], z[aj], z[ak]
    r0 = r0_table[zc, zj] * r0_table[zc, zk] * r0_table[zj, zk]
    e = _atm((vj * vj).sum(-1), (vk * vk).sum(-1), r2jk, c6_mat[center, aj], c6_mat[center, ak],
             c6_mat[aj, ak], r0, s9, alp3)
    if width > 0.0:
        e = e * sw[pj] * sw[pk] * _switch(torch.sqrt(r2jk), cutoff, width)
    return torch.where(keep, e, torch.zeros_like(e)) / 3.0


def _switch_slope(r, cutoff: float, width: float):
    """The quintic switch and its slope in ``r``."""
    w = min(width, cutoff)
    x = torch.clamp((cutoff - r) / w, 0.0, 1.0)
    return x ** 3 * (10.0 + x * (6.0 * x - 15.0)), -30.0 * x * x * (1.0 - x) * (1.0 - x) / w


def _atm_slopes(x, y, w, c9, tt, ip1, alp3: float):
    """ATM energy of triangles from the squared sides ``x`` (first edge), ``y`` (second) and ``w`` (third),
    the C9 coefficient, the damping power ``6 (R0 / sqrt(x y w))^alp3`` and ``1 / sqrt(x y w)``, and its
    derivatives in the squared sides."""
    ip2 = ip1 * ip1
    ip3 = ip2 * ip1
    ip5 = ip3 * ip2
    damp = 1.0 / (1.0 + tt)
    a, b, c = x + w - y, x - w + y, -x + w + y
    abc = a * b * c
    ang = 0.375 * abc * ip5 + ip3
    k = 0.9375 * abc * ip5 + 1.5 * ip3
    ix, iy, iw = y * w * ip2, x * w * ip2, x * y * ip2
    hd = damp * damp * (0.5 * alp3) * tt
    ex = c9 * ((0.375 * (b * c + a * c - a * b) * ip5 - k * ix) * damp + ang * hd * ix)
    ey = c9 * ((0.375 * (-b * c + a * c + a * b) * ip5 - k * iy) * damp + ang * hd * iy)
    ew = c9 * ((0.375 * (b * c - a * c + a * b) * ip5 - k * iw) * damp + ang * hd * iw)
    return c9 * ang * damp, ex, ey, ew


def _switched(e, ex, ey, ew, ra, rb, rk, cutoff: float, width: float):
    """``_atm_slopes`` times the switches of the three sides (lengths ``ra``, ``rb``, ``rk``)."""
    sa, da = _switch_slope(ra, cutoff, width)
    sb, db = _switch_slope(rb, cutoff, width)
    sk, dk = _switch_slope(rk, cutoff, width)
    s = sa * sb * sk
    return (e * s, ex * s + e * sb * sk * da / (2.0 * ra), ey * s + e * sa * sk * db / (2.0 * rb),
            ew * s + e * sa * sb * dk / (2.0 * rk))


def _tile_gradient(v, r, q, src, z, s6_mat, q_mat, dc6m, dc6n, r0_table, center, a0, b0, end, live, s9,
                   alp3: float, cut2: float, cutoff: float, width: float, T: int, factored: bool):
    """Tile-pair jobs as :func:`_tile_energy`, with the gradient of the summed energies (unit weights): per
    first and second neighbor the edge-vector gradient ``(B, T, 3)``, the sum of the triangle energies (for the
    C6 of the center's side) and the coordination-number derivatives through the third side's C6. ``s6_mat`` is
    ``sqrt(C6)``; with ``factored`` the damping power is the product of the per-edge factors ``q``
    (``(R0 / r)^alp3``) of the center's sides and ``q_mat`` of the third side."""
    ar = torch.arange(T, device=v.device)
    ra, rb = a0[:, None] + ar, b0[:, None] + ar
    ok_a, ok_b = ra < end[:, None], rb < end[:, None]
    ra, rb = torch.minimum(ra, end[:, None] - 1), torch.minimum(rb, end[:, None] - 1)
    va, vb = v[ra], v[rb]                                          # (B, T, 3)
    atom_a, atom_b = src[ra], src[rb]
    same = (a0 == b0)[:, None, None]
    mask = (ok_a[:, :, None] & ok_b[:, None, :] & live[:, None, None]
            & (~same | (ar[:, None] < ar[None, :])) & (atom_a[:, :, None] != atom_b[:, None, :]))
    vjk = va[:, :, None, :] - vb[:, None, :, :]
    w = (vjk * vjk).sum(-1)
    keep = mask & (w <= cut2) & (w > EPS)
    w = torch.where(keep, w, torch.ones_like(w))
    x, y = (va * va).sum(-1)[:, :, None], (vb * vb).sum(-1)[:, None, :]
    c = center[:, None]
    zc, za, zb = z[c], z[atom_a], z[atom_b]
    ab = (atom_a[:, :, None], atom_b[:, None, :])
    sjk = s6_mat[ab]
    c6jk = sjk * sjk
    c9 = s9 * s6_mat[c, atom_a][:, :, None] * s6_mat[c, atom_b][:, None, :] * sjk
    ip1 = torch.rsqrt(x * y * w)
    if factored:
        tt = 6.0 * q[ra][:, :, None] * q[rb][:, None, :] * q_mat[ab]
    else:
        r0 = (r0_table[zc, za][:, :, None] * r0_table[zc, zb][:, None, :]
              * r0_table[za[:, :, None], zb[:, None, :]])
        tt = 6.0 * (r0 * ip1) ** alp3
    e, ex, ey, ew = _atm_slopes(x, y, w, c9, tt, ip1, alp3)
    if width > 0.0:
        e, ex, ey, ew = _switched(e, ex, ey, ew, r[ra][:, :, None], r[rb][:, None, :], torch.sqrt(w), cutoff, width)
    zero = torch.zeros_like(e)
    e, ex, ey, ew = (torch.where(keep, t, zero) for t in (e, ex, ey, ew))
    third = e / 3.0
    g_a = 2.0 * (va * (ex.sum(2) + ew.sum(2))[..., None] - torch.matmul(ew, vb))
    g_b = 2.0 * (vb * (ey.sum(1) + ew.sum(1))[..., None] - torch.matmul(ew.transpose(1, 2), va))
    bk = 0.5 * e / c6jk
    return (third.sum((1, 2)), third.sum(2), third.sum(1), g_a, g_b, e.sum(2), e.sum(1),
            (bk * dc6m[ab]).sum(2), (bk * dc6n[ab]).sum(1))


def _tile_gradient_packed(v, r, q, src, z, s6_mat, q_mat, dc6m, dc6n, r0_table, center, a0, b0, end, live, s9,
                          alp3: float, cut2: float, cutoff: float, width: float, T: int, factored: bool):
    """:func:`_tile_gradient` with each side's results packed per job ``(B, 6 T)``: per neighbor its share of
    the energy, the C6 sum, the coordination-number derivative and the three gradient components."""
    e_c, e_a, e_b, g_a, g_b, m_a, m_b, dn_a, dn_b = _tile_gradient(
        v, r, q, src, z, s6_mat, q_mat, dc6m, dc6n, r0_table, center, a0, b0, end, live, s9, alp3, cut2, cutoff,
        width, T, factored)
    side_a = torch.cat([e_a.unsqueeze(-1), m_a.unsqueeze(-1), dn_a.unsqueeze(-1), g_a], -1)
    side_b = torch.cat([e_b.unsqueeze(-1), m_b.unsqueeze(-1), dn_b.unsqueeze(-1), g_b], -1)
    return e_c, side_a.reshape(side_a.shape[0], -1), side_b.reshape(side_b.shape[0], -1)


def _pair_gradient(v, r, src, z, s6_mat, dc6m, dc6n, r0_table, center, pj, pk, live, s9, alp3: float, cut2: float,
                   cutoff: float, width: float):
    """Repeated-atom pairs as :func:`_pair_energy`, with the gradient (a third of the triangle per visit): the
    energy ``(P,)``, the edge-vector gradients of the two edges ``(P, 3)``, the triangle energy share for the
    C6 of the center's sides and the coordination-number derivatives through the third side's C6."""
    vj, vk = v[pj], v[pk]
    d = vj - vk
    w = (d * d).sum(-1)
    keep = live & (w <= cut2) & (w > EPS)
    w = torch.where(keep, w, torch.ones_like(w))
    aj, ak = src[pj], src[pk]
    zc, zj, zk = z[center], z[aj], z[ak]
    r0 = r0_table[zc, zj] * r0_table[zc, zk] * r0_table[zj, zk]
    sjk = s6_mat[aj, ak]
    c6jk = sjk * sjk
    x, y = (vj * vj).sum(-1), (vk * vk).sum(-1)
    ip1 = torch.rsqrt(x * y * w)
    e, ex, ey, ew = _atm_slopes(x, y, w, s9 * s6_mat[center, aj] * s6_mat[center, ak] * sjk,
                                6.0 * (r0 * ip1) ** alp3, ip1, alp3)
    if width > 0.0:
        e, ex, ey, ew = _switched(e, ex, ey, ew, r[pj], r[pk], torch.sqrt(w), cutoff, width)
    zero = torch.zeros_like(e)
    e, ex, ey, ew = (torch.where(keep, t, zero) / 3.0 for t in (e, ex, ey, ew))
    g_j = 2.0 * (ex[:, None] * vj + ew[:, None] * d)
    g_k = 2.0 * (ey[:, None] * vk - ew[:, None] * d)
    bk = 0.5 * e / c6jk
    return e, g_j, g_k, bk * dc6m[aj, ak], bk * dc6n[aj, ak]



def _repeated_slopes(vj, vk, aj, ak, c, rj, rk, z, s6_mat, dc6m, dc6n, r0_table, keep, s9, alp3: float,
                     cut2: float, cutoff: float, width: float):
    """:func:`_pair_gradient` on blocks of pairs (any broadcast shape): a third of the triangle energy, the
    gradients of the two edges and their coordination-number derivatives."""
    d = vj - vk
    w = (d * d).sum(-1)
    keep = keep & (w <= cut2) & (w > EPS)
    w = torch.where(keep, w, torch.ones_like(w))
    zc, zj, zk = z[c], z[aj], z[ak]
    r0 = r0_table[zc, zj] * r0_table[zc, zk] * r0_table[zj, zk]
    sjk = s6_mat[aj, ak]
    c6jk = sjk * sjk
    x, y = (vj * vj).sum(-1), (vk * vk).sum(-1)
    ip1 = torch.rsqrt(x * y * w)
    e, ex, ey, ew = _atm_slopes(x, y, w, s9 * s6_mat[c, aj] * s6_mat[c, ak] * sjk, 6.0 * (r0 * ip1) ** alp3, ip1,
                                alp3)
    if width > 0.0:
        e, ex, ey, ew = _switched(e, ex, ey, ew, rj, rk, torch.sqrt(w), cutoff, width)
    zero = torch.zeros_like(e)
    e, ex, ey, ew = (torch.where(keep, t, zero) / 3.0 for t in (e, ex, ey, ew))
    g_j = 2.0 * (ex.unsqueeze(-1) * vj + ew.unsqueeze(-1) * d)
    g_k = 2.0 * (ey.unsqueeze(-1) * vk - ew.unsqueeze(-1) * d)
    bk = 0.5 * e / c6jk
    return e, g_j, g_k, bk * dc6m[aj, ak], bk * dc6n[aj, ak]


def _pack(e, dn, g):
    """Per row: (energy share 0, C6 sum, coordination-number derivative, gradient), ``(..., 6)``."""
    return torch.cat([torch.zeros_like(e).unsqueeze(-1), e.unsqueeze(-1), dn.unsqueeze(-1), g], -1)


def _self_gradient(v, r, src, z, s6_mat, dc6m, dc6n, r0_table, center, s, o0, end, lo, hi, live, s9, alp3: float,
                   cut2: float, cutoff: float, width: float, T: int):
    """Images of a center (row ``s``) with ``T`` other rows of the center from ``o0`` (outside its own group
    ``[lo, hi)``): the center's energies ``(B,)``, the image row's packed results ``(B, 6)`` and the other rows'
    ``(B, 6 T)``."""
    ar = torch.arange(T, device=v.device)
    ro = o0[:, None] + ar
    ok = live[:, None] & (ro < end[:, None]) & ((ro < lo[:, None]) | (ro >= hi[:, None]))
    ro = torch.minimum(ro, end[:, None] - 1)
    e, g_j, g_k, dn_j, dn_k = _repeated_slopes(
        v[s][:, None, :], v[ro], src[s][:, None], src[ro], center[:, None], r[s][:, None], r[ro], z, s6_mat, dc6m,
        dc6n, r0_table, ok, s9, alp3, cut2, cutoff, width)
    side_o = _pack(e, dn_k, g_k)
    return e.sum(1), _pack(e.sum(1), dn_j.sum(1), g_j.sum(1)), side_o.reshape(side_o.shape[0], -1)


def _group_gradient(v, r, src, z, s6_mat, dc6m, dc6n, r0_table, center, g0, size, live, s9, alp3: float,
                    cut2: float, cutoff: float, width: float, K: int):
    """The pairs inside groups of edges to one atom (rows ``g0`` to ``g0 + size``, at most ``K``): the
    center's energies ``(B,)`` and the group rows' packed results ``(B, K, 6)``."""
    ar = torch.arange(K, device=v.device)
    ok = live[:, None] & (ar < size[:, None])
    rows = torch.minimum(g0[:, None] + ar, (g0 + size - 1)[:, None])
    vr, rr = v[rows], r[rows]
    atom = src[g0][:, None, None]
    keep = ok[:, :, None] & ok[:, None, :] & (ar[:, None] < ar[None, :])
    e, g_j, g_k, dn_j, dn_k = _repeated_slopes(
        vr[:, :, None, :], vr[:, None, :, :], atom, atom, center[:, None, None], rr[:, :, None], rr[:, None, :], z,
        s6_mat, dc6m, dc6n, r0_table, keep, s9, alp3, cut2, cutoff, width)
    side = _pack(e.sum(2) + e.sum(1), dn_j.sum(2) + dn_k.sum(1), g_j.sum(2) + g_k.sum(1))
    return e.sum((1, 2)), side


@functools.lru_cache(maxsize=None)
def _compiled(name: str) -> Callable:
    fn = {"tiles": _tile_energy, "pairs": _pair_energy, "tiles_grad": _tile_gradient_packed,
          "pairs_grad": _pair_gradient, "self_grad": _self_gradient, "group_grad": _group_gradient}[name]
    compiled = torch.compile(fn, dynamic=False)

    def call(*args):
        # parallel loops over the jobs whatever the size of the first, compiling call: inductor decides on
        # threads from the compile-time size (a small molecule compiled first left larger systems serial)
        with torch._inductor.config.patch({"cpp.min_chunk_size": 1}):
            return compiled(*args)
    return call


@functools.lru_cache(maxsize=None)
def available() -> bool:
    """``torch.compile`` produces CPU code here (inductor and a C++ compiler); checked once."""
    try:
        f = torch.compile(lambda x: torch.exp(x) * 2.0 + 1.0, dynamic=False)
        x = torch.linspace(0.0, 1.0, 7, dtype=torch.float64)
        return bool(torch.allclose(f(x), torch.exp(x) * 2.0 + 1.0))
    except Exception:  # noqa: BLE001  (any compiler failure: the reference runs)
        return False


def _pairs_in_groups(starts: Tensor, sizes: Tensor) -> Tuple[Tensor, Tensor]:
    """Every pair ``(p, q)``, ``p < q``, of positions inside each group ``[start, start + size)``."""
    n_pairs = sizes * (sizes - 1) // 2
    keep = n_pairs > 0
    starts, sizes, n_pairs = starts[keep], sizes[keep], n_pairs[keep]
    if starts.numel() == 0:
        empty = starts.new_zeros(0)
        return empty, empty
    m = int(sizes.max())
    tri = torch.tril_indices(m, m, -1, device=starts.device)          # (2, m (m - 1) / 2): row > col
    local = torch.arange(int(n_pairs.sum()), device=starts.device) - torch.repeat_interleave(
        torch.cumsum(n_pairs, 0) - n_pairs, n_pairs)
    base = torch.repeat_interleave(starts, n_pairs)
    return base + tri[1][local], base + tri[0][local]


class Plan:
    """The tile-pair jobs and repeated-atom pairs of one edge list (sorted by center, then neighbor).

    Attributes
    ----------
    order : Tensor
        The edges in center, then neighbor order ``(E,)``.
    src, dst : Tensor
        Neighbor and center of the sorted edges.
    jobs : Tensor
        ``(J, 4)``: center, first row of tile a, first row of tile b, end of the center's rows.
    job_tiles : Tensor
        ``(J, 2)``: the numbers of tiles a and b among all centers' tiles.
    tile_rows, tile_end : Tensor
        Every tile's first row and its center's end of rows.
    pair_center, pair_j, pair_k : Tensor
        The repeated-atom pairs (sorted rows ``j < k``) and their center (built on first use).
    groups : Tensor
        ``(G, 3)``: center, first row and size of every group of two or more edges to one atom.
    self_rows : Tensor
        The rows of the centers' own images.
    """

    def __init__(self, edge_index: Tensor, n_atoms: int, tile: Optional[int] = None):
        src, dst = edge_index[0], edge_index[1]
        keys = dst * n_atoms + src
        self.order = torch.argsort(keys, stable=True)
        keys = keys[self.order]
        self.src, self.dst = src[self.order].contiguous(), dst[self.order].contiguous()
        atoms = torch.arange(n_atoms, device=src.device)
        ends = torch.searchsorted(keys, (atoms + 1) * n_atoms)
        # the first neighbor of larger index: a center pairs only those (one visit per triangle)
        first = torch.searchsorted(keys, atoms * n_atoms + atoms, right=True)
        if tile is None:
            most = int((ends - first).max()) if n_atoms > 0 else 0
            tile = MIN_TILE
            while tile < min(most, TILE):
                tile *= 2
        self.tile = tile
        m = torch.div(ends - first + tile - 1, tile, rounding_mode="floor").clamp(min=0)
        n_jobs = m * (m + 1) // 2
        total = int(n_jobs.sum())
        if total > 0:
            mm = int(m.max())
            tri = torch.tril_indices(mm, mm, 0, device=src.device)     # (tb, ta), ta <= tb, row-major
            local = torch.arange(total, device=src.device) - torch.repeat_interleave(
                torch.cumsum(n_jobs, 0) - n_jobs, n_jobs)
            center = torch.repeat_interleave(atoms, n_jobs)
            f = first[center]
            self.jobs = torch.stack([center, f + tri[1][local] * tile, f + tri[0][local] * tile, ends[center]], 1)
            base = torch.cumsum(m, 0) - m
            self.job_tiles = torch.stack([base[center] + tri[1][local], base[center] + tri[0][local]], 1)
            tile_center = torch.repeat_interleave(atoms, m)
            k = torch.arange(int(m.sum()), device=src.device) - torch.repeat_interleave(base, m)
            self.tile_rows, self.tile_end = first[tile_center] + k * tile, ends[tile_center]
        else:
            self.jobs = torch.zeros((0, 4), dtype=torch.long, device=src.device)
            self.job_tiles = torch.zeros((0, 2), dtype=torch.long, device=src.device)
            self.tile_rows = self.tile_end = torch.zeros(0, dtype=torch.long, device=src.device)
        # repeated atoms: pairs inside a group of edges to one atom (two images, or images of
        # the center among themselves), and every image of the center with every other neighbor
        self._keys, self._ends, self._n_atoms = keys, ends, n_atoms
        uniq, counts = torch.unique_consecutive(keys, return_counts=True)
        group_start = torch.cumsum(counts, 0) - counts
        multi = counts > 1
        self.groups = torch.stack([torch.div(uniq[multi], n_atoms, rounding_mode="floor"), group_start[multi],
                                   counts[multi]], 1)
        self.self_rows = torch.nonzero(self.src == self.dst).squeeze(-1)
        self.repeated = self.groups.shape[0] > 0 or self.self_rows.numel() > 0
        self._pairs = None

    def _repeated_pairs(self) -> Tuple[Tensor, Tensor, Tensor]:
        """The repeated-atom pairs, one by one (built on first use)."""
        if self._pairs is None:
            keys, ends, n_atoms = self._keys, self._ends, self._n_atoms
            pj, pk = _pairs_in_groups(self.groups[:, 1], self.groups[:, 2])
            self_rows = self.self_rows
            if self_rows.numel() > 0:
                c = self.dst[self_rows]
                starts = torch.cat([ends.new_zeros(1), ends[:-1]])     # first row of every center
                s_lo = torch.searchsorted(keys, c * n_atoms + c)         # the center's own group
                s_hi = torch.searchsorted(keys, c * n_atoms + c, right=True)
                n_other = (ends[c] - starts[c]) - (s_hi - s_lo)
                q = torch.arange(int(n_other.sum()), device=keys.device) - torch.repeat_interleave(
                    torch.cumsum(n_other, 0) - n_other, n_other)
                row = torch.repeat_interleave(self_rows, n_other)
                lo, hi, st = (torch.repeat_interleave(t, n_other) for t in (s_lo, s_hi, starts[c]))
                other = torch.where(q < lo - st, st + q, hi + (q - (lo - st)))
                pj, pk = torch.cat([pj, torch.minimum(row, other)]), torch.cat([pk, torch.maximum(row, other)])
            self._pairs = (self.dst[pj] if pj.numel() > 0 else pj, pj, pk)
        return self._pairs

    @property
    def pair_center(self) -> Tensor:
        return self._repeated_pairs()[0]

    @property
    def pair_j(self) -> Tensor:
        return self._repeated_pairs()[1]

    @property
    def pair_k(self) -> Tensor:
        return self._repeated_pairs()[2]

    def self_jobs(self, tile: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Every image of a center with every ``tile`` rows of the center's other neighbors: the jobs ``(S, 6)``
        (center, image row, first row of the tile, end of the center's rows, the center's own group), each
        job's tile number among all centers' tiles of ``tile`` rows, and those tiles' first rows and ends."""
        keys, ends, n_atoms = self._keys, self._ends, self._n_atoms
        starts = torch.cat([ends.new_zeros(1), ends[:-1]])
        n_tiles = torch.div(ends - starts + tile - 1, tile, rounding_mode="floor")
        base = torch.cumsum(n_tiles, 0) - n_tiles
        atoms = torch.arange(n_atoms, device=keys.device)
        tile_center = torch.repeat_interleave(atoms, n_tiles)
        k = torch.arange(int(n_tiles.sum()), device=keys.device) - torch.repeat_interleave(base, n_tiles)
        tile_rows, tile_end = starts[tile_center] + k * tile, ends[tile_center]
        c = self.dst[self.self_rows]
        lo = torch.searchsorted(keys, c * n_atoms + c)
        hi = torch.searchsorted(keys, c * n_atoms + c, right=True)
        per = n_tiles[c]
        kk = torch.arange(int(per.sum()), device=keys.device) - torch.repeat_interleave(torch.cumsum(per, 0) - per, per)
        cc = torch.repeat_interleave(c, per)
        jobs = torch.stack([cc, torch.repeat_interleave(self.self_rows, per), starts[cc] + kk * tile, ends[cc],
                            torch.repeat_interleave(lo, per), torch.repeat_interleave(hi, per)], 1)
        return jobs, base[cc] + kk, tile_rows, tile_end


def three_body_energy_cpu(z: Tensor, edge_index: Tensor, edge_vec: Tensor, r: Tensor, c6_mat: Tensor,
                          r0_table: Tensor, s9: Tensor, alp3: float, cutoff: float, width: float,
                          n_atoms: int, plan: Optional[Plan] = None) -> Tensor:
    """Per-atom ATM energies ``(N,)`` in hartree: the values of
    :func:`~xnn.common.models.dispersion.three_body_energy` (same arguments) on the compiled tiles.

    ``edge_index`` holds the edges within the three-body cutoff, ``edge_vec`` and ``r`` their
    vectors and lengths in bohr, ``c6_mat`` the dense pair C6; gradients reach ``edge_vec``,
    ``r`` and ``c6_mat``. ``plan`` reuses the jobs of an unchanged edge list.
    """
    from ..dispersion import switching_function
    if plan is None:
        plan = Plan(edge_index, n_atoms)
    energy = edge_vec.new_zeros(n_atoms)
    if plan.jobs.shape[0] == 0 and plan.pair_j.numel() == 0:
        return energy
    v = edge_vec[plan.order].contiguous()
    sw = switching_function(r, cutoff, width)[plan.order].contiguous() if width > 0.0 else r[plan.order]
    z = z.contiguous()
    c6_mat = c6_mat.contiguous()
    r0_table = r0_table.to(edge_vec.dtype).contiguous()
    s9 = torch.as_tensor(s9, dtype=edge_vec.dtype)
    cut2 = cutoff * cutoff
    # the sources change size with every structure: dynamic, so no new compilation per size
    for t, dims in ((v, (0,)), (sw, (0,)), (plan.src, (0,)), (z, (0,)), (c6_mat, (0, 1))):
        for d in dims:
            torch._dynamo.mark_dynamic(t, d)
    tiles = _compiled("tiles")
    jobs = plan.jobs
    n_jobs = jobs.shape[0]
    tile = plan.tile
    ar = torch.arange(tile, device=v.device)
    for i in range(0, n_jobs, BATCH):
        jb = jobs[i:i + BATCH]
        n = jb.shape[0]
        # the batch is as long as the jobs left (dynamic: no padding, no new compilation); a
        # single job is padded to two, as a size-one dimension would be specialized
        live = torch.arange(max(n, 2), device=v.device) < n
        if n < 2:
            jb = torch.cat([jb, jb[:1]])
        cols = [jb[:, k].contiguous() for k in range(4)]
        for t in cols + [live]:
            torch._dynamo.mark_dynamic(t, 0)
        e_c, e_a, e_b = tiles(v, sw, plan.src, z, c6_mat, r0_table, cols[0], cols[1], cols[2], cols[3], live,
                              s9, alp3, cut2, cutoff, width, tile)
        ra = torch.minimum(jb[:, 1:2] + ar, jb[:, 3:4] - 1)
        rb = torch.minimum(jb[:, 2:3] + ar, jb[:, 3:4] - 1)
        energy = (energy.index_add(0, jb[:, 0], e_c).index_add(0, plan.src[ra].reshape(-1), e_a.reshape(-1))
                  .index_add(0, plan.src[rb].reshape(-1), e_b.reshape(-1)))
    n_pairs = plan.pair_j.numel()
    if n_pairs > 0:
        pairs = _compiled("pairs")
        for i in range(0, n_pairs, PAIRS):
            sl = slice(i, i + PAIRS)
            c, pj, pk = plan.pair_center[sl], plan.pair_j[sl], plan.pair_k[sl]
            n = c.numel()
            live = torch.arange(max(n, 2), device=v.device) < n
            if n < 2:
                c, pj, pk = (torch.cat([t, t[:1]]) for t in (c, pj, pk))
            c, pj, pk = c.contiguous(), pj.contiguous(), pk.contiguous()
            for t in (c, pj, pk, live):
                torch._dynamo.mark_dynamic(t, 0)
            e = pairs(v, sw, plan.src, z, c6_mat, r0_table, c, pj, pk, live, s9, alp3, cut2, cutoff, width)
            energy = energy.index_add(0, c, e)
    return energy



def _tile_sums(tiles: Tensor, side: Tensor) -> Tuple[Tensor, Tensor]:
    """The rows of ``side`` summed per tile number in ``tiles``: the distinct tiles and their sums."""
    order = torch.argsort(tiles)
    uniq, counts = torch.unique_consecutive(tiles[order], return_counts=True)
    return uniq, torch.segment_reduce(side[order], "sum", lengths=counts)

def three_body_direct(z: Tensor, plan: Plan, v: Tensor, r: Tensor, c6_mat: Tensor, dc6m: Tensor, dc6n: Tensor,
                      r0_table: Tensor, s9: float, alp3: float, cutoff: float, width: float,
                      n_atoms: int) -> Tuple[Tensor, Tensor, Tensor]:
    """The ATM energies per atom ``(N,)`` and their gradient without autograd: in the edge vectors (in the plan's
    edge order, ``(E, 3)``) and in the coordination numbers ``(N,)``, through the pair C6 (``dc6m[i, j]`` and
    ``dc6n[i, j]``: the derivatives of ``c6_mat[i, j]`` in the coordination numbers of ``i`` and ``j``). ``v`` and
    ``r`` are the edge vectors and lengths (bohr) in the plan's order, of the edges within the cutoff."""
    energy = v.new_zeros(n_atoms)
    grad = torch.zeros_like(v)
    m_edge = v.new_zeros(v.shape[0])
    dn_edge = v.new_zeros(v.shape[0])
    cut2 = cutoff * cutoff
    z = z.contiguous()
    r0_table = r0_table.to(v.dtype).contiguous()
    s9 = torch.as_tensor(s9, dtype=v.dtype)
    s6_mat = torch.sqrt(c6_mat).contiguous()
    # without repeated-atom pairs (molecules, cells wider than twice the cutoff) every side of a triangle is an
    # edge, and the damping power splits into per-edge factors (R0 / r)^alp3
    factored = not plan.repeated
    q = (r0_table[z[plan.dst], z[plan.src]] / r) ** alp3
    q_mat = (v.new_zeros((n_atoms, n_atoms)).index_put_((plan.dst, plan.src), q) if factored
             else v.new_zeros((1, 1)))
    for t, dims in ((v, (0,)), (r, (0,)), (q, (0,)), (plan.src, (0,)), (z, (0,)), (s6_mat, (0, 1)),
                    (dc6m, (0, 1)), (dc6n, (0, 1))) + (((q_mat, (0, 1)),) if factored else ()):
        for d in dims:
            torch._dynamo.mark_dynamic(t, d)
    jobs, tile = plan.jobs, plan.tile
    ar = torch.arange(tile, device=v.device)
    # as many triangle slots per call as BATCH jobs of the largest tiles
    batch = BATCH * max(1, (TILE // tile) ** 2)
    if jobs.shape[0] > 0:
        tiles = _compiled("tiles_grad")
        # the jobs' results summed per tile (a batch's jobs cover few distinct tiles of a side), the tile sums into
        # the accumulator, and the accumulator onto the edges once at the end
        acc = v.new_zeros((plan.tile_rows.shape[0], 6 * tile))
        for i in range(0, jobs.shape[0], batch):
            jb, jt = jobs[i:i + batch], plan.job_tiles[i:i + batch]
            n = jb.shape[0]
            live = torch.arange(max(n, 2), device=v.device) < n
            if n < 2:
                jb, jt = torch.cat([jb, jb[:1]]), torch.cat([jt, jt[:1]])
            cols = [jb[:, k].contiguous() for k in range(4)]
            for t in cols + [live]:
                torch._dynamo.mark_dynamic(t, 0)
            e_c, side_a, side_b = tiles(
                v, r, q, plan.src, z, s6_mat, q_mat, dc6m, dc6n, r0_table, cols[0], cols[1], cols[2], cols[3], live,
                s9, alp3, cut2, cutoff, width, tile, factored)
            energy.index_add_(0, jb[:, 0], e_c)
            for tiles_of, side in ((jt[:, 0], side_a), (jt[:, 1], side_b)):
                acc.index_add_(0, *_tile_sums(tiles_of, side))
        rows = plan.tile_rows[:, None] + ar
        valid = rows < plan.tile_end[:, None]
        rows = rows[valid]
        acc = acc.reshape(-1, tile, 6)[valid]
        energy.index_add_(0, plan.src[rows], acc[:, 0])
        m_edge.index_add_(0, rows, acc[:, 1])
        dn_edge.index_add_(0, rows, acc[:, 2])
        grad.index_add_(0, rows, acc[:, 3:])
    if plan.repeated:
        # repeated atoms as blocks of rows (see Plan): their results per row, onto the edges at the end
        rows_acc = v.new_zeros((v.shape[0], 6))
        groups = plan.groups
        if groups.shape[0] > 0:
            kernel = _compiled("group_grad")
            k = -(-int(groups[:, 2].max()) // 8) * 8
            ar_k = torch.arange(k, device=v.device)
            step = max(1, BATCH * TILE * TILE // (k * k))
            for i in range(0, groups.shape[0], step):
                gb = groups[i:i + step]
                n = gb.shape[0]
                live = torch.arange(max(n, 2), device=v.device) < n
                if n < 2:
                    gb = torch.cat([gb, gb[:1]])
                cols = [gb[:, j].contiguous() for j in range(3)]
                for t in cols + [live]:
                    torch._dynamo.mark_dynamic(t, 0)
                e_c, side = kernel(v, r, plan.src, z, s6_mat, dc6m, dc6n, r0_table, cols[0], cols[1], cols[2], live,
                                   s9, alp3, cut2, cutoff, width, k)
                energy.index_add_(0, cols[0], e_c)
                valid = ar_k < cols[2][:, None]
                rows_acc.index_add_(0, (cols[1][:, None] + ar_k)[valid], side[valid])
        if plan.self_rows.numel() > 0:
            kernel = _compiled("self_grad")
            jobs_s, tiles_s, tile_rows_s, tile_end_s = plan.self_jobs(tile)
            acc_s = v.new_zeros((tile_rows_s.shape[0], 6 * tile))
            step = BATCH * TILE * TILE // tile
            for i in range(0, jobs_s.shape[0], step):
                jb, jt = jobs_s[i:i + step], tiles_s[i:i + step]
                n = jb.shape[0]
                live = torch.arange(max(n, 2), device=v.device) < n
                if n < 2:
                    jb, jt = torch.cat([jb, jb[:1]]), torch.cat([jt, jt[:1]])
                cols = [jb[:, j].contiguous() for j in range(6)]
                for t in cols + [live]:
                    torch._dynamo.mark_dynamic(t, 0)
                e_c, side_s, side_o = kernel(v, r, plan.src, z, s6_mat, dc6m, dc6n, r0_table, *cols, live, s9, alp3,
                                             cut2, cutoff, width, tile)
                energy.index_add_(0, cols[0], e_c)
                rows_acc.index_add_(0, cols[1], side_s)
                acc_s.index_add_(0, *_tile_sums(jt, side_o))
            rows = tile_rows_s[:, None] + ar
            valid = rows < tile_end_s[:, None]
            rows_acc.index_add_(0, rows[valid], acc_s.reshape(-1, tile, 6)[valid])
        m_edge += rows_acc[:, 1]
        dn_edge += rows_acc[:, 2]
        grad += rows_acc[:, 3:]
    # the C6 of the center's sides: through both coordination numbers of every edge
    c6e = c6_mat[plan.dst, plan.src]
    a6 = 0.5 * m_edge / torch.where(c6e != 0, c6e, torch.ones_like(c6e))
    dcn = (v.new_zeros(n_atoms).index_add(0, plan.dst, a6 * dc6m[plan.dst, plan.src])
           .index_add(0, plan.src, a6 * dc6n[plan.dst, plan.src] + dn_edge))
    return energy, grad, dcn
