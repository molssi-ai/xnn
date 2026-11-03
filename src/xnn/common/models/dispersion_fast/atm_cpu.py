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


@functools.lru_cache(maxsize=None)
def _compiled(name: str) -> Callable:
    fn = {"tiles": _tile_energy, "pairs": _pair_energy}[name]
    return torch.compile(fn, dynamic=False)


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
    pair_center, pair_j, pair_k : Tensor
        The repeated-atom pairs (sorted rows ``j < k``) and their center.
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
        else:
            self.jobs = torch.zeros((0, 4), dtype=torch.long, device=src.device)
        # repeated atoms: pairs inside a group of edges to one atom (two images, or images of
        # the center among themselves), and every image of the center with every other neighbor
        uniq, counts = torch.unique_consecutive(keys, return_counts=True)
        group_start = torch.cumsum(counts, 0) - counts
        pj, pk = _pairs_in_groups(group_start, counts)
        self_rows = torch.nonzero(self.src == self.dst).squeeze(-1)
        if self_rows.numel() > 0:
            c = self.dst[self_rows]
            starts = torch.cat([ends.new_zeros(1), ends[:-1]])         # first row of every center
            s_lo = torch.searchsorted(keys, c * n_atoms + c)             # the center's own group
            s_hi = torch.searchsorted(keys, c * n_atoms + c, right=True)
            n_other = (ends[c] - starts[c]) - (s_hi - s_lo)
            q = torch.arange(int(n_other.sum()), device=src.device) - torch.repeat_interleave(
                torch.cumsum(n_other, 0) - n_other, n_other)
            row = torch.repeat_interleave(self_rows, n_other)
            lo, hi, st = (torch.repeat_interleave(t, n_other) for t in (s_lo, s_hi, starts[c]))
            other = torch.where(q < lo - st, st + q, hi + (q - (lo - st)))
            pj, pk = torch.cat([pj, torch.minimum(row, other)]), torch.cat([pk, torch.maximum(row, other)])
        self.pair_j, self.pair_k = pj, pk
        self.pair_center = self.dst[pj] if pj.numel() > 0 else pj


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
