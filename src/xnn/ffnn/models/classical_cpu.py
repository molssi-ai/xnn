"""The classical kernels on the CPU: :func:`xnn.common.cpu_kernels.launch` runs them as C++ loops, with the
lane-wise twins of the helpers that reduce over a block, and :class:`Plan` holds the launches of one
molecular-dynamics step."""
from typing import Optional

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra import libdevice

from xnn.common import cpu_kernels

from . import classical_kernels as K

# the cell gradient of a vector: each lane adds its own shift (x) g (the GPU kernel sums the lanes of the
# block's first structure first)
cpu_kernels.override(K._cell_grad, '''
def _cell_grad(gcell_ptr, bat_ptr, s_ptr, a, e, m, gx, gy, gz):
    c = tl.load(bat_ptr + a, mask=m, other=0).to(tl.int64) * 9
    for i in tl.static_range(3):
        si = tl.load(s_ptr + e * 3 + i, mask=m, other=0.0).to(tl.float64)
        tl.atomic_add(gcell_ptr + c + 3 * i, si * gx, mask=m, sem="relaxed")
        tl.atomic_add(gcell_ptr + c + 3 * i + 1, si * gy, mask=m, sem="relaxed")
        tl.atomic_add(gcell_ptr + c + 3 * i + 2, si * gz, mask=m, sem="relaxed")
''')

launch = cpu_kernels.launch
# the lanes of a row of the pair kernel by rows (each lane walks every ROW_LANES-th partner)
ROW_LANES = 1


@triton.jit
def _moved(new_ptr, pos_ptr, ref_ptr, sc_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Copy the new positions ``(n, 3)`` into ``pos`` and count the atoms more than ``sc[0]`` (half the skin)
    from ``ref`` into ``out``."""
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n
    d2 = tl.zeros([BLOCK], dtype=tl.float64)
    for c in tl.static_range(3):
        x = tl.load(new_ptr + e * 3 + c, mask=m, other=0.0)
        tl.store(pos_ptr + e * 3 + c, x, mask=m)
        d = x - tl.load(ref_ptr + e * 3 + c, mask=m, other=0.0)
        d2 += d * d
    half = tl.load(sc_ptr)
    tl.atomic_add(out_ptr, tl.sum(tl.where(m & (d2 > half * half), 1, 0), 0), sem="relaxed")


@triton.jit
def _forces(g_ptr, f_ptr, n, BLOCK: tl.constexpr):
    """``f = -g`` and ``g = 0`` (the step's forces from its gradient, the gradient ready for the next step)."""
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n
    tl.store(f_ptr + e, -tl.load(g_ptr + e, mask=m, other=0.0), mask=m)
    tl.store(g_ptr + e, tl.zeros([BLOCK], dtype=tl.float64), mask=m)


@triton.jit
def _excl_reach(xp_ptr, xk_ptr, pos_ptr, inv_ptr, len_ptr, out_ptr, BLOCK: tl.constexpr):
    """The largest squared minimum-image distance of atom ``i`` to its partners in the exclusion rows."""
    i = tl.program_id(0).to(tl.int64)
    far = tl.load(pos_ptr + 3 * i) * 0.0
    for j in range(tl.load(xp_ptr + i), tl.load(xp_ptr + i + 1)):
        kk = tl.load(xk_ptr + j).to(tl.int64)
        r2 = far * 0.0
        for c in tl.static_range(3):
            d = tl.load(pos_ptr + 3 * kk + c) - tl.load(pos_ptr + 3 * i + c)
            d = d - tl.floor(d * tl.load(inv_ptr + c) + 0.5) * tl.load(len_ptr + c)
            r2 += d * d
        far = tl.maximum(far, r2)
    tl.store(out_ptr + i, far)


@triton.jit
def _kept(cnt_ptr, k_ptr, code_ptr, xp_ptr, xk_ptr, pos_ptr, off_ptr, sc_ptr, tk_ptr, tc_ptr, out_ptr, stride,
          EXCL: tl.constexpr, MAXE: tl.constexpr, BLOCK: tl.constexpr):
    """Atom ``i``'s staged partners (:func:`~xnn.common.data.grid_pairs.grid_pairs_staged`) that are not in its
    row of the exclusion rows (tested only within the distance ``sqrt(sc[0])``, the farthest excluded pair's),
    moved to the front of its slots in order (those within ``sqrt(sc[1])`` first, the others after them through the
    scratch slots ``tk``, ``tc``), and their number into ``out``."""
    i = tl.program_id(0).to(tl.int64)
    c = tl.load(cnt_ptr + i)
    reach = tl.load(sc_ptr)
    inner = tl.load(sc_ptr + 1)
    w = i * 0
    o = i * 0
    for j in range(0, c):
        kk = tl.load(k_ptr + i * stride + j)
        code = tl.load(code_ptr + i * stride + j)
        q = code.to(tl.int64) * 3
        x = tl.load(pos_ptr + 3 * kk + 0) - tl.load(pos_ptr + 3 * i + 0) + tl.load(off_ptr + q + 0)
        y = tl.load(pos_ptr + 3 * kk + 1) - tl.load(pos_ptr + 3 * i + 1) + tl.load(off_ptr + q + 1)
        z = tl.load(pos_ptr + 3 * kk + 2) - tl.load(pos_ptr + 3 * i + 2) + tl.load(off_ptr + q + 2)
        r2 = x * x + y * y + z * z
        drop = kk < 0
        if EXCL:
            if r2 <= reach:
                # the exclusion rows hold each pair in the row of its lower atom
                drop = K._excluded_row(xp_ptr, xk_ptr, tl.minimum(i, kk), tl.maximum(i, kk), kk >= 0, MAXE)
        if not drop:
            if r2 >= inner:
                tl.store(tk_ptr + i * stride + o, kk)
                tl.store(tc_ptr + i * stride + o, code)
                o += 1
            else:
                tl.store(k_ptr + i * stride + w, kk)
                tl.store(code_ptr + i * stride + w, code)
                w += 1
    for j in range(0, o):
        tl.store(k_ptr + i * stride + w + j, tl.load(tk_ptr + i * stride + j))
        tl.store(code_ptr + i * stride + w + j, tl.load(tc_ptr + i * stride + j))
    tl.store(out_ptr + i, w + o)


@triton.jit
def _compact(cnt_ptr, k_ptr, code_ptr, start_ptr, src_ptr, dst_ptr, img_ptr, stride, SRC: tl.constexpr,
             BLOCK: tl.constexpr):
    """Atom ``i``'s first ``cnt[i]`` staged partners from ``start[i]``: the pairs ``(i, k)`` (``i`` only with
    ``SRC``) and their image indices, in order."""
    i = tl.program_id(0).to(tl.int64)
    w = tl.load(start_ptr + i)
    for j in range(0, tl.load(cnt_ptr + i)):
        if SRC:
            tl.store(src_ptr + w + j, i)
        tl.store(dst_ptr + w + j, tl.load(k_ptr + i * stride + j))
        tl.store(img_ptr + w + j, tl.load(code_ptr + i * stride + j))


def _slots(work: Optional[dict], name: str, need: int) -> Optional[tuple]:
    """Staged partner and image-index buffers (int32, int8) of at least ``need`` entries kept in ``work`` from
    one list to the next (a quarter more than the first that needs them), new ones without ``work``."""
    if work is None:
        return torch.empty(need, dtype=torch.int32), torch.empty(need, dtype=torch.int8)
    if name not in work or work[name][0].numel() < need:
        work[name] = (torch.empty(need + need // 4, dtype=torch.int32), torch.empty(need + need // 4, dtype=torch.int8))
    return work[name]


def stepper_pairs(pos: Tensor, cutoff: float, grid: tuple, rows: Optional[tuple], inner: Optional[float] = None,
                  work: Optional[dict] = None, compact: bool = True, half: bool = False) -> Optional[tuple]:
    """A stepper's pairs ``i < k`` within ``cutoff`` on the CPU from one grid search (``grid`` of
    :func:`~xnn.common.data.neighborlist.cell_grid`), or from testing every pair where ``grid`` is the edges of a
    cell too small for one (:func:`small_cell`), the excluded pairs of the compressed rows ``rows`` (or ``None``)
    left out, each atom's partners within ``inner`` first where given: ``i``, ``k`` (int32), the image indices (int8) of
    :func:`~xnn.common.data.grid_pairs.grid_pairs_staged` and where each atom's pairs start ``(N + 1,)`` (int64),
    or ``None`` when the search needs its two passes; ``work`` a dict that keeps the search's buffers between
    calls. Without ``compact``: no list of ``i`` (a reader of rows needs none), ``(None, k, img, start, (beg,
    end))`` with atom ``i``'s partners ``k[beg[i]:end[i]]``. ``half`` (a grid): each pair once in the row of either
    atom (``k`` may be lower than ``i``; half the search's candidates)."""
    from xnn.common.data.grid_pairs import all_pairs_staged, grid_pairs_staged
    lengths = grid[0] if isinstance(grid, tuple) else grid
    n = int(pos.shape[0])
    volume = float(torch.prod(lengths.double()))
    # room for every partner of the atom with the most (an atom's partners k > i are at most all its neighbors)
    cap = min(int(1.3 * n / volume * 4.0 / 3.0 * 3.141592653589793 * cutoff ** 3) + 32, n)
    out = _slots(work, "slots", n * (cap + 1))
    if isinstance(grid, tuple):
        staged = grid_pairs_staged(pos, cutoff, lengths, grid[1], grid[3], cap, out, half=half)
    else:
        staged = all_pairs_staged(pos, cutoff, lengths, cap, out)
    if staged is None:
        return None
    counts, k, code = staged
    pos = pos.detach().contiguous()
    one = torch.arange(-1, 2, dtype=pos.dtype)
    off = (torch.cartesian_prod(one, one, one) * lengths.to(pos.dtype)).contiguous()
    if rows is not None or inner is not None:
        reach = 0.0
        if rows is not None:
            # no pair farther apart than the farthest excluded one is excluded
            far = torch.zeros(n, dtype=pos.dtype)
            L = lengths.to(pos.dtype).contiguous()
            cpu_kernels.launch(_excl_reach, (n,), rows[0], rows[1], pos, 1.0 / L, L, far, BLOCK=1)
            reach = float(far.max()) * (1.0 + 1.0e-9) + 1.0e-12 if n else 0.0
        sc = torch.tensor([reach, float("inf") if inner is None else float(inner) ** 2], dtype=pos.dtype)
        scratch = _slots(work, "scratch", int(k.numel()))
        xp, xk = (rows[0], rows[1]) if rows is not None else (counts, counts)
        kept = torch.empty(n, dtype=torch.int64)
        cpu_kernels.launch(_kept, (n,), counts, k, code, xp, xk, pos, off, sc, *scratch, kept, cap + 1,
                           EXCL=rows is not None, MAXE=int(rows[2]) if rows is not None else 0, BLOCK=1)
    else:
        kept = counts.to(torch.int64)
    start = torch.zeros(n + 1, dtype=torch.int64) if work is None else work.setdefault("start", torch.zeros(n + 1,
                                                                                                       dtype=torch.int64))
    torch.cumsum(kept, 0, out=start[1:])
    E = int(start[-1])
    if work is not None:
        # the list in buffers kept the same way: the pairs ``(2, E)`` as rows of one tensor
        if "pairs" not in work or work["pairs"].shape[1] < E:
            work["pairs"] = torch.empty(2, E + E // 4, dtype=torch.int32)
            work["img"] = torch.empty(E + E // 4, dtype=torch.int8)
        src, dst, img = work["pairs"][0, :E], work["pairs"][1, :E], work["img"][:E]
    else:
        src, dst = torch.empty(E, dtype=torch.int32), torch.empty(E, dtype=torch.int32)
        img = torch.empty(E, dtype=torch.int8)
    if E:
        cpu_kernels.launch(_compact, (n,), kept, k, code, start, src, dst, img, cap + 1, SRC=compact, BLOCK=1)
    if not compact:
        # no first atoms (read by rows): ``(None, k, img, start, (beg, end))``
        return None, dst, img, start, (start[:-1], start[1:])
    return src, dst, img, start


ALL_PAIRS_ATOMS = 8192


def small_cell(pos: Tensor, cutoff: float, cell: Optional[Tensor], pbc: Optional[Tensor]) -> Optional[Tensor]:
    """The edges of an orthorhombic cell, periodic on every axis, each edge over twice ``cutoff``, of a structure
    of at most ``ALL_PAIRS_ATOMS`` atoms on the CPU: the cells :func:`stepper_pairs` searches by testing every
    pair (no room for a grid); otherwise ``None``."""
    if cell is None or pbc is None or not bool(pbc.all()) or pos.device.type != "cpu":
        return None
    if int(pos.shape[0]) > ALL_PAIRS_ATOMS or not cpu_kernels.available():
        return None
    rows = cell.detach().reshape(3, 3).double()
    if bool((rows - torch.diag(torch.diagonal(rows))).abs().max() > 0):
        return None
    lengths = torch.diagonal(cell.detach().reshape(3, 3)).abs()
    return lengths if float(lengths.min()) > 2.0 * cutoff else None


def cpu_pairs(pos: Tensor, cutoff: float, cell: Optional[Tensor], pbc: Optional[Tensor],
              rows: Optional[tuple] = None, half: bool = False) -> Optional[tuple]:
    """:func:`stepper_pairs` on the grid of :func:`~xnn.common.data.neighborlist.cell_grid` or on a
    :func:`small_cell`, or ``None`` where neither applies."""
    from xnn.common.data.neighborlist import cell_grid
    grid = cell_grid(pos, cutoff, cell, pbc)
    if grid is None:
        grid = small_cell(pos, cutoff, cell, pbc)
    elif len(grid) != 4:
        return None
    return None if grid is None else stepper_pairs(pos, cutoff, grid, rows, half=half)


def decode_images(img: Tensor) -> Tensor:
    """The integer shifts ``(E, 3)`` of the image indices of :func:`stepper_pairs`."""
    c = img.to(torch.int64)
    return torch.stack([c // 9 - 1, (c // 3) % 3 - 1, c % 3 - 1], 1)


@triton.jit
def _min_images(pos_ptr, a_ptr, b_ptr, inv_ptr, out_ptr, M, BLOCK: tl.constexpr):
    """The shifts ``-rint((pos[b] - pos[a]) / L)`` per axis of one orthorhombic cell (``inv`` ``1 / L``)."""
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < M
    a = tl.load(a_ptr + e, mask=m, other=0).to(tl.int64)
    b = tl.load(b_ptr + e, mask=m, other=0).to(tl.int64)
    for c in tl.static_range(3):
        d = tl.load(pos_ptr + 3 * b + c, mask=m, other=0.0) - tl.load(pos_ptr + 3 * a + c, mask=m, other=0.0)
        tl.store(out_ptr + 3 * e + c, -libdevice.rint(d * tl.load(inv_ptr + c)), mask=m)


@triton.jit
def _hb_pair(ptr_ptr, dst_ptr, code_ptr, pos_ptr, off_ptr, hyd_ptr, acc_ptr, cut2, i, e, R: tl.constexpr):
    """List pair ``e`` of row ``i``: its partner, image index, whether it joins a donor ``i`` to an acceptor
    (``fwd``) or an acceptor ``i`` to a donor (``bwd``) within the cutoff, and the hydrogens of either donor."""
    k = tl.load(dst_ptr + e).to(tl.int64)
    c = tl.load(code_ptr + e).to(tl.int64)
    di = tl.load(hyd_ptr + i * R) >= 0
    dk = tl.load(hyd_ptr + k * R) >= 0
    fwd = di & (tl.load(acc_ptr + k) != 0)
    bwd = dk & (tl.load(acc_ptr + i) != 0)
    near = fwd & False
    if fwd | bwd:
        x = tl.load(pos_ptr + 3 * k + 0) - tl.load(pos_ptr + 3 * i + 0) + tl.load(off_ptr + 3 * c + 0)
        y = tl.load(pos_ptr + 3 * k + 1) - tl.load(pos_ptr + 3 * i + 1) + tl.load(off_ptr + 3 * c + 1)
        z = tl.load(pos_ptr + 3 * k + 2) - tl.load(pos_ptr + 3 * i + 2) + tl.load(off_ptr + 3 * c + 2)
        near = x * x + y * y + z * z < cut2
    return k, c, fwd & near, bwd & near


@triton.jit
def _hb_count(beg_ptr, end_ptr, dst_ptr, code_ptr, pos_ptr, off_ptr, hyd_ptr, acc_ptr, sc_ptr, nf_ptr, nb_ptr,
              R: tl.constexpr, BLOCK: tl.constexpr):
    """The hydrogen-bond triplets of row ``i`` of a stepper's list with ``i`` the donor (``nf``) and with ``i``
    the acceptor (``nb``)."""
    i = tl.program_id(0).to(tl.int64)
    cut2 = tl.load(sc_ptr)
    lo = tl.load(beg_ptr + i)
    # a row adds triplets only where its first atom is a donor or an acceptor
    role = (tl.load(hyd_ptr + i * R) >= 0) | (tl.load(acc_ptr + i) != 0)
    hi = tl.where(role, tl.load(end_ptr + i), lo)
    nh = i * 0
    for r in tl.static_range(R):
        nh += (tl.load(hyd_ptr + i * R + r) >= 0).to(tl.int64)
    nf = i * 0
    nb = i * 0
    for e in range(lo, hi):
        k, c, fwd, bwd = _hb_pair(beg_ptr, dst_ptr, code_ptr, pos_ptr, off_ptr, hyd_ptr, acc_ptr, cut2, i, e, R)
        if fwd:
            nf += nh
        if bwd:
            for t in tl.static_range(R):
                nb += (tl.load(hyd_ptr + k * R + t) >= 0).to(tl.int64)
    tl.store(nf_ptr + i, nf)
    tl.store(nb_ptr + i, nb)


@triton.jit
def _hb_emit(pos_ptr, hyd_ptr, sc_ptr, idx_ptr, sh_ptr, d, a, c, w, sign, H, R: tl.constexpr):
    """The triplets of donor ``d`` and acceptor ``a`` (image index ``c`` of ``a`` from ``d`` times ``sign``) from
    slot ``w``, one per hydrogen of ``d``; the next free slot."""
    for r in tl.static_range(R):
        h = tl.load(hyd_ptr + d * R + r)
        if h >= 0:
            tl.store(idx_ptr + w, d)
            tl.store(idx_ptr + H + w, h)
            tl.store(idx_ptr + 2 * H + w, a)
            tl.store(sh_ptr + 3 * w + 0, sign * (c // 9 - 1).to(tl.float64))
            tl.store(sh_ptr + 3 * w + 1, sign * ((c // 3) % 3 - 1).to(tl.float64))
            tl.store(sh_ptr + 3 * w + 2, sign * (c % 3 - 1).to(tl.float64))
            for q in tl.static_range(3):
                v = tl.load(pos_ptr + 3 * d + q) - tl.load(pos_ptr + 3 * h + q)
                tl.store(sh_ptr + 3 * H + 3 * w + q, -libdevice.rint(v * tl.load(sc_ptr + 1 + q)))
            w += 1
    return w


@triton.jit
def _hb_write(beg_ptr, end_ptr, dst_ptr, code_ptr, pos_ptr, off_ptr, hyd_ptr, acc_ptr, sc_ptr, of_ptr, ob_ptr, idx_ptr,
              sh_ptr, H, R: tl.constexpr, BLOCK: tl.constexpr):
    """Row ``i``'s triplets ``(donor, H, acceptor)`` from ``of[i]`` (``i`` the donor) and ``ob[i]`` (``i`` the
    acceptor), once per hydrogen, with the image shifts of the donor-acceptor and hydrogen-donor vectors
    ``(2, H, 3)``."""
    i = tl.program_id(0).to(tl.int64)
    cut2 = tl.load(sc_ptr)
    lo = tl.load(beg_ptr + i)
    role = (tl.load(hyd_ptr + i * R) >= 0) | (tl.load(acc_ptr + i) != 0)
    hi = tl.where(role, tl.load(end_ptr + i), lo)
    wf = tl.load(of_ptr + i)
    wb = tl.load(ob_ptr + i)
    for e in range(lo, hi):
        k, c, fwd, bwd = _hb_pair(beg_ptr, dst_ptr, code_ptr, pos_ptr, off_ptr, hyd_ptr, acc_ptr, cut2, i, e, R)
        if fwd:
            wf = _hb_emit(pos_ptr, hyd_ptr, sc_ptr, idx_ptr, sh_ptr, i, k, c, wf, 1.0, H, R)
        if bwd:
            wb = _hb_emit(pos_ptr, hyd_ptr, sc_ptr, idx_ptr, sh_ptr, k, i, c, wb, -1.0, H, R)


def hbond_triplets(rows: tuple, dst: Tensor, img: Tensor, pos: Tensor, lengths: Tensor, hyd: Tensor,
                   acceptor: Tensor, cutoff: float, out: Optional[tuple] = None) -> tuple:
    """The hydrogen-bond triplets ``(donor, H, acceptor)`` ``(3, H)`` of a stepper's list (atom ``i``'s partners
    ``dst[beg[i]:end[i]]`` for ``rows`` ``(beg, end)`` of :func:`stepper_pairs`, one orthorhombic cell of edges
    ``lengths``) within ``cutoff``, once per hydrogen of the
    donor (``hyd`` ``(N, R)``, -1 padded), the pairs with the donor first then those with the acceptor first, and
    the image shifts ``(2, H, 3)`` of their donor-acceptor and hydrogen-donor vectors, and ``H``; written into the
    first ``H`` columns of the buffers ``out`` ``(3, C)``, ``(2, C, 3)`` where they fit (``idx`` and the shifts
    are then those buffers)."""
    n = int(pos.shape[0])
    R = int(hyd.shape[1])
    one = torch.arange(-1, 2, dtype=pos.dtype)
    off = (torch.cartesian_prod(one, one, one) * lengths.to(pos.dtype)).contiguous()
    sc = torch.cat([torch.tensor([float(cutoff) ** 2], dtype=torch.float64), 1.0 / lengths.double()])
    hyd = hyd.to(torch.int64).contiguous()
    acc = acceptor.to(torch.int8).contiguous()
    nf, nb = torch.empty(n, dtype=torch.int64), torch.empty(n, dtype=torch.int64)
    args = (rows[0], rows[1], dst, img, pos, off, hyd, acc, sc)
    cpu_kernels.launch(_hb_count, (n,), *args, nf, nb, R=R, BLOCK=1)
    of, ob = torch.cumsum(nf, 0) - nf, torch.cumsum(nb, 0) - nb
    F = int(of[-1] + nf[-1]) if n else 0
    H = F + (int(ob[-1] + nb[-1]) if n else 0)
    if out is not None and int(out[0].shape[1]) >= H and out[1] is not None and out[1].dtype == pos.dtype:
        idx, sh = out
    else:
        idx, sh = torch.empty(3, H, dtype=torch.long), torch.empty(2, H, 3, dtype=pos.dtype)
    if H:
        # the columns of the buffers (their width the stride of the rows)
        cpu_kernels.launch(_hb_write, (n,), *args, of, ob + F, idx, sh, int(idx.shape[1]), R=R, BLOCK=1)
    return idx, sh, H


class BondedImages:
    """The image shifts of the vectors of a stepper's bonded terms in one orthorhombic cell, as
    :func:`~xnn.ffnn.models.classical_fast.image_shifts` gives them, from one launch: ``terms`` the
    ``(idx, kind)`` of each class."""

    def __init__(self, pos: Tensor, terms: list):
        from .classical_fast import VECTORS
        self.layout = [(len(VECTORS[kind]), int(idx.shape[1])) for idx, kind in terms]
        a = torch.cat([idx[f] for idx, kind in terms for f, _ in VECTORS[kind]]).contiguous()
        b = torch.cat([idx[t] for idx, kind in terms for _, t in VECTORS[kind]]).contiguous()
        M = int(a.shape[0])
        self.inv = torch.zeros(3, dtype=pos.dtype)
        self.out = torch.zeros(M, 3, dtype=pos.dtype)
        self.run = cpu_kernels.prepare(_min_images, (max(1, triton.cdiv(M, 1024)),), pos, a, b, self.inv, self.out,
                                       M, BLOCK=1024)

    def __call__(self, pos: Tensor, lengths: Tensor) -> list:
        self.run.bind(0, pos)
        self.inv.copy_(1.0 / lengths)
        self.run()
        out, start = [], 0
        for V, M in self.layout:
            out.append(self.out[start:start + V * M].reshape(V, M, 3))
            start += V * M
        return out


class Moves:
    """The positions of a step copied into a stepper's buffer and its list's test (whether an atom moved by
    more than half the skin) in one launch."""

    def __init__(self, new: Tensor, pos: Tensor, ref: Tensor, half_skin: float):
        self.count = torch.zeros(1, dtype=torch.int32)
        sc = torch.tensor([float(half_skin)], dtype=torch.float64)
        n = int(pos.shape[0])
        self.run = cpu_kernels.prepare(_moved, (triton.cdiv(n, 1024),), new, pos, ref, sc, self.count, n, BLOCK=1024)

    def __call__(self, new: Tensor) -> bool:
        self.run.bind(0, new)
        self.count.zero_()
        self.run()
        return bool(self.count[0] > 0)


_COEFFICIENTS: dict = {}


def torsion_coefficients(k: Tensor, n: Tensor, cg: Tensor, sg: Tensor, form: int, nmax: int) -> Tensor:
    """The torsion terms ``(M, S)`` of form 0 or 1 of
    :func:`~xnn.ffnn.models.classical_kernels._torsion` summed by multiplicity: per torsion ``sum K``, then
    ``a_n = sum K cg`` for ``n`` 1 to ``nmax`` (a multiplicity outside 2 to ``nmax`` counts as 1, as in the
    kernel), and for form 1 ``b_n = sum K sg``; the coefficient rows of forms 3 and 4 (kept for the same
    parameters)."""
    key = (k.data_ptr(), n.data_ptr(), cg.data_ptr(), sg.data_ptr(), k._version, tuple(k.shape), form, nmax)
    if key not in _COEFFICIENTS:
        M = int(k.shape[0])
        k, n, cg, sg = (x.reshape(M, -1) for x in (k, n, cg, sg))
        q = torch.where((n >= 2) & (n <= nmax), n, torch.ones_like(n)).long() - 1
        a = torch.zeros(M, nmax, dtype=k.dtype).scatter_add_(1, q, k * cg)
        cols = [k.sum(1, keepdim=True), a]
        if form == 1:
            cols.append(torch.zeros(M, nmax, dtype=k.dtype).scatter_add_(1, q, k * sg))
        if len(_COEFFICIENTS) > 32:
            _COEFFICIENTS.clear()
        _COEFFICIENTS[key] = torch.cat(cols, 1).contiguous()
    return _COEFFICIENTS[key]


class Plan:
    """The energy and the forces of one structure at the positions in ``pos`` (the tensor a stepper writes
    them into) from launches prepared once per list: each class of bonded terms and the pairs by one pass
    (:func:`~xnn.ffnn.models.classical_kernels._bonded_energy_forces`,
    :func:`~xnn.ffnn.models.classical_kernels._pairs_energy_forces`)."""

    def __init__(self, pos: Tensor, cell: Optional[Tensor], batch: Tensor, dtype: torch.dtype):
        self.pos, self.cell, self.batch = pos, cell, batch
        self.dtype = dtype
        self.dt = tl.float64 if dtype == torch.float64 else tl.float32
        n = int(pos.shape[0])
        self.gpos = torch.zeros(n, 3, dtype=torch.float64)
        self.forces = torch.zeros(n, 3, dtype=torch.float64)
        self.flip = cpu_kernels.prepare(_forces, (triton.cdiv(3 * n, 4096),), self.gpos, self.forces, 3 * n,
                                        BLOCK=4096)
        self.outs, self.specs = [], []
        self.sequences = {}
        # one structure: the kernels read its cell once (HAS_CELL 2)
        self.single = cell is not None and cell.reshape(-1, 9).shape[0] == 1

    def bonded(self, idx: Tensor, shifts: Optional[Tensor], kind: str, params: tuple, tiny: float, bound: float,
               options: Optional[dict] = None, extra: tuple = (0.0, 0.0)):
        """One class of bonded terms (the arguments of :func:`~xnn.ffnn.models.classical_kernels.bonded_terms`)."""
        M = int(idx.shape[1])
        if not M:
            return
        cell_t, bat_t, s_t, has = K._cell_args(self.pos, self.cell, self.batch, shifts)
        has = 2 if has and self.single else has
        sc = K._scalars((float(tiny), float(bound)) + tuple(float(x) for x in extra), self.pos.device)
        params = tuple(p.detach().contiguous() for p in params)
        params = params + (self.pos,) * (6 - len(params))
        out = torch.zeros(1, dtype=torch.float64)
        opts = dict(FORM=0, S=1, NMAX=1)
        if kind == "torsion":
            opts.update(options or {})
            if opts["FORM"] in (0, 1) and opts["S"] > 1:
                # the terms of each torsion summed by multiplicity once (constant parameters): one recurrence
                coef = torsion_coefficients(params[1], params[2], params[3], params[4], opts["FORM"], opts["NMAX"])
                params = (params[0], coef, coef, coef, coef, params[5])
                opts.update(FORM=opts["FORM"] + 3, S=1)
        self.specs.append((K._bonded_energy_forces, (triton.cdiv(M, K.BLOCK),),
                           (self.pos, cell_t, bat_t, s_t, idx.contiguous(), *params, sc, out, self.gpos, self.gpos, M),
                           dict(KIND=K._KINDS[kind], DT=self.dt, HAS_CELL=has, CELL_GRAD=False, BLOCK=K.BLOCK,
                                **opts), True))
        self.outs.append(out)

    def pairs(self, src: Tensor, dst: Tensor, shifts: Optional[Tensor], n_local: int, t: Tensor, q: Tensor,
              table: Tensor, ke: float, tiny: float, cutoff: float, width: float, form: int, by_rows: bool = False,
              images: bool = False, ptr: Optional[Tensor] = None, live: bool = False,
              rows: Optional[tuple] = None):
        """Pairs ``i < k`` without exclusions (the arguments of
        :func:`~xnn.ffnn.models.classical_kernels.pair_terms`); ``by_rows`` a list sorted by ``i`` of one
        structure, by rows (:func:`~xnn.ffnn.models.classical_kernels._pairs_rows_energy_forces`); ``images``
        ``shifts`` the image indices of :func:`stepper_pairs` (with ``by_rows``), ``ptr`` its rows (the offsets of
        a compact list) and ``rows`` ``(beg, end)`` where the rows are not contiguous (``src`` then ``None``);
        ``live`` the shifts read from their own buffer at each call (one the caller refills)."""
        E = int(ptr[-1]) if rows is not None else int(src.shape[0])
        if not E:
            return
        sc = K._scalars((float(ke), float(tiny), float(cutoff) ** 2, float(cutoff - width),
                         1.0 / width if width > 0 else 0.0), self.pos.device)
        # the list as 32-bit atoms and 8-bit image shifts: a pair kernel on a CPU streams its list from memory
        if int(self.pos.shape[0]) < 2 ** 31:
            src = None if src is None else src.to(torch.int32)
            dst = dst.to(torch.int32)
        # the image indices of stepper_pairs: no shifts to convert (their offsets below)
        has = self.cell is not None and shifts is not None and not images
        cell_t = bat_t = s_t = self.pos
        if has:
            cell_t = self.cell.reshape(-1, 9).to(self.pos.dtype).contiguous()
            bat_t = self.batch.contiguous()
            # ``live``: the shifts read where they are (a buffer the caller refills), else as bytes where they fit
            small = not live and (shifts.dtype == torch.int8 or float(shifts.abs().max()) <= 127)
            s_t = shifts.to(torch.int8 if small else self.pos.dtype).contiguous()
            has = 2 if self.single else True
        out = torch.zeros(2, dtype=torch.float64)
        q_t, tab_t = q.detach().contiguous().to(self.dtype), table.detach().contiguous().to(self.dtype)
        if by_rows and (images or not has or self.single):
            n = int(self.pos.shape[0])
            # where each first atom's pairs start (the list is sorted by it)
            if ptr is None:
                ptr = torch.searchsorted(src, torch.arange(n + 1, dtype=src.dtype))
            beg, end = rows if rows is not None else (ptr[:-1], ptr[1:])
            rows_cell, rows_sh, mode = cell_t, s_t, bool(has)
            if images:
                one = torch.arange(-1, 2, dtype=self.pos.dtype)
                lattice = self.cell.reshape(3, 3).to(self.pos.dtype)
                rows_cell = (torch.cartesian_prod(one, one, one) @ lattice).contiguous()
                rows_sh, mode = shifts, 2
            elif has and (E == 0 or int(s_t.abs().max()) <= 1):
                # one byte per pair: its image among the 27 offsets of shifts -1, 0, 1 (the cell is fixed)
                one = torch.arange(-1, 2, dtype=self.pos.dtype)
                images = torch.cartesian_prod(one, one, one)
                rows_cell = (images @ self.cell.reshape(3, 3).to(self.pos.dtype)).contiguous()
                rows_sh = ((s_t[:, 0].to(torch.int16) + 1) * 9 + (s_t[:, 1] + 1) * 3 + (s_t[:, 2] + 1)).to(torch.int8)
                mode = 2
            self.specs.append((K._pairs_rows_energy_forces, (n,),
                               (beg, end, dst.contiguous(), rows_sh.contiguous(), self.pos, rows_cell,
                                t.to(torch.int32),
                                q_t, tab_t, sc, out, self.gpos, int(table.shape[0])),
                               dict(DT=self.dt, HAS_CELL=mode, FORM=int(form), SWITCH=width > 0, BLOCK=ROW_LANES,
                                    balance=ptr.to(torch.int64).contiguous()), True))
            self.outs.append(out)
            return
        args = (src.contiguous(), dst.contiguous(), s_t, self.pos, cell_t, bat_t, self.pos, self.pos, t.contiguous(),
                q_t, tab_t, sc)
        self.specs.append((K._pairs_energy_forces, (triton.cdiv(E, K.BLOCK),),
                           (*args, out, self.gpos, self.gpos, E, int(n_local), int(table.shape[0])),
                           dict(DT=self.dt, HAS_CELL=has, CELL_GRAD=False, EXCL=False, MAXE=0, FORM=int(form),
                                SWITCH=width > 0, BLOCK=K.BLOCK), False))
        self.outs.append(out)

    def __call__(self, energy: bool = True) -> tuple:
        """The energy (float64, a 0-d tensor; ``None`` without ``energy``, the kernels then sum no energy) and the
        forces ``(N, 3)`` in the positions' precision."""
        if energy not in self.sequences:
            # the launches in one parallel region, the forces from the gradient last
            runs = [cpu_kernels.prepare(kernel, grid, *args, **meta, **({"ENERGY": energy} if flag else {}))
                    for kernel, grid, args, meta, flag in self.specs]
            # the term kernels add into shared buffers with no barrier between them, the gradient summed once, the
            # forces from it after
            self.sequences[energy] = cpu_kernels.Deferred(runs, [self.flip])
        for out in self.outs:
            out.zero_()
        self.sequences[energy]()
        forces = self.forces.to(self.pos.dtype, copy=True)
        if not energy:
            return None, forces
        return torch.tensor(sum(float(out.sum()) for out in self.outs), dtype=torch.float64), forces
