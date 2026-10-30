"""The unique pairs of an orthorhombic periodic cell from a grid of cells, on the GPU.

The cell is divided into at least three cells per axis, each at least the cutoff
wide, so every pair within the cutoff joins an atom to one of the 27 cells around
it, and in exactly one image. One program per atom walks those cells twice: first
to count its partners of higher index, then to write them at its offset in the
prefix sum of the counts. The pairs come out sorted by their first atom, in the
same order in every run, with no atomic adds and no host read but the total.
"""
from typing import Optional

import torch
import triton
import triton.language as tl
from torch import Tensor

GRID_BLOCK, GRID_WARPS = 64, 2


@triton.jit
def _grid_pairs(pos_ptr, box_ptr, cell_ptr, atoms_ptr, start_ptr, count_ptr, offset_ptr, i_ptr, k_ptr, shift_ptr,
                NX, NY, NZ, cap, WRITE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    lx = tl.load(box_ptr + 0)
    ly = tl.load(box_ptr + 1)
    lz = tl.load(box_ptr + 2)
    cut2 = tl.load(box_ptr + 3)
    px = tl.load(pos_ptr + 3 * i + 0)
    py = tl.load(pos_ptr + 3 * i + 1)
    pz = tl.load(pos_ptr + 3 * i + 2)
    cx = tl.load(cell_ptr + 3 * i + 0)
    cy = tl.load(cell_ptr + 3 * i + 1)
    cz = tl.load(cell_ptr + 3 * i + 2)
    offs = tl.arange(0, BLOCK)
    base = tl.load(offset_ptr + i)
    count = tl.sum(tl.zeros([BLOCK], dtype=tl.int32), 0)
    for d in tl.static_range(27):
        c = ((((cx + d // 9 - 1 + NX) % NX) * NY + (cy + (d // 3) % 3 - 1 + NY) % NY) * NZ
             + (cz + d % 3 - 1 + NZ) % NZ)
        s = tl.load(start_ptr + c)
        e = tl.load(start_ptr + c + 1)
        for j0 in range(s, e, BLOCK):
            j = j0 + offs
            m = j < e
            k = tl.load(atoms_ptr + j, mask=m, other=0)
            # the minimum image: the cell is at least three cutoffs wide
            dx = tl.load(pos_ptr + 3 * k + 0, mask=m, other=0.0) - px
            dy = tl.load(pos_ptr + 3 * k + 1, mask=m, other=0.0) - py
            dz = tl.load(pos_ptr + 3 * k + 2, mask=m, other=0.0) - pz
            ix = tl.floor(dx / lx + 0.5)
            iy = tl.floor(dy / ly + 0.5)
            iz = tl.floor(dz / lz + 0.5)
            dx = dx - ix * lx
            dy = dy - iy * ly
            dz = dz - iz * lz
            hit = m & (k > i) & (dx * dx + dy * dy + dz * dz <= cut2)
            if WRITE:
                at = base + count + tl.cumsum(hit.to(tl.int32), 0) - 1
                # the pairs beyond the buffer's capacity are not written (the caller sees the count)
                put = hit & (at < cap)
                tl.store(i_ptr + at, tl.zeros([BLOCK], dtype=tl.int64) + i, mask=put)
                tl.store(k_ptr + at, k.to(tl.int64), mask=put)
                tl.store(shift_ptr + 3 * at + 0, (-ix).to(tl.int64), mask=put)
                tl.store(shift_ptr + 3 * at + 1, (-iy).to(tl.int64), mask=put)
                tl.store(shift_ptr + 3 * at + 2, (-iz).to(tl.int64), mask=put)
            count += tl.sum(hit.to(tl.int32), 0)
    if not WRITE:
        tl.store(count_ptr + i, count)


def grid_shape(lengths: list, cutoff: float) -> Optional[tuple]:
    """The cells per axis of a grid over an orthorhombic cell with edges ``lengths``:
    as many as fit at least ``cutoff`` wide, or ``None`` with fewer than three on an axis."""
    shape = []
    for length in lengths:
        n = int(length // cutoff)
        if n > 0 and length / n < cutoff:
            n -= 1
        if n < 3:
            return None
        shape.append(n)
    return tuple(shape)


def grid_pairs(pos: Tensor, cutoff: float, lengths: Tensor, shape: tuple, capacity: Optional[int] = None) -> tuple:
    """Every pair within ``cutoff`` once, as ``(i, k, shift)`` with ``i < k``, sorted by ``i``.

    Parameters
    ----------
    pos : Tensor
        Cartesian positions ``(N, 3)`` on the GPU, wrapped into the cell or not.
    cutoff : float
        Neighbor cutoff radius.
    lengths : Tensor
        The edges of the orthorhombic cell ``(3,)``.
    shape : tuple of int
        The cells per axis, from :func:`grid_shape`.
    capacity : int, optional
        Write the pairs into buffers of this many entries, without reading their number
        on the host: the entries past it repeat the first pair, and their number comes
        back on the device (a pair list staged behind a running step).

    Returns
    -------
    tuple of Tensor
        ``i`` and ``k`` ``(M,)`` and the integer image shifts ``(M, 3)``, with the
        displacement ``pos[k] - pos[i] + shift * lengths``; with ``capacity`` the
        buffers ``(capacity,)`` and the number of pairs ``(1,)`` (more than the capacity
        when they did not fit).
    """
    N, device = int(pos.shape[0]), pos.device
    pos = pos.detach().contiguous()
    lengths = lengths.to(device=device, dtype=pos.dtype)
    # the edges and the squared cutoff, read by the kernel in the positions' precision
    box = torch.cat([lengths, torch.full((1,), float(cutoff) ** 2, dtype=pos.dtype, device=device)])
    dims = torch.tensor(shape, device=device)
    frac = pos / lengths
    cell = ((frac - torch.floor(frac)) * dims).to(torch.int64).clamp_max(dims - 1)
    flat = (cell[:, 0] * shape[1] + cell[:, 1]) * shape[2] + cell[:, 2]
    # the atoms by cell (stable: by index within a cell) and where each cell starts
    atoms = torch.argsort(flat.to(torch.int32), stable=True).to(torch.int32)
    per_cell = torch.zeros(shape[0] * shape[1] * shape[2], dtype=torch.int64, device=device)
    per_cell.index_add_(0, flat, torch.ones_like(flat))
    start = torch.zeros(per_cell.shape[0] + 1, dtype=torch.int32, device=device)
    start[1:] = torch.cumsum(per_cell, 0)
    cell = cell.to(torch.int32).contiguous()
    counts = torch.empty(N, dtype=torch.int32, device=device)
    args = (pos, box, cell, atoms, start)
    sizes = shape
    if N:
        _grid_pairs[(N,)](*args, counts, counts, counts, counts, counts, *sizes, 0, WRITE=False, BLOCK=GRID_BLOCK,
                          num_warps=GRID_WARPS)
    offset = torch.zeros(N + 1, dtype=torch.int64, device=device)
    offset[1:] = torch.cumsum(counts, 0)
    M = int(offset[-1]) if capacity is None else int(capacity)
    i = torch.zeros(M, dtype=torch.int64, device=device)
    k = torch.zeros(M, dtype=torch.int64, device=device)
    shift = torch.zeros(M, 3, dtype=torch.int64, device=device)
    if M and N:
        _grid_pairs[(N,)](*args, counts, offset, i, k, shift, *sizes, M, WRITE=True, BLOCK=GRID_BLOCK,
                          num_warps=GRID_WARPS)
    if capacity is None:
        return i, k, shift
    # the padding repeats the first pair (a pair of distinct atoms: no zero distance in the arithmetic)
    total = offset[-1:]
    pad = torch.arange(M, device=device) >= total
    return (torch.where(pad, i[:1], i), torch.where(pad, k[:1], k), torch.where(pad[:, None], shift[:1], shift),
            total)
