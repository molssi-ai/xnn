"""The unique pairs of an orthorhombic periodic cell from a grid of cells, on the GPU (or on the CPU, one lane per
program, through :mod:`xnn.common.cpu_kernels`).

The cell is divided into at least ``2 SPAN + 1`` cells per axis, each at least the
cutoff over ``SPAN`` wide, so every pair within the cutoff joins an atom to one of the
``(2 SPAN + 1)^3`` cells around it, and in exactly one image (``SPAN`` 1 on a GPU, 2 on
a CPU where the grid allows: fewer candidates in smaller cells). One program per atom walks those cells twice: first
to count its partners of higher index, then to write them at its offset in the
prefix sum of the counts. The pairs come out sorted by their first atom, in the
same order in every run, with no atomic adds and no host read but the total.
A cell too small for the grid, with every edge over the cutoff, is searched
without one: each atom against every atom of higher index (a GPU).
"""
from typing import Optional

import torch
import triton
import triton.language as tl
from torch import Tensor

GRID_BLOCK, GRID_WARPS = 64, 2


@triton.jit
def _grid_pairs(pos_ptr, box_ptr, cell_ptr, atoms_ptr, start_ptr, count_ptr, offset_ptr, i_ptr, k_ptr, shift_ptr,
                NX, NY, NZ, cap, WRITE: tl.constexpr, BLOCK: tl.constexpr, SPAN: tl.constexpr = 1):
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
    for d in tl.static_range((2 * SPAN + 1) ** 3):
        c = ((((cx + d // ((2 * SPAN + 1) * (2 * SPAN + 1)) - SPAN + NX) % NX) * NY
              + (cy + (d // (2 * SPAN + 1)) % (2 * SPAN + 1) - SPAN + NY) % NY) * NZ
             + (cz + d % (2 * SPAN + 1) - SPAN + NZ) % NZ)
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
                # in the outputs' types (the stores convert)
                tl.store(i_ptr + at, tl.zeros([BLOCK], dtype=tl.int64) + i, mask=put)
                tl.store(k_ptr + at, k.to(tl.int64), mask=put)
                tl.store(shift_ptr + 3 * at + 0, -ix, mask=put)
                tl.store(shift_ptr + 3 * at + 1, -iy, mask=put)
                tl.store(shift_ptr + 3 * at + 2, -iz, mask=put)
            count += tl.sum(hit.to(tl.int32), 0)
    if not WRITE:
        tl.store(count_ptr + i, count)


@triton.jit
def _small_cell_pairs(pos_ptr, box_ptr, count_ptr, offset_ptr, i_ptr, k_ptr, shift_ptr, N, cap,
                      WRITE: tl.constexpr, BLOCK: tl.constexpr):
    """:func:`_grid_pairs` for a cell with every edge over the cutoff and no room for a grid: atom ``i`` against
    every atom of higher index, in the minimum image and, on each axis, the image on the far side (any other
    image lies a whole edge away)."""
    i = tl.program_id(0)
    lx = tl.load(box_ptr + 0)
    ly = tl.load(box_ptr + 1)
    lz = tl.load(box_ptr + 2)
    cut2 = tl.load(box_ptr + 3)
    px = tl.load(pos_ptr + 3 * i + 0)
    py = tl.load(pos_ptr + 3 * i + 1)
    pz = tl.load(pos_ptr + 3 * i + 2)
    offs = tl.arange(0, BLOCK)
    base = tl.load(offset_ptr + i)
    count = tl.sum(tl.zeros([BLOCK], dtype=tl.int32), 0)
    # a while loop: its bound depends on the program (Triton's interpreter runs no such range)
    k0 = i + 1
    while k0 < N:
        k = k0 + offs
        m = k < N
        dx = tl.load(pos_ptr + 3 * k + 0, mask=m, other=0.0) - px
        dy = tl.load(pos_ptr + 3 * k + 1, mask=m, other=0.0) - py
        dz = tl.load(pos_ptr + 3 * k + 2, mask=m, other=0.0) - pz
        ix = tl.floor(dx / lx + 0.5)
        iy = tl.floor(dy / ly + 0.5)
        iz = tl.floor(dz / lz + 0.5)
        dx = dx - ix * lx
        dy = dy - iy * ly
        dz = dz - iz * lz
        # the lattice step to the image on the far side of each axis
        ax = tl.where(dx > 0, -1.0, 1.0)
        ay = tl.where(dy > 0, -1.0, 1.0)
        az = tl.where(dz > 0, -1.0, 1.0)
        for c in tl.static_range(8):
            bx = (c // 4) % 2
            by = (c // 2) % 2
            bz = c % 2
            ex = dx + ax * lx * bx
            ey = dy + ay * ly * by
            ez = dz + az * lz * bz
            hit = m & (ex * ex + ey * ey + ez * ez <= cut2)
            if WRITE:
                at = base + count + tl.cumsum(hit.to(tl.int32), 0) - 1
                put = hit & (at < cap)
                tl.store(i_ptr + at, tl.zeros([BLOCK], dtype=tl.int64) + i, mask=put)
                tl.store(k_ptr + at, k.to(tl.int64), mask=put)
                tl.store(shift_ptr + 3 * at + 0, ax * bx - ix, mask=put)
                tl.store(shift_ptr + 3 * at + 1, ay * by - iy, mask=put)
                tl.store(shift_ptr + 3 * at + 2, az * bz - iz, mask=put)
            count += tl.sum(hit.to(tl.int32), 0)
        k0 += BLOCK
    if not WRITE:
        tl.store(count_ptr + i, count)


@triton.jit
def _bin_cells(pos_ptr, box_ptr, cell_ptr, fill_ptr, start_ptr, atoms_ptr, spos_ptr, swrap_ptr, ext_ptr, N, NX, NY, NZ,
               BLOCK: tl.constexpr):
    """The atoms by cell in one program on the CPU (a counting sort, by index within a cell): each atom's cell
    ``(N, 3)``, where each cell starts in ``atoms``, and in that order the positions wrapped into the cell
    (``spos``) with the whole edges taken off (``swrap``); ``fill`` the zeroed counts per cell; ``ext`` the
    positions' extent over the edges (the largest per axis)."""
    lx = tl.load(box_ptr + 0)
    ly = tl.load(box_ptr + 1)
    lz = tl.load(box_ptr + 2)
    x0 = tl.load(pos_ptr + 0)
    y0 = tl.load(pos_ptr + 1)
    z0 = tl.load(pos_ptr + 2)
    x1 = x0
    y1 = y0
    z1 = z0
    for a in range(0, N):
        px = tl.load(pos_ptr + 3 * a + 0)
        py = tl.load(pos_ptr + 3 * a + 1)
        pz = tl.load(pos_ptr + 3 * a + 2)
        x0 = tl.minimum(x0, px)
        y0 = tl.minimum(y0, py)
        z0 = tl.minimum(z0, pz)
        x1 = tl.maximum(x1, px)
        y1 = tl.maximum(y1, py)
        z1 = tl.maximum(z1, pz)
        # the cell of the wrapped position (the search's, of the same expressions)
        fx = (px - tl.floor(px * (1.0 / lx)) * lx) * (1.0 / lx)
        fy = (py - tl.floor(py * (1.0 / ly)) * ly) * (1.0 / ly)
        fz = (pz - tl.floor(pz * (1.0 / lz)) * lz) * (1.0 / lz)
        cx = tl.minimum(tl.maximum((fx * NX).to(tl.int32), 0), NX - 1)
        cy = tl.minimum(tl.maximum((fy * NY).to(tl.int32), 0), NY - 1)
        cz = tl.minimum(tl.maximum((fz * NZ).to(tl.int32), 0), NZ - 1)
        tl.store(cell_ptr + 3 * a + 0, cx)
        tl.store(cell_ptr + 3 * a + 1, cy)
        tl.store(cell_ptr + 3 * a + 2, cz)
        c = (cx * NY + cy) * NZ + cz
        tl.store(fill_ptr + c, tl.load(fill_ptr + c) + 1)
    w = N * 0
    for c in range(0, NX * NY * NZ):
        tl.store(start_ptr + c, w)
        n = tl.load(fill_ptr + c)
        tl.store(fill_ptr + c, w)
        w += n
    tl.store(start_ptr + NX * NY * NZ, w)
    for a in range(0, N):
        c = (tl.load(cell_ptr + 3 * a + 0) * NY + tl.load(cell_ptr + 3 * a + 1)) * NZ + tl.load(cell_ptr + 3 * a + 2)
        at = tl.load(fill_ptr + c)
        tl.store(fill_ptr + c, at + 1)
        tl.store(atoms_ptr + at, a)
        for q in tl.static_range(3):
            p = tl.load(pos_ptr + 3 * a + q)
            edge = tl.load(box_ptr + q)
            n = tl.floor(p * (1.0 / edge))
            tl.store(spos_ptr + 3 * at + q, p - n * edge)
            tl.store(swrap_ptr + 3 * at + q, n.to(tl.int32))
    tl.store(ext_ptr, tl.maximum(tl.maximum((x1 - x0) / lx, (y1 - y0) / ly), (z1 - z0) / lz))


@triton.jit
def _stage_candidate(k_ptr, code_ptr, spos_ptr, swrap_ptr, j, k, base, count, cap, ex, ey, ez, bx, by, bz, cut2):
    """Candidate ``j`` (atom ``k``) into the next free slot of the row at ``base``; the count with it where it is
    within the cutoff (``e`` the cell's image less the atom's position, ``b`` its shift index terms)."""
    dx = tl.load(spos_ptr + 3 * j + 0) + ex
    dy = tl.load(spos_ptr + 3 * j + 1) + ey
    dz = tl.load(spos_ptr + 3 * j + 2) + ez
    slot = base + tl.minimum(count, cap)
    tl.store(k_ptr + slot, k)
    tl.store(code_ptr + slot, (bx - tl.load(swrap_ptr + 3 * j + 0)) * 9 + (by - tl.load(swrap_ptr + 3 * j + 1)) * 3
             + (bz - tl.load(swrap_ptr + 3 * j + 2)))
    return count + (dx * dx + dy * dy + dz * dz <= cut2).to(tl.int32)


@triton.jit
def _grid_pairs_staged(pos_ptr, box_ptr, cell_ptr, atoms_ptr, spos_ptr, swrap_ptr, start_ptr, count_ptr, k_ptr,
                       code_ptr, NX, NY, NZ, cap, SPAN: tl.constexpr, HALF: tl.constexpr, BLOCK: tl.constexpr):
    """One search on the CPU, one atom per program and its candidates one at a time: atom i's partners in slots
    i (cap + 1) onwards (the count is kept; more than cap are not written), with the index of their image among
    the 27 of shifts -1, 0, 1. A cell farther than the cutoff from the atom is not read; the positions of the
    candidates in the order of the cells, wrapped into the cell (``spos``, ``swrap`` the edges taken off), so a
    neighboring cell's image is one shift for all its atoms. Every candidate is written to the next free slot, which
    only a partner takes (no branch per candidate; slot cap the last one written to). ``HALF``: the cells after the
    atom's own in the stencil's order and the atoms of higher index in its own (each pair once, in the row of either
    atom), else every cell and the atoms of higher index."""
    i = tl.program_id(0)
    lx = tl.load(box_ptr + 0)
    ly = tl.load(box_ptr + 1)
    lz = tl.load(box_ptr + 2)
    cut2 = tl.load(box_ptr + 3)
    wx = lx / NX
    wy = ly / NY
    wz = lz / NZ
    px = tl.load(pos_ptr + 3 * i + 0)
    py = tl.load(pos_ptr + 3 * i + 1)
    pz = tl.load(pos_ptr + 3 * i + 2)
    cx = tl.load(cell_ptr + 3 * i + 0)
    cy = tl.load(cell_ptr + 3 * i + 1)
    cz = tl.load(cell_ptr + 3 * i + 2)
    # the atom's place in its cell, and the squared cutoff with room for its rounding
    # the atom wrapped into the cell as its candidates are (whole edges nx, ny, nz off), its place in its cell
    nx = tl.floor(px * (1.0 / lx))
    ny = tl.floor(py * (1.0 / ly))
    nz = tl.floor(pz * (1.0 / lz))
    px = px - nx * lx
    py = py - ny * ly
    pz = pz - nz * lz
    ux = px - cx * wx
    uy = py - cy * wy
    uz = pz - cz * wz
    jx = nx.to(tl.int32)
    jy = ny.to(tl.int32)
    jz = nz.to(tl.int32)
    reach = cut2 * 1.000001 + 1.0e-9
    base = i.to(tl.int64) * (cap + 1)
    count = i * 0
    for d in tl.static_range(((2 * SPAN + 1) ** 3 - 1) // 2 if HALF else 0, (2 * SPAN + 1) ** 3):
        ox = d // ((2 * SPAN + 1) * (2 * SPAN + 1)) - SPAN
        oy = (d // (2 * SPAN + 1)) % (2 * SPAN + 1) - SPAN
        oz = d % (2 * SPAN + 1) - SPAN
        gx = ux * 0.0
        gy = uy * 0.0
        gz = uz * 0.0
        if ox > 0:
            gx = ox * wx - ux
        if ox < 0:
            gx = ux + (-ox - 1) * wx
        if oy > 0:
            gy = oy * wy - uy
        if oy < 0:
            gy = uy + (-oy - 1) * wy
        if oz > 0:
            gz = oz * wz - uz
        if oz < 0:
            gz = uz + (-oz - 1) * wz
        if gx * gx + gy * gy + gz * gz <= reach:
            c = (((cx + ox + NX) % NX) * NY + (cy + oy + NY) % NY) * NZ + (cz + oz + NZ) % NZ
            s = tl.load(start_ptr + c)
            e = tl.load(start_ptr + c + 1)
            # the cell's image: its atoms by a whole edge where the stencil wraps around (the same for all of them)
            sx = (cx + ox >= NX).to(tl.int32) - (cx + ox < 0).to(tl.int32)
            sy = (cy + oy >= NY).to(tl.int32) - (cy + oy < 0).to(tl.int32)
            sz = (cz + oz >= NZ).to(tl.int32) - (cz + oz < 0).to(tl.int32)
            ex = sx * lx - px
            ey = sy * ly - py
            ez = sz * lz - pz
            # a partner's shift for the positions as given: the cell's image less the edges between the two
            bx = 1 + sx + jx
            by = 1 + sy + jy
            bz = 1 + sz + jz
            for j in range(s, e):
                k = tl.load(atoms_ptr + j)
                if HALF and d * 2 + 1 != (2 * SPAN + 1) ** 3:
                    # a cell after the own one: all its atoms
                    count = _stage_candidate(k_ptr, code_ptr, spos_ptr, swrap_ptr, j, k, base, count, cap, ex, ey,
                                             ez, bx, by, bz, cut2)
                else:
                    if k > i:
                        count = _stage_candidate(k_ptr, code_ptr, spos_ptr, swrap_ptr, j, k, base, count, cap, ex,
                                                 ey, ez, bx, by, bz, cut2)
    tl.store(count_ptr + i, count)


@triton.jit
def _all_pairs_staged(pos_ptr, box_ptr, count_ptr, k_ptr, code_ptr, N, cap, BLOCK: tl.constexpr):
    """The staged search of a cell too small for a grid (each edge over twice the cutoff, the minimum image
    the one image within it): atom ``i`` against every atom of higher index."""
    i = tl.program_id(0)
    lx = tl.load(box_ptr + 0)
    ly = tl.load(box_ptr + 1)
    lz = tl.load(box_ptr + 2)
    cut2 = tl.load(box_ptr + 3)
    ilx = 1.0 / lx
    ily = 1.0 / ly
    ilz = 1.0 / lz
    px = tl.load(pos_ptr + 3 * i + 0)
    py = tl.load(pos_ptr + 3 * i + 1)
    pz = tl.load(pos_ptr + 3 * i + 2)
    base = i.to(tl.int64) * (cap + 1)
    count = i * 0
    for k in range(i + 1, N):
        dx = tl.load(pos_ptr + 3 * k + 0) - px
        dy = tl.load(pos_ptr + 3 * k + 1) - py
        dz = tl.load(pos_ptr + 3 * k + 2) - pz
        ix = tl.floor(dx * ilx + 0.5)
        iy = tl.floor(dy * ily + 0.5)
        iz = tl.floor(dz * ilz + 0.5)
        dx = dx - ix * lx
        dy = dy - iy * ly
        dz = dz - iz * lz
        slot = base + tl.minimum(count, cap)
        tl.store(k_ptr + slot, k)
        tl.store(code_ptr + slot, ((1.0 - ix) * 9.0 + (1.0 - iy) * 3.0 + (1.0 - iz)).to(tl.int32))
        count += (dx * dx + dy * dy + dz * dz <= cut2).to(tl.int32)
    tl.store(count_ptr + i, count)


def all_pairs_staged(pos: Tensor, cutoff: float, lengths: Tensor, cap: int,
                     out: Optional[tuple] = None) -> Optional[tuple]:
    """:func:`grid_pairs_staged` for an orthorhombic cell with every edge over twice ``cutoff`` and no room for
    a grid: every pair tested (a few thousand atoms)."""
    from xnn.common import cpu_kernels
    N = int(pos.shape[0])
    pos = pos.detach().contiguous()
    lengths = lengths.to(dtype=pos.dtype)
    if N and float(((pos.amax(0) - pos.amin(0)) / lengths).max()) >= 1.5:
        return None
    box = torch.cat([lengths, torch.full((1,), float(cutoff) ** 2, dtype=pos.dtype)])
    counts = torch.zeros(N, dtype=torch.int32)
    k, code = _slots(N * (cap + 1), out)
    if N:
        cpu_kernels.launch(_all_pairs_staged, (N,), pos, box, counts, k, code, N, cap, BLOCK=1)
    if N and int(counts.max()) > cap:
        return None
    return counts, k, code


def _slots(n: int, out: Optional[tuple]) -> tuple:
    """The staged partners and image indices: the first ``n`` of the buffers ``out`` (int32, int8) or new."""
    if out is not None:
        return out[0][:n], out[1][:n]
    return torch.empty(n, dtype=torch.int32), torch.empty(n, dtype=torch.int8)


def grid_pairs_staged(pos: Tensor, cutoff: float, lengths: Tensor, shape: tuple, span: int,
                      cap: int, out: Optional[tuple] = None, half: bool = False) -> Optional[tuple]:
    """The pairs ``i < k`` within ``cutoff`` from one search on the CPU (:mod:`xnn.common.cpu_kernels`): each
    atom's partners in ``cap`` slots of its own, with the index of their image among the 27 shifts of -1, 0, 1
    per axis (``(s_x + 1) 9 + (s_y + 1) 3 + s_z + 1``, the displacement ``pos[k] - pos[i] + shift * lengths``).
    ``half``: each pair once in the row of one of its atoms, ``k`` lower than ``i`` too (half the candidates).

    Returns
    -------
    tuple or None
        The counts ``(N,)`` (int32), the partners ``(N (cap + 1),)`` (int32) and their image indices (int8), atom
        ``i``'s from ``i (cap + 1)`` (in the buffers ``out`` where given, at least that long), or ``None`` when
        an atom has more than ``cap`` partners or the positions spread over 1.5 edges on an axis (a pair's shift
        may then be beyond -1, 0, 1).
    """
    from xnn.common import cpu_kernels
    N = int(pos.shape[0])
    pos = pos.detach().contiguous()
    lengths = lengths.to(dtype=pos.dtype)
    box = torch.cat([lengths, torch.full((1,), float(cutoff) ** 2, dtype=pos.dtype)])
    C = shape[0] * shape[1] * shape[2]
    cell = torch.empty(N, 3, dtype=torch.int32)
    fill = torch.zeros(C, dtype=torch.int32)
    start = torch.empty(C + 1, dtype=torch.int32)
    atoms = torch.empty(N, dtype=torch.int32)
    spos = torch.empty_like(pos)
    swrap = torch.empty(N, 3, dtype=torch.int32)
    ext = torch.zeros(1, dtype=pos.dtype)
    counts = torch.zeros(N, dtype=torch.int32)
    k, code = _slots(N * (cap + 1), out)
    if N:
        cpu_kernels.launch(_bin_cells, (1,), pos, box, cell, fill, start, atoms, spos, swrap, ext, N, *shape, BLOCK=1)
        if float(ext) >= 1.5:
            return None
        cpu_kernels.launch(_grid_pairs_staged, (N,), pos, box, cell, atoms, spos, swrap, start, counts, k, code, *shape,
                           cap, SPAN=span, HALF=half, BLOCK=1)
    if N and int(counts.max()) > cap:
        return None
    return counts, k, code


def grid_shape(lengths: list, cutoff: float, span: int = 1) -> Optional[tuple]:
    """The cells per axis of a grid over an orthorhombic cell with edges ``lengths``:
    as many as fit at least ``cutoff / span`` wide, or ``None`` with fewer than
    ``2 span + 1`` on an axis."""
    shape = []
    width = cutoff / span
    for length in lengths:
        n = int(length // width)
        if n > 0 and length / n < width:
            n -= 1
        if n < 2 * span + 1:
            return None
        shape.append(n)
    return tuple(shape)


def grid_pairs(pos: Tensor, cutoff: float, lengths: Tensor, shape: tuple, capacity: Optional[int] = None,
               span: int = 1, index_dtype: torch.dtype = torch.int64, shift_dtype: torch.dtype = torch.int64) -> tuple:
    """Every pair within ``cutoff`` once, as ``(i, k, shift)`` with ``i < k``, sorted by ``i``.

    Parameters
    ----------
    pos : Tensor
        Cartesian positions ``(N, 3)`` on the GPU, wrapped into the cell or not.
    cutoff : float
        Neighbor cutoff radius.
    lengths : Tensor
        The edges of the orthorhombic cell ``(3,)``.
    shape : tuple of int or None
        The cells per axis, from :func:`grid_shape` with ``span``; ``None`` searches
        without a grid, every atom against those of higher index (a cell too small
        for one, every edge over ``cutoff``; positions on a GPU, or on the CPU under
        Triton's interpreter).
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
    span : int, optional
        The cells per cutoff on each axis (a CPU search; 1 on a GPU).
    index_dtype, shift_dtype : torch.dtype, optional
        The types of the atom indices and of the shifts (``int64`` by default).
    """
    N, device = int(pos.shape[0]), pos.device
    pos = pos.detach().contiguous()
    lengths = lengths.to(device=device, dtype=pos.dtype)
    # the edges and the squared cutoff, read by the kernel in the positions' precision
    box = torch.cat([lengths, torch.full((1,), float(cutoff) ** 2, dtype=pos.dtype, device=device)])
    counts = torch.empty(N, dtype=torch.int32, device=device)
    if shape is None:
        args, sizes, block = (pos, box), (N,), GRID_BLOCK
        run = lambda *a, **k: _small_cell_pairs[(N,)](*a, **k)  # noqa: E731
        return _two_passes(run, args, sizes, block, counts, N, capacity, index_dtype, shift_dtype)
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
    args = (pos, box, cell, atoms, start)
    if pos.is_cuda:
        run, block = (lambda *a, **k: _grid_pairs[(N,)](*a, **k)), GRID_BLOCK
    else:
        from xnn.common import cpu_kernels
        run, block = (lambda *a, **k: cpu_kernels.launch(_grid_pairs, (N,), *a, SPAN=span, **k)), 1
    return _two_passes(run, args, shape, block, counts, N, capacity, index_dtype, shift_dtype)


def _two_passes(run, args: tuple, sizes: tuple, block: int, counts: Tensor, N: int, capacity: Optional[int],
                index_dtype: torch.dtype, shift_dtype: torch.dtype) -> tuple:
    """The two passes of a search kernel ``run`` (the counts per atom, then the pairs at their offsets) and the
    outputs of :func:`grid_pairs`."""
    device = counts.device
    if N:
        run(*args, counts, counts, counts, counts, counts, *sizes, 0, WRITE=False, BLOCK=block, num_warps=GRID_WARPS)
    offset = torch.zeros(N + 1, dtype=torch.int64, device=device)
    offset[1:] = torch.cumsum(counts, 0)
    M = int(offset[-1]) if capacity is None else int(capacity)
    i = torch.zeros(M, dtype=index_dtype, device=device)
    k = torch.zeros(M, dtype=index_dtype, device=device)
    shift = torch.zeros(M, 3, dtype=shift_dtype, device=device)
    if M and N:
        run(*args, counts, offset, i, k, shift, *sizes, M, WRITE=True, BLOCK=block, num_warps=GRID_WARPS)
    if capacity is None:
        return i, k, shift
    # the padding repeats the first pair (a pair of distinct atoms: no zero distance in the arithmetic)
    total = offset[-1:]
    pad = torch.arange(M, device=device) >= total
    return (torch.where(pad, i[:1], i), torch.where(pad, k[:1], k), torch.where(pad[:, None], shift[:1], shift),
            total)
