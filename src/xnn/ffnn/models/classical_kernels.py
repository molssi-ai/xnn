"""Fused Triton kernels of the nonbonded pair terms of the classical force fields (OPLS, AMBER, CHARMM,
DREIDING).

One program handles a block of pairs of a half neighbor list: the squared distance, the van der Waals
energy from a table per pair of atom types, the Coulomb energy, the quintic switch, the exclusion weight
and the cutoff, written as the per-atom halves of both energies with atomic adds into a float64 buffer.
The backward kernel forms the gradient with respect to the pair vectors from the incoming per-atom
gradients (no atomic adds), so forces and stress follow through the pair vectors' own graph. The forms
are those of the eager functions of the models:

* ``FORM 0`` (12-6): ``A (x^2 - B x)`` with ``x = (R^2 / (r^2 + tiny))^3``: OPLS with ``A = 4 epsilon``,
  ``B = 1`` and ``R = sigma``; AMBER, CHARMM and DREIDING with ``A = epsilon``, ``B = 2`` and ``R`` the
  minimum-energy distance.
* ``FORM 1`` (exponential-6): ``a exp(-c r) - b / (r^6 + tiny)`` with ``r = sqrt(r^2 + tiny)`` (DREIDING X6).

The Coulomb term is ``k q_i q_k / r``; the switch ``1 - x^3 (10 - 15 x + 6 x^2)`` over
``[cutoff - width, cutoff]`` multiplies both terms.

Imported only on CUDA devices, so that ``triton`` is needed only there.
"""
# no postponed annotations: Triton's interpreter (TRITON_INTERPRET=1) recognizes the
# tl.constexpr parameters of a kernel only from evaluated annotations
from typing import Optional

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra import libdevice

from .amoeba_kernels import _add_sorted

BLOCK, WARPS = 128, 4


@triton.jit
def _switch(r, start, inv_width, SWITCH: tl.constexpr):
    """The quintic switch and its derivative with respect to ``r`` (1 and 0 without a switch)."""
    if SWITCH:
        x = tl.minimum(tl.maximum((r - start) * inv_width, 0.0), 1.0)
        s = 1.0 - x * x * x * (10.0 - 15.0 * x + 6.0 * x * x)
        ds = -30.0 * x * x * (1.0 - x) * (1.0 - x) * inv_width
    else:
        s = r * 0.0 + 1.0
        ds = r * 0.0
    return s, ds


@triton.jit
def _pair_energy(r2, t_ptr, tab_ptr, qi, qk, ke, tiny, m, FORM: tl.constexpr):
    """The van der Waals and Coulomb energies of a pair and their derivatives with respect to ``r^2``."""
    if FORM == 0:
        A = tl.load(tab_ptr + t_ptr * 3, mask=m, other=0.0)
        Bc = tl.load(tab_ptr + t_ptr * 3 + 1, mask=m, other=0.0)
        R2 = tl.load(tab_ptr + t_ptr * 3 + 2, mask=m, other=0.0)
        inv = 1.0 / (r2 + tiny)
        y = R2 * inv
        x = y * y * y
        e_v = A * (x * x - Bc * x)
        # dx/dr2 = -3 x / (r2 + tiny)
        de_v = A * (2.0 * x - Bc) * (-3.0 * x * inv)
    else:
        a = tl.load(tab_ptr + t_ptr * 3, mask=m, other=0.0)
        b = tl.load(tab_ptr + t_ptr * 3 + 1, mask=m, other=0.0)
        c = tl.load(tab_ptr + t_ptr * 3 + 2, mask=m, other=0.0)
        r = tl.sqrt(r2 + tiny)
        ex = a * tl.exp(-c * r)
        r6 = r2 * r2 * r2
        e_v = ex - b / (r6 + tiny)
        de_v = -c * ex * 0.5 / r + b * 3.0 * r2 * r2 / ((r6 + tiny) * (r6 + tiny))
    # one division and one square root for both energies of the 12-6 form
    inv_c = 1.0 / (r2 + tiny)
    e_c = ke * qi * qk * tl.sqrt(inv_c)
    de_c = -0.5 * e_c * inv_c
    return e_v, de_v, e_c, de_c


@triton.jit
def _segments(src_ptr, i, m, n, BLOCK: tl.constexpr):
    """The first and the last first atom of the block's pairs and the lanes of each (see
    :func:`~xnn.ffnn.models.amoeba_kernels._add_sorted`)."""
    start = tl.program_id(0) * BLOCK
    i0 = tl.load(src_ptr + start).to(tl.int64)
    i1 = tl.load(src_ptr + tl.minimum(start + BLOCK - 1, n - 1)).to(tl.int64)
    same = m & (i == i0)
    tail = m & (i == i1) & (i1 != i0)
    rest = m & (i != i0) & (i != i1)
    return i0, i1, same, tail, rest


@triton.jit
def _excluded_row(xp_ptr, xk_ptr, a, b, m, MAXE: tl.constexpr):
    """Whether ``b`` is in row ``a`` of the compressed exclusion rows ``(ptr, cols)``."""
    lo = tl.load(xp_ptr + a, mask=m, other=0)
    hi = tl.load(xp_ptr + a + 1, mask=m, other=0)
    hit = m & (lo < 0)
    for j in tl.static_range(MAXE):
        at = lo + j
        hit = hit | ((at < hi) & (tl.load(xk_ptr + at, mask=m & (at < hi), other=-1) == b))
    return hit


@triton.jit
def _pair(src_ptr, dst_ptr, sh_ptr, pos_ptr, cell_ptr, bat_ptr, xp_ptr, xk_ptr, t_ptr, q_ptr, tab_ptr, sc_ptr, e, m0,
          n_local, n_types, DT: tl.constexpr, HAS_CELL: tl.constexpr, EXCL: tl.constexpr, MAXE: tl.constexpr,
          FORM: tl.constexpr, SWITCH: tl.constexpr):
    """A pair ``i < k`` of a directed pair list, or an atom's own periodic image ``i = k``: its vector, both
    energies switched and weighted, and ``dE_v/dr2``, ``dE_c/dr2`` of the weighted, switched energies."""
    i = tl.load(src_ptr + e, mask=m0, other=0).to(tl.int64)
    k = tl.load(dst_ptr + e, mask=m0, other=0).to(tl.int64)
    m = m0 & (i <= k)
    x, y, z = _vec(pos_ptr, cell_ptr, bat_ptr, sh_ptr, i, k, e, m, HAS_CELL)
    x, y, z = x.to(DT), y.to(DT), z.to(DT)
    r2 = x * x + y * y + z * z
    # an atom's own image (a cell shorter than the cutoff) is in the directed list once per direction: half
    # weight each; a pair of an atom with itself at no distance (the padding 0, 0 of a stepper) is inactive
    own = i == k
    m = m & (~own | (r2 > 0.0))
    # float64 scalars in the working precision (a float kernel argument would be a float32 constant)
    ke = tl.load(sc_ptr).to(DT)
    tiny = tl.load(sc_ptr + 1).to(DT)
    cut2 = tl.load(sc_ptr + 2).to(DT)
    start = tl.load(sc_ptr + 3).to(DT)
    inv_width = tl.load(sc_ptr + 4).to(DT)
    # the pairs of a list with a skin beyond the cutoff are inactive (no load, no add)
    m = m & (r2 < cut2)
    w = tl.where(m, tl.where(own, 0.5, 1.0), 0.0).to(DT)
    if EXCL:
        # the excluded partners of the first atom (its row of the exclusions of one structure)
        hit = _excluded_row(xp_ptr, xk_ptr, i % n_local, (k % n_local).to(tl.int32), m, MAXE)
        w = tl.where(hit, 0.0, w).to(DT)
    tp = tl.load(t_ptr + i, mask=m, other=0) * n_types + tl.load(t_ptr + k, mask=m, other=0)
    qi = tl.load(q_ptr + i, mask=m, other=0.0)
    qk = tl.load(q_ptr + k, mask=m, other=0.0)
    e_v, de_v, e_c, de_c = _pair_energy(r2, tp, tab_ptr, qi, qk, ke, tiny, m, FORM)
    r = tl.sqrt(r2 + tiny)
    s, ds = _switch(r, start, inv_width, SWITCH)
    # d(s e)/dr2 = s de/dr2 + e ds/dr dr/dr2, dr/dr2 = 1 / (2 r) (without a switch ds is 0)
    if SWITCH:
        dsr = ds * 0.5 / r
    else:
        dsr = ds
    return i, k, m, x, y, z, w * s * e_v, w * (s * de_v + e_v * dsr), w * s * e_c, w * (s * de_c + e_c * dsr)


@triton.jit
def _pairs_forward(src_ptr, dst_ptr, sh_ptr, pos_ptr, cell_ptr, bat_ptr, xp_ptr, xk_ptr, t_ptr, q_ptr, tab_ptr,
                   sc_ptr, e_ptr, n_edges, n_local, n_types, DT: tl.constexpr, HAS_CELL: tl.constexpr,
                   EXCL: tl.constexpr, MAXE: tl.constexpr, FORM: tl.constexpr, SWITCH: tl.constexpr,
                   BLOCK: tl.constexpr):
    """The per-atom halves of the van der Waals and Coulomb energies ``(N, 2)``, added into ``e``."""
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    i, k, m, _, _, _, ev, _, ec, _ = _pair(src_ptr, dst_ptr, sh_ptr, pos_ptr, cell_ptr, bat_ptr, xp_ptr, xk_ptr,
                                           t_ptr, q_ptr, tab_ptr, sc_ptr, e, e < n_edges, n_local, n_types, DT,
                                           HAS_CELL, EXCL, MAXE, FORM, SWITCH)
    hv = (0.5 * ev).to(tl.float64)
    hc = (0.5 * ec).to(tl.float64)
    # a list sorted by its first atom: those halves as one add per atom and block (any order stays correct)
    i0, i1, same, tail, rest = _segments(src_ptr, i, m, n_edges, BLOCK)
    _add_sorted(e_ptr, i0 * 2, i1 * 2, i * 2, hv, same, tail, rest, TAIL=1)
    _add_sorted(e_ptr + 1, i0 * 2, i1 * 2, i * 2, hc, same, tail, rest, TAIL=1)
    tl.atomic_add(e_ptr + k * 2, hv, mask=m, sem="relaxed")
    tl.atomic_add(e_ptr + k * 2 + 1, hc, mask=m, sem="relaxed")


@triton.jit
def _pairs_backward(src_ptr, dst_ptr, sh_ptr, pos_ptr, cell_ptr, bat_ptr, xp_ptr, xk_ptr, t_ptr, q_ptr, tab_ptr,
                    sc_ptr, g_ptr, gpos_ptr, gcell_ptr, n_edges, n_local, n_types, DT: tl.constexpr,
                    HAS_CELL: tl.constexpr, CELL_GRAD: tl.constexpr, EXCL: tl.constexpr, MAXE: tl.constexpr,
                    FORM: tl.constexpr, SWITCH: tl.constexpr, BLOCK: tl.constexpr):
    """The position and cell gradients from the incoming gradients of the per-atom energies ``g`` ``(N, 2)``."""
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    i, k, m, x, y, z, _, dv, _, dc = _pair(src_ptr, dst_ptr, sh_ptr, pos_ptr, cell_ptr, bat_ptr, xp_ptr, xk_ptr,
                                           t_ptr, q_ptr, tab_ptr, sc_ptr, e, e < n_edges, n_local, n_types, DT,
                                           HAS_CELL, EXCL, MAXE, FORM, SWITCH)
    gv = 0.5 * (tl.load(g_ptr + i * 2, mask=m, other=0.0) + tl.load(g_ptr + k * 2, mask=m, other=0.0))
    gc = 0.5 * (tl.load(g_ptr + i * 2 + 1, mask=m, other=0.0) + tl.load(g_ptr + k * 2 + 1, mask=m, other=0.0))
    d = 2.0 * (gv.to(DT) * dv + gc.to(DT) * dc)
    i0, i1, same, tail, rest = _segments(src_ptr, i, m, n_edges, BLOCK)
    _vec_grad_sorted(gpos_ptr, gcell_ptr, bat_ptr, sh_ptr, i, k, e, m, d * x, d * y, d * z, i0, i1, same, tail, rest,
                     HAS_CELL, CELL_GRAD)


@triton.jit
def _pairs_energy_forces(src_ptr, dst_ptr, sh_ptr, pos_ptr, cell_ptr, bat_ptr, xp_ptr, xk_ptr, t_ptr, q_ptr, tab_ptr,
                         sc_ptr, e_ptr, gpos_ptr, gcell_ptr, n_edges, n_local, n_types, DT: tl.constexpr,
                         HAS_CELL: tl.constexpr, CELL_GRAD: tl.constexpr, EXCL: tl.constexpr, MAXE: tl.constexpr,
                         FORM: tl.constexpr, SWITCH: tl.constexpr, BLOCK: tl.constexpr):
    """A step's pairs in one pass: the total van der Waals and Coulomb energies added into ``e`` ``(2,)`` and
    their position (and cell) gradients into ``gpos`` (``gcell``)."""
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    i, k, m, x, y, z, ev, dv, ec, dc = _pair(src_ptr, dst_ptr, sh_ptr, pos_ptr, cell_ptr, bat_ptr, xp_ptr, xk_ptr,
                                             t_ptr, q_ptr, tab_ptr, sc_ptr, e, e < n_edges, n_local, n_types, DT,
                                             HAS_CELL, EXCL, MAXE, FORM, SWITCH)
    tl.atomic_add(e_ptr, tl.sum(tl.where(m, ev.to(tl.float64), 0.0), 0), sem="relaxed")
    tl.atomic_add(e_ptr + 1, tl.sum(tl.where(m, ec.to(tl.float64), 0.0), 0), sem="relaxed")
    d = 2.0 * (dv + dc)
    i0, i1, same, tail, rest = _segments(src_ptr, i, m, n_edges, BLOCK)
    _vec_grad_sorted(gpos_ptr, gcell_ptr, bat_ptr, sh_ptr, i, k, e, m, d * x, d * y, d * z, i0, i1, same, tail, rest,
                     HAS_CELL, CELL_GRAD)


@triton.jit
def _pairs_rows_energy_forces(beg_ptr, end_ptr, nbr_ptr, sh_ptr, pos_ptr, cell_ptr, t_ptr, q_ptr, tab_ptr, sc_ptr,
                              e_ptr, gpos_ptr, n_types, DT: tl.constexpr, HAS_CELL: tl.constexpr,
                              FORM: tl.constexpr, SWITCH: tl.constexpr, BLOCK: tl.constexpr,
                              ENERGY: tl.constexpr = True):
    """:func:`_pairs_energy_forces` by rows, for a CPU: one program per first atom ``i``, its partners ``k > i``
    (``nbr[beg[i]:end[i]]``, no excluded pair, one structure, ``HAS_CELL`` its one cell) in chunks of
    ``BLOCK`` lanes, ``i``'s position, charge and type read once and its force summed over the row. ``HAS_CELL``
    2: ``sh`` one byte per pair, the index of its image among 27 Cartesian offsets ``cell (27, 3)`` (shifts of
    -1, 0 or 1 per axis); otherwise ``sh`` the integer shifts ``(E, 3)`` and ``cell`` the cell. ``ENERGY`` off:
    the forces alone."""
    i = tl.program_id(0).to(tl.int64)
    lo = tl.load(beg_ptr + i).to(tl.int64)
    hi = tl.load(end_ptr + i).to(tl.int64)
    xi = tl.load(pos_ptr + i * 3)
    yi = tl.load(pos_ptr + i * 3 + 1)
    zi = tl.load(pos_ptr + i * 3 + 2)
    qi = tl.load(q_ptr + i)
    ti = tl.load(t_ptr + i) * n_types
    ke = tl.load(sc_ptr).to(DT)
    tiny = tl.load(sc_ptr + 1).to(DT)
    cut2 = tl.load(sc_ptr + 2).to(DT)
    start = tl.load(sc_ptr + 3).to(DT)
    inv_width = tl.load(sc_ptr + 4).to(DT)
    # no value carried between the chunks: on a CPU a partner beyond the cutoff skips the rest of its chunk
    for e0 in range(lo, hi, BLOCK):
        e = e0 + tl.arange(0, BLOCK)
        m = e < hi
        k = tl.load(nbr_ptr + e, mask=m, other=0).to(tl.int64)
        x = tl.load(pos_ptr + k * 3, mask=m, other=0.0) - xi
        y = tl.load(pos_ptr + k * 3 + 1, mask=m, other=0.0) - yi
        z = tl.load(pos_ptr + k * 3 + 2, mask=m, other=0.0) - zi
        if HAS_CELL == 2:
            c = tl.load(sh_ptr + e, mask=m, other=13).to(tl.int64) * 3
            x += tl.load(cell_ptr + c, mask=m, other=0.0)
            y += tl.load(cell_ptr + c + 1, mask=m, other=0.0)
            z += tl.load(cell_ptr + c + 2, mask=m, other=0.0)
        elif HAS_CELL:
            s0 = tl.load(sh_ptr + e * 3, mask=m, other=0)
            s1 = tl.load(sh_ptr + e * 3 + 1, mask=m, other=0)
            s2 = tl.load(sh_ptr + e * 3 + 2, mask=m, other=0)
            x += s0 * tl.load(cell_ptr) + s1 * tl.load(cell_ptr + 3) + s2 * tl.load(cell_ptr + 6)
            y += s0 * tl.load(cell_ptr + 1) + s1 * tl.load(cell_ptr + 4) + s2 * tl.load(cell_ptr + 7)
            z += s0 * tl.load(cell_ptr + 2) + s1 * tl.load(cell_ptr + 5) + s2 * tl.load(cell_ptr + 8)
        x, y, z = x.to(DT), y.to(DT), z.to(DT)
        r2 = x * x + y * y + z * z
        m = m & (r2 < cut2)
        tp = ti + tl.load(t_ptr + k, mask=m, other=0)
        qk = tl.load(q_ptr + k, mask=m, other=0.0)
        e_v, de_v, e_c, de_c = _pair_energy(r2, tp, tab_ptr, qi, qk, ke, tiny, m, FORM)
        if SWITCH:
            r = tl.sqrt(r2 + tiny)
            s, ds = _switch(r, start, inv_width, SWITCH)
            dsr = ds * 0.5 / r
            d = 2.0 * ((s * de_v + e_v * dsr) + (s * de_c + e_c * dsr))
            e_v = s * e_v
            e_c = s * e_c
        else:
            # no switch: neither a distance nor a factor of one (the same numbers)
            d = 2.0 * (de_v + de_c)
        if ENERGY:
            tl.atomic_add(e_ptr, tl.sum(tl.where(m, e_v.to(tl.float64), 0.0), 0), sem="relaxed")
            tl.atomic_add(e_ptr + 1, tl.sum(tl.where(m, e_c.to(tl.float64), 0.0), 0), sem="relaxed")
        gx = (d * x).to(tl.float64)
        gy = (d * y).to(tl.float64)
        gz = (d * z).to(tl.float64)
        tl.atomic_add(gpos_ptr + k * 3, gx, mask=m, sem="relaxed")
        tl.atomic_add(gpos_ptr + k * 3 + 1, gy, mask=m, sem="relaxed")
        tl.atomic_add(gpos_ptr + k * 3 + 2, gz, mask=m, sem="relaxed")
        tl.atomic_add(gpos_ptr + i * 3, tl.sum(tl.where(m, -gx, 0.0), 0), sem="relaxed")
        tl.atomic_add(gpos_ptr + i * 3 + 1, tl.sum(tl.where(m, -gy, 0.0), 0), sem="relaxed")
        tl.atomic_add(gpos_ptr + i * 3 + 2, tl.sum(tl.where(m, -gz, 0.0), 0), sem="relaxed")


@triton.jit
def _bonded_energy_forces(pos_ptr, cell_ptr, bat_ptr, s_ptr, idx_ptr, p0_ptr, p1_ptr, p2_ptr, p3_ptr, p4_ptr, p5_ptr,
                          sc_ptr, e_ptr, gpos_ptr, gcell_ptr, n_terms, KIND: tl.constexpr, DT: tl.constexpr,
                          HAS_CELL: tl.constexpr, CELL_GRAD: tl.constexpr, FORM: tl.constexpr, S: tl.constexpr,
                          NMAX: tl.constexpr, BLOCK: tl.constexpr, ENERGY: tl.constexpr = True):
    """A step's terms of one class in one pass: their total energy added into ``e`` ``(1,)`` and its position (and
    cell) gradient into ``gpos`` (``gcell``). ``KIND`` 0 to 5: bonds, Morse bonds, angles, torsions, inversions,
    hydrogen bonds, with the parameter arrays of :data:`_FORWARD`'s kernel in order (the rest unused); ``ENERGY``
    off: the gradient alone."""
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n_terms
    tiny = tl.load(sc_ptr).to(DT)
    if KIND == 0:
        a, b, en, gx, gy, gz = _bond(pos_ptr, cell_ptr, bat_ptr, s_ptr, idx_ptr, p0_ptr, p1_ptr, e, m, n_terms, tiny,
                                     DT, HAS_CELL)
        _vec_grad(gpos_ptr, gcell_ptr, bat_ptr, s_ptr, a, b, e, m, gx, gy, gz, HAS_CELL, CELL_GRAD)
    elif KIND == 1:
        a, b, en, gx, gy, gz = _morse(pos_ptr, cell_ptr, bat_ptr, s_ptr, idx_ptr, p0_ptr, p1_ptr, p2_ptr, e, m,
                                      n_terms, tiny, DT, HAS_CELL)
        _vec_grad(gpos_ptr, gcell_ptr, bat_ptr, s_ptr, a, b, e, m, gx, gy, gz, HAS_CELL, CELL_GRAD)
    elif KIND == 2:
        bound = tl.load(sc_ptr + 1).to(DT)
        a, c, b, en, gux, guy, guz, gvx, gvy, gvz = _angle(pos_ptr, cell_ptr, bat_ptr, s_ptr, idx_ptr, p0_ptr, p1_ptr,
                                                           p2_ptr, e, m, n_terms, tiny, bound, DT, HAS_CELL)
        _vec_grad(gpos_ptr, gcell_ptr, bat_ptr, s_ptr, c, a, e, m, gux, guy, guz, HAS_CELL, CELL_GRAD)
        _vec_grad(gpos_ptr, gcell_ptr, bat_ptr, s_ptr + n_terms * 3, c, b, e, m, gvx, gvy, gvz, HAS_CELL, CELL_GRAD)
    elif KIND == 3:
        i0, i1, i2, i3, en, ax, ay, az, bx, by, bz, cx, cy, cz = _torsion(
            pos_ptr, cell_ptr, bat_ptr, s_ptr, idx_ptr, p1_ptr, p2_ptr, p3_ptr, p4_ptr, p5_ptr, e, m, n_terms, tiny, DT,
            HAS_CELL, FORM, S, NMAX)
        _vec_grad(gpos_ptr, gcell_ptr, bat_ptr, s_ptr, i0, i1, e, m, ax, ay, az, HAS_CELL, CELL_GRAD)
        _vec_grad(gpos_ptr, gcell_ptr, bat_ptr, s_ptr + n_terms * 3, i1, i2, e, m, bx, by, bz, HAS_CELL, CELL_GRAD)
        _vec_grad(gpos_ptr, gcell_ptr, bat_ptr, s_ptr + n_terms * 6, i2, i3, e, m, cx, cy, cz, HAS_CELL, CELL_GRAD)
    elif KIND == 4:
        c, a, b, d, en, gix, giy, giz, gjx, gjy, gjz, glx, gly, glz = _inversion(
            pos_ptr, cell_ptr, bat_ptr, s_ptr, idx_ptr, p0_ptr, p1_ptr, p2_ptr, e, m, n_terms, tiny, DT, HAS_CELL)
        _vec_grad(gpos_ptr, gcell_ptr, bat_ptr, s_ptr, c, a, e, m, gix, giy, giz, HAS_CELL, CELL_GRAD)
        _vec_grad(gpos_ptr, gcell_ptr, bat_ptr, s_ptr + n_terms * 3, c, b, e, m, gjx, gjy, gjz, HAS_CELL, CELL_GRAD)
        _vec_grad(gpos_ptr, gcell_ptr, bat_ptr, s_ptr + n_terms * 6, c, d, e, m, glx, gly, glz, HAS_CELL, CELL_GRAD)
    else:
        cut2 = tl.load(sc_ptr + 2).to(DT)
        cos_max = tl.load(sc_ptr + 3).to(DT)
        dd, hh, aa, en, gax, gay, gaz, ghx, ghy, ghz = _hbond(pos_ptr, cell_ptr, bat_ptr, s_ptr, idx_ptr, p0_ptr,
                                                              p1_ptr, e, m, n_terms, tiny, cut2, cos_max, DT, HAS_CELL)
        _vec_grad(gpos_ptr, gcell_ptr, bat_ptr, s_ptr, dd, aa, e, m, gax, gay, gaz, HAS_CELL, CELL_GRAD)
        _vec_grad(gpos_ptr, gcell_ptr, bat_ptr, s_ptr + n_terms * 3, hh, dd, e, m, ghx, ghy, ghz, HAS_CELL, CELL_GRAD)
    if ENERGY:
        tl.atomic_add(e_ptr, tl.sum(tl.where(m, en.to(tl.float64), 0.0), 0), sem="relaxed")


_KINDS = {"bond": 0, "morse": 1, "angle": 2, "torsion": 3, "inversion": 4, "hbond": 5}


def _go(kernel, grid, *args, **meta):
    """``kernel[grid](*args, **meta)``, on CPU tensors as C++ loops (:mod:`.classical_cpu`)."""
    if args[0].device.type == "cpu":
        from .classical_cpu import launch
        launch(kernel, grid, *args, **meta)
    else:
        kernel[grid](*args, **meta)


_SCALARS: dict = {}


def _scalars(values: tuple, device: torch.device) -> Tensor:
    """The kernels' scalars as a float64 tensor on ``device``, made once per set of values."""
    key = (values, str(device))
    if key not in _SCALARS:
        _SCALARS[key] = torch.tensor(values, dtype=torch.float64, device=device)
    return _SCALARS[key]


# bonded terms: positions, image shifts and the cell in, per-atom energies out; the backward kernels add
# the gradients of the positions and of the cells atomically (float64)

@triton.jit
def _vec(pos_ptr, cell_ptr, bat_ptr, s_ptr, a, b, e, m, HAS_CELL: tl.constexpr):
    """``pos[b] - pos[a] + shift @ cell`` in the precision of the positions (:func:`pair_vectors`); ``HAS_CELL`` 2:
    one cell for all atoms (its entries read once, not per lane)."""
    x = tl.load(pos_ptr + b * 3, mask=m, other=0.0) - tl.load(pos_ptr + a * 3, mask=m, other=0.0)
    y = tl.load(pos_ptr + b * 3 + 1, mask=m, other=0.0) - tl.load(pos_ptr + a * 3 + 1, mask=m, other=0.0)
    z = tl.load(pos_ptr + b * 3 + 2, mask=m, other=0.0) - tl.load(pos_ptr + a * 3 + 2, mask=m, other=0.0)
    if HAS_CELL:
        s0 = tl.load(s_ptr + e * 3, mask=m, other=0.0)
        s1 = tl.load(s_ptr + e * 3 + 1, mask=m, other=0.0)
        s2 = tl.load(s_ptr + e * 3 + 2, mask=m, other=0.0)
        if HAS_CELL == 2:
            x += s0 * tl.load(cell_ptr) + s1 * tl.load(cell_ptr + 3) + s2 * tl.load(cell_ptr + 6)
            y += s0 * tl.load(cell_ptr + 1) + s1 * tl.load(cell_ptr + 4) + s2 * tl.load(cell_ptr + 7)
            z += s0 * tl.load(cell_ptr + 2) + s1 * tl.load(cell_ptr + 5) + s2 * tl.load(cell_ptr + 8)
        else:
            c = tl.load(bat_ptr + a, mask=m, other=0).to(tl.int64) * 9
            x += s0 * tl.load(cell_ptr + c, mask=m, other=0.0) + s1 * tl.load(cell_ptr + c + 3, mask=m, other=0.0) \
                + s2 * tl.load(cell_ptr + c + 6, mask=m, other=0.0)
            y += s0 * tl.load(cell_ptr + c + 1, mask=m, other=0.0) + s1 * tl.load(cell_ptr + c + 4, mask=m, other=0.0) \
                + s2 * tl.load(cell_ptr + c + 7, mask=m, other=0.0)
            z += s0 * tl.load(cell_ptr + c + 2, mask=m, other=0.0) + s1 * tl.load(cell_ptr + c + 5, mask=m, other=0.0) \
                + s2 * tl.load(cell_ptr + c + 8, mask=m, other=0.0)
    return x, y, z


@triton.jit
def _cell_grad(gcell_ptr, bat_ptr, s_ptr, a, e, m, gx, gy, gz):
    """Add ``shift (x) g`` of a vector of :func:`_vec` to its cell: the block's share of the cell of its first
    structure in one add per entry, other lanes one by one (float64 ``g``)."""
    c = tl.load(bat_ptr + a, mask=m, other=0).to(tl.int64) * 9
    c0 = tl.min(tl.where(m, c, 2147483647), 0)
    same = m & (c == c0)
    other = m & (c != c0)
    for i in tl.static_range(3):
        si = tl.load(s_ptr + e * 3 + i, mask=m, other=0.0).to(tl.float64)
        vx = si * gx
        vy = si * gy
        vz = si * gz
        tl.atomic_add(gcell_ptr + c0 + 3 * i, tl.sum(tl.where(same, vx, 0.0), 0), sem="relaxed")
        tl.atomic_add(gcell_ptr + c0 + 3 * i + 1, tl.sum(tl.where(same, vy, 0.0), 0), sem="relaxed")
        tl.atomic_add(gcell_ptr + c0 + 3 * i + 2, tl.sum(tl.where(same, vz, 0.0), 0), sem="relaxed")
        tl.atomic_add(gcell_ptr + c + 3 * i, vx, mask=other, sem="relaxed")
        tl.atomic_add(gcell_ptr + c + 3 * i + 1, vy, mask=other, sem="relaxed")
        tl.atomic_add(gcell_ptr + c + 3 * i + 2, vz, mask=other, sem="relaxed")


@triton.jit
def _vec_grad(gpos_ptr, gcell_ptr, bat_ptr, s_ptr, a, b, e, m, gx, gy, gz, HAS_CELL: tl.constexpr,
              CELL_GRAD: tl.constexpr):
    """Add the gradient ``g`` of a vector of :func:`_vec` to its two atoms and, with ``CELL_GRAD``, to its
    cell (:func:`_cell_grad`)."""
    gx = gx.to(tl.float64)
    gy = gy.to(tl.float64)
    gz = gz.to(tl.float64)
    tl.atomic_add(gpos_ptr + b * 3, gx, mask=m, sem="relaxed")
    tl.atomic_add(gpos_ptr + b * 3 + 1, gy, mask=m, sem="relaxed")
    tl.atomic_add(gpos_ptr + b * 3 + 2, gz, mask=m, sem="relaxed")
    tl.atomic_add(gpos_ptr + a * 3, -gx, mask=m, sem="relaxed")
    tl.atomic_add(gpos_ptr + a * 3 + 1, -gy, mask=m, sem="relaxed")
    tl.atomic_add(gpos_ptr + a * 3 + 2, -gz, mask=m, sem="relaxed")
    if HAS_CELL and CELL_GRAD:
        _cell_grad(gcell_ptr, bat_ptr, s_ptr, a, e, m, gx, gy, gz)


@triton.jit
def _vec_grad_sorted(gpos_ptr, gcell_ptr, bat_ptr, s_ptr, a, b, e, m, gx, gy, gz, first, last, same, tail, rest,
                     HAS_CELL: tl.constexpr, CELL_GRAD: tl.constexpr):
    """:func:`_vec_grad` for a block sorted by its first atoms ``a``: theirs as segments (:func:`_segments`)."""
    gx = gx.to(tl.float64)
    gy = gy.to(tl.float64)
    gz = gz.to(tl.float64)
    tl.atomic_add(gpos_ptr + b * 3, gx, mask=m, sem="relaxed")
    tl.atomic_add(gpos_ptr + b * 3 + 1, gy, mask=m, sem="relaxed")
    tl.atomic_add(gpos_ptr + b * 3 + 2, gz, mask=m, sem="relaxed")
    _add_sorted(gpos_ptr, first * 3, last * 3, a * 3, -gx, same, tail, rest, TAIL=1)
    _add_sorted(gpos_ptr + 1, first * 3, last * 3, a * 3, -gy, same, tail, rest, TAIL=1)
    _add_sorted(gpos_ptr + 2, first * 3, last * 3, a * 3, -gz, same, tail, rest, TAIL=1)
    if HAS_CELL and CELL_GRAD:
        _cell_grad(gcell_ptr, bat_ptr, s_ptr, a, e, m, gx, gy, gz)


@triton.jit
def _bond(pos_ptr, cell_ptr, bat_ptr, s_ptr, idx_ptr, k_ptr, r0_ptr, e, m, n_terms, tiny, DT: tl.constexpr,
          HAS_CELL: tl.constexpr):
    """``k (r - r0)^2`` of a pair of atoms, and ``dE/dvec``."""
    a = tl.load(idx_ptr + e, mask=m, other=0).to(tl.int64)
    b = tl.load(idx_ptr + n_terms + e, mask=m, other=0).to(tl.int64)
    x, y, z = _vec(pos_ptr, cell_ptr, bat_ptr, s_ptr, a, b, e, m, HAS_CELL)
    x, y, z = x.to(DT), y.to(DT), z.to(DT)
    r = tl.sqrt(x * x + y * y + z * z + tiny)
    k = tl.load(k_ptr + e, mask=m, other=0.0)
    dr = r - tl.load(r0_ptr + e, mask=m, other=0.0)
    d = 2.0 * k * dr / r
    return a, b, k * dr * dr, d * x, d * y, d * z


@triton.jit
def _bonds_forward(pos_ptr, cell_ptr, bat_ptr, s_ptr, idx_ptr, k_ptr, r0_ptr, sc_ptr, out_ptr, n_terms,
                   DT: tl.constexpr, HAS_CELL: tl.constexpr, BLOCK: tl.constexpr):
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n_terms
    tiny = tl.load(sc_ptr).to(DT)
    a, b, en, _, _, _ = _bond(pos_ptr, cell_ptr, bat_ptr, s_ptr, idx_ptr, k_ptr, r0_ptr, e, m, n_terms, tiny, DT,
                              HAS_CELL)
    h = (0.5 * en).to(tl.float64)
    tl.atomic_add(out_ptr + a, h, mask=m, sem="relaxed")
    tl.atomic_add(out_ptr + b, h, mask=m, sem="relaxed")


@triton.jit
def _bonds_backward(pos_ptr, cell_ptr, bat_ptr, s_ptr, idx_ptr, k_ptr, r0_ptr, sc_ptr, g_ptr, gpos_ptr, gcell_ptr,
                    n_terms, DT: tl.constexpr, HAS_CELL: tl.constexpr, CELL_GRAD: tl.constexpr, BLOCK: tl.constexpr):
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n_terms
    tiny = tl.load(sc_ptr).to(DT)
    a, b, _, gx, gy, gz = _bond(pos_ptr, cell_ptr, bat_ptr, s_ptr, idx_ptr, k_ptr, r0_ptr, e, m, n_terms, tiny, DT,
                                HAS_CELL)
    g = 0.5 * (tl.load(g_ptr + a, mask=m, other=0.0) + tl.load(g_ptr + b, mask=m, other=0.0)).to(DT)
    _vec_grad(gpos_ptr, gcell_ptr, bat_ptr, s_ptr, a, b, e, m, g * gx, g * gy, g * gz, HAS_CELL, CELL_GRAD)


@triton.jit
def _morse(pos_ptr, cell_ptr, bat_ptr, s_ptr, idx_ptr, d_ptr, al_ptr, r0_ptr, e, m, n_terms, tiny,
           DT: tl.constexpr, HAS_CELL: tl.constexpr):
    """``D (exp(-alpha (r - r0)) - 1)^2`` of a pair of atoms, and ``dE/dvec``."""
    a = tl.load(idx_ptr + e, mask=m, other=0).to(tl.int64)
    b = tl.load(idx_ptr + n_terms + e, mask=m, other=0).to(tl.int64)
    x, y, z = _vec(pos_ptr, cell_ptr, bat_ptr, s_ptr, a, b, e, m, HAS_CELL)
    x, y, z = x.to(DT), y.to(DT), z.to(DT)
    r = tl.sqrt(x * x + y * y + z * z + tiny)
    dd = tl.load(d_ptr + e, mask=m, other=0.0)
    al = tl.load(al_ptr + e, mask=m, other=0.0)
    ex = tl.exp(-al * (r - tl.load(r0_ptr + e, mask=m, other=0.0)))
    f = ex - 1.0
    # dE/dr = -2 D alpha exp f
    d = -2.0 * dd * al * ex * f / r
    return a, b, dd * f * f, d * x, d * y, d * z


@triton.jit
def _morses_forward(pos_ptr, cell_ptr, bat_ptr, s_ptr, idx_ptr, d_ptr, al_ptr, r0_ptr, sc_ptr, out_ptr, n_terms,
                    DT: tl.constexpr, HAS_CELL: tl.constexpr, BLOCK: tl.constexpr):
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n_terms
    tiny = tl.load(sc_ptr).to(DT)
    a, b, en, _, _, _ = _morse(pos_ptr, cell_ptr, bat_ptr, s_ptr, idx_ptr, d_ptr, al_ptr, r0_ptr, e, m, n_terms,
                               tiny, DT, HAS_CELL)
    h = (0.5 * en).to(tl.float64)
    tl.atomic_add(out_ptr + a, h, mask=m, sem="relaxed")
    tl.atomic_add(out_ptr + b, h, mask=m, sem="relaxed")


@triton.jit
def _morses_backward(pos_ptr, cell_ptr, bat_ptr, s_ptr, idx_ptr, d_ptr, al_ptr, r0_ptr, sc_ptr, g_ptr, gpos_ptr,
                     gcell_ptr, n_terms, DT: tl.constexpr, HAS_CELL: tl.constexpr, CELL_GRAD: tl.constexpr,
                     BLOCK: tl.constexpr):
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n_terms
    tiny = tl.load(sc_ptr).to(DT)
    a, b, _, gx, gy, gz = _morse(pos_ptr, cell_ptr, bat_ptr, s_ptr, idx_ptr, d_ptr, al_ptr, r0_ptr, e, m, n_terms,
                                 tiny, DT, HAS_CELL)
    g = 0.5 * (tl.load(g_ptr + a, mask=m, other=0.0) + tl.load(g_ptr + b, mask=m, other=0.0)).to(DT)
    _vec_grad(gpos_ptr, gcell_ptr, bat_ptr, s_ptr, a, b, e, m, g * gx, g * gy, g * gz, HAS_CELL, CELL_GRAD)


@triton.jit
def _angle(pos_ptr, cell_ptr, bat_ptr, s_ptr, idx_ptr, st_ptr, k_ptr, x0_ptr, e, m, n_terms, tiny, bound,
           DT: tl.constexpr, HAS_CELL: tl.constexpr):
    """An angle term (style 0: ``k (theta - theta0)^2``, 1: ``k (cos - x0)^2``, 2: ``k (1 + cos)``) about the
    center atom, and ``dE/du``, ``dE/dv`` of its two vectors from the center."""
    a = tl.load(idx_ptr + e, mask=m, other=0).to(tl.int64)
    c = tl.load(idx_ptr + n_terms + e, mask=m, other=0).to(tl.int64)
    b = tl.load(idx_ptr + 2 * n_terms + e, mask=m, other=0).to(tl.int64)
    ux, uy, uz = _vec(pos_ptr, cell_ptr, bat_ptr, s_ptr, c, a, e, m, HAS_CELL)
    vx, vy, vz = _vec(pos_ptr, cell_ptr, bat_ptr, s_ptr + n_terms * 3, c, b, e, m, HAS_CELL)
    ux, uy, uz = ux.to(DT), uy.to(DT), uz.to(DT)
    vx, vy, vz = vx.to(DT), vy.to(DT), vz.to(DT)
    uu = ux * ux + uy * uy + uz * uz
    vv = vx * vx + vy * vy + vz * vz
    p = ux * vx + uy * vy + uz * vz
    d = tl.sqrt(uu * vv + tiny)
    cs = p / d
    st = tl.load(st_ptr + e, mask=m, other=0)
    k = tl.load(k_ptr + e, mask=m, other=0.0)
    x0 = tl.load(x0_ptr + e, mask=m, other=0.0)
    cc = tl.minimum(tl.maximum(cs, -bound), bound)
    th = libdevice.acos(cc)
    inside = (cs > -bound) & (cs < bound)
    e_h = k * (th - x0) * (th - x0)
    de_h = tl.where(inside, -2.0 * k * (th - x0) / tl.sqrt(1.0 - cc * cc), 0.0)
    e_c = k * (cs - x0) * (cs - x0)
    de_c = 2.0 * k * (cs - x0)
    en = tl.where(st == 0, e_h, tl.where(st == 1, e_c, k * (1.0 + cs)))
    dc = tl.where(st == 0, de_h, tl.where(st == 1, de_c, k))
    # dcos/du = v / d - p vv u / d^3
    d3 = d * d * d
    gux = dc * (vx / d - p * vv * ux / d3)
    guy = dc * (vy / d - p * vv * uy / d3)
    guz = dc * (vz / d - p * vv * uz / d3)
    gvx = dc * (ux / d - p * uu * vx / d3)
    gvy = dc * (uy / d - p * uu * vy / d3)
    gvz = dc * (uz / d - p * uu * vz / d3)
    return a, c, b, en, gux, guy, guz, gvx, gvy, gvz


@triton.jit
def _angles_forward(pos_ptr, cell_ptr, bat_ptr, s_ptr, idx_ptr, st_ptr, k_ptr, x0_ptr, sc_ptr, out_ptr, n_terms,
                    DT: tl.constexpr, HAS_CELL: tl.constexpr, BLOCK: tl.constexpr):
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n_terms
    tiny = tl.load(sc_ptr).to(DT)
    bound = tl.load(sc_ptr + 1).to(DT)
    a, c, b, en, _, _, _, _, _, _ = _angle(pos_ptr, cell_ptr, bat_ptr, s_ptr, idx_ptr, st_ptr, k_ptr, x0_ptr, e, m,
                                           n_terms, tiny, bound, DT, HAS_CELL)
    tl.atomic_add(out_ptr + c, en.to(tl.float64), mask=m, sem="relaxed")


@triton.jit
def _angles_backward(pos_ptr, cell_ptr, bat_ptr, s_ptr, idx_ptr, st_ptr, k_ptr, x0_ptr, sc_ptr, g_ptr, gpos_ptr,
                     gcell_ptr, n_terms, DT: tl.constexpr, HAS_CELL: tl.constexpr, CELL_GRAD: tl.constexpr,
                     BLOCK: tl.constexpr):
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n_terms
    tiny = tl.load(sc_ptr).to(DT)
    bound = tl.load(sc_ptr + 1).to(DT)
    a, c, b, _, gux, guy, guz, gvx, gvy, gvz = _angle(pos_ptr, cell_ptr, bat_ptr, s_ptr, idx_ptr, st_ptr, k_ptr,
                                                      x0_ptr, e, m, n_terms, tiny, bound, DT, HAS_CELL)
    g = tl.load(g_ptr + c, mask=m, other=0.0).to(DT)
    _vec_grad(gpos_ptr, gcell_ptr, bat_ptr, s_ptr, c, a, e, m, g * gux, g * guy, g * guz, HAS_CELL, CELL_GRAD)
    _vec_grad(gpos_ptr, gcell_ptr, bat_ptr, s_ptr + n_terms * 3, c, b, e, m, g * gvx, g * gvy, g * gvz, HAS_CELL,
              CELL_GRAD)


@triton.jit
def _cross(ax, ay, az, bx, by, bz):
    return ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx


@triton.jit
def _torsion(pos_ptr, cell_ptr, bat_ptr, s_ptr, idx_ptr, k_ptr, n_ptr, cg_ptr, sg_ptr, c0_ptr, e, m, n_terms, tiny,
             DT: tl.constexpr, HAS_CELL: tl.constexpr, FORM: tl.constexpr, S: tl.constexpr,
             NMAX: tl.constexpr):
    """A torsion term over four atoms and the gradients of its three bond vectors. FORM 0: ``const + sum_s K_s
    (1 + cg_s T_n(cos phi))`` (Chebyshev polynomials of the cosine); 1: ``sum_s K_s (1 + cg_s cos(n phi) +
    sg_s sin(n phi))`` of the signed angle; 2: ``K (phi - phi0)^2`` with ``(cos, sin)`` of ``phi0`` in
    ``(cg, sg)``. FORM 3 and 4: forms 0 and 1 with the terms summed by multiplicity, ``k`` a row of
    ``1 + NMAX`` (3: ``sum K``, then ``a_n``) or ``1 + 2 NMAX`` (4: then ``b_n``) coefficients per torsion
    (:func:`~xnn.ffnn.models.classical_cpu.torsion_coefficients`), one recurrence for all its terms."""
    i0 = tl.load(idx_ptr + e, mask=m, other=0).to(tl.int64)
    i1 = tl.load(idx_ptr + n_terms + e, mask=m, other=0).to(tl.int64)
    i2 = tl.load(idx_ptr + 2 * n_terms + e, mask=m, other=0).to(tl.int64)
    i3 = tl.load(idx_ptr + 3 * n_terms + e, mask=m, other=0).to(tl.int64)
    ax, ay, az = _vec(pos_ptr, cell_ptr, bat_ptr, s_ptr, i0, i1, e, m, HAS_CELL)
    bx, by, bz = _vec(pos_ptr, cell_ptr, bat_ptr, s_ptr + n_terms * 3, i1, i2, e, m, HAS_CELL)
    cx, cy, cz = _vec(pos_ptr, cell_ptr, bat_ptr, s_ptr + n_terms * 6, i2, i3, e, m, HAS_CELL)
    ax, ay, az = ax.to(DT), ay.to(DT), az.to(DT)
    bx, by, bz = bx.to(DT), by.to(DT), bz.to(DT)
    cx, cy, cz = cx.to(DT), cy.to(DT), cz.to(DT)
    n1x, n1y, n1z = _cross(ax, ay, az, bx, by, bz)
    n2x, n2y, n2z = _cross(bx, by, bz, cx, cy, cz)
    A = n1x * n1x + n1y * n1y + n1z * n1z
    Bn = n2x * n2x + n2y * n2y + n2z * n2z
    P = n1x * n2x + n1y * n2y + n1z * n2z
    D = tl.sqrt(A * Bn + tiny)
    c = P / D
    D3 = D * D * D
    # dc/dn1, dc/dn2
    c1x = n2x / D - P * Bn * n1x / D3
    c1y = n2y / D - P * Bn * n1y / D3
    c1z = n2z / D - P * Bn * n1z / D3
    c2x = n1x / D - P * A * n2x / D3
    c2y = n1y / D - P * A * n2y / D3
    c2z = n1z / D - P * A * n2z / D3
    en = tl.load(c0_ptr + e, mask=m, other=0.0)
    ec = c * 0.0
    es = c * 0.0
    if FORM == 3:
        row = e * (NMAX + 1)
        a1 = tl.load(k_ptr + row + 1, mask=m, other=0.0)
        en += tl.load(k_ptr + row, mask=m, other=0.0) + a1 * c
        ec += a1
        tp = c * 0.0 + 1.0
        tc = c
        dp = c * 0.0
        dcur = c * 0.0 + 1.0
        for q in tl.static_range(2, NMAX + 1):
            tnew = 2.0 * c * tc - tp
            dnew = 2.0 * tc + 2.0 * c * dcur - dp
            tp = tc
            tc = tnew
            dp = dcur
            dcur = dnew
            aq = tl.load(k_ptr + row + q, mask=m, other=0.0)
            en += aq * tc
            ec += aq * dcur
        s = c * 0.0
        Lb = c * 0.0 + 1.0
        Q = c * 0.0
    elif FORM == 0:
        for j in tl.static_range(S):
            K = tl.load(k_ptr + e * S + j, mask=m, other=0.0)
            nn = tl.load(n_ptr + e * S + j, mask=m, other=1)
            cg = tl.load(cg_ptr + e * S + j, mask=m, other=0.0)
            # T_n and T_n' by the recurrences T_{n+1} = 2 c T_n - T_{n-1}, T'_{n+1} = 2 T_n + 2 c T'_n - T'_{n-1}
            tp = c * 0.0 + 1.0
            tc = c
            dp = c * 0.0
            dcur = c * 0.0 + 1.0
            tn = tc
            dn = dcur
            for q in tl.static_range(2, NMAX + 1):
                tnew = 2.0 * c * tc - tp
                dnew = 2.0 * tc + 2.0 * c * dcur - dp
                tp = tc
                tc = tnew
                dp = dcur
                dcur = dnew
                tn = tl.where(nn == q, tc, tn)
                dn = tl.where(nn == q, dcur, dn)
            en += K * (1.0 + cg * tn)
            ec += K * cg * dn
        s = c * 0.0
        Lb = c * 0.0 + 1.0
        Q = c * 0.0
    else:
        Lb = tl.sqrt(bx * bx + by * by + bz * bz + tiny)
        qx, qy, qz = _cross(n1x, n1y, n1z, n2x, n2y, n2z)
        Q = qx * bx + qy * by + qz * bz
        s = -Q / (D * Lb)
        if FORM == 4:
            row = e * (2 * NMAX + 1)
            a1 = tl.load(k_ptr + row + 1, mask=m, other=0.0)
            b1 = tl.load(k_ptr + row + 1 + NMAX, mask=m, other=0.0)
            en += tl.load(k_ptr + row, mask=m, other=0.0) + a1 * c + b1 * s
            ec += a1
            es += b1
            C = c
            Sn = s
            Cc = c * 0.0 + 1.0
            Cs = c * 0.0
            Sc = c * 0.0
            Ss = c * 0.0 + 1.0
            for q in tl.static_range(2, NMAX + 1):
                Cn = C * c - Sn * s
                Snn = Sn * c + C * s
                Ccn = Cc * c + C - Sc * s
                Csn = Cs * c - Ss * s - Sn
                Scn = Sc * c + Sn + Cc * s
                Ssn = Ss * c + Cs * s + C
                C = Cn
                Sn = Snn
                Cc = Ccn
                Cs = Csn
                Sc = Scn
                Ss = Ssn
                aq = tl.load(k_ptr + row + q, mask=m, other=0.0)
                bq = tl.load(k_ptr + row + q + NMAX, mask=m, other=0.0)
                en += aq * C + bq * Sn
                ec += aq * Cc + bq * Sc
                es += aq * Cs + bq * Ss
        elif FORM == 1:
            for j in tl.static_range(S):
                K = tl.load(k_ptr + e * S + j, mask=m, other=0.0)
                nn = tl.load(n_ptr + e * S + j, mask=m, other=1)
                cg = tl.load(cg_ptr + e * S + j, mask=m, other=0.0)
                sg = tl.load(sg_ptr + e * S + j, mask=m, other=0.0)
                # cos(n phi), sin(n phi) and their derivatives in (c, s) by the angle-addition recurrences
                C = c
                Sn = s
                Cc = c * 0.0 + 1.0
                Cs = c * 0.0
                Sc = c * 0.0
                Ss = c * 0.0 + 1.0
                cn = C
                sn = Sn
                cnc = Cc
                cns = Cs
                snc = Sc
                sns = Ss
                for q in tl.static_range(2, NMAX + 1):
                    Cn = C * c - Sn * s
                    Snn = Sn * c + C * s
                    Ccn = Cc * c + C - Sc * s
                    Csn = Cs * c - Ss * s - Sn
                    Scn = Sc * c + Sn + Cc * s
                    Ssn = Ss * c + Cs * s + C
                    C = Cn
                    Sn = Snn
                    Cc = Ccn
                    Cs = Csn
                    Sc = Scn
                    Ss = Ssn
                    sel = nn == q
                    cn = tl.where(sel, C, cn)
                    sn = tl.where(sel, Sn, sn)
                    cnc = tl.where(sel, Cc, cnc)
                    cns = tl.where(sel, Cs, cns)
                    snc = tl.where(sel, Sc, snc)
                    sns = tl.where(sel, Ss, sns)
                en += K * (1.0 + cg * cn + sg * sn)
                ec += K * (cg * cnc + sg * snc)
                es += K * (cg * cns + sg * sns)
        else:
            K = tl.load(k_ptr + e, mask=m, other=0.0)
            c0 = tl.load(cg_ptr + e, mask=m, other=1.0)
            s0 = tl.load(sg_ptr + e, mask=m, other=0.0)
            yy = s * c0 - c * s0
            xx = c * c0 + s * s0
            dphi = libdevice.atan2(yy, xx)
            r2 = xx * xx + yy * yy
            en += K * dphi * dphi
            dd = 2.0 * K * dphi / r2
            ec += dd * (xx * (-s0) - yy * c0)
            es += dd * (xx * c0 - yy * s0)
    # the gradients of the normals: dE/dn = Ec dc/dn + Es ds/dn
    g1x = ec * c1x
    g1y = ec * c1y
    g1z = ec * c1z
    g2x = ec * c2x
    g2y = ec * c2y
    g2z = ec * c2z
    gbx = c * 0.0
    gby = c * 0.0
    gbz = c * 0.0
    if FORM != 0:
        DL = D * Lb
        # Q = (n1 x n2) . b2: dQ/dn1 = n2 x b2, dQ/dn2 = b2 x n1, dQ/db2 = n1 x n2; s = -Q / (D L)
        p1x, p1y, p1z = _cross(n2x, n2y, n2z, bx, by, bz)
        p2x, p2y, p2z = _cross(bx, by, bz, n1x, n1y, n1z)
        qx, qy, qz = _cross(n1x, n1y, n1z, n2x, n2y, n2z)
        f = Q / (D3 * Lb)
        g1x += es * (-p1x / DL + f * Bn * n1x)
        g1y += es * (-p1y / DL + f * Bn * n1y)
        g1z += es * (-p1z / DL + f * Bn * n1z)
        g2x += es * (-p2x / DL + f * A * n2x)
        g2y += es * (-p2y / DL + f * A * n2y)
        g2z += es * (-p2z / DL + f * A * n2z)
        h = Q / (D * Lb * Lb * Lb)
        gbx += es * (-qx / DL + h * bx)
        gby += es * (-qy / DL + h * by)
        gbz += es * (-qz / DL + h * bz)
    # n1 = b1 x b2, n2 = b2 x b3: for n = u x w, d(g.n)/du = w x g and d(g.n)/dw = g x u
    ga_x, ga_y, ga_z = _cross(bx, by, bz, g1x, g1y, g1z)
    t1x, t1y, t1z = _cross(g1x, g1y, g1z, ax, ay, az)
    t2x, t2y, t2z = _cross(cx, cy, cz, g2x, g2y, g2z)
    gc_x, gc_y, gc_z = _cross(g2x, g2y, g2z, bx, by, bz)
    gbx += t1x + t2x
    gby += t1y + t2y
    gbz += t1z + t2z
    return i0, i1, i2, i3, en, ga_x, ga_y, ga_z, gbx, gby, gbz, gc_x, gc_y, gc_z


@triton.jit
def _torsions_forward(pos_ptr, cell_ptr, bat_ptr, s_ptr, idx_ptr, own_ptr, k_ptr, n_ptr, cg_ptr, sg_ptr, c0_ptr,
                      sc_ptr, out_ptr, n_terms, DT: tl.constexpr, HAS_CELL: tl.constexpr, FORM: tl.constexpr,
                      S: tl.constexpr, NMAX: tl.constexpr, BLOCK: tl.constexpr):
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n_terms
    tiny = tl.load(sc_ptr).to(DT)
    _, _, _, _, en, _, _, _, _, _, _, _, _, _ = _torsion(pos_ptr, cell_ptr, bat_ptr, s_ptr, idx_ptr, k_ptr, n_ptr,
                                                         cg_ptr, sg_ptr, c0_ptr, e, m, n_terms, tiny, DT, HAS_CELL,
                                                         FORM, S, NMAX)
    own = tl.load(own_ptr + e, mask=m, other=0).to(tl.int64)
    tl.atomic_add(out_ptr + own, en.to(tl.float64), mask=m, sem="relaxed")


@triton.jit
def _torsions_backward(pos_ptr, cell_ptr, bat_ptr, s_ptr, idx_ptr, own_ptr, k_ptr, n_ptr, cg_ptr, sg_ptr, c0_ptr,
                       sc_ptr, g_ptr, gpos_ptr, gcell_ptr, n_terms, DT: tl.constexpr, HAS_CELL: tl.constexpr,
                       CELL_GRAD: tl.constexpr, FORM: tl.constexpr, S: tl.constexpr, NMAX: tl.constexpr,
                       BLOCK: tl.constexpr):
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n_terms
    tiny = tl.load(sc_ptr).to(DT)
    i0, i1, i2, i3, _, ax, ay, az, bx, by, bz, cx, cy, cz = _torsion(
        pos_ptr, cell_ptr, bat_ptr, s_ptr, idx_ptr, k_ptr, n_ptr, cg_ptr, sg_ptr, c0_ptr, e, m, n_terms, tiny, DT,
        HAS_CELL, FORM, S, NMAX)
    own = tl.load(own_ptr + e, mask=m, other=0).to(tl.int64)
    g = tl.load(g_ptr + own, mask=m, other=0.0).to(DT)
    _vec_grad(gpos_ptr, gcell_ptr, bat_ptr, s_ptr, i0, i1, e, m, g * ax, g * ay, g * az, HAS_CELL, CELL_GRAD)
    _vec_grad(gpos_ptr, gcell_ptr, bat_ptr, s_ptr + n_terms * 3, i1, i2, e, m, g * bx, g * by, g * bz, HAS_CELL,
              CELL_GRAD)
    _vec_grad(gpos_ptr, gcell_ptr, bat_ptr, s_ptr + n_terms * 6, i2, i3, e, m, g * cx, g * cy, g * cz, HAS_CELL,
              CELL_GRAD)


@triton.jit
def _inversion(pos_ptr, cell_ptr, bat_ptr, s_ptr, idx_ptr, k_ptr, pl_ptr, c0_ptr, e, m, n_terms, tiny,
               DT: tl.constexpr, HAS_CELL: tl.constexpr):
    """A DREIDING inversion about the center atom ``c`` (``c, a, b, d``): ``sin psi = n . u``, ``n`` the unit
    normal of ``(r_ca, r_cb)`` and ``u`` the unit ``r_cd``, ``cos psi = sqrt(max(1 - sin^2, tiny))``; the
    energy ``K (1 - cos psi)`` (planar) or ``K (cos psi - cos psi0)^2`` (the prefactors already in ``K``), and
    the gradients of the three vectors from the center."""
    c = tl.load(idx_ptr + e, mask=m, other=0).to(tl.int64)
    a = tl.load(idx_ptr + n_terms + e, mask=m, other=0).to(tl.int64)
    b = tl.load(idx_ptr + 2 * n_terms + e, mask=m, other=0).to(tl.int64)
    d = tl.load(idx_ptr + 3 * n_terms + e, mask=m, other=0).to(tl.int64)
    ix, iy, iz = _vec(pos_ptr, cell_ptr, bat_ptr, s_ptr, c, a, e, m, HAS_CELL)
    jx, jy, jz = _vec(pos_ptr, cell_ptr, bat_ptr, s_ptr + n_terms * 3, c, b, e, m, HAS_CELL)
    lx, ly, lz = _vec(pos_ptr, cell_ptr, bat_ptr, s_ptr + n_terms * 6, c, d, e, m, HAS_CELL)
    ix, iy, iz = ix.to(DT), iy.to(DT), iz.to(DT)
    jx, jy, jz = jx.to(DT), jy.to(DT), jz.to(DT)
    lx, ly, lz = lx.to(DT), ly.to(DT), lz.to(DT)
    Nx, Ny, Nz = _cross(ix, iy, iz, jx, jy, jz)
    Nn = tl.sqrt(Nx * Nx + Ny * Ny + Nz * Nz + tiny)
    Ln = tl.sqrt(lx * lx + ly * ly + lz * lz + tiny)
    nx, ny, nz = Nx / Nn, Ny / Nn, Nz / Nn
    ux, uy, uz = lx / Ln, ly / Ln, lz / Ln
    sn = nx * ux + ny * uy + nz * uz
    one = 1.0 - sn * sn
    cs = tl.sqrt(tl.maximum(one, tiny))
    K = tl.load(k_ptr + e, mask=m, other=0.0)
    planar = tl.load(pl_ptr + e, mask=m, other=0) != 0
    c0 = tl.load(c0_ptr + e, mask=m, other=0.0)
    en = tl.where(planar, K * (1.0 - cs), K * (cs - c0) * (cs - c0))
    dcs = tl.where(planar, -K, 2.0 * K * (cs - c0))
    # dcos/dsin = -sin / cos where the clamp is inactive
    ds = tl.where(one > tiny, -dcs * sn / cs, 0.0)
    # ds/dr_cd = (n - s u) / |r_cd|; ds/dN = (u - s n) / |N|, N = r_ca x r_cb
    glx = ds * (nx - sn * ux) / Ln
    gly = ds * (ny - sn * uy) / Ln
    glz = ds * (nz - sn * uz) / Ln
    gNx = ds * (ux - sn * nx) / Nn
    gNy = ds * (uy - sn * ny) / Nn
    gNz = ds * (uz - sn * nz) / Nn
    gix, giy, giz = _cross(jx, jy, jz, gNx, gNy, gNz)
    gjx, gjy, gjz = _cross(gNx, gNy, gNz, ix, iy, iz)
    return c, a, b, d, en, gix, giy, giz, gjx, gjy, gjz, glx, gly, glz


@triton.jit
def _inversions_forward(pos_ptr, cell_ptr, bat_ptr, s_ptr, idx_ptr, k_ptr, pl_ptr, c0_ptr, sc_ptr, out_ptr, n_terms,
                        DT: tl.constexpr, HAS_CELL: tl.constexpr, BLOCK: tl.constexpr):
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n_terms
    tiny = tl.load(sc_ptr).to(DT)
    c, _, _, _, en, _, _, _, _, _, _, _, _, _ = _inversion(pos_ptr, cell_ptr, bat_ptr, s_ptr, idx_ptr, k_ptr, pl_ptr,
                                                           c0_ptr, e, m, n_terms, tiny, DT, HAS_CELL)
    tl.atomic_add(out_ptr + c, en.to(tl.float64), mask=m, sem="relaxed")


@triton.jit
def _inversions_backward(pos_ptr, cell_ptr, bat_ptr, s_ptr, idx_ptr, k_ptr, pl_ptr, c0_ptr, sc_ptr, g_ptr, gpos_ptr,
                         gcell_ptr, n_terms, DT: tl.constexpr, HAS_CELL: tl.constexpr, CELL_GRAD: tl.constexpr,
                         BLOCK: tl.constexpr):
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n_terms
    tiny = tl.load(sc_ptr).to(DT)
    c, a, b, d, _, gix, giy, giz, gjx, gjy, gjz, glx, gly, glz = _inversion(
        pos_ptr, cell_ptr, bat_ptr, s_ptr, idx_ptr, k_ptr, pl_ptr, c0_ptr, e, m, n_terms, tiny, DT, HAS_CELL)
    g = tl.load(g_ptr + c, mask=m, other=0.0).to(DT)
    _vec_grad(gpos_ptr, gcell_ptr, bat_ptr, s_ptr, c, a, e, m, g * gix, g * giy, g * giz, HAS_CELL, CELL_GRAD)
    _vec_grad(gpos_ptr, gcell_ptr, bat_ptr, s_ptr + n_terms * 3, c, b, e, m, g * gjx, g * gjy, g * gjz, HAS_CELL,
              CELL_GRAD)
    _vec_grad(gpos_ptr, gcell_ptr, bat_ptr, s_ptr + n_terms * 6, c, d, e, m, g * glx, g * gly, g * glz, HAS_CELL,
              CELL_GRAD)


@triton.jit
def _hbond(pos_ptr, cell_ptr, bat_ptr, s_ptr, idx_ptr, d0_ptr, r0_ptr, e, m, n_terms, tiny, cut2, cos_max,
           DT: tl.constexpr, HAS_CELL: tl.constexpr):
    """A DREIDING hydrogen bond over ``(donor, H, acceptor)``: ``D0 (5 rho^12 - 6 rho^10) cos^4 theta``,
    ``rho = R0 / r_DA``, ``theta`` the angle at H, counted where ``r_DA^2`` is below ``cut2`` and ``cos theta``
    below ``cos_max``; the gradients of the vectors donor to acceptor and H to donor."""
    dd = tl.load(idx_ptr + e, mask=m, other=0).to(tl.int64)
    hh = tl.load(idx_ptr + n_terms + e, mask=m, other=0).to(tl.int64)
    aa = tl.load(idx_ptr + 2 * n_terms + e, mask=m, other=0).to(tl.int64)
    ax, ay, az = _vec(pos_ptr, cell_ptr, bat_ptr, s_ptr, dd, aa, e, m, HAS_CELL)
    hx, hy, hz = _vec(pos_ptr, cell_ptr, bat_ptr, s_ptr + n_terms * 3, hh, dd, e, m, HAS_CELL)
    ax, ay, az = ax.to(DT), ay.to(DT), az.to(DT)
    hx, hy, hz = hx.to(DT), hy.to(DT), hz.to(DT)
    # H to acceptor through the donor
    wx, wy, wz = hx + ax, hy + ay, hz + az
    r2 = ax * ax + ay * ay + az * az
    r = tl.sqrt(r2 + tiny)
    hh2 = hx * hx + hy * hy + hz * hz
    ww2 = wx * wx + wy * wy + wz * wz
    p = hx * wx + hy * wy + hz * wz
    dn = tl.sqrt(hh2 * ww2 + tiny)
    c = p / dn
    d0 = tl.load(d0_ptr + e, mask=m, other=0.0)
    r0 = tl.load(r0_ptr + e, mask=m, other=0.0)
    rho2 = (r0 / r) * (r0 / r)
    rho10 = rho2 * rho2 * rho2 * rho2 * rho2
    f = 5.0 * rho10 * rho2 - 6.0 * rho10
    # a padding slot (D0 = 0, coincident atoms) stays off: its rho^12 overflows in float32
    on = m & (d0 != 0.0) & (r2 < cut2) & (c < cos_max)
    c2 = c * c
    en = tl.where(on, d0 * f * c2 * c2, 0.0)
    # df/dr = (-60 rho^12 + 60 rho^10) / r
    dfr = tl.where(on, d0 * c2 * c2 * (60.0 * rho10 - 60.0 * rho10 * rho2) / r, 0.0)
    dc = tl.where(on, 4.0 * d0 * f * c2 * c, 0.0)
    d3 = dn * dn * dn
    # dc/dh = w / dn - p ww h / dn^3, dc/dw = h / dn - p hh w / dn^3
    gwx = dc * (hx / dn - p * hh2 * wx / d3)
    gwy = dc * (hy / dn - p * hh2 * wy / d3)
    gwz = dc * (hz / dn - p * hh2 * wz / d3)
    ghx = dc * (wx / dn - p * ww2 * hx / d3) + gwx
    ghy = dc * (wy / dn - p * ww2 * hy / d3) + gwy
    ghz = dc * (wz / dn - p * ww2 * hz / d3) + gwz
    gax = dfr * ax / r + gwx
    gay = dfr * ay / r + gwy
    gaz = dfr * az / r + gwz
    return dd, hh, aa, en, gax, gay, gaz, ghx, ghy, ghz


@triton.jit
def _hbonds_forward(pos_ptr, cell_ptr, bat_ptr, s_ptr, idx_ptr, d0_ptr, r0_ptr, sc_ptr, out_ptr, n_terms,
                    DT: tl.constexpr, HAS_CELL: tl.constexpr, BLOCK: tl.constexpr):
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n_terms
    tiny = tl.load(sc_ptr).to(DT)
    cut2 = tl.load(sc_ptr + 2).to(DT)
    cos_max = tl.load(sc_ptr + 3).to(DT)
    _, hh, _, en, _, _, _, _, _, _ = _hbond(pos_ptr, cell_ptr, bat_ptr, s_ptr, idx_ptr, d0_ptr, r0_ptr, e, m, n_terms,
                                           tiny, cut2, cos_max, DT, HAS_CELL)
    tl.atomic_add(out_ptr + hh, en.to(tl.float64), mask=m, sem="relaxed")


@triton.jit
def _hbonds_backward(pos_ptr, cell_ptr, bat_ptr, s_ptr, idx_ptr, d0_ptr, r0_ptr, sc_ptr, g_ptr, gpos_ptr, gcell_ptr,
                     n_terms, DT: tl.constexpr, HAS_CELL: tl.constexpr, CELL_GRAD: tl.constexpr,
                     BLOCK: tl.constexpr):
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n_terms
    tiny = tl.load(sc_ptr).to(DT)
    cut2 = tl.load(sc_ptr + 2).to(DT)
    cos_max = tl.load(sc_ptr + 3).to(DT)
    dd, hh, aa, _, gax, gay, gaz, ghx, ghy, ghz = _hbond(pos_ptr, cell_ptr, bat_ptr, s_ptr, idx_ptr, d0_ptr, r0_ptr,
                                                         e, m, n_terms, tiny, cut2, cos_max, DT, HAS_CELL)
    g = tl.load(g_ptr + hh, mask=m, other=0.0).to(DT)
    _vec_grad(gpos_ptr, gcell_ptr, bat_ptr, s_ptr, dd, aa, e, m, g * gax, g * gay, g * gaz, HAS_CELL, CELL_GRAD)
    _vec_grad(gpos_ptr, gcell_ptr, bat_ptr, s_ptr + n_terms * 3, hh, dd, e, m, g * ghx, g * ghy, g * ghz, HAS_CELL,
              CELL_GRAD)


class _OnceOnly(torch.autograd.Function):
    """A first derivative passed through unchanged, whose own derivative raises: tied to the inputs of the
    evaluation, so any second derivative through it reaches the error."""

    @staticmethod
    def forward(ctx, value, *anchors):
        return value.clone()

    @staticmethod
    def backward(ctx, g):
        raise RuntimeError("the kernels with a cell gradient (stress) are differentiable once: a second "
                           "derivative needs use_fast=False")


def _delayed(anchors: tuple, *outputs):
    """The outputs of a backward differentiable once: when a graph is recorded they pass through
    :class:`_OnceOnly` tied to ``anchors`` (the evaluation's positions and cell), so first derivatives pass and
    a second one raises instead of coming out silently zero (``once_differentiable`` guards only when an
    incoming gradient requires one, and the energy's does not)."""
    if not torch.is_grad_enabled():
        return outputs
    tied = tuple(a for a in anchors if isinstance(a, Tensor) and a.requires_grad)
    return tuple(_OnceOnly.apply(t, *tied) if isinstance(t, Tensor) else t for t in outputs)


def _cell_args(pos: Tensor, cell: Optional[Tensor], batch: Tensor, shifts: Optional[Tensor]):
    has = cell is not None and shifts is not None
    if not has:
        return pos, pos, pos, False
    return cell.reshape(-1, 9).to(pos.dtype).contiguous(), batch.contiguous(), shifts.to(pos.dtype).contiguous(), True


_FORWARD = {"bond": _bonds_forward, "morse": _morses_forward, "angle": _angles_forward, "torsion": _torsions_forward,
            "inversion": _inversions_forward, "hbond": _hbonds_forward}
_BACKWARD = {"bond": _bonds_backward, "morse": _morses_backward, "angle": _angles_backward,
             "torsion": _torsions_backward, "inversion": _inversions_backward, "hbond": _hbonds_backward}


class BondedTerms(torch.autograd.Function):
    """The per-atom energies ``(N,)`` (float64) of one class of bonded terms (``kind`` ``"bond"``, ``"morse"``,
    ``"angle"``, ``"torsion"``, ``"inversion"`` or ``"hbond"``) from the positions ``(N, 3)``, the cells
    ``(B, 3, 3)``, the atoms' structures, the term atoms ``idx`` ``(A, M)``, their image shifts
    ``(A - 1, M, 3)`` (or ``None`` without cells) and the term parameters ``params`` (see the kernels); the
    model's precision ``dtype``. Differentiable once, in the positions and the cells."""

    @staticmethod
    def forward(ctx, pos, cell, batch, idx, shifts, kind: str, params: tuple, dtype, scalars, options: dict):
        N = int(pos.shape[0])
        M = int(idx.shape[1])
        out = torch.zeros(N, dtype=torch.float64, device=pos.device)
        cell_t, bat_t, s_t, has = _cell_args(pos, cell, batch, shifts)
        pos_c = pos.detach().contiguous()
        idx = idx.contiguous()
        params = tuple(p.contiguous() for p in params)
        dt = tl.float64 if dtype == torch.float64 else tl.float32
        grid = (triton.cdiv(max(M, 1), BLOCK),)
        if M:
            _go(_FORWARD[kind], grid, pos_c, cell_t, bat_t, s_t, idx, *params, scalars, out, M, DT=dt, HAS_CELL=has,
                BLOCK=BLOCK, num_warps=WARPS, **(options if kind == "torsion" else {}))
        # the inputs themselves too: a further derivative of the backward's outputs must reach them
        ctx.save_for_backward(pos_c, cell_t, bat_t, s_t, idx, scalars, *params, pos,
                              cell if cell is not None else pos)
        ctx.kind, ctx.dt, ctx.has, ctx.options, ctx.M = kind, dt, has, options, M
        ctx.cell_grad = has and bool(ctx.needs_input_grad[1])
        ctx.cell_shape = None if cell is None else cell.shape
        ctx.cell_dtype = None if cell is None else cell.dtype
        return out

    @staticmethod
    def backward(ctx, g):
        pos_c, cell_t, bat_t, s_t, idx, scalars, *params, pos_in, cell_in = ctx.saved_tensors
        N = int(pos_c.shape[0])
        gpos = torch.zeros(N, 3, dtype=torch.float64, device=pos_c.device)
        gcell = torch.zeros(cell_t.shape[0], 9, dtype=torch.float64, device=pos_c.device) if ctx.has else gpos
        g = g.contiguous()
        grid = (triton.cdiv(max(ctx.M, 1), BLOCK),)
        if ctx.M:
            args = (pos_c, cell_t, bat_t, s_t, idx, *params, scalars, g, gpos, gcell, ctx.M)
            _go(_BACKWARD[ctx.kind], grid, *args, DT=ctx.dt, HAS_CELL=ctx.has, CELL_GRAD=ctx.cell_grad, BLOCK=BLOCK,
                num_warps=WARPS, **(ctx.options if ctx.kind == "torsion" else {}))
        g_cell = None
        if ctx.cell_shape is not None and ctx.cell_grad:
            g_cell = gcell.reshape(ctx.cell_shape).to(ctx.cell_dtype)
        return _delayed((pos_in, cell_in), gpos.to(pos_c.dtype), g_cell, None, None, None, None, None, None, None,
                        None)


def bonded_terms(pos: Tensor, cell: Optional[Tensor], batch: Tensor, idx: Tensor, shifts: Optional[Tensor], kind: str,
                 params: tuple, dtype: torch.dtype, tiny: float, bound: float = 1.0, options: Optional[dict] = None,
                 extra: tuple = (0.0, 0.0)) -> Tensor:
    """The per-atom energies ``(N,)`` in ``dtype`` of one class of bonded terms (see :class:`BondedTerms`);
    ``extra`` the squared cutoff and the cosine bound of the hydrogen bonds."""
    sc = _scalars((float(tiny), float(bound)) + tuple(float(x) for x in extra), pos.device)
    e = BondedTerms.apply(pos, cell, batch, idx, shifts, kind, tuple(p.detach() for p in params), dtype, sc,
                          options or {})
    return e.to(dtype)


class PairTerms(torch.autograd.Function):
    """The van der Waals and Coulomb energies per atom ``(N, 2)`` (float64) of the pairs ``i < k`` of a directed
    pair list ``(src, dst)`` with image shifts ``shifts`` ``(E, 3)`` (or ``None``), from the positions and cells,
    without the pairs in the exclusion rows ``(ptr, cols, width)`` of one structure of ``n_local`` atoms (or
    ``None``), with compact atom types ``t``, charges ``q`` and the ``(T, T, 3)`` table of the form;
    differentiable once, in the positions and the cells."""

    @staticmethod
    def forward(ctx, pos, cell, batch, src, dst, shifts, exclusions, n_local, t, q, table, dtype, scalars, options):
        N = int(pos.shape[0])
        E = int(src.shape[0])
        out = torch.zeros(N, 2, dtype=torch.float64, device=pos.device)
        cell_t, bat_t, s_t, has = _cell_args(pos, cell, batch, shifts)
        pos_c = pos.detach().contiguous()
        xp, xk, width = exclusions if exclusions is not None else (pos_c, pos_c, 0)
        dt = tl.float64 if dtype == torch.float64 else tl.float32
        flags = dict(DT=dt, HAS_CELL=has, EXCL=exclusions is not None, MAXE=int(width), **options)
        args = (src.contiguous(), dst.contiguous(), s_t, pos_c, cell_t, bat_t, xp, xk, t.contiguous(),
                q.contiguous().to(dtype), table.contiguous().to(dtype), scalars)
        if E:
            _go(_pairs_forward, (triton.cdiv(E, BLOCK),), *args, out, E, int(n_local), int(table.shape[0]), **flags,
                                                     BLOCK=BLOCK, num_warps=WARPS)
        ctx.save_for_backward(*[a for a in args if isinstance(a, Tensor)], pos, cell if cell is not None else pos)
        ctx.flags, ctx.E, ctx.n_local, ctx.T, ctx.has = flags, E, int(n_local), int(table.shape[0]), has
        ctx.cell_grad = has and bool(ctx.needs_input_grad[1])
        ctx.cell_shape = None if cell is None else cell.shape
        ctx.cell_dtype = None if cell is None else cell.dtype
        return out

    @staticmethod
    def backward(ctx, g):
        *args, pos_in, cell_in = ctx.saved_tensors
        pos_c, cell_t = args[3], args[4]
        gpos = torch.zeros(pos_c.shape[0], 3, dtype=torch.float64, device=pos_c.device)
        gcell = torch.zeros(cell_t.shape[0], 9, dtype=torch.float64, device=pos_c.device) if ctx.has else gpos
        if ctx.E:
            _go(_pairs_backward, (triton.cdiv(ctx.E, BLOCK),), *args, g.contiguous(), gpos, gcell, ctx.E, ctx.n_local,
                                                          ctx.T, **ctx.flags, CELL_GRAD=ctx.cell_grad, BLOCK=BLOCK,
                                                          num_warps=WARPS)
        g_cell = gcell.reshape(ctx.cell_shape).to(ctx.cell_dtype) if ctx.cell_shape is not None and ctx.cell_grad \
            else None
        return _delayed((pos_in, cell_in), gpos.to(pos_c.dtype), g_cell, *((None,) * 12))


def pair_terms(pos: Tensor, cell: Optional[Tensor], batch: Tensor, src: Tensor, dst: Tensor,
               shifts: Optional[Tensor], exclusions: Optional[tuple], n_local: int, t: Tensor, q: Tensor,
               table: Tensor, dtype: torch.dtype, ke: float, tiny: float, cutoff: float, width: float,
               form: int) -> tuple:
    """The per-atom van der Waals and Coulomb energies ``(N,)`` each in ``dtype`` of :class:`PairTerms`."""
    sc = _scalars((float(ke), float(tiny), float(cutoff) ** 2, float(cutoff - width),
                   1.0 / width if width > 0 else 0.0), pos.device)
    e = PairTerms.apply(pos, cell, batch, src, dst, shifts, exclusions, n_local, t, q.detach(), table.detach(),
                        dtype, sc, dict(FORM=int(form), SWITCH=width > 0))
    return e[:, 0].to(dtype), e[:, 1].to(dtype)



@triton.jit
def _excluded_kernel(src_ptr, dst_ptr, xp_ptr, xk_ptr, out_ptr, n_pairs, n_local, MAXE: tl.constexpr,
                     BLOCK: tl.constexpr):
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n_pairs
    i = tl.load(src_ptr + e, mask=m, other=0).to(tl.int64) % n_local
    k = tl.load(dst_ptr + e, mask=m, other=0).to(tl.int64) % n_local
    # the row of the lower atom (the rows hold each excluded pair once, in the row of its first atom)
    hit = _excluded_row(xp_ptr, xk_ptr, tl.minimum(i, k), tl.maximum(i, k).to(tl.int32), m, MAXE)
    tl.store(out_ptr + e, hit.to(tl.int8), mask=m)


def excluded(src: Tensor, dst: Tensor, exclusions: tuple, n_local: int) -> Tensor:
    """Whether each pair ``(src, dst)`` (either order, atoms of a batch of one structure's copies) is one of the
    exclusions ``(ptr, cols, width)`` of :func:`~xnn.ffnn.models.classical_fast.exclusion_rows`, as ``(P,)``
    bool."""
    ptr, cols, width = exclusions
    n = int(src.shape[0])
    out = torch.empty(n, dtype=torch.int8, device=src.device)
    if n > 0:
        BLOCK = 512
        _go(_excluded_kernel, (triton.cdiv(n, BLOCK),), src.contiguous(), dst.contiguous(), ptr, cols, out, n,
            n_local, MAXE=int(width), BLOCK=BLOCK)
    return out.bool()


def available(device: Optional[torch.device] = None) -> bool:
    """Whether the kernels can run (a CUDA device and Triton)."""
    return torch.cuda.is_available() and (device is None or device.type == "cuda")


__all__ = ["PairTerms", "pair_terms", "BondedTerms", "bonded_terms", "excluded", "available"]
