"""Triton kernels of the ReaxFF terms (the fast path of :class:`~xnn.ffnn.models.ReaxFF`), the energies of
one evaluation from the positions: the tapered van der Waals and shielded Coulomb pairs. Their backward kernels
are generated from these forward kernels (``tools/generate_amoeba_derivatives.py --target reaxff``: the reverse
sweep with the positions and the inputs declared as the loaded arrays), so each kernel here is written once.
"""
from typing import Optional

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra import libdevice

from .classical_kernels import _add_sorted, _segments, _vec  # noqa: F401  (device functions of the kernels)

BLOCK, WARPS = 128, 4

# the per-species-pair table of the nonbonded kernel: 2 rvdw, alfa, Devdw, (1 / gammaw)^vdw1, (1 / gamma)^3,
# the inner-wall radius (1 where there is none), its exponent and its energy (0 where there is none)
NB_COLUMNS = 8


@triton.jit
def _taper(r, rc):
    """The 7th-order nonbonded taper ``1 - 35 x^4 + 84 x^5 - 70 x^6 + 20 x^7``, ``x = r / rc``."""
    x = r / rc
    x2 = x * x
    x4 = x2 * x2
    return 1.0 + x4 * (-35.0 + x * (84.0 + x * (-70.0 + 20.0 * x)))


@triton.jit
def _pow(a, b):
    """``a^b`` for ``a > 0`` as ``exp(b log a)`` (differentiable by the generator's rules)."""
    return tl.exp(b * tl.log(a))


@triton.jit
def _nb_pair(src_ptr, dst_ptr, sh_ptr, pos_ptr, cell_ptr, bat_ptr, t_ptr, q_ptr, tab_ptr, sc_ptr, e, m0, n_types,
             DT: tl.constexpr, HAS_CELL: tl.constexpr, SHIELD: tl.constexpr, WALL: tl.constexpr):
    """A pair ``i <= k`` of a directed pair list (an atom's own image ``i = k`` at half weight, a pair of an atom
    with itself at no distance inactive): its van der Waals and Coulomb energies, tapered and weighted."""
    i = tl.load(src_ptr + e, mask=m0, other=0).to(tl.int64)
    k = tl.load(dst_ptr + e, mask=m0, other=0).to(tl.int64)
    m = m0 & (i <= k)
    x, y, z = _vec(pos_ptr, cell_ptr, bat_ptr, sh_ptr, i, k, e, m, HAS_CELL)
    x = x.to(DT)
    y = y.to(DT)
    z = z.to(DT)
    r2 = x * x + y * y + z * z
    own = i == k
    m = m & (~own | (r2 > 0.0))
    # float64 scalars in the working precision: the Coulomb constant, the cutoff, the exponent vdw1, the
    # distance offset of the eager terms (r = sqrt(r^2 + safety)) and 1/3
    ke = tl.load(sc_ptr).to(DT)
    rc = tl.load(sc_ptr + 1).to(DT)
    p = tl.load(sc_ptr + 2).to(DT)
    tiny = tl.load(sc_ptr + 3).to(DT)
    # 1/3 in float64 (a Python float in a kernel is a float32 constant)
    third = tl.load(sc_ptr + 4).to(DT)
    on = m & (r2 < rc * rc)
    w = tl.where(on, tl.where(own, 0.5, 1.0), 0.0).to(DT)
    r = tl.sqrt(r2 + tiny)
    tp = (tl.load(t_ptr + i, mask=m, other=0) * n_types + tl.load(t_ptr + k, mask=m, other=0)) * 8
    rv2 = tl.load(tab_ptr + tp, mask=m, other=1.0)
    alfa = tl.load(tab_ptr + tp + 1, mask=m, other=0.0)
    dv = tl.load(tab_ptr + tp + 2, mask=m, other=0.0)
    gw = tl.load(tab_ptr + tp + 3, mask=m, other=0.0)
    gm3 = tl.load(tab_ptr + tp + 4, mask=m, other=1.0)
    tap = _taper(r, rc)
    if SHIELD:
        f13 = _pow(_pow(r, p) + gw, 1.0 / p)
    else:
        f13 = r
    ex = tl.exp(0.5 * alfa * (1.0 - f13 / rv2))
    ev = dv * (ex * ex - 2.0 * ex)
    if WALL:
        rcore = tl.load(tab_ptr + tp + 5, mask=m, other=1.0)
        acore = tl.load(tab_ptr + tp + 6, mask=m, other=0.0)
        ecore = tl.load(tab_ptr + tp + 7, mask=m, other=0.0)
        ev += ecore * tl.exp(acore * (1.0 - r / rcore))
    qi = tl.load(q_ptr + i, mask=m, other=0.0)
    qk = tl.load(q_ptr + k, mask=m, other=0.0)
    rth = _pow(r * r * r + gm3, third)
    ec = ke * qi * qk / rth
    return i, k, m, w * tap * ev, w * tap * ec


@triton.jit
def _nonbonded_forward(src_ptr, dst_ptr, sh_ptr, pos_ptr, cell_ptr, bat_ptr, t_ptr, q_ptr, tab_ptr, sc_ptr, e_ptr,
                       n_edges, n_types, DT: tl.constexpr, HAS_CELL: tl.constexpr, SHIELD: tl.constexpr,
                       WALL: tl.constexpr, BLOCK: tl.constexpr):
    """The per-atom halves of the van der Waals and Coulomb energies ``(N, 2)`` of the pairs, added into ``e``."""
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    i, k, m, ev, ec = _nb_pair(src_ptr, dst_ptr, sh_ptr, pos_ptr, cell_ptr, bat_ptr, t_ptr, q_ptr, tab_ptr, sc_ptr,
                               e, e < n_edges, n_types, DT, HAS_CELL, SHIELD, WALL)
    hv = (0.5 * ev).to(tl.float64)
    hc = (0.5 * ec).to(tl.float64)
    tl.atomic_add(e_ptr + i * 2, hv, mask=m, sem="relaxed")
    tl.atomic_add(e_ptr + i * 2 + 1, hc, mask=m, sem="relaxed")
    tl.atomic_add(e_ptr + k * 2, hv, mask=m, sem="relaxed")
    tl.atomic_add(e_ptr + k * 2 + 1, hc, mask=m, sem="relaxed")


# the per-species-pair table of the bond-order corrections and the bond energy: boc3, boc4, boc5, the
# over-coordination and 1-3 corrections switched on (1 / 0), Desi, Depi, Depp, be1, be2
BC_COLUMNS = 10


@triton.jit
def _gexp(x):
    """``exp(min(x, 40))`` (the guarded exponential of the eager terms)."""
    return tl.exp(tl.minimum(x, 40.0))


@triton.jit
def _bond_corr(src_ptr, dst_ptr, bop_ptr, bpi_ptr, bpp_ptr, dp_ptr, t_ptr, spt_ptr, tab_ptr, gen_ptr, e, m,
               n_types, DT: tl.constexpr):
    """A bond's corrected bond orders (LAMMPS conventions: ``f1`` on the pi orders whenever the over-coordination
    correction is on, orders below 1e-10 zero) and its bond energy ``e_si``."""
    a = tl.load(dst_ptr + e, mask=m, other=0).to(tl.int64)
    b = tl.load(src_ptr + e, mask=m, other=0).to(tl.int64)
    ta = tl.load(t_ptr + a, mask=m, other=0)
    tb = tl.load(t_ptr + b, mask=m, other=0)
    bop_si = tl.load(bop_ptr + e, mask=m, other=0.0).to(DT)
    bpi = tl.load(bpi_ptr + e, mask=m, other=0.0).to(DT)
    bpp = tl.load(bpp_ptr + e, mask=m, other=0.0).to(DT)
    bop = bop_si + bpi + bpp
    dpa = tl.load(dp_ptr + a, mask=m, other=0.0).to(DT)
    dpb = tl.load(dp_ptr + b, mask=m, other=0.0).to(DT)
    val_a = tl.load(spt_ptr + ta * 2, mask=m, other=1.0).to(DT)
    val_b = tl.load(spt_ptr + tb * 2, mask=m, other=1.0).to(DT)
    vboc_a = tl.load(spt_ptr + ta * 2 + 1, mask=m, other=0.0).to(DT)
    vboc_b = tl.load(spt_ptr + tb * 2 + 1, mask=m, other=0.0).to(DT)
    boc1 = tl.load(gen_ptr).to(DT)
    boc2 = tl.load(gen_ptr + 1).to(DT)
    dva = dpa - val_a
    dvb = dpb - val_b
    f2 = _gexp(-boc1 * dva) + _gexp(-boc1 * dvb)
    f3 = (-1.0 / boc2) * tl.log(0.5 * (_gexp(-boc2 * dva) + _gexp(-boc2 * dvb)))
    f1 = 0.5 * ((val_a + f2) / (val_a + f2 + f3) + (val_b + f2) / (val_b + f2 + f3))
    tp = (ta * n_types + tb) * 10
    boc3 = tl.load(tab_ptr + tp, mask=m, other=0.0).to(DT)
    boc4 = tl.load(tab_ptr + tp + 1, mask=m, other=0.0).to(DT)
    boc5 = tl.load(tab_ptr + tp + 2, mask=m, other=0.0).to(DT)
    ov = tl.load(tab_ptr + tp + 3, mask=m, other=0.0)
    c13 = tl.load(tab_ptr + tp + 4, mask=m, other=0.0)
    f4 = 1.0 / (1.0 + tl.exp(-(boc3 * (boc4 * bop * bop - (dpa - vboc_a)) - boc5)))
    f5 = 1.0 / (1.0 + tl.exp(-(boc3 * (boc4 * bop * bop - (dpb - vboc_b)) - boc5)))
    f1 = tl.where(ov > 0.5, f1, 1.0)
    a0 = f1 * tl.where(c13 > 0.5, f4 * f5, 1.0)
    bo0 = bop * a0
    bopi = bpi * a0 * f1
    bopp = bpp * a0 * f1
    bosi = bo0 - bopi - bopp
    bo0 = tl.where(bo0 < 1.0e-10, 0.0, bo0)
    bosi = tl.where(bosi < 1.0e-10, 0.0, bosi)
    bopi = tl.where(bopi < 1.0e-10, 0.0, bopi)
    bopp = tl.where(bopp < 1.0e-10, 0.0, bopp)
    desi = tl.load(tab_ptr + tp + 5, mask=m, other=0.0).to(DT)
    depi = tl.load(tab_ptr + tp + 6, mask=m, other=0.0).to(DT)
    depp = tl.load(tab_ptr + tp + 7, mask=m, other=0.0).to(DT)
    be1 = tl.load(tab_ptr + tp + 8, mask=m, other=0.0).to(DT)
    be2 = tl.load(tab_ptr + tp + 9, mask=m, other=1.0).to(DT)
    on = bosi > 0.0
    powb = tl.where(on, _pow(tl.where(on, bosi, 1.0), be2), 0.0)
    esi = desi * bosi * tl.exp(be1 * (1.0 - powb)) + depi * bopi + depp * bopp
    return a, bo0, bosi, bopi, bopp, esi


@triton.jit
def _bond_corrections_forward(src_ptr, dst_ptr, bop_ptr, bpi_ptr, bpp_ptr, dp_ptr, t_ptr, spt_ptr, tab_ptr, gen_ptr,
                              bo0_ptr, bosi_ptr, bopi_ptr, bopp_ptr, e_ptr, n_bonds, n_types, DT: tl.constexpr,
                              BLOCK: tl.constexpr):
    """The corrected bond orders per bond (stored) and ``-e_si / 2`` per bond added to its first atom."""
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n_bonds
    a, bo0, bosi, bopi, bopp, esi = _bond_corr(src_ptr, dst_ptr, bop_ptr, bpi_ptr, bpp_ptr, dp_ptr, t_ptr, spt_ptr,
                                               tab_ptr, gen_ptr, e, m, n_types, DT)
    tl.store(bo0_ptr + e, bo0, mask=m)
    tl.store(bosi_ptr + e, bosi, mask=m)
    tl.store(bopi_ptr + e, bopi, mask=m)
    tl.store(bopp_ptr + e, bopp, mask=m)
    tl.atomic_add(e_ptr + a, (-0.5 * esi).to(tl.float64), mask=m, sem="relaxed")


# the per-angle-type table (i, j, k species) of the angle kernel: val1, theta0, val4, val7, val2, pen1, coa1
ANG_COLUMNS = 7


@triton.jit
def _angle(e1_ptr, e2_ptr, c_ptr, i_ptr, k_ptr, bv_ptr, br_ptr, bo_ptr, fbo_ptr, t_ptr, sbo_ptr, dang_ptr, f9_ptr,
           sig_ptr, del_ptr, ang_ptr, sp_ptr, gen_ptr, e, m, n_types, DT: tl.constexpr, NN: tl.constexpr):
    """A valence angle's energy, its penalty and its three-body conjugation. LAMMPS conventions: the bond orders
    enter as ``BO - thb``, a negative ``val1`` as ``-val1 exp(-val2 dtheta^2)``. ``NN`` the ReaxFF-nn conventions:
    the cosine by the law of cosines, the bond orders ``bo`` (already ``relu(BO - atol)``) with their tapers
    ``fbo``, ``(bo + safety)^val4``, ``theta0`` in degrees over 57.29577951."""
    e1 = tl.load(e1_ptr + e, mask=m, other=0).to(tl.int64)
    e2 = tl.load(e2_ptr + e, mask=m, other=0).to(tl.int64)
    c = tl.load(c_ptr + e, mask=m, other=0).to(tl.int64)
    ai = tl.load(i_ptr + e, mask=m, other=0).to(tl.int64)
    ak = tl.load(k_ptr + e, mask=m, other=0).to(tl.int64)
    ux = tl.load(bv_ptr + e1 * 3, mask=m, other=1.0).to(DT)
    uy = tl.load(bv_ptr + e1 * 3 + 1, mask=m, other=0.0).to(DT)
    uz = tl.load(bv_ptr + e1 * 3 + 2, mask=m, other=0.0).to(DT)
    vx = tl.load(bv_ptr + e2 * 3, mask=m, other=1.0).to(DT)
    vy = tl.load(bv_ptr + e2 * 3 + 1, mask=m, other=0.0).to(DT)
    vz = tl.load(bv_ptr + e2 * 3 + 2, mask=m, other=0.0).to(DT)
    ru = tl.load(br_ptr + e1, mask=m, other=1.0).to(DT)
    rv = tl.load(br_ptr + e2, mask=m, other=1.0).to(DT)
    val6 = tl.load(gen_ptr).to(DT)
    val10 = tl.load(gen_ptr + 1).to(DT)
    pen2 = tl.load(gen_ptr + 2).to(DT)
    coa3 = tl.load(gen_ptr + 3).to(DT)
    coa4 = tl.load(gen_ptr + 4).to(DT)
    thb = tl.load(gen_ptr + 5).to(DT)
    pi = tl.load(gen_ptr + 6).to(DT)
    bound = tl.load(gen_ptr + 7).to(DT)
    if NN:
        wx = ux - vx
        wy = uy - vy
        wz = uz - vz
        cs = (ru * ru + rv * rv - (wx * wx + wy * wy + wz * wz)) / (2.0 * ru * rv)
    else:
        cs = (ux * vx + uy * vy + uz * vz) / (ru * rv)
    theta = libdevice.acos(tl.minimum(tl.maximum(cs, -bound), bound))
    ti = tl.load(t_ptr + ai, mask=m, other=0)
    tj = tl.load(t_ptr + c, mask=m, other=0)
    tk = tl.load(t_ptr + ak, mask=m, other=0)
    ta = ((ti * n_types + tj) * n_types + tk) * 7
    val1 = tl.load(ang_ptr + ta, mask=m, other=0.0).to(DT)
    th0p = tl.load(ang_ptr + ta + 1, mask=m, other=0.0).to(DT)
    val4 = tl.load(ang_ptr + ta + 2, mask=m, other=1.0).to(DT)
    val7 = tl.load(ang_ptr + ta + 3, mask=m, other=0.0).to(DT)
    val2 = tl.load(ang_ptr + ta + 4, mask=m, other=0.0).to(DT)
    pen1 = tl.load(ang_ptr + ta + 5, mask=m, other=0.0).to(DT)
    coa1 = tl.load(ang_ptr + ta + 6, mask=m, other=0.0).to(DT)
    val3 = tl.load(sp_ptr + tj * 2, mask=m, other=0.0).to(DT)
    val5 = tl.load(sp_ptr + tj * 2 + 1, mask=m, other=1.0).to(DT)
    sbo2 = tl.load(sbo_ptr + c, mask=m, other=0.0).to(DT)
    dang = tl.load(dang_ptr + c, mask=m, other=0.0).to(DT)
    f9 = tl.load(f9_ptr + c, mask=m, other=0.0).to(DT)
    sig = tl.load(sig_ptr + c, mask=m, other=0.0).to(DT)
    d_i = tl.load(del_ptr + ai, mask=m, other=0.0).to(DT)
    d_k = tl.load(del_ptr + ak, mask=m, other=0.0).to(DT)
    if NN:
        # the ReaxFF-nn constants: the safety offset and the degrees of a radian in gen 5 and 6
        boa_ij = tl.load(bo_ptr + e1, mask=m, other=1.0).to(DT)
        boa_jk = tl.load(bo_ptr + e2, mask=m, other=1.0).to(DT)
        fijk = tl.load(fbo_ptr + e1, mask=m, other=0.0).to(DT) * tl.load(fbo_ptr + e2, mask=m, other=0.0).to(DT)
        theta0 = (180.0 - th0p * (1.0 - tl.exp(-val10 * (2.0 - sbo2)))) / pi
        f7 = (1.0 - tl.exp(-val3 * _pow(boa_ij + thb, val4))) * (1.0 - tl.exp(-val3 * _pow(boa_jk + thb, val4)))
    else:
        boa_ij = tl.load(bo_ptr + e1, mask=m, other=1.0).to(DT) - thb
        boa_jk = tl.load(bo_ptr + e2, mask=m, other=1.0).to(DT) - thb
        fijk = boa_ij * 0.0 + 1.0
        theta0 = (180.0 - th0p * (1.0 - tl.exp(-val10 * (2.0 - sbo2)))) * (pi / 180.0)
        f7 = (1.0 - tl.exp(-val3 * _pow(boa_ij, val4))) * (1.0 - tl.exp(-val3 * _pow(boa_jk, val4)))
    exp6 = _gexp(val6 * dang)
    exp7 = _gexp(-val7 * dang)
    f8 = val5 - (val5 - 1.0) * (2.0 + exp6) / (1.0 + exp6 + exp7)
    dth = theta0 - theta
    expang = tl.exp(-val2 * dth * dth)
    if NN:
        eang = fijk * f7 * f8 * (val1 - val1 * expang)
    else:
        eang = f7 * f8 * tl.where(val1 >= 0.0, val1 * (1.0 - expang), -val1 * expang)
    bi2 = boa_ij - 2.0
    bk2 = boa_jk - 2.0
    epen = pen1 * f9 * tl.exp(-pen2 * bi2 * bi2) * tl.exp(-pen2 * bk2 * bk2) * fijk
    ci = d_i - boa_ij
    ck = d_k - boa_jk
    hi = boa_ij - 1.5
    hk = boa_jk - 1.5
    etcon = coa1 * sig * tl.exp(-coa3 * ci * ci) * tl.exp(-coa3 * ck * ck) * tl.exp(-coa4 * hi * hi) \
        * tl.exp(-coa4 * hk * hk) * fijk
    return c, eang, epen, etcon


@triton.jit
def _angles_forward(e1_ptr, e2_ptr, c_ptr, i_ptr, k_ptr, bv_ptr, br_ptr, bo_ptr, fbo_ptr, t_ptr, sbo_ptr, dang_ptr,
                    f9_ptr, sig_ptr, del_ptr, ang_ptr, sp_ptr, gen_ptr, out_ptr, n_angles, n_types, DT: tl.constexpr,
                    NN: tl.constexpr, BLOCK: tl.constexpr):
    """The angle, penalty and three-body conjugation energies ``(N, 3)`` on the angles' centers."""
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n_angles
    c, eang, epen, etcon = _angle(e1_ptr, e2_ptr, c_ptr, i_ptr, k_ptr, bv_ptr, br_ptr, bo_ptr, fbo_ptr, t_ptr, sbo_ptr,
                                  dang_ptr, f9_ptr, sig_ptr, del_ptr, ang_ptr, sp_ptr, gen_ptr, e, m, n_types, DT, NN)
    tl.atomic_add(out_ptr + c * 3, eang.to(tl.float64), mask=m, sem="relaxed")
    tl.atomic_add(out_ptr + c * 3 + 1, epen.to(tl.float64), mask=m, sem="relaxed")
    tl.atomic_add(out_ptr + c * 3 + 2, etcon.to(tl.float64), mask=m, sem="relaxed")


# the per-torsion-type table (i, j, k, l species) of the torsion kernel: tor1, V1, V2, V3, cot1
TOR_COLUMNS = 5


@triton.jit
def _torsion(ei_ptr, ec_ptr, el_ptr, ai_ptr, aj_ptr, ak_ptr, al_ptr, bv_ptr, br_ptr, bo_ptr, bpi_ptr, t_ptr,
             dang_ptr, tor_ptr, gen_ptr, e, m, n_types, DT: tl.constexpr):
    """A torsion's energy and its four-body conjugation (LAMMPS conventions) over the bonds i-j, j-k (central),
    k-l with ``b1 = x_j - x_i``, ``b2 = x_k - x_j``, ``b3 = x_l - x_k`` from the bond vectors."""
    ei = tl.load(ei_ptr + e, mask=m, other=0).to(tl.int64)
    ec = tl.load(ec_ptr + e, mask=m, other=0).to(tl.int64)
    el = tl.load(el_ptr + e, mask=m, other=0).to(tl.int64)
    aj = tl.load(aj_ptr + e, mask=m, other=0).to(tl.int64)
    ak = tl.load(ak_ptr + e, mask=m, other=0).to(tl.int64)
    t1 = tl.load(t_ptr + tl.load(ai_ptr + e, mask=m, other=0).to(tl.int64), mask=m, other=0)
    t2 = tl.load(t_ptr + aj, mask=m, other=0)
    t3 = tl.load(t_ptr + ak, mask=m, other=0)
    t4 = tl.load(t_ptr + tl.load(al_ptr + e, mask=m, other=0).to(tl.int64), mask=m, other=0)
    ax = tl.load(bv_ptr + ei * 3, mask=m, other=1.0).to(DT)
    ay = tl.load(bv_ptr + ei * 3 + 1, mask=m, other=0.0).to(DT)
    az = tl.load(bv_ptr + ei * 3 + 2, mask=m, other=0.0).to(DT)
    cx = tl.load(bv_ptr + ec * 3, mask=m, other=1.0).to(DT)
    cy = tl.load(bv_ptr + ec * 3 + 1, mask=m, other=0.0).to(DT)
    cz = tl.load(bv_ptr + ec * 3 + 2, mask=m, other=0.0).to(DT)
    lx = tl.load(bv_ptr + el * 3, mask=m, other=1.0).to(DT)
    ly = tl.load(bv_ptr + el * 3 + 1, mask=m, other=0.0).to(DT)
    lz = tl.load(bv_ptr + el * 3 + 2, mask=m, other=0.0).to(DT)
    b2x = -cx
    b2y = -cy
    b2z = -cz
    b3x = -lx
    b3y = -ly
    b3z = -lz
    rij = tl.load(br_ptr + ei, mask=m, other=1.0).to(DT)
    rjk = tl.load(br_ptr + ec, mask=m, other=1.0).to(DT)
    rkl = tl.load(br_ptr + el, mask=m, other=1.0).to(DT)
    c_ijk = -(ax * b2x + ay * b2y + az * b2z) / (rij * rjk)
    c_jkl = -(b2x * b3x + b2y * b3y + b2z * b3z) / (rjk * rkl)
    s_ijk = tl.sqrt(tl.maximum(1.0 - c_ijk * c_ijk, 1.0e-20))
    s_jkl = tl.sqrt(tl.maximum(1.0 - c_jkl * c_jkl, 1.0e-20))
    n1x = ay * b2z - az * b2y
    n1y = az * b2x - ax * b2z
    n1z = ax * b2y - ay * b2x
    n2x = b2y * b3z - b2z * b3y
    n2y = b2z * b3x - b2x * b3z
    n2z = b2x * b3y - b2y * b3x
    den = tl.sqrt((n1x * n1x + n1y * n1y + n1z * n1z) * (n2x * n2x + n2y * n2y + n2z * n2z))
    ok = den > 1.0e-20
    cw = tl.where(ok, (n1x * n2x + n1y * n2y + n1z * n2z) / tl.where(ok, den, 1.0), 1.0)
    cw = tl.minimum(tl.maximum(cw, -1.0), 1.0)
    c2w = 2.0 * cw * cw - 1.0
    c3w = 4.0 * cw * cw * cw - 3.0 * cw
    tor2 = tl.load(gen_ptr).to(DT)
    tor3 = tl.load(gen_ptr + 1).to(DT)
    tor4 = tl.load(gen_ptr + 2).to(DT)
    cot2 = tl.load(gen_ptr + 3).to(DT)
    thb = tl.load(gen_ptr + 4).to(DT)
    bij = tl.load(bo_ptr + ei, mask=m, other=1.0).to(DT) - thb
    bjk = tl.load(bo_ptr + ec, mask=m, other=1.0).to(DT) - thb
    bkl = tl.load(bo_ptr + el, mask=m, other=1.0).to(DT) - thb
    f10 = (1.0 - tl.exp(-tor2 * bij)) * (1.0 - tl.exp(-tor2 * bjk)) * (1.0 - tl.exp(-tor2 * bkl))
    delt = tl.load(dang_ptr + aj, mask=m, other=0.0).to(DT) + tl.load(dang_ptr + ak, mask=m, other=0.0).to(DT)
    exp3 = _gexp(-tor3 * delt)
    f11 = (2.0 + exp3) / (1.0 + exp3 + _gexp(tor4 * delt))
    tt = (((t1 * n_types + t2) * n_types + t3) * n_types + t4) * 5
    tor1 = tl.load(tor_ptr + tt, mask=m, other=0.0).to(DT)
    v1 = tl.load(tor_ptr + tt + 1, mask=m, other=0.0).to(DT)
    v2 = tl.load(tor_ptr + tt + 2, mask=m, other=0.0).to(DT)
    v3 = tl.load(tor_ptr + tt + 3, mask=m, other=0.0).to(DT)
    cot1 = tl.load(tor_ptr + tt + 4, mask=m, other=0.0).to(DT)
    q = 2.0 - tl.load(bpi_ptr + ec, mask=m, other=0.0).to(DT) - f11
    expv2 = tl.exp(tor1 * q * q)
    etor = f10 * s_ijk * s_jkl * (0.5 * v1 * (1.0 + cw) + 0.5 * v2 * expv2 * (1.0 - c2w) + 0.5 * v3 * (1.0 + c3w))
    hi = bij - 1.5
    hj = bjk - 1.5
    hk = bkl - 1.5
    f12 = tl.exp(-cot2 * (hi * hi + hj * hj + hk * hk))
    efcon = cot1 * f12 * (1.0 + (cw * cw - 1.0) * s_ijk * s_jkl)
    return aj, etor, efcon


@triton.jit
def _torsions_forward(ei_ptr, ec_ptr, el_ptr, ai_ptr, aj_ptr, ak_ptr, al_ptr, bv_ptr, br_ptr, bo_ptr, bpi_ptr, t_ptr,
                      dang_ptr, tor_ptr, gen_ptr, out_ptr, n_torsions, n_types, DT: tl.constexpr,
                      BLOCK: tl.constexpr):
    """The torsion and four-body conjugation energies ``(N, 2)`` on the torsions' first central atoms."""
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n_torsions
    aj, etor, efcon = _torsion(ei_ptr, ec_ptr, el_ptr, ai_ptr, aj_ptr, ak_ptr, al_ptr, bv_ptr, br_ptr, bo_ptr,
                               bpi_ptr, t_ptr, dang_ptr, tor_ptr, gen_ptr, e, m, n_types, DT)
    tl.atomic_add(out_ptr + aj * 2, etor.to(tl.float64), mask=m, sem="relaxed")
    tl.atomic_add(out_ptr + aj * 2 + 1, efcon.to(tl.float64), mask=m, sem="relaxed")


@triton.jit
def _torsion_nn(ei_ptr, ec_ptr, el_ptr, ai_ptr, aj_ptr, ak_ptr, al_ptr, bv_ptr, br_ptr, bo_ptr, bpi_ptr, fbo_ptr,
                t_ptr, dang_ptr, tor_ptr, gen_ptr, e, m, n_types, DT: tl.constexpr):
    """A torsion's energy and its four-body conjugation (ReaxFF-nn conventions): the dihedral by the law of
    cosines from ``x_i - x_j``, ``x_j - x_k``, ``x_k - x_l``, the cut bond orders with their tapers. The masked
    lanes take three orthogonal bonds, which keeps their derivatives finite."""
    ei = tl.load(ei_ptr + e, mask=m, other=0).to(tl.int64)
    ec = tl.load(ec_ptr + e, mask=m, other=0).to(tl.int64)
    el = tl.load(el_ptr + e, mask=m, other=0).to(tl.int64)
    aj = tl.load(aj_ptr + e, mask=m, other=0).to(tl.int64)
    ak = tl.load(ak_ptr + e, mask=m, other=0).to(tl.int64)
    t1 = tl.load(t_ptr + tl.load(ai_ptr + e, mask=m, other=0).to(tl.int64), mask=m, other=0)
    t2 = tl.load(t_ptr + aj, mask=m, other=0)
    t3 = tl.load(t_ptr + ak, mask=m, other=0)
    t4 = tl.load(t_ptr + tl.load(al_ptr + e, mask=m, other=0).to(tl.int64), mask=m, other=0)
    ax = -tl.load(bv_ptr + ei * 3, mask=m, other=1.0).to(DT)
    ay = -tl.load(bv_ptr + ei * 3 + 1, mask=m, other=0.0).to(DT)
    az = -tl.load(bv_ptr + ei * 3 + 2, mask=m, other=0.0).to(DT)
    cx = tl.load(bv_ptr + ec * 3, mask=m, other=0.0).to(DT)
    cy = tl.load(bv_ptr + ec * 3 + 1, mask=m, other=1.0).to(DT)
    cz = tl.load(bv_ptr + ec * 3 + 2, mask=m, other=0.0).to(DT)
    lx = tl.load(bv_ptr + el * 3, mask=m, other=0.0).to(DT)
    ly = tl.load(bv_ptr + el * 3 + 1, mask=m, other=0.0).to(DT)
    lz = tl.load(bv_ptr + el * 3 + 2, mask=m, other=1.0).to(DT)
    rij = tl.load(br_ptr + ei, mask=m, other=1.0).to(DT)
    rjk = tl.load(br_ptr + ec, mask=m, other=1.0).to(DT)
    rkl = tl.load(br_ptr + el, mask=m, other=1.0).to(DT)
    jlx = cx + lx
    jly = cy + ly
    jlz = cz + lz
    rjl = tl.sqrt(jlx * jlx + jly * jly + jlz * jlz)
    ilx = ax + jlx
    ily = ay + jly
    ilz = az + jlz
    ril = tl.sqrt(ilx * ilx + ily * ily + ilz * ilz)
    ikx = ax + cx
    iky = ay + cy
    ikz = az + cz
    rik = tl.sqrt(ikx * ikx + iky * iky + ikz * ikz)
    rij2 = rij * rij
    rjk2 = rjk * rjk
    rkl2 = rkl * rkl
    rjl2 = rjl * rjl
    ril2 = ril * ril
    rik2 = rik * rik
    c_ijk = (rij2 + rjk2 - rik2) / (2.0 * rij * rjk)
    s_ijk = tl.sqrt(tl.abs(1.0 - c_ijk * c_ijk) + 1.0e-12)
    c_jkl = (rjk2 + rkl2 - rjl2) / (2.0 * rjk * rkl)
    s_jkl = tl.sqrt(tl.abs(1.0 - c_jkl * c_jkl) + 1.0e-12)
    c_kjl = (rjk2 + rjl2 - rkl2) / (2.0 * rjk * rjl)
    s_kjl = tl.sqrt(tl.abs(1.0 - c_kjl * c_kjl) + 1.0e-12)
    fz = rij2 + rjl2 - ril2 - 2.0 * rij * rjl * c_ijk * c_kjl
    fm = rij * rjl * s_ijk * s_kjl
    tiny = (fm <= 0.000001) & (fm >= -0.000001)
    cw = tl.where(tiny, 0.0, 0.5 * fz / tl.where(tiny, 1.0, fm))
    cw = tl.where(cw > 0.9999999, 1.0, cw)
    cw = tl.where(cw < -0.999999, -1.0, cw)
    c2w = 2.0 * cw * cw - 1.0
    c3w = 4.0 * cw * cw * cw - 3.0 * cw
    tor2 = tl.load(gen_ptr).to(DT)
    tor3 = tl.load(gen_ptr + 1).to(DT)
    tor4 = tl.load(gen_ptr + 2).to(DT)
    cot2 = tl.load(gen_ptr + 3).to(DT)
    exptol = tl.load(gen_ptr + 4).to(DT)
    bij = tl.load(bo_ptr + ei, mask=m, other=1.0).to(DT)
    bjk = tl.load(bo_ptr + ec, mask=m, other=1.0).to(DT)
    bkl = tl.load(bo_ptr + el, mask=m, other=1.0).to(DT)
    fijkl = tl.load(fbo_ptr + ei, mask=m, other=0.0).to(DT) * tl.load(fbo_ptr + ec, mask=m, other=0.0).to(DT) \
        * tl.load(fbo_ptr + el, mask=m, other=0.0).to(DT)
    f10 = (1.0 - tl.exp(-tor2 * bij)) * (1.0 - tl.exp(-tor2 * bjk)) * (1.0 - tl.exp(-tor2 * bkl))
    delt = tl.load(dang_ptr + aj, mask=m, other=0.0).to(DT) + tl.load(dang_ptr + ak, mask=m, other=0.0).to(DT)
    exp3 = _gexp(-tor3 * delt)
    f11 = (2.0 + exp3) / (1.0 + exp3 + _gexp(tor4 * delt))
    tt = (((t1 * n_types + t2) * n_types + t3) * n_types + t4) * 5
    tor1 = tl.load(tor_ptr + tt, mask=m, other=0.0).to(DT)
    v1 = tl.load(tor_ptr + tt + 1, mask=m, other=0.0).to(DT)
    v2 = tl.load(tor_ptr + tt + 2, mask=m, other=0.0).to(DT)
    v3 = tl.load(tor_ptr + tt + 3, mask=m, other=0.0).to(DT)
    cot1 = tl.load(tor_ptr + tt + 4, mask=m, other=0.0).to(DT)
    q = 2.0 - tl.load(bpi_ptr + ec, mask=m, other=0.0).to(DT) - f11
    expv2 = tl.exp(tor1 * q * q)
    etor = fijkl * f10 * s_ijk * s_jkl * (0.5 * v1 * (1.0 + cw) + 0.5 * v2 * expv2 * (1.0 - c2w)
                                          + 0.5 * v3 * (1.0 + c3w))
    hi = bij - 1.5
    hj = bjk - 1.5
    hk = bkl - 1.5
    f12 = (tl.exp(-cot2 * hi * hi) - exptol) * (tl.exp(-cot2 * hj * hj) - exptol) \
        * (tl.exp(-cot2 * hk * hk) - exptol)
    efcon = fijkl * f12 * cot1 * (1.0 + (cw * cw - 1.0) * s_ijk * s_jkl)
    return aj, etor, efcon


@triton.jit
def _torsions_nn_forward(ei_ptr, ec_ptr, el_ptr, ai_ptr, aj_ptr, ak_ptr, al_ptr, bv_ptr, br_ptr, bo_ptr, bpi_ptr,
                         fbo_ptr, t_ptr, dang_ptr, tor_ptr, gen_ptr, out_ptr, n_torsions, n_types,
                         DT: tl.constexpr, BLOCK: tl.constexpr):
    """The torsion and four-body conjugation energies ``(N, 2)`` (ReaxFF-nn conventions) on the torsions' first
    central atoms."""
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n_torsions
    aj, etor, efcon = _torsion_nn(ei_ptr, ec_ptr, el_ptr, ai_ptr, aj_ptr, ak_ptr, al_ptr, bv_ptr, br_ptr, bo_ptr,
                                  bpi_ptr, fbo_ptr, t_ptr, dang_ptr, tor_ptr, gen_ptr, e, m, n_types, DT)
    tl.atomic_add(out_ptr + aj * 2, etor.to(tl.float64), mask=m, sem="relaxed")
    tl.atomic_add(out_ptr + aj * 2 + 1, efcon.to(tl.float64), mask=m, sem="relaxed")


# the per-(X, H, Z)-species table of the hydrogen-bond kernel: rohb (1 where none), on (1 / 0), Dehb, hb1, hb2
HB_COLUMNS = 5


@triton.jit
def _hbond(d_ptr, a_ptr, x_ptr, h_ptr, z_ptr, bv_ptr, br_ptr, bo_ptr, ev_ptr, er_ptr, t_ptr, hb_ptr, e, m, n_types,
           DT: tl.constexpr):
    """A hydrogen bond X-H...Z (LAMMPS conventions) from the X-H bond ``d`` and the H...Z pair edge ``a``."""
    d = tl.load(d_ptr + e, mask=m, other=0).to(tl.int64)
    a = tl.load(a_ptr + e, mask=m, other=0).to(tl.int64)
    hh = tl.load(h_ptr + e, mask=m, other=0).to(tl.int64)
    tx = tl.load(t_ptr + tl.load(x_ptr + e, mask=m, other=0).to(tl.int64), mask=m, other=0)
    th = tl.load(t_ptr + hh, mask=m, other=0)
    tz = tl.load(t_ptr + tl.load(z_ptr + e, mask=m, other=0).to(tl.int64), mask=m, other=0)
    ux = tl.load(bv_ptr + d * 3, mask=m, other=1.0).to(DT)
    uy = tl.load(bv_ptr + d * 3 + 1, mask=m, other=0.0).to(DT)
    uz = tl.load(bv_ptr + d * 3 + 2, mask=m, other=0.0).to(DT)
    vx = tl.load(ev_ptr + a * 3, mask=m, other=1.0).to(DT)
    vy = tl.load(ev_ptr + a * 3 + 1, mask=m, other=0.0).to(DT)
    vz = tl.load(ev_ptr + a * 3 + 2, mask=m, other=0.0).to(DT)
    rd = tl.load(br_ptr + d, mask=m, other=1.0).to(DT)
    r_hz = tl.load(er_ptr + a, mask=m, other=1.0).to(DT)
    cs = (ux * vx + uy * vy + uz * vz) / (rd * r_hz)
    sh = 0.5 - 0.5 * tl.minimum(tl.maximum(cs, -1.0), 1.0)
    sin4 = sh * sh
    tt = ((tx * n_types + th) * n_types + tz) * 5
    rohb = tl.load(hb_ptr + tt, mask=m, other=1.0).to(DT)
    on = tl.load(hb_ptr + tt + 1, mask=m, other=0.0)
    dehb = tl.load(hb_ptr + tt + 2, mask=m, other=0.0).to(DT)
    hb1 = tl.load(hb_ptr + tt + 3, mask=m, other=0.0).to(DT)
    hb2 = tl.load(hb_ptr + tt + 4, mask=m, other=0.0).to(DT)
    hbsum = rohb / r_hz + r_hz / rohb - 2.0
    bo = tl.load(bo_ptr + d, mask=m, other=0.0).to(DT)
    ehb = dehb * (1.0 - tl.exp(-hb1 * bo)) * tl.exp(-hb2 * hbsum) * sin4
    return hh, tl.where(on > 0.5, ehb, 0.0)


@triton.jit
def _hbonds_forward(d_ptr, a_ptr, x_ptr, h_ptr, z_ptr, bv_ptr, br_ptr, bo_ptr, ev_ptr, er_ptr, t_ptr, hb_ptr, out_ptr,
                    n_hbonds, n_types, DT: tl.constexpr, BLOCK: tl.constexpr):
    """The hydrogen-bond energies ``(N, 1)`` on the hydrogens."""
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n_hbonds
    hh, ehb = _hbond(d_ptr, a_ptr, x_ptr, h_ptr, z_ptr, bv_ptr, br_ptr, bo_ptr, ev_ptr, er_ptr, t_ptr, hb_ptr, e, m,
                     n_types, DT)
    tl.atomic_add(out_ptr + hh, ehb.to(tl.float64), mask=m, sem="relaxed")


# the per-(X, H, Z)-species table of the ReaxFF-nn hydrogen-bond kernel: rohb, Dehb, hb1, hb2
HB_NN_COLUMNS = 4


@triton.jit
def _hbond_nn(d_ptr, a_ptr, x_ptr, h_ptr, z_ptr, bv_ptr, br_ptr, bo_ptr, fhb_ptr, ev_ptr, er_ptr, t_ptr, hb_ptr,
              gen_ptr, e, m, n_types, DT: tl.constexpr):
    """A hydrogen bond X-H...Z (ReaxFF-nn conventions) from the X-H bond ``d`` and the H...Z pair edge ``a``: the
    published libraries' angle cosine (``r_XH``, not its square, in the numerator), the X...Z taper from the
    ``gen`` window ``(rmin, rmax, 1 / (rmax - rmin)^3)``."""
    d = tl.load(d_ptr + e, mask=m, other=0).to(tl.int64)
    a = tl.load(a_ptr + e, mask=m, other=0).to(tl.int64)
    hh = tl.load(h_ptr + e, mask=m, other=0).to(tl.int64)
    tx = tl.load(t_ptr + tl.load(x_ptr + e, mask=m, other=0).to(tl.int64), mask=m, other=0)
    th = tl.load(t_ptr + hh, mask=m, other=0)
    tz = tl.load(t_ptr + tl.load(z_ptr + e, mask=m, other=0).to(tl.int64), mask=m, other=0)
    ux = tl.load(bv_ptr + d * 3, mask=m, other=1.0).to(DT)
    uy = tl.load(bv_ptr + d * 3 + 1, mask=m, other=0.0).to(DT)
    uz = tl.load(bv_ptr + d * 3 + 2, mask=m, other=0.0).to(DT)
    vx = tl.load(ev_ptr + a * 3, mask=m, other=0.0).to(DT)
    vy = tl.load(ev_ptr + a * 3 + 1, mask=m, other=1.0).to(DT)
    vz = tl.load(ev_ptr + a * 3 + 2, mask=m, other=0.0).to(DT)
    rij = tl.load(br_ptr + d, mask=m, other=1.0).to(DT)
    rjk = tl.load(er_ptr + a, mask=m, other=1.0).to(DT)
    wx = ux + vx
    wy = uy + vy
    wz = uz + vz
    rik = tl.sqrt(wx * wx + wy * wy + wz * wz)
    cs = (rij + rjk * rjk - rik * rik) / (2.0 * rij * rjk)
    hbthe = 0.5 - 0.5 * cs
    rmin = tl.load(gen_ptr).to(DT)
    rmax = tl.load(gen_ptr + 1).to(DT)
    rterm = tl.load(gen_ptr + 2).to(DT)
    below = tl.where(rik < rmin, 1.0, 0.0)
    inside = (rik <= rmax) & (rik > rmin)
    x2 = tl.where(inside, rik, 0.0)
    one2 = tl.where(inside, 1.0, 0.0)
    rm = rmax * one2
    rd = rm - x2
    trm1 = rm + 2.0 * x2 - 3.0 * rmin * one2
    frhb = rterm * rd * rd * trm1 + below
    tt = ((tx * n_types + th) * n_types + tz) * 4
    rohb = tl.load(hb_ptr + tt, mask=m, other=1.0).to(DT)
    dehb = tl.load(hb_ptr + tt + 1, mask=m, other=0.0).to(DT)
    hb1 = tl.load(hb_ptr + tt + 2, mask=m, other=0.0).to(DT)
    hb2 = tl.load(hb_ptr + tt + 3, mask=m, other=0.0).to(DT)
    bo = tl.load(bo_ptr + d, mask=m, other=0.0).to(DT)
    fhb = tl.load(fhb_ptr + d, mask=m, other=0.0).to(DT)
    hbsum = rohb / rjk + rjk / rohb - 2.0
    ehb = fhb * frhb * dehb * (1.0 - tl.exp(-hb1 * bo)) * tl.exp(-hb2 * hbsum) * hbthe * hbthe
    return hh, ehb


@triton.jit
def _hbonds_nn_forward(d_ptr, a_ptr, x_ptr, h_ptr, z_ptr, bv_ptr, br_ptr, bo_ptr, fhb_ptr, ev_ptr, er_ptr, t_ptr,
                       hb_ptr, gen_ptr, out_ptr, n_hbonds, n_types, DT: tl.constexpr, BLOCK: tl.constexpr):
    """The hydrogen-bond energies ``(N, 1)`` (ReaxFF-nn conventions) on the hydrogens."""
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n_hbonds
    hh, ehb = _hbond_nn(d_ptr, a_ptr, x_ptr, h_ptr, z_ptr, bv_ptr, br_ptr, bo_ptr, fhb_ptr, ev_ptr, er_ptr, t_ptr,
                        hb_ptr, gen_ptr, e, m, n_types, DT)
    tl.atomic_add(out_ptr + hh, ehb.to(tl.float64), mask=m, sem="relaxed")


@triton.jit
def _eem_products(src_ptr, dst_ptr, g_ptr, x_ptr, y_ptr, n_edges, BLOCK: tl.constexpr):
    """``y[dst] += g x[src]`` over the edges for two right-hand sides ``(N, 2)``: the off-diagonal part of the
    EEM matrix-vector product of the conjugate-gradient solve, in one pass over the edges."""
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n_edges
    i = tl.load(src_ptr + e, mask=m, other=0).to(tl.int64)
    k = tl.load(dst_ptr + e, mask=m, other=0).to(tl.int64)
    g = tl.load(g_ptr + e, mask=m, other=0.0)
    tl.atomic_add(y_ptr + k * 2, g * tl.load(x_ptr + i * 2, mask=m, other=0.0), mask=m, sem="relaxed")
    tl.atomic_add(y_ptr + k * 2 + 1, g * tl.load(x_ptr + i * 2 + 1, mask=m, other=0.0), mask=m, sem="relaxed")


def eem_products(src: Tensor, dst: Tensor, g: Tensor):
    """The matrix-vector product ``x -> H x`` (``H[dst, src] += g``) of :func:`_eem_products` for ``(N, 2)``
    blocks."""
    src_c, dst_c, g_c = src.contiguous(), dst.contiguous(), g.contiguous()
    n = int(src.shape[0])

    def product(x: Tensor) -> Tensor:
        x = x.contiguous()
        y = torch.zeros_like(x)
        if n:
            _eem_products[(triton.cdiv(n, 512),)](src_c, dst_c, g_c, x, y, n, BLOCK=512, num_warps=4)
        return y
    return product


def nonbonded_table(rvdw: Tensor, alfa: Tensor, devdw: Tensor, gammaw: Tensor, gamma: Tensor, vdw1: Tensor,
                    rcore: Optional[Tensor] = None, acore: Optional[Tensor] = None,
                    ecore: Optional[Tensor] = None) -> Tensor:
    """The ``(S, S, NB_COLUMNS)`` table of :func:`_nonbonded_forward` from the per-species-pair parameters
    ``(S, S)`` each (differentiable in them)."""
    one = torch.ones_like(rvdw)
    if rcore is None:
        rcore, acore, ecore = one, torch.zeros_like(rvdw), torch.zeros_like(rvdw)
    else:
        on = rcore > 0.0
        rcore = torch.where(on, rcore, one)
        acore = torch.where(on, acore, torch.zeros_like(acore))
        ecore = torch.where(on, ecore, torch.zeros_like(ecore))
    return torch.stack([2.0 * rvdw, alfa, devdw, (1.0 / gammaw) ** vdw1, (1.0 / gamma) ** 3, rcore, acore, ecore],
                       -1)


__all__ = ["NB_COLUMNS", "BC_COLUMNS", "ANG_COLUMNS", "TOR_COLUMNS", "HB_COLUMNS", "nonbonded_table", "eem_products"]
