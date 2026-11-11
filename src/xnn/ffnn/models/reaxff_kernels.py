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

from .classical_kernels import _add_sorted, _go, _segments, _vec, _vec_grad  # noqa: F401  (device functions of the kernels)

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
def _nb_force_terms(src_ptr, dst_ptr, sh_ptr, pos_ptr, cell_ptr, bat_ptr, t_ptr, q_ptr, tab_ptr, sc_ptr, e, m0, n_types,
                    DT: tl.constexpr, HAS_CELL: tl.constexpr, SHIELD: tl.constexpr, WALL: tl.constexpr):
    """A pair of :func:`_nonbonded_forward` for its forces: its atoms, mask and vector, weight, taper and the
    taper's derivative in ``r``, the van der Waals and Coulomb energies before the taper with their derivatives
    in ``r``, and ``r``."""
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
    ke = tl.load(sc_ptr).to(DT)
    rc = tl.load(sc_ptr + 1).to(DT)
    p = tl.load(sc_ptr + 2).to(DT)
    tiny = tl.load(sc_ptr + 3).to(DT)
    third = tl.load(sc_ptr + 4).to(DT)
    # the pairs of a list with a skin beyond the cutoff are inactive (no load, no add)
    m = m & (r2 < rc * rc)
    w = tl.where(m, tl.where(own, 0.5, 1.0), 0.0).to(DT)
    r = tl.sqrt(r2 + tiny)
    tp = (tl.load(t_ptr + i, mask=m, other=0) * n_types + tl.load(t_ptr + k, mask=m, other=0)) * 8
    rv2 = tl.load(tab_ptr + tp, mask=m, other=1.0)
    alfa = tl.load(tab_ptr + tp + 1, mask=m, other=0.0)
    dv = tl.load(tab_ptr + tp + 2, mask=m, other=0.0)
    gw = tl.load(tab_ptr + tp + 3, mask=m, other=0.0)
    gm3 = tl.load(tab_ptr + tp + 4, mask=m, other=1.0)
    # the taper and its derivative in r
    u = r / rc
    u2 = u * u
    u4 = u2 * u2
    poly = -35.0 + u * (84.0 + u * (-70.0 + 20.0 * u))
    tap = 1.0 + u4 * poly
    dtap = (4.0 * u2 * u * poly + u4 * (84.0 + u * (-140.0 + 60.0 * u))) / rc
    # van der Waals: f13 = (r^p + gw)^(1/p) shielded (else r), ev = dv (ex^2 - 2 ex)
    if SHIELD:
        rp = _pow(r, p)
        sp = rp + gw
        f13 = _pow(sp, 1.0 / p)
        df13 = f13 * rp / (r * sp)
    else:
        f13 = r
        df13 = r * 0.0 + 1.0
    ex = tl.exp(0.5 * alfa * (1.0 - f13 / rv2))
    ev = dv * (ex * ex - 2.0 * ex)
    dev = dv * (2.0 * ex - 2.0) * ex * (-0.5 * alfa / rv2) * df13
    if WALL:
        rcore = tl.load(tab_ptr + tp + 5, mask=m, other=1.0)
        acore = tl.load(tab_ptr + tp + 6, mask=m, other=0.0)
        ecore = tl.load(tab_ptr + tp + 7, mask=m, other=0.0)
        core = ecore * tl.exp(acore * (1.0 - r / rcore))
        ev += core
        dev += -acore / rcore * core
    # Coulomb: ec = ke qi qk / (r^3 + gm3)^(1/3)
    qi = tl.load(q_ptr + i, mask=m, other=0.0)
    qk = tl.load(q_ptr + k, mask=m, other=0.0)
    cube = r * r * r + gm3
    rth = _pow(cube, third)
    ec = ke * qi * qk / rth
    dec = -ec * r * r / cube
    return i, k, m, x, y, z, w, tap, dtap, ev, dev, ec, dec, r


@triton.jit
def _nonbonded_forces(src_ptr, dst_ptr, sh_ptr, pos_ptr, cell_ptr, bat_ptr, t_ptr, q_ptr, tab_ptr, sc_ptr, g_ptr,
                      gpos_ptr, n_edges, n_types, DT: tl.constexpr, HAS_CELL: tl.constexpr, SHIELD: tl.constexpr,
                      WALL: tl.constexpr, BLOCK: tl.constexpr):
    """The position gradient (into ``gpos``, float64) of the pair energies of :func:`_nonbonded_forward` from
    the incoming gradients ``g`` ``(N, 2)`` of the per-atom halves, the charges and the table held fixed: each
    pair's ``dE/dr`` along its vector, in one pass."""
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    i, k, m, x, y, z, w, tap, dtap, ev, dev, ec, dec, r = _nb_force_terms(
        src_ptr, dst_ptr, sh_ptr, pos_ptr, cell_ptr, bat_ptr, t_ptr, q_ptr, tab_ptr, sc_ptr, e, e < n_edges, n_types,
        DT, HAS_CELL, SHIELD, WALL)
    # each atom holds half of each pair energy
    gv = 0.5 * (tl.load(g_ptr + i * 2, mask=m, other=0.0) + tl.load(g_ptr + k * 2, mask=m, other=0.0)).to(DT)
    gc = 0.5 * (tl.load(g_ptr + i * 2 + 1, mask=m, other=0.0) + tl.load(g_ptr + k * 2 + 1, mask=m, other=0.0)).to(DT)
    coef = w * (gv * (dtap * ev + tap * dev) + gc * (dtap * ec + tap * dec)) / r
    _vec_grad(gpos_ptr, gpos_ptr, bat_ptr, sh_ptr, i, k, e, m, coef * x, coef * y, coef * z, HAS_CELL, False)


@triton.jit
def _nonbonded_energy_forces(src_ptr, dst_ptr, sh_ptr, pos_ptr, cell_ptr, bat_ptr, t_ptr, q_ptr, tab_ptr, sc_ptr,
                             e_ptr, gpos_ptr, n_edges, n_types, DT: tl.constexpr, HAS_CELL: tl.constexpr,
                             SHIELD: tl.constexpr, WALL: tl.constexpr, BLOCK: tl.constexpr):
    """A step's pairs in one pass: the total van der Waals and Coulomb energies added into ``e`` ``(2,)`` and
    their position gradient into ``gpos`` (float64), the charges and the table held fixed."""
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    i, k, m, x, y, z, w, tap, dtap, ev, dev, ec, dec, r = _nb_force_terms(
        src_ptr, dst_ptr, sh_ptr, pos_ptr, cell_ptr, bat_ptr, t_ptr, q_ptr, tab_ptr, sc_ptr, e, e < n_edges, n_types,
        DT, HAS_CELL, SHIELD, WALL)
    tl.atomic_add(e_ptr, tl.sum(tl.where(m, (w * tap * ev).to(tl.float64), 0.0), 0), sem="relaxed")
    tl.atomic_add(e_ptr + 1, tl.sum(tl.where(m, (w * tap * ec).to(tl.float64), 0.0), 0), sem="relaxed")
    coef = w * ((dtap * ev + tap * dev) + (dtap * ec + tap * dec)) / r
    _vec_grad(gpos_ptr, gpos_ptr, bat_ptr, sh_ptr, i, k, e, m, coef * x, coef * y, coef * z, HAS_CELL, False)


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
                              act_ptr, bo0_ptr, bosi_ptr, bopi_ptr, bopp_ptr, e_ptr, n_bonds, n_types,
                              DT: tl.constexpr, ACT: tl.constexpr, BLOCK: tl.constexpr):
    """The corrected bond orders per bond (stored) and ``-e_si / 2`` per bond added to its first atom; ``ACT``
    only the bonds whose ``act`` flag is set (the others keep their outputs)."""
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n_bonds
    if ACT:
        # the stepper's candidate lists: an entry takes part where its flag is set
        m = m & (tl.load(act_ptr + e, mask=m, other=0) != 0)
    a, bo0, bosi, bopi, bopp, esi = _bond_corr(src_ptr, dst_ptr, bop_ptr, bpi_ptr, bpp_ptr, dp_ptr, t_ptr, spt_ptr,
                                               tab_ptr, gen_ptr, e, m, n_types, DT)
    tl.store(bo0_ptr + e, bo0, mask=m)
    tl.store(bosi_ptr + e, bosi, mask=m)
    tl.store(bopi_ptr + e, bopi, mask=m)
    tl.store(bopp_ptr + e, bopp, mask=m)
    tl.atomic_add(e_ptr + a, (-0.5 * esi).to(tl.float64), mask=m, sem="relaxed")


# the bond-order stage of a stepper (LAMMPS conventions, constant parameters): the per-species-pair table of the
# uncorrected bond orders (rosi, bo1, bo2, ropi, bo3, bo4, ropp, bo5, bo6 and the three channels switched on),
# the per-species table of the atom terms (val, vale, lighter than 21 amu, carbon, lp2, ovun2, ovun5), the
# per-species-pair table (ovun1 Desi, the triple-bond stabilization on) and the general parameters (lp1, ovun3,
# ovun4, ovun6, ovun7, ovun8, kc2, the C-C correction on, trip1, trip2, trip3, trip4)
BO_COLUMNS, AT_COLUMNS, AP_COLUMNS = 12, 7, 2


@triton.jit
def _channel(ch_ptr, tp, j: tl.constexpr, r, m, DT: tl.constexpr):
    """``exp(p1 (r / r0)^p2)`` of channel ``j`` (0 sigma, 1 pi, 2 double pi) of the pair table row ``tp``, zero
    where the channel is off."""
    on = tl.load(ch_ptr + tp + 9 + j, mask=m, other=0.0) > 0.5
    rad = tl.where(on, tl.load(ch_ptr + tp + 3 * j, mask=m, other=1.0).to(DT), 1.0)
    p1 = tl.load(ch_ptr + tp + 3 * j + 1, mask=m, other=0.0).to(DT)
    p2 = tl.load(ch_ptr + tp + 3 * j + 2, mask=m, other=1.0).to(DT)
    x = p1 * _pow(r / rad, p2)
    return tl.where(on, tl.exp(tl.where(on, x, 0.0)), 0.0)


@triton.jit
def _bond_orders_forward(src_ptr, dst_ptr, br_ptr, valid_ptr, t_ptr, ch_ptr, sc_ptr, bsi_ptr, bpi_ptr, bpp_ptr,
                         act_ptr, dp_ptr, n_bonds, n_types, DT: tl.constexpr, BLOCK: tl.constexpr):
    """The uncorrected bond orders of a stepper's bond candidates at the distances ``br``: sigma (less ``botol``),
    pi and double pi per bond (zero where a candidate does not take part: flagged off in ``valid``, beyond the
    bond cutoff or below ``botol``), the flags of those that do into ``act`` and their sum added to the first
    atom's uncorrected coordination ``dp``; ``sc`` (botol, the bond cutoff)."""
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n_bonds
    a = tl.load(dst_ptr + e, mask=m, other=0).to(tl.int64)
    b = tl.load(src_ptr + e, mask=m, other=0).to(tl.int64)
    ta = tl.load(t_ptr + a, mask=m, other=0)
    tb = tl.load(t_ptr + b, mask=m, other=0)
    r = tl.load(br_ptr + e, mask=m, other=1.0).to(DT)
    botol = tl.load(sc_ptr).to(DT)
    cut = tl.load(sc_ptr + 1).to(DT)
    tp = (ta * n_types + tb) * 12
    e1 = (1.0 + botol) * _channel(ch_ptr, tp, 0, r, m, DT)
    e2 = _channel(ch_ptr, tp, 1, r, m, DT)
    e3 = _channel(ch_ptr, tp, 2, r, m, DT)
    act = m & (tl.load(valid_ptr + e, mask=m, other=0) != 0) & (r <= cut) & (e1 + e2 + e3 >= botol)
    bsi = tl.where(act, e1 - botol, 0.0)
    bpi = tl.where(act, e2, 0.0)
    bpp = tl.where(act, e3, 0.0)
    tl.store(bsi_ptr + e, bsi, mask=m)
    tl.store(bpi_ptr + e, bpi, mask=m)
    tl.store(bpp_ptr + e, bpp, mask=m)
    tl.store(act_ptr + e, act.to(tl.int32), mask=m)
    tl.atomic_add(dp_ptr + a, bsi + bpi + bpp, mask=m, sem="relaxed")


@triton.jit
def _lone_pairs(delta, vale, val, lp1):
    """LAMMPS's lone pairs of an atom of coordination ``delta`` (from the truncated half electron excess) and
    their deviation from the optimum."""
    half = libdevice.trunc(0.5 * (delta - vale))
    vlpex = delta - vale - 2.0 * half
    nlp = tl.exp(-lp1 * (2.0 + vlpex) * (2.0 + vlpex)) - half
    return nlp, 0.5 * (vale - val) - nlp


@triton.jit
def _sigm(x):
    """The logistic sigmoid, finite (and its derivative) at any argument."""
    z = tl.exp(-tl.abs(x))
    return tl.where(x >= 0.0, 1.0 / (1.0 + z), z / (1.0 + z))


@triton.jit
def _bond_atom_sums_forward(src_ptr, dst_ptr, bo0_ptr, bpi_ptr, bpp_ptr, del_ptr, t_ptr, at_ptr, ap_ptr, gen_ptr,
                            so_ptr, dpil_ptr, e_ptr, n_bonds, n_types, DT: tl.constexpr, BLOCK: tl.constexpr):
    """The per-atom sums of the over- and under-coordination terms over a stepper's bonds (``ovun1 Desi BO`` into
    ``so``, the pi orders weighted by the second atom's deviation into ``dpil``) from the corrected bond orders and
    the coordinations ``del``, and the bonds' C-C over-coordination and triple-bond stabilization energies added
    to their first atoms (``e``)."""
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n_bonds
    a = tl.load(dst_ptr + e, mask=m, other=0).to(tl.int64)
    b = tl.load(src_ptr + e, mask=m, other=0).to(tl.int64)
    ta = tl.load(t_ptr + a, mask=m, other=0)
    tb = tl.load(t_ptr + b, mask=m, other=0)
    bo0 = tl.load(bo0_ptr + e, mask=m, other=0.0).to(DT)
    pis = tl.load(bpi_ptr + e, mask=m, other=0.0).to(DT) + tl.load(bpp_ptr + e, mask=m, other=0.0).to(DT)
    da = tl.load(del_ptr + a, mask=m, other=0.0).to(DT)
    db = tl.load(del_ptr + b, mask=m, other=0.0).to(DT)
    val_a = tl.load(at_ptr + ta * 7, mask=m, other=0.0).to(DT)
    val_b = tl.load(at_ptr + tb * 7, mask=m, other=0.0).to(DT)
    vale_b = tl.load(at_ptr + tb * 7 + 1, mask=m, other=0.0).to(DT)
    light_a = tl.load(at_ptr + ta * 7 + 2, mask=m, other=0.0).to(DT)
    light_b = tl.load(at_ptr + tb * 7 + 2, mask=m, other=0.0).to(DT)
    carbon = (tl.load(at_ptr + ta * 7 + 3, mask=m, other=0.0) > 0.5) & (tl.load(at_ptr + tb * 7 + 3, mask=m,
                                                                                other=0.0) > 0.5)
    tp = (ta * n_types + tb) * 2
    lp1 = tl.load(gen_ptr).to(DT)
    dva = da - val_a
    dvb = db - val_b
    _, dlp_b = _lone_pairs(db, vale_b, val_b, lp1)
    tl.atomic_add(so_ptr + a, tl.load(ap_ptr + tp, mask=m, other=0.0).to(DT) * bo0, mask=m, sem="relaxed")
    tl.atomic_add(dpil_ptr + a, (dvb - light_a * (light_b * dlp_b)) * pis, mask=m, sem="relaxed")
    # the C-C over-coordination correction
    kc2 = tl.load(gen_ptr + 6).to(DT)
    vov3 = bo0 - dva - (dva * dva) * (dva * dva) / 25.0  # 0.04 as a literal would be a float32 constant
    on_c2 = carbon & (tl.load(gen_ptr + 7) > 0.5) & (vov3 > 3.0)
    e_c2 = tl.where(on_c2, kc2 * (vov3 - 3.0) * (vov3 - 3.0), 0.0)
    # the terminal triple-bond stabilization, half on each atom of the bond
    trip1 = tl.load(gen_ptr + 8).to(DT)
    trip2 = tl.load(gen_ptr + 9).to(DT)
    trip3 = tl.load(gen_ptr + 10).to(DT)
    trip4 = tl.load(gen_ptr + 11).to(DT)
    on_t = (bo0 >= 1.0) & (tl.load(ap_ptr + tp + 1, mask=m, other=0.0) > 0.5)
    exphu = tl.exp(-trip2 * (bo0 - 2.5) * (bo0 - 2.5))
    exphua = _gexp(-trip4 * (da - bo0))
    exphub = _gexp(-trip4 * (db - bo0))
    exphuov = _gexp(trip3 * (dva + dvb))
    e_t = tl.where(on_t, trip1 * exphu / (1.0 + 25.0 * exphuov) * (exphua + exphub), 0.0)
    tl.atomic_add(e_ptr + a, (e_c2 + 0.5 * e_t).to(tl.float64), mask=m, sem="relaxed")


@triton.jit
def _atom_terms_forward(del_ptr, so_ptr, dpil_ptr, t_ptr, at_ptr, gen_ptr, nlp_ptr, e_ptr, n_atoms, DT: tl.constexpr,
                        BLOCK: tl.constexpr):
    """The lone-pair, over- and under-coordination energies per atom (into ``e``) and the lone pairs ``nlp`` from
    the coordinations ``del`` and the sums of :func:`_bond_atom_sums_forward`."""
    i = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = i < n_atoms
    t = tl.load(t_ptr + i, mask=m, other=0)
    delta = tl.load(del_ptr + i, mask=m, other=0.0).to(DT)
    so = tl.load(so_ptr + i, mask=m, other=0.0).to(DT)
    dpil = tl.load(dpil_ptr + i, mask=m, other=0.0).to(DT)
    val = tl.load(at_ptr + t * 7, mask=m, other=1.0).to(DT)
    vale = tl.load(at_ptr + t * 7 + 1, mask=m, other=0.0).to(DT)
    light = tl.load(at_ptr + t * 7 + 2, mask=m, other=0.0).to(DT)
    lp2 = tl.load(at_ptr + t * 7 + 4, mask=m, other=0.0).to(DT)
    ovun2 = tl.load(at_ptr + t * 7 + 5, mask=m, other=0.0).to(DT)
    ovun5 = tl.load(at_ptr + t * 7 + 6, mask=m, other=0.0).to(DT)
    lp1 = tl.load(gen_ptr).to(DT)
    ovun3 = tl.load(gen_ptr + 1).to(DT)
    ovun4 = tl.load(gen_ptr + 2).to(DT)
    ovun6 = tl.load(gen_ptr + 3).to(DT)
    ovun7 = tl.load(gen_ptr + 4).to(DT)
    ovun8 = tl.load(gen_ptr + 5).to(DT)
    nlp, delta_lp = _lone_pairs(delta, vale, val, lp1)
    e_lone = lp2 * delta_lp * _sigm(75.0 * delta_lp)
    lpcorr = light * (light * delta_lp) / (1.0 + ovun3 * _gexp(ovun4 * dpil))
    dlc = delta - val - lpcorr
    e_over = so * (1.0 / (dlc + val + 1.0e-8)) * dlc * _sigm(-ovun2 * dlc)
    eu2 = 1.0 / (1.0 + ovun7 * _gexp(ovun8 * dpil))
    e_under = -ovun5 * (1.0 - _gexp(ovun6 * dlc)) * _sigm(ovun2 * dlc) * eu2
    tl.store(nlp_ptr + i, nlp, mask=m)
    tl.atomic_add(e_ptr + i, (e_lone + e_over + e_under).to(tl.float64), mask=m, sem="relaxed")


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
                    f9_ptr, sig_ptr, del_ptr, ang_ptr, sp_ptr, gen_ptr, act_ptr, out_ptr, n_angles, n_types,
                    DT: tl.constexpr, NN: tl.constexpr, ACT: tl.constexpr, BLOCK: tl.constexpr):
    """The angle, penalty and three-body conjugation energies ``(N, 3)`` on the angles' centers (``ACT``: of
    the angles whose ``act`` flag is set)."""
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n_angles
    if ACT:
        # the stepper's candidate lists: an entry takes part where its flag is set
        m = m & (tl.load(act_ptr + e, mask=m, other=0) != 0)
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
                      dang_ptr, tor_ptr, gen_ptr, act_ptr, out_ptr, n_torsions, n_types, DT: tl.constexpr,
                      ACT: tl.constexpr, BLOCK: tl.constexpr, THB: tl.constexpr = 0.0):
    """The torsion and four-body conjugation energies ``(N, 2)`` on the torsions' first central atoms (``ACT``:
    of the torsions whose ``act`` flag is set; ``ACT`` 2 also those whose three bond orders and their product
    exceed ``THB`` and whose 1-4 vector is not zero)."""
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n_torsions
    if ACT:
        # the stepper's candidate lists: an entry takes part where its flag is set
        m = m & (tl.load(act_ptr + e, mask=m, other=0) != 0)
        if ACT == 2:
            ei = tl.load(ei_ptr + e, mask=m, other=0).to(tl.int64)
            ec = tl.load(ec_ptr + e, mask=m, other=0).to(tl.int64)
            el = tl.load(el_ptr + e, mask=m, other=0).to(tl.int64)
            bi = tl.load(bo_ptr + ei, mask=m, other=0.0)
            bc = tl.load(bo_ptr + ec, mask=m, other=0.0)
            bl = tl.load(bo_ptr + el, mask=m, other=0.0)
            vx = (tl.load(bv_ptr + ei * 3, mask=m, other=0.0) - tl.load(bv_ptr + ec * 3, mask=m, other=0.0)
                  - tl.load(bv_ptr + el * 3, mask=m, other=0.0))
            vy = (tl.load(bv_ptr + ei * 3 + 1, mask=m, other=0.0) - tl.load(bv_ptr + ec * 3 + 1, mask=m, other=0.0)
                  - tl.load(bv_ptr + el * 3 + 1, mask=m, other=0.0))
            vz = (tl.load(bv_ptr + ei * 3 + 2, mask=m, other=0.0) - tl.load(bv_ptr + ec * 3 + 2, mask=m, other=0.0)
                  - tl.load(bv_ptr + el * 3 + 2, mask=m, other=0.0))
            m = (m & (bi > THB) & (bc > THB) & (bl > THB) & (bi * bc * bl > THB)
                 & (vx * vx + vy * vy + vz * vz > 1.0e-10))
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
def _hbond(d_ptr, a_ptr, x_ptr, h_ptr, z_ptr, bv_ptr, br_ptr, bo_ptr, ev_ptr, er_ptr, t_ptr, hb_ptr, off_ptr, e, m,
           n_types, DT: tl.constexpr, POS: tl.constexpr):
    """A hydrogen bond X-H...Z (LAMMPS conventions) from the X-H bond ``d`` and the H...Z pair edge ``a`` (its
    vector and length in ``ev``, ``er``), or with ``POS`` from the positions ``ev`` and the hydrogen bond's
    Cartesian image offset ``off`` (``x_H - x_Z + off``, its length ``sqrt(r^2 + 1e-9)``)."""
    d = tl.load(d_ptr + e, mask=m, other=0).to(tl.int64)
    a = tl.load(a_ptr + e, mask=m, other=0).to(tl.int64)
    hh = tl.load(h_ptr + e, mask=m, other=0).to(tl.int64)
    tx = tl.load(t_ptr + tl.load(x_ptr + e, mask=m, other=0).to(tl.int64), mask=m, other=0)
    th = tl.load(t_ptr + hh, mask=m, other=0)
    tz = tl.load(t_ptr + tl.load(z_ptr + e, mask=m, other=0).to(tl.int64), mask=m, other=0)
    ux = tl.load(bv_ptr + d * 3, mask=m, other=1.0).to(DT)
    uy = tl.load(bv_ptr + d * 3 + 1, mask=m, other=0.0).to(DT)
    uz = tl.load(bv_ptr + d * 3 + 2, mask=m, other=0.0).to(DT)
    if POS:
        zz = tl.load(z_ptr + e, mask=m, other=0).to(tl.int64)
        vx = (tl.load(ev_ptr + hh * 3, mask=m, other=0.0) - tl.load(ev_ptr + zz * 3, mask=m, other=0.0)
              + tl.load(off_ptr + e * 3, mask=m, other=0.0)).to(DT)
        vy = (tl.load(ev_ptr + hh * 3 + 1, mask=m, other=0.0) - tl.load(ev_ptr + zz * 3 + 1, mask=m, other=0.0)
              + tl.load(off_ptr + e * 3 + 1, mask=m, other=0.0)).to(DT)
        vz = (tl.load(ev_ptr + hh * 3 + 2, mask=m, other=0.0) - tl.load(ev_ptr + zz * 3 + 2, mask=m, other=0.0)
              + tl.load(off_ptr + e * 3 + 2, mask=m, other=0.0)).to(DT)
        r_hz = tl.sqrt(vx * vx + vy * vy + vz * vz + 1.0e-9)
    else:
        vx = tl.load(ev_ptr + a * 3, mask=m, other=1.0).to(DT)
        vy = tl.load(ev_ptr + a * 3 + 1, mask=m, other=0.0).to(DT)
        vz = tl.load(ev_ptr + a * 3 + 2, mask=m, other=0.0).to(DT)
        r_hz = tl.load(er_ptr + a, mask=m, other=1.0).to(DT)
    rd = tl.load(br_ptr + d, mask=m, other=1.0).to(DT)
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
def _hbonds_forward(d_ptr, a_ptr, x_ptr, h_ptr, z_ptr, bv_ptr, br_ptr, bo_ptr, ev_ptr, er_ptr, t_ptr, hb_ptr, off_ptr,
                    act_ptr, out_ptr, n_hbonds, n_types, DT: tl.constexpr, ACT: tl.constexpr, POS: tl.constexpr,
                    BLOCK: tl.constexpr, HBT: tl.constexpr = 0.0, HBL2: tl.constexpr = 0.0,
                    SAFE: tl.constexpr = 0.0):
    """The hydrogen-bond energies ``(N, 1)`` on the hydrogens (``ACT``: of the hydrogen bonds whose ``act``
    flag is set; ``ACT`` 2 with ``POS`` also those whose X-H bond order is at least ``HBT`` and whose H...Z
    vector has ``|v|^2 + SAFE <= HBL2``)."""
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n_hbonds
    if ACT:
        # the stepper's candidate lists: an entry takes part where its flag is set
        m = m & (tl.load(act_ptr + e, mask=m, other=0) != 0)
        if ACT == 2:
            dd = tl.load(d_ptr + e, mask=m, other=0).to(tl.int64)
            ih = tl.load(h_ptr + e, mask=m, other=0).to(tl.int64)
            iz = tl.load(z_ptr + e, mask=m, other=0).to(tl.int64)
            vx = (tl.load(ev_ptr + ih * 3, mask=m, other=0.0) - tl.load(ev_ptr + iz * 3, mask=m, other=0.0)
                  + tl.load(off_ptr + e * 3, mask=m, other=0.0))
            vy = (tl.load(ev_ptr + ih * 3 + 1, mask=m, other=0.0) - tl.load(ev_ptr + iz * 3 + 1, mask=m, other=0.0)
                  + tl.load(off_ptr + e * 3 + 1, mask=m, other=0.0))
            vz = (tl.load(ev_ptr + ih * 3 + 2, mask=m, other=0.0) - tl.load(ev_ptr + iz * 3 + 2, mask=m, other=0.0)
                  + tl.load(off_ptr + e * 3 + 2, mask=m, other=0.0))
            m = (m & (tl.load(bo_ptr + dd, mask=m, other=0.0) >= HBT)
                 & ((vx * vx + vy * vy + vz * vz).to(DT) + SAFE <= HBL2))
    hh, ehb = _hbond(d_ptr, a_ptr, x_ptr, h_ptr, z_ptr, bv_ptr, br_ptr, bo_ptr, ev_ptr, er_ptr, t_ptr, hb_ptr,
                     off_ptr, e, m, n_types, DT, POS)
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


@triton.jit
def _eem_products_pairs(src_ptr, dst_ptr, g_ptr, x_ptr, y_ptr, n_pairs, BLOCK: tl.constexpr):
    """``y[dst] += g x[src]`` and ``y[src] += g x[dst]`` over the pairs (each once) for two right-hand sides
    ``(N, 2)``: the symmetric off-diagonal product of :func:`_eem_products` from a half list. A list sorted by
    ``src`` adds the first atoms' sums by segments (any order gives the same sums)."""
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n_pairs
    i = tl.load(src_ptr + e, mask=m, other=0).to(tl.int64)
    k = tl.load(dst_ptr + e, mask=m, other=0).to(tl.int64)
    g = tl.load(g_ptr + e, mask=m, other=0.0)
    xi0 = tl.load(x_ptr + i * 2, mask=m, other=0.0)
    xi1 = tl.load(x_ptr + i * 2 + 1, mask=m, other=0.0)
    xk0 = tl.load(x_ptr + k * 2, mask=m, other=0.0)
    xk1 = tl.load(x_ptr + k * 2 + 1, mask=m, other=0.0)
    tl.atomic_add(y_ptr + k * 2, g * xi0, mask=m, sem="relaxed")
    tl.atomic_add(y_ptr + k * 2 + 1, g * xi1, mask=m, sem="relaxed")
    i0, i1, same, tail, rest = _segments(src_ptr, i, m, n_pairs, BLOCK)
    _add_sorted(y_ptr, i0 * 2, i1 * 2, i * 2, g * xk0, same, tail, rest, TAIL=1)
    _add_sorted(y_ptr + 1, i0 * 2, i1 * 2, i * 2, g * xk1, same, tail, rest, TAIL=1)


@triton.jit
def _eem_rows(ptr_ptr, col_ptr, idx_ptr, g_ptr, x_ptr, y_ptr, DT: tl.constexpr, INDIRECT: tl.constexpr,
              BLOCK: tl.constexpr):
    """``y[r] = sum_e g_e x[col_e]`` over the entries ``ptr[r] <= e < ptr[r + 1]`` of row ``r`` (one program per
    row) for two right-hand sides ``(N, 2)``: the off-diagonal product of a compressed-row matrix, with no
    atomic adds; ``INDIRECT`` the values ``g[idx_e]`` (one per pair of a symmetric matrix)."""
    r = tl.program_id(0).to(tl.int64)
    lo = tl.load(ptr_ptr + r).to(tl.int64)
    hi = tl.load(ptr_ptr + r + 1).to(tl.int64)
    acc0 = tl.zeros([BLOCK], dtype=DT)
    acc1 = tl.zeros([BLOCK], dtype=DT)
    start = lo
    while start < hi:
        e = start + tl.arange(0, BLOCK)
        m = e < hi
        c = tl.load(col_ptr + e, mask=m, other=0).to(tl.int64)
        if INDIRECT:
            g = tl.load(g_ptr + tl.load(idx_ptr + e, mask=m, other=0).to(tl.int64), mask=m, other=0.0)
        else:
            g = tl.load(g_ptr + e, mask=m, other=0.0)
        acc0 += g * tl.load(x_ptr + c * 2, mask=m, other=0.0)
        acc1 += g * tl.load(x_ptr + c * 2 + 1, mask=m, other=0.0)
        start += BLOCK
    tl.store(y_ptr + r * 2, tl.sum(acc0, 0))
    tl.store(y_ptr + r * 2 + 1, tl.sum(acc1, 0))


@triton.jit
def _pair_geometry(i_ptr, k_ptr, sh_ptr, pos_ptr, cell_ptr, r_ptr, off_ptr, n, HAS_CELL: tl.constexpr,
                   BLOCK: tl.constexpr):
    """The distances ``r`` and Cartesian image offsets ``off = sh @ cell`` ``(n, 3)`` of the pairs ``(i, k)`` with
    integer shifts ``sh`` (one cell ``(3, 3)``)."""
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n
    i = tl.load(i_ptr + e, mask=m, other=0).to(tl.int64)
    k = tl.load(k_ptr + e, mask=m, other=0).to(tl.int64)
    r2 = tl.zeros([BLOCK], dtype=tl.float64)
    for c in tl.static_range(3):
        d = tl.load(pos_ptr + k * 3 + c, mask=m, other=0.0) - tl.load(pos_ptr + i * 3 + c, mask=m, other=0.0)
        if HAS_CELL:
            o = (tl.load(sh_ptr + e * 3, mask=m, other=0.0) * tl.load(cell_ptr + c)
                 + tl.load(sh_ptr + e * 3 + 1, mask=m, other=0.0) * tl.load(cell_ptr + 3 + c)
                 + tl.load(sh_ptr + e * 3 + 2, mask=m, other=0.0) * tl.load(cell_ptr + 6 + c))
            tl.store(off_ptr + e * 3 + c, o, mask=m)
            d = d + o
        r2 += (d * d).to(tl.float64)
    tl.store(r_ptr + e, tl.sqrt(r2), mask=m)


def pair_geometry(i: Tensor, k: Tensor, shifts: Optional[Tensor], pos: Tensor, cell: Optional[Tensor]) -> tuple:
    """The distances ``(E,)`` and Cartesian image offsets ``(E, 3)`` (zero without a cell) of a pair list in the
    positions' precision (:func:`_pair_geometry`)."""
    E = int(i.shape[0])
    r = torch.empty(E, dtype=pos.dtype, device=pos.device)
    off = torch.zeros(E, 3, dtype=pos.dtype, device=pos.device)
    has = cell is not None and shifts is not None
    sh = shifts.to(pos.dtype).contiguous() if has else pos
    c = cell.reshape(9).to(pos.dtype).contiguous() if has else pos
    if E:
        _go(_pair_geometry, (triton.cdiv(E, 1024),), i.contiguous(), k.contiguous(), sh, pos.detach().contiguous(),
            c, r, off, E, HAS_CELL=has, BLOCK=1024)
    return r, off


@triton.jit
def _cg_matvec(ptr_ptr, nbr_ptr, g_ptr, diag_ptr, p_ptr, ap_ptr, BLOCK: tl.constexpr):
    """``Ap += diag p + H p`` (``Ap`` zero before) for two right-hand sides, one program per row ``i`` of the half
    list (its partners ``k > i``, ``nbr[ptr[i]:ptr[i + 1]]``, values ``g`` per pair): the row's sum in registers,
    the partners' terms added to their rows."""
    i = tl.program_id(0).to(tl.int64)
    lo = tl.load(ptr_ptr + i)
    hi = tl.load(ptr_ptr + i + 1)
    p0 = tl.load(p_ptr + i * 2)
    p1 = tl.load(p_ptr + i * 2 + 1)
    for e0 in range(lo, hi, BLOCK):
        e = e0 + tl.arange(0, BLOCK)
        m = e < hi
        k = tl.load(nbr_ptr + e, mask=m, other=0).to(tl.int64)
        g = tl.load(g_ptr + e, mask=m, other=0.0)
        tl.atomic_add(ap_ptr + k * 2, g * p0, mask=m, sem="relaxed")
        tl.atomic_add(ap_ptr + k * 2 + 1, g * p1, mask=m, sem="relaxed")
        tl.atomic_add(ap_ptr + i * 2, tl.sum(tl.where(m, g * tl.load(p_ptr + k * 2, mask=m, other=0.0), 0.0), 0),
                      sem="relaxed")
        tl.atomic_add(ap_ptr + i * 2 + 1,
                      tl.sum(tl.where(m, g * tl.load(p_ptr + k * 2 + 1, mask=m, other=0.0), 0.0), 0), sem="relaxed")
    d = tl.load(diag_ptr + i)
    tl.atomic_add(ap_ptr + i * 2, d * p0, sem="relaxed")
    tl.atomic_add(ap_ptr + i * 2 + 1, d * p1, sem="relaxed")


@triton.jit
def _cg_dot(p_ptr, ap_ptr, dot_ptr, n, BLOCK: tl.constexpr):
    """``p . Ap`` per right-hand side, added into ``dot`` ``(2,)``."""
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n
    for c in tl.static_range(2):
        v = tl.load(p_ptr + e * 2 + c, mask=m, other=0.0) * tl.load(ap_ptr + e * 2 + c, mask=m, other=0.0)
        tl.atomic_add(dot_ptr + c, tl.sum(tl.where(m, v.to(tl.float64), 0.0), 0), sem="relaxed")


@triton.jit
def _cg_update(x_ptr, res_ptr, z_ptr, p_ptr, ap_ptr, inv_ptr, rz_ptr, dot_ptr, acc_ptr, n, BLOCK: tl.constexpr):
    """``x += alpha p``, ``res -= alpha Ap``, ``z = res / diag`` with ``alpha = rz / (p . Ap)``, adding
    ``res . z`` and ``res . res`` into ``acc`` ``(4,)``."""
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n
    inv = tl.load(inv_ptr + e, mask=m, other=0.0)
    for c in tl.static_range(2):
        rz = tl.load(rz_ptr + c)
        pap = tl.load(dot_ptr + c).to(rz.dtype)
        alpha = tl.where(pap != 0.0, rz / tl.where(pap != 0.0, pap, 1.0), 0.0)
        p = tl.load(p_ptr + e * 2 + c, mask=m, other=0.0)
        x = tl.load(x_ptr + e * 2 + c, mask=m, other=0.0)
        tl.store(x_ptr + e * 2 + c, x + alpha * p, mask=m)
        res = tl.load(res_ptr + e * 2 + c, mask=m, other=0.0) - alpha * tl.load(ap_ptr + e * 2 + c, mask=m, other=0.0)
        tl.store(res_ptr + e * 2 + c, res, mask=m)
        z = res * inv
        tl.store(z_ptr + e * 2 + c, z, mask=m)
        tl.atomic_add(acc_ptr + c, tl.sum(tl.where(m, (res * z).to(tl.float64), 0.0), 0), sem="relaxed")
        tl.atomic_add(acc_ptr + 2 + c, tl.sum(tl.where(m, (res * res).to(tl.float64), 0.0), 0), sem="relaxed")


@triton.jit
def _cg_direction(z_ptr, p_ptr, ap_ptr, sc_ptr, n, BLOCK: tl.constexpr):
    """``p = z + beta p`` with ``beta = rz_new / rz`` (``sc`` ``(4,)``: ``rz_new`` then ``rz``), and ``Ap`` zeroed
    for the next product."""
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n
    for c in tl.static_range(2):
        new = tl.load(sc_ptr + c)
        old = tl.load(sc_ptr + 2 + c)
        beta = tl.where(old != 0.0, new / tl.where(old != 0.0, old, 1.0), 0.0)
        p = tl.load(z_ptr + e * 2 + c, mask=m, other=0.0) + beta * tl.load(p_ptr + e * 2 + c, mask=m, other=0.0)
        tl.store(p_ptr + e * 2 + c, p, mask=m)
        tl.store(ap_ptr + e * 2 + c, p * 0.0, mask=m)


@triton.jit
def _hb_cross(dptr_ptr, dsel_ptr, aptr_ptr, asel_ptr, bsrc_ptr, did_ptr, asrc_ptr, aoff_ptr, at_ptr, n_ptr, d_ptr,
              x_ptr, h_ptr, z_ptr, off_ptr, WRITE: tl.constexpr, BLOCK: tl.constexpr):
    """The hydrogen-bond candidates of hydrogen ``h`` on the CPU: each of its X-H bonds (``dsel``, rows ``dptr``)
    with each of its H...Z pairs (``asel``, rows ``aptr``) whose Z is not X; their number into ``n`` or, with
    ``WRITE``, the bond, X, H, Z and the pair's offset from ``at[h]`` on."""
    h = tl.program_id(0).to(tl.int64)
    a0 = tl.load(aptr_ptr + h)
    a1 = tl.load(aptr_ptr + h + 1)
    w = tl.load(at_ptr + h)
    for j in range(tl.load(dptr_ptr + h), tl.load(dptr_ptr + h + 1)):
        dj = tl.load(dsel_ptr + j)
        d = tl.load(did_ptr + dj)
        x = tl.load(bsrc_ptr + d)
        for q in range(a0, a1):
            aq = tl.load(asel_ptr + q)
            z = tl.load(asrc_ptr + aq)
            if z != x:
                if WRITE:
                    tl.store(d_ptr + w, d)
                    tl.store(x_ptr + w, x)
                    tl.store(h_ptr + w, h)
                    tl.store(z_ptr + w, z)
                    for c in tl.static_range(3):
                        tl.store(off_ptr + 3 * w + c, tl.load(aoff_ptr + 3 * aq + c))
                w += 1
    if not WRITE:
        tl.store(n_ptr + h, w)


def hbond_cross(d_ids: Tensor, d_key: Tensor, b_src: Tensor, a_src: Tensor, a_key: Tensor, a_off: Tensor,
                n: int) -> tuple:
    """The hydrogen-bond candidates on the CPU as the grouped cross product of
    :func:`~xnn.ffnn.models.reaxff._group_cross` gives them (by hydrogen, its X-H bonds ``d_ids`` (hydrogen
    ``d_key``) times its H...Z pairs (hydrogen ``a_key``, Z ``a_src``, offsets ``a_off``), Z not X): the bonds,
    X, H, Z and the offsets ``(C, 3)``."""
    from xnn.common import cpu_kernels

    def rows(key):
        order = torch.argsort(key, stable=True)
        ptr = torch.zeros(n + 1, dtype=torch.int64)
        ptr[1:] = torch.cumsum(torch.bincount(key, minlength=n), 0)
        return ptr, order
    dptr, dsel = rows(d_key)
    aptr, asel = rows(a_key)
    zero = torch.zeros(n, dtype=torch.int64)
    a_off = a_off.contiguous()
    args = (dptr, dsel, aptr, asel, b_src.contiguous(), d_ids.contiguous(), a_src.contiguous(), a_off)
    cnt = torch.empty(n, dtype=torch.int64)
    cpu_kernels.launch(_hb_cross, (n,), *args, zero, cnt, zero, zero, zero, zero, a_off, WRITE=False, BLOCK=1)
    at = torch.cumsum(cnt, 0) - cnt
    C = int(cnt.sum())
    d, x, h, z = (torch.empty(C, dtype=torch.long) for _ in range(4))
    off = torch.empty(C, 3, dtype=a_off.dtype)
    if C:
        cpu_kernels.launch(_hb_cross, (n,), *args, at, cnt, d, x, h, z, off, WRITE=True, BLOCK=1)
    return d, x, h, z, off


@triton.jit
def _pair_cands(ptr_ptr, k_ptr, r_ptr, t_ptr, bt_ptr, fl_ptr, sc_ptr, at_ptr, nb_ptr, nf_ptr, nr_ptr, b_ptr, f_ptr,
                g_ptr, n_rows, n_types, WRITE: tl.constexpr, HB: tl.constexpr, BLOCK: tl.constexpr):
    """Row ``i`` of a CPU stepper's half list (pairs ``ptr[i]`` to ``ptr[i + 1]``, partners ``k``, distances
    ``r``): its bond candidates (within ``bt`` of the species pair, negative where the pair has no bond), and with
    ``HB`` its pairs of a hydrogen ``k`` and an acceptor ``i`` (forward) or the reverse within ``sc[0]``; their
    numbers into ``nb``, ``nf``, ``nr`` or, with ``WRITE``, their pair indices from the offsets ``at`` (3, N)."""
    i = tl.program_id(0).to(tl.int64)
    ti = tl.load(t_ptr + i)
    fi = tl.load(fl_ptr + ti)
    cut = tl.load(sc_ptr)
    wb = tl.load(at_ptr + i)
    wf = tl.load(at_ptr + n_rows + i)
    wr = tl.load(at_ptr + 2 * n_rows + i)
    for e in range(tl.load(ptr_ptr + i), tl.load(ptr_ptr + i + 1)):
        k = tl.load(k_ptr + e).to(tl.int64)
        tk = tl.load(t_ptr + k)
        r = tl.load(r_ptr + e)
        if r <= tl.load(bt_ptr + tk * n_types + ti):
            if WRITE:
                tl.store(b_ptr + wb, e)
            wb += 1
        if HB:
            if r <= cut:
                fk = tl.load(fl_ptr + tk)
                if (fk == 1) & (fi == 2):
                    if WRITE:
                        tl.store(f_ptr + wf, e)
                    wf += 1
                if (fi == 1) & (fk == 2):
                    if WRITE:
                        tl.store(g_ptr + wr, e)
                    wr += 1
    if not WRITE:
        tl.store(nb_ptr + i, wb)
        tl.store(nf_ptr + i, wf)
        tl.store(nr_ptr + i, wr)


def pair_candidates(ptr: Tensor, k: Tensor, r: Tensor, species: Tensor, bt: Tensor, flag: Optional[Tensor],
                    hb_cut: float) -> tuple:
    """A CPU stepper's candidates from its half list by rows (``ptr``, partners ``k``, distances ``r``): the
    bond candidates (within the per-species-pair distance ``bt`` ``(S, S)``, indexed by the partner's species
    first; negative for no bond), and with the hydrogen-bond flags ``flag`` ``(S,)`` (1 hydrogen, 2 acceptor) the
    pairs within ``hb_cut`` of a hydrogen partner and an acceptor first atom, then those of the reverse; pair
    indices in increasing order (as ``nonzero`` gives them)."""
    from xnn.common import cpu_kernels
    n = int(ptr.shape[0]) - 1
    S = int(bt.shape[0])
    hb = flag is not None
    fl = (flag if hb else torch.zeros(S, dtype=torch.long)).to(torch.int32).contiguous()
    sc = torch.tensor([float(hb_cut)], dtype=r.dtype)
    t = species.to(torch.int32).contiguous()
    args = (ptr.contiguous(), k.contiguous(), r.contiguous(), t, bt.to(r.dtype).contiguous(), fl, sc)
    counts = torch.zeros(3, n, dtype=torch.int64)
    none = torch.zeros(1, dtype=torch.int64)
    cpu_kernels.launch(_pair_cands, (n,), *args, torch.zeros(3 * n, dtype=torch.int64), counts[0], counts[1],
                       counts[2], none, none, none, n, S, WRITE=False, HB=hb, BLOCK=1)
    total = counts.sum(1)
    at = (torch.cumsum(counts, 1) - counts).contiguous()
    out = [torch.empty(int(c), dtype=torch.int64) for c in total.tolist()]
    if n:
        cpu_kernels.launch(_pair_cands, (n,), *args, at.reshape(-1), none, none, none, *out, n, S, WRITE=True, HB=hb,
                           BLOCK=1)
    return tuple(out)


class CGSolver:
    """Jacobi-preconditioned conjugate gradients for the two right-hand sides of the EEM system on the CPU, from
    the half list's matrix values (each pair once, its rows ``ptr``, ``nbr``): an iteration is two runs of prepared
    kernel launches (the product, its dot and the update in one parallel region; the new direction in another)
    with the scalars in small tensors."""

    def __init__(self, ptr: Tensor, nbr: Tensor, g: Tensor, diag: Tensor, b: Tensor):
        from xnn.common import cpu_kernels
        n = int(diag.shape[0])
        dtype = b.dtype
        self.n, self.b, self.diag = n, b, diag.contiguous()
        self.bnorm2 = (b * b).sum(0).tolist()
        self.x = torch.zeros(n, 2, dtype=dtype)
        self.res, self.z = torch.zeros_like(self.x), torch.zeros_like(self.x)
        self.p, self.ap = torch.zeros_like(self.x), torch.zeros_like(self.x)
        self.inv = (1.0 / diag).contiguous()
        self.rz = torch.zeros(2, dtype=dtype)
        self.dot = torch.zeros(2, dtype=torch.float64)
        self.acc = torch.zeros(4, dtype=torch.float64)
        self.sc = torch.zeros(4, dtype=dtype)
        ptr = ptr.to(torch.int64).contiguous()
        grid = (triton.cdiv(n, 512),)
        prep = cpu_kernels.prepare
        self.matvec = prep(_cg_matvec, (n,), ptr, nbr.contiguous(), g, self.diag, self.p, self.ap, BLOCK=1,
                           balance=ptr)
        dot = prep(_cg_dot, grid, self.p, self.ap, self.dot, n, BLOCK=512)
        update = prep(_cg_update, grid, self.x, self.res, self.z, self.p, self.ap, self.inv, self.rz, self.dot,
                      self.acc, n, BLOCK=512)
        self.step = cpu_kernels.Sequence([self.matvec, dot, update])
        self.direction = cpu_kernels.Sequence([prep(_cg_direction, grid, self.z, self.p, self.ap, self.sc, n,
                                                    BLOCK=512)])

    def solve(self, x0: Tensor, tol: float) -> tuple:
        """From ``x0`` until every relative residual is at most ``tol``: the solution, the relative residuals
        ``(2,)`` and the iterations."""
        self.x.copy_(x0)
        # the first residual from one product of x0
        self.p.copy_(x0)
        self.ap.zero_()
        self.matvec()
        torch.sub(self.b, self.ap, out=self.res)
        torch.mul(self.res, self.inv[:, None], out=self.z)
        self.p.copy_(self.z)
        self.ap.zero_()
        rz = (self.res * self.z).sum(0).tolist()
        r2 = (self.res * self.res).sum(0).tolist()
        self.rz.copy_(torch.tensor(rz, dtype=self.rz.dtype))
        bound = [tol * tol * b for b in self.bnorm2]
        it = 0
        while not (r2[0] <= bound[0] and r2[1] <= bound[1]):
            self.dot.zero_()
            self.acc.zero_()
            self.step()
            acc = self.acc.tolist()
            rz_new, r2 = acc[:2], acc[2:]
            self.sc.copy_(torch.tensor(rz_new + rz, dtype=self.sc.dtype))
            self.direction()
            self.rz.copy_(self.sc[:2])
            rz = rz_new
            it += 1
        res = torch.tensor([(r2[c] / self.bnorm2[c]) ** 0.5 for c in range(2)], dtype=self.b.dtype)
        return self.x, res, it


@triton.jit
def _eem_matrix(src_ptr, dst_ptr, off_ptr, valid_ptr, gm3_ptr, pos_ptr, sc_ptr, g_ptr, n_pairs, DT: tl.constexpr,
                BLOCK: tl.constexpr):
    """The off-diagonal coefficient ``ke Tap(r) / (r^3 + gm3)^(1/3)`` of each pair within the cutoff (0 beyond it
    and for invalid entries), ``r`` from the positions and the Cartesian image offset, ``r = sqrt(r^2 + tiny)``;
    ``sc`` the float64 (ke, cutoff, tiny, 1/3)."""
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n_pairs
    i = tl.load(src_ptr + e, mask=m, other=0).to(tl.int64)
    k = tl.load(dst_ptr + e, mask=m, other=0).to(tl.int64)
    x = (tl.load(pos_ptr + k * 3, mask=m, other=0.0) - tl.load(pos_ptr + i * 3, mask=m, other=0.0)
         + tl.load(off_ptr + e * 3, mask=m, other=0.0)).to(DT)
    y = (tl.load(pos_ptr + k * 3 + 1, mask=m, other=0.0) - tl.load(pos_ptr + i * 3 + 1, mask=m, other=0.0)
         + tl.load(off_ptr + e * 3 + 1, mask=m, other=0.0)).to(DT)
    z = (tl.load(pos_ptr + k * 3 + 2, mask=m, other=0.0) - tl.load(pos_ptr + i * 3 + 2, mask=m, other=0.0)
         + tl.load(off_ptr + e * 3 + 2, mask=m, other=0.0)).to(DT)
    ke = tl.load(sc_ptr).to(DT)
    rc = tl.load(sc_ptr + 1).to(DT)
    tiny = tl.load(sc_ptr + 2).to(DT)
    third = tl.load(sc_ptr + 3).to(DT)
    r2 = x * x + y * y + z * z + tiny
    r = tl.sqrt(r2)
    on = m & (tl.load(valid_ptr + e, mask=m, other=0) != 0) & (r2 < rc * rc + tiny)
    gm3 = tl.load(gm3_ptr + e, mask=m, other=1.0).to(DT)
    g = ke * _taper(r, rc) / _pow(r * r * r + gm3, third)
    tl.store(g_ptr + e, tl.where(on, g, 0.0), mask=m)


def eem_matrix(src: Tensor, dst: Tensor, off: Tensor, valid: Tensor, gm3: Tensor, pos: Tensor, out: Tensor,
               ke: float, cutoff: float, tiny: float) -> None:
    """The pair coefficients of :func:`_eem_matrix` into ``out`` (the model's precision)."""
    from .classical_kernels import _scalars
    n = int(src.shape[0])
    dt = tl.float64 if out.dtype == torch.float64 else tl.float32
    if n:
        _go(_eem_matrix, (triton.cdiv(n, 512),), src, dst, off, valid, gm3, pos.detach(), _scalars((float(ke),
            float(cutoff), float(tiny), 1.0 / 3.0), pos.device), out, n, DT=dt, BLOCK=512, num_warps=4)


def eem_rows(ptr: Tensor, col: Tensor, g: Tensor, idx: Optional[Tensor] = None):
    """The matrix-vector product ``x -> H x`` for ``(N, 2)`` blocks of the compressed-row off-diagonal part
    of the EEM matrix (row ``r`` holds the entries ``ptr[r]`` to ``ptr[r + 1]``, columns ``col``, values ``g``,
    or ``g[idx]`` with ``idx``) of :func:`_eem_rows`."""
    ptr_c, col_c, g_c = ptr.contiguous(), col.contiguous(), g.contiguous()
    idx_c = col_c if idx is None else idx.contiguous()
    n = int(ptr.shape[0]) - 1
    dt = tl.float64 if g.dtype == torch.float64 else tl.float32
    # a row's entries in chunks of a warp's width on a GPU, of a few lanes on a CPU (each lane one iteration)
    block = 128 if g.is_cuda else 8

    def product(x: Tensor) -> Tensor:
        x = x.contiguous()
        y = torch.empty_like(x)
        if n:
            _go(_eem_rows, (n,), ptr_c, col_c, idx_c, g_c, x, y, DT=dt, INDIRECT=idx is not None, BLOCK=block,
                num_warps=4)
        return y
    return product


def eem_products(src: Tensor, dst: Tensor, g: Tensor, pairs: bool = False):
    """The matrix-vector product ``x -> H x`` (``H[dst, src] += g``) of :func:`_eem_products` for ``(N, 2)``
    blocks; ``pairs`` the list holds each pair once and ``H`` is symmetric (:func:`_eem_products_pairs`)."""
    src_c, dst_c, g_c = src.contiguous(), dst.contiguous(), g.contiguous()
    n = int(src.shape[0])
    kernel = _eem_products_pairs if pairs else _eem_products

    def product(x: Tensor) -> Tensor:
        x = x.contiguous()
        y = torch.zeros_like(x)
        if n:
            _go(kernel, (triton.cdiv(n, 512),), src_c, dst_c, g_c, x, y, n, BLOCK=512, num_warps=4)
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
