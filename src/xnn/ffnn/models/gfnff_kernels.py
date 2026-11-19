"""The GFN-FF energy terms as fused kernels: every kernel adds the energy of its rows and their position
gradient, one row (or a block of pairs) per program. :class:`Plan` holds the launches of one energy-and-gradient
evaluation of a bound :class:`~xnn.ffnn.models.gfnff.GFNFF` model, on a GPU through Triton and on the CPU through
:mod:`xnn.common.cpu_kernels`."""
import itertools
import math

import numpy as np
import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra import libdevice

from xnn.common import cpu_kernels

#: Inexact constants of the kernels (float literals of a kernel are float32).
CONST = torch.tensor([0.1, (225.0 / 238.0) ** 2, (225.0 / 202.0) ** 2, 1.8, 0.8, 0.41, 0.06, 16.0 / 45.0,
                      1.0 / math.sqrt(math.pi), 4.4,
                      math.log(1.0 + math.exp(4.4)), 4.0 / 3.0, math.pi], dtype=torch.float64)
#: Atom count above which the C6 coefficients come from matrix products, and the largest system of charge
#: equations factorized in a kernel.
DENSE_FROM, SMALL_SOLVE = 256, 48
#: Atom count up to which the evaluation of a few atoms runs on one thread.
SERIAL_UP_TO = 10
#: Pairs per program of the pair kernels on a GPU.
PAIR_BLOCK = 128
#: Slots of the term energies in the energy buffer, in the order of ``gfnff.TERMS``.
BOND, ANGLE, TORSION, REPULSION, BATM, DISPERSION, HB, XB, ELECTROSTAT = range(9)


@triton.jit
def _delta(x_ptr, i, j):
    """``x_i - x_j`` and its length."""
    dx = tl.load(x_ptr + 3 * i) - tl.load(x_ptr + 3 * j)
    dy = tl.load(x_ptr + 3 * i + 1) - tl.load(x_ptr + 3 * j + 1)
    dz = tl.load(x_ptr + 3 * i + 2) - tl.load(x_ptr + 3 * j + 2)
    return dx, dy, dz, tl.sqrt(dx * dx + dy * dy + dz * dz)


@triton.jit
def _add(g_ptr, i, fx, fy, fz):
    tl.atomic_add(g_ptr + 3 * i, fx, sem="relaxed")
    tl.atomic_add(g_ptr + 3 * i + 1, fy, sem="relaxed")
    tl.atomic_add(g_ptr + 3 * i + 2, fz, sem="relaxed")


@triton.jit
def _push(g_ptr, i, j, dx, dy, dz, r, de):
    """The gradient of a term with ``dE/dr = de`` along ``x_i - x_j`` of length ``r``."""
    s = de / r
    _add(g_ptr, i, s * dx, s * dy, s * dz)
    _add(g_ptr, j, -s * dx, -s * dy, -s * dz)


@triton.jit
def _quartic(r, rad, c):
    """The damping ``1 / (1 + c (r / rad)^4)`` and its derivative in ``r``."""
    y = r / rad
    y2 = y * y
    f = 1.0 / (1.0 + c * y2 * y2)
    return f, -4.0 * c * y2 * y / rad * f * f


@triton.jit
def _dihedral(x_ptr, p0, p1, p2, p3):
    """The dihedral angle of the chain ``p0-p1-p2-p3`` (the sign of ``atan2((n1 x b1) . n2, n1 . n2)``), its
    gradients at ``p0`` and ``p3`` and the two projection ratios that give those at ``p1`` and ``p2``; ``ok`` is
    false when three of the atoms are collinear."""
    ax, ay, az, _ = _delta(x_ptr, p1, p0)
    bx, by, bz, rb = _delta(x_ptr, p2, p1)
    cx, cy, cz, _ = _delta(x_ptr, p3, p2)
    ux = ay * bz - az * by
    uy = az * bx - ax * bz
    uz = ax * by - ay * bx
    vx = by * cz - bz * cy
    vy = bz * cx - bx * cz
    vz = bx * cy - by * cx
    uu = ux * ux + uy * uy + uz * uz
    vv = vx * vx + vy * vy + vz * vz
    ok = (uu > 0.0) & (vv > 0.0)
    su = tl.where(ok, rb / uu, 0.0)
    sv = tl.where(ok, rb / vv, 0.0)
    sin = -rb * (ax * vx + ay * vy + az * vz)
    cos = ux * vx + uy * vy + uz * vz
    phi = libdevice.atan2(sin, cos)
    fa = (ax * bx + ay * by + az * bz) / (rb * rb)
    fc = (cx * bx + cy * by + cz * bz) / (rb * rb)
    return phi, ok, su * ux, su * uy, su * uz, -sv * vx, -sv * vy, -sv * vz, fa, fc


@triton.jit
def _add_dihedral(g_ptr, p0, p1, p2, p3, de, g0x, g0y, g0z, g3x, g3y, g3z, fa, fc):
    """Add ``de`` times the gradient of a dihedral angle given by :func:`_dihedral`."""
    g1x = -g0x - fa * g0x + fc * g3x
    g1y = -g0y - fa * g0y + fc * g3y
    g1z = -g0z - fa * g0z + fc * g3z
    _add(g_ptr, p0, de * g0x, de * g0y, de * g0z)
    _add(g_ptr, p1, de * g1x, de * g1y, de * g1z)
    _add(g_ptr, p2, -de * (g0x + g1x + g3x), -de * (g0y + g1y + g3y), -de * (g0z + g1z + g3z))
    _add(g_ptr, p3, de * g3x, de * g3y, de * g3z)


@triton.jit
def _vdelta(x_ptr, i, j, m):
    """``x_i - x_j`` and its length for the lanes in ``m`` (masked lanes read atom 0)."""
    dx = tl.load(x_ptr + 3 * i, mask=m, other=0.0) - tl.load(x_ptr + 3 * j, mask=m, other=0.0)
    dy = tl.load(x_ptr + 3 * i + 1, mask=m, other=0.0) - tl.load(x_ptr + 3 * j + 1, mask=m, other=0.0)
    dz = tl.load(x_ptr + 3 * i + 2, mask=m, other=0.0) - tl.load(x_ptr + 3 * j + 2, mask=m, other=0.0)
    return dx, dy, dz, tl.sqrt(dx * dx + dy * dy + dz * dz)


@triton.jit
def _vpush(g_ptr, i, j, dx, dy, dz, r, de, m):
    """:func:`_push` for the lanes in ``m``."""
    s = tl.where(m, de / tl.where(m, r, 1.0), 0.0)
    tl.atomic_add(g_ptr + 3 * i, s * dx, mask=m, sem="relaxed")
    tl.atomic_add(g_ptr + 3 * i + 1, s * dy, mask=m, sem="relaxed")
    tl.atomic_add(g_ptr + 3 * i + 2, s * dz, mask=m, sem="relaxed")
    tl.atomic_add(g_ptr + 3 * j, -s * dx, mask=m, sem="relaxed")
    tl.atomic_add(g_ptr + 3 * j + 1, -s * dy, mask=m, sem="relaxed")
    tl.atomic_add(g_ptr + 3 * j + 2, -s * dz, mask=m, sem="relaxed")


@triton.jit
def _pair_lanes(ij_ptr, npair, BLOCK: tl.constexpr):
    p = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = p < npair
    i = tl.load(ij_ptr + 2 * p, mask=m, other=0)
    j = tl.load(ij_ptr + 2 * p + 1, mask=m, other=1)
    return p, m, i, j


@triton.jit
def _pair_first(x_ptr, ij_ptr, rcov_ptr, gam_ptr, cnt_ptr, mat_ptr, c_ptr, n, dim, npair, BLOCK: tl.constexpr):
    """Pairs of a block of ``BLOCK``: their shares of the coordination counts and their Coulomb matrix elements."""
    p, m, i, j = _pair_lanes(ij_ptr, npair, BLOCK)
    inv_sqrt_pi = tl.load(c_ptr + 8)  # noqa: F841, read by the erf tangent
    dx, dy, dz, r = _vdelta(x_ptr, i, j, m)
    r = tl.where(m, r, 1.0)
    rc = tl.load(c_ptr + 11) * (tl.load(rcov_ptr + i, mask=m, other=1.0) + tl.load(rcov_ptr + j, mask=m, other=1.0))
    near = m & (r < 2.0 * rc)
    v = 0.5 * (1.0 + libdevice.erf(-7.5 * (r / rc - 1.0)))
    tl.atomic_add(cnt_ptr + i, v, mask=near, sem="relaxed")
    tl.atomic_add(cnt_ptr + j, v, mask=near, sem="relaxed")
    a = libdevice.erf(tl.load(gam_ptr + i * n + j, mask=m, other=0.0) * r) / r
    tl.store(mat_ptr + i * dim + j, a, mask=m)
    tl.store(mat_ptr + j * dim + i, a, mask=m)


@triton.jit
def _atom_terms(cnt_ptr, lcn_ptr, dl_ptr, rhs_ptr, dxl_ptr, x0_ptr, kap_ptr, ref_ptr, nref_ptr, al_ptr, sw_ptr, s_ptr,
                ds_ptr, c_ptr, n, NREF: tl.constexpr, NFREQ: tl.constexpr, BLOCK: tl.constexpr):
    """Atom ``i``: the logarithmic coordination number and its derivative in the count, the right-hand side of the
    charge equations with its derivative, and the weighted reference polarizabilities with their derivatives."""
    i = tl.program_id(0).to(tl.int64)
    ex = tl.exp(tl.load(c_ptr + 9) - tl.load(cnt_ptr + i))
    lcn = tl.load(c_ptr + 10) - tl.log(1.0 + ex)
    tl.store(lcn_ptr + i, lcn)
    tl.store(dl_ptr + i, ex / (1.0 + ex))
    kap = tl.load(kap_ptr + i)
    sq = tl.where(lcn > 0.0, tl.sqrt(tl.where(lcn > 0.0, lcn, 1.0)), 0.0)
    tl.store(rhs_ptr + i, tl.load(x0_ptr + i) + kap * sq)
    tl.store(dxl_ptr + i, tl.where(sq > 0.0, 0.5 * kap / sq, 0.0))
    nr = tl.load(nref_ptr + i)
    norm = lcn * 0.0
    dnorm = lcn * 0.0
    for r in tl.static_range(NREF):
        d = lcn - tl.load(ref_ptr + i * NREF + r)
        w = tl.where(nr > r, tl.exp(-4.0 * d * d), 0.0)
        norm += w
        dnorm += -8.0 * d * w
    for k in tl.static_range(NFREQ):
        acc = lcn * 0.0
        dacc = lcn * 0.0
        for r in tl.static_range(NREF):
            d = lcn - tl.load(ref_ptr + i * NREF + r)
            w = tl.where(nr > r, tl.exp(-4.0 * d * d), 0.0)
            a = tl.load(al_ptr + (k * NREF + r) * n + i)
            acc += w * a
            dacc += -8.0 * d * w * a
        alpha = acc / norm
        sw = tl.load(sw_ptr + k)
        tl.store(s_ptr + i * NFREQ + k, alpha * sw)
        tl.store(ds_ptr + i * NFREQ + k, (dacc - alpha * dnorm) / norm * sw)


@triton.jit
def _pair_second(x_ptr, ij_ptr, rc_ptr, rm_ptr, rcut_ptr, zz_ptr, dq_ptr, r06_ptr, r08_ptr, c6_ptr, dc_ptr, gam_ptr,
                 q_ptr, g_ptr, glog_ptr, e_ptr, c_ptr, n, dcut, npair, NFREQ: tl.constexpr, BLOCK: tl.constexpr):
    """Pairs of a block: repulsion, dispersion and the position gradient of the electrostatic energy. With ``NFREQ``
    the C6 coefficient and its derivatives are summed here from the weighted polarizabilities in ``c6`` and ``dc``
    ``(n, NFREQ)``; without, they are read from the ``(n, n)`` tables."""
    p, m, i, j = _pair_lanes(ij_ptr, npair, BLOCK)
    dx, dy, dz, r = _vdelta(x_ptr, i, j, m)
    r = tl.where(m, r, 1.0)
    k = i * n + j
    gam = tl.load(gam_ptr + k, mask=m, other=0.0)
    inv_sqrt_pi = tl.load(c_ptr + 8)
    de = tl.load(q_ptr + i, mask=m, other=0.0) * tl.load(q_ptr + j, mask=m, other=0.0) * (
        2.0 * gam * inv_sqrt_pi * tl.exp(-gam * gam * r * r) / r - libdevice.erf(gam * r) / (r * r))
    mr = m & (r < tl.load(rcut_ptr + p, mask=m, other=0.0))
    mm = tl.load(rm_ptr + p, mask=mr, other=0.0)
    sr = tl.sqrt(r)
    en = tl.where(mr, tl.load(rc_ptr + p, mask=mr, other=0.0) / r * tl.exp(-mm * r * sr), 0.0)
    tl.atomic_add(e_ptr + 3, tl.sum(en, 0), sem="relaxed")
    de += -en * (1.0 / r + 1.5 * mm * sr)
    md = m & (r < dcut)
    r2 = r * r
    r6 = r2 * r2 * r2
    t6 = 1.0 / (r6 + tl.load(r06_ptr + k, mask=md, other=1.0))
    t8 = 1.0 / (r6 * r2 + tl.load(r08_ptr + k, mask=md, other=1.0))
    q2 = 2.0 * tl.load(dq_ptr + k, mask=md, other=0.0)
    f = t6 + q2 * t8
    zc = tl.load(zz_ptr + k, mask=md, other=0.0)
    c6 = r * 0.0
    di = r * 0.0
    dj = r * 0.0
    if NFREQ > 0:
        for w in tl.static_range(NFREQ):
            si = tl.load(c6_ptr + i * NFREQ + w, mask=md, other=0.0)
            sj = tl.load(c6_ptr + j * NFREQ + w, mask=md, other=0.0)
            c6 += si * sj
            di += tl.load(dc_ptr + i * NFREQ + w, mask=md, other=0.0) * sj
            dj += tl.load(dc_ptr + j * NFREQ + w, mask=md, other=0.0) * si
    else:
        c6 = tl.load(c6_ptr + k, mask=md, other=0.0)
        di = tl.load(dc_ptr + k, mask=md, other=0.0)
        dj = tl.load(dc_ptr + j * n + i, mask=md, other=0.0)
    tl.atomic_add(e_ptr + 5, tl.sum(tl.where(md, -zc * c6 * f, 0.0), 0), sem="relaxed")
    de += tl.where(md, zc * c6 * (6.0 * r6 / r * t6 * t6 + 8.0 * q2 * r6 * r * t8 * t8), 0.0)
    tl.atomic_add(glog_ptr + i, -zc * f * di, mask=md, sem="relaxed")
    tl.atomic_add(glog_ptr + j, -zc * f * dj, mask=md, sem="relaxed")
    _vpush(g_ptr, i, j, dx, dy, dz, r, de, m)


@triton.jit
def _count_gradient(x_ptr, ij_ptr, rcov_ptr, dl_ptr, glog_ptr, g_ptr, c_ptr, npair, BLOCK: tl.constexpr):
    """Pairs of a block: the gradient that reaches the positions through the coordination numbers."""
    p, m, i, j = _pair_lanes(ij_ptr, npair, BLOCK)
    dx, dy, dz, r = _vdelta(x_ptr, i, j, m)
    r = tl.where(m, r, 1.0)
    rc = tl.load(c_ptr + 11) * (tl.load(rcov_ptr + i, mask=m, other=1.0) + tl.load(rcov_ptr + j, mask=m, other=1.0))
    near = m & (r < 2.0 * rc)
    u = 7.5 * (r / rc - 1.0)
    w = tl.load(glog_ptr + i, mask=near, other=0.0) * tl.load(dl_ptr + i, mask=near, other=0.0) \
        + tl.load(glog_ptr + j, mask=near, other=0.0) * tl.load(dl_ptr + j, mask=near, other=0.0)
    _vpush(g_ptr, i, j, dx, dy, dz, r, -7.5 / rc * tl.load(c_ptr + 8) * tl.exp(-u * u) * w, near)


@triton.jit
def _bonds(x_ptr, ij_ptr, par_ptr, lcn_ptr, accp_ptr, acc_ptr, accd_ptr, g_ptr, glog_ptr, e_ptr, c_ptr,
           BLOCK: tl.constexpr):
    """Bond ``p`` with the parameters (pair factor, radius sum, the two coordination slopes, shift, k, alpha) in
    seven columns and its hydrogen-bond acceptors in rows ``accp[p]`` to ``accp[p + 1]`` of ``acc`` (H, acceptor)."""
    p = tl.program_id(0).to(tl.int64)
    i = tl.load(ij_ptr + 2 * p)
    j = tl.load(ij_ptr + 2 * p + 1)
    dx, dy, dz, r = _delta(x_ptr, i, j)
    fp = tl.load(par_ptr + 7 * p)
    ci = tl.load(par_ptr + 7 * p + 2)
    cj = tl.load(par_ptr + 7 * p + 3)
    r0 = fp * (tl.load(par_ptr + 7 * p + 1) + ci * tl.load(lcn_ptr + i) + cj * tl.load(lcn_ptr + j)
               + tl.load(par_ptr + 7 * p + 4))
    kk = tl.load(par_ptr + 7 * p + 5)
    al = tl.load(par_ptr + 7 * p + 6)
    lo = tl.load(accp_ptr + p)
    hi = tl.load(accp_ptr + p + 1)
    inv_sqrt_pi = tl.load(c_ptr + 8)  # noqa: F841, read by the erf tangent
    mm = r * 0.0
    for t in range(lo, hi):
        _, _, _, rh = _delta(x_ptr, tl.load(acc_ptr + 2 * t), tl.load(acc_ptr + 2 * t + 1))
        mm += 0.5 * (1.0 + libdevice.erf(-27.5 * (rh / tl.load(accd_ptr + t) - 1.0)))
    tenth = tl.load(c_ptr)
    u = r - r0
    ae = al * (1.0 - tenth * mm)
    en = kk * tl.exp(-ae * u * u)
    tl.atomic_add(e_ptr, en, sem="relaxed")
    de = 2.0 * ae * u * en
    _push(g_ptr, i, j, dx, dy, dz, r, -de)
    tl.atomic_add(glog_ptr + i, de * fp * ci, sem="relaxed")
    tl.atomic_add(glog_ptr + j, de * fp * cj, sem="relaxed")
    dm = en * tenth * al * u * u
    for t in range(lo, hi):
        h = tl.load(acc_ptr + 2 * t)
        b = tl.load(acc_ptr + 2 * t + 1)
        hx, hy, hz, rh = _delta(x_ptr, h, b)
        dd = tl.load(accd_ptr + t)
        w = 27.5 * (rh / dd - 1.0)
        _push(g_ptr, h, b, hx, hy, hz, rh, -27.5 / dd * tl.load(c_ptr + 8) * tl.exp(-w * w) * dm)


@triton.jit
def _angles(x_ptr, idx_ptr, par_ptr, g_ptr, e_ptr, c_ptr, BLOCK: tl.constexpr):
    """Angle ``p`` (centre, a, b) with the parameters (cos t0, linear flag, k, damping radius of each arm)."""
    p = tl.program_id(0).to(tl.int64)
    c = tl.load(idx_ptr + 3 * p)
    a = tl.load(idx_ptr + 3 * p + 1)
    b = tl.load(idx_ptr + 3 * p + 2)
    ux, uy, uz, ru = _delta(x_ptr, a, c)
    vx, vy, vz, rv = _delta(x_ptr, b, c)
    cs = (ux * vx + uy * vy + uz * vz) / (ru * rv)
    kk = tl.load(par_ptr + 5 * p + 2)
    cd = tl.load(c_ptr + 1)
    fa, dfa = _quartic(ru, tl.load(par_ptr + 5 * p + 3), cd)
    fb, dfb = _quartic(rv, tl.load(par_ptr + 5 * p + 4), cd)
    ea = cs - tl.load(par_ptr + 5 * p)
    dea = 2.0 * ea
    ea = ea * ea
    if tl.load(par_ptr + 5 * p + 1) > 0.5:
        # pi minus the angle from the cross and dot products, well conditioned near 180 degrees
        wx = uy * vz - uz * vy
        wy = uz * vx - ux * vz
        wz = ux * vy - uy * vx
        sw = tl.sqrt(wx * wx + wy * wy + wz * wz)
        th = libdevice.atan2(sw, -(ux * vx + uy * vy + uz * vz))
        sn = sw / (ru * rv)
        ea = th * th
        steep = sn > 0.0
        dea = tl.where(steep, 2.0 * th / tl.where(steep, sn, 1.0), 2.0)
    en = kk * ea * fa * fb
    tl.atomic_add(e_ptr + 1, en, sem="relaxed")
    w = kk * fa * fb * dea
    iuv = 1.0 / (ru * rv)
    su = kk * ea * fb * dfa / ru - w * cs / (ru * ru)
    sv = kk * ea * fa * dfb / rv - w * cs / (rv * rv)
    gux = w * iuv * vx + su * ux
    guy = w * iuv * vy + su * uy
    guz = w * iuv * vz + su * uz
    gvx = w * iuv * ux + sv * vx
    gvy = w * iuv * uy + sv * vy
    gvz = w * iuv * uz + sv * vz
    _add(g_ptr, a, gux, guy, guz)
    _add(g_ptr, b, gvx, gvy, gvz)
    _add(g_ptr, c, -gux - gvx, -guy - gvy, -guz - gvz)


@triton.jit
def _torsions(x_ptr, idx_ptr, par_ptr, g_ptr, e_ptr, c_ptr, BLOCK: tl.constexpr):
    """Torsion ``p`` over the chain (idx0, idx2, idx1, idx3) with the parameters (n, phi0, k, three damping
    radii)."""
    p = tl.program_id(0).to(tl.int64)
    a = tl.load(idx_ptr + 4 * p)
    b = tl.load(idx_ptr + 4 * p + 1)
    c = tl.load(idx_ptr + 4 * p + 2)
    l = tl.load(idx_ptr + 4 * p + 3)
    phi, ok, g0x, g0y, g0z, g3x, g3y, g3z, fa, fc = _dihedral(x_ptr, a, c, b, l)
    x1, y1, z1, r1 = _delta(x_ptr, a, c)
    x2, y2, z2, r2 = _delta(x_ptr, c, b)
    x3, y3, z3, r3 = _delta(x_ptr, b, l)
    cd = tl.load(c_ptr + 2)
    f1, d1 = _quartic(r1, tl.load(par_ptr + 6 * p + 3), cd)
    f2, d2 = _quartic(r2, tl.load(par_ptr + 6 * p + 4), cd)
    f3, d3 = _quartic(r3, tl.load(par_ptr + 6 * p + 5), cd)
    nn = tl.load(par_ptr + 6 * p)
    arg = nn * (phi - tl.load(par_ptr + 6 * p + 1)) + tl.load(c_ptr + 12)
    kk = tl.load(par_ptr + 6 * p + 2)
    ang = kk * (1.0 + tl.cos(arg))
    tl.atomic_add(e_ptr + 2, ang * f1 * f2 * f3, sem="relaxed")
    _add_dihedral(g_ptr, a, c, b, l, -kk * nn * tl.sin(arg) * f1 * f2 * f3, g0x, g0y, g0z, g3x, g3y, g3z, fa, fc)
    _push(g_ptr, a, c, x1, y1, z1, r1, ang * d1 * f2 * f3)
    _push(g_ptr, c, b, x2, y2, z2, r2, ang * f1 * d2 * f3)
    _push(g_ptr, b, l, x3, y3, z3, r3, ang * f1 * f2 * d3)


@triton.jit
def _inversions(x_ptr, idx_ptr, par_ptr, g_ptr, e_ptr, c_ptr, BLOCK: tl.constexpr):
    """Inversion ``p`` (centre, j, k, l) with the parameters (sign flag, cos w0, k, three damping radii)."""
    p = tl.program_id(0).to(tl.int64)
    c = tl.load(idx_ptr + 4 * p)
    j = tl.load(idx_ptr + 4 * p + 1)
    k = tl.load(idx_ptr + 4 * p + 2)
    l = tl.load(idx_ptr + 4 * p + 3)
    px, py, pz, r1 = _delta(x_ptr, j, c)
    qx, qy, qz, _ = _delta(x_ptr, k, c)
    vx, vy, vz, rv = _delta(x_ptr, l, c)
    nx = py * qz - pz * qy
    ny = pz * qx - px * qz
    nz = px * qy - py * qx
    rn = tl.sqrt(nx * nx + ny * ny + nz * nz)
    sw = (nx * vx + ny * vy + nz * vz) / (rn * rv)
    cw = tl.sqrt(tl.maximum(1.0 - sw * sw, 0.0))
    x2, y2, z2, r2 = _delta(x_ptr, j, k)
    x3, y3, z3, r3 = _delta(x_ptr, j, l)
    cd = tl.load(c_ptr + 2)
    f1, d1 = _quartic(r1, tl.load(par_ptr + 6 * p + 3), cd)
    f2, d2 = _quartic(r2, tl.load(par_ptr + 6 * p + 4), cd)
    f3, d3 = _quartic(r3, tl.load(par_ptr + 6 * p + 5), cd)
    kk = tl.load(par_ptr + 6 * p + 2)
    ea = kk * (1.0 - cw)
    dea = -kk
    if tl.load(par_ptr + 6 * p) > 0.5:
        u = cw - tl.load(par_ptr + 6 * p + 1)
        ea = kk * u * u
        dea = 2.0 * kk * u
    tl.atomic_add(e_ptr + 2, ea * f1 * f2 * f3, sem="relaxed")
    w = tl.where(cw > 0.0, -dea * sw / tl.where(cw > 0.0, cw, 1.0), 0.0) * f1 * f2 * f3
    inv = 1.0 / (rn * rv)
    hx = w * (vx * inv - sw * nx / (rn * rn))
    hy = w * (vy * inv - sw * ny / (rn * rn))
    hz = w * (vz * inv - sw * nz / (rn * rn))
    gjx = qy * hz - qz * hy
    gjy = qz * hx - qx * hz
    gjz = qx * hy - qy * hx
    gkx = hy * pz - hz * py
    gky = hz * px - hx * pz
    gkz = hx * py - hy * px
    glx = w * (nx * inv - sw * vx / (rv * rv))
    gly = w * (ny * inv - sw * vy / (rv * rv))
    glz = w * (nz * inv - sw * vz / (rv * rv))
    _add(g_ptr, j, gjx, gjy, gjz)
    _add(g_ptr, k, gkx, gky, gkz)
    _add(g_ptr, l, glx, gly, glz)
    _add(g_ptr, c, -gjx - gkx - glx, -gjy - gky - gly, -gjz - gkz - glz)
    _push(g_ptr, j, c, px, py, pz, r1, ea * d1 * f2 * f3)
    _push(g_ptr, j, k, x2, y2, z2, r2, ea * f1 * d2 * f3)
    _push(g_ptr, j, l, x3, y3, z3, r3, ea * f1 * f2 * d3)


@triton.jit
def _triples(x_ptr, idx_ptr, co_ptr, g_ptr, e_ptr, BLOCK: tl.constexpr):
    """Bonded three-body term of the triple ``p``."""
    p = tl.program_id(0).to(tl.int64)
    i = tl.load(idx_ptr + 3 * p)
    j = tl.load(idx_ptr + 3 * p + 1)
    k = tl.load(idx_ptr + 3 * p + 2)
    ax, ay, az, a = _delta(x_ptr, i, j)
    bx, by, bz, b = _delta(x_ptr, i, k)
    cx, cy, cz, c = _delta(x_ptr, j, k)
    ci = (a * a + b * b - c * c) / (2.0 * a * b)
    cj = (a * a + c * c - b * b) / (2.0 * a * c)
    ck = (b * b + c * c - a * a) / (2.0 * b * c)
    pr = a * b * c
    p3 = tl.load(co_ptr + p) / (pr * pr * pr)
    ang = 3.0 * ci * cj * ck + 1.0
    tl.atomic_add(e_ptr + 4, p3 * ang, sem="relaxed")
    da = 3.0 * ((1.0 / b - ci / a) * cj * ck + ci * (1.0 / c - cj / a) * ck - ci * cj * a / (b * c)) - 3.0 * ang / a
    db = 3.0 * ((1.0 / a - ci / b) * cj * ck - ci * b / (a * c) * ck + ci * cj * (1.0 / c - ck / b)) - 3.0 * ang / b
    dc = 3.0 * (-c / (a * b) * cj * ck + ci * (1.0 / a - cj / c) * ck + ci * cj * (1.0 / b - ck / c)) - 3.0 * ang / c
    _push(g_ptr, i, j, ax, ay, az, a, p3 * da)
    _push(g_ptr, i, k, bx, by, bz, b, p3 * db)
    _push(g_ptr, j, k, cx, cy, cz, c, p3 * dc)


@triton.jit
def _hb_radial(rr, rab):
    """The distance damping of a hydrogen bond and its logarithmic derivative in ``rab`` times ``rab``."""
    y = 22.0 * rr / (rab * rab)
    y2 = y * y
    a = y2 * y2 * y2
    z = rab * rab / 85.0
    z2 = z * z
    b = z2 * z2 * z2
    return 1.0 / ((1.0 + a) * (1.0 + b)), 12.0 * a / (1.0 + a) - 12.0 * b / (1.0 + b)


@triton.jit
def _hb_bonded(x_ptr, idx_ptr, par_ptr, nbp_ptr, nb_ptr, aux_ptr, yp_ptr, y_ptr, g_ptr, e_ptr, c_ptr, cut,
               BLOCK: tl.constexpr):
    """Hydrogen bond ``p`` of an H bonded to its donor (donor, acceptor, H) with the parameters (radius sum,
    strength, neighbour exponent). The neighbours of the acceptor are rows ``nbp[p]`` to ``nbp[p + 1]`` of ``nb``.
    ``aux`` holds the kind (1: acceptor with one neighbour ``aux1`` whose other neighbours are rows ``yp`` of
    ``y``, 2: two-coordinate N with the neighbours ``aux1``, ``aux2``). Rows with ``r_AB^2`` at or above ``cut``
    contribute nothing."""
    p = tl.program_id(0).to(tl.int64)
    a = tl.load(idx_ptr + 3 * p)
    b = tl.load(idx_ptr + 3 * p + 1)
    h = tl.load(idx_ptr + 3 * p + 2)
    rr = tl.load(par_ptr + 3 * p)
    cnb = tl.load(par_ptr + 3 * p + 2)
    abx, aby, abz, rab = _delta(x_ptr, a, b)
    ahx, ahy, ahz, rah = _delta(x_ptr, a, h)
    bhx, bhy, bhz, rbh = _delta(x_ptr, b, h)
    t = 49.0 / rr * ((rah + rbh) / rab - 1.0)
    on = (t < 15.0) & (rab * rab < cut)
    o = tl.where(on, 2.0 / (1.0 + tl.exp(tl.minimum(t, 15.0))), 0.0)
    do = tl.where(on, -o * (1.0 - 0.5 * o), 0.0) * 49.0 / rr
    prod = rab * 0.0 + 1.0
    lo = tl.load(nbp_ptr + p)
    hi = tl.load(nbp_ptr + p + 1)
    for s in range(lo, hi):
        i = tl.load(nb_ptr + s)
        _, _, _, rai = _delta(x_ptr, a, i)
        _, _, _, rbi = _delta(x_ptr, b, i)
        prod *= 2.0 / (1.0 + tl.exp(-cnb * ((rai + rbi) / rab - 1.0))) - 1.0
    kind = tl.load(aux_ptr + 3 * p)
    xb = tl.load(aux_ptr + 3 * p + 1)
    x2 = tl.load(aux_ptr + 3 * p + 2)
    ylo = tl.load(yp_ptr + p)
    yhi = tl.load(yp_ptr + p + 1)
    f1 = rab * 0.0 + 1.0
    f2 = rab * 0.0 + 1.0
    ux, uy, uz, ru = _delta(x_ptr, xb, b)
    cs = -(ux * bhx + uy * bhy + uz * bhz) / (ru * rbh)
    if kind == 1:
        f1 = 1.0 - tl.load(c_ptr + 7) * (cs + 0.5) * (cs + 0.5)
        for s in range(ylo, yhi):
            phi, ok, g0x, g0y, g0z, g3x, g3y, g3z, fa, fc = _dihedral(x_ptr, h, b, xb, tl.load(y_ptr + s))
            sn = tl.where(ok, tl.sin(phi), 1.0)  # a neighbour on the acceptor axis counts as perpendicular
            prod *= 1.0 - tl.load(c_ptr + 6) * sn * sn
    lone = tl.load(c_ptr + 5)
    mx = tl.load(x_ptr + 3 * b) - 0.5 * (tl.load(x_ptr + 3 * xb) + tl.load(x_ptr + 3 * x2))
    my = tl.load(x_ptr + 3 * b + 1) - 0.5 * (tl.load(x_ptr + 3 * xb + 1) + tl.load(x_ptr + 3 * x2 + 1))
    mz = tl.load(x_ptr + 3 * b + 2) - 0.5 * (tl.load(x_ptr + 3 * xb + 2) + tl.load(x_ptr + 3 * x2 + 2))
    rm = tl.sqrt(mx * mx + my * my + mz * mz)
    wx = abx - lone * mx / rm
    wy = aby - lone * my / rm
    wz = abz - lone * mz / rm
    rw = tl.sqrt(wx * wx + wy * wy + wz * wz)
    t2 = 56.0 / rr * ((lone + rw) / rab - 1.0)
    df2 = rab * 0.0
    if kind == 2:
        f2 = 2.0 / (1.0 + tl.exp(tl.minimum(t2, 300.0)))
        df2 = tl.where(t2 < 300.0, -f2 * (1.0 - 0.5 * f2), 0.0) * 56.0 / rr
    damp, dlog = _hb_radial(rr, rab)
    rad = tl.load(c_ptr + 3) / (rbh * rbh * rbh) - tl.load(c_ptr + 4) / (rab * rab * rab)
    pre = -tl.load(par_ptr + 3 * p + 1)
    en = pre * o * prod * f1 * f2 * damp * rad
    tl.atomic_add(e_ptr + 6, en, sem="relaxed")
    rest = pre * prod * f1 * f2 * damp
    # the switch in (rah + rbh) / rab, the damping and the radial part
    co = rest * rad * do / rab
    dab = -co * (rah + rbh) / rab + en * dlog / rab + rest * o * 3.0 * tl.load(c_ptr + 4) / (rab * rab * rab * rab)
    dbh = co - rest * o * 3.0 * tl.load(c_ptr + 3) / (rbh * rbh * rbh * rbh)
    _push(g_ptr, a, h, ahx, ahy, ahz, rah, co)
    for s in range(lo, hi):
        i = tl.load(nb_ptr + s)
        aix, aiy, aiz, rai = _delta(x_ptr, a, i)
        bix, biy, biz, rbi = _delta(x_ptr, b, i)
        arm = (rai + rbi) / rab
        sg = 2.0 / (1.0 + tl.exp(-cnb * (arm - 1.0)))
        fac = sg - 1.0
        cf = tl.where(fac != 0.0, en / tl.where(fac != 0.0, fac, 1.0), 0.0) * sg * (1.0 - 0.5 * sg) * cnb / rab
        _push(g_ptr, a, i, aix, aiy, aiz, rai, cf)
        _push(g_ptr, b, i, bix, biy, biz, rbi, cf)
        dab += -cf * arm
    if kind == 1:
        # the angle at the acceptor between its neighbour and the H
        w = -en / f1 * 2.0 * tl.load(c_ptr + 7) * (cs + 0.5)
        iuv = 1.0 / (ru * rbh)
        gux = w * (-bhx * iuv - cs * ux / (ru * ru))
        guy = w * (-bhy * iuv - cs * uy / (ru * ru))
        guz = w * (-bhz * iuv - cs * uz / (ru * ru))
        gvx = w * (ux * iuv + cs * bhx / (rbh * rbh))
        gvy = w * (uy * iuv + cs * bhy / (rbh * rbh))
        gvz = w * (uz * iuv + cs * bhz / (rbh * rbh))
        _add(g_ptr, xb, gux, guy, guz)
        _add(g_ptr, h, gvx, gvy, gvz)
        _add(g_ptr, b, -gux - gvx, -guy - gvy, -guz - gvz)
        for s in range(ylo, yhi):
            y = tl.load(y_ptr + s)
            phi, ok, g0x, g0y, g0z, g3x, g3y, g3z, fa, fc = _dihedral(x_ptr, h, b, xb, y)
            sn = tl.where(ok, tl.sin(phi), 1.0)  # a neighbour on the acceptor axis counts as perpendicular
            fy = 1.0 - tl.load(c_ptr + 6) * sn * sn
            _add_dihedral(g_ptr, h, b, xb, y, -en / fy * tl.load(c_ptr + 6) * tl.sin(2.0 * phi), g0x, g0y, g0z,
                          g3x, g3y, g3z, fa, fc)
    if kind == 2:
        # the lone pair at lone * m / |m| from the acceptor, m from the midpoint of its neighbours
        cl = pre * o * prod * f1 * damp * rad * df2 / rab
        dab += -cl * (lone + rw) / rab
        ex = cl * wx / rw
        ey = cl * wy / rw
        ez = cl * wz / rw
        dot = (ex * mx + ey * my + ez * mz) / (rm * rm)
        hx = -lone * (ex - dot * mx) / rm
        hy = -lone * (ey - dot * my) / rm
        hz = -lone * (ez - dot * mz) / rm
        _add(g_ptr, a, ex, ey, ez)
        _add(g_ptr, b, hx - ex, hy - ey, hz - ez)
        _add(g_ptr, xb, -0.5 * hx, -0.5 * hy, -0.5 * hz)
        _add(g_ptr, x2, -0.5 * hx, -0.5 * hy, -0.5 * hz)
    _push(g_ptr, a, b, abx, aby, abz, rab, dab)
    _push(g_ptr, b, h, bhx, bhy, bhz, rbh, dbh)


@triton.jit
def _hb_loose(x_ptr, idx_ptr, par_ptr, g_ptr, e_ptr, cut, cut3, BLOCK: tl.constexpr):
    """Hydrogen bond ``p`` of an H bonded to neither or to both partners (A, B, H) with the parameters (radius sum,
    acidity of A and of B, basicity of A and of B, the charge factor of H). Rows with ``r_AB^2`` at or above ``cut``
    or ``r_AB^2 + r_AH^2 + r_BH^2`` at or above ``cut3`` contribute nothing."""
    p = tl.program_id(0).to(tl.int64)
    a = tl.load(idx_ptr + 3 * p)
    b = tl.load(idx_ptr + 3 * p + 1)
    h = tl.load(idx_ptr + 3 * p + 2)
    rr = tl.load(par_ptr + 6 * p)
    ca = tl.load(par_ptr + 6 * p + 1)
    cb = tl.load(par_ptr + 6 * p + 2)
    ba = tl.load(par_ptr + 6 * p + 3)
    bb = tl.load(par_ptr + 6 * p + 4)
    abx, aby, abz, rab = _delta(x_ptr, a, b)
    ahx, ahy, ahz, rah = _delta(x_ptr, a, h)
    bhx, bhy, bhz, rbh = _delta(x_ptr, b, h)
    t = 49.0 / rr * ((rah + rbh) / rab - 1.0)
    on = (t < 15.0) & (rab * rab < cut) & (rab * rab + rah * rah + rbh * rbh < cut3)
    o = tl.where(on, 2.0 / (1.0 + tl.exp(tl.minimum(t, 15.0))), 0.0)
    do = tl.where(on, -o * (1.0 - 0.5 * o), 0.0) * 49.0 / rr
    wa = rbh * rbh * rbh * rbh
    wb = rah * rah * rah * rah
    acid = (ca * wa + cb * wb) / (wa + wb)
    base = (ba * wb + bb * wa) / (wa + wb)
    damp, dlog = _hb_radial(rr, rab)
    pre = -5.0 * tl.load(par_ptr + 6 * p + 5) * damp / (rab * rab * rab)
    en = pre * o * acid * base
    tl.atomic_add(e_ptr + 6, en, sem="relaxed")
    co = pre * acid * base * do / rab
    sa = 4.0 * wb / rah / (wa + wb)
    sb = 4.0 * wa / rbh / (wa + wb)
    dah = co + pre * o * sa * ((cb - acid) * base + acid * (ba - base))
    dbh = co + pre * o * sb * ((ca - acid) * base + acid * (bb - base))
    dab = -co * (rah + rbh) / rab + en * (dlog - 3.0) / rab
    _push(g_ptr, a, b, abx, aby, abz, rab, dab)
    _push(g_ptr, a, h, ahx, ahy, ahz, rah, dah)
    _push(g_ptr, b, h, bhx, bhy, bhz, rbh, dbh)


@triton.jit
def _halogen_bonds(x_ptr, idx_ptr, par_ptr, g_ptr, e_ptr, cut, BLOCK: tl.constexpr):
    """Halogen bond ``p`` (donor, acceptor, halogen) with the parameters (radius sum, strength). Rows with
    ``r_DY^2`` at or above ``cut`` contribute nothing."""
    p = tl.program_id(0).to(tl.int64)
    d = tl.load(idx_ptr + 3 * p)
    y = tl.load(idx_ptr + 3 * p + 1)
    xa = tl.load(idx_ptr + 3 * p + 2)
    ax, ay, az, rdx = _delta(x_ptr, d, xa)
    bx, by, bz, rxy = _delta(x_ptr, xa, y)
    cx, cy, cz, rdy = _delta(x_ptr, d, y)
    t = 70.0 * ((rdx + rxy) / rdy - 1.0)
    on = (t < 15.0) & (rdy * rdy < cut)
    o = tl.where(on, 2.0 / (1.0 + tl.exp(tl.minimum(t, 15.0))), 0.0)
    do = tl.where(on, -o * (1.0 - 0.5 * o), 0.0) * 70.0 / rdy
    u = 5.0 * tl.load(par_ptr + 2 * p) / (rxy * rxy)
    u2 = u * u
    sa = u2 * u2 * u2
    v = rxy * rxy / 70.0
    v2 = v * v
    sb = v2 * v2 * v2
    rest = -tl.load(par_ptr + 2 * p + 1) / ((1.0 + sa) * (1.0 + sb) * rxy * rxy * rxy)
    en = rest * o
    tl.atomic_add(e_ptr + 7, en, sem="relaxed")
    _push(g_ptr, d, xa, ax, ay, az, rdx, rest * do)
    _push(g_ptr, xa, y, bx, by, bz, rxy, rest * do + en * (12.0 * sa / (1.0 + sa) - 12.0 * sb / (1.0 + sb) - 3.0) / rxy)
    _push(g_ptr, d, y, cx, cy, cz, rdy, -rest * do * (rdx + rxy) / rdy)


@triton.jit
def _charge_energy(sol_ptr, rhs_ptr, e_ptr, dim, BLOCK: tl.constexpr):
    """The electrostatic energy, minus half the product of the solution and the right-hand side."""
    acc = tl.load(sol_ptr) * 0.0
    for k in range(0, dim):
        acc += tl.load(sol_ptr + k) * tl.load(rhs_ptr + k)
    tl.store(e_ptr + 8, -0.5 * acc)


@triton.jit
def _prepare(pos_ptr, x_ptr, work_ptr, bohr_ptr, ref_ptr, lim_ptr, flag_ptr, n, n3, nw, BLOCK: tl.constexpr):
    """The positions in Bohr and the accumulators at zero, one entry of each per program; the flag set when an atom
    has moved by more than half the skin (``4 d^2`` above the squared skin) from the positions ``ref`` of the
    hydrogen- and halogen-bond lists."""
    p = tl.program_id(0)
    if p < n3:
        tl.store(x_ptr + p, tl.load(pos_ptr + p) / tl.load(bohr_ptr))
    if p < nw:
        tl.store(work_ptr + p, tl.load(bohr_ptr) * 0.0)
    if p < n:
        dx = tl.load(pos_ptr + 3 * p) - tl.load(ref_ptr + 3 * p)
        dy = tl.load(pos_ptr + 3 * p + 1) - tl.load(ref_ptr + 3 * p + 1)
        dz = tl.load(pos_ptr + 3 * p + 2) - tl.load(ref_ptr + 3 * p + 2)
        if 4.0 * (dx * dx + dy * dy + dz * dz) > tl.load(lim_ptr):
            tl.store(flag_ptr, tl.load(lim_ptr) * 0.0 + 1.0)


@triton.jit
def _solve_small(mat_ptr, rhs_ptr, fac_ptr, sol_ptr, dxl_ptr, glog_ptr, e_ptr, dim, n, BLOCK: tl.constexpr):
    """The charge equations by an L D L^T factorization without pivoting (the matrix is a positive definite block
    bordered by the constraints, which this order factorizes stably): ``fac`` holds L below and D on the diagonal.
    Then the charge term of the count gradient, ``-q dx/dlogCN`` per atom, and the electrostatic energy."""
    for j in range(0, dim):
        d = tl.load(mat_ptr + j * dim + j)
        for k in range(0, j):
            l = tl.load(fac_ptr + j * dim + k)
            d -= l * l * tl.load(fac_ptr + k * dim + k)
        tl.store(fac_ptr + j * dim + j, d)
        for i in range(j + 1, dim):
            v = tl.load(mat_ptr + i * dim + j)
            for k in range(0, j):
                v -= tl.load(fac_ptr + i * dim + k) * tl.load(fac_ptr + j * dim + k) * tl.load(fac_ptr + k * dim + k)
            tl.store(fac_ptr + i * dim + j, v / d)
    for i in range(0, dim):
        v = tl.load(rhs_ptr + i)
        for k in range(0, i):
            v -= tl.load(fac_ptr + i * dim + k) * tl.load(sol_ptr + k)
        tl.store(sol_ptr + i, v)
    for i in range(0, dim):
        tl.store(sol_ptr + i, tl.load(sol_ptr + i) / tl.load(fac_ptr + i * dim + i))
    for r in range(0, dim):
        i = dim - 1 - r
        v = tl.load(sol_ptr + i)
        for k in range(i + 1, dim):
            v -= tl.load(fac_ptr + k * dim + i) * tl.load(sol_ptr + k)
        tl.store(sol_ptr + i, v)
    for i in range(0, n):
        tl.store(glog_ptr + i, tl.load(glog_ptr + i) - tl.load(sol_ptr + i) * tl.load(dxl_ptr + i))
    acc = tl.load(sol_ptr) * 0.0
    for k in range(0, dim):
        acc += tl.load(sol_ptr + k) * tl.load(rhs_ptr + k)
    tl.store(e_ptr + 8, -0.5 * acc)


class _GpuLaunch:
    """``kernel[grid](*args, **meta)`` on GPU tensors: one warp per row, four per block of pairs."""

    def __init__(self, kernel, grid, *args, **meta):
        self.kernel, self.grid, self.args, self.meta = kernel, grid, args, meta
        self.n = int(grid[0])

    def __call__(self):
        if self.n > 0:
            self.kernel[self.grid](*self.args, **self.meta, num_warps=4 if self.meta.get("BLOCK", 1) > 1 else 1)


class _Sequence:
    """Launches run one after another (the GPU orders them on its stream)."""

    def __init__(self, launches: list):
        self.launches = launches

    def __call__(self):
        for run in self.launches:
            run()


def _recorded(P):
    """``P`` (a launch maker) whose launches keep their kernel and arguments, from which :func:`_twin` makes the
    launches of the tangent kernels."""

    def launch(kernel, grid, *args, **meta):
        run = P(kernel, grid, *args, **meta)
        run.spec = (kernel, grid, args, meta)
        return run

    return launch


def _key(t: Tensor) -> tuple:
    return t.data_ptr(), t.numel()


def _twin(P, run, twins: dict):
    """The launch of the tangent kernel of ``run`` (made by :func:`_recorded`) on the same arguments, every pointer
    of its request bound to the tangent buffer that ``twins`` keeps for that argument."""
    from . import gfnff_tangents

    kernel, grid, args, meta = run.spec
    tangent = getattr(gfnff_tangents, getattr(kernel, "fn", kernel).__name__ + "_t")
    names = kernel.arg_names
    extra = {name: twins[_key(args[names.index(name[:-2])])] for name in tangent.arg_names
             if name.endswith("_t") and name[:-2] in names}
    return P(tangent, grid, *args, **extra, **meta)


def _twins(pairs) -> dict:
    """Tangent buffers by their primal buffers."""
    return {_key(a): b for a, b in pairs}


class _Serial(cpu_kernels.Sequence):
    """:class:`~xnn.common.cpu_kernels.Sequence` on one thread (the few atoms that do not fill several)."""

    def __call__(self):
        if self.launches:
            self.lib.run_sequence(len(self.launches), self.setups, self.bodies, self.tps, self.fps, self.ips,
                                  self.nps, self.programs, self.wait, 1)


def _launches(runs: list, gpu: bool, n: int):
    """Launches of a system of ``n`` atoms run in order: a GPU stream's sequence or one parallel region on the CPU
    (one thread up to :data:`SERIAL_UP_TO` atoms)."""
    if gpu:
        return _Sequence(runs)
    return _Serial(runs) if n <= SERIAL_UP_TO else cpu_kernels.Sequence(runs)


def _pointer(rows: Tensor, n: int) -> Tensor:
    """Compressed-row pointer ``(n + 1,)`` of sorted row numbers."""
    out = torch.zeros(n + 1, dtype=torch.int64, device=rows.device)
    out[1:] = torch.cumsum(torch.bincount(rows, minlength=n), 0)
    return out


def _ragged(idx: Tensor, mask: Tensor) -> tuple:
    """Compressed rows of a padded index table."""
    counts = mask.sum(1)
    out = torch.zeros(idx.shape[0] + 1, dtype=torch.int64, device=idx.device)
    out[1:] = torch.cumsum(counts, 0)
    return out, idx[mask].contiguous()


class Plan:
    """One energy-and-gradient evaluation of a bound GFN-FF model on the CPU.

    Calling it with positions ``(N, 3)`` in Angstrom returns the nine term energies (Hartree, in the order of
    ``gfnff.TERMS``) and the gradient of their sum in Hartree per Angstrom.
    """

    def __init__(self, model):
        f64, i64 = torch.float64, torch.int64
        dev = self.device = model.rcov.device
        n = self.n = int(model.num_atoms)
        col = lambda *ts: torch.stack([t.to(f64) for t in ts], 1).contiguous()  # noqa: E731
        z = lambda *shape: torch.zeros(*shape, dtype=f64, device=dev)  # noqa: E731
        from . import gfnfflib as lib
        from .gfnff import DISPERSION_CUT

        self.bohr = float(lib.BOHR)
        # the buffers that start every evaluation at zero, in one tensor
        self.work = z(5 * n + 9)
        self.g, self.glog, self.cnt = self.work[:3 * n].view(n, 3), self.work[3 * n:4 * n], self.work[4 * n:5 * n]
        self.e = self.work[5 * n:]
        self.x, self.lcn, self.dl, self.dxl = z(n, 3), z(n), z(n), z(n)
        nc = int(model.e_tot.shape[0])
        self.dim = n + nc
        self.mat, self.rhs, self.sol = z(self.dim, self.dim), z(self.dim), z(self.dim)
        self.q = self.sol[:n]
        self.mat[:n, :n] = torch.diag(model.e_diag)
        self.mat[n:, :n] = model.e_con
        self.mat[:n, n:] = model.e_con.T
        self.rhs[n:] = model.e_tot
        nfreq, nref = int(model.d_alpha.shape[0]), int(model.d_ref.shape[1])
        if int(model.d_alpha.shape[1]) != nref:
            raise cpu_kernels.Unsupported("reference tables of different widths")
        if bool((model.d_w < 0).any()):
            raise cpu_kernels.Unsupported("negative integration weights")
        # small molecules sum the C6 coefficients in the pair kernel, large ones take them from two matrix products
        self.dense = n > DENSE_FROM
        self.s, self.ds = z(n, nfreq), z(n, nfreq)
        self.c6, self.dc = (z(n, n), z(n, n)) if self.dense else (self.s, self.ds)
        sw = torch.sqrt(3.0 / math.pi * model.d_w.to(f64)).contiguous()
        pairs = model.r_ij.contiguous()
        npair = int(pairs.shape[0])
        rcov, gam = model.rcov.contiguous(), model.e_g.contiguous()
        c = CONST.to(dev)
        gpu = dev.type == "cuda"
        P = _recorded(_GpuLaunch if gpu else cpu_kernels.prepare)
        pb = PAIR_BLOCK if gpu else 1
        pgrid = (triton.cdiv(npair, pb),)
        first = P(_pair_first, pgrid, self.x, pairs, rcov, gam, self.cnt, self.mat, c, n, self.dim, npair, BLOCK=pb)
        atoms = P(_atom_terms, (n,), self.cnt, self.lcn, self.dl, self.rhs, self.dxl, model.e_x0.contiguous(),
                  model.e_kappa.contiguous(), model.d_ref.contiguous(), model.d_valid.sum(1).to(i64).contiguous(),
                  model.d_alpha.to(f64).contiguous(), sw, self.s, self.ds, c, n, NREF=nref, NFREQ=nfreq, BLOCK=1)
        self.first = _Sequence([first, atoms])
        self.firsts = (first, atoms)
        runs = []
        acc = model.b_acc
        runs.append(P(_bonds, (int(model.b_ij.shape[0]),), self.x, model.b_ij.contiguous(),
                      col(model.b_fp, model.b_ra, model.b_ci, model.b_cj, model.b_sh, model.b_k, model.b_al), self.lcn,
                      _pointer(acc[:, 0], int(model.b_ij.shape[0])), acc[:, 1:].contiguous(),
                      model.b_accd.contiguous(), self.g, self.glog, self.e, c, BLOCK=1))
        ai, rad = model.a_idx, model.arad
        lin = ((model.a_t0 - math.pi).abs() < 1e-6).to(f64)
        runs.append(P(_angles, (int(ai.shape[0]),), self.x, ai.contiguous(),
                      col(torch.cos(model.a_t0), lin, model.a_k, rad[ai[:, 0]] + rad[ai[:, 1]],
                          rad[ai[:, 0]] + rad[ai[:, 2]]), self.g, self.e, c, BLOCK=1))
        ti = model.t_idx
        runs.append(P(_torsions, (int(ti.shape[0]),), self.x, ti.contiguous(),
                      col(model.t_n, model.t_p0, model.t_k, rad[ti[:, 0]] + rad[ti[:, 2]],
                          rad[ti[:, 2]] + rad[ti[:, 1]], rad[ti[:, 1]] + rad[ti[:, 3]]), self.g, self.e, c, BLOCK=1))
        ii = model.i_idx
        runs.append(P(_inversions, (int(ii.shape[0]),), self.x, ii.contiguous(),
                      col(model.i_neg, model.i_c0, model.i_k, rad[ii[:, 0]] + rad[ii[:, 1]],
                          rad[ii[:, 1]] + rad[ii[:, 2]], rad[ii[:, 1]] + rad[ii[:, 3]]), self.g, self.e, c, BLOCK=1))
        runs.append(P(_triples, (int(model.m_idx.shape[0]),), self.x, model.m_idx.contiguous(),
                      model.m_c.contiguous(), self.g, self.e, BLOCK=1))
        cut, cut3, xcut = (float(v) for v in model.list_cuts)
        bb = model.h_bb
        nbb = int(bb.shape[0])
        aux = torch.zeros(nbb, 3, dtype=i64, device=dev)
        ymask = torch.zeros(nbb, int(model.h_o1y.shape[1]), dtype=torch.bool, device=dev)
        yidx = torch.zeros(nbb, int(model.h_o1y.shape[1]), dtype=i64, device=dev)
        if int(model.h_o1.shape[0]):
            aux[model.h_o1, 0], aux[model.h_o1, 1] = 1, model.h_o1x
            ymask[model.h_o1], yidx[model.h_o1] = model.h_o1ym, model.h_o1y
        if int(model.h_n2.shape[0]):
            aux[model.h_n2, 0] = 2
            aux[model.h_n2, 1:] = model.h_n2nb
        empty = (torch.zeros(1, dtype=i64, device=dev), torch.zeros(0, dtype=i64, device=dev))
        nbp, nbs = _ragged(model.h_nb, model.h_nbm) if nbb else empty
        yp, ys = _ragged(yidx, ymask)
        runs.append(P(_hb_bonded, (nbb,), self.x, bb.contiguous(), col(model.h_rr, model.h_p, model.h_cnb),
                      nbp, nbs, aux, yp, ys, self.g, self.e, c, cut, BLOCK=1))
        runs.append(P(_hb_loose, (int(model.h_bl.shape[0]),), self.x, model.h_bl.contiguous(),
                      col(model.l_rr, model.l_ca, model.l_cb, model.l_ba, model.l_bb, model.l_qh), self.g, self.e,
                      cut, cut3, BLOCK=1))
        runs.append(P(_halogen_bonds, (int(model.x_idx.shape[0]),), self.x, model.x_idx.contiguous(),
                      col(model.x_rad, model.x_p), self.g, self.e, xcut, BLOCK=1))
        runs.append(P(_pair_second, pgrid, self.x, pairs, model.r_c.contiguous(), model.r_m.contiguous(),
                      model.r_cut.contiguous(), model.d_zz.contiguous(), model.d_q.contiguous(),
                      model.d_r06.contiguous(), model.d_r08.contiguous(), self.c6, self.dc, gam, self.sol, self.g,
                      self.glog, self.e, c, n, float(DISPERSION_CUT), npair, NFREQ=0 if self.dense else nfreq,
                      BLOCK=pb))
        self.runs = runs
        self.third = P(_count_gradient, pgrid, self.x, pairs, rcov, self.dl, self.glog, self.g, c, npair, BLOCK=pb)
        self.last = P(_charge_energy, (1,), self.sol, self.rhs, self.e, self.dim, BLOCK=1)
        # few atoms: the factorization as a loop costs less than a LAPACK call
        self.small = None
        if self.dim <= SMALL_SOLVE and not gpu:
            self.fac = z(self.dim, self.dim)
            self.small = P(_solve_small, (1,), self.mat, self.rhs, self.fac, self.sol, self.dxl, self.glog, self.e,
                           self.dim, n, BLOCK=1)
        self.seq, self.checks = None, False
        self.gpu, self.P, self.tangent = gpu, P, None
        if gpu:
            self.second = _Sequence(runs)
            self.tail = _Sequence([self.third, self.last])
        elif self.small is not None:
            # few atoms: the whole evaluation in one call and one parallel region, the positions read in place, and
            # the list check of the model (see GFNFF.follow) on the same positions as a flag
            self.checks, self.skin = not model.fixed_lists, float(model.skin)
            self.pos, self.flag = z(n, 3), z(1)
            self.flag_np = self.flag.numpy()
            self.bohr_t = torch.tensor([self.bohr], dtype=f64)
            self.ref = torch.as_tensor(model._ref, dtype=f64).contiguous()
            self.lim = torch.tensor([float(model.skin) ** 2 if self.checks else math.inf], dtype=f64)
            nw = int(self.work.numel())
            self.prep = P(_prepare, (max(3 * n, nw),), self.pos, self.x, self.work, self.bohr_t, self.ref, self.lim,
                          self.flag, n, 3 * n, nw, BLOCK=1)
            self.seq = cpu_kernels.Sequence([self.prep, first, atoms, self.small, *runs, self.third])
        else:
            # one parallel region per stage on the CPU: the count gradient and the charge energy follow the pair
            # sums
            self.first = cpu_kernels.Deferred([first], [atoms])
            self.second = cpu_kernels.Deferred(runs, [self.third, self.last])
            self.tail = None
        self.keep = (pairs, rcov, gam, sw, c)

    def has_moved(self) -> bool:
        """Whether the last evaluation found an atom farther than half the skin from the positions of the
        hydrogen- and halogen-bond lists (only plans with ``checks`` look)."""
        return bool(self.flag_np[0] > 0.0)

    def __call__(self, positions: Tensor) -> tuple:
        if self.seq is not None:
            pos = positions
            if pos.dtype != torch.float64 or pos.device.type != "cpu" or not pos.is_contiguous():
                pos = pos.detach().to(device="cpu", dtype=torch.float64).contiguous()
            self.prep.tp[0] = pos.data_ptr()
            self.flag_np[0] = 0.0
            s = self.seq
            threads = 1 if self.n <= SERIAL_UP_TO else torch.get_num_threads()
            s.lib.run_sequence(len(s.launches), s.setups, s.bodies, s.tps, s.fps, s.ips, s.nps, s.programs, s.wait,
                               threads)
            return self.e.clone(), self.g / self.bohr
        pos = positions.detach()
        if pos.dtype != torch.float64:
            pos = pos.to(torch.float64)
        torch.div(pos, self.bohr, out=self.x)
        self.work.zero_()
        self.first()
        if self.dense:
            torch.mm(self.s, self.s.T, out=self.c6)
            torch.mm(self.ds, self.s.T, out=self.dc)
        if self.small is None:
            torch.linalg.solve(self.mat, self.rhs, out=self.sol)
        self.glog.addcmul_(self.q, self.dxl, value=-1.0)
        self.second()
        if self.tail is not None:
            self.tail()
        return self.e.clone(), self.g / self.bohr

    def tangents(self) -> None:
        """Make the launches of the tangent kernels (once; :class:`~xnn.common.cpu_kernels.Unsupported` where the
        CPU kernels do not cover them)."""
        if self.tangent is not None:
            return
        z = torch.zeros_like
        n, P = self.n, self.P
        self.twork = z(self.work)
        self.x_t, self.mat_t, self.rhs_t, self.sol_t = z(self.x), z(self.mat), z(self.rhs), z(self.sol)
        self.g_t, self.glog_t = self.twork[:3 * n].view(n, 3), self.twork[3 * n:4 * n]
        self.s_t, self.ds_t = z(self.s), z(self.ds)
        self.c6_t, self.dc_t = (z(self.c6), z(self.dc)) if self.dense else (self.s_t, self.ds_t)
        pairs = [(self.x, self.x_t), (self.cnt, self.twork[4 * n:5 * n]), (self.mat, self.mat_t),
                 (self.rhs, self.rhs_t), (self.sol, self.sol_t), (self.g, self.g_t), (self.glog, self.glog_t),
                 (self.lcn, z(self.lcn)), (self.dl, z(self.dl)), (self.dxl, z(self.dxl)), (self.s, self.s_t),
                 (self.ds, self.ds_t), (self.c6, self.c6_t), (self.dc, self.dc_t)]
        if self.small is not None:
            pairs.append((self.fac, z(self.fac)))
        twins = _twins(pairs)
        self.q_t, self.dxl_t = self.sol_t[:n], twins[_key(self.dxl)]
        first = [_twin(P, run, twins) for run in self.firsts]
        second = [_twin(P, run, twins) for run in self.runs] + [_twin(P, self.third, twins)]
        if self.small is not None:
            self.tangent = (_launches(first + [_twin(P, self.small, twins)] + second, self.gpu, n), None)
        else:
            self.tangent = (_launches(first, self.gpu, n), _launches(second, self.gpu, n))

    def hvp(self, positions: Tensor, direction: Tensor) -> Tensor:
        """The derivative of the gradient (Hartree per Angstrom) at ``positions`` ``(N, 3)`` along ``direction``
        ``(N, 3)`` (Angstrom), the Hessian times ``direction``: the tangent kernels run the evaluation again with
        the tangents of every buffer."""
        self.tangents()
        f64 = torch.float64
        torch.div(positions.detach().to(device=self.device, dtype=f64), self.bohr, out=self.x)
        torch.div(direction.detach().to(device=self.device, dtype=f64), self.bohr, out=self.x_t)
        self.work.zero_()
        self.twork.zero_()
        first, second = self.tangent
        first()
        if second is not None:
            if self.dense:
                torch.mm(self.s, self.s.T, out=self.c6)
                torch.mm(self.ds, self.s.T, out=self.dc)
                torch.mm(self.s_t, self.s.T, out=self.c6_t)
                self.c6_t.addmm_(self.s, self.s_t.T)
                torch.mm(self.ds_t, self.s.T, out=self.dc_t)
                self.dc_t.addmm_(self.ds, self.s_t.T)
            lu = torch.linalg.lu_factor(self.mat)
            self.sol.copy_(torch.linalg.lu_solve(*lu, self.rhs[:, None])[:, 0])
            self.sol_t.copy_(torch.linalg.lu_solve(*lu, (self.rhs_t - self.mat_t @ self.sol)[:, None])[:, 0])
            self.glog_t.addcmul_(self.q_t, self.dxl, value=-1.0)
            self.glog_t.addcmul_(self.q, self.dxl_t, value=-1.0)
            self.glog.addcmul_(self.q, self.dxl, value=-1.0)
            second()
        return self.g_t / self.bohr


#: Inexact constants of the periodic kernels: 1/sqrt(pi), 2/sqrt(pi), 4 pi, 2 pi, the distance (Bohr) within which
#: images of a pair count as minimum images and a floor of their squared bound, the smallest squared distance of a
#: pair (Bohr^2).
PCONST = torch.tensor([1.0 / math.sqrt(math.pi), 2.0 / math.sqrt(math.pi), 4.0 * math.pi, 2.0 * math.pi,
                       0.01, 1e-14, 1e-20], dtype=torch.float64)


@triton.jit
def _cell(geo_ptr):
    """The cell rows in Bohr from the first nine entries of ``geo``."""
    return (tl.load(geo_ptr), tl.load(geo_ptr + 1), tl.load(geo_ptr + 2), tl.load(geo_ptr + 3), tl.load(geo_ptr + 4),
            tl.load(geo_ptr + 5), tl.load(geo_ptr + 6), tl.load(geo_ptr + 7), tl.load(geo_ptr + 8))


@triton.jit
def _p_prepare(pos_ptr, cell_ptr, wrap_ptr, base_ptr, shift_ptr, x_ptr, geo_ptr, work_ptr, bohr_ptr, n, ne, nw,
               BLOCK: tl.constexpr):
    """The cell in Bohr with its inverse (``geo``: cell rows, inverse, volume), the atoms wrapped into the cell as the
    topology saw them followed by the images the rows reach (``x``, Bohr), and the accumulators at zero; one entry of
    each per program."""
    p = tl.program_id(0)
    bohr = tl.load(bohr_ptr)
    a0 = tl.load(cell_ptr) / bohr
    a1 = tl.load(cell_ptr + 1) / bohr
    a2 = tl.load(cell_ptr + 2) / bohr
    b0 = tl.load(cell_ptr + 3) / bohr
    b1 = tl.load(cell_ptr + 4) / bohr
    b2 = tl.load(cell_ptr + 5) / bohr
    c0 = tl.load(cell_ptr + 6) / bohr
    c1 = tl.load(cell_ptr + 7) / bohr
    c2 = tl.load(cell_ptr + 8) / bohr
    if p == 0:
        det = a0 * (b1 * c2 - b2 * c1) - a1 * (b0 * c2 - b2 * c0) + a2 * (b0 * c1 - b1 * c0)
        tl.store(geo_ptr, a0)
        tl.store(geo_ptr + 1, a1)
        tl.store(geo_ptr + 2, a2)
        tl.store(geo_ptr + 3, b0)
        tl.store(geo_ptr + 4, b1)
        tl.store(geo_ptr + 5, b2)
        tl.store(geo_ptr + 6, c0)
        tl.store(geo_ptr + 7, c1)
        tl.store(geo_ptr + 8, c2)
        # the inverse: fractional coordinates are positions times it
        tl.store(geo_ptr + 9, (b1 * c2 - b2 * c1) / det)
        tl.store(geo_ptr + 10, (a2 * c1 - a1 * c2) / det)
        tl.store(geo_ptr + 11, (a1 * b2 - a2 * b1) / det)
        tl.store(geo_ptr + 12, (b2 * c0 - b0 * c2) / det)
        tl.store(geo_ptr + 13, (a0 * c2 - a2 * c0) / det)
        tl.store(geo_ptr + 14, (a2 * b0 - a0 * b2) / det)
        tl.store(geo_ptr + 15, (b0 * c1 - b1 * c0) / det)
        tl.store(geo_ptr + 16, (a1 * c0 - a0 * c1) / det)
        tl.store(geo_ptr + 17, (a0 * b1 - a1 * b0) / det)
        tl.store(geo_ptr + 18, tl.abs(det))
    if p < ne:
        k = p.to(tl.int64)
        if p >= n:
            k = tl.load(base_ptr + p - n)
        w0 = tl.load(wrap_ptr + 3 * k)
        w1 = tl.load(wrap_ptr + 3 * k + 1)
        w2 = tl.load(wrap_ptr + 3 * k + 2)
        s0 = w0 * 0.0
        s1 = w0 * 0.0
        s2 = w0 * 0.0
        if p >= n:
            s0 = tl.load(shift_ptr + 3 * (p - n))
            s1 = tl.load(shift_ptr + 3 * (p - n) + 1)
            s2 = tl.load(shift_ptr + 3 * (p - n) + 2)
        u0 = s0 - w0
        u1 = s1 - w1
        u2 = s2 - w2
        tl.store(x_ptr + 3 * p, tl.load(pos_ptr + 3 * k) / bohr + u0 * a0 + u1 * b0 + u2 * c0)
        tl.store(x_ptr + 3 * p + 1, tl.load(pos_ptr + 3 * k + 1) / bohr + u0 * a1 + u1 * b1 + u2 * c1)
        tl.store(x_ptr + 3 * p + 2, tl.load(pos_ptr + 3 * k + 2) / bohr + u0 * a2 + u1 * b2 + u2 * c2)
    if p < nw:
        tl.store(work_ptr + p, bohr * 0.0)


@triton.jit
def _p_counts(x_ptr, ij_ptr, tr_ptr, geo_ptr, rcov_ptr, cnt_ptr, cg_ptr, c_ptr, ntr, BLOCK: tl.constexpr):
    """Pair ``p`` (i <= j) over the translations of the count list: its shares of the coordination counts (an atom
    and its own images from both sides), and in ``cg`` the sums over the images of ``exp(-u^2) / r`` times the pair
    vector (three) and times its outer product (nine) that give the count gradient."""
    p = tl.program_id(0).to(tl.int64)
    i = tl.load(ij_ptr + 2 * p)
    j = tl.load(ij_ptr + 2 * p + 1)
    a0, a1, a2, b0, b1, b2, c0, c1, c2 = _cell(geo_ptr)
    ex = tl.load(x_ptr + 3 * j) - tl.load(x_ptr + 3 * i)
    ey = tl.load(x_ptr + 3 * j + 1) - tl.load(x_ptr + 3 * i + 1)
    ez = tl.load(x_ptr + 3 * j + 2) - tl.load(x_ptr + 3 * i + 2)
    rc = tl.load(c_ptr + 11) * (tl.load(rcov_ptr + i) + tl.load(rcov_ptr + j))
    inv_sqrt_pi = tl.load(c_ptr + 8)  # noqa: F841, read by the erf tangent
    acc = ex * 0.0
    gx = ex * 0.0
    gy = ex * 0.0
    gz = ex * 0.0
    v00 = ex * 0.0
    v01 = ex * 0.0
    v02 = ex * 0.0
    v11 = ex * 0.0
    v12 = ex * 0.0
    v22 = ex * 0.0
    for t in range(0, ntr):
        s0 = tl.load(tr_ptr + 3 * t)
        s1 = tl.load(tr_ptr + 3 * t + 1)
        s2 = tl.load(tr_ptr + 3 * t + 2)
        dx = ex + s0 * a0 + s1 * b0 + s2 * c0
        dy = ey + s0 * a1 + s1 * b1 + s2 * c1
        dz = ez + s0 * a2 + s1 * b2 + s2 * c2
        r = tl.sqrt(dx * dx + dy * dy + dz * dz)
        own = (i == j) & (s0 * s0 + s1 * s1 + s2 * s2 == 0.0)
        near = tl.where(own, r < 0.0, r < 2.0 * rc)
        u = 7.5 * (r / rc - 1.0)
        acc += tl.where(near, 0.5 * (1.0 + libdevice.erf(-u)), 0.0)
        s = tl.where(near, tl.exp(-u * u) / tl.where(near, r, 1.0), 0.0)
        gx += s * dx
        gy += s * dy
        gz += s * dz
        v00 += s * dx * dx
        v01 += s * dx * dy
        v02 += s * dx * dz
        v11 += s * dy * dy
        v12 += s * dy * dz
        v22 += s * dz * dz
    tl.atomic_add(cnt_ptr + i, tl.where(i == j, 2.0 * acc, acc), sem="relaxed")
    tl.atomic_add(cnt_ptr + j, tl.where(i == j, 0.0 * acc, acc), sem="relaxed")
    o = 9 * p
    tl.store(cg_ptr + o, gx)
    tl.store(cg_ptr + o + 1, gy)
    tl.store(cg_ptr + o + 2, gz)
    tl.store(cg_ptr + o + 3, v00)
    tl.store(cg_ptr + o + 4, v01)
    tl.store(cg_ptr + o + 5, v02)
    tl.store(cg_ptr + o + 6, v11)
    tl.store(cg_ptr + o + 7, v12)
    tl.store(cg_ptr + o + 8, v22)


@triton.jit
def _p_count_gradient(ij_ptr, cg_ptr, rcov_ptr, dl_ptr, glog_ptr, g_ptr, vir_ptr, c_ptr, BLOCK: tl.constexpr):
    """Pair ``p`` (i <= j): the gradient that reaches the positions and the cell through the coordination
    numbers, from the image sums in ``cg``."""
    p = tl.program_id(0).to(tl.int64)
    i = tl.load(ij_ptr + 2 * p)
    j = tl.load(ij_ptr + 2 * p + 1)
    rc = tl.load(c_ptr + 11) * (tl.load(rcov_ptr + i) + tl.load(rcov_ptr + j))
    wi = tl.load(glog_ptr + i) * tl.load(dl_ptr + i)
    w = tl.where(i == j, 2.0 * wi, wi + tl.load(glog_ptr + j) * tl.load(dl_ptr + j)) * (-7.5 / rc) * tl.load(c_ptr + 8)
    o = 9 * p
    gx = w * tl.load(cg_ptr + o)
    gy = w * tl.load(cg_ptr + o + 1)
    gz = w * tl.load(cg_ptr + o + 2)
    v00 = w * tl.load(cg_ptr + o + 3)
    v01 = w * tl.load(cg_ptr + o + 4)
    v02 = w * tl.load(cg_ptr + o + 5)
    v11 = w * tl.load(cg_ptr + o + 6)
    v12 = w * tl.load(cg_ptr + o + 7)
    v22 = w * tl.load(cg_ptr + o + 8)
    _add(g_ptr, j, gx, gy, gz)
    _add(g_ptr, i, -gx, -gy, -gz)
    _virial(vir_ptr, v00, v01, v02, v01, v11, v12, v02, v12, v22)


@triton.jit
def _virial(vir_ptr, v00, v01, v02, v10, v11, v12, v20, v21, v22):
    tl.atomic_add(vir_ptr, v00, sem="relaxed")
    tl.atomic_add(vir_ptr + 1, v01, sem="relaxed")
    tl.atomic_add(vir_ptr + 2, v02, sem="relaxed")
    tl.atomic_add(vir_ptr + 3, v10, sem="relaxed")
    tl.atomic_add(vir_ptr + 4, v11, sem="relaxed")
    tl.atomic_add(vir_ptr + 5, v12, sem="relaxed")
    tl.atomic_add(vir_ptr + 6, v20, sem="relaxed")
    tl.atomic_add(vir_ptr + 7, v21, sem="relaxed")
    tl.atomic_add(vir_ptr + 8, v22, sem="relaxed")


@triton.jit
def _p_expand(lcn_ptr, lcne_ptr, base_ptr, n, ne, BLOCK: tl.constexpr):
    """The logarithmic coordination numbers of the atoms and of the images the rows reach."""
    p = tl.program_id(0).to(tl.int64)
    k = p
    if p >= n:
        k = tl.load(base_ptr + p - n)
    tl.store(lcne_ptr + p, tl.load(lcn_ptr + k))


@triton.jit
def _p_phases(x_ptr, gv_ptr, cs_ptr, ng, BLOCK: tl.constexpr):
    """Atom ``p``: ``cos`` and ``sin`` of its phase ``k.x`` for every reciprocal vector ``k``."""
    p = tl.program_id(0).to(tl.int64)
    px = tl.load(x_ptr + 3 * p)
    py = tl.load(x_ptr + 3 * p + 1)
    pz = tl.load(x_ptr + 3 * p + 2)
    for m in range(0, ng):
        arg = tl.load(gv_ptr + 3 * m) * px + tl.load(gv_ptr + 3 * m + 1) * py + tl.load(gv_ptr + 3 * m + 2) * pz
        tl.store(cs_ptr + (p * ng + m) * 2, tl.cos(arg))
        tl.store(cs_ptr + (p * ng + m) * 2 + 1, tl.sin(arg))


@triton.jit
def _p_ewald(x_ptr, ij_ptr, geo_ptr, cand_ptr, cube_ptr, gv_ptr, gw_ptr, cs_ptr, gam_ptr, diag_ptr, eta_ptr, mat_ptr,
             pg_ptr, pc_ptr, n, dim, ng, ncand, ncube, BLOCK: tl.constexpr):
    """Pair ``p`` (i <= j): its element of the Ewald matrix of the charges, and the derivatives of that element in
    ``x_j - x_i`` (three) and in a strain of the cell (nine, row of the vector times column of the derivative) in
    ``pg``. Real space over the fixed +-2 cube around the minimum images of the pair (averaged over every image
    within 0.01 Bohr of the shortest),
    reciprocal space over the fixed +-2 cube (``gw`` holds each vector's weight and the factor of its strain
    derivative), and on the diagonal the self term and the hardness. ``ncand`` and ``ncube`` count the candidate
    images and the cube's translations (loops at run time, not unrolled on the CPU)."""
    p = tl.program_id(0).to(tl.int64)
    i = tl.load(ij_ptr + 2 * p)
    j = tl.load(ij_ptr + 2 * p + 1)
    a0, a1, a2, b0, b1, b2, c0, c1, c2 = _cell(geo_ptr)
    eta = tl.load(eta_ptr)
    gam = tl.load(gam_ptr + i * n + j)
    ex = tl.load(x_ptr + 3 * j) - tl.load(x_ptr + 3 * i)
    ey = tl.load(x_ptr + 3 * j + 1) - tl.load(x_ptr + 3 * i + 1)
    ez = tl.load(x_ptr + 3 * j + 2) - tl.load(x_ptr + 3 * i + 2)
    f0 = ex * tl.load(geo_ptr + 9) + ey * tl.load(geo_ptr + 12) + ez * tl.load(geo_ptr + 15)
    f1 = ex * tl.load(geo_ptr + 10) + ey * tl.load(geo_ptr + 13) + ez * tl.load(geo_ptr + 16)
    f2 = ex * tl.load(geo_ptr + 11) + ey * tl.load(geo_ptr + 14) + ez * tl.load(geo_ptr + 17)
    lmin = ex * ex + ey * ey + ez * ez
    for k in range(0, ncand):
        u0 = f0 + tl.load(cand_ptr + 3 * k)
        u1 = f1 + tl.load(cand_ptr + 3 * k + 1)
        u2 = f2 + tl.load(cand_ptr + 3 * k + 2)
        vx = u0 * a0 + u1 * b0 + u2 * c0
        vy = u0 * a1 + u1 * b1 + u2 * c1
        vz = u0 * a2 + u1 * b2 + u2 * c2
        lmin = tl.minimum(lmin, vx * vx + vy * vy + vz * vz)
    near = tl.sqrt(lmin) + tl.load(pc_ptr + 4)
    lim = near * near + tl.load(pc_ptr + 5)
    tiny = tl.load(pc_ptr + 6)
    e2 = tl.load(pc_ptr + 1)
    inv_sqrt_pi = tl.load(pc_ptr)  # noqa: F841, read by the erf tangent
    nt = ex * 0.0
    for k in range(0, ncand):
        u0 = f0 + tl.load(cand_ptr + 3 * k)
        u1 = f1 + tl.load(cand_ptr + 3 * k + 1)
        u2 = f2 + tl.load(cand_ptr + 3 * k + 2)
        vx = u0 * a0 + u1 * b0 + u2 * c0
        vy = u0 * a1 + u1 * b1 + u2 * c1
        vz = u0 * a2 + u1 * b2 + u2 * c2
        nt += tl.where(vx * vx + vy * vy + vz * vz <= lim, 1.0, 0.0)
    acc = ex * 0.0
    gx = ex * 0.0
    gy = ex * 0.0
    gz = ex * 0.0
    v00 = ex * 0.0
    v01 = ex * 0.0
    v02 = ex * 0.0
    v10 = ex * 0.0
    v11 = ex * 0.0
    v12 = ex * 0.0
    v20 = ex * 0.0
    v21 = ex * 0.0
    v22 = ex * 0.0
    for k in range(0, ncand):
        u0 = f0 + tl.load(cand_ptr + 3 * k)
        u1 = f1 + tl.load(cand_ptr + 3 * k + 1)
        u2 = f2 + tl.load(cand_ptr + 3 * k + 2)
        vx = u0 * a0 + u1 * b0 + u2 * c0
        vy = u0 * a1 + u1 * b1 + u2 * c1
        vz = u0 * a2 + u1 * b2 + u2 * c2
        if vx * vx + vy * vy + vz * vz <= lim:
            for m in range(0, ncube):
                s0 = tl.load(cube_ptr + 3 * m)
                s1 = tl.load(cube_ptr + 3 * m + 1)
                s2 = tl.load(cube_ptr + 3 * m + 2)
                dx = vx + s0 * a0 + s1 * b0 + s2 * c0
                dy = vy + s0 * a1 + s1 * b1 + s2 * c1
                dz = vz + s0 * a2 + s1 * b2 + s2 * c2
                r2 = dx * dx + dy * dy + dz * dz
                ok = r2 >= tiny
                r2 = tl.where(ok, r2, 1.0)
                r = tl.sqrt(r2)
                phi = (libdevice.erfc(eta * r) - libdevice.erfc(gam * r)) / r
                acc += tl.where(ok, phi, 0.0)
                dphi = (e2 * (gam * tl.exp(-gam * gam * r2) - eta * tl.exp(-eta * eta * r2)) - phi) / r
                s = tl.where(ok, dphi / (r * nt), 0.0)
                gx += s * dx
                gy += s * dy
                gz += s * dz
                v00 += s * dx * dx
                v01 += s * dx * dy
                v02 += s * dx * dz
                v10 += s * dy * dx
                v11 += s * dy * dy
                v12 += s * dy * dz
                v20 += s * dz * dx
                v21 += s * dz * dy
                v22 += s * dz * dz
    a = acc / nt
    # reciprocal space from the phases: cos k.(x_j - x_i) = c_j c_i + s_j s_i, sin = s_j c_i - c_j s_i
    rec = ex * 0.0
    for m in range(0, ng):
        ci = tl.load(cs_ptr + (i * ng + m) * 2)
        si = tl.load(cs_ptr + (i * ng + m) * 2 + 1)
        cj = tl.load(cs_ptr + (j * ng + m) * 2)
        sj = tl.load(cs_ptr + (j * ng + m) * 2 + 1)
        wk = tl.load(gw_ptr + 2 * m)
        cs = wk * (cj * ci + sj * si)
        sn = wk * (sj * ci - cj * si)
        rec += cs
        kx = tl.load(gv_ptr + 3 * m)
        ky = tl.load(gv_ptr + 3 * m + 1)
        kz = tl.load(gv_ptr + 3 * m + 2)
        gx += -sn * kx
        gy += -sn * ky
        gz += -sn * kz
        fk = tl.load(gw_ptr + 2 * m + 1) * cs
        v00 += fk * kx * kx - cs
        v01 += fk * kx * ky
        v02 += fk * kx * kz
        v10 += fk * ky * kx
        v11 += fk * ky * ky - cs
        v12 += fk * ky * kz
        v20 += fk * kz * kx
        v21 += fk * kz * ky
        v22 += fk * kz * kz - cs
    a += rec
    if i == j:
        a += tl.load(diag_ptr + i) - 2.0 * eta * tl.load(pc_ptr)
    tl.store(mat_ptr + i * dim + j, a)
    tl.store(mat_ptr + j * dim + i, a)
    o = 12 * p
    tl.store(pg_ptr + o, gx)
    tl.store(pg_ptr + o + 1, gy)
    tl.store(pg_ptr + o + 2, gz)
    tl.store(pg_ptr + o + 3, v00)
    tl.store(pg_ptr + o + 4, v01)
    tl.store(pg_ptr + o + 5, v02)
    tl.store(pg_ptr + o + 6, v10)
    tl.store(pg_ptr + o + 7, v11)
    tl.store(pg_ptr + o + 8, v12)
    tl.store(pg_ptr + o + 9, v20)
    tl.store(pg_ptr + o + 10, v21)
    tl.store(pg_ptr + o + 11, v22)


@triton.jit
def _p_pair_gradient(ij_ptr, pg_ptr, q_ptr, w_ptr, g_ptr, vir_ptr, BLOCK: tl.constexpr):
    """Pair ``p`` (i <= j): the gradient in the positions and the cell of half the charge-weighted Ewald element,
    ``(q_i w_j + w_i q_j) / 2`` off the diagonal and ``q_i w_i / 2`` on it, from the derivatives in ``pg``."""
    p = tl.program_id(0).to(tl.int64)
    i = tl.load(ij_ptr + 2 * p)
    j = tl.load(ij_ptr + 2 * p + 1)
    qi = tl.load(q_ptr + i)
    qj = tl.load(q_ptr + j)
    wgt = tl.where(i == j, 0.5 * qi * tl.load(w_ptr + i), 0.5 * (qi * tl.load(w_ptr + j) + tl.load(w_ptr + i) * qj))
    o = 12 * p
    gx = wgt * tl.load(pg_ptr + o)
    gy = wgt * tl.load(pg_ptr + o + 1)
    gz = wgt * tl.load(pg_ptr + o + 2)
    _add(g_ptr, j, gx, gy, gz)
    _add(g_ptr, i, -gx, -gy, -gz)
    _virial(vir_ptr, wgt * tl.load(pg_ptr + o + 3), wgt * tl.load(pg_ptr + o + 4), wgt * tl.load(pg_ptr + o + 5),
            wgt * tl.load(pg_ptr + o + 6), wgt * tl.load(pg_ptr + o + 7), wgt * tl.load(pg_ptr + o + 8),
            wgt * tl.load(pg_ptr + o + 9), wgt * tl.load(pg_ptr + o + 10), wgt * tl.load(pg_ptr + o + 11))


@triton.jit
def _p_solve(mat_ptr, rhs_ptr, tot_ptr, fac_ptr, sol_ptr, uu_ptr, w_ptr, dxl_ptr, glog_ptr, e_ptr, dim, n,
             FACTOR: tl.constexpr, BLOCK: tl.constexpr):
    """The charge equations (with ``FACTOR`` by an L D L^T factorization in ``fac`` as :func:`_solve_small`; else
    ``sol`` and ``uu`` hold the solutions already): ``sol`` for the right-hand side and ``uu`` for the constraint
    charges alone. Then ``w = q - 2 u``, the charge term of the count gradient ``-(q - u) dx/dlogCN`` and the
    electrostatic energy ``-q.x/2 + lambda.Q/2``."""
    if FACTOR:
        for j in range(0, dim):
            d = tl.load(mat_ptr + j * dim + j)
            for k in range(0, j):
                l = tl.load(fac_ptr + j * dim + k)
                d -= l * l * tl.load(fac_ptr + k * dim + k)
            tl.store(fac_ptr + j * dim + j, d)
            for i in range(j + 1, dim):
                v = tl.load(mat_ptr + i * dim + j)
                for k in range(0, j):
                    v -= tl.load(fac_ptr + i * dim + k) * tl.load(fac_ptr + j * dim + k) * tl.load(
                        fac_ptr + k * dim + k)
                tl.store(fac_ptr + i * dim + j, v / d)
        for i in range(0, dim):
            v = tl.load(rhs_ptr + i)
            u = tl.load(tot_ptr + i)
            for k in range(0, i):
                l = tl.load(fac_ptr + i * dim + k)
                v -= l * tl.load(sol_ptr + k)
                u -= l * tl.load(uu_ptr + k)
            tl.store(sol_ptr + i, v)
            tl.store(uu_ptr + i, u)
        for i in range(0, dim):
            dd = tl.load(fac_ptr + i * dim + i)
            tl.store(sol_ptr + i, tl.load(sol_ptr + i) / dd)
            tl.store(uu_ptr + i, tl.load(uu_ptr + i) / dd)
        for r in range(0, dim):
            i = dim - 1 - r
            v = tl.load(sol_ptr + i)
            u = tl.load(uu_ptr + i)
            for k in range(i + 1, dim):
                l = tl.load(fac_ptr + k * dim + i)
                v -= l * tl.load(sol_ptr + k)
                u -= l * tl.load(uu_ptr + k)
            tl.store(sol_ptr + i, v)
            tl.store(uu_ptr + i, u)
    for i in range(0, n):
        q = tl.load(sol_ptr + i)
        u = tl.load(uu_ptr + i)
        tl.store(w_ptr + i, q - 2.0 * u)
        tl.store(glog_ptr + i, tl.load(glog_ptr + i) - (q - u) * tl.load(dxl_ptr + i))
    acc = tl.load(sol_ptr) * 0.0
    for k in range(0, n):
        acc += -tl.load(sol_ptr + k) * tl.load(rhs_ptr + k)
    for k in range(n, dim):
        acc += tl.load(sol_ptr + k) * tl.load(rhs_ptr + k)
    tl.store(e_ptr + 8, 0.5 * acc)


@triton.jit
def _p_dispersion(x_ptr, ij_ptr, tr_ptr, geo_ptr, s_ptr, ds_ptr, zz_ptr, dq_ptr, r06_ptr, r08_ptr, g_ptr, vir_ptr,
                  glog_ptr, e_ptr, pc_ptr, n, ntr, cut2, NFREQ: tl.constexpr, BLOCK: tl.constexpr):
    """Pair ``p`` (i <= j) over the dispersion translations: the dispersion energy (an atom and its own images at full
    weight), its gradient in the positions, the cell and the coordination numbers."""
    p = tl.program_id(0).to(tl.int64)
    i = tl.load(ij_ptr + 2 * p)
    j = tl.load(ij_ptr + 2 * p + 1)
    a0, a1, a2, b0, b1, b2, c0, c1, c2 = _cell(geo_ptr)
    ex = tl.load(x_ptr + 3 * j) - tl.load(x_ptr + 3 * i)
    ey = tl.load(x_ptr + 3 * j + 1) - tl.load(x_ptr + 3 * i + 1)
    ez = tl.load(x_ptr + 3 * j + 2) - tl.load(x_ptr + 3 * i + 2)
    c6 = ex * 0.0
    di = ex * 0.0
    dj = ex * 0.0
    for w in tl.static_range(NFREQ):
        si = tl.load(s_ptr + i * NFREQ + w)
        sj = tl.load(s_ptr + j * NFREQ + w)
        c6 += si * sj
        di += tl.load(ds_ptr + i * NFREQ + w) * sj
        dj += tl.load(ds_ptr + j * NFREQ + w) * si
    k = i * n + j
    zc = tl.load(zz_ptr + k)
    r06 = tl.load(r06_ptr + k)
    r08 = tl.load(r08_ptr + k)
    q2 = 2.0 * tl.load(dq_ptr + k)
    tiny = tl.load(pc_ptr + 6)
    sf = ex * 0.0
    gx = ex * 0.0
    gy = ex * 0.0
    gz = ex * 0.0
    v00 = ex * 0.0
    v01 = ex * 0.0
    v02 = ex * 0.0
    v10 = ex * 0.0
    v11 = ex * 0.0
    v12 = ex * 0.0
    v20 = ex * 0.0
    v21 = ex * 0.0
    v22 = ex * 0.0
    for t in range(0, ntr):
        s0 = tl.load(tr_ptr + 3 * t)
        s1 = tl.load(tr_ptr + 3 * t + 1)
        s2 = tl.load(tr_ptr + 3 * t + 2)
        dx = ex + s0 * a0 + s1 * b0 + s2 * c0
        dy = ey + s0 * a1 + s1 * b1 + s2 * c1
        dz = ez + s0 * a2 + s1 * b2 + s2 * c2
        r2 = dx * dx + dy * dy + dz * dz
        ok = (r2 > tiny) & (r2 < cut2)
        r2 = tl.where(ok, r2, 1.0)
        r6 = r2 * r2 * r2
        t6 = 1.0 / (r6 + r06)
        t8 = 1.0 / (r6 * r2 + r08)
        sf += tl.where(ok, t6 + q2 * t8, 0.0)
        s = tl.where(ok, zc * c6 * (6.0 * r6 / r2 * t6 * t6 + 8.0 * q2 * r6 * t8 * t8), 0.0)
        gx += s * dx
        gy += s * dy
        gz += s * dz
        v00 += s * dx * dx
        v01 += s * dx * dy
        v02 += s * dx * dz
        v10 += s * dy * dx
        v11 += s * dy * dy
        v12 += s * dy * dz
        v20 += s * dz * dx
        v21 += s * dz * dy
        v22 += s * dz * dz
    tl.atomic_add(e_ptr + 5, -zc * c6 * sf, sem="relaxed")
    tl.atomic_add(glog_ptr + i, tl.where(i == j, -zc * sf * (di + dj), -zc * sf * di), sem="relaxed")
    tl.atomic_add(glog_ptr + j, tl.where(i == j, 0.0 * sf, -zc * sf * dj), sem="relaxed")
    _add(g_ptr, j, gx, gy, gz)
    _add(g_ptr, i, -gx, -gy, -gz)
    _virial(vir_ptr, v00, v01, v02, v10, v11, v12, v20, v21, v22)


@triton.jit
def _p_repulsion(x_ptr, ij_ptr, sh_ptr, par_ptr, geo_ptr, g_ptr, vir_ptr, e_ptr, BLOCK: tl.constexpr):
    """Repulsion pair ``p`` (atom i, atom j in the image of lattice shift ``sh``) with the parameters (prefactor,
    exponent, cut)."""
    p = tl.program_id(0).to(tl.int64)
    i = tl.load(ij_ptr + 2 * p)
    j = tl.load(ij_ptr + 2 * p + 1)
    a0, a1, a2, b0, b1, b2, c0, c1, c2 = _cell(geo_ptr)
    s0 = tl.load(sh_ptr + 3 * p)
    s1 = tl.load(sh_ptr + 3 * p + 1)
    s2 = tl.load(sh_ptr + 3 * p + 2)
    dx = tl.load(x_ptr + 3 * j) - tl.load(x_ptr + 3 * i) + s0 * a0 + s1 * b0 + s2 * c0
    dy = tl.load(x_ptr + 3 * j + 1) - tl.load(x_ptr + 3 * i + 1) + s0 * a1 + s1 * b1 + s2 * c1
    dz = tl.load(x_ptr + 3 * j + 2) - tl.load(x_ptr + 3 * i + 2) + s0 * a2 + s1 * b2 + s2 * c2
    r = tl.sqrt(dx * dx + dy * dy + dz * dz)
    if r < tl.load(par_ptr + 3 * p + 2):
        mm = tl.load(par_ptr + 3 * p + 1)
        sr = tl.sqrt(r)
        en = tl.load(par_ptr + 3 * p) / r * tl.exp(-mm * r * sr)
        tl.atomic_add(e_ptr + 3, en, sem="relaxed")
        s = -en * (1.0 / r + 1.5 * mm * sr) / r
        _add(g_ptr, j, s * dx, s * dy, s * dz)
        _add(g_ptr, i, -s * dx, -s * dy, -s * dz)
        _virial(vir_ptr, s * dx * dx, s * dx * dy, s * dx * dz, s * dy * dx, s * dy * dy, s * dy * dz, s * dz * dx,
                s * dz * dy, s * dz * dz)


@triton.jit
def _p_fold(x_ptr, gb_ptr, glb_ptr, base_ptr, g_ptr, glog_ptr, vir_ptr, n, BLOCK: tl.constexpr):
    """Atom or image ``p`` of the rows: its gradient onto its atom of the cell, and its share of the cell gradient
    (position times gradient, the rows being sums over differences)."""
    p = tl.program_id(0).to(tl.int64)
    k = p
    if p >= n:
        k = tl.load(base_ptr + p - n)
    fx = tl.load(gb_ptr + 3 * p)
    fy = tl.load(gb_ptr + 3 * p + 1)
    fz = tl.load(gb_ptr + 3 * p + 2)
    _add(g_ptr, k, fx, fy, fz)
    tl.atomic_add(glog_ptr + k, tl.load(glb_ptr + p), sem="relaxed")
    px = tl.load(x_ptr + 3 * p)
    py = tl.load(x_ptr + 3 * p + 1)
    pz = tl.load(x_ptr + 3 * p + 2)
    _virial(vir_ptr, px * fx, px * fy, px * fz, py * fx, py * fy, py * fz, pz * fx, pz * fy, pz * fz)


@triton.jit
def _p_finish(pos_ptr, cell_ptr, g_ptr, vir_ptr, out_ptr, bohr_ptr, n, BLOCK: tl.constexpr):
    """The position gradient in Hartree per Angstrom (``out[:3n]``) and the cell gradient at fixed positions
    (``out[3n:]``): ``cell^-T vir`` (the gradient at fixed fractional coordinates) minus fractional coordinates times
    position gradient."""
    bohr = tl.load(bohr_ptr)
    a0 = tl.load(cell_ptr)
    a1 = tl.load(cell_ptr + 1)
    a2 = tl.load(cell_ptr + 2)
    b0 = tl.load(cell_ptr + 3)
    b1 = tl.load(cell_ptr + 4)
    b2 = tl.load(cell_ptr + 5)
    c0 = tl.load(cell_ptr + 6)
    c1 = tl.load(cell_ptr + 7)
    c2 = tl.load(cell_ptr + 8)
    det = a0 * (b1 * c2 - b2 * c1) - a1 * (b0 * c2 - b2 * c0) + a2 * (b0 * c1 - b1 * c0)
    m00 = (b1 * c2 - b2 * c1) / det
    m01 = (a2 * c1 - a1 * c2) / det
    m02 = (a1 * b2 - a2 * b1) / det
    m10 = (b2 * c0 - b0 * c2) / det
    m11 = (a0 * c2 - a2 * c0) / det
    m12 = (a2 * b0 - a0 * b2) / det
    m20 = (b0 * c1 - b1 * c0) / det
    m21 = (a1 * c0 - a0 * c1) / det
    m22 = (a0 * b1 - a1 * b0) / det
    w00 = tl.load(vir_ptr)
    w01 = tl.load(vir_ptr + 1)
    w02 = tl.load(vir_ptr + 2)
    w10 = tl.load(vir_ptr + 3)
    w11 = tl.load(vir_ptr + 4)
    w12 = tl.load(vir_ptr + 5)
    w20 = tl.load(vir_ptr + 6)
    w21 = tl.load(vir_ptr + 7)
    w22 = tl.load(vir_ptr + 8)
    # cell^-T vir: element (a, b) = sum_c inv[c][a] vir[c][b]
    h00 = m00 * w00 + m10 * w10 + m20 * w20
    h01 = m00 * w01 + m10 * w11 + m20 * w21
    h02 = m00 * w02 + m10 * w12 + m20 * w22
    h10 = m01 * w00 + m11 * w10 + m21 * w20
    h11 = m01 * w01 + m11 * w11 + m21 * w21
    h12 = m01 * w02 + m11 * w12 + m21 * w22
    h20 = m02 * w00 + m12 * w10 + m22 * w20
    h21 = m02 * w01 + m12 * w11 + m22 * w21
    h22 = m02 * w02 + m12 * w12 + m22 * w22
    for i in range(0, n):
        gx = tl.load(g_ptr + 3 * i) / bohr
        gy = tl.load(g_ptr + 3 * i + 1) / bohr
        gz = tl.load(g_ptr + 3 * i + 2) / bohr
        tl.store(out_ptr + 3 * i, gx)
        tl.store(out_ptr + 3 * i + 1, gy)
        tl.store(out_ptr + 3 * i + 2, gz)
        px = tl.load(pos_ptr + 3 * i)
        py = tl.load(pos_ptr + 3 * i + 1)
        pz = tl.load(pos_ptr + 3 * i + 2)
        f0 = px * m00 + py * m10 + pz * m20
        f1 = px * m01 + py * m11 + pz * m21
        f2 = px * m02 + py * m12 + pz * m22
        h00 -= f0 * gx
        h01 -= f0 * gy
        h02 -= f0 * gz
        h10 -= f1 * gx
        h11 -= f1 * gy
        h12 -= f1 * gz
        h20 -= f2 * gx
        h21 -= f2 * gy
        h22 -= f2 * gz
    o = 3 * n
    tl.store(out_ptr + o, h00)
    tl.store(out_ptr + o + 1, h01)
    tl.store(out_ptr + o + 2, h02)
    tl.store(out_ptr + o + 3, h10)
    tl.store(out_ptr + o + 4, h11)
    tl.store(out_ptr + o + 5, h12)
    tl.store(out_ptr + o + 6, h20)
    tl.store(out_ptr + o + 7, h21)
    tl.store(out_ptr + o + 8, h22)


class PeriodicPlan:
    """One energy-and-gradient evaluation of a bound periodic GFN-FF model on the CPU or a GPU.

    Calling it with positions ``(N, 3)`` and the cell ``(3, 3)`` in Angstrom returns the nine term energies (Hartree,
    in the order of ``gfnff.TERMS``), the gradient of their sum in the positions (Hartree per Angstrom) and in the
    cell at fixed positions (Hartree per Angstrom). The Ewald splitting parameter and the translation lists follow the
    cell they were made for and are made again when it changes.
    """

    def __init__(self, model):
        from . import gfnff_periodic as per
        from . import gfnfflib as lib

        f64, i64 = torch.float64, torch.int64
        dev = self.device = model.rcov.device
        self.model = model
        hb = per.kernel_tables(model)
        n = self.n = int(model.num_atoms)
        base = model.x_base.contiguous()
        ne = self.ne = n + int(base.shape[0])
        col = lambda *ts: torch.stack([t.to(f64) for t in ts], 1).contiguous()  # noqa: E731
        z = lambda *shape: torch.zeros(*shape, dtype=f64, device=dev)  # noqa: E731
        self.bohr = float(lib.BOHR)
        # the accumulators zeroed by the first kernel, in one tensor
        sizes = (3 * ne, ne, 3 * n, n, n, 9, 9)
        self.work = z(sum(sizes))
        parts, o = [], 0
        for s in sizes:
            parts.append(self.work[o:o + s])
            o += s
        self.gb, self.glb, self.g, self.glog, self.cnt, self.vir, self.e = parts
        self.x, self.geo, self.lcn, self.lcne, self.dl, self.dxl = z(ne, 3), z(19), z(n), z(ne), z(n), z(n)
        self.out = z(3 * n + 9)
        self.eta = z(1)
        con, tot = per._constraints(model, f64, dev)
        nc = int(tot.shape[0])
        self.dim = n + nc
        self.mat, self.rhs, self.sol = z(self.dim, self.dim), z(self.dim), z(self.dim)
        self.uu, self.w = z(self.dim), z(n)
        self.mat[n:, :n] = con
        self.mat[:n, n:] = con.T
        self.rhs[n:] = tot
        self.tot = z(self.dim)
        self.tot[n:] = tot
        self.q = self.sol[:n]
        nfreq, nref = int(model.d_alpha.shape[0]), int(model.d_ref.shape[1])
        if int(model.d_alpha.shape[1]) != nref:
            raise cpu_kernels.Unsupported("reference tables of different widths")
        if bool((model.d_w < 0).any()):
            raise cpu_kernels.Unsupported("negative integration weights")
        self.s, self.ds = z(n, nfreq), z(n, nfreq)
        sw = torch.sqrt(3.0 / math.pi * model.d_w.to(f64)).contiguous()
        iu = torch.triu_indices(n, n, device=dev)
        pairs = iu.T.contiguous()
        npair = int(pairs.shape[0])
        c, pc = CONST.to(dev), PCONST.to(dev)
        self.cell_ref = None
        self.cntr = model.cn_shift.to(f64).contiguous()
        self.dtr = z(1, 3)
        cube = torch.tensor(list(itertools.product(range(-2, 3), repeat=3)), dtype=f64, device=dev)
        cand = torch.tensor(list(itertools.product(range(-1, 2), repeat=3)), dtype=f64, device=dev)
        self.mill = cube[(cube != 0).any(1)].contiguous()
        ng = int(self.mill.shape[0])
        self.gv, self.gw, self.cs = z(ng, 3), z(ng, 2), z(n, ng, 2)
        self.pg = z(npair, 12)
        gpu = dev.type == "cuda"
        P = _recorded(_GpuLaunch if gpu else cpu_kernels.prepare)
        self.pos, self.cell = z(n, 3), z(3, 3)
        self.bohr_t = torch.tensor([self.bohr], dtype=f64, device=dev)
        nw = int(self.work.numel())
        self.prep = P(_p_prepare, (max(ne, nw),), self.pos, self.cell, model.x_wrap.to(f64).contiguous(), base,
                      model.x_shift.to(f64).contiguous(), self.x, self.geo, self.work, self.bohr_t, n, ne, nw, BLOCK=1)
        rcov = model.rcov.to(f64).contiguous()
        self.cg = z(npair, 9)
        counts = P(_p_counts, (npair,), self.x, pairs, self.cntr, self.geo, rcov, self.cnt, self.cg, c,
                   int(self.cntr.shape[0]), BLOCK=1)
        atoms = P(_atom_terms, (n,), self.cnt, self.lcn, self.dl, self.rhs, self.dxl, model.e_x0.contiguous(),
                  model.e_kappa.contiguous(), model.d_ref.contiguous(), model.d_valid.sum(1).to(i64).contiguous(),
                  model.d_alpha.to(f64).contiguous(), sw, self.s, self.ds, c, n, NREF=nref, NFREQ=nfreq, BLOCK=1)
        expand = P(_p_expand, (ne,), self.lcn, self.lcne, base, n, ne, BLOCK=1)
        gam = model.e_g.to(f64).contiguous()
        phases = P(_p_phases, (n,), self.x, self.gv, self.cs, ng, BLOCK=1)
        ewald = P(_p_ewald, (npair,), self.x, pairs, self.geo, cand, cube, self.gv, self.gw, self.cs, gam,
                  model.e_diag.contiguous(), self.eta, self.mat, self.pg, pc, n, self.dim, ng, int(cand.shape[0]),
                  int(cube.shape[0]), BLOCK=1)
        small = self.dim <= SMALL_SOLVE and not gpu
        self.fac = z(self.dim, self.dim)
        solve = P(_p_solve, (1,), self.mat, self.rhs, self.tot, self.fac, self.sol, self.uu, self.w, self.dxl,
                  self.glog, self.e, self.dim, n, FACTOR=small, BLOCK=1)
        runs = []
        acc = model.b_acc
        rad = torch.cat([model.arad, model.arad[base]])
        runs.append(P(_bonds, (int(model.b_ij.shape[0]),), self.x, model.b_ij.contiguous(),
                      col(model.b_fp, model.b_ra, model.b_ci, model.b_cj, model.b_sh, model.b_k, model.b_al), self.lcne,
                      _pointer(acc[:, 0], int(model.b_ij.shape[0])), acc[:, 1:].contiguous(),
                      model.b_accd.contiguous(), self.gb, self.glb, self.e, c, BLOCK=1))
        ai = model.a_idx
        lin = ((model.a_t0 - math.pi).abs() < 1e-6).to(f64)
        runs.append(P(_angles, (int(ai.shape[0]),), self.x, ai.contiguous(),
                      col(torch.cos(model.a_t0), lin, model.a_k, rad[ai[:, 0]] + rad[ai[:, 1]],
                          rad[ai[:, 0]] + rad[ai[:, 2]]), self.gb, self.e, c, BLOCK=1))
        ti = model.t_idx
        runs.append(P(_torsions, (int(ti.shape[0]),), self.x, ti.contiguous(),
                      col(model.t_n, model.t_p0, model.t_k, rad[ti[:, 0]] + rad[ti[:, 2]],
                          rad[ti[:, 2]] + rad[ti[:, 1]], rad[ti[:, 1]] + rad[ti[:, 3]]), self.gb, self.e, c, BLOCK=1))
        ii = model.i_idx
        runs.append(P(_inversions, (int(ii.shape[0]),), self.x, ii.contiguous(),
                      col(model.i_neg, model.i_c0, model.i_k, rad[ii[:, 0]] + rad[ii[:, 1]],
                          rad[ii[:, 1]] + rad[ii[:, 2]], rad[ii[:, 1]] + rad[ii[:, 3]]), self.gb, self.e, c, BLOCK=1))
        runs.append(P(_triples, (int(model.m_idx.shape[0]),), self.x, model.m_idx.contiguous(),
                      model.m_c.contiguous(), self.gb, self.e, BLOCK=1))
        runs.append(P(_p_repulsion, (int(model.r_ij.shape[0]),), self.x, model.r_ij.contiguous(),
                      model.r_sh.to(f64).contiguous(), col(model.r_c, model.r_m, model.r_cut), self.geo, self.g,
                      self.vir, self.e, BLOCK=1))
        cut, cut3, xcut = float(lib.HB_CUT), float(lib.HB_CUT3), float(lib.XB_CUT)
        runs.append(P(_hb_bonded, (int(hb["bb"].shape[0]),), self.x, hb["bb"], hb["bb_par"], hb["nbp"], hb["nbs"],
                      hb["aux"], hb["yp"], hb["ys"], self.gb, self.e, c, cut, BLOCK=1))
        runs.append(P(_hb_loose, (int(hb["bl"].shape[0]),), self.x, hb["bl"], hb["bl_par"], self.gb, self.e, cut, cut3,
                      BLOCK=1))
        runs.append(P(_halogen_bonds, (int(hb["xb"].shape[0]),), self.x, hb["xb"], hb["xb_par"], self.gb, self.e,
                      xcut, BLOCK=1))
        self.disp = P(_p_dispersion, (npair,), self.x, pairs, self.dtr, self.geo, self.s, self.ds,
                      model.d_zz.to(f64).contiguous(), model.d_q.to(f64).contiguous(),
                      model.d_r06.to(f64).contiguous(), model.d_r08.to(f64).contiguous(), self.g, self.vir, self.glog,
                      self.e, pc, n, 0, float(per.DISP_CUT) ** 2, NFREQ=nfreq, BLOCK=1)
        ewg = P(_p_pair_gradient, (npair,), pairs, self.pg, self.sol, self.w, self.g, self.vir, BLOCK=1)
        fold = P(_p_fold, (ne,), self.x, self.gb, self.glb, base, self.g, self.glog, self.vir, n, BLOCK=1)
        third = P(_p_count_gradient, (npair,), pairs, self.cg, rcov, self.dl, self.glog, self.g, self.vir, c, BLOCK=1)
        finish = P(_p_finish, (1,), self.pos, self.cell, self.g, self.vir, self.out, self.bohr_t, n, BLOCK=1)
        self.small = small
        self.parts = (counts, atoms, expand, phases, ewald, solve, runs, ewg, fold, third, finish)
        self.gpu, self.P = gpu, P
        self.npair, self.pairs, self.nfreq = npair, pairs, nfreq
        self.seq = None
        self.tangent = None
        self.keep = (pairs, rcov, gam, sw, c, pc, cube, cand, rad, hb, base)

    def _cell_lists(self, cell: Tensor) -> None:
        """The Ewald splitting parameter, the reciprocal vectors and the dispersion translations of ``cell``."""
        from . import gfnff_periodic as per
        from . import gfnfflib as lib

        c = cell.detach().to(device="cpu", dtype=torch.float64)
        if self.cell_ref is not None and torch.equal(c, self.cell_ref):
            return
        self.cell_ref = c.clone()
        cn = c.numpy()
        cellb = cn / lib.BOHR
        self.eta.fill_(per.ewald_eta(cellb, per.mean_gamma(self.model)))
        rec = 2.0 * math.pi * torch.linalg.inv(torch.as_tensor(cellb)).T
        gv = self.mill.cpu() @ rec
        k2 = (gv ** 2).sum(1)
        eta = float(self.eta[0])
        vol = abs(float(np.linalg.det(cellb)))
        # each reciprocal vector's weight and the factor of the strain derivative of that weight
        gw = torch.stack([4.0 * math.pi / vol * torch.exp(-k2 / (4.0 * eta ** 2)) / k2,
                          2.0 * (1.0 / (4.0 * eta ** 2) + 1.0 / k2)], 1)
        self.gv.copy_(gv.to(self.gv.device))
        self.gw.copy_(gw.to(self.gw.device))
        # every translation of the capped list; the cutoff applies to the pair distances only
        sh = np.asarray(lib.translation_list(cn, per.DISP_LIST2, cap=True), dtype=float)
        self.dtr = torch.as_tensor(sh, dtype=torch.float64, device=self.device).contiguous()
        counts, atoms, expand, phases, ewald, solve, runs, ewg, fold, third, finish = self.parts
        P = self.P
        m = self.model
        self.disp = P(_p_dispersion, (self.npair,), self.x, self.pairs, self.dtr, self.geo, self.s, self.ds,
                      m.d_zz.to(torch.float64).contiguous(), m.d_q.to(torch.float64).contiguous(),
                      m.d_r06.to(torch.float64).contiguous(), m.d_r08.to(torch.float64).contiguous(), self.g, self.vir,
                      self.glog, self.e, PCONST.to(self.device), self.n, int(self.dtr.shape[0]),
                      float(per.DISP_CUT) ** 2, NFREQ=self.nfreq, BLOCK=1)
        self._keep_d = (m.d_zz, m.d_q, m.d_r06, m.d_r08)
        second = [*runs, self.disp, ewg]
        first = [self.prep, counts, atoms, expand, phases, ewald]
        if self.gpu:
            self.stages = [_Sequence(first), None, _Sequence([solve, *second, fold, third, finish])]
        elif self.small:
            self.stages = [cpu_kernels.Sequence([*first, solve, *second, fold, third, finish])]
        else:
            self.stages = [cpu_kernels.Sequence(first), None, cpu_kernels.Sequence([solve, *second, fold, third,
                                                                                     finish])]

    def __call__(self, positions: Tensor, cell: Tensor) -> tuple:
        self._cell_lists(cell)
        self.pos.copy_(positions.detach())
        self.cell.copy_(cell.detach())
        for stage in self.stages:
            if stage is None:
                sol = torch.linalg.solve(self.mat, torch.stack([self.rhs, self.tot], 1))
                self.sol.copy_(sol[:, 0])
                self.uu.copy_(sol[:, 1])
            else:
                stage()
        n = self.n
        return self.e.clone(), self.out[:3 * n].view(n, 3).clone(), self.out[3 * n:].view(3, 3).clone()

    def tangents(self) -> None:
        """Make the tangent buffers and the launches of the tangent kernels that do not follow the cell (once)."""
        if self.tangent is not None:
            return
        z = torch.zeros_like
        n, ne = self.n, self.ne
        self.twork = z(self.work)
        sizes = (3 * ne, ne, 3 * n, n, n, 9, 9)
        parts, o = [], 0
        for s in sizes:
            parts.append(self.twork[o:o + s])
            o += s
        self.pos_t, self.cell_t, self.out_t = z(self.pos), z(self.cell), z(self.out)
        self.gv_t, self.gw_t = z(self.gv), z(self.gw)
        self.mat_t, self.rhs_t, self.sol_t, self.uu_t = z(self.mat), z(self.rhs), z(self.sol), z(self.uu)
        pairs = list(zip((self.gb, self.glb, self.g, self.glog, self.cnt, self.vir, self.e), parts))
        pairs += [(self.pos, self.pos_t), (self.cell, self.cell_t), (self.out, self.out_t), (self.gv, self.gv_t),
                  (self.gw, self.gw_t), (self.mat, self.mat_t), (self.rhs, self.rhs_t), (self.sol, self.sol_t),
                  (self.uu, self.uu_t)]
        pairs += [(t, z(t)) for t in (self.x, self.geo, self.lcn, self.lcne, self.dl, self.dxl, self.w, self.fac,
                                      self.s, self.ds, self.cs, self.cg, self.pg)]
        self.twins = _twins(pairs)
        self.dxl_t = self.twins[_key(self.dxl)]
        self.tangent = {}

    def _tangent_stages(self) -> list:
        """The stages of the tangent kernels for the current dispersion translations."""
        if self.tangent.get("disp") is self.disp:
            return self.tangent["stages"]
        P, tw = self.P, self.twins
        counts, atoms, expand, phases, ewald, solve, runs, ewg, fold, third, finish = self.parts
        first = [_twin(P, run, tw) for run in (self.prep, counts, atoms, expand, phases, ewald)]
        second = [_twin(P, run, tw) for run in (solve, *runs, self.disp, ewg, fold, third, finish)]
        if self.small:
            stages = [_launches(first + second, self.gpu, self.n)]
        else:
            stages = [_launches(first, self.gpu, self.n), None, _launches(second, self.gpu, self.n)]
        self.tangent.update(disp=self.disp, stages=stages)
        return stages

    def _table_tangents(self, cell: Tensor, direction: Tensor) -> None:
        """The tangents of the reciprocal vectors and their weights along a change ``direction`` of the cell (the
        splitting parameter fixed, as in the gradient)."""
        from . import gfnfflib as lib

        f64 = torch.float64
        cb = cell.detach().to(device="cpu", dtype=f64) / lib.BOHR
        db = direction.detach().to(device="cpu", dtype=f64) / lib.BOHR
        inv = torch.linalg.inv(cb)
        rec = 2.0 * math.pi * inv.T
        rec_t = -2.0 * math.pi * (inv @ db @ inv).T
        mill = self.mill.cpu()
        gv, gv_t = mill @ rec, mill @ rec_t
        k2 = (gv ** 2).sum(1)
        k2_t = 2.0 * (gv * gv_t).sum(1)
        eta = float(self.eta[0])
        vol_t = torch.trace(inv @ db)
        w0 = 4.0 * math.pi / abs(float(torch.linalg.det(cb))) * torch.exp(-k2 / (4.0 * eta ** 2)) / k2
        self.gv_t.copy_(gv_t.to(self.device))
        self.gw_t.copy_(torch.stack([w0 * (-vol_t - k2_t / (4.0 * eta ** 2) - k2_t / k2),
                                     -2.0 * k2_t / (k2 * k2)], 1).to(self.device))

    def hvp(self, positions: Tensor, cell: Tensor, dpos: Tensor, dcell: Tensor) -> tuple:
        """The derivatives of the gradients in the positions and in the cell (Hartree per Angstrom) at
        ``positions`` ``(N, 3)`` and ``cell`` ``(3, 3)`` along the changes ``dpos`` and ``dcell`` (Angstrom): the
        Hessian of the energy in both times the direction."""
        self.tangents()
        self._cell_lists(cell)
        stages = self._tangent_stages()
        self._table_tangents(cell, dcell)
        self.pos.copy_(positions.detach())
        self.cell.copy_(cell.detach())
        self.pos_t.copy_(dpos.detach())
        self.cell_t.copy_(dcell.detach())
        self.twork.zero_()
        for stage in stages:
            if stage is None:
                lu = torch.linalg.lu_factor(self.mat)
                sol = torch.linalg.lu_solve(*lu, torch.stack([self.rhs, self.tot], 1))
                self.sol.copy_(sol[:, 0])
                self.uu.copy_(sol[:, 1])
                rhs = torch.stack([self.rhs_t - self.mat_t @ self.sol, -(self.mat_t @ self.uu)], 1)
                sol_t = torch.linalg.lu_solve(*lu, rhs)
                self.sol_t.copy_(sol_t[:, 0])
                self.uu_t.copy_(sol_t[:, 1])
            else:
                stage()
        n = self.n
        return self.out_t[:3 * n].view(n, 3).clone(), self.out_t[3 * n:].view(3, 3).clone()


def available(device: torch.device = torch.device("cpu")) -> bool:
    """Whether the kernels can run on ``device``: a C++ compiler for the CPU, Triton for a GPU."""
    if device.type == "cuda":
        try:
            import triton  # noqa: F401
        except ImportError:
            return False
        return True
    return cpu_kernels.available()


__all__ = ["Plan", "PeriodicPlan", "available", "CONST", "PCONST"]
