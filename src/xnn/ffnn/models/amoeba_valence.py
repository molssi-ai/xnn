"""The valence terms of AMOEBA and HIPPO on fused kernels: bonds, angles (in-plane included), stretch-bends,
Urey-Bradley terms, out-of-plane bends and torsions of a step plan, from its bonded vectors.

One launch forms every term's energy and adds it to the per-atom rows of its term; one launch forms the
gradient of a weighted sum of those rows with respect to every bonded vector. Each vector belongs to one
term, so the gradient is stored, not added. The arithmetic is float64 in either precision of the model,
with the inexact constants from a float64 table (a Triton float literal is float32). The forms are those of
:meth:`~xnn.ffnn.models.amoeba.Amoeba._valence`; the kernels serve evaluations with first derivatives.
"""
from typing import Optional

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import once_differentiable
from triton.language.extra import libdevice

#: The rows of the output, in the order of the valence terms of the model.
TERMS = ("e_bond", "e_angle", "e_strbnd", "e_urey", "e_opbend", "e_torsion", "e_pitors", "e_tortor")
#: The entries of the float64 table of constants.
SCALARS = ("tiny", "deg", "bond_cubic", "bond_quartic", "angle_cubic", "angle_quartic", "angle_pentic",
           "angle_sextic", "opbend_cubic", "opbend_quartic", "opbend_pentic", "opbend_sextic", "angle_bound",
           "opbend_bound")
BLOCK = 128


@triton.jit
def _ld(ptr, row, m):
    x = tl.load(ptr + row * 3, mask=m, other=0.0).to(tl.float64)
    y = tl.load(ptr + row * 3 + 1, mask=m, other=0.0).to(tl.float64)
    z = tl.load(ptr + row * 3 + 2, mask=m, other=0.0).to(tl.float64)
    return x, y, z


@triton.jit
def _st(ptr, vir_ptr, row, m, x, y, z, rx, ry, rz, VIRIAL: tl.constexpr):
    """Store the gradient ``(x, y, z)`` of the vectors ``r`` of a block; with ``VIRIAL`` also add the
    block's sum of ``r (x) gradient`` to the strain derivative ``(3, 3)`` (masked lanes add zeros)."""
    tl.store(ptr + row * 3, x.to(ptr.dtype.element_ty), mask=m)
    tl.store(ptr + row * 3 + 1, y.to(ptr.dtype.element_ty), mask=m)
    tl.store(ptr + row * 3 + 2, z.to(ptr.dtype.element_ty), mask=m)
    if VIRIAL:
        tl.atomic_add(vir_ptr + 0, tl.sum(rx * x, 0), sem="relaxed")
        tl.atomic_add(vir_ptr + 1, tl.sum(rx * y, 0), sem="relaxed")
        tl.atomic_add(vir_ptr + 2, tl.sum(rx * z, 0), sem="relaxed")
        tl.atomic_add(vir_ptr + 3, tl.sum(ry * x, 0), sem="relaxed")
        tl.atomic_add(vir_ptr + 4, tl.sum(ry * y, 0), sem="relaxed")
        tl.atomic_add(vir_ptr + 5, tl.sum(ry * z, 0), sem="relaxed")
        tl.atomic_add(vir_ptr + 6, tl.sum(rz * x, 0), sem="relaxed")
        tl.atomic_add(vir_ptr + 7, tl.sum(rz * y, 0), sem="relaxed")
        tl.atomic_add(vir_ptr + 8, tl.sum(rz * z, 0), sem="relaxed")


@triton.jit
def _dot(ax, ay, az, bx, by, bz):
    return ax * bx + ay * by + az * bz


@triton.jit
def _cross(ax, ay, az, bx, by, bz):
    return ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx


@triton.jit
def _f64(ptr, t, m):
    return tl.load(ptr + t, mask=m, other=0.0).to(tl.float64)


@triton.jit
def _add(out_ptr, row, n_atoms, atom, m, e):
    tl.atomic_add(out_ptr + row * n_atoms + atom, e.to(out_ptr.dtype.element_ty), mask=m, sem="relaxed")


@triton.jit
def _weight(g_ptr, row, n_atoms, atom, m):
    return tl.load(g_ptr + row * n_atoms + atom, mask=m, other=0.0).to(tl.float64)


@triton.jit
def _poly4(x, c3, c4, c5, c6):
    """``1 + c3 x + c4 x^2 + c5 x^3 + c6 x^4`` and its derivative."""
    p = 1.0 + c3 * x + c4 * x * x + c5 * x * x * x + c6 * x * x * x * x
    dp = c3 + 2.0 * c4 * x + 3.0 * c5 * x * x + 4.0 * c6 * x * x * x
    return p, dp


@triton.jit
def _angle(ux, uy, uz, vx, vy, vz, tiny, bound):
    """The angle between ``u`` and ``v`` (``arccos`` of the clamped cosine) and what its gradient needs."""
    uu = _dot(ux, uy, uz, ux, uy, uz)
    vv = _dot(vx, vy, vz, vx, vy, vz)
    uv = _dot(ux, uy, uz, vx, vy, vz)
    d = tl.sqrt(uu * vv + tiny)
    c = uv / d
    cc = tl.minimum(tl.maximum(c, -bound), bound)
    return libdevice.acos(cc), c, cc, uu, vv, uv, d


@triton.jit
def _angle_grad(ux, uy, uz, vx, vy, vz, gth, c, cc, uu, vv, uv, d, bound):
    """The gradient of ``gth`` times the angle with respect to ``u`` and ``v``; the clamp passes it inside
    its closed interval only."""
    inside = (c >= -bound) & (c <= bound)
    gc = tl.where(inside, -gth / tl.sqrt(1.0 - cc * cc), 0.0)
    a = gc / d
    b = gc * uv / (d * d * d)
    return (a * vx - b * vv * ux, a * vy - b * vv * uy, a * vz - b * vv * uz,
            a * ux - b * uu * vx, a * uy - b * uu * vy, a * uz - b * uu * vz)


@triton.jit
def _bond(vec_ptr, out_ptr, g_ptr, gvec_ptr, vir_ptr, scal_ptr, t, m, n_atoms, base, a0_ptr, a1_ptr, k_ptr, r0_ptr,
          VIRIAL: tl.constexpr, GRAD: tl.constexpr):
    tiny = tl.load(scal_ptr + 0)
    c3 = tl.load(scal_ptr + 2)
    c4 = tl.load(scal_ptr + 3)
    row = base + t
    bx, by, bz = _ld(vec_ptr, row, m)
    a0 = tl.load(a0_ptr + t, mask=m, other=0)
    a1 = tl.load(a1_ptr + t, mask=m, other=0)
    k = _f64(k_ptr, t, m)
    r = tl.sqrt(_dot(bx, by, bz, bx, by, bz) + tiny)
    dr = r - _f64(r0_ptr, t, m)
    if GRAD:
        w = 0.5 * (_weight(g_ptr, 0, n_atoms, a0, m) + _weight(g_ptr, 0, n_atoms, a1, m))
        s = w * k * (2.0 * dr + 3.0 * c3 * dr * dr + 4.0 * c4 * dr * dr * dr) / r
        _st(gvec_ptr, vir_ptr, row, m, s * bx, s * by, s * bz, bx, by, bz, VIRIAL)
    else:
        e = 0.5 * k * dr * dr * (1.0 + c3 * dr + c4 * dr * dr)
        _add(out_ptr, 0, n_atoms, a0, m, e)
        _add(out_ptr, 0, n_atoms, a1, m, e)


@triton.jit
def _angles(vec_ptr, out_ptr, g_ptr, gvec_ptr, vir_ptr, scal_ptr, t, m, n_atoms, bu, bv, bw, c_ptr, k_ptr, t0_ptr,
            ip_ptr, INPLANE: tl.constexpr, VIRIAL: tl.constexpr, GRAD: tl.constexpr):
    tiny = tl.load(scal_ptr + 0)
    deg = tl.load(scal_ptr + 1)
    bound = tl.load(scal_ptr + 12)
    ux, uy, uz = _ld(vec_ptr, bu + t, m)
    vx, vy, vz = _ld(vec_ptr, bv + t, m)
    atom = tl.load(c_ptr + t, mask=m, other=0)
    k = _f64(k_ptr, t, m)
    t0 = _f64(t0_ptr, t, m)
    th, c, cc, uu, vv, uv, d = _angle(ux, uy, uz, vx, vy, vz, tiny, bound)
    if INPLANE:
        ip = tl.load(ip_ptr + t, mask=m, other=0) != 0
        wx, wy, wz = _ld(vec_ptr, bw + t, m)
        adx, ady, adz = ux - wx, uy - wy, uz - wz
        cdx, cdy, cdz = vx - wx, vy - wy, vz - wz
        nx, ny, nz = _cross(adx, ady, adz, cdx, cdy, cdz)
        num = _dot(nx, ny, nz, wx, wy, wz)
        den = _dot(nx, ny, nz, nx, ny, nz) + tiny
        delta = num / den
        px, py, pz = ux - delta * nx, uy - delta * ny, uz - delta * nz
        qx, qy, qz = vx - delta * nx, vy - delta * ny, vz - delta * nz
        th2, c2, cc2, pp, qq, pq, d2 = _angle(px, py, pz, qx, qy, qz, tiny, bound)
        th = tl.where(ip, th2, th)
    dt = th - t0
    x = dt / deg
    p, dp = _poly4(x, tl.load(scal_ptr + 4), tl.load(scal_ptr + 5), tl.load(scal_ptr + 6), tl.load(scal_ptr + 7))
    if GRAD:
        gth = _weight(g_ptr, 1, n_atoms, atom, m) * k * (2.0 * dt * p + dt * dt * dp / deg)
        if INPLANE:
            g_plain = tl.where(ip, 0.0, gth)
            g_proj = tl.where(ip, gth, 0.0)
        else:
            g_plain = gth
        gux, guy, guz, gvx, gvy, gvz = _angle_grad(ux, uy, uz, vx, vy, vz, g_plain, c, cc, uu, vv, uv, d, bound)
        if INPLANE:
            gpx, gpy, gpz, gqx, gqy, gqz = _angle_grad(px, py, pz, qx, qy, qz, g_proj, c2, cc2, pp, qq, pq, d2,
                                                       bound)
            gux, guy, guz = gux + gpx, guy + gpy, guz + gpz
            gvx, gvy, gvz = gvx + gqx, gvy + gqy, gvz + gqz
            # p = u - delta n, q = v - delta n
            gsx, gsy, gsz = -(gpx + gqx), -(gpy + gqy), -(gpz + gqz)
            gdelta = _dot(gsx, gsy, gsz, nx, ny, nz)
            gnx, gny, gnz = delta * gsx, delta * gsy, delta * gsz
            gnum = gdelta / den
            gden = -gdelta * num / (den * den)
            gnx, gny, gnz = gnx + gnum * wx + 2.0 * gden * nx, gny + gnum * wy + 2.0 * gden * ny, \
                gnz + gnum * wz + 2.0 * gden * nz
            gwx, gwy, gwz = gnum * nx, gnum * ny, gnum * nz
            # n = ad x cd
            gax, gay, gaz = _cross(cdx, cdy, cdz, gnx, gny, gnz)
            gcx, gcy, gcz = _cross(gnx, gny, gnz, adx, ady, adz)
            gux, guy, guz = gux + gax, guy + gay, guz + gaz
            gvx, gvy, gvz = gvx + gcx, gvy + gcy, gvz + gcz
            gwx, gwy, gwz = gwx - gax - gcx, gwy - gay - gcy, gwz - gaz - gcz
            _st(gvec_ptr, vir_ptr, bw + t, m, gwx, gwy, gwz, wx, wy, wz, VIRIAL)
        _st(gvec_ptr, vir_ptr, bu + t, m, gux, guy, guz, ux, uy, uz, VIRIAL)
        _st(gvec_ptr, vir_ptr, bv + t, m, gvx, gvy, gvz, vx, vy, vz, VIRIAL)
    else:
        _add(out_ptr, 1, n_atoms, atom, m, k * dt * dt * p)


@triton.jit
def _strbnds(vec_ptr, out_ptr, g_ptr, gvec_ptr, vir_ptr, scal_ptr, t, m, n_atoms, b1, b2, c_ptr, k1_ptr, k2_ptr,
             r01_ptr, r02_ptr, t0_ptr, VIRIAL: tl.constexpr, GRAD: tl.constexpr):
    tiny = tl.load(scal_ptr + 0)
    bound = tl.load(scal_ptr + 12)
    ux, uy, uz = _ld(vec_ptr, b1 + t, m)
    vx, vy, vz = _ld(vec_ptr, b2 + t, m)
    atom = tl.load(c_ptr + t, mask=m, other=0)
    k1 = _f64(k1_ptr, t, m)
    k2 = _f64(k2_ptr, t, m)
    r1 = tl.sqrt(_dot(ux, uy, uz, ux, uy, uz) + tiny)
    r2 = tl.sqrt(_dot(vx, vy, vz, vx, vy, vz) + tiny)
    th, c, cc, uu, vv, uv, d = _angle(ux, uy, uz, vx, vy, vz, tiny, bound)
    stretch = k1 * (r1 - _f64(r01_ptr, t, m)) + k2 * (r2 - _f64(r02_ptr, t, m))
    bend = th - _f64(t0_ptr, t, m)
    if GRAD:
        w = _weight(g_ptr, 2, n_atoms, atom, m)
        gux, guy, guz, gvx, gvy, gvz = _angle_grad(ux, uy, uz, vx, vy, vz, w * stretch, c, cc, uu, vv, uv, d, bound)
        s1 = w * bend * k1 / r1
        s2 = w * bend * k2 / r2
        _st(gvec_ptr, vir_ptr, b1 + t, m, gux + s1 * ux, guy + s1 * uy, guz + s1 * uz, ux, uy, uz,
            VIRIAL)
        _st(gvec_ptr, vir_ptr, b2 + t, m, gvx + s2 * vx, gvy + s2 * vy, gvz + s2 * vz, vx, vy, vz,
            VIRIAL)
    else:
        _add(out_ptr, 2, n_atoms, atom, m, stretch * bend)


@triton.jit
def _ureys(vec_ptr, out_ptr, g_ptr, gvec_ptr, vir_ptr, scal_ptr, t, m, n_atoms, base, c_ptr, k_ptr, r0_ptr,
           VIRIAL: tl.constexpr, GRAD: tl.constexpr):
    tiny = tl.load(scal_ptr + 0)
    x, y, z = _ld(vec_ptr, base + t, m)
    atom = tl.load(c_ptr + t, mask=m, other=0)
    k = _f64(k_ptr, t, m)
    r = tl.sqrt(_dot(x, y, z, x, y, z) + tiny)
    dr = r - _f64(r0_ptr, t, m)
    if GRAD:
        s = _weight(g_ptr, 3, n_atoms, atom, m) * 2.0 * k * dr / r
        _st(gvec_ptr, vir_ptr, base + t, m, s * x, s * y, s * z, x, y, z, VIRIAL)
    else:
        _add(out_ptr, 3, n_atoms, atom, m, k * dr * dr)


@triton.jit
def _opbends(vec_ptr, out_ptr, g_ptr, gvec_ptr, vir_ptr, scal_ptr, t, m, n_atoms, bab, bcb, bdb, c_ptr, k_ptr,
             ALLINGER: tl.constexpr, VIRIAL: tl.constexpr, GRAD: tl.constexpr):
    tiny = tl.load(scal_ptr + 0)
    deg = tl.load(scal_ptr + 1)
    upper = tl.load(scal_ptr + 13)
    ax, ay, az = _ld(vec_ptr, bab + t, m)
    cx, cy, cz = _ld(vec_ptr, bcb + t, m)
    dx, dy, dz = _ld(vec_ptr, bdb + t, m)
    atom = tl.load(c_ptr + t, mask=m, other=0)
    k = _f64(k_ptr, t, m)
    xx, xy, xz = _cross(ax, ay, az, cx, cy, cz)
    ee = _dot(dx, dy, dz, xx, xy, xz)
    if ALLINGER:
        px, py, pz = ax - dx, ay - dy, az - dz
        qx, qy, qz = cx - dx, cy - dy, cz - dz
        nx, ny, nz = _cross(px, py, pz, qx, qy, qz)
    else:
        nx, ny, nz = xx, xy, xz
    cc = _dot(nx, ny, nz, nx, ny, nz)
    rdb2 = _dot(dx, dy, dz, dx, dy, dz)
    den = cc * rdb2 + tiny
    sin2 = ee * ee / den
    clamped = tl.minimum(tl.maximum(sin2, 0.0), upper)
    sn = tl.sqrt(clamped)
    chi = libdevice.asin(sn)
    x = chi / deg
    p, dp = _poly4(x, tl.load(scal_ptr + 8), tl.load(scal_ptr + 9), tl.load(scal_ptr + 10), tl.load(scal_ptr + 11))
    if GRAD:
        gchi = _weight(g_ptr, 4, n_atoms, atom, m) * k * (2.0 * chi * p + chi * chi * dp / deg)
        inside = (sin2 >= 0.0) & (sin2 <= upper) & (sn > 0.0)
        gs2 = tl.where(inside, gchi / tl.sqrt(1.0 - sn * sn) * 0.5 / tl.where(sn > 0.0, sn, 1.0), 0.0)
        gee = gs2 * 2.0 * ee / den
        gden = -gs2 * ee * ee / (den * den)
        gcc = gden * rdb2
        grdb2 = gden * cc
        # ee = d . (a x c)
        gdx, gdy, gdz = gee * xx + 2.0 * grdb2 * dx, gee * xy + 2.0 * grdb2 * dy, gee * xz + 2.0 * grdb2 * dz
        gax, gay, gaz = _cross(cx, cy, cz, gee * dx, gee * dy, gee * dz)
        gcx, gcy, gcz = _cross(gee * dx, gee * dy, gee * dz, ax, ay, az)
        gnx, gny, gnz = 2.0 * gcc * nx, 2.0 * gcc * ny, 2.0 * gcc * nz
        if ALLINGER:
            gpx, gpy, gpz = _cross(qx, qy, qz, gnx, gny, gnz)
            gqx, gqy, gqz = _cross(gnx, gny, gnz, px, py, pz)
            gax, gay, gaz = gax + gpx, gay + gpy, gaz + gpz
            gcx, gcy, gcz = gcx + gqx, gcy + gqy, gcz + gqz
            gdx, gdy, gdz = gdx - gpx - gqx, gdy - gpy - gqy, gdz - gpz - gqz
        else:
            gpx, gpy, gpz = _cross(cx, cy, cz, gnx, gny, gnz)
            gqx, gqy, gqz = _cross(gnx, gny, gnz, ax, ay, az)
            gax, gay, gaz = gax + gpx, gay + gpy, gaz + gpz
            gcx, gcy, gcz = gcx + gqx, gcy + gqy, gcz + gqz
        _st(gvec_ptr, vir_ptr, bab + t, m, gax, gay, gaz, ax, ay, az, VIRIAL)
        _st(gvec_ptr, vir_ptr, bcb + t, m, gcx, gcy, gcz, cx, cy, cz, VIRIAL)
        _st(gvec_ptr, vir_ptr, bdb + t, m, gdx, gdy, gdz, dx, dy, dz, VIRIAL)
    else:
        _add(out_ptr, 4, n_atoms, atom, m, k * chi * chi * p)


@triton.jit
def _pick(n, v, g1, g2, g3, g4, g5, g6):
    """``v`` added to the accumulator of multiplicity ``n``."""
    return (g1 + tl.where(n == 1, v, 0.0), g2 + tl.where(n == 2, v, 0.0), g3 + tl.where(n == 3, v, 0.0),
            g4 + tl.where(n == 4, v, 0.0), g5 + tl.where(n == 5, v, 0.0), g6 + tl.where(n == 6, v, 0.0))


@triton.jit
def _back_step(gc, gs, gcn, gsn, ck, sk, c1, s1, mc, ms):
    """The reverse of ``c_(k+1) = c_k c1 - s_k s1``, ``s_(k+1) = s_k c1 + c_k s1``."""
    return (gc + gcn * c1 + gsn * s1, gs - gcn * s1 + gsn * c1,
            mc + gcn * ck + gsn * sk, ms - gcn * sk + gsn * ck)


@triton.jit
def _torsions(vec_ptr, out_ptr, g_ptr, gvec_ptr, vir_ptr, scal_ptr, t, m, n_atoms, b1, b2, b3, c_ptr, k_ptr, n_ptr,
              cos_ptr, sin_ptr, NMAX: tl.constexpr, NS: tl.constexpr, VIRIAL: tl.constexpr,
              GRAD: tl.constexpr):
    tiny = tl.load(scal_ptr + 0)
    ax, ay, az = _ld(vec_ptr, b1 + t, m)
    bx, by, bz = _ld(vec_ptr, b2 + t, m)
    cx, cy, cz = _ld(vec_ptr, b3 + t, m)
    atom = tl.load(c_ptr + t, mask=m, other=0)
    n1x, n1y, n1z = _cross(ax, ay, az, bx, by, bz)
    n2x, n2y, n2z = _cross(bx, by, bz, cx, cy, cz)
    A = _dot(n1x, n1y, n1z, n1x, n1y, n1z)
    Bn = _dot(n2x, n2y, n2z, n2x, n2y, n2z)
    norm = tl.sqrt(A * Bn + tiny)
    dot = _dot(n1x, n1y, n1z, n2x, n2y, n2z)
    c1 = dot / norm
    blen = tl.sqrt(_dot(bx, by, bz, bx, by, bz) + tiny)
    mx, my, mz = _cross(n1x, n1y, n1z, n2x, n2y, n2z)
    tt = _dot(mx, my, mz, bx, by, bz)
    P = norm * blen
    s1 = -tt / P
    zero = c1 * 0.0
    c2, s2, c3, s3, c4, s4, c5, s5, c6, s6 = zero, zero, zero, zero, zero, zero, zero, zero, zero, zero
    if NMAX >= 2:
        c2, s2 = c1 * c1 - s1 * s1, s1 * c1 + c1 * s1
    if NMAX >= 3:
        c3, s3 = c2 * c1 - s2 * s1, s2 * c1 + c2 * s1
    if NMAX >= 4:
        c4, s4 = c3 * c1 - s3 * s1, s3 * c1 + c3 * s1
    if NMAX >= 5:
        c5, s5 = c4 * c1 - s4 * s1, s4 * c1 + c4 * s1
    if NMAX >= 6:
        c6, s6 = c5 * c1 - s5 * s1, s5 * c1 + c5 * s1
    e = zero
    gc1, gc2, gc3, gc4, gc5, gc6 = zero, zero, zero, zero, zero, zero
    gs1, gs2, gs3, gs4, gs5, gs6 = zero, zero, zero, zero, zero, zero
    for j in tl.static_range(NS):
        kj = _f64(k_ptr, t * NS + j, m)
        nj = tl.load(n_ptr + t * NS + j, mask=m, other=1)
        cj = _f64(cos_ptr, t * NS + j, m)
        sj = _f64(sin_ptr, t * NS + j, m)
        if GRAD:
            gc1, gc2, gc3, gc4, gc5, gc6 = _pick(nj, kj * cj, gc1, gc2, gc3, gc4, gc5, gc6)
            gs1, gs2, gs3, gs4, gs5, gs6 = _pick(nj, kj * sj, gs1, gs2, gs3, gs4, gs5, gs6)
        else:
            cn = tl.where(nj == 1, c1, tl.where(nj == 2, c2, tl.where(nj == 3, c3, tl.where(
                nj == 4, c4, tl.where(nj == 5, c5, c6)))))
            sn = tl.where(nj == 1, s1, tl.where(nj == 2, s2, tl.where(nj == 3, s3, tl.where(
                nj == 4, s4, tl.where(nj == 5, s5, s6)))))
            e = e + kj * (1.0 + cj * cn + sj * sn)
    if GRAD:
        w = _weight(g_ptr, 5, n_atoms, atom, m)
        mc, ms = zero, zero
        if NMAX >= 6:
            gc5, gs5, mc, ms = _back_step(gc5, gs5, gc6, gs6, c5, s5, c1, s1, mc, ms)
        if NMAX >= 5:
            gc4, gs4, mc, ms = _back_step(gc4, gs4, gc5, gs5, c4, s4, c1, s1, mc, ms)
        if NMAX >= 4:
            gc3, gs3, mc, ms = _back_step(gc3, gs3, gc4, gs4, c3, s3, c1, s1, mc, ms)
        if NMAX >= 3:
            gc2, gs2, mc, ms = _back_step(gc2, gs2, gc3, gs3, c2, s2, c1, s1, mc, ms)
        if NMAX >= 2:
            gc1, gs1, mc, ms = _back_step(gc1, gs1, gc2, gs2, c1, s1, c1, s1, mc, ms)
        gc = w * (gc1 + mc)
        gs = w * (gs1 + ms)
        # cos = dot / norm, sin = -(m . b) / (norm |b|)
        gdot = gc / norm
        gnorm = -gc * dot / (norm * norm)
        gt = -gs / P
        gP = gs * tt / (P * P)
        gnorm = gnorm + gP * blen
        gblen = gP * norm
        gmx, gmy, gmz = gt * bx, gt * by, gt * bz
        gbx, gby, gbz = gt * mx + gblen * bx / blen, gt * my + gblen * by / blen, gt * mz + gblen * bz / blen
        g1x, g1y, g1z = _cross(n2x, n2y, n2z, gmx, gmy, gmz)
        g2x, g2y, g2z = _cross(gmx, gmy, gmz, n1x, n1y, n1z)
        gAB = gnorm / (2.0 * norm)
        g1x, g1y, g1z = g1x + gdot * n2x + 2.0 * gAB * Bn * n1x, g1y + gdot * n2y + 2.0 * gAB * Bn * n1y, \
            g1z + gdot * n2z + 2.0 * gAB * Bn * n1z
        g2x, g2y, g2z = g2x + gdot * n1x + 2.0 * gAB * A * n2x, g2y + gdot * n1y + 2.0 * gAB * A * n2y, \
            g2z + gdot * n1z + 2.0 * gAB * A * n2z
        # n1 = a x b, n2 = b x c
        gax, gay, gaz = _cross(bx, by, bz, g1x, g1y, g1z)
        hx, hy, hz = _cross(g1x, g1y, g1z, ax, ay, az)
        jx, jy, jz = _cross(cx, cy, cz, g2x, g2y, g2z)
        gcx, gcy, gcz = _cross(g2x, g2y, g2z, bx, by, bz)
        _st(gvec_ptr, vir_ptr, b1 + t, m, gax, gay, gaz, ax, ay, az, VIRIAL)
        _st(gvec_ptr, vir_ptr, b2 + t, m, gbx + hx + jx, gby + hy + jy, gbz + hz + jz, bx, by, bz,
            VIRIAL)
        _st(gvec_ptr, vir_ptr, b3 + t, m, gcx, gcy, gcz, cx, cy, cz, VIRIAL)
    else:
        _add(out_ptr, 5, n_atoms, atom, m, e)


@triton.jit
def _valence(vec_ptr, out_ptr, g_ptr, gvec_ptr, vir_ptr, scal_ptr, n_atoms,
             nb, bb, b_a0, b_a1, b_k, b_r0,
             na, au, av, aw, a_c, a_k, a_t0, a_ip,
             ns, s1, s2, s_c, s_k1, s_k2, s_r01, s_r02, s_t0,
             nu, ub, u_c, u_k, u_r0,
             no, oab, ocb, odb, o_c, o_k,
             nt, t1, t2, t3, t_c, t_k, t_n, t_cos, t_sin,
             e1, e2, e3, e4, e5,
             INPLANE: tl.constexpr, ALLINGER: tl.constexpr, NMAX: tl.constexpr, NS: tl.constexpr,
             VIRIAL: tl.constexpr, GRAD: tl.constexpr, BLOCK: tl.constexpr):
    """One block of terms of one kind per program: the programs below ``e1`` the bonds, then the angles up to
    ``e2``, the stretch-bends up to ``e3``, the Urey-Bradley terms up to ``e4``, the out-of-plane bends up to
    ``e5`` and the torsions. ``GRAD`` stores the vectors' gradient instead of adding the energies, with
    ``VIRIAL`` adding the sum of every vector times its gradient to ``vir_ptr`` ``(3, 3)`` too."""
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    if pid < e1:
        t = pid * BLOCK + lane
        _bond(vec_ptr, out_ptr, g_ptr, gvec_ptr, vir_ptr, scal_ptr, t, t < nb, n_atoms, bb, b_a0, b_a1, b_k, b_r0,
              VIRIAL, GRAD)
    elif pid < e2:
        t = (pid - e1) * BLOCK + lane
        _angles(vec_ptr, out_ptr, g_ptr, gvec_ptr, vir_ptr, scal_ptr, t, t < na, n_atoms, au, av, aw, a_c, a_k, a_t0,
                a_ip, INPLANE, VIRIAL, GRAD)
    elif pid < e3:
        t = (pid - e2) * BLOCK + lane
        _strbnds(vec_ptr, out_ptr, g_ptr, gvec_ptr, vir_ptr, scal_ptr, t, t < ns, n_atoms, s1, s2, s_c, s_k1, s_k2,
                 s_r01, s_r02, s_t0, VIRIAL, GRAD)
    elif pid < e4:
        t = (pid - e3) * BLOCK + lane
        _ureys(vec_ptr, out_ptr, g_ptr, gvec_ptr, vir_ptr, scal_ptr, t, t < nu, n_atoms, ub, u_c, u_k, u_r0,
               VIRIAL, GRAD)
    elif pid < e5:
        t = (pid - e4) * BLOCK + lane
        _opbends(vec_ptr, out_ptr, g_ptr, gvec_ptr, vir_ptr, scal_ptr, t, t < no, n_atoms, oab, ocb, odb, o_c, o_k,
                 ALLINGER, VIRIAL, GRAD)
    else:
        t = (pid - e5) * BLOCK + lane
        _torsions(vec_ptr, out_ptr, g_ptr, gvec_ptr, vir_ptr, scal_ptr, t, t < nt, n_atoms, t1, t2, t3, t_c, t_k, t_n,
                  t_cos, t_sin, NMAX, NS, VIRIAL, GRAD)


class ValenceTable:
    """What the valence kernels read for the terms of a step plan: per kind, the count, the first row of
    every vector the kind uses in the plan's stacked bonded vectors, the atom (or atoms) its energy goes
    to and its parameters, plus the float64 constants and the switches of the forms.

    Parameters
    ----------
    n_atoms : int
        Atoms of the batch.
    kinds : dict
        Per kind (``"bond"``, ``"angle"``, ``"strbnd"``, ``"urey"``, ``"opbend"``, ``"torsion"``) a tuple of
        its count, its first rows and its tensors, in the order of the kernel's arguments.
    scalars : Tensor
        The float64 constants, in the order of :data:`SCALARS`.
    inplane, allinger : bool
        Whether in-plane angles occur and whether the out-of-plane bends take Allinger's angle.
    nmax, slots : int
        The largest torsion multiplicity and the number of a torsion's harmonic slots.
    """

    def __init__(self, n_atoms: int, kinds: dict, scalars: Tensor, inplane: bool, allinger: bool, nmax: int,
                 slots: int):
        self.n_atoms, self.kinds, self.scalars = int(n_atoms), kinds, scalars
        self.inplane, self.allinger, self.nmax, self.slots = bool(inplane), bool(allinger), int(nmax), int(slots)

    def unit(self, like: Tensor) -> Tensor:
        """Unit weights of every row ``(8, N)`` in the dtype and on the device of ``like``, kept."""
        key = (like.dtype, like.device)
        if getattr(self, "_unit", None) is None or self._unit[0] != key:
            self._unit = (key, torch.ones(len(TERMS), self.n_atoms, dtype=like.dtype, device=like.device))
        return self._unit[1]

    def arguments(self) -> tuple:
        args, ends, programs = [], [], 0
        for kind in ("bond", "angle", "strbnd", "urey", "opbend", "torsion"):
            count, rows, tensors = self.kinds[kind]
            args += [count, *rows, *tensors]
            programs += triton.cdiv(count, BLOCK)
            ends.append(programs)
        return args, ends


def _launch(vec: Tensor, out: Tensor, g: Tensor, gvec: Tensor, table: ValenceTable, grad: bool,
            virial: Optional[Tensor] = None) -> None:
    args, ends = table.arguments()
    if ends[-1]:
        _valence[(ends[-1],)](vec, out, g, gvec, table.scalars if virial is None else virial, table.scalars,
                              table.n_atoms, *args, *ends[:5], INPLANE=table.inplane, ALLINGER=table.allinger,
                              NMAX=table.nmax, NS=table.slots, VIRIAL=virial is not None, GRAD=grad, BLOCK=BLOCK,
                              num_warps=4)


def energies_and_gradient(vec: Tensor, table: ValenceTable) -> tuple:
    """The per-atom rows ``(8, N)`` of the valence terms, the gradient of their sum with respect to the
    stacked bonded vectors ``vec`` ``(M, 3)`` and the sum of every vector times its gradient ``(3, 3)``
    in float64 (no autograd)."""
    vec = vec.contiguous()
    out = torch.zeros(len(TERMS), table.n_atoms, dtype=vec.dtype, device=vec.device)
    _launch(vec, out, out, vec, table, False)
    unit = table.unit(vec)
    gvec = torch.zeros_like(vec)
    virial = torch.zeros(3, 3, dtype=torch.float64, device=vec.device)
    _launch(vec, unit, unit, gvec, table, True, virial)
    return out, gvec, virial


class ValenceTerms(torch.autograd.Function):
    """The per-atom rows ``(8, N)`` of the valence terms (:data:`TERMS`; the pi-torsion and torsion-torsion
    rows stay zero) from the stacked bonded vectors ``vec`` ``(M, 3)`` of a :class:`ValenceTable`,
    differentiable once with respect to the vectors."""

    @staticmethod
    def forward(ctx, vec: Tensor, table: ValenceTable) -> Tensor:
        vec = vec.contiguous()
        out = torch.zeros(len(TERMS), table.n_atoms, dtype=vec.dtype, device=vec.device)
        _launch(vec, out, out, vec, table, False)
        ctx.save_for_backward(vec)
        ctx.table = table
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, g: Tensor):
        (vec,) = ctx.saved_tensors
        # the vectors of in-plane angles' fourth atoms that no in-plane angle reads keep a zero gradient
        gvec = torch.zeros_like(vec)
        _launch(vec, g.contiguous(), g.contiguous(), gvec, ctx.table, True)
        return gvec, None
