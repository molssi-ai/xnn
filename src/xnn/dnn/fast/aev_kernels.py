"""Triton kernels and autograd functions of the AEV fast path (see :mod:`xnn.dnn.fast.aev`).

Imported only when the fast path runs, so that ``triton`` is needed only there.

The radial term sums ``g_l(r_e) = p exp(-eta_l (r_e - Rs_l)^2) fc(r_e)`` over the edges
of a center into the block of the neighbor's species; the angular term sums
``g_l = A_l(theta_t) R_l(r1_t, r2_t)`` over the neighbor pairs of a center into the block
of their species pair, with ``A_l = ((1 + cos(theta - theta_s,l)) / 2)^zeta_l`` and
``R_l = 2 exp(-eta_l ((r1 + r2) / 2 - Rs_l)^2) fc(r1) fc(r2)`` (the ANI form, the
``2^(1-zeta)`` folded in). The kernels take the per-edge distances and the per-triplet
``(r1, r2, theta)``; the geometry before them (norms, the angle) stays in PyTorch.

Every term has four kernels: the sums (one program per center over its edges or triplets
in center order, accumulated in registers, so no atomic adds and the same result in every
run), the gradient with respect to the inputs (one program per block of edges or triplets),
and the two derivatives of that gradient, with respect to the incoming gradient (the sums
again, of the first partial derivatives along a tangent) and with respect to the inputs
(second partial derivatives). The gradient is therefore differentiable once, which is what
a force loss needs for training.
"""
# no postponed annotations: Triton's interpreter (TRITON_INTERPRET=1) recognizes the
# tl.constexpr parameters of a kernel only from evaluated annotations

import math
from typing import Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor

from xnn.common.models.dispersion_fast._triton import next_power_of_two

#: Edges or triplets per program of the gradient kernels.
BLOCK = 64
#: Triplets per program of the factored angular gradient (its tiles are three-dimensional).
BLOCK_FACTORED = 32


@triton.jit
def _cutoff(r, rc, pi_rc):
    """The cosine cutoff and its first two derivatives at ``r``."""
    inside = r < rc
    c = tl.cos(pi_rc * r)
    s = tl.sin(pi_rc * r)
    f = tl.where(inside, 0.5 * c + 0.5, 0.0)
    f1 = tl.where(inside, -0.5 * pi_rc * s, 0.0)
    f2 = tl.where(inside, -0.5 * pi_rc * pi_rc * c, 0.0)
    return f, f1, f2


@triton.jit
def _radial_terms(r, eta, rs, pref, rc, pi_rc):
    """``g``, ``dg/dr`` and ``d2g/dr2`` of every lane at ``r`` (broadcast)."""
    u = r - rs
    e = tl.exp(-eta * u * u)
    e1 = -2.0 * eta * u * e
    e2 = (4.0 * eta * eta * u * u - 2.0 * eta) * e
    f, f1, f2 = _cutoff(r, rc, pi_rc)
    g = pref * e * f
    g1 = pref * (e1 * f + e * f1)
    g2 = pref * (e2 * f + 2.0 * e1 * f1 + e * f2)
    return g, g1, g2


@triton.jit
def _radial_sums_kernel(r_ptr, s_ptr, t_ptr, order_ptr, ptr_ptr, eta_ptr, rs_ptr, out_ptr,
                        c_ptr, TANGENT: tl.constexpr,
                        P: tl.constexpr, PP: tl.constexpr, S: tl.constexpr, SP: tl.constexpr):
    """``out[i, s] = sum_e g(r_e)`` over the edges of center ``i`` with neighbor species ``s``
    (``TANGENT``: ``sum_e t_e dg/dr(r_e)``)."""
    i = tl.program_id(0).to(tl.int64)
    pref, rc, pi_rc = tl.load(c_ptr), tl.load(c_ptr + 1), tl.load(c_ptr + 2)
    lane = tl.arange(0, PP)
    row = tl.arange(0, SP)
    lm = lane < P
    eta = tl.load(eta_ptr + lane, mask=lm, other=0.0)
    rs = tl.load(rs_ptr + lane, mask=lm, other=0.0)
    start = tl.load(ptr_ptr + i)
    end = tl.load(ptr_ptr + i + 1)
    acc = tl.zeros([SP, PP], dtype=out_ptr.dtype.element_ty)
    n = start
    while n < end:
        e = tl.load(order_ptr + n)
        r = tl.load(r_ptr + e)
        s = tl.load(s_ptr + e)
        g, g1, g2 = _radial_terms(r, eta, rs, pref, rc, pi_rc)
        if TANGENT:
            g = g1 * tl.load(t_ptr + e)
        acc += tl.where(row[:, None] == s, g[None, :], 0.0)
        n += 1
    m = (row[:, None] < S) & lm[None, :]
    tl.store(out_ptr + i * (S * P) + row[:, None] * P + lane[None, :], acc, mask=m)


@triton.jit
def _radial_grad_kernel(r_ptr, s_ptr, c_ptr, w_ptr, t_ptr, eta_ptr, rs_ptr, out_ptr, n_edges,
                        k_ptr, SECOND: tl.constexpr,
                        P: tl.constexpr, PP: tl.constexpr, S: tl.constexpr, B: tl.constexpr):
    """Per edge ``sum_l w[c, s, l] dg_l/dr`` (``SECOND``: ``t_e sum_l w[c, s, l] d2g_l/dr2``)."""
    e = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    pref, rc, pi_rc = tl.load(k_ptr), tl.load(k_ptr + 1), tl.load(k_ptr + 2)
    em = e < n_edges
    lane = tl.arange(0, PP)
    lm = lane < P
    eta = tl.load(eta_ptr + lane, mask=lm, other=0.0)[None, :]
    rs = tl.load(rs_ptr + lane, mask=lm, other=0.0)[None, :]
    r = tl.load(r_ptr + e, mask=em, other=1.0)[:, None]
    s = tl.load(s_ptr + e, mask=em, other=-1)
    c = tl.load(c_ptr + e, mask=em, other=0)
    ok = em & (s >= 0)
    w = tl.load(w_ptr + c[:, None] * (S * P) + s[:, None] * P + lane[None, :],
                mask=ok[:, None] & lm[None, :], other=0.0)
    g, g1, g2 = _radial_terms(r, eta, rs, pref, rc, pi_rc)
    if SECOND:
        val = tl.sum(w * g2, axis=1) * tl.load(t_ptr + e, mask=em, other=0.0)
    else:
        val = tl.sum(w * g1, axis=1)
    tl.store(out_ptr + e, val, mask=em)


@triton.jit
def _ang_factor(cth, sth, zeta, cts, sts):
    """The angular factor ``A`` and its two theta derivatives (broadcast). ``cth, sth`` are the
    cosine and sine of the angle, ``cts, sts`` those of the shifts: the shifted cosine and sine by
    the addition theorems, no trigonometry per shift."""
    cd = cth * cts + sth * sts                                   # cos(theta - theta_s)
    sd = sth * cts - cth * sts                                   # sin(theta - theta_s)
    b = 0.5 + 0.5 * cd
    # one logarithm and one exponential per lane: b^(zeta-1) and b^(zeta-2) by division (float64
    # transcendentals are long instruction sequences); at b = 0: 0^0 = 1 and 0^k = 0 for k > 0 (a
    # negative power only meets a zero coefficient)
    pos = b > 0
    safe = tl.where(pos, b, 1.0)
    p0 = tl.exp(zeta * tl.log(safe))
    p1 = p0 / safe
    p2 = p1 / safe
    a = tl.where(pos, p0, tl.where(zeta == 0.0, 1.0, 0.0))
    q1 = tl.where(pos, p1, tl.where(zeta == 1.0, 1.0, 0.0))
    q2 = tl.where(pos, p2, tl.where(zeta == 2.0, 1.0, 0.0))
    a1 = zeta * q1 * (-0.5 * sd)
    a2 = zeta * (zeta - 1.0) * q2 * (0.25 * sd * sd) + zeta * q1 * (-0.5 * cd)
    return a, a1, a2


@triton.jit
def _rad_factor(r1, r2, eta, rs, rc, pi_rc):
    """The radial factor ``R`` (cutoffs and the 2 of the ANI form included) with its first and
    second partial derivatives in ``r1`` and ``r2`` (broadcast)."""
    u = 0.5 * (r1 + r2) - rs
    e = tl.exp(-eta * u * u)
    e1 = -2.0 * eta * u * e
    e2 = (4.0 * eta * eta * u * u - 2.0 * eta) * e
    f1, f1d, f1dd = _cutoff(r1, rc, pi_rc)
    f2, f2d, f2dd = _cutoff(r2, rc, pi_rc)
    rr = 2.0 * e * f1 * f2
    r_1 = 2.0 * (0.5 * e1 * f1 * f2 + e * f1d * f2)
    r_2 = 2.0 * (0.5 * e1 * f1 * f2 + e * f1 * f2d)
    r_11 = 2.0 * (0.25 * e2 * f1 * f2 + e1 * f1d * f2 + e * f1dd * f2)
    r_22 = 2.0 * (0.25 * e2 * f1 * f2 + e1 * f1 * f2d + e * f1 * f2dd)
    r_12 = 2.0 * (0.25 * e2 * f1 * f2 + 0.5 * e1 * f1 * f2d + 0.5 * e1 * f1d * f2 + e * f1d * f2d)
    return rr, r_1, r_2, r_11, r_22, r_12


@triton.jit
def _angular_terms(r1, r2, cth, sth, eta, zeta, rs, cts, sts, rc, pi_rc):
    """Both factors on the same lanes (a grid with several zeta values)."""
    a, a1, a2 = _ang_factor(cth, sth, zeta, cts, sts)
    rr, r_1, r_2, r_11, r_22, r_12 = _rad_factor(r1, r2, eta, rs, rc, pi_rc)
    return a, a1, a2, rr, r_1, r_2, r_11, r_22, r_12


@triton.jit
def _angular_sums_kernel(r1_ptr, r2_ptr, th_ptr, b_ptr, t1_ptr, t2_ptr, tt_ptr, ptr_ptr,
                         eta_ptr, zeta_ptr, rs_ptr, cts_ptr, sts_ptr, out_ptr, k_ptr, TANGENT: tl.constexpr,
                         P: tl.constexpr, PP: tl.constexpr, NB: tl.constexpr, NBP: tl.constexpr):
    """``out[i, b] = sum_t g(t)`` over the triplets of center ``i`` with species pair ``b``
    (``TANGENT``: the derivative of ``g`` along the tangent ``(t1, t2, tt)`` of
    ``(r1, r2, theta)``)."""
    i = tl.program_id(0).to(tl.int64)
    rc, pi_rc = tl.load(k_ptr + 1), tl.load(k_ptr + 2)
    lane = tl.arange(0, PP)
    row = tl.arange(0, NBP)
    lm = lane < P
    eta = tl.load(eta_ptr + lane, mask=lm, other=0.0)
    zeta = tl.load(zeta_ptr + lane, mask=lm, other=1.0)
    rs = tl.load(rs_ptr + lane, mask=lm, other=0.0)
    cts = tl.load(cts_ptr + lane, mask=lm, other=1.0)
    sts = tl.load(sts_ptr + lane, mask=lm, other=0.0)
    start = tl.load(ptr_ptr + i)
    end = tl.load(ptr_ptr + i + 1)
    acc = tl.zeros([NBP, PP], dtype=out_ptr.dtype.element_ty)
    n = start
    while n < end:
        r1 = tl.load(r1_ptr + n)
        r2 = tl.load(r2_ptr + n)
        th = tl.load(th_ptr + n)
        bk = tl.load(b_ptr + n)
        a, a1, a2, rr, r_1, r_2, r_11, r_22, r_12 = _angular_terms(r1, r2, tl.cos(th), tl.sin(th), eta, zeta, rs,
                                                                   cts, sts, rc, pi_rc)
        if TANGENT:
            g = a * (r_1 * tl.load(t1_ptr + n) + r_2 * tl.load(t2_ptr + n)) + a1 * rr * tl.load(tt_ptr + n)
        else:
            g = a * rr
        acc += tl.where(row[:, None] == bk, g[None, :], 0.0)
        n += 1
    m = (row[:, None] < NB) & lm[None, :]
    tl.store(out_ptr + i * (NB * P) + row[:, None] * P + lane[None, :], acc, mask=m)


@triton.jit
def _angular_grad_kernel(r1_ptr, r2_ptr, th_ptr, b_ptr, c_ptr, w_ptr, t1_ptr, t2_ptr, tt_ptr,
                         eta_ptr, zeta_ptr, rs_ptr, cts_ptr, sts_ptr, o1_ptr, o2_ptr, ot_ptr, n_trip, k_ptr,
                         SECOND: tl.constexpr, P: tl.constexpr, PP: tl.constexpr, NB: tl.constexpr,
                         B: tl.constexpr):
    """Per triplet the gradient ``sum_l w[c, b, l] dg_l/d(r1, r2, theta)`` (``SECOND``: the
    second derivatives contracted with ``w`` and the tangent ``(t1, t2, tt)``)."""
    t = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    rc, pi_rc = tl.load(k_ptr + 1), tl.load(k_ptr + 2)
    tm = t < n_trip
    lane = tl.arange(0, PP)
    lm = lane < P
    eta = tl.load(eta_ptr + lane, mask=lm, other=0.0)[None, :]
    zeta = tl.load(zeta_ptr + lane, mask=lm, other=1.0)[None, :]
    rs = tl.load(rs_ptr + lane, mask=lm, other=0.0)[None, :]
    cts = tl.load(cts_ptr + lane, mask=lm, other=1.0)[None, :]
    sts = tl.load(sts_ptr + lane, mask=lm, other=0.0)[None, :]
    r1 = tl.load(r1_ptr + t, mask=tm, other=1.0)[:, None]
    r2 = tl.load(r2_ptr + t, mask=tm, other=1.0)[:, None]
    th = tl.load(th_ptr + t, mask=tm, other=0.0)[:, None]
    bk = tl.load(b_ptr + t, mask=tm, other=-1)
    c = tl.load(c_ptr + t, mask=tm, other=0)
    ok = tm & (bk >= 0)
    w = tl.load(w_ptr + c[:, None] * (NB * P) + bk[:, None] * P + lane[None, :],
                mask=ok[:, None] & lm[None, :], other=0.0)
    a, a1, a2, rr, r_1, r_2, r_11, r_22, r_12 = _angular_terms(r1, r2, tl.cos(th), tl.sin(th), eta, zeta, rs,
                                                               cts, sts, rc, pi_rc)
    if SECOND:
        v1 = tl.load(t1_ptr + t, mask=tm, other=0.0)[:, None]
        v2 = tl.load(t2_ptr + t, mask=tm, other=0.0)[:, None]
        vt = tl.load(tt_ptr + t, mask=tm, other=0.0)[:, None]
        d1 = tl.sum(w * (a * (r_11 * v1 + r_12 * v2) + a1 * r_1 * vt), axis=1)
        d2 = tl.sum(w * (a * (r_12 * v1 + r_22 * v2) + a1 * r_2 * vt), axis=1)
        dt = tl.sum(w * (a1 * (r_1 * v1 + r_2 * v2) + a2 * rr * vt), axis=1)
    else:
        d1 = tl.sum(w * a * r_1, axis=1)
        d2 = tl.sum(w * a * r_2, axis=1)
        dt = tl.sum(w * a1 * rr, axis=1)
    tl.store(o1_ptr + t, d1, mask=tm)
    tl.store(o2_ptr + t, d2, mask=tm)
    tl.store(ot_ptr + t, dt, mask=tm)


@triton.jit
def _angular_sums_factored_kernel(r1_ptr, r2_ptr, th_ptr, b_ptr, t1_ptr, t2_ptr, tt_ptr, ptr_ptr,
                                  eta_ptr, rs_ptr, zeta_ptr, cts_ptr, sts_ptr, out_ptr, k_ptr,
                                  TANGENT: tl.constexpr, R: tl.constexpr, RP: tl.constexpr, T: tl.constexpr,
                                  TP: tl.constexpr, NB: tl.constexpr, NBP: tl.constexpr):
    """:func:`_angular_sums_kernel` for a grid with one zeta: the radial factor on the ``R`` rows
    ``(eta, Rs)`` and the angular factor on the ``T`` columns ``theta_s``, each evaluated once per
    triplet, their product the ``R T`` functions (column ``r T + t``)."""
    i = tl.program_id(0).to(tl.int64)
    rc, pi_rc = tl.load(k_ptr + 1), tl.load(k_ptr + 2)
    row = tl.arange(0, RP)
    col = tl.arange(0, TP)
    pair = tl.arange(0, NBP)
    rm = row < R
    cm = col < T
    eta = tl.load(eta_ptr + row, mask=rm, other=0.0)
    rs = tl.load(rs_ptr + row, mask=rm, other=0.0)
    zeta = tl.load(zeta_ptr)
    cts = tl.load(cts_ptr + col, mask=cm, other=1.0)
    sts = tl.load(sts_ptr + col, mask=cm, other=0.0)
    start = tl.load(ptr_ptr + i)
    end = tl.load(ptr_ptr + i + 1)
    acc = tl.zeros([NBP, RP, TP], dtype=out_ptr.dtype.element_ty)
    n = start
    while n < end:
        r1 = tl.load(r1_ptr + n)
        r2 = tl.load(r2_ptr + n)
        th = tl.load(th_ptr + n)
        bk = tl.load(b_ptr + n)
        a, a1, a2 = _ang_factor(tl.cos(th), tl.sin(th), zeta, cts, sts)
        rr, r_1, r_2, r_11, r_22, r_12 = _rad_factor(r1, r2, eta, rs, rc, pi_rc)
        if TANGENT:
            g = (a[None, :] * (r_1 * tl.load(t1_ptr + n) + r_2 * tl.load(t2_ptr + n))[:, None]
                 + a1[None, :] * (rr * tl.load(tt_ptr + n))[:, None])
        else:
            g = rr[:, None] * a[None, :]
        acc += tl.where(pair[:, None, None] == bk, g[None, :, :], 0.0)
        n += 1
    m = (pair[:, None, None] < NB) & rm[None, :, None] & cm[None, None, :]
    offs = pair[:, None, None] * (R * T) + row[None, :, None] * T + col[None, None, :]
    tl.store(out_ptr + i * (NB * R * T) + offs, acc, mask=m)


@triton.jit
def _angular_grad_factored_kernel(r1_ptr, r2_ptr, th_ptr, b_ptr, c_ptr, w_ptr, t1_ptr, t2_ptr, tt_ptr,
                                  eta_ptr, rs_ptr, zeta_ptr, cts_ptr, sts_ptr, o1_ptr, o2_ptr, ot_ptr, n_trip,
                                  k_ptr, SECOND: tl.constexpr, R: tl.constexpr, RP: tl.constexpr,
                                  T: tl.constexpr, TP: tl.constexpr, NB: tl.constexpr, B: tl.constexpr):
    """:func:`_angular_grad_kernel` for a grid with one zeta (rows and columns as in
    :func:`_angular_sums_factored_kernel`)."""
    t = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    rc, pi_rc = tl.load(k_ptr + 1), tl.load(k_ptr + 2)
    tm = t < n_trip
    row = tl.arange(0, RP)
    col = tl.arange(0, TP)
    rm = row < R
    cm = col < T
    eta = tl.load(eta_ptr + row, mask=rm, other=0.0)[None, :]
    rs = tl.load(rs_ptr + row, mask=rm, other=0.0)[None, :]
    zeta = tl.load(zeta_ptr)
    cts = tl.load(cts_ptr + col, mask=cm, other=1.0)[None, :]
    sts = tl.load(sts_ptr + col, mask=cm, other=0.0)[None, :]
    r1 = tl.load(r1_ptr + t, mask=tm, other=1.0)[:, None]
    r2 = tl.load(r2_ptr + t, mask=tm, other=1.0)[:, None]
    th = tl.load(th_ptr + t, mask=tm, other=0.0)[:, None]
    bk = tl.load(b_ptr + t, mask=tm, other=-1)
    c = tl.load(c_ptr + t, mask=tm, other=0)
    ok = tm & (bk >= 0)
    offs = row[None, :, None] * T + col[None, None, :]
    w = tl.load(w_ptr + (c * (NB * R * T) + bk * (R * T))[:, None, None] + offs,
                mask=ok[:, None, None] & rm[None, :, None] & cm[None, None, :], other=0.0)
    a, a1, a2 = _ang_factor(tl.cos(th), tl.sin(th), zeta, cts, sts)            # (B, TP)
    rr, r_1, r_2, r_11, r_22, r_12 = _rad_factor(r1, r2, eta, rs, rc, pi_rc)  # (B, RP)
    # the contractions of w with each column factor first: (B, RP) per angular factor
    wa = tl.sum(w * a[:, None, :], axis=2)
    wa1 = tl.sum(w * a1[:, None, :], axis=2)
    if SECOND:
        wa2 = tl.sum(w * a2[:, None, :], axis=2)
        v1 = tl.load(t1_ptr + t, mask=tm, other=0.0)[:, None]
        v2 = tl.load(t2_ptr + t, mask=tm, other=0.0)[:, None]
        vt = tl.load(tt_ptr + t, mask=tm, other=0.0)[:, None]
        d1 = tl.sum(wa * (r_11 * v1 + r_12 * v2) + wa1 * r_1 * vt, axis=1)
        d2 = tl.sum(wa * (r_12 * v1 + r_22 * v2) + wa1 * r_2 * vt, axis=1)
        dt = tl.sum(wa1 * (r_1 * v1 + r_2 * v2) + wa2 * rr * vt, axis=1)
    else:
        d1 = tl.sum(wa * r_1, axis=1)
        d2 = tl.sum(wa * r_2, axis=1)
        dt = tl.sum(wa1 * rr, axis=1)
    tl.store(o1_ptr + t, d1, mask=tm)
    tl.store(o2_ptr + t, d2, mask=tm)
    tl.store(ot_ptr + t, dt, mask=tm)


class RadialSpec:
    """The constants of a radial term: lane tables ``(P,)``, species count, prefactor, cutoff."""

    def __init__(self, eta: Tensor, rs: Tensor, n_species: int, prefactor: float, cutoff: float):
        self.eta, self.rs = eta.contiguous(), rs.contiguous()
        self.p, self.s = int(eta.numel()), int(n_species)
        # (prefactor, cutoff, pi / cutoff) in the inputs' precision: a Python float reaches a
        # kernel as float32
        self.consts = torch.tensor([float(prefactor), float(cutoff), math.pi / float(cutoff)],
                                   dtype=eta.dtype).to(eta.device)


class AngularSpec:
    """The constants of an angular term from the axes of its grid ``(eta, zeta, Rs, theta_s)``.

    With one zeta (every ANI preset) the term is factored: the rows ``(eta, Rs)`` and the columns
    ``theta_s`` (``factored``); otherwise every function is a lane of flat tables ``(P,)``.
    """

    def __init__(self, etas: Tensor, zetas: Tensor, rs: Tensor, ts: Tensor, n_pairs: int, cutoff: float):
        self.factored = zetas.numel() == 1
        if self.factored:
            self.eta = etas.repeat_interleave(rs.numel()).contiguous()          # rows (eta, Rs)
            self.rs = rs.repeat(etas.numel()).contiguous()
            self.zeta = zetas.reshape(1).contiguous()
            self.cts, self.sts = torch.cos(ts).contiguous(), torch.sin(ts).contiguous()
            self.r, self.t = int(self.eta.numel()), int(ts.numel())
        else:
            grid = torch.cartesian_prod(etas, zetas, rs, ts)
            self.eta, self.zeta, self.rs = (grid[:, k].contiguous() for k in range(3))
            self.cts, self.sts = torch.cos(grid[:, 3]).contiguous(), torch.sin(grid[:, 3]).contiguous()
        self.p, self.nb = int(etas.numel() * zetas.numel() * rs.numel() * ts.numel()), int(n_pairs)
        # (unused, cutoff, pi / cutoff) in the inputs' precision (see RadialSpec)
        self.consts = torch.tensor([1.0, float(cutoff), math.pi / float(cutoff)], dtype=etas.dtype).to(etas.device)


def _radial_sums(r: Tensor, species: Tensor, tangent: Tensor, rows: Tuple[Tensor, Tensor], n: int,
                 spec: RadialSpec, use_tangent: bool) -> Tensor:
    out = torch.empty(n, spec.s * spec.p, dtype=r.dtype, device=r.device)
    if n == 0:
        return out
    order, ptr = rows
    _radial_sums_kernel[(n,)](r, species, tangent, order, ptr, spec.eta, spec.rs, out, spec.consts,
                              TANGENT=use_tangent, P=spec.p, PP=next_power_of_two(spec.p),
                              S=spec.s, SP=next_power_of_two(spec.s, 2))
    return out


def _radial_grad(r: Tensor, species: Tensor, center: Tensor, w: Tensor, tangent: Tensor,
                 spec: RadialSpec, second: bool) -> Tensor:
    out = torch.empty_like(r)
    if r.numel() == 0:
        return out
    _radial_grad_kernel[(triton.cdiv(r.numel(), BLOCK),)](
        r, species, center, w.contiguous(), tangent, spec.eta, spec.rs, out, r.numel(), spec.consts,
        SECOND=second, P=spec.p, PP=next_power_of_two(spec.p), S=spec.s, B=BLOCK)
    return out


class RadialFunction(torch.autograd.Function):
    """The radial AEV ``(N, S P)`` from the edge distances ``(E,)``."""

    @staticmethod
    def forward(ctx, r, species, center, rows, n, spec):
        ctx.save_for_backward(r)
        ctx.species, ctx.center, ctx.rows, ctx.spec = species, center, rows, spec
        return _radial_sums(r, species, r, rows, n, spec, False)

    @staticmethod
    def backward(ctx, w):
        (r,) = ctx.saved_tensors
        return (RadialGradFunction.apply(w, r, ctx.species, ctx.center, ctx.rows, ctx.spec),
                None, None, None, None, None)


class RadialGradFunction(torch.autograd.Function):
    """The gradient of :class:`RadialFunction` with respect to the distances, differentiable once."""

    @staticmethod
    def forward(ctx, w, r, species, center, rows, spec):
        w = w.contiguous()
        ctx.save_for_backward(w, r)
        ctx.species, ctx.center, ctx.rows, ctx.spec = species, center, rows, spec
        return _radial_grad(r, species, center, w, r, spec, False)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, v):
        w, r = ctx.saved_tensors
        v = v.contiguous()
        grad_w = grad_r = None
        if ctx.needs_input_grad[0]:
            grad_w = _radial_sums(r, ctx.species, v, ctx.rows, w.shape[0], ctx.spec, True)
        if ctx.needs_input_grad[1]:
            grad_r = _radial_grad(r, ctx.species, ctx.center, w, v, ctx.spec, True)
        return grad_w, grad_r, None, None, None, None


def _angular_sums(r1: Tensor, r2: Tensor, th: Tensor, bucket: Tensor, tangents, ptr: Tensor, n: int,
                  spec: AngularSpec) -> Tensor:
    out = torch.empty(n, spec.nb * spec.p, dtype=r1.dtype, device=r1.device)
    if n == 0:
        return out
    t1, t2, tt = tangents if tangents is not None else (r1, r2, th)
    if spec.factored:
        _angular_sums_factored_kernel[(n,)](
            r1, r2, th, bucket, t1, t2, tt, ptr, spec.eta, spec.rs, spec.zeta, spec.cts, spec.sts, out, spec.consts,
            TANGENT=tangents is not None, R=spec.r, RP=next_power_of_two(spec.r, 1), T=spec.t,
            TP=next_power_of_two(spec.t, 1), NB=spec.nb, NBP=next_power_of_two(spec.nb, 2))
        return out
    _angular_sums_kernel[(n,)](r1, r2, th, bucket, t1, t2, tt, ptr, spec.eta, spec.zeta, spec.rs, spec.cts, spec.sts,
                               out, spec.consts, TANGENT=tangents is not None, P=spec.p,
                               PP=next_power_of_two(spec.p), NB=spec.nb, NBP=next_power_of_two(spec.nb, 2))
    return out


def _angular_grad(r1: Tensor, r2: Tensor, th: Tensor, bucket: Tensor, center: Tensor, w: Tensor, tangents,
                  spec: AngularSpec) -> Tuple[Tensor, Tensor, Tensor]:
    o1, o2, ot = torch.empty_like(r1), torch.empty_like(r1), torch.empty_like(r1)
    if r1.numel() == 0:
        return o1, o2, ot
    t1, t2, tt = tangents if tangents is not None else (r1, r2, th)
    if spec.factored:
        _angular_grad_factored_kernel[(triton.cdiv(r1.numel(), BLOCK_FACTORED),)](
            r1, r2, th, bucket, center, w.contiguous(), t1, t2, tt, spec.eta, spec.rs, spec.zeta, spec.cts, spec.sts,
            o1, o2, ot, r1.numel(), spec.consts, SECOND=tangents is not None, R=spec.r,
            RP=next_power_of_two(spec.r, 1), T=spec.t, TP=next_power_of_two(spec.t, 1), NB=spec.nb, B=BLOCK_FACTORED)
        return o1, o2, ot
    _angular_grad_kernel[(triton.cdiv(r1.numel(), BLOCK),)](
        r1, r2, th, bucket, center, w.contiguous(), t1, t2, tt, spec.eta, spec.zeta, spec.rs, spec.cts, spec.sts,
        o1, o2, ot, r1.numel(), spec.consts, SECOND=tangents is not None, P=spec.p,
        PP=next_power_of_two(spec.p), NB=spec.nb, B=BLOCK)
    return o1, o2, ot


class AngularFunction(torch.autograd.Function):
    """The angular AEV ``(N, NB P)`` from the triplets' ``(r1, r2, theta)`` (center order)."""

    @staticmethod
    def forward(ctx, r1, r2, th, bucket, center, ptr, n, spec):
        ctx.save_for_backward(r1, r2, th)
        ctx.bucket, ctx.center, ctx.ptr, ctx.spec = bucket, center, ptr, spec
        return _angular_sums(r1, r2, th, bucket, None, ptr, n, spec)

    @staticmethod
    def backward(ctx, w):
        r1, r2, th = ctx.saved_tensors
        g1, g2, gt = AngularGradFunction.apply(w, r1, r2, th, ctx.bucket, ctx.center, ctx.ptr, ctx.spec)
        return g1, g2, gt, None, None, None, None, None


class AngularGradFunction(torch.autograd.Function):
    """The gradient of :class:`AngularFunction` with respect to ``(r1, r2, theta)``,
    differentiable once."""

    @staticmethod
    def forward(ctx, w, r1, r2, th, bucket, center, ptr, spec):
        w = w.contiguous()
        ctx.save_for_backward(w, r1, r2, th)
        ctx.bucket, ctx.center, ctx.ptr, ctx.spec = bucket, center, ptr, spec
        return _angular_grad(r1, r2, th, bucket, center, w, None, spec)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, v1, v2, vt):
        w, r1, r2, th = ctx.saved_tensors
        tangents = tuple(torch.zeros_like(r1) if v is None else v.contiguous() for v in (v1, v2, vt))
        grad_w = None
        if ctx.needs_input_grad[0]:
            grad_w = _angular_sums(r1, r2, th, ctx.bucket, tangents, ctx.ptr, w.shape[0], ctx.spec)
        g1 = g2 = gt = None
        if any(ctx.needs_input_grad[1:4]):
            g1, g2, gt = _angular_grad(r1, r2, th, ctx.bucket, ctx.center, w, tangents, ctx.spec)
        return grad_w, g1, g2, gt, None, None, None, None
