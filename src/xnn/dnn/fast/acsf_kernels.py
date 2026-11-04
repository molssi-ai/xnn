"""Triton kernels and autograd functions of the HDNNP symmetry functions (see :mod:`xnn.dnn.fast.acsf`).

Imported only when the kernels run, so that ``triton`` is needed only there.

The radial functions (types 1 and 2) of an edge depend on its length ``r``; the angular functions
(types 3 and 9) of a triplet factor into ``A(c) h(r1) h(r2) h(r3)``, with ``c`` the cosine of the
angle, ``A(c) = 2^(1-zeta) max(1 + lambda c, 0)^zeta``, ``h(r) = exp(-eta r^2) fc(r)`` and no
``h(r3)`` for type 9. The kernels take the per-edge lengths and the per-triplet ``(r1, r2, r3, c)``;
the geometry before them stays in PyTorch.

The edges (triplets) are sorted by center and neighbor species (species pair), and one program
sums a segment: the lanes are the functions of that center element and neighbor species (pair),
so no lane is masked, and every program writes its own output columns (no atomic adds, the same
result in every run). The cutoff values of the at most two declared cutoffs are evaluated once per
edge or triplet and shared by the lanes. Each order has its own kernels: the values, the first
derivatives (the gradient with respect to the inputs and the sums of a tangent's first
derivatives) and the second derivatives (contracted with a tangent). The gradient is therefore
differentiable once, which is what a force loss needs for training.
"""
# no postponed annotations: Triton's interpreter (TRITON_INTERPRET=1) recognizes the
# tl.constexpr parameters of a kernel only from evaluated annotations

from typing import Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor

from xnn.common.models.dispersion_fast._triton import next_power_of_two
from xnn.dnn.fast.acsf_tables import Tables, pair_index, tables  # noqa: F401

#: Edges or triplets per step of the segment kernels and per program of the per-item kernels.
BLOCK = 32

@triton.jit
def _cutoff(r, kind, rc, rin, inv, pi_inv, ORDER: tl.constexpr):
    """``fc`` and, up to ``ORDER``, its derivatives at ``r`` (``kind`` 0 cosine, 1 tanh; ``inv = 1 / (rc -
    rin)``, ``pi_inv = pi inv``); constant below ``rin``, zero from ``rc``."""
    inside = r < rc
    moving = (r > rin) & inside
    x = tl.minimum(tl.maximum((r - rin) * inv, 0.0), 1.0)
    arg = pi_inv * (r - rin) * moving.to(r.dtype)
    c = tl.cos(arg)
    y = 1.0 - x
    q = tl.exp(-2.0 * y)
    t = (1.0 - q) / (1.0 + q)
    is_tanh = kind == 1
    f = tl.where(inside, tl.where(is_tanh, t * t * t, 0.5 * c + 0.5), 0.0)
    f1 = f
    f2 = f
    if ORDER >= 1:
        s = tl.sin(arg)
        dt = 1.0 - t * t
        f1 = tl.where(moving, tl.where(is_tanh, -3.0 * t * t * dt * inv, -0.5 * pi_inv * s), 0.0)
        if ORDER >= 2:
            f2 = tl.where(moving, tl.where(is_tanh, (6.0 * t - 12.0 * t * t * t) * dt * inv * inv,
                                           -0.5 * pi_inv * pi_inv * c), 0.0)
    return f, f1, f2


@triton.jit
def _cutoff_set(r, cut_ptr, ci, NC: tl.constexpr, ORDER: tl.constexpr):
    """The cutoff values of each lane's cutoff ``ci`` at ``r`` (the at most two declared cutoffs evaluated
    once and selected)."""
    f, f1, f2 = _cutoff(r, tl.load(cut_ptr), tl.load(cut_ptr + 1), tl.load(cut_ptr + 2), tl.load(cut_ptr + 3),
                        tl.load(cut_ptr + 4), ORDER)
    if NC == 2:
        g, g1, g2 = _cutoff(r, tl.load(cut_ptr + 5), tl.load(cut_ptr + 6), tl.load(cut_ptr + 7),
                            tl.load(cut_ptr + 8), tl.load(cut_ptr + 9), ORDER)
        one = ci == 1
        f = tl.where(one, g, f)
        f1 = tl.where(one, g1, f1)
        f2 = tl.where(one, g2, f2)
    return f, f1, f2


@triton.jit
def _radial(r, typ, eta, rs, power, f, f1, f2, ORDER: tl.constexpr, HAS1: tl.constexpr):
    """``g`` and, up to ``ORDER``, its derivatives in ``r`` of type 1 (``fc^power``) and type 2
    (``exp(-eta (r - rs)^2) fc``) lanes."""
    u = r - rs
    e = tl.exp(-eta * u * u)
    g = e * f
    g1 = g
    g2 = g
    if ORDER >= 1:
        g1 = e * (f1 - 2.0 * eta * u * f)
    if ORDER >= 2:
        g2 = e * (f2 - 4.0 * eta * u * f1 + (4.0 * eta * eta * u * u - 2.0 * eta) * f)
    if HAS1:
        pos = f > 0.0
        fs = tl.where(pos, f, 1.0)
        pw = tl.where(pos, tl.exp(power * tl.log(fs)), 0.0)
        one = typ == 1
        g = tl.where(one, pw, g)
        if ORDER >= 1:
            pw1 = pw / fs
            g1 = tl.where(one, power * pw1 * f1, g1)
            if ORDER >= 2:
                g2 = tl.where(one, power * (power - 1.0) * (pw1 / fs) * f1 * f1 + power * pw1 * f2, g2)
    return g, g1, g2


@triton.jit
def _radial_sums_kernel(r_ptr, t_ptr, ptr_ptr, sp_ptr, col_ptr, typ_ptr, eta_ptr, rs_ptr, pw_ptr, ci_ptr, cut_ptr,
                        out_ptr, TANGENT: tl.constexpr, NC: tl.constexpr, HAS1: tl.constexpr, Q: tl.constexpr,
                        K: tl.constexpr, KP: tl.constexpr, D: tl.constexpr, B: tl.constexpr):
    """``out[i, col_l] = sum_e g_l(r_e)`` over the edges of segment (center ``i``, neighbor species), in the
    sorted order (``TANGENT``: ``sum_e t_e dg_l/dr``)."""
    seg = tl.program_id(0).to(tl.int64)
    i = seg // Q
    row = tl.load(sp_ptr + i).to(tl.int64) * Q + seg % Q
    lane = tl.arange(0, KP)[None, :]
    lm = lane < K
    at = row * K + lane
    col = tl.load(col_ptr + at, mask=lm, other=-1)
    typ = tl.load(typ_ptr + at, mask=lm, other=2)
    eta = tl.load(eta_ptr + at, mask=lm, other=0.0)
    rs = tl.load(rs_ptr + at, mask=lm, other=0.0)
    pw = tl.load(pw_ptr + at, mask=lm, other=1.0)
    ci = tl.load(ci_ptr + at, mask=lm, other=0)
    bb = tl.arange(0, B)
    start = tl.load(ptr_ptr + seg)
    end = tl.load(ptr_ptr + seg + 1)
    acc = tl.zeros([B, KP], dtype=out_ptr.dtype.element_ty)
    n = start
    while n < end:
        em = (n + bb) < end
        r = tl.load(r_ptr + n + bb, mask=em, other=1.0)[:, None]
        if TANGENT:
            f, f1, f2 = _cutoff_set(r, cut_ptr, ci, NC, 1)
            g, g1, g2 = _radial(r, typ, eta, rs, pw, f, f1, f2, 1, HAS1)
            g = g1 * tl.load(t_ptr + n + bb, mask=em, other=0.0)[:, None]
        else:
            f, f1, f2 = _cutoff_set(r, cut_ptr, ci, NC, 0)
            g, g1, g2 = _radial(r, typ, eta, rs, pw, f, f1, f2, 0, HAS1)
        acc += tl.where(em[:, None], g, 0.0)
        n += B
    ok = (col >= 0) & lm
    tl.store(out_ptr + i * D + col, tl.sum(acc, axis=0)[None, :], mask=ok)


@triton.jit
def _radial_grad_kernel(r_ptr, row_ptr, c_ptr, w_ptr, t_ptr, col_ptr, typ_ptr, eta_ptr, rs_ptr, pw_ptr, ci_ptr,
                        cut_ptr, out_ptr, n_edges, SECOND: tl.constexpr, NC: tl.constexpr, HAS1: tl.constexpr,
                        K: tl.constexpr, KP: tl.constexpr, D: tl.constexpr, B: tl.constexpr):
    """Per edge ``sum_l w[c, col_l] dg_l/dr`` (``SECOND``: ``t_e sum_l w[c, col_l] d2g_l/dr2``)."""
    e = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    em = e < n_edges
    lane = tl.arange(0, KP)[None, :]
    r = tl.load(r_ptr + e, mask=em, other=1.0)[:, None]
    row = tl.load(row_ptr + e, mask=em, other=0).to(tl.int64)[:, None]
    c = tl.load(c_ptr + e, mask=em, other=0).to(tl.int64)[:, None]
    m2 = em[:, None] & (lane < K)
    at = row * K + lane
    col = tl.load(col_ptr + at, mask=m2, other=-1)
    ok = m2 & (col >= 0)
    w = tl.load(w_ptr + c * D + col, mask=ok, other=0.0)
    ci = tl.load(ci_ptr + at, mask=m2, other=0)
    f, f1, f2 = _cutoff_set(r, cut_ptr, ci, NC, 1 + SECOND)
    g, g1, g2 = _radial(r, tl.load(typ_ptr + at, mask=m2, other=2), tl.load(eta_ptr + at, mask=m2, other=0.0),
                        tl.load(rs_ptr + at, mask=m2, other=0.0), tl.load(pw_ptr + at, mask=m2, other=1.0),
                        f, f1, f2, 1 + SECOND, HAS1)
    if SECOND:
        val = tl.sum(w * g2, axis=1) * tl.load(t_ptr + e, mask=em, other=0.0)
    else:
        val = tl.sum(w * g1, axis=1)
    tl.store(out_ptr + e, val, mask=em)


@triton.jit
def _ang_factors(c, r1, r2, r3, typ, eta, lam, zeta, pref, ci, cut_ptr, NC: tl.constexpr, ORDER: tl.constexpr):
    """Per lane ``A E``, ``A' E``, ``A'' E`` (``E = exp(-eta S)``, ``S = r1^2 + r2^2 (+ r3^2)``), the cutoff
    values ``f_k`` and, up to ``ORDER``, ``q_k = f_k' - 2 eta r_k f_k`` and ``p_k = f_k'' - 4 eta r_k f_k' +
    (4 eta^2 r_k^2 - 2 eta) f_k`` (``f_3 = 1``, ``q_3 = p_3 = 0`` for type 9): ``g = A E f1 f2 f3`` and its
    derivatives are products of these."""
    three = typ == 3
    e = tl.exp(-eta * (r1 * r1 + r2 * r2 + tl.where(three, r3 * r3, 0.0)))
    base = 1.0 + lam * c
    pos = base > 0.0
    bs = tl.where(pos, base, 1.0)
    ae = tl.where(pos, pref * tl.exp(zeta * tl.log(bs)), 0.0) * e
    f1, d1, s1 = _cutoff_set(r1, cut_ptr, ci, NC, ORDER)
    f2, d2, s2 = _cutoff_set(r2, cut_ptr, ci, NC, ORDER)
    f3, d3, s3 = _cutoff_set(r3, cut_ptr, ci, NC, ORDER)
    f3 = tl.where(three, f3, 1.0)
    a1e = ae
    a2e = ae
    q1 = f1
    q2 = f2
    q3 = f3
    p1 = f1
    p2 = f2
    p3 = f3
    if ORDER >= 1:
        a1e = zeta * lam * ae / bs
        q1 = d1 - 2.0 * eta * r1 * f1
        q2 = d2 - 2.0 * eta * r2 * f2
        q3 = tl.where(three, d3 - 2.0 * eta * r3 * f3, 0.0)
        if ORDER >= 2:
            a2e = zeta * (zeta - 1.0) * lam * lam * ae / (bs * bs)
            p1 = s1 - 4.0 * eta * r1 * d1 + (4.0 * eta * eta * r1 * r1 - 2.0 * eta) * f1
            p2 = s2 - 4.0 * eta * r2 * d2 + (4.0 * eta * eta * r2 * r2 - 2.0 * eta) * f2
            p3 = tl.where(three, s3 - 4.0 * eta * r3 * d3 + (4.0 * eta * eta * r3 * r3 - 2.0 * eta) * f3, 0.0)
    return ae, a1e, a2e, f1, q1, p1, f2, q2, p2, f3, q3, p3


@triton.jit
def _angular_sums_kernel(r1_ptr, r2_ptr, r3_ptr, c_ptr, t1_ptr, t2_ptr, t3_ptr, tc_ptr, ptr_ptr, sp_ptr, col_ptr,
                         typ_ptr, eta_ptr, lam_ptr, zeta_ptr, pref_ptr, ci_ptr, cut_ptr, out_ptr,
                         TANGENT: tl.constexpr, NC: tl.constexpr, Q: tl.constexpr, K: tl.constexpr,
                         KP: tl.constexpr, D: tl.constexpr, B: tl.constexpr):
    """``out[i, col_l] = sum_t g_l(t)`` over the triplets of segment (center ``i``, species pair), in the
    sorted order (``TANGENT``: the derivative of ``g`` along the tangent ``(t1, t2, t3, tc)``)."""
    seg = tl.program_id(0).to(tl.int64)
    i = seg // Q
    row = tl.load(sp_ptr + i).to(tl.int64) * Q + seg % Q
    lane = tl.arange(0, KP)[None, :]
    lm = lane < K
    at = row * K + lane
    col = tl.load(col_ptr + at, mask=lm, other=-1)
    typ = tl.load(typ_ptr + at, mask=lm, other=9)
    eta = tl.load(eta_ptr + at, mask=lm, other=0.0)
    lam = tl.load(lam_ptr + at, mask=lm, other=1.0)
    zeta = tl.load(zeta_ptr + at, mask=lm, other=1.0)
    pref = tl.load(pref_ptr + at, mask=lm, other=0.0)
    ci = tl.load(ci_ptr + at, mask=lm, other=0)
    bb = tl.arange(0, B)
    start = tl.load(ptr_ptr + seg)
    end = tl.load(ptr_ptr + seg + 1)
    acc = tl.zeros([B, KP], dtype=out_ptr.dtype.element_ty)
    n = start
    while n < end:
        tm = (n + bb) < end
        t = n + bb
        r1 = tl.load(r1_ptr + t, mask=tm, other=1.0)[:, None]
        r2 = tl.load(r2_ptr + t, mask=tm, other=1.0)[:, None]
        r3 = tl.load(r3_ptr + t, mask=tm, other=1.0)[:, None]
        c = tl.load(c_ptr + t, mask=tm, other=0.0)[:, None]
        if TANGENT:
            ae, a1e, a2e, f1, q1, p1, f2, q2, p2, f3, q3, p3 = _ang_factors(
                c, r1, r2, r3, typ, eta, lam, zeta, pref, ci, cut_ptr, NC, 1)
            g = (ae * (q1 * f2 * f3 * tl.load(t1_ptr + t, mask=tm, other=0.0)[:, None]
                       + f1 * q2 * f3 * tl.load(t2_ptr + t, mask=tm, other=0.0)[:, None]
                       + f1 * f2 * q3 * tl.load(t3_ptr + t, mask=tm, other=0.0)[:, None])
                 + a1e * f1 * f2 * f3 * tl.load(tc_ptr + t, mask=tm, other=0.0)[:, None])
        else:
            ae, a1e, a2e, f1, q1, p1, f2, q2, p2, f3, q3, p3 = _ang_factors(
                c, r1, r2, r3, typ, eta, lam, zeta, pref, ci, cut_ptr, NC, 0)
            g = ae * f1 * f2 * f3
        acc += tl.where(tm[:, None], g, 0.0)
        n += B
    ok = (col >= 0) & lm
    tl.store(out_ptr + i * D + col, tl.sum(acc, axis=0)[None, :], mask=ok)


@triton.jit
def _angular_grad_kernel(r1_ptr, r2_ptr, r3_ptr, c_ptr, row_ptr, ctr_ptr, w_ptr, t1_ptr, t2_ptr, t3_ptr, tc_ptr,
                         col_ptr, typ_ptr, eta_ptr, lam_ptr, zeta_ptr, pref_ptr, ci_ptr, cut_ptr, o1_ptr, o2_ptr,
                         o3_ptr, oc_ptr, n_trip, SECOND: tl.constexpr, NC: tl.constexpr, K: tl.constexpr,
                         KP: tl.constexpr, D: tl.constexpr, B: tl.constexpr):
    """Per triplet ``sum_l w[ctr, col_l] dg_l/d(r1, r2, r3, c)`` (``SECOND``: the second derivatives
    contracted with ``w`` and the tangent ``(t1, t2, t3, tc)``)."""
    t = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    tm = t < n_trip
    lane = tl.arange(0, KP)[None, :]
    r1 = tl.load(r1_ptr + t, mask=tm, other=1.0)[:, None]
    r2 = tl.load(r2_ptr + t, mask=tm, other=1.0)[:, None]
    r3 = tl.load(r3_ptr + t, mask=tm, other=1.0)[:, None]
    c = tl.load(c_ptr + t, mask=tm, other=0.0)[:, None]
    row = tl.load(row_ptr + t, mask=tm, other=0).to(tl.int64)[:, None]
    ctr = tl.load(ctr_ptr + t, mask=tm, other=0).to(tl.int64)[:, None]
    m2 = tm[:, None] & (lane < K)
    at = row * K + lane
    col = tl.load(col_ptr + at, mask=m2, other=-1)
    ok = m2 & (col >= 0)
    w = tl.load(w_ptr + ctr * D + col, mask=ok, other=0.0)
    ae, a1e, a2e, f1, q1, p1, f2, q2, p2, f3, q3, p3 = _ang_factors(
        c, r1, r2, r3, tl.load(typ_ptr + at, mask=m2, other=9), tl.load(eta_ptr + at, mask=m2, other=0.0),
        tl.load(lam_ptr + at, mask=m2, other=1.0), tl.load(zeta_ptr + at, mask=m2, other=1.0),
        tl.load(pref_ptr + at, mask=m2, other=0.0), tl.load(ci_ptr + at, mask=m2, other=0), cut_ptr, NC,
        1 + SECOND)
    if SECOND:
        v1 = tl.load(t1_ptr + t, mask=tm, other=0.0)[:, None]
        v2 = tl.load(t2_ptr + t, mask=tm, other=0.0)[:, None]
        v3 = tl.load(t3_ptr + t, mask=tm, other=0.0)[:, None]
        vc = tl.load(tc_ptr + t, mask=tm, other=0.0)[:, None]
        d1 = ae * (p1 * f2 * f3 * v1 + q1 * q2 * f3 * v2 + q1 * f2 * q3 * v3) + a1e * q1 * f2 * f3 * vc
        d2 = ae * (q1 * q2 * f3 * v1 + f1 * p2 * f3 * v2 + f1 * q2 * q3 * v3) + a1e * f1 * q2 * f3 * vc
        d3 = ae * (q1 * f2 * q3 * v1 + f1 * q2 * q3 * v2 + f1 * f2 * p3 * v3) + a1e * f1 * f2 * q3 * vc
        dc = a1e * (q1 * f2 * f3 * v1 + f1 * q2 * f3 * v2 + f1 * f2 * q3 * v3) + a2e * f1 * f2 * f3 * vc
    else:
        d1 = ae * q1 * f2 * f3
        d2 = ae * f1 * q2 * f3
        d3 = ae * f1 * f2 * q3
        dc = a1e * f1 * f2 * f3
    tl.store(o1_ptr + t, tl.sum(w * d1, axis=1), mask=tm)
    tl.store(o2_ptr + t, tl.sum(w * d2, axis=1), mask=tm)
    tl.store(o3_ptr + t, tl.sum(w * d3, axis=1), mask=tm)
    tl.store(oc_ptr + t, tl.sum(w * dc, axis=1), mask=tm)


def segments(center: Tensor, q: Tensor, n: int, n_q: int) -> Tuple[Tensor, Tensor]:
    """The order that sorts items by (center, ``q``) and the segment pointers ``(n n_q + 1,)``."""
    key = center * n_q + q
    order = torch.argsort(key, stable=True)
    counts = torch.zeros(n * n_q, dtype=torch.long, device=center.device).index_add_(0, key, torch.ones_like(key))
    return order, torch.cat([counts.new_zeros(1), torch.cumsum(counts, 0)])


def _radial_sums(r: Tensor, tangent: Tensor, ptr: Tensor, sp: Tensor, n: int, tab: Tables, use_tangent: bool) -> Tensor:
    out = torch.zeros(n, tab.d, dtype=r.dtype, device=r.device)
    if n == 0:
        return out
    _radial_sums_kernel[(n * tab.q,)](r, tangent, ptr, sp, *tab.arrays, out, TANGENT=use_tangent, NC=tab.nc,
                                      HAS1=tab.has1, Q=tab.q, K=tab.k, KP=next_power_of_two(tab.k, 1), D=tab.d,
                                      B=BLOCK)
    return out


def _radial_grad(r: Tensor, row: Tensor, center: Tensor, w: Tensor, tangent: Tensor, tab: Tables,
                 second: bool) -> Tensor:
    out = torch.empty_like(r)
    if r.numel() == 0:
        return out
    _radial_grad_kernel[(triton.cdiv(r.numel(), BLOCK),)](
        r, row, center, w.contiguous(), tangent, *tab.arrays, out, r.numel(), SECOND=second, NC=tab.nc,
        HAS1=tab.has1, K=tab.k, KP=next_power_of_two(tab.k, 1), D=tab.d, B=BLOCK)
    return out


class RadialFunction(torch.autograd.Function):
    """The radial symmetry functions ``(N, D)`` from the edge lengths ``(E,)`` in segment order (row ``(E,)``:
    each edge's table row, center the edge's center, ``ptr`` the segment pointers, ``sp`` the atoms' species)."""

    @staticmethod
    def forward(ctx, r, row, center, ptr, sp, n, tab):
        ctx.save_for_backward(r)
        ctx.row, ctx.center, ctx.ptr, ctx.sp, ctx.tab = row, center, ptr, sp, tab
        return _radial_sums(r, r, ptr, sp, n, tab, False)

    @staticmethod
    def backward(ctx, w):
        (r,) = ctx.saved_tensors
        return (RadialGradFunction.apply(w, r, ctx.row, ctx.center, ctx.ptr, ctx.sp, ctx.tab),
                None, None, None, None, None, None)


class RadialGradFunction(torch.autograd.Function):
    """The gradient of :class:`RadialFunction` with respect to the lengths, differentiable once."""

    @staticmethod
    def forward(ctx, w, r, row, center, ptr, sp, tab):
        w = w.contiguous()
        ctx.save_for_backward(w, r)
        ctx.row, ctx.center, ctx.ptr, ctx.sp, ctx.tab = row, center, ptr, sp, tab
        return _radial_grad(r, row, center, w, r, tab, False)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, v):
        w, r = ctx.saved_tensors
        v = v.contiguous()
        grad_w = grad_r = None
        if ctx.needs_input_grad[0]:
            grad_w = _radial_sums(r, v, ctx.ptr, ctx.sp, w.shape[0], ctx.tab, True)
        if ctx.needs_input_grad[1]:
            grad_r = _radial_grad(r, ctx.row, ctx.center, w, v, ctx.tab, True)
        return grad_w, grad_r, None, None, None, None, None


def _angular_sums(x: Tuple[Tensor, Tensor, Tensor, Tensor], tangents, ptr: Tensor, sp: Tensor, n: int,
                  tab: Tables) -> Tensor:
    out = torch.zeros(n, tab.d, dtype=x[0].dtype, device=x[0].device)
    if n == 0:
        return out
    t = tangents if tangents is not None else x
    _angular_sums_kernel[(n * tab.q,)](*x, *t, ptr, sp, *tab.arrays, out, TANGENT=tangents is not None, NC=tab.nc,
                                       Q=tab.q, K=tab.k, KP=next_power_of_two(tab.k, 1), D=tab.d, B=BLOCK)
    return out


def _angular_grad(x: Tuple[Tensor, Tensor, Tensor, Tensor], row: Tensor, center: Tensor, w: Tensor, tangents,
                  tab: Tables) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    out = tuple(torch.empty_like(x[0]) for _ in range(4))
    if x[0].numel() == 0:
        return out
    t = tangents if tangents is not None else x
    _angular_grad_kernel[(triton.cdiv(x[0].numel(), BLOCK),)](
        *x, row, center, w.contiguous(), *t, *tab.arrays, *out, x[0].numel(), SECOND=tangents is not None,
        NC=tab.nc, K=tab.k, KP=next_power_of_two(tab.k, 1), D=tab.d, B=BLOCK)
    return out


class AngularFunction(torch.autograd.Function):
    """The angular symmetry functions ``(N, D)`` from the triplets' ``(r1, r2, r3, cos)`` in segment order."""

    @staticmethod
    def forward(ctx, r1, r2, r3, c, row, center, ptr, sp, n, tab):
        ctx.save_for_backward(r1, r2, r3, c)
        ctx.row, ctx.center, ctx.ptr, ctx.sp, ctx.tab = row, center, ptr, sp, tab
        return _angular_sums((r1, r2, r3, c), None, ptr, sp, n, tab)

    @staticmethod
    def backward(ctx, w):
        x = ctx.saved_tensors
        g = AngularGradFunction.apply(w, *x, ctx.row, ctx.center, ctx.ptr, ctx.sp, ctx.tab)
        return (*g, None, None, None, None, None, None)


class AngularGradFunction(torch.autograd.Function):
    """The gradient of :class:`AngularFunction` with respect to ``(r1, r2, r3, cos)``, differentiable once."""

    @staticmethod
    def forward(ctx, w, r1, r2, r3, c, row, center, ptr, sp, tab):
        w = w.contiguous()
        ctx.save_for_backward(w, r1, r2, r3, c)
        ctx.row, ctx.center, ctx.ptr, ctx.sp, ctx.tab = row, center, ptr, sp, tab
        return _angular_grad((r1, r2, r3, c), row, center, w, None, tab)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, v1, v2, v3, vc):
        w, *x = ctx.saved_tensors
        x = tuple(x)
        tangents = tuple(torch.zeros_like(x[0]) if v is None else v.contiguous() for v in (v1, v2, v3, vc))
        grad_w = None
        if ctx.needs_input_grad[0]:
            grad_w = _angular_sums(x, tangents, ctx.ptr, ctx.sp, w.shape[0], ctx.tab)
        g = (None, None, None, None)
        if any(ctx.needs_input_grad[1:5]):
            g = _angular_grad(x, ctx.row, ctx.center, w, tangents, ctx.tab)
        return (grad_w, *g, None, None, None, None, None, None)
