"""Triton kernels and autograd functions of the sGDML fast path (see :mod:`xnn.krnn.fast.sgdml`).

Imported only when the fast path runs, so that ``triton`` is needed only there.
"""
# no postponed annotations: Triton's interpreter (TRITON_INTERPRET=1) recognizes the
# tl.constexpr parameters of a kernel only from evaluated annotations

from functools import lru_cache
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import once_differentiable

from ..kernels import Matern, _deriv

#: entries of the ``(B, M Q)`` coefficient matrices formed at once (64 MB each in float64)
PAIR_CHUNK = 1 << 23
#: block sizes of the pair kernel (structures, training terms, descriptor entries)
BLOCK_B, BLOCK_J, BLOCK_D = 8, 32, 8
#: contraction programs per streaming multiprocessor; the contraction splits the training terms to reach them
PROGRAMS_PER_SM = 4


@triton.jit
def _poly(c_ptr, off: tl.constexpr, n: tl.constexpr, r):
    """``sum_k c[off + k] r^k`` by Horner's rule (float64 coefficients from memory)."""
    out = tl.zeros_like(r) + tl.load(c_ptr + off + n - 1)
    for k in tl.static_range(n - 1):
        out = out * r + tl.load(c_ptr + off + n - 2 - k)
    return out


@triton.jit
def _pairs_kernel(x_ptr, v_ptr, xp_ptr, wp_ptr, ec_ptr, c_ptr, alpha_ptr, beta_ptr, part_ptr, B, J, D,
                  HAS_E: tl.constexpr, HVP: tl.constexpr, STORE: tl.constexpr,
                  NP: tl.constexpr, NQ: tl.constexpr, NT: tl.constexpr,
                  BB: tl.constexpr, BJ: tl.constexpr, BD: tl.constexpr):
    """Kernel factors of every pair of a structure ``b`` and a training term ``j``.

    Energy pass: ``alpha = g``, ``beta = t s - g a`` and the partial energies
    ``sum_j g s + k a``. Hessian pass (``HVP``, direction ``v``): ``alpha = t (D.v)``,
    ``beta = u (D.v) s + t (w.v) - a t (D.v)`` and the partial sums of ``t s - g a``.
    Here ``D = x_b - xp_j``, ``s = D.w_j``, ``a = ec_j`` and ``u = t'/r``.
    """
    pid_b = tl.program_id(0)
    pid_j = tl.program_id(1)
    ob = pid_b * BB + tl.arange(0, BB)
    oj = pid_j * BJ + tl.arange(0, BJ)
    mb = ob < B
    mj = oj < J
    ob64 = ob.to(tl.int64)
    oj64 = oj.to(tl.int64)
    r2 = tl.zeros((BB, BJ), tl.float64)
    s = tl.zeros((BB, BJ), tl.float64)
    dv = tl.zeros((BB, BJ), tl.float64)
    wv = tl.zeros((BB, BJ), tl.float64)
    d0 = 0
    while d0 < D:
        od = d0 + tl.arange(0, BD)
        md = od < D
        xm = mb[:, None] & md[None, :]
        pm = mj[:, None] & md[None, :]
        xt = tl.load(x_ptr + ob64[:, None] * D + od[None, :], mask=xm, other=0.0)
        pt = tl.load(xp_ptr + oj64[:, None] * D + od[None, :], mask=pm, other=0.0)
        wt = tl.load(wp_ptr + oj64[:, None] * D + od[None, :], mask=pm, other=0.0)
        diff = xt[:, None, :] - pt[None, :, :]
        r2 += tl.sum(diff * diff, axis=2)
        s += tl.sum(diff * wt[None, :, :], axis=2)
        if HVP:
            vt = tl.load(v_ptr + ob64[:, None] * D + od[None, :], mask=xm, other=0.0)
            dv += tl.sum(diff * vt[:, None, :], axis=2)
            wv += tl.sum(wt[None, :, :] * vt[:, None, :], axis=2)
        d0 += BD
    a = tl.load(c_ptr)
    r = tl.sqrt(r2)
    ex = tl.exp(-a * r)
    g = ex * _poly(c_ptr, 1 + NP, NQ, r)
    t = ex * _poly(c_ptr, 1 + NP + NQ, NT, r)
    m2 = mb[:, None] & mj[None, :]
    if HAS_E:
        ej = tl.load(ec_ptr + oj, mask=mj, other=0.0)[None, :]
    else:
        ej = tl.zeros((1, BJ), tl.float64)
    if HVP:
        pos = r > 0
        u = tl.where(pos, ex * _poly(c_ptr, 1 + NP + NQ + NT, NT, r) / tl.where(pos, r, 1.0), 0.0)
        alpha = t * dv
        beta = u * dv * s + t * wv - t * dv * ej
        part = t * s - g * ej
    else:
        alpha = g
        beta = t * s - g * ej
        part = g * s + ex * _poly(c_ptr, 1, NP, r) * ej
    tl.store(part_ptr + pid_j * B + ob, tl.sum(tl.where(m2, part, 0.0), axis=1), mask=mb)
    if STORE:
        tl.store(alpha_ptr + ob64[:, None] * J + oj64[None, :], alpha, mask=m2)
        tl.store(beta_ptr + ob64[:, None] * J + oj64[None, :], beta, mask=m2)


@triton.jit
def _contract_kernel(x_ptr, xp_ptr, wp_ptr, alpha_ptr, beta_ptr, out_ptr, B, J, D, J_PER,
                     BB: tl.constexpr, BJ: tl.constexpr, BD: tl.constexpr):
    """``out[split, b] = sum_j alpha_bj w_j + beta_bj (x_b - xp_j)`` over one split of the terms."""
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_s = tl.program_id(2)
    ob = pid_b * BB + tl.arange(0, BB)
    od = pid_d * BD + tl.arange(0, BD)
    mb = ob < B
    md = od < D
    ob64 = ob.to(tl.int64)
    xm = mb[:, None] & md[None, :]
    xt = tl.load(x_ptr + ob64[:, None] * D + od[None, :], mask=xm, other=0.0)
    acc = tl.zeros((BB, BD), tl.float64)
    j0 = pid_s * J_PER
    j_end = tl.minimum(j0 + J_PER, J)
    while j0 < j_end:
        oj = j0 + tl.arange(0, BJ)
        mj = oj < j_end
        oj64 = oj.to(tl.int64)
        cm = mb[:, None] & mj[None, :]
        al = tl.load(alpha_ptr + ob64[:, None] * J + oj64[None, :], mask=cm, other=0.0)
        be = tl.load(beta_ptr + ob64[:, None] * J + oj64[None, :], mask=cm, other=0.0)
        pm = mj[:, None] & md[None, :]
        pt = tl.load(xp_ptr + oj64[:, None] * D + od[None, :], mask=pm, other=0.0)
        wt = tl.load(wp_ptr + oj64[:, None] * D + od[None, :], mask=pm, other=0.0)
        acc += tl.sum(al[:, :, None] * wt[None, :, :] + be[:, :, None] * (xt[:, None, :] - pt[None, :, :]), axis=1)
        j0 += BJ
    tl.store(out_ptr + pid_s * B * D + ob64[:, None] * D + od[None, :], acc, mask=xm)


def coefficients(kernel: Matern, device: torch.device) -> Tensor:
    """``[a, p, q, t, (t' - a t)]`` of the kernel's radial functions, float64 on ``device`` (cached)."""
    cache = kernel.__dict__.setdefault("_fast_coefficients", {})
    if device not in cache:
        tp = (_deriv(kernel.t) + [0.0] * len(kernel.t))[:len(kernel.t)]
        v = [d - kernel.a * c for d, c in zip(tp, kernel.t)]
        cache[device] = torch.tensor([kernel.a] + kernel.p + kernel.q + kernel.t + v, dtype=torch.float64,
                                     device=device)
    return cache[device]


def _pair_pass(x, v, xp, wp, ec, kernel, store: bool) -> Tuple[Optional[Tensor], Optional[Tensor], Tensor]:
    """``alpha``, ``beta`` ``(B, J)`` (with ``store``) and the summed partials ``(B,)`` of :func:`_pairs_kernel`."""
    b, d = x.shape
    j = xp.shape[0]
    nb, nj = triton.cdiv(b, BLOCK_B), triton.cdiv(j, BLOCK_J)
    part = x.new_empty(nj, b)
    alpha = x.new_empty(b, j) if store else part
    beta = x.new_empty(b, j) if store else part
    has_e = ec.numel() > 0
    _pairs_kernel[(nb, nj)](x, x if v is None else v, xp, wp, ec if has_e else x, coefficients(kernel, x.device),
                            alpha, beta, part, b, j, d, HAS_E=has_e, HVP=v is not None, STORE=store,
                            NP=len(kernel.p), NQ=len(kernel.q), NT=len(kernel.t),
                            BB=BLOCK_B, BJ=BLOCK_J, BD=BLOCK_D)
    return (alpha, beta, part.sum(0)) if store else (None, None, part.sum(0))


@lru_cache(maxsize=None)
def _multiprocessors(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def target_programs(device: torch.device) -> int:
    """Contraction programs that fill ``device``: ``PROGRAMS_PER_SM`` per multiprocessor
    (as if one multiprocessor off CUDA, where only Triton's interpreter runs them)."""
    if device.type != "cuda":
        return PROGRAMS_PER_SM
    index = torch.cuda.current_device() if device.index is None else device.index
    return PROGRAMS_PER_SM * _multiprocessors(index)


def _contract(x, xp, wp, alpha, beta) -> Tensor:
    """``(B, D)`` sums ``sum_j alpha_bj w_j + beta_bj (x_b - xp_j)``."""
    b, d = x.shape
    j = xp.shape[0]
    nb, nd = triton.cdiv(b, BLOCK_B), triton.cdiv(d, BLOCK_D)
    split = max(1, min(triton.cdiv(j, BLOCK_J), target_programs(x.device) // (nb * nd)))
    per = triton.cdiv(triton.cdiv(j, split), BLOCK_J) * BLOCK_J
    split = triton.cdiv(j, per)
    out = x.new_empty(split, b, d)
    _contract_kernel[(nb, nd, split)](x, xp, wp, alpha, beta, out, b, j, d, per,
                                      BB=BLOCK_B, BJ=BLOCK_J, BD=BLOCK_D)
    return out.sum(0)


def _flatten(xp: Tensor, wp: Tensor, ec: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """The ``(M Q, D)`` training terms and their energy coefficients."""
    m, q, d = xp.shape
    ec = ec.repeat_interleave(q).contiguous() if ec.numel() > 0 else ec
    return xp.reshape(m * q, d).contiguous(), wp.reshape(m * q, d).contiguous(), ec


def energy_gradient(x: Tensor, xp: Tensor, wp: Tensor, ec: Tensor, kernel: Matern,
                    gradient: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
    """Energies ``(B,)`` and (with ``gradient``) gradients ``(B, D)`` of the kernel expansion.

    The same sums as :func:`xnn.krnn.models.sgdml.kernel_energy`, in two kernels:
    one forms the kernel factors of every pair of a structure and a training term,
    the other contracts them with the training terms.
    """
    xp, wp, ec = _flatten(xp, wp, ec)
    x = x.contiguous()
    step = max(1, PAIR_CHUNK // max(1, xp.shape[0]))
    es, gs = [], []
    for s in range(0, x.shape[0], step):
        xs = x[s:s + step]
        alpha, beta, e = _pair_pass(xs, None, xp, wp, ec, kernel, gradient)
        es.append(e)
        if gradient:
            gs.append(_contract(xs, xp, wp, alpha, beta))
    return torch.cat(es), (torch.cat(gs) if gradient else None)


def hessian_vector(x: Tensor, v: Tensor, xp: Tensor, wp: Tensor, ec: Tensor, kernel: Matern) -> Tensor:
    """``(B, D)`` products of the Hessians of the kernel expansion with the directions ``v``.

    ``sum_j t (D.v) w + [u (D.v) s + t (w.v) - a t (D.v)] D + (t s - g a) v`` with
    ``u = t'/r`` (zero at ``r = 0``, where its term vanishes).
    """
    xp, wp, ec = _flatten(xp, wp, ec)
    x, v = x.contiguous(), v.contiguous()
    step = max(1, PAIR_CHUNK // max(1, xp.shape[0]))
    out = []
    for s in range(0, x.shape[0], step):
        xs, vs = x[s:s + step], v[s:s + step]
        alpha, beta, gamma = _pair_pass(xs, vs, xp, wp, ec, kernel, True)
        out.append(_contract(xs, xp, wp, alpha, beta) + gamma[:, None] * vs)
    return torch.cat(out)


class KernelEnergy(torch.autograd.Function):
    """:func:`energy_gradient` with the gradient as the derivative and the Hessian beyond it."""

    @staticmethod
    def forward(ctx, x, xp, wp, ec, kernel):
        ctx.kernel = kernel
        if not ctx.needs_input_grad[0]:
            return energy_gradient(x, xp, wp, ec, kernel, gradient=False)[0]
        e, grad = energy_gradient(x, xp, wp, ec, kernel)
        ctx.save_for_backward(x, xp, wp, ec, grad)
        return e

    @staticmethod
    def backward(ctx, grad_e):
        x, xp, wp, ec, grad = ctx.saved_tensors
        if torch.is_grad_enabled():
            grad = KernelGradient.apply(x, grad, xp, wp, ec, ctx.kernel)
        return grad_e[:, None] * grad, None, None, None, None


class KernelGradient(torch.autograd.Function):
    """The gradient formed by :class:`KernelEnergy`, differentiable once more (Hessian products)."""

    @staticmethod
    def forward(ctx, x, grad, xp, wp, ec, kernel):
        ctx.kernel = kernel
        ctx.save_for_backward(x, xp, wp, ec)
        return grad.clone()

    @staticmethod
    @once_differentiable
    def backward(ctx, v):
        x, xp, wp, ec = ctx.saved_tensors
        return hessian_vector(x, v, xp, wp, ec, ctx.kernel), None, None, None, None, None
