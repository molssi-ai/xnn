"""Triton kernels and autograd functions of the SchNet fast path (see :mod:`xnn.gnn.fast.schnet`).

Imported only when the fast path runs, so that ``triton`` is needed only there.

One program handles one atom and walks its edges in a sorted order (compressed
rows, :mod:`xnn.common.models.csr`: by center for the sums onto centers, by
neighbor for the sums onto neighbors and the per-edge gradients), a chunk of
edges at a time, accumulating in registers: no atomic adds, and the same
result in every run.
"""
# no postponed annotations: Triton's interpreter (TRITON_INTERPRET=1) recognizes the
# tl.constexpr parameters of a kernel only from evaluated annotations

import math
from typing import Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import once_differentiable

from xnn.common.models.csr import orderings

LN2 = math.log(2.0)


@triton.jit
def _shifted_softplus(z, ln2):
    """``max(z, 0) + log1p(exp(-|z|)) - ln 2``, with ``log1p(t) = t log(u) / (u - 1)``, ``u = 1 + t``.

    ``ln2`` is loaded in the dtype of ``z``: a float literal would be a float32 constant.
    """
    t = tl.exp(-tl.abs(z))
    u = 1.0 + t
    log1p = tl.where(u == 1.0, t, tl.log(u) * t / (u - 1.0))
    return tl.maximum(z, 0.0) + log1p - ln2


@triton.jit
def _forward_kernel(h_ptr, z_ptr, env_ptr, src_ptr, order_ptr, ptr_ptr, out_ptr, ln2_ptr,
                    F: tl.constexpr, B: tl.constexpr, HAS_ENV: tl.constexpr, ACT: tl.constexpr):
    """``out[i] = sum_e h[src_e] * act(z_e) * env_e`` over the edges of center ``i``."""
    i = tl.program_id(0).to(tl.int64)
    b = tl.arange(0, B)[:, None]
    f = tl.arange(0, F)[None, :]
    start = tl.load(ptr_ptr + i)
    end = tl.load(ptr_ptr + i + 1)
    acc = tl.zeros([B, F], dtype=out_ptr.dtype.element_ty)
    k = start
    while k < end:
        mk = (k + b) < end
        m2 = mk & (f < F)
        e = tl.load(order_ptr + k + b, mask=mk, other=0)
        j = tl.load(src_ptr + e, mask=mk, other=0)
        w = tl.load(z_ptr + e * F + f, mask=m2, other=0.0)
        if ACT:
            w = _shifted_softplus(w, tl.load(ln2_ptr))
        if HAS_ENV:
            w = w * tl.load(env_ptr + e, mask=mk, other=0.0)
        acc += tl.load(h_ptr + j * F + f, mask=m2, other=0.0) * w
        k += B
    tl.store(out_ptr + i * F + tl.arange(0, F), tl.sum(acc, axis=0))


@triton.jit
def _backward_kernel(h_ptr, z_ptr, env_ptr, dst_ptr, order_ptr, ptr_ptr, g_ptr, gh_ptr, gz_ptr,
                     genv_ptr, ln2_ptr, F: tl.constexpr, B: tl.constexpr, HAS_ENV: tl.constexpr,
                     ACT: tl.constexpr):
    """Over the edges of neighbor ``j``: ``gh[j] = sum_e g[dst_e] act(z_e) env_e``, and per edge
    the gradients of ``z`` and of the envelope."""
    j = tl.program_id(0).to(tl.int64)
    b = tl.arange(0, B)[:, None]
    f = tl.arange(0, F)[None, :]
    start = tl.load(ptr_ptr + j)
    end = tl.load(ptr_ptr + j + 1)
    hj = tl.load(h_ptr + j * F + f)
    acc = tl.zeros([B, F], dtype=gh_ptr.dtype.element_ty)
    k = start
    while k < end:
        mk = (k + b) < end
        m2 = mk & (f < F)
        e = tl.load(order_ptr + k + b, mask=mk, other=0)
        d = tl.load(dst_ptr + e, mask=mk, other=0)
        z = tl.load(z_ptr + e * F + f, mask=m2, other=0.0)
        g = tl.load(g_ptr + d * F + f, mask=m2, other=0.0)
        if ACT:
            s = _shifted_softplus(z, tl.load(ln2_ptr))
        else:
            s = z
        dw = g * hj
        if HAS_ENV:
            c = tl.load(env_ptr + e, mask=mk, other=0.0)
            acc += g * (s * c)
            tl.store(genv_ptr + e, tl.sum(dw * s, axis=1)[:, None], mask=mk)
            gz = dw * c
        else:
            acc += g * s
            gz = dw
        if ACT:
            gz = gz / (1.0 + tl.exp(-z))                     # times ssp'(z) = sigmoid(z)
        tl.store(gz_ptr + e * F + f, gz, mask=m2)
        k += B
    tl.store(gh_ptr + j * F + tl.arange(0, F), tl.sum(acc, axis=0))


@triton.jit
def _derivatives(z, ln2_ptr, ACT: tl.constexpr):
    """``act(z)``, ``act'(z)`` and ``act''(z)`` (the identity or the shifted softplus)."""
    if ACT:
        s = _shifted_softplus(z, tl.load(ln2_ptr))
        a1 = 1.0 / (1.0 + tl.exp(-z))
        a2 = a1 * (1.0 - a1)
    else:
        s = z
        a1 = tl.full(z.shape, 1.0, z.dtype)
        a2 = tl.zeros_like(z)
    return s, a1, a2


@triton.jit
def _double_center(h_ptr, z_ptr, env_ptr, th_ptr, tz_ptr, tenv_ptr, src_ptr, order_ptr, ptr_ptr,
                   dg_ptr, ln2_ptr, F: tl.constexpr, B: tl.constexpr, HAS_ENV: tl.constexpr,
                   ACT: tl.constexpr, HAS_TH: tl.constexpr, HAS_TZ: tl.constexpr,
                   HAS_TENV: tl.constexpr):
    """Over the edges of center ``i``: the derivative with respect to ``g[i]`` of
    ``<th, gh> + <tz, gz> + <tenv, genv>``, ``sum_e th[src] s c + tz h[src] c a1 + tenv h[src] s``
    (``s, a1 = act(z), act'(z)``, ``c`` the envelope)."""
    i = tl.program_id(0).to(tl.int64)
    b = tl.arange(0, B)[:, None]
    f = tl.arange(0, F)[None, :]
    start = tl.load(ptr_ptr + i)
    end = tl.load(ptr_ptr + i + 1)
    acc = tl.zeros([B, F], dtype=dg_ptr.dtype.element_ty)
    k = start
    while k < end:
        mk = (k + b) < end
        m2 = mk & (f < F)
        e = tl.load(order_ptr + k + b, mask=mk, other=0)
        j = tl.load(src_ptr + e, mask=mk, other=0)
        z = tl.load(z_ptr + e * F + f, mask=m2, other=0.0)
        s, a1, a2 = _derivatives(z, ln2_ptr, ACT)
        if HAS_ENV:
            c = tl.load(env_ptr + e, mask=mk, other=0.0)
        else:
            c = tl.full(z.shape, 1.0, z.dtype)
        if HAS_TH:
            acc += tl.load(th_ptr + j * F + f, mask=m2, other=0.0) * s * c
        if HAS_TZ or HAS_TENV:
            hs = tl.load(h_ptr + j * F + f, mask=m2, other=0.0)
            if HAS_TZ:
                acc += tl.load(tz_ptr + e * F + f, mask=m2, other=0.0) * hs * c * a1
            if HAS_TENV:
                acc += tl.load(tenv_ptr + e, mask=mk, other=0.0) * hs * s
        k += B
    tl.store(dg_ptr + i * F + tl.arange(0, F), tl.sum(acc, axis=0))


@triton.jit
def _double_neighbor(h_ptr, z_ptr, env_ptr, g_ptr, th_ptr, tz_ptr, tenv_ptr, dst_ptr, order_ptr,
                     ptr_ptr, dh_ptr, dz_ptr, denv_ptr, ln2_ptr, F: tl.constexpr, B: tl.constexpr,
                     HAS_ENV: tl.constexpr, ACT: tl.constexpr, HAS_TH: tl.constexpr,
                     HAS_TZ: tl.constexpr, HAS_TENV: tl.constexpr):
    """Over the edges of neighbor ``j``: the derivatives of ``<th, gh> + <tz, gz> + <tenv, genv>``
    with respect to ``h[j]`` (``sum_e tz g c a1 + tenv g s``), and per edge with respect to ``z``
    (``th[j] g c a1 + tz g h[j] c a2 + tenv g h[j] a1``) and the envelope
    (``sum_f th[j] g s + tz g h[j] a1``), with ``g = g[dst_e]``."""
    j = tl.program_id(0).to(tl.int64)
    b = tl.arange(0, B)[:, None]
    f = tl.arange(0, F)[None, :]
    start = tl.load(ptr_ptr + j)
    end = tl.load(ptr_ptr + j + 1)
    hj = tl.load(h_ptr + j * F + f)
    if HAS_TH:
        thj = tl.load(th_ptr + j * F + f)
    acc = tl.zeros([B, F], dtype=dh_ptr.dtype.element_ty)
    k = start
    while k < end:
        mk = (k + b) < end
        m2 = mk & (f < F)
        e = tl.load(order_ptr + k + b, mask=mk, other=0)
        d = tl.load(dst_ptr + e, mask=mk, other=0)
        z = tl.load(z_ptr + e * F + f, mask=m2, other=0.0)
        s, a1, a2 = _derivatives(z, ln2_ptr, ACT)
        if HAS_ENV:
            c = tl.load(env_ptr + e, mask=mk, other=0.0)
        else:
            c = tl.full(z.shape, 1.0, z.dtype)
        g = tl.load(g_ptr + d * F + f, mask=m2, other=0.0)
        dz = tl.zeros_like(z)
        dc = tl.zeros_like(z)
        if HAS_TH:
            dz += thj * g * c * a1
            dc += thj * g * s
        if HAS_TZ:
            tz = tl.load(tz_ptr + e * F + f, mask=m2, other=0.0)
            acc += tz * g * c * a1
            dz += tz * g * hj * c * a2
            dc += tz * g * hj * a1
        if HAS_TENV:
            te = tl.load(tenv_ptr + e, mask=mk, other=0.0)
            acc += te * g * s
            dz += te * g * hj * a1
        tl.store(dz_ptr + e * F + f, dz, mask=m2)
        if HAS_ENV:
            tl.store(denv_ptr + e, tl.sum(dc, axis=1)[:, None], mask=mk)
        k += B
    tl.store(dh_ptr + j * F + tl.arange(0, F), tl.sum(acc, axis=0))


def _ln2(like: Tensor) -> Tensor:
    """``ln 2`` as a one-element tensor of the kernel's dtype and device."""
    return torch.full((1,), LN2, dtype=like.dtype, device=like.device)


#: Values per step (edges per step times features) and warps per program, per precision: a sweep
#: on three GPUs chose 1024 values over two warps in float32 and 512 values over one warp in
#: float64, within a fifth of the best shape of every kernel on every GPU.
SHAPES = {torch.float32: (1024, 2), torch.float64: (512, 1)}


def _launch(n_features: int, dtype: torch.dtype) -> Tuple[int, int]:
    """Edges per step and warps per program."""
    values, warps = SHAPES[dtype]
    return max(1, values // n_features), warps


def _contiguous(t):
    return None if t is None else t.contiguous()


class CFConvFunction(torch.autograd.Function):
    """``out[i] = sum_{e: dst_e = i} h[src_e] * act(z_e) * env_e``."""

    @staticmethod
    def forward(ctx, h: Tensor, z: Tensor, env: Tensor, src: Tensor, dst: Tensor, rows: tuple,
                activated: bool) -> Tensor:
        has_env = env.numel() > 0
        n, F = h.shape
        out = torch.empty_like(h)
        order_dst, ptr_dst = rows[0], rows[1]
        if n:
            B, warps = _launch(F, h.dtype)
            _forward_kernel[(n,)](h, z, env if has_env else z, src, order_dst, ptr_dst, out,
                                  _ln2(z), F=F, B=B, HAS_ENV=has_env, ACT=activated,
                                  num_warps=warps)
        ctx.activated = activated
        ctx.rows = rows
        ctx.save_for_backward(h, z, env, src, dst)
        return out

    @staticmethod
    def backward(ctx, gout: Tensor):
        h, z, env, src, dst = ctx.saved_tensors
        gh, gz, genv = CFConvGradFunction.apply(h, z, env, gout.contiguous(), src, dst, ctx.rows,
                                                ctx.activated)
        return gh, gz, (genv if env.numel() else None), None, None, None, None


class CFConvGradFunction(torch.autograd.Function):
    """The backward of :class:`CFConvFunction` (the gradients of ``h``, ``z`` and the
    envelope for the incoming gradient ``g``), differentiable once more."""

    @staticmethod
    def forward(ctx, h: Tensor, z: Tensor, env: Tensor, g: Tensor, src: Tensor, dst: Tensor,
                rows: tuple, activated: bool):
        has_env = env.numel() > 0
        n, F = h.shape
        gh = torch.empty_like(h)
        gz = torch.empty_like(z)
        genv = torch.empty_like(env)
        order_src, ptr_src = rows[2], rows[3]
        if n:
            B, warps = _launch(F, h.dtype)
            _backward_kernel[(n,)](h, z, env if has_env else z, dst, order_src, ptr_src, g, gh, gz,
                                   genv if has_env else z, _ln2(z), F=F, B=B, HAS_ENV=has_env,
                                   ACT=activated, num_warps=warps)
        ctx.activated = activated
        ctx.rows = rows
        ctx.save_for_backward(h, z, env, g, src, dst)
        return gh, gz, genv

    @staticmethod
    @once_differentiable
    def backward(ctx, th, tz, tenv):
        h, z, env, g, src, dst = ctx.saved_tensors
        has_env = env.numel() > 0
        if not has_env:
            tenv = None
        th, tz, tenv = _contiguous(th), _contiguous(tz), _contiguous(tenv)
        n, F = h.shape
        order_dst, ptr_dst, order_src, ptr_src = ctx.rows
        dg = torch.empty_like(g)
        dh = torch.empty_like(h)
        dz = torch.empty_like(z)
        denv = torch.empty_like(env)
        if n:
            B, warps = _launch(F, h.dtype)
            flags = dict(F=F, B=B, HAS_ENV=has_env, ACT=ctx.activated, HAS_TH=th is not None,
                         HAS_TZ=tz is not None, HAS_TENV=tenv is not None, num_warps=warps)
            th_, tz_, tenv_ = (h if th is None else th), (z if tz is None else tz), (z if tenv is None else tenv)
            env_ = env if has_env else z
            _double_center[(n,)](h, z, env_, th_, tz_, tenv_, src, order_dst, ptr_dst, dg, _ln2(z),
                                 **flags)
            _double_neighbor[(n,)](h, z, env_, g, th_, tz_, tenv_, dst, order_src, ptr_src, dh, dz,
                                   denv if has_env else z, _ln2(z), **flags)
        return dh, dz, (denv if has_env else None), dg, None, None, None, None


def cfconv(h: Tensor, z: Tensor, envelope, edge_index: Tensor,
           activated: bool = False) -> Tensor:
    """The fused continuous-filter convolution of one block (see :mod:`xnn.gnn.fast.schnet`).

    Parameters
    ----------
    h : Tensor
        Features after the block's input layer, shape ``(N, F)``.
    z : Tensor
        Output of the filter network's second dense layer, shape ``(E, F)``.
    envelope : Tensor or None
        Filter envelope per edge, shape ``(E,)``.
    edge_index : Tensor
        Source (row 0) and destination (row 1) of every edge.
    activated : bool, optional
        Whether the shifted softplus follows the second dense layer.

    Returns
    -------
    Tensor
        ``sum_j h_j o W(r_ij)`` per atom, shape ``(N, F)``.
    """
    env = envelope.contiguous() if envelope is not None else z.new_empty(0)
    src, dst = edge_index[0].contiguous(), edge_index[1].contiguous()
    # the blocks of one evaluation pass the same edge index: one sort for all of them
    rows = orderings(src, dst, h.shape[0], key=edge_index)
    return CFConvFunction.apply(h.contiguous(), z.contiguous(), env, src, dst, rows, activated)
