"""Triton kernels and autograd function of the SchNet fast path (see :mod:`xnn.gnn.fast.schnet`).

Imported only when the fast path runs, so that ``triton`` is needed only there.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import once_differentiable

from xnn.common.models.ops import exact_softplus

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
def _forward_kernel(h_ptr, z_ptr, env_ptr, src_ptr, dst_ptr, out_ptr, ln2_ptr, n_edges,
                    F: tl.constexpr, BLOCK_E: tl.constexpr, HAS_ENV: tl.constexpr,
                    ACT: tl.constexpr):
    """``out[dst] += h[src] * act(z) * env`` for a tile of edges (``act``: ssp or identity)."""
    e = (tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)).to(tl.int64)
    me = e < n_edges
    f = tl.arange(0, F)[None, :]
    m2 = me[:, None] & (f < F)
    src = tl.load(src_ptr + e, mask=me, other=0)
    dst = tl.load(dst_ptr + e, mask=me, other=0)
    w = tl.load(z_ptr + e[:, None] * F + f, mask=m2, other=0.0)
    if ACT:
        w = _shifted_softplus(w, tl.load(ln2_ptr))
    if HAS_ENV:
        w = w * tl.load(env_ptr + e, mask=me, other=0.0)[:, None]
    hs = tl.load(h_ptr + src[:, None] * F + f, mask=m2, other=0.0)
    tl.atomic_add(out_ptr + dst[:, None] * F + f, hs * w, mask=m2)


@triton.jit
def _backward_kernel(h_ptr, z_ptr, env_ptr, src_ptr, dst_ptr, gout_ptr, gh_ptr, dz_ptr,
                     denv_ptr, ln2_ptr, n_edges, F: tl.constexpr, BLOCK_E: tl.constexpr,
                     HAS_ENV: tl.constexpr, ACT: tl.constexpr):
    """The feature gradient (onto ``src``), the gradient of ``z`` and of the envelope."""
    e = (tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)).to(tl.int64)
    me = e < n_edges
    f = tl.arange(0, F)[None, :]
    m2 = me[:, None] & (f < F)
    src = tl.load(src_ptr + e, mask=me, other=0)
    dst = tl.load(dst_ptr + e, mask=me, other=0)
    z = tl.load(z_ptr + e[:, None] * F + f, mask=m2, other=0.0)
    if ACT:
        s = _shifted_softplus(z, tl.load(ln2_ptr))
    else:
        s = z
    hs = tl.load(h_ptr + src[:, None] * F + f, mask=m2, other=0.0)
    g = tl.load(gout_ptr + dst[:, None] * F + f, mask=m2, other=0.0)
    dw = g * hs
    if HAS_ENV:
        env = tl.load(env_ptr + e, mask=me, other=0.0)[:, None]
        tl.atomic_add(gh_ptr + src[:, None] * F + f, g * (s * env), mask=m2)
        tl.store(denv_ptr + e, tl.sum(dw * s, axis=1), mask=me)
        dz = dw * env
    else:
        tl.atomic_add(gh_ptr + src[:, None] * F + f, g * s, mask=m2)
        dz = dw
    if ACT:
        dz = dz / (1.0 + tl.exp(-z))                         # times ssp'(z) = sigmoid(z)
    tl.store(dz_ptr + e[:, None] * F + f, dz, mask=m2)


def _ln2(like: Tensor) -> Tensor:
    """``ln 2`` as a one-element tensor of the kernel's dtype and device."""
    return torch.full((1,), LN2, dtype=like.dtype, device=like.device)


def _block_edges(n_features: int) -> int:
    """Edges per program: a tile of about 4096 values."""
    return max(16, 4096 // n_features)


class CFConvFunction(torch.autograd.Function):
    """``out[i] = sum_{e: dst_e = i} h[src_e] * act(W2 ssp(W1 rbf_e + b1) + b2) * env_e``."""

    @staticmethod
    def forward(ctx, h: Tensor, rbf: Tensor, env: Tensor, w1: Tensor, b1: Tensor, w2: Tensor,
                b2: Tensor, src: Tensor, dst: Tensor, activated: bool) -> Tensor:
        has_env = env.numel() > 0
        z1 = torch.addmm(b1, rbf, w1.t())
        # ssp(z1) = softplus(z1) - ln 2: the shift moves into the second bias
        z2 = torch.addmm(b2 - LN2 * w2.sum(1), exact_softplus(z1), w2.t())
        h = h.contiguous()
        out = torch.zeros_like(h)
        n_edges, F = z2.shape
        if n_edges:
            block = _block_edges(F)
            _forward_kernel[(triton.cdiv(n_edges, block),)](
                h, z2, env if has_env else z2, src, dst, out, _ln2(z2), n_edges,
                F=F, BLOCK_E=block, HAS_ENV=has_env, ACT=activated)
        ctx.has_env = has_env
        ctx.activated = activated
        ctx.save_for_backward(h, env, z1, z2, w1, w2, src, dst)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, gout: Tensor):
        h, env, z1, z2, w1, w2, src, dst = ctx.saved_tensors
        gout = gout.contiguous()
        gh = torch.zeros_like(h)
        dz2 = torch.empty_like(z2)
        denv = torch.empty_like(env) if ctx.has_env else None
        n_edges, F = z2.shape
        if n_edges:
            block = _block_edges(F)
            _backward_kernel[(triton.cdiv(n_edges, block),)](
                h, z2, env if ctx.has_env else z2, src, dst, gout, gh, dz2,
                denv if ctx.has_env else z2, _ln2(z2), n_edges, F=F, BLOCK_E=block,
                HAS_ENV=ctx.has_env, ACT=ctx.activated)
        else:
            dz2.zero_()
            if denv is not None:
                denv.zero_()
        dz1 = (dz2 @ w2).mul_(torch.sigmoid(z1))
        grbf = dz1 @ w1
        return gh, grbf, denv, None, None, None, None, None, None, None


def cfconv(h: Tensor, rbf: Tensor, envelope, w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor,
           edge_index: Tensor, activated: bool = True) -> Tensor:
    """The fused continuous-filter convolution of one block (see the module docstring).

    Parameters
    ----------
    h : Tensor
        Features after the block's input layer, shape ``(N, F)``.
    rbf : Tensor
        Radial expansion, shape ``(E, K)``.
    envelope : Tensor or None
        Filter envelope per edge, shape ``(E,)``.
    w1, b1, w2, b2 : Tensor
        The weights and biases of the two dense layers of the filter network.
    edge_index : Tensor
        Source (row 0) and destination (row 1) of every edge.
    activated : bool, optional
        Whether the second dense layer is followed by the shifted softplus.

    Returns
    -------
    Tensor
        ``sum_j h_j o W(r_ij)`` per atom, shape ``(N, F)``.
    """
    env = envelope.contiguous() if envelope is not None else rbf.new_empty(0)
    return CFConvFunction.apply(h, rbf.contiguous(), env, w1, b1, w2, b2,
                                edge_index[0].contiguous(), edge_index[1].contiguous(), activated)
