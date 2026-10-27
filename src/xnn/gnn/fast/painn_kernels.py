"""Triton kernels and autograd function of the PaiNN fast path (see :mod:`xnn.gnn.fast.painn`).

Imported only when the fast path runs, so that ``triton`` is needed only there.
"""
from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import once_differentiable


@triton.jit
def _forward_kernel(x_ptr, w_ptr, w_stride, dir_ptr, v_ptr, src_ptr, dst_ptr, ds_ptr, dv_ptr,
                    n_edges, F: tl.constexpr, BLOCK_E: tl.constexpr):
    """``ds[dst] += x_s[src] W_s`` and ``dv[dst] += x_vs[src] W_vs dir + x_vv[src] W_vv v[src]``."""
    e = (tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)).to(tl.int64)
    me = e < n_edges
    f = tl.arange(0, F)[None, :]
    m2 = me[:, None] & (f < F)
    src = tl.load(src_ptr + e, mask=me, other=0)[:, None]
    dst = tl.load(dst_ptr + e, mask=me, other=0)[:, None]
    x_row = x_ptr + src * (3 * F) + f
    w_row = w_ptr + e[:, None] * w_stride + f
    p_s = tl.load(x_row, mask=m2, other=0.0) * tl.load(w_row, mask=m2, other=0.0)
    p_vv = tl.load(x_row + F, mask=m2, other=0.0) * tl.load(w_row + F, mask=m2, other=0.0)
    p_vs = tl.load(x_row + 2 * F, mask=m2, other=0.0) * tl.load(w_row + 2 * F, mask=m2, other=0.0)
    tl.atomic_add(ds_ptr + dst * F + f, p_s, mask=m2)
    for k in tl.static_range(3):
        d = tl.load(dir_ptr + e * 3 + k, mask=me, other=0.0)[:, None]
        vk = tl.load(v_ptr + src * (3 * F) + k * F + f, mask=m2, other=0.0)
        tl.atomic_add(dv_ptr + dst * (3 * F) + k * F + f, p_vs * d + p_vv * vk, mask=m2)


@triton.jit
def _backward_kernel(x_ptr, w_ptr, w_stride, dir_ptr, v_ptr, src_ptr, dst_ptr, gs_ptr, gv_ptr,
                     gx_ptr, gw_ptr, gdir_ptr, gvin_ptr, n_edges, F: tl.constexpr,
                     BLOCK_E: tl.constexpr):
    """The gradients of ``x`` and ``v`` (onto ``src``), of ``W`` and of the directions."""
    e = (tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)).to(tl.int64)
    me = e < n_edges
    f = tl.arange(0, F)[None, :]
    m2 = me[:, None] & (f < F)
    src = tl.load(src_ptr + e, mask=me, other=0)[:, None]
    dst = tl.load(dst_ptr + e, mask=me, other=0)[:, None]
    x_row = x_ptr + src * (3 * F) + f
    w_row = w_ptr + e[:, None] * w_stride + f
    x_s = tl.load(x_row, mask=m2, other=0.0)
    x_vv = tl.load(x_row + F, mask=m2, other=0.0)
    x_vs = tl.load(x_row + 2 * F, mask=m2, other=0.0)
    w_s = tl.load(w_row, mask=m2, other=0.0)
    w_vv = tl.load(w_row + F, mask=m2, other=0.0)
    w_vs = tl.load(w_row + 2 * F, mask=m2, other=0.0)
    p_vv = x_vv * w_vv
    p_vs = x_vs * w_vs
    g_s = tl.load(gs_ptr + dst * F + f, mask=m2, other=0.0)
    g_vv = tl.zeros_like(g_s)        # d L / d p_vv = sum_k g_v[k] v[src][k]
    g_vs = tl.zeros_like(g_s)        # d L / d p_vs = sum_k g_v[k] dir[k]
    for k in tl.static_range(3):
        d = tl.load(dir_ptr + e * 3 + k, mask=me, other=0.0)[:, None]
        gk = tl.load(gv_ptr + dst * (3 * F) + k * F + f, mask=m2, other=0.0)
        vk = tl.load(v_ptr + src * (3 * F) + k * F + f, mask=m2, other=0.0)
        g_vv += gk * vk
        g_vs += gk * d
        tl.store(gdir_ptr + e * 3 + k, tl.sum(gk * p_vs, axis=1), mask=me)
        tl.atomic_add(gvin_ptr + src * (3 * F) + k * F + f, gk * p_vv, mask=m2)
    gx_row = gx_ptr + src * (3 * F) + f
    tl.atomic_add(gx_row, g_s * w_s, mask=m2)
    tl.atomic_add(gx_row + F, g_vv * w_vv, mask=m2)
    tl.atomic_add(gx_row + 2 * F, g_vs * w_vs, mask=m2)
    gw_row = gw_ptr + e[:, None] * (3 * F) + f
    tl.store(gw_row, g_s * x_s, mask=m2)
    tl.store(gw_row + F, g_vv * x_vv, mask=m2)
    tl.store(gw_row + 2 * F, g_vs * x_vs, mask=m2)


#: Values per tile (edges per program times features) and warps per program. The backward
#: holds about a dozen tiles at once; small tiles over eight warps keep them in registers,
#: which on the GPUs measured is several times faster than tiles of a few thousand values.
TILE, WARPS = 512, 8


def _launch(n_features: int) -> Tuple[int, int]:
    """Edges per program and warps per program."""
    return max(4, TILE // n_features), WARPS


class MessageFunction(torch.autograd.Function):
    """The message block of PaiNN on given ``x = phi(s)`` and filters ``W``."""

    @staticmethod
    def forward(ctx, x: Tensor, W: Tensor, direction: Tensor, v: Tensor, src: Tensor,
                dst: Tensor) -> Tuple[Tensor, Tensor]:
        n, F = v.shape[0], v.shape[2]
        ds = x.new_zeros((n, F))
        dv = x.new_zeros((n, 3, F))
        n_edges = W.shape[0]
        if n_edges:
            block, warps = _launch(F)
            _forward_kernel[(triton.cdiv(n_edges, block),)](
                x, W, W.stride(0), direction, v, src, dst, ds, dv, n_edges, F=F, BLOCK_E=block,
                num_warps=warps)
        ctx.save_for_backward(x, W, direction, v, src, dst)
        return ds, dv

    @staticmethod
    @once_differentiable
    def backward(ctx, g_ds, g_dv):
        x, W, direction, v, src, dst = ctx.saved_tensors
        g_ds = torch.zeros_like(x[:, : v.shape[2]]) if g_ds is None else g_ds.contiguous()
        g_dv = torch.zeros_like(v) if g_dv is None else g_dv.contiguous()
        gx = torch.zeros_like(x)
        gv = torch.zeros_like(v)
        n_edges, F = W.shape[0], v.shape[2]
        gW = W.new_empty((n_edges, 3 * F))
        gdir = direction.new_empty(direction.shape)
        if n_edges:
            block, warps = _launch(F)
            _backward_kernel[(triton.cdiv(n_edges, block),)](
                x, W, W.stride(0), direction, v, src, dst, g_ds, g_dv, gx, gW, gdir, gv, n_edges,
                F=F, BLOCK_E=block, num_warps=warps)
        return gx, gW, gdir, gv, None, None


def message(x: Tensor, W: Tensor, direction: Tensor, v: Tensor,
            edge_index: Tensor) -> Tuple[Tensor, Tensor]:
    """The fused message block (eq 7 and 8 of the PaiNN paper).

    Parameters
    ----------
    x : Tensor
        ``phi(s)``, shape ``(N, 3F)``, parts in the order ``(s, vv, vs)``.
    W : Tensor
        The block's filters, the cutoff included, shape ``(E, 3F)`` (rows may be
        strided, as for one block's columns of the stacked filters).
    direction : Tensor
        Unit vectors ``r_ij / |r_ij|``, shape ``(E, 3)``.
    v : Tensor
        Vector features, shape ``(N, 3, F)``.
    edge_index : Tensor
        Neighbor (row 0) and center (row 1) of every edge.

    Returns
    -------
    tuple of Tensor
        ``ds`` of shape ``(N, F)`` and ``dv`` of shape ``(N, 3, F)``.
    """
    if W.stride(1) != 1:
        W = W.contiguous()
    return MessageFunction.apply(x.contiguous(), W, direction.contiguous(), v.contiguous(),
                                 edge_index[0].contiguous(), edge_index[1].contiguous())
