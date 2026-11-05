"""Triton kernels and autograd function of the PaiNN fast path (see :mod:`xnn.gnn.fast.painn`).

Imported only when the fast path runs, so that ``triton`` is needed only there.
"""
# no postponed annotations: Triton's interpreter (TRITON_INTERPRET=1) recognizes the
# tl.constexpr parameters of a kernel only from evaluated annotations

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


@triton.jit
def _double_kernel(x_ptr, tx_ptr, w_ptr, w_stride, tw_ptr, dir_ptr, tdir_ptr, v_ptr, tv_ptr,
                   src_ptr, dst_ptr, gs_ptr, gv_ptr, dgs_ptr, dgv_ptr, dx_ptr, dw_ptr, ddir_ptr,
                   dv_ptr, n_edges, F: tl.constexpr, BLOCK_E: tl.constexpr, HAS_TX: tl.constexpr,
                   HAS_TW: tl.constexpr, HAS_TDIR: tl.constexpr, HAS_TV: tl.constexpr):
    """The derivatives of ``<tx, gx> + <tW, gW> + <tdir, gdir> + <tv, gv>`` (the backward's
    outputs against their incoming gradients) with respect to the incoming gradients
    ``g_s``, ``g_v`` (onto ``dst``: the forward's tangent), ``x`` and ``v`` (onto ``src``),
    ``W`` and the directions. With the tangents of the products ``A = tx W + x tW`` and the
    sums over the three components ``G_vv = sum_k g_k v_k``, ``G_vs = sum_k g_k d_k``,
    ``T_vv = sum_k g_k tv_k``, ``T_vs = sum_k g_k td_k``:
    ``dg_s = A_s``, ``dg_k = A_vv v_k + A_vs d_k + p_vs td_k + p_vv tv_k``,
    ``dx = (g_s tW_s, tW_vv G_vv + W_vv T_vv, tW_vs G_vs + W_vs T_vs)``,
    ``dW = (g_s tx_s, tx_vv G_vv + x_vv T_vv, tx_vs G_vs + x_vs T_vs)``,
    ``dd_k = sum_f g_k A_vs`` and ``dv_k = g_k A_vv``."""
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
    a_s = tl.zeros_like(x_s)
    a_vv = tl.zeros_like(x_s)
    a_vs = tl.zeros_like(x_s)
    if HAS_TX:
        tx_row = tx_ptr + src * (3 * F) + f
        tx_s = tl.load(tx_row, mask=m2, other=0.0)
        tx_vv = tl.load(tx_row + F, mask=m2, other=0.0)
        tx_vs = tl.load(tx_row + 2 * F, mask=m2, other=0.0)
        a_s += tx_s * w_s
        a_vv += tx_vv * w_vv
        a_vs += tx_vs * w_vs
    if HAS_TW:
        tw_row = tw_ptr + e[:, None] * (3 * F) + f
        tw_s = tl.load(tw_row, mask=m2, other=0.0)
        tw_vv = tl.load(tw_row + F, mask=m2, other=0.0)
        tw_vs = tl.load(tw_row + 2 * F, mask=m2, other=0.0)
        a_s += x_s * tw_s
        a_vv += x_vv * tw_vv
        a_vs += x_vs * tw_vs
    tl.atomic_add(dgs_ptr + dst * F + f, a_s, mask=m2)
    g_s = tl.load(gs_ptr + dst * F + f, mask=m2, other=0.0)
    G_vv = tl.zeros_like(x_s)
    G_vs = tl.zeros_like(x_s)
    T_vv = tl.zeros_like(x_s)
    T_vs = tl.zeros_like(x_s)
    for k in tl.static_range(3):
        d = tl.load(dir_ptr + e * 3 + k, mask=me, other=0.0)[:, None]
        vk = tl.load(v_ptr + src * (3 * F) + k * F + f, mask=m2, other=0.0)
        gk = tl.load(gv_ptr + dst * (3 * F) + k * F + f, mask=m2, other=0.0)
        G_vv += gk * vk
        G_vs += gk * d
        dgk = a_vv * vk + a_vs * d
        if HAS_TDIR:
            td = tl.load(tdir_ptr + e * 3 + k, mask=me, other=0.0)[:, None]
            dgk += p_vs * td
            T_vs += gk * td
        if HAS_TV:
            tvk = tl.load(tv_ptr + src * (3 * F) + k * F + f, mask=m2, other=0.0)
            dgk += p_vv * tvk
            T_vv += gk * tvk
        tl.atomic_add(dgv_ptr + dst * (3 * F) + k * F + f, dgk, mask=m2)
        tl.store(ddir_ptr + e * 3 + k, tl.sum(gk * a_vs, axis=1), mask=me)
        tl.atomic_add(dv_ptr + src * (3 * F) + k * F + f, gk * a_vv, mask=m2)
    dx_s = tl.zeros_like(x_s)
    dx_vv = w_vv * T_vv
    dx_vs = w_vs * T_vs
    if HAS_TW:
        dx_s += g_s * tw_s
        dx_vv += tw_vv * G_vv
        dx_vs += tw_vs * G_vs
    dx_row = dx_ptr + src * (3 * F) + f
    tl.atomic_add(dx_row, dx_s, mask=m2)
    tl.atomic_add(dx_row + F, dx_vv, mask=m2)
    tl.atomic_add(dx_row + 2 * F, dx_vs, mask=m2)
    dw_s = tl.zeros_like(x_s)
    dw_vv = x_vv * T_vv
    dw_vs = x_vs * T_vs
    if HAS_TX:
        dw_s += g_s * tx_s
        dw_vv += tx_vv * G_vv
        dw_vs += tx_vs * G_vs
    dw_row = dw_ptr + e[:, None] * (3 * F) + f
    tl.store(dw_row, dw_s, mask=m2)
    tl.store(dw_row + F, dw_vv, mask=m2)
    tl.store(dw_row + 2 * F, dw_vs, mask=m2)


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
    def backward(ctx, g_ds, g_dv):
        x, W, direction, v, src, dst = ctx.saved_tensors
        g_ds = torch.zeros_like(x[:, : v.shape[2]]) if g_ds is None else g_ds.contiguous()
        g_dv = torch.zeros_like(v) if g_dv is None else g_dv.contiguous()
        return (*MessageGradFunction.apply(x, W, direction, v, g_ds, g_dv, src, dst), None, None)


class MessageGradFunction(torch.autograd.Function):
    """The backward of :class:`MessageFunction` (the gradients of ``x``, ``W``, the
    directions and ``v`` for the incoming gradients), differentiable once more."""

    @staticmethod
    def forward(ctx, x: Tensor, W: Tensor, direction: Tensor, v: Tensor, g_ds: Tensor,
                g_dv: Tensor, src: Tensor, dst: Tensor):
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
        ctx.save_for_backward(x, W, direction, v, g_ds, g_dv, src, dst)
        return gx, gW, gdir, gv

    @staticmethod
    @once_differentiable
    def backward(ctx, tx, tW, tdir, tv):
        x, W, direction, v, g_ds, g_dv, src, dst = ctx.saved_tensors
        tx, tW, tdir, tv = (None if t is None else t.contiguous() for t in (tx, tW, tdir, tv))
        n_edges, F = W.shape[0], v.shape[2]
        dgs = torch.zeros_like(g_ds)
        dgv = torch.zeros_like(g_dv)
        dx = torch.zeros_like(x)
        dv = torch.zeros_like(v)
        dW = W.new_empty((n_edges, 3 * F))
        ddir = direction.new_empty(direction.shape)
        if n_edges:
            block, warps = _launch(F)
            _double_kernel[(triton.cdiv(n_edges, block),)](
                x, x if tx is None else tx, W, W.stride(0), dW if tW is None else tW, direction,
                direction if tdir is None else tdir, v, v if tv is None else tv, src, dst, g_ds,
                g_dv, dgs, dgv, dx, dW, ddir, dv, n_edges, F=F, BLOCK_E=block,
                HAS_TX=tx is not None, HAS_TW=tW is not None, HAS_TDIR=tdir is not None,
                HAS_TV=tv is not None, num_warps=warps)
        return dx, dW, ddir, dv, dgs, dgv, None, None


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
