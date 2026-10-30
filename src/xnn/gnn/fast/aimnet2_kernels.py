"""Triton kernels and autograd functions of the AIMNet2 fast path (see :mod:`xnn.gnn.fast.aimnet2`).

Imported only when the fast path runs, so that ``triton`` is needed only there.

The shell sums ``z[i, c, s, k] = sum_{e: dst_e = i} a[src_e, c, s] g[e, s, k]``
take one program per center over its edges in center order; their gradients
``gg[e, s, k] = sum_c w[dst_e, c, s, k] a[src_e, c, s]`` and the edge's share
``u[e, c, s] = sum_k w[dst_e, c, s, k] g[e, s, k]`` one program per center (its
gradient row read once), and ``ga[j] = sum_{e: src_e = j} u[e]`` one program per
neighbor over its edges in neighbor order (compressed rows,
:mod:`xnn.common.models.csr`). The sums run in registers, without atomic adds,
and come out the same in every run.
"""
# no postponed annotations: Triton's interpreter (TRITON_INTERPRET=1) recognizes the
# tl.constexpr parameters of a kernel only from evaluated annotations

from typing import Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor

from xnn.common.models.csr import orderings
from xnn.common.models.dispersion_fast._triton import next_power_of_two


@triton.jit
def _sums_kernel(a_ptr, g_ptr, src_ptr, order_ptr, ptr_ptr, out_ptr,
                 C: tl.constexpr, CP: tl.constexpr, S: tl.constexpr, K: tl.constexpr):
    """``out[i] = sum_e a[src_e] (x) g_e`` (outer product over ``k``) over the edges of center ``i``."""
    i = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, CP)[:, None, None]
    s = tl.arange(0, S)[None, :, None]
    k = tl.arange(0, K)[None, None, :]
    mc = c < C
    start = tl.load(ptr_ptr + i)
    end = tl.load(ptr_ptr + i + 1)
    acc = tl.zeros([CP, S, K], dtype=out_ptr.dtype.element_ty)
    n = start
    while n < end:
        e = tl.load(order_ptr + n)
        j = tl.load(src_ptr + e)
        av = tl.load(a_ptr + j * (C * S) + c * S + s, mask=mc, other=0.0)
        gv = tl.load(g_ptr + e * (S * K) + s * K + k)
        acc += av * gv
        n += 1
    tl.store(out_ptr + i * (C * S * K) + c * (S * K) + s * K + k, acc, mask=mc & (s < S) & (k < K))


@triton.jit
def _grads_kernel(w_ptr, g_ptr, a_ptr, src_ptr, order_ptr, ptr_ptr, u_ptr, gg_ptr,
                  C: tl.constexpr, CP: tl.constexpr, S: tl.constexpr, K: tl.constexpr):
    """Over the edges of center ``i`` (its gradient row loaded once): per edge
    ``gg_e = sum_c w[i] a[src_e]`` and the edge's share of the neighbor's gradient,
    ``u_e = sum_k w[i] g_e``."""
    i = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, CP)[:, None, None]
    s = tl.arange(0, S)[None, :, None]
    k = tl.arange(0, K)[None, None, :]
    mc = c < C
    s2 = tl.arange(0, S)[:, None]
    k2 = tl.arange(0, K)[None, :]
    c3 = tl.arange(0, CP)[:, None]
    s3 = tl.arange(0, S)[None, :]
    wi = tl.load(w_ptr + i * (C * S * K) + c * (S * K) + s * K + k, mask=mc, other=0.0)
    start = tl.load(ptr_ptr + i)
    end = tl.load(ptr_ptr + i + 1)
    n = start
    while n < end:
        e = tl.load(order_ptr + n)
        j = tl.load(src_ptr + e)
        aj = tl.load(a_ptr + j * (C * S) + c * S + s, mask=mc, other=0.0)
        gv = tl.load(g_ptr + e * (S * K) + s * K + k)
        tl.store(gg_ptr + e * (S * K) + s2 * K + k2, tl.sum(wi * aj, axis=0))
        tl.store(u_ptr + e * (C * S) + c3 * S + s3, tl.sum(wi * gv, axis=2), mask=(c3 < C) & (s3 < S))
        n += 1


@triton.jit
def _rows_kernel(u_ptr, order_ptr, ptr_ptr, ga_ptr, CS: tl.constexpr, CSP: tl.constexpr):
    """``ga[j] = sum_e u_e`` over the edges of neighbor ``j``, in order."""
    j = tl.program_id(0).to(tl.int64)
    v = tl.arange(0, CSP)
    start = tl.load(ptr_ptr + j)
    end = tl.load(ptr_ptr + j + 1)
    acc = tl.zeros([CSP], dtype=ga_ptr.dtype.element_ty)
    n = start
    while n < end:
        e = tl.load(order_ptr + n)
        acc += tl.load(u_ptr + e * CS + v, mask=v < CS, other=0.0)
        n += 1
    tl.store(ga_ptr + j * CS + v, acc, mask=v < CS)


#: Warps per program of the sums, the gradients and the row pass of the gradients.
WARPS = {"sums": 4, "grads": 4, "rows": 2}


def _shape(a: Tensor) -> Tuple[int, int, int]:
    n, channels, shells = a.shape
    return channels, next_power_of_two(channels, 2), shells


def _sums(a: Tensor, g: Tensor, src: Tensor, rows: tuple) -> Tensor:
    channels, padded, shells = _shape(a)
    out = torch.empty(a.shape + (g.shape[-1],), dtype=a.dtype, device=a.device)
    n = a.shape[0]
    if n:
        _sums_kernel[(n,)](a, g, src, rows[0], rows[1], out, C=channels, CP=padded, S=shells,
                           K=g.shape[-1], num_warps=WARPS["sums"])
    return out


def _grads(w: Tensor, a: Tensor, g: Tensor, src: Tensor, rows: tuple) -> Tuple[Tensor, Tensor]:
    channels, padded, shells = _shape(a)
    ga = torch.empty_like(a)
    gg = torch.empty_like(g)
    u = torch.empty((g.shape[0], channels, shells), dtype=a.dtype, device=a.device)
    n = a.shape[0]
    if n:
        _grads_kernel[(n,)](w, g, a, src, rows[0], rows[1], u, gg, C=channels, CP=padded, S=shells,
                            K=g.shape[-1], num_warps=WARPS["grads"])
        _rows_kernel[(n,)](u, rows[2], rows[3], ga, CS=channels * shells,
                           CSP=next_power_of_two(channels * shells, 16), num_warps=WARPS["rows"])
    return ga, gg


class ShellSumsFunction(torch.autograd.Function):
    """``z[i] = sum_{e: dst_e = i} a[src_e] (x) g_e``, differentiable at any order."""

    @staticmethod
    def forward(ctx, a: Tensor, g: Tensor, src: Tensor, dst: Tensor, rows: tuple) -> Tensor:
        ctx.rows = rows
        ctx.save_for_backward(a, g, src, dst)
        return _sums(a, g, src, rows)

    @staticmethod
    def backward(ctx, w: Tensor):
        a, g, src, dst = ctx.saved_tensors
        ga, gg = ShellGradsFunction.apply(w.contiguous(), a, g, src, dst, ctx.rows)
        return ga, gg, None, None, None


class ShellGradsFunction(torch.autograd.Function):
    """``(sum_k w[dst] g, sum_c w[dst] a)`` onto the neighbors and edges: the gradients of
    :class:`ShellSumsFunction` for the incoming ``w``. Bilinear like the sums, so its own
    derivatives are the two functions again."""

    @staticmethod
    def forward(ctx, w: Tensor, a: Tensor, g: Tensor, src: Tensor, dst: Tensor,
                rows: tuple) -> Tuple[Tensor, Tensor]:
        ctx.rows = rows
        ctx.save_for_backward(w, a, g, src, dst)
        return _grads(w, a, g, src, rows)

    @staticmethod
    def backward(ctx, ta: Tensor, tg: Tensor):
        w, a, g, src, dst = ctx.saved_tensors
        rows = ctx.rows
        ta = None if ta is None else ta.contiguous()
        tg = None if tg is None else tg.contiguous()
        dw = None
        if ta is not None:
            dw = ShellSumsFunction.apply(ta, g, src, dst, rows)
        if tg is not None:
            term = ShellSumsFunction.apply(a, tg, src, dst, rows)
            dw = term if dw is None else dw + term
        # d/da = sum_k w tg and d/dg = sum_c w ta: the same two contractions
        da = dg = None
        if ta is not None or tg is not None:
            da, dg = ShellGradsFunction.apply(w, torch.zeros_like(a) if ta is None else ta,
                                              torch.zeros_like(g) if tg is None else tg, src, dst, rows)
        return dw, da, dg, None, None, None


def shell_sums(a: Tensor, g: Tensor, edge_index: Tensor) -> Tensor:
    """The four shell sums of the AIMNet2 convolution (see :mod:`xnn.gnn.fast.aimnet2`).

    Parameters
    ----------
    a : Tensor
        The convolved per-atom quantity, shape ``(N, C, S)``.
    g : Tensor
        Per edge the radial basis and its products with the three components
        of the unit bond vector, shape ``(E, S, 4)``.
    edge_index : Tensor
        Source (row 0, the neighbor) and destination (row 1, the center) of
        every edge.

    Returns
    -------
    Tensor
        ``sum_{j -> i} a[j, c, s] g[ij, s, k]``, shape ``(N, C, S, 4)``.
    """
    src, dst = edge_index[0].contiguous(), edge_index[1].contiguous()
    # the passes of one evaluation pass the same edge index: one sort for all of them
    rows = orderings(src, dst, a.shape[0], key=edge_index)
    return ShellSumsFunction.apply(a.contiguous(), g.contiguous(), src, dst, rows)
