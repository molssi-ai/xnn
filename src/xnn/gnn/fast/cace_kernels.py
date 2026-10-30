"""Triton kernels and autograd functions of the CACE fast path (see :mod:`xnn.gnn.fast.cace`).

Imported only when the fast path runs, so that ``triton`` is needed only there.

The ``Ar`` messages summed onto the centers,
``m[i, a, s] = sum_{e: dst_e = i} A[src_e, a, s] D[e, g(a), s]`` (``a`` the angular
entry, ``g(a)`` its order, ``s`` the radial and edge channels), take one program per
center over its edges in center order. Their gradients take one program per center as
well (its gradient row read once): per edge ``dD[e, g, s] = sum_{a: g(a) = g} w[i, a, s]
A[src_e, a, s]`` and the edge's share ``u[e, a, s] = w[i, a, s] D[e, g(a), s]``, then one
program per neighbor sums the shares of its edges (compressed rows,
:mod:`xnn.common.models.csr`). No atomic adds: the sums come out the same in every run.
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
def _sums_kernel(a_ptr, d_ptr, order_ptr_g, src_ptr, order_ptr, ptr_ptr, out_ptr,
                 L: tl.constexpr, LP: tl.constexpr, G: tl.constexpr, S: tl.constexpr, SP: tl.constexpr):
    """``out[i] = sum_e A[src_e] * D_e[g(a)]`` over the edges of center ``i``."""
    i = tl.program_id(0).to(tl.int64)
    a = tl.arange(0, LP)[:, None]
    s = tl.arange(0, SP)[None, :]
    m = (a < L) & (s < S)
    g = tl.load(order_ptr_g + tl.arange(0, LP), mask=tl.arange(0, LP) < L, other=0)[:, None]
    start = tl.load(ptr_ptr + i)
    end = tl.load(ptr_ptr + i + 1)
    acc = tl.zeros([LP, SP], dtype=out_ptr.dtype.element_ty)
    n = start
    while n < end:
        e = tl.load(order_ptr + n)
        j = tl.load(src_ptr + e)
        av = tl.load(a_ptr + j * (L * S) + a * S + s, mask=m, other=0.0)
        dv = tl.load(d_ptr + e * (G * S) + g * S + s, mask=m, other=0.0)
        acc += av * dv
        n += 1
    tl.store(out_ptr + i * (L * S) + a * S + s, acc, mask=m)


@triton.jit
def _grads_kernel(w_ptr, a_ptr, d_ptr, order_ptr_g, src_ptr, order_ptr, ptr_ptr, u_ptr, gd_ptr,
                  L: tl.constexpr, LP: tl.constexpr, G: tl.constexpr, S: tl.constexpr, SP: tl.constexpr):
    """Over the edges of center ``i``: ``dD_e[g] = sum_{a in g} w[i] A[src_e]`` and the share
    ``u_e = w[i] D_e[g(a)]``."""
    i = tl.program_id(0).to(tl.int64)
    a = tl.arange(0, LP)[:, None]
    s = tl.arange(0, SP)[None, :]
    m = (a < L) & (s < S)
    g = tl.load(order_ptr_g + tl.arange(0, LP), mask=tl.arange(0, LP) < L, other=G)[:, None]
    s1 = tl.arange(0, SP)
    wi = tl.load(w_ptr + i * (L * S) + a * S + s, mask=m, other=0.0)
    start = tl.load(ptr_ptr + i)
    end = tl.load(ptr_ptr + i + 1)
    n = start
    while n < end:
        e = tl.load(order_ptr + n)
        j = tl.load(src_ptr + e)
        av = tl.load(a_ptr + j * (L * S) + a * S + s, mask=m, other=0.0)
        dv = tl.load(d_ptr + e * (G * S) + g * S + s, mask=m, other=0.0)
        tl.store(u_ptr + e * (L * S) + a * S + s, wi * dv, mask=m)
        prod = wi * av
        for k in tl.static_range(G):
            tl.store(gd_ptr + e * (G * S) + k * S + s1, tl.sum(tl.where(g == k, prod, 0.0), axis=0),
                     mask=s1 < S)
        n += 1


@triton.jit
def _rows_kernel(u_ptr, order_ptr, ptr_ptr, ga_ptr, LS: tl.constexpr, LSP: tl.constexpr):
    """``ga[j] = sum_e u_e`` over the edges of neighbor ``j``, in order."""
    j = tl.program_id(0).to(tl.int64)
    v = tl.arange(0, LSP)
    start = tl.load(ptr_ptr + j)
    end = tl.load(ptr_ptr + j + 1)
    acc = tl.zeros([LSP], dtype=ga_ptr.dtype.element_ty)
    n = start
    while n < end:
        e = tl.load(order_ptr + n)
        acc += tl.load(u_ptr + e * LS + v, mask=v < LS, other=0.0)
        n += 1
    tl.store(ga_ptr + j * LS + v, acc, mask=v < LS)


def _sums(a: Tensor, d: Tensor, groups: Tensor, src: Tensor, rows: tuple) -> Tensor:
    n, n_l, n_s = a.shape
    out = torch.empty_like(a)
    if n:
        _sums_kernel[(n,)](a, d, groups, src, rows[0], rows[1], out, L=n_l, LP=next_power_of_two(n_l, 2),
                           G=d.shape[1], S=n_s, SP=next_power_of_two(n_s, 16), num_warps=4)
    return out


def _grads(w: Tensor, a: Tensor, d: Tensor, groups: Tensor, src: Tensor, rows: tuple) -> Tuple[Tensor, Tensor]:
    n, n_l, n_s = a.shape
    ga = torch.empty_like(a)
    gd = torch.empty_like(d)
    u = torch.empty((d.shape[0], n_l, n_s), dtype=a.dtype, device=a.device)
    if n:
        _grads_kernel[(n,)](w, a, d, groups, src, rows[0], rows[1], u, gd, L=n_l, LP=next_power_of_two(n_l, 2),
                            G=d.shape[1], S=n_s, SP=next_power_of_two(n_s, 16), num_warps=4)
        _rows_kernel[(n,)](u, rows[2], rows[3], ga, LS=n_l * n_s, LSP=next_power_of_two(n_l * n_s, 16),
                           num_warps=4)
    return ga, gd


class MessageSumsFunction(torch.autograd.Function):
    """``m[i] = sum_{e: dst_e = i} A[src_e] * D_e[g(a)]``, differentiable at any order."""

    @staticmethod
    def forward(ctx, a: Tensor, d: Tensor, groups: Tensor, src: Tensor, dst: Tensor, rows: tuple) -> Tensor:
        ctx.rows = rows
        ctx.save_for_backward(a, d, groups, src, dst)
        return _sums(a, d, groups, src, rows)

    @staticmethod
    def backward(ctx, w: Tensor):
        a, d, groups, src, dst = ctx.saved_tensors
        ga, gd = MessageGradsFunction.apply(w.contiguous(), a, d, groups, src, dst, ctx.rows)
        return ga, gd, None, None, None, None


class MessageGradsFunction(torch.autograd.Function):
    """The gradients of :class:`MessageSumsFunction` for the incoming ``w``: bilinear like the
    sums, so its own derivatives are the two functions again."""

    @staticmethod
    def forward(ctx, w: Tensor, a: Tensor, d: Tensor, groups: Tensor, src: Tensor, dst: Tensor,
                rows: tuple) -> Tuple[Tensor, Tensor]:
        ctx.rows = rows
        ctx.save_for_backward(w, a, d, groups, src, dst)
        return _grads(w, a, d, groups, src, rows)

    @staticmethod
    def backward(ctx, ta: Tensor, td: Tensor):
        w, a, d, groups, src, dst = ctx.saved_tensors
        rows = ctx.rows
        ta = None if ta is None else ta.contiguous()
        td = None if td is None else td.contiguous()
        dw = None
        if ta is not None:
            dw = MessageSumsFunction.apply(ta, d, groups, src, dst, rows)
        if td is not None:
            term = MessageSumsFunction.apply(a, td, groups, src, dst, rows)
            dw = term if dw is None else dw + term
        # d/da = sum w td[g(a)] and d/dd = sum_{a in g} w ta: the same two contractions
        da = dd = None
        if ta is not None or td is not None:
            da, dd = MessageGradsFunction.apply(w, torch.zeros_like(a) if ta is None else ta,
                                                torch.zeros_like(d) if td is None else td, groups, src, dst, rows)
        return dw, da, dd, None, None, None, None


def message_sums(a: Tensor, d: Tensor, groups: Tensor, edge_index: Tensor) -> Tensor:
    """The ``Ar`` messages summed onto the centers (see :mod:`xnn.gnn.fast.cace`).

    Parameters
    ----------
    a : Tensor
        The senders' features, shape ``(N, L, S)`` (angular entries, then the radial and
        edge channels flattened).
    d : Tensor
        The decay of every edge per angular order, shape ``(E, G, S)``.
    groups : Tensor
        The order ``g(a)`` of every angular entry, shape ``(L,)``, int32 or int64.
    edge_index : Tensor
        Source (row 0, the sender) and destination (row 1, the center) of every edge.

    Returns
    -------
    Tensor
        ``sum_{e -> i} a[src_e, a, s] d[e, g(a), s]``, shape ``(N, L, S)``.
    """
    src, dst = edge_index[0].contiguous(), edge_index[1].contiguous()
    rows = orderings(src, dst, a.shape[0], key=edge_index)
    return MessageSumsFunction.apply(a.contiguous(), d.contiguous(), groups.contiguous(), src, dst, rows)
