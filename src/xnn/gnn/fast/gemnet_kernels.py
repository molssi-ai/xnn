"""Triton kernels and autograd functions of the GemNet fast path (see :mod:`xnn.gnn.fast.gemnet`).

Imported only when the fast path runs, so that ``triton`` is needed only there.

The neighbor sums of the efficient bilinear layer,
``S[e, a, f] = sum_{t: out_t = e} s[t, a] x[in_t, f]``, split the items of every
output edge (triplets or quadruplets, in output order) into chunks of at most
:data:`CHUNK`: one program per chunk forms the chunk's product ``s^T x`` (a
matrix-unit tile, IEEE float32 or float64) and one program per output adds
its chunks' partial sums in order, so long rows (quadruplets) occupy the whole
GPU. The gradients take one program per chunk as well (the output's gradient
row read once): per item ``gs[t, a] = sum_f w[out_t, a, f] x[in_t, f]`` and the
share ``u[t, f] = sum_a w[out_t, a, f] s[t, a]``, then one program per input
row sums the shares of its items, ``gx[r] = sum_{t: in_t = r} u[t]``
(compressed rows, :mod:`xnn.common.models.csr`). No atomic adds: the sums come
out the same in every run.
"""
# no postponed annotations: Triton's interpreter (TRITON_INTERPRET=1) recognizes the
# tl.constexpr parameters of a kernel only from evaluated annotations

from typing import Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor

from xnn.common.models.csr import compressed
from xnn.common.models.dispersion_fast._triton import next_power_of_two


@triton.jit
def _sums_kernel(x_ptr, s_ptr, in_ptr, order_ptr, cstart_ptr, cend_ptr, part_ptr,
                 A: tl.constexpr, AP: tl.constexpr, F: tl.constexpr, FP: tl.constexpr, CH: tl.constexpr,
                 SUB: tl.constexpr, USE_DOT: tl.constexpr, IEEE: tl.constexpr):
    """The partial sum ``sum_t s_t (x) x[in_t]`` over the items of chunk ``c`` (at most ``CH``
    items of one output), ``SUB`` items per step."""
    c = tl.program_id(0).to(tl.int64)
    start = tl.load(cstart_ptr + c)
    end = tl.load(cend_ptr + c)
    a = tl.arange(0, AP)
    f = tl.arange(0, FP)
    acc = tl.zeros([AP, FP], dtype=part_ptr.dtype.element_ty)
    for i in range(0, CH, SUB):
        k = start + i + tl.arange(0, SUB)
        mk = k < end
        t = tl.load(order_ptr + k, mask=mk, other=0)
        r = tl.load(in_ptr + t, mask=mk, other=0)
        sv = tl.load(s_ptr + t[:, None] * A + a[None, :], mask=mk[:, None] & (a[None, :] < A), other=0.0)
        xv = tl.load(x_ptr + r[:, None] * F + f[None, :], mask=mk[:, None] & (f[None, :] < F), other=0.0)
        if USE_DOT:
            if IEEE:
                acc += tl.dot(tl.trans(sv), xv, input_precision="ieee")
            else:
                acc += tl.dot(tl.trans(sv), xv)
        else:
            acc += tl.sum(sv[:, :, None] * xv[:, None, :], axis=0)
    tl.store(part_ptr + c * (A * F) + a[:, None] * F + f[None, :], acc, mask=(a[:, None] < A) & (f[None, :] < F))


@triton.jit
def _reduce_kernel(part_ptr, cptr_ptr, out_ptr, AF: tl.constexpr, AFP: tl.constexpr):
    """``out[e] = sum`` of the partial sums of the chunks of output ``e``, in order."""
    e = tl.program_id(0).to(tl.int64)
    j = tl.arange(0, AFP)
    start = tl.load(cptr_ptr + e)
    end = tl.load(cptr_ptr + e + 1)
    acc = tl.zeros([AFP], dtype=out_ptr.dtype.element_ty)
    n = start
    while n < end:
        acc += tl.load(part_ptr + n * AF + j, mask=j < AF, other=0.0)
        n += 1
    tl.store(out_ptr + e * AF + j, acc, mask=j < AF)


@triton.jit
def _grads_kernel(w_ptr, s_ptr, x_ptr, in_ptr, order_ptr, cout_ptr, cstart_ptr, cend_ptr, u_ptr, gs_ptr,
                  A: tl.constexpr, AP: tl.constexpr, F: tl.constexpr, FP: tl.constexpr, CH: tl.constexpr,
                  SUB: tl.constexpr, USE_DOT: tl.constexpr, IEEE: tl.constexpr):
    """Over the items of chunk ``c`` of output ``e`` (its gradient row loaded once): per item
    ``u_t = sum_a w[e] s_t`` (the item's share of the input-row gradient) and
    ``gs_t = sum_f w[e] x[in_t]``."""
    c = tl.program_id(0).to(tl.int64)
    e = tl.load(cout_ptr + c)
    start = tl.load(cstart_ptr + c)
    end = tl.load(cend_ptr + c)
    a = tl.arange(0, AP)
    f = tl.arange(0, FP)
    we = tl.load(w_ptr + e * (A * F) + a[:, None] * F + f[None, :], mask=(a[:, None] < A) & (f[None, :] < F),
                 other=0.0)
    for i in range(0, CH, SUB):
        k = start + i + tl.arange(0, SUB)
        mk = k < end
        t = tl.load(order_ptr + k, mask=mk, other=0)
        r = tl.load(in_ptr + t, mask=mk, other=0)
        sv = tl.load(s_ptr + t[:, None] * A + a[None, :], mask=mk[:, None] & (a[None, :] < A), other=0.0)
        xv = tl.load(x_ptr + r[:, None] * F + f[None, :], mask=mk[:, None] & (f[None, :] < F), other=0.0)
        if USE_DOT:
            if IEEE:
                u = tl.dot(sv, we, input_precision="ieee")
                gs = tl.dot(xv, tl.trans(we), input_precision="ieee")
            else:
                u = tl.dot(sv, we)
                gs = tl.dot(xv, tl.trans(we))
        else:
            u = tl.sum(sv[:, :, None] * we[None, :, :], axis=1)
            gs = tl.sum(xv[:, None, :] * we[None, :, :], axis=2)
        tl.store(u_ptr + t[:, None] * F + f[None, :], u, mask=mk[:, None] & (f[None, :] < F))
        tl.store(gs_ptr + t[:, None] * A + a[None, :], gs, mask=mk[:, None] & (a[None, :] < A))


@triton.jit
def _rows_kernel(u_ptr, order_ptr, ptr_ptr, gx_ptr, F: tl.constexpr, FP: tl.constexpr, B: tl.constexpr):
    """``gx[r] = sum_t u_t`` over the items of input row ``r``."""
    r = tl.program_id(0).to(tl.int64)
    b = tl.arange(0, B)[:, None]
    f = tl.arange(0, FP)[None, :]
    start = tl.load(ptr_ptr + r)
    end = tl.load(ptr_ptr + r + 1)
    acc = tl.zeros([B, FP], dtype=gx_ptr.dtype.element_ty)
    n = start
    while n < end:
        mb = (n + b) < end
        t = tl.load(order_ptr + n + b, mask=mb, other=0)
        acc += tl.load(u_ptr + t * F + f, mask=mb & (f < F), other=0.0)
        n += B
    f1 = tl.arange(0, FP)
    tl.store(gx_ptr + r * F + f1, tl.sum(acc, axis=0), mask=f1 < F)


#: Items per chunk: one program each, the partial sums of an output's chunks added in order.
CHUNK = 64

#: Triton tile products (``tl.dot``) in float64 too: Triton has no float64 tile product (it
#: rejects the operands), so float64 accumulates by multiply-adds.
DOT_FLOAT64 = False


def _launch(s: Tensor, x: Tensor):
    """``(A, AP, F, FP, SUB, USE_DOT, IEEE)``: the widths, their blocks (powers of two, at least
    16 for a tile product), the items per step and the product kind (float32 tiles are IEEE)."""
    width_a, width_f = s.shape[1], x.shape[1]
    use_dot = x.dtype == torch.float32 or DOT_FLOAT64
    least = 16 if use_dot else 2
    padded_a, padded_f = next_power_of_two(width_a, least), next_power_of_two(width_f, least)
    sub = CHUNK if use_dot else max(1, 2048 // (padded_a * padded_f))
    return width_a, padded_a, width_f, padded_f, sub, use_dot, True


def _sums(x: Tensor, s: Tensor, in_idx: Tensor, rows: tuple, num_out: int) -> Tensor:
    width_a, padded_a, width_f, padded_f, sub, use_dot, ieee = _launch(s, x)
    order_out, chunk_out, chunk_start, chunk_end, chunk_ptr = rows[0], rows[4], rows[5], rows[6], rows[7]
    out = torch.empty((num_out, width_a, width_f), dtype=x.dtype, device=x.device)
    n_chunks = chunk_out.shape[0]
    if num_out and n_chunks:
        part = torch.empty((n_chunks, width_a, width_f), dtype=x.dtype, device=x.device)
        _sums_kernel[(n_chunks,)](x, s, in_idx, order_out, chunk_start, chunk_end, part, A=width_a, AP=padded_a,
                                  F=width_f, FP=padded_f, CH=CHUNK, SUB=sub, USE_DOT=use_dot, IEEE=ieee,
                                  num_warps=4)
        _reduce_kernel[(num_out,)](part, chunk_ptr, out, AF=width_a * width_f,
                                   AFP=next_power_of_two(width_a * width_f, 16), num_warps=4)
    else:
        out.zero_()
    return out


def _grads(w: Tensor, s: Tensor, x: Tensor, in_idx: Tensor, rows: tuple) -> Tuple[Tensor, Tensor]:
    width_a, padded_a, width_f, padded_f, sub, use_dot, ieee = _launch(s, x)
    order_out, chunk_out, chunk_start, chunk_end = rows[0], rows[4], rows[5], rows[6]
    gx = torch.empty_like(x)
    gs = torch.empty_like(s)
    u = torch.empty((s.shape[0], width_f), dtype=x.dtype, device=x.device)
    if chunk_out.shape[0]:
        _grads_kernel[(chunk_out.shape[0],)](w, s, x, in_idx, order_out, chunk_out, chunk_start, chunk_end, u, gs,
                                             A=width_a, AP=padded_a, F=width_f, FP=padded_f, CH=CHUNK, SUB=sub,
                                             USE_DOT=use_dot, IEEE=ieee, num_warps=4)
    if x.shape[0]:
        _rows_kernel[(x.shape[0],)](u, rows[2], rows[3], gx, F=width_f, FP=padded_f,
                                    B=max(1, 1024 // padded_f), num_warps=2)
    return gx, gs


class NeighborSumsFunction(torch.autograd.Function):
    """``S[e] = sum_{t: out_t = e} s_t (x) x[in_t]``, differentiable at any order."""

    @staticmethod
    def forward(ctx, x: Tensor, s: Tensor, in_idx: Tensor, out_idx: Tensor, rows: tuple,
                num_out: int) -> Tensor:
        ctx.rows, ctx.num_out = rows, num_out
        ctx.save_for_backward(x, s, in_idx, out_idx)
        return _sums(x, s, in_idx, rows, num_out)

    @staticmethod
    def backward(ctx, w: Tensor):
        x, s, in_idx, out_idx = ctx.saved_tensors
        gx, gs = NeighborGradsFunction.apply(w.contiguous(), x, s, in_idx, out_idx, ctx.rows)
        return gx, gs, None, None, None, None


class NeighborGradsFunction(torch.autograd.Function):
    """``(sum_a w[out] s, sum_f w[out] x[in])`` onto the input rows and the items: the
    gradients of :class:`NeighborSumsFunction` for the incoming ``w``. Bilinear like the
    sums, so its own derivatives are the two functions again."""

    @staticmethod
    def forward(ctx, w: Tensor, x: Tensor, s: Tensor, in_idx: Tensor, out_idx: Tensor,
                rows: tuple) -> Tuple[Tensor, Tensor]:
        ctx.rows = rows
        ctx.save_for_backward(w, x, s, in_idx, out_idx)
        return _grads(w, s, x, in_idx, rows)

    @staticmethod
    def backward(ctx, tx: Tensor, ts: Tensor):
        w, x, s, in_idx, out_idx = ctx.saved_tensors
        rows = ctx.rows
        tx = None if tx is None else tx.contiguous()
        ts = None if ts is None else ts.contiguous()
        dw = None
        if tx is not None:
            dw = NeighborSumsFunction.apply(tx, s, in_idx, out_idx, rows, w.shape[0])
        if ts is not None:
            term = NeighborSumsFunction.apply(x, ts, in_idx, out_idx, rows, w.shape[0])
            dw = term if dw is None else dw + term
        # d/dx = sum_a w ts and d/ds = sum_f w tx: the same two contractions
        dx = ds = None
        if tx is not None or ts is not None:
            dx, ds = NeighborGradsFunction.apply(w, torch.zeros_like(x) if tx is None else tx,
                                                 torch.zeros_like(s) if ts is None else ts, in_idx, out_idx, rows)
        return dw, dx, ds, None, None, None


#: the compressed rows of the last item list: the blocks of one evaluation share them
_last = [None]


def item_rows(in_idx: Tensor, out_idx: Tensor, num_in: int, num_out: int) -> tuple:
    """The items by output and by input and the output chunks: ``(order_out, ptr_out, order_in,
    ptr_in, chunk_out, chunk_start, chunk_end, chunk_ptr)``, cached for the same index tensors
    (the interaction blocks of one evaluation pass the same ones)."""
    hit = _last[0]
    if hit is not None and hit[0] is in_idx and hit[1] is out_idx and hit[2] == (num_in, num_out):
        return hit[3]
    order_out, ptr_out = compressed(out_idx, num_out)
    # the chunks of every output: at most CHUNK items each, in output order
    counts = ptr_out[1:] - ptr_out[:-1]
    per = (counts + CHUNK - 1) // CHUNK
    chunk_ptr = torch.zeros(num_out + 1, dtype=torch.int64, device=out_idx.device)
    chunk_ptr[1:] = torch.cumsum(per, 0)
    total = int(chunk_ptr[-1])                       # one read-back per item list (cached)
    chunk_out = torch.repeat_interleave(torch.arange(num_out, device=out_idx.device), per, output_size=total)
    local = torch.arange(total, device=out_idx.device) - chunk_ptr[chunk_out]
    chunk_start = ptr_out[chunk_out] + local * CHUNK
    chunk_end = torch.minimum(chunk_start + CHUNK, ptr_out[chunk_out + 1])
    rows = (order_out, ptr_out, *compressed(in_idx, num_in), chunk_out, chunk_start, chunk_end, chunk_ptr)
    _last[0] = (in_idx, out_idx, (num_in, num_out), rows)
    return rows


def neighbor_sums(x: Tensor, s: Tensor, in_idx: Tensor, out_idx: Tensor, num_out: int) -> Tensor:
    """The neighbor sums of the efficient bilinear layer (see :mod:`xnn.gnn.fast.gemnet`).

    Parameters
    ----------
    x : Tensor
        Features of the input rows (edges or intermediate triplets), shape ``(R, F)``.
    s : Tensor
        Angular basis of every item (triplet or quadruplet), shape ``(T, A)``.
    in_idx, out_idx : Tensor
        Input row and output edge of every item, shape ``(T,)``.
    num_out : int
        Number of output edges.

    Returns
    -------
    Tensor
        ``sum_{t: out_t = e} s[t, a] x[in_t, f]``, shape ``(num_out, A, F)``.
    """
    rows = item_rows(in_idx, out_idx, x.shape[0], num_out)
    return NeighborSumsFunction.apply(x.contiguous(), s.contiguous(), in_idx.contiguous(),
                                      out_idx.contiguous(), rows, num_out)
