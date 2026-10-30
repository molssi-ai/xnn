"""Triton kernels and autograd functions of the TensorNet fast path (see :mod:`xnn.gnn.fast.tensornet`).

Imported only when the fast path runs, so that ``triton`` is needed only there.

Per atom the kernels read and write ten ``F``-wide rows: the scalar, the three
components of the vector (the axial vector of the skew part) and the six
distinct entries ``xx, xy, xz, yy, yz, zz`` of a symmetric part. One program
handles one atom and walks its pairs in a sorted order (compressed rows: by
center for the sums onto centers, by neighbor for the gradients onto
neighbors), accumulating in registers, so the sums need no atomic adds (one
remains, for the neighbor term of the pair embedding) and are the same in
every run.
"""
# no postponed annotations: Triton's interpreter (TRITON_INTERPRET=1) recognizes the
# tl.constexpr parameters of a kernel only from evaluated annotations

from typing import Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import once_differentiable

from xnn.common.models.csr import orderings

#: rows per atom: scalar, vector (3), symmetric (6)
ROWS = 10

@triton.jit
def _message_forward(x_ptr, a_ptr, a_stride, cut_ptr, src_ptr, order_ptr, ptr_ptr, out_ptr,
                     F: tl.constexpr):
    """``out[i, r] = sum_j cut_ij a_ij,p(r) x[j, r]`` over the pairs of center ``i``."""
    i = tl.program_id(0).to(tl.int64)
    r = tl.arange(0, 16)[:, None]
    f = tl.arange(0, F)[None, :]
    m = (r < 10) & (f < F)
    start = tl.load(ptr_ptr + i)
    end = tl.load(ptr_ptr + i + 1)
    acc = tl.zeros([16, F], dtype=out_ptr.dtype.element_ty)
    k = start
    while k < end:
        e = tl.load(order_ptr + k)
        s = tl.load(src_ptr + e)
        c = tl.load(cut_ptr + e)
        a_row = a_ptr + e * a_stride + f
        w = tl.where(r == 0, tl.load(a_row),
                     tl.where(r < 4, tl.load(a_row + F), tl.load(a_row + 2 * F)))
        x = tl.load(x_ptr + s * (10 * F) + r * F + f, mask=m, other=0.0)
        acc += (w * c) * x
        k += 1
    tl.store(out_ptr + i * (10 * F) + r * F + f, acc, mask=m)


@triton.jit
def _message_backward(x_ptr, a_ptr, a_stride, cut_ptr, dst_ptr, order_ptr, ptr_ptr, g_ptr,
                      gx_ptr, ga_ptr, gcut_ptr, F: tl.constexpr):
    """Over the pairs of neighbor ``j``: the gradient of ``x[j]``, and per pair those of ``a`` and the cutoff.

    The ten rows are separate vectors, so the sums over the rows of a part are register
    arithmetic; one sum over the features per pair (the cutoff's gradient) remains.
    """
    j = tl.program_id(0).to(tl.int64)
    f = tl.arange(0, F)
    start = tl.load(ptr_ptr + j)
    end = tl.load(ptr_ptr + j + 1)
    xr = x_ptr + j * (10 * F) + f
    x0 = tl.load(xr + 0 * F)
    x1 = tl.load(xr + 1 * F)
    x2 = tl.load(xr + 2 * F)
    x3 = tl.load(xr + 3 * F)
    x4 = tl.load(xr + 4 * F)
    x5 = tl.load(xr + 5 * F)
    x6 = tl.load(xr + 6 * F)
    x7 = tl.load(xr + 7 * F)
    x8 = tl.load(xr + 8 * F)
    x9 = tl.load(xr + 9 * F)
    zero = tl.zeros([F], dtype=gx_ptr.dtype.element_ty)
    q0 = zero
    q1 = zero
    q2 = zero
    q3 = zero
    q4 = zero
    q5 = zero
    q6 = zero
    q7 = zero
    q8 = zero
    q9 = zero
    k = start
    while k < end:
        e = tl.load(order_ptr + k)
        d = tl.load(dst_ptr + e)
        c = tl.load(cut_ptr + e)
        ar = a_ptr + e * a_stride + f
        a0 = tl.load(ar)
        a1 = tl.load(ar + F)
        a2 = tl.load(ar + 2 * F)
        w0 = a0 * c
        w1 = a1 * c
        w2 = a2 * c
        gr = g_ptr + d * (10 * F) + f
        g0 = tl.load(gr + 0 * F)
        g1 = tl.load(gr + 1 * F)
        g2 = tl.load(gr + 2 * F)
        g3 = tl.load(gr + 3 * F)
        g4 = tl.load(gr + 4 * F)
        g5 = tl.load(gr + 5 * F)
        g6 = tl.load(gr + 6 * F)
        g7 = tl.load(gr + 7 * F)
        g8 = tl.load(gr + 8 * F)
        g9 = tl.load(gr + 9 * F)
        q0 += w0 * g0
        q1 += w1 * g1
        q2 += w1 * g2
        q3 += w1 * g3
        q4 += w2 * g4
        q5 += w2 * g5
        q6 += w2 * g6
        q7 += w2 * g7
        q8 += w2 * g8
        q9 += w2 * g9
        h0 = g0 * x0                                   # d L / d (cut a_p), per part
        h1 = g1 * x1 + g2 * x2 + g3 * x3
        h2 = g4 * x4 + g5 * x5 + g6 * x6 + g7 * x7 + g8 * x8 + g9 * x9
        gar = ga_ptr + e * (3 * F) + f
        tl.store(gar, c * h0)
        tl.store(gar + F, c * h1)
        tl.store(gar + 2 * F, c * h2)
        tl.store(gcut_ptr + e, tl.sum(a0 * h0 + a1 * h1 + a2 * h2, axis=0))
        k += 1
    gxr = gx_ptr + j * (10 * F) + f
    tl.store(gxr + 0 * F, q0)
    tl.store(gxr + 1 * F, q1)
    tl.store(gxr + 2 * F, q2)
    tl.store(gxr + 3 * F, q3)
    tl.store(gxr + 4 * F, q4)
    tl.store(gxr + 5 * F, q5)
    tl.store(gxr + 6 * F, q6)
    tl.store(gxr + 7 * F, q7)
    tl.store(gxr + 8 * F, q8)
    tl.store(gxr + 9 * F, q9)


@triton.jit
def _direction_rows(r, ux, uy, uz):
    """The factor of every row: ``1``, ``u_k`` and ``u_k u_l`` (``k <= l``), zero past row 9."""
    return tl.where(r == 0, 1.0, tl.where(r == 1, ux, tl.where(r == 2, uy, tl.where(r == 3, uz,
           tl.where(r == 4, ux * ux, tl.where(r == 5, ux * uy, tl.where(r == 6, ux * uz,
           tl.where(r == 7, uy * uy, tl.where(r == 8, uy * uz,
           tl.where(r == 9, uz * uz, 0.0))))))))))


@triton.jit
def _embedding_forward(left_ptr, right_ptr, cut_ptr, d1_ptr, d2_ptr, d3_ptr, u_ptr, src_ptr,
                       order_ptr, ptr_ptr, out_ptr, F: tl.constexpr):
    """The ten rows of the embedding sums of center ``i``, ``w = cut (left[i] + right[j])``.

    Row 0 gains ``w d_1``, rows 1-3 ``w d_2 u_k``, rows 4-9 ``w d_3 u_k u_l`` (``k <= l``).
    """
    i = tl.program_id(0).to(tl.int64)
    r = tl.arange(0, 16)[:, None]
    f = tl.arange(0, F)[None, :]
    m = (r < 10) & (f < F)
    start = tl.load(ptr_ptr + i)
    end = tl.load(ptr_ptr + i + 1)
    left = tl.load(left_ptr + i * F + f)
    acc = tl.zeros([16, F], dtype=out_ptr.dtype.element_ty)
    k = start
    while k < end:
        e = tl.load(order_ptr + k)
        s = tl.load(src_ptr + e)
        w = tl.load(cut_ptr + e) * (left + tl.load(right_ptr + s * F + f))
        t = tl.where(r == 0, w * tl.load(d1_ptr + e * F + f),
                     tl.where(r < 4, w * tl.load(d2_ptr + e * F + f), w * tl.load(d3_ptr + e * F + f)))
        coef = _direction_rows(r, tl.load(u_ptr + e * 3), tl.load(u_ptr + e * 3 + 1),
                               tl.load(u_ptr + e * 3 + 2))
        acc += t * coef
        k += 1
    tl.store(out_ptr + i * (10 * F) + r * F + f, acc, mask=m)


@triton.jit
def _embedding_backward(left_ptr, right_ptr, cut_ptr, d1_ptr, d2_ptr, d3_ptr, u_ptr, src_ptr,
                        order_ptr, ptr_ptr, g_ptr, gleft_ptr, gright_ptr, gd_ptr, dots_ptr,
                        F: tl.constexpr):
    """Over the pairs of center ``i``: the gradients of ``left[i]``, of ``right`` (onto the
    neighbors, atomic adds) and of ``d``, and per pair the sixteen sums over the features
    ``dots`` from which :func:`_cut_and_direction_gradients` forms those of the cutoff and of ``u``.
    """
    i = tl.program_id(0).to(tl.int64)
    r = tl.arange(0, 16)[:, None]
    f = tl.arange(0, F)
    m = (r < 10) & (f[None, :] < F)
    start = tl.load(ptr_ptr + i)
    end = tl.load(ptr_ptr + i + 1)
    left = tl.load(left_ptr + i * F + f)
    gr = g_ptr + i * (10 * F) + f
    g0 = tl.load(gr + 0 * F)
    g1 = tl.load(gr + 1 * F)
    g2 = tl.load(gr + 2 * F)
    g3 = tl.load(gr + 3 * F)
    g4 = tl.load(gr + 4 * F)
    g5 = tl.load(gr + 5 * F)
    g6 = tl.load(gr + 6 * F)
    g7 = tl.load(gr + 7 * F)
    g8 = tl.load(gr + 8 * F)
    g9 = tl.load(gr + 9 * F)
    G = tl.load(g_ptr + i * (10 * F) + r * F + f[None, :], mask=m, other=0.0)
    acc = tl.zeros([F], dtype=gleft_ptr.dtype.element_ty)
    k = start
    while k < end:
        e = tl.load(order_ptr + k)
        s = tl.load(src_ptr + e)
        c = tl.load(cut_ptr + e)
        pair = left + tl.load(right_ptr + s * F + f)
        w = c * pair
        d1 = tl.load(d1_ptr + e * F + f)
        d2 = tl.load(d2_ptr + e * F + f)
        d3 = tl.load(d3_ptr + e * F + f)
        ux = tl.load(u_ptr + e * 3)
        uy = tl.load(u_ptr + e * 3 + 1)
        uz = tl.load(u_ptr + e * 3 + 2)
        # d L / d t for the three products t_p = w d_p
        y1 = g0
        y2 = ux * g1 + uy * g2 + uz * g3
        y3 = ((ux * ux) * g4 + (ux * uy) * g5 + (ux * uz) * g6 + (uy * uy) * g7
              + (uy * uz) * g8 + (uz * uz) * g9)
        gdr = gd_ptr + e * (3 * F) + f
        tl.store(gdr, w * y1)
        tl.store(gdr + F, w * y2)
        tl.store(gdr + 2 * F, w * y3)
        gw = y1 * d1 + y2 * d2 + y3 * d3
        gp = c * gw
        acc += gp
        tl.atomic_add(gright_ptr + s * F + f, gp)
        # one sum over the features for all rows: row 0 the cutoff's gradient, rows 1-9 the
        # dots of the rows' gradients with their products w d_2 and w d_3
        prod = tl.where(r < 4, (w * d2)[None, :], (w * d3)[None, :])
        M = tl.where(r == 0, (gw * pair)[None, :], G * prod)
        tl.store(dots_ptr + e * 16 + tl.arange(0, 16), tl.sum(M, axis=1))
        k += 1
    tl.store(gleft_ptr + i * F + f, acc)


# Second derivatives. Both pair sums are linear in each of their inputs, so the gradient of a
# first derivative contracted with the cotangents ``t*`` of its outputs is again a sum over the
# pairs of the same form: the kernels below walk the pairs as the first ones do. A missing
# cotangent (zero) is a compile-time flag, its pointer a placeholder that is never read.

@triton.jit
def _message_double_center(x_ptr, tx_ptr, a_ptr, a_stride, ta_ptr, cut_ptr, tcut_ptr, src_ptr,
                           order_ptr, ptr_ptr, out_ptr, F: tl.constexpr, HAS_TX: tl.constexpr,
                           HAS_TA: tl.constexpr, HAS_TCUT: tl.constexpr):
    """The gradient of the messages' backward with respect to its incoming gradient ``g``:
    ``out[i, r] = sum_j cut a_p(r) tx[j, r] + (cut ta_p(r) + tcut a_p(r)) x[j, r]``."""
    i = tl.program_id(0).to(tl.int64)
    r = tl.arange(0, 16)[:, None]
    f = tl.arange(0, F)[None, :]
    m = (r < 10) & (f < F)
    start = tl.load(ptr_ptr + i)
    end = tl.load(ptr_ptr + i + 1)
    acc = tl.zeros([16, F], dtype=out_ptr.dtype.element_ty)
    k = start
    while k < end:
        e = tl.load(order_ptr + k)
        s = tl.load(src_ptr + e)
        c = tl.load(cut_ptr + e)
        a_row = a_ptr + e * a_stride + f
        w = tl.where(r == 0, tl.load(a_row),
                     tl.where(r < 4, tl.load(a_row + F), tl.load(a_row + 2 * F)))
        if HAS_TX:
            acc += (w * c) * tl.load(tx_ptr + s * (10 * F) + r * F + f, mask=m, other=0.0)
        if HAS_TA or HAS_TCUT:
            v = tl.zeros([16, F], dtype=out_ptr.dtype.element_ty)
            if HAS_TA:
                ta_row = ta_ptr + e * (3 * F) + f
                v += c * tl.where(r == 0, tl.load(ta_row),
                                  tl.where(r < 4, tl.load(ta_row + F), tl.load(ta_row + 2 * F)))
            if HAS_TCUT:
                v += tl.load(tcut_ptr + e) * w
            acc += v * tl.load(x_ptr + s * (10 * F) + r * F + f, mask=m, other=0.0)
        k += 1
    tl.store(out_ptr + i * (10 * F) + r * F + f, acc, mask=m)


@triton.jit
def _message_double_neighbor(x_ptr, tx_ptr, a_ptr, a_stride, ta_ptr, cut_ptr, tcut_ptr, dst_ptr,
                             order_ptr, ptr_ptr, g_ptr, dx_ptr, da_ptr, dcut_ptr, F: tl.constexpr,
                             HAS_TX: tl.constexpr, HAS_TA: tl.constexpr, HAS_TCUT: tl.constexpr):
    """Over the pairs of neighbor ``j``: the gradients of the messages' backward with respect to
    ``x[j]`` and, per pair, to ``a`` and the cutoff."""
    j = tl.program_id(0).to(tl.int64)
    f = tl.arange(0, F)
    start = tl.load(ptr_ptr + j)
    end = tl.load(ptr_ptr + j + 1)
    zero = tl.zeros([F], dtype=dx_ptr.dtype.element_ty)
    xr = x_ptr + j * (10 * F) + f
    x0 = tl.load(xr + 0 * F)
    x1 = tl.load(xr + 1 * F)
    x2 = tl.load(xr + 2 * F)
    x3 = tl.load(xr + 3 * F)
    x4 = tl.load(xr + 4 * F)
    x5 = tl.load(xr + 5 * F)
    x6 = tl.load(xr + 6 * F)
    x7 = tl.load(xr + 7 * F)
    x8 = tl.load(xr + 8 * F)
    x9 = tl.load(xr + 9 * F)
    t0 = zero
    t1 = zero
    t2 = zero
    t3 = zero
    t4 = zero
    t5 = zero
    t6 = zero
    t7 = zero
    t8 = zero
    t9 = zero
    if HAS_TX:
        tr = tx_ptr + j * (10 * F) + f
        t0 = tl.load(tr + 0 * F)
        t1 = tl.load(tr + 1 * F)
        t2 = tl.load(tr + 2 * F)
        t3 = tl.load(tr + 3 * F)
        t4 = tl.load(tr + 4 * F)
        t5 = tl.load(tr + 5 * F)
        t6 = tl.load(tr + 6 * F)
        t7 = tl.load(tr + 7 * F)
        t8 = tl.load(tr + 8 * F)
        t9 = tl.load(tr + 9 * F)
    q0 = zero
    q1 = zero
    q2 = zero
    q3 = zero
    q4 = zero
    q5 = zero
    q6 = zero
    q7 = zero
    q8 = zero
    q9 = zero
    k = start
    while k < end:
        e = tl.load(order_ptr + k)
        d = tl.load(dst_ptr + e)
        c = tl.load(cut_ptr + e)
        ar = a_ptr + e * a_stride + f
        a0 = tl.load(ar)
        a1 = tl.load(ar + F)
        a2 = tl.load(ar + 2 * F)
        ta0 = zero
        ta1 = zero
        ta2 = zero
        if HAS_TA:
            tar = ta_ptr + e * (3 * F) + f
            ta0 = tl.load(tar)
            ta1 = tl.load(tar + F)
            ta2 = tl.load(tar + 2 * F)
        tc = 0.0
        if HAS_TCUT:
            tc = tl.load(tcut_ptr + e)
        # the weights of x in the derivative: cut ta + tcut a, per part
        v0 = c * ta0 + tc * a0
        v1 = c * ta1 + tc * a1
        v2 = c * ta2 + tc * a2
        gr = g_ptr + d * (10 * F) + f
        g0 = tl.load(gr + 0 * F)
        g1 = tl.load(gr + 1 * F)
        g2 = tl.load(gr + 2 * F)
        g3 = tl.load(gr + 3 * F)
        g4 = tl.load(gr + 4 * F)
        g5 = tl.load(gr + 5 * F)
        g6 = tl.load(gr + 6 * F)
        g7 = tl.load(gr + 7 * F)
        g8 = tl.load(gr + 8 * F)
        g9 = tl.load(gr + 9 * F)
        q0 += v0 * g0
        q1 += v1 * g1
        q2 += v1 * g2
        q3 += v1 * g3
        q4 += v2 * g4
        q5 += v2 * g5
        q6 += v2 * g6
        q7 += v2 * g7
        q8 += v2 * g8
        q9 += v2 * g9
        h0 = g0 * x0                                   # g . x and g . tx, per part
        h1 = g1 * x1 + g2 * x2 + g3 * x3
        h2 = g4 * x4 + g5 * x5 + g6 * x6 + g7 * x7 + g8 * x8 + g9 * x9
        p0 = g0 * t0
        p1 = g1 * t1 + g2 * t2 + g3 * t3
        p2 = g4 * t4 + g5 * t5 + g6 * t6 + g7 * t7 + g8 * t8 + g9 * t9
        dar = da_ptr + e * (3 * F) + f
        tl.store(dar, c * p0 + tc * h0)
        tl.store(dar + F, c * p1 + tc * h1)
        tl.store(dar + 2 * F, c * p2 + tc * h2)
        tl.store(dcut_ptr + e, tl.sum(a0 * p0 + a1 * p1 + a2 * p2 + ta0 * h0 + ta1 * h1 + ta2 * h2,
                                      axis=0))
        k += 1
    dxr = dx_ptr + j * (10 * F) + f
    tl.store(dxr + 0 * F, q0)
    tl.store(dxr + 1 * F, q1)
    tl.store(dxr + 2 * F, q2)
    tl.store(dxr + 3 * F, q3)
    tl.store(dxr + 4 * F, q4)
    tl.store(dxr + 5 * F, q5)
    tl.store(dxr + 6 * F, q6)
    tl.store(dxr + 7 * F, q7)
    tl.store(dxr + 8 * F, q8)
    tl.store(dxr + 9 * F, q9)


@triton.jit
def _embedding_double(left_ptr, right_ptr, tleft_ptr, tright_ptr, cut_ptr, d1_ptr, d2_ptr, d3_ptr,
                      td_ptr, tdots_ptr, u_ptr, src_ptr, order_ptr, ptr_ptr, g_ptr, dg_ptr,
                      dleft_ptr, dright_ptr, dd_ptr, ddots_ptr, F: tl.constexpr,
                      HAS_TP: tl.constexpr, HAS_TD: tl.constexpr, HAS_TT: tl.constexpr):
    """Over the pairs of center ``i``: the gradients of the embedding's backward (its outputs
    ``gleft``, ``gright``, ``gd`` and ``dots`` with cotangents ``tleft``, ``tright``, ``td``,
    ``tdots``) with respect to ``g[i]``, ``left[i]``, ``right`` (onto the neighbors, atomic adds)
    and ``d``, and per pair the sixteen sums ``ddots`` of :func:`_cut_and_direction_gradients`.

    With ``pair = left[i] + right[j]``, ``w = cut pair``, ``y_q`` the products of ``g[i]`` with
    the direction factors (as in the backward), ``kappa = cut (tleft[i] + tright[j]) + tdots_0
    pair``, ``z_2 = sum_m tdots_m g_m`` and ``z_3 = sum_kl tdots_kl g_kl``, the contracted
    backward is ``sum_f y_q (kappa d_q + w td_q) + w (d_2 z_2 + d_3 z_3)`` per pair.
    """
    i = tl.program_id(0).to(tl.int64)
    r = tl.arange(0, 16)[:, None]
    f = tl.arange(0, F)
    m = (r < 10) & (f[None, :] < F)
    start = tl.load(ptr_ptr + i)
    end = tl.load(ptr_ptr + i + 1)
    left = tl.load(left_ptr + i * F + f)
    zero = tl.zeros([F], dtype=dleft_ptr.dtype.element_ty)
    tleft = zero
    if HAS_TP:
        tleft = tl.load(tleft_ptr + i * F + f)
    gr = g_ptr + i * (10 * F) + f
    g0 = tl.load(gr + 0 * F)
    g1 = tl.load(gr + 1 * F)
    g2 = tl.load(gr + 2 * F)
    g3 = tl.load(gr + 3 * F)
    g4 = tl.load(gr + 4 * F)
    g5 = tl.load(gr + 5 * F)
    g6 = tl.load(gr + 6 * F)
    g7 = tl.load(gr + 7 * F)
    g8 = tl.load(gr + 8 * F)
    g9 = tl.load(gr + 9 * F)
    G = tl.load(g_ptr + i * (10 * F) + r * F + f[None, :], mask=m, other=0.0)
    acc = tl.zeros([16, F], dtype=dg_ptr.dtype.element_ty)
    acc_left = zero
    k = start
    while k < end:
        e = tl.load(order_ptr + k)
        s = tl.load(src_ptr + e)
        c = tl.load(cut_ptr + e)
        pair = left + tl.load(right_ptr + s * F + f)
        w = c * pair
        tp = zero
        if HAS_TP:
            tp = tleft + tl.load(tright_ptr + s * F + f)
        tau0 = 0.0
        z2 = zero
        z3 = zero
        if HAS_TT:
            tt = tdots_ptr + e * 16
            tau0 = tl.load(tt)
            z2 = tl.load(tt + 1) * g1 + tl.load(tt + 2) * g2 + tl.load(tt + 3) * g3
            z3 = (tl.load(tt + 4) * g4 + tl.load(tt + 5) * g5 + tl.load(tt + 6) * g6
                  + tl.load(tt + 7) * g7 + tl.load(tt + 8) * g8 + tl.load(tt + 9) * g9)
        kappa = c * tp + tau0 * pair
        d1 = tl.load(d1_ptr + e * F + f)
        d2 = tl.load(d2_ptr + e * F + f)
        d3 = tl.load(d3_ptr + e * F + f)
        td1 = zero
        td2 = zero
        td3 = zero
        if HAS_TD:
            tdr = td_ptr + e * (3 * F) + f
            td1 = tl.load(tdr)
            td2 = tl.load(tdr + F)
            td3 = tl.load(tdr + 2 * F)
        ux = tl.load(u_ptr + e * 3)
        uy = tl.load(u_ptr + e * 3 + 1)
        uz = tl.load(u_ptr + e * 3 + 2)
        y1 = g0
        y2 = ux * g1 + uy * g2 + uz * g3
        y3 = ((ux * ux) * g4 + (ux * uy) * g5 + (ux * uz) * g6 + (uy * uy) * g7
              + (uy * uz) * g8 + (uz * uz) * g9)
        rho1 = kappa * d1 + w * td1
        rho2 = kappa * d2 + w * td2
        rho3 = kappa * d3 + w * td3
        ddr = dd_ptr + e * (3 * F) + f
        tl.store(ddr, kappa * y1)
        tl.store(ddr + F, kappa * y2 + w * z2)
        tl.store(ddr + 2 * F, kappa * y3 + w * z3)
        # d / d w and d / d kappa of the contracted backward
        dw = y1 * td1 + y2 * td2 + y3 * td3 + d2 * z2 + d3 * z3
        dk = y1 * d1 + y2 * d2 + y3 * d3
        dpair = c * dw + tau0 * dk
        acc_left += dpair
        tl.atomic_add(dright_ptr + s * F + f, dpair)
        # row 0 the cutoff's gradient, rows 1-9 the dots of g with rho_2 and rho_3 (the
        # direction factors' gradients, formed as in the backward)
        M = tl.where(r == 0, (pair * dw + tp * dk)[None, :],
                     G * tl.where(r < 4, rho2[None, :], rho3[None, :]))
        tl.store(ddots_ptr + e * 16 + tl.arange(0, 16), tl.sum(M, axis=1))
        # d / d g[i]: the rows' direction factors times rho, and tdots_r w d (rows 1-9)
        coef = _direction_rows(r, ux, uy, uz)
        t = tl.where(r == 0, rho1[None, :], tl.where(r < 4, rho2[None, :], rho3[None, :])) * coef
        if HAS_TT:
            tau = tl.load(tdots_ptr + e * 16 + r, mask=(r >= 1) & (r < 10), other=0.0)
            t += tau * tl.where(r < 4, (w * d2)[None, :], (w * d3)[None, :])
        acc += t
        k += 1
    tl.store(dg_ptr + i * (10 * F) + r * F + f[None, :], acc, mask=m)
    tl.store(dleft_ptr + i * F + f, acc_left)


def _cut_and_direction_gradients(dots: Tensor, u: Tensor) -> Tuple[Tensor, Tensor]:
    """The gradients of the cutoff and of the directions from the embedding backward's sums.

    ``dots[:, 0]`` is the cutoff's; with ``a_m = dots[:, 1 + m]`` and ``h_kl`` the sums of the
    rows ``u_k u_l`` (``k <= l``, columns 4-9), ``d L / d u_m = a_m + sum_{k <= l} h_kl
    (delta_km u_l + delta_lm u_k)``.
    """
    ux, uy, uz = u.unbind(1)
    a, h = dots[:, 1:4], dots[:, 4:10]
    hxx, hxy, hxz, hyy, hyz, hzz = h.unbind(1)
    gu = torch.stack([a[:, 0] + (hxx + hxx) * ux + hxy * uy + hxz * uz,
                      a[:, 1] + hxy * ux + (hyy + hyy) * uy + hyz * uz,
                      a[:, 2] + hxz * ux + hyz * uy + (hzz + hzz) * uz], dim=1)
    return dots[:, 0], gu


def _warps(n_features: int) -> int:
    """Warps per program: one atom's ten rows (16 with padding) of ``F`` values."""
    return 4 if n_features <= 128 else 8


def _contiguous(t):
    return None if t is None else t.contiguous()


class MessageFunction(torch.autograd.Function):
    """The interaction messages on the ten rows ``x`` with filters ``cut * a``."""

    @staticmethod
    def forward(ctx, x: Tensor, a: Tensor, cut: Tensor, src: Tensor, dst: Tensor) -> Tensor:
        n, F = x.shape[0], x.shape[2]
        out = x.new_empty((n, ROWS, F))
        order_dst, ptr_dst, order_src, ptr_src = orderings(src, dst, n)
        if n:
            _message_forward[(n,)](x, a, a.stride(0), cut, src, order_dst, ptr_dst, out, F=F,
                                   num_warps=_warps(F))
        ctx.save_for_backward(x, a, cut, src, dst, order_dst, ptr_dst, order_src, ptr_src)
        return out

    @staticmethod
    def backward(ctx, g):
        return (*MessageGradFunction.apply(*ctx.saved_tensors[:3], g.contiguous(),
                                           *ctx.saved_tensors[3:]), None, None)


class MessageGradFunction(torch.autograd.Function):
    """The backward of :class:`MessageFunction` (the gradients of ``x``, ``a`` and the cutoff
    for the incoming gradient ``g``), differentiable once more."""

    @staticmethod
    def forward(ctx, x: Tensor, a: Tensor, cut: Tensor, g: Tensor, src: Tensor, dst: Tensor,
                order_dst: Tensor, ptr_dst: Tensor, order_src: Tensor, ptr_src: Tensor):
        n, F, n_pairs = x.shape[0], x.shape[2], a.shape[0]
        gx = torch.empty_like(x)
        ga = a.new_empty((n_pairs, 3 * F))
        gcut = cut.new_empty((n_pairs,))
        if n:
            _message_backward[(n,)](x, a, a.stride(0), cut, dst, order_src, ptr_src, g, gx, ga,
                                    gcut, F=F, num_warps=_warps(F))
        ctx.save_for_backward(x, a, cut, g, src, dst, order_dst, ptr_dst, order_src, ptr_src)
        return gx, ga, gcut

    @staticmethod
    @once_differentiable
    def backward(ctx, tx, ta, tcut):
        x, a, cut, g, src, dst, order_dst, ptr_dst, order_src, ptr_src = ctx.saved_tensors
        tx, ta, tcut = _contiguous(tx), _contiguous(ta), _contiguous(tcut)
        n, F, n_pairs = x.shape[0], x.shape[2], a.shape[0]
        flags = dict(HAS_TX=tx is not None, HAS_TA=ta is not None, HAS_TCUT=tcut is not None)
        dx = da = dcut = dg = None
        if ctx.needs_input_grad[3]:
            dg = torch.empty_like(g)
            if n:
                _message_double_center[(n,)](x, x if tx is None else tx, a, a.stride(0),
                                             a if ta is None else ta, cut,
                                             cut if tcut is None else tcut, src, order_dst,
                                             ptr_dst, dg, F=F, num_warps=_warps(F), **flags)
        if any(ctx.needs_input_grad[:3]):
            dx = torch.empty_like(x)
            da = a.new_empty((n_pairs, 3 * F))
            dcut = cut.new_empty((n_pairs,))
            if n:
                _message_double_neighbor[(n,)](x, x if tx is None else tx, a, a.stride(0),
                                               a if ta is None else ta, cut,
                                               cut if tcut is None else tcut, dst, order_src,
                                               ptr_src, g, dx, da, dcut, F=F,
                                               num_warps=_warps(F), **flags)
        return dx, da, dcut, dg, None, None, None, None, None, None


class EmbeddingFunction(torch.autograd.Function):
    """The pair sums of the embedding on the ten rows."""

    @staticmethod
    def forward(ctx, left: Tensor, right: Tensor, cut: Tensor, d1: Tensor, d2: Tensor,
                d3: Tensor, u: Tensor, src: Tensor, dst: Tensor) -> Tensor:
        n, F = left.shape
        out = left.new_empty((n, ROWS, F))
        order_dst, ptr_dst, _, _ = orderings(src, dst, n)
        if n:
            _embedding_forward[(n,)](left, right, cut, d1, d2, d3, u, src, order_dst, ptr_dst, out,
                                     F=F, num_warps=_warps(F))
        ctx.save_for_backward(left, right, cut, d1, d2, d3, u, src, order_dst, ptr_dst)
        return out

    @staticmethod
    def backward(ctx, g):
        left, right, cut, d1, d2, d3, u, src, order_dst, ptr_dst = ctx.saved_tensors
        F = left.shape[1]
        gleft, gright, gd, dots = EmbeddingGradFunction.apply(
            left, right, cut, d1, d2, d3, u, g.contiguous(), src, order_dst, ptr_dst)
        # outside the kernel, so that autograd differentiates the direction factors' products
        gcut, gu = _cut_and_direction_gradients(dots, u)
        return (gleft, gright, gcut, gd[:, :F], gd[:, F:2 * F], gd[:, 2 * F:], gu, None,
                None)


class EmbeddingGradFunction(torch.autograd.Function):
    """The backward of :class:`EmbeddingFunction` up to the per-pair sums ``dots``,
    differentiable once more."""

    @staticmethod
    def forward(ctx, left: Tensor, right: Tensor, cut: Tensor, d1: Tensor, d2: Tensor,
                d3: Tensor, u: Tensor, g: Tensor, src: Tensor, order_dst: Tensor,
                ptr_dst: Tensor):
        n, F, n_pairs = left.shape[0], left.shape[1], d1.shape[0]
        gleft = torch.empty_like(left)
        gright = torch.zeros_like(right)
        gd = d1.new_empty((n_pairs, 3 * F))
        dots = d1.new_empty((n_pairs, 16))
        if n:
            _embedding_backward[(n,)](left, right, cut, d1, d2, d3, u, src, order_dst, ptr_dst, g,
                                      gleft, gright, gd, dots, F=F, num_warps=_warps(F))
        ctx.save_for_backward(left, right, cut, d1, d2, d3, u, g, src, order_dst, ptr_dst)
        return gleft, gright, gd, dots

    @staticmethod
    @once_differentiable
    def backward(ctx, tleft, tright, td, tdots):
        left, right, cut, d1, d2, d3, u, g, src, order_dst, ptr_dst = ctx.saved_tensors
        n, F, n_pairs = left.shape[0], left.shape[1], d1.shape[0]
        if (tleft is None) != (tright is None):
            tleft = torch.zeros_like(left) if tleft is None else tleft
            tright = torch.zeros_like(right) if tright is None else tright
        tleft, tright, td, tdots = (_contiguous(t) for t in (tleft, tright, td, tdots))
        dg = torch.empty_like(g)
        dleft = torch.empty_like(left)
        dright = torch.zeros_like(right)
        dd = d1.new_empty((n_pairs, 3 * F))
        ddots = d1.new_empty((n_pairs, 16))
        if n:
            _embedding_double[(n,)](left, right, left if tleft is None else tleft,
                                    right if tright is None else tright, cut, d1, d2, d3,
                                    d1 if td is None else td, d1 if tdots is None else tdots, u,
                                    src, order_dst, ptr_dst, g, dg, dleft, dright, dd, ddots, F=F,
                                    num_warps=_warps(F), HAS_TP=tleft is not None,
                                    HAS_TD=td is not None, HAS_TT=tdots is not None)
        dcut, du = _cut_and_direction_gradients(ddots, u)
        return (dleft, dright, dcut, dd[:, :F], dd[:, F:2 * F], dd[:, 2 * F:], du, dg, None,
                None, None)


def messages(x: Tensor, a: Tensor, cut: Tensor, src: Tensor, dst: Tensor) -> Tensor:
    """The fused interaction messages.

    Parameters
    ----------
    x : Tensor
        The ten rows of every atom, ``(N, 10, F)``.
    a : Tensor
        The filter network's output, without the cutoff, ``(P, 3F)``: the
        scalar, vector and symmetric parts.
    cut : Tensor
        The cosine cutoff of every pair ``(P,)``.
    src, dst : Tensor
        Neighbor and center of every pair ``(P,)``.

    Returns
    -------
    Tensor
        ``sum_j cut_ij a_ij x_j`` on the ten rows, ``(N, 10, F)``.
    """
    if a.stride(1) != 1:
        a = a.contiguous()
    return MessageFunction.apply(x.contiguous(), a, cut.contiguous(), src.contiguous(),
                                 dst.contiguous())


def embedding(left: Tensor, right: Tensor, cut: Tensor, d1: Tensor, d2: Tensor, d3: Tensor,
              u: Tensor, src: Tensor, dst: Tensor) -> Tensor:
    """The fused pair sums of the tensor embedding.

    Parameters
    ----------
    left, right : Tensor
        The pair embedding's terms of the center and of the neighbor, ``(N, F)``
        each (the bias in ``left``).
    cut : Tensor
        The cosine cutoff of every pair ``(P,)``.
    d1, d2, d3 : Tensor
        The three radial projections, ``(P, F)`` each.
    u : Tensor
        Unit vectors of the pairs (zero for a self pair), ``(P, 3)``.
    src, dst : Tensor
        Neighbor and center of every pair ``(P,)``.

    Returns
    -------
    Tensor
        The ten rows (scalar, vector, the six distinct entries of the outer-product
        sum, its trace not removed), ``(N, 10, F)``.
    """
    return EmbeddingFunction.apply(left.contiguous(), right.contiguous(), cut.contiguous(),
                                   d1.contiguous(), d2.contiguous(), d3.contiguous(),
                                   u.contiguous(), src.contiguous(), dst.contiguous())
