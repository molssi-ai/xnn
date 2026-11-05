"""Triton kernels and multilinear blocks of the ET fast path (see :mod:`xnn.trfnn.fast.et`).

Imported only when the fast path runs, so that ``triton`` is needed only there.

Features are ``F = H D`` wide (``H`` heads of ``D``); a kernel holds them as an
``(H, D)`` tile, padded to powers of two. One program handles one atom and
walks its pairs in a sorted order (compressed rows: by center for the sums onto
centers, by neighbor for the gradients onto neighbors), accumulating in
registers, so the sums need no atomic adds and are the same in every run.

Two blocks, each linear in every operand:

* the logits ``l[e, h] = sum_(f in h) q[i, f] k[j, f] dk[e, f]`` of every pair
  ``e`` (center ``i``, neighbor ``j``);
* the messages, ``dx[i] = sum_e a[e, h] s3[e]`` and
  ``dv[i, c] = sum_e v[j, c] s1[e] + s2[e] u[e, c]``, with the values
  ``s_k[e] = w_k[j] dw_k[e] cw[e]`` (``w`` the value projections, ``dw`` the
  value filters, ``cw`` the cutoff of the values, ``u`` the pair directions),
  returned as four rows per atom: ``dx`` and the three of ``dv``.

Their adjoints are kernels of the same form, and the derivatives of the
adjoints are values and adjoints with one operand replaced by its cotangent
(:mod:`xnn.common.models.multilinear`), so the fused layers train with a force
loss.
"""
# no postponed annotations: Triton's interpreter (TRITON_INTERPRET=1) recognizes the
# tl.constexpr parameters of a kernel only from evaluated annotations

from typing import List, Optional

import torch
import triton
import triton.language as tl
from torch import Tensor

from xnn.common.models.csr import orderings
from xnn.common.models.multilinear import MultilinearBlock, multilinear


@triton.jit
def _logits_value(q_ptr, k_ptr, dk_ptr, src_ptr, order_ptr, ptr_ptr, out_ptr, H: tl.constexpr,
                  D: tl.constexpr, HP: tl.constexpr, DP: tl.constexpr, HAS_DK: tl.constexpr):
    """``out[e, h] = sum_(f in h) q[i] k[j] dk[e]`` over the pairs of center ``i``."""
    i = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, HP)[:, None]
    col = h * D + tl.arange(0, DP)[None, :]
    fm = (h < H) & (tl.arange(0, DP)[None, :] < D)
    hv = tl.arange(0, HP)
    qi = tl.load(q_ptr + i * (H * D) + col, mask=fm, other=0.0)
    p = tl.load(ptr_ptr + i)
    end = tl.load(ptr_ptr + i + 1)
    while p < end:
        e = tl.load(order_ptr + p).to(tl.int64)
        j = tl.load(src_ptr + e).to(tl.int64)
        t = qi * tl.load(k_ptr + j * (H * D) + col, mask=fm, other=0.0)
        if HAS_DK:
            t = t * tl.load(dk_ptr + e * (H * D) + col, mask=fm, other=0.0)
        tl.store(out_ptr + e * H + hv, tl.sum(t, axis=1), mask=hv < H)
        p += 1


@triton.jit
def _logits_grad_center(g_ptr, k_ptr, dk_ptr, src_ptr, order_ptr, ptr_ptr, gq_ptr, H: tl.constexpr,
                        D: tl.constexpr, HP: tl.constexpr, DP: tl.constexpr, HAS_DK: tl.constexpr):
    """``gq[i] = sum_e g[e, h] k[j] dk[e]`` over the pairs of center ``i``."""
    i = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, HP)[:, None]
    col = h * D + tl.arange(0, DP)[None, :]
    fm = (h < H) & (tl.arange(0, DP)[None, :] < D)
    acc = tl.zeros([HP, DP], dtype=gq_ptr.dtype.element_ty)
    p = tl.load(ptr_ptr + i)
    end = tl.load(ptr_ptr + i + 1)
    while p < end:
        e = tl.load(order_ptr + p).to(tl.int64)
        j = tl.load(src_ptr + e).to(tl.int64)
        t = tl.load(g_ptr + e * H + h, mask=h < H, other=0.0) * tl.load(
            k_ptr + j * (H * D) + col, mask=fm, other=0.0)
        if HAS_DK:
            t = t * tl.load(dk_ptr + e * (H * D) + col, mask=fm, other=0.0)
        acc += t
        p += 1
    tl.store(gq_ptr + i * (H * D) + col, acc, mask=fm)


@triton.jit
def _logits_grad_neighbor(g_ptr, q_ptr, k_ptr, dk_ptr, dst_ptr, order_ptr, ptr_ptr, gk_ptr, gdk_ptr,
                          H: tl.constexpr, D: tl.constexpr, HP: tl.constexpr, DP: tl.constexpr,
                          HAS_DK: tl.constexpr, NEED_K: tl.constexpr, NEED_DK: tl.constexpr):
    """Over the pairs of neighbor ``j``: ``gk[j] = sum_e g[e, h] q[i] dk[e]`` and ``gdk[e] = g[e, h] q[i] k[j]``."""
    j = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, HP)[:, None]
    col = h * D + tl.arange(0, DP)[None, :]
    fm = (h < H) & (tl.arange(0, DP)[None, :] < D)
    acc = tl.zeros([HP, DP], dtype=q_ptr.dtype.element_ty)
    kj = tl.load(k_ptr + j * (H * D) + col, mask=fm, other=0.0)
    p = tl.load(ptr_ptr + j)
    end = tl.load(ptr_ptr + j + 1)
    while p < end:
        e = tl.load(order_ptr + p).to(tl.int64)
        i = tl.load(dst_ptr + e).to(tl.int64)
        t = tl.load(g_ptr + e * H + h, mask=h < H, other=0.0) * tl.load(
            q_ptr + i * (H * D) + col, mask=fm, other=0.0)
        if NEED_DK:
            tl.store(gdk_ptr + e * (H * D) + col, t * kj, mask=fm)
        if NEED_K:
            if HAS_DK:
                t = t * tl.load(dk_ptr + e * (H * D) + col, mask=fm, other=0.0)
            acc += t
        p += 1
    if NEED_K:
        tl.store(gk_ptr + j * (H * D) + col, acc, mask=fm)


@triton.jit
def _messages_value(a_ptr, w_ptr, dw_ptr, cw_ptr, v_ptr, u_ptr, src_ptr, order_ptr, ptr_ptr, out_ptr,
                    H: tl.constexpr, D: tl.constexpr, HP: tl.constexpr, DP: tl.constexpr,
                    HAS_DW: tl.constexpr, HAS_CW: tl.constexpr, T1: tl.constexpr, T2: tl.constexpr,
                    T3: tl.constexpr):
    """The four rows ``dx, dv_x, dv_y, dv_z`` of center ``i`` over its pairs (terms ``T1`` to ``T3``)."""
    i = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, HP)[:, None]
    dd = tl.arange(0, DP)[None, :]
    col = h * D + dd
    col3 = h * (3 * D) + dd
    fm = (h < H) & (dd < D)
    acc0 = tl.zeros([HP, DP], dtype=out_ptr.dtype.element_ty)
    acc1 = tl.zeros([HP, DP], dtype=out_ptr.dtype.element_ty)
    acc2 = tl.zeros([HP, DP], dtype=out_ptr.dtype.element_ty)
    acc3 = tl.zeros([HP, DP], dtype=out_ptr.dtype.element_ty)
    p = tl.load(ptr_ptr + i)
    end = tl.load(ptr_ptr + i + 1)
    while p < end:
        e = tl.load(order_ptr + p).to(tl.int64)
        j = tl.load(src_ptr + e).to(tl.int64)
        wj = w_ptr + j * (3 * H * D) + col3
        dwe = dw_ptr + e * (3 * H * D) + col3
        if HAS_CW:
            c = tl.load(cw_ptr + e)
        else:
            c = 1.0
        if T1:
            s3 = tl.load(wj, mask=fm, other=0.0) * c
            if HAS_DW:
                s3 = s3 * tl.load(dwe, mask=fm, other=0.0)
            acc0 += tl.load(a_ptr + e * H + h, mask=h < H, other=0.0) * s3
        if T2:
            s1 = tl.load(wj + D, mask=fm, other=0.0) * c
            if HAS_DW:
                s1 = s1 * tl.load(dwe + D, mask=fm, other=0.0)
            vj = v_ptr + j * (3 * H * D) + col
            acc1 += tl.load(vj, mask=fm, other=0.0) * s1
            acc2 += tl.load(vj + H * D, mask=fm, other=0.0) * s1
            acc3 += tl.load(vj + 2 * H * D, mask=fm, other=0.0) * s1
        if T3:
            s2 = tl.load(wj + 2 * D, mask=fm, other=0.0) * c
            if HAS_DW:
                s2 = s2 * tl.load(dwe + 2 * D, mask=fm, other=0.0)
            acc1 += s2 * tl.load(u_ptr + e * 3)
            acc2 += s2 * tl.load(u_ptr + e * 3 + 1)
            acc3 += s2 * tl.load(u_ptr + e * 3 + 2)
        p += 1
    row = out_ptr + i * (4 * H * D) + col
    tl.store(row, acc0, mask=fm)
    tl.store(row + H * D, acc1, mask=fm)
    tl.store(row + 2 * H * D, acc2, mask=fm)
    tl.store(row + 3 * H * D, acc3, mask=fm)


@triton.jit
def _messages_grad(g_ptr, a_ptr, w_ptr, dw_ptr, cw_ptr, v_ptr, u_ptr, dst_ptr, order_ptr, ptr_ptr,
                   ga_ptr, gw_ptr, gdw_ptr, gcw_ptr, gv_ptr, gu_ptr, H: tl.constexpr, D: tl.constexpr,
                   HP: tl.constexpr, DP: tl.constexpr, HAS_DW: tl.constexpr, HAS_CW: tl.constexpr,
                   T1: tl.constexpr, T2: tl.constexpr, T3: tl.constexpr, NA: tl.constexpr,
                   NW: tl.constexpr, NDW: tl.constexpr, NCW: tl.constexpr, NV: tl.constexpr,
                   NU: tl.constexpr):
    """Over the pairs of neighbor ``j`` for the incoming rows ``g``: the gradients of ``w[j]`` and
    ``v[j]`` (sums) and per pair those of ``a``, ``dw``, ``cw`` and ``u``."""
    j = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, HP)[:, None]
    dd = tl.arange(0, DP)[None, :]
    col = h * D + dd
    col3 = h * (3 * D) + dd
    fm = (h < H) & (dd < D)
    hv = tl.arange(0, HP)
    dtype = w_ptr.dtype.element_ty
    wj = w_ptr + j * (3 * H * D) + col3
    w3 = tl.load(wj, mask=fm, other=0.0)
    w1 = tl.load(wj + D, mask=fm, other=0.0)
    w2 = tl.load(wj + 2 * D, mask=fm, other=0.0)
    vj = v_ptr + j * (3 * H * D) + col
    v0 = tl.load(vj, mask=fm, other=0.0)
    v1 = tl.load(vj + H * D, mask=fm, other=0.0)
    v2 = tl.load(vj + 2 * H * D, mask=fm, other=0.0)
    acc_w3 = tl.zeros([HP, DP], dtype=dtype)
    acc_w1 = tl.zeros([HP, DP], dtype=dtype)
    acc_w2 = tl.zeros([HP, DP], dtype=dtype)
    acc_v0 = tl.zeros([HP, DP], dtype=dtype)
    acc_v1 = tl.zeros([HP, DP], dtype=dtype)
    acc_v2 = tl.zeros([HP, DP], dtype=dtype)
    p = tl.load(ptr_ptr + j)
    end = tl.load(ptr_ptr + j + 1)
    while p < end:
        e = tl.load(order_ptr + p).to(tl.int64)
        i = tl.load(dst_ptr + e).to(tl.int64)
        gi = g_ptr + i * (4 * H * D) + col
        dwe = dw_ptr + e * (3 * H * D) + col3
        if HAS_CW:
            c = tl.load(cw_ptr + e)
        else:
            c = 1.0
        gcw = tl.sum(tl.sum(tl.zeros([HP, DP], dtype=dtype), axis=1), axis=0)
        if T1:
            g0 = tl.load(gi, mask=fm, other=0.0)
            a = tl.load(a_ptr + e * H + h, mask=h < H, other=0.0)
            if HAS_DW:
                dw3 = tl.load(dwe, mask=fm, other=0.0)
            else:
                dw3 = 1.0
            t = g0 * w3 * dw3
            if NA:
                tl.store(ga_ptr + e * H + hv, tl.sum(t, axis=1) * c, mask=hv < H)
            if NCW:
                gcw += tl.sum(tl.sum(t * a, axis=1), axis=0)
            if NW:
                acc_w3 += a * g0 * dw3 * c
            if NDW:
                tl.store(gdw_ptr + e * (3 * H * D) + col3, a * g0 * w3 * c, mask=fm)
        if T2 or T3:
            g1 = tl.load(gi + H * D, mask=fm, other=0.0)
            g2 = tl.load(gi + 2 * H * D, mask=fm, other=0.0)
            g3 = tl.load(gi + 3 * H * D, mask=fm, other=0.0)
        if T2:
            if HAS_DW:
                dw1 = tl.load(dwe + D, mask=fm, other=0.0)
            else:
                dw1 = 1.0
            gv = g1 * v0 + g2 * v1 + g3 * v2
            if NCW:
                gcw += tl.sum(tl.sum(gv * w1 * dw1, axis=1), axis=0)
            if NW:
                acc_w1 += gv * dw1 * c
            if NDW:
                tl.store(gdw_ptr + e * (3 * H * D) + col3 + D, gv * w1 * c, mask=fm)
            if NV:
                s1 = w1 * dw1 * c
                acc_v0 += g1 * s1
                acc_v1 += g2 * s1
                acc_v2 += g3 * s1
        if T3:
            if HAS_DW:
                dw2 = tl.load(dwe + 2 * D, mask=fm, other=0.0)
            else:
                dw2 = 1.0
            ux = tl.load(u_ptr + e * 3)
            uy = tl.load(u_ptr + e * 3 + 1)
            uz = tl.load(u_ptr + e * 3 + 2)
            gu = g1 * ux + g2 * uy + g3 * uz
            if NCW:
                gcw += tl.sum(tl.sum(gu * w2 * dw2, axis=1), axis=0)
            if NW:
                acc_w2 += gu * dw2 * c
            if NDW:
                tl.store(gdw_ptr + e * (3 * H * D) + col3 + 2 * D, gu * w2 * c, mask=fm)
            if NU:
                s2 = w2 * dw2 * c
                tl.store(gu_ptr + e * 3, tl.sum(tl.sum(g1 * s2, axis=1), axis=0))
                tl.store(gu_ptr + e * 3 + 1, tl.sum(tl.sum(g2 * s2, axis=1), axis=0))
                tl.store(gu_ptr + e * 3 + 2, tl.sum(tl.sum(g3 * s2, axis=1), axis=0))
        if NCW:
            tl.store(gcw_ptr + e, gcw)
        p += 1
    if NW:
        gwj = gw_ptr + j * (3 * H * D) + col3
        tl.store(gwj, acc_w3, mask=fm)
        tl.store(gwj + D, acc_w1, mask=fm)
        tl.store(gwj + 2 * D, acc_w2, mask=fm)
    if NV:
        gvj = gv_ptr + j * (3 * H * D) + col
        tl.store(gvj, acc_v0, mask=fm)
        tl.store(gvj + H * D, acc_v1, mask=fm)
        tl.store(gvj + 2 * H * D, acc_v2, mask=fm)


def _pow2(n: int) -> int:
    return 1 << max(n - 1, 0).bit_length()


def _warps(n_features: int) -> int:
    return 4 if n_features <= 128 else 8


class _Pairs:
    """The pair list of a layer: neighbors, centers, atom count, head layout and the sorted orders."""

    def __init__(self, src: Tensor, dst: Tensor, n: int, heads: int, head_dim: int,
                 key: Optional[Tensor] = None):
        self.src, self.dst, self.n = src, dst, n
        self.order_dst, self.ptr_dst, self.order_src, self.ptr_src = orderings(src, dst, n, key=key)
        self.sizes = dict(H=heads, D=head_dim, HP=_pow2(heads), DP=_pow2(head_dim))
        self.warps = _warps(heads * head_dim)


class _Logits(MultilinearBlock):
    """The logits ``(P, H)``; operands ``q``, ``k`` ``(N, F)`` and, optionally, ``dk`` ``(P, F)``."""

    def __init__(self, pairs: _Pairs, has_dk: bool):
        self.pairs, self.has_dk = pairs, has_dk

    def value(self, ops: List[Tensor], only: int = -1) -> Tensor:
        pr = self.pairs
        q, k = ops[0].contiguous(), ops[1].contiguous()
        dk = ops[2].contiguous() if self.has_dk else q
        out = q.new_empty((pr.src.shape[0], pr.sizes["H"]))
        if pr.n:
            _logits_value[(pr.n,)](q, k, dk, pr.src, pr.order_dst, pr.ptr_dst, out, HAS_DK=self.has_dk,
                                   num_warps=pr.warps, **pr.sizes)
        return out

    def adjoints(self, g: Tensor, ops: List[Tensor], needs: List[bool],
                 only: int = -1) -> List[Optional[Tensor]]:
        pr = self.pairs
        g = g.contiguous()
        q, k = ops[0].contiguous(), ops[1].contiguous()
        dk = ops[2].contiguous() if self.has_dk else q
        out: List[Optional[Tensor]] = [None] * len(ops)
        if needs[0]:
            out[0] = torch.empty_like(q)
            if pr.n:
                _logits_grad_center[(pr.n,)](g, k, dk, pr.src, pr.order_dst, pr.ptr_dst, out[0],
                                             HAS_DK=self.has_dk, num_warps=pr.warps, **pr.sizes)
        need_dk = self.has_dk and needs[2]
        if needs[1] or need_dk:
            gk = torch.empty_like(k) if needs[1] else k
            gdk = torch.empty_like(dk) if need_dk else k
            if pr.n:
                _logits_grad_neighbor[(pr.n,)](g, q, k, dk, pr.dst, pr.order_src, pr.ptr_src, gk, gdk,
                                               HAS_DK=self.has_dk, NEED_K=needs[1], NEED_DK=need_dk,
                                               num_warps=pr.warps, **pr.sizes)
            if needs[1]:
                out[1] = gk
            if need_dk:
                out[2] = gdk
        return out


class _Messages(MultilinearBlock):
    """The four rows ``(N, 4, F)`` per atom; operands ``a`` ``(P, H)``, ``w`` ``(N, 3F)``,
    optionally ``dw`` ``(P, 3F)`` and ``cw`` ``(P,)``, then ``v`` ``(N, 3, F)`` and ``u`` ``(P, 3)``."""

    def __init__(self, pairs: _Pairs, has_dw: bool, has_cw: bool):
        self.pairs, self.has_dw, self.has_cw = pairs, has_dw, has_cw
        names = ["a", "w"] + (["dw"] if has_dw else []) + (["cw"] if has_cw else []) + ["v", "u"]
        self.index = {name: i for i, name in enumerate(names)}
        # the terms (dx, the neighbor vectors, the directions) each operand enters
        everywhere = (True, True, True)
        terms = {"a": (True, False, False), "w": everywhere, "dw": everywhere, "cw": everywhere,
                 "v": (False, True, False), "u": (False, False, True)}
        self.terms = [terms[name] for name in names]

    def _unpack(self, ops: List[Tensor]):
        get = {name: ops[i].contiguous() for name, i in self.index.items()}
        w = get["w"]
        return get["a"], w, get.get("dw", w), get.get("cw", w), get["v"], get["u"]

    def _active(self, only: int):
        return (True, True, True) if only < 0 else self.terms[only]

    def value(self, ops: List[Tensor], only: int = -1) -> Tensor:
        pr = self.pairs
        a, w, dw, cw, v, u = self._unpack(ops)
        t1, t2, t3 = self._active(only)
        out = v.new_empty((pr.n, 4, v.shape[2]))
        if pr.n:
            _messages_value[(pr.n,)](a, w, dw, cw, v, u, pr.src, pr.order_dst, pr.ptr_dst, out,
                                     HAS_DW=self.has_dw, HAS_CW=self.has_cw, T1=t1, T2=t2, T3=t3,
                                     num_warps=pr.warps, **pr.sizes)
        return out

    def adjoints(self, g: Tensor, ops: List[Tensor], needs: List[bool],
                 only: int = -1) -> List[Optional[Tensor]]:
        pr = self.pairs
        a, w, dw, cw, v, u = self._unpack(ops)
        active = self._active(only)
        t1, t2, t3 = active
        # an operand outside every active term has no gradient here
        needs = [need and any(x and y for x, y in zip(self.terms[i], active))
                 for i, need in enumerate(needs)]
        idx = self.index
        want = {name: needs[i] for name, i in idx.items()}
        full = t1 and t2 and t3
        alloc = torch.empty_like if full else torch.zeros_like
        ga = torch.empty_like(a) if want["a"] else a
        gw = alloc(w) if want["w"] else w
        gdw = alloc(dw) if want.get("dw", False) else w
        gcw = torch.empty_like(cw) if want.get("cw", False) else w
        gv = torch.empty_like(v) if want["v"] else v
        gu = torch.empty_like(u) if want["u"] else u
        if pr.n and any(needs):
            _messages_grad[(pr.n,)](g.contiguous(), a, w, dw, cw, v, u, pr.dst, pr.order_src, pr.ptr_src,
                                    ga, gw, gdw, gcw, gv, gu, HAS_DW=self.has_dw, HAS_CW=self.has_cw,
                                    T1=t1, T2=t2, T3=t3, NA=want["a"], NW=want["w"],
                                    NDW=want.get("dw", False), NCW=want.get("cw", False), NV=want["v"],
                                    NU=want["u"], num_warps=pr.warps, **pr.sizes)
        grads = {"a": ga, "w": gw, "dw": gdw, "cw": gcw, "v": gv, "u": gu}
        return [grads[name] if needs[i] else None for name, i in idx.items()]


def attention_logits(q: Tensor, k: Tensor, dk: Optional[Tensor], src: Tensor, dst: Tensor,
                     heads: int, key: Optional[Tensor] = None) -> Tensor:
    """The fused logits ``sum_(f in h) q[dst] k[src] dk`` of every pair and head, ``(P, H)``.

    Parameters
    ----------
    q, k : Tensor
        Queries and keys ``(N, F)``.
    dk : Tensor, optional
        The key filters of the pairs ``(P, F)``.
    src, dst : Tensor
        Neighbor and center of every pair ``(P,)``.
    heads : int
        Number of heads ``H`` (``F`` divisible by it).
    key : Tensor, optional
        The object that identifies the pair list for the sort cache (for example the edge
        index ``src`` and ``dst`` are rows of); by default ``src`` and ``dst`` themselves.

    Returns
    -------
    Tensor
        ``(P, H)``.
    """
    pairs = _Pairs(src, dst, q.shape[0], heads, q.shape[1] // heads, key)
    ops = [q, k] + ([dk] if dk is not None else [])
    return multilinear(_Logits(pairs, dk is not None), *ops)


def attention_messages(a: Tensor, w: Tensor, dw: Optional[Tensor], cw: Optional[Tensor], v: Tensor,
                       u: Tensor, src: Tensor, dst: Tensor, heads: int) -> Tensor:
    """The fused messages of every center: ``dx`` and the three rows of ``dv``, ``(N, 4, F)``.

    Parameters
    ----------
    a : Tensor
        Attention weights ``(P, H)``.
    w : Tensor
        Value projections ``(N, 3F)``, per head ``[s3 | s1 | s2]``.
    dw : Tensor, optional
        Value filters of the pairs ``(P, 3F)``, the same layout.
    cw : Tensor, optional
        Cutoff on the values ``(P,)``.
    v : Tensor
        Vector features ``(N, 3, F)``.
    u : Tensor
        Pair directions ``(P, 3)``.
    src, dst : Tensor
        Neighbor and center of every pair ``(P,)``.
    heads : int
        Number of heads.

    Returns
    -------
    Tensor
        ``(N, 4, F)``.
    """
    pairs = _Pairs(src, dst, v.shape[0], heads, v.shape[2] // heads)
    ops = [a, w] + ([dw] if dw is not None else []) + ([cw] if cw is not None else []) + [v, u]
    return multilinear(_Messages(pairs, dw is not None, cw is not None), *ops)
