"""Triton kernels and autograd functions of the NewtonNet fast path (see :mod:`xnn.gnn.fast.newtonnet`).

Imported only when the fast path runs, so that ``triton`` is needed only there.

Two kernels cover every edge sum of the force and displacement features and all their derivatives:

* the scatter kernel adds ``c1_e u_e`` (an outer product, ``(F,)`` by ``(3,)``) and ``c2_e v[g_e]``
  (feature-wise) onto the atoms ``s_e``;
* the gather kernel forms, per edge, ``sum_k G[s_e, k] u_e,k``, ``sum_f G[s_e, :, f] c_e,f`` and
  ``sum_k G[s_e, k] v[g_e, k]`` from an atom quantity ``G``.

Both maps are bilinear, and the derivatives of each are maps of the other kind, so the autograd function of
each calls the other in its backward pass: forces, the second derivatives of a force loss and higher ones
run fused, and no ``(E, 3, F)`` tensor is written.
"""
# no postponed annotations: Triton's interpreter (TRITON_INTERPRET=1) recognizes the
# tl.constexpr parameters of a kernel only from evaluated annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor

#: Values per program and warps: small tiles over eight warps stay in registers (the PaiNN fast path's
#: measured choice for kernels of this form).
TILE, WARPS = 512, 8


@triton.jit
def _scatter_kernel(c1_ptr, u_ptr, c2_ptr, v_ptr, g_ptr, s_ptr, out_ptr, n_edges, F: tl.constexpr,
                    FP: tl.constexpr, OUTER: tl.constexpr, PROD: tl.constexpr, BLOCK_E: tl.constexpr):
    """``out[s] += c1 u`` (with ``OUTER``) ``+ c2 v[g]`` (with ``PROD``), per edge."""
    e = (tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)).to(tl.int64)
    me = e < n_edges
    f = tl.arange(0, FP)[None, :]
    m2 = me[:, None] & (f < F)
    s = tl.load(s_ptr + e, mask=me, other=0)[:, None]
    row = out_ptr + s * (3 * F) + f
    if OUTER:
        c1 = tl.load(c1_ptr + e[:, None] * F + f, mask=m2, other=0.0)
        u0 = tl.load(u_ptr + e * 3, mask=me, other=0.0)[:, None]
        u1 = tl.load(u_ptr + e * 3 + 1, mask=me, other=0.0)[:, None]
        u2 = tl.load(u_ptr + e * 3 + 2, mask=me, other=0.0)[:, None]
        if PROD:
            g = tl.load(g_ptr + e, mask=me, other=0)[:, None]
            c2 = tl.load(c2_ptr + e[:, None] * F + f, mask=m2, other=0.0)
            v_row = v_ptr + g * (3 * F) + f
            tl.atomic_add(row, c1 * u0 + c2 * tl.load(v_row, mask=m2, other=0.0), mask=m2)
            tl.atomic_add(row + F, c1 * u1 + c2 * tl.load(v_row + F, mask=m2, other=0.0), mask=m2)
            tl.atomic_add(row + 2 * F, c1 * u2 + c2 * tl.load(v_row + 2 * F, mask=m2, other=0.0), mask=m2)
        else:
            tl.atomic_add(row, c1 * u0, mask=m2)
            tl.atomic_add(row + F, c1 * u1, mask=m2)
            tl.atomic_add(row + 2 * F, c1 * u2, mask=m2)
    else:
        g = tl.load(g_ptr + e, mask=me, other=0)[:, None]
        c2 = tl.load(c2_ptr + e[:, None] * F + f, mask=m2, other=0.0)
        v_row = v_ptr + g * (3 * F) + f
        tl.atomic_add(row, c2 * tl.load(v_row, mask=m2, other=0.0), mask=m2)
        tl.atomic_add(row + F, c2 * tl.load(v_row + F, mask=m2, other=0.0), mask=m2)
        tl.atomic_add(row + 2 * F, c2 * tl.load(v_row + 2 * F, mask=m2, other=0.0), mask=m2)


@triton.jit
def _gather_kernel(G_ptr, u_ptr, c_ptr, v_ptr, s_ptr, g_ptr, ok_ptr, of_ptr, op_ptr, n_edges,
                   F: tl.constexpr, FP: tl.constexpr, K: tl.constexpr, FF: tl.constexpr, P: tl.constexpr,
                   BLOCK_E: tl.constexpr):
    """Per edge: ``ok = sum_k G[s, k] u_k`` (``K``), ``of_k = sum_f G[s, k, f] c_f`` (``FF``) and
    ``op = sum_k G[s, k] v[g, k]`` (``P``)."""
    e = (tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)).to(tl.int64)
    me = e < n_edges
    f = tl.arange(0, FP)[None, :]
    m2 = me[:, None] & (f < F)
    s = tl.load(s_ptr + e, mask=me, other=0)[:, None]
    G_row = G_ptr + s * (3 * F) + f
    G0 = tl.load(G_row, mask=m2, other=0.0)
    G1 = tl.load(G_row + F, mask=m2, other=0.0)
    G2 = tl.load(G_row + 2 * F, mask=m2, other=0.0)
    out_row = e[:, None] * F + f
    if K:
        u0 = tl.load(u_ptr + e * 3, mask=me, other=0.0)[:, None]
        u1 = tl.load(u_ptr + e * 3 + 1, mask=me, other=0.0)[:, None]
        u2 = tl.load(u_ptr + e * 3 + 2, mask=me, other=0.0)[:, None]
        tl.store(ok_ptr + out_row, G0 * u0 + G1 * u1 + G2 * u2, mask=m2)
    if FF:
        c = tl.load(c_ptr + out_row, mask=m2, other=0.0)
        tl.store(of_ptr + e * 3, tl.sum(G0 * c, axis=1), mask=me)
        tl.store(of_ptr + e * 3 + 1, tl.sum(G1 * c, axis=1), mask=me)
        tl.store(of_ptr + e * 3 + 2, tl.sum(G2 * c, axis=1), mask=me)
    if P:
        g = tl.load(g_ptr + e, mask=me, other=0)[:, None]
        v_row = v_ptr + g * (3 * F) + f
        tl.store(op_ptr + out_row, G0 * tl.load(v_row, mask=m2, other=0.0)
                 + G1 * tl.load(v_row + F, mask=m2, other=0.0)
                 + G2 * tl.load(v_row + 2 * F, mask=m2, other=0.0), mask=m2)


def _launch(n_features: int) -> Tuple[int, int, int]:
    """Padded feature width, edges per program and warps per program."""
    fp = triton.next_power_of_2(n_features)
    return fp, max(4, TILE // fp), WARPS


def _c(t: Optional[Tensor]) -> Optional[Tensor]:
    return None if t is None else t.contiguous()


def _empty(t: Optional[Tensor]) -> Optional[Tensor]:
    """``None`` for the zero-size placeholder of an output that was not formed."""
    return None if t is None or t.numel() == 0 else t


class Scatter(torch.autograd.Function):
    """``out[s_e] += c1_e u_e + c2_e v[g_e]`` (either term may be absent), differentiable to any order."""

    @staticmethod
    def forward(ctx, c1: Optional[Tensor], u: Optional[Tensor], c2: Optional[Tensor], v: Optional[Tensor],
                g: Optional[Tensor], s: Tensor, n_out: int) -> Tensor:
        like = c1 if c1 is not None else c2
        n_edges, F = like.shape
        out = like.new_zeros((n_out, 3, F))
        if n_edges:
            fp, block, warps = _launch(F)
            _scatter_kernel[(triton.cdiv(n_edges, block),)](
                like if c1 is None else c1, like if u is None else u, like if c2 is None else c2,
                like if v is None else v, s if g is None else g, s, out, n_edges, F=F, FP=fp,
                OUTER=c1 is not None, PROD=c2 is not None, BLOCK_E=block, num_warps=warps)
        ctx.save_for_backward(c1, u, c2, v, g, s)
        ctx.n_v = 0 if v is None else v.shape[0]
        return out

    @staticmethod
    def backward(ctx, g_out):
        c1, u, c2, v, g, s = ctx.saved_tensors
        g_out = g_out.contiguous()
        outer, prod = c1 is not None, c2 is not None
        g_c1, g_u, g_c2 = (_empty(t) for t in Gather.apply(
            g_out, u if outer else None, c1 if outer else None, v if prod else None, s, g if prod else None))
        g_v = Scatter.apply(None, None, c2, g_out, s, g, ctx.n_v) if prod else None
        return g_c1, g_u, g_c2, g_v, None, None, None


class Gather(torch.autograd.Function):
    """Per edge ``sum_k G[s, k] u_k``, ``sum_f G[s, :, f] c_f`` and ``sum_k G[s, k] v[g, k]`` (each formed
    only when its second factor is given; zero-size placeholders otherwise), differentiable to any order."""

    @staticmethod
    def forward(ctx, G: Tensor, u: Optional[Tensor], c: Optional[Tensor], v: Optional[Tensor], s: Tensor,
                g: Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        n_edges, F = s.shape[0], G.shape[2]
        o_k = G.new_empty((n_edges, F)) if u is not None else G.new_empty(0)
        o_f = G.new_empty((n_edges, 3)) if c is not None else G.new_empty(0)
        o_p = G.new_empty((n_edges, F)) if v is not None else G.new_empty(0)
        if n_edges:
            fp, block, warps = _launch(F)
            _gather_kernel[(triton.cdiv(n_edges, block),)](
                G, G if u is None else u, G if c is None else c, G if v is None else v, s,
                s if g is None else g, G if u is None else o_k, G if c is None else o_f,
                G if v is None else o_p, n_edges, F=F, FP=fp, K=u is not None, FF=c is not None,
                P=v is not None, BLOCK_E=block, num_warps=warps)
        ctx.save_for_backward(G, u, c, v, s, g)
        return o_k, o_f, o_p

    @staticmethod
    def backward(ctx, g_k, g_f, g_p):
        G, u, c, v, s, g = ctx.saved_tensors
        n = G.shape[0]
        g_k = _c(g_k) if u is not None else None
        g_f = _c(g_f) if c is not None else None
        g_p = _c(g_p) if v is not None else None
        g_G = None
        if g_k is not None or g_p is not None:
            g_G = Scatter.apply(g_k, u if g_k is not None else None, g_p, v if g_p is not None else None,
                                g, s, n)
        if g_f is not None:
            part = Scatter.apply(c, g_f, None, None, None, s, n)
            g_G = part if g_G is None else g_G + part
        g_u = g_c = None
        if g_k is not None or g_f is not None:
            # d(sum_k G u_k)/du_k = sum_f G[s, k, f] g_k,f and d(sum_f G c_f)/dc_f = sum_k G[s, k, f] g_f,k
            from_f, from_k, _ = Gather.apply(G, g_f, g_k, None, s, None)
            g_u, g_c = _empty(from_k), _empty(from_f)
        g_v = Scatter.apply(None, None, g_p, G, s, g, v.shape[0]) if g_p is not None else None
        return g_G, g_u, g_c, g_v, None, None


def edge_sum(n_atoms: int, src: Tensor, dst: Tensor, c1: Optional[Tensor] = None, u: Optional[Tensor] = None,
             c2: Optional[Tensor] = None, v: Optional[Tensor] = None) -> Tensor:
    """``out_i = sum_(j -> i) c1_ij u_ij + c2_ij v_j``, shape ``(n_atoms, 3, F)``.

    Parameters
    ----------
    n_atoms : int
        Number of atoms.
    src, dst : Tensor
        Neighbor ``j`` and center ``i`` of every edge, ``(E,)``.
    c1 : Tensor, optional
        Edge coefficients of the directional term, ``(E, F)``.
    u : Tensor, optional
        Edge vectors of the directional term, ``(E, 3)``.
    c2 : Tensor, optional
        Edge coefficients of the neighbor term, ``(E, F)``.
    v : Tensor, optional
        Atom vectors of the neighbor term, ``(N, 3, F)``.

    Returns
    -------
    Tensor
        The sums ``(n_atoms, 3, F)``.
    """
    return Scatter.apply(_c(c1), _c(u), _c(c2), _c(v), src.contiguous() if c2 is not None else None,
                         dst.contiguous(), n_atoms)
