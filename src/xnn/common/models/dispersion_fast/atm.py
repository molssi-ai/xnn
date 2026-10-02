"""The Axilrod-Teller-Muto three-body term of D3 and D4 as fused Triton kernels.

The reference (:func:`~xnn.common.models.dispersion.three_body_energy_chunked`)
enumerates the triplets of every block of centers into index tensors, gathers
per-triplet inputs, evaluates the summand and accumulates the results, in a
forward and a closed-form backward pass. Here one kernel does all of it in
registers. Each program takes one center and a block of its edges (sorted by
center, then neighbor) and walks the pairs of that block with the later edges
of the center: a ``BLOCK x BLOCK`` tile of pairs at a time, with the geometry,
the three pair ``C6``, the damping, the switching functions and the energy (and,
in the backward kernel, every derivative) evaluated per pair and accumulated
without ever storing a triplet.

Bookkeeping, as in the reference:

* A triangle of three distinct atoms is evaluated once, from the corner with
  the smallest index (the edges of a center are sorted by neighbor, so these
  are the pairs among the edges to larger-index neighbors), and a third of its
  energy goes to each corner.
* A triangle with a repeated atom (periodic self-images: a center with its own
  image, or two images of one neighbor) is evaluated at every visit with a
  third to the center. These exist only when a cell is narrower than twice the
  cutoff; a second launch covers the centers that have them.

The pair ``C6`` enters through per-atom factors: ``C6_xy = <f_x, g_y>`` with
``f, g (N, K)``. For D4 ``f = g = alpha_x(i w) sqrt(3 w / pi)`` over the 23
Casimir-Polder frequencies; for D3 ``f`` holds the reference-weighted ``C6``
rows of each species and ``g`` the reference weights placed at the atom's own
species. No ``(N, N)`` table and no per-triplet gather of 23 values. The pair
damping radii come from a small table over the species present.

Derivatives: the backward kernel returns the gradients with respect to the
edge vectors and both factor matrices in closed form; they are not themselves
differentiable. When a second derivative is needed (``create_graph=True``, e.g.
force training) the backward pass runs the reference implementation instead,
which is differentiable to second order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor

import triton
import triton.language as tl

#: pairs per tile side: a program holds ``BLOCK x BLOCK`` pairs in registers
BLOCK = 32
#: edges per chunk of the per-edge C6 gradient (bounds its ``(E, K)`` temporaries)
_EDGE_CHUNK = 1 << 20
_EPS = 2.220446049250313e-16       # the reference's lower bound on r_jk^2


@triton.jit
def _switch(r, cutoff, width, HAS_SWITCH: tl.constexpr):
    """The quintic switch and its derivative in ``r`` (1 and 0 without a window)."""
    if HAS_SWITCH:
        u = tl.minimum(tl.maximum((cutoff - r) / width, 0.0), 1.0)
        sw = u * u * u * (10.0 + u * (6.0 * u - 15.0))
        slope = tl.where((u > 0.0) & (u < 1.0), -30.0 * u * u * (1.0 - u) * (1.0 - u) / width, 0.0)
    else:
        sw = r * 0.0 + 1.0
        slope = r * 0.0
    return sw, slope


@triton.jit
def _pair_mask(ia, ib, ma, mb, ja, kb, c, w, cut2, eps, MODE: tl.constexpr):
    pair = ma[:, None] & mb[None, :] & (ia[:, None] < ib[None, :])
    if MODE == 1:
        pair = pair & (ja[:, None] != kb[None, :])
    else:
        pair = pair & ((ja[:, None] == c) | (kb[None, :] == c) | (ja[:, None] == kb[None, :]))
    return pair & (w <= cut2) & (w > eps)


@triton.jit
def _atm_forward(vec_ptr, src_ptr, centers_ptr, lo_ptr, hi_ptr, spec_ptr, fa_ptr, fb_ptr,
                 table_ptr, const_ptr, energy_ptr, K, S,
                 MODE: tl.constexpr, BLOCK: tl.constexpr, HAS_SWITCH: tl.constexpr):
    pid = tl.program_id(0)
    c = tl.load(centers_ptr + pid)
    lo = tl.load(lo_ptr + pid)
    hi = tl.load(hi_ptr + pid)
    a0 = lo + tl.program_id(1) * BLOCK
    if a0 < hi:
        s9 = tl.load(const_ptr + 0)
        alp3 = tl.load(const_ptr + 1)
        cut2 = tl.load(const_ptr + 2)
        cutoff = tl.load(const_ptr + 3)
        width = tl.load(const_ptr + 4)
        eps = tl.load(const_ptr + 5)
        ia = a0 + tl.arange(0, BLOCK)
        ma = ia < hi
        ax = tl.load(vec_ptr + ia * 3 + 0, mask=ma, other=0.0)
        ay = tl.load(vec_ptr + ia * 3 + 1, mask=ma, other=0.0)
        az = tl.load(vec_ptr + ia * 3 + 2, mask=ma, other=0.0)
        xa = ax * ax + ay * ay + az * az
        ja = tl.load(src_ptr + ia, mask=ma, other=0)
        sc = tl.load(spec_ptr + c)
        sj = tl.load(spec_ptr + ja, mask=ma, other=0)
        c6a = xa * 0.0
        for k in range(K):
            c6a += tl.load(fa_ptr + c * K + k) * tl.load(fb_ptr + ja * K + k, mask=ma, other=0.0)
        pa = tl.load(table_ptr + sc * S + sj, mask=ma, other=1.0)
        swa, _sa = _switch(tl.sqrt(xa), cutoff, width, HAS_SWITCH)
        e_row = xa * 0.0
        for b0 in range(a0, hi, BLOCK):
            ib = b0 + tl.arange(0, BLOCK)
            mb = ib < hi
            bx = tl.load(vec_ptr + ib * 3 + 0, mask=mb, other=0.0)
            by = tl.load(vec_ptr + ib * 3 + 1, mask=mb, other=0.0)
            bz = tl.load(vec_ptr + ib * 3 + 2, mask=mb, other=0.0)
            yb = bx * bx + by * by + bz * bz
            kb = tl.load(src_ptr + ib, mask=mb, other=0)
            sk = tl.load(spec_ptr + kb, mask=mb, other=0)
            c6b = yb * 0.0
            for k in range(K):
                c6b += tl.load(fa_ptr + c * K + k) * tl.load(fb_ptr + kb * K + k, mask=mb, other=0.0)
            pb = tl.load(table_ptr + sc * S + sk, mask=mb, other=1.0)
            swb, _sb = _switch(tl.sqrt(yb), cutoff, width, HAS_SWITCH)
            dx = ax[:, None] - bx[None, :]
            dy = ay[:, None] - by[None, :]
            dz = az[:, None] - bz[None, :]
            w = dx * dx + dy * dy + dz * dz
            keep = _pair_mask(ia, ib, ma, mb, ja, kb, c, w, cut2, eps, MODE)
            c6jk = w * 0.0
            for k in range(K):
                c6jk += (tl.load(fa_ptr + ja * K + k, mask=ma, other=0.0)[:, None]
                         * tl.load(fb_ptr + kb * K + k, mask=mb, other=0.0)[None, :])
            pjk = tl.load(table_ptr + sj[:, None] * S + sk[None, :],
                          mask=ma[:, None] & mb[None, :], other=1.0)
            x = tl.where(keep, xa[:, None] + w * 0.0, 1.0)
            y = tl.where(keep, yb[None, :] + w * 0.0, 1.0)
            w = tl.where(keep, w, 1.0)
            prod2 = x * y * w
            prod1 = tl.sqrt(prod2)
            prod3 = prod2 * prod1
            prod5 = prod3 * prod2
            r0 = tl.where(keep, pa[:, None] * pb[None, :] * pjk, 1.0)
            t = tl.exp(alp3 * tl.log(r0 / prod1))
            damp = 1.0 / (1.0 + 6.0 * t)
            f1 = x + w - y
            f2 = x - w + y
            f3 = -x + w + y
            ang = 0.375 * f1 * f2 * f3 / prod5 + 1.0 / prod3
            swjk, _sjk = _switch(tl.sqrt(w), cutoff, width, HAS_SWITCH)
            c9 = tl.sqrt(tl.abs(c6a[:, None] * c6b[None, :] * c6jk))
            e = tl.where(keep, s9 * c9 * ang * damp * swa[:, None] * swb[None, :] * swjk / 3.0, 0.0)
            e_row += tl.sum(e, axis=1)
            if MODE == 1:
                tl.atomic_add(energy_ptr + kb, tl.sum(e, axis=0), mask=mb)
        tl.atomic_add(energy_ptr + c, tl.sum(e_row, axis=0))
        if MODE == 1:
            tl.atomic_add(energy_ptr + ja, e_row, mask=ma)


@triton.jit
def _atm_backward(vec_ptr, src_ptr, centers_ptr, lo_ptr, hi_ptr, spec_ptr, fa_ptr, fb_ptr,
                  table_ptr, const_ptr, go_ptr, gvec_ptr, gc6_ptr, gfa_ptr, gfb_ptr, K, S,
                  MODE: tl.constexpr, BLOCK: tl.constexpr, HAS_SWITCH: tl.constexpr):
    pid = tl.program_id(0)
    c = tl.load(centers_ptr + pid)
    lo = tl.load(lo_ptr + pid)
    hi = tl.load(hi_ptr + pid)
    a0 = lo + tl.program_id(1) * BLOCK
    if a0 < hi:
        s9 = tl.load(const_ptr + 0)
        alp3 = tl.load(const_ptr + 1)
        cut2 = tl.load(const_ptr + 2)
        cutoff = tl.load(const_ptr + 3)
        width = tl.load(const_ptr + 4)
        eps = tl.load(const_ptr + 5)
        ia = a0 + tl.arange(0, BLOCK)
        ma = ia < hi
        ax = tl.load(vec_ptr + ia * 3 + 0, mask=ma, other=0.0)
        ay = tl.load(vec_ptr + ia * 3 + 1, mask=ma, other=0.0)
        az = tl.load(vec_ptr + ia * 3 + 2, mask=ma, other=0.0)
        xa = ax * ax + ay * ay + az * az
        ja = tl.load(src_ptr + ia, mask=ma, other=0)
        sc = tl.load(spec_ptr + c)
        sj = tl.load(spec_ptr + ja, mask=ma, other=0)
        c6a = xa * 0.0
        for k in range(K):
            c6a += tl.load(fa_ptr + c * K + k) * tl.load(fb_ptr + ja * K + k, mask=ma, other=0.0)
        c6a_safe = tl.where(c6a != 0.0, c6a, 1.0)
        pa = tl.load(table_ptr + sc * S + sj, mask=ma, other=1.0)
        ra = tl.sqrt(tl.where(ma, xa, 1.0))
        swa, swa_r = _switch(ra, cutoff, width, HAS_SWITCH)
        go_c = tl.load(go_ptr + c)
        go_j = tl.load(go_ptr + ja, mask=ma, other=0.0)
        g_ax = xa * 0.0
        g_ay = xa * 0.0
        g_az = xa * 0.0
        g_c6a = xa * 0.0
        for b0 in range(a0, hi, BLOCK):
            ib = b0 + tl.arange(0, BLOCK)
            mb = ib < hi
            bx = tl.load(vec_ptr + ib * 3 + 0, mask=mb, other=0.0)
            by = tl.load(vec_ptr + ib * 3 + 1, mask=mb, other=0.0)
            bz = tl.load(vec_ptr + ib * 3 + 2, mask=mb, other=0.0)
            yb = bx * bx + by * by + bz * bz
            kb = tl.load(src_ptr + ib, mask=mb, other=0)
            sk = tl.load(spec_ptr + kb, mask=mb, other=0)
            c6b = yb * 0.0
            for k in range(K):
                c6b += tl.load(fa_ptr + c * K + k) * tl.load(fb_ptr + kb * K + k, mask=mb, other=0.0)
            c6b_safe = tl.where(c6b != 0.0, c6b, 1.0)
            pb = tl.load(table_ptr + sc * S + sk, mask=mb, other=1.0)
            rb = tl.sqrt(tl.where(mb, yb, 1.0))
            swb, swb_r = _switch(rb, cutoff, width, HAS_SWITCH)
            dx = ax[:, None] - bx[None, :]
            dy = ay[:, None] - by[None, :]
            dz = az[:, None] - bz[None, :]
            w = dx * dx + dy * dy + dz * dz
            keep = _pair_mask(ia, ib, ma, mb, ja, kb, c, w, cut2, eps, MODE)
            c6jk = w * 0.0
            for k in range(K):
                c6jk += (tl.load(fa_ptr + ja * K + k, mask=ma, other=0.0)[:, None]
                         * tl.load(fb_ptr + kb * K + k, mask=mb, other=0.0)[None, :])
            c6jk_safe = tl.where(c6jk != 0.0, c6jk, 1.0)
            pjk = tl.load(table_ptr + sj[:, None] * S + sk[None, :],
                          mask=ma[:, None] & mb[None, :], other=1.0)
            weight = go_c + w * 0.0
            if MODE == 1:
                go_k = tl.load(go_ptr + kb, mask=mb, other=0.0)
                weight = weight + go_j[:, None] + go_k[None, :]
            x = tl.where(keep, xa[:, None] + w * 0.0, 1.0)
            y = tl.where(keep, yb[None, :] + w * 0.0, 1.0)
            w = tl.where(keep, w, 1.0)
            rjk = tl.sqrt(w)
            prod2 = x * y * w
            prod1 = tl.sqrt(prod2)
            prod3 = prod2 * prod1
            prod5 = prod3 * prod2
            r0 = tl.where(keep, pa[:, None] * pb[None, :] * pjk, 1.0)
            t = tl.exp(alp3 * tl.log(r0 / prod1))
            damp = 1.0 / (1.0 + 6.0 * t)
            f1 = x + w - y
            f2 = x - w + y
            f3 = -x + w + y
            nn = f1 * f2 * f3
            ang = 0.375 * nn / prod5 + 1.0 / prod3
            sjk, sjk_r = _switch(rjk, cutoff, width, HAS_SWITCH)
            sab = swa[:, None] * swb[None, :]
            sw = sab * sjk
            c9 = tl.sqrt(tl.abs(c6a[:, None] * c6b[None, :] * c6jk))
            base = tl.where(keep, weight * s9 * c9 / 3.0, 0.0)
            e = base * ang * damp * sw
            # derivatives in the squared distances x (c-j), y (c-k), w (j-k)
            n_x = f2 * f3 + f1 * f3 - f1 * f2
            n_y = -f2 * f3 + f1 * f3 + f1 * f2
            n_w = f2 * f3 - f1 * f3 + f1 * f2
            ang_x = 0.375 * (n_x / prod5 - 2.5 * nn / (prod5 * x)) - 1.5 / (prod3 * x)
            ang_y = 0.375 * (n_y / prod5 - 2.5 * nn / (prod5 * y)) - 1.5 / (prod3 * y)
            ang_w = 0.375 * (n_w / prod5 - 2.5 * nn / (prod5 * w)) - 1.5 / (prod3 * w)
            d_common = 3.0 * alp3 * t * damp * damp
            sw_x = (swa_r / (2.0 * ra))[:, None] * swb[None, :] * sjk
            sw_y = (swb_r / (2.0 * rb))[None, :] * swa[:, None] * sjk
            sw_w = sjk_r / (2.0 * rjk) * sab
            g_x = base * (ang_x * damp * sw + ang * (d_common / x) * sw + ang * damp * sw_x)
            g_y = base * (ang_y * damp * sw + ang * (d_common / y) * sw + ang * damp * sw_y)
            g_w = base * (ang_w * damp * sw + ang * (d_common / w) * sw + ang * damp * sw_w)
            g_x = tl.where(keep, g_x, 0.0)
            g_y = tl.where(keep, g_y, 0.0)
            g_w = tl.where(keep, g_w, 0.0)
            # the two edges: v_jk = v_a - v_b
            g_ax += tl.sum(2.0 * (g_x * ax[:, None] + g_w * dx), axis=1)
            g_ay += tl.sum(2.0 * (g_x * ay[:, None] + g_w * dy), axis=1)
            g_az += tl.sum(2.0 * (g_x * az[:, None] + g_w * dz), axis=1)
            tl.atomic_add(gvec_ptr + ib * 3 + 0, tl.sum(2.0 * (g_y * bx[None, :] - g_w * dx), axis=0), mask=mb)
            tl.atomic_add(gvec_ptr + ib * 3 + 1, tl.sum(2.0 * (g_y * by[None, :] - g_w * dy), axis=0), mask=mb)
            tl.atomic_add(gvec_ptr + ib * 3 + 2, tl.sum(2.0 * (g_y * bz[None, :] - g_w * dz), axis=0), mask=mb)
            # C6 (through C9 = sqrt(C6 C6 C6)): dE / dC6 = E / (2 C6)
            g_c6a += tl.sum(tl.where(keep, e / (2.0 * c6a_safe[:, None]), 0.0), axis=1)
            tl.atomic_add(gc6_ptr + ib, tl.sum(tl.where(keep, e / (2.0 * c6b_safe[None, :]), 0.0), axis=0),
                          mask=mb)
            g3 = tl.where(keep, e / (2.0 * c6jk_safe), 0.0)
            for k in range(K):
                fj = tl.load(fa_ptr + ja * K + k, mask=ma, other=0.0)
                gk = tl.load(fb_ptr + kb * K + k, mask=mb, other=0.0)
                tl.atomic_add(gfa_ptr + ja * K + k, tl.sum(g3 * gk[None, :], axis=1), mask=ma)
                tl.atomic_add(gfb_ptr + kb * K + k, tl.sum(g3 * fj[:, None], axis=0), mask=mb)
        tl.atomic_add(gvec_ptr + ia * 3 + 0, g_ax, mask=ma)
        tl.atomic_add(gvec_ptr + ia * 3 + 1, g_ay, mask=ma)
        tl.atomic_add(gvec_ptr + ia * 3 + 2, g_az, mask=ma)
        tl.atomic_add(gc6_ptr + ia, g_c6a, mask=ma)


@dataclass
class _Launch:
    """One kernel launch: the centers and the edge range of each."""

    centers: Tensor     # (P,) int32
    lo: Tensor          # (P,) int32
    hi: Tensor          # (P,) int32
    blocks: int         # row blocks per center (grid axis 1)
    mode: int


@dataclass
class TripletPlan:
    """The center-sorted edge list of one evaluation and its kernel launches."""

    order: Tensor       # (E,) the sort of the input edges by (center, neighbor)
    src: Tensor         # (E,) int32 neighbor of each sorted edge
    dst: Tensor         # (E,) int64 center of each sorted edge
    launches: list
    n_atoms: int


def plan_triplets(edge_index: Tensor, n_atoms: int) -> TripletPlan:
    """Sort the edges by (center, neighbor) and lay out the kernel launches."""
    with torch.no_grad():
        src, dst = edge_index[0], edge_index[1]
        key = dst * n_atoms + src
        key_sorted, order = torch.sort(key)
        src_s, dst_s = src.index_select(0, order), dst.index_select(0, order)
        counts = torch.bincount(dst_s, minlength=n_atoms)
        offsets = torch.zeros(n_atoms + 1, dtype=torch.long, device=src.device)
        offsets[1:] = torch.cumsum(counts, 0)
        lower = torch.bincount(dst_s[src_s <= dst_s], minlength=n_atoms)
        lo1 = offsets[:-1] + lower
        hi = offsets[1:]
        launches = []
        span = int((hi - lo1).max()) if n_atoms else 0
        if span >= 2:
            launches.append(_Launch(torch.arange(n_atoms, dtype=torch.int32, device=src.device),
                                    lo1.to(torch.int32), hi.to(torch.int32),
                                    triton.cdiv(span, BLOCK), 1))
        # centers with self-image triangles: an edge to their own image, or two
        # edges to images of one neighbor (equal sorted keys)
        repeated = src_s == dst_s
        if key_sorted.numel() > 1:
            same = key_sorted[1:] == key_sorted[:-1]
            repeated[1:] |= same
            repeated[:-1] |= same
        centers2 = torch.unique(dst_s[repeated])
        if centers2.numel():
            lo2, hi2 = offsets[centers2], offsets[centers2 + 1]
            span2 = int((hi2 - lo2).max())
            if span2 >= 2:
                launches.append(_Launch(centers2.to(torch.int32), lo2.to(torch.int32),
                                        hi2.to(torch.int32), triton.cdiv(span2, BLOCK), 2))
        return TripletPlan(order, src_s.to(torch.int32).contiguous(), dst_s, launches, n_atoms)


@dataclass
class ATMConstants:
    """The scalars and tables of one ATM term (species-indexed)."""

    species: Tensor     # (N,) int32 index into the table
    table: Tensor       # (S, S) pair factors of R0 (the damping radius is their product)
    s9: float
    alp3: float
    cutoff: float       # bohr
    width: float        # switching width, bohr (<= 0: sharp cutoff)


def _const_tensor(consts: ATMConstants, like: Tensor) -> Tensor:
    width = min(consts.width, consts.cutoff) if consts.width > 0.0 else 1.0
    return torch.tensor([consts.s9, consts.alp3, consts.cutoff * consts.cutoff, consts.cutoff,
                         width, _EPS], dtype=like.dtype, device=like.device)


def _launch_meta(like: Tensor) -> dict:
    return {"num_warps": 4 if like.dtype == torch.float32 else 8}


class _ATMEnergy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, edge_vec: Tensor, feat_a: Tensor, feat_b: Tensor, plan: TripletPlan,
                consts: ATMConstants, reference: Optional[Callable]):
        vec = edge_vec.detach().index_select(0, plan.order).contiguous()
        fa = feat_a.detach().contiguous()
        fb = feat_b.detach().contiguous()
        cst = _const_tensor(consts, vec)
        energy = torch.zeros(plan.n_atoms, dtype=vec.dtype, device=vec.device)
        table = consts.table.to(vec.dtype).contiguous()
        k, s = int(fa.shape[1]), int(table.shape[0])
        for launch in plan.launches:
            grid = (int(launch.centers.shape[0]), launch.blocks)
            _atm_forward[grid](vec, plan.src, launch.centers, launch.lo, launch.hi,
                               consts.species, fa, fb, table, cst, energy, k, s,
                               MODE=launch.mode, BLOCK=BLOCK, HAS_SWITCH=consts.width > 0.0,
                               **_launch_meta(vec))
        ctx.save_for_backward(edge_vec, feat_a, feat_b)
        ctx.plan, ctx.consts, ctx.reference = plan, consts, reference
        ctx.shared = feat_a is feat_b
        return energy

    @staticmethod
    def backward(ctx, grad_energy: Tensor):
        edge_vec, feat_a, feat_b = ctx.saved_tensors
        if torch.is_grad_enabled() and ctx.reference is not None:
            # a differentiable backward is asked for: the reference is
            # differentiable to second order, the kernel to first
            inputs = [edge_vec, feat_a] if ctx.shared else [edge_vec, feat_a, feat_b]
            wanted = [t for t in inputs if t.requires_grad]
            with torch.enable_grad():
                energy = ctx.reference(edge_vec, feat_a, feat_a if ctx.shared else feat_b)
                found = iter(torch.autograd.grad(energy, wanted, grad_energy, create_graph=True,
                                                 allow_unused=True))
            grads = [next(found) if t.requires_grad else None for t in inputs]
            if ctx.shared:
                grads.append(None)
            return (*grads, None, None, None)
        plan, consts = ctx.plan, ctx.consts
        with torch.no_grad():
            vec = edge_vec.detach().index_select(0, plan.order).contiguous()
            fa = feat_a.detach().contiguous()
            fb = feat_b.detach().contiguous()
            go = grad_energy.detach().to(vec.dtype).contiguous()
            cst = _const_tensor(consts, vec)
            table = consts.table.to(vec.dtype).contiguous()
            g_vec = torch.zeros_like(vec)
            g_c6 = torch.zeros(vec.shape[0], dtype=vec.dtype, device=vec.device)
            g_fa = torch.zeros_like(fa)
            g_fb = torch.zeros_like(fb)
            k, s = int(fa.shape[1]), int(table.shape[0])
            for launch in plan.launches:
                grid = (int(launch.centers.shape[0]), launch.blocks)
                _atm_backward[grid](vec, plan.src, launch.centers, launch.lo, launch.hi,
                                    consts.species, fa, fb, table, cst, go, g_vec, g_c6, g_fa, g_fb,
                                    k, s, MODE=launch.mode, BLOCK=BLOCK,
                                    HAS_SWITCH=consts.width > 0.0, **_launch_meta(vec))
            # the center sides: C6_cj = <f_c, g_j> per sorted edge
            src_l = plan.src.to(torch.long)
            for e0 in range(0, vec.shape[0], _EDGE_CHUNK):
                e1 = min(e0 + _EDGE_CHUNK, vec.shape[0])
                gc = g_c6[e0:e1, None]
                c_idx, j_idx = plan.dst[e0:e1], src_l[e0:e1]
                g_fa.index_add_(0, c_idx, gc * fb.index_select(0, j_idx))
                g_fb.index_add_(0, j_idx, gc * fa.index_select(0, c_idx))
            grad_vec = torch.zeros_like(g_vec).index_copy_(0, plan.order, g_vec)
        return grad_vec.to(edge_vec.dtype), g_fa, g_fb, None, None, None


def atm_energy(edge_index: Tensor, edge_vec: Tensor, feat_a: Tensor, feat_b: Tensor,
               consts: ATMConstants, n_atoms: int,
               reference: Optional[Callable] = None) -> Tensor:
    """Per-atom ATM energies ``(N,)`` in hartree, with the Triton kernels.

    Parameters
    ----------
    edge_index : Tensor
        Directed edges ``[src, dst]`` within the three-body cutoff, ``(2, E)``,
        both directions of every pair.
    edge_vec : Tensor
        Their vectors in bohr, ``(E, 3)`` (differentiable).
    feat_a, feat_b : Tensor
        Pair-C6 factors ``(N, K)``, ``C6_xy = <feat_a[x], feat_b[y]>``
        (differentiable; may be the same tensor).
    consts : ATMConstants
        Species, the damping-radius table and the scalars.
    n_atoms : int
        Number of atoms.
    reference : callable, optional
        ``reference(edge_vec, feat_a, feat_b) -> (N,)``: the same energy in
        differentiable torch ops, run by the backward pass when a second
        derivative is needed; without it the kernel's (first-order) gradient
        is used throughout.
    """
    plan = plan_triplets(edge_index, n_atoms)
    return _ATMEnergy.apply(edge_vec, feat_a, feat_b, plan, consts, reference)
