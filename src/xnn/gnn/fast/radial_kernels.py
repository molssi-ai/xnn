"""Triton kernels of the fused hidden layers of a radial network (see :class:`~xnn.gnn.models.blocks.RadialNet`).

Three hidden layers ``s_i = silu(s_{i-1} A_i)`` of a per-edge radial network run in one kernel per tile of
edges, the intermediate activations in registers; the backward kernel recomputes them and returns the input
gradient (forces and stress). When the pass also wants the weights' gradients, or the backward is itself
differentiated (``create_graph``), the backward is the same formula in PyTorch operations.
Products are float32 without TF32. Imported only when the fast path runs.
"""
# no postponed annotations: Triton's interpreter (TRITON_INTERPRET=1) recognizes the
# tl.constexpr parameters of a kernel only from evaluated annotations
from typing import List, Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor

#: Edges per tile.
BLOCK = 64
#: Warps per program.
WARPS = 8


@triton.jit
def _weight(a_ptr, K: tl.constexpr, H: tl.constexpr, KP: tl.constexpr, HP: tl.constexpr):
    k = tl.arange(0, KP)
    h = tl.arange(0, HP)
    return tl.load(a_ptr + k[:, None] * H + h[None, :], mask=(k[:, None] < K) & (h[None, :] < H), other=0.0)


@triton.jit
def _silu(z):
    return z * tl.sigmoid(z)


@triton.jit
def _dsilu(z):
    s = tl.sigmoid(z)
    return s * (1.0 + z * (1.0 - s))


@triton.jit
def _forward_kernel(x_ptr, a1_ptr, a2_ptr, a3_ptr, out_ptr, n_rows,
                    K0: tl.constexpr, H1: tl.constexpr, H2: tl.constexpr, H3: tl.constexpr,
                    K0P: tl.constexpr, H1P: tl.constexpr, H2P: tl.constexpr, H3P: tl.constexpr,
                    BLOCK: tl.constexpr):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rmask = rows < n_rows
    k0 = tl.arange(0, K0P)
    x = tl.load(x_ptr + rows[:, None] * K0 + k0[None, :], mask=rmask[:, None] & (k0[None, :] < K0), other=0.0)
    s = _silu(tl.dot(x, _weight(a1_ptr, K0, H1, K0P, H1P), input_precision="ieee"))
    s = _silu(tl.dot(s, _weight(a2_ptr, H1, H2, H1P, H2P), input_precision="ieee"))
    s = _silu(tl.dot(s, _weight(a3_ptr, H2, H3, H2P, H3P), input_precision="ieee"))
    h = tl.arange(0, H3P)
    tl.store(out_ptr + rows[:, None] * H3 + h[None, :], s, mask=rmask[:, None] & (h[None, :] < H3))


@triton.jit
def _backward_kernel(x_ptr, g_ptr, a1_ptr, a2_ptr, a3_ptr, t1_ptr, t2_ptr, t3_ptr, gx_ptr, n_rows,
                     K0: tl.constexpr, H1: tl.constexpr, H2: tl.constexpr, H3: tl.constexpr,
                     K0P: tl.constexpr, H1P: tl.constexpr, H2P: tl.constexpr, H3P: tl.constexpr,
                     BLOCK: tl.constexpr):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rmask = rows < n_rows
    k0 = tl.arange(0, K0P)
    j3 = tl.arange(0, H3P)
    x = tl.load(x_ptr + rows[:, None] * K0 + k0[None, :], mask=rmask[:, None] & (k0[None, :] < K0), other=0.0)
    z1 = tl.dot(x, _weight(a1_ptr, K0, H1, K0P, H1P), input_precision="ieee")
    z2 = tl.dot(_silu(z1), _weight(a2_ptr, H1, H2, H1P, H2P), input_precision="ieee")
    z3 = tl.dot(_silu(z2), _weight(a3_ptr, H2, H3, H2P, H3P), input_precision="ieee")
    g = tl.load(g_ptr + rows[:, None] * H3 + j3[None, :], mask=rmask[:, None] & (j3[None, :] < H3), other=0.0)
    # the transposed matrices come from memory: transposing loaded tiles costs more than the products
    g = tl.dot(g * _dsilu(z3), _weight(t3_ptr, H3, H2, H3P, H2P), input_precision="ieee") * _dsilu(z2)
    g = tl.dot(g, _weight(t2_ptr, H2, H1, H2P, H1P), input_precision="ieee") * _dsilu(z1)
    gx = tl.dot(g, _weight(t1_ptr, H1, K0, H1P, K0P), input_precision="ieee")
    tl.store(gx_ptr + rows[:, None] * K0 + k0[None, :], gx, mask=rmask[:, None] & (k0[None, :] < K0))


def _padded(n: int) -> int:
    return max(16, triton.next_power_of_2(n))


def _sizes(x: Tensor, a: Tuple[Tensor, ...]) -> Tuple[int, ...]:
    dims = (x.shape[1], a[0].shape[1], a[1].shape[1], a[2].shape[1])
    return dims + tuple(_padded(d) for d in dims)


def _dsilu_torch(z: Tensor) -> Tensor:
    s = torch.sigmoid(z)
    return s * (1.0 + z * (1.0 - s))


def _vjp_torch(x: Tensor, a: Tuple[Tensor, ...], g: Tensor) -> List[Tensor]:
    """The backward in PyTorch operations (differentiable again)."""
    z1 = x @ a[0]
    s1 = torch.nn.functional.silu(z1)
    z2 = s1 @ a[1]
    s2 = torch.nn.functional.silu(z2)
    g3 = g * _dsilu_torch(s2 @ a[2])
    g2 = (g3 @ a[2].t()) * _dsilu_torch(z2)
    g1 = (g2 @ a[1].t()) * _dsilu_torch(z1)
    return [g1 @ a[0].t(), x.t() @ g1, s1.t() @ g2, s2.t() @ g3]


class FusedHidden(torch.autograd.Function):
    """``silu(silu(silu(x A1) A2) A3)`` for ``x`` of shape ``(E, K)`` in float32."""

    @staticmethod
    def forward(ctx, x: Tensor, a1: Tensor, a2: Tensor, a3: Tensor) -> Tensor:
        x, a = x.contiguous(), (a1.contiguous(), a2.contiguous(), a3.contiguous())
        sizes = _sizes(x, a)
        out = x.new_empty(x.shape[0], sizes[3])
        if x.shape[0]:
            _forward_kernel[(triton.cdiv(x.shape[0], BLOCK),)](x, *a, out, x.shape[0], *sizes, BLOCK=BLOCK,
                                                                 num_warps=WARPS)
        ctx.save_for_backward(x, *a)
        return out

    @staticmethod
    def backward(ctx, g: Tensor):
        x, a1, a2, a3 = ctx.saved_tensors
        a = (a1, a2, a3)
        need = ctx.needs_input_grad
        if torch.is_grad_enabled() or any(_executes(t) for t, n in zip(a, need[1:]) if n):
            # differentiated again, or the weights' gradients are wanted: PyTorch operations
            grads = _vjp_torch(x, a, g)
            return tuple(gr if nd else None for gr, nd in zip(grads, need))
        gx = torch.empty_like(x)
        if x.shape[0]:
            t = tuple(w.t().contiguous() for w in a)
            _backward_kernel[(triton.cdiv(x.shape[0], BLOCK),)](x, g.contiguous(), *a, *t, gx, x.shape[0],
                                                               *_sizes(x, a), BLOCK=BLOCK, num_warps=WARPS)
        return gx if need[0] else None, None, None, None


def _executes(t: Tensor) -> bool:
    """Whether the running backward pass reaches the gradient of ``t`` (``True`` when unknown)."""
    fn = t.grad_fn
    if fn is None:
        return True
    try:
        return bool(torch._C._will_engine_execute_node(fn))
    except Exception:
        return True


def fused_hidden(x: Tensor, a1: Tensor, a2: Tensor, a3: Tensor) -> Tensor:
    """Three hidden SiLU layers with the (scaled) matrices ``a1``, ``a2``, ``a3``; see the module notes."""
    return FusedHidden.apply(x, a1, a2, a3)


__all__ = ["BLOCK", "WARPS", "FusedHidden", "fused_hidden"]
