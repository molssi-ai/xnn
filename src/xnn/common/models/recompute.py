"""Recomputation blocks whose memory stays bounded through a double backward.

:func:`torch.utils.checkpoint.checkpoint` drops a block's intermediates after
the forward pass and recomputes them in the backward pass. That bounds memory
for a single backward. Under ``create_graph=True`` -- which
:class:`~xnn.common.models.outputs.ForceStressOutput` uses in training mode so
that a force loss can be differentiated -- the recomputation is recorded and
the intermediates of every block are retained again until the second
backward; the same happens when the backward is wrapped in a checkpoint,
because the autograd engine evaluates the derivative formulas on its own
thread, outside the checkpoint's saved-tensor hooks. On a 1500-atom cell the
checkpointed three-body term needed 1.6 GB in eval mode and 47 GB in training
mode either way.

:func:`recompute` closes that gap with two explicit autograd functions:

* :class:`_RecomputeBlock` evaluates the block without recording and saves
  only its inputs. Its backward is the second function.
* :class:`_RecomputeGrad` computes the block's gradient in a *local* autograd
  graph that is freed on exit (forward), and, when a second backward asks for
  it, re-runs block and gradient once more with ``create_graph=True`` and
  contracts with the incoming cotangents, again locally (backward).

Memory is one block at every stage; the price is one extra evaluation of the
block per order of differentiation. Third derivatives are not supported.
Eager only: custom autograd functions do not script.
"""
from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch import Tensor


def _expand(grads: Sequence, needs: Sequence[bool]):
    """Place gradients of the needed inputs at their positions, ``None`` elsewhere."""
    out, k = [], 0
    for need in needs:
        if need:
            out.append(grads[k])
            k += 1
        else:
            out.append(None)
    return out


class _RecomputeGrad(torch.autograd.Function):
    """``d fn(*tensors) / d tensors`` contracted with ``grad_out``, recomputed on demand."""

    @staticmethod
    def forward(ctx, fn: Callable[..., Tensor], needs: Sequence[bool], grad_out: Tensor,
                *tensors: Tensor):
        ctx.fn, ctx.needs = fn, needs
        ctx.save_for_backward(grad_out, *tensors)
        with torch.enable_grad():
            live = [t.detach().requires_grad_(need) for t, need in zip(tensors, needs)]
            y = fn(*live)
            wrt = [t for t, need in zip(live, needs) if need]
            grads = torch.autograd.grad(y, wrt, grad_out.detach(), allow_unused=True)
        return tuple(torch.zeros_like(t) if g is None else g.detach() for g, t in zip(grads, wrt))

    @staticmethod
    def backward(ctx, *cotangents: Tensor):
        grad_out, *tensors = ctx.saved_tensors
        fn, needs = ctx.fn, ctx.needs
        with torch.enable_grad():
            live = [t.detach().requires_grad_(need) for t, need in zip(tensors, needs)]
            g_live = grad_out.detach().requires_grad_(True)
            y = fn(*live)
            wrt = [t for t, need in zip(live, needs) if need]
            grads = torch.autograd.grad(y, wrt, g_live, create_graph=True, allow_unused=True)
            total = sum((g * c).sum() for g, c in zip(grads, cotangents) if g is not None)
            if not torch.is_tensor(total) or not total.requires_grad:
                second = [None] * (1 + len(wrt))
            else:
                second = torch.autograd.grad(total, [g_live] + wrt, allow_unused=True)
        g_grad_out = second[0] if second[0] is not None else torch.zeros_like(grad_out)
        g_inputs = [torch.zeros_like(t) if g is None else g for g, t in zip(second[1:], wrt)]
        return (None, None, g_grad_out, *_expand(g_inputs, needs))


class _RecomputeBlock(torch.autograd.Function):
    """``fn(*tensors)`` evaluated without recording; gradients via :class:`_RecomputeGrad`."""

    @staticmethod
    def forward(ctx, fn: Callable[..., Tensor], *tensors: Tensor) -> Tensor:
        ctx.fn = fn
        ctx.needs = [bool(t.requires_grad) for t in tensors]
        ctx.save_for_backward(*tensors)
        with torch.no_grad():
            return fn(*tensors)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        grads = _RecomputeGrad.apply(ctx.fn, ctx.needs, grad_out, *ctx.saved_tensors)
        return (None, *_expand(grads, ctx.needs))


def recompute(fn: Callable[..., Tensor], *tensors: Tensor) -> Tensor:
    """Evaluate ``fn(*tensors)`` as a block that is recomputed rather than stored.

    Parameters
    ----------
    fn : callable
        A function of tensors only (bind other arguments with a closure or
        :func:`functools.partial`) returning one tensor.
    *tensors : Tensor
        Its inputs; gradients flow to those with ``requires_grad``.

    Returns
    -------
    Tensor
        ``fn(*tensors)``, with first and second derivatives evaluated by
        re-running the block, so the retained memory is its inputs only.
    """
    if not torch.is_grad_enabled() or not any(t.requires_grad for t in tensors):
        with torch.no_grad():
            return fn(*tensors)
    return _RecomputeBlock.apply(fn, *tensors)
