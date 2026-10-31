"""Multilinear blocks: autograd functions that keep only their operands.

A block ``f(op_0, ..., op_k)`` linear in each operand (a sum of terms, each
linear in the operands it contains) is evaluated without recording, and its
derivatives come in closed form from the block itself:

* the adjoint ``bar f_i(g)`` of each operand (the gradient for the incoming
  gradient ``g``), and
* their own derivatives (the second backward of force training). For each
  term, ``<c_i, bar f_i(g)> = <g, f(op_i <- c_i)>``, so the derivatives of the
  adjoints are values and adjoints of the same block with one operand replaced
  by its cotangent, over the terms that contain it.

Nothing but the operands (and, in a second backward, the incoming gradient) is
kept. The blocks can be pair sums written as fused kernels, whose adjoints are
kernels of the same form. Third derivatives are not supported. Eager only:
custom autograd functions do not script.
"""
from typing import List, Optional, Tuple

import torch
from torch import Tensor


def add_optional(a: Optional[Tensor], b: Tensor) -> Tensor:
    """``a + b``, or ``b`` when ``a`` is ``None``."""
    return b if a is None else a + b


class MultilinearBlock:
    """Interface of a block for :func:`multilinear`.

    ``value(ops, only)`` evaluates the block; ``adjoints(g, ops, needs, only)``
    returns one gradient per operand (``None`` where ``needs`` is false). With
    ``only = i`` both run over the terms that contain operand ``i`` (all terms
    for ``only = -1``). :meth:`second` derives the derivatives of the adjoints
    from them; a block may override it to share work.
    """

    def value(self, ops: List[Tensor], only: int = -1) -> Tensor:
        raise NotImplementedError

    def adjoints(self, g: Tensor, ops: List[Tensor], needs: List[bool],
                 only: int = -1) -> List[Optional[Tensor]]:
        raise NotImplementedError

    def second(self, g: Tensor, ops: List[Tensor], cots: List[Optional[Tensor]], want_g: bool,
               wants: List[bool]) -> Tuple[Optional[Tensor], List[Optional[Tensor]]]:
        """The gradients of ``<cots, adjoints(g, ops)>`` for ``g`` and the operands."""
        g_out: Optional[Tensor] = None
        g_ops: List[Optional[Tensor]] = [None] * len(ops)
        for i, c in enumerate(cots):
            if c is None:
                continue
            swapped = ops[:i] + [c] + ops[i + 1:]
            if want_g:
                g_out = add_optional(g_out, self.value(swapped, i))
            needs = [want and j != i for j, want in enumerate(wants)]
            if any(needs):
                for j, t in enumerate(self.adjoints(g, swapped, needs, i)):
                    if t is not None:
                        g_ops[j] = add_optional(g_ops[j], t)
        return g_out, g_ops


class _MultilinearGrad(torch.autograd.Function):
    """The adjoints of a block, differentiable once more through :meth:`MultilinearBlock.second`."""

    @staticmethod
    def forward(ctx, block, needs: List[bool], grad_out: Tensor, *ops: Tensor):
        ctx.block, ctx.needs = block, needs
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(grad_out, *ops)
        return tuple(a for a in block.adjoints(grad_out, list(ops), needs) if a is not None)

    @staticmethod
    def backward(ctx, *cotangents: Optional[Tensor]):
        grad_out, *ops = ctx.saved_tensors
        given = iter(cotangents)
        cots = [next(given) if need else None for need in ctx.needs]
        g_out, g_ops = ctx.block.second(grad_out, ops, cots, ctx.needs_input_grad[2],
                                        list(ctx.needs_input_grad[3:]))
        return (None, None, g_out, *g_ops)


class _Multilinear(torch.autograd.Function):
    """A block evaluated without recording; it saves its operands only."""

    @staticmethod
    def forward(ctx, block, *ops: Tensor) -> Tensor:
        ctx.block = block
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(*ops)
        return block.value(list(ops))

    @staticmethod
    def backward(ctx, grad_out: Optional[Tensor]):
        needs = list(ctx.needs_input_grad[1:])
        if grad_out is None:
            return (None,) * (1 + len(needs))
        grads = iter(_MultilinearGrad.apply(ctx.block, needs, grad_out.contiguous(), *ctx.saved_tensors))
        return (None, *[next(grads) if need else None for need in needs])


def multilinear(block: MultilinearBlock, *ops: Tensor) -> Tensor:
    """``block.value(ops)``, with first and second derivatives from the block.

    Parameters
    ----------
    block : MultilinearBlock
        The block (its index tensors and sizes bound in the object).
    *ops : Tensor
        Its operands; gradients reach those that require them.

    Returns
    -------
    Tensor
        The block's value.
    """
    return _Multilinear.apply(block, *ops)
