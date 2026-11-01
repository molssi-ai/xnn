"""Transformer blocks compiled into fused kernels (the fast paths of Equiformer and EquiformerV2).

A block is a module whose ``_reference(*args)`` is its reference forward. On the
compiled path, in evaluation mode, :func:`torch.compile` runs ``_reference`` (with
a compiled first derivative for the forces). In training mode the block runs as a
function of its weights and floating-point inputs whose first and second
derivatives are compiled as well
(:func:`~xnn.common.models.fast.compile_twice_differentiable`), with the integer
inputs (indices) as constants, so a force loss trains through it. A block with
dropout that acts (training mode) runs the reference.
"""
import sys

import torch
from torch import Tensor

from xnn.common.models.fast import compile_twice_differentiable


def supported(device: torch.device, dtype: torch.dtype) -> bool:
    """A CUDA device and a Python that :func:`torch.compile` supports."""
    return device.type == "cuda" and sys.version_info < (3, 13) and dtype in (torch.float32, torch.float64)


def _stochastic(block: torch.nn.Module) -> bool:
    """Whether the block has dropout that acts in training mode."""
    for m in block.modules():
        for name in ("p", "drop_prob"):
            value = getattr(m, name, 0.0)
            if isinstance(value, float) and value > 0.0:
                return True
    return False


def _kind(a) -> str:
    if isinstance(a, Tensor):
        return "d" if a.is_floating_point() else "c"
    return "v"


def _structure(block: torch.nn.Module) -> tuple:
    """Blocks with the same structure (class, layers and their settings, weight and buffer shapes)
    share their compiled code."""
    return (type(block), str(block), tuple((n, tuple(t.shape), t.dtype, t.device) for n, t in block.named_parameters()),
            tuple((n, tuple(t.shape), t.dtype, t.device) for n, t in block.named_buffers()))


#: compiled functions shared by blocks of one structure
_SHARED: dict = {}


def _states(block: torch.nn.Module):
    params = [p for _, p in block.named_parameters()]
    buffers = [b for _, b in block.named_buffers()]
    return params, buffers


def _functional(template: torch.nn.Module, n_args: int, kinds, values):
    """``template._reference`` as a function of (weights, buffers, differentiated inputs, constants)."""
    from torch.nn.utils.stateless import _reparametrize_module
    names = [n for n, _ in template.named_parameters()] + [n for n, _ in template.named_buffers()]
    k_params = sum(1 for _ in template.named_parameters())
    k_buffers = len(names) - k_params
    nd = kinds.count("d")

    def fn(*xs):
        # weights, differentiated inputs, then buffers and constant inputs
        params = list(xs[:k_params])
        diff = iter(xs[k_params:k_params + nd])
        rest = xs[k_params + nd:]
        buffers, const = list(rest[:k_buffers]), iter(rest[k_buffers:])
        full = [next(diff) if kind == "d" else next(const) if kind == "c" else value
                for kind, value in zip(kinds, values)]
        # the reference forward under these weights (functional_call would enter forward, which
        # takes the compiled path again when the derivatives are traced, in the backward pass)
        with _reparametrize_module(template, dict(zip(names, params + buffers))):
            return template._reference(*full)

    fn.__name__ = f"{type(template).__name__}_{''.join(kinds)}"
    return fn, k_params + nd, k_buffers + kinds.count("c")


def run_compiled(block: torch.nn.Module, *args):
    """``block._reference(*args)`` on the compiled path (the reference where dropout acts).

    Python values among ``args`` (sizes, ``None``) are compiled in as constants. Blocks of one
    structure share one compiled function (their weights are its inputs).
    """
    if block.training and _stochastic(block):
        return block._reference(*args)
    kinds = tuple(_kind(a) for a in args)
    values = tuple(a if k == "v" else None for k, a in zip(kinds, args))
    key = (_structure(block), block.training, kinds, tuple(v for v in values if v is not None))
    fn = _SHARED.get(key)
    if fn is None:
        raw, n_diff, n_const = _functional(block, len(args), list(kinds), list(values))
        fn = _SHARED[key] = (compile_twice_differentiable(raw, n_diff, n_const) if block.training
                             else torch.compile(raw, dynamic=True))
    params, buffers = _states(block)
    # contiguous inputs: a transposed view as an input of the twice-differentiated graph trips
    # the compiler (symbolic strides)
    diff = [a.contiguous() for a in args if _kind(a) == "d"]
    const = [a for a in args if _kind(a) == "c"]
    return fn(*params, *diff, *buffers, *const)
