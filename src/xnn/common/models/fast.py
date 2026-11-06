"""Fast paths: optimized kernels behind the same parameters as the reference code.

A module with a fast path keeps exactly the parameters, buffers and
``state_dict`` of its reference implementation and adds a second way of
evaluating them (for example a fused GPU kernel). Checkpoints, the model hub and
the trainer never see the difference, and the reference stays the canonical
implementation: TorchScript export, CPU runs and anything the fast backend does
not support use it.

Every such module subclasses :class:`FastPathModule`. A model owning them
chooses, once per evaluation, whether they run fast:

``use_fast=False``
    always the reference code.
``use_fast=True``
    the fast path wherever it is available (the reference otherwise).
``use_fast="auto"`` (the default)
    the fast path when it is available *and* measured to be faster for this
    input, i.e. on a GPU and above a size threshold per GPU model (see
    :class:`AutoPolicy`).

The fast kernels compute the same function with a different summation order,
so results agree with the reference to rounding: about ``1e-6`` relative in
float32 and ``1e-14`` in float64. A one-time :class:`FastPathPrecisionWarning`
says so when a fast path first runs in float32.
"""
from __future__ import annotations

import contextlib
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import torch
from torch import nn

UseFast = Union[bool, str]
MODES = ("auto", True, False)


class FastPathPrecisionWarning(UserWarning):
    """A fast path ran in float32: results differ from the reference at ~1e-6."""


_warned_float32 = False


def warn_float32_once() -> None:
    """Issue the float32 precision warning, once per process."""
    global _warned_float32
    if _warned_float32:
        return
    _warned_float32 = True
    warnings.warn(
        "A fast path (fused GPU kernels) is running in float32. It computes the same "
        "function as the reference implementation with a different summation order, so "
        "energies and forces differ from the reference at about 1e-6 relative (float64: "
        "about 1e-14). Pass use_fast=False for the reference implementation.",
        FastPathPrecisionWarning, stacklevel=3)


def resolve_use_fast(use_fast: UseFast) -> UseFast:
    """Validate a ``use_fast`` value (``True``, ``False`` or ``"auto"``)."""
    if isinstance(use_fast, str):
        if use_fast.lower() != "auto":
            raise ValueError(f"use_fast must be True, False or 'auto', got {use_fast!r}")
        return "auto"
    if isinstance(use_fast, bool):
        return use_fast
    raise TypeError(f"use_fast must be True, False or 'auto', got {type(use_fast).__name__}")


class FastPathModule:
    """Mixin of a module with a reference and a fast implementation.

    A fast-path module subclasses its reference module (so its parameters and
    ``state_dict`` are the reference ones) and this mixin. It implements the
    fast evaluation in methods marked ``@torch.jit.unused`` and branches on
    :attr:`fast_active`, which the owning model sets before each evaluation (see
    :func:`select`); its ``__init__`` sets ``self.fast_active = False``. Under
    TorchScript the flag is always ``False``.

    Attributes
    ----------
    fast_active : bool
        Whether the next evaluation uses the fast path.
    """

    def fast_supported(self, device: torch.device, dtype: torch.dtype) -> bool:
        """Whether the fast path can run for inputs on ``device`` in ``dtype``."""
        return False


def fast_modules(model: nn.Module) -> List[FastPathModule]:
    """Every :class:`FastPathModule` inside ``model``."""
    return [m for m in model.modules() if isinstance(m, FastPathModule)]


@dataclass
class AutoPolicy:
    """When ``use_fast="auto"`` takes the fast path.

    The fast kernels have a fixed per-call cost, so they lose on small inputs and
    win on large ones. The crossover is measured per GPU model and precision and
    stored as a minimum edge count of the neighbor graph; a GPU without a
    measured entry uses the default.

    Parameters
    ----------
    min_edges : dict of str to int
        float32 thresholds, keyed by a substring of the CUDA device name
        (matched case-insensitively, first match wins).
    min_edges_float64 : dict of str to int
        The same for float64 (where the reference is slower, so the kernels win
        earlier).
    default_min_edges, default_min_edges_float64 : int
        The thresholds of a GPU in neither table.
    max_edges, max_edges_float64 : dict of str to int
        Upper bounds, for fast paths that win only on small inputs (compiled
        blocks, which save kernel launches): the fast path is taken below them.
        A GPU in neither table has none unless the defaults set one.
    default_max_edges, default_max_edges_float64 : int, optional
        The upper bounds of a GPU in neither table (``None``: no bound).
    cpu_min_edges, cpu_min_edges_float64 : int, optional
        The thresholds on a CPU, for fast paths that have a CPU implementation
        (``None``, the default: never on a CPU).
    """

    min_edges: Dict[str, int] = field(default_factory=dict)
    min_edges_float64: Dict[str, int] = field(default_factory=dict)
    default_min_edges: int = 0
    default_min_edges_float64: int = 0
    max_edges: Dict[str, int] = field(default_factory=dict)
    max_edges_float64: Dict[str, int] = field(default_factory=dict)
    default_max_edges: Optional[int] = None
    default_max_edges_float64: Optional[int] = None
    cpu_min_edges: Optional[int] = None
    cpu_min_edges_float64: Optional[int] = None

    def threshold(self, device: torch.device, dtype: torch.dtype = torch.float32) -> int:
        """The minimum edge count for ``device`` and ``dtype``."""
        f64 = dtype == torch.float64
        table = self.min_edges_float64 if f64 else self.min_edges
        name = torch.cuda.get_device_name(device).lower()
        for key, value in table.items():
            if key.lower() in name:
                return int(value)
        return int(self.default_min_edges_float64 if f64 else self.default_min_edges)

    def decide(self, device: torch.device, dtype: torch.dtype, num_edges: int) -> bool:
        """Whether a graph with ``num_edges`` edges takes the fast path."""
        if device.type == "cpu":
            t = self.cpu_min_edges_float64 if dtype == torch.float64 else self.cpu_min_edges
            return t is not None and num_edges >= t
        if device.type != "cuda" or num_edges < self.threshold(device, dtype):
            return False
        bound = self.upper_bound(device, dtype)
        return bound is None or num_edges < bound

    def upper_bound(self, device: torch.device, dtype: torch.dtype = torch.float32) -> Optional[int]:
        """The edge count from which the fast path is no longer taken (``None``: no bound)."""
        f64 = dtype == torch.float64
        table = self.max_edges_float64 if f64 else self.max_edges
        name = torch.cuda.get_device_name(device).lower()
        for key, value in table.items():
            if key.lower() in name:
                return int(value)
        return self.default_max_edges_float64 if f64 else self.default_max_edges


def select(modules: List[FastPathModule], use_fast: UseFast, policy: Optional[AutoPolicy],
           device: torch.device, dtype: torch.dtype, num_edges: int) -> bool:
    """Set :attr:`FastPathModule.fast_active` on ``modules`` for one evaluation.

    Parameters
    ----------
    modules : list of FastPathModule
        The model's fast-path modules (from :func:`fast_modules`).
    use_fast : bool or str
        The model's setting (``True``, ``False`` or ``"auto"``).
    policy : AutoPolicy or None
        The ``"auto"`` policy; ``None`` treats ``"auto"`` as ``True``.
    device, dtype : torch.device, torch.dtype
        Where and in which precision the evaluation runs.
    num_edges : int
        The edge count of the input graph.

    Returns
    -------
    bool
        Whether any module runs its fast path.
    """
    want = bool(use_fast) if not isinstance(use_fast, str) else (
        policy is None or policy.decide(device, dtype, num_edges))
    any_fast = False
    for m in modules:
        m.fast_active = bool(want and m.fast_supported(device, dtype))
        any_fast = any_fast or m.fast_active
    if any_fast and dtype == torch.float32:
        warn_float32_once()
    return any_fast


def set_use_fast(model: nn.Module, use_fast: UseFast) -> nn.Module:
    """Set ``use_fast`` on every model inside ``model`` that has fast paths.

    ``model`` may be a wrapper (forces, dispersion, long-range): the setting
    reaches every potential inside it with a ``set_use_fast`` method.

    Returns
    -------
    torch.nn.Module
        ``model``.
    """
    mode = resolve_use_fast(use_fast)
    for m in model.modules():
        if callable(getattr(m, "set_use_fast", None)):
            m.set_use_fast(mode)
    return model


def deactivate(model: nn.Module) -> None:
    """Switch every fast path inside ``model`` off (before TorchScript export)."""
    for m in fast_modules(model):
        m.fast_active = False


def iter_fast_capable(model: nn.Module, device: torch.device,
                      dtype: torch.dtype) -> Iterator[FastPathModule]:
    """The fast-path modules of ``model`` that can run on ``device`` in ``dtype``."""
    for m in fast_modules(model):
        if m.fast_supported(device, dtype):
            yield m


def separate_shapes():
    """A context in which :func:`torch.compile` gives every dynamic size its own symbol.

    By default sizes that are equal when a graph is traced share one symbol ("duck
    shaping"); an edge count that happens to equal a feature width then ties the
    two, and tracing the second derivatives of such a graph can crash the process
    (torch 2.5). The compiled functions here are called inside this context, so
    their graphs are traced without it.
    """
    try:
        import torch.fx.experimental._config as fx_config
        return fx_config.patch(use_duck_shape=False)
    except (ImportError, AttributeError):
        return contextlib.nullcontext()


def compile_twice_differentiable(fn: Callable[..., Any], n_inputs: int,
                                 n_constants: int = 0, dynamic: bool = True,
                                 options: Optional[dict] = None) -> Callable[..., Any]:
    """``fn`` compiled by :func:`torch.compile` and differentiable twice.

    Compiled code has a first derivative but no second, which a force loss
    needs. Here ``fn`` runs inside an autograd function whose backward is the
    compiled vector-Jacobian product of ``fn`` (:func:`torch.func.vjp`), itself
    an autograd function whose backward is the compiled vector-Jacobian product
    of that. Each order is a few fused kernels, as in evaluation mode. Third
    derivatives are not supported.

    Parameters
    ----------
    fn : callable
        A function of ``n_inputs`` tensors returning a tensor or a tuple of
        tensors, written in differentiable torch operations.
    n_inputs : int
        The number of its tensor arguments that are differentiated.
    n_constants : int, optional
        The number of further arguments, after the inputs, that are not
        differentiated (index tensors, sizes). Default 0.
    dynamic : bool, optional
        Compile with symbolic sizes (default) or for the sizes of the first
        call (recompiled when they change).
    options : dict, optional
        Inductor options of the three compiled stages.

    Returns
    -------
    callable
        ``fn`` with the same arguments and outputs.
    """
    from torch.func import vjp

    m = n_inputs + n_constants

    def as_tuple(*xs):
        out = fn(*xs)
        return tuple(out) if isinstance(out, (tuple, list)) else (out,)

    def first(*args):
        # inputs, constants, then one incoming gradient per output
        consts = args[n_inputs:m]
        _, pull = vjp(lambda *xs: as_tuple(*xs, *consts), *args[:n_inputs])
        return pull(tuple(args[m:]))

    def second(*args):
        # the arguments of first (inputs, constants, incoming gradients), then one cotangent per input
        k = len(args) - n_inputs
        consts = args[n_inputs:m]

        def first_of(*xs):
            return first(*xs[:n_inputs], *consts, *xs[n_inputs:])

        _, pull = vjp(first_of, *args[:n_inputs], *args[m:k])
        out = pull(tuple(args[k:]))
        return out[:n_inputs], out[n_inputs:]

    # torch.compile keeps a few compiled versions per code object; the closures of every call
    # here share theirs, so each wrapped function gets its own copies (named after it)
    name = getattr(fn, "__name__", "fn")
    for f, kind in ((as_tuple, "forward"), (first, "vjp"), (second, "vjp_vjp")):
        f.__code__ = f.__code__.replace(co_name=f"{name}_{kind}")
    forward = torch.compile(as_tuple, dynamic=dynamic, options=options)
    first_c = torch.compile(first, dynamic=dynamic, options=options)
    second_c = torch.compile(second, dynamic=dynamic, options=options)

    class Grad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            ctx.save_for_backward(*args[:n_inputs], *args[m:])
            ctx.consts = args[n_inputs:m]
            with separate_shapes():
                return tuple(first_c(*args))

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, *cotangents):
            saved = ctx.saved_tensors
            with separate_shapes():
                g_inputs, g_grads = second_c(*saved[:n_inputs], *ctx.consts, *saved[n_inputs:], *cotangents)
            return (*g_inputs, *([None] * n_constants), *g_grads)

    class Stage(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *xs):
            ctx.save_for_backward(*xs[:n_inputs])
            ctx.consts = xs[n_inputs:]
            with separate_shapes():
                return forward(*xs)

        @staticmethod
        def backward(ctx, *grads):
            g = Grad.apply(*ctx.saved_tensors, *ctx.consts, *grads)
            return (*g, *([None] * n_constants))

    def apply(*xs):
        out = Stage.apply(*xs)
        return out[0] if len(out) == 1 else out

    return apply
