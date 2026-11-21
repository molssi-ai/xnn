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


_CAPTURE = {"depth": 0}


@contextlib.contextmanager
def graph_capture(graph: "torch.cuda.CUDAGraph", **kwargs) -> Iterator[None]:
    """``torch.cuda.graph(graph, **kwargs)`` (``pool``, ``stream``, ``capture_error_mode``) with the cuBLAS
    workspaces cleared before and after the recording.

    cuBLAS keeps a workspace per handle and stream, made at its first use and kept. Cleared, a workspace that
    the recording needs is made in the graph's memory pool and lives as long as the graph. A workspace kept from
    elsewhere (from launches, or from another graph's pool) could be freed while the graph still wrote into it:
    every recording starts by releasing the allocator's free memory, so a later recording or launch could reuse
    it. A workspace made in the pool must not serve launches outside the graph either.
    """
    torch._C._cuda_clearCublasWorkspaces()
    try:
        with torch.cuda.graph(graph, **kwargs):
            yield
    finally:
        torch._C._cuda_clearCublasWorkspaces()


@contextlib.contextmanager
def capture_preparation() -> Iterator[None]:
    """A context in which an evaluation is warmed up for or recorded into a CUDA graph.

    Inside it models leave out steps that read a result on the host to choose their
    work (a recording cannot read one, and its warmups must run the code it records),
    such as MACE's pairing of the two directions of its bonds.
    """
    _CAPTURE["depth"] += 1
    try:
        yield
    finally:
        _CAPTURE["depth"] -= 1


def preparing_capture() -> bool:
    """Whether :func:`capture_preparation` is active."""
    return _CAPTURE["depth"] > 0


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


def _dense(grads: tuple) -> tuple:
    """Incoming gradients as dense tensors of their own. Autograd hands on views, such as the zero-stride
    expansion of a sum's gradient or a slice of a concatenation's, and the compiled derivative stages are
    specialized to their inputs' strides and storage offsets (every new offset a recompilation); the GPU code
    inductor (torch 2.11) generated for such an expanded gradient of a batch gave wrong derivatives."""
    return tuple(g if not isinstance(g, torch.Tensor) or (g.is_contiguous() and g.storage_offset() == 0)
                 else g.clone(memory_format=torch.contiguous_format) for g in grads)


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
    forward, vjp_function = _vjp_parts(fn, n_inputs, n_constants, dynamic, options)

    class Stage(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *xs):
            ctx.save_for_backward(*xs[:n_inputs])
            ctx.consts = xs[n_inputs:]
            with separate_shapes():
                return forward(*xs)

        @staticmethod
        def backward(ctx, *grads):
            g = vjp_function.apply(*ctx.saved_tensors, *ctx.consts, *_dense(grads))
            return (*g, *([None] * n_constants))

    def apply(*xs):
        out = Stage.apply(*xs)
        return out[0] if len(out) == 1 else out

    return apply


def _vjp_parts(fn: Callable[..., Any], n_inputs: int, n_constants: int, dynamic: bool,
               options: Optional[dict]) -> tuple:
    """The compiled ``fn`` (returning a tuple) and an autograd function of its vector-Jacobian product.

    The autograd function maps ``(inputs, constants, one cotangent per output)`` to one gradient per
    input. It differentiates ``fn`` at its own copies of the inputs, so no path of a graph around them
    enters, and its backward (the compiled product of second order) makes it differentiable once more.
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
                g_inputs, g_grads = second_c(*saved[:n_inputs], *ctx.consts, *saved[n_inputs:],
                                             *_dense(cotangents))
            return (*g_inputs, *([None] * n_constants), *g_grads)

    return forward, Grad


def _detached_copy(x: torch.Tensor) -> torch.Tensor:
    """``x`` detached as a leaf that needs a gradient, with the dynamic sizes and size hints given to the
    compiler for ``x`` (:func:`torch._dynamo.mark_dynamic`). The sizes of a parameter stay static, as the
    compiler keeps those of the parameters it is given (kernels specialized to the layer widths)."""
    y = x.detach().requires_grad_(True)
    if isinstance(x, torch.nn.Parameter) and not getattr(x, "_dynamo_dynamic_indices", None):
        torch._dynamo.mark_static(y)
        return y
    hints = getattr(x, "_dynamo_hint_overrides", {})
    for dim in getattr(x, "_dynamo_dynamic_indices", ()):
        if dim in hints:
            torch._dynamo.mark_dynamic(y, dim, hint_override=hints[dim])
        else:
            torch._dynamo.mark_dynamic(y, dim)
    return y


def compile_differentiable(fn: Callable[..., Any], n_inputs: int, n_constants: int = 0,
                           dynamic: bool = True, options: Optional[dict] = None) -> Callable[..., Any]:
    """``fn`` compiled by :func:`torch.compile`, with a fast first derivative and a second one on request.

    :func:`compile_twice_differentiable` recomputes ``fn`` inside every backward pass, the price of a
    backward that can itself be differentiated. Here the forward pass keeps the compiled graph of
    ``fn`` (as :func:`torch.compile` does), and a backward pass runs its compiled backward. A backward
    pass that builds a graph (``create_graph=True``), or a second backward pass through the same graph
    (``retain_graph=True``), takes the vector-Jacobian product of :func:`compile_twice_differentiable`,
    which agrees with the compiled backward to rounding; second derivatives stay available.

    Parameters
    ----------
    fn : callable
        A function of ``n_inputs`` tensors returning a tensor or a tuple of tensors, written in
        differentiable torch operations.
    n_inputs : int
        The number of its tensor arguments that are differentiated.
    n_constants : int, optional
        The number of further arguments, after the inputs, that are not differentiated. Default 0.
    dynamic : bool, optional
        Compile with symbolic sizes (default) or for the sizes of the first call.
    options : dict, optional
        Inductor options of the compiled functions.

    Returns
    -------
    callable
        ``fn`` with the same arguments and outputs.
    """
    once = torch.compile(fn, dynamic=dynamic, options=options)
    _, vjp_function = _vjp_parts(fn, n_inputs, n_constants, dynamic, options)

    class Stage(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *xs):
            ins = [_detached_copy(x) if x.requires_grad else x for x in xs[:n_inputs]]
            # traced with shared size symbols: without them inductor reduces the rows of the backward slowly
            with torch.enable_grad():
                out = once(*ins, *xs[n_inputs:])
            outs = tuple(out) if isinstance(out, (tuple, list)) else (out,)
            ctx.save_for_backward(*xs[:n_inputs])
            ctx.consts = xs[n_inputs:]
            ctx.ins, ctx.outs = ins, outs
            ctx.wanted = [i for i, x in enumerate(ins) if x.requires_grad]
            return tuple(o.detach() for o in outs)

        @staticmethod
        def backward(ctx, *grads):
            grads = _dense(grads)
            if torch.is_grad_enabled() or ctx.outs is None:
                # a backward that is itself differentiated, or a second one (the compiled backward runs once and
                # gives its buffers to its own computation): the vector-Jacobian product, the same to rounding
                g = vjp_function.apply(*ctx.saved_tensors, *ctx.consts, *grads)
                return (*g, *([None] * n_constants))
            got = torch.autograd.grad(ctx.outs, [ctx.ins[i] for i in ctx.wanted], grads, allow_unused=True)
            ctx.ins = ctx.outs = None
            result = [None] * (n_inputs + n_constants)
            for i, g in zip(ctx.wanted, got):
                result[i] = g
            return tuple(result)

    def apply(*xs):
        out = Stage.apply(*xs)
        return out[0] if len(out) == 1 else out

    return apply
