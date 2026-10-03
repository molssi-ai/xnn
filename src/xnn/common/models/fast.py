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

import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Union

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
    """

    min_edges: Dict[str, int] = field(default_factory=dict)
    min_edges_float64: Dict[str, int] = field(default_factory=dict)
    default_min_edges: int = 0
    default_min_edges_float64: int = 0

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
        return device.type == "cuda" and num_edges >= self.threshold(device, dtype)


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
