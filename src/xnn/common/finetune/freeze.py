"""Freezing parameters by name pattern (layer-freezing fine-tuning)."""
from __future__ import annotations

import fnmatch
from typing import Sequence

from torch import nn


def freeze_parameters(model: nn.Module, freeze: Sequence[str] = (),
                      train_only: Sequence[str] = ()) -> tuple[int, int]:
    """Set ``requires_grad`` of the parameters of ``model`` by name pattern.

    Parameters
    ----------
    model : torch.nn.Module
        The model.
    freeze : sequence of str, optional
        ``fnmatch`` patterns of parameter names to freeze
        (``["model.interactions.0.*", "*node_embedding*"]``).
    train_only : sequence of str, optional
        Patterns of the parameters to keep trainable; every other parameter
        is frozen (``["*readouts*", "*atom_ref*"]`` trains the readout only).
        ``freeze`` is applied afterwards.

    Returns
    -------
    tuple of int
        ``(trainable, total)`` parameter counts after the change.
    """
    for name, p in model.named_parameters():
        if train_only:
            p.requires_grad = any(fnmatch.fnmatchcase(name, pat) for pat in train_only)
        if any(fnmatch.fnmatchcase(name, pat) for pat in freeze):
            p.requires_grad = False
    return count_parameters(model, trainable_only=True), count_parameters(model)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Number of parameters of ``model`` (optionally only the trainable ones)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
