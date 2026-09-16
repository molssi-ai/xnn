"""Smooth cutoff functions shared across families."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class CosineCutoff(nn.Module):
    """Behler cosine cutoff function.

    Computes ``0.5 * (cos(pi * r / rc) + 1)`` for distances inside the cutoff
    and returns zero beyond it, giving a smooth decay to zero at ``r = rc``.

    Parameters
    ----------
    cutoff : float
        Cutoff radius ``rc``, in the same units as the input distances.
    """

    def __init__(self, cutoff: float):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, r: Tensor) -> Tensor:
        """Evaluate the cosine cutoff.

        Parameters
        ----------
        r : Tensor
            Interatomic distances of arbitrary shape ``(...)``.

        Returns
        -------
        Tensor
            Cutoff weights of the same shape as ``r``, in ``[0, 1]`` and zero
            where ``r >= cutoff``.
        """
        out = 0.5 * (torch.cos(math.pi * r / self.cutoff) + 1.0)
        return out * (r < self.cutoff)
