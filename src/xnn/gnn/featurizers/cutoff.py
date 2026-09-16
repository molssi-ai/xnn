"""Polynomial cutoff envelope used by the equivariant GNN edge embedding (NequIP/MACE)."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class PolynomialCutoff(nn.Module):
    """Smooth p-degree polynomial envelope (used by NequIP/MACE), C^2 at rc.

    Multiplicative envelope that decays smoothly from 1 at ``r = 0`` to exactly
    0 at the cutoff ``rc``, with continuous value and derivatives (``C^2``) at
    ``rc``. Applied to radial embeddings so that edge features vanish smoothly
    as neighbours leave the cutoff sphere.

    Parameters
    ----------
    cutoff : float
        Cutoff radius ``rc``; the envelope is zero for ``r >= cutoff``.
    p : int, optional
        Polynomial degree controlling the smoothness/sharpness of the decay.
        Default is 6.
    """

    def __init__(self, cutoff: float, p: int = 6):
        super().__init__()
        self.cutoff = cutoff
        self.p = p

    def forward(self, r: Tensor) -> Tensor:
        """Evaluate the cutoff envelope at the given distances.

        Parameters
        ----------
        r : Tensor
            Interatomic distances of arbitrary shape ``(...)``.

        Returns
        -------
        Tensor
            The envelope values, same shape as ``r``, decaying from 1 to 0 and
            exactly 0 for ``r >= cutoff``.
        """
        x = r / self.cutoff
        p = self.p
        env = (1.0
               - ((p + 1) * (p + 2) / 2) * x ** p
               + p * (p + 2) * x ** (p + 1)
               - (p * (p + 1) / 2) * x ** (p + 2))
        return env * (r < self.cutoff)
