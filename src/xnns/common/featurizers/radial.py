"""Radial basis expansion shared across families (SchNet + equivariant GNN edge embedding)."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class GaussianRBF(nn.Module):
    """Gaussian radial basis expansion (SchNet-style).

    Expands scalar interatomic distances onto a set of ``n_rbf`` Gaussians whose
    centers are fixed and evenly spaced on ``[0, cutoff]`` and whose width is
    fixed to the center spacing. Registered as a non-trainable buffer.

    Parameters
    ----------
    n_rbf : int, optional
        Number of Gaussian basis functions (centers). Defaults to ``50``.
    cutoff : float, optional
        Upper bound of the center range, in the same units as the distances.
        Defaults to ``5.0``.

    Notes
    -----
    The width is the spacing between adjacent centers, or ``1.0`` when
    ``n_rbf == 1``.
    """

    def __init__(self, n_rbf: int = 50, cutoff: float = 5.0):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, n_rbf)
        self.register_buffer("centers", centers)
        self.width = (centers[1] - centers[0]).item() if n_rbf > 1 else 1.0

    def forward(self, r: Tensor) -> Tensor:
        """Expand distances onto the Gaussian basis.

        Parameters
        ----------
        r : Tensor
            Interatomic distances of arbitrary shape ``(...)``.

        Returns
        -------
        Tensor
            The radial basis expansion of shape ``(..., n_rbf)``.
        """
        diff = r[..., None] - self.centers
        return torch.exp(-0.5 * (diff / self.width) ** 2)
