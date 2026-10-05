"""Radial basis expansion shared across families (SchNet + equivariant GNN edge embedding)."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class GaussianRBF(nn.Module):
    """Gaussian radial basis expansion (SchNet-style).

    Expands scalar interatomic distances onto a set of ``n_rbf`` Gaussians
    ``e_k(r) = exp(-gamma (r - mu_k)^2)`` whose centers ``mu_k`` are fixed and
    evenly spaced on ``[start, cutoff]``. Registered as a non-trainable buffer.

    Parameters
    ----------
    n_rbf : int, optional
        Number of Gaussian basis functions (centers). Defaults to ``50``.
    cutoff : float, optional
        Upper bound of the center range, in the same units as the distances.
        Defaults to ``5.0``.
    gamma : float or None, optional
        Width parameter of the Gaussians. ``None`` (default) sets the standard
        deviation to the spacing between adjacent centers, i.e.
        ``gamma = 0.5 / spacing**2``. SchNet (Schuett et al., NIPS 2017) fixes
        ``gamma = 10`` per Angstrom^2 on a 0.1 Angstrom center grid.
    start : float, optional
        Position of the first center, by default ``0.0``.
    endpoint : bool, optional
        Whether the last center sits at ``cutoff`` (default) or the grid stops
        one spacing short of it, ``mu_k = start + k (cutoff - start) / n_rbf``
        (the AIMNet2 layout, whose Gaussians all keep some weight inside the
        cutoff envelope).

    Notes
    -----
    With ``gamma=None`` the width is the spacing between adjacent centers, or
    ``1.0`` when ``n_rbf == 1``.
    """

    def __init__(self, n_rbf: int = 50, cutoff: float = 5.0,
                 gamma: float | None = None, start: float = 0.0,
                 endpoint: bool = True):
        super().__init__()
        if endpoint:
            centers = torch.linspace(start, cutoff, n_rbf)
        else:
            centers = torch.linspace(start, cutoff, n_rbf + 1)[:n_rbf]
        self.register_buffer("centers", centers)
        self.width = (centers[1] - centers[0]).item() if n_rbf > 1 else 1.0
        self.gamma = float(gamma) if gamma is not None \
            else 0.5 / self.width ** 2

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
        return torch.exp(-self.gamma * diff ** 2)
