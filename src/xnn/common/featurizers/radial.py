"""Radial basis expansions shared across families: the Gaussian basis of SchNet and the
equivariant GNN edge embedding, and the exponential-normal basis of the TorchMD-Net models
(ET, TensorNet) and BAMBOO."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .cutoff import CosineCutoff


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


class ExpNormalSmearing(nn.Module):
    r"""Exponential-normal radial basis (TorchMD-Net ET / TensorNet, BAMBOO).

    Expands a distance ``r`` onto ``n_rbf`` Gaussians placed in exponential
    distance space, dense at short range and sparse near the cutoff, times a
    cosine cutoff envelope::

        phi_k(r) = cos_cutoff(r) *
                   exp(-beta_k * (exp(alpha * (cutoff_lower - r)) - mu_k) ** 2)

    with ``alpha = 5 / (cutoff - cutoff_lower)``. The centers ``mu_k`` run
    linearly from ``exp(cutoff_lower - cutoff)`` to ``1`` and every width
    starts at ``(2 / n_rbf * (1 - exp(cutoff_lower - cutoff))) ** -2``
    (Thoelke & De Fabritiis 2022, eq 1, the PhysNet convention).

    Parameters
    ----------
    n_rbf : int, optional
        Number of basis functions, by default 32.
    cutoff : float, optional
        Upper cutoff in the units of the distances, by default 5.0.
    cutoff_lower : float, optional
        Lower cutoff, by default 0.0.
    trainable : bool, optional
        Learn the centers and widths (``True``, the default) or keep them as
        buffers.

    Attributes
    ----------
    means : Tensor
        The centers ``mu_k``, shape ``(n_rbf,)``.
    betas : Tensor
        The inverse widths ``beta_k``, shape ``(n_rbf,)``.
    alpha : float
        The exponential-space scale ``5 / (cutoff - cutoff_lower)``.
    """

    def __init__(self, n_rbf: int = 32, cutoff: float = 5.0,
                 cutoff_lower: float = 0.0, trainable: bool = True):
        super().__init__()
        self.n_rbf = n_rbf
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff
        self.trainable = trainable
        self.cutoff_fn = CosineCutoff(cutoff)
        self.alpha = 5.0 / (self.cutoff_upper - self.cutoff_lower)

        if trainable:
            self.means = nn.Parameter(torch.empty(n_rbf))
            self.betas = nn.Parameter(torch.empty(n_rbf))
        else:
            self.register_buffer("means", torch.empty(n_rbf))
            self.register_buffer("betas", torch.empty(n_rbf))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the centers and widths to their default values in place."""
        first_center = math.exp(self.cutoff_lower - self.cutoff_upper)
        shared_width = (2.0 / self.n_rbf * (1.0 - first_center)) ** -2
        with torch.no_grad():
            self.means.copy_(torch.linspace(first_center, 1.0, self.n_rbf))
            self.betas.fill_(shared_width)

    def forward(self, r: Tensor) -> Tensor:
        """Expand distances onto the basis.

        Parameters
        ----------
        r : Tensor
            Interatomic distances of arbitrary shape ``(...)``.

        Returns
        -------
        Tensor
            The expansion of shape ``(..., n_rbf)``, zero at and beyond the
            cutoff.
        """
        r = r.unsqueeze(-1)
        # the distance mapped to exponential space, where the centers live
        u = torch.exp(self.alpha * (self.cutoff_lower - r))
        return self.cutoff_fn(r) * torch.exp(-self.betas * (u - self.means) ** 2)
