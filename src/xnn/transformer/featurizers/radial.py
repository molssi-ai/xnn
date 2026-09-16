"""Exponential-normal radial basis (TorchMD-Net / TensorNet / BAMBOO).

The exponential-normal smearing expands an interatomic distance onto a set of
Gaussians placed in *exponential* distance space rather than linearly, so the
basis is dense at short range (where the physics is stiff) and sparse near the
cutoff. It is the radial basis shared by the graph-transformer potentials
(TorchMD-Net ``ExpNormalSmearing``, TensorNet, and BAMBOO's GET). The whole
basis is smoothly damped to zero at the cutoff by the shared Behler
:class:`~xnn.common.featurizers.CosineCutoff`.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from xnn.common.featurizers import CosineCutoff


class ExpNormalSmearing(nn.Module):
    r"""Exponential-normal radial basis expansion (TorchMD-Net-style).

    Expands a scalar distance ``r`` onto ``n_rbf`` Gaussians in exponential
    distance space, multiplied by a cosine cutoff envelope::

        phi_k(r) = cos_cutoff(r) *
                   exp(-beta_k * (exp(alpha * (cutoff_lower - r)) - mu_k) ** 2)

    with ``alpha = 5 / (cutoff_upper - cutoff_lower)``. The centers ``mu_k`` are
    spaced linearly from ``exp(cutoff_lower - cutoff_upper)`` to ``1`` and the
    widths ``beta_k`` are initialised to the TorchMD-Net/PhysNet default
    ``(2 / n_rbf * (1 - exp(cutoff_lower - cutoff_upper))) ** -2``.

    Parameters
    ----------
    n_rbf : int, optional
        Number of basis functions (output width). Default is 32.
    cutoff : float, optional
        Upper cutoff ``cutoff_upper`` in the same units as the distances.
        Default is 5.0.
    cutoff_lower : float, optional
        Lower cutoff, by default 0.0.
    trainable : bool, optional
        If ``True`` (the default) the centers ``means`` and widths ``betas``
        are learnable :class:`torch.nn.Parameter`; otherwise fixed buffers.

    Attributes
    ----------
    means : Tensor
        The Gaussian centers ``mu_k`` of shape ``(n_rbf,)`` (parameter or
        buffer depending on ``trainable``).
    betas : Tensor
        The Gaussian inverse-widths ``beta_k`` of shape ``(n_rbf,)``.
    alpha : float
        The exponential-space scale ``5 / (cutoff_upper - cutoff_lower)``.
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
        """Fill ``means`` and ``betas`` with their default values in place.

        The defaults follow the TorchMD-Net/PhysNet convention: in exponential
        distance space the centers run linearly from
        ``exp(cutoff_lower - cutoff_upper)`` (the image of the upper cutoff) up
        to ``1`` (the image of the lower cutoff), and every Gaussian starts
        with the same width, matched to the center spacing.
        """
        first_center = math.exp(self.cutoff_lower - self.cutoff_upper)
        shared_width = (2.0 / self.n_rbf * (1.0 - first_center)) ** -2
        with torch.no_grad():
            self.means.copy_(torch.linspace(first_center, 1.0, self.n_rbf))
            self.betas.fill_(shared_width)

    def forward(self, r: Tensor) -> Tensor:
        """Expand distances onto the exponential-normal basis.

        Parameters
        ----------
        r : Tensor
            Interatomic distances of arbitrary shape ``(...)``.

        Returns
        -------
        Tensor
            The radial embedding of shape ``(..., n_rbf)``, smoothly zero at
            and beyond the cutoff.
        """
        r = r.unsqueeze(-1)
        # image of the distance in exponential space, where the centers live
        u = torch.exp(self.alpha * (self.cutoff_lower - r))
        gaussians = torch.exp(-self.betas * (u - self.means) ** 2)
        return self.cutoff_fn(r) * gaussians
