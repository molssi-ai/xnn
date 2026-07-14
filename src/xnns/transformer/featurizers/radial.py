"""Exponential-normal radial basis (TorchMD-Net / TensorNet / BAMBOO).

The exponential-normal smearing expands an interatomic distance onto a set of
Gaussians placed in *exponential* distance space rather than linearly, so the
basis is dense at short range (where the physics is stiff) and sparse near the
cutoff. It is the radial basis shared by the graph-transformer potentials
(TorchMD-Net ``ExpNormalSmearing``, TensorNet, and BAMBOO's GET). The whole
basis is smoothly damped to zero at the cutoff by the shared Behler
:class:`~xnns.common.featurizers.CosineCutoff`.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from xnns.common.featurizers import CosineCutoff


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

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self) -> tuple[Tensor, Tensor]:
        """Initial centers and widths (TorchMD-Net/PhysNet convention).

        Returns
        -------
        tuple of Tensor
            The ``(means, betas)`` initial values, each of shape ``(n_rbf,)``.
        """
        start_value = math.exp(-self.cutoff_upper + self.cutoff_lower)
        means = torch.linspace(start_value, 1.0, self.n_rbf)
        betas = torch.full(
            (self.n_rbf,),
            (2.0 / self.n_rbf * (1.0 - start_value)) ** -2,
        )
        return means, betas

    def reset_parameters(self) -> None:
        """Reset ``means`` and ``betas`` to their initial values in place."""
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

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
        return self.cutoff_fn(r) * torch.exp(
            -self.betas
            * (torch.exp(self.alpha * (-r + self.cutoff_lower)) - self.means) ** 2
        )
