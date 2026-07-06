"""Bessel radial basis for the equivariant GNN edge embedding (NequIP/MACE/DimeNet-style)."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class BesselRBF(nn.Module):
    """Bessel radial basis (NequIP/DimeNet-style), smoother & fewer functions.

    Expands an interatomic distance ``r`` into a set of ``n_rbf`` invariant
    radial features using the normalized sinc/Bessel functions
    ``prefactor * sin(n*pi*r/rc) / r`` for ``n = 1, ..., n_rbf``. Compared with
    a Gaussian basis this is smoother and needs fewer functions to cover the
    cutoff sphere.

    Parameters
    ----------
    n_rbf : int, optional
        Number of Bessel basis functions (output width). Default is 8.
    cutoff : float, optional
        Cutoff radius ``rc`` used to set the basis frequencies and the
        normalization. Default is 5.0.
    trainable : bool, optional
        If ``True``, the ``n * pi`` frequencies are a learnable
        :class:`torch.nn.Parameter` (the NequIP ``BesselBasis`` default);
        otherwise a fixed buffer (the MACE default). Default is ``False``.
    prefactor : float, optional
        Overall normalization factor. ``None`` (default) uses the
        DimeNet/MACE convention ``sqrt(2 / cutoff)``; the original NequIP
        uses ``2 / cutoff``.

    Attributes
    ----------
    freqs : Tensor
        The angular frequencies ``n * pi`` of shape ``(n_rbf,)`` -- a
        registered buffer, or a :class:`torch.nn.Parameter` when
        ``trainable``.
    norm : float
        The scalar normalization ``prefactor``.
    """

    def __init__(self, n_rbf: int = 8, cutoff: float = 5.0,
                 trainable: bool = False, prefactor: float | None = None):
        super().__init__()
        self.cutoff = cutoff
        freqs = math.pi * torch.arange(1, n_rbf + 1, dtype=torch.get_default_dtype())
        if trainable:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs)
        self.norm = math.sqrt(2.0 / cutoff) if prefactor is None else float(prefactor)

    def forward(self, r: Tensor) -> Tensor:
        """Expand distances into the Bessel radial basis.

        Parameters
        ----------
        r : Tensor
            Interatomic distances of arbitrary shape ``(...)``. Values are
            clamped to a small positive minimum to avoid division by zero.

        Returns
        -------
        Tensor
            The radial embedding of shape ``(..., n_rbf)``.
        """
        r = r.clamp(min=1e-8)[..., None]
        return self.norm * torch.sin(self.freqs * r / self.cutoff) / r
