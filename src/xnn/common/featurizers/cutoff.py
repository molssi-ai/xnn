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


class MollifierCutoff(nn.Module):
    """Bump-function envelope ``exp(1 - 1 / (1 - (r / rc)^2))``, smooth to every order.

    Equal to 1 at ``r = 0`` and to 0 for ``r >= rc`` with every derivative
    continuous at the cutoff. AIMNet2 (Anstine *et al.*, *Chem. Sci.* 2025)
    uses it to switch off the short-range part of the Coulomb energy that its
    network has learned implicitly; it is also the radial cutoff of SpookyNet
    (Unke *et al.*, *Nat. Commun.* 2021, eq 16).

    Parameters
    ----------
    cutoff : float
        Cutoff radius ``rc``; the envelope is zero for ``r >= cutoff``.
    """

    def __init__(self, cutoff: float):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, r: Tensor) -> Tensor:
        """Evaluate the envelope at the given distances.

        Parameters
        ----------
        r : Tensor
            Interatomic distances of arbitrary shape ``(...)``.

        Returns
        -------
        Tensor
            The envelope values, same shape as ``r``, in ``[0, 1]`` and zero
            for ``r >= cutoff``.
        """
        # the argument stops a hair short of 1 so the exponent stays finite;
        # the value there is exp(-5e5), zero in every floating format
        x = (r / self.cutoff).clamp(0.0, 1.0 - 1e-6)
        return torch.exp(1.0 - 1.0 / (1.0 - x * x))


class PolynomialCutoff(nn.Module):
    """Smooth p-degree polynomial envelope (NequIP, MACE, DimeNet, the Equiformers), C^2 at rc.

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
        # 1 - (p+1)(p+2)/2 x^p + p(p+2) x^(p+1) - p(p+1)/2 x^(p+2), with one power:
        # each power's derivative costs kernels in both force passes of training
        tail = ((-(p * (p + 1) / 2)) * x + p * (p + 2)) * x - (p + 1) * (p + 2) / 2
        env = 1.0 + tail * x ** p
        return env * (r < self.cutoff)
