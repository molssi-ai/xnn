"""Matern kernels of half-integer order, as radial functions with their derivatives.

The Matern kernel of order ``nu = n + 1/2`` with length scale ``sigma`` is

``k(r) = exp(-a r) P_n(r)``,  ``a = sqrt(2 nu) / sigma``,
``P_n(r) = sum_{k=0..n} (n + k)! / (2n)! * C(n, k) * (2 a r)^(n - k)``

(Chmiela et al., Sci. Adv. 3, e1603015, 2017, eq. 3). Gradient-domain models
need ``g(r) = -k'(r) / r`` and ``t(r) = g'(r) / r``; for ``n >= 2`` both are
``exp(-a r)`` times polynomials, finite at ``r = 0`` (the numerators vanish
there), whose coefficients are formed here once.
"""
from __future__ import annotations

import math
from typing import List

import torch
from torch import Tensor


def _polyval(coef: List[float], r: Tensor) -> Tensor:
    """``sum_k coef[k] r^k`` by Horner's rule."""
    out = torch.full_like(r, coef[-1])
    for c in reversed(coef[:-1]):
        out = out * r + c
    return out


def _deriv(coef: List[float]) -> List[float]:
    return [k * c for k, c in enumerate(coef)][1:] or [0.0]


def _divide_by_r(coef: List[float]) -> List[float]:
    """``coef / r`` for a polynomial whose constant term vanishes (up to rounding, dropped)."""
    return coef[1:] or [0.0]


class Matern:
    """Matern kernel of order ``n + 1/2`` as a radial function.

    Parameters
    ----------
    sigma : float
        Length scale.
    n : int, optional
        Order ``nu = n + 1/2``, by default 2 (``nu = 5/2``); at least 2.
    """

    def __init__(self, sigma: float, n: int = 2):
        self.sigma = float(sigma)
        self.n = int(n)
        if self.n < 2:
            raise ValueError("the gradient-domain kernel needs a Matern order of at least 5/2 (n >= 2)")
        self.a = math.sqrt(2 * self.n + 1) / self.sigma
        f = math.factorial
        p = [0.0] * (self.n + 1)
        for k in range(self.n + 1):
            p[self.n - k] = (f(self.n + k) / f(2 * self.n) * math.comb(self.n, k) * (2 * self.a) ** (self.n - k))
        self.p = p
        # k' = exp(-a r) (P' - a P), so g = -k'/r = exp(-a r) (a P - P') / r
        q = [self.a * c for c in p]
        for k, c in enumerate(_deriv(p)):
            q[k] -= c
        self.q = _divide_by_r(q)
        # g' = exp(-a r) (Q' - a Q), so t = g'/r = exp(-a r) (Q' - a Q) / r
        t = [-self.a * c for c in self.q]
        for k, c in enumerate(_deriv(self.q)):
            t[k] += c
        self.t = _divide_by_r(t + [0.0])

    def k(self, r: Tensor) -> Tensor:
        """The kernel ``k(r)``."""
        return torch.exp(-self.a * r) * _polyval(self.p, r)

    def g(self, r: Tensor) -> Tensor:
        """``-k'(r) / r``."""
        return torch.exp(-self.a * r) * _polyval(self.q, r)

    def t_of(self, r: Tensor) -> Tensor:
        """``g'(r) / r``."""
        return torch.exp(-self.a * r) * _polyval(self.t, r)

    def g_and_t(self, r: Tensor):
        """``g(r)`` and ``t(r)`` with one exponential."""
        ex = torch.exp(-self.a * r)
        return ex * _polyval(self.q, r), ex * _polyval(self.t, r)


def safe_norm(x: Tensor) -> Tensor:
    """Euclidean norm over the last axis with a zero gradient at the origin.

    Exact for the Matern kernels of order 5/2 and above, whose radial functions
    have zero slope at ``r = 0``.
    """
    s = (x * x).sum(-1)
    pos = s > 0
    return torch.where(pos, torch.sqrt(torch.where(pos, s, torch.ones_like(s))), torch.zeros_like(s))
