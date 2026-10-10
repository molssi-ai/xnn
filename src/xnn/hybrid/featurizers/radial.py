"""Radial basis of the hybrid family: the exponential Bernstein polynomials of SpookyNet."""
import math

import torch
from torch import Tensor, nn

from xnn.common.models.ops import softplus_inverse


class ExponentialBernsteinRBF(nn.Module):
    r"""Exponential Bernstein polynomials, the radial basis of SpookyNet.

    Unke *et al.*, *Nat. Commun.* 12, 7273 (2021), eqs 14 and 15: the
    distance is mapped to ``x = exp(-gamma r)`` in ``(0, 1]`` and expanded in
    the ``K`` Bernstein polynomials of degree ``K - 1``,

        ``b_k(x) = binom(K - 1, k) x^k (1 - x)^(K - 1 - k)``,  ``k = 0, ..., K - 1``,

    which form a partition of unity and approximate any continuous function
    of ``x`` as ``K`` grows. The decay ``gamma`` is one positive parameter
    shared by all functions (kept positive through a softplus), so the rate
    at which a learned radial function can vary falls off with the distance.
    The polynomials are evaluated in log space,
    ``log binom + k log x + (K - 1 - k) log(1 - x)``, which keeps the large
    binomials and small powers apart. A cutoff envelope is not included; the
    model multiplies it in.

    Parameters
    ----------
    n_rbf : int, optional
        Number of basis functions ``K``, by default 16.
    gamma : float, optional
        Initial decay ``gamma`` in inverse length units, by default 0.5 / bohr
        (0.9449 / Angstrom), the paper's value.
    trainable : bool, optional
        Learn ``gamma``, by default ``True`` (the paper).
    exp_weighting : bool, optional
        Multiply every function by ``x = exp(-gamma r)``, an option of the
        reference code that is not part of the paper's eq 14. Default
        ``False``.

    Attributes
    ----------
    gamma_raw : Tensor
        ``softplus^-1(gamma)``, a parameter when ``trainable``.
    """

    __jit_unused_properties__ = ["gamma"]

    def __init__(self, n_rbf: int = 16, gamma: float = 0.5 / 0.5291772109044924,
                 trainable: bool = True, exp_weighting: bool = False):
        super().__init__()
        self.n_rbf = int(n_rbf)
        self.exp_weighting = bool(exp_weighting)
        k = torch.arange(self.n_rbf, dtype=torch.float64)
        n = float(self.n_rbf - 1)
        log_binom = (math.lgamma(n + 1.0) - torch.lgamma(k + 1.0)
                     - torch.lgamma(n - k + 1.0))
        dtype = torch.get_default_dtype()
        self.register_buffer("log_binom", log_binom.to(dtype), persistent=False)
        self.register_buffer("power_x", k.to(dtype), persistent=False)
        self.register_buffer("power_1mx", (n - k).to(dtype), persistent=False)
        raw = torch.tensor(float(softplus_inverse(float(gamma))), dtype=dtype)
        if trainable:
            self.gamma_raw = nn.Parameter(raw)
        else:
            self.register_buffer("gamma_raw", raw)

    @property
    def gamma(self) -> Tensor:
        """The decay ``gamma = softplus(gamma_raw)``."""
        return nn.functional.softplus(self.gamma_raw)

    def exact_constants(self) -> bool:
        """Rebuild float64 log-binomials exactly after a cast; whether they changed."""
        buf = self.log_binom
        if buf.dtype != torch.float64:
            return False
        k = torch.arange(self.n_rbf, dtype=torch.float64, device=buf.device)
        n = float(self.n_rbf - 1)
        exact = math.lgamma(n + 1.0) - torch.lgamma(k + 1.0) - torch.lgamma(n - k + 1.0)
        if torch.equal(buf, exact):
            return False
        with torch.no_grad():
            buf.copy_(exact)
        return True

    def forward(self, r: Tensor) -> Tensor:
        """Expand distances into the basis.

        Parameters
        ----------
        r : Tensor
            Interatomic distances ``(E,)``, positive.

        Returns
        -------
        Tensor
            ``(E, n_rbf)``; column ``k`` is ``b_k(exp(-gamma r))``.
        """
        log_x = -nn.functional.softplus(self.gamma_raw) * r.unsqueeze(-1)
        log_b = (self.log_binom + self.power_x * log_x
                 + self.power_1mx * torch.log(-torch.expm1(log_x)))
        rbf = torch.exp(log_b)
        if self.exp_weighting:
            rbf = rbf * torch.exp(log_x)
        return rbf
