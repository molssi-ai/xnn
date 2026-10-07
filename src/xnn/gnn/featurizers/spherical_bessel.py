"""The 2D spherical Fourier-Bessel basis of DimeNet (Gasteiger et al., ICLR 2020).

DimeNet represents the distance ``d`` and the angle ``alpha`` of a pair of
messages jointly in the regular solutions of the Helmholtz equation inside the
cutoff sphere (paper section 5, eq 6),

    a_ln(d, alpha) = sqrt(2 / (c^3 j_{l+1}(z_ln)^2)) * j_l(z_ln d / c) * Y_l^0(alpha)

with the spherical Bessel functions ``j_l``, their ``n``-th positive zero
``z_ln`` (so every function vanishes at the cutoff ``c``), and the ``m = 0``
spherical harmonics ``Y_l^0``. The functions are orthonormal on the cutoff
sphere, ``l = 0 .. N_SHBF - 1`` and ``n = 1 .. N_SRBF``; the radial-only
member of the family (``l = 0``, zeros ``n pi``) is the Bessel radial basis
:class:`~xnn.gnn.featurizers.BesselRBF` (eq 7).

Everything here is evaluated in plain torch: the spherical Bessel functions
by the power series close to the origin and the upward recurrence beyond
(the switch keeps both exact to double precision), their zeros by bisection
between the interlacing zeros of the previous order, and ``Y_l^0`` through
the Legendre recurrence. The internal arithmetic is float64 whatever the
model dtype, so a float32 model gets correctly rounded basis values.
"""
# NOTE: no `from __future__ import annotations` -- the module is part of the
# TorchScript core of DimeNet and TorchScript needs the real annotations.
import math
from typing import List

import numpy as np
import torch
from torch import Tensor, nn

#: Default number of terms of the power series of ``j_l`` (double precision for
#: ``x < l + 2``, where it is used).
_SERIES_TERMS = 30


def spherical_bessel_jn(l_max: int, x: Tensor, n_terms: int = _SERIES_TERMS) -> Tensor:
    """Spherical Bessel functions of the first kind ``j_0(x) .. j_lmax(x)``.

    Parameters
    ----------
    l_max : int
        Highest order.
    x : Tensor
        Arguments of any shape, ``x >= 0``.
    n_terms : int, optional
        Terms of the power series, by default 30 (double precision over its
        range of use).

    Returns
    -------
    Tensor
        ``j_l(x)`` stacked along a new last axis, shape ``(..., l_max + 1)``,
        in the dtype of ``x``.

    Notes
    -----
    Order ``l`` uses the power series
    ``x^l / (2l+1)!! * sum_k (-x^2/2)^k / (k! (2l+3)(2l+5)...(2l+2k+1))``
    for ``x < l + 2`` and the upward recurrence
    ``j_{l+1} = (2l+1)/x j_l - j_{l-1}`` from ``j_0 = sin x / x`` and
    ``j_1 = (sin x / x - cos x) / x`` beyond; the recurrence loses digits for
    ``x`` below the order, the series needs more terms above it, and the
    switch at ``l + 2`` keeps both at double precision. The arithmetic is
    float64 internally.
    """
    x64 = x.to(torch.float64)
    # the power series of every order
    minus_half_x2 = -0.5 * x64 * x64
    series: List[Tensor] = []
    double_factorial = 1.0
    for l in range(l_max + 1):
        double_factorial *= 2 * l + 1
        term = x64 ** l / double_factorial
        total = term
        for k in range(1, n_terms):
            term = term * minus_half_x2 / float(k * (2 * l + 2 * k + 1))
            total = total + term
        series.append(total)
    # the upward recurrence (the argument stays away from zero: this branch
    # is only selected at x >= l + 2 >= 2)
    xs = x64.clamp(min=1.0)
    j_prev = torch.sin(xs) / xs
    j_curr = (j_prev - torch.cos(xs)) / xs
    recurrence: List[Tensor] = [j_prev, j_curr]
    for l in range(1, l_max):
        j_next = (2 * l + 1) / xs * j_curr - j_prev
        recurrence.append(j_next)
        j_prev, j_curr = j_curr, j_next
    out: List[Tensor] = []
    for l in range(l_max + 1):
        out.append(torch.where(x64 < float(l + 2), series[l], recurrence[l]))
    return torch.stack(out, dim=-1).to(x.dtype)


def spherical_bessel_zeros(l_max: int, n: int) -> np.ndarray:
    """The first ``n`` positive zeros ``z_ln`` of ``j_l`` for ``l = 0 .. l_max``.

    Parameters
    ----------
    l_max : int
        Highest order.
    n : int
        Zeros per order.

    Returns
    -------
    numpy.ndarray
        Float64 zeros of shape ``(l_max + 1, n)``, ``z_l1 < z_l2 < ...``.

    Notes
    -----
    ``j_0`` vanishes at ``n pi``; the zeros of consecutive orders interlace
    (``z_{l-1,n} < z_ln < z_{l-1,n+1}``), so each ``z_ln`` is bracketed by two
    zeros of the previous order and found by bisection to machine precision.
    ``l_max`` extra zeros of ``j_0`` seed the recursion so that every order
    keeps ``n`` brackets.
    """
    zeros = np.zeros((l_max + 1, n + l_max))
    zeros[0] = np.arange(1, n + l_max + 1) * np.pi
    for l in range(1, l_max + 1):
        count = n + l_max - l
        lo = torch.tensor(zeros[l - 1, :count])
        hi = torch.tensor(zeros[l - 1, 1:count + 1])
        f_lo = spherical_bessel_jn(l, lo)[:, l]
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            f_mid = spherical_bessel_jn(l, mid)[:, l]
            same_side = (f_mid > 0) == (f_lo > 0)
            lo = torch.where(same_side, mid, lo)
            f_lo = torch.where(same_side, f_mid, f_lo)
            hi = torch.where(same_side, hi, mid)
            if float((hi - lo).max()) <= 4.0 * np.finfo(np.float64).eps * float(hi.max()):
                break
        zeros[l, :count] = (0.5 * (lo + hi)).numpy()
    return zeros[:, :n]


def zonal_harmonics(l_max: int, cos_alpha: Tensor) -> Tensor:
    """The ``m = 0`` real spherical harmonics ``Y_l^0`` of a polar angle.

    Parameters
    ----------
    l_max : int
        Highest degree.
    cos_alpha : Tensor
        Cosine of the polar angle, any shape.

    Returns
    -------
    Tensor
        ``Y_l^0(alpha) = sqrt((2l+1) / 4pi) P_l(cos alpha)`` for
        ``l = 0 .. l_max`` along a new last axis, shape ``(..., l_max + 1)``,
        in the dtype of ``cos_alpha``. The Legendre polynomials come from
        the three-term recurrence, evaluated in float64.
    """
    c = cos_alpha.to(torch.float64)
    p_prev = torch.ones_like(c)
    p_curr = c
    out: List[Tensor] = [p_prev * math.sqrt(1.0 / (4.0 * math.pi))]
    if l_max >= 1:
        out.append(p_curr * math.sqrt(3.0 / (4.0 * math.pi)))
    for l in range(1, l_max):
        p_next = ((2 * l + 1) * c * p_curr - l * p_prev) / (l + 1)
        out.append(p_next * math.sqrt((2 * l + 3) / (4.0 * math.pi)))
        p_prev, p_curr = p_curr, p_next
    return torch.stack(out, dim=-1).to(cos_alpha.dtype)


class SphericalBesselBasis(nn.Module):
    """The 2D spherical Fourier-Bessel basis ``a_ln(d, alpha)`` of DimeNet (eq 6).

    Joint representation of a distance ``d`` and an angle ``alpha`` in the
    ``N_SHBF * N_SRBF`` orthonormal functions
    ``sqrt(2 / (c^3 j_{l+1}(z_ln)^2)) j_l(z_ln d / c) Y_l^0(alpha)``. The
    basis itself ends with a step at the cutoff; DimeNet multiplies it by the
    polynomial envelope of eq 8 (:class:`~xnn.gnn.featurizers.PolynomialCutoff`)
    to make it twice continuously differentiable.

    The radial and angular factors are exposed separately
    (:meth:`radial`, :meth:`angular`) because in the directional message
    passing the distance belongs to an edge and the angle to a triplet of
    atoms, so the radial part is evaluated once per edge and gathered onto the
    triplets.

    Parameters
    ----------
    n_spherical : int, optional
        Number of spherical-harmonic degrees ``N_SHBF`` (``l = 0 .. N_SHBF-1``),
        by default 7 (the paper's value).
    n_radial : int, optional
        Zeros per degree ``N_SRBF`` (``n = 1 .. N_SRBF``), by default 6.
    cutoff : float, optional
        Cutoff radius ``c`` in Angstrom, by default 5.0.

    Attributes
    ----------
    zeros : Tensor
        Buffer of shape ``(n_spherical, n_radial)`` holding ``z_ln``.
    norm : Tensor
        Buffer of the same shape holding ``sqrt(2 / (c^3 j_{l+1}(z_ln)^2))``.
    """

    def __init__(self, n_spherical: int = 7, n_radial: int = 6, cutoff: float = 5.0):
        super().__init__()
        self.n_spherical = n_spherical
        self.n_radial = n_radial
        self.cutoff = cutoff
        zeros = torch.tensor(spherical_bessel_zeros(n_spherical - 1, n_radial))
        # j_{l+1} at the zeros of j_l, for the normalization
        j_all = spherical_bessel_jn(n_spherical, zeros)             # (L, N, L + 1)
        j_next = torch.stack([j_all[l, :, l + 1] for l in range(n_spherical)])
        norm = torch.sqrt(2.0 / (cutoff ** 3 * j_next ** 2))
        dtype = torch.get_default_dtype()
        self.register_buffer("zeros", zeros.to(dtype))
        self.register_buffer("norm", norm.to(dtype))
        #: Number of basis functions ``n_spherical * n_radial``.
        self.output_dim = n_spherical * n_radial

    def radial(self, r: Tensor) -> Tensor:
        """The radial factors ``sqrt(2 / (c^3 j_{l+1}(z_ln)^2)) j_l(z_ln r / c)``.

        Parameters
        ----------
        r : Tensor
            Distances, shape ``(E,)``.

        Returns
        -------
        Tensor
            Shape ``(E, n_spherical, n_radial)``, in the dtype of ``r``.
        """
        x = (r.to(torch.float64) / self.cutoff)[:, None, None] * self.zeros.to(torch.float64)
        parts: List[Tensor] = []
        for l in range(self.n_spherical):
            parts.append(spherical_bessel_jn(l, x[:, l, :])[..., l])
        j = torch.stack(parts, dim=1)                                 # (E, L, N)
        return (self.norm.to(torch.float64) * j).to(r.dtype)

    def angular(self, cos_alpha: Tensor) -> Tensor:
        """The angular factors ``Y_l^0(alpha)``.

        Parameters
        ----------
        cos_alpha : Tensor
            Cosines of the angles, shape ``(T,)``.

        Returns
        -------
        Tensor
            Shape ``(T, n_spherical)``.
        """
        return zonal_harmonics(self.n_spherical - 1, cos_alpha)

    def forward(self, r: Tensor, cos_alpha: Tensor) -> Tensor:
        """Evaluate the full basis for distance/angle pairs.

        Parameters
        ----------
        r : Tensor
            Distances, shape ``(T,)``.
        cos_alpha : Tensor
            Cosines of the angles paired with them, shape ``(T,)``.

        Returns
        -------
        Tensor
            ``a_ln(r, alpha)`` of shape ``(T, n_spherical * n_radial)``, the
            degree ``l`` running slowest (column ``l * n_radial + n - 1``).
        """
        out = self.radial(r) * self.angular(cos_alpha)[:, :, None]
        return out.reshape(r.shape[0], self.n_spherical * self.n_radial)
