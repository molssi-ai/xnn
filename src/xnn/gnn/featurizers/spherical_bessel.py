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
    """Spherical Bessel functions of the first kind, orders 0 to ``l_max``.

    See :func:`_spherical_bessel_values` for the evaluation. With a gradient
    to track, the derivative comes from the identity
    ``j_l' = (l j_{l-1} - (l+1) j_{l+1}) / (2l+1)``: autograd then keeps ``x``
    only, not the intermediates of the series, and higher derivatives
    (force training) come from the same identity. TorchScript evaluates
    :func:`_spherical_bessel_values` directly.

    Parameters
    ----------
    l_max : int
        Highest order.
    x : Tensor
        Arguments of any shape, ``x >= 0``.
    n_terms : int, optional
        Terms of the power series, by default 30.

    Returns
    -------
    Tensor
        ``j_l(x)`` stacked along a new last axis, shape ``(..., l_max + 1)``,
        in the dtype of ``x``.
    """
    if not torch.jit.is_scripting():
        if torch.is_grad_enabled() and x.requires_grad:
            return _SphericalBessel.apply(l_max, x, n_terms)
    return _spherical_bessel_values(l_max, x, n_terms)


class _SphericalBessel(torch.autograd.Function):
    """``j_0 .. j_{l_max}`` of ``x`` whose derivative is the recurrence identity."""

    @staticmethod
    def forward(ctx, l_max: int, x: Tensor, n_terms: int) -> Tensor:
        ctx.l_max, ctx.n_terms = l_max, n_terms
        ctx.save_for_backward(x)
        return _spherical_bessel_values(l_max, x, n_terms)

    @staticmethod
    def backward(ctx, grad: Tensor):
        (x,) = ctx.saved_tensors
        l_max = ctx.l_max
        # differentiable: the second derivative comes through this same function
        j = spherical_bessel_jn(l_max + 1, x, ctx.n_terms)
        orders = torch.arange(l_max + 1, dtype=j.dtype, device=j.device)
        lower = torch.cat([torch.zeros_like(j[..., :1]), j[..., :l_max]], dim=-1)
        deriv = (orders * lower - (orders + 1.0) * j[..., 1:]) / (2.0 * orders + 1.0)
        return None, (grad * deriv).sum(-1), None


def _spherical_bessel_values(l_max: int, x: Tensor, n_terms: int = _SERIES_TERMS) -> Tensor:
    """Spherical Bessel functions of the first kind, orders 0 to ``l_max``.

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
    switch at ``l + 2`` keeps both at double precision. The series of all
    orders is evaluated at once by Horner's scheme in ``-x^2 / 2`` (one fused
    multiply-add per term). The arithmetic is float64 internally.
    """
    x64 = x.to(torch.float64)
    n_l = l_max + 1
    orders = torch.arange(n_l, dtype=torch.float64, device=x.device)
    # the power series of every order at once, by Horner's scheme in y = -x^2/2:
    # sum_k a_k(l) y^k with a_k(l) = prod_{i <= k} 1 / (i (2l + 2i + 1))
    i = torch.arange(1, n_terms, dtype=torch.float64, device=x.device).unsqueeze(-1)
    coef = torch.cat([torch.ones(1, n_l, dtype=torch.float64, device=x.device),
                      torch.cumprod(1.0 / (i * (2.0 * orders + 2.0 * i + 1.0)), dim=0)])
    y = (-0.5 * x64 * x64).unsqueeze(-1)
    total = coef[n_terms - 1].expand(list(x64.shape) + [n_l])
    for k in range(n_terms - 2, -1, -1):
        total = torch.addcmul(coef[k], total, y)
    # x^l / (2l+1)!!, the powers by repeated products (finite gradients at x = 0;
    # a scan over the short last axis is slow on large inputs)
    power = torch.ones_like(x64)
    powers: List[Tensor] = [power]
    for _ in range(l_max):
        power = power * x64
        powers.append(power)
    series = total * torch.stack(powers, dim=-1) / torch.cumprod(2.0 * orders + 1.0, dim=0)
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
    rec = torch.stack(recurrence[:n_l], dim=-1)
    return torch.where(x64.unsqueeze(-1) < orders + 2.0, series, rec).to(x.dtype)


def spherical_bessel_rows(x: Tensor, first: int = 0, n_terms: int = _SERIES_TERMS) -> Tensor:
    """Spherical Bessel functions with one order per row: ``j_{first + r}(x[..., r, :])``.

    The radial factors of the 2D basis evaluate ``j_l`` at the zeros of the
    same order only, so the order runs along the row axis (``-2``) of ``x``.
    One evaluation for all rows gives the values of :func:`spherical_bessel_jn`
    bitwise, without the orders a row does not use. With a gradient to track,
    the same evaluation also takes the neighboring orders and keeps the
    derivative ``j_l' = (l j_{l-1} - (l+1) j_{l+1}) / (2l+1)``; the second
    derivative follows from the spherical Bessel equation,
    ``j_l'' = -2 j_l' / x - (1 - l(l+1) / x^2) j_l``, so no derivative order
    evaluates the series again.

    Parameters
    ----------
    x : Tensor
        Arguments of shape ``(..., R, N)``, ``x >= 0`` (``x > 0`` where
        differentiated twice).
    first : int, optional
        Order of row 0, by default 0; row ``r`` has order ``first + r``.
    n_terms : int, optional
        Terms of the power series, by default 30.

    Returns
    -------
    Tensor
        Same shape and dtype as ``x``.
    """
    if not torch.jit.is_scripting():
        if torch.is_grad_enabled() and x.requires_grad:
            with torch.no_grad():
                both = _spherical_bessel_row_values(x, first, n_terms, True)   # (..., 3, R, N)
                order = torch.arange(first, first + x.shape[-2], dtype=x.dtype,
                                     device=x.device).unsqueeze(-1)
                value = both[..., 1, :, :]
                slope = (order * both[..., 0, :, :] - (order + 1.0) * both[..., 2, :, :]) / (2.0 * order + 1.0)
            return _BesselRows.apply(x, value, slope, first)
    return _spherical_bessel_row_values(x, first, n_terms, False)[..., 0, :, :]


class _BesselRows(torch.autograd.Function):
    """``j_l`` per row from its precomputed value; the derivative is :class:`_BesselRowsSlope`."""

    @staticmethod
    def forward(ctx, x: Tensor, value: Tensor, slope: Tensor, first: int) -> Tensor:
        ctx.first = first
        ctx.save_for_backward(x, value, slope)
        return value.clone()

    @staticmethod
    def backward(ctx, grad: Tensor):
        x, value, slope = ctx.saved_tensors
        return grad * _BesselRowsSlope.apply(x, value, slope, ctx.first), None, None, None


class _BesselRowsSlope(torch.autograd.Function):
    """``j_l'`` per row; its derivative from the spherical Bessel equation (differentiable again)."""

    @staticmethod
    def forward(ctx, x: Tensor, value: Tensor, slope: Tensor, first: int) -> Tensor:
        ctx.first = first
        ctx.save_for_backward(x, value, slope)
        return slope.clone()

    @staticmethod
    def backward(ctx, grad: Tensor):
        x, value, slope = ctx.saved_tensors
        first = ctx.first
        order = torch.arange(first, first + x.shape[-2], dtype=x.dtype, device=x.device).unsqueeze(-1)
        j = _BesselRows.apply(x, value, slope, first)
        dj = _BesselRowsSlope.apply(x, value, slope, first)
        curvature = -2.0 / x * dj - (1.0 - order * (order + 1.0) / (x * x)) * j
        return grad * curvature, None, None, None


def _spherical_bessel_row_values(x: Tensor, first: int, n_terms: int = _SERIES_TERMS,
                                 neighbors: bool = False) -> Tensor:
    """:func:`spherical_bessel_rows` evaluated: the series and recurrence of
    :func:`_spherical_bessel_values` with the order taken per row.

    Returns shape ``(..., S, R, N)``: ``S = 1`` (order ``first + r``) or, with
    ``neighbors``, ``S = 3`` (orders ``first + r - 1``, ``first + r`` and
    ``first + r + 1``; an order below zero is evaluated as order zero).
    """
    x64 = x.to(torch.float64).unsqueeze(-3)                                   # (..., 1, R, N)
    n_rows = x.shape[-2]
    low = -1 if neighbors else 0
    high = 1 if neighbors else 0
    last = first + n_rows - 1 + high
    order = (torch.arange(low, high + 1, device=x.device).view(-1, 1, 1)
             + torch.arange(first, first + n_rows, device=x.device).view(1, -1, 1)).clamp(min=0)
    order_f = order.to(torch.float64)                                         # (S, R, 1)
    i = torch.arange(1, n_terms, dtype=torch.float64, device=x.device).view(-1, 1, 1, 1)
    coef = torch.cat([torch.ones_like(order_f).unsqueeze(0),
                      torch.cumprod(1.0 / (i * (2.0 * order_f + 2.0 * i + 1.0)), dim=0)])
    y = -0.5 * x64 * x64
    total = coef[n_terms - 1]                                  # broadcast by the first step
    for k in range(n_terms - 2, -1, -1):
        total = torch.addcmul(coef[k], total, y)
    power = torch.ones_like(x64)
    x_l = power
    for l in range(1, last + 1):
        power = power * x64
        x_l = torch.where(order == l, power, x_l)
    all_orders = torch.arange(last + 1, dtype=torch.float64, device=x.device)
    double_factorial = torch.cumprod(2.0 * all_orders + 1.0, dim=0)[order]
    series = total * x_l / double_factorial
    xs = x64.clamp(min=1.0)
    j_prev = torch.sin(xs) / xs
    j_curr = (j_prev - torch.cos(xs)) / xs
    rec = j_prev
    if last >= 1:
        rec = torch.where(order == 1, j_curr, rec)
    for l in range(1, last):
        j_next = (2 * l + 1) / xs * j_curr - j_prev
        rec = torch.where(order == l + 1, j_next, rec)
        j_prev, j_curr = j_curr, j_next
    return torch.where(x64 < order_f + 2.0, series, rec).to(x.dtype)


def spherical_bessel_zeros(l_max: int, n: int) -> np.ndarray:
    """The first ``n`` positive zeros ``z_ln`` of ``j_l`` for every order up to ``l_max``.

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


def real_spherical_harmonics(l_max: int, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    """Real spherical harmonics ``Y_lm`` of a unit vector, degrees 0 to ``l_max``.

    Orthonormal on the sphere, without the Condon-Shortley phase:
    ``Y_l0 = N_l0 P_l(z)``, ``Y_lm = sqrt(2) N_lm P_l^m(z) cos(m phi)`` and
    ``Y_l,-m = sqrt(2) N_lm P_l^m(z) sin(m phi)`` for ``m > 0``, with
    ``N_lm = sqrt((2l+1)/(4 pi) (l-m)!/(l+m)!)``. They are evaluated as
    polynomials in the Cartesian components (float64 internally), so they are
    smooth everywhere on the sphere, the poles included.

    Parameters
    ----------
    l_max : int
        Highest degree.
    x, y, z : Tensor
        Components of unit vectors, all of the same shape; ``z`` is the polar
        axis and the azimuth is measured from ``x`` towards ``y``.

    Returns
    -------
    Tensor
        Shape ``(..., (l_max + 1)**2)``, degree-major with ``m = -l .. l``
        within each degree (column ``l**2 + l + m``), in the dtype of ``z``.
    """
    x64, y64, z64 = x.to(torch.float64), y.to(torch.float64), z.to(torch.float64)
    # cos/sin parts r^m (cos m phi, sin m phi) = Re/Im (x + i y)^m
    cos_m: List[Tensor] = [torch.ones_like(z64)]
    sin_m: List[Tensor] = [torch.zeros_like(z64)]
    for m in range(l_max):
        cos_m.append(cos_m[m] * x64 - sin_m[m] * y64)
        sin_m.append(sin_m[m] * x64 + cos_m[m] * y64)
    # q[l][m] = P_l^m(z) / sin^m(theta): polynomials in z, from the
    # recurrences of the associated Legendre functions
    q: List[List[Tensor]] = []
    for l in range(l_max + 1):
        q.append([torch.zeros_like(z64) for _ in range(l + 1)])
    double_factorial = 1.0
    for m in range(l_max + 1):
        if m > 0:
            double_factorial *= 2 * m - 1
        q[m][m] = torch.full_like(z64, double_factorial)
        if m + 1 <= l_max:
            q[m + 1][m] = (2 * m + 1) * z64 * q[m][m]
        for l in range(m + 2, l_max + 1):
            q[l][m] = ((2 * l - 1) * z64 * q[l - 1][m] - (l + m - 1) * q[l - 2][m]) / (l - m)
    out: List[Tensor] = []
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            a = abs(m)
            ratio = 1.0                                   # (l - a)! / (l + a)!
            for k in range(l - a + 1, l + a + 1):
                ratio = ratio / k
            norm = math.sqrt((2 * l + 1) / (4.0 * math.pi) * ratio)
            if m == 0:
                out.append(norm * q[l][0])
            elif m > 0:
                out.append(math.sqrt(2.0) * norm * q[l][a] * cos_m[a])
            else:
                out.append(math.sqrt(2.0) * norm * q[l][a] * sin_m[a])
    return torch.stack(out, dim=-1).to(z.dtype)


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
        j = spherical_bessel_rows(x)                                  # (E, L, N), order l in row l
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
