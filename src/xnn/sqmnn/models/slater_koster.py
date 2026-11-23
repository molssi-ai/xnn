"""Slater-Koster two-center matrices of density-functional tight binding.

Orbitals are real spherical harmonics ordered ``m = -l, ..., l`` in every shell
(``p``: ``y, z, x``; ``d``: ``xy, yz, z2, xz, x2-y2``; ``f``: ``y(3x2-y2), xyz,
yz2, z3, xz2, z(x2-y2), x(x2-3y2)``).

In the frame of a bond, with ``z`` along the vector from atom ``A`` to atom
``B``, a two-center integral couples only orbitals of the same ``m`` and depends
on ``|m|`` (sigma, pi, delta, phi). In the molecular frame (Slater and Koster,
Phys. Rev. 94, 1498 (1954))

    M_mu,nu = sum_k V_k(R) P_k,mu,nu(u),    P_k,mu,nu = sum_{|m| = k} D_mu,m D_nu,m,

with ``D`` the rotation of the real harmonics that takes ``z`` to the unit bond
vector ``u``. The sum over ``m = +-k`` makes ``P_k`` independent of the rotation
about the bond: it is a homogeneous polynomial of degree ``l1 + l2`` in the
components of ``u``. Its coefficients are computed once with numpy (exact
quadrature of the rotated harmonics and a least-squares fit, unique for
homogeneous polynomials on the sphere) and evaluated with torch, so the
matrices are differentiable to any order in the positions.
"""
from __future__ import annotations

import functools
import math
from typing import Tuple

import numpy as np
import torch
from torch import Tensor

#: Number of orbitals of a shell of angular momentum ``l``.
N_ORBITALS = (1, 3, 5, 7)

_PI = math.pi


def real_harmonics(l: int, v: np.ndarray) -> np.ndarray:
    """Real normalized spherical harmonics of order ``l`` (``m = -l ... l``) at unit vectors ``v``,
    shape ``(..., 2l + 1)``."""
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    if l == 0:
        return np.full(v.shape[:-1] + (1,), 0.5 / math.sqrt(_PI))
    if l == 1:
        c = math.sqrt(3.0 / (4.0 * _PI))
        return np.stack([c * y, c * z, c * x], -1)
    if l == 2:
        a, b = math.sqrt(15.0 / _PI) / 2.0, math.sqrt(5.0 / _PI) / 4.0
        return np.stack([a * x * y, a * y * z, b * (3.0 * z * z - 1.0), a * x * z, 0.5 * a * (x * x - y * y)], -1)
    if l == 3:
        c3 = math.sqrt(35.0 / (2.0 * _PI)) / 4.0
        c2 = math.sqrt(105.0 / _PI) / 2.0
        c1 = math.sqrt(21.0 / (2.0 * _PI)) / 4.0
        c0 = math.sqrt(7.0 / _PI) / 4.0
        return np.stack([c3 * y * (3.0 * x * x - y * y), c2 * x * y * z, c1 * y * (5.0 * z * z - 1.0),
                         c0 * z * (5.0 * z * z - 3.0), c1 * x * (5.0 * z * z - 1.0), 0.5 * c2 * z * (x * x - y * y),
                         c3 * x * (x * x - 3.0 * y * y)], -1)
    raise ValueError(f"angular momentum {l} is not supported")


@functools.lru_cache(maxsize=None)
def _sphere_quadrature(n_theta: int = 10, n_phi: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Points and weights exact for polynomials of degree below ``min(2 n_theta, n_phi)`` on the sphere."""
    t, wt = np.polynomial.legendre.leggauss(n_theta)
    phi = 2.0 * _PI * np.arange(n_phi) / n_phi
    st = np.sqrt(1.0 - t * t)
    pts = np.stack([np.outer(st, np.cos(phi)), np.outer(st, np.sin(phi)), np.outer(t, np.ones(n_phi))], -1)
    w = np.outer(wt, np.full(n_phi, 2.0 * _PI / n_phi))
    return pts.reshape(-1, 3), w.reshape(-1)


def _rotation(l: int, u: np.ndarray) -> np.ndarray:
    """``D[mu, m]``: the bond-frame harmonic ``m`` (``z`` along ``u``) in the molecular harmonics ``mu``."""
    u = u / np.linalg.norm(u)
    helper = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = helper - u * (helper @ u)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    pts, w = _sphere_quadrature()
    local = np.stack([pts @ e1, pts @ e2, pts @ u], -1)
    return np.einsum("q,qa,qb->ab", w, real_harmonics(l, pts), real_harmonics(l, local))


@functools.lru_cache(maxsize=None)
def monomial_powers(n: int) -> np.ndarray:
    """Exponents ``(n_monomials, 3)`` of the homogeneous monomials of degree ``n`` in ``x, y, z``."""
    return np.array([(a, b, n - a - b) for a in range(n, -1, -1) for b in range(n - a, -1, -1)], dtype=np.int64)


@functools.lru_cache(maxsize=None)
def rotation_polynomials(l1: int, l2: int) -> np.ndarray:
    """Coefficients ``(n_monomials, min(l1, l2) + 1, 2 l1 + 1, 2 l2 + 1)`` of ``P_k,mu,nu`` in the
    homogeneous monomials of degree ``l1 + l2`` (:func:`monomial_powers`)."""
    n = l1 + l2
    powers = monomial_powers(n)
    rng = np.random.default_rng(1954)
    dirs = rng.normal(size=(4 * len(powers) + 16, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    kmax = min(l1, l2)
    values = np.zeros((len(dirs), kmax + 1, 2 * l1 + 1, 2 * l2 + 1))
    for q, u in enumerate(dirs):
        d1, d2 = _rotation(l1, u), _rotation(l2, u)
        for m in range(-kmax, kmax + 1):
            values[q, abs(m)] += np.outer(d1[:, l1 + m], d2[:, l2 + m])
    basis = np.prod(dirs[:, None, :] ** powers[None, :, :], axis=-1)
    coef, *_ = np.linalg.lstsq(basis, values.reshape(len(dirs), -1), rcond=None)
    coef[np.abs(coef) < 1e-13] = 0.0
    return coef.reshape(len(powers), kmax + 1, 2 * l1 + 1, 2 * l2 + 1)


@functools.lru_cache(maxsize=None)
def _polynomials_on(l1: int, l2: int, dtype: torch.dtype, device: torch.device) -> Tuple[Tensor, Tensor]:
    coef = rotation_polynomials(l1, l2)
    return (torch.as_tensor(monomial_powers(l1 + l2), device=device),
            torch.as_tensor(coef, dtype=dtype, device=device))


def monomials(u: Tensor, n: int) -> Tensor:
    """The homogeneous monomials of degree ``n`` of the vectors ``u`` ``(..., 3)``, ``(..., n_monomials)``."""
    powers = monomial_powers(n)
    if n == 0:
        return torch.ones(u.shape[:-1] + (1,), dtype=u.dtype, device=u.device)
    pw = [torch.ones_like(u)]
    for _ in range(n):
        pw.append(pw[-1] * u)
    stacked = torch.stack(pw, -2)                                # (..., n + 1, 3)
    return (stacked[..., powers[:, 0], 0] * stacked[..., powers[:, 1], 1] * stacked[..., powers[:, 2], 2])


def shell_block(u: Tensor, integrals: Tensor, l1: int, l2: int) -> Tensor:
    """Two-center blocks ``(..., 2 l1 + 1, 2 l2 + 1)`` between a shell ``l1`` on ``A`` and a shell ``l2``
    on ``B`` from the unit vectors ``u`` from ``A`` to ``B`` ``(..., 3)`` and the bond-frame integrals
    ``integrals`` ``(..., min(l1, l2) + 1)`` (sigma first)."""
    _, coef = _polynomials_on(l1, l2, u.dtype, u.device)
    mono = monomials(u, l1 + l2)                                  # (..., n_mono)
    angular = torch.tensordot(mono, coef, dims=([-1], [0]))       # (..., k, n1, n2)
    return (angular * integrals[..., :, None, None]).sum(-3)
