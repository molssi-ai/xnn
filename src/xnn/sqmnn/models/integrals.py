"""Integrals of the NDDO methods over Slater orbitals, in PyTorch.

* :func:`overlap_local` -- two-center overlaps in the diatomic frame, by the
  prolate-spheroidal expansion ``S = sum_ij c_ij A_i(p) B_j(t)``;
* :func:`slater_condon` -- radial one-center two-electron integrals ``R^K``;
* :class:`PointCharges` / :func:`config_interaction` -- the point-charge
  multipole configurations of the two-center two-electron and
  electron-core integrals (Dewar & Thiel, Theor. Chim. Acta 46, 89, 1977;
  Thiel & Voityuk, Theor. Chim. Acta 81, 391, 1992) and their
  Klopman-Ohno interaction ``sum q_i q_j / sqrt(r_ij^2 + rho^2)``;
* :func:`local_frames` -- the orbital rotation from the diatomic to the
  molecular frame for ``s``, ``p`` and ``d`` shells.

Lengths are in bohr inside this module and energies in eV; every function
is differentiable with respect to positions and parameters.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from ..common.constants import HARTREE
from .angular import L_OF, LOCAL, M_OF, MOLECULAR

# polynomial in (xi, eta) as a dict {(i, j): coefficient}


def _pmul(a: Dict, b: Dict) -> Dict:
    out: Dict = {}
    for (i1, j1), c1 in a.items():
        for (i2, j2), c2 in b.items():
            key = (i1 + i2, j1 + j2)
            out[key] = out.get(key, 0.0) + c1 * c2
    return {k: v for k, v in out.items() if v != 0.0}


def _padd(a: Dict, b: Dict, sb: float = 1.0) -> Dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0.0) + sb * v
    return {k: v for k, v in out.items() if v != 0.0}


def _ppow(a: Dict, n: int) -> Dict:
    out = {(0, 0): 1.0}
    for _ in range(n):
        out = _pmul(out, a)
    return out


# spheroidal coordinates in units of R/2: rA = xi + eta, rB = xi - eta,
# zA = 1 + xi eta, zB = xi eta - 1, rho^2 = (xi^2 - 1)(1 - eta^2)
_RA = {(1, 0): 1.0, (0, 1): 1.0}
_RB = {(1, 0): 1.0, (0, 1): -1.0}
_ZA = {(0, 0): 1.0, (1, 1): 1.0}
_ZB = {(1, 1): 1.0, (0, 0): -1.0}
_RHO2 = _pmul({(2, 0): 1.0, (0, 0): -1.0}, {(0, 0): 1.0, (0, 2): -1.0})
_VOLUME = {(2, 0): 1.0, (0, 2): -1.0}

# angular polynomial (without the azimuthal factor) and normalization of a real
# harmonic of given (l, |m|): Y = C * poly / r^l * trig(m phi)
_HARMONIC_NORM = {(0, 0): 1.0 / math.sqrt(4 * math.pi), (1, 0): math.sqrt(3 / (4 * math.pi)),
                  (1, 1): math.sqrt(3 / (4 * math.pi)), (2, 0): math.sqrt(5 / (16 * math.pi)),
                  (2, 1): math.sqrt(15 / (4 * math.pi)), (2, 2): math.sqrt(15 / (16 * math.pi))}


def _angular_poly(l: int, m: int, z: Dict, rho2: Dict) -> Dict:
    """``r^l Y_lm`` up to the constant and the azimuthal factor; odd powers of
    rho are paired between the two centers, so a single rho is returned as
    the marker ``None`` and handled by the caller."""
    if (l, m) == (0, 0):
        return {(0, 0): 1.0}
    if (l, m) == (1, 0):
        return z
    if (l, m) == (2, 0):
        return _padd(_pmul({(0, 0): 2.0}, _pmul(z, z)), rho2, -1.0)
    if (l, m) == (2, 2):
        return rho2
    if (l, m) == (1, 1):
        return {(0, 0): 1.0}          # times rho
    if (l, m) == (2, 1):
        return z                      # times rho
    raise ValueError((l, m))


@lru_cache(maxsize=None)
def _overlap_poly(na: int, la: int, nb: int, lb: int, m: int) -> Tuple[np.ndarray, float]:
    """Coefficients ``c_ij`` and the constant of the local-frame overlap of
    ``(na, la, m)`` on A with ``(nb, lb, m)`` on B (z from A to B)."""
    pa = _pmul(_ppow(_RA, na - 1 - la), _angular_poly(la, m, _ZA, _RHO2))
    pb = _pmul(_ppow(_RB, nb - 1 - lb), _angular_poly(lb, m, _ZB, _RHO2))
    poly = _pmul(_pmul(pa, pb), _VOLUME)
    if m == 1:      # rho_A rho_B
        poly = _pmul(poly, _RHO2)
    phi = 2.0 * math.pi if m == 0 else math.pi
    const = _HARMONIC_NORM[(la, m)] * _HARMONIC_NORM[(lb, m)] * phi
    deg = max(max(i for i, _ in poly), max(j for _, j in poly)) + 1
    c = np.zeros((deg, deg))
    for (i, j), v in poly.items():
        c[i, j] = v
    return c, const


def _a_integrals(p: Tensor, n: int) -> Tensor:
    """``A_k(p) = int_1^inf xi^k exp(-p xi) dxi`` for ``k < n``; shape ``(..., n)``."""
    e = torch.exp(-p)
    out = [e / p]
    for k in range(1, n):
        out.append((k * out[-1] + e) / p)
    return torch.stack(out, dim=-1)


_SERIES_LIMIT = 6.0
_SERIES_TERMS = 80


def _b_integrals(t: Tensor, n: int, order: Optional[Tensor] = None) -> Tensor:
    """``B_k(t) = int_-1^1 eta^k exp(-t eta) deta`` for ``k < n``; shape ``(..., n)``.

    Upward recursion for ``|t| >= 6`` and the power series below, where the
    recursion loses digits. ``order`` (optional, per element) truncates the
    series after the term ``t^order``, the approximation of the reference
    program (:func:`series_order`).
    """
    full = _SERIES_TERMS - 1
    if order is None:
        order = torch.full(t.shape, full, dtype=torch.long, device=t.device)
    small = (t.abs() < _SERIES_LIMIT) | (order < full)
    ts = torch.where(small, torch.full_like(t, _SERIES_LIMIT), t)
    ep, em = torch.exp(ts), torch.exp(-ts)
    rec = [(ep - em) / ts]
    for k in range(1, n):
        rec.append((k * rec[-1] + (-1) ** k * ep - em) / ts)
    rec = torch.stack(rec, dim=-1)
    tz = torch.where(small, t, torch.zeros_like(t))
    k = torch.arange(_SERIES_TERMS, dtype=t.dtype, device=t.device)
    # (-t)^k / k! as a running product (no 0^0 in the derivative)
    step = (-tz).unsqueeze(-1) / k[1:].clamp(min=1.0)
    terms = torch.cat([torch.ones_like(tz).unsqueeze(-1), torch.cumprod(step, dim=-1)], -1)
    terms = torch.where(k.long() <= order.unsqueeze(-1), terms, torch.zeros_like(terms))
    j = torch.arange(n, device=t.device)
    jk = j.unsqueeze(-1) + k.long()                                      # (n, K)
    weight = torch.where(jk % 2 == 0, 2.0 / (jk + 1).to(t.dtype), torch.zeros((), dtype=t.dtype))
    series = terms.unsqueeze(-2).mul(weight).sum(-1)                     # (..., n)
    return torch.where(small.unsqueeze(-1), series, rec)


def series_order(t: Tensor, scheme: Tensor) -> Tensor:
    """Number of series terms of the ``B`` integrals in the reference program.

    MOPAC sums the power series of ``B_k(t)`` only up to ``t^6`` for ``|t| <=
    0.5``, ``t^7`` up to 1, ``t^12`` up to 2 and ``t^15`` up to 3 (``scheme``
    0, the general ``spd`` overlap routine); its routine for pairs of ``sp``
    elements up to Cl (``scheme`` = highest index ``k`` of the integrals it
    forms, 2-6) truncates only for ``|t| <= 0.5`` (and ``t^7`` up to 1 when
    ``k = 6``). ``|t| <= 1e-6`` keeps the ``t^0`` term. These truncations
    change overlaps by up to ~1e-6 relative and are part of the published
    methods' numbers; :data:`_SERIES_TERMS` - 1 means the exact value.
    """
    a = t.detach().abs()
    full = _SERIES_TERMS - 1
    general = scheme == 0
    o = torch.full(t.shape, full, dtype=torch.long, device=t.device)
    o = torch.where((a > 2.0) & (a <= 3.0) & general, 15, o)
    o = torch.where((a > 1.0) & (a <= 2.0) & general, 12, o)
    o = torch.where((a > 0.5) & (a <= 1.0) & (general | (scheme > 5)), 7, o)
    o = torch.where(a <= 0.5, 6, o)
    o = torch.where(a <= 1e-6, 0, o)
    return o


def _sto_norm(n: int, zeta: Tensor) -> Tensor:
    return (2.0 * zeta) ** (n + 0.5) / math.sqrt(math.factorial(2 * n))


def overlap_local(na: int, la: int, za: Tensor, nb: int, lb: int, zb: Tensor, r: Tensor,
                  m: int, scheme: Optional[Tensor] = None) -> Tensor:
    """Overlap of real Slater orbitals in the diatomic frame.

    ``(na, la, m)`` on atom A at the origin, ``(nb, lb, m)`` on atom B at
    ``r`` along ``+z``; the orbitals of both atoms are oriented along the
    same axes, so the overlap depends on ``|m|`` only.

    Parameters
    ----------
    na, la, nb, lb, m : int
        Principal and angular quantum numbers, ``|m|``.
    za, zb, r : Tensor
        Exponents (bohr^-1) and separations (bohr), broadcastable.
    scheme : Tensor, optional
        Per-element truncation scheme of the auxiliary series
        (:func:`series_order`); exact integrals when omitted.

    Returns
    -------
    Tensor
        The overlap integrals.
    """
    c, const = _overlap_poly(na, la, nb, lb, m)
    half = 0.5 * r
    p = half * (za + zb)
    t = half * (za - zb)
    a = _a_integrals(p, c.shape[0])
    order = None if scheme is None else series_order(t, scheme)
    b = _b_integrals(t, c.shape[1], order)
    cc = torch.as_tensor(c, dtype=r.dtype, device=r.device)
    s = torch.einsum("...i,ij,...j->...", a, cc, b)
    return const * _sto_norm(na, za) * _sto_norm(nb, zb) * half ** (na + nb + 1) * s


def slater_condon(k: int, na: int, za: Tensor, nb: int, zb: Tensor, nc: int, zc: Tensor,
                  nd: int, zd: Tensor) -> Tensor:
    """Radial Slater-Condon integral ``R^k(ab, cd)`` of Slater orbitals, eV.

    ``int int R_a R_b (1) r_<^k / r_>^(k+1) R_c R_d (2) r1^2 r2^2 dr1 dr2``
    with electron 1 in ``a b`` and electron 2 in ``c d``.
    """
    p, q = na + nb, nc + nd
    alpha, beta = za + zb, zc + zd
    s = alpha + beta
    f = math.factorial
    inner = f(p - k - 1) / alpha ** (p - k)
    for i in range(q + k + 1):
        inner = inner - beta ** i / f(i) * f(p - k - 1 + i) / s ** (p - k + i)
    term1 = f(q + k) / beta ** (q + k + 1) * inner
    term2 = torch.zeros_like(s)
    for i in range(q - k):
        term2 = term2 + beta ** i / f(i) * f(p + k + i) / s ** (p + k + i + 1)
    term2 = f(q - k - 1) / beta ** (q - k) * term2
    norm = _sto_norm(na, za) * _sto_norm(nb, zb) * _sto_norm(nc, zc) * _sto_norm(nd, zd)
    return HARTREE * norm * (term1 + term2)


# point-charge configurations: (charge, unit position); actual position = D * unit
_S2 = 1.0 / math.sqrt(2.0)
CONFIGS = {
    "Q": [(1.0, (0, 0, 0))],
    "Dz": [(0.5, (0, 0, 1)), (-0.5, (0, 0, -1))],
    "Dx": [(0.5, (1, 0, 0)), (-0.5, (-1, 0, 0))],
    "Dy": [(0.5, (0, 1, 0)), (-0.5, (0, -1, 0))],
    # linear quadrupoles of the sp basis (separation 2 qq)
    "Qzz": [(0.25, (0, 0, 1)), (0.25, (0, 0, -1)), (-0.5, (0, 0, 0))],
    "Qxx": [(0.25, (1, 0, 0)), (0.25, (-1, 0, 0)), (-0.5, (0, 0, 0))],
    "Qyy": [(0.25, (0, 1, 0)), (0.25, (0, -1, 0)), (-0.5, (0, 0, 0))],
    # square quadrupoles of the sp basis (separation qq)
    "Qxz": [(0.25, (1, 0, 1)), (0.25, (-1, 0, -1)), (-0.25, (1, 0, -1)), (-0.25, (-1, 0, 1))],
    "Qyz": [(0.25, (0, 1, 1)), (0.25, (0, -1, -1)), (-0.25, (0, 1, -1)), (-0.25, (0, -1, 1))],
    # quadrupoles of the spd scheme, one per real harmonic of order 2
    "z2": [(0.25, (0, 0, 1)), (0.25, (0, 0, -1)), (-0.125, (1, 0, 0)), (-0.125, (-1, 0, 0)),
           (-0.125, (0, 1, 0)), (-0.125, (0, -1, 0))],
    "xz": [(0.25, (_S2, 0, _S2)), (0.25, (-_S2, 0, -_S2)), (-0.25, (_S2, 0, -_S2)),
           (-0.25, (-_S2, 0, _S2))],
    "yz": [(0.25, (0, _S2, _S2)), (0.25, (0, -_S2, -_S2)), (-0.25, (0, _S2, -_S2)),
           (-0.25, (0, -_S2, _S2))],
    "x2y2": [(0.25, (1, 0, 0)), (0.25, (-1, 0, 0)), (-0.25, (0, 1, 0)), (-0.25, (0, -1, 0))],
    "xy": [(0.25, (_S2, _S2, 0)), (0.25, (-_S2, -_S2, 0)), (-0.25, (_S2, -_S2, 0)),
           (-0.25, (-_S2, _S2, 0))],
}
#: Configuration name of the dipole / quadrupole multipole components of the spd scheme.
SPD_CONFIG = {"s": "Q", "z": "Dz", "x": "Dx", "y": "Dy", "z2": "z2", "xz": "xz", "yz": "yz",
              "x2y2": "x2y2", "xy": "xy"}
_MAX_CHARGES = 6


def config_tensors(names, dtype, device) -> Tuple[Tensor, Tensor]:
    """Charges ``(S, 6)`` and unit positions ``(S, 6, 3)`` of configurations, zero-padded."""
    q = np.zeros((len(names), _MAX_CHARGES))
    u = np.zeros((len(names), _MAX_CHARGES, 3))
    for s, name in enumerate(names):
        for i, (c, pos) in enumerate(CONFIGS[name]):
            q[s, i] = c
            u[s, i] = pos
    return (torch.as_tensor(q, dtype=dtype, device=device),
            torch.as_tensor(u, dtype=dtype, device=device))


def config_interaction(qa: Tensor, ua: Tensor, da: Tensor, qb: Tensor, ub: Tensor, db: Tensor,
                       r: Tensor, rho: Tensor) -> Tensor:
    """Klopman-Ohno interaction of point-charge configurations on two centers, eV.

    Configuration ``a`` sits at the origin, ``b`` at ``(0, 0, r)``.

    Parameters
    ----------
    qa, ua : Tensor
        Charges ``(S, n)`` and unit positions ``(S, n, 3)`` of the ``a`` side.
    da : Tensor
        Charge separations ``(P, S)`` (bohr).
    qb, ub, db : Tensor
        The same for the ``b`` side.
    r : Tensor
        Separations ``(P,)`` (bohr).
    rho : Tensor
        Additive terms ``(P, S)`` (bohr), the sum of both sides'.

    Returns
    -------
    Tensor
        ``(P, S)`` interaction energies.
    """
    pa = ua.unsqueeze(0) * da[..., None, None]                     # (P, S, n, 3)
    pb = ub.unsqueeze(0) * db[..., None, None]
    pb = pb + torch.stack([torch.zeros_like(r), torch.zeros_like(r), r], -1)[:, None, None, :]
    diff = pb.unsqueeze(2) - pa.unsqueeze(3)                       # (P, S, na, nb, 3)
    d2 = diff.pow(2).sum(-1) + rho.pow(2)[..., None, None]
    qq = qa.unsqueeze(-1) * qb.unsqueeze(-2)                       # (S, na, nb)
    return HARTREE * (qq / torch.sqrt(d2)).sum((-1, -2))


# rotation from the diatomic frame to the molecular frame

def _d_forms(dtype, device) -> Tensor:
    """Quadratic forms of the real d harmonics, ``(5, 3, 3)``, molecular order."""
    c20 = math.sqrt(5.0 / (16.0 * math.pi))
    c21 = math.sqrt(15.0 / (4.0 * math.pi))
    c22 = math.sqrt(15.0 / (16.0 * math.pi))
    m = np.zeros((5, 3, 3))
    m[0] = c22 * np.diag([1.0, -1.0, 0.0])                         # x2-y2
    m[1][0, 2] = m[1][2, 0] = 0.5 * c21                            # xz
    m[2] = c20 * np.diag([-1.0, -1.0, 2.0])                        # z2
    m[3][1, 2] = m[3][2, 1] = 0.5 * c21                            # yz
    m[4][0, 1] = m[4][1, 0] = 0.5 * c21                            # xy
    return torch.as_tensor(m, dtype=dtype, device=device)


# local d order (z2, xz, yz, x2-y2, xy) as indices into the molecular d order
_LOCAL_D_IN_MOLECULAR = [2, 1, 3, 0, 4]


def local_frames(vec: Tensor, n_orb: int) -> Tensor:
    """Orbital rotation matrices from the diatomic frame of each pair.

    The diatomic ``z`` axis is the unit vector ``vec / |vec|`` (from atom A to
    atom B); ``x`` is the component of the molecular ``z`` axis (or, for
    bonds within 25 degrees of it, the molecular ``x`` axis) perpendicular to
    the bond, ``y = z x x``. The integrals are invariant to this choice, which
    only keeps the derivatives finite.

    Parameters
    ----------
    vec : Tensor
        Bond vectors ``(P, 3)``.
    n_orb : int
        Orbitals per atom (1, 4 or 9).

    Returns
    -------
    Tensor
        ``(P, n_orb, n_orb)`` with ``T[k, i]`` the coefficient of molecular
        orbital ``i`` in local orbital ``k``.
    """
    P = vec.shape[0]
    T = vec.new_zeros((P, n_orb, n_orb))
    T[:, 0, 0] = 1.0
    if n_orb == 1:
        return T
    ez = vec / vec.norm(dim=-1, keepdim=True)
    near_z = (ez[:, 2].abs() > math.cos(math.radians(25.0))).to(vec.dtype).detach()
    ref = torch.stack([near_z, torch.zeros_like(near_z), 1.0 - near_z], dim=-1)
    ex = ref - (ref * ez).sum(-1, keepdim=True) * ez
    ex = ex / ex.norm(dim=-1, keepdim=True)
    ey = torch.cross(ez, ex, dim=-1)
    E = torch.stack([ex, ey, ez], dim=1)                     # rows: local x, y, z axes
    T[:, 1:4, 1:4] = torch.stack([ez, ex, ey], dim=1)        # local order pz, px, py
    if n_orb == 9:
        M = _d_forms(vec.dtype, vec.device)                  # (5, 3, 3)
        # local form k expressed in molecular coordinates: E^T M_k E
        Mloc = M[_LOCAL_D_IN_MOLECULAR]
        rot = torch.einsum("pai,kab,pbj->pkij", E, Mloc, E)  # (P, 5, 3, 3)
        T[:, 4:, 4:] = torch.einsum("pkij,mij->pkm", rot, M) * (8.0 * math.pi / 15.0)
    return T


#: Local orbital labels by orbital count.
def local_labels(n_orb: int):
    """Labels of the diatomic-frame orbitals of an atom with ``n_orb`` orbitals."""
    return LOCAL[:n_orb]


def molecular_labels(n_orb: int):
    """Labels of the molecular-frame orbitals of an atom with ``n_orb`` orbitals."""
    return MOLECULAR[:n_orb]


__all__ = ["overlap_local", "slater_condon", "CONFIGS", "SPD_CONFIG", "config_tensors",
           "config_interaction", "local_frames", "local_labels", "molecular_labels",
           "L_OF", "M_OF"]
