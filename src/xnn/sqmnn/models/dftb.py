"""DFTB1, DFTB2 (SCC-DFTB) and DFTB3 density-functional tight binding as xnn models.

The energy of density-functional tight binding (Porezag et al., Phys. Rev. B 51,
12947 (1995); Elstner et al., Phys. Rev. B 58, 7260 (1998); Gaus, Cui and
Elstner, J. Chem. Theory Comput. 7, 931 (2011); review: Hourahine et al., J.
Chem. Phys. 152, 124101 (2020)) in a minimal valence basis,

    E = sum_mu,nu P_mu,nu H0_mu,nu + E2 + E3 + E_rep (+ E_disp) - T S_el,

* ``H0`` and the overlap ``S`` are two-center matrices from Slater-Koster tables
  (:mod:`xnn.sqmnn.common.skf`), rotated into the molecular frame
  (:mod:`xnn.sqmnn.models.slater_koster`); the diagonal holds the atomic
  orbital energies;
* ``E2 = 1/2 sum_AB gamma_AB dq_A dq_B`` couples the Mulliken charge
  fluctuations through the gamma function of exponentially decaying atomic
  densities with widths ``tau = 16 U / 5`` (``U`` the Hubbard parameter); it
  tends to ``1 / R`` at long range and to ``U`` on an atom (DFTB2 and DFTB3);
* ``E3 = 1/3 sum_AB dq_A^2 dq_B Gamma_AB`` with ``Gamma_AB = dgamma_AB / dU_A
  U^d_A`` and the Hubbard derivatives ``U^d`` (DFTB3); DFTB3 also damps the
  short-range part of gamma between hydrogen and other atoms by
  ``exp(-((U_A + U_B) / 2)^zeta R^2)``;
* ``E_rep`` is the sum of the repulsive pair potentials of the tables.

DFTB1 is the first-order method without charge self-consistency.

The tables are read as the DFTB+ program reads them: the rows a file states
(less its last one) are interpolated by 8-point Lagrange polynomials over
consecutive rows, the windows kept inside the table; past the last row a
quintic joins the interpolant's value, slope and curvature to zero within one
bohr.

The model's ``energy`` is the total (Mermin free) energy in eV; forces are its
negative gradient. Molecules and structures periodic in three directions (at
the Gamma point of the cell, with Ewald electrostatics and stress) are handled;
collinear spin polarization, long-range corrected exchange, the onsite
correction and DFTB+U are options.
"""
from __future__ import annotations

import copy
import functools
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from xnn.common.models import register_model

from ..common.skf import COLUMNS, EXTENDED_COLUMNS, SlaterKosterFile, SlaterKosterSet, load_set
from .slater_koster import shell_block
from .tightbinding import TightBinding, _all_pairs

#: Angular momentum letters.
LETTERS = "spdf"
#: Atoms whose homonuclear file states a mass below this (atomic mass units) count as hydrogen.
HYDROGEN_MASS = 3.5
#: Length (bohr) over which the integrals fall to zero past the last row of a table.
TAIL_LENGTH = 1.0
#: Density widths (inverse bohr) closer than this use the gamma function of equal widths.
EQUAL_WIDTHS = 1e-5
#: Density widths closer than this take exponentials rounded as the C library rounds them in gamma.
CLOSE_WIDTHS = 0.1
#: Ewald splitting parameter times the cube root of the cell volume (balances the real and reciprocal sums).
EWALD_SPLIT = 5.0
#: Pair-image distances the real-space Ewald sum forms at once.
EWALD_CHUNK = 250_000
#: Repulsion of hydrogen pairs of DFTB3-D3H5: height (hartree), steepness and midpoint (Angstrom).
HH_REPULSION = (4.784005158e-4, 14.31, 2.35)
#: Hydrogen pairs from this distance (bohr) on have no D3H5 repulsion.
HH_CUTOFF = 10.0
#: Parameters the ``trainable`` option accepts.
TRAINABLE = ("tables", "repulsive", "onsite", "hubbard")
#: Van der Waals radii (Angstrom) of Bondi, J. Phys. Chem. 68, 441 (1964), used by the H5 correction.
BONDI_RADII = {1: 1.20, 2: 1.40, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 10: 1.54, 14: 2.10, 15: 1.80, 16: 1.80,
               17: 1.75, 18: 1.88, 33: 1.85, 34: 1.90, 35: 1.85, 36: 2.02, 52: 2.06, 53: 1.98, 54: 2.16}
#: Defaults of the H5 correction (Rezac, J. Chem. Theory Comput. 13, 4804 (2017)): scales of the centre and
#: width of the Gaussian and its height per element (others off).
H5_DEFAULTS = {"r_scaling": 0.714, "w_scaling": 0.25, "scaling": {7: 0.18, 8: 0.06, 16: 0.21}}


@functools.lru_cache(maxsize=None)
def _end_weights() -> np.ndarray:
    """Weights of the last 8 rows giving the 8-point interpolant and its first two derivatives (in rows)
    at the last row, ``(3, 8)``."""
    out = np.zeros((3, 8))
    for j in range(8):
        y = np.zeros(8)
        y[j] = 1.0
        c = np.polynomial.Polynomial.fit(np.arange(8.0), y, 7, domain=[0, 7], window=[0, 7]).convert()
        for d in range(3):
            out[d, j] = c.deriv(d)(7.0) if d else c(7.0)
    return out


def _lagrange8(t: Tensor) -> Tensor:
    """Lagrange weights ``(..., 8)`` of the nodes ``0, ..., 7`` at ``t``: the products of ``(t - m) / (j - m)`` over
    the other nodes, all eight weights at once."""
    j = torch.arange(8, dtype=t.dtype, device=t.device)
    w = torch.ones(t.shape + (8,), dtype=t.dtype, device=t.device)
    for m in range(8):
        step = w * (t - m)[..., None] / torch.where(j == m, torch.ones_like(j), j - m)
        w = torch.where(j == m, w, step)
    return w


def _point_value(table: np.ndarray, grid: float, r: float) -> np.ndarray:
    """The 8-point interpolant of the columns of ``table`` at ``r`` by Neville's tableau for an equidistant grid,
    in float64 arithmetic: the nodes are the row positions ``(k + 1) * grid``, the tableau starts at the row below
    ``r``, and the differences of order ``m`` are multiplied by the reciprocal of the distance between the first
    node of the window and its ``m``-th neighbor."""
    n = table.shape[0]
    below = math.floor(r / grid - 1.0)
    i0 = min(max(below - 3, 0), n - 8)
    start = min(max(below - i0, 0), 7)
    xs = [(i0 + 1 + k) * grid for k in range(8)]
    y = table[i0:i0 + 8]
    c, d = [row.copy() for row in y], [row.copy() for row in y]
    out = y[start].copy()
    ns = start - 1
    for m in range(1, 8):
        inverse = 1.0 / (xs[0] - xs[m])
        for i in range(8 - m):
            ho, hp = xs[i] - r, xs[i + m] - r
            den = (c[i + 1] - d[i]) * inverse
            d[i], c[i] = hp * den, ho * den
        if 2 * (ns + 1) < 8 - m:
            out = out + c[ns + 1]
        else:
            out = out + d[ns]
            ns -= 1
    return out


def tail_join(table: np.ndarray, grid: float, step: float = 1e-5) -> np.ndarray:
    """Value, slope and curvature ``(3, C)`` (bohr) that the tail of the table columns joins at the last row.

    The value is the last row; slope and curvature are central differences of the interpolant with a step of
    ``step`` bohr about it, which the published parameter sets assume.
    """
    r0 = table.shape[0] * grid
    plus, minus = _point_value(table, grid, r0 + step), _point_value(table, grid, r0 - step)
    return np.stack([table[-1], (plus - minus) / (2.0 * step), (plus + minus - 2.0 * table[-1]) / step ** 2])


def tail_correction(table: np.ndarray, grid: float) -> np.ndarray:
    """``tail_join`` less the value and the exact derivatives of the interpolant at the last row ``(3, C)``."""
    exact = _end_weights() @ table[-8:] * np.array([1.0, 1.0 / grid, 1.0 / grid ** 2])[:, None]
    return tail_join(table, grid) - exact


def table_values(table: Tensor, grid: float, r: Tensor, tail: float = TAIL_LENGTH,
                 ends: Optional[Tensor] = None) -> Tensor:
    """Interpolated table columns ``(P, C)`` at the distances ``r`` ``(P,)`` (bohr).

    ``table`` ``(n, C)`` holds the rows used, row ``k`` at ``(k + 1) * grid``. Inside the table an
    8-point Lagrange polynomial through rows ``i - 3, ..., i + 4`` (``i`` the row below ``r``),
    shifted to stay inside the table; from the last row to ``tail`` beyond it the quintic that joins
    the interpolant's value, slope and curvature to zero; zero further out. ``ends`` ``(3, C)`` is added to
    that value, slope and curvature (bohr; ``tail_correction``).
    """
    n = table.shape[0]
    x = r / grid - 1.0
    i0 = (torch.floor(x.detach()) - 3).clamp(0, n - 8).long()
    t = x - i0.to(x.dtype)
    rows = i0[:, None] + torch.arange(8, device=r.device)
    inner = (_lagrange8(t)[..., None] * table[rows]).sum(1)
    c = tail_polynomial(table, grid, tail, ends)
    r0 = n * grid
    s = ((r - r0) / tail).clamp(0.0, 1.0)[:, None]
    outer = c[0] + s * (c[1] + s * (c[2] + s * (c[3] + s * (c[4] + s * c[5]))))
    rr = r[:, None]
    return torch.where(rr < r0, inner, torch.where(rr < r0 + tail, outer, torch.zeros_like(outer)))


def tail_polynomial(table: Tensor, grid: float, tail: float = TAIL_LENGTH, ends: Optional[Tensor] = None) -> Tensor:
    """Coefficients ``(6, C)`` of the tail of ``table`` (see :func:`table_values`) in powers of
    ``(r - r0) / tail``, the second one halved."""
    n = table.shape[0]
    w = torch.as_tensor(_end_weights(), dtype=table.dtype, device=table.device)
    f = w @ table[n - 8:]                                       # (3, C) in rows
    f0, f1, f2 = f[0], f[1] * (tail / grid), f[2] * (tail / grid) ** 2
    if ends is not None:
        f0, f1, f2 = f0 + ends[0], f1 + ends[1] * tail, f2 + ends[2] * tail ** 2
    c3 = -10.0 * f0 - 6.0 * f1 - 1.5 * f2
    c4 = 15.0 * f0 + 8.0 * f1 + 1.5 * f2
    c5 = -6.0 * f0 - 3.0 * f1 - 0.5 * f2
    return torch.stack([f0, f1, 0.5 * f2, c3, c4, c5])


def stacked_values(rows: Tensor, start: Tensor, count: Tensor, grid: Tensor, coef: Tensor, r: Tensor,
                   tail: float = TAIL_LENGTH) -> Tensor:
    """:func:`table_values` of several tables at once: the rows of all tables ``rows`` ``(R, C)``, and for every
    distance ``r`` ``(P,)`` the first row ``start``, the row count ``count`` and the grid ``grid`` of its table
    and the coefficients of its tail ``coef`` ``(P, 6, C)`` (:func:`tail_polynomial`)."""
    x = r / grid - 1.0
    i0 = torch.minimum((torch.floor(x.detach()) - 3).clamp(min=0), (count - 8).to(x.dtype)).long()
    t = x - i0.to(x.dtype)
    index = (start + i0)[:, None] + torch.arange(8, device=r.device)
    inner = (_lagrange8(t)[..., None] * rows[index]).sum(1)
    r0 = count.to(r.dtype) * grid
    s = ((r - r0) / tail).clamp(0.0, 1.0)[:, None]
    c = coef.unbind(1)
    outer = c[0] + s * (c[1] + s * (c[2] + s * (c[3] + s * (c[4] + s * c[5]))))
    rr, r0 = r[:, None], r0[:, None]
    return torch.where(rr < r0, inner, torch.where(rr < r0 + tail, outer, torch.zeros_like(outer)))


def gamma_short(ta: Tensor, tb: Tensor, r: Tensor, equal_widths: float = EQUAL_WIDTHS,
                close_widths: float = CLOSE_WIDTHS) -> Tensor:
    """Short-range part ``S`` of the gamma function, ``gamma = 1 / R - S``, for atomic density widths
    ``ta``, ``tb`` and distances ``r > 0`` (atomic units; Elstner et al. 1998).

    Widths closer than ``EQUAL_WIDTHS`` take the form of equal widths at their mean. The form of unequal widths
    cancels strongly for close widths; its products are grouped as in DFTB+, and for widths closer than
    ``CLOSE_WIDTHS`` its exponentials are rounded correctly, so that both round alike.
    """
    equal = (ta - tb).abs() < equal_widths
    # equal widths: the symmetric form in their mean, whose partial derivatives in either width are halves
    tm = 0.5 * (ta + tb)
    same = torch.exp(-tm * r) * (1.0 / r + 11.0 * tm / 16.0 + 3.0 * tm * tm * r / 16.0 + tm ** 3 * r * r / 48.0)
    a2, b2 = ta * ta, tb * tb
    a3, b3 = a2 * ta, b2 * tb
    a4, b4, a6, b6 = a2 * a2, b2 * b2, a3 * a3, b3 * b3
    d = torch.where(equal, torch.ones_like(ta), a2 - b2)
    d2 = d * d
    d3 = d2 * d
    close = ((ta - tb).abs() < close_widths) & ~equal
    fa = 0.5 * b4 * ta / d2 - (b6 - 3.0 * b4 * a2) / (r * d3)
    fb = 0.5 * a4 * tb / d2 - (a6 - 3.0 * a4 * b2) / (-(r * d3))
    other = _exp_rounded(-ta * r, close) * fa + _exp_rounded(-tb * r, close) * fb
    return torch.where(equal, same, other)


def gamma_short_slope(ta: Tensor, tb: Tensor, r: Tensor, equal_widths: float = EQUAL_WIDTHS) -> Tensor:
    """The derivative of :func:`gamma_short` in its first width ``ta``; the one in ``tb`` is the same function
    with the widths exchanged (``S`` is symmetric in them)."""
    equal = (ta - tb).abs() < equal_widths
    tm = 0.5 * (ta + tb)
    em = torch.exp(-tm * r)
    same = em * (1.0 / r + 11.0 * tm / 16.0 + 3.0 * tm * tm * r / 16.0 + tm ** 3 * r * r / 48.0)
    d_same = -r * same + em * (11.0 / 16.0 + 6.0 * tm * r / 16.0 + tm * tm * r * r / 16.0)
    a2, b2 = ta * ta, tb * tb
    a3, b3 = a2 * ta, b2 * tb
    a4, b4, a6, b6 = a2 * a2, b2 * b2, a3 * a3, b3 * b3
    d = torch.where(equal, torch.ones_like(ta), a2 - b2)
    d2 = d * d
    d3, d4 = d2 * d, d2 * d2
    fa = 0.5 * b4 * ta / d2 - (b6 - 3.0 * b4 * a2) / (r * d3)
    dfa = b4 / (2.0 * d2) - 2.0 * b4 * a2 / d3 + (6.0 * ta * b4 * d + 6.0 * ta * (b6 - 3.0 * a2 * b4)) / (d4 * r)
    dfb = (2.0 * a3 * tb / d2 - 2.0 * a4 * ta * tb / d3
           + ((6.0 * a4 * ta - 12.0 * b2 * a3) * d - 6.0 * ta * (a6 - 3.0 * b2 * a4)) / (d4 * r))
    ea, eb = torch.exp(-ta * r), torch.exp(-tb * r)
    other = -r * ea * fa + ea * dfa + eb * dfb
    return torch.where(equal, 0.5 * d_same, other)


def _exp_rounded(x: Tensor, where: Tensor) -> Tensor:
    """``exp(x)``, correctly rounded (the C library's) at the entries ``where``, with the gradient of ``exp``."""
    e = torch.exp(x)
    if not torch.jit.is_scripting():
        if bool(where.any()) and x.dtype == torch.float64:
            values = x.detach()[where].cpu().tolist()
            exact = torch.tensor([math.exp(v) for v in values], dtype=x.dtype, device=x.device)
            correction = torch.zeros_like(e)
            correction[where] = exact - e.detach()[where]
            return e + correction
    return e


def gamma_onsite(ta: Tensor, tb: Tensor) -> Tensor:
    """The gamma function of two densities on one atom (``R = 0``); ``U`` when ``ta = tb``."""
    s = ta + tb
    return 0.5 * (ta * tb / s + ta * ta * tb * tb / s ** 3)


def short_range_cutoff(ta: float, tb: float, tolerance: float = 1e-10) -> float:
    """Distance (bohr) from which the short-range gamma of widths ``ta``, ``tb`` is left out of the pair sums.

    A bisection toward ``S = tolerance`` on ``[-0.1, b]``, with ``b = 32.1`` replaced by ``2 b - 0.1`` until ``S``
    there is below the tolerance, ends once ``S`` changes by less than the tolerance across the interval; the
    cutoff is its upper end.
    """
    ta_t, tb_t = torch.tensor([ta], dtype=torch.float64), torch.tensor([tb], dtype=torch.float64)

    def short(r: float) -> float:
        return float(gamma_short(ta_t, tb_t, torch.tensor([r], dtype=torch.float64)))

    lo, hi = -0.1, 32.1
    s_hi = short(hi)
    while s_hi > tolerance:
        hi = 2.0 * hi - 0.1
        s_hi = short(hi)
    s_lo = math.inf
    while s_lo - s_hi >= tolerance:
        mid = 0.5 * (lo + hi)
        s = short(mid)
        if s > tolerance:
            lo, s_lo = mid, s
        else:
            hi, s_hi = mid, s
    return hi


def gamma_yukawa(ta: Tensor, tb: Tensor, r: Tensor, omega: float) -> Tensor:
    """The Yukawa-screened gamma of two exponential densities with widths ``ta``, ``tb`` at distances ``r > 0``,
    ``exp(-omega r12) / r12`` between them (the inverse Fourier transform of Lutsker, Aradi and Niehaus, J.
    Chem. Phys. 143, 184107 (2015), eq 46, by partial fractions; equal widths by their own form)."""
    w2 = omega * omega
    equal = ta == tb
    a, b = ta, torch.where(equal, ta + 1.0, tb)                  # safe unequal widths where they are equal
    a2, b2 = a * a, b * b
    ea, eb, ew = torch.exp(-a * r), torch.exp(-b * r), torch.exp(-omega * r)
    d = (a2 - b2) ** 3 * (a2 - w2) ** 2 * (b2 - w2) ** 2
    num = (-0.5 * r * a * (a2 - b2) * (a2 - w2) ** 2 * (b2 - w2) * eb
           - 0.5 * r * b * (a2 - b2) * (a2 - w2) * (b2 - w2) ** 2 * ea
           + a * b * (a2 - b2) ** 3 * ew
           - a * b * (a2 - w2) ** 2 * (a2 - 3.0 * b2 + 2.0 * w2) * eb
           + a * b * (b2 - w2) ** 2 * (-3.0 * a2 + b2 + 2.0 * w2) * ea)
    unequal = a2 * a * b2 * b * num / (r * d)
    t, t2 = ta, ta * ta
    e = torch.exp(-t * r)
    num_eq = (-0.5 * r * t2 * t2 * (t2 - w2) * e - 0.125 * r * t2 * (t2 - w2) ** 2 * (r * t + 1.0) * e
              - r * (t2 - w2) ** 3 * (r * r * t2 + 3.0 * r * t + 3.0) * e / 48.0
              + t2 * t2 * t * (ew - e))
    same = t2 * t * num_eq / (r * (t2 - w2) ** 4)
    return torch.where(equal, same, unequal)


def gamma_yukawa_onsite(t: Tensor, omega: float) -> Tensor:
    """The Yukawa-screened gamma of a density with itself (``R = 0``)."""
    w = omega
    t2 = t * t
    return t2 * t * (5.0 * t ** 6 - 16.0 * t ** 5 * w + 15.0 * t2 * t2 * w ** 2 - 5.0 * t2 * w ** 4 + w ** 6) / (
        16.0 * (t2 - w * w) ** 4)


def lattice_translations(cell: Tensor, pbc: Tensor, reach: float) -> Tensor:
    """Integer translations ``(N_T, 3)`` whose lattice vectors cover every distance up to ``reach`` (bohr) along
    the periodic directions (rows of ``cell`` are the lattice vectors)."""
    c = cell.detach()
    volume = torch.det(c).abs()
    ranges = []
    for k in range(3):
        if not bool(pbc[k]):
            ranges.append(torch.zeros(1, dtype=torch.long, device=c.device))
            continue
        height = float(volume / torch.linalg.cross(c[(k + 1) % 3], c[(k + 2) % 3]).norm())
        n = int(math.ceil(reach / height))
        ranges.append(torch.arange(-n, n + 1, device=c.device))
    return torch.cartesian_prod(*ranges).reshape(-1, 3)


def ewald_matrix(pos: Tensor, cell: Tensor, accuracy: float = 1e-16, split: float = EWALD_SPLIT) -> Tensor:
    """The lattice sum of ``1 / |r_j - r_i + T|`` of point charges in a neutralizing background, ``(N, N)``
    (bohr, positions and lattice vectors as rows); the diagonal holds the interaction of a charge with its
    images. Ewald splitting (``alpha = split / V^(1/3)``) with real and reciprocal sums converged to
    ``accuracy``; the reciprocal sum by structure factors."""
    from xnn.common.models.eeq import ewald_alpha, reciprocal_vectors
    volume = torch.det(cell).abs()
    alpha = split / float(volume.detach()) ** (1.0 / 3.0)
    reach = ewald_alpha(1.0, accuracy) / alpha                   # erfc(alpha reach) = accuracy
    n = pos.shape[0]
    d0 = pos[None, :, :] - pos[:, None, :]                        # (N, N, 3) r_j - r_i
    # the nearest images of every pair: the sums run over all translations anyway
    frac = torch.linalg.solve(cell.detach().t(), d0.detach().reshape(-1, 3).t()).t().reshape(n, n, 3)
    d0 = d0 - torch.round(frac) @ cell
    extent = float(d0.detach().norm(dim=-1).max()) if n > 1 else 0.0
    shifts = lattice_translations(cell, torch.ones(3, dtype=torch.bool), reach + extent).to(pos.dtype) @ cell
    real = torch.zeros(n, n, dtype=pos.dtype, device=pos.device)
    chunk = max(1, EWALD_CHUNK // max(n * n, 1))                  # images per step, within a memory budget
    for t in range(0, shifts.shape[0], chunk):
        v = d0[:, :, None, :] + shifts[None, None, t:t + chunk]
        r = v.detach().norm(dim=-1)
        keep = (r > 0.0) & (r <= reach)
        if bool(keep.any()):
            # a charge and itself (no translation) leave the sum before the norm, whose curvature is singular there
            rs = torch.where(keep[..., None], v, torch.ones_like(v)).norm(dim=-1)
            real = real + torch.where(keep, torch.erfc(alpha * rs) / rs, torch.zeros_like(r)).sum(-1)
    _, gvec, g = reciprocal_vectors(cell, alpha, accuracy)
    phase = pos @ gvec.t()                                         # (N, N_G)
    c, s = torch.cos(phase), torch.sin(phase)
    recip = (c * g) @ c.t() + (s * g) @ s.t()
    eye = torch.eye(n, dtype=pos.dtype, device=pos.device)
    return real + recip - math.pi / (alpha * alpha * volume) - 2.0 * alpha / math.sqrt(math.pi) * eye


class _DFTBBasis:
    """Shells ``l = 0, ..., lmax`` of every atom and their atomic orbitals, for a batch of molecules."""

    def __init__(self, lmax: Dict[int, int], z: List[int], batch: List[int], n_struct: int, dtype, device):
        shell_atom, shell_l, atom_first = [], [], []
        ao_shell, ao_atom, ao_index, ao_struct = [], [], [], []
        counts = [0] * n_struct
        for a, (za, b) in enumerate(zip(z, batch)):
            atom_first.append(counts[b])
            for l in range(lmax[za] + 1):
                s = len(shell_atom)
                shell_atom.append(a)
                shell_l.append(l)
                for _ in range(2 * l + 1):
                    ao_shell.append(s)
                    ao_atom.append(a)
                    ao_struct.append(b)
                    ao_index.append(counts[b])
                    counts[b] += 1
        t = lambda v: torch.tensor(v, dtype=torch.long, device=device)  # noqa: E731
        self.z = z
        self.shell_atom, self.shell_l, self.shell_l_t = t(shell_atom), shell_l, t(shell_l)
        self.ao_shell, self.ao_atom, self.ao_struct, self.ao_index = t(ao_shell), t(ao_atom), t(ao_struct), t(ao_index)
        self.n_ao = counts
        self.nb = max(counts)
        self.shell_struct = [batch[a] for a in shell_atom]
        self.shell_struct_t = t(self.shell_struct)
        self.atom_first = t(atom_first)                          # first orbital of each atom in its structure


class DFTB(TightBinding):
    """Density-functional tight binding; see the module docstring.

    Parameters
    ----------
    parameters : str, path or SlaterKosterSet, optional
        A published Slater-Koster set (``"mio-1-1"``, ``"3ob-3-1"``, ``"ob2-1-1"``, ``"pbc-0-3"``,
        ``"matsci-0-3"`` or their short names), a folder of ``A-B.skf`` files, or a read set; defaults to
        the method's set.
    max_angular_momentum : dict, optional
        Highest shell (``"s"``, ``"p"``, ``"d"``, ``"f"`` or 0 to 3) per element symbol or atomic number.
        Elements not given take the value the set's README recommends, else the number of shells its
        homonuclear file states, else its highest shell with a level or electrons.
    scc : bool
        Self-consistent charges (DFTB2, DFTB3); ``False`` is DFTB1.
    shell_resolved : bool
        Charges and Hubbard parameters per shell; by default one per atom, the s shell's ``U``.
    third_order : str, optional
        ``"full"`` (DFTB3), ``"onsite"`` (the on-site terms only) or ``None``.
    hubbard_derivatives : dict, optional
        ``U^d`` (hartree per electron) per element; defaults to the set's README values.
    damping_exponent : float, optional
        ``zeta`` of the damping of gamma between hydrogen and other atoms; ``None`` for no damping.
    electronic_temperature : float
        Fermi smearing temperature in kelvin (default 0).
    scc_tolerance : float
        The cycle ends when no charge changes by more than this between iterations (default 1e-5).
    max_iter : int
        Iteration limit of the self-consistent cycle.
    mixing, history : float, int
        Damping and depth of the Anderson mixing of the charges.
    h5 : bool or dict, optional
        The H5 correction of hydrogen bonds: gamma between H and an element X is multiplied by
        ``1 + k_X exp(-(R - r0)^2 / (2 w^2))`` with ``r0 = r_scaling (R_X + R_H)`` and ``w = w_scaling (R_X + R_H)
        / (2 sqrt(2 ln 2))`` (Bondi radii, Angstrom). ``True`` takes the published values (``k`` of N 0.18,
        O 0.06, S 0.21; ``r_scaling`` 0.714, ``w_scaling`` 0.25); a dict replaces any of them (``scaling`` per
        element symbol or atomic number, negative for off). It replaces the hydrogen damping and scales the
        second-order gamma only (the third-order terms keep the unscaled one).
    spin_constants : dict, optional
        Collinear spin polarization with these constants ``W`` (hartree) per element symbol or atomic number:
        one value (it couples the atomic magnetizations), or the matrix of the element's shells as a flat
        list, the second shell running fastest (it couples the shell magnetizations). The energy gains
        ``1/2 sum_A sum_ll' W_ll' m_l m_l'`` of the Mulliken magnetizations ``m``; each structure's
        ``spin_multiplicity`` fixes the electrons of the two channels (by default one unpaired electron for odd
        counts).
    onsite_correction : dict, optional
        The onsite correction of the one-center integrals (Dominguez, Niehaus and Frauenheim, J. Phys. Chem. A
        119, 3535 (2015)): per element symbol or atomic number ``{"uu": [...], "ud": [...]}``, the same-spin
        and opposite-spin constants (hartree) of the element's shell pairs as flat matrices (the s-s entry is
        ignored). The dual density matrix becomes self-consistent with the charges.
    orbital_potential : dict, optional
        DFTB+U (Hourahine et al., J. Phys. Chem. A 111, 5671 (2007)) on chosen shells of spin-polarized runs:
        ``{"functional": "fll" or "psic", "shells": {element: [{"shells": [l, ...], "uj": U - J}, ...]}}``
        (angular momenta, hartree). Each block's spin occupation matrix ``n`` (the dual density of its
        orbitals) adds ``(U - J) / 2 Tr(n - n n)`` per spin (fully localized limit) or
        ``-(U - J) / 2 Tr(n n)`` (pseudo self-interaction correction).
    range_separation : float or bool, optional
        Long-range corrected exchange (LC-DFTB, Lutsker, Aradi and Niehaus, J. Chem. Phys. 143, 184107
        (2015)) with this range-separation parameter (inverse bohr). ``None`` (default) takes the value the
        set's files state (the ob2 set) and no exchange for other sets; ``False`` switches it off.
    kpoints : sequence of int or dict, optional
        k-points of periodic structures: a Monkhorst-Pack grid ``(n1, n2, n3)``, or ``{"grid": (n1, n2, n3),
        "shift": (s1, s2, s3)}`` with the points ``((m1 + s1) / n1, ...)`` in units of the reciprocal vectors.
        ``None`` (default) samples the Gamma point only. Range separation, the onsite correction and DFTB+U need
        the Gamma point.
    exchange_cutoff : float, optional
        Distance (bohr) from which the long-range exchange gamma is dropped in a periodic cell, where its lattice
        sum would not converge (a truncated Coulomb operator). ``None`` (default) takes the radius of the sphere
        inscribed in the cell.
    trainable : bool, str or sequence of str
        Parameters to train: ``"tables"`` (the Hamiltonian and overlap rows of every file), ``"repulsive"``
        (the spline coefficients and exponential heads), ``"onsite"`` (orbital energies) and ``"hubbard"``
        (Hubbard parameters); ``True`` trains all of them, ``False`` (default) none. :meth:`write_set`
        writes the trained set as Slater-Koster files.
    dispersion : dict, optional
        A D3 dispersion correction: ``{"model": "d3", "damping": "bj", "s8": ..., "a1": ..., "a2": ...}``
        (``s6`` 1 and ``s9`` 0 unless given) or ``"damping": "zero"`` with ``rs6``, ``s8`` and ``alp``;
        pairs within ``cutoff`` (default ``sqrt(9000)`` bohr) and coordination numbers within ``cutoff_cn``
        (default 40 bohr). Or a D4 correction with charges of the electronegativity equilibration model:
        ``{"model": "d4", "s8": ..., "s9": ..., "a1": ..., "a2": ...}`` (``s6`` 1, pairs within 64 bohr, triples
        within 40 bohr, covalent radii in the Bohr radius of the model). With D3, ``"hh_repulsion": True`` adds
        the repulsion of every pair of hydrogen atoms ``k / (1 + exp(e (R / r0 - 1)))`` (``k`` 4.784e-4 hartree,
        ``e`` 14.31, ``r0`` 2.35 Angstrom) of the DFTB3-D3H5 method (Rezac, J. Chem. Theory Comput. 13, 4804
        (2017)). The method has no dispersion by default.
    use_fast : bool or str, optional
        The fast path (see :meth:`set_use_fast`): ``"auto"`` (default) and ``True`` build the matrices on GPU
        kernels and evaluate small batches on a GPU with a copy of the model on the host; ``False`` is the reference.
    """

    method = ""
    #: Largest basis on a GPU evaluated on the host by the fast path: batches whose sum of squared basis sizes is
    #: at most its square (one such structure of this size, or many smaller ones).
    host_basis = 110
    #: Bohr radius (Angstrom) of the DFTB+ program, so that coordinates in Angstrom agree with it.
    bohr = 0.529177249
    #: Smallest smearing energy (hartree), as in the DFTB+ program: at zero temperature degenerate partly filled
    #: levels share their electrons and the energy includes their entropy term.
    minimum_kt = 1e-8
    #: Default parameter set of the method.
    default_parameters = "mio-1-1"
    scc_terms = ("second_order", "third_order", "spin", "exchange", "onsite", "orbital")
    reported_terms = ("second_order", "third_order", "spin", "exchange", "onsite", "orbital")
    trainable: Tuple[str, ...] = ()

    def __init__(self, parameters: Union[str, SlaterKosterSet, None] = None,
                 max_angular_momentum: Optional[Dict[Union[str, int], Union[str, int]]] = None,
                 scc: bool = True, shell_resolved: bool = False, third_order: Optional[str] = None,
                 hubbard_derivatives: Optional[Dict[Union[str, int], float]] = None,
                 damping_exponent: Optional[float] = None, electronic_temperature: float = 0.0,
                 scc_tolerance: float = 1e-5, max_iter: int = 100, mixing: float = 0.4, history: int = 8,
                 dispersion: Optional[Dict[str, Union[str, float]]] = None,
                 h5: Union[bool, Dict, None] = None,
                 spin_constants: Optional[Dict[Union[str, int], Union[float, List[float]]]] = None,
                 trainable: Union[bool, str, Tuple[str, ...], List[str]] = False,
                 range_separation: Union[float, bool, None] = None,
                 exchange_cutoff: Optional[float] = None,
                 kpoints: Union[Tuple[int, int, int], List[int], Dict, None] = None,
                 onsite_correction: Optional[Dict[Union[str, int], Dict[str, List[float]]]] = None,
                 orbital_potential: Optional[Dict] = None, use_fast: Union[bool, str] = "auto"):
        super().__init__()
        from xnn.common.data.elements import CHEMICAL_SYMBOLS
        if third_order not in (None, "full", "onsite"):
            raise ValueError(f"third_order must be 'full', 'onsite' or None, got {third_order!r}")
        if third_order is not None and not scc:
            raise ValueError("the third-order terms need self-consistent charges")
        self.skset = load_set(parameters or self.default_parameters)
        self.scc, self.shell_resolved, self.third_order = bool(scc), bool(shell_resolved), third_order
        self.damping_exponent = None if damping_exponent is None else float(damping_exponent)
        self.electronic_temperature = float(electronic_temperature)
        self.scc_tolerance, self.max_iter = float(scc_tolerance), int(max_iter)
        self.mixing, self.history = float(mixing), int(history)
        self.last_scf: Optional[Dict[str, Tensor]] = None
        self.set_use_fast(use_fast)
        self.cutoff = 0.1                                            # the model forms all atom pairs itself
        number = {s: z for z, s in enumerate(CHEMICAL_SYMBOLS)}
        key = lambda k: number[k] if isinstance(k, str) else int(k)  # noqa: E731
        recommended = self.skset.recommended()
        given = {key(k): ("spdf".index(v) if isinstance(v, str) else int(v))
                 for k, v in (max_angular_momentum or {}).items()}
        self.lmax: Dict[int, int] = {}
        for z in self.skset.elements():
            if z in given:
                self.lmax[z] = given[z]
            elif z in recommended["max_angular_momentum"]:
                self.lmax[z] = "spdf".index(recommended["max_angular_momentum"][z])
            else:
                self.lmax[z] = self._shells_of(self.skset.pair(z, z)) - 1
        self.supported = tuple(sorted(self.lmax))
        derivs = dict(recommended["hubbard_derivatives"])
        derivs.update({key(k): float(v) for k, v in (hubbard_derivatives or {}).items()})
        self.hubbard_derivatives = derivs
        self._tables: Dict[tuple, tuple] = {}
        self.h5 = self._h5_options(h5, key)
        if self.h5 is not None and self.damping_exponent is not None:
            raise ValueError("the H5 correction replaces the hydrogen damping: give one of them")
        self.spin_constants = None
        if spin_constants:
            if not scc:
                raise ValueError("spin polarization needs self-consistent charges")
            self.spin_constants = {key(k): np.atleast_1d(np.asarray(v, dtype=np.float64))
                                   for k, v in spin_constants.items()}
            for z, w in self.spin_constants.items():
                n = self.lmax.get(z, 0) + 1
                if w.size not in (1, n * n):
                    raise ValueError(f"spin constants of Z = {z}: {w.size} values, expected 1 or {n * n}")
        self.spin_polarized = self.spin_constants is not None
        if range_separation is None:
            stated = {self.skset.pair(z, z).range_separation for z in self.supported}
            omega = next(iter(stated))[1] if len(stated) == 1 and None not in stated else None
        else:
            omega = None if range_separation is False else float(range_separation)
        if omega is not None and (not scc or self.spin_polarized):
            raise ValueError("range separation needs self-consistent charges and no spin polarization")
        self.omega = omega
        self.exchange_cutoff = None if exchange_cutoff is None else float(exchange_cutoff)
        self.kpoints = None
        if kpoints is not None:
            grid = kpoints.get("grid") if isinstance(kpoints, dict) else kpoints
            shift = kpoints.get("shift", (0.0, 0.0, 0.0)) if isinstance(kpoints, dict) else (0.0, 0.0, 0.0)
            if len(grid) != 3 or len(shift) != 3 or min(int(g) for g in grid) < 1:
                raise ValueError(f"kpoints needs three grid sizes of at least 1 and three shifts, got {kpoints!r}")
            self.kpoints = {"grid": [int(g) for g in grid], "shift": [float(s) for s in shift]}
        self.onsite = None
        if onsite_correction:
            if not scc or self.spin_polarized:
                raise ValueError("the onsite correction needs self-consistent charges and no spin polarization")
            self.onsite = {}
            for k, v in onsite_correction.items():
                z = key(k)
                n = self.lmax.get(z, 0) + 1
                uu, ud = (np.asarray(v[name], dtype=np.float64).reshape(-1) for name in ("uu", "ud"))
                if uu.size != n * n or ud.size != n * n:
                    raise ValueError(f"onsite constants of Z = {z}: expected {n * n} values for uu and ud")
                lam = 0.5 * (uu + ud).reshape(n, n)                    # both spin channels carry half the density
                lam[0, 0] = 0.0
                self.onsite[z] = lam
        self.plus_u = None
        if orbital_potential:
            functional = str(orbital_potential.get("functional", "fll")).lower()
            if functional not in ("fll", "psic"):
                raise ValueError(f"orbital potential functional must be 'fll' or 'psic', got {functional!r}")
            if not self.spin_polarized:
                raise ValueError("DFTB+U needs spin polarization (spin_constants)")
            blocks = {key(k): [(tuple(int(l) for l in b["shells"]), float(b["uj"])) for b in v]
                      for k, v in orbital_potential.get("shells", {}).items()}
            self.plus_u = (functional, blocks)
        if self.spin_polarized and (omega is not None or self.onsite is not None):
            raise ValueError("range separation and the onsite correction work without spin polarization")
        self.density_mixing = omega is not None or self.onsite is not None or self.plus_u is not None
        self.dispersion = dict(dispersion) if dispersion else None
        options = dict(self.dispersion or {})
        self.hh_repulsion = bool(options.pop("hh_repulsion", False))
        self.d3 = self._dispersion_model(options) if self.dispersion else None
        names = TRAINABLE if trainable is True else ((trainable,) if isinstance(trainable, str) else
                                                     tuple(trainable or ()))
        unknown = sorted(set(names) - set(TRAINABLE))
        if unknown:
            raise ValueError(f"not trainable: {unknown}; choose from {list(TRAINABLE)}")
        self.trainable = tuple(n for n in TRAINABLE if n in names)
        self.params = torch.nn.ParameterDict(self._trained_parameters())

    @staticmethod
    def _h5_options(h5, key) -> Optional[Dict]:
        """The resolved ``h5`` option."""
        if not h5:
            return None
        out = {"r_scaling": H5_DEFAULTS["r_scaling"], "w_scaling": H5_DEFAULTS["w_scaling"],
               "scaling": dict(H5_DEFAULTS["scaling"])}
        if isinstance(h5, dict):
            out["r_scaling"] = float(h5.get("r_scaling", out["r_scaling"]))
            out["w_scaling"] = float(h5.get("w_scaling", out["w_scaling"]))
            out["scaling"].update({key(k): float(v) for k, v in h5.get("scaling", {}).items()})
        out["scaling"] = {z: k for z, k in out["scaling"].items() if k >= 0.0}
        missing = sorted(z for z in out["scaling"] if z not in BONDI_RADII)
        if missing:
            raise ValueError(f"no van der Waals radius for the H5 correction of Z = {missing}")
        return out

    @staticmethod
    def _dispersion_model(options: Dict[str, Union[str, float]]):
        """The D3 or D4 term of the ``dispersion`` option."""
        from xnn.common.models.d3 import BOHR, DFTD3
        opts = dict(options)
        kind = str(opts.pop("model", "d3")).lower()
        if kind == "d4":
            from xnn.common.models.d4 import DFTD4
            opts.setdefault("s6", 1.0)
            cutoff = {"cutoff_pair": 64.0, "cutoff_triple": 40.0, "cutoff_cn": 40.0, "cutoff_eeq_cn": 40.0}
            previous = torch.get_default_dtype()
            torch.set_default_dtype(torch.float64)
            try:
                d4 = DFTD4(**opts, **{k: v * BOHR for k, v in cutoff.items()})
            finally:
                torch.set_default_dtype(previous)
            # covalent radii tabulated in Angstrom, converted with the Bohr radius of the DFTB models
            d4.rcov.mul_(d4.bohr / DFTB.bohr)
            return d4
        if kind != "d3":
            raise ValueError(f"dispersion model must be 'd3' or 'd4', got {options.get('model')!r}")
        cutoff, cutoff_cn = float(opts.pop("cutoff", math.sqrt(9000.0))), float(opts.pop("cutoff_cn", 40.0))
        opts.setdefault("s6", 1.0)
        opts.setdefault("s9", 0.0)
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            return DFTD3(**opts, references="2024", cutoff_pair=cutoff * BOHR, cutoff_cn=cutoff_cn * BOHR,
                         cutoff_triple=cutoff_cn * BOHR)
        finally:
            torch.set_default_dtype(previous)

    @staticmethod
    def _shells_of(f: SlaterKosterFile) -> int:
        """Number of shells of an element without an explicit or recommended value."""
        if f.n_shells:
            return f.n_shells
        used = [l for l in range(len(f.energies)) if f.energies[l] != 0.0 or f.occupations[l] != 0.0]
        return max(used) + 1 if used else 1

    #: Options :meth:`from_config` reads from ``cfg.extra``.
    config_keys = ("parameters", "max_angular_momentum", "scc", "shell_resolved", "third_order",
                   "hubbard_derivatives", "damping_exponent", "electronic_temperature", "scc_tolerance", "max_iter",
                   "mixing", "history", "dispersion", "h5", "spin_constants", "trainable", "range_separation",
                   "exchange_cutoff", "kpoints", "onsite_correction", "orbital_potential", "use_fast")

    @classmethod
    def from_config(cls, cfg) -> "DFTB":
        """Build from a :class:`~xnn.common.config.schema.ModelConfig` (options in ``cfg.extra``, named as in
        the constructor)."""
        from xnn.common.config.options import model_options

        extra = model_options(cfg)
        return cls(**{k: extra[k] for k in cls.config_keys if k in extra})

    def set_use_fast(self, use_fast) -> "DFTB":
        """Choose the fast path: ``True``, ``False`` or ``"auto"``.

        On a GPU, batches up to ``host_basis`` (see :attr:`host_basis`) are evaluated by a copy of the model on
        the host, where they take less time than the GPU's kernel launches; their results return to the GPU.
        Larger batches build the overlap and Hamiltonian on the kernels of :mod:`~xnn.sqmnn.models.dftb_kernels`
        (first and second derivatives included). K-points take the reference.
        """
        from xnn.common.models.fast import resolve_use_fast
        self.use_fast = resolve_use_fast(use_fast)
        return self

    def _on_host(self, data) -> bool:
        """Whether a batch on a GPU is small enough for the host (fast path, see :attr:`host_basis`)."""
        if self.use_fast is False or data.pos.device.type != "cuda" or self.trainable:
            return False
        per_element = torch.zeros(max(self.lmax) + 1, dtype=torch.float64)
        for z, l in self.lmax.items():
            per_element[z] = (l + 1) ** 2
        z = data.atomic_numbers.long().cpu().clamp(max=per_element.shape[0] - 1)
        counts = torch.zeros(data.num_graphs, dtype=torch.float64).index_add(0, data.batch.cpu(), per_element[z])
        return float((counts * counts).sum()) <= float(self.host_basis) ** 2

    def _host_model(self) -> "DFTB":
        """This model on the host with the same options (kept apart from the registered modules, so that it stays
        there)."""
        options = (self.scc_tolerance, self.max_iter, self.mixing, self.history, self.electronic_temperature,
                   self.damping_exponent, self.use_fast)
        twin = self.__dict__.get("_twin")
        if twin is None or self.__dict__.get("_twin_options") != options:
            self.__dict__["_twin"] = None
            # device caches and the state of the current evaluation stay with this model
            kept = {k: self.__dict__.pop(k) for k in ("_lattice", "_kgrid", "_kweight", "_density_positions",
                                                      "_second_order_density", "_total_charge", "_last_h0", "_last_s")
                    if k in self.__dict__}
            kept["_tables"], kept["last_scf"] = self._tables, self.last_scf
            self._tables, self.last_scf = {}, None
            try:
                twin = copy.deepcopy(self).to("cpu")
            finally:
                self.__dict__.update(kept)
            self.__dict__["_twin"], self.__dict__["_twin_options"] = twin, options
        return twin.train(self.training)

    # parameters

    def _element(self, z: int) -> SlaterKosterFile:
        return self.skset.pair(z, z)

    def hubbard(self, z: int, l: int) -> float:
        """Hubbard parameter of shell ``l`` of element ``z`` (the s shell's unless shell-resolved)."""
        return float(self._element(z).hubbard[l if self.shell_resolved else 0])

    def reference_occupations(self, z: int) -> List[float]:
        occ = self._element(z).occupations
        return [float(occ[l]) for l in range(self.lmax[z] + 1)]

    def coordination_numbers(self, z: Tensor, pos: Tensor, batch: Tensor) -> Optional[Tensor]:
        return None

    def _trained_parameters(self) -> Dict[str, torch.nn.Parameter]:
        """The trained values (float64), from the set: tables and splines per file, levels and Hubbard
        parameters per element and angular momentum."""
        out: Dict[str, torch.nn.Parameter] = {}
        elements = self.supported
        p = lambda a: torch.nn.Parameter(torch.as_tensor(np.array(a, dtype=np.float64)))  # noqa: E731
        for za in elements:
            for zb in elements:
                if not self.skset.has(za, zb):
                    continue
                f = self.skset.pair(za, zb)
                if "tables" in self.trainable:
                    out[f"H_{za}_{zb}"] = p(f.hamiltonian[:f.n_grid - 1])
                    out[f"S_{za}_{zb}"] = p(f.overlap[:f.n_grid - 1])
                if "repulsive" in self.trainable and za <= zb and f.spline is not None:
                    out[f"spline_{za}_{zb}"] = p(f.spline.coefficients)
                    out[f"head_{za}_{zb}"] = p(f.spline.head)
        if "onsite" in self.trainable:
            table = np.zeros((max(elements) + 1, 4))
            for z in elements:
                e = self._element(z).energies
                table[z, :len(e)] = e
            out["onsite"] = p(table)
        if "hubbard" in self.trainable:
            table = np.zeros((max(elements) + 1, 4))
            for z in elements:
                u = self._element(z).hubbard
                table[z, :len(u)] = u
            out["hubbard"] = p(table)
        return out

    def write_set(self, folder: Union[str, Path]) -> None:
        """Write the model's Slater-Koster files, with the current values of the trained parameters, into
        ``folder`` (``A-B.skf`` for every pair of supported elements the set has); DFTB+ reads them."""
        import dataclasses as dc

        from ..common.skf import Spline, format_skf

        out = Path(folder)
        out.mkdir(parents=True, exist_ok=True)
        value = lambda name: self.params[name].detach().cpu().numpy()  # noqa: E731
        for za in self.supported:
            for zb in self.supported:
                if not self.skset.has(za, zb):
                    continue
                f = self.skset.pair(za, zb)
                changes = {}
                if "tables" in self.trainable:
                    n = f.n_grid - 1
                    changes["hamiltonian"] = np.concatenate([value(f"H_{za}_{zb}"), f.hamiltonian[n:]])
                    changes["overlap"] = np.concatenate([value(f"S_{za}_{zb}"), f.overlap[n:]])
                key = f"{min(za, zb)}_{max(za, zb)}"
                if "repulsive" in self.trainable and f.spline is not None:
                    changes["spline"] = Spline(f.spline.cutoff, tuple(float(x) for x in value(f"head_{key}")),
                                               f.spline.knots, value(f"spline_{key}"))
                if za == zb and "onsite" in self.trainable:
                    changes["energies"] = tuple(float(x) for x in value("onsite")[za, :len(f.energies)])
                if za == zb and "hubbard" in self.trainable:
                    changes["hubbard"] = tuple(float(x) for x in value("hubbard")[za, :len(f.hubbard)])
                text = format_skf(dc.replace(f, **changes))
                (out / self.skset.path(za, zb).name).write_text(text)

    def _integrals_trained(self, basis: _DFTBBasis) -> bool:
        return "tables" in self.trainable and torch.is_grad_enabled()

    def _table(self, za: int, zb: int, which: str, dtype, device) -> Tuple[Tensor, float, Tensor]:
        """The rows of one matrix of the file ``za-zb`` that the interpolation uses, its grid and the
        correction of its tail join (``tail_correction``)."""
        f = self.skset.pair(za, zb)
        if "tables" in self.trainable:
            table = self.params[f"{which}_{za}_{zb}"].to(device=device, dtype=dtype)
            ends = tail_correction(table.detach().to("cpu", torch.float64).numpy(), f.grid)
            return table, f.grid, torch.as_tensor(ends, dtype=dtype, device=device)
        k = (za, zb, which, dtype, device)
        if k not in self._tables:
            rows = (f.hamiltonian if which == "H" else f.overlap)[:f.n_grid - 1]
            self._tables[k] = (torch.as_tensor(rows, dtype=dtype, device=device),
                               torch.as_tensor(tail_correction(np.asarray(rows, dtype=np.float64), f.grid),
                                               dtype=dtype, device=device))
        table, ends = self._tables[k]
        return table, f.grid, ends

    def _stacked(self, pairs: List[Tuple[int, int]], dtype, device) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor,
                                                                            Tuple[str, ...]]:
        """The H tables, then the S tables, of the ordered element pairs ``pairs`` in the columns ``names`` (all
        files' columns): their rows one after the other, and per table the first row, the row count, the grid
        and the tail coefficients ``(6, C)``; returns those and ``names``."""
        key = ("stacked", tuple(pairs), dtype, device)
        if "tables" not in self.trainable and key in self._tables:
            return self._tables[key]
        names = EXTENDED_COLUMNS if any(self.skset.has(a, b) and self.skset.pair(a, b).extended
                                        for a, b in pairs) else COLUMNS
        rows, start, count, grid, coef = [], [], [], [], []
        first = 0
        for which in ("H", "S"):
            for a, b in pairs:
                if not self.skset.has(a, b):                          # no file: a placeholder no entry reads
                    table, g, ends = torch.zeros(8, len(names), dtype=dtype, device=device), 1.0, None
                    cols = names
                else:
                    table, g, ends = self._table(a, b, which, dtype, device)
                    cols = self.skset.pair(a, b).columns
                if ends is None:
                    ends = torch.zeros(3, len(cols), dtype=dtype, device=device)
                zero = torch.zeros_like(table[:, 0])
                full = torch.stack([table[:, cols.index(c)] if c in cols else zero for c in names], 1)
                full_ends = torch.stack([ends[:, cols.index(c)] if c in cols else torch.zeros_like(ends[:, 0])
                                         for c in names], 1)
                rows.append(full)
                start.append(first)
                count.append(full.shape[0])
                grid.append(g)
                coef.append(tail_polynomial(full, g, TAIL_LENGTH, full_ends))
                first += full.shape[0]
        as_long = lambda v: torch.tensor(v, dtype=torch.long, device=device)  # noqa: E731
        out = (torch.cat(rows), as_long(start), as_long(count), torch.tensor(grid, dtype=dtype, device=device),
               torch.stack(coef), names)
        if "tables" not in self.trainable:
            self._tables[key] = out
        return out

    def _range(self, za: int, zb: int) -> float:
        """Distance (bohr) beyond which the integrals of the pair vanish."""
        f = self.skset.pair(za, zb)
        return (f.n_grid - 1) * f.grid + TAIL_LENGTH

    # the integrals and the Hamiltonian

    def export_core(self) -> torch.nn.Module:
        """The scriptable core of :func:`~xnn.common.deploy.export_torchscript_potential` (see
        :mod:`~xnn.sqmnn.models.dftb_export`): this model's tables, one molecule per call."""
        from .dftb_export import DFTBExport
        return DFTBExport(self)

    def _make_basis(self, z: List[int], batch: List[int], n_struct: int, dtype, device) -> _DFTBBasis:
        basis = _DFTBBasis(self.lmax, z, batch, n_struct, dtype, device)
        kgrid, lattice = self.__dict__.get("_kgrid"), self.__dict__.get("_lattice")
        if kgrid is None or lattice is None:
            return basis
        # several k-points: a structure in a cell has one row of the matrices per k-point, holding the real form
        # of the complex matrices (the orbitals once for the real and once for the imaginary part)
        periodic = lattice[1].tolist()
        n_k = kgrid.shape[0]
        phys = list(basis.n_ao)
        atom_struct = torch.zeros(len(z), dtype=torch.long, device=device)
        atom_struct[basis.ao_atom] = basis.ao_struct
        rows: List[List[int]] = []
        row_struct, row_k = [], []
        for b in range(n_struct):
            rows.append(list(range(len(row_struct), len(row_struct) + (n_k if periodic[b] else 1))))
            row_struct += [b] * len(rows[b])
            row_k += list(range(n_k)) if periodic[b] else [0]
        kweight = self.__dict__["_kweight"].tolist()
        copies = [int(round(2 * sum(kweight))) if periodic[b] else 1 for b in range(n_struct)]
        row_weight = [kweight[k] if periodic[b] else 1.0 for b, k in zip(row_struct, row_k)]
        t = lambda v: torch.tensor(v, dtype=torch.long, device=device)  # noqa: E731
        # every orbital once per row of its structure, twice (real, imaginary part) in a cell
        b = basis.ao_struct
        cell = t([int(x) for x in periodic])[b].bool()
        reps = torch.where(cell, 2 * n_k, 1)
        src = torch.repeat_interleave(torch.arange(b.shape[0], device=device), reps)
        q = torch.arange(src.shape[0], device=device) - (torch.cumsum(reps, 0) - reps)[src]
        bs = b[src]
        on_cell = cell[src]
        row = t([r[0] for r in rows])[bs] + torch.where(on_cell, q // 2, torch.zeros_like(q))
        index = basis.ao_index[src] + torch.where(on_cell, q % 2, torch.zeros_like(q)) * t(phys)[bs]
        basis.ao_shell, basis.ao_atom = basis.ao_shell[src], basis.ao_atom[src]
        basis.ao_struct, basis.ao_index = row, index
        basis.atom_struct_t = atom_struct
        basis.n_phys, basis.copies = phys, copies
        basis.row_struct_t, basis.row_k, basis.row_first = t(row_struct), t(row_k), [r[0] for r in rows]
        basis.n_ao = [phys[b] * (2 if periodic[b] else 1) for b in row_struct]
        basis.nb = max(basis.n_ao)
        basis.row_weight_t = torch.tensor(row_weight, dtype=dtype, device=device)
        basis.ao_copies = (torch.tensor([copies[s] for s in row_struct], dtype=dtype, device=device)
                           / basis.row_weight_t)[row]
        return basis

    def forward(self, data) -> Dict[str, Tensor]:
        """Total energies (eV) and Mulliken charges of a batch of molecules or periodic structures (the latter at
        the Gamma point of their cell or on the grid of the ``kpoints`` option)."""
        cell, pbc = getattr(data, "cell", None), getattr(data, "pbc", None)
        lattice = None
        if cell is not None and pbc is not None and bool(pbc.any()):
            if bool((pbc.any(-1) & ~pbc.all(-1)).any()):
                raise NotImplementedError("DFTB handles molecules and structures periodic in all three directions")
            lattice = (cell.to(torch.float64) / self.bohr, pbc.all(-1))
        kgrid = None
        if lattice is not None and self.kpoints is not None and math.prod(self.kpoints["grid"]) > 1:
            if self.density_mixing:
                raise NotImplementedError("range separation, the onsite correction and DFTB+U need the Gamma point")
            axes = [(torch.arange(n, dtype=torch.float64) + s) / n
                    for n, s in zip(self.kpoints["grid"], self.kpoints["shift"])]
            kgrid = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
            kgrid, kweight = self._time_reversal(kgrid)
            self.__dict__["_kweight"] = kweight
        self.__dict__["_lattice"], self.__dict__["_kgrid"] = lattice, kgrid
        try:
            return super().forward(data)
        finally:
            self.__dict__.pop("_lattice", None)
            self.__dict__.pop("_kgrid", None)
            self.__dict__.pop("_kweight", None)

    @staticmethod
    def _time_reversal(kgrid: Tensor) -> Tuple[Tensor, Tensor]:
        """The k-points of a grid with ``-k`` merged into ``k`` (equal spectra, conjugate densities) and their
        weights (2 for a merged pair, 1 for a point that is its own inverse)."""
        keep: List[List[float]] = []
        weight: List[float] = []
        index: Dict[tuple, int] = {}
        for k in kgrid.tolist():
            key = tuple(round(x % 1.0, 12) % 1.0 for x in k)
            inverse = tuple(round(-x % 1.0, 12) % 1.0 for x in k)
            if inverse in index:
                weight[index[inverse]] += 1.0
                continue
            index[key] = len(keep)
            keep.append(k)
            weight.append(1.0)
        return torch.tensor(keep, dtype=kgrid.dtype), torch.tensor(weight, dtype=kgrid.dtype)

    def _begin(self, z: Tensor, pos: Tensor, batch: Tensor, n_struct: int, charge: Tensor) -> None:
        self.__dict__["_total_charge"] = charge                     # the dispersion charges need it

    def _pair_entries(self, atom_struct: Tensor, pos: Tensor, reach: float) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Atom pairs within ``reach`` (bohr): ``i, j``, the vectors from ``i`` to ``j`` and a flag of the images
        of an atom itself. A molecule gives its pairs ``i < j``; a periodic structure the pairs ``i < j`` with every
        lattice translation and the images ``i = j`` with every non-zero one."""
        device = pos.device
        lattice = self.__dict__.get("_lattice")
        periodic = torch.zeros(max(int(atom_struct.max()) + 1, 1), dtype=torch.bool, device=device)
        if lattice is not None:
            periodic = lattice[1].to(device)
        i, j = _all_pairs(atom_struct)
        keep = ~periodic[atom_struct[i]]
        i, j = i[keep], j[keep]
        parts = [(i, j, pos.index_select(0, j) - pos.index_select(0, i), torch.zeros_like(i, dtype=torch.bool))]
        for b in torch.nonzero(periodic).flatten().tolist():
            members = torch.nonzero(atom_struct == b).flatten()
            p = pos.index_select(0, members)
            m = members.shape[0]
            ii, jj = (t.reshape(-1) for t in torch.meshgrid(torch.arange(m, device=device),
                                                            torch.arange(m, device=device), indexing="ij"))
            cell_b = lattice[0][b].to(device=device, dtype=pos.dtype)
            d0 = p.index_select(0, jj) - p.index_select(0, ii)
            # the nearest image of every pair, so that few translations reach every image within reach
            d0 = d0 - torch.round(torch.linalg.solve(cell_b.detach().t(), d0.detach().t()).t()) @ cell_b
            extent = float(d0.detach().norm(dim=-1).max())
            n = lattice_translations(lattice[0][b], torch.ones(3, dtype=torch.bool), reach + extent)
            shifts = n.to(pos.dtype) @ lattice[0][b]
            d = d0[:, None, :] + shifts[None, :, :]
            r = d.detach().norm(dim=-1)
            zero = (n == 0).all(-1)
            mask = (r < reach) & ((ii < jj)[:, None] | ((ii == jj)[:, None] & ~zero[None, :]))
            pair, t = torch.nonzero(mask, as_tuple=True)
            parts.append((members[ii[pair]], members[jj[pair]], d[pair, t], (ii == jj)[pair]))
        return tuple(torch.cat([q[k] for q in parts]) for k in range(4))

    def _two_center(self, basis: _DFTBBasis, pos: Tensor, n_struct: int) -> Tuple[Tensor, Tensor]:
        """Dense overlap and Hamiltonian ``(B, nb, nb)``, one matrix per row of the basis (per structure, or per
        k-point of a structure in a cell)."""
        dtype, device = pos.dtype, pos.device
        z = basis.z
        n_atoms = len(z)
        atom_struct = getattr(basis, "atom_struct_t", None)
        if atom_struct is None:
            atom_struct = torch.zeros(n_atoms, dtype=torch.long, device=device)
            atom_struct[basis.ao_atom] = basis.ao_struct
        n_rows = len(basis.n_ao)
        reach = max((self._range(za, zb) for za in set(z) for zb in set(z) if self.skset.has(za, zb)), default=0.0)
        i, j, vec, itself = self._pair_entries(atom_struct, pos, reach)
        b_idx, rows, cols, h_vals, s_vals, once, entry = [], [], [], [], [], [], []
        # every table at once: the entries of all element pairs interpolate their own tables
        elements = sorted(set(z))
        pairs = [(a, b) for a in elements for b in elements]
        lookup = torch.zeros(max(elements) + 1, dtype=torch.long)
        lookup[torch.tensor(elements)] = torch.arange(len(elements))
        zt = lookup.to(device)[torch.tensor(z, dtype=torch.long, device=device)]          # element index per atom
        ab, ba = zt[i] * len(elements) + zt[j], zt[j] * len(elements) + zt[i]           # table of A-B and of B-A
        ranges = torch.tensor([self._range(a, b) if self.skset.has(a, b) else 0.0 for a, b in pairs], dtype=dtype,
                              device=device)
        sel = torch.nonzero(vec.detach().norm(dim=-1) < ranges[ab]).squeeze(1)
        ii, jj, ab, ba = i[sel], j[sel], ab[sel], ba[sel]
        stacked = self._stacked(pairs, dtype, device)
        table_rows, start, count, grid, coef, names = stacked
        lmax = torch.tensor([self.lmax[a] for a in elements], dtype=torch.long, device=device)
        la_max, lb_max = lmax[zt[ii]], lmax[zt[jj]]
        top = max(self.lmax[a] for a in elements)
        nb = basis.nb
        if self._fused_integrals(basis, pos):
            from . import dftb_kernels
            tables = dftb_kernels.PairTables(basis.atom_first[ii], basis.atom_first[jj], atom_struct[ii], itself[sel],
                                             la_max, lb_max, ab, ba)
            S, H = dftb_kernels.TwoCenter.apply(vec[sel], table_rows, coef, tables, (start, count, grid, names), top,
                                                n_rows, nb)
            return self._with_diagonal(S, H, basis)
        rr = vec[sel].norm(dim=-1)
        u = vec[sel] / rr[:, None]
        n_pairs = len(pairs)
        values = {}
        for side, ids in (("ab", ab), ("ba", ba)):
            for k, which in enumerate(("H", "S")):
                p = ids + k * n_pairs
                values[side, which] = stacked_values(table_rows, start[p], count[p], grid[p], coef[p], rr)
        for la in range(top + 1):
            for lb in range(top + 1):
                m = torch.nonzero((la_max >= la) & (lb_max >= lb)).squeeze(1)
                lo, hi = min(la, lb), max(la, lb)
                idx = [names.index(f"{LETTERS[lo]}{LETTERS[hi]}{k}") for k in range(lo + 1)]
                src, sign = ("ab", 1.0) if la <= lb else ("ba", (-1.0) ** (la + lb))
                oa = basis.atom_first[ii[m]] + la * la
                ob = basis.atom_first[jj[m]] + lb * lb
                ra = oa[:, None, None] + torch.arange(2 * la + 1, device=device)[None, :, None]
                cb = ob[:, None, None] + torch.arange(2 * lb + 1, device=device)[None, None, :]
                ra, cb = torch.broadcast_tensors(ra, cb)
                bb = atom_struct[ii[m]][:, None, None].expand_as(ra)
                for which, store in (("H", h_vals), ("S", s_vals)):
                    v = sign * values[src, which][m][:, idx]
                    store.append(shell_block(u[m], v, la, lb).reshape(-1))
                b_idx.append(bb.reshape(-1))
                rows.append(ra.reshape(-1))
                cols.append(cb.reshape(-1))
                once.append(itself[sel[m]][:, None, None].expand_as(ra).reshape(-1))
                entry.append(sel[m][:, None, None].expand_as(ra).reshape(-1))
        S = torch.zeros(n_rows, nb, nb, dtype=dtype, device=device)
        H = torch.zeros(n_rows, nb, nb, dtype=dtype, device=device)
        if b_idx:
            bi, ri, ci = torch.cat(b_idx), torch.cat(rows), torch.cat(cols)
            hv, sv = torch.cat(h_vals), torch.cat(s_vals)
            # pairs of different atoms fill both blocks; the images of an atom (both T and -T present) fill its own
            # block once
            pair = ~torch.cat(once)
            copies = getattr(basis, "copies", None)
            folded = torch.ones_like(pair) if copies is None else torch.as_tensor(copies, device=device)[bi] == 1
            fb, fr, fc, fp = bi[folded], ri[folded], ci[folded], pair[folded]
            if copies is not None:
                fb = torch.as_tensor(basis.row_first, device=device)[fb]           # the single row of the structure
            S = S.index_put((fb, fr, fc), sv[folded], accumulate=True).index_put(
                (fb[fp], fc[fp], fr[fp]), sv[folded][fp], accumulate=True)
            H = H.index_put((fb, fr, fc), hv[folded], accumulate=True).index_put(
                (fb[fp], fc[fp], fr[fp]), hv[folded][fp], accumulate=True)
            if not bool(folded.all()):
                ent = torch.cat(entry)[~folded]
                S, H = self._scatter_kpoints(S, H, basis, bi[~folded], ri[~folded], ci[~folded], sv[~folded],
                                             hv[~folded], pair[~folded], pos, i[ent], j[ent], vec[ent])
        return self._with_diagonal(S, H, basis)

    def _fused_integrals(self, basis: _DFTBBasis, pos: Tensor) -> bool:
        """Whether the two-center matrices take the kernels of :mod:`~xnn.sqmnn.models.dftb_kernels`: the fast path
        on a GPU, except with k-points (the kernels give first and second derivatives in the positions and the
        tables)."""
        if self.use_fast is False or not pos.is_cuda:
            return False
        if getattr(basis, "copies", None) is not None:
            return False
        from .dftb_kernels import available
        return available(pos.device)

    def _with_diagonal(self, S: Tensor, H: Tensor, basis: _DFTBBasis) -> Tuple[Tensor, Tensor]:
        """``S`` and ``H`` with their diagonals: unit overlaps (padding included) and the orbital energies."""
        z, dtype, device, n_rows, nb = basis.z, S.dtype, S.device, S.shape[0], basis.nb
        if "onsite" in self.trainable:
            zs = torch.tensor([z[a] for a in basis.shell_atom.tolist()], device=device)
            levels = self.params["onsite"].to(device=device, dtype=dtype)[zs, basis.shell_l_t]
        else:
            levels = torch.tensor([self._element(z[a]).energies[basis.shell_l[s]]
                                   for s, a in enumerate(basis.shell_atom.tolist())], dtype=dtype, device=device)
        diag_h = torch.zeros(n_rows, nb, dtype=dtype, device=device)
        diag_h[basis.ao_struct, basis.ao_index] = levels[basis.ao_shell]
        diag_s = torch.ones(n_rows, nb, dtype=dtype, device=device)        # padding keeps a unit overlap
        return S + torch.diag_embed(diag_s), H + torch.diag_embed(diag_h)

    def _scatter_kpoints(self, S: Tensor, H: Tensor, basis: _DFTBBasis, b: Tensor, row: Tensor, col: Tensor,
                         s_val: Tensor, h_val: Tensor, pair: Tensor, pos: Tensor, ai: Tensor, aj: Tensor,
                         vec: Tensor) -> Tuple[Tensor, Tensor]:
        """Add the integrals of structures sampled at several k-points: per k-point the real form
        ``[[A, -B], [B, A]]`` of the complex matrix ``A + iB = sum_T M(T) exp(2 pi i k.n)`` (``n`` the lattice
        translation of an entry), in the row of the basis of that k-point."""
        lattice = self.__dict__["_lattice"]
        cells = lattice[0].to(device=pos.device, dtype=pos.dtype)
        kgrid = self.__dict__["_kgrid"].to(device=pos.device, dtype=pos.dtype)
        # lattice translation of every element, in units of the cell vectors
        shift = (vec - (pos.index_select(0, aj) - pos.index_select(0, ai))).detach()
        n = torch.round(torch.linalg.solve(cells[b].transpose(-1, -2), shift.unsqueeze(-1)).squeeze(-1))
        phase = 2.0 * math.pi * (n @ kgrid.T)                                          # (E, Nk)
        cos, sin = torch.cos(phase), torch.sin(phase)
        nb = torch.as_tensor(basis.n_phys, device=pos.device)[b][:, None]
        first = torch.as_tensor(basis.row_first, device=pos.device)[b][:, None]
        bb = first + torch.arange(kgrid.shape[0], device=pos.device)[None, :]          # the row of each k-point
        r, c = row[:, None].expand_as(bb), col[:, None].expand_as(bb)
        width = int(nb.max())
        p = pair
        out = []
        for M, v in ((S, s_val), (H, h_val)):
            # real and imaginary parts of the complex matrices, then their real forms row by row
            re, im = v[:, None] * cos, v[:, None] * sin
            A = M.new_zeros(M.shape[0], width, width).index_put((bb, r, c), re, accumulate=True)
            A = A.index_put((bb[p], c[p], r[p]), re[p], accumulate=True)
            B = M.new_zeros(M.shape[0], width, width).index_put((bb, r, c), im, accumulate=True)
            B = B.index_put((bb[p], c[p], r[p]), -im[p], accumulate=True)
            for s in torch.unique(b).tolist():
                n = int(basis.n_phys[s])
                rows = torch.arange(basis.row_first[s], basis.row_first[s] + kgrid.shape[0], device=pos.device)
                a, i = A[rows, :n, :n], B[rows, :n, :n]
                block = torch.cat([torch.cat([a, -i], -1), torch.cat([i, a], -1)], -2)
                M = M.index_put((rows[:, None, None], torch.arange(2 * n, device=pos.device)[None, :, None],
                                 torch.arange(2 * n, device=pos.device)[None, None, :]), block, accumulate=True)
            out.append(M)
        return out[0], out[1]

    def _integrals(self, basis: _DFTBBasis, pos: Tensor, n_struct: int) -> Tuple[Tensor, Tensor, Tensor]:
        S, H = self._two_center(basis, pos, n_struct)
        self.__dict__["_last_h0"] = (pos, H)
        self.__dict__["_last_s"] = (pos, S)
        empty = S.new_zeros(S.shape[0], 0, basis.nb, basis.nb)
        return S, empty, empty

    def _hamiltonian(self, z: List[int], basis: _DFTBBasis, pos: Tensor, S: Tensor, cn: Optional[Tensor],
                     n_struct: int) -> Tensor:
        cached = self.__dict__.pop("_last_h0", None)
        if cached is not None and cached[0] is pos:
            return cached[1]
        return self._two_center(basis, pos, n_struct)[1]

    # the charge-dependent energy

    def _electron_counts(self, n_el: List[int], multiplicity: Optional[List[int]], dtype, device
                         ) -> Tuple[Tensor, Tensor]:
        if not self.spin_polarized:
            # one Fermi level for all electrons (the smearing is never zero): equal channels
            half = torch.tensor([n / 2.0 for n in n_el], dtype=dtype, device=device)
            return half, half.clone()
        return super()._electron_counts(n_el, multiplicity, dtype, device)

    def _scc_threshold(self) -> float:
        return self.scc_tolerance if self.scc else math.inf

    def _scc_error(self, r: Tensor, owner: Tensor, count: Tensor) -> Tensor:
        return torch.zeros_like(count).index_reduce(0, owner, r.detach().abs(), "amax", include_self=True)

    def scc_context(self, z: Tensor, pos: Tensor, basis: _DFTBBasis, cn, batch: Tensor, n_struct: int) -> dict:
        """Gamma (and Gamma of the third order) between the charge sites: atoms, or shells when resolved."""
        if not self.scc:
            return {"n_struct": n_struct}
        dtype, device = pos.dtype, pos.device
        outer = torch.is_grad_enabled()
        spin = self._spin_context(basis, dtype, device) if self.spin_polarized else {}
        zl = basis.z
        if self.shell_resolved:
            site_atom = basis.shell_atom
            site_u = [self.hubbard(zl[a], l) for a, l in zip(basis.shell_atom.tolist(), basis.shell_l)]
            shell_site = torch.arange(len(basis.shell_l), device=device)
        else:
            site_atom = torch.arange(len(zl), device=device)
            site_u = [self.hubbard(za, 0) for za in zl]
            shell_site = basis.shell_atom
        u = torch.tensor(site_u, dtype=dtype, device=device)
        if "hubbard" in self.trainable:
            site_z = torch.tensor([zl[a] for a in site_atom.tolist()], device=device)
            site_l = basis.shell_l_t if self.shell_resolved else torch.zeros_like(site_z)
            u = self.params["hubbard"].to(device=device, dtype=dtype)[site_z, site_l]
        n_sites = u.shape[0]
        site_struct = batch.index_select(0, site_atom)
        si, sj = _all_pairs(site_struct)
        hydrogen = torch.tensor([self._element(za).mass < HYDROGEN_MASS for za in zl], device=device)
        with torch.enable_grad():
            # Gamma differentiates gamma in the Hubbard parameters: aliases of them (leaves when not trained)
            ua, ub = u.index_select(0, si), u.index_select(0, sj)
            if not ua.requires_grad:
                ua, ub = ua.detach().requires_grad_(True), ub.detach().requires_grad_(True)
            short, h, r, same, damp = self._gamma_parts(ua, ub, site_atom[si], site_atom[sj], pos, hydrogen)
            f = self._h5_factor(site_atom[si], site_atom[sj], r, zl)
            gam = torch.where(same, gamma_onsite(3.2 * ua, 3.2 * ub), (1.0 / r - short * h) * f)
            out = {"si": si, "sj": sj, "gamma": gam, "onsite": u, "site_atom": site_atom, "n_struct": n_struct,
                   "shell_site": shell_site, "n_sites": n_sites, "site_struct": site_struct}
            if self.third_order is not None:
                derivs = torch.tensor([self.hubbard_derivatives.get(zl[a], 0.0) for a in site_atom.tolist()],
                                      dtype=dtype, device=device)
                out["Gamma_onsite"] = 0.5 * derivs
                if self.third_order == "full":
                    if bool(same.any()):
                        raise NotImplementedError("the full third order needs one charge per atom")
                    sa, sb = torch.autograd.grad(short.sum(), (ua, ub), create_graph=outer)
                    # equal Hubbard parameters: the derivative of the short-range part S(U, U) in the common U
                    # (twice the partial one), as the method defines Gamma for that case
                    close = 3.2 * (ua.detach() - ub.detach()).abs() < EQUAL_WIDTHS
                    both = torch.where(close, 2.0, 1.0).to(dtype)
                    dh = self._damping_slope(ua, ub, r, damp) * h                    # dh / dU_a = dh / dU_b
                    # the H5 factor scales gamma in the second-order energy only, not Gamma
                    out["Gamma_ab"] = -(both * sa * h + short * dh) * derivs.index_select(0, si)
                    out["Gamma_ba"] = -(both * sb * h + short * dh) * derivs.index_select(0, sj)
            lattice = self.__dict__.get("_lattice")
            if lattice is not None and bool(lattice[1].any()):
                self._periodic_gamma(out, pos, batch, u, site_atom, hydrogen, zl, lattice, outer)
        out.update(spin)
        if self.density_mixing:
            out.update(self._exchange_context(basis, pos, batch, n_struct, u if not self.shell_resolved else None))
        if self.onsite is not None:
            out.update(self._onsite_context(basis, n_struct, dtype, device))
        if self.plus_u is not None:
            out.update(self._plus_u_context(basis, n_struct, dtype, device))
        if not outer:
            out = {k: (v.detach() if isinstance(v, Tensor) else v) for k, v in out.items()}
        return out

    def _gamma_parts(self, ua: Tensor, ub: Tensor, ai: Tensor, aj: Tensor, pos: Tensor, hydrogen: Tensor):
        """The short-range part ``S`` and the damping factor ``h`` of gamma between sites with Hubbard
        parameters ``ua``, ``ub`` on atoms ``ai``, ``aj`` (``gamma = 1 / R - S h``; ``h = 1`` unless one atom
        is hydrogen and the damping is on), the distances (1 where zero), the zero-distance mask and the
        mask of damped pairs. ``S`` is zero from the pair's cutoff on."""
        r = (pos.index_select(0, aj) - pos.index_select(0, ai)).norm(dim=-1)
        same = r == 0.0
        rs = torch.where(same, torch.ones_like(r), r)
        short = gamma_short(3.2 * ua, 3.2 * ub, rs)
        short = torch.where(rs < self._short_cutoff(ua, ub), short, torch.zeros_like(short))
        h = torch.ones_like(rs)
        damp = torch.zeros_like(same)
        if self.damping_exponent is not None:
            damp = hydrogen.index_select(0, ai) | hydrogen.index_select(0, aj)
            h = torch.where(damp, torch.exp(-(0.5 * (ua + ub)) ** self.damping_exponent * rs * rs), h)
        return short, h, rs, same, damp

    def _short_cutoff(self, ua: Tensor, ub: Tensor) -> Tensor:
        """The cutoff (bohr) of the short-range gamma between sites with Hubbard parameters ``ua``, ``ub``."""
        cache = self.__dict__.setdefault("_cutoffs", {})
        if ua.numel() == 0:
            return torch.zeros_like(ua)
        kinds, inverse = torch.unique(torch.cat([ua.detach(), ub.detach()]), return_inverse=True)
        values = kinds.tolist()
        for a in values:
            for b in values:
                if (a, b) not in cache:
                    cache[(a, b)] = short_range_cutoff(3.2 * a, 3.2 * b)
        table = torch.tensor([[cache[(a, b)] for b in values] for a in values], dtype=ua.dtype, device=ua.device)
        n = ua.shape[0]
        return table[inverse[:n], inverse[n:]]

    def _periodic_gamma(self, out: dict, pos: Tensor, batch: Tensor, u: Tensor, site_atom: Tensor, hydrogen: Tensor,
                        zl: List[int], lattice, outer: bool) -> None:
        """Gamma (and Gamma of the third order) of the periodic structures at the Gamma point, in place of the
        molecular values: the Ewald sum of ``1 / R`` between the atoms of the sites plus the short-range rest summed
        over images (``(1/R - S h) f - 1/R``); the images of an atom add to the block of its own sites."""
        dtype, device = pos.dtype, pos.device
        cell, periodic = lattice[0].to(device), lattice[1].to(device)
        n_atoms, n_sites = pos.shape[0], u.shape[0]
        kinds = torch.unique(u.detach())
        reach = float(self._short_cutoff(kinds.repeat_interleave(len(kinds)), kinds.repeat(len(kinds))).max())
        ai, aj, vec, itself = self._pair_entries(batch, pos, reach)
        keep = periodic[batch.index_select(0, ai)]
        ai, aj, vec, itself = ai[keep], aj[keep], vec[keep], itself[keep]
        # every pair of sites of the two atoms of an entry
        count = torch.zeros(n_atoms, dtype=torch.long, device=device).index_add(
            0, site_atom, torch.ones_like(site_atom))
        width = max(int(count.max()), 1)
        first = torch.cumsum(count, 0) - count
        rank = torch.arange(n_sites, device=device) - first.index_select(0, site_atom)
        table = torch.full((n_atoms, width), -1, dtype=torch.long, device=device)
        table[site_atom, rank] = torch.arange(n_sites, device=device)
        s = table.index_select(0, ai)[:, :, None].expand(-1, width, width)
        t = table.index_select(0, aj)[:, None, :].expand(-1, width, width)
        valid = (s >= 0) & (t >= 0)
        entry = torch.arange(ai.shape[0], device=device)[:, None, None].expand(-1, width, width)[valid]
        i, j = s[valid], t[valid]
        ai, aj, itself = ai[entry], aj[entry], itself[entry]
        r = vec.norm(dim=-1)[entry]
        within = r < self._short_cutoff(u.index_select(0, i), u.index_select(0, j))
        i, j, ai, aj, itself, r = i[within], j[within], ai[within], aj[within], itself[within], r[within]
        ua, ub = u.index_select(0, i), u.index_select(0, j)
        if not ua.requires_grad:
            ua, ub = ua.detach().requires_grad_(True), ub.detach().requires_grad_(True)
        short = gamma_short(3.2 * ua, 3.2 * ub, r)
        h = torch.ones_like(r)
        damp = torch.zeros_like(itself)
        if self.damping_exponent is not None:
            damp = hydrogen.index_select(0, ai) | hydrogen.index_select(0, aj)
            h = torch.where(damp, torch.exp(-(0.5 * (ua + ub)) ** self.damping_exponent * r * r), h)
        f = self._h5_factor(ai, aj, r, zl)
        rest = -short * h * f + (f - 1.0) / r
        A = torch.zeros(n_atoms, n_atoms, dtype=dtype, device=device)
        for b in torch.nonzero(periodic).flatten().tolist():
            members = torch.nonzero(batch == b).flatten()
            block = ewald_matrix(pos.index_select(0, members), cell[b])
            A = A.index_put((members[:, None].expand_as(block), members[None, :].expand_as(block)), block)
        G = A.index_select(0, site_atom).index_select(1, site_atom)
        G = G.index_put((i, j), rest, accumulate=True)                 # images of an atom: its own block
        si, sj = out["si"], out["sj"]
        on_lattice = periodic[batch.index_select(0, site_atom)]
        both_periodic = on_lattice.index_select(0, si)
        same_atom = site_atom.index_select(0, si) == site_atom.index_select(0, sj)
        lattice_gamma = G[si, sj] + torch.where(same_atom, out["gamma"], torch.zeros_like(out["gamma"]))
        out["gamma"] = torch.where(both_periodic, lattice_gamma, out["gamma"])
        out["onsite"] = torch.where(on_lattice, u + G.diagonal(), out["onsite"])
        if self.third_order == "full":
            derivs = torch.tensor([self.hubbard_derivatives.get(zl[a], 0.0) for a in site_atom.tolist()],
                                  dtype=dtype, device=device)
            sa, sb = torch.autograd.grad(short.sum(), (ua, ub), create_graph=outer)
            both = torch.where(3.2 * (ua.detach() - ub.detach()).abs() < EQUAL_WIDTHS, 2.0, 1.0).to(dtype)
            dh = self._damping_slope(ua, ub, r, damp) * h
            ga = -(both * sa * h + short * dh) * derivs.index_select(0, i)
            gb = -(both * sb * h + short * dh) * derivs.index_select(0, j)
            pair = ~itself
            zeros = torch.zeros(n_sites, n_sites, dtype=dtype, device=device)
            GA = zeros.index_put((i[pair], j[pair]), ga[pair], accumulate=True)
            GB = zeros.index_put((i[pair], j[pair]), gb[pair], accumulate=True)
            out["Gamma_ab"] = torch.where(both_periodic, GA[si, sj], out["Gamma_ab"])
            out["Gamma_ba"] = torch.where(both_periodic, GB[si, sj], out["Gamma_ba"])
            images = torch.zeros(n_sites, dtype=dtype, device=device).index_add(0, i[itself], ga[itself])
            out["Gamma_onsite"] = torch.where(on_lattice, out["Gamma_onsite"] + images, out["Gamma_onsite"])

    def _h5_factor(self, ai: Tensor, aj: Tensor, r: Tensor, z: List[int]) -> Tensor:
        """The H5 factor of gamma between atoms ``ai`` and ``aj`` at distances ``r`` (bohr); 1 without H5."""
        if self.h5 is None:
            return torch.ones_like(r)
        zt = torch.tensor(z, device=r.device)
        za, zb = zt.index_select(0, ai), zt.index_select(0, aj)
        other = torch.where(za == 1, zb, za)
        pair = (za == 1) ^ (zb == 1)
        k = torch.zeros_like(r)
        radius = torch.zeros_like(r)
        for zx, kx in self.h5["scaling"].items():
            sel = pair & (other == zx)
            k = torch.where(sel, torch.full_like(r, kx), k)
            radius = torch.where(sel, torch.full_like(r, BONDI_RADII[zx] + BONDI_RADII[1]), radius)
        r0 = self.h5["r_scaling"] * radius
        w = self.h5["w_scaling"] * radius / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        w = torch.where(k > 0.0, w, torch.ones_like(w))
        x = r * self.bohr - r0
        return 1.0 + k * torch.exp(-x * x / (2.0 * w * w))

    def _damping_slope(self, ua: Tensor, ub: Tensor, r: Tensor, damp: Tensor) -> Tensor:
        """``(dh / dU_a) / h`` of the damping factor (zero for undamped pairs)."""
        if self.damping_exponent is None:
            return torch.zeros_like(r)
        z = self.damping_exponent
        slope = -0.5 * z * (0.5 * (ua + ub)) ** (z - 1.0) * r * r
        return torch.where(damp, slope, torch.zeros_like(slope))

    def _terms(self, q_shell: Tensor, mu: Tensor, theta: Tensor, ctx: dict) -> Dict[str, Tensor]:
        zero = q_shell.new_zeros(int(ctx["n_struct"]))
        if not self.scc:
            return {"second_order": zero, "third_order": zero, "spin": zero, "exchange": zero, "onsite": zero,
                    "orbital": zero}
        e_spin = zero
        if "m_shell" in ctx and self.spin_polarized:
            m = ctx["m_shell"]
            e_spin = zero.index_add(0, ctx["spin_struct"], 0.5 * m.index_select(0, ctx["spin_i"])
                                    * ctx["spin_w"] * m.index_select(0, ctx["spin_j"]))
        q = q_shell.new_zeros(ctx["n_sites"]).index_add(0, ctx["shell_site"], q_shell)
        si, sj, owner = ctx["si"], ctx["sj"], ctx["site_struct"]
        qi, qj = q.index_select(0, si), q.index_select(0, sj)
        e2 = zero.index_add(0, owner.index_select(0, si), ctx["gamma"] * qi * qj)
        e2 = e2.index_add(0, owner, 0.5 * ctx["onsite"] * q * q)
        e3 = zero
        if self.third_order is not None:
            # q is the charge (electron deficit): dq = -q in the expansion's sign convention
            e3 = zero.index_add(0, owner, -ctx["Gamma_onsite"] * q ** 3 / 3.0)
            if self.third_order == "full":
                e3 = e3.index_add(0, owner.index_select(0, si),
                                  -(ctx["Gamma_ab"] * qi * qi * qj + ctx["Gamma_ba"] * qj * qj * qi) / 3.0)
        e_x = e_ons = e_u = zero
        if self.density_mixing and "P" in ctx and self.plus_u is not None:
            e_u = self._orbital_potential(ctx["P"], ctx)[0]
        elif self.density_mixing and "P" in ctx:
            dP = ctx["P"] - ctx["p0"]
            if self.omega is not None:
                e_x = 0.5 * (dP * self._exchange_fock(dP, ctx)).sum((-1, -2))
            if self.onsite is not None:
                e_ons = self._onsite(ctx["P"], ctx)[0]
        return {"second_order": e2, "third_order": e3, "spin": e_spin, "exchange": e_x, "onsite": e_ons,
                "orbital": e_u}

    def potentials(self, q_shell: Tensor, mu: Tensor, theta: Tensor, ctx: dict) -> Tuple[Tensor, Tensor, Tensor]:
        """Derivatives of the charge-dependent energy in the shell charges (closed form; no moments)."""
        zm, zt = torch.zeros_like(mu), torch.zeros_like(theta)
        if not self.scc:
            return torch.zeros_like(q_shell), zm, zt
        q = q_shell.new_zeros(ctx["n_sites"]).index_add(0, ctx["shell_site"], q_shell)
        si, sj, gam = ctx["si"], ctx["sj"], ctx["gamma"]
        qi, qj = q.index_select(0, si), q.index_select(0, sj)
        v = ctx["onsite"] * q
        v = v.index_add(0, si, gam * qj).index_add(0, sj, gam * qi)
        if self.third_order is not None:
            v = v - ctx["Gamma_onsite"] * q * q
            if self.third_order == "full":
                gab, gba = ctx["Gamma_ab"], ctx["Gamma_ba"]
                v = v.index_add(0, si, -(2.0 * gab * qi * qj + gba * qj * qj) / 3.0)
                v = v.index_add(0, sj, -(gab * qi * qi + 2.0 * gba * qi * qj) / 3.0)
        return v.index_select(0, ctx["shell_site"]), zm, zt

    def density_fock(self, P: Tensor, ctx: dict) -> Tensor:
        """The density-dependent Fock terms: long-range exchange and the onsite correction."""
        out = torch.zeros_like(P)
        if self.plus_u is not None:
            return self._orbital_potential(P, ctx)[1]
        if self.omega is not None:
            out = out + self._exchange_fock(P - ctx["p0"], ctx)
        if self.onsite is not None:
            out = out + self._onsite(P, ctx)[1]
        return out

    def _exchange_fock(self, dP: Tensor, ctx: dict) -> Tensor:
        """Long-range exchange ``-1/8 sum S dP S (g + g + g + g)`` in matrix form (Mulliken approximation of the
        exchange integrals; ``dP`` the density less the reference atoms', ``g`` the long-range gamma)."""
        S, G = ctx["S"], ctx["lr_gamma"]
        SP, PS = S @ dP, dP @ S
        return -0.125 * ((SP * G) @ S + G * (SP @ S) + S @ (dP * G) @ S + S @ (PS * G))

    def _onsite(self, P: Tensor, ctx: dict) -> Tuple[Tensor, Tensor]:
        """Energy per structure and Fock term of the onsite correction from the dual density
        ``Q = (P S + S P) / 2`` less the reference atoms': ``sum W Q^2`` over orbital pairs of an atom plus, per p
        or d shell, ``c ((2l + 1) sum_m Q_mm^2 - (sum_m Q_mm)^2)``; the Fock term is ``(G S + S G) / 2`` with ``G``
        the derivative in ``Q``."""
        S = ctx["S"]
        Q = 0.5 * (P @ S + S @ P) - ctx["p0"]
        W = ctx["ons_w"]
        x = Q.diagonal(dim1=-2, dim2=-1)                                   # (B, nb)
        flat = x.reshape(-1)
        shell, coef, width = ctx["ons_shell"], ctx["ons_c"], ctx["ons_width"]
        n_sh = coef.shape[0]
        sum_x = flat.new_zeros(n_sh + 1).index_add(0, shell, flat)[:n_sh]
        sum_x2 = flat.new_zeros(n_sh + 1).index_add(0, shell, flat * flat)[:n_sh]
        e_shell = coef * (width * sum_x2 - sum_x * sum_x)
        energy = (W * Q * Q).sum((-1, -2)) + x.new_zeros(Q.shape[0]).index_add(0, ctx["ons_struct"], e_shell)
        coef_p = torch.cat([coef, coef.new_zeros(1)])
        width_p = torch.cat([width, width.new_zeros(1)])
        sum_p = torch.cat([sum_x, sum_x.new_zeros(1)])
        g_diag = 2.0 * coef_p[shell] * (width_p[shell] * flat - sum_p[shell])
        G = 2.0 * W * Q + torch.diag_embed(g_diag.view_as(x))
        return energy, 0.5 * (G @ S + S @ G)

    def _exchange_context(self, basis: _DFTBBasis, pos: Tensor, batch: Tensor, n_struct: int,
                          u_atoms: Optional[Tensor]) -> dict:
        """The overlap, the reference density and the long-range gamma between the orbitals."""
        dtype, device = pos.dtype, pos.device
        cached = self.__dict__.get("_last_s")
        S = cached[1] if cached is not None and cached[0] is pos else self._two_center(basis, pos, n_struct)[0]
        n0 = torch.tensor([o for za in basis.z for o in self.reference_occupations(za)], dtype=dtype, device=device)
        p0 = self.reference_density(basis, n0, S)
        n_atoms = len(basis.z)
        if self.omega is None:
            return {"S": S, "p0": p0}
        if u_atoms is None:
            u_atoms = torch.tensor([self.hubbard(z, 0) for z in basis.z], dtype=dtype, device=device)
        tau = 3.2 * u_atoms
        # in a cell the long-range gamma is summed over the images within the exchange cutoff
        lattice = self.__dict__.get("_lattice")
        cut = torch.full((n_struct,), math.inf, dtype=dtype, device=device)
        if lattice is not None:
            a = lattice[0].to(device=device, dtype=dtype)
            faces = torch.linalg.cross(a.roll(-1, dims=1), a.roll(-2, dims=1)).norm(dim=-1)
            inscribed = 0.5 * (torch.linalg.det(a).abs()[:, None] / faces).min(dim=-1).values
            given = torch.full_like(inscribed, self.exchange_cutoff or 0.0)
            cut = torch.where(lattice[1].to(device), given if self.exchange_cutoff else inscribed, cut)
        finite = torch.isfinite(cut)
        i, j, vec, itself = self._pair_entries(batch, pos, float(cut[finite].max()) if bool(finite.any()) else 0.0)
        r = vec.norm(dim=-1)
        keep = r < cut.index_select(0, batch.index_select(0, i))
        i, j, r, itself = i[keep], j[keep], r[keep], itself[keep]
        ti, tj = tau.index_select(0, i), tau.index_select(0, j)
        coulomb = 1.0 / r - gamma_short(ti, tj, r)
        pair = coulomb - gamma_yukawa(ti, tj, r, self.omega)
        onsite = 5.0 * tau / 16.0 - gamma_yukawa_onsite(tau, self.omega)
        G = torch.zeros(n_atoms + 1, n_atoms + 1, dtype=dtype, device=device)
        other = ~itself
        G = G.index_put((i[other], j[other]), pair[other], accumulate=True)
        G = G.index_put((j[other], i[other]), pair[other], accumulate=True)
        idx = torch.arange(n_atoms, device=device)
        G = G.index_put((i[itself], i[itself]), pair[itself], accumulate=True)
        G = G.index_put((idx, idx), onsite, accumulate=True)
        atom_of = torch.full((n_struct, basis.nb), n_atoms, dtype=torch.long, device=device)
        atom_of[basis.ao_struct, basis.ao_index] = basis.ao_atom
        lr = G[atom_of[:, :, None], atom_of[:, None, :]]
        return {"S": S, "p0": p0, "lr_gamma": lr}

    def _orbital_potential(self, P2: Tensor, ctx: dict) -> Tuple[Tensor, Tensor]:
        """Energy per structure and channel Fock terms of DFTB+U from the stacked channel densities ``P2``: the
        block occupation matrices ``n = (P S + S P) / 2``, ``dE / dn`` mapped back by ``(G S + S G) / 2``."""
        S = torch.cat([ctx["S"], ctx["S"]])
        n = 0.5 * (P2 @ S + S @ P2)
        uj, diag = torch.cat([ctx["u_block"]] * 2), torch.cat([ctx["u_diag"]] * 2)     # U - J on blocks and diagonals
        if self.plus_u[0] == "fll":
            energy2 = 0.5 * ((diag * n.diagonal(dim1=-2, dim2=-1)).sum(-1) - (uj * n * n).sum((-1, -2)))
            G = torch.diag_embed(0.5 * diag) - uj * n
        else:
            energy2 = -0.5 * (uj * n * n).sum((-1, -2))
            G = -uj * n
        half = n.shape[0] // 2
        return energy2[:half] + energy2[half:], 0.5 * (G @ S + S @ G)

    def _plus_u_context(self, basis: _DFTBBasis, n_struct: int, dtype, device) -> dict:
        """``U - J`` on the orbital pairs of each block and on its diagonal."""
        nb = basis.nb
        block = torch.zeros(n_struct, nb, nb, dtype=dtype, device=device)
        diag = torch.zeros(n_struct, nb, dtype=dtype, device=device)
        first = basis.atom_first.tolist()
        for a, z in enumerate(basis.z):
            b = int(basis.ao_struct[(basis.ao_atom == a).nonzero()[0, 0]])
            for shells, uj in self.plus_u[1].get(z, []):
                orbitals = [first[a] + l * l + m for l in shells for m in range(2 * l + 1)]
                idx = torch.tensor(orbitals, device=device)
                block[b, idx[:, None], idx[None, :]] = uj
                diag[b, idx] = uj
        return {"u_block": block, "u_diag": diag}

    def _onsite_context(self, basis: _DFTBBasis, n_struct: int, dtype, device) -> dict:
        """Weights of the onsite correction: ``W`` of the orbital pairs of an atom (shell pair constant, zero on
        the diagonal) and, per p or d shell, its coefficient ``lambda_ll / (2l + 1)`` and width ``2l + 1``."""
        nb = basis.nb
        W = torch.zeros(n_struct, nb, nb, dtype=dtype, device=device)
        shell_of = torch.full((n_struct * nb,), -1, dtype=torch.long, device=device)
        coef, width, struct = [], [], []
        first = basis.atom_first.tolist()
        rows, vals = [], []
        for a, z in enumerate(basis.z):
            lam = self.onsite.get(z)
            if lam is None:
                continue
            b = int(basis.ao_struct[(basis.ao_atom == a).nonzero()[0, 0]])
            n_l = self.lmax[z] + 1
            orbitals = [(l, first[a] + l * l + m) for l in range(n_l) for m in range(2 * l + 1)]
            for l1, i in orbitals:
                for l2, j in orbitals:
                    if i != j:
                        rows.append((b, i, j))
                        vals.append(lam[l1, l2])
            for l in range(1, n_l):
                k = len(coef)
                coef.append(lam[l, l] / (2 * l + 1))
                width.append(float(2 * l + 1))
                struct.append(b)
                for m in range(2 * l + 1):
                    shell_of[b * nb + first[a] + l * l + m] = k
        if rows:
            idx = torch.tensor(rows, dtype=torch.long, device=device)
            W[idx[:, 0], idx[:, 1], idx[:, 2]] = torch.tensor(vals, dtype=dtype, device=device)
        n_sh = len(coef)
        shell_of = torch.where(shell_of < 0, torch.full_like(shell_of, n_sh), shell_of)
        return {"ons_w": W, "ons_shell": shell_of, "ons_c": torch.tensor(coef, dtype=dtype, device=device),
                "ons_width": torch.tensor(width, dtype=dtype, device=device),
                "ons_struct": torch.tensor(struct, dtype=torch.long, device=device)}

    def spin_potentials(self, m_shell: Tensor, ctx: dict) -> Tensor:
        """``sum_l' W_ll' m_l'`` per shell (with atomic constants, ``W m_A`` on every shell of the atom)."""
        v = ctx["spin_w"] * m_shell.index_select(0, ctx["spin_j"])
        return torch.zeros_like(m_shell).index_add(0, ctx["spin_i"], v)

    def _spin_context(self, basis: _DFTBBasis, dtype, device) -> dict:
        """Pairs of shells of one atom with their spin constants (all pairs of the atom's shells carry the
        atomic constant when it is one value per atom)."""
        si, sj, w = [], [], []
        shells_of: Dict[int, List[int]] = {}
        for s, a in enumerate(basis.shell_atom.tolist()):
            shells_of.setdefault(a, []).append(s)
        for a, shells in shells_of.items():
            z = basis.z[a]
            if z not in self.spin_constants:
                raise ValueError(f"no spin constants for Z = {z}")
            table = self.spin_constants[z]
            n = len(shells)
            for x, s1 in enumerate(shells):
                for y, s2 in enumerate(shells):
                    si.append(s1)
                    sj.append(s2)
                    w.append(float(table[x * n + y]) if table.size > 1 else float(table[0]))
        si_t = torch.tensor(si, dtype=torch.long, device=device)
        return {"spin_i": si_t, "spin_j": torch.tensor(sj, dtype=torch.long, device=device),
                "spin_w": torch.tensor(w, dtype=dtype, device=device), "spin_struct": basis.shell_struct_t[si_t]}

    # the classical energy

    def classical_energy(self, z: Tensor, pos: Tensor, batch: Tensor, n_struct: int) -> Dict[str, Tensor]:
        """Repulsive pair energies per structure (hartree)."""
        zs = sorted(set(z.tolist()))
        reach = max((self._repulsive_range(za, zb) for za in zs for zb in zs
                     if self.skset.has(min(za, zb), max(za, zb))), default=0.0)
        i, j, vec, itself = self._pair_entries(batch, pos, reach)
        zi, zj = z[i], z[j]
        r = vec.norm(dim=-1)
        e = torch.zeros_like(r)
        for za, zb in sorted({(int(a), int(b)) for a, b in zip(zi.tolist(), zj.tolist())}):
            sel = torch.nonzero((zi == za) & (zj == zb)).squeeze(1)
            e = e.index_put((sel,), self.repulsive(za, zb, r.index_select(0, sel)))
        e = torch.where(itself, 0.5 * e, e)                         # the images of an atom: T and -T both present
        rep = torch.zeros(n_struct, dtype=pos.dtype, device=pos.device).index_add(0, batch.index_select(0, i), e)
        if self.d3 is None:
            return {"total": rep, "repulsion": rep}
        d3 = self.d3.to(device=pos.device, dtype=pos.dtype)
        cutoffs = [getattr(d3, k, 0.0) for k in ("cutoff_pair", "cutoff_triple", "cutoff_cn", "cutoff_eeq_cn")]
        pi, pj, dij, image = self._pair_entries(batch, pos, max(cutoffs) / d3.bohr)
        both = ~image                                               # the images of an atom come in both directions
        ei = torch.cat([torch.stack([pi, pj]), torch.stack([pj[both], pi[both]])], 1)
        zero = torch.zeros(n_struct, dtype=pos.dtype, device=pos.device)
        charge = self.__dict__.get("_total_charge", zero)
        cell = torch.zeros(n_struct, 3, 3, dtype=pos.dtype, device=pos.device)
        pbc = torch.zeros(n_struct, 3, dtype=torch.bool, device=pos.device)
        lattice = self.__dict__.get("_lattice")
        if lattice is not None:
            cell = lattice[0].to(device=pos.device, dtype=pos.dtype) * d3.bohr
            pbc = lattice[1].to(pos.device)[:, None].expand(-1, 3)
        node = d3.evaluate(z, pos * d3.bohr, ei, torch.cat([dij, -dij[both]]) * d3.bohr, batch, n_struct, cell, pbc,
                           charge)["node_energy"]
        disp = zero.index_add(0, batch, node / d3.hartree)
        out = {"total": rep + disp, "repulsion": rep, "dispersion": disp}
        if self.hh_repulsion:
            hh = (z.index_select(0, pi) == 1) & (z.index_select(0, pj) == 1)
            r_aa = dij.norm(dim=-1) * self.bohr
            height, steepness, middle = HH_REPULSION
            e_hh = height / (1.0 + torch.exp(steepness * (r_aa / middle - 1.0)))
            hh = hh & (dij.norm(dim=-1) < HH_CUTOFF)
            e_hh = torch.where(hh, torch.where(image, 0.5 * e_hh, e_hh), torch.zeros_like(r_aa))
            out["hh_repulsion"] = zero.index_add(0, batch.index_select(0, pi), e_hh)
            out["total"] = out["total"] + out["hh_repulsion"]
        return out

    def _repulsive_range(self, za: int, zb: int) -> float:
        """Distance (bohr) beyond which the repulsive potential of the pair vanishes."""
        f = self.skset.pair(min(za, zb), max(za, zb))
        return f.spline.cutoff if f.spline is not None else f.polynomial_cutoff

    def repulsive(self, za: int, zb: int, r: Tensor) -> Tensor:
        """The repulsive pair potential (hartree) of elements ``za`` and ``zb`` at distances ``r`` (bohr):
        the spline of their file, or its polynomial ``sum_k c_k (r_cut - r)^k`` without one."""
        f = self.skset.pair(min(za, zb), max(za, zb))
        spline = f.spline
        if spline is None:
            out = torch.zeros_like(r)
            x = (f.polynomial_cutoff - r).clamp_min(0.0)
            for k, c in enumerate(f.polynomial, start=2):
                out = out + c * x ** k
            return out
        starts = torch.as_tensor(spline.knots[:-1], dtype=r.dtype, device=r.device)
        coef = torch.as_tensor(spline.coefficients, dtype=r.dtype, device=r.device)
        a1, a2, a3 = spline.head
        key = f"{min(za, zb)}_{max(za, zb)}"
        if "repulsive" in self.trainable:
            coef = self.params[f"spline_{key}"].to(device=r.device, dtype=r.dtype)
            a1, a2, a3 = self.params[f"head_{key}"].to(device=r.device, dtype=r.dtype)
        seg = (torch.searchsorted(starts, r.detach().contiguous(), right=True) - 1).clamp(0, starts.shape[0] - 1)
        t = r - starts[seg]
        c = coef[seg]
        poly = c[:, 0] + t * (c[:, 1] + t * (c[:, 2] + t * (c[:, 3] + t * (c[:, 4] + t * c[:, 5]))))
        head = torch.exp(-a1 * r + a2) + a3
        return torch.where(r < starts[0], head, torch.where(r < spline.cutoff, poly, torch.zeros_like(r)))


@register_model("dftb1")
class DFTB1(DFTB):
    """DFTB1: non-self-consistent tight binding (Porezag et al. 1995); see :class:`DFTB`."""

    method = "dftb1"

    def __init__(self, parameters=None, **kwargs):
        kwargs.setdefault("scc", False)
        super().__init__(parameters, **kwargs)


@register_model("dftb2")
class DFTB2(DFTB):
    """DFTB2 (SCC-DFTB, Elstner et al. 1998) with the mio-1-1 set by default; see :class:`DFTB`."""

    method = "dftb2"


@register_model("dftb3")
class DFTB3(DFTB):
    """DFTB3 (Gaus, Cui and Elstner 2011) with the 3ob-3-1 set by default: full third order and the
    damped hydrogen gamma with the set's exponent (``damping_exponent=None``) and Hubbard derivatives; see
    :class:`DFTB`."""

    method = "dftb3"
    default_parameters = "3ob-3-1"

    def __init__(self, parameters=None, **kwargs):
        kwargs.setdefault("third_order", "full")
        super().__init__(parameters, **kwargs)
        if self.damping_exponent is None and self.h5 is None:
            zeta = self.skset.recommended()["damping_exponent"].get(0)
            self.damping_exponent = None if zeta is None else float(zeta)
