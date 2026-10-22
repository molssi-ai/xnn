"""AMOEBA and HIPPO: polarizable atomic multipole force fields.

AMOEBA (Ren and Ponder, *J. Phys. Chem. B* 107, 5933, 2003; Ponder *et al.*,
*J. Phys. Chem. B* 114, 2549, 2010; Shi *et al.*, *JCTC* 9, 4046, 2013)
writes the energy as

    U = U_bond + U_angle + U_strbnd + U_oop + U_torsion + U_vdw
        + U_ele^perm + U_ele^ind

with the anharmonic MM3 bond and angle polynomials, a stretch-bend cross
term, Allinger out-of-plane bending at trigonal centers, Fourier torsions
(plus pi-torsions across double bonds and splined torsion-torsion maps),
Halgren's buffered 14-7 van der Waals form with hydrogen sites displaced
along their bonds, permanent atomic multipoles through the quadrupole in
local frames defined by the bonded neighbours, and induced dipoles obtained
from mutual induction with Thole damping. Interactions between atoms
separated by one or two bonds are left out, 1,4 and 1,5 multipole
interactions are scaled, and permanent multipoles do not polarize atoms of
their own polarization group.

HIPPO (Rackers *et al.*, *JCTC* 17, 7056, 2021) keeps the valence terms and
the multipoles but replaces every intermolecular term by one derived from a
model electron density: each atom is a point core charge plus a
hydrogen-like valence density, so permanent electrostatics acquire charge
penetration (damped multipole interactions), the same damping enters the
induction, the exchange repulsion is an anisotropic multipolar overlap
model, the dispersion is a damped C6 term and a short-range charge-transfer
term is added; a charge-flux term lets the monopoles follow the bond and
angle geometry.

Both fields are one implementation here: :class:`Amoeba` evaluates the shared
valence and induction machinery with the AMOEBA nonbonded terms,
:class:`Hippo` the same with the HIPPO ones. Parameters come from the
``.frc`` libraries of :mod:`xnn.ffnn.models.amoebalib`; the model binds a
library to one :class:`~xnn.ffnn.models.topology.MolecularTopology` (types,
bonds and everything derived from them, including the polarization groups
and the multipole frames) and evaluates any conformation of that system.
Forces are autograd derivatives: the local frames are built from the
positions inside the graph, so the torques of the anisotropic terms become
forces without any explicit conversion, and the induced dipoles are solved
with a differentiable dense linear solve, so the gradient of the
non-variational polarization energy (the induced dipoles see the group-based
*direct* field, the energy the connectivity-based *polar* field) is exact.

Conventions and scope
---------------------
* Units inside the model are eV and Angstrom; multipoles stay in e, e A and
  e A^2. The Coulomb constant is Tinker's 332.063713 kcal A / (mol e^2).
* Dipoles and quadrupoles in the parameter files are in atomic units (e bohr
  and e bohr^2, as the distributed multipole analysis delivers them); they
  are converted to e Angstrom at assembly and the traceless quadrupoles are
  divided by three (the convention of the Tinker files), so the interaction
  tensors are the plain derivatives of the kernel.
* Nonbonded interactions come from the neighbor list within ``cutoff``. For
  clusters they are evaluated in full (no switching, no cutoffs below the
  neighbor cutoff), as the reference calculations of the papers do. For
  periodic cells the multipoles, the induction and HIPPO's dispersion are
  Ewald sums with the reciprocal part on a particle mesh, and the
  short-range terms are switched off smoothly between a taper distance and
  their cutoff (``amoeba_ewald.py``); the induced dipoles of a cell are
  solved iteratively (preconditioned conjugate gradients) with the gradient
  from the converged dipoles through an implicit backward pass.
* The HIPPO repulsion is ``K_i K_j S^2_total / R`` with the multipole
  expansion applied to the squared overlap of two normalised 1s Slater
  functions (the paper's ``f_exp``), the charge-penetration kernels are the
  Coulomb potentials of the model densities; their derivative ladders are
  formed exactly from exponential-polynomial series rather than from
  transcribed formulas.
"""
from __future__ import annotations

import ast
import math
import warnings
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import Parameter, ParameterDict

from xnn.common.data import AtomicGraph
from xnn.common.models import InteratomicPotential, register_model
from xnn.common.models.ops import cell_volume, scatter_sum

from .amoeba_ewald import (
    dispersion_pme,
    dispersion_real_kernel,
    ewald_coefficient,
    ewald_ladder,
    mesh_self_energy,
    mesh_self_moments,
    multipole_pme,
    multipole_reciprocal_energy,
    multipole_self_energy,
    pme_grid_size,
    self_field_coefficient,
    switching,
)
from .amoebalib import DEFAULT_SETTINGS, KCAL_TO_EV, MAX_TORSION_TERMS, WILDCARD, AmoebaLibrary, read_amoeba
from .geometry import TINY, dihedral_cos_sin, pair_vectors
from .topology import MolecularTopology, read_topology, residue_name_list

# Tinker's Coulomb constant in kcal A / (mol e^2) and in eV A / e^2
COULOMB_KCAL = 332.063713
COULOMB = COULOMB_KCAL * KCAL_TO_EV
# the parameter files give dipoles in e bohr and traceless quadrupoles in
# e bohr^2 (the atomic units of the distributed multipole analysis); the
# model works in e Angstrom with the quadrupole divided by three
BOHR = 0.52917721067
DEG = math.pi / 180.0
_DEBYE = 4.80320427            # Debye per e Angstrom (the unit of the induction convergence)
_MAX_Z = 100

# frame types of the permanent multipoles
FRAME_NONE, FRAME_ZONLY, FRAME_ZTHENX, FRAME_BISECTOR, FRAME_ZBISECT, FRAME_THREEFOLD = range(6)

# trainable parameter groups of the force field, and the model-level ones
_GROUPS = ("vdw", "bond", "angle", "strbnd", "urey", "opbend", "torsion", "pitors", "tortor",
           "charge", "dipole", "quadrupole", "polarizability", "thole", "chgpen", "chgtrn",
           "repulsion", "dispersion", "cflux")
_MODEL_GROUPS = ("atom_ref",)


def _split_trainable(trainable: Union[Sequence[str], str]):
    if isinstance(trainable, str):
        if trainable.lower() == "all":
            return "all", set(_MODEL_GROUPS)
        if trainable in _MODEL_GROUPS:
            return (), {trainable}
        return (trainable,), set()
    groups = tuple(g for g in trainable if g not in _MODEL_GROUPS)
    return groups, {g for g in trainable if g in _MODEL_GROUPS}


def _as_library(source) -> AmoebaLibrary:
    if isinstance(source, AmoebaLibrary):
        return source
    return read_amoeba(str(source))


def _unit(v: Tensor) -> Tensor:
    return v / torch.sqrt((v * v).sum(-1, keepdim=True) + TINY)


# radial kernels as exponential-polynomial series
class RadialSeries:
    """A radial function ``sum_k c_k R^p_k exp(-(n_k zeta_i + m_k zeta_j) R / 2)``.

    The kernels of the density-based terms (the Coulomb potential of a
    hydrogen-like density, the overlap of two such densities, its square)
    all have this form, and so does every member of their derivative ladder
    ``D_n = (1/R d/dR)^n g``, which is what the Cartesian multipole
    interaction tensors are built from. Coefficients are tensors over the
    pairs (or Python floats); the exponent bookkeeping is exact.

    Parameters
    ----------
    terms : dict
        ``{(p, n, m): coefficient}``.
    """

    def __init__(self, terms: dict):
        self.terms = dict(terms)

    def shift(self, dp: int) -> "RadialSeries":
        """Multiply by ``R^dp``."""
        return RadialSeries({(p + dp, n, m): c for (p, n, m), c in self.terms.items()})

    def evaluate(self, R: Tensor, zi: Tensor, zj: Tensor) -> Tensor:
        ei = torch.exp(-0.5 * zi * R)
        ej = torch.exp(-0.5 * zj * R)
        powers: dict = {0: torch.ones_like(R)}
        exps: dict = {(0, 0): torch.ones_like(R)}

        def power(p):
            if p not in powers:
                powers[p] = R ** p
            return powers[p]

        def expo(n, m):
            if (n, m) not in exps:
                exps[(n, m)] = ei ** n * ej ** m if (n and m) else (ei ** n if n else ej ** m)
            return exps[(n, m)]

        total = torch.zeros_like(R)
        for (p, n, m), c in self.terms.items():
            coef = _resolve(c, zi, zj)
            total = total + coef * power(p) * expo(n, m)
        return total


def _resolve(c, zi, zj):
    """Coefficients may hold deferred factors ``("exp", c, n, m)`` meaning
    ``-c (n zeta_i + m zeta_j) / 2`` (from differentiating the exponential)."""
    if isinstance(c, tuple):
        if c[0] == "exp":
            inner = _resolve(c[1], zi, zj)
            return -0.5 * inner * (c[2] * zi + c[3] * zj)
        if c[0] == "sum":
            return _resolve(c[1], zi, zj) + _resolve(c[2], zi, zj)
        if c[0] == "mul":
            return _resolve(c[1], zi, zj) * _resolve(c[2], zi, zj)
    return c


class _Deferred:
    """Arithmetic on deferred coefficients (kept symbolic until evaluation)."""
    @staticmethod
    def add(a, b):
        if isinstance(a, tuple) or isinstance(b, tuple):
            return ("sum", a, b)
        return a + b

    @staticmethod
    def mul(a, b):
        if isinstance(a, tuple) or isinstance(b, tuple):
            return ("mul", a, b)
        return a * b


def _series_ladder(self: RadialSeries) -> RadialSeries:
    """``(1/R) d/dR`` of a series: ``c R^p e^-aR -> c p R^(p-2) e^-aR - c a R^(p-1) e^-aR``."""
    out: dict = {}

    def add(key, value):
        out[key] = _Deferred.add(out[key], value) if key in out else value

    for (p, n, m), c in self.terms.items():
        if p != 0:
            add((p - 2, n, m), _Deferred.mul(c, float(p)))
        if n or m:
            add((p - 1, n, m), ("exp", c, n, m))
    return RadialSeries(out)


def _series_mul(self: RadialSeries, other: RadialSeries) -> RadialSeries:
    out: dict = {}
    for (p1, n1, m1), c1 in self.terms.items():
        for (p2, n2, m2), c2 in other.terms.items():
            key = (p1 + p2, n1 + n2, m1 + m2)
            prod = _Deferred.mul(c1, c2)
            out[key] = _Deferred.add(out[key], prod) if key in out else prod
    return RadialSeries(out)


RadialSeries.ladder = _series_ladder          # type: ignore[assignment]
RadialSeries.__mul__ = _series_mul            # type: ignore[assignment]


def coulomb_kernel() -> RadialSeries:
    """``1/R``."""
    return RadialSeries({(-1, 0, 0): 1.0})


def core_valence_kernel(zeta: Tensor, slot: str = "j") -> RadialSeries:
    """Potential of a hydrogen-like valence density of exponent ``zeta``:
    ``[1 - (1 + zeta R / 2) exp(-zeta R)] / R``. ``slot`` says which exponent
    slot of :meth:`RadialSeries.evaluate` (``"i"`` or ``"j"``) ``zeta`` is
    passed in, so the exponential and the polynomial agree."""
    label = (2, 0) if slot == "i" else (0, 2)
    return RadialSeries({(-1, 0, 0): 1.0, (-1,) + label: -1.0, (0,) + label: -0.5 * zeta})


def overlap_coulomb_kernel(zi: Tensor, zj: Tensor, equal: bool) -> RadialSeries:
    """Coulomb interaction kernel of two hydrogen-like densities (the
    two-center charge-penetration damping): for equal exponents
    ``[1 - (1 + 11/16 x + 3/16 x^2 + 1/48 x^3) e^-x] / R`` with ``x = zeta R``,
    otherwise the closed form with ``A = zeta_j^2 / (zeta_j^2 - zeta_i^2)``
    and ``B = zeta_i^2 / (zeta_i^2 - zeta_j^2)``."""
    if equal:
        z = zi
        return RadialSeries({(-1, 0, 0): 1.0, (-1, 2, 0): -1.0, (0, 2, 0): -11.0 / 16.0 * z,
                             (1, 2, 0): -3.0 / 16.0 * z ** 2, (2, 2, 0): -z ** 3 / 48.0})
    A = zj ** 2 / (zj ** 2 - zi ** 2)
    B = zi ** 2 / (zi ** 2 - zj ** 2)
    return RadialSeries({(-1, 0, 0): 1.0,
                         (-1, 2, 0): -A * A * (1.0 + 2.0 * B), (0, 2, 0): -A * A * 0.5 * zi,
                         (-1, 0, 2): -B * B * (1.0 + 2.0 * A), (0, 0, 2): -B * B * 0.5 * zj})


def slater_overlap(zi: Tensor, zj: Tensor, equal: bool) -> RadialSeries:
    """Overlap of two normalised 1s Slater functions with exponents
    ``zeta_i / 2`` and ``zeta_j / 2`` (the HIPPO pseudo-orbitals)."""
    if equal:
        z = zi
        return RadialSeries({(0, 1, 0): 1.0, (1, 1, 0): 0.5 * z, (2, 1, 0): z ** 2 / 12.0})
    delta = zi ** 2 - zj ** 2
    pref = 8.0 * (zi * zj) ** 1.5 / delta ** 3
    return RadialSeries({(0, 0, 1): pref * zi * delta, (-1, 0, 1): -8.0 * pref * zi * zj,
                         (0, 1, 0): pref * zj * delta, (-1, 1, 0): 8.0 * pref * zi * zj})


def derivative_ladder(kernel: RadialSeries, R: Tensor, zi: Tensor, zj: Tensor, order: int
                      ) -> list[Tensor]:
    """``[D_0, ..., D_order]`` with ``D_n = (1/R d/dR)^n g`` evaluated per pair."""
    dtype = R.dtype
    R64, zi64, zj64 = R.double(), zi.double(), zj.double()
    out = []
    series = kernel
    for n in range(order + 1):
        # the series of unequal exponents cancel strongly: evaluate in float64 and
        # return in the working precision
        out.append(series.evaluate(R64, zi64, zj64).to(dtype))
        if n < order:
            series = series.ladder()
    return out


def coulomb_ladder(R: Tensor, order: int) -> list[Tensor]:
    """The undamped ladder ``1/R, -1/R^3, 3/R^5, -15/R^7, 105/R^9``."""
    inv = 1.0 / R
    inv2 = inv * inv
    out = [inv, -inv * inv2, 3.0 * inv * inv2 ** 2, -15.0 * inv * inv2 ** 3, 105.0 * inv * inv2 ** 4]
    return out[:order + 1]


def thole_ladder(R: Tensor, alpha_i: Tensor, alpha_j: Tensor, thole_i: Tensor, thole_j: Tensor,
                 order: int = 3) -> list[Tensor]:
    """The Thole-damped ladder ``D_1`` to ``D_3`` (``D_0`` is the plain Coulomb
    ``1/R``): ``u = R / (alpha_i alpha_j)^(1/6)``, ``a = min(thole_i, thole_j)``
    (the other when one is zero, undamped when both are),
    ``lambda_3 = 1 - e^-au^3``, ``lambda_5 = 1 - (1 + au^3) e^-au^3``,
    ``lambda_7 = 1 - (1 + au^3 + 3/5 a^2 u^6) e^-au^3``."""
    a = torch.where(thole_i > 0, torch.where(thole_j > 0, torch.minimum(thole_i, thole_j), thole_i), thole_j)
    damp_scale = (alpha_i * alpha_j) ** (1.0 / 6.0)
    u3 = (R / damp_scale.clamp_min(1e-12)) ** 3
    d = torch.clamp(a * u3, max=50.0)
    ex = torch.exp(-d)
    undamped = a <= 0
    l3 = torch.where(undamped, torch.ones_like(d), 1.0 - ex)
    l5 = torch.where(undamped, torch.ones_like(d), 1.0 - (1.0 + d) * ex)
    l7 = torch.where(undamped, torch.ones_like(d), 1.0 - (1.0 + d + 0.6 * d * d) * ex)
    inv = 1.0 / R
    inv2 = inv * inv
    out = [inv, -l3 * inv * inv2, 3.0 * l5 * inv * inv2 ** 2, -15.0 * l7 * inv * inv2 ** 3]
    return out[:order + 1]


def multipole_energy(ci, di, Qi, ck, dk, Qk, r, D) -> Tensor:
    """Interaction energy of two multipole sets through a radial kernel.

    ``E = [c_k + d_k . grad + Q_k : grad grad] [c_i - d_i . grad + Q_i : grad grad] g(r)``
    with ``r = r_k - r_i`` and the ladder ``D = [D_0 .. D_4]`` of ``g`` (five
    entries; pass zeros for the orders a term set does not need).

    Parameters
    ----------
    ci, ck : Tensor
        Charges ``(M,)``.
    di, dk : Tensor
        Dipoles ``(M, 3)``.
    Qi, Qk : Tensor
        Quadrupoles ``(M, 3, 3)`` (traceless, divided by three).
    r : Tensor
        ``r_k - r_i`` ``(M, 3)``.
    D : sequence of Tensor
        The kernel ladder.

    Returns
    -------
    Tensor
        Per-pair energies ``(M,)`` (without the Coulomb constant).
    """
    D0, D1, D2, D3, D4 = D
    dir_ = (di * r).sum(-1)
    dkr = (dk * r).sum(-1)
    dik = (di * dk).sum(-1)
    Qir = torch.einsum("mab,mb->ma", Qi, r)
    Qkr = torch.einsum("mab,mb->ma", Qk, r)
    qir = (Qir * r).sum(-1)
    qkr = (Qkr * r).sum(-1)
    qik = (Qir * Qkr).sum(-1)
    dkqi = (dk * Qir).sum(-1)
    diqk = (di * Qkr).sum(-1)
    qiqk = (Qi * Qk).sum((-1, -2))
    return (ci * ck * D0
            + (ci * dkr - ck * dir_) * D1
            - (D2 * dir_ * dkr + D1 * dik)
            + (ck * qir + ci * qkr) * D2
            + D3 * (dkr * qir - dir_ * qkr) + 2.0 * D2 * (dkqi - diqk)
            + D4 * qir * qkr + 4.0 * D3 * qik + 2.0 * D2 * qiqk)


def multipole_field(c, d, Q, r, D, at_k: bool) -> Tensor:
    """Electric field of a multipole set through a radial kernel.

    With ``r = r_k - r_i``: the field at ``k`` due to the set at ``i``
    (``at_k=True``) is ``-c D_1 r + D_2 r (d.r) + D_1 d - D_3 r (r.Q.r) - 2 D_2 Q r``;
    the field at ``i`` due to the set at ``k`` (``at_k=False``) is
    ``c D_1 r + D_2 r (d.r) + D_1 d + D_3 r (r.Q.r) + 2 D_2 Q r``.
    """
    D1, D2, D3 = D[1], D[2], D[3]
    dr = (d * r).sum(-1, keepdim=True)
    Qr = torch.einsum("mab,mb->ma", Q, r)
    qrr = (Qr * r).sum(-1, keepdim=True)
    sign = -1.0 if at_k else 1.0
    return (sign * c[:, None] * D1[:, None] * r + D2[:, None] * r * dr + D1[:, None] * d
            + sign * D3[:, None] * r * qrr + sign * 2.0 * D2[:, None] * Qr)


def dipole_field_tensor(r: Tensor, D1: Tensor, D2: Tensor) -> Tensor:
    """``(M, 3, 3)`` tensor ``T`` with ``field = T mu`` for a dipole at the
    other end of ``r``: ``D_2 r r^T + D_1 I``."""
    eye = torch.eye(3, dtype=r.dtype, device=r.device)
    return D2[:, None, None] * r[:, :, None] * r[:, None, :] + D1[:, None, None] * eye


# bicubic spline of the torsion-torsion grids
def _periodic_spline_derivative(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    """First derivatives of a periodic cubic spline through ``values`` on the
    grid ``x`` (first and last point coincide), along axis 0."""
    n = values.shape[0] - 1                 # independent points
    h = np.diff(x)                          # (n,)
    # cyclic tridiagonal system for the derivatives at the n independent points
    A = np.zeros((n, n))
    rhs = np.zeros((n,) + values.shape[1:])
    for i in range(n):
        im, ip = (i - 1) % n, (i + 1) % n
        hm, hp = h[(i - 1) % n], h[i]
        A[i, im] += 1.0 / hm
        A[i, i] += 2.0 / hm + 2.0 / hp
        A[i, ip] += 1.0 / hp
        rhs[i] = 3.0 * ((values[i] - values[im]) / hm ** 2 + (values[ip] - values[i]) / hp ** 2)
    d = np.linalg.solve(A, rhs.reshape(n, -1)).reshape(rhs.shape)
    return np.concatenate([d, d[:1]], axis=0)


def tortor_spline_tables(angles1, angles2, values) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(df/dx, df/dy, d2f/dxdy)`` on a periodic torsion-torsion grid."""
    f = np.asarray(values, dtype=float)
    x = np.asarray(angles1, dtype=float)
    y = np.asarray(angles2, dtype=float)
    fx = _periodic_spline_derivative(f, x)
    fy = _periodic_spline_derivative(f.T, y).T
    fxy = _periodic_spline_derivative(fx.T, y).T
    return fx, fy, fxy


_BCU_WT = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [-3, 0, 0, 3, 0, 0, 0, 0, -2, 0, 0, -1, 0, 0, 0, 0],
    [2, 0, 0, -2, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, -3, 0, 0, 3, 0, 0, 0, 0, -2, 0, 0, -1],
    [0, 0, 0, 0, 2, 0, 0, -2, 0, 0, 0, 0, 1, 0, 0, 1],
    [-3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0],
    [9, -9, 9, -9, 6, 3, -3, -6, 6, -6, -3, 3, 4, 2, 1, 2],
    [-6, 6, -6, 6, -4, -2, 2, 4, -3, 3, 3, -3, -2, -1, -1, -2],
    [2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0],
    [-6, 6, -6, 6, -3, -3, 3, 3, -4, 4, 2, -2, -2, -2, -1, -1],
    [4, -4, 4, -4, 2, 2, -2, -2, 2, -2, -2, 2, 1, 1, 1, 1]], dtype=float)


def bicubic_interpolate(f, fx, fy, fxy, x, y, x1, x2, y1, y2):
    """Bicubic interpolation (the Numerical Recipes construction) inside the
    cell with corner values ``f`` ``(M, 4)`` and derivatives, corners counted
    counter-clockwise from the lower left: ``(x1,y1), (x2,y1), (x2,y2), (x1,y2)``."""
    d1 = x2 - x1
    d2 = y2 - y1
    wt = torch.as_tensor(_BCU_WT, dtype=f.dtype, device=f.device)
    packed = torch.cat([f, fx * d1[:, None], fy * d2[:, None], fxy * (d1 * d2)[:, None]], dim=1)
    c = torch.einsum("ij,mj->mi", wt, packed).reshape(-1, 4, 4)
    t = ((x - x1) / d1)[:, None]
    u = ((y - y1) / d2)[:, None]
    tp = torch.cat([torch.ones_like(t), t, t * t, t * t * t], 1)
    up = torch.cat([torch.ones_like(u), u, u * u, u * u * u], 1)
    return torch.einsum("mi,mij,mj->m", tp, c, up)


class AmoebaForceField(nn.Module):
    """The trainable parameter tensors of an AMOEBA / HIPPO library.

    Parameters
    ----------
    library : AmoebaLibrary, str or Path
        The library (a shipped set such as ``"amoeba-water-2003"``, a
        ``.frc`` / ``.prm`` / JSON path, see
        :func:`~xnn.ffnn.models.amoebalib.read_amoeba`).
    trainable : sequence of str or "all", optional
        Groups to expose to the optimizer: ``"vdw"``, ``"bond"``,
        ``"angle"``, ``"strbnd"``, ``"urey"``, ``"opbend"``, ``"torsion"``,
        ``"pitors"``, ``"tortor"``, ``"charge"``, ``"dipole"``,
        ``"quadrupole"``, ``"polarizability"``, ``"thole"``, ``"chgpen"``,
        ``"chgtrn"``, ``"repulsion"``, ``"dispersion"``, ``"cflux"``;
        ``"all"`` unfreezes every group.
    symmetry_mask : bool, optional
        Keep the dipole and quadrupole components that the library leaves
        zero (by the symmetry of the local frame) at zero while they train,
        by default ``True``.

    Attributes
    ----------
    params : torch.nn.ParameterDict
        The parameter tensors in eV / Angstrom / radian / e (see the source
        for each group's layout).
    settings : dict
        The force-field definition settings.
    library : AmoebaLibrary
        The library the tensors were built from (names and keys).
    """

    def __init__(self, library: Union[AmoebaLibrary, str, Path],
                 trainable: Union[Sequence[str], str] = (),
                 symmetry_mask: bool = True):
        super().__init__()
        lib = _as_library(library)
        self.library = lib
        dtype = torch.get_default_dtype()
        self.name = lib.name
        self.model = lib.model
        self.settings = {k: lib.setting(k) for k in DEFAULT_SETTINGS}
        self.settings.update({k: v for k, v in lib.settings.items() if k not in DEFAULT_SETTINGS})
        if isinstance(trainable, str):
            trainable_set = set(_GROUPS) if trainable.lower() == "all" else {trainable}
        else:
            trainable_set = set(trainable)
        unknown = trainable_set - set(_GROUPS)
        if unknown:
            raise ValueError(f"unknown trainable group(s) {sorted(unknown)}; available: {list(_GROUPS)}")

        self.type_names = list(lib.atom_types)
        self.class_names = lib.classes()
        self._type_index = {n: i for i, n in enumerate(self.type_names)}
        self._class_index = {n: i for i, n in enumerate(self.class_names)}
        self.type_class = [self._class_index[lib.cls(t)] for t in self.type_names]
        self.templates = dict(lib.templates)
        self.fragments = dict(lib.fragments)
        self.metadata = dict(lib.metadata)
        self.references = list(lib.references)
        self.register_buffer("type_z", torch.tensor([int(lib.atom_types[t]["element"]) for t in self.type_names],
                                                    dtype=torch.long))
        self.register_buffer("type_mass", torch.tensor([float(lib.atom_types[t].get("mass", 0.0))
                                                        for t in self.type_names], dtype=dtype))
        self.params = ParameterDict()

        def make(name, values, group, shape=(-1,), scale=1.0):
            t = torch.tensor(values, dtype=torch.float64).reshape(*shape) * scale
            self.params[name] = Parameter(t.to(dtype), requires_grad=group in trainable_set)

        nc = len(self.class_names)
        # van der Waals per class (missing classes get zero depth)
        vdw = [lib.vdw_types.get(c, {"rmin": 1.0, "eps": 0.0, "reduction": 1.0}) for c in self.class_names]
        make("vdw_rmin", [float(v["rmin"]) for v in vdw], "vdw")
        make("vdw_eps", [float(v["eps"]) for v in vdw], "vdw", scale=KCAL_TO_EV)
        self.register_buffer("vdw_reduction", torch.tensor([float(v.get("reduction", 1.0)) for v in vdw],
                                                           dtype=dtype))
        self.vdw_pair_keys = list(lib.vdw_pair_types)
        make("vdw_pair_rmin", [float(lib.vdw_pair_types[k]["rmin"]) for k in self.vdw_pair_keys] or [0.0], "vdw")
        make("vdw_pair_eps", [float(lib.vdw_pair_types[k]["eps"]) for k in self.vdw_pair_keys] or [0.0], "vdw",
             scale=KCAL_TO_EV)
        # dense per-class-pair override tables (index -1 = none)
        pair_idx = torch.full((nc, nc), -1, dtype=torch.long)
        for m, key in enumerate(self.vdw_pair_keys):
            a, b = key.split("-")
            ia, ib = self._class_index[a], self._class_index[b]
            pair_idx[ia, ib] = pair_idx[ib, ia] = m
        self.register_buffer("vdw_pair_index", pair_idx)

        # valence tables
        self.bond_keys = list(lib.bond_types)
        make("bond_k", [lib.bond_types[k]["k"] for k in self.bond_keys] or [0.0], "bond", scale=KCAL_TO_EV)
        make("bond_r0", [lib.bond_types[k]["r0"] for k in self.bond_keys] or [0.0], "bond")
        self.angle_keys = list(lib.angle_types)
        self.angle_inplane_keys = list(lib.angle_inplane_types)
        all_angles = [lib.angle_types[k] for k in self.angle_keys] + \
                     [lib.angle_inplane_types[k] for k in self.angle_inplane_keys]
        make("angle_k", [a["k"] for a in all_angles] or [0.0], "angle", scale=KCAL_TO_EV)

        def pad3(v):
            v = list(v)[:3]
            return v + [v[-1]] * (3 - len(v)) if v else [0.0, 0.0, 0.0]

        make("angle_theta0", [pad3(a["theta0"]) for a in all_angles] or [[0.0] * 3], "angle",
             shape=(-1, 3), scale=DEG)
        self.strbnd_keys = list(lib.strbnd_types)
        make("strbnd_k", [[lib.strbnd_types[k]["k1"], lib.strbnd_types[k]["k2"]] for k in self.strbnd_keys]
             or [[0.0, 0.0]], "strbnd", shape=(-1, 2), scale=KCAL_TO_EV)
        self.urey_keys = list(lib.urey_types)
        make("urey_k", [lib.urey_types[k]["k"] for k in self.urey_keys] or [0.0], "urey", scale=KCAL_TO_EV)
        make("urey_r0", [lib.urey_types[k]["r0"] for k in self.urey_keys] or [0.0], "urey")
        self.opbend_keys = list(lib.opbend_types)
        make("opbend_k", [lib.opbend_types[k]["k"] for k in self.opbend_keys] or [0.0], "opbend",
             scale=KCAL_TO_EV)
        self.torsion_keys = list(lib.torsion_types)
        ks, ns, phs, ms = [], [], [], []
        for key in self.torsion_keys:
            terms = list(lib.torsion_types[key]["terms"])[:MAX_TORSION_TERMS]
            pad = MAX_TORSION_TERMS - len(terms)
            ks.append([float(t[0]) * KCAL_TO_EV for t in terms] + [0.0] * pad)
            ns.append([max(int(t[1]), 1) for t in terms] + [1] * pad)
            phs.append([float(t[2]) * DEG for t in terms] + [0.0] * pad)
            ms.append([1.0] * len(terms) + [0.0] * pad)
        if not ks:
            ks, ns, phs, ms = [[0.0] * MAX_TORSION_TERMS], [[1] * MAX_TORSION_TERMS], \
                [[0.0] * MAX_TORSION_TERMS], [[0.0] * MAX_TORSION_TERMS]
        self.params["torsion_k"] = Parameter(torch.tensor(ks, dtype=dtype), requires_grad="torsion" in trainable_set)
        self.register_buffer("torsion_n", torch.tensor(ns, dtype=torch.long))
        self.register_buffer("torsion_mask", torch.tensor(ms, dtype=dtype))
        ph = torch.tensor(phs, dtype=torch.float64)
        c, s = torch.cos(ph), torch.sin(ph)
        for t in (c, s):
            for target in (-1.0, 0.0, 1.0):
                t[(t - target).abs() < 1e-12] = target
        self.register_buffer("torsion_cos", c.to(dtype))
        self.register_buffer("torsion_sin", s.to(dtype))
        self.pitors_keys = list(lib.pitors_types)
        make("pitors_k", [lib.pitors_types[k]["k"] for k in self.pitors_keys] or [0.0], "pitors",
             scale=KCAL_TO_EV)
        self.tortor_keys = list(lib.tortor_types)
        self.tortor_grids: list[tuple[Tensor, Tensor]] = []
        for m, key in enumerate(self.tortor_keys):
            row = lib.tortor_types[key]
            self.register_buffer(f"tortor_x_{m}", torch.tensor(row["angles1"], dtype=dtype))
            self.register_buffer(f"tortor_y_{m}", torch.tensor(row["angles2"], dtype=dtype))
            self.params[f"tortor_f_{m}"] = Parameter(
                torch.tensor(row["values"], dtype=dtype) * KCAL_TO_EV, requires_grad="tortor" in trainable_set)

        # permanent multipoles, one row per (type, frame) entry
        self.multipole_rows: list[tuple[str, tuple]] = []
        qs, ds, Qs = [], [], []
        for t, entries in lib.multipoles.items():
            for e in entries:
                self.multipole_rows.append((t, tuple(str(x) for x in e["frame"])))
                qs.append(float(e["q"]))
                ds.append([float(x) for x in e["dipole"]])
                Qs.append([[float(x) for x in row] for row in e["quadrupole"]])
        if not qs:
            qs, ds, Qs = [0.0], [[0.0] * 3], [[[0.0] * 3] * 3]
        make("mpole_q", qs, "charge")
        make("mpole_d", ds, "dipole", shape=(-1, 3), scale=BOHR)
        make("mpole_Q", Qs, "quadrupole", shape=(-1, 3, 3), scale=BOHR * BOHR / 3.0)
        # components the library leaves zero stay zero while the multipoles train
        d_mask = torch.tensor(ds, dtype=dtype) != 0
        Q_mask = torch.tensor(Qs, dtype=dtype) != 0
        if not symmetry_mask:
            d_mask, Q_mask = torch.ones_like(d_mask), torch.ones_like(Q_mask)
        self.register_buffer("mpole_d_mask", d_mask.to(dtype), persistent=False)
        self.register_buffer("mpole_Q_mask", Q_mask.to(dtype), persistent=False)
        # total-charge constraints of the bound topologies (rows of multipole-row counts, their
        # totals and the pseudo-inverse of the rows); derived from the topologies, so not part
        # of the state dict
        n_rows = len(qs)
        self.register_buffer("charge_rows", torch.zeros(0, n_rows, dtype=dtype), persistent=False)
        self.register_buffer("charge_totals", torch.zeros(0, dtype=dtype), persistent=False)
        self.register_buffer("charge_proj", torch.zeros(n_rows, 0, dtype=dtype), persistent=False)
        # polarizability per type
        pol = [lib.polarize.get(t, {"alpha": 0.0, "thole": 0.0, "groups": []}) for t in self.type_names]
        make("polarizability", [float(p["alpha"]) for p in pol], "polarizability")
        make("thole", [float(p.get("thole", 0.0) or 0.0) for p in pol], "thole")
        self.polar_groups = [[str(g) for g in p.get("groups", [])] for p in pol]
        # HIPPO per-class density parameters
        has_pen = bool(lib.chgpen)
        cp = [lib.chgpen.get(c, {"core": 0.0, "alpha": 0.0}) for c in self.class_names]
        make("chgpen_core", [float(v["core"]) for v in cp], "chgpen")
        make("chgpen_alpha", [float(v["alpha"]) for v in cp], "chgpen")
        ct = [lib.chgtrn.get(c, {"eps": 0.0, "alpha": 0.0}) for c in self.class_names]
        # the charge-transfer prefactor is a charge-like quantity: the energy is
        # eps exp(-eta R) times the Coulomb constant (Tinker's convention)
        make("chgtrn_eps", [float(v["eps"]) for v in ct], "chgtrn", scale=COULOMB)
        make("chgtrn_alpha", [float(v["alpha"]) for v in ct], "chgtrn")
        rp = [lib.repulsion.get(c, {"k": 0.0, "alpha": 1.0, "valence": 0.0}) for c in self.class_names]
        make("rep_k", [float(v["k"]) for v in rp], "repulsion", scale=math.sqrt(KCAL_TO_EV))
        make("rep_alpha", [float(v["alpha"]) for v in rp], "repulsion")
        make("rep_valence", [float(v["valence"]) for v in rp], "repulsion")
        dp = [lib.dispersion.get(c, {"c6": 0.0, "alpha": 1.0}) for c in self.class_names]
        make("disp_c6", [float(v["c6"]) for v in dp], "dispersion", scale=math.sqrt(KCAL_TO_EV))
        make("disp_alpha", [float(v["alpha"]) for v in dp], "dispersion")
        self.has_penetration = has_pen
        # charge flux
        self.bndcflux_keys = list(lib.bndcflux)
        make("bndcflux_j", [lib.bndcflux[k]["j"] for k in self.bndcflux_keys] or [0.0], "cflux")
        self.angcflux_keys = list(lib.angcflux)
        make("angcflux", [[lib.angcflux[k]["jt1"], lib.angcflux[k]["jt2"],
                           lib.angcflux[k]["jb1"], lib.angcflux[k]["jb2"]] for k in self.angcflux_keys]
             or [[0.0] * 4], "cflux", shape=(-1, 4))
        self._bond_index = {k: i for i, k in enumerate(self.bond_keys)}
        self._angle_index = {k: i for i, k in enumerate(self.angle_keys)}
        self._angle_inplane_index = {k: i + len(self.angle_keys) for i, k in enumerate(self.angle_inplane_keys)}
        self._strbnd_index = {k: i for i, k in enumerate(self.strbnd_keys)}
        self._urey_index = {k: i for i, k in enumerate(self.urey_keys)}
        self._opbend_index = {k: i for i, k in enumerate(self.opbend_keys)}
        self._torsion_index = {k: i for i, k in enumerate(self.torsion_keys)}
        self._pitors_index = {k: i for i, k in enumerate(self.pitors_keys)}
        self._tortor_index = {k: i for i, k in enumerate(self.tortor_keys)}
        self._mpole_index: dict[str, list[int]] = {}
        for m, (t, _frame) in enumerate(self.multipole_rows):
            self._mpole_index.setdefault(t, []).append(m)
        self._bndcflux_index = {k: i for i, k in enumerate(self.bndcflux_keys)}
        self._angcflux_index = {k: i for i, k in enumerate(self.angcflux_keys)}

    # lookups
    def type_index(self, name: str) -> int:
        if name not in self._type_index:
            raise KeyError(f"atom type {name!r} is not in library {self.name!r}")
        return self._type_index[name]

    def class_of(self, type_name: str) -> str:
        return self.class_names[self.type_class[self.type_index(type_name)]]

    # multipole constraints
    def constrain_charges(self, counts: Tensor, total: float) -> None:
        """Fix a molecule's total charge across refits of ``charge``.

        Adds the linear constraint ``counts . q == total`` on the multipole
        row charges ``q``; :attr:`charges` is then the ``mpole_q`` parameter
        projected onto every constraint registered so far, so the molecule
        keeps its net charge whatever the optimizer does to the row values.
        :class:`Amoeba` registers every molecule of its topology at
        construction; a force field shared by several models accumulates
        their constraints, identical molecules once.

        Parameters
        ----------
        counts : Tensor
            ``(R,)``, how many atoms of the molecule use each multipole row.
        total : float
            The molecule's total charge in e.
        """
        row = counts.detach().to(self.charge_rows).reshape(1, -1)
        if self.charge_rows.shape[0] and bool((self.charge_rows == row).all(1).any()):
            return
        self.charge_rows = torch.cat([self.charge_rows, row])
        self.charge_totals = torch.cat([self.charge_totals, row.new_tensor([float(total)])])
        self.charge_proj = torch.linalg.pinv(self.charge_rows)

    @property
    def charges(self) -> Tensor:
        """Tensor : The row charges satisfying the registered constraints.

        The minimum-norm correction ``A^+ (A q - b)`` is removed from the
        ``mpole_q`` parameter ``q``, with ``A`` the constraint rows and ``b``
        their totals; without constraints this is ``q`` itself.
        """
        q = self.params["mpole_q"]
        if self.charge_rows.shape[0] == 0:
            return q
        return q - self.charge_proj @ (self.charge_rows @ q - self.charge_totals)

    def effective_multipoles(self) -> tuple[Tensor, Tensor, Tensor]:
        """The multipoles the model uses.

        The constrained charges, the masked dipoles and the quadrupoles made
        symmetric, masked and traceless.

        Returns
        -------
        tuple of Tensor
            ``(R,)`` charges in e, ``(R, 3)`` dipoles in e Angstrom and
            ``(R, 3, 3)`` quadrupoles in e Angstrom^2 / 3, one row per
            ``(type, frame)`` entry, in the local frames.
        """
        d = self.params["mpole_d"] * self.mpole_d_mask
        Q = self.params["mpole_Q"]
        Q = 0.5 * (Q + Q.transpose(1, 2)) * self.mpole_Q_mask
        tr = torch.diagonal(Q, dim1=1, dim2=2).sum(-1) / 3.0
        Q = Q - tr[:, None, None] * torch.eye(3, dtype=Q.dtype, device=Q.device)
        return self.charges, d, Q

    def export_library(self) -> AmoebaLibrary:
        """The current (possibly trained) parameters as an :class:`AmoebaLibrary`.

        The multipoles are the ones the model uses (see
        :meth:`effective_multipoles`), so a program reading the exported file
        evaluates the same model.

        Returns
        -------
        AmoebaLibrary
            In the library's units (kcal/mol, Angstrom, degrees).
        """
        import copy
        lib = copy.deepcopy(self.library)
        P = {k: v.detach().cpu().double() for k, v in self.params.items()}
        for i, c in enumerate(self.class_names):
            if c in lib.vdw_types:
                lib.vdw_types[c]["rmin"] = float(P["vdw_rmin"][i])
                lib.vdw_types[c]["eps"] = float(P["vdw_eps"][i]) / KCAL_TO_EV
            if c in lib.chgpen:
                lib.chgpen[c] = {"core": float(P["chgpen_core"][i]), "alpha": float(P["chgpen_alpha"][i])}
            if c in lib.chgtrn:
                lib.chgtrn[c] = {"eps": float(P["chgtrn_eps"][i]) / COULOMB,
                                 "alpha": float(P["chgtrn_alpha"][i])}
            if c in lib.repulsion:
                lib.repulsion[c] = {"k": float(P["rep_k"][i]) / math.sqrt(KCAL_TO_EV),
                                    "alpha": float(P["rep_alpha"][i]), "valence": float(P["rep_valence"][i])}
            if c in lib.dispersion:
                lib.dispersion[c] = {"c6": float(P["disp_c6"][i]) / math.sqrt(KCAL_TO_EV),
                                     "alpha": float(P["disp_alpha"][i])}
        for m, k in enumerate(self.vdw_pair_keys):
            lib.vdw_pair_types[k] = {"rmin": float(P["vdw_pair_rmin"][m]),
                                     "eps": float(P["vdw_pair_eps"][m]) / KCAL_TO_EV}
        for m, k in enumerate(self.bond_keys):
            lib.bond_types[k] = {"k": float(P["bond_k"][m]) / KCAL_TO_EV, "r0": float(P["bond_r0"][m])}
        for m, k in enumerate(self.angle_keys):
            n = len(lib.angle_types[k]["theta0"])
            lib.angle_types[k] = {"k": float(P["angle_k"][m]) / KCAL_TO_EV,
                                  "theta0": [float(x) / DEG for x in P["angle_theta0"][m][:n]]}
        off = len(self.angle_keys)
        for m, k in enumerate(self.angle_inplane_keys):
            n = len(lib.angle_inplane_types[k]["theta0"])
            lib.angle_inplane_types[k] = {"k": float(P["angle_k"][off + m]) / KCAL_TO_EV,
                                          "theta0": [float(x) / DEG for x in P["angle_theta0"][off + m][:n]]}
        for m, k in enumerate(self.strbnd_keys):
            lib.strbnd_types[k] = {"k1": float(P["strbnd_k"][m, 0]) / KCAL_TO_EV,
                                   "k2": float(P["strbnd_k"][m, 1]) / KCAL_TO_EV}
        for m, k in enumerate(self.urey_keys):
            lib.urey_types[k] = {"k": float(P["urey_k"][m]) / KCAL_TO_EV, "r0": float(P["urey_r0"][m])}
        for m, k in enumerate(self.opbend_keys):
            lib.opbend_types[k] = {"k": float(P["opbend_k"][m]) / KCAL_TO_EV}
        for m, k in enumerate(self.torsion_keys):
            terms = lib.torsion_types[k]["terms"]
            lib.torsion_types[k] = {"terms": [[float(P["torsion_k"][m, j]) / KCAL_TO_EV, int(n), float(ph)]
                                              for j, (_, n, ph) in enumerate(terms[:MAX_TORSION_TERMS])]}
        for m, k in enumerate(self.pitors_keys):
            lib.pitors_types[k] = {"k": float(P["pitors_k"][m]) / KCAL_TO_EV}
        for m, k in enumerate(self.tortor_keys):
            lib.tortor_types[k]["values"] = (P[f"tortor_f_{m}"] / KCAL_TO_EV).tolist()
        q_rows, d_rows, Q_rows = (x.detach().cpu().double() for x in self.effective_multipoles())
        for m, (t, frame) in enumerate(self.multipole_rows):
            for e in lib.multipoles[t]:
                if tuple(str(x) for x in e["frame"]) == frame:
                    e["q"] = float(q_rows[m])
                    e["dipole"] = (d_rows[m] / BOHR).tolist()
                    e["quadrupole"] = (Q_rows[m] * 3.0 / (BOHR * BOHR)).tolist()
        for i, t in enumerate(self.type_names):
            if t in lib.polarize:
                lib.polarize[t]["alpha"] = float(P["polarizability"][i])
                lib.polarize[t]["thole"] = float(P["thole"][i])
        for m, k in enumerate(self.bndcflux_keys):
            lib.bndcflux[k] = {"j": float(P["bndcflux_j"][m])}
        for m, k in enumerate(self.angcflux_keys):
            v = P["angcflux"][m]
            lib.angcflux[k] = {"jt1": float(v[0]), "jt2": float(v[1]), "jb1": float(v[2]), "jb2": float(v[3])}
        return lib


def _bfs_separations(n: int, neighbors: list[list[int]], max_sep: int) -> dict[tuple[int, int], int]:
    """Bonded separation (number of bonds on the shortest path) of every
    pair up to ``max_sep``, as ``{(i, j): sep}`` with ``i < j``."""
    out: dict[tuple[int, int], int] = {}
    for start in range(n):
        dist = {start: 0}
        frontier = [start]
        for sep in range(1, max_sep + 1):
            nxt = []
            for a in frontier:
                for b in neighbors[a]:
                    if b not in dist:
                        dist[b] = sep
                        nxt.append(b)
            frontier = nxt
        for b, sep in dist.items():
            if b > start:
                out[(start, b)] = sep
    return out


@register_model("amoeba")
class Amoeba(InteratomicPotential):
    """The AMOEBA polarizable multipole force field.

    An instance binds a parameter library to one molecular topology; the
    valence terms, multipole frames, polarization groups and nonbonded scale
    factors are resolved once, and ``forward`` evaluates the energy on any
    conformation (or batch of conformations) of that system. Forces come
    from autograd via :class:`~xnn.common.models.outputs.ForceStressOutput`.

    Parameters
    ----------
    ffield : AmoebaLibrary, AmoebaForceField, str or Path
        The parameter library: a parsed :class:`~xnn.ffnn.models.amoebalib.AmoebaLibrary`,
        a set shipped with xnn (``"amoeba-water-2003"``, ``"amoebapro13"``),
        a ``.frc`` / Tinker ``.prm`` / JSON path, or an :class:`AmoebaForceField`
        to share parameters with other models.
    topology : MolecularTopology, str or Path
        The system's topology (or a topology JSON path).
    cutoff : float, optional
        Neighbor-list cutoff in Angstrom, by default 10.0. For clusters use
        a value that encloses the whole system (every pair interacts in
        full); for cells it bounds the real-space cutoffs below.
    trainable : sequence of str or "all", optional
        Force-field groups (see :class:`AmoebaForceField`) plus the model
        group ``"atom_ref"``; ``"all"`` unfreezes everything.
    polarization : str, optional
        ``"mutual"`` solves the induced dipoles self-consistently,
        ``"direct"`` uses the permanent field only and ``"none"`` switches
        the induction off; by default the library's ``polarization``
        setting (``MUTUAL`` unless the file says ``DIRECT``).
    keep_intermediates : bool, optional
        Stash the intermediate tensors of the last evaluation in
        ``self.intermediates``.
    atomic_energies : mapping, optional
        Per-element reference energies in eV loaded into ``atom_ref``.
    ewald : bool, optional
        Ewald sums for the electrostatics of periodic cells: ``None``
        (default) switches them on for every structure that has a cell,
        ``False`` truncates at the cutoff instead (minimum image). A slab
        (two periodic axes) is summed on the full cell, which must hold a
        vacuum gap along the free axis, with the Yeh-Berkowitz correction.
    ewald_alpha, pewald_alpha, dewald_alpha : float, optional
        Ewald coefficients (1/Angstrom) of the permanent multipoles, the
        induction and HIPPO's dispersion; the defaults make the screened
        interaction at the real-space cutoff ``1e-8``, the induction one
        equal to the multipole one.
    ewald_cutoff : float, optional
        Real-space cutoff of the Ewald sums, by default the smaller of
        7 Angstrom and ``cutoff``.
    pme_grid, dpme_grid : sequence of int, optional
        Mesh dimensions of the electrostatic and dispersion particle-mesh
        sums; by default 1.2 (0.8) points per Angstrom rounded up to an
        FFT-friendly even size, at least 16.
    pme_order, ppme_order, dpme_order : int, optional
        B-spline orders of the multipole, induction and dispersion meshes
        (5, the multipole one, 4).
    vdw_cutoff, repulsion_cutoff, chgtrn_cutoff, dispersion_cutoff : float, optional
        Cutoffs of the switched short-range terms, by default ``cutoff``
        (the dispersion one is the Ewald real-space cutoff when its sum is
        on).
    vdw_taper, taper : float, optional
        Start of the C2 switch of the van der Waals term, and of every
        switched term, in Angstrom; by default 0.9 of each cutoff for
        cells and no switching for molecules.
    mesh_self : bool, optional
        Remove the spurious, position-averaged mesh self-interaction of
        every multipole from the particle-mesh energy, from the field
        that polarizes it and from the dispersion sum. Off by default:
        the plain smooth PME sums, the same algorithm as Tinker. On, the
        mean discretisation error of the self-interaction leaves the
        energies and the stress, which helps the induction, the
        dispersion and the pressure at coarse meshes.
    polar_eps : float, optional
        Convergence of the iterative induced-dipole solve, the rms change
        per iteration in Debye (``1e-8``).
    solver : str, optional
        ``"auto"`` (dense for molecules up to a thousand atoms, iterative
        otherwise), ``"dense"`` or ``"iterative"``.
    max_iter : int, optional
        Iteration limit of the solve (100).
    conserve_charge : bool, optional
        Hold the total charge of every molecule at its library value while
        ``charge`` trains (a minimum-norm projection of the row charges,
        registered on the force field), by default ``True``.
    symmetry_mask : bool, optional
        Keep the dipole and quadrupole components that the library leaves
        zero at zero while they train, by default ``True``. The quadrupoles
        stay symmetric and traceless in any case, and ``export_library``
        writes the multipoles the model uses.

    Notes
    -----
    ``forward`` returns ``node_energy`` / ``energy`` / ``node_features``,
    the per-atom ``charges`` (monopoles, charge flux included), the
    ``induced_dipoles`` ``(N, 3)`` in e Angstrom, the molecular ``dipole``
    ``(B, 3)`` in e Angstrom, and one per-structure tensor per term:
    ``e_bond``, ``e_angle``, ``e_strbnd``, ``e_urey``, ``e_opbend``,
    ``e_torsion``, ``e_pitors``, ``e_tortor``, ``e_vdw``, ``e_multipole``,
    ``e_polar`` (and for HIPPO ``e_repulsion``, ``e_dispersion``,
    ``e_chgtrn`` instead of ``e_vdw``), and ``e_ref``.
    """

    nonbonded = "amoeba"

    def __init__(self, ffield, topology: Union[MolecularTopology, str, Path], *,
                 cutoff: float = 10.0,
                 trainable: Union[Sequence[str], str] = (),
                 polarization: Optional[str] = None,
                 keep_intermediates: bool = False,
                 atomic_energies=None,
                 ewald: Optional[bool] = None,
                 ewald_alpha: Optional[float] = None,
                 pewald_alpha: Optional[float] = None,
                 dewald_alpha: Optional[float] = None,
                 ewald_cutoff: Optional[float] = None,
                 pme_grid: Optional[Sequence[int]] = None,
                 pme_order: int = 5,
                 ppme_order: Optional[int] = None,
                 dpme_grid: Optional[Sequence[int]] = None,
                 dpme_order: int = 4,
                 vdw_cutoff: Optional[float] = None,
                 vdw_taper: Optional[float] = None,
                 taper: Optional[float] = None,
                 repulsion_cutoff: Optional[float] = None,
                 chgtrn_cutoff: Optional[float] = None,
                 dispersion_cutoff: Optional[float] = None,
                 polar_eps: float = 1e-8,
                 mesh_self: bool = False,
                 solver: str = "auto",
                 max_iter: int = 100,
                 conserve_charge: bool = True,
                 symmetry_mask: bool = True):
        super().__init__()
        self.ewald = ewald
        self.ewald_alpha = None if ewald_alpha is None else float(ewald_alpha)
        self.pewald_alpha = None if pewald_alpha is None else float(pewald_alpha)
        self.dewald_alpha = None if dewald_alpha is None else float(dewald_alpha)
        self.ewald_cutoff = None if ewald_cutoff is None else float(ewald_cutoff)
        self.pme_grid = _grid_option(pme_grid)
        self.pme_order = int(pme_order)
        self.ppme_order = None if ppme_order is None else int(ppme_order)
        self.dpme_grid = _grid_option(dpme_grid)
        self.dpme_order = int(dpme_order)
        self.vdw_cutoff = None if vdw_cutoff is None else float(vdw_cutoff)
        self.vdw_taper = None if vdw_taper is None else float(vdw_taper)
        self.taper = None if taper is None else float(taper)
        self.repulsion_cutoff = None if repulsion_cutoff is None else float(repulsion_cutoff)
        self.chgtrn_cutoff = None if chgtrn_cutoff is None else float(chgtrn_cutoff)
        self.dispersion_cutoff = None if dispersion_cutoff is None else float(dispersion_cutoff)
        self.polar_eps = float(polar_eps)
        self.mesh_self = bool(mesh_self)
        if solver not in ("auto", "dense", "iterative"):
            raise ValueError("solver must be 'auto', 'dense' or 'iterative'")
        self.solver = solver
        self.max_iter = int(max_iter)
        self._force_dense = False
        groups, model_groups = _split_trainable(trainable)
        self.ff = (ffield if isinstance(ffield, AmoebaForceField)
                   else AmoebaForceField(ffield, trainable=groups, symmetry_mask=symmetry_mask))
        self.conserve_charge = bool(conserve_charge)
        if self.ff.model != self.nonbonded:
            raise ValueError(f"library {self.ff.name!r} carries {self.ff.model.upper()} nonbonded terms; "
                             f"use the {self.ff.model.capitalize()} model class")
        self.atom_ref = nn.Embedding(_MAX_Z, 1)
        nn.init.zeros_(self.atom_ref.weight)
        self.atom_ref.weight.requires_grad_("atom_ref" in model_groups)
        self.cutoff = float(cutoff)
        if polarization is None:
            polarization = str(self.ff.settings.get("polarization", "MUTUAL"))
        polarization = str(polarization).lower()
        if polarization not in ("mutual", "direct", "none"):
            raise ValueError("polarization must be 'mutual', 'direct' or 'none'")
        self.polarization = polarization
        self.keep_intermediates = bool(keep_intermediates)
        self.intermediates: dict = {}
        self.node_feature_dim = 1
        self._bind_topology(topology if isinstance(topology, MolecularTopology) else read_topology(topology))
        self.species = sorted({int(z) for z in self.top_z.tolist()})
        if atomic_energies is not None:
            from ...common.finetune.reference import set_atomic_energies
            set_atomic_energies(self, atomic_energies)

    # construction helpers
    @classmethod
    def from_atoms(cls, structure, ffield="amoeba-water-2003", *, charge: int = 0,
                   bonds: Optional[Sequence[Sequence[int]]] = None, orders: bool = True, **kwargs):
        """Build a model for a structure, typing it from the library's templates.

        Parameters
        ----------
        structure : object
            An ``ase.Atoms``, a ``(positions, atomic_numbers)`` pair, a dict
            with ``"pos"`` / ``"atomic_numbers"``, an RDKit molecule or a
            SMILES string.
        ffield : AmoebaLibrary, AmoebaForceField, str or Path, optional
            The parameter library (must carry templates or fragments).
        charge : int, optional
            Total charge (for bond-order perception).
        bonds : sequence of (int, int), optional
            Known connectivity.
        orders : bool, optional
            With ``bonds`` given, ``False`` skips bond-order perception.
        **kwargs
            Forwarded to the constructor.

        Returns
        -------
        Amoeba or Hippo
            The model bound to the derived topology.
        """
        from ..common.typing import assign_atom_types, perceive_bonds
        lib = ffield if isinstance(ffield, AmoebaForceField) else _as_library(ffield)
        types, details = assign_atom_types(structure, lib, charge=charge, bonds=bonds,
                                           return_details=True, orders=orders)
        residues = details["residues"]
        has_res = any(r is not None for r in residues)
        top = MolecularTopology.from_bonds(
            types, perceive_bonds(details["mol"]),
            residues=residues if has_res else None,
            residue_names=residue_name_list(residues, details["fragment_names"]) if has_res else None)
        return cls(lib, top, **kwargs)

    @classmethod
    def from_tinker_xyz(cls, path: Union[str, Path], ffield, type_map: Optional[dict] = None, **kwargs):
        """Build a model from a Tinker ``.xyz`` file (types and connectivity).

        Parameters
        ----------
        path : str or Path
            The coordinate file.
        ffield : AmoebaLibrary, AmoebaForceField, str or Path
            The parameter library; the file's integer types must be the
            library's type names, or be translated by ``type_map``.
        type_map : dict, optional
            ``{integer type: type name}``.
        **kwargs
            Forwarded to the constructor.

        Returns
        -------
        tuple
            ``(model, structure)`` with ``structure`` a dict of ``pos``,
            ``atomic_numbers`` (and ``cell`` when the file has one) ready for
            :func:`~xnn.common.data.structure_to_graph`.
        """
        from .amoebalib import read_tinker_xyz
        lib = ffield if isinstance(ffield, AmoebaForceField) else _as_library(ffield)
        data = read_tinker_xyz(path, library=lib.library if isinstance(lib, AmoebaForceField) else lib,
                               type_map=type_map)
        top = MolecularTopology.from_bonds(data["types"], data["bonds"])
        model = cls(lib, top, **kwargs)
        structure = {"pos": np.asarray(data["positions"], dtype=float),
                     "atomic_numbers": np.asarray([int(z) for z in model.top_z.tolist()])}
        if data["cell"] is not None:
            a, b, c = data["cell"][:3]
            structure["cell"] = np.diag([a, b, c])
        return model, structure

    # topology binding
    def _bind_topology(self, top: MolecularTopology) -> None:
        """Resolve the topology against the force field into index buffers."""
        self.topology = top
        ff = self.ff
        lib = ff.library
        n = top.n_atoms
        dtype = torch.get_default_dtype()
        missing: list[str] = []

        def miss(msg: str):
            if msg not in missing:
                missing.append(msg)

        t_idx = []
        for name in top.types:
            try:
                t_idx.append(ff.type_index(name))
            except KeyError as err:
                miss(str(err.args[0]))
                t_idx.append(0)
        cls = [ff.class_names[ff.type_class[i]] for i in t_idx]
        neighbors: list[list[int]] = [[] for _ in range(n)]
        for i, j in top.bonds:
            neighbors[i].append(j)
            neighbors[j].append(i)
        for nb in neighbors:
            nb.sort()
        z = [int(ff.type_z[i]) for i in t_idx]
        n_hyd = [sum(1 for j in nb if z[j] == 1) for nb in neighbors]

        def buf(name, data, shape):
            self.register_buffer(name, torch.tensor(data, dtype=torch.long).reshape(*shape))

        # bonds
        b_type = []
        for i, j in top.bonds:
            key = _key2(cls[i], cls[j])
            if key not in ff._bond_index:
                miss(f"no bond parameters for classes {key}")
                b_type.append(0)
            else:
                b_type.append(ff._bond_index[key])
        buf("bond_index", [list(b) for b in top.bonds], (-1, 2))
        self.bond_index = self.bond_index.t().contiguous()
        buf("bond_type", b_type, (-1,))

        # angles (in-plane at trigonal centers when a projected row exists),
        # with the hydrogen-environment choice of the ideal angle
        a_type, a_inplane, a_env, a_fourth = [], [], [], []
        for i, j, k in top.angles:
            key = _key3(cls[i], cls[j], cls[k])
            row = None
            inplane = False
            if len(neighbors[j]) == 3 and key in ff._angle_inplane_index:
                row = ff._angle_inplane_index[key]
                inplane = True
            elif key in ff._angle_index:
                row = ff._angle_index[key]
            if row is None:
                miss(f"no angle parameters for classes {key}")
                row = 0
            a_type.append(row)
            a_inplane.append(inplane)
            nh = n_hyd[j] - (1 if z[i] == 1 else 0) - (1 if z[k] == 1 else 0)
            table = lib.angle_inplane_types if inplane else lib.angle_types
            n_theta = len(table.get(key, {"theta0": [0.0]})["theta0"]) if key in table else 1
            a_env.append(min(max(nh, 0), max(n_theta - 1, 0)))
            fourth = [m for m in neighbors[j] if m not in (i, k)]
            a_fourth.append(fourth[0] if inplane else -1)
        buf("angle_index", [list(a) for a in top.angles], (-1, 3))
        self.angle_index = self.angle_index.t().contiguous()
        buf("angle_type", a_type, (-1,))
        buf("angle_env", a_env, (-1,))
        buf("angle_fourth", a_fourth, (-1,))
        self.register_buffer("angle_inplane", torch.tensor(a_inplane, dtype=torch.bool))

        # stretch-bend and Urey-Bradley on the angles that have them
        sb_idx, sb_type, sb_flip = [], [], []
        ub_idx, ub_type = [], []
        for m, (i, j, k) in enumerate(top.angles):
            key, flipped = _canon3(cls[i], cls[j], cls[k])
            if key in ff._strbnd_index:
                sb_idx.append(m)
                sb_type.append(ff._strbnd_index[key])
                sb_flip.append(flipped)
            if key in ff._urey_index:
                ub_idx.append(m)
                ub_type.append(ff._urey_index[key])
        buf("strbnd_angle", sb_idx, (-1,))
        buf("strbnd_type", sb_type, (-1,))
        self.register_buffer("strbnd_flip", torch.tensor(sb_flip, dtype=torch.bool))
        buf("urey_angle", ub_idx, (-1,))
        buf("urey_type", ub_type, (-1,))

        # out-of-plane bends at trigonal centers
        op_idx, op_type = [], []
        for j in range(n):
            if len(neighbors[j]) != 3:
                continue
            a, b, c = neighbors[j]
            for d, (p, q) in ((a, (b, c)), (b, (a, c)), (c, (a, b))):
                row = lib.opbend(cls[d], cls[j], cls[p], cls[q])
                if row is None:
                    continue
                key = _opbend_key(lib, cls[d], cls[j], cls[p], cls[q])
                op_idx.append([d, j, p, q])
                op_type.append(ff._opbend_index[key])
        buf("opbend_index", op_idx, (-1, 4))
        self.opbend_index = self.opbend_index.t().contiguous()
        buf("opbend_type", op_type, (-1,))

        # torsions
        from .oplslib import resolve_dihedral_type
        d_idx, d_type = [], []
        for i, j, k, l in top.dihedrals:
            key = resolve_dihedral_type(lib.torsion_types, cls[i], cls[j], cls[k], cls[l])
            if key is None:
                continue
            d_idx.append([i, j, k, l])
            d_type.append(ff._torsion_index[key])
        buf("torsion_index", d_idx, (-1, 4))
        self.torsion_index = self.torsion_index.t().contiguous()
        buf("torsion_type", d_type, (-1,))

        # pi-torsions: bonds between two trigonal centers
        pt_idx, pt_type = [], []
        for j, k in top.bonds:
            if len(neighbors[j]) != 3 or len(neighbors[k]) != 3:
                continue
            key = _key2(cls[j], cls[k])
            if key not in ff._pitors_index:
                continue
            a, b = [m for m in neighbors[j] if m != k]
            e, g = [m for m in neighbors[k] if m != j]
            pt_idx.append([a, b, j, k, e, g])
            pt_type.append(ff._pitors_index[key])
        buf("pitors_index", pt_idx, (-1, 6))
        self.pitors_index = self.pitors_index.t().contiguous()
        buf("pitors_type", pt_type, (-1,))

        # torsion-torsions: chains of five bonded atoms
        tt_idx, tt_type, tt_rev = [], [], []
        if lib.tortor_types:
            seen = set()
            for j, k, l, m in top.dihedrals:
                for i in neighbors[j]:
                    if i in (k, l, m):
                        continue
                    chain = (i, j, k, l, m)
                    canon = min(chain, chain[::-1])
                    if canon in seen:
                        continue
                    seen.add(canon)
                    hit = lib.tortor(cls[i], cls[j], cls[k], cls[l], cls[m])
                    if hit is None:
                        continue
                    row, reverse = hit
                    key = _fmt5(cls[i], cls[j], cls[k], cls[l], cls[m]) if not reverse \
                        else _fmt5(cls[m], cls[l], cls[k], cls[j], cls[i])
                    tt_idx.append(list(chain))
                    tt_type.append(ff._tortor_index[key])
                    tt_rev.append(reverse)
        buf("tortor_index", tt_idx, (-1, 5))
        self.tortor_index = self.tortor_index.t().contiguous()
        buf("tortor_type", tt_type, (-1,))
        self.register_buffer("tortor_reverse", torch.tensor(tt_rev, dtype=torch.bool))
        self._tortor_tables = []
        for m in range(len(ff.tortor_keys)):
            row = lib.tortor_types[ff.tortor_keys[m]]
            fx, fy, fxy = tortor_spline_tables(row["angles1"], row["angles2"], row["values"])
            self.register_buffer(f"tortor_fx_{m}", torch.tensor(fx * KCAL_TO_EV, dtype=dtype))
            self.register_buffer(f"tortor_fy_{m}", torch.tensor(fy * KCAL_TO_EV, dtype=dtype))
            self.register_buffer(f"tortor_fxy_{m}", torch.tensor(fxy * KCAL_TO_EV, dtype=dtype))

        # multipole frames
        frame_type, frame_z, frame_x, frame_y, mp_row = [], [], [], [], []
        for i in range(n):
            t = top.types[i]
            choice = self._choose_frame(i, t, top.types, neighbors, lib, ff)
            if choice is None:
                if t in ff._mpole_index:
                    miss(f"no multipole frame of type {t} fits atom {i} (neighbours "
                         f"{[top.types[j] for j in neighbors[i]]})")
                frame_type.append(FRAME_NONE)
                frame_z.append(i)
                frame_x.append(i)
                frame_y.append(i)
                mp_row.append(-1)
                continue
            row, ftype, iz, ix, iy = choice
            frame_type.append(ftype)
            frame_z.append(iz if iz >= 0 else i)
            frame_x.append(ix if ix >= 0 else i)
            frame_y.append(iy if iy >= 0 else i)
            mp_row.append(row)
        buf("frame_type", frame_type, (-1,))
        buf("frame_z", frame_z, (-1,))
        buf("frame_x", frame_x, (-1,))
        buf("frame_y", frame_y, (-1,))
        buf("mpole_row", mp_row, (-1,))
        self.register_buffer("has_mpole", torch.tensor([r >= 0 for r in mp_row], dtype=torch.bool))

        # every molecule (connected component) keeps its total charge while the charges train
        if self.conserve_charge and any(r >= 0 for r in mp_row):
            q_rows = ff.charges.detach()
            seen = [False] * n
            for start in range(n):
                if seen[start]:
                    continue
                members, stack = [], [start]
                seen[start] = True
                while stack:
                    a = stack.pop()
                    members.append(a)
                    for b in neighbors[a]:
                        if not seen[b]:
                            seen[b] = True
                            stack.append(b)
                counts = torch.zeros(q_rows.shape[0], dtype=q_rows.dtype, device=q_rows.device)
                for a in members:
                    if mp_row[a] >= 0:
                        counts[mp_row[a]] += 1.0
                if float(counts.sum()) > 0:
                    ff.constrain_charges(counts, float(counts @ q_rows))

        # polarization groups and their connectivity
        group = list(range(n))

        def find(a):
            while group[a] != a:
                group[a] = group[group[a]]
                a = group[a]
            return a

        for i, j in top.bonds:
            gi, gj = ff.polar_groups[t_idx[i]], ff.polar_groups[t_idx[j]]
            if top.types[j] in gi or top.types[i] in gj:
                group[find(i)] = find(j)
        roots = [find(i) for i in range(n)]
        labels = {r: m for m, r in enumerate(dict.fromkeys(roots))}
        atom_group = [labels[r] for r in roots]
        n_groups = len(labels)
        gneigh: list[set] = [set() for _ in range(n_groups)]
        for i, j in top.bonds:
            if atom_group[i] != atom_group[j]:
                gneigh[atom_group[i]].add(atom_group[j])
                gneigh[atom_group[j]].add(atom_group[i])
        gsep = _bfs_separations(n_groups, [sorted(s) for s in gneigh], 3)
        buf("atom_group", atom_group, (-1,))

        # pair scale factors by bonded separation and group separation
        sep = _bfs_separations(n, neighbors, 4)
        s = ff.settings
        mpole = [1.0] + [float(s[f"mpole-1{k}-scale"]) for k in (2, 3, 4, 5)]
        vdw = [1.0] + [float(s[f"vdw-1{k}-scale"]) for k in (2, 3, 4, 5)]
        polar = [1.0] + [float(s[f"polar-1{k}-scale"]) for k in (2, 3, 4, 5)]
        intra = [1.0] + [float(s[f"polar-1{k}-intra"]) for k in (2, 3, 4, 5)]
        induce = [1.0] + [float(s[f"induce-1{k}-scale"]) for k in (2, 3, 4, 5)]
        rep = [1.0] + [float(s[f"rep-1{k}-scale"]) for k in (2, 3, 4, 5)]
        disp = [1.0] + [float(s[f"disp-1{k}-scale"]) for k in (2, 3, 4, 5)]
        direct = [float(s[f"direct-1{k}-scale"]) for k in (1, 2, 3, 4)]
        mutual = [float(s[f"mutual-1{k}-scale"]) for k in (1, 2, 3, 4)]
        special: dict[tuple[int, int], list] = {}
        for (i, j), k in sep.items():
            same = atom_group[i] == atom_group[j]
            special[(i, j)] = [vdw[k], mpole[k], intra[k] if same else polar[k], induce[k], rep[k], disp[k],
                               1.0, 1.0]
        for i in range(n):
            for j in range(i + 1, n):
                gi, gj = atom_group[i], atom_group[j]
                g = 0 if gi == gj else gsep.get((min(gi, gj), max(gi, gj)), 99)
                if g < 4:
                    row = special.setdefault((i, j), [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
                    row[6] = direct[g]
                    row[7] = mutual[g]
        if bool(s.get("d-equals-p", False)):
            # Tinker's d-equals-p: the direct field uses the polar (connectivity) scale factors, and
            # the mutual interactions the induce factors alone
            for row in special.values():
                row[6] = row[2]
                row[7] = 1.0
        keys = sorted(special)
        self.register_buffer("special_keys", torch.tensor([i * n + j for i, j in keys], dtype=torch.long))
        self.register_buffer("special_scales", torch.tensor([special[k] for k in keys] or [[1.0] * 8],
                                                            dtype=dtype).reshape(-1, 8))

        # van der Waals sites: hydrogens sit on their bond at the reduction factor
        vdw_parent = list(range(n))
        vdw_red = [1.0] * n
        for i in range(n):
            red = float(ff.vdw_reduction[ff.type_class[t_idx[i]]])
            if len(neighbors[i]) == 1 and red != 1.0:
                vdw_parent[i] = neighbors[i][0]
                vdw_red[i] = red
        buf("vdw_parent", vdw_parent, (-1,))
        self.register_buffer("vdw_red", torch.tensor(vdw_red, dtype=dtype))

        # charge flux terms
        cf_b_idx, cf_b_type, cf_b_sign = [], [], []
        for m, (i, j) in enumerate(top.bonds):
            key, flipped = _canon2(cls[i], cls[j])
            if key in ff._bndcflux_index:
                cf_b_idx.append(m)
                cf_b_type.append(ff._bndcflux_index[key])
                cf_b_sign.append(-1.0 if flipped else 1.0)
        buf("cflux_bond", cf_b_idx, (-1,))
        buf("cflux_bond_type", cf_b_type, (-1,))
        self.register_buffer("cflux_bond_sign", torch.tensor(cf_b_sign or [0.0], dtype=dtype)[:len(cf_b_idx)])
        cf_a_idx, cf_a_type, cf_a_flip, cf_a_bonds = [], [], [], []
        bond_of = {b: m for m, b in enumerate(top.bonds)}
        for m, (i, j, k) in enumerate(top.angles):
            key, flipped = _canon3(cls[i], cls[j], cls[k])
            if key in ff._angcflux_index:
                cf_a_idx.append(m)
                cf_a_type.append(ff._angcflux_index[key])
                cf_a_flip.append(flipped)
                cf_a_bonds.append([bond_of[(min(i, j), max(i, j))], bond_of[(min(j, k), max(j, k))]])
        buf("cflux_angle", cf_a_idx, (-1,))
        buf("cflux_angle_type", cf_a_type, (-1,))
        self.register_buffer("cflux_angle_flip", torch.tensor(cf_a_flip, dtype=torch.bool))
        buf("cflux_angle_bonds", cf_a_bonds, (-1, 2))

        if missing:
            raise KeyError("the topology needs parameters the library does not provide:\n  "
                           + "\n  ".join(missing))
        buf("top_type", t_idx, (-1,))
        buf("top_class", [ff.type_class[i] for i in t_idx], (-1,))
        self.register_buffer("top_z", ff.type_z[self.top_type].clone())
        self.n_bonded_sites = n

    @staticmethod
    def _choose_frame(i: int, t: str, types: Sequence[str], neighbors, lib: AmoebaLibrary, ff):
        """Pick the multipole row of type ``t`` whose frame atoms exist around
        atom ``i``: the z atom among the bonded neighbours, the x (and y)
        atoms among the bonded neighbours, or, if no row resolves that way,
        among the bonded and the 1,3 atoms (the water hydrogen's x atom is
        the other hydrogen); within a pass the first row in file order wins
        (the alpha carbon next to a terminal residue gets the row naming the
        terminal types)."""
        rows = ff._mpole_index.get(t, [])
        if not rows:
            return None
        bonded = list(neighbors[i])
        one_three = sorted({k for j in bonded for k in neighbors[j] if k != i and k not in bonded})
        # two passes: rows whose frame atoms are all bonded neighbours win over rows
        # that need a 1,3 atom for the x or y axis (the z atom is always bonded)
        for pools in ([bonded], [bonded, one_three]):
            for row in rows:
                _, (kz, kx, ky) = ff.multipole_rows[row]
                z_name, x_name, y_name = (str(kz), str(kx), str(ky))
                if z_name in ("0", ""):
                    return row, FRAME_NONE, -1, -1, -1
                neg_z = z_name.startswith("-")
                neg_x = x_name.startswith("-")
                neg_y = y_name.startswith("-")
                z_t, x_t, y_t = z_name.lstrip("-"), x_name.lstrip("-"), y_name.lstrip("-")
                used: list[int] = []

                def pick(name, candidates):
                    for pool in candidates:
                        for a in pool:
                            if a not in used and types[a] == name:   # noqa: B023  # called in this iteration
                                used.append(a)   # noqa: B023
                                return a
                    return -1

                iz = pick(z_t, [bonded])
                if iz < 0:
                    continue
                if x_name in ("0", ""):
                    return row, FRAME_ZONLY, iz, -1, -1
                ix = pick(x_t, pools)
                if ix < 0:
                    continue
                if y_name in ("0", ""):
                    ftype = FRAME_BISECTOR if (neg_z and neg_x) else FRAME_ZTHENX
                    return row, ftype, iz, ix, -1
                iy = pick(y_t, pools)
                if iy < 0:
                    continue
                if neg_z and neg_x and neg_y:
                    ftype = FRAME_THREEFOLD
                elif neg_x and neg_y:
                    ftype = FRAME_ZBISECT
                else:
                    ftype = FRAME_ZTHENX
                return row, ftype, iz, ix, iy
        return None

    # geometry helpers
    def _frames(self, data: AtomicGraph) -> tuple[Tensor, Tensor]:
        """Rotation matrices ``(N, 3, 3)`` (columns x, y, z in the global
        frame) of every atom's local multipole frame, plus the chirality
        sign ``(N,)`` for frames with a y-defining atom."""
        B = data.num_graphs
        n = int(self.frame_type.shape[0])
        starts = torch.arange(B, device=data.pos.device) * n
        idx = torch.arange(n, device=data.pos.device).repeat(B) + starts.repeat_interleave(n)
        ftype = self.frame_type.repeat(B)
        iz = self.frame_z.repeat(B) + starts.repeat_interleave(n)
        ix = self.frame_x.repeat(B) + starts.repeat_interleave(n)
        iy = self.frame_y.repeat(B) + starts.repeat_interleave(n)
        vz = pair_vectors(data, idx, iz)
        vx = pair_vectors(data, idx, ix)
        vy = pair_vectors(data, idx, iy)
        uz, ux, uy = _unit(vz), _unit(vx), _unit(vy)
        is_bis = (ftype == FRAME_BISECTOR)[:, None]
        is_zb = (ftype == FRAME_ZBISECT)[:, None]
        is_3f = (ftype == FRAME_THREEFOLD)[:, None]
        is_zo = (ftype == FRAME_ZONLY)[:, None]
        is_none = (ftype == FRAME_NONE)[:, None]
        zaxis = torch.where(is_bis, uz + ux, torch.where(is_3f, uz + ux + uy, uz))
        zaxis = _unit(torch.where(is_none, torch.tensor([0.0, 0.0, 1.0], dtype=uz.dtype, device=uz.device)
                                  .expand_as(uz), zaxis))
        xref = torch.where(is_zb, ux + uy, ux)
        ex = torch.tensor([1.0, 0.0, 0.0], dtype=uz.dtype, device=uz.device).expand_as(uz)
        ey = torch.tensor([0.0, 1.0, 0.0], dtype=uz.dtype, device=uz.device).expand_as(uz)
        zonly_ref = torch.where((zaxis[:, :1].abs() > 0.866), ey, ex)
        xref = torch.where(is_zo | is_none, zonly_ref, xref)
        xaxis = _unit(xref - (xref * zaxis).sum(-1, keepdim=True) * zaxis)
        yaxis = torch.cross(zaxis, xaxis, dim=1)
        # chirality: a y-defining atom on the negative side flips the y-odd components
        has_y = (ftype == FRAME_ZTHENX) & (self.frame_y.repeat(B) != torch.arange(n, device=data.pos.device).repeat(B))
        side = (vy * yaxis).sum(-1)
        chir = torch.where(has_y & (side < 0), -torch.ones_like(side), torch.ones_like(side))
        return torch.stack([xaxis, yaxis, zaxis], dim=2), chir

    def _rotated_multipoles(self, data: AtomicGraph, B: int):
        """Global-frame charge, dipole and quadrupole of every atom."""
        row = self.mpole_row.repeat(B).clamp_min(0)
        has = self.has_mpole.repeat(B)
        q_rows, d_rows, Q_rows = self.ff.effective_multipoles()
        q = torch.where(has, q_rows[row], torch.zeros_like(q_rows[row]))
        d_loc = d_rows[row] * has[:, None].to(d_rows.dtype)
        Q_loc = Q_rows[row] * has[:, None, None].to(Q_rows.dtype)
        R, chir = self._frames(data)
        flip = torch.ones_like(d_loc)
        flip[:, 1] = chir
        d_loc = d_loc * flip
        Qflip = flip[:, :, None] * flip[:, None, :]
        Q_loc = Q_loc * Qflip
        d = torch.einsum("nab,nb->na", R, d_loc)
        Q = torch.einsum("nab,nbc,ndc->nad", R, Q_loc, R)
        return q, d, Q

    # pair bookkeeping
    def _pairs(self, data: AtomicGraph):
        """Unique nonbonded pairs ``(i, k, r_vec, r)`` of the neighbor list
        with their scale factors ``(M, 8)``: vdw, mpole, polar, induce, rep,
        disp, direct, mutual."""
        src, dst = data.edge_index[0], data.edge_index[1]
        vec = data.edge_vectors()
        keep = src < dst
        i, k, r = src[keep], dst[keep], vec[keep]
        n = int(self.frame_type.shape[0])
        start = data.batch[i] * n
        key = (i - start) * n + (k - start)
        if self.special_keys.numel() == 0:
            return i, k, r, torch.ones(i.shape[0], 8, dtype=r.dtype, device=r.device)
        pos = torch.searchsorted(self.special_keys, key).clamp_max(self.special_keys.shape[0] - 1)
        hit = self.special_keys[pos] == key
        if data.cell is not None and bool(hit.any()):
            # the scale factors belong to the minimum image of a bonded pair; any other
            # image within the cutoff is a plain pair
            sel = hit.nonzero()[:, 0]
            closest = pair_vectors(data, i[sel], k[sel]).detach()
            same = ((r[sel].detach() - closest) ** 2).sum(-1) < 1e-10
            hit = hit.clone()
            hit[sel] = same
        scales = torch.where(hit[:, None], self.special_scales[pos],
                             torch.ones_like(self.special_scales[pos]))
        return i, k, r, scales

    # forward
    def forward(self, data: AtomicGraph) -> dict[str, Tensor]:
        """Evaluate the energy on a (batched) atomic graph.

        Parameters
        ----------
        data : AtomicGraph
            The batched graph; every structure must be a conformation of
            the bound topology, with the neighbor list built with this
            model's ``cutoff``.

        Returns
        -------
        dict of str to Tensor
            See the class notes.

        Raises
        ------
        ValueError
            If a structure in the batch does not match the bound topology.
        """
        N = data.num_nodes
        B = data.num_graphs
        n = int(self.top_type.shape[0])
        device = data.pos.device
        P = self.ff.params
        ff = self.ff
        s = ff.settings
        if not bool((data.n_atoms == n).all()):
            raise ValueError(f"every structure must have the bound topology's {n} atoms "
                             f"(got {data.n_atoms.tolist()})")
        if not bool((data.atomic_numbers == self.top_z.repeat(B)).all()):
            raise ValueError("atomic numbers do not match the bound topology's element sequence")
        dtype = P["bond_k"].dtype
        starts = torch.arange(B, device=device) * n
        cls_idx = self.top_class.repeat(B)
        typ_idx = self.top_type.repeat(B)

        def expand(idx: Tensor) -> Tensor:
            return (idx.unsqueeze(1) + starts.view(1, B, 1)).reshape(idx.shape[0], -1)

        def tile(v: Tensor) -> Tensor:
            return v.repeat(B)

        # indices into the per-structure angle list (stretch-bend, Urey-Bradley) step by the
        # number of angles, not the number of atoms
        angle_starts = torch.arange(B, device=device) * int(self.angle_index.shape[1])

        def expand_angles(idx: Tensor) -> Tensor:
            return (idx.unsqueeze(1) + angle_starts.view(1, B, 1)).reshape(idx.shape[0], -1)

        out_terms: dict[str, Tensor] = {}
        zeros_n = torch.zeros(N, dtype=dtype, device=device)

        # bonds: K dr^2 (1 + c3 dr + c4 dr^2)
        bi = expand(self.bond_index)
        bt = tile(self.bond_type)
        bvec = pair_vectors(data, bi[0], bi[1])
        br = torch.sqrt((bvec * bvec).sum(1) + TINY)
        dr = br - P["bond_r0"][bt]
        e_bond = P["bond_k"][bt] * dr * dr * (1.0 + float(s["bond-cubic"]) * dr + float(s["bond-quartic"]) * dr * dr)
        out_terms["e_bond"] = scatter_sum(0.5 * e_bond, bi[0], N) + scatter_sum(0.5 * e_bond, bi[1], N)

        # angles (harmonic or in-plane projected), sextic polynomial in degrees
        ai = expand(self.angle_index)
        at = tile(self.angle_type)
        aenv = tile(self.angle_env)
        u = pair_vectors(data, ai[1], ai[0])
        v = pair_vectors(data, ai[1], ai[2])
        theta_plain = _angle(u, v)
        if bool(self.angle_inplane.any()):
            fourth = expand(self.angle_fourth.clamp_min(0).unsqueeze(0))[0]
            w = pair_vectors(data, ai[1], fourth)          # D - B
            ad = u - w                                     # A - D
            cd = v - w                                     # C - D
            nrm = torch.cross(ad, cd, dim=1)
            # projection of the center onto the plane of the three substituents
            delta = (nrm * w).sum(1) / ((nrm * nrm).sum(1) + TINY)
            shift = delta[:, None] * nrm                   # B' - B
            theta_ip = _angle(u - shift, v - shift)
            theta = torch.where(tile(self.angle_inplane), theta_ip, theta_plain)
        else:
            theta = theta_plain
        theta0 = torch.gather(P["angle_theta0"][at], 1, aenv[:, None])[:, 0]
        dt = theta - theta0
        dtd = dt / DEG
        e_angle = P["angle_k"][at] * dt * dt * (
            1.0 + float(s["angle-cubic"]) * dtd + float(s["angle-quartic"]) * dtd ** 2
            + float(s["angle-pentic"]) * dtd ** 3 + float(s["angle-sextic"]) * dtd ** 4)
        out_terms["e_angle"] = scatter_sum(e_angle, ai[1], N)

        # stretch-bend: (k1 dr_ab + k2 dr_cb) dtheta, with the plain angle
        if self.strbnd_angle.numel():
            sbi = expand_angles(self.strbnd_angle.unsqueeze(0))[0]
            sbt = tile(self.strbnd_type)
            flip = tile(self.strbnd_flip)
            a_sel = ai[:, sbi]
            r1 = torch.sqrt((pair_vectors(data, a_sel[1], a_sel[0]) ** 2).sum(1) + TINY)
            r2 = torch.sqrt((pair_vectors(data, a_sel[1], a_sel[2]) ** 2).sum(1) + TINY)
            k = P["strbnd_k"][sbt]
            k1 = torch.where(flip, k[:, 1], k[:, 0])
            k2 = torch.where(flip, k[:, 0], k[:, 1])
            # ideal bond lengths of the two bonds
            r01 = self._bond_r0_of(a_sel[1], a_sel[0], B)
            r02 = self._bond_r0_of(a_sel[1], a_sel[2], B)
            e_sb = (k1 * (r1 - r01) + k2 * (r2 - r02)) * (theta_plain[sbi] - theta0[sbi])
            out_terms["e_strbnd"] = scatter_sum(e_sb, a_sel[1], N)
        else:
            out_terms["e_strbnd"] = zeros_n

        # Urey-Bradley
        if self.urey_angle.numel():
            ubi = expand_angles(self.urey_angle.unsqueeze(0))[0]
            ubt = tile(self.urey_type)
            a_sel = ai[:, ubi]
            rv = pair_vectors(data, a_sel[0], a_sel[2])
            rr = torch.sqrt((rv * rv).sum(1) + TINY)
            e_ub = P["urey_k"][ubt] * (rr - P["urey_r0"][ubt]) ** 2
            out_terms["e_urey"] = scatter_sum(e_ub, a_sel[1], N)
        else:
            out_terms["e_urey"] = zeros_n

        # out-of-plane bends (Allinger or W-D-C angle), sextic polynomial in degrees
        if self.opbend_type.numel():
            oi = expand(self.opbend_index)
            ot = tile(self.opbend_type)
            d_, b_, a_, c_ = oi[0], oi[1], oi[2], oi[3]
            ab = pair_vectors(data, b_, a_)
            cb = pair_vectors(data, b_, c_)
            db = pair_vectors(data, b_, d_)
            ee = (db * torch.cross(ab, cb, dim=1)).sum(1)
            if str(s["opbendtype"]).upper().startswith("ALL"):
                ad = ab - db
                cd = cb - db
                cc = (torch.cross(ad, cd, dim=1) ** 2).sum(1)
            else:
                cc = (torch.cross(ab, cb, dim=1) ** 2).sum(1)
            rdb2 = (db * db).sum(1)
            sin2 = (ee * ee) / (cc * rdb2 + TINY)
            sin_chi = torch.sqrt(torch.clamp(sin2, 0.0, 1.0 - 1e-15))
            chi = torch.arcsin(sin_chi)
            chid = chi / DEG
            e_op = P["opbend_k"][ot] * chi * chi * (
                1.0 + float(s["opbend-cubic"]) * chid + float(s["opbend-quartic"]) * chid ** 2
                + float(s["opbend-pentic"]) * chid ** 3 + float(s["opbend-sextic"]) * chid ** 4)
            out_terms["e_opbend"] = scatter_sum(e_op, b_, N)
        else:
            out_terms["e_opbend"] = zeros_n

        # torsions
        if self.torsion_type.numel():
            di = expand(self.torsion_index)
            dtp = tile(self.torsion_type)
            c1, s1 = dihedral_cos_sin(data, di)
            nmax = int(ff.torsion_n.max())
            cs, ss = [c1], [s1]
            for _ in range(1, nmax):
                cs.append(cs[-1] * c1 - ss[-1] * s1)
                ss.append(ss[-1] * c1 + cs[-2] * s1)
            cos_n = torch.stack(cs, 1)
            sin_n = torch.stack(ss, 1)
            nn_ = ff.torsion_n[dtp]
            cn = torch.gather(cos_n, 1, nn_ - 1)
            sn = torch.gather(sin_n, 1, nn_ - 1)
            e_t = (P["torsion_k"][dtp] * ff.torsion_mask[dtp]
                   * (1.0 + ff.torsion_cos[dtp] * cn + ff.torsion_sin[dtp] * sn)).sum(1)
            out_terms["e_torsion"] = scatter_sum(e_t, di[1], N)
        else:
            out_terms["e_torsion"] = zeros_n

        # pi-torsions: dihedral between the p-orbital directions of two trigonal centers
        if self.pitors_type.numel():
            pi_ = expand(self.pitors_index)
            ptt = tile(self.pitors_type)
            a_, b_, c_, d_, e_, g_ = pi_
            pc = data.pos[c_].to(dtype)
            pd = data.pos[d_].to(dtype)
            ad = pair_vectors(data, d_, a_)
            bd = pair_vectors(data, d_, b_)
            ec = pair_vectors(data, c_, e_)
            gc = pair_vectors(data, c_, g_)
            p_site = pc + torch.cross(ad, bd, dim=1)
            q_site = pd + torch.cross(ec, gc, dim=1)
            b1 = pc - p_site
            b2 = pair_vectors(data, c_, d_)
            b3 = q_site - pd
            n1 = torch.cross(b1, b2, dim=1)
            n2 = torch.cross(b2, b3, dim=1)
            cosphi = (n1 * n2).sum(1) / torch.sqrt((n1 * n1).sum(1) * (n2 * n2).sum(1) + TINY)
            cos2 = 2.0 * cosphi * cosphi - 1.0
            e_pt = P["pitors_k"][ptt] * (1.0 - cos2)
            out_terms["e_pitors"] = scatter_sum(0.5 * e_pt, c_, N) + scatter_sum(0.5 * e_pt, d_, N)
        else:
            out_terms["e_pitors"] = zeros_n

        # torsion-torsion maps
        if self.tortor_type.numel():
            ti = expand(self.tortor_index)
            ttt = tile(self.tortor_type)
            rev = tile(self.tortor_reverse)
            c_a, s_a = dihedral_cos_sin(data, ti[:4])
            c_b, s_b = dihedral_cos_sin(data, ti[1:5])
            phi_a = torch.atan2(s_a, c_a) / DEG
            phi_b = torch.atan2(s_b, c_b) / DEG
            # the grid's first angle is the torsion of the chain's last four atoms
            # and the second the torsion of its first four, with the sign convention
            # of the Tinker torsion angle (the negative of the IUPAC angle used
            # elsewhere in this module); the chirality flip of chiral map centers
            # is not implemented (the AMOEBA 2013 maps belong to glycine)
            x = -torch.where(rev, phi_a, phi_b)
            y = -torch.where(rev, phi_b, phi_a)
            e_tt = torch.zeros_like(x)
            for m in range(len(ff.tortor_keys)):
                sel = ttt == m
                if not bool(sel.any()):
                    continue
                e_tt = e_tt + torch.where(sel, self._tortor_energy(m, x, y), torch.zeros_like(x))
            out_terms["e_tortor"] = scatter_sum(e_tt, ti[2], N)
        else:
            out_terms["e_tortor"] = zeros_n

        # permanent multipoles (global frame) and the charge flux
        q_perm, d_perm, Q_perm = self._rotated_multipoles(data, B)
        dq = self._charge_flux(data, B, br, bt, bi, theta_plain, theta0, ai, N) if self._has_cflux() else None
        if dq is not None:
            q_perm = q_perm + dq

        # nonbonded pairs
        i_, k_, r_vec, scales = self._pairs(data)
        R = torch.sqrt((r_vec * r_vec).sum(1) + TINY)
        ew = self._ewald_setup(data, i_, R)
        nb = self._nonbonded(data, i_, k_, r_vec, R, scales, q_perm, d_perm, Q_perm, cls_idx, typ_idx, B, N, ew)
        out_terms.update(nb["terms"])

        e_ref_node = self.atom_ref(data.atomic_numbers).squeeze(-1)
        out_terms["e_ref"] = e_ref_node
        node_energy = zeros_n
        for v in out_terms.values():
            node_energy = node_energy + v
        energy = self.aggregate_energy(node_energy, data)

        # molecular dipole: charges, permanent and induced dipoles
        pos = data.pos.to(dtype)
        mu_ind = nb["induced"]
        dip_atoms = q_perm[:, None] * pos + d_perm + mu_ind
        dipole = scatter_sum(dip_atoms, data.batch, B)

        if self.keep_intermediates:
            self.intermediates = dict(bond_r=br, e_bond=e_bond, angle_theta=theta, e_angle=e_angle,
                                      q=q_perm, d=d_perm, Q=Q_perm, induced=mu_ind, pair_i=i_, pair_k=k_,
                                      pair_r=R, pair_scales=scales, ewald=None if ew is None else ew["summary"],
                                      **nb.get("extra", {}))
        result = {"node_energy": node_energy, "energy": energy, "charges": q_perm,
                  "node_features": q_perm.unsqueeze(1), "induced_dipoles": mu_ind, "dipole": dipole}
        for k, v in out_terms.items():
            result[k] = self.aggregate_energy(v, data)
        return result

    # pieces of the forward
    def _bond_r0_of(self, a: Tensor, b: Tensor, B: int) -> Tensor:
        """Ideal length of the bond between atoms ``a`` and ``b`` (gathered
        from the bond table through the topology's bond list)."""
        n = int(self.top_type.shape[0])
        key_lookup = getattr(self, "_bond_key_lookup", None)
        if key_lookup is None:
            keys = torch.full((n * n,), -1, dtype=torch.long, device=a.device)
            bonds = self.bond_index
            idx = bonds[0] * n + bonds[1]
            keys[idx] = torch.arange(bonds.shape[1], device=a.device)
            keys[bonds[1] * n + bonds[0]] = torch.arange(bonds.shape[1], device=a.device)
            self._bond_key_lookup = keys
            key_lookup = keys
        la = a % n
        lb = b % n
        m = key_lookup[la * n + lb]
        return self.ff.params["bond_r0"][self.bond_type[m]]

    def _has_cflux(self) -> bool:
        return bool(self.cflux_bond.numel() or self.cflux_angle.numel())

    def _charge_flux(self, data, B, br, bt, bi, theta, theta0, ai, N) -> Tensor:
        """Per-atom charge changes from the bond and angle charge-flux terms."""
        P = self.ff.params
        dq = torch.zeros(N, dtype=br.dtype, device=br.device)
        if self.cflux_bond.numel():
            nbond = int(self.bond_index.shape[1])
            sel = (self.cflux_bond.unsqueeze(1) + torch.arange(B, device=br.device).view(1, B) * nbond).reshape(-1)
            j = P["bndcflux_j"][self.cflux_bond_type.repeat(B)] * self.cflux_bond_sign.repeat(B)
            flux = j * (br[sel] - P["bond_r0"][bt[sel]])
            dq = dq.index_add(0, bi[0][sel], -flux).index_add(0, bi[1][sel], flux)
        if self.cflux_angle.numel():
            nang = int(self.angle_index.shape[1])
            sel = (self.cflux_angle.unsqueeze(1) + torch.arange(B, device=br.device).view(1, B) * nang).reshape(-1)
            typ = self.cflux_angle_type.repeat(B)
            flip = self.cflux_angle_flip.repeat(B)
            par = P["angcflux"][typ]
            jt1 = torch.where(flip, par[:, 1], par[:, 0])
            jt2 = torch.where(flip, par[:, 0], par[:, 1])
            jb1 = torch.where(flip, par[:, 3], par[:, 2])
            jb2 = torch.where(flip, par[:, 2], par[:, 3])
            nbond = int(self.bond_index.shape[1])
            bsel = (self.cflux_angle_bonds.unsqueeze(1)
                    + torch.arange(B, device=br.device).view(1, B, 1) * nbond).reshape(-1, 2)
            dr_ab = br[bsel[:, 0]] - P["bond_r0"][bt[bsel[:, 0]]]
            dr_cb = br[bsel[:, 1]] - P["bond_r0"][bt[bsel[:, 1]]]
            dth = theta[sel] - theta0[sel]
            dq_a = jt1 * dth + jb1 * dr_cb
            dq_c = jt2 * dth + jb2 * dr_ab
            a_sel = ai[:, sel]
            dq = dq.index_add(0, a_sel[0], dq_a).index_add(0, a_sel[2], dq_c).index_add(0, a_sel[1], -(dq_a + dq_c))
        return dq

    def _tortor_energy(self, m: int, x: Tensor, y: Tensor) -> Tensor:
        """Bicubic spline energy of torsion-torsion map ``m`` at ``(x, y)`` degrees."""
        gx = getattr(self.ff, f"tortor_x_{m}")
        gy = getattr(self.ff, f"tortor_y_{m}")
        f = self.ff.params[f"tortor_f_{m}"]
        fx = getattr(self, f"tortor_fx_{m}")
        fy = getattr(self, f"tortor_fy_{m}")
        fxy = getattr(self, f"tortor_fxy_{m}")
        x = torch.clamp(x, float(gx[0]), float(gx[-1]) - 1e-9)
        y = torch.clamp(y, float(gy[0]), float(gy[-1]) - 1e-9)
        ix = (torch.searchsorted(gx, x.detach(), right=True) - 1).clamp(0, gx.shape[0] - 2)
        iy = (torch.searchsorted(gy, y.detach(), right=True) - 1).clamp(0, gy.shape[0] - 2)
        x1, x2 = gx[ix], gx[ix + 1]
        y1, y2 = gy[iy], gy[iy + 1]

        def corners(t):
            return torch.stack([t[ix, iy], t[ix + 1, iy], t[ix + 1, iy + 1], t[ix, iy + 1]], 1)

        return bicubic_interpolate(corners(f), corners(fx), corners(fy), corners(fxy), x, y, x1, x2, y1, y2)

    def _vdw_sites(self, data: AtomicGraph, B: int) -> Tensor:
        """Positions of the van der Waals sites (hydrogens reduced along their bond)."""
        n = int(self.top_type.shape[0])
        starts = torch.arange(B, device=data.pos.device) * n
        idx = torch.arange(n, device=data.pos.device).repeat(B) + starts.repeat_interleave(n)
        parent = self.vdw_parent.repeat(B) + starts.repeat_interleave(n)
        red = self.vdw_red.repeat(B)
        pos = data.pos
        vec = pair_vectors(data, parent, idx)         # H - X
        return pos[parent].to(vec.dtype) + red[:, None] * vec

    def _induced(self, i_, k_, r_vec, R, field_d, field_p, D1, D2, N, B, alpha, data, ew):
        """Solve the mutual induction and return ``(mu, e_polar_node)``.

        ``field_d`` / ``field_p`` are the direct- and polar-scaled permanent
        fields at every atom (mesh and self contributions included for
        cells); ``D1, D2`` the mutual-induction ladder per pair (scale
        factors and, for cells, the real-space Ewald part included);
        ``alpha`` the per-atom polarizability. Molecules are solved with a
        dense matrix (or iteratively when large), cells iteratively with
        the mesh field of the induced dipoles.
        """
        n = int(self.top_type.shape[0])
        polar = alpha > 0
        mu_dir = alpha[:, None] * field_d
        if self.polarization == "direct" or not bool(polar.any()):
            mu = mu_dir
        else:
            T = dipole_field_tensor(r_vec, D1, D2)                        # (M, 3, 3)
            if self._use_dense(N, ew):
                A = torch.zeros(B, n, n, 3, 3, dtype=R.dtype, device=R.device)
                bi = i_ // n
                li, lk = i_ % n, k_ % n
                A[bi, li, lk] = -T
                A[bi, lk, li] = -T.transpose(1, 2)
                inv_alpha = torch.where(polar, 1.0 / alpha.clamp_min(1e-30), torch.full_like(alpha, 1.0))
                eye = torch.eye(3, dtype=R.dtype, device=R.device)
                diag = inv_alpha.view(B, n)[:, :, None, None] * eye
                ar = torch.arange(n, device=R.device)
                A[:, ar, ar] = diag
                Afull = A.permute(0, 1, 3, 2, 4).reshape(B, 3 * n, 3 * n)
                rhs = (field_d * polar[:, None].to(R.dtype)).reshape(B, 3 * n, 1)
                mu = torch.linalg.solve(Afull, rhs).reshape(N, 3)
                mu = mu * polar[:, None].to(R.dtype)
                if self.keep_intermediates:
                    self._induction_matrix = Afull.detach()
            else:
                dtype = R.dtype
                pos = data.pos.to(dtype)
                cell = (ew["cells"] if ew is not None
                        else torch.zeros(B, 3, 3, dtype=dtype, device=R.device))
                op = _MutualField(i_, k_, N, polar, ew, self.polar_eps, self.max_iter)
                mu = _ImplicitInduction.apply(field_d * polar[:, None].to(dtype), alpha, T, pos, cell, op)
                mu = mu * polar[:, None].to(dtype)
        e_pol = -0.5 * (mu * field_p).sum(1) * COULOMB
        return mu, e_pol

    def _use_dense(self, N: int, ew) -> bool:
        if self.solver == "dense" or self._force_dense:
            if ew is not None:
                raise ValueError("the dense induction solve has no reciprocal part; cells need "
                                 "solver='iterative' (and the polarizability tensor is for molecules)")
            return True
        if self.solver == "iterative":
            return False
        return ew is None and N <= 1000

    # Ewald sums of periodic cells
    def _ewald_setup(self, data: AtomicGraph, i_: Tensor, R: Tensor):
        """Ewald parameters of the batch, or ``None`` when no structure is a
        periodic cell (or Ewald is off)."""
        if self.ewald is False or data.cell is None:
            return None
        B = data.num_graphs
        dtype, device = R.dtype, R.device
        cells = data.cell.to(dtype)
        periodic = cell_volume(cells.detach()) > 1e-8
        n_pbc = torch.full((B,), 3, dtype=torch.long, device=device)
        if data.pbc is not None:
            pbc = data.pbc.to(device)
            n_pbc = pbc.sum(-1)
            if bool((periodic & (n_pbc == 1)).any()):
                raise ValueError("Ewald sums need two (slab) or three periodic axes")
            periodic = periodic & (n_pbc >= 2)
        if not bool(periodic.any()):
            if self.ewald is True:
                raise ValueError("ewald=True needs periodic structures with a cell")
            return None
        rc = self.ewald_cutoff if self.ewald_cutoff is not None else min(7.0, self.cutoff)
        if rc > self.cutoff + 1e-9:
            raise ValueError(f"ewald_cutoff {rc} exceeds the neighbor cutoff {self.cutoff}")
        alpha_e = self.ewald_alpha if self.ewald_alpha is not None else ewald_coefficient(rc)
        alpha_p = self.pewald_alpha if self.pewald_alpha is not None else alpha_e
        rc_d = self.dispersion_cutoff if self.dispersion_cutoff is not None else rc
        if rc_d > self.cutoff + 1e-9:
            raise ValueError(f"dispersion_cutoff {rc_d} exceeds the neighbor cutoff {self.cutoff}")
        alpha_d = self.dewald_alpha if self.dewald_alpha is not None else ewald_coefficient(rc_d)
        lengths = torch.linalg.norm(cells.detach(), dim=-1)
        structures = []
        for b in range(B):
            if not bool(periodic[b]):
                continue
            idx = (data.batch == b).nonzero()[:, 0]
            grid = list(self.pme_grid) if self.pme_grid is not None else pme_grid_size(lengths[b].tolist())
            dgrid = (list(self.dpme_grid) if self.dpme_grid is not None
                     else pme_grid_size(lengths[b].tolist(), density=0.8))
            normal = None
            if int(n_pbc[b]) == 2:
                # a slab: the unit normal of the periodic plane, for the Yeh-Berkowitz correction
                free = int((~pbc[b]).nonzero()[0, 0])
                a1, a2 = cells[b][(free + 1) % 3], cells[b][(free + 2) % 3]
                normal = torch.linalg.cross(a1, a2)
                normal = normal / torch.linalg.norm(normal)
            structures.append((b, idx, grid, dgrid, normal))
        pair_periodic = periodic[data.batch[i_]]
        ones = torch.ones_like(R)
        pf = pair_periodic.to(dtype)
        summary = {"alpha": alpha_e, "pewald_alpha": alpha_p, "dewald_alpha": alpha_d, "cutoff": rc,
                   "dispersion_cutoff": rc_d, "grids": [(b, grid, dgrid) for b, _, grid, dgrid, _ in structures],
                   "slabs": [b for b, _, _, _, normal in structures if normal is not None],
                   "mesh_self": self.mesh_self}
        return {"periodic": periodic, "mesh_self": self.mesh_self, "pair_periodic": pair_periodic,
                "structures": structures, "cells": cells,
                "alpha_e": alpha_e * pf, "alpha_p": alpha_p * pf, "alpha_d": alpha_d * pf,
                "keep": torch.where(pair_periodic, (R < rc).to(dtype), ones),
                "keep_d": torch.where(pair_periodic, (R < rc_d).to(dtype), ones),
                "alpha_e_value": alpha_e, "alpha_p_value": alpha_p, "alpha_d_value": alpha_d,
                "order": self.pme_order, "porder": self.ppme_order or self.pme_order, "dorder": self.dpme_order,
                "summary": summary}

    def _switch(self, R: Tensor, pair_periodic: Optional[Tensor], cutoff_value: Optional[float],
                taper_value: Optional[float]) -> Optional[Tensor]:
        """The switch of a short-range term, or ``None`` when nothing is switched
        (molecules without explicit cutoffs)."""
        explicit = cutoff_value is not None or taper_value is not None or self.taper is not None
        if pair_periodic is None and not explicit:
            return None
        cut = float(cutoff_value) if cutoff_value is not None else self.cutoff
        if cut > self.cutoff + 1e-9:
            raise ValueError(f"a term cutoff of {cut} exceeds the neighbor cutoff {self.cutoff}")
        tap = (float(taper_value) if taper_value is not None
               else (self.taper if self.taper is not None else 0.9 * cut))
        sw = switching(R, tap, cut)
        if not explicit:
            sw = torch.where(pair_periodic, sw, torch.ones_like(sw))
        return sw

    def _reciprocal_energy(self, data: AtomicGraph, ew, q: Tensor, d: Tensor, Q: Tensor) -> Tensor:
        """Per-atom mesh and self energies of the permanent multipoles, ``e^2 / A``."""
        pos = data.pos.to(q.dtype)
        out = torch.zeros_like(q)
        ae = ew["alpha_e_value"]
        for b, idx, grid, _, normal in ew["structures"]:
            e = multipole_reciprocal_energy(q[idx], d[idx], Q[idx], pos[idx], ew["cells"][b], ae, grid, ew["order"])
            e = e + multipole_self_energy(q[idx], d[idx], Q[idx], torch.full_like(q[idx], ae))
            if self.mesh_self:
                e = e - mesh_self_energy(q[idx], d[idx], Q[idx],
                                         mesh_self_moments(ew["cells"][b], ae, grid, ew["order"], "coulomb"))
            if normal is not None:
                # Yeh-Berkowitz: 2 pi / V M_n^2 with M_n the dipole moment along the slab normal
                moment = q[idx] * (pos[idx] @ normal) + d[idx] @ normal
                e = e + 2.0 * math.pi / cell_volume(ew["cells"][b]) * moment * moment.sum()
            out = out.index_add(0, idx, e)
        return out

    def _reciprocal_field(self, data: AtomicGraph, ew, q: Tensor, d: Tensor, Q: Tensor) -> Tensor:
        """Mesh field of the permanent multipoles at every atom, self field included."""
        pos = data.pos.to(q.dtype)
        out = torch.zeros_like(d)
        ap = ew["alpha_p_value"]
        coef = self_field_coefficient(ap)
        for b, idx, grid, _, normal in ew["structures"]:
            _, dphi, _ = multipole_pme(q[idx], d[idx], Q[idx], pos[idx], ew["cells"][b], ap, grid,
                                       ew["porder"], hessian=False)
            field = -dphi + coef * d[idx]
            if self.mesh_self:
                s2 = mesh_self_moments(ew["cells"][b], ap, grid, ew["porder"], "coulomb", max_rank=2)[1]
                field = field + 2.0 * d[idx] @ s2
            if normal is not None:
                moment = (q[idx] * (pos[idx] @ normal) + d[idx] @ normal).sum()
                field = field - 4.0 * math.pi / cell_volume(ew["cells"][b]) * moment * normal
            out = out.index_add(0, idx, field)
        return out

    def _reciprocal_dispersion(self, data: AtomicGraph, ew, c: Tensor) -> Tensor:
        """Per-atom mesh (and self) energies of the ``-c_i c_j / R^6`` sum."""
        pos = data.pos.to(c.dtype)
        out = torch.zeros_like(c)
        ad = ew["alpha_d_value"]
        for b, idx, _, dgrid, _ in ew["structures"]:
            e = dispersion_pme(c[idx], pos[idx], ew["cells"][b], ad, dgrid, ew["dorder"])
            if self.mesh_self:
                s0 = mesh_self_moments(ew["cells"][b], ad, dgrid, ew["dorder"], "r6")[0]
                e = e + s0 * c[idx] * c[idx]
            out = out.index_add(0, idx, e)
        return out

    def _ewald_fields(self, data, ew, i_, k_, r_vec, R, scales, q, d, Q, f_damp_k, f_damp_i, D1m, D2m,
                      u_scale, N):
        """Direct- and polar-scaled permanent fields and the mutual ladder of a
        batch with cells: the real-space Ewald field of the total multipoles
        minus their bare field, plus the scaled damped field of the vacuum
        model, plus the mesh and self fields."""
        keep = ew["keep"]
        Dc = coulomb_ladder(R, 3)
        Dp = ewald_ladder(R, ew["alpha_p"], 3)
        Dew = [(a - c) * keep for a, c in zip(Dp, Dc)]
        f_ew_k = multipole_field(q[i_], d[i_], Q[i_], r_vec, Dew, at_k=True)
        f_ew_i = multipole_field(q[k_], d[k_], Q[k_], r_vec, Dew, at_k=False)
        kp = keep[:, None]
        field_d = (scatter_sum(f_ew_k + f_damp_k * (scales[:, 6:7] * kp), k_, N)
                   + scatter_sum(f_ew_i + f_damp_i * (scales[:, 6:7] * kp), i_, N))
        field_p = (scatter_sum(f_ew_k + f_damp_k * (scales[:, 2:3] * kp), k_, N)
                   + scatter_sum(f_ew_i + f_damp_i * (scales[:, 2:3] * kp), i_, N))
        rec = self._reciprocal_field(data, ew, q, d, Q)
        D1u = (Dp[1] - Dc[1] + D1m * u_scale) * keep
        D2u = (Dp[2] - Dc[2] + D2m * u_scale) * keep
        return field_d + rec, field_p + rec, D1u, D2u

    def molecular_polarizability(self, data: AtomicGraph) -> Tensor:
        """Static dipole polarizability of every structure from the induction model.

        A uniform field is applied to every polarizable site, the induced
        dipoles are solved with the mutual-induction matrix of the structure
        (Thole or density damping included) and summed, column by column, into
        the ``(B, 3, 3)`` tensor in Angstrom^3. Permanent multipoles and the
        polarization groups play no role: the induced dipoles see every other
        induced dipole as in a mutual induction run.

        Parameters
        ----------
        data : AtomicGraph
            The (batched) graph of conformations of the bound topology.

        Returns
        -------
        Tensor
            Polarizability tensors ``(B, 3, 3)``.
        """
        keep = self.keep_intermediates
        self.keep_intermediates = True
        self._force_dense = True
        try:
            with torch.no_grad():
                self.forward(data)
                A = getattr(self, "_induction_matrix", None)
                if A is None:
                    raise RuntimeError("the polarizability needs mutual polarization and polarizable atoms")
                B = A.shape[0]
                n = A.shape[1] // 3
                alpha = self.ff.params["polarizability"][self.top_type.repeat(B)]
                polar = (alpha > 0).to(A.dtype).view(B, n, 1)
                eye3 = torch.eye(3, dtype=A.dtype, device=A.device).expand(B, n, 3, 3)
                rhs = (eye3 * polar[..., None]).reshape(B, 3 * n, 3)
                mu = torch.linalg.solve(A, rhs).reshape(B, n, 3, 3)
                return mu.sum(1)
        finally:
            self.keep_intermediates = keep
            self._force_dense = False

    def _nonbonded(self, data, i_, k_, r_vec, R, scales, q, d, Q, cls_idx, typ_idx, B, N, ew) -> dict:
        """AMOEBA nonbonded terms: buffered 14-7, permanent multipoles, Thole induction."""
        P = self.ff.params
        s = self.ff.settings
        terms: dict[str, Tensor] = {}
        ci, ck = cls_idx[i_], cls_idx[k_]
        per = None if ew is None else ew["pair_periodic"]
        # van der Waals between the reduced sites
        sites = self._vdw_sites(data, B)
        vvec = r_vec + (sites[k_] - data.pos[k_].to(r_vec.dtype)) - (sites[i_] - data.pos[i_].to(r_vec.dtype))
        rv = torch.sqrt((vvec * vvec).sum(1) + TINY)
        rmin_i, rmin_k = P["vdw_rmin"][ci], P["vdw_rmin"][ck]
        eps_i, eps_k = P["vdw_eps"][ci], P["vdw_eps"][ck]
        rule = str(s["radiusrule"]).upper()
        if rule.startswith("CUBIC"):
            rmin = (rmin_i ** 3 + rmin_k ** 3) / (rmin_i ** 2 + rmin_k ** 2 + TINY)
        elif rule.startswith("GEO"):
            rmin = torch.sqrt(rmin_i * rmin_k)
        else:
            rmin = 0.5 * (rmin_i + rmin_k)
        erule = str(s["epsilonrule"]).upper()
        if erule == "HHG":
            se_i, se_k = torch.sqrt(eps_i.clamp_min(0.0)), torch.sqrt(eps_k.clamp_min(0.0))
            eps = 4.0 * eps_i * eps_k / ((se_i + se_k) ** 2 + TINY)
        elif erule.startswith("ARITH"):
            eps = 0.5 * (eps_i + eps_k)
        else:
            eps = torch.sqrt((eps_i * eps_k).clamp_min(0.0))
        pidx = self.ff.vdw_pair_index[ci, ck]
        has_pair = pidx >= 0
        if bool(has_pair.any()):
            rmin = torch.where(has_pair, P["vdw_pair_rmin"][pidx.clamp_min(0)], rmin)
            eps = torch.where(has_pair, P["vdw_pair_eps"][pidx.clamp_min(0)], eps)
        delta = float(s["delta-halgren"])
        gamma = float(s["gamma-halgren"])
        rho = rv / rmin
        rho7 = rho ** 7
        e_vdw = eps * ((1.0 + delta) / (rho + delta)) ** 7 * ((1.0 + gamma) / (rho7 + gamma) - 2.0)
        e_vdw = e_vdw * scales[:, 0]
        e_vdw_raw = e_vdw
        sw = self._switch(rv, per, self.vdw_cutoff, self.vdw_taper)
        if sw is not None:
            e_vdw = e_vdw * sw
        terms["e_vdw"] = scatter_sum(0.5 * e_vdw, i_, N) + scatter_sum(0.5 * e_vdw, k_, N)

        # permanent multipole interactions: the Coulomb ladder for molecules; for cells
        # the real-space Ewald ladder with the excluded fraction of the bare interaction
        # removed, plus the mesh and self terms
        Dc = coulomb_ladder(R, 4)
        if ew is None:
            e_mp = multipole_energy(q[i_], d[i_], Q[i_], q[k_], d[k_], Q[k_], r_vec, Dc) * scales[:, 1] * COULOMB
        else:
            keep = ew["keep"]
            De = ewald_ladder(R, ew["alpha_e"], 4)
            Dm = [(a - (1.0 - scales[:, 1]) * c) * keep for a, c in zip(De, Dc)]
            e_mp = multipole_energy(q[i_], d[i_], Q[i_], q[k_], d[k_], Q[k_], r_vec, Dm) * COULOMB
        e_node = scatter_sum(0.5 * e_mp, i_, N) + scatter_sum(0.5 * e_mp, k_, N)
        if ew is not None:
            e_node = e_node + self._reciprocal_energy(data, ew, q, d, Q) * COULOMB
        terms["e_multipole"] = e_node

        # induction with Thole damping
        mu = torch.zeros_like(d)
        if self.polarization != "none":
            alpha = P["polarizability"][typ_idx]
            thole = P["thole"][typ_idx]
            Dt = thole_ladder(R, alpha[i_], alpha[k_], thole[i_], thole[k_], 3)
            u_scale = scales[:, 7] * scales[:, 3]
            f_at_k = multipole_field(q[i_], d[i_], Q[i_], r_vec, Dt, at_k=True)
            f_at_i = multipole_field(q[k_], d[k_], Q[k_], r_vec, Dt, at_k=False)
            if ew is None:
                field_d = scatter_sum(f_at_k * scales[:, 6:7], k_, N) + scatter_sum(f_at_i * scales[:, 6:7], i_, N)
                field_p = scatter_sum(f_at_k * scales[:, 2:3], k_, N) + scatter_sum(f_at_i * scales[:, 2:3], i_, N)
                D1u, D2u = Dt[1] * u_scale, Dt[2] * u_scale
            else:
                field_d, field_p, D1u, D2u = self._ewald_fields(data, ew, i_, k_, r_vec, R, scales, q, d, Q,
                                                                f_at_k, f_at_i, Dt[1], Dt[2], u_scale, N)
            mu, e_pol = self._induced(i_, k_, r_vec, R, field_d, field_p, D1u, D2u, N, B, alpha, data, ew)
            terms["e_polar"] = e_pol
        else:
            terms["e_polar"] = torch.zeros(N, dtype=R.dtype, device=R.device)
        return {"terms": terms, "induced": mu,
                "extra": {"vdw_r": rv, "e_vdw_pair": e_vdw, "e_vdw_pair_raw": e_vdw_raw, "e_mpole_pair": e_mp}}

    # conveniences
    def export_library(self) -> AmoebaLibrary:
        """Export the current parameters as an :class:`~xnn.ffnn.models.amoebalib.AmoebaLibrary`."""
        return self.ff.export_library()

    @property
    def masses(self) -> Tensor:
        """Tensor : Per-atom masses (u) of the bound topology."""
        return self.ff.type_mass[self.top_type]

    @property
    def polarization_groups(self) -> list[int]:
        """list of int : The polarization group index of every atom."""
        return self.atom_group.tolist()

    @classmethod
    def from_config(cls, cfg) -> "Amoeba":
        """Construct the model from a core model config.

        Core field: ``cfg.cutoff``. From ``cfg.extra``: ``library`` and either
        ``topology`` (a topology JSON path), ``tinker_xyz`` (a Tinker
        coordinate file) or ``types`` + ``bonds`` inline; optional
        ``polarization``, ``trainable``, ``atomic_energies``.

        Parameters
        ----------
        cfg : xnn.common.config.schema.ModelConfig
            The core model config.

        Returns
        -------
        Amoeba
            The model.
        """
        extra = dict(cfg.extra or {})

        def listy(value):
            if isinstance(value, str) and value.lstrip().startswith(("[", "(")):
                return ast.literal_eval(value)
            return value

        library = extra.get("library", cls.default_library)
        kwargs = dict(cutoff=float(cfg.cutoff), polarization=extra.get("polarization"),
                      trainable=listy(extra.get("trainable")) or (),
                      atomic_energies=extra.get("atomic_energies"))
        for key in _OPTION_KEYS:
            if extra.get(key) is not None:
                kwargs[key] = listy(extra[key])
        if extra.get("tinker_xyz"):
            model, _ = cls.from_tinker_xyz(extra["tinker_xyz"], library, **kwargs)
            return model
        topo_path = extra.get("topology")
        if topo_path is not None:
            topology: MolecularTopology = read_topology(topo_path)
        else:
            types = listy(extra.get("types"))
            bonds = listy(extra.get("bonds"))
            if types is None or bonds is None:
                raise ValueError(f"{cls.__name__} needs a topology: set model.topology to a topology JSON "
                                 "path, model.tinker_xyz to a Tinker coordinate file, or give model.types "
                                 "and model.bonds inline")
            topology = MolecularTopology.from_bonds(types, bonds)
        return cls(library, topology, **kwargs)

    default_library = "amoeba-water-2003"


@register_model("hippo")
class Hippo(Amoeba):
    """The HIPPO force field (Rackers *et al.* 2021): AMOEBA's valence terms
    and multipoles with density-based intermolecular terms.

    The constructor and bookkeeping are those of :class:`Amoeba`; the
    nonbonded terms are the charge-penetration electrostatics (core charge
    plus damped valence multipoles), the induction damped by the same model
    densities, the multipolar Pauli repulsion, the damped dispersion and the
    pairwise charge transfer, with the charge flux of the library applied to
    the monopoles. ``forward`` reports ``e_multipole``, ``e_polar``,
    ``e_repulsion``, ``e_dispersion`` and ``e_chgtrn``.
    """

    nonbonded = "hippo"
    default_library = "hippo-water-2021"

    def _nonbonded(self, data, i_, k_, r_vec, R, scales, q, d, Q, cls_idx, typ_idx, B, N, ew) -> dict:
        P = self.ff.params
        terms: dict[str, Tensor] = {}
        ci, ck = cls_idx[i_], cls_idx[k_]
        per = None if ew is None else ew["pair_periodic"]
        core_i, core_k = P["chgpen_core"][ci], P["chgpen_core"][ck]
        zi, zk = P["chgpen_alpha"][ci], P["chgpen_alpha"][ck]
        val_i = q[i_] - core_i
        val_k = q[k_] - core_k
        zero3 = torch.zeros_like(d[i_])
        zeroQ = torch.zeros_like(Q[i_])
        zero = torch.zeros_like(R)

        # permanent electrostatics: core-core, core-valence (one-center damping), valence-valence;
        # in cells the real-space Ewald interaction of the total multipoles minus their bare
        # interaction is added and the mesh and self terms complete the sum
        Dc = coulomb_ladder(R, 4)
        # valence of i seen by core k, and of k seen by core i
        D_ki = derivative_ladder(core_valence_kernel(zi.double(), "i"), R, zi, zk, 3) + [zero]
        D_ik = derivative_ladder(core_valence_kernel(zk.double(), "j"), R, zi, zk, 3) + [zero]
        D_ov = self._overlap_ladder(R, zi, zk, 4)
        e_cc = core_i * core_k * Dc[0]
        e_ck = multipole_energy(val_i, d[i_], Q[i_], core_k, zero3, zeroQ, r_vec, D_ki)
        e_ik = multipole_energy(core_i, zero3, zeroQ, val_k, d[k_], Q[k_], r_vec, D_ik)
        e_vv = multipole_energy(val_i, d[i_], Q[i_], val_k, d[k_], Q[k_], r_vec, D_ov)
        e_damped = (e_cc + e_ck + e_ik + e_vv) * scales[:, 1]
        if ew is None:
            e_mp = e_damped * COULOMB
        else:
            keep = ew["keep"]
            De = ewald_ladder(R, ew["alpha_e"], 4)
            e_full = multipole_energy(q[i_], d[i_], Q[i_], q[k_], d[k_], Q[k_], r_vec,
                                      [(a - c) * keep for a, c in zip(De, Dc)])
            e_mp = (e_full + e_damped * keep) * COULOMB
        e_node = scatter_sum(0.5 * e_mp, i_, N) + scatter_sum(0.5 * e_mp, k_, N)
        if ew is not None:
            e_node = e_node + self._reciprocal_energy(data, ew, q, d, Q) * COULOMB
        terms["e_multipole"] = e_node

        # Pauli repulsion: K_i K_j S^2_total / R. The multipole expansion acts on the
        # squared overlap S^2 of the valence densities (its derivative ladder), the
        # 1/R stays outside, and the valence electron density carries the valence
        # electron count as its monopole with dipoles and quadrupoles of the opposite
        # sign to those of the total charge distribution
        rk_i, rk_k = P["rep_k"][ci], P["rep_k"][ck]
        ra_i, ra_k = P["rep_alpha"][ci], P["rep_alpha"][ck]
        D_rep = self._repulsion_ladder(R, ra_i, ra_k, 4)
        e_rep = rk_i * rk_k * multipole_energy(P["rep_valence"][ci], -d[i_], -Q[i_], P["rep_valence"][ck],
                                               -d[k_], -Q[k_], r_vec, D_rep) / R * scales[:, 4]
        sw = self._switch(R, per, self.repulsion_cutoff, None)
        if sw is not None:
            e_rep = e_rep * sw
        terms["e_repulsion"] = scatter_sum(0.5 * e_rep, i_, N) + scatter_sum(0.5 * e_rep, k_, N)

        # damped dispersion: -C6_i C6_k / R^6 [ (3 lambda_5 - lambda_3) / 2 ]^2; in cells the
        # Ewald real-space share replaces the bare 1/R^6 and the mesh adds the rest
        da_i, da_k = P["disp_alpha"][ci], P["disp_alpha"][ck]
        D_d = self._overlap_ladder(R, da_i, da_k, 2)
        lam3 = -D_d[1] * R ** 3
        lam5 = D_d[2] * R ** 5 / 3.0
        fdamp = 0.5 * (3.0 * lam5 - lam3)
        c6 = P["disp_c6"][ci] * P["disp_c6"][ck]
        if ew is None:
            e_disp = -c6 / R ** 6 * fdamp * fdamp * scales[:, 5]
            sw = self._switch(R, per, self.dispersion_cutoff, None)
            if sw is not None:
                e_disp = e_disp * sw
        else:
            g = dispersion_real_kernel(R, ew["alpha_d"])
            e_disp = -c6 / R ** 6 * (scales[:, 5] * fdamp * fdamp - 1.0 + g) * ew["keep_d"]
        e_dnode = scatter_sum(0.5 * e_disp, i_, N) + scatter_sum(0.5 * e_disp, k_, N)
        if ew is not None:
            e_dnode = e_dnode + self._reciprocal_dispersion(data, ew, P["disp_c6"][cls_idx])
        terms["e_dispersion"] = e_dnode

        # charge transfer: -(eps_i e^-eta_k R + eps_k e^-eta_i R); a zero eta switches a term off
        eps_i, eps_k = P["chgtrn_eps"][ci], P["chgtrn_eps"][ck]
        eta_i, eta_k = P["chgtrn_alpha"][ci], P["chgtrn_alpha"][ck]
        t1 = torch.where(eta_k > 0, eps_i * torch.exp(-eta_k * R), zero)
        t2 = torch.where(eta_i > 0, eps_k * torch.exp(-eta_i * R), zero)
        e_ct = -(t1 + t2) * scales[:, 1]
        sw = self._switch(R, per, self.chgtrn_cutoff, None)
        if sw is not None:
            e_ct = e_ct * sw
        terms["e_chgtrn"] = scatter_sum(0.5 * e_ct, i_, N) + scatter_sum(0.5 * e_ct, k_, N)

        # induction: permanent field with one-center damping (the source atom's
        # exponent), mutual induction with the two-center overlap damping
        mu = torch.zeros_like(d)
        if self.polarization != "none":
            alpha = P["polarizability"][typ_idx]
            # field at k from atom i: core (Coulomb) + valence (damped by zeta_i)
            f_core_k = multipole_field(core_i, zero3, zeroQ, r_vec, Dc, at_k=True)
            f_val_k = multipole_field(val_i, d[i_], Q[i_], r_vec, D_ki, at_k=True)
            f_core_i = multipole_field(core_k, zero3, zeroQ, r_vec, Dc, at_k=False)
            f_val_i = multipole_field(val_k, d[k_], Q[k_], r_vec, D_ik, at_k=False)
            f_at_k = f_core_k + f_val_k
            f_at_i = f_core_i + f_val_i
            D_mut = self._overlap_ladder(R, zi, zk, 2)
            u_scale = scales[:, 7] * scales[:, 3]
            if ew is None:
                field_d = scatter_sum(f_at_k * scales[:, 6:7], k_, N) + scatter_sum(f_at_i * scales[:, 6:7], i_, N)
                field_p = scatter_sum(f_at_k * scales[:, 2:3], k_, N) + scatter_sum(f_at_i * scales[:, 2:3], i_, N)
                D1u, D2u = D_mut[1] * u_scale, D_mut[2] * u_scale
            else:
                field_d, field_p, D1u, D2u = self._ewald_fields(data, ew, i_, k_, r_vec, R, scales, q, d, Q,
                                                                f_at_k, f_at_i, D_mut[1], D_mut[2], u_scale, N)
            mu, e_pol = self._induced(i_, k_, r_vec, R, field_d, field_p, D1u, D2u, N, B, alpha, data, ew)
            terms["e_polar"] = e_pol
        else:
            terms["e_polar"] = torch.zeros(N, dtype=R.dtype, device=R.device)
        return {"terms": terms, "induced": mu,
                "extra": {"e_mpole_pair": e_mp, "e_rep_pair": e_rep, "e_disp_pair": e_disp, "e_ct_pair": e_ct}}

    @staticmethod
    def _split_equal(zi: Tensor, zk: Tensor):
        eq = (zi - zk).abs() <= 1e-9 * torch.maximum(zi.abs(), zk.abs())
        return eq

    def _overlap_ladder(self, R, zi, zk, order):
        """Ladder of the two-center Coulomb kernel, handling equal and
        unequal exponents on their own subsets."""
        eq = self._split_equal(zi, zk)
        out = [torch.zeros_like(R) for _ in range(order + 1)]
        for mask, equal in ((eq, True), (~eq, False)):
            if not bool(mask.any()):
                continue
            Rm, zim, zkm = R[mask], zi[mask].double(), zk[mask].double()
            vals = derivative_ladder(overlap_coulomb_kernel(zim, zkm, equal), Rm, zim, zkm, order)
            for n in range(order + 1):
                out[n] = out[n].index_put((mask.nonzero()[:, 0],), vals[n])
        return out

    def _repulsion_ladder(self, R, zi, zk, order):
        eq = self._split_equal(zi, zk)
        out = [torch.zeros_like(R) for _ in range(order + 1)]
        for mask, equal in ((eq, True), (~eq, False)):
            if not bool(mask.any()):
                continue
            Rm, zim, zkm = R[mask], zi[mask].double(), zk[mask].double()
            S = slater_overlap(zim, zkm, equal)
            vals = derivative_ladder(S * S, Rm, zim, zkm, order)
            for n in range(order + 1):
                out[n] = out[n].index_put((mask.nonzero()[:, 0],), vals[n])
        return out


# the periodic and solver options of the constructor, also read by from_config
_OPTION_KEYS = ("ewald", "ewald_alpha", "pewald_alpha", "dewald_alpha", "ewald_cutoff", "pme_grid", "pme_order",
                "ppme_order", "dpme_grid", "dpme_order", "vdw_cutoff", "vdw_taper", "taper", "repulsion_cutoff",
                "chgtrn_cutoff", "dispersion_cutoff", "polar_eps", "mesh_self", "solver", "max_iter",
                "conserve_charge", "symmetry_mask")


def _grid_option(value) -> Optional[list]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return [int(value)] * 3
    out = [int(x) for x in value]
    return out * 3 if len(out) == 1 else out


# the iterative induced-dipole solve of large molecules and periodic cells
class _MutualField:
    """The field of a set of induced dipoles at every atom: the real-space
    pairs through their mutual tensors ``T`` (:meth:`real`) and, in cells,
    the mesh field plus the self field and, for slabs, the Yeh-Berkowitz
    field (:meth:`mesh`). The two parts depend on disjoint inputs (``T``;
    the positions and cells), which keeps the products of the implicit
    backward pass exact."""

    def __init__(self, i_: Tensor, k_: Tensor, N: int, polar: Tensor, ew, tol: float, max_iter: int):
        self.i, self.k, self.N = i_, k_, N
        self.polar = polar
        self.tol, self.max_iter = tol, max_iter
        self.structures = [] if ew is None else [(b, idx, grid, normal) for b, idx, grid, _, normal in ew["structures"]]
        self.alpha_p = 0.0 if ew is None else ew["alpha_p_value"]
        self.order = 0 if ew is None else ew["porder"]
        self.mesh_self = False if ew is None else bool(ew["mesh_self"])

    def real(self, mu: Tensor, T: Tensor) -> Tensor:
        return (scatter_sum(torch.einsum("mab,mb->ma", T, mu[self.i]), self.k, self.N)
                + scatter_sum(torch.einsum("mab,mb->ma", T, mu[self.k]), self.i, self.N))

    def mesh(self, mu: Tensor, pos: Tensor, cell: Tensor) -> Optional[Tensor]:
        if not self.structures:
            return None
        f = torch.zeros_like(mu)
        coef = self_field_coefficient(self.alpha_p)
        for b, idx, grid, normal in self.structures:
            m = mu[idx]
            _, dphi, _ = multipole_pme(torch.zeros_like(m[:, 0]), m, None, pos[idx], cell[b], self.alpha_p, grid,
                                       self.order, hessian=False)
            field = -dphi + coef * m
            if self.mesh_self:
                s2 = mesh_self_moments(cell[b], self.alpha_p, grid, self.order, "coulomb", max_rank=2)[1]
                field = field + 2.0 * m @ s2
            if normal is not None:
                field = field - 4.0 * math.pi / cell_volume(cell[b]) * (m @ normal).sum() * normal
            f = f.index_add(0, idx, field)
        return f

    def __call__(self, mu: Tensor, T: Tensor, pos: Tensor, cell: Tensor) -> Tensor:
        f = self.real(mu, T)
        g = self.mesh(mu, pos, cell)
        return f if g is None else f + g


def _pcg(rhs: Tensor, alpha: Tensor, field_of, polar: Tensor, tol: float, max_iter: int) -> Tensor:
    """Preconditioned conjugate gradients for ``(1/alpha - T) mu = rhs`` on the
    polarizable atoms, preconditioned by ``alpha``, converged when the rms
    change of the dipoles per iteration drops below ``tol`` Debye."""
    mask = polar[:, None].to(rhs.dtype)
    n_polar = max(int(polar.sum()), 1)
    inv_alpha = torch.where(polar, 1.0 / alpha.clamp_min(1e-30), torch.zeros_like(alpha))[:, None]
    diag = alpha[:, None] * mask

    def apply(v):
        return (v * inv_alpha - field_of(v)) * mask

    x = diag * rhs
    r = rhs * mask - apply(x)
    z = diag * r
    p = z
    rz = (r * z).sum()
    for _ in range(max_iter):
        ap = apply(p)
        denom = (p * ap).sum()
        if float(denom) == 0.0:
            break
        a = rz / denom
        step = a * p
        x = x + step
        r = r - a * ap
        if math.sqrt(float((step * step).sum()) / n_polar) * _DEBYE < tol:
            break
        z = diag * r
        rz_new = (r * z).sum()
        p = z + (rz_new / rz) * p
        rz = rz_new
    else:
        warnings.warn(f"the induced dipoles did not converge to {tol} Debye in {max_iter} iterations",
                      stacklevel=2)
    return x


class _ImplicitInduction(torch.autograd.Function):
    """``mu = (1/alpha - T)^-1 rhs`` by conjugate gradients, with the gradient of
    the converged dipoles through the adjoint system (the same solve on the
    incoming gradient): neither the iterations nor a dense matrix are stored.

    The backward pass forms ``lambda^T (dA/dtheta) mu`` explicitly: the
    real-space part is the outer product of the adjoint and the dipoles on
    every pair, the mesh part the position and cell derivatives of
    ``lambda . mesh(mu)`` with both vectors held fixed. For a second
    derivative the dipoles and the adjoint are recomputed as differentiable
    solves and the bilinear mesh term is re-attached to them on detached
    copies of the positions, so every second derivative through the solve
    (force training, Hessians) is exact.
    """

    @staticmethod
    def forward(ctx, rhs, alpha, T, pos, cell, op):
        with torch.no_grad():
            mu = _pcg(rhs, alpha, lambda v: op(v, T, pos, cell), op.polar, op.tol, op.max_iter)
        ctx.op = op
        ctx.save_for_backward(mu, rhs, alpha, T, pos, cell)
        return mu

    @staticmethod
    def backward(ctx, grad_mu):
        mu, rhs, alpha, T, pos, cell = ctx.saved_tensors
        op = ctx.op
        mask = op.polar[:, None].to(mu.dtype)
        create_graph = torch.is_grad_enabled()
        if create_graph:
            mu = _ImplicitInduction.apply(rhs, alpha, T, pos, cell, op)
        lam = _ImplicitInduction.apply(grad_mu * mask, alpha, T, pos, cell, op)
        grad_T = grad_pos = grad_cell = None
        if T.requires_grad:
            # d/dT of lam . (T mu[i] at k, T mu[k] at i) summed over the pairs
            grad_T = lam[op.k][:, :, None] * mu[op.i][:, None, :] + lam[op.i][:, :, None] * mu[op.k][:, None, :]
        if op.structures and (pos.requires_grad or cell.requires_grad):
            mu_d, lam_d = mu.detach(), lam.detach()
            with torch.enable_grad():
                # the explicit dependence on the positions and the cell, with the vectors fixed
                inputs = [t for t in (pos, cell) if t.requires_grad]
                scalar = (lam_d * op.mesh(mu_d, pos, cell)).sum()
                grads = list(torch.autograd.grad(scalar, inputs, create_graph=create_graph, allow_unused=True))
                if create_graph:
                    # the term is bilinear in the adjoint and the dipoles: their (zero-valued)
                    # variations carry the graphs of the solves into the returned gradients. They
                    # are differentiated on detached copies of the positions, as a separate pass,
                    # so that this backward is not entered again through the recomputed solves.
                    pv = pos.detach().requires_grad_(bool(pos.requires_grad))
                    cv = cell.detach().requires_grad_(bool(cell.requires_grad))
                    copies = [t for t in (pv, cv) if t.requires_grad]
                    scalar = (lam_d * op.mesh(mu - mu_d, pv, cv)).sum() + ((lam - lam_d) * op.mesh(mu_d, pv, cv)).sum()
                    more = torch.autograd.grad(scalar, copies, create_graph=True, allow_unused=True)
                    grads = [g if m is None else (m if g is None else g + m) for g, m in zip(grads, more)]
            found = dict(zip([id(t) for t in inputs], grads))
            grad_pos, grad_cell = found.get(id(pos)), found.get(id(cell))
        a2 = torch.where(op.polar, alpha * alpha, torch.ones_like(alpha))
        grad_alpha = torch.where(op.polar, (lam * mu).sum(1) / a2, torch.zeros_like(alpha))
        return lam, grad_alpha, grad_T, grad_pos, grad_cell, None


# small helpers
def _angle(u: Tensor, v: Tensor) -> Tensor:
    cos_th = (u * v).sum(1) / torch.sqrt((u * u).sum(1) * (v * v).sum(1) + TINY)
    bound = min(0.9999999999, 1.0 - 4.0 * torch.finfo(cos_th.dtype).eps)
    return torch.arccos(torch.clamp(cos_th, -bound, bound))


def _key2(a: str, b: str) -> str:
    return "-".join(sorted((a, b)))


def _canon2(a: str, b: str) -> tuple[str, bool]:
    if a > b:
        return f"{b}-{a}", True
    return f"{a}-{b}", False


def _key3(a: str, b: str, c: str) -> str:
    return f"{c}-{b}-{a}" if a > c else f"{a}-{b}-{c}"


def _canon3(a: str, b: str, c: str) -> tuple[str, bool]:
    if a > c:
        return f"{c}-{b}-{a}", True
    return f"{a}-{b}-{c}", False


def _fmt5(*parts: str) -> str:
    return "-".join(parts)


def _opbend_key(lib: AmoebaLibrary, d: str, b: str, a: str, c: str) -> str:
    a, c = sorted((a, c))
    for pat in ((d, b, a, c), (d, b, a, WILDCARD), (d, b, c, WILDCARD), (d, b, WILDCARD, WILDCARD)):
        key = "-".join(pat)
        if key in lib.opbend_types:
            return key
    raise KeyError(f"no out-of-plane parameters for {d}-{b}-{a}-{c}")


__all__ = ["Amoeba", "Hippo", "AmoebaForceField", "COULOMB", "COULOMB_KCAL", "RadialSeries",
           "multipole_energy", "multipole_field", "thole_ladder", "coulomb_ladder",
           "core_valence_kernel", "overlap_coulomb_kernel", "slater_overlap", "derivative_ladder",
           "tortor_spline_tables", "bicubic_interpolate", "ewald_ladder", "multipole_pme"]
