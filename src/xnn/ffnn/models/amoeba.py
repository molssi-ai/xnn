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
from xnn.common.data.neighborlist import unique_pairs
from xnn.common.models import InteratomicPotential, register_model
from xnn.common.models.csr import narrow_keys
from xnn.common.models.fast import AutoPolicy, resolve_use_fast
from xnn.common.models.ops import cell_volume, scatter_sum, structure_sums

from .amoeba_ewald import (
    MeshPlan,
    dispersion_real_kernel,
    ewald_coefficient,
    ewald_ladder,
    fft_mesh_size,
    fusable,
    fused,
    fused_active,
    matmul3,
    matvec,
    multipole_pme,
    multipole_self_energy,
    pme_grid_size,
    region,
    self_field_coefficient,
    switching,
)
from .amoebalib import DEFAULT_SETTINGS, KCAL_TO_EV, MAX_TORSION_TERMS, WILDCARD, AmoebaLibrary, read_amoeba
from .geometry import TINY, dihedral_cos_sin, pair_image_shifts, pair_vectors
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
# the z-only frame's x axis switches from the lab x axis to the lab y axis when the z axis is within
# 45 degrees of the lab x axis: cos(45 degrees)
_ZONLY_SWITCH = math.sqrt(0.5)

# trainable parameter groups of the force field, and the model-level ones
_GROUPS = ("vdw", "bond", "angle", "strbnd", "urey", "opbend", "torsion", "pitors", "tortor",
           "charge", "dipole", "quadrupole", "polarizability", "thole", "chgpen", "chgtrn",
           "repulsion", "dispersion", "exchpol", "cflux")
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


# the series in q of R g = 1 - e^-p sum_k q^2k P_k(p) of the overlap Coulomb kernel, p = (zi + zj) R / 2 and
# q = (zi - zj) R / 2: {power of p: coefficient} of P_0 .. P_6, P_0 the equal-exponent form
_COULOMB_NEAR = (
    {0: 1.0, 1: 11 / 16, 2: 3 / 16, 3: 1 / 48},
    {-1: 3 / 8, 0: 3 / 8, 1: 5 / 32, 2: 1 / 32, 3: 1 / 480},
    {-3: -1 / 16, -2: -1 / 16, 0: 1 / 48, 1: 19 / 1920, 2: 1 / 640, 3: 1 / 13440},
    {-3: -1 / 96, -2: -1 / 96, -1: -1 / 320, 0: 1 / 2880, 1: 23 / 80640, 2: 1 / 26880, 3: 1 / 725760},
    {-3: -1 / 1920, -2: -1 / 1920, -1: -1 / 6720, 1: 1 / 215040, 2: 1 / 1935360, 3: 1 / 63866880},
    {-3: -1 / 80640, -2: -1 / 80640, -1: -1 / 322560, 0: -1 / 14515200, 1: 31 / 638668800,
     2: 1 / 212889600, 3: 1 / 8302694400},
    {-3: -1 / 5806080, -2: -1 / 5806080, -1: -1 / 26611200, 0: -1 / 958003200, 1: 1 / 2846638080,
     2: 1 / 33210777600, 3: 1 / 1494484992000})
_NEAR_TERMS = len(_COULOMB_NEAR) - 1
# the series of c1(t) = (t cosh t - sinh t) / t^3 and c2(t) = sinh t / t of the Slater overlap
_C1 = tuple((2 * k + 2) / math.factorial(2 * k + 3) for k in range(_NEAR_TERMS + 1))
_C2 = tuple(1.0 / math.factorial(2 * k + 1) for k in range(_NEAR_TERMS + 1))


def _near_weights(table: dict) -> tuple:
    """``(powers, W)``: the consecutive R powers of a near-equal series and
    ``W[power, k]``, the coefficient of ``rho^2k`` in the coefficient of that power."""
    powers = list(range(min(table), max(table) + 1))
    return powers, [[table.get(p, {}).get(k, 0.0) for k in range(_NEAR_TERMS + 1)] for p in powers]


# R g of the Coulomb kernel: the term rho^2k P_k,m p^(2k+m) sits at the power 2k + m of p
_COULOMB_POWERS, _COULOMB_W = _near_weights(
    {pw: {k: c for k, row in enumerate(_COULOMB_NEAR) for m, c in row.items() if 2 * k + m == pw}
     for pw in sorted({2 * k + m for k, row in enumerate(_COULOMB_NEAR) for m in row})})
# the Slater overlap: rho^2k [c1_k s^(2k+2) + c2_k (s^(2k+1) + s^2k)]
_SLATER_POWERS, _SLATER_W = _near_weights(
    {pw: {k: (_C1[k] if pw == 2 * k + 2 else 0.0) + (_C2[k] if pw in (2 * k, 2 * k + 1) else 0.0)
          for k in range(_NEAR_TERMS + 1) if pw in (2 * k, 2 * k + 1, 2 * k + 2)}
     for pw in range(2 * _NEAR_TERMS + 3)})


_NEAR_CONSTANTS: dict = {}


def _near_table(name: str, like: Tensor) -> tuple:
    """``(powers, W)`` of the near-equal series ``name`` ("coulomb" or
    "slater"), ``W`` on the device and dtype of ``like``, made once."""
    powers, W = {"coulomb": (_COULOMB_POWERS, _COULOMB_W), "slater": (_SLATER_POWERS, _SLATER_W)}[name]
    key = (name, str(like.device), like.dtype)
    if key not in _NEAR_CONSTANTS:
        _NEAR_CONSTANTS[key] = torch.tensor(W, dtype=like.dtype, device=like.device)
    return powers, _NEAR_CONSTANTS[key]


def _nonneg_powers(x: Tensor, n: int) -> Tensor:
    """``x^0 .. x^n`` along a new last axis: ``x^0 .. x^3`` times ``x^0, x^4,
    x^8 ...`` as one outer product, a handful of elementwise products in all
    (a cumulative product over a short axis runs as a scan, with a scan for its
    backward, each many times slower than the products themselves)."""
    small = [torch.ones_like(x)]
    while len(small) < min(n, 3) + 1:
        small.append(small[-1] * x)
    a = torch.stack(small, -1)
    if n < 4:
        return a
    x4 = small[3] * x
    big = [torch.ones_like(x), x4]
    while len(big) < n // 4 + 1:
        big.append(big[-1] * x4)
    b = torch.stack(big, -1)
    return (b[..., :, None] * a[..., None, :]).reshape(*x.shape, b.shape[-1] * a.shape[-1])[..., :n + 1]


def _powers(x: Tensor, lo: int, hi: int) -> Tensor:
    """``x^lo .. x^hi`` (``lo <= 0 <= hi``) along a new last axis for a positive ``x``."""
    parts = []
    if lo < 0:
        parts.append(_nonneg_powers(1.0 / x, -lo).flip(-1)[..., :-1])
    parts.append(_nonneg_powers(x, hi))
    return torch.cat(parts, -1)


def _near_coefficients(rho: Tensor, scale: Tensor, name: str) -> Tensor:
    """``scale^power sum_k rho^2k W[power, k]`` for every power of the
    near-equal series ``name``, ``(..., n_powers)``."""
    powers, W = _near_table(name, rho)
    rho2 = rho * rho
    rho_pow = [torch.ones_like(rho2)]
    for _ in range(_NEAR_TERMS):
        rho_pow.append(rho_pow[-1] * rho2)
    return (torch.stack(rho_pow, -1) @ W.t()) * _powers(scale, powers[0], powers[-1])


def overlap_coulomb_kernel(zi: Tensor, zj: Tensor, near: bool) -> RadialSeries:
    """Coulomb interaction kernel of two hydrogen-like densities (the
    two-center charge-penetration damping): the closed form
    ``[1 - A^2 (1 + 2 B + zeta_i R / 2) e^(-zeta_i R) - B^2 (1 + 2 A + zeta_j R / 2) e^(-zeta_j R)] / R``
    with ``A = zeta_j^2 / (zeta_j^2 - zeta_i^2)`` and
    ``B = zeta_i^2 / (zeta_i^2 - zeta_j^2)``, or with ``near`` its series
    ``[1 - e^-p sum_k q^2k P_k(p)] / R`` in ``q = (zeta_i - zeta_j) R / 2``,
    ``p = (zeta_i + zeta_j) R / 2``, which holds at equal and nearly equal
    exponents where the closed form cancels (``P_0`` is the equal-exponent
    form ``1 + 11/16 p + 3/16 p^2 + 1/48 p^3``)."""
    if near:
        half = 0.5 * (zi + zj)
        coef = _near_coefficients((zi - zj) / (zi + zj), half, "coulomb")
        terms = {(-1, 0, 0): 1.0}
        for n, pw in enumerate(_COULOMB_POWERS):
            terms[(pw - 1, 1, 1)] = -coef[..., n]
        return RadialSeries(terms)
    A = zj ** 2 / (zj ** 2 - zi ** 2)
    B = zi ** 2 / (zi ** 2 - zj ** 2)
    return RadialSeries({(-1, 0, 0): 1.0,
                         (-1, 2, 0): -A * A * (1.0 + 2.0 * B), (0, 2, 0): -A * A * 0.5 * zi,
                         (-1, 0, 2): -B * B * (1.0 + 2.0 * A), (0, 0, 2): -B * B * 0.5 * zj})


def slater_overlap(zi: Tensor, zj: Tensor, near: bool) -> RadialSeries:
    """Overlap of two normalised 1s Slater functions with exponents
    ``zeta_i / 2`` and ``zeta_j / 2`` (the HIPPO pseudo-orbitals): the closed
    form, or with ``near`` the identity
    ``S = (1 - rho^2)^(3/2) e^-s [s^2 c1(t) + (s + 1) c2(t)]`` with
    ``s = (zeta_i + zeta_j) R / 4``, ``t = (zeta_i - zeta_j) R / 4``,
    ``rho = t / s``, ``c2 = sinh t / t`` and
    ``c1 = (t cosh t - sinh t) / t^3`` as series in ``t``, which holds at
    equal and nearly equal exponents (``e^-s (1 + s + s^2 / 3)`` at equal
    ones)."""
    if near:
        rho = (zi - zj) / (zi + zj)
        coef = _near_coefficients(rho, 0.25 * (zi + zj), "slater")
        coef = coef * (1.0 - rho * rho).clamp_min(0.0)[..., None] ** 1.5
        return RadialSeries({(pw, 0.5, 0.5): coef[..., n] for n, pw in enumerate(_SLATER_POWERS)})
    delta = zi ** 2 - zj ** 2
    pref = 8.0 * (zi * zj) ** 1.5 / delta ** 3
    return RadialSeries({(0, 0, 1): pref * zi * delta, (-1, 0, 1): -8.0 * pref * zi * zj,
                         (0, 1, 0): pref * zj * delta, (-1, 1, 0): 8.0 * pref * zi * zj})


class _LadderPlan:
    """The derivative ladder of an exponential-polynomial series, derived once.

    Differentiating ``R^p e^{-(n zeta_i + m zeta_j) R / 2}`` with ``1/R d/dR``
    only multiplies by ``p`` and by ``-(n zeta_i + m zeta_j) / 2``, so every
    coefficient of ``D_k`` is a sum of the kernel's own coefficients ``c_t``
    times monomials ``zeta_i^a zeta_j^b`` with fixed weights. The plan derives
    those weights from the term keys of a series once; a ladder is then a few
    gathers, one matrix product and one contraction per call.

    Parameters
    ----------
    keys : tuple
        The ``(p, n, m)`` keys of the series, in the order of its coefficients.
    order : int
        Highest ladder member.
    """

    def __init__(self, keys: tuple, order: int):
        current: dict = {}
        for t, key in enumerate(keys):
            current.setdefault(key, {})[(t, 0, 0)] = 1.0
        levels = [current]
        for _ in range(order):
            nxt: dict = {}
            for (p, n, m), poly in current.items():
                if p != 0:
                    target = nxt.setdefault((p - 2, n, m), {})
                    for mono, w in poly.items():
                        target[mono] = target.get(mono, 0.0) + w * p
                if n or m:
                    target = nxt.setdefault((p - 1, n, m), {})
                    for (t, a, b), w in poly.items():
                        if n:
                            target[(t, a + 1, b)] = target.get((t, a + 1, b), 0.0) - 0.5 * n * w
                        if m:
                            target[(t, a, b + 1)] = target.get((t, a, b + 1), 0.0) - 0.5 * m * w
            levels.append(nxt)
            current = nxt
        features = sorted({mono for level in levels for poly in level.values() for mono in poly})
        basis = sorted({key for level in levels for key in level})
        self.order = int(order)
        self.powers = sorted({p for p, _, _ in basis})
        self.exps = sorted({(n, m) for _, n, m in basis})
        self.feature_t = [f[0] for f in features]
        self.feature_a = [f[1] for f in features]
        self.feature_b = [f[2] for f in features]
        self.basis_p = [self.powers.index(p) for p, _, _ in basis]
        self.basis_e = [self.exps.index((n, m)) for _, n, m in basis]
        f_index = {f: j for j, f in enumerate(features)}
        b_index = {key: j for j, key in enumerate(basis)}
        n_out = len(basis) * (self.order + 1)
        weights = [[0.0] * n_out for _ in features]
        for o, level in enumerate(levels):
            for key, poly in level.items():
                for mono, w in poly.items():
                    weights[f_index[mono]][b_index[key] * (self.order + 1) + o] += w
        self.weights = weights
        self.n_basis = len(basis)
        self._on: dict = {}

    def _tensors(self, device):
        key = str(device)
        if key not in self._on:
            def ints(v):
                return torch.tensor(v, dtype=torch.long, device=device)
            f64 = torch.float64
            self._on[key] = (ints(self.feature_t), ints(self.feature_a), ints(self.feature_b),
                             ints(self.basis_p), ints(self.basis_e),
                             torch.tensor([n for n, _ in self.exps], dtype=f64, device=device),
                             torch.tensor([m for _, m in self.exps], dtype=f64, device=device),
                             torch.tensor(self.weights, dtype=f64, device=device))
        return self._on[key]

    def evaluate(self, coeffs: Tensor, R: Tensor, zi: Tensor, zj: Tensor) -> Tensor:
        """The ladder ``(M, order + 1)`` for float64 coefficients ``(M, T)`` and pairs."""
        ft, fa, fb, bp, be, en, em, W = self._tensors(R.device)
        zi_pow = torch.stack([zi ** a for a in range(max(self.feature_a) + 1)], 1)
        zj_pow = torch.stack([zj ** b for b in range(max(self.feature_b) + 1)], 1)
        features = coeffs[:, ft] * zi_pow[:, fa] * zj_pow[:, fb]                   # (M, F)
        r_pow = torch.stack([R ** p for p in self.powers], 1)
        decay = torch.exp(-0.5 * R[:, None] * (en * zi[:, None] + em * zj[:, None]))
        basis = r_pow[:, bp] * decay[:, be]                                         # (M, nB)
        coef = (features @ W).view(R.shape[0], self.n_basis, self.order + 1)
        return (coef * basis[:, :, None]).sum(1)


_LADDER_PLANS: dict = {}


def _ladder_plan(keys: tuple, order: int) -> _LadderPlan:
    if (keys, order) not in _LADDER_PLANS:
        _LADDER_PLANS[(keys, order)] = _LadderPlan(keys, order)
    return _LADDER_PLANS[(keys, order)]


def _coulomb64(R64: Tensor, order: int) -> list[Tensor]:
    """The ladder of ``1/R`` in float64, ``(-1)^n (2n - 1)!! / R^(2n + 1)``."""
    inv = 1.0 / R64
    inv2 = inv * inv
    out, c = [], inv
    for n in range(order + 1):
        out.append(c)
        c = c * (-(2 * n + 1)) * inv2
    return out


def _finish_ladder(values64: list, R64: Tensor, dtype: torch.dtype, deficit: bool) -> list[Tensor]:
    """The float64 ladder in the working precision, or its deficit ``C_n - D_n`` from the
    Coulomb ladder of ``1/R``: the damping's exponential parts alone, formed before the cast so
    that no core-sized term is cancelled in the working precision."""
    if deficit:
        values64 = [c - v for c, v in zip(_coulomb64(R64, len(values64) - 1), values64)]
    return [v.to(dtype) for v in values64]


@region("hippo.ladder")
def derivative_ladder(kernel: RadialSeries, R: Tensor, zi: Tensor, zj: Tensor, order: int, deficit: bool = False
                      ) -> list[Tensor]:
    """``[D_0, ..., D_order]`` with ``D_n = (1/R d/dR)^n g`` evaluated per pair, or with
    ``deficit`` the deficits ``C_n - D_n`` from the Coulomb ladder.

    The ladder of a series with plain coefficients comes from its
    :class:`_LadderPlan`, derived once per term layout; a series that is
    itself a symbolic ladder is differentiated term by term.
    """
    dtype = R.dtype
    # the series of unequal exponents cancel strongly: evaluate in float64 and
    # return in the working precision
    R64, zi64, zj64 = R.double(), zi.double(), zj.double()
    coeffs = list(kernel.terms.values())
    if any(isinstance(c, tuple) for c in coeffs):
        out = []
        series = kernel
        for n in range(order + 1):
            out.append(series.evaluate(R64, zi64, zj64))
            if n < order:
                series = series.ladder()
        return _finish_ladder(out, R64, dtype, deficit)
    parts = [c.double() if isinstance(c, Tensor) else R64.new_tensor(float(c)) for c in coeffs]
    shape = torch.broadcast_shapes(R64.shape, zi64.shape, zj64.shape, *[c.shape for c in parts])

    def flat(t: Tensor) -> Tensor:
        return t.expand(shape).reshape(-1)

    stacked = torch.stack([flat(c) for c in parts], 1)
    values = _ladder_plan(tuple(kernel.terms), order).evaluate(stacked, flat(R64), flat(zi64), flat(zj64))
    return _finish_ladder([values[:, n].reshape(shape) for n in range(order + 1)], R64.expand(shape), dtype, deficit)


def _warm_hippo_ladders() -> None:
    """Derive the ladder plans of every HIPPO kernel layout (charge penetration,
    the closed forms of the two-center overlap and the squared Slater overlap)
    ahead of the first evaluation."""
    z1 = torch.ones(1, dtype=torch.float64)
    z2 = 2.0 * z1
    kernels = [(core_valence_kernel(z1, "i"), 3), (core_valence_kernel(z1, "j"), 3)]
    kernels += [(overlap_coulomb_kernel(z1, z2, False), 4), (overlap_coulomb_kernel(z1, z2, False), 2),
                (squared_overlap(z1, z2, False), 4)]
    for kernel, order in kernels:
        _ladder_plan(tuple(kernel.terms), order)


@fusable
def core_valence_ladder(R: Tensor, zi: Tensor, zk: Tensor, slot: str, order: int, deficit: bool = False
                        ) -> list[Tensor]:
    """The ladder of the one-center (core-valence) kernel of the exponent of
    ``zi`` (``slot="i"``) or ``zk`` (``slot="j"``), or its deficit, with a zero
    ``D_4`` (a core carries no quadrupole)."""
    zeta = zi if slot == "i" else zk
    ladder = derivative_ladder(core_valence_kernel(zeta.double(), slot), R, zi, zk, order, deficit)
    return ladder + [torch.zeros_like(R)]


# below this |zeta_i - zeta_k| R the two-center kernels take their series in the exponent difference, exact
# at equal exponents, and above it their closed forms, whose cancellation is then mild
_NEAR_EQUAL = 2.0


def _near_pairs(R: Tensor, zi: Tensor, zk: Tensor) -> Tensor:
    return (zi - zk).abs() * R.detach() < _NEAR_EQUAL


@fusable
def two_center_ladder(R: Tensor, zi: Tensor, zk: Tensor, order: int, kernel, deficit: bool = False) -> list[Tensor]:
    """The ladder of a two-center kernel (:func:`overlap_coulomb_kernel` or
    :func:`squared_overlap`), or its deficit, on every pair, selected between
    its series for equal and nearly equal exponents and its closed form without
    a host synchronization; each form sees a stand-in exponent on the pairs of
    the other (equal exponents for the series, distinct ones for the closed form)."""
    near = _near_pairs(R, zi, zk)
    zi64, zk64 = zi.double(), zk.double()
    zk_un = torch.where(near, 1.5 * zi64 + 1.0, zk64)
    series = _near_ladder(kernel, R, zi64, torch.where(near, zk64, zi64), order, deficit)
    closed = derivative_ladder(kernel(zi64, zk_un, False), R, zi64, zk_un, order, deficit)
    return [torch.where(near, a, b) for a, b in zip(series, closed)]


def squared_overlap(zi: Tensor, zj: Tensor, near: bool) -> RadialSeries:
    """The squared overlap ``S^2`` of two Slater densities (HIPPO repulsion)."""
    S = slater_overlap(zi, zj, near)
    return S * S


def _coulomb_near_form(zi: Tensor, zk: Tensor) -> tuple:
    coef = _near_coefficients((zi - zk) / (zi + zk), 0.5 * (zi + zk), "coulomb")
    return 1.0, -coef, _COULOMB_POWERS[0] - 1, 0.5 * (zi + zk)


def _slater_near_form(zi: Tensor, zk: Tensor) -> tuple:
    rho = (zi - zk) / (zi + zk)
    w = (1.0 - rho * rho).clamp_min(0.0)
    S = _near_coefficients(rho, 0.25 * (zi + zk), "slater") * (w * w.sqrt())[..., None]
    return 0.0, S, _SLATER_POWERS[0], 0.25 * (zi + zk)


# the near-equal series of the two-center kernels as c / R + e^(-lambda R) sum_j Q_j R^(lowest + j), and
# whether the kernel is its square: kernel -> ((zi, zk) -> (c, Q, lowest, lambda), squared)
_NEAR_FORMS = {overlap_coulomb_kernel: (_coulomb_near_form, False), squared_overlap: (_slater_near_form, True)}


def _near_ladder(kernel, R: Tensor, zi: Tensor, zk: Tensor, order: int, deficit: bool = False) -> list[Tensor]:
    """The ladder of the near-equal series of a two-center kernel, or its deficit. With a single
    exponential, ``(1/R d/dR) e^(-lambda R) Q = e^(-lambda R) (Q' - lambda Q) / R``
    is a recurrence on the coefficients of ``Q``, ``(1/R d/dR)^n (1/R)`` is
    ``(-1)(-3)..(1 - 2n) R^(-1-2n)``, and the ladder of a square follows from
    that of its root by the Leibniz rule (``1/R d/dR`` is a derivation)."""
    form, squared = _NEAR_FORMS[kernel]
    shape = torch.broadcast_shapes(R.shape, zi.shape, zk.shape)
    R64 = R.double().expand(shape)
    inverse, Q, lowest, lam = form(zi.double().expand(shape), zk.double().expand(shape))
    start = min(lowest, -1) - 2 * order
    r_pow = _powers(R64, start, lowest + Q.shape[-1] - 1)
    decay = torch.exp(-lam * R64)
    ladder, low, factor = [], lowest, 1.0
    for level in range(order + 1):
        lo = low - start
        value = decay * (Q * r_pow[..., lo:lo + Q.shape[-1]]).sum(-1)
        if inverse:
            value = value + inverse * factor * r_pow[..., -1 - 2 * level - start]
        ladder.append(value)
        if level < order:
            pw = torch.arange(low, low + Q.shape[-1], dtype=Q.dtype, device=Q.device)
            Q = (torch.nn.functional.pad(Q * pw, (0, 1))
                 - torch.nn.functional.pad(lam[..., None] * Q, (1, 0)))
            low -= 2
            factor *= -1.0 - 2.0 * level
    if squared:
        ladder = [sum(math.comb(n, k) * ladder[k] * ladder[n - k] for k in range(n + 1)) for n in range(order + 1)]
    return _finish_ladder(ladder, R64, R.dtype, deficit)


def two_center(R: Tensor, zi: Tensor, zk: Tensor, order: int, kernel, deficit: bool = False) -> list[Tensor]:
    """The ladder of a two-center kernel (:func:`overlap_coulomb_kernel` or
    :func:`squared_overlap`), or its deficit, with its series in the exponent difference where
    ``|zeta_i - zeta_k| R`` is small (equal exponents included) and its closed
    form elsewhere. Each form runs on its own pairs, which takes a host
    synchronization; on a GPU inside the fused path, a compiled region or a
    CUDA-graph capture both forms run on every pair instead
    (:func:`two_center_ladder`)."""
    if R.is_cuda and (fused_active() or torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing()):
        return two_center_ladder(R, zi, zk, order, kernel, deficit)
    near_pairs = _near_pairs(R, zi, zk)
    # the near pairs first; an empty side still runs, so its device tables exist before any graph capture
    perm = torch.argsort((~near_pairs).to(torch.int8), stable=True)
    n_near = int(near_pairs.sum())
    inverse = torch.empty_like(perm).scatter_(0, perm, torch.arange(perm.shape[0], device=perm.device))
    Rp, zip_, zkp = R.index_select(0, perm), zi.double().index_select(0, perm), zk.double().index_select(0, perm)
    near = _near_ladder(kernel, Rp[:n_near], zip_[:n_near], zkp[:n_near], order, deficit)
    zi_far, zk_far = zip_[n_near:], zkp[n_near:]
    far = derivative_ladder(kernel(zi_far, zk_far, False), Rp[n_near:], zi_far, zk_far, order, deficit)
    values = torch.stack([torch.cat([a, b]) for a, b in zip(near, far)]).index_select(1, inverse)
    return [values[n] for n in range(order + 1)]


def _real_space_fields(f_at_k, f_at_i, D1m, D2m, scales, i_, k_, N: int, r_vec, R, q_i, q_k, d_i, d_k, Q_i, Q_k,
                       keep, alpha_p):
    """The real-space permanent fields at the atoms, direct- and polar-scaled,
    and the mutual ladder of the solve, from the damped pair fields
    ``f_at_k`` / ``f_at_i`` and mutual ladder ``D1m, D2m``. Molecules
    (``keep`` ``None``): the scaled damped fields. Cells: the Ewald field of
    the total multipoles minus their bare field, plus the scaled damped field
    of the pairs kept; the mesh and self fields are added by the caller."""
    u_scale = scales[:, 7] * scales[:, 3]
    if keep is None:
        sd, sp = scales[:, 6:7], scales[:, 2:3]
        field_d = scatter_sum(f_at_k * sd, k_, N) + scatter_sum(f_at_i * sd, i_, N)
        field_p = scatter_sum(f_at_k * sp, k_, N) + scatter_sum(f_at_i * sp, i_, N)
        return field_d, field_p, D1m * u_scale, D2m * u_scale
    Dc = coulomb_ladder(R, 3)
    Dp = ewald_ladder(R, alpha_p, 3)
    Dew = [(a - c) * keep for a, c in zip(Dp, Dc)]
    f_ew_k = multipole_field(q_i, d_i, Q_i, r_vec, Dew, at_k=True)
    f_ew_i = multipole_field(q_k, d_k, Q_k, r_vec, Dew, at_k=False)
    kp = keep[:, None]
    field_d = (scatter_sum(f_ew_k + f_at_k * (scales[:, 6:7] * kp), k_, N)
               + scatter_sum(f_ew_i + f_at_i * (scales[:, 6:7] * kp), i_, N))
    field_p = (scatter_sum(f_ew_k + f_at_k * (scales[:, 2:3] * kp), k_, N)
               + scatter_sum(f_ew_i + f_at_i * (scales[:, 2:3] * kp), i_, N))
    D1u = (Dp[1] - Dc[1] + D1m * u_scale) * keep
    D2u = (Dp[2] - Dc[2] + D2m * u_scale) * keep
    return field_d, field_p, D1u, D2u


@fusable
def amoeba_pair_block(R, r_vec, scales, q_i, q_k, d_i, d_k, Q_i, Q_k, i_, k_, N: int, keep, alpha_e, alpha_p,
                      thole) -> dict:
    """The AMOEBA multipole pairs as one function of per-pair tensors: the
    permanent multipole energies (per pair and summed to the atoms) and, with
    ``thole = (alpha_i, alpha_k, thole_i, thole_k)``, the real-space permanent
    fields and mutual ladder of the induction (:func:`_real_space_fields`).
    ``keep``, ``alpha_e`` and ``alpha_p`` are the Ewald inputs of cells
    (``None`` for molecules)."""
    Dc = coulomb_ladder(R, 4)
    if keep is None:
        e_mp = multipole_energy(q_i, d_i, Q_i, q_k, d_k, Q_k, r_vec, Dc) * scales[:, 1] * COULOMB
    else:
        De = ewald_ladder(R, alpha_e, 4)
        Dm = [(a - (1.0 - scales[:, 1]) * c) * keep for a, c in zip(De, Dc)]
        e_mp = multipole_energy(q_i, d_i, Q_i, q_k, d_k, Q_k, r_vec, Dm) * COULOMB
    out = {"e_mp": e_mp, "e_mp_node": scatter_sum(0.5 * e_mp, i_, N) + scatter_sum(0.5 * e_mp, k_, N)}
    if thole is not None:
        Dt = thole_ladder(R, *thole, 3)
        f_at_k = multipole_field(q_i, d_i, Q_i, r_vec, Dt, at_k=True)
        f_at_i = multipole_field(q_k, d_k, Q_k, r_vec, Dt, at_k=False)
        out["fields"] = _real_space_fields(f_at_k, f_at_i, Dt[1], Dt[2], scales, i_, k_, N, r_vec, R,
                                           q_i, q_k, d_i, d_k, Q_i, Q_k, keep, alpha_p)
    return out


@fusable
def hippo_pair_block(R, r_vec, scales, q_i, q_k, d_i, d_k, Q_i, Q_k, i_, k_, N: int, params: dict, keep,
                     alpha_e, alpha_p, alpha_d, keep_d, switches: dict, polar: bool) -> dict:
    """The HIPPO pairs as one function of per-pair tensors: charge-penetration
    electrostatics, Pauli repulsion, damped dispersion and charge transfer
    (per pair and summed to the atoms) and, with ``polar``, the real-space
    permanent fields and mutual ladder of the induction. ``params`` holds the
    per-pair class parameters (``"name_i"`` / ``"name_k"``), ``switches`` the
    switches of the short-range terms (or ``None``); ``keep``, the Ewald
    coefficients and ``keep_d`` are the inputs of cells (``None`` for
    molecules)."""
    p = params
    core_i, core_k, zi, zk = p["core_i"], p["core_k"], p["zeta_i"], p["zeta_k"]
    val_i = q_i - core_i
    val_k = q_k - core_k
    zero3 = torch.zeros_like(d_i)
    zeroQ = torch.zeros_like(Q_i)
    zero = torch.zeros_like(R)
    out = {}

    # permanent electrostatics: the total multipoles through the point ladder (the Coulomb
    # ladder scaled, in cells plus the real-space Ewald interaction minus the bare one) minus the
    # scaled deficits of the damped kernels (core-valence with the one-center damping both ways,
    # valence-valence with the two-center overlap), so that no core-sized term is cancelled in the
    # working precision; the mesh and self terms complete the sum of a cell
    Dc = coulomb_ladder(R, 4)
    X_ki = core_valence_ladder(R, zi, zk, "i", 3, deficit=True)
    X_ik = core_valence_ladder(R, zi, zk, "j", 3, deficit=True)
    X_ov = two_center(R, zi, zk, 4, overlap_coulomb_kernel, deficit=True)
    s1 = scales[:, 1]
    if keep is None:
        U = [s1 * c for c in Dc]
        sk = s1
    else:
        De = ewald_ladder(R, alpha_e, 4)
        U = [(s1 * c + a - c) * keep for a, c in zip(De, Dc)]
        sk = s1 * keep
    e_def = (multipole_energy(val_i, d_i, Q_i, core_k, zero3, zeroQ, r_vec, X_ki)
             + multipole_energy(core_i, zero3, zeroQ, val_k, d_k, Q_k, r_vec, X_ik)
             + multipole_energy(val_i, d_i, Q_i, val_k, d_k, Q_k, r_vec, X_ov))
    e_mp = (multipole_energy(q_i, d_i, Q_i, q_k, d_k, Q_k, r_vec, U) - e_def * sk) * COULOMB
    out["e_mp"] = e_mp

    # Pauli repulsion: K_i K_j S^2_total / R. The multipole expansion acts on the
    # squared overlap S^2 of the valence densities (its derivative ladder), the
    # 1/R stays outside, and the valence electron density carries the valence
    # electron count as its monopole with dipoles and quadrupoles of the opposite
    # sign to those of the total charge distribution
    D_rep = two_center(R, p["rep_alpha_i"], p["rep_alpha_k"], 4, squared_overlap)
    e_rep = p["rep_k_i"] * p["rep_k_k"] * multipole_energy(p["rep_valence_i"], -d_i, -Q_i, p["rep_valence_k"],
                                                           -d_k, -Q_k, r_vec, D_rep) / R * scales[:, 4]
    if switches["repulsion"] is not None:
        e_rep = e_rep * switches["repulsion"]
    out["e_rep"] = e_rep

    # damped dispersion: -C6_i C6_k / R^6 [ (3 lambda_5 - lambda_3) / 2 ]^2; in cells the
    # Ewald real-space share replaces the bare 1/R^6 and the mesh adds the rest
    D_d = two_center(R, p["disp_alpha_i"], p["disp_alpha_k"], 2, overlap_coulomb_kernel)
    lam3 = -D_d[1] * R ** 3
    lam5 = D_d[2] * R ** 5 / 3.0
    fdamp = 0.5 * (3.0 * lam5 - lam3)
    c6 = p["disp_c6_i"] * p["disp_c6_k"]
    if keep is None:
        e_disp = -c6 / R ** 6 * fdamp * fdamp * scales[:, 5]
        if switches["dispersion"] is not None:
            e_disp = e_disp * switches["dispersion"]
    else:
        g = dispersion_real_kernel(R, alpha_d)
        e_disp = -c6 / R ** 6 * (scales[:, 5] * fdamp * fdamp - 1.0 + g) * keep_d
    out["e_disp"] = e_disp

    # charge transfer: -(eps_i e^-eta_k R + eps_k e^-eta_i R); a zero eta switches a term off
    eta_i, eta_k = p["chgtrn_alpha_i"], p["chgtrn_alpha_k"]
    t1 = torch.where(eta_k > 0, p["chgtrn_eps_i"] * torch.exp(-eta_k * R), zero)
    t2 = torch.where(eta_i > 0, p["chgtrn_eps_k"] * torch.exp(-eta_i * R), zero)
    e_ct = -(t1 + t2) * scales[:, 1]
    if switches["chgtrn"] is not None:
        e_ct = e_ct * switches["chgtrn"]
    out["e_ct"] = e_ct
    for name in ("e_mp", "e_rep", "e_disp", "e_ct"):
        out[name + "_node"] = scatter_sum(0.5 * out[name], i_, N) + scatter_sum(0.5 * out[name], k_, N)

    # induction: permanent field with one-center damping (the source atom's
    # exponent), mutual induction with the two-center overlap damping
    if polar:
        # field at k from atom i: the total multipoles through the Coulomb ladder minus the
        # valence's deficit (damped by zeta_i)
        f_at_k = (multipole_field(q_i, d_i, Q_i, r_vec, Dc, at_k=True)
                  - multipole_field(val_i, d_i, Q_i, r_vec, X_ki, at_k=True))
        f_at_i = (multipole_field(q_k, d_k, Q_k, r_vec, Dc, at_k=False)
                  - multipole_field(val_k, d_k, Q_k, r_vec, X_ik, at_k=False))
        D_mut = two_center(R, zi, zk, 2, overlap_coulomb_kernel)
        out["fields"] = _real_space_fields(f_at_k, f_at_i, D_mut[1], D_mut[2], scales, i_, k_, N, r_vec, R,
                                           q_i, q_k, d_i, d_k, Q_i, Q_k, keep, alpha_p)
    return out


@fusable
def buffered_14_7(rv: Tensor, rmin: Tensor, eps: Tensor, delta: float, gamma: float) -> Tensor:
    """Halgren's buffered 14-7 energy ``eps ((1 + delta) / (rho + delta))^7
    ((1 + gamma) / (rho^7 + gamma) - 2)`` with ``rho = rv / rmin``."""
    rho = rv / rmin
    rho7 = rho ** 7
    return eps * ((1.0 + delta) / (rho + delta)) ** 7 * ((1.0 + gamma) / (rho7 + gamma) - 2.0)


@fusable
def coulomb_ladder(R: Tensor, order: int) -> list[Tensor]:
    """The undamped ladder ``1/R, -1/R^3, 3/R^5, -15/R^7, 105/R^9``."""
    inv = 1.0 / R
    inv2 = inv * inv
    out = [inv, -inv * inv2, 3.0 * inv * inv2 ** 2, -15.0 * inv * inv2 ** 3, 105.0 * inv * inv2 ** 4]
    return out[:order + 1]


@fusable
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


@fusable
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
    Qir = matvec(Qi, r)
    Qkr = matvec(Qk, r)
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


@fusable
def multipole_field(c, d, Q, r, D, at_k: bool) -> Tensor:
    """Electric field of a multipole set through a radial kernel.

    With ``r = r_k - r_i``: the field at ``k`` due to the set at ``i``
    (``at_k=True``) is ``-c D_1 r + D_2 r (d.r) + D_1 d - D_3 r (r.Q.r) - 2 D_2 Q r``;
    the field at ``i`` due to the set at ``k`` (``at_k=False``) is
    ``c D_1 r + D_2 r (d.r) + D_1 d + D_3 r (r.Q.r) + 2 D_2 Q r``.
    """
    D1, D2, D3 = D[1], D[2], D[3]
    dr = (d * r).sum(-1, keepdim=True)
    Qr = matvec(Q, r)
    qrr = (Qr * r).sum(-1, keepdim=True)
    sign = -1.0 if at_k else 1.0
    return (sign * c[:, None] * D1[:, None] * r + D2[:, None] * r * dr + D1[:, None] * d
            + sign * D3[:, None] * r * qrr + sign * 2.0 * D2[:, None] * Qr)


def exchange_polarization_tensor(r_vec: Tensor, R: Tensor, i_: Tensor, k_: Tensor, N: int, spring_i: Tensor,
                                 spring_k: Tensor, size_i: Tensor, size_k: Tensor, damp_i: Tensor, damp_k: Tensor,
                                 weight: Tensor) -> Tensor:
    """The exchange-polarization tensors of the atoms, ``(N, 3, 3)``.

    Chung, Wang, Rackers and Ponder, *J. Phys. Chem. B* **126**, 7579 (2022):
    the overlap of two hydrogen-like orbitals,
    ``S = sqrt(q_i q_k) (1 + x + x^2 / 3) e^{-x}`` with ``x = zeta R / 2`` and
    ``zeta = sqrt(zeta_i zeta_k)`` (their eqs 8c, 8d, 8g), damps the
    polarizability of each atom of a pair along the pair:
    ``K_i = k_i sum_j S_ij^2 r_ij r_ij^T / R_ij^2`` and the polarizability
    becomes ``alpha_i (I + K_i)^-1`` (eqs 7d, 9, 11a). ``weight`` scales
    every pair (zero drops it).
    """
    x = 0.5 * torch.sqrt(damp_i * damp_k) * R
    s2 = size_i * size_k * (1.0 + x + x * x / 3.0).square() * torch.exp(-2.0 * x) * weight
    u = r_vec / R[:, None]
    outer = (u[:, :, None] * u[:, None, :]).reshape(-1, 9)
    K = scatter_sum((spring_i * s2)[:, None] * outer, i_, N) + scatter_sum((spring_k * s2)[:, None] * outer, k_, N)
    return K.view(N, 3, 3)


def _kernels():
    """The fused pair kernels (imported on first use: they need Triton)."""
    from . import amoeba_kernels
    return amoeba_kernels


def _kernels_supported(device: torch.device, dtype: torch.dtype) -> bool:
    """Whether the fused pair kernels run on ``device`` in ``dtype``."""
    if device.type != "cuda" or dtype not in (torch.float32, torch.float64):
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def _take(t: Tensor, idx: Tensor) -> Tensor:
    """Rows ``t[idx]`` by ``index_select``: its backward is an atomic ``index_add``,
    where the advanced-index gather sorts the indices and sums every atom's many
    pairs one after the other, far below the memory bandwidth."""
    return t.index_select(0, idx)


def _plan_vectors(data: AtomicGraph, i: Tensor, k: Tensor, shift: Optional[Tensor], count: Optional[Tensor] = None,
                  kernels: bool = False, rows: Optional[tuple] = None) -> Tensor:
    """``pos[k] - pos[i]`` plus the image shift of the plan's pairs. The gathers
    are :func:`_take`, whose backward adds atomically: the advanced indexing of
    :func:`pair_vectors` sorts the indices of every pair in its backward. With
    ``kernels`` the difference is :class:`~.amoeba_kernels.PairDifference`,
    whose backward kernel adds each block's gradients to the first atom of the
    block in one operation (the pairs of a plan are sorted by that atom) and
    skips the padding by ``count``; with one cell its kernel also adds the
    image shifts' product with the cell and stores the model's precision, so
    the vectors cost one launch; given the plan's compressed ``rows``, its
    backward sums each atom's row instead (no atomic adds). Otherwise a single
    cell enters as one product rather than a copy per pair."""
    pos = data.pos
    if kernels and (shift is None or (data.cell.shape[0] == 1 and not data.cell.requires_grad)):
        cell = None if shift is None else data.cell[0].to(pos.dtype)
        return _kernels().PairDifference.apply(pos, i, k, count, shift, cell, data.model_dtype, rows)
    vec = _kernels().PairDifference.apply(pos, i, k, count) if kernels else _take(pos, k) - _take(pos, i)
    if shift is not None:
        cell = data.cell.to(vec.dtype)
        if cell.shape[0] == 1:
            vec = vec + shift @ cell[0]
        else:
            vec = vec + (shift[:, :, None] * _take(cell, _take(data.batch, i))).sum(1)
    return vec.to(data.model_dtype)


# the phases of the induced-dipole solves a step plan runs at fixed length, in the order of ``StepPlan.needed``
SOLVE_PHASES = ("forward", "adjoint")


class StepPlan:
    """What an evaluation of a structure decides on the host, recorded once and
    reused while the structure moves by small steps (molecular dynamics between
    rebuilds of the pair list): the image shifts of every bonded vector, the
    unique pairs of the neighbor list with their image shifts and scale
    factors, the layout of the Ewald sums and the polarizable atoms. An
    evaluation on a plan reads these instead of deciding them, so it waits for
    the device only in the convergence tests of the induced-dipole solve.

    The pairs form a Verlet list: built with a neighbor list of ``cutoff +
    skin``, they hold every pair within the cutoff until an atom has moved
    by more than half the skin (:meth:`outdated`); the pairs beyond the
    cutoff and the padding up to a fixed capacity get zero weight.

    ``solve_iterations``, a dict from ``"forward"`` and ``"adjoint"`` to a
    number of iterations, makes those solves run exactly that long without
    reading the device (the counts of earlier steps are in ``solver_stats``);
    ``converged`` then tells on the device whether every solve of the last
    evaluation and its backward pass converged, and ``needed`` holds, per
    phase in the order of ``SOLVE_PHASES``, the iterations the last solve of
    that phase needed (zero for a phase that did not run). Build a plan with
    :meth:`Amoeba.plan_step`.
    """

    def __init__(self, skin: float = 0.0, capacity: Optional[int] = None):
        self.recording = True
        self.skin = float(skin)
        self.capacity = capacity
        self.shifts: list = []
        # the atom pairs of the bonded vectors (constants of the topology) and how many of them the
        # valence block takes, which it then forms in one gather
        self.image_pairs: list = []
        self.valence_calls = 0
        self.pairs: Optional[tuple] = None
        self.valid: Optional[Tensor] = None
        # the number of valid pairs as an int32 tensor on the device, read by the pair kernels
        self.count: Optional[Tensor] = None
        # per evaluation on the pair kernels: the pairs within the cutoff they visit and their number
        # (select_pairs), and the per-atom parameter table, a constant of the plan
        self.select: Optional[tuple] = None
        self.kernel_params: Optional[Tensor] = None
        # the pairs in compressed rows per atom, for the induction's field kernel (on a GPU)
        self.rows: Optional[tuple] = None
        self.scales: Optional[Tensor] = None
        self.reference: Optional[Tensor] = None
        self.ewald = None
        self.has_polar: Optional[bool] = None
        self.n_polar: Optional[int] = None
        self.solve_iterations: Optional[dict] = None
        self.converged: Optional[Tensor] = None
        self.needed: Optional[Tensor] = None

    def replaying(self) -> bool:
        """Whether the plan is complete and evaluations read it."""
        return not self.recording

    def outdated(self, pos: Tensor) -> Tensor:
        """Whether an atom at ``pos`` has moved by more than half the skin since
        the plan was built, as a boolean on the device (the pair list then
        misses pairs and the plan must be built again).

        Parameters
        ----------
        pos : Tensor
            Positions ``(N, 3)``.

        Returns
        -------
        Tensor
            A boolean scalar.
        """
        step = pos.detach().to(self.reference.dtype) - self.reference
        return (step * step).sum(-1).max() > (0.5 * self.skin) ** 2

    def due(self, pos: Tensor, last: Tensor, steps: int = 1) -> Tensor:
        """Whether an atom at ``pos`` will have moved by more than half the skin
        after ``steps`` more steps like the one from ``last``, with one step to
        spare, as a boolean on the device: the pair list is still valid now and
        should be built again within ``steps`` steps.

        Parameters
        ----------
        pos : Tensor
            Positions ``(N, 3)``.
        last : Tensor
            The positions of the step before ``(N, 3)``.
        steps : int, optional
            The steps the pair list must still hold, by default 1 (it is built
            again before the next step).

        Returns
        -------
        Tensor
            A boolean scalar.
        """
        pos = pos.detach().to(self.reference.dtype)
        moved = ((pos - self.reference) ** 2).sum(-1).sqrt()
        stride = ((pos - last.to(pos.dtype)) ** 2).sum(-1).sqrt()
        return (moved + (steps + 1) * stride).max() > 0.5 * self.skin

    def checks(self, pos: Tensor, last: Tensor, steps: int = 1) -> Tensor:
        """:meth:`outdated` and :meth:`due` of ``pos`` with the solves' convergence and needed
        iterations, as one float64 tensor ``[outdated, due, converged, *needed]``, and ``last`` set to
        ``pos`` after: one kernel on a GPU, so a recorded CUDA graph ends a step with one launch
        and its caller reads everything at once.

        Parameters
        ----------
        pos : Tensor
            Positions ``(N, 3)``.
        last : Tensor
            The positions of the step before ``(N, 3)``, overwritten with ``pos``.
        steps : int, optional
            As in :meth:`due`, by default 1.

        Returns
        -------
        Tensor
            ``(3 + len(needed),)``, float64.
        """
        if _kernels_supported(pos.device, torch.float64) and last.is_contiguous():
            return _kernels().plan_checks(pos, self.reference, last, self.converged, self.needed, self.skin, steps)
        out = torch.cat([torch.stack([self.outdated(pos), self.due(pos, last, steps),
                                      self.converged.reshape(())]).to(torch.float64),
                         self.needed.reshape(-1).to(torch.float64)])
        last.copy_(pos.detach())
        return out

    def update(self, other: "StepPlan", bonded: bool = True) -> bool:
        """Copy the pairs (with their rows), image shifts and reference
        positions of ``other``, a plan of the same structure built later, into
        this plan's tensors in place, so a CUDA graph recorded on this plan
        replays the new ones.

        Parameters
        ----------
        other : StepPlan
            The new plan.
        bonded : bool, optional
            Copy the image shifts of the bonded vectors too (``other`` recorded
            a whole evaluation), by default ``True``.

        Returns
        -------
        bool
            Whether every tensor matched in shape and was copied; nothing is
            copied otherwise.
        """
        mine = [*self.pairs, self.valid, self.count, self.scales, self.reference, *(self.rows or ()),
                *(self.shifts if bonded else [])]
        theirs = [*other.pairs, other.valid, other.count, other.scales, other.reference, *(other.rows or ()),
                  *(other.shifts if bonded else [])]
        if len(mine) != len(theirs) or any(
                (a is None) != (b is None) or (a is not None and a.shape != b.shape) for a, b in zip(mine, theirs)):
            return False
        for a, b in zip(mine, theirs):
            if a is not None:
                a.copy_(b)
        return True


class _Images:
    """The displacement vectors of the bonded terms of one evaluation: with a
    recording plan the image shifts of every call are searched and recorded,
    with a complete plan they are replayed in the same order."""

    def __init__(self, data: AtomicGraph, plan: Optional[StepPlan], shifts: Optional[list] = None,
                 pairs: Optional[list] = None):
        # the recorded shifts may be given on their own (a compiled block takes the list, not the
        # plan: an object whose attributes change would be guarded by identity and recompiled)
        self.data, self.plan, self.calls = data, plan, 0
        self.shifts = shifts if shifts is not None else (None if plan is None else plan.shifts)
        # with the recorded atom pairs of the first calls, those vectors are formed in one gather and
        # served call by call: one scatter per side in the backward pass instead of one per call
        self.pairs = pairs
        self._parts: Optional[tuple] = None

    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        if self.shifts is None or self.data.cell is None:
            return pair_vectors(self.data, a, b)
        if self.plan is not None and self.plan.recording:
            shift = pair_image_shifts(self.data, a, b)
            self.plan.shifts.append(shift)
            self.plan.image_pairs.append((a.detach(), b.detach()))
        elif self.pairs and self.calls < len(self.pairs):
            if self._parts is None:
                k = len(self.pairs)
                vec = pair_vectors(self.data, torch.cat([pa for pa, _ in self.pairs]),
                                   torch.cat([pb for _, pb in self.pairs]), shift=torch.cat(self.shifts[:k]))
                self._parts = vec.split([int(pa.shape[0]) for pa, _ in self.pairs])
            self.calls += 1
            return self._parts[self.calls - 1]
        else:
            shift = self.shifts[self.calls]
        self.calls += 1
        return pair_vectors(self.data, a, b, shift=shift)


def dipole_field(m: Tensor, r: Tensor, D1: Tensor, D2: Tensor) -> Tensor:
    """The field of dipoles ``m`` ``(..., M, 3)`` at the other ends of the pairs.

    It is ``T m = D_1 m + D_2 (r m) r`` (``r m`` the dot product), computed
    without forming ``T``.
    """
    return D1[:, None] * m + (D2 * (r * m).sum(-1))[..., None] * r


@fusable
def mutual_field_real(mu: Tensor, r: Tensor, D1: Tensor, D2: Tensor, ik: Tensor, ki: Tensor, N: int) -> Tensor:
    """The real-space field of dipoles ``mu`` ``(N, 3)`` through the pair ladders,
    both directions of every pair at once: gather by ``ik = [i; k]``, scatter by
    ``ki = [k; i]``."""
    f = dipole_field(mu.index_select(0, ik).view(2, -1, 3), r, D1, D2)
    return scatter_sum(f.view(-1, 3), ki, N)


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


_BCU_ON: dict = {}


def bicubic_interpolate(f, fx, fy, fxy, x, y, x1, x2, y1, y2):
    """Bicubic interpolation (the Numerical Recipes construction) inside the
    cell with corner values ``f`` ``(M, 4)`` and derivatives, corners counted
    counter-clockwise from the lower left: ``(x1,y1), (x2,y1), (x2,y2), (x1,y2)``."""
    d1 = x2 - x1
    d2 = y2 - y1
    # kept on the device: a copy from the host at every evaluation cannot be recorded in a CUDA graph
    key = (f.dtype, str(f.device))
    if key not in _BCU_ON:
        _BCU_ON[key] = torch.as_tensor(_BCU_WT, dtype=f.dtype, device=f.device)
    wt = _BCU_ON[key]
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
        ``"chgtrn"``, ``"repulsion"``, ``"dispersion"``, ``"exchpol"``,
        ``"cflux"``; ``"all"`` unfreezes every group.
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
        # the highest periodicity, on the host: reading it from the buffer waits for the device
        self.torsion_nmax = max([n for row in ns for n in row] or [1])
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
        # exchange polarization per class; a pair takes part when either of its classes is used
        xp = [lib.exchpol.get(c, {"spring": 0.0, "size": 0.0, "damp": 1.0, "use": False}) for c in self.class_names]
        make("expol_spring", [float(v["spring"]) for v in xp], "exchpol")
        make("expol_size", [float(v["size"]) for v in xp], "exchpol")
        make("expol_damp", [float(v["damp"]) for v in xp], "exchpol")
        self.register_buffer("expol_use", torch.tensor([bool(v["use"]) for v in xp], dtype=torch.bool),
                             persistent=False)
        self.has_exchpol = bool(lib.exchpol)
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
            if c in lib.exchpol:
                lib.exchpol[c] = {"spring": float(P["expol_spring"][i]), "size": float(P["expol_size"][i]),
                                  "damp": float(P["expol_damp"][i]), "use": lib.exchpol[c]["use"]}
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
        sums, by default 1.2 (0.8) points per Angstrom. As in Tinker, every
        size is rounded up to an even one with prime factors 2, 3 and 5, at
        least 16 (a requested 27 becomes 30).
    pme_order, ppme_order, dpme_order : int, optional
        B-spline orders of the multipole, induction and dispersion meshes
        (5, the multipole one, 4).
    vdw_cutoff, repulsion_cutoff, chgtrn_cutoff, dispersion_cutoff : float, optional
        Cutoffs of the switched short-range terms, by default ``cutoff``
        (the dispersion one is the Ewald real-space cutoff when its sum is
        on). The van der Waals term acts between the reduced sites, so a
        pair of sites within ``vdw_cutoff`` can belong to atoms just beyond
        it: with ``cutoff`` above ``vdw_cutoff`` by twice the largest site
        displacement (about 0.2 Angstrom for hydrogens reduced to 0.9 of
        their bond), the neighbor list holds every such pair, as Tinker's
        list of the sites does.
    vdw_taper, taper : float, optional
        Start of the C2 switch of the van der Waals term, and of every
        switched term, in Angstrom; by default 0.9 of each cutoff for
        cells and no switching for molecules.
    mesh_self : bool, optional
        Remove the spurious, position-averaged mesh self-interaction of
        every multipole from the particle-mesh energy, from the field
        that polarizes it and from the dispersion sum. Off by default,
        as in Tinker. On, the mean discretisation error of the
        self-interaction leaves the energies and the stress.
    polar_eps : float, optional
        Convergence of the iterative induced-dipole solve, the rms change
        per iteration in Debye (``1e-8``). The adjoint solve of the forces
        converges the dipoles of the polar field to the same tolerance.
    preconditioner : str, optional
        Preconditioner of the iterative solve: ``"diagonal"`` (default,
        the polarizabilities), ``"block"`` (the exact inverse of every
        molecule's real-space block, polarization groups for molecules of
        more than 32 atoms), ``"neumann"`` (the diagonal plus a first-order
        correction from the pairs within ``precond_cutoff``) or
        ``"block-neumann"`` (both). The dipoles are converged to
        ``polar_eps`` either way; only the number of iterations changes.
    precond_cutoff : float, optional
        Pair distance of the Neumann correction, in Angstrom (4.5).
    solver : str, optional
        ``"auto"`` (dense for molecules up to a thousand atoms, iterative
        otherwise), ``"dense"`` or ``"iterative"``.
    max_iter : int, optional
        Iteration limit of the solve (100).
    predictor : str, optional
        Predict the starting dipoles of the iterative solve from the
        converged dipoles of the last evaluations (:class:`DipolePredictor`:
        ``"aspc"``, ``"gear"`` or ``"previous"``), for molecular dynamics
        where each structure follows the last. Used in evaluation mode only;
        by default ``None`` (the MDI engine turns ASPC on). The result agrees
        with a cold start to ``polar_eps``.
    predictor_history : int, optional
        Number of past dipole sets the predictor uses (6).
    conserve_charge : bool, optional
        Hold the total charge of every molecule at its library value while
        ``charge`` trains (a minimum-norm projection of the row charges,
        registered on the force field), by default ``True``.
    symmetry_mask : bool, optional
        Keep the dipole and quadrupole components that the library leaves
        zero at zero while they train, by default ``True``. The quadrupoles
        stay symmetric and traceless in any case, and ``export_library``
        writes the multipoles the model uses.
    mixed_precision : bool, optional
        How the fast implementation evaluates the radial ladders of the pair
        terms in a float32 model: ``False`` (default) in float64, cast
        afterwards, at the round-off of the float32 reference; ``True`` in
        float32 throughout, as Tinker9's mixed precision does, faster, with
        the energies at the float32 round-off and the forces up to one digit
        off. No effect on a float64 model.
    use_fast : bool or str, optional
        The fast path: ``False`` runs the reference implementation, the
        eager composition of the equations (any device, second derivatives);
        ``True`` runs the fast implementation, which computes the same
        numbers to round-off with the same derivatives: the fused GPU
        kernels of the pair terms, the fields and the induction's products
        (:mod:`.amoeba_kernels`) and the mesh spreading and interpolation
        kernels. The pair kernels are differentiable twice (their generated
        tangents and parameter adjoints, :mod:`.amoeba_tangents`), so force
        losses and parameter gradients train through them; in training the
        mesh and the solver's products run eager. A second derivative of a
        periodic cell in evaluation mode (a Hessian) needs
        ``use_fast=False``: the mesh kernels are differentiable once. When
        forces follow, AMOEBA's fast implementation solves the dipoles of
        the direct and the polar field together, one set of iterations for
        both (Tinker's scheme), so its dipoles and forces agree with the
        reference's to the solver tolerance. ``"auto"`` (default) takes the
        fast implementation on a GPU above the measured crossover of the mode
        (evaluation or training) and the reference otherwise.
    mixed_precision : bool, optional
        How the fast implementation of HIPPO evaluates the radial ladders of
        the pair terms in a float32 model: ``False`` (default) in float64,
        cast afterwards, which keeps the float32 numbers at the round-off of
        the float32 reference; ``True`` in float32 throughout, as Tinker9's
        mixed precision does, which is faster, with the energies at the
        float32 round-off and the forces up to one digit off. No effect on a
        float64 model or on AMOEBA.

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
                 preconditioner: str = "diagonal",
                 precond_cutoff: float = 4.5,
                 mesh_self: bool = False,
                 solver: str = "auto",
                 max_iter: int = 100,
                 predictor: Optional[str] = None,
                 predictor_history: int = 6,
                 conserve_charge: bool = True,
                 symmetry_mask: bool = True,
                 use_fast: Union[bool, str] = "auto",
                 mixed_precision: bool = False):
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
        if preconditioner not in _Preconditioner.KINDS:
            raise ValueError(f"preconditioner must be one of {_Preconditioner.KINDS} (got {preconditioner!r})")
        self.preconditioner = preconditioner
        self.precond_cutoff = float(precond_cutoff)
        self.mesh_self = bool(mesh_self)
        if solver not in ("auto", "dense", "iterative"):
            raise ValueError("solver must be 'auto', 'dense' or 'iterative'")
        self.solver = solver
        self.max_iter = int(max_iter)
        self.set_use_fast(use_fast)
        self.mixed_precision = bool(mixed_precision)
        self.predictor: Optional[DipolePredictor] = None
        self.set_dipole_predictor(predictor, predictor_history)
        # a one-shot starting guess for the next iterative solve, and the last adjoint of each
        # shape, which starts the next backward solve
        self.dipole_guess: Optional[Tensor] = None
        self._adjoints: dict = {}
        self.solver_stats: dict = {}
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
        if self.nonbonded == "hippo":
            _warm_hippo_ladders()

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
        self._inplane_any = any(a_inplane)

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
        self._tortor_maps = set(tt_type)
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
        # a z-only frame leaves the azimuth of its x axis to the lab axes, a frame-less atom its whole
        # orientation: multipole components that are not symmetric about the free axis (a dipole off
        # the z axis, Qxx != Qyy, off-diagonal quadrupoles) make the energy depend on the orientation
        # of the system, and Tinker's analytical forces on such atoms are not the gradient of its energy
        with torch.no_grad():
            _, d_rows, Q_rows = ff.effective_multipoles()
        loose: list[str] = []
        for i, (ftype, row) in enumerate(zip(frame_type, mp_row)):
            if row < 0 or ftype not in (FRAME_ZONLY, FRAME_NONE) or top.types[i] in loose:
                continue
            d, Q = d_rows[row], Q_rows[row]
            if ftype == FRAME_ZONLY:
                off = torch.stack([d[0], d[1], Q[0, 0] - Q[1, 1], Q[0, 1], Q[0, 2], Q[1, 2]]).abs().max()
            else:
                off = torch.maximum(d.abs().max(), Q.abs().max())
            if float(off) > 1e-10:
                loose.append(top.types[i])
        if loose:
            warnings.warn(f"the multipoles of types {loose} are not symmetric about the free axis of their "
                          "z-only or frame-less frame: the energy depends on the orientation of the system",
                          stacklevel=2)

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

        # blocks of the block-Jacobi preconditioner: whole molecules up to _BLOCK_ATOMS atoms,
        # otherwise polarization groups; atoms of a larger group stand alone
        component = [-1] * n
        for start in range(n):
            if component[start] >= 0:
                continue
            component[start] = start
            stack = [start]
            while stack:
                a = stack.pop()
                for b in neighbors[a]:
                    if component[b] < 0:
                        component[b] = start
                        stack.append(b)
        comp_size: dict = {}
        group_size: dict = {}
        for i in range(n):
            comp_size[component[i]] = comp_size.get(component[i], 0) + 1
            group_size[atom_group[i]] = group_size.get(atom_group[i], 0) + 1
        block_ids: dict = {}
        fill: dict = {}
        pc_block, pc_slot = [], []
        for i in range(n):
            if comp_size[component[i]] <= _BLOCK_ATOMS:
                key = ("molecule", component[i])
            elif group_size[atom_group[i]] <= _BLOCK_ATOMS:
                key = ("group", atom_group[i])
            else:
                key = ("atom", i)
            block = block_ids.setdefault(key, len(block_ids))
            pc_block.append(block)
            pc_slot.append(fill.get(block, 0))
            fill[block] = fill.get(block, 0) + 1
        buf("pc_block", pc_block, (-1,))
        buf("pc_slot", pc_slot, (-1,))
        self.pc_blocks = len(block_ids)
        self.pc_size = max(fill.values()) if fill else 1

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
    def _frames(self, data: AtomicGraph, img) -> tuple[Tensor, Tensor]:
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
        vz = img(idx, iz)
        vx = img(idx, ix)
        vy = img(idx, iy)
        uz, ux, uy = _unit(vz), _unit(vx), _unit(vy)
        is_bis = (ftype == FRAME_BISECTOR)[:, None]
        is_zb = (ftype == FRAME_ZBISECT)[:, None]
        is_3f = (ftype == FRAME_THREEFOLD)[:, None]
        is_zo = (ftype == FRAME_ZONLY)[:, None]
        is_none = (ftype == FRAME_NONE)[:, None]
        # the axes as rows of an identity made on the device (no host-to-device copy)
        eye = torch.eye(3, dtype=uz.dtype, device=uz.device)
        ex, ey, ez = (eye[c].expand_as(uz) for c in range(3))
        zaxis = torch.where(is_bis, uz + ux, torch.where(is_3f, uz + ux + uy, uz))
        zaxis = _unit(torch.where(is_none, ez, zaxis))
        xref = torch.where(is_zb, ux + uy, ux)
        # a z-only frame (and a frame-less atom) takes its x axis from the lab x axis unless the
        # z axis lies within 45 degrees of it, then from the lab y axis, Tinker's convention
        # (Tinker9 switches at 30 degrees instead); the choice matters only for a local dipole
        # or quadrupole with components off the z axis
        zonly_ref = torch.where((zaxis[:, :1].abs() > _ZONLY_SWITCH), ey, ex)
        xref = torch.where(is_zo | is_none, zonly_ref, xref)
        xaxis = _unit(xref - (xref * zaxis).sum(-1, keepdim=True) * zaxis)
        yaxis = torch.cross(zaxis, xaxis, dim=1)
        # chirality: a y-defining atom on the negative side flips the y-odd components
        has_y = (ftype == FRAME_ZTHENX) & (self.frame_y.repeat(B) != torch.arange(n, device=data.pos.device).repeat(B))
        side = (vy * yaxis).sum(-1)
        chir = torch.where(has_y & (side < 0), -torch.ones_like(side), torch.ones_like(side))
        return torch.stack([xaxis, yaxis, zaxis], dim=2), chir

    @region("amoeba.multipoles")
    def _rotated_multipoles(self, data: AtomicGraph, B: int, img):
        """Global-frame charge, dipole and quadrupole of every atom."""
        row = self.mpole_row.repeat(B).clamp_min(0)
        has = self.has_mpole.repeat(B)
        q_rows, d_rows, Q_rows = self.ff.effective_multipoles()
        q = torch.where(has, q_rows[row], torch.zeros_like(q_rows[row]))
        d_loc = d_rows[row] * has[:, None].to(d_rows.dtype)
        Q_loc = Q_rows[row] * has[:, None, None].to(Q_rows.dtype)
        R, chir = self._frames(data, img)
        flip = torch.ones_like(d_loc)
        flip[:, 1] = chir
        d_loc = d_loc * flip
        Qflip = flip[:, :, None] * flip[:, None, :]
        Q_loc = Q_loc * Qflip
        d = matvec(R, d_loc)
        Q = matmul3(matmul3(R, Q_loc), R.transpose(1, 2))
        return q, d, Q

    # pair bookkeeping
    @region("amoeba.pairs")
    def _pairs(self, data: AtomicGraph, plan: Optional[StepPlan] = None):
        """Unique nonbonded pairs ``(i, k, r_vec)`` of the neighbor list
        with their scale factors ``(M, 8)``: vdw, mpole, polar, induce, rep,
        disp, direct, mutual, and their distances. A recording plan stores the pairs, their image
        shifts and scale factors; on a plan the vectors come from the
        positions, and the pairs beyond the cutoff and the padding get zero
        scale factors (every pair term of a molecule carries one, and those of a
        cell are cut by their switches or the Ewald ``keep`` in addition)."""
        if plan is None:
            src, dst = data.edge_index[0], data.edge_index[1]
            i, k, r, scales = self._select_pairs(data, src, dst, data.edge_vectors())
            return i, k, r, scales, torch.sqrt((r * r).sum(1) + TINY)
        if plan.recording:
            self._record_pairs(data, plan)
        i, k, shift = plan.pairs
        kernels = (bool(self.use_fast) and not self.training
                   and _kernels_supported(data.pos.device, data.pos.dtype))
        r = _plan_vectors(data, i, k, shift, plan.count, kernels, plan.rows if kernels else None)
        plan.select = None
        if self._pair_kernels_active(r) and (self.nonbonded == "hippo" or not self.ff.has_exchpol):
            # nothing differentiates the distances on the pair kernels (AMOEBA's exchange polarization
            # would): the distances, the cutoff mask and the visited pairs from two kernels and a scan
            key = (r.dtype, str(r.device))
            cut = self.__dict__.setdefault("_cutoff_sq", {})
            if key not in cut:
                cut[key] = torch.full((1,), self.cutoff ** 2, dtype=r.dtype, device=r.device)
            R, sel, count = _kernels().select_pairs_within(r, cut[key], plan.valid)
            plan.select = (sel, count)
            return i, k, r, plan.scales, R
        # one pass for the squared distances: the distances and the cutoff mask
        r2 = (r * r).sum(1)
        inside = r2.detach() < self.cutoff ** 2
        R = torch.sqrt(r2 + TINY)
        plan.select = None
        if self._pair_kernels_active(R):
            # the pair kernels visit the valid pairs within the cutoff, in order, and no others: the
            # scale factors need no mask
            active = inside if plan.valid is None else inside & (plan.valid > 0)
            plan.select = _kernels().select_pairs(active)
            return i, k, r, plan.scales, R
        return i, k, r, plan.scales * inside.to(r.dtype)[:, None], R

    def _record_pairs(self, data: AtomicGraph, plan: StepPlan, pairs: Optional[tuple] = None,
                      single_image: bool = False) -> None:
        """Store the unique pairs of the neighbor list of ``data`` in ``plan``: atoms,
        image shifts and scale factors, padded to the plan's capacity, and the
        positions they were found at. ``pairs``, the unique pairs ``(i, k, shift)``
        of one structure sorted by ``i``
        (:func:`~xnn.common.data.neighborlist.unique_pairs`), takes the place of the
        graph's neighbor list; ``single_image`` as in :meth:`stage_pairs`. Four entries
        ``(i, k, shift, count)`` are buffers of the plan's capacity with the number of
        pairs on the device (single images, :func:`~xnn.common.data.grid_pairs.grid_pairs`)."""
        # sorted by the first atom: the kernels add each block's contributions to that atom in
        # one operation, and the gathers of an atom's rows hit the cache
        if pairs is None:
            src, dst = data.edge_index[0], data.edge_index[1]
            i, k, _, scales = self._select_pairs(data, src, dst, data.edge_vectors())
            shift = None
            if data.cell is not None:
                shift = data.cell_shifts[(src < dst).nonzero()[:, 0]].to(data.pos.dtype)
            order = torch.argsort(narrow_keys(i, data.num_nodes), stable=True)
            i, k, scales = i[order], k[order], scales[order]
            if shift is not None:
                shift = shift[order]
        elif len(pairs) == 4:
            # buffers of the plan's capacity with the number of pairs on the device (a pair list
            # staged behind a running step reads nothing on the host): the entries past it get no
            # weight
            i, k, shift, total = pairs
            shift = None if data.cell is None else shift.to(data.pos.dtype)
            scales = self._pair_scales(data, i, k, None).detach()
            valid = (torch.arange(int(i.shape[0]), device=i.device) < total).to(scales.dtype)
            plan.valid, plan.count = valid, total.to(torch.int32)
            plan.pairs, plan.scales = (i, k, shift), scales * valid[:, None]
            plan.reference = data.pos.detach().clone()
            if bool(self.use_fast) and not self.training and _kernels_supported(data.pos.device, data.pos.dtype):
                plan.rows = _kernels().pair_rows(i, k, data.num_nodes, total, sorted_i=True)
            return
        else:
            i, k, shift = pairs
            shift = None if data.cell is None else shift.to(data.pos.dtype)
            r = None
            if not single_image:
                r = data.pos[k] - data.pos[i]
                if shift is not None:
                    r = r + shift @ data.cell.reshape(-1, 3, 3)[0].to(data.pos.dtype)
                r = r.to(data.model_dtype)
            scales = self._pair_scales(data, i, k, r)
        scales = scales.detach()
        M = int(i.shape[0])
        cap = M if plan.capacity is None else int(plan.capacity)
        if cap < M:
            raise ValueError(f"the plan holds {cap} pairs, the neighbor list has {M}")
        if cap > M and M > 0:
            # the padding repeats the first pair with zero weight
            fill = torch.zeros(cap - M, dtype=torch.long, device=i.device)
            i, k = torch.cat([i, i[fill]]), torch.cat([k, k[fill]])
            shift = None if shift is None else torch.cat([shift, shift[fill]])
            scales = torch.cat([scales, torch.zeros_like(scales[fill])])
        if plan.capacity is not None:
            plan.valid = (torch.arange(int(i.shape[0]), device=i.device) < M).to(scales.dtype)
            plan.count = torch.tensor([M], dtype=torch.int32, device=i.device)
        plan.pairs, plan.scales = (i, k, shift), scales
        plan.reference = data.pos.detach().clone()
        if bool(self.use_fast) and not self.training and _kernels_supported(data.pos.device, data.pos.dtype):
            # the rows of the induction's field kernel change only with the pairs
            plan.rows = _kernels().pair_rows(i, k, data.num_nodes, M, sorted_i=True)

    def plan_pairs(self, data: AtomicGraph, plan: StepPlan) -> bool:
        """Refresh the pair list of ``plan`` from the neighbor list of ``data`` in
        place, keeping the rest of the plan: a cheap rebuild while the atoms move
        continuously (an atom wrapped back into the cell changes the image shifts
        of its bonded vectors too and needs :meth:`plan_step`).

        Parameters
        ----------
        data : AtomicGraph
            The batched graph at the current positions, its neighbor list built
            with ``cutoff + skin``.
        plan : StepPlan
            The plan to refresh.

        Returns
        -------
        bool
            Whether the pairs fitted the plan's capacity and were copied; the
            plan is unchanged otherwise.
        """
        src, dst = data.edge_index[0], data.edge_index[1]
        if plan.capacity is not None and int((src < dst).sum()) > plan.capacity:
            return False
        fresh = StepPlan(plan.skin, plan.capacity)
        self._record_pairs(data, fresh)
        return plan.update(fresh, bonded=False)

    def refresh_pairs(self, data: AtomicGraph, plan: StepPlan, cutoff: float) -> bool:
        """Refresh the pair list of ``plan`` in place at the positions and cell of
        ``data``, one structure, from a half neighbor list of ``cutoff`` (the
        model's cutoff plus the skin), without building a graph: the same pairs
        as :meth:`plan_pairs`, found once each on the device.

        Parameters
        ----------
        data : AtomicGraph
            The plan's graph, its positions and cell at the current step.
        plan : StepPlan
            The plan to refresh.
        cutoff : float
            The radius of the pair list.

        Returns
        -------
        bool
            Whether the pairs fitted the plan's capacity and were copied; the
            plan is unchanged otherwise.
        """
        cell = None if data.cell is None else data.cell.detach().reshape(-1, 3, 3)[0]
        pbc = None if data.pbc is None else data.pbc.reshape(-1, 3)[0]
        fresh = self.stage_pairs(data, plan, unique_pairs(data.pos.detach(), cutoff, cell, pbc))
        return fresh is not None and plan.update(fresh, bonded=False)

    def stage_pairs(self, data: AtomicGraph, plan: StepPlan, pairs: tuple,
                    single_image: bool = False) -> Optional[StepPlan]:
        """The unique pairs ``(i, k, shift)`` of ``data``, one structure, sorted by
        ``i`` (:func:`~xnn.common.data.neighborlist.unique_pairs`), recorded as the
        pair list of a new plan for ``plan.update(fresh, bonded=False)``. ``plan``
        is not touched, so a step can still run on it meanwhile.

        Parameters
        ----------
        data : AtomicGraph
            The plan's graph, its positions and cell where the pairs were found.
        plan : StepPlan
            The plan whose capacity and skin the new one takes.
        pairs : tuple of Tensor
            The pairs and their integer image shifts, or those in buffers of the plan's
            capacity followed by their number on the device.
        single_image : bool, optional
            Every pair has one image within the list's cutoff (a cell more than
            twice the cutoff high on every axis), so a bonded pair's scale
            factors need no minimum-image test.

        Returns
        -------
        StepPlan or None
            The new pair list, or ``None`` when the pairs exceed the capacity.
        """
        if plan.capacity is not None and int(pairs[0].shape[0]) > plan.capacity:
            return None
        fresh = StepPlan(plan.skin, plan.capacity)
        with torch.no_grad():
            self._record_pairs(data, fresh, pairs, single_image)
        return fresh

    def _select_pairs(self, data: AtomicGraph, src: Tensor, dst: Tensor, vec: Tensor):
        """The unique pairs of the neighbor list and their scale factors, decided on the host."""
        sel = (src < dst).nonzero()[:, 0]
        i, k, r = src[sel], dst[sel], _take(vec, sel)
        return i, k, r, self._pair_scales(data, i, k, r)

    def _pair_scales(self, data: AtomicGraph, i: Tensor, k: Tensor, r: Optional[Tensor]) -> Tensor:
        """The eight scale factors of the unique pairs ``(i, k)`` with displacements ``r``:
        those of the bonded pairs from the topology (the minimum image of a bonded pair in a
        cell; any other image is a plain pair), ones otherwise. Without ``r`` every pair has a
        single image within the list's cutoff, which is then the minimum image."""
        n = int(self.frame_type.shape[0])
        start = data.batch[i] * n
        key = (i - start) * n + (k - start)
        if self.special_keys.numel() == 0:
            return torch.ones(i.shape[0], 8, dtype=data.model_dtype, device=i.device)
        pos = torch.searchsorted(self.special_keys, key).clamp_max(self.special_keys.shape[0] - 1)
        hit = self.special_keys[pos] == key
        if data.cell is not None and r is not None:
            # the scale factors belong to the minimum image of a bonded pair; any other
            # image within the cutoff is a plain pair
            sel = hit.nonzero()[:, 0]
            closest = pair_vectors(data, i[sel], k[sel]).detach()
            same = ((r[sel].detach() - closest) ** 2).sum(-1) < 1e-10
            hit = hit.clone()
            hit[sel] = same
        return torch.where(hit[:, None], self.special_scales[pos], torch.ones_like(self.special_scales[pos]))

    # forward
    def forward(self, data: AtomicGraph, plan: Optional[StepPlan] = None) -> dict[str, Tensor]:
        """Evaluate the energy on a (batched) atomic graph.

        Parameters
        ----------
        data : AtomicGraph
            The batched graph; every structure must be a conformation of
            the bound topology, with the neighbor list built with this
            model's ``cutoff``.
        plan : StepPlan, optional
            The host decisions of this structure, from :meth:`plan_step`:
            the evaluation reads them instead of deciding them again, and
            takes its pairs from the plan instead of the neighbor list.

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
        P = self.ff.params
        if plan is None or plan.recording:
            if not bool((data.n_atoms == n).all()):
                raise ValueError(f"every structure must have the bound topology's {n} atoms "
                                 f"(got {data.n_atoms.tolist()})")
            if not bool((data.atomic_numbers == self.top_z.repeat(B)).all()):
                raise ValueError("atomic numbers do not match the bound topology's element sequence")
        else:
            # the solves of this evaluation and of its backward pass report here
            plan.converged.fill_(True)
        for predictor in (self.predictor, self.adjoint_predictor):
            if predictor is not None:
                # set again by this evaluation's pushes, so a rewind takes back only what it stored
                predictor.pushed = False
        dtype = P["bond_k"].dtype
        cls_idx = self.top_class.repeat(B)
        typ_idx = self.top_type.repeat(B)

        # the nonbonded pairs first: their count decides whether the evaluation runs fused
        i_, k_, r_vec, scales, R = self._pairs(data, plan)
        ew = self._ewald_setup(data, i_, R, plan)

        # the glue between the kernels is compiled on a step plan only (molecular dynamics): a compiled
        # block is differentiable once, a plain evaluation may be differentiated twice (Hessians, force
        # losses through the kernels)
        active = self._fused_active(R) and plan is not None
        # the valence terms and the frames compiled as one block (not while a plan records its
        # image shifts: that evaluation runs once)
        recording = plan is not None and plan.recording
        with fused(active and not recording):
            val = self._valence(data, plan if recording else None, None if plan is None or recording else plan.shifts,
                                B, N, None if plan is None or recording else plan.image_pairs[:plan.valence_calls])
        # the image calls of the terms that follow continue the recorded sequence
        img = _Images(data, plan)
        img.calls = val["img_calls"]
        if recording:
            plan.valence_calls = int(val["img_calls"])
        out_terms = val["terms"]
        q_perm, d_perm, Q_perm = val["q"], val["d"], val["Q"]
        br, bt, bi, theta, theta_plain, theta0, ai = (val[k] for k in ("br", "bt", "bi", "theta", "theta_plain",
                                                                      "theta0", "ai"))
        e_bond, e_angle = val["e_bond"], val["e_angle"]
        with fused(active):
            nb = self._nonbonded(data, i_, k_, r_vec, R, scales, q_perm, d_perm, Q_perm, cls_idx, typ_idx, B, N, ew,
                                 img, plan)
        out_terms.update(nb["terms"])

        e_ref_node = self.atom_ref(data.atomic_numbers).squeeze(-1)
        out_terms["e_ref"] = e_ref_node
        mu_ind = nb["induced"]
        with fused(active):
            stacked, node_energy, dipole = _assemble(list(out_terms.values()), data.batch, B, q_perm, data.pos,
                                                     d_perm, mu_ind, dtype)
        energy = self.aggregate_energy(node_energy, data)
        # every term's totals in one scatter, outside the compiled block (whose reduction over the atoms of a
        # few rows the compiler spreads over a few programs)
        totals = structure_sums(stacked, data.batch, B)

        if self.keep_intermediates:
            self.intermediates = dict(bond_r=br, e_bond=e_bond, angle_theta=theta, e_angle=e_angle,
                                      q=q_perm, d=d_perm, Q=Q_perm, induced=mu_ind, pair_i=i_, pair_k=k_,
                                      pair_r=R, pair_scales=scales, ewald=None if ew is None else ew["summary"],
                                      **nb.get("extra", {}))
        result = {"node_energy": node_energy, "energy": energy, "charges": q_perm,
                  "node_features": q_perm.unsqueeze(1), "induced_dipoles": mu_ind, "dipole": dipole}
        for t, k in enumerate(out_terms):
            result[k] = totals[t]
        return result

    @fusable
    @region("amoeba.valence")
    def _valence(self, data: AtomicGraph, plan: Optional[StepPlan], shifts: Optional[list], B: int, N: int,
                 pairs: Optional[list] = None) -> dict:
        """The valence terms per atom, the global-frame multipoles with the charge flux, and
        the bond and angle quantities the intermediates report; one compiled block on the
        fused path. ``plan`` is given while it records its image shifts, ``shifts`` (its list)
        when they are replayed, with ``pairs`` (the plan's atom pairs of these calls) in one gather;
        the count of image calls is returned for the terms that follow."""
        img = _Images(data, plan, shifts, pairs)
        n = int(self.top_type.shape[0])
        device = data.pos.device
        P = self.ff.params
        ff = self.ff
        s = ff.settings
        dtype = P["bond_k"].dtype
        starts = torch.arange(B, device=device) * n

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
        bvec = img(bi[0], bi[1])
        br = torch.sqrt((bvec * bvec).sum(1) + TINY)
        dr = br - P["bond_r0"][bt]
        e_bond = P["bond_k"][bt] * dr * dr * (1.0 + float(s["bond-cubic"]) * dr + float(s["bond-quartic"]) * dr * dr)
        out_terms["e_bond"] = scatter_sum(0.5 * e_bond, bi[0], N) + scatter_sum(0.5 * e_bond, bi[1], N)

        # angles (harmonic or in-plane projected), sextic polynomial in degrees
        ai = expand(self.angle_index)
        at = tile(self.angle_type)
        aenv = tile(self.angle_env)
        u = img(ai[1], ai[0])
        v = img(ai[1], ai[2])
        theta_plain = _angle(u, v)
        if self._inplane_any:
            fourth = expand(self.angle_fourth.clamp_min(0).unsqueeze(0))[0]
            w = img(ai[1], fourth)          # D - B
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
            r1 = torch.sqrt((img(a_sel[1], a_sel[0]) ** 2).sum(1) + TINY)
            r2 = torch.sqrt((img(a_sel[1], a_sel[2]) ** 2).sum(1) + TINY)
            k = P["strbnd_k"][sbt]
            k1 = torch.where(flip, k[:, 1], k[:, 0])
            k2 = torch.where(flip, k[:, 0], k[:, 1])
            # ideal bond lengths of the two bonds
            r01 = self._bond_r0_of(a_sel[1], a_sel[0], B)
            r02 = self._bond_r0_of(a_sel[1], a_sel[2], B)
            e_sb = (k1 * (r1 - r01) + k2 * (r2 - r02)) * (_take(theta_plain, sbi) - _take(theta0, sbi))
            out_terms["e_strbnd"] = scatter_sum(e_sb, a_sel[1], N)
        else:
            out_terms["e_strbnd"] = zeros_n

        # Urey-Bradley
        if self.urey_angle.numel():
            ubi = expand_angles(self.urey_angle.unsqueeze(0))[0]
            ubt = tile(self.urey_type)
            a_sel = ai[:, ubi]
            rv = img(a_sel[0], a_sel[2])
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
            ab = img(b_, a_)
            cb = img(b_, c_)
            db = img(b_, d_)
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
            c1, s1 = dihedral_cos_sin(data, di, img)
            nmax = ff.torsion_nmax
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
            pc = _take(data.pos, c_).to(dtype)
            pd = _take(data.pos, d_).to(dtype)
            ad = img(d_, a_)
            bd = img(d_, b_)
            ec = img(c_, e_)
            gc = img(c_, g_)
            p_site = pc + torch.cross(ad, bd, dim=1)
            q_site = pd + torch.cross(ec, gc, dim=1)
            b1 = pc - p_site
            b2 = img(c_, d_)
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
            c_a, s_a = dihedral_cos_sin(data, ti[:4], img)
            c_b, s_b = dihedral_cos_sin(data, ti[1:5], img)
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
                if m not in self._tortor_maps:
                    continue
                sel = ttt == m
                e_tt = e_tt + torch.where(sel, self._tortor_energy(m, x, y), torch.zeros_like(x))
            out_terms["e_tortor"] = scatter_sum(e_tt, ti[2], N)
        else:
            out_terms["e_tortor"] = zeros_n

        # permanent multipoles (global frame) and the charge flux
        q_perm, d_perm, Q_perm = self._rotated_multipoles(data, B, img)
        dq = self._charge_flux(data, B, br, bt, bi, theta_plain, theta0, ai, N) if self._has_cflux() else None
        if dq is not None:
            q_perm = q_perm + dq

        return {"terms": out_terms, "q": q_perm, "d": d_perm, "Q": Q_perm, "br": br, "bt": bt, "bi": bi,
                "theta": theta, "theta_plain": theta_plain, "theta0": theta0, "ai": ai, "e_bond": e_bond,
                "e_angle": e_angle, "img_calls": img.calls}

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
            flux = j * (_take(br, sel) - P["bond_r0"][bt[sel]])
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
            dr_ab = _take(br, bsel[:, 0]) - P["bond_r0"][bt[bsel[:, 0]]]
            dr_cb = _take(br, bsel[:, 1]) - P["bond_r0"][bt[bsel[:, 1]]]
            dth = _take(theta, sel) - _take(theta0, sel)
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

    def _vdw_offsets(self, data: AtomicGraph, B: int, img) -> Tensor:
        """The van der Waals sites as offsets from their atoms (hydrogens reduced along their bond):
        ``(red - 1) (H - X)`` from the bonded vector alone, so an atom wrapped into another image of
        the cell than its parent keeps its site."""
        device = data.pos.device
        key = (B, str(device), self.vdw_red.dtype)
        cache = self.__dict__.setdefault("_vdw_index", {})
        if key not in cache:
            # constants of the topology and the batch size, made once
            n = int(self.top_type.shape[0])
            starts = torch.arange(B, device=device) * n
            idx = torch.arange(n, device=device).repeat(B) + starts.repeat_interleave(n)
            parent = self.vdw_parent.repeat(B) + starts.repeat_interleave(n)
            cache[key] = (idx, parent, (self.vdw_red.repeat(B) - 1.0)[:, None])
        idx, parent, shrink = cache[key]
        vec = img(parent, idx)                        # H - X
        return shrink.to(vec.dtype) * vec

    def plan_step(self, data: AtomicGraph, skin: float = 0.0, capacity: Optional[int] = None) -> StepPlan:
        """Record the host decisions of an evaluation of ``data`` for later
        evaluations of the same structure as it moves (see :class:`StepPlan`);
        the recording evaluation itself runs without gradients and leaves the
        dipole predictor alone.

        Parameters
        ----------
        data : AtomicGraph
            The batched graph, its neighbor list built with ``cutoff + skin``.
        skin : float, optional
            The margin of the neighbor list beyond the model's cutoff in
            Angstrom, by default 0 (the plan then holds at these positions only).
        capacity : int, optional
            The number of pairs the plan holds, padded beyond the unique
            pairs of the neighbor list; by default exactly those.

        Returns
        -------
        StepPlan
            The plan, for ``forward(data, plan)``.

        Raises
        ------
        ValueError
            If the neighbor list has more unique pairs than ``capacity``.
        """
        plan = StepPlan(skin, capacity)
        with torch.no_grad():
            self.forward(data, plan)
        plan.recording = False
        plan.converged = torch.ones((), dtype=torch.bool, device=data.pos.device)
        plan.needed = torch.zeros(len(SOLVE_PHASES), dtype=torch.float64, device=data.pos.device)
        return plan

    def set_use_fast(self, use_fast: Union[bool, str]) -> "Amoeba":
        """Set when the fast implementation runs (see ``use_fast`` of the class).

        Parameters
        ----------
        use_fast : bool or str
            ``True``, ``False`` or ``"auto"``.

        Returns
        -------
        Amoeba
            The model.
        """
        self.use_fast = resolve_use_fast(use_fast)
        return self

    def _kernels_active(self, R: Tensor) -> bool:
        """Whether this evaluation runs the fused pair kernels: the fast path on a CUDA device
        where Triton is available (the compiled arithmetic serves it otherwise)."""
        return self._fused_active(R) and _kernels_supported(R.device, R.dtype)

    def _pair_kernels_active(self, R: Tensor) -> bool:
        """Whether the nonbonded pairs run on the fused pair kernels: the kernels are active and the
        library has no pair-specific van der Waals values (which the eager function applies)."""
        return self._kernels_active(R) and (self.nonbonded == "hippo" or not self.ff.vdw_pair_keys)

    def _fused_active(self, R: Tensor) -> bool:
        """Whether this evaluation takes the fast implementation: on a GPU with ``use_fast=True`` in
        evaluation and training alike (the pair kernels are differentiable twice); with ``"auto"`` above
        the measured crossover of the mode (``FAST_POLICY``, ``TRAINING_POLICY``)."""
        if not self.use_fast or not R.is_cuda:
            return False
        if isinstance(self.use_fast, str):
            policy = TRAINING_POLICY if self.training else FAST_POLICY
            return policy.decide(R.device, R.dtype, int(R.shape[0]))
        return True

    def rewind_predictors(self) -> None:
        """Take the dipoles of the last evaluation back out of the predictors that
        stored them, before the step is evaluated again (see
        :meth:`DipolePredictor.rewind`)."""
        for predictor in (self.predictor, self.adjoint_predictor):
            if predictor is not None:
                predictor.rewind()

    def set_dipole_predictor(self, kind: Optional[str], history: int = 6) -> None:
        """Switch the induced-dipole predictor on (``"aspc"``, ``"gear"``,
        ``"previous"``) or off (``None`` or ``"none"``); see :class:`DipolePredictor`.

        Parameters
        ----------
        kind : str or None
            The extrapolation scheme.
        history : int, optional
            Number of past dipole sets it uses (6).
        """
        if kind is None or str(kind).lower() == "none":
            self.predictor = None
            self.adjoint_predictor = None
        else:
            self.predictor = DipolePredictor(str(kind).lower(), history)
            # the adjoint of the backward solve changes as smoothly as the dipoles
            self.adjoint_predictor = DipolePredictor(str(kind).lower(), history)

    @region("amoeba.induction")
    def _induced(self, i_, k_, r_vec, R, field_d, field_p, D1, D2, N, B, alpha, data, ew,
                 x0: Optional[Tensor] = None, plan: Optional[StepPlan] = None, expol: Optional[Tensor] = None,
                 kernels: bool = False):
        """Solve the mutual induction and return ``(mu, e_polar_node)``.

        ``field_d`` / ``field_p`` are the direct- and polar-scaled permanent
        fields at every atom (mesh and self contributions included for
        cells); ``D1, D2`` the mutual-induction ladder per pair (scale
        factors and, for cells, the real-space Ewald part included);
        ``alpha`` the per-atom polarizability. Molecules are solved with a
        dense matrix (or iteratively when large), cells iteratively with
        the mesh field of the induced dipoles. ``x0`` is a starting guess
        of the iterative solve; without it the model's ``dipole_guess`` (used
        once) or the predictor's extrapolation is tried, and the solve starts
        from whichever of the guess and the direct dipoles has the smaller
        residual. A plan supplies the count of polarizable atoms, and its
        recording evaluation leaves the predictor alone. ``expol``, the
        exchange-polarization tensors ``K`` (:func:`exchange_polarization_tensor`),
        turns the polarizabilities into ``alpha (I + K)^-1``.
        """
        n = int(self.top_type.shape[0])
        polar = alpha > 0
        if expol is None:
            mu_dir = alpha[:, None] * field_d
        else:
            eye = torch.eye(3, dtype=R.dtype, device=R.device)
            mu_dir = alpha[:, None] * matvec(_inverse3(eye + expol), field_d)
        if plan is not None and plan.replaying():
            has_polar, n_polar = plan.has_polar, plan.n_polar
        else:
            n_polar = int(polar.sum())
            has_polar = n_polar > 0
            if plan is not None:
                plan.has_polar, plan.n_polar = has_polar, n_polar
        if self.polarization == "direct" or not has_polar:
            mu = mu_dir
        else:
            if self._use_dense(n, B, ew):
                T = dipole_field_tensor(r_vec, D1, D2)                    # (M, 3, 3)
                A = torch.zeros(B, n, n, 3, 3, dtype=R.dtype, device=R.device)
                bi = i_ // n
                li, lk = i_ % n, k_ % n
                # accumulated: the padding of a plan repeats a pair (with zero weight)
                A.index_put_((bi, li, lk), -T, accumulate=True)
                A.index_put_((bi, lk, li), -T.transpose(1, 2), accumulate=True)
                inv_alpha = torch.where(polar, 1.0 / alpha.clamp_min(1e-30), torch.full_like(alpha, 1.0))
                eye = torch.eye(3, dtype=R.dtype, device=R.device)
                inv = inv_alpha[:, None, None] * eye
                if expol is not None:
                    inv = torch.where(polar[:, None, None], inv_alpha[:, None, None] * (eye + expol), eye)
                diag = inv.view(B, n, 3, 3)
                ar = torch.arange(n, device=R.device)
                A[:, ar, ar] = diag
                Afull = A.permute(0, 1, 3, 2, 4).reshape(B, 3 * n, 3 * n)
                rhs = (field_d * polar[:, None].to(R.dtype)).reshape(B, 3 * n, 1)
                # solve_ex leaves the error flag on the device instead of reading it
                mu = torch.linalg.solve_ex(Afull, rhs)[0].reshape(N, 3)
                mu = mu * polar[:, None].to(R.dtype)
                if self.keep_intermediates:
                    self._induction_matrix = Afull.detach()
            else:
                dtype = R.dtype
                pos = ew["pos"] if ew is not None else data.pos.to(dtype)
                cell = (ew["cells"] if ew is not None
                        else torch.zeros(B, 3, 3, dtype=dtype, device=R.device))
                op = _MutualField(i_, k_, N, polar, ew, self.polar_eps, self.max_iter, pos, cell)
                op.kernels = kernels
                op.count = None if plan is None else plan.count
                if plan is not None and plan.select is not None:
                    op.select = (plan.select[1], plan.select[0])
                else:
                    op.select = (op.count, None)
                op.rows = None if plan is None else plan.rows
                # under d-equals-p the direct and polar fields coincide, so the adjoint of the
                # polarization energy is -C/2 times the dipoles themselves (the backward checks that
                # its incoming gradient is that energy's before taking the shortcut)
                d_equals_p = bool(self.ff.settings.get("d-equals-p", False))
                # otherwise, when a gradient will follow on the kernels, the dipoles of the polar field
                # (the adjoint of the polarization energy, times -2/C) are solved with those of the
                # direct field in one set of iterations (the paired solve)
                paired = (not d_equals_p and kernels and not self.training and field_d.requires_grad
                          and self.preconditioner == "diagonal"
                          and op.pairable())
                op.polar_field = field_p.detach() * polar[:, None].to(dtype) if (d_equals_p or paired) else None
                op.paired_rhs = op.polar_field if paired else None
                op.stats, op.adjoints = self.solver_stats, self._adjoints
                op.n_polar = max(n_polar, 1)
                # the statistics read the device; evaluations on a plan do without
                op.collect_stats = plan is None
                if plan is not None and plan.replaying():
                    op.fixed, op.converged, op.needed = plan.solve_iterations, plan.converged, plan.needed
                if self.preconditioner != "diagonal":
                    shift = (torch.arange(B, device=R.device) * self.pc_blocks).repeat_interleave(n)
                    op.precond = _Preconditioner(self.preconditioner, alpha, polar, r_vec, D1, D2, i_, k_, R,
                                                 self.pc_block.repeat(B) + shift, self.pc_slot.repeat(B),
                                                 B * self.pc_blocks, self.pc_size, self.precond_cutoff, N,
                                                 expol)
                op.adjoint_predictor = self.adjoint_predictor if (self.predictor is not None
                                                                  and not self.training) else None
                predict = (self.predictor is not None and not self.training
                           and not (plan is not None and plan.recording))
                if x0 is None and self.dipole_guess is not None:
                    x0, self.dipole_guess = self.dipole_guess, None
                if x0 is None and predict:
                    x0 = self.predictor.predict(field_d)
                if x0 is not None and x0.shape == field_d.shape:
                    op.guess = x0.detach().to(device=field_d.device, dtype=dtype)
                if op.paired_rhs is not None and op.guess is not None:
                    # the polar field's dipoles from the extrapolated or the last adjoint
                    lam = self.adjoint_predictor.predict(field_d) if (predict and self.adjoint_predictor
                                                                      is not None) else None
                    if lam is None:
                        lam = self._adjoints.get((tuple(field_d.shape), dtype, str(field_d.device), False))
                    op.paired_guess = None if lam is None else lam.detach() / (-0.5 * COULOMB)
                mu = _ImplicitInduction.apply(field_d * polar[:, None].to(dtype), alpha, expol, r_vec, D1, D2, pos,
                                              cell, op)
                mu = mu * polar[:, None].to(dtype)
                if predict:
                    self.predictor.push(mu.detach())
        e_pol = -0.5 * (mu * field_p).sum(1) * COULOMB
        return mu, e_pol

    def _use_dense(self, n: int, B: int, ew) -> bool:
        """Whether the induced dipoles of a batch of ``B`` molecules of ``n`` atoms are
        solved directly: one block-diagonal system, ``B`` dense blocks of ``3n``."""
        if self.solver == "dense" or self._force_dense:
            if ew is not None:
                raise ValueError("the dense induction solve has no reciprocal part; cells need "
                                 "solver='iterative' (and the polarizability tensor is for molecules)")
            return True
        if self.solver == "iterative":
            return False
        return ew is None and n <= _DENSE_ATOMS and B * (3 * n) ** 2 <= _DENSE_ENTRIES

    # Ewald sums of periodic cells
    def _ewald_layout(self, data: AtomicGraph, dtype: torch.dtype) -> Optional[dict]:
        """The host decisions of the Ewald sums of a batch: which structures are
        cells, their meshes and slab axes, the cutoffs and coefficients; ``None``
        when no structure is a periodic cell (or Ewald is off)."""
        B = data.num_graphs
        cells = data.cell.to(dtype)
        periodic = cell_volume(cells.detach()) > 1e-8
        n_pbc = torch.full((B,), 3, dtype=torch.long, device=cells.device)
        if data.pbc is not None:
            n_pbc = data.pbc.to(cells.device).sum(-1)
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
        rc_d = self.dispersion_cutoff if self.dispersion_cutoff is not None else rc
        if rc_d > self.cutoff + 1e-9:
            raise ValueError(f"dispersion_cutoff {rc_d} exceeds the neighbor cutoff {self.cutoff}")
        alpha_e = self.ewald_alpha if self.ewald_alpha is not None else ewald_coefficient(rc)
        # the per-structure decisions are made on host copies (one transfer, no per-structure sync)
        lengths = torch.linalg.norm(cells.detach().cpu(), dim=-1).tolist()
        periodic_host = periodic.cpu().tolist()
        n_pbc_host = n_pbc.cpu().tolist()
        pbc_host = data.pbc.cpu() if data.pbc is not None else None
        structures = []
        for b in range(B):
            if not periodic_host[b]:
                continue
            grid = list(self.pme_grid) if self.pme_grid is not None else pme_grid_size(lengths[b])
            dgrid = (list(self.dpme_grid) if self.dpme_grid is not None
                     else pme_grid_size(lengths[b], density=0.8))
            # a slab: the free axis, whose normal enters the Yeh-Berkowitz correction
            free = int((~pbc_host[b]).nonzero()[0, 0]) if int(n_pbc_host[b]) == 2 else None
            structures.append((b, grid, dgrid, free))
        return {"structures": structures, "rc": rc, "rc_d": rc_d, "alpha_e": alpha_e,
                "alpha_p": self.pewald_alpha if self.pewald_alpha is not None else alpha_e,
                "alpha_d": self.dewald_alpha if self.dewald_alpha is not None else ewald_coefficient(rc_d)}

    @region("amoeba.ewald_setup")
    def _ewald_setup(self, data: AtomicGraph, i_: Tensor, R: Tensor, plan: Optional[StepPlan] = None):
        """Ewald parameters of the batch, or ``None`` when no structure is a
        periodic cell (or Ewald is off). The host decisions come from the plan
        when it has them (:meth:`_ewald_layout`)."""
        if self.ewald is False or data.cell is None:
            return None
        dtype, device = R.dtype, R.device
        if plan is not None and plan.replaying():
            layout = plan.ewald
        else:
            layout = self._ewald_layout(data, dtype)
            if plan is not None:
                plan.ewald = layout
        if layout is None:
            return None
        rc, rc_d = layout["rc"], layout["rc_d"]
        alpha_e, alpha_p, alpha_d = layout["alpha_e"], layout["alpha_p"], layout["alpha_d"]
        cells = data.cell.to(dtype)
        # the per-structure and per-pair constants of a plan (which structures are periodic, which
        # pairs belong to one, the Ewald coefficients per pair) are kept with the plan's layout
        consts = None if plan is None or plan.recording else layout.get("pair_consts")
        if consts is None:
            periodic = cell_volume(cells.detach()) > 1e-8
            if data.pbc is not None:
                periodic = periodic & (data.pbc.to(device).sum(-1) >= 2)
        else:
            periodic = consts["periodic"]
        n = int(self.top_type.shape[0])
        structures = []
        for b, grid, dgrid, free in layout["structures"]:
            idx = torch.arange(b * n, (b + 1) * n, device=device)
            normal = None
            if free is not None:
                a1, a2 = cells[b][(free + 1) % 3], cells[b][(free + 2) % 3]
                normal = torch.linalg.cross(a1, a2)
                normal = normal / torch.linalg.norm(normal)
            structures.append((b, idx, grid, dgrid, normal))

        def grouped(slot: int) -> list:
            # structures sharing a mesh are summed together, each on its own copy of the mesh; the
            # membership and its index tensors are fixed for a structure and kept in the plan
            key = "groups" if slot == 2 else "dgroups"
            members = None if plan is None or plan.recording else layout.get(key)
            if members is None:
                by_grid: dict = {}
                for j, st in enumerate(structures):
                    by_grid.setdefault(tuple(st[slot]), []).append(j)
                members = [(list(grid), js,
                            torch.tensor([structures[j][0] for j in js], dtype=torch.long, device=device),
                            torch.cat([structures[j][1] for j in js]),
                            torch.arange(len(js), device=device).repeat_interleave(n))
                           for grid, js in by_grid.items()]
                if plan is not None:
                    layout[key] = members
            out = []
            # a plan keeps the influence functions of its fixed cells across evaluations
            caches = None if plan is None else layout.setdefault("influence_" + key, [{} for _ in members])
            for j, (grid, js, bs, idx, batch) in enumerate(members):
                normals = None
                if any(structures[j][4] is not None for j in js):
                    normals = torch.stack([structures[j][4] if structures[j][4] is not None
                                           else torch.zeros(3, dtype=dtype, device=device) for j in js])
                # a group of every structure of the batch covers every atom: no gathers, no scatters
                out.append({"bs": bs, "idx": idx, "batch": batch, "grid": list(grid), "normals": normals,
                            "influence": None if caches is None else caches[j],
                            "whole": len(js) == int(data.num_graphs)})
            return out
        if consts is None:
            pair_periodic = periodic[data.batch[i_]]
            ones = torch.ones_like(R)
            # the padding of a plan's pair list is kept out of the Ewald sums too
            valid = ones if plan is None or plan.valid is None else plan.valid
            pf = pair_periodic.to(dtype)
            consts = {"periodic": periodic, "pair_periodic": pair_periodic, "ones": ones, "valid": valid,
                      "per": pf, "alpha_e": alpha_e * pf, "alpha_p": alpha_p * pf, "alpha_d": alpha_d * pf}
            if plan is not None:
                layout["pair_consts"] = consts
        pair_periodic, ones, valid = consts["pair_periodic"], consts["ones"], consts["valid"]
        summary = {"alpha": alpha_e, "pewald_alpha": alpha_p, "dewald_alpha": alpha_d, "cutoff": rc,
                   "dispersion_cutoff": rc_d, "grids": [(b, grid, dgrid) for b, _, grid, dgrid, _ in structures],
                   "slabs": [b for b, _, _, _, normal in structures if normal is not None],
                   "mesh_self": self.mesh_self}
        kernels = self._kernels_active(R)
        # the pair kernels form the real-space factors from the distances themselves
        in_kernels = self._pair_kernels_active(R)
        return {"periodic": periodic, "mesh_self": self.mesh_self, "pair_periodic": pair_periodic,
                "structures": structures, "cells": cells, "pos": data.pos.to(dtype),
                # the mesh kernels are differentiable once: training takes the eager mesh
                "groups": grouped(2), "dgroups": grouped(3), "plans": {}, "kernels": kernels and not self.training,
                "alpha_e": consts["alpha_e"], "alpha_p": consts["alpha_p"], "alpha_d": consts["alpha_d"],
                "per": consts.get("per", pair_periodic.to(dtype)), "rc": rc, "rc_d": rc_d,
                "keep": None if in_kernels else torch.where(pair_periodic, (R < rc).to(dtype), ones) * valid,
                "keep_d": None if in_kernels else torch.where(pair_periodic, (R < rc_d).to(dtype), ones) * valid,
                "alpha_e_value": alpha_e, "alpha_p_value": alpha_p, "alpha_d_value": alpha_d,
                "order": self.pme_order, "porder": self.ppme_order or self.pme_order, "dorder": self.dpme_order,
                "summary": summary}

    def _exchange_polarization(self, i_: Tensor, k_: Tensor, r_vec: Tensor, R: Tensor, scales: Tensor,
                               cls_idx: Tensor, N: int, switch: Optional[Tensor],
                               plan: Optional[StepPlan] = None) -> Optional[Tensor]:
        """The exchange-polarization tensors of the atoms (:func:`exchange_polarization_tensor`), or
        ``None`` when the library has no exchange polarization. A pair takes part when either of its
        classes is used, with the polarization scale factors of the permanent field (the
        intra-group ones within a polarization group) and, in a cell, the switch of the repulsion."""
        if not self.ff.has_exchpol:
            return None
        P = self.ff.params
        ci, ck = cls_idx[i_], cls_idx[k_]
        use = self.ff.expol_use
        weight = scales[:, 2] * (_take(use, ci) | _take(use, ck)).to(R.dtype)
        weight = weight * (R.detach() < self.cutoff).to(R.dtype)
        if plan is not None and plan.valid is not None:
            weight = weight * plan.valid
        if switch is not None:
            weight = weight * switch
        return exchange_polarization_tensor(r_vec, R, i_, k_, N, _take(P["expol_spring"], ci),
                                            _take(P["expol_spring"], ck), _take(P["expol_size"], ci),
                                            _take(P["expol_size"], ck), _take(P["expol_damp"], ci),
                                            _take(P["expol_damp"], ck), weight)

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

    @region("amoeba.mesh_energy")
    def _reciprocal_energy(self, data: AtomicGraph, ew, q: Tensor, d: Tensor, Q: Tensor) -> Tensor:
        """Per-atom mesh and self energies of the permanent multipoles, ``e^2 / A``."""
        pos = ew["pos"]
        out = torch.zeros_like(q)
        ae = ew["alpha_e_value"]
        for g, plan in _mesh_plans(ew, "m", ew["order"]):
            idx, b = g["idx"], g["batch"]
            qi, di, Qi = ((t if g["whole"] else _take(t, idx)) for t in (q, d, Q))
            phi, dphi, d2phi = plan.potential(qi, di, Qi, ae, hessian=True)
            # the potential gradient serves the induction's field too when its Ewald coefficient and
            # mesh are these (Tinker's defaults): one mesh pass, one backward through it
            ew.setdefault("mesh_gradients", {})[id(plan)] = (float(ae), dphi)
            e = _mesh_energy_terms(qi, di, Qi, phi, dphi, d2phi, float(ae))
            if self.mesh_self:
                s0, s2, s4 = plan.self_moments(ae, "coulomb")
                s2, s4 = s2[b], s4[b]
                e = e - (qi * qi * s0[b] + torch.einsum("na,nab,nb->n", di, s2, di)
                         - 2.0 * qi * torch.einsum("nab,nab->n", Qi, s2)
                         + torch.einsum("nab,nabcd,ncd->n", Qi, s4, Qi))
            if g["normals"] is not None:
                # Yeh-Berkowitz: 2 pi / V M_n^2 with M_n the dipole moment along the slab normal
                normal = g["normals"][b]
                moment = qi * (pos[idx] * normal).sum(-1) + (di * normal).sum(-1)
                total = scatter_sum(moment, b, plan.groups)[b]
                e = e + 2.0 * math.pi / plan.volume[b] * moment * total
            out = out + e if g["whole"] else out.index_add(0, idx, e)
        return out

    @region("amoeba.mesh_field")
    def _reciprocal_field(self, data: AtomicGraph, ew, q: Tensor, d: Tensor, Q: Tensor) -> Tensor:
        """Mesh field of the permanent multipoles at every atom, self field included."""
        pos = ew["pos"]
        out = torch.zeros_like(d)
        ap = ew["alpha_p_value"]
        coef = self_field_coefficient(ap)
        for g, plan in _mesh_plans(ew, "m", ew["porder"]):
            idx, b = g["idx"], g["batch"]
            qi, di, Qi = ((t if g["whole"] else _take(t, idx)) for t in (q, d, Q))
            known = ew.get("mesh_gradients", {}).get(id(plan))
            if known is not None and known[0] == float(ap):
                dphi = known[1]
            else:
                _, dphi, _ = plan.potential(qi, di, Qi, ap, hessian=False)
            field = -dphi + coef * di
            if self.mesh_self:
                s2 = plan.self_moments(ap, "coulomb", max_rank=2)[1]
                field = field + 2.0 * torch.einsum("na,nab->nb", di, s2[b])
            if g["normals"] is not None:
                normal = g["normals"][b]
                moment = scatter_sum(qi * (pos[idx] * normal).sum(-1) + (di * normal).sum(-1), b, plan.groups)[b]
                field = field - (4.0 * math.pi / plan.volume[b] * moment)[:, None] * normal
            out = out + field if g["whole"] else out.index_add(0, idx, field)
        return out

    @region("hippo.mesh_dispersion")
    def _reciprocal_dispersion(self, data: AtomicGraph, ew, c: Tensor) -> Tensor:
        """Per-atom mesh (and self) energies of the ``-c_i c_j / R^6`` sum."""
        out = torch.zeros_like(c)
        ad = ew["alpha_d_value"]
        for g, plan in _mesh_plans(ew, "d", ew["dorder"]):
            idx = g["idx"]
            ci = c if g["whole"] else _take(c, idx)
            e = plan.dispersion(ci, ad)
            if self.mesh_self:
                s0 = plan.self_moments(ad, "r6")[0]
                e = e + s0[g["batch"]] * ci * ci
            out = out + e if g["whole"] else out.index_add(0, idx, e)
        return out

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

    @region("amoeba.nonbonded")
    def _nonbonded(self, data, i_, k_, r_vec, R, scales, q, d, Q, cls_idx, typ_idx, B, N, ew, img=None,
                   plan=None) -> dict:
        """AMOEBA nonbonded terms: buffered 14-7, permanent multipoles, Thole induction."""
        if self._pair_kernels_active(R):
            return self._nonbonded_kernels(data, i_, k_, r_vec, R, scales, q, d, Q, cls_idx, typ_idx, B, N, ew, plan,
                                           img)
        P = self.ff.params
        s = self.ff.settings
        terms: dict[str, Tensor] = {}
        ci, ck = cls_idx[i_], cls_idx[k_]
        # every per-atom quantity is gathered once per side of the pairs
        q_i, q_k, d_i, d_k, Q_i, Q_k = (_take(t, idx) for t in (q, d, Q) for idx in (i_, k_))
        per = None if ew is None else ew["pair_periodic"]
        # van der Waals between the reduced sites
        offset = self._vdw_offsets(data, B, img).to(r_vec.dtype)
        vvec = r_vec + _take(offset, k_) - _take(offset, i_)
        rv = torch.sqrt((vvec * vvec).sum(1) + TINY)
        rmin_i, rmin_k = _take(P["vdw_rmin"], ci), _take(P["vdw_rmin"], ck)
        eps_i, eps_k = _take(P["vdw_eps"], ci), _take(P["vdw_eps"], ck)
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
        if self.ff.vdw_pair_keys:
            # pair-specific values where the library has them (decided on the host, no synchronization)
            pidx = self.ff.vdw_pair_index[ci, ck]
            has_pair = pidx >= 0
            rmin = torch.where(has_pair, P["vdw_pair_rmin"][pidx.clamp_min(0)], rmin)
            eps = torch.where(has_pair, P["vdw_pair_eps"][pidx.clamp_min(0)], eps)
        e_vdw = buffered_14_7(rv, rmin, eps, float(s["delta-halgren"]), float(s["gamma-halgren"]))
        e_vdw = e_vdw * scales[:, 0]
        e_vdw_raw = e_vdw
        sw = self._switch(rv, per, self.vdw_cutoff, self.vdw_taper)
        if sw is not None:
            e_vdw = e_vdw * sw
        terms["e_vdw"] = scatter_sum(0.5 * e_vdw, i_, N) + scatter_sum(0.5 * e_vdw, k_, N)

        # permanent multipoles and the real-space fields of the induction, as one function of the
        # pairs: the Coulomb ladder for molecules; for cells the real-space Ewald ladder with the
        # excluded fraction of the bare interaction removed, plus the mesh and self terms
        polar = self.polarization != "none"
        alpha = P["polarizability"][typ_idx] if polar else None
        thole = None
        if polar:
            th = P["thole"][typ_idx]
            thole = (_take(alpha, i_), _take(alpha, k_), _take(th, i_), _take(th, k_))
        pb = amoeba_pair_block(R, r_vec, scales, q_i, q_k, d_i, d_k, Q_i, Q_k, i_, k_, N,
                               None if ew is None else ew["keep"], None if ew is None else ew["alpha_e"],
                               None if ew is None else ew["alpha_p"], thole)
        e_mp = pb["e_mp"]
        e_node = pb["e_mp_node"]
        if ew is not None:
            e_node = e_node + self._reciprocal_energy(data, ew, q, d, Q) * COULOMB
        terms["e_multipole"] = e_node

        # induction with Thole damping
        mu = torch.zeros_like(d)
        if polar:
            field_d, field_p, D1u, D2u = pb["fields"]
            if ew is not None:
                rec = self._reciprocal_field(data, ew, q, d, Q)
                field_d, field_p = field_d + rec, field_p + rec
            expol = self._exchange_polarization(i_, k_, r_vec, R, scales, cls_idx, N,
                                                self._switch(R, per, self.repulsion_cutoff, None), plan)
            mu, e_pol = self._induced(i_, k_, r_vec, R, field_d, field_p, D1u, D2u, N, B, alpha, data, ew,
                                      plan=plan, expol=expol)
            terms["e_polar"] = e_pol
        else:
            terms["e_polar"] = torch.zeros(N, dtype=R.dtype, device=R.device)
        return {"terms": terms, "induced": mu,
                "extra": {"vdw_r": rv, "e_vdw_pair": e_vdw, "e_vdw_pair_raw": e_vdw_raw, "e_mpole_pair": e_mp}}

    def _nonbonded_kernels(self, data, i_, k_, r_vec, R, scales, q, d, Q, cls_idx, typ_idx, B, N, ew, plan,
                           img=None) -> dict:
        """The AMOEBA nonbonded terms through the fused pair kernels (:mod:`.amoeba_kernels`): the
        same quantities as the eager function, one kernel for the pair terms and fields, one for
        their gradients, and the induction's real-space products in kernels."""
        P = self.ff.params
        s = self.ff.settings
        km = _kernels()
        dtype, device = R.dtype, R.device
        M = int(R.shape[0])
        polar = self.polarization != "none"
        # the per-atom table, made once per plan (see the HIPPO kernels)
        params = None if plan is None else plan.kernel_params
        if params is None or params.dtype != dtype or params.device != device:
            zeros = torch.zeros(N, dtype=dtype, device=device)
            cols = [P["vdw_rmin"][cls_idx], P["vdw_eps"][cls_idx],
                    P["polarizability"][typ_idx] if polar else zeros, P["thole"][typ_idx] if polar else zeros]
            # outside a plan the table keeps its graph: the kernels return its gradient (training,
            # parameter derivatives); a plan's table is a constant of the molecular dynamics
            params = torch.stack(cols, 1)
            if plan is not None:
                params = params.detach()
                plan.kernel_params = params
        # the van der Waals sites as offsets from the atoms (hydrogens reduced along their bond)
        offsets = self._vdw_offsets(data, B, img).to(r_vec.dtype)
        if ew is None:
            keep = torch.ones(M, dtype=dtype, device=device)
            per = torch.zeros(M, dtype=dtype, device=device)
            alpha_e = alpha_p = 0.0
        else:
            keep, per = ew["keep"], ew["per"]
            alpha_e, alpha_p = ew["alpha_e_value"], ew["alpha_p_value"]
        # the switch of the van der Waals term (as _switch decides it) and the combining rules
        explicit = self.vdw_cutoff is not None or self.vdw_taper is not None or self.taper is not None
        cut = float(self.vdw_cutoff) if self.vdw_cutoff is not None else self.cutoff
        if cut > self.cutoff + 1e-9:
            raise ValueError(f"a term cutoff of {cut} exceeds the neighbor cutoff {self.cutoff}")
        tap = (float(self.vdw_taper) if self.vdw_taper is not None
               else (self.taper if self.taper is not None else 0.9 * cut))
        rule, erule = str(s["radiusrule"]).upper(), str(s["epsilonrule"]).upper()
        modes = (2 if explicit else (1 if ew is not None else 0),
                 2 if rule.startswith("CUBIC") else (1 if rule.startswith("GEO") else 0),
                 2 if erule == "HHG" else (1 if erule.startswith("ARITH") else 0))
        window = (tap, cut) if modes[0] else (0.0, 0.0)
        rc = 0.0 if ew is None else float(ew["rc"])
        key = (float(alpha_e), float(alpha_p), window, rc, str(device))
        tables = self.__dict__.setdefault("_kernel_scalars", {})
        if key not in tables:
            tables[key] = km.amoeba_scalar_table(COULOMB, alpha_e, alpha_p, window, rc, float(s["delta-halgren"]),
                                                 float(s["gamma-halgren"]), TINY, device)
        select, count = None, None if plan is None else plan.count
        if plan is not None and plan.select is not None:
            select, count = plan.select
        e_node, field_d, field_p, du, e_pair = km.AmoebaPairs.apply(
            r_vec, q, d, Q, offsets, i_, k_, scales, keep, per, params, tables[key], polar, modes,
            bool(self.keep_intermediates), count, select)
        e_mp, e_vdw = e_node.unbind(1)
        if ew is not None:
            e_mp = e_mp + self._reciprocal_energy(data, ew, q, d, Q) * COULOMB
        terms = {"e_vdw": e_vdw, "e_multipole": e_mp}
        mu = torch.zeros_like(d)
        if polar:
            alpha = P["polarizability"][typ_idx]
            if ew is not None:
                rec = self._reciprocal_field(data, ew, q, d, Q)
                field_d, field_p = field_d + rec, field_p + rec
            expol = None
            if self.ff.has_exchpol:
                expol = self._exchange_polarization(i_, k_, r_vec, R, scales, cls_idx, N,
                                                    self._switch(R, None if ew is None else ew["pair_periodic"],
                                                                 self.repulsion_cutoff, None), plan)
            du1, du2 = du.unbind(1)
            mu, e_pol = self._induced(i_, k_, r_vec, R, field_d, field_p, du1, du2, N, B, alpha, data, ew,
                                      plan=plan, expol=expol, kernels=True)
            terms["e_polar"] = e_pol
        else:
            terms["e_polar"] = torch.zeros(N, dtype=dtype, device=device)
        extra = {}
        if self.keep_intermediates:
            extra = {"e_mpole_pair": e_pair[:, 0], "e_vdw_pair": e_pair[:, 1]}
        return {"terms": terms, "induced": mu, "extra": extra}

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

        from xnn.common.config.options import model_options

        extra = model_options(cfg)

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

    @region("hippo.nonbonded")
    def _nonbonded(self, data, i_, k_, r_vec, R, scales, q, d, Q, cls_idx, typ_idx, B, N, ew, img=None,
                   plan=None) -> dict:
        if self._kernels_active(R):
            return self._nonbonded_kernels(data, i_, k_, r_vec, R, scales, q, d, Q, cls_idx, typ_idx, B, N, ew, plan)
        P = self.ff.params
        terms: dict[str, Tensor] = {}
        ci, ck = cls_idx[i_], cls_idx[k_]
        # every per-atom quantity is gathered once per side of the pairs
        q_i, q_k, d_i, d_k, Q_i, Q_k = (_take(t, idx) for t in (q, d, Q) for idx in (i_, k_))
        per = None if ew is None else ew["pair_periodic"]
        params = {}
        for key, name in (("core", "chgpen_core"), ("zeta", "chgpen_alpha"), ("rep_k", "rep_k"),
                          ("rep_alpha", "rep_alpha"), ("rep_valence", "rep_valence"), ("disp_alpha", "disp_alpha"),
                          ("disp_c6", "disp_c6"), ("chgtrn_eps", "chgtrn_eps"), ("chgtrn_alpha", "chgtrn_alpha")):
            params[key + "_i"], params[key + "_k"] = _take(P[name], ci), _take(P[name], ck)
        switches = {"repulsion": self._switch(R, per, self.repulsion_cutoff, None),
                    "dispersion": None if ew is not None else self._switch(R, per, self.dispersion_cutoff, None),
                    "chgtrn": self._switch(R, per, self.chgtrn_cutoff, None)}
        polar = self.polarization != "none"
        # one compiled block wins on small systems; on large ones its backward recomputes the
        # exponential ladders, and the block runs with its parts compiled one by one instead
        block = hippo_pair_block if int(R.shape[0]) < _HIPPO_BLOCK_PAIRS else hippo_pair_block.__wrapped__
        pb = block(R, r_vec, scales, q_i, q_k, d_i, d_k, Q_i, Q_k, i_, k_, N, params,
                              None if ew is None else ew["keep"], None if ew is None else ew["alpha_e"],
                              None if ew is None else ew["alpha_p"], None if ew is None else ew["alpha_d"],
                              None if ew is None else ew["keep_d"], switches, polar)
        e_mp, e_rep, e_disp, e_ct = pb["e_mp"], pb["e_rep"], pb["e_disp"], pb["e_ct"]
        e_node, e_dnode = pb["e_mp_node"], pb["e_disp_node"]
        if ew is not None:
            e_node = e_node + self._reciprocal_energy(data, ew, q, d, Q) * COULOMB
            e_dnode = e_dnode + self._reciprocal_dispersion(data, ew, P["disp_c6"][cls_idx])
        terms["e_multipole"] = e_node
        terms["e_repulsion"] = pb["e_rep_node"]
        terms["e_dispersion"] = e_dnode
        terms["e_chgtrn"] = pb["e_ct_node"]

        mu = torch.zeros_like(d)
        if polar:
            alpha = P["polarizability"][typ_idx]
            field_d, field_p, D1u, D2u = pb["fields"]
            if ew is not None:
                rec = self._reciprocal_field(data, ew, q, d, Q)
                field_d, field_p = field_d + rec, field_p + rec
            expol = self._exchange_polarization(i_, k_, r_vec, R, scales, cls_idx, N, switches["repulsion"],
                                                plan)
            mu, e_pol = self._induced(i_, k_, r_vec, R, field_d, field_p, D1u, D2u, N, B, alpha, data, ew,
                                      plan=plan, expol=expol)
            terms["e_polar"] = e_pol
        else:
            terms["e_polar"] = torch.zeros(N, dtype=R.dtype, device=R.device)
        return {"terms": terms, "induced": mu,
                "extra": {"e_mpole_pair": e_mp, "e_rep_pair": e_rep, "e_disp_pair": e_disp, "e_ct_pair": e_ct}}


    def _nonbonded_kernels(self, data, i_, k_, r_vec, R, scales, q, d, Q, cls_idx, typ_idx, B, N, ew, plan,
                           img=None) -> dict:
        """The HIPPO nonbonded terms through the fused pair kernels (:mod:`.amoeba_kernels`): the
        same quantities as the eager functions, one kernel for the pair terms and fields, one for
        their gradients, and the induction's real-space products in kernels."""
        P = self.ff.params
        km = _kernels()
        dtype, device = R.dtype, R.device
        M = int(R.shape[0])
        per_bool = None if ew is None else ew["pair_periodic"]
        # the per-atom parameter table, made once per plan, and the scalars
        params = None if plan is None else plan.kernel_params
        if params is None or params.dtype != dtype or params.device != device:
            cols = [P[name][cls_idx] for name in km.PARAMS[:9]]
            if self.ff.has_exchpol:
                cols += [P[name][cls_idx] for name in km.PARAMS[9:]] + [self.ff.expol_use[cls_idx].to(dtype)]
            else:
                cols += [torch.zeros(N, dtype=dtype, device=device)] * 4
            # outside a plan the table keeps its graph: the kernels return its gradient (training,
            # parameter derivatives); a plan's table is a constant of the molecular dynamics
            params = torch.stack(cols, 1)
            if plan is not None:
                params = params.detach()
                plan.kernel_params = params
        if ew is None:
            keep = keep_d = torch.ones(M, dtype=dtype, device=device)
            per = torch.zeros(M, dtype=dtype, device=device)
            alpha_e = alpha_p = alpha_d = 0.0
        else:
            keep, keep_d, per = ew["keep"], ew["keep_d"], ew["per"]
            alpha_e, alpha_p, alpha_d = ew["alpha_e_value"], ew["alpha_p_value"], ew["alpha_d_value"]
        windows: dict = {}
        modes = []
        for key, cutoff_value in (("rep", self.repulsion_cutoff), ("ct", self.chgtrn_cutoff),
                                  ("disp", self.dispersion_cutoff)):
            explicit = cutoff_value is not None or self.taper is not None
            if (key == "disp" and ew is not None) or (per_bool is None and not explicit):
                modes.append(0)
                continue
            cut = float(cutoff_value) if cutoff_value is not None else self.cutoff
            if cut > self.cutoff + 1e-9:
                raise ValueError(f"a term cutoff of {cut} exceeds the neighbor cutoff {self.cutoff}")
            windows[key] = (self.taper if self.taper is not None else 0.9 * cut, cut)
            modes.append(2 if explicit else 1)
        # the scalar table is made once per set of values (no host-to-device copy per step, which
        # would block the capture of a CUDA graph)
        rcs = (0.0, 0.0) if ew is None else (float(ew["rc"]), float(ew["rc_d"]))
        key = (float(alpha_e), float(alpha_p), float(alpha_d), tuple(sorted(windows.items())), rcs, str(device))
        tables = self.__dict__.setdefault("_kernel_scalars", {})
        if key not in tables:
            tables[key] = km.scalar_table(COULOMB, alpha_e, alpha_p, alpha_d, windows, self.cutoff, device, rcs)
        scalars = tables[key]
        polar = self.polarization != "none"
        expol = polar and bool(self.ff.has_exchpol)
        # on a plan the kernels visit its pairs within the cutoff, in order (selected by _pairs): the
        # pairs of its skin and its padding add nothing (zero scale factors, and in a cell zero Ewald
        # keep beyond the cutoff)
        select, count = None, None if plan is None else plan.count
        if plan is not None and plan.select is not None:
            select, count = plan.select
        e_node, field_d, field_p, du, K, e_pair = km.HippoPairs.apply(
            r_vec, q, d, Q, i_, k_, scales, keep, keep_d, per, params, scalars, polar, expol, tuple(modes),
            bool(self.keep_intermediates), count, bool(self.mixed_precision), select)
        e_mp, e_rep, e_disp, e_ct = e_node.unbind(1)
        if ew is not None:
            e_mp = e_mp + self._reciprocal_energy(data, ew, q, d, Q) * COULOMB
            e_disp = e_disp + self._reciprocal_dispersion(data, ew, P["disp_c6"][cls_idx])
        terms = {"e_multipole": e_mp, "e_repulsion": e_rep, "e_dispersion": e_disp, "e_chgtrn": e_ct}
        mu = torch.zeros_like(d)
        if polar:
            alpha = P["polarizability"][typ_idx]
            if ew is not None:
                rec = self._reciprocal_field(data, ew, q, d, Q)
                field_d, field_p = field_d + rec, field_p + rec
            du1, du2 = du.unbind(1)
            mu, e_pol = self._induced(i_, k_, r_vec, R, field_d, field_p, du1, du2, N, B, alpha, data, ew,
                                      plan=plan, expol=K if expol else None, kernels=True)
            terms["e_polar"] = e_pol
        else:
            terms["e_polar"] = torch.zeros(N, dtype=dtype, device=device)
        extra = {}
        if self.keep_intermediates:
            extra = {"e_mpole_pair": e_pair[:, 0], "e_rep_pair": e_pair[:, 1], "e_disp_pair": e_pair[:, 2],
                     "e_ct_pair": e_pair[:, 3]}
        return {"terms": terms, "induced": mu, "extra": extra}


# the periodic and solver options of the constructor, also read by from_config
_OPTION_KEYS = ("ewald", "ewald_alpha", "pewald_alpha", "dewald_alpha", "ewald_cutoff", "pme_grid", "pme_order",
                "ppme_order", "dpme_grid", "dpme_order", "vdw_cutoff", "vdw_taper", "taper", "repulsion_cutoff",
                "chgtrn_cutoff", "dispersion_cutoff", "polar_eps", "mesh_self", "solver", "max_iter",
                "predictor", "predictor_history", "preconditioner", "precond_cutoff", "conserve_charge",
                "symmetry_mask", "use_fast", "mixed_precision")


def _grid_option(value) -> Optional[list]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        value = [value]
    out = [fft_mesh_size(x) for x in value]
    return out * 3 if len(out) == 1 else out


class DipolePredictor:
    """Extrapolate the induced dipoles of the next step from the last few.

    In molecular dynamics the converged dipoles change smoothly from step to
    step, so a linear combination of the last ``m`` sets,
    ``mu_p = sum_j B_j mu(t - j dt)``, is a much better start for the
    iterative solve than the direct dipoles. The schemes:

    * ``"aspc"``: the always stable predictor-corrector coefficients of
      Kolafa, J. Comput. Chem. 25, 335 (2004), with ``k = m - 2``:
      ``B_j = (-1)^(j+1) j C(2k+4, k+2-j) / C(2k+2, k+1)``; time reversible
      and stable, exact for linear motion;
    * ``"gear"``: polynomial extrapolation through the ``m`` points,
      ``B_j = (-1)^(j+1) C(m, j)``, exact for polynomials of degree ``m - 1``;
    * ``"previous"``: the last dipoles.

    Fewer stored sets than ``m`` (the first steps) use the same scheme on the
    sets there are. A change of shape, dtype or device starts a new history.
    The solve is converged to the model's tolerance, so the prediction only
    changes the number of iterations, not the dipoles.

    Parameters
    ----------
    kind : str, optional
        ``"aspc"`` (default), ``"gear"`` or ``"previous"``.
    history : int, optional
        The number ``m`` of past dipole sets (6).
    """

    KINDS = ("aspc", "gear", "previous")

    def __init__(self, kind: str = "aspc", history: int = 6):
        if kind not in self.KINDS:
            raise ValueError(f"predictor must be one of {self.KINDS} (got {kind!r})")
        if int(history) < 1:
            raise ValueError("the predictor history needs at least one dipole set")
        self.kind = kind
        self.history = 1 if kind == "previous" else int(history)
        self._sets: list = []
        # a full history: the sets and a spare (the oldest set of the last push, for rewind) as the rows of
        # one buffer, so a push shifts them in place with one copy
        self._buffer: Optional[Tensor] = None
        self._grew = False
        # whether the evaluation that is running or last ran stored dipoles here
        self.pushed = False

    @staticmethod
    def coefficients(kind: str, m: int) -> list:
        """The weights ``B_1 .. B_m`` of the last ``m`` sets, newest first.

        Parameters
        ----------
        kind : str
            ``"aspc"``, ``"gear"`` or ``"previous"``.
        m : int
            Number of sets.

        Returns
        -------
        list of float
            The weights (they sum to one).
        """
        if kind == "previous" or m == 1:
            return [1.0]
        if kind == "gear":
            return [float((-1) ** (j + 1) * math.comb(m, j)) for j in range(1, m + 1)]
        k = m - 2
        return [(-1) ** (j + 1) * j * math.comb(2 * k + 4, k + 2 - j) / math.comb(2 * k + 2, k + 1)
                for j in range(1, k + 3)]

    def reset(self) -> None:
        """Forget the stored dipoles."""
        self._sets = []
        self._buffer = None
        self._grew = False

    def push(self, mu: Tensor) -> None:
        """Store the converged dipoles of a step.

        Parameters
        ----------
        mu : Tensor
            The dipoles ``(N, 3)``.
        """
        if self._sets and (self._sets[0].shape != mu.shape or self._sets[0].dtype != mu.dtype
                           or self._sets[0].device != mu.device):
            self.reset()
        self.pushed = True
        if len(self._sets) == self.history:
            # a full history shifts in place, so a recorded CUDA graph advances it on replay; the
            # oldest set goes to the spare row, so a step evaluated again can take this one back
            if self._buffer is None:
                self._buffer = torch.stack(self._sets + [self._sets[-1]])
                self._sets = list(self._buffer[:self.history].unbind(0))
            self._buffer.copy_(torch.cat([mu.detach()[None].to(self._buffer.dtype), self._buffer[:self.history]]))
            self._grew = False
        else:
            self._sets = [mu.detach().clone()] + self._sets
            self._grew = True

    def rewind(self) -> None:
        """Take back the dipoles of the last :meth:`push`, as many times as it ran
        (a recorded CUDA graph pushes on every replay), before a step is evaluated
        again at the same positions: the history then holds each step once. Only
        the last push can be taken back."""
        if not self.pushed or not self._sets:
            return
        if self._grew:
            self._sets = self._sets[1:]
            self._grew = False
        elif self._buffer is not None:
            self._buffer[:self.history].copy_(self._buffer[1:].clone())
        self.pushed = False

    def predict(self, like: Tensor) -> Optional[Tensor]:
        """The extrapolated dipoles, or ``None`` without a compatible history.

        Parameters
        ----------
        like : Tensor
            A tensor of the dipoles' shape, dtype and device.

        Returns
        -------
        Tensor or None
            The prediction ``(N, 3)``.
        """
        if not self._sets or self._sets[0].shape != like.shape or self._sets[0].dtype != like.dtype \
                or self._sets[0].device != like.device:
            return None
        return _extrapolate(tuple(self.coefficients(self.kind, len(self._sets))), tuple(self._sets))


@fusable
def _assemble(values: list, batch: Tensor, B: int, q: Tensor, pos: Tensor, d: Tensor, mu: Tensor,
              dtype: torch.dtype) -> tuple:
    """The terms ``values`` stacked ``(T, N)``, the per-atom energies (their sum) and the molecular
    dipoles (charges, permanent and induced dipoles); one compiled block on the fused path."""
    stacked = torch.stack([v.to(dtype) for v in values])
    dip_atoms = q[:, None] * pos.to(dtype) + d + mu
    return stacked, stacked.sum(0), scatter_sum(dip_atoms, batch, B)


@fusable
def _extrapolate(weights: tuple, sets: tuple) -> Tensor:
    """``sum(w * mu)`` over the stored dipole sets: one generated kernel on the fused path."""
    out = weights[0] * sets[0]
    for w, mu in zip(weights[1:], sets[1:]):
        out = out + w * mu
    return out


#: ``use_fast="auto"``: the minimum number of pairs per GPU model and precision from which the fast
#: implementation runs in evaluation; it is faster at every size measured, so it runs on every GPU
FAST_POLICY = AutoPolicy()
#: The same for training (an energy and force loss through the kernels)
TRAINING_POLICY = AutoPolicy()

# pairs from which the solver's real-space product runs compiled under use_fast (measured on an
# A30: a loss of 5-9 % at about 5e4 pairs, a gain of 3-22 % from about 4e5)
_FUSED_SOLVE_PAIRS = 200_000
# pairs up to which the HIPPO pair terms run as one compiled block (A30: 8-27 % faster up to about
# 1.6e5 pairs, 5-6 % slower from 4e5)
_HIPPO_BLOCK_PAIRS = 200_000

# the direct solve: molecules up to this size, and batches up to this many matrix entries
_DENSE_ATOMS = 1000
_DENSE_ENTRIES = 30_000_000
# molecules up to this size are blocks of the block-Jacobi preconditioner
_BLOCK_ATOMS = 32


class _Preconditioner:
    """Approximate inverses of the induced-dipole matrix ``A = 1/alpha - T``.

    ``"block"`` inverts the real-space blocks of whole molecules (or of
    polarization groups): ``B`` holds ``1/alpha`` on the diagonal and ``-T``
    of the pairs inside a block, every block is factorized by Cholesky once
    per evaluation and inverted (a block that is not positive definite falls
    back to its diagonal). ``"neumann"`` adds the first term of the Neumann
    series of the remaining near-field pairs, ``M^-1 = B^-1 + B^-1 T_near
    B^-1``, with ``B`` the blocks or the diagonal; it is symmetric, and
    positive definite as long as the near field is weaker than the blocks,
    which the solver checks. Everything is built from detached tensors: the
    preconditioner changes the iterations, not the solution or its gradient.
    With exchange-polarization tensors ``K`` the diagonal blocks are
    ``(I + K) / alpha``.
    """

    KINDS = ("diagonal", "block", "neumann", "block-neumann")

    def __init__(self, kind: str, alpha: Tensor, polar: Tensor, r: Tensor, D1: Tensor, D2: Tensor, i_: Tensor,
                 k_: Tensor, R: Tensor, block: Tensor, slot: Tensor, n_blocks: int, size: int, cutoff: float,
                 N: int, expol: Optional[Tensor] = None):
        with torch.no_grad():
            r, D1, D2 = r.detach(), D1.detach(), D2.detach()
            dtype, device = r.dtype, r.device
            self.N = N
            self.mask = polar[:, None].to(dtype)
            expol = None if expol is None else expol.detach()
            self.diag = _diagonal_inverse(alpha.detach(), expol, self.mask)
            pair_ok = polar[i_] & polar[k_]
            self.binv = None
            same = torch.zeros_like(pair_ok)
            if kind in ("block", "block-neumann") and size > 1:
                same = pair_ok & (block[i_] == block[k_])
                s = int(size)
                Ab = torch.zeros(n_blocks, s, s, 3, 3, dtype=dtype, device=device)
                g, a, b = block[i_][same], slot[i_][same], slot[k_][same]
                t = dipole_field_tensor(r[same], D1[same], D2[same])
                Ab.index_put_((g, a, b), -t, accumulate=True)
                Ab.index_put_((g, b, a), -t.transpose(1, 2), accumulate=True)
                inv_alpha = torch.where(polar, 1.0 / alpha.detach().clamp_min(1e-30), torch.ones_like(alpha))
                eye = torch.eye(3, dtype=dtype, device=device)
                occupied = torch.zeros(n_blocks, s, dtype=torch.bool, device=device)
                occupied[block, slot] = True
                own = eye if expol is None else eye + expol
                Ab[block, slot, slot] = Ab[block, slot, slot] + inv_alpha[:, None, None] * own
                pad = (~occupied).nonzero()
                Ab[pad[:, 0], pad[:, 1], pad[:, 1]] = eye
                full = Ab.permute(0, 1, 3, 2, 4).reshape(n_blocks, 3 * s, 3 * s)
                factor, info = torch.linalg.cholesky_ex(full)
                binv = torch.cholesky_inverse(factor)
                failed = info != 0
                if bool(failed.any()):
                    d = torch.diagonal(full[failed], dim1=1, dim2=2)
                    binv[failed] = torch.diag_embed(1.0 / d)
                self.binv, self.block, self.slot, self.n_blocks, self.size = binv, block, slot, n_blocks, s
            self.near = None
            if kind in ("neumann", "block-neumann"):
                near = pair_ok & ~same & (R.detach() < cutoff)
                self.near = (i_[near], k_[near], r[near], D1[near], D2[near])

    def _blocks(self, r: Tensor) -> Tensor:
        if self.binv is None:
            return self.diag(r)
        rb = torch.zeros(self.n_blocks, self.size, 3, dtype=r.dtype, device=r.device)
        rb[self.block, self.slot] = r
        zb = torch.bmm(self.binv, rb.reshape(self.n_blocks, 3 * self.size, 1)).reshape(self.n_blocks, self.size, 3)
        return zb[self.block, self.slot] * self.mask

    def __call__(self, r: Tensor) -> Tensor:
        z = self._blocks(r)
        if self.near is not None:
            i, k, r, D1, D2 = self.near
            f = scatter_sum(dipole_field(z[i], r, D1, D2), k, self.N) + scatter_sum(dipole_field(z[k], r, D1, D2), i,
                                                                                  self.N)
            z = z + self._blocks(f * self.mask)
        return z


@fusable
def _mesh_energy_terms(q: Tensor, d: Tensor, Q: Tensor, phi: Tensor, dphi: Tensor, d2phi: Tensor,
                       alpha: float) -> Tensor:
    """Per-atom mesh energy of the multipoles in their mesh potential, plus the self energy."""
    e = 0.5 * (q * phi + (d * dphi).sum(-1) + (Q * d2phi).sum((-1, -2)))
    return e + multipole_self_energy(q, d, Q, torch.full_like(q, alpha))


def _mesh_plans(ew, kind: str, order: int, pos: Optional[Tensor] = None, cells: Optional[Tensor] = None) -> list:
    """The mesh plans of the periodic structures of a batch, one per group of
    structures sharing a mesh (``kind`` ``"m"`` for the multipole mesh, ``"d"``
    for dispersion), paired with their groups. Without ``pos`` and ``cells``
    they are built once per evaluation from the batch's positions and cells
    and shared by every caller; with them they are built afresh."""
    groups = ew["groups"] if kind == "m" else ew["dgroups"]
    if pos is None:
        key = (kind, int(order))
        if key not in ew["plans"]:
            ew["plans"][key] = [MeshPlan(ew["pos"] if g["whole"] else _take(ew["pos"], g["idx"]),
                                         _take(ew["cells"], g["bs"]), g["batch"],
                                         g["grid"], order, kernels=ew.get("kernels", False),
                                         influence_cache=g.get("influence"))
                                for g in groups]
        return list(zip(groups, ew["plans"][key]))
    return [(g, MeshPlan(pos if g["whole"] else _take(pos, g["idx"]), _take(cells, g["bs"]), g["batch"], g["grid"],
                         order,
                         kernels=ew.get("kernels", False))) for g in groups]


# the iterative induced-dipole solve of large molecules and periodic cells
class _MutualField:
    """The field of a set of induced dipoles at every atom: the real-space
    pairs through their mutual ladders ``(r, D1, D2)`` (:meth:`real`, matrix
    free, :func:`dipole_field`) and, in cells,
    the mesh field plus the self field and, for slabs, the Yeh-Berkowitz
    field (:meth:`mesh`). The two parts depend on disjoint inputs (the
    ladders; the positions and cells), which keeps the products of the implicit
    backward pass exact. The mesh part runs on the evaluation's shared mesh
    plans, built from ``pos`` and ``cell``, so the solve and its adjoint
    redo none of the spline or influence-function setup."""

    def __init__(self, i_: Tensor, k_: Tensor, N: int, polar: Tensor, ew, tol: float, max_iter: int,
                 pos: Tensor, cell: Tensor):
        self.i, self.k, self.N = i_, k_, N
        # one gather and one scatter per product: both directions of every pair at once, made
        # when the eager product first needs them
        self._ik: Optional[tuple] = None
        # the real-space product runs compiled when the evaluation is fused and has enough pairs
        # to repay the fixed cost of a compiled call on every iteration, in the forward solve and
        # in the adjoint solve of the backward pass alike
        self.fused = fused_active() and int(i_.shape[0]) >= _FUSED_SOLVE_PAIRS
        # the mesh field's stencil arithmetic, compiled whenever the evaluation is fused
        self.fused_mesh = fused_active()
        # the real-space product through the fused kernel (the mutual ladders stacked once per solve)
        self.kernels = False
        # the count of valid pairs of a plan (the kernels skip the padding), and the count and
        # selection of the pairs a kernel visits per pair (those within the cutoff)
        self.count: Optional[Tensor] = None
        self.select: tuple = (None, None)
        # the pairs in compressed rows per atom for the field kernel (a plan's, or built at first use)
        self.rows: Optional[tuple] = None
        self._du: Optional[tuple] = None
        self.polar_field: Optional[Tensor] = None
        # the paired solve: the right-hand side of the polar field's dipoles, their guess, and the
        # dipoles themselves once solved
        self.paired_rhs: Optional[Tensor] = None
        self.paired_guess: Optional[Tensor] = None
        self.polar_dipoles: Optional[Tensor] = None
        self.polar = polar
        self.tol, self.max_iter = tol, max_iter
        self.ew = ew
        self.pos, self.cell = pos, cell
        self.structures = [] if ew is None else ew["structures"]
        self.alpha_p = 0.0 if ew is None else ew["alpha_p_value"]
        self.order = 0 if ew is None else ew["porder"]
        self.mesh_self = False if ew is None else bool(ew["mesh_self"])
        self.guess: Optional[Tensor] = None
        self.precond: Optional["_Preconditioner"] = None
        self.n_polar: Optional[int] = None
        # solves of fixed length per phase, their convergence and the iterations they needed, on the
        # device (a plan's)
        self.fixed: Optional[dict] = None
        self.converged: Optional[Tensor] = None
        self.needed: Optional[Tensor] = None
        self.collect_stats = True
        self.phase = "forward"
        self.stats: dict = {}
        self.adjoints: dict = {}
        self.adjoint_predictor: Optional[DipolePredictor] = None
        # the shared plans must carry the graph of the positions and cells: build them here,
        # outside the no-grad solve, if no earlier term has
        self.plans()

    def _directions(self) -> tuple:
        if self._ik is None:
            self._ik = (torch.cat([self.i, self.k]), torch.cat([self.k, self.i]))
        return self._ik

    @property
    def ik(self) -> Tensor:
        """The pairs' first atoms, then their second atoms ``(2 M,)``."""
        return self._directions()[0]

    @property
    def ki(self) -> Tensor:
        """The pairs' second atoms, then their first atoms ``(2 M,)``."""
        return self._directions()[1]

    def real(self, mu: Tensor, pair, guard: Optional[Tensor] = None) -> Tensor:
        r, D1, D2 = pair
        if self.kernels:
            if self._du is None or self._du[0] is not pair:
                # the two columns of the pair kernel's (M, 2) output as they are, else stacked
                du = D1._base
                if (du is None or du is not D2._base or tuple(du.shape) != (D1.shape[0], 2)
                        or not du.is_contiguous() or D1.data_ptr() != du.data_ptr()
                        or D2.data_ptr() != du.data_ptr() + du.element_size()):
                    du = torch.stack([D1, D2], 1)
                self._du = (pair, du)
            if self.rows is None:
                self.rows = _kernels().pair_rows(self.i, self.k, self.N)
            return _kernels().mutual_field(mu, r, self._du[1], self.i, self.k, self.N, self.count, guard, self.rows)
        with fused(self.fused):
            return mutual_field_real(mu, r, D1, D2, self.ik, self.ki, self.N)

    def plans(self, pos: Optional[Tensor] = None, cell: Optional[Tensor] = None) -> list:
        """The mesh plans: the evaluation's shared ones, or new ones on ``pos`` and ``cell``."""
        if not self.structures:
            return []
        return _mesh_plans(self.ew, "m", self.order, pos, cell)

    def mesh(self, mu: Tensor, plans: Optional[list] = None) -> Optional[Tensor]:
        if not self.structures:
            return None
        with fused(self.fused_mesh):
            return self._mesh(mu, plans)

    def _mesh(self, mu: Tensor, plans: Optional[list]) -> Tensor:
        f = torch.zeros_like(mu)
        coef = self_field_coefficient(self.alpha_p)
        for g, plan in (self.plans() if plans is None else plans):
            idx, b = g["idx"], g["batch"]
            m = mu if g["whole"] else _take(mu, idx)
            _, dphi, _ = plan.potential(None, m, None, self.alpha_p, hessian=False)
            field = -dphi + coef * m
            if self.mesh_self:
                s2 = plan.self_moments(self.alpha_p, "coulomb", max_rank=2)[1]
                field = field + 2.0 * torch.einsum("na,nab->nb", m, s2[b])
            if g["normals"] is not None:
                normal = g["normals"][b]
                moment = scatter_sum((m * normal).sum(-1), b, plan.groups)[b]
                field = field - (4.0 * math.pi / plan.volume[b] * moment)[:, None] * normal
            f = f + field if g["whole"] else f.index_add(0, idx, field)
        return f

    def __call__(self, mu: Tensor, pair, guard: Optional[Tensor] = None) -> Tensor:
        # guard: the scalars of a conjugate-gradient iteration on the kernels, whose kernels skip
        # their work once the solve has converged (the field is then read by no update)
        if mu.shape[0] != self.N:
            return self._paired(mu, pair, guard)
        f = self.real(mu, pair, guard)
        if self.kernels and self.structures and not torch.is_grad_enabled() and self._fusable_mesh():
            # the solve's operator on the kernels: the mesh field accumulated onto the real-space
            # field by the gather kernel itself (one whole group, no slab or self-moment terms)
            g, plan = self.plans()[0]
            return plan.field_into(mu, self.alpha_p, f, float(self_field_coefficient(self.alpha_p)), guard)
        g = self.mesh(mu)
        return f if g is None else f + g

    def _paired(self, mu: Tensor, pair, guard: Optional[Tensor]) -> Tensor:
        """The operator of the paired solve on two sets of dipoles stacked ``(2 N, 3)``: the
        real-space field of each, and the mesh field of both in one pass (:meth:`pairable`)."""
        f = self.real(mu, pair, guard)
        if not self.structures:
            return f
        g, plan = self.plans()[0]
        return plan.paired_field_into(mu, self.alpha_p, f, float(self_field_coefficient(self.alpha_p)), guard)

    def pairable(self) -> bool:
        """Whether two sets of dipoles can be solved together here: on the kernels, with no mesh
        or the one whole group that the gather kernel accumulates into."""
        return self.kernels and (not self.structures or self._fusable_mesh())

    def _fusable_mesh(self) -> bool:
        plans = self.plans()
        if len(plans) != 1 or self.mesh_self:
            return False
        g, plan = plans[0]
        return bool(g["whole"]) and g["normals"] is None and plan.kernels and not plan.cells.requires_grad


@region("amoeba.pcg")
@fusable
def _inverse3(M: Tensor) -> Tensor:
    """Inverses of a batch of 3 x 3 matrices by their adjugate: no host synchronization, so it
    runs inside a captured CUDA graph (the matrices ``I + K`` have determinants of at least one);
    on the fused path one generated kernel instead of some forty elementwise ones."""
    a, b, c = M[..., 0, 0], M[..., 0, 1], M[..., 0, 2]
    d, e, f = M[..., 1, 0], M[..., 1, 1], M[..., 1, 2]
    g, h, i = M[..., 2, 0], M[..., 2, 1], M[..., 2, 2]
    adj = torch.stack([torch.stack([e * i - f * h, c * h - b * i, b * f - c * e], -1),
                       torch.stack([f * g - d * i, a * i - c * g, c * d - a * f], -1),
                       torch.stack([d * h - e * g, b * g - a * h, a * e - b * d], -1)], -2)
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    return adj / det[..., None, None]


def _diagonal_inverse(alpha: Tensor, expol: Optional[Tensor], mask: Tensor):
    """The inverse of the diagonal blocks of the induction matrix as a function of a
    field ``(N, 3)``: ``alpha``, or ``alpha (I + K)^-1`` with exchange polarization."""
    if expol is None:
        diag = alpha[:, None] * mask
        return lambda v: diag * v
    eye = torch.eye(3, dtype=expol.dtype, device=expol.device)
    blocks = alpha[:, None, None] * _inverse3(eye + expol) * mask[:, :, None]
    return lambda v: matvec(blocks, v)


def _pcg(rhs: Tensor, alpha: Tensor, field_of, polar: Tensor, tol: float, max_iter: int,
         x0: Optional[Tensor] = None, precond=None, n_polar: Optional[int] = None,
         fixed: Optional[int] = None, expol: Optional[Tensor] = None,
         kernels: bool = False) -> tuple[Tensor, int, Tensor, Optional[Tensor], Optional[Tensor]]:
    """Preconditioned conjugate gradients for ``(1/alpha - T) mu = rhs`` on the
    polarizable atoms (``((I + K)/alpha - T) mu = rhs`` with the
    exchange-polarization tensors ``expol``), converged when the rms change of
    the dipoles per iteration drops below ``tol`` Debye.

    ``precond`` applies ``M^-1`` (:class:`_Preconditioner`); by default the
    polarizabilities. The solve starts from ``x0`` when its residual is below
    that of a zero start (the right-hand side itself, which costs no operator
    application; the first iteration from zero steps to the direct dipoles
    ``M^-1 rhs``), so a guess costs one operator application and a poor one
    never more iterations. One host synchronization per
    iteration (the convergence test, which also checks that ``r . M^-1 r``
    stays positive; if a preconditioner ever loses definiteness the solve
    restarts from the current dipoles with the diagonal one). Returns the
    dipoles, the number of iterations, whether the guess was used (on the
    device) and, for a solve of ``fixed`` length, whether it converged and
    the number of iterations it needed (both on the device; otherwise
    ``None``). ``n_polar``, the count of polarizable atoms, saves reading it.

    A solve of ``fixed`` length runs that many iterations and reads nothing
    from the device: once converged, the iterate stops changing, so with
    enough iterations it returns the dipoles of the tested solve. The
    iterations it needed are the tested solve's count, or one more than the
    length when it did not converge.
    """
    mask = polar[:, None].to(rhs.dtype)
    n_polar = max(int(polar.sum()), 1) if n_polar is None else n_polar
    if kernels and precond is None:
        # the start in one compiled block on the fused path (the operator on the guess comes first), then
        # the iteration's vector operations in three kernels (xnn.ffnn.models.amoeba_kernels), the field
        # of the direction from the operator as before
        blocks = None if expol is None else _kernels().blocks_inverse(alpha, expol, mask[:, 0])
        xg = None if x0 is None else x0 * mask
        x, r, z, p, rz, used, inv_alpha, flat, diag = _pcg_start(rhs, polar, alpha, xg,
                                                                 None if xg is None else field_of(xg), expol,
                                                                 blocks)
        x, iterations, converged, needed = _pcg_kernels(x, r, z, p, rz, field_of, inv_alpha, expol, flat, diag,
                                                        blocks, tol, max_iter, n_polar, fixed)
        return x, iterations, used, converged, needed
    inv_alpha = torch.where(polar, 1.0 / alpha.clamp_min(1e-30), torch.zeros_like(alpha))[:, None]
    diagonal = _diagonal_inverse(alpha, expol, mask)
    precond = diagonal if precond is None else precond

    def apply(v):
        own = v if expol is None else v + matvec(expol, v)
        return (own * inv_alpha - field_of(v)) * mask

    b = rhs * mask
    if x0 is not None:
        xg = x0 * mask
        rg = b - apply(xg)
        # the guess when its residual is the smaller one, chosen on the device
        used = (rg * rg).sum() < (b * b).sum()
        x, r = torch.where(used, xg, torch.zeros_like(xg)), torch.where(used, rg, b)
    else:
        used = torch.zeros((), dtype=torch.bool, device=rhs.device)
        x, r = torch.zeros_like(b), b
    z = precond(r)
    p = z
    rz = (r * z).sum()
    if fixed is not None:
        x, iterations, converged, needed = _pcg_fixed(x, r, z, p, rz, apply, precond, tol, n_polar, fixed)
        return x, iterations, used, converged, needed
    iterations = 0
    for _ in range(max_iter):
        iterations += 1
        ap = apply(p)
        denom = (p * ap).sum()
        # a vanishing curvature (an exact solution) gives a zero step, which converges
        a = torch.where(denom != 0, rz / torch.where(denom != 0, denom, torch.ones_like(denom)),
                        torch.zeros_like(denom))
        step = a * p
        x = x + step
        r = r - a * ap
        z = precond(r)
        rz_new = (r * z).sum()
        change, curvature = torch.stack([(step * step).sum(), rz_new]).tolist()
        if math.sqrt(change / n_polar) * _DEBYE < tol:
            break
        if curvature <= 0.0 and precond is not diagonal:
            precond = diagonal
            z = diagonal(r)
            p = z
            rz = (r * z).sum()
            continue
        p = z + (rz_new / torch.where(rz != 0, rz, torch.ones_like(rz))) * p
        rz = rz_new
    else:
        warnings.warn(f"the induced dipoles did not converge to {tol} Debye in {max_iter} iterations",
                      stacklevel=2)
    return x, iterations, used, None, None


@fusable
def _pcg_start(rhs: Tensor, polar: Tensor, alpha: Tensor, xg: Optional[Tensor], field: Optional[Tensor],
               expol: Optional[Tensor], blocks: Optional[Tensor]) -> tuple:
    """The start of :func:`_pcg` on the kernels with the diagonal preconditioner (the polarizabilities,
    or the blocks ``alpha (I + K)^-1``): the guess ``xg`` when its residual (from the operator's field
    of it, ``field``) is below the right-hand side's, else zero; the residual, the preconditioned
    residual, the direction, ``r . z`` and whether the guess was taken, then the inverse
    polarizabilities, the mask and the diagonal (``None`` with blocks) as the kernels take them.
    Fresh tensors, which the kernels update in place; one compiled block on the fused path."""
    flat = polar.to(rhs.dtype)
    mask = flat[:, None]
    inv_alpha = torch.where(polar, 1.0 / alpha.clamp_min(1e-30), torch.zeros_like(alpha))
    diag = None if blocks is not None else alpha * flat
    b = rhs * mask
    if xg is not None:
        own = xg if expol is None else xg + matvec(expol, xg)
        rg = b - (own * inv_alpha[:, None] - field) * mask
        used = (rg * rg).sum() < (b * b).sum()
        x, r = torch.where(used, xg, torch.zeros_like(xg)), torch.where(used, rg, b)
    else:
        used = torch.zeros((), dtype=torch.bool, device=rhs.device)
        x, r = torch.zeros_like(b), b
    z = matvec(blocks, r) if blocks is not None else diag[:, None] * r
    return x, r, z, z.clone(), (r * z).sum(), used, inv_alpha, flat, diag


def _pcg_kernels(x, r, z, p, rz, field_of, inv_alpha, expol, mask, diag, blocks, tol: float, max_iter: int,
                 n_polar: int, fixed: Optional[int]):
    """The iterations of :func:`_pcg` on the kernels: the vectors are updated in place and the
    scalars of an iteration alternate between two device buffers, so a solve of fixed length
    reads nothing from the device; the convergence test of a free solve reads its two numbers.
    Returns the dipoles, the iterations and, for a fixed length, whether the solve converged and
    the iterations it needed (on the device)."""
    km = _kernels()
    scal = km.pcg_scalars(rz)
    bound = (tol / _DEBYE) ** 2 * n_polar if fixed is not None else -1.0
    k = 0
    iterations = 0
    for _ in range(max_iter if fixed is None else fixed):
        iterations += 1
        # the kernels of an iteration after convergence return at once (a solve of fixed length)
        ap = km.pcg_apply(p, field_of(p, scal[k]), inv_alpha, expol, mask, scal[k])
        km.pcg_step(x, r, z, p, ap, diag, blocks, scal[k])
        if fixed is None:
            change = float(scal[k, 3])
            if math.sqrt(change / n_polar) * _DEBYE < tol:
                return x, iterations, None, None
        km.pcg_direction(p, z, scal[k], scal[1 - k], bound)
        k = 1 - k
    if fixed is None:
        warnings.warn(f"the induced dipoles did not converge to {tol} Debye in {max_iter} iterations",
                      stacklevel=2)
        return x, iterations, None, None
    return x, iterations, scal[k, 4] != 0.0, scal[k, 5] + 1.0


def _pcg_fixed(x, r, z, p, rz, apply, precond, tol: float, n_polar: int, length: int):
    """The iterations of :func:`_pcg` for a solve of fixed length: every update
    is taken while the solve has not converged, the test on the device. Returns
    the dipoles, the length, whether the solve converged and the iterations it
    needed (on the device)."""
    # the convergence test of _pcg, squared: rms change per atom below tol Debye
    bound = (tol / _DEBYE) ** 2 * n_polar
    done = torch.zeros((), dtype=torch.bool, device=x.device)
    taken = torch.zeros((), dtype=torch.float64, device=x.device)
    for _ in range(length):
        ap = apply(p)
        denom = (p * ap).sum()
        a = torch.where(denom != 0, rz / torch.where(denom != 0, denom, torch.ones_like(denom)),
                        torch.zeros_like(denom))
        step = a * p
        x = torch.where(done, x, x + step)
        r = torch.where(done, r, r - a * ap)
        z = precond(r)
        rz_new = (r * z).sum()
        now = done | ((step * step).sum() < bound)
        p = torch.where(now, p, z + (rz_new / torch.where(rz != 0, rz, torch.ones_like(rz))) * p)
        rz = torch.where(now, rz, rz_new)
        taken = torch.where(now, taken, taken + 1.0)
        done = now
    return x, length, done.clone(), taken + 1.0


class _ImplicitInduction(torch.autograd.Function):
    """``mu = ((I + K)/alpha - T)^-1 rhs`` by conjugate gradients (``K`` the
    exchange-polarization tensors, or ``None``), with the gradient of
    the converged dipoles through the adjoint system (the same solve on the
    incoming gradient): neither the iterations nor a dense matrix are stored.
    The backward pass forms ``lambda^T (dA/dtheta) mu`` explicitly: the
    real-space part is the derivative of ``lambda . T mu`` on every pair with
    respect to its ladders and its vector (``T = D_2 r r^T + D_1 I``), the
    mesh part the position and cell derivatives of
    ``lambda . mesh(mu)`` with both vectors held fixed, taken through the
    evaluation's shared mesh plans. For a second derivative the dipoles and
    the adjoint are recomputed as differentiable solves and the bilinear
    mesh term is re-attached to them on detached copies of the positions, so
    every second derivative through the solve (force training, Hessians) is
    exact.
    """

    @staticmethod
    def forward(ctx, rhs, alpha, expol, r, D1, D2, pos, cell, op):
        with torch.no_grad():
            pair = (r.detach(), D1.detach(), D2.detach())
            fixed = None if op.fixed is None else op.fixed.get(op.phase)
            def field_of(v, guard=None):
                return op(v, pair, guard)

            # the adjoint of the polarization energy is -C/2 times the dipoles of the polar field: its
            # tolerance is theirs times C/2, so that both dipole sets converge alike (as in Tinker)
            tol = op.tol * (0.5 * COULOMB if op.phase == "adjoint" else 1.0)
            ex = None if expol is None else expol.detach()
            if op.paired_rhs is not None and op.phase == "forward":
                # the paired solve: the dipoles of the direct and of the polar field stacked, one set of
                # iterations for both (the rms change of the pair below tol bounds each set's)
                n = int(rhs.shape[0])
                x0 = (None if op.guess is None or op.paired_guess is None
                      else torch.cat([op.guess, op.paired_guess]))
                both, iterations, used, converged, needed = _pcg(
                    torch.cat([rhs, op.paired_rhs]), torch.cat([alpha, alpha]), field_of,
                    torch.cat([op.polar, op.polar]), tol, op.max_iter, x0, None, op.n_polar, fixed,
                    None if ex is None else torch.cat([ex, ex]), kernels=op.kernels)
                mu, op.polar_dipoles = both[:n], both[n:]
            else:
                mu, iterations, used, converged, needed = _pcg(rhs, alpha, field_of, op.polar, tol, op.max_iter,
                                                               op.guess, op.precond, op.n_polar, fixed, ex,
                                                               kernels=op.kernels)
            if converged is not None and op.converged is not None:
                op.converged.logical_and_(converged)
            if needed is not None and op.needed is not None and op.phase in SOLVE_PHASES:
                op.needed[SOLVE_PHASES.index(op.phase)].copy_(needed)
        op.stats[op.phase] = iterations
        if op.collect_stats:
            op.stats[op.phase + "_guess"] = bool(used)
        if op.guess is not None and op.collect_stats:
            # how far the guess was from the solution, rms over the polarizable atoms in Debye
            m = op.polar[:, None].to(mu.dtype)
            op.stats[op.phase + "_guess_error"] = float(
                ((op.guess * m - mu) ** 2).sum() / op.n_polar) ** 0.5 * _DEBYE
        ctx.op = op
        ctx.save_for_backward(mu, rhs, alpha, expol, r, D1, D2, pos, cell)
        return mu

    @staticmethod
    def backward(ctx, grad_mu):
        mu, rhs, alpha, expol, r, D1, D2, pos, cell = ctx.saved_tensors
        op = ctx.op
        mask = op.polar[:, None].to(mu.dtype)
        create_graph = torch.is_grad_enabled()
        guess, phase = op.guess, op.phase
        if create_graph:
            # the differentiable recompute starts from the converged dipoles
            op.guess, op.phase = mu.detach(), "recompute"
            mu = _ImplicitInduction.apply(rhs, alpha, expol, r, D1, D2, pos, cell, op)
        field_p = getattr(op, "polar_field", None)
        shortcut = field_p is not None and not create_graph
        if shortcut and not (grad_mu.is_cuda and torch.cuda.is_current_stream_capturing()):
            # the identity holds when the incoming gradient is the polarization energy's; checked
            # with one synchronization, except inside a CUDA-graph capture (a step plan evaluates
            # the energy and nothing else depends on the dipoles there)
            shortcut = bool(torch.allclose(grad_mu * mask, -0.5 * COULOMB * field_p, rtol=1e-9, atol=1e-14))
        key = (tuple(grad_mu.shape), grad_mu.dtype, str(grad_mu.device), create_graph)
        if shortcut:
            # lam solves A lam = -C/2 E_p: lam = -C/2 times the dipoles of the polar field, which are
            # those of the direct field when the two fields coincide (d equals p), and were solved
            # with them otherwise (the paired solve), no solve here
            paired = op.polar_dipoles is not None
            lam = -0.5 * COULOMB * (op.polar_dipoles if paired else mu)
            op.stats["adjoint"] = 0
            if op.collect_stats:
                op.stats["adjoint_guess"] = False
            if paired:
                # remembered as a solved adjoint is, for the next guess of the paired solve
                last = op.adjoints.get(key)
                if last is None:
                    op.adjoints[key] = lam.detach().clone()
                else:
                    last.copy_(lam.detach())
                if op.adjoint_predictor is not None:
                    op.adjoint_predictor.push(lam.detach())
        else:
            # the adjoint starts from the last adjoint of this shape (molecular dynamics steps,
            # repeated training batches), when that is closer than the direct start
            predictor = op.adjoint_predictor if not create_graph else None
            start = predictor.predict(grad_mu) if predictor is not None else None
            op.guess, op.phase = (start if start is not None else op.adjoints.get(key)), "adjoint"
            lam = _ImplicitInduction.apply(grad_mu * mask, alpha, expol, r, D1, D2, pos, cell, op)
            # updated in place, so a recorded graph refreshes it on replay
            last = op.adjoints.get(key)
            if last is None:
                op.adjoints[key] = lam.detach().clone()
            else:
                last.copy_(lam.detach())
            if predictor is not None:
                predictor.push(lam.detach())
            op.guess, op.phase = guess, phase
        grad_r = grad_D1 = grad_D2 = grad_pos = grad_cell = grad_expol = None
        if any(ctx.needs_input_grad[3:6]) and op.kernels and not create_graph:
            # the kernel products are differentiable once: a second derivative takes the eager ones below
            du = op._du[1] if op._du is not None else torch.stack([D1, D2], 1)
            gr, gdu = _kernels().mutual_backward(mu, lam, r, du, op.i, op.k, *op.select)
            grad_r = gr if ctx.needs_input_grad[3] else None
            grad_D1 = gdu[:, 0] if ctx.needs_input_grad[4] else None
            grad_D2 = gdu[:, 1] if ctx.needs_input_grad[5] else None
        elif any(ctx.needs_input_grad[3:6]):
            # lam . T mu summed over both directions of the pairs: D1 (l . m) + D2 (r . l)(r . m)
            m, l_ = mu[op.ik].view(2, -1, 3), lam[op.ki].view(2, -1, 3)
            rm, rl = (r * m).sum(-1), (r * l_).sum(-1)
            if ctx.needs_input_grad[3]:
                grad_r = D2[:, None] * (rm[..., None] * l_ + rl[..., None] * m).sum(0)
            if ctx.needs_input_grad[4]:
                grad_D1 = (l_ * m).sum((0, 2))
            if ctx.needs_input_grad[5]:
                grad_D2 = (rm * rl).sum(0)
        if op.structures and (pos.requires_grad or cell.requires_grad):
            mu_d, lam_d = mu.detach(), lam.detach()
            with torch.enable_grad():
                # the explicit dependence on the positions and the cell, with the vectors fixed; the
                # shared plans were built from op.pos and op.cell, the very inputs of this function
                inputs = [t for t in (op.pos, op.cell) if t.requires_grad]
                if not create_graph and op.kernels and op._fusable_mesh():
                    # on the kernels the gradient of lam . (mesh field of mu) with respect to the scaled
                    # coordinates comes from the two potentials and one adjoint kernel, then through the
                    # coordinates' graph to the positions (lam is a multiple of mu under the shortcut)
                    _, plan = op.plans()[0]
                    scale = -0.5 * COULOMB if (shortcut and op.polar_dipoles is None) else None
                    gu = plan.dipole_adjoint(mu_d, lam_d, op.alpha_p, scale)
                    grads = list(torch.autograd.grad(plan.u, inputs, gu, retain_graph=True, allow_unused=True))
                else:
                    scalar = (lam_d * op.mesh(mu_d)).sum()
                    grads = list(torch.autograd.grad(scalar, inputs, create_graph=create_graph, retain_graph=True,
                                                     allow_unused=True))
                if create_graph:
                    # the term is bilinear in the adjoint and the dipoles: their (zero-valued)
                    # variations carry the graphs of the solves into the returned gradients. They
                    # are differentiated on detached copies of the positions, as a separate pass,
                    # so that this backward is not entered again through the recomputed solves.
                    pv = pos.detach().requires_grad_(bool(pos.requires_grad))
                    cv = cell.detach().requires_grad_(bool(cell.requires_grad))
                    copies = [t for t in (pv, cv) if t.requires_grad]
                    plans = op.plans(pv, cv)
                    scalar = ((lam_d * op.mesh(mu - mu_d, plans)).sum()
                              + ((lam - lam_d) * op.mesh(mu_d, plans)).sum())
                    more = torch.autograd.grad(scalar, copies, create_graph=True, allow_unused=True)
                    grads = [g if m is None else (m if g is None else g + m) for g, m in zip(grads, more)]
            found = dict(zip([id(t) for t in inputs], grads))
            grad_pos, grad_cell = found.get(id(op.pos)), found.get(id(op.cell))
        grad_alpha = None
        if ctx.needs_input_grad[1]:
            # the diagonal blocks (I + K) / alpha: -lam^T (d/d theta) mu for alpha and for K
            a2 = torch.where(op.polar, alpha * alpha, torch.ones_like(alpha))
            own = mu if expol is None else mu + matvec(expol, mu)
            grad_alpha = torch.where(op.polar, (lam * own).sum(1) / a2, torch.zeros_like(alpha))
        if expol is not None and ctx.needs_input_grad[2]:
            inv = torch.where(op.polar, 1.0 / torch.where(op.polar, alpha, torch.ones_like(alpha)),
                              torch.zeros_like(alpha))
            grad_expol = -inv[:, None, None] * lam[:, :, None] * mu[:, None, :]
        return lam, grad_alpha, grad_expol, grad_r, grad_D1, grad_D2, grad_pos, grad_cell, None


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
           "tortor_spline_tables", "bicubic_interpolate", "ewald_ladder", "multipole_pme", "DipolePredictor",
           "StepPlan", "SOLVE_PHASES"]
