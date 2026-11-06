"""ByteFF and ByteFF-Pol: classical force fields parameterized by a graph transformer.

ByteFF (Zheng et al., Chem. Sci. 16, 2730 (2025),
https://doi.org/10.1039/D4SC06640E) predicts every parameter of an AMBER-form force
field for a molecule from its two-dimensional graph: a local edge-augmented graph
transformer (:class:`~xnn.gnn.models.egt.MolecularGraphTransformer`) reads the atoms
and bonds, and output heads map its states to

* harmonic bond and angle parameters, a four-term Fourier series per proper torsion
  (phases 0 for odd and pi for even multiplicities, zero barriers about linear
  centers) and a twofold improper per trigonal carbon or nitrogen;
* per-atom Lennard-Jones ``sigma`` and ``epsilon`` (Lorentz-Berthelot combination)
  and partial charges built as bond charge increments on the formal charges, so the
  total charge is exact.

ByteFF-Pol (Zheng et al., Nat. Commun. 17, 8276 (2026),
https://doi.org/10.1038/s41467-026-73566-3) keeps the bonded terms and replaces the
nonbonded part by five physically separated terms fitted to an energy decomposition:
Born-Mayer repulsion with a short-range wall, damped C6 dispersion, point-charge
electrostatics, Thole-damped induced dipoles and a charge-transfer term. Atomic C6,
polarizability and van der Waals radius follow from a predicted effective atomic
volume by Tkatchenko-Scheffler scaling of free-atom values.

The network sees no coordinates. The parameters of a bound system are predicted once
per distinct molecule (and cached between evaluations while the weights do not
change); the energy of a conformation is then a classical force field evaluation.
Several independently trained networks (an ensemble) average their parameters.

Conventions and scope
---------------------
* A model is bound to one system (:meth:`ByteFF.bind`): an RDKit molecule of one or
  more fragments, a list of molecules, a SMILES string or an ``ase.Atoms`` (bonds
  perceived). Every structure evaluated must be a conformation of that system.
* Parameters are in kcal/mol, Angstrom, radians and e; energies are returned in eV.
* Without ``cutoff`` every pair of a structure interacts (molecules and clusters,
  no cell). ByteFF also runs with a neighbor-list cutoff and periodic cells
  (truncated or switched, without Ewald sums). ByteFF-Pol solves its dipoles
  directly per structure without a cutoff; with one it evaluates periodic cells:
  the short-range terms truncated at the cutoff with a long-range correction of
  their tails, the charges and induced dipoles as Ewald sums with the reciprocal
  part on a particle mesh, and the dipoles solved iteratively.
"""
from __future__ import annotations

import inspect
import math
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn

from xnn.common.config.options import model_options
from xnn.common.data import AtomicGraph
from xnn.common.data.neighborlist import unique_pairs
from xnn.common.models import InteratomicPotential, register_model
from xnn.common.models.blocks import MLP
from xnn.common.models.compiled import run_compiled
from xnn.common.models.fast import resolve_use_fast
from xnn.common.models.ops import cell_volume, scatter_sum
from xnn.ffnn.models.amoeba import (
    SOLVE_PHASES,
    DipolePredictor,
    StepPlan,
    _ImplicitInduction,
    _kernels,
    _kernels_supported,
    _mesh_plans,
    _MutualField,
    _plan_vectors,
    coulomb_ladder,
    thole_ladder,
)
from xnn.ffnn.models.amoeba_ewald import SQRT_PI, ewald_ladder, fusable, fused
from xnn.ffnn.models.classical_fast import (
    ClassicalFastPath,
    fast_bonded,
    fast_nonbonded,
    fast_pairs14,
    image_shifts,
    lj_table,
    needs_twice,
    wants_fast,
)
from xnn.ffnn.models.geometry import excluded_pairs, geometric_mean, half_edges, pair_image_shifts, pair_vectors
from xnn.ffnn.models.oplslib import KCAL_TO_EV
from xnn.ffnn.models.terms import (
    bond_angle,
    charge_transfer,
    damped_dispersion,
    exp_repulsion,
    fourier_series,
    harmonic,
    lennard_jones,
    polarization_system,
    solve_dipoles,
    torsion_cos,
)
from xnn.gnn.models.egt import MolecularGraphTransformer, class_mean

#: Coulomb constant of the released models, kcal Angstrom / (mol e^2).
COULOMB = 332.0637141491396
_BOHR = 1.0 / 1.88973
_HARTREE = 627.509474

# Free-atom volume (bohr^3), polarizability (bohr^3), C6 (hartree bohr^6) and van der
# Waals radius (bohr) of the Tkatchenko-Scheffler scheme (Tkatchenko & Scheffler, Phys.
# Rev. Lett. 102, 073005 (2009)); lithium carries the helium-like values of Li+ for the
# volume.
_FREE_ATOMS = {
    1: (7.9, 4.5, 6.5, 3.1), 3: (89.0, 1.38, 1.46, 2.65), 5: (47.1, 21.0, 99.5, 3.89),
    6: (35.7, 12.0, 46.6, 3.59), 7: (27.0, 7.4, 24.2, 3.34), 8: (22.7, 5.4, 15.6, 3.19),
    9: (18.6, 3.8, 9.5, 3.04), 11: (114.3, 2.67, 6.38, 2.91), 14: (99.5, 37.0, 305.0, 4.2),
    15: (95.7, 25.0, 185.0, 4.01), 16: (77.0, 19.6, 134.0, 3.86), 17: (66.7, 15.0, 94.6, 3.71),
    19: (219.4, 11.1, 64.3, 3.55), 35: (97.0, 20.0, 162.0, 3.93), 53: (152.2, 35.0, 385.0, 4.17),
}
_N_ELEMENTS = 86


def free_atom_table(column: int) -> list:
    """One free-atom quantity per element (row ``Z - 1``) in Angstrom and kcal/mol:
    0 volume, 1 polarizability, 2 C6, 3 radius."""
    unit = (_BOHR ** 3, _BOHR ** 3, _BOHR ** 6 * _HARTREE, _BOHR)[column]
    return [_FREE_ATOMS.get(z, (0.0,) * 4)[column] * unit for z in range(1, _N_ELEMENTS + 1)]


def _mlp(n_in: int, hidden: int, n_out: int, layers: int, activation: str = "gelu", bias: bool = True) -> MLP:
    return MLP([n_in] + [hidden] * (layers - 1) + [n_out], activation, bias=bias)


def _tail_sums(P: dict, counts: Tensor, cutoff: float, prefactor: float, wall: float, damping: float,
               nodes: int = 64) -> Tensor:
    """The pair-weighted tail integrals of the short-range terms of a periodic system.

    ``sum_{a <= b} c_ab int_rc^inf r^2 U_ab(r) dr`` over the atom classes ``a, b`` (the
    distinct molecules' atoms, ``counts`` copies each), ``c_aa = n_a (n_a + 1) / 2`` and
    ``c_ab = n_a n_b``: the uniform-density estimate of the energy beyond the cutoff
    is ``4 pi N / (N + 1)`` times this over the volume (``N`` atoms). One sum each for
    the repulsion (with its wall), the dispersion and the charge transfer, by
    Gauss-Legendre quadrature in ``x = r_c / r``.
    """
    x, w = np.polynomial.legendre.leggauss(nodes)
    dtype, device = P["eps"].dtype, P["eps"].device
    x = torch.as_tensor(0.5 * (x + 1.0), dtype=dtype, device=device)
    w = torch.as_tensor(0.5 * w, dtype=dtype, device=device) * cutoff ** 3 / x ** 4
    r = (cutoff / x).view(1, 1, -1)

    def pair(k: str, mean: bool = False) -> Tensor:
        v = P[k]
        return (0.5 * (v[:, None] + v[None, :]) if mean else torch.sqrt(v[:, None] * v[None, :])).unsqueeze(-1)

    r0 = pair("r0", mean=True)
    rep = exp_repulsion(r, pair("eps"), pair("lam"), r0, prefactor, wall)
    disp = damped_dispersion(r, pair("c6"), r0, damping)
    ct = charge_transfer(r, pair("ct_eps"), pair("ct_lam"), r0)
    n = counts.to(dtype)
    weight = 0.5 * (n[:, None] * n[None, :] + torch.diag(n))
    return torch.stack([((u * w).sum(-1) * weight).sum() for u in (rep, disp, ct)])


def _fft_size(n: int) -> int:
    """The smallest size of at least ``n`` whose prime factors are at most 7 (OpenMM's GPU platforms round
    their meshes so)."""
    size = n
    smooth = False
    while not smooth:
        m = size
        for f in [2, 3, 5, 7]:
            while m % f == 0:
                m = m // f
        smooth = m == 1
        if not smooth:
            size += 1
    return size


def _pol_local_terms(vec: Tensor, sizes: List[int], bond_k: Tensor, bond_x0: Tensor, angle_k: Tensor, angle_x0: Tensor,
               torsion_k: Tensor, sign: Tensor, improper: Tensor, vdw_scale: Tensor, eps: Tensor, lam: Tensor,
               r0: Tensor, c6: Tensor, b0: Tensor, b1: Tensor, a1: Tensor, t_owner: Tensor, p0: Tensor, p1: Tensor,
                     N: int, prefactor: float, wall: float, damping: float) -> Tensor:
    """The bonded terms and the fixed 1,4 / 1,5 pairs of periodic ByteFF-Pol from their vectors ``vec`` (bonds,
    the two arms of the angles, the three bonds of the torsions, the pairs, in ``sizes`` blocks), as per-atom
    energies ``(N, 6)``: bond, angle, torsion, improper, repulsion, dispersion; one scatter. Compiled on a step
    plan's fast path (:func:`~xnn.ffnn.models.amoeba_ewald.fusable`)."""
    ends = [0]
    for k in sizes:
        ends.append(ends[-1] + k)
    parts = [vec[ends[i]:ends[i + 1]] for i in range(7)]
    vb, vu, vv, t1, t2, t3, vp = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6]
    e_bond = harmonic(torch.sqrt((vb * vb).sum(1)), bond_k, bond_x0)
    e_angle = harmonic(bond_angle(vu, vv), angle_k, angle_x0)
    e_tors = fourier_series(torsion_cos(t1, t2, t3), torsion_k, [1, 2, 3, 4], sign)
    rp = torch.sqrt((vp * vp).sum(1))
    rep = vdw_scale * exp_repulsion(rp, eps, lam, r0, prefactor, wall)
    disp = vdw_scale * damped_dispersion(rp, c6, r0, damping)
    K = 6
    t_col = 2 + improper.long()
    values = torch.cat([0.5 * e_bond, 0.5 * e_bond, e_angle, e_tors, 0.5 * rep, 0.5 * rep, 0.5 * disp, 0.5 * disp])
    index = torch.cat([b0 * K, b1 * K, a1 * K + 1, t_owner * K + t_col, p0 * K + 4, p1 * K + 4, p0 * K + 5,
                       p1 * K + 5])
    return scatter_sum(values, index, N * K).view(N, K)


_pol_local = fusable(_pol_local_terms)


def _pol_scaled_terms(vc: Tensor, ci: Tensor, cj: Tensor, cm: Tensor, cs: Tensor, q: Tensor, damp: Tensor, N: int,
                      ke: float, thole: float) -> Tensor:
    """The corrections of periodic ByteFF-Pol's Ewald sums on the pairs within 1,5 (vectors ``vc``): the charge
    scale less one times the bare Coulomb energy and the field scale less one times the Thole-damped field, as
    ``(N, 4)``: electrostatic energy and field; one scatter per atom of the pair. Compiled on a step plan's fast
    path."""
    r = torch.sqrt((vc * vc).sum(1))
    q_i, q_j = q.index_select(0, ci), q.index_select(0, cj)
    e = 0.5 * ke * cm * q_i * q_j / r
    tc = torch.full_like(r, thole)
    fc = cs * thole_ladder(r, damp.index_select(0, ci), damp.index_select(0, cj), tc, tc, 1)[1]
    at_i = torch.cat([e[:, None], (q_j * fc)[:, None] * vc], 1)
    at_j = torch.cat([e[:, None], -(q_i * fc)[:, None] * vc], 1)
    return scatter_sum(at_i, ci, N) + scatter_sum(at_j, cj, N)


_pol_scaled_pairs = fusable(_pol_scaled_terms)


def _pol_real_space(vn: Tensor, valid: Tensor, beyond: Tensor, src: Tensor, dst: Tensor, atoms: Tensor, N: int,
                    cutoff2: float, alpha: float, thole: float, ke: float, prefactor: float, wall: float,
                    damping: float) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """The neighbor pairs of periodic ByteFF-Pol in one function of per-pair tensors: their per-atom energies
    ``(N, 4)`` (repulsion, dispersion and charge transfer of the pairs beyond 1,5, real-space Ewald
    electrostatics of every pair), the charges' field at the atoms (the Ewald field less the bare one plus
    the Thole-damped one) and the mutual-induction ladders ``D1, D2`` per pair. ``valid`` weighs the pairs
    (zero for a plan's padding, whose distance is set to one), ``beyond`` the short-range ones; ``atoms``
    stacks the per-atom rows ``(N, 8)``: charge, damping volume, eps, lambda, r0, C6, CT eps, CT lambda.
    The reference; a step plan's fast path runs :mod:`~xnn.hybnn.models.byteff_kernels` instead."""
    r2 = torch.where(valid > 0, (vn * vn).sum(1), torch.ones_like(valid))
    within = (r2 < cutoff2).to(vn.dtype) * valid
    w = within * beyond
    r = torch.sqrt(r2)
    a_i, a_k = atoms.index_select(0, src), atoms.index_select(0, dst)
    q_i, q_k = a_i[:, 0], a_k[:, 0]
    mix = torch.sqrt(a_i[:, 2:8] * a_k[:, 2:8])
    r0 = 0.5 * (a_i[:, 4] + a_k[:, 4])
    rep = w * exp_repulsion(r, mix[:, 0], mix[:, 1], r0, prefactor, wall)
    disp = w * damped_dispersion(r, mix[:, 3], r0, damping)
    ct = w * charge_transfer(r, mix[:, 4], mix[:, 5], r0)
    De = ewald_ladder(r, torch.full_like(r, alpha), 2)
    Dc = coulomb_ladder(r, 2)
    th = torch.full_like(r, thole)
    Dt = thole_ladder(r, a_i[:, 1], a_k[:, 1], th, th, 2)
    elec = within * ke * q_i * q_k * De[0]
    f1 = within * (De[1] - Dc[1] + Dt[1])
    half = 0.5 * torch.stack([rep, disp, ct, elec], 1)
    node = scatter_sum(half, src, N) + scatter_sum(half, dst, N)
    field = scatter_sum((q_k * f1)[:, None] * vn, src, N) - scatter_sum((q_i * f1)[:, None] * vn, dst, N)
    return node, field, f1, within * (De[2] - Dc[2] + Dt[2])


def _clamp(x: Tensor, lo: float, hi: float) -> Tensor:
    """``x`` clamped to ``[lo, hi]`` with the gradient of the identity."""
    return x + (torch.clamp(x, lo, hi) - x).detach()


# Atoms per bonded term, and the parameters each head predicts: name -> (width, std, mean,
# lower, upper); a raw output y becomes clamp(std t tanh(y / ... ) + mean).
_TERM_ATOMS = {"bond": 2, "angle": 3, "proper": 4, "improper": 2}
_HARMONIC = {
    "bond": {"bond_k": (1, 200.0, 700.0, 80.0, 4000.0), "bond_r0": (1, 0.19, 1.3, 0.5, 5.0)},
    "angle": {"angle_k": (1, 60.0, 130.0, 40.0, 1000.0), "angle_d0": (1, 10.0, 120.0, 20.0, 180.0)},
    "proper": {"proper_k": (4, 1.2, 0.4, -20.0, 20.0)},
    "improper": {"improper_k": (1, 3.7, 4.3, 0.0, 20.0)},
}
_TWO_WELL = {
    "bond": {"bond_k1": (1, 50.0, 1000.0, 0.0, 4000.0), "bond_k2": (1, 25.0, 500.0, 0.0, 4000.0)},
    "angle": {"angle_k1": (1, 5.0, 100.0, 0.0, 400.0), "angle_k2": (1, 5.0, 100.0, 0.0, 400.0)},
    "proper": {"proper_k": (4, 1.2, 0.4, -20.0, 20.0)},
    "improper": {"improper_k": (1, 3.7, 4.3, 0.0, 40.0)},
}
#: Anchors of the two-well form: bond lengths (Angstrom) and angles (radians).
TWO_WELL_BONDS = (0.5, 4.0)
TWO_WELL_ANGLES = (0.25 * math.pi, 1.05 * math.pi)


class BondedHead(nn.Module):
    """Bond, angle, torsion and improper parameters from atom and edge states.

    The states along a term are concatenated in both directions (impropers: center,
    edge and one neighbor, for each neighbor) and summed through one MLP, so the
    parameters do not depend on the order the term is written in.

    Parameters
    ----------
    node_dim, edge_dim : int
        Widths of the atom and edge states.
    pre : tuple of int
        ``(hidden, out, layers)`` of the pooling MLPs (and of the MLP after pooling).
    out : tuple of int
        ``(hidden, layers)`` of the per-parameter output MLPs.
    tanh_output : float
        Bound of the raw outputs (``t tanh(y)``); 0 leaves them unbounded.
    form : str
        ``"harmonic"`` (force constant and equilibrium value) or ``"two_well"``
        (two force constants about the fixed anchors, an equivalent harmonic term).
    activation : str
        Activation of the MLPs.
    """

    def __init__(self, node_dim: int, edge_dim: int, pre=(256, 256, 3), out=(256, 4), tanh_output: float = 20.0,
                 form: str = "harmonic", activation: str = "gelu"):
        super().__init__()
        if form not in ("harmonic", "two_well"):
            raise ValueError(f"form must be 'harmonic' or 'two_well', got {form!r}")
        self.form, self.tanh_output = form, float(tanh_output)
        self.table = _HARMONIC if form == "harmonic" else _TWO_WELL
        self.pool = nn.ModuleDict({t: _mlp(a * node_dim + (a - 1) * edge_dim, pre[0], pre[1], pre[2], activation)
                                   for t, a in _TERM_ATOMS.items()})
        self.post = nn.ModuleDict({t: _mlp(pre[1], pre[0], pre[1], pre[2], activation) for t in _TERM_ATOMS})
        self.out = nn.ModuleDict({p: _mlp(pre[1], out[0], spec[0], out[1], activation)
                                  for t in _TERM_ATOMS for p, spec in self.table[t].items()})

    def forward(self, x: Tensor, e: Tensor, terms: dict) -> dict:
        """Raw clamped parameters ``{name: (M, width)}`` of the terms ``terms[kind] = (atoms, edges)``."""
        params = {}
        for kind in _TERM_ATOMS:
            atoms, edges = terms[kind]
            if kind == "improper":
                parts = [torch.cat([x[atoms[:, 0]], e[edges[:, i]], x[atoms[:, i + 1]]], -1) for i in range(3)]
            else:
                seq = []
                for i in range(atoms.shape[1]):
                    seq.append(x[atoms[:, i]])
                    if i < edges.shape[1]:
                        seq.append(e[edges[:, i]])
                parts = [torch.cat(seq, -1), torch.cat(seq[::-1], -1)]
            # every orientation through the pooling MLP in one batch, then summed in order
            m = atoms.shape[0]
            pooled = self.pool[kind](torch.cat(parts, 0))
            y = pooled[:m]
            for k in range(1, len(parts)):
                y = y + pooled[k * m:(k + 1) * m]
            y = self.post[kind](y)
            for name, (_, std, mean, lo, hi) in self.table[kind].items():
                d = self.out[name](y)
                if self.tanh_output > 0.0:
                    d = self.tanh_output * torch.tanh(d)
                params[name] = _clamp(d * std + mean, lo, hi)
        return params


class _BondChargeIncrements(nn.Module):
    """Charges as formal charges (averaged over equivalent atoms) plus antisymmetric
    bond charge increments."""

    def __init__(self, node_dim: int, edge_dim: int, pre, out, bound: float, activation: str):
        super().__init__()
        self.bound = float(bound)
        self.pre = _mlp(2 * node_dim + edge_dim, pre[0], pre[1], pre[2], activation)
        self.out = _mlp(pre[1], out[0], 1, out[1], "tanh", bias=False)

    def forward(self, x: Tensor, e: Tensor, bonds: Tensor, bond_edges: Tensor, formal: Tensor,
                atom_class: Tensor) -> Tensor:
        i, j, b = bonds[:, 0], bonds[:, 1], bond_edges
        m = i.shape[0]
        both = self.pre(torch.cat([torch.cat([x[i], e[b], x[j]], -1), torch.cat([x[j], e[b], x[i]], -1)], 0))
        fwd, bwd = both[:m], both[m:]
        inc = self.bound * torch.tanh(self.out(fwd - bwd).squeeze(-1))
        q = class_mean(formal.to(inc.dtype).unsqueeze(-1), atom_class).squeeze(-1)
        return q + scatter_sum(inc, i, q.shape[0]) - scatter_sum(inc, j, q.shape[0])


class LennardJonesChargeHead(nn.Module):
    """Per-atom Lennard-Jones ``sigma``, ``epsilon`` and partial charges (ByteFF).

    Parameters
    ----------
    node_dim, edge_dim : int
        Widths of the atom and edge states.
    pre : tuple of int
        ``(hidden, out, layers)`` of the charge-increment MLP over bonds.
    out : tuple of int
        ``(hidden, layers)`` of the output MLPs.
    tanh_output : float
        ``sigma`` and ``epsilon`` are ``t (tanh(y) + 1)``; also the bound of the
        bond charge increments.
    activation : str
        Activation of the MLPs.
    """

    def __init__(self, node_dim: int, edge_dim: int, pre=(512, 512, 4), out=(256, 3), tanh_output: float = 20.0,
                 activation: str = "gelu"):
        super().__init__()
        self.t = float(tanh_output)
        self.sigma = _mlp(node_dim, out[0], 1, out[1], activation)
        self.epsilon = _mlp(node_dim, out[0], 1, out[1], activation)
        self.charge = _BondChargeIncrements(node_dim, edge_dim, pre, out, tanh_output, activation)

    def forward(self, x: Tensor, e: Tensor, g: dict) -> dict:
        """``{"charge", "sigma", "epsilon"}``, each ``(n,)``."""
        return {"charge": self.charge(x, e, g["bonds"], g["bond_edges"], g["formal"], g["atom_class"]),
                "sigma": self.t * torch.tanh(self.sigma(x).squeeze(-1)) + self.t,
                "epsilon": self.t * torch.tanh(self.epsilon(x).squeeze(-1)) + self.t}


class PolarizableHead(nn.Module):
    """Charges and the exp-6, polarization and charge-transfer parameters (ByteFF-Pol).

    The effective atomic volume ``V`` scales the free-atom values:
    ``C6 = C6_free (V/V_free)^2``, ``alpha = alpha_free (V/V_free)^(4/3)`` and
    ``R = R_free (V/V_free)^(4/21)``; the free-atom tables are trainable.

    Parameters
    ----------
    node_dim, edge_dim : int
        Widths of the atom and edge states.
    pre, out : tuple of int
        MLP shapes as in :class:`LennardJonesChargeHead`.
    charge_bound : float
        Bound of the bond charge increments.
    lithium_volume : float
        Fixed effective volume of lithium (Angstrom^3).
    lithium_alpha : float or None
        Fixed polarizability of lithium, or ``None``.
    lithium_damping : float
        Lower bound of lithium's Thole damping volume.
    damping_floor : float
        Lower bound of every damping volume.
    lambda_scale : float
        Scale of the repulsion exponents.
    c6_scale : float
        Scale at which the free-atom C6 table is stored.
    activation : str
        Activation of the MLPs.
    """

    def __init__(self, node_dim: int, edge_dim: int, pre=(256, 256, 3), out=(256, 3), charge_bound: float = 4.0,
                 lithium_volume: float = 13.0, lithium_alpha: Optional[float] = 1.0e-6,
                 lithium_damping: float = 0.2, damping_floor: float = 1.0e-6, lambda_scale: float = 5.0,
                 c6_scale: float = 1000.0, activation: str = "gelu"):
        super().__init__()
        self.lithium_volume, self.lithium_alpha = float(lithium_volume), lithium_alpha
        self.lithium_damping, self.damping_floor = float(lithium_damping), float(damping_floor)
        self.lambda_scale, self.c6_scale = float(lambda_scale), float(c6_scale)
        self.charge = _BondChargeIncrements(node_dim, edge_dim, pre, out, charge_bound, activation)
        self.volume = _mlp(node_dim, out[0], 1, out[1], activation)
        for name in ("lam", "eps", "ct_eps", "ct_lam"):
            setattr(self, name, _mlp(node_dim, out[0], 1, out[1], activation))
        self.register_buffer("volume_ref", torch.tensor(free_atom_table(0)), persistent=False)
        self.v_free = nn.Parameter(torch.tensor(free_atom_table(0)))
        self.alpha_free = nn.Parameter(torch.tensor(free_atom_table(1)))
        self.c6_free = nn.Parameter(torch.tensor(free_atom_table(2)) / self.c6_scale)
        self.r_free = nn.Parameter(torch.tensor(free_atom_table(3)))

    def forward(self, x: Tensor, e: Tensor, g: dict) -> dict:
        """Per-atom ``charge``, ``volume``, ``c6``, ``r0``, ``alpha``, ``damp``, ``lam``,
        ``eps``, ``ct_eps`` and ``ct_lam``, each ``(n,)``."""
        z = g["element"]
        li = z == 2
        q = self.charge(x, e, g["bonds"], g["bond_edges"], g["formal"], g["atom_class"])
        volume = torch.exp(self.volume(x).squeeze(-1)) * self.volume_ref.to(x.dtype)[z]
        volume = torch.where(li, torch.full_like(volume, self.lithium_volume), volume)
        ratio = volume / self.v_free[z]
        alpha = torch.clamp(self.alpha_free[z] * ratio ** (4.0 / 3.0), min=1.0e-6)
        if self.lithium_alpha is not None:
            alpha = torch.where(li, torch.full_like(alpha, float(self.lithium_alpha)), alpha)
        damp = alpha
        if self.lithium_damping > 0.0:
            damp = torch.where(li, torch.clamp(alpha, min=self.lithium_damping), alpha)
        return {"charge": q, "volume": volume,
                "c6": torch.clamp(self.c6_free[z] * ratio ** 2 * self.c6_scale, min=1.0e-6),
                "r0": self.r_free[z] * ratio ** (4.0 / 21.0), "alpha": alpha,
                "damp": torch.clamp(damp, min=self.damping_floor),
                "lam": torch.exp(self.lam(x).squeeze(-1)) * self.lambda_scale,
                "eps": torch.exp(self.eps(x).squeeze(-1)),
                "ct_eps": torch.exp(self.ct_eps(x).squeeze(-1)),
                "ct_lam": torch.exp(self.ct_lam(x).squeeze(-1))}


class _Member(nn.Module):
    """One network of an ensemble: graph transformer, bonded head, nonbonded head."""

    def __init__(self, graph: dict, bonded: dict, nonbonded: dict, polarizable: bool):
        super().__init__()
        self.graph = MolecularGraphTransformer(**graph)
        nd, ed = self.graph.node_dim, self.graph.edge_dim
        self.bonded = BondedHead(nd, ed, **bonded)
        self.nonbonded = (PolarizableHead if polarizable else LennardJonesChargeHead)(nd, ed, **nonbonded)

    def forward(self, g: dict) -> dict:
        x, e = self.graph(g["features"], g["edge_features"], g["edge_index"], g["atom_class"], g["edge_class"])
        out = self.bonded(x, e, g["terms"])
        out.update(self.nonbonded(x, e, g))
        return out


def _molecules(structure, charge: int = 0) -> list:
    """The molecules of a system as RDKit molecules with the global index of each atom."""
    from xnn.ffnn.common.typing import to_rdkit
    from xnn.gnn.featurizers.molgraph import split_molecules
    items = list(structure) if isinstance(structure, (list, tuple)) else [structure]
    out, offset = [], 0
    for item in items:
        mol = to_rdkit(item, charge=charge if len(items) == 1 else 0)
        for frag, atoms in split_molecules(mol):
            out.append((frag, [offset + a for a in atoms]))
        offset += mol.GetNumAtoms()
    return out


class _GraphParameterizedFF(InteratomicPotential):
    """A classical force field whose parameters a graph network predicts for a bound
    system (the shared part of :class:`ByteFF` and :class:`ByteFFPol`)."""

    polarizable = False
    foundation_default = "byteff-26"
    _pair_names: tuple = ()
    _atom_keys: tuple = ("charge",)

    def __init__(self, structure=None, *, graph: Optional[dict] = None, bonded: Optional[dict] = None,
                 nonbonded: Optional[dict] = None, members: int = 1,
                 supported_elements: Sequence[int] = (), cutoff: Optional[float] = None,
                 switch_width: float = 0.0, coulomb_constant: float = COULOMB, charge: int = 0):
        super().__init__()
        self.graph_options = dict(graph or {})
        self.bonded_options = dict(bonded or {})
        self.nonbonded_options = dict(nonbonded or {})
        self.members = nn.ModuleList(_Member(self.graph_options, self.bonded_options, self.nonbonded_options,
                                             self.polarizable) for _ in range(int(members)))
        self.supported_elements = tuple(int(z) for z in supported_elements)
        self.coulomb_constant = float(coulomb_constant)
        self.all_pairs = cutoff is None
        self.cutoff = 0.1 if cutoff is None else float(cutoff)
        self.switch_width = float(switch_width)
        if not self.all_pairs and not 0.0 <= self.switch_width < self.cutoff:
            raise ValueError("switch_width must satisfy 0 <= switch_width < cutoff")
        self.node_feature_dim = 1
        self._cache: Optional[tuple] = None
        self._block_calls: dict = {}
        self._compile_failed: set = set()
        self._compile_checked: set = set()
        self.bound = False
        if structure is not None:
            self.bind(structure, charge=charge)

    def bind(self, structure, charge: int = 0, reference_positions=None) -> "_GraphParameterizedFF":
        """Bind the system every evaluated structure is a conformation of.

        Parameters
        ----------
        structure : rdkit.Chem.Mol, str, ase.Atoms or list
            An RDKit molecule with explicit hydrogens (fragments are the
            molecules), a SMILES, an ``ase.Atoms`` (bonds and bond orders
            perceived), or a list of these concatenated in order.
        charge : int, optional
            Total charge, for bond perception from an ``ase.Atoms``.
        reference_positions : array_like, optional
            Positions ``(n, 3)`` of the system (an optimized geometry) giving the
            equilibrium lengths and angles of the bonds and angles at hypervalent
            phosphorus and sulfur, whose force constants are fixed (1000
            kcal/mol/Angstrom^2 and 278.44 kcal/mol/rad^2) instead of predicted.
            Taken from an ``ase.Atoms`` by default; without it those equilibrium
            values are zero.

        Returns
        -------
        _GraphParameterizedFF
            The model.
        """
        from xnn.gnn.featurizers.molgraph import graph_key, molecular_graph
        if reference_positions is None and hasattr(structure, "get_positions"):
            reference_positions = structure.get_positions()
        mols = _molecules(structure, charge)
        species, keys, graphs = [], {}, []
        for mol, atoms in mols:
            key = graph_key(mol)
            if key not in keys:
                keys[key] = len(graphs)
                graphs.append(molecular_graph(mol))
            species.append((keys[key], atoms))
        self._bind_graphs(graphs, species)
        self._bind_hypervalent(graphs, species, reference_positions)
        return self

    def _bind_hypervalent(self, graphs: list, species: list, positions) -> None:
        """Masks of the bonded terms at hypervalent atoms and their reference values."""
        hyper = torch.as_tensor(np.concatenate([g.hypervalent for g in graphs]), device=self._u_features.device)
        masks = {"bond": hyper[self._t_bond_atoms].any(1), "angle": hyper[self._t_angle_atoms[:, 1]],
                 "proper": hyper[self._t_proper_atoms].any(1), "improper": hyper[self._t_improper_atoms].any(1)}
        for kind, m in masks.items():
            self.register_buffer(f"_t_{kind}_hyper", m, persistent=False)
        ref = {k: torch.full((int(masks[k].shape[0]),), float("nan"), dtype=torch.float64) for k in ("bond", "angle")}
        if bool(hyper.any()) and positions is not None:
            pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
            node_off = np.cumsum([0] + [g.n_atoms for g in graphs])
            to_global = np.zeros(int(node_off[-1]), dtype=np.int64)
            seen = set()
            for s, atoms in species:
                if s not in seen:
                    seen.add(s)
                    to_global[node_off[s]:node_off[s + 1]] = atoms
            x = torch.as_tensor(pos[to_global])
            b = self._t_bond_atoms.cpu()
            ref["bond"] = torch.linalg.norm(x[b[:, 1]] - x[b[:, 0]], dim=1)
            a = self._t_angle_atoms.cpu()
            ref["angle"] = bond_angle(x[a[:, 0]] - x[a[:, 1]], x[a[:, 2]] - x[a[:, 1]])
        elif bool(hyper.any()):
            import warnings
            warnings.warn("hypervalent P/S centers: their bonds and angles have no predicted equilibrium values; "
                          "pass reference_positions (an optimized geometry) to bind()", stacklevel=3)
        for k, v in ref.items():
            v = torch.where(masks[k].cpu(), v, torch.full_like(v, float("nan")))
            self.register_buffer(f"_t_{k}_x0", v.to(self._u_features.device), persistent=False)
        self._cache = None

    @property
    def atomic_numbers(self) -> Tensor:
        """Atomic numbers ``(n,)`` of the bound system, in its atom order."""
        return self._z

    def term_atoms(self, kind: str) -> Tensor:
        """The atoms of the rows of :meth:`predicted_parameters` of one kind of term.

        Parameters
        ----------
        kind : str
            ``"bond"``, ``"angle"``, ``"proper"`` or ``"improper"``.

        Returns
        -------
        Tensor
            ``(M, k)`` atom indices over the distinct molecules' atoms (the bound
            system's own indices when it is one molecule).
        """
        return getattr(self, f"_t_{kind}_atoms")

    @classmethod
    def from_foundation(cls, name: Optional[str] = None, **kwargs):
        """Load released weights (see :func:`~xnn.hybnn.models.byteff_foundation.from_foundation`).

        Parameters
        ----------
        name : str, optional
            The released model; by default this class's (ByteFF-26 or ByteFF-Pol-25).
        **kwargs
            Options of :func:`~xnn.hybnn.models.byteff_foundation.from_foundation`.

        Returns
        -------
        ByteFF or ByteFFPol
            The model in evaluation mode.
        """
        from .byteff_foundation import from_foundation
        return from_foundation(name or cls.foundation_default, **kwargs)

    def _bind_graphs(self, graphs: list, species: list) -> None:
        cat = np.concatenate
        node_off = np.cumsum([0] + [g.n_atoms for g in graphs])
        edge_off = np.cumsum([0] + [2 * g.bonds.shape[0] for g in graphs])
        names = ("bonds", "angles", "propers", "impropers")
        term_off = {k: np.cumsum([0] + [getattr(g, k).shape[0] for g in graphs]) for k in names}
        u = {
            "features": cat([np.stack([g.atomic_numbers - 1, g.degree, g.formal_charges, g.ring_degree,
                                       g.min_ring], 1) for g in graphs]),
            "edge_features": cat([np.repeat(np.stack([g.bond_in_ring, g.bond_order], 1), 2, axis=0)
                                  for g in graphs]),
            "edge_index": cat([np.stack([np.stack([g.bonds[:, 0], g.bonds[:, 1]], 1).reshape(-1),
                                         np.stack([g.bonds[:, 1], g.bonds[:, 0]], 1).reshape(-1)]) + o
                               for g, o in zip(graphs, node_off)], 1),
            "atom_class": cat([g.atom_class + o for g, o in zip(graphs, node_off)]),
            "edge_class": cat([g.edge_class + o for g, o in zip(graphs, edge_off)]),
            "formal": cat([g.formal_charges for g in graphs]),
            "bonds": cat([g.bonds + o for g, o in zip(graphs, node_off)]),
            "bond_edges": cat([2 * np.arange(g.bonds.shape[0]) + o for g, o in zip(graphs, edge_off)]),
            "linear": cat([g.proper_linear for g in graphs]),
        }
        u["element"] = u["features"][:, 0]
        edges = {"bond": None, "angle": "angle_edges", "proper": "proper_edges", "improper": "improper_edges"}
        terms = {}
        for kind, name in zip(edges, names):
            atoms = cat([getattr(g, name) + o for g, o in zip(graphs, node_off)])
            if kind == "bond":
                ed = u["bond_edges"].reshape(-1, 1)
            else:
                ed = cat([getattr(g, edges[kind]) + o for g, o in zip(graphs, edge_off)])
            terms[kind] = (atoms, ed)
        # the system: global atom indices of every term, and the unique row it takes its parameters from
        sys_atom, sys_terms, sys_src = [], {k: [] for k in names}, {k: [] for k in names}
        sys_pairs = {k: [] for k in ("pairs12", "pairs13", "pairs14", "pairs15")}
        mol_of = []
        for m, (s, atoms) in enumerate(species):
            g, idx = graphs[s], np.asarray(atoms, dtype=np.int64)
            sys_atom.append(np.arange(g.n_atoms) + node_off[s])
            mol_of.append(np.full(g.n_atoms, m))
            for k in names:
                # torsions about linear centers have no barrier and an undefined angle at collinearity
                keep = ~g.proper_linear if k == "propers" else np.ones(getattr(g, k).shape[0], dtype=bool)
                sys_terms[k].append(idx[getattr(g, k)][keep])
                sys_src[k].append((np.arange(getattr(g, k).shape[0]) + term_off[k][s])[keep])
            for k in sys_pairs:
                sys_pairs[k].append(idx[getattr(g, k)])
        order = np.argsort(cat([np.asarray(a) for _, a in species]))
        n = order.shape[0]
        self.n_atoms = n
        dev = next(self.parameters()).device

        def buf(name, value, dtype=torch.long):
            self.register_buffer(name, torch.as_tensor(np.asarray(value), dtype=dtype, device=dev), persistent=False)

        for k, v in u.items():
            buf("_u_" + k, v, torch.bool if k == "linear" else torch.long)
        for kind, (atoms, ed) in terms.items():
            buf(f"_t_{kind}_atoms", atoms.reshape(-1, _width(kind)))
            buf(f"_t_{kind}_edges", ed.reshape(-1, 3 if kind == "improper" else _width(kind) - 1))
        buf("_atom_src", cat(sys_atom)[order])
        buf("_z", (u["features"][:, 0] + 1)[cat(sys_atom)][order])
        for k in names:
            buf(f"_{k}", cat(sys_terms[k]).reshape(-1, _width(k[:-1])).T)
            buf(f"_{k}_src", cat(sys_src[k]))
        for k, v in sys_pairs.items():
            buf("_" + k, cat(v).reshape(-1, 2).T)
        buf("_molecule", cat(mol_of)[order])
        self._bind_pairs(n)
        self.species = sorted({int(z) for z in self._z.tolist()})
        self.bound = True
        self._cache = None

    def _bind_pairs(self, n: int) -> None:
        """The pair lists of the nonbonded terms: the 1,4 pairs, then (with every pair interacting)
        the pairs beyond 1,4 of a structure; and the excluded pairs of a neighbor list."""
        excl = torch.cat([self._pairs12, self._pairs13, self._pairs14], 1)
        self.register_buffer("_excluded", excl, persistent=False)
        pairs, is15 = self._pairs14, torch.zeros(self._pairs14.shape[1], dtype=torch.bool, device=excl.device)
        if self.all_pairs:
            i, j = torch.triu_indices(n, n, 1, device=excl.device)
            code = torch.zeros(n, n, dtype=torch.int8, device=excl.device)
            for c, pp in ((1, self._pairs12), (1, self._pairs13), (2, self._pairs14), (3, self._pairs15)):
                code[pp[0], pp[1]] = c
            cij = code[i, j]
            far = (cij != 1) & (cij != 2)
            pairs = torch.cat([pairs, torch.stack([i[far], j[far]])], 1)
            is15 = torch.cat([is15, cij[far] == 3])
        self.register_buffer("_pair_list", pairs, persistent=False)
        self.register_buffer("_pair_is15", is15, persistent=False)
        self._plans, self._row_cache = {}, {}

    def _unique_graph(self) -> dict:
        g = {k[3:]: getattr(self, k) for k in ("_u_features", "_u_edge_features", "_u_edge_index", "_u_atom_class",
                                               "_u_edge_class", "_u_formal", "_u_bonds", "_u_bond_edges",
                                               "_u_element")}
        g["terms"] = {k: (getattr(self, f"_t_{k}_atoms"), getattr(self, f"_t_{k}_edges")) for k in _TERM_ATOMS}
        return g

    def _patch(self, p: dict) -> dict:
        """Fixed force constants and zero equilibrium values and barriers for the terms at
        hypervalent atoms (their predictions are not used)."""
        out = dict(p)
        for kind, k_fixed in (("bond", 1000.0), ("angle", 278.44)):
            m = getattr(self, f"_t_{kind}_hyper").unsqueeze(-1)
            for name in (f"{kind}_k", f"{kind}_k1"):
                if name in out:
                    out[name] = torch.where(m, torch.full_like(out[name], k_fixed), out[name])
            for name in (f"{kind}_r0" if kind == "bond" else "angle_d0", f"{kind}_k2"):
                if name in out:
                    out[name] = torch.where(m, torch.zeros_like(out[name]), out[name])
        for kind in ("proper", "improper"):
            m = getattr(self, f"_t_{kind}_hyper").unsqueeze(-1)
            out[f"{kind}_k"] = torch.where(m, torch.zeros_like(out[f"{kind}_k"]), out[f"{kind}_k"])
        return out

    def _derive(self, p: dict) -> dict:
        """Energy parameters from the averaged head outputs (unique-molecule rows)."""
        out = {k: v for k, v in p.items() if not k.startswith(("bond_", "angle_"))}
        if "bond_k1" in p:
            for kind, (b1, b2) in (("bond", TWO_WELL_BONDS), ("angle", TWO_WELL_ANGLES)):
                k1, k2 = p[f"{kind}_k1"].squeeze(-1), p[f"{kind}_k2"].squeeze(-1)
                out[f"{kind}_k"] = k1 + k2
                out[f"{kind}_x0"] = (k1 * b1 + k2 * b2) / (k1 + k2)
        else:
            out["bond_k"], out["bond_x0"] = p["bond_k"].squeeze(-1), p["bond_r0"].squeeze(-1)
            out["angle_k"], out["angle_x0"] = p["angle_k"].squeeze(-1), torch.deg2rad(p["angle_d0"].squeeze(-1))
        for kind in ("bond", "angle"):
            ref = getattr(self, f"_t_{kind}_x0").to(out[f"{kind}_x0"].dtype)
            out[f"{kind}_x0"] = torch.where(torch.isnan(ref), out[f"{kind}_x0"], ref)
        out["proper_k"] = p["proper_k"] * (~self._u_linear).to(p["proper_k"].dtype).unsqueeze(-1)
        out["improper_k"] = p["improper_k"]
        return out

    def predicted_parameters(self, per_member: bool = False) -> dict:
        """The force-field parameters of the bound system's distinct molecules.

        Parameters
        ----------
        per_member : bool, optional
            Return the raw head outputs of every ensemble member stacked on a
            leading axis instead of the averaged energy parameters.

        Returns
        -------
        dict of str to Tensor
            Per-atom rows over the distinct molecules' atoms and per-term rows over
            their terms, in kcal/mol, Angstrom, radians and e.
        """
        if not self.bound:
            raise RuntimeError("bind a system first (model.bind(structure))")
        g = self._unique_graph()
        outs = [self._patch(m(g)) for m in self.members]
        if per_member:
            return {k: torch.stack([o[k] for o in outs]) for k in outs[0]}
        mean = {k: torch.stack([o[k] for o in outs]).mean(0) for k in outs[0]}
        return self._derive(mean)

    def _parameters_cached(self) -> dict:
        """The system's energy parameters, recomputed when the weights change (and
        always while training)."""
        if self.training:
            self._cache = None
            return self.predicted_parameters()
        from xnn.common.models import compiled
        cached = self.__dict__.get("_param_list")
        if cached is None or cached[0] != compiled._registrations:
            # kept until anything registers a parameter or module (walking the tree costs more than the check)
            cached = self.__dict__["_param_list"] = (compiled._registrations, tuple(self.parameters()))
        params = cached[1]
        key = (tuple(p._version for p in params), tuple(p.data_ptr() for p in params), params[0].dtype,
               params[0].device)
        if self._cache is None or self._cache[0] != key:
            with torch.no_grad():
                self._cache = (key, self.predicted_parameters())
            self._row_cache = {}
        return self._cache[1]

    def index_planner(self):
        """The planner of a captured evaluation (:class:`~xnn.common.deploy.GraphedEvaluation`): without a
        cutoff the batch of the bound system fixes every shape, so it is replayed unpadded; ``None`` with a
        cutoff (periodic cells run on step plans instead, :class:`~xnn.common.deploy.StepGraph`)."""
        from xnn.common.train.graphed import FixedShapePlanner
        return FixedShapePlanner() if self.all_pairs else None

    def _check(self, data: AtomicGraph) -> int:
        if not self.bound:
            raise RuntimeError("bind a system first (model.bind(structure))")
        n, B = self.n_atoms, data.num_graphs
        if not bool((data.n_atoms == n).all()):
            raise ValueError(f"every structure must have the bound system's {n} atoms "
                             f"(got {data.n_atoms.tolist()})")
        if not bool((data.atomic_numbers.view(B, n) == self._z).all()):
            raise ValueError("atomic numbers do not match the bound system's element sequence")
        if not self.training and self.supported_elements:
            bad = sorted(set(self.species) - set(self.supported_elements))
            if bad:
                raise ValueError(f"elements {bad} are not supported by this model "
                                 f"(supported: {list(self.supported_elements)})")
        if self.all_pairs and data.cell is not None:
            raise ValueError("this model evaluates every pair of a structure and takes no cell; "
                             "set a cutoff for periodic systems")
        return B

    def _plan(self, B: int, device) -> dict:
        """The index layout of a batch of ``B`` copies of the system (cached): the terms of each
        structure in one block, the vectors of every term gathered at once, and the atom each term's
        energy is assigned to (pair energies half to each atom)."""
        key = (B, str(device))
        if key in self._plans:
            return self._plans[key]
        off = torch.arange(B, device=device) * self.n_atoms

        def ex(idx: Tensor) -> Tensor:
            return (idx.unsqueeze(1) + off.view(1, B, 1)).reshape(idx.shape[0], -1)

        tors = torch.cat([self._propers, self._impropers], 1)
        b, a, t, p = ex(self._bonds), ex(self._angles), ex(tors), ex(self._pair_list)
        n14 = self._pairs14.shape[1]
        improper = (torch.arange(tors.shape[1], device=device) >= self._propers.shape[1]).repeat(B)
        owners = [b[0], b[1], a[1], torch.where(improper, t[0], t[1])] + [p[k] for _ in self._pair_names
                                                                           for k in (0, 1)]
        plan = {"bonds": b, "angles": a, "propers": ex(self._propers), "impropers": ex(self._impropers),
                "pairs": p, "improper": improper, "t_owner": torch.where(improper, t[0], t[1]),
                "is14": (torch.arange(self._pair_list.shape[1], device=device) < n14).repeat(B),
                "pairs14": ex(self._pair_list[:, :n14]), "far": ex(self._pair_list[:, n14:]),
                "src": torch.cat([b[0], a[1], a[1], t[0], t[1], t[2], p[0]]),
                "dst": torch.cat([b[1], a[0], a[2], t[1], t[2], t[3], p[1]]),
                "sizes": [b.shape[1], a.shape[1], a.shape[1]] + [t.shape[1]] * 3 + [p.shape[1]],
                "owners": torch.cat(owners)}
        self._plans[key] = plan
        return plan

    def _rows(self, P: dict, B: int, dtype: torch.dtype) -> dict:
        """The parameters tiled over the terms, atoms and pairs of a batch (cached while the
        parameters are)."""
        key = (B, dtype)
        if not self.training and key in self._row_cache:
            return self._row_cache[key]
        P = {k: v.to(dtype) for k, v in P.items()}
        improper = P["improper_k"][self._impropers_src] * torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=dtype,
                                                                      device=P["improper_k"].device)
        R = {"bond_k": P["bond_k"][self._bonds_src].repeat(B), "bond_x0": P["bond_x0"][self._bonds_src].repeat(B),
             "angle_k": P["angle_k"][self._angles_src].repeat(B),
             "angle_x0": P["angle_x0"][self._angles_src].repeat(B),
             "torsion_k": torch.cat([P["proper_k"][self._propers_src], improper]).repeat(B, 1),
             "torsion_sign": torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=dtype, device=improper.device)}
        atoms = {k: P[k][self._atom_src] for k in self._atom_keys}
        R["charge"] = atoms["charge"].repeat(B)
        i, j = self._pair_list
        R.update({k: v.repeat(B) for k, v in self._pair_rows(atoms, i, j).items()})
        R.update(self._atom_rows(atoms, B))
        R.update(self._unique_rows(P, B))
        if not self.training:
            self._row_cache[key] = R
        return R

    def _atom_rows(self, atoms: dict, B: int) -> dict:
        """Per-atom parameters tiled over the batch, for terms beyond the fixed pair list."""
        return {}

    def _unique_rows(self, P: dict, B: int) -> dict:
        """Parameter arrays over the distinct molecules' atoms and terms (for the kernels)."""
        return {}

    #: evaluations of one batch shape after which ``use_fast="auto"`` compiles the block: compiling takes
    #: tens of seconds per shape and saves milliseconds per call
    compile_after = 1000

    def _run_block(self, block: nn.Module, data: AtomicGraph, B: int, args: tuple):
        """An energy block, compiled where ``use_fast`` asks for it (eager where compiling fails)."""
        from xnn.common.models import compiled
        key = (type(block).__name__, B, data.model_dtype, str(data.pos.device), self.training)
        if self.use_fast is False or not compiled.supported(data.pos.device, data.model_dtype) \
                or key in self._compile_failed:
            return block._reference(*args)
        if self.use_fast == "auto":
            self._block_calls[key] = self._block_calls.get(key, 0) + 1
            if self._block_calls[key] <= self.compile_after:
                return block._reference(*args)
        if key not in self._compile_checked:
            # the compiled derivatives are built at their first use: try them once on copies of the inputs, so
            # a compiler failure for this shape falls back to the (identical) eager block
            try:
                with torch.enable_grad():
                    trial = tuple(a.detach().requires_grad_(True) if isinstance(a, Tensor) and a.is_floating_point()
                                  else a for a in args)
                    out = run_compiled(block, *trial)
                    out = out if isinstance(out, tuple) else (out,)
                    g = torch.autograd.grad(sum(o.sum() for o in out), trial[0], create_graph=self.training)[0]
                    if self.training:
                        torch.autograd.grad(g.sum(), trial[1], allow_unused=True)
            except Exception as err:
                import warnings
                self._compile_failed.add(key)
                warnings.warn(f"compiling the energy block failed for this shape; evaluating it eagerly "
                              f"({type(err).__name__})", RuntimeWarning, stacklevel=2)
                return block._reference(*args)
            self._compile_checked.add(key)
        return run_compiled(block, *args)

    def _kernel_terms(self, data: AtomicGraph, R: dict, B: int, plan: dict, step: Optional[StepPlan] = None):
        """Per-atom energies ``{term: (N,)}`` and extra outputs from fused kernels, or ``None`` for the
        reference (``step``: a step plan, see :meth:`ByteFFPol.plan_step`)."""
        return None

    def _extra(self, data: AtomicGraph, R: dict, B: int) -> tuple:
        """Per-atom energies ``{term: (N,)}`` of terms outside the fixed layout, and extra outputs."""
        return {}, {}

    def _reference_terms(self, data: AtomicGraph, R: dict, B: int, plan: dict, total) -> tuple:
        """The terms block on the bond and pair vectors (gathered here, with periodic images), compiled
        where ``use_fast`` asks for it, plus the terms outside it: per-atom energies, per-structure term
        totals and extra outputs."""
        vec = pair_vectors(data, plan["src"], plan["dst"])
        b, a, p = plan["bonds"], plan["angles"], plan["pairs"]
        args = (vec, R["bond_k"], R["bond_x0"], R["angle_k"], R["angle_x0"], R["torsion_k"], R["torsion_sign"],
                plan["improper"].to(vec.dtype), R["sigma2"], R["eps4"], R["qq"], plan["is14"].to(vec.dtype),
                b[0], b[1], a[1], plan["t_owner"], p[0], p[1], tuple(plan["sizes"]), data.num_nodes)
        e = self._run_block(self._terms, data, B, args)
        node_terms = {k: e[:, j] for j, k in enumerate(_ByteFFTerms.TERMS)}
        extra_node, extra_out = self._extra(data, R, B)
        for k, v in extra_node.items():
            node_terms[k] = node_terms[k] + v
        return sum(node_terms.values()), {k: total(v) for k, v in node_terms.items()}, extra_out

    def forward(self, data: AtomicGraph, plan: Optional[StepPlan] = None) -> dict:
        """Evaluate the energy of conformations of the bound system.

        Parameters
        ----------
        data : AtomicGraph
            The batched graph, every structure a conformation of the bound system.
        plan : StepPlan, optional
            A step plan of a moving periodic structure (:meth:`ByteFFPol.plan_step`), whose
            evaluations decide nothing on the host.

        Returns
        -------
        dict of str to Tensor
            ``node_energy`` ``(N,)``, ``energy`` ``(B,)``, ``charges`` ``(N,)``,
            ``node_features`` ``(N, 1)`` and one per-structure energy per term (eV).
        """
        step = plan
        if step is not None and not hasattr(self, "plan_step"):
            raise ValueError(f"{type(self).__name__} evaluates no step plans")
        # a CUDA-graph capture (xnn.common.deploy.GraphedEvaluation) reads nothing from the device: the checks
        # ran on the evaluations before it
        capturing = data.pos.is_cuda and torch.cuda.is_current_stream_capturing()
        replaying = step is not None and not step.recording
        B = data.num_graphs if replaying or capturing else self._check(data)
        plan = self._plan(B, data.pos.device)
        R = self._rows(self._parameters_cached(), B, data.model_dtype)

        def total(v: Tensor) -> Tensor:
            return v.view(B, -1).to(torch.float64).sum(1)

        kernel = self._kernel_terms(data, R, B, plan, step)
        if kernel is not None:
            kernel, extra_out = kernel
            node = sum(kernel.values())
            terms = {k: total(v) for k, v in kernel.items()}
        else:
            node, terms, extra_out = self._reference_terms(data, R, B, plan, total)
        node = node * KCAL_TO_EV
        out = {"node_energy": node, "energy": self.aggregate_energy(node, data)}
        out.update({k: v * KCAL_TO_EV for k, v in terms.items()})
        out["charges"] = R["charge"]
        out["node_features"] = R["charge"].unsqueeze(1)
        out.update(extra_out)
        return out


def _width(kind: str) -> int:
    return {"bond": 2, "angle": 3, "proper": 4, "improper": 4}[kind]


_BYTEFF_GRAPH = dict(atom_dim=32, feature_dim=8, node_mlp=(64, 256, 2), edge_mlp=(64, 256, 2), hidden=256,
                     layers=4, heads=16, head_dim=16, ffn=(256, 2), out_dim=256, jumping_knowledge="cat",
                     activation="gelu")


class _ByteFFTerms(nn.Module):
    """ByteFF's bonded terms and fixed pairs (the 1,4 pairs, and without a cutoff every pair beyond them) of
    a batch as a function of their vectors and the parameter rows: per-atom energies (kcal/mol) of every
    term. Evaluated eagerly it is the reference; the fast path compiles it."""

    TERMS = ("e_bond", "e_angle", "e_torsion", "e_improper", "e_vdw", "e_coulomb", "e_vdw14", "e_coulomb14")
    #: a bound system has fixed sizes; symbolic ones break the compiled second derivatives (torch 2.5)
    static_shapes = True

    def _reference(self, vec, bond_k, bond_x0, angle_k, angle_x0, torsion_k, sign, improper, sigma2, eps4, qq,
                   is14, b0, b1, a1, t_owner, p0, p1, sizes, N):
        ends = [0]
        for k in sizes:
            ends.append(ends[-1] + k)
        vb, vu, vv, t1, t2, t3, vp = (vec[ends[i]:ends[i + 1]] for i in range(7))
        e_bond = harmonic(torch.sqrt((vb * vb).sum(1)), bond_k, bond_x0)
        e_angle = harmonic(bond_angle(vu, vv), angle_k, angle_x0)
        e_tors = fourier_series(torsion_cos(t1, t2, t3), torsion_k, [1, 2, 3, 4], sign)
        r2 = (vp * vp).sum(1)
        x = (sigma2 / r2) ** 3
        h_vdw = 0.5 * eps4 * (x * x - x)
        h_coul = 0.5 * qq / torch.sqrt(r2)
        h_bond = 0.5 * e_bond
        # one scatter into (atom, term): the torsions' and pairs' columns from their masks
        K = len(self.TERMS)
        t_col = 2 + improper.long()
        p_col = 4 + 2 * is14.long()
        values = torch.cat([h_bond, h_bond, e_angle, e_tors, h_vdw, h_vdw, h_coul, h_coul])
        index = torch.cat([b0 * K, b1 * K, a1 * K + 1, t_owner * K + t_col, p0 * K + p_col, p1 * K + p_col,
                           p0 * K + p_col + 1, p1 * K + p_col + 1])
        return scatter_sum(values, index, N * K).view(N, K)


@register_model("byteff")
class ByteFF(ClassicalFastPath, _GraphParameterizedFF):
    """ByteFF: an AMBER-form force field whose parameters a graph transformer predicts.

    Parameters
    ----------
    structure : rdkit.Chem.Mol, str, ase.Atoms or list, optional
        The system to bind (see :meth:`bind`); a model can be bound later.
    graph : dict, optional
        Options of :class:`~xnn.gnn.models.egt.MolecularGraphTransformer`.
    bonded : dict, optional
        Options of :class:`BondedHead`.
    nonbonded : dict, optional
        Options of :class:`LennardJonesChargeHead`.
    members : int, optional
        Networks of the ensemble whose parameters are averaged. Default 1.
    supported_elements : sequence of int, optional
        Elements an evaluation outside training accepts (all when empty).
    cutoff : float, optional
        Nonbonded cutoff in Angstrom on the neighbor list; ``None`` (default) takes
        every pair of a structure (no cell).
    switch_width : float, optional
        Width of a quintic switch to zero at the cutoff. Default 0.
    coulomb_constant : float, optional
        In kcal Angstrom / (mol e^2).
    fudge_qq, fudge_lj : float, optional
        Scale factors of the 1,4 Coulomb and Lennard-Jones terms.
    charge : int, optional
        Total charge for bond perception when binding an ``ase.Atoms``.
    use_fast : bool or "auto", optional
        Evaluate the energy terms with the fused GPU kernels of
        :mod:`~xnn.ffnn.models.classical_kernels` (the same numbers to round-off,
        differentiable twice for force training, with parameter gradients) when the
        nonbonded pairs come from a neighbor list (a cutoff), on a CUDA device with
        Triton. Without a cutoff (molecule batches) the terms are compiled instead,
        which takes tens of seconds per batch shape: ``"auto"`` (default) compiles
        after :attr:`compile_after` evaluations of one shape, ``True`` at once.
        ``False`` evaluates every term eagerly.

    Notes
    -----
    ``forward`` returns ``e_bond``, ``e_angle``, ``e_torsion``, ``e_improper``,
    ``e_vdw``, ``e_coulomb``, ``e_vdw14`` and ``e_coulomb14`` besides the energies.
    """

    def __init__(self, structure=None, *, graph: Optional[dict] = None, bonded: Optional[dict] = None,
                 nonbonded: Optional[dict] = None, members: int = 1,
                 supported_elements: Sequence[int] = (1, 6, 7, 8, 9, 15, 16, 17, 35, 53),
                 cutoff: Optional[float] = None, switch_width: float = 0.0, coulomb_constant: float = COULOMB,
                 fudge_qq: float = 0.8333333333, fudge_lj: float = 0.5, charge: int = 0,
                 use_fast: Union[bool, str] = "auto"):
        self.fudge_qq, self.fudge_lj = float(fudge_qq), float(fudge_lj)
        self.use_fast = resolve_use_fast(use_fast)
        super().__init__(structure, graph={**_BYTEFF_GRAPH, **(graph or {})},
                         bonded={**dict(pre=(256, 256, 3), out=(256, 4), tanh_output=20.0, form="harmonic"),
                                 **(bonded or {})},
                         nonbonded={**dict(pre=(512, 512, 4), out=(256, 3), tanh_output=20.0), **(nonbonded or {})},
                         members=members, supported_elements=supported_elements, cutoff=cutoff,
                         switch_width=switch_width, coulomb_constant=coulomb_constant, charge=charge)
        self._terms = _ByteFFTerms()

    _pair_names = ("vdw", "coulomb")
    _atom_keys = ("charge", "sigma", "epsilon")

    def _pair_rows(self, atoms: dict, i: Tensor, j: Tensor) -> dict:
        """Combined Lennard-Jones and Coulomb parameters of the fixed pairs (1,4 pairs scaled)."""
        n14 = self._pairs14.shape[1]
        scale = torch.ones(i.shape[0], dtype=atoms["charge"].dtype, device=i.device)
        lj, qq = scale.clone(), scale.clone()
        lj[:n14], qq[:n14] = self.fudge_lj, self.fudge_qq
        sig, eps, q = atoms["sigma"], atoms["epsilon"], atoms["charge"]
        sigma = 0.5 * (sig[i] + sig[j])
        return {"sigma2": sigma * sigma, "eps4": 4.0 * lj * geometric_mean(eps[i], eps[j]),
                "qq": self.coulomb_constant * qq * q[i] * q[j]}

    def _atom_rows(self, atoms: dict, B: int) -> dict:
        return {} if self.all_pairs else {"sigma": atoms["sigma"].repeat(B), "epsilon": atoms["epsilon"].repeat(B)}

    def _extra(self, data: AtomicGraph, R: dict, B: int) -> tuple:
        """With a cutoff: the van der Waals and Coulomb energies of the neighbor list's pairs beyond 1,4."""
        if self.all_pairs:
            return {}, {}
        N = data.num_nodes
        src, dst, _, r2, half, included, _ = half_edges(data, self._excluded, self.n_atoms, self)
        r = torch.sqrt(r2)
        w = included.to(r.dtype) * half
        if self.switch_width > 0.0:
            x = torch.clamp((r - (self.cutoff - self.switch_width)) / self.switch_width, 0.0, 1.0)
            w = w * (1.0 - x ** 3 * (10.0 - 15.0 * x + 6.0 * x * x))
        sigma = 0.5 * (R["sigma"][src] + R["sigma"][dst])
        vdw = w * lennard_jones(r * r, sigma, geometric_mean(R["epsilon"][src], R["epsilon"][dst]))
        coul = w * self.coulomb_constant * R["charge"][src] * R["charge"][dst] / r
        node = {}
        for name, e in (("e_vdw", vdw), ("e_coulomb", coul)):
            node[name] = scatter_sum(0.5 * e, src, N) + scatter_sum(0.5 * e, dst, N)
        return node, {}

    def _unique_rows(self, P: dict, B: int) -> dict:
        """The Lennard-Jones tables over pairs of the distinct molecules' atoms (plain and 1,4) and
        the barrier rows of the torsions, for the kernels."""
        sig, eps = P["sigma"], P["epsilon"]
        T = sig.shape[0]
        sigma = 0.5 * (sig[:, None] + sig[None, :])
        well = 4.0 * geometric_mean(eps[:, None].expand(T, T), eps[None, :].expand(T, T))
        return {"lj_table": lj_table(well, 1.0, sigma), "lj_table14": lj_table(self.fudge_lj * well, 1.0, sigma),
                "proper_rows": P["proper_k"][self._propers_src].repeat(B, 1),
                "improper_rows": P["improper_k"][self._impropers_src].repeat(B, 1)}

    def _kernel_terms(self, data: AtomicGraph, R: dict, B: int, plan: dict, step: Optional[StepPlan] = None):
        """The terms from the kernels of :mod:`~xnn.ffnn.models.classical_kernels` (harmonic terms as
        ``K (x - x0)^2`` with ``K = k / 2``, the torsions as phased Fourier series, Lennard-Jones and
        Coulomb from the pair kernel), or ``None`` where the reference runs."""
        if self.all_pairs:
            return None            # molecule batches: the compiled terms block instead (launch-bound kernels)
        if not wants_fast(self, self.use_fast, data, R["charge"]):
            return None
        twice = needs_twice(self, data)
        b, a, tp, ti, p14, far = (plan[k] for k in ("bonds", "angles", "propers", "impropers", "pairs14", "far"))
        kinds = [(b, "bond"), (a, "angle"), (tp, "torsion"), (ti, "torsion"), (p14, "bond"), (far, "bond")]
        sh = image_shifts(data, kinds)

        def series(k: Tensor, n: list, phase_cos: list, owner: Tensor) -> tuple:
            M, S = k.shape
            mult = torch.tensor(n, dtype=torch.int32, device=k.device).expand(M, S).contiguous()
            c = torch.tensor(phase_cos, dtype=k.dtype, device=k.device).expand(M, S).contiguous()
            return (owner, k, mult, c, torch.zeros_like(c), torch.zeros_like(k[:, 0])), \
                dict(FORM=1, S=S, NMAX=max(n))

        node = {"e_bond": fast_bonded(data, b, "bond", (0.5 * R["bond_k"], R["bond_x0"]), sh[0], twice=twice, tiny=0.0),
                "e_angle": fast_bonded(data, a, "angle", (torch.zeros_like(a[0], dtype=torch.int32),
                                                          0.5 * R["angle_k"], R["angle_x0"]), sh[1], twice=twice,
                                       tiny=0.0)}
        prm, opt = series(R["proper_rows"], [1, 2, 3, 4], [1.0, -1.0, 1.0, -1.0], tp[1])
        node["e_torsion"] = fast_bonded(data, tp, "torsion", prm, sh[2], opt, twice=twice, tiny=0.0)
        prm, opt = series(R["improper_rows"], [2], [-1.0], ti[0])
        node["e_improper"] = fast_bonded(data, ti, "torsion", prm, sh[3], opt, twice=twice, tiny=0.0)
        compact, q = self._atom_src, R["charge"]
        node["e_vdw14"], node["e_coulomb14"] = fast_pairs14(data, p14, sh[4], compact, B, q, R["lj_table14"],
                                                            self.coulomb_constant * self.fudge_qq, twice, 0.0)
        if not self.all_pairs:
            node["e_vdw"], node["e_coulomb"] = fast_nonbonded(
                data, self.n_atoms, B, compact, q, R["lj_table"], self.coulomb_constant, self._excluded, self.cutoff,
                self.switch_width, 0, self, twice, 0.0)
        elif far.shape[1] > 0:
            node["e_vdw"], node["e_coulomb"] = fast_pairs14(data, far, sh[5], compact, B, q, R["lj_table"],
                                                            self.coulomb_constant, twice, 0.0)
        else:
            node["e_vdw"] = node["e_coulomb"] = torch.zeros_like(node["e_bond"])
        return node, {}

    def lammps_input(self, pos, cell=None) -> tuple:
        """The LAMMPS data file and commands that evaluate this model's energy for the bound system.

        The predicted parameters become LAMMPS styles in real units: harmonic bonds and angles
        (``K = k / 2``), ``dihedral fourier``, ``improper cvff`` (``d = -1``, ``n = 2``, center first),
        ``lj/cut/coul/cut`` with arithmetic mixing and the 1,4 scale factors, one atom type per atom of
        the distinct molecules; the charges are scaled to LAMMPS's Coulomb constant. Without a cutoff the
        pair cutoff is the structure's extent plus 2 Angstrom (every pair interacts). Used by
        :func:`~xnn.ffnn.models.lammps_input.write_lammps_input`.

        Parameters
        ----------
        pos : array_like
            Positions ``(n, 3)`` in Angstrom, in the bound system's order.
        cell : array_like, optional
            An orthorhombic cell ``(3, 3)``, with a cutoff only.

        Returns
        -------
        tuple
            The data file text, the commands (``read_data DATAFILE``) and the energy offset (0).
        """
        from xnn.common.models.ops import ATOMIC_MASSES
        from xnn.ffnn.models.lammps_input import LAMMPS_COULOMB, _box, _check_switch, _data, _r, _styles
        _check_switch(self)
        pos = np.asarray(pos, float).reshape(-1, 3)
        if self.all_pairs:
            if cell is not None:
                raise ValueError("a model without a cutoff takes no cell")
            cutoff = float(np.linalg.norm(pos[:, None] - pos[None], axis=-1).max()) + 2.0
        else:
            cutoff = self.cutoff
        pos, bounds, boundary = _box(pos, cell, cutoff)
        with torch.no_grad():
            P = {k: v.detach().double().cpu() for k, v in self.predicted_parameters().items()}
        src = self._atom_src.cpu().numpy()
        z = self._z.cpu().numpy()
        q = P["charge"].numpy()[src] * math.sqrt(self.coulomb_constant / LAMMPS_COULOMB)
        types = [int(u) + 1 for u in src]
        masses = {int(u) + 1: ATOMIC_MASSES[int(zz)] for u, zz in sorted(set(zip(src.tolist(), z.tolist())))}
        sections, tables = {}, {}

        def rows(section, kind, key):
            idx, srcs = getattr(self, f"_{kind}s").cpu().numpy(), getattr(self, f"_{kind}s_src").cpu().numpy()
            table, out = {}, []
            for m in range(idx.shape[1]):
                k = key(int(srcs[m]))
                if k is None:
                    continue
                table.setdefault(k, len(table) + 1)
                out.append((table[k], *[int(a) + 1 for a in idx[:, m]]))
            sections[section], tables[section] = out, table

        rows("Bonds", "bond", lambda r: (_r(0.5 * P["bond_k"][r]), _r(P["bond_x0"][r])))
        rows("Angles", "angle", lambda r: (_r(0.5 * P["angle_k"][r]), _r(math.degrees(float(P["angle_x0"][r])))))
        rows("Dihedrals", "proper", lambda r: tuple((_r(P["proper_k"][r, s]), s + 1, 0 if s % 2 == 0 else 180)
                                                    for s in range(4) if float(P["proper_k"][r, s]) != 0.0) or None)
        rows("Impropers", "improper", lambda r: (_r(P["improper_k"][r, 0]),))
        data = _data("xnn ByteFF", pos, bounds, masses, types, q, sections,
                     {s: len(tables[s]) for s in tables})
        cmds = ["units real", "atom_style full", f"boundary {boundary}",
                f"pair_style lj/cut/coul/cut {cutoff!r}", "pair_modify mix arithmetic",
                f"special_bonds lj 0.0 0.0 {self.fudge_lj!r} coul 0.0 0.0 {self.fudge_qq!r}"]
        cmds += _styles(sections, {"bond": "harmonic", "angle": "harmonic", "dihedral": "fourier",
                                   "improper": "cvff"})
        cmds.append("read_data DATAFILE")
        for t in sorted(masses):
            cmds.append(f"pair_coeff {t} {t} {float(P['epsilon'][t - 1]):.15g} {float(P['sigma'][t - 1]):.15g}")
        cmds += [f"bond_coeff {i} {k[0]:.15g} {k[1]:.15g}" for k, i in tables["Bonds"].items()]
        cmds += [f"angle_coeff {i} {k[0]:.15g} {k[1]:.15g}" for k, i in tables["Angles"].items()]
        cmds += [f"dihedral_coeff {i} {len(k)} " + " ".join(f"{a:.15g} {n} {d}" for a, n, d in k)
                 for k, i in tables["Dihedrals"].items()]
        cmds += [f"improper_coeff {i} {k[0]:.15g} -1 2" for k, i in tables["Impropers"].items()]
        return data, cmds, 0.0

    @classmethod
    def from_config(cls, cfg) -> "ByteFF":
        """Construct from a model config: the keyword options of the class from ``cfg.extra``;
        ``all_pairs`` (default true) evaluates every pair, otherwise ``cfg.cutoff`` is the
        nonbonded cutoff."""
        opts = _options(cfg, cls)
        opts["cutoff"] = None if model_options(cfg).get("all_pairs", True) else float(cfg.cutoff)
        return cls(**opts)


class _PolEnergy(nn.Module):
    """The ByteFF-Pol terms of a batch of copies of a bound system as a function of the positions and the
    parameter rows: per-atom energies (kcal/mol) of every term but polarization, and the polarization
    system ``A mu = E``, which is solved outside the block (a factorization inside compiled code loses its
    gradients). Evaluated eagerly it is the reference; the fast path compiles it."""

    TERMS = ("e_bond", "e_angle", "e_torsion", "e_improper", "e_repulsion", "e_dispersion", "e_electrostatics",
             "e_charge_transfer")
    #: a bound system has fixed sizes; symbolic ones break the compiled second derivatives (torch 2.5)
    static_shapes = True
    #: the inductor's coalescing tiling analysis fails on some of this block's shapes (torch 2.11: an
    #: uncaught CantSplit), so it is compiled without it
    compile_options = {"triton.coalesce_tiling_analysis": False}

    def __init__(self, prefactor: float, wall: float, damping: float, ke: float, thole: float):
        super().__init__()
        self.prefactor, self.wall, self.damping = float(prefactor), float(wall), float(damping)
        self.ke, self.thole = float(ke), float(thole)

    def extra_repr(self) -> str:
        return f"prefactor={self.prefactor}, wall={self.wall}, damping={self.damping}, ke={self.ke}, thole={self.thole}"

    def _reference(self, pos: Tensor, bond_k: Tensor, bond_x0: Tensor, angle_k: Tensor, angle_x0: Tensor,
                   torsion_k: Tensor, sign: Tensor, improper: Tensor, vdw_scale: Tensor, lam: Tensor, eps: Tensor,
                   r0: Tensor, c6: Tensor, qq: Tensor, ct_eps: Tensor, ct_lam: Tensor, charge: Tensor, alpha: Tensor,
                   damp: Tensor, field_scale: Tensor, src: Tensor, dst: Tensor, b0: Tensor, b1: Tensor, a1: Tensor,
                   t_owner: Tensor, p0: Tensor, p1: Tensor, sizes: List[int], B: int,
                   n: int) -> Tuple[Tensor, Tensor, Tensor]:
        N = B * n
        vec = pos.index_select(0, dst) - pos.index_select(0, src)
        # slices at fixed offsets (the compiled second derivative of a split with a size list fails)
        ends = [0]
        for k in sizes:
            ends.append(ends[-1] + k)
        parts = [vec[ends[i]:ends[i + 1]] for i in range(7)]
        vb, vu, vv, t1, t2, t3, vp = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6]
        e_bond = harmonic(torch.sqrt((vb * vb).sum(1)), bond_k, bond_x0)
        e_angle = harmonic(bond_angle(vu, vv), angle_k, angle_x0)
        e_tors = fourier_series(torsion_cos(t1, t2, t3), torsion_k, [1, 2, 3, 4], sign)
        r = torch.sqrt((vp * vp).sum(1))
        rep = vdw_scale * exp_repulsion(r, eps, lam, r0, self.prefactor, self.wall)
        disp = vdw_scale * damped_dispersion(r, c6, r0, self.damping)
        ct = charge_transfer(r, ct_eps, ct_lam, r0)
        elec = qq / r
        A, field = polarization_system(pos.view(B, n, 3), charge.view(B, n), alpha.view(B, n), damp.view(B, n),
                                       field_scale, self.thole)

        # one scatter into (atom, term), a column per entry of TERMS
        K = 8
        t_col = 2 + improper.long()
        values = [0.5 * e_bond, 0.5 * e_bond, e_angle, e_tors, 0.5 * rep, 0.5 * rep, 0.5 * disp, 0.5 * disp,
                  0.5 * elec, 0.5 * elec, 0.5 * ct, 0.5 * ct]
        index = [b0 * K, b1 * K, a1 * K + 1, t_owner * K + t_col, p0 * K + 4, p1 * K + 4, p0 * K + 5, p1 * K + 5,
                 p0 * K + 6, p1 * K + 6, p0 * K + 7, p1 * K + 7]
        energies = scatter_sum(torch.cat(values), torch.cat(index), N * K).view(N, K)
        return energies, A, field


@register_model("byteff_pol")
class ByteFFPol(_GraphParameterizedFF):
    """ByteFF-Pol: a polarizable force field whose parameters a graph transformer predicts.

    Parameters
    ----------
    structure : rdkit.Chem.Mol, str, ase.Atoms or list, optional
        The system to bind (see :meth:`bind`).
    graph, bonded : dict, optional
        As for :class:`ByteFF` (the bonded head defaults to the two-well form).
    nonbonded : dict, optional
        Options of :class:`PolarizableHead`.
    members : int, optional
        Networks of the ensemble. Default 1.
    supported_elements : sequence of int, optional
        Elements an evaluation outside training accepts.
    coulomb_constant : float, optional
        In kcal Angstrom / (mol e^2).
    charge14, vdw14 : float, optional
        Scale factors of the 1,4 electrostatic and repulsion/dispersion terms.
    induction14, induction15 : float, optional
        Scale factors of the charges' polarizing field at 1,4 and 1,5 pairs (1,2 and
        1,3 pairs do not polarize each other).
    thole : float, optional
        Thole damping strength.
    dispersion_damping : float, optional
        ``s`` of the damped C6 term.
    wall : float, optional
        ``s12`` of the short-range repulsive wall, in Angstrom (kcal/mol)^(1/12).
    repulsion_prefactor : float, optional
        Prefactor of the Born-Mayer term (``a`` of ``a eps exp[lam (1 - r/r0)]``).
    charge : int, optional
        Total charge for bond perception when binding an ``ase.Atoms``.
    use_fast : bool or "auto", optional
        Evaluate the energy block compiled on a CUDA device (the same numbers to
        round-off, its first and second derivatives compiled for force training;
        the dipoles are solved outside it). Compiling takes tens of seconds per
        batch shape: ``"auto"`` (default) compiles after :attr:`compile_after`
        evaluations of one shape, ``True`` at once (molecular dynamics,
        training), ``False`` never. In cells, ``True`` and ``"auto"`` solve the
        dipoles and run the mesh on fused GPU kernels when evaluating (the same
        numbers to round-off), training stays on the reference.
    cutoff : float, optional
        ``None`` (default): every pair of a structure interacts (molecules and
        clusters, no cell). A cutoff in Angstrom evaluates periodic cells: the
        repulsion, dispersion and charge transfer of the neighbor pairs truncated
        there, the charges and induced dipoles as Ewald sums.
    ewald_tolerance : float, optional
        Sets the Ewald coefficient ``sqrt(-ln(2 tol)) / cutoff`` and the mesh,
        ``ceil(2 alpha L / (3 tol^(1/5)))`` points (at least six) along a cell
        side ``L``, rounded up to a size whose prime factors are at most 7 (as
        OpenMM's GPU platforms do). Default ``5e-5``.
    ewald_alpha : float, optional
        The Ewald coefficient in 1/Angstrom instead.
    pme_grid : sequence of int, optional
        The mesh instead.
    pme_order : int, optional
        B-spline order of the mesh. Default 5.
    polar_eps : float, optional
        Convergence of the iterative dipole solve of cells, the rms change per
        iteration in Debye. Default ``1e-5``.
    max_iter : int, optional
        Iterations of that solve at most. Default 100.
    long_range_correction : bool, optional
        Add the uniform-density estimate of the short-range terms beyond the
        cutoff. Default ``True``.
    predictor : str, optional
        Extrapolate the dipoles of the next step from the last ones as the start
        of the iterative solve in cells (molecular dynamics): ``"aspc"``,
        ``"gear"`` or ``"previous"`` (see
        :class:`~xnn.ffnn.models.DipolePredictor`); ``None`` (default) starts from
        the direct dipoles. The converged dipoles do not change.
    predictor_history : int, optional
        Dipole sets the predictor uses. Default 6.

    Notes
    -----
    ``forward`` returns ``e_repulsion``, ``e_dispersion``, ``e_electrostatics``,
    ``e_polarization`` and ``e_charge_transfer`` besides the bonded terms, and the
    ``induced_dipoles`` ``(N, 3)`` in e Angstrom. A cell's sides must be at least
    twice the cutoff long (its perpendicular widths), so that every pair interacts
    with one image at most; the 1,2 to 1,5 pairs of a molecule take their nearest
    images. In cells, :meth:`plan_step` records the host decisions of an evaluation
    for :class:`~xnn.common.deploy.StepGraph`, which replays molecular dynamics
    steps as CUDA graphs.
    """

    polarizable = True
    foundation_default = "byteff-pol-25"
    # the MDI engine's StepGraph reads each step's status during the next step for this model
    late_checks = True

    def __init__(self, structure=None, *, graph: Optional[dict] = None, bonded: Optional[dict] = None,
                 nonbonded: Optional[dict] = None, members: int = 1,
                 supported_elements: Sequence[int] = (1, 3, 6, 7, 8, 9, 15, 16, 17),
                 coulomb_constant: float = COULOMB, charge14: float = 0.5, vdw14: float = 0.5,
                 induction14: float = 0.5, induction15: float = 1.0, thole: float = 0.39,
                 dispersion_damping: float = 120.0, wall: float = 1.5, repulsion_prefactor: float = 6.0,
                 charge: int = 0, use_fast: Union[bool, str] = "auto", cutoff: Optional[float] = None,
                 ewald_tolerance: float = 5e-5, ewald_alpha: Optional[float] = None,
                 pme_grid: Optional[Sequence[int]] = None, pme_order: int = 5, polar_eps: float = 1e-5,
                 max_iter: int = 100, long_range_correction: bool = True, predictor: Optional[str] = None,
                 predictor_history: int = 6):
        self.ewald_tolerance = float(ewald_tolerance)
        self.solver_stats: dict = {}
        self._adjoints: dict = {}
        self.set_dipole_predictor(predictor, predictor_history)
        self.ewald_alpha = None if ewald_alpha is None else float(ewald_alpha)
        self.pme_grid = None if pme_grid is None else [int(k) for k in pme_grid]
        self.pme_order, self.polar_eps, self.max_iter = int(pme_order), float(polar_eps), int(max_iter)
        self.long_range_correction = bool(long_range_correction)
        self.charge14, self.vdw14 = float(charge14), float(vdw14)
        self.use_fast = resolve_use_fast(use_fast)
        self.induction14, self.induction15 = float(induction14), float(induction15)
        self.thole, self.dispersion_damping = float(thole), float(dispersion_damping)
        self.wall, self.repulsion_prefactor = float(wall), float(repulsion_prefactor)
        super().__init__(structure, graph={**_BYTEFF_GRAPH, **(graph or {})},
                         bonded={**dict(pre=(256, 256, 3), out=(256, 4), tanh_output=20.0, form="two_well"),
                                 **(bonded or {})},
                         nonbonded={**dict(pre=(256, 256, 3), out=(256, 3)), **(nonbonded or {})},
                         members=members, supported_elements=supported_elements, cutoff=cutoff,
                         coulomb_constant=coulomb_constant, charge=charge)
        self._energy = _PolEnergy(self.repulsion_prefactor, self.wall, self.dispersion_damping,
                                  self.coulomb_constant, self.thole)


    def _bind_pairs(self, n: int) -> None:
        super()._bind_pairs(n)
        scale = 1.0 - torch.eye(n, dtype=torch.float64, device=self._excluded.device)
        for p, v in ((self._pairs12, 0.0), (self._pairs13, 0.0), (self._pairs14, self.induction14),
                     (self._pairs15, self.induction15)):
            scale[p[0], p[1]] = v
            scale[p[1], p[0]] = v
        self.register_buffer("_field_scale", scale, persistent=False)
        if not self.all_pairs:
            # cells: the fixed pairs are the 1,4 and 1,5 pairs (repulsion and dispersion), the neighbor pairs
            # exclude every pair within 1,5; the Ewald sums take every pair, corrected on those within 1,5
            # by the charge scale (energy) and the field scale (induction) less one
            p14, p15 = self._pairs14, self._pairs15
            within = torch.cat([self._pairs12, self._pairs13, p14, p15], 1)
            self.register_buffer("_pair_list", torch.cat([p14, p15], 1), persistent=False)
            self.register_buffer("_pair_is15", torch.cat([torch.zeros(p14.shape[1], dtype=torch.bool),
                                                          torch.ones(p15.shape[1], dtype=torch.bool)]).to(
                within.device), persistent=False)
            self.register_buffer("_excluded", within, persistent=False)
            charge_scale, field_scale = [], []
            for p, m, f in ((self._pairs12, 0.0, 0.0), (self._pairs13, 0.0, 0.0),
                            (p14, self.charge14, self.induction14), (p15, 1.0, self.induction15)):
                charge_scale += [m - 1.0] * p.shape[1]
                field_scale += [f - 1.0] * p.shape[1]
            self.register_buffer("_corrected", within, persistent=False)
            self.register_buffer("_corr_scales", torch.tensor([charge_scale, field_scale], dtype=torch.float64,
                                                              device=within.device).reshape(2, -1), persistent=False)

    _pair_names = ("repulsion", "dispersion", "electrostatics", "charge_transfer")
    _atom_keys = ("charge", "volume", "c6", "r0", "alpha", "damp", "lam", "eps", "ct_eps", "ct_lam")

    def _pair_rows(self, atoms: dict, i: Tensor, j: Tensor) -> dict:
        """Combined pair parameters (Lorentz-Berthelot) of the fixed pairs: 1,4 pairs scaled, no charge
        transfer within 1,5."""
        n14 = self._pairs14.shape[1]
        a = atoms
        vdw = torch.ones(i.shape[0], dtype=a["charge"].dtype, device=i.device)
        qq = vdw.clone()
        vdw[:n14], qq[:n14] = self.vdw14, self.charge14
        ct = (~self._pair_is15).to(vdw.dtype)
        ct[:n14] = 0.0
        return {"vdw_scale": vdw, "lam": torch.sqrt(a["lam"][i] * a["lam"][j]),
                "eps": torch.sqrt(a["eps"][i] * a["eps"][j]), "r0": 0.5 * (a["r0"][i] + a["r0"][j]),
                "c6": torch.sqrt(a["c6"][i] * a["c6"][j]),
                "qq": self.coulomb_constant * qq * a["charge"][i] * a["charge"][j],
                "ct_eps": ct * torch.sqrt(a["ct_eps"][i] * a["ct_eps"][j]),
                "ct_lam": torch.sqrt(a["ct_lam"][i] * a["ct_lam"][j])}

    def _atom_rows(self, atoms: dict, B: int) -> dict:
        rows = {k: atoms[k].repeat(B) for k in ("alpha", "damp")}
        if not self.all_pairs:
            # the neighbor pairs of cells combine per-atom rows (named apart from the fixed pairs' rows)
            rows.update({"atom_" + k: atoms[k].repeat(B) for k in ("lam", "eps", "r0", "c6", "ct_eps", "ct_lam")})
        return rows

    def _unique_rows(self, P: dict, B: int) -> dict:
        """With a cutoff: the tail sums of the long-range correction (:func:`_tail_sums`)."""
        if self.all_pairs or not self.long_range_correction:
            return {}
        counts = torch.bincount(self._atom_src, minlength=P["eps"].shape[0])
        return {"tails": _tail_sums(P, counts, self.cutoff, self.repulsion_prefactor, self.wall,
                                    self.dispersion_damping)}

    def _kernel_terms(self, data: AtomicGraph, R: dict, B: int, plan: dict, step: Optional[StepPlan] = None):
        """Every term from the energy block :class:`_PolEnergy`, eager (the reference) or compiled (the
        fast path), and the induced dipoles of its polarization system; with a cutoff the periodic terms
        (:meth:`_periodic_terms`)."""
        if not self.all_pairs:
            return self._periodic_terms(data, R, B, plan, step)
        n, dtype = self.n_atoms, data.model_dtype
        b, a, p = plan["bonds"], plan["angles"], plan["pairs"]
        args = (data.pos.to(dtype), R["bond_k"], R["bond_x0"], R["angle_k"], R["angle_x0"], R["torsion_k"],
                R["torsion_sign"], plan["improper"].to(dtype), R["vdw_scale"], R["lam"], R["eps"], R["r0"], R["c6"],
                R["qq"], R["ct_eps"], R["ct_lam"], R["charge"], R["alpha"], R["damp"], self._field_scale.to(dtype),
                plan["src"], plan["dst"], b[0], b[1], a[1], plan["t_owner"], p[0], p[1], tuple(plan["sizes"]), B, n)
        e, A, field = self._run_block(self._energy, data, B, args)
        mu = solve_dipoles(A, field)
        terms = {k: e[:, j] for j, k in enumerate(_PolEnergy.TERMS)}
        terms["e_polarization"] = -0.5 * self.coulomb_constant * (mu * field).sum(-1).reshape(-1)
        return terms, {"induced_dipoles": mu.reshape(-1, 3)}

    def _ewald_context(self, data: AtomicGraph, dtype: torch.dtype, step: Optional[StepPlan] = None) -> dict:
        """The Ewald coefficient and meshes of a batch of cells, with the mesh groups of the induction
        solve (:class:`~xnn.ffnn.models.amoeba._MutualField`): structures on one mesh share a plan. The
        host decisions are a step plan's when it has them."""
        layout = None if step is None else step.ewald
        if layout is None:
            layout = self._ewald_layout(data, dtype)
            if step is not None:
                step.ewald = layout
        return {"structures": layout["structures"], "groups": layout["groups"], "plans": {},
                "pos": data.pos.to(dtype), "cells": data.cell.to(dtype), "alpha_p_value": layout["alpha"],
                "porder": self.pme_order, "mesh_self": False, "kernels": False}

    def _ewald_layout(self, data: AtomicGraph, dtype: torch.dtype) -> dict:
        """The host decisions of the Ewald sums of a batch (checks, coefficient, meshes and groups)."""
        message = "a ByteFF-Pol model with a cutoff evaluates periodic cells; give every structure one"
        if data.cell is None:
            raise ValueError(message)
        cells = data.cell.to(dtype)
        B, n, rc = data.num_graphs, self.n_atoms, self.cutoff
        host = cells.detach().cpu().double()
        volume = torch.abs(torch.linalg.det(host))
        if bool((volume < 1e-8).any()) or (data.pbc is not None and not bool(data.pbc.all())):
            raise ValueError(message + " (periodic along all three axes)")
        widths = volume[:, None] / torch.linalg.norm(torch.linalg.cross(host[:, [1, 2, 0]], host[:, [2, 0, 1]]),
                                                     dim=-1)
        if bool((widths < 2.0 * rc - 1e-9).any()):
            raise ValueError(f"every cell must be at least twice the cutoff ({2.0 * rc} Angstrom) wide")
        tol = self.ewald_tolerance
        alpha = self.ewald_alpha if self.ewald_alpha is not None else math.sqrt(-math.log(2.0 * tol)) / rc
        by_mesh: dict = {}
        for b in range(B):
            mesh = (self.pme_grid if self.pme_grid is not None
                    else [_fft_size(max(6, math.ceil(2.0 * alpha * float(host[b, i, i]) / (3.0 * tol ** 0.2))))
                          for i in range(3)])
            by_mesh.setdefault(tuple(mesh), []).append(b)
        device = cells.device
        groups, structures = [], []
        for mesh, bs in by_mesh.items():
            idx = torch.cat([torch.arange(b * n, (b + 1) * n, device=device) for b in bs])
            groups.append({"bs": torch.tensor(bs, dtype=torch.long, device=device), "idx": idx,
                           "batch": torch.arange(len(bs), device=device).repeat_interleave(n), "grid": list(mesh),
                           "normals": None, "influence": None, "whole": len(bs) == B})
            structures += [(b, None, list(mesh), None, None) for b in bs]
        return {"structures": structures, "groups": groups, "alpha": alpha}

    def _fast(self, data: AtomicGraph, dtype: torch.dtype) -> bool:
        """Whether a periodic evaluation runs on the fused kernels: on CUDA, not training."""
        return (self.use_fast is not False and not self.training and data.pos.is_cuda
                and _kernels_supported(data.pos.device, dtype))

    def plan_step(self, data: AtomicGraph, skin: float = 0.0, capacity: Optional[int] = None) -> StepPlan:
        """Record the host decisions of an evaluation of a periodic structure for later evaluations of it
        as it moves (see :class:`~xnn.ffnn.models.StepPlan`): the neighbor pairs as a Verlet list of
        ``cutoff + skin``, the image shifts of the bonded vectors, the meshes and the Ewald coefficient.
        Evaluations on the plan read nothing from the device but the convergence of the dipole solve,
        and with fixed solve lengths not even that. The recording evaluation runs without gradients and
        leaves the dipole predictor alone.

        Parameters
        ----------
        data : AtomicGraph
            The batched graph, its neighbor list built with ``cutoff + skin``.
        skin : float, optional
            The margin of the neighbor list beyond the cutoff in Angstrom. Default 0.
        capacity : int, optional
            The number of pairs the plan holds, padded beyond the unique pairs of the neighbor
            list; by default exactly those.

        Returns
        -------
        StepPlan
            The plan, for ``forward(data, plan)``.
        """
        if self.all_pairs:
            raise ValueError("step plans need a cutoff (periodic cells)")
        plan = StepPlan(skin, capacity)
        with torch.no_grad():
            self.forward(data, plan)
        plan.recording = False
        plan.converged = torch.ones((), dtype=torch.bool, device=data.pos.device)
        plan.needed = torch.zeros(len(SOLVE_PHASES), dtype=torch.float64, device=data.pos.device)
        return plan

    def plan_pairs(self, data: AtomicGraph, plan: StepPlan) -> bool:
        """Refresh the pair list of ``plan`` from the neighbor list of ``data`` in place.

        Parameters
        ----------
        data : AtomicGraph
            The graph at the current positions, its neighbor list built with ``cutoff + skin``.
        plan : StepPlan
            The plan to refresh.

        Returns
        -------
        bool
            Whether the pairs fitted the plan's capacity and were copied.
        """
        src, dst = data.edge_index[0], data.edge_index[1]
        if plan.capacity is not None and int((src < dst).sum()) > plan.capacity:
            return False
        fresh = StepPlan(plan.skin, plan.capacity)
        with torch.no_grad():
            self._record_pairs(data, fresh)
        return plan.update(fresh, bonded=False)

    def refresh_pairs(self, data: AtomicGraph, plan: StepPlan, cutoff: float) -> bool:
        """Refresh the pair list of ``plan`` in place at the positions and cell of ``data``, one
        structure, from a half neighbor list of ``cutoff`` (the cutoff plus the skin) found on the
        device, without building a graph.

        Parameters
        ----------
        data : AtomicGraph
            The plan's graph at the current step.
        plan : StepPlan
            The plan to refresh.
        cutoff : float
            The radius of the pair list.

        Returns
        -------
        bool
            Whether the pairs fitted the plan's capacity and were copied.
        """
        cell = data.cell.detach().reshape(-1, 3, 3)[0]
        pbc = None if data.pbc is None else data.pbc.reshape(-1, 3)[0]
        fresh = self.stage_pairs(data, plan, unique_pairs(data.pos.detach(), cutoff, cell, pbc))
        return fresh is not None and plan.update(fresh, bonded=False)

    def stage_pairs(self, data: AtomicGraph, plan: StepPlan, pairs: tuple,
                    single_image: bool = False) -> Optional[StepPlan]:
        """The unique pairs ``(i, k, shift)`` of ``data``, one structure, sorted by ``i``, recorded as
        the pair list of a new plan for ``plan.update(fresh, bonded=False)``; ``plan`` is not touched,
        so a step can still run on it meanwhile.

        Parameters
        ----------
        data : AtomicGraph
            The plan's graph, its positions and cell where the pairs were found.
        plan : StepPlan
            The plan whose capacity and skin the new one takes.
        pairs : tuple of Tensor
            The pairs and their integer image shifts, or those in buffers of the plan's capacity
            followed by their number on the device.
        single_image : bool, optional
            Every pair has one image within the list's cutoff. Unused: every image of an excluded
            pair is excluded, its far images lying beyond the cutoff anyway.

        Returns
        -------
        StepPlan or None
            The new pair list, or ``None`` when the pairs exceed the capacity.
        """
        if plan.capacity is not None and int(pairs[0].shape[0]) > plan.capacity:
            return None
        fresh = StepPlan(plan.skin, plan.capacity)
        with torch.no_grad():
            self._record_pairs(data, fresh, pairs)
        return fresh

    def _record_pairs(self, data: AtomicGraph, plan: StepPlan, pairs: Optional[tuple] = None) -> None:
        """Store the unique neighbor pairs of ``data`` in ``plan``, sorted by their first atom and
        padded to the plan's capacity: atoms, image shifts, whether each lies beyond 1,5 (its
        short-range weight) and the positions they were found at. ``pairs`` takes the place of the
        graph's neighbor list: ``(i, k, shift)`` sorted by ``i``, or those in buffers of the plan's
        capacity with their number on the device (nothing is read on the host)."""
        dtype = data.model_dtype
        n, B = self.n_atoms, data.num_graphs
        if pairs is not None and len(pairs) == 4:
            i, k, shift, total = pairs
            weight = (~excluded_pairs(i, k, self._excluded, n, B, self)).to(dtype)
            valid = (torch.arange(int(i.shape[0]), device=i.device) < total).to(dtype)
            plan.pairs, plan.valid, plan.scales = (i, k, shift.to(data.pos.dtype)), valid, weight * valid
            plan.count = total.reshape(1).to(torch.int32)
            plan.reference = data.pos.detach().clone()
            if self._fast(data, dtype):
                plan.rows = _kernels().pair_rows(i, k, data.num_nodes, total, sorted_i=True)
            return
        if pairs is None:
            src, dst = data.edge_index[0], data.edge_index[1]
            sel = (src < dst).nonzero()[:, 0]
            i, k = src.index_select(0, sel), dst.index_select(0, sel)
            shift = data.cell_shifts.index_select(0, sel)
            order = torch.argsort(i, stable=True)
            i, k, shift = i[order], k[order], shift[order]
        else:
            i, k, shift = pairs
        shift = shift.to(data.pos.dtype)
        weight = (~excluded_pairs(i, k, self._excluded, n, B, self)).to(dtype)
        M = int(i.shape[0])
        cap = M if plan.capacity is None else int(plan.capacity)
        if cap < M:
            raise ValueError(f"the plan holds {cap} pairs, the neighbor list has {M}")
        if cap > M > 0:
            # the padding repeats the first pair with zero weight
            fill = torch.zeros(cap - M, dtype=torch.long, device=i.device)
            i, k, shift = torch.cat([i, i[fill]]), torch.cat([k, k[fill]]), torch.cat([shift, shift[fill]])
            weight = torch.cat([weight, torch.zeros_like(weight[fill])])
        valid = (torch.arange(int(i.shape[0]), device=i.device) < M).to(dtype)
        plan.pairs, plan.valid, plan.scales = (i, k, shift), valid, weight * valid
        plan.count = torch.tensor([M], dtype=torch.int32, device=i.device)
        plan.reference = data.pos.detach().clone()
        if self._fast(data, dtype):
            plan.rows = _kernels().pair_rows(i, k, data.num_nodes, M, sorted_i=True)

    def export_core(self) -> nn.Module:
        """The scriptable core of :func:`~xnn.common.deploy.export_torchscript_potential`: the parameters of the
        bound system with the reference's terms, for molecules and clusters
        (:class:`~xnn.hybnn.models.byteff_export.ByteFFPolExport`) or, with a cutoff, periodic cells
        (:class:`~xnn.hybnn.models.byteff_export.ByteFFPolCellExport`)."""
        from .byteff_export import ByteFFPolCellExport, ByteFFPolExport
        return ByteFFPolExport(self) if self.all_pairs else ByteFFPolCellExport(self)

    def set_dipole_predictor(self, kind: Optional[str], history: int = 6) -> None:
        """Switch the dipole predictor of the iterative solve in cells on (``"aspc"``, ``"gear"``,
        ``"previous"``) or off (``None``).

        Parameters
        ----------
        kind : str or None
            The extrapolation scheme (:class:`~xnn.ffnn.models.DipolePredictor`).
        history : int, optional
            Dipole sets it uses. Default 6.
        """
        if kind is None or str(kind).lower() == "none":
            self.predictor = None
        else:
            self.predictor = DipolePredictor(str(kind).lower(), history)

    def rewind_predictors(self) -> None:
        """Take the dipoles of the last evaluation back out of the predictor, before a step is
        evaluated again."""
        if self.predictor is not None:
            self.predictor.rewind()

    def _periodic_terms(self, data: AtomicGraph, R: dict, B: int, plan: dict, step: Optional[StepPlan] = None):
        """The terms of periodic cells: bonded terms and the 1,4 / 1,5 pairs on minimum-image vectors, the
        short-range terms of the neighbor pairs beyond 1,5 within the cutoff (plus their long-range
        correction), the Ewald sums of the charges (real-space pairs within the cutoff, mesh, self term,
        and the scaled pairs within 1,5 corrected) and of the induced dipoles, which are solved by
        conjugate gradients with the gradient through the adjoint system. On a step plan the pairs, image
        shifts and Ewald layout are the plan's."""
        n, N, dtype = self.n_atoms, data.num_nodes, data.model_dtype
        C, rc, thole = self.coulomb_constant, self.cutoff, self.thole
        recording = step is not None and step.recording
        ew = self._ewald_context(data, dtype, step)
        # the fast path of an evaluation on CUDA: the induction solve and the mesh on AMOEBA's kernels
        # (differentiable once: training stays eager)
        fast = self._fast(data, dtype)
        ew["kernels"] = fast
        off = (torch.arange(B, device=data.pos.device) * n).view(1, B, 1)
        ci, cj = (self._corrected.unsqueeze(1) + off).reshape(2, -1)
        if recording:
            step.shifts = [pair_image_shifts(data, plan["src"], plan["dst"]), pair_image_shifts(data, ci, cj)]
            self._record_pairs(data, step)
        shifts = (None, None) if step is None else step.shifts
        alpha_e = ew["alpha_p_value"]
        node = {}

        # the glue between the kernels runs compiled on a step plan's fast path (first derivatives only)
        glue = fast and step is not None and not recording

        # bonded terms and the fixed 1,4 and 1,5 pairs (repulsion and dispersion)
        vec = pair_vectors(data, plan["src"], plan["dst"], shifts[0])
        b, a, p = plan["bonds"], plan["angles"], plan["pairs"]
        with fused(glue):
            local = _pol_local(vec, tuple(plan["sizes"]), R["bond_k"], R["bond_x0"], R["angle_k"], R["angle_x0"],
                               R["torsion_k"], R["torsion_sign"], plan["improper"], R["vdw_scale"], R["eps"], R["lam"],
                               R["r0"], R["c6"], b[0], b[1], a[1], plan["t_owner"], p[0], p[1], N,
                               self.repulsion_prefactor, self.wall, self.dispersion_damping)
        node.update({k: local[:, j] for j, k in enumerate(("e_bond", "e_angle", "e_torsion", "e_improper"))})
        rep, disp = local[:, 4], local[:, 5]

        # neighbor pairs, once each: short-range terms beyond 1,5, real-space Ewald terms of every pair
        if step is None:
            src, dst, vn, _, _, beyond, _ = half_edges(data, self._excluded, n, self)
            valid, beyond = torch.ones_like(vn[:, 0]), beyond.to(dtype)
        else:
            src, dst, pair_shift = step.pairs
            vn = _plan_vectors(data, src, dst, pair_shift, step.count, fast, step.rows if fast else None)
            valid, beyond = step.valid, step.scales
        q, damp = R["charge"], R["damp"]
        atoms = torch.stack([q, damp] + [R["atom_" + k] for k in ("eps", "lam", "r0", "c6", "ct_eps", "ct_lam")], 1)
        constants = (rc * rc, alpha_e, thole, C, self.repulsion_prefactor, self.wall, self.dispersion_damping)
        if fast and step is not None and not recording and step.rows is not None:
            # a step plan's fast path (molecular dynamics, first derivatives only): the fused kernels walk
            # the plan's rows
            from .byteff_kernels import pol_real_space
            pairs_node, field, D1u, D2u = pol_real_space(vn, src, dst, beyond, atoms, step.rows, step.count,
                                                         *constants)
        else:
            pairs_node, field, D1u, D2u = _pol_real_space(vn, valid, beyond, src, dst, atoms, N, *constants)
        rep, disp = rep + pairs_node[:, 0], disp + pairs_node[:, 1]
        ct, elec = pairs_node[:, 2], pairs_node[:, 3]

        # the pairs within 1,5: charge scale and field scale less one
        cm, cs = self._corr_scales.to(dtype).repeat(1, B)
        vc = pair_vectors(data, ci, cj, shifts[1])
        with fused(glue):
            scaled = _pol_scaled_pairs(vc, ci, cj, cm, cs, q, damp, N, C, thole)
        elec, field = elec + scaled[:, 0], field + scaled[:, 1:]

        # the mesh: reciprocal energy and field of the charges, and the self energy
        for g, mp in _mesh_plans(ew, "m", self.pme_order):
            idx = g["idx"]
            qg = q if g["whole"] else q.index_select(0, idx)
            phi, dphi, _ = mp.potential(qg, None, None, alpha_e, hessian=False)
            if g["whole"]:
                elec, field = elec + 0.5 * C * qg * phi, field - dphi
            else:
                elec, field = elec.index_add(0, idx, 0.5 * C * qg * phi), field.index_add(0, idx, -dphi)
        elec = elec - C * alpha_e / SQRT_PI * q * q

        # the induced dipoles: A mu = E with the real-space ladders, the mesh field and the self field
        alpha = R["alpha"]
        polar = alpha > 0
        op = _MutualField(src, dst, N, polar, ew, self.polar_eps, self.max_iter, ew["pos"], ew["cells"])
        op.n_polar, op.kernels, op.coulomb = N, fast, C * KCAL_TO_EV
        # one field polarizes and is polarized: the adjoint of the polarization energy is -C/2 times the
        # dipoles themselves, with no second solve (checked on the incoming gradient)
        op.polar_field = (field * polar[:, None].to(dtype)).detach()
        op.stats, op.adjoints = self.solver_stats, self._adjoints
        if step is not None:
            op.count, op.rows, op.collect_stats = step.count, step.rows if fast else None, False
            op.select = (op.count, None)
            if not recording:
                op.fixed, op.converged, op.needed = step.solve_iterations, step.converged, step.needed
        predict = self.predictor is not None and not self.training and not recording
        if predict:
            guess = self.predictor.predict(field)
            if guess is not None and guess.shape == field.shape:
                op.guess = guess.detach()
        mu = _ImplicitInduction.apply(field, alpha, None, vn, D1u, D2u, ew["pos"], ew["cells"], op)
        if predict:
            self.predictor.push(mu.detach())
        node["e_polarization"] = -0.5 * C * (mu * field).sum(1)

        # the tails beyond the cutoff
        if "tails" in R:
            volume = cell_volume(ew["cells"])
            tail = (4.0 * math.pi * n / (n + 1.0)) * R["tails"][None, :] / volume[:, None]     # (B, 3)
            tail = tail.repeat_interleave(n, 0) / n
            rep, disp, ct = rep + tail[:, 0], disp + tail[:, 1], ct + tail[:, 2]
        node.update({"e_repulsion": rep, "e_dispersion": disp, "e_electrostatics": elec, "e_charge_transfer": ct})
        return node, {"induced_dipoles": mu}

    def set_use_fast(self, use_fast):
        """Set when the energy block runs compiled (see ``use_fast`` of the model).

        Parameters
        ----------
        use_fast : bool or str
            ``True``, ``False`` or ``"auto"``.

        Returns
        -------
        nn.Module
            The model.
        """
        self.use_fast = resolve_use_fast(use_fast)
        return self

    @classmethod
    def from_config(cls, cfg) -> "ByteFFPol":
        """Construct from a model config; the options are read from ``cfg.extra``
        (the keyword arguments of the class). ``all_pairs`` (default true) evaluates every
        pair, otherwise ``cfg.cutoff`` is the cutoff of periodic cells."""
        opts = _options(cfg, cls)
        opts["cutoff"] = None if model_options(cfg).get("all_pairs", True) else float(cfg.cutoff)
        return cls(**opts)


def _options(cfg, cls) -> dict:
    """The keyword arguments of ``cls`` (but the cutoff) present in a model config's options."""
    extra = model_options(cfg)
    names = [p for p in inspect.signature(cls.__init__).parameters if p not in ("self", "cutoff")]
    return {k: extra.get(k) for k in names if extra.get(k) is not None}


__all__ = ["ByteFF", "ByteFFPol", "BondedHead", "LennardJonesChargeHead", "PolarizableHead", "COULOMB",
           "free_atom_table", "TWO_WELL_BONDS", "TWO_WELL_ANGLES"]
