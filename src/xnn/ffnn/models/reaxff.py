"""ReaxFF: the reactive force field, classical and machine-learned (ReaxFF-nn).

ReaxFF (van Duin, Dasgupta, Lorant & Goddard, *J. Phys. Chem. A* 105, 9396,
2001) writes the total energy as a sum of bond-order-dependent valence terms
plus shielded, tapered nonbonded terms evaluated between *all* atom pairs:

    E = E_bond + E_lp + E_over + E_under + E_val + E_pen + E_coa
      + E_tors + E_conj + E_hbond + E_vdWaals + E_Coulomb  (+ E_self)

Bond orders are computed directly from interatomic distances (sigma, pi and
double-pi contributions, eq 2 of the paper), corrected for over-coordination
and residual 1-3 contributions (eq 3), and every valence term is written in
terms of these bond orders so that it vanishes smoothly as bonds break --
which is what makes the force field *reactive*. Partial charges are
re-equilibrated at every geometry with the electronegativity-equalization
method (EEM), giving geometry-dependent Coulomb energies. The
transition-metal extension (Nielson et al., *J. Phys. Chem. A* 109, 493,
2005) and the modern standard form (Senftle et al., *npj Comput. Mater.* 2,
15011, 2016) share this term structure.

ReaxFF-nn (Guo et al., *Comput. Mater. Sci.* 172, 109393, 2020; Xue et al.,
*Phys. Chem. Chem. Phys.* 23, 19457, 2021 -- ReaxFF-MPNN; Guo et al., 2023 --
ReaxFF-nn in GULP) replaces the closed-form bond-order correction with a
message-passing neural network acting on the uncorrected bond orders and
(optionally) the closed-form bond energy with a small per-bond network,
keeping every other ReaxFF term intact. Both variants are implemented here in
one model; every term is checked against the published equations in
``tests/test_reaxff.py`` (see the fidelity notes in the documentation for why
no third-party verification artifact is distributed).

Conventions and scope
---------------------
* All parameters come from a standard ``ffield`` text library or a
  ReaxFF-nn JSON library (see :mod:`xnn.ffnn.models.ffield`); energies are
  converted from kcal/mol to eV at assembly.
* Any parameter group can be made trainable (``trainable=...``): the ReaxFF
  functional form is differentiable end-to-end, so classical parameters can
  be refit by gradient descent (Guo et al. 2020). In ReaxFF-nn mode the
  network weights are trainable by default.
* Nonbonded terms (vdW / Coulomb / EEM) are evaluated on the model's neighbor
  list within ``vdw_cutoff`` with the standard 7th-order taper, which has
  zero value and slope at the cutoff.
* Valence angles and torsions use the composed edge geometry
  (``r_ik = |r_ij + r_jk|``), which is well defined for any cell size.
* A handful of conventions (switching-function regularization, unit handling
  of unlisted angle types, the hydrogen-bond angle numerator) follow the
  behavior under which published ReaxFF-nn parameter libraries were trained,
  so that such libraries evaluate correctly; each is marked in the code.
* The neural vdW taper of the ReaxFF-nn JSON format (``VdwFunction > 0``
  with vdW network weights) and the per-molecule training offsets
  (``MolEnergy``) are not implemented.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import torch
from torch import Tensor
from torch.nn import Parameter, ParameterDict

from xnn.common.data import AtomicGraph
from xnn.common.models import InteratomicPotential, register_model
from xnn.common.models.ops import scatter_sum, build_triplets

from .ffield import (FFieldLibrary, read_ffield, complete_off_diagonal,
                     complete_hbonds, resolve_torsion, cutoff_table,
                     SYMBOL_TO_Z, CHEMICAL_SYMBOLS, P_ANGLE, P_TORSION,
                     P_HBOND)

# kcal/mol -> eV, the conversion constant used by ReaxFF codes.
KCAL_TO_EV = 4.3364432032e-2
# Coulomb constant e^2 / (4 pi eps_0) in eV * Angstrom.
KE = 14.39975840
# Small numerical guard applied to powers and square roots.
SAFETY = 1.0e-9

# Parameter names whose ffield values are in kcal/mol.
_KCAL_PARAMS = frozenset({"Desi", "Depi", "Depp", "lp2", "ovun5", "val1",
                          "coa1", "V1", "V2", "V3", "cot1", "pen1", "Devdw",
                          "Dehb"})

_P_GENERAL = ("boc1", "boc2", "coa2", "ovun6", "lp1", "ovun7", "ovun8",
              "val6", "val8", "val9", "val10", "tor2", "tor3", "tor4",
              "cot2", "coa3", "coa4", "ovun3", "ovun4", "pen2", "pen3",
              "pen4", "vdw1")
_P_SPECIES = ("val", "vale", "valang", "valboc", "lp2", "ovun2", "ovun5",
              "atomic", "chi", "mu", "gamma", "gammaw", "boc3", "boc4",
              "boc5", "val3", "val5")
_P_PAIR = ("Desi", "Depi", "Depp", "be1", "be2", "bo1", "bo2", "bo3", "bo4",
           "bo5", "bo6", "ovun1", "rosi", "ropi", "ropp", "Devdw", "rvdw",
           "alfa", "corr13", "ovcorr")
# Pair parameters needed for the nonbonded terms of *every* pair, bonded or
# not (filled by the geometric-mean combination rule when unlisted).
_P_PAIR_VDW = ("Devdw", "rvdw", "alfa")


def taper_up(x: Tensor, rmin: float, rmax: float) -> Tensor:
    """Ascending bond-order taper: 0 below ``rmin``, 1 above ``rmax``.

    The cubic switching polynomial applied to the raw bond-order
    exponentials, with the regularized denominator
    ``(rmin - rmax)^3 + 1e-7`` under which published ReaxFF-nn parameter
    libraries were trained -- for the tiny bond-order thresholds this taper
    is applied to, the constant dominates the cube, so it acts as a
    near-step at ``rmin`` rather than a smooth switch.

    Parameters
    ----------
    x : Tensor
        Values to taper (bond orders).
    rmin, rmax : float
        Lower and upper edge of the switching window.

    Returns
    -------
    Tensor
        Taper values.
    """
    above = (x > rmax).to(x.dtype)
    inside = (x <= rmax) & (x > rmin)
    x2 = torch.where(inside, x, torch.zeros_like(x))
    one2 = inside.to(x.dtype)
    rterm = 1.0 / ((rmin - rmax) ** 3 + 1.0e-7)
    rm = rmin * one2
    rd = rm - x2
    trm1 = rm + 2.0 * x2 - 3.0 * rmax * one2
    return rterm * rd * rd * trm1 + above


def taper_down(x: Tensor, rmin: float, rmax: float) -> Tensor:
    """Descending distance taper: 1 below ``rmin``, 0 above ``rmax``.

    The cubic switching polynomial of the hydrogen-bond donor--acceptor
    distance window.

    Parameters
    ----------
    x : Tensor
        Values to taper (distances).
    rmin, rmax : float
        Lower and upper edge of the switching window.

    Returns
    -------
    Tensor
        Taper values, elementwise in ``[0, 1]``.
    """
    below = (x < rmin).to(x.dtype)
    inside = (x <= rmax) & (x > rmin)
    x2 = torch.where(inside, x, torch.zeros_like(x))
    one2 = inside.to(x.dtype)
    rterm = 1.0 / (rmax - rmin) ** 3
    rm = rmax * one2
    rd = rm - x2
    trm1 = rm + 2.0 * x2 - 3.0 * rmin * one2
    return rterm * rd * rd * trm1 + below


def nonbonded_taper(r: Tensor, cutoff: float) -> Tensor:
    """The 7th-order ReaxFF nonbonded taper ``Tap(r)``.

    ``Tap = 1 - 35 (r/rc)^4 + 84 (r/rc)^5 - 70 (r/rc)^6 + 20 (r/rc)^7`` is 1
    at ``r = 0`` and has zero value and slope at the cutoff. Used for the van
    der Waals, Coulomb and EEM kernels.

    Parameters
    ----------
    r : Tensor
        Interatomic distances.
    cutoff : float
        The nonbonded cutoff radius ``rc``.

    Returns
    -------
    Tensor
        Taper values.
    """
    return (1.0 - 35.0 / cutoff ** 4 * r ** 4 + 84.0 / cutoff ** 5 * r ** 5
            - 70.0 / cutoff ** 6 * r ** 6 + 20.0 / cutoff ** 7 * r ** 7)


def guarded_exp(x: Tensor) -> Tensor:
    """``exp`` with the argument capped at 40.

    Every exponential this guards appears inside a saturating ratio (the
    ReaxFF correction and coordination functions), where capping the argument
    changes the result by less than 1e-17 -- but keeps single and double
    backward passes finite in float32 when training drives the model through
    strongly over/under-coordinated configurations.

    Parameters
    ----------
    x : Tensor
        Exponent.

    Returns
    -------
    Tensor
        ``exp(min(x, 40))``.
    """
    return torch.exp(torch.clamp(x, max=40.0))


def _lexsort(cols: Sequence[Tensor]) -> Tensor:
    """Indices that sort rows lexicographically by the given key columns.

    Parameters
    ----------
    cols : sequence of Tensor
        Key columns, highest priority first, each of shape ``(E,)``.

    Returns
    -------
    Tensor
        Permutation of ``arange(E)`` sorting the rows.
    """
    idx = torch.arange(cols[0].shape[0], device=cols[0].device)
    for c in reversed(cols):
        idx = idx[torch.argsort(c[idx], stable=True)]
    return idx


def reverse_edge_permutation(src: Tensor, dst: Tensor, shifts: Tensor) -> Tensor:
    """For each directed edge, the index of its reversed counterpart.

    Edge ``(src, dst, shift)`` is matched with ``(dst, src, -shift)``; the
    edge list must be closed under reversal (true for any neighbor list built
    from a symmetric distance criterion).

    Parameters
    ----------
    src, dst : Tensor
        Edge endpoint indices, shape ``(E,)``.
    shifts : Tensor
        Integer periodic image shifts, shape ``(E, 3)``.

    Returns
    -------
    Tensor
        ``rev`` of shape ``(E,)`` with ``rev[e]`` the index of the reverse of
        edge ``e``.
    """
    fwd = (src, dst, shifts[:, 0], shifts[:, 1], shifts[:, 2])
    bwd = (dst, src, -shifts[:, 0], -shifts[:, 1], -shifts[:, 2])
    ia = _lexsort(fwd)
    ib = _lexsort(bwd)
    rev = torch.empty_like(ia)
    rev[ib] = ia
    return rev


def _group_cross(key_a: Tensor, key_b: Tensor, n_keys: int
                 ) -> tuple[Tensor, Tensor]:
    """All index pairs ``(a, b)`` whose keys match: ``key_a[a] == key_b[b]``.

    Vectorised grouped cross-product, used to attach torsion end-atoms to a
    central bond and hydrogen-bond acceptors to a donor pair.

    Parameters
    ----------
    key_a, key_b : Tensor
        Integer group keys for the two index sets, shapes ``(A,)`` / ``(B,)``.
    n_keys : int
        Number of distinct key values (keys are ``0 <= key < n_keys``).

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(idx_a, idx_b)`` of equal length, enumerating every matching pair.
    """
    device = key_a.device
    order_a = torch.argsort(key_a, stable=True)
    order_b = torch.argsort(key_b, stable=True)
    cnt_a = torch.bincount(key_a, minlength=n_keys)
    cnt_b = torch.bincount(key_b, minlength=n_keys)
    off_a = torch.cumsum(cnt_a, 0) - cnt_a
    off_b = torch.cumsum(cnt_b, 0) - cnt_b
    n_pairs = cnt_a * cnt_b
    total = int(n_pairs.sum())
    if total == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    key = torch.repeat_interleave(torch.arange(n_keys, device=device), n_pairs)
    start = torch.cumsum(n_pairs, 0) - n_pairs
    pos = torch.arange(total, device=device) - start.repeat_interleave(n_pairs)
    a_local = torch.div(pos, cnt_b[key], rounding_mode="floor")
    b_local = pos - a_local * cnt_b[key]
    idx_a = order_a[off_a[key] + a_local]
    idx_b = order_b[off_b[key] + b_local]
    return idx_a, idx_b


def _mlp(x: Tensor, wi: Tensor, bi: Tensor, wh: Tensor, bh: Tensor,
         wo: Tensor, bo: Tensor) -> Tensor:
    """The all-sigmoid feed-forward network of ReaxFF-nn, per item.

    Every layer (input, hidden and output) applies a logistic sigmoid (Xue
    et al. 2021, eq 5). Weights are gathered per item (per edge / per atom)
    so that each chemical species or bond type uses its own network.

    Parameters
    ----------
    x : Tensor
        Inputs of shape ``(E, n_in)``.
    wi, bi : Tensor
        Input layer weights ``(E, n_in, W)`` and biases ``(E, W)``.
    wh, bh : Tensor
        Hidden layer weights ``(L, E, W, W)`` and biases ``(L, E, W)``.
    wo, bo : Tensor
        Output layer weights ``(E, W, n_out)`` and biases ``(E, n_out)``.

    Returns
    -------
    Tensor
        Network outputs of shape ``(E, n_out)``.
    """
    h = torch.sigmoid(torch.einsum("ei,eio->eo", x, wi) + bi)
    for layer in range(wh.shape[0]):
        h = torch.sigmoid(torch.einsum("ei,eio->eo", h, wh[layer]) + bh[layer])
    return torch.sigmoid(torch.einsum("ei,eio->eo", h, wo) + bo)


@register_model("reaxff")
class ReaxFF(InteratomicPotential):
    """ReaxFF / ReaxFF-nn reactive force field.

    The model is fully specified by a parameter library (``ffield`` text or
    ReaxFF-nn JSON); it evaluates the complete ReaxFF energy --
    bond, lone-pair, over/under-coordination, valence angle, penalty,
    three-body conjugation, torsion, four-body conjugation, hydrogen bond,
    tapered/shielded van der Waals and Coulomb terms with EEM-equilibrated
    charges -- and, in ``nn`` mode, the ReaxFF-nn message-passing bond-order
    and bond-energy networks. Forces and stress come from autograd via
    :class:`~xnn.common.models.outputs.ForceStressOutput`.

    Parameters
    ----------
    ffield : str or Path or FFieldLibrary
        The parameter library (path, or a pre-parsed
        :class:`~xnn.ffnn.models.ffield.FFieldLibrary`).
    species : sequence of str or int, optional
        Chemical symbols (or atomic numbers) this instance supports; must be
        a subset of the library's species. Default: every species in the
        library.
    nn : bool, optional
        Use the ReaxFF-nn neural bond blocks. Default: ``True`` exactly when
        the library carries network weights.
    messages : int, optional
        Number of message-passing steps ``T`` (nn mode). Default: the
        library's value.
    vdw_cutoff : float, optional
        Nonbonded (vdW / Coulomb / EEM) cutoff in Angstrom, and the model's
        neighbor-list ``cutoff``; by default 10.0, the ReaxFF standard.
    hb_short, hb_long : float, optional
        Donor--acceptor (X..Z) distance window of the hydrogen-bond taper, by
        default 6.75 and 7.5 Angstrom.
    trainable : sequence of str or "all", optional
        Which classical parameter groups to expose to the optimizer, by
        library name (e.g. ``("Desi", "be1", "V2")``; angle, torsion and
        hydrogen-bond groups are addressed as ``"ang_theta0"``, ``"tor_V2"``,
        ``"hb_Dehb"``, ...); ``"all"`` unfreezes every group. Network weights
        (nn mode) are always trainable. Default: none (a fixed classical
        force field).
    keep_intermediates : bool, optional
        If ``True``, stash the intermediate tensors of the last evaluation
        (bond orders, per-interaction energies, angle/torsion indices, ...)
        in ``self.intermediates`` for inspection -- used by the fidelity
        notebook. Default ``False``.

    Notes
    -----
    ``forward`` returns, besides the standard ``node_energy`` / ``energy`` /
    ``node_features`` keys, the EEM ``charges`` ``(N,)`` and one
    per-structure tensor per energy term (``e_bond``, ``e_lone``, ``e_over``,
    ``e_under``, ``e_angle``, ``e_penalty``, ``e_three_conj``, ``e_torsion``,
    ``e_four_conj``, ``e_vdw``, ``e_coulomb``, ``e_hbond``, ``e_self``).
    A per-structure ``total_charge`` attribute on the graph (shape ``(B,)``)
    is honored by the EEM solve; the default is charge neutrality.
    """

    def __init__(self, ffield: Union[str, Path, FFieldLibrary], *,
                 species: Optional[Sequence] = None,
                 nn: Optional[bool] = None,
                 messages: Optional[int] = None,
                 vdw_cutoff: float = 10.0,
                 hb_short: float = 6.75,
                 hb_long: float = 7.5,
                 trainable: Union[Sequence[str], str] = (),
                 keep_intermediates: bool = False):
        super().__init__()
        lib = ffield if isinstance(ffield, FFieldLibrary) else read_ffield(ffield)
        self.cutoff = float(vdw_cutoff)
        self.hb_short = float(hb_short)
        self.hb_long = float(hb_long)
        self.keep_intermediates = bool(keep_intermediates)
        self.intermediates: dict = {}

        if species is None:
            self.species = list(lib.spec)
        else:
            self.species = [CHEMICAL_SYMBOLS[s] if isinstance(s, int) else str(s)
                            for s in species]
            missing = [s for s in self.species if s not in lib.spec]
            if missing:
                raise ValueError(f"species {missing} not in the ffield library "
                                 f"(has {lib.spec})")

        self.nn = lib.is_nn if nn is None else bool(nn)
        self.messages = int(lib.messages if messages is None else messages)
        self.bo_function = int(lib.bo_function)
        self.energy_function = int(lib.energy_function)
        self.message_function = int(lib.message_function)
        if self.nn:
            # the neural vdW taper is only ever active when the library
            # actually carries vdW network weights
            if lib.vdw_function and lib.vdw_layer is not None:
                raise NotImplementedError("the neural vdW taper "
                                          "(VdwFunction > 0) is not implemented")
            if self.message_function not in (0, 1, 2, 3):
                raise NotImplementedError(
                    f"MessageFunction {self.message_function} is not implemented")
            if self.energy_function not in (0, 1, 2, 3, 5):
                raise NotImplementedError(
                    f"EnergyFunction {self.energy_function} is not implemented")
            if self.bo_function not in (0, 1, 2):
                raise NotImplementedError(
                    f"BOFunction {self.bo_function} is not implemented")
        self.bo_layer = lib.bo_layer
        self.mf_layer = lib.mf_layer
        self.be_layer = lib.be_layer

        p = dict(lib.p)
        bonds = list(lib.bonds)
        hbs = list(lib.hbs)
        complete_off_diagonal(p, lib.spec, bonds)
        complete_hbonds(p, lib.spec, hbs)

        self.botol = 0.01 * p["cutoff"]
        self.atol = p["acut"]
        self.hbtol = p["hbtol"]

        # species -> model index; Z -> model index
        z_to_index = torch.full((len(CHEMICAL_SYMBOLS),), -1, dtype=torch.long)
        for i, sym in enumerate(self.species):
            z_to_index[SYMBOL_TO_Z[sym]] = i
        self.register_buffer("z_to_index", z_to_index)
        self._h_index = self.species.index("H") if "H" in self.species else -1

        self._assemble_parameters(p, bonds, lib, trainable)
        if self.nn:
            self._assemble_networks(lib.m or {})
        self.node_feature_dim = 3

    # ------------------------------------------------------------------
    # parameter assembly
    # ------------------------------------------------------------------
    def _assemble_parameters(self, p: dict, bonds: list, lib: FFieldLibrary,
                             trainable: Union[Sequence[str], str]) -> None:
        """Build the dense parameter tensors from the flat library dict.

        Per-species vectors ``(S,)``, pair matrices ``(S, S)``,
        valence-angle tensors ``(S, S, S)``, torsion tensors ``(S, S, S, S)``
        (wildcards resolved) and hydrogen-bond tensors ``(S, S, S)`` are
        assembled in eV/Angstrom units; the bond-order cutoff matrix ``rcbo``
        and the valence cutoff matrix ``rcuta`` are precomputed as buffers.

        Parameters
        ----------
        p : dict
            Completed flat parameter dictionary.
        bonds : list[str]
            Bond types with explicit bond parameters.
        lib : FFieldLibrary
            The source library (for cutoff tables and type lists).
        trainable : sequence of str or str
            Parameter groups to leave trainable.
        """
        spec = self.species
        S = len(spec)
        dtype = torch.get_default_dtype()
        if isinstance(trainable, str):
            trainable_all = trainable.lower() == "all"
            trainable_set = set() if trainable_all else {trainable}
        else:
            trainable_all = False
            trainable_set = set(trainable)

        def unit(name):
            """Unit conversion factor for parameter ``name``."""
            return KCAL_TO_EV if name in _KCAL_PARAMS else 1.0

        def make_param(name, tensor):
            """Register ``tensor`` as the (possibly trainable) group ``name``."""
            req = trainable_all or name in trainable_set
            self.params[name] = Parameter(tensor, requires_grad=req)

        self.params = ParameterDict()
        for name in _P_GENERAL:
            make_param(name, torch.tensor(float(p[name]) * unit(name),
                                          dtype=dtype))

        for name in _P_SPECIES:
            vals = [float(p[f"{name}_{sp}"]) * unit(name) for sp in spec]
            make_param(name, torch.tensor(vals, dtype=dtype))

        bond_set = set(bonds)
        has_bond = torch.zeros(S, S, dtype=torch.bool)
        for i, a in enumerate(spec):
            for j, b in enumerate(spec):
                has_bond[i, j] = (f"{a}-{b}" in bond_set) or (f"{b}-{a}" in bond_set)
        self.register_buffer("has_bond", has_bond)

        for name in _P_PAIR:
            mat = torch.zeros(S, S, dtype=dtype)
            for i, a in enumerate(spec):
                for j, b in enumerate(spec):
                    key = f"{name}_{a}-{b}"
                    if key not in p:
                        key = f"{name}_{b}-{a}"
                    if key in p:
                        mat[i, j] = float(p[key]) * unit(name)
                    elif name in _P_PAIR_VDW:
                        # combination rule for non-bonded-only pairs
                        va, vb = float(p[f"{name}_{a}"]), float(p[f"{name}_{b}"])
                        mat[i, j] = (va * vb) ** 0.5 * unit(name) \
                            if va > 0.0 and vb > 0.0 else 0.0
                    elif name in ("rosi", "ropi", "ropp"):
                        mat[i, j] = 1.0     # inert placeholder, no bond formed
            make_param(name, mat)

        # Angle parameters convert kcal -> eV only for angle types listed in
        # the library (those with a `theta0` entry); values of unlisted types
        # are used raw. Published ReaxFF-nn parameter libraries were trained
        # under this convention, so it is load-bearing when evaluating them.
        listed_angles = set(lib.angs)
        for name in P_ANGLE:
            ten = torch.zeros(S, S, S, dtype=dtype)
            for i, a in enumerate(spec):
                for j, b in enumerate(spec):
                    for k, c in enumerate(spec):
                        key = f"{name}_{a}-{b}-{c}"
                        ang_type = f"{a}-{b}-{c}"
                        if key not in p:
                            key = f"{name}_{c}-{b}-{a}"
                            ang_type = f"{c}-{b}-{a}"
                        u = unit(name) if ang_type in listed_angles else 1.0
                        ten[i, j, k] = float(p.get(key, 0.0)) * u
            make_param(f"ang_{name}", ten)

        for name in P_TORSION:
            ten = torch.zeros(S, S, S, S, dtype=dtype)
            for i, a in enumerate(spec):
                for j, b in enumerate(spec):
                    for k, c in enumerate(spec):
                        for l, d in enumerate(spec):
                            tor = f"{a}-{b}-{c}-{d}"
                            ten[i, j, k, l] = resolve_torsion(
                                p, lib.torp, tor, name) * unit(name)
            make_param(f"tor_{name}", ten)

        for name in P_HBOND:
            ten = torch.zeros(S, S, S, dtype=dtype)
            if self._h_index >= 0:
                for i, a in enumerate(spec):
                    for k, c in enumerate(spec):
                        key = f"{name}_{a}-H-{c}"
                        if key in p:
                            ten[i, self._h_index, k] = float(p[key]) * unit(name)
            make_param(f"hb_{name}", ten)

        # --- cutoff tables (fixed buffers) ---
        rcut = cutoff_table(lib.rcut, "rcut", spec)
        rcuta = cutoff_table(lib.rcuta, "rcuta", spec)
        rcut_m = torch.zeros(S, S, dtype=dtype)
        rcuta_m = torch.zeros(S, S, dtype=dtype)
        rcbo = torch.zeros(S, S, dtype=dtype)
        log_botol = math.log(self.botol / (1.0 + self.botol))
        for i, a in enumerate(spec):
            for j, b in enumerate(spec):
                rcut_m[i, j] = rcut[f"{a}-{b}"]
                rcuta_m[i, j] = rcuta[f"{a}-{b}"]
                if has_bond[i, j]:
                    bd = f"{a}-{b}" if f"{a}-{b}" in bond_set else f"{b}-{a}"
                    ref = a if a == b else bd
                    rc = p[f"rosi_{ref}"] * (log_botol / p[f"bo1_{bd}"]) \
                        ** (1.0 / p[f"bo2_{bd}"])
                    rcbo[i, j] = min(rcut[f"{a}-{b}"], rc)
        self.register_buffer("rcut_pair", rcut_m)
        self.register_buffer("rcuta_pair", rcuta_m)
        self.register_buffer("rcbo", rcbo)

    def _assemble_networks(self, m: dict) -> None:
        """Build the ReaxFF-nn weight tensors from the library's ``m`` dict.

        Weights are stacked per species (message network ``fm``) or per bond
        pair (bond-order networks ``fsi``/``fpi``/``fpp`` and bond-energy
        network ``fe``) into dense tensors indexed by species, so a whole
        batch of edges evaluates its per-type networks with one gather and a
        batched matmul.

        Parameters
        ----------
        m : dict
            The library's weight dictionary (``"fmwi_C"``, ``"few_C-H"``,
            nested lists).
        """
        spec = self.species
        S = len(spec)
        dtype = torch.get_default_dtype()
        self.weights = ParameterDict()

        def get(key, a, b=None):
            """Fetch weight ``key`` for species ``a`` (or pair ``a-b``)."""
            if b is None:
                return torch.tensor(m[f"{key}_{a}"], dtype=dtype)
            for bd in (f"{a}-{b}", f"{b}-{a}"):
                if f"{key}_{bd}" in m:
                    return torch.tensor(m[f"{key}_{bd}"], dtype=dtype)
            raise KeyError(f"{key}_{a}-{b}")

        def stack_species(prefix, n_hidden):
            """Stack the per-species network ``prefix`` into dense tensors."""
            wi = torch.stack([get(prefix + "wi", sp) for sp in spec])
            bi = torch.stack([get(prefix + "bi", sp) for sp in spec])
            wo = torch.stack([get(prefix + "wo", sp) for sp in spec])
            bo = torch.stack([get(prefix + "bo", sp) for sp in spec])
            width = wi.shape[-1]
            if n_hidden:
                wh = torch.stack([torch.stack([get(prefix + "w", sp)[layer]
                                               for sp in spec])
                                  for layer in range(n_hidden)])
                bh = torch.stack([torch.stack([get(prefix + "b", sp)[layer]
                                               for sp in spec])
                                  for layer in range(n_hidden)])
            else:
                wh = torch.zeros(0, S, width, width, dtype=dtype)
                bh = torch.zeros(0, S, width, dtype=dtype)
            for name, t in (("wi", wi), ("bi", bi), ("w", wh), ("b", bh),
                            ("wo", wo), ("bo", bo)):
                self.weights[f"{prefix}{name}"] = Parameter(t)

        def stack_pairs(prefix, n_hidden):
            """Stack the per-bond-pair network ``prefix`` into dense tensors."""
            wi = torch.stack([torch.stack([get(prefix + "wi", a, b)
                                           for b in spec]) for a in spec])
            bi = torch.stack([torch.stack([get(prefix + "bi", a, b)
                                           for b in spec]) for a in spec])
            wo = torch.stack([torch.stack([get(prefix + "wo", a, b)
                                           for b in spec]) for a in spec])
            bo = torch.stack([torch.stack([get(prefix + "bo", a, b)
                                           for b in spec]) for a in spec])
            width = wi.shape[-1]
            if n_hidden:
                wh = torch.stack([torch.stack(
                    [torch.stack([get(prefix + "w", a, b)[layer] for b in spec])
                     for a in spec]) for layer in range(n_hidden)])
                bh = torch.stack([torch.stack(
                    [torch.stack([get(prefix + "b", a, b)[layer] for b in spec])
                     for a in spec]) for layer in range(n_hidden)])
            else:
                wh = torch.zeros(0, S, S, width, width, dtype=dtype)
                bh = torch.zeros(0, S, S, width, dtype=dtype)
            for name, t in (("wi", wi), ("bi", bi), ("w", wh), ("b", bh),
                            ("wo", wo), ("bo", bo)):
                self.weights[f"{prefix}{name}"] = Parameter(t)

        stack_species("fm", self.mf_layer[1])
        stack_pairs("fe", self.be_layer[1])
        if self.bo_function in (1, 2):
            for prefix in ("fsi", "fpi", "fpp"):
                stack_pairs(prefix, self.bo_layer[1])

    # ------------------------------------------------------------------
    # parameter access helpers
    # ------------------------------------------------------------------
    def _pair(self, name: str) -> Tensor:
        """Pair parameter matrix ``(S, S)``, symmetrized to tie gradients."""
        mat = self.params[name]
        return 0.5 * (mat + mat.transpose(0, 1))

    def _comb(self, name: str) -> Tensor:
        """Geometric-mean pair combination of a per-species parameter."""
        v = self.params[name]
        return torch.sqrt(v[:, None] * v[None, :])

    def _ang(self, name: str) -> Tensor:
        """Angle parameter tensor ``(S, S, S)``, i--k symmetrized."""
        ten = self.params[f"ang_{name}"]
        return 0.5 * (ten + ten.permute(2, 1, 0))

    def _tor(self, name: str) -> Tensor:
        """Torsion parameter tensor ``(S, S, S, S)``, reversal-symmetrized."""
        ten = self.params[f"tor_{name}"]
        return 0.5 * (ten + ten.permute(3, 2, 1, 0))

    def _net(self, prefix: str, idx_a: Tensor, idx_b: Optional[Tensor],
             x: Tensor) -> Tensor:
        """Evaluate the per-type network ``prefix`` on gathered items.

        Parameters
        ----------
        prefix : str
            Network name (``"fm"``, ``"fe"``, ``"fsi"``, ...).
        idx_a : Tensor
            Species index of the first (or only) type axis, per item.
        idx_b : Tensor or None
            Species index of the second type axis (pair networks; the weight
            tensors are symmetrized over the pair axes to tie gradients), or
            ``None`` for per-species networks.
        x : Tensor
            Network inputs of shape ``(E, n_in)``.

        Returns
        -------
        Tensor
            Network outputs of shape ``(E, n_out)``.
        """
        w = self.weights
        if idx_b is None:
            wi, bi = w[prefix + "wi"][idx_a], w[prefix + "bi"][idx_a]
            wo, bo = w[prefix + "wo"][idx_a], w[prefix + "bo"][idx_a]
            wh, bh = w[prefix + "w"][:, idx_a], w[prefix + "b"][:, idx_a]
        else:
            def sym(t, axis):
                """Symmetrize the two pair axes ``axis``/``axis+1`` of ``t``."""
                perm = list(range(t.dim()))
                perm[axis], perm[axis + 1] = perm[axis + 1], perm[axis]
                return 0.5 * (t + t.permute(*perm))
            wi = sym(w[prefix + "wi"], 0)[idx_a, idx_b]
            bi = sym(w[prefix + "bi"], 0)[idx_a, idx_b]
            wo = sym(w[prefix + "wo"], 0)[idx_a, idx_b]
            bo = sym(w[prefix + "bo"], 0)[idx_a, idx_b]
            wh = sym(w[prefix + "w"], 1)[:, idx_a, idx_b]
            bh = sym(w[prefix + "b"], 1)[:, idx_a, idx_b]
        return _mlp(x, wi, bi, wh, bh, wo, bo)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, data: AtomicGraph) -> dict[str, Tensor]:
        """Evaluate the ReaxFF energy on a (batched) atomic graph.

        Parameters
        ----------
        data : AtomicGraph
            The batched atomic graph; its neighbor list must have been built
            with this model's ``cutoff`` (the nonbonded cutoff).

        Returns
        -------
        dict of str to Tensor
            ``node_energy`` ``(N,)``, ``energy`` ``(B,)``, ``charges``
            ``(N,)``, ``node_features`` ``(N, 3)`` and the per-structure
            energy decomposition (see the class docstring).
        """
        N = data.num_nodes
        P = self.params
        inter: dict = {}

        s = self.z_to_index[data.atomic_numbers]
        if bool((s < 0).any()):
            bad = sorted(set(int(z) for z in data.atomic_numbers[s < 0].tolist()))
            raise ValueError(f"atomic numbers {bad} are not covered by this "
                             f"ffield (species {self.species})")

        src, dst = data.edge_index[0], data.edge_index[1]
        vec = data.edge_vectors()
        r = torch.sqrt((vec * vec).sum(dim=1) + SAFETY)
        si, sj = s[dst], s[src]

        # --- EEM charges and the shielded Coulomb kernel -------------------
        gamma_e = self._comb("gamma")[si, sj]
        gm3 = (1.0 / gamma_e) ** 3
        rth = (r ** 3 + gm3) ** (1.0 / 3.0)        # shielded distance kernel
        tap = nonbonded_taper(r, self.cutoff)
        q = self._eem_charges(data, s, tap / rth)
        inter["q"] = q

        # --- bonded subset ---------------------------------------------------
        bmask = (r < self.rcbo[si, sj]) & (src != dst)
        b_src, b_dst = src[bmask], dst[bmask]
        b_r = r[bmask]
        b_vec = vec[bmask]
        b_shift = data.cell_shifts[bmask]
        bsi, bsj = s[b_dst], s[b_src]

        def pb(name):
            """Gather pair parameter ``name`` per bonded edge."""
            return self._pair(name)[bsi, bsj]

        # --- uncorrected bond orders (sigma / pi / double-pi), eq 2 ----------
        eterm1 = (1.0 + self.botol) * torch.exp(
            pb("bo1") * (b_r / pb("rosi")) ** pb("bo2"))
        eterm2 = torch.exp(pb("bo3") * (b_r / pb("ropi")) ** pb("bo4"))
        eterm3 = torch.exp(pb("bo5") * (b_r / pb("ropp")) ** pb("bo6"))
        if self.nn and self.bo_function in (1, 2):
            sign = 1.0 if self.bo_function == 1 else -1.0
            bop_si = self._net("fsi", bsi, bsj, sign * eterm1[:, None])[:, 0]
            bop_pi = self._net("fpi", bsi, bsj, sign * eterm2[:, None])[:, 0]
            bop_pp = self._net("fpp", bsi, bsj, sign * eterm3[:, None])[:, 0]
        else:
            bop_si = taper_up(eterm1, self.botol, 2.0 * self.botol) \
                * (eterm1 - self.botol)
            bop_pi = taper_up(eterm2, self.botol, 2.0 * self.botol) * eterm2
            bop_pp = taper_up(eterm3, self.botol, 2.0 * self.botol) * eterm3
        bop = bop_si + bop_pi + bop_pp
        deltap = scatter_sum(bop, b_dst, N)
        inter.update(bond_mask=bmask, eterm1=eterm1, eterm2=eterm2,
                     eterm3=eterm3, bop_si=bop_si, bop_pi=bop_pi,
                     bop_pp=bop_pp, bop=bop, deltap=deltap)

        # --- corrected bond orders -------------------------------------------
        if self.nn:
            bo0, bosi, bopi, bopp = self._message_passing(
                s, b_src, b_dst, b_shift, bop, bop_si, bop_pi, bop_pp, N)
        else:
            bo0, bosi, bopi, bopp = self._bond_order_corrections(
                s, b_src, b_dst, bop, bop_si, bop_pi, bop_pp, deltap, inter)
        bo = torch.relu(bo0 - self.atol)
        delta = scatter_sum(bo0, b_dst, N)
        dv = delta - P["val"][s]
        dpi = scatter_sum(bopi + bopp, b_dst, N)
        so = scatter_sum(pb("ovun1") * pb("Desi") * bo0, b_dst, N)
        fbo = taper_up(bo0, self.atol, 2.0 * self.atol)
        fhb = taper_up(bo0, self.hbtol, 2.0 * self.hbtol)
        inter.update(bo0=bo0, bo=bo, bosi=bosi, bopi=bopi, bopp=bopp,
                     delta=delta, dpi=dpi)

        # --- bond energy -------------------------------------------------------
        esi = self._bond_energy(bsi, bsj, bo0, bosi, bopi, bopp, pb)
        e_bond_node = scatter_sum(-0.5 * esi, b_dst, N)
        inter["esi"] = esi

        # --- lone pair / over- / under-coordination ----------------------------
        vale, val = P["vale"][s], P["val"][s]
        delta_e = 0.5 * (delta - vale)
        de = torch.relu(-torch.ceil(delta_e))
        nlp = de + torch.exp(-P["lp1"] * 4.0 * (1.0 + delta_e + de) ** 2)
        nlpopt = 0.5 * (vale - val)
        delta_lp = nlpopt - nlp
        dlp = delta - val - delta_lp
        dpil = scatter_sum(dlp[b_src] * (bopi + bopp), b_dst, N)
        # delta_lp / (1 + exp(-75 delta_lp)), written with the (exact,
        # overflow-safe) logistic sigmoid
        e_lone_node = P["lp2"][s] * delta_lp * torch.sigmoid(75.0 * delta_lp)

        lpcorr = delta_lp / (1.0 + P["ovun3"] * guarded_exp(P["ovun4"] * dpil))
        delta_lpcorr = dv - lpcorr
        otrm1 = 1.0 / (delta_lpcorr + val)
        otrm2 = torch.sigmoid(-P["ovun2"][s] * delta_lpcorr)
        e_over_node = so * otrm1 * delta_lpcorr * otrm2

        expeu1 = guarded_exp(P["ovun6"] * delta_lpcorr)
        eu1 = torch.sigmoid(P["ovun2"][s] * delta_lpcorr)
        eu2 = 1.0 / (1.0 + P["ovun7"] * guarded_exp(P["ovun8"] * dpil))
        e_under_node = -P["ovun5"][s] * (1.0 - expeu1) * eu1 * eu2
        inter.update(nlp=nlp, delta_lp=delta_lp, dpil=dpil,
                     delta_lpcorr=delta_lpcorr)

        # --- valence angles (+ penalty + three-body conjugation) ---------------
        amask = b_r < self.rcuta_pair[bsi, bsj]
        e_ang_node, e_pen_node, e_tcon_node = self._angle_terms(
            s, N, b_src, b_dst, b_vec, b_r, amask, bo, fbo, delta, dv, dpi,
            nlp, inter)

        # --- torsions (+ four-body conjugation) ---------------------------------
        e_tor_node, e_fcon_node = self._torsion_terms(
            s, N, b_src, b_dst, b_vec, b_r, b_shift, amask, bo, bopi, fbo,
            delta, inter)

        # --- hydrogen bonds ------------------------------------------------------
        e_hb_node = self._hbond_terms(s, N, src, dst, vec, r, b_src, b_dst,
                                      b_vec, b_r, amask, bo0, fhb, inter)

        # --- van der Waals + Coulomb ----------------------------------------------
        f13 = (r ** P["vdw1"]
               + (1.0 / self._comb("gammaw")[si, sj]) ** P["vdw1"]) \
            ** (1.0 / P["vdw1"])
        expvdw1 = torch.exp(0.5 * self._pair("alfa")[si, sj]
                            * (1.0 - f13 / (2.0 * self._pair("rvdw")[si, sj])))
        evdw_e = tap * self._pair("Devdw")[si, sj] * (expvdw1 ** 2 - 2.0 * expvdw1)
        ecoul_e = tap * KE * q[dst] * q[src] / rth
        e_vdw_node = scatter_sum(0.5 * evdw_e, dst, N)
        e_coul_node = scatter_sum(0.5 * ecoul_e, dst, N)
        inter.update(f13=f13, evdw=evdw_e, ecoul=ecoul_e)

        # --- charge self energy + atomic reference ----------------------------------
        e_self_node = q * (P["chi"][s] + q * P["mu"][s])
        e_atomic_node = -P["atomic"][s]

        node_energy = (e_bond_node + e_lone_node + e_over_node + e_under_node
                       + e_ang_node + e_pen_node + e_tcon_node
                       + e_tor_node + e_fcon_node
                       + e_vdw_node + e_coul_node + e_hb_node
                       + e_self_node + e_atomic_node)
        energy = self.aggregate_energy(node_energy, data)

        out = {
            "node_energy": node_energy,
            "energy": energy,
            "charges": q,
            "node_features": torch.stack([delta, nlp, q], dim=1),
            "e_bond": self.aggregate_energy(e_bond_node, data),
            "e_lone": self.aggregate_energy(e_lone_node, data),
            "e_over": self.aggregate_energy(e_over_node, data),
            "e_under": self.aggregate_energy(e_under_node, data),
            "e_angle": self.aggregate_energy(e_ang_node, data),
            "e_penalty": self.aggregate_energy(e_pen_node, data),
            "e_three_conj": self.aggregate_energy(e_tcon_node, data),
            "e_torsion": self.aggregate_energy(e_tor_node, data),
            "e_four_conj": self.aggregate_energy(e_fcon_node, data),
            "e_vdw": self.aggregate_energy(e_vdw_node, data),
            "e_coulomb": self.aggregate_energy(e_coul_node, data),
            "e_hbond": self.aggregate_energy(e_hb_node, data),
            "e_self": self.aggregate_energy(e_self_node, data),
        }
        if self.keep_intermediates:
            self.intermediates = inter
        return out

    # ------------------------------------------------------------------
    # blocks
    # ------------------------------------------------------------------
    def _correction_factors(self, s: Tensor, b_src: Tensor, b_dst: Tensor,
                            bop: Tensor, deltap: Tensor
                            ) -> tuple[Tensor, Tensor, Tensor]:
        """The classical bond-order correction factors ``f1``, ``f4``, ``f5``.

        ``f1`` (from ``f2``/``f3``, eqs 3b-d of van Duin 2001) corrects for
        over-coordination of the two atoms; ``f4``/``f5`` (eqs 3e-f) remove
        residual 1-3 bond orders, with the ``boc3/4/5`` pair values combined
        geometrically from the atomic parameters.

        Parameters
        ----------
        s : Tensor
            Species index per atom.
        b_src, b_dst : Tensor
            Bonded-edge endpoints.
        bop : Tensor
            Uncorrected total bond order per edge.
        deltap : Tensor
            Uncorrected coordination ``Delta'`` per atom.

        Returns
        -------
        tuple of Tensor
            ``(f1, f4, f5)`` per bonded edge.
        """
        P = self.params
        bsi, bsj = s[b_dst], s[b_src]
        dv_atoms = deltap - P["val"][s]
        f2 = guarded_exp(-P["boc1"] * dv_atoms[b_dst]) \
            + guarded_exp(-P["boc1"] * dv_atoms[b_src])
        f3 = (-1.0 / P["boc2"]) * torch.log(
            0.5 * (guarded_exp(-P["boc2"] * dv_atoms[b_dst])
                   + guarded_exp(-P["boc2"] * dv_atoms[b_src])))
        val_i, val_j = P["val"][bsi], P["val"][bsj]
        f1 = 0.5 * ((val_i + f2) / (val_i + f2 + f3)
                    + (val_j + f2) / (val_j + f2 + f3))

        d_boc = deltap - P["valboc"][s]
        boc3 = self._comb("boc3")[bsi, bsj]
        boc4 = self._comb("boc4")[bsi, bsj]
        boc5 = self._comb("boc5")[bsi, bsj]
        f4 = torch.sigmoid(boc3 * (boc4 * bop ** 2 - d_boc[b_dst]) - boc5)
        f5 = torch.sigmoid(boc3 * (boc4 * bop ** 2 - d_boc[b_src]) - boc5)
        return f1, f4, f5

    def _bond_order_corrections(self, s, b_src, b_dst, bop, bop_si, bop_pi,
                                bop_pp, deltap, inter):
        """Classical corrected bond orders (van Duin 2001 eq 3a).

        The ``ovcorr`` / ``corr13`` pair flags select which corrections
        apply; the sigma component is recovered as the remainder
        ``BO - BO_pi - BO_pipi``.

        Parameters
        ----------
        s : Tensor
            Species index per atom.
        b_src, b_dst : Tensor
            Bonded-edge endpoints.
        bop, bop_si, bop_pi, bop_pp : Tensor
            Uncorrected bond orders per edge.
        deltap : Tensor
            Uncorrected coordination per atom.
        inter : dict
            Intermediate stash.

        Returns
        -------
        tuple of Tensor
            ``(bo0, bosi, bopi, bopp)`` corrected bond orders per edge.
        """
        bsi, bsj = s[b_dst], s[b_src]
        f1, f4, f5 = self._correction_factors(s, b_src, b_dst, bop, deltap)
        ovcorr = self._pair("ovcorr")[bsi, bsj]
        corr13 = self._pair("corr13")[bsi, bsj]
        one = torch.ones_like(f1)
        f11 = torch.where(ovcorr >= 0.0001, f1, one)
        f12 = torch.where((ovcorr >= 0.0001) & (corr13 >= 0.0001), f1, one)
        f45 = torch.where(corr13 >= 0.0001, f4 * f5, one)
        F = f11 * f12 * f45
        bo0 = bop * f11 * f45
        bopi = bop_pi * F
        bopp = bop_pp * F
        bosi = torch.relu(bo0 - bopi - bopp)
        inter.update(f1=f1, f4=f4, f5=f5)
        return bo0, bosi, bopi, bopp

    def _message_passing(self, s, b_src, b_dst, b_shift, bop, bop_si,
                         bop_pi, bop_pp, N):
        """ReaxFF-nn message passing over the uncorrected bond orders.

        Implements the ReaxFF-nn message-function variants 0-3: 0 applies the
        closed-form ``f1^2 f4 f5`` correction each step; 1-3 evaluate the
        per-species network ``fm`` on each directed edge and multiply the two
        directed messages of a bond (Xue et al. 2021, eqs 8-10). Message 2
        *replaces* the state with the network output; 1 and 3 rescale it.

        Parameters
        ----------
        s : Tensor
            Species index per atom.
        b_src, b_dst : Tensor
            Bonded-edge endpoints.
        b_shift : Tensor
            Integer image shifts of the bonded edges (to pair each directed
            edge with its reverse).
        bop, bop_si, bop_pi, bop_pp : Tensor
            Uncorrected bond orders per edge (the ``t = 0`` state).
        N : int
            Number of atoms.

        Returns
        -------
        tuple of Tensor
            ``(bo0, bosi, bopi, bopp)`` final-state bond orders per edge.
        """
        rev = reverse_edge_permutation(b_src, b_dst, b_shift)
        bsi = s[b_dst]
        h = bop
        hsi, hpi, hpp = bop_si, bop_pi, bop_pp
        d = scatter_sum(h, b_dst, N)
        d_si = scatter_sum(hsi, b_dst, N)
        d_pi = scatter_sum(hpi, b_dst, N)
        d_pp = scatter_sum(hpp, b_dst, N)

        for _ in range(self.messages):
            replace = False
            if self.message_function == 0:
                deltap = scatter_sum(bop, b_dst, N)
                f1, f4, f5 = self._correction_factors(s, b_src, b_dst, bop,
                                                      deltap)
                fsi = fpi = fpp = f1 * f1 * f4 * f5
            elif self.message_function == 1:
                x = torch.stack([d_si[b_src] - hsi, d_pi[b_src] - hpi,
                                 d_pp[b_src] - hpp, h,
                                 d_pp[b_dst] - hpp, d_pi[b_dst] - hpi,
                                 d_si[b_dst] - hsi], dim=1)
                f_dir = self._net("fm", bsi, None, x)
                f = f_dir * f_dir[rev]
                fsi, fpi, fpp = f[:, 0], f[:, 1], f[:, 2]
            else:                                     # message functions 2, 3
                x = torch.stack([d[b_dst] - h, h, d[b_src] - h], dim=1)
                f_dir = self._net("fm", bsi, None, x)
                f = f_dir * f_dir[rev]
                fsi, fpi, fpp = f[:, 0], f[:, 1], f[:, 2]
                replace = self.message_function == 2
            if replace:
                hsi, hpi, hpp = fsi, fpi, fpp
            else:
                hsi, hpi, hpp = hsi * fsi, hpi * fpi, hpp * fpp
            h = hsi + hpi + hpp
            d = scatter_sum(h, b_dst, N)
            d_si = scatter_sum(hsi, b_dst, N)
            d_pi = scatter_sum(hpi, b_dst, N)
            d_pp = scatter_sum(hpp, b_dst, N)
        return h, hsi, hpi, hpp

    def _bond_energy(self, bsi, bsj, bo0, bosi, bopi, bopp,
                     pb: Callable[[str], Tensor]) -> Tensor:
        """Per-edge bond energy ``e_si`` (positive; ``E_bond = -1/2 sum``).

        Classical form (van Duin 2001 eq 5) or, in nn mode, the
        ``EnergyFunction`` selected by the library: 0 (guarded classical),
        1/2 (per-bond network on the +-bond orders, Xue 2021 eq 12), 3
        (network scaled by the total bond order) or 5 (linear).

        Parameters
        ----------
        bsi, bsj : Tensor
            Species indices per bonded edge.
        bo0, bosi, bopi, bopp : Tensor
            Corrected bond orders per edge.
        pb : callable
            Pair-parameter gather (``name -> per-edge values``).

        Returns
        -------
        Tensor
            ``e_si`` per directed bonded edge, in eV.
        """
        if not self.nn:
            powb = (bosi + SAFETY) ** pb("be2")
            expb = torch.exp(pb("be1") * (1.0 - powb))
            return pb("Desi") * bosi * expb + pb("Depi") * bopi \
                + pb("Depp") * bopp
        ef = self.energy_function
        if ef == 0:
            fc = (bosi < 0.000001).to(bosi.dtype)
            powb = (bosi + fc) ** pb("be2")
            expb = torch.exp(pb("be1") * (1.0 - powb)) * (1.0 - fc)
            return pb("Desi") * bosi * expb + pb("Depi") * bopi \
                + pb("Depp") * bopp
        if ef in (1, 2):
            sign = 1.0 if ef == 1 else -1.0
            e = self._net("fe", bsi, bsj,
                          sign * torch.stack([bosi, bopi, bopp], dim=1))[:, 0]
            return pb("Desi") * e * (bo0 >= 0.0000001).to(e.dtype)
        if ef == 3:
            e = self._net("fe", bsi, bsj,
                          torch.stack([bosi, bopi, bopp], dim=1))[:, 0]
            return pb("Desi") * bo0 * e
        # energy function 5 (linear; guarded in __init__)
        return pb("Desi") * bosi + pb("Depi") * bopi - pb("Depp") * bopp

    def _angle_terms(self, s, N, b_src, b_dst, b_vec, b_r, amask, bo, fbo,
                     delta, dv, dpi, nlp, inter):
        """Valence angle, penalty and three-body-conjugation energies.

        Angles ``i-j-k`` are every unordered pair of valence neighbors of a
        non-hydrogen center ``j`` (neighbors within the per-pair valence
        cutoff ``rcuta``). Implements eqs 8a-d (angle), 9a-b (penalty, e.g.
        allene) and the three-body conjugation term of van Duin 2001 /
        Senftle 2016, with the equilibrium angle driven by the pi-bond
        environment ``SBO`` of the center.

        Parameters
        ----------
        s : Tensor
            Species index per atom.
        N : int
            Number of atoms.
        b_src, b_dst, b_vec, b_r : Tensor
            The bonded-edge arrays.
        amask : Tensor
            Boolean mask selecting valence (``rcuta``) edges among the
            bonded edges.
        bo, fbo : Tensor
            Per-bonded-edge cut bond order and the angle bond-order taper.
        delta, dv, dpi, nlp : Tensor
            Per-atom coordination quantities.
        inter : dict
            Intermediate stash.

        Returns
        -------
        tuple of Tensor
            ``(e_angle, e_penalty, e_three_conj)`` per-atom energies (on the
            angle centers).
        """
        P = self.params
        zero = torch.zeros(N, dtype=delta.dtype, device=delta.device)
        a_ids = torch.nonzero(amask, as_tuple=False)[:, 0]
        if a_ids.numel() == 0:
            return zero, zero.clone(), zero.clone()
        a_edge_index = torch.stack([b_src[a_ids], b_dst[a_ids]], dim=0)
        e1_loc, e2_loc, center = build_triplets(a_edge_index, N)
        keep = s[center] != self._h_index
        e1_loc, e2_loc, center = e1_loc[keep], e2_loc[keep], center[keep]
        if center.numel() == 0:
            return zero, zero.clone(), zero.clone()
        e1, e2 = a_ids[e1_loc], a_ids[e2_loc]          # bonded-edge indices
        ai, ak = b_src[e1], b_src[e2]                   # end atoms i, k

        # geometry: r_ij, r_jk from the two edges; r_ik composed
        rij, rjk = b_r[e1], b_r[e2]
        vik = b_vec[e1] - b_vec[e2]                     # (x_j-x_i) - (x_j-x_k)
        rik2 = (vik * vik).sum(dim=1)
        cos_theta = (rij ** 2 + rjk ** 2 - rik2) / (2.0 * rij * rjk)
        # clamp inside the arccos domain; the bound backs off far enough from
        # +-1 for the running dtype that the backward pass stays finite
        bound = min(0.9999999999,
                    1.0 - 4.0 * torch.finfo(cos_theta.dtype).eps)
        theta = torch.arccos(torch.clamp(cos_theta, -bound, bound))

        sa_i, sa_j, sa_k = s[ai], s[center], s[ak]

        def pa(name):
            """Gather angle parameter ``name`` per angle."""
            return self._ang(name)[sa_i, sa_j, sa_k]

        # equilibrium angle from the pi-bond environment of the center (eq 8d)
        pbo = torch.exp(scatter_sum(-((bo + SAFETY) ** 8), b_dst, N))
        dang = delta - P["valang"][s]
        sbo = dpi[center] - (1.0 - pbo[center]) * (dang[center]
                                                   + P["val8"] * nlp[center])
        # masked-out entries use a base of 1 so that pow() stays smooth to
        # second order (forces in the training loss differentiate twice)
        ok1 = (sbo <= 1.0) & (sbo > 0.0)
        s1 = torch.where(ok1, sbo, torch.ones_like(sbo))
        sbo01 = torch.where(ok1, s1 ** P["val9"], torch.zeros_like(sbo))
        ok2 = (sbo < 2.0) & (sbo > 1.0)
        s2 = torch.where(ok2, 2.0 - sbo, torch.ones_like(sbo))
        sbo12 = torch.where(ok2, 2.0 - s2 ** P["val9"], torch.zeros_like(sbo))
        sbo3 = sbo01 + sbo12 + 2.0 * (sbo > 2.0).to(sbo.dtype)
        theta0 = (180.0 - pa("theta0")
                  * (1.0 - torch.exp(-P["val10"] * (2.0 - sbo3)))) / 57.29577951

        boij, bojk = bo[e1], bo[e2]
        fijk = fbo[e1] * fbo[e2]
        expang = torch.exp(-pa("val2") * (theta0 - theta) ** 2)
        f7 = (1.0 - torch.exp(-P["val3"][sa_j] * (boij + SAFETY) ** pa("val4"))) \
            * (1.0 - torch.exp(-P["val3"][sa_j] * (bojk + SAFETY) ** pa("val4")))
        exp6 = guarded_exp(P["val6"] * dang[center])
        exp7 = guarded_exp(-pa("val7") * dang[center])
        val5 = P["val5"][sa_j]
        f8 = val5 - (val5 - 1.0) * (2.0 + exp6) / (1.0 + exp6 + exp7)
        eang = fijk * f7 * f8 * (pa("val1") - pa("val1") * expang)

        # penalty for two double bonds sharing an atom (eq 9), e.g. allene
        exp3 = guarded_exp(-P["pen3"] * dv[center])
        exp4 = guarded_exp(P["pen4"] * dv[center])
        f9 = (2.0 + exp3) / (1.0 + exp3 + exp4)
        epen = pa("pen1") * f9 * torch.exp(-P["pen2"] * (boij - 2.0) ** 2) \
            * torch.exp(-P["pen2"] * (bojk - 2.0) ** 2) * fijk

        # three-body conjugation
        dcoa = (delta - P["valboc"][s])[center]
        etcon = pa("coa1") * torch.sigmoid(-P["coa2"] * dcoa) \
            * torch.exp(-P["coa3"] * (delta[ai] - boij) ** 2) \
            * torch.exp(-P["coa3"] * (delta[ak] - bojk) ** 2) \
            * torch.exp(-P["coa4"] * (boij - 1.5) ** 2) \
            * torch.exp(-P["coa4"] * (bojk - 1.5) ** 2) * fijk

        inter.update(angle_i=ai, angle_center=center, angle_k=ak,
                     theta=theta, theta0=theta0, sbo3=sbo3, eang=eang,
                     epen=epen, etcon=etcon)
        return (scatter_sum(eang, center, N), scatter_sum(epen, center, N),
                scatter_sum(etcon, center, N))

    def _torsion_terms(self, s, N, b_src, b_dst, b_vec, b_r, b_shift, amask,
                       bo, bopi, fbo, delta, inter):
        """Torsion and four-body-conjugation energies.

        Torsions ``i-j-k-l`` are enumerated around every valence bond
        ``j-k`` whose two central atoms are both heavy, with ``i``/``l``
        valence neighbors of ``j``/``k`` (``i != k``, ``l != j``,
        ``l != i``). Implements eqs 10a-c (torsion, with the pi-bond order of
        the central bond driving the ``V2`` barrier) and 11a-b (conjugation)
        of van Duin 2001.

        Parameters
        ----------
        s : Tensor
            Species index per atom.
        N : int
            Number of atoms.
        b_src, b_dst, b_vec, b_r, b_shift : Tensor
            The bonded-edge arrays.
        amask : Tensor
            Valence-edge mask over the bonded edges.
        bo, bopi, fbo : Tensor
            Per-bonded-edge bond orders and the taper.
        delta : Tensor
            Corrected coordination per atom.
        inter : dict
            Intermediate stash.

        Returns
        -------
        tuple of Tensor
            ``(e_torsion, e_four_conj)`` per-atom energies (attributed to
            the first central atom).
        """
        P = self.params
        zero = torch.zeros(N, dtype=delta.dtype, device=delta.device)
        a_ids = torch.nonzero(amask, as_tuple=False)[:, 0]
        if a_ids.numel() == 0:
            return zero, zero.clone()
        a_src, a_dst = b_src[a_ids], b_dst[a_ids]
        a_shift = b_shift[a_ids]
        heavy = (s[a_src] != self._h_index) & (s[a_dst] != self._h_index)
        rev = reverse_edge_permutation(a_src, a_dst, a_shift)
        arange = torch.arange(a_ids.numel(), device=a_ids.device)
        central = torch.nonzero(heavy & (arange < rev), as_tuple=False)[:, 0]
        if central.numel() == 0:
            return zero, zero.clone()
        jc, kc = a_dst[central], a_src[central]        # central bond j-k

        # attach i (valence neighbor of j), then l (valence neighbor of k)
        c_idx, i_loc = _group_cross(jc, a_dst, N)
        ok = a_src[i_loc] != kc[c_idx]
        c_idx, i_loc = c_idx[ok], i_loc[ok]
        if c_idx.numel() == 0:
            return zero, zero.clone()
        pair_idx, l_loc = _group_cross(kc[c_idx], a_dst, N)
        c2, i2 = c_idx[pair_idx], i_loc[pair_idx]
        ok = (a_src[l_loc] != jc[c2]) & (a_src[l_loc] != a_src[i2])
        c2, i2, l_loc = c2[ok], i2[ok], l_loc[ok]
        if c2.numel() == 0:
            return zero, zero.clone()

        e_c = a_ids[central[c2]]                        # bonded-edge indices
        e_i, e_l = a_ids[i2], a_ids[l_loc]
        ti, tj, tk, tl = b_src[e_i], b_dst[e_i], b_src[e_c], b_src[e_l]

        v_ij = -b_vec[e_i]                              # x_i - x_j
        v_jk = b_vec[e_c]                               # x_j - x_k
        v_kl = b_vec[e_l]                               # x_k - x_l
        rij, rjk, rkl = b_r[e_i], b_r[e_c], b_r[e_l]
        v_jl = v_jk + v_kl
        rjl = torch.sqrt((v_jl * v_jl).sum(1))
        v_il = v_ij + v_jl
        ril = torch.sqrt((v_il * v_il).sum(1))
        v_ik = v_ij + v_jk
        rik = torch.sqrt((v_ik * v_ik).sum(1))

        rij2, rjk2, rkl2 = rij ** 2, rjk ** 2, rkl ** 2
        rjl2, ril2, rik2 = rjl ** 2, ril ** 2, rik ** 2
        # sin factors of the three bond angles; the 1e-12 floor keeps the
        # sqrt differentiable when an angle passes through collinearity
        c_ijk = (rij2 + rjk2 - rik2) / (2.0 * rij * rjk)
        s_ijk = torch.sqrt(torch.abs(1.0 - c_ijk ** 2) + 1.0e-12)
        c_jkl = (rjk2 + rkl2 - rjl2) / (2.0 * rjk * rkl)
        s_jkl = torch.sqrt(torch.abs(1.0 - c_jkl ** 2) + 1.0e-12)
        c_kjl = (rjk2 + rjl2 - rkl2) / (2.0 * rjk * rjl)
        s_kjl = torch.sqrt(torch.abs(1.0 - c_kjl ** 2) + 1.0e-12)

        fz = rij2 + rjl2 - ril2 - 2.0 * rij * rjl * c_ijk * c_kjl
        fm_den = rij * rjl * s_ijk * s_kjl
        tiny = (fm_den <= 0.000001) & (fm_den >= -0.000001)
        fm_den = torch.where(tiny, torch.ones_like(fm_den), fm_den)
        cos_w = 0.5 * fz * (~tiny).to(fz.dtype) / fm_den
        cos_w = torch.where(cos_w > 0.9999999, torch.ones_like(cos_w), cos_w)
        cos_w = torch.where(cos_w < -0.999999, -torch.ones_like(cos_w), cos_w)
        # cos(2w) and cos(3w) via the Chebyshev identities: exactly the
        # cosine multiples of w = arccos(cos_w), but with no arccos in the
        # graph (whose derivative diverges for the planar torsions every
        # conjugated molecule has)
        cos2w = 2.0 * cos_w ** 2 - 1.0
        cos3w = 4.0 * cos_w ** 3 - 3.0 * cos_w

        st_i, st_j, st_k, st_l = s[ti], s[tj], s[tk], s[tl]

        def pt(name):
            """Gather torsion parameter ``name`` per torsion."""
            return self._tor(name)[st_i, st_j, st_k, st_l]

        botij, botjk, botkl = bo[e_i], bo[e_c], bo[e_l]
        fijkl = fbo[e_i] * fbo[e_c] * fbo[e_l]
        f10 = (1.0 - torch.exp(-P["tor2"] * botij)) \
            * (1.0 - torch.exp(-P["tor2"] * botjk)) \
            * (1.0 - torch.exp(-P["tor2"] * botkl))
        dang = delta - P["valang"][s]
        delt = dang[tj] + dang[tk]
        f11 = (2.0 + guarded_exp(-P["tor3"] * delt)) \
            / (1.0 + guarded_exp(-P["tor3"] * delt)
               + guarded_exp(P["tor4"] * delt))
        expv2 = torch.exp(pt("tor1") * (2.0 - bopi[e_c] - f11) ** 2)
        etor = fijkl * f10 * s_ijk * s_jkl * (
            0.5 * pt("V1") * (1.0 + cos_w)
            + 0.5 * pt("V2") * expv2 * (1.0 - cos2w)
            + 0.5 * pt("V3") * (1.0 + cos3w))

        exptol = torch.exp(-P["cot2"] * (self.atol - 1.5) ** 2)
        f12 = (torch.exp(-P["cot2"] * (botij - 1.5) ** 2) - exptol) \
            * (torch.exp(-P["cot2"] * (botjk - 1.5) ** 2) - exptol) \
            * (torch.exp(-P["cot2"] * (botkl - 1.5) ** 2) - exptol)
        prod = 1.0 + (cos_w ** 2 - 1.0) * s_ijk * s_jkl
        efcon = fijkl * f12 * pt("cot1") * prod

        inter.update(tor_i=ti, tor_j=tj, tor_k=tk, tor_l=tl, tor_cos_w=cos_w,
                     etor=etor, efcon=efcon)
        return scatter_sum(etor, tj, N), scatter_sum(efcon, tj, N)

    def _hbond_terms(self, s, N, src, dst, vec, r, b_src, b_dst, b_vec, b_r,
                     amask, bo0, fhb, inter):
        """Hydrogen-bond energy over ``X-H .. Z`` triples.

        The donor pair ``X-H`` is a valence bond; acceptors ``Z`` are any
        heavy atom in the nonbonded neighbor list of the hydrogen (excluding
        the donor), with the donor--acceptor distance ``r_XZ`` switched off
        between ``hb_short`` and ``hb_long``.

        Parameters
        ----------
        s : Tensor
            Species index per atom.
        N : int
            Number of atoms.
        src, dst, vec, r : Tensor
            The full nonbonded edge arrays.
        b_src, b_dst, b_vec, b_r : Tensor
            The bonded-edge arrays.
        amask : Tensor
            Valence-edge mask over the bonded edges.
        bo0, fhb : Tensor
            Bond order and hydrogen-bond taper per bonded edge.
        inter : dict
            Intermediate stash.

        Returns
        -------
        Tensor
            Per-atom hydrogen-bond energy (attributed to the hydrogen).
        """
        zero = torch.zeros(N, dtype=r.dtype, device=r.device)
        if self._h_index < 0:
            return zero
        donor = amask & (s[b_dst] != self._h_index) & (s[b_src] == self._h_index)
        d_ids = torch.nonzero(donor, as_tuple=False)[:, 0]
        if d_ids.numel() == 0:
            return zero
        acc = (s[src] != self._h_index) & (s[dst] == self._h_index)
        a_ids = torch.nonzero(acc, as_tuple=False)[:, 0]
        if a_ids.numel() == 0:
            return zero
        di, al = _group_cross(b_src[d_ids], dst[a_ids], N)
        d_e, a_e = d_ids[di], a_ids[al]
        ok = src[a_e] != b_dst[d_e]                    # acceptor != donor X
        d_e, a_e = d_e[ok], a_e[ok]
        if d_e.numel() == 0:
            return zero

        hi, hj, hk = b_dst[d_e], b_src[d_e], src[a_e]  # X, H, Z
        rij = b_r[d_e]
        rjk = r[a_e]
        v_ik = b_vec[d_e] + vec[a_e]                   # (x_i-x_j) + (x_j-x_k)
        rik = torch.sqrt((v_ik * v_ik).sum(1))
        # NOTE: published ReaxFF-nn parameter libraries are trained with
        # r_XH (not r_XH^2) in this law-of-cosines numerator; kept for
        # compatibility with those libraries.
        cos_th = (rij + rjk ** 2 - rik ** 2) / (2.0 * rij * rjk)
        hbthe = 0.5 - 0.5 * cos_th
        frhb = taper_down(rik, self.hb_short, self.hb_long)

        sh_i, sh_j, sh_k = s[hi], s[hj], s[hk]

        def ph(name):
            """Gather hydrogen-bond parameter ``name`` per triple."""
            return self.params[f"hb_{name}"][sh_i, sh_j, sh_k]

        exphb1 = 1.0 - torch.exp(-ph("hb1") * bo0[d_e])
        hbsum = ph("rohb") / rjk + rjk / ph("rohb") - 2.0
        exphb2 = torch.exp(-ph("hb2") * hbsum)
        ehb = fhb[d_e] * frhb * ph("Dehb") * exphb1 * exphb2 * hbthe ** 2
        inter.update(hb_x=hi, hb_h=hj, hb_z=hk, ehb=ehb)
        return scatter_sum(ehb, hj, N)

    def _eem_charges(self, data: AtomicGraph, s: Tensor,
                     kernel: Tensor) -> Tensor:
        """Equilibrate partial charges with EEM, per structure.

        Solves, for each structure, the electronegativity-equalization
        system ``2 mu_i q_i + sum_j H_ij q_j + chi_i = lambda`` under the
        total-charge constraint, with the tapered, gamma-shielded Coulomb
        kernel ``H_ij = ke Tap(r) / (r^3 + gamma_ij^-3)^(1/3)`` accumulated
        over the nonbonded neighbor list (periodic images included). This is
        the variational minimum of ``E_self + E_Coulomb``, so making the
        charges geometry-dependent through the (differentiable) solve leaves
        the forces conservative.

        Parameters
        ----------
        data : AtomicGraph
            The batched graph.
        s : Tensor
            Species index per atom.
        kernel : Tensor
            Per-edge shielded, tapered Coulomb kernel (without the ``ke``
            prefactor), shape ``(E,)``.

        Returns
        -------
        Tensor
            Partial charges ``(N,)``.
        """
        P = self.params
        dtype = kernel.dtype
        device = kernel.device
        src, dst = data.edge_index[0], data.edge_index[1]
        gterm = kernel * KE
        chi = P["chi"][s]
        mu = P["mu"][s]
        total = getattr(data, "total_charge", None)

        q = torch.zeros(data.num_nodes, dtype=dtype, device=device)
        offsets = torch.cumsum(data.n_atoms, 0) - data.n_atoms
        batch_dst = data.batch[dst]
        for g in range(data.num_graphs):
            n = int(data.n_atoms[g])
            o = int(offsets[g])
            emask = batch_dst == g
            h_mat = torch.zeros(n, n, dtype=dtype, device=device).index_put(
                (dst[emask] - o, src[emask] - o), gterm[emask], accumulate=True)
            a = torch.zeros(n + 1, n + 1, dtype=dtype, device=device)
            a[:n, :n] = h_mat + torch.diag(2.0 * mu[o:o + n])
            a[:n, n] = 1.0
            a[n, :n] = 1.0
            b = torch.zeros(n + 1, dtype=dtype, device=device)
            b[:n] = -chi[o:o + n]
            if total is not None:
                b[n] = total[g].to(dtype) if torch.is_tensor(total) \
                    else float(total)
            q[o:o + n] = torch.linalg.solve(a, b)[:n]
        return q

    def export_library(self) -> FFieldLibrary:
        """Export the current parameters as a portable :class:`FFieldLibrary`.

        The inverse of construction: dense tensors are unpacked into the flat
        library dictionary (energies converted back to kcal/mol), every pair /
        angle / torsion / hydrogen-bond type is listed explicitly, and the
        network weights (nn mode) are included. ``FFieldLibrary.save`` writes
        the result as a ReaxFF-nn JSON file, so a trained force field can be
        used outside xnn.

        Returns
        -------
        FFieldLibrary
            The exported library.
        """
        spec = self.species
        S = len(spec)
        P = self.params
        p: dict = {}

        def unit(name):
            """Inverse unit factor (eV -> kcal/mol where applicable)."""
            return 1.0 / KCAL_TO_EV if name in _KCAL_PARAMS else 1.0

        for name in _P_GENERAL:
            p[name] = float(P[name]) * unit(name)
        p["cutoff"] = self.botol / 0.01
        p["acut"] = self.atol
        p["hbtol"] = self.hbtol
        for name in _P_SPECIES:
            for i, sp in enumerate(spec):
                p[f"{name}_{sp}"] = float(P[name][i]) * unit(name)
        for i, sp in enumerate(spec):
            p[f"mass_{sp}"] = float(SYMBOL_TO_Z[sp])
            p[f"hbond_{sp}"] = 1.0 if sp == "H" else -1.0
            # species-level pair quantities, re-exported from the diagonal
            # (identifies the species on reload and feeds combination rules)
            for name in ("rosi", "ropi", "ropp", "rvdw", "Devdw", "alfa"):
                p[f"{name}_{sp}"] = float(self._pair(name)[i, i]) * unit(name)

        bonds, offd = [], []
        for i, a in enumerate(spec):
            for j in range(i, S):
                b = spec[j]
                bd = f"{a}-{b}"
                bonds.append(bd)
                if a != b:
                    offd.append(bd)
                for name in _P_PAIR:
                    p[f"{name}_{bd}"] = float(self._pair(name)[i, j]) * unit(name)

        angs = []
        for j, b in enumerate(spec):
            for i, a in enumerate(spec):
                for k in range(i, S):
                    c = spec[k]
                    ang = f"{a}-{b}-{c}"
                    angs.append(ang)
                    for name in P_ANGLE:
                        p[f"{name}_{ang}"] = \
                            float(self._ang(name)[i, j, k]) * unit(name)

        torp = []
        for i, a in enumerate(spec):
            for j, b in enumerate(spec):
                for k, c in enumerate(spec):
                    for l, d in enumerate(spec):
                        if (l, k, j, i) < (i, j, k, l):
                            continue          # keep one spelling per torsion
                        tor = f"{a}-{b}-{c}-{d}"
                        torp.append(tor)
                        for name in P_TORSION:
                            p[f"{name}_{tor}"] = \
                                float(self._tor(name)[i, j, k, l]) * unit(name)

        hbs = []
        if self._h_index >= 0:
            h = self._h_index
            for i, a in enumerate(spec):
                for k, c in enumerate(spec):
                    if a == "H" or c == "H":
                        continue
                    hb = f"{a}-H-{c}"
                    hbs.append(hb)
                    for name in P_HBOND:
                        p[f"{name}_{hb}"] = \
                            float(self.params[f"hb_{name}"][i, h, k]) * unit(name)

        rcut, rcuta = {}, {}
        for i, a in enumerate(spec):
            for j, b in enumerate(spec):
                rcut[f"{a}-{b}"] = float(self.rcut_pair[i, j])
                rcuta[f"{a}-{b}"] = float(self.rcuta_pair[i, j])

        m = None
        if self.nn:
            m = {}

            def put_species(prefix):
                """Unpack a per-species network into the flat weight dict."""
                w = self.weights
                for i, sp in enumerate(spec):
                    m[f"{prefix}wi_{sp}"] = w[prefix + "wi"][i].tolist()
                    m[f"{prefix}bi_{sp}"] = w[prefix + "bi"][i].tolist()
                    m[f"{prefix}w_{sp}"] = w[prefix + "w"][:, i].tolist()
                    m[f"{prefix}b_{sp}"] = w[prefix + "b"][:, i].tolist()
                    m[f"{prefix}wo_{sp}"] = w[prefix + "wo"][i].tolist()
                    m[f"{prefix}bo_{sp}"] = w[prefix + "bo"][i].tolist()

            def put_pairs(prefix):
                """Unpack a per-pair network (symmetrized) into the dict."""
                w = self.weights

                def sym(t, axis):
                    """Symmetrize the pair axes ``axis``/``axis+1``."""
                    perm = list(range(t.dim()))
                    perm[axis], perm[axis + 1] = perm[axis + 1], perm[axis]
                    return 0.5 * (t + t.permute(*perm))

                for i, a in enumerate(spec):
                    for j in range(i, S):
                        bd = f"{a}-{spec[j]}"
                        m[f"{prefix}wi_{bd}"] = sym(w[prefix + "wi"], 0)[i, j].tolist()
                        m[f"{prefix}bi_{bd}"] = sym(w[prefix + "bi"], 0)[i, j].tolist()
                        m[f"{prefix}w_{bd}"] = sym(w[prefix + "w"], 1)[:, i, j].tolist()
                        m[f"{prefix}b_{bd}"] = sym(w[prefix + "b"], 1)[:, i, j].tolist()
                        m[f"{prefix}wo_{bd}"] = sym(w[prefix + "wo"], 0)[i, j].tolist()
                        m[f"{prefix}bo_{bd}"] = sym(w[prefix + "bo"], 0)[i, j].tolist()

            put_species("fm")
            put_pairs("fe")
            if self.bo_function in (1, 2):
                for prefix in ("fsi", "fpi", "fpp"):
                    put_pairs(prefix)

        return FFieldLibrary(
            p=p, m=m, spec=list(spec), bonds=bonds, offd=offd, angs=angs,
            torp=torp, hbs=hbs, messages=self.messages,
            bo_function=self.bo_function,
            energy_function=self.energy_function,
            message_function=self.message_function, vdw_function=0,
            bo_layer=self.bo_layer, mf_layer=self.mf_layer,
            be_layer=self.be_layer, vdw_layer=None, rcut=rcut, rcuta=rcuta)

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def from_ffield(cls, path, **kwargs) -> "ReaxFF":
        """Construct a :class:`ReaxFF` from a parameter library on disk.

        Parameters
        ----------
        path : str or Path
            Path to a ``ffield`` text library or a ReaxFF-nn JSON library.
        **kwargs
            Forwarded to the constructor.

        Returns
        -------
        ReaxFF
            The model.
        """
        return cls(path, **kwargs)

    @classmethod
    def from_config(cls, cfg) -> "ReaxFF":
        """Construct a :class:`ReaxFF` from a core model config.

        Core field: ``cfg.cutoff`` is the nonbonded (vdW / Coulomb / EEM)
        cutoff. Everything else is read from ``cfg.extra`` (``ffield`` is
        required); alternative spellings used by other ReaxFF tools are
        translated by
        :mod:`xnn.common.config.translate`.

        Parameters
        ----------
        cfg : xnn.common.config.schema.ModelConfig
            The core model config.

        Returns
        -------
        ReaxFF
            The model.
        """
        extra = dict(cfg.extra or {})
        ffield = extra.get("ffield")
        if ffield is None:
            raise ValueError("ReaxFF needs a parameter library: set "
                             "model.ffield to a ffield / ffield.json path")

        def listy(value):
            """Coerce a possibly stringified list to a Python value."""
            if isinstance(value, str) and value.lstrip().startswith(("[", "(")):
                return ast.literal_eval(value)
            return value

        def boolean(value):
            """Coerce a possibly stringified boolean."""
            if isinstance(value, str):
                return value.lower() in ("1", "true", "yes")
            return bool(value)

        nn = extra.get("nn")
        messages = extra.get("messages")
        return cls(
            ffield,
            species=listy(extra.get("species")),
            nn=None if nn is None else boolean(nn),
            messages=None if messages is None else int(messages),
            vdw_cutoff=float(cfg.cutoff),
            hb_short=float(extra.get("hb_short", 6.75)),
            hb_long=float(extra.get("hb_long", 7.5)),
            trainable=listy(extra.get("trainable", ())) or (),
            keep_intermediates=boolean(extra.get("keep_intermediates", False)),
        )
