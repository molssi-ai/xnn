"""AMBER: the fixed-topology force field of Cornell et al. (1995).

The Cornell *et al.* force field (ff94; *J. Am. Chem. Soc.* 117, 5179,
1995, eq 1) writes the energy of a molecule or assembly as

    E = sum_bonds K_r (r - r_eq)^2 + sum_angles K_theta (theta - theta_eq)^2
        + sum_dihedrals sum_n V_n/2 [1 + cos(n phi - gamma)]
        + sum_{i<j} [ A_ij / R_ij^12 - B_ij / R_ij^6 + q_i q_j / (eps R_ij) ]

with harmonic bonds and angles, a Fourier series per proper dihedral,
impropers of the same periodic form centred on the third atom of the
quadruple, and nonbonded interactions between atoms separated by three or
more bonds, the 1,4 pairs scaled by 1/2 (van der Waals) and 1/1.2
(electrostatics). The van der Waals term is written with per-type well
depths ``epsilon_i`` and radii ``R*_i`` combined as ``R*_ij = R*_i + R*_j``
and ``epsilon_ij = (epsilon_i epsilon_j)^1/2`` (Table 13 of the paper), so
that ``A_ij = epsilon_ij R*_ij^12`` and ``B_ij = 2 epsilon_ij R*_ij^6``.
Partial charges are RESP charges that belong to residue templates rather
than to atom types.

Like OPLS, the model binds a parameter library to one molecular topology
(:mod:`xnn.ffnn.models.topology`): per-atom types, charges and bonds,
from which angles, dihedrals, exclusions and 1,4 pairs follow, with the
impropers placed by the AMBER rules (:func:`~xnn.ffnn.models.amberlib.place_impropers`).
:meth:`Amber.from_atoms` types a structure from the residue templates and
atom-type SMARTS of the parameter file. Later parameter sets of the same
functional form (ff99, ff14SB, GAFF, ...) are libraries for this one model.

Conventions and scope
---------------------
* Every parameter lives as a plain tensor in an :class:`AmberForceField`
  (eV / Angstrom / radians at assembly) and any group can be trained;
  several :class:`Amber` instances can share one force field. Charges are
  per atom and belong to the model (``trainable=("charge",)``), held to the
  bound system's total charge by default.
* Torsion phases ``gamma`` and periodicities ``n`` are fixed; the barrier
  heights are the trainable part.
* Nonbonded interactions come from the neighbor list within ``cutoff``
  (optionally smoothed over ``switch_width``); 1,4 pairs are evaluated
  exactly from the topology. Excluded pairs are excluded in every periodic
  image.
* The Coulomb constant is the one of the AMBER programs,
  ``18.2223^2 = 332.0522`` kcal Angstrom / (mol e^2).
* The 10-12 hydrogen-bond term of earlier AMBER force fields is not part of
  the Cornell *et al.* model (its ``parm94`` table is zero) and is not
  implemented.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Optional, Sequence, Union

import torch
from torch import Tensor, nn
from torch.nn import Parameter, ParameterDict

from xnn.common.data import AtomicGraph
from xnn.common.data.elements import CHEMICAL_SYMBOLS
from xnn.common.models import InteratomicPotential, register_model
from xnn.common.models.ops import scatter_sum

from .geometry import TINY, geometric_mean, pair_vectors, dihedral_cos_sin
from .amberlib import (AmberLibrary, read_amber, resolve_bond_type,
                       resolve_angle_type, resolve_dihedral_type,
                       resolve_improper_type, place_impropers, KCAL_TO_EV,
                       MAX_TERMS, MAX_IMPROPER_TERMS)
from .topology import MolecularTopology, read_topology, residue_name_list

# Coulomb constant of the AMBER programs in eV Angstrom: charges in the
# AMBER topology files are scaled by 18.2223 so that q_i q_j / r is in
# kcal/mol (the thermochemical kcal of xnn's OPLS constant converts to eV).
KE_KCAL = 18.2223 ** 2
KE = KE_KCAL * KCAL_TO_EV
DEG = math.pi / 180.0

# Trainable parameter groups of the force field ("charge" and "atom_ref"
# are model-level groups).
_GROUPS = ("rstar", "epsilon", "bond_k", "bond_r0", "angle_k",
           "angle_theta0", "dihedral_k", "improper_k")
_MODEL_GROUPS = ("charge", "atom_ref")
_MAX_Z = 100


def _split_trainable(trainable: Union[Sequence[str], str]
                     ) -> tuple[Union[Sequence[str], str], set]:
    """Separate the model-level groups from the force-field groups."""
    if isinstance(trainable, str):
        if trainable.lower() == "all":
            return "all", set(_MODEL_GROUPS)
        if trainable in _MODEL_GROUPS:
            return (), {trainable}
        return (trainable,), set()
    groups = tuple(g for g in trainable if g not in _MODEL_GROUPS)
    return groups, {g for g in trainable if g in _MODEL_GROUPS}


def _as_library(source: Union[AmberLibrary, str, Path]) -> AmberLibrary:
    """Coerce a library argument to an :class:`AmberLibrary`."""
    if isinstance(source, AmberLibrary):
        return source
    return read_amber(str(source))


def _exact_trig(values: Tensor) -> tuple[Tensor, Tensor]:
    """``(cos, sin)`` of phase angles with the values at multiples of 90
    degrees made exact (``sin(pi)`` is not zero in floating point)."""
    c, s = torch.cos(values), torch.sin(values)
    for t in (c, s):
        for target in (-1.0, 0.0, 1.0):
            t[(t - target).abs() < 1e-12] = target
    return c, s


def multiple_angles(cos1: Tensor, sin1: Tensor, n_max: int
                    ) -> tuple[Tensor, Tensor]:
    """``cos(n phi)`` and ``sin(n phi)`` for ``n = 1..n_max`` by recurrence.

    Parameters
    ----------
    cos1, sin1 : Tensor
        ``cos phi`` and ``sin phi`` of shape ``(M,)``.
    n_max : int
        Highest multiple.

    Returns
    -------
    tuple of Tensor
        Two ``(M, n_max)`` tensors, column ``n - 1`` holding the multiple
        ``n``.
    """
    cs, ss = [cos1], [sin1]
    for _ in range(1, n_max):
        cs.append(cs[-1] * cos1 - ss[-1] * sin1)
        ss.append(ss[-1] * cos1 + cs[-2] * sin1)
    return torch.stack(cs, 1), torch.stack(ss, 1)


class AmberForceField(nn.Module):
    """The trainable parameter tensors of an AMBER parameter library.

    Parameters
    ----------
    library : AmberLibrary, str or Path
        The parameter library (or a shipped set such as ``"ff94"``, a
        ``.frc`` path or a native JSON path, see
        :func:`~xnn.ffnn.models.amberlib.read_amber`).
    trainable : sequence of str or "all", optional
        Groups to expose to the optimizer: ``"rstar"``, ``"epsilon"``,
        ``"bond_k"``, ``"bond_r0"``, ``"angle_k"``, ``"angle_theta0"``,
        ``"dihedral_k"``, ``"improper_k"``; ``"all"`` unfreezes every group.

    Attributes
    ----------
    params : torch.nn.ParameterDict
        ``rstar`` ``(T,)`` in Angstrom, ``epsilon`` ``(T,)`` in eV,
        ``bond_k`` ``(NB,)`` in eV/A^2, ``bond_r0`` ``(NB,)``, ``angle_k``
        ``(NA,)`` in eV/rad^2, ``angle_theta0`` ``(NA,)`` in rad,
        ``dihedral_k`` ``(ND, 4)`` and ``improper_k`` ``(NI, 2)`` in eV (one
        column per Fourier term, zero where a row has fewer terms).
    dihedral_n, dihedral_cos, dihedral_sin, dihedral_mask : Tensor
        Periodicities, the cosine / sine of the phases and a 0/1 mask of the
        used term slots of the torsion rows (fixed buffers; a trained
        ``dihedral_k`` never grows a term in an unused slot); likewise
        ``improper_n`` / ``improper_cos`` / ``improper_sin`` /
        ``improper_mask``.
    type_names, bond_keys, angle_keys, dihedral_keys, improper_keys : list
        Row labels of the parameter tensors.
    fudge_lj, fudge_qq : float
        The library's 1,4 scaling factors.
    """

    def __init__(self, library: Union[AmberLibrary, str, Path],
                 trainable: Union[Sequence[str], str] = ()):
        super().__init__()
        lib = _as_library(library)
        dtype = torch.get_default_dtype()
        self.name = lib.name
        self.fudge_lj = float(lib.fudge_lj)
        self.fudge_qq = float(lib.fudge_qq)
        if isinstance(trainable, str):
            trainable_set = set(_GROUPS) if trainable.lower() == "all" \
                else {trainable}
        else:
            trainable_set = set(trainable)
        unknown = trainable_set - set(_GROUPS)
        if unknown:
            raise ValueError(f"unknown trainable group(s) {sorted(unknown)}; "
                             f"available: {list(_GROUPS)}")

        self.type_names = list(lib.atom_types)
        self._type_index = {n: i for i, n in enumerate(self.type_names)}
        self.type_cls = [lib.cls(n, "bond") for n in self.type_names]
        self.type_cls_angle = [lib.cls(n, "angle") for n in self.type_names]
        self.type_cls_torsion = [lib.cls(n, "torsion") for n in self.type_names]
        self.type_cls_oop = [lib.cls(n, "oop") for n in self.type_names]
        self.templates = dict(lib.templates)
        self.fragments = dict(lib.fragments)
        self.metadata = dict(lib.metadata)
        self.references = list(lib.references)
        at = [lib.atom_types[n] for n in self.type_names]
        self.register_buffer("type_z", torch.tensor(
            [int(a["element"]) for a in at], dtype=torch.long))
        self.register_buffer("type_mass", torch.tensor(
            [float(a.get("mass", 0.0)) for a in at], dtype=dtype))

        self.bond_keys = list(lib.bond_types)
        self.angle_keys = list(lib.angle_types)
        self.dihedral_keys = list(lib.dihedral_types)
        self.improper_keys = list(lib.improper_types)
        self._bond_types = dict(lib.bond_types)
        self._angle_types = dict(lib.angle_types)
        self._dihedral_types = dict(lib.dihedral_types)
        self._improper_types = dict(lib.improper_types)
        self._bond_index = {k: i for i, k in enumerate(self.bond_keys)}
        self._angle_index = {k: i for i, k in enumerate(self.angle_keys)}
        self._dihedral_index = {k: i for i, k in enumerate(self.dihedral_keys)}
        self._improper_index = {k: i for i, k in enumerate(self.improper_keys)}

        self.params = ParameterDict()

        def make(name, values, shape=(-1,)):
            """Register value list as the (possibly trainable) group ``name``."""
            self.params[name] = Parameter(
                torch.tensor(values, dtype=dtype).reshape(*shape),
                requires_grad=name in trainable_set)

        make("rstar", [float(a["rstar"]) for a in at])
        make("epsilon", [float(a["epsilon"]) * KCAL_TO_EV for a in at])
        make("bond_k", [float(lib.bond_types[k]["k"]) * KCAL_TO_EV
                        for k in self.bond_keys])
        make("bond_r0", [float(lib.bond_types[k]["r0"]) for k in self.bond_keys])
        make("angle_k", [float(lib.angle_types[k]["k"]) * KCAL_TO_EV
                         for k in self.angle_keys])
        make("angle_theta0", [float(lib.angle_types[k]["theta0"]) * DEG
                              for k in self.angle_keys])

        def term_tables(table, keys, width):
            """Per-row (K, n, phase, mask) with zero-padded unused term slots;
            the mask keeps the padding out of the energy and its gradient."""
            ks, ns, phs, ms = [], [], [], []
            for key in keys:
                terms = list(table[key]["terms"])[:width]
                pad = width - len(terms)
                ks.append([float(t[0]) * KCAL_TO_EV for t in terms] + [0.0] * pad)
                ns.append([max(int(t[1]), 1) for t in terms] + [1] * pad)
                phs.append([float(t[2]) * DEG for t in terms] + [0.0] * pad)
                ms.append([1.0] * len(terms) + [0.0] * pad)
            return (torch.tensor(ks, dtype=dtype).reshape(len(keys), width),
                    torch.tensor(ns, dtype=torch.long).reshape(len(keys), width),
                    torch.tensor(phs, dtype=torch.float64).reshape(len(keys), width),
                    torch.tensor(ms, dtype=dtype).reshape(len(keys), width))

        k, n, ph, mask = term_tables(lib.dihedral_types, self.dihedral_keys, MAX_TERMS)
        self.params["dihedral_k"] = Parameter(k, requires_grad="dihedral_k" in trainable_set)
        self.register_buffer("dihedral_n", n)
        self.register_buffer("dihedral_mask", mask)
        c, s = _exact_trig(ph)
        self.register_buffer("dihedral_cos", c.to(dtype))
        self.register_buffer("dihedral_sin", s.to(dtype))
        k, n, ph, mask = term_tables(lib.improper_types, self.improper_keys,
                                     MAX_IMPROPER_TERMS)
        self.params["improper_k"] = Parameter(k, requires_grad="improper_k" in trainable_set)
        self.register_buffer("improper_n", n)
        self.register_buffer("improper_mask", mask)
        c, s = _exact_trig(ph)
        self.register_buffer("improper_cos", c.to(dtype))
        self.register_buffer("improper_sin", s.to(dtype))

    # resolution (topology binding)
    def type_index(self, name: str) -> int:
        """Row index of atom type ``name`` (``KeyError`` if unknown)."""
        if name not in self._type_index:
            raise KeyError(f"atom type {name!r} is not in library {self.name!r}")
        return self._type_index[name]

    def resolve_bond(self, a: str, b: str) -> int:
        """Bond-type row for the class pair (``KeyError`` if missing)."""
        key = resolve_bond_type(self._bond_types, a, b)
        if key is None:
            raise KeyError(f"no bond type {a}-{b} in library {self.name!r}")
        return self._bond_index[key]

    def resolve_angle(self, a: str, b: str, c: str) -> int:
        """Angle-type row for the class triple, center second."""
        key = resolve_angle_type(self._angle_types, a, b, c)
        if key is None:
            raise KeyError(f"no angle type {a}-{b}-{c} in library {self.name!r}")
        return self._angle_index[key]

    def resolve_dihedral(self, a: str, b: str, c: str, d: str) -> int:
        """Dihedral-type row: the exact quadruple (either direction) wins
        over the ``X-B-C-X`` generic entry."""
        key = resolve_dihedral_type(self._dihedral_types, a, b, c, d)
        if key is None:
            raise KeyError(f"no dihedral type {a}-{b}-{c}-{d} in library "
                           f"{self.name!r}")
        return self._dihedral_index[key]

    def resolve_improper(self, *args) -> int:
        """Improper-type row, by exact key or by the four classes in
        evaluation order (center third, distinguished atom last)."""
        if len(args) == 1:
            key = args[0]
            if key not in self._improper_index:
                raise KeyError(f"no improper type {key!r} in library {self.name!r}")
            return self._improper_index[key]
        i, j, k, l = args
        key = resolve_improper_type(self._improper_types, i, j, k, l)
        if key is None:
            raise KeyError(f"no improper type for {i}-{j}-{k}-{l} (center {k}) "
                           f"in library {self.name!r}")
        return self._improper_index[key]

    # export
    def export_library(self) -> AmberLibrary:
        """The current (possibly trained) parameters as an :class:`AmberLibrary`.

        Returns
        -------
        AmberLibrary
            In AMBER units (kcal/mol, Angstrom, degrees); the residue
            templates keep the library's charges (trained per-atom charges
            belong to the models, see :meth:`Amber.export_library`).
        """
        P = {k: v.detach().cpu() for k, v in self.params.items()}
        atom_types = {}
        for i, name in enumerate(self.type_names):
            atom_types[name] = {
                "cls_nonbond": name, "cls_bond": self.type_cls[i],
                "cls_angle": self.type_cls_angle[i],
                "cls_torsion": self.type_cls_torsion[i],
                "cls_oop": self.type_cls_oop[i],
                "element": int(self.type_z[i]), "mass": float(self.type_mass[i]),
                "rstar": float(P["rstar"][i]),
                "epsilon": float(P["epsilon"][i]) / KCAL_TO_EV, "comment": ""}

        def terms(table, keys, kpar):
            """The library's term lists with the current barrier heights."""
            out = {}
            for i, key in enumerate(keys):
                rows = [[float(kpar[i, m]) / KCAL_TO_EV, int(n), float(ph)]
                        for m, (_, n, ph) in enumerate(table[key]["terms"][:kpar.shape[1]])]
                out[key] = {"terms": rows}
            return out

        return AmberLibrary(
            atom_types=atom_types,
            bond_types={k: {"k": float(P["bond_k"][i]) / KCAL_TO_EV,
                            "r0": float(P["bond_r0"][i])}
                        for i, k in enumerate(self.bond_keys)},
            angle_types={k: {"k": float(P["angle_k"][i]) / KCAL_TO_EV,
                             "theta0": float(P["angle_theta0"][i]) / DEG}
                         for i, k in enumerate(self.angle_keys)},
            dihedral_types=terms(self._dihedral_types, self.dihedral_keys,
                                 P["dihedral_k"]),
            improper_types=terms(self._improper_types, self.improper_keys,
                                 P["improper_k"]),
            fudge_lj=self.fudge_lj, fudge_qq=self.fudge_qq, name=self.name,
            references=list(self.references), templates=dict(self.templates),
            fragments=dict(self.fragments), metadata=dict(self.metadata))


@register_model("amber")
class Amber(InteratomicPotential):
    """The AMBER fixed-topology force field (Cornell *et al.* 1995 form).

    An instance binds a parameter library to one molecular topology; the
    bonded terms, impropers, exclusions and scaled 1,4 pairs are resolved
    once, and ``forward`` evaluates the energy on any conformation (or batch
    of conformations) of that system. Forces and stress come from autograd
    via :class:`~xnn.common.models.outputs.ForceStressOutput`.

    Parameters
    ----------
    ffield : AmberLibrary, AmberForceField, str or Path
        The parameter library -- a parsed :class:`AmberLibrary`, a set
        shipped with xnn (``"ff94"``), a ``.frc`` or native JSON path, or an
        :class:`AmberForceField` to share parameters with other models.
    topology : MolecularTopology, str or Path
        The system's topology (or a topology JSON path). Its ``charges``
        are the partial charges unless ``charges`` is given; its
        ``residues`` order the impropers (:func:`~xnn.ffnn.models.amberlib.place_impropers`).
    cutoff : float, optional
        Nonbonded cutoff in Angstrom (the neighbor-list cutoff); by default
        10.0. 1,4 pairs are independent of it.
    switch_width : float, optional
        Width of a quintic switching function taking the nonbonded
        interactions smoothly to zero at the cutoff; 0.0 (plain truncation)
        by default.
    charges : sequence of float, optional
        Per-atom partial charges in e, overriding the topology's.
    fudge_lj, fudge_qq : float, optional
        Override the library's 1,4 scaling factors (1/2 and 1/1.2).
    trainable : sequence of str or "all", optional
        Force-field groups (see :class:`AmberForceField`) plus the model
        groups ``"charge"`` (per-atom charges) and ``"atom_ref"``
        (per-element reference energies); ``"all"`` unfreezes everything.
    keep_intermediates : bool, optional
        Stash the intermediate tensors of the last evaluation in
        ``self.intermediates``. Default ``False``.
    auto_impropers : bool, optional
        When the topology lists no impropers, place them by the AMBER rules
        (one per three-connected atom with a matching pattern). Default
        ``True``. Listed impropers are used as given, resolved by their
        atom classes in the listed order (or by ``improper_keys``).
    constrain_charge : bool or float, optional
        Hold the bound system's total charge when ``charge`` trains:
        ``True`` (default) keeps the sum of the initial charges, a number
        fixes that value in e, ``False`` registers no constraint. The
        charges the model evaluates are the per-atom parameters shifted
        uniformly onto the constraint.
    atomic_energies : mapping, optional
        Per-element reference energies in eV, ``{Z or symbol: E0}``, loaded
        into ``atom_ref``; the offset between the labels' zero and the force
        field's. All zero by default.

    Notes
    -----
    ``forward`` returns, besides ``node_energy`` / ``energy`` /
    ``node_features``, the partial ``charges`` ``(N,)`` and one
    per-structure tensor per term: ``e_bond``, ``e_angle``, ``e_torsion``,
    ``e_improper``, ``e_vdw``, ``e_coulomb``, ``e_vdw14``, ``e_coulomb14``
    and ``e_ref``. Every structure in a batch must have this topology's
    atom count and element sequence.

    Attributes
    ----------
    atom_ref : torch.nn.Embedding
        The per-element reference energies, row ``Z``.
    species : list of int
        The atomic numbers of the bound topology.
    impropers : list of tuple
        The improper quadruples in evaluation order (center third).
    """

    def __init__(self, ffield: Union[AmberLibrary, AmberForceField, str, Path],
                 topology: Union[MolecularTopology, str, Path], *,
                 cutoff: float = 10.0,
                 switch_width: float = 0.0,
                 charges: Optional[Sequence[float]] = None,
                 fudge_lj: Optional[float] = None,
                 fudge_qq: Optional[float] = None,
                 trainable: Union[Sequence[str], str] = (),
                 keep_intermediates: bool = False,
                 auto_impropers: bool = True,
                 constrain_charge: Union[bool, float] = True,
                 atomic_energies=None):
        super().__init__()
        groups, model_groups = _split_trainable(trainable)
        self.ff = ffield if isinstance(ffield, AmberForceField) \
            else AmberForceField(ffield, trainable=groups)
        self.atom_ref = nn.Embedding(_MAX_Z, 1)
        nn.init.zeros_(self.atom_ref.weight)
        self.atom_ref.weight.requires_grad_("atom_ref" in model_groups)
        self._train_charge = "charge" in model_groups
        self.auto_impropers = bool(auto_impropers)
        self.cutoff = float(cutoff)
        self.switch_width = float(switch_width)
        if not 0.0 <= self.switch_width < self.cutoff:
            raise ValueError("switch_width must satisfy 0 <= switch_width < cutoff")
        self.fudge_lj = self.ff.fudge_lj if fudge_lj is None else float(fudge_lj)
        self.fudge_qq = self.ff.fudge_qq if fudge_qq is None else float(fudge_qq)
        self.keep_intermediates = bool(keep_intermediates)
        self.intermediates: dict = {}
        self.node_feature_dim = 1
        self._bind_topology(
            topology if isinstance(topology, MolecularTopology)
            else read_topology(topology), charges)
        self.species = sorted({int(z) for z in self.top_z.tolist()})
        if atomic_energies is not None:
            from ...common.finetune.reference import set_atomic_energies
            set_atomic_energies(self, atomic_energies)
        total = float(self.charge.detach().sum()) if constrain_charge is True \
            else (None if constrain_charge is False else float(constrain_charge))
        self.register_buffer("charge_total", torch.tensor(
            float("nan") if total is None else total,
            dtype=self.charge.dtype), persistent=False)

    @classmethod
    def from_atoms(cls, structure, ffield: Union[AmberLibrary, AmberForceField,
                                                 str, Path] = "ff94",
                   *, charge: int = 0,
                   bonds: Optional[Sequence[Sequence[int]]] = None,
                   charges: Optional[Sequence[float]] = None,
                   orders: bool = True,
                   **kwargs) -> "Amber":
        """Build an AMBER model for a structure, typing it from the library.

        Residue templates (the ``#fragments`` of the parameter file) assign
        atom types *and* partial charges to every atom they match; atoms
        outside any residue are typed by the per-type SMARTS templates and
        carry zero charge unless ``charges`` is given.

        Parameters
        ----------
        structure : object
            An ``ase.Atoms``, a ``(positions, atomic_numbers)`` pair, a dict
            with ``"pos"`` / ``"atomic_numbers"``, an RDKit molecule or a
            SMILES string.
        ffield : AmberLibrary, AmberForceField, str or Path, optional
            The parameter library (must carry templates); default ``"ff94"``.
        charge : int, optional
            Total charge of the structure (for bond-order perception).
        bonds : sequence of (int, int), optional
            Known connectivity; bond orders are then perceived, not bonds.
        charges : sequence of float, optional
            Per-atom partial charges overriding the templates' (e.g. RESP
            charges of a small molecule).
        orders : bool, optional
            With ``bonds`` given, ``False`` skips the perception of bond
            orders and formal charges (the residue templates need only the
            connectivity), which is robust for charged systems and ions.
        **kwargs
            Forwarded to the constructor (``cutoff``, ``trainable``, ...).

        Returns
        -------
        Amber
            The model, bound to the derived topology.
        """
        from ..common.typing import assign_atom_types, perceive_bonds
        lib = ffield if isinstance(ffield, AmberForceField) else _as_library(ffield)
        types, details = assign_atom_types(structure, lib, charge=charge,
                                           bonds=bonds, return_details=True,
                                           orders=orders)
        template_q = [0.0 if q is None else float(q) for q in details["charges"]]
        residues = details["residues"]
        has_res = any(r is not None for r in residues)
        top = MolecularTopology.from_bonds(
            types, perceive_bonds(details["mol"]), charges=template_q,
            residues=residues if has_res else None,
            residue_names=residue_name_list(residues, details["fragment_names"])
            if has_res else None)
        return cls(lib, top, charges=charges, **kwargs)

    # topology binding
    def _bind_topology(self, top: MolecularTopology,
                       charges: Optional[Sequence[float]]) -> None:
        """Resolve the topology against the force field into index buffers."""
        self.topology = top
        ff = self.ff
        dtype = torch.get_default_dtype()
        missing: list[str] = []

        def gather(fn, *args) -> int:
            """Resolve one type, recording a miss instead of raising."""
            try:
                return fn(*args)
            except KeyError as err:
                msg = str(err.args[0])
                if msg not in missing:
                    missing.append(msg)
                return 0

        t_idx = [gather(ff.type_index, name) for name in top.types]
        cls = [ff.type_cls[i] for i in t_idx]
        cls_a = [ff.type_cls_angle[i] for i in t_idx]
        cls_t = [ff.type_cls_torsion[i] for i in t_idx]
        cls_o = [ff.type_cls_oop[i] for i in t_idx]
        b_type = [gather(ff.resolve_bond, cls[i], cls[j]) for i, j in top.bonds]
        a_type = [gather(ff.resolve_angle, cls_a[i], cls_a[j], cls_a[k])
                  for i, j, k in top.angles]
        d_type = [gather(ff.resolve_dihedral, cls_t[i], cls_t[j], cls_t[k], cls_t[l])
                  for i, j, k, l in top.dihedrals]
        impropers = [tuple(int(x) for x in im) for im in top.impropers]
        if top.improper_keys:
            i_type = [gather(ff.resolve_improper, key) for key in top.improper_keys]
        elif impropers:
            i_type = [gather(ff.resolve_improper, cls_o[i], cls_o[j], cls_o[k], cls_o[l])
                      for i, j, k, l in impropers]
        elif self.auto_impropers:
            elements = [CHEMICAL_SYMBOLS[int(ff.type_z[i])] if 0 < int(ff.type_z[i])
                        < len(CHEMICAL_SYMBOLS) else "X" for i in t_idx]
            impropers, keys = place_impropers(top, cls_o, elements, ff._improper_types)
            i_type = [ff.resolve_improper(key) for key in keys]
        else:
            i_type = []
        if missing:
            raise KeyError("the topology needs parameters the library does "
                           "not provide:\n  " + "\n  ".join(missing))

        def buf(name, data, shape):
            """Register an integer index buffer (possibly empty)."""
            self.register_buffer(name, torch.tensor(data, dtype=torch.long).reshape(*shape))

        n = top.n_atoms
        buf("top_type", t_idx, (-1,))
        self.register_buffer("top_z", ff.type_z[self.top_type].clone())
        buf("bond_index", [list(b) for b in top.bonds], (-1, 2))
        self.bond_index = self.bond_index.t().contiguous()
        buf("bond_type", b_type, (-1,))
        buf("angle_index", [list(a) for a in top.angles], (-1, 3))
        self.angle_index = self.angle_index.t().contiguous()
        buf("angle_type", a_type, (-1,))
        buf("dihedral_index", [list(d) for d in top.dihedrals], (-1, 4))
        self.dihedral_index = self.dihedral_index.t().contiguous()
        buf("dihedral_type", d_type, (-1,))
        self.impropers = impropers
        buf("improper_index", [list(im) for im in impropers], (-1, 4))
        self.improper_index = self.improper_index.t().contiguous()
        buf("improper_type", i_type, (-1,))
        buf("pair14_index", [list(p) for p in top.pairs14], (-1, 2))
        self.pair14_index = self.pair14_index.t().contiguous()
        excl = sorted(set(top.exclusions) | set(top.pairs14))
        buf("excl_index", [list(p) for p in excl], (-1, 2))
        self.excl_index = self.excl_index.t().contiguous()

        # charges: explicit, else the topology's, else zero
        if charges is None:
            charges = top.charges if top.charges else [0.0] * n
        q = torch.as_tensor([float(c) for c in charges], dtype=dtype)
        if q.shape != (n,):
            raise ValueError(f"got {q.numel()} charges for {n} atoms")
        if self._train_charge:
            self.charge = Parameter(q)
        else:
            self.register_buffer("charge", q)

    @property
    def charges(self) -> Tensor:
        """Tensor : The per-atom charges the model evaluates: the ``charge``
        parameters shifted uniformly so that they sum to the constrained
        total (the parameters themselves without a constraint)."""
        q = self.charge
        if bool(torch.isnan(self.charge_total)):
            return q
        return q - (q.sum() - self.charge_total) / q.shape[0]

    # energy terms
    def _vdw_coulomb(self, rs_i, rs_j, eps_i, eps_j, q_i, q_j, r, r2
                     ) -> tuple[Tensor, Tensor]:
        """Per-pair van der Waals (``eps [(R*/r)^12 - 2 (R*/r)^6]``, eq 1 with
        ``R*_ij = R*_i + R*_j``) and Coulomb energies in eV."""
        rstar = rs_i + rs_j
        eps = geometric_mean(eps_i, eps_j)
        x = (rstar * rstar / (r2 + TINY)) ** 3
        return eps * (x * x - 2.0 * x), KE * q_i * q_j / r

    def _periodic(self, idx: Tensor, types: Tensor, k_par: Tensor, n_buf: Tensor,
                  cos_buf: Tensor, sin_buf: Tensor, mask: Tensor, data: AtomicGraph
                  ) -> tuple[Tensor, Tensor]:
        """``sum_m K_m [1 + cos(n_m phi - gamma_m)]`` per quadruple, from the
        dihedral's ``cos phi`` / ``sin phi`` by multiple-angle recurrences."""
        c1, s1 = dihedral_cos_sin(data, idx)
        n = n_buf[types]
        n_max = int(n.max()) if n.numel() else 1
        cos_n, sin_n = multiple_angles(c1, s1, n_max)
        cn = torch.gather(cos_n, 1, n - 1)
        sn = torch.gather(sin_n, 1, n - 1)
        e = k_par[types] * mask[types] * (1.0 + cos_buf[types] * cn + sin_buf[types] * sn)
        return e.sum(1), c1

    # forward
    def forward(self, data: AtomicGraph) -> dict[str, Tensor]:
        """Evaluate the AMBER energy on a (batched) atomic graph.

        Parameters
        ----------
        data : AtomicGraph
            The batched graph; every structure must be a conformation of
            the bound topology, with the neighbor list built with this
            model's ``cutoff``.

        Returns
        -------
        dict of str to Tensor
            ``node_energy`` ``(N,)``, ``energy`` ``(B,)``, ``charges``
            ``(N,)``, ``node_features`` ``(N, 1)`` and the per-structure
            energy decomposition (see the class docstring).

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

        if not bool((data.n_atoms == n).all()):
            raise ValueError(
                f"every structure must have the bound topology's {n} atoms "
                f"(got {data.n_atoms.tolist()})")
        if not bool((data.atomic_numbers == self.top_z.repeat(B)).all()):
            raise ValueError("atomic numbers do not match the bound topology's "
                             "element sequence")

        t = self.top_type.repeat(B)
        starts = torch.arange(B, device=device) * n

        def expand(idx: Tensor) -> Tensor:
            """Tile per-structure index rows ``(k, m)`` to ``(k, B*m)``."""
            return (idx.unsqueeze(1) + starts.view(1, B, 1)).reshape(idx.shape[0], -1)

        def tile(v: Tensor) -> Tensor:
            """Tile per-interaction type rows ``(m,)`` to ``(B*m,)``."""
            return v.repeat(B)

        # bonds
        bi = expand(self.bond_index)
        bt = tile(self.bond_type)
        bvec = pair_vectors(data, bi[0], bi[1])
        br = torch.sqrt((bvec * bvec).sum(1) + TINY)
        e_bond = P["bond_k"][bt] * (br - P["bond_r0"][bt]) ** 2
        e_bond_node = scatter_sum(0.5 * e_bond, bi[0], N) \
            + scatter_sum(0.5 * e_bond, bi[1], N)

        # angles
        ai = expand(self.angle_index)
        at = tile(self.angle_type)
        u = pair_vectors(data, ai[1], ai[0])
        v = pair_vectors(data, ai[1], ai[2])
        cos_th = (u * v).sum(1) / torch.sqrt((u * u).sum(1) * (v * v).sum(1) + TINY)
        bound = min(0.9999999999, 1.0 - 4.0 * torch.finfo(cos_th.dtype).eps)
        theta = torch.arccos(torch.clamp(cos_th, -bound, bound))
        e_angle = P["angle_k"][at] * (theta - P["angle_theta0"][at]) ** 2
        e_angle_node = scatter_sum(e_angle, ai[1], N)

        # proper dihedrals
        di = expand(self.dihedral_index)
        dt = tile(self.dihedral_type)
        e_tors, c_tors = self._periodic(di, dt, P["dihedral_k"], self.ff.dihedral_n,
                                        self.ff.dihedral_cos, self.ff.dihedral_sin,
                                        self.ff.dihedral_mask, data)
        e_tors_node = scatter_sum(e_tors, di[1], N)

        # impropers (center third)
        ii = expand(self.improper_index)
        it = tile(self.improper_type)
        e_impr, c_impr = self._periodic(ii, it, P["improper_k"], self.ff.improper_n,
                                        self.ff.improper_cos, self.ff.improper_sin,
                                        self.ff.improper_mask, data)
        e_impr_node = scatter_sum(e_impr, ii[2], N)

        # nonbonded: neighbor list minus exclusions
        src, dst = data.edge_index[0], data.edge_index[1]
        vec = data.edge_vectors()
        r2 = (vec * vec).sum(1)
        r = torch.sqrt(r2 + TINY)
        q = self.charges.repeat(B)
        rs_t, eps_t = P["rstar"][t], P["epsilon"][t]
        vdw_e, coul_e = self._vdw_coulomb(rs_t[src], rs_t[dst], eps_t[src], eps_t[dst],
                                          q[src], q[dst], r, r2)
        lo = torch.minimum(src, dst)
        hi = torch.maximum(src, dst)
        edge_key = lo * N + hi
        if self.excl_index.shape[1] > 0:
            ex = expand(self.excl_index)
            w = (~torch.isin(edge_key, ex[0] * N + ex[1])).to(r.dtype)
        else:
            w = torch.ones_like(r)
        if self.switch_width > 0.0:
            x = torch.clamp((r - (self.cutoff - self.switch_width)) / self.switch_width,
                            0.0, 1.0)
            w = w * (1.0 - x ** 3 * (10.0 - 15.0 * x + 6.0 * x * x))
        e_vdw_node = scatter_sum(0.5 * w * vdw_e, dst, N)
        e_coul_node = scatter_sum(0.5 * w * coul_e, dst, N)

        # scaled 1,4 pairs (exact, cutoff-independent)
        pi = expand(self.pair14_index)
        p0, p1 = t[pi[0]], t[pi[1]]
        pvec = pair_vectors(data, pi[0], pi[1])
        pr2 = (pvec * pvec).sum(1)
        pr = torch.sqrt(pr2 + TINY)
        vdw14, coul14 = self._vdw_coulomb(P["rstar"][p0], P["rstar"][p1],
                                          P["epsilon"][p0], P["epsilon"][p1],
                                          q[pi[0]], q[pi[1]], pr, pr2)
        vdw14 = self.fudge_lj * vdw14
        coul14 = self.fudge_qq * coul14
        e_vdw14_node = scatter_sum(0.5 * vdw14, pi[0], N) + scatter_sum(0.5 * vdw14, pi[1], N)
        e_coul14_node = scatter_sum(0.5 * coul14, pi[0], N) + scatter_sum(0.5 * coul14, pi[1], N)

        e_ref_node = self.atom_ref(data.atomic_numbers).squeeze(-1)
        node_energy = (e_bond_node + e_angle_node + e_tors_node + e_impr_node
                       + e_vdw_node + e_coul_node + e_vdw14_node + e_coul14_node
                       + e_ref_node)
        energy = self.aggregate_energy(node_energy, data)

        if self.keep_intermediates:
            self.intermediates = dict(
                bond_r=br, e_bond=e_bond, angle_theta=theta, e_angle=e_angle,
                dihedral_cos=c_tors, e_torsion=e_tors, improper_cos=c_impr,
                e_improper=e_impr, nb_r=r, nb_weight=w, nb_vdw=vdw_e,
                nb_coulomb=coul_e, pair14_r=pr, e_vdw14=vdw14, e_coulomb14=coul14,
                type_index=t)

        return {
            "node_energy": node_energy,
            "energy": energy,
            "charges": q,
            "node_features": q.unsqueeze(1),
            "e_bond": self.aggregate_energy(e_bond_node, data),
            "e_angle": self.aggregate_energy(e_angle_node, data),
            "e_torsion": self.aggregate_energy(e_tors_node, data),
            "e_improper": self.aggregate_energy(e_impr_node, data),
            "e_vdw": self.aggregate_energy(e_vdw_node, data),
            "e_coulomb": self.aggregate_energy(e_coul_node, data),
            "e_vdw14": self.aggregate_energy(e_vdw14_node, data),
            "e_coulomb14": self.aggregate_energy(e_coul14_node, data),
            "e_ref": self.aggregate_energy(e_ref_node, data),
        }

    # conveniences
    def export_library(self) -> AmberLibrary:
        """Export the current parameters as an :class:`AmberLibrary`.

        The per-atom charges of this model are written into the residue
        templates of the library wherever the bound topology carries
        residue positions (every atom of a matched residue), so a refit of
        ``charge`` round-trips through the ``.frc`` file.

        Returns
        -------
        AmberLibrary
            See :meth:`AmberForceField.export_library`.
        """
        lib = self.ff.export_library()
        top = self.topology
        names = top.residue_names
        if top.residues and names:
            q = self.charges.detach().cpu()
            for a, r in enumerate(top.residues):
                if r is None or r[0] >= len(names) or names[r[0]] not in lib.fragments:
                    continue
                frag = dict(lib.fragments[names[r[0]]])
                charges = list(frag.get("charges", []))
                if r[1] < len(charges):
                    charges[r[1]] = float(q[a])
                    frag["charges"] = charges
                    lib.fragments[names[r[0]]] = frag
        return lib

    @property
    def masses(self) -> Tensor:
        """Tensor : Per-atom masses (u) of the bound topology."""
        return self.ff.type_mass[self.top_type]

    @classmethod
    def from_config(cls, cfg) -> "Amber":
        """Construct an :class:`Amber` model from a core model config.

        Core field: ``cfg.cutoff``. From ``cfg.extra``: ``library``
        (default ``ff94``) and either ``topology`` (a topology JSON path) or
        ``types`` + ``bonds`` (+ optional ``impropers`` / ``improper_keys``
        / ``charges`` / ``residues``) inline. Optional: ``switch_width``,
        ``fudge_lj``, ``fudge_qq`` (or the AMBER divisors ``scnb`` /
        ``scee``), ``trainable``, ``constrain_charge`` and
        ``atomic_energies``. Alternative spellings are translated by
        :mod:`xnn.common.config.translate`.

        Parameters
        ----------
        cfg : xnn.common.config.schema.ModelConfig
            The core model config.

        Returns
        -------
        Amber
            The model.

        Raises
        ------
        ValueError
            If the topology specification is missing.
        """
        extra = dict(cfg.extra or {})

        def listy(value):
            """Coerce a possibly stringified list to a Python value."""
            if isinstance(value, str) and value.lstrip().startswith(("[", "(")):
                return ast.literal_eval(value)
            return value

        library = extra.get("library", "ff94")
        topo_path = extra.get("topology")
        if topo_path is not None:
            topology: MolecularTopology = read_topology(topo_path)
        else:
            types = listy(extra.get("types"))
            bonds = listy(extra.get("bonds"))
            if types is None or bonds is None:
                raise ValueError("Amber needs a topology: set model.topology "
                                 "to a topology JSON path, or give "
                                 "model.types and model.bonds inline")
            topology = MolecularTopology.from_bonds(
                types, bonds,
                impropers=listy(extra.get("impropers")) or (),
                improper_keys=listy(extra.get("improper_keys")) or (),
                charges=listy(extra.get("charges")),
                residues=listy(extra.get("residues")))

        def floaty(key, inverse_key=None):
            """Optional float from extra (or the inverse of the AMBER-style
            ``scee`` / ``scnb`` divisors)."""
            value = extra.get(key)
            if value is None and inverse_key is not None and extra.get(inverse_key) is not None:
                return 1.0 / float(extra[inverse_key])
            return None if value is None else float(value)

        constrain = extra.get("constrain_charge", True)
        if isinstance(constrain, str):
            constrain = {"true": True, "false": False}.get(constrain.strip().lower(),
                                                           constrain)
            if isinstance(constrain, str):
                constrain = float(constrain)
        return cls(library, topology, cutoff=float(cfg.cutoff),
                   switch_width=float(extra.get("switch_width") or 0.0),
                   fudge_lj=floaty("fudge_lj", "scnb"),
                   fudge_qq=floaty("fudge_qq", "scee"),
                   trainable=listy(extra.get("trainable")) or (),
                   constrain_charge=constrain,
                   atomic_energies=extra.get("atomic_energies"))
