"""CHARMM: the all-atom protein force field of MacKerell et al. (1998).

The CHARMM22 energy function (*J. Phys. Chem. B* 102, 3586, 1998, eq 1) is

    E = sum_bonds K_b (b - b0)^2 + sum_UB K_UB (S - S0)^2
        + sum_angles K_theta (theta - theta0)^2
        + sum_dihedrals K_chi [1 + cos(n chi - delta)]
        + sum_impropers K_imp (phi - phi0)^2
        + sum_{i<j} [ eps_ij ((Rmin_ij / r_ij)^12 - 2 (Rmin_ij / r_ij)^6)
                      + q_i q_j / (eps_1 r_ij) ]

with harmonic bonds, angles and Urey-Bradley 1,3 terms, one or more
Fourier terms per dihedral, harmonic impropers in the dihedral angle over
the four atoms (central atom first), and nonbonded interactions between
atoms separated by three or more bonds. The 1,4 interactions are not
scaled; instead some atom types carry their own Lennard-Jones parameters
for 1,4 pairs, and pair-specific overrides (NBFIX) replace the
Lorentz-Berthelot combination for listed type pairs. Charges belong to
residue templates.

Like OPLS and AMBER, the model binds a parameter library to one molecular
topology (:mod:`xnn.ffnn.models.topology`): per-atom types, charges and
bonds, from which angles, Urey-Bradley pairs, dihedrals, exclusions and 1,4
pairs follow, with the impropers taken from the residue templates
(:func:`~xnn.ffnn.models.charmmlib.place_impropers`).
:meth:`Charmm.from_atoms` types a structure from the residue templates and
atom-type SMARTS of the parameter file.

Conventions and scope
---------------------
* Every parameter lives as a plain tensor in a :class:`CharmmForceField`
  (eV / Angstrom / radians at assembly) and any group can be trained;
  several :class:`Charmm` instances can share one force field. Charges are
  per atom and belong to the model (``trainable=("charge",)``), held to the
  bound system's total charge by default.
* Dihedral phases and periodicities and the improper reference angles are
  fixed; the force constants are the trainable part.
* The 1,4 pairs are the dihedral end atoms that are not also 1,2 or 1,3
  pairs, each counted once (:class:`~xnn.ffnn.models.topology.MolecularTopology`),
  which is the CHARMM bookkeeping also for rings of six atoms or fewer.
* Nonbonded interactions come from the neighbor list within ``cutoff``
  (optionally smoothed over ``switch_width``); the shifting and switching
  functions of the CHARMM program are not reproduced. 1,4 pairs are
  evaluated exactly from the topology. Excluded pairs are excluded in every
  periodic image.
* The Coulomb constant is the CHARMM program's, 332.0716 kcal Angstrom /
  (mol e^2), unless the library or the constructor says otherwise.
* The CMAP cross term of CHARMM22/CMAP and later parameter sets is not
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

from .amber import _exact_trig, multiple_angles
from .charmmlib import KCAL_TO_EV, MAX_TERMS, CharmmLibrary, place_impropers, read_charmm, resolve_improper_type
from .geometry import TINY, dihedral_cos_sin, geometric_mean, pair_vectors
from .oplslib import resolve_angle_type, resolve_bond_type, resolve_dihedral_type
from .topology import MolecularTopology, read_topology, residue_name_list

DEG = math.pi / 180.0

# Trainable parameter groups of the force field ("charge" and "atom_ref"
# are model-level groups).
_GROUPS = ("rstar", "epsilon", "rstar14", "epsilon14", "nbfix_rmin",
           "nbfix_eps", "bond_k", "bond_r0", "angle_k", "angle_theta0",
           "ub_k", "ub_s0", "dihedral_k", "improper_k")
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


def _as_library(source: Union[CharmmLibrary, str, Path]) -> CharmmLibrary:
    """Coerce a library argument to a :class:`CharmmLibrary`."""
    if isinstance(source, CharmmLibrary):
        return source
    return read_charmm(str(source))


class CharmmForceField(nn.Module):
    """The trainable parameter tensors of a CHARMM parameter library.

    Parameters
    ----------
    library : CharmmLibrary, str or Path
        The parameter library (or a shipped set such as ``"charmm22"``, a
        ``.frc`` path or a native JSON path, see
        :func:`~xnn.ffnn.models.charmmlib.read_charmm`).
    trainable : sequence of str or "all", optional
        Groups to expose to the optimizer: ``"rstar"``, ``"epsilon"``,
        ``"rstar14"``, ``"epsilon14"``, ``"nbfix_rmin"``, ``"nbfix_eps"``,
        ``"bond_k"``, ``"bond_r0"``, ``"angle_k"``, ``"angle_theta0"``,
        ``"ub_k"``, ``"ub_s0"``, ``"dihedral_k"``, ``"improper_k"``;
        ``"all"`` unfreezes every group.

    Attributes
    ----------
    params : torch.nn.ParameterDict
        ``rstar`` ``(T,)`` (``Rmin/2``) in Angstrom and ``epsilon`` ``(T,)``
        in eV, ``rstar14`` / ``epsilon14`` ``(T,)`` the 1,4 values (equal to
        the regular ones where a type has none; ``own14`` marks the types
        whose values count), ``nbfix_rmin`` / ``nbfix_eps`` and
        ``nbfix_rmin14`` / ``nbfix_eps14`` ``(M,)`` the pair overrides,
        ``bond_k`` ``(NB,)`` in eV/A^2, ``bond_r0`` ``(NB,)``, ``angle_k``
        ``(NA,)`` in eV/rad^2, ``angle_theta0`` ``(NA,)`` in rad, ``ub_k``
        ``(NU,)`` in eV/A^2, ``ub_s0`` ``(NU,)``, ``dihedral_k`` ``(ND, 6)``
        in eV (one column per Fourier term, zero where a row has fewer
        terms) and ``improper_k`` ``(NI,)`` in eV/rad^2.
    dihedral_n, dihedral_cos, dihedral_sin, dihedral_mask : Tensor
        Periodicities, the cosine / sine of the phases and a 0/1 mask of the
        used term slots of the torsion rows (fixed buffers).
    improper_cos, improper_sin : Tensor
        The cosine / sine of the improper reference angles (fixed buffers).
    type_names, bond_keys, angle_keys, ub_keys, dihedral_keys, improper_keys, nbfix_keys : list
        Row labels of the parameter tensors.
    coulomb_constant, fudge_qq : float
        The library's Coulomb constant (kcal Angstrom / mol / e^2) and its
        1,4 electrostatic scaling.
    """

    def __init__(self, library: Union[CharmmLibrary, str, Path],
                 trainable: Union[Sequence[str], str] = ()):
        super().__init__()
        lib = _as_library(library)
        dtype = torch.get_default_dtype()
        self.name = lib.name
        self.coulomb_constant = float(lib.coulomb_constant)
        self.fudge_qq = float(lib.scale_14_coulomb)
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
        self.ub_keys = list(lib.urey_bradley_types)
        self.dihedral_keys = list(lib.dihedral_types)
        self.improper_keys = list(lib.improper_types)
        self.nbfix_keys = list(lib.nbfix_types)
        self._bond_types = dict(lib.bond_types)
        self._angle_types = dict(lib.angle_types)
        self._ub_types = dict(lib.urey_bradley_types)
        self._dihedral_types = dict(lib.dihedral_types)
        self._improper_types = dict(lib.improper_types)
        self._bond_index = {k: i for i, k in enumerate(self.bond_keys)}
        self._angle_index = {k: i for i, k in enumerate(self.angle_keys)}
        self._ub_index = {k: i for i, k in enumerate(self.ub_keys)}
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
        own14 = [a.get("rstar14") is not None and a.get("epsilon14") is not None
                 for a in at]
        make("rstar14", [float(a["rstar14"]) if o else float(a["rstar"])
                         for a, o in zip(at, own14)])
        make("epsilon14", [float(a["epsilon14"] if o else a["epsilon"]) * KCAL_TO_EV
                           for a, o in zip(at, own14)])
        self.register_buffer("own14", torch.tensor(own14, dtype=torch.bool))

        nb_i, nb_j = [], []
        for key in self.nbfix_keys:
            a, b = key.split("-")
            nb_i.append(self.type_index(a))
            nb_j.append(self.type_index(b))
        self.register_buffer("nbfix_i", torch.tensor(nb_i, dtype=torch.long))
        self.register_buffer("nbfix_j", torch.tensor(nb_j, dtype=torch.long))
        rows = [lib.nbfix_types[k] for k in self.nbfix_keys]
        make("nbfix_rmin", [float(r["rmin"]) for r in rows])
        make("nbfix_eps", [float(r["eps"]) * KCAL_TO_EV for r in rows])
        make("nbfix_rmin14", [float(r["rmin"] if r.get("rmin14") is None else r["rmin14"])
                              for r in rows])
        make("nbfix_eps14", [float(r["eps"] if r.get("eps14") is None else r["eps14"])
                             * KCAL_TO_EV for r in rows])
        # the 1,4 override parameters move with the regular ones unless listed
        self.params["nbfix_rmin14"].requires_grad_("nbfix_rmin" in trainable_set)
        self.params["nbfix_eps14"].requires_grad_("nbfix_eps" in trainable_set)

        make("bond_k", [float(lib.bond_types[k]["k"]) * KCAL_TO_EV
                        for k in self.bond_keys])
        make("bond_r0", [float(lib.bond_types[k]["r0"]) for k in self.bond_keys])
        make("angle_k", [float(lib.angle_types[k]["k"]) * KCAL_TO_EV
                         for k in self.angle_keys])
        make("angle_theta0", [float(lib.angle_types[k]["theta0"]) * DEG
                              for k in self.angle_keys])
        make("ub_k", [float(lib.urey_bradley_types[k]["k"]) * KCAL_TO_EV
                      for k in self.ub_keys])
        make("ub_s0", [float(lib.urey_bradley_types[k]["s0"]) for k in self.ub_keys])

        ks, ns, phs, ms = [], [], [], []
        for key in self.dihedral_keys:
            terms = list(lib.dihedral_types[key]["terms"])[:MAX_TERMS]
            pad = MAX_TERMS - len(terms)
            ks.append([float(t[0]) * KCAL_TO_EV for t in terms] + [0.0] * pad)
            ns.append([max(int(t[1]), 1) for t in terms] + [1] * pad)
            phs.append([float(t[2]) * DEG for t in terms] + [0.0] * pad)
            ms.append([1.0] * len(terms) + [0.0] * pad)
        nd = len(self.dihedral_keys)
        self.params["dihedral_k"] = Parameter(
            torch.tensor(ks, dtype=dtype).reshape(nd, MAX_TERMS),
            requires_grad="dihedral_k" in trainable_set)
        self.register_buffer("dihedral_n", torch.tensor(ns, dtype=torch.long).reshape(nd, MAX_TERMS))
        self.register_buffer("dihedral_mask", torch.tensor(ms, dtype=dtype).reshape(nd, MAX_TERMS))
        c, s = _exact_trig(torch.tensor(phs, dtype=torch.float64).reshape(nd, MAX_TERMS))
        self.register_buffer("dihedral_cos", c.to(dtype))
        self.register_buffer("dihedral_sin", s.to(dtype))

        make("improper_k", [float(lib.improper_types[k]["k"]) * KCAL_TO_EV
                            for k in self.improper_keys])
        c, s = _exact_trig(torch.tensor(
            [float(lib.improper_types[k].get("psi0", 0.0)) * DEG for k in self.improper_keys],
            dtype=torch.float64))
        self.register_buffer("improper_cos", c.to(dtype))
        self.register_buffer("improper_sin", s.to(dtype))

    # resolution (topology binding)
    def type_index(self, name: str) -> int:
        """Row index of atom type ``name`` (``KeyError`` if unknown)."""
        if name not in self._type_index:
            raise KeyError(f"atom type {name!r} is not in library {self.name!r}")
        return self._type_index[name]

    def resolve_bond(self, a: str, b: str) -> int:
        """Bond-type row for the type pair (``KeyError`` if missing)."""
        key = resolve_bond_type(self._bond_types, a, b)
        if key is None:
            raise KeyError(f"no bond type {a}-{b} in library {self.name!r}")
        return self._bond_index[key]

    def resolve_angle(self, a: str, b: str, c: str) -> int:
        """Angle-type row for the type triple, center second."""
        key = resolve_angle_type(self._angle_types, a, b, c)
        if key is None:
            raise KeyError(f"no angle type {a}-{b}-{c} in library {self.name!r}")
        return self._angle_index[key]

    def resolve_urey_bradley(self, a: str, b: str, c: str) -> Optional[int]:
        """Urey-Bradley row for the type triple, or ``None`` when the angle has no such term."""
        key = resolve_angle_type(self._ub_types, a, b, c)
        return None if key is None else self._ub_index[key]

    def resolve_dihedral(self, a: str, b: str, c: str, d: str) -> int:
        """Dihedral-type row: the exact quadruple (either direction) wins
        over the ``X-B-C-X`` generic entry."""
        key = resolve_dihedral_type(self._dihedral_types, a, b, c, d)
        if key is None:
            raise KeyError(f"no dihedral type {a}-{b}-{c}-{d} in library "
                           f"{self.name!r}")
        return self._dihedral_index[key]

    def resolve_improper(self, *args) -> int:
        """Improper-type row, by exact key or by the four types in
        evaluation order (central atom first)."""
        if len(args) == 1:
            key = args[0]
            if key not in self._improper_index:
                rev = "-".join(key.split("-")[::-1])
                if rev not in self._improper_index:
                    raise KeyError(f"no improper type {key!r} in library {self.name!r}")
                key = rev
            return self._improper_index[key]
        i, j, k, l = args
        key = resolve_improper_type(self._improper_types, i, j, k, l)
        if key is None:
            raise KeyError(f"no improper type for {i}-{j}-{k}-{l} (center {i}) "
                           f"in library {self.name!r}")
        return self._improper_index[key]

    # pair tables
    def pair_tables(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """The per-type-pair Lennard-Jones tables ``(rmin, eps, rmin14, eps14)``.

        Returns
        -------
        tuple of Tensor
            Four ``(T, T)`` tensors: the Lorentz-Berthelot combination of
            the per-type parameters (the regular ones, and the 1,4 ones for
            the types that have their own), with the pair overrides written
            over the listed pairs. ``rmin`` is the pair's minimum distance
            in Angstrom, ``eps`` its well depth in eV.
        """
        P = self.params
        rs, eps = P["rstar"], P["epsilon"]
        rs14 = torch.where(self.own14, P["rstar14"], rs)
        eps14 = torch.where(self.own14, P["epsilon14"], eps)
        tables = []
        for r, e, pr, pe in ((rs, eps, P["nbfix_rmin"], P["nbfix_eps"]),
                             (rs14, eps14, P["nbfix_rmin14"], P["nbfix_eps14"])):
            rmin = r[:, None] + r[None, :]
            well = geometric_mean(e[:, None].expand_as(rmin), e[None, :].expand_as(rmin))
            if self.nbfix_i.numel() > 0:
                ii = torch.cat([self.nbfix_i, self.nbfix_j])
                jj = torch.cat([self.nbfix_j, self.nbfix_i])
                rmin = rmin.index_put((ii, jj), torch.cat([pr, pr]))
                well = well.index_put((ii, jj), torch.cat([pe, pe]))
            tables += [rmin, well]
        return tuple(tables)

    # export
    def export_library(self) -> CharmmLibrary:
        """The current (possibly trained) parameters as a :class:`CharmmLibrary`.

        Returns
        -------
        CharmmLibrary
            In CHARMM units (kcal/mol, Angstrom, degrees); the residue
            templates keep the library's charges (trained per-atom charges
            belong to the models, see :meth:`Charmm.export_library`).
        """
        P = {k: v.detach().cpu() for k, v in self.params.items()}
        atom_types = {}
        for i, name in enumerate(self.type_names):
            own = bool(self.own14[i])
            atom_types[name] = {
                "element": int(self.type_z[i]), "mass": float(self.type_mass[i]),
                "rstar": float(P["rstar"][i]),
                "epsilon": float(P["epsilon"][i]) / KCAL_TO_EV,
                "rstar14": float(P["rstar14"][i]) if own else None,
                "epsilon14": float(P["epsilon14"][i]) / KCAL_TO_EV if own else None,
                "connections": 0, "comment": ""}
        dihedral_types = {}
        for i, key in enumerate(self.dihedral_keys):
            terms = self._dihedral_types[key]["terms"][:MAX_TERMS]
            dihedral_types[key] = {"terms": [
                [float(P["dihedral_k"][i, m]) / KCAL_TO_EV, int(n), float(ph)]
                for m, (_, n, ph) in enumerate(terms)]}
        return CharmmLibrary(
            atom_types=atom_types,
            bond_types={k: {"k": float(P["bond_k"][i]) / KCAL_TO_EV,
                            "r0": float(P["bond_r0"][i])}
                        for i, k in enumerate(self.bond_keys)},
            angle_types={k: {"k": float(P["angle_k"][i]) / KCAL_TO_EV,
                             "theta0": float(P["angle_theta0"][i]) / DEG}
                         for i, k in enumerate(self.angle_keys)},
            urey_bradley_types={k: {"k": float(P["ub_k"][i]) / KCAL_TO_EV,
                                    "s0": float(P["ub_s0"][i])}
                                for i, k in enumerate(self.ub_keys)},
            dihedral_types=dihedral_types,
            improper_types={k: {"k": float(P["improper_k"][i]) / KCAL_TO_EV,
                                "psi0": float(self._improper_types[k].get("psi0", 0.0))}
                            for i, k in enumerate(self.improper_keys)},
            nbfix_types={k: {"rmin": float(P["nbfix_rmin"][i]),
                             "eps": float(P["nbfix_eps"][i]) / KCAL_TO_EV,
                             "rmin14": float(P["nbfix_rmin14"][i]),
                             "eps14": float(P["nbfix_eps14"][i]) / KCAL_TO_EV}
                         for i, k in enumerate(self.nbfix_keys)},
            coulomb_constant=self.coulomb_constant, scale_14_coulomb=self.fudge_qq,
            name=self.name, references=list(self.references),
            templates=dict(self.templates), fragments=dict(self.fragments),
            metadata=dict(self.metadata))


@register_model("charmm")
class Charmm(InteratomicPotential):
    """The CHARMM fixed-topology force field (MacKerell *et al.* 1998 form).

    An instance binds a parameter library to one molecular topology; the
    bonded terms, impropers, exclusions and 1,4 pairs are resolved once,
    and ``forward`` evaluates the energy on any conformation (or batch of
    conformations) of that system. Forces and stress come from autograd via
    :class:`~xnn.common.models.outputs.ForceStressOutput`.

    Parameters
    ----------
    ffield : CharmmLibrary, CharmmForceField, str or Path
        The parameter library -- a parsed :class:`CharmmLibrary`, a set
        shipped with xnn (``"charmm22"``), a ``.frc`` or native JSON path,
        or a :class:`CharmmForceField` to share parameters with other
        models.
    topology : MolecularTopology, str or Path
        The system's topology (or a topology JSON path). Its ``charges``
        are the partial charges unless ``charges`` is given; its
        ``residues`` select the impropers of the residue templates.
    cutoff : float, optional
        Nonbonded cutoff in Angstrom (the neighbor-list cutoff); by default
        12.0. 1,4 pairs are independent of it.
    switch_width : float, optional
        Width of a quintic switching function taking the nonbonded
        interactions smoothly to zero at the cutoff; 0.0 (plain truncation)
        by default.
    charges : sequence of float, optional
        Per-atom partial charges in e, overriding the topology's.
    fudge_qq : float, optional
        Override the library's 1,4 electrostatic scaling (1.0, CHARMM's
        ``e14fac``).
    coulomb_constant : float, optional
        Override the library's Coulomb constant (kcal Angstrom / mol / e^2;
        332.0716 in the CHARMM program, 332.0636 in NAMD, 332.06371 in
        LAMMPS ``real`` units).
    trainable : sequence of str or "all", optional
        Force-field groups (see :class:`CharmmForceField`) plus the model
        groups ``"charge"`` (per-atom charges) and ``"atom_ref"``
        (per-element reference energies); ``"all"`` unfreezes everything.
    keep_intermediates : bool, optional
        Stash the intermediate tensors of the last evaluation in
        ``self.intermediates``. Default ``False``.
    auto_impropers : bool, optional
        When the topology lists no impropers, take them from the residue
        templates of the matched residues
        (:func:`~xnn.ffnn.models.charmmlib.place_impropers`). Default
        ``True``. Listed impropers are used as given, central atom first,
        resolved by their atom types (or by ``improper_keys``).
    constrain_charge : bool or float, optional
        Hold the bound system's total charge when ``charge`` trains:
        ``True`` (default) keeps the sum of the initial charges, a number
        fixes that value in e, ``False`` registers no constraint.
    atomic_energies : mapping, optional
        Per-element reference energies in eV, ``{Z or symbol: E0}``, loaded
        into ``atom_ref``. All zero by default.

    Notes
    -----
    ``forward`` returns, besides ``node_energy`` / ``energy`` /
    ``node_features``, the partial ``charges`` ``(N,)`` and one
    per-structure tensor per term: ``e_bond``, ``e_angle``,
    ``e_urey_bradley``, ``e_torsion``, ``e_improper``, ``e_vdw``,
    ``e_coulomb``, ``e_vdw14``, ``e_coulomb14`` and ``e_ref``. Every
    structure in a batch must have this topology's atom count and element
    sequence.

    Attributes
    ----------
    atom_ref : torch.nn.Embedding
        The per-element reference energies, row ``Z``.
    species : list of int
        The atomic numbers of the bound topology.
    impropers : list of tuple
        The improper quadruples in evaluation order (central atom first).
    """

    def __init__(self, ffield: Union[CharmmLibrary, CharmmForceField, str, Path],
                 topology: Union[MolecularTopology, str, Path], *,
                 cutoff: float = 12.0,
                 switch_width: float = 0.0,
                 charges: Optional[Sequence[float]] = None,
                 fudge_qq: Optional[float] = None,
                 coulomb_constant: Optional[float] = None,
                 trainable: Union[Sequence[str], str] = (),
                 keep_intermediates: bool = False,
                 auto_impropers: bool = True,
                 constrain_charge: Union[bool, float] = True,
                 atomic_energies=None):
        super().__init__()
        groups, model_groups = _split_trainable(trainable)
        self.ff = ffield if isinstance(ffield, CharmmForceField) \
            else CharmmForceField(ffield, trainable=groups)
        self.atom_ref = nn.Embedding(_MAX_Z, 1)
        nn.init.zeros_(self.atom_ref.weight)
        self.atom_ref.weight.requires_grad_("atom_ref" in model_groups)
        self._train_charge = "charge" in model_groups
        self.auto_impropers = bool(auto_impropers)
        self.cutoff = float(cutoff)
        self.switch_width = float(switch_width)
        if not 0.0 <= self.switch_width < self.cutoff:
            raise ValueError("switch_width must satisfy 0 <= switch_width < cutoff")
        self.fudge_qq = self.ff.fudge_qq if fudge_qq is None else float(fudge_qq)
        self.coulomb_constant = self.ff.coulomb_constant if coulomb_constant is None \
            else float(coulomb_constant)
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
    def from_atoms(cls, structure, ffield: Union[CharmmLibrary, CharmmForceField,
                                                 str, Path] = "charmm22",
                   *, charge: int = 0,
                   bonds: Optional[Sequence[Sequence[int]]] = None,
                   charges: Optional[Sequence[float]] = None,
                   orders: bool = True,
                   **kwargs) -> "Charmm":
        """Build a CHARMM model for a structure, typing it from the library.

        Residue templates (the ``#fragments`` of the parameter file) assign
        atom types, partial charges and impropers to every atom they match;
        atoms outside any residue are typed by the per-type SMARTS templates
        and carry zero charge unless ``charges`` is given.

        Parameters
        ----------
        structure : object
            An ``ase.Atoms``, a ``(positions, atomic_numbers)`` pair, a dict
            with ``"pos"`` / ``"atomic_numbers"``, an RDKit molecule or a
            SMILES string.
        ffield : CharmmLibrary, CharmmForceField, str or Path, optional
            The parameter library (must carry templates); default
            ``"charmm22"``.
        charge : int, optional
            Total charge of the structure (for bond-order perception).
        bonds : sequence of (int, int), optional
            Known connectivity; bond orders are then perceived, not bonds.
        charges : sequence of float, optional
            Per-atom partial charges overriding the templates'.
        orders : bool, optional
            With ``bonds`` given, ``False`` skips the perception of bond
            orders and formal charges (the residue templates need only the
            connectivity), which is robust for charged systems and ions.
        **kwargs
            Forwarded to the constructor (``cutoff``, ``trainable``, ...).

        Returns
        -------
        Charmm
            The model, bound to the derived topology.
        """
        from ..common.typing import assign_atom_types, perceive_bonds
        lib = ffield if isinstance(ffield, CharmmForceField) else _as_library(ffield)
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
        names = [ff.type_names[i] for i in t_idx]
        b_type = [gather(ff.resolve_bond, names[i], names[j]) for i, j in top.bonds]
        a_type = [gather(ff.resolve_angle, names[i], names[j], names[k])
                  for i, j, k in top.angles]
        ub_pairs, ub_type = [], []
        for i, j, k in top.angles:
            row = ff.resolve_urey_bradley(names[i], names[j], names[k])
            if row is not None:
                ub_pairs.append((i, k))
                ub_type.append(row)
        d_type = [gather(ff.resolve_dihedral, names[i], names[j], names[k], names[l])
                  for i, j, k, l in top.dihedrals]
        impropers = [tuple(int(x) for x in im) for im in top.impropers]
        if top.improper_keys:
            i_type = [gather(ff.resolve_improper, key) for key in top.improper_keys]
        elif impropers:
            i_type = [gather(ff.resolve_improper, names[i], names[j], names[k], names[l])
                      for i, j, k, l in impropers]
        elif self.auto_impropers and top.residues:
            elements = [CHEMICAL_SYMBOLS[int(ff.type_z[i])] if 0 < int(ff.type_z[i])
                        < len(CHEMICAL_SYMBOLS) else "X" for i in t_idx]
            impropers = place_impropers(top, ff.fragments, elements)
            i_type = [gather(ff.resolve_improper, names[i], names[j], names[k], names[l])
                      for i, j, k, l in impropers]
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
        buf("ub_index", [list(p) for p in ub_pairs], (-1, 2))
        self.ub_index = self.ub_index.t().contiguous()
        buf("ub_type", ub_type, (-1,))
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
    @staticmethod
    def _lj(rmin: Tensor, eps: Tensor, r2: Tensor) -> Tensor:
        """``eps [(Rmin/r)^12 - 2 (Rmin/r)^6]`` per pair."""
        x = (rmin * rmin / (r2 + TINY)) ** 3
        return eps * (x * x - 2.0 * x)

    # forward
    def forward(self, data: AtomicGraph) -> dict[str, Tensor]:
        """Evaluate the CHARMM energy on a (batched) atomic graph.

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

        # Urey-Bradley 1,3 terms
        ui = expand(self.ub_index)
        ut = tile(self.ub_type)
        uvec = pair_vectors(data, ui[0], ui[1])
        ur = torch.sqrt((uvec * uvec).sum(1) + TINY)
        e_ub = P["ub_k"][ut] * (ur - P["ub_s0"][ut]) ** 2
        e_ub_node = scatter_sum(0.5 * e_ub, ui[0], N) + scatter_sum(0.5 * e_ub, ui[1], N)

        # proper dihedrals
        di = expand(self.dihedral_index)
        dt = tile(self.dihedral_type)
        c1, s1 = dihedral_cos_sin(data, di)
        n_per = self.ff.dihedral_n[dt]
        n_max = int(n_per.max()) if n_per.numel() else 1
        cos_n, sin_n = multiple_angles(c1, s1, n_max)
        cn = torch.gather(cos_n, 1, n_per - 1)
        sn = torch.gather(sin_n, 1, n_per - 1)
        e_tors = (P["dihedral_k"][dt] * self.ff.dihedral_mask[dt]
                  * (1.0 + self.ff.dihedral_cos[dt] * cn + self.ff.dihedral_sin[dt] * sn)).sum(1)
        e_tors_node = scatter_sum(e_tors, di[1], N)

        # impropers: K (phi - phi0)^2 on the signed dihedral over the atoms as listed
        ii = expand(self.improper_index)
        it = tile(self.improper_type)
        ci, si = dihedral_cos_sin(data, ii)
        c0, s0 = self.ff.improper_cos[it], self.ff.improper_sin[it]
        dphi = torch.atan2(si * c0 - ci * s0, ci * c0 + si * s0)
        e_impr = P["improper_k"][it] * dphi * dphi
        e_impr_node = scatter_sum(e_impr, ii[0], N)

        # nonbonded: neighbor list minus exclusions, pair tables with overrides
        rmin_tab, eps_tab, rmin14_tab, eps14_tab = self.ff.pair_tables()
        src, dst = data.edge_index[0], data.edge_index[1]
        vec = data.edge_vectors()
        r2 = (vec * vec).sum(1)
        r = torch.sqrt(r2 + TINY)
        q = self.charges.repeat(B)
        ke = self.coulomb_constant * KCAL_TO_EV
        ts, td = t[src], t[dst]
        vdw_e = self._lj(rmin_tab[ts, td], eps_tab[ts, td], r2)
        coul_e = ke * q[src] * q[dst] / r
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

        # 1,4 pairs (exact, cutoff-independent) with their own LJ parameters
        pi = expand(self.pair14_index)
        p0, p1 = t[pi[0]], t[pi[1]]
        pvec = pair_vectors(data, pi[0], pi[1])
        pr2 = (pvec * pvec).sum(1)
        pr = torch.sqrt(pr2 + TINY)
        vdw14 = self._lj(rmin14_tab[p0, p1], eps14_tab[p0, p1], pr2)
        coul14 = self.fudge_qq * ke * q[pi[0]] * q[pi[1]] / pr
        e_vdw14_node = scatter_sum(0.5 * vdw14, pi[0], N) + scatter_sum(0.5 * vdw14, pi[1], N)
        e_coul14_node = scatter_sum(0.5 * coul14, pi[0], N) + scatter_sum(0.5 * coul14, pi[1], N)

        e_ref_node = self.atom_ref(data.atomic_numbers).squeeze(-1)
        node_energy = (e_bond_node + e_angle_node + e_ub_node + e_tors_node
                       + e_impr_node + e_vdw_node + e_coul_node + e_vdw14_node
                       + e_coul14_node + e_ref_node)
        energy = self.aggregate_energy(node_energy, data)

        if self.keep_intermediates:
            self.intermediates = dict(
                bond_r=br, e_bond=e_bond, angle_theta=theta, e_angle=e_angle,
                ub_r=ur, e_urey_bradley=e_ub, dihedral_cos=c1, dihedral_sin=s1,
                e_torsion=e_tors, improper_dphi=dphi, e_improper=e_impr, nb_r=r,
                nb_weight=w, nb_vdw=vdw_e, nb_coulomb=coul_e, pair14_r=pr,
                e_vdw14=vdw14, e_coulomb14=coul14, type_index=t)

        return {
            "node_energy": node_energy,
            "energy": energy,
            "charges": q,
            "node_features": q.unsqueeze(1),
            "e_bond": self.aggregate_energy(e_bond_node, data),
            "e_angle": self.aggregate_energy(e_angle_node, data),
            "e_urey_bradley": self.aggregate_energy(e_ub_node, data),
            "e_torsion": self.aggregate_energy(e_tors_node, data),
            "e_improper": self.aggregate_energy(e_impr_node, data),
            "e_vdw": self.aggregate_energy(e_vdw_node, data),
            "e_coulomb": self.aggregate_energy(e_coul_node, data),
            "e_vdw14": self.aggregate_energy(e_vdw14_node, data),
            "e_coulomb14": self.aggregate_energy(e_coul14_node, data),
            "e_ref": self.aggregate_energy(e_ref_node, data),
        }

    # conveniences
    def export_library(self) -> CharmmLibrary:
        """Export the current parameters as a :class:`CharmmLibrary`.

        The per-atom charges of this model are written into the residue
        templates of the library wherever the bound topology carries
        residue positions, so a refit of ``charge`` round-trips through the
        ``.frc`` file.

        Returns
        -------
        CharmmLibrary
            See :meth:`CharmmForceField.export_library`.
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
    def from_config(cls, cfg) -> "Charmm":
        """Construct a :class:`Charmm` model from a core model config.

        Core field: ``cfg.cutoff``. From ``cfg.extra``: ``library``
        (default ``charmm22``) and either ``topology`` (a topology JSON
        path) or ``types`` + ``bonds`` (+ optional ``impropers`` /
        ``improper_keys`` / ``charges`` / ``residues`` / ``residue_names``)
        inline. Optional: ``switch_width``, ``fudge_qq`` (CHARMM's
        ``e14fac``), ``coulomb_constant``, ``trainable``,
        ``constrain_charge`` and ``atomic_energies``. Alternative spellings
        are translated by :mod:`xnn.common.config.translate`.

        Parameters
        ----------
        cfg : xnn.common.config.schema.ModelConfig
            The core model config.

        Returns
        -------
        Charmm
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

        library = extra.get("library", "charmm22")
        topo_path = extra.get("topology")
        if topo_path is not None:
            topology: MolecularTopology = read_topology(topo_path)
        else:
            types = listy(extra.get("types"))
            bonds = listy(extra.get("bonds"))
            if types is None or bonds is None:
                raise ValueError("Charmm needs a topology: set model.topology "
                                 "to a topology JSON path, or give "
                                 "model.types and model.bonds inline")
            topology = MolecularTopology.from_bonds(
                types, bonds,
                impropers=listy(extra.get("impropers")) or (),
                improper_keys=listy(extra.get("improper_keys")) or (),
                charges=listy(extra.get("charges")),
                residues=listy(extra.get("residues")),
                residue_names=listy(extra.get("residue_names")))

        def floaty(key):
            """Optional float from extra."""
            value = extra.get(key)
            return None if value is None else float(value)

        constrain = extra.get("constrain_charge", True)
        if isinstance(constrain, str):
            constrain = {"true": True, "false": False}.get(constrain.strip().lower(),
                                                           constrain)
            if isinstance(constrain, str):
                constrain = float(constrain)
        return cls(library, topology, cutoff=float(cfg.cutoff),
                   switch_width=float(extra.get("switch_width") or 0.0),
                   fudge_qq=floaty("fudge_qq"),
                   coulomb_constant=floaty("coulomb_constant"),
                   trainable=listy(extra.get("trainable")) or (),
                   constrain_charge=constrain,
                   atomic_energies=extra.get("atomic_energies"))
