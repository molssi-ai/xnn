"""OPLS: the all-atom (and united-atom) fixed-topology force field.

OPLS (optimized potentials for liquid simulations; Jorgensen, Maxwell &
Tirado-Rives, *J. Am. Chem. Soc.* 118, 11225, 1996) writes the total energy
as harmonic bond and angle terms, a three(four)-term Fourier series per
proper dihedral, ``V2``-only improper dihedrals at trigonal centers, and
Coulomb plus Lennard-Jones nonbonded interactions between all pairs of atoms
separated by three or more bonds:

    E_ab = sum_ij [ q_i q_j e^2 / r_ij
                    + 4 eps_ij (sigma_ij^12/r_ij^12 - sigma_ij^6/r_ij^6) ] f_ij

with geometric combining rules ``sigma_ij = (sigma_i sigma_j)^1/2``,
``eps_ij = (eps_i eps_j)^1/2`` and ``f_ij = 1`` except for intramolecular
1,4 pairs, where ``f_ij = 1/2`` (eqs 1-4 of the paper). The united-atom
variant (OPLS-UA) and reparameterizations such as L-OPLS for long
hydrocarbons (Siu, Pluhackova & Boeckmann, *J. Chem. Theory Comput.* 8,
1459, 2012) share this functional form and differ only in their parameter
libraries, so all of them are served by this one model plus a library
(see :mod:`xnns.ffnn.models.oplslib`).

Unlike ReaxFF, OPLS is not reactive: it needs a fixed molecular topology --
per-atom OPLS types and the bond list, from which angles, dihedrals,
exclusions and 1,4 pairs follow (:mod:`xnns.ffnn.models.topology`). The
topology is bound to the model instance; every structure evaluated by the
model must be a conformation of that same system (this is what lets batches
of conformers flow through the standard dataset / trainer / ASE-calculator
machinery unchanged). Bonded terms and 1,4 pairs use minimum-image
displacements, so molecules may wrap across periodic boundaries.

Conventions and scope
---------------------
* All parameters live in a :class:`OPLSForceField` module as plain tensors
  (converted to eV / Angstrom / radians at assembly); any group can be made
  trainable (``trainable=...``), so classical parameters can be refit by
  gradient descent exactly like ReaxFF's. Several :class:`OPLS` instances
  (different molecules) can share one :class:`OPLSForceField` to fit
  transferable parameters jointly.
* Nonbonded interactions are evaluated on the model's neighbor list within
  ``cutoff`` and truncated there (optionally smoothed over the last
  ``switch_width`` Angstrom with a quintic switching function); 1,4 pairs
  are evaluated exactly from the topology, independent of the cutoff, and
  scaled by the library's ``fudge`` factors.
* Excluded (1,2 / 1,3 / 1,4) pairs are excluded in *every* periodic image,
  the standard molecular-mechanics convention; keep the cutoff below half
  the box length, as usual.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Optional, Sequence, Union

import torch
from torch import Tensor, nn
from torch.nn import Parameter, ParameterDict

from xnns.common.data import AtomicGraph
from xnns.common.models import InteratomicPotential, register_model
from xnns.common.models.ops import scatter_sum

from .oplslib import (OPLSLibrary, read_opls, resolve_bond_type,
                      resolve_angle_type, resolve_dihedral_type,
                      resolve_improper_type, KCAL_TO_EV)
from .topology import MolecularTopology, read_topology

# Coulomb constant e^2 / (4 pi eps_0) in eV * Angstrom: the OpenMM/CODATA
# value 138.935456 kJ/mol nm converted with 96.48533212331 kJ/mol per eV.
# (ReaxFF uses its own historical constant; see the note in oplslib.py.)
KE = 14.399645307997487
# Small numerical guard applied under square roots.
TINY = 1.0e-12
DEG = math.pi / 180.0

# Trainable parameter groups of the force field.
_GROUPS = ("charge", "sigma", "epsilon", "bond_k", "bond_r0", "angle_k",
           "angle_theta0", "dihedral_v", "improper_v2")


def _as_library(source: Union[OPLSLibrary, str, Path]) -> OPLSLibrary:
    """Coerce a library argument to an :class:`OPLSLibrary`.

    Parameters
    ----------
    source : OPLSLibrary, str or Path
        A parsed library, or anything :func:`~xnns.ffnn.models.oplslib.read_opls`
        accepts: a built-in variant name (``"oplsaa"``, ``"oplsaa-1996"``,
        ``"lopls"``, ``"CL&P"``), a ``.frc`` path (``"file.frc:variant"`` to
        pick one of several), or a native JSON path.

    Returns
    -------
    OPLSLibrary
        The library.
    """
    if isinstance(source, OPLSLibrary):
        return source
    return read_opls(str(source))


def _geometric_mean(x: Tensor, y: Tensor) -> Tensor:
    """Geometric mean with a differentiable zero (for zero-LJ hydrogens).

    Parameters
    ----------
    x, y : Tensor
        Non-negative parameter values, gathered per pair.

    Returns
    -------
    Tensor
        ``sqrt(x * y)``, exactly zero (with zero gradient) where the
        product vanishes.
    """
    prod = x * y
    safe = torch.sqrt(torch.clamp(prod, min=TINY))
    return torch.where(prod > 0.0, safe, torch.zeros_like(prod))


class OPLSForceField(nn.Module):
    """The trainable parameter tensors of an OPLS parameter library.

    Assembles the library's per-atom-type nonbonded parameters and per-class
    bonded parameter tables into dense tensors (eV / Angstrom / radians),
    exposed as a :class:`torch.nn.ParameterDict` so any group can be refit
    by gradient descent. One instance can be shared by several
    :class:`OPLS` models (different molecules) to train transferable
    parameters jointly.

    Parameters
    ----------
    library : OPLSLibrary, str or Path
        The parameter library (or a built-in set name / file path, see
        :func:`~xnns.ffnn.models.oplslib.read_opls`).
    trainable : sequence of str or "all", optional
        Parameter groups to expose to the optimizer: any of ``"charge"``,
        ``"sigma"``, ``"epsilon"``, ``"bond_k"``, ``"bond_r0"``,
        ``"angle_k"``, ``"angle_theta0"``, ``"dihedral_v"``,
        ``"improper_v2"``; ``"all"`` unfreezes every group. Default: none
        (a fixed classical force field).

    Attributes
    ----------
    params : torch.nn.ParameterDict
        ``charge`` ``(T,)`` in e, ``sigma`` ``(T,)`` in Angstrom,
        ``epsilon`` ``(T,)`` in eV, ``bond_k`` ``(NB,)`` in eV/A^2,
        ``bond_r0`` ``(NB,)``, ``angle_k`` ``(NA,)`` in eV/rad^2,
        ``angle_theta0`` ``(NA,)`` in rad, ``dihedral_v`` ``(ND, 5)`` in eV
        and ``improper_v2`` ``(NI,)`` in eV.
    type_names, bond_keys, angle_keys, dihedral_keys, improper_keys : list
        Row labels of the parameter tensors, in assembly order.
    fudge_lj, fudge_qq : float
        The library's 1,4 scaling factors.
    """

    def __init__(self, library: Union[OPLSLibrary, str, Path],
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
        # per-term equivalence classes (a .frc equivalence table may point a
        # type at different classes for bonds, angles, torsions and oops)
        self.type_cls = [lib.cls(n, "bond") for n in self.type_names]
        self.type_cls_angle = [lib.cls(n, "angle") for n in self.type_names]
        self.type_cls_torsion = [lib.cls(n, "torsion") for n in self.type_names]
        self.type_cls_oop = [lib.cls(n, "oop") for n in self.type_names]
        self.templates = dict(lib.templates)
        self.fragments = dict(lib.fragments)
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
        self._bond_index = {k: i for i, k in enumerate(self.bond_keys)}
        self._angle_index = {k: i for i, k in enumerate(self.angle_keys)}
        self._dihedral_index = {k: i for i, k in
                                enumerate(self.dihedral_keys)}
        self._improper_types = dict(lib.improper_types)
        self._improper_index = {k: i for i, k in
                                enumerate(self.improper_keys)}

        def make(name, values):
            """Register value list as the (possibly trainable) group ``name``."""
            self.params[name] = Parameter(
                torch.tensor(values, dtype=dtype),
                requires_grad=name in trainable_set)

        self.params = ParameterDict()
        make("charge", [float(a["charge"]) for a in at])
        make("sigma", [float(a["sigma"]) for a in at])
        make("epsilon", [float(a["epsilon"]) * KCAL_TO_EV for a in at])
        make("bond_k", [float(lib.bond_types[k]["k"]) * KCAL_TO_EV
                        for k in self.bond_keys])
        make("bond_r0", [float(lib.bond_types[k]["r0"])
                         for k in self.bond_keys])
        make("angle_k", [float(lib.angle_types[k]["k"]) * KCAL_TO_EV
                         for k in self.angle_keys])
        make("angle_theta0", [float(lib.angle_types[k]["theta0"]) * DEG
                              for k in self.angle_keys])
        vs = [[float(x) * KCAL_TO_EV
               for x in (list(lib.dihedral_types[k]["v"]) + [0.0] * 5)[:5]]
              for k in self.dihedral_keys]
        self.params["dihedral_v"] = Parameter(
            torch.tensor(vs, dtype=dtype).reshape(len(vs), 5),
            requires_grad="dihedral_v" in trainable_set)
        make("improper_v2", [float(lib.improper_types[k]["v2"]) * KCAL_TO_EV
                             for k in self.improper_keys])

    # ------------------------------------------------------------------
    # resolution (topology binding)
    # ------------------------------------------------------------------
    def type_index(self, name: str) -> int:
        """Row index of atom type ``name``.

        Parameters
        ----------
        name : str
            The atom-type name (e.g. ``"opls_135"``).

        Returns
        -------
        int
            Index into the per-type parameter tensors.

        Raises
        ------
        KeyError
            If the library has no such atom type.
        """
        if name not in self._type_index:
            raise KeyError(f"atom type {name!r} is not in library "
                           f"{self.name!r}")
        return self._type_index[name]

    def resolve_bond(self, a: str, b: str) -> int:
        """Bond-type row for the class pair ``(a, b)``.

        Parameters
        ----------
        a, b : str
            Atom classes of the bond's ends.

        Returns
        -------
        int
            Row index into ``bond_k`` / ``bond_r0``.

        Raises
        ------
        KeyError
            If the library has no parameters for this bond type.
        """
        key = resolve_bond_type(self._bond_types, a, b)
        if key is None:
            raise KeyError(f"no bond type {a}-{b} in library {self.name!r}")
        return self._bond_index[key]

    def resolve_angle(self, a: str, b: str, c: str) -> int:
        """Angle-type row for the class triple ``(a, b, c)``.

        Parameters
        ----------
        a, b, c : str
            Atom classes, center second.

        Returns
        -------
        int
            Row index into ``angle_k`` / ``angle_theta0``.

        Raises
        ------
        KeyError
            If the library has no parameters for this angle type.
        """
        key = resolve_angle_type(self._angle_types, a, b, c)
        if key is None:
            raise KeyError(f"no angle type {a}-{b}-{c} in library "
                           f"{self.name!r}")
        return self._angle_index[key]

    def resolve_dihedral(self, a: str, b: str, c: str, d: str) -> int:
        """Dihedral-type row for the class quadruple (wildcards allowed).

        Parameters
        ----------
        a, b, c, d : str
            Atom classes along the dihedral.

        Returns
        -------
        int
            Row index into ``dihedral_v``.

        Raises
        ------
        KeyError
            If no library entry (exact or wildcard) matches.
        """
        key = resolve_dihedral_type(self._dihedral_types, a, b, c, d)
        if key is None:
            raise KeyError(f"no dihedral type {a}-{b}-{c}-{d} in library "
                           f"{self.name!r}")
        return self._dihedral_index[key]

    def resolve_improper(self, *args) -> int:
        """Improper-type row, by exact key or by class quadruple.

        Parameters
        ----------
        *args : str
            Either one opaque key (``"Z-CM-X-Y"``, from a topology's
            ``improper_keys``), or four atom classes ``i, j, k, l`` with
            ``k`` the trigonal center, matched against the library's
            ``"I-J-K-L"`` patterns (``X`` wildcards) in SEAMM's precedence
            order (:func:`~xnns.ffnn.models.oplslib.resolve_improper_type`).

        Returns
        -------
        int
            Row index into ``improper_v2``.

        Raises
        ------
        KeyError
            If nothing in the library matches.
        """
        if len(args) == 1:
            key = args[0]
            if key not in self._improper_index:
                raise KeyError(f"no improper type {key!r} in library "
                               f"{self.name!r}")
            return self._improper_index[key]
        i, j, k, l = args
        key = resolve_improper_type(self._improper_types, i, j, k, l)
        if key is None:
            raise KeyError(f"no improper type for {i}-{j}-{k}-{l} (center "
                           f"{k}) in library {self.name!r}")
        return self._improper_index[key]

    def improper_defined(self, i: str, j: str, k: str, l: str) -> bool:
        """Whether an improper pattern exists for the class quadruple."""
        return resolve_improper_type(self._improper_types, i, j, k, l) is not None

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------
    def export_library(self) -> OPLSLibrary:
        """Write the current (possibly trained) parameters back to a library.

        Returns
        -------
        OPLSLibrary
            A new library in OPLS units (kcal/mol, Angstrom, degrees),
            round-trippable through :meth:`OPLSLibrary.save` /
            :func:`~xnns.ffnn.models.oplslib.read_opls`.
        """
        P = {k: v.detach().cpu() for k, v in self.params.items()}
        atom_types = {}
        for i, name in enumerate(self.type_names):
            atom_types[name] = {
                "cls": self.type_cls[i], "cls_nonbond": name,
                "cls_bond": self.type_cls[i],
                "cls_angle": self.type_cls_angle[i],
                "cls_torsion": self.type_cls_torsion[i],
                "cls_oop": self.type_cls_oop[i],
                "element": int(self.type_z[i]),
                "mass": float(self.type_mass[i]),
                "charge": float(P["charge"][i]),
                "sigma": float(P["sigma"][i]),
                "epsilon": float(P["epsilon"][i]) / KCAL_TO_EV,
                "comment": ""}
        return OPLSLibrary(
            atom_types=atom_types,
            bond_types={k: {"k": float(P["bond_k"][i]) / KCAL_TO_EV,
                            "r0": float(P["bond_r0"][i])}
                        for i, k in enumerate(self.bond_keys)},
            angle_types={k: {"k": float(P["angle_k"][i]) / KCAL_TO_EV,
                             "theta0": float(P["angle_theta0"][i]) / DEG}
                         for i, k in enumerate(self.angle_keys)},
            dihedral_types={k: {"v": [float(x) / KCAL_TO_EV
                                      for x in P["dihedral_v"][i]]}
                            for i, k in enumerate(self.dihedral_keys)},
            improper_types={k: {"v2": float(P["improper_v2"][i]) / KCAL_TO_EV}
                            for i, k in enumerate(self.improper_keys)},
            fudge_lj=self.fudge_lj, fudge_qq=self.fudge_qq, name=self.name,
            templates=dict(self.templates), fragments=dict(self.fragments))


@register_model("opls")
class OPLS(InteratomicPotential):
    """The OPLS fixed-topology force field (all-atom or united-atom).

    An instance binds a parameter library to one molecular topology: bonded
    terms, exclusions and scaled 1,4 pairs are resolved once at
    construction, and ``forward`` evaluates the full OPLS energy on any
    conformation (or batch of conformations) of that system. Forces and
    stress come from autograd via
    :class:`~xnns.common.models.outputs.ForceStressOutput`.

    Parameters
    ----------
    ffield : OPLSLibrary, OPLSForceField, str or Path
        The parameter library -- a parsed :class:`OPLSLibrary`, a variant
        shipped with xnns (``"oplsaa"``, ``"oplsaa-1996"``, ``"lopls"``,
        ``"CL&P"``), a ``.frc`` or native JSON path, or an existing
        :class:`OPLSForceField` to *share* parameters with other models.
    topology : MolecularTopology, str or Path
        The system's topology (or a path to a topology JSON file).
    cutoff : float, optional
        Nonbonded (Lennard-Jones / Coulomb) cutoff in Angstrom, and the
        model's neighbor-list ``cutoff``; by default 10.0. 1,4 pairs are
        independent of this cutoff.
    switch_width : float, optional
        Width in Angstrom of a quintic switching function that takes the
        nonbonded interactions smoothly to zero at the cutoff; by default
        0.0 (plain truncation).
    fudge_lj, fudge_qq : float, optional
        Override the library's 1,4 scaling factors (both 0.5 for OPLS).
    trainable : sequence of str or "all", optional
        Trainable parameter groups, forwarded to :class:`OPLSForceField`
        (ignored when ``ffield`` is already an ``OPLSForceField``).
    keep_intermediates : bool, optional
        If ``True``, stash the intermediate tensors of the last evaluation
        (bond lengths, angles, dihedral cosines, per-interaction energies)
        in ``self.intermediates``. Default ``False``.
    auto_impropers : bool, optional
        When the topology lists no impropers, place one improper at every
        three-connected atom whose classes match an ``improper_opls`` pattern
        of the library (the SEAMM convention; centers without a pattern get
        none). Default ``True``. Explicitly listed impropers are always used
        as given.

    Notes
    -----
    ``forward`` returns, besides the standard ``node_energy`` / ``energy`` /
    ``node_features`` keys, the fixed partial ``charges`` ``(N,)`` and one
    per-structure tensor per term: ``e_bond``, ``e_angle``, ``e_torsion``,
    ``e_improper``, ``e_lj``, ``e_coulomb``, ``e_lj14``, ``e_coulomb14``.
    Every structure in a batch must have this topology's atom count and
    element sequence (conformers of the bound system).
    """

    def __init__(self, ffield: Union[OPLSLibrary, OPLSForceField, str, Path],
                 topology: Union[MolecularTopology, str, Path], *,
                 cutoff: float = 10.0,
                 switch_width: float = 0.0,
                 fudge_lj: Optional[float] = None,
                 fudge_qq: Optional[float] = None,
                 trainable: Union[Sequence[str], str] = (),
                 keep_intermediates: bool = False,
                 auto_impropers: bool = True):
        super().__init__()
        self.ff = ffield if isinstance(ffield, OPLSForceField) \
            else OPLSForceField(ffield, trainable=trainable)
        self.auto_impropers = bool(auto_impropers)
        self.cutoff = float(cutoff)
        self.switch_width = float(switch_width)
        if not 0.0 <= self.switch_width < self.cutoff:
            raise ValueError("switch_width must satisfy "
                             "0 <= switch_width < cutoff")
        self.fudge_lj = self.ff.fudge_lj if fudge_lj is None \
            else float(fudge_lj)
        self.fudge_qq = self.ff.fudge_qq if fudge_qq is None \
            else float(fudge_qq)
        self.keep_intermediates = bool(keep_intermediates)
        self.intermediates: dict = {}
        self.node_feature_dim = 1
        self._bind_topology(
            topology if isinstance(topology, MolecularTopology)
            else read_topology(topology))

    @classmethod
    def from_atoms(cls, structure, ffield: Union[OPLSLibrary, str, Path] = "oplsaa",
                   *, charge: int = 0,
                   bonds: Optional[Sequence[Sequence[int]]] = None,
                   **kwargs) -> "OPLS":
        """Build an OPLS model for a structure, typing it with the library's
        SMARTS templates.

        Bonding is perceived from the coordinates (or taken from ``bonds``),
        atom types are assigned by :func:`~xnns.ffnn.common.typing.assign_atom_types`,
        and the topology is derived from the perceived bonds -- so nothing
        about the force field's own type names has to be known in advance.

        Parameters
        ----------
        structure : object
            An ``ase.Atoms``, a ``(positions, atomic_numbers)`` pair, a dict
            with ``"pos"`` / ``"atomic_numbers"``, an RDKit molecule or a
            SMILES string.
        ffield : OPLSLibrary, str or Path, optional
            The parameter library (must carry templates); default
            ``"oplsaa"``.
        charge : int, optional
            Total charge of the structure (for bond-order perception).
        bonds : sequence of (int, int), optional
            Known connectivity; bond orders are then perceived, not bonds.
        **kwargs
            Forwarded to the constructor (``cutoff``, ``trainable``, ...).

        Returns
        -------
        OPLS
            The model, bound to the derived topology.
        """
        from ..common.typing import assign_atom_types, perceive_bonds
        lib = ffield if isinstance(ffield, OPLSForceField) else _as_library(ffield)
        source = lib if isinstance(lib, OPLSForceField) else lib
        types, mol = assign_atom_types(structure, source, charge=charge,
                                       bonds=bonds, return_mol=True)
        top = MolecularTopology.from_bonds(types, perceive_bonds(mol))
        return cls(lib, top, **kwargs)

    # ------------------------------------------------------------------
    # topology binding
    # ------------------------------------------------------------------
    def _bind_topology(self, top: MolecularTopology) -> None:
        """Resolve the topology against the force field into index buffers.

        Per-atom type indices, per-interaction parameter rows, exclusions
        and 1,4 pairs are stored as integer buffers; unresolvable atom or
        interaction types are collected and reported together.

        Parameters
        ----------
        top : MolecularTopology
            The topology to bind.

        Raises
        ------
        KeyError
            Listing every atom type or bonded type the library lacks.
        """
        self.topology = top
        ff = self.ff
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
        b_type = [gather(ff.resolve_bond, cls[i], cls[j])
                  for i, j in top.bonds]
        a_type = [gather(ff.resolve_angle, cls_a[i], cls_a[j], cls_a[k])
                  for i, j, k in top.angles]
        d_type = [gather(ff.resolve_dihedral, cls_t[i], cls_t[j], cls_t[k],
                         cls_t[l]) for i, j, k, l in top.dihedrals]
        impropers = [tuple(int(x) for x in im) for im in top.impropers]
        if top.improper_keys:
            i_type = [gather(ff.resolve_improper, key)
                      for key in top.improper_keys]
        elif impropers:
            i_type = [gather(ff.resolve_improper, cls_o[i], cls_o[j], cls_o[k],
                             cls_o[l]) for i, j, k, l in impropers]
        elif self.auto_impropers:
            # one improper per three-connected center, outer atoms in index
            # order, center third (SEAMM's setup_topology); centers the
            # library has no pattern for get none
            neighbors: dict[int, list[int]] = {}
            for i, j in top.bonds:
                neighbors.setdefault(i, []).append(j)
                neighbors.setdefault(j, []).append(i)
            i_type = []
            for m, nb in sorted(neighbors.items()):
                if len(nb) != 3:
                    continue
                n1, n2, n3 = sorted(nb)
                if ff.improper_defined(cls_o[n1], cls_o[n2], cls_o[m], cls_o[n3]):
                    impropers.append((n1, n2, m, n3))
                    i_type.append(ff.resolve_improper(cls_o[n1], cls_o[n2],
                                                      cls_o[m], cls_o[n3]))
        else:
            i_type = []
        if missing:
            raise KeyError("the topology needs parameters the library does "
                           "not provide:\n  " + "\n  ".join(missing))

        def buf(name, data, shape):
            """Register an integer index buffer (possibly empty)."""
            t = torch.tensor(data, dtype=torch.long)
            self.register_buffer(name, t.reshape(*shape))

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
        # every topologically excluded or scaled pair is removed from the
        # neighbor-list interactions (1,4 pairs are added back exactly)
        excl = sorted(set(top.exclusions) | set(top.pairs14))
        buf("excl_index", [list(p) for p in excl], (-1, 2))
        self.excl_index = self.excl_index.t().contiguous()

    # ------------------------------------------------------------------
    # geometry helpers
    # ------------------------------------------------------------------
    def _pair_vectors(self, data: AtomicGraph, a: Tensor, b: Tensor
                      ) -> Tensor:
        """Minimum-image displacement vectors ``pos[b] - pos[a]``.

        For periodic structures the integer image shift is recomputed from
        the fractional displacement (rounded, detached), so bonded terms are
        correct for molecules wrapped across the boundary while gradients
        still flow to positions and cell.

        Parameters
        ----------
        data : AtomicGraph
            The batched graph.
        a, b : Tensor
            Atom indices of shape ``(M,)``.

        Returns
        -------
        Tensor
            Displacements of shape ``(M, 3)``.
        """
        vec = data.pos[b] - data.pos[a]
        if data.cell is None or vec.shape[0] == 0:
            return vec
        cell = data.cell[data.batch[a]]                       # (M, 3, 3)
        inv = torch.linalg.inv(data.cell)[data.batch[a]]
        frac = torch.einsum("mi,mij->mj", vec, inv)
        shift = -torch.round(frac).detach()
        if data.pbc is not None:
            shift = shift * data.pbc[data.batch[a]].to(shift.dtype)
        return vec + torch.einsum("mi,mij->mj", shift, cell)

    def _dihedral_cos(self, data: AtomicGraph, idx: Tensor) -> Tensor:
        """Cosine of the dihedral angle over each atom quadruple.

        Uses the plane-normal formula with ``phi = 0`` at *cis* (the OPLS
        convention). Only ``cos phi`` is returned; the Fourier terms are
        even in ``phi``, so the multiple angles come from Chebyshev
        identities and no ``arccos``/``atan2`` (whose derivatives are
        singular at planar geometries) enters the graph.

        Parameters
        ----------
        data : AtomicGraph
            The batched graph.
        idx : Tensor
            Atom indices of shape ``(4, M)``.

        Returns
        -------
        Tensor
            ``cos phi`` of shape ``(M,)``.
        """
        b1 = self._pair_vectors(data, idx[0], idx[1])
        b2 = self._pair_vectors(data, idx[1], idx[2])
        b3 = self._pair_vectors(data, idx[2], idx[3])
        n1 = torch.cross(b1, b2, dim=1)
        n2 = torch.cross(b2, b3, dim=1)
        denom = torch.sqrt((n1 * n1).sum(1) * (n2 * n2).sum(1) + TINY)
        return (n1 * n2).sum(1) / denom

    def _lj_coulomb(self, sig_i, sig_j, eps_i, eps_j, q_i, q_j, r, r2
                    ) -> tuple[Tensor, Tensor]:
        """Per-pair Lennard-Jones and Coulomb energies (eq 1).

        Parameters
        ----------
        sig_i, sig_j, eps_i, eps_j, q_i, q_j : Tensor
            Per-pair gathered atomic parameters.
        r, r2 : Tensor
            Pair distances and squared distances.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(e_lj, e_coulomb)`` per pair, in eV.
        """
        sig = _geometric_mean(sig_i, sig_j)
        eps = _geometric_mean(eps_i, eps_j)
        s6 = (sig * sig / (r2 + TINY)) ** 3
        return 4.0 * eps * (s6 * s6 - s6), KE * q_i * q_j / r

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, data: AtomicGraph) -> dict[str, Tensor]:
        """Evaluate the OPLS energy on a (batched) atomic graph.

        Parameters
        ----------
        data : AtomicGraph
            The batched graph; every structure must be a conformation of
            the bound topology (same atom count and element sequence), and
            the neighbor list must have been built with this model's
            ``cutoff``.

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
        inter: dict = {}

        if not bool((data.n_atoms == n).all()):
            raise ValueError(
                f"every structure must have the bound topology's {n} atoms "
                f"(got {data.n_atoms.tolist()})")
        if not bool((data.atomic_numbers
                     == self.top_z.repeat(B)).all()):
            raise ValueError("atomic numbers do not match the bound "
                             "topology's element sequence")

        t = self.top_type.repeat(B)
        starts = torch.arange(B, device=device) * n

        def expand(idx: Tensor) -> Tensor:
            """Tile per-structure index rows ``(k, m)`` to ``(k, B*m)``."""
            return (idx.unsqueeze(1) + starts.view(1, B, 1)).reshape(
                idx.shape[0], -1)

        def tile(v: Tensor) -> Tensor:
            """Tile per-interaction type rows ``(m,)`` to ``(B*m,)``."""
            return v.repeat(B)

        # --- bonds -----------------------------------------------------
        bi = expand(self.bond_index)
        bt = tile(self.bond_type)
        bvec = self._pair_vectors(data, bi[0], bi[1])
        br = torch.sqrt((bvec * bvec).sum(1) + TINY)
        e_bond = P["bond_k"][bt] * (br - P["bond_r0"][bt]) ** 2
        e_bond_node = scatter_sum(0.5 * e_bond, bi[0], N) \
            + scatter_sum(0.5 * e_bond, bi[1], N)

        # --- angles ----------------------------------------------------
        ai = expand(self.angle_index)
        at = tile(self.angle_type)
        u = self._pair_vectors(data, ai[1], ai[0])
        v = self._pair_vectors(data, ai[1], ai[2])
        cos_th = (u * v).sum(1) / torch.sqrt(
            (u * u).sum(1) * (v * v).sum(1) + TINY)
        bound = min(0.9999999999,
                    1.0 - 4.0 * torch.finfo(cos_th.dtype).eps)
        theta = torch.arccos(torch.clamp(cos_th, -bound, bound))
        e_angle = P["angle_k"][at] * (theta - P["angle_theta0"][at]) ** 2
        e_angle_node = scatter_sum(e_angle, ai[1], N)

        # --- proper dihedrals (Fourier series, eq 4) --------------------
        di = expand(self.dihedral_index)
        dt = tile(self.dihedral_type)
        c1 = self._dihedral_cos(data, di)
        c2 = 2.0 * c1 * c1 - 1.0
        c3 = c1 * (4.0 * c1 * c1 - 3.0)
        c4 = 2.0 * c2 * c2 - 1.0
        vd = P["dihedral_v"][dt]
        e_tors = (vd[:, 0] + 0.5 * (vd[:, 1] * (1.0 + c1)
                                    + vd[:, 2] * (1.0 - c2)
                                    + vd[:, 3] * (1.0 + c3)
                                    + vd[:, 4] * (1.0 - c4)))
        e_tors_node = scatter_sum(e_tors, di[1], N)

        # --- improper dihedrals -----------------------------------------
        ii = expand(self.improper_index)
        it = tile(self.improper_type)
        ci = self._dihedral_cos(data, ii)
        e_impr = 0.5 * P["improper_v2"][it] * (1.0 - (2.0 * ci * ci - 1.0))
        e_impr_node = scatter_sum(e_impr, ii[2], N)

        # --- nonbonded: neighbor list minus exclusions -------------------
        src, dst = data.edge_index[0], data.edge_index[1]
        vec = data.edge_vectors()
        r2 = (vec * vec).sum(1)
        r = torch.sqrt(r2 + TINY)
        q = P["charge"][t]
        sig_t, eps_t = P["sigma"][t], P["epsilon"][t]

        lj_e, coul_e = self._lj_coulomb(sig_t[src], sig_t[dst],
                                        eps_t[src], eps_t[dst],
                                        q[src], q[dst], r, r2)
        lo = torch.minimum(src, dst)
        hi = torch.maximum(src, dst)
        edge_key = lo * N + hi
        if self.excl_index.shape[1] > 0:
            ex = expand(self.excl_index)
            excl_keys = ex[0] * N + ex[1]
            w = (~torch.isin(edge_key, excl_keys)).to(r.dtype)
        else:
            w = torch.ones_like(r)
        if self.switch_width > 0.0:
            x = torch.clamp((r - (self.cutoff - self.switch_width))
                            / self.switch_width, 0.0, 1.0)
            w = w * (1.0 - x ** 3 * (10.0 - 15.0 * x + 6.0 * x * x))
        e_lj_node = scatter_sum(0.5 * w * lj_e, dst, N)
        e_coul_node = scatter_sum(0.5 * w * coul_e, dst, N)

        # --- scaled 1,4 pairs (exact, cutoff-independent) -----------------
        pi = expand(self.pair14_index)
        p_t0, p_t1 = t[pi[0]], t[pi[1]]
        pvec = self._pair_vectors(data, pi[0], pi[1])
        pr2 = (pvec * pvec).sum(1)
        pr = torch.sqrt(pr2 + TINY)
        lj14, coul14 = self._lj_coulomb(
            P["sigma"][p_t0], P["sigma"][p_t1],
            P["epsilon"][p_t0], P["epsilon"][p_t1],
            P["charge"][p_t0], P["charge"][p_t1], pr, pr2)
        lj14 = self.fudge_lj * lj14
        coul14 = self.fudge_qq * coul14
        e_lj14_node = scatter_sum(0.5 * lj14, pi[0], N) \
            + scatter_sum(0.5 * lj14, pi[1], N)
        e_coul14_node = scatter_sum(0.5 * coul14, pi[0], N) \
            + scatter_sum(0.5 * coul14, pi[1], N)

        node_energy = (e_bond_node + e_angle_node + e_tors_node
                       + e_impr_node + e_lj_node + e_coul_node
                       + e_lj14_node + e_coul14_node)
        energy = self.aggregate_energy(node_energy, data)

        if self.keep_intermediates:
            inter.update(bond_r=br, e_bond=e_bond, angle_theta=theta,
                         e_angle=e_angle, dihedral_cos=c1, e_torsion=e_tors,
                         improper_cos=ci, e_improper=e_impr, nb_r=r,
                         nb_weight=w, nb_lj=lj_e, nb_coulomb=coul_e,
                         pair14_r=pr, e_lj14=lj14, e_coulomb14=coul14,
                         type_index=t)
            self.intermediates = inter

        return {
            "node_energy": node_energy,
            "energy": energy,
            "charges": q,
            "node_features": q.unsqueeze(1),
            "e_bond": self.aggregate_energy(e_bond_node, data),
            "e_angle": self.aggregate_energy(e_angle_node, data),
            "e_torsion": self.aggregate_energy(e_tors_node, data),
            "e_improper": self.aggregate_energy(e_impr_node, data),
            "e_lj": self.aggregate_energy(e_lj_node, data),
            "e_coulomb": self.aggregate_energy(e_coul_node, data),
            "e_lj14": self.aggregate_energy(e_lj14_node, data),
            "e_coulomb14": self.aggregate_energy(e_coul14_node, data),
        }

    # ------------------------------------------------------------------
    # conveniences
    # ------------------------------------------------------------------
    def export_library(self) -> OPLSLibrary:
        """Export the current parameters as an :class:`OPLSLibrary`.

        Returns
        -------
        OPLSLibrary
            See :meth:`OPLSForceField.export_library`.
        """
        return self.ff.export_library()

    @property
    def masses(self) -> Tensor:
        """Tensor : Per-atom masses (u) of the bound topology."""
        return self.ff.type_mass[self.top_type]

    @classmethod
    def from_config(cls, cfg) -> "OPLS":
        """Construct an :class:`OPLS` model from a core model config.

        Core field: ``cfg.cutoff`` is the nonbonded cutoff. Everything else
        is read from ``cfg.extra``: ``library`` (required; a variant shipped
        with xnns such as ``oplsaa``, a ``.frc`` path or a native JSON path)
        and either ``topology`` (path to a topology JSON file) or ``types``
        + ``bonds`` (+ optional ``impropers`` / ``improper_keys``) inline.
        Optional: ``switch_width``, ``fudge_lj``, ``fudge_qq``,
        ``trainable``. Alternative spellings used by other MD packages are
        translated by :mod:`xnns.common.config.translate`.

        Parameters
        ----------
        cfg : xnns.common.config.schema.ModelConfig
            The core model config.

        Returns
        -------
        OPLS
            The model.

        Raises
        ------
        ValueError
            If the library or the topology specification is missing.
        """
        extra = dict(cfg.extra or {})

        def listy(value):
            """Coerce a possibly stringified list to a Python value."""
            if isinstance(value, str) and value.lstrip().startswith(("[", "(")):
                return ast.literal_eval(value)
            return value

        library = extra.get("library")
        if library is None:
            raise ValueError("OPLS needs a parameter library: set "
                             "model.library to 'oplsaa', 'lopls', 'CL&P', a "
                             ".frc path or a native JSON path")
        topo_path = extra.get("topology")
        if topo_path is not None:
            topology: MolecularTopology = read_topology(topo_path)
        else:
            types = listy(extra.get("types"))
            bonds = listy(extra.get("bonds"))
            if types is None or bonds is None:
                raise ValueError("OPLS needs a topology: set model.topology "
                                 "to a topology JSON path, or give "
                                 "model.types and model.bonds inline")
            topology = MolecularTopology.from_bonds(
                types, bonds,
                impropers=listy(extra.get("impropers")) or (),
                improper_keys=listy(extra.get("improper_keys")) or ())

        def floaty(key):
            """Optional float from extra."""
            value = extra.get(key)
            return None if value is None else float(value)

        return cls(library, topology, cutoff=float(cfg.cutoff),
                   switch_width=float(extra.get("switch_width") or 0.0),
                   fudge_lj=floaty("fudge_lj"), fudge_qq=floaty("fudge_qq"),
                   trainable=listy(extra.get("trainable")) or ())
