"""DREIDING parameter library and generation rules.

DREIDING (Mayo, Olafson & Goddard, *J. Phys. Chem.* 94, 8897, 1990) is a
*rule-generated* force field: instead of tabulating parameters per bond,
angle or torsion type, it derives every valence term from a handful of
per-atom generators and global constants by hybridization rules.

* Bonds (eqs 6-9): ``R0_IJ = R0_I + R0_J - delta`` with ``delta = 0.01`` A
  from the per-type *bond radii* of Table I, and one force constant
  ``K(n) = n * 700 (kcal/mol)/A^2`` (well depth ``D(n) = n * 70 kcal/mol``
  for the Morse variant) scaled by the bond order ``n``.
* Angles (eqs 10-12): the equilibrium angle ``theta0`` depends only on the
  *central* atom (Table I) and every angle shares ``K = 100
  (kcal/mol)/rad^2``.
* Torsions (eqs 13-23): the barrier ``V``, periodicity ``n`` and equilibrium
  angle ``phi0`` follow from the hybridizations of the two central atoms and
  the bond order between them; the barrier is a per-*bond* total, divided
  evenly over the dihedrals sharing that central bond
  (:data:`TORSION_RULES`, :func:`torsion_rule`).
* Inversions (eqs 28): one spectroscopic umbrella term per listed planar (or
  stereo) center, all three axis choices averaged with weight 1/3.
* van der Waals: per-type ``(R0, D0)`` for the Lennard-Jones form (eq 31')
  or ``(R0, D0, zeta)`` for the exponential-6 form (eq 32'), with geometric
  combination of well depths and arithmetic (LJ default, eq 36c) or
  geometric combination of radii.
* Hydrogen bonds (eq 38): an explicit 12-10 term on donor-H...acceptor
  triplets involving the dedicated ``H__HB`` hydrogen type.

The per-atom generators come from the SEAMM ``dreiding.frc`` distribution
shipped with xnn (variants ``"dreiding"`` for Lennard-Jones and
``"dreiding/X6"`` for exponential-6 nonbonds), which also carries the SMARTS
templates that assign DREIDING atom types to a structure. The global rule
constants are the published ones and live here. A library can also be
round-tripped through the native JSON format of this module
(:meth:`DreidingLibrary.save` / :func:`read_dreiding`) to persist retrained
parameters.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from ..common.elements import CHEMICAL_SYMBOLS

# Bond-additivity correction delta of eq 6, in Angstrom.
DELTA = 0.01
# Single-bond stretch force constant K(1) of eq 7, kcal/mol/A^2.
BOND_K1 = 700.0
# Single-bond well depth D(1) of eq 8 (Morse form), kcal/mol.
BOND_D1 = 70.0
# Universal angle-bend force constant of eq 12, kcal/mol/rad^2.
ANGLE_K = 100.0
# Hydrogen-bond well depth / distance (Table V, the no-charges convention;
# use 7.0 with Gasteiger charges and 4.0 with experiment-quality charges).
HBOND_D0 = 9.0
HBOND_R0 = 2.75

# Torsion rules of eqs 14-23: rule id -> (total barrier V [kcal/mol],
# periodicity n, equilibrium angle phi0 [deg]). Rule (g) (a central sp1,
# monovalent or metal atom) has V = 0 and generates no term.
TORSION_RULES: dict[str, tuple[float, int, float]] = {
    "a": (2.0, 3, 180.0),    # sp3 - sp3 single bond (eq 14)
    "b": (1.0, 6, 0.0),      # sp2/resonant - sp3 single bond (eq 15)
    "c": (45.0, 2, 180.0),   # sp2 = sp2 double bond (eq 16)
    "d": (25.0, 2, 180.0),   # resonance bond, order 1.5 (eq 17)
    "e": (5.0, 2, 180.0),    # single bond between sp2/resonant atoms (eq 18)
    "f": (10.0, 2, 180.0),   # exocyclic aromatic-aromatic single bond (eq 19)
    "h": (2.0, 2, 90.0),     # sp3 - sp3, both of the oxygen column (eq 21)
    "i": (2.0, 2, 180.0),    # oxygen-column sp3 - other-column sp2 (eq 22)
    "j": (2.0, 3, 180.0),    # propene-like sp2 - sp3 dihedral (eq 23)
}
# Elements of the oxygen column (group 16), for torsion rules (h) and (i).
OXYGEN_COLUMN = frozenset({"O", "S", "Se", "Te", "Po"})
# Elements that act as hydrogen-bond acceptors (and donors), section II.I.
HB_ACCEPTOR_ELEMENTS = frozenset({"N", "O", "F"})
# Atom types that mark an explicit hydrogen-bonding hydrogen.
HB_DONOR_TYPES = frozenset({"H__HB"})


def hybridization(type_name: str) -> str:
    """Hybridization class of a DREIDING atom type, from its mnemonic.

    The third character of the five-character DREIDING label encodes the
    geometry: ``1`` linear (sp1), ``2`` trigonal (sp2), ``3`` tetrahedral
    (sp3), ``R`` resonant/aromatic. Types without one (``H_``, halogens,
    alkali/alkaline-earth and transition metals) are monovalent or ionic
    centers, class ``"0"``, which never carry a torsion (rule g).

    Parameters
    ----------
    type_name : str
        The DREIDING atom type (e.g. ``"C_R"``, ``"O_3"``, ``"Na"``).

    Returns
    -------
    str
        One of ``"0"``, ``"1"``, ``"2"``, ``"3"``, ``"R"``.
    """
    if len(type_name) > 2 and type_name[2] in "123R":
        return type_name[2]
    return "0"


def element_of(type_name: str) -> str:
    """Chemical element implied by a DREIDING type mnemonic.

    The first one or two characters name the element, with ``_`` padding
    single-letter symbols (``N_``, ``H__HB``).

    Parameters
    ----------
    type_name : str
        The DREIDING atom type.

    Returns
    -------
    str
        The element symbol (empty if the mnemonic matches no element).
    """
    two = type_name[:2]
    if two in CHEMICAL_SYMBOLS:
        return two
    one = type_name[:1]
    return one if one in CHEMICAL_SYMBOLS else ""


@dataclass
class DreidingLibrary:
    """The per-atom-type generators of a DREIDING parameter set.

    Values are stored in the paper's units (kcal/mol, Angstrom, degree);
    the model converts to eV / radians when it assembles tensors.

    Attributes
    ----------
    atom_types : dict
        ``type -> {"element", "mass", "connections", "comment"}``.
    radius : dict
        Bond radius ``R0`` per type, Angstrom (Table I).
    theta0 : dict
        Equilibrium angle per type as an angle *center*, degrees (Table I).
    vdw_r0, vdw_d0 : dict
        van der Waals minimum distance (Angstrom) and well depth (kcal/mol)
        per type (Table II).
    x6_zeta : dict
        Dimensionless exponential-6 scaling parameter ``zeta`` per type
        (Table II); empty for a Lennard-Jones library.
    oop : dict
        ``central type -> (K [kcal/mol/rad^2], Psi0 [deg])`` inversion rows.
    form : str
        ``"lj"`` or ``"x6"``: which nonbond form the library was defined
        with.
    combination : str
        Combination rule for the LJ ``R0`` (``"arithmetic"``, the DREIDING
        default of eq 36c, or ``"geometric"``); well depths always combine
        geometrically.
    delta, bond_k1, bond_d1, angle_k : float
        The global valence generators (eqs 6-12).
    torsion_v : dict
        Total torsion barrier per rule id, kcal/mol (eqs 14-23).
    hbond_d0, hbond_r0 : float
        Hydrogen-bond well depth (kcal/mol) and distance (Angstrom) of
        eq 38.
    hb_donor_types : set
        Atom types marking hydrogen-bonding hydrogens.
    templates, fragments : dict
        SMARTS typing templates (see :mod:`xnn.ffnn.common.typing`).
    name : str
        Library name.
    """

    atom_types: dict = field(default_factory=dict)
    radius: dict = field(default_factory=dict)
    theta0: dict = field(default_factory=dict)
    vdw_r0: dict = field(default_factory=dict)
    vdw_d0: dict = field(default_factory=dict)
    x6_zeta: dict = field(default_factory=dict)
    oop: dict = field(default_factory=dict)
    form: str = "lj"
    combination: str = "arithmetic"
    delta: float = DELTA
    bond_k1: float = BOND_K1
    bond_d1: float = BOND_D1
    angle_k: float = ANGLE_K
    torsion_v: dict = field(
        default_factory=lambda: {r: v for r, (v, _, _) in
                                 TORSION_RULES.items()})
    hbond_d0: float = HBOND_D0
    hbond_r0: float = HBOND_R0
    hb_donor_types: set = field(default_factory=lambda: set(HB_DONOR_TYPES))
    templates: dict = field(default_factory=dict)
    fragments: dict = field(default_factory=dict)
    name: str = "dreiding"

    # rule helpers
    def hybrid(self, type_name: str) -> str:
        """Hybridization class of a type (see :func:`hybridization`)."""
        return hybridization(type_name)

    def element(self, type_name: str) -> str:
        """Element of a type (the ``atom_types`` entry, else the mnemonic)."""
        entry = self.atom_types.get(type_name)
        if entry and entry.get("element"):
            return str(entry["element"])
        return element_of(type_name)

    def oxygen_column(self, type_name: str) -> bool:
        """Whether the type's element sits in the oxygen column."""
        return self.element(type_name) in OXYGEN_COLUMN

    def is_acceptor(self, type_name: str) -> bool:
        """Whether the type can accept a hydrogen bond (N, O, F)."""
        return self.element(type_name) in HB_ACCEPTOR_ELEMENTS

    def save(self, path: Union[str, Path]) -> Path:
        """Write the library as JSON (read back by :func:`read_dreiding`).

        Parameters
        ----------
        path : str or Path
            Output file path.

        Returns
        -------
        Path
            The written path.
        """
        data = {
            "format": "xnn-dreiding-1",
            "name": self.name, "form": self.form,
            "combination": self.combination,
            "delta": self.delta, "bond_k1": self.bond_k1,
            "bond_d1": self.bond_d1, "angle_k": self.angle_k,
            "torsion_v": self.torsion_v,
            "hbond_d0": self.hbond_d0, "hbond_r0": self.hbond_r0,
            "hb_donor_types": sorted(self.hb_donor_types),
            "atom_types": self.atom_types,
            "radius": self.radius, "theta0": self.theta0,
            "vdw_r0": self.vdw_r0, "vdw_d0": self.vdw_d0,
            "x6_zeta": self.x6_zeta,
            "oop": {k: list(v) for k, v in self.oop.items()},
            "templates": self.templates, "fragments": self.fragments,
        }
        p = Path(path)
        p.write_text(json.dumps(data, indent=2) + "\n")
        return p


def torsion_rule(lib: DreidingLibrary, t_i: str, t_j: str, t_k: str,
                 t_l: str, order: float = 1.0) -> Optional[str]:
    """DREIDING torsion rule for one dihedral ``I-J-K-L`` (eqs 14-23).

    ``J`` and ``K`` are the central atoms and ``order`` the bond order of
    the ``J-K`` bond. The rules depend only on the central atoms except for
    the propene exception (j), which distinguishes dihedrals by the outer
    atom on the sp2 side: an sp2/resonant outer atom keeps the 6-fold rule
    (b), any other outer atom gets the 3-fold rule (j) -- this reproduces
    the paper's own barriers for propene and the acetate anion.

    Parameters
    ----------
    lib : DreidingLibrary
        The library (for hybridizations and elements).
    t_i, t_j, t_k, t_l : str
        Atom types along the dihedral.
    order : float, optional
        Bond order of the central bond (1, 1.5, 2, 3); by default 1.

    Returns
    -------
    str or None
        The rule id (a key of :data:`TORSION_RULES`), or ``None`` when the
        dihedral carries no torsion (rule g: a central sp1, monovalent or
        metal atom).
    """
    hj, hk = lib.hybrid(t_j), lib.hybrid(t_k)
    if hj in ("0", "1") or hk in ("0", "1"):
        return None                                            # (g)
    sp2 = ("2", "R")
    if hj in sp2 and hk in sp2:
        if order >= 1.75:
            return "c"                                         # double bond
        if abs(order - 1.5) < 0.25:
            return "d"                                         # resonance
        if hj == "R" and hk == "R":
            return "f"                                         # biphenyl-like
        return "e"                                             # butadiene-like
    if hj == "3" and hk == "3":
        if lib.oxygen_column(t_j) and lib.oxygen_column(t_k):
            return "h"                                         # HOOH-like
        return "a"                                             # ethane-like
    # one sp2/resonant center, one sp3 center
    if hj in sp2:
        t_sp2, t_sp3, t_outer = t_j, t_k, t_i
    else:
        t_sp2, t_sp3, t_outer = t_k, t_j, t_l
    if lib.oxygen_column(t_sp3) and not lib.oxygen_column(t_sp2):
        return "i"                                             # ester-like
    if lib.hybrid(t_outer) in sp2:
        return "b"                                             # acetate-like
    return "j"                                                 # propene-like


# reading
def _from_json(path: Path) -> DreidingLibrary:
    """Load a native JSON library written by :meth:`DreidingLibrary.save`."""
    data = json.loads(path.read_text())
    return DreidingLibrary(
        atom_types=data.get("atom_types", {}),
        radius=data.get("radius", {}), theta0=data.get("theta0", {}),
        vdw_r0=data.get("vdw_r0", {}), vdw_d0=data.get("vdw_d0", {}),
        x6_zeta=data.get("x6_zeta", {}),
        oop={k: tuple(v) for k, v in data.get("oop", {}).items()},
        form=data.get("form", "lj"),
        combination=data.get("combination", "arithmetic"),
        delta=float(data.get("delta", DELTA)),
        bond_k1=float(data.get("bond_k1", BOND_K1)),
        bond_d1=float(data.get("bond_d1", BOND_D1)),
        angle_k=float(data.get("angle_k", ANGLE_K)),
        torsion_v={k: float(v) for k, v in data.get(
            "torsion_v", {r: v for r, (v, _, _) in
                          TORSION_RULES.items()}).items()},
        hbond_d0=float(data.get("hbond_d0", HBOND_D0)),
        hbond_r0=float(data.get("hbond_r0", HBOND_R0)),
        hb_donor_types=set(data.get("hb_donor_types", HB_DONOR_TYPES)),
        templates=data.get("templates", {}),
        fragments=data.get("fragments", {}),
        name=data.get("name", "dreiding"))


def read_dreiding(spec: Union[str, Path] = "dreiding") -> DreidingLibrary:
    """Read a DREIDING library from a ``.frc`` variant or a native JSON.

    Parameters
    ----------
    spec : str or Path, optional
        ``"dreiding"`` (the shipped Lennard-Jones variant, the default),
        ``"dreiding/X6"`` (the shipped exponential-6 variant), a ``.frc``
        path (``"file.frc:variant"`` to pick one of several), or a native
        JSON path written by :meth:`DreidingLibrary.save`.

    Returns
    -------
    DreidingLibrary
        The library.

    Raises
    ------
    ValueError
        If the resolved force field is not of the DREIDING form.
    """
    p = Path(str(spec))
    if p.suffix.lower() == ".json" and p.is_file():
        return _from_json(p)

    from ..common.frc import read_forcefield
    ff = read_forcefield(str(spec))
    if ff.ff_form != "dreiding":
        raise ValueError(f"force field {ff.name!r} has ff_form "
                         f"{ff.ff_form!r}, not 'dreiding'")

    lib = DreidingLibrary(name=ff.name, templates=dict(ff.templates),
                          fragments=dict(ff.fragments))

    def ensure_type(t: str) -> None:
        """Register a type that appears only in a parameter table."""
        if t not in lib.atom_types:
            lib.atom_types[t] = {"element": element_of(t), "mass": 0.0,
                                 "connections": 0, "comment": ""}

    for key, row in ff.rows("atom_types").items():
        v = row.values
        lib.atom_types[key[0]] = {
            "element": str(v.get("El", "")) or element_of(key[0]),
            "mass": float(v.get("Mass", 0.0)),
            "connections": int(v.get("connections", 0)),
            "comment": str(v.get("Comment", ""))}
    for key, row in ff.rows("dreiding_atomic_parameters").items():
        ensure_type(key[0])
        lib.radius[key[0]] = float(row.values["Radius"])
        lib.theta0[key[0]] = float(row.values["Theta0"])
    if "buckingham" in ff.sections:
        lib.form = "x6"
        for key, row in ff.rows("buckingham").items():
            ensure_type(key[0])
            lib.vdw_r0[key[0]] = float(row.values["rho"])
            lib.vdw_d0[key[0]] = float(row.values["eps"])
            lib.x6_zeta[key[0]] = float(row.values["S"])
    else:
        lib.form = "lj"
        lib.combination = ff.combination("nonbond(12-6)")
        two_sixth = 2.0 ** (1.0 / 6.0)
        for key, row in ff.rows("nonbond(12-6)").items():
            ensure_type(key[0])
            # rows were normalised to sigma-eps on read; DREIDING works
            # with the vdW minimum distance R0 = sigma * 2^(1/6)
            lib.vdw_r0[key[0]] = float(row.values["sigma"]) * two_sixth
            lib.vdw_d0[key[0]] = float(row.values["eps"])
    for key, row in ff.rows("dreiding_out_of_plane").items():
        center = key[1]                       # like_oop: central atom second
        ensure_type(center)
        lib.oop[center] = (float(row.values["K2"]),
                           float(row.values["Psi0"]))
    return lib
