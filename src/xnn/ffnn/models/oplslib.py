"""OPLS parameter libraries: the SEAMM ``.frc`` format and native JSON.

An OPLS model (Jorgensen, Maxwell & Tirado-Rives, *J. Am. Chem. Soc.* 118,
11225, 1996) is fully specified by a parameter library: per-atom-type
nonbonded parameters (partial charge, Lennard-Jones sigma / epsilon) plus
bonded parameters keyed by the *equivalent* atom types of each term --
harmonic bonds and angles, Fourier proper dihedrals and ``V2``-only improper
dihedrals. Two sources are supported:

* the MolSSI/SEAMM ``.frc`` force-field format (:mod:`xnn.ffnn.common.frc`).
  The OPLS-AA distribution ships with xnn as ``oplsaa.frc`` (variants
  ``"oplsaa"``, ``"CL&P"``, ``"oplsaa+"``), together with ``"lopls"`` (Siu,
  Pluhackova & Boeckmann, *JCTC* 8, 1459, 2012) and ``"oplsaa-1996"`` (the
  paper's original alkane torsions) layered over it. A ``.frc`` library
  carries the SMARTS **templates** that assign its atom types to a structure
  (:mod:`xnn.ffnn.common.typing`, :meth:`~xnn.ffnn.models.opls.OPLS.from_atoms`);
* the **native JSON** format of this module (:func:`read_opls` /
  :meth:`OPLSLibrary.save`), a direct dump of :class:`OPLSLibrary` used to
  round-trip trained parameters.

Energies are converted from kcal/mol to eV only when the model assembles its
parameter tensors, mirroring :mod:`xnn.ffnn.models.ffield`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

from ..common.elements import CHEMICAL_SYMBOLS, atomic_number

# kJ/mol -> kcal/mol (thermochemical calorie).
KCAL_PER_KJ = 1.0 / 4.184
# kcal/mol -> eV as 4.184 (exact) / 96.48533212331 (CODATA 2018 kJ/mol per
# eV). OPLS parameters are defined in the thermochemical-kcal / kJ ecosystem
# (BOSS, GROMACS, OpenMM), so this choice makes xnn energies agree with
# those codes to their own precision; it differs from ReaxFF's historical
# constant by ~8e-6 relative, deliberately.
KCAL_TO_EV = 4.336410424180094e-2

WILDCARD = "X"
_TERMS = ("nonbond", "bond", "angle", "torsion", "oop")


def fourier_to_rb(v: Sequence[float]) -> list[float]:
    """Convert OPLS Fourier coefficients to Ryckaert-Bellemans form.

    The OPLS proper-dihedral energy (all phase angles zero) is::

        E = V0 + V1/2 (1+cos phi) + V2/2 (1-cos 2 phi)
              + V3/2 (1+cos 3 phi) + V4/2 (1-cos 4 phi)

    with ``phi = 0`` at *cis*; the Ryckaert-Bellemans form is
    ``E = sum_n C_n cos(psi)^n`` with ``psi = phi - 180``. ``V0`` is a
    constant offset (zero for original OPLS types; L-OPLS uses it to make the
    two forms match exactly).

    Parameters
    ----------
    v : sequence of float
        ``(V0, V1, V2, V3, V4)``.

    Returns
    -------
    list of float
        ``[C0, C1, C2, C3, C4, C5]`` in the same energy unit.
    """
    v0, v1, v2, v3, v4 = (float(x) for x in v)
    return [v0 + v2 + 0.5 * (v1 + v3), 0.5 * (-v1 + 3.0 * v3),
            -v2 + 4.0 * v4, -2.0 * v3, -4.0 * v4, 0.0]


def rb_to_fourier(c: Sequence[float]) -> list[float]:
    """Convert Ryckaert-Bellemans coefficients to OPLS Fourier form.

    Exact inverse of :func:`fourier_to_rb`; any constant that the four
    cosine terms cannot represent lands in ``V0``.

    Parameters
    ----------
    c : sequence of float
        ``(C0, C1, C2, C3, C4)`` or ``(C0, ..., C5)`` with ``C5 = 0``.

    Returns
    -------
    list of float
        ``[V0, V1, V2, V3, V4]`` in the same energy unit.

    Raises
    ------
    ValueError
        If a non-zero ``C5`` is given (not representable by the OPLS form).
    """
    cs = [float(x) for x in c] + [0.0] * (6 - len(c))
    if abs(cs[5]) > 1.0e-10:
        raise ValueError(f"C5 = {cs[5]} is not representable by the OPLS "
                         "Fourier form")
    v3 = -0.5 * cs[3]
    v4 = -0.25 * cs[4]
    v2 = -cs[2] - cs[4]
    v1 = -1.5 * cs[3] - 2.0 * cs[1]
    v0 = cs[0] - v2 - 0.5 * (v1 + v3)
    return [v0, v1, v2, v3, v4]


@dataclass
class OPLSLibrary:
    """A parsed OPLS parameter library, in OPLS units.

    All energies are kcal/mol, lengths Angstrom, angles degrees. Bonded
    parameters are keyed by the *equivalent* atom types of the term joined
    with ``-`` (``"opls_18-opls_18"``, ``"opls_85-opls_18-opls_18-opls_85"``),
    exactly as a ``.frc`` file keys them; dihedral and improper keys may use
    the wildcard ``X`` (``"X-opls_86-opls_86-X"``). Every atom type records,
    per term, which equivalent type its parameters are looked up under
    (``cls_bond`` etc.; ``cls`` is a synonym of ``cls_bond``).

    Parameters and attributes
    -------------------------
    atom_types : dict[str, dict]
        ``name -> {"cls", "cls_nonbond", "cls_bond", "cls_angle",
        "cls_torsion", "cls_oop", "element", "mass", "charge", "sigma",
        "epsilon", "connections", "comment"}`` where ``element`` is the
        atomic number and ``sigma`` / ``epsilon`` the Lennard-Jones
        parameters (Angstrom, kcal/mol).
    bond_types : dict[str, dict]
        ``"A-B" -> {"k", "r0"}`` for ``E = k (r - r0)^2`` (OPLS/AMBER
        convention, *without* the 1/2).
    angle_types : dict[str, dict]
        ``"A-B-C" -> {"k", "theta0"}`` for ``E = k (theta - theta0)^2`` with
        ``theta0`` in degrees.
    dihedral_types : dict[str, dict]
        ``"A-B-C-D" -> {"v": [V0, V1, V2, V3, V4]}`` Fourier coefficients
        (see :func:`fourier_to_rb` for the energy expression).
    improper_types : dict[str, dict]
        ``"I-J-K-L" -> {"v2"}`` for the improper energy ``V2/2 (1 - cos 2
        phi)``, with ``K`` the central atom and ``X`` wildcards, resolved by
        :func:`resolve_improper_type`. Libraries written before this format
        may carry opaque keys referenced from
        :attr:`~xnn.ffnn.models.topology.MolecularTopology.improper_keys`.
    fudge_lj, fudge_qq : float
        Scaling factors for 1,4 Lennard-Jones and Coulomb interactions
        (0.5 and 0.5 for OPLS).
    name : str
        A short label for the parameter set.
    references : list[str]
        Literature provenance of the parameters.
    templates : dict
        SMARTS atom-typing templates (``type -> {"smarts", "description",
        ...}``) when the library came from a ``.frc`` file; consumed by
        :func:`~xnn.ffnn.common.typing.assign_atom_types`.
    fragments : dict
        Whole-molecule typing fragments, likewise.
    metadata : dict
        The ``#metadata`` entries (``ff_form``, ``charges``).
    notes : list[str]
        Anything the reader had to fill in (e.g. atom types with no
        Lennard-Jones entry, set to zero).
    """

    atom_types: dict = field(default_factory=dict)
    bond_types: dict = field(default_factory=dict)
    angle_types: dict = field(default_factory=dict)
    dihedral_types: dict = field(default_factory=dict)
    improper_types: dict = field(default_factory=dict)
    fudge_lj: float = 0.5
    fudge_qq: float = 0.5
    name: str = "opls"
    references: list = field(default_factory=list)
    templates: dict = field(default_factory=dict)
    fragments: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def cls(self, type_name: str, term: str = "bond") -> str:
        """The equivalent type used for ``term`` by an atom type.

        Parameters
        ----------
        type_name : str
            An atom type of the library.
        term : str
            ``"nonbond"``, ``"bond"``, ``"angle"``, ``"torsion"`` or ``"oop"``.

        Returns
        -------
        str
            The class (equivalent type); the type itself when unspecified.
        """
        entry = self.atom_types[type_name]
        return str(entry.get(f"cls_{term}", entry.get("cls", type_name)))

    def save(self, path: Union[str, Path]) -> None:
        """Write the library as native JSON (read back by :func:`read_opls`).

        Parameters
        ----------
        path : str or Path
            Output file path (conventionally ``*.json``).
        """
        data = {"name": self.name, "references": self.references,
                "fudge_lj": self.fudge_lj, "fudge_qq": self.fudge_qq,
                "atom_types": self.atom_types, "bond_types": self.bond_types,
                "angle_types": self.angle_types,
                "dihedral_types": self.dihedral_types,
                "improper_types": self.improper_types,
                "templates": self.templates, "fragments": self.fragments,
                "metadata": self.metadata}
        Path(path).write_text(json.dumps(data, indent=1, sort_keys=True)
                              + "\n")

    def save_frc(self, path: Union[str, Path], name: Optional[str] = None,
                 version: str = "1.0") -> Path:
        """Write the library as a ``.frc`` force-field file.

        See :func:`to_forcefield`.
        """
        return to_forcefield(self, name=name, version=version).write(path)


# bonded-type resolution
def resolve_bond_type(bond_types: dict, a: str, b: str) -> Optional[str]:
    """Find the bond-type key for the class pair ``(a, b)``.

    Parameters
    ----------
    bond_types : dict
        The library's bond-type table.
    a, b : str
        Atom classes of the two ends.

    Returns
    -------
    str or None
        The matching key (either atom order), or ``None``.
    """
    for key in (f"{a}-{b}", f"{b}-{a}"):
        if key in bond_types:
            return key
    return None


def resolve_angle_type(angle_types: dict, a: str, b: str, c: str
                       ) -> Optional[str]:
    """Find the angle-type key for the class triple ``(a, b, c)``.

    Parameters
    ----------
    angle_types : dict
        The library's angle-type table.
    a, b, c : str
        Atom classes, center second.

    Returns
    -------
    str or None
        The matching key (forward or reversed), or ``None``.
    """
    for key in (f"{a}-{b}-{c}", f"{c}-{b}-{a}"):
        if key in angle_types:
            return key
    return None


def resolve_dihedral_type(dihedral_types: dict, a: str, b: str, c: str,
                          d: str) -> Optional[str]:
    """Find the best dihedral-type key for the class quadruple.

    Both atom orders are tried; among the keys that match (with ``X``
    matching any class), the one with the most non-wildcard positions wins,
    ties broken alphabetically for determinism. This is the usual exact-
    before-wildcard precedence of OPLS implementations.

    Parameters
    ----------
    dihedral_types : dict
        The library's dihedral-type table.
    a, b, c, d : str
        Atom classes along the dihedral.

    Returns
    -------
    str or None
        The best matching key, or ``None``.
    """
    quad = (a, b, c, d)
    best_key, best_score = None, -1
    for key in dihedral_types:
        pattern = tuple(key.split("-"))
        if len(pattern) != 4:
            continue
        for cand in (quad, quad[::-1]):
            if all(p == WILDCARD or p == q for p, q in zip(pattern, cand)):
                score = sum(p != WILDCARD for p in pattern)
                if score > best_score or (score == best_score
                                          and key < best_key):
                    best_key, best_score = key, score
    return best_key


def improper_key(i: str, j: str, k: str, l: str) -> str:
    """Canonical improper key: outer classes sorted, central ``k`` third."""
    a, b, c = sorted((i, j, l))
    return f"{a}-{b}-{k}-{c}"


def resolve_improper_type(improper_types: dict, i: str, j: str, k: str,
                          l: str) -> Optional[str]:
    """Find the improper-type key for the class quadruple, ``k`` central.

    Follows the precedence of SEAMM's force-field reader: the exact triple
    of outer classes first, then one outer wildcard (each position), then
    two, then all three. Outer classes are order-insensitive.

    Parameters
    ----------
    improper_types : dict
        The library's improper-type table (``"I-J-K-L"`` keys, ``X``
        wildcards).
    i, j, k, l : str
        Atom classes; ``k`` is the trigonal center.

    Returns
    -------
    str or None
        The matching key, or ``None``.
    """
    X = WILDCARD
    for pat in ((i, j, l), (X, j, l), (i, X, l), (i, j, X),
                (X, X, l), (X, j, X), (i, X, X), (X, X, X)):
        key = improper_key(pat[0], pat[1], k, pat[2])
        if key in improper_types:
            return key
    return None


# readers
def _read_json(path: Path) -> OPLSLibrary:
    """Read the native JSON library format."""
    data = json.loads(path.read_text())
    return OPLSLibrary(
        atom_types=data.get("atom_types", {}),
        bond_types=data.get("bond_types", {}),
        angle_types=data.get("angle_types", {}),
        dihedral_types=data.get("dihedral_types", {}),
        improper_types=data.get("improper_types", {}),
        fudge_lj=float(data.get("fudge_lj", 0.5)),
        fudge_qq=float(data.get("fudge_qq", 0.5)),
        name=data.get("name", path.stem),
        references=list(data.get("references", [])),
        templates=dict(data.get("templates", {})),
        fragments=dict(data.get("fragments", {})),
        metadata=dict(data.get("metadata", {})))


def from_forcefield(ff, strict: bool = True) -> OPLSLibrary:
    """Build an :class:`OPLSLibrary` from a resolved ``.frc`` force field.

    Atom types come from ``#atom_types``; their per-term classes from
    ``#equivalence`` (identity when absent); charges and Lennard-Jones
    parameters are looked up through the nonbond equivalence, with any
    ``@type`` (``rmin-eps``, ``A-B``, ...) and ``@units`` converted to
    sigma/epsilon in Angstrom and kcal/mol. Bonded tables are copied under
    their canonical keys with ``*`` written as ``X``.

    Parameters
    ----------
    ff : xnn.ffnn.common.frc.ForceField
        A force field of the OPLS functional form: ``quadratic_bond``,
        ``quadratic_angle``, ``torsion_opls``, ``improper_opls`` and a
        ``nonbond(12-6)`` section with geometric combination.
    strict : bool, optional
        If ``True`` (default) refuse a force field that also carries
        functional forms the OPLS model does not implement (e.g. the
        ``tabulated_angle`` of the CL&P ``PF6-`` anion). If ``False``, load
        it anyway, ignore those sections and record them in ``notes``; the
        library is then correct for every structure that uses none of the
        skipped types.

    Returns
    -------
    OPLSLibrary
        The library, including the file's SMARTS templates.

    Raises
    ------
    ValueError
        If the nonbond combination rule is not geometric (the OPLS model
        hard-codes geometric mixing), or a bonded section of a functional
        form the OPLS model does not implement is present.
    """
    lib = OPLSLibrary(name=ff.name)
    lib.metadata = dict(ff.metadata)
    lib.templates = dict(ff.templates)
    lib.fragments = dict(ff.fragments)
    lib.references = [r.text for r in ff.references_used()]

    unsupported = [k for k in ff.sections
                   if k in ("simple_fourier_angle", "tabulated_angle",
                            "quartic_bond", "quartic_angle", "torsion_1",
                            "torsion_3", "buckingham", "nonbond(9-6)",
                            "wilson_out_of_plane")]
    if unsupported:
        detail = {k: len(ff.sections[k]) for k in unsupported}
        if strict:
            raise ValueError(
                f"force field {ff.name!r} uses functional forms the OPLS "
                f"model does not implement: {detail} (section: rows). Pass "
                "strict=False to load it without those sections, if your "
                "structures use none of the affected types")
        for k, n in detail.items():
            lib.notes.append(f"ignored section {k} ({n} rows): functional "
                             "form not implemented by the OPLS model")
    if "nonbond(12-6)" in ff.sections and ff.combination() != "geometric":
        raise ValueError(f"force field {ff.name!r} uses the "
                         f"{ff.combination()!r} combination rule; OPLS "
                         "requires geometric")

    for name, at in ff.atom_types.items():
        entry = {f"cls_{t}": ff.equivalent(name, t) for t in _TERMS}
        entry["cls"] = entry["cls_bond"]
        try:
            entry["element"] = atomic_number(str(at.get("El", "")))
        except KeyError:
            entry["element"] = 0
        entry["mass"] = float(at.get("Mass", 0.0))
        conns = at.get("connections", 0)
        try:
            entry["connections"] = int(conns)
        except (TypeError, ValueError):      # e.g. "Dummy" for virtual sites
            entry["connections"] = 0
        entry["comment"] = str(at.get("Comment", ""))
        entry["charge"] = ff.charge(name)
        lj = ff.nonbond(name)
        if lj is None:
            lj = (0.0, 0.0)
            lib.notes.append(f"atom type {name} has no nonbond(12-6) entry; "
                             "sigma = epsilon = 0")
        entry["sigma"], entry["epsilon"] = float(lj[0]), float(lj[1])
        lib.atom_types[name] = entry

    def key_of(key) -> str:
        return "-".join(WILDCARD if x == "*" else x for x in key)

    for key, row in ff.rows("quadratic_bond").items():
        lib.bond_types[key_of(key)] = {"k": float(row.values["K2"]),
                                       "r0": float(row.values["R0"])}
    for key, row in ff.rows("quadratic_angle").items():
        lib.angle_types[key_of(key)] = {"k": float(row.values["K2"]),
                                        "theta0": float(row.values["Theta0"])}
    for key, row in ff.rows("torsion_opls").items():
        v = row.values
        lib.dihedral_types[key_of(key)] = {
            "v": [0.0, float(v["V1"]), float(v["V2"]), float(v["V3"]),
                  float(v.get("V4", 0.0))]}
    for key, row in ff.rows("improper_opls").items():
        lib.improper_types[key_of(key)] = {"v2": float(row.values["V2"])}
    return lib


def to_forcefield(lib: OPLSLibrary, name: Optional[str] = None,
                  version: str = "1.0"):
    """Write an :class:`OPLSLibrary` as an in-memory ``.frc`` file.

    The inverse of :func:`from_forcefield`: atom types, equivalences,
    charges, ``nonbond(12-6)`` (sigma-eps, geometric), the four bonded
    tables, the templates and fragments, a ``#metadata`` and a ``#define``.
    Dihedral ``V0`` constants are not representable in ``torsion_opls`` and
    are dropped (they do not affect forces or energy differences).

    Parameters
    ----------
    lib : OPLSLibrary
        The library.
    name : str, optional
        The ``#define`` name; default ``lib.name``.
    version : str, optional
        Version stamp of every row.

    Returns
    -------
    xnn.ffnn.common.frc.FrcFile
        Save it with ``.write(path)``.
    """
    from ..common.frc import FrcFile, Define, Reference, make_section
    name = name or lib.name or "opls"
    label = name
    frc = FrcFile.empty()

    def sym(z):
        return CHEMICAL_SYMBOLS[int(z)] if 0 < int(z) < len(CHEMICAL_SYMBOLS) else "X"

    def split(key):
        return tuple("*" if x == WILDCARD else x for x in key.split("-"))

    sections = [
        make_section("metadata", label, ["Parameter"], ["Value", "Description"],
                     [(("ff_form",), {"Value": lib.metadata.get("ff_form", "oplsaa"),
                                      "Description": "The functional form of the forcefield"}),
                      (("charges",), {"Value": lib.metadata.get("charges", "point"),
                                      "Description": "How charges should be handled"})],
                     version=version),
        make_section("atom_types", label, ["Type"],
                     ["Mass", "El", "connections", "Comment"],
                     [((n,), {"Mass": float(a.get("mass", 0.0)),
                              "El": sym(a.get("element", 0)),
                              "connections": int(a.get("connections", 0) or 0),
                              "Comment": str(a.get("comment", ""))})
                      for n, a in lib.atom_types.items()], version=version),
        make_section("equivalence", label, ["Type"],
                     ["NonB", "Bond", "Angle", "Torsion", "OOP"],
                     [((n,), {"NonB": lib.cls(n, "nonbond"), "Bond": lib.cls(n, "bond"),
                              "Angle": lib.cls(n, "angle"),
                              "Torsion": lib.cls(n, "torsion"),
                              "OOP": lib.cls(n, "oop")})
                      for n in lib.atom_types], version=version),
        make_section("charges", label, ["I"], ["Q"],
                     [((n,), {"Q": float(a.get("charge", 0.0))})
                      for n, a in lib.atom_types.items()
                      if lib.cls(n, "nonbond") == n], version=version),
        make_section("nonbond(12-6)", label, ["I"], ["sigma", "eps"],
                     [((n,), {"sigma": float(a.get("sigma", 0.0)),
                              "eps": float(a.get("epsilon", 0.0))})
                      for n, a in lib.atom_types.items()
                      if lib.cls(n, "nonbond") == n], version=version,
                     annotations=["E = 4 * eps(ij) * [(sigma(ij)/r(ij))**12 - "
                                  "(sigma(ij)/r(ij))**6]"],
                     modifiers={"type": [["sigma-eps"]],
                                "combination": [["geometric"]]}),
        make_section("quadratic_bond", label, ["I", "J"], ["R0", "K2"],
                     [(split(k), {"R0": float(v["r0"]), "K2": float(v["k"])})
                      for k, v in lib.bond_types.items()], version=version,
                     annotations=["E = K2 * (R - R0)^2"]),
        make_section("quadratic_angle", label, ["I", "J", "K"], ["Theta0", "K2"],
                     [(split(k), {"Theta0": float(v["theta0"]), "K2": float(v["k"])})
                      for k, v in lib.angle_types.items()], version=version,
                     annotations=["E = K2 * (Theta - Theta0)^2"]),
        make_section("torsion_opls", label, ["I", "J", "K", "L"],
                     ["V1", "V2", "V3", "V4"],
                     [(split(k), {"V1": float(v["v"][1]), "V2": float(v["v"][2]),
                                  "V3": float(v["v"][3]),
                                  "V4": float(v["v"][4]) if len(v["v"]) > 4 else 0.0})
                      for k, v in lib.dihedral_types.items()], version=version,
                     annotations=["E = 1/2*V1*[1 + cos(phi)] + 1/2*V2*[1 - cos(2*phi)]"
                                  " + 1/2*V3*[1 + cos(3*phi)] + 1/2*V4*[1 - cos(4*phi)]"]),
        make_section("improper_opls", label, ["I", "J", "K", "L"], ["V2"],
                     [(split(k), {"V2": float(v["v2"])})
                      for k, v in lib.improper_types.items()
                      if len(k.split("-")) == 4], version=version,
                     annotations=["E = 1/2*V2*[1 - cos(2*phi)]", "k is the central atom"]),
    ]
    from ..common.frc import Section
    if lib.templates:
        tsec = Section(kind="templates", label=label)
        tsec.data = {t: {str(e.get("version", version)):
                         {k: v for k, v in e.items() if k != "version"}}
                     for t, e in lib.templates.items()}
        sections.append(tsec)
    if lib.fragments:
        fsec = Section(kind="fragments", label=label)
        fsec.data = {t: {str(e.get("version", version)):
                         {k: v for k, v in e.items() if k != "version"}}
                     for t, e in lib.fragments.items()}
        sections.append(fsec)
    for sec in sections:
        frc.sections[(sec.kind, sec.label)] = sec
    define = Define(name=name)
    for sec in sections:
        define.entries.append((version, "1", sec.kind, [label]))
    frc.defines[name] = define
    text = "\n".join(lib.references) or f"OPLS parameters exported from xnn ({name})."
    frc.references[("<memory>", "1")] = Reference(number="1", text=text, author="xnn")
    return frc


def read_opls(source: Union[str, Path], strict: bool = True) -> OPLSLibrary:
    """Read an OPLS parameter library.

    Parameters
    ----------
    source : str or Path
        A ``.json`` file in the native format, or a ``.frc`` force-field spec
        understood by :func:`~xnn.ffnn.common.frc.find_forcefield`: a
        variant shipped with xnn by name (``"oplsaa"``, ``"oplsaa-1996"``,
        ``"lopls"``, ``"CL&P"``, ``"oplsaa+"``), a ``.frc`` path, or
        ``"<path>.frc:<variant>"``.
    strict : bool, optional
        See :func:`from_forcefield`.

    Returns
    -------
    OPLSLibrary
        The parsed library.
    """
    s = str(source)
    if s.lower().endswith(".json"):
        return _read_json(Path(s))
    from ..common.frc import read_forcefield
    return from_forcefield(read_forcefield(s), strict=strict)


def builtin_library(name: str = "oplsaa", strict: bool = True) -> OPLSLibrary:
    """Return one of the parameter sets shipped with xnn.

    ``"oplsaa"`` is the OPLS-AA distribution (SEAMM's ``oplsaa.frc``), with
    the alkane torsions of its late-1999 revision; ``"oplsaa-1996"`` restores
    the original 1996 alkane torsions (reproduces Table 1 of the paper);
    ``"lopls"`` layers the L-OPLS long-hydrocarbon refit of Siu et al. (2012)
    on top (new ``lopls_*`` atom types and refit alkane / alkene torsions);
    ``"CL&P"`` is the Canongia Lopes & Padua ionic-liquid extension and
    ``"oplsaa+"`` the union of everything. See
    :func:`~xnn.ffnn.common.frc.list_forcefields` for the full list.

    Parameters
    ----------
    name : str, optional
        The variant name, by default ``"oplsaa"``.
    strict : bool, optional
        See :func:`from_forcefield`; ``"CL&P"`` and ``"oplsaa+"`` need
        ``strict=False`` because of the tabulated ``PF6-`` angle.

    Returns
    -------
    OPLSLibrary
        A fresh library instance (safe to modify).
    """
    return read_opls(name, strict=strict)
