"""CHARMM parameter libraries: the SEAMM ``.frc`` format and native JSON.

A CHARMM model (MacKerell *et al.*, *J. Phys. Chem. B* 102, 3586, 1998) is
specified by the parameter tables of its ``.prm`` file and the residue
definitions of its ``.rtf`` file. Both live in one ``.frc`` file
(:mod:`xnn.ffnn.common.frc`): ``quadratic_bond``, ``quadratic_angle``,
``urey_bradley``, ``torsion_fourier``, ``improper_harmonic``,
``nonbond(12-6)`` (``Rmin/2`` and ``epsilon`` per type),
``nonbond_1-4(12-6)`` (the types with their own 1,4 values) and
``nonbond_pair(12-6)`` (the NBFIX pair overrides) sections, and the
residues as ``#fragments`` whose SMARTS carry atom maps in topology order
together with ``atom names``, ``atom types``, ``charges`` and the
residue's ``impropers`` (atom names; a ``-`` or ``+`` prefix names the
atom of the previous or next residue bonded to the improper's central
atom). ``charmm22`` ships with xnn as ``charmm22.frc``.

Energies are converted from kcal/mol to eV only when the model assembles
its parameter tensors, as for OPLS and AMBER.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

from xnn.common.data.elements import CHEMICAL_SYMBOLS, atomic_number

from .oplslib import (KCAL_TO_EV, WILDCARD, resolve_bond_type,
                      resolve_angle_type, resolve_dihedral_type)

# The Coulomb constant of the CHARMM program (CCELEC) in kcal Angstrom / (mol e^2).
COULOMB_CONSTANT = 332.0716
# 1,4 electrostatics are not scaled (e14fac = 1).
SCALE_14_COULOMB = 1.0
TWO_SIXTH = 2.0 ** (1.0 / 6.0)
# at most this many Fourier terms per torsion row
MAX_TERMS = 6


@dataclass
class CharmmLibrary:
    """A parsed CHARMM parameter library, in CHARMM units.

    Energies are kcal/mol, lengths Angstrom, angles degrees. Bonded
    parameters are keyed by the atom types of the term joined with ``-``
    (``"CT3-CT3"``, ``"X-CT3-CT3-X"``); ``X`` is the wildcard.

    Parameters and attributes
    -------------------------
    atom_types : dict[str, dict]
        ``name -> {"element", "mass", "rstar", "epsilon", "rstar14",
        "epsilon14", "connections", "comment"}``; ``rstar`` is ``Rmin/2``
        in Angstrom and ``epsilon`` the well depth in kcal/mol;
        ``rstar14`` / ``epsilon14`` are the type's own 1,4 values or
        ``None`` when it has none (the regular values then apply).
    bond_types : dict[str, dict]
        ``"A-B" -> {"k", "r0"}`` for ``E = k (r - r0)^2``.
    angle_types : dict[str, dict]
        ``"A-B-C" -> {"k", "theta0"}`` for ``E = k (theta - theta0)^2``.
    urey_bradley_types : dict[str, dict]
        ``"A-B-C" -> {"k", "s0"}`` for ``E = k (S - s0)^2`` on the A...C
        distance (only the angles that carry a Urey-Bradley term).
    dihedral_types : dict[str, dict]
        ``"A-B-C-D" -> {"terms": [[K, n, delta], ...]}`` for
        ``E = sum K [1 + cos(n phi - delta)]`` per dihedral.
    improper_types : dict[str, dict]
        ``"I-J-K-L" -> {"k", "psi0"}`` for ``E = k (psi - psi0)^2`` with
        ``psi`` the dihedral angle over the four atoms in the order given;
        keys are kept as written (the central atom first in CHARMM's
        topology files, last in many rows of its parameter files) and
        match a quadruple in either direction.
    nbfix_types : dict[str, dict]
        ``"A-B" -> {"rmin", "eps", "rmin14", "eps14"}`` pair overrides of
        the Lennard-Jones parameters (``rmin`` is the pair's own minimum
        distance; the 1,4 entries may be ``None``, meaning the same).
    coulomb_constant : float
        The Coulomb constant in kcal Angstrom / (mol e^2).
    scale_14_coulomb : float
        Scaling of the 1,4 electrostatics (1 for CHARMM).
    name : str
        A short label for the parameter set.
    references : list[str]
        Literature provenance of the parameters.
    templates : dict
        Per-type SMARTS templates (``type -> {"smarts": [...], ...}``).
    fragments : dict
        Residue templates (``name -> {"SMARTS", "atom names", "atom types",
        "charges", "impropers", ...}``), consumed by
        :func:`~xnn.ffnn.common.typing.assign_atom_types` and
        :func:`place_impropers`.
    metadata : dict
        The ``#metadata`` entries.
    notes : list[str]
        Anything the reader had to fill in.
    """

    atom_types: dict = field(default_factory=dict)
    bond_types: dict = field(default_factory=dict)
    angle_types: dict = field(default_factory=dict)
    urey_bradley_types: dict = field(default_factory=dict)
    dihedral_types: dict = field(default_factory=dict)
    improper_types: dict = field(default_factory=dict)
    nbfix_types: dict = field(default_factory=dict)
    coulomb_constant: float = COULOMB_CONSTANT
    scale_14_coulomb: float = SCALE_14_COULOMB
    name: str = "charmm"
    references: list = field(default_factory=list)
    templates: dict = field(default_factory=dict)
    fragments: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def bond(self, key: str) -> Optional[dict]:
        """The bond entry of ``"A-B"`` in either orientation (``None`` if absent)."""
        a, b = key.split("-")
        k = resolve_bond_type(self.bond_types, a, b)
        return None if k is None else self.bond_types[k]

    def angle(self, key: str) -> Optional[dict]:
        """The angle entry of ``"A-B-C"`` in either orientation (``None`` if absent)."""
        a, b, c = key.split("-")
        k = resolve_angle_type(self.angle_types, a, b, c)
        return None if k is None else self.angle_types[k]

    def urey_bradley(self, key: str) -> Optional[dict]:
        """The Urey-Bradley entry of ``"A-B-C"`` in either orientation (``None`` if absent)."""
        a, b, c = key.split("-")
        k = resolve_angle_type(self.urey_bradley_types, a, b, c)
        return None if k is None else self.urey_bradley_types[k]

    def torsion(self, key: str) -> Optional[dict]:
        """The torsion entry whose key is ``"A-B-C-D"`` or its reverse, exactly
        (no wildcard matching; ``None`` if absent)."""
        parts = key.split("-")
        for cand in ("-".join(parts), "-".join(parts[::-1])):
            if cand in self.dihedral_types:
                return self.dihedral_types[cand]
        return None

    def improper(self, key: str) -> Optional[dict]:
        """The improper entry of ``"I-J-K-L"`` as written or reversed (``None`` if absent)."""
        parts = key.split("-")
        for cand in ("-".join(parts), "-".join(parts[::-1])):
            if cand in self.improper_types:
                return self.improper_types[cand]
        return None

    def nbfix(self, key: str) -> Optional[dict]:
        """The pair override of ``"A-B"`` in either orientation (``None`` if absent)."""
        a, b = key.split("-")
        k = resolve_bond_type(self.nbfix_types, a, b)
        return None if k is None else self.nbfix_types[k]

    def residue_charges(self, name: str) -> dict[str, float]:
        """``atom name -> charge`` of a residue template.

        Parameters
        ----------
        name : str
            A fragment (residue) name, e.g. ``"ALA"``.

        Returns
        -------
        dict
            The template's charges by atom name.

        Raises
        ------
        KeyError
            If the library has no such residue.
        """
        frag = self.fragments[name]
        return {str(a): float(q) for a, q in
                zip(frag.get("atom names", []), frag.get("charges", []))}

    def save(self, path: Union[str, Path]) -> None:
        """Write the library as native JSON (read back by :func:`read_charmm`).

        Parameters
        ----------
        path : str or Path
            Output file path (conventionally ``*.json``).
        """
        data = {"name": self.name, "references": self.references,
                "coulomb_constant": self.coulomb_constant,
                "scale_14_coulomb": self.scale_14_coulomb,
                "atom_types": self.atom_types, "bond_types": self.bond_types,
                "angle_types": self.angle_types,
                "urey_bradley_types": self.urey_bradley_types,
                "dihedral_types": self.dihedral_types,
                "improper_types": self.improper_types,
                "nbfix_types": self.nbfix_types,
                "templates": self.templates, "fragments": self.fragments,
                "metadata": self.metadata}
        Path(path).write_text(json.dumps(data, indent=1, sort_keys=True)
                              + "\n")

    def save_frc(self, path: Union[str, Path], name: Optional[str] = None,
                 version: str = "1.0") -> Path:
        """Write the library as a ``.frc`` force-field file (see
        :func:`to_forcefield`)."""
        return to_forcefield(self, name=name, version=version).write(path)


# improper resolution and placement
def resolve_improper_type(improper_types: dict, i: str, j: str, k: str,
                          l: str) -> Optional[str]:
    """Find the improper key for four atom types in evaluation order.

    A row matches the quadruple as written or reversed; rows without
    wildcards are preferred, then the wildcard rows in table order, each
    tried in both directions (the CHARMM rule).

    Parameters
    ----------
    improper_types : dict
        The library's improper table.
    i, j, k, l : str
        Atom types with ``i`` the central atom.

    Returns
    -------
    str or None
        The key, or ``None``.
    """
    quad = (i, j, k, l)
    for cand in (quad, quad[::-1]):
        key = "-".join(cand)
        if key in improper_types:
            return key
    for key in improper_types:
        pat = tuple(key.split("-"))
        if len(pat) != 4 or WILDCARD not in pat:
            continue
        for cand in (quad, quad[::-1]):
            if all(p == WILDCARD or p == q for p, q in zip(pat, cand)):
                return key
    return None


def place_impropers(topology, fragments: dict, elements: Sequence[str]
                    ) -> list[tuple[int, int, int, int]]:
    """The improper quadruples of a topology from its residue templates.

    Every residue template lists its impropers by atom name with the
    central atom first; a name prefixed by ``-`` or ``+`` denotes the atom
    of a neighbouring residue bonded to the central atom (the carbonyl
    carbon ``-C`` before an amide nitrogen, the nitrogen ``+N`` after a
    carbonyl carbon). An improper whose neighbouring-residue atom does not
    exist (a chain end) is dropped, as the CHARMM builder does.

    Parameters
    ----------
    topology : MolecularTopology
        The topology; ``residues`` and ``residue_names`` must be set (by
        template typing).
    fragments : dict
        The library's residue templates.
    elements : sequence of str
        Per-atom element symbols, used to pick the neighbouring-residue atom
        by the element its name starts with.

    Returns
    -------
    list of tuple
        The ``(i, j, k, l)`` quadruples, central atom first, in residue
        order.

    Raises
    ------
    ValueError
        If a neighbouring-residue reference is ambiguous.
    """
    neighbors: dict[int, list[int]] = {}
    for i, j in topology.bonds:
        neighbors.setdefault(i, []).append(j)
        neighbors.setdefault(j, []).append(i)
    by_residue: dict[int, dict[str, int]] = {}
    members: dict[int, set] = {}
    names = list(topology.residue_names)
    for atom, r in enumerate(topology.residues):
        if r is None or r[0] >= len(names) or names[r[0]] not in fragments:
            continue
        frag = fragments[names[r[0]]]
        atom_names = frag.get("atom names", [])
        if r[1] < len(atom_names):
            by_residue.setdefault(r[0], {})[str(atom_names[r[1]])] = atom
        members.setdefault(r[0], set()).add(atom)

    def element_of(name: str) -> str:
        """The element a CHARMM protein atom name starts with (``HN`` -> ``H``)."""
        return name[0].upper()

    impropers = []
    for r in sorted(by_residue):
        frag = fragments[names[r]]
        for quad in frag.get("impropers", []):
            center_name = str(quad[0])
            if center_name not in by_residue[r]:
                continue
            center = by_residue[r][center_name]
            atoms, ok = [], True
            for name in quad:
                name = str(name)
                if name[0] in "+-":
                    want = element_of(name[1:])
                    cand = [a for a in neighbors.get(center, [])
                            if a not in members[r] and elements[a] == want]
                    if not cand:
                        ok = False
                        break
                    if len(cand) > 1:
                        raise ValueError(
                            f"improper {quad} of residue {names[r]}: the "
                            f"reference {name!r} matches {len(cand)} atoms")
                    atoms.append(cand[0])
                elif name in by_residue[r]:
                    atoms.append(by_residue[r][name])
                else:
                    ok = False
                    break
            if ok:
                impropers.append(tuple(atoms))
    return impropers


# readers
def _read_json(path: Path) -> CharmmLibrary:
    """Read the native JSON library format."""
    data = json.loads(path.read_text())
    return CharmmLibrary(
        atom_types=data.get("atom_types", {}),
        bond_types=data.get("bond_types", {}),
        angle_types=data.get("angle_types", {}),
        urey_bradley_types=data.get("urey_bradley_types", {}),
        dihedral_types=data.get("dihedral_types", {}),
        improper_types=data.get("improper_types", {}),
        nbfix_types=data.get("nbfix_types", {}),
        coulomb_constant=float(data.get("coulomb_constant", COULOMB_CONSTANT)),
        scale_14_coulomb=float(data.get("scale_14_coulomb", SCALE_14_COULOMB)),
        name=data.get("name", path.stem),
        references=list(data.get("references", [])),
        templates=dict(data.get("templates", {})),
        fragments=dict(data.get("fragments", {})),
        metadata=dict(data.get("metadata", {})))


def _terms_of(values: dict, max_terms: int) -> list[list[float]]:
    """``[[K, n, delta], ...]`` from the ``K1 n1 Phi0_1 ...`` columns of a row."""
    terms = []
    for m in range(1, max_terms + 1):
        k = values.get(f"K{m}")
        if k is None or k == "":
            break
        n = values.get(f"n{m}", 0)
        ph = values.get(f"Phi0_{m}", 0.0)
        terms.append([float(k), int(round(float(n))), float(ph)])
    return terms


def _optional(values: dict, name: str) -> Optional[float]:
    """A value column that may be absent or blank."""
    v = values.get(name)
    return None if v is None or v == "" else float(v)


def from_forcefield(ff, strict: bool = True) -> CharmmLibrary:
    """Build a :class:`CharmmLibrary` from a resolved ``.frc`` force field.

    Parameters
    ----------
    ff : xnn.ffnn.common.frc.ForceField
        A force field of the CHARMM functional form (see the module
        docstring); the reader already normalised the nonbond sections to
        sigma/epsilon, and ``Rmin/2`` is recovered as ``sigma 2^(1/6) / 2``.
    strict : bool, optional
        If ``True`` (default) refuse a force field that carries functional
        forms this model does not implement; if ``False`` ignore those
        sections and record them in ``notes``.

    Returns
    -------
    CharmmLibrary
        The library, including the file's templates and residue fragments.

    Raises
    ------
    ValueError
        If the nonbond combination rule is not arithmetic, or an
        unsupported section is present under ``strict``.
    """
    lib = CharmmLibrary(name=ff.name)
    lib.metadata = dict(ff.metadata)
    lib.templates = dict(ff.templates)
    lib.fragments = dict(ff.fragments)
    lib.references = [r.text for r in ff.references_used()]
    lib.coulomb_constant = float(lib.metadata.get("coulomb_constant", COULOMB_CONSTANT))
    lib.scale_14_coulomb = float(lib.metadata.get("scale_14_coulomb", SCALE_14_COULOMB))

    unsupported = [k for k in ff.sections
                   if k in ("simple_fourier_angle", "tabulated_angle",
                            "quartic_bond", "quartic_angle", "torsion_1",
                            "torsion_3", "torsion_opls", "improper_opls",
                            "improper_fourier", "buckingham", "nonbond(9-6)",
                            "wilson_out_of_plane")]
    if unsupported:
        detail = {k: len(ff.sections[k]) for k in unsupported}
        if strict:
            raise ValueError(
                f"force field {ff.name!r} uses functional forms the CHARMM "
                f"model does not implement: {detail} (section: rows). Pass "
                "strict=False to load it without those sections")
        for k, n in detail.items():
            lib.notes.append(f"ignored section {k} ({n} rows): functional "
                             "form not implemented by the CHARMM model")
    if "nonbond(12-6)" in ff.sections and ff.combination() != "arithmetic":
        raise ValueError(f"force field {ff.name!r} uses the "
                         f"{ff.combination()!r} combination rule; CHARMM "
                         "requires arithmetic (Lorentz-Berthelot)")

    rows14 = ff.rows("nonbond_1-4(12-6)")
    for name, at in ff.atom_types.items():
        entry = {}
        try:
            entry["element"] = atomic_number(str(at.get("El", "")))
        except KeyError:
            entry["element"] = 0
        entry["mass"] = float(at.get("Mass", 0.0))
        conns = at.get("connections", 0)
        try:
            entry["connections"] = int(conns)
        except (TypeError, ValueError):
            entry["connections"] = 0
        entry["comment"] = str(at.get("Comment", ""))
        lj = ff.nonbond(name)
        if lj is None:
            lj = (0.0, 0.0)
            lib.notes.append(f"atom type {name} has no nonbond(12-6) entry; "
                             "Rmin/2 = epsilon = 0")
        entry["rstar"] = float(lj[0]) * TWO_SIXTH / 2.0
        entry["epsilon"] = float(lj[1])
        row = rows14.get((name,))
        if row is None:
            entry["rstar14"], entry["epsilon14"] = None, None
        else:
            vals = list(row.values.values())
            entry["rstar14"] = float(vals[0]) * TWO_SIXTH / 2.0
            entry["epsilon14"] = float(vals[1])
        lib.atom_types[name] = entry

    def key_of(key) -> str:
        return "-".join(WILDCARD if x == "*" else x for x in key)

    for key, row in ff.rows("quadratic_bond").items():
        lib.bond_types[key_of(key)] = {"k": float(row.values["K2"]),
                                       "r0": float(row.values["R0"])}
    for key, row in ff.rows("quadratic_angle").items():
        lib.angle_types[key_of(key)] = {"k": float(row.values["K2"]),
                                        "theta0": float(row.values["Theta0"])}
    for key, row in ff.rows("urey_bradley").items():
        lib.urey_bradley_types[key_of(key)] = {"k": float(row.values["K2"]),
                                               "s0": float(row.values["S0"])}
    for key, row in ff.rows("torsion_fourier").items():
        lib.dihedral_types[key_of(key)] = {
            "terms": _terms_of(row.values, MAX_TERMS)}
    for key, row in ff.rows("improper_harmonic").items():
        lib.improper_types[key_of(key)] = {
            "k": float(row.values["K2"]),
            "psi0": float(row.values.get("Chi0", 0.0) or 0.0)}
    for key, row in ff.rows("nonbond_pair(12-6)").items():
        lib.nbfix_types[key_of(key)] = {
            "rmin": float(row.values["rmin"]), "eps": float(row.values["eps"]),
            "rmin14": _optional(row.values, "rmin14"),
            "eps14": _optional(row.values, "eps14")}
    return lib


def to_forcefield(lib: CharmmLibrary, name: Optional[str] = None,
                  version: str = "1.0"):
    """Write a :class:`CharmmLibrary` as an in-memory ``.frc`` file.

    The inverse of :func:`from_forcefield`: atom types, ``nonbond(12-6)``
    and ``nonbond_1-4(12-6)`` in ``rmin-eps`` form (``rmin = 2 Rmin/2``, the
    like-pair minimum) with arithmetic combination, the pair overrides, the
    bonded tables, the templates and fragments, a ``#metadata`` with the
    Coulomb constant and the 1,4 scaling, and a ``#define``.

    Parameters
    ----------
    lib : CharmmLibrary
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
    from ..common.frc import FrcFile, Define, Reference, Section, make_section
    name = name or lib.name or "charmm"
    label = name
    frc = FrcFile.empty()

    def sym(z):
        return CHEMICAL_SYMBOLS[int(z)] if 0 < int(z) < len(CHEMICAL_SYMBOLS) else "X"

    def split(key):
        return tuple("*" if x == WILDCARD else x for x in key.split("-"))

    def term_columns(terms):
        out = {}
        for m, (k, n, ph) in enumerate(terms[:MAX_TERMS], start=1):
            out[f"K{m}"] = float(k)
            out[f"n{m}"] = int(n)
            out[f"Phi0_{m}"] = float(ph)
        return out

    n_cols = max([len(v["terms"]) for v in lib.dihedral_types.values()] + [1])
    tcols = [c for m in range(1, min(n_cols, MAX_TERMS) + 1)
             for c in (f"K{m}", f"n{m}", f"Phi0_{m}")]
    meta = {"ff_form": lib.metadata.get("ff_form", "charmm"),
            "charges": lib.metadata.get("charges", "template"),
            "coulomb_constant": lib.coulomb_constant,
            "scale_14_coulomb": lib.scale_14_coulomb}
    descriptions = {"ff_form": "The functional form of the forcefield",
                    "charges": "How charges are assigned (residue templates)",
                    "coulomb_constant": "The Coulomb constant in kcal*Angstrom/mol/e^2",
                    "scale_14_coulomb": "Scaling of the 1,4 Coulomb interactions"}

    def pair_values(v):
        out = {"rmin": float(v["rmin"]), "eps": float(v["eps"])}
        if v.get("rmin14") is not None and v.get("eps14") is not None:
            out["rmin14"] = float(v["rmin14"])
            out["eps14"] = float(v["eps14"])
        return out

    sections = [
        make_section("metadata", label, ["Parameter"], ["Value", "Description"],
                     [((k,), {"Value": v, "Description": descriptions[k]})
                      for k, v in meta.items()], version=version),
        make_section("atom_types", label, ["Type"],
                     ["Mass", "El", "connections", "Comment"],
                     [((n,), {"Mass": float(a.get("mass", 0.0)),
                              "El": sym(a.get("element", 0)),
                              "connections": int(a.get("connections", 0) or 0),
                              "Comment": str(a.get("comment", ""))})
                      for n, a in lib.atom_types.items()], version=version),
        make_section("nonbond(12-6)", label, ["I"], ["rmin", "eps"],
                     [((n,), {"rmin": 2.0 * float(a.get("rstar", 0.0)),
                              "eps": float(a.get("epsilon", 0.0))})
                      for n, a in lib.atom_types.items()], version=version,
                     annotations=["E = eps(ij) * [(rmin(ij)/r(ij))**12 - 2 * (rmin(ij)/r(ij))**6]",
                                  "rmin(ij) = (rmin(i) + rmin(j)) / 2 = Rmin/2(i) + Rmin/2(j)",
                                  "eps(ij) = sqrt(eps(i) * eps(j))",
                                  "rmin(i) is the like-pair minimum distance, twice the Rmin/2 of the CHARMM parameter files"],
                     modifiers={"type": [["rmin-eps"]],
                                "combination": [["arithmetic"]]}),
        make_section("nonbond_1-4(12-6)", label, ["I"], ["rmin", "eps"],
                     [((n,), {"rmin": 2.0 * float(a["rstar14"]),
                              "eps": float(a["epsilon14"])})
                      for n, a in lib.atom_types.items()
                      if a.get("rstar14") is not None and a.get("epsilon14") is not None],
                     version=version,
                     annotations=["The Lennard-Jones parameters of the listed types in their 1,4 interactions",
                                  "(same form and combination as nonbond(12-6)); other types keep their regular values"],
                     modifiers={"type": [["rmin-eps"]],
                                "combination": [["arithmetic"]]}),
        make_section("nonbond_pair(12-6)", label, ["I", "J"],
                     ["rmin", "eps", "rmin14", "eps14"],
                     [(split(k), pair_values(v)) for k, v in lib.nbfix_types.items()],
                     version=version,
                     annotations=["Pair-specific overrides (NBFIX): rmin is the minimum distance of the pair I-J itself,",
                                  "eps its well depth; rmin14 / eps14 apply to the pair's 1,4 interactions (same as rmin / eps when absent)"]),
        make_section("quadratic_bond", label, ["I", "J"], ["R0", "K2"],
                     [(split(k), {"R0": float(v["r0"]), "K2": float(v["k"])})
                      for k, v in lib.bond_types.items()], version=version,
                     annotations=["E = K2 * (R - R0)^2"]),
        make_section("quadratic_angle", label, ["I", "J", "K"], ["Theta0", "K2"],
                     [(split(k), {"Theta0": float(v["theta0"]), "K2": float(v["k"])})
                      for k, v in lib.angle_types.items()], version=version,
                     annotations=["E = K2 * (Theta - Theta0)^2"]),
        make_section("urey_bradley", label, ["I", "J", "K"], ["S0", "K2"],
                     [(split(k), {"S0": float(v["s0"]), "K2": float(v["k"])})
                      for k, v in lib.urey_bradley_types.items()], version=version,
                     annotations=["E = K2 * (S - S0)^2, S the distance between the outer atoms I and K"]),
        make_section("torsion_fourier", label, ["I", "J", "K", "L"], tcols,
                     [(split(k), term_columns(v["terms"]))
                      for k, v in lib.dihedral_types.items()], version=version,
                     annotations=["E = sum_i K_i * [1 + cos(n_i * Phi - Phi0_i)], K_i per dihedral"]),
        make_section("improper_harmonic", label, ["I", "J", "K", "L"], ["K2", "Chi0"],
                     [(split(k), {"K2": float(v["k"]), "Chi0": float(v.get("psi0", 0.0))})
                      for k, v in lib.improper_types.items()], version=version,
                     annotations=["E = K2 * (Chi - Chi0)^2",
                                  "Chi is the dihedral angle over the atoms in the order given; the central atom is first in the",
                                  "topology templates, and a row matches a quadruple as written or reversed, X matching any type"]),
    ]
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
    text = "\n".join(lib.references) or f"CHARMM parameters exported from xnn ({name})."
    frc.references[("<memory>", "1")] = Reference(number="1", text=text, author="xnn")
    return frc


def read_charmm(source: Union[str, Path], strict: bool = True) -> CharmmLibrary:
    """Read a CHARMM parameter library.

    Parameters
    ----------
    source : str or Path
        A ``.json`` file in the native format, or a ``.frc`` force-field
        spec understood by :func:`~xnn.ffnn.common.frc.find_forcefield`: a
        variant shipped with xnn by name (``"charmm22"``), a ``.frc`` path,
        or ``"<path>.frc:<variant>"``.
    strict : bool, optional
        See :func:`from_forcefield`.

    Returns
    -------
    CharmmLibrary
        The parsed library.
    """
    s = str(source)
    if s.lower().endswith(".json"):
        return _read_json(Path(s))
    from ..common.frc import read_forcefield
    return from_forcefield(read_forcefield(s), strict=strict)


def builtin_library(name: str = "charmm22", strict: bool = True) -> CharmmLibrary:
    """Return one of the CHARMM parameter sets shipped with xnn.

    ``"charmm22"`` is the all-atom protein force field of MacKerell *et al.*
    (1998): the protein parameter and topology files (without the later
    CMAP correction), TIP3P water and ions.

    Parameters
    ----------
    name : str, optional
        The variant name, by default ``"charmm22"``.
    strict : bool, optional
        See :func:`from_forcefield`.

    Returns
    -------
    CharmmLibrary
        A fresh library instance (safe to modify).
    """
    return read_charmm(name, strict=strict)
