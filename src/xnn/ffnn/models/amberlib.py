"""AMBER parameter libraries: the SEAMM ``.frc`` format and native JSON.

An AMBER model (Cornell *et al.*, *J. Am. Chem. Soc.* 117, 5179, 1995) is
specified by two kinds of information:

* **per-atom-type parameters** -- the van der Waals well depth ``epsilon``
  and radius ``R*`` (half the like-pair minimum distance), plus the bonded
  tables keyed by atom types: harmonic bonds and angles, periodic torsions
  with one or more Fourier terms ``K [1 + cos(n phi - gamma)]`` per
  quadruple, and impropers of the same periodic form centred on the third
  atom (Table 14 of the paper, ``parm94.dat`` in the AMBER distribution);
* **residue templates** -- the atom names, atom types and RESP partial
  charges of every amino acid, nucleotide, capping group, ion and water
  (Figures 4-6 of the paper, the ``.lib`` residue libraries of AMBER).
  Charges belong to the residue, not to the atom type, which is what sets
  AMBER apart from OPLS or DREIDING.

Both live in one ``.frc`` file (:mod:`xnn.ffnn.common.frc`): the parameter
tables as ``quadratic_bond`` / ``quadratic_angle`` / ``torsion_fourier`` /
``improper_fourier`` / ``nonbond(12-6)`` sections, and the residue templates
as ``#fragments`` whose SMARTS carry atom maps in template order together
with ``atom names``, ``atom types`` and ``charges`` lists. ``#templates``
(per-type SMARTS) type small organic molecules outside the residue
libraries, which then need charges from the user. ``ff94`` ships with xnn
as ``amber.frc``. A library can also be round-tripped through the native
JSON of this module (:meth:`AmberLibrary.save` / :func:`read_amber`).

Energies are converted from kcal/mol to eV only when the model assembles
its parameter tensors, as for OPLS.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

from xnn.common.data.elements import CHEMICAL_SYMBOLS, atomic_number

from .oplslib import (  # noqa: F401  # KCAL_TO_EV and resolve_dihedral_type are re-exported
    KCAL_TO_EV,
    WILDCARD,
    resolve_angle_type,
    resolve_bond_type,
    resolve_dihedral_type,
)

# 1,4 scaling of the Cornell et al. force field (section C and E of the
# paper): van der Waals by 1/2, electrostatics by 1/1.2.
SCALE_14_VDW = 0.5
SCALE_14_COULOMB = 1.0 / 1.2
TWO_SIXTH = 2.0 ** (1.0 / 6.0)
_TERMS = ("nonbond", "bond", "angle", "torsion", "oop")
# at most this many Fourier terms per torsion / improper row
MAX_TERMS = 4
MAX_IMPROPER_TERMS = 2


@dataclass
class AmberLibrary:
    """A parsed AMBER parameter library, in AMBER units.

    Energies are kcal/mol, lengths Angstrom, angles degrees. Bonded
    parameters are keyed by the atom types of the term joined with ``-``
    (``"CT-CT"``, ``"X-CT-CT-X"``); ``X`` is the wildcard.

    Parameters and attributes
    -------------------------
    atom_types : dict[str, dict]
        ``name -> {"element", "mass", "rstar", "epsilon", "cls_nonbond",
        "cls_bond", "cls_angle", "cls_torsion", "cls_oop", "connections",
        "comment"}``; ``rstar`` is the van der Waals radius ``R*`` (half the
        like-pair minimum distance) in Angstrom and ``epsilon`` the well
        depth in kcal/mol.
    bond_types : dict[str, dict]
        ``"A-B" -> {"k", "r0"}`` for ``E = k (r - r0)^2``.
    angle_types : dict[str, dict]
        ``"A-B-C" -> {"k", "theta0"}`` for ``E = k (theta - theta0)^2``,
        ``theta0`` in degrees.
    dihedral_types : dict[str, dict]
        ``"A-B-C-D" -> {"terms": [[K, n, gamma], ...]}`` for
        ``E = sum K [1 + cos(n phi - gamma)]`` per dihedral (``K`` already
        divided by the number of paths of the AMBER file).
    improper_types : dict[str, dict]
        ``"I-J-K-L" -> {"terms": [[K, n, gamma], ...]}`` with ``K`` the
        central atom and ``L`` the distinguished outer atom, in the order
        written (the positions carry meaning, see :func:`match_improper`).
    fudge_lj, fudge_qq : float
        Scaling factors of the 1,4 van der Waals and Coulomb interactions
        (1/2 and 1/1.2 for AMBER).
    name : str
        A short label for the parameter set.
    references : list[str]
        Literature provenance of the parameters.
    templates : dict
        Per-type SMARTS templates (``type -> {"smarts": [...], ...}``).
    fragments : dict
        Residue templates (``name -> {"SMARTS", "atom names", "atom types",
        "charges", ...}``), consumed by
        :func:`~xnn.ffnn.common.typing.assign_atom_types`.
    metadata : dict
        The ``#metadata`` entries.
    notes : list[str]
        Anything the reader had to fill in.
    """

    atom_types: dict = field(default_factory=dict)
    bond_types: dict = field(default_factory=dict)
    angle_types: dict = field(default_factory=dict)
    dihedral_types: dict = field(default_factory=dict)
    improper_types: dict = field(default_factory=dict)
    fudge_lj: float = SCALE_14_VDW
    fudge_qq: float = SCALE_14_COULOMB
    name: str = "amber"
    references: list = field(default_factory=list)
    templates: dict = field(default_factory=dict)
    fragments: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def cls(self, type_name: str, term: str = "bond") -> str:
        """The equivalent type an atom type uses for ``term``.

        Parameters
        ----------
        type_name : str
            An atom type of the library.
        term : str
            ``"nonbond"``, ``"bond"``, ``"angle"``, ``"torsion"`` or ``"oop"``.

        Returns
        -------
        str
            The equivalent type; the type itself when none is recorded.
        """
        entry = self.atom_types[type_name]
        return str(entry.get(f"cls_{term}", type_name))

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

    def torsion(self, key: str) -> Optional[dict]:
        """The torsion entry whose key is ``"A-B-C-D"`` or its reverse, exactly
        (no wildcard matching; ``None`` if absent)."""
        parts = key.split("-")
        for cand in ("-".join(parts), "-".join(parts[::-1])):
            if cand in self.dihedral_types:
                return self.dihedral_types[cand]
        return None

    def improper(self, key: str) -> Optional[dict]:
        """The improper entry of ``"I-J-K-L"`` (first two positions in either
        order; ``None`` if absent)."""
        i, j, k, l = key.split("-")
        for cand in (f"{i}-{j}-{k}-{l}", f"{j}-{i}-{k}-{l}"):
            if cand in self.improper_types:
                return self.improper_types[cand]
        return None

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
        """Write the library as native JSON (read back by :func:`read_amber`).

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
        """Write the library as a ``.frc`` force-field file (see
        :func:`to_forcefield`)."""
        return to_forcefield(self, name=name, version=version).write(path)


# improper placement and resolution
def _pattern_matches(pattern: Sequence[str], types: Sequence[str]) -> bool:
    """Whether the outer-position pattern (wildcards allowed) matches ``types``."""
    return all(p == WILDCARD or p == t for p, t in zip(pattern, types))


def match_improper(improper_types: dict, center: str, outer: Sequence[tuple]
                   ) -> Optional[tuple[str, tuple]]:
    """Pick the improper parameter and atom order for one trigonal center.

    Follows the conventions of the AMBER builder (LEaP) as reproduced by the
    ``amber`` improper ordering of OpenMM: a pattern ``I-J-K-L`` matches a
    center ``K`` when its three neighbours can be assigned to ``I``, ``J``
    and ``L`` (``X`` matches anything); a pattern without wildcards is
    preferred over one with, the last matching wildcard-free pattern wins
    among those, else the first matching wildcard pattern (table order).
    The matched atoms are then ordered: between positions of equal type
    (wildcard-free patterns) or of equal element involving ``L`` (wildcard
    patterns), and always between the two leading positions of a wildcard
    pattern, the atom that comes first in its residue template goes first.

    Parameters
    ----------
    improper_types : dict
        The library's improper table in file order, keys ``"I-J-K-L"``.
    center : str
        Atom type of the central atom.
    outer : sequence of (int, str, str, tuple)
        The three neighbours as ``(atom index, atom type, element symbol,
        order key)``, where the order key is ``(residue index, position in
        the residue template)``.

    Returns
    -------
    tuple of (str, tuple) or None
        The matching key and the atom indices ``(i, j, k, l)`` in
        evaluation order, or ``None`` when no pattern applies.
    """
    best = None
    for key in improper_types:
        pat = key.split("-")
        if len(pat) != 4 or pat[2] != center:
            continue
        has_x = WILDCARD in (pat[0], pat[1], pat[3])
        if best is not None and has_x:
            continue
        for perm in itertools.permutations(outer):
            if not _pattern_matches((pat[0], pat[1], pat[3]),
                                    (perm[0][1], perm[1][1], perm[2][1])):
                continue
            a, b, c = perm                     # positions I, J, L
            if not has_x:
                if a[1] == c[1] and a[3] > c[3]:
                    a, c = c, a
                if b[1] == c[1] and b[3] > c[3]:
                    b, c = c, b
                if a[1] == b[1] and a[3] > b[3]:
                    a, b = b, a
            else:
                if a[2] == c[2] and a[3] > c[3]:
                    a, c = c, a
                if b[2] == c[2] and b[3] > c[3]:
                    b, c = c, b
                if a[3] > b[3]:
                    a, b = b, a
            best = (key, (a[0], b[0]), c[0])
            break
    if best is None:
        return None
    key, (i, j), l = best
    return key, (i, j, l)


def place_impropers(topology, type_cls: Sequence[str], elements: Sequence[str],
                    improper_types: dict) -> tuple[list, list]:
    """Generate the improper list of a topology the AMBER way.

    One improper per atom with exactly three bonded neighbours for which
    :func:`match_improper` finds a pattern (centers without one get none).

    Parameters
    ----------
    topology : MolecularTopology
        The topology; its ``residues`` give the template order used to
        order wildcard positions (atoms without one order by index).
    type_cls : sequence of str
        Per-atom out-of-plane class (the atom type for AMBER).
    elements : sequence of str
        Per-atom element symbols.
    improper_types : dict
        The library's improper table.

    Returns
    -------
    tuple of (list, list)
        ``(impropers, keys)``: the ``(i, j, k, l)`` quadruples (center
        third) and the matching library key of each.
    """
    neighbors: dict[int, list[int]] = {}
    for i, j in topology.bonds:
        neighbors.setdefault(i, []).append(j)
        neighbors.setdefault(j, []).append(i)
    residues = list(topology.residues)

    def order_key(a: int) -> tuple:
        r = residues[a] if a < len(residues) else None
        return (-1, a) if r is None else (int(r[0]), int(r[1]))

    impropers, keys = [], []
    for center in sorted(neighbors):
        nb = sorted(neighbors[center])
        if len(nb) != 3:
            continue
        outer = [(a, type_cls[a], elements[a], order_key(a)) for a in nb]
        found = match_improper(improper_types, type_cls[center], outer)
        if found is None:
            continue
        key, (i, j, l) = found
        impropers.append((i, j, center, l))
        keys.append(key)
    return impropers, keys


def resolve_improper_type(improper_types: dict, i: str, j: str, k: str,
                          l: str) -> Optional[str]:
    """Find the improper key for atom types already in evaluation order.

    Parameters
    ----------
    improper_types : dict
        The library's improper table.
    i, j, k, l : str
        Atom types with ``k`` central and ``l`` the distinguished atom.

    Returns
    -------
    str or None
        The key (exact positions, then wildcard patterns in table order),
        or ``None``.
    """
    key = f"{i}-{j}-{k}-{l}"
    if key in improper_types:
        return key
    alt = f"{j}-{i}-{k}-{l}"
    if alt in improper_types:
        return alt
    best = None
    for cand in improper_types:
        pat = cand.split("-")
        if len(pat) != 4 or pat[2] != k:
            continue
        if _pattern_matches((pat[0], pat[1], pat[3]), (i, j, l)) or \
                _pattern_matches((pat[0], pat[1], pat[3]), (j, i, l)):
            if best is None:
                best = cand
    return best


# readers
def _read_json(path: Path) -> AmberLibrary:
    """Read the native JSON library format."""
    data = json.loads(path.read_text())
    return AmberLibrary(
        atom_types=data.get("atom_types", {}),
        bond_types=data.get("bond_types", {}),
        angle_types=data.get("angle_types", {}),
        dihedral_types=data.get("dihedral_types", {}),
        improper_types=data.get("improper_types", {}),
        fudge_lj=float(data.get("fudge_lj", SCALE_14_VDW)),
        fudge_qq=float(data.get("fudge_qq", SCALE_14_COULOMB)),
        name=data.get("name", path.stem),
        references=list(data.get("references", [])),
        templates=dict(data.get("templates", {})),
        fragments=dict(data.get("fragments", {})),
        metadata=dict(data.get("metadata", {})))


def _terms_of(values: dict, max_terms: int) -> list[list[float]]:
    """``[[K, n, gamma], ...]`` from the ``K1 n1 Phi0_1 ...`` columns of a row."""
    terms = []
    for m in range(1, max_terms + 1):
        k = values.get(f"K{m}")
        if k is None or k == "":
            break
        n = values.get(f"n{m}", 0)
        ph = values.get(f"Phi0_{m}", 0.0)
        terms.append([float(k), int(round(float(n))), float(ph)])
    return terms


def from_forcefield(ff, strict: bool = True) -> AmberLibrary:
    """Build an :class:`AmberLibrary` from a resolved ``.frc`` force field.

    Parameters
    ----------
    ff : xnn.ffnn.common.frc.ForceField
        A force field of the AMBER functional form: ``quadratic_bond``,
        ``quadratic_angle``, ``torsion_fourier``, ``improper_fourier`` and a
        ``nonbond(12-6)`` section with arithmetic combination (the reader
        already normalised its ``@type`` to sigma/epsilon; ``R*`` is
        recovered as ``sigma 2^(1/6) / 2``).
    strict : bool, optional
        If ``True`` (default) refuse a force field that carries functional
        forms this model does not implement; if ``False`` ignore those
        sections and record them in ``notes``.

    Returns
    -------
    AmberLibrary
        The library, including the file's templates and residue fragments.

    Raises
    ------
    ValueError
        If the nonbond combination rule is not arithmetic, or an
        unsupported section is present under ``strict``.
    """
    lib = AmberLibrary(name=ff.name)
    lib.metadata = dict(ff.metadata)
    lib.templates = dict(ff.templates)
    lib.fragments = dict(ff.fragments)
    lib.references = [r.text for r in ff.references_used()]
    lib.fudge_lj = float(lib.metadata.get("scale_14_vdw", SCALE_14_VDW))
    lib.fudge_qq = float(lib.metadata.get("scale_14_coulomb", SCALE_14_COULOMB))

    unsupported = [k for k in ff.sections
                   if k in ("simple_fourier_angle", "tabulated_angle",
                            "quartic_bond", "quartic_angle", "torsion_1",
                            "torsion_3", "torsion_opls", "improper_opls",
                            "buckingham", "nonbond(9-6)",
                            "wilson_out_of_plane")]
    if unsupported:
        detail = {k: len(ff.sections[k]) for k in unsupported}
        if strict:
            raise ValueError(
                f"force field {ff.name!r} uses functional forms the AMBER "
                f"model does not implement: {detail} (section: rows). Pass "
                "strict=False to load it without those sections")
        for k, n in detail.items():
            lib.notes.append(f"ignored section {k} ({n} rows): functional "
                             "form not implemented by the AMBER model")
    if "nonbond(12-6)" in ff.sections and ff.combination() != "arithmetic":
        raise ValueError(f"force field {ff.name!r} uses the "
                         f"{ff.combination()!r} combination rule; AMBER "
                         "requires arithmetic (Lorentz-Berthelot)")

    for name, at in ff.atom_types.items():
        entry = {f"cls_{t}": ff.equivalent(name, t) for t in _TERMS}
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
                             "R* = epsilon = 0")
        entry["rstar"] = float(lj[0]) * TWO_SIXTH / 2.0
        entry["epsilon"] = float(lj[1])
        lib.atom_types[name] = entry

    def key_of(key) -> str:
        return "-".join(WILDCARD if x == "*" else x for x in key)

    for key, row in ff.rows("quadratic_bond").items():
        lib.bond_types[key_of(key)] = {"k": float(row.values["K2"]),
                                       "r0": float(row.values["R0"])}
    for key, row in ff.rows("quadratic_angle").items():
        lib.angle_types[key_of(key)] = {"k": float(row.values["K2"]),
                                        "theta0": float(row.values["Theta0"])}
    for key, row in ff.rows("torsion_fourier").items():
        lib.dihedral_types[key_of(key)] = {
            "terms": _terms_of(row.values, MAX_TERMS)}
    for key, row in ff.rows("improper_fourier").items():
        lib.improper_types[key_of(key)] = {
            "terms": _terms_of(row.values, MAX_IMPROPER_TERMS)}
    return lib


def to_forcefield(lib: AmberLibrary, name: Optional[str] = None,
                  version: str = "1.0"):
    """Write an :class:`AmberLibrary` as an in-memory ``.frc`` file.

    The inverse of :func:`from_forcefield`: atom types, equivalences,
    ``nonbond(12-6)`` in ``rmin-eps`` form (``rmin = 2 R*``, the like-pair
    minimum) with arithmetic combination, the four bonded tables, the
    templates and fragments, a ``#metadata`` with the 1,4 scaling factors,
    and a ``#define``.

    Parameters
    ----------
    lib : AmberLibrary
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
    from ..common.frc import Define, FrcFile, Reference, Section, make_section
    name = name or lib.name or "amber"
    label = name
    frc = FrcFile.empty()

    def sym(z):
        return CHEMICAL_SYMBOLS[int(z)] if 0 < int(z) < len(CHEMICAL_SYMBOLS) else "X"

    def split(key):
        return tuple("*" if x == WILDCARD else x for x in key.split("-"))

    def term_columns(terms, max_terms):
        out = {}
        for m, (k, n, ph) in enumerate(terms[:max_terms], start=1):
            out[f"K{m}"] = float(k)
            out[f"n{m}"] = int(n)
            out[f"Phi0_{m}"] = float(ph)
        return out

    tcols = [c for m in range(1, MAX_TERMS + 1) for c in (f"K{m}", f"n{m}", f"Phi0_{m}")]
    icols = [c for m in range(1, MAX_IMPROPER_TERMS + 1)
             for c in (f"K{m}", f"n{m}", f"Phi0_{m}")]
    meta = {"ff_form": lib.metadata.get("ff_form", "amber"),
            "charges": lib.metadata.get("charges", "template"),
            "scale_14_vdw": lib.fudge_lj, "scale_14_coulomb": lib.fudge_qq}
    descriptions = {"ff_form": "The functional form of the forcefield",
                    "charges": "How charges are assigned (residue templates)",
                    "scale_14_vdw": "Scaling of the 1,4 van der Waals interactions",
                    "scale_14_coulomb": "Scaling of the 1,4 Coulomb interactions"}
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
        make_section("equivalence", label, ["Type"],
                     ["NonB", "Bond", "Angle", "Torsion", "OOP"],
                     [((n,), {"NonB": lib.cls(n, "nonbond"), "Bond": lib.cls(n, "bond"),
                              "Angle": lib.cls(n, "angle"),
                              "Torsion": lib.cls(n, "torsion"),
                              "OOP": lib.cls(n, "oop")})
                      for n in lib.atom_types], version=version),
        make_section("nonbond(12-6)", label, ["I"], ["rmin", "eps"],
                     [((n,), {"rmin": 2.0 * float(a.get("rstar", 0.0)),
                              "eps": float(a.get("epsilon", 0.0))})
                      for n, a in lib.atom_types.items()
                      if lib.cls(n, "nonbond") == n], version=version,
                     annotations=["E = eps(ij) * [(rmin(ij)/r(ij))**12 - 2 * (rmin(ij)/r(ij))**6]",
                                  "rmin(ij) = (rmin(i) + rmin(j)) / 2 = R*(i) + R*(j)",
                                  "eps(ij) = sqrt(eps(i) * eps(j))",
                                  "rmin(i) is the like-pair minimum distance, "
                                  "twice the R* of the Amber parameter files"],
                     modifiers={"type": [["rmin-eps"]],
                                "combination": [["arithmetic"]]}),
        make_section("quadratic_bond", label, ["I", "J"], ["R0", "K2"],
                     [(split(k), {"R0": float(v["r0"]), "K2": float(v["k"])})
                      for k, v in lib.bond_types.items()], version=version,
                     annotations=["E = K2 * (R - R0)^2"]),
        make_section("quadratic_angle", label, ["I", "J", "K"], ["Theta0", "K2"],
                     [(split(k), {"Theta0": float(v["theta0"]), "K2": float(v["k"])})
                      for k, v in lib.angle_types.items()], version=version,
                     annotations=["E = K2 * (Theta - Theta0)^2"]),
        make_section("torsion_fourier", label, ["I", "J", "K", "L"], tcols,
                     [(split(k), term_columns(v["terms"], MAX_TERMS))
                      for k, v in lib.dihedral_types.items()], version=version,
                     annotations=["E = sum_i K_i * [1 + cos(n_i * Phi - Phi0_i)]",
                                  "K_i is per dihedral: the Amber barrier divided by its number of paths"]),
        make_section("improper_fourier", label, ["I", "J", "K", "L"], icols,
                     [(split(k), term_columns(v["terms"], MAX_IMPROPER_TERMS))
                      for k, v in lib.improper_types.items()], version=version,
                     annotations=["E = sum_i K_i * [1 + cos(n_i * Phi - Phi0_i)]",
                                  "Phi is the dihedral angle over the atoms in the order given, K is the central atom",
                                  "L is the distinguished outer atom; wildcards X precede it"]),
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
    text = "\n".join(lib.references) or f"AMBER parameters exported from xnn ({name})."
    frc.references[("<memory>", "1")] = Reference(number="1", text=text, author="xnn")
    return frc


def read_amber(source: Union[str, Path], strict: bool = True) -> AmberLibrary:
    """Read an AMBER parameter library.

    Parameters
    ----------
    source : str or Path
        A ``.json`` file in the native format, or a ``.frc`` force-field
        spec understood by :func:`~xnn.ffnn.common.frc.find_forcefield`: a
        variant shipped with xnn by name (``"ff94"``), a ``.frc`` path, or
        ``"<path>.frc:<variant>"``.
    strict : bool, optional
        See :func:`from_forcefield`.

    Returns
    -------
    AmberLibrary
        The parsed library.
    """
    s = str(source)
    if s.lower().endswith(".json"):
        return _read_json(Path(s))
    from ..common.frc import read_forcefield
    return from_forcefield(read_forcefield(s), strict=strict)


def builtin_library(name: str = "ff94", strict: bool = True) -> AmberLibrary:
    """Return one of the AMBER parameter sets shipped with xnn.

    ``"ff94"`` is the Cornell *et al.* (1995) force field: ``parm94.dat``
    with the ff94 residue libraries (amino acids with their N- and
    C-terminal forms, nucleotides, capping groups, TIP3P water, ions).

    Parameters
    ----------
    name : str, optional
        The variant name, by default ``"ff94"``.
    strict : bool, optional
        See :func:`from_forcefield`.

    Returns
    -------
    AmberLibrary
        A fresh library instance (safe to modify).
    """
    return read_amber(name, strict=strict)
