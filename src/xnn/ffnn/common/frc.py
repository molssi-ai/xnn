"""The MolSSI/SEAMM ``.frc`` force-field file format.

This is the one parameter-file format the ``ffnn`` family reads. It is the
format of the SEAMM force-field distribution
(https://github.com/molssi-seamm/forcefield_step, ``forcefield_step/data``),
a descendant of the Biosym ``.frc`` files, and it is force-field agnostic:
OPLS-AA, ReaxFF, PCFF, Dreiding and the Buckingham potentials of battery
materials all ship in it. What makes it worth adopting wholesale is that a
file carries not only parameters but the *typing rules*: a ``#templates``
section of SMARTS patterns that assign the file's own atom types to any
structure (see :mod:`xnn.ffnn.common.typing`).

Grammar
-------
A file opens with a ``!MolSSI forcefield 1`` line: the word after ``!`` names
the file's dialect (``MolSSI`` today, ``BIOSYM`` in the Biosym-era files the
format descends from), and the trailing number is the **format version**,
that is the version of the file grammar, not of the parameters inside. The
current and only published format version is 1, which is what this module
implements and writes (:data:`FRC_FORMAT_VERSION`); a file declaring a newer
number is still parsed, with a warning. Parameter versions live elsewhere: in
the ``Version`` column of every data row and ``#define`` row (a date or a
dotted number), and the reader always keeps the newest one per key.
Everything else is organised in sections that start at a line beginning with
``#`` and run to the next such line::

    #<kind> <label>            e.g.  #quadratic_bond oplsaa

Inside a section, ``!`` lines are comments (the last one before the data is
the column header), ``>`` lines are annotations (the energy expression),
``@`` lines are modifiers (``@units K2 kJ/mol/nm^2``, ``@type rmin-eps``,
``@combination geometric``), and every other line is a data row::

    Version     Ref   <atom-type key columns>   <value columns>
    2023.01.29  1     opls_18   opls_18        1.5290   268.00

Special sections: ``#define <name>`` lists, per functional form, which
labelled sections make up the force-field variant ``<name>`` (a file may
define several variants, so ``oplsaa.frc`` defines ``oplsaa``, ``CL&P`` and
``oplsaa+``); ``#include <file> [missing_ok]`` splices another file in;
``#templates <label>`` and ``#fragments <label>`` hold JSON;
``#reference <n>`` holds free-text provenance; ``#end`` closes a section and
is otherwise ignored.

Resolution
----------
:meth:`FrcFile.forcefield` turns a ``#define`` into a :class:`ForceField`:
for every functional form, the newest row of the define (or the newest not
above a requested version) names the labels to use, in order; the labelled
sections are merged in that order so a later label overrides an earlier one
for the same key (this is how ``CL&P`` extends ``oplsaa``), and within a
section the newest version of each key wins. A label ending in
``:optional`` may be absent. The result offers the bonded-parameter lookups
force-field codes need -- direct key, then the type's equivalences, then the
wildcard patterns in SEAMM's precedence order -- and the merged templates.

The reader is *schema driven*: ``SECTION_SCHEMA`` records, per section
kind, how many leading columns form the atom-type key, the key's symmetry
(``like_bond``: ``i <= j``; ``like_angle``: ``i <= k``; ``like_torsion``;
``like_improper``: central atom third), and the value columns with their
default units, so ``@units`` modifiers are honoured. Kinds without a schema
are read from their header line. New force fields register their sections
with :func:`register_section_schema`.
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

try:
    from packaging.version import Version as _Version
except ImportError:  # pragma: no cover - packaging ships with setuptools
    _Version = None

logger = logging.getLogger(__name__)

FRC_FORMAT_VERSION = 1
"""The ``.frc`` format version this module reads and writes (the trailing
number of the ``!MolSSI forcefield 1`` header). SEAMM has published only
version 1 so far; every file shipped in :mod:`xnn.ffnn.data` declares it."""
FRC_HEADER = f"!MolSSI forcefield {FRC_FORMAT_VERSION}"


def parse_header(line: str) -> tuple[str, str, Optional[int]]:
    """Split a ``!<dialect> <kind> <version>`` header line.

    Returns ``(dialect, kind, version)``, e.g. ``("MolSSI", "forcefield", 1)``
    for the standard header or ``("BIOSYM", "forcefield", 1)`` for a legacy
    Biosym file. Missing or non-numeric parts come back as ``""`` / ``None``.
    """
    words = line.strip()[1:].split() if line.strip().startswith("!") else []
    dialect = words[0] if words else ""
    kind = words[1] if len(words) > 1 else ""
    version: Optional[int] = None
    if len(words) > 2:
        try:
            version = int(words[2])
        except ValueError:
            version = None
    return dialect, kind, version
WILDCARD = "*"


# ----------------------------------------------------------------------
# versions
# ----------------------------------------------------------------------
def parse_version(text: str):
    """Comparable form of a version token such as ``2023.01.29`` or ``1.0``.

    Parameters
    ----------
    text : str
        The version column of a ``.frc`` row.

    Returns
    -------
    object
        A totally ordered key (``packaging.version.Version`` when available,
        else a tuple of integers).
    """
    if _Version is not None:
        try:
            return _Version(text)
        except Exception:
            pass
    return tuple(int(x) if x.isdigit() else 0 for x in re.split(r"[.\-]", text))


# ----------------------------------------------------------------------
# units
# ----------------------------------------------------------------------
# dimension vector: (energy, length, angle, charge, mass); scale converts to
# the reference units kcal/mol, Angstrom, degree, e, Dalton
_BASE_UNITS: dict[str, tuple[float, tuple[int, int, int, int, int]]] = {
    "kcal/mol": (1.0, (1, 0, 0, 0, 0)),
    "kJ/mol": (1.0 / 4.184, (1, 0, 0, 0, 0)),
    "eV": (23.060547830619026, (1, 0, 0, 0, 0)),         # 96.48533212331 kJ/mol / 4.184
    "hartree": (627.5094740631, (1, 0, 0, 0, 0)),
    "Å": (1.0, (0, 1, 0, 0, 0)),
    "angstrom": (1.0, (0, 1, 0, 0, 0)),
    "A": (1.0, (0, 1, 0, 0, 0)),
    "nm": (10.0, (0, 1, 0, 0, 0)),
    "pm": (0.01, (0, 1, 0, 0, 0)),
    "bohr": (0.529177210903, (0, 1, 0, 0, 0)),
    "degree": (1.0, (0, 0, 1, 0, 0)),
    "degrees": (1.0, (0, 0, 1, 0, 0)),
    "deg": (1.0, (0, 0, 1, 0, 0)),
    "radian": (180.0 / math.pi, (0, 0, 1, 0, 0)),
    "rad": (180.0 / math.pi, (0, 0, 1, 0, 0)),
    "e": (1.0, (0, 0, 0, 1, 0)),
    "Dalton": (1.0, (0, 0, 0, 0, 1)),
    "Da": (1.0, (0, 0, 0, 0, 1)),
    "amu": (1.0, (0, 0, 0, 0, 1)),
    "g/mol": (1.0, (0, 0, 0, 0, 1)),
    "": (1.0, (0, 0, 0, 0, 0)),
}
# compound tokens that appear verbatim in files (the tokenizer splits on
# operators, so ``kcal/mol`` must be recognised as one unit, not kcal per mol)
_UNIT_TOKEN = re.compile(r"kcal/mol|kJ/mol|g/mol|[A-Za-zÅ]+")


def _parse_unit(text: str) -> tuple[float, tuple[float, ...]]:
    """Parse a unit expression into ``(scale, dimension)``.

    Supports products, quotients, parentheses and integer or rational powers
    written ``^n``, ``**n`` or ``**(p/q)``, e.g. ``kJ/mol/nm^2``,
    ``kcal/mol*Å**6``, ``(kJ/mol)**(1/12)*nm``.
    """
    s = text.strip().replace("**", "^")
    pos = [0]

    def peek():
        return s[pos[0]] if pos[0] < len(s) else ""

    def skip():
        while peek() == " ":
            pos[0] += 1

    def number():
        skip()
        m = re.match(r"[+-]?\d+(\.\d+)?", s[pos[0]:])
        if not m:
            raise ValueError(f"bad exponent in unit {text!r}")
        pos[0] += m.end()
        return float(m.group(0))

    def exponent():
        skip()
        if peek() == "(":
            pos[0] += 1
            p = number()
            skip()
            q = 1.0
            if peek() == "/":
                pos[0] += 1
                q = number()
            skip()
            if peek() != ")":
                raise ValueError(f"unbalanced exponent in unit {text!r}")
            pos[0] += 1
            return p / q
        return number()

    def atom():
        skip()
        if peek() == "(":
            pos[0] += 1
            val = expr()
            skip()
            if peek() != ")":
                raise ValueError(f"unbalanced parentheses in unit {text!r}")
            pos[0] += 1
            return val
        m = _UNIT_TOKEN.match(s, pos[0])
        if not m:
            raise ValueError(f"cannot parse unit {text!r} at {s[pos[0]:]!r}")
        pos[0] = m.end()
        name = m.group(0)
        if name not in _BASE_UNITS:
            raise ValueError(f"unknown unit {name!r} in {text!r}")
        scale, dim = _BASE_UNITS[name]
        return scale, tuple(float(d) for d in dim)

    def term():
        scale, dim = atom()
        skip()
        if peek() == "^":
            pos[0] += 1
            e = exponent()
            scale, dim = scale ** e, tuple(d * e for d in dim)
        return scale, dim

    def expr():
        scale, dim = term()
        while True:
            skip()
            op = peek()
            if op == "*":
                pos[0] += 1
                s2, d2 = term()
                scale, dim = scale * s2, tuple(a + b for a, b in zip(dim, d2))
            elif op == "/":
                pos[0] += 1
                s2, d2 = term()
                scale, dim = scale / s2, tuple(a - b for a, b in zip(dim, d2))
            else:
                return scale, dim

    if not s:
        return 1.0, (0.0,) * 5
    result = expr()
    skip()
    if pos[0] != len(s):
        raise ValueError(f"trailing text in unit {text!r}: {s[pos[0]:]!r}")
    return result


def convert_units(value: float, src: str, dst: str) -> float:
    """Convert ``value`` from unit ``src`` to unit ``dst``.

    Parameters
    ----------
    value : float
        The quantity in ``src`` units.
    src, dst : str
        Unit expressions: products, quotients, parentheses and powers of the
        base units (``kJ/mol/nm^2``, ``kcal/mol*Å**6``, ``(kJ/mol)**(1/12)*nm``).

    Returns
    -------
    float
        The quantity in ``dst`` units.

    Raises
    ------
    ValueError
        If the two units have different dimensions or cannot be parsed.
    """
    if src.strip() == dst.strip():
        return value
    s_scale, s_dim = _parse_unit(src)
    d_scale, d_dim = _parse_unit(dst)
    if any(abs(a - b) > 1e-9 for a, b in zip(s_dim, d_dim)):
        raise ValueError(f"cannot convert {src!r} to {dst!r}: different "
                         "dimensions")
    return value * s_scale / d_scale


def unit_factor(src: str, dst: str) -> float:
    """Multiplicative factor taking ``src`` units to ``dst`` units."""
    return convert_units(1.0, src, dst)


# ----------------------------------------------------------------------
# section schema
# ----------------------------------------------------------------------
def _auto(tok: str):
    """Float if the token parses as one, else the token itself."""
    try:
        return float(tok)
    except ValueError:
        return tok


# kind -> {"n_atoms", "symmetry", "type", "constants": [(name, unit, ctor)]
#          or None (take names from the header), "text": trailing text column
#          name or None, "flip": n (swap the first n and next n values when
#          the key was flipped into canonical order)}
SECTION_SCHEMA: dict[str, dict] = {
    "atom_types": {"n_atoms": 1, "symmetry": "none", "type": "atom type",
                   "constants": [("Mass", "Da", float), ("El", "", str),
                                 ("connections", "", int)],
                   "text": "Comment"},
    "equivalence": {"n_atoms": 1, "symmetry": "none", "type": "equivalence",
                    "constants": [("NonB", "", str), ("Bond", "", str),
                                  ("Angle", "", str), ("Torsion", "", str),
                                  ("OOP", "", str)]},
    "auto_equivalence": {"n_atoms": 1, "symmetry": "none",
                         "type": "auto equivalence", "constants": None},
    "metadata": {"n_atoms": 1, "symmetry": "none", "type": "metadata",
                 "constants": [("Value", "", _auto)], "text": "Description"},
    "charges": {"n_atoms": 1, "symmetry": "none", "type": "atomic charge",
                "constants": [("Q", "e", float)]},
    "bond_increments": {"n_atoms": 2, "symmetry": "like_bond",
                        "type": "bond charge increment",
                        "constants": [("deltaij", "e", float),
                                      ("deltaji", "e", float)], "flip": 1},
    "nonbond(12-6)": {"n_atoms": 1, "symmetry": "none", "type": "pair",
                      "form": "sigma-eps",
                      "constants": [("sigma", "Å", float),
                                    ("eps", "kcal/mol", float)]},
    "nonbond(9-6)": {"n_atoms": 1, "symmetry": "none", "type": "pair",
                     "form": "rmin-eps",
                     "constants": [("rmin", "Å", float),
                                   ("eps", "kcal/mol", float)]},
    # Buckingham / exponential-6 sections come in two shapes: the classic
    # pairwise one (key I J, columns A rho C) and the per-atom generator form
    # of the SEAMM dreiding.frc (key I, columns rho eps S -- the vdW minimum
    # distance R0, the well depth D0 and the dimensionless scaling parameter
    # zeta of Mayo et al. 1990, eq 32'). ``n_atoms: None`` means "read the key
    # width and the value columns from the section's own header line".
    "buckingham": {"n_atoms": None, "symmetry": "like_bond", "type": "pair",
                   "constants": None},
    # DREIDING per-atom generators: the bond radius R0 and the angle Theta0
    # of the atom as an angle center (Mayo et al. 1990, Table I)
    "dreiding_atomic_parameters": {
        "n_atoms": 1, "symmetry": "none", "type": "dreiding atomic",
        "constants": [("Radius", "Å", float), ("Theta0", "degree", float)]},
    # DREIDING inversions, keyed by the central atom (second column)
    "dreiding_out_of_plane": {
        "n_atoms": 4, "symmetry": "like_oop", "type": "out-of-plane",
        "constants": [("K2", "kcal/mol/radian^2", float),
                      ("Psi0", "degree", float)]},
    "quadratic_bond": {"n_atoms": 2, "symmetry": "like_bond", "type": "bond",
                       "constants": [("R0", "Å", float),
                                     ("K2", "kcal/mol/Å^2", float)]},
    "quartic_bond": {"n_atoms": 2, "symmetry": "like_bond", "type": "bond",
                     "constants": [("R0", "Å", float),
                                   ("K2", "kcal/mol/Å^2", float),
                                   ("K3", "kcal/mol/Å^3", float),
                                   ("K4", "kcal/mol/Å^4", float)]},
    "quadratic_angle": {"n_atoms": 3, "symmetry": "like_angle",
                        "type": "angle",
                        "constants": [("Theta0", "degree", float),
                                      ("K2", "kcal/mol/radian^2", float)]},
    "quartic_angle": {"n_atoms": 3, "symmetry": "like_angle", "type": "angle",
                      "constants": [("Theta0", "degree", float),
                                    ("K2", "kcal/mol/radian^2", float),
                                    ("K3", "kcal/mol/radian^3", float),
                                    ("K4", "kcal/mol/radian^4", float)]},
    "simple_fourier_angle": {"n_atoms": 3, "symmetry": "like_angle",
                             "type": "angle",
                             "constants": [("K", "kcal/mol", float),
                                           ("n", "", int)]},
    "tabulated_angle": {"n_atoms": 3, "symmetry": "like_angle",
                        "type": "angle",
                        "constants": [("Eqn", "", str), ("K", "kcal/mol", float),
                                      ("n", "", int), ("Rb", "Å", float),
                                      ("A", "kcal/mol*Å^12", float),
                                      ("zero-shift", "degree", float)]},
    "torsion_opls": {"n_atoms": 4, "symmetry": "like_torsion",
                     "type": "torsion",
                     "constants": [("V1", "kcal/mol", float),
                                   ("V2", "kcal/mol", float),
                                   ("V3", "kcal/mol", float),
                                   ("V4", "kcal/mol", float)]},
    "torsion_1": {"n_atoms": 4, "symmetry": "like_torsion", "type": "torsion",
                  "constants": [("KPhi", "kcal/mol", float), ("n", "", int),
                                ("Phi0", "degree", float)]},
    "torsion_3": {"n_atoms": 4, "symmetry": "like_torsion", "type": "torsion",
                  "constants": [("V1", "kcal/mol", float),
                                ("Phi0_1", "degree", float),
                                ("V2", "kcal/mol", float),
                                ("Phi0_2", "degree", float),
                                ("V3", "kcal/mol", float),
                                ("Phi0_3", "degree", float)]},
    "improper_opls": {"n_atoms": 4, "symmetry": "like_improper",
                      "type": "out-of-plane",
                      "constants": [("V2", "kcal/mol", float)]},
    "wilson_out_of_plane": {"n_atoms": 4, "symmetry": "like_oop",
                            "type": "out-of-plane",
                            "constants": [("K", "kcal/mol/radian^2", float),
                                          ("Chi0", "degree", float)]},
    # ReaxFF sections: keys are element symbols exactly as written (the
    # library keeps the file's orientation of bond and angle types), value
    # columns are named by the header line
    "reaxff_general_parameters": {"n_atoms": 1, "symmetry": "none",
                                  "type": "reaxff general",
                                  "constants": [("Value", "", float)],
                                  "text": "Description"},
    "reaxff_off-diagonal_parameters": {"n_atoms": 2, "symmetry": "none",
                                       "type": "reaxff off-diagonal",
                                       "constants": None},
    "reaxff_angle_parameters": {"n_atoms": 3, "symmetry": "none",
                                "type": "reaxff angle", "constants": None},
    "reaxff_torsion_parameters": {"n_atoms": 4, "symmetry": "none",
                                  "type": "reaxff torsion", "constants": None},
    "reaxff_hydrogen-bond_parameters": {"n_atoms": 3, "symmetry": "none",
                                        "type": "reaxff hydrogen bond",
                                        "constants": None},
}
for _g in ("1-8", "9-16", "17-24", "25-32"):
    SECTION_SCHEMA[f"reaxff_atomic_parameters_{_g}"] = {
        "n_atoms": 1, "symmetry": "none", "type": "reaxff atomic",
        "constants": None}
for _g in ("1-8", "9-16"):
    SECTION_SCHEMA[f"reaxff_bond_parameters_{_g}"] = {
        "n_atoms": 2, "symmetry": "none", "type": "reaxff bond",
        "constants": None}

JSON_SECTIONS = ("templates", "fragments")
# parameter names of each nonbond @type, the names @units modifiers refer to
_NONBOND_FORM_PARAMS = {"sigma-eps": ("sigma", "eps"), "rmin-eps": ("rmin", "eps"),
                        "eps-rmin": ("eps", "rmin"), "A-B": ("A", "B"),
                        "A/r-B/r": ("A", "B")}
_KEY_HEADER_TOKENS = {"I", "J", "K", "L", "M", "N", "Type", "Center",
                      "Parameter", "Atom"}


def register_section_schema(kind: str, n_atoms: Optional[int], symmetry: str,
                            constants: Optional[Sequence[tuple]],
                            type: str = "", text: Optional[str] = None,
                            flip: int = 0, **extra) -> None:
    """Register (or replace) the schema of a section kind.

    Parameters
    ----------
    kind : str
        The section kind, the word after ``#`` (e.g. ``"quadratic_bond"``).
    n_atoms : int
        Number of leading atom-type key columns after ``Version`` and ``Ref``.
    n_atoms : int or None
        Number of leading atom-type key columns after ``Version`` and ``Ref``;
        ``None`` reads the key width from the section's own header line (for
        kinds such as ``buckingham`` that appear in both pairwise and
        per-atom shapes), in which case ``constants`` should be ``None`` too.
    symmetry : str
        ``"none"``, ``"like_bond"``, ``"like_angle"``, ``"like_torsion"``,
        ``"like_improper"`` (central atom third) or ``"like_oop"`` (central
        atom second); governs the canonical ordering of the key.
    constants : sequence of (name, unit[, ctor]) or None
        The value columns with their default units; ``None`` takes the names
        from the section's header comment and reads floats.
    type : str, optional
        The interaction class (``"bond"``, ``"angle"``, ``"torsion"``,
        ``"out-of-plane"``, ``"pair"``, ...), used by
        :meth:`ForceField.terms`.
    text : str, optional
        Name of a trailing free-text column (the remainder of the row).
    flip : int, optional
        When the key was flipped into canonical order, swap the first ``flip``
        values with the next ``flip`` (bond increments).
    """
    SECTION_SCHEMA[kind] = {"n_atoms": n_atoms, "symmetry": symmetry,
                            "type": type, "constants": (
                                None if constants is None else
                                [tuple(c) if len(c) > 2 else (c[0], c[1], float)
                                 for c in constants]),
                            "text": text, "flip": flip, **extra}


def canonical_key(symmetry: str, atoms: Sequence[str]) -> tuple[tuple, bool]:
    """Order an atom-type key canonically under a symmetry.

    Mirrors the SEAMM conventions so that keys written in either orientation
    in a file resolve to the same entry.

    Parameters
    ----------
    symmetry : str
        See :func:`register_section_schema`.
    atoms : sequence of str
        The key columns as written.

    Returns
    -------
    key : tuple of str
        The canonical key.
    flipped : bool
        Whether the order was changed.
    """
    a = list(atoms)
    n = len(a)
    if n == 2 and symmetry == "like_bond":
        if a[0] > a[1]:
            return (a[1], a[0]), True
        return (a[0], a[1]), False
    if n == 3 and symmetry == "like_angle":
        if a[0] > a[2]:
            return (a[2], a[1], a[0]), True
        return tuple(a), False
    if n == 4:
        i, j, k, l = a
        if symmetry == "like_torsion":
            if j == k and i > l:
                return (l, j, k, i), True
            if j > k:
                return (l, k, j, i), True
            return tuple(a), False
        if symmetry == "like_improper":          # k is the central atom
            i2, j2, l2 = sorted((i, j, l))
            key = (i2, j2, k, l2)
            return key, list(key) != a
        if symmetry == "like_oop":               # j is the central atom
            i2, k2, l2 = sorted((i, k, l))
            key = (i2, j, k2, l2)
            return key, list(key) != a
    return tuple(a), False


# ----------------------------------------------------------------------
# data classes
# ----------------------------------------------------------------------
@dataclass
class Row:
    """One data row of a parameter section.

    Attributes
    ----------
    version : str
        The row's version token (newest wins among rows with the same key).
    reference : str
        The ``#reference`` number the row cites.
    key : tuple of str
        The canonical atom-type key (empty for keyless rows).
    values : dict
        Column name -> value, in the schema's default units.
    source : str
        Path of the file the row came from (references are per file).
    """
    version: str
    reference: str
    key: tuple
    values: dict
    source: str = ""

    @property
    def version_key(self):
        """Comparable version."""
        return parse_version(self.version)


@dataclass
class Section:
    """A ``#<kind> <label>`` section.

    Attributes
    ----------
    kind, label : str
        The section kind and label.
    comments : list of str
        ``!`` lines (without the marker).
    annotations : list of str
        ``>`` lines: the energy expression and notes.
    modifiers : dict
        ``@`` lines grouped by keyword: ``{"units": [["K2", "kJ/mol/nm^2"]],
        "type": [["rmin-eps"]], ...}``.
    key_columns, columns : list of str
        Names of the key columns and of the value columns.
    rows : list of Row
        The data rows (empty for JSON sections).
    data : object
        Parsed JSON for ``templates`` / ``fragments``, else ``None``.
    source : str
        Originating file.
    """
    kind: str
    label: str
    comments: list = field(default_factory=list)
    annotations: list = field(default_factory=list)
    modifiers: dict = field(default_factory=dict)
    key_columns: list = field(default_factory=list)
    columns: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    data: Any = None
    source: str = ""

    @property
    def schema(self) -> Optional[dict]:
        """The registered schema for this kind (``None`` if unknown)."""
        return SECTION_SCHEMA.get(self.kind)

    def units(self, column: str) -> str:
        """The unit a value column is stored in (schema default)."""
        sch = self.schema
        if sch and sch.get("constants"):
            for c in sch["constants"]:
                if c[0] == column:
                    return c[1]
        return ""

    def modifier(self, keyword: str) -> Optional[list]:
        """First ``@keyword`` modifier's arguments, or ``None``."""
        items = self.modifiers.get(keyword)
        return items[0] if items else None

    def latest(self) -> dict:
        """Key -> newest :class:`Row` within this section."""
        out: dict = {}
        for row in self.rows:
            cur = out.get(row.key)
            if cur is None or row.version_key >= cur.version_key:
                out[row.key] = row
        return out


@dataclass
class Define:
    """A ``#define <name>`` section: which labelled sections form a variant.

    Attributes
    ----------
    name : str
        The force-field variant name.
    entries : list of (version, reference, function, labels)
        One per row of the define.
    """
    name: str
    entries: list = field(default_factory=list)
    source: str = ""

    def sections_for(self, version=None) -> dict[str, list[str]]:
        """Functional form -> labels, picking one row per form by version.

        Parameters
        ----------
        version : str, optional
            Use the newest row not above this version; default the newest.

        Returns
        -------
        dict
            Form -> ordered label list (labels keep any ``:optional`` suffix).
        """
        want = parse_version(version) if version is not None else None
        chosen: dict[str, tuple] = {}
        for ver, ref, form, labels in self.entries:
            vk = parse_version(ver)
            if want is not None and vk > want:
                continue
            cur = chosen.get(form)
            if cur is None or vk >= cur[0]:
                chosen[form] = (vk, labels)
        return {form: list(labels) for form, (_, labels) in chosen.items()}


@dataclass
class Reference:
    """A ``#reference <n>`` block."""
    number: str
    text: str
    author: str = ""
    date: str = ""
    source: str = ""


# ----------------------------------------------------------------------
# reading
# ----------------------------------------------------------------------
_RULE = re.compile(r"^!\s*-+[\s-]*$")


def builtin_data_dir() -> Path:
    """Directory of the ``.frc`` files shipped with xnn."""
    return Path(str(resources.files("xnn.ffnn").joinpath("data")))


class FrcFile:
    """A parsed ``.frc`` file (plus everything it ``#include``\\ s).

    Parameters
    ----------
    path : str or Path
        The file to read.
    include_dirs : sequence of str or Path, optional
        Directories searched for ``#include local:...`` targets (after the
        including file's directory); the xnn data directory is always
        searched last.

    Attributes
    ----------
    path : Path
        The top-level file.
    header : str
        The first line, verbatim.
    header_dialect : str
        The word after ``!`` in the header (``"MolSSI"``, or ``"BIOSYM"`` for
        legacy files).
    format_version : int or None
        The format version declared by the header (the trailing number, 1 for
        every current file); ``None`` if the header carries no number.
    sections : dict
        ``(kind, label) -> Section`` for every parameter section read.
    defines : dict
        ``name -> Define``.
    references : dict
        ``(source, number) -> Reference``.
    files : list of Path
        Every file read, in order.
    missing_includes : list of str
        ``missing_ok`` includes that were not found.
    """

    def __init__(self, path: Union[str, Path],
                 include_dirs: Sequence[Union[str, Path]] = ()):
        self.path = Path(path)
        self.include_dirs = [Path(d) for d in include_dirs]
        self.header = ""
        self.header_dialect = ""
        self.format_version: Optional[int] = None
        self.version_lines: list[str] = []
        self.sections: dict[tuple, Section] = {}
        self.defines: dict[str, Define] = {}
        self.references: dict[tuple, Reference] = {}
        self.files: list[Path] = []
        self.missing_includes: list[str] = []
        self._read(self.path, top=True)

    # -- construction from parts (for writers) --
    @classmethod
    def empty(cls, header: str = FRC_HEADER) -> "FrcFile":
        """A blank in-memory file to fill with sections and defines."""
        obj = cls.__new__(cls)
        obj.path = Path("<memory>")
        obj.include_dirs = []
        obj.header = header
        obj.version_lines = []
        obj.sections = {}
        obj.defines = {}
        obj.references = {}
        obj.files = []
        obj.missing_includes = []
        return obj

    @property
    def forcefields(self) -> list[str]:
        """Names of the variants this file defines, in file order."""
        return list(self.defines)

    # -- parsing --
    def _resolve_include(self, target: str, current: Path) -> Optional[Path]:
        """Locate an ``#include`` target, or ``None``."""
        if target.startswith("local:"):
            rel = target[len("local:"):]
            candidates = [d / rel for d in self.include_dirs]
            candidates.append(builtin_data_dir() / rel)
        else:
            rel = target
            candidates = [current.parent / rel]
            candidates += [d / rel for d in self.include_dirs]
            candidates.append(builtin_data_dir() / rel)
        for c in candidates:
            if c.is_file():
                return c
        return None

    def _read(self, path: Path, top: bool = False) -> None:
        """Parse one file into the shared pools (recursing on includes)."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        self.files.append(path)
        lines = path.read_text(encoding="utf-8").splitlines()
        if top:
            self.header = lines[0].strip() if lines else ""
            self.header_dialect, kind, self.format_version = parse_header(self.header)
            if not self.header.startswith("!") or kind != "forcefield":
                logger.warning("%s: expected a '%s' header, got %r",
                               path, FRC_HEADER, self.header)
            elif self.format_version is None:
                logger.warning("%s: header %r carries no format version; "
                               "assuming version %d", path, self.header,
                               FRC_FORMAT_VERSION)
            elif self.format_version > FRC_FORMAT_VERSION:
                logger.warning("%s declares .frc format version %d, newer than "
                               "the version %d this reader implements; parsing "
                               "it as version %d", path, self.format_version,
                               FRC_FORMAT_VERSION, FRC_FORMAT_VERSION)
            else:
                logger.debug("%s: %s forcefield file, format version %d",
                             path, self.header_dialect, self.format_version)
        start = 1 if top else 0
        # split into blocks at '#' lines
        i = start
        n = len(lines)
        # skip anything before the first '#'
        while i < n and not lines[i].lstrip().startswith("#"):
            i += 1
        while i < n:
            head = lines[i].strip()
            words = head[1:].split()
            kind = words[0] if words else ""
            j = i + 1
            while j < n and not lines[j].lstrip().startswith("#"):
                j += 1
            body = lines[i + 1:j]
            i = j
            if kind in ("end", ""):
                continue
            if kind == "version":
                self.version_lines.append(head)
                continue
            if kind == "include":
                target = words[1] if len(words) > 1 else ""
                missing_ok = "missing_ok" in words[2:]
                found = self._resolve_include(target, path)
                if found is None:
                    if missing_ok:
                        self.missing_includes.append(target)
                        continue
                    raise FileNotFoundError(
                        f"{path}: cannot find included force-field file "
                        f"{target!r}")
                self._read(found)
                continue
            label = words[1] if len(words) > 1 else "missing"
            if kind == "define":
                self._parse_define(label, body, path)
            elif kind == "reference":
                self._parse_reference(label, body, path)
            else:
                sec = self._parse_section(kind, label, body, path)
                key = (kind, label)
                if key in self.sections:
                    raise ValueError(f"{path}: section '#{kind} {label}' is "
                                     "defined more than once")
                self.sections[key] = sec

    def _parse_define(self, name: str, body: list[str], path: Path) -> None:
        if name in self.defines:
            raise ValueError(f"{path}: force field {name!r} is defined twice")
        d = Define(name=name, source=str(path))
        for ln in body:
            s = ln.strip()
            if not s or s.startswith(("!", ">", "@")):
                continue
            w = s.split()
            if len(w) < 4:
                logger.warning("%s: short line in #define %s: %r", path, name, s)
                continue
            d.entries.append((w[0], w[1], w[2], w[3:]))
        self.defines[name] = d

    def _parse_reference(self, number: str, body: list[str], path: Path) -> None:
        author = date = ""
        text_lines = []
        for ln in body:
            s = ln.rstrip()
            if s.strip().lower().startswith("@author"):
                author = s.split(None, 1)[1].strip() if len(s.split()) > 1 else ""
            elif s.strip().lower().startswith("@date"):
                date = s.split(None, 1)[1].strip() if len(s.split()) > 1 else ""
            else:
                text_lines.append(s)
        self.references[(str(path), number)] = Reference(
            number=number, text="\n".join(text_lines).strip(), author=author,
            date=date, source=str(path))

    @staticmethod
    def _header_tokens(comments: list[str]) -> list[str]:
        """Column names from the last non-rule comment line."""
        for c in reversed(comments):
            if _RULE.match("!" + c):
                continue
            toks = c.split()
            if not toks:
                continue
            # drop the leading Version / Ref columns (spelled variously)
            while toks and re.fullmatch(r"(Ver(sion)?|Ref)", toks[0], re.I):
                toks.pop(0)
            return toks
        return []

    def _parse_section(self, kind: str, label: str, body: list[str],
                       path: Path) -> Section:
        sec = Section(kind=kind, label=label, source=str(path))
        data_lines: list[str] = []
        for ln in body:
            s = ln.strip()
            if not s:
                continue
            if s.startswith("!"):
                sec.comments.append(s[1:])
            elif s.startswith(">"):
                sec.annotations.append(s[1:].strip())
            elif s.startswith("@") and not s.lower().startswith("@bibtex"):
                w = s[1:].split()
                if w:
                    sec.modifiers.setdefault(w[0], []).append(w[1:])
            else:
                data_lines.append(ln.rstrip())
        if kind in JSON_SECTIONS:
            text = "\n".join(data_lines).strip()
            sec.data = json.loads(text) if text else {}
            return sec

        sch = SECTION_SCHEMA.get(kind)
        header = [t for t in self._header_tokens(sec.comments) if t != "#"]
        if sch is not None and sch["n_atoms"] is not None:
            n_atoms = sch["n_atoms"]
            symmetry = sch["symmetry"]
        else:
            # no schema, or a schema that defers the key width to the file:
            # the leading I/J/K/L-style header tokens are the key columns
            n_atoms = 0
            for tok in header:
                if tok in _KEY_HEADER_TOKENS:
                    n_atoms += 1
                else:
                    break
            symmetry = sch["symmetry"] if sch is not None else "none"
        sec.key_columns = header[:n_atoms] if len(header) >= n_atoms \
            else [f"k{i}" for i in range(n_atoms)]

        # value columns: schema constants, else the header
        constants = sch["constants"] if sch else None
        if constants is None:
            names = header[n_atoms:]
            # collapse header fragments like "# conns" that split on spaces
            constants = [(nm, "", _auto) for nm in names]
            text_col = None
        else:
            text_col = sch.get("text")
        sec.columns = [c[0] for c in constants] + ([text_col] if text_col else [])

        # per-column unit factors from @units against the schema defaults
        factors = {}
        for mod in sec.modifiers.get("units", []):
            if len(mod) >= 2:
                which, unit = mod[0], " ".join(mod[1:])
                for c in constants:
                    if c[0] == which and c[1]:
                        factors[which] = unit_factor(unit, c[1])
        # Nonbond sections come in several forms (@type sigma-eps, rmin-eps,
        # A-B, A/r-B/r) with their own @units, and a force field may merge
        # sections of different forms under one label list -- so every row
        # is normalised to the schema's form and default units here, where
        # the section's modifiers still apply to it. The modifiers are then
        # rewritten to the canonical form so a written file re-reads as is.
        form_kind = bool(sch and sch.get("form"))
        if form_kind:
            mod = sec.modifier("type")
            in_form = mod[0] if mod else sch["form"]
            # @units name the form's own parameters (rmin, eps, A, B), as in
            # SEAMM, whatever the header calls the columns
            in_units = {m[0].lower(): " ".join(m[1:]) for m in
                        sec.modifiers.get("units", []) if len(m) >= 2}
            form_names = _NONBOND_FORM_PARAMS.get(in_form, ("sigma", "eps"))
            raw_names = header[n_atoms:n_atoms + 2]
            if len(raw_names) < 2:
                raw_names = list(form_names)
            out_names = [c[0] for c in constants[:2]]
            factors = {}
            constants = [(nm, "", float) for nm in raw_names]
            sec.columns = list(out_names)

        flip = sch.get("flip", 0) if sch else 0
        for ln in data_lines:
            w = ln.split()
            if len(w) < 2 + n_atoms:
                logger.warning("%s: short row in #%s %s: %r", path, kind, label, ln)
                continue
            version, ref = w[0], w[1]
            key, flipped = canonical_key(symmetry, w[2:2 + n_atoms])
            vals = w[2 + n_atoms:]
            if flipped and flip:
                vals = vals[flip:2 * flip] + vals[:flip] + vals[2 * flip:]
            values: dict = {}
            for idx, c in enumerate(constants):
                if idx >= len(vals):
                    break
                name, unit = c[0], c[1]
                ctor = c[2] if len(c) > 2 else float
                tok = vals[idx]
                try:
                    v = ctor(tok)
                except (TypeError, ValueError):
                    v = _auto(tok)
                if name in factors and isinstance(v, float):
                    v *= factors[name]
                values[name] = v
            if text_col:
                values[text_col] = " ".join(vals[len(constants):])
            if form_kind:
                v = list(values.values())
                u1 = in_units.get(form_names[0].lower()) or \
                    in_units.get(raw_names[0].lower())
                u2 = in_units.get(form_names[1].lower()) or \
                    in_units.get(raw_names[1].lower())
                sig, eps = nonbond_to_sigma_eps(in_form, float(v[0]),
                                                float(v[1]), u1, u2)
                if sch["form"] == "rmin-eps":
                    sig = sig * 2.0 ** (1.0 / 6.0)
                values = {out_names[0]: sig, out_names[1]: eps}
            sec.rows.append(Row(version=version, reference=ref, key=key,
                                values=values, source=str(path)))
        if form_kind:
            sec.modifiers["type"] = [[sch["form"]]]
        if form_kind or factors:
            # the rows are now in the schema's default units, so the file's
            # @units no longer describe them (and would convert them twice
            # if written back out)
            sec.modifiers.pop("units", None)
        return sec

    # -- resolution --
    def forcefield(self, name: Optional[str] = None,
                   version: Optional[str] = None) -> "ForceField":
        """Resolve a ``#define`` into a :class:`ForceField`.

        Parameters
        ----------
        name : str, optional
            The variant to build; default: the only one, or the one named
            like the file, or the first.
        version : str, optional
            Build the force field as of this version (rows newer than it are
            ignored); default the newest.

        Returns
        -------
        ForceField
            The merged, ready-to-query force field.

        Raises
        ------
        KeyError
            If ``name`` is not defined in the file.
        """
        if not self.defines:
            raise ValueError(f"{self.path} has no #define section")
        if name is None:
            if len(self.defines) == 1:
                name = next(iter(self.defines))
            elif self.path.stem in self.defines:
                name = self.path.stem
            else:
                name = next(iter(self.defines))
        if name not in self.defines:
            raise KeyError(f"{self.path} does not define {name!r}; available: "
                           f"{self.forcefields}")
        ff = ForceField(name=name, file=self, version=version)
        want = parse_version(version) if version is not None else None
        for form, labels in self.defines[name].sections_for(version).items():
            merged: dict = {}
            used: list[str] = []
            modifiers: dict = {}
            annotations: list[str] = []
            for label in labels:
                optional = label.endswith(":optional")
                lab = label[:-9] if optional else label
                sec = self.sections.get((form, lab))
                if sec is None:
                    if optional:
                        continue
                    raise KeyError(f"{self.path}: force field {name!r} needs "
                                   f"section '#{form} {lab}', which is missing")
                used.append(lab)
                for k, v in sec.modifiers.items():
                    modifiers[k] = v
                annotations = annotations or list(sec.annotations)
                if sec.kind in JSON_SECTIONS:
                    latest = _latest_json(sec.data, want)
                    merged.update(latest)
                    continue
                for key, row in _rows_as_of(sec, want).items():
                    merged[key] = row
            ff.sections[form] = merged
            ff.labels[form] = used
            ff.modifiers[form] = modifiers
            ff.annotations[form] = annotations
            ff.columns[form] = self._columns_for(form, used)
        ff._finish()
        return ff

    def _columns_for(self, form: str, labels: list[str]) -> list[str]:
        for lab in labels:
            sec = self.sections.get((form, lab))
            if sec is not None and sec.columns:
                return list(sec.columns)
        return []

    # -- writing --
    def write(self, path: Union[str, Path]) -> Path:
        """Write the file in ``.frc`` syntax (see :func:`write_frc`)."""
        return write_frc(self, path)


def _rows_as_of(sec: Section, want) -> dict:
    """Key -> newest row of ``sec`` not newer than ``want``."""
    out: dict = {}
    for row in sec.rows:
        vk = row.version_key
        if want is not None and vk > want:
            continue
        cur = out.get(row.key)
        if cur is None or vk >= cur.version_key:
            out[row.key] = row
    return out


def _latest_json(data: dict, want) -> dict:
    """Newest version entry per item of a templates/fragments JSON blob.

    The JSON is ``{item: {version: entry}}``; the returned dict is
    ``{item: entry}`` with the entry's version stored under ``"version"``.
    """
    out: dict = {}
    for item, versions in (data or {}).items():
        best = None
        for ver, entry in versions.items():
            vk = parse_version(ver)
            if want is not None and vk > want:
                continue
            if best is None or vk >= best[0]:
                best = (vk, ver, entry)
        if best is not None:
            e = dict(best[2])
            e.setdefault("version", best[1])
            out[item] = e
    return out


# ----------------------------------------------------------------------
# resolved force field
# ----------------------------------------------------------------------
class ForceField:
    """One resolved force-field variant: merged sections plus lookups.

    Built by :meth:`FrcFile.forcefield`; not constructed directly.

    Attributes
    ----------
    name : str
        The variant name (the ``#define``).
    file : FrcFile
        The file it was resolved from.
    sections : dict
        ``kind -> {key: Row}`` for every parameter section of the variant.
    labels, modifiers, annotations, columns : dict
        Per kind: the labels merged, the ``@`` modifiers, the ``>``
        annotations, and the value-column names.
    templates : dict
        ``atom type -> {"smarts": [...], "description": str, "overrides":
        [...], "version": str}`` in file order (later templates take
        precedence when typing).
    fragments : dict
        ``fragment name -> entry`` (``SMARTS``, ``atom types``, optional
        ``charges``).
    metadata : dict
        The ``#metadata`` parameters (``ff_form``, ``charges``, ...).
    """

    def __init__(self, name: str, file: FrcFile, version: Optional[str]):
        self.name = name
        self.file = file
        self.version = version
        self.sections: dict[str, dict] = {}
        self.labels: dict[str, list] = {}
        self.modifiers: dict[str, dict] = {}
        self.annotations: dict[str, list] = {}
        self.columns: dict[str, list] = {}
        self.templates: dict = {}
        self.fragments: dict = {}
        self.metadata: dict = {}
        self._equiv: dict = {}

    def _finish(self) -> None:
        self.templates = dict(self.sections.pop("templates", {}))
        self.fragments = dict(self.sections.pop("fragments", {}))
        meta = self.sections.get("metadata", {})
        self.metadata = {k[0]: r.values.get("Value") for k, r in meta.items()}
        eq = self.sections.get("equivalence", {})
        self._equiv = {k[0]: r.values for k, r in eq.items()}

    # -- introspection --
    @property
    def ff_form(self) -> str:
        """The ``ff_form`` metadata entry (``"oplsaa"``, ``"reaxff"``, ...)."""
        return str(self.metadata.get("ff_form", ""))

    @property
    def terms(self) -> dict[str, list[str]]:
        """Interaction class -> section kinds present (``"bond"`` -> ...)."""
        out: dict[str, list[str]] = {}
        for kind in self.sections:
            sch = SECTION_SCHEMA.get(kind)
            t = sch.get("type", "") if sch else ""
            if t:
                out.setdefault(t, []).append(kind)
        return out

    def kinds_of(self, term: str) -> list[str]:
        """Section kinds implementing interaction class ``term``."""
        return self.terms.get(term, [])

    @property
    def atom_types(self) -> dict[str, dict]:
        """``type -> {"Mass", "El", "connections", "Comment"}``."""
        return {k[0]: r.values for k, r in self.sections.get("atom_types", {}).items()}

    def rows(self, kind: str) -> dict:
        """``key -> Row`` of a section kind (empty if absent)."""
        return self.sections.get(kind, {})

    def references_used(self) -> list[Reference]:
        """The ``#reference`` blocks cited by the merged rows, deduplicated."""
        seen = set()
        out = []
        for kind, rows in self.sections.items():
            for row in rows.values():
                key = (row.source, row.reference)
                if key in seen:
                    continue
                seen.add(key)
                ref = self.file.references.get(key)
                if ref is not None:
                    out.append(ref)
        return out

    # -- equivalences --
    def equivalent(self, atom_type: str, term: str) -> str:
        """The equivalent type used for ``term`` (``"nonbond"``, ``"bond"``,
        ``"angle"``, ``"torsion"``, ``"oop"``); the type itself if none."""
        col = {"nonbond": "NonB", "bond": "Bond", "angle": "Angle",
               "torsion": "Torsion", "oop": "OOP"}[term]
        e = self._equiv.get(atom_type)
        return str(e[col]) if e and col in e else atom_type

    # -- per-atom lookups --
    def charge(self, atom_type: str) -> float:
        """Partial charge of an atom type (direct, then NonB equivalent, else 0)."""
        rows = self.sections.get("charges", {})
        for t in (atom_type, self.equivalent(atom_type, "nonbond")):
            r = rows.get((t,))
            if r is not None:
                return float(r.values["Q"])
        return 0.0

    def nonbond(self, atom_type: str, kind: str = "nonbond(12-6)"
                ) -> Optional[tuple[float, float]]:
        """Lennard-Jones parameters of an atom type, ``(sigma [Å], epsilon
        [kcal/mol])`` for ``nonbond(12-6)`` (``(rmin, epsilon)`` for
        ``nonbond(9-6)``).

        Every ``@type`` / ``@units`` variant a section may use was already
        normalised to this form when the file was read. Looks up the type,
        then its nonbond equivalent, then a ``*`` wildcard row; ``None`` if
        nothing matches.
        """
        rows = self.sections.get(kind, {})
        row = None
        for t in (atom_type, self.equivalent(atom_type, "nonbond")):
            row = rows.get((t,))
            if row is not None:
                break
        if row is None:
            row = rows.get((WILDCARD,))
        if row is None:
            return None
        vals = list(row.values.values())
        if len(vals) < 2:
            return None
        return float(vals[0]), float(vals[1])

    def combination(self, kind: str = "nonbond(12-6)") -> str:
        """The ``@combination`` rule of a nonbond section (default geometric)."""
        m = self.modifiers.get(kind, {}).get("combination")
        return m[0][0] if m else "geometric"

    # -- bonded lookups (direct key, then equivalences, then wildcards) --
    def _lookup(self, term: str, symmetry: str, atoms: Sequence[str],
                patterns) -> Optional[tuple[str, tuple, Row]]:
        kinds = self.kinds_of(term)
        eq = tuple(self.equivalent(a, term if term != "out-of-plane" else "oop")
                   for a in atoms)
        for cand in (tuple(atoms), eq):
            for kind in kinds:
                rows = self.sections.get(kind, {})
                for pat in patterns(cand):
                    key, _ = canonical_key(symmetry, pat)
                    if key in rows:
                        return kind, key, rows[key]
        return None

    def bond(self, i: str, j: str):
        """``(kind, key, Row)`` for a bond between types ``i`` and ``j``, or ``None``."""
        return self._lookup("bond", "like_bond", (i, j), lambda a: [a])

    def angle(self, i: str, j: str, k: str):
        """``(kind, key, Row)`` for an angle (``j`` central), or ``None``."""
        return self._lookup("angle", "like_angle", (i, j, k),
                            lambda a: [a, (WILDCARD, a[1], WILDCARD)])

    def torsion(self, i: str, j: str, k: str, l: str):
        """``(kind, key, Row)`` for a proper torsion, wildcards in SEAMM order."""
        def pats(a):
            return [a, (WILDCARD, a[1], a[2], a[3]), (a[0], a[1], a[2], WILDCARD),
                    (WILDCARD, a[1], a[2], WILDCARD)]
        return self._lookup("torsion", "like_torsion", (i, j, k, l), pats)

    def improper(self, i: str, j: str, k: str, l: str):
        """``(kind, key, Row)`` for an improper with ``k`` central, or ``None``.

        The wildcard order follows SEAMM: exact; one outer wildcard (each
        position); two outer wildcards; all outer wildcards.
        """
        X = WILDCARD

        def pats(a):
            i_, j_, k_, l_ = a
            return [(i_, j_, k_, l_), (X, j_, k_, l_), (i_, X, k_, l_),
                    (i_, j_, k_, X), (X, X, k_, l_), (X, j_, k_, X),
                    (i_, X, k_, X), (X, X, k_, X)]
        return self._lookup("out-of-plane", "like_improper", (i, j, k, l), pats)

    def __repr__(self) -> str:
        return (f"ForceField({self.name!r}, file={self.file.path.name!r}, "
                f"sections={sorted(self.sections)}, "
                f"templates={len(self.templates)})")


def nonbond_to_sigma_eps(form: str, v1: float, v2: float,
                         unit1: Optional[str] = None,
                         unit2: Optional[str] = None) -> tuple[float, float]:
    """Convert a nonbond row to ``(sigma [Å], epsilon [kcal/mol])``.

    Parameters
    ----------
    form : str
        ``"sigma-eps"``, ``"rmin-eps"``, ``"eps-rmin"``, ``"A-B"`` or
        ``"A/r-B/r"`` (the ``@type`` of the section).
    v1, v2 : float
        The two value columns as written.
    unit1, unit2 : str, optional
        Their units when not the defaults (Å and kcal/mol for the sigma/rmin
        forms; ``kcal/mol*Å^12`` / ``kcal/mol*Å^6`` for ``A-B``).

    Returns
    -------
    tuple of float
        ``(sigma, epsilon)``.
    """
    two_sixth = 2.0 ** (1.0 / 6.0)
    if form == "sigma-eps":
        s = convert_units(v1, unit1 or "Å", "Å")
        e = convert_units(v2, unit2 or "kcal/mol", "kcal/mol")
        return s, e
    if form in ("rmin-eps", "eps-rmin"):
        if form == "eps-rmin":
            v1, v2, unit1, unit2 = v2, v1, unit2, unit1
        r = convert_units(v1, unit1 or "Å", "Å")
        e = convert_units(v2, unit2 or "kcal/mol", "kcal/mol")
        return r / two_sixth, e
    if form in ("A-B", "A/r-B/r"):
        if form == "A/r-B/r":
            # A/r and B/r carry units of (E)^(1/12)*L and (E)^(1/6)*L
            u1 = unit1 or "(kcal/mol)**(1/12)*Å"
            u2 = unit2 or "(kcal/mol)**(1/6)*Å"
            a = convert_units(v1, u1, "(kcal/mol)**(1/12)*Å") ** 12
            b = convert_units(v2, u2, "(kcal/mol)**(1/6)*Å") ** 6
        else:
            a = convert_units(v1, unit1 or "kcal/mol*Å^12", "kcal/mol*Å^12")
            b = convert_units(v2, unit2 or "kcal/mol*Å^6", "kcal/mol*Å^6")
        if a == 0.0 or b == 0.0:
            return 0.0, 0.0
        return (a / b) ** (1.0 / 6.0), b * b / (4.0 * a)
    raise ValueError(f"unknown nonbond form {form!r}")


# ----------------------------------------------------------------------
# writing
# ----------------------------------------------------------------------
def _fmt(v) -> str:
    """Format a cell: floats in their shortest round-trippable form."""
    if isinstance(v, float):
        return repr(float(v))
    return str(v)


def write_frc(frc: FrcFile, path: Union[str, Path]) -> Path:
    """Write a :class:`FrcFile` to disk in ``.frc`` syntax.

    Sections are written with aligned columns and a header comment naming
    them; JSON sections are pretty-printed; ``#define`` and ``#reference``
    blocks are reproduced. The output re-reads with :func:`read_frc`.

    Parameters
    ----------
    frc : FrcFile
        The file object (built by hand with :meth:`FrcFile.empty` or read).
    path : str or Path
        Destination.

    Returns
    -------
    Path
        The written path.
    """
    out: list[str] = [frc.header or FRC_HEADER, ""]
    out += list(frc.version_lines)
    for d in frc.defines.values():
        out += [f"#define {d.name}", "",
                "!Version      Ref  Section                          Label",
                "!---------  -----  -------------------------------  --------------"]
        for ver, ref, form, labels in d.entries:
            out.append(f"{ver:<11} {ref:>4}  {form:<32} {' '.join(labels)}")
        out += ["#end", ""]
    for (kind, label), sec in frc.sections.items():
        out += [f"#{kind} {label}", ""]
        for a in sec.annotations:
            out.append(f"> {a}")
        if sec.annotations:
            out.append("")
        for kw, items in sec.modifiers.items():
            for it in items:
                out.append(f"@{kw} {' '.join(it)}")
        if sec.modifiers:
            out.append("")
        if kind in JSON_SECTIONS:
            out += [json.dumps(sec.data, indent=4), ""]
            continue
        keycols = sec.key_columns or [f"k{i}" for i in range(
            len(sec.rows[0].key) if sec.rows else 0)]
        cols = sec.columns or (list(sec.rows[0].values) if sec.rows else [])
        table = []
        for r in sec.rows:
            table.append([r.version, r.reference, *r.key,
                          *[_fmt(r.values.get(c, "")) for c in cols]])
        header = ["Version", "Ref", *keycols, *cols]
        widths = [max(len(str(x)) for x in col)
                  for col in zip(header, *table)] if table else \
            [len(h) for h in header]
        out.append("!" + "  ".join(f"{h:<{w}}" for h, w in zip(header, widths)).rstrip())
        out.append("!" + "  ".join("-" * w for w in widths))
        for row in table:
            out.append(" " + "  ".join(f"{str(x):<{w}}" for x, w in zip(row, widths)).rstrip())
        out.append("")
    for ref in frc.references.values():
        out += [f"#reference {ref.number}"]
        if ref.author:
            out.append(f"@Author {ref.author}")
        if ref.date:
            out.append(f"@Date {ref.date}")
        out += [ref.text, ""]
    out.append("#end")
    p = Path(path)
    p.write_text("\n".join(out) + "\n", encoding="utf-8")
    return p


def make_section(kind: str, label: str, key_columns: Sequence[str],
                 columns: Sequence[str], rows: Iterable[tuple],
                 version: str = "1.0", reference: str = "1",
                 annotations: Sequence[str] = (),
                 modifiers: Optional[dict] = None) -> Section:
    """Build a :class:`Section` from ``(key_tuple, values_dict)`` rows.

    A convenience for writers: keys are canonicalised under the kind's
    schema symmetry and every row gets the given version and reference.
    """
    sch = SECTION_SCHEMA.get(kind)
    sym = sch["symmetry"] if sch else "none"
    sec = Section(kind=kind, label=label, key_columns=list(key_columns),
                  columns=list(columns), annotations=list(annotations),
                  modifiers=dict(modifiers or {}))
    for key, values in rows:
        ck, _ = canonical_key(sym, key)
        sec.rows.append(Row(version=version, reference=reference, key=ck,
                            values=dict(values)))
    return sec


# ----------------------------------------------------------------------
# registry of shipped files and the spec syntax
# ----------------------------------------------------------------------
_DEFINE_RE = re.compile(r"^#define\s+(\S+)", re.M)


def list_forcefields(include_dirs: Sequence[Union[str, Path]] = ()
                     ) -> dict[str, Path]:
    """Every force-field variant available by name.

    Scans the ``.frc`` files shipped with xnn (and any ``include_dirs``)
    for their ``#define`` names.

    Returns
    -------
    dict
        ``variant name -> file``; ReaxFF fields are named
        ``reaxff/<field>`` as in their files, and are also reachable by the
        bare ``<field>`` (see :func:`find_forcefield`).
    """
    out: dict[str, Path] = {}
    dirs = [Path(d) for d in include_dirs] + [builtin_data_dir()]
    for d in dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.rglob("*.frc")):
            try:
                text = f.read_text(encoding="utf-8")
            except OSError:
                continue
            for name in _DEFINE_RE.findall(text):
                out.setdefault(name, f)
    return out


def find_forcefield(spec: str, include_dirs: Sequence[Union[str, Path]] = ()
                    ) -> tuple[Path, Optional[str]]:
    """Resolve a force-field spec to ``(file, variant)``.

    Parameters
    ----------
    spec : str
        One of: a variant name shipped with xnn (``"oplsaa"``, ``"CL&P"``,
        ``"lopls"``, ``"reaxff/CHO_cho_2008"`` or just ``"CHO_cho_2008"``);
        a path to a ``.frc`` file; ``"<path>.frc:<variant>"`` to pick one of
        several variants in a file.
    include_dirs : sequence of path, optional
        Extra directories searched for files and variants.

    Returns
    -------
    tuple
        ``(path, variant or None)``.

    Raises
    ------
    FileNotFoundError
        If nothing matches.
    """
    s = str(spec)
    variant = None
    if ".frc" in s:
        head, _, tail = s.rpartition(".frc")
        candidate = head + ".frc"
        if tail.startswith(":"):
            variant = tail[1:] or None
        elif tail:
            candidate = s
        p = Path(candidate)
        for base in [Path("."), *[Path(d) for d in include_dirs],
                     builtin_data_dir()]:
            q = p if p.is_absolute() else base / p
            if q.is_file():
                return q, variant
        raise FileNotFoundError(f"force-field file {candidate!r} not found")
    known = list_forcefields(include_dirs)
    if s in known:
        return known[s], s
    # bare ReaxFF field name, file stems, case-insensitive fallbacks
    for name, path in known.items():
        if name.rsplit("/", 1)[-1] == s or path.stem == s:
            return path, name
    low = s.lower()
    for name, path in known.items():
        if name.lower() == low or name.rsplit("/", 1)[-1].lower() == low \
                or path.stem.lower() == low:
            return path, name
    raise FileNotFoundError(
        f"unknown force field {spec!r}; shipped: {sorted(known)}; or give a "
        ".frc path (optionally '<path>.frc:<variant>')")


def read_frc(path: Union[str, Path],
             include_dirs: Sequence[Union[str, Path]] = ()) -> FrcFile:
    """Parse a ``.frc`` file (and its includes) into a :class:`FrcFile`."""
    return FrcFile(path, include_dirs=include_dirs)


def read_forcefield(spec: str, version: Optional[str] = None,
                    include_dirs: Sequence[Union[str, Path]] = ()
                    ) -> ForceField:
    """Resolve a spec (see :func:`find_forcefield`) straight to a
    :class:`ForceField`."""
    path, variant = find_forcefield(spec, include_dirs)
    return FrcFile(path, include_dirs=include_dirs).forcefield(variant, version)
