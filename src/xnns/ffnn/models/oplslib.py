"""OPLS parameter libraries: native JSON, GROMACS ``.itp`` and built-in sets.

An OPLS model (Jorgensen, Maxwell & Tirado-Rives, *J. Am. Chem. Soc.* 118,
11225, 1996) is fully specified by a parameter library: per-atom-type
nonbonded parameters (partial charge, Lennard-Jones sigma / epsilon) plus
bonded parameters keyed by *atom class* (the bonded type, e.g. ``CT`` for an
sp3 carbon) -- harmonic bonds and angles, Fourier proper dihedrals and
``V2``-only improper dihedrals. Three sources are supported:

* the **native JSON** format of this module (:func:`read_opls` /
  :meth:`OPLSLibrary.save`), a direct dump of :class:`OPLSLibrary` in OPLS
  units -- kcal/mol, Angstrom, degrees, Fourier coefficients;
* **GROMACS** ``oplsaa.ff``-style ``.itp`` files (``ffnonbonded.itp`` /
  ``ffbonded.itp`` / ``forcefield.itp``), the most common distribution of
  OPLS parameters. The reader translates at load time -- kJ/mol to kcal/mol,
  nm to Angstrom, Ryckaert-Bellemans torsion coefficients back to the OPLS
  Fourier form -- so that everything downstream sees one canonical
  convention;
* **built-in curated subsets** (:func:`builtin_library`): ``"oplsaa"``, the
  published OPLS-AA parameters for alkanes, alkenes, benzene rings and
  monoalcohols (Jorgensen et al. 1996 and later revisions from the Jorgensen
  lab, as tabulated in the standard OPLS-AA distribution), and ``"lopls"``,
  the L-OPLS reparameterization for long hydrocarbons (Siu, Pluhackova &
  Boeckmann, *J. Chem. Theory Comput.* 8, 1459, 2012, Table 2).

Energies are converted from kcal/mol to eV only when the model assembles its
parameter tensors, mirroring :mod:`xnns.ffnn.models.ffield`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

# kJ/mol -> kcal/mol (thermochemical calorie), the GROMACS unit translation.
KCAL_PER_KJ = 1.0 / 4.184
# kcal/mol -> eV as 4.184 (exact) / 96.48533212331 (CODATA 2018 kJ/mol per
# eV). OPLS parameters are defined in the thermochemical-kcal / kJ ecosystem
# (BOSS, GROMACS, OpenMM), so this choice makes xnns energies agree with
# those codes to their own constant precision. (ReaxFF keeps its historical
# Fortran-era constant in reaxff.py for ffield compatibility; the two differ
# by 8e-6 relative.)
KCAL_TO_EV = 4.336410424180094e-2

# Atomic masses (u) for the common organic elements, used to infer the
# element of a GROMACS atom type when the file omits the atomic number.
_MASS_TO_Z = ((1.008, 1), (12.011, 6), (14.007, 7), (15.999, 8), (18.998, 9),
              (28.086, 14), (30.974, 15), (32.06, 16), (35.45, 17),
              (79.904, 35), (126.9, 53))


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
    parameters are keyed by atom *class* strings joined with ``-``
    (``"CT-CT"``, ``"CT-CT-HC"``, ``"CT-CT-CT-CT"``); dihedral keys may use
    the wildcard class ``X`` (``"X-CM-CM-X"``).

    Parameters and attributes
    -------------------------
    atom_types : dict[str, dict]
        ``name -> {"cls", "element", "mass", "charge", "sigma", "epsilon",
        "comment"}`` where ``cls`` is the bonded class, ``element`` the
        atomic number and ``sigma`` / ``epsilon`` the Lennard-Jones
        parameters.
    bond_types : dict[str, dict]
        ``"A-B" -> {"k", "r0"}`` for ``E = k (r - r0)^2`` (note: OPLS/AMBER
        convention, *without* the 1/2).
    angle_types : dict[str, dict]
        ``"A-B-C" -> {"k", "theta0"}`` for ``E = k (theta - theta0)^2`` with
        ``theta0`` in degrees.
    dihedral_types : dict[str, dict]
        ``"A-B-C-D" -> {"v": [V0, V1, V2, V3, V4]}`` Fourier coefficients
        (see :func:`fourier_to_rb` for the energy expression).
    improper_types : dict[str, dict]
        ``key -> {"v2"}`` for the improper energy ``V2/2 (1 - cos 2 phi)``;
        keys are opaque strings referenced by
        :attr:`~xnns.ffnn.models.topology.MolecularTopology.improper_keys`
        (the GROMACS-style names ``"O-C-X-Y"``, ``"Z-CM-X-Y"``, ... in the
        built-in sets).
    fudge_lj, fudge_qq : float
        Scaling factors for 1,4 Lennard-Jones and Coulomb interactions
        (0.5 and 0.5 for OPLS).
    name : str
        A short label for the parameter set.
    references : list[str]
        Literature provenance of the parameters.
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
                "improper_types": self.improper_types}
        Path(path).write_text(json.dumps(data, indent=1, sort_keys=True)
                              + "\n")


# ----------------------------------------------------------------------
# bonded-type resolution
# ----------------------------------------------------------------------
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
            if all(p == "X" or p == q for p, q in zip(pattern, cand)):
                score = sum(p != "X" for p in pattern)
                if score > best_score or (score == best_score
                                          and key < best_key):
                    best_key, best_score = key, score
    return best_key


# ----------------------------------------------------------------------
# readers
# ----------------------------------------------------------------------
def _read_json(path: Path) -> OPLSLibrary:
    """Read the native JSON library format.

    Parameters
    ----------
    path : Path
        The JSON file.

    Returns
    -------
    OPLSLibrary
        The parsed library.
    """
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
        references=list(data.get("references", [])))


def _element_from_mass(mass: float) -> int:
    """Guess the atomic number from an atomic mass.

    Parameters
    ----------
    mass : float
        Atomic mass in u.

    Returns
    -------
    int
        The atomic number of the closest tabulated element (0 for a
        massless virtual site).
    """
    if mass < 0.5:
        return 0
    return min(_MASS_TO_Z, key=lambda mz: abs(mz[0] - mass))[1]


def _itp_lines(paths: Iterable[Path]):
    """Iterate over the meaningful lines of GROMACS ``.itp`` files.

    Comments are stripped; ``[ section ]`` headers switch the section;
    ``#define improper_*`` macros are passed through as pseudo-lines in the
    ``"improper_defines"`` section.

    Parameters
    ----------
    paths : iterable of Path
        The files to read, in order.

    Yields
    ------
    tuple[str, list[str]]
        ``(section, tokens)`` per content line.
    """
    section = ""
    for path in paths:
        for raw in path.read_text().splitlines():
            line = raw.split(";")[0].strip()
            if not line:
                continue
            if line.startswith("["):
                section = line.strip("[] ").strip().lower()
                continue
            if line.startswith("#define") and "improper_" in line:
                yield "improper_defines", line.split()
                continue
            if line.startswith("#"):
                continue
            yield section, line.split()


def read_gromacs_opls(source: Union[str, Path, Sequence[Union[str, Path]]]
                      ) -> OPLSLibrary:
    """Read OPLS parameters from GROMACS ``oplsaa.ff``-style ``.itp`` files.

    Accepts a force-field directory (reads ``forcefield.itp``,
    ``ffnonbonded.itp`` and ``ffbonded.itp`` from it, each optional) or an
    explicit list of ``.itp`` files. Units and functional forms are
    translated to the OPLS conventions of :class:`OPLSLibrary`: kJ/mol to
    kcal/mol, nm to Angstrom, GROMACS ``k/2 (r-r0)^2`` harmonics to the OPLS
    ``k (r-r0)^2`` convention, Ryckaert-Bellemans (function 3) dihedrals to
    Fourier coefficients, and the ``#define improper_*`` periodic macros to
    ``V2`` improper types. Duplicate type keys keep the first definition.

    Parameters
    ----------
    source : str, Path or sequence thereof
        Force-field directory or ``.itp`` file(s).

    Returns
    -------
    OPLSLibrary
        The translated library.

    Raises
    ------
    ValueError
        If the ``[ defaults ]`` section declares a non-geometric combination
        rule (the library would not be an OPLS force field), or an improper
        macro is not a multiplicity-2, 180-degree-phase torsion.
    FileNotFoundError
        If no input file exists.
    """
    if isinstance(source, (str, Path)):
        src = Path(source)
        if src.is_dir():
            paths = [src / n for n in ("forcefield.itp", "ffnonbonded.itp",
                                       "ffbonded.itp")]
            paths = [p for p in paths if p.exists()]
        else:
            paths = [src]
    else:
        paths = [Path(p) for p in source]
    if not paths:
        raise FileNotFoundError(f"no .itp files found in {source}")
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(str(p))

    lib = OPLSLibrary(name="gromacs-opls",
                      references=["translated from GROMACS .itp files"])
    for section, parts in _itp_lines(paths):
        if section == "defaults" and len(parts) >= 2:
            comb = parts[1]
            if comb != "3":
                raise ValueError(
                    f"combination rule {comb} is not the geometric rule "
                    "(3) of OPLS; refusing to translate this force field")
            if len(parts) >= 5:
                lib.fudge_lj = float(parts[3])
                lib.fudge_qq = float(parts[4])
        elif section == "atomtypes" and len(parts) >= 6:
            # `name [class] [at.num] mass charge ptype sigma epsilon`;
            # the particle-type column anchors the layout.
            name = parts[0]
            if name in lib.atom_types:
                continue
            ptype = [i for i in range(3, len(parts) - 2)
                     if parts[i] in ("A", "S", "V", "D")]
            if not ptype:
                continue
            ip = ptype[-1]
            mass, charge = float(parts[ip - 2]), float(parts[ip - 1])
            sigma, eps = float(parts[ip + 1]), float(parts[ip + 2])
            cls, z = name, None
            for tok in parts[1:ip - 2]:
                if _is_number(tok):
                    z = int(float(tok))
                else:
                    cls = tok
            lib.atom_types[name] = {
                "cls": cls, "element": z if z is not None
                else _element_from_mass(mass),
                "mass": mass, "charge": charge, "sigma": sigma * 10.0,
                "epsilon": eps * KCAL_PER_KJ, "comment": ""}
        elif section == "bondtypes" and len(parts) >= 5:
            a, b, func, b0, kb = parts[:5]
            key = f"{a}-{b}"
            if func == "1" and resolve_bond_type(lib.bond_types, a, b) is None:
                lib.bond_types[key] = {
                    "k": 0.5 * float(kb) * KCAL_PER_KJ / 100.0,
                    "r0": float(b0) * 10.0}
        elif section == "angletypes" and len(parts) >= 6:
            a, b, c, func, th0, k = parts[:6]
            key = f"{a}-{b}-{c}"
            if func == "1" and resolve_angle_type(lib.angle_types,
                                                  a, b, c) is None:
                lib.angle_types[key] = {"k": 0.5 * float(k) * KCAL_PER_KJ,
                                        "theta0": float(th0)}
        elif section == "dihedraltypes" and len(parts) >= 11:
            a, b, c, d, func = parts[:5]
            if func != "3":
                continue
            key = f"{a}-{b}-{c}-{d}"
            if key in lib.dihedral_types or f"{d}-{c}-{b}-{a}" \
                    in lib.dihedral_types:
                continue
            cs = [float(x) * KCAL_PER_KJ for x in parts[5:11]]
            lib.dihedral_types[key] = {"v": rb_to_fourier(cs)}
        elif section == "improper_defines" and len(parts) >= 5:
            # `#define improper_O_C_X_Y  180.0  43.932  2`
            name = parts[1].removeprefix("improper_").replace("_", "-")
            phase, k, mult = float(parts[2]), float(parts[3]), int(parts[4])
            if mult != 2 or abs(phase - 180.0) > 1.0e-6:
                raise ValueError(f"improper macro {parts[1]} is not the "
                                 "OPLS V2 form (multiplicity 2, phase 180)")
            if name not in lib.improper_types:
                # k (1 + cos(2 phi - 180)) == (2k)/2 (1 - cos 2 phi)
                lib.improper_types[name] = {"v2": 2.0 * k * KCAL_PER_KJ}
    return lib


def _is_number(token: str) -> bool:
    """Whether ``token`` parses as a float.

    Parameters
    ----------
    token : str
        The token.

    Returns
    -------
    bool
        ``True`` if ``float(token)`` succeeds.
    """
    try:
        float(token)
        return True
    except ValueError:
        return False


def read_opls(source: Union[str, Path, Sequence[Union[str, Path]]]
              ) -> OPLSLibrary:
    """Read an OPLS parameter library from disk.

    Dispatches on the input: a ``.json`` file is read as the native format,
    anything else (a GROMACS force-field directory or ``.itp`` file(s)) goes
    through :func:`read_gromacs_opls`.

    Parameters
    ----------
    source : str, Path or sequence thereof
        Library file, force-field directory, or list of ``.itp`` files.

    Returns
    -------
    OPLSLibrary
        The parsed library.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() == ".json":
            return _read_json(path)
    return read_gromacs_opls(source)


# ----------------------------------------------------------------------
# built-in parameter sets
# ----------------------------------------------------------------------
# OPLS-AA subset for alkanes, alkenes, benzene rings and monoalcohols.
# Values are the published OPLS-AA parameters (Jorgensen, Maxwell &
# Tirado-Rives, JACS 118, 11225, 1996, and later revisions from the
# Jorgensen lab), in kcal/mol / Angstrom / degrees, exactly as tabulated in
# the standard OPLS-AA distribution. Atom-type numbering follows the
# original ffoplsaa convention.
#   name: (class, Z, mass, charge/e, sigma/A, epsilon/kcal, comment)
_OPLSAA_ATOM_TYPES = {
    "opls_135": ("CT", 6, 12.011, -0.18, 3.5, 0.066, "methyl C, alkanes"),
    "opls_136": ("CT", 6, 12.011, -0.12, 3.5, 0.066, "methylene C, alkanes"),
    "opls_137": ("CT", 6, 12.011, -0.06, 3.5, 0.066, "methine C, alkanes"),
    "opls_138": ("CT", 6, 12.011, -0.24, 3.5, 0.066, "methane C"),
    "opls_139": ("CT", 6, 12.011, 0.0, 3.5, 0.066, "quaternary C, alkanes"),
    "opls_140": ("HC", 1, 1.008, 0.06, 2.5, 0.03, "H on alkane C"),
    "opls_141": ("CM", 6, 12.011, 0.0, 3.55, 0.076,
                 "disubstituted sp2 C, alkenes"),
    "opls_142": ("CM", 6, 12.011, -0.115, 3.55, 0.076,
                 "monosubstituted sp2 C, alkenes"),
    "opls_143": ("CM", 6, 12.011, -0.23, 3.55, 0.076,
                 "terminal =CH2 C, alkenes"),
    "opls_144": ("HC", 1, 1.008, 0.115, 2.42, 0.03, "H on alkene C"),
    "opls_145": ("CA", 6, 12.011, -0.115, 3.55, 0.07, "benzene C (12-site)"),
    "opls_146": ("HA", 1, 1.008, 0.115, 2.42, 0.03, "benzene H (12-site)"),
    "opls_154": ("OH", 8, 15.9994, -0.683, 3.12, 0.17, "monoalcohol O"),
    "opls_155": ("HO", 1, 1.008, 0.418, 0.0, 0.0, "monoalcohol H(O)"),
    "opls_156": ("HC", 1, 1.008, 0.04, 2.5, 0.03, "methanol H(C)"),
    "opls_157": ("CT", 6, 12.011, 0.145, 3.5, 0.066, "alcohol CH3 / CH2"),
    "opls_158": ("CT", 6, 12.011, 0.205, 3.5, 0.066, "alcohol CH"),
    "opls_159": ("CT", 6, 12.011, 0.265, 3.5, 0.066, "alcohol C"),
}

# "A-B": (k / kcal mol^-1 A^-2, r0 / A) for E = k (r - r0)^2.
_OPLSAA_BOND_TYPES = {
    "CT-CT": (268.0, 1.529), "CT-HC": (340.0, 1.09),
    "CM-CM": (549.0, 1.34), "CM-C=": (549.0, 1.34), "C=-C=": (385.0, 1.46),
    "CM-CT": (317.0, 1.51), "C=-CT": (317.0, 1.51),
    "CM-HC": (340.0, 1.08), "C=-HC": (340.0, 1.08),
    "CA-CA": (469.0, 1.4), "CA-CT": (317.0, 1.51), "CA-HA": (367.0, 1.08),
    "CA-CM": (427.0, 1.433), "CA-OH": (450.0, 1.364), "CM-OH": (450.0, 1.37),
    "CT-OH": (320.0, 1.41), "HO-OH": (553.0, 0.945),
}

# "A-B-C": (k / kcal mol^-1 rad^-2, theta0 / deg) for E = k (th - th0)^2.
_OPLSAA_ANGLE_TYPES = {
    "CT-CT-CT": (58.35, 112.7), "CT-CT-HC": (37.5, 110.7),
    "HC-CT-HC": (33.0, 107.8),
    "CT-CT-OH": (50.0, 109.5), "HC-CT-OH": (35.0, 109.5),
    "CT-OH-HO": (55.0, 108.5),
    "CM-CM-CT": (70.0, 124.0), "CM-C=-C=": (70.0, 124.0),
    "CM-C=-CT": (70.0, 124.0), "C=-C=-CT": (70.0, 124.0),
    "CT-CM-C=": (70.0, 124.0), "CM-CT-CM": (63.0, 112.4),
    "CM-CM-HC": (35.0, 120.0), "CM-C=-HC": (35.0, 120.0),
    "C=-CM-HC": (35.0, 120.0), "C=-C=-HC": (35.0, 120.0),
    "CT-CM-HC": (35.0, 117.0), "HC-CM-HC": (35.0, 117.0),
    "CT-CM-CT": (70.0, 130.0), "CM-CT-CT": (63.0, 111.1),
    "CM-CT-HC": (35.0, 109.5), "C=-CT-HC": (35.0, 109.5),
    "CM-CT-OH": (50.0, 109.5), "CM-CM-OH": (70.0, 123.0),
    "C=-CM-OH": (70.0, 123.0), "CM-OH-HO": (35.0, 109.0),
    "CA-CA-CA": (63.0, 120.0), "CA-CA-HA": (35.0, 120.0),
    "CA-CA-CT": (70.0, 120.0), "CA-CT-HC": (35.0, 109.5),
    "CA-CT-CT": (63.0, 114.0), "CA-CT-CA": (40.0, 109.5),
    "CM-CT-CA": (40.0, 109.5), "CA-CA-CM": (70.0, 124.0),
    "CA-CM-CT": (85.0, 119.7), "CA-CM-C=": (85.0, 117.0),
    "CA-CM-CM": (85.0, 117.0), "CA-CM-HC": (35.0, 123.3),
    "CA-CA-OH": (70.0, 120.0), "CA-CT-OH": (50.0, 109.5),
    "CA-OH-HO": (35.0, 113.0),
}

# "A-B-C-D": (V0, V1, V2, V3, V4) / kcal mol^-1 (X = wildcard).
_OPLSAA_DIHEDRAL_TYPES = {
    "CT-CT-CT-CT": (0.0, 1.3, -0.05, 0.2, 0.0),
    "CT-CT-CT-HC": (0.0, 0.0, 0.0, 0.3, 0.0),
    "HC-CT-CT-HC": (0.0, 0.0, 0.0, 0.3, 0.0),
    "CT-CT-CT-OH": (0.0, 1.711, -0.5, 0.663, 0.0),
    "HC-CT-CT-OH": (0.0, 0.0, 0.0, 0.468, 0.0),
    "CT-CT-OH-HO": (0.0, -0.356, -0.174, 0.492, 0.0),
    "HC-CT-OH-HO": (0.0, 0.0, 0.0, 0.45, 0.0),
    "OH-CT-CT-OH": (0.0, 9.066, 0.0, 0.0, 0.0),
    "X-CM-CM-X": (0.0, 0.0, 14.0, 0.0, 0.0),
    "CM-C=-C=-CM": (0.0, 1.423, 4.055, 0.858, 0.0),
    "CM-C=-C=-CT": (0.0, 0.0, 0.0, -0.372, 0.0),
    "CM-C=-C=-HC": (0.0, 0.0, 0.0, -0.372, 0.0),
    "CT-C=-C=-HC": (0.0, 0.0, 0.0, 0.3, 0.0),
    "HC-C=-C=-HC": (0.0, 0.0, 0.0, 0.3, 0.0),
    "CT-C=-CM-CT": (0.0, 0.0, 14.0, 0.0, 0.0),
    "CT-C=-CM-HC": (0.0, 0.0, 14.0, 0.0, 0.0),
    "CT-CM-C=-HC": (0.0, 0.0, 14.0, 0.0, 0.0),
    "HC-C=-CM-HC": (0.0, 0.0, 14.0, 0.0, 0.0),
    "CM-CM-CT-CT": (0.0, 0.346, 0.405, -0.904, 0.0),
    "C=-CM-CT-CT": (0.0, 0.346, 0.405, -0.904, 0.0),
    "CM-CM-CT-HC": (0.0, 0.0, 0.0, -0.372, 0.0),
    "HC-CM-CT-HC": (0.0, 0.0, 0.0, 0.318, 0.0),
    "CT-CM-CT-CT": (0.0, 2.817, -0.169, 0.543, 0.0),
    "CT-CM-CT-HC": (0.0, 0.0, 0.0, 0.3, 0.0),
    "CM-CT-CT-CT": (0.0, 1.3, -0.05, 0.2, 0.0),
    "CM-CT-CT-HC": (0.0, 0.0, 0.0, 0.366, 0.0),
    "CM-CT-CT-OH": (0.0, 1.711, -0.5, 0.663, 0.0),
    "CT-CM-CT-OH": (0.0, 1.711, -0.5, 0.663, 0.0),
    "HC-CM-CT-OH": (0.0, 0.0, 0.0, 0.468, 0.0),
    "CM-CT-OH-HO": (0.0, -0.9, 0.0, 0.0, 0.0),
    "C=-CT-OH-HO": (0.0, -0.9, 0.0, 0.0, 0.0),
    "CM-CM-CT-OH": (0.0, 0.5, 0.0, 0.0, 0.0),
    "C=-CM-CT-OH": (0.0, 0.5, 0.0, 0.0, 0.0),
    "X-CA-CA-X": (0.0, 0.0, 7.25, 0.0, 0.0),
    "CA-CA-CT-X": (0.0, 0.0, 0.0, 0.0, 0.0),
    "CA-CA-CT-HC": (0.0, 0.0, 0.0, 0.0, 0.0),
    "CA-CA-CT-CT": (0.0, 0.0, 0.0, 0.0, 0.0),
    "CA-CA-CT-OH": (0.0, 0.0, 0.0, 0.0, 0.0),
    "CA-CT-CT-CT": (0.0, 1.3, -0.05, 0.2, 0.0),
    "CA-CT-CT-HC": (0.0, 0.0, 0.0, 0.462, 0.0),
    "CA-CT-CT-OH": (0.0, 1.711, -0.5, 0.663, 0.0),
    "CA-CT-OH-HO": (0.0, -0.9, 0.0, 0.0, 0.0),
    "CA-CA-CM-CM": (0.0, 1.241, 3.353, -0.286, 0.0),
    "C=-CM-CA-CA": (0.0, 1.241, 3.353, -0.286, 0.0),
    "CA-CA-CM-CT": (0.0, 0.205, -0.531, 0.0, 0.0),
    "CA-CA-OH-HO": (0.0, 0.0, 1.682, 0.0, 0.0),
}

# key: V2 / kcal mol^-1 for E = V2/2 (1 - cos 2 phi); keys follow the
# GROMACS macro names (second field = the trigonal center's class family).
_OPLSAA_IMPROPER_TYPES = {
    "O-C-X-Y": 21.0, "Z-N-X-Y": 2.0, "Z-CM-X-Y": 30.0, "Z-CA-X-Y": 2.2,
}

# L-OPLS (Siu, Pluhackova & Boeckmann, JCTC 8, 1459, 2012, Table 2):
# refit hydrocarbon torsions (given in kJ/mol; converted below), new
# per-connectivity nonbonded types with adjusted charges and a reduced
# epsilon for methylene hydrogens. Bonds / angles / everything else are the
# unchanged OPLS-AA values.
#   name: (class, Z, mass, charge/e, sigma/A, epsilon/(kJ/mol), comment)
_LOPLS_ATOM_TYPES = {
    "lopls_CT_CH3": ("CT", 6, 12.011, -0.222, 3.5, 0.276144, "alkane CH3 C"),
    "lopls_CT_CH2": ("CT", 6, 12.011, -0.148, 3.5, 0.276144, "alkane CH2 C"),
    "lopls_CM_CH": ("CM", 6, 12.011, -0.16, 3.55, 0.317984, "alkene CH C"),
    "lopls_HC_CH3": ("HC", 1, 1.008, 0.074, 2.5, 0.12552, "CH3 hydrogen"),
    "lopls_HC_CH2": ("HC", 1, 1.008, 0.074, 2.5, 0.11, "CH2 hydrogen"),
    "lopls_HC_CH": ("HC", 1, 1.008, 0.16, 2.42, 0.12552, "alkene hydrogen"),
}

# "A-B-C-D": (V0, V1, V2, V3, V4) / kJ mol^-1 (converted below).
_LOPLS_DIHEDRAL_TYPES = {
    "CT-CT-CT-CT": (-0.305938, 2.697394, -0.896807, 0.74567, 0.0),
    "X-CM-CM-X": (0.0, 0.0, 51.2551, 0.0, 0.0),
    "CM-CM-CT-CT": (-1.49571, -3.368171, 1.34679, -0.4321105, 0.0),
    "CM-CT-CT-CT": (1.843356, 2.017484, 0.562197, 0.74369, 0.0),
}


def _oplsaa_library() -> OPLSLibrary:
    """Assemble the built-in OPLS-AA subset library.

    Returns
    -------
    OPLSLibrary
        The curated OPLS-AA subset.
    """
    atom_types = {name: {"cls": t[0], "element": t[1], "mass": t[2],
                         "charge": t[3], "sigma": t[4], "epsilon": t[5],
                         "comment": t[6]}
                  for name, t in _OPLSAA_ATOM_TYPES.items()}
    return OPLSLibrary(
        atom_types=atom_types,
        bond_types={k: {"k": v[0], "r0": v[1]}
                    for k, v in _OPLSAA_BOND_TYPES.items()},
        angle_types={k: {"k": v[0], "theta0": v[1]}
                     for k, v in _OPLSAA_ANGLE_TYPES.items()},
        dihedral_types={k: {"v": list(v)}
                        for k, v in _OPLSAA_DIHEDRAL_TYPES.items()},
        improper_types={k: {"v2": v}
                        for k, v in _OPLSAA_IMPROPER_TYPES.items()},
        name="oplsaa",
        references=["Jorgensen, Maxwell & Tirado-Rives, JACS 118, 11225 "
                    "(1996)", "Kaminski et al., J. Phys. Chem. B 105, 6474 "
                    "(2001)"])


def _lopls_library() -> OPLSLibrary:
    """Assemble the built-in L-OPLS library (OPLS-AA base + Siu 2012 refit).

    Returns
    -------
    OPLSLibrary
        The L-OPLS library.
    """
    lib = _oplsaa_library()
    lib.name = "lopls"
    lib.references = ["Siu, Pluhackova & Boeckmann, JCTC 8, 1459 (2012)",
                      "base parameters: Jorgensen, Maxwell & Tirado-Rives, "
                      "JACS 118, 11225 (1996)"]
    for name, t in _LOPLS_ATOM_TYPES.items():
        lib.atom_types[name] = {"cls": t[0], "element": t[1], "mass": t[2],
                                "charge": t[3], "sigma": t[4],
                                "epsilon": t[5] * KCAL_PER_KJ,
                                "comment": t[6]}
    for key, v in _LOPLS_DIHEDRAL_TYPES.items():
        lib.dihedral_types[key] = {"v": [x * KCAL_PER_KJ for x in v]}
    return lib


def _oplsaa_1996_library() -> OPLSLibrary:
    """The built-in OPLS-AA subset with the original 1996 alkane torsions.

    The current OPLS-AA distribution carries slightly revised alkane
    torsional parameters (a late-1999 update from the Jorgensen lab); this
    variant restores the three alkane torsions of the original paper
    (JACS 118, 11225, 1996, Supporting Information Table 7), which
    reproduce the conformational energies of the paper's Table 1 exactly.

    Returns
    -------
    OPLSLibrary
        The 1996-torsion variant.
    """
    lib = _oplsaa_library()
    lib.name = "oplsaa-1996"
    lib.dihedral_types["CT-CT-CT-CT"] = {"v": [0.0, 1.740, -0.157, 0.279,
                                               0.0]}
    lib.dihedral_types["CT-CT-CT-HC"] = {"v": [0.0, 0.0, 0.0, 0.366, 0.0]}
    lib.dihedral_types["HC-CT-CT-HC"] = {"v": [0.0, 0.0, 0.0, 0.318, 0.0]}
    return lib


def builtin_library(name: str = "oplsaa") -> OPLSLibrary:
    """Return one of the built-in curated parameter sets.

    ``"oplsaa"`` covers alkanes, alkenes, benzene rings and monoalcohols
    with the published OPLS-AA parameters; ``"oplsaa-1996"`` is that set
    with the original 1996 alkane torsions (reproduces Table 1 of the
    paper); ``"lopls"`` is the base set with the L-OPLS long-hydrocarbon
    refit of Siu et al. (2012) layered on top (new ``lopls_*`` atom types
    and refit ``CT-CT-CT-CT`` / alkene torsions).

    Parameters
    ----------
    name : str, optional
        ``"oplsaa"`` (default), ``"oplsaa-1996"`` or ``"lopls"``.

    Returns
    -------
    OPLSLibrary
        A fresh library instance (safe to modify).

    Raises
    ------
    KeyError
        For an unknown library name.
    """
    builders = {"oplsaa": _oplsaa_library, "oplsaa1996": _oplsaa_1996_library,
                "lopls": _lopls_library}
    key = name.lower().replace("-", "").replace("_", "")
    if key not in builders:
        raise KeyError(f"unknown built-in OPLS library {name!r} "
                       f"(available: {sorted(builders)})")
    return builders[key]()
