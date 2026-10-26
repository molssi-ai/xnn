"""AMOEBA and HIPPO parameter libraries: ``.frc`` bridge and Tinker file formats.

The polarizable multipole force fields of the Ponder group are specified by
two kinds of atom labels: *atom types* carry the electrostatic parameters
(permanent multipoles with their local-frame definitions, polarizabilities
and polarization groups, the HIPPO density parameters), *atom classes* carry
the valence and van der Waals parameters (several types share one class).
An :class:`AmoebaLibrary` keeps both, in the units of the papers and of the
Tinker parameter files (kcal/mol, Angstrom, degrees, e), together with the
force-field definition settings (anharmonic coefficients, combining rules,
scale factors) that the functional form needs.

Libraries live in the MolSSI/SEAMM ``.frc`` format (:mod:`xnn.ffnn.common.frc`)
like every other field of the ``ffnn`` family: the valence tables are the
``amoeba_bond`` / ``amoeba_angle`` / ``amoeba_angle_inplane`` /
``stretch_bend`` / ``urey_bradley`` / ``amoeba_out_of_plane`` /
``torsion_fourier`` / ``pi_torsion`` sections and the JSON
``torsion_torsion`` grids, the nonbonded ones ``buffered_14-7`` (with pair
overrides), ``atomic_multipole``, ``polarizability``, ``charge_penetration``,
``charge_transfer``, ``pauli_repulsion``, ``damped_dispersion``,
``exchange_polarization`` and the two charge-flux kinds; the settings are ``#metadata`` rows and the type-to-class
map is the ``#equivalence`` section. ``amoeba-water-2003`` (Ren and Ponder
2003), ``hippo-water-2021`` (Rackers *et al.* 2021) and ``amoebapro13``
(Shi *et al.* 2013) ship with xnn.

Because Tinker's text formats are the lingua franca of these force fields,
the module also reads Tinker parameter files (:func:`read_tinker_prm`), reads
Tinker coordinate files with their types and connectivity
(:func:`read_tinker_xyz`), and writes a library as a Tinker parameter file
(:func:`write_tinker_prm`) that Tinker and the LAMMPS AMOEBA package read as
is, which is how the fidelity notebooks drive both reference codes from one
set of xnn parameters.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

from xnn.common.data.elements import CHEMICAL_SYMBOLS, SYMBOL_TO_Z, atomic_number

from .oplslib import KCAL_TO_EV, WILDCARD

# the force-field definition keywords an AMOEBA / HIPPO file carries, with
# the Tinker defaults that apply when a file leaves one out
DEFAULT_SETTINGS = {
    "forcefield": "AMOEBA",
    "bond-cubic": 0.0, "bond-quartic": 0.0,
    "angle-cubic": 0.0, "angle-quartic": 0.0, "angle-pentic": 0.0, "angle-sextic": 0.0,
    "opbendtype": "W-D-C",
    "opbend-cubic": 0.0, "opbend-quartic": 0.0, "opbend-pentic": 0.0, "opbend-sextic": 0.0,
    "torsionunit": 1.0, "pitorsunit": 1.0, "tortorunit": 1.0,
    "vdwtype": "BUFFERED-14-7", "radiusrule": "CUBIC-MEAN", "radiustype": "R-MIN",
    "radiussize": "DIAMETER", "epsilonrule": "HHG",
    "delta-halgren": 0.07, "gamma-halgren": 0.12,
    "dielectric": 1.0, "polarization": "MUTUAL", "d-equals-p": False,
    "vdw-12-scale": 0.0, "vdw-13-scale": 0.0, "vdw-14-scale": 1.0, "vdw-15-scale": 1.0,
    "mpole-12-scale": 0.0, "mpole-13-scale": 0.0, "mpole-14-scale": 1.0, "mpole-15-scale": 1.0,
    "polar-12-scale": 0.0, "polar-13-scale": 0.0, "polar-14-scale": 1.0, "polar-15-scale": 1.0,
    "polar-12-intra": 0.0, "polar-13-intra": 0.0, "polar-14-intra": 0.5, "polar-15-intra": 1.0,
    "direct-11-scale": 0.0, "direct-12-scale": 1.0, "direct-13-scale": 1.0, "direct-14-scale": 1.0,
    "mutual-11-scale": 1.0, "mutual-12-scale": 1.0, "mutual-13-scale": 1.0, "mutual-14-scale": 1.0,
    "induce-12-scale": 1.0, "induce-13-scale": 1.0, "induce-14-scale": 1.0, "induce-15-scale": 1.0,
    "rep-12-scale": 0.0, "rep-13-scale": 0.0, "rep-14-scale": 1.0, "rep-15-scale": 1.0,
    "disp-12-scale": 0.0, "disp-13-scale": 0.0, "disp-14-scale": 1.0, "disp-15-scale": 1.0,
}
_FLAG_SETTINGS = ("d-equals-p",)
_TEXT_SETTINGS = ("forcefield", "opbendtype", "vdwtype", "radiusrule", "radiustype",
                  "radiussize", "epsilonrule", "polarization")
MAX_TORSION_TERMS = 6


def _fmt_key(parts: Sequence[str]) -> str:
    return "-".join(WILDCARD if str(p) in ("0", "*") else str(p) for p in parts)


def _split_key(key: str) -> tuple[str, ...]:
    return tuple(key.split("-"))


def _canonical(parts: Sequence[str], symmetry: str) -> tuple[tuple, bool]:
    from ..common.frc import canonical_key
    return canonical_key(symmetry, [str(p) for p in parts])


@dataclass
class AmoebaLibrary:
    """A parsed AMOEBA / HIPPO parameter library, in Tinker units.

    Energies are kcal/mol, lengths Angstrom, angles degrees, charges e,
    dipoles e bohr, quadrupoles e bohr^2, polarizabilities Angstrom^3, damping
    exponents 1/Angstrom. Bonded tables
    are keyed by the atom *classes* of the term joined with ``-`` in the
    canonical order of the ``.frc`` reader (bond and angle keys sorted end to
    end, torsions by their central pair); the out-of-plane key keeps its
    written order (out-of-plane atom, center, the two others); ``X`` is the
    wildcard.

    Parameters and attributes
    -------------------------
    atom_types : dict[str, dict]
        ``type -> {"class", "element", "mass", "connections", "comment"}``.
    bond_types : dict[str, dict]
        ``"A-B" -> {"k", "r0"}`` for ``k dr^2 (1 + c3 dr + c4 dr^2)`` with
        the cubic and quartic coefficients from ``settings``.
    angle_types, angle_inplane_types : dict[str, dict]
        ``"A-B-C" -> {"k", "theta0": [theta_1, (theta_2, theta_3)]}``: the
        ideal angles for a center carrying 0, 1 or 2 further hydrogens; the
        in-plane table applies to trigonal centers (Tinker ``anglep``).
    strbnd_types : dict[str, dict]
        ``"A-B-C" -> {"k1", "k2"}``, ``k1`` for the A-B bond of the key.
    urey_types : dict[str, dict]
        ``"A-B-C" -> {"k", "r0"}`` on the A..C distance.
    opbend_types : dict[str, dict]
        ``"D-B-A-C" -> {"k"}`` with D the out-of-plane atom, B the trigonal
        center, A and C the other substituents (sorted, ``X`` wildcards).
    torsion_types : dict[str, dict]
        ``"A-B-C-D" -> {"terms": [[K, n, phase], ...]}`` for
        ``E = sum K [1 + cos(n phi - phase)]`` per dihedral (``K`` already
        includes the file's ``torsionunit``).
    pitors_types : dict[str, dict]
        ``"B-C" -> {"k"}`` for ``k [1 - cos(2 phi)]`` on the p-orbital angle.
    tortor_types : dict[str, dict]
        ``"A-B-C-D-E" -> {"angles1", "angles2", "values"}``: the two angle
        grids (degrees, ``-180 .. 180`` inclusive) and the ``(nx, ny)`` energy
        table (kcal/mol, with ``tortorunit`` applied).
    vdw_types : dict[str, dict]
        ``class -> {"rmin", "eps", "reduction"}``: the like-pair minimum
        distance (Tinker's diameter), well depth and hydrogen reduction
        factor (1 for heavy atoms).
    vdw_pair_types : dict[str, dict]
        ``"A-B" -> {"rmin", "eps"}`` pair overrides (Tinker ``vdwpr``).
    multipoles : dict[str, list]
        ``type -> [{"frame": (kz, kx, ky), "q", "dipole": [3],
        "quadrupole": [[3 x 3]]}, ...]``, one entry per local-frame
        definition of the type, in file order. Frame entries are atom type
        names with Tinker's sign convention (``"-HW"``) or ``"0"``; the
        dipole is in e bohr and the quadrupole traceless in e bohr^2, the
        atomic units of the parameter files and the papers.
    polarize : dict[str, dict]
        ``type -> {"alpha", "thole", "groups": [types]}``; ``thole = 0``
        means charge-penetration damping (HIPPO).
    chgpen, chgtrn, repulsion, dispersion : dict[str, dict]
        Per class: ``{"core", "alpha"}``, ``{"eps", "alpha"}``,
        ``{"k", "alpha", "valence"}``, ``{"c6", "alpha"}``.
    exchpol : dict[str, dict]
        Per class: ``{"spring", "size", "damp", "use"}``, the exchange
        polarization of Chung *et al.* (2022): the spring constant ``k``, the
        orbital size ``q`` and exponent ``zeta`` of the overlap, and the use
        flag (a pair takes part when either of its classes is used).
    bndcflux : dict[str, dict]
        ``"A-B" -> {"j"}``: the bond charge flux, in e/Angstrom, moved from
        A to B per unit stretch of the key as stored.
    angcflux : dict[str, dict]
        ``"A-B-C" -> {"jt1", "jt2", "jb1", "jb2"}`` (e/radian, e/Angstrom).
    settings : dict
        The force-field definition keywords (see :data:`DEFAULT_SETTINGS`).
    name : str
        Short label of the parameter set.
    references, templates, fragments, metadata, notes
        As for the other ``ffnn`` libraries.
    """

    atom_types: dict = field(default_factory=dict)
    bond_types: dict = field(default_factory=dict)
    angle_types: dict = field(default_factory=dict)
    angle_inplane_types: dict = field(default_factory=dict)
    strbnd_types: dict = field(default_factory=dict)
    urey_types: dict = field(default_factory=dict)
    opbend_types: dict = field(default_factory=dict)
    torsion_types: dict = field(default_factory=dict)
    pitors_types: dict = field(default_factory=dict)
    tortor_types: dict = field(default_factory=dict)
    vdw_types: dict = field(default_factory=dict)
    vdw_pair_types: dict = field(default_factory=dict)
    multipoles: dict = field(default_factory=dict)
    polarize: dict = field(default_factory=dict)
    chgpen: dict = field(default_factory=dict)
    chgtrn: dict = field(default_factory=dict)
    repulsion: dict = field(default_factory=dict)
    dispersion: dict = field(default_factory=dict)
    exchpol: dict = field(default_factory=dict)
    bndcflux: dict = field(default_factory=dict)
    angcflux: dict = field(default_factory=dict)
    settings: dict = field(default_factory=lambda: dict(DEFAULT_SETTINGS))
    name: str = "amoeba"
    references: list = field(default_factory=list)
    templates: dict = field(default_factory=dict)
    fragments: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    # introspection
    @property
    def model(self) -> str:
        """``"hippo"`` when the library carries charge-penetration (HIPPO)
        nonbonded terms, else ``"amoeba"``."""
        return "hippo" if (self.chgpen or self.repulsion or self.dispersion) else "amoeba"

    def cls(self, type_name: str) -> str:
        """The atom class of an atom type (``KeyError`` if unknown)."""
        return str(self.atom_types[type_name]["class"])

    def classes(self) -> list[str]:
        """The atom classes, in first-appearance order of the types."""
        out: list[str] = []
        for a in self.atom_types.values():
            c = str(a["class"])
            if c not in out:
                out.append(c)
        return out

    def setting(self, key: str):
        """A force-field setting, with the Tinker default when unset."""
        return self.settings.get(key, DEFAULT_SETTINGS.get(key))

    def scale(self, term: str, sep: int) -> float:
        """The scale factor of ``term`` (``"vdw"``, ``"mpole"``, ``"polar"``,
        ``"polar-intra"``, ``"direct"``, ``"mutual"``, ``"induce"``, ``"rep"``,
        ``"disp"``) for atoms separated by ``sep`` bonds (1,1 .. 1,5)."""
        if term == "polar-intra":
            return float(self.setting(f"polar-1{sep}-intra"))
        return float(self.setting(f"{term}-1{sep}-scale"))

    # canonical accessors
    def bond(self, a: str, b: str) -> Optional[dict]:
        return self.bond_types.get(_fmt_key(_canonical((a, b), "like_bond")[0]))

    def angle(self, a: str, b: str, c: str, inplane: bool = False) -> Optional[dict]:
        table = self.angle_inplane_types if inplane else self.angle_types
        return table.get(_fmt_key(_canonical((a, b, c), "like_angle")[0]))

    def urey(self, a: str, b: str, c: str) -> Optional[dict]:
        return self.urey_types.get(_fmt_key(_canonical((a, b, c), "like_angle")[0]))

    def strbnd(self, a: str, b: str, c: str) -> Optional[tuple[float, float]]:
        """``(k1, k2)`` oriented to the bonds ``a-b`` and ``b-c`` as asked."""
        key, flipped = _canonical((a, b, c), "like_angle")
        row = self.strbnd_types.get(_fmt_key(key))
        if row is None:
            return None
        k1, k2 = float(row["k1"]), float(row["k2"])
        return (k2, k1) if flipped else (k1, k2)

    def angle_flux(self, a: str, b: str, c: str) -> Optional[tuple[float, float, float, float]]:
        """``(jt1, jt2, jb1, jb2)`` oriented to the atoms ``a, b, c`` as asked."""
        key, flipped = _canonical((a, b, c), "like_angle")
        row = self.angcflux.get(_fmt_key(key))
        if row is None:
            return None
        v = (float(row["jt1"]), float(row["jt2"]), float(row["jb1"]), float(row["jb2"]))
        return (v[1], v[0], v[3], v[2]) if flipped else v

    def bond_flux(self, a: str, b: str) -> Optional[float]:
        """The charge moved from ``a`` to ``b`` per unit stretch (sign
        follows the orientation asked)."""
        key, flipped = _canonical((a, b), "like_bond")
        row = self.bndcflux.get(_fmt_key(key))
        if row is None:
            return None
        return -float(row["j"]) if flipped else float(row["j"])

    def vdw_pair(self, a: str, b: str) -> Optional[dict]:
        return self.vdw_pair_types.get(_fmt_key(_canonical((a, b), "like_bond")[0]))

    def pitors(self, a: str, b: str) -> Optional[dict]:
        return self.pitors_types.get(_fmt_key(_canonical((a, b), "like_bond")[0]))

    def torsion(self, a: str, b: str, c: str, d: str) -> Optional[dict]:
        """The torsion row for the class quadruple (the exact key, in either
        direction, beats wildcard rows)."""
        from .oplslib import resolve_dihedral_type
        key = resolve_dihedral_type(self.torsion_types, a, b, c, d)
        return None if key is None else self.torsion_types[key]

    def opbend(self, d: str, b: str, a: str, c: str) -> Optional[dict]:
        """The out-of-plane row for the out-of-plane atom ``d`` at center
        ``b`` whose other substituents are ``a`` and ``c``: the exact row,
        then the wildcard rows for the other substituents."""
        a, c = sorted((a, c))
        for pat in ((d, b, a, c), (d, b, a, WILDCARD), (d, b, c, WILDCARD),
                    (d, b, WILDCARD, WILDCARD)):
            row = self.opbend_types.get(_fmt_key(pat))
            if row is not None:
                return row
        return None

    def tortor(self, a: str, b: str, c: str, d: str, e: str
               ) -> Optional[tuple[dict, bool]]:
        """The torsion-torsion row for five classes in chain order and
        whether it matched in reverse (the two angles then swap)."""
        row = self.tortor_types.get(_fmt_key((a, b, c, d, e)))
        if row is not None:
            return row, False
        row = self.tortor_types.get(_fmt_key((e, d, c, b, a)))
        if row is not None:
            return row, True
        return None

    # persistence
    def save(self, path: Union[str, Path]) -> None:
        """Write the library as native JSON."""
        data = {k: getattr(self, k) for k in self.__dataclass_fields__}
        Path(path).write_text(json.dumps(data, indent=1) + "\n")

    def save_frc(self, path: Union[str, Path], name: Optional[str] = None,
                 version: str = "1.0") -> Path:
        """Write the library as a ``.frc`` force-field file."""
        return to_forcefield(self, name=name, version=version).write(path)


def _read_json(path: Path) -> AmoebaLibrary:
    data = json.loads(path.read_text())
    lib = AmoebaLibrary()
    for k, v in data.items():
        if k in lib.__dataclass_fields__:
            setattr(lib, k, v)
    return lib


# .frc bridge
_SETTING_DESCRIPTIONS = {
    "forcefield": "Name of the parameter set", "bond-cubic": "Cubic bond coefficient (1/A)",
    "bond-quartic": "Quartic bond coefficient (1/A^2)",
    "angle-cubic": "Cubic angle coefficient (1/degree)",
    "angle-quartic": "Quartic angle coefficient (1/degree^2)",
    "angle-pentic": "Pentic angle coefficient (1/degree^3)",
    "angle-sextic": "Sextic angle coefficient (1/degree^4)",
    "opbendtype": "Out-of-plane angle definition (ALLINGER or W-D-C)",
    "opbend-cubic": "Cubic out-of-plane coefficient (1/degree)",
    "opbend-quartic": "Quartic out-of-plane coefficient (1/degree^2)",
    "opbend-pentic": "Pentic out-of-plane coefficient (1/degree^3)",
    "opbend-sextic": "Sextic out-of-plane coefficient (1/degree^4)",
    "torsionunit": "Factor the Tinker torsion amplitudes were multiplied by",
    "pitorsunit": "Factor the Tinker pi-torsion amplitudes were multiplied by",
    "tortorunit": "Factor the Tinker torsion-torsion grids were multiplied by",
    "vdwtype": "Van der Waals functional form", "radiusrule": "Combining rule of the minimum distance",
    "radiustype": "Meaning of the size parameter", "radiussize": "Radius or diameter",
    "epsilonrule": "Combining rule of the well depth",
    "delta-halgren": "Buffering constant delta of the 14-7 form",
    "gamma-halgren": "Buffering constant gamma of the 14-7 form",
    "dielectric": "Dielectric constant", "polarization": "Induced dipole model (MUTUAL or DIRECT)",
    "d-equals-p": "Use one set of induced dipoles for energy and gradient (HIPPO)",
}


def from_forcefield(ff, strict: bool = True) -> AmoebaLibrary:
    """Build an :class:`AmoebaLibrary` from a resolved ``.frc`` force field.

    Parameters
    ----------
    ff : xnn.ffnn.common.frc.ForceField
        A force field of the AMOEBA / HIPPO functional form (see the module
        notes for the section kinds).
    strict : bool, optional
        If ``True`` (default) refuse functional forms this model does not
        implement; else ignore them and record a note.

    Returns
    -------
    AmoebaLibrary
        The library, templates and fragments included.
    """
    lib = AmoebaLibrary(name=ff.name)
    lib.metadata = dict(ff.metadata)
    lib.templates = dict(ff.templates)
    lib.fragments = dict(ff.fragments)
    lib.references = [r.text for r in ff.references_used()]
    settings = dict(DEFAULT_SETTINGS)
    for key, value in lib.metadata.items():
        if key in DEFAULT_SETTINGS:
            if key in _FLAG_SETTINGS:
                settings[key] = str(value).strip().lower() in ("1", "true", "yes", "1.0")
            elif key in _TEXT_SETTINGS:
                settings[key] = str(value)
            else:
                settings[key] = float(value)
    lib.settings = settings

    known = {"metadata", "atom_types", "equivalence", "amoeba_bond", "amoeba_angle",
             "amoeba_angle_inplane", "stretch_bend", "urey_bradley", "amoeba_out_of_plane",
             "torsion_fourier", "pi_torsion", "torsion_torsion", "buffered_14-7",
             "buffered_14-7_pair", "atomic_multipole", "polarizability",
             "charge_penetration", "charge_transfer", "pauli_repulsion",
             "damped_dispersion", "exchange_polarization", "bond_charge_flux", "angle_charge_flux"}
    unsupported = {k: len(v) for k, v in ff.sections.items() if k not in known}
    if unsupported:
        if strict:
            raise ValueError(
                f"force field {ff.name!r} uses functional forms the AMOEBA / HIPPO "
                f"model does not implement: {unsupported} (section: rows). Pass "
                "strict=False to load it without those sections")
        for k, n in unsupported.items():
            lib.notes.append(f"ignored section {k} ({n} rows)")

    for name, at in ff.atom_types.items():
        try:
            element = atomic_number(str(at.get("El", "")))
        except KeyError:
            element = 0
        conns = at.get("connections", 0)
        try:
            conns = int(conns)
        except (TypeError, ValueError):
            conns = 0
        lib.atom_types[name] = {"class": ff.equivalent(name, "nonbond"),
                                "element": element, "mass": float(at.get("Mass", 0.0)),
                                "connections": conns,
                                "comment": str(at.get("Comment", ""))}

    def key_of(key) -> str:
        return _fmt_key(key)

    for key, row in ff.rows("amoeba_bond").items():
        lib.bond_types[key_of(key)] = {"k": float(row.values["K2"]), "r0": float(row.values["R0"])}
    for kind, table in (("amoeba_angle", lib.angle_types),
                        ("amoeba_angle_inplane", lib.angle_inplane_types)):
        for key, row in ff.rows(kind).items():
            thetas = [float(row.values[c]) for c in ("Theta0_1", "Theta0_2", "Theta0_3")
                      if c in row.values and row.values[c] is not None]
            table[key_of(key)] = {"k": float(row.values["K2"]), "theta0": thetas}
    for key, row in ff.rows("stretch_bend").items():
        lib.strbnd_types[key_of(key)] = {"k1": float(row.values["K1"]), "k2": float(row.values["K2"])}
    for key, row in ff.rows("urey_bradley").items():
        lib.urey_types[key_of(key)] = {"k": float(row.values["K2"]), "r0": float(row.values["S0"])}
    for key, row in ff.rows("amoeba_out_of_plane").items():
        lib.opbend_types[key_of(key)] = {"k": float(row.values["K2"])}
    for key, row in ff.rows("torsion_fourier").items():
        terms = []
        for m in range(1, MAX_TORSION_TERMS + 1):
            if f"K{m}" not in row.values or row.values[f"K{m}"] is None:
                break
            terms.append([float(row.values[f"K{m}"]), int(row.values[f"n{m}"]),
                          float(row.values[f"Phi0_{m}"])])
        lib.torsion_types[key_of(key)] = {"terms": terms}
    for key, row in ff.rows("pi_torsion").items():
        lib.pitors_types[key_of(key)] = {"k": float(row.values["K2"])}
    for key, entry in ff.rows("torsion_torsion").items():
        lib.tortor_types[str(key)] = {"angles1": [float(x) for x in entry["angles1"]],
                                      "angles2": [float(x) for x in entry["angles2"]],
                                      "values": [[float(x) for x in r] for r in entry["values"]]}
    for key, row in ff.rows("buffered_14-7").items():
        red = row.values.get("reduction")
        lib.vdw_types[key[0]] = {"rmin": float(row.values["rmin"]), "eps": float(row.values["eps"]),
                                 "reduction": 1.0 if red is None else float(red)}
    for key, row in ff.rows("buffered_14-7_pair").items():
        lib.vdw_pair_types[key_of(key)] = {"rmin": float(row.values["rmin"]),
                                           "eps": float(row.values["eps"])}
    for key, row in ff.rows("atomic_multipole").items():
        t, kz, kx, ky = key
        v = row.values
        lib.multipoles.setdefault(t, []).append({
            "frame": (str(kz), str(kx), str(ky)), "q": float(v["Q"]),
            "dipole": [float(v["Dx"]), float(v["Dy"]), float(v["Dz"])],
            "quadrupole": [[float(v["Qxx"]), float(v["Qxy"]), float(v["Qxz"])],
                           [float(v["Qxy"]), float(v["Qyy"]), float(v["Qyz"])],
                           [float(v["Qxz"]), float(v["Qyz"]), float(v["Qzz"])]]})
    for key, row in ff.rows("polarizability").items():
        groups = str(row.values.get("Group", "") or "").split()
        lib.polarize[key[0]] = {"alpha": float(row.values["alpha"]),
                                "thole": float(row.values.get("thole", 0.0) or 0.0),
                                "groups": groups}
    for key, row in ff.rows("charge_penetration").items():
        lib.chgpen[key[0]] = {"core": float(row.values["Core"]), "alpha": float(row.values["alpha"])}
    for key, row in ff.rows("charge_transfer").items():
        lib.chgtrn[key[0]] = {"eps": float(row.values["eps"]), "alpha": float(row.values["alpha"])}
    for key, row in ff.rows("pauli_repulsion").items():
        lib.repulsion[key[0]] = {"k": float(row.values["K"]), "alpha": float(row.values["alpha"]),
                                 "valence": float(row.values["valence"])}
    for key, row in ff.rows("damped_dispersion").items():
        lib.dispersion[key[0]] = {"c6": float(row.values["C6"]), "alpha": float(row.values["alpha"])}
    for key, row in ff.rows("exchange_polarization").items():
        lib.exchpol[key[0]] = {"spring": float(row.values["spring"]), "size": float(row.values["size"]),
                               "damp": float(row.values["damp"]), "use": float(row.values["use"]) != 0.0}
    for key, row in ff.rows("bond_charge_flux").items():
        lib.bndcflux[key_of(key)] = {"j": float(row.values["J"])}
    for key, row in ff.rows("angle_charge_flux").items():
        lib.angcflux[key_of(key)] = {k.lower(): float(row.values[k]) for k in ("Jt1", "Jt2", "Jb1", "Jb2")}
    return lib


def to_forcefield(lib: AmoebaLibrary, name: Optional[str] = None, version: str = "1.0"):
    """Write an :class:`AmoebaLibrary` as an in-memory ``.frc`` file.

    Parameters
    ----------
    lib : AmoebaLibrary
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
    name = name or lib.name or "amoeba"
    label = name
    frc = FrcFile.empty()

    def sym(z):
        return CHEMICAL_SYMBOLS[int(z)] if 0 < int(z) < len(CHEMICAL_SYMBOLS) else "X"

    meta_rows = [(("ff_form",), {"Value": lib.model, "Description": "The functional form of the forcefield"}),
                 (("charges",), {"Value": "multipole", "Description": "Permanent atomic multipoles per atom type"})]
    for key in DEFAULT_SETTINGS:
        value = lib.settings.get(key, DEFAULT_SETTINGS[key])
        if key in _FLAG_SETTINGS:
            value = "true" if value else "false"
        meta_rows.append(((key,), {"Value": value, "Description": _SETTING_DESCRIPTIONS.get(key, "")}))
    for key, value in lib.metadata.items():
        if key not in DEFAULT_SETTINGS and key not in ("ff_form", "charges"):
            meta_rows.append(((key,), {"Value": value, "Description": ""}))

    def thetas(v):
        out = {}
        for m, t in enumerate(v["theta0"][:3], start=1):
            out[f"Theta0_{m}"] = float(t)
        return out

    def term_columns(terms):
        out = {}
        for m, (k, n, ph) in enumerate(terms[:MAX_TORSION_TERMS], start=1):
            out[f"K{m}"] = float(k)
            out[f"n{m}"] = int(n)
            out[f"Phi0_{m}"] = float(ph)
        return out

    tcols = [c for m in range(1, MAX_TORSION_TERMS + 1) for c in (f"K{m}", f"n{m}", f"Phi0_{m}")]
    sections = [
        make_section("metadata", label, ["Parameter"], ["Value", "Description"], meta_rows, version=version),
        make_section("atom_types", label, ["Type"], ["Mass", "El", "connections", "Comment"],
                     [((n,), {"Mass": float(a.get("mass", 0.0)), "El": sym(a.get("element", 0)),
                              "connections": int(a.get("connections", 0) or 0),
                              "Comment": str(a.get("comment", ""))})
                      for n, a in lib.atom_types.items()], version=version),
        make_section("equivalence", label, ["Type"], ["NonB", "Bond", "Angle", "Torsion", "OOP"],
                     [((n,), {c: str(a["class"]) for c in ("NonB", "Bond", "Angle", "Torsion", "OOP")})
                      for n, a in lib.atom_types.items()], version=version,
                     annotations=["The equivalent 'type' is the AMOEBA atom class carrying the valence "
                                  "and van der Waals parameters"]),
    ]
    if lib.vdw_types:
        sections.append(make_section(
            "buffered_14-7", label, ["I"], ["rmin", "eps", "reduction"],
            [((c,), {"rmin": float(v["rmin"]), "eps": float(v["eps"]),
                     "reduction": float(v.get("reduction", 1.0))}) for c, v in lib.vdw_types.items()],
            version=version,
            annotations=["E = eps(ij) [(1 + delta)/(rho + delta)]^7 [(1 + gamma)/(rho^7 + gamma) - 2], "
                         "rho = r/rmin(ij)",
                         "rmin is the like-pair minimum distance; rmin(ij) and eps(ij) follow radiusrule "
                         "and epsilonrule",
                         "the hydrogen site sits on its bond at reduction x the bond length from the heavy atom"]))
    if lib.vdw_pair_types:
        sections.append(make_section(
            "buffered_14-7_pair", label, ["I", "J"], ["rmin", "eps"],
            [(_split_key(k), {"rmin": float(v["rmin"]), "eps": float(v["eps"])})
             for k, v in lib.vdw_pair_types.items()], version=version))
    if lib.bond_types:
        sections.append(make_section(
            "amoeba_bond", label, ["I", "J"], ["R0", "K2"],
            [(_split_key(k), {"R0": float(v["r0"]), "K2": float(v["k"])}) for k, v in lib.bond_types.items()],
            version=version, annotations=["E = K2 dr^2 [1 + bond-cubic dr + bond-quartic dr^2], dr = R - R0"]))
    for kind, table, cols in (("amoeba_angle", lib.angle_types, ["K2", "Theta0_1", "Theta0_2", "Theta0_3"]),
                              ("amoeba_angle_inplane", lib.angle_inplane_types, ["K2", "Theta0_1", "Theta0_2"])):
        if table:
            sections.append(make_section(
                kind, label, ["I", "J", "K"], cols,
                [(_split_key(k), {"K2": float(v["k"]), **thetas(v)}) for k, v in table.items()],
                version=version,
                annotations=["E = K2 dTheta^2 [1 + angle-cubic dTheta + ... + angle-sextic dTheta^4] "
                             "(dTheta in degrees, K2 per radian^2)",
                             "Theta0_2 / Theta0_3 apply when the center carries 1 / 2 further hydrogens"]))
    if lib.strbnd_types:
        sections.append(make_section(
            "stretch_bend", label, ["I", "J", "K"], ["K1", "K2"],
            [(_split_key(k), {"K1": float(v["k1"]), "K2": float(v["k2"])}) for k, v in lib.strbnd_types.items()],
            version=version, annotations=["E = (K1 dr_IJ + K2 dr_JK) dTheta"]))
    if lib.urey_types:
        sections.append(make_section(
            "urey_bradley", label, ["I", "J", "K"], ["S0", "K2"],
            [(_split_key(k), {"S0": float(v["r0"]), "K2": float(v["k"])}) for k, v in lib.urey_types.items()],
            version=version, annotations=["E = K2 (S - S0)^2 on the I..K distance"]))
    if lib.opbend_types:
        sections.append(make_section(
            "amoeba_out_of_plane", label, ["I", "J", "K", "L"], ["K2"],
            [(_split_key(k), {"K2": float(v["k"])}) for k, v in lib.opbend_types.items()],
            version=version,
            annotations=["E = K2 chi^2 [1 + opbend-cubic chi + ...]: I the out-of-plane atom, J the trigonal center, "
                         "K and L its other substituents",
                         "chi (degrees) follows opbendtype"]))
    if lib.torsion_types:
        sections.append(make_section(
            "torsion_fourier", label, ["I", "J", "K", "L"], tcols,
            [(_split_key(k), term_columns(v["terms"])) for k, v in lib.torsion_types.items()],
            version=version,
            annotations=["E = sum_i K_i [1 + cos(n_i Phi - Phi0_i)]; K_i includes the Tinker torsionunit"]))
    if lib.pitors_types:
        sections.append(make_section(
            "pi_torsion", label, ["I", "J"], ["K2"],
            [(_split_key(k), {"K2": float(v["k"])}) for k, v in lib.pitors_types.items()],
            version=version,
            annotations=["E = K2 [1 - cos(2 Phi)], Phi the angle between the p-orbitals of the two trigonal centers"]))
    if lib.multipoles:
        rows = []
        for t, entries in lib.multipoles.items():
            for e in entries:
                Q = e["quadrupole"]
                rows.append(((t,) + tuple(str(x) for x in e["frame"]),
                             {"Q": float(e["q"]), "Dx": float(e["dipole"][0]), "Dy": float(e["dipole"][1]),
                              "Dz": float(e["dipole"][2]), "Qxx": float(Q[0][0]), "Qxy": float(Q[0][1]),
                              "Qxz": float(Q[0][2]), "Qyy": float(Q[1][1]), "Qyz": float(Q[1][2]),
                              "Qzz": float(Q[2][2])}))
        sections.append(make_section(
            "atomic_multipole", label, ["Type", "Z", "X", "Y"],
            ["Q", "Dx", "Dy", "Dz", "Qxx", "Qxy", "Qxz", "Qyy", "Qyz", "Qzz"], rows, version=version,
            annotations=["Local frame from the bonded (or 1-3) atoms of the given types: "
                         "Z X = z-then-x, -Z -X = bisector,",
                         "Z -X -Y = z-bisect, -Z -X -Y = 3-fold, Z 0 = z-only, 0 = none; "
                         "a fourth atom Y fixes the chirality",
                         "dipole in e bohr, traceless quadrupole in e bohr^2 "
                         "(atomic units, as in the papers and the Tinker files)"]))
    if lib.polarize:
        sections.append(make_section(
            "polarizability", label, ["I"], ["alpha", "thole", "Group"],
            [((t,), {"alpha": float(v["alpha"]), "thole": float(v.get("thole", 0.0)),
                     "Group": " ".join(str(g) for g in v.get("groups", []))}) for t, v in lib.polarize.items()],
            version=version,
            annotations=["thole = 0 selects charge-penetration damping; "
                         "Group lists the atom types of the polarization group"]))
    for kind, table, cols, conv in (
            ("charge_penetration", lib.chgpen, ["Core", "alpha"], lambda v: {"Core": v["core"], "alpha": v["alpha"]}),
            ("charge_transfer", lib.chgtrn, ["eps", "alpha"], lambda v: {"eps": v["eps"], "alpha": v["alpha"]}),
            ("pauli_repulsion", lib.repulsion, ["K", "alpha", "valence"],
             lambda v: {"K": v["k"], "alpha": v["alpha"], "valence": v["valence"]}),
            ("damped_dispersion", lib.dispersion, ["C6", "alpha"], lambda v: {"C6": v["c6"], "alpha": v["alpha"]}),
            ("exchange_polarization", lib.exchpol, ["spring", "size", "damp", "use"],
             lambda v: {"spring": v["spring"], "size": v["size"], "damp": v["damp"], "use": 1.0 if v["use"] else 0.0})):
        if table:
            sections.append(make_section(kind, label, ["I"], cols,
                                         [((c,), {k: float(x) for k, x in conv(v).items()}) for c, v in table.items()],
                                         version=version))
    if lib.bndcflux:
        sections.append(make_section(
            "bond_charge_flux", label, ["I", "J"], ["J"],
            [(_split_key(k), {"J": float(v["j"])}) for k, v in lib.bndcflux.items()], version=version,
            annotations=["dq_J = -dq_I = J (R - R0) moved from I to J"]))
    if lib.angcflux:
        sections.append(make_section(
            "angle_charge_flux", label, ["I", "J", "K"], ["Jt1", "Jt2", "Jb1", "Jb2"],
            [(_split_key(k), {"Jt1": float(v["jt1"]), "Jt2": float(v["jt2"]),
                              "Jb1": float(v["jb1"]), "Jb2": float(v["jb2"])})
             for k, v in lib.angcflux.items()], version=version,
            annotations=["dq_I = Jt1 dTheta + Jb1 dr_JK, dq_K = Jt2 dTheta + Jb2 dr_IJ, dq_J = -(dq_I + dq_K)"]))
    if lib.tortor_types:
        tsec = Section(kind="torsion_torsion", label=label)
        tsec.data = {k: {version: {"angles1": v["angles1"], "angles2": v["angles2"],
                                   "values": v["values"]}} for k, v in lib.tortor_types.items()}
        sections.append(tsec)
    if lib.templates:
        tsec = Section(kind="templates", label=label)
        tsec.data = {t: {str(e.get("version", version)): {k: v for k, v in e.items() if k != "version"}}
                     for t, e in lib.templates.items()}
        sections.append(tsec)
    if lib.fragments:
        fsec = Section(kind="fragments", label=label)
        fsec.data = {t: {str(e.get("version", version)): {k: v for k, v in e.items() if k != "version"}}
                     for t, e in lib.fragments.items()}
        sections.append(fsec)
    for sec in sections:
        frc.sections[(sec.kind, sec.label)] = sec
    define = Define(name=name)
    for sec in sections:
        define.entries.append((version, "1", sec.kind, [label]))
    frc.defines[name] = define
    text = "\n".join(lib.references) or f"AMOEBA / HIPPO parameters exported from xnn ({name})."
    frc.references[("<memory>", "1")] = Reference(number="1", text=text, author="xnn")
    return frc


def read_amoeba(source: Union[str, Path], strict: bool = True) -> AmoebaLibrary:
    """Read an AMOEBA / HIPPO parameter library.

    Parameters
    ----------
    source : str or Path
        A ``.json`` file in the native format, a Tinker ``.prm`` file, or a
        ``.frc`` force-field spec understood by
        :func:`~xnn.ffnn.common.frc.find_forcefield`: a variant shipped with
        xnn by name (``"amoeba-water-2003"``, ``"hippo-water-2021"``,
        ``"amoebapro13"``), a ``.frc`` path, or ``"<path>.frc:<variant>"``.
    strict : bool, optional
        See :func:`from_forcefield`.

    Returns
    -------
    AmoebaLibrary
        The parsed library.
    """
    s = str(source)
    if s.lower().endswith(".json"):
        return _read_json(Path(s))
    if s.lower().endswith(".prm"):
        return read_tinker_prm(s)
    from ..common.frc import read_forcefield
    return from_forcefield(read_forcefield(s), strict=strict)


def builtin_library(name: str = "amoeba-water-2003", strict: bool = True) -> AmoebaLibrary:
    """Return one of the AMOEBA / HIPPO parameter sets shipped with xnn.

    ``"amoeba-water-2003"`` is the water model of Ren and Ponder (2003),
    ``"hippo-water-2021"`` the HIPPO water model of Rackers *et al.* (2021)
    and ``"amoebapro13"`` the AMOEBA protein force field of Shi *et al.*
    (2013) with its water and ions.

    Parameters
    ----------
    name : str, optional
        The variant name.
    strict : bool, optional
        See :func:`from_forcefield`.

    Returns
    -------
    AmoebaLibrary
        A fresh library instance (safe to modify).
    """
    return read_amoeba(name, strict=strict)


# Tinker parameter files
def _tokens(line: str) -> list[str]:
    for mark in ("!!", "#"):
        if mark in line:
            line = line.split(mark, 1)[0]
    return line.replace(",", " ").split()


def read_tinker_prm(path: Union[str, Path]) -> AmoebaLibrary:
    """Read a Tinker parameter file of the AMOEBA or HIPPO force field.

    Atom types and classes keep their integer labels as strings; torsion
    amplitudes are multiplied by the file's ``torsionunit`` (and pi-torsion /
    torsion-torsion values by theirs) so the library holds the energy
    coefficients; ``biotype`` rows and keywords the model does not use are
    recorded in ``notes``.

    Parameters
    ----------
    path : str or Path
        The ``.prm`` file.

    Returns
    -------
    AmoebaLibrary
        The library (without typing templates).
    """
    lines = Path(path).read_text(errors="replace").splitlines()
    lib = AmoebaLibrary(name=Path(path).stem)
    settings = dict(DEFAULT_SETTINGS)
    ignored: dict[str, int] = {}
    torsion_rows: list[tuple[tuple, list]] = []
    pitors_rows: list[tuple[tuple, float]] = []
    tortor_rows: list = []
    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i]
        i += 1
        w = _tokens(raw)
        if not w:
            continue
        kw = w[0].lower()
        if kw in DEFAULT_SETTINGS:
            if kw in _FLAG_SETTINGS:
                settings[kw] = True
            elif kw in _TEXT_SETTINGS:
                settings[kw] = w[1] if len(w) > 1 else ""
            else:
                settings[kw] = float(w[1])
            continue
        if kw == "atom":
            # atom  type class symbol "description" Z mass connections
            text = raw
            desc = ""
            if '"' in text:
                desc = text.split('"')[1].strip()
                rest = text.split('"')[0].split() + text.split('"')[2].split()
            else:
                rest = w
            t, c, symbol = rest[1], rest[2], rest[3]
            z = int(float(rest[4])) if len(rest) > 4 else SYMBOL_TO_Z.get(symbol, 0)
            lib.atom_types[t] = {"class": c, "element": z,
                                 "mass": float(rest[5]) if len(rest) > 5 else 0.0,
                                 "connections": int(float(rest[6])) if len(rest) > 6 else 0,
                                 "comment": desc}
        elif kw == "vdw":
            lib.vdw_types[w[1]] = {"rmin": float(w[2]), "eps": float(w[3]),
                                   "reduction": float(w[4]) if len(w) > 4 else 1.0}
        elif kw == "vdwpr":
            lib.vdw_pair_types[_fmt_key(_canonical(w[1:3], "like_bond")[0])] = {
                "rmin": float(w[3]), "eps": float(w[4])}
        elif kw == "bond":
            lib.bond_types[_fmt_key(_canonical(w[1:3], "like_bond")[0])] = {
                "k": float(w[3]), "r0": float(w[4])}
        elif kw in ("angle", "anglep"):
            key, _ = _canonical(w[1:4], "like_angle")
            table = lib.angle_inplane_types if kw == "anglep" else lib.angle_types
            table[_fmt_key(key)] = {"k": float(w[4]), "theta0": [float(x) for x in w[5:8]]}
        elif kw == "strbnd":
            key, flipped = _canonical(w[1:4], "like_angle")
            k1, k2 = float(w[4]), float(w[5])
            lib.strbnd_types[_fmt_key(key)] = {"k1": k2 if flipped else k1, "k2": k1 if flipped else k2}
        elif kw == "ureybrad":
            lib.urey_types[_fmt_key(_canonical(w[1:4], "like_angle")[0])] = {
                "k": float(w[4]), "r0": float(w[5])}
        elif kw == "opbend":
            d, b, a, c = w[1:5]
            a, c = sorted((a, c), key=lambda x: _fmt_key((x,)))
            lib.opbend_types[_fmt_key((d, b, a, c))] = {"k": float(w[5])}
        elif kw == "torsion":
            key, _ = _canonical(w[1:5], "like_torsion")
            vals = w[5:]
            terms = []
            for m in range(0, len(vals) - 2, 3):
                terms.append([float(vals[m]), int(float(vals[m + 2])), float(vals[m + 1])])
            torsion_rows.append((key, terms))
        elif kw == "pitors":
            pitors_rows.append((_canonical(w[1:3], "like_bond")[0], float(w[3])))
        elif kw == "tortors":
            classes = tuple(w[1:6])
            nx, ny = int(w[6]), int(w[7])
            grid = []
            while len(grid) < nx * ny and i < n:
                g = _tokens(lines[i])
                i += 1
                if len(g) >= 3:
                    grid.append((float(g[0]), float(g[1]), float(g[2])))
            tortor_rows.append((classes, nx, ny, grid))
        elif kw == "multipole":
            frame = w[2:-1] if len(w) > 2 else []
            q = float(w[-1])
            kz = frame[0] if len(frame) > 0 else "0"
            kx = frame[1] if len(frame) > 1 else "0"
            ky = frame[2] if len(frame) > 2 else "0"
            d = [float(x) for x in _tokens(lines[i])[:3]]
            q1 = [float(x) for x in _tokens(lines[i + 1])[:1]]
            q2 = [float(x) for x in _tokens(lines[i + 2])[:2]]
            q3 = [float(x) for x in _tokens(lines[i + 3])[:3]]
            i += 4
            Q = [[q1[0], q2[0], q3[0]], [q2[0], q2[1], q3[1]], [q3[0], q3[1], q3[2]]]
            lib.multipoles.setdefault(w[1], []).append(
                {"frame": (kz, kx, ky), "q": q, "dipole": d, "quadrupole": Q})
        elif kw == "polarize":
            reals, ints = [], []
            for tok in w[2:]:
                if "." in tok or "e" in tok.lower():
                    reals.append(float(tok))
                else:
                    ints.append(tok)
            alpha = reals[0] if reals else 0.0
            thole = reals[1] if len(reals) > 1 else 0.0
            lib.polarize[w[1]] = {"alpha": alpha, "thole": thole, "groups": ints}
        elif kw == "chgpen":
            lib.chgpen[w[1]] = {"core": float(w[2]), "alpha": float(w[3])}
        elif kw == "chgtrn":
            lib.chgtrn[w[1]] = {"eps": float(w[2]), "alpha": float(w[3])}
        elif kw == "repulsion":
            lib.repulsion[w[1]] = {"k": float(w[2]), "alpha": float(w[3]), "valence": float(w[4])}
        elif kw == "dispersion":
            lib.dispersion[w[1]] = {"c6": float(w[2]), "alpha": float(w[3])}
        elif kw == "exchpol":
            # class, spring constant, size, damping exponent, whether the class is damped
            lib.exchpol[w[1]] = {"spring": float(w[2]), "size": float(w[3]), "damp": float(w[4]),
                                 "use": len(w) < 6 or int(float(w[5])) != 0}
        elif kw == "bndcflux":
            key, flipped = _canonical(w[1:3], "like_bond")
            j = float(w[3])
            lib.bndcflux[_fmt_key(key)] = {"j": -j if flipped else j}
        elif kw == "angcflux":
            key, flipped = _canonical(w[1:4], "like_angle")
            jt1, jt2, jb1, jb2 = (float(x) for x in w[4:8])
            if flipped:
                jt1, jt2, jb1, jb2 = jt2, jt1, jb2, jb1
            lib.angcflux[_fmt_key(key)] = {"jt1": jt1, "jt2": jt2, "jb1": jb1, "jb2": jb2}
        elif kw in ("biotype", "forcefield", "parameters"):
            ignored[kw] = ignored.get(kw, 0) + 1
        elif len(w) > 1 and _looks_like_keyword(w):
            ignored[kw] = ignored.get(kw, 0) + 1
    tu = float(settings["torsionunit"])
    for key, terms in torsion_rows:
        lib.torsion_types[_fmt_key(key)] = {"terms": [[k * tu, nn, ph] for k, nn, ph in terms]}
    pu = float(settings["pitorsunit"])
    for key, k in pitors_rows:
        lib.pitors_types[_fmt_key(key)] = {"k": k * pu}
    ttu = float(settings["tortorunit"])
    for classes, nx, ny, grid in tortor_rows:
        a1 = sorted({g[0] for g in grid})
        a2 = sorted({g[1] for g in grid})
        table = [[0.0] * ny for _ in range(nx)]
        for x, y, v in grid:
            table[a1.index(x)][a2.index(y)] = v * ttu
        lib.tortor_types[_fmt_key(classes)] = {"angles1": a1, "angles2": a2, "values": table}
    lib.settings = settings
    lib.name = str(settings.get("forcefield") or lib.name)
    for kw, count in sorted(ignored.items()):
        lib.notes.append(f"ignored keyword {kw} ({count} rows)")
    lib.references = [f"Read from the Tinker parameter file {Path(path).name}"]
    return lib


def _looks_like_keyword(w: list[str]) -> bool:
    """A parameter-file line (keyword followed by numbers), not prose."""
    try:
        float(w[1])
        return True
    except ValueError:
        return w[0].isupper() or w[0].islower() and len(w) <= 3


def _prm_number(x: float, width: int = 10, decimals: int = 5) -> str:
    return f"{x:{width}.{decimals}f}"


def _banner(title: str) -> list[str]:
    """The five-line section banner of the Tinker parameter files; the LAMMPS
    reader recognises a section by this exact shape (bar, blank frame, title,
    blank frame, bar)."""
    width = len(title) + 8
    bar = "      " + "#" * width
    frame = "      ##" + " " * (width - 4) + "##"
    return ["", bar, frame, f"      ##  {title}  ##", frame, bar, ""]


def _own_numbers(names, kind: str) -> dict:
    """``{name: int(name)}``, checked to be positive and distinct."""
    out = {}
    for name in names:
        try:
            number = int(str(name))
        except ValueError:
            raise ValueError(f"atom {kind} {name!r} is not an integer; write with keep_numbers=False") from None
        if number < 1:
            raise ValueError(f"atom {kind} {name!r} is not a positive integer")
        out[name] = number
    if len(set(out.values())) != len(out):
        raise ValueError(f"two atom {kind} names give the same number")
    return out


def write_tinker_prm(lib: AmoebaLibrary, path: Union[str, Path],
                     title: Optional[str] = None, keep_numbers: bool = False) -> dict:
    """Write a library as a Tinker parameter file.

    Atom types and classes are renumbered 1..N in library order (Tinker
    needs integers), or keep their own numbers with ``keep_numbers``; the
    mapping is returned so coordinate files can be written with matching
    type numbers. Torsion amplitudes are written divided by
    ``torsionunit`` again. The file is read by Tinker and by the LAMMPS
    AMOEBA package.

    Parameters
    ----------
    lib : AmoebaLibrary
        The library.
    path : str or Path
        Output path.
    title : str, optional
        The ``forcefield`` name written; default the library's setting.
    keep_numbers : bool, optional
        Write the type and class names as they are, for a library whose
        names are Tinker's integers (one read from a ``.prm`` file), so
        existing coordinate files stay valid.

    Returns
    -------
    dict
        ``{"types": {name: int}, "classes": {name: int}}``.

    Raises
    ------
    ValueError
        With ``keep_numbers``, when a type or class name is not a positive
        integer or two names give the same number.
    """
    if keep_numbers:
        types = _own_numbers(lib.atom_types, "type")
        classes = _own_numbers(lib.classes(), "class")
    else:
        types = {t: i + 1 for i, t in enumerate(lib.atom_types)}
        classes = {c: i + 1 for i, c in enumerate(lib.classes())}
    s = lib.settings

    def cl(name: str) -> str:
        return "0" if name in (WILDCARD, "0") else str(classes[name])

    def ty(name: str) -> str:
        name = str(name)
        if name in ("0", ""):
            return "0"
        if name.startswith("-"):
            return "-" + str(types[name[1:]])
        return str(types[name])

    out = ["", *_banner("Force Field Definition"),
           f"forcefield              {title or s.get('forcefield', lib.name)}", ""]
    for key in ("bond-cubic", "bond-quartic", "angle-cubic", "angle-quartic", "angle-pentic",
                "angle-sextic", "opbendtype", "opbend-cubic", "opbend-quartic", "opbend-pentic",
                "opbend-sextic", "torsionunit", "pitorsunit", "tortorunit", "vdwtype", "radiusrule",
                "radiustype", "radiussize", "epsilonrule", "dielectric", "polarization"):
        value = lib.setting(key)
        if lib.model == "hippo" and key in ("vdwtype", "radiusrule", "radiustype", "radiussize", "epsilonrule"):
            continue
        if key in ("pitorsunit", "tortorunit") and float(value) == 1.0:
            continue
        out.append(f"{key:<24}{value}")
    if lib.setting("d-equals-p"):
        out.append("d-equals-p")
    for term in ("vdw", "rep", "disp", "mpole", "polar"):
        if term == "vdw" and lib.model == "hippo":
            continue
        if term in ("rep", "disp") and lib.model != "hippo":
            continue
        for sep in (2, 3, 4, 5):
            out.append(f"{term}-1{sep}-scale{'':<10}{lib.scale(term, sep)}")
    for sep in (2, 3, 4, 5):
        out.append(f"polar-1{sep}-intra{'':<10}{lib.setting(f'polar-1{sep}-intra')}")
    for term in ("direct", "mutual"):
        for sep in (1, 2, 3, 4):
            out.append(f"{term}-1{sep}-scale{'':<9}{lib.scale(term, sep)}")
    if lib.model == "hippo":
        for sep in (2, 3, 4, 5):
            out.append(f"induce-1{sep}-scale{'':<9}{lib.scale('induce', sep)}")
    out += [*_banner("Literature References")]
    for ref in lib.references:
        # as comments: both Tinker and LAMMPS parse every line by its first word, and a
        # reference line may begin with a keyword such as "repulsion"
        out += ["!! " + ln for ln in str(ref).splitlines()]
        out.append("")
    out += [*_banner("Atom Type Definitions")]
    for t, a in lib.atom_types.items():
        z = int(a.get("element", 0))
        symbol = CHEMICAL_SYMBOLS[z] if 0 < z < len(CHEMICAL_SYMBOLS) else "X"
        desc = '"' + str(a.get("comment", t))[:30] + '"'
        out.append(f"atom {types[t]:10d} {classes[str(a['class'])]:4d}    {symbol:<4} "
                   f"{desc:<32} {z:4d} {float(a.get('mass', 0.0)):12.6f} "
                   f"{int(a.get('connections', 0) or 0):5d}")
    out.append("")
    if lib.vdw_types:
        out += [*_banner("Van der Waals Parameters")]
        for c, v in lib.vdw_types.items():
            red = float(v.get("reduction", 1.0))
            line = f"vdw {classes[c]:11d} {'':>13} {v['rmin']:14.8f} {v['eps']:14.8f}"
            out.append(line + (f" {red:12.6f}" if red != 1.0 else ""))
        if lib.vdw_pair_types:
            out += _banner("Van der Waals Pair Parameters")
            for k, v in lib.vdw_pair_types.items():
                a, b = _split_key(k)
                out.append(f"vdwpr {classes[a]:9d} {classes[b]:4d} {'':>8} {v['rmin']:14.8f} {v['eps']:14.8f}")
        out.append("")
    if lib.bond_types:
        out += [*_banner("Bond Stretching Parameters")]
        for k, v in lib.bond_types.items():
            a, b = _split_key(k)
            out.append(f"bond {classes[a]:10d} {classes[b]:4d} {'':>9} {v['k']:14.8f} {v['r0']:14.8f}")
        out.append("")
    for kw, table in (("angle", lib.angle_types), ("anglep", lib.angle_inplane_types)):
        if table:
            out += [*_banner("Angle Bending Parameters")]
            for k, v in table.items():
                a, b, c = _split_key(k)
                thetas = " ".join(f"{t:14.8f}" for t in v["theta0"])
                out.append(f"{kw:<6} {classes[a]:8d} {classes[b]:4d} {classes[c]:4d} {'':>4} {v['k']:14.8f} {thetas}")
            out.append("")
    if lib.strbnd_types:
        out += [*_banner("Stretch-Bend Parameters")]
        for k, v in lib.strbnd_types.items():
            a, b, c = _split_key(k)
            out.append(f"strbnd {classes[a]:8d} {classes[b]:4d} {classes[c]:4d} {'':>4} "
                       f"{v['k1']:14.8f} {v['k2']:14.8f}")
        out.append("")
    if lib.urey_types:
        out += [*_banner("Urey-Bradley Parameters")]
        for k, v in lib.urey_types.items():
            a, b, c = _split_key(k)
            out.append(f"ureybrad {classes[a]:6d} {classes[b]:4d} {classes[c]:4d} {'':>4} "
                       f"{v['k']:14.8f} {v['r0']:14.8f}")
        out.append("")
    if lib.opbend_types:
        out += [*_banner("Out-of-Plane Bend Parameters")]
        for k, v in lib.opbend_types.items():
            d, b, a, c = _split_key(k)
            out.append(f"opbend {classes[d]:8d} {classes[b]:4d} {cl(a):>4} {cl(c):>4} {'':>4} {v['k']:14.8f}")
        out.append("")
    if lib.torsion_types:
        tu = float(lib.setting("torsionunit"))
        out += [*_banner("Torsional Parameters")]
        for k, v in lib.torsion_types.items():
            a, b, c, d = _split_key(k)
            terms = " ".join(f"{K / tu:12.6f} {ph:8.2f} {nn:2d}" for K, nn, ph in v["terms"])
            out.append(f"torsion {cl(a):>6} {cl(b):>4} {cl(c):>4} {cl(d):>4} {terms}")
        out.append("")
    if lib.pitors_types:
        pu = float(lib.setting("pitorsunit"))
        out += [*_banner("Pi-Torsion Parameters")]
        for k, v in lib.pitors_types.items():
            a, b = _split_key(k)
            out.append(f"pitors {classes[a]:8d} {classes[b]:4d} {'':>9} {v['k'] / pu:14.8f}")
        out.append("")
    if lib.tortor_types:
        ttu = float(lib.setting("tortorunit"))
        out += [*_banner("Torsion-Torsion Parameters")]
        for k, v in lib.tortor_types.items():
            cls5 = [cl(x) for x in _split_key(k)]
            nx, ny = len(v["angles1"]), len(v["angles2"])
            out.append(f"tortors {' '.join(f'{c:>4}' for c in cls5)} {nx:8d} {ny:4d}")
            for ix, x in enumerate(v["angles1"]):
                for iy, y in enumerate(v["angles2"]):
                    out.append(f" {x:8.1f} {y:8.1f} {v['values'][ix][iy] / ttu:14.8f}")
        out.append("")
    if lib.multipoles:
        out += [*_banner("Atomic Multipole Parameters")]
        for t, entries in lib.multipoles.items():
            for e in entries:
                kz, kx, ky = e["frame"]
                frame = [ty(kz), ty(kx)] + ([ty(ky)] if str(ky) not in ("0", "") else [])
                Q = e["quadrupole"]
                out.append(f"multipole {types[t]:5d} {' '.join(f'{f:>4}' for f in frame):<15} {e['q']:16.8f}")
                out.append(f"{'':>39}{e['dipole'][0]:14.8f} {e['dipole'][1]:14.8f} {e['dipole'][2]:14.8f}")
                out.append(f"{'':>39}{Q[0][0]:14.8f}")
                out.append(f"{'':>39}{Q[0][1]:14.8f} {Q[1][1]:14.8f}")
                out.append(f"{'':>39}{Q[0][2]:14.8f} {Q[1][2]:14.8f} {Q[2][2]:14.8f}")
        out.append("")
    if lib.polarize:
        out += [*_banner("Dipole Polarizability Parameters")]
        for t, v in lib.polarize.items():
            groups = " ".join(str(types[g]) for g in v.get("groups", []) if g in types)
            keep_thole = lib.model == "amoeba" or float(v.get("thole", 0.0)) != 0.0
            thole = f"{v['thole']:14.8f}" if keep_thole else "          "
            out.append(f"polarize {types[t]:7d} {'':>7} {v['alpha']:14.8f} {thole} {groups}")
        out.append("")
    for kw, title, table, fmt in (
            ("chgpen", "Charge Penetration Parameters", lib.chgpen,
             lambda v: f"{v['core']:14.8f} {v['alpha']:14.8f}"),
            ("repulsion", "Pauli Repulsion Parameters", lib.repulsion,
             lambda v: f"{v['k']:14.8f} {v['alpha']:14.8f} {v['valence']:14.8f}"),
            ("dispersion", "Dispersion Parameters", lib.dispersion,
             lambda v: f"{v['c6']:14.8f} {v['alpha']:14.8f}"),
            ("chgtrn", "Charge Transfer Parameters", lib.chgtrn,
             lambda v: f"{v['eps']:14.8f} {v['alpha']:14.8f}"),
            ("exchpol", "Exchange Polarization Parameters", lib.exchpol,
             lambda v: f"{v['spring']:14.8f} {v['size']:14.8f} {v['damp']:14.8f} {1 if v['use'] else 0:3d}")):
        if table:
            # one banner per block: the LAMMPS reader parses the file by section titles
            out += _banner(title)
            for c, v in table.items():
                out.append(f"{kw:<10} {classes[c]:7d} {'':>7} {fmt(v)}")
    if lib.bndcflux or lib.angcflux:
        out += [*_banner("Charge Flux Parameters")]
        for k, v in lib.bndcflux.items():
            a, b = _split_key(k)
            out.append(f"bndcflux {classes[a]:6d} {classes[b]:4d} {'':>9} {v['j']:14.8f}")
        for k, v in lib.angcflux.items():
            a, b, c = _split_key(k)
            out.append(f"angcflux {classes[a]:6d} {classes[b]:4d} {classes[c]:4d} {'':>4} "
                       f"{v['jt1']:14.8f} {v['jt2']:14.8f} {v['jb1']:14.8f} {v['jb2']:14.8f}")
    out.append("")
    Path(path).write_text("\n".join(out), encoding="utf-8")
    return {"types": types, "classes": classes}


# Tinker coordinate files
def read_tinker_xyz(path: Union[str, Path], library: Optional[AmoebaLibrary] = None,
                    type_map: Optional[dict] = None):
    """Read a Tinker ``.xyz`` coordinate file (atoms with types and bonds).

    Parameters
    ----------
    path : str or Path
        The file: a title line (atom count), an optional cell line, then one
        line per atom with index, name, x, y, z, atom type and the indices of
        the bonded atoms.
    library : AmoebaLibrary, optional
        Used to translate the file's integer types to the library's type
        names (when the library keeps Tinker's integers, as a library read
        from a ``.prm`` file does) and to assign the elements.
    type_map : dict, optional
        ``{integer type: type name}`` translation (as returned by
        :func:`write_tinker_prm`, inverted), taking precedence over the
        library's own names.

    Returns
    -------
    dict
        ``positions`` ``(N, 3)`` in Angstrom (a list of lists), ``symbols``,
        ``atomic_numbers``, ``types`` (names), ``bonds`` ``(i, j)`` pairs
        with ``i < j``, ``cell`` (``None`` or the six lattice parameters) and
        ``title``.
    """
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
    head = lines[0].split()
    natoms = int(head[0])
    title = " ".join(head[1:])
    start = 1
    cell = None
    second = lines[1].split()
    if len(second) == 6 and all(_is_float(x) for x in second):
        cell = [float(x) for x in second]
        start = 2
    positions, names, types, bonds = [], [], [], set()
    for ln in lines[start:start + natoms]:
        w = ln.split()
        idx = int(w[0]) - 1
        names.append(w[1])
        positions.append([float(w[2]), float(w[3]), float(w[4])])
        types.append(w[5])
        for b in w[6:]:
            j = int(b) - 1
            bonds.add((min(idx, j), max(idx, j)))
    inv = dict(type_map or {})
    type_names = [inv.get(int(t), inv.get(t, t)) for t in types]
    symbols = []
    for name, t in zip(names, type_names):
        z = None
        if library is not None and t in library.atom_types:
            z = int(library.atom_types[t]["element"])
        if not z:
            letters = "".join(ch for ch in name if ch.isalpha())
            for cand in (letters[:2].capitalize(), letters[:1].upper()):
                if cand in SYMBOL_TO_Z:
                    z = SYMBOL_TO_Z[cand]
                    break
        symbols.append(CHEMICAL_SYMBOLS[z] if z else "X")
    return {"positions": positions, "symbols": symbols,
            "atomic_numbers": [SYMBOL_TO_Z.get(s, 0) for s in symbols],
            "types": type_names, "names": names, "bonds": sorted(bonds), "cell": cell,
            "title": title}


def write_tinker_xyz(path: Union[str, Path], positions, types: Sequence[str],
                     bonds: Sequence[Sequence[int]], type_numbers: dict,
                     symbols: Optional[Sequence[str]] = None, title: str = "xnn",
                     cell: Optional[Sequence[float]] = None) -> Path:
    """Write a Tinker ``.xyz`` coordinate file.

    Parameters
    ----------
    path : str or Path
        Output path.
    positions : array_like
        Cartesian coordinates ``(N, 3)`` in Angstrom.
    types : sequence of str
        Per-atom type names of the library.
    bonds : sequence of (int, int)
        The connectivity.
    type_numbers : dict
        ``{type name: int}`` as returned by :func:`write_tinker_prm`.
    symbols : sequence of str, optional
        Atom names (element symbols by default).
    title : str, optional
        Title written after the atom count.
    cell : sequence of float, optional
        The six lattice parameters ``a b c alpha beta gamma`` for a
        periodic system.

    Returns
    -------
    Path
        The written file.
    """
    n = len(types)
    nb: list[list[int]] = [[] for _ in range(n)]
    for i, j in bonds:
        nb[int(i)].append(int(j) + 1)
        nb[int(j)].append(int(i) + 1)
    lines = [f"{n:6d}  {title}"]
    if cell is not None:
        lines.append(" ".join(f"{float(x):12.6f}" for x in cell))
    for i in range(n):
        p = positions[i]
        name = symbols[i] if symbols is not None else "X"
        lines.append(f"{i + 1:6d}  {name:<3} {float(p[0]):16.10f} {float(p[1]):16.10f} {float(p[2]):16.10f} "
                     f"{type_numbers[types[i]]:5d} " + " ".join(f"{b:5d}" for b in sorted(nb[i])))
    out = Path(path)
    out.write_text("\n".join(lines) + "\n")
    return out


def _is_float(x: str) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        return False


__all__ = ["AmoebaLibrary", "DEFAULT_SETTINGS", "MAX_TORSION_TERMS", "from_forcefield",
           "to_forcefield", "read_amoeba", "builtin_library", "read_tinker_prm",
           "write_tinker_prm", "read_tinker_xyz", "write_tinker_xyz", "KCAL_TO_EV"]
