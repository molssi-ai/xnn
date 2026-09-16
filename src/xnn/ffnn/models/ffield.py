"""ReaxFF parameter libraries: the SEAMM ``.frc`` format and ReaxFF-nn JSON.

A ReaxFF model is fully specified by its parameter library. Two on-disk
formats are supported:

* the MolSSI/SEAMM ``.frc`` force-field format (:mod:`xnn.ffnn.common.frc`),
  in which the standard ReaxFF parameter blocks appear as named sections
  (``#reaxff_general_parameters``, ``#reaxff_atomic_parameters_1-8`` ...
  ``#reaxff_hydrogen-bond_parameters``) with every parameter identified by
  name rather than by column position. Published fields translated to this
  format ship with xnn (``ReaxFF("CHO_cho_2008")``; see
  :func:`~xnn.ffnn.common.frc.list_forcefields`);
* the JSON parameter-library format used by ReaxFF-nn parameter sets
  (Guo et al., *Comput. Mater. Sci.* 172, 109393, 2020; Xue et al., *PCCP*
  23, 19457, 2021), which stores the same parameters as a flat
  ``"<name>_<type>"`` dictionary plus the neural-network weight matrices and
  the function/layer selectors. This is also the format
  :meth:`FFieldLibrary.save` writes, since network weights have no place in
  a ``.frc`` file.

Parameters are kept in the flat naming convention of ReaxFF libraries
(``"Desi_C-C"``, ``"val_C"``, ``"theta0_H-C-H"``, ...) and in the file's
units (kcal/mol where applicable -- the model converts to eV when it
assembles its tensors). :func:`read_ffield` returns a :class:`FFieldLibrary`;
the completion helpers below apply the same conventions ReaxFF codes apply
after reading a library (off-diagonal combination rules, hydrogen-bond
defaults, torsion wildcard resolution).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union


from ..common.elements import CHEMICAL_SYMBOLS, SYMBOL_TO_Z  # noqa: F401  (re-exported)


# Layout of the standard `ffield` text library (column -> parameter name).
# This is the published file format shared by every ReaxFF code; "n.u." marks
# columns that are unused in the current functional form.
GENERAL_PARAMS = (
    "boc1", "boc2", "coa2", "trip4", "trip3", "kc2", "ovun6", "trip2",
    "ovun7", "ovun8", "trip1", "swa", "swb", "n.u.", "val6", "lp1",
    "val9", "val10", "n.u.", "pen2", "pen3", "pen4", "n.u.", "tor2",
    "tor3", "tor4", "n.u.", "cot2", "vdw1", "cutoff", "coa4", "ovun4",
    "ovun3", "val8", "acut", "hbtol", "n.u.", "n.u.", "coa3",
)

SPECIES_LINES = (
    ("rosi", "val", "mass", "rvdw", "Devdw", "gamma", "ropi", "vale"),
    ("alfa", "gammaw", "valang", "ovun5", "n.u.", "chi", "mu", "hbond"),
    ("ropp", "lp2", "n.u.", "boc4", "boc3", "boc5", "n.u.", "n.u."),
    ("ovun2", "val3", "n.u.", "valboc", "val5", "n.u.", "n.u.", "atomic"),
)

BOND_LINES = (
    ("Desi", "Depi", "Depp", "be1", "bo5", "corr13", "bo6", "ovun1"),
    ("be2", "bo3", "bo4", "n.u.", "bo1", "bo2", "ovcorr", "n.u."),
)

OFFDIAG_PARAMS = ("Devdw", "rvdw", "alfa", "rosi", "ropi", "ropp")
ANGLE_PARAMS = ("theta0", "val1", "val2", "coa1", "val7", "pen1", "val4")
TORSION_PARAMS = ("V1", "V2", "V3", "tor1", "cot1", "n.u.", "n.u.")
HBOND_PARAMS = ("rohb", "Dehb", "hb1", "hb2")

# Per-type parameter names looked up per valence angle / torsion / H-bond.
P_ANGLE = ("theta0", "val1", "val2", "coa1", "val7", "val4", "pen1")
P_TORSION = ("V1", "V2", "V3", "tor1", "cot1")
P_HBOND = ("rohb", "Dehb", "hb1", "hb2")

# Fallback cutoffs used when a library does not carry its own tables: the
# bond-order cutoff `rcut` should reach past the first coordination shell,
# the valence cutoff `rcuta` only up to it (angle/torsion/H-bond
# enumeration). The heuristic below (tighter windows for pairs involving
# hydrogen) covers organic elements; libraries for other chemistries should
# ship their own `rcut` / `rcutBond` tables.
def default_pair_cutoff(a: str, b: str, kind: str) -> float:
    """Heuristic fallback cutoff for a species pair, in Angstrom.

    Parameters
    ----------
    a, b : str
        Chemical symbols of the pair.
    kind : str
        ``"rcut"`` for the bond-order cutoff or ``"rcuta"`` for the valence
        cutoff.

    Returns
    -------
    float
        The fallback cutoff.
    """
    n_h = int(a == "H") + int(b == "H")
    if kind == "rcut":
        return 2.0 if n_h >= 1 else 2.5
    return {0: 1.95, 1: 1.75, 2: 1.35}[n_h]


@dataclass
class FFieldLibrary:
    """A parsed ReaxFF parameter library.

    Parameters and attributes
    -------------------------
    p : dict[str, float]
        Flat parameter dictionary in the ReaxFF naming convention:
        general parameters by bare name (``"boc1"``), per-species as
        ``"<name>_<El>"``, per-bond as ``"<name>_<El>-<El>"``, valence angles
        as ``"<name>_<El>-<El>-<El>"``, torsions (``X`` wildcards allowed) and
        hydrogen bonds likewise. Values are in the file's units (energies in
        kcal/mol).
    m : dict or None
        ReaxFF-nn weight matrices (``"fmwi_C"``, ``"few_C-H"``, ...) as nested
        lists, or ``None`` / empty when the library is a classical ReaxFF.
    spec : list[str]
        Chemical symbols, in library order.
    bonds : list[str]
        Bond types (``"C-H"``) with bond parameters, in library order.
    offd : list[str]
        Off-diagonal (unlike-pair) types with explicit vdW/bond radii.
    angs : list[str]
        Valence-angle types with parameters.
    torp : list[str]
        Torsion types with parameters (may contain ``X`` wildcards).
    hbs : list[str]
        Hydrogen-bond triples ``"X-H-Z"`` with parameters.
    messages : int
        Number of ReaxFF-nn message-passing steps ``T``.
    bo_function, energy_function, message_function, vdw_function : int
        ReaxFF-nn function-form selectors (see :class:`~xnn.ffnn.models.reaxff.ReaxFF`).
    bo_layer, mf_layer, be_layer, vdw_layer : tuple or None
        ``(width, n_hidden)`` of the bond-order / message / bond-energy / vdW
        neural networks.
    rcut, rcuta : dict[str, float] or None
        Per-bond-type bond-order and valence cutoff tables (with an
        ``"others"`` fallback), or ``None`` to use the defaults.
    mol_energy : dict[str, float]
        Per-molecule reference-energy offsets some ReaxFF-nn training
        workflows record (not used by the model; kept for round-tripping).
    heat_increment : dict[str, float]
        The per-species atomic heat increments (``Hat``, species line 3,
        column 3 of the classical layout) as read from a ``.frc`` file. Kept
        for round-tripping only: ReaxFF MD codes (LAMMPS) do not add them to
        the energy, and neither does :class:`~xnn.ffnn.models.reaxff.ReaxFF`.
    name : str
        The force-field name (the ``#define`` of a ``.frc`` file).
    references : list[str]
        Provenance text of the parameters, when the file carries it.
    """

    p: dict
    m: Optional[dict] = None
    spec: list = field(default_factory=list)
    bonds: list = field(default_factory=list)
    offd: list = field(default_factory=list)
    angs: list = field(default_factory=list)
    torp: list = field(default_factory=list)
    hbs: list = field(default_factory=list)
    messages: int = 1
    bo_function: int = 0
    energy_function: int = 0
    message_function: int = 0
    vdw_function: int = 0
    bo_layer: Optional[tuple] = None
    mf_layer: Optional[tuple] = None
    be_layer: Optional[tuple] = None
    vdw_layer: Optional[tuple] = None
    rcut: Optional[dict] = None
    rcuta: Optional[dict] = None
    mol_energy: dict = field(default_factory=dict)
    heat_increment: dict = field(default_factory=dict)
    name: str = ""
    references: list = field(default_factory=list)

    @property
    def is_nn(self) -> bool:
        """bool : Whether the library carries ReaxFF-nn network weights."""
        return bool(self.m)

    def save(self, path) -> None:
        """Write the library to disk in the ReaxFF-nn JSON format.

        The file can be read back with :func:`read_ffield` (and by other
        tools that consume the same format).

        Parameters
        ----------
        path : str or Path
            Destination path (conventionally ``ffield.json``).
        """
        def tup(x):
            """JSON-encode a layer spec (tuple or None)."""
            return list(x) if x is not None else None

        payload = {
            "p": {k: float(v) for k, v in self.p.items()},
            "m": self.m,
            "MolEnergy": self.mol_energy,
            "messages": self.messages,
            "BOFunction": self.bo_function,
            "EnergyFunction": self.energy_function,
            "MessageFunction": self.message_function,
            "VdwFunction": self.vdw_function,
            "bo_layer": tup(self.bo_layer),
            "mf_layer": tup(self.mf_layer),
            "be_layer": tup(self.be_layer),
            "vdw_layer": tup(self.vdw_layer),
            "rcut": self.rcut,
            "rcutBond": self.rcuta,
            "rEquilibrium": None,
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, sort_keys=True, indent=2)


def _scan_types(p: dict) -> tuple[list, list, list, list, list, list]:
    """Recover the species/bond/angle/torsion/H-bond type lists from ``p``.

    The JSON format stores only the flat parameter dictionary; the typed
    lists are implied by which keys exist (``bo1_*`` for bonds, single-species
    ``rosi_*`` for species, ``theta0_*`` for angles, ``tor1_*`` for torsions,
    ``rohb_*`` for hydrogen bonds).

    Parameters
    ----------
    p : dict
        Flat parameter dictionary.

    Returns
    -------
    tuple of lists
        ``(spec, bonds, offd, angs, torp, hbs)``.
    """
    spec, bonds, offd, angs, torp, hbs = [], [], [], [], [], []
    for key in p:
        head, _, tail = key.partition("_")
        if not tail:
            continue
        if head == "bo1":
            bonds.append(tail)
        elif head == "rosi":
            parts = tail.split("-")
            if len(parts) == 1:
                spec.append(tail)
            elif parts[0] != parts[1]:
                offd.append(tail)
        elif head == "theta0":
            angs.append(tail)
        elif head == "tor1":
            torp.append(tail)
        elif head == "rohb":
            hbs.append(tail)
    return spec, bonds, offd, angs, torp, hbs


def dedup_torsion_types(torp: list) -> list:
    """Drop torsion types that are reversed spellings of an earlier entry.

    A torsion ``i-j-k-l`` is the same type as ``l-k-j-i`` (the full
    reversal). Swapping only the two central atoms (``i-k-j-l``) is *not* an
    equivalence -- it changes the central bond -- and published fields list
    such pairs (``C-O-C-H`` and ``H-O-C-C``) with different parameters, so
    both are kept.

    Parameters
    ----------
    torp : list[str]
        Torsion type names, possibly containing duplicates.

    Returns
    -------
    list[str]
        The deduplicated list (order preserved).
    """
    kept: list = []
    for tor in torp:
        t1, t2, t3, t4 = tor.split("-")
        rev = f"{t4}-{t3}-{t2}-{t1}"
        if tor not in kept and rev not in kept:
            kept.append(tor)
    return kept


def complete_off_diagonal(p: dict, spec: list, bonds: list) -> None:
    """Fill missing off-diagonal pair parameters in place.

    Applies the standard ReaxFF conventions: like pairs inherit the atomic
    value; unlike pairs without an explicit off-diagonal entry get the
    geometric-mean combination rule (or ``-1`` when either atomic value is
    non-positive); and negative pi / double-pi bond radii -- the ffield
    convention for "this pair has no pi bond" -- are replaced by a fraction of
    the sigma radius while the corresponding bond-order exponentials are
    switched off (``bo3/bo5 = -50``, ``bo4/bo6 = 0``).

    Parameters
    ----------
    p : dict
        Flat parameter dictionary, modified in place.
    spec : list[str]
        Species symbols.
    bonds : list[str]
        Bond types.
    """
    keys = ("Devdw", "rvdw", "alfa", "rosi", "ropi", "ropp")
    for key in keys:
        for sp in spec:
            if f"{key}_{sp}" in p:
                p.setdefault(f"{key}_{sp}-{sp}", p[f"{key}_{sp}"])
    for bd in bonds:
        a, b = bd.split("-")
        if f"rvdw_{bd}" not in p:
            for key in keys:
                va, vb = p[f"{key}_{a}"], p[f"{key}_{b}"]
                p[f"{key}_{bd}"] = math.sqrt(va * vb) if va > 0.0 and vb > 0.0 else -1.0
    for bd in bonds:
        if p[f"ropi_{bd}"] < 0.0:
            p[f"ropi_{bd}"] = 0.3 * p[f"rosi_{bd}"]
            p[f"bo3_{bd}"] = -50.0
            p[f"bo4_{bd}"] = 0.0
        if p[f"ropp_{bd}"] < 0.0:
            p[f"ropp_{bd}"] = 0.2 * p[f"rosi_{bd}"]
            p[f"bo5_{bd}"] = -50.0
            p[f"bo6_{bd}"] = 0.0


def complete_hbonds(p: dict, spec: list, hbs: list) -> None:
    """Add placeholder parameters for unlisted ``X-H-Z`` hydrogen-bond triples.

    Every heavy-donor / heavy-acceptor combination not present in the library
    gets an inert entry (``Dehb = 0`` so it contributes no energy).
    ``hbs`` is extended in place.

    Parameters
    ----------
    p : dict
        Flat parameter dictionary, modified in place.
    spec : list[str]
        Species symbols.
    hbs : list[str]
        Hydrogen-bond triples, extended in place.
    """
    if "H" not in spec:
        return
    for sp1 in spec:
        if sp1 == "H":
            continue
        for sp2 in spec:
            if sp2 == "H":
                continue
            hb = f"{sp1}-H-{sp2}"
            if hb not in hbs:
                hbs.append(hb)
                p[f"rohb_{hb}"] = 1.9
                p[f"Dehb_{hb}"] = 0.0
                p[f"hb1_{hb}"] = 2.0
                p[f"hb2_{hb}"] = 19.0


def resolve_torsion(p: dict, torp: list, tor: str, key: str) -> float:
    """Look up a torsion parameter, resolving reversal and wildcards.

    The lookup order is that of ReaxFF codes (LAMMPS ``pair_style reaxff``):
    the exact type, its reversal ``l-k-j-i``, then the ``X-j-k-X`` and
    ``X-k-j-X`` wildcards. As a last resort the central-bond swaps
    ``i-k-j-l`` / ``l-j-k-i`` are tried, so a type that a library spells
    only that way still finds parameters; they never shadow an exact,
    reversed or wildcard match. Unmatched types get ``0.0``.

    Parameters
    ----------
    p : dict
        Flat parameter dictionary.
    torp : list[str]
        Torsion types present in the library.
    tor : str
        The concrete torsion type ``"i-j-k-l"`` to resolve.
    key : str
        Parameter name (one of :data:`P_TORSION`).

    Returns
    -------
    float
        The resolved parameter value.
    """
    if tor in torp:
        return p.get(f"{key}_{tor}", 0.0)
    t1, t2, t3, t4 = tor.split("-")
    for cand in (f"{t4}-{t3}-{t2}-{t1}", f"X-{t2}-{t3}-X", f"X-{t3}-{t2}-X",
                 f"{t1}-{t3}-{t2}-{t4}", f"{t4}-{t2}-{t3}-{t1}"):
        if cand in torp:
            return p.get(f"{key}_{cand}", 0.0)
    return 0.0


def _read_json(path: Path) -> FFieldLibrary:
    """Parse a ReaxFF-nn JSON parameter library.

    Parameters
    ----------
    path : Path
        Path to the JSON file.

    Returns
    -------
    FFieldLibrary
        The parsed library.
    """
    with open(path) as fh:
        j = json.load(fh)
    p = {k: float(v) for k, v in j["p"].items()}
    spec, bonds, offd, angs, torp, hbs = _scan_types(p)
    torp = dedup_torsion_types(torp)

    def tup(x):
        """Coerce a JSON layer spec (list or None) to a tuple or None."""
        return tuple(x) if x is not None else None

    return FFieldLibrary(
        p=p,
        m=j.get("m") or None,
        spec=spec, bonds=bonds, offd=offd, angs=angs, torp=torp, hbs=hbs,
        messages=int(j.get("messages") or 0),
        bo_function=int(j.get("BOFunction") or 0),
        energy_function=int(j.get("EnergyFunction") or 0),
        message_function=int(j.get("MessageFunction") or 0),
        vdw_function=int(j.get("VdwFunction") or 0),
        bo_layer=tup(j.get("bo_layer")),
        mf_layer=tup(j.get("mf_layer")),
        be_layer=tup(j.get("be_layer")),
        vdw_layer=tup(j.get("vdw_layer")),
        rcut=j.get("rcut"),
        rcuta=j.get("rcutBond"),
        mol_energy=j.get("MolEnergy") or {},
    )


# ----------------------------------------------------------------------
# the SEAMM .frc format
# ----------------------------------------------------------------------
# SEAMM parameter names, in the column order of the classical ``ffield``
# layout, so that ``zip(FRC_*, <positional xnn names>)`` is the name map.
# Taken from SEAMM's own translation tables (seamm_ff_util.reaxff.metadata).
FRC_GENERAL = (
    "Pboc,1", "Pboc,2", "Pcoa,2", "Ptrip,4", "Ptrip,3", "kc2", "Povun,6",
    "Ptrip,2", "Povun,7", "Povun,8", "Ptrip,1", "Rtaper,lower", "Rtaper,upper",
    "Pfe1", "Pval,7", "Plp,1", "Pval,9", "Pval,10", "not_used_1", "Ppen,2",
    "Ppen,3", "Ppen,4", "not_used_2", "Ptor,2", "Ptor,3", "Ptor,4",
    "not_used_3", "Pcot,2", "PvdW,1", "BO_cutoff", "Pcoa,4", "Povun,4",
    "Povun,3", "Pval,8", "not_used_4", "not_used_5", "not_used_6",
    "not_used_7", "Pcoa,3",
)
FRC_ATOMIC = (
    "R0,alpha", "Val", "m", "RvdW", "Dij", "gamma", "R0,pi", "Val,e",
    "alpha", "gamma,w", "Val,angle", "Povun,5", "not_used_1", "chi", "eta",
    "Phbond",
    "R0,pi-pi", "Plp,2", "Hat", "Pboc,4", "Pboc,3", "Pboc,5", "C_i", "alpha_e",
    "Povun,2", "Pval,3", "beta", "Val,boc", "Pval,5", "Rcore,2", "Ecore,2",
    "Acore,2",
)
FRC_BOND = (
    "De,sigma", "De,pi", "De,pi-pi", "Pbe,1", "Pbo,5", "13_boc", "Pbo,6",
    "Povun,1",
    "Pbe,2", "Pbo,3", "Pbo,4", "not_used_1", "Pbo_1", "Pbo,2", "ovc",
    "not_used_2",
)
FRC_OFFDIAG = ("Dij", "RvdW", "alpha", "R0,sigma", "R0,pi", "R0,pi-pi")
FRC_ANGLE = ("Theta0", "Pval,1", "Pval,2", "Pcoa,1", "Pval,7", "Ppen,1",
             "Pval,4")
FRC_TORSION = ("V1", "V2", "V3", "Ptor,1", "Pcot,1", "not_used_1",
               "not_used_2")
FRC_HBOND = ("Rhb", "Ehb", "Thb", "Phb3")

_XNN_ATOMIC = tuple(n for line in SPECIES_LINES for n in line)
_XNN_BOND = tuple(n for line in BOND_LINES for n in line)

# .frc name -> xnn name per block ("n.u." entries are dropped on read)
GENERAL_FROM_FRC = dict(zip(FRC_GENERAL, GENERAL_PARAMS))
ATOMIC_FROM_FRC = dict(zip(FRC_ATOMIC, _XNN_ATOMIC))
BOND_FROM_FRC = dict(zip(FRC_BOND, _XNN_BOND))
OFFDIAG_FROM_FRC = dict(zip(FRC_OFFDIAG, OFFDIAG_PARAMS))
ANGLE_FROM_FRC = dict(zip(FRC_ANGLE, ANGLE_PARAMS))
TORSION_FROM_FRC = dict(zip(FRC_TORSION, TORSION_PARAMS))
HBOND_FROM_FRC = dict(zip(FRC_HBOND, HBOND_PARAMS))

# customary valence / hydrogen-bond bond-order thresholds for published
# fields, whose general block carries zeros in the corresponding slots
_ACUT_DEFAULT = 1.0e-4
_HBTOL_DEFAULT = 1.0e-4

_REAXFF_SECTIONS = ("reaxff_general_parameters", "reaxff_atomic_parameters_1-8",
                    "reaxff_atomic_parameters_9-16",
                    "reaxff_atomic_parameters_17-24",
                    "reaxff_atomic_parameters_25-32",
                    "reaxff_bond_parameters_1-8", "reaxff_bond_parameters_9-16",
                    "reaxff_off-diagonal_parameters", "reaxff_angle_parameters",
                    "reaxff_torsion_parameters",
                    "reaxff_hydrogen-bond_parameters")


def _frc_wild(sym: str) -> str:
    """``*`` (SEAMM wildcard) -> ``X`` (ReaxFF library wildcard)."""
    return "X" if sym == "*" else sym


def from_forcefield(ff) -> FFieldLibrary:
    """Build a classical :class:`FFieldLibrary` from a resolved ``.frc`` force field.

    Every value is copied as written (kcal/mol, Angstrom, degrees) under its
    xnn name; bond, angle, torsion and hydrogen-bond types keep the
    orientation of the file. Angle and torsion types listed in both
    orientations are deduplicated (reversal only; see
    :func:`dedup_torsion_types`).

    Parameters
    ----------
    ff : xnn.ffnn.common.frc.ForceField
        A ReaxFF force field (``ff_form = reaxff``).

    Returns
    -------
    FFieldLibrary
        The library (``messages = 0``, no network weights).

    Raises
    ------
    ValueError
        If the force field has no ReaxFF parameter sections.
    """
    missing = [k for k in _REAXFF_SECTIONS[:-1] if k not in ff.sections]
    if missing:
        raise ValueError(f"force field {ff.name!r} is not a ReaxFF field; "
                         f"missing sections {missing}")
    p: dict = {}
    heat: dict = {}

    for key, row in ff.rows("reaxff_general_parameters").items():
        name = GENERAL_FROM_FRC.get(key[0])
        if name and name != "n.u.":
            p[name] = float(row.values["Value"])

    spec: list = []
    for kind in _REAXFF_SECTIONS[1:5]:
        for key, row in ff.rows(kind).items():
            sym = key[0]
            if sym not in spec:
                spec.append(sym)
            for col, val in row.values.items():
                if col == "Hat":
                    heat[sym] = float(val)
                name = ATOMIC_FROM_FRC.get(col)
                if name and name != "n.u.":
                    p[f"{name}_{sym}"] = float(val)

    bonds: list = []
    for kind in _REAXFF_SECTIONS[5:7]:
        for key, row in ff.rows(kind).items():
            bd = f"{key[0]}-{key[1]}"
            if bd not in bonds:
                bonds.append(bd)
            for col, val in row.values.items():
                name = BOND_FROM_FRC.get(col)
                if name and name != "n.u.":
                    p[f"{name}_{bd}"] = float(val)

    offd: list = []
    for key, row in ff.rows("reaxff_off-diagonal_parameters").items():
        bd = f"{key[0]}-{key[1]}"
        offd.append(bd)
        for col, val in row.values.items():
            name = OFFDIAG_FROM_FRC.get(col)
            if name:
                p[f"{name}_{bd}"] = float(val)

    angs: list = []
    for key, row in ff.rows("reaxff_angle_parameters").items():
        a = "-".join(key)
        rev = "-".join(reversed(key))
        if a in angs or rev in angs:
            continue
        angs.append(a)
        for col, val in row.values.items():
            name = ANGLE_FROM_FRC.get(col)
            if name:
                p[f"{name}_{a}"] = float(val)

    torp: list = []
    for key, row in ff.rows("reaxff_torsion_parameters").items():
        t1, t2, t3, t4 = (_frc_wild(x) for x in key)
        tor = f"{t1}-{t2}-{t3}-{t4}"
        if tor in torp or f"{t4}-{t3}-{t2}-{t1}" in torp:
            continue
        torp.append(tor)
        for col, val in row.values.items():
            name = TORSION_FROM_FRC.get(col)
            if name and name != "n.u.":
                p[f"{name}_{tor}"] = float(val)

    hbs: list = []
    for key, row in ff.rows("reaxff_hydrogen-bond_parameters").items():
        hb = "-".join(key)
        hbs.append(hb)
        for col, val in row.values.items():
            name = HBOND_FROM_FRC.get(col)
            if name:
                p[f"{name}_{hb}"] = float(val)

    # the classical layout's slots 35/36 ("not_used_4/5" to SEAMM) are where
    # xnn keeps the valence / hydrogen-bond bond-order thresholds; published
    # fields have zeros there and get the customary 1e-4, while a library
    # written by ``to_forcefield`` (a seed library, say) keeps its own values
    for name, default in (("acut", _ACUT_DEFAULT), ("hbtol", _HBTOL_DEFAULT)):
        if not (p.get(name, 0.0) > 0.0):
            p[name] = default
    return FFieldLibrary(
        p=p, m=None, spec=spec, bonds=bonds, offd=offd, angs=angs, torp=torp,
        hbs=hbs, messages=0, heat_increment=heat, name=ff.name,
        references=[r.text for r in ff.references_used()])


def to_forcefield(lib: FFieldLibrary, name: Optional[str] = None,
                  version: str = "1.0", reference_text: str = ""):
    """Write a classical :class:`FFieldLibrary` as an in-memory ``.frc`` file.

    The inverse of :func:`from_forcefield`: the standard blocks become the
    ``#reaxff_*`` sections (value columns in SEAMM's alphabetical order, as
    SEAMM itself writes them), plus a ``#define`` and ``#metadata``.

    Parameters
    ----------
    lib : FFieldLibrary
        The library; must be classical (``not lib.is_nn``).
    name : str, optional
        The ``#define`` name; default ``"reaxff/<lib.name>"``.
    version, reference_text : str, optional
        Version stamp of every row and the ``#reference 1`` text.

    Returns
    -------
    xnn.ffnn.common.frc.FrcFile
        Save it with ``.write(path)``.

    Raises
    ------
    ValueError
        For a ReaxFF-nn library (network weights cannot be expressed in the
        format; use :meth:`FFieldLibrary.save`).
    """
    from ..common.frc import FrcFile, Define, Reference, make_section
    if lib.is_nn:
        raise ValueError("a ReaxFF-nn library carries network weights, which "
                         "the .frc format cannot hold; use FFieldLibrary.save() "
                         "for the JSON format")
    base = lib.name or "reaxff"
    label = base.rsplit("/", 1)[-1]
    name = name or (base if base.startswith("reaxff/") else f"reaxff/{label}")
    frc = FrcFile.empty()
    p = lib.p

    def rows_for(keys, frc_names, xnn_names, sep_key):
        out = []
        for key in keys:
            vals = {}
            for fn, xn in zip(frc_names, xnn_names):
                if xn == "n.u.":
                    vals[fn] = 0.0
                else:
                    vals[fn] = float(p.get(f"{xn}_{sep_key(key)}", 0.0))
            out.append((tuple(key), vals))
        return out

    sections = []
    general = [((fn,), {"Value": float(p.get(xn, 0.0)) if xn != "n.u." else 0.0,
                        "Description": ""})
               for fn, xn in zip(FRC_GENERAL, GENERAL_PARAMS)]
    general.sort(key=lambda kv: kv[0][0])
    sections.append(make_section("reaxff_general_parameters", label,
                                 ["Parameter"], ["Value", "Description"],
                                 general, version=version))
    atomic_names = sorted(FRC_ATOMIC)
    for g in range(4):
        cols = atomic_names[8 * g: 8 * g + 8]
        rows = []
        for sym in lib.spec:
            vals = {}
            for fn in cols:
                xn = ATOMIC_FROM_FRC[fn]
                if fn == "Hat":
                    vals[fn] = float(lib.heat_increment.get(sym, 0.0))
                elif xn == "n.u.":
                    vals[fn] = 0.0
                else:
                    vals[fn] = float(p.get(f"{xn}_{sym}", 0.0))
            rows.append(((sym,), vals))
        sections.append(make_section(f"reaxff_atomic_parameters_{8*g+1}-{8*g+8}",
                                     label, ["Center"], cols, rows,
                                     version=version))
    bond_names = sorted(FRC_BOND)
    for g in range(2):
        cols = bond_names[8 * g: 8 * g + 8]
        rows = []
        for bd in lib.bonds:
            i, j = bd.split("-")
            vals = {fn: (0.0 if BOND_FROM_FRC[fn] == "n.u." else
                         float(p.get(f"{BOND_FROM_FRC[fn]}_{bd}", 0.0)))
                    for fn in cols}
            rows.append(((i, j), vals))
        sections.append(make_section(f"reaxff_bond_parameters_{8*g+1}-{8*g+8}",
                                     label, ["I", "J"], cols, rows,
                                     version=version))
    cols = sorted(FRC_OFFDIAG)
    # like pairs may carry explicit off-diagonal values without being listed
    # in ``offd`` (a seed library switching hydrogen's pi channels off does);
    # write them too, filling what the library never stored from the atomic
    # values, which is exactly what ``complete_off_diagonal`` would do
    offd_pairs = list(lib.offd)
    for sp in lib.spec:
        like = f"{sp}-{sp}"
        if like not in offd_pairs and any(f"{k}_{like}" in p
                                          for k in OFFDIAG_PARAMS):
            offd_pairs.append(like)
    rows = []
    for bd in offd_pairs:
        a, b = bd.split("-")
        vals = {}
        for fn in cols:
            xn = OFFDIAG_FROM_FRC[fn]
            fallback = p.get(f"{xn}_{a}", 0.0) if a == b else 0.0
            vals[fn] = float(p.get(f"{xn}_{bd}", fallback))
        rows.append(((a, b), vals))
    sections.append(make_section("reaxff_off-diagonal_parameters", label,
                                 ["I", "J"], cols, rows, version=version))
    cols = sorted(FRC_ANGLE)
    rows = [(tuple(a.split("-")),
             {fn: float(p.get(f"{ANGLE_FROM_FRC[fn]}_{a}", 0.0)) for fn in cols})
            for a in lib.angs]
    sections.append(make_section("reaxff_angle_parameters", label,
                                 ["I", "J", "K"], cols, rows, version=version))
    cols = sorted(FRC_TORSION)
    rows = []
    for tor in lib.torp:
        key = tuple("*" if x == "X" else x for x in tor.split("-"))
        vals = {fn: (0.0 if TORSION_FROM_FRC[fn] == "n.u." else
                     float(p.get(f"{TORSION_FROM_FRC[fn]}_{tor}", 0.0)))
                for fn in cols}
        rows.append((key, vals))
    sections.append(make_section("reaxff_torsion_parameters", label,
                                 ["I", "J", "K", "L"], cols, rows,
                                 version=version))
    if lib.hbs:
        cols = sorted(FRC_HBOND)
        rows = [(tuple(hb.split("-")),
                 {fn: float(p.get(f"{HBOND_FROM_FRC[fn]}_{hb}", 0.0)) for fn in cols})
                for hb in lib.hbs]
        sections.append(make_section("reaxff_hydrogen-bond_parameters", label,
                                     ["I", "J", "K"], cols, rows,
                                     version=version))
    meta = make_section("metadata", label, ["Parameter"],
                        ["Value", "Description"],
                        [(("ff_form",), {"Value": "reaxff",
                                         "Description": "The functional form of the forcefield"}),
                         (("charges",), {"Value": "qeq/reaxff",
                                         "Description": "How charges should be handled"})],
                        version=version)
    frc.sections[("metadata", label)] = meta
    for sec in sections:
        frc.sections[(sec.kind, sec.label)] = sec
    define = Define(name=name)
    define.entries.append((version, "1", "metadata", [label]))
    for sec in sections:
        define.entries.append((version, "1", sec.kind, [label]))
    frc.defines[name] = define
    text = reference_text or "\n".join(lib.references) or \
        f"ReaxFF parameters exported from xnn ({lib.name or 'library'})."
    frc.references[("<memory>", "1")] = Reference(number="1", text=text,
                                                  author="xnn")
    return frc


def read_ffield(path: Union[str, Path]) -> FFieldLibrary:
    """Read a ReaxFF parameter library.

    Parameters
    ----------
    path : str or Path
        Either a ``.json`` file (a ReaxFF-nn library, possibly with network
        weights, or one written by :meth:`FFieldLibrary.save`), or a
        ``.frc`` force-field spec understood by
        :func:`~xnn.ffnn.common.frc.find_forcefield`: a field shipped with
        xnn by name (``"CHO_cho_2008"``, ``"reaxff/CHO_cho_2008"``), a
        ``.frc`` path, or ``"<path>.frc:<variant>"``.

    Returns
    -------
    FFieldLibrary
        The parsed library.

    Raises
    ------
    FileNotFoundError
        If the spec matches nothing.
    """
    s = str(path)
    if s.lower().endswith(".json"):
        return _read_json(Path(s))
    from ..common.frc import read_forcefield
    ff = read_forcefield(s)
    return from_forcefield(ff)


def cutoff_table(table: Optional[dict], kind: str, spec: list) -> dict:
    """Build a complete per-bond-type cutoff table for the given species.

    Every ordered pair of species gets an entry, taken (in order of
    preference) from the library's table, its reversed spelling, its
    ``"others"`` fallback, then the heuristic default of
    :func:`default_pair_cutoff`.

    Parameters
    ----------
    table : dict or None
        The library's cutoff table (may be partial), or ``None``.
    kind : str
        ``"rcut"`` (bond order) or ``"rcuta"`` (valence).
    spec : list[str]
        Species symbols.

    Returns
    -------
    dict[str, float]
        Cutoff per ordered bond-type string ``"A-B"``.
    """
    src = dict(table or {})
    out = {}
    for a in spec:
        for b in spec:
            bd, rev = f"{a}-{b}", f"{b}-{a}"
            if bd in src:
                out[bd] = float(src[bd])
            elif rev in src:
                out[bd] = float(src[rev])
            elif "others" in src:
                out[bd] = float(src["others"])
            else:
                out[bd] = default_pair_cutoff(a, b, kind)
    return out


# Generic per-element seed values for the template library: covalent radii
# (Angstrom, Cordero et al. 2008), EEM electronegativity / hardness (eV,
# round numbers on the scale typical of ReaxFF EEM parameterizations, cf.
# Mortier et al. 1986), and the vdW radius parameter of the shielded Morse
# term (which ReaxFF evaluates against 2 * rvdw).
_COVALENT_RADIUS = {"H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66,
                    "F": 0.57, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02}
_EEM_CHI = {"H": 4.0, "B": 5.0, "C": 5.5, "N": 6.8, "O": 8.0, "F": 10.0,
            "Si": 4.2, "P": 5.6, "S": 6.5, "Cl": 8.5}
_EEM_MU = {"H": 9.5, "B": 7.0, "C": 7.0, "N": 7.0, "O": 8.0, "F": 10.0,
           "Si": 6.0, "P": 6.5, "S": 7.5, "Cl": 9.0}
_RVDW = {"H": 1.5, "B": 1.9, "C": 1.9, "N": 1.9, "O": 2.0, "F": 1.9,
         "Si": 2.1, "P": 2.1, "S": 2.1, "Cl": 2.1}
_VALENCE = {"H": 1.0, "B": 3.0, "C": 4.0, "N": 3.0, "O": 2.0, "F": 1.0,
            "Si": 4.0, "P": 3.0, "S": 2.0, "Cl": 1.0}
_N_VALENCE_EL = {"H": 1.0, "B": 3.0, "C": 4.0, "N": 5.0, "O": 6.0, "F": 7.0,
                 "Si": 4.0, "P": 5.0, "S": 6.0, "Cl": 7.0}


def template_library(species, *, nn: bool = True, messages: int = 1,
                     message_function: int = 3, energy_function: int = 1,
                     bo_function: int = 0, mf_layer: tuple = (8, 1),
                     be_layer: tuple = (8, 1), seed: int = 0) -> FFieldLibrary:
    """Build a generic starting-point (seed) parameter library.

    The library carries plausible, *untrained* values for every classical
    parameter -- magnitudes follow the published general-parameter and
    hydrocarbon tables of van Duin et al. (2001), radii follow covalent
    radii, and the EEM electronegativities / hardnesses are the standard
    Pearson values -- plus, when ``nn`` is on, randomly initialized ReaxFF-nn
    message / bond-energy networks. It is a seed for training (the point of
    the ``ffnn`` family), a template whose values can be replaced by a
    published parameterization, and the fixture for self-contained tests. It
    is **not** a validated force field.

    Parameters
    ----------
    species : sequence of str
        Chemical symbols to parameterize (organic elements are tabulated;
        others fall back to carbon-like values).
    nn : bool, optional
        Include ReaxFF-nn network weights, by default ``True``.
    messages : int, optional
        Message-passing steps ``T``, by default 1.
    message_function, energy_function, bo_function : int, optional
        ReaxFF-nn function-form selectors, by default 3 / 1 / 0 (the
        combination used by published ReaxFF-nn parameter sets).
    mf_layer, be_layer : tuple, optional
        ``(width, n_hidden)`` of the message and bond-energy networks, by
        default ``(8, 1)``.
    seed : int, optional
        Seed for the network-weight initialization, by default 0.

    Returns
    -------
    FFieldLibrary
        The template library.
    """
    import random

    spec = [str(s) for s in species]

    def bond_order_anchor(a, b):
        """Per-pair sigma/pi bond-order exponentials, anchored so that
        BO(r_e) = 0.85 at the covalent bond length and BO = 1e-4 at the
        bond-order cutoff -- which makes bond orders (and with them every
        valence term) vanish smoothly at the neighbor-list boundaries."""
        re_ab = _COVALENT_RADIUS.get(a, 0.76) + _COVALENT_RADIUS.get(b, 0.76)
        rosi = re_ab / 1.05
        rcut = default_pair_cutoff(a, b, "rcut")
        bo2 = math.log(math.log(1.0e-4) / math.log(0.85)) \
            / math.log(rcut / re_ab)
        bo1 = math.log(0.85) / 1.05 ** bo2
        return re_ab, rosi, bo1, bo2
    p: dict = {
        # general parameters (magnitudes from van Duin et al. 2001, Table 1)
        "boc1": 50.0, "boc2": 15.61, "coa2": 2.17, "trip4": 0.0, "trip3": 0.0,
        "kc2": 0.0, "ovun6": 1.94, "trip2": 0.0, "ovun7": 12.38,
        "ovun8": 13.4, "trip1": 0.0, "swa": 0.0, "swb": 10.0, "val6": 33.87,
        "lp1": 16.0, "val9": 1.06, "val10": 2.04, "pen2": 7.98, "pen3": 0.40,
        "pen4": 4.00, "tor2": 3.17, "tor3": 10.0, "tor4": 0.90, "cot2": 2.17,
        "vdw1": 1.69, "cutoff": 0.01, "coa4": 2.55, "ovun4": 3.0,
        "ovun3": 2.7, "val8": 1.06, "acut": 0.001, "hbtol": 0.001,
        "coa3": 1.05,
    }
    for sp in spec:
        rcov = _COVALENT_RADIUS.get(sp, 0.76)
        p[f"rosi_{sp}"] = 2.0 * rcov / 1.05
        p[f"ropi_{sp}"] = 0.85 * p[f"rosi_{sp}"]
        p[f"ropp_{sp}"] = 0.75 * p[f"rosi_{sp}"]
        p[f"val_{sp}"] = _VALENCE.get(sp, 4.0)
        p[f"vale_{sp}"] = _N_VALENCE_EL.get(sp, 4.0)
        p[f"valang_{sp}"] = _VALENCE.get(sp, 4.0)
        p[f"valboc_{sp}"] = _VALENCE.get(sp, 4.0)
        p[f"mass_{sp}"] = float(SYMBOL_TO_Z.get(sp, 12))
        p[f"chi_{sp}"] = _EEM_CHI.get(sp, 5.5)
        p[f"mu_{sp}"] = _EEM_MU.get(sp, 7.0)
        p[f"gamma_{sp}"] = 0.85
        p[f"gammaw_{sp}"] = 4.0
        p[f"boc3_{sp}"] = 5.02
        p[f"boc4_{sp}"] = 18.32
        p[f"boc5_{sp}"] = 8.32
        p[f"lp2_{sp}"] = 0.0 if p[f"vale_{sp}"] == p[f"val_{sp}"] else 10.0
        p[f"ovun2_{sp}"] = -5.0
        p[f"ovun5_{sp}"] = 30.0
        p[f"atomic_{sp}"] = 0.0
        p[f"val3_{sp}"] = 3.0
        p[f"val5_{sp}"] = 1.0
        p[f"rvdw_{sp}"] = _RVDW.get(sp, 1.9)
        p[f"Devdw_{sp}"] = 0.1
        p[f"alfa_{sp}"] = 10.5

    bonds, offd = [], []
    for i, a in enumerate(spec):
        for b in spec[i:]:
            bd = f"{a}-{b}"
            bonds.append(bd)
            re_ab, rosi, bo1, bo2 = bond_order_anchor(a, b)
            p[f"Desi_{bd}"] = 120.0
            p[f"Depi_{bd}"] = 40.0
            p[f"Depp_{bd}"] = 30.0
            p[f"be1_{bd}"] = 0.3
            p[f"be2_{bd}"] = 0.65
            p[f"bo1_{bd}"] = bo1
            p[f"bo2_{bd}"] = bo2
            # pi channels anchored at BO_pi(r_e) = 0.25, BO_pipi(r_e) = 0.10,
            # with the same exponent (short-ranged by their smaller radii)
            p[f"bo3_{bd}"] = math.log(0.25) / (1.05 / 0.85) ** bo2
            p[f"bo4_{bd}"] = bo2
            p[f"bo5_{bd}"] = math.log(0.10) / (1.05 / 0.75) ** bo2
            p[f"bo6_{bd}"] = bo2
            p[f"ovun1_{bd}"] = 0.5
            p[f"corr13_{bd}"] = 1.0
            p[f"ovcorr_{bd}"] = 1.0
            if a != b:
                offd.append(bd)
                for key in OFFDIAG_PARAMS:
                    p[f"{key}_{bd}"] = math.sqrt(p[f"{key}_{a}"] * p[f"{key}_{b}"])
                p[f"rosi_{bd}"] = rosi
                p[f"ropi_{bd}"] = 0.85 * rosi
                p[f"ropp_{bd}"] = 0.75 * rosi
            # H has no pi orbitals: switch the pi channels off
            if a == "H" or b == "H":
                p[f"ropi_{bd}"] = 0.3 * p.get(f"rosi_{bd}", p[f"rosi_{a}"])
                p[f"ropp_{bd}"] = 0.2 * p.get(f"rosi_{bd}", p[f"rosi_{a}"])
                p[f"bo3_{bd}"] = -50.0
                p[f"bo4_{bd}"] = 0.0
                p[f"bo5_{bd}"] = -50.0
                p[f"bo6_{bd}"] = 0.0

    angs = []
    for b in spec:
        if b == "H":
            continue                      # hydrogen cannot center an angle
        for i, a in enumerate(spec):
            for c in spec[i:]:
                ang = f"{a}-{b}-{c}"
                angs.append(ang)
                p[f"theta0_{ang}"] = 71.5   # equilibrium angle ~108.5 degrees
                p[f"val1_{ang}"] = 30.0
                p[f"val2_{ang}"] = 2.0
                p[f"val4_{ang}"] = 1.0
                p[f"val7_{ang}"] = 1.0
                p[f"coa1_{ang}"] = 0.0
                p[f"pen1_{ang}"] = 0.0

    torp = []
    heavy = [sp for sp in spec if sp != "H"]
    for i, b in enumerate(heavy):
        for c in heavy[i:]:
            tor = f"X-{b}-{c}-X"
            torp.append(tor)
            p[f"V1_{tor}"] = 0.0
            p[f"V2_{tor}"] = 10.0
            p[f"V3_{tor}"] = 0.5
            p[f"tor1_{tor}"] = -2.0
            p[f"cot1_{tor}"] = 0.0

    hbs = []
    if "H" in spec:
        for a in heavy:
            for c in heavy:
                hb = f"{a}-H-{c}"
                hbs.append(hb)
                p[f"rohb_{hb}"] = 1.9
                p[f"Dehb_{hb}"] = -2.5
                p[f"hb1_{hb}"] = 2.0
                p[f"hb2_{hb}"] = 19.0

    m = None
    if nn:
        rng = random.Random(seed)

        def mat(rows, cols):
            """A random (rows x cols) nested list, small uniform values."""
            return [[rng.uniform(-0.5, 0.5) for _ in range(cols)]
                    for _ in range(rows)]

        def vec(cols, shift=0.0):
            """A random length-`cols` list, small uniform values plus shift."""
            return [rng.uniform(-0.5, 0.5) + shift for _ in range(cols)]

        m = {}
        w_m, l_m = mf_layer
        n_in_m = 7 if message_function == 1 else 3
        for sp in spec:
            m[f"fmwi_{sp}"] = mat(n_in_m, w_m)
            m[f"fmbi_{sp}"] = vec(w_m)
            m[f"fmw_{sp}"] = [mat(w_m, w_m) for _ in range(l_m)]
            m[f"fmb_{sp}"] = [vec(w_m) for _ in range(l_m)]
            m[f"fmwo_{sp}"] = mat(w_m, 3)
            # bias the message output toward 1 so the seed bond orders are
            # not crushed before training starts
            m[f"fmbo_{sp}"] = vec(3, shift=2.0)
        w_e, l_e = be_layer
        for bd in bonds:
            m[f"fewi_{bd}"] = mat(3, w_e)
            m[f"febi_{bd}"] = vec(w_e)
            m[f"few_{bd}"] = [mat(w_e, w_e) for _ in range(l_e)]
            m[f"feb_{bd}"] = [vec(w_e) for _ in range(l_e)]
            m[f"fewo_{bd}"] = mat(w_e, 1)
            m[f"febo_{bd}"] = vec(1)

    # the template widens the valence (angle/torsion/H-bond) windows toward
    # the bond-order cutoff, where its anchored bond orders are ~1e-4 -- so
    # valence terms switch on/off smoothly as pairs enter the lists
    rcut_table, rcuta_table = {}, {}
    for i, a in enumerate(spec):
        for b in spec[i:]:
            rc = default_pair_cutoff(a, b, "rcut")
            rcut_table[f"{a}-{b}"] = rc
            rcuta_table[f"{a}-{b}"] = rc - 0.05

    return FFieldLibrary(
        p=p, m=m, spec=spec, bonds=bonds, offd=offd, angs=angs, torp=torp,
        hbs=hbs, messages=messages if nn else 0,
        bo_function=bo_function if nn else 0,
        energy_function=energy_function if nn else 0,
        message_function=message_function if nn else 0,
        vdw_function=0, bo_layer=None,
        mf_layer=tuple(mf_layer) if nn else None,
        be_layer=tuple(be_layer) if nn else None,
        vdw_layer=None, rcut=rcut_table, rcuta=rcuta_table)
