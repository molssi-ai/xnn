"""ReaxFF parameter libraries: the standard ``ffield`` text format and JSON.

A ReaxFF model is fully specified by its parameter library. Two on-disk
formats are supported:

* the standard ReaxFF ``ffield`` text library (the format introduced with the
  original Fortran code and shared by LAMMPS / GULP / AMS): a block of general
  parameters followed by per-species, per-bond, off-diagonal, valence-angle,
  torsion and hydrogen-bond blocks;
* the JSON parameter-library format used by ReaxFF-nn parameter sets
  (Guo et al., *Comput. Mater. Sci.* 172, 109393, 2020; Xue et al., *PCCP*
  23, 19457, 2021), which stores the same parameters as a flat
  ``"<name>_<type>"`` dictionary plus the neural-network weight matrices and
  the function/layer selectors.

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
from typing import Optional


# Chemical symbols indexed by atomic number (Z = index).
CHEMICAL_SYMBOLS = (
    "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
    "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb",
    "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm",
    "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta",
    "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At",
    "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu",
)

SYMBOL_TO_Z = {sym: z for z, sym in enumerate(CHEMICAL_SYMBOLS) if z > 0}


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
        ReaxFF-nn function-form selectors (see :class:`~xnns.ffnn.models.reaxff.ReaxFF`).
    bo_layer, mf_layer, be_layer, vdw_layer : tuple or None
        ``(width, n_hidden)`` of the bond-order / message / bond-energy / vdW
        neural networks.
    rcut, rcuta : dict[str, float] or None
        Per-bond-type bond-order and valence cutoff tables (with an
        ``"others"`` fallback), or ``None`` to use the defaults.
    mol_energy : dict[str, float]
        Per-molecule reference-energy offsets some ReaxFF-nn training
        workflows record (not used by the model; kept for round-tripping).
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
    """Drop torsion types that are index-permuted duplicates of another entry.

    A torsion ``i-j-k-l`` is equivalent to ``l-k-j-i`` (full reversal) and to
    the central-bond swaps ``i-k-j-l`` / ``l-j-k-i``; only one spelling is
    kept.

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
        variants = {f"{t1}-{t3}-{t2}-{t4}", f"{t4}-{t3}-{t2}-{t1}",
                    f"{t4}-{t2}-{t3}-{t1}"}
        variants.discard(tor)
        if not any(v in kept for v in variants):
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
    """Look up a torsion parameter, resolving permutations and wildcards.

    The lookup order matches the convention of ReaxFF codes: the exact type,
    its central-bond swaps and reversal, then the ``X-j-k-X`` and ``X-k-j-X``
    wildcards; unmatched types get ``0.0``.

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
    for cand in (f"{t1}-{t3}-{t2}-{t4}", f"{t4}-{t3}-{t2}-{t1}",
                 f"{t4}-{t2}-{t3}-{t1}", f"X-{t2}-{t3}-X", f"X-{t3}-{t2}-X"):
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


def _read_text(path: Path) -> FFieldLibrary:
    """Parse a standard ReaxFF ``ffield`` text library.

    The layout follows the published format: a header line, the general
    parameter block, then per-species (4 lines x 8 columns), per-bond
    (2 x 8), off-diagonal, valence-angle, torsion and hydrogen-bond blocks
    (see the ``*_LINES`` / ``*_PARAMS`` module constants for the column
    meanings). ``acut`` / ``hbtol`` -- the bond-order thresholds of the
    valence and hydrogen-bond terms -- are set to ``1e-4``, the customary
    value for text libraries (JSON libraries carry their own).

    Parameters
    ----------
    path : Path
        Path to the text library.

    Returns
    -------
    FFieldLibrary
        The parsed library.
    """
    lines = Path(path).read_text().splitlines()
    p: dict = {}
    n_general = int(lines[1].split()[0])
    if n_general > len(GENERAL_PARAMS):
        raise ValueError(f"ffield declares {n_general} general parameters; "
                         f"at most {len(GENERAL_PARAMS)} are supported")
    for i in range(n_general):
        p[GENERAL_PARAMS[i]] = float(lines[2 + i].split()[0])

    row = 2 + n_general                      # first line of the species block
    n_spec = int(lines[row].split()[0])
    row += len(SPECIES_LINES)                # skip the 4 column-header lines
    spec = []
    for _ in range(n_spec):
        cols = lines[row].split()
        spec.append(cols[0])
        for il, names in enumerate(SPECIES_LINES):
            cols = lines[row + il].split()
            first = 1 if il == 0 else 0      # first species line starts with the symbol
            for ip, name in enumerate(names):
                p[f"{name}_{spec[-1]}"] = float(cols[first + ip])
        row += len(SPECIES_LINES)

    n_bond = int(lines[row].split()[0])
    row += len(BOND_LINES)
    bonds = []
    for _ in range(n_bond):
        cols = lines[row].split()
        bd = f"{spec[int(cols[0]) - 1]}-{spec[int(cols[1]) - 1]}"
        bonds.append(bd)
        for il, names in enumerate(BOND_LINES):
            cols = lines[row + il].split()
            first = 2 if il == 0 else 0      # first bond line starts with the two indices
            for ip, name in enumerate(names):
                p[f"{name}_{bd}"] = float(cols[first + ip])
        row += len(BOND_LINES)

    n_offd = int(lines[row].split()[0])
    row += 1
    offd = []
    for _ in range(n_offd):
        cols = lines[row].split()
        bd = f"{spec[int(cols[0]) - 1]}-{spec[int(cols[1]) - 1]}"
        offd.append(bd)
        for ip, name in enumerate(OFFDIAG_PARAMS):
            p[f"{name}_{bd}"] = float(cols[2 + ip])
        row += 1

    n_ang = int(lines[row].split()[0])
    row += 1
    angs = []
    for _ in range(n_ang):
        cols = lines[row].split()
        a = "-".join(spec[int(c) - 1] for c in cols[:3])
        rev = "-".join(reversed(a.split("-")))
        if a not in angs and rev not in angs:
            angs.append(a)
            for ip, name in enumerate(ANGLE_PARAMS):
                p[f"{name}_{a}"] = float(cols[3 + ip])
        row += 1

    n_tor = int(lines[row].split()[0])
    row += 1
    torp = []
    for _ in range(n_tor):
        cols = lines[row].split()
        names4 = ["X" if int(c) == 0 else spec[int(c) - 1] for c in cols[:4]]
        tor = "-".join(names4)
        t1, t2, t3, t4 = names4
        variants = {f"{t4}-{t3}-{t2}-{t1}", f"{t1}-{t3}-{t2}-{t4}",
                    f"{t4}-{t2}-{t3}-{t1}"}
        if tor not in torp and not any(v in torp for v in variants):
            torp.append(tor)
            for ip, name in enumerate(TORSION_PARAMS[:5]):
                p[f"{name}_{tor}"] = float(cols[4 + ip])
        row += 1

    n_hb = int(lines[row].split()[0])
    row += 1
    hbs = []
    for _ in range(n_hb):
        cols = lines[row].split()
        hb = "-".join(spec[int(c) - 1] for c in cols[:3])
        hbs.append(hb)
        for ip, name in enumerate(HBOND_PARAMS):
            p[f"{name}_{hb}"] = float(cols[3 + ip])
        row += 1

    p["acut"] = 1.0e-4
    p["hbtol"] = 1.0e-4
    return FFieldLibrary(p=p, m=None, spec=spec, bonds=bonds, offd=offd,
                         angs=angs, torp=torp, hbs=hbs, messages=0)


def read_ffield(path) -> FFieldLibrary:
    """Read a ReaxFF parameter library from disk.

    Files ending in ``.json`` are parsed as ReaxFF-nn JSON libraries (which
    may carry network weights); anything else is parsed as a standard
    ``ffield`` text library.

    Parameters
    ----------
    path : str or Path
        Path to the parameter file.

    Returns
    -------
    FFieldLibrary
        The parsed library.
    """
    path = Path(path)
    if path.suffix == ".json":
        return _read_json(path)
    return _read_text(path)


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
