"""Builder for the GMTKN55 benchmark database (Goerigk *et al.* 2017).

GMTKN55 collects 55 benchmark sets of general main-group thermochemistry,
kinetics and noncovalent interactions: 1505 relative energies computed from
2462 single-point structures, with reference values at the W4 / W1-F12 /
W2-F12 / CCSD(T)/CBS level (Goerigk, Hansen, Bauer, Ehrlich, Najibi &
Grimme, *Phys. Chem. Chem. Phys.* 19, 32184, 2017, DOI
10.1039/C7CP04913G). The sets fall into five categories: basic properties
and reactions of small systems (18 sets), reactions of large systems and
isomerisations (9), barrier heights (7), intermolecular (12) and
intramolecular (9) noncovalent interactions; see :data:`CATEGORIES`.

Data
----
The files are those of the maintainers' repository
`grimme-lab/GMTKN55 <https://github.com/grimme-lab/GMTKN55>`_ (dataset
license CC BY 4.0; cite the paper above). Two versions are available, each
pinned to a commit and verified by MD5:

* ``"v2"`` (default): the ``v2`` branch, with the maintainers' updated
  reference values (251 reactions in C60ISO, ISOL24, UPU23, S66, G21EA,
  G21IP, WATER27, BH76, BH76RC, BHPERI, PX13 and BHDIV10) and 10 reactions
  removed whose references were affected by spin contamination (Liang &
  Head-Gordon, *J. Chem. Theory Comput.* 2025, DOI 10.1021/acs.jctc.5c01380):
  1495 reactions, 2448 structures. The changes are listed in the
  repository's ``REFERENCE_UPDATES.md``.
* ``"v1"``: the ``v1`` branch, the reference values of the 2017 paper: 1505
  reactions, 2462 structures. The geometries are identical in both versions.

Only the geometries, charges, spins, reaction definitions, bibliographies
and the maintainers' PBEh-3c single-point outputs are extracted from the
archive (about 40 MB); the evaluation scripts of the repository are not
used. Everything is cached under ``datasets/gmtkn55/raw/``; nothing ships
with xnn.

Contents
--------
``load_dataset("gmtkn55")`` returns two lists:

* ``structures``: one structure dict per species of every loaded set (2442
  distinct species in ``v1``, 2420 in ``v2``) with ``atomic_numbers``,
  ``pos`` (Angstrom), ``total_charge``,
  ``spin_multiplicity`` (unpaired electrons + 1), ``subset``, ``category``,
  ``name`` (the species folder of the repository) and ``pbeh3c_energy``,
  the maintainers' PBEh-3c/def2-mSVP total energy in eV (a ready baseline
  and the oracle of the verification notebook);
* ``reactions``: one record per relative energy with ``subset``,
  ``category``, ``index`` (1-based within the set), ``species`` (names),
  ``stoichiometry`` (signed integers) and ``reference`` in kcal/mol. A
  reaction energy is ``sum(coefficient * E(species))``; the sign convention
  is the repository's (e.g. S22 and S66 binding energies are positive).

The BH76RC set is defined on the structures of the BH76 folder; its species
are listed under ``subset="BH76RC"`` as well.

The helpers :func:`reaction_energies`, :func:`subset_statistics` and
:func:`wtmad2` score a model from a mapping of its per-species energies.
WTMAD-2 (eq. 2 of the paper) scales every set's mean absolute deviation by
``56.84 / <|dE|>`` with 56.84 kcal/mol the paper's average of the 55 mean
absolute reference energies (:data:`PAPER_SCALE`), weights it by the number
of reactions and divides by the total count. The paper's constant is kept
by default so that scores compare with the published ones; the actual
average of the data is 57.82 kcal/mol in ``v1`` (Table 1 of the paper
prints 414.73 for the MB16-43 average where the data give 468.39) and
57.40 in ``v2``; ``scale="data"`` recomputes it from the loaded sets.

Notes
-----
* In ``v2`` the species names of the WATER27 reaction file do not match the
  folder names (``H1O`` for ``H2O``, ``H2Op`` for ``H3Op``, ``OHmH1O`` for
  ``OHmH2O``); the builder maps them back, and every reaction of both
  versions resolves to a geometry.
* The reaction files carry two trailing columns (an energy offset and an
  error multiplier of the maintainers' tool); they are 0 and 1 throughout
  and are checked, not applied.
* Charged (up to 2 e) and open-shell species (up to three unpaired
  electrons), 60 elements up to Bi and systems of up to 81 atoms occur; the
  ``elements``, ``max_abs_charge`` and ``closed_shell`` filters restrict the
  reactions to what a model can evaluate.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union

import numpy as np

from ..elements import atomic_number
from ._download import download_file, extract_archive
from .base import DatasetBuilder, register_dataset
from .units import HARTREE_TO_EV, KCAL_MOL_TO_EV

_URL = "https://github.com/grimme-lab/GMTKN55/archive/{commit}.zip"

#: version -> (commit, archive MD5)
_VERSIONS = {
    "v1": ("8d485b37a1ca8837e395042671ca5ba4e0714691", "1c83333b9009045e83c7cf3a6ccdf5cc"),
    "v2": ("ccabc16c8c4677fc60c19870ae96ac25405844a2", "5c92f8fc05e350bb35cf9c5d82d317ce"),
}

#: archive members kept (relative to the repository root)
_KEEP_SUFFIXES = ("/struc.xyz", "/.CHRG", "/.UHF", "/.res", "/.resRC", "/.bib",
                  "/PBEh-3c/orca.out")
_KEEP_FILES = ("README.md", "LICENSE", "REFERENCE_UPDATES.md", ".list.dirs", ".bib")

#: the 55 sets by category, in the order of Table 1 of the paper
CATEGORIES: dict[str, tuple[str, ...]] = {
    "small_reactions": ("W4-11", "G21EA", "G21IP", "DIPCS10", "PA26", "SIE4x4", "ALKBDE10",
                        "YBDE18", "AL2X6", "HEAVYSB11", "NBPRC", "ALK8", "RC21", "G2RC",
                        "BH76RC", "FH51", "TAUT15", "DC13"),
    "large_reactions": ("MB16-43", "DARC", "RSE43", "BSR36", "CDIE20", "ISO34", "ISOL24",
                        "C60ISO", "PArel"),
    "barrier_heights": ("BH76", "BHPERI", "BHDIV10", "INV24", "BHROT27", "PX13", "WCPT18"),
    "intermolecular_nci": ("RG18", "ADIM6", "S22", "S66", "HEAVY28", "WATER27", "CARBHB12",
                           "PNICO23", "HAL59", "AHB21", "CHB6", "IL16"),
    "intramolecular_nci": ("IDISP", "ICONF", "ACONF", "Amino20x4", "PCONF21", "MCONF",
                           "SCONF", "UPU23", "BUT14DIOL"),
}
#: all 55 set names
SUBSETS: tuple[str, ...] = tuple(s for sets in CATEGORIES.values() for s in sets)
_CATEGORY_OF = {s: c for c, sets in CATEGORIES.items() for s in sets}
_LOOKUP = {s.lower(): s for s in SUBSETS}
# sets whose structures live in another folder, and their reaction file
_FOLDER = {"BH76RC": "BH76"}
_RES_FILE = {"BH76RC": ".resRC"}
# species-name repairs per version (regex -> replacement, applied to the start)
_SPECIES_FIXES = {
    "v2": {"WATER27": (("H1O", "H2O"), ("H2Op", "H3Op"), ("OHmH1O", "OHmH2O"))},
}

#: the paper's WTMAD-2 normalisation constant (kcal/mol)
PAPER_SCALE = 56.84


def _expand(token: str) -> list[str]:
    """Species names of one path token of a reaction line (``01{,a}/$f``)."""
    name = token.split("/")[0]
    m = re.fullmatch(r"([^{}]*)\{([^{}]*)\}([^{}]*)", name)
    if m:
        return [m.group(1) + part + m.group(3) for part in m.group(2).split(",")]
    if "{" in name or "}" in name:
        raise ValueError(f"cannot expand species token {token!r}")
    return [name]


def parse_reactions(text: str) -> list[tuple[list[str], list[int], float]]:
    """Reactions of one ``.res`` file: (species names, coefficients, reference in kcal/mol).

    A reaction line names its species (bash brace groups allowed), then
    ``x``, the signed coefficients, ``$w`` and the reference value;
    everything after ``#`` is a comment.
    """
    out = []
    for raw in text.splitlines():
        tokens = raw.split("#", 1)[0].split()
        if len(tokens) < 5 or tokens[0] not in ("$tmer", "tmer2++"):
            continue
        if "x" not in tokens or "$w" not in tokens:
            continue
        ix, iw = tokens.index("x"), tokens.index("$w")
        species = [name for tok in tokens[1:ix] for name in _expand(tok)]
        coefficients = [int(t) for t in tokens[ix + 1:iw]]
        if len(species) != len(coefficients):
            raise ValueError(f"species and coefficients differ in length: {raw!r}")
        trailing = []
        for t in tokens[iw + 2:]:
            try:
                trailing.append(float(t))
            except ValueError:
                break
        if trailing and (trailing[0] != 0.0 or (len(trailing) > 1 and trailing[1] != 1.0)):
            raise ValueError(f"unsupported energy offset / error multiplier: {raw!r}")
        out.append((species, coefficients, float(tokens[iw + 1])))
    return out


def _read_xyz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text().splitlines()
    n = int(lines[0].split()[0])
    numbers, xyz = [], []
    for line in lines[2:2 + n]:
        tok = line.split()
        numbers.append(atomic_number(tok[0]))
        xyz.append([float(v) for v in tok[1:4]])
    return np.array(numbers, dtype=np.int64), np.array(xyz, dtype=np.float64)


def _read_int(path: Path, default: int = 0) -> int:
    return int(path.read_text().split()[0]) if path.exists() else default


def _read_orca_energy(path: Path) -> Optional[float]:
    """Last ``FINAL SINGLE POINT ENERGY`` of an ORCA output, in eV."""
    if not path.exists():
        return None
    energy = None
    for line in path.read_text().splitlines():
        if "FINAL SINGLE POINT ENERGY" in line:
            energy = float(line.split()[-1])
    return None if energy is None else energy * HARTREE_TO_EV


def _canonical(subset: str) -> str:
    key = subset.strip().lower().replace("_", "-")
    if key not in _LOOKUP:
        raise ValueError(f"unknown GMTKN55 set {subset!r}; choose from {', '.join(SUBSETS)}")
    return _LOOKUP[key]


def _element_set(elements: Iterable[Union[int, str]]) -> set[int]:
    return {int(e) if not isinstance(e, str) else atomic_number(e) for e in elements}


class GMTKN55Builder(DatasetBuilder):
    """Builder for the GMTKN55 database (see the module docstring)."""

    name = "gmtkn55"
    description = ("GMTKN55 (Goerigk 2017): 55 sets, 1505 main-group reaction energies, "
                   "barrier heights and noncovalent interactions with CCSD(T)-quality "
                   "references (kcal/mol) on 2462 structures.")

    def load(self, *, split: Optional[str] = None, cache_dir: Path, version: str = "v2",
             subsets: Optional[Union[str, Sequence[str]]] = None,
             elements: Optional[Iterable[Union[int, str]]] = None,
             max_abs_charge: Optional[float] = None, closed_shell: bool = False,
             quiet: bool = False) -> Union[dict[str, list[dict]], list[dict]]:
        """Download, extract and read the database.

        Parameters
        ----------
        split : str or None
            ``None`` returns ``{"structures": [...], "reactions": [...]}``;
            ``"structures"`` (or ``"all"``) and ``"reactions"`` return one list.
        cache_dir : pathlib.Path
            Base cache directory; files go under ``cache_dir/"gmtkn55"/"raw"``.
        version : str, optional
            ``"v2"`` (default, updated references) or ``"v1"`` (the paper's).
        subsets : str or sequence of str, optional
            Set names to load (case-insensitive); ``None`` loads all 55.
        elements : iterable of int or str, optional
            Keep only reactions whose species contain these elements alone.
        max_abs_charge : float, optional
            Keep only reactions whose species have ``|total_charge|`` at most
            this value.
        closed_shell : bool, optional
            Keep only reactions whose species have no unpaired electrons.
        quiet : bool, optional
            Suppress the download progress bar.

        Returns
        -------
        dict of {str: list of dict} or list of dict
            The structures and the reactions (see the module docstring). After
            filtering, the structures are those used by the kept reactions.
        """
        if split not in (None, "all", "structures", "reactions"):
            raise ValueError("split must be None, 'structures' ('all') or 'reactions'")
        if version not in _VERSIONS:
            raise ValueError(f"version must be one of {sorted(_VERSIONS)}")
        if isinstance(subsets, str):
            subsets = [subsets]
        names = SUBSETS if subsets is None else [_canonical(s) for s in subsets]

        root = self._fetch(Path(cache_dir) / self.name / "raw", version, quiet)
        structures: list[dict] = []
        reactions: list[dict] = []
        allowed = None if elements is None else _element_set(elements)
        for subset in names:
            folder = root / _FOLDER.get(subset, subset)
            fixes = _SPECIES_FIXES.get(version, {}).get(subset, ())
            cache: dict[str, dict] = {}
            for k, (species, coefficients, reference) in enumerate(
                    parse_reactions((folder / _RES_FILE.get(subset, ".res")).read_text()), 1):
                species = [self._fix(s, fixes) for s in species]
                for s in species:
                    if s not in cache:
                        cache[s] = self._structure(folder / s, subset, s)
                if not self._keep(species, cache, allowed, max_abs_charge, closed_shell):
                    continue
                reactions.append({"subset": subset, "category": _CATEGORY_OF[subset],
                                  "index": k, "species": species,
                                  "stoichiometry": coefficients, "reference": reference})
            used = {s for r in reactions if r["subset"] == subset for s in r["species"]}
            structures.extend(cache[s] for s in cache if s in used)

        splits = {"structures": structures, "reactions": reactions}
        if split is None:
            return splits
        return splits["structures" if split == "all" else split]

    @staticmethod
    def _fix(name: str, fixes: Sequence[tuple[str, str]]) -> str:
        for wrong, right in fixes:
            if name.startswith(wrong):
                return right + name[len(wrong):]
        return name

    @staticmethod
    def _keep(species, cache, allowed, max_abs_charge, closed_shell) -> bool:
        for s in species:
            d = cache[s]
            if allowed is not None and not set(d["atomic_numbers"].tolist()) <= allowed:
                return False
            if max_abs_charge is not None and abs(d["total_charge"]) > max_abs_charge:
                return False
            if closed_shell and d["spin_multiplicity"] != 1.0:
                return False
        return True

    @staticmethod
    def _structure(folder: Path, subset: str, name: str) -> dict:
        if not (folder / "struc.xyz").exists():
            raise FileNotFoundError(f"no geometry for species {name!r} of {subset} ({folder})")
        numbers, pos = _read_xyz(folder / "struc.xyz")
        d = {"atomic_numbers": numbers, "pos": pos,
             "total_charge": float(_read_int(folder / ".CHRG")),
             "spin_multiplicity": float(_read_int(folder / ".UHF")) + 1.0,
             "subset": subset, "category": _CATEGORY_OF[subset], "name": name}
        energy = _read_orca_energy(folder / "PBEh-3c" / "orca.out")
        if energy is not None:
            d["pbeh3c_energy"] = energy
        return d

    @staticmethod
    def _fetch(raw: Path, version: str, quiet: bool) -> Path:
        """Download the pinned archive once and extract the needed members."""
        root = raw / version
        marker = root / ".extracted"
        if marker.exists():
            return root
        commit, md5 = _VERSIONS[version]
        archive = download_file(_URL.format(commit=commit), raw / f"GMTKN55-{version}.zip",
                                md5, quiet=quiet)

        def select(member: str) -> bool:
            rel = member.split("/", 1)[1] if "/" in member else ""
            return rel in _KEEP_FILES or member.endswith(_KEEP_SUFFIXES)

        extract_archive(archive, root, select=select, strip=1)
        marker.write_text(commit + "\n")
        return root


register_dataset(GMTKN55Builder())


def reaction_energies(reactions: Sequence[Mapping], energies: Mapping[tuple[str, str], float],
                      units: str = "eV") -> np.ndarray:
    """Reaction energies (kcal/mol) from per-species energies.

    Parameters
    ----------
    reactions : sequence of dict
        Reaction records of :func:`~xnn.common.data.hub.load_dataset`.
    energies : mapping of {(subset, name): float}
        Total energy of every species, keyed by its set and name.
    units : str, optional
        Units of ``energies``: ``"eV"`` (default), ``"kcal/mol"`` or
        ``"hartree"``.

    Returns
    -------
    numpy.ndarray
        One reaction energy per record, ``sum(coefficient * E)`` in kcal/mol.
    """
    factor = {"ev": 1.0 / KCAL_MOL_TO_EV, "kcal/mol": 1.0,
              "hartree": HARTREE_TO_EV / KCAL_MOL_TO_EV}[units.lower()]
    out = np.empty(len(reactions))
    for k, r in enumerate(reactions):
        out[k] = factor * sum(c * energies[(r["subset"], s)]
                              for s, c in zip(r["species"], r["stoichiometry"]))
    return out


def subset_statistics(reactions: Sequence[Mapping],
                      predicted: Sequence[float]) -> dict[str, dict[str, float]]:
    """Per-set error statistics of predicted reaction energies.

    Parameters
    ----------
    reactions : sequence of dict
        Reaction records.
    predicted : sequence of float
        Predicted reaction energies (kcal/mol), aligned with ``reactions``.

    Returns
    -------
    dict of {str: dict}
        For every set (in :data:`SUBSETS` order): ``n``, ``mean_abs_ref``
        (the set's average absolute reference energy), ``mad``, ``md``
        (mean signed deviation), ``rmsd`` and ``max_abs`` in kcal/mol.
    """
    pred = np.asarray(predicted, dtype=float)
    if pred.shape != (len(reactions),):
        raise ValueError("predicted must hold one value per reaction")
    ref = np.array([r["reference"] for r in reactions], dtype=float)
    sets = np.array([r["subset"] for r in reactions])
    stats = {}
    for s in SUBSETS:
        m = sets == s
        if not m.any():
            continue
        err = pred[m] - ref[m]
        stats[s] = {"n": int(m.sum()), "mean_abs_ref": float(np.abs(ref[m]).mean()),
                    "mad": float(np.abs(err).mean()), "md": float(err.mean()),
                    "rmsd": float(np.sqrt((err ** 2).mean())),
                    "max_abs": float(np.abs(err).max())}
    return stats


def wtmad2(reactions: Sequence[Mapping], predicted: Sequence[float],
           scale: Union[float, str] = PAPER_SCALE) -> dict[str, float]:
    """WTMAD-2 of predicted reaction energies (eq. 2 of the paper).

    Parameters
    ----------
    reactions : sequence of dict
        Reaction records.
    predicted : sequence of float
        Predicted reaction energies (kcal/mol), aligned with ``reactions``.
    scale : float or "data", optional
        The constant that scales the sets' mean absolute deviations:
        :data:`PAPER_SCALE` (56.84 kcal/mol, default) or ``"data"`` for the
        average of the loaded sets' mean absolute reference energies.

    Returns
    -------
    dict of {str: float}
        ``total`` and one entry per category of :data:`CATEGORIES` present
        (plus ``nci`` for both noncovalent categories together), each the
        weighted sum over its sets divided by its number of reactions.
    """
    stats = subset_statistics(reactions, predicted)
    s = float(np.mean([v["mean_abs_ref"] for v in stats.values()])) if scale == "data" \
        else float(scale)
    weighted = {name: v["n"] * s / v["mean_abs_ref"] * v["mad"] for name, v in stats.items()}

    def score(names):
        n = sum(stats[k]["n"] for k in names)
        return sum(weighted[k] for k in names) / n if n else float("nan")

    out = {"total": score(list(stats))}
    for cat, sets in CATEGORIES.items():
        present = [k for k in sets if k in stats]
        if present:
            out[cat] = score(present)
    nci = [k for k in stats if _CATEGORY_OF[k].endswith("_nci")]
    if nci:
        out["nci"] = score(nci)
    return out
