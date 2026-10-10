"""Builder for the benchmark systems of the fourth-generation HDNNP paper.

Ko, Finkler, Goedecker and Behler (*Nat. Commun.* 12, 398, 2021) introduced
four systems in which charge transfer is non-local, each with total energies,
forces and Hirshfeld charges from FHI-aims (PBE):

* ``carbon_chain``: 10019 C10H2 and C10H3+ chains;
* ``ag_clusters``: 11013 Ag3+ and Ag3- clusters;
* ``nacl_clusters``: 5000 Na9Cl8+ and Na8Cl8+ clusters;
* ``au2_mgo``: 5000 periodic structures of an Au2 dimer on an undoped or an
  Al-doped MgO(001) slab (110 atoms).

The data is read from extended XYZ copies in angstrom and eV: the molecules
from the data repository of the CACE long-range paper (Kim, King, Zhong and
Cheng, *Nat. Commun.* 2025; CC BY-NC 4.0), ``au2_mgo`` from the repository of
the latent Ewald summation paper (Cheng, *npj Comput. Mater.* 2025).

Notes
-----
* There is no official split (the paper split 90/10 at random), so the data
  comes as one ``"all"`` split.
* The total charge of a structure is the sum of its Hirshfeld charges.
* Reading the files needs the ``ase`` extra.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from ._download import download_file
from .base import DatasetBuilder, register_dataset

_SOURCES = {
    "cace": "https://raw.githubusercontent.com/BingqingCheng/cace-lr-fit/"
            "c009da6edf720b9302b7eadfe2a00b6694dfd70c/{path}",
    "les": "https://raw.githubusercontent.com/ChengUCB/les_fit/"
           "a886785caa4182a0effb95d95f0e402881adfc4d/{path}",
}

# system -> [(source, path in the repository, md5)]
_FILES = {
    "carbon_chain": [("cace", "fit-4hdnnp-carbon-chain/both.xyz",
                      "be48cbff4b6bbe288fa5ecbe3615127c")],
    "ag_clusters": [("cace", "fit-4hdnnp-Ag-cluster/Ag-cluster.xyz",
                     "dd601bfbbf1e42f031046c55cd92b9b0")],
    "nacl_clusters": [("cace", "fit-4hdnnp-NaCl/NaCl.xyz",
                       "cd28edb3ea3893ad0e25f995064363f9")],
    "au2_mgo": [("les", "data-benchmark/train-Au-MgO-Al.xyz",
                 "e5c2290ae3e58b492e8add5e054e4df9"),
                ("les", "data-benchmark/test-Au-MgO-Al.xyz",
                 "ace9a86f28262db5ef1022ef523d24f9")],
}

_ALIASES = {
    "carbon": "carbon_chain", "c10h2": "carbon_chain",
    "ag": "ag_clusters", "ag3": "ag_clusters", "ag_cluster": "ag_clusters",
    "nacl": "nacl_clusters", "nacl_cluster": "nacl_clusters",
    "aumgo": "au2_mgo", "au_mgo": "au2_mgo", "au2mgo": "au2_mgo",
}


class HDNNP4GBuilder(DatasetBuilder):
    """Builder for the four benchmark systems of Ko et al. (2021).

    See the module docstring for the systems, the sources and the citations.
    """

    name = "hdnnp4g"
    description = ("4G-HDNNP benchmarks (Ko et al. 2021): carbon chains, Ag3 and NaCl "
                   "clusters, Au2 on MgO; energies, forces, Hirshfeld charges.")

    def load(self, *, split: Optional[str] = None, cache_dir: Path,
             system: str = "au2_mgo",
             quiet: bool = False) -> Union[dict[str, list[dict]], list[dict]]:
        """Download and convert one system.

        Parameters
        ----------
        split : str or None
            ``None`` returns ``{"all": structures}``; ``"all"`` returns the list.
        cache_dir : pathlib.Path
            Base cache directory; files are stored under ``cache_dir/"hdnnp4g"``.
        system : str, optional
            ``carbon_chain``, ``ag_clusters``, ``nacl_clusters`` or ``au2_mgo``
            (default).
        quiet : bool, optional
            Suppress progress output. Defaults to ``False``.

        Returns
        -------
        dict of {str: list of dict} or list of dict
            Structure dicts with ``pos``, ``atomic_numbers``, ``energy``,
            ``forces``, ``charges`` and ``total_charge``; ``cell`` and ``pbc``
            are periodic for ``au2_mgo`` only.

        Raises
        ------
        ValueError
            If ``system`` or ``split`` is unknown.
        """
        key = _ALIASES.get(system.strip().lower(), system.strip().lower())
        if key not in _FILES:
            raise ValueError(f"unknown hdnnp4g system {system!r}; choose one of: "
                             + ", ".join(_FILES))
        if split not in (None, "all"):
            raise ValueError(f"unknown split {split!r}; hdnnp4g has no official split, "
                             "use split=None or split='all'")
        root = Path(cache_dir) / self.name / "raw"
        structures = []
        for source, path, md5 in _FILES[key]:
            local = download_file(_SOURCES[source].format(path=path),
                                  root / source / Path(path).name, md5, quiet=quiet)
            structures += _read_extxyz(local)
        return structures if split == "all" else {"all": structures}


def _atomic_charges(atoms) -> np.ndarray:
    """The per-atom charges of a frame, whichever column name the file uses."""
    for key in ("initial_charges", "charge", "charges"):
        if key in atoms.arrays:
            return np.asarray(atoms.arrays[key], dtype=float).reshape(-1)
    if atoms.calc is not None and "charges" in atoms.calc.results:
        return np.asarray(atoms.calc.results["charges"], dtype=float).reshape(-1)
    raise KeyError("frame without atomic charges")


def _read_extxyz(path: Path) -> list[dict]:
    """Structure dicts of an extended XYZ file with per-atom charges."""
    from ase.io import read

    from ..ase_io import atoms_to_structure

    structures = []
    for atoms in read(str(path), index=":"):
        d = atoms_to_structure(atoms)
        q = _atomic_charges(atoms)
        d["charges"] = q
        d["total_charge"] = float(np.round(q.sum()))
        structures.append(d)
    return structures


register_dataset(HDNNP4GBuilder())
