"""Builder for the electronic-state datasets of the SpookyNet paper.

Unke *et al.* (*Nat. Commun.* 12, 7273, 2021, Fig. 4) published two GFN2-xTB
datasets in which the same geometries occur in two electronic states, so a
model can only tell them apart from the total charge or the spin:

* ``ag3``: 2200 Ag3+ and Ag3- clusters;
* ``carbene``: 2200 singlet and triplet CH2 molecules.

550 structures were sampled around the minimum of each state by normal-mode
sampling at 1000 K and every structure was recomputed in the other state.
Each entry has the total charge, the number of unpaired electrons
(``spin_multiplicity = S + 1``), the energy (eV), the forces (eV/Angstrom)
and the dipole moment about the origin (e Angstrom).

Notes
-----
* Data: Zenodo record 10.5281/zenodo.5115732 (CC BY 4.0), SQLite files read
  with the standard library.
* The paper trained on 1000 random structures; the builder draws the
  splits from a seeded permutation (1000 / 100 / 1100 by default), which is
  reproducible but not the authors' own.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ._download import download_file
from .base import DatasetBuilder, register_dataset

_URL = "https://zenodo.org/records/5115732/files/{name}?download=1"
_FILES = {"ag3": ("ag3_2200.db", "941ee5359039b43f38d2b3a85cd02e63"),
          "carbene": ("carbene_2200.db", "c3b31993e7f306f1556cd0b60141014a")}
_ALIASES = {"ag": "ag3", "ag3+/ag3-": "ag3", "ch2": "carbene"}


def _read(path: Path) -> list[dict]:
    """Structure dicts of one SQLite file, in file order."""
    structures = []
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        rows = db.execute("SELECT Q, S, Z, R, E, F, D FROM data ORDER BY id").fetchall()
    for q, s, z, r, e, f, d in rows:
        numbers = np.frombuffer(z, dtype="<i4").astype(np.int64)
        n = numbers.shape[0]
        structures.append({
            "atomic_numbers": numbers,
            "pos": np.frombuffer(r, dtype="<f4").reshape(n, 3).astype(np.float64),
            "energy": float(e),
            "forces": np.frombuffer(f, dtype="<f4").reshape(n, 3).astype(np.float64),
            "dipole": np.frombuffer(d, dtype="<f4").reshape(3).astype(np.float64),
            "total_charge": float(q or 0.0),
            "spin_multiplicity": float(s or 0.0) + 1.0,
        })
    return structures


class SpookyNetStatesBuilder(DatasetBuilder):
    """Builder for the Ag3+/Ag3- and singlet/triplet CH2 sets."""

    name = "spookynet_states"
    description = ("SpookyNet electronic states (Unke 2021, Fig. 4): Ag3+/Ag3- and "
                   "singlet/triplet CH2, 2200 GFN2-xTB structures each, with charges "
                   "and spins.")

    def load(self, *, split: Optional[str] = None, cache_dir: Path, system: str = "ag3",
             n_train: int = 1000, n_val: int = 100, seed: int = 0,
             quiet: bool = False) -> Union[dict[str, list[dict]], list[dict]]:
        """Download and read one of the two sets.

        Parameters
        ----------
        split : str or None
            ``None`` returns ``{"train": ..., "val": ..., "test": ...}``;
            ``"train"`` / ``"val"`` / ``"test"`` returns that split; ``"all"``
            returns every structure in file order.
        cache_dir : pathlib.Path
            Base cache directory; the files are stored under
            ``cache_dir/"spookynet_states"``.
        system : str, optional
            ``"ag3"`` (default) or ``"carbene"``.
        n_train, n_val : int, optional
            Sizes of the training and validation splits (1000 and 100); the
            remaining structures are the test split.
        seed : int, optional
            Seed of the permutation. Defaults to 0.
        quiet : bool, optional
            Suppress download progress output.

        Returns
        -------
        dict of {str: list of dict} or list of dict
            Structure dicts with ``pos``, ``atomic_numbers``, ``energy``,
            ``forces``, ``dipole``, ``total_charge`` and ``spin_multiplicity``.

        Raises
        ------
        ValueError
            For an unknown ``system`` or ``split``, or splits larger than the set.
        """
        key = _ALIASES.get(str(system).lower(), str(system).lower())
        if key not in _FILES:
            raise ValueError(f"system must be one of {sorted(_FILES)}, got {system!r}")
        if split not in (None, "train", "val", "test", "all"):
            raise ValueError(f"split must be None, 'train', 'val', 'test' or 'all', got {split!r}")
        name, md5 = _FILES[key]
        path = download_file(_URL.format(name=name), cache_dir / self.name / "raw" / name, md5,
                             quiet=quiet)
        structures = _read(path)
        if split == "all":
            return structures
        if n_train + n_val > len(structures):
            raise ValueError(f"n_train + n_val = {n_train + n_val} exceeds the "
                             f"{len(structures)} structures")
        perm = np.random.default_rng(seed).permutation(len(structures))
        splits = {"train": [structures[i] for i in perm[:n_train]],
                  "val": [structures[i] for i in perm[n_train:n_train + n_val]],
                  "test": [structures[i] for i in perm[n_train + n_val:]]}
        return splits if split is None else splits[split]


register_dataset(SpookyNetStatesBuilder())
