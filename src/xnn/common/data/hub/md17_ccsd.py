"""MD17 at the coupled-cluster level (the sGDML CCSD(T) data sets) builder.

Conformations of five small molecules from MD17 trajectories, recomputed with CCSD(T) (aspirin with CCSD) by
the sGDML authors, with energies and forces and fixed train and test splits: 1000 training structures and 500
test structures (ethanol 1000, toluene 501).

Reference
---------
Chmiela, Sauceda, Müller and Tkatchenko, "Towards exact molecular dynamics simulations with machine-learned
force fields", *Nat. Commun.* **9**, 3887 (2018). Data: http://www.sgdml.org/#datasets

Notes
-----
* Upstream units are kcal/mol (energies) and kcal/mol/angstrom (forces); positions are in angstrom. By
  default these are converted to eV and eV/A, controlled by the ``units`` argument.
* Levels of theory: aspirin CCSD/cc-pVDZ, ethanol CCSD(T)/cc-pVTZ, the others CCSD(T)/cc-pVDZ.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from ._download import download_file, extract_archive
from .base import DatasetBuilder, register_dataset
from .units import KCAL_MOL_TO_EV as _KCAL_MOL_TO_EV

_URL = "https://sgdml.org/secure_proxy.php?file=data/npz/{stem}.zip"

# molecule -> (archive stem, md5 of the archive)
_MOLECULES: dict[str, tuple[str, str]] = {
    "aspirin":       ("aspirin_ccsd", "42c19ed61d1904c227c0a7d1794409f6"),
    "benzene":       ("benzene_ccsd_t", "39aebc564c099e5ceb74504fb6fa9ab6"),
    "ethanol":       ("ethanol_ccsd_t", "ab7c156fa546fce11ff31176fcfa1fc2"),
    "malonaldehyde": ("malonaldehyde_ccsd_t", "11a81be8e89acb43d2905aacad68ee2f"),
    "toluene":       ("toluene_ccsd_t", "7ac4fa7128ee1861d4611a576f8c84d0"),
}


class MD17CCSDBuilder(DatasetBuilder):
    """Builder for the coupled-cluster MD17 data sets (see the module docstring)."""

    name = "md17_ccsd"
    description = "MD17 at CCSD(T): 5 small molecules with fixed train/test splits."

    def load(self, *, split: Optional[str] = None, cache_dir: Path, molecule: Optional[str] = None,
             units: str = "eV", n_train: Optional[int] = None, n_test: Optional[int] = None,
             quiet: bool = False) -> Union[dict[str, list[dict]], list[dict]]:
        """Download and preprocess one molecule.

        Parameters
        ----------
        split : str or None
            ``None`` returns ``{"train": ..., "test": ...}``; ``"train"`` or ``"test"`` returns that split.
        cache_dir : pathlib.Path
            Base cache directory; files are stored under ``cache_dir/"md17_ccsd"``.
        molecule : str
            ``aspirin``, ``benzene``, ``ethanol``, ``malonaldehyde`` or ``toluene`` (required).
        units : str, optional
            ``"eV"`` (default) converts energies to eV and forces to eV/A; ``"kcal/mol"`` keeps the upstream
            units.
        n_train, n_test : int, optional
            Truncate the splits to their first ``n`` structures.
        quiet : bool, optional
            Suppress download progress output.

        Returns
        -------
        dict of {str: list of dict} or list of dict
            Structure dicts with ``pos``, ``atomic_numbers``, ``energy`` and ``forces``.

        Raises
        ------
        ValueError
            For a missing or unknown molecule, split or unit.
        """
        if molecule is None or molecule.strip().lower() not in _MOLECULES:
            raise ValueError(f"md17_ccsd needs a `molecule`, one of: {', '.join(sorted(_MOLECULES))}; "
                             f"got {molecule!r}")
        if units not in ("eV", "kcal/mol", "kcal"):
            raise ValueError(f"units must be 'eV' or 'kcal/mol', got {units!r}")
        if split not in (None, "train", "test"):
            raise ValueError(f"unknown split {split!r}; use 'train', 'test' or None")
        stem, md5 = _MOLECULES[molecule.strip().lower()]
        raw = Path(cache_dir) / self.name / "raw"
        archive = download_file(_URL.format(stem=stem), raw / f"{stem}.zip", md5, quiet=quiet)
        names = {part: f"{stem}-{part}.npz" for part in ("train", "test")}
        if not all((raw / n).exists() for n in names.values()):
            extract_archive(archive, raw, select=lambda member: member in names.values())
        scale = 1.0 if units in ("kcal/mol", "kcal") else _KCAL_MOL_TO_EV
        limits = {"train": n_train, "test": n_test}
        splits = {}
        for part, filename in names.items():
            with np.load(raw / filename) as data:      # every key access of an npz reads its array again
                z, R, E, F = data["z"].astype(np.int64), data["R"], data["E"].reshape(-1), data["F"]
            count = len(E) if limits[part] is None else min(limits[part], len(E))
            splits[part] = [{"pos": R[i], "atomic_numbers": z, "energy": float(E[i]) * scale,
                             "forces": F[i] * scale} for i in range(count)]
        return splits if split is None else splits[split]


register_dataset(MD17CCSDBuilder())
