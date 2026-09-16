"""Argon MD dataset builder (bundled with xnn).

A small dataset of periodic **argon** configurations with reference energies,
forces and stress, used by the example notebooks
(``examples/*/*_argon_train_test.ipynb`` and ``*_argon_density_md.ipynb``). It is
stored in the MACE convention (extxyz with ``REF_energy`` / ``REF_forces`` /
``REF_stress`` and a leading ``config_type=IsolatedAtom`` reference frame) and
ships **inside the repository** under ``datasets/argon_md/``, so it loads offline
with no download.

Notes
-----
* Energies are in eV, forces in eV/A, positions in angstrom, stress in eV/A^3 --
  the conventions used elsewhere in xnn.
* The ``config_type=IsolatedAtom`` reference frames (the per-element ``E0``) are
  dropped from the returned structures; for argon ``E0`` is ``0.0``, so no
  atomic-energy shift is needed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from ..ase_io import atoms_to_structure
from .base import DatasetBuilder, register_dataset

_FILES = {"train": "argon_train.xyz", "test": "argon_test.xyz"}


class ArgonMDBuilder(DatasetBuilder):
    """Builder for the bundled argon MD dataset.

    Reads the extxyz files shipped under ``datasets/argon_md/`` (MACE
    convention) and converts them to xnn structure dicts. No download: the
    files travel with the repository.
    """

    name = "argon_md"
    description = ("Argon MD: periodic Ar configurations with reference "
                   "energies, forces and stress (bundled, MACE convention).")

    def load(self, *, split: Optional[str] = None, cache_dir: Path,
             quiet: bool = False) -> Union[dict[str, list[dict]], list[dict]]:
        """Load the argon MD dataset.

        Parameters
        ----------
        split : str or None
            ``None`` returns ``{"train": ..., "test": ...}``; ``"train"`` /
            ``"test"`` returns that split's list; ``"all"`` returns train + test
            concatenated.
        cache_dir : pathlib.Path
            Base directory; files are read from ``cache_dir/"argon_md"``
            (the repo's ``datasets/argon_md/`` by default).
        quiet : bool, optional
            Unused (kept for interface consistency). Defaults to ``False``.

        Returns
        -------
        dict of {str: list of dict} or list of dict
            Structure dicts with keys ``pos``, ``atomic_numbers``, ``cell``,
            ``pbc``, ``energy``, ``forces`` and ``stress``. The
            ``config_type=IsolatedAtom`` reference frames are dropped.

        Raises
        ------
        ValueError
            If ``split`` is unrecognized.
        FileNotFoundError
            If the bundled extxyz files are missing.
        ImportError
            If ``ase`` is not installed.
        """
        root = Path(cache_dir) / self.name
        splits: dict[str, list[dict]] = {}
        for name, fname in _FILES.items():
            path = root / fname
            if not path.exists():
                raise FileNotFoundError(
                    f"missing {path}; the argon_md dataset ships under "
                    f"datasets/argon_md/ in the repository")
            splits[name] = self._read(path)

        if split is None:
            return splits
        if split == "all":
            return splits["train"] + splits["test"]
        if split not in splits:
            raise ValueError(
                f"unknown split {split!r}; use 'train', 'test', 'all', or None")
        return splits[split]

    @staticmethod
    def _read(path: Path) -> list[dict]:
        """Parse one extxyz file into structure dicts, skipping IsolatedAtom."""
        from ase.io import read

        out: list[dict] = []
        for atoms in read(str(path), index=":"):
            if atoms.info.get("config_type") == "IsolatedAtom":
                continue
            out.append(atoms_to_structure(
                atoms, energy_key="REF_energy", forces_key="REF_forces",
                stress_key="REF_stress"))
        return out


register_dataset(ArgonMDBuilder())
