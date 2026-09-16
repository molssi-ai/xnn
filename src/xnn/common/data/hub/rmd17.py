"""Revised MD17 (rMD17) dataset builder.

rMD17 recomputes 100,000 conformations of ten small molecules from the original
MD17 at the PBE/def2-SVP level with tight SCF and dense grids, so it is
practically free of numerical noise. Each molecule ships as a ``.npz`` and the
authors provide five fixed train/test index splits of 1000 structures each.

Reference
---------
Christensen & von Lilienfeld, "On the role of gradients for machine learning of
molecular energies and forces", *Mach. Learn.: Sci. Technol.* **1** 045018
(2020). Data: https://figshare.com/articles/dataset/12672038

Notes
-----
* Upstream units are kcal/mol (energies) and kcal/mol/angstrom (forces);
  positions are in angstrom. By default these are converted to eV and eV/A --
  the convention used elsewhere in xnn and by MACE/NequIP -- controlled by the
  ``units`` argument.
* The authors warn: **do not train on more than 1000 samples**, as the
  conformations are consecutive MD frames and thus correlated. The five official
  splits respect this; ``split="all"`` (100k frames) is offered only for custom
  splitting.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
from tqdm.auto import tqdm

from ._download import download_file
from .base import DatasetBuilder, register_dataset

_FIGSHARE = "https://ndownloader.figshare.com/files/{file_id}"

# eV per kcal/mol == ase.units.kcal / ase.units.mol (hard-coded to keep this a
# pure-numpy path with no ASE dependency).
_KCAL_MOL_TO_EV = 0.0433641153087705

# molecule -> (figshare file id, md5) for the per-molecule .npz files.
_MOLECULES: dict[str, tuple[str, str]] = {
    "aspirin":       ("62265757", "17fd6fb69066888613f7e16b358a7553"),
    "azobenzene":    ("62265754", "be79df918468eb3579aa73a0becf7390"),
    "benzene":       ("62265739", "18c9242bc90fbf28215f6dd81e650f16"),
    "ethanol":       ("62265733", "eb837fb8deb27d4e0d52f71a03aff776"),
    "malonaldehyde": ("62265736", "cdc1d70c0c34062ddde6e5071eb6fe21"),
    "naphthalene":   ("62265751", "0efba19c9907e3852318b1e6008b3b9e"),
    "paracetamol":   ("62265760", "ba10784f7b67635427085f6d7ec2dd97"),
    "salicylic":     ("62265748", "900fedf242da438400fa4293348d7dd1"),
    "toluene":       ("62265742", "0f2913d51f8149c90ab28d697a076f64"),
    "uracil":        ("62265745", "992a4479c28a07e0cce6da964805be31"),
}

# common spellings mapped to the canonical molecule key.
_ALIASES: dict[str, str] = {
    "salicylic_acid": "salicylic",
    "salicylic acid": "salicylic",
}

# (split, fold) -> (figshare file id, md5) for the index CSVs (1000 ints each).
_SPLITS: dict[tuple[str, int], tuple[str, str]] = {
    ("train", 1): ("62265793", "22264dcf2e42d6133362411abbbe4ce8"),
    ("train", 2): ("62265772", "28dbc9a3d5e7c47e653b1cc01809df35"),
    ("train", 3): ("62265775", "8de68b1b288c790e633c9d05ff5f04dd"),
    ("train", 4): ("62265766", "3764257abf8121a0905f364468ba83da"),
    ("train", 5): ("62265778", "4670362bd554c9c22581c936432236b3"),
    ("test", 1):  ("62265781", "124c8a0849e84e244d6ce8424e1c05cf"),
    ("test", 2):  ("62265769", "9dc098724c1680daca038b785ee4f602"),
    ("test", 3):  ("62265787", "fd96f4623291dc23dd80b24dd634ca0b"),
    ("test", 4):  ("62265790", "f0acce759fa4fe46b3a0b2a450e7e260"),
    ("test", 5):  ("62265784", "4892da12e88a3c30c9a15f1f95e6eaac"),
}


class RMD17Builder(DatasetBuilder):
    """Builder for the revised MD17 dataset.

    See the module docstring for the dataset description and citation. Only the
    requested molecule's ``.npz`` (and, for the standard splits, the two index
    CSVs) are downloaded, so a single molecule costs far less than the 1 GB
    bundle.
    """

    name = "rmd17"
    description = "Revised MD17: 10 small molecules, PBE/def2-SVP energies & forces."

    def load(self, *, split: Optional[str] = None, cache_dir: Path,
             molecule: Optional[str] = None, fold: int = 1, units: str = "eV",
             n_train: Optional[int] = None, n_test: Optional[int] = None,
             quiet: bool = False) -> Union[dict[str, list[dict]], list[dict]]:
        """Download and preprocess one rMD17 molecule.

        Parameters
        ----------
        split : str or None
            ``None`` returns ``{"train": ..., "test": ...}`` for the chosen
            ``fold``; ``"train"`` / ``"test"`` returns that split; ``"all"``
            returns every conformation (no split filtering), for custom splits.
        cache_dir : pathlib.Path
            Base cache directory; files are stored under ``cache_dir/"rmd17"``.
        molecule : str
            Which molecule to load (required). One of
            ``aspirin, azobenzene, benzene, ethanol, malonaldehyde,
            naphthalene, paracetamol, salicylic, toluene, uracil``.
        fold : int, optional
            Which of the five official splits (1-5) to use. Defaults to ``1``.
            Ignored when ``split="all"``.
        units : str, optional
            ``"eV"`` (default) converts energies to eV and forces to eV/A;
            ``"kcal/mol"`` keeps the upstream units.
        n_train, n_test : int, optional
            Truncate the train/test split to the first ``n`` structures. The
            official splits hold 1000 each; the authors warn against training on
            more than 1000 samples.
        quiet : bool, optional
            Suppress download progress output. Defaults to ``False``.

        Returns
        -------
        dict of {str: list of dict} or list of dict
            Structure dicts with keys ``pos`` ``(N, 3)``, ``atomic_numbers``
            ``(N,)``, ``energy`` (scalar) and ``forces`` ``(N, 3)``. rMD17 is
            molecular, so no ``cell`` / ``pbc`` is set.

        Raises
        ------
        ValueError
            If ``molecule`` is missing/unknown, ``fold`` is not in 1-5, or
            ``units`` / ``split`` is unrecognized.
        """
        mol = self._resolve_molecule(molecule)
        if units not in ("eV", "kcal/mol", "kcal"):
            raise ValueError(f"units must be 'eV' or 'kcal/mol', got {units!r}")
        root = Path(cache_dir) / self.name
        raw = root / "raw"

        file_id, md5 = _MOLECULES[mol]
        npz_path = download_file(_FIGSHARE.format(file_id=file_id),
                                 raw / f"rmd17_{mol}.npz", md5, quiet=quiet)
        data = np.load(npz_path)
        z = data["nuclear_charges"].astype(np.int64)
        coords = data["coords"]        # (n_conf, n_atoms, 3) angstrom
        energies = data["energies"]    # (n_conf,) kcal/mol
        forces = data["forces"]        # (n_conf, n_atoms, 3) kcal/mol/angstrom

        scale = 1.0 if units in ("kcal/mol", "kcal") else _KCAL_MOL_TO_EV

        def build(indices, desc: str) -> list[dict]:
            """Materialize structure dicts for the given conformation indices."""
            return [
                {
                    "pos": coords[i],
                    "atomic_numbers": z,
                    "energy": float(energies[i]) * scale,
                    "forces": forces[i] * scale,
                }
                for i in tqdm(indices, desc=desc, unit=" struct",
                              disable=quiet, leave=False)
            ]

        if split == "all":
            return build(range(len(energies)), f"rmd17:{mol} all")

        train_idx = self._read_split(root, "train", fold, quiet)
        test_idx = self._read_split(root, "test", fold, quiet)
        if n_train is not None:
            train_idx = train_idx[:n_train]
        if n_test is not None:
            test_idx = test_idx[:n_test]

        splits = {"train": build(train_idx, f"rmd17:{mol} train"),
                  "test": build(test_idx, f"rmd17:{mol} test")}
        if split is None:
            return splits
        if split not in splits:
            raise ValueError(
                f"unknown split {split!r}; use 'train', 'test', 'all', or None")
        return splits[split]

    @staticmethod
    def _resolve_molecule(molecule: Optional[str]) -> str:
        """Normalize a molecule name to a canonical key, validating it.

        Parameters
        ----------
        molecule : str or None
            User-supplied name (case-insensitive; a few aliases accepted).

        Returns
        -------
        str
            Canonical molecule key present in :data:`_MOLECULES`.

        Raises
        ------
        ValueError
            If ``molecule`` is ``None`` or not a known molecule.
        """
        if molecule is None:
            raise ValueError(
                "rmd17 requires a `molecule` argument; choose one of: "
                + ", ".join(sorted(_MOLECULES)))
        key = molecule.strip().lower()
        key = _ALIASES.get(key, key)
        if key not in _MOLECULES:
            raise ValueError(
                f"unknown rmd17 molecule {molecule!r}; choose one of: "
                + ", ".join(sorted(_MOLECULES)))
        return key

    @staticmethod
    def _read_split(root: Path, split: str, fold: int, quiet: bool) -> list[int]:
        """Download and parse an official split's index CSV.

        Parameters
        ----------
        root : pathlib.Path
            Dataset cache root (``cache_dir/"rmd17"``).
        split : str
            ``"train"`` or ``"test"``.
        fold : int
            Split number, 1-5.
        quiet : bool
            Suppress download progress output.

        Returns
        -------
        list of int
            Zero-based conformation indices into the molecule's arrays.

        Raises
        ------
        ValueError
            If ``fold`` is not one of the five available splits.
        """
        try:
            file_id, md5 = _SPLITS[(split, int(fold))]
        except KeyError:
            raise ValueError(f"fold must be 1-5, got {fold!r}") from None
        path = download_file(_FIGSHARE.format(file_id=file_id),
                             root / "raw" / f"index_{split}_{int(fold):02d}.csv",
                             md5, quiet=quiet)
        return [int(v) for v in np.loadtxt(path, dtype=np.int64).ravel()]


register_dataset(RMD17Builder())
