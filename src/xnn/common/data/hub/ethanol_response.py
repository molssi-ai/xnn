"""Ethanol response-property dataset builder (the FieldSchNet reference data).

10,000 ethanol conformations, drawn at random from MD17 and recomputed at
the PBE0/def2-TZVP level (ORCA 4.0.1.2) with energies, forces, dipole
moments, polarizability tensors and nuclear shielding tensors. Released
with the FieldSchNet paper and the training data of the PaiNN spectra
experiment (8000 / 1000 / 1000 structures for training, validation and
test).

References
----------
M. Gastegger, K. T. Schuett, K.-R. Mueller, "Machine learning of solvent
effects on molecular spectra and reactions", *Chem. Sci.* **12**, 11473
(2021). K. T. Schuett, O. T. Unke, M. Gastegger, "Equivariant message
passing for the prediction of tensorial properties and molecular spectra",
ICML 2021. Data: http://www.quantum-machine.org/datasets/
(``ethanol_vacuum.tgz``).

Notes
-----
* Upstream units are atomic units throughout (Bohr, Hartree, Hartree/Bohr,
  e Bohr, Bohr^3); by default they are converted to Angstrom, eV, eV/Angstrom,
  e Angstrom and Angstrom^3 (``units="eV"``). ``units="au"`` keeps them.
* The archive holds an ASE database, so reading it needs ``ase``.
* The published split is a random 8000 / 1000 / 1000 partition; the builder
  draws it from a seeded permutation (``seed``), so it is reproducible but
  not the authors' own.
"""
from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ._download import download_file
from .base import DatasetBuilder, register_dataset
from .units import BOHR_TO_ANGSTROM, HARTREE_TO_EV

_URL = "https://quantum-machine.org/data/fieldschnet/ethanol_vacuum.tgz"
_MD5 = "3d4e3d45d6496dc0430d5ba7d0db39d0"
_DB = "ethanol_vacuum.db"


class EthanolResponseBuilder(DatasetBuilder):
    """Builder for the ethanol energies / forces / dipoles / polarizabilities set."""

    name = "ethanol_response"
    description = ("Ethanol (FieldSchNet / PaiNN): 10k MD17 conformations at PBE0/def2-TZVP "
                   "with energies, forces, dipole moments and polarizability tensors.")

    def load(self, *, split: Optional[str] = None, cache_dir: Path,
             n_train: int = 8000, n_val: int = 1000, n_test: int = 1000,
             seed: int = 0, units: str = "eV", shielding: bool = False,
             quiet: bool = False) -> Union[dict[str, list[dict]], list[dict]]:
        """Download and preprocess the ethanol set.

        Parameters
        ----------
        split : str or None
            ``None`` returns ``{"train": ..., "val": ..., "test": ...}``;
            ``"train"`` / ``"val"`` / ``"test"`` returns that split; ``"all"``
            returns every conformation in file order.
        cache_dir : pathlib.Path
            Base cache directory; the archive is stored under
            ``cache_dir/"ethanol_response"``.
        n_train, n_val, n_test : int, optional
            Sizes of the three splits (8000 / 1000 / 1000 in the papers),
            drawn in that order from a seeded random permutation.
        seed : int, optional
            Seed of the permutation. Defaults to 0.
        units : str, optional
            ``"eV"`` (default: Angstrom, eV, eV/Angstrom, e Angstrom,
            Angstrom^3) or ``"au"`` (the published atomic units).
        shielding : bool, optional
            Also return the nuclear shielding tensors (``(N, 3, 3)``, ppm) as
            ``"shielding"``. Defaults to ``False``.
        quiet : bool, optional
            Suppress download progress output.

        Returns
        -------
        dict of {str: list of dict} or list of dict
            Structure dicts with ``pos`` ``(N, 3)``, ``atomic_numbers``
            ``(N,)``, ``energy``, ``forces`` ``(N, 3)``, ``dipole`` ``(3,)``
            and ``polarizability`` ``(3, 3)``.

        Raises
        ------
        ImportError
            If ase is not installed.
        ValueError
            For an unknown ``units`` or ``split``, or splits larger than the set.
        """
        if units not in ("eV", "au"):
            raise ValueError(f"units must be 'eV' or 'au', got {units!r}")
        if split not in (None, "train", "val", "test", "all"):
            raise ValueError(f"split must be None, 'train', 'val', 'test' or 'all', got {split!r}")
        try:
            from ase.db import connect
        except ImportError as err:
            raise ImportError("reading the ethanol database needs ase: "
                              'pip install "xnn[ase]"') from err
        root = cache_dir / self.name
        archive = download_file(_URL, root / "raw" / "ethanol_vacuum.tgz", _MD5, quiet=quiet)
        db_path = root / "raw" / _DB
        if not db_path.exists():
            with tarfile.open(archive) as tar:
                member = next(m for m in tar.getmembers() if m.name.endswith(_DB))
                member.name = _DB
                tar.extract(member, root / "raw")

        r_scale = BOHR_TO_ANGSTROM if units == "eV" else 1.0
        e_scale = HARTREE_TO_EV if units == "eV" else 1.0
        structures = []
        with connect(str(db_path)) as db:
            for row in db.select():
                d = row.data
                s = {"pos": np.asarray(row.positions, dtype=np.float64) * r_scale,
                     "atomic_numbers": np.asarray(row.numbers, dtype=np.int64),
                     "energy": float(np.asarray(d["energy"]).reshape(-1)[0]) * e_scale,
                     "forces": np.asarray(d["forces"], dtype=np.float64) * (e_scale / r_scale),
                     "dipole": np.asarray(d["dipole_moment"], dtype=np.float64).reshape(3) * r_scale,
                     "polarizability": (np.asarray(d["polarizability"], dtype=np.float64)
                                        .reshape(3, 3) * r_scale ** 3)}
                if shielding:
                    s["shielding"] = np.asarray(d["shielding"], dtype=np.float64)
                structures.append(s)

        if split == "all":
            return structures
        n_total = n_train + n_val + n_test
        if n_total > len(structures):
            raise ValueError(f"n_train + n_val + n_test = {n_total} exceeds the "
                             f"{len(structures)} conformations")
        perm = np.random.default_rng(seed).permutation(len(structures))
        splits = {"train": [structures[i] for i in perm[:n_train]],
                  "val": [structures[i] for i in perm[n_train:n_train + n_val]],
                  "test": [structures[i] for i in perm[n_train + n_val:n_total]]}
        return splits if split is None else splits[split]


register_dataset(EthanolResponseBuilder())
