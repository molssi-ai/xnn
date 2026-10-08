"""QM7 dataset builder.

QM7 (Blum & Reymond 2009; Rupp *et al.*, PRL 108, 058301, 2012) holds 7165
organic molecules of up to 23 atoms (H, C, N, O, S) from GDB-13 with their
PBE0 atomization energies, the benchmark of the spherical CNN of Cohen *et
al.* (ICLR 2018, Sec. 5.4). The authors provide a stratified five-fold split.

Data: http://quantum-machine.org/datasets/ (``qm7.mat``).

Notes
-----
* Upstream positions are in Bohr and energies in kcal/mol; by default both
  are converted to Angstrom and eV (``units="eV"``).
* The file is a MATLAB ``.mat``; reading it needs ``scipy``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from ._download import download_file
from .base import DatasetBuilder, register_dataset
from .units import BOHR_TO_ANGSTROM, KCAL_MOL_TO_EV

_URL = "http://quantum-machine.org/data/qm7.mat"
_MD5 = "2aea1a6ac88793faeafcd61a2491f059"


class QM7Builder(DatasetBuilder):
    """Builder for QM7 (7165 molecules, atomization energies, five folds)."""

    name = "qm7"
    description = "QM7: 7165 GDB-13 molecules (H, C, N, O, S), PBE0 atomization energies."

    def load(self, *, split: Optional[str] = None, cache_dir: Path, fold: int = 1,
             units: str = "eV", n_train: Optional[int] = None, n_test: Optional[int] = None,
             quiet: bool = False) -> Union[dict[str, list[dict]], list[dict]]:
        """Download and preprocess QM7.

        Parameters
        ----------
        split : str or None
            ``None`` returns ``{"train": ..., "test": ...}`` for the chosen
            ``fold``; ``"train"`` / ``"test"`` returns that split; ``"all"``
            returns every molecule in file order.
        cache_dir : pathlib.Path
            Base cache directory; the file is stored under ``cache_dir/"qm7"``.
        fold : int, optional
            Which of the five stratified folds (1-5) is the test set; the
            other four are the training set. Defaults to 1.
        units : str, optional
            ``"eV"`` (default, with positions in Angstrom) or ``"kcal/mol"``
            (energies as published, positions in Bohr).
        n_train, n_test : int, optional
            Truncate the splits to their first ``n`` molecules.
        quiet : bool, optional
            Suppress download progress output.

        Returns
        -------
        dict of {str: list of dict} or list of dict
            Structure dicts with ``pos`` ``(N, 3)``, ``atomic_numbers`` ``(N,)``
            and ``energy`` (the atomization energy).

        Raises
        ------
        ImportError
            If scipy is not installed.
        ValueError
            For an unknown ``units``, ``split`` or ``fold``.
        """
        if units not in ("eV", "kcal/mol", "kcal"):
            raise ValueError(f"units must be 'eV' or 'kcal/mol', got {units!r}")
        if not 1 <= int(fold) <= 5:
            raise ValueError(f"fold must be 1-5, got {fold!r}")
        try:
            from scipy.io import loadmat
        except ImportError as err:
            raise ImportError("reading qm7.mat needs scipy: pip install scipy") from err
        path = download_file(_URL, cache_dir / self.name / "raw" / "qm7.mat", _MD5, quiet=quiet)
        raw = loadmat(path)
        positions, charges = raw["R"], raw["Z"]
        energies, folds = raw["T"].reshape(-1), raw["P"]
        e_scale = KCAL_MOL_TO_EV if units == "eV" else 1.0
        r_scale = BOHR_TO_ANGSTROM if units == "eV" else 1.0

        def build(indices):
            out = []
            for i in indices:
                keep = charges[i] > 0
                out.append({"pos": np.asarray(positions[i][keep], dtype=np.float64) * r_scale,
                            "atomic_numbers": np.asarray(charges[i][keep], dtype=np.int64),
                            "energy": float(energies[i]) * e_scale})
            return out

        if split == "all":
            return build(range(len(energies)))
        test_idx = np.sort(folds[int(fold) - 1].astype(np.int64))
        train_idx = np.sort(np.concatenate([folds[k] for k in range(5) if k != int(fold) - 1]).astype(np.int64))
        if n_train is not None:
            train_idx = train_idx[:n_train]
        if n_test is not None:
            test_idx = test_idx[:n_test]
        splits = {"train": build(train_idx), "test": build(test_idx)}
        if split is None:
            return splits
        if split not in splits:
            raise ValueError(f"unknown split {split!r}; use 'train', 'test', 'all', or None")
        return splits[split]


register_dataset(QM7Builder())
