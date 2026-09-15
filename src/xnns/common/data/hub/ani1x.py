"""ANI-1x dataset builder (single pyanitools HDF5 from figshare).

The ANI-1x data set holds ~5 million off-equilibrium conformations for H/C/N/O
organic molecules, selected by **active learning** rather than the dense
Normal-Mode Sampling of :mod:`~xnns.common.data.hub.ani1`. This is the data the
ANI-1x *model* (:meth:`xnns.dnn.models.ani.ANI.ani1x`) was trained on, so it is
the natural companion to that preset. Unlike ANI-1 it also ships **forces** and
several levels of theory in one ~5.6 GB file ``ani1x-release.h5``. Each molecule
is a top-level HDF5 group holding ``coordinates`` / ``atomic_numbers`` and a set
of per-property datasets keyed ``<method>.<property>`` (e.g. ``wb97x_dz.energy``,
``wb97x_dz.forces``, ``ccsd(t)_cbs.energy``). Not every property is computed for
every conformation, so entries are NaN-masked per conformation on read.

Reference
---------
Smith *et al.*, "Less is more: Sampling chemical space with active learning",
*J. Chem. Phys.* **148**, 241733 (2018) (the ANI-1x model), and Smith *et al.*,
"The ANI-1ccx and ANI-1x data sets, coupled-cluster and density functional
theory properties for molecules", *Sci. Data* **7**, 134 (2020) (the release
used here). Data: https://doi.org/10.6084/m9.figshare.10047041
Format / reader spec: https://github.com/aiqm/ANI1x_datasets

Notes
-----
* Upstream energies are in **Hartree**, forces in **Hartree/angstrom**, and
  positions in **angstrom**. By default energies/forces are converted to eV and
  eV/angstrom (the convention used elsewhere in xnns); ``units="hartree"`` keeps
  the raw values.
* ``level`` selects the level of theory. ``"wb97x_dz"`` (default, wB97X/6-31G(d)
  -- the level the ANI-1x model was fit to) and ``"wb97x_tz"`` (wB97X/def2-TZVPP)
  carry forces; ``"ccsd(t)_cbs"`` (the ANI-1ccx target) is energy-only. The
  coupled-cluster subset also has its own registered name --
  ``load_dataset("ani1ccx")`` (see :mod:`~xnns.common.data.hub.ani1ccx`), which
  shares this builder and the cached release file.
* The one ~5.6 GB file is downloaded once (cached and MD5-verified). Use
  ``max_molecules`` / ``max_conformations`` to cap the amount materialised --
  the full set will not fit in memory at once.
* ANI-1x ships no official split. ``split`` in ``{"train", "val", "test"}``
  applies a per-conformation 80/10/10 partition with a fixed seed, so splits are
  disjoint and reproducible; ``split=None`` returns ``{"all": ...}``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
from tqdm.auto import tqdm

from ._download import download_file
from .base import DatasetBuilder, register_dataset

# figshare: ani1x-release.h5 (ANI-1x Dataset Release, article 10047041)
_URL = "https://ndownloader.figshare.com/files/18112775"
_MD5 = "98090dd6679106da861f52bed825ffb7"
_FILENAME = "ani1x-release.h5"

# 1 Hartree in eV (CODATA 2018), matching ase.units.Hartree.
_HARTREE_TO_EV = 27.211386245988

# Level of theory -> (energy key, forces key or None). Only these levels are
# exposed; others in the file (hf_*, mp2_*, npno_ccsd(t)_*, ...) are correlation
# energies or single properties without a matching force field.
_LEVELS = {
    "wb97x_dz": ("wb97x_dz.energy", "wb97x_dz.forces"),
    "wb97x_tz": ("wb97x_tz.energy", "wb97x_tz.forces"),
    "ccsd(t)_cbs": ("ccsd(t)_cbs.energy", None),
}


class ANI1xBuilder(DatasetBuilder):
    """Builder for the ANI-1x data set (active-learning DFT energies & forces).

    See the module docstring for the dataset description and citation. The one
    ~5.6 GB ``ani1x-release.h5`` file is downloaded once; its molecule groups are
    then parsed into xnns structure dicts, keeping only the conformations for
    which the requested ``level`` has non-NaN values.
    """

    name = "ani1x"
    description = ("ANI-1x: ~5M active-learning-selected conformations with "
                  "wB97X energies & forces for H/C/N/O molecules.")

    def load(self, *, split: Optional[str] = None, cache_dir: Path,
             level: str = "wb97x_dz", forces: bool = True, units: str = "eV",
             max_molecules: Optional[int] = None,
             max_conformations: Optional[int] = None, seed: int = 1234,
             quiet: bool = False) -> Union[dict[str, list[dict]], list[dict]]:
        """Download and preprocess the ANI-1x data set.

        Parameters
        ----------
        split : str or None
            ``None`` returns ``{"all": ...}``; ``"train"`` / ``"val"`` /
            ``"test"`` returns a per-conformation 80/10/10 partition (fixed
            ``seed``, disjoint splits).
        cache_dir : pathlib.Path
            Base cache directory; files live under ``cache_dir/"ani1x"``.
        level : str, optional
            Level of theory: ``"wb97x_dz"`` (default, the level the ANI-1x model
            was trained on), ``"wb97x_tz"``, or ``"ccsd(t)_cbs"`` (energy-only).
        forces : bool, optional
            Include forces (default ``True``). Ignored for energy-only levels
            like ``"ccsd(t)_cbs"``; raises if forces are requested for a level
            that has none.
        units : str, optional
            ``"eV"`` (default) converts energies to eV and forces to eV/A;
            ``"hartree"`` keeps the raw upstream values.
        max_molecules : int, optional
            Cap the number of molecule groups read (useful for demos).
        max_conformations : int, optional
            Cap the number of (non-NaN) conformations kept per molecule.
        seed : int, optional
            Seed for the reproducible 80/10/10 split. Defaults to ``1234``.
        quiet : bool, optional
            Suppress progress output. Defaults to ``False``.

        Returns
        -------
        dict of {str: list of dict} or list of dict
            Structure dicts with keys ``pos`` ``(N, 3)``, ``atomic_numbers``
            ``(N,)``, ``energy`` (scalar) and, when available and requested,
            ``forces`` ``(N, 3)``. ANI-1x is molecular, so no ``cell`` / ``pbc``.

        Raises
        ------
        ValueError
            If ``units``, ``split``, or ``level`` is unrecognized, or forces are
            requested for an energy-only level.
        ImportError
            If ``h5py`` is not installed.
        """
        if units not in ("eV", "hartree", "Hartree"):
            raise ValueError(f"units must be 'eV' or 'hartree', got {units!r}")
        if level not in _LEVELS:
            raise ValueError(
                f"unknown level {level!r}; choose from {sorted(_LEVELS)}")
        energy_key, force_key = _LEVELS[level]
        if forces and force_key is None:
            raise ValueError(
                f"level {level!r} has no forces; pass forces=False")
        want_forces = forces and force_key is not None
        scale = 1.0 if units.lower() == "hartree" else _HARTREE_TO_EV

        root = Path(cache_dir) / self.name
        h5_path = self._ensure_file(root, quiet)

        structures = self._read_h5(
            h5_path, energy_key, force_key if want_forces else None, scale,
            max_molecules, max_conformations, quiet)

        if split is None:
            return {"all": structures}
        if split not in ("train", "val", "test"):
            raise ValueError(
                f"unknown split {split!r}; use 'train', 'val', 'test', or None")
        return self._partition(structures, seed)[split]

    def _ensure_file(self, root: Path, quiet: bool) -> Path:
        """Download (once, MD5-verified) the release file; return its path."""
        return download_file(_URL, root / "raw" / _FILENAME, _MD5, quiet=quiet)

    def _read_h5(self, h5_path: Path, energy_key: str,
                 force_key: Optional[str], scale: float,
                 max_molecules: Optional[int],
                 max_conformations: Optional[int], quiet: bool) -> list[dict]:
        """Parse ``ani1x-release.h5`` into structure dicts, NaN-masked per level.

        Each top-level group is one molecule holding all its conformations. A
        conformation is kept only if the requested energy (and forces, when
        requested) are finite -- upstream leaves un-computed properties as NaN.
        """
        import h5py

        out: list[dict] = []
        with h5py.File(h5_path, "r") as f:
            groups = list(f.values())
            if max_molecules is not None:
                groups = groups[:max_molecules]
            for g in tqdm(groups, desc=f"{self.name}:{energy_key}", unit=" mol",
                          disable=quiet, leave=False):
                if energy_key not in g:
                    continue
                z = np.asarray(g["atomic_numbers"][()], dtype=np.int64)
                coords = np.asarray(g["coordinates"][()], dtype=np.float64)
                energies = np.asarray(g[energy_key][()], dtype=np.float64)

                mask = np.isfinite(energies)
                forces = None
                if force_key is not None:
                    if force_key not in g:
                        continue
                    forces = np.asarray(g[force_key][()], dtype=np.float64)
                    mask &= np.isfinite(forces).all(axis=(1, 2))
                keep = np.nonzero(mask)[0]
                if max_conformations is not None:
                    keep = keep[:max_conformations]

                for i in keep:
                    rec = {
                        "pos": coords[i],
                        "atomic_numbers": z,
                        "energy": float(energies[i]) * scale,
                    }
                    if forces is not None:
                        rec["forces"] = forces[i] * scale
                    out.append(rec)
        return out

    @staticmethod
    def _partition(structures: list[dict],
                   seed: int) -> dict[str, list[dict]]:
        """Deterministic 80/10/10 train/val/test partition."""
        idx = np.arange(len(structures))
        np.random.default_rng(seed).shuffle(idx)
        n_train = int(0.8 * len(idx))
        n_val = int(0.1 * len(idx))
        parts = {"train": idx[:n_train],
                 "val": idx[n_train:n_train + n_val],
                 "test": idx[n_train + n_val:]}
        return {k: [structures[i] for i in v] for k, v in parts.items()}


register_dataset(ANI1xBuilder())
