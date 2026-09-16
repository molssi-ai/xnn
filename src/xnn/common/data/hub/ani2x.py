"""ANI-2x dataset builder (pyanitools HDF5 from the Zenodo release).

The ANI-2x data set (Devereux et al., *J. Chem. Theory Comput.* **16**, 4192,
2020) extends ANI to **seven** elements (adds S, F, Cl to H, C, N, O). It holds
~9.6 million active-learning-selected off-equilibrium conformations with
wB97X/6-31G(d) energies **and forces**, the level the ANI-2x *model*
(:meth:`xnn.dnn.models.ani.ANI.ani2x`) was trained on. Upstream distributes it
as a ~3.7 GB ``ANI-2x-wB97X-631Gd.tar.gz`` archive holding one pyanitools HDF5
file whose top-level groups are keyed by **number of atoms** (``"002"``,
``"003"``, ...). Each group holds ``coordinates`` ``(Nc, Na, 3)``, ``species``
``(Nc, Na)`` (as atomic numbers), ``energies`` ``(Nc,)`` and ``forces``
``(Nc, Na, 3)`` -- so within a group each conformation can be a different
molecule with the same atom count.

Reference
---------
Devereux et al., "Extending the Applicability of the ANI Deep Learning
Molecular Potential to Sulfur and Halogens", *J. Chem. Theory Comput.* **16**,
4192 (2020). Data: https://doi.org/10.5281/zenodo.10108942
Format / reader spec: the Zenodo record's ``sample_data_loader.py`` and
``supplementary_information.pdf`` ("ANI-2x Data Set Guidelines").

Notes
-----
* Upstream energies are in **Hartree**, forces in **Hartree/angstrom**, and
  positions in **angstrom**. By default energies/forces are converted to eV and
  eV/angstrom (the convention used elsewhere in xnn); ``units="hartree"``
  keeps the raw values.
* Only the wB97X/6-31G(d) release is used: it is the level ANI-2x was fit to and
  the only released ANI-2x level that ships usable forces (the def2-TZVPP files
  omit forces upstream).
* The one ~3.7 GB archive is downloaded once (cached and MD5-verified) and
  extracted. Use ``n_atoms`` to select atom-count groups and
  ``max_conformations`` to cap the amount materialised -- the full set is
  ~9.6 M conformations and will not fit in memory at once.
* ANI-2x ships no official split. ``split`` in ``{"train", "val", "test"}``
  applies a per-conformation 80/10/10 partition with a fixed seed, so splits are
  disjoint and reproducible; ``split=None`` returns ``{"all": ...}``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
from tqdm.auto import tqdm

from ._download import download_file, extract_archive
from .base import DatasetBuilder, register_dataset

# Zenodo: ANI-2x Release, record 10108942, wB97X/6-31G(d) archive (with forces).
_URL = "https://zenodo.org/api/records/10108942/files/ANI-2x-wB97X-631Gd.tar.gz/content"
_MD5 = "cb1d9effb3d07fc1cc6ced7cd0b1e1f2"
_ARCHIVE = "ANI-2x-wB97X-631Gd.tar.gz"

# 1 Hartree in eV (CODATA 2018), matching ase.units.Hartree.
_HARTREE_TO_EV = 27.211386245988


class ANI2xBuilder(DatasetBuilder):
    """Builder for the ANI-2x data set (7-element wB97X energies & forces).

    See the module docstring for the dataset description and citation. The one
    ~3.7 GB archive is downloaded and extracted once; its atom-count groups are
    then parsed into xnn structure dicts.
    """

    name = "ani2x"
    description = ("ANI-2x: ~9.6M active-learning conformations with wB97X "
                   "energies & forces for H/C/N/O/S/F/Cl molecules.")

    def load(self, *, split: Optional[str] = None, cache_dir: Path,
             n_atoms: Union[int, list, tuple, None] = None,
             forces: bool = True, units: str = "eV",
             max_groups: Optional[int] = None,
             max_conformations: Optional[int] = None, seed: int = 1234,
             quiet: bool = False) -> Union[dict[str, list[dict]], list[dict]]:
        """Download and preprocess the ANI-2x data set.

        Parameters
        ----------
        split : str or None
            ``None`` returns ``{"all": ...}``; ``"train"`` / ``"val"`` /
            ``"test"`` returns a per-conformation 80/10/10 partition (fixed
            ``seed``, disjoint splits).
        cache_dir : pathlib.Path
            Base cache directory; files live under ``cache_dir/"ani2x"``.
        n_atoms : int or sequence of int, optional
            Which atom-count group(s) to load (e.g. ``5`` or ``[4, 5, 6]``).
            Defaults to every group. Fewer/smaller groups mean far less data.
        forces : bool, optional
            Include forces (default ``True``).
        units : str, optional
            ``"eV"`` (default) converts energies to eV and forces to eV/A;
            ``"hartree"`` keeps the raw upstream values.
        max_groups : int, optional
            Cap the number of atom-count groups read (useful for demos).
        max_conformations : int, optional
            Cap the number of conformations kept per group.
        seed : int, optional
            Seed for the reproducible 80/10/10 split. Defaults to ``1234``.
        quiet : bool, optional
            Suppress progress output. Defaults to ``False``.

        Returns
        -------
        dict of {str: list of dict} or list of dict
            Structure dicts with keys ``pos`` ``(N, 3)``, ``atomic_numbers``
            ``(N,)``, ``energy`` (scalar) and, when requested, ``forces``
            ``(N, 3)``. ANI-2x is molecular, so no ``cell`` / ``pbc``.

        Raises
        ------
        ValueError
            If ``units`` or ``split`` is unrecognized.
        ImportError
            If ``h5py`` is not installed.
        """
        if units not in ("eV", "hartree", "Hartree"):
            raise ValueError(f"units must be 'eV' or 'hartree', got {units!r}")
        keep = self._resolve_n_atoms(n_atoms)
        scale = 1.0 if units.lower() == "hartree" else _HARTREE_TO_EV

        root = Path(cache_dir) / self.name
        h5_path = self._ensure_extracted(root, quiet)

        structures = self._read_h5(h5_path, keep, forces, scale, max_groups,
                                   max_conformations, quiet)

        if split is None:
            return {"all": structures}
        if split not in ("train", "val", "test"):
            raise ValueError(
                f"unknown split {split!r}; use 'train', 'val', 'test', or None")
        return self._partition(structures, seed)[split]

    @staticmethod
    def _resolve_n_atoms(n_atoms) -> Optional[set]:
        """Normalize the ``n_atoms`` selection to a set (or ``None`` for all)."""
        if n_atoms is None:
            return None
        vals = [n_atoms] if isinstance(n_atoms, int) else list(n_atoms)
        return {int(v) for v in vals}

    def _ensure_extracted(self, root: Path, quiet: bool) -> Path:
        """Download (once) and extract the archive; return the .h5 file path."""
        raw = root / "raw"
        existing = sorted(raw.glob("**/*.h5"))
        if existing:
            return existing[0]
        archive = download_file(_URL, raw / _ARCHIVE, _MD5, quiet=quiet)
        extract_archive(archive, raw)
        found = sorted(raw.glob("**/*.h5"))
        if not found:
            raise FileNotFoundError(
                f"no .h5 file found after extracting {archive}")
        return found[0]

    @staticmethod
    def _read_h5(h5_path: Path, keep: Optional[set], want_forces: bool,
                 scale: float, max_groups: Optional[int],
                 max_conformations: Optional[int],
                 quiet: bool) -> list[dict]:
        """Parse the ANI-2x HDF5 (groups keyed by atom count) into dicts.

        ``species`` is stored as atomic numbers per conformation ``(Nc, Na)``;
        a shared ``(Na,)`` layout is also handled defensively. Each atom-count
        group mixes many different molecules ordered by molecule, so when
        ``max_conformations`` caps a group the kept conformations are taken
        **evenly spaced** across it (a deterministic stride), not as a
        contiguous head; otherwise a cap would silently keep only the first
        molecule(s) and drop whole elements (e.g. all S/F/Cl).
        """
        import h5py

        out: list[dict] = []
        with h5py.File(h5_path, "r") as f:
            names = sorted(f.keys())
            if keep is not None:
                names = [n for n in names if _atom_count(n) in keep]
            if max_groups is not None:
                names = names[:max_groups]
            for name in tqdm(names, desc="ani2x", unit=" group",
                             disable=quiet, leave=False):
                g = f[name]
                coords = np.asarray(g["coordinates"][()], dtype=np.float64)
                species = np.asarray(g["species"][()])
                energies = np.asarray(g["energies"][()], dtype=np.float64)
                forces = (np.asarray(g["forces"][()], dtype=np.float64)
                          if want_forces and "forces" in g else None)

                total = len(energies)
                if max_conformations is not None and max_conformations < total:
                    sel = np.linspace(0, total - 1, max_conformations,
                                      dtype=np.int64)
                else:
                    sel = np.arange(total)
                shared_z = species.ndim == 1
                for i in sel:
                    z = species if shared_z else species[i]
                    rec = {
                        "pos": coords[i],
                        "atomic_numbers": np.asarray(z, dtype=np.int64),
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


def _atom_count(name: str) -> int:
    """Parse an atom-count group name (e.g. ``"002"``) to an int."""
    try:
        return int(name)
    except ValueError:
        return -1


register_dataset(ANI2xBuilder())
