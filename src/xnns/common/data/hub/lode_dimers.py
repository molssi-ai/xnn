"""LODE non-bonded interactions dataset builder (molecular dimers & toy systems).

The dataset accompanies the extension of the long-distance equivariant (LODE)
framework to diverse long-range interactions, and bundles several sub-datasets:
biomolecular sidechain dimers with DFT energies & forces, their isolated
monomers, two point-charge toy systems (pure Coulomb and pure ``1/r**6``
dispersion), and Xenon dimers/trimers.

All files are ASE-readable extxyz in eV / eV.A units (positions in angstrom),
carrying rich per-frame ``info`` -- most usefully ``label`` on the biomolecular
dimers, tagging each pair by the polarity of its two fragments: ``CC``
(charged-charged), ``CP`` (charged-polar), ``PP`` (polar-polar), plus ``AA`` /
``CA`` / ``PA`` involving apolar fragments.

Reference
---------
K. K. Huguenin-Dumittan, P. Loche, H. Ni, M. Ceriotti, "Physics-inspired
equivariant descriptors of non-bonded interactions", *J. Phys. Chem. Lett.*
(2023). Data: https://archive.materialscloud.org/records/405an-d8183
(DOI 10.24435/materialscloud:23-99).

Notes
-----
* Frames are periodic only nominally -- each system sits in a large (>=30 A)
  cubic box so it is effectively isolated -- so ``cell`` / ``pbc`` are populated
  from the file.
* Reading extxyz requires the ``ase`` extra (``pip install xnns[ase]``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from ._download import download_file
from .base import DatasetBuilder, register_dataset

_BASE = "https://archive.materialscloud.org/api/records/405an-d8183/files/{fname}/content"

# subset -> (remote filename, md5, has_forces). The biomolecular dimers and
# xenon carry forces; the point-charge toy sets are energy-only.
_SUBSETS: dict[str, tuple[str, str, bool]] = {
    "bio":                       ("bio_dimers.xyz", "cdf5f3e139bd56a2c31014c8e50e8b93", True),
    "monomers":                  ("bio_dimers_monomers.xyz", "3d4b87ec5e1a79b75c41c48a78f78552", True),
    "point_charges_coulomb":     ("point_charges_Training_set_p1.xyz", "9e76b4a8972a2e829335163fe31c7d7c", False),
    "point_charges_dispersion":  ("point_charges_Training_set_p6.xyz", "57c54e0ea0b01be89e0a0622f0f2ff96", False),
    "xenon":                     ("xenon.xyz", "9371ed17727e39635fa0c11e27425ace", True),
}

# common spellings mapped to the canonical subset key.
_ALIASES: dict[str, str] = {
    "bio_dimers": "bio",
    "dimers": "bio",
    "coulomb": "point_charges_coulomb",
    "p1": "point_charges_coulomb",
    "dispersion": "point_charges_dispersion",
    "p6": "point_charges_dispersion",
    "xe": "xenon",
}

# valid `label` values on the biomolecular dimers (fragment polarity pair).
_LABELS = frozenset({"AA", "CA", "CC", "CP", "PA", "PP"})


class LODEDimersBuilder(DatasetBuilder):
    """Builder for the LODE non-bonded interactions dataset.

    See the module docstring for the dataset description and citation. Only the
    requested ``subset``'s file is downloaded.
    """

    name = "lode_dimers"
    description = ("LODE non-bonded dataset: biomolecular dimers (CC/CP/PP...), "
                   "monomers, point-charge toys, and Xe clusters.")

    def load(self, *, split: Optional[str] = None, cache_dir: Path,
             subset: str = "bio", label: Optional[str] = None,
             return_info: bool = False,
             quiet: bool = False) -> Union[dict[str, list[dict]], list[dict]]:
        """Download and preprocess one sub-dataset.

        Parameters
        ----------
        split : str or None
            The dataset ships no official train/test split. ``None`` returns
            ``{"all": structures}``; ``"all"`` returns the list directly. (Split
            it yourself -- e.g. by ``distance`` for the paper's near/far
            extrapolation task.)
        cache_dir : pathlib.Path
            Base cache directory; files are stored under
            ``cache_dir/"lode_dimers"``.
        subset : str, optional
            Which sub-dataset to load. One of ``bio`` (default; biomolecular
            dimers with energies & forces), ``monomers``,
            ``point_charges_coulomb``, ``point_charges_dispersion``, ``xenon``.
        label : str, optional
            For ``subset="bio"`` only: keep only dimers of this fragment-polarity
            class -- one of ``AA``, ``CA``, ``CC``, ``CP``, ``PA``, ``PP``.
            ``None`` (default) keeps all classes.
        return_info : bool, optional
            If ``True``, attach the frame's extxyz ``info`` (the per-frame
            metadata) to each structure dict under the ``"info"`` key. For the
            biomolecular dimers this carries ``label``, ``dimer_id``,
            ``distance``, the per-monomer ``energyA`` / ``energyB`` and
            ``chargeA`` / ``chargeB``, etc. -- enough to compute binding energies
            ``E_dimer - energyA - energyB`` directly. Downstream graph building
            ignores the extra key, so it is safe to combine with ``cutoff``.
            Defaults to ``False``.
        quiet : bool, optional
            Suppress download progress output. Defaults to ``False``.

        Returns
        -------
        dict of {str: list of dict} or list of dict
            Structure dicts with keys ``pos`` ``(N, 3)``, ``atomic_numbers``
            ``(N,)``, ``cell`` ``(3, 3)``, ``pbc`` ``(3,)``, ``energy``, --
            where the subset provides them -- ``forces`` ``(N, 3)``, and (with
            ``return_info``) ``info``.

        Raises
        ------
        ValueError
            If ``subset`` / ``label`` is unknown, ``label`` is used with a
            non-``bio`` subset, or ``split`` is unrecognized.
        """
        sub = self._resolve_subset(subset)
        if split not in (None, "all"):
            raise ValueError(
                f"unknown split {split!r}; lode_dimers has no train/test split, "
                "use split=None or split='all'")

        fname, md5, _ = _SUBSETS[sub]
        root = Path(cache_dir) / self.name
        path = download_file(_BASE.format(fname=fname), root / "raw" / fname,
                             md5, quiet=quiet)

        structures = self._read_xyz(path, sub, label, return_info, quiet)
        if split == "all":
            return structures
        return {"all": structures}

    @staticmethod
    def _resolve_subset(subset: str) -> str:
        """Normalize a subset name to a canonical key, validating it.

        Parameters
        ----------
        subset : str
            User-supplied subset name (case-insensitive; a few aliases accepted).

        Returns
        -------
        str
            Canonical subset key present in :data:`_SUBSETS`.

        Raises
        ------
        ValueError
            If ``subset`` is not a known sub-dataset.
        """
        key = subset.strip().lower()
        key = _ALIASES.get(key, key)
        if key not in _SUBSETS:
            raise ValueError(
                f"unknown lode_dimers subset {subset!r}; choose one of: "
                + ", ".join(sorted(_SUBSETS)))
        return key

    @staticmethod
    def _read_xyz(path: Path, subset: str, label: Optional[str],
                  return_info: bool, quiet: bool) -> list[dict]:
        """Read an extxyz file into structure dicts, optionally filtered by label.

        Parameters
        ----------
        path : pathlib.Path
            Local extxyz file.
        subset : str
            Canonical subset key (only ``bio`` supports ``label`` filtering).
        label : str or None
            Fragment-polarity class to keep, or ``None`` for all.
        return_info : bool
            Attach each frame's ``info`` dict under the structure dict's
            ``"info"`` key.
        quiet : bool
            Suppress the conversion progress bar.

        Returns
        -------
        list of dict
            Structure dicts (see :func:`~xnns.common.data.ase_io.atoms_to_structure`).

        Raises
        ------
        ValueError
            If ``label`` is invalid or requested for a non-``bio`` subset.
        """
        from tqdm.auto import tqdm

        from ase.io import read

        from ..ase_io import atoms_to_structure

        if label is not None:
            if subset != "bio":
                raise ValueError(
                    "`label` filtering is only supported for subset='bio'")
            label = label.strip().upper()
            if label not in _LABELS:
                raise ValueError(
                    f"unknown label {label!r}; choose one of: "
                    + ", ".join(sorted(_LABELS)))

        frames = read(str(path), index=":")
        if label is not None:
            frames = [a for a in frames if a.info.get("label") == label]
        desc = f"lode_dimers:{subset}" + (f" [{label}]" if label else "")

        structures = []
        for a in tqdm(frames, desc=desc, unit=" struct",
                      disable=quiet, leave=False):
            d = atoms_to_structure(a)
            if return_info:
                d["info"] = dict(a.info)
            structures.append(d)
        return structures


register_dataset(LODEDimersBuilder())
