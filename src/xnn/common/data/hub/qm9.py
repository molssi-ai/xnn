"""QM9 dataset builder.

QM9 (Ramakrishnan *et al.*, *Sci. Data* **1**, 140022 (2014)) holds 133,885
small organic molecules of up to nine heavy atoms (C, N, O, F) with
B3LYP/6-31G(2df,p) geometries and 15 properties each, the benchmark of the
TorchMD-Net models (ET: Table 1 of Thoelke & De Fabritiis 2022; TensorNet:
Table 1 of Pelaez *et al.* 2024).

Data: https://doi.org/10.6084/m9.figshare.978904 (``dsgdb9nsd.xyz.tar.bz2``
and ``uncharacterized.txt``).

Notes
-----
* The 3054 molecules that failed the authors' consistency check are excluded
  by default (``exclude_uncharacterized``), leaving the 130,831 molecules of
  the usual benchmark splits (110,000 / 10,000 / 10,831).
* Energies (``U0``, ``U``, ``H``, ``G``, ``zpve``, ``homo``, ``lumo``,
  ``gap``) are Hartree upstream and converted to eV by default
  (``units="eV"``); the other properties keep their units. The per-element
  reference energies of the energies (``atomref.txt``) are available as
  :func:`atomic_energies`.
* The archive of 133,885 xyz files is parsed once and cached as an ``.npz``.
"""
from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ._download import download_file
from .base import DatasetBuilder, register_dataset
from .units import HARTREE_TO_EV

_XYZ_URL = "https://ndownloader.figshare.com/files/3195389"
_XYZ_MD5 = "ad1ebd51ee7f5b3a6e32e974e5d54012"
_UNCHARACTERIZED_URL = "https://ndownloader.figshare.com/files/3195404"
_UNCHARACTERIZED_MD5 = "a361887bacb427b8a0ce7903d92a53b4"

#: The 15 properties of every molecule, in file order.
PROPERTIES = ("A", "B", "C", "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve",
              "U0", "U", "H", "G", "Cv")
#: The properties given in Hartree upstream.
ENERGY_PROPERTIES = ("homo", "lumo", "gap", "zpve", "U0", "U", "H", "G")
_SYMBOLS = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}

#: Per-element reference energies (Hartree) of the authors' ``atomref.txt``.
_ATOMIC_REFERENCES = {
    "U0": {1: -0.500273, 6: -37.846772, 7: -54.583861, 8: -75.064579, 9: -99.718730},
    "U": {1: -0.498857, 6: -37.845355, 7: -54.582445, 8: -75.063163, 9: -99.717314},
    "H": {1: -0.497912, 6: -37.844411, 7: -54.581501, 8: -75.062219, 9: -99.716370},
    "G": {1: -0.510927, 6: -37.861317, 7: -54.598897, 8: -75.079532, 9: -99.733544},
}


def atomic_energies(label: str = "U0", units: str = "eV") -> dict[int, float]:
    """The per-element reference energies of an energy label.

    Parameters
    ----------
    label : str, optional
        ``"U0"`` (default), ``"U"``, ``"H"`` or ``"G"``.
    units : str, optional
        ``"eV"`` (default) or ``"hartree"``.

    Returns
    -------
    dict of int to float
        ``{Z: E0}`` for H, C, N, O and F.
    """
    if label not in _ATOMIC_REFERENCES:
        raise ValueError(f"no atomic references for {label!r}; "
                         f"choose from {sorted(_ATOMIC_REFERENCES)}")
    scale = HARTREE_TO_EV if units == "eV" else 1.0
    return {z: e * scale for z, e in _ATOMIC_REFERENCES[label].items()}


def parse_xyz(text: str) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Parse one QM9 xyz file into ``(gdb index, Z, positions, properties)``."""
    lines = text.splitlines()
    n_atoms = int(lines[0].strip())
    header = lines[1].split()
    index = int(header[1])
    props = np.array([float(v) for v in header[2:17]], dtype=np.float64)
    z = np.empty(n_atoms, dtype=np.int64)
    pos = np.empty((n_atoms, 3), dtype=np.float64)
    for i, line in enumerate(lines[2:2 + n_atoms]):
        fields = line.replace("*^", "e").split()
        z[i] = _SYMBOLS[fields[0]]
        pos[i] = [float(v) for v in fields[1:4]]
    return index, z, pos, props


def _parse_archive(path: Path) -> dict[str, np.ndarray]:
    """Parse the whole xyz archive into flat arrays (atoms concatenated, with offsets)."""
    indices, zs, positions, props, counts = [], [], [], [], []
    with tarfile.open(path, "r:*") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".xyz"):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            index, z, pos, prop = parse_xyz(io.TextIOWrapper(handle, encoding="utf-8").read())
            indices.append(index)
            zs.append(z)
            positions.append(pos)
            props.append(prop)
            counts.append(len(z))
    order = np.argsort(np.array(indices))
    counts = np.array(counts, dtype=np.int64)[order]
    offsets = np.concatenate([[0], np.cumsum(counts)])
    return {"index": np.array(indices, dtype=np.int64)[order],
            "atomic_numbers": np.concatenate([zs[i] for i in order]),
            "pos": np.concatenate([positions[i] for i in order]),
            "properties": np.stack([props[i] for i in order]),
            "offsets": offsets}


def _uncharacterized(path: Path) -> np.ndarray:
    """The gdb indices listed in ``uncharacterized.txt``."""
    ids = []
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if fields and fields[0].isdigit():
            ids.append(int(fields[0]))
    return np.array(sorted(set(ids)), dtype=np.int64)


class QM9Builder(DatasetBuilder):
    """Builder for QM9 (133,885 molecules, 15 properties, random benchmark splits)."""

    name = "qm9"
    description = ("QM9: 133,885 GDB-9 molecules (H, C, N, O, F) with B3LYP energies and "
                   "14 further properties.")

    def load(self, *, split: Optional[str] = None, cache_dir: Path, label: str = "U0",
             units: str = "eV", n_train: int = 110000, n_val: int = 10000,
             n_test: Optional[int] = None, seed: int = 0,
             exclude_uncharacterized: bool = True, return_info: bool = False,
             quiet: bool = False) -> Union[dict[str, list[dict]], list[dict]]:
        """Download, parse and split QM9.

        Parameters
        ----------
        split : str or None
            ``None`` returns ``{"train": ..., "val": ..., "test": ...}``;
            ``"train"`` / ``"val"`` / ``"test"`` returns that split; ``"all"``
            returns every molecule in gdb order.
        cache_dir : pathlib.Path
            Base cache directory; the files are stored under ``cache_dir/"qm9"``.
        label : str, optional
            The property stored as the structure ``energy``, by default
            ``"U0"`` (see :data:`PROPERTIES`).
        units : str, optional
            ``"eV"`` (default; Hartree properties converted) or
            ``"hartree"`` (as published).
        n_train, n_val : int, optional
            Sizes of the random training and validation splits, by default
            110,000 and 10,000; the rest is the test split.
        n_test : int, optional
            Truncate the test split to its first ``n_test`` molecules.
        seed : int, optional
            Seed of the random split, by default 0.
        exclude_uncharacterized : bool, optional
            Drop the 3054 molecules of ``uncharacterized.txt`` (default).
        return_info : bool, optional
            Add an ``info`` dict with the gdb ``index`` and all 15 properties
            (in the chosen units) to every structure.
        quiet : bool, optional
            Suppress download progress output.

        Returns
        -------
        dict of {str: list of dict} or list of dict
            Structure dicts with ``pos`` ``(N, 3)`` (Angstrom),
            ``atomic_numbers`` ``(N,)`` and ``energy`` (the ``label``).

        Raises
        ------
        ValueError
            For an unknown ``units``, ``label`` or ``split``.
        """
        if units not in ("eV", "hartree"):
            raise ValueError(f"units must be 'eV' or 'hartree', got {units!r}")
        if label not in PROPERTIES:
            raise ValueError(f"unknown label {label!r}; choose from {PROPERTIES}")
        root = cache_dir / self.name
        cache = root / "qm9.npz"
        if not cache.exists():
            archive = download_file(_XYZ_URL, root / "raw" / "dsgdb9nsd.xyz.tar.bz2",
                                    _XYZ_MD5, quiet=quiet)
            np.savez(cache, **_parse_archive(archive))
        data = np.load(cache)
        index, offsets = data["index"], data["offsets"]
        positions, numbers = data["pos"], data["atomic_numbers"]
        keep = np.ones(len(index), dtype=bool)
        if exclude_uncharacterized:
            bad = _uncharacterized(download_file(
                _UNCHARACTERIZED_URL, root / "raw" / "uncharacterized.txt",
                _UNCHARACTERIZED_MD5, quiet=quiet))
            keep &= ~np.isin(index, bad)
        scale = np.ones(len(PROPERTIES))
        if units == "eV":
            for name in ENERGY_PROPERTIES:
                scale[PROPERTIES.index(name)] = HARTREE_TO_EV
        props = data["properties"] * scale
        column = PROPERTIES.index(label)

        def build(rows):
            out = []
            for i in rows:
                a, b = offsets[i], offsets[i + 1]
                s = {"pos": positions[a:b].copy(),
                     "atomic_numbers": numbers[a:b].copy(),
                     "energy": float(props[i, column])}
                if return_info:
                    s["info"] = {"index": int(index[i]),
                                 **{k: float(v) for k, v in zip(PROPERTIES, props[i])}}
                out.append(s)
            return out

        rows = np.flatnonzero(keep)
        if split == "all":
            return build(rows)
        perm = rows[np.random.default_rng(seed).permutation(len(rows))]
        splits = {"train": build(np.sort(perm[:n_train])),
                  "val": build(np.sort(perm[n_train:n_train + n_val])),
                  "test": build(np.sort(perm[n_train + n_val:][:n_test]))}
        if split is None:
            return splits
        if split not in splits:
            raise ValueError(f"unknown split {split!r}; use 'train', 'val', 'test', 'all', or None")
        return splits[split]


register_dataset(QM9Builder())
