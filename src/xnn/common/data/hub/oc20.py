"""Builder for the OC20 S2EF data (structure to energy and forces), the 200K training split.

The Open Catalyst 2020 dataset (Chanussot et al., *ACS Catal.* 11, 6059, 2021;
CC BY 4.0) holds adsorbates on catalyst slabs (metals, intermetallics and
their oxides and nitrides) from RPBE relaxations. The S2EF ``200K`` split is the
smallest official training set: 200,000 frames, each from a different
adsorbate-slab system, about 70 atoms each, periodic in the slab plane with
vacuum along the third lattice vector.

Notes
-----
* ``energy="adsorption"`` (default) stores the S2EF target, the DFT energy minus
  the reference energy of the bare slab and the gas-phase adsorbate references;
  ``"total"`` stores the DFT energy itself.
* Atoms the relaxation kept fixed (the slab's bottom layers) get an
  ``atom_force_weight`` of zero, so a force loss counts the free atoms only, as
  in the OC20 training protocol. ``info`` keeps the ``tags`` (0 subsurface,
  1 surface, 2 adsorbate), the system id and the reference energy.
* There is no official validation subset of the 200K split; ``val`` is a random
  subset of it (the official validation splits are separate archives).
"""
from __future__ import annotations

import lzma
import re
import tarfile
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ..elements import SYMBOL_TO_Z
from ._download import download_file
from .base import DatasetBuilder, register_dataset

_URL = "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_200K.tar"
_MD5 = None
_PROPERTIES = "species:S:1:pos:R:3:move_mask:L:1:tags:I:1:forces:R:3"
_LATTICE = re.compile(r'Lattice="([^"]+)"')
_ENERGY = re.compile(r'(?<![\w])energy=(\S+)')
_PROPS = re.compile(r'Properties=(\S+)')


def parse_extxyz(text: str):
    """Parse the frames of one OC20 S2EF ``.extxyz`` file.

    Parameters
    ----------
    text : str
        The decompressed file.

    Returns
    -------
    list of tuple
        ``(cell (3, 3), energy, Z (N,), pos (N, 3), fixed (N,), tags (N,), forces (N, 3))``
        per frame.

    Raises
    ------
    ValueError
        If a frame does not have the OC20 S2EF columns.
    """
    lines = text.splitlines()
    frames, i = [], 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n = int(lines[i])
        header = lines[i + 1]
        props = _PROPS.search(header)
        if props is None or props.group(1) != _PROPERTIES:
            raise ValueError(f"unexpected extxyz columns: {props.group(1) if props else header!r}")
        cell = np.array([float(v) for v in _LATTICE.search(header).group(1).split()]).reshape(3, 3)
        energy = float(_ENERGY.search(header).group(1))
        rows = [line.split() for line in lines[i + 2:i + 2 + n]]
        z = np.array([SYMBOL_TO_Z[r[0]] for r in rows], dtype=np.int64)
        pos = np.array([[float(v) for v in r[1:4]] for r in rows])
        fixed = np.array([r[4] == "F" for r in rows])
        tags = np.array([int(r[5]) for r in rows], dtype=np.int64)
        forces = np.array([[float(v) for v in r[6:9]] for r in rows])
        frames.append((cell, energy, z, pos, fixed, tags, forces))
        i += 2 + n
    return frames


def parse_txt(text: str) -> list[tuple[str, int, float]]:
    """Parse the ``system_id,frame_number,reference_energy`` lines of a ``.txt`` file."""
    out = []
    for line in text.splitlines():
        if line.strip():
            sid, frame, ref = line.split(",")
            out.append((sid, int(frame.replace("frame", "")), float(ref)))
    return out


def _parse_archive(path: Path) -> dict[str, np.ndarray]:
    """Parse every ``<k>.extxyz.xz`` / ``<k>.txt.xz`` pair of the archive into flat arrays."""
    members: dict[str, dict[str, bytes]] = {}
    with tarfile.open(path, "r:*") as archive:
        for member in archive:
            name = Path(member.name).name
            if member.isfile() and (name.endswith(".extxyz.xz") or name.endswith(".txt.xz")):
                key, kind = name.split(".")[0], name.split(".")[1]
                members.setdefault(key, {})[kind] = archive.extractfile(member).read()
    cells, energies, refs, sids, counts = [], [], [], [], []
    zs, positions, fixed, tags, forces = [], [], [], [], []
    for key in sorted(members, key=lambda k: int(k) if k.isdigit() else k):
        pair = members[key]
        frames = parse_extxyz(lzma.decompress(pair["extxyz"]).decode())
        info = parse_txt(lzma.decompress(pair["txt"]).decode())
        if len(frames) != len(info):
            raise ValueError(f"{key}: {len(frames)} frames but {len(info)} reference lines")
        for (cell, e, z, pos, fx, tg, f), (sid, _, ref) in zip(frames, info):
            cells.append(cell); energies.append(e); refs.append(ref); sids.append(sid)
            counts.append(len(z)); zs.append(z); positions.append(pos)
            fixed.append(fx); tags.append(tg); forces.append(f)
    return {"cell": np.stack(cells), "energy": np.array(energies), "reference_energy": np.array(refs),
            "sid": np.array(sids), "offsets": np.concatenate([[0], np.cumsum(counts)]),
            "atomic_numbers": np.concatenate(zs), "pos": np.concatenate(positions),
            "fixed": np.concatenate(fixed), "tags": np.concatenate(tags),
            "forces": np.concatenate(forces)}


class OC20Builder(DatasetBuilder):
    """Builder for the OC20 S2EF 200K split (adsorbates on catalyst slabs, energies and forces)."""

    name = "oc20"
    description = ("OC20 S2EF 200K: 200,000 adsorbate+slab frames (RPBE) with adsorption "
                   "energies and forces; periodic slabs.")

    def load(self, *, split: Optional[str] = None, cache_dir: Path, energy: str = "adsorption",
             n_val: int = 10000, n_train: Optional[int] = None, seed: int = 0,
             return_info: bool = False, quiet: bool = False) -> Union[dict[str, list[dict]], list[dict]]:
        """Download, parse and split the S2EF 200K data.

        Parameters
        ----------
        split : str or None
            ``None`` returns ``{"train": ..., "val": ...}``; ``"train"`` / ``"val"``
            returns that split; ``"all"`` returns every frame in file order.
        cache_dir : pathlib.Path
            Base cache directory; the files are stored under ``cache_dir/"oc20"``.
        energy : str, optional
            ``"adsorption"`` (default, the S2EF target) or ``"total"``.
        n_val : int, optional
            Size of the random validation subset, by default 10,000.
        n_train : int, optional
            Truncate the training split to its first ``n_train`` frames.
        seed : int, optional
            Seed of the random split, by default 0.
        return_info : bool, optional
            Add an ``info`` dict (``sid``, ``tags``, ``reference_energy``,
            ``total_energy``) to every structure.
        quiet : bool, optional
            Suppress download progress output.

        Returns
        -------
        dict of {str: list of dict} or list of dict
            Structure dicts with ``pos``, ``atomic_numbers``, ``cell``, ``pbc``,
            ``energy`` (eV), ``forces`` (eV/Angstrom) and ``atom_force_weight``.

        Raises
        ------
        ValueError
            For an unknown ``energy`` or ``split``.
        """
        if energy not in ("adsorption", "total"):
            raise ValueError(f"energy must be 'adsorption' or 'total', got {energy!r}")
        root = cache_dir / self.name
        cache = root / "s2ef_train_200K.npz"
        if not cache.exists():
            archive = download_file(_URL, root / "raw" / "s2ef_train_200K.tar", _MD5, quiet=quiet)
            # written under a temporary name and renamed, so a concurrent reader never sees half a file
            partial = cache.with_suffix(".partial.npz")
            np.savez(partial, **_parse_archive(archive))
            partial.replace(cache)
        with np.load(cache) as npz:
            data = {key: npz[key] for key in npz.files}
        offsets = data["offsets"]
        targets = data["energy"] - (data["reference_energy"] if energy == "adsorption" else 0.0)

        def build(rows):
            out = []
            for i in rows:
                a, b = offsets[i], offsets[i + 1]
                s = {"pos": data["pos"][a:b].copy(), "atomic_numbers": data["atomic_numbers"][a:b].copy(),
                     "cell": data["cell"][i].copy(), "pbc": [True, True, True],
                     "energy": float(targets[i]), "forces": data["forces"][a:b].copy(),
                     "atom_force_weight": (~data["fixed"][a:b]).astype(np.float64)}
                if return_info:
                    s["info"] = {"sid": str(data["sid"][i]), "tags": data["tags"][a:b].copy(),
                                 "reference_energy": float(data["reference_energy"][i]),
                                 "total_energy": float(data["energy"][i])}
                out.append(s)
            return out

        n = len(targets)
        if split == "all":
            return build(range(n))
        perm = np.random.default_rng(seed).permutation(n)
        splits = {"train": build(np.sort(perm[n_val:])[:n_train]), "val": build(np.sort(perm[:n_val]))}
        if split is None:
            return splits
        if split not in splits:
            raise ValueError(f"unknown split {split!r}; use 'train', 'val', 'all', or None")
        return splits[split]


register_dataset(OC20Builder())
