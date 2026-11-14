"""Materials Project datasets of the MEGNet papers.

``mp_2018``: the 69,239 Materials Project crystals of 2018.6.1 with their
formation energies per atom, band gaps and, for 5830 of them, bulk and shear
moduli (Chen *et al.*, *Chem. Mater.* **31**, 3564 (2019);
https://doi.org/10.6084/m9.figshare.7451351, CC BY 4.0).

``multifidelity_band_gap``: band gaps of Materials Project crystals at four
levels of theory, PBE (52,348), GLLB-SC (2290), HSE (6030) and SCAN (472)
(Chen *et al.*, *Nat. Comput. Sci.* **1**, 46 (2021);
https://doi.org/10.6084/m9.figshare.13040330, CC BY 4.0), with the
structures of the 2019.04.01 database
(https://doi.org/10.6084/m9.figshare.8097992, MIT). The experimental band gaps
of the paper are not included: their structures come from the ICSD, which is
not freely available.

Both files are read once, entry by entry, and cached as ``.npz``.
"""
from __future__ import annotations

import gzip
import io
import json
import math
import zipfile
from pathlib import Path
from typing import Iterator, Optional, Sequence, Union

import numpy as np

from ..elements import SYMBOL_TO_Z
from ._download import download_file
from .base import DatasetBuilder, register_dataset

_MP2018 = ("https://ndownloader.figshare.com/files/15087992", "4599ea88419c6a90bfb0dad2c2bf3733",
           "mp.2018.6.1.json.zip")
_GAPS = ("https://ndownloader.figshare.com/files/24940055", "11c0b00503056e638ab25089a73d818a",
         "band_gap_no_structs.gz")
_MP2019 = ("https://ndownloader.figshare.com/files/15108200", "bdb151464a4bff4ca9080cf2883394d0",
           "mp.2019.04.01.json.zip")


def iter_json_list(stream, chunk: int = 1 << 24) -> Iterator:
    """The items of a JSON list read from a text ``stream`` one at a time, so a file larger than memory can be
    read."""
    decoder = json.JSONDecoder()
    buf, pos, done = "", 0, False

    def fill():
        nonlocal buf, pos, done
        more = stream.read(chunk)
        done = not more
        buf, pos = buf[pos:] + more, 0

    fill()
    while True:
        while pos < len(buf) and buf[pos] in " \t\r\n,[":
            pos += 1
        if pos < len(buf) and buf[pos] == "]":
            return
        if pos >= len(buf):
            if done:
                return
            fill()
            continue
        try:
            item, end = decoder.raw_decode(buf, pos)
        except json.JSONDecodeError:
            if done:
                raise
            fill()
            continue
        pos = end
        yield item


def parse_p1_cif(text: str) -> tuple:
    """Cell ``(3, 3)`` (lattice vectors as rows, Angstrom), atomic numbers ``(N,)`` and Cartesian positions
    ``(N, 3)`` of a CIF in space group P1 with fully occupied sites (the files pymatgen writes).

    Raises
    ------
    ValueError
        For other symmetry operations or partial occupancies.
    """
    params, header, rows, ops = {}, [], [], []
    lines = [line.strip() for line in text.splitlines()]
    i = 0
    while i < len(lines):
        s = lines[i]
        i += 1
        if s.startswith("_cell_length_") or s.startswith("_cell_angle_"):
            key, value = s.split()[:2]
            params[key] = float(value)
        elif s == "loop_":
            # the loop's field names, then its rows up to the next blank line, field or loop
            fields, body = [], []
            while i < len(lines) and lines[i].startswith("_"):
                fields.append(lines[i])
                i += 1
            while i < len(lines) and lines[i] and not lines[i].startswith("_") and lines[i] != "loop_":
                body.append(lines[i])
                i += 1
            if "_symmetry_equiv_pos_as_xyz" in fields:
                ops = body
            elif "_atom_site_fract_x" in fields:
                header, rows = fields, body
    if len(ops) > 1 or (ops and ops[0].split(None, 1)[-1].strip("'\" ").replace(" ", "") != "x,y,z"):
        raise ValueError("only P1 CIFs (one symmetry operation) are read")
    if not rows:
        raise ValueError("the CIF has no atom sites")
    col = {f: i for i, f in enumerate(header)}
    symbols, frac = [], []
    for row in rows:
        v = row.split()
        if "_atom_site_occupancy" in col and abs(float(v[col["_atom_site_occupancy"]]) - 1.0) > 1e-6:
            raise ValueError("partially occupied sites are not read")
        symbols.append("".join(ch for ch in v[col["_atom_site_type_symbol"]] if ch.isalpha()))
        frac.append([float(v[col[f"_atom_site_fract_{k}"]]) for k in "xyz"])
    a, b, c = (params[f"_cell_length_{k}"] for k in "abc")
    al, be, ga = (math.radians(params[f"_cell_angle_{k}"]) for k in ("alpha", "beta", "gamma"))
    cx = math.cos(be)
    cy = (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
    cell = np.array([[a, 0.0, 0.0], [b * math.cos(ga), b * math.sin(ga), 0.0],
                     [c * cx, c * cy, c * math.sqrt(max(1.0 - cx * cx - cy * cy, 0.0))]])
    numbers = np.array([SYMBOL_TO_Z[s] for s in symbols], dtype=np.int64)
    return cell, numbers, np.array(frac, dtype=np.float64).reshape(-1, 3) @ cell


def _pack(structures: list) -> dict:
    """Concatenated arrays of ``(cell, numbers, positions)`` triples."""
    sizes = [len(s[1]) for s in structures]
    return {"cell": np.stack([s[0] for s in structures]) if structures else np.zeros((0, 3, 3)),
            "atomic_numbers": np.concatenate([s[1] for s in structures]) if structures else np.zeros(0, int),
            "pos": np.concatenate([s[2] for s in structures]) if structures else np.zeros((0, 3)),
            "offsets": np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)}


def _structure(data, i: int, value: float, state=None, info=None) -> dict:
    a, b = data["offsets"][i], data["offsets"][i + 1]
    s = {"pos": data["pos"][a:b].copy(), "atomic_numbers": data["atomic_numbers"][a:b].copy(),
         "cell": data["cell"][i].copy(), "pbc": [True, True, True], "properties": float(value)}
    if state is not None:
        s["state"] = int(state)
    if info is not None:
        s["info"] = info
    return s


def _split(rows: np.ndarray, n_train: int, n_val: int, seed: int) -> dict:
    perm = rows[np.random.default_rng(seed).permutation(len(rows))]
    return {"train": np.sort(perm[:n_train]), "val": np.sort(perm[n_train:n_train + n_val]),
            "test": np.sort(perm[n_train + n_val:])}


def _pick(splits: dict, split: Optional[str]):
    if split is None:
        return splits
    if split not in splits:
        raise ValueError(f"unknown split {split!r}; use 'train', 'val', 'test', 'all', or None")
    return splits[split]


class MP2018Builder(DatasetBuilder):
    """Builder for the Materials Project crystals of 2018.6.1 (the MEGNet crystal benchmarks)."""

    name = "mp_2018"
    description = ("Materials Project 2018.6.1: 69,239 crystals with formation energies, band gaps and "
                   "5830 bulk and shear moduli (MEGNet).")
    #: The labels a structure's ``properties`` can hold.
    LABELS = ("formation_energy", "band_gap", "bulk_modulus", "shear_modulus", "log10_bulk_modulus",
              "log10_shear_modulus")

    def _arrays(self, cache_dir: Path, quiet: bool):
        root = cache_dir / self.name
        cache = root / "mp_2018.npz"
        if not cache.exists():
            url, md5, file = _MP2018
            archive = download_file(url, root / "raw" / file, md5, quiet=quiet)
            structures, ids, values = [], [], []
            with zipfile.ZipFile(archive) as z, z.open(z.namelist()[0]) as raw:
                for item in iter_json_list(io.TextIOWrapper(raw, encoding="utf-8")):
                    structures.append(parse_p1_cif(item["structure"]))
                    ids.append(item["material_id"])
                    values.append([item["formation_energy_per_atom"], item["band_gap"],
                                   item.get("K", np.nan), item.get("G", np.nan)])
            np.savez(cache, **_pack(structures), material_id=np.array(ids), values=np.array(values, dtype=float))
        with np.load(cache) as archive:   # every array read once (an NpzFile reads on every access)
            return dict(archive)

    def load(self, *, split: Optional[str] = None, cache_dir: Path, label: str = "formation_energy",
             n_train: Optional[int] = None, n_val: Optional[int] = None, seed: int = 0,
             nonzero_band_gap: Optional[bool] = None, return_info: bool = False,
             quiet: bool = False) -> Union[dict, list]:
        """Download, parse and split the crystals.

        Parameters
        ----------
        split : str or None
            ``None`` returns ``{"train", "val", "test"}``; ``"train"`` / ``"val"`` / ``"test"`` that split;
            ``"all"`` every crystal with the label.
        cache_dir : pathlib.Path
            Base cache directory; the files go to ``cache_dir/"mp_2018"``.
        label : str, optional
            The structure's ``properties``: ``"formation_energy"`` (eV/atom, default), ``"band_gap"`` (eV),
            ``"bulk_modulus"`` / ``"shear_modulus"`` (GPa) or their base-10 logarithms
            ``"log10_bulk_modulus"`` / ``"log10_shear_modulus"`` (the papers' targets).
        n_train, n_val : int, optional
            Split sizes. By default the paper's: 60,000 training crystals for the formation energy and the
            rest halved into validation and test; 80 % / 10 % / 10 % for the other labels.
        seed : int, optional
            Seed of the random split, by default 0.
        nonzero_band_gap : bool, optional
            Keep only the crystals with a band gap above zero (the paper's band-gap regression, 45,901
            crystals). By default on for ``label="band_gap"``.
        return_info : bool, optional
            Add ``info = {"material_id": ...}`` to every structure.
        quiet : bool, optional
            Suppress download progress output.

        Returns
        -------
        dict of {str: list of dict} or list of dict
            Structure dicts with ``pos``, ``atomic_numbers``, ``cell``, ``pbc`` and ``properties``.

        Raises
        ------
        ValueError
            For an unknown ``label`` or ``split``.
        """
        if label not in self.LABELS:
            raise ValueError(f"unknown label {label!r}; choose from {self.LABELS}")
        data = self._arrays(cache_dir, quiet)
        base = label.replace("log10_", "")
        column = ("formation_energy", "band_gap", "bulk_modulus", "shear_modulus").index(base)
        values = data["values"][:, column]
        keep = np.isfinite(values)
        if label.startswith("log10_"):
            keep &= values > 0
            values = np.where(keep, np.log10(np.where(keep, values, 1.0)), np.nan)
        if nonzero_band_gap if nonzero_band_gap is not None else label == "band_gap":
            keep &= data["values"][:, 1] > 0
        rows = np.flatnonzero(keep)
        ids = data["material_id"]

        def build(index):
            return [_structure(data, i, values[i], info={"material_id": str(ids[i])} if return_info else None)
                    for i in index]

        if split == "all":
            return build(rows)
        if n_train is None:
            n_train = 60000 if label == "formation_energy" else int(0.8 * len(rows))
        if n_val is None:
            n_val = (len(rows) - n_train) // 2
        return _pick({k: build(v) for k, v in _split(rows, n_train, n_val, seed).items()}, split)


class MultiFidelityBandGapBuilder(DatasetBuilder):
    """Builder for the multi-fidelity band gaps of Materials Project crystals (PBE, GLLB-SC, HSE, SCAN)."""

    name = "multifidelity_band_gap"
    description = ("Band gaps of Materials Project crystals at PBE, GLLB-SC, HSE and SCAN "
                   "(multi-fidelity MEGNet; 2019.04.01 structures).")
    #: The fidelities, in the order of the state labels of MatGL's pretrained band-gap model.
    FIDELITIES = ("pbe", "gllb-sc", "hse", "scan")

    def _arrays(self, cache_dir: Path, quiet: bool):
        root = cache_dir / self.name
        cache = root / "multifidelity_band_gap.npz"
        if not cache.exists():
            url, md5, file = _GAPS
            with gzip.open(download_file(url, root / "raw" / file, md5, quiet=quiet), "rt") as f:
                gaps = json.load(f)
            wanted = set().union(*(gaps[k] for k in self.FIDELITIES))
            url, md5, file = _MP2019
            archive = download_file(url, root / "raw" / file, md5, quiet=quiet)
            structures, ids = [], []
            with zipfile.ZipFile(archive) as z, z.open(z.namelist()[0]) as raw:
                for item in iter_json_list(io.TextIOWrapper(raw, encoding="utf-8")):
                    if item["material_id"] in wanted:
                        structures.append(parse_p1_cif(item["structure"]))
                        ids.append(item["material_id"])
            index = {m: i for i, m in enumerate(ids)}
            rows, fidelity, value = [], [], []
            for k, name in enumerate(self.FIDELITIES):
                for m, gap in gaps[name].items():
                    if m in index:
                        rows.append(index[m]); fidelity.append(k); value.append(gap)
            np.savez(cache, **_pack(structures), material_id=np.array(ids), row=np.array(rows, dtype=np.int64),
                     fidelity=np.array(fidelity, dtype=np.int64), value=np.array(value, dtype=float))
        with np.load(cache) as archive:   # every array read once (an NpzFile reads on every access)
            return dict(archive)

    def load(self, *, split: Optional[str] = None, cache_dir: Path,
             fidelities: Sequence[str] = FIDELITIES, train_fraction: float = 0.8, val_fraction: float = 0.1,
             seed: int = 0, return_info: bool = False, quiet: bool = False) -> Union[dict, list]:
        """Download, parse and split the band gaps.

        Every label is one structure dict: the same crystal appears once per fidelity that has its band gap,
        as in the paper.

        Parameters
        ----------
        split : str or None
            ``None`` returns ``{"train", "val", "test"}``; ``"train"`` / ``"val"`` / ``"test"`` that split;
            ``"all"`` every label.
        cache_dir : pathlib.Path
            Base cache directory; the files go to ``cache_dir/"multifidelity_band_gap"``.
        fidelities : sequence of str, optional
            The fidelities to load, from ``"pbe"``, ``"gllb-sc"``, ``"hse"`` and ``"scan"``. A structure's
            ``state`` is the position of its fidelity in this sequence; the default order is that of MatGL's
            pretrained model.
        train_fraction, val_fraction : float, optional
            Split fractions, drawn within every fidelity separately, by default 0.8 and 0.1 (the paper's).
        seed : int, optional
            Seed of the random split, by default 0.
        return_info : bool, optional
            Add ``info = {"material_id": ..., "fidelity": ...}`` to every structure.
        quiet : bool, optional
            Suppress download progress output.

        Returns
        -------
        dict of {str: list of dict} or list of dict
            Structure dicts with ``pos``, ``atomic_numbers``, ``cell``, ``pbc``, ``properties`` (the band gap
            in eV) and ``state`` (the fidelity label).

        Raises
        ------
        ValueError
            For an unknown fidelity or ``split``.
        """
        unknown = [f for f in fidelities if f not in self.FIDELITIES]
        if unknown:
            raise ValueError(f"unknown fidelities {unknown}; choose from {self.FIDELITIES}")
        data = self._arrays(cache_dir, quiet)
        ids = data["material_id"]
        labels = [(k, self.FIDELITIES.index(f)) for k, f in enumerate(fidelities)]
        parts = {"train": [], "val": [], "test": [], "all": []}
        for state, k in labels:
            rows = np.flatnonzero(data["fidelity"] == k)
            n_train = int(round(train_fraction * len(rows)))
            n_val = int(round(val_fraction * len(rows)))
            groups = _split(rows, n_train, n_val, seed + k)
            groups["all"] = rows
            for name, index in groups.items():
                parts[name] += [_structure(data, data["row"][j], data["value"][j], state,
                                           {"material_id": str(ids[data["row"][j]]), "fidelity": fidelities[state]}
                                           if return_info else None) for j in index]
        if split == "all":
            return parts["all"]
        return _pick({k: parts[k] for k in ("train", "val", "test")}, split)


register_dataset(MP2018Builder())
register_dataset(MultiFidelityBandGapBuilder())
