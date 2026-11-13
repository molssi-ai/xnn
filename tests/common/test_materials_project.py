"""The Materials Project datasets of the MEGNet papers: CIF and JSON readers, splits, labels and states."""
import gzip
import io
import json
import math
import zipfile

import numpy as np
import pytest

from xnn.common.data.hub import load_dataset
from xnn.common.data.hub import materials_project as mp


def cif(a=4.0, b=5.0, c=6.0, alpha=80.0, beta=95.0, gamma=100.0, sites=(("Na", 0, 0, 0), ("Cl", 0.5, 0.5, 0.5)),
        ops=("x, y, z",), occupancy=1):
    """A CIF in the layout pymatgen writes."""
    lines = ["# generated", "data_X", "_symmetry_space_group_name_H-M   'P 1'", f"_cell_length_a   {a}",
             f"_cell_length_b   {b}", f"_cell_length_c   {c}", f"_cell_angle_alpha   {alpha}",
             f"_cell_angle_beta   {beta}", f"_cell_angle_gamma   {gamma}", "_symmetry_Int_Tables_number   1",
             "loop_", " _symmetry_equiv_pos_site_id", " _symmetry_equiv_pos_as_xyz"]
    lines += [f"  {i + 1}  '{op}'" for i, op in enumerate(ops)]
    lines += ["loop_", " _atom_site_type_symbol", " _atom_site_label", " _atom_site_symmetry_multiplicity",
              " _atom_site_fract_x", " _atom_site_fract_y", " _atom_site_fract_z", " _atom_site_occupancy"]
    lines += [f"  {s}  {s}{i}  1  {x}  {y}  {z}  {occupancy}" for i, (s, x, y, z) in enumerate(sites)]
    return "\n".join(lines) + "\n"


def test_p1_cif_cell_and_positions():
    """The cell has the CIF's lengths and angles; positions are fractional coordinates times the cell."""
    cell, numbers, pos = mp.parse_p1_cif(cif())
    lengths = np.linalg.norm(cell, axis=1)
    assert np.allclose(lengths, [4.0, 5.0, 6.0])

    def angle(u, v):
        return math.degrees(math.acos(u @ v / np.linalg.norm(u) / np.linalg.norm(v)))
    assert np.allclose([angle(cell[1], cell[2]), angle(cell[0], cell[2]), angle(cell[0], cell[1])],
                       [80.0, 95.0, 100.0])
    assert numbers.tolist() == [11, 17] and np.allclose(pos[1], 0.5 * cell.sum(0))
    _, numbers, _ = mp.parse_p1_cif(cif(sites=(("Fe2+", 0, 0, 0),)))
    assert numbers.tolist() == [26]


def test_p1_cif_refuses_symmetry_and_disorder():
    """Other space groups and partial occupancies are refused."""
    with pytest.raises(ValueError, match="P1"):
        mp.parse_p1_cif(cif(ops=("x, y, z", "-x, -y, -z")))
    with pytest.raises(ValueError, match="P1"):
        mp.parse_p1_cif(cif(ops=("-x, y, z",)))
    with pytest.raises(ValueError, match="occupied"):
        mp.parse_p1_cif(cif(occupancy=0.5))
    with pytest.raises(ValueError, match="no atom sites"):
        mp.parse_p1_cif(cif(sites=()))


def test_json_list_is_read_item_by_item_across_chunks():
    """Items cut by the chunk boundaries are completed; an empty list gives nothing; a broken file raises."""
    items = [{"a": i, "s": "x" * i, "l": list(range(i))} for i in range(12)]
    text = "[ " + " ,\n".join(json.dumps(x) for x in items) + " ]"
    assert list(mp.iter_json_list(io.StringIO(text), chunk=7)) == items
    assert list(mp.iter_json_list(io.StringIO("[]"))) == []
    assert list(mp.iter_json_list(io.StringIO("[ {\"a\": 1}"))) == [{"a": 1}]
    with pytest.raises(json.JSONDecodeError):
        list(mp.iter_json_list(io.StringIO('[{"a": 1'), chunk=4))


def _zip(path, name, items):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(name, json.dumps(items))


def _entries(n, rng, moduli=()):
    out = []
    for i in range(n):
        e = {"material_id": f"mp-{i}", "graph": {"atom": [1], "bond": [1.0]}, "state": [[0, 0]],
             "structure": cif(a=3 + rng.uniform(), sites=(("Li", 0, 0, 0), ("O", 0.5, 0.5, 0.5))),
             "formation_energy_per_atom": -float(i), "band_gap": float(i % 3)}
        if i in moduli:
            e["K"], e["G"] = 10.0 * (i + 1), float(i)
        out.append(e)
    return out


@pytest.fixture
def fake_files(monkeypatch):
    """Small synthetic stand-ins for the three downloads, written where the builders expect them."""
    rng = np.random.default_rng(0)
    calls = []

    def download(url, dest, md5=None, quiet=False):
        calls.append(dest.name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.name == "mp.2018.6.1.json.zip":
            _zip(dest, "mp.2018.6.1.json", _entries(20, rng, moduli=range(10)))
        elif dest.name == "band_gap_no_structs.gz":
            gaps = {"pbe": {f"mp-{i}": float(i % 4) for i in range(20)},
                    "gllb-sc": {f"mp-{i}": 2.0 + i for i in range(10)},
                    "hse": {"mp-1": 1.5, "mp-2": 0.0, "mp-99": 3.0},   # mp-99 has no structure
                    "scan": {f"mp-{i}": 1.0 for i in range(5)}, "ordered_exp": {"icsd-1": 1.0}}
            with gzip.open(dest, "wt") as f:
                json.dump(gaps, f)
        else:
            _zip(dest, "mp.2019.04.01.json", _entries(30, rng))
        return dest

    monkeypatch.setattr(mp, "download_file", download)
    return calls


def test_mp_2018_labels_and_splits(tmp_path, fake_files):
    """Formation energies, band gaps (nonzero by default) and moduli (GPa or log10) with the requested splits."""
    splits = load_dataset("mp_2018", cache_dir=tmp_path, n_train=12, n_val=4, return_info=True)
    assert [len(splits[k]) for k in ("train", "val", "test")] == [12, 4, 4]
    s = splits["train"][0]
    assert s["properties"] == -int(s["info"]["material_id"][3:]) and s["pbc"] == [True] * 3
    assert s["cell"].shape == (3, 3) and s["atomic_numbers"].tolist() == [3, 8]
    gaps = load_dataset("mp_2018", cache_dir=tmp_path, label="band_gap", split="all")
    assert len(gaps) == 13 and all(g["properties"] > 0 for g in gaps)
    assert len(load_dataset("mp_2018", cache_dir=tmp_path, label="band_gap", split="all", nonzero_band_gap=False)) == 20
    bulk = load_dataset("mp_2018", cache_dir=tmp_path, label="bulk_modulus", split="all")
    assert sorted(b["properties"] for b in bulk) == [10.0 * (i + 1) for i in range(10)]
    log_shear = load_dataset("mp_2018", cache_dir=tmp_path, label="log10_shear_modulus", split="all")
    assert len(log_shear) == 9 and min(s["properties"] for s in log_shear) == 0.0
    parts = load_dataset("mp_2018", cache_dir=tmp_path, label="bulk_modulus")
    assert [len(parts[k]) for k in ("train", "val", "test")] == [8, 1, 1]
    assert fake_files.count("mp.2018.6.1.json.zip") == 1       # parsed once, then cached
    with pytest.raises(ValueError, match="label"):
        load_dataset("mp_2018", cache_dir=tmp_path, label="volume")
    with pytest.raises(ValueError, match="split"):
        load_dataset("mp_2018", cache_dir=tmp_path, n_train=12, n_val=4, split="dev")


def test_mp_2018_feeds_megnet(tmp_path, fake_files):
    """With a cutoff the splits are datasets that MEGNet evaluates."""
    from xnn.common.data import collate
    from xnn.gnn.models import MEGNet
    train = load_dataset("mp_2018", cache_dir=tmp_path, n_train=12, n_val=4, split="train", cutoff=4.0)
    batch = collate([train[i] for i in range(3)])
    assert batch.properties.shape == (3,)
    assert MEGNet()(batch)["properties"].shape == (3,)


def test_multifidelity_band_gaps(tmp_path, fake_files):
    """One structure per label, the fidelity as state, splits within every fidelity, unknown ids dropped."""
    everything = load_dataset("multifidelity_band_gap", cache_dir=tmp_path, split="all", return_info=True)
    counts = {f: sum(s["info"]["fidelity"] == f for s in everything) for f in ("pbe", "gllb-sc", "hse", "scan")}
    assert counts == {"pbe": 20, "gllb-sc": 10, "hse": 2, "scan": 5}
    assert {s["state"] for s in everything if s["info"]["fidelity"] == "hse"} == {2}
    hse = [s for s in everything if s["info"]["fidelity"] == "hse"]
    assert {(s["info"]["material_id"], s["properties"]) for s in hse} == {("mp-1", 1.5), ("mp-2", 0.0)}
    splits = load_dataset("multifidelity_band_gap", cache_dir=tmp_path, fidelities=("hse", "pbe"))
    assert sum(len(v) for v in splits.values()) == 22
    assert {s["state"] for s in splits["train"]} == {0, 1}
    assert len([s for s in splits["train"] if s["state"] == 1]) == 16
    assert fake_files.count("mp.2019.04.01.json.zip") == 1
    with pytest.raises(ValueError, match="fidelities"):
        load_dataset("multifidelity_band_gap", cache_dir=tmp_path, fidelities=("experiment",))
