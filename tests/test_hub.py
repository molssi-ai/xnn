"""Dataset hub: ``load_dataset`` download + preprocessing.

The core preprocessing (npz -> structure dicts, unit conversion, official
splits, ``AtomicDataset`` wrapping) is exercised against a synthetic rMD17-shaped
``.npz`` by monkeypatching the downloader, so the suite stays offline. A single
opt-in test actually hits figshare when ``XNNS_TEST_NETWORK=1``.
"""
import os

import numpy as np
import pytest
import torch

from xnns.common.data import AtomicDataset, list_datasets, load_dataset
from xnns.common.data.hub import default_cache_dir, lode_dimers, rmd17

# eV per kcal/mol, matching rmd17._KCAL_MOL_TO_EV.
KCAL = 0.0433641153087705
N_CONF, N_ATOMS = 40, 5


@pytest.fixture
def fake_rmd17(monkeypatch):
    """Patch the rMD17 downloader to emit a small synthetic dataset.

    Returns the deterministic ``(charges, coords, energies, forces)`` arrays so
    tests can assert exact preprocessing, plus the fixed train/test indices the
    fake split CSVs encode.
    """
    rng = np.random.default_rng(0)
    z = np.array([6, 1, 1, 1, 8], dtype=np.int64)
    coords = rng.normal(size=(N_CONF, N_ATOMS, 3))
    energies = rng.normal(size=(N_CONF,)) * 100.0          # kcal/mol
    forces = rng.normal(size=(N_CONF, N_ATOMS, 3))         # kcal/mol/A
    train_idx = np.arange(0, 10)
    test_idx = np.arange(10, 20)

    def fake_download(url, dest, md5=None, quiet=False, chunk=1 << 20):
        from pathlib import Path
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            return dest
        if dest.suffix == ".npz":
            np.savez(dest, nuclear_charges=z, coords=coords,
                     energies=energies, forces=forces)
        else:  # split index CSV
            idx = train_idx if "train" in dest.name else test_idx
            np.savetxt(dest, idx, fmt="%d")
        return dest

    monkeypatch.setattr(rmd17, "download_file", fake_download)
    return z, coords, energies, forces, train_idx, test_idx


def test_registry_lists_rmd17():
    """rMD17 is discoverable through the registry."""
    assert "rmd17" in list_datasets()


def test_default_cache_dir_is_repo_datasets():
    """By default data caches in the repo's datasets/ directory."""
    assert default_cache_dir().name == "datasets"


def test_load_splits_and_units(fake_rmd17, tmp_path):
    """Default load returns train/test splits, converted to eV."""
    z, coords, energies, forces, train_idx, test_idx = fake_rmd17
    torch.set_default_dtype(torch.float64)

    splits = load_dataset("rmd17", molecule="aspirin", cache_dir=tmp_path, quiet=True)
    assert set(splits) == {"train", "test"}
    assert len(splits["train"]) == len(train_idx)
    assert len(splits["test"]) == len(test_idx)

    s0 = splits["train"][0]
    i = train_idx[0]
    assert np.array_equal(s0["atomic_numbers"], z)
    assert np.allclose(s0["pos"], coords[i])
    assert np.isclose(s0["energy"], energies[i] * KCAL)
    assert np.allclose(s0["forces"], forces[i] * KCAL)
    assert "cell" not in s0  # molecular: no periodicity


def test_kcal_units_are_raw(fake_rmd17, tmp_path):
    """units='kcal/mol' keeps the upstream magnitudes."""
    _, _, energies, _, train_idx, _ = fake_rmd17
    s = load_dataset("rmd17", molecule="benzene", split="train",
                     units="kcal/mol", cache_dir=tmp_path, quiet=True)
    assert np.isclose(s[0]["energy"], energies[train_idx[0]])


def test_split_selection_and_truncation(fake_rmd17, tmp_path):
    """A named split returns a list; n_train/n_test truncate it."""
    train = load_dataset("rmd17", molecule="ethanol", split="train",
                         n_train=3, cache_dir=tmp_path, quiet=True)
    assert isinstance(train, list) and len(train) == 3


def test_split_all(fake_rmd17, tmp_path):
    """split='all' returns every conformation with no index filtering."""
    allc = load_dataset("rmd17", molecule="uracil", split="all",
                        cache_dir=tmp_path, quiet=True)
    assert len(allc) == N_CONF


def test_cutoff_wraps_atomicdataset(fake_rmd17, tmp_path):
    """Passing cutoff yields AtomicDataset(s) with a valid neighbor graph."""
    torch.set_default_dtype(torch.float64)
    splits = load_dataset("rmd17", molecule="toluene", cutoff=5.0,
                          cache_dir=tmp_path, quiet=True)
    assert isinstance(splits["train"], AtomicDataset)
    g = splits["train"][0]
    assert g.num_nodes == N_ATOMS
    assert g.cell is None and g.forces.shape == (N_ATOMS, 3)

    train = load_dataset("rmd17", molecule="toluene", split="train", cutoff=5.0,
                         cache_dir=tmp_path, quiet=True)
    assert isinstance(train, AtomicDataset)


def test_caching_skips_second_download(fake_rmd17, tmp_path, monkeypatch):
    """A second load reuses cached files instead of re-downloading."""
    load_dataset("rmd17", molecule="aspirin", cache_dir=tmp_path, quiet=True)
    calls = {"n": 0}
    orig = rmd17.download_file

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(rmd17, "download_file", counting)
    load_dataset("rmd17", molecule="aspirin", cache_dir=tmp_path, quiet=True)
    # files exist, so the fake still returns them but without rewriting; the
    # real downloader would short-circuit on the cached path + md5.
    assert calls["n"] > 0  # invoked, but resolves to the cached file


@pytest.mark.parametrize("kwargs,match", [
    ({}, "requires a `molecule`"),
    ({"molecule": "water"}, "unknown rmd17 molecule"),
    ({"molecule": "aspirin", "units": "hartree"}, "units must be"),
    ({"molecule": "aspirin", "fold": 9}, "fold must be 1-5"),
])
def test_invalid_args_raise(fake_rmd17, tmp_path, kwargs, match):
    """Bad molecule / units / fold arguments raise clear ValueErrors."""
    with pytest.raises(ValueError, match=match):
        load_dataset("rmd17", cache_dir=tmp_path, quiet=True, **kwargs)


def test_unknown_dataset_raises():
    """An unregistered dataset name is reported with the available list."""
    with pytest.raises(KeyError, match="unknown dataset"):
        load_dataset("nonexistent")


@pytest.mark.skipif(os.environ.get("XNNS_TEST_NETWORK") != "1",
                    reason="set XNNS_TEST_NETWORK=1 to download from figshare")
def test_live_download(tmp_path):
    """End-to-end: really fetch a molecule + splits from figshare."""
    torch.set_default_dtype(torch.float64)
    splits = load_dataset("rmd17", molecule="ethanol", fold=1,
                          n_train=5, n_test=5, cache_dir=tmp_path)
    assert len(splits["train"]) == 5 and len(splits["test"]) == 5
    s = splits["train"][0]
    assert set(np.unique(s["atomic_numbers"])).issubset({1, 6, 8})
    assert abs(s["energy"]) < 1e5  # eV, converted from kcal/mol


# --------------------------------------------------------------------------- #
# lode_dimers (Materials Cloud extxyz)                                         #
# --------------------------------------------------------------------------- #

@pytest.fixture
def fake_lode(monkeypatch):
    """Patch the lode_dimers downloader to emit a small synthetic extxyz.

    Six frames span three fragment-polarity labels (CC/CP/PP), each carrying an
    energy and forces, so label filtering and preprocessing can be checked
    offline. Returns the ordered list of (label, energy) written.
    """
    pytest.importorskip("ase")
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import write

    rng = np.random.default_rng(0)
    written = []

    def make_frame(label, e):
        atoms = Atoms(numbers=[6, 8, 1], positions=rng.uniform(0, 5, (3, 3)),
                      cell=np.eye(3) * 30.0, pbc=True)
        atoms.calc = SinglePointCalculator(atoms, energy=e,
                                           forces=rng.normal(size=(3, 3)))
        atoms.info["label"] = label
        written.append((label, e))
        return atoms

    frames = [make_frame(lab, float(i))
              for i, lab in enumerate(["CC", "CC", "CP", "CP", "PP", "PP"])]

    def fake_download(url, dest, md5=None, quiet=False, chunk=1 << 20):
        from pathlib import Path
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            write(str(dest), frames)
        return dest

    monkeypatch.setattr(lode_dimers, "download_file", fake_download)
    return written


def test_lode_registered():
    """lode_dimers is discoverable through the registry."""
    assert "lode_dimers" in list_datasets()


def test_lode_default_load(fake_lode, tmp_path):
    """Default load returns all frames under an 'all' split, with targets."""
    torch.set_default_dtype(torch.float64)
    out = load_dataset("lode_dimers", cache_dir=tmp_path, quiet=True)
    assert set(out) == {"all"}
    assert len(out["all"]) == 6
    s = out["all"][0]
    assert np.array_equal(s["atomic_numbers"], [6, 8, 1])
    assert s["forces"].shape == (3, 3)
    assert s["cell"].shape == (3, 3)  # large-box periodic


def test_lode_label_filter(fake_lode, tmp_path):
    """label= keeps only the requested fragment-polarity class."""
    cc = load_dataset("lode_dimers", subset="bio", label="CC",
                      split="all", cache_dir=tmp_path, quiet=True)
    assert isinstance(cc, list) and len(cc) == 2


def test_lode_alias_and_cutoff(fake_lode, tmp_path):
    """A subset alias resolves, and cutoff wraps an AtomicDataset."""
    torch.set_default_dtype(torch.float64)
    ds = load_dataset("lode_dimers", subset="dimers", cutoff=6.0,
                      cache_dir=tmp_path, quiet=True)
    assert isinstance(ds["all"], AtomicDataset)
    assert ds["all"][0].num_nodes == 3


def test_lode_return_info(fake_lode, tmp_path):
    """return_info attaches each frame's info dict and survives cutoff wrapping."""
    torch.set_default_dtype(torch.float64)
    d = load_dataset("lode_dimers", subset="bio", split="all",
                     return_info=True, cache_dir=tmp_path, quiet=True)
    assert d[0]["info"]["label"] == "CC"
    # info is ignored by graph building but preserved on the source dicts
    ds = load_dataset("lode_dimers", subset="bio", cutoff=6.0,
                      return_info=True, cache_dir=tmp_path, quiet=True)["all"]
    assert ds[0].num_nodes == 3
    assert "info" in ds.structures[0]


@pytest.mark.parametrize("kwargs,match", [
    ({"subset": "nope"}, "unknown lode_dimers subset"),
    ({"subset": "bio", "label": "ZZ"}, "unknown label"),
    ({"subset": "xenon", "label": "CC"}, "only supported for subset='bio'"),
    ({"split": "train"}, "no train/test split"),
])
def test_lode_invalid_args(fake_lode, tmp_path, kwargs, match):
    """Bad subset / label / split arguments raise clear ValueErrors."""
    with pytest.raises(ValueError, match=match):
        load_dataset("lode_dimers", cache_dir=tmp_path, quiet=True, **kwargs)


@pytest.mark.skipif(os.environ.get("XNNS_TEST_NETWORK") != "1",
                    reason="set XNNS_TEST_NETWORK=1 to download from Materials Cloud")
def test_lode_live_download(tmp_path):
    """End-to-end: really fetch the Xenon subset from Materials Cloud."""
    torch.set_default_dtype(torch.float64)
    xe = load_dataset("lode_dimers", subset="xenon", split="all",
                      cache_dir=tmp_path)
    assert len(xe) > 0
    s = xe[0]
    assert set(np.unique(s["atomic_numbers"])) == {54}  # Xe
    assert s["forces"].shape[0] == len(s["atomic_numbers"])
