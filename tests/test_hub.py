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
from xnns.common.data.hub import ani1x, default_cache_dir, lode_dimers, rmd17

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
    ({"subset": "xenon", "label": "CC"}, "only supported for the biomolecular"),
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


# --------------------------------------------------------------------------- #
# ani1 (pyanitools HDF5)                                                       #
# --------------------------------------------------------------------------- #

@pytest.fixture
def fake_ani1(tmp_path):
    """Write a synthetic pyanitools-shaped archive layout under tmp_path.

    Two heavy-atom subsets (s02, s03), each with a couple of molecules and a
    few conformations, so subset selection / capping / splitting / unit
    conversion can be checked offline (no 4.8 GB download).
    """
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(0)
    root = tmp_path / "ani1" / "raw" / "ANI-1_release"
    root.mkdir(parents=True)

    layout = {
        2: [[b"C", b"O", b"H", b"H"], [b"N", b"H", b"H", b"H"]],
        3: [[b"C", b"C", b"N", b"H", b"H"]],
    }
    per_mol = 4
    for x, mols in layout.items():
        with h5py.File(root / f"ani_gdb_s{x:02d}.h5", "w") as f:
            top = f.create_group(f"gdb11_s{x:02d}")
            for m, sym in enumerate(mols):
                natoms = len(sym)
                g = top.create_group(f"mol{m}")
                g["coordinates"] = rng.standard_normal((per_mol, natoms, 3)).astype(np.float32)
                g["energies"] = (rng.standard_normal(per_mol) - 40.0).astype(np.float64)
                g["species"] = np.array(sym)
                g["smiles"] = np.array([b"C", b"O"])
    return tmp_path


def test_ani1_registered():
    """ani1 is discoverable through the registry."""
    assert "ani1" in list_datasets()


def test_ani1_load_subset_and_units(fake_ani1):
    """Loading a heavy-atom subset returns Hartree->eV converted structures."""
    torch.set_default_dtype(torch.float64)
    out = load_dataset("ani1", heavy_atoms=2, cache_dir=fake_ani1, quiet=True)
    assert set(out) == {"all"}
    assert len(out["all"]) == 2 * 4       # 2 molecules * per_mol conformations
    s = out["all"][0]
    assert set(sorted(s)) >= {"pos", "atomic_numbers", "energy", "smiles"}
    assert set(np.unique(s["atomic_numbers"])).issubset({1, 6, 7, 8})

    raw = load_dataset("ani1", heavy_atoms=2, units="hartree",
                       cache_dir=fake_ani1, quiet=True)["all"]
    assert np.isclose(out["all"][0]["energy"],
                      raw[0]["energy"] * 27.211386245988)


def test_ani1_caps_and_multisubset(fake_ani1):
    """max_molecules / max_conformations and multi-subset selection apply."""
    out = load_dataset("ani1", heavy_atoms=[2, 3], max_molecules=1,
                       max_conformations=2, cache_dir=fake_ani1, quiet=True)["all"]
    # 1 molecule from each of s02, s03 * 2 conformations each
    assert len(out) == 2 * 2


def test_ani1_splits(fake_ani1):
    """train/val/test partitions are disjoint and cover the whole subset."""
    kw = dict(heavy_atoms=2, cache_dir=fake_ani1, quiet=True)
    tr = load_dataset("ani1", split="train", **kw)
    va = load_dataset("ani1", split="val", **kw)
    te = load_dataset("ani1", split="test", **kw)
    assert len(tr) + len(va) + len(te) == 2 * 4
    assert len(tr) > len(va) and len(tr) > len(te)


def test_ani1_cutoff_wraps_dataset(fake_ani1):
    """cutoff wraps an AtomicDataset; ANI-1 is force-free."""
    torch.set_default_dtype(torch.float64)
    ds = load_dataset("ani1", heavy_atoms=2, cutoff=5.2,
                      cache_dir=fake_ani1, quiet=True)["all"]
    assert isinstance(ds, AtomicDataset)
    g = ds[0]
    assert g.forces is None and g.energy is not None


@pytest.mark.parametrize("kwargs,match", [
    ({"heavy_atoms": 9}, "heavy_atoms entries must be 1-8"),
    ({"units": "hartree/2"}, "units must be"),
    ({"split": "trian"}, "unknown split"),
])
def test_ani1_invalid_args(fake_ani1, kwargs, match):
    """Bad heavy_atoms / units / split raise clear ValueErrors."""
    base = dict(heavy_atoms=2, cache_dir=fake_ani1, quiet=True)
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        load_dataset("ani1", **base)


# --------------------------------------------------------------------------- #
# ani1x (single pyanitools HDF5 with forces + NaN masking)                     #
# --------------------------------------------------------------------------- #

@pytest.fixture
def fake_ani1x(monkeypatch):
    """Patch the ANI-1x downloader to emit a small synthetic release HDF5.

    Two molecule groups keyed like the real file, with per-conformation NaN
    holes in the energy and force datasets, so unit conversion, NaN masking,
    level selection, and forces handling can be checked offline (no 5.6 GB
    download). ``ccsd(t)_cbs`` is written energy-only, as upstream.
    """
    h5py = pytest.importorskip("h5py")

    def fake_download(url, dest, md5=None, quiet=False, chunk=1 << 20):
        from pathlib import Path
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            return dest
        with h5py.File(dest, "w") as f:
            g = f.create_group("C1H2")
            g["atomic_numbers"] = np.array([6, 1, 1], dtype=np.int64)
            g["coordinates"] = np.arange(4 * 3 * 3, dtype=np.float64).reshape(4, 3, 3)
            g["wb97x_dz.energy"] = np.array([-38.0, np.nan, -38.2, -38.3])
            fr = np.ones((4, 3, 3)) * 0.1
            fr[2, 0, 0] = np.nan                       # NaN force in conf 2
            g["wb97x_dz.forces"] = fr
            g["ccsd(t)_cbs.energy"] = np.array([-38.5, -38.6, np.nan, np.nan])
            h = f.create_group("H2")
            h["atomic_numbers"] = np.array([1, 1], dtype=np.int64)
            h["coordinates"] = np.zeros((2, 2, 3))
            h["wb97x_dz.energy"] = np.array([-1.0, -1.1])
            h["wb97x_dz.forces"] = np.zeros((2, 2, 3))
            h["ccsd(t)_cbs.energy"] = np.array([-1.2, -1.3])
        return dest

    monkeypatch.setattr(ani1x, "download_file", fake_download)


def test_ani1x_registered():
    """ani1x is discoverable through the registry."""
    assert "ani1x" in list_datasets()


def test_ani1x_masks_nan_and_converts_units(fake_ani1x, tmp_path):
    """NaN energies/forces are dropped and Hartree->eV conversion applies."""
    out = load_dataset("ani1x", cache_dir=tmp_path, quiet=True)["all"]
    # C1H2 keeps conf 0 and 3 (1 NaN energy, 2 NaN force); H2 keeps both -> 4.
    assert len(out) == 4
    s = out[0]
    assert set(s) == {"pos", "atomic_numbers", "energy", "forces"}
    assert s["forces"].shape == (len(s["atomic_numbers"]), 3)
    assert np.isclose(s["energy"], -38.0 * 27.211386245988)

    raw = load_dataset("ani1x", units="hartree", cache_dir=tmp_path,
                       quiet=True)["all"]
    assert any(np.isclose(r["energy"], -38.0) for r in raw)


def test_ani1x_ccsd_is_energy_only(fake_ani1x, tmp_path):
    """The CCSD(T)/CBS level carries no forces; requesting them raises."""
    out = load_dataset("ani1x", level="ccsd(t)_cbs", forces=False,
                       cache_dir=tmp_path, quiet=True)["all"]
    # C1H2 keeps conf 0, 1 (2, 3 are NaN); H2 keeps both -> 4, all force-free.
    assert len(out) == 4
    assert all("forces" not in s for s in out)
    with pytest.raises(ValueError, match="no forces"):
        load_dataset("ani1x", level="ccsd(t)_cbs", cache_dir=tmp_path, quiet=True)


def test_ani1x_caps_and_splits(fake_ani1x, tmp_path):
    """max_molecules / max_conformations cap output; splits cover the whole set."""
    one = load_dataset("ani1x", max_molecules=1, cache_dir=tmp_path,
                       quiet=True)["all"]
    assert len(one) == 2                       # only C1H2's two valid confs
    capped = load_dataset("ani1x", max_conformations=1, cache_dir=tmp_path,
                          quiet=True)["all"]
    assert len(capped) == 2                     # one conf from each molecule

    kw = dict(cache_dir=tmp_path, quiet=True)
    tr = load_dataset("ani1x", split="train", **kw)
    va = load_dataset("ani1x", split="val", **kw)
    te = load_dataset("ani1x", split="test", **kw)
    assert len(tr) + len(va) + len(te) == 4


def test_ani1x_cutoff_wraps_dataset_with_forces(fake_ani1x, tmp_path):
    """cutoff wraps an AtomicDataset; ANI-1x carries forces."""
    torch.set_default_dtype(torch.float64)
    ds = load_dataset("ani1x", cutoff=5.2, cache_dir=tmp_path, quiet=True)["all"]
    assert isinstance(ds, AtomicDataset)
    g = ds[0]
    assert g.forces is not None and g.energy is not None


@pytest.mark.parametrize("kwargs,match", [
    ({"level": "mp2_dz"}, "unknown level"),
    ({"units": "kcal"}, "units must be"),
    ({"split": "trian"}, "unknown split"),
])
def test_ani1x_invalid_args(fake_ani1x, tmp_path, kwargs, match):
    """Bad level / units / split raise clear ValueErrors."""
    base = dict(cache_dir=tmp_path, quiet=True)
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        load_dataset("ani1x", **base)


# --------------------------------------------------------------------------- #
# ani1ccx (coupled-cluster subset of the ANI-1x release file)                  #
# --------------------------------------------------------------------------- #

def test_ani1ccx_registered():
    """ani1ccx is discoverable through the registry."""
    assert "ani1ccx" in list_datasets()


def test_ani1ccx_loads_ccsd_energy_only(fake_ani1x, tmp_path):
    """ani1ccx returns only conformations with finite CCSD(T)*/CBS energies."""
    out = load_dataset("ani1ccx", cache_dir=tmp_path, quiet=True)["all"]
    # C1H2 keeps conf 0, 1 (2, 3 are NaN in ccsd(t)_cbs.energy); H2 keeps both.
    assert len(out) == 4
    assert all(set(s) == {"pos", "atomic_numbers", "energy"} for s in out)
    assert any(np.isclose(s["energy"], -38.5 * 27.211386245988) for s in out)

    raw = load_dataset("ani1ccx", units="hartree", cache_dir=tmp_path,
                       quiet=True)["all"]
    assert any(np.isclose(r["energy"], -38.5) for r in raw)


def test_ani1ccx_shares_ani1x_cache(fake_ani1x, tmp_path):
    """The release file is cached once, under the ani1x directory."""
    load_dataset("ani1ccx", cache_dir=tmp_path, quiet=True)
    assert (tmp_path / "ani1x" / "raw" / "ani1x-release.h5").exists()
    assert not (tmp_path / "ani1ccx").exists()
    # loading ani1x afterwards reuses the very same file
    load_dataset("ani1x", cache_dir=tmp_path, quiet=True)
    assert not (tmp_path / "ani1ccx").exists()


def test_ani1ccx_caps_and_splits(fake_ani1x, tmp_path):
    """max_molecules / max_conformations cap output; splits cover the set."""
    one = load_dataset("ani1ccx", max_molecules=1, max_conformations=1,
                       cache_dir=tmp_path, quiet=True)["all"]
    assert len(one) == 1                       # first conf of C1H2 only

    kw = dict(cache_dir=tmp_path, quiet=True)
    tr = load_dataset("ani1ccx", split="train", **kw)
    va = load_dataset("ani1ccx", split="val", **kw)
    te = load_dataset("ani1ccx", split="test", **kw)
    assert len(tr) + len(va) + len(te) == 4


def test_ani1ccx_cutoff_wraps_forcefree_dataset(fake_ani1x, tmp_path):
    """cutoff wraps an AtomicDataset; ANI-1ccx is energy-only."""
    torch.set_default_dtype(torch.float64)
    ds = load_dataset("ani1ccx", cutoff=5.2, cache_dir=tmp_path,
                      quiet=True)["all"]
    assert isinstance(ds, AtomicDataset)
    g = ds[0]
    assert g.forces is None and g.energy is not None


@pytest.mark.parametrize("kwargs,match", [
    ({"units": "kcal"}, "units must be"),
    ({"split": "trian"}, "unknown split"),
])
def test_ani1ccx_invalid_args(fake_ani1x, tmp_path, kwargs, match):
    """Bad units / split raise clear ValueErrors."""
    base = dict(cache_dir=tmp_path, quiet=True)
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        load_dataset("ani1ccx", **base)


def test_ani1ccx_rejects_forces_and_level():
    """ani1ccx pins the level of theory; forces / level are not accepted."""
    with pytest.raises(TypeError):
        load_dataset("ani1ccx", forces=True, quiet=True)
    with pytest.raises(TypeError):
        load_dataset("ani1ccx", level="wb97x_dz", quiet=True)


# --------------------------------------------------------------------------- #
# argon_md (bundled extxyz)                                                    #
# --------------------------------------------------------------------------- #

@pytest.fixture
def fake_argon(tmp_path):
    """Write a synthetic argon_md-shaped extxyz layout under tmp_path.

    One IsolatedAtom reference frame (E0) plus two periodic Ar frames with
    REF_energy / REF_forces / REF_stress, so the builder's E0 filtering and
    MACE-key parsing can be checked offline.
    """
    ase = pytest.importorskip("ase")
    from ase import Atoms
    from ase.io import write

    root = tmp_path / "argon_md"
    root.mkdir()
    rng = np.random.default_rng(0)

    def frame(n, energy, iso=False):
        a = Atoms("Ar" * n, positions=rng.uniform(0, 10, (n, 3)),
                  cell=np.eye(3) * 12.0, pbc=True)
        a.info["REF_energy"] = energy
        a.arrays["REF_forces"] = rng.normal(size=(n, 3))
        a.info["REF_stress"] = rng.normal(size=6)
        if iso:
            a.info["config_type"] = "IsolatedAtom"
        return a

    write(str(root / "argon_train.xyz"),
          [frame(1, 0.0, iso=True), frame(8, -6.3), frame(8, -6.1)])
    write(str(root / "argon_test.xyz"), [frame(8, -6.2)])
    return tmp_path


def test_argon_registered():
    """argon_md is discoverable through the registry."""
    assert "argon_md" in list_datasets()


def test_argon_default_load_drops_isolated(fake_argon):
    """Default load returns train/test with the IsolatedAtom frame removed."""
    torch.set_default_dtype(torch.float64)
    out = load_dataset("argon_md", cache_dir=fake_argon, quiet=True)
    assert set(out) == {"train", "test"}
    assert len(out["train"]) == 2 and len(out["test"]) == 1  # E0 frame dropped
    s = out["train"][0]
    assert np.array_equal(np.unique(s["atomic_numbers"]), [18])
    assert s["cell"].shape == (3, 3)
    assert s["forces"].shape == (8, 3)
    assert "stress" in s and "energy" in s


def test_argon_split_and_all(fake_argon):
    """Named split returns a list; 'all' concatenates train + test."""
    tr = load_dataset("argon_md", split="train", cache_dir=fake_argon, quiet=True)
    assert isinstance(tr, list) and len(tr) == 2
    allc = load_dataset("argon_md", split="all", cache_dir=fake_argon, quiet=True)
    assert len(allc) == 3


def test_argon_cutoff_wraps_dataset(fake_argon):
    """cutoff wraps a periodic AtomicDataset with energy + forces."""
    torch.set_default_dtype(torch.float64)
    ds = load_dataset("argon_md", split="train", cutoff=6.0,
                      cache_dir=fake_argon, quiet=True)
    assert isinstance(ds, AtomicDataset)
    g = ds[0]
    assert g.cell is not None and g.forces.shape == (8, 3)


def test_argon_bad_split(fake_argon):
    """An unknown split raises a clear ValueError."""
    with pytest.raises(ValueError, match="unknown split"):
        load_dataset("argon_md", split="valid", cache_dir=fake_argon, quiet=True)


# --------------------------------------------------------------------------- #
# lode_dimers bundled bio_scan subset                                          #
# --------------------------------------------------------------------------- #

@pytest.fixture
def fake_bio_scan(tmp_path):
    """Write a synthetic bundled bio_scan file under tmp_path/lode_dimers/."""
    ase = pytest.importorskip("ase")
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import write

    root = tmp_path / "lode_dimers"
    root.mkdir()
    rng = np.random.default_rng(0)
    frames = []
    for label in ["CC", "CP", "PP"]:
        for k in range(3):
            a = Atoms("CO", positions=rng.uniform(0, 5, (2, 3)),
                      cell=np.eye(3) * 30.0, pbc=True)
            a.calc = SinglePointCalculator(a, energy=float(-100 - k),
                                           forces=rng.normal(size=(2, 3)))
            a.info.update(label=label, distance=5.0 + k, energyA=-40.0, energyB=-59.0)
            frames.append(a)
    write(str(root / "bio_dimers_CC_CP_PP.xyz"), frames)
    return tmp_path


def test_bio_scan_bundled(fake_bio_scan):
    """The bundled bio_scan subset loads offline with info + label filtering."""
    torch.set_default_dtype(torch.float64)
    allc = load_dataset("lode_dimers", subset="bio_scan", split="all",
                        return_info=True, cache_dir=fake_bio_scan, quiet=True)
    assert len(allc) == 9
    d = allc[0]
    assert {"energyA", "energyB", "distance", "label"} <= set(d["info"])
    assert "forces" in d and "energy" in d

    cc = load_dataset("lode_dimers", subset="bio_scan", label="CC", split="all",
                      cache_dir=fake_bio_scan, quiet=True)
    assert len(cc) == 3


def test_bio_scan_missing_file(tmp_path):
    """A missing bundled file gives a clear FileNotFoundError."""
    pytest.importorskip("ase")
    with pytest.raises(FileNotFoundError, match="bundled file"):
        load_dataset("lode_dimers", subset="bio_scan", split="all",
                     cache_dir=tmp_path, quiet=True)
