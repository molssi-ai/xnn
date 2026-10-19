"""Tests of the GMTKN55 dataset builder and its scoring helpers."""
import io
import os
import re
import urllib.request
import zipfile

import numpy as np
import pytest

from xnn.common.data import load_dataset, list_datasets
from xnn.common.data.hub import gmtkn55
from xnn.common.data.hub.gmtkn55 import (
    CATEGORIES, PAPER_SCALE, SUBSETS, parse_reactions, reaction_energies,
    subset_statistics, wtmad2,
)

HARTREE_TO_KCAL = gmtkn55.HARTREE_TO_EV / gmtkn55.KCAL_MOL_TO_EV


def _xyz(symbols, scale=1.0):
    lines = [str(len(symbols)), ""]
    for k, s in enumerate(symbols):
        lines.append(f"{s} {0.9 * k * scale:.6f} 0.0 0.0")
    return "\n".join(lines) + "\n"


def _orca(energy):
    return f"some header\nFINAL SINGLE POINT ENERGY {energy:.12f}\nmore text\n"


# species of the synthetic archive: name -> (symbols, charge, unpaired, PBEh-3c energy / Ha)
_S22 = {"01": ("NNHHHHHH", 0, 0, -113.0), "01a": ("NHHH", 0, 0, -56.6),
        "02": ("OOHHHH", 0, 0, -152.9), "02a": ("OHH", 0, 0, -76.5)}
_BH76 = {"h": ("H", 0, 1, -0.5), "n2o": ("NNO", 0, 0, -184.5), "oh": ("OH", 0, 1, -75.7),
         "n2": ("NN", 0, 0, -109.4), "n2ohts": ("NNOH", 0, 1, -184.9), "f-": ("F", -1, 0, -99.8),
         "ch3cl": ("CHHHCl", 0, 0, -499.9), "cl-": ("Cl", -1, 0, -460.2), "ch3f": ("CHHHF", 0, 0, -139.6)}
_WATER27 = {"H2O": ("OHH", 0, 0, -76.4), "H2O2": ("OHHOHH", 0, 0, -152.8), "H3Op": ("OHHH", 1, 0, -76.7),
            "H3OpH2O": ("OHHHOHH", 1, 0, -153.2)}

_S22_RES = """if [ "$TMER" == "" ]
then
  tmer=tmer2++
fi
f=$1
w=$2
#Revised reference values
$tmer 01{,a}/$f    x -1 2    $w 3.133   0   1  nh3
$tmer 02{,a}/$f    x -1 2    $w 4.989  0  1 h2o  # trailing comment
"""
_BH76_RES = """f=$1
w=$2
$tmer  h/$f  n2o/$f  n2ohts/$f  x  -1  -1  1  $w         17.19  # W4
$tmer  oh/$f  n2/$f  n2ohts/$f  x  -1  -1  1  $w         82.14
"""
_BH76_RESRC = """f=$1
w=$2
tmer2++\th/$f\tn2o/$f\toh/$f\tn2/$f\tx\t-1\t-1\t1\t1\t$w\t-64.95\t# W4
tmer2++\tf-/$f\tch3cl/$f\tcl-/$f\tch3f/$f\tx\t-1\t-1\t1\t1\t$w\t-32.16
"""
_WATER27_RES = {
    "v1": "$tmer {H2O2,H2O}/$f            x -1  2  $w 5.01\n"
          "$tmer {H3OpH2O,H3Op,H2O}/$f    x -1 1 1 $w  33.5\n",
    "v2": "$tmer {H1O2,H2O}/$f            x -1  2  $w   4.97\n"
          "$tmer {H2OpH2O,H3Op,H2O}/$f    x -1 1 1 $w  33.74\n",
}


def _archive(version):
    """Bytes of a small GMTKN55-like zip wrapped in one top-level folder."""
    commit = gmtkn55._VERSIONS[version][0]
    top = f"GMTKN55-{commit}/"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(top + ".list.dirs", "BH76\nS22\nWATER27\n")
        z.writestr(top + "LICENSE", "CC BY 4.0\n")
        z.writestr(top + "eval.py", "print('not extracted')\n")
        for folder, species, res in (("S22", _S22, _S22_RES), ("BH76", _BH76, _BH76_RES),
                                     ("WATER27", _WATER27, _WATER27_RES[version])):
            z.writestr(f"{top}{folder}/.res", res)
            z.writestr(f"{top}{folder}/.bib", "@article{x, doi = {10.1039/C7CP04913G}}\n")
            for name, (symbols, charge, unpaired, energy) in species.items():
                symbols = re.findall(r"[A-Z][a-z]?", symbols)
                z.writestr(f"{top}{folder}/{name}/struc.xyz", _xyz(symbols))
                z.writestr(f"{top}{folder}/{name}/PBEh-3c/orca.out", _orca(energy))
                if charge:
                    z.writestr(f"{top}{folder}/{name}/.CHRG", f"{charge}\n")
                if unpaired:
                    z.writestr(f"{top}{folder}/{name}/.UHF", f"{unpaired}\n")
        z.writestr(top + "BH76/.resRC", _BH76_RESRC)
    return buf.getvalue()


@pytest.fixture
def fake_gmtkn55(monkeypatch, tmp_path):
    """Serve synthetic v1 and v2 archives from a local cache, offline."""
    monkeypatch.setattr(gmtkn55, "SUBSETS", ("BH76RC", "BH76", "S22", "WATER27"))
    monkeypatch.setitem(gmtkn55._VERSIONS, "v1", (gmtkn55._VERSIONS["v1"][0], None))
    monkeypatch.setitem(gmtkn55._VERSIONS, "v2", (gmtkn55._VERSIONS["v2"][0], None))

    def fake_download(url, dest, md5=None, quiet=False, chunk=1 << 20):
        version = "v1" if gmtkn55._VERSIONS["v1"][0] in url else "v2"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(_archive(version))
        return dest

    monkeypatch.setattr(gmtkn55, "download_file", fake_download)
    return tmp_path


def test_registered():
    """The builder is listed and the 55 sets are grouped in five categories."""
    assert "gmtkn55" in list_datasets()
    assert len(SUBSETS) == 55 and len(set(SUBSETS)) == 55
    assert [len(v) for v in CATEGORIES.values()] == [18, 9, 7, 12, 9]


def test_parse_reactions_formats():
    """Brace groups, comments, both command spellings and trailing columns parse."""
    rx = parse_reactions(_S22_RES + _BH76_RESRC + "$tmer {h2,h}/$f  x -1 2 $w 109.493\n"
                         "$tmer B_{T,G}/$f x -1 1 $w 0.598\n$tmer 2O{,_A,_B}/$f x -1 1 1 $w 6.047\n")
    assert rx[0] == (["01", "01a"], [-1, 2], 3.133)
    assert rx[2] == (["h", "n2o", "oh", "n2"], [-1, -1, 1, 1], -64.95)
    assert rx[4] == (["h2", "h"], [-1, 2], 109.493)
    assert rx[5] == (["B_T", "B_G"], [-1, 1], 0.598)
    assert rx[6] == (["2O", "2O_A", "2O_B"], [-1, 1, 1], 6.047)
    with pytest.raises(ValueError, match="offset"):
        parse_reactions("$tmer 01{,a}/$f x -1 2 $w 3.1 0.5 1\n")
    with pytest.raises(ValueError, match="length"):
        parse_reactions("$tmer 01{,a}/$f x -1 $w 3.1\n")


def test_load_structures_and_reactions(fake_gmtkn55):
    """Species carry geometry, charge, spin and PBEh-3c energy; reactions carry references."""
    d = load_dataset("gmtkn55", cache_dir=fake_gmtkn55, quiet=True)
    assert set(d) == {"structures", "reactions"}
    by_key = {(s["subset"], s["name"]): s for s in d["structures"]}
    f = by_key[("BH76RC", "f-")]
    assert f["total_charge"] == -1.0 and f["spin_multiplicity"] == 1.0
    assert f["atomic_numbers"].tolist() == [9] and f["pos"].shape == (1, 3)
    assert f["category"] == "small_reactions"
    assert by_key[("BH76", "oh")]["spin_multiplicity"] == 2.0
    assert by_key[("S22", "01")]["pbeh3c_energy"] == pytest.approx(-113.0 * gmtkn55.HARTREE_TO_EV)
    assert ("BH76RC", "ch3cl") in by_key and ("BH76", "ch3cl") not in by_key
    r = [x for x in d["reactions"] if x["subset"] == "S22"]
    assert r[1] == {"subset": "S22", "category": "intermolecular_nci", "index": 2,
                    "species": ["02", "02a"], "stoichiometry": [-1, 2], "reference": 4.989}
    assert len(d["reactions"]) == 8
    assert not (fake_gmtkn55 / "gmtkn55" / "raw" / "v2" / "eval.py").exists()


def test_versions_and_water27_names(fake_gmtkn55):
    """v1 and v2 are cached apart and the v2 WATER27 species names resolve."""
    v1 = load_dataset("gmtkn55", cache_dir=fake_gmtkn55, subsets="water27", version="v1", quiet=True)
    v2 = load_dataset("gmtkn55", cache_dir=fake_gmtkn55, subsets="WATER27", quiet=True)
    assert [r["species"] for r in v1["reactions"]] == [r["species"] for r in v2["reactions"]]
    assert [r["reference"] for r in v1["reactions"]] == [5.01, 33.5]
    assert [r["reference"] for r in v2["reactions"]] == [4.97, 33.74]
    assert (fake_gmtkn55 / "gmtkn55" / "raw" / "v1" / ".extracted").exists()
    assert (fake_gmtkn55 / "gmtkn55" / "raw" / "v2" / ".extracted").exists()


def test_filters(fake_gmtkn55):
    """Element, charge and closed-shell filters drop reactions and their unused species."""
    d = load_dataset("gmtkn55", cache_dir=fake_gmtkn55, subsets=["BH76", "BH76RC"],
                     closed_shell=True, quiet=True)
    assert [r["index"] for r in d["reactions"]] == [2]
    assert {s["name"] for s in d["structures"]} == {"f-", "ch3cl", "cl-", "ch3f"}
    d = load_dataset("gmtkn55", cache_dir=fake_gmtkn55, subsets=["BH76", "BH76RC"],
                     closed_shell=True, max_abs_charge=0, quiet=True)
    assert d["reactions"] == [] and d["structures"] == []
    d = load_dataset("gmtkn55", cache_dir=fake_gmtkn55, elements=["H", "O"], quiet=True)
    assert {r["subset"] for r in d["reactions"]} == {"S22", "WATER27"}
    assert all(set(s["atomic_numbers"].tolist()) <= {1, 8} for s in d["structures"])
    d = load_dataset("gmtkn55", cache_dir=fake_gmtkn55, elements=[1, 8], max_abs_charge=0, quiet=True)
    assert [r["subset"] for r in d["reactions"]] == ["S22", "WATER27"]


def test_split_and_cutoff(fake_gmtkn55):
    """Named splits return lists; cutoff wraps the structures only."""
    structures = load_dataset("gmtkn55", cache_dir=fake_gmtkn55, split="all", quiet=True)
    reactions = load_dataset("gmtkn55", cache_dir=fake_gmtkn55, split="reactions", quiet=True)
    assert isinstance(structures, list) and "pos" in structures[0]
    assert isinstance(reactions, list) and "stoichiometry" in reactions[0]
    d = load_dataset("gmtkn55", cache_dir=fake_gmtkn55, subsets="S22", cutoff=4.0, quiet=True)
    assert type(d["structures"]).__name__ == "AtomicDataset" and len(d["structures"]) == 4
    assert isinstance(d["reactions"], list)
    g = d["structures"][0]
    assert g.total_charge is not None and g.spin_multiplicity is not None


@pytest.mark.parametrize("kwargs,match", [
    ({"split": "train"}, "split must be"),
    ({"version": "v3"}, "version must be"),
    ({"subsets": "S23"}, "unknown GMTKN55 set"),
])
def test_invalid_args(fake_gmtkn55, kwargs, match):
    """Bad arguments raise ValueError with a clear message."""
    with pytest.raises(ValueError, match=match):
        load_dataset("gmtkn55", cache_dir=fake_gmtkn55, quiet=True, **kwargs)


def test_reaction_energies_and_wtmad2(fake_gmtkn55):
    """Reaction energies follow the stoichiometry and WTMAD-2 follows eq. 2 of the paper."""
    d = load_dataset("gmtkn55", cache_dir=fake_gmtkn55, quiet=True)
    energies = {(s["subset"], s["name"]): s["pbeh3c_energy"] for s in d["structures"]}
    pred = reaction_energies(d["reactions"], energies)
    s22 = [r for r in d["reactions"]].index(next(r for r in d["reactions"] if r["subset"] == "S22"))
    assert pred[s22] == pytest.approx((-(-113.0) + 2 * (-56.6)) * HARTREE_TO_KCAL)
    hartree = {k: v / gmtkn55.HARTREE_TO_EV for k, v in energies.items()}
    assert reaction_energies(d["reactions"], hartree, units="hartree") == pytest.approx(pred)

    ref = np.array([r["reference"] for r in d["reactions"]])
    stats = subset_statistics(d["reactions"], ref + 1.0)
    assert list(stats) == ["BH76RC", "BH76", "S22", "WATER27"]
    assert all(v["mad"] == pytest.approx(1.0) and v["md"] == pytest.approx(1.0) for v in stats.values())
    assert stats["S22"]["mean_abs_ref"] == pytest.approx((3.133 + 4.989) / 2)
    # equal MADs: WTMAD-2 = scale * MAD * sum_i N_i / |dE|_i / sum_i N_i
    n = np.array([v["n"] for v in stats.values()])
    e = np.array([v["mean_abs_ref"] for v in stats.values()])
    expected = PAPER_SCALE * (n / e).sum() / n.sum()
    scores = wtmad2(d["reactions"], ref + 1.0)
    assert scores["total"] == pytest.approx(expected)
    assert scores["intermolecular_nci"] == scores["nci"]
    assert set(scores) == {"total", "small_reactions", "barrier_heights", "intermolecular_nci", "nci"}
    assert wtmad2(d["reactions"], ref + 1.0, scale="data")["total"] == pytest.approx(expected * e.mean() / PAPER_SCALE)
    assert wtmad2(d["reactions"], ref)["total"] == 0.0
    with pytest.raises(ValueError, match="one value per reaction"):
        subset_statistics(d["reactions"], ref[:-1])


@pytest.mark.skipif(os.environ.get("XNN_TEST_NETWORK") != "1",
                    reason="set XNN_TEST_NETWORK=1 to download from GitHub")
def test_live_download(tmp_path):
    """End-to-end: the pinned v2 archive gives the published counts."""
    d = load_dataset("gmtkn55", cache_dir=tmp_path, quiet=True)
    assert len(d["reactions"]) == 1495
    assert len({(s["name"], gmtkn55._FOLDER.get(s["subset"], s["subset"])) for s in d["structures"]}) == 2420
    assert all("pbeh3c_energy" in s for s in d["structures"])


@pytest.mark.skipif(os.environ.get("XNN_TEST_NETWORK") != "1",
                    reason="set XNN_TEST_NETWORK=1 to check the dataset card's links")
def test_card_links_resolve():
    """Every DOI and URL named in the module docstring resolves."""
    import json

    doc = gmtkn55.__doc__
    dois = {m.rstrip(").,:;") for m in re.findall(r"10\.\d{4,9}/[^\s>`]+", doc)}
    urls = {m.rstrip(").,:;") for m in re.findall(r"https?://[^\s>`]+", doc)}
    urls |= {gmtkn55._URL.format(commit=c) for c, _ in gmtkn55._VERSIONS.values()}
    assert dois and urls
    for doi in dois:
        with urllib.request.urlopen(f"https://doi.org/api/handles/{doi}", timeout=30) as resp:
            assert json.load(resp)["responseCode"] == 1, doi
    for url in urls:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "xnn"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            assert resp.status == 200, url
