"""Model hub: ``from_pretrained`` / ``save_pretrained`` / ``list_models``.

Everything runs offline. Remote files are served from ``file://`` URLs (the
real downloader, with its MD5 check, handles those like any other URL) and
the Zenodo records API is pointed at local JSON files, so the download,
verification, unpacking, conversion and caching paths are exercised for
real. The MACE foundation tests use a tiny upstream ``mace-torch`` model
built in-process (skipped without ``mace-torch``), plus the cached
MACE-OFF23 small checkpoint when one is on disk. One opt-in test downloads
from Zenodo when ``XNN_TEST_NETWORK=1``.
"""
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from xnn.common.config import from_dict
from xnn.common.data import structure_to_graph
from xnn.common.models import (ForceStressOutput, ModelCard, build_model, from_pretrained,
                               list_models, model_card, register_pretrained, save_pretrained)
from xnn.common.models.hub import (default_model_cache_dir, fetch_model, load_checkpoint,
                                   load_pretrained, registry, zenodo)
from xnn.common.models.hub.cache import is_cache_key, url_slot
from xnn.common.models.hub.checkpoint import config_to_dict

SCHNET = {"name": "schnet", "cutoff": 4.0, "n_interactions": 1, "n_rbf": 6, "n_features": 8}
WATER = {"pos": np.array([[0.0, 0.0, 0.119262], [0.0, 0.763239, -0.477047],
                          [0.0, -0.763239, -0.477047]]),
         "atomic_numbers": np.array([8, 1, 1])}


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """No cache or offline settings from the environment; registry restored.

    Other test modules switch the default dtype to float64 at import, so it
    is pinned here: the tiny models are built in float32, as a training run's.
    """
    for var in ("XNN_MODELS", "XNN_CACHE", "XNN_OFFLINE"):
        monkeypatch.delenv(var, raising=False)
    saved = dict(registry._REGISTRY)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(prev)
    registry._REGISTRY.clear()
    registry._REGISTRY.update(saved)


def _model(seed=0, spec=SCHNET):
    cfg = from_dict({"model": dict(spec)})
    torch.manual_seed(seed)
    return ForceStressOutput(build_model(cfg.model), compute_stress=True), cfg


def _energy(model, structure=WATER, cutoff=4.0):
    g = structure_to_graph(structure, cutoff)
    g.pos = g.pos.to(next(model.parameters()).dtype)
    out = model(g)
    return float(out["energy"]), out["forces"].detach()


def _trainer_checkpoint(path, seed=0):
    model, cfg = _model(seed)
    torch.save({"model": model.state_dict(), "cfg": cfg}, path)
    return model, cfg


def _md5(path):
    import hashlib
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


# registry and cards
def test_registry_lists_foundation_and_xnn_models():
    names = list_models()
    assert {"mace-off23-small", "mace-mp-0-medium", "mace-mh-0", "xnn-mace-argon"} <= set(names)
    assert names == sorted(names)
    rows = {r["name"]: r for r in list_models(details=True)}
    assert rows["mace-off23-small"]["format"] == "mace-torch"
    assert rows["mace-off23-small"]["license"] == "ASL"
    assert rows["mace-mh-0"]["heads"] == 7
    assert rows["mace-mh-1"]["cached"] == "unsupported"
    assert rows["xnn-mace-argon"]["format"] == "xnn"
    assert set(list_models(format="xnn")) >= {"xnn-mace-argon"}
    assert "xnn-mace-argon" not in list_models(format="mace-torch")
    assert "mace-off23-small" in list_models(tag="organic")
    for name in list_models(format="mace-torch"):
        card = model_card(name)
        assert card.url.startswith("https://") and len(card.md5) == 32
        assert card.architecture == "mace" and card.cutoff > 0 and card.species


def test_card_roundtrip_ignores_unknown_keys(tmp_path):
    card = ModelCard(name="m", description="d", species=[1, 8], tags=["a"])
    path = card.save(tmp_path / "card.json")
    data = json.loads(path.read_text())
    assert "doi" not in data                       # unset fields are left out
    data["field_from_the_future"] = 1
    path.write_text(json.dumps(data))
    assert ModelCard.load(path) == card


def test_register_validates_names():
    register_pretrained(name="lab/model-1", doi="10.5281/zenodo.1")
    assert model_card("lab/model-1").doi == "10.5281/zenodo.1"
    for bad in ("../x", "/abs", "a b", "", "x/../y"):
        with pytest.raises(ValueError):
            register_pretrained(name=bad)
    assert is_cache_key("zenodo.1/file.model") and not is_cache_key("https://x")


# cache location
def test_cache_dir_precedence(monkeypatch, tmp_path):
    repo = Path(__file__).resolve().parents[1]
    assert default_model_cache_dir() == repo / "models"
    monkeypatch.setenv("XNN_CACHE", str(tmp_path / "c"))
    assert default_model_cache_dir() == tmp_path / "c" / "models"
    monkeypatch.setenv("XNN_MODELS", str(tmp_path / "m"))
    assert default_model_cache_dir() == tmp_path / "m"


# portable directories
def test_save_pretrained_writes_a_portable_directory(tmp_path):
    model, cfg = _model()
    out = save_pretrained(model, tmp_path / "tiny", config=cfg, description="tiny",
                          license="MIT")
    assert sorted(p.name for p in out.iterdir()) == ["card.json", "config.yaml", "model.pt"]
    card = ModelCard.load(out / "card.json")
    assert (card.name, card.architecture, card.cutoff, card.dtype) == \
        ("tiny", "schnet", 4.0, "float32")
    assert card.files == {f: _md5(out / f) for f in ("config.yaml", "model.pt")}
    # no pickled objects: the weights load with weights_only=True
    torch.load(out / "model.pt", weights_only=True)
    assert "/" not in json.dumps(card.to_dict()).replace("\\/", "")  # no paths
    # a bare model (no force wrapper) saves the same weights
    bare = save_pretrained(model.model, tmp_path / "bare", config=cfg.model)
    assert _md5(bare / "model.pt") != ""   # written
    a = load_checkpoint(out).state_dict
    b = load_checkpoint(bare).state_dict
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_from_pretrained_reproduces_the_model(tmp_path):
    model, cfg = _model()
    out = save_pretrained(model, tmp_path / "tiny", config=cfg)
    loaded = from_pretrained(out)
    assert not loaded.training
    e0, f0 = _energy(model)
    e1, f1 = _energy(loaded)
    assert e1 == pytest.approx(e0, abs=1e-6) and torch.allclose(f0, f1, atol=1e-6)
    bare = from_pretrained(out, wrap=False)
    assert type(bare).__name__ == "SchNet"
    f64 = from_pretrained(out, dtype="float64")
    assert next(f64.parameters()).dtype == torch.float64


def test_trainer_checkpoint_loads_and_packs(tmp_path):
    model, cfg = _trainer_checkpoint(tmp_path / "best.pt")
    e0, _ = _energy(model)
    assert _energy(from_pretrained(tmp_path / "best.pt"))[0] == pytest.approx(e0, abs=1e-6)
    out = save_pretrained(tmp_path / "best.pt", tmp_path / "packed", license="MIT")
    assert ModelCard.load(out / "card.json").source == "best.pt"
    assert _energy(from_pretrained(out))[0] == pytest.approx(e0, abs=1e-6)
    # the packed config reads back as the same model section
    assert config_to_dict(load_checkpoint(out).config)["model"] == config_to_dict(cfg)["model"]


def test_bare_state_dict_needs_a_config(tmp_path):
    model, cfg = _model()
    torch.save(model.state_dict(), tmp_path / "sd.pt")
    ck = load_checkpoint(tmp_path / "sd.pt")
    assert ck.config is None and ck.state_dict.keys() == model.state_dict().keys()
    with pytest.raises(ValueError, match="embeds no config"):
        from_pretrained(tmp_path / "sd.pt")
    with pytest.raises(ValueError, match="needs the model's config"):
        save_pretrained(tmp_path / "sd.pt", tmp_path / "out")
    save_pretrained(tmp_path / "sd.pt", tmp_path / "out", config=cfg)   # with one it works


def test_file_references_are_flagged(tmp_path, caplog):
    import logging

    model, cfg = _model()
    (tmp_path / "topology.json").write_text("{}")
    cfg.model.extra["topology"] = str(tmp_path / "topology.json")
    with caplog.at_level(logging.WARNING):
        save_pretrained(model, tmp_path / "m", config=cfg, verify=False)
    assert "refers to files outside" in caplog.text and "topology.json" in caplog.text


def test_save_pretrained_rejects_a_mismatched_config(tmp_path):
    model, cfg = _model()
    wrong = from_dict({"model": {**SCHNET, "n_features": 16}})
    with pytest.raises(ValueError, match="does not rebuild"):
        save_pretrained(model, tmp_path / "x", config=wrong)
    assert not (tmp_path / "x" / "card.json").exists()


def test_config_values_are_written_as_plain_yaml(tmp_path):
    model, cfg = _model()
    cfg.model.extra["species"] = np.array([1, 8])
    cfg.model.extra["atomic_energies"] = torch.tensor([-13.6, -2041.0], dtype=torch.float64)
    cfg.model.extra["tags"] = ("a", "b")
    out = save_pretrained(model, tmp_path / "m", config=cfg, verify=False)
    extra = load_checkpoint(out).config.model.extra
    assert extra["species"] == [1, 8] and extra["tags"] == ["a", "b"]
    assert extra["atomic_energies"] == [-13.6, -2041.0]       # exact float round trip
    cfg.model.extra["bad"] = object()
    with pytest.raises(TypeError, match="model.extra.bad"):
        save_pretrained(model, tmp_path / "m2", config=cfg, verify=False)


def test_copied_cache_is_portable(tmp_path):
    """A cache (or one model of it) copied elsewhere loads there by name."""
    model, cfg = _model()
    save_pretrained(model, tmp_path / "siteA" / "my-model", config=cfg)
    shutil.copytree(tmp_path / "siteA", tmp_path / "elsewhere" / "siteB")
    shutil.rmtree(tmp_path / "siteA")
    cache = tmp_path / "elsewhere" / "siteB"
    loaded = from_pretrained("my-model", cache_dir=cache, local_files_only=True)
    assert _energy(loaded)[0] == pytest.approx(_energy(model)[0], abs=1e-6)
    rows = {r["name"]: r for r in list_models(cache, details=True)}
    assert rows["my-model"]["cached"] == "ready"
    assert "my-model" in list_models(cache, cached_only=True)
    assert model_card("my-model", cache).architecture == "schnet"


# downloads (file:// URLs stand in for remote hosts)
def test_registered_url_download_cache_and_md5(tmp_path):
    _trainer_checkpoint(tmp_path / "best.pt")
    url = (tmp_path / "best.pt").as_uri()
    register_pretrained(name="tiny-url", url=url, md5=_md5(tmp_path / "best.pt"),
                        description="tiny", license="MIT")
    cache = tmp_path / "cache"
    assert list_models(cache, details=True)[-1]["cached"] in ("", "ready")
    path = fetch_model("tiny-url", cache_dir=cache, quiet=True)
    assert path == cache / "tiny-url"
    card = ModelCard.load(path / "card.json")
    assert (card.description, card.license, card.source) == ("tiny", "MIT", url)
    assert not (path / ".download").exists() and not (path / "raw").exists()
    # cached: works without the source
    (tmp_path / "best.pt").unlink()
    assert fetch_model("tiny-url", cache_dir=cache, local_files_only=True) == path
    assert {r["name"]: r for r in list_models(cache, details=True)}["tiny-url"]["cached"] == "ready"

    _trainer_checkpoint(tmp_path / "other.pt", seed=3)
    register_pretrained(name="tiny-bad", url=(tmp_path / "other.pt").as_uri(), md5="0" * 32)
    with pytest.raises(ValueError, match="MD5 mismatch"):
        fetch_model("tiny-bad", cache_dir=cache, quiet=True)
    assert not (cache / "tiny-bad" / "card.json").exists()


def test_offline_and_unknown_sources(tmp_path, monkeypatch):
    _trainer_checkpoint(tmp_path / "best.pt")
    register_pretrained(name="tiny-url", url=(tmp_path / "best.pt").as_uri())
    with pytest.raises(FileNotFoundError, match="network access is disabled"):
        fetch_model("tiny-url", cache_dir=tmp_path / "c", local_files_only=True)
    monkeypatch.setenv("XNN_OFFLINE", "1")
    with pytest.raises(FileNotFoundError, match="network access is disabled"):
        fetch_model("tiny-url", cache_dir=tmp_path / "c")
    monkeypatch.delenv("XNN_OFFLINE")
    register_pretrained(name="unpublished", architecture="mace")
    with pytest.raises(FileNotFoundError, match="no download source"):
        fetch_model("unpublished", cache_dir=tmp_path / "c")
    with pytest.raises(FileNotFoundError, match="list_models"):
        fetch_model("no-such-model", cache_dir=tmp_path / "c")
    with pytest.raises(ValueError, match="only Zenodo DOIs"):
        fetch_model("doi:10.6084/m9.figshare.10047041", cache_dir=tmp_path / "c")
    with pytest.raises(NotImplementedError, match="cannot be loaded"):
        fetch_model("mace-mh-1", cache_dir=tmp_path / "c")


def test_head_selection_is_checked_before_downloading(tmp_path):
    register_pretrained(name="two-heads", url="https://example.invalid/x.model",
                        format="mace-torch", heads=["a", "b"])
    with pytest.raises(ValueError, match="pass head="):
        fetch_model("two-heads", cache_dir=tmp_path)
    with pytest.raises(ValueError, match="unknown head"):
        fetch_model("two-heads", head="c", cache_dir=tmp_path)
    register_pretrained(name="one-head", url="https://example.invalid/y.model",
                        format="mace-torch", heads=["only"])
    with pytest.raises(ValueError, match="single head"):
        fetch_model("one-head", head="other", cache_dir=tmp_path)


def test_archive_of_a_portable_directory(tmp_path):
    model, cfg = _model()
    save_pretrained(model, tmp_path / "upload" / "lab-model", config=cfg,
                    description="the uploader's description", citation="Lab (2026)")
    archive = shutil.make_archive(str(tmp_path / "lab-model"), "zip",
                                  root_dir=tmp_path / "upload", base_dir="lab-model")
    register_pretrained(name="lab-model", url=Path(archive).as_uri(),
                        description="registry text", license="CC-BY-4.0")
    path = fetch_model("lab-model", cache_dir=tmp_path / "cache", quiet=True)
    card = ModelCard.load(path / "card.json")
    # the uploader's card wins, the registry fills in what it lacks
    assert card.description == "the uploader's description" and card.citation == "Lab (2026)"
    assert card.license == "CC-BY-4.0" and card.name == "lab-model"
    assert card.source == Path(archive).as_uri()          # where this copy came from
    assert _energy(from_pretrained(path))[0] == pytest.approx(_energy(model)[0], abs=1e-6)


def test_tar_archive_refuses_path_traversal(tmp_path):
    import tarfile
    if not hasattr(tarfile, "data_filter"):
        pytest.skip("tarfile data filter unavailable")
    evil = tmp_path / "evil.tar.gz"
    payload = tmp_path / "payload.txt"
    payload.write_text("x")
    with tarfile.open(evil, "w:gz") as t:
        t.add(payload, arcname="../escaped.txt")
    register_pretrained(name="evil", url=evil.as_uri())
    with pytest.raises(Exception):
        fetch_model("evil", cache_dir=tmp_path / "cache", quiet=True)
    assert not (tmp_path / "escaped.txt").exists()


# Zenodo
def _fake_zenodo(tmp_path, monkeypatch, recid, files, title="A model record"):
    """Serve a record JSON with file:// links, as the records API would."""
    api = tmp_path / "api"
    api.mkdir(exist_ok=True)
    record = {"id": int(recid), "doi": f"10.5281/zenodo.{recid}",
              "metadata": {"title": title, "license": {"id": "cc-by-4.0"},
                           "publication_date": "2026-05-01",
                           "creators": [{"name": "Doe, Jane"}, {"name": "Roe, R."}]},
              "files": [{"key": Path(f).name, "size": Path(f).stat().st_size,
                         "checksum": f"md5:{_md5(f)}",
                         "links": {"self": Path(f).as_uri()}} for f in files]}
    (api / f"{recid}.json").write_text(json.dumps(record))
    monkeypatch.setattr(zenodo, "API", (api.as_uri() + "/{recid}.json"))
    return record


@pytest.mark.parametrize("ref", [
    "doi:10.5281/zenodo.4242", "10.5281/zenodo.4242", "https://doi.org/10.5281/zenodo.4242",
    "https://zenodo.org/records/4242", "https://zenodo.org/record/4242/", "zenodo:4242",
    "https://zenodo.org/doi/10.5281/zenodo.4242"])
def test_zenodo_reference_spellings(ref):
    assert zenodo.parse(ref) == ("4242", None)


def test_zenodo_file_links_select_the_file():
    assert zenodo.parse("https://zenodo.org/records/4242/files/a%20b.model?download=1") == \
        ("4242", "a b.model")
    assert zenodo.parse("https://zenodo.org/api/records/4242/files/m.pt/content") == ("4242", "m.pt")
    assert zenodo.parse("https://example.org/records/4242") is None
    assert zenodo.is_doi("10.6084/m9.figshare.1") and not zenodo.is_doi("mace-off23-small")


def test_zenodo_record_with_several_models(tmp_path, monkeypatch):
    _trainer_checkpoint(tmp_path / "a.pt", seed=1)
    model_b, _ = _trainer_checkpoint(tmp_path / "b.pt", seed=2)
    (tmp_path / "README.md").write_text("hello")
    _fake_zenodo(tmp_path, monkeypatch, "4242",
                 [tmp_path / "a.pt", tmp_path / "b.pt", tmp_path / "README.md"])
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="pass filename="):
        fetch_model("doi:10.5281/zenodo.4242", cache_dir=cache)
    with pytest.raises(FileNotFoundError, match="no file"):
        fetch_model("doi:10.5281/zenodo.4242", filename="c.pt", cache_dir=cache)
    loaded = load_pretrained("doi:10.5281/zenodo.4242", filename="b.pt", cache_dir=cache,
                             quiet=True)
    assert loaded.path == cache / "zenodo.4242" / "b.pt"
    assert loaded.card.license == "cc-by-4.0" and "Doe, Jane and Roe, R." in loaded.card.citation
    assert loaded.card.doi == "10.5281/zenodo.4242"
    assert _energy(loaded.model)[0] == pytest.approx(_energy(model_b)[0], abs=1e-6)
    assert (cache / "zenodo.4242" / "record.json").is_file()
    # offline, by record link or by cache key
    (tmp_path / "api" / "4242.json").unlink()
    link = "https://zenodo.org/records/4242/files/b.pt"
    assert fetch_model(link, cache_dir=cache, local_files_only=True) == loaded.path
    assert fetch_model("zenodo.4242/b.pt", cache_dir=cache, local_files_only=True) == loaded.path
    assert "zenodo.4242/b.pt" in list_models(cache, cached_only=True)


def test_zenodo_record_of_a_portable_upload(tmp_path, monkeypatch):
    model, cfg = _model(seed=5)
    up = save_pretrained(model, tmp_path / "upload", config=cfg, description="uploaded")
    _fake_zenodo(tmp_path, monkeypatch, "777",
                 [up / "card.json", up / "config.yaml", up / "model.pt"])
    register_pretrained(name="lab-zenodo", doi="10.5281/zenodo.777", license="MIT")
    for source in ("lab-zenodo", "doi:10.5281/zenodo.777"):
        path = fetch_model(source, cache_dir=tmp_path / "cache", quiet=True)
        card = ModelCard.load(path / "card.json")
        assert card.description == "uploaded"
        assert _energy(from_pretrained(path))[0] == pytest.approx(_energy(model)[0], abs=1e-6)
    assert (tmp_path / "cache" / "lab-zenodo").is_dir() and (tmp_path / "cache" / "zenodo.777").is_dir()


def test_plain_url_cache_key(tmp_path, monkeypatch):
    assert url_slot("https://h.org/m/best.pt?x=1").startswith("url/best.pt-")
    assert url_slot("https://a.org/best.pt") != url_slot("https://b.org/best.pt")


# consumers of the shared loader
def test_engine_calculator_and_classmethod_take_names(tmp_path):
    pytest.importorskip("ase")
    from ase import Atoms

    from xnn.gnn.models.schnet import SchNet
    from xnn.common.deploy import MDIEngine, XNNCalculator
    from xnn.gnn.models.mace import MACE

    model, cfg = _model()
    save_pretrained(model, tmp_path / "tiny-schnet", config=cfg)
    e0, _ = _energy(model)
    engine = MDIEngine.from_checkpoint("tiny-schnet", cache_dir=tmp_path)
    assert engine.cutoff == 4.0 and engine.model.compute_stress
    calc = XNNCalculator.from_pretrained("tiny-schnet", cache_dir=tmp_path)
    atoms = Atoms(numbers=WATER["atomic_numbers"], positions=WATER["pos"])
    atoms.calc = calc
    assert atoms.get_potential_energy() == pytest.approx(e0, abs=1e-5)
    # a float64 model in a float32 session: the calculator follows the model
    calc64 = XNNCalculator.from_pretrained("tiny-schnet", cache_dir=tmp_path, dtype="float64")
    assert torch.get_default_dtype() == torch.float32 and calc64.dtype == torch.float64
    atoms.calc = calc64
    assert atoms.get_potential_energy() == pytest.approx(e0, abs=1e-5)
    assert isinstance(SchNet.from_pretrained("tiny-schnet", cache_dir=tmp_path), SchNet)
    with pytest.raises(TypeError, match="not a MACE"):
        MACE.from_pretrained("tiny-schnet", cache_dir=tmp_path)


def test_trainer_save_pretrained(tmp_path):
    from xnn.common.config import Config
    from xnn.common.data import AtomicDataset
    from xnn.common.train import Trainer

    cfg = Config()
    cfg.model = from_dict({"model": SCHNET}).model
    cfg.data.cutoff = 4.0
    cfg.data.batch_size = 2
    cfg.optim.epochs = 1
    cfg.device = "cpu"
    cfg.output_dir = str(tmp_path / "run")
    rng = np.random.default_rng(0)
    data = [{**WATER, "pos": WATER["pos"] + rng.normal(0, 0.02, (3, 3)),
             "energy": float(rng.normal()), "forces": rng.normal(0, 0.1, (3, 3))}
            for _ in range(6)]
    trainer = Trainer(cfg, AtomicDataset(data, 4.0))
    trainer.fit()
    out = trainer.save_pretrained(tmp_path / "trained", description="toy", license="MIT")
    loaded = load_pretrained(out)
    assert loaded.card.description == "toy" and loaded.config.optim.epochs == 1
    ref = trainer.module.to("cpu").eval()
    assert _energy(loaded.model)[0] == pytest.approx(_energy(ref)[0], abs=1e-5)


def test_benchmark_and_export_resolve_hub_sources(tmp_path):
    model, cfg = _model()
    save_pretrained(model, tmp_path / "cache" / "tiny-schnet", config=cfg)
    out = tmp_path / "deployed.pt"
    env = {**os.environ, "XNN_MODELS": str(tmp_path / "cache")}
    run = subprocess.run([sys.executable, "-m", "xnn.common.cli.main", "export", "--ckpt",
                          "tiny-schnet", "--out", str(out)], capture_output=True, text=True,
                         env=env)
    assert run.returncode == 0, run.stderr
    assert torch.jit.load(str(out)) is not None


def test_cli_list_info_pull_pack(tmp_path, capsys):
    from xnn.common.models.hub.cli import main

    _trainer_checkpoint(tmp_path / "best.pt")
    main(["pack", str(tmp_path / "best.pt"), str(tmp_path / "cache" / "xnn-mace-argon"),
          "--name", "xnn-mace-argon", "--archive"])
    card = ModelCard.load(tmp_path / "cache" / "xnn-mace-argon" / "card.json")
    # a registered name lends its description and license
    assert card.license == "MIT" and "argon" in card.description
    assert zipfile.is_zipfile(tmp_path / "cache" / "xnn-mace-argon.zip")
    capsys.readouterr()
    main(["list", "--cached", "--format", "xnn", "--cache-dir", str(tmp_path / "cache")])
    table = capsys.readouterr().out
    assert "xnn-mace-argon" in table and "ready" in table
    main(["info", "mace-off23-small"])
    assert json.loads(capsys.readouterr().out)["license"] == "ASL"
    register_pretrained(name="tiny-url", url=(tmp_path / "best.pt").as_uri())
    main(["pull", "tiny-url", "--cache-dir", str(tmp_path / "c2")])
    assert capsys.readouterr().out.strip() == str(tmp_path / "c2" / "tiny-url")


# the mace-torch format: MACE foundation models under the same API
def _tiny_upstream_mace(heads=None):
    """A small upstream ScaleShiftMACE (Agnesi + ZBL + density blocks)."""
    import mace.modules as mm
    from e3nn import o3

    n = len(heads) if heads else 1
    e0 = np.array([[0.5, -1.3, -2.1], [0.1, -0.4, -0.9]])[:n]
    up = mm.ScaleShiftMACE(
        atomic_inter_scale=0.83 if n == 1 else np.array([0.8, 1.1]),
        atomic_inter_shift=0.11 if n == 1 else np.array([0.1, -0.2]),
        r_max=4.0, num_bessel=6, num_polynomial_cutoff=5, max_ell=2,
        interaction_cls=mm.RealAgnosticDensityResidualInteractionBlock,
        interaction_cls_first=mm.RealAgnosticDensityInteractionBlock, num_interactions=2,
        num_elements=3, hidden_irreps=o3.Irreps("8x0e+8x1o"),
        MLP_irreps=o3.Irreps(f"{16 * n}x0e"), atomic_energies=e0.squeeze(0) if n == 1 else e0,
        avg_num_neighbors=3.1, atomic_numbers=[1, 6, 8], correlation=3,
        gate=torch.nn.functional.silu, radial_MLP=[16, 16], radial_type="bessel",
        distance_transform="Agnesi", pair_repulsion=True, heads=heads,
        use_reduced_cg=False, apply_cutoff=True).double()
    torch.manual_seed(11)
    with torch.no_grad():
        for p in up.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return up


ETHANOL_ISH = {"pos": np.array([[0.0, 0.0, 0.0], [1.4, 0.1, 0.0], [2.0, 1.3, 0.2],
                                [-0.5, 0.9, 0.3], [-0.4, -0.9, 0.4], [1.8, -0.7, -0.6]]),
               "atomic_numbers": np.array([6, 6, 8, 1, 1, 1])}


def test_mace_torch_model_converts_once_and_caches(tmp_path):
    pytest.importorskip("mace")
    pytest.importorskip("e3nn")
    from xnn.gnn.models.mace_foundation import from_mace_torch

    up = _tiny_upstream_mace()
    torch.save(up, tmp_path / "tiny.model")
    register_pretrained(name="tiny-foundation", url=(tmp_path / "tiny.model").as_uri(),
                        md5=_md5(tmp_path / "tiny.model"), format="mace-torch",
                        architecture="mace", license="MIT")
    cache = tmp_path / "cache"
    direct = ForceStressOutput(from_mace_torch(up))
    first = from_pretrained("tiny-foundation", cache_dir=cache, quiet=True)
    slot = cache / "tiny-foundation"
    assert sorted(p.name for p in slot.iterdir()) == ["card.json", "config.yaml", "model.pt"]
    assert ModelCard.load(slot / "card.json").format == "xnn"     # stored converted
    again = from_pretrained("tiny-foundation", cache_dir=cache, local_files_only=True)
    e0, f0 = _energy(direct, ETHANOL_ISH)
    for m in (first, again):
        e, f = _energy(m, ETHANOL_ISH)
        assert abs(e - e0) < 1e-12 and (f - f0).abs().max() < 1e-12

    # the cached copy loads without mace-torch
    code = ("import sys; sys.modules['mace'] = None\n"
            "from xnn.common.models import from_pretrained\n"
            f"m = from_pretrained('tiny-foundation', cache_dir={str(cache)!r}, "
            "local_files_only=True)\nprint(type(m.model).__name__)")
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert run.returncode == 0 and run.stdout.strip() == "MACE", run.stderr


def test_mace_torch_multihead_caches_each_head(tmp_path):
    pytest.importorskip("mace")
    from xnn.gnn.models.mace_foundation import from_mace_torch

    up = _tiny_upstream_mace(heads=["ha", "hb"])
    torch.save(up, tmp_path / "mh.model")
    register_pretrained(name="tiny-mh", url=(tmp_path / "mh.model").as_uri(),
                        format="mace-torch", heads=["ha", "hb"])
    cache = tmp_path / "cache"
    for head in ("ha", "hb"):
        m = from_pretrained("tiny-mh", head=head, cache_dir=cache, quiet=True)
        e0, _ = _energy(ForceStressOutput(from_mace_torch(up, head=head)), ETHANOL_ISH)
        assert abs(_energy(m, ETHANOL_ISH)[0] - e0) < 1e-12
    assert (cache / "tiny-mh" / "raw" / "mh.model").is_file()     # kept for other heads
    assert {r["name"]: r for r in list_models(cache, details=True)}["tiny-mh"]["cached"] \
        == "ready: ha, hb"


def test_mace_torch_from_a_plain_url_and_a_local_file(tmp_path):
    pytest.importorskip("mace")
    from xnn.gnn.models.mace import MACE

    up = _tiny_upstream_mace()
    torch.save(up, tmp_path / "tiny.model")
    local = from_pretrained(tmp_path / "tiny.model", wrap=False)    # converted in memory
    assert isinstance(local, MACE)
    assert not any((tmp_path).glob("**/card.json"))
    # MACE.from_foundation goes through the same hub cache
    via = MACE.from_foundation(tmp_path / "tiny.model")
    assert isinstance(via, MACE)


def test_foundation_alias_through_the_hub(tmp_path):
    """A registered foundation alias converts once into the given cache."""
    pytest.importorskip("mace")
    from xnn.gnn.models.mace import MACE
    from xnn.gnn.models.mace_foundation import _legacy_cached, from_mace_torch, load_foundation

    cached = _legacy_cached(model_card("mace-off23-small"))
    if cached is None:
        pytest.skip("no cached MACE-OFF23 small checkpoint (tests never download)")
    model = MACE.from_foundation("mace-off23-small", cache_dir=tmp_path)
    assert (tmp_path / "mace-off23-small" / "card.json").is_file()
    direct = from_mace_torch(load_foundation(cached))
    e0, f0 = _energy(ForceStressOutput(direct), ETHANOL_ISH, cutoff=4.5)
    e1, f1 = _energy(ForceStressOutput(model), ETHANOL_ISH, cutoff=4.5)
    assert abs(e1 - e0) < 1e-10 and (f1 - f0).abs().max() < 1e-10
    cfg = from_dict({"model": {"name": "mace", "cutoff": 4.5,
                               "foundation": "mace-off23-small"}})
    assert isinstance(build_model(cfg.model), MACE)


@pytest.mark.skipif(os.environ.get("XNN_TEST_NETWORK") != "1",
                    reason="network test; set XNN_TEST_NETWORK=1")
def test_network_zenodo_mace_model(tmp_path):
    pytest.importorskip("mace")
    loaded = load_pretrained("doi:10.5281/zenodo.18957344",
                             filename="mace_csfapbbri_al_5_1_stagetwo.model",
                             cache_dir=tmp_path)
    assert loaded.card.license == "cc-by-4.0" and loaded.cutoff == 5.0
    assert set(loaded.card.species) == {1, 6, 7, 35, 53, 55, 82}
