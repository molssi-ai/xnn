"""Benchmarking feature: config parsing, metrics, report writers, scoring driver.

Benchmarking scores *pre-trained* models, so the end-to-end tests build a
checkpoint directly (a wrapped model's ``state_dict``) rather than training.
"""
import csv
import json

import numpy as np
import pytest
import torch

from xnns.common.benchmark import (
    BenchmarkConfig, from_dict, run_benchmark, Benchmark,
    register_metric, get_metric, available_metrics, score,
    register_writer, available_writers, write_all, format_table,
)
from xnns.common.benchmark.metrics import mae, mse, rmse, collect_predictions
from xnns.common.benchmark.report import columns
from xnns.common.benchmark.energy import (
    build_e0_lookup, fit_atomic_energies, dataset_structures,
)
from xnns.common.config import from_dict as cfg_from_dict
from xnns.common.models import build_model, ForceStressOutput


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _write_dataset(path, n=12):
    """Write a tiny extxyz file with per-frame energy and forces."""
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import write
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(n):
        at = Atoms(numbers=[1, 6, 8, 1], positions=rng.uniform(0, 4, (4, 3)))
        at.calc = SinglePointCalculator(
            at, energy=float(rng.normal()), forces=rng.normal(0, 1, (4, 3)))
        frames.append(at)
    write(str(path), frames, format="extxyz")
    return str(path)


def _model_spec(label=None, **over):
    spec = {"name": "schnet", "n_features": 16, "n_interactions": 1,
            "n_rbf": 8, "cutoff": 4.0}
    if label:
        spec["label"] = label
    spec.update(over)
    return spec


def _make_checkpoint(path, spec=None):
    """Build the model for ``spec`` and save its (random) weights as a checkpoint.

    Mirrors what ``xnns train`` writes (``{"model": state_dict}``) without the
    training cost -- benchmarking only needs weights to load and score.
    """
    spec = spec or _model_spec()
    cfg = cfg_from_dict({"model": {k: v for k, v in spec.items()
                                   if k != "label"}})
    wrapped = ForceStressOutput(build_model(cfg.model))
    torch.save({"model": wrapped.state_dict(), "cfg": cfg}, str(path))
    return str(path)


def _bench_dict(data_path, models, **over):
    d = {
        "models": models,
        "metrics": ["mae", "rmse"],
        "targets": ["energy", "forces"],
        "data": {"test_path": data_path, "batch_size": 4},
        "device": "cpu",
        "seed": 0,
    }
    d.update(over)
    return d


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_models_accept_strings_and_mappings():
    cfg = from_dict({"models": ["schnet", {"name": "mace", "cutoff": 5.0}]})
    assert [m.label for m in cfg.models] == ["schnet", "mace"]
    assert cfg.models[1].model["cutoff"] == 5.0


def test_single_string_model_is_wrapped_to_list():
    cfg = from_dict({"models": "schnet"})
    assert [m.label for m in cfg.models] == ["schnet"]


def test_duplicate_labels_are_disambiguated():
    cfg = from_dict({"models": ["schnet", "schnet", {"name": "schnet"}]})
    assert [m.label for m in cfg.models] == ["schnet", "schnet#2", "schnet#3"]


def test_entry_keys_split_from_model_section():
    cfg = from_dict({"models": [{"name": "mace", "checkpoint": "a/best.pt",
                                 "label": "m1"}]})
    e = cfg.models[0]
    assert e.label == "m1"
    assert e.checkpoint == "a/best.pt"
    assert e.model == {"name": "mace"}        # entry keys stripped out
    # benchmarking-only: no training-oriented fields on the entry
    assert not hasattr(e, "optim") and not hasattr(e, "output_dir")


def test_model_config_from_file(tmp_path):
    import yaml
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump({"name": "schnet", "cutoff": 6.0,
                                 "n_features": 8}))
    # inline keys override the file
    cfg = from_dict({"models": [{"config": str(p), "n_features": 32,
                                 "checkpoint": "c.pt"}]})
    assert cfg.models[0].model["cutoff"] == 6.0
    assert cfg.models[0].model["n_features"] == 32
    assert cfg.models[0].checkpoint == "c.pt"


def test_to_config_folds_model_and_shared_data():
    cfg = from_dict({
        "models": [{"name": "schnet", "cutoff": 5.0, "checkpoint": "c.pt"}],
        "data": {"batch_size": 8},
        "seed": 42, "device": "cpu",
    })
    rc = cfg.models[0].to_config(cfg)
    assert rc.model.name == "schnet"
    assert rc.model.cutoff == 5.0
    assert rc.data.cutoff == 5.0           # kept in lockstep by Config
    assert rc.seed == 42 and rc.device == "cpu"


def test_config_has_no_phases_or_optim():
    cfg = from_dict({"models": ["schnet"]})
    assert not hasattr(cfg, "phases")
    assert not hasattr(cfg, "optim")


def test_scalar_fields_accept_scalar_or_list():
    cfg = from_dict({"metrics": "mae", "targets": "energy"})
    assert cfg.metrics == ["mae"]
    assert cfg.targets == ["energy"]


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_builtin_metrics_values():
    p = torch.tensor([1.0, 2.0, 3.0])
    t = torch.tensor([1.0, 4.0, 3.0])
    assert mae(p, t) == pytest.approx(2.0 / 3.0)
    assert mse(p, t) == pytest.approx(4.0 / 3.0)
    assert rmse(p, t) == pytest.approx((4.0 / 3.0) ** 0.5)


def test_metric_registry_and_score():
    for m in ("mae", "mse", "rmse"):
        assert m in available_metrics()
    assert get_metric("MAE") is mae   # case-insensitive
    pairs = {"energy": (torch.zeros(3), torch.ones(3))}
    out = score(pairs, ["mae", "rmse"])
    assert out == {"energy_mae": pytest.approx(1.0),
                   "energy_rmse": pytest.approx(1.0)}


def test_custom_metric_registration_via_decorator():
    @register_metric("halfmae")
    def _half(pred, target):
        return 0.5 * float((pred - target).abs().mean())
    assert "halfmae" in available_metrics()
    out = score({"forces": (torch.zeros(2), torch.full((2,), 4.0))}, ["halfmae"])
    assert out["forces_halfmae"] == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# atomization / interaction energy
# --------------------------------------------------------------------------- #
def test_e0_lookup_from_dict():
    e0 = build_e0_lookup({1: -13.6, 8: -2042.0})
    assert e0[1] == pytest.approx(-13.6)
    assert e0[8] == pytest.approx(-2042.0)
    assert e0[6] == 0.0            # unspecified elements stay zero


def test_e0_lookup_from_symbols():
    e0 = build_e0_lookup({"H": -13.6, "O": -2042.0})
    assert e0[1] == pytest.approx(-13.6) and e0[8] == pytest.approx(-2042.0)


def test_e0_lookup_from_list_needs_species():
    e0 = build_e0_lookup([-13.6, -1029.0, -2042.0], species=[1, 6, 8])
    assert e0[6] == pytest.approx(-1029.0)
    with pytest.raises(ValueError, match="species"):
        build_e0_lookup([-13.6, -1029.0])         # no species alignment


def test_e0_lookup_none_disables():
    assert build_e0_lookup(None) is None


def test_e0_average_requires_dataset():
    with pytest.raises(ValueError, match="dataset"):
        build_e0_lookup("average")


def test_fit_atomic_energies_recovers_known_e0s():
    # Energies built exactly as sum of known per-element E0s -> lstsq recovers them.
    rng = np.random.default_rng(1)
    true = {1: -13.6, 6: -1029.0, 8: -2042.0}
    structs = []
    for _ in range(20):
        counts = {1: int(rng.integers(1, 5)), 6: int(rng.integers(1, 5)),
                  8: int(rng.integers(1, 5))}
        zs = [z for z, c in counts.items() for _ in range(c)]
        e = sum(true[z] for z in zs)
        structs.append({"atomic_numbers": zs, "energy": e})
    species, values = fit_atomic_energies(structs)
    assert species == [1, 6, 8]
    for z, v in zip(species, values):
        # float32 lstsq on ~2000-magnitude E0s -> ~1e-4 relative precision
        assert v == pytest.approx(true[z], rel=1e-3)


def test_dataset_structures_reads_atomicdataset(tmp_path):
    from xnns.common.data import AtomicDataset
    ds = AtomicDataset.from_file(_write_dataset(tmp_path / "d.extxyz"), 4.0)
    structs = dataset_structures(ds)
    assert len(structs) == 12
    assert all("atomic_numbers" in s for s in structs)


def test_atomization_leaves_difference_metrics_invariant():
    """Subtracting the same E0 offset from pred and ref cannot change MAE/RMSE."""
    from xnns.common.data import AtomicDataset, collate
    from torch.utils.data import DataLoader

    structs = [{"pos": np.random.default_rng(i).uniform(0, 4, (4, 3)),
                "atomic_numbers": [1, 6, 8, 1],
                "energy": float(i), "forces": np.zeros((4, 3))}
               for i in range(6)]
    ds = AtomicDataset(structs, 4.0)
    loader = DataLoader(ds, batch_size=3, collate_fn=collate)
    cfg = cfg_from_dict({"model": {"name": "schnet", "n_features": 16,
                                   "n_interactions": 1, "cutoff": 4.0}})
    model = ForceStressOutput(build_model(cfg.model))
    dev = torch.device("cpu")

    e0 = build_e0_lookup({1: -13.6, 6: -1029.0, 8: -2042.0})
    plain = score(collect_predictions(model, loader, dev, ["energy"],
                                      atomic_energies=None), ["mae", "rmse"])
    atomz = score(collect_predictions(model, loader, dev, ["energy"],
                                      atomic_energies=e0), ["mae", "rmse"])
    # Exact in real arithmetic; float32 cancellation of the large E0 offset
    # leaves a tiny residual, so compare with a relative tolerance.
    assert plain["energy_mae"] == pytest.approx(atomz["energy_mae"], rel=1e-4)
    assert plain["energy_rmse"] == pytest.approx(atomz["energy_rmse"], rel=1e-4)


def test_atomization_average_end_to_end(tmp_path):
    data = _write_dataset(tmp_path / "data.extxyz")
    ckpt = _make_checkpoint(tmp_path / "m.pt")
    cfg = from_dict(_bench_dict(
        data, [_model_spec(checkpoint=ckpt)], atomic_energies="average",
        output={"dir": str(tmp_path / "bavg"), "formats": ["json"]}))
    rows = run_benchmark(cfg)
    assert any("energy_mae" in r for r in rows)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def test_columns_lead_with_model():
    rows = [{"model": "a", "energy_mae": 1.0, "n_params": 5}]
    assert columns(rows)[0] == "model"
    assert set(columns(rows)) == {"model", "energy_mae", "n_params"}


def test_write_all_csv_json_md(tmp_path):
    rows = [{"model": "a", "n_params": 3, "energy_mae": 0.5},
            {"model": "b", "n_params": 9, "energy_mae": 1.5}]
    paths = write_all(rows, ["csv", "json", "md"], str(tmp_path), "res")
    assert {p.rsplit(".", 1)[1] for p in paths} == {"csv", "json", "md"}

    with open(tmp_path / "res.json") as f:
        loaded = json.load(f)
    assert loaded[0]["model"] == "a" and loaded[1]["energy_mae"] == 1.5

    with open(tmp_path / "res.csv") as f:
        rowsback = list(csv.DictReader(f))
    assert rowsback[0]["model"] == "a"
    assert float(rowsback[1]["energy_mae"]) == 1.5

    md = (tmp_path / "res.md").read_text()
    assert md.startswith("| model |")


def test_custom_writer_registration(tmp_path):
    @register_writer("tsv")
    def _tsv(rows, cols, path):
        with open(path, "w") as f:
            f.write("\t".join(cols) + "\n")
            for r in rows:
                f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    assert "tsv" in available_writers()
    rows = [{"model": "a", "energy_mae": 0.5}]
    (path,) = write_all(rows, ["tsv"], str(tmp_path), "res")
    assert "\t" in open(path).read()


def test_format_table_renders_header_and_rows():
    rows = [{"model": "a", "energy_mae": 0.5}]
    txt = format_table(rows)
    assert "model" in txt and "a" in txt
    assert format_table([]) == "(no results)"


def test_format_table_shows_units_in_front():
    rows = [{"model": "a", "energy_mae": 0.5, "n_params": 3}]
    txt = format_table(rows, {"energy_mae": "eV/atom"})
    assert "energy_mae [eV/atom]" in txt
    assert "\nn_params" not in txt        # unitless columns unchanged


def test_column_units_defaults_and_override():
    cfg = from_dict({"targets": ["energy", "forces"], "energy_per_atom": True,
                     "units": {"forces": "meV/A"}})
    b = Benchmark(cfg)
    b.rows = [{"model": "m", "n_params": 1, "energy_mae": 0.1,
               "forces_rmse": 0.2}]
    u = b._column_units()
    assert u["energy_mae"] == "eV/atom"     # default
    assert u["forces_rmse"] == "meV/A"      # overridden
    assert "n_params" not in u              # unitless


def test_column_units_energy_total_when_not_per_atom():
    cfg = from_dict({"targets": ["energy"], "energy_per_atom": False})
    b = Benchmark(cfg)
    b.rows = [{"model": "m", "energy_mae": 0.1}]
    assert b._column_units()["energy_mae"] == "eV"


# --------------------------------------------------------------------------- #
# end-to-end scoring driver
# --------------------------------------------------------------------------- #
def test_benchmark_scores_pretrained_models(tmp_path):
    data = _write_dataset(tmp_path / "data.extxyz")
    ck1 = _make_checkpoint(tmp_path / "m1.pt")
    ck2 = _make_checkpoint(tmp_path / "m2.pt")
    out_dir = tmp_path / "bench"
    cfg = from_dict(_bench_dict(
        data,
        [_model_spec(label="a", checkpoint=ck1),
         _model_spec(label="b", checkpoint=ck2)],
        output={"dir": str(out_dir), "filename": "results",
                "formats": ["csv", "json"]}))
    rows = run_benchmark(cfg)

    assert [r["model"] for r in rows] == ["a", "b"]
    for r in rows:
        assert {"energy_mae", "energy_rmse", "forces_mae", "forces_rmse"} <= set(r)
        assert r["n_params"] > 0
        assert "split" not in r          # benchmarking-only: single dataset
    assert (out_dir / "results.csv").exists()
    assert (out_dir / "results.json").exists()


def test_architecture_read_from_checkpoint(tmp_path):
    # A checkpoint written by xnns embeds its Config, so the entry needs no
    # architecture -- just the checkpoint (and an optional label).
    data = _write_dataset(tmp_path / "data.extxyz")
    ckpt = _make_checkpoint(tmp_path / "m.pt", _model_spec())
    cfg = from_dict(_bench_dict(
        data, [{"label": "fromckpt", "checkpoint": ckpt}],
        output={"dir": str(tmp_path / "b"), "formats": ["json"]}))
    assert cfg.models[0].model == {}          # no architecture declared
    rows = run_benchmark(cfg)
    assert rows[0]["model"] == "fromckpt"
    assert rows[0]["n_params"] > 0
    assert "energy_mae" in rows[0]


def test_units_in_written_files_but_rows_stay_plain(tmp_path):
    data = _write_dataset(tmp_path / "data.extxyz")
    ckpt = _make_checkpoint(tmp_path / "m.pt")
    out_dir = tmp_path / "out"
    cfg = from_dict(_bench_dict(
        data, [_model_spec(label="a", checkpoint=ckpt)],
        units={"energy": "eV/atom", "forces": "eV/A"},
        output={"dir": str(out_dir), "formats": ["csv", "json", "md"]}))
    rows = run_benchmark(cfg)

    # returned rows keep plain keys for programmatic use
    assert "energy_mae" in rows[0] and "energy_mae [eV/atom]" not in rows[0]

    # written files carry unit-annotated headers
    loaded = json.load(open(out_dir / "results.json"))
    assert "energy_mae [eV/atom]" in loaded[0]
    assert "forces_mae [eV/A]" in loaded[0]
    assert "energy_mae" not in loaded[0]        # relabeled, not duplicated

    header = open(out_dir / "results.csv").readline()
    assert "energy_mae [eV/atom]" in header
    assert "model" in header and "n_params" in header   # unitless cols plain

    md = (out_dir / "results.md").read_text()
    assert "forces_mae [eV/A]" in md


def test_missing_checkpoint_raises(tmp_path):
    data = _write_dataset(tmp_path / "data.extxyz")
    cfg = from_dict(_bench_dict(data, [_model_spec()]))   # no checkpoint
    with pytest.raises(ValueError, match="checkpoint"):
        run_benchmark(cfg)


def test_checkpoint_not_found_raises(tmp_path):
    data = _write_dataset(tmp_path / "data.extxyz")
    cfg = from_dict(_bench_dict(
        data, [_model_spec(checkpoint=str(tmp_path / "nope.pt"))]))
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        run_benchmark(cfg)


def test_no_dataset_path_raises(tmp_path):
    ckpt = _make_checkpoint(tmp_path / "m.pt")
    cfg = from_dict({"models": [_model_spec(checkpoint=ckpt)],
                     "data": {}, "device": "cpu"})
    with pytest.raises(ValueError, match="dataset"):
        run_benchmark(cfg)


def test_dataset_shared_per_cutoff(tmp_path):
    data = _write_dataset(tmp_path / "data.extxyz")
    ckpt = _make_checkpoint(tmp_path / "m.pt")
    cfg = from_dict(_bench_dict(
        data, [_model_spec(checkpoint=ckpt), _model_spec(checkpoint=ckpt)]))
    b = Benchmark(cfg)
    assert b._dataset(4.0) is b._dataset(4.0)      # cached, identical object


def test_cli_benchmark(tmp_path):
    import yaml
    from xnns.common.cli import main
    data = _write_dataset(tmp_path / "data.extxyz")
    ckpt = _make_checkpoint(tmp_path / "m.pt")
    out_dir = tmp_path / "out"
    cfg_yaml = tmp_path / "bench.yaml"
    cfg_yaml.write_text(yaml.safe_dump(_bench_dict(
        data, [_model_spec(checkpoint=ckpt)],
        output={"dir": str(out_dir), "formats": ["json"]})))
    main(["benchmark", "--config", str(cfg_yaml), "--set", "targets=['energy']"])
    assert (out_dir / "results.json").exists()
    rows = json.load(open(out_dir / "results.json"))
    keys = list(rows[0])            # unit-annotated in files (e.g. "energy_mae [eV/atom]")
    assert any("energy_mae" in k for k in keys)
    assert not any("forces_mae" in k for k in keys)
