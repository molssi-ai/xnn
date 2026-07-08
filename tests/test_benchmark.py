"""Benchmarking feature: config parsing, metrics, report writers, phase driver."""
import copy
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


# --------------------------------------------------------------------------- #
# data helpers
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


def _base_dict(train_path, **over):
    d = {
        "models": [_model_spec()],
        "phases": ["train", "evaluate", "benchmark"],
        "metrics": ["mae", "rmse"],
        "targets": ["energy", "forces"],
        "data": {"train_path": train_path, "batch_size": 4,
                 "val_fraction": 0.25, "test_fraction": 0.25},
        "optim": {"epochs": 1, "scheduler": "none"},
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
                                 "optim": {"epochs": 5}, "output_dir": "o"}]})
    e = cfg.models[0]
    assert e.checkpoint == "a/best.pt"
    assert e.optim == {"epochs": 5}
    assert e.output_dir == "o"
    assert "checkpoint" not in e.model and "optim" not in e.model


def test_model_config_from_file(tmp_path):
    import yaml
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump({"name": "schnet", "cutoff": 6.0,
                                 "n_features": 8}))
    # inline keys override the file
    cfg = from_dict({"models": [{"config": str(p), "n_features": 32}]})
    assert cfg.models[0].model["cutoff"] == 6.0
    assert cfg.models[0].model["n_features"] == 32


def test_to_run_config_merges_shared_and_per_model():
    cfg = from_dict({
        "models": [{"name": "schnet", "cutoff": 5.0, "optim": {"epochs": 7}}],
        "optim": {"epochs": 100, "lr": 1e-3},
        "data": {"batch_size": 8},
        "output": {"dir": "runs/b"},
        "seed": 42, "device": "cpu",
    })
    rc = cfg.models[0].to_run_config(cfg)
    assert rc.model.name == "schnet"
    assert rc.model.cutoff == 5.0
    assert rc.data.cutoff == 5.0           # kept in lockstep by Config
    assert rc.data.batch_size == 8
    assert rc.optim.epochs == 7            # per-model override wins
    assert rc.optim.lr == 1e-3             # shared value preserved
    assert rc.output_dir == "runs/b/schnet"
    assert rc.seed == 42 and rc.device == "cpu"


def test_scalar_fields_accept_scalar_or_list():
    cfg = from_dict({"phases": "benchmark", "metrics": "mae",
                     "targets": "energy"})
    assert cfg.phases == ["benchmark"]
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
# report
# --------------------------------------------------------------------------- #
def test_columns_lead_with_model_and_split():
    rows = [{"model": "a", "split": "test", "energy_mae": 1.0, "n_params": 5}]
    assert columns(rows)[:2] == ["model", "split"]
    assert set(columns(rows)) == {"model", "split", "energy_mae", "n_params"}


def test_write_all_csv_json_md(tmp_path):
    rows = [{"model": "a", "split": "test", "energy_mae": 0.5, "n_params": 3},
            {"model": "b", "split": "test", "energy_mae": 1.5, "n_params": 9}]
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
    assert md.startswith("| model | split |")


def test_custom_writer_registration(tmp_path):
    @register_writer("tsv")
    def _tsv(rows, cols, path):
        with open(path, "w") as f:
            f.write("\t".join(cols) + "\n")
            for r in rows:
                f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    assert "tsv" in available_writers()
    rows = [{"model": "a", "split": "test", "energy_mae": 0.5}]
    (path,) = write_all(rows, ["tsv"], str(tmp_path), "res")
    assert "\t" in open(path).read()


def test_format_table_renders_header_and_rows():
    rows = [{"model": "a", "split": "test", "energy_mae": 0.5}]
    txt = format_table(rows)
    assert "model" in txt and "a" in txt
    assert format_table([]) == "(no results)"


# --------------------------------------------------------------------------- #
# end-to-end phase driver
# --------------------------------------------------------------------------- #
def test_train_evaluate_benchmark(tmp_path):
    train_path = _write_dataset(tmp_path / "data.extxyz")
    out_dir = tmp_path / "bench"
    cfg = from_dict(_base_dict(
        train_path, output={"dir": str(out_dir), "filename": "results",
                             "formats": ["csv", "json"]}))
    rows = run_benchmark(cfg)

    # one val row (evaluate) + one test row (benchmark)
    splits = sorted(r["split"] for r in rows)
    assert splits == ["test", "val"]
    for r in rows:
        assert {"energy_mae", "energy_rmse", "forces_mae", "forces_rmse"} <= set(r)
        assert r["n_params"] > 0
    assert (out_dir / "results.csv").exists()
    assert (out_dir / "results.json").exists()
    # a checkpoint was trained
    assert (out_dir / "schnet" / "last.pt").exists()


def test_benchmark_only_on_pretrained_checkpoint(tmp_path):
    # First train + save a checkpoint via a full run.
    train_path = _write_dataset(tmp_path / "data.extxyz")
    first = from_dict(_base_dict(
        train_path, models=[_model_spec()],
        output={"dir": str(tmp_path / "b1"), "formats": ["json"]}))
    run_benchmark(first)
    ckpt = tmp_path / "b1" / "schnet" / "last.pt"
    assert ckpt.exists()

    # Now benchmark ONLY, loading the pre-trained checkpoint (no training).
    cfg = from_dict(_base_dict(
        train_path,
        models=[_model_spec(label="pretrained", checkpoint=str(ckpt))],
        phases=["benchmark"],
        output={"dir": str(tmp_path / "b2"), "formats": ["csv"]}))
    b = Benchmark(cfg)
    rows = b.run()
    assert len(rows) == 1
    assert rows[0]["model"] == "pretrained"
    assert rows[0]["split"] == "test"
    # no training happened for b2 -> no checkpoints written under its out dir
    assert not (tmp_path / "b2" / "pretrained").exists()
    assert (tmp_path / "b2" / "results.csv").exists()


def test_evaluate_benchmark_on_pretrained(tmp_path):
    train_path = _write_dataset(tmp_path / "data.extxyz")
    first = from_dict(_base_dict(train_path,
                                 output={"dir": str(tmp_path / "b1")}))
    run_benchmark(first)
    ckpt = tmp_path / "b1" / "schnet" / "last.pt"

    cfg = from_dict(_base_dict(
        train_path,
        models=[_model_spec(checkpoint=str(ckpt))],
        phases=["evaluate", "benchmark"],
        output={"dir": str(tmp_path / "b3"), "formats": ["json"]}))
    rows = run_benchmark(cfg)
    assert sorted(r["split"] for r in rows) == ["test", "val"]


def test_no_checkpoint_without_train_phase_raises(tmp_path):
    train_path = _write_dataset(tmp_path / "data.extxyz")
    cfg = from_dict(_base_dict(
        train_path, phases=["benchmark"],
        output={"dir": str(tmp_path / "b4")}))
    with pytest.raises(ValueError, match="no checkpoint"):
        run_benchmark(cfg)


def test_splits_are_shared_per_cutoff(tmp_path):
    train_path = _write_dataset(tmp_path / "data.extxyz")
    cfg = from_dict(_base_dict(
        train_path, models=[_model_spec(), _model_spec()],
        output={"dir": str(tmp_path / "b5")}))
    b = Benchmark(cfg)
    s1 = b._splits_for(4.0)
    s2 = b._splits_for(4.0)
    assert s1 is s2      # cached, identical objects
