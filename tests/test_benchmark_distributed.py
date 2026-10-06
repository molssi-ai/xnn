"""Distributed benchmark smoke test: 2-process CPU scoring through torchrun.

Scores the same checkpoints once in a single process and once in two
Gloo-backed ranks launched exactly as a user would with
``torchrun --nproc-per-node 2 -m xnn benchmark``. The ranks score disjoint
strided shards of the dataset (no padding) and gather their predictions, so
the metrics must equal the single-process ones; only rank 0 writes.
"""
import json
import os
import subprocess
import sys

import pytest
import yaml

from test_benchmark import _bench_dict, _make_checkpoint, _model_spec, _write_dataset
from xnn.common.benchmark import from_dict, run_benchmark

pytest.importorskip("ase")


def _torchrun(args, nproc=2, env=None):
    return subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", str(nproc),
         # c10d rendezvous on port 0 grabs a free port, so parallel test runs
         # on a shared machine do not collide
         "--rdzv-backend", "c10d", "--rdzv-endpoint", "localhost:0",
         "-m", "xnn", "benchmark", *args],
        capture_output=True, text=True, timeout=600, env=env)


def test_two_ranks_match_single_process(tmp_path):
    # 13 frames: not divisible by 2, so the shards differ in length
    data = _write_dataset(tmp_path / "data.extxyz", n=13)
    ck1 = _make_checkpoint(tmp_path / "m1.pt")
    ck2 = _make_checkpoint(tmp_path / "m2.pt", _model_spec(n_features=8))
    models = [_model_spec(label="a", checkpoint=ck1),
              _model_spec(label="b", n_features=8, checkpoint=ck2)]

    serial = run_benchmark(from_dict(_bench_dict(
        data, models, output={"dir": str(tmp_path / "serial"), "formats": ["json"]})))

    raw = _bench_dict(data, models, output={"dir": str(tmp_path / "ddp"), "formats": ["json"]})
    cfg_path = tmp_path / "bench.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))
    result = _torchrun(["--config", str(cfg_path)])
    assert result.returncode == 0, result.stdout + result.stderr

    with open(tmp_path / "ddp" / "results.json") as f:
        ddp = json.load(f)
    assert [r["model"] for r in ddp] == ["a", "b"]
    for s, d in zip(serial, ddp):
        for col, val in s.items():
            key = next(k for k in d if k == col or k.startswith(col + " ["))
            if isinstance(val, float):
                # same vectors, summed in a different order
                assert d[key] == pytest.approx(val, rel=1e-5), col
            else:
                assert d[key] == val
    # rank 0 alone printed and wrote the table
    assert result.stdout.count("wrote") == 1


def test_shard_part_under_torchrun(tmp_path):
    data = _write_dataset(tmp_path / "data.extxyz", n=6)
    ck = _make_checkpoint(tmp_path / "m.pt")
    models = [_model_spec(label=l, checkpoint=ck) for l in ("a", "b", "c")]
    out = tmp_path / "out"
    raw = _bench_dict(data, models, output={"dir": str(out), "formats": ["csv"]})
    cfg_path = tmp_path / "bench.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))

    # the job-array pattern: one torchrun per shard, then a merge
    for i in range(2):
        result = _torchrun(["--config", str(cfg_path), "--shard", f"{i}/2"])
        assert result.returncode == 0, result.stdout + result.stderr
    assert sorted(os.listdir(out / "parts")) == ["results.a+c.json", "results.b.json"]
    assert not (out / "results.csv").exists()

    result = subprocess.run(
        [sys.executable, "-m", "xnn", "benchmark", "--config", str(cfg_path), "--merge"],
        capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    lines = (out / "results.csv").read_text().splitlines()
    assert [l.split(",")[0] for l in lines[1:]] == ["a", "b", "c"]
