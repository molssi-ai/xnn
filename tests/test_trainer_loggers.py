"""Trainer loggers: console, CSV, TensorBoard and user subclasses."""
import csv
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from xnn.common.config import from_dict
from xnn.common.data import AtomicDataset
from xnn.common.train import CSVLogger, Logger, Trainer

CUTOFF = 3.0


def _dataset(n, seed):
    rng = np.random.default_rng(seed)
    structures = [{"pos": rng.uniform(0.0, 3.0, (3, 3)), "atomic_numbers": [1, 8, 1],
                   "energy": float(rng.normal()), "forces": rng.normal(size=(3, 3))}
                  for _ in range(n)]
    return AtomicDataset(structures, CUTOFF)


def _cfg(tmp_path, **trainer):
    return from_dict({"model": {"name": "schnet", "n_features": 8, "n_interactions": 1,
                                "cutoff": CUTOFF},
                      "optim": {"epochs": 2, "scheduler": "none"},
                      "data": {"batch_size": 2, "val_fraction": 0.0},
                      "trainer": trainer, "device": "cpu", "output_dir": str(tmp_path)})


class _Recorder(Logger):
    def __init__(self):
        self.calls = []

    def setup(self, output_dir):
        self.calls.append(("setup", output_dir))

    def log_hyperparams(self, params):
        self.calls.append(("hparams", params))

    def log_metrics(self, metrics, step):
        self.calls.append(("metrics", dict(metrics), step))

    def finalize(self, status):
        self.calls.append(("finalize", status))


def test_custom_logger_receives_every_hook(tmp_path):
    """A Logger subclass gets setup, hyperparameters, epoch/test metrics and finalize."""
    rec = _Recorder()
    trainer = Trainer(_cfg(tmp_path), _dataset(5, 0), _dataset(2, 1), _dataset(2, 2),
                      logger=rec)
    out = trainer.fit()
    kinds = [c[0] for c in rec.calls]
    assert kinds == ["setup", "hparams", "metrics", "metrics", "metrics", "finalize"]
    assert rec.calls[1][1]["model"]["name"] == "schnet"
    epochs = [c for c in rec.calls if c[0] == "metrics" and "epoch" in c[1]]
    assert [c[1]["epoch"] for c in epochs] == [0, 1]
    assert [c[2] for c in epochs] == [3, 6]
    assert epochs[-1][1]["train/loss"] == out["train"]["loss"]
    assert epochs[-1][1]["val/loss"] == out["val"]["loss"]
    assert epochs[0][1]["lr"] == pytest.approx(1e-3)
    test = rec.calls[-2]
    assert test[1] == {f"test/{k}": v for k, v in out["test"].items()}
    assert rec.calls[-1] == ("finalize", "success")


def test_step_logging_and_failed_status(tmp_path):
    """log_every_n_steps logs batch values; an exception in fit finalizes as failed."""
    class Failing(_Recorder):
        def log_metrics(self, metrics, step):
            super().log_metrics(metrics, step)
            if "epoch" in metrics:
                raise RuntimeError("stop")

    rec = Failing()
    trainer = Trainer(_cfg(tmp_path, log_every_n_steps=2), _dataset(5, 0), logger=[rec])
    with pytest.raises(RuntimeError, match="stop"):
        trainer.fit()
    steps = [c for c in rec.calls if c[0] == "metrics" and "train_step/loss" in c[1]]
    assert [c[2] for c in steps] == [2]
    assert rec.calls[-1] == ("finalize", "failed")


def test_csv_logger_writes_rows_and_hparams(tmp_path):
    """The CSV file holds one row per logged dict, with columns added as they appear."""
    trainer = Trainer(_cfg(tmp_path, loggers=["csv"]), _dataset(5, 0), _dataset(2, 1),
                      _dataset(2, 2))
    out = trainer.fit()
    with open(tmp_path / "metrics.csv") as f:
        rows = list(csv.DictReader(f))
    assert [r["epoch"] for r in rows] == ["0", "1", ""]
    assert float(rows[1]["val/loss"]) == pytest.approx(out["val"]["loss"], rel=1e-12)
    assert float(rows[2]["test/loss"]) == pytest.approx(out["test"]["loss"], rel=1e-12)
    assert rows[2]["step"] == "6"
    assert (tmp_path / "hparams.yaml").exists()


def test_console_logger_prints_epoch_and_test_lines(tmp_path, capsys):
    """The default console logger prints the epoch lines and the test loss."""
    Trainer(_cfg(tmp_path), _dataset(5, 0), _dataset(2, 1), _dataset(2, 2)).fit()
    out = capsys.readouterr().out
    assert "epoch    0 | train loss " in out and " | val loss " in out
    assert "test loss " in out


def test_logger_false_logs_nothing(tmp_path, capsys):
    """logger=False turns logging off."""
    Trainer(_cfg(tmp_path, loggers=["csv"]), _dataset(4, 0), logger=False).fit()
    assert "epoch" not in capsys.readouterr().out
    assert not (tmp_path / "metrics.csv").exists()


def test_config_loggers_are_validated(tmp_path):
    """A single name becomes a list; an unknown name is rejected."""
    assert _cfg(tmp_path, loggers="CSV").trainer.loggers == ["csv"]
    with pytest.raises(ValueError, match="unknown"):
        _cfg(tmp_path, loggers=["wandb"])


def test_explicit_save_dir_wins(tmp_path):
    """A logger built with a directory keeps it."""
    lg = CSVLogger(save_dir=str(tmp_path / "logs"))
    Trainer(_cfg(tmp_path), _dataset(4, 0), logger=lg).fit()
    assert os.path.exists(tmp_path / "logs" / "metrics.csv")


def test_tensorboard_logger_writes_scalars(tmp_path):
    """TensorBoard receives one scalar series per metric."""
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    trainer = Trainer(_cfg(tmp_path, loggers=["tensorboard"]), _dataset(5, 0), _dataset(2, 1))
    out = trainer.fit()
    acc = EventAccumulator(str(tmp_path / "tensorboard"))
    acc.Reload()
    events = acc.Scalars("val/loss")
    assert [e.step for e in events] == [3, 6]
    assert events[-1].value == pytest.approx(out["val"]["loss"], rel=1e-6)


# The same run under every launcher: PaiNN with every loss term (energy, forces,
# stress, dipole, polarizability, charges) on clusters and cells mixed in uneven
# batches, each term labelled on some structures only; under DDP the strided
# validation shards leave rank 1 without stress, dipole and polarizability labels
# and rank 0 without charges. Every hook is recorded per rank.
_SCRIPT = """
import json
import os

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
from xnn.common.benchmark.metrics import prediction_pairs
from xnn.common.config import from_dict
from xnn.common.data import AtomicDataset, collate
from xnn.common.train import Logger, Trainer
from xnn.common.train.losses import weighted_loss


class Record(Logger):
    def __init__(self):
        self.calls = []

    def log_hyperparams(self, params):
        self.calls.append(["hparams"])

    def log_metrics(self, metrics, step):
        self.calls.append(["metrics", metrics, step])

    def finalize(self, status):
        self.calls.append(["finalize", status])


TARGETS = ("energy", "forces", "stress", "dipole", "polarizability", "charges")
rng = np.random.default_rng(0)
def structs(n, start, cells, labels):
    out = []
    for i in range(n):
        s = {"pos": rng.uniform(0, 4, (4, 3)), "atomic_numbers": [1, 6, 8, 1]}
        if i in cells:
            s["cell"] = np.eye(3) * 6.0
            s["pbc"] = [True, True, True]
        for key, where in labels.items():
            if i not in where:
                continue
            if key == "energy":
                s["energy"] = float(start + i)
            elif key == "stress":
                s["stress"] = rng.normal(scale=0.01, size=(3, 3))
            else:
                shape = {"forces": (4, 3), "dipole": (3,), "polarizability": (3, 3),
                         "charges": (4,)}[key]
                s[key] = rng.normal(scale=0.2, size=shape)
        out.append(s)
    return out


cfg = from_dict({"model": {"name": "painn", "n_features": 8, "n_interactions": 1, "n_rbf": 4,
                           "cutoff": 3.0, "extra": {"dipole": True, "polarizability": True}},
                 "optim": {"epochs": 3, "lr": 2e-2, "scheduler": "plateau",
                           "energy_weight": 1.0, "force_weight": 10.0, "stress_weight": 100.0,
                           "dipole_weight": 1.0, "polarizability_weight": 1.0,
                           "charge_weight": 1.0},
                 "data": {"batch_size": 2, "val_fraction": 0.0},
                 "trainer": {"log_every_n_steps": 1, "step_average": "running",
                             "monitor": "val/forces_rmse"},
                 "device": "cpu", "output_dir": OUTPUT_DIR})
cut = cfg.model.cutoff
train = AtomicDataset(structs(9, 0, {2, 5}, {
    "energy": range(9), "forces": {0, 1, 2, 3, 4, 5, 6, 8}, "stress": {2, 5},
    "dipole": {0, 4, 6, 8}, "polarizability": {0, 3}, "charges": {1, 4, 8}}), cut)
val = AtomicDataset(structs(7, 100, {4}, {
    "energy": {0, 1, 2, 4, 5, 6}, "forces": {0, 2, 3, 4, 5, 6}, "stress": {4},
    "dipole": {0, 2}, "polarizability": {6}, "charges": {1, 5}}), cut)
test = AtomicDataset(structs(5, 200, {1}, {
    "energy": range(5), "forces": {0, 1, 3, 4}, "stress": {1},
    "dipole": {0, 4}, "polarizability": {3}, "charges": {2}}), cut)
rec = Record()
loggers = ["console", "csv", rec]
try:
    import tensorboard  # noqa: F401
    loggers.append("tensorboard")
except ImportError:
    pass
trainer = Trainer(cfg, train, val, test, logger=loggers)
metrics = trainer.fit()

o = cfg.optim
def one_batch(ds):
    data = collate([ds[i] for i in range(len(ds))])
    pred = trainer.eval_module(data)
    _, ref = weighted_loss(pred, data, o.energy_weight, o.force_weight, o.stress_weight,
                           dipole_weight=o.dipole_weight,
                           polarizability_weight=o.polarizability_weight,
                           charge_weight=o.charge_weight)
    for target, (p, r) in prediction_pairs(pred, data, list(TARGETS)).items():
        d = (p - r).double()
        ref[target + "_mae"] = float(d.abs().mean())
        ref[target + "_rmse"] = float(d.pow(2).mean().sqrt())
    return ref
last = {"val": one_batch(val), "test": one_batch(test)}
ckpt = torch.load(os.path.join(OUTPUT_DIR, "best.pt"), weights_only=False)
trainer.eval_module.load_state_dict(ckpt["model"])
out = {"calls": rec.calls, "metrics": metrics, "global_step": trainer.global_step,
       "last": last, "best": one_batch(val)}
with open(os.path.join(OUTPUT_DIR, "rank_" + os.environ.get("RANK", "0") + ".json"), "w") as fh:
    json.dump(out, fh)
"""

TERMS = ({"loss", "energy_mse", "force_mse", "stress_mse", "dipole_mse", "polarizability_mse",
          "charge_mse"}
         | {f"{t}_{m}" for t in ("energy", "forces", "stress", "dipole", "polarizability",
                                 "charges") for m in ("mae", "rmse")})

_TORCHRUN = [sys.executable, "-m", "torch.distributed.run",
             "--rdzv-backend", "c10d", "--rdzv-endpoint", "localhost:0"]


@pytest.mark.parametrize("launch,world", [([sys.executable], 1),
                                          (_TORCHRUN + ["--nproc-per-node", "1"], 1),
                                          (_TORCHRUN + ["--nproc-per-node", "2"], 2)],
                         ids=["python", "torchrun-1", "ddp-2"])
def test_logging_and_best_checkpoint_under_every_launcher(tmp_path, launch, world):
    """Rank-0 logging, all-term dataset-level losses and errors, running step values and the
    monitored best.pt agree under every launcher."""
    script = tmp_path / "run.py"
    script.write_text(f"OUTPUT_DIR = {str(tmp_path)!r}\n" + _SCRIPT)
    result = subprocess.run(launch + [str(script)], capture_output=True, text=True,
                            timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    ranks = [json.loads((tmp_path / f"rank_{r}.json").read_text()) for r in range(world)]
    r0 = ranks[0]

    # only rank 0 logs; every rank took the same steps and sees the same metrics
    for other in ranks[1:]:
        assert other["calls"] == []
        assert other["global_step"] == r0["global_step"]
        for split, values in r0["metrics"].items():
            assert other["metrics"][split] == pytest.approx(values, rel=1e-12)
    assert result.stdout.count("epoch    0 |") == 1
    assert result.stdout.count("test loss ") == 1

    calls = r0["calls"]
    assert calls[0] == ["hparams"] and calls[-1] == ["finalize", "success"]
    logged = [c for c in calls if c[0] == "metrics"]
    epochs = [c for c in logged if "epoch" in c[1]]
    batches = [c for c in logged if "train_step/loss" in c[1]]
    assert [c[1]["epoch"] for c in epochs] == [0, 1, 2]
    assert epochs[-1][2] == r0["global_step"]
    assert [c[2] for c in batches] == list(range(1, r0["global_step"] + 1))
    # running step values: the last step of an epoch is that epoch's training value
    for c in epochs:
        end = next(b for b in batches if b[2] == c[2])
        train = {k[6:]: v for k, v in c[1].items() if k.startswith("train/")}
        assert {k[11:]: v for k, v in end[1].items()} == pytest.approx(train, rel=1e-12)

    # every term reaches every split, and the logged values are the
    # dataset-level losses of the whole set
    for split in ("train", "val", "test"):
        assert set(r0["metrics"][split]) == TERMS, split
    final = {k[4:]: v for k, v in epochs[-1][1].items() if k.startswith("val/")}
    assert final == pytest.approx(r0["last"]["val"], rel=1e-8)
    test = {k[5:]: v for k, v in logged[-1][1].items() if k.startswith("test/")}
    assert test == pytest.approx(r0["last"]["test"], rel=1e-8)
    assert test == pytest.approx(r0["metrics"]["test"], rel=1e-12)
    # best.pt holds the epoch with the lowest monitored value (val/forces_rmse)
    assert r0["best"]["forces_rmse"] == pytest.approx(
        min(c[1]["val/forces_rmse"] for c in epochs), rel=1e-8)

    with open(tmp_path / "metrics.csv") as f:
        assert len(list(csv.DictReader(f))) == len(logged)
    if (tmp_path / "tensorboard").exists():
        assert len(list((tmp_path / "tensorboard").glob("events.out.tfevents.*"))) == 1
