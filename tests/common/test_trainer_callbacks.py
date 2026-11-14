"""Trainer callbacks and options: monitor, early stopping, frequencies, checkpoints, error metrics."""
import math

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from xnn.common.benchmark.metrics import prediction_pairs
from xnn.common.config import from_dict
from xnn.common.data import AtomicDataset, collate
from xnn.common.train import Callback, EarlyStopping, Logger, ModelCheckpoint, Trainer

CUTOFF = 3.0


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _dataset(n, seed, forces=lambda i: True):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        s = {"pos": rng.uniform(0.0, 3.0, (3, 3)), "atomic_numbers": [1, 8, 1],
             "energy": float(rng.normal())}
        if forces(i):
            s["forces"] = rng.normal(size=(3, 3))
        out.append(s)
    return AtomicDataset(out, CUTOFF)


def _cfg(tmp_path, epochs=4, **trainer):
    return from_dict({"model": {"name": "schnet", "n_features": 8, "n_interactions": 1,
                                "cutoff": CUTOFF},
                      "optim": {"epochs": epochs, "lr": 2e-2, "scheduler": "none"},
                      "data": {"batch_size": 2, "val_fraction": 0.0},
                      "trainer": trainer, "device": "cpu", "output_dir": str(tmp_path)})


class _Record(Logger):
    def __init__(self):
        self.rows = []

    def log_metrics(self, metrics, step):
        self.rows.append((dict(metrics), step))

    def epochs(self):
        return [m for m, _ in self.rows if "epoch" in m]


def test_error_metrics_equal_one_batch_of_the_whole_set(tmp_path):
    """evaluate() MAE / RMSE over uneven, partially labelled batches equal those of one batch."""
    val = _dataset(7, 1, forces=lambda i: i % 3 != 1)
    trainer = Trainer(_cfg(tmp_path), _dataset(4, 0), val, logger=False)
    got = trainer.evaluate(DataLoader(val, batch_size=3, collate_fn=collate))
    data = collate([val[i] for i in range(len(val))])
    pairs = prediction_pairs(trainer.eval_module(data), data, ["energy", "forces"])
    for target, (p, r) in pairs.items():
        d = p - r
        assert got[f"{target}_mae"] == pytest.approx(float(d.abs().mean()), rel=1e-10)
        assert got[f"{target}_rmse"] == pytest.approx(float(d.pow(2).mean().sqrt()), rel=1e-10)


def test_metrics_option_selects_the_error_metrics(tmp_path):
    """trainer.metrics picks MAE and/or RMSE; an empty list keeps the loss terms only."""
    trainer = Trainer(_cfg(tmp_path, metrics=["mae"]), _dataset(4, 0), _dataset(3, 1), logger=False)
    assert {"energy_mae", "forces_mae"} <= set(trainer.evaluate(trainer.val_loader))
    assert not any(k.endswith("_rmse") for k in trainer.evaluate(trainer.val_loader))
    trainer = Trainer(_cfg(tmp_path, metrics=[]), _dataset(4, 0), _dataset(3, 1), logger=False)
    assert set(trainer.evaluate(trainer.val_loader)) == {"loss", "energy_mse", "force_mse"}


def test_best_checkpoint_follows_the_monitor(tmp_path):
    """best.pt holds the epoch with the best monitored value, in either mode."""
    val = _dataset(5, 1)
    for mode, pick in (("min", min), ("max", max)):
        rec = _Record()
        trainer = Trainer(_cfg(tmp_path / mode, monitor="val/forces_mae", mode=mode),
                          _dataset(6, 0), val, logger=rec)
        trainer.fit()
        values = [m["val/forces_mae"] for m in rec.epochs()]
        assert len(set(values)) == len(values)
        ckpt = torch.load(tmp_path / mode / "best.pt", weights_only=False)
        trainer.eval_module.load_state_dict(ckpt["model"])
        assert trainer.evaluate(trainer.val_loader)["forces_mae"] == pytest.approx(
            pick(values), rel=1e-10)
        cb = next(c for c in trainer.callbacks if isinstance(c, ModelCheckpoint))
        assert cb.best_score == pytest.approx(pick(values), rel=1e-12)


def test_val_monitor_falls_back_to_train_and_unknown_keys_fail(tmp_path):
    """Without a validation set val/ reads train/; a monitor that is never logged raises."""
    trainer = Trainer(_cfg(tmp_path, epochs=1), _dataset(4, 0), logger=False)
    assert trainer.monitor == "train/loss"
    trainer.fit()
    assert (tmp_path / "best.pt").exists()
    trainer = Trainer(_cfg(tmp_path, epochs=1, monitor="val/stress_mae"), _dataset(4, 0),
                      _dataset(2, 1), logger=False)
    with pytest.raises(KeyError, match="stress_mae"):
        trainer.fit()


def test_validation_and_logging_frequencies(tmp_path):
    """val_every_n_epochs and log_every_n_epochs thin the work and the log, keeping the last epoch."""
    rec = _Record()
    Trainer(_cfg(tmp_path, epochs=5, val_every_n_epochs=2, log_every_n_epochs=2),
            _dataset(4, 0), _dataset(2, 1), logger=rec).fit()
    epochs = rec.epochs()
    assert [m["epoch"] for m in epochs] == [1, 3, 4]
    assert all("val/loss" in m for m in epochs)

    rec = _Record()
    Trainer(_cfg(tmp_path, epochs=3, val_every_n_epochs=2), _dataset(4, 0), _dataset(2, 1),
            logger=rec).fit()
    assert ["val/loss" in m for m in rec.epochs()] == [False, True, True]


def test_running_and_batch_step_values(tmp_path):
    """Running step values end each epoch at its training value; batch values are per batch."""
    rec = _Record()
    Trainer(_cfg(tmp_path, epochs=2, log_every_n_steps=1, step_average="running"),
            _dataset(5, 0, forces=lambda i: i != 2), logger=rec).fit()
    for metrics, step in rec.rows:
        if "epoch" in metrics:
            end = next(m for m, s in rec.rows if s == step and "epoch" not in m)
            train = {k[6:]: v for k, v in metrics.items() if k.startswith("train/")}
            assert {k[11:]: v for k, v in end.items()} == pytest.approx(train, rel=1e-12)

    rec = _Record()
    Trainer(_cfg(tmp_path, epochs=1, log_every_n_steps=1), _dataset(5, 0), logger=rec).fit()
    steps = [m for m, _ in rec.rows if "epoch" not in m]
    assert len(steps) == 3 and steps[0] != steps[-1]


def test_early_stopping_patience_threshold_and_non_finite(tmp_path):
    """Early stopping fires on patience, on a threshold and on a non-finite value."""
    trainer = Trainer(_cfg(tmp_path, epochs=50, monitor="train/loss",
                           early_stopping_patience=1, early_stopping_min_delta=1e9),
                      _dataset(4, 0), logger=False)
    trainer.fit()
    stop = next(c for c in trainer.callbacks if isinstance(c, EarlyStopping))
    assert stop.stopped_epoch == 1 and "did not improve" in stop.reason

    rec = _Record()
    trainer = Trainer(_cfg(tmp_path, epochs=50, monitor="train/loss",
                           early_stopping_threshold=1e9), _dataset(4, 0), logger=rec)
    out = trainer.fit()
    stop = next(c for c in trainer.callbacks if isinstance(c, EarlyStopping))
    assert stop.stopped_epoch == 0 and "threshold" in stop.reason
    assert [m["epoch"] for m in rec.epochs()] == [0]
    assert out["train"]["loss"] == rec.epochs()[0]["train/loss"]
    assert (tmp_path / "last.pt").exists()

    class Poison(Callback):
        def on_epoch_end(self, trainer, epoch, metrics):
            metrics["train/loss"] = math.nan

    trainer = Trainer(_cfg(tmp_path, epochs=5), _dataset(4, 0), logger=False,
                      callbacks=[Poison(), EarlyStopping(monitor="train/loss", patience=0)])
    trainer.fit()
    stop = next(c for c in trainer.callbacks if isinstance(c, EarlyStopping))
    assert stop.stopped_epoch == 0 and "not finite" in stop.reason


def test_periodic_checkpoints_and_custom_checkpoint(tmp_path):
    """save_every_n_epochs keeps epoch_<n>.pt; a given ModelCheckpoint replaces the default."""
    Trainer(_cfg(tmp_path / "a", epochs=4, save_every_n_epochs=2), _dataset(4, 0),
            logger=False).fit()
    assert sorted(p.name for p in (tmp_path / "a").glob("*.pt")) == [
        "best.pt", "epoch_2.pt", "epoch_4.pt", "last.pt"]

    cb = ModelCheckpoint(monitor="train/forces_rmse", filename="lowest_forces.pt",
                         save_last=False)
    trainer = Trainer(_cfg(tmp_path / "b", epochs=2), _dataset(4, 0), logger=False,
                      callbacks=[cb])
    trainer.fit()
    assert [c for c in trainer.callbacks if isinstance(c, ModelCheckpoint)] == [cb]
    assert sorted(p.name for p in (tmp_path / "b").glob("*.pt")) == ["lowest_forces.pt"]


def test_callback_hooks_run_in_order(tmp_path):
    """A custom callback sees every hook, with the trainer and the epoch metrics."""
    seen = []

    class Hooks(Callback):
        def on_fit_start(self, trainer):
            seen.append("fit_start")

        def on_train_epoch_start(self, trainer, epoch):
            seen.append(f"epoch_start {epoch}")

        def on_train_batch_end(self, trainer, logs, batch_idx):
            seen.append(f"batch {batch_idx}")

        def on_epoch_end(self, trainer, epoch, metrics):
            seen.append(f"epoch_end {epoch} {'val/loss' in metrics}")

        def on_train_end(self, trainer):
            seen.append("train_end")

        def on_fit_end(self, trainer, results):
            seen.append(f"fit_end {sorted(results)}")

    Trainer(_cfg(tmp_path, epochs=2, progress_bar=True), _dataset(3, 0), _dataset(2, 1),
            _dataset(2, 2), logger=False, callbacks=[Hooks()]).fit()
    assert seen == ["fit_start", "epoch_start 0", "batch 0", "batch 1", "epoch_end 0 True",
                    "epoch_start 1", "batch 0", "batch 1", "epoch_end 1 True", "train_end",
                    "fit_end ['test', 'train', 'val']"]


def test_trainer_options_are_validated(tmp_path):
    """Unknown metrics, modes and averages are rejected when the config is built."""
    for bad in ({"metrics": ["r2"]}, {"mode": "best"}, {"step_average": "epoch"},
                {"val_every_n_epochs": 0}):
        with pytest.raises(ValueError):
            _cfg(tmp_path, **bad)
