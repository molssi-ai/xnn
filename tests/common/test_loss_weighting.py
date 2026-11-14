"""Learned weights of the loss terms (``optim.loss_weighting: uncertainty``)."""
import math
import os

import numpy as np
import pytest
import torch

from xnn.common import distributed
from xnn.common.config import from_dict
from xnn.common.data import AtomicDataset, collate
from xnn.common.train import Trainer, UncertaintyWeights, weighted_loss


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _structures(n=6):
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        q = rng.normal(scale=0.2, size=4)
        out.append({"pos": rng.uniform(0, 3, (4, 3)), "atomic_numbers": [8, 1, 6, 1],
                    "energy": float(rng.normal()), "forces": rng.normal(size=(4, 3)),
                    "charges": q - q.mean(), "dipole": rng.normal(size=3)})
    return out


def _cfg(tmp_path, **optim):
    return from_dict({
        "model": {"name": "qrnn", "cutoff": 5.2, "n_features": 0, "n_interactions": 0, "n_rbf": 0,
                  "extra": {"species": [1, 6, 8], "hidden": [8]}},
        "optim": {"epochs": 2, "lr": 1e-2, "charge_weight": 1.0, "dipole_weight": 0.5,
                  "scheduler": "none", "loss_weighting": "uncertainty", **optim},
        "data": {"batch_size": 3, "val_fraction": 0.0},
        "trainer": {"loggers": []},
        "device": "cpu", "output_dir": str(tmp_path)})


def test_objective_of_the_terms():
    """Each term enters as L / (2 sigma^2) + log sigma; a term without a scale adds as it is."""
    uw = UncertaintyWeights(["energy_mse", "charge_mse"])
    with torch.no_grad():
        uw.log_sigma["energy_mse"].fill_(0.3)
    parts = {"energy_mse": torch.tensor(2.0), "charge_mse": torch.tensor(0.5), "other": torch.tensor(1.5)}
    expected = 2.0 / (2 * math.exp(0.6)) + 0.3 + 0.5 / 2 + 1.5
    assert float(uw(parts)) == pytest.approx(expected, rel=1e-15)
    assert float(uw({})) == 0.0
    assert uw.sigmas()["energy_mse"] == pytest.approx(math.exp(0.3), rel=1e-15)


def test_parts_are_the_summands_of_the_loss(tmp_path):
    trainer = Trainer(_cfg(tmp_path, loss_weighting="fixed"), AtomicDataset(_structures(), 5.2))
    data = collate([trainer.train_loader.dataset[i] for i in range(3)])
    pred = trainer.module(data)
    parts = {}
    loss, logs = weighted_loss(pred, data, 1.0, 10.0, 0.0, charge_weight=1.0, dipole_weight=0.5, parts=parts)
    assert set(parts) == {"energy_mse", "force_mse", "charge_mse", "dipole_mse"}
    assert float(sum(parts.values())) == pytest.approx(float(loss), rel=1e-14)
    assert float(parts["dipole_mse"]) == pytest.approx(0.5 * logs["dipole_mse"], rel=1e-14)
    assert trainer.loss_weights is None


def test_training_learns_the_scales(tmp_path):
    trainer = Trainer(_cfg(tmp_path), AtomicDataset(_structures(), 5.2))
    names = set(trainer.loss_weights.log_sigma.keys())
    assert names == {"energy_mse", "force_mse", "charge_mse", "dipole_mse"}
    assert trainer.opt.param_groups[1]["weight_decay"] == 0.0
    metrics = trainer.fit()["train"]
    assert {f"sigma_{n}" for n in names} <= set(metrics)
    assert any(abs(float(s)) > 1e-6 for s in trainer.loss_weights.log_sigma.values())
    ckpt = torch.load(os.path.join(str(tmp_path), "last.pt"), weights_only=False)
    assert ckpt["loss_sigmas"] == trainer.loss_weights.sigmas()


def test_distributed_steps_average_the_scale_gradients(tmp_path, monkeypatch):
    trainer = Trainer(_cfg(tmp_path), AtomicDataset(_structures(), 5.2))
    calls = []
    monkeypatch.setattr(distributed, "average_gradients", lambda params: calls.append(len(params)))
    trainer.distributed = True
    trainer._step(collate([trainer.train_loader.dataset[i] for i in range(3)]), True)
    assert calls == [4]


def test_average_gradients_in_a_process_group(tmp_path):
    """One rank: the average is the gradient itself; a missing gradient becomes zero."""
    torch.distributed.init_process_group("gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1)
    try:
        a, b = torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(3))
        a.grad = torch.tensor([2.0, 4.0])
        distributed.average_gradients([a, b])
        assert torch.equal(a.grad, torch.tensor([2.0, 4.0])) and torch.equal(b.grad, torch.zeros(3))
    finally:
        torch.distributed.destroy_process_group()
    distributed.average_gradients([a])                       # outside a run: nothing happens


@pytest.mark.parametrize("change, match", [
    ({"loss_weighting": "learned"}, "fixed' or 'uncertainty"),
    ({"energy_weight": 0.0, "force_weight": 0.0, "charge_weight": 0.0, "dipole_weight": 0.0}, "positive weight"),
])
def test_option_errors(tmp_path, change, match):
    with pytest.raises(ValueError, match=match):
        Trainer(_cfg(tmp_path, **change), AtomicDataset(_structures(), 5.2))


def test_multi_head_batches_refused(tmp_path):
    trainer = Trainer(_cfg(tmp_path, loss_weighting="fixed"), AtomicDataset(_structures(), 5.2))
    data = collate([trainer.train_loader.dataset[i] for i in range(2)])
    data.head = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="one head"):
        weighted_loss(trainer.module(data), data, 1.0, 0.0, 0.0, parts={})


def test_captured_steps_and_several_heads_refused(tmp_path):
    trainer = Trainer(_cfg(tmp_path), AtomicDataset(_structures(), 5.2))
    trainer._graph_mode = True
    with pytest.raises(ValueError, match="captured steps"):
        trainer._make_loss_weights(trainer.cfg.optim)
    trainer._graph_mode, trainer.heads = False, ["a", "b"]
    with pytest.raises(ValueError, match="single-head"):
        trainer._make_loss_weights(trainer.cfg.optim)
