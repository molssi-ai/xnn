"""The trainer's pull of the trainable parameters toward their starting values."""
import numpy as np
import pytest
import torch

from xnn.common.config import from_dict
from xnn.common.data import AtomicDataset, collate
from xnn.common.train import Trainer

POS = np.array([[0.0, 0.0, 0.0], [1.53, 0.0, 0.0], [2.05, 1.44, 0.0], [3.58, 1.44, 0.0],
                [-0.4, -0.5, 0.9], [-0.4, -0.5, -0.9], [-0.4, 1.0, 0.0],
                [1.93, -0.52, 0.88], [1.93, -0.52, -0.88],
                [1.65, 1.96, -0.88], [1.65, 1.96, 0.88],
                [3.98, 0.44, 0.0], [3.98, 1.96, 0.88], [3.98, 1.96, -0.88]])
Z = [6, 6, 6, 6] + [1] * 10
TYPES = ["opls_80", "opls_81", "opls_81", "opls_80"] + ["opls_85"] * 10
BONDS = [[0, 1], [1, 2], [2, 3], [0, 4], [0, 5], [0, 6], [1, 7], [1, 8],
         [2, 9], [2, 10], [3, 11], [3, 12], [3, 13]]


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _structures(n=8):
    rng = np.random.default_rng(0)
    return [{"pos": POS + rng.normal(scale=0.03, size=POS.shape), "atomic_numbers": Z,
             "energy": float(rng.normal())} for _ in range(n)]


def _cfg(tmp_path, **optim):
    return from_dict({
        "model": {"name": "opls", "cutoff": 12.0, "library": "oplsaa", "types": TYPES,
                  "bonds": BONDS, "trainable": ["dihedral_v", "charge"]},
        "optim": {"epochs": 20, "lr": 1e-2, "energy_weight": 1.0, "force_weight": 0.0,
                  "scheduler": "none", **optim},
        "data": {"batch_size": 4, "val_fraction": 0.0},
        "trainer": {"loggers": []},
        "device": "cpu", "output_dir": str(tmp_path)})


def _trainable(trainer):
    return {n: p for n, p in trainer.module.named_parameters() if p.requires_grad}


def _drift(trainer, start):
    return {n: float((p.detach() - start[n]).abs().max())
            for n, p in _trainable(trainer).items()}


def test_prior_holds_parameters_near_their_start(tmp_path):
    ds = AtomicDataset(_structures(), 12.0)
    free = Trainer(_cfg(tmp_path / "free"), ds)
    start = {n: p.detach().clone() for n, p in _trainable(free).items()}
    free.fit()
    held = Trainer(_cfg(tmp_path / "held", prior_weight=1e3), ds)
    assert _trainable(held).keys() == start.keys()
    held.fit()
    d_free, d_held = _drift(free, start), _drift(held, start)
    for name in start:
        assert d_free[name] > 0.0
        assert d_held[name] < 0.5 * d_free[name], name


def test_prior_term_is_logged_and_in_the_loss(tmp_path):
    ds = AtomicDataset(_structures(), 12.0)
    trainer = Trainer(_cfg(tmp_path, prior_weight=2.0), ds)
    start = {n: p.detach().clone() for n, p in _trainable(trainer).items()}
    metrics = trainer._train_epoch()
    assert metrics["loss"] == pytest.approx(metrics["energy_mse"] + metrics["prior"], rel=1e-10)
    # the pull is the weighted squared distance from the start, before the step
    hand = sum(2.0 * float(((p.detach() - start[n]) ** 2).sum())
               for n, p in _trainable(trainer).items())
    batch = collate([ds[i] for i in range(4)])
    logs, counts = trainer._step(batch, True)
    assert logs["prior"] == pytest.approx(hand, rel=1e-10)
    assert counts["prior"] == (1.0, 1.0)
    assert logs["loss"] == pytest.approx(logs["energy_mse"] + logs["prior"], rel=1e-10)
    plain, _ = Trainer(_cfg(tmp_path / "plain"), ds)._step(batch, True)
    assert "prior" not in plain


def test_prior_weights_patterns_pick_parameters(tmp_path):
    ds = AtomicDataset(_structures(), 12.0)
    trainer = Trainer(_cfg(tmp_path, prior_weight=1.0,
                           prior_weights={"*charge": 100.0, "*dihedral_v": 0.0}), ds)
    weights = {id(p): w for p, _, w in trainer._prior}
    by_name = {n: id(p) for n, p in trainer.module.named_parameters()}
    assert weights[by_name["model.ff.params.charge"]] == 100.0
    assert by_name["model.ff.params.dihedral_v"] not in weights
    assert by_name["model.atom_ref.weight"] not in weights
