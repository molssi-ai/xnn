"""Smoke tests: the core pipeline must run on CPU for molecular & periodic."""
import numpy as np
import torch

from xnns.common.config import Config, from_dict
from xnns.common.data import AtomicDataset, structure_to_graph
from xnns.common.models import build_model, ForceStressOutput, available_models
from xnns.common.train import Trainer


def _structs(periodic):
    rng = np.random.default_rng(0)
    out = []
    for _ in range(8):
        n = 4
        s = {"pos": rng.uniform(0, 4, (n, 3)), "atomic_numbers": [1, 6, 8, 1],
             "energy": float(rng.normal()), "forces": rng.normal(0, 1, (n, 3))}
        if periodic:
            s["cell"] = np.eye(3) * 6.0
            s["pbc"] = [True, True, True]
        out.append(s)
    return out


def test_registry():
    for m in ["schnet", "hdnnp", "ani", "nequip", "mace", "allegro"]:
        assert m in available_models()


def test_forces_molecular():
    cfg = Config(); cfg.model.n_features = 16; cfg.model.n_interactions = 1
    model = ForceStressOutput(build_model(cfg.model))
    g = structure_to_graph(_structs(False)[0], cfg.model.cutoff)
    out = model(g)
    assert out["forces"].shape == g.pos.shape
    assert torch.isfinite(out["energy"]).all()


def test_stress_periodic():
    cfg = from_dict({"model": {"name": "schnet", "n_features": 16,
                               "n_interactions": 1, "cutoff": 5.0}})
    model = ForceStressOutput(build_model(cfg.model), compute_stress=True)
    g = structure_to_graph(_structs(True)[0], cfg.model.cutoff)
    out = model(g)
    assert out["stress"].shape == (1, 3, 3)


def test_train_step_batch():
    cfg = Config(); cfg.model.n_features = 16; cfg.model.n_interactions = 1
    cfg.optim.epochs = 2; cfg.data.batch_size = 4; cfg.data.val_fraction = 0.25
    ds = AtomicDataset(_structs(False), cfg.model.cutoff)
    Trainer(cfg, ds).fit()


def test_train_val_test_split_fractions(tmp_path):
    """test_fraction carves a test split; fit() evaluates it and returns metrics."""
    cfg = Config(); cfg.model.n_features = 16; cfg.model.n_interactions = 1
    cfg.optim.epochs = 1; cfg.data.batch_size = 4
    cfg.data.val_fraction = 0.25; cfg.data.test_fraction = 0.25
    cfg.output_dir = str(tmp_path)
    ds = AtomicDataset(_structs(False), cfg.model.cutoff)   # 8 structures
    t = Trainer(cfg, ds)
    assert len(t.train_loader.dataset) == 4
    assert len(t.val_loader.dataset) == 2
    assert len(t.test_loader.dataset) == 2
    metrics = t.fit()
    assert all(k in metrics for k in ("train", "val", "test"))
    assert "loss" in metrics["test"]


def test_explicit_val_and_test_sets(tmp_path):
    """Explicit val/test datasets are used as-is; the training set is not split."""
    cfg = Config(); cfg.model.n_features = 16; cfg.model.n_interactions = 1
    cfg.optim.epochs = 1; cfg.data.batch_size = 4
    cfg.output_dir = str(tmp_path)
    cutoff = cfg.model.cutoff
    train = AtomicDataset(_structs(False), cutoff)
    val = AtomicDataset(_structs(True), cutoff)
    test = AtomicDataset(_structs(False)[:3], cutoff)
    t = Trainer(cfg, train, val, test)
    assert len(t.train_loader.dataset) == 8      # untouched despite val_fraction=0.1
    assert len(t.test_loader.dataset) == 3
    metrics = t.fit()
    assert "loss" in metrics["val"] and "loss" in metrics["test"]


def test_no_test_split_by_default(tmp_path):
    """Default config (test_fraction=0, no test set) keeps prior behavior."""
    cfg = Config(); cfg.model.n_features = 16; cfg.model.n_interactions = 1
    cfg.optim.epochs = 1; cfg.data.batch_size = 4; cfg.data.val_fraction = 0.25
    cfg.output_dir = str(tmp_path)
    t = Trainer(cfg, AtomicDataset(_structs(False), cfg.model.cutoff))
    assert t.test_loader is None
    assert t.fit()["test"] == {}
