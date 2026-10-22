"""Dataset-level metrics of the trainer over uneven, partially labelled batches."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from xnn.common.config import from_dict
from xnn.common.data import AtomicDataset, collate, structure_to_graph
from xnn.common.train import Trainer
from xnn.common.train.loggers import ConsoleLogger, _summary
from xnn.common.train.losses import weighted_loss

BOX = 6.0
CUTOFF = 3.0
# sizes, periodicity and labels: stress only on the cells, forces missing on two
# clusters, so the batches of three carry different numbers of each label
SPEC = [(3, False, ("energy", "forces")), (5, True, ("energy", "forces", "stress")),
        (2, False, ("energy",)), (4, False, ("energy", "forces")),
        (6, True, ("energy", "forces", "stress")), (3, False, ("energy",)),
        (4, False, ("energy", "forces"))]


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _structures(spec=SPEC, heads=None):
    out = []
    for i, (n, periodic, labels) in enumerate(spec):
        rng = np.random.default_rng(i)
        s = {"pos": rng.uniform(0.0, BOX - 0.5, (n, 3)), "atomic_numbers": ([1, 8] * n)[:n],
             "energy": float(rng.normal())}
        if periodic:
            s["cell"] = np.eye(3) * BOX
            s["pbc"] = [True, True, True]
        if "forces" in labels:
            s["forces"] = rng.normal(size=(n, 3))
        if "stress" in labels:
            s["stress"] = rng.normal(scale=0.01, size=(3, 3))
        if heads is not None:
            s["head"] = heads[i]
        out.append(s)
    return out


def _cfg(tmp_path):
    return from_dict({"model": {"name": "schnet", "n_features": 8, "n_interactions": 1,
                                "cutoff": CUTOFF},
                      "optim": {"epochs": 1, "energy_weight": 1.0, "force_weight": 10.0,
                                "stress_weight": 100.0},
                      "data": {"batch_size": 3, "val_fraction": 0.0},
                      "device": "cpu", "output_dir": str(tmp_path)})


def _metrics(steps):
    """Accumulate (logs, counts) pairs the way a single-process trainer does."""
    return Trainer._dataset_metrics(SimpleNamespace(distributed=False), steps)


def test_evaluate_equals_one_batch_of_all_structures(tmp_path):
    """evaluate() over uneven batches equals the loss over one batch of everything."""
    cfg = _cfg(tmp_path)
    ds = AtomicDataset(_structures(), CUTOFF)
    trainer = Trainer(cfg, ds)
    got = trainer.evaluate(DataLoader(ds, batch_size=3, collate_fn=collate))

    data = collate([ds[i] for i in range(len(ds))])
    o = cfg.optim
    _, ref = weighted_loss(trainer.eval_module(data), data,
                           o.energy_weight, o.force_weight, o.stress_weight)
    assert set(ref) <= set(got)
    for key in ref:
        assert got[key] == pytest.approx(ref[key], rel=1e-10), key

    # the mean of the batch losses is a different number on these batches
    naive = np.mean([logs["loss"] for logs, _ in
                     (trainer._step(d, False) for d in DataLoader(ds, batch_size=3,
                                                                  collate_fn=collate))])
    assert abs(naive - ref["loss"]) > 1e-3 * ref["loss"]



def test_best_checkpoint_is_the_lowest_dataset_level_val_loss(tmp_path):
    """best.pt scored on the whole validation set reproduces the lowest logged val/loss."""
    from xnn.common.train import Logger

    class Record(Logger):
        def __init__(self):
            self.val = []

        def log_metrics(self, metrics, step):
            if "val/loss" in metrics:
                self.val.append(metrics["val/loss"])

    cfg = _cfg(tmp_path)
    cfg.optim.epochs, cfg.optim.lr, cfg.optim.scheduler = 6, 2e-2, "none"
    rec = Record()
    val = AtomicDataset(_structures(), CUTOFF)
    trainer = Trainer(cfg, AtomicDataset(_structures(), CUTOFF), val, logger=rec)
    trainer.fit()
    assert len(set(rec.val)) == len(rec.val)

    ckpt = torch.load(tmp_path / "best.pt", weights_only=False)
    trainer.eval_module.load_state_dict(ckpt["model"])
    data = collate([val[i] for i in range(len(val))])
    o = cfg.optim
    _, ref = weighted_loss(trainer.eval_module(data), data,
                           o.energy_weight, o.force_weight, o.stress_weight)
    assert ref["loss"] == pytest.approx(min(rec.val), rel=1e-10)

def _slice(pred, data, lo, hi):
    """The predictions of structures lo..hi-1 of a collated batch."""
    atoms = (data.batch >= lo) & (data.batch < hi)
    return {"energy": pred["energy"][lo:hi], "forces": pred["forces"][atoms],
            "stress": pred["stress"][lo:hi]}


@pytest.mark.parametrize("heads", [None, [0, 0, 0, 1, 1, 0, 1]])
def test_counts_recombine_batches_into_the_one_batch_loss(heads):
    """Batch logs and counts recombine into the whole-dataset loss, per head too."""
    structs = _structures(heads=heads)
    for i, s in enumerate(structs):
        s["weight"] = 0.5 + 0.25 * i
    graphs = [structure_to_graph(s, CUTOFF) for s in structs]
    whole = collate(graphs)
    g = torch.Generator().manual_seed(0)
    pred = {"energy": whole.energy + torch.randn(whole.energy.shape, generator=g),
            "forces": torch.randn(whole.pos.shape, generator=g),
            "stress": torch.randn((len(graphs), 3, 3), generator=g) * 0.01}
    kw = {"head_names": ["a", "b"], "head_weights": {1: (2.0, 5.0, 50.0)}} if heads else {}
    _, ref = weighted_loss(pred, whole, 1.0, 10.0, 100.0, **kw)

    steps = []
    for lo, hi in ((0, 3), (3, 5), (5, 7)):
        counts = {}
        _, logs = weighted_loss(_slice(pred, whole, lo, hi), collate(graphs[lo:hi]),
                                1.0, 10.0, 100.0, counts=counts, **kw)
        assert set(counts) == {k for k in logs if not k.endswith("loss")}
        steps.append((logs, counts))
    got = _metrics(steps)
    assert set(got) == set(ref)
    for key in ref:
        assert got[key] == pytest.approx(ref[key], rel=1e-12), key


def test_single_head_batch_logs_under_its_head_name():
    """A one-head batch of a multi-head run logs its terms under the head's name."""
    structs = _structures(heads=[1] * len(SPEC))
    data = collate([structure_to_graph(s, CUTOFF) for s in structs])
    pred = {"energy": data.energy + 0.1, "forces": torch.zeros_like(data.pos),
            "stress": torch.zeros(len(structs), 3, 3)}
    counts = {}
    loss, logs = weighted_loss(pred, data, 1.0, 10.0, 100.0, head_names=["a", "b"],
                               counts=counts)
    assert set(logs) == {"b/energy_mse", "b/force_mse", "b/stress_mse", "b/loss", "loss"}
    assert set(counts) == {"b/energy_mse", "b/force_mse", "b/stress_mse"}
    assert logs["b/loss"] == logs["loss"] == float(loss)


def test_log_line_reports_the_terms(capsys):
    """The epoch line shows the train and validation terms and the head losses."""
    tr = {"loss": 1.5, "energy_mse": 0.25, "force_mse": 0.125}
    va = {"loss": 2.0, "a/energy_mse": 0.5, "a/loss": 0.5, "b/loss": 1.5}
    ConsoleLogger().log_metrics({"epoch": 3, "lr": 1e-3, **{f"train/{k}": v for k, v in tr.items()},
                                 **{f"val/{k}": v for k, v in va.items()}}, step=30)
    line = capsys.readouterr().out
    assert "train loss 1.5000e+00 (energy 2.500e-01, force 1.250e-01)" in line
    assert "val loss 2.0000e+00 (a 5.000e-01, b 1.500e+00)" in line
    assert line.startswith("epoch    3 | ")
    assert _summary({}) == "0.0000e+00"
