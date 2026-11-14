"""Distributed smoke test: 2-process CPU DDP through torchrun.

Launches the trainer in two Gloo-backed processes exactly as a user would with
``torchrun --nproc-per-node 2``, covering the distributed code paths the
single-process suite cannot: process-group setup, ``DistributedSampler``
sharding, the double backward through the force loss under DDP, metric
all-reduce, and rank-0-only checkpointing.
"""
import subprocess
import sys

import torch

# The training script run by every rank. The dataset matches the one built by
# tests/common/test_smoke.py; forces are trained (default force_weight > 0), so the
# backward pass goes through the create_graph force gradients under DDP.
_SCRIPT = """
import numpy as np

from xnn.common.config import Config
from xnn.common.data import AtomicDataset
from xnn.common.train import Trainer

rng = np.random.default_rng(0)
structs = [{"pos": rng.uniform(0, 4, (4, 3)), "atomic_numbers": [1, 6, 8, 1],
            "energy": float(rng.normal()), "forces": rng.normal(0, 1, (4, 3))}
           for _ in range(8)]

cfg = Config()
cfg.device = "cpu"
cfg.model.n_features = 16
cfg.model.n_interactions = 1
cfg.optim.epochs = 2
cfg.data.batch_size = 2
cfg.data.val_fraction = 0.25
cfg.output_dir = OUTPUT_DIR

trainer = Trainer(cfg, AtomicDataset(structs, cfg.model.cutoff))
assert trainer.distributed and trainer.device.type == "cpu"
metrics = trainer.fit()
assert metrics["train"]["loss"] > 0 and metrics["val"]["loss"] > 0
"""


def test_ddp_two_cpu_processes(tmp_path):
    script = tmp_path / "train_ddp.py"
    script.write_text(f"OUTPUT_DIR = {str(tmp_path)!r}\n" + _SCRIPT)
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run",
         "--nproc-per-node", "2",
         # c10d rendezvous on port 0 grabs a free port, so parallel test runs
         # on a shared machine do not collide
         "--rdzv-backend", "c10d", "--rdzv-endpoint", "localhost:0",
         str(script)],
        capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr

    # rank 0 wrote both checkpoints, with bare (non-DDP) state-dict keys
    for name in ("best.pt", "last.pt"):
        ckpt = torch.load(tmp_path / name, weights_only=False)
        assert not any(k.startswith("module.") for k in ckpt["model"])

    # only rank 0 logs: one line per epoch plus the initial epoch line count
    assert result.stdout.count("epoch    0") == 1


# multi-head replay under DDP: a batch may lack a head, so the trainer wraps the
# model with find_unused_parameters and the gradient sync must still complete
_SCRIPT_HEADS = """
import numpy as np

from xnn.common.config import Config
from xnn.common.data import AtomicDataset
from xnn.common.train import Trainer

rng = np.random.default_rng(0)
def structs(n, labels=True):
    out = []
    for _ in range(n):
        s = {"pos": rng.uniform(0, 4, (4, 3)), "atomic_numbers": [1, 6, 8, 1]}
        if labels:
            s["energy"] = float(rng.normal()); s["forces"] = rng.normal(0, 1, (4, 3))
        out.append(s)
    return out

cfg = Config()
cfg.device = "cpu"
cfg.model.name = "schnet"
cfg.model.n_features = 16
cfg.model.n_interactions = 1
cfg.model.extra = {"species": [1, 6, 8], "heads": ["pt_head", "Default"]}
cfg.optim.epochs = 2
cfg.optim.head_weights = {"pt_head": {"energy_weight": 1.0}}
cfg.data.batch_size = 1
cfg.data.val_fraction = 0.25
cfg.data.replay_pseudolabel = True
cfg.output_dir = OUTPUT_DIR

trainer = Trainer(cfg, AtomicDataset(structs(8), cfg.model.cutoff),
                  replay_set=AtomicDataset(structs(4, labels=False), cfg.model.cutoff))
assert trainer.distributed and trainer.heads == ["pt_head", "Default"]
metrics = trainer.fit()
assert metrics["train"]["loss"] > 0 and metrics["val"]["loss"] > 0
"""


def test_ddp_multihead_replay(tmp_path):
    script = tmp_path / "train_ddp_heads.py"
    script.write_text(f"OUTPUT_DIR = {str(tmp_path)!r}\n" + _SCRIPT_HEADS)
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run",
         "--nproc-per-node", "2",
         "--rdzv-backend", "c10d", "--rdzv-endpoint", "localhost:0",
         str(script)],
        capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    ckpt = torch.load(tmp_path / "last.pt", weights_only=False)
    assert any(k.startswith("model.extra_heads.0.") for k in ckpt["model"])


# shuffling and dataset-level metrics under DDP: every epoch reshuffles, the
# training shards cover the set, and the val/test metrics equal one batch of
# the whole set. The strided evaluation shards give rank 1 structures 1 and 3,
# neither with forces, so its batches carry no force term at all.
_SCRIPT_METRICS = """
import json
import os

import numpy as np

from xnn.common.config import Config
from xnn.common.data import AtomicDataset, collate
from xnn.common.train import Trainer
from xnn.common.train.losses import weighted_loss

rng = np.random.default_rng(0)
def structs(n, start, with_forces=lambda i: True):
    out = []
    for i in range(n):
        s = {"pos": rng.uniform(0, 4, (4, 3)), "atomic_numbers": [1, 6, 8, 1],
             "energy": float(start + i)}
        if with_forces(i):
            s["forces"] = rng.normal(0, 1, (4, 3))
        out.append(s)
    return out

cfg = Config()
cfg.device = "cpu"
cfg.model.name = "schnet"
cfg.model.n_features = 16
cfg.model.n_interactions = 1
cfg.optim.epochs = 2
cfg.data.batch_size = 2
cfg.data.val_fraction = 0.0
cfg.output_dir = OUTPUT_DIR

cut = cfg.model.cutoff
train = AtomicDataset(structs(9, 0), cut)
val = AtomicDataset(structs(5, 100, lambda i: i not in (1, 3)), cut)
test = AtomicDataset(structs(5, 200, lambda i: i not in (1, 3)), cut)
trainer = Trainer(cfg, train, val, test)
assert trainer.distributed
seen = {}
step = trainer._step
def record(data, train):
    if train:
        epoch = trainer.train_loader.sampler.epoch
        seen.setdefault(epoch, []).extend(round(e) for e in data.energy.tolist())
    return step(data, train)
trainer._step = record
metrics = trainer.fit()

o = cfg.optim
for name, ds in (("val", val), ("test", test)):
    data = collate([ds[i] for i in range(len(ds))])
    _, ref = weighted_loss(trainer.eval_module(data), data,
                           o.energy_weight, o.force_weight, o.stress_weight)
    assert set(ref) <= set(metrics[name]), (name, metrics[name], ref)
    for k in ref:
        assert abs(metrics[name][k] - ref[k]) <= 1e-5 * abs(ref[k]), (name, k, metrics[name][k], ref[k])
# after fit() has ended the process group, evaluate() still scores the whole set
for name, loader in (("val", trainer.val_loader), ("test", trainer.test_loader)):
    after = trainer.evaluate(loader)
    for k, v in metrics[name].items():
        assert abs(after[k] - v) <= 1e-5 * abs(v), (name, k, after[k], v)
with open(os.path.join(OUTPUT_DIR, "seen_" + os.environ["RANK"] + ".json"), "w") as fh:
    json.dump(seen, fh)
"""


def test_ddp_reshuffles_and_reports_dataset_metrics(tmp_path):
    import json

    script = tmp_path / "train_ddp_metrics.py"
    script.write_text(f"OUTPUT_DIR = {str(tmp_path)!r}\n" + _SCRIPT_METRICS)
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run",
         "--nproc-per-node", "2",
         "--rdzv-backend", "c10d", "--rdzv-endpoint", "localhost:0",
         str(script)],
        capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    seen = [json.loads((tmp_path / f"seen_{r}.json").read_text()) for r in (0, 1)]
    for epoch in ("0", "1"):
        both = seen[0][epoch] + seen[1][epoch]
        # DistributedSampler pads 9 structures to 10: one repeats
        assert len(both) == 10 and sorted(set(both)) == list(range(9))
    assert any(s["0"] != s["1"] for s in seen)


# an output head without labels or weight (a PaiNN dipole head trained on
# energies and forces only) leaves its parameters without gradients under DDP
_SCRIPT_UNUSED_HEAD = """
import numpy as np

from xnn.common.config import from_dict
from xnn.common.data import AtomicDataset
from xnn.common.train import Trainer

rng = np.random.default_rng(0)
structs = [{"pos": rng.uniform(0, 4, (4, 3)), "atomic_numbers": [1, 6, 8, 1],
            "energy": float(rng.normal()), "forces": rng.normal(0, 1, (4, 3))}
           for _ in range(8)]
cfg = from_dict({"model": {"name": "painn", "n_features": 8, "n_interactions": 1, "n_rbf": 4,
                           "cutoff": 3.0, "extra": {"dipole": True, "polarizability": True}},
                 "optim": {"epochs": 2},
                 "data": {"batch_size": 2, "val_fraction": 0.25},
                 "device": "cpu", "output_dir": OUTPUT_DIR})
trainer = Trainer(cfg, AtomicDataset(structs, cfg.model.cutoff))
assert trainer.distributed
assert trainer.fit()["train"]["loss"] > 0
"""


def test_ddp_heads_without_labels(tmp_path):
    script = tmp_path / "train_ddp_unused.py"
    script.write_text(f"OUTPUT_DIR = {str(tmp_path)!r}\n" + _SCRIPT_UNUSED_HEAD)
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run",
         "--nproc-per-node", "2",
         "--rdzv-backend", "c10d", "--rdzv-endpoint", "localhost:0",
         str(script)],
        capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
