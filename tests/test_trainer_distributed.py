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
# tests/test_smoke.py; forces are trained (default force_weight > 0), so the
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
