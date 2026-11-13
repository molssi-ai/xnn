"""The Trainer reproduces a plain PyTorch training loop bit for bit, under every launcher."""
import subprocess
import sys

import pytest

# A toy system (water-like triatomics with a harmonic bond potential), a SchNet
# trained twice from the same seed: by a hand-written PyTorch loop and by the
# Trainer. Run as python, torchrun with one process, or torchrun DDP; the
# reference then uses DistributedSampler, DistributedDataParallel and strided
# evaluation shards with an all-reduce of the summed errors.
_SCRIPT = r"""
import copy
import math
import os

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

torch.set_default_dtype(torch.float64)
from xnn.common.config import from_dict
from xnn.common.data import AtomicDataset, collate
from xnn.common.models import ForceStressOutput, build_model
from xnn.common.train import Logger, Trainer

WORLD = int(os.environ.get("WORLD_SIZE", "1"))
DDP = WORLD > 1
if DDP:
    dist.init_process_group("gloo")
RANK = dist.get_rank() if DDP else 0
CUT = 3.0


def water(n, seed):
    rng = np.random.default_rng(seed)
    th = np.deg2rad(104.5)
    base = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.96 * np.cos(th), 0.96 * np.sin(th), 0.0]])
    bonds = {(0, 1): (0.96, 40.0), (0, 2): (0.96, 40.0), (1, 2): (1.52, 10.0)}
    out = []
    for _ in range(n):
        pos = base + rng.normal(scale=0.08, size=(3, 3))
        e, f = 0.0, np.zeros((3, 3))
        for (i, j), (r0, k) in bonds.items():
            v = pos[i] - pos[j]
            r = np.linalg.norm(v)
            e += 0.5 * k * (r - r0) ** 2
            g = k * (r - r0) * v / r
            f[i] -= g
            f[j] += g
        out.append({"pos": pos, "atomic_numbers": [8, 1, 1], "energy": e - 3.0, "forces": f})
    return out


TRAIN, VAL, TEST = water(22, 0), water(7, 1), water(5, 2)


class Sums:
    # dataset-level sums: squared and absolute errors and their counts
    def __init__(self):
        self.v = {}

    def add(self, key, value, count):
        s, c = self.v.get(key, (0.0, 0.0))
        self.v[key] = (s + float(value), c + float(count))

    def means(self):
        keys = sorted(self.v)
        flat = [self.v[k][0] for k in keys] + [self.v[k][1] for k in keys]
        if DDP:
            t = torch.tensor(flat, dtype=torch.float64)
            dist.all_reduce(t)
            flat = t.tolist()
        return {k: flat[i] / flat[len(keys) + i] for i, k in enumerate(keys)}


def batch_loss(o, pred, data):
    n = data.n_atoms.to(pred["energy"].dtype)
    de = (pred["energy"] - data.energy) / n
    df = pred["forces"] - data.forces
    e_mse, f_mse = (de ** 2).mean(), (df ** 2).mean()
    return o.energy_weight * e_mse + o.force_weight * f_mse, de.detach(), df.detach()


def score(sums, de, df):
    sums.add("energy_se", (de ** 2).sum(), de.numel())
    sums.add("energy_ae", de.abs().sum(), de.numel())
    sums.add("forces_se", (df ** 2).sum(), df.numel())
    sums.add("forces_ae", df.abs().sum(), df.numel())


def summary(o, sums):
    m = sums.means()
    out = {"energy_mse": m["energy_se"], "force_mse": m["forces_se"],
           "energy_mae": m["energy_ae"], "energy_rmse": math.sqrt(m["energy_se"]),
           "forces_mae": m["forces_ae"], "forces_rmse": math.sqrt(m["forces_se"])}
    out["loss"] = o.energy_weight * out["energy_mse"] + o.force_weight * out["force_mse"]
    return out


def evaluate(o, model, loader):
    sums = Sums()
    for data in loader:
        _, de, df = batch_loss(o, model(data), data)
        score(sums, de, df)
    return summary(o, sums)


def reference(cfg):
    o, bs = cfg.optim, cfg.data.batch_size
    train, val, test = (AtomicDataset(s, CUT) for s in (TRAIN, VAL, TEST))
    torch.manual_seed(cfg.seed)
    model = ForceStressOutput(build_model(cfg.model), compute_forces=True, compute_stress=False)
    init = copy.deepcopy(model.state_dict())
    net = DistributedDataParallel(model) if DDP else model
    if DDP:
        sampler = DistributedSampler(train, shuffle=True, seed=cfg.seed)
        train_loader = DataLoader(train, batch_size=bs, sampler=sampler, collate_fn=collate)
        shard = lambda ds: range(RANK, len(ds), WORLD)
    else:
        train_loader = DataLoader(train, batch_size=bs, shuffle=True, collate_fn=collate)
        shard = lambda ds: None
    val_loader = DataLoader(val, batch_size=bs, sampler=shard(val), collate_fn=collate)
    test_loader = DataLoader(test, batch_size=bs, sampler=shard(test), collate_fn=collate)
    params = list(net.parameters())
    opt = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[o.optimizer](
        params, lr=o.lr, weight_decay=o.weight_decay)
    if o.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=o.epochs)
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10)
    ema, shadow, n_avg = None, None, 0
    if o.ema_decay > 0:
        ema = copy.deepcopy(model)
        shadow = [p.detach().clone() for p in ema.parameters()]

    best, best_state, epochs = math.inf, None, []
    for epoch in range(o.epochs):
        if DDP:
            sampler.set_epoch(epoch)
        lr = opt.param_groups[0]["lr"]
        net.train()
        sums = Sums()
        for data in train_loader:
            loss, de, df = batch_loss(o, net(data), data)
            opt.zero_grad()
            loss.backward()
            if o.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(params, o.clip_grad)
            opt.step()
            if ema is not None:
                d = o.ema_decay if n_avg > 0 else 0.0
                with torch.no_grad():
                    for s, a, p in zip(shadow, ema.parameters(), model.parameters()):
                        s.mul_(d).add_(p.detach(), alpha=1.0 - d)
                        a.copy_(s)
                n_avg += 1
            score(sums, de, df)
        tr = summary(o, sums)
        evaluated = ema if ema is not None else model
        net.eval()
        va = evaluate(o, evaluated, val_loader)
        if va["loss"] < best:
            best, best_state = va["loss"], copy.deepcopy(evaluated.state_dict())
        if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sched.step(va["loss"])
        else:
            sched.step()
        epochs.append({"lr": lr, "train": tr, "val": va})
    evaluated = ema if ema is not None else model
    return {"init": init, "model": model, "ema": ema, "epochs": epochs, "best": best_state,
            "last": copy.deepcopy(evaluated.state_dict()),
            "test": evaluate(o, evaluated, test_loader)}


class Record(Logger):
    def __init__(self):
        self.epochs = []

    def log_metrics(self, metrics, step):
        if "epoch" in metrics:
            self.epochs.append(dict(metrics))


def same_tensors(a, b):
    return all(torch.equal(a[k], b[k]) for k in a) and set(a) == set(b)


def close(a, b, what):
    for k, v in b.items():
        assert math.isclose(a[k], v, rel_tol=1e-12, abs_tol=0.0), (what, k, a[k], v)


CONFIGS = {
    "adam-cosine": {"lr": 5e-3, "scheduler": "cosine"},
    "adamw-plateau-clip-ema": {"lr": 5e-3, "scheduler": "plateau", "optimizer": "adamw",
                               "weight_decay": 1e-2, "clip_grad": 0.5, "ema_decay": 0.9},
}
for name, optim in CONFIGS.items():
    cfg = from_dict({"model": {"name": "schnet", "n_features": 16, "n_interactions": 2,
                               "cutoff": CUT},
                     "optim": {"epochs": 6, "energy_weight": 1.0, "force_weight": 10.0, **optim},
                     "data": {"batch_size": 4},
                     "device": "cpu", "seed": 7, "output_dir": os.path.join(OUTPUT_DIR, name)})
    ref = reference(cfg)

    rec = Record()
    trainer = Trainer(cfg, AtomicDataset(TRAIN, CUT), AtomicDataset(VAL, CUT),
                      AtomicDataset(TEST, CUT), logger=rec)
    assert same_tensors(trainer.module.state_dict(), ref["init"]), (name, "initial weights")
    out = trainer.fit()

    assert same_tensors(trainer.module.state_dict(), ref["model"].state_dict()), (name, "final weights")
    if ref["ema"] is not None:
        assert same_tensors(trainer.eval_module.state_dict(), ref["ema"].state_dict()), (name, "EMA")
    if RANK == 0:
        assert len(rec.epochs) == len(ref["epochs"])
    for e, r in enumerate(ref["epochs"]):
        if RANK == 0:
            got = rec.epochs[e]
            assert got["lr"] == r["lr"], (name, e, got["lr"], r["lr"])
            close({k[6:]: v for k, v in got.items() if k.startswith("train/")}, r["train"], (name, e, "train"))
            close({k[4:]: v for k, v in got.items() if k.startswith("val/")}, r["val"], (name, e, "val"))
    close(out["val"], ref["epochs"][-1]["val"], (name, "fit val"))
    close(out["test"], ref["test"], (name, "test"))
    if RANK == 0:
        for f, key in (("best.pt", "best"), ("last.pt", "last")):
            ckpt = torch.load(os.path.join(cfg.output_dir, f), weights_only=False)
            assert same_tensors(ckpt["model"], ref[key]), (name, f)
        for e, r in enumerate(ref["epochs"]):
            got = rec.epochs[e]
            print(f"{name} epoch {e} lr {r['lr']:.6e} | train loss ref {r['train']['loss']:.15e} "
                  f"xnn {got['train/loss']:.15e} | val loss ref {r['val']['loss']:.15e} "
                  f"xnn {got['val/loss']:.15e}", flush=True)
    print(f"PARITY OK {name} rank {RANK}", flush=True)
if DDP:
    dist.destroy_process_group()
"""

_TORCHRUN = [sys.executable, "-m", "torch.distributed.run",
             "--rdzv-backend", "c10d", "--rdzv-endpoint", "localhost:0"]


@pytest.mark.parametrize("launch,world", [([sys.executable], 1),
                                          (_TORCHRUN + ["--nproc-per-node", "1"], 1),
                                          (_TORCHRUN + ["--nproc-per-node", "2"], 2)],
                         ids=["python", "torchrun-1", "ddp-2"])
def test_trainer_matches_a_plain_pytorch_loop(tmp_path, launch, world):
    """Weights, EMA, learning rates, checkpoints and dataset-level metrics equal a hand-written loop."""
    script = tmp_path / "parity.py"
    script.write_text(f"OUTPUT_DIR = {str(tmp_path)!r}\n" + _SCRIPT)
    result = subprocess.run(launch + [str(script)], capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout)
    assert result.stdout.count("PARITY OK") == 2 * world
