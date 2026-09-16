"""End-to-end smoke example: build data, train, get forces, deploy to ASE.

Runs on CPU in a few seconds with a toy dataset. Demonstrates that the full
pipeline is wired together. Run:  python examples/quickstart.py
"""
import numpy as np
import torch

from xnn.common.config import Config
from xnn.common.data import AtomicDataset
from xnn.common.train import Trainer
from xnn.common.models import build_model, ForceStressOutput, available_models


def toy_structures(n=24, periodic=False):
    """Random small structures with a smooth synthetic energy/forces target."""
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        natoms = rng.integers(3, 6)
        pos = rng.uniform(0, 4, size=(natoms, 3))
        z = rng.choice([1, 6, 8], size=natoms)
        # toy target: pairwise gaussian well (just to have a learnable signal)
        d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
        e = float(-np.exp(-((d - 1.5) ** 2)).sum())
        s = {"pos": pos, "atomic_numbers": z, "energy": e,
             "forces": rng.normal(0, 0.1, size=(natoms, 3))}
        if periodic:
            s["cell"] = np.eye(3) * 6.0
            s["pbc"] = [True, True, True]
        out.append(s)
    return out


def main():
    print("registered models:", available_models())
    cfg = Config()
    cfg.model.name = "schnet"
    cfg.model.cutoff = 5.0
    cfg.model.n_features = 32
    cfg.model.n_interactions = 2
    cfg.data.batch_size = 4          # >1 == batch training; set 1 to disable
    cfg.optim.epochs = 3
    cfg.optim.force_weight = 1.0
    cfg.device = "auto"              # picks cuda if present, else cpu
    cfg.output_dir = "runs/quickstart"

    dataset = AtomicDataset(toy_structures(periodic=False), cfg.data.cutoff)
    trainer = Trainer(cfg, dataset)
    print(f"training on {trainer.device} ...")
    trainer.fit()

    # inference with forces on a single structure (with the trained weights;
    # a freshly built SchNet predicts exactly the energy shift, since its
    # readout head starts zero-initialized per the DTNN convention)
    model = trainer.module.to("cpu").eval()
    g = dataset[0]
    out = model(g)
    print("energy:", round(float(out["energy"].detach()), 4),
          "| forces shape:", tuple(out["forces"].shape))


if __name__ == "__main__":
    main()
