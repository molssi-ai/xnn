# xnn

[![Tests](https://github.com/molssi-ai/xnn/actions/workflows/tests.yml/badge.svg)](https://github.com/molssi-ai/xnn/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/molssi-ai/xnn/graph/badge.svg)](https://codecov.io/gh/molssi-ai/xnn)
[![Discussions](https://img.shields.io/github/discussions/molssi-ai/xnn)](https://github.com/molssi-ai/xnn/discussions)
[![Issues](https://img.shields.io/github/issues/molssi-ai/xnn)](https://github.com/molssi-ai/xnn/issues)
[![Pull requests](https://img.shields.io/github/issues-pr/molssi-ai/xnn)](https://github.com/molssi-ai/xnn/pulls)

Machine-learning interatomic potentials for molecules and materials behind one
PyTorch interface. 

**x** in **xnn** refers to the neural network model architectures: **g** stands
for graph, **d** for dense, **c** for convolutional, **ff** for classical
force-fields, **sqm** for semi-empirical quantum mechanics, **trf** for
transformers, **hyb** for hybrid models, all sharing one data object, one
trainer and one deployment path. This categorization is reflected in the package
structure.

**Documentation:** https://molssi-ai.github.io/xnn/

## Installation

```
pip install xnns                  # core (torch, numpy, pyyaml)
pip install "xnns[gnn,ase]"       # + e3nn for NequIP/MACE/Allegro, + ASE calculator
pip install -e ".[all]"           # from a clone: everything, for development
```

The distribution is `xnns`; the import name and the command line are `xnn`.

## Design principles

1. **One data object.** Every model takes an `AtomicGraph` and returns
   `{"node_energy", "energy"}`. Periodicity lives only in the edge vectors, so
   molecules and crystals look the same to a model.
2. **Featurizers are first-class.** Descriptors (symmetry functions, AEV) and
   equivariant edge features (spherical harmonics, Cartesian monomials) are
   standalone modules that models compose and you can inspect on their own.
3. **Forces and stress in one place.** `ForceStressOutput` differentiates any
   model's energy for forces and stress; models never implement them. Wrappers
   in the same style add long-range electrostatics (LES) and D3 / D4
   dispersion to any model.
4. **One config, three frontends.** `@register_model` exposes a model to the
   YAML, argparse and Hydra interfaces, all of which fill one `Config`
   dataclass.

## Quick start

One script loads data, trains a model, loads pre-trained models, fine-tunes
one and deploys it. Structures are plain dictionaries (`pos`, `atomic_numbers`, optional
`cell`, `pbc` and `energy` / `forces` / `stress` labels), any ASE-readable
file, or a dataset from the xnn hub:

```python
from ase import Atoms, units
from ase.md.verlet import VelocityVerlet

from xnn.common.config import Config
from xnn.common.data import AtomicDataset, load_dataset
from xnn.common.deploy import XNNCalculator, export_to_lammps
from xnn.common.models import from_pretrained, list_models
from xnn.common.train import Trainer
from xnn.gnn.models import TensorNet

# Load data: 250 aspirin conformations of rMD17 (eV, Angstrom) from the data hub
structures = load_dataset("rmd17", molecule="aspirin", split="train", n_train=250)
data = AtomicDataset(structures, cutoff=5.0)
# or from a list of dicts or a file: 
# AtomicDataset.from_file("trajectory.extxyz", cutoff=5.0)

# Train: a small NequIP model, checkpoints in runs/nequip (best.pt, last.pt)
cfg = Config()
cfg.model.name = "nequip"
cfg.model.cutoff = cfg.data.cutoff = 5.0
cfg.model.extra = {"species": [1, 6, 8], "l_max": 2}
cfg.optim.epochs = 10
cfg.output_dir = "runs/nequip"
Trainer(cfg, data).fit()

# Pre-trained models: foundation models, your own checkpoints, the local cache
print(list_models())
mace = from_pretrained("mace-off23-small")                            # MACE-OFF23, converted once and cached
tensornet = TensorNet.from_foundation("tensornet-matpes-pbe-2025.2")  # MatGL's TensorNet trained on MatPES
mine = from_pretrained("runs/nequip/best.pt")

# Fine-tune: MACE-OFF23 small with LoRA adapters, its reference energies re-estimated from the data
ft = Config()
ft.model.name = "mace"
ft.model.cutoff = ft.data.cutoff = 4.5                                # the checkpoint's cutoff
ft.model.extra = {"pretrained": "mace-off23-small", "lora": {"rank": 16},
                  "atomic_energies": "estimated"}
ft.optim.epochs = 10
ft.output_dir = "runs/mace-off23-lora"
Trainer(ft, AtomicDataset(structures, cutoff=4.5)).fit()

# Deploy: the fine-tuned model (LoRA updates folded in) as an ASE calculator, 20 MD steps of aspirin
atoms = Atoms(numbers=structures[0]["atomic_numbers"], positions=structures[0]["pos"])
atoms.calc = XNNCalculator.from_pretrained("runs/mace-off23-lora/best.pt")
print(atoms.get_potential_energy(), atoms.get_forces().shape)
VelocityVerlet(atoms, timestep=0.5 * units.fs).run(20)

# Deploy: the same model as a TorchScript file for LAMMPS (no xnn needed to run it)
export_to_lammps(from_pretrained("runs/mace-off23-lora/best.pt", wrap=False), cutoff=4.5,
                 path="mace-off23-lora.pt")
```

- The same training runs from the command line:
  `xnn train --config configs/train.yaml --set optim.epochs=50`.
- Fine-tuning also covers full fine-tuning, multi-head replay
  (`heads: [pt_head, Default]` with `data.replay_path`, original or
  pseudolabelled replay data) and layer freezing (`optim.freeze`).
- Score pre-trained checkpoints on one dataset (energy / force / stress MAE, MSE
  or RMSE, per atom or as atomization energies), tabulated per model and written
  to CSV, JSON or Markdown with `xnn benchmark --config configs/benchmark.yaml`
- `MLPotential("xnn", model="runs/mace-off23-lora/best.pt")` runs the model in
  OpenMM (openmm-ml), alone or as the ML part of an ML/MM system.
- `xnn mdi --ckpt runs/mace-off23-lora/best.pt` serves a model to an external MD
  code through the MolSSI Driver Interface.
- `DomainParallel` splits one large box over the GPUs of a `torchrun` launch,
  with the same energies, forces and stress as one GPU.

## Models

| Family | Models |
|---|---|
| Generic add-ons (`common`) | LES long-range electrostatics (k-space or particle-mesh sum; optionally charge-constrained or charge-equilibrated; latent charges and dipoles trainable on labels); D3 (zero, BJ, mzero, op damping) and D4 dispersion |
| Graph neural networks (`gnn`) | SchNet, DimeNet, DimeNet++, GemNet (-Q, -T, -dQ, -dT), PaiNN, TensorNet, NequIP, MACE, Allegro, CACE, AIMNet2 |
| Dense neural networks (`dnn`) | ANI (ANI-1, ANI-1x, ANI-1ccx, ANI-2x), PhysNet, HDNNP (generations 1 to 4) |
| Convolutional neural networks (`cnn`) | 3D steerable-/ conventional- / spherical-CNN |
| Force field neural networks (`ffnn`) | ReaxFF / ReaxFF-nn, OPLS-AA / L-OPLS, DREIDING / X6, AMBER (ff94), CHARMM22, AMOEBA, HIPPO |
| Transformer neural networks (`trfnn`) | ET (the TorchMD-Net equivariant transformer), Equiformer, EquiformerV2 |
| Hybrid neural networks (`hybnn`) | BAMBOO, SpookyNet, ByteFF, ByteFF-Pol |
| Semiempirical quantum mechanics neural networks (`sqmnn`) | PM3, PM6, PM6-ORG, PM7 (classics or trainable) |

Each model is checked against its reference code or paper
(`notebooks/fidelity_checks/`). SchNet, DimeNet, GemNet, PaiNN, TensorNet, ET, Equiformer, EquiformerV2, SpookyNet, NequIP,
MACE, Allegro, CACE, AIMNet2, ANI, PhysNet, HDNNP, BAMBOO, PM3, PM6, PM6-ORG, PM7 and ByteFF-Pol also export to TorchScript / LAMMPS; every model deploys through ASE and the MDI
engine.

## Examples and tests

Executed notebooks for every model are under `notebooks/examples/`
(`pip install -e ".[examples]"`); `notebooks/examples/quickstart.py` is a minimal script.
Run the tests with `pytest tests/`.

## Contributing

Bug reports, fixes and new models are welcome. Read [CONTRIBUTING.md](https://github.com/molssi-ai/xnn/blob/main/CONTRIBUTING.md)
first: it explains how to report a bug with a reproducible example and what a pull request
needs to be merged.

## License

MIT

## How to cite

If you use xnn in your work, please cite it as:

```bibtex
@software{Mostafanejad:2026:xnn,
  author  = {Mostafanejad, Mohammad},
  title   = {xnn: Machine-learning interatomic potentials for molecules and materials},
  year    = {2026},
  version = {0.26.0},
  url     = {https://github.com/molssi-ai/xnn}
}
```

**Author:** Mohammad Mostafanejad, The Molecular Sciences Software Institute
(MolSSI)
