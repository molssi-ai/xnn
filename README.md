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
pip install xnns                  # Core (torch, numpy, pyyaml)
pip install "xnns[gnn,ase]"       # + e3nn for NequIP/MACE/Allegro, + ASE calculator
pip install -e ".[all]"           # From a clone: everything, for development
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

The steps below load data, train a model, load pre-trained models, fine-tune
one and deploy it. Each step builds on the ones before it.

### Load data

```python
from xnn.common.data import AtomicDataset, load_dataset

# 250 aspirin conformations of rMD17 (eV, Angstrom) from the data hub
structures = load_dataset("rmd17", molecule="aspirin", split="train", n_train=250)
data = AtomicDataset(structures, cutoff=5.0)
```

- A structure is a plain dictionary: `pos`, `atomic_numbers`, optional `cell`
  and `pbc`, and `energy` / `forces` / `stress` labels.
- Any ASE-readable file works too:
  `AtomicDataset.from_file("trajectory.extxyz", cutoff=5.0)`.

### Train a model

```python
from xnn.common.config import Config
from xnn.common.train import Trainer

# A small NequIP model, checkpoints in runs/nequip (best.pt, last.pt)
cfg = Config()
cfg.model.name = "nequip"
cfg.model.cutoff = cfg.data.cutoff = 5.0
cfg.model.extra = {"species": [1, 6, 8], "l_max": 2}
cfg.optim.epochs = 10
cfg.output_dir = "runs/nequip"
Trainer(cfg, data).fit()
```

The same training runs from the command line:
`xnn train --config configs/train.yaml --set optim.epochs=50`.

### Use pre-trained models

```python
from xnn.common.models import from_pretrained, list_models
from xnn.gnn.models import MEGNet, QET, TensorNet

print(list_models())                                                  # Foundation models and the local cache
mace = from_pretrained("mace-off23-small")                            # MACE-OFF23, converted once and cached
tensornet = TensorNet.from_foundation("tensornet-matpes-pbe-2025.2")  # MatGL's TensorNet trained on MatPES
qet = QET.from_foundation("qet-matpes-pbe-2025.2")                    # MatGL's charge-equilibrated TensorNet
gap = MEGNet.from_foundation("megnet-mp-2019.4.1-bandgap-mfi")        # MatGL's multi-fidelity band-gap MEGNet
mine = from_pretrained("runs/nequip/best.pt")                         # Your own checkpoint
```

Score checkpoints on one dataset with
`xnn benchmark --config configs/benchmark.yaml`. It reports energy, force and
stress errors (MAE, MSE or RMSE, per atom or as atomization energies) per model,
in various formats such as CSV, JSON or Markdown.

### Fine-tune

```python
# MACE-OFF23 small with LoRA adapters, its reference energies re-estimated from the data
ft = Config()
ft.model.name = "mace"
ft.model.cutoff = ft.data.cutoff = 4.5      # The checkpoint's cutoff
ft.model.extra = {"pretrained": "mace-off23-small", "lora": {"rank": 16},
                  "atomic_energies": "estimated"}
ft.optim.epochs = 10
ft.output_dir = "runs/mace-off23-lora"
Trainer(ft, AtomicDataset(structures, cutoff=4.5)).fit()
```

- Full fine-tuning and layer freezing (`optim.freeze`) work the same way.
- Multi-head replay keeps the pre-trained head: `heads: [pt_head, Default]`
  with `data.replay_path` (original or pseudolabelled replay data).

### Deploy

The fine-tuned model (its LoRA updates folded in) can be deployed as an ASE
calculator and used to drive a 20-step MD simulation of aspirin:

```python
from ase import Atoms, units
from ase.md.verlet import VelocityVerlet

from xnn.common.deploy import XNNCalculator

atoms = Atoms(numbers=structures[0]["atomic_numbers"], positions=structures[0]["pos"])
atoms.calc = XNNCalculator.from_pretrained("runs/mace-off23-lora/best.pt")
print(atoms.get_potential_energy(), atoms.get_forces().shape)
VelocityVerlet(atoms, timestep=0.5 * units.fs).run(20)
```

The same model can be deployed as a TorchScript file for LAMMPS (no xnn needed
to run it):

```python
from xnn.common.deploy import export_to_lammps

export_to_lammps(from_pretrained("runs/mace-off23-lora/best.pt", wrap=False), cutoff=4.5,
                 path="mace-off23-lora.pt")
```

Other deployment routes:

- OpenMM (openmm-ml): `MLPotential("xnn",
  model="runs/mace-off23-lora/best.pt")`, adopted alone or as the ML part of an
  ML/MM system.
- An external MD code used through the MolSSI Driver Interface: `xnn mdi --ckpt
  runs/mace-off23-lora/best.pt`.
- Decompose large systems across several GPUs: `DomainParallel` splits one large
  box over the GPUs of a `torchrun` launch

## Models

| Family | Models |
|---|---|
| Generic add-ons (`common`) | LES long-range electrostatics (k-space or particle-mesh sum; optionally charge-constrained or charge-equilibrated; latent charges and dipoles trainable on labels); D3 (zero, BJ, mzero, op damping) and D4 dispersion |
| Graph neural networks (`gnn`) | SchNet, DimeNet, DimeNet++, GemNet (-Q, -T, -dQ, -dT), PaiNN, TensorNet, QET, NequIP, MACE, Allegro, CACE, AIMNet2, MEGNet (single- and multi-fidelity property models) |
| Dense neural networks (`dnn`) | ANI (ANI-1, ANI-1x, ANI-1ccx, ANI-2x), PhysNet, HDNNP (generations 1 to 4), QRNN |
| Convolutional neural networks (`cnn`) | 3D steerable-/ conventional- / spherical-CNN |
| Force field neural networks (`ffnn`) | ReaxFF / ReaxFF-nn, OPLS-AA / L-OPLS, DREIDING / X6, AMBER (ff94), CHARMM22, AMOEBA, HIPPO |
| Transformer neural networks (`trfnn`) | ET (the TorchMD-Net equivariant transformer), Equiformer, EquiformerV2 |
| Hybrid neural networks (`hybnn`) | BAMBOO, SpookyNet, ByteFF, ByteFF-Pol |
| Kernel regression neural networks (`krnn`) | GDML, sGDML |
| Semiempirical quantum mechanics neural networks (`sqmnn`) | PM3, PM6, PM6-ORG, PM7 (classics or trainable) |

Each model is checked against its reference code or paper
(`notebooks/fidelity_checks/`). See the User Guide on [Model
Support](https://molssi-ai.github.io/xnn/user_guide/model_support.html) for full
details on the models and their supported features.

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
  version = {0.28.0},
  url     = {https://github.com/molssi-ai/xnn}
}
```

**Author:** Mohammad Mostafanejad, The Molecular Sciences Software Institute
(MolSSI)
