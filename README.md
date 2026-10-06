# xnn

[![Tests](https://github.com/molssi-ai/xnn/actions/workflows/tests.yml/badge.svg)](https://github.com/molssi-ai/xnn/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/molssi-ai/xnn/graph/badge.svg)](https://codecov.io/gh/molssi-ai/xnn)

Machine-learning interatomic potentials for molecules and materials behind one
PyTorch interface. 

**x** in **xnn** refers to the neural network model architectures: **g** stands
for graph, **d** for dense, **c** for convolutional, **ff** for classical
force-fields, **hybrid** for mixed models, all sharing one data object, one
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
   in the same style add long-range electrostatics (LES) and DFT-D3 / DFT-D4
   dispersion to any model.
4. **One config, three frontends.** `@register_model` exposes a model to the
   YAML, argparse and Hydra interfaces, all of which fill one `Config`
   dataclass.

## Load data

Structures are plain dictionaries (`pos`, `atomic_numbers`, optional `cell`,
`pbc` and `energy` / `forces` / `stress` targets), any ASE-readable file, or a
dataset from the hub:

```python
from xnn.common.data import AtomicDataset, load_dataset

# From a list of structures dict
data = AtomicDataset(structures, cutoff=5.0)

# From a file (e.g., .extxyz, .cif, VASP, ...)
data = AtomicDataset.from_file("trajectory.extxyz", cutoff=5.0)

# From the xnn data hub (e.g., RMD17 dataset)
data = load_dataset("rmd17", molecule="aspirin", split="train", cutoff=5.0)
```

## Train

```python
from xnn.common.config import Config
from xnn.common.train import Trainer

# Initialize the configuration
cfg = Config()

# Set the model and data configuration
cfg.model.name = "nequip"
cfg.model.extra = {"species": [1, 6, 8], "l_max": 2}
cfg.data.batch_size = 16

# Train the model
Trainer(cfg, data).fit()
```

or from the command line: `xnn train --config configs/train.yaml --set optim.epochs=50`.

## Pretrained models

```python
from xnn.common.models import from_pretrained, list_models

# Foundation models, xnn models, local cache
list_models()

# MACE-MP / MACE-OFF, converted once and cached
model = from_pretrained("mace-off23-small")

# Your own checkpoints
model = from_pretrained("runs/exp/best.pt")
```

## Fine-tune

Any pretrained model continues training through the same `Trainer`: naive
fine-tuning, LoRA, multi-head replay (with original or pseudolabelled replay
data), layer freezing and model-aware reference-energy reestimation, all from
the config:

```yaml
model:
  pretrained: mace-off23-small
  cutoff: 5.0
  lora: {rank: 16}                      # or heads: [pt_head, Default] with data.replay_path
  atomic_energies: estimated
```

## Benchmark

Score pre-trained checkpoints on one dataset (energy / force / stress MAE, MSE
or RMSE, per atom or as atomization energies), tabulated per model and written
to CSV, JSON or Markdown:

```bash
xnn benchmark --config configs/benchmark.yaml
```

## Deploy

```python
from xnn.common.deploy import XNNCalculator, export_to_lammps

# Export as ASE calculator
atoms.calc = XNNCalculator(model, cutoff=5.0)

# Export to LAMMPS (TorchScript)
export_to_lammps(model, cutoff=5.0, path="deployed.pt")
```

`xnn mdi --ckpt runs/exp/best.pt` serves a model to an external MD code through
the MolSSI Driver Interface.

## Models

| Family | Models |
|---|---|
| Generic add-ons (`common`) | LES long-range electrostatics; DFT-D3 (zero, BJ, mzero, op damping) and DFT-D4 dispersion |
| Graph neural networks (`gnn`) | SchNet, NequIP, MACE, Allegro, CACE, AIMNet2 |
| Dense neural networks (`dnn`) | ANI (ANI-1, ANI-1x, ANI-1ccx, ANI-2x), PhysNet, HDNNP* |
| Convolutional neural networks (`cnn`) | 3D steerable CNN (SE(3)-equivariant), conventional 3D CNN |
| Force field neural networks (`ffnn`) | ReaxFF / ReaxFF-nn, OPLS-AA / L-OPLS, DREIDING |
| Hybrid neural networks (`hybrid`) | BAMBOO |

\* under development. Each model is checked against its reference code or paper
(`examples/fidelity_checks/`). SchNet, NequIP, MACE and Allegro also export to
TorchScript / LAMMPS; the others deploy through ASE.

## Examples and tests

Executed notebooks for every model are under `examples/`
(`pip install -e ".[examples]"`); `examples/quickstart.py` is a minimal script.
Run the tests with `pytest tests/`.

## License

MIT

## How to cite

If you use xnn in your work, please cite it as:

```bibtex
@software{Mostafanejad:2026:xnn,
  author  = {Mostafanejad, Mohammad},
  title   = {xnn: Machine-learning interatomic potentials for molecules and materials},
  year    = {2026},
  version = {0.8.0},
  url     = {https://github.com/molssi-ai/xnn}
}
```

**Author:** Mohammad Mostafanejad, The Molecular Sciences Software Institute
(MolSSI)
