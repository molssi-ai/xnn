# xnns

Machine-learning interatomic potentials in PyTorch, for **molecular and
periodic** systems behind a single coherent PyTorch interface.

## Installation

```
pip install -e .            # core (torch, numpy, pyyaml)
pip install -e ".[ase]"     # + ASE calculator
pip install -e ".[gnn]"     # + e3nn for NequIP/MACE/Allegro
pip install -e ".[hydra]"   # + Hydra/OmegaConf config
pip install -e ".[examples]" # + ASE, e3nn, mace-torch, nequip, jupyter (runs the notebooks)
pip install -e ".[all]"
```

The `examples` notebooks validate xnns models against their reference
implementations. For example, the MACE implementation in xnns is validated
against that of [ACEsuit/mace](https://github.com/ACEsuit/mace) (`mace-torch`),
the NequIP implementation is validated against that of
[mir-group/nequip](https://github.com/mir-group/nequip) and Allegro is validated
against [mir-group/allegro](https://github.com/mir-group/allegro), which pin
`e3nn==0.4.4`; xnns has been thoroughly tested on this pin. The BAMBOO graph
equivariant transformer is validated block-by-block against
[bytedance/bamboo](https://github.com/bytedance/bamboo). SchNet is built and
verified directly against the manuscripts' equations (block-by-block, in
`examples/fidelity_checks/schnet_verification.ipynb`). The `pyproject.toml` also
carries a `uv` setup that reproduces the GPU `.venv` that was used to create the
notebooks (we adopted `torch 2.5.1+cu121` from the PyTorch cu121 index that are
compatible with CUDA-12.x drivers).

## Quick start

```bash
# builds trains a model on toy data and predicts on a test set
python examples/quickstart.py

# smoke tests for all models and families
pytest tests/
```

```python
from xnns.common.config import Config
from xnns.common.data import AtomicDataset
from xnns.common.train import Trainer

cfg = Config()
# E.g. schnet | hdnnp | ani | physnet | nequip | mace | allegro | cace
cfg.model.name = "nequip"
# Unshared model specific hyperparameters go here
cfg.model.extra = {"species": [1, 6, 8], "l_max": 2}
# batch_size = 1 disables batch training
cfg.data.batch_size = 16             
# auto | cpu | cuda | cuda:0
cfg.device = "auto"
Trainer(cfg, AtomicDataset(structures, cfg.model.cutoff)).fit()
```

`structures` is a list of plain dictionaries (`pos`, `atomic_numbers`, and
optionally, `cell`/`pbc` and `energy`/`forces`/`stress` targets). Data in any
ASE-native format loads directly and targets can also be included. So, no
pre-wrapping is required:

```python
# xnns supports ase-io native formats such as .extxyz, .cif, VASP, ...
train_set = AtomicDataset.from_file("trajectory.extxyz", cutoff=4.0)
```

## Package layout

The package is organized **by model family** (`gnn`, `cnn`, `dnn`, `hybrid`),
with shared resources factored into the `common` modules and reusable
transformer building blocks in `transformer`. An object (e.g., function, module
etc.) lives with the model family that uses it, or with `common` if more than
one family needs it. Of course, layers are designed as stand-alone entities and
can be imported on their own.

```
src/xnns/
├── __main__.py                 `python -m xnns` entry point
├── common/                     shared across all model families
│   ├── data/                   - common data abstractions
│   │   └── …                     + AtomicGraph (the one data object), PBC neighbor list, AtomicDataset, ASE I/O
│   ├── featurizers/            - common featurizers
│   │   └── …                     + Featurizer base + shared basis functions (GaussianRBF, CosineCutoff)
│   ├── config/                 - one dataclass schema; loaders for yaml / argparse / hydra
│   │   └── …                     + schema, loaders, translate, coerce
│   ├── models/                 - InteratomicPotential interface, registry, ForceStressOutput, ops (scatter_sum, shifted_softplus)
│   │   └── …                     + base, registry, outputs, les, ops
│   ├── train/                  - Trainer (batch + device aware), weighted energy/force/stress loss
│   │   └── …                     + trainer, losses
│   ├── benchmark/              - score pre-trained models on a dataset (metrics, atomization energy, report writers)
│   │   └── …                     + config, runner, metrics, energy, report
│   ├── deploy/                 - ASE Calculator, and LAMMPS/TorchScript export
│   │   └── …                     + ase_calculator, lammps
│   └── cli/                    - the `xnns` command-line interface
│       └── main.py
├── gnn/                        graph potentials
│   ├── featurizers/            - GNN featurizers
│   │   └── …                     + spherical, cartesian, radial, cutoff
│   └── models/                 - base (GNNPotential, EquivariantGNN), blocks, nequip, mace, allegro, cace
│       └── …                     + base, blocks, nequip, mace, allegro, cace
├── cnn/                        continuous-filter conv net
│   └── models/                 schnet
├── dnn/                        descriptor + per-element networks, and PhysNet
│   ├── featurizers/            - DNN featurizers
│   │   └── …                     + symmetry functions, AEV
│   └── models/                 base (DescriptorPotential), hdnnp, ani, physnet and ported Grimme's D3
│       └── …
├── transformer/                shared graph-transformer building blocks
│   ├── attention.py            - EdgeMultiheadAttention (multi-head QKV attention on edges)
│   └── featurizers/            - ExpNormalSmearing radial basis
└── hybrid/                     GNN + transformer potentials with a physics energy split
    └── models/                 - bamboo (BAMBOO graph equivariant transformer), dispersion (D3(CSO))
        └── …
```

```python
# Data (structures dict -> AtomicGraph)
from xnns.common.data import AtomicDataset, build_neighbor_list
ds = AtomicDataset(structures, cutoff=5.0)
graph = ds[0]

# Featurizers (AtomicGraph -> model inputs)
from xnns.dnn.featurizers import AEV, RadialSymmetryFunctions
from xnns.gnn.featurizers import SphericalHarmonicEdgeEmbedding
# (N, D) invariant per-atom AEV
descriptor = AEV(species=[1, 6, 8])(graph)
# Equivariant edge attributes
edges = SphericalHarmonicEdgeEmbedding(l_max=2)(graph)

# Models (model inputs -> energy)
from xnns.common.models import build_model, ForceStressOutput, available_models
# Any registered model + autograd forces/stress
model = ForceStressOutput(build_model(cfg.model))
```

Four ideas hold xnns together:

1. **Unified data object.** Every model consumes an `AtomicGraph` and returns
   `{"node_energy", "energy"}`. Molecular vs. periodic is invisible to models:
   periodicity lives only in `edge_vectors()` (`r_ij = pos[dst] - pos[src] +
   cell_shift @ cell`), keeping forces and stress differentiable.
2. **Featurizers are first-class.** A `Featurizer` (subclass of `nn.Module`)
   turns a graph into invariant descriptors (symmetry functions, AEV) or
   equivariant edge attributes (spherical harmonics, Cartesian monomials).
   Descriptor models (HDNNP, ANI) and GNNs (NequIP/MACE/Allegro/CACE) are thin
   compositions over featurizers, so the featurization is reusable and
   inspectable on its own.
3. **Forces/stress in one place.** `ForceStressOutput` wraps any model and
   differentiates energy w.r.t. positions (forces) and a symmetric strain
   (stress). Models never implement them. The same wrapper idea powers
   `LatentEwald` (Latent Ewald Summation, Cheng 2025): every model exposes
   invariant `"node_features"`, so long-range electrostatics/dispersion can be
   added to *any* short-range model with `extra: {long_range: {...}}` — a
   faithful port of the CACE-LR reference implementation (see
   `tests/test_les.py` and `examples/gnn/les/`).
4. **Extensibility via registry + one config, three frontends.**
   `@register_model("name")` + a `from_config` classmethod makes a model usable
   from any of YAML / argparse / Hydra, which all funnel into one
   `Config` dataclass.

## Config frontends (interchangeable)

```python
from xnns.common.config import from_yaml, from_argparse, from_hydra
cfg = from_yaml("configs/train.yaml")
cfg = from_argparse(["--config", "configs/train.yaml", "--set", "model.cutoff=6.0"])
```

CLI: `xnns train --config configs/train.yaml --set optim.epochs=50`
(also `xnns benchmark --config configs/benchmark.yaml` and
`xnns export --config ... --ckpt ... --to lammps|torchscript`).

Model keys copied verbatim from an upstream code's yaml also work: a per-model
key-translation registry (`xnns.common.config.translate`) rewrites the foreign
spellings (MACE-CLI `r_max`/`num_radial_basis`/`atomic_numbers`/`E0s`, NequIP
`num_layers`, ...) to the xnns canonical names at config-load time; the xnns
spelling wins if both are given. Extend it for another code with
`register_key_translation("name", {...})`.

## Deployment

```python
from xnns.common.deploy import XNNSCalculator, export_to_lammps
atoms.calc = XNNSCalculator(model, cutoff=5.0)          # ASE
export_to_lammps(model, cutoff=5.0, path="deployed.pt") # TorchScript for LAMMPS
```

Pair the exported `.pt` with the matching C++ pair style (pair_nequip /
pair_mace / pair_allegro pattern). The `LAMMPSWrapper` in
`common/deploy/lammps.py` defines the tensor ABI. A model is exportable when it
provides the scriptable `node_energy(atomic_numbers, edge_index, edge_vec)`
core -- SchNet, NequIP, MACE and Allegro all do (the scripted models reproduce
the eager ones to ~1e-15, verified in `tests/test_schnet.py` /
`tests/test_mace.py` / `tests/test_nequip.py` / `tests/test_allegro.py`). For NequIP this required a scriptable, bit-exact stand-in
for e3nn's `Gate` (`xnns.gnn.models.nequip._Gate`), which the e3nn 0.4.4
original cannot do on torch 2.x.

## Benchmarking

Score a set of **pre-trained** models on one dataset with
`xnns.common.benchmark` and tabulate their errors. A single config lists the
`models` (each an architecture plus the `checkpoint` to load), the error
`metrics` (`mae` / `mse` / `rmse`, or custom callables), and the `targets` to
score (`energy` / `forces` / `stress`). Results are tabulated per model and
written to CSV / JSON / Markdown (or a user-registered format). Energy can be
scored per atom or, with `atomic_energies` (a `{Z: E0}` map or `average` to fit
from data), as the physically meaningful atomization (interaction) energy.
Benchmarking does not train — produce the checkpoints first with `xnns train`.

```bash
xnns benchmark --config configs/benchmark.yaml
```

```python
from xnns.common.benchmark import from_yaml, run_benchmark
rows = run_benchmark(from_yaml("configs/benchmark.yaml"))
```

The benchmark builds models with the same `Config` and model registry as a
single run; new metrics and output formats plug in via `@register_metric` and
`@register_writer`, mirroring `@register_model`.

## Models and fidelity

| Model | Family | Featurizers | State |
|---|---|---|---|
| SchNet | cnn | Gaussian RBF | Under development |
| PhysNet | dnn | exp-Gaussian rbf + attention masks | Complete: Training, Evaluation, Deployment (ASE only) |
| HDNNP | dnn | radial symmetry functions (G2) | Under development |
| ANI | dnn | AEV (radial + angular symmetry functions) | Complete: Training, Evaluation, Deployment (ASE only) |
| NequIP | gnn | spherical-harmonic edges | Complete: Training, Evaluation, Deployment (TorchScript, LAMMPS, ASE) |
| MACE | gnn | spherical-harmonic edges | Complete: Training, Evaluation, Deployment (TorchScript, LAMMPS, ASE) |
| CACE | gnn | Cartesian monomial edges | Complete: Training, Evaluation, Deployment (ASE only) |
| Allegro | gnn | spherical-harmonic edges | Complete: Training, Evaluation, Deployment (TorchScript, LAMMPS, ASE) |
| BAMBOO | hybrid | exp-normal rbf + multi-head edge attention | Complete: Training, Evaluation, Deployment (ASE only) |

## Examples

Runnable notebooks are grouped into separate directories based on model family
types within `examples/<xnn>` (`pip install -e ".[examples]"`) where `x` refers
to the architecture types (e.g., `g` in `gnn` for graph neural networks, `d` in
`dnn` for deep neural networks, and `c` in `cnn` for convolutional neural
networks): each holds a `<model>_argon_train_test.ipynb` and a
`<model>_argon_density_md.ipynb` (LES has `les_molecular_dimers.ipynb`; ANI has
`examples/dnn/ani/ani_rmd17_train.ipynb` and `ani1_dataset.ipynb`). Every
training/MD notebook pulls its data through the one-line dataset hub
(`load_dataset("argon_md")`, `load_dataset("rmd17", ...)`,
`load_dataset("ani1", ...)`, `load_dataset("lode_dimers", subset="bio_scan")`).
The block-by-block numerical verifications against the upstream codes are
collected under `examples/fidelity_checks/` as `<model>_verification.ipynb`. The
`examples/quickstart.py` module presents a minimal train/predict workflow on
toy-data.

## Extension points

- **Implement a new model:** 
  + Pick your model family package (`gnn` / `cnn` / `dnn`,  or add one) and add
    a module under `<family>/models/`.
  + Subclass `xnns.common.models.InteratomicPotential` (or the family base, e.g.,
  `gnn.models.base.EquivariantGNN`) and implement its `forward(data)` method.
  + register your model implementation using `@register_model` decorator.
  + Add a YAML config file for your model `configs/model/<name>.yaml`. 
  + Import the family package so the model registers.
  + For TorchScript/LAMMPS export, it is important to expose a scriptable
  `node_energy(atomic_numbers, edge_index, edge_vec)` core (SchNet shows the
  pattern; e3nn models need e3nn's JIT support for this).
- **Add a new featurizer:**
  + Subclass `xnns.common.featurizers.Featurizer` and implement `output_dim` and
  `forward(data)`. Put the resulting featurizer module in the
  `common/featurizers/` if shared by more than one model family or under the
  using family's `featurizers/` if it is only used by that one model family.
