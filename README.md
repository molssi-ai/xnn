# xnn

Machine-learning interatomic potentials for **molecules and materials** behind a
single coherent PyTorch interface.

## Installation

From PyPI, where the distribution is called `xnns` (the import name and the
command line are `xnn`):

```
pip install xnns                 # core (torch, numpy, pyyaml)
pip install "xnns[gnn,ase]"      # + e3nn for NequIP/MACE/Allegro, + ASE calculator
```

From a clone, for development:

```
pip install -e .             # core (torch, numpy, pyyaml)
pip install -e ".[ase]"      # + ASE calculator
pip install -e ".[gnn]"      # + e3nn for NequIP/MACE/Allegro
pip install -e ".[hydra]"    # + Hydra/OmegaConf config
pip install -e ".[examples]" # + ASE, e3nn, mace-torch, nequip, jupyter (runs the notebooks)
pip install -e ".[all]"
```

The Allegro reference implementation used by one fidelity notebook is not on
PyPI; install it separately with
`pip install "git+https://github.com/mir-group/allegro@v0.3.0"`.

## Quick start

```bash
# builds trains a model on toy data and predicts on a test set
python examples/quickstart.py

# smoke tests for all models and families
pytest tests/
```

```python
from xnn.common.config import Config
from xnn.common.data import AtomicDataset
from xnn.common.train import Trainer

cfg = Config()
# E.g. schnet | hdnnp | ani | physnet | nequip | mace | allegro | cace | reaxff
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
# xnn supports ase-io native formats such as .extxyz, .cif, VASP, ...
train_set = AtomicDataset.from_file("trajectory.extxyz", cutoff=4.0)
```

## Design principles

Four ideas hold `xnn` together:

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


```python
# Data (structures dict -> AtomicGraph)
from xnn.common.data import AtomicDataset, build_neighbor_list
ds = AtomicDataset(structures, cutoff=5.0)
graph = ds[0]

# Featurizers (AtomicGraph -> model inputs)
from xnn.dnn.featurizers import AEV, RadialSymmetryFunctions
from xnn.gnn.featurizers import SphericalHarmonicEdgeEmbedding
# (N, D) invariant per-atom AEV
descriptor = AEV(species=[1, 6, 8])(graph)
# Equivariant edge attributes
edges = SphericalHarmonicEdgeEmbedding(l_max=2)(graph)

# Models (model inputs -> energy)
from xnn.common.models import build_model, ForceStressOutput, available_models
# Any registered model + autograd forces/stress
model = ForceStressOutput(build_model(cfg.model))
```

## Config frontends (interchangeable)

```python
from xnn.common.config import from_yaml, from_argparse, from_hydra
cfg = from_yaml("configs/train.yaml")
cfg = from_argparse(["--config", "configs/train.yaml", "--set", "model.cutoff=6.0"])
```

CLI: `xnn train --config configs/train.yaml --set optim.epochs=50`
(also `xnn benchmark --config configs/benchmark.yaml` and
`xnn export --config ... --ckpt ... --to lammps|torchscript`).

Model keys copied verbatim from an upstream code's yaml also work: a per-model
key-translation registry (`xnn.common.config.translate`) rewrites the foreign
spellings (MACE-CLI `r_max`/`num_radial_basis`/`atomic_numbers`/`E0s`, NequIP
`num_layers`, ...) to the xnn canonical names at config-load time; the xnn
spelling wins if both are given. Extend it for another code with
`register_key_translation("name", {...})`.

## Deployment

```python
from xnn.common.deploy import XNNCalculator, export_to_lammps
atoms.calc = XNNCalculator(model, cutoff=5.0)          # ASE
export_to_lammps(model, cutoff=5.0, path="deployed.pt") # TorchScript for LAMMPS
```

Pair the exported `.pt` with the matching C++ pair style (pair_nequip /
pair_mace / pair_allegro pattern). The `LAMMPSWrapper` in
`common/deploy/lammps.py` defines the tensor ABI. A model is exportable when it
provides the scriptable `node_energy(atomic_numbers, edge_index, edge_vec)`
core -- SchNet, NequIP, MACE and Allegro all do (the scripted models reproduce
the eager ones to ~1e-15, verified in `tests/test_schnet.py` /
`tests/test_mace.py` / `tests/test_nequip.py` / `tests/test_allegro.py`). For NequIP this required a scriptable, bit-exact stand-in
for e3nn's `Gate` (`xnn.gnn.models.nequip._Gate`), which the e3nn 0.4.4
original cannot do on torch 2.x.

## Benchmarking

Score a set of **pre-trained** models on one dataset with
`xnn.common.benchmark` and tabulate their errors. A single config lists the
`models` (each an architecture plus the `checkpoint` to load) and the
`metrics` mapping, which ties each target quantity (`energy` / `forces` /
`stress`) to the error metrics reported for it (`mae` / `mse` / `rmse`, or
custom callables). Results are tabulated per model and
written to CSV / JSON / Markdown (or a user-registered format). Energy can be
scored per atom or, with `atomic_energies` (a `{Z: E0}` map or `average` to fit
from data), as the physically meaningful atomization (interaction) energy.
Benchmarking does not train — produce the checkpoints first with `xnn train`.

```bash
xnn benchmark --config configs/benchmark.yaml
```

```python
from xnn.common.benchmark import from_yaml, run_benchmark
rows = run_benchmark(from_yaml("configs/benchmark.yaml"))
```

The benchmark builds models with the same `Config` and model registry as a
single run; new metrics and output formats plug in via `@register_metric` and
`@register_writer`, mirroring `@register_model`.

## Available models

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
| ReaxFF / ReaxFF-nn | ffnn | bond orders + EEM charges (the force field is the model) | Complete: Training, Evaluation, Deployment (ASE only) |
| OPLS / OPLS-AA / L-OPLS | ffnn | fixed valence topology (the force field is the model) | Complete: Training, Evaluation, Deployment (ASE only) |


<details> <!-- Start Package layout -->
<summary><h2 style="display:inline-block">Package layout</h2></summary>

The package is organized **by model family** (`gnn`, `cnn`, `dnn`, `ffnn`, `hybrid`),
with shared resources factored into the `common` modules and reusable
transformer building blocks in `transformer`. An object (e.g., function, module
etc.) lives with the model family that uses it, or with `common` if more than
one family needs it. Of course, layers are designed as stand-alone entities and
can be imported on their own.

```
src/xnn/
├── __main__.py                 `python -m xnn` entry point
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
│   └── cli/                    - the `xnn` command-line interface
│       └── main.py
├── gnn/                        graph potentials
│   ├── featurizers/            - GNN featurizers
│   │   └── …                     + spherical, cartesian, radial, cutoff
│   └── models/                 - base (GNNPotential, EquivariantGNN), blocks, nequip, mace, allegro, cace
│       └── …                     + base, blocks, nequip, mace, allegro, cace
├── cnn/                        continuous-filter conv net
│   └── models/                 - schnet
├── dnn/                        descriptor + per-element networks, and PhysNet
│   ├── featurizers/            - DNN featurizers
│   │   └── …                     + symmetry functions, AEV
│   └── models/                 base (DescriptorPotential), hdnnp, ani, physnet and ported Grimme's D3
│       └── …
├── ffnn/                       learnable classical force fields
│   ├── common/                 - frc (SEAMM .frc force-field files: reader, resolver, writer, registry),
│   │                             typing (SMARTS atom typing), elements
│   ├── data/                   - shipped parameter files: oplsaa.frc, lopls.frc, oplsaa_1996.frc, reaxff/*.frc
│   └── models/                 - reaxff, ffield, opls, oplslib, topology
│       └── …                     + ReaxFF / ReaxFF-nn reactive force field, OPLS / L-OPLS fixed-topology
│                                   force field, the .frc <-> model parameter bridges
├── transformer/                shared graph-transformer building blocks
│   ├── attention.py            - EdgeMultiheadAttention (multi-head QKV attention on edges)
│   └── featurizers/            - ExpNormalSmearing radial basis
└── hybrid/                     GNN + transformer potentials with a physics energy split
    └── models/                 - bamboo (BAMBOO graph equivariant transformer), dispersion (D3(CSO))
        └── …
```

</details> <!-- End Package layout -->

<details> <!-- Start Examples -->
<summary><h2 style="display:inline-block">Examples</h2></summary>

Runnable, pre-executed notebooks live under `examples/` (`pip install -e
".[examples]"`). They are grouped by model family, where the leading letter of
`<xnn>` names the architecture type (`gnn` graph, `cnn` convolutional, `dnn`
deep/descriptor, `ffnn` force-field, `hybrid` mixed):

- **Per-model training and MD** (`examples/<xnn>/<model>/`):
  + `<model>_argon_train_test.ipynb` and `<model>_argon_density_md.ipynb` for
    MACE, NequIP, Allegro, CACE and PhysNet.
  + `schnet_rmd17_train.ipynb` and `schnet_ethanol_md.ipynb` for SchNet.
  + `ani_rmd17_train.ipynb` plus the `ani1*_dataset.ipynb` / `ani2x_dataset.ipynb`
    dataset walk-throughs for ANI.
  + `les_molecular_dimers.ipynb` for the LES long-range wrapper (charged and
    polar dimers) and `recreate_mace_architecture.ipynb` for a block-by-block
    MACE rebuild.
  + `bamboo_dimer_electrostatics.ipynb` and `bamboo_charge_analysis.ipynb` for
    the hybrid BAMBOO model.
- **Force fields** (`examples/ffnn/`):
  + `reaxff/reaxff_rmd17_train_test.ipynb` trains the ReaxFF-nn reactive force
    field on rMD17 and `reaxff/reaxff_md_bond_orders.ipynb` analyses its bond
    orders, EEM charges and bond dissociation in MD. Both benchmark against the
    published classical Chenoweth 2008 C/H/O field, loaded from the shipped
    SEAMM `.frc` file (`ReaxFF("CHO_cho_2008")`).
  + `opls/opls_conformational_energetics.ipynb` reproduces Table 1 of the 1996
    OPLS-AA paper with a relaxed dihedral driver, typing every molecule from
    coordinates with the SMARTS templates of `oplsaa.frc`.
  + `opls/opls_lopls_torsion_refit.ipynb` re-derives the L-OPLS hydrocarbon
    torsion refit of Siu et al. (2012) by gradient descent
    (`trainable=("dihedral_v",)`) and writes the trained field back out as a
    `.frc` file.
- **Data and deployment:**
  + `data/load_dataset_tutorial.ipynb` covers the one-line dataset hub used by
    every training notebook (`load_dataset("argon_md")`, `load_dataset("rmd17",
    ...)`, `load_dataset("ani1", ...)`, `load_dataset("lode_dimers",
    subset="bio_scan")`).
  + `deploy/mdi_argon_md.ipynb` and `deploy/mdi_argon_lammps.ipynb` drive a
    trained model from an external MD code through the MDI engine.
  + `examples/quickstart.py` is a minimal train/predict script on toy data.
- **Fidelity checks** (`examples/fidelity_checks/<model>_verification.ipynb`):
  + Block-by-block numerical comparisons against the upstream codes for MACE,
    NequIP, Allegro, CACE, SchNet, PhysNet, ANI, BAMBOO and LES.
  + OPLS is verified against OpenMM (an independent MD engine, optional
    dependency) in `opls_verification.ipynb` and `tests/test_opls.py`.
  + **ReaxFF is the exception:** the authors' reference implementation of
    ReaxFF-nn is AGPL-licensed, so no verification notebook or test depending
    on it (and no code derived from it) is distributed with this MIT-licensed
    code base. The implementation follows the published equations, was checked
    against that implementation during development without redistributing
    anything from it, and ships with self-contained equation-by-equation tests
    instead (`tests/test_reaxff.py`; see the fidelity notes in the
    documentation).

</details> <!-- End Examples -->

<details> <!-- Start Extension points -->
<summary><h2 style="display:inline-block">Extension points</h2></summary>

- **Implement a new model:** 
  + Pick your model family package (`gnn` / `cnn` / `dnn`/ `ffnn`,  or add one)
    and add a module under `<family>/models/`.
  + Subclass `xnn.common.models.InteratomicPotential` (or the family base,
  e.g., `gnn.models.base.EquivariantGNN`) and implement its `forward(data)`
  method.
  + register your model implementation using `@register_model` decorator.
  + Add a YAML config file for your model `configs/model/<name>.yaml`. 
  + Import the family package so the model registers.
  + For TorchScript/LAMMPS export, it is important to expose a scriptable
  `node_energy(atomic_numbers, edge_index, edge_vec)` core (SchNet shows the
  pattern; e3nn models need e3nn's JIT support for this).
- **Add a new featurizer:**
  + Subclass `xnn.common.featurizers.Featurizer` and implement `output_dim` and
  `forward(data)`. Put the resulting featurizer module in the
  `common/featurizers/` if shared by more than one model family or under the
  using family's `featurizers/` if it is only used by that one model family.

</details> <!-- End Extension points -->
