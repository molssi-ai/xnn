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

The `examples` notebooks benchmark xnns models against their reference
implementations. For example, the MACE implementation in xnns is validated
against that of [ACEsuit/mace](https://github.com/ACEsuit/mace) (`mace-torch`),
the NequIP implementation is validated against that of
[mir-group/nequip](https://github.com/mir-group/nequip) and Allegro is validated
against [mir-group/allegro](https://github.com/mir-group/allegro), which pin
`e3nn==0.4.4`; xnns has been thoroughly tested on this pin. The `pyproject.toml`
also carries a `uv` setup that reproduces the GPU `.venv` that was used to
create the notebooks (we adopted `torch 2.5.1+cu121` from the PyTorch cu121
index that are compatible with CUDA-12.x drivers).

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
# e.g. schnet | hdnnp | ani | physnet | nequip | mace | allegro | cace
cfg.model.name = "nequip"
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
# works ase-io native formats such as .extxyz, .cif, VASP, ... formats
train_set = AtomicDataset.from_file("trajectory.extxyz", cutoff=4.0)
```

## Package layout

The package is organized **by model family** (`gnn`, `cnn`, `dnn`), with shared
resources factored into the `common` modules. An object (e.g., function, module
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
│   ├── models/                 - InteratomicPotential interface, registry, ForceStressOutput, ops (scatter_sum)
│   │   └── …                     + base, registry, outputs, les, ops
│   ├── train/                  - Trainer (batch + device aware), weighted energy/force/stress loss
│   │   └── …                     + trainer, losses
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
└── dnn/                        descriptor + per-element networks, and PhysNet
    ├── featurizers/            - DNN featurizers
    │   └── …                     + symmetry functions, AEV
    └── models/                 base (DescriptorPotential), hdnnp, ani, physnet and ported Grimme's D3
        └── …
```

```python
# Data (structures dict -> AtomicGraph)
from xnns.common.data import AtomicDataset, build_neighbor_list
ds = AtomicDataset(structures, cutoff=5.0); graph = ds[0]

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
(also `xnns export --config ... --ckpt ... --to lammps|torchscript`).

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
the eager ones to ~1e-15, verified in `tests/test_mace.py` /
`tests/test_nequip.py` / `tests/test_allegro.py`). For NequIP this required a scriptable, bit-exact stand-in
for e3nn's `Gate` (`xnns.gnn.models.nequip._Gate`), which the e3nn 0.4.4
original cannot do on torch 2.x.

## Models and fidelity

| Model | Family | Featurizer | State |
|---|---|---|---|
| SchNet | cnn | Gaussian RBF | full; trainable; TorchScript/LAMMPS-deployable |
| HDNNP | dnn | radial symmetry functions (G2) | full; trainable |
| ANI | dnn | AEV (radial + angular) | full; trainable |
| NequIP | gnn | spherical-harmonic edges | faithful; matches mir-group/nequip (see note); TorchScript/LAMMPS-deployable |
| MACE | gnn | spherical-harmonic edges | faithful; learned symmetric contraction; matches ACEsuit/mace (see note); TorchScript/LAMMPS-deployable |
| Allegro | gnn | spherical-harmonic edges | faithful; matches mir-group/allegro (see note); TorchScript/LAMMPS-deployable |
| CACE | gnn | Cartesian monomial edges | faithful; matches BingqingCheng/cace (see note); no e3nn; ASE-deployable |
| PhysNet | dnn | exp-Gaussian rbf + attention masks | faithful; matches MMunibas/PhysNet TF (see note); charges/dipoles/electrostatics/D3; ASE-deployable |

Equivariance is verified in `tests/test_gnn.py` and `tests/test_mace.py` (rotate
inputs → energy invariant, forces co-rotate; errors ~1e-7).

**Fidelity notes for the equivariant models.**
- *MACE* is a faithful, self-contained re-implementation — the real
  `RealAgnostic(Residual)InteractionBlock` and the paper's *learned symmetric
  contraction* over Clebsch-Gordan paths (`correlation` order). The CG
  coupling basis (`U_matrix_real`) is
  bit-identical to `mace-torch` and the contraction reproduces it to ~1e-16
  given the same weights. It needs only `e3nn` — no `mace-torch`,
  `cuequivariance`, or `opt_einsum_fx`. It also adds ZBL `pair_repulsion` and
  makes the message-passing depth fully flexible (`num_interactions` = T = 0..N,
  vs. upstream's fixed 2). The `examples/` notebooks verify it block-by-block
  and end-to-end against `mace-torch` on Argon MD data.
- *NequIP* is a faithful, self-contained re-implementation of the upstream
  `EnergyModel`: the real `InteractionBlock` (with upstream parameter names, so
  state dicts transplant directly), per-layer `tp_path_exists` irreps pruning,
  the gated nonlinearity, NequIP's radial conventions (trainable Bessel with
  the `2/r_max` prefactor, `1/sqrt(avg_num_neighbors)` message normalization,
  the `r_j - r_i` edge orientation) and the per-species energy scale/shift.
  Given the same weights it reproduces `nequip` to ~1e-16 (energies, forces
  and stress; `tests/test_nequip.py`), needing only `e3nn`. The `examples/`
  notebooks verify it block-by-block and end-to-end on Argon MD data.
- *Allegro* is a faithful, self-contained re-implementation of the original
  mir-group/allegro (v0.3.0, the e3nn-era reference, default `uuulin` mode):
  the two-body product type embedding, the per-channel weightless Wigner-3j
  tensor products with the embedded-environment density trick, the strided
  channel-mixing linears (same flat weight layout, so state dicts transplant
  directly), the cumulative-softmax latent resnet, Allegro's radial
  conventions (trainable "normalized sinc" Bessel with the `r_max/pi`
  prefactor, `1/sqrt(avg_num_neighbors - 1)` environment and
  `1/sqrt(avg_num_neighbors)` energy-sum normalization) and the per-species
  scale/shift. Given the same weights it reproduces `allegro` to ~1e-15
  (energies and forces; `tests/test_allegro.py`), needing only `e3nn`. The
  `examples/` notebooks verify it block-by-block and end-to-end on Argon MD
  data.
- *CACE* is a faithful, self-contained re-implementation of BingqingCheng/cace
  (the Cartesian atomic cluster expansion, Cheng 2024) — the only model here
  that needs no spherical harmonics or e3nn at all: the Cartesian monomial
  angular basis (`CartesianAngularBasis`, same autograd-safe recursion), the
  exact multinomial symmetrization rules (identical B-feature ordering), the
  tensor-product element-embedding edge type, the per-(l, c) trainable radial
  channel coupling, all three message-passing mechanisms (`M`/`Ar`/`Bchi`),
  and the linear + MLP readout. Given the same weights it reproduces `cace`
  to ~1e-16 relative (energies and forces, molecular and periodic;
  `tests/test_cace.py`). Like upstream (which has no LAMMPS interface), it
  deploys through ASE rather than TorchScript. The `examples/gnn/cace/`
  notebooks verify it block-by-block and end-to-end on Argon MD data.
- *PhysNet* is a faithful **pure-PyTorch translation of the original
  TensorFlow 1.x implementation** (MMunibas/PhysNet): the exponential-Gaussian
  radial basis, distance-based attention masks, pre-activation residual
  blocks, per-module (energy, charge) output heads with per-element
  scale/shift tables, the exact charge correction, switched/shielded
  electrostatics, and a statement-for-statement port of the bundled Grimme
  D3(BJ) module (tables shipped in-package, coefficients learnable). Given the
  same weights it reproduces the original TF graph to ~1e-15 in energies,
  forces, charges, and the non-hierarchicality penalty
  (`tests/test_physnet.py`; TF + an upstream clone required for the parity
  test). It also returns `"charges"`, `"dipole"`, and `"nh_loss"` from
  `forward`. The `examples/dnn/physnet/` notebooks verify it block-by-block
  and end-to-end on Argon MD data against the original TF graph.

Everything downstream (data, featurizers, autograd forces/stress, training,
ASE/LAMMPS deploy) is identical across all models.

## Examples

Runnable notebooks in `examples/gnn/mace/` (`pip install -e ".[examples]"`), each
validating the faithful MACE against the reference `mace-torch`:

- **`01_mace_block_by_block_vs_original.ipynb`** — reproduces every MACE
  architectural block from the papers and checks each one numerically against
  the original `mace-torch` block.
- **`02_mace_argon_train_test.ipynb`** — a full train/test pipeline on real
  Argon MD data, run twice (xnns vs. original MACE) and compared at every stage.
- **`03_mace_argon_density_md.ipynb`** — liquid-Argon mass density from NPT MD
  through ASE, comparing xnns against `mace-torch` (identical weights → ~zero
  difference, plus independently trained models).
- **`04_recreate_mace_architecture.ipynb`** — a step-by-step tutorial that rebuilds
  the MACE architecture block by block in *both* `mace-torch` and xnns (with the
  defining equations and architecture figures in `figures/`), transplants a whole
  model, and reproduces its energy and forces to ~1e-15.

The same trilogy exists for NequIP in `examples/gnn/nequip/`, validating the
faithful NequIP against the reference `nequip` package (the Argon data is shared
from `examples/gnn/mace/data/`):

- **`01_nequip_block_by_block_vs_original.ipynb`** — every NequIP block
  (embedding, trainable Bessel basis, spherical harmonics, interaction block,
  gate, readout, per-species scale/shift) checked numerically against the
  original, ending with a whole-model weight transplant (~1e-16).
- **`02_nequip_argon_train_test.ipynb`** — the full train/test pipeline on the
  Argon MD data, run twice (xnns vs. original NequIP) and compared at every
  stage.
- **`03_nequip_argon_density_md.ipynb`** — liquid-Argon mass density from NPT MD
  through ASE, comparing xnns against `nequip` (identical weights → ~zero
  difference, plus independently trained models).

And for Allegro in `examples/gnn/allegro/`, validating the faithful Allegro
against the reference `allegro` package, and for CACE in `examples/gnn/cace/`,
validating the faithful CACE against the reference `cace` package (same
01 block-by-block / 02 Argon train-test / 03 NPT-density trilogy).

PhysNet has its trilogy in `examples/dnn/physnet/`, validated against the
**original TensorFlow implementation** (run with a venv that has both
`tensorflow` and `torch`; the notebooks clone MMunibas/PhysNet on demand and
drive its TF1 graph through `tf.compat.v1`).

`examples/quickstart.py` is the minimal toy-data train/predict loop.

## Extension points

- **New model:** pick the family package (`gnn` / `cnn` / `dnn`, or add one),
  add a module under `<family>/models/`, subclass
  `xnns.common.models.InteratomicPotential` (or the family base, e.g.
  `gnn.models.base.EquivariantGNN`), implement `forward(data)`, register with
  `@register_model`, and add a `configs/model/<name>.yaml`. Import the family
  package so the model registers. For TorchScript/LAMMPS export also expose a
  scriptable `node_energy(atomic_numbers, edge_index, edge_vec)` core (SchNet
  shows the pattern; e3nn models need e3nn's JIT support for this).
- **New featurizer:** subclass `xnns.common.featurizers.Featurizer`, implement
  `output_dim` and `forward(data)`; put it in `common/featurizers/` if shared,
  else under the using family's `featurizers/`, and compose it into a model.
- **Neighbor list:** the reference builder is correct but brute-force; swap in a
  cell-list / `matscipy` for large periodic systems — the
  `edge_index`/`cell_shifts` interface is unchanged.
