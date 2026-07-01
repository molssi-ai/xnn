# xnns

Machine-learning interatomic potentials in PyTorch, for **molecular and
periodic** systems behind a single coherent `nn.Module` interface.

```
pip install -e .            # core (torch, numpy, pyyaml)
pip install -e ".[ase]"     # + ASE calculator
pip install -e ".[gnn]"     # + e3nn for NequIP/MACE/Allegro
pip install -e ".[hydra]"   # + Hydra/OmegaConf config
pip install -e ".[examples]" # + ASE, e3nn, mace-torch, jupyter (runs the notebooks)
pip install -e ".[all]"
```

The `examples` notebooks benchmark xnns against the reference
[ACEsuit/mace](https://github.com/ACEsuit/mace) (`mace-torch`), which pins
`e3nn==0.4.4`; xnns runs fine on that pin. `pyproject.toml` also carries a `uv`
setup that reproduces the GPU `.venv` the notebooks were built in (torch
`2.5.1+cu121` from the PyTorch cu121 index, for CUDA-12.x drivers).

## Quick start

```bash
python examples/quickstart.py        # builds toy data, trains, predicts forces
pytest tests/                        # smoke + featurizer + neighbor list + GNN/MACE equivariance
```

```python
from xnns.config import Config
from xnns.data import AtomicDataset
from xnns.train import Trainer

cfg = Config()
cfg.model.name = "nequip"            # schnet | hdnnp | ani | nequip | mace | allegro
cfg.model.extra = {"species": [1, 6, 8], "l_max": 2}
cfg.data.batch_size = 16             # 1 disables batch training
cfg.device = "auto"                  # auto | cpu | cuda | cuda:0
Trainer(cfg, AtomicDataset(structures, cfg.model.cutoff)).fit()
```

## Independently usable abstractions

The package is organized so each layer stands alone and can be imported and used
without the others.

```
src/xnns/
  data/         AtomicGraph (the one data object), PBC neighbor list, AtomicDataset, batching
  featurizers/  Featurizer base; symmetry functions, AEV, spherical-harmonic edge embeddings
  models/       InteratomicPotential interface + registry + ForceStressOutput; cnn/ dnn/ gnn/
  config/       one dataclass schema; loaders for yaml / toml / argparse / hydra
  train/        Trainer (batch + device aware), weighted energy/force/stress loss
  deploy/       ASE Calculator, LAMMPS/TorchScript export
```

```python
# Data on its own
from xnns.data import AtomicDataset, build_neighbor_list
ds = AtomicDataset(structures, cutoff=5.0); graph = ds[0]

# Featurizers on their own (AtomicGraph -> model inputs)
from xnns.featurizers import AEV, RadialSymmetryFunctions, SphericalHarmonicEdgeEmbedding
descriptor = AEV(species=[1, 6, 8])(graph)            # (N, D) invariant per-atom AEV
edges = SphericalHarmonicEdgeEmbedding(l_max=2)(graph) # equivariant edge attributes

# Models on their own
from xnns.models import build_model, ForceStressOutput, available_models
model = ForceStressOutput(build_model(cfg.model))      # any model + autograd forces/stress
```

Four ideas hold it together:

1. **One data object.** Every model consumes an `AtomicGraph` and returns
   `{"node_energy", "energy"}`. Molecular vs. periodic is invisible to models —
   periodicity lives only in `edge_vectors()`
   (`r_ij = pos[dst] - pos[src] + cell_shift @ cell`), keeping forces and stress
   differentiable.
2. **Featurizers are first-class.** A `Featurizer` (subclass of `nn.Module`)
   turns a graph into invariant descriptors (symmetry functions, AEV) or
   equivariant edge attributes (spherical harmonics). Descriptor models (HDNNP,
   ANI) and GNNs (NequIP/MACE/Allegro) are thin compositions over featurizers,
   so the featurization is reusable and inspectable on its own.
3. **Forces/stress in one place.** `ForceStressOutput` wraps any model and
   differentiates energy w.r.t. positions (forces) and a symmetric strain
   (stress). Models never implement them.
4. **Extensibility via registry + one config, four frontends.**
   `@register_model("name")` + a `from_config` classmethod makes a model usable
   from any of YAML / TOML / argparse / Hydra, which all funnel into one
   `Config` dataclass.

## Config formats (interchangeable)

```python
from xnns.config import from_yaml, from_toml, from_argparse, from_hydra
cfg = from_yaml("configs/train.yaml")
cfg = from_argparse(["--config", "configs/train.yaml", "--set", "model.cutoff=6.0"])
```

CLI: `xnns train --config configs/train.yaml --set optim.epochs=50`
(also `xnns export --config ... --ckpt ... --to lammps|torchscript`).

**Full MACE config.** `MACEConfig` mirrors the entire `mace_run_train` flag set
(same names and defaults) as one flat dataclass, so a MACE-CLI YAML drops in
unchanged. It stays separate from the minimal core `Config`;
`to_core_config()` bridges to the `Trainer` and `build_model()` builds the model.

```python
from xnns.config import mace_from_yaml, mace_from_hydra
mcfg = mace_from_yaml("configs/mace.yaml")
model = mcfg.build_model(species=[18])          # or infer from cfg.atomic_numbers
```

## Deployment

```python
from xnns.deploy import XNNSCalculator, export_to_lammps
atoms.calc = XNNSCalculator(model, cutoff=5.0)          # ASE
export_to_lammps(model, cutoff=5.0, path="deployed.pt") # TorchScript for LAMMPS
```

Pair the exported `.pt` with the matching C++ pair style (pair_nequip /
pair_mace / pair_allegro pattern). The `LAMMPSWrapper` in `deploy/lammps.py`
defines the tensor ABI.

## Models and fidelity

| Model | Family | Featurizer | State |
|---|---|---|---|
| SchNet | cnn | Gaussian RBF | full; trainable; TorchScript/LAMMPS-deployable |
| HDNNP | dnn | radial symmetry functions (G2) | full; trainable |
| ANI | dnn | AEV (radial + angular) | full; trainable |
| NequIP | gnn | spherical-harmonic edges | full; equivariant (verified); trainable |
| MACE | gnn | spherical-harmonic edges | faithful; learned symmetric contraction; matches ACEsuit/mace (see note) |
| Allegro | gnn | spherical-harmonic edges | equivariant; strictly local (see note) |

Equivariance is verified in `tests/test_gnn.py` and `tests/test_mace.py` (rotate
inputs → energy invariant, forces co-rotate; errors ~1e-7).

**Fidelity notes for the equivariant models.**
- *MACE* is now a faithful, self-contained re-implementation — the real
  `RealAgnostic(Residual)InteractionBlock` and the paper's *learned symmetric
  contraction* over Clebsch-Gordan paths (`correlation` order), not the earlier
  `TensorSquare` approximation. The CG coupling basis (`U_matrix_real`) is
  bit-identical to `mace-torch` and the contraction reproduces it to ~1e-16
  given the same weights. It needs only `e3nn` — no `mace-torch`,
  `cuequivariance`, or `opt_einsum_fx`. It also adds ZBL `pair_repulsion` and
  makes the message-passing depth fully flexible (`num_interactions` = T = 0..N,
  vs. upstream's fixed 2). The `examples/` notebooks verify it block-by-block
  and end-to-end against `mace-torch` on Argon MD data.
- *Allegro* implements the defining property (strict locality, no message
  passing, latent-MLP-driven equivariant edge updates) without the original's
  two-body bootstrap / normalization details.

Everything downstream (data, featurizers, autograd forces/stress, training,
ASE/LAMMPS deploy) is identical across all models.

## Examples

Runnable notebooks in `examples/` (`pip install -e ".[examples]"`), each
validating the faithful MACE against the reference `mace-torch`:

- **`01_mace_block_by_block_vs_original.ipynb`** — reproduces every MACE
  architectural block from the papers and checks each one numerically against
  the original `mace-torch` block.
- **`02_mace_argon_train_test.ipynb`** — a full train/test pipeline on real
  Argon MD data, run twice (xnns vs. original MACE) and compared at every stage.
- **`03_mace_argon_density_md.ipynb`** — liquid-Argon mass density from NPT MD
  through ASE, comparing xnns against `mace-torch` (identical weights → ~zero
  difference, plus independently trained models).

`examples/quickstart.py` is the minimal toy-data train/predict loop.

## Extension points

- **New model:** subclass `InteratomicPotential`, implement `forward(data)`,
  register with `@register_model`, add a `configs/model/<name>.yaml`. For
  TorchScript/LAMMPS export also expose a scriptable
  `node_energy(atomic_numbers, edge_index, edge_vec)` core (SchNet shows the
  pattern; e3nn models need e3nn's JIT support for this).
- **New featurizer:** subclass `Featurizer`, implement `output_dim` and
  `forward(data)`; compose it into a model.
- **Neighbor list:** the reference builder is correct but brute-force; swap in a
  cell-list / `matscipy` for large periodic systems — the
  `edge_index`/`cell_shifts` interface is unchanged.
