# Benchmarking example — NequIP / Allegro / MACE / PhysNet on Argon

Scores four pre-trained models on the shared Argon test set
([`../../datasets/argon_md/argon_test.xyz`](../../datasets/argon_md/argon_test.xyz)) and
tabulates their energy/force errors — a one-config, apples-to-apples comparison.

Benchmarking only *scores* pre-trained models; it does not train. The
checkpoints used here are produced by the per-model training notebooks
(`examples/<family>/<model>_argon_train_test.ipynb`). Run those first if a
checkpoint is missing.

## Run it

From the **repository root** (paths in the config are relative to it):

```bash
xnn benchmark --config examples/benchmark/argon_benchmark.yaml
```

or from Python:

```python
from xnn.common.benchmark import from_yaml, run_benchmark

rows = run_benchmark(from_yaml("examples/benchmark/argon_benchmark.yaml"))
```

The table is printed and written to
`examples/benchmark/runs/argon_benchmark/results.{csv,json,md}`.

## What you get

One row per model. Each metric column shows its unit (energy per atom, forces
per component — see the `units:` block in the config):

```
model    n_params  energy_mae [eV/atom]  energy_rmse [eV/atom]  forces_mae [eV/A]  forces_rmse [eV/A]
-------  --------  --------------------  ---------------------  -----------------  ------------------
nequip   208128    0.0123838             0.0153581              0.00124883         0.00200672
allegro  124432    0.0125408             0.0166653              0.00102537         0.00164191
mace     73048     0.0128105             0.0152518              0.00131497         0.00202244
physnet  200448    0.014039              0.0170653              0.00173706         0.00307508
```

(Exact numbers depend on the checkpoints in each `runs/argon_xnn/` directory.)
The `results.{csv,json,md}` files carry the same unit-annotated headers
(`energy_mae [eV/atom]`, ...).

## How it works

- **Architecture from the checkpoint.** Each xnn checkpoint embeds the
  `Config` it was trained with, so every entry in the config is just a `label`
  and a `checkpoint` — the benchmark rebuilds the exact architecture from the
  checkpoint before loading the weights.
- **One dataset, one cutoff per model.** The Argon test set is read once per
  cutoff (models sharing a cutoff share the neighbor-list build).
- **Target keys.** The Argon frames store references under the MACE-convention
  keys `REF_energy` / `REF_forces`, set via `data.energy_key` / `forces_key`.

## Variations

Override any config key on the command line with `--set`:

```bash
# forces only
xnn benchmark --config examples/benchmark/argon_benchmark.yaml \
    --set "metrics={'forces': ['mae','rmse']}"

# report atomization energy (fit the per-atom Ar reference from the data)
xnn benchmark --config examples/benchmark/argon_benchmark.yaml --set atomic_energies=average

# add MSE, keep MAE only for energy, and write only JSON
xnn benchmark --config examples/benchmark/argon_benchmark.yaml \
    --set "metrics={'energy': ['mae'], 'forces': ['mae','mse','rmse']}" "output.formats=['json']"
```

See the [How-To guide](../../docs/how_tos/benchmark_models.rst) for custom
metrics and output formats.
