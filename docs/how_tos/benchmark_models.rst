.. _howto-benchmark:

*****************************
Benchmark Several Models
*****************************

The :mod:`xnns.common.benchmark` package scores any set of **pre-trained**
models on one dataset and writes a table of error metrics. It does one thing --
benchmarking -- and does not train: produce the checkpoints first (for example
with ``xnns train``; see :ref:`howto-train-config`), then point each model at
its checkpoint. Models are built with the same model registry and
:class:`~xnns.common.models.ForceStressOutput` a single run uses.

Write a benchmark config
========================
A benchmark config lists the ``models`` to compare, the ``data`` to score on,
and the ``metrics`` / ``targets`` to report:

.. code-block:: yaml

   # benchmark.yaml
   models:
     - label: mace
       checkpoint: runs/mace/best.pt            # architecture read from the checkpoint
     - label: nequip
       checkpoint: runs/nequip/best.pt

   metrics: [mae, rmse]                         # mae / mse / rmse or custom
   targets: [energy, forces]                    # energy / forces / stress

   data:
     test_path: data/argon_test.extxyz         # the dataset to score on
     batch_size: 16

   output:
     dir: runs/benchmark
     filename: results
     formats: [csv, json, md]                   # any registered writer

   device: auto
   seed: 1234

The bundled ``configs/benchmark.yaml`` is a complete, commented example, and
``examples/benchmark/`` benchmarks NequIP, Allegro, MACE and PhysNet on the
Argon test set.

**Every entry needs a** ``checkpoint`` -- benchmarking scores pre-trained
models. An xnns checkpoint embeds the :class:`~xnns.common.config.schema.Config`
it was trained with, so the benchmark rebuilds the exact architecture from the
checkpoint and an entry usually needs only its ``checkpoint`` (plus an optional
``label`` for the row name). Give an explicit architecture -- inline keys, a
``name``, or a ``config`` file -- only for checkpoints that do **not** embed a
config; upstream key spellings (MACE ``r_max`` ...) are translated to the xnns
canonical names exactly as in a normal run (see :ref:`howto-upstream-configs`).

The dataset is resolved from the ``data`` section: ``test_path`` (the natural
held-out benchmark set), falling back to ``val_path`` then ``train_path``.

Run the benchmark
=================
From the command line:

.. code-block:: bash

   xnns benchmark --config benchmark.yaml
   xnns benchmark --config benchmark.yaml --set "targets=['energy']"

or from Python:

.. code-block:: python

   from xnns.common.benchmark import from_yaml, run_benchmark

   rows = run_benchmark(from_yaml("benchmark.yaml"))

The comparison table is printed and written to ``output.dir`` in every
configured format, one row per model, with a ``n_params`` column and one
``<target>_<metric>`` column per scored quantity (e.g. ``energy_mae``,
``forces_rmse``).

The tables show the physical unit after each metric name (e.g.
``energy_mae [eV/atom]``). xnns is unit-agnostic, so these are labels only:
they default to ``eV/atom`` (or ``eV`` with ``energy_per_atom: false``) for
energy, ``eV/A`` for forces and ``eV/A**3`` for stress, and are overridden per
target with ``units`` to match your data:

.. code-block:: yaml

   units:
     energy: meV/atom
     forces: meV/A

The written ``results.{csv,json,md}`` files carry the same unit-annotated
headers (the returned rows keep plain keys for programmatic use).

Report atomization (interaction) energy
=======================================
By default energy is scored per atom (total energy divided by the atom count).
Set ``atomic_energies`` to score the **atomization / interaction energy**
instead -- the total minus the summed per-element reference energies (E0s),
which is the physically meaningful quantity:

.. code-block:: yaml

   atomic_energies: {1: -13.663, 6: -1029.863, 8: -2042.785}   # {Z: E0}, eV
   # atomic_energies: average        # or fit E0s from the benchmark data (lstsq)
   # species: [H, C, O]              # only needed for the list / scalar forms
   energy_per_atom: true             # divide the (atomization) energy by atom count

The same E0 offset is subtracted from both the prediction and the reference, so
plain difference metrics (MAE/MSE/RMSE) are unchanged while the reported values
become meaningful; the effect is visible in reference-dependent custom metrics
(relative error, :math:`R^2`) and in reported magnitudes. Accepted forms are a
``{Z: E0}`` / ``{symbol: E0}`` dict, a list aligned with ``species``, a single
number, or ``"average"`` to fit the E0s from the benchmark dataset
(:func:`~xnns.common.benchmark.energy.build_e0_lookup`).

Add custom metrics and output formats
=====================================
Custom error metrics are registered like models. Either import them in the
config with a ``module:function`` path (each callable has the signature
``fn(pred, target) -> float``):

.. code-block:: yaml

   custom_metrics:
     - {name: maxae, path: my_metrics:max_abs_error}
   metrics: [mae, rmse, maxae]

or register them in Python before running:

.. code-block:: python

   from xnns.common.benchmark import register_metric, register_writer

   @register_metric("maxae")
   def max_abs_error(pred, target):
       return float((pred - target).abs().max())

   @register_writer("tsv")            # a new output format
   def write_tsv(rows, columns, path):
       with open(path, "w") as f:
           f.write("\t".join(columns) + "\n")
           for r in rows:
               f.write("\t".join(str(r.get(c, "")) for c in columns) + "\n")

Any registered writer name can then be listed under ``output.formats``.
