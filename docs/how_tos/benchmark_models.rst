.. _howto-benchmark:

*****************************
Benchmark Several Models
*****************************

The :mod:`xnn.common.benchmark` package scores any set of **pre-trained**
models on one dataset and writes a table of error metrics. It only does one
thing: benchmarking. Thus, no other operation such as data processing and model
training is performed in this module. The users must produce the model
checkpoints first (for example with ``xnn train``; see
:ref:`howto-train-a-model`), and then include the checkpoint location in the
config. Then the benchmark module will load the model from the checkpoint, run
it on the dataset, and report the metrics.

Write a benchmark config
========================
A benchmark config lists the ``models`` to compare, the ``data`` to score on,
and the ``metrics`` to report:

.. code-block:: yaml

   # benchmark.yaml
   models:
     - label: mace
       checkpoint: runs/mace/best.pt            # architecture read from the checkpoint
     - label: nequip
       checkpoint: runs/nequip/best.pt

   metrics:                                     # target -> its reported metrics
     energy: [mae, rmse]                        # energy / forces / stress
     forces: [mae, rmse]                        # mae / mse / rmse or custom

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

**Every model entry needs a** ``checkpoint``: The :mod:`xnn.common.benchmark`
module scores pre-trained models. An xnn checkpoint embeds the
:class:`~xnn.common.config.schema.Config` it was trained with, so the benchmark
rebuilds the exact architecture from the checkpoint. A model entry usually needs
only its ``checkpoint``, and an optional ``label`` for the row name. The users
should provide an explicit architecture -- inline keys, a ``name``, or a
``config`` file -- only for checkpoints that do **not** embed a config object.
The upstream key spellings (e.g., for MACE, ``r_max``, ...) are translated to
the xnn canonical names exactly as in a normal run (see
:ref:`howto-upstream-configs`).

The dataset is resolved from the ``data`` section: ``test_path`` (the natural
held-out benchmark set), falling back to ``val_path`` then ``train_path``.

Connect metrics to targets
==========================
The ``metrics`` mapping states unambiguously *which* metric is reported for
*which* target quantity, so different targets can carry different metrics.
Targets are any subset of ``energy`` / ``forces`` / ``stress`` (anything else is
rejected at load time). Only the targets named in the mapping are scored. Three
equivalent spellings are accepted. The mapping is the canonical form, in which
the targets are keys and the metrics, corresponding to each target, can be a
single name or a list of names:

.. code-block:: yaml

   metrics:
     energy: mae
     forces: [mae, rmse]

The same connection can be written as ``(target, metrics)`` pairs: 2-item lists
in YAML are turned into tuples in Python. Repeated targets accumulate their
metrics:

.. code-block:: yaml

   metrics:
     - [energy, mae]
     - [forces, [mae, rmse]]

Finally, the flat cross-product shorthand applies every metric to every
target:

.. code-block:: yaml

   metrics: [mae, rmse]
   targets: [energy, forces]

The separate ``targets`` key belongs to the flat shorthand only. combining it
with the mapping or pair form raises an error. An omitted side of the flat form
falls back to its default (``metrics: [mae, rmse]``, ``targets: [energy,
forces]``), as does a mapping target given without metrics (e.g. ``energy:
null``). Whatever the spelling, every loader normalizes the inputs into the same
per-target mapping on :attr:`BenchmarkConfig.metrics
<xnn.common.benchmark.BenchmarkConfig>`, and the results table carries one
``<target>_<metric>`` column per connected pair.

Run the benchmark
=================
From the command line:

.. code-block:: bash

   xnn benchmark --config benchmark.yaml
   xnn benchmark --config benchmark.yaml --set "metrics={'energy': ['mae']}"

or from Python:

.. code-block:: python

   from xnn.common.benchmark import from_yaml, run_benchmark

   rows = run_benchmark(from_yaml("benchmark.yaml"))

The comparison table is printed and written to ``output.dir`` in every
configured format, one row per model, with a ``n_params`` column showing the
number of model parameters and one ``<target>_<metric>`` column per scored
quantity (e.g. ``energy_mae``, ``forces_rmse``).

The tables show the physical unit after each metric name (e.g. ``energy_mae
[eV/atom]``). xnn is unit-agnostic, so these are labels only: they default to
``eV/atom`` (or ``eV`` with ``energy_per_atom: false``) for energy, ``eV/A`` for
forces and ``eV/A**3`` for stress, and are overridden per target with ``units``
to match your data:

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
instead: the total energy minus the summed per-element reference energies (E0s),
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
(see :func:`~xnn.common.benchmark.energy.build_e0_lookup` for details  ).

Add custom metrics and output formats
=====================================
Custom error metrics are registered like models. Either import them in the
config with a ``module:function`` path (each callable has the signature
``fn(pred, target) -> float``):

.. code-block:: yaml

   custom_metrics:
     # import a metric from a Python module
     - {name: maxae, path: my_metrics:max_abs_error}
   metrics:
     energy: [mae, rmse]
     # use the custom metric by name
     forces: [mae, rmse, maxae]

or register them in Python before running:

.. code-block:: python

   from xnn.common.benchmark import register_metric, register_writer

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
