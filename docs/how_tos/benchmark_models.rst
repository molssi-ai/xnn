.. _howto-benchmark:

*****************************
Benchmark Several Models
*****************************

``xnn benchmark`` scores pre-trained checkpoints on one dataset and writes a
table of error metrics. It does not train: produce the checkpoints first
(``xnn train``) and list them in the config.

Write a benchmark config
========================

.. code-block:: yaml

   # benchmark.yaml
   models:
     - label: mace
       checkpoint: runs/mace/best.pt       # the architecture comes from the checkpoint
     - label: nequip
       checkpoint: runs/nequip/best.pt

   metrics:                                # target -> metrics to report
     energy: [mae, rmse]
     forces: [mae, rmse]

   data:
     test_path: data/argon_test.extxyz
     batch_size: 16

   output:
     dir: runs/benchmark
     filename: results
     formats: [csv, json, md]

   device: auto

``configs/benchmark.yaml`` is a commented example and ``examples/benchmark/``
scores NequIP, Allegro, MACE and PhysNet on the argon test set. A
``checkpoint`` can be anything ``from_pretrained()`` accepts. Only
checkpoints without an embedded config need an explicit architecture
(inline keys, a ``name`` or a ``config`` file). The dataset comes from
``test_path``, falling back to ``val_path`` and ``train_path``.

Run it
======

.. code-block:: bash

   xnn benchmark --config benchmark.yaml
   xnn benchmark --config benchmark.yaml --set "metrics={'energy': ['mae']}"

.. code-block:: python

   from xnn.common.benchmark import from_yaml, run_benchmark

   rows = run_benchmark(from_yaml("benchmark.yaml"))

The table has one row per model, a ``n_params`` column and one
``<target>_<metric>`` column per scored pair, with the unit after each name
(``energy_mae [eV/atom]``). Units are labels: set them per target to match
your data with ``units: {energy: meV/atom, forces: meV/A}``.

Several GPUs or nodes
=====================
Two independent kinds of parallelism combine.

**Across the dataset, one model at a time.** Start the same command through a
distributed launcher, as for training (see :ref:`training`). Each rank scores
a disjoint strided shard of the dataset and the predictions are gathered
before the metrics are computed, so the numbers equal a serial run. Only
rank 0 writes. ``data.batch_size`` is per rank.

.. code-block:: bash

   torchrun --nproc-per-node 4 -m xnn benchmark --config benchmark.yaml

   # two nodes: one such command per node (e.g. one Slurm step each)
   torchrun --nnodes 2 --nproc-per-node 4 \
            --rdzv-backend c10d --rdzv-endpoint "$HEAD_NODE":29500 \
            -m xnn benchmark --config benchmark.yaml

**Across models, one GPU each.** Restrict a run with ``--models`` (labels or
positions) or ``--shard I/N`` (every ``N``-th entry from ``I``). Such a run
writes its rows to a part file under ``output.dir/parts/`` instead of the
final table; ``--merge`` assembles the table from the parts in config order.

.. code-block:: bash

   xnn benchmark --config benchmark.yaml --shard 0/2      # entries 0, 2, 4, ...
   xnn benchmark --config benchmark.yaml --shard 1/2      # entries 1, 3, 5, ...
   xnn benchmark --config benchmark.yaml --merge          # -> results.{csv,json,md}

On Slurm the shards are the tasks of one job array, ``--shard slurm`` reads
``I/N`` from the array variables, and the merge runs after the array:

.. code-block:: bash

   # bench.sh: one GPU per task, one task per model
   #SBATCH --gres=gpu:1
   xnn benchmark --config benchmark.yaml --shard slurm

   JOB=$(sbatch --parsable --array=0-3 bench.sh)                 # 4 models
   sbatch --dependency=afterok:$JOB --wrap "xnn benchmark --config benchmark.yaml --merge"

To also score each model data-parallel, run ``torchrun --nproc-per-node 4 -m
xnn benchmark`` inside ``bench.sh`` with ``--gres=gpu:4``. The merge names a
model whose part is missing (a failed task) and leaves its row out: rerun that
shard and merge again.

On one multi-GPU machine without a scheduler, ``--parallel`` does the fan-out
and the merge in one command: one worker process per GPU (or ``--parallel N``
workers), each pinned to its GPU through ``CUDA_VISIBLE_DEVICES`` and scoring
one model at a time, logging to ``output.dir/parts/<label>.log``.

.. code-block:: bash

   xnn benchmark --config benchmark.yaml --parallel

In Python these are :func:`~xnn.common.benchmark.select_entries`,
:class:`~xnn.common.benchmark.Benchmark` (``entries`` argument),
:func:`~xnn.common.benchmark.merge_parts` and
:func:`~xnn.common.benchmark.run_parallel`.

Metrics
=======
Targets are ``energy``, ``forces`` and ``stress``; metrics are ``mae``,
``mse``, ``rmse`` or a custom one. Two shorter spellings of the mapping are
accepted: ``(target, metrics)`` pairs, and the flat cross product
``metrics: [mae, rmse]`` with ``targets: [energy, forces]``.

Energy is scored per atom. To score the atomization energy instead, give the
per-element reference energies, or fit them from the benchmark data:

.. code-block:: yaml

   atomic_energies: {1: -13.663, 6: -1029.863, 8: -2042.785}   # {Z: E0}, or "average"
   energy_per_atom: true

Custom metrics and formats
==========================
Point at a ``module:function`` with the signature ``fn(pred, target) -> float``,
or register in Python:

.. code-block:: yaml

   custom_metrics:
     - {name: maxae, path: my_metrics:max_abs_error}
   metrics:
     forces: [mae, maxae]

.. code-block:: python

   from xnn.common.benchmark import register_metric, register_writer

   @register_metric("maxae")
   def max_abs_error(pred, target):
       return float((pred - target).abs().max())

   @register_writer("tsv")
   def write_tsv(rows, columns, path):
       ...

A registered writer name can then be listed under ``output.formats``.
