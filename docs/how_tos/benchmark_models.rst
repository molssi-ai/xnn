.. _howto-benchmark:

*****************************
Benchmark Several Models
*****************************

The ``xnn benchmark`` driver scores pre-trained checkpoints on one dataset and
writes a table of error/score measures. This module does not train the model!
You should produce the checkpoints first (using ``xnn train``) and list them in the
benchmarking config file.

Write a benchmark config
========================

A typical benchmark config YAML file can have the following sections:

.. code-block:: yaml

   # benchmark.yaml

   models:
     - label: mace
       checkpoint: runs/mace/best.pt       # The architecture comes from the checkpoint
     - label: nequip
       checkpoint: runs/nequip/best.pt

   metrics:                                # Target -> metrics to report
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

The bundled ``configs/benchmark.yaml`` file offers a general commented example
and ``notebooks/examples/common/benchmark/`` provides specific cases where the benchmark task
scores NequIP, Allegro, MACE and PhysNet on the argon test set that is also
available from the xnn's data hub and is shipped with the package. A
``checkpoint`` can be anything ``from_pretrained()`` accepts. Only checkpoints
without an embedded config need an explicit architecture (inline keys, a
``name`` or a ``config`` file). The dataset comes from ``test_path``, and will
fallback to ``val_path`` and ``train_path``, if not present. 

Run the benchmark
=================
You can run the benchmark from the command line or in Python. The command line
accepts a config file and optional overrides:

.. code-block:: bash

   xnn benchmark --config benchmark.yaml
   xnn benchmark --config benchmark.yaml --set "metrics={'energy': ['mae']}"

The Python API is in ``xnn.common.benchmark``. The simplest way to run a
benchmark is to load a config and call ``run_benchmark()``:

.. code-block:: python

   from xnn.common.benchmark import from_yaml, run_benchmark

   rows = run_benchmark(from_yaml("benchmark.yaml"))

The resulting table will have one row per model, a ``n_params`` column and one
``<target>_<metric>`` column per scored pair, with the unit after each name
(e.g., ``energy_mae [eV/atom]``). Units can be set per target to match your data
through ``units: {energy: meV/atom, forces: meV/A}``.

Benchmarking on multiple GPUs
=============================
xnn allows for parallelism across two independent dimensions and their
combination for benchmarking: Parallel execution across the dataset and across
the models.

**Across the dataset, one model at a time.** Start the same command through a
distributed launcher, as for multi-GPU distributed data-parallel (DDP) training
(see :ref:`training`). Each rank scores a disjoint strided shard of the dataset
and the predictions are gathered before the metrics are computed, so the numbers
match that of a serial run. Only rank 0 writes the outcomes. The ``data.batch_size``
variable is per-rank (per-GPU).

.. code-block:: bash

   # One node, 4 GPUs
   torchrun --nproc-per-node 4 -m xnn benchmark --config benchmark.yaml

   # Two nodes: one such command per node (e.g. one Slurm step each)
   torchrun --nnodes 2 --nproc-per-node 4 \
            --rdzv-backend c10d --rdzv-endpoint "$HEAD_NODE":29500 \
            -m xnn benchmark --config benchmark.yaml

**Across models, one GPU each.** Restrict a run with ``--models`` (labels or
positions) or ``--shard I/N`` (every ``N``-th entry from ``I``). Such a run
writes its rows to a part file under ``output.dir/parts/`` instead of the final
table. Calling benchmark driver with ``--merge`` assembles the table from the
parts in the config order.

.. code-block:: bash

   xnn benchmark --config benchmark.yaml --shard 0/2      # Entries 0, 2, 4, ...
   xnn benchmark --config benchmark.yaml --shard 1/2      # Entries 1, 3, 5, ...
   xnn benchmark --config benchmark.yaml --merge          # -> results.{csv,json,md}

On Slurm, the shards are the tasks of one job array. That is, the ``--shard
slurm`` reads the ``I/N`` entries from the array variables, and the merge runs
after the array completes to assemble the final table. 

.. code-block:: bash

   # bench.sh: One GPU per task, one task per model

   #SBATCH --gres=gpu:1
   xnn benchmark --config benchmark.yaml --shard slurm

   JOB=$(sbatch --parsable --array=0-3 bench.sh)      # 4 models
   sbatch --dependency=afterok:$JOB --wrap "xnn benchmark --config benchmark.yaml --merge"

It is also possible to score each model in DDP mode. Just run ``torchrun
--nproc-per-node 4 -m xnn benchmark`` inside ``bench.sh`` script with ``#SBATCH
--gres=gpu:4``. The merge names a model whose part is missing (a failed task)
and leaves its row out. You can rerun that shard and merge again to recover the
missing row.

On one multi-GPU machine without a scheduler, adding the ``--parallel`` flag
does the process fan-out and the merge in one command. Here, one worker process
per GPU (or ``--parallel N`` workers), each pinned to its GPU through
``CUDA_VISIBLE_DEVICES`` scores one model at a time and logs to
``output.dir/parts/<label>.log``.

.. code-block:: bash

   xnn benchmark --config benchmark.yaml --parallel

In Python API, these functionalities are handled via
:func:`~xnn.common.benchmark.select_entries`,
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
Custom metrics can be defined by pointing at a ``module:function`` with the
signature ``fn(pred, target) -> float``, or registering them in Python. In the
config, the ``custom_metrics`` section lists the name and path of each custom
metric:

.. code-block:: yaml

   custom_metrics:
     - {name: maxae, path: my_metrics:max_abs_error}
   
   metrics:
     forces: [mae, maxae]

In Python, the ``@register_metric`` and ``@register_writer`` decorators can be
used to register custom metrics and output formats:

.. code-block:: python

   from xnn.common.benchmark import register_metric, register_writer

   @register_metric("maxae")
   def max_abs_error(pred, target):
       return float((pred - target).abs().max())

   @register_writer("tsv")
   def write_tsv(rows, columns, path):
       ...

A registered writer name can then be listed under ``output.formats``.
