.. _cli:

****************
The Command Line
****************

Installing xnn provides the ``xnn`` command (entry point
``xnn.common.cli:main``) with three subcommands. Reading structure files
requires the ``ase`` extra.

xnn train
==========
Train a model from a config file:

.. code-block:: bash

   xnn train --config configs/train.yaml
   xnn train --config configs/train.yaml --set optim.epochs=50 model.cutoff=6.0

- ``--config``: a YAML config (see :ref:`configuration`)
- ``--set KEY=VALUE``: dotted-key overrides, repeatable

Structures are read from ``data.train_path`` / ``data.val_path`` /
``data.test_path`` with ``ase.io.read`` (any ASE-readable format: extxyz,
VASP, ...); when no ``val_path`` (``test_path``) is given,
``data.val_fraction`` (``data.test_fraction``) of the training set is held
out instead. The optional test set is evaluated once after training.
Checkpoints (``best.pt``, ``last.pt``) go to ``output_dir``.

The same command runs data-parallel on several GPUs or nodes when started
through a distributed launcher (``torchrun --nproc-per-node 2 -m xnn train
--config configs/train.yaml``) with no config changes; see
:ref:`training`.

xnn benchmark
==============
Score several **pre-trained** models on one dataset and write a results table:

.. code-block:: bash

   xnn benchmark --config configs/benchmark.yaml
   xnn benchmark --config configs/benchmark.yaml --set "metrics={'energy': ['mae']}"

- ``--config``: a YAML benchmark config (see :ref:`howto-benchmark`)
- ``--set KEY=VALUE``: dotted-key overrides applied to the config, repeatable

Each model listed in ``models`` is built from its architecture, loaded from its
``checkpoint`` (benchmarking does not train; produce checkpoints with
``xnn train`` first), and scored with the configured ``metrics`` -- a mapping
from each target (energy / forces / stress) to the error metrics (MAE / MSE /
RMSE or custom) reported for it. The comparison table is printed and written to
``output.dir`` in every configured format (CSV / JSON / Markdown). See
:ref:`howto-benchmark` for the full config.

xnn export
===========
Export a trained checkpoint for deployment:

.. code-block:: bash

   xnn export --config configs/train.yaml --ckpt runs/exp/best.pt --to lammps
   xnn export --config configs/train.yaml --ckpt runs/exp/best.pt --to torchscript

- ``--ckpt``: a checkpoint written by ``xnn train``
- ``--to``: ``lammps`` (TorchScript wrapped in the LAMMPS tensor ABI) or
  ``torchscript`` (plain scripted model)

See :ref:`deployment` for what to do with the exported file.
