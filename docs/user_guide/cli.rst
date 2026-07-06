.. _cli:

****************
The Command Line
****************

Installing xnns provides the ``xnns`` command (entry point
``xnns.common.cli:main``) with two subcommands. Reading structure files
requires the ``ase`` extra.

xnns train
==========
Train a model from a config file:

.. code-block:: bash

   xnns train --config configs/train.yaml
   xnns train --config configs/train.yaml --set optim.epochs=50 model.cutoff=6.0

- ``--config`` — a YAML config (see :ref:`configuration`)
- ``--set KEY=VALUE`` — dotted-key overrides, repeatable

Structures are read from ``data.train_path`` / ``data.val_path`` /
``data.test_path`` with ``ase.io.read`` (any ASE-readable format: extxyz,
VASP, ...); when no ``val_path`` (``test_path``) is given,
``data.val_fraction`` (``data.test_fraction``) of the training set is held
out instead. The optional test set is evaluated once after training.
Checkpoints (``best.pt``, ``last.pt``) go to ``output_dir``.

xnns export
===========
Export a trained checkpoint for deployment:

.. code-block:: bash

   xnns export --config configs/train.yaml --ckpt runs/exp/best.pt --to lammps
   xnns export --config configs/train.yaml --ckpt runs/exp/best.pt --to torchscript

- ``--ckpt`` — a checkpoint written by ``xnns train``
- ``--to`` — ``lammps`` (TorchScript wrapped in the LAMMPS tensor ABI) or
  ``torchscript`` (plain scripted model)

See :ref:`deployment` for what to do with the exported file.
