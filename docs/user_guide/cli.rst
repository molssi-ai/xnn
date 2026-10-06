.. _cli:

****************
The Command Line
****************

Installing xnn provides the ``xnn`` command with five subcommands. Reading
structure files needs the ``ase`` extra. Every ``--ckpt`` accepts a trainer
checkpoint, a portable model directory, a hub name, a URL or a Zenodo DOI.

xnn train
=========

.. code-block:: bash

   xnn train --config configs/train.yaml
   xnn train --config configs/train.yaml --set optim.epochs=50 model.cutoff=6.0
   torchrun --nproc-per-node 2 -m xnn train --config configs/train.yaml   # data parallel

Reads ``data.train_path`` / ``val_path`` / ``test_path`` (and
``replay_path`` for multi-head fine-tuning) with ASE, trains, and writes
``best.pt`` and ``last.pt`` to ``output_dir``. ``--set KEY=VALUE`` overrides
any config key. See :ref:`howto-train-a-model`.

xnn benchmark
=============

.. code-block:: bash

   xnn benchmark --config configs/benchmark.yaml
   xnn benchmark --config configs/benchmark.yaml --set "metrics={'energy': ['mae']}"

Scores pre-trained checkpoints on one dataset and writes the results table in
every configured format. See :ref:`howto-benchmark`.

xnn export
==========

.. code-block:: bash

   xnn export --ckpt runs/exp/best.pt --out deployed.pt
   xnn export --ckpt mace-off23-small --out mace_off.pt

Writes a self-contained TorchScript file with the whole-system entry point
``forward`` and the pair-style ``forward_lammps``. Options: ``--config`` for
a checkpoint without an embedded config, ``--head`` for a multi-head
checkpoint, ``--total-charge`` and ``--spin-multiplicity`` for the charge
state, ``--no-dispersion`` to skip a recorded subtracted dispersion. See
:ref:`howto-lammps`.

xnn mdi
=======

.. code-block:: bash

   xnn mdi --ckpt runs/exp/best.pt -mdi "-role ENGINE -name xnn -method TCP -port 8021 -hostname localhost"

Serves a checkpoint as a `MolSSI Driver Interface
<https://github.com/MolSSI-MDI/MDI_Library>`_ engine (``mdi`` extra) for
LAMMPS ``fix mdi/qm`` or any MDI driver. Options: ``--device``, ``--dtype``
(default: the checkpoint's), ``--dispersion d3|d4|"{...}"`` to add a term
to a plain checkpoint, ``--no-dispersion``, ``--head``, ``--total-charge``,
``--eeq-reuse`` (carry the D4 charge solve between MD steps) and ``--fast
auto|on|off``. See :ref:`deployment`.

xnn models
==========

.. code-block:: bash

   xnn models list --cached                 # registered and cached models
   xnn models info mace-mh-0                # the model card
   xnn models pull mace-off23-small         # download and convert into the cache
   xnn models pack runs/exp/best.pt my-model --license MIT --archive

See :ref:`howto-pretrained-models`.
