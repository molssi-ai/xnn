.. _cli:

****************
The Command Line
****************

Installing xnn provides the ``xnn`` command with five subcommands. Reading
structure files needs the ``ase`` extra dependency installed. Every ``--ckpt``
accepts a trainer checkpoint, a portable model directory, a model hub name, a
URL or a Zenodo DOI.

xnn train
=========

You can train a model with the ``xnn train`` command. It reads the training,
validation and test datasets, trains the model, and writes the best and last
checkpoints to the specified output directory. You can override any config key
with the ``--set KEY=VALUE`` option. For distributed training, use ``torchrun``
with ``--nproc-per-node N`` to launch N processes on one node. See
:ref:`howto-train-a-model` guide for more information.

.. code-block:: bash

   xnn train --config configs/train.yaml
   xnn train --config configs/train.yaml --set optim.epochs=50 model.cutoff=6.0
   torchrun --nproc-per-node 2 -m xnn train --config configs/train.yaml   # data parallel

xnn benchmark
=============

Benchmark a model on one or more datasets with the ``xnn benchmark`` command. It
reads the datasets, runs the model and writes a results table in every
configured format.

.. code-block:: bash

   xnn benchmark --config configs/benchmark.yaml
   xnn benchmark --config configs/benchmark.yaml --set "metrics={'energy': ['mae']}"

See the :ref:`howto-benchmark` guide for more details and options.

xnn export
==========

The ``xnn export`` command writes a self-contained TorchScript file with the
whole-system entry point ``forward`` and the pair-style ``forward_lammps``. It
accepts a trainer checkpoint, a portable model directory, a model hub name, a
URL or a Zenodo DOI.

.. code-block:: bash

   xnn export --ckpt runs/exp/best.pt --out deployed.pt
   xnn export --ckpt mace-off23-small --out mace_off.pt

You can set the ``--config`` flag for a checkpoint without an embedded config,
``--head`` for a multi-head checkpoint, ``--total-charge`` and
``--spin-multiplicity`` for fixing the charge and spin states, and
``--no-dispersion`` to skip a recorded subtracted dispersion. See the
:ref:`deployment` page for more details and options.

xnn mdi
=======
The ``xnn mdi`` command serves a checkpoint as a `MolSSI Driver Interface
<https://github.com/MolSSI-MDI/MDI_Library>`_ engine (requires ``mdi`` extra
dependency installed) for LAMMPS ``fix mdi/qm`` or any MDI-compatible MD driver.

.. code-block:: bash

   xnn mdi --ckpt runs/exp/best.pt -mdi "-role ENGINE -name xnn -method TCP -port 8021 -hostname localhost"

You can set the ``--device``, and ``--dtype`` (default: the checkpoint's), and
use the ``--dispersion d3|d4|"{...}"`` flag to add a dispersion term to a plain
short-range model checkpoint. AMOEBA and HIPPO also take ``--terms
valence|nonbonded`` for RESPA drivers and ``--dtype mixed``. See the
:ref:`deployment` page for more details and options.

xnn models
==========

The ``xnn models`` command lists, shows information, downloads and packs models. 

.. code-block:: bash

   xnn models list --cached                 # Registered and cached models
   xnn models info mace-mh-0                # The model card
   xnn models pull mace-off23-small         # Download and convert into the cache
   xnn models pack runs/exp/best.pt my-model --license MIT --archive

See :ref:`howto-pretrained-models` for model details.
