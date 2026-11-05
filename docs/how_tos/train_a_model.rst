.. _howto-train-a-model:

*************
Train a Model
*************

xnn offers one :class:`~xnn.common.config.schema.Config` schema but three
interfaces to populate it: YAML, the command line, and Hydra.

Write a config file
===================
One of the simplest ways to control the training, evaluation, deployment and
benchmarking of a model in xnn is to write a YAML file. There are different
sections in the YAML file. For example, ``model``, ``data``, ``optim``,
``device``, ``seed``, and ``output_dir``. A ``model`` entry has two groups of
keywords: The core group which is general to all models, and the model-specific
group which is defined in the ``model.extra`` field. 

Below, we create a new config file ``my_train.yaml`` for training a MACE model
on argon data (``datasets/argon_md/`` in the repository).

.. code-block:: yaml

   # my_train.yaml

   model:
     name: mace
     cutoff: 4.0
     species: [18]
     n_features: 32
     n_interactions: 2
     max_ell: 3
     correlation: 3
     avg_num_neighbors: 20.0

   data:
     train_path: datasets/argon_md/argon_train.xyz
     test_path: datasets/argon_md/argon_test.xyz    # Optional; evaluated once after training
     energy_key: REF_energy
     forces_key: REF_forces
     stress_key: REF_stress
     batch_size: 8

   optim:
     lr: 1.0e-3
     epochs: 100
     energy_weight: 1.0
     force_weight: 10.0
     scheduler: plateau

   device: auto
   seed: 1234
   output_dir: runs/argon_mace

xnn comes with pre-made config files such as ``configs/train.yaml`` which can be
combined with other per-model YAML files from ``configs/model/`` to create
hierarchical composite Hydra-style files with a ``defaults`` list.

Run it
======

You can run the training with the command line interface using the config file
you just created:

.. code-block:: bash

   xnn train --config my_train.yaml
   xnn train --config my_train.yaml --set optim.epochs=50 model.cutoff=6.0

These command read the ASE structure files (``*.xyz``) which require ``ase``
extra dependency to be installed. The same training job can be driven from the
xnn's Python API:

.. code-block:: python

   from xnn.common.config import from_yaml
   from xnn.common.data import AtomicDataset
   from xnn.common.train import Trainer

   cfg = from_yaml("my_train.yaml")
   train = AtomicDataset.from_file(cfg.data.train_path, cfg.model.cutoff)
   Trainer(cfg, train).fit()

Multi-GPU Training
==================
As all xnn models are implemented in PyTorch, you can prepend the same xnn
command through with a ``torchrun`` driver to distribute the training on
multiple GPUs. The trainer will run the training in data-parallel mode by
default with no config changes:

.. code-block:: bash

   torchrun --nproc-per-node 4 -m xnn train --config my_train.yaml

Note that the ``data.batch_size`` is a per-process (per-GPU) variable. See
:ref:`training` for multi-node run settings.

Hydra
=====
With the ``hydra`` extra dependency installed, a Hydra-driven application can
hand its ``DictConfig`` object to xnn seamlessly:

.. code-block:: python

   from xnn.common.config import from_hydra

   cfg = from_hydra(hydra_dict_config)
