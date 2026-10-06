.. _howto-train-a-model:

*************
Train a Model
*************

One :class:`~xnn.common.config.schema.Config` schema, three ways to fill it:
YAML, the command line, and Hydra.

Write a config file
===================
Any ``model`` key that is not a core field goes into ``model.extra``, so model
options are written flat:

.. code-block:: yaml

   # my_train.yaml
   model:
     name: mace
     cutoff: 4.0
     species: [18]
     num_channels: 32
     num_interactions: 2
     max_ell: 3
     correlation: 3
     avg_num_neighbors: 20.0

   data:
     train_path: data/argon_train.xyz
     val_path: data/argon_val.xyz
     test_path: data/argon_test.xyz    # optional, evaluated once after training
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

The bundled ``configs/train.yaml`` composes per-model files from
``configs/model/`` with a Hydra-style ``defaults`` list.

Run it
======

.. code-block:: bash

   xnn train --config my_train.yaml
   xnn train --config my_train.yaml --set optim.epochs=50 model.cutoff=6.0

The command reads the structure files with ASE (``ase`` extra). The same from
Python:

.. code-block:: python

   from xnn.common.config import from_yaml
   from xnn.common.data import AtomicDataset
   from xnn.common.train import Trainer

   cfg = from_yaml("my_train.yaml")
   train = AtomicDataset.from_file(cfg.data.train_path, cfg.model.cutoff)
   Trainer(cfg, train).fit()

Several GPUs
============
Start the same command through ``torchrun`` and the trainer runs data
parallel with no config changes:

.. code-block:: bash

   torchrun --nproc-per-node 4 -m xnn train --config my_train.yaml

``data.batch_size`` is per process. See :ref:`training` for multi-node runs.

Hydra
=====
With the ``hydra`` extra, a Hydra application hands its ``DictConfig`` to xnn
directly:

.. code-block:: python

   from xnn.common.config import from_hydra

   cfg = from_hydra(hydra_dict_config)
