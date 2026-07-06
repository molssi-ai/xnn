.. _howto-train-config:

****************************
Train from a Config File
****************************

xnns has one configuration schema — the
:class:`~xnns.common.config.schema.Config` dataclass — and three
interchangeable frontends to fill it: YAML, argparse, and Hydra.

Write a config file
===================
A YAML training config sets the ``model``, ``data``, and ``optim`` sections
(any key that is not a core :class:`~xnns.common.config.schema.ModelConfig`
field is folded into ``model.extra`` automatically):

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
     val_path: data/argon_test.xyz
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

The bundled ``configs/train.yaml`` additionally shows the Hydra-style
``defaults:`` list that composes per-model files from ``configs/model/`` and
data settings from ``configs/data/``.

Train from Python
=================

.. code-block:: python

   from xnns.common.config import from_yaml
   from xnns.common.data import AtomicDataset
   from xnns.common.train import Trainer

   cfg = from_yaml("my_train.yaml")
   Trainer(cfg, AtomicDataset(structures, cfg.model.cutoff)).fit()

Train from the command line
===========================
The ``xnns`` command reads the structure files named in ``data.train_path`` /
``data.val_path`` with ASE (requires the ``ase`` extra) and runs the same
trainer:

.. code-block:: bash

   xnns train --config my_train.yaml

Override any key at the command line
====================================
Every frontend supports dotted-key overrides, so a config file can stay
generic while runs vary:

.. code-block:: bash

   xnns train --config my_train.yaml --set optim.epochs=50 model.cutoff=6.0

or from Python:

.. code-block:: python

   from xnns.common.config import from_argparse

   cfg = from_argparse(["--config", "my_train.yaml", "--set", "model.cutoff=6.0"])

Use Hydra
=========
With the ``hydra`` extra installed, an existing Hydra application can hand
its ``DictConfig`` straight to xnns:

.. code-block:: python

   from xnns.common.config import from_hydra

   cfg = from_hydra(hydra_dict_config)
