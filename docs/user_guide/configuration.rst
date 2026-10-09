.. _configuration:

*************
Configuration
*************

All configuration funnels into one dataclass tree
(:mod:`xnn.common.config.schema`): a :class:`~xnn.common.config.schema.Config`
with ``model``, ``data`` and ``optim`` sections. The complete schema with its
defaults, as YAML:

.. code-block:: yaml

   model:
     name: mace                 # any registered model name
     cutoff: 4.0                # neighbor-list radius, Å
     n_features: 32             # shared core sizes; the model may use its own names
     n_interactions: 2
     n_rbf: 8
     # every other key is model-specific and goes into model.extra, e.g.
     species: [1, 6, 8]
     atomic_energies: {1: -13.6, 6: -1029.9, 8: -2042.8}
     dispersion: {name: d4}     # add a D3 / D4 term (see Models)
     long_range: {n_channels: 4}   # add the LES term
     pretrained: mace-off23-small  # fine-tuning: start from a hub model
     lora: {rank: 4}               # and / or adapt it with LoRA
     heads: {pt_head: null, Default: {}}   # multi-head replay

   data:
     train_path: null           # structure files, read with ASE by the CLI
     val_path: null
     test_path: null            # optional, evaluated once after training
     cutoff: 4.0                # synchronized to model.cutoff
     batch_size: 16             # 1 disables batching
     num_workers: 0
     val_fraction: 0.1          # held out of the training set when val_path is absent
     test_fraction: 0.0         # same for the test set; 0 = none
     energy_key: energy         # names of the targets in the file (e.g. REF_energy)
     forces_key: forces
     stress_key: stress
     head: null                 # multi-head fine-tuning: head of the training structures
     replay_path: null          # replay structures and their head
     replay_head: pt_head
     replay_samples: null       # random subsample size
     replay_filter: subset      # none / subset / exact / superset (elements vs the training set)
     replay_pseudolabel: false  # relabel the replay set with the pretrained model

   optim:
     lr: 1.0e-3
     weight_decay: 0.0
     optimizer: adam            # adam / adamw
     epochs: 100
     energy_weight: 1.0         # loss weights; nonzero enables the head
     force_weight: 10.0
     stress_weight: 0.0
     dipole_weight: 0.0         # dipole / polarizability labels (PaiNN; PhysNet, AIMNet2 dipoles)
     polarizability_weight: 0.0
     huber_delta: 0.0           # > 0 clips the loss tails; per-term huber_delta_energy / _forces / _stress
     scheduler: plateau         # none / cosine / plateau
     clip_grad: 0.0             # max gradient norm per step; 0 = off
     ema_decay: 0.0             # > 0 keeps a weight average for validation and checkpoints
     head_weights: null         # per-head loss weights of a multi-head model
     freeze: []                 # parameter-name patterns to freeze
     train_only: []             # patterns of the only parameters to train

   device: auto                 # auto / cpu / cuda / cuda:0
   seed: 1234
   output_dir: runs/exp
   subtracted_dispersion: null  # what was subtracted from the labels, added back on deployment

The model options are described in :ref:`models`, the fine-tuning keys in
:ref:`howto-finetune`, the loss options in :ref:`training` and
``subtracted_dispersion`` in :ref:`deployment`.

Frontends
=========

.. code-block:: python

   from xnn.common.config import Config, from_yaml, from_argparse, from_hydra, apply_overrides

   cfg = Config()                                    # Python
   cfg = from_yaml("configs/train.yaml")             # YAML
   cfg = from_argparse(["--config", "train.yaml", "--set", "model.cutoff=6.0"])
   cfg = from_hydra(dict_config)                     # Hydra / OmegaConf (hydra extra)
   apply_overrides(cfg, {"optim.epochs": 50})        # dotted-key overrides

``xnn train --set KEY=VALUE`` applies the same overrides from the shell.
Keys spelled the upstream way (MACE's ``r_max``, NequIP's ``num_layers``)
are translated when the config loads (:ref:`howto-upstream-configs`).

Bundled configs
===============

.. code-block:: text

   configs/
     train.yaml          training config with a Hydra-style defaults list
     benchmark.yaml      benchmark config
     data/default.yaml
     model/{mace,nequip,allegro,cace,aimnet2,schnet,dimenet,dimenet_pp,painn,se3cnn,cnn3d,physnet,hdnnp,ani,bamboo,reaxff,opls,dreiding}.yaml
