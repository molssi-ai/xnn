.. _configuration:

*************
Configuration
*************

All configuration details funnels into one dataclass tree
(:mod:`xnn.common.config.schema`). A :class:`~xnn.common.config.schema.Config`
class object with ``model``, ``data`` and ``optim`` sections exposes the
complete schema with its defaults, in Python. The same schema is also available
in YAML format, and can be overridden from YAML files, command line arguments,
or Hydra / OmegaConf dictionaries. The following example shows a typical
training configuration in YAML format, with the default values for each section.

.. code-block:: yaml

   model:
     name: mace                 # Any registered model name
     cutoff: 4.0                # Neighbor-list radius, Å
     n_features: 32             # Shared core sizes; the model may use its own names
     n_interactions: 2
     n_rbf: 8
     # Every other key is model-specific and goes into model.extra, e.g.
     species: [1, 6, 8]
     atomic_energies: {1: -13.6, 6: -1029.9, 8: -2042.8}
     dispersion: {name: d4}        # Add a D3 / D4 term (see Models)
     long_range: {n_channels: 4}   # Add the LES term
     pretrained: mace-off23-small  # Fine-tuning: start from a hub model
     lora: {rank: 4}               # And / or adapt it with LoRA
     heads: {pt_head: null, Default: {}}   # Multi-head replay

   data:
     train_path: null           # Structure files, read with ASE by the CLI
     val_path: null
     test_path: null            # Optional, evaluated once after training
     cutoff: 4.0                # Synchronized to model.cutoff
     batch_size: 16             # 1 disables batching
     num_workers: 0
     val_fraction: 0.1          # Held out of the training set when val_path is absent
     test_fraction: 0.0         # Same for the test set; 0 = none
     energy_key: energy         # Names of the targets in the file (e.g. REF_energy)
     forces_key: forces
     stress_key: stress
     head: null                 # Multi-head fine-tuning: head of the training structures
     replay_path: null          # Replay structures and their head
     replay_head: pt_head
     replay_samples: null       # Random subsample size
     replay_filter: subset      # None / subset / exact / superset (elements vs the training set)
     replay_pseudolabel: false  # Re-label the replay set with the pretrained model

   optim:
     lr: 1.0e-3
     weight_decay: 0.0
     optimizer: adam            # adam / amsgrad / adamw
     epochs: 100
     energy_weight: 1.0         # Loss weights; nonzero enables the head
     force_weight: 10.0
     stress_weight: 0.0
     dipole_weight: 0.0         # Dipole / polarizability labels (PaiNN; PhysNet, AIMNet2 dipoles)
     polarizability_weight: 0.0
     charge_weight: 0.0         # Partial-charge labels (HDNNP 3G / 4G; any model with charges)
     les_dipole_weight: 0.0     # LES latent dipole against dipole labels
     les_charge_weight: 0.0     # LES latent charges against charges labels
     huber_delta: 0.0           # > 0 clips the loss tails; per-term huber_delta_energy / _forces / _stress
     scheduler: plateau         # none / cosine / plateau
     clip_grad: 0.0             # Max gradient norm per step; 0 = off
     ema_decay: 0.0             # > 0 keeps a weight average for validation and checkpoints
     head_weights: null         # Per-head loss weights of a multi-head model
     freeze: []                 # Parameter-name patterns to freeze
     train_only: []             # Patterns of the only parameters to train
     prior_weight: 0.0          # > 0 pulls the trainable parameters toward their starting values
     prior_weights: null        # The same per parameter-name pattern, {pattern: weight}

   trainer:
     loggers: [console]         # Any of console / csv / tensorboard
     log_every_n_epochs: 1
     log_every_n_steps: 0       # > 0 also logs training values every n steps
     step_average: batch        # Step values: batch / running (the epoch so far)
     metrics: [mae, rmse]       # Error metrics besides the loss terms
     monitor: val/loss          # Picks best.pt, drives plateau and early stopping
     mode: min                  # min / max
     val_every_n_epochs: 1
     save_every_n_epochs: 0     # > 0 also keeps epoch_<n>.pt
     early_stopping_patience: 0 # Validations without improvement; 0 = off
     early_stopping_min_delta: 0.0
     early_stopping_threshold: null
     progress_bar: false

   device: auto                 # auto / cpu / cuda / cuda:0
   seed: 1234
   output_dir: runs/exp
   subtracted_dispersion: null  # What was subtracted from the labels, added back on deployment

The model options are described in :ref:`models`, the fine-tuning keys in
:ref:`howto-finetune`, the loss, metric, checkpoint and logging options in
:ref:`training` and ``subtracted_dispersion`` in :ref:`deployment`.

Frontends
=========
The configuration can be loaded from Python, YAML, command line arguments, or
Hydra / OmegaConf dictionaries. The following example shows how to load a
configuration from each of these frontends, and how to apply dotted-key
overrides from Python.

.. code-block:: python

   from xnn.common.config import Config, from_yaml, from_argparse, from_hydra, apply_overrides

   cfg = Config()                                    # Python
   cfg = from_yaml("configs/train.yaml")             # YAML
   cfg = from_argparse(["--config", "train.yaml", "--set", "model.cutoff=6.0"])
   cfg = from_hydra(dict_config)                     # Hydra / OmegaConf (hydra extra)
   apply_overrides(cfg, {"optim.epochs": 50})        # Dotted-key overrides

The command ``xnn train --set KEY=VALUE`` applies the same overrides from a
shell. Keys which are spelled using the upstream code's convention (e.g., MACE's
``r_max``, NequIP's ``num_layers``) are automatically translated to xnn's
convention when the config loads at the runtime (See
:ref:`howto-upstream-configs` for details).

Bundled configs
===============
xnn ships with a set of pre-built configs to get you started with training and
benchmarking models. The configs are organized in a directory structure as follows:

.. code-block:: text

   configs/
     train.yaml          Training config with a Hydra-style defaults list
     benchmark.yaml      Benchmark config
     data/default.yaml
     model/{mace, nequip, allegro, cace, aimnet2, ...}.yaml
