.. _configuration:

*************
Configuration
*************

One schema
==========
All configuration funnels into a single dataclass tree
(:mod:`xnns.common.config.schema`):

.. list-table::
   :header-rows: 1
   :widths: 20 22 58

   * - Section
     - Field
     - Default / meaning
   * - ``Config``
     - ``model`` / ``data`` / ``optim``
     - the three sections below
   * -
     - ``device``
     - ``"auto"``: ``auto | cpu | cuda | cuda:0 ...``
   * -
     - ``seed``
     - ``1234``
   * -
     - ``output_dir``
     - ``"runs/exp"``: where checkpoints are written
   * - ``ModelConfig``
     - ``name``
     - ``"mace"``: any registered model name
   * -
     - ``cutoff``
     - ``4.0`` Å
   * -
     - ``n_features`` / ``n_interactions`` / ``n_rbf``
     - ``32`` / ``2`` / ``8``: shared core sizes
   * -
     - ``extra``
     - dict of model-specific options (see :ref:`models`)
   * - ``DataConfig``
     - ``train_path`` / ``val_path`` / ``test_path``
     - structure files (read with ASE by the CLI); ``test_path`` is optional
       and evaluated once after training
   * -
     - ``cutoff``
     - synchronized to ``model.cutoff`` automatically
   * -
     - ``batch_size``
     - ``16`` (``1`` disables batch training)
   * -
     - ``num_workers``
     - ``0``
   * -
     - ``val_fraction`` / ``test_fraction``
     - ``0.1`` / ``0.0``: fractions of the training set held out when no
       ``val_path`` / ``test_path`` is given (``0`` disables the split)
   * -
     - ``energy_key`` / ``forces_key`` / ``stress_key``
     - ``"energy"`` / ``"forces"`` / ``"stress"``: names the targets are
       stored under in the file (e.g. ``REF_energy`` for MACE-style datasets)
   * - ``OptimConfig``
     - ``lr`` / ``weight_decay``
     - ``1e-3`` / ``0.0`` (Adam)
   * -
     - ``epochs``
     - ``100``
   * -
     - ``energy_weight`` / ``force_weight`` / ``stress_weight``
     - ``1.0`` / ``10.0`` / ``0.0``: loss weights; nonzero enables the head
   * -
     - ``scheduler``
     - ``"plateau"``: ``cosine | plateau |`` none

When loading from a file, any ``model`` key that is not a core
``ModelConfig`` field is folded into ``model.extra``, so model options are
written flat.

Frontends
=========
.. code-block:: python

   from xnns.common.config import (
       Config, from_dict, from_yaml, from_argparse, from_hydra,
       apply_overrides,
   )

   cfg = Config()                                   # pure Python
   cfg = from_yaml("configs/train.yaml")            # YAML
   cfg = from_argparse(["--config", "configs/train.yaml",
                        "--set", "model.cutoff=6.0"])  # CLI-style
   cfg = from_hydra(dict_config)                    # Hydra / OmegaConf

``apply_overrides(cfg, {"optim.epochs": 50})`` applies dotted-key overrides
to an existing config; ``--set KEY=VALUE`` (repeatable) does the same from
the command line.

Bundled configs
===============
The repository ships composable templates:

.. code-block:: text

   configs/
     train.yaml        top-level training config (Hydra-style defaults list)
     data/default.yaml
     model/{mace,nequip,allegro,cace,schnet,physnet,hdnnp,ani,bamboo}.yaml

Upstream key translation
========================
Model keys copied verbatim from an upstream code's YAML also work: a
per-model key-translation registry
(:mod:`xnns.common.config.translate`) rewrites foreign spellings (MACE-CLI
``r_max`` / ``num_radial_basis`` / ``atomic_numbers`` / ``E0s``, NequIP
``num_layers``, ...) to the xnns canonical names at load time. The xnns
spelling wins if both are given. See :ref:`howto-upstream-configs`.
