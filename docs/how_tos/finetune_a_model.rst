.. _howto-finetune:

******************************
Fine-tune a Pretrained Model
******************************

The ordinary training config files allow for fine-tuning any model that the
xnn's ``from_pretrained()`` can load. A variety of fine-tuning methods are
available: naive fine-tuning, layer freezing, LoRA and multi-head replay all of
which can also be specified as config options.

The ``notebooks/examples/gnn/mace/mace_finetuning_strategies.ipynb`` notebook compares
these methods via fine-tuning a MACE-OFF23 foundation model. The classical force
fields have no readout layer to be exposed to adapters. Therefore, they should
be fine-tuned with the naive or freezing methods (see
:ref:`howto-forcefield-files`).

Naive fine-tuning
=================
The ``model.pretrained`` names the starting model; the architecture comes from
the checkpoint:

.. code-block:: yaml

   model:
     pretrained: mace-off23-small    # Or runs/exp/best.pt, a directory, a DOI
     cutoff: 5.0                     # Must equal the checkpoint's
     atomic_energies: estimated      # Set the E0s from the training data statistics

   data:
     train_path: ethanol_train.xyz
     batch_size: 5

   optim:
     lr: 1.0e-3
     epochs: 200
     energy_weight: 10.0
     force_weight: 10.0
     optimizer: adamw
     weight_decay: 0.0
     ema_decay: 0.999
     clip_grad: 1.0

By default, every weight contributes to the training process. The resulting
fine-tuned model checkpoint can be loaded back with
``from_pretrained("runs/exp/best.pt")``.

Reference energies
==================
A fine-tuning dataset with labels calculated at a different level of theory from
those in the pretraining data will mostly differ by per-element constants. As
such, you can re-set them before the starting the training using the
``atomic_energies`` option in the model config:

- Setting ``atomic_energies: estimated`` aligns the pretrained model's predictions
  with the new energies (model-aware re-estimation), and
- ``atomic_energies: average`` fits the energies to the compositions alone.
- A ``{Z: E0}`` mapping allows you to set the atomic energies to explicit values
  per element. In Python, the same task can be done as follows:

.. code-block:: python

   from xnn.common.finetune import estimate_atomic_energies, set_atomic_energies

   set_atomic_energies(model, estimate_atomic_energies(model, train_structures))

Freezing layers
===============
The ``fnmatch`` pattern for parameter names, with the ``model.`` prefix allows
you to freeze any subset of the model's weights. The ``train_only`` option is
the complement of ``freeze``:

.. code-block:: yaml

   optim:
     # Readout-only fine-tuning
     train_only: ["*readouts*", "*atom_ref*"]
     # Or freeze specific layers
     freeze: ["model.node_embedding*", "model.interactions.0.*"]

LoRA
====
The ``model.lora`` option allows you to freeze the pretrained weights and train
using low-rank updates to every linear layer, block by block, per irrep. So,
even using LoRA adapters, the model's equivariance is preserved.

.. code-block:: yaml

   model:
     pretrained: mace-off23-small
     cutoff: 5.0
     lora: {rank: 4, alpha: 1.0}       # Or lora: 16
   optim:
     lr: 1.0e-2                        # LoRA takes a ten times larger learning-rate
     ema_decay: 0.99
     clip_grad: 10.0

The model checkpoint artifacts preserve the adapters. Therefore,
``from_pretrained``, ``xnn export`` and ``xnn mdi`` fold them into the base
weights such that a deployed LoRA model mimics the behavior of the original
architecture. In xnn's Python API, :func:`~xnn.common.finetune.inject_lora` and
:func:`~xnn.common.finetune.merge_lora` perform the same operations.

Multi-head replay
=================
In this mode, the target data trains one readout head and a replay set of the
pre-training distribution trains another, over one shared model backbone. So,
the model can keeps its foundational behaviour while being fine-tuned on the
target data:

.. code-block:: yaml

   model:
     pretrained: mace-mp-0-small
     cutoff: 6.0
     heads: {pt_head: null, Default: {atomic_energies: estimated}}

   data:
     train_path: target.xyz           # Trains the Default head
     replay_path: mptrj_subset.xyz    # Trains pt_head
     replay_filter: subset            # Keep replay structures made of the target's elements
     replay_samples: 2000             # Random subsample (seeded)
     replay_pseudolabel: true         # Label the replay set with the pretrained model

   optim:
     lr: 1.0e-4
     energy_weight: 10.0
     force_weight: 10.0
     head_weights: {pt_head: {energy_weight: 1.0, force_weight: 10.0}}
     ema_decay: 0.9999
     clip_grad: 1.0

Head with index ``0`` keeps the pre-trained readout layer. The other heads start
as copies of the pre-trained layer. With ``replay_pseudolabel``, any
structurally diverse set serves as a replay data. The ``heads`` and ``lora``
options can be combined. A multi-head checkpoint is served one head at a time in
all interfaces: ``from_pretrained(path, head="Default")``, ``xnn export --head
Default``, and ``xnn mdi --head Default``. The Python API side is mainly handled
by :class:`~xnn.common.finetune.MultiHead` class, as well as the helper
functions :func:`~xnn.common.finetune.select_replay` and
:func:`~xnn.common.finetune.pseudolabel`.

Settings that work
==================
From the fine-tuning study of `Tompa et al., arXiv:2606.12704 (2026)
<https://arxiv.org/abs/2606.12704>`_
the following settings are recommended for fine-tuning MACE models:
``weight_decay: 0`` for every method; constant loss weights (energy 10, forces
10); learning rates of 1e-3 (naive, freezing), 1e-2 (LoRA) and 1e-4 (multi-head
replay); ``ema_decay`` 0.999 / 0.99 / 0.9999; ``clip_grad`` 1 / 10 / 1; and the
replay head weight set energy at 1, and forces at 10.
