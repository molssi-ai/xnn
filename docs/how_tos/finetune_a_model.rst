.. _howto-finetune:

******************************
Fine-tune a Pretrained Model
******************************

The ordinary training config fine-tunes any model that ``from_pretrained()``
loads: naive fine-tuning, layer freezing, LoRA and multi-head replay are all
config options. ``examples/gnn/mace/mace_finetuning_strategies.ipynb``
compares them on a MACE-OFF23 model. The classical force fields have no
readout to adapt and are refit directly (:ref:`howto-forcefield-files`).

Naive fine-tuning
=================
``model.pretrained`` names the starting model; the architecture comes from
the checkpoint:

.. code-block:: yaml

   model:
     pretrained: mace-off23-small    # or runs/exp/best.pt, a directory, a DOI
     cutoff: 5.0                     # must equal the checkpoint's
     atomic_energies: estimated      # re-set the E0s from the training data

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

Every weight trains. The resulting checkpoint loads back with
``from_pretrained("runs/exp/best.pt")``.

Reference energies
==================
A fine-tuning set at another level of theory differs from the pretraining
labels mostly by per-element constants, so re-set them before the fit:

- ``atomic_energies: estimated`` aligns the pretrained model's predictions
  with the new energies (model-aware reestimation).
- ``atomic_energies: average`` fits the energies to the compositions alone.
- A ``{Z: E0}`` mapping sets explicit values.

.. code-block:: python

   from xnn.common.finetune import estimate_atomic_energies, set_atomic_energies

   set_atomic_energies(model, estimate_atomic_energies(model, train_structures))

Freeze layers
=============
``fnmatch`` patterns of parameter names, with the ``model.`` prefix of the
force wrapper:

.. code-block:: yaml

   optim:
     train_only: ["*readouts*", "*atom_ref*"]        # readout-only fine-tuning
     # or
     freeze: ["model.node_embedding*", "model.interactions.0.*"]

LoRA
====
``model.lora`` freezes the pretrained weights and trains low-rank updates of
every linear layer, block by block per irrep so equivariance is kept:

.. code-block:: yaml

   model:
     pretrained: mace-off23-small
     cutoff: 5.0
     lora: {rank: 4, alpha: 1.0}       # or lora: 16
   optim:
     lr: 1.0e-2                        # LoRA takes a ten times larger rate
     ema_decay: 0.99
     clip_grad: 10.0

Checkpoints keep the adapters; ``from_pretrained``, ``xnn export`` and
``xnn mdi`` fold them into the base weights, so a deployed LoRA model has
the original architecture and cost. In Python,
:func:`~xnn.common.finetune.inject_lora` and
:func:`~xnn.common.finetune.merge_lora` do the same.

Multi-head replay
=================
The target data trains one readout head and a replay set of the pretraining
distribution trains another, over one shared trunk, so the model keeps the
foundation model's behaviour away from the target data:

.. code-block:: yaml

   model:
     pretrained: mace-mp-0-small
     cutoff: 6.0
     heads: {pt_head: null, Default: {atomic_energies: estimated}}

   data:
     train_path: target.xyz           # trains the Default head
     replay_path: mptrj_subset.xyz    # trains pt_head
     replay_filter: subset            # keep replay structures made of the target's elements
     replay_samples: 2000             # random subsample (seeded)
     replay_pseudolabel: true         # label the replay set with the pretrained model

   optim:
     lr: 1.0e-4
     energy_weight: 10.0
     force_weight: 10.0
     head_weights: {pt_head: {energy_weight: 1.0, force_weight: 10.0}}
     ema_decay: 0.9999
     clip_grad: 1.0

Head 0 keeps the pretrained readout; the others start as copies. With
``replay_pseudolabel`` any structurally diverse set serves as replay data.
``heads`` and ``lora`` combine. A multi-head checkpoint is served one head at
a time: ``from_pretrained(path, head="Default")``, ``xnn export --head
Default``, ``xnn mdi --head Default``. The Python side is
:class:`~xnn.common.finetune.MultiHead`,
:func:`~xnn.common.finetune.select_replay` and
:func:`~xnn.common.finetune.pseudolabel`.

Settings that work
==================
From the fine-tuning study of Tompa *et al.*: ``weight_decay: 0`` for every
method; constant loss weights (energy 10, forces 10); learning rates of 1e-3
(naive, freezing), 1e-2 (LoRA) and 1e-4 (multi-head replay), with
``ema_decay`` 0.999 / 0.99 / 0.9999 and ``clip_grad`` 1 / 10 / 1; the replay
head at energy 1, forces 10.
