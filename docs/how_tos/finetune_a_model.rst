.. _howto-finetune:

******************************
Fine-tune a Pretrained Model
******************************

Every fine-tuning strategy of the MACE fine-tuning protocols is available
for xnn models through the ordinary training config and the
:mod:`xnn.common.finetune` tools: naive fine-tuning, layer freezing, LoRA,
multi-head replay (with original or pseudolabelled replay data) and the
reference-energy estimators. The strategies apply to every model that
declares its readout (MACE, NequIP, Allegro, CACE, AIMNet2, SchNet,
PhysNet, BAMBOO, HDNNP/ANI); the classical force fields have no readout
and no linear layers to adapt. ``examples/gnn/mace/mace_finetuning_strategies.ipynb``
runs the strategies side by side on a MACE-OFF23 foundation model.

Start from any pretrained model
===============================

``model.pretrained`` names anything :func:`~xnn.common.models.from_pretrained`
loads (a registered foundation model, a trainer checkpoint, a portable model
directory, a URL or a Zenodo DOI). The architecture comes from the
checkpoint; the config only needs the model's cutoff and the fine-tuning
options:

.. code-block:: yaml

   model:
     pretrained: mace-off23-small     # or runs/exp/best.pt, a directory, a DOI
     cutoff: 5.0                      # must equal the checkpoint's
     dtype: float32                   # optional cast of the weights
     atomic_energies: estimated       # re-set the E0s from the training data

   data:
     train_path: ethanol_train.xyz
     batch_size: 5

   optim:
     lr: 1.0e-3
     epochs: 200
     energy_weight: 10.0
     force_weight: 10.0
     weight_decay: 0.0
     ema_decay: 0.999
     clip_grad: 1.0
     optimizer: adamw

This is naive fine-tuning: every weight trains. The checkpoint written by
the trainer records the resolved architecture, so it rebuilds from its
weights alone, and loads back with ``from_pretrained("runs/exp/best.pt")``.

Reference energies
==================

A fine-tuning set computed at another level of theory differs from the
pretraining labels mostly by per-element constants, so the reference
energies are re-set before the fit. ``atomic_energies: estimated`` solves
for the per-element corrections that align the *pretrained model's
predictions* with the new energies (the model-aware reestimation of
Tompa *et al.*); ``atomic_energies: average`` fits the energies to the
compositions alone; a ``{Z: E0}`` mapping sets explicit values (for example
isolated-atom energies at the new level of theory). From Python:

.. code-block:: python

   from xnn.common.finetune import estimate_atomic_energies, set_atomic_energies

   e0 = estimate_atomic_energies(model, train_structures)   # {Z: E0}
   set_atomic_energies(model, e0)

Freeze layers
=============

``optim.freeze`` and ``optim.train_only`` take ``fnmatch`` patterns of
parameter names (as printed by ``model.named_parameters()``, with the
``model.`` prefix of the force wrapper):

.. code-block:: yaml

   optim:
     train_only: ["*readouts*", "*atom_ref*"]      # readout-only fine-tuning
     # or
     freeze: ["model.node_embedding*", "model.interactions.0.*"]

LoRA
====

``model.lora`` freezes the pretrained weights and trains low-rank updates of
every linear layer, including the equivariant ones (block by block per
irrep, so equivariance is kept):

.. code-block:: yaml

   model:
     pretrained: mace-off23-small
     cutoff: 5.0
     lora: {rank: 4, alpha: 1.0}       # or lora: 16; trainable: ["*atom_ref*"] keeps the E0s free
   optim:
     lr: 1.0e-2                        # LoRA tolerates a ten times larger learning rate
     ema_decay: 0.99
     clip_grad: 10.0

The adapted model computes exactly what the pretrained one did until the
updates are trained. Its checkpoints keep the adapters (``lora_in`` /
``lora_out`` parameters); ``from_pretrained``, ``xnn export`` and ``xnn mdi``
fold them into the base weights, so a deployed LoRA model has the original
architecture and inference cost. From Python,
:func:`~xnn.common.finetune.inject_lora` and
:func:`~xnn.common.finetune.merge_lora` do the same on a model object.

Multi-head replay
=================

Multi-head replay trains the target data on one readout head and a replay
set of the pretraining distribution on another, over one shared trunk, so
the model keeps the foundation model's behaviour away from the target data.
Head 0 keeps the pretrained readout and reference energies; the other heads
start as copies of it:

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

The loss sums every head's terms, each averaged over that head's own
structures, with ``head_weights`` giving a head its own energy, force and
stress weights. Replay structures with their original labels are the
replay-fine-tuning protocol; with ``replay_pseudolabel`` any structurally
diverse set serves, and the replay head reproduces the pretrained model.
Combine ``heads`` with ``lora`` for multi-head LoRA. From Python:

.. code-block:: python

   from xnn.common.finetune import MultiHead, pseudolabel, select_replay

   model = from_pretrained("mace-mp-0-small", wrap=False)
   multi = MultiHead(model, ["pt_head", "Default"])
   replay = pseudolabel(multi, select_replay(candidates, species_of(target), n=2000))
   multi.label(target, "Default"); multi.label(replay, "pt_head")

A multi-head checkpoint is served one head at a time:
``from_pretrained(path, head="Default")``, ``xnn export --head Default``,
``xnn mdi --head Default``; ``multi.select("Default")`` returns the plain
single-head potential in Python. ``torchrun`` data-parallel training works
as for any other model (a batch missing one head is handled).

Hyperparameters
===============

The settings that the fine-tuning study of Tompa *et al.* found stable, as
xnn options: ``weight_decay: 0.0`` for every method; constant target loss
weights (``energy_weight: 10, force_weight: 10``) rather than a two-stage
schedule; ``lr`` of 1e-3 (naive, freezing), 1e-2 (LoRA) and 1e-4
(multi-head replay); ``ema_decay`` 0.999, 0.99 and 0.9999 with ``clip_grad``
1, 10 and 1 respectively; the replay head at ``energy_weight: 1,
force_weight: 10``.
