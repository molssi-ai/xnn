.. _training:

********
Training
********

The Trainer
===========
:class:`~xnn.common.train.trainer.Trainer` owns the whole loop:

.. code-block:: python

   from xnn.common.train import Trainer

   trainer = Trainer(cfg, train_set, val_set, test_set)   # val and test optional
   metrics = trainer.fit()                                # {"train": ..., "val": ..., "test": ...}

It builds the model from ``cfg.model``, wraps it in
:class:`~xnn.common.models.outputs.ForceStressOutput` (force and stress
heads follow the nonzero loss weights), resolves the device, and sets up the
optimizer, scheduler and batched loaders. When no validation or test set is
given, ``data.val_fraction`` and ``data.test_fraction`` carve them out of
the training set with the run's seed.

``fit()`` validates every epoch and writes ``best.pt`` (lowest validation
loss) and ``last.pt`` to ``cfg.output_dir``. A test set is evaluated once
after the last epoch. A checkpoint holds the state dict and the config, and
loads with ``from_pretrained(path)``; to score the best checkpoint on the
test set, load it and call :meth:`~xnn.common.train.trainer.Trainer.evaluate`.
``trainer.save_pretrained("my-model")`` writes a portable model directory
(:ref:`howto-pretrained-models`).

The loss
========
A weighted sum of the per-property errors:

.. math::

   \mathcal{L} = w_E \, \mathcal{L}_\text{energy}
               + w_F \, \mathcal{L}_\text{forces}
               + w_\sigma \, \mathcal{L}_\text{stress}

with the defaults :math:`w_E = 1`, :math:`w_F = 10`, :math:`w_\sigma = 0`.
A property enters only when the data provides it and its weight is nonzero.
Train on forces whenever you have them; it is far more data-efficient than
energies alone. The energy term is per atom and averages over structures;
the force term averages over atoms.

- **Per-structure weights.** A ``weight`` in the structure dictionary
  scales that structure's share of every term, for instance to balance a
  sparse set against a dense scan. Uniform weights reproduce the plain loss.
- **Huber tails.** ``optim.huber_delta`` (or the per-term
  ``huber_delta_energy`` / ``_forces`` / ``_stress``) replaces the squared
  error by a function that is quadratic up to :math:`\delta` and linear
  beyond, which caps the pull of a few large residuals. It is scaled to
  agree with :math:`d^2` below :math:`\delta`, twice the textbook Huber, so
  halve ``delta`` values ported from MACE.
- **Several heads.** With head labels in the batch the loss sums each head's
  terms over that head's own structures; ``optim.head_weights`` gives a head
  its own weights (:ref:`howto-finetune`).

Optimizer, clipping, weight averaging
=====================================
``optim.optimizer`` selects Adam, AMSGrad or AdamW (decoupled weight decay),
``optim.clip_grad`` clips the gradient norm, and ``optim.ema_decay`` keeps an
exponential moving average of the weights that validation and the
checkpoints use. ``optim.freeze`` and ``optim.train_only`` select trainable
parameters by name pattern. All are off by default; the fine-tuning recipes
rely on them. ``model.pretrained``, ``model.lora`` and ``model.heads`` turn
the same trainer into a fine-tuning run (:ref:`howto-finetune`).

Several GPUs or nodes
=====================
Distributed training is native PyTorch DDP and needs no code or config
changes, only a launcher that sets ``RANK``, ``LOCAL_RANK`` and
``WORLD_SIZE``:

.. code-block:: bash

   torchrun --nproc-per-node 2 -m xnn train --config train.yaml

   torchrun --nnodes 2 --nproc-per-node 4 \
            --rdzv-backend c10d --rdzv-endpoint "$HEAD_NODE":29500 \
            -m xnn train --config train.yaml

Each rank is pinned to its GPU, every loader is sharded, metrics are
all-reduced so the best-checkpoint decision and the scheduler agree across
ranks, and only rank 0 prints and writes. Checkpoints hold the bare model
and are interchangeable with serial runs. ``data.batch_size`` is per
process, so the effective batch is ``batch_size × WORLD_SIZE``. Sharded
strategies (FSDP, DeepSpeed) are not supported: force and stress losses
need a double backward, which only DDP provides.

Reproducibility
===============
``cfg.seed`` seeds the run. Bit-exact results across devices and CUDA
versions are not guaranteed by PyTorch itself.
