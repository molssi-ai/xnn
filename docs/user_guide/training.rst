.. _training:

********
Training
********

The Trainer
===========
:class:`~xnns.common.train.trainer.Trainer` owns the whole loop:

.. code-block:: python

   from xnns.common.train import Trainer

   trainer = Trainer(cfg, train_set, val_set, test_set)   # val/test optional
   metrics = trainer.fit()   # {"train": ..., "val": ..., "test": ...}

Both a train/val and a train/val/test workflow are supported: pass the
splits explicitly, or let the trainer carve them out of ``train_set`` —
when ``val_set`` (``test_set``) is ``None`` and ``data.val_fraction``
(``data.test_fraction``) is positive, that fraction is held out using a
single ``cfg.seed``-seeded permutation. The default ``test_fraction = 0``
means no test split unless you ask for one.

Constructing a ``Trainer``:

1. builds the model from ``cfg.model`` via
   :func:`~xnns.common.models.registry.build_model`;
2. wraps it in :class:`~xnns.common.models.outputs.ForceStressOutput`, with
   the force and stress heads enabled by nonzero ``optim.force_weight`` /
   ``optim.stress_weight``;
3. resolves the device (``cfg.device``, with ``"auto"`` choosing CUDA when
   available) and moves everything there;
4. sets up the Adam optimizer (``lr``, ``weight_decay``), the learning-rate
   scheduler (``cosine``, ``plateau``, or none), and batched data loaders
   using :func:`~xnns.common.data.dataset.collate`.

``fit()`` trains for ``cfg.optim.epochs`` epochs, validating each epoch when
a validation set is available, and writes checkpoints to
``cfg.output_dir``:

- ``best.pt`` — lowest validation loss so far
- ``last.pt`` — most recent epoch

A checkpoint is ``{"model": state_dict, "cfg": Config}``; load it with
``torch.load(path, weights_only=False)``.

When a test set exists, ``fit()`` evaluates it once after the final epoch
(with the final-epoch weights) and reports the test loss; the returned
metrics dict carries the numbers. To test the *best* checkpoint instead,
load it and call :meth:`~xnns.common.train.trainer.Trainer.evaluate`:

.. code-block:: python

   import os, torch

   state = torch.load(os.path.join(cfg.output_dir, "best.pt"), weights_only=False)
   trainer.model.load_state_dict(state["model"])
   test_metrics = trainer.evaluate()             # uses trainer.test_loader

The loss
========
:func:`~xnns.common.train.losses.weighted_loss` combines the per-property
mean-squared errors:

.. math::

   \mathcal{L} = w_E \, \mathcal{L}_\text{energy}
               + w_F \, \mathcal{L}_\text{forces}
               + w_\sigma \, \mathcal{L}_\text{stress}

with the weights from ``OptimConfig`` (defaults
:math:`w_E = 1`, :math:`w_F = 10`, :math:`w_\sigma = 0`). A property enters
the loss only when the dataset provides the target and its weight is
nonzero. Force training is strongly recommended whenever forces are
available — it is dramatically more data-efficient than energies alone.

Devices and batching
====================
``cfg.device = "auto" | "cpu" | "cuda" | "cuda:0"`` — resolved by
:func:`~xnns.common.train.trainer.resolve_device`. Batching is by graph
concatenation (see :ref:`data`); ``data.batch_size = 1`` disables batch
training entirely.

Reproducibility
===============
``cfg.seed`` seeds the run. Note that exact bit-reproducibility across
devices and CUDA versions is not guaranteed by PyTorch itself.
