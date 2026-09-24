.. _training:

********
Training
********

The Trainer
===========
:class:`~xnn.common.train.trainer.Trainer` owns the whole loop:

.. code-block:: python

   from xnn.common.train import Trainer

   trainer = Trainer(cfg, train_set, val_set, test_set)   # val/test optional
   metrics = trainer.fit()   # {"train": ..., "val": ..., "test": ...}

Both a train/val and a train/val/test workflow are supported: pass the
splits explicitly, or let the trainer carve them out of ``train_set``:
when ``val_set`` (``test_set``) is ``None`` and ``data.val_fraction``
(``data.test_fraction``) is positive, that fraction is held out using a
single ``cfg.seed``-seeded permutation. The default ``test_fraction = 0``
means no test split unless you ask for one.

Constructing a ``Trainer``:

1. builds the model from ``cfg.model`` via
   :func:`~xnn.common.models.registry.build_model`;
2. wraps it in :class:`~xnn.common.models.outputs.ForceStressOutput`, with
   the force and stress heads enabled by nonzero ``optim.force_weight`` /
   ``optim.stress_weight``;
3. resolves the device (``cfg.device``, with ``"auto"`` choosing CUDA when
   available) and moves everything there;
4. sets up the Adam optimizer (``lr``, ``weight_decay``), the learning-rate
   scheduler (``cosine``, ``plateau``, or none), and batched data loaders
   using :func:`~xnn.common.data.dataset.collate`.

``fit()`` trains for ``cfg.optim.epochs`` epochs, validating each epoch when
a validation set is available, and writes checkpoints to
``cfg.output_dir``:

- ``best.pt``: lowest validation loss so far
- ``last.pt``: most recent epoch

A checkpoint is ``{"model": state_dict, "cfg": Config}``; load it with
``torch.load(path, weights_only=False)``.

When a test set exists, ``fit()`` evaluates it once after the final epoch
(with the final-epoch weights) and reports the test loss; the returned
metrics dict carries the numbers. To test the *best* checkpoint instead,
load it and call :meth:`~xnn.common.train.trainer.Trainer.evaluate`:

.. code-block:: python

   import os, torch

   state = torch.load(os.path.join(cfg.output_dir, "best.pt"), weights_only=False)
   trainer.model.load_state_dict(state["model"])
   test_metrics = trainer.evaluate()             # uses trainer.test_loader

The loss
========
:func:`~xnn.common.train.losses.weighted_loss` combines the per-property
mean-squared errors:

.. math::

   \mathcal{L} = w_E \, \mathcal{L}_\text{energy}
               + w_F \, \mathcal{L}_\text{forces}
               + w_\sigma \, \mathcal{L}_\text{stress}

with the weights from ``OptimConfig`` (defaults
:math:`w_E = 1`, :math:`w_F = 10`, :math:`w_\sigma = 0`). A property enters
the loss only when the dataset provides the target and its weight is
nonzero. Force training is strongly recommended whenever forces are
available; it is dramatically more data-efficient than energies alone.

The energy term divides by the atom count and averages over *structures*;
the force term averages over every *atom* in the batch. The two therefore
disagree about what one sample is, which quietly allocates the fit when a
dataset mixes structure sizes, or mixes densely sampled scans with sparse
ones. Two knobs address that.

Per-structure weights
---------------------
Give a structure dict a ``weight`` (see
:func:`~xnn.common.data.dataset.to_graph`) and every term becomes a weighted
mean: structure :math:`b` counts in proportion to ``weight[b]``, and the
force term spreads that weight over its atoms. Uniform weights reproduce the
unweighted loss exactly, so this changes nothing until a dataset asks for it.

.. code-block:: python

   # equalize two sources that differ 10-fold in frame count
   for s in scan_frames:    s["weight"] = 1.0
   for s in cluster_frames: s["weight"] = 10.0

Weights are how you allocate the fit on purpose rather than by accident.
Diagnose first: measure each subset's share of the loss, not just its share
of the frames, since a subset with larger typical forces carries a share
that grows with the *square* of that scale.

Huber tails
-----------
``optim.huber_delta`` (with ``huber_delta_energy`` / ``huber_delta_forces`` /
``huber_delta_stress`` overriding it per term) replaces the squared error by
a function that is quadratic up to :math:`\delta` and linear beyond, capping
the pull of a few large residuals:

.. math::

   \ell(d) = \begin{cases} d^2 & |d| \le \delta \\
                            2\delta|d| - \delta^2 & |d| > \delta \end{cases}

The default ``0.0`` is plain squared error. Note this is scaled to agree with
:math:`d^2` below :math:`\delta`, twice the textbook Huber, so the loss
weights and learning rate keep their meaning when it is switched on. MACE
uses the textbook half-square form; halve ported ``delta`` values
accordingly. Per-term deltas matter because per-atom energies (eV) and
forces (eV/A) differ in scale by an order of magnitude.

Devices and batching
====================
``cfg.device = "auto" | "cpu" | "cuda" | "cuda:0"`` is resolved by
:func:`~xnn.common.train.trainer.resolve_device`. Batching is by graph
concatenation (see :ref:`data`); ``data.batch_size = 1`` disables batch
training entirely.

Multi-GPU and multi-node training
=================================
Distributed data parallelism is native PyTorch DDP and needs no code or
configuration changes, only a distributed launcher. When the trainer finds
the launcher's ``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE`` environment
variables it joins the process group (NCCL on GPUs, Gloo on CPUs), pins each
rank to ``cuda:LOCAL_RANK``, shards all loaders with ``DistributedSampler``,
wraps the model in ``DistributedDataParallel``, and all-reduces the logged
metrics so the best-checkpoint decision and the plateau scheduler stay in
lockstep across ranks. Only rank 0 prints and writes checkpoints, and the
saved state dict is that of the bare model, so checkpoints from serial and
distributed runs are interchangeable.

Single node, all (or ``N``) GPUs:

.. code-block:: bash

   torchrun --nproc-per-node 2 -m xnn train --config train.yaml

Multi-node (one such command per node, e.g. from a Slurm step):

.. code-block:: bash

   torchrun --nnodes 2 --nproc-per-node 4 \
            --rdzv-backend c10d --rdzv-endpoint "$HEAD_NODE":29500 \
            -m xnn train --config train.yaml

Hugging Face's ``accelerate launch`` works as well (it exports the same
environment variables), e.g. ``accelerate launch --multi_gpu --num_processes 2
-m xnn train --config train.yaml``, but note that it acts purely as a
process launcher here: FSDP or DeepSpeed options in an accelerate config are
not picked up, since the trainer deliberately uses DDP only. Sharded
strategies cannot train forces or stress anyway: those losses back-propagate
through gradients taken with ``create_graph=True`` (a double backward), which
DDP supports and FSDP/DeepSpeed do not.

``data.batch_size`` is per process, so the effective batch is
``batch_size × WORLD_SIZE``; scale the learning rate (or the batch size)
accordingly. Validation and test sets are sharded too, and
``DistributedSampler`` pads uneven shards by repeating a few samples, so
metrics can differ negligibly from a serial run when the split size is not
divisible by the world size.

Reproducibility
===============
``cfg.seed`` seeds the run. Note that exact bit-reproducibility across
devices and CUDA versions is not guaranteed by PyTorch itself.
