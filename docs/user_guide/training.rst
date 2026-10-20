.. _training:

********
Training
********

The Trainer
===========
The :class:`~xnn.common.train.trainer.Trainer` class is responsible for running
the whole training loop:

.. code-block:: python

   from xnn.common.train import Trainer

   trainer = Trainer(cfg, train_set, val_set, test_set)   # Validation and test sets are optional
   metrics = trainer.fit()                                # {"train": ..., "val": ..., "test": ...}

Cascading a ``fit()`` function call on a
:class:`~xnn.common.train.trainer.Trainer` instance object, does the following:

- Builds the model from ``cfg.model`` and adds force and stress heads for the
  nonzero loss weights,
- Holds out ``data.val_fraction`` and ``data.test_fraction`` of the training
  set when no validation or test set is given,
- Validates every epoch and writes ``best.pt`` and ``last.pt`` to
  ``cfg.output_dir``, and
- Evaluates the test set once after the last epoch.

A checkpoint loads back with ``from_pretrained(path)``, and
``trainer.save_pretrained("my-model")`` writes a portable model directory (see
the :ref:`howto-pretrained-models` guide for mode details).

The loss
========
The loss is a weighted sum of the per-property errors:

.. math::

   \mathcal{L} = w_E \, \mathcal{L}_\text{energy}
               + w_F \, \mathcal{L}_\text{forces}
               + w_\sigma \, \mathcal{L}_\text{stress}

The defaults are :math:`w_E = 1`, :math:`w_F = 10` and :math:`w_\sigma = 0`, can
be set by ``energy_weight``, ``force_weight`` and ``stress_weight`` in ``optim``
section of the config file. A property contributes to the loss only when the
data provides it and its weight is nonzero. Train on forces whenever you have
them, since they are often far more informative to the training than energies
alone.

You can shape the loss further:

- **Per-structure weights.** A ``weight`` in a structure's dictionary scales
  its share of every term, for instance to balance a sparse set against a dense
  scan.
- **Per-label weights.** ``energy_weight``, ``force_weight`` and
  ``stress_weight`` in a structure scale one of its terms, and
  ``atom_force_weight`` the force error of each atom. A weight of 0 drops the
  label (see :ref:`howto-weight-force-labels`).
- **Huber loss.** ``optim.huber_delta`` keeps the error quadratic up to
  :math:`\delta` and linear beyond, which limits the pull of a few large
  outliers. Below :math:`\delta` it equals the squared error, so the loss
  weights keep their meaning when you switch it on (MACE's Huber loss is half
  as large).
- **Several heads.** A multi-head model sums each head's terms, and
  ``optim.head_weights`` gives a head its own weights (see
  :ref:`howto-finetune`).

Metrics, checkpoints and stopping
=================================
Each epoch prints the loss terms for the training and validation sets, then
the MAE and RMSE of every labelled property (``energy_mae`` per atom,
``forces_rmse``, ...). Every value is averaged over the whole set, so batch
sizes do not skew it. The ``trainer`` section decides what drives the run:

.. code-block:: yaml

   trainer:
     monitor: val/forces_mae         # Picks best.pt and drives early stopping
     mode: min
     val_every_n_epochs: 1
     save_every_n_epochs: 0          # > 0 also keeps epoch_<n>.pt
     early_stopping_patience: 20     # Validations without improvement; 0 turns it off
     early_stopping_threshold: 0.02  # Stop once forces_mae drops below 0.02

Any logged key can be monitored. In Python the same features are callbacks,
and you can add your own:

.. code-block:: python

   from xnn.common.train import Callback, EarlyStopping, ModelCheckpoint

   class Snapshot(Callback):
       def on_epoch_end(self, trainer, epoch, metrics):
           if epoch % 50 == 0:
               trainer.save(f"runs/exp/epoch_{epoch}.pt")

   trainer = Trainer(cfg, train_set, val_set, callbacks=[
       ModelCheckpoint(monitor="val/energy_rmse", filename="best_energy.pt"),
       EarlyStopping(monitor="val/forces_mae", patience=30),
       Snapshot()])

Logging
=======
Metrics go to the console and to ``metrics.csv`` in ``output_dir`` by
default. TensorBoard needs the ``tensorboard`` extra:

.. code-block:: yaml

   trainer:
     loggers: [console, csv, tensorboard]
     log_every_n_steps: 0            # > 0 also logs training values every n steps

In Python, pass ``logger=[...]`` to the trainer, or subclass
:class:`~xnn.common.train.loggers.Logger` to send the metrics anywhere.

Optimizer and fine-tuning
=========================
The ``optim`` section holds the optimizer settings:

.. code-block:: yaml

   optim:
     optimizer: adamw                # adam, amsgrad or adamw
     lr: 1.0e-3
     scheduler: plateau              # none, cosine or plateau
     clip_grad: 1.0                  # Max gradient norm; 0 turns it off
     ema_decay: 0.999                # Validate and save an average of the weights
     freeze: ["model.node_embedding*"]   # Parameter-name patterns to keep fixed
     prior_weight: 0.0               # > 0 pulls the parameters toward their starting values
     prior_weights: {"*charge": 10.0}    # The same per parameter-name pattern

``prior_weight`` adds ``prior_weight * sum((p - p0)^2)`` over the trainable
parameters to the training loss (logged as ``prior``), in the parameters' own
units; it is the regularization of a force-field refit whose data leave some
parameter directions free. ``model.pretrained``, ``model.lora`` and
``model.heads`` turn the same trainer into a fine-tuning run (see
:ref:`howto-finetune`).

Several GPUs or nodes
=====================
Distributed training uses PyTorch DDP and needs no code or config changes, only
a launcher such as ``torchrun``:

.. code-block:: bash

   torchrun --nproc-per-node 2 -m xnn train --config train.yaml

   torchrun --nnodes 2 --nproc-per-node 4 \
            --rdzv-backend c10d --rdzv-endpoint "$HEAD_NODE":29500 \
            -m xnn train --config train.yaml

Note that:

- the ``data.batch_size`` is a per GPU micro-batch size. The effective batch is
  ``batch_size`` times the total number of GPUs (over a single node or multiple
  nodes).
- Only the first rank prints and writes checkpoints, and checkpoints are
  interchangeable with single-GPU runs.
- The FSDP and DeepSpeed ZeRo methods are not currently supported, because force
  and stress losses need a double backward pass that only DDP provides.

Custom batching
===============
The training loader reshuffles the data at every epoch. To batch differently,
for instance molecules and periodic cells in separate batches, pass a
``train_batch_sampler`` that yields lists of dataset indices:

.. code-block:: python

   trainer = Trainer(cfg, train_set, val_set, train_batch_sampler=my_sampler)

If the sampler has a ``set_epoch(epoch)`` method, the trainer calls it before
every epoch.

Reproducibility
===============
The ``cfg.seed`` sets the seed for the run. PyTorch itself does not guarantee
bit-identical results across devices or CUDA versions.
