.. _first-training:

***********************
Your First Training Run
***********************

This page trains a MACE potential from start to finish. It needs the ``gnn``
extra.

1. Prepare the data
===================
Structures are dictionaries with ``pos`` and ``atomic_numbers``; the targets
and the cell are optional:

.. code-block:: python

   structures = [
       {
           "pos": frame.positions,            # (N, 3)
           "atomic_numbers": frame.numbers,   # (N,)
           "cell": frame.cell,                # (3, 3), periodic systems
           "pbc": frame.pbc,                  # (3,)
           "energy": frame.energy,            # scalar target
           "forces": frame.forces,            # (N, 3) target
       }
       for frame in my_trajectory
   ]

If the data is in a file ASE can read, skip the dictionaries:

.. code-block:: python

   from xnn.common.data import AtomicDataset

   train_set = AtomicDataset.from_file("train.extxyz", cutoff=4.0)
   val_set = AtomicDataset.from_file("val.extxyz", cutoff=4.0)

Energies, forces and stresses stored in the file are picked up automatically.

2. Configure the model
======================
Model-specific options go into ``cfg.model.extra``:

.. code-block:: python

   from xnn.common.config import Config

   cfg = Config()
   cfg.model.name = "mace"
   cfg.model.cutoff = 4.0
   cfg.model.extra = {
       "species": [18],              # atomic numbers in the dataset
       "max_ell": 3,
       "num_channels": 32,
       "num_interactions": 2,
       "correlation": 3,
       "avg_num_neighbors": 20.0,    # compute from your data
   }
   cfg.data.batch_size = 8
   cfg.optim.lr = 1e-3
   cfg.optim.epochs = 100
   cfg.optim.force_weight = 10.0     # > 0 trains on forces
   cfg.output_dir = "runs/my_first_run"

The same settings can come from a YAML file (:ref:`configuration`) or the
templates in ``configs/``.

3. Train
========

.. code-block:: python

   from xnn.common.train import Trainer

   trainer = Trainer(cfg, train_set, val_set)
   trainer.fit()

The trainer builds the model, wraps it in
:class:`~xnn.common.models.outputs.ForceStressOutput`, and writes ``best.pt``
(lowest validation loss) and ``last.pt`` to the output directory. Pass a
third dataset, or set ``cfg.data.test_fraction``, to evaluate a test set
once after the last epoch.

4. Predict
==========
A checkpoint loads through the model hub:

.. code-block:: python

   from xnn.common.models import from_pretrained

   model = from_pretrained("runs/my_first_run/best.pt")
   out = model(AtomicDataset(test_structures, cutoff=4.0)[0])
   print(out["energy"], out["forces"].shape)

Next
====
- Drive the run from a config file: :ref:`howto-train-a-model`
- Run molecular dynamics with ASE: :ref:`howto-ase`
- Export for LAMMPS: :ref:`howto-lammps`
