.. _first-training:

***********************
Your First Training Run
***********************

This tutorial focuses on training a MACE potential from start to finish. It
needs the ``gnn`` extra optional dependency in place.

1. Prepare the data
===================
Structures are dictionaries with ``pos`` and ``atomic_numbers``. However, the targets
and the ``cell`` entries are optional:

.. code-block:: python

   structures = [
       {
           "pos": frame.positions,            # (N, 3)
           "atomic_numbers": frame.numbers,   # (N,)
           "cell": frame.cell,                # (3, 3), periodic systems
           "pbc": frame.pbc,                  # (3,)
           "energy": frame.energy,            # Scalar target
           "forces": frame.forces,            # (N, 3) target
       }
       for frame in my_trajectory
   ]

If the atomic data frames are in a file format that ASE can read, you can skip
creating the dictionaries and read the data directly from the file:

.. code-block:: python

   from xnn.common.data import AtomicDataset

   train_set = AtomicDataset.from_file("train.extxyz", cutoff=4.0)
   val_set = AtomicDataset.from_file("val.extxyz", cutoff=4.0)

Energies, forces and stresses that are stored in the file will be picked up
automatically.

2. Configure the model
======================
Model-specific options should go into ``cfg.model.extra`` entry in the configuration object:

.. code-block:: python

   from xnn.common.config import Config

   cfg = Config()
   cfg.model.name = "mace"
   cfg.model.cutoff = 4.0
   cfg.model.extra = {
       "species": [18],              # Atomic numbers in the dataset
       "max_ell": 3,
       "num_channels": 32,
       "num_interactions": 2,
       "correlation": 3,
       "avg_num_neighbors": 20.0,    # Compute from your data
   }
   cfg.data.batch_size = 8
   cfg.optim.lr = 1e-3
   cfg.optim.epochs = 100
   cfg.optim.force_weight = 10.0     # > 0 trains on forces as well
   cfg.output_dir = "runs/my_first_run"

The same settings can come from a YAML file (:ref:`configuration`) or the
templates in the repository's ``configs/`` directory.

3. Train
========
You can train a model by passing the configuration and datasets to the
:class:`~xnn.common.train.trainer.Trainer` class:

.. code-block:: python

   from xnn.common.train import Trainer

   trainer = Trainer(cfg, train_set, val_set)
   trainer.fit()

The ``trainer`` builds the model, wraps it in
:class:`~xnn.common.models.outputs.ForceStressOutput`, and writes the
``best.pt`` (lowest validation loss) and ``last.pt`` checkpoint artifacts to the
output directory. You can pass a third dataset argument, or set
``cfg.data.test_fraction`` in the configuration object, to evaluate a test set
after training.

1. Predict
==========
The checkpoint artifact of a pre-trained model can be fetched through the model hub or loaded from 
disk using the :func:`~xnn.common.models.from_pretrained` function:

.. code-block:: python

   from xnn.common.models import from_pretrained

   # Load the best checkpoint from the training run
   model = from_pretrained("runs/my_first_run/best.pt")

   # Evaluate on a single structure
   out = model(AtomicDataset(test_structures, cutoff=4.0)[0])

   # Print the predicted energy and shape of the predicted atomic force tensor
   print(out["energy"], out["forces"].shape)

Next
====
- Drive the run from a config file: :ref:`howto-train-a-model`
- Run molecular dynamics with ASE: :ref:`howto-ase`
- Export for LAMMPS: :ref:`howto-lammps`
