.. _first-training:

***********************
Your First Training Run
***********************

This tutorial trains a MACE potential on a small dataset from start to
finish, explaining each step. It requires the ``gnn`` extra
(``pip install -e ".[gnn]"``).

1. Prepare the data
===================
xnns consumes structures as plain Python dictionaries. Required keys are
``pos`` (an ``(N, 3)`` array of positions) and ``atomic_numbers`` (an
``(N,)`` array); training targets and periodicity are optional:

.. code-block:: python

   import numpy as np

   structures = []
   for frame in my_trajectory:            # however you obtain your data
       structures.append(
           {
               "pos": frame.positions,             # (N, 3), required
               "atomic_numbers": frame.numbers,    # (N,),   required
               "cell": frame.cell,                 # (3, 3), optional (periodic)
               "pbc": frame.pbc,                   # (3,),   optional
               "energy": frame.energy,             # scalar, training target
               "forces": frame.forces,             # (N, 3), training target
               # "stress": frame.stress,           # (3, 3), optional target
           }
       )

If your data lives in a file format ASE can read (extxyz, CIF, VASP, ...),
skip the dictionaries entirely and load the file directly (requires the
``ase`` extra):

.. code-block:: python

   from xnns.common.data import AtomicDataset

   dataset = AtomicDataset.from_file("my_trajectory.extxyz", cutoff=4.0)

Energy / forces / stress targets stored in the file are picked up
automatically; see :ref:`data`.

2. Build the dataset
====================
:class:`~xnns.common.data.dataset.AtomicDataset` turns the structure
dictionaries into :class:`~xnns.common.data.atomic_data.AtomicGraph` objects,
building the (PBC-aware) neighbor list at the given cutoff:

.. code-block:: python

   from xnns.common.data import AtomicDataset

   cutoff = 4.0
   train_set = AtomicDataset(structures[:800], cutoff)
   val_set = AtomicDataset(structures[800:], cutoff)

Molecular and periodic systems go through the same class — periodicity is
handled entirely inside the neighbor list and edge vectors, and models never
see the difference.

3. Configure the model
======================
The :class:`~xnns.common.config.schema.Config` dataclass holds everything:
model, data, and optimizer settings. Model-specific options go into
``cfg.model.extra``:

.. code-block:: python

   from xnns.common.config import Config

   cfg = Config()
   cfg.model.name = "mace"
   cfg.model.cutoff = 4.0
   cfg.model.extra = {
       "species": [18],              # atomic numbers in the dataset (Argon here)
       "max_ell": 3,                 # spherical-harmonic order of the edges
       "num_channels": 32,           # feature channels
       "num_interactions": 2,        # message-passing layers (T)
       "correlation": 3,             # order of the symmetric contraction
       "avg_num_neighbors": 20.0,    # normalization; compute from your data
   }

   cfg.data.batch_size = 8
   cfg.optim.lr = 1e-3
   cfg.optim.epochs = 100
   cfg.optim.energy_weight = 1.0
   cfg.optim.force_weight = 10.0    # > 0 enables force training
   cfg.device = "auto"
   cfg.output_dir = "runs/my_first_run"

Alternatively, load the same settings from a YAML file — see
:ref:`configuration` — or start from the templates in ``configs/``.

4. Train
========
:class:`~xnns.common.train.trainer.Trainer` builds the model from the config,
wraps it in :class:`~xnns.common.models.outputs.ForceStressOutput` (force and
stress heads are switched on by nonzero loss weights), sets up the Adam
optimizer, scheduler, and data loaders, and runs the loop:

.. code-block:: python

   from xnns.common.train import Trainer

   trainer = Trainer(cfg, train_set, val_set)
   trainer.fit()

Progress is printed per epoch; ``best.pt`` (lowest validation loss) and
``last.pt`` are written to ``cfg.output_dir``. A checkpoint is a dictionary
``{"model": state_dict, "cfg": Config}``. A held-out test set is optional —
pass it as a third dataset (``Trainer(cfg, train_set, val_set, test_set)``)
or set ``cfg.data.test_fraction`` to carve one out of the training data; it
is evaluated once after the last epoch (see :ref:`training`).

5. Predict
==========
Load the checkpoint and evaluate on new structures:

.. code-block:: python

   import torch
   from xnns.common.data import AtomicDataset
   from xnns.common.models import build_model, ForceStressOutput

   ckpt = torch.load("runs/my_first_run/best.pt", weights_only=False)
   model = build_model(ckpt["cfg"].model)
   model.load_state_dict(ckpt["model"])
   model = ForceStressOutput(model).eval()

   graph = AtomicDataset(test_structures, cutoff)[0]
   out = model(graph)
   print(out["energy"], out["forces"].shape)

6. Where to go from here
========================
- Drive the same run from a config file and the CLI: :ref:`howto-train-a-model`
- Attach the model to ASE for molecular dynamics: :ref:`howto-ase`
- Export it for production LAMMPS runs: :ref:`howto-lammps`
- The example notebooks in ``examples/gnn/`` repeat this workflow on real
  Argon MD data and validate every step against the reference MACE, NequIP,
  and Allegro implementations.
