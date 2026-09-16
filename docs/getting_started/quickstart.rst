.. _quickstart:

**********
Quickstart
**********

The fastest way to see xnn in action is the bundled quickstart script, which
builds toy data, trains a small SchNet for a few epochs, and predicts energies
and forces with the trained model:

.. code-block:: bash

   python examples/quickstart.py

It runs on CPU in a few seconds (and picks up a GPU automatically when one is
present). The rest of this page walks through exactly what the script does.

1. Build data as plain dictionaries
===================================
xnn consumes structures as plain dictionaries with keys ``pos`` and
``atomic_numbers`` (and optionally ``cell``, ``pbc``, ``energy``, ``forces``,
``stress``). The quickstart generates a toy set of small random H/C/O
structures with a smooth synthetic target, just to have a learnable signal:

.. code-block:: python

   import numpy as np

   def toy_structures(n=24, periodic=False):
       """Random small structures with a smooth synthetic energy/forces target."""
       rng = np.random.default_rng(0)
       out = []
       for _ in range(n):
           natoms = rng.integers(3, 6)
           pos = rng.uniform(0, 4, size=(natoms, 3))
           z = rng.choice([1, 6, 8], size=natoms)
           # toy target: pairwise gaussian well
           d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
           e = float(-np.exp(-((d - 1.5) ** 2)).sum())
           s = {"pos": pos, "atomic_numbers": z, "energy": e,
                "forces": rng.normal(0, 0.1, size=(natoms, 3))}
           if periodic:
               s["cell"] = np.eye(3) * 6.0
               s["pbc"] = [True, True, True]
           out.append(s)
       return out

:class:`~xnn.common.data.dataset.AtomicDataset` converts each dictionary into
an :class:`~xnn.common.data.atomic_data.AtomicGraph`, the single data object
every xnn model consumes. Real data loads just as easily: any ASE-readable
file format (extxyz, CIF, VASP, ...) with
``AtomicDataset.from_file("trajectory.extxyz", cutoff)``, and standard
benchmark datasets download in one line with
:func:`~xnn.common.data.hub.base.load_dataset` (e.g.
``load_dataset("rmd17", molecule="aspirin", cutoff=5.0)``); see :ref:`data`.

2. Configure and train
======================
Everything in xnn funnels through one
:class:`~xnn.common.config.schema.Config` dataclass, one dataset class, and
one trainer. The quickstart trains a small SchNet, but any registered model
name works here:

.. code-block:: python

   from xnn.common.config import Config
   from xnn.common.data import AtomicDataset
   from xnn.common.train import Trainer

   cfg = Config()
   cfg.model.name = "schnet"        # schnet|hdnnp|ani|physnet|nequip|mace|allegro|cace|bamboo
   cfg.model.cutoff = 5.0
   cfg.model.n_features = 32
   cfg.model.n_interactions = 2
   cfg.data.batch_size = 4          # >1 == batch training; set 1 to disable
   cfg.optim.epochs = 3
   cfg.optim.force_weight = 1.0
   cfg.device = "auto"              # picks cuda if present, else cpu
   cfg.output_dir = "runs/quickstart"

   dataset = AtomicDataset(toy_structures(periodic=False), cfg.data.cutoff)
   trainer = Trainer(cfg, dataset)
   trainer.fit()

Training prints a per-epoch summary and writes ``best.pt`` and ``last.pt``
checkpoints to ``cfg.output_dir`` (default ``runs/exp``).

3. Predict energies and forces
==============================
The trainer wraps the model in
:class:`~xnn.common.models.outputs.ForceStressOutput`, which adds
conservative forces (and optionally stress) by automatic differentiation of
the predicted energy. ``trainer.module`` is that trained, wrapped model,
ready for inference:

.. code-block:: python

   model = trainer.module.to("cpu").eval()
   g = dataset[0]
   out = model(g)                   # {"energy", "node_energy", "forces", ...}
   print("energy:", round(float(out["energy"].detach()), 4),
         "| forces shape:", tuple(out["forces"].shape))

.. note::

   Forces are computed by autograd, so do **not** wrap inference in
   ``torch.no_grad()``. Also predict with the *trained* module (as above): a
   freshly built SchNet predicts exactly its energy shift, because its
   readout head starts zero-initialized (the DTNN convention).

Run end to end, the script prints something like::

   registered models: ['allegro', 'ani', 'bamboo', 'cace', 'hdnnp', 'mace', 'nequip', 'opls', 'physnet', 'reaxff', 'schnet']
   training on cuda ...
   epoch    0 | train loss 2.0641e+00 | val loss 9.3133e-01
   epoch    1 | train loss 1.8768e+00 | val loss 8.7541e-01
   epoch    2 | train loss 1.7326e+00 | val loss 7.9425e-01
   energy: -0.5449 | forces shape: (5, 3)

The pieces also work on their own
=================================
Each layer of xnn is independently importable: data, featurizers, and
models compose but do not require each other:

.. code-block:: python

   # Data on its own
   from xnn.common.data import AtomicDataset, build_neighbor_list
   ds = AtomicDataset(structures, cutoff=5.0)
   graph = ds[0]

   # Featurizers on their own (AtomicGraph -> model inputs)
   from xnn.dnn.featurizers import AEV, RadialSymmetryFunctions
   from xnn.gnn.featurizers import SphericalHarmonicEdgeEmbedding
   descriptor = AEV(species=[1, 6, 8])(graph)              # (N, D) invariant AEV
   edges = SphericalHarmonicEdgeEmbedding(l_max=2)(graph)  # equivariant edges

   # Models on their own
   from xnn.common.models import build_model, ForceStressOutput, available_models
   model = ForceStressOutput(build_model(cfg.model))

From the command line
=====================
The same workflow is available through the ``xnn`` command:

.. code-block:: bash

   xnn train --config configs/train.yaml --set optim.epochs=50
   xnn export --config configs/train.yaml --ckpt runs/exp/best.pt --to lammps

Next steps
==========
- :ref:`first-training`: a complete, annotated training walk-through
- :ref:`how-tos`: training from config files, deploying to ASE and LAMMPS, etc.
- :ref:`user-guide`: full reference for data, models, configs, and training
