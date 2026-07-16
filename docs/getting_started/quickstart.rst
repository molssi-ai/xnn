.. _quickstart:

**********
Quickstart
**********

The fastest way to see xnns in action is the bundled quickstart script, which
builds toy data, trains a small model for a few epochs, and predicts energies
and forces:

.. code-block:: bash

   python examples/quickstart.py

Five lines to a trained potential
=================================
Everything in xnns funnels through one :class:`~xnns.common.config.schema.Config`
dataclass, one dataset class, and one trainer:

.. code-block:: python

   from xnns.common.config import Config
   from xnns.common.data import AtomicDataset
   from xnns.common.train import Trainer

   cfg = Config()
   cfg.model.name = "nequip"            # schnet|hdnnp|ani|physnet|nequip|mace|allegro|cace|bamboo
   cfg.model.extra = {"species": [1, 6, 8], "l_max": 2}
   cfg.data.batch_size = 16             # 1 disables batch training
   cfg.device = "auto"                  # auto | cpu | cuda | cuda:0

   Trainer(cfg, AtomicDataset(structures, cfg.model.cutoff)).fit()

Here ``structures`` is a list of plain dictionaries, one per structure, with
keys ``pos`` and ``atomic_numbers`` (and optionally ``cell``, ``pbc``,
``energy``, ``forces``, ``stress``). :class:`~xnns.common.data.dataset.AtomicDataset`
converts each into an :class:`~xnns.common.data.atomic_data.AtomicGraph`, the
single data object every xnns model consumes. Data in any ASE-readable file
format (extxyz, CIF, VASP, ...) loads directly with
``AtomicDataset.from_file("trajectory.extxyz", cutoff)``, and standard benchmark
datasets download in one line with
:func:`~xnns.common.data.hub.base.load_dataset` (e.g.
``load_dataset("rmd17", molecule="aspirin", cutoff=5.0)``); see :ref:`data`.

Training writes ``best.pt`` and ``last.pt`` checkpoints to
``cfg.output_dir`` (default ``runs/exp``).

Predicting energies and forces
==============================
Any model can be wrapped in
:class:`~xnns.common.models.outputs.ForceStressOutput`, which adds forces (and
optionally stress) by automatic differentiation of the predicted energy:

.. code-block:: python

   from xnns.common.models import build_model, ForceStressOutput

   model = ForceStressOutput(build_model(cfg.model)).eval()
   out = model(graph)          # {"energy", "node_energy", "forces"}

.. note::

   Forces are computed by autograd, so do **not** wrap inference in
   ``torch.no_grad()``.

The pieces also work on their own
=================================
Each layer of xnns is independently importable: data, featurizers, and
models compose but do not require each other:

.. code-block:: python

   # Data on its own
   from xnns.common.data import AtomicDataset, build_neighbor_list
   ds = AtomicDataset(structures, cutoff=5.0)
   graph = ds[0]

   # Featurizers on their own (AtomicGraph -> model inputs)
   from xnns.dnn.featurizers import AEV, RadialSymmetryFunctions
   from xnns.gnn.featurizers import SphericalHarmonicEdgeEmbedding
   descriptor = AEV(species=[1, 6, 8])(graph)              # (N, D) invariant AEV
   edges = SphericalHarmonicEdgeEmbedding(l_max=2)(graph)  # equivariant edges

   # Models on their own
   from xnns.common.models import build_model, ForceStressOutput, available_models
   model = ForceStressOutput(build_model(cfg.model))

From the command line
=====================
The same workflow is available through the ``xnns`` command:

.. code-block:: bash

   xnns train --config configs/train.yaml --set optim.epochs=50
   xnns export --config configs/train.yaml --ckpt runs/exp/best.pt --to lammps

Next steps
==========
- :ref:`first-training`: a complete, annotated training walk-through
- :ref:`how-tos`: training from config files, deploying to ASE and LAMMPS
- :ref:`user-guide`: full reference for data, models, configs, and training
