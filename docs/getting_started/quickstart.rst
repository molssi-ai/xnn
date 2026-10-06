.. _quickstart:

**********
Quickstart
**********

The bundled script builds toy data, trains a small SchNet for three epochs
and predicts energies and forces with it. It runs on CPU in seconds and uses
a GPU when one is present:

.. code-block:: bash

   python examples/quickstart.py

The rest of this page is what the script does.

1. Data
=======
A structure is a plain dictionary with ``pos`` and ``atomic_numbers``, plus
optional ``cell``, ``pbc`` and the targets ``energy``, ``forces`` and
``stress``. :class:`~xnn.common.data.dataset.AtomicDataset` turns a list of
them into neighbor graphs at a cutoff:

.. code-block:: python

   from xnn.common.data import AtomicDataset

   dataset = AtomicDataset(structures, cutoff=5.0)

Real data loads the same way from any ASE-readable file
(``AtomicDataset.from_file("trajectory.extxyz", cutoff=5.0)``) or from the
data hub (``load_dataset("rmd17", molecule="aspirin", cutoff=5.0)``); see
:ref:`data`.

2. Configure and train
======================
One :class:`~xnn.common.config.schema.Config` object holds the model, data
and optimizer settings, and one :class:`~xnn.common.train.trainer.Trainer`
runs the loop:

.. code-block:: python

   from xnn.common.config import Config
   from xnn.common.train import Trainer

   cfg = Config()
   cfg.model.name = "schnet"        # schnet|hdnnp|ani|physnet|nequip|mace|allegro|cace|bamboo|se3cnn|cnn3d
   cfg.model.cutoff = 5.0
   cfg.model.n_features = 32
   cfg.model.n_interactions = 2
   cfg.data.batch_size = 4
   cfg.optim.epochs = 3
   cfg.optim.force_weight = 1.0
   cfg.output_dir = "runs/quickstart"

   trainer = Trainer(cfg, dataset)
   trainer.fit()

Training prints one line per epoch and writes ``best.pt`` and ``last.pt`` to
the output directory.

3. Predict
==========
``trainer.module`` is the trained model wrapped in
:class:`~xnn.common.models.outputs.ForceStressOutput`, which adds forces by
autograd:

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

   registered models: ['allegro', 'ani', 'bamboo', 'cace', 'cnn3d', 'hdnnp', 'mace', 'nequip', 'opls', 'physnet', 'reaxff', 'schnet', 'se3cnn']
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
The same run from a YAML config:

.. code-block:: bash

   xnn train --config configs/train.yaml --set optim.epochs=50
   xnn export --ckpt runs/exp/best.pt --to lammps

Next steps
==========
- :ref:`first-training`: a complete, annotated training run
- :ref:`howto-pretrained-models`: load a foundation model in one line
- :ref:`how-tos`: config files, fine-tuning, ASE and LAMMPS deployment
