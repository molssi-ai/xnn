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
One :class:`~xnn.common.config.schema.Config` holds the model, data and
optimizer settings; one :class:`~xnn.common.train.trainer.Trainer` runs the
loop:

.. code-block:: python

   from xnn.common.config import Config
   from xnn.common.train import Trainer

   cfg = Config()
   cfg.model.name = "schnet"          # any name from available_models()
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
   out = model(dataset[0])
   print(out["energy"], out["forces"].shape)

Forces come from autograd, so do not wrap inference in ``torch.no_grad()``.
Predict with the trained module: a fresh SchNet predicts only its energy
shift, because its readout starts zero-initialized.

The pieces work on their own
============================
Data, featurizers and models compose but do not require each other:

.. code-block:: python

   from xnn.dnn.featurizers import AEV
   from xnn.gnn.featurizers import SphericalHarmonicEdgeEmbedding
   from xnn.common.models import build_model, ForceStressOutput

   graph = dataset[0]
   descriptor = AEV(species=[1, 6, 8])(graph)               # (N, D) invariant
   edges = SphericalHarmonicEdgeEmbedding(l_max=2)(graph)   # equivariant edges
   model = ForceStressOutput(build_model(cfg.model))

From the command line
=====================

.. code-block:: bash

   xnn train --config configs/train.yaml --set optim.epochs=50
   xnn export --ckpt runs/exp/best.pt --out deployed.pt

Next steps
==========
- :ref:`first-training`: a complete, annotated training run
- :ref:`howto-pretrained-models`: load a foundation model in one line
- :ref:`how-tos`: config files, fine-tuning, ASE and LAMMPS deployment
