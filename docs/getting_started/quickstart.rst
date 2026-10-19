.. _quickstart:

**********
Quickstart
**********

The bundled ``quickstart.py`` script builds a toy data, trains a small SchNet
model on it for three epochs and predicts the energies and forces using it. This
code runs on CPU in seconds and uses a GPU when one is present:

.. code-block:: bash

   python notebooks/examples/quickstart.py

The rest of this page details what the ``quickstart`` script does in more
detail. 

1. Data
=======
A structure is a plain dictionary with ``pos`` and ``atomic_numbers``, plus
optional ``cell``, ``pbc`` and target properties such as ``energy``, ``forces`` and
``stress``. The :class:`~xnn.common.data.dataset.AtomicDataset` class turns a
list of these structures into neighbor graphs at a cutoff radius:

.. code-block:: python

   from xnn.common.data import AtomicDataset

   dataset = AtomicDataset(structures, cutoff=5.0)

The structures can be provided to xnn in a similar way from any ASE-readable file
(``AtomicDataset.from_file("trajectory.extxyz", cutoff=5.0)``) or from the xnn's
own data hub (``load_dataset("rmd17", molecule="aspirin", cutoff=5.0)``). See
the :ref:`data` user guide for more details.

2. Configure and train
======================
Once the data is ready, we can focus on the model preparation. xnn uses a configuration object,
:class:`~xnn.common.config.schema.Config`, to hold all the settings for the model, data, 
optimizer etc. The :class:`~xnn.common.train.trainer.Trainer` class can use the
dataset and configuration settings to run the training loop:

.. code-block:: python

   from xnn.common.config import Config
   from xnn.common.train import Trainer

   cfg = Config()
   cfg.model.name = "schnet"          # Any name from available_models()
   cfg.model.cutoff = 5.0
   cfg.model.n_features = 32
   cfg.model.n_interactions = 2
   cfg.data.batch_size = 4
   cfg.optim.epochs = 3
   cfg.optim.force_weight = 1.0
   cfg.output_dir = "runs/quickstart"

   trainer = Trainer(cfg, dataset)
   trainer.fit()

Training prints one line per epoch and writes ``best.pt`` and ``last.pt``
checkpoints to the output directory.

3. Predict
==========
The ``trainer.module`` wraps the trained model in the
:class:`~xnn.common.models.outputs.ForceStressOutput` class which computes the atomic 
forces by autograd:

.. code-block:: python

   model = trainer.module.to("cpu").eval()
   out = model(dataset[0])
   print(out["energy"], out["forces"].shape)

Because forces are calculated by autograd, you should not wrap the module within
a ``torch.no_grad()`` context.

The pieces work on their own
============================
Data, featurizers and models compose but do not require each other:

.. code-block:: python

   from xnn.dnn.featurizers import AEV
   from xnn.gnn.featurizers import SphericalHarmonicEdgeEmbedding
   from xnn.common.models import build_model, ForceStressOutput

   # Get a graph from the dataset
   graph = dataset[0]

   # Featurize the graph
   descriptor = AEV(species=[1, 6, 8])(graph)               # (N, D) invariant
   edges = SphericalHarmonicEdgeEmbedding(l_max=2)(graph)   # Equivariant edges

   # Build a model and wrap it in the ForceStressOutput class
   model = ForceStressOutput(build_model(cfg.model))

From the command line
=====================

xnn can be run from the command line using the ``xnn`` entry point.

.. code-block:: bash

   # Train a model for 50 epochs
   xnn train --config configs/train.yaml --set optim.epochs=50

   # Export the best checkpoint to a deployable model
   xnn export --ckpt runs/exp/best.pt --out deployed.pt

Next steps
==========
- :ref:`first-training`: a complete, annotated training run
- :ref:`howto-pretrained-models`: load a foundation model in one line
- :ref:`how-tos`: config files, fine-tuning, ASE and LAMMPS deployment
