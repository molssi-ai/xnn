.. _api:

*************
API Reference
*************

Generated from the docstrings in ``src/xnn``. Start from the subpackage
matching what you need:

- :mod:`xnn.common`: data pipeline and hub, featurizer base, configuration,
  model registry, hub and add-ons, fine-tuning, training, benchmarking,
  deployment, CLI
- :mod:`xnn.gnn`: NequIP, MACE, Allegro, CACE, AIMNet2 and their
  featurizers and fast paths (NequIP, MACE and Allegro need ``e3nn``)
- :mod:`xnn.cnn`: SchNet
- :mod:`xnn.dnn`: HDNNP, ANI, PhysNet and the symmetry-function / AEV
  featurizers
- :mod:`xnn.ffnn`: ReaxFF, OPLS, DREIDING and the ``.frc`` force-field reader
- :mod:`xnn.transformer`: edge attention and the exponential-normal basis
- :mod:`xnn.hybrid`: BAMBOO

.. autosummary::
   :toctree: generated
   :recursive:

   xnn.common
   xnn.gnn
   xnn.cnn
   xnn.dnn
   xnn.ffnn
   xnn.transformer
   xnn.hybrid
