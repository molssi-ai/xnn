.. _api:

*************
API Reference
*************

Generated from the docstrings in ``src/xnn``. Start from the subpackage
matching what you need:

- :mod:`xnn.common`: data pipeline and hub, featurizer base, configuration,
  model registry, hub and add-ons (LES, D3, D4), fine-tuning, training,
  benchmarking, deployment, CLI
- :mod:`xnn.gnn`: SchNet, DimeNet, DimeNet++, PaiNN, NequIP, MACE, Allegro, CACE,
  AIMNet2, their featurizers and fast paths (NequIP, MACE and Allegro need
  ``e3nn``)
- :mod:`xnn.cnn`: the SE(3) steerable CNN and the conventional 3D CNN over
  voxelized environments, the spherical CNN over spherical signals, and
  their featurizers
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
