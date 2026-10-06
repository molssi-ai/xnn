.. _api:

*************
API Reference
*************

Generated from the docstrings in ``src/xnn``. Start from the subpackage
matching what you need:

- :mod:`xnn.common`: data pipeline, featurizer base, configuration,
  model registry and outputs, training, benchmarking, deployment, CLI
- :mod:`xnn.gnn`: graph-network models (SchNet, and NequIP, MACE, Allegro,
  CACE, AIMNet2, which require ``e3nn``) and featurizers
- :mod:`xnn.cnn`: volumetric convolution models over voxelized environments
  (the 3D steerable CNN and the conventional 3D CNN) and the voxel featurizer
- :mod:`xnn.dnn`: descriptor models (HDNNP, ANI, PhysNet) and
  symmetry-function / AEV featurizers
- :mod:`xnn.ffnn`: learnable classical force fields
  (ReaxFF / ReaxFF-nn) and their parameter-library I/O
- :mod:`xnn.transformer`: shared graph-transformer building blocks
  (multi-head edge attention, exponential-normal radial basis)
- :mod:`xnn.hybrid`: GNN + transformer models with a physics energy split
  (BAMBOO)

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
