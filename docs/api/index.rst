.. _api:

*************
API Reference
*************

Complete reference documentation, generated from the docstrings in
``src/xnn``. Start from the subpackage matching what you need:

- :mod:`xnn.common`: data pipeline, featurizer base, configuration,
  model registry and outputs, training, benchmarking, deployment, CLI
- :mod:`xnn.gnn`: E(3)-equivariant models (NequIP, MACE, Allegro, CACE) and
  featurizers (requires ``e3nn``)
- :mod:`xnn.cnn`: continuous-filter convolution models (SchNet)
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
