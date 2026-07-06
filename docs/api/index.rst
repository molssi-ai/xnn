.. _api:

*************
API Reference
*************

Complete reference documentation, generated from the docstrings in
``src/xnns``. Start from the subpackage matching what you need:

- :mod:`xnns.common` — data pipeline, featurizer base, configuration,
  model registry and outputs, training, deployment, CLI
- :mod:`xnns.gnn` — E(3)-equivariant models (NequIP, MACE, Allegro) and
  featurizers (requires ``e3nn``)
- :mod:`xnns.cnn` — continuous-filter convolution models (SchNet)
- :mod:`xnns.dnn` — descriptor models (HDNNP, ANI) and symmetry-function /
  AEV featurizers

.. autosummary::
   :toctree: generated
   :recursive:

   xnns.common
   xnns.gnn
   xnns.cnn
   xnns.dnn
