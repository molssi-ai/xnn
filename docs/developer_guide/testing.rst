.. _developer-guide-testing:

*******
Testing
*******

.. code-block:: bash

   pip install -e ".[dev,gnn]"
   pytest tests/

The suite is a flat ``tests/`` directory, one file per model or subsystem.
Tests that need a reference package (``mace-torch``, ``nequip``,
``torchani``, ``dftd4``, OpenMM, ...) skip when it is not installed, so
the suite runs with the core extras alone.

What the tests check
====================
- **Every model**: registration, rotation and translation invariance of the
  energy and equivariance of the forces (to about 1e-7), periodic stress,
  forces against finite differences, size extensivity, batching, config
  key translation, and TorchScript parity with the eager model where the
  model exports.
- **Faithful implementations** (MACE, NequIP, Allegro, CACE, AIMNet2, DimeNet, SpookyNet,
  PaiNN, PhysNet, ANI, HDNNP, BAMBOO, LES, D3, D4): parity with the upstream code under
  transplanted weights, to about 1e-15 (HDNNP against the RuNNer executable
  when ``XNN_RUNNER`` points to it) (``test_mace`` also covers the
  foundation-checkpoint conversion in process, without downloads).
- **Clean-room implementations** (SchNet, the SE(3) steerable CNN,
  ReaxFF, OPLS, DREIDING): parity
  with an independent implementation of the papers' equations, hand-recomputed
  terms, the published tables and, for OPLS, OpenMM.
- **Pipeline**: neighbor lists against ASE and the vesin backend, mixed
  batches and label masks, float64 geometry, the data and model hubs, ASE
  file I/O, the losses, the benchmark runner, the MDI engine, the fast
  paths against the reference, and ``torchrun`` DDP training.

Conventions for new code
========================
- A new model gets, at minimum: a registration test, an invariance or
  equivariance test, a periodic-stress test and, if it exports, a
  script-versus-eager parity test.
- A faithful re-implementation also pins parity with the upstream code under
  transplanted weights, guarded by an import check. A clean-room build pins
  parity with an independent implementation of the paper's equations instead.
