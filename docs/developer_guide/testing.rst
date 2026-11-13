.. _developer-guide-testing:

*******
Testing
*******

.. code-block:: bash

   pip install -e ".[dev,gnn,ase,ffnn,vesin,cpu-kernels]"
   pytest tests/                   # the whole suite
   pytest tests/gnn                # one family
   pytest tests/gnn/test_mace.py   # one file

The tests live in one folder per model family (``tests/gnn/``, ``tests/ffnn/``,
...), one file per model or subsystem. ``tests/common/`` holds the tests of the
shared code and every test that uses models of several families.
Tests that need a reference package (``mace-torch``, ``nequip``,
``torchani``, ``dftd4``, OpenMM, ...) skip when it is not installed, so
the suite runs with the core extras alone. Without a CUDA device, the Triton
kernels run under Triton's interpreter (``tests/conftest.py``). The CPU-only
PyTorch wheel has no Triton, so install ``triton`` as well (CI uses 3.3.1).

What the tests check
====================
- **Every model**: registration, rotation and translation invariance of the
  energy and equivariance of the forces (to about 1e-7), periodic stress,
  forces against finite differences, size extensivity, batching, config
  key translation, and TorchScript parity with the eager model where the
  model exports.
- **Fidelity** (every model, whatever the implementation method): numerical
  agreement with the reference to float64 round-off, and to float32 round-off
  in single precision. The reference is the upstream code under identical
  weights where one exists (MACE, NequIP, Allegro, CACE, AIMNet2, DimeNet,
  GemNet, SpookyNet, SchNet, PaiNN, TensorNet, MEGNet, ET, Equiformer, EquiformerV2, PhysNet,
  ANI, the spherical CNN, HDNNP against the RuNNer executable when
  ``XNN_RUNNER`` points to it,
  BAMBOO, ByteFF, ByteFF-Pol, GDML, sGDML, LES, D3, D4), the reference program (OpenMM for OPLS, LAMMPS for ReaxFF,
  AMBER, CHARMM and DREIDING, Tinker for AMOEBA and HIPPO, MOPAC for the NDDO
  methods), and otherwise an independent implementation of the paper's
  equations with its published tables (the SE(3) steerable CNN).
  ``test_mace`` also covers the foundation-checkpoint conversion in
  process, without downloads; ``test_amber``, ``test_charmm`` and
  ``test_amoeba`` also check float32 against float64, the improper and
  multipole-frame conventions, for CHARMM the 1,4 bookkeeping in small rings
  and for HIPPO the density kernels against their closed forms.
- **Performance** (every model with an upstream code): the reference
  implementation is at least as fast and, preferably, as memory-efficient as the
  upstream code with the same weights and inputs, in float32 and float64, for
  inference and for a training step with forces. It is timed on the upstream's
  own hardware: on CPUs when the upstream runs on CPUs. A fast path agrees with
  the reference to round-off in energies and every derivative, and is timed
  against the upstream's GPU build where one exists, otherwise against the
  reference. The model's fidelity notebook reports the comparison.
- **Pipeline**: neighbor lists against ASE and the vesin backend, mixed
  batches and label masks, float64 geometry, the data and model hubs, ASE
  file I/O, the losses, the benchmark runner, the MDI engine, the fast
  paths against the reference, and ``torchrun`` DDP training.
- **Numerical edge cases** (``test_numerical_stability``): first and second
  activation derivatives, collinear angular descriptors in float32 and
  float64, nearest images in skewed cells, and periodic invariance under
  lattice translations and rotations. Deployment tests also check LES
  charge solves and stress with rotated cells.

Continuous integration
======================
The Tests workflow runs each folder of ``tests/`` as its own job, in parallel,
on the oldest supported and the current PyTorch.

- A pull request runs only the folders its changes can affect: the family that
  changed, the families that import from it (``dnn`` and ``hybnn`` import
  ``gnn``; ``hybnn`` also imports ``ffnn`` and ``trfnn``) and ``tests/common/``.
- A change to ``xnn.common``, ``configs/``, ``tools/``, ``pyproject.toml`` or
  ``tests/conftest.py`` runs every folder.
- Every folder runs on each merge into ``main`` and once a week, at midnight US
  Eastern time on Monday. To run them all on a branch, start the workflow by
  hand (Actions, Tests, Run workflow).
- The ``Tests passed`` job reports the overall result. It is the check that
  branch protection requires.

Put a new test file in the folder of the family it tests. Put it in
``tests/common/`` when it tests shared code or uses models of several families.
Keep file names unique across the folders, because test files import helpers
from each other by name.

Conventions for new code
========================
- A new model gets, at minimum: a registration test, an invariance or
  equivariance test, a periodic-stress test and, if it exports, a
  script-versus-eager parity test.
- Every model also pins its fidelity: agreement with its reference to round-off,
  whatever the implementation method. The reference is the upstream code under
  identical weights (guarded by an import check), the reference program, or,
  where neither exists, an independent implementation of the paper's equations
  and its published numbers.
- A model with an upstream code is also at least as fast and, preferably, as
  memory-efficient as that code, in float32 and float64, on the upstream's own
  hardware, before it is merged. A fast path matches the reference's numbers
  and derivatives to round-off and gets a test under Triton's interpreter.
- Its defaults are those of the upstream code as shipped; where the paper
  differs, the paper's values are options, which the tests and notebooks pass.
  Options draw on the shared abstractions (radial bases, cutoffs, activations)
  wherever they fit. A config key or option that nothing uses is ignored with
  an ``IgnoredOptionWarning`` (``tests/common/test_config_options.py``).
