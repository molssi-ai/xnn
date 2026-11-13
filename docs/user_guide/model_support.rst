.. _model-support:

*************
Model support
*************

.. role:: check-green
.. role:: check-orange
.. role:: cross-red

.. |Y| replace:: :check-green:`✔`
.. |O| replace:: :check-orange:`✔`
.. |N| replace:: :cross-red:`✘`

The table shows which deployment routes and implementations each model
supports. |Y| means that the feature is complete and available, |O| that it
is available with a limit given below the table, and |N| that it is not
available yet. A |Y| in the Reference-Path column means that the reference
implementation matches its upstream code to round-off in float32 and float64
and is at least as fast, for evaluation and for a training step.

.. list-table::
   :header-rows: 1
   :widths: 28 8 14 8 12 15 15

   * - Model
     - ASE
     - Torchscript-LAMMPS
     - MDI
     - Reference-Path
     - Fast-Path (Training)
     - Fast-Path (Inference)
   * - SchNet
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
   * - DimeNet / DimeNet++
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
   * - GemNet
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
   * - PaiNN
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
   * - TensorNet
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
   * - QET
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
   * - MEGNet
     - |N|
     - |N|
     - |N|
     - |Y|
     - |Y|
     - |Y|
   * - NequIP
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
   * - MACE
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
   * - Allegro
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
   * - CACE
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
   * - AIMNet2
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
   * - ANI
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
   * - PhysNet
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
   * - HDNNP
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
   * - QRNN
     - |Y|
     - |Y|
     - |Y|
     - |O|
     - |Y|
     - |Y|
   * - SE(3) steerable CNN / 3D CNN
     - |Y|
     - |N|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
   * - Spherical CNN
     - |Y|
     - |N|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
   * - ReaxFF / ReaxFF-nn
     - |Y|
     - |O|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
   * - OPLS-AA / L-OPLS
     - |Y|
     - |O|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
   * - DREIDING / X6
     - |Y|
     - |O|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
   * - AMBER (ff94)
     - |Y|
     - |O|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
   * - CHARMM22
     - |Y|
     - |O|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
   * - AMOEBA
     - |Y|
     - |O|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
   * - HIPPO
     - |Y|
     - |O|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
   * - ET
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
   * - Equiformer
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
   * - EquiformerV2
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
   * - BAMBOO
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
   * - SpookyNet
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |O|
     - |Y|
   * - ByteFF
     - |O|
     - |O|
     - |O|
     - |Y|
     - |Y|
     - |Y|
   * - ByteFF-Pol
     - |O|
     - |O|
     - |O|
     - |Y|
     - |Y|
     - |Y|
   * - PM3 / PM6 / PM6-ORG / PM7
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
   * - GDML / sGDML
     - |Y|
     - |N|
     - |Y|
     - |Y|
     - |Y|
     - |Y|

- **ASE and MDI** serve every model, with or without a D3 / D4 term (see
  :ref:`deployment`).
- **Torchscript-LAMMPS** exports the reference implementation with
  ``xnn export``. A model with long-range terms (LES, D4, SpookyNet, or the
  charges of AIMNet2, BAMBOO, PhysNet, QET, QRNN and HDNNP 3G / 4G) needs the whole system on one
  rank, and so do PM3, PM6, PM6-ORG and PM7, whose energy takes every atom
  pair of a molecule.
- **Reference-Path** rates the reference implementation against the upstream
  code:

  - |Y| It matches the upstream code to round-off in float32 and float64. It
    is at least as fast and usually as memory efficient.
  - |O| It matches the paper. No upstream code is available to compare compute
    and memory efficiencies with. Or it matches the upstream code, but is
    faster than it only with its optional C++ kernels.
  - |N| It is slower or uses more memory than the upstream code, or the
    comparison and development is still in progress.
- **Fast-Path (Inference)** gives energies, forces and stress from fused GPU
  kernels that agree with the reference to round-off.
- **Fast-Path (Training)** also gives the second derivatives of a force loss,
  so training runs through the kernels.
- ``use_fast="auto"`` picks the fast path above a measured system size (see
  :ref:`howto-fast-paths`). For Equiformer, EquiformerV2 and the evaluation of
  SpookyNet it is the other way round: their fast paths win on small systems, so ``"auto"`` takes them below a
  measured size.

Why a cell is orange or red:

- **SE(3) steerable CNN, 3D CNN and spherical CNN, LAMMPS (red).** They build
  a grid around every atom at every step. That suits datasets of small
  molecules, not long molecular dynamics, so they have no LAMMPS export. They
  run molecular dynamics through ASE and MDI.
- **OPLS, AMBER, CHARMM, DREIDING and ReaxFF, LAMMPS (orange).** They deploy
  as inputs for LAMMPS's own styles, not as a TorchScript file. LAMMPS
  evaluates the same energy to round-off, but only in orthorhombic cells and
  without a switched cutoff. ReaxFF-nn has no LAMMPS route: LAMMPS has no
  style for its neural bond orders.
- **AMOEBA and HIPPO, LAMMPS (orange).** They deploy to LAMMPS's AMOEBA
  package, which has no charge flux, no exchange polarization and none of
  HIPPO's ion terms. In a cell,
  the forces of LAMMPS 29Aug2024 do not match its energy.
- **SpookyNet, Fast-Path (Training) (orange).** In ordinary training steps its
  fast path is faster only on large systems. Captured training steps are
  faster with it at every size.
- **ByteFF and ByteFF-Pol, ASE and MDI (orange).** Their parameters are
  predicted from the molecules, so a model serves the system it is bound to.
- **ByteFF, LAMMPS (orange).** It deploys as inputs for LAMMPS's own styles,
  with the parameters of the system it is bound to.
- **ByteFF-Pol, LAMMPS (orange).** It exports as TorchScript with the
  parameters of the system it is bound to, for molecules, clusters and
  periodic cells.
- **D4, Reference-Path (orange).** See the add-on terms below.
- **MEGNet, ASE, LAMMPS and MDI (red).** MEGNet predicts one property per
  structure, such as a band gap, rather than a potential energy. It has no
  forces for molecular dynamics.
- **GDML and sGDML, LAMMPS (red).** Each model serves one molecule with its atoms in a fixed order. They
  run molecular dynamics through ASE and MDI.

Notes on individual models:

- **GDML and sGDML.** Training is one closed-form fit, run on the GPU when there
  is one. When the kernel matrix does not fit in the GPU's memory, the fit
  switches to an iterative solve.
- **QRNN.** Its Coulomb and dispersion terms take every atom pair of a molecule,
  so in LAMMPS the whole molecule must be on one rank in a non-periodic box (a
  zero cell). Periodic cells need ``coulomb: false`` and ``dispersion: false``.
  No code was published with the paper. The reference matches the paper's
  equations. The fast path runs the AEV on the kernels of ANI and,
  when every element has networks of the same shape (the default), evaluates
  them together. ``"auto"`` takes it below about 300k edges, where it is faster.
- **DimeNet.** The upstream times its steps on TensorFlow's graph executor.
  Compare it with a captured xnn step (``trainer.cuda_graph: true``, see
  :ref:`howto-fast-paths`), which runs the same reference implementation and
  is faster than the upstream at every size.
- **ANI.** The upstream is torchani 2.2.4. Its fast path is faster at every
  size in float32. In float64,
  ``"auto"`` keeps the reference on a T4 or an L40S, whose float64 throughput
  is low.
- **PhysNet.** The upstream is MMunibas/PhysNet at commit ``e243e2c``. It
  runs on TensorFlow's graph executor. Compare it
  with captured xnn steps (``trainer.cuda_graph: true`` for training,
  ``XNNCalculator(..., cuda_graph=True)`` for evaluations, see
  :ref:`howto-fast-paths`). They run the same reference implementation and are
  faster than the upstream at every size. The fast path wins on large systems,
  from a few hundred to about a thousand atoms depending on the GPU.
- **HDNNP.** On a GPU, Triton kernels evaluate the symmetry functions of
  types 1, 2, 3 and 9 with cosine or tanh cutoffs, for evaluation and
  training. Other function sets, and every set on a CPU, run the symmetry
  functions compiled with ``torch.compile``. That code is differentiable only
  once, so a training step with those runs the reference. RuNNer (2.0.5) runs
  only on CPUs. There, a 2G model forms its forces from written-out derivatives
  in one compiled pass, which takes 0.54 to 0.89 of RuNNer's time on one and
  eight threads.
- **SE(3) steerable CNN and 3D CNN.** Their fast path runs the 3D
  convolutions on cuDNN's channels-last kernels, in batch chunks that bound
  the kernels' workspace, with cuDNN's algorithm search. From a few molecules
  on it is faster than the reference and needs less memory, in evaluation and
  training; ``"auto"`` keeps the reference below that, where the kernels'
  workspace would need more memory. cuDNN searches once for every new input
  shape, so a training run pays a few seconds for each batch size it meets.
- **Spherical CNN.** Its fast path computes the Fourier transforms on Triton
  kernels and the products of the spectra as one matrix product per degree.
  It is faster than the reference at every size, and faster than the
  upstream's CUDA kernels, which run in float32 only and cannot be
  differentiated twice, so the upstream cannot train on forces. On a single
  molecule in float32 evaluation, it pads the spectra to take all degrees in
  one product, which needs about a third more memory than the upstream.
- **ByteFF and ByteFF-Pol.** A model serves one system at a time: bind it with
  ``model.bind(...)`` before an ASE calculator evaluates it, or with the
  ``structure`` option of the checkpoint's config for the MDI engine. The
  upstream is ByteDance-Seed/byteff2 1.1.0 (commit ``663451d``). For molecules
  and clusters, the reference is faster than it on its CPU and on an A100. In
  periodic cells, ByteFF-Pol reproduces byteff2's OpenMM setup, whose AMOEBA
  scale factors byteff2 patches. There the reference is faster than OpenMM's CPU
  platform.
- **ByteFF and ByteFF-Pol fast paths.** With a neighbor-list cutoff, ByteFF runs
  its terms on the classical force-field kernels. Without one, ByteFF and
  ByteFF-Pol compile their energy terms, after 1000 evaluations of one system
  under ``"auto"`` or at once with ``use_fast=True``. In cells, ByteFF-Pol
  solves its dipoles on the AMOEBA kernels, and molecular dynamics runs on step
  plans replayed as CUDA graphs (:class:`~xnn.common.deploy.StepGraph`), with
  fused pair kernels. The MDI engine runs them with the ASPC dipole predictor
  and late checks (``StepGraph(..., late_checks=True)``). On an A100 a float32
  step then takes 0.84 to 0.92 times as long as in byteff2's OpenMM CUDA setup.
  Without them, it takes up to 1.09 times as long. A float64 step takes 0.38 to
  0.56 times as long as OpenMM's double-precision step.
- **BAMBOO.** The upstream is release 202404 (commit ``386ac0f``). Its forces
  are the exact gradient of the energy. The forces the
  upstream code reports differ from that gradient by about one part in 10^10.
- **SpookyNet.** The upstream is OUnke/SpookyNet 1.0 (commit ``d57b1fc``). Its
  time goes to launching many small kernels. For evaluation, its fast path
  compiles all interaction modules into one block. Evaluations up to about 30k
  edges run 1.4 to 1.5 times faster with less memory, and ``"auto"`` takes the
  fast path there. For training, its fast path fuses every residual MLP with its
  second derivatives. Captured training steps run up to 1.2 times faster with it
  at every size, and ``"auto"`` takes it there. In ordinary steps it wins only
  on large systems, from 16k to 150k edges depending on the GPU and precision
  (on an A100 only in float64), and ``"auto"`` takes it there. Captured steps themselves make training and evaluation 3 to 5 times
  faster on small and medium systems (see :ref:`howto-fast-paths`).
- **DimeNet++ and GemNet in float64.** Both fast paths run in float32 and
  float64, but Triton has no float64 matrix products. In float64, ``"auto"``
  keeps the reference for DimeNet++ on an A100 (its fast path is slower there
  but needs half the memory) and takes it from about 4k edges on the other
  GPUs. For GemNet, it keeps the reference for the quadruplet layers and, on an
  A100 or an A30, in training. It takes the fast path for evaluation from a few
  hundred atoms on.
- **OPLS, AMBER, CHARMM and DREIDING.** All four match LAMMPS to round-off.
  DREIDING / X6 has no code from its authors, so LAMMPS serves as its upstream.
  On CPUs, an MD step with :class:`~xnn.ffnn.models.ClassicalStepper` is faster
  than LAMMPS on the same cores, list rebuilds included.
  On a GPU, all four evaluate and train on fused kernels. An MD step with
  :class:`~xnn.ffnn.models.ClassicalStepper` is faster than with the GPU build
  of LAMMPS, list rebuilds included. A training step is two to two and a half
  times faster than on the eager terms, which still handle stress losses.
- **ReaxFF.** It matches LAMMPS to round-off. On CPUs, an MD step with
  :class:`~xnn.ffnn.models.ReaxFFStepper` is faster than LAMMPS on the same
  cores, list rebuilds included. Above 2048 atoms, it solves for
  the charges by conjugate gradients instead of a dense solve. On a GPU,
  ReaxFF and ReaxFF-nn evaluate energies, forces and stress on fused kernels
  in both conventions. The neural bond orders of ReaxFF-nn stay in PyTorch.
  Training runs on the kernels as well, 1.1 to 2.7 times faster than on the
  eager terms. A stress loss trains on the eager terms. An MD step of a
  classical field with :class:`~xnn.ffnn.models.ReaxFFStepper` is faster than
  with the GPU build of LAMMPS, list rebuilds included.
- **Force fields in LAMMPS.** OPLS, AMBER, CHARMM, DREIDING, ByteFF and classical
  ReaxFF deploy as inputs for LAMMPS's own styles
  (:func:`~xnn.ffnn.models.write_lammps_input`), not as a TorchScript file.
  These styles evaluate the model to round-off. ByteFF writes the parameters
  of the system it is bound to. ReaxFF-nn has no LAMMPS route. ByteFF-Pol
  exports as TorchScript (see :ref:`howto-lammps`).
- **PM3, PM6, PM6-ORG and PM7.** All four match MOPAC to its printed precision.
  MOPAC runs only on CPUs. On the same cores and at the same SCF convergence
  (MOPAC's ``PRECISE``), PM3 and PM6 are faster than MOPAC at every size.
  PM6-ORG and PM7 are as fast as MOPAC below about 30 atoms and faster above.
  Like MOPAC, the SCF of large systems rotates the previous orbitals near
  convergence instead of diagonalizing the Fock matrix. The hydrogen-bond, H-H
  and dispersion corrections run on the host with their own gradients. Second
  derivatives come from their PyTorch form. The fast path replays CUDA graphs of
  the reference's launches (see :ref:`howto-fast-paths`). An evaluation is 2.2
  to 3.7 times as fast below 100 atoms and 1.3 to 2.4 times faster above. A
  force-loss training step also replays the integral build with its second
  derivatives and the density response. It is 1.6 to 3 times faster. The
  TorchScript export runs the same reference code and matches it to round-off
  (see :ref:`howto-lammps`).
- **AMOEBA and HIPPO.** Both match Tinker 8 to round-off. On CPUs, their
  kernels run as C++ loops (see :ref:`howto-fast-paths-cpu`) and are faster
  than Tinker 8. On GPUs, both run on fused kernels. Both deploy as inputs for
  LAMMPS's AMOEBA package (see :ref:`howto-lammps`), which has no charge flux.

Add-on terms
============

An add-on term wraps any model above and adds its energy to the model's
prediction. Enable it in the model's config: ``dispersion: {name: d3}`` or
``dispersion: {name: d4}`` for a dispersion term, ``long_range: {n_channels: 4}``
for the LES long-range term (see :ref:`configuration`). The columns and marks
are the same as above.

.. list-table::
   :header-rows: 1
   :widths: 28 8 14 8 12 15 15

   * - Add-on
     - ASE
     - Torchscript-LAMMPS
     - MDI
     - Reference-Path
     - Fast-Path (Training)
     - Fast-Path (Inference)
   * - D3
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
   * - D4
     - |Y|
     - |Y|
     - |Y|
     - |O|
     - |Y|
     - |Y|
   * - LES
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|

- **D3, D4 and LES.** All three match their upstream codes to round-off. LES is
  faster than its upstream on the same GPU, for evaluation and training. The D3
  and D4 upstreams (simple-dftd3 1.6.0 and dftd4 4.2.0) run only on the CPU.
  There, xnn's PyTorch path computes a molecule over all atom pairs and
  triangles in one compiled call, and a cell with compiled three-body tiles,
  both with written-out forces. D3 is faster than simple-dftd3 at every size,
  0.9 of its time at 21 atoms and down to 0.03 for large systems. D4 is faster
  than dftd4 everywhere except a 21-atom molecule on eight threads, where it
  takes 1.13 to 1.28 of dftd4's time. The optional C++ kernels (see :ref:`howto-fast-paths-cpu`)
  are faster than dftd4 at every size, so the D4 Reference-Path cell is orange.
  On a GPU the three-body kernels are twice differentiable, so a training step
  with a force loss also runs them. They do not differentiate ``s9``, ``a1`` or
  ``a2``. When those are trained, the reference three-body term runs.
