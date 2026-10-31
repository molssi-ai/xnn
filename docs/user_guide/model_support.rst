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
     - |N|
     - |Y|
     - |N|
     - |N|
     - |N|
   * - PhysNet
     - |Y|
     - |N|
     - |Y|
     - |N|
     - |N|
     - |N|
   * - HDNNP
     - |Y|
     - |N|
     - |Y|
     - |N|
     - |N|
     - |N|
   * - SE(3) steerable CNN / 3D CNN
     - |Y|
     - |N|
     - |Y|
     - |N|
     - |N|
     - |N|
   * - Spherical CNN
     - |Y|
     - |N|
     - |Y|
     - |N|
     - |N|
     - |N|
   * - ReaxFF / ReaxFF-nn
     - |Y|
     - |O|
     - |Y|
     - |N|
     - |N|
     - |N|
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
     - |N|
     - |Y|
     - |N|
     - |Y|
     - |Y|
   * - HIPPO
     - |Y|
     - |N|
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
     - |N|
     - |N|
   * - EquiformerV2
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |N|
     - |N|
   * - BAMBOO
     - |Y|
     - |N|
     - |Y|
     - |N|
     - |N|
     - |N|
   * - SpookyNet
     - |Y|
     - |Y|
     - |Y|
     - |N|
     - |N|
     - |N|
   * - PM3 / PM6 / PM6-ORG / PM7
     - |Y|
     - |N|
     - |Y|
     - |N|
     - |N|
     - |N|

- **ASE and MDI** serve every model, with or without a D3 / D4 term (see
  :ref:`deployment`).
- **Torchscript-LAMMPS** exports the reference implementation with
  ``xnn export``. A long-range (LES) model needs the whole system on one rank.
- **Reference-Path** rates the reference implementation against the upstream
  code:

  - |Y| It matches the upstream code to round-off in float32 and float64. It
    is at least as fast and usually as memory efficient.
  - |O| It matches the paper. No upstream code is available to compare compute
    and memory efficiencies with.
  - |N| It is slower or uses more memory than the upstream code, or the
    comparison and development is still in progress.
- **Fast-Path (Inference)** gives energies, forces and stress from fused GPU
  kernels that agree with the reference to round-off.
- **Fast-Path (Training)** also gives the second derivatives of a force loss,
  so training runs through the kernels.
- ``use_fast="auto"`` picks the fast path above a measured system size (see
  :ref:`howto-fast-paths`).

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
     - |N|
     - |N|
     - |Y|
   * - D4
     - |Y|
     - |Y|
     - |Y|
     - |N|
     - |N|
     - |Y|
   * - LES
     - |Y|
     - |Y|
     - |Y|
     - |N|
     - |Y|
     - |Y|

- **DimeNet.** The upstream times its steps on TensorFlow's graph executor.
  Compare it with a captured xnn step (``trainer.cuda_graph: true``, see
  :ref:`howto-fast-paths`), which runs the same reference implementation and
  is faster than the upstream at every size.
- **DimeNet++ and GemNet in float64.** Both fast paths run in float32 and
  float64, but Triton has no float64 matrix products. In float64, ``"auto"``
  keeps the reference for DimeNet++ (its fast path is slower there but needs
  half the memory). For GemNet, it keeps the reference in training and for the
  quadruplet layers, and takes the fast path for evaluation from a few hundred
  atoms on.
- **OPLS, AMBER, CHARMM and DREIDING.** All four match LAMMPS to round-off.
  DREIDING / X6 has no code from its authors, so LAMMPS serves as its upstream.
  On a GPU, all four evaluate and train on fused kernels. An MD step with
  :class:`~xnn.ffnn.models.ClassicalStepper` is faster than with the GPU build
  of LAMMPS, list rebuilds included. A training step is two to two and a half
  times faster than on the eager terms, which still handle stress losses.
- **ReaxFF.** It matches LAMMPS to round-off but is slower than LAMMPS's GPU
  build, as it has no fused kernels yet. Above 2048 atoms, it solves for the
  charges by conjugate gradients instead of a dense solve.
- **Force fields in LAMMPS.** OPLS, AMBER, CHARMM, DREIDING and classical
  ReaxFF deploy as inputs for LAMMPS's own styles
  (:func:`~xnn.ffnn.models.write_lammps_input`), not as a TorchScript file.
  These styles evaluate the model to round-off. ReaxFF-nn has no LAMMPS route.
- **D3, D4 and LES.** All three match their upstream codes to round-off. We
  have not compared their speed with those codes yet. The dispersion kernels
  are differentiable only once, so training uses the reference three-body term
  and the dispersion fast path serves evaluation only.
