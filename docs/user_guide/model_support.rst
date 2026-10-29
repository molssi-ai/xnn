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
supports. |Y| means that the feature is complete and available, |N| that it
is not available yet.

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
     - |N|
     - |N|
     - |N|
   * - GemNet
     - |Y|
     - |Y|
     - |Y|
     - |N|
     - |N|
     - |N|
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
     - |N|
     - |Y|
     - |Y|
   * - MACE
     - |Y|
     - |Y|
     - |Y|
     - |N|
     - |Y|
     - |Y|
   * - Allegro
     - |Y|
     - |Y|
     - |Y|
     - |N|
     - |N|
     - |N|
   * - CACE
     - |Y|
     - |N|
     - |Y|
     - |N|
     - |N|
     - |N|
   * - AIMNet2
     - |Y|
     - |Y|
     - |Y|
     - |N|
     - |N|
     - |N|
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
     - |N|
     - |Y|
     - |N|
     - |N|
     - |N|
   * - OPLS-AA / L-OPLS
     - |Y|
     - |N|
     - |Y|
     - |N|
     - |N|
     - |N|
   * - DREIDING / X6
     - |Y|
     - |N|
     - |Y|
     - |O|
     - |N|
     - |N|
   * - AMBER (ff94)
     - |Y|
     - |N|
     - |Y|
     - |N|
     - |N|
     - |N|
   * - CHARMM22
     - |Y|
     - |N|
     - |Y|
     - |N|
     - |N|
     - |N|
   * - AMOEBA
     - |Y|
     - |N|
     - |Y|
     - |N|
     - |N|
     - |Y|
   * - HIPPO
     - |Y|
     - |N|
     - |Y|
     - |N|
     - |N|
     - |Y|
   * - ET
     - |Y|
     - |Y|
     - |Y|
     - |N|
     - |N|
     - |N|
   * - Equiformer
     - |Y|
     - |Y|
     - |Y|
     - |N|
     - |N|
     - |N|
   * - EquiformerV2
     - |Y|
     - |Y|
     - |Y|
     - |N|
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

- The D3, D4 and LES terms match their upstream codes to round-off. Their
  speed against those codes has not been compared yet.
- The dispersion kernels are differentiable once. A training step therefore
  uses the reference three-body term, and the fast path serves evaluation.
