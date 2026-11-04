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
     - |N|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
   * - SpookyNet
     - |Y|
     - |Y|
     - |Y|
     - |Y|
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
  ``xnn export``. A model with long-range terms (LES, or the charges of
  AIMNet2, PhysNet and HDNNP 3G / 4G) needs the whole system on one rank.
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
  :ref:`howto-fast-paths`). For Equiformer and EquiformerV2 it is the other way
  round: their fast paths win on small systems, so ``"auto"`` takes them below a
  measured size.

Notes on individual models:

- **DimeNet.** The upstream times its steps on TensorFlow's graph executor.
  Compare it with a captured xnn step (``trainer.cuda_graph: true``, see
  :ref:`howto-fast-paths`), which runs the same reference implementation and
  is faster than the upstream at every size.
- **ANI.** Its fast path is faster at every size in float32. In float64,
  ``"auto"`` keeps the reference on a T4 or an L40S, whose float64 throughput
  is low.
- **PhysNet.** The upstream runs on TensorFlow's graph executor. Compare it
  with captured xnn steps (``trainer.cuda_graph: true`` for training,
  ``XNNCalculator(..., cuda_graph=True)`` for evaluations, see
  :ref:`howto-fast-paths`). They run the same reference implementation and are
  faster than the upstream at every size. The fast path wins from about a
  thousand atoms.
- **HDNNP.** On a GPU, Triton kernels evaluate the symmetry functions of
  types 1, 2, 3 and 9 with cosine or tanh cutoffs, for evaluation and
  training. Other function sets, and every set on a CPU, run the symmetry
  functions compiled with ``torch.compile``. That code is differentiable only
  once, so a training step with those runs the reference. RuNNer runs only on
  CPUs. There, xnn's C++ kernels take about half of RuNNer's time on one and
  eight threads, and the Reference-Path cell rates them.
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
- **BAMBOO.** Its forces are the exact gradient of the energy. The forces the
  upstream code reports differ from that gradient by about one part in 10^10.
- **DimeNet++ and GemNet in float64.** Both fast paths run in float32 and
  float64, but Triton has no float64 matrix products. In float64, ``"auto"``
  keeps the reference for DimeNet++ (its fast path is slower there but needs
  half the memory). For GemNet, it keeps the reference in training and for the
  quadruplet layers, and takes the fast path for evaluation from a few hundred
  atoms on.
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
- **Force fields in LAMMPS.** OPLS, AMBER, CHARMM, DREIDING and classical
  ReaxFF deploy as inputs for LAMMPS's own styles
  (:func:`~xnn.ffnn.models.write_lammps_input`), not as a TorchScript file.
  These styles evaluate the model to round-off. ReaxFF-nn has no LAMMPS route.
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
     - |N|
     - |Y|
     - |Y|
   * - D4
     - |Y|
     - |Y|
     - |Y|
     - |N|
     - |Y|
     - |Y|
   * - LES
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|
     - |Y|

- **D3, D4 and LES.** All three match their upstream codes to round-off. LES
  is faster than its upstream on the same GPU, for evaluation and training. The
  D3 and D4 upstreams run only on the CPU. There, xnn's C++ kernels (see
  :ref:`howto-fast-paths-cpu`) take 0.04 to 0.51 of simple-dftd3's time and
  0.11 to 0.61 of dftd4's from 81 atoms on, on one and eight threads. On a
  21-atom molecule they take two to six times as long: the whole upstream call
  is shorter than the Python overhead of an xnn evaluation, so the
  Reference-Path cells stay red. On a GPU the three-body kernels are twice
  differentiable, so a training step with a force loss also runs them. They do
  not differentiate ``s9``, ``a1`` or ``a2``. When those are trained, the
  reference three-body term runs.
