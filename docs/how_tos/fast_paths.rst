.. _howto-fast-paths:

*******************************
Run Models on Fused GPU Kernels
*******************************

The blocks that dominate the cost of an equivariant model have two
implementations in xnn: the reference one, which is the canonical, faithful
code, and a fast path on fused GPU kernels from `cuEquivariance
<https://github.com/NVIDIA/cuEquivariance>`_. Both use the same parameters, so
a model, a checkpoint or a hub entry is the same object either way, and the
choice is made when the model runs.

Blocks with a fast path:

* the message step of the equivariant convolution (gather, ``uvu`` tensor
  product, sum onto the receivers), used by MACE and NequIP, as one kernel
  (:class:`~xnn.gnn.fast.ConvTensorProduct`);
* the symmetric contraction of MACE (the product basis);
* the dispersion terms DFT-D3 and DFT-D4: the Axilrod-Teller-Muto three-body
  term as fused Triton kernels, and D4's large-system EEQ solve with a
  factorized Ewald operator (see `Dispersion terms`_);
* the reciprocal sum of the Latent Ewald Summation (LES) long-range term (see
  `Long-range term (LES)`_).

Install
=======

.. code-block:: bash

   pip install "xnns[cueq]"

The extra pins the three cuEquivariance packages to the tested release. Their
CUDA kernels are distributed by NVIDIA under its own license, which is why
they are optional; without them, or on a CPU, every model runs its reference
implementation.

The kernels need cuBLAS 12.5 or newer, so install a PyTorch build for CUDA 12.4
or later (``cu124``, ``cu126``) first; a ``cu121`` build pins an older cuBLAS
and the extra does not resolve against it.

Choose the implementation
=========================

Every loader takes ``use_fast``:

.. code-block:: python

   from xnn.common.models import from_pretrained
   from xnn.common.deploy import XNNCalculator

   model = from_pretrained("mace-mp-0-medium", device="cuda")                  # "auto"
   model = from_pretrained("mace-mp-0-medium", device="cuda", use_fast=True)   # kernels wherever available
   model = from_pretrained("mace-mp-0-medium", device="cuda", use_fast=False)  # always the reference
   atoms.calc = XNNCalculator.from_pretrained("mace-off23-small", device="cuda", use_fast="auto")

A model already built changes with :func:`~xnn.common.models.fast.set_use_fast`
(it reaches the potential inside force, dispersion and long-range wrappers), and
``xnn mdi`` takes ``--fast auto|on|off``.

``"auto"``, the default, takes the kernels on a CUDA device when cuEquivariance
is installed and the neighbor graph is large enough for them to be faster. The
kernels have a fixed cost per evaluation, so they lose on small systems. The
thresholds per GPU and precision are in :data:`xnn.gnn.fast.AUTO_POLICY` and
come from measured crossovers.

Precision
=========

The kernels compute the same function as the reference with a different
summation order. In float64 the results agree to about ``1e-14`` relative; in
float32 to about ``1e-6``, and the first fast evaluation in float32 issues a
:class:`~xnn.common.models.fast.FastPathPrecisionWarning` that says so. Pass
``use_fast=False`` when results must match the reference bit for bit.

Training
========

The fast paths are differentiable twice, so a force-matching loss trains
through them, and the trainer saves the same checkpoints as without them.

Dispersion terms
================

:class:`~xnn.common.models.d4.D4Dispersion` and
:class:`~xnn.common.models.d3.D3Dispersion`, standalone or wrapped around a
model, take the same ``use_fast`` setting (:mod:`xnn.common.models.dispersion_fast`).
They need no extra package: the kernels are written in Triton, which ships with
the CUDA builds of PyTorch. Measured on A100, A30 and V100 GPUs they are faster
at every size, from 1.2-2x on clusters of a few dozen atoms to 7-15x above
10 000 atoms, so ``"auto"`` takes them on any CUDA device.

* **Three-body term (D3 and D4).** One kernel evaluates the geometry, the
  three pair C6, the damping and the energy of every triplet in registers, and
  a second one its gradient in closed form, without storing the triplets. The
  pair C6 comes from per-atom factors, so D3 no longer needs its dense
  ``(N, N)`` C6 matrix: the three-body term has no atom limit, and the
  ``"c6_matrix"`` output is returned empty on the fast path
  (:meth:`~xnn.common.models.d3.DFTD3.c6_matrix` forms it on request).
* **EEQ charges (D4, large regime).** The Ewald reciprocal sum is applied
  through factorized phase tables built once per structure, so the
  conjugate-gradient loop has no trigonometry; the real-space part is a sparse
  matrix, and the charge and constraint solves run together.

Results agree with the reference to rounding for the three-body term (about
``1e-14`` relative in float64) and to the solver tolerance for the EEQ charges
(about ``1e-12`` in float64). The three-body kernels are differentiable once,
which is what molecular dynamics, relaxations and inference need. A training
step (a force loss differentiates the forces again) therefore runs the
reference three-body term, as does a model with trainable damping parameters
(``trainable=True``), and a second derivative taken in evaluation mode (a
Hessian) falls back to the reference in the backward pass; all of these are
exact.

Long-range term (LES)
=====================

:class:`~xnn.common.models.les.LatentEwald` follows the same ``use_fast``
setting (and passes it on to the model it wraps). For periodic structures the
reference forms the structure factors from ``(N, M)`` sines and cosines (``M``
wave vectors), all kept for the force backward pass; that grows as ``N^2`` and
does not fit even a 141 GB GPU at 24 000 water atoms. The fast path
(:meth:`~xnn.common.models.les.EwaldSummation.reciprocal_fast`) sums the same
wave vectors with the same weights but builds the structure factors from
factorized phase tables (:mod:`xnn.common.models.reciprocal`, shared with the
D4 EEQ operator), in column blocks that are recomputed in the backward pass. It
is plain PyTorch, differentiable to every order, so force training runs
through it. Results agree with the reference to rounding. It has a fixed cost
of about 2 ms per structure, so ``"auto"`` takes it, structure by structure,
for periodic structures of at least 3000 atoms
(:data:`xnn.common.models.les.AUTO_POLICY`), where it is 2-9 times faster and
needs 10-30 times less memory. Molecular structures take the real-space sum
either way.

Export
======

TorchScript and LAMMPS export always script the reference implementation (the
kernels are eager-only), so an exported model runs anywhere TorchScript does.
