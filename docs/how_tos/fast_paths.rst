.. _howto-fast-paths:

*******************************
Run Models on Fused GPU Kernels
*******************************

The expensive blocks of a model have two implementations: the reference,
which is the faithful code, and a fast path on fused GPU kernels. Both use
the same parameters, so a model or checkpoint is the same object either way
and the choice is made when the model runs.

Fast paths exist for the message step and the symmetric contraction of MACE
and NequIP (`cuEquivariance <https://github.com/NVIDIA/cuEquivariance>`_),
the three-body term of DFT-D3 and DFT-D4 and the large-system EEQ solve of
D4 (Triton kernels, shipped with CUDA builds of PyTorch), and the reciprocal
sum of the LES long-range term (plain PyTorch).

Install
=======
Only the MACE / NequIP kernels need an extra package. They require a CUDA
12.6 or newer PyTorch build:

.. code-block:: bash

   pip install cuequivariance==0.6.1 cuequivariance-torch==0.6.1 cuequivariance-ops-torch-cu12==0.6.1

The three packages must come from one release; 0.6.1 is the tested one.
Without them, or on CPU, every model runs its reference implementation.

Choose the implementation
=========================
Every loader takes ``use_fast``:

.. code-block:: python

   from xnn.common.models import from_pretrained
   from xnn.common.models.fast import set_use_fast

   model = from_pretrained("mace-mp-0-medium", device="cuda")                  # "auto"
   model = from_pretrained("mace-mp-0-medium", device="cuda", use_fast=True)   # kernels wherever available
   model = from_pretrained("mace-mp-0-medium", device="cuda", use_fast=False)  # always the reference
   set_use_fast(model, "auto")                                                 # change a built model

``XNNCalculator.from_pretrained(..., use_fast=...)`` and ``xnn mdi --fast
auto|on|off`` do the same. ``"auto"`` takes a kernel on a CUDA device when it
is installed and the system is large enough for it to be faster; the
thresholds per GPU and precision are measured crossovers
(:data:`xnn.gnn.fast.AUTO_POLICY`, :data:`xnn.common.models.les.AUTO_POLICY`).
The dispersion kernels are faster at every size and are always taken on a
GPU.

What to expect
==============
- **Precision.** The kernels sum in a different order. Float64 agrees with the
  reference to about 1e-14 relative, float32 to about 1e-6; the first float32
  evaluation warns once
  (:class:`~xnn.common.models.fast.FastPathPrecisionWarning`). Use
  ``use_fast=False`` when results must match bit for bit.
- **Training.** The MACE / NequIP and LES fast paths are differentiable
  twice, so force training runs through them. The dispersion three-body
  kernels are differentiable once; a training step or a model with
  trainable damping parameters falls back to the exact reference term.
- **Memory.** Above one million edges the radial networks of MACE and NequIP
  recompute their activations in the backward pass instead of keeping them
  (:func:`~xnn.gnn.models.set_recompute_radial` sets it per model). The LES
  fast path needs 10 to 30 times less memory than the reference on large
  periodic cells.
- **Export.** TorchScript and LAMMPS export always script the reference
  implementation, so an exported model runs anywhere TorchScript does.

``examples/parity_checks/`` compares every fast path with its reference on
real systems.
