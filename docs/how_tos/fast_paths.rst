.. _howto-fast-paths:

*******************************
Run Models on Fused GPU Kernels
*******************************

The expensive blocks of a model have two implementations: the **reference**
implementation, which is the faithful to its upstream counterpart, and a
**fast** implementation on fused GPU kernels. Both use the same set of
parameters. Thus, a model or a checkpoint will have same object type either way
and the path choice is made at runtime. 

Currently, the fast paths exist for the message phase and the symmetric
contraction of MACE and NequIP (where we use `cuEquivariance
<https://github.com/NVIDIA/cuEquivariance>`_), the continuous-filter
convolution of SchNet, the message block of PaiNN, the embedding and
interaction layers of TensorNet, the three-body term of D3
and D4 and the large-system EEQ solve of D4 (where we use the Triton kernels,
shipped with CUDA builds of PyTorch), and the reciprocal sum of the LES
long-range term (for which, we use plain PyTorch).

Install
=======
At the moment (v0.14.0), only the MACE and NequIP kernels need an extra
dependency package installed to be activated. The installation requires a CUDA
12.6 or newer PyTorch build. You can install the cuEquivariance packages using
pip:

.. code-block:: bash

   pip install cuequivariance==0.6.1 cuequivariance-torch==0.6.1 cuequivariance-ops-torch-cu12==0.6.1

The three packages must come from one xnn release. Without them, or on CPU,
every model falls back to its reference implementation.

Choose the implementation
=========================
Every model loader function exposes a ``use_fast`` argument. Set it to ``True``
to use the fast kernels wherever available:

.. code-block:: python

   from xnn.common.models import from_pretrained
   from xnn.common.models.fast import set_use_fast

   model = from_pretrained("mace-mp-0-medium", device="cuda")                  # "auto"
   model = from_pretrained("mace-mp-0-medium", device="cuda", use_fast=True)   # kernels wherever available
   model = from_pretrained("mace-mp-0-medium", device="cuda", use_fast=False)  # always the reference
   set_use_fast(model, "auto")                                                 # change a built model

The xnn's ``XNNCalculator.from_pretrained(..., use_fast=...)`` and ``xnn mdi --fast
auto|on|off`` allow for the same choices. 

Note the ``"auto"`` option for the ``use_fast`` argument: it selects a kernel on
a CUDA device when it is installed and when the system is large enough for it to
be faster. The thresholds per GPU and precision are based on our own
experimental benchmarks with measured crossovers
(:data:`xnn.gnn.fast.AUTO_POLICY`, :data:`xnn.gnn.fast.schnet.AUTO_POLICY`,
:data:`xnn.gnn.fast.painn.AUTO_POLICY`, :data:`xnn.gnn.fast.tensornet.AUTO_POLICY`,
:data:`xnn.common.models.les.AUTO_POLICY`).
A model in training mode has its own thresholds, since a training step has its
own crossover (:data:`xnn.gnn.fast.schnet.AUTO_POLICY_TRAINING`,
:data:`xnn.gnn.fast.painn.AUTO_POLICY_TRAINING`,
:data:`xnn.gnn.fast.tensornet.AUTO_POLICY_TRAINING`).
These thresholds are subject to change in any release. So, make sure to clarify
the xnn's version in your reports if you choose to use the ``"auto"`` option. The
dispersion and PaiNN kernels are faster at every system size, so they are always
selected on a GPU.

What to expect from fast implementations
========================================
- **Precision.** The kernels sum in a different order. In float64, the agreement
  between the fast and reference implementations are about 1e-14 in relative
  error. In float32, the agreement is about 1e-6. Performing a float32
  evaluation warns the user once on the first usage with a
  :class:`~xnn.common.models.fast.FastPathPrecisionWarning`. Use
  ``use_fast=False`` when results must match the reference implementation, bit
  for bit.
- **Mixed precision (HIPPO).** The HIPPO kernels evaluate the radial ladders of
  the pair terms in float64 and cast, so a float32 model stays at the round-off
  of the float32 reference. ``mixed_precision=True`` (a model option, also a
  config key) evaluates them in float32 throughout, as Tinker9's mixed precision
  does. It is ten to thirty percent faster per MD step, depending on the GPU.
  The energies stay at the float32 round-off, the forces lose up to one digit.
  ``use_fast`` alone never changes the numbers.
- **Training.** The MACE, NequIP, LES, SchNet, PaiNN and TensorNet fast paths
  are twice differentiable. So, the force training runs through them, LoRA
  adapters included. The three-body kernels in dispersion module are
  differentiable once. Thus, a training step or a model with trainable damping
  parameters falls back to the exact reference term.
- **Memory.** Above one million edges, the radial networks of MACE and NequIP,
  the filter networks of TensorNet and the filters of PaiNN are recomputed in
  the backward pass instead of being kept in memory
  (:func:`~xnn.gnn.models.set_recompute_radial` sets it per model; ``True``
  helps on small GPUs). For PaiNN this needs about a third of the memory of an
  evaluation for 25 to 50 % more time. The SchNet, PaiNN and TensorNet kernels
  need about half the memory of their reference counterparts. The LES fast path
  needs 10 to 30 times less memory than its reference counterpart on large
  periodic cells.
- **Export.** TorchScript and LAMMPS export frameworks always script the
  reference implementation. Therefore, an exported model runs anywhere
  TorchScript does.
- **Compilation.** The TensorNet fast path compiles its atom-wise steps with
  ``torch.compile`` on the first evaluation in each precision (seconds to a
  minute). Later evaluations of any system size reuse them.

The ``notebooks/parity_checks/`` notebooks compare every fast paths with their
reference implementations on real systems.
