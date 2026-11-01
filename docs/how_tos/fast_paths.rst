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
interaction layers of TensorNet, the shell convolutions of AIMNet2, the neighbor sums of
GemNet's bilinear layers, the triplet sums of DimeNet and DimeNet++, the tensor products of Allegro, the edge-basis sums and ``Ar`` messages of CACE,
the pair sums of ET's attention layers,
every energy term of the OPLS, AMBER, CHARMM and DREIDING force fields, the terms and the charge solve of
classical ReaxFF fields, the three-body term of D3
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

With ``"auto"``, a model takes the fast path on a CUDA device when the kernels
are installed and the system is large enough for them to be faster. We measured
the crossovers per GPU model and precision, separately for evaluation and
training, and store them as minimum edge counts (atom counts for LES):

- MACE: :data:`xnn.gnn.fast.AUTO_POLICY`,
  :data:`xnn.gnn.fast.AUTO_POLICY_TRAINING`
- NequIP: :data:`xnn.gnn.fast.NEQUIP_AUTO_POLICY`,
  :data:`xnn.gnn.fast.NEQUIP_AUTO_POLICY_TRAINING`
- SchNet: :data:`xnn.gnn.fast.schnet.AUTO_POLICY`,
  :data:`xnn.gnn.fast.schnet.AUTO_POLICY_TRAINING`
- PaiNN: :data:`xnn.gnn.fast.painn.AUTO_POLICY`,
  :data:`xnn.gnn.fast.painn.AUTO_POLICY_TRAINING`
- TensorNet: :data:`xnn.gnn.fast.tensornet.AUTO_POLICY`,
  :data:`xnn.gnn.fast.tensornet.AUTO_POLICY_TRAINING`
- AIMNet2: :data:`xnn.gnn.fast.aimnet2.AUTO_POLICY`,
  :data:`xnn.gnn.fast.aimnet2.AUTO_POLICY_TRAINING`
- GemNet: :data:`xnn.gnn.fast.gemnet.AUTO_POLICY`,
  :data:`xnn.gnn.fast.gemnet.AUTO_POLICY_TRAINING`, and for the quadruplet
  variants :data:`xnn.gnn.fast.gemnet.QUADRUPLET_AUTO_POLICY`,
  :data:`xnn.gnn.fast.gemnet.QUADRUPLET_AUTO_POLICY_TRAINING`
- DimeNet: :data:`xnn.gnn.fast.dimenet.AUTO_POLICY`,
  :data:`xnn.gnn.fast.dimenet.AUTO_POLICY_TRAINING`, and for DimeNet++
  :data:`xnn.gnn.fast.dimenet.PP_AUTO_POLICY`,
  :data:`xnn.gnn.fast.dimenet.PP_AUTO_POLICY_TRAINING`
- Allegro: :data:`xnn.gnn.fast.allegro.AUTO_POLICY`,
  :data:`xnn.gnn.fast.allegro.AUTO_POLICY_TRAINING`
- CACE: :data:`xnn.gnn.fast.cace.AUTO_POLICY`,
  :data:`xnn.gnn.fast.cace.AUTO_POLICY_TRAINING`
- ET: :data:`xnn.trfnn.fast.et.AUTO_POLICY`,
  :data:`xnn.trfnn.fast.et.AUTO_POLICY_TRAINING`
- LES: :data:`xnn.common.models.les.AUTO_POLICY`

The dispersion, PaiNN and TensorNet kernels, and those of the OPLS, AMBER,
CHARMM and DREIDING force fields, are faster at every system size, so
``"auto"`` always selects them on a GPU. So are the AIMNet2 kernels, except
for training on small systems on an A30 or an L40S. The thresholds may change in
any release as we rerun our benchmarks, so report the xnn version with results
obtained with ``"auto"``.

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
- **Training.** The MACE, NequIP, LES, SchNet, PaiNN, TensorNet, AIMNet2, GemNet,
  DimeNet, Allegro and CACE fast paths are twice differentiable. So, the force training runs through them, LoRA
  adapters included. The three-body kernels in dispersion module are
  differentiable once. Thus, a training step or a model with trainable damping
  parameters falls back to the exact reference term. The kernels of the
  classical force fields are twice differentiable, so force training and
  Hessians run through them. With a cell gradient (stress) they are
  differentiable once, so a stress loss trains on the reference terms.
- **Memory.** Above one million edges, the radial networks of MACE and NequIP,
  the filter networks of TensorNet and the filters of PaiNN are recomputed in
  the backward pass instead of being kept in memory
  (:func:`~xnn.gnn.models.set_recompute_radial` sets it per model; ``True``
  helps on small GPUs). For PaiNN this needs about a third of the memory of an
  evaluation for 25 to 60 % more time. The SchNet, PaiNN and TensorNet kernels
  need about half the memory of their reference counterparts, the AIMNet2 kernels
  a quarter to three quarters, the Allegro kernels a half to two thirds, the CACE
  kernels under half, the DimeNet kernels a tenth to two fifths and the DimeNet++
  kernels about half from a hundred atoms on. The LES fast path
  needs 10 to 30 times less memory than its reference counterpart on large
  periodic cells.
- **Export.** TorchScript and LAMMPS export frameworks always script the
  reference implementation. Therefore, an exported model runs anywhere
  TorchScript does.
- **Compilation.** The TensorNet fast path compiles its atom-wise steps with
  ``torch.compile`` on the first evaluation in each precision (seconds to a
  minute). Later evaluations of any system size reuse them.

Captured training steps
=======================
A training step on small molecules is bound by kernel launches, not by the
GPU's arithmetic. For DimeNet, DimeNet++ and GemNet, the trainer can record
the whole step once as a CUDA graph and replay it for every batch:

.. code-block:: yaml

   trainer:
     cuda_graph: true

The data loader pads every batch to fixed counts with one inert structure
and builds the model's triplets (and GemNet's quadruplets) on the CPU. The
step runs the reference implementation. The results equal those of ordinary
steps up to round-off. Where ``use_fast`` takes the fused kernels for the
padded batch size, the first steps time ordinary steps on the kernels against
captured ones, and training goes on with the faster. It needs one GPU and a
single head, and drops the last, smaller batch of each epoch. Validation runs
ordinary steps.

Molecular dynamics with classical force fields
==============================================
For an MD run of one OPLS, AMBER, CHARMM or DREIDING structure,
:class:`~xnn.ffnn.models.ClassicalStepper` runs the kernels without the
per-evaluation overhead. It keeps a neighbor list with a skin, rebuilds it
only when an atom has moved by half the skin, and replays each step as a CUDA
graph:

.. code-block:: python

   from xnn.ffnn.models import ClassicalStepper

   stepper = ClassicalStepper(model.to("cuda").eval(), skin=2.0)
   out = stepper({"pos": pos, "atomic_numbers": z, "cell": cell, "pbc": pbc})
   out["energy"], out["forces"]

Each call returns the energy and forces at new positions of the same atoms and
cell. The numbers are those of the model's evaluation to round-off.

The ``notebooks/parity_checks/`` notebooks compare every fast paths with their
reference implementations on real systems.
