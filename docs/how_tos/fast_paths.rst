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
<https://github.com/NVIDIA/cuEquivariance>`_), the continuous-filter convolution
of SchNet, the message block of PaiNN, the embedding and interaction layers of
TensorNet, the shell convolutions of AIMNet2, the neighbor sums of GemNet's
bilinear layers, the triplet sums of DimeNet and DimeNet++, the tensor products
of Allegro, the edge-basis sums and ``Ar`` messages of CACE, the pair sums of
ET's attention layers, the transformer blocks of Equiformer and EquiformerV2
(compiled with ``torch.compile``), the 3D convolutions of the SE(3) steerable
CNN and the 3D CNN (on cuDNN's channels-last kernels), the Fourier transforms
and spectral products of the spherical CNN, the atomic environment vectors of
ANI, the message sums of PhysNet, the symmetry functions of HDNNP, the local
message sums of SpookyNet, the attention messages of BAMBOO, every energy term
of the OPLS, AMBER, CHARMM and DREIDING force fields, the terms and the charge
solve of ReaxFF and ReaxFF-nn, the three-body term of D3 and D4 and the
large-system EEQ solve of D4 (where we use the Triton kernels, shipped with CUDA
builds of PyTorch), and the reciprocal sum of the LES long-range term (for
which, we use plain PyTorch). On a CPU, D3, D4 and the symmetry functions of
HDNNP run on C++ kernels (see :ref:`howto-fast-paths-cpu`).

Install
=======
At the moment (v0.23.0), only the MACE and NequIP kernels need an extra
dependency package installed to be activated. The installation requires a CUDA
12.6 or newer PyTorch build. You can install the cuEquivariance packages using
pip:

.. code-block:: bash

   pip install cuequivariance==0.6.1 cuequivariance-torch==0.6.1 cuequivariance-ops-torch-cu12==0.6.1

The three packages must come from one xnn release. Without them, or on a CPU,
MACE and NequIP fall back to their reference implementation.

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
- Equiformer: :data:`xnn.trfnn.fast.equiformer.AUTO_POLICY`,
  :data:`xnn.trfnn.fast.equiformer.AUTO_POLICY_TRAINING` (maximum edge counts)
- EquiformerV2: :data:`xnn.trfnn.fast.equiformer_v2.AUTO_POLICY`,
  :data:`xnn.trfnn.fast.equiformer_v2.AUTO_POLICY_TRAINING` (maximum edge
  counts: the compiled blocks are faster on small systems only)
- SE(3) steerable CNN and 3D CNN: :data:`xnn.cnn.fast.AUTO_POLICY`,
  :data:`xnn.cnn.fast.AUTO_POLICY_TRAINING`
- Spherical CNN: :data:`xnn.cnn.fast.spherical.AUTO_POLICY`,
  :data:`xnn.cnn.fast.spherical.AUTO_POLICY_TRAINING`
- ANI: :data:`xnn.dnn.fast.aev.AUTO_POLICY`,
  :data:`xnn.dnn.fast.aev.AUTO_POLICY_TRAINING`
- PhysNet: :data:`xnn.dnn.fast.physnet.AUTO_POLICY`,
  :data:`xnn.dnn.fast.physnet.AUTO_POLICY_TRAINING`
- HDNNP: :data:`xnn.dnn.fast.acsf.AUTO_POLICY`,
  :data:`xnn.dnn.fast.acsf.AUTO_POLICY_TRAINING`
- SpookyNet: :data:`xnn.hybnn.fast.spookynet.AUTO_POLICY`,
  :data:`xnn.hybnn.fast.spookynet.AUTO_POLICY_TRAINING`
- BAMBOO: :data:`xnn.hybnn.fast.bamboo.AUTO_POLICY`,
  :data:`xnn.hybnn.fast.bamboo.AUTO_POLICY_TRAINING`
- LES: :data:`xnn.common.models.les.AUTO_POLICY`

The dispersion, PaiNN and TensorNet kernels, and those of the OPLS, AMBER,
CHARMM and DREIDING force fields, are faster at every system size, so
``"auto"`` always selects them on a GPU. So are the AIMNet2 kernels, except
for training on small systems on an A30 or an L40S. The ANI kernels are faster
at every size in float32. In float64 they win at every size on an A100 and on
small systems on an A30 or a V100. They are not taken on a T4 or an L40S, whose
float64 throughput is low. The PhysNet kernels win on large systems, from about
20k to 300k edges depending on the GPU, and the BAMBOO kernels from about 5k to
60k. The HDNNP kernels are faster at every size, for evaluation and training. ``"auto"`` keeps the
reference for SpookyNet, whose fused local sums are not faster. The thresholds may change in
any release as we rerun our benchmarks, so report the xnn version with results
obtained with ``"auto"``.

.. _howto-fast-paths-cpu:

On a CPU
========
D3, D4 and the symmetry functions of HDNNP also have a fast path on CPUs: C++
and OpenMP kernels that loop over the atoms, their neighbors and neighbor pairs,
as the Fortran codes do. D3 runs entirely in them. D4 runs its polarizabilities
and its two- and three-body sums in them, and its charges in PyTorch. They take
a fraction of the time of simple-dftd3, dftd4 and RuNNer from about 80 atoms on.
The kernels are built on first use, which needs a C++ compiler with OpenMP and
``ninja`` (``pip install "xnns[cpu-kernels]"``):

.. code-block:: python

   model.set_use_fast(True)     # the C++ kernels on a CPU, where they build

- **First call.** The build takes about a minute, once per machine. ``"auto"``
  takes the kernels at every size where they build.
- **Fallback.** Without the build, D3 and D4 compile their three-body term with
  ``torch.compile`` (10 to 20 seconds on the first call; ``"auto"`` takes it above
  5000 edges, about 75 atoms). HDNNP function sets the kernels do not cover use
  ``torch.compile`` only with ``use_fast=True``. Compiled code keeps the OpenMP
  thread count of the process that compiled it, so give runs with different
  thread counts their own ``TORCHINDUCTOR_CACHE_DIR``. Where neither works, the
  reference path runs.
- **Evaluation only.** Covers the energies, forces and stress. A training step
  runs the reference path.

AMOEBA and HIPPO run their GPU kernels on a CPU as well. With
``use_fast=True``, the pair, mutual-field, solver and mesh kernels become C++
loops on OpenMP threads (:mod:`xnn.common.cpu_kernels`). They give the
reference's energies, forces and stress. As on a GPU, a training step takes the
pair kernels and runs the mesh and the solver eagerly.

- **First call.** The system C++ compiler builds the kernels on first use, which
  takes about a minute. They are cached in ``~/.cache/xnn/cpu_kernels``, or in
  ``XNN_CPU_KERNELS_DIR`` when it is set.
- **Choice.** ``"auto"`` takes the kernels on a CPU from about 50 000 pairs
  (about 1500 atoms of water), in evaluation and in training. Below that, it
  keeps the reference. ``use_fast=True`` takes the kernels at every size.

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
  adapters included. The three-body kernels of D3 and D4 are twice
  differentiable on a GPU, stress included, so force training runs through
  them. With trainable damping parameters (``s9``, ``a1``, ``a2``), the
  reference three-body term runs. The kernels of the
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
GPU's arithmetic. For DimeNet, DimeNet++, GemNet and PhysNet, the trainer can
record the whole step once as a CUDA graph and replay it for every batch:

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

Captured evaluations
====================
An evaluation of a small structure is bound by kernel launches in the same
way. For DimeNet, DimeNet++, GemNet and PhysNet, the ASE calculator can record
the evaluation once as a CUDA graph and replay it at every step:

.. code-block:: python

   from xnn.common.deploy import XNNCalculator

   atoms.calc = XNNCalculator(model, cutoff=model.cutoff, device="cuda", cuda_graph=True)

Each structure is padded to fixed atom and edge counts with one inert
structure, with 5 % headroom. A structure that outgrows the counts is recorded
again. The replay runs the reference implementation, so the results equal
those of ordinary evaluations up to round-off.
:class:`~xnn.common.deploy.GraphedEvaluation` does the same for any
evaluation of a graph.

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

On a CPU, pass a model on the CPU. The stepper then runs the same kernels as C++
loops over the CPU threads (:mod:`xnn.common.cpu_kernels`), so the machine needs a
C++ compiler. The first call compiles each kernel and keeps it in
``~/.cache/xnn/cpu_kernels`` (or ``XNN_CPU_KERNELS_DIR``). Later runs reuse it.
``stepper(struct, energy=False)`` returns the forces alone. Use it between the steps
that report the energy: the kernels then skip the energy sums.

Molecular dynamics with ReaxFF
==============================
:class:`~xnn.ffnn.models.ReaxFFStepper` does the same for a classical ReaxFF
field in the LAMMPS conventions. At a list rebuild it keeps the candidates of
every term: the pairs, the bonds that can reach the bond-order cutoff within
the skin, and their angles, torsions and hydrogen bonds. Each step flags the
candidates that take part, solves for the charges from an extrapolation of the
previous steps' charges, and replays the kernels as CUDA graphs:

.. code-block:: python

   from xnn.ffnn.models import ReaxFFStepper

   stepper = ReaxFFStepper(model.to("cuda").eval(), tol=1e-6)
   out = stepper({"pos": pos, "atomic_numbers": z, "cell": cell, "pbc": pbc})
   out["energy"], out["forces"], out["charges"]

``tol`` is the relative residual of the charge solve. LAMMPS's
``fix qeq/reaxff`` commonly takes 1e-6. The default is the model's tighter
one. The numbers are those of the model's evaluation, with the charges
converged to ``tol``. On a CPU it runs as :class:`~xnn.ffnn.models.ClassicalStepper`
does, without graphs.

The ``notebooks/parity_checks/`` notebooks compare every fast paths with their
reference implementations on real systems.
