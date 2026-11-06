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
and the interaction modules of SpookyNet (compiled with ``torch.compile``), the
residual MLPs of SpookyNet in training, the
3D convolutions of the SE(3) steerable CNN and the 3D CNN (on cuDNN's
channels-last kernels), the Fourier transforms and spectral products of the
spherical CNN, the atomic environment vectors of ANI, the message sums of
PhysNet, the symmetry functions of HDNNP, the attention messages of BAMBOO,
every energy term of the OPLS, AMBER, CHARMM and DREIDING force fields, the terms and the charge
solve of ReaxFF and ReaxFF-nn, the pair terms, fields, induction solve and
mesh sums of AMOEBA and HIPPO, the three-body term of D3 and D4 and the
large-system EEQ solve of D4 (where we use the Triton kernels, shipped with CUDA
builds of PyTorch), and the reciprocal sum of the LES long-range term (for
which, we use plain PyTorch). On a CPU, the three-body term of D3 and D4 and the
symmetry functions of HDNNP run compiled (see :ref:`howto-fast-paths-cpu`).

Install
=======
At the moment (v0.23.0), only the MACE and NequIP kernels need an extra
dependency package installed to be activated. The installation requires a CUDA
12.6 or newer PyTorch build. You can install the cuEquivariance packages using
pip:

.. code-block:: bash

   pip install cuequivariance==0.6.1 cuequivariance-torch==0.6.1 cuequivariance-ops-torch-cu12==0.6.1

The three packages must come from one cuEquivariance release. Without them, or on a CPU,
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
- SpookyNet: :data:`xnn.hybnn.fast.spookynet.AUTO_POLICY` (maximum edge
  counts), :data:`xnn.hybnn.fast.spookynet.AUTO_POLICY_TRAINING` and, for
  captured steps, :data:`xnn.hybnn.fast.spookynet.AUTO_POLICY_CAPTURED`
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
20k to 300k edges depending on the GPU (at every size in float64 on a T4), and the BAMBOO kernels from about 5k to
60k. The HDNNP kernels are faster at every size, for evaluation and training,
except in float64 on an L40S or a T4. ``"auto"`` takes the
compiled block of SpookyNet for evaluations below about 30k edges (at every
size in float32 on a T4, and never in float64 on a T4 or an L40S). It trains
SpookyNet on its fused MLPs in captured steps (not in float64 on a T4 or an
L40S), and in ordinary steps from 16k to 150k edges depending
on the GPU and precision. The thresholds may change in any release as we rerun our benchmarks, so report the xnn version with results
obtained with ``"auto"``.

.. _howto-fast-paths-cpu:

On a CPU
========
D3, D4 and the symmetry functions of HDNNP also have a fast path on CPUs. It is
the same PyTorch computation compiled with ``torch.compile`` into C++ and OpenMP
loops. So, the code needs a C++ compiler on the machine:

.. code-block:: python

   model.set_use_fast(True)     # the compiled path on a CPU, where torch.compile works

- **First call.** The first evaluation compiles, which takes 10 to 60 seconds.
  Later calls of any system size reuse the compiled code.
- **Choice.** A standalone D3 or D4 term computes a molecule of up to 2048
  atoms, or a batch of molecules of up to 256 atoms, over all atom pairs and
  triangles in one compiled call. ``"auto"`` takes that at every size. For cells
  and larger systems, ``"auto"`` takes the compiled path above 5000 edges (about
  75 atoms). A 2G HDNNP takes its direct path (below) at every size.
- **Threads.** The compiled code keeps the OpenMP thread count of the process that
  compiled it. Give runs with different thread counts their own
  ``TORCHINDUCTOR_CACHE_DIR``. With exactly as many threads as allotted cores,
  the compiled code's OpenMP threads can stall against PyTorch's own. Allot a
  few more cores than threads, or set ``OMP_WAIT_POLICY=PASSIVE``.
- **Evaluation only.** Covers the energies, forces and stress. A training step
  runs the reference path. A standalone D3 or D4 term (no wrapped model), and a
  2G HDNNP, form their forces and stress from written-out derivatives instead
  of an autograd pass. Their outputs then carry no autograd graph.
- **Fallback.** Where ``torch.compile`` cannot produce CPU code, the reference
  path runs.

**Optional C++ kernels.** For D3, D4 and HDNNP, xnn also ships hand-written C++
and OpenMP kernels with the loops of the Fortran codes. They are off by default.
Turn them on with ``XNN_CPP_KERNELS=1`` in the environment or in Python:

.. code-block:: python

   from xnn.common.models import cpu_extension
   cpu_extension.enable()

They need ``ninja`` (``pip install "xnns[cpu-kernels]"``) and build in about a
minute on first use. ``"auto"`` then takes them at every size. They take a
fraction of the time of simple-dftd3, dftd4 and RuNNer.

AMOEBA and HIPPO also run their GPU kernels on a CPU, as C++ loops. The
results match the reference.

- **First call.** The kernels compile on first use, in about a minute. They are
  cached in ``~/.cache/xnn/cpu_kernels`` (or ``XNN_CPU_KERNELS_DIR``).
- **Choice.** ``"auto"`` uses them from about 50 000 atom pairs (about 1500
  water atoms). ``use_fast=True`` uses them at every size.
- **Training.** A training step uses the pair kernels. The mesh and the dipole
  solver run eagerly.

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
- **Training.** Every fast path with a Fast-Path (Training) mark in
  :ref:`model-support` is twice differentiable, so force training runs
  through it. The MACE, NequIP, LES, SchNet, PaiNN, TensorNet, AIMNet2, GemNet,
  DimeNet, Allegro and CACE fast paths also take LoRA adapters. The three-body kernels of D3 and D4 are twice
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
  evaluation for 25 to 60 % more time. When the per-edge weights of all edges would
  not fit in the free GPU memory, MACE evaluates its message pass in blocks of
  edges and repeats each block in the backward pass, so larger structures fit
  (:func:`~xnn.gnn.models.set_message_blocks` sets it per model). The SchNet, PaiNN and TensorNet kernels
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
  minute). Later evaluations of any system size reuse them. SpookyNet's
  compiled block takes about a minute the first time. The compiled blocks of
  Equiformer and EquiformerV2 also compile on their first evaluation.

Captured training steps
=======================
A training step on small molecules is bound by kernel launches, not by the
GPU's arithmetic. For DimeNet, DimeNet++, GemNet, PhysNet, SpookyNet and MACE, the
trainer can record the whole step once as a CUDA graph and replay it for every
batch:

.. code-block:: yaml

   trainer:
     cuda_graph: true

The data loader pads every batch to fixed counts with one inert structure
and builds the model's index work on the CPU: the triplets of DimeNet, the
quadruplets of GemNet, and the local edges and long-range pairs of SpookyNet.
The step runs the reference implementation (SpookyNet's runs its fused MLPs).
SpookyNet steps are 3 to 5 times faster this way on small and medium systems. They do not support the Ewald
sum. The results equal those of ordinary
steps up to round-off. Where ``use_fast`` takes the fused kernels for the
padded batch size, the first steps time ordinary steps on the kernels against
captured ones, and training goes on with the faster. It needs one GPU and a
single head, and drops the last, smaller batch of each epoch. Validation runs
ordinary steps.

Captured evaluations
====================
An evaluation of a small structure is bound by kernel launches in the same
way. For DimeNet, DimeNet++, GemNet, PhysNet, SpookyNet and MACE, the ASE calculator
can record the evaluation once as a CUDA graph and replay it at every step:

.. code-block:: python

   from xnn.common.deploy import XNNCalculator

   atoms.calc = XNNCalculator(model, cutoff=model.cutoff, device="cuda", cuda_graph=True)

Each structure is padded to fixed atom and edge counts with one inert
structure, with 5 % headroom. A structure that outgrows the counts is recorded
again. The replay runs the same code as an ordinary evaluation, so the results equal
those of ordinary evaluations up to round-off. After each recording, a few
replays are timed against ordinary evaluations and the faster is kept: large
structures, where the padding costs more than the launches, run ordinarily. SpookyNet's replay also uses
its compiled block where ``"auto"`` takes it.
:class:`~xnn.common.deploy.GraphedEvaluation` does the same for any
evaluation of a graph.

Semiempirical methods
=====================
PM3, PM6, PM6-ORG and PM7 have no separate kernels. On a GPU the model replays
CUDA graphs of its own launches instead. From the third evaluation of a batch
composition (the same atoms, charges and spins), it replays the SCF iterations
and the integral build with its derivatives:

.. code-block:: python

   from xnn.sqmnn.models import PM7

   model = PM7().to("cuda").eval()      # use_fast="auto": graphs from the third evaluation
   model.set_use_fast(False)            # always launched

The eigenvalue problem runs between the replays. The results equal those of
launched evaluations up to round-off. In a molecular dynamics run or a geometry
optimization on a GPU, each SCF starts from the previous step's orbitals and
rotates them from its first iteration. That saves one diagonalization per step. Gradients taken with ``create_graph``
(Hessians, force training) and their gradients replay graphs of their own,
recorded at the next evaluation after the first such gradient. With trainable
parameters the build is replayed from the parameters and the density response
from graphs of its iterations, so a force-loss training step runs on the graphs
too. Each recording takes a few tenths of a second. A
composition that only recurs after many others, as in a training set cycled
through in epochs, is never recorded.

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
