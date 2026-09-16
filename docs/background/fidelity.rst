.. _fidelity:

********************************
Model Fidelity to Upstream Codes
********************************

The literature models in xnns (MACE, NequIP, Allegro, CACE, PhysNet, BAMBOO,
ANI, SchNet) are not approximations or "inspired-by" re-implementations: they
are faithful, self-contained xnns-native reproductions of the reference codes,
verified numerically block by block and end to end. Models based on
spherical-harmonic features will only need ``e3nn``: no additional packages such
as  ``mace-torch``, ``nequip``, ``cuequivariance``, or ``opt_einsum_fx`` is
required at the runtime. Furthermore, CACE, PhysNet, BAMBOO, ANI, and SchNet
need no extra dependency at all. Moreover, PhysNet's original implementation in
TensorFlow has been ported to xnns's ecosystem in pure PyTorch. The BAMBOO model
uses Cartesian vector channels, so, it does not need the e3nn package. The ANI
model is written in pure PyTorch and use symmetry functions. The SchNet model is
also written in plain PyTorch.

The equivariance itself is verified in the test suite (rotate the inputs
$\rightarrow$ check to see whether the energy remains invariant and the forces
co-rotate, numerically, to the threshold of ~1e-7; see ``tests/test_gnn.py`` and
``tests/test_mace.py``).

MACE
====
MACE in xnns reproduces the upstream `ACEsuit/mace
<https://github.com/ACEsuit/mace>`_ (``mace-torch``). Notably, the xnns
implementation reproduces:

- the real ``RealAgnostic(Residual)InteractionBlock`` and the paper's
  *learned symmetric contraction* over Clebsch-Gordan paths (``correlation``
  order);
- the CG coupling basis (``U_matrix_real``), which is bit-identical to
  ``mace-torch``, and the contraction reproduces it to ~1e-16 given the same
  weights;
- the ZBL ``pair_repulsion``, which makes the message-passing depth fully
  flexible (contrary to the original code, where the ``num_interactions``, T, is
  fixed to 2, our implementation allows for T to be set to 0, ..., N).

The notebooks in ``examples/fidelity_checks`` verify the xnns's implementation
and compare it against the upstream version block by block. The notebooks in
``examples/gnn/mace/`` offer end to end examples which compare the xnns's MACE
implementation against ``mace-torch`` on argon molecular dynamics data.

NequIP
======
The NequIP model in xnns offers a faithful re-implementation of the upstream
``EnergyModel`` of `mir-group/nequip <https://github.com/mir-group/nequip>`_ and
involves:

- the real ``InteractionBlock``, with upstream parameter names, so state
  dicts transplant directly;
- per-layer ``tp_path_exists`` irreps pruning and the gated nonlinearity;
- NequIP's radial conventions: trainable Bessel with the :math:`2/r_{\max}`
  prefactor, :math:`1/\sqrt{\langle n_\text{neigh}\rangle}` message
  normalization, and the :math:`\mathbf{r}_j - \mathbf{r}_i` edge
  orientation;
- the per-species energy scale/shift.

Given the same weights it reproduces ``nequip`` to ~1e-16 in energies,
forces, and stress (``tests/test_nequip.py``). TorchScript export required a
scriptable, bit-exact stand-in for e3nn's ``Gate``
(``xnns.gnn.models.nequip._Gate``), which the e3nn 0.4.4 original cannot
provide on torch 2.x.

Allegro
=======
A faithful re-implementation of the original `mir-group/allegro
<https://github.com/mir-group/allegro>`_ (v0.3.0, the e3nn-era reference,
default ``uuulin`` mode):

- the two-body product type embedding and the per-channel weightless
  Wigner-3j tensor products with the embedded-environment density trick;
- the strided channel-mixing linears with the same flat weight layout, so
  state dicts transplant directly;
- the cumulative-softmax latent resnet;
- Allegro's radial conventions: trainable "normalized sinc" Bessel with the
  :math:`r_{\max}/\pi` prefactor,
  :math:`1/\sqrt{\langle n_\text{neigh}\rangle - 1}` environment and
  :math:`1/\sqrt{\langle n_\text{neigh}\rangle}` energy-sum normalization;
- the per-species scale/shift.

Given the same weights it reproduces ``allegro`` to ~1e-15 in energies and
forces (``tests/test_allegro.py``).

CACE
====
A faithful re-implementation of `BingqingCheng/cace
<https://github.com/BingqingCheng/cace>`_ (Cheng, *npj Comput Mater* 2024),
the Cartesian atomic cluster expansion, which needs no spherical
harmonics or e3nn at all:

- the Cartesian monomial angular basis
  (:class:`~xnns.gnn.featurizers.cartesian.CartesianAngularBasis`,
  evaluated with the same autograd-safe multiply recursion) and the exact
  multinomial symmetrization rules of upstream
  ``find_combo_vectors_nu{2,3,4}``, so B-feature ordering is identical;
- the tensor-product element-embedding edge type, the per-\ :math:`(l, c)`
  trainable radial channel coupling (upstream's per-\ :math:`l` weight list
  stacked into one einsum), and all three message-passing mechanisms
  (node memory ``M``, exponential-decay filter ``Ar``, recursive edge
  embedding ``Bchi``);
- CACE's radial conventions: trainable Bessel with the MACE
  :math:`\sqrt{2/r_{\max}}` prefactor, degree-6 polynomial cutoff,
  normalized :math:`\mathbf{r}_i - \mathbf{r}_j` edge vectors, and the
  :math:`1/\sqrt{\langle n_\text{neigh}\rangle}` message normalization;
- the linear + MLP readout on the concatenated per-layer B features; the
  per-species reference energy lives in the standard xnns ``atom_ref``
  (upstream subtracts it from the training labels instead).

Given the same weights it reproduces ``cace`` to ~1e-16 (relative) in
energies and forces, molecular and periodic, for any message-type subset
(``tests/test_cace.py``).

PhysNet
=======
A faithful **pure-PyTorch translation** of the original TensorFlow 1.x
implementation `MMunibas/PhysNet <https://github.com/MMunibas/PhysNet>`_
(Unke & Meuwly, JCTC 2019):

- the exponential-Gaussian radial basis with its quintic cutoff, the
  distance-based attention masks (zero-initialized ``k2f``), pre-activation
  residual blocks, gated feature updates, and the zero-initialized
  per-module ``(energy, charge)`` output heads, with per-element
  scale/shift tables of length 95 (elements are embedded directly by
  nuclear charge -- alchemical by construction);
- the exact charge-correction, the switched/shielded Coulomb term
  (``kehalf`` constant and the force-shifted long-range form included), and
  an independently written D3(BJ) dispersion module that reproduces the
  behavior of upstream's bundled Grimme D3 code exactly, with its reference
  tables (shipped compressed in ``xnns/dnn/models/d3_tables.npz``) and
  softplus-learnable coefficients;
- the shifted-softplus is evaluated in its exact form
  ``max(x, 0) + log1p(exp(-|x|))`` -- PyTorch's ``F.softplus`` goes linear
  above its threshold and would cost ~1e-9.

Given the same weights it reproduces the original TF graph to ~1e-15 in
energies, forces, corrected charges, and the non-hierarchicality penalty
(``tests/test_physnet.py``; the parity test needs TensorFlow and a clone of
the upstream repo via ``PHYSNET_UPSTREAM_PATH``, and the block-by-block
notebook documents three dtype-only harness patches that let the float32-era
TF graph run in float64). Dropout (upstream ``keep_prob``, default off) is
not implemented.

ANI
===
A faithful, from-scratch re-implementation of the ANI method (Smith, Isayev &
Roitberg, *Chem. Sci.* 2017), verified against `aiqm/torchani
<https://github.com/aiqm/torchani>`_ (pure-PyTorch symmetry functions, no
extra dependency):

- the Atomic Environment Vector
  (:class:`~xnns.dnn.featurizers.AEV`): element-resolved radial (Behler
  :math:`G^2`) and angular symmetry functions, bucketed by neighbour species
  and by the unordered neighbour-species pair in torchani's upper-triangular
  order, with the parameter grids laid out in torchani's ``(EtaR, ShfR)`` and
  ``(EtaA, Zeta, ShfA, ShfZ)`` orderings;
- the ANI / NeuroChem conventions torchani follows over the paper as written:
  the ``0.25`` radial prefactor and the ``0.95`` scaling of :math:`\cos\theta`
  inside ``acos`` (both configurable; set to ``1.0`` for the literal
  Behler-Parrinello form);
- per-element neural networks with per-element architectures (:meth:`ANI.ani1x`
  uses torchani's H ``160:128:96``, C ``144:112:96``, N/O ``128:112:96`` widths
  and the ``CELU`` activation), and per-element self atomic energies;
- the published parameterisations as presets: :meth:`ANI.ani1` (the
  paper's 768-length AEV, 4.6/3.1 Å cutoffs, ``768:128:128:64:1`` networks with
  a Gaussian activation), :meth:`ANI.ani1x` (the 384-length ANI-1x grid,
  5.2/3.5 Å cutoffs), :meth:`ANI.ani1ccx` (Smith *et al.* 2019: identical
  architecture to ANI-1x, transfer-learned to CCSD(T)*/CBS coupled-cluster
  data; the preset reuses :meth:`ANI.ani1x` and swaps in the coupled-cluster
  self atomic energies), and :meth:`ANI.ani2x` (Devereux *et al.* 2020: the
  seven-element model, adding S, F, and Cl to give a 1008-length AEV with
  5.1/3.5 Å cutoffs and wider per-element networks).

The AEV matches ``torchani.AEVComputer`` element-for-element to ~1e-16 (for the
ANI-1, ANI-1x, and ANI-2x grids), and transplanting torchani's **pretrained**
ANI-1x, ANI-1ccx, or ANI-2x weights reproduces their energies to ~1e-8 Ha and
forces to ~2e-7 Ha/Å, for a single network and the full 8-model ensemble
(``examples/fidelity_checks/ani_verification.ipynb``, ``tests/test_ani.py``; the
parity tests need ``torchani``, the ``[ani]`` extra). All four training sets
are in the hub: the original ANI-1 data as ``load_dataset("ani1")``, the
active-learning ANI-1x data (energies and forces) as ``load_dataset("ani1x")``,
its coupled-cluster subset (CCSD(T)*/CBS energies, shared release file) as
``load_dataset("ani1ccx")``, and the seven-element ANI-2x data (wB97X energies
and forces for H/C/N/O/S/F/Cl) as ``load_dataset("ani2x")``.

SchNet
======
A faithful implementation of the **manuscripts** — Schütt *et al.*, NIPS 30
(2017), plus the DTNN predecessor (*Nat. Commun.* **8**, 13890, 2017) for
the conventions SchNet inherits. By design it is a *clean-room* build: no
code is taken from (or compared against) schnetpack; the verification
reference is an independent NumPy implementation of the papers' equations
run with the same weights:

- the atom-type embedding (eq. 3), the Gaussian radial basis
  :math:`e_k(r) = \exp(-\gamma (r - \mu_k)^2)` on the paper's grid
  (:math:`\gamma = 10` Å\ :sup:`-2`, centers every 0.1 Å — the shared
  :class:`~xnns.common.featurizers.GaussianRBF` with an explicit ``gamma``);
- the continuous-filter convolution (eq. 2) with the two-dense-layer
  shifted-softplus filter-generating network, and the Fig. 2 interaction
  block (*atom-wise → cfconv → atom-wise → ssp → atom-wise*, residual,
  no weight sharing across the :math:`T` blocks);
- the exact shifted softplus :math:`\ln(0.5 e^x + 0.5)` (shared with
  PhysNet in :func:`~xnns.common.models.ops.shifted_softplus`), the
  atom-wise :math:`F \to F/2 \to 1` readout with a zero-initialized head,
  and the DTNN per-atom energy standardization
  :math:`E_i = E_\sigma \hat E_i + E_\mu`;
- one documented, off-by-default deviation: ``cutoff_fn="cosine"``
  multiplies the filters by a smooth envelope for finite-cutoff condensed-
  phase use (the paper trains cutoff-free; its RBF grid simply ends at
  30 Å).

Given the same weights it reproduces the equation-by-equation reference to
~1e-15 in energies, block by block and end to end, with autograd forces
matching finite differences to ~1e-10 and exact TorchScript parity
(``tests/test_schnet.py``,
``examples/fidelity_checks/schnet_verification.ipynb``).

BAMBOO
======
A faithful, from-scratch re-implementation of `bytedance/bamboo
<https://github.com/bytedance/bamboo>`_ (Gong *et al.* 2024), the graph
equivariant transformer with a physics energy split, built on the xnns
abstractions with no upstream code vendored:

- the multi-head QKV edge attention (the reusable
  :class:`~xnns.transformer.attention.EdgeMultiheadAttention`), the
  scalar/vector GET layers with their inner-product coupling (first / middle /
  last variants), and the exponential-normal radial basis
  (:class:`~xnns.transformer.featurizers.ExpNormalSmearing`);
- the charge-equilibrium electrostatics: per-atom electronegativity/hardness
  energies from the initial embedding, charges squashed by ``tanh`` and
  conserved to the total charge, and the damped all-pairs Coulomb sum (whose
  short-range softplus damping differentiates exactly to the paper's sigmoid
  force damping);
- BAMBOO's native kcal/mol / Å units and its ``ele_factor`` constant; elements
  are embedded directly by atomic number.

Given the same weights, and driven from the same geometry, it reproduces the
original model to machine precision: every GET layer, the partial charges,
the dipole, and the component energies match to ~1e-15, block by block
(``examples/fidelity_checks/bamboo_verification.ipynb``,
``tests/test_bamboo.py`` with a clone of the upstream repo via
``BAMBOO_UPSTREAM_PATH``). The one deliberate difference is the force
convention: xnns returns the full conservative ``-dE/dr`` through
:class:`~xnns.common.models.outputs.ForceStressOutput`, which equals the
upstream ``forces + qeq_force`` to ~1e-14; upstream reports only ``forces``
(``nn + coul`` with charges held fixed) and regularises the
charge-equilibrium residual ``qeq_force`` toward zero during training
(Supplementary Theorem A.2). The optional D3(CSO) dispersion (off by default,
as in the paper's DFT training) reuses xnns's standard Grimme-D3 reference
tables with the CSO damping and is therefore not on the machine-precision
path.

Latent Ewald Summation (LES)
============================
The long-range add-on :class:`~xnns.common.models.les.LatentEwald` is an
independently written implementation of the algorithm of the reference
``cace.modules.EwaldPotential`` (Cheng, *npj Comput Mater* 2025) and the
``cace-lr-fit`` training-script conventions, verified against them:

- the reciprocal-space sum with hemisphere symmetry factors, tinfoil
  boundary conditions (no ``k = 0`` term), triclinic cells, the optional
  Gaussian self-interaction removal, and the ``1/r^6`` dispersion kernel;
- the ``erf``-converged real-space direct sum for non-periodic structures;
- the reference latent-charge head (bias-free ``[24, 12]`` MLP plus a
  parallel bias-free linear layer) on the model's invariant features.

Given the same weights, ``LatentEwald(CACE)`` reproduces the upstream
CACE-LR composition (representation + two ``Atomwise`` heads +
``EwaldPotential`` + ``FeatureAdd``) to float32 round-off end to end, and
each kernel matches upstream to ~1e-16 in float64 (``tests/test_les.py``).
Documented deviations (floating-point robustness, not physics): the k-grid
follows the input dtype (upstream hard-casts it to float32 and cannot run in
float64), exact ties at the ``|k| = k_c`` shell resolve consistently so the
energy is exactly rotation-invariant, and the self-interaction term is
subtracted once (upstream subtracts it once per ``q`` channel).

ReaxFF / ReaxFF-nn
==================
A faithful implementation of the **published equations**: the classical
reactive force field of van Duin *et al.* (*J. Phys. Chem. A* 105, 9396,
2001; the transition-metal extension of Nielson *et al.* 2005 and the
standard form reviewed by Senftle *et al.* 2016) and the machine-learned
ReaxFF-nn variant (Guo *et al.*, *Comput. Mater. Sci.* 172, 109393, 2020;
Xue *et al.*, *PCCP* 23, 19457, 2021), including the conventions required to
evaluate published parameter libraries (SEAMM ``.frc`` files and ReaxFF-nn
JSON):
kcal/mol units and their per-term application rules, off-diagonal
combination rules, torsion wildcards, hydrogen-bond defaults, and the
bond-order switching behavior the libraries were trained under.

The classical evaluation was additionally cross-checked against standalone
LAMMPS ``pair_style reaxff`` on the published C/H/O combustion field
(Chenoweth *et al.* 2008; xnns ships SEAMM's ``.frc`` translation of it,
which rounds eight bond-energy parameters to three decimals -- ``De`` values
differ from the LAMMPS ``ffield.reax.cho`` by up to 4e-4 kcal/mol, about
3e-5 eV on a small molecule, and nothing else differs):
per-term energies agree at the level set by the two codes' different
bond-list truncation conventions (nonbonded van der Waals and Coulomb terms
to ~1e-6 eV; bond, angle, and total energies to ~0.1 percent), and forces
match within a fraction of a percent. LAMMPS is used there only as an
external oracle; nothing in xnns depends on it.

**Licensing note.** The authors' reference implementation of ReaxFF-nn is
distributed under the AGPL, which is incompatible with redistributing any
derived verification artifact alongside this MIT-licensed code base. During
development the xnns implementation was checked term by term against that
publicly available implementation (agreement at its float32 working
precision on molecular and periodic CHNO systems, classical and neural
variants alike), but **no fidelity notebook, test, or vendored code that
depends on it is distributed here, and none of its code was copied**. The
distributed verification is therefore self-contained, following the same
clean-room convention as SchNet: ``tests/test_reaxff.py`` recomputes every
energy term from the papers' equations (uncorrected and corrected bond
orders, bond energy, the analytic two-atom EEM solution and its
self/Coulomb energies, the shielded-Morse van der Waals dimer, the valence
angle of water via eqs 8a-8d, hydrogen bonds, valence/torsion enumeration by
brute force) and checks rotation/translation invariance, autograd forces
against finite differences, size extensivity, batching, charge-constraint
handling, trainability, and library round-trips.

Documented conventions: valence angles and torsions use the composed edge
geometry (well defined for any cell); nonbonded terms are evaluated on a
true periodic neighbor list under the standard 7th-order taper; the torsion
angle enters through Chebyshev cosine identities (no ``arccos`` in the
graph, whose derivative diverges for the planar torsions every conjugated
molecule has); and a handful of rational-exponential terms are evaluated in
overflow-safe sigmoid form (exact) or with an argument-capped ``exp``
(differing below 1e-17, inside saturating ratios only) so that float32
training with force losses stays finite.

OPLS / OPLS-AA / L-OPLS
=======================
A clean-room implementation of the **published functional form** (Jorgensen,
Maxwell & Tirado-Rives, *JACS* 118, 11225, 1996, eqs 1-4), with parameters
taken from the published tables rather than any upstream code. Fidelity is
established two independent ways:

* **Parity with OpenMM**: ``tests/test_opls.py`` and
  ``examples/fidelity_checks/opls_verification.ipynb`` assemble the same
  parameter library and topology into an OpenMM ``System`` (Coulomb via
  ``NonbondedForce``, Lennard-Jones via a ``CustomNonbondedForce`` with OPLS
  geometric mixing, exact 1,4 exceptions, Fourier torsions as phased
  periodic torsions) and compare on randomized conformations of butane,
  ethanol and ethylene: energies agree to ~1e-7 kJ/mol and forces to ~1e-6
  kJ/mol/nm — the precision at which the two codes' physical constants are
  written down. OpenMM is an optional test dependency; the suite skips the
  parity test without it.
* **The papers' own numbers**: relaxed dihedral-driver scans reproduce the
  OPLS-AA column of Table 1 of the 1996 paper to a few hundredths of a
  kcal/mol across ethane, propane, butane, methanol and ethanol
  (``examples/ffnn/opls/opls_conformational_energetics.ipynb``), and the
  Fourier/Ryckaert-Bellemans conversion reproduces the dual-form torsion
  rows of Table 2 of the L-OPLS paper (Siu *et al.*, *JCTC* 8, 1459, 2012)
  exactly. The hexane gauche-trans gap of the built-in ``"lopls"`` library
  matches the published refit (~2 kJ/mol vs OPLS-AA's ~5).

Documented conventions: ``"oplsaa"`` is SEAMM's OPLS-AA distribution, whose
alkane torsions are the late-1999 revision by the Jorgensen lab and whose
``H-C-O-H`` torsion (``V3`` = 0.352 kcal/mol) and ``C-C-C-O`` torsion differ
from the 1996 paper as well; ``"oplsaa-1996"`` restores the paper's values
(alkanes from Supporting Information Table 7, ``H-C-O-H`` ``V3`` = 0.45 and
``C-C-C-O`` as distributed with GROMACS), which are what Table 1 was computed
with -- with them every Table 1 entry is reproduced to 0.01 kcal/mol. Atom
types are assigned from the file's SMARTS templates, so the SEAMM type names
(``opls_80`` for an alkane CH3 carbon, not GROMACS' ``opls_135``) never need
to be known. Parameters use the thermochemical
calorie (4.184 kJ exactly) and the CODATA Coulomb constant, matching the
kJ-based ecosystem (BOSS/GROMACS/OpenMM) in which OPLS parameters are
distributed — ReaxFF keeps its own historical Fortran-era constants for
``ffield`` compatibility, and the two differ by ~8e-6 relative. Dihedral
angles enter through Chebyshev cosine identities (the OPLS Fourier terms
are even in the angle, so no ``arccos``/``atan2`` is needed); excluded
pairs are excluded in every periodic image, the standard MM convention.

Why this matters
================
Fidelity means results published with the reference codes can be reproduced,
weights can be transplanted in either direction (see
:ref:`howto-transplant`), and the xnns implementations can serve as readable,
single-dependency references for how these architectures actually work: the
``examples/fidelity_checks/<model>_verification.ipynb`` notebooks double as
annotated tours of each architecture.
