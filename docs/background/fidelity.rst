.. _fidelity:

********************************
Model Fidelity to Upstream Codes
********************************

The literature models in xnns (MACE, NequIP, Allegro, CACE, PhysNet, BAMBOO)
are not approximations or "inspired-by" re-implementations: they are faithful,
self-contained reproductions of the reference codes, verified numerically
block by block and end to end. The spherical-harmonic models need only
``e3nn`` — no ``mace-torch``, ``nequip``, ``cuequivariance``, or
``opt_einsum_fx`` at runtime; CACE, PhysNet, and BAMBOO need no extra
dependency at all (PhysNet's original is TensorFlow — the xnns version is pure
PyTorch; BAMBOO uses Cartesian vector channels, so it needs no e3nn).

Equivariance itself is verified in the test suite (rotate the inputs → the
energy is invariant and the forces co-rotate, to ~1e-7; see
``tests/test_gnn.py`` and ``tests/test_mace.py``).

MACE
====
A faithful re-implementation of `ACEsuit/mace <https://github.com/ACEsuit/mace>`_
(``mace-torch``):

- the real ``RealAgnostic(Residual)InteractionBlock`` and the paper's
  *learned symmetric contraction* over Clebsch–Gordan paths (``correlation``
  order);
- the CG coupling basis (``U_matrix_real``) is bit-identical to
  ``mace-torch``, and the contraction reproduces it to ~1e-16 given the same
  weights;
- adds ZBL ``pair_repulsion``, and makes the message-passing depth fully
  flexible (``num_interactions`` = T = 0..N, vs. upstream's fixed 2).

The notebooks in ``examples/gnn/mace/`` verify it block by block and end to
end against ``mace-torch`` on Argon MD data.

NequIP
======
A faithful re-implementation of the upstream ``EnergyModel`` of
`mir-group/nequip <https://github.com/mir-group/nequip>`_:

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
<https://github.com/BingqingCheng/cace>`_ (Cheng, *npj Comput Mater* 2024)
— the Cartesian atomic cluster expansion, which needs no spherical
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
  a statement-for-statement port of the bundled Grimme D3(BJ) module with
  its reference tables (shipped compressed in
  ``xnns/dnn/models/d3_tables.npz``) and softplus-learnable coefficients;
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

BAMBOO
======
A faithful, from-scratch re-implementation of `bytedance/bamboo
<https://github.com/bytedance/bamboo>`_ (Gong *et al.* 2024) — the graph
equivariant transformer with a physics energy split — built on the xnns
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
original model to machine precision — every GET layer, the partial charges,
the dipole, and the component energies match to ~1e-15, block by block
(``examples/fidelity_checks/bamboo_verification.ipynb``,
``tests/test_bamboo.py`` with a clone of the upstream repo via
``BAMBOO_UPSTREAM_PATH``). The one deliberate difference is the force
convention: xnns returns the full conservative ``-dE/dr`` through
:class:`~xnns.common.models.outputs.ForceStressOutput`, which equals the
upstream ``forces + qeq_force`` to ~1e-14 — upstream reports only ``forces``
(``nn + coul`` with charges held fixed) and regularises the
charge-equilibrium residual ``qeq_force`` toward zero during training
(Supplementary Theorem A.2). The optional D3(CSO) dispersion (off by default,
as in the paper's DFT training) reuses xnns's standard Grimme-D3 reference
tables with the CSO damping and is therefore not on the machine-precision
path.

Latent Ewald Summation (LES)
============================
The long-range add-on :class:`~xnns.common.models.les.LatentEwald` is a
faithful port of the reference ``cace.modules.EwaldPotential`` (Cheng,
*npj Comput Mater* 2025) and the ``cace-lr-fit`` training-script conventions:

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

Why this matters
================
Fidelity means results published with the reference codes can be reproduced,
weights can be transplanted in either direction (see
:ref:`howto-transplant`), and the xnns implementations can serve as readable,
single-dependency references for how these architectures actually work — the
``examples/fidelity_checks/<model>_verification.ipynb`` notebooks double as
annotated tours of each architecture.
