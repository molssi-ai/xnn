.. _fidelity:

********************************
Model Fidelity to Upstream Codes
********************************

The literature models in xnn are not "inspired-by" re-implementations. Each
one reproduces its reference code, or the paper when the reference is a
manuscript, and is verified block by block and end to end with transplanted
weights. The reference packages are used only as external oracles in the
tests and the ``examples/fidelity_checks`` notebooks; at run time NequIP,
MACE and Allegro need ``e3nn`` and nothing else, and every other model is
plain PyTorch.

The summary per model, with the precision reached under identical weights:

MACE
====
Reproduces `ACEsuit/mace <https://github.com/ACEsuit/mace>`_: the
interaction blocks, the learned symmetric contraction with a bit-identical
Clebsch-Gordan basis, ZBL pair repulsion, the ScaleShift energy expression,
the Agnesi and Soft distance transforms and the density-normalized blocks
of the newer foundation generations. The message-passing depth is free (any
T from 0). Agreement about 1e-16. ``from_pretrained`` converts 16 of the 17
released MACE-MP / MACE-OFF checkpoints plus every head of ``mace-mh-0``
with energy parity about 1e-15 eV/atom and forces about 1e-13 eV/Å;
``mace-mh-1`` (a next-generation architecture) is rejected with an explicit
error.

NequIP
======
Reproduces the ``EnergyModel`` of `mir-group/nequip
<https://github.com/mir-group/nequip>`_ with upstream parameter names, so
state dicts transplant directly: irreps pruning, the gated nonlinearity, the
trainable Bessel basis with its :math:`2/r_{\max}` prefactor, the
:math:`1/\sqrt{\langle n \rangle}` message normalization and the per-species
scale and shift. Agreement about 1e-16 in energies, forces and stress.
TorchScript export uses a scriptable, bit-exact stand-in for e3nn's ``Gate``.

Allegro
=======
Reproduces `mir-group/allegro <https://github.com/mir-group/allegro>`_
v0.3.0 (``uuulin`` mode): the two-body embedding, the weightless Wigner-3j
tensor products, the strided channel-mixing linears in the upstream weight
layout, the latent resnet and the radial conventions. Agreement about 1e-15.

CACE
====
Reproduces `BingqingCheng/cace <https://github.com/BingqingCheng/cace>`_:
the Cartesian monomial basis with identical feature ordering, the element
embedding, the trainable radial coupling, all three message-passing
mechanisms and the readout. The per-species reference energy lives in the
model (upstream subtracts it from the labels). Agreement about 1e-16,
molecular and periodic.

AIMNet2
=======
Written from the paper (Anstine, Zubatyuk and Isayev 2025) with the
conventions of `isayevlab/aimnetcentral
<https://github.com/isayevlab/aimnetcentral>`_ where the paper leaves them
open, so the published checkpoints transplant weight for weight: the radial
basis and shell embeddings, the pass layout with neural charge
equilibration, the float64 energy shifts, the short-range Coulomb
subtraction and the all-pairs, damped shifted-force, Ewald and particle-mesh
Ewald sums, and the two-channel open-shell variant. Two deliberate
differences stay within the target accuracy: xnn keeps the real-space
cutoff at ``lr_cutoff`` and balances the splitting parameter to it, and its
PME uses the exact Euler-spline coefficients. The published models are
served with the two-body D3(BJ) term of their reference calculator. Float32
round-off against the ``aimnet`` package for all six families; the lattice
sums reproduce the rock-salt Madelung constant to 1e-7.

PhysNet
=======
A pure-PyTorch translation of the TensorFlow `MMunibas/PhysNet
<https://github.com/MMunibas/PhysNet>`_ (Unke and Meuwly 2019): the radial
basis, attention masks, residual blocks, the per-module energy and charge
heads, charge correction, shielded electrostatics and an independently
written D3(BJ) module with upstream's reference tables. The shifted
softplus is evaluated exactly. Agreement about 1e-15 in energies, forces,
charges and the non-hierarchicality penalty. Dropout is not implemented.

ANI
===
Written from the paper (Smith, Isayev and Roitberg 2017) and verified
against `aiqm/torchani <https://github.com/aiqm/torchani>`_: the atomic
environment vector in torchani's parameter and species-pair ordering, with
the NeuroChem conventions it follows (the 0.25 radial prefactor, the 0.95
cosine scaling; both configurable), per-element networks and self energies,
and the ANI-1, ANI-1x, ANI-1ccx and ANI-2x presets. The AEV matches to
about 1e-16; torchani's pretrained ANI-1x, ANI-1ccx and ANI-2x weights
reproduce its energies to about 1e-8 Ha, for one network and the full
ensemble.

SchNet
======
A clean-room implementation of the manuscripts (Schütt *et al.* 2017 and
the DTNN predecessor); no code from schnetpack is used or compared against.
The reference is an independent NumPy implementation of the equations:
embedding, Gaussian basis on the paper's grid, continuous-filter
convolution, the interaction block, the exact shifted softplus, the
zero-initialized readout and the DTNN energy standardization. One
off-by-default deviation, ``cutoff_fn="cosine"``, adds a smooth envelope for
finite-cutoff use. Agreement about 1e-15, forces against finite differences
about 1e-10, exact TorchScript parity.

BAMBOO
======
Written from scratch against `bytedance/bamboo
<https://github.com/bytedance/bamboo>`_ (Gong *et al.* 2024): the edge
attention, the scalar/vector layers, the exponential-normal basis, the
charge-equilibrium electrostatics with their damped Coulomb sum, and the
native kcal/mol units. Every layer, the charges, the dipole and the
component energies match to about 1e-15. The one deliberate difference is
the force: xnn returns the conservative ``-dE/dr``, equal to upstream's
``forces + qeq_force`` to about 1e-14, where upstream reports the first
term and regularizes the second toward zero during training.

Latent Ewald Summation
======================
An independent implementation of the reference ``cace.modules.EwaldPotential``
(Cheng 2025): the reciprocal sum with hemisphere symmetry, tinfoil boundary
conditions, triclinic cells, optional self-interaction removal, the
:math:`1/r^6` kernel, the real-space sum for molecules and the reference
latent-charge head. ``LatentEwald(CACE)`` reproduces the upstream CACE-LR
composition to float32 round-off; each kernel matches to about 1e-16 in
float64. Three robustness deviations: the k-grid follows the input dtype,
ties at the k-space cutoff resolve consistently, and the self-interaction is
subtracted once rather than once per channel.

DFT-D4 and DFT-D3
=================
Independent implementations verified against `dftd4
<https://github.com/dftd4/dftd4>`_ 4.2.0 and `simple-dftd3
<https://github.com/dftd3/simple-dftd3>`_ 1.6.0 for molecules, ions and
crystals: coordination numbers, the EEQ charge model with its Ewald sum
(D4), reference polarizabilities and Casimir-Polder C6 integration, the
BJ-damped two-body and ATM three-body terms, the four D3 damping functions
and the upstream cutoffs. Energies agree to about 1e-16 hartree, gradients
and virials to about 1e-17. The reference data are extracted from the
upstream sources by ``tools/build_d{3,4}_reference.py``. D4's matrix-free
``large`` regime differs from ``dftd4`` by that code's own Ewald tolerance,
about 1e-8 in the charges; isolated atoms differ by about 1e-9 e because xnn
evaluates the coordination-number cap exactly. D3 ships both the 2024 and
Grimme's 2010 reference systems (identical up to Z = 86); PhysNet and
BAMBOO use the 2010 set, so their tables are those of their upstream codes.

ReaxFF / ReaxFF-nn
==================
A faithful implementation of the published equations (van Duin *et al.*
2001, Nielson *et al.* 2005, Senftle *et al.* 2016; ReaxFF-nn of Guo *et al.*
2020 and Xue *et al.* 2021), with the conventions needed to evaluate
published libraries. Cross-checked against LAMMPS ``pair_style reaxff`` on
the C/H/O combustion field: nonbonded terms to about 1e-6 eV, bond, angle
and total energies to about 0.1 percent (the two codes truncate bond lists
differently), forces within a fraction of a percent. SEAMM's ``.frc``
translation of that field rounds eight bond-energy parameters to three
decimals, about 3e-5 eV on a small molecule.

**Licensing note.** The authors' reference implementation of ReaxFF-nn is
AGPL-licensed, which is incompatible with distributing any derived
verification artifact alongside this MIT-licensed code. During development
the implementation was checked term by term against it (agreement at its
float32 working precision, classical and neural), but no notebook, test or
code that depends on it is distributed and none of its code was copied. The
distributed verification is self-contained: ``tests/test_reaxff.py``
recomputes every energy term from the papers' equations.

Conventions: valence angles and torsions use the composed edge geometry,
nonbonded terms a true periodic neighbor list under the 7th-order taper,
the torsion angle enters through Chebyshev cosine identities (no ``arccos``,
whose derivative diverges for planar torsions), and a few rational
exponentials are evaluated in overflow-safe form so float32 force training
stays finite.

OPLS / L-OPLS
=============
A clean-room implementation of the published functional form (Jorgensen,
Maxwell and Tirado-Rives 1996), verified two ways: parity with OpenMM on
randomized conformations of butane, ethanol and ethylene (energies to about
1e-7 kJ/mol, forces to about 1e-6 kJ/mol/nm), and the papers' own numbers
(Table 1 of the 1996 paper to 0.01 kcal/mol with the ``"oplsaa-1996"``
library, the dual-form torsion rows of the L-OPLS paper exactly).
``"oplsaa"`` is SEAMM's distribution, whose alkane and alcohol torsions are
later revisions; ``"oplsaa-1996"`` restores the paper's values. Atom types
come from the file's SMARTS templates. Parameters use the thermochemical
calorie and the CODATA Coulomb constant, matching the kJ-based ecosystem
OPLS is distributed in.

DREIDING / DREIDING-X6
======================
A clean-room implementation of the published functional form (Mayo, Olafson
and Goddard 1990), with the generators from SEAMM's ``dreiding.frc`` and the
rule constants from the paper. Verified three ways: the published Tables I,
II, III and V entry by entry; parity with LAMMPS's own DREIDING styles on
nine molecules covering every rule branch, term by term to about 1e-10
kcal/mol; and the paper's Table XI torsional barriers (mean difference about
0.01 kcal/mol) and Table XII entries from relaxed scans. Conventions: the
inversion of eq 28 is averaged over the three axis choices, 1,4 pairs count
in full, the Coulomb constant is the paper's 332.0637, and DREIDING
prescribes no charges (``charges="gasteiger"`` supplies the paper's
recommendation).

Why this matters
================
Results published with the reference codes can be reproduced, weights move
in either direction (:ref:`howto-transplant`), and the xnn implementations
serve as readable single-dependency references for how these architectures
work. The ``examples/fidelity_checks`` notebooks double as annotated tours
of each one.
