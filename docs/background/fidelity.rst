.. _fidelity:

********************************
Model Fidelity to Upstream Codes
********************************

Each model reproduces its reference code and its paper results when available,
and the code is verified block by block with transplanted weights from the
original models when accessible (``tests/``, ``examples/fidelity_checks/``). The
reference packages are used only as external oracles and references for
validating our implementation; at run time NequIP, MACE and Allegro need
``e3nn`` and every other model is plain PyTorch. Numerical agreement under
identical weights involves:

- **MACE** (`ACEsuit/mace <https://github.com/ACEsuit/mace>`_): about 1e-16,
  including ScaleShift, the Agnesi and Soft distance transforms and the
  density-normalized blocks. 16 of the 17 released MACE-MP / MACE-OFF
  checkpoints and every head of ``mace-mh-0`` convert with energy parity
  about 1e-15 eV/atom; ``mace-mh-1`` is rejected with an explicit error.
- **NequIP** (`mir-group/nequip <https://github.com/mir-group/nequip>`_):
  about 1e-16 in energies, forces and stress; parameter names match, so
  state dicts transplant directly.
- **Allegro** (`mir-group/allegro <https://github.com/mir-group/allegro>`_
  v0.3.0): about 1e-15, with the upstream weight layout.
- **CACE** (`BingqingCheng/cace <https://github.com/BingqingCheng/cace>`_):
  about 1e-16, molecular and periodic, for every message-passing mechanism.
- **AIMNet2** (`isayevlab/aimnetcentral
  <https://github.com/isayevlab/aimnetcentral>`_): float32 round-off for all
  six published families, including the DSF, Ewald and PME lattice sums.
  Served with the two-body D3(BJ) term of the reference calculator.
- **PhysNet** (`MMunibas/PhysNet <https://github.com/MMunibas/PhysNet>`_,
  TensorFlow): about 1e-15 in energies, forces and charges from a
  pure-PyTorch translation. Dropout is not implemented.
- **ANI** (`aiqm/torchani <https://github.com/aiqm/torchani>`_): the AEV to
  about 1e-16; the pretrained ANI-1x, ANI-1ccx and ANI-2x ensembles to about
  1e-8 Ha.
- **HDNNP** (`RuNNer <https://gitlab.com/runner-suite/runner2>`_ 2.0.5,
  compiled and run as an external program): RuNNer model directories load
  unchanged; energies, forces and charges agree to about 1e-16 relative for
  molecules in every generation (2G, 3G with point or Gaussian charges and
  screening, 4G with element or network hardnesses and charged structures)
  and to the Ewald truncation (about 1e-12) in periodic cells, also for the
  published 4G Au2/MgO model. 1G follows the review's definition.
- **SchNet** (Schütt *et al.* 2017): a clean-room build, verified against an
  independent implementation of the paper's equations to about 1e-15. No
  schnetpack code is used or compared against.
- **DimeNet and DimeNet++** (`gasteigerjo/dimenet
  <https://github.com/gasteigerjo/dimenet>`_, TensorFlow): about 1e-13
  relative in energies and forces under transplanted weights in float64;
  the published DimeNet++ QM9 model reproduces to float32 round-off.
- **PaiNN** (Schütt *et al.* 2021; reference implementation in
  `schnetpack <https://github.com/atomistic-machine-learning/schnetpack>`_):
  a build from the paper, verified against an independent implementation
  of its equations and against ``schnetpack`` under transplanted weights:
  energies, forces, latent charges, dipoles and polarizabilities agree to
  float64 and float32 round-off.
- **SpookyNet** (`OUnke/SpookyNet <https://github.com/OUnke/SpookyNet>`_):
  a build from the paper, verified against an independent implementation of
  its equations and against the reference code under transplanted weights:
  energies, forces, charges and dipoles agree to float64 round-off (about
  1e-14) for molecules in several charge and spin states, batches and
  periodic cells, and the published example model reproduces to 1e-15. The
  D4 term uses the dftd4 data of :mod:`~xnn.common.models.d4`; the reference
  code ships float32-rounded covalent radii, which moves its dispersion
  energies by about 1e-8 relative.
- **SE(3) steerable CNN** (Weiler *et al.* 2018): a clean-room build from
  the paper; the steerable kernel basis is formed from Clebsch-Gordan
  coefficients and checked against a numerical solution of the paper's
  constraint. The energy is exactly invariant under the rotations of the
  grid onto itself and invariant to the bandlimit under every other one.
- **Spherical CNN** (`jonas-koehler/s2cnn
  <https://github.com/jonas-koehler/s2cnn>`_): a clean-room build from the
  paper; the Wigner d-matrices, the :math:`S^2` / :math:`SO(3)` transforms,
  the integral, the rotation operator and the correlation layers (with
  transplanted filters) match the reference to float32 and float64
  round-off, the correlations match their direct evaluation for point
  filters, and the layers are exactly equivariant on bandlimited signals.
- **BAMBOO** (`bytedance/bamboo <https://github.com/bytedance/bamboo>`_):
  about 1e-15 layer by layer. xnn returns the conservative force, equal to
  upstream's ``forces + qeq_force``.
- **LES** (``cace.modules.EwaldPotential``): ``LatentEwald(CACE)`` reproduces
  CACE-LR to float32 round-off; each kernel to about 1e-16 in float64.
- **DFT-D4 and DFT-D3** (`dftd4 <https://github.com/dftd4/dftd4>`_,
  `simple-dftd3 <https://github.com/dftd3/simple-dftd3>`_): energies to
  about 1e-16 hartree, gradients and virials to about 1e-17, for every
  damping function; the matrix-free D4 regime differs by the reference's
  own Ewald tolerance (about 1e-8 in the charges).
- **ReaxFF / ReaxFF-nn**: the published equations, cross-checked against
  LAMMPS ``pair_style reaxff`` (nonbonded terms to about 1e-6 eV, totals to
  about 0.1 percent from differing bond-list truncation). The authors'
  ReaxFF-nn code is AGPL-licensed, so nothing derived from it is distributed
  and none of its code was copied; the shipped tests recompute every term
  from the papers.
- **OPLS / L-OPLS** (Jorgensen *et al.* 1996): OpenMM to about 1e-7 kJ/mol,
  and Table 1 of the paper to 0.01 kcal/mol with the ``"oplsaa-1996"``
  library.
- **DREIDING** (Mayo *et al.* 1990): the LAMMPS DREIDING styles to about
  1e-10 kcal/mol, the paper's parameter tables entry by entry, and its
  Table XI barriers to about 0.01 kcal/mol.

Fidelity means published results can be reproduced, weights move in either
direction (:ref:`howto-transplant`), and the implementations serve as
readable references for the architectures; the fidelity notebooks double as
annotated tours of each one.
