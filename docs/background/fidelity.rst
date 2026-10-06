.. _fidelity:

********************************
Model Fidelity to Upstream Codes
********************************

Each model reproduces its reference code, or the paper when the reference
is a manuscript, and is verified block by block with transplanted weights
(``tests/``, ``examples/fidelity_checks/``). The reference packages are used
only as external oracles; at run time NequIP, MACE and Allegro need ``e3nn``
and every other model is plain PyTorch. Agreement under identical weights:

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
- **SchNet** (Schütt *et al.* 2017): a clean-room build, verified against an
  independent implementation of the paper's equations to about 1e-15. No
  schnetpack code is used or compared against.
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
