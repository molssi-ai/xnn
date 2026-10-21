.. _fidelity:

********************************
Model Fidelity to Upstream Codes
********************************

Each model reproduces its reference code and its paper results when available,
and the code is verified block by block with transplanted weights from the
original models when accessible (``tests/``, ``notebooks/fidelity_checks/``). The
reference packages are used only as external oracles and references for
validating our implementation.

.. note::

   **xnn does not ship any pre-trained models or third-party weights.**
   Published models are downloaded from their developers' own release
   locations when you first ask for one, and they stay under their original
   licenses.


Numerical agreement under identical weights involves:

- **MACE** (`ACEsuit/mace <https://github.com/ACEsuit/mace>`_): about 1e-16,
  including ScaleShift, the Agnesi and Soft distance transforms and the
  density-normalized blocks. 16 of the 17 released MACE-MP, MACE-OFF checkpoints
  and every head of ``mace-mh-0`` convert with energy parity of about 1e-15
  eV/atom. The ``mace-mh-1`` model is not currently supported and will be
  rejected with an explicit error if requested by a ``from_pretrained()`` call.
- **NequIP** (`mir-group/nequip <https://github.com/mir-group/nequip>`_): about
  1e-16 in energies, forces and stress. The parameter names match between xnn
  and the original code. Therefore, the state dicts can be transplanted
  directly.
- **Allegro** (`mir-group/allegro <https://github.com/mir-group/allegro>`_
  v0.3.0): about 1e-15, with the upstream weight layout.
- **CACE** (`BingqingCheng/cace <https://github.com/BingqingCheng/cace>`_):
  about 1e-16, molecular and periodic, for every message-passing mechanism.
- **AIMNet2** (`isayevlab/aimnetcentral
  <https://github.com/isayevlab/aimnetcentral>`_): float32 round-off for all six
  published families, including the DSF, Ewald and PME lattice sums. The
  implementation is served with the two-body D3(BJ) term of the reference
  calculator.
- **PhysNet** (`MMunibas/PhysNet <https://github.com/MMunibas/PhysNet>`_,
  TensorFlow): about 1e-15 in energies, forces and charges from a
  pure-PyTorch translation. The dropout layer is not implemented.
- **ANI** (`aiqm/torchani <https://github.com/aiqm/torchani>`_): the AEV to
  about 1e-16. The pretrained ANI-1x, ANI-1ccx and ANI-2x ensembles are
  reproduced to about 1e-8 Ha in energies.
- **HDNNP** (`RuNNer <https://gitlab.com/runner-suite/runner2>`_ 2.0.5, compiled
  and run as an external program): RuNNer model directories load unchanged.
  Energies, forces and charges agree to about 1e-16 relative for molecules in
  every generation (2G, 3G with point or Gaussian charges and screening, 4G with
  element or network hardnesses and charged structures) and to the Ewald
  truncation (about 1e-12) in periodic cells as well as for the published 4G
  Au2/MgO model. Our implementation of 1G follows the `review's
  <https://doi.org/10.1021/acs.chemrev.0c00868>`_ definition.
- **SchNet** (Schütt *et al.* 2017): a clean-room build, verified against an
  independent implementation of the paper's equations to about 1e-15. No
  schnetpack code is used or compared against as an oracle.
- **DimeNet and DimeNet++** (`gasteigerjo/dimenet
  <https://github.com/gasteigerjo/dimenet>`_, TensorFlow): about 1e-13 relative
  in energies and forces under transplanted weights in float64. The published
  DimeNet++ QM9 model is reproduced up to float32 round-off.
- **GemNet** (`TUM-DAML/gemnet_pytorch
  <https://github.com/TUM-DAML/gemnet_pytorch>`_): about 1e-13 relative in
  energies and forces under transplanted weights in float64, for GemNet-Q, -T,
  -dQ and -dT. The published GemNet-Q and GemNet-T weights are reproduced to
  float32 round-off.
- **PaiNN** (Schütt *et al.* 2021; reference implementation in
  `schnetpack <https://github.com/atomistic-machine-learning/schnetpack>`_):
  a build from the paper, verified against an independent implementation
  of its equations and against ``schnetpack`` under transplanted weights:
  energies, forces, latent charges, dipoles and polarizabilities agree to
  float64 and float32 round-off.
- **ET and TensorNet** (`torchmd/torchmd-net
  <https://github.com/torchmd/torchmd-net>`_): built from the papers, verified
  against independent implementations of their equations and against the
  reference code under transplanted weights: energies and forces agree to
  float64 round-off (about 1e-14) for molecules, batches, charged structures and
  both ET cutoff conventions and TensorNet groups, and to float32 round-off in
  single precision.
- **SpookyNet** (`OUnke/SpookyNet <https://github.com/OUnke/SpookyNet>`_): a
  build from the paper, verified against an independent implementation of its
  equations and against the reference code under transplanted weights: energies,
  forces, charges and dipoles agree to float64 round-off (about 1e-14 in worst
  case) for molecules in several charge and spin states, batches and periodic
  cells, and the published example model reproduces to 1e-15. The D4 term uses
  the ``dftd4`` data of :mod:`~xnn.common.models.d4`. The reference code ships
  float32-rounded covalent radii, which moves its dispersion energies by about
  1e-8 relative.
- **SE(3) steerable CNN** (Weiler *et al.* 2018): a clean-room build from the
  paper. The steerable kernel basis is formed from Clebsch-Gordan coefficients
  and checked against a numerical solution of the paper's constraint. The energy
  is exactly invariant under the rotations of the grid onto itself and invariant
  to the bandlimit under every other one.
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
  CACE-LR to float32 round-off and each kernel to about 1e-16 in float64.
- **D4 and D3** (`dftd4 <https://github.com/dftd4/dftd4>`_,
  `simple-dftd3 <https://github.com/dftd3/simple-dftd3>`_): energies to about
  1e-16 hartree, gradients and virials to about 1e-17, for every damping
  function. The matrix-free D4 regime differs by the reference's own Ewald
  tolerance (about 1e-8 in the charges).
- **ReaxFF / ReaxFF-nn**: based on the published equations with
  ``conventions="lammps"`` (the default for classical fields), LAMMPS
  ``pair_style reaxff`` with ``fix qeq/reaxff`` on water, NaCl(aq) and C/H/O
  structures of up to 1536 atoms. The energies match to about 3e-10 eV per atom,
  forces to about 1e-6 eV/Angstrom, charges to 1e-9 e. The vendored
  ``CHO_cho_2008`` file rounds eight bond energies to three decimals, which
  moves its energies by about 1e-5 eV per bond. The authors' ReaxFF-nn code is
  AGPL-licensed. So, nothing derived from it is distributed and none of its
  code was copied. The shipped tests recompute every term from the papers.
- **OPLS / L-OPLS** (Jorgensen *et al.* 1996): OpenMM to about 1e-7 kJ/mol,
  and Table 1 of the paper to 0.01 kcal/mol with the ``"oplsaa-1996"``
  library.
- **DREIDING** (Mayo *et al.* 1990): the LAMMPS DREIDING styles to about
  1e-10 kcal/mol, the paper's parameter tables entry by entry, and its
  Table XI barriers to about 0.01 kcal/mol.
- **AMBER ff94** (Cornell *et al.* 1995): LAMMPS (harmonic, fourier, cvff and
  lj/cut/coul/cut styles with the AMBER 1,4 factors) to about 1e-8 kcal/mol on
  capped peptides, water and ions; the atom types, RESP charges and improper
  ordering of ``tleap`` and the energy components of ``sander`` on
  sequence-built peptides and nucleic acids; Table 14 and the residue charges
  of the paper entry by entry; float32 within 1e-6 relative of float64.
- **CHARMM22** (MacKerell *et al.* 1998): LAMMPS (harmonic, charmm angle and
  dihedral, harmonic improper and lj/charmm/coul/charmm styles with the
  CHARMM 1,4 weights) to about 1e-9 kcal/mol on capped peptides, water and
  ions; the atom types, charges and impropers of ``psfgen`` and the energy
  components of NAMD on sequence-built, patched peptides; the parameter files
  and the paper's quoted values entry by entry; float32 within 1e-6 relative
  of float64.
- **PM3, PM6, PM6-ORG, PM7** (Stewart 1989, 2007, 2013, 2023): MOPAC 23.2.5
  to about 1e-6 kcal/mol in heats of formation and 1e-7 in charges, open
  shells and transition metals included, and the PM6 and PM7 S22 interaction
  energies of the PM7 paper (Table 5). Overlaps keep MOPAC's truncated
  auxiliary series; its PM6-ORG gradient is not the derivative of its
  energy, xnn's is.

Fidelity in xnn has a specific meaning: it means published results can be
reproduced, weights can move in either direction (:ref:`howto-transplant`), and
the implementations serve as readable references for the architectures.
