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
  and every head of ``mace-mh-0`` convert with an energy fidelity of about 1e-15
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
  Au2/MgO model. Our implementation of 1G follows the definition of
  `Behler, Chem. Rev. 121, 10037 (2021) <https://doi.org/10.1021/acs.chemrev.0c00868>`_.
- **SchNet** (`Schütt et al., Adv. Neural Inf. Process. Syst. 30 (2017)
  <https://proceedings.neurips.cc/paper/2017/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html>`_;
  reference implementation in
  `schnetpack <https://github.com/atomistic-machine-learning/schnetpack>`_ 2.2.0):
  built from the paper and verified against an independent implementation of
  its equations and against ``schnetpack`` under transplanted weights, with its
  filter network assembled to the paper's two activations. Energies and forces
  agree to float64 round-off (about 1e-15) and float32 round-off for molecules,
  batches and periodic cells.
- **DimeNet and DimeNet++** (`gasteigerjo/dimenet
  <https://github.com/gasteigerjo/dimenet>`_, TensorFlow): about 1e-13 relative
  in energies and forces under transplanted weights in float64. The published
  DimeNet++ QM9 model is reproduced up to float32 round-off.
- **GemNet** (`TUM-DAML/gemnet_pytorch
  <https://github.com/TUM-DAML/gemnet_pytorch>`_): about 1e-13 relative in
  energies and forces under transplanted weights in float64, for GemNet-Q, -T,
  -dQ and -dT. The published GemNet-Q and GemNet-T weights are reproduced to
  float32 round-off.
- **PaiNN** (`Schütt et al., Proc. Mach. Learn. Res. 139, 9377 (2021)
  <https://proceedings.mlr.press/v139/schutt21a.html>`_; reference
  implementation in
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
- **Equiformer and EquiformerV2**
  (`atomicarchitects/equiformer <https://github.com/atomicarchitects/equiformer>`_;
  `fairchem-core 1.10 <https://github.com/facebookresearch/fairchem>`_): built
  from the papers and verified against the reference codes under transplanted
  weights. Energies and forces agree to float64 round-off (about 1e-14) and to
  float32 round-off (about 1e-6) for every attention, norm, activation and
  radial-basis variant, on molecules, batches and periodic cells. The reference
  EquiformerV2 detaches its edge frames from the positions and samples their
  roll at random. xnn uses deterministic frames and differentiates through
  them, so its forces are the exact gradient of its energy.
- **SpookyNet** (`OUnke/SpookyNet <https://github.com/OUnke/SpookyNet>`_): a
  build from the paper, verified against an independent implementation of its
  equations and against the reference code under transplanted weights: energies,
  forces, charges and dipoles agree to float64 round-off (about 1e-14 in worst
  case) for molecules in several charge and spin states, batches and periodic
  cells, and the published example model reproduces to 1e-15. The D4 term uses
  the ``dftd4`` data of :mod:`~xnn.common.models.d4`. The reference code ships
  float32-rounded covalent radii, which moves its dispersion energies by about
  1e-8 relative.
- **SE(3) steerable CNN** (`Weiler et al., Adv. Neural Inf. Process. Syst. 31
  (2018)
  <https://proceedings.neurips.cc/paper/2018/hash/488e4104520c6aab692863cc1dba45af-Abstract.html>`_):
  built from the
  paper. The steerable kernel basis is formed from Clebsch-Gordan coefficients
  and checked against a numerical solution of the paper's constraint. The energy
  is exactly invariant under the rotations of the grid onto itself and invariant
  to the bandlimit under every other one.
- **Spherical CNN** (`jonas-koehler/s2cnn
  <https://github.com/jonas-koehler/s2cnn>`_): built from the
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
- **OPLS / L-OPLS** (`Jorgensen et al., J. Am. Chem. Soc. 118, 11225 (1996)
  <https://doi.org/10.1021/ja9621760>`_): OpenMM to about 1e-7 kJ/mol,
  and Table 1 of the paper to 0.01 kcal/mol with the ``"oplsaa-1996"``
  library.
- **DREIDING** (`Mayo et al., J. Phys. Chem. 94, 8897 (1990)
  <https://doi.org/10.1021/j100389a010>`_): the LAMMPS DREIDING styles to about
  1e-10 kcal/mol, the paper's parameter tables entry by entry, and its
  Table XI barriers to about 0.01 kcal/mol.
- **AMBER ff94** (`Cornell et al., J. Am. Chem. Soc. 117, 5179 (1995)
  <https://doi.org/10.1021/ja00124a002>`_): LAMMPS (harmonic, fourier, cvff and
  lj/cut/coul/cut styles with the AMBER 1,4 factors) to about 1e-8 kcal/mol on
  capped peptides, water and ions; the atom types, RESP charges and improper
  ordering of ``tleap`` and the energy components of ``sander`` on
  sequence-built peptides and nucleic acids; Table 14 and the residue charges
  of the paper entry by entry; float32 within 1e-6 relative of float64.
- **CHARMM22** (`MacKerell et al., J. Phys. Chem. B 102, 3586 (1998)
  <https://doi.org/10.1021/jp973084f>`_): LAMMPS (harmonic, charmm angle and
  dihedral, harmonic improper and lj/charmm/coul/charmm styles with the
  CHARMM 1,4 weights) to about 1e-9 kcal/mol on capped peptides, water and
  ions; the atom types, charges and impropers of ``psfgen`` and the energy
  components of NAMD on sequence-built, patched peptides; the parameter files
  and the paper's quoted values entry by entry; float32 within 1e-6 relative
  of float64.
- **AMOEBA and HIPPO** (`Ren and Ponder, J. Phys. Chem. B 107, 5933 (2003)
  <https://doi.org/10.1021/jp027815+>`_;
  `Ponder et al., J. Phys. Chem. B 114, 2549 (2010)
  <https://doi.org/10.1021/jp910674d>`_; `Shi et al., J. Chem. Theory Comput. 9,
  4046 (2013) <https://doi.org/10.1021/ct4003702>`_;
  `Rackers et al., J. Chem. Theory Comput. 17, 7056 (2021)
  <https://doi.org/10.1021/acs.jctc.1c00628>`_): Tinker 8.11.3 term by term
  (valence terms
  including in-plane angles, stretch-bend, out-of-plane, pi-torsions and the
  glycine torsion-torsion map, buffered 14-7, permanent multipoles, Thole and
  density-damped induction, charge penetration, Pauli repulsion, dispersion,
  charge transfer, charge flux) to about 1e-8 kcal/mol and 1e-7 kcal/mol/A on
  water dimers, clusters, distorted monomers and capped dipeptides with
  terminal residues; the LAMMPS AMOEBA package on the water systems; the
  water monomer moments, dimer and cluster energies and vibrational
  frequencies of the papers; float32 within 2e-5 relative of float64.
  Periodic cells (particle-mesh Ewald multipoles, induction and HIPPO
  dispersion, switched short-range terms, iterative induced dipoles): the
  mesh sums against explicit reciprocal-space and lattice sums to 1e-8,
  slab sums independent of the vacuum gap, forces, stress and Hessians
  against finite differences; the optional mesh self-interaction
  correction equals a brute-force position average to 1e-7, vanishes at
  a fine mesh and cuts the coarse-mesh dispersion error of a 64-water
  box by an order of magnitude; a 2x2x2 replica of a cell gives the same
  energy per cell and the same forces to 1e-10 meV/A;
  Tinker 8 on 64-water boxes (AMOEBA and HIPPO, 48^3 meshes) within 5e-3
  kcal/mol per term and 2e-3 kcal/mol/A in forces; on 320- and 2560-atom
  ethylene carbonate cells and 648-atom water boxes (Tinker's own meshes)
  within 0.01 kcal/mol in the multipole energy per 320 atoms and 0.007 to
  0.19 meV/A rms in atomic forces, the level at which Tinker's GPU build and
  the LAMMPS AMOEBA package agree with Tinker.
- **PM3, PM6, PM6-ORG, PM7** (`Stewart, J. Comput. Chem. 10, 209 (1989)
  <https://doi.org/10.1002/jcc.540100208>`_; `Stewart, J. Mol. Model. 13, 1173
  (2007) <https://doi.org/10.1007/s00894-007-0233-4>`_;
  `Stewart, J. Mol. Model. 19, 1 (2013)
  <https://doi.org/10.1007/s00894-012-1667-x>`_; `Stewart and Stewart, J. Mol.
  Model. 29, 284 (2023) <https://doi.org/10.1007/s00894-023-05695-1>`_): MOPAC
  23.2.5
  to about 1e-6 kcal/mol in heats of formation and 1e-7 in charges, open
  shells and transition metals included, and the PM6 and PM7 S22 interaction
  energies of the PM7 paper (Table 5). Overlaps keep MOPAC's truncated
  auxiliary series; its PM6-ORG gradient is not the derivative of its
  energy, xnn's is.

Fidelity in xnn has a specific meaning: it means published results can be
reproduced, weights can move in either direction (:ref:`howto-transplant`), and
the implementations serve as readable references for the architectures.
