.. _fidelity:

********************************
Model Fidelity to Upstream Codes
********************************

Each model reproduces its reference code and, when available, its paper
results. The code is checked block by block against the original models, with
transplanted weights where they are accessible (``tests/``,
``notebooks/fidelity_checks/``). The reference packages serve only as external
oracles.

Each reference implementation must also be at least as fast as its upstream
code, for evaluation and for a training step, on the upstream's own hardware
(the same CPU cores when the upstream runs on CPUs). The fast paths
(:ref:`howto-fast-paths`) are separate implementations that give the
reference's numbers and derivatives to round-off. :ref:`model-support` rates
every model against this standard.

.. note::

   **xnn does not ship any pre-trained models or third-party weights.**
   Published models are downloaded from their developers' own release
   locations when you first ask for one, and they stay under their original
   licenses.


Agreement with the upstream codes under identical weights:

- **MACE** (`ACEsuit/mace <https://github.com/ACEsuit/mace>`_): about 1e-16,
  including ScaleShift, the Agnesi and Soft distance transforms and the
  density-normalized blocks. 16 of the 17 released MACE-MP and MACE-OFF
  checkpoints and every head of ``mace-mh-0`` convert to about 1e-15 eV/atom.
  ``mace-mh-1`` is not supported, and ``from_pretrained()`` rejects it with an
  explicit error.
- **NequIP** (`mir-group/nequip <https://github.com/mir-group/nequip>`_): about
  1e-16 in energies, forces and stress. The parameter names match the original
  code, so state dicts transplant directly.
- **Allegro** (`mir-group/allegro <https://github.com/mir-group/allegro>`_
  v0.3.0): about 1e-15, with the upstream weight layout.
- **CACE** (`BingqingCheng/cace <https://github.com/BingqingCheng/cace>`_):
  about 1e-16, molecular and periodic, for every message-passing mechanism.
- **AIMNet2** (`isayevlab/aimnetcentral
  <https://github.com/isayevlab/aimnetcentral>`_): float32 round-off for all six
  published families, including the DSF, Ewald and PME lattice sums. The models
  come with the two-body D3(BJ) term of the reference calculator.
- **PhysNet** (`MMunibas/PhysNet <https://github.com/MMunibas/PhysNet>`_,
  TensorFlow): about 1e-15 in energies, forces and charges. Dropout is not
  implemented.
- **ANI** (`aiqm/torchani <https://github.com/aiqm/torchani>`_): about 1e-16 in
  the AEV. The pretrained ANI-1x, ANI-1ccx and ANI-2x ensembles reproduce to
  about 1e-8 Ha.
- **HDNNP** (`RuNNer <https://gitlab.com/runner-suite/runner2>`_ 2.0.5, run as
  an external program): RuNNer model directories load unchanged. Energies,
  forces and charges agree to about 1e-16 relative for molecules in every
  generation (2G, 3G and 4G, charged structures included). In periodic cells,
  including the published 4G Au2/MgO model, they agree to the Ewald truncation
  (about 1e-12). 1G follows `Behler, Chem. Rev. 121, 10037 (2021)
  <https://doi.org/10.1021/acs.chemrev.0c00868>`_.
- **SchNet** (`Schütt et al., Adv. Neural Inf. Process. Syst. 30 (2017)
  <https://proceedings.neurips.cc/paper/2017/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html>`_;
  reference implementation in
  `schnetpack <https://github.com/atomistic-machine-learning/schnetpack>`_ 2.2.0):
  energies and forces agree to about 1e-15 in float64 and to float32 round-off,
  for molecules, batches and periodic cells. They are checked against an
  independent implementation of the paper's equations and against
  ``schnetpack`` with transplanted weights, with both the ``schnetpack``
  defaults and the paper's options.
- **DimeNet and DimeNet++** (`gasteigerjo/dimenet
  <https://github.com/gasteigerjo/dimenet>`_, TensorFlow): about 1e-13 relative
  in energies and forces in float64. The published DimeNet++ QM9 model
  reproduces to float32 round-off.
- **GemNet** (`TUM-DAML/gemnet_pytorch
  <https://github.com/TUM-DAML/gemnet_pytorch>`_): about 1e-13 relative in
  energies and forces in float64, for GemNet-Q, -T, -dQ and -dT. The published
  GemNet-Q and GemNet-T weights reproduce to float32 round-off.
- **PaiNN** (`Schütt et al., Proc. Mach. Learn. Res. 139, 9377 (2021)
  <https://proceedings.mlr.press/v139/schutt21a.html>`_; reference
  implementation in
  `schnetpack <https://github.com/atomistic-machine-learning/schnetpack>`_):
  energies, forces, latent charges, dipoles and polarizabilities agree to
  float64 and float32 round-off. They are checked against an independent
  implementation of the paper's equations and against ``schnetpack`` with
  transplanted weights, with the ``schnetpack`` defaults, the paper's options
  and shared weights.
- **ET and TensorNet** (`torchmd/torchmd-net
  <https://github.com/torchmd/torchmd-net>`_): built from the papers. Energies
  and forces agree to about 1e-14 in float64 and to float32 round-off, for
  molecules, batches, charged structures, both ET cutoff conventions and all
  TensorNet groups.
- **MEGNet** (`MatGL 4.1.4 <https://github.com/materialyzeai/matgl>`_): MatGL's
  pretrained formation energy and multi-fidelity band gap models agree to about
  1e-14 in float64 and to float32 round-off, at every fidelity, for crystals,
  molecules and batches. Disordered sites match the paper's code (`megnet 1.3.2
  <https://github.com/materialyzeai/megnet>`_) as closely as possible.
- **Equiformer and EquiformerV2**
  (`atomicarchitects/equiformer <https://github.com/atomicarchitects/equiformer>`_;
  `fairchem-core 1.10 <https://github.com/facebookresearch/fairchem>`_): built
  from the papers. Energies and forces agree to about 1e-14 in float64 and about
  1e-6 in float32, for every attention, norm, activation and radial-basis
  variant, on molecules, batches and periodic cells. The reference EquiformerV2
  detaches its edge frames from the positions and rolls them at random. xnn
  uses deterministic frames and differentiates through them, so its forces are
  the exact gradient of its energy.
- **SpookyNet** (`OUnke/SpookyNet <https://github.com/OUnke/SpookyNet>`_):
  built from the paper. Energies, forces, charges and dipoles agree to about
  1e-14 in float64, for molecules in several charge and spin states, batches
  and periodic cells. The published example model reproduces to 1e-15. The D4
  term uses the ``dftd4`` data of :mod:`~xnn.common.models.d4`. The reference
  code rounds its covalent radii to float32, which moves its dispersion
  energies by about 1e-8 relative.
- **GDML and sGDML** (`stefanch/sGDML <https://github.com/stefanch/sGDML>`_): models trained by
  the sGDML package load into xnn and give the same energies and forces to about 1e-13 in float64. A
  float32 model rounds only its positions, as the package's GPU backend does, and keeps its coefficients
  in float64. Refitting the same training data gives the same predictions to 1e-9 or better, and the
  symmetry search finds the same permutations.
- **SE(3) steerable CNN** (`Weiler et al., Adv. Neural Inf. Process. Syst. 31
  (2018)
  <https://proceedings.neurips.cc/paper/2018/hash/488e4104520c6aab692863cc1dba45af-Abstract.html>`_):
  built from the paper. The steerable kernel basis comes from Clebsch-Gordan
  coefficients and matches a numerical solution of the paper's constraint. The
  energy is exactly invariant under the rotations that map the grid onto
  itself, and invariant to the bandlimit under all others.
- **Spherical CNN** (`jonas-koehler/s2cnn
  <https://github.com/jonas-koehler/s2cnn>`_): built from the paper. The Wigner
  d-matrices, the :math:`S^2` and :math:`SO(3)` transforms, the rotation
  operator and the correlation layers match the reference to float32 and
  float64 round-off. The layers are exactly equivariant on bandlimited signals.
- **BAMBOO** (`bytedance/bamboo <https://github.com/bytedance/bamboo>`_):
  about 1e-15 layer by layer. xnn's forces are the exact gradient of its
  energy. The upstream's forces differ from that gradient by about one part in
  10^10.
- **ByteFF and ByteFF-Pol** (`ByteDance-Seed/byteff2
  <https://github.com/ByteDance-Seed/byteff2>`_): the molecular graphs and the
  predicted parameters are identical bit for bit. Energies and forces of
  molecules and clusters agree to about 1e-14 with the released ByteFF-26 and
  ByteFF-Pol-25 networks. xnn leaves out a small polarization self-interaction
  term that the upstream keeps in float64 and the paper does not have. Periodic
  ByteFF-Pol matches byteff2's OpenMM setup to 1e-15 in the short-range terms
  and 1e-10 or better in electrostatics and induction. The remaining
  differences come from OpenMM's single-precision steps.
- **LES** (``cace.modules.EwaldPotential``): ``LatentEwald(CACE)`` reproduces
  CACE-LR to float32 round-off and each kernel to about 1e-16 in float64.
- **D4 and D3** (`dftd4 <https://github.com/dftd4/dftd4>`_,
  `simple-dftd3 <https://github.com/dftd3/simple-dftd3>`_): energies to about
  1e-16 hartree, gradients and virials to about 1e-17, for every damping
  function. The matrix-free D4 regime agrees to the reference's own Ewald
  tolerance (about 1e-8 in the charges).
- **ReaxFF / ReaxFF-nn**: built from the published equations. With
  ``conventions="lammps"`` (the default for classical fields), it matches
  LAMMPS ``pair_style reaxff`` with ``fix qeq/reaxff`` on water, NaCl(aq) and
  C/H/O structures of up to 1536 atoms. Energies agree to about 3e-10 eV per
  atom, forces to about 1e-6 eV/Angstrom and charges to 1e-9 e. The vendored
  ``CHO_cho_2008`` file rounds eight bond energies to three decimals, about
  1e-5 eV per bond. The authors' ReaxFF-nn code is AGPL-licensed, so none of
  its code is copied and nothing derived from it is distributed. The tests
  recompute every term from the papers.
- **OPLS / L-OPLS** (`Jorgensen et al., J. Am. Chem. Soc. 118, 11225 (1996)
  <https://doi.org/10.1021/ja9621760>`_): OpenMM to about 1e-7 kJ/mol, and
  Table 1 of the paper to 0.01 kcal/mol with the ``"oplsaa-1996"`` library.
- **DREIDING** (`Mayo et al., J. Phys. Chem. 94, 8897 (1990)
  <https://doi.org/10.1021/j100389a010>`_): the LAMMPS DREIDING styles to about
  1e-10 kcal/mol. The parameter tables match the paper entry by entry, and its
  Table XI barriers to about 0.01 kcal/mol.
- **AMBER ff94** (`Cornell et al., J. Am. Chem. Soc. 117, 5179 (1995)
  <https://doi.org/10.1021/ja00124a002>`_): LAMMPS to about 1e-8 kcal/mol on
  capped peptides, water and ions. The atom types, RESP charges and impropers
  match ``tleap``, and the energy components match ``sander`` on peptides and
  nucleic acids. Table 14 and the residue charges of the paper match entry by
  entry. Float32 stays within 1e-6 relative of float64.
- **CHARMM22** (`MacKerell et al., J. Phys. Chem. B 102, 3586 (1998)
  <https://doi.org/10.1021/jp973084f>`_): LAMMPS to about 1e-9 kcal/mol on
  capped peptides, water and ions. The atom types, charges and impropers match
  ``psfgen``, and the energy components match NAMD on patched peptides. The
  parameter files and the paper's quoted values match entry by entry. Float32
  stays within 1e-6 relative of float64.
- **AMOEBA and HIPPO** (`Ren and Ponder, J. Phys. Chem. B 107, 5933 (2003)
  <https://doi.org/10.1021/jp027815+>`_;
  `Ponder et al., J. Phys. Chem. B 114, 2549 (2010)
  <https://doi.org/10.1021/jp910674d>`_; `Shi et al., J. Chem. Theory Comput. 9,
  4046 (2013) <https://doi.org/10.1021/ct4003702>`_;
  `Rackers et al., J. Chem. Theory Comput. 17, 7056 (2021)
  <https://doi.org/10.1021/acs.jctc.1c00628>`_): Tinker 8.11.3 term by term, to
  about 1e-8 kcal/mol and 1e-7 kcal/mol/A, on water dimers and clusters,
  distorted monomers and capped dipeptides. The water systems also match the
  LAMMPS AMOEBA package and the water moments, cluster energies and
  vibrational frequencies of the papers. Float32 stays within 2e-5 relative of
  float64.

  In periodic cells, the particle-mesh sums match explicit lattice sums to
  1e-8, and forces, stress and Hessians match finite differences. On 64-water
  boxes, xnn agrees with Tinker 8 within 5e-3 kcal/mol per term and 2e-3
  kcal/mol/A in forces. On larger water and ethylene carbonate cells, with
  Tinker's own meshes, the forces agree to 0.007 to 0.19 meV/A rms. That is
  the level at which Tinker's GPU build and LAMMPS agree with Tinker.
- **PM3, PM6, PM6-ORG, PM7** (`Stewart, J. Comput. Chem. 10, 209 (1989)
  <https://doi.org/10.1002/jcc.540100208>`_; `Stewart, J. Mol. Model. 13, 1173
  (2007) <https://doi.org/10.1007/s00894-007-0233-4>`_;
  `Stewart, J. Mol. Model. 19, 1 (2013)
  <https://doi.org/10.1007/s00894-012-1667-x>`_; `Stewart and Stewart, J. Mol.
  Model. 29, 284 (2023) <https://doi.org/10.1007/s00894-023-05695-1>`_): MOPAC
  23.2.5 to about 1e-6 kcal/mol in heats of formation and 1e-7 in charges,
  open shells and transition metals included. The PM6 and PM7 S22 interaction
  energies of the PM7 paper (Table 5) are reproduced. MOPAC's PM6-ORG gradient
  is not the derivative of its energy, while xnn's is.

In xnn, fidelity means that published results can be reproduced, that weights
move in either direction (:ref:`howto-transplant`), and that the
implementations serve as readable references for the architectures.
