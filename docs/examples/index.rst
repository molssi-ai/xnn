.. _examples:

********
Examples
********

Every notebook of the repository's `notebooks/
<https://github.com/molssi-ai/xnn/tree/main/notebooks>`_ directory, rendered
with its executed outputs. Its ``examples/``, ``fidelity_checks/`` and
``parity_checks/`` folders each follow the package layout
(``<family>/<model>/``). To run one yourself, see :ref:`howto-examples`
for the environments. Entries marked *coming soon* are being reviewed and
become links as they are published.

Data
====

The data hub: ``load_dataset()``, splits, units, caching.

.. example-toctree::
   :caption: Data

   Loading datasets with `xnn`: the data hub & `load_dataset()` <nb/examples/common/data/load_dataset_tutorial>
   GMTKN55 in one line: reaction energies, WTMAD-2 and pretrained models on the benchmark <nb/examples/common/data/gmtkn55_dataset>

Models
======

The model hub: foundation models, xnn-trained models and Zenodo uploads
under one ``from_pretrained()`` call, the cache layout and sharing your own.

.. example-toctree::
   :caption: Models

   Pre-trained models with `xnn`: the model hub & `from_pretrained()` <nb/examples/common/models/pretrained_models_tutorial>

ANI (dnn)
=========

ANI from scratch on rMD17, then the four published ANI datasets, each paired
with its model preset.

.. example-toctree::
   :caption: ANI (dnn)

   Training ANI from scratch on rMD17 (paracetamol) <nb/examples/dnn/ani/ani_rmd17_train>
   The ANI-1 dataset in one line, and a paper-style correlation test <nb/examples/dnn/ani/ani1_dataset>
   The ANI-1x dataset in one line: forces, active learning, and the `ani-1x` preset <nb/examples/dnn/ani/ani1x_dataset>
   The ANI-1ccx dataset: coupled-cluster labels and transfer learning with the `ani-1ccx` preset <nb/examples/dnn/ani/ani1ccx_dataset>
   The ANI-2x dataset: seven elements (S, F, Cl) and the `ani-2x` preset <nb/examples/dnn/ani/ani2x_dataset>

HDNNP (dnn)
===========

The four generations of high-dimensional neural network potentials: their
defining properties on untrained models, first and second generation on
rMD17, and the 2G / 3G / 4G comparison of the fourth-generation paper
(`Ko et al., Nat. Commun. 12, 398 (2021)
<https://doi.org/10.1038/s41467-020-20427-2>`_).

.. example-toctree::
   :caption: HDNNP (dnn)

   The four generations of HDNNPs: symmetry functions, locality, non-local charge transfer <nb/examples/dnn/hdnnp/hdnnp_paper_checks>
   First- and second-generation potentials on rMD17 ethanol <nb/examples/dnn/hdnnp/hdnnp_rmd17_train>
   2G, 3G and 4G HDNNPs on the benchmarks of the fourth-generation paper <nb/examples/dnn/hdnnp/hdnnp_4g_benchmarks>

PhysNet (dnn)
=============

Training and MD against the original TensorFlow PhysNet.

.. example-toctree::
   :caption: PhysNet (dnn)

   Training & testing PhysNet on Argon MD data: `xnn` vs the original PhysNet, step by step <nb/examples/dnn/physnet/physnet_argon_train_test>
   Argon MD with PhysNet: `xnn` NPT density + lock-step NVE against the original <nb/examples/dnn/physnet/physnet_argon_density_md>

SchNet (gnn)
============

The architecture of `Schütt et al., Adv. Neural Inf. Process. Syst. 30 (2017)
<https://proceedings.neurips.cc/paper/2017/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html>`_
trained on rMD17 ethanol, then thermostat-free NVE dynamics to show energy
conservation by construction.

.. example-toctree::
   :caption: SchNet (gnn)

   Training SchNet on rMD17 (ethanol) <nb/examples/gnn/schnet/schnet_rmd17_train>
   SchNet-driven NVE dynamics: energy conservation by construction <nb/examples/gnn/schnet/schnet_ethanol_md>

DimeNet and DimeNet++ (gnn)
===========================

The directional message passing papers (`Gasteiger et al., Int. Conf. Learn.
Represent. (2020) <https://arxiv.org/abs/2003.03123>`_
and `Gasteiger et al., NeurIPS Workshop on Machine Learning for Molecules (2020)
<https://arxiv.org/abs/2011.14115>`_) checked with untrained models: the Bessel
and spherical
Fourier-Bessel bases, the envelope, twice continuous differentiability across
the cutoff, the distinguishability argument and the DimeNet++ speed-up. Then
the MD17 experiment of the paper on rMD17 ethanol: DimeNet, DimeNet++ and
SchNet trained on 1000 structures, with the learned 2D filters.

.. example-toctree::
   :caption: DimeNet and DimeNet++ (gnn)

   DimeNet from the paper: bases, smoothness, directionality, and the DimeNet++ speed-up <nb/examples/gnn/dimenet/dimenet_paper_checks>
   DimeNet and DimeNet++ on rMD17 ethanol: energies and forces from 1000 structures <nb/examples/gnn/dimenet/dimenet_rmd17_train>

GemNet (gnn)
============

The geometric message passing paper
(`Gasteiger et al., Adv. Neural Inf. Process. Syst. 34, 6790 (2021)
<https://proceedings.neurips.cc/paper/2021/hash/35cf8659cfcb13224cbd47863a34fc58-Abstract.html>`_)
checked with untrained models: the three bases, triplets and quadruplets,
what the dihedral angle adds, the variance scaling of Fig. 3, smoothness at
both cutoffs, direct versus conservative forces and the cost of the variants.
Then GemNet-T, -Q and -dT trained on rMD17 ethanol, with NVE energy
conservation.

.. example-toctree::
   :caption: GemNet (gnn)

   GemNet from the paper: bases, two-hop message passing, variance scaling, direct forces <nb/examples/gnn/gemnet/gemnet_paper_checks>
   GemNet on rMD17 ethanol: GemNet-T, GemNet-Q and direct forces <nb/examples/gnn/gemnet/gemnet_rmd17_train>

PaiNN (gnn)
===========

The equivariant message passing paper
(`Schütt et al., Proc. Mach. Learn. Res. 139, 9377 (2021)
<https://proceedings.mlr.press/v139/schutt21a.html>`_): the architecture and its
claims checked with untrained models, the MD17
experiments and ablations of Tables III and IV, the rotational profile of
substituted ferrocene at short cutoffs (Fig. 4), and the infrared and Raman
spectra of ethanol from classical and ring-polymer dynamics (Fig. 5).

.. example-toctree::
   :caption: PaiNN (gnn)

   PaiNN from the paper: equivariant messages, directional information and tensorial outputs <nb/examples/gnn/painn/painn_paper_checks>
   PaiNN on MD17: energies and forces from 1000 structures, and the ablations <nb/examples/gnn/painn/painn_md17>
   Propagation of directional information: the substituted ferrocene rotation <nb/examples/gnn/painn/painn_ferrocene>
   Infrared and Raman spectra of ethanol from classical and ring-polymer dynamics <nb/examples/gnn/painn/painn_spectra>

TensorNet (gnn)
===============

The Cartesian tensor network of TorchMD-Net 2.0
(`Simeon and De Fabritiis, Adv. Neural Inf. Process. Syst. 36, 37334 (2023)
<https://doi.org/10.52202/075280-1623>`_;
`Pelaez et al., J. Chem. Theory Comput. 20, 4076 (2024)
<https://doi.org/10.1021/acs.jctc.4c00253>`_): the irreducible decomposition,
O(3)
against SO(3) on enantiomers, the total-charge coupling, smoothness and the cost
of 0 to 3 layers. Then TensorNet on rMD17 aspirin, and TensorNet against ET on a
QM9 subset at a short budget.

.. example-toctree::
   :caption: TensorNet (gnn)

   TensorNet from the paper: Cartesian tensors, O(3) and SO(3), charge coupling and layer depth <nb/examples/gnn/tensornet/tensornet_paper_checks>
   TensorNet on rMD17 aspirin: energies and forces from 950 structures <nb/examples/gnn/tensornet/tensornet_rmd17_train>
   TensorNet and ET on QM9 U0: the layer-depth trend of TorchMD-Net 2.0 at a short budget <nb/examples/gnn/tensornet/tensornet_qm9_u0>


3D steerable CNN (cnn)
======================

The Tetris experiment of the 3D steerable CNN paper
(`Weiler et al., Adv. Neural Inf. Process. Syst. 31 (2018)
<https://proceedings.neurips.cc/paper/2018/hash/488e4104520c6aab692863cc1dba45af-Abstract.html>`_):
steerable and conventional kernels trained on the eight
pieces in one orientation and tested under random rotations, with the
paper's low-pass ablation, the kernel-basis figure and the equivariance of
the internal fields. Then the same comparison for interatomic potentials:
the steerable and the conventional 3D CNN trained on rMD17 ethanol and
evaluated on rotated structures.

.. example-toctree::
   :caption: 3D steerable CNN (cnn)

   The Tetris experiment: rotation equivariance of 3D steerable CNNs <nb/examples/cnn/se3cnn/tetris_equivariance>
   Steerable against conventional 3D CNN potentials on rMD17 under rotations <nb/examples/cnn/se3cnn/se3cnn_rmd17_rotation>

Spherical CNN (cnn)
===================

The QM7 experiment of the spherical CNN paper
(`Cohen et al., Int. Conf. Learn. Represent. (2018)
<https://arxiv.org/abs/1801.10130>`_): atomization energies from the potentials
of the atoms sampled on spheres,
with the Table 3 network and the DeepSet readout, and the rotation
invariance of the energy.

.. example-toctree::
   :caption: Spherical CNN (cnn)

   Spherical CNN on QM7: atomization energies from spherical signals <nb/examples/cnn/s2cnn/s2cnn_qm7>

NequIP, MACE, Allegro, CACE, AIMNet2 (gnn)
==========================================

The argon train, evaluate and MD series, one pair of notebooks per model
against its reference code; a block-by-block walkthrough of the MACE
architecture; the MACE-OFF23 and MACE-MP foundation models on organic
chemistry and materials; the fine-tuning strategies (naive, readout-only,
LoRA, multi-head replay) side by side; and the published AIMNet2 models on
neutral, charged and open-shell molecules.

.. example-toctree::
   :caption: NequIP, MACE, Allegro, CACE, AIMNet2 (gnn)

   Training & testing NequIP on Argon MD data: `xnn` vs the original NequIP, step by step <nb/examples/gnn/nequip/nequip_argon_train_test>
   Argon density from MD: `xnn` vs the original NequIP <nb/examples/gnn/nequip/nequip_argon_density_md>
   Training & testing MACE on Argon MD data: `xnn` vs the original MACE, step by step <nb/examples/gnn/mace/mace_argon_train_test>
   Argon density from MD: `xnn` vs the original MACE <nb/examples/gnn/mace/mace_argon_density_md>
   Recreating the MACE architecture, block by block: original MACE **and** `xnn` <nb/examples/gnn/mace/recreate_mace_architecture>
   MACE-OFF23 in xnn: organic chemistry with a pretrained foundation model <nb/examples/gnn/mace/mace_foundation_molecules>
   Fine-tuning strategies for a foundation model: naive, LoRA and multi-head replay <nb/examples/gnn/mace/mace_finetuning_strategies>
   MACE-MP foundation models in xnn: materials properties across generations <nb/examples/gnn/mace/mace_foundation_materials>
   Training & testing Allegro on Argon MD data: `xnn` vs the original Allegro, step by step <nb/examples/gnn/allegro/allegro_argon_train_test>
   Argon density from MD: `xnn` vs the original Allegro <nb/examples/gnn/allegro/allegro_argon_density_md>
   Training & testing CACE on Argon MD data: `xnn` vs the original CACE, step by step <nb/examples/gnn/cace/cace_argon_train_test>
   Argon density from MD: `xnn` vs the original CACE <nb/examples/gnn/cace/cace_argon_density_md>
   AIMNet2 in xnn: neutral, charged and open-shell molecules with a pretrained foundation model <nb/examples/gnn/aimnet2/aimnet2_foundation_molecules>

Long-range: Latent Ewald Summation (common)
===========================================

.. example-toctree::
   :caption: Long-range: Latent Ewald Summation (common)

   Why long-range matters: molecular-dimer binding curves (`xnn` LES vs a short-range model) <nb/examples/common/les/les_molecular_dimers>

Dispersion: D4 (common)
=======================

Reproductions of the examples of the D4 paper
(`Caldeweyher et al., J. Chem. Phys. 150, 154122 (2019)
<https://doi.org/10.1063/1.5090222>`_), a benchmark against the reference
``dftd4`` with deployment through every channel, and the large-system regime
pushed to the memory limit of one GPU.

.. example-toctree::
   :caption: Dispersion: D4 (common)

   D4 in xnn: reproducing examples of the D4 paper <nb/examples/common/d4/d4_paper_examples>
   D4 in xnn vs the reference `dftd4`: accuracy, speed, and deployment with an MLIP <nb/examples/common/d4/d4_benchmark>
   Large-scale D4: liquid ethanol from rMD17 conformers, up to the memory limit <nb/examples/common/d4/d4_large_scale_ethanol>

Dispersion: D3 (common)
=======================

Reproductions of the two Grimme papers (`Grimme et al., J. Chem. Phys. 132,
154104 (2010) <https://doi.org/10.1063/1.3382344>`_;
`Grimme et al., J. Comput. Chem. 32, 1456 (2011)
<https://doi.org/10.1002/jcc.21759>`_) and a benchmark against ``simple-dftd3``
for every damping function.

.. example-toctree::
   :caption: Dispersion: D3 (common)

   D3 in xnn: reproducing examples of the two Grimme papers <nb/examples/common/d3/d3_paper_examples>
   D3 in xnn vs the reference `simple-dftd3`: accuracy, speed, and deployment with an MLIP <nb/examples/common/d3/d3_benchmark>

ET (trfnn)
================

The equivariant transformer paper
(`Thölke and De Fabritiis, Int. Conf. Learn. Represent. (2022)
<https://arxiv.org/abs/2202.02541>`_): the sizes of Tables 4 and 6, the
attention weights of section 3.1, equivariance, and the cutoff discontinuity
that TorchMD-Net 2.0 (`Pelaez et al., J. Chem. Theory Comput. 20, 4076 (2024)
<https://doi.org/10.1021/acs.jctc.4c00253>`_) removed. Then the MD17 protocol of
Table 2 on rMD17 aspirin, with and without the fix.

.. example-toctree::
   :caption: ET (trfnn)

   ET from the paper: sizes, attention, equivariance and the cutoff fix of TorchMD-Net 2.0 <nb/examples/trfnn/et/et_paper_checks>
   ET on rMD17 aspirin: the MD17 protocol of the paper at a short budget <nb/examples/trfnn/et/et_md17_train>

Equiformer (trfnn)
==================

The equivariant graph attention paper
(`Liao and Smidt, Int. Conf. Learn. Represent. (2023)
<https://arxiv.org/abs/2206.11990>`_): the sizes of Table 6 and section E.1,
the irreps and tensor-product paths, SE(3) against E(3) under reflections, the
cost of the attention variants and the cutoff. Then the MD17 protocol on rMD17
aspirin, with and without the smooth cutoff.

.. example-toctree::
   :caption: Equiformer (trfnn)

   Equiformer from the paper: sizes, equivariance, attention variants and the cutoff <nb/examples/trfnn/equiformer/equiformer_paper_checks>
   Equiformer on rMD17 aspirin: the MD17 protocol of the paper at a short budget <nb/examples/trfnn/equiformer/equiformer_md17_train>

EquiformerV2 (trfnn)
====================

The EquiformerV2 paper (`Liao et al., Int. Conf. Learn. Represent. (2024)
<https://arxiv.org/abs/2306.12059>`_): the sizes of Tables 8 and 13, the eSCN
edge frames, the sphere sampling, forces through the frames and liquid argon.
Then Equiformer and EquiformerV2 on a QM9 subset, and EquiformerV2 on OC20 S2EF
with conservative and direct forces.

.. example-toctree::
   :caption: EquiformerV2 (trfnn)

   EquiformerV2 from the paper: sizes, eSCN frames, sphere sampling, exact forces and a liquid <nb/examples/trfnn/equiformer_v2/equiformer_v2_paper_checks>
   Equiformer and EquiformerV2 on QM9 U0 at a short budget <nb/examples/trfnn/equiformer_v2/equiformer_v2_qm9_u0>
   EquiformerV2 on OC20 S2EF: adsorbates on catalyst slabs <nb/examples/trfnn/equiformer_v2/equiformer_v2_oc20_s2ef>

BAMBOO (hybnn)
===============

.. example-toctree::
   :caption: BAMBOO (hybnn)

   BAMBOO charges, energy decomposition, and deployment <nb/examples/hybnn/bamboo/bamboo_charge_analysis>
   BAMBOO on charged / polar molecular dimers: why the charge-equilibrium term matters <nb/examples/hybnn/bamboo/bamboo_dimer_electrostatics>

SpookyNet (hybnn)
==================

The SpookyNet paper (`Unke et al., Nat. Commun. 12, 7273 (2021)
<https://doi.org/10.1038/s41467-021-27504-0>`_): the building blocks
checked with untrained models and the published example model, the
electronic-state surfaces of Ag3+ / Ag3- and singlet / triplet CH2 (Fig. 4),
the dissociation of diatomics with and without nonlocal interactions
(Fig. 5), the 4G-HDNNP charge-transfer benchmarks (Table 1) and MD17
(Table 3).

.. example-toctree::
   :caption: SpookyNet (hybnn)

   SpookyNet from the paper: electronic states, orbital-like interactions and nonlocality <nb/examples/hybnn/spookynet/spookynet_paper_checks>
   Electronic states: Ag3+ / Ag3- and singlet / triplet CH2 <nb/examples/hybnn/spookynet/spookynet_electronic_states>
   Nonlocal effects: dissociation of diatomic molecules <nb/examples/hybnn/spookynet/spookynet_nonlocal>
   Nonlocal charge transfer: the 4G-HDNNP benchmarks <nb/examples/hybnn/spookynet/spookynet_charge_transfer>
   SpookyNet on MD17 <nb/examples/hybnn/spookynet/spookynet_md17>

ReaxFF / ReaxFF-nn (ffnn)
=========================

A reactive force field fit by gradient descent to rMD17 malonaldehyde, then
MD with the trained library: bond orders, EEM charges and a dissociation
scan.

.. example-toctree::
   :caption: ReaxFF / ReaxFF-nn (ffnn)

   Training ReaxFF-nn on rMD17 (malonaldehyde) <nb/examples/ffnn/reaxff/reaxff_rmd17_train_test>
   ReaxFF in action: bond orders, charges, MD and dissociation <nb/examples/ffnn/reaxff/reaxff_md_bond_orders>

OPLS / L-OPLS (ffnn)
====================

Table 1 of the OPLS-AA paper (`Jorgensen et al., J. Am. Chem. Soc. 118, 11225
(1996) <https://doi.org/10.1021/ja9621760>`_)
from relaxed dihedral scans, then the L-OPLS hexane torsion
(`Siu et al., J. Chem. Theory Comput. 8, 1459 (2012)
<https://doi.org/10.1021/ct200908r>`_) re-derived by gradient descent to
the published coefficients.

.. example-toctree::
   :caption: OPLS / L-OPLS (ffnn)

   OPLS-AA conformational energetics: reproducing Table 1 of Jorgensen et al. (1996) <nb/examples/ffnn/opls/opls_conformational_energetics>
   L-OPLS, and refitting OPLS torsions by gradient descent <nb/examples/ffnn/opls/opls_lopls_torsion_refit>

DREIDING (ffnn)
===============

Tables XI and XII of the DREIDING paper (`Mayo et al., J. Phys. Chem. 94, 8897
(1990) <https://doi.org/10.1021/j100389a010>`_)
from relaxed scans, then the generators
refit on benzene and tested for transfer to naphthalene and toluene.

.. example-toctree::
   :caption: DREIDING (ffnn)

   DREIDING conformational energetics: reproducing Tables XI and XII of the 1990 paper <nb/examples/ffnn/dreiding/dreiding_conformational_energetics>
   Retraining DREIDING: refitting the generators, and testing whether they transfer <nb/examples/ffnn/dreiding/dreiding_refit_aromatics>

AMBER (ffnn)
============

Tables 3, 4 and 11 and the dipeptide maps of Figure 2 of the ff94 paper
(`Cornell et al., J. Am. Chem. Soc. 117, 5179 (1995)
<https://doi.org/10.1021/ja00124a002>`_) from relaxed scans, the normal modes of
Tables 6 and 7 from the autograd Hessian,
then the backbone torsions refit by gradient descent.

.. example-toctree::
   :caption: AMBER (ffnn)

   AMBER ff94 conformational energetics: reproducing Tables 3, 4 and 11 and Figure 2 of Cornell et al. (1995) <nb/examples/ffnn/amber/amber_conformational_energetics>
   AMBER ff94 normal modes: reproducing Tables 6 and 7 of Cornell et al. (1995) <nb/examples/ffnn/amber/amber_normal_modes>
   Refitting the AMBER backbone torsions by gradient descent <nb/examples/ffnn/amber/amber_backbone_refit>

CHARMM22 (ffnn)
===============

The backbone geometries, conformational energies, dipeptide map and NMA
interaction energies of the CHARMM22 paper
(`MacKerell et al., J. Phys. Chem. B 102, 3586 (1998)
<https://doi.org/10.1021/jp973084f>`_), the normal modes of NMA and the
alanine dipeptide from the autograd Hessian, then the backbone torsions
refit by gradient descent.

.. example-toctree::
   :caption: CHARMM22 (ffnn)

   CHARMM22 backbone energetics: reproducing Tables 1, 2, 7, 9, 10 and 11 and Figure 2 of MacKerell et al. (1998) <nb/examples/ffnn/charmm/charmm_backbone_energetics>
   CHARMM22 normal modes: reproducing Tables 3 and 8 of MacKerell et al. (1998) <nb/examples/ffnn/charmm/charmm_normal_modes>
   Refitting the CHARMM22 backbone torsions by gradient descent <nb/examples/ffnn/charmm/charmm_backbone_refit>

AMOEBA and HIPPO (ffnn)
=======================

The water dimer and cluster energetics of the AMOEBA and HIPPO water papers
(`Ren and Ponder, J. Phys. Chem. B 107, 5933 (2003)
<https://doi.org/10.1021/jp027815+>`_;
`Rackers et al., J. Chem. Theory Comput. 17, 7056 (2021)
<https://doi.org/10.1021/acs.jctc.1c00628>`_; the ten Smith dimers with their
energy decomposition, the cyclic clusters), the monomer moments, an AMOEBA
water model refit to HIPPO reference data by gradient descent, the AMOEBA
protein force field (`Shi et al., J. Chem. Theory Comput. 9, 4046 (2013)
<https://doi.org/10.1021/ct4003702>`_) on capped
peptides with the alanine dipeptide map, and a periodic water box with
particle-mesh Ewald sums and a short microcanonical run.

.. example-toctree::
   :caption: AMOEBA and HIPPO (ffnn)

   AMOEBA water: reproducing the dimer and cluster results of Ren and Ponder (2003) <nb/examples/ffnn/amoeba/amoeba_water_clusters>
   HIPPO water: the Smith dimers and their SAPT decomposition from Rackers et al. (2021) <nb/examples/ffnn/hippo/hippo_water_dimers>
   Refitting the AMOEBA water model to HIPPO reference data <nb/examples/ffnn/amoeba/amoeba_water_refit>
   AMOEBA protein force field: typing peptides and the alanine dipeptide map of Shi et al. (2013) <nb/examples/ffnn/amoeba/amoeba_protein_backbone>
   AMOEBA and HIPPO water in a periodic box: Ewald sums, pressure and a short MD <nb/examples/ffnn/amoeba/amoeba_water_box_pme>

Semiempirical methods (sqmnn)
=============================

The tables of the PM3, PM6, PM7 and PM6-ORG papers (`Stewart, J. Comput. Chem.
10, 209 (1989) <https://doi.org/10.1002/jcc.540100208>`_;
`Stewart, J. Mol. Model. 13, 1173 (2007)
<https://doi.org/10.1007/s00894-007-0233-4>`_; `Stewart, J. Mol. Model. 19, 1
(2013) <https://doi.org/10.1007/s00894-012-1667-x>`_;
`Stewart and Stewart, J. Mol. Model. 29, 284 (2023)
<https://doi.org/10.1007/s00894-023-05695-1>`_) reproduced with the published
parameters: heats of formation, dipoles and ionization potentials, the modified
core-core terms, and the S22 / S66 interaction energies; then the parameters refit
by gradient descent and written back for MOPAC.

.. example-toctree::
   :caption: Semiempirical methods (sqmnn)

   PM3 heats of formation, dipole moments and ionization potentials (Stewart 1989) <nb/examples/sqmnn/nddo/pm3_heats_of_formation>
   PM6: the modified NDDO approximations of the 2007 paper <nb/examples/sqmnn/nddo/pm6_approximations>
   Noncovalent interactions: PM7 (S22, S66) and PM6-ORG <nb/examples/sqmnn/nddo/pm7_pm6org_noncovalent>
   Refitting NDDO parameters: hydrogen bonds, MOPAC export and force training <nb/examples/sqmnn/nddo/sqmnn_parameter_refit>

Deployment: MDI (common)
========================

A checkpoint served as an MDI engine, driven first from a minimal Python
driver and then from LAMMPS ``fix mdi/qm``.

.. example-toctree::
   :caption: Deployment: MDI (common)

   Deploying a trained model as an MDI engine: liquid argon over the MolSSI Driver Interface <nb/examples/common/deploy/mdi_argon_md>
   Driving the xnn MDI engine from LAMMPS: liquid-argon NVE and $g(r)$ <nb/examples/common/deploy/mdi_argon_lammps>

Fidelity checks
===============

Block-by-block verification of each implementation against its upstream
reference (:ref:`fidelity`), ending with a weight transplant.

.. example-toctree::
   :caption: Fidelity checks

   SchNet, block by block: the paper's equations, the reference implementation, speed and memory <nb/fidelity_checks/gnn/schnet/schnet_verification>
   3D steerable CNN, block by block: the kernel basis, equivariance and the reference implementation <nb/fidelity_checks/cnn/se3cnn/se3cnn_verification>
   Spherical CNN, block by block: harmonic analysis, correlations, equivariance and the reference implementation <nb/fidelity_checks/cnn/s2cnn/s2cnn_verification>
   DimeNet and DimeNet++ against the authors' TensorFlow implementation, including the published QM9 model <nb/fidelity_checks/gnn/dimenet/dimenet_verification>
   GemNet against the reference implementation, including the published weights <nb/fidelity_checks/gnn/gemnet/gemnet_verification>
   PaiNN, block by block: the paper's equations, the reference implementation, speed and memory <nb/fidelity_checks/gnn/painn/painn_verification>
   SpookyNet, block by block: the paper's equations and the reference implementation <nb/fidelity_checks/hybnn/spookynet/spookynet_verification>
   ET and TensorNet, block by block: the papers' equations and the TorchMD-Net reference implementation <nb/fidelity_checks/trfnn/et/torchmdnet_verification>
   Equiformer, block by block: the authors' implementation in float64 and float32, and the speed of a training step <nb/fidelity_checks/trfnn/equiformer/equiformer_verification>
   EquiformerV2 against fairchem: every variant, Wigner matrices, sphere grids, SO(2) convolutions, speed and memory <nb/fidelity_checks/trfnn/equiformer_v2/fairchem_verification>
   GMTKN55 against the paper and the maintainers' repository: counts, averages, PBEh-3c and WTMAD-2 <nb/fidelity_checks/common/data/gmtkn55_verification>
   HDNNP (2G, 3G, 4G) against the RuNNer code, including a published 4G model <nb/fidelity_checks/dnn/hdnnp/hdnnp_verification>
   ANI-1, block by block: reproducing the original implementation with `xnn` <nb/fidelity_checks/dnn/ani/ani_verification>
   PhysNet, block by block: reproducing the original implementation with `xnn` <nb/fidelity_checks/dnn/physnet/physnet_verification>
   NequIP, block by block: reproducing the original implementation with `xnn` <nb/fidelity_checks/gnn/nequip/nequip_verification>
   MACE, block by block: reproducing the original implementation with `xnn` <nb/fidelity_checks/gnn/mace/mace_verification>
   MACE foundation models: verifying `MACE.from_foundation()` against `mace-torch` <nb/fidelity_checks/gnn/mace/mace_foundation_verification>
   AIMNet2 foundation models: verifying `AIMNet2.from_foundation()` against the `aimnet` package <nb/fidelity_checks/gnn/aimnet2/aimnet2_verification>
   Allegro, block by block: reproducing the original implementation with `xnn` <nb/fidelity_checks/gnn/allegro/allegro_verification>
   CACE, block by block: reproducing the original implementation with `xnn` <nb/fidelity_checks/gnn/cace/cace_verification>
   Latent Ewald Summation (LES), block by block: reproducing the original implementation with `xnn` <nb/fidelity_checks/common/les/les_verification>
   D4 dispersion, block by block: reproducing the reference `dftd4` with `xnn` <nb/fidelity_checks/common/d4/d4_verification>
   D3 dispersion, block by block: reproducing the reference `simple-dftd3` with `xnn` <nb/fidelity_checks/common/d3/d3_verification>
   BAMBOO fidelity check: `xnn` vs the original `bytedance/bamboo`, block by block <nb/fidelity_checks/hybnn/bamboo/bamboo_verification>
   OPLS: verifying the `xnn` implementation against OpenMM and the 1996 paper <nb/fidelity_checks/ffnn/opls/opls_verification>
   DREIDING: verifying the `xnn` implementation against LAMMPS and the 1990 paper <nb/fidelity_checks/ffnn/dreiding/dreiding_verification>
   AMBER (ff94): verifying the `xnn` implementation against LAMMPS, AmberTools and the 1995 paper <nb/fidelity_checks/ffnn/amber/amber_verification>
   CHARMM22: verifying the `xnn` implementation against LAMMPS, NAMD/psfgen and the 1998 paper <nb/fidelity_checks/ffnn/charmm/charmm_verification>
   AMOEBA: verifying the `xnn` implementation against Tinker, LAMMPS and the papers <nb/fidelity_checks/ffnn/amoeba/amoeba_verification>
   HIPPO: verifying the `xnn` implementation against Tinker, LAMMPS and the 2021 paper <nb/fidelity_checks/ffnn/hippo/hippo_verification>
   PM3, PM6, PM6-ORG and PM7 against the MOPAC program <nb/fidelity_checks/sqmnn/nddo/sqmnn_verification>

Parity checks
=============

The fast paths (:ref:`howto-fast-paths`) against the reference
implementations they replace: energies, forces, stress, charges, training
gradients and timings.

.. example-toctree::
   :caption: Parity checks

   MACE and NequIP on cuEquivariance: parity with the reference <nb/parity_checks/gnn/mace/mace_cueq_parity>
   Dispersion fast paths: parity with the reference (D3, D4) <nb/parity_checks/common/dispersion/dispersion_parity>
   SchNet fast path: parity with the reference <nb/parity_checks/gnn/schnet/schnet_parity>
   PaiNN fast path: parity with the reference <nb/parity_checks/gnn/painn/painn_parity>
