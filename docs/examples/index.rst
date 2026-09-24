.. _examples:

********
Examples
********

Every example notebook in the repository's `examples/
<https://github.com/molssi-ai/xnn/tree/main/examples>`_ directory, rendered
with its executed outputs (training curves, parity plots, MD observables, and
the fidelity tables) so you can read them without running anything. To run
one yourself, see :ref:`howto-examples` for the required environments and
kernels; the notebooks below are the committed, fully executed versions.

Data
====

.. toctree::
   :maxdepth: 1
   :caption: Data

   nb/data/load_dataset_tutorial

ANI (dnn)
=========

Training ANI from scratch on rMD17, then the four published training sets
(ANI-1, ANI-1x, the coupled-cluster ANI-1ccx with its transfer-learning recipe,
and the seven-element ANI-2x set), each paired with its matching model preset.

.. toctree::
   :maxdepth: 1
   :caption: ANI (dnn)

   nb/dnn/ani/ani_rmd17_train
   nb/dnn/ani/ani1_dataset
   nb/dnn/ani/ani1x_dataset
   nb/dnn/ani/ani1ccx_dataset
   nb/dnn/ani/ani2x_dataset

PhysNet (dnn)
=============

.. toctree::
   :maxdepth: 1
   :caption: PhysNet (dnn)

   nb/dnn/physnet/physnet_argon_train_test
   nb/dnn/physnet/physnet_argon_density_md

SchNet (cnn)
============

Training the paper-architecture SchNet on its own MD17-style benchmark
(rMD17 ethanol, energies + forces through the hub), then driving
thermostat-free NVE dynamics with the trained model to demonstrate the
paper's energy-conservation-by-construction claim.

.. toctree::
   :maxdepth: 1
   :caption: SchNet (cnn)

   nb/cnn/schnet/schnet_rmd17_train
   nb/cnn/schnet/schnet_ethanol_md

NequIP, MACE, Allegro, CACE (gnn)
=================================

The shared Argon train/evaluate/deploy series (one pair of notebooks per
model), plus a block-by-block walkthrough of the MACE architecture, and
the MACE **foundation models** in action:
``mace_foundation_molecules.ipynb`` loads MACE-OFF23 with one
``MACE.from_foundation()`` call and runs the butane torsion profile
against OPLS-AA, the water dimer against CCSD(T)/CBS, and a
``Trainer`` fine-tune to a new DFT reference (rMD17 malonaldehyde);
``mace_foundation_materials.ipynb`` screens equations of state (Si, Al,
NaCl) across the MACE-MP generations (MP-0, MPA-0, OMAT-0).

.. toctree::
   :maxdepth: 1
   :caption: NequIP, MACE, Allegro, CACE (gnn)

   nb/gnn/nequip/nequip_argon_train_test
   nb/gnn/nequip/nequip_argon_density_md
   nb/gnn/mace/mace_argon_train_test
   nb/gnn/mace/mace_argon_density_md
   nb/gnn/mace/recreate_mace_architecture
   nb/gnn/mace/mace_foundation_molecules
   nb/gnn/mace/mace_foundation_materials
   nb/gnn/allegro/allegro_argon_train_test
   nb/gnn/allegro/allegro_argon_density_md
   nb/gnn/cace/cace_argon_train_test
   nb/gnn/cace/cace_argon_density_md

Long-range: Latent Ewald Summation (gnn)
========================================

.. toctree::
   :maxdepth: 1
   :caption: Long-range: Latent Ewald Summation (gnn)

   nb/gnn/les/les_molecular_dimers

Dispersion: DFT-D4 (common)
===========================

The charge-dependent DFT-D4 dispersion correction as a model-agnostic add-on.
``d4_paper_examples.ipynb`` reproduces examples of the D4 paper (Caldeweyher
*et al.* 2019): the charge-scaling function of fig 2, the charge- and
CN-dependence of the carbon and hydrogen polarizabilities of fig 5, the
atom-in-molecule polarizabilities and the molecular C6 coefficient of
(3Z)-hexen-1-yne (fig 3b), the molecular C6 coefficients of the DOSD
benchmark against the experimental dipole-oscillator-strength values (table
III), and the D4 vs D3(BJ) dispersion contributions to the S22 interaction
energies. ``d4_benchmark.ipynb`` benchmarks the xnn implementation against
the reference ``dftd4`` code (accuracy on S22 and crystals; timing on CPU and
GPU versus system size, with the upstream and MD-friendly cutoffs) and shows
D4 correcting a short-range MLIP through every deploy channel.

.. toctree::
   :maxdepth: 1
   :caption: Dispersion: DFT-D4 (common)

   nb/common/d4/d4_paper_examples
   nb/common/d4/d4_benchmark

Dispersion: DFT-D3 (common)
===========================

The geometry-dependent DFT-D3 correction (Grimme *et al.* 2010; BJ damping
Grimme, Ehrlich & Goerigk 2011) as the same model-agnostic add-on.
``d3_paper_examples.ipynb`` reproduces examples of the two papers: the
rare-gas and carbon C6 coefficients of table II and the rare-gas C9 of table
III (2010), the CN-dependent C6 curves of fig 5, the two-carbon dispersion
energy of fig 1, the zero- vs BJ-damped argon dimer of fig 1 of the 2011
paper, the three-body share of the graphene bilayer binding (table VII), and
the DOSD molecular C6 comparison of fig 6. ``d3_benchmark.ipynb`` benchmarks
xnn against the reference ``s-dftd3`` (S22, crystals, all damping functions;
timing on CPU and GPU) and deploys a D3-corrected MLIP through every channel.

.. toctree::
   :maxdepth: 1
   :caption: Dispersion: DFT-D3 (common)

   nb/common/d3/d3_paper_examples
   nb/common/d3/d3_benchmark

BAMBOO (hybrid)
===============

.. toctree::
   :maxdepth: 1
   :caption: BAMBOO (hybrid)

   nb/hybrid/bamboo/bamboo_charge_analysis
   nb/hybrid/bamboo/bamboo_dimer_electrostatics

ReaxFF / ReaxFF-nn (ffnn)
=========================

Training a reactive force field by gradient descent: a generic seed library
is fit to rMD17 malonaldehyde energies and forces through the standard
pipeline, then exported as a portable ``ffield.json``. The companion
notebook runs ASE molecular dynamics with the trained library and analyses
the reactive descriptors -- per-pair bond orders, geometry-dependent EEM
charges, and a smooth bond-dissociation scan -- including an honest look at
what equilibrium-only training data cannot constrain.

.. toctree::
   :maxdepth: 1
   :caption: ReaxFF / ReaxFF-nn (ffnn)

   nb/ffnn/reaxff/reaxff_rmd17_train_test
   nb/ffnn/reaxff/reaxff_md_bond_orders

OPLS / L-OPLS (ffnn)
====================

The fixed-topology classical force field, validated against its own
literature: ``opls_conformational_energetics.ipynb`` reproduces the relaxed
torsional energies of Table 1 of Jorgensen *et al.* (1996) with the paper's
dihedral-driver protocol (ethane, propane, butane, methanol, ethanol), and
``opls_lopls_torsion_refit.ipynb`` first compares the hexane torsion
profile of OPLS-AA and L-OPLS (Siu *et al.* 2012) and then *re-derives* the
L-OPLS ``CT-CT-CT-CT`` torsion by gradient descent — mark ``dihedral_v``
trainable, fit conformer energies, recover the published Fourier
coefficients to machine precision — before exporting the trained library
and checking NVE energy conservation.

.. toctree::
   :maxdepth: 1
   :caption: OPLS / L-OPLS (ffnn)

   nb/ffnn/opls/opls_conformational_energetics
   nb/ffnn/opls/opls_lopls_torsion_refit

DREIDING (ffnn)
===============

The rule-generated generic force field, tested against the paper that
defined it: ``dreiding_conformational_energetics.ipynb`` reproduces the
single-bond rotational barriers of Table XI (fourteen molecules, mean
difference from the paper's own calculated column ~0.01 kcal/mol) and the
butane and cyclohexane entries of Table XII, all from relaxed scans, and
shows where DREIDING's deliberate simplifications part company with
experiment. ``dreiding_refit_aromatics.ipynb`` then treats the generators
as trainable parameters: refit them on benzene alone against rMD17 PBE
forces and ask what that does to naphthalene and toluene, neither of which
was trained on — a direct test of the transferability DREIDING claims.

.. toctree::
   :maxdepth: 1
   :caption: DREIDING (ffnn)

   nb/ffnn/dreiding/dreiding_conformational_energetics
   nb/ffnn/dreiding/dreiding_refit_aromatics

Deployment: MDI (common)
========================

Serving a trained checkpoint as a `MolSSI Driver Interface
<https://github.com/MolSSI-MDI/MDI_Library>`_ engine: train on the hub argon
data, launch ``xnn mdi``, validate the wire protocol against direct
evaluation, and drive NVE molecular dynamics from a minimal Python driver.
The companion notebook then replaces the Python driver with **LAMMPS**
(``fix mdi/qm``): same engine, production driver, with LAMMPS-side
thermodynamics and a radial distribution function. The engine is model
agnostic, so the same workflow serves any family's checkpoint.

.. toctree::
   :maxdepth: 1
   :caption: Deployment: MDI (common)

   nb/deploy/mdi_argon_md
   nb/deploy/mdi_argon_lammps

Fidelity checks
===============

Block-by-block numerical verification of each xnn implementation against its
upstream reference (see :ref:`fidelity` for the summary of what matches and to
what precision). SchNet is the exception that proves the rule: a clean-room
build verified against the manuscripts' equations instead of a reference code.

.. toctree::
   :maxdepth: 1
   :caption: Fidelity checks

   nb/fidelity_checks/schnet_verification
   nb/fidelity_checks/ani_verification
   nb/fidelity_checks/physnet_verification
   nb/fidelity_checks/nequip_verification
   nb/fidelity_checks/mace_verification
   nb/fidelity_checks/mace_foundation_verification
   nb/fidelity_checks/allegro_verification
   nb/fidelity_checks/cace_verification
   nb/fidelity_checks/les_verification
   nb/fidelity_checks/d4_verification
   nb/fidelity_checks/d3_verification
   nb/fidelity_checks/bamboo_verification
   nb/fidelity_checks/opls_verification
   nb/fidelity_checks/dreiding_verification
