.. _examples:

********
Examples
********

Every example notebook in the repository's `examples/
<https://github.com/molssi-ai/xnn/tree/main/examples>`_ directory, rendered
with its executed outputs (training curves, parity plots, MD observables, and
the fidelity tables) so you can read them without running anything. To run
one yourself, see :ref:`howto-examples` for the required environments and
kernels. The pages listed below are the committed, fully executed versions;
entries marked *coming soon* are being reviewed and become links as they are
published.

Data
====

.. example-toctree::
   :caption: Data

   Loading datasets with `xnn`: the data hub & `load_dataset()` <nb/data/load_dataset_tutorial>

Models
======

The model hub: MACE foundation models, xnn-trained models and Zenodo uploads
under one ``from_pretrained()`` call, the cache layout, portability, and
sharing your own models.

.. example-toctree::
   :caption: Models

   Pre-trained models with `xnn`: the model hub & `from_pretrained()` <nb/models/pretrained_models_tutorial>

ANI (dnn)
=========

Training ANI from scratch on rMD17, then the four published training sets
(ANI-1, ANI-1x, the coupled-cluster ANI-1ccx with its transfer-learning recipe,
and the seven-element ANI-2x set), each paired with its matching model preset.

.. example-toctree::
   :caption: ANI (dnn)

   Training ANI from scratch on rMD17 (paracetamol) <nb/dnn/ani/ani_rmd17_train>
   The ANI-1 dataset in one line, and a paper-style correlation test <nb/dnn/ani/ani1_dataset>
   The ANI-1x dataset in one line: forces, active learning, and the `ani-1x` preset <nb/dnn/ani/ani1x_dataset>
   The ANI-1ccx dataset: coupled-cluster labels and transfer learning with the `ani-1ccx` preset <nb/dnn/ani/ani1ccx_dataset>
   The ANI-2x dataset: seven elements (S, F, Cl) and the `ani-2x` preset <nb/dnn/ani/ani2x_dataset>

PhysNet (dnn)
=============

.. example-toctree::
   :caption: PhysNet (dnn)

   Training & testing PhysNet on Argon MD data: `xnn` vs the original PhysNet, step by step <nb/dnn/physnet/physnet_argon_train_test>
   Argon MD with PhysNet: `xnn` NPT density + lock-step NVE against the original <nb/dnn/physnet/physnet_argon_density_md>

SchNet (cnn)
============

Training the paper-architecture SchNet on its own MD17-style benchmark
(rMD17 ethanol, energies + forces through the hub), then driving
thermostat-free NVE dynamics with the trained model to demonstrate the
paper's energy-conservation-by-construction claim.

.. example-toctree::
   :caption: SchNet (cnn)

   Training SchNet on rMD17 (ethanol) <nb/cnn/schnet/schnet_rmd17_train>
   SchNet-driven NVE dynamics: energy conservation by construction <nb/cnn/schnet/schnet_ethanol_md>

NequIP, MACE, Allegro, CACE, AIMNet2 (gnn)
==========================================

The shared Argon train/evaluate/deploy series (one pair of notebooks per
model), plus a block-by-block walkthrough of the MACE architecture, and
the MACE and AIMNet2 **foundation models** in action:
``mace_foundation_molecules.ipynb`` loads MACE-OFF23 with one
``MACE.from_foundation()`` call and runs the butane torsion profile
against OPLS-AA, the water dimer against CCSD(T)/CBS, and a
``Trainer`` fine-tune to a new DFT reference (rMD17 malonaldehyde);
``mace_finetuning_strategies.ipynb`` compares the fine-tuning strategies
(naive, readout-only, LoRA, multi-head pseudolabel replay) of the same
foundation model on 50 rMD17 ethanol structures, with the model-aware
reference-energy reestimation and the drift away from the foundation model
on other molecules;
``mace_foundation_materials.ipynb`` screens equations of state (Si, Al,
NaCl) across the MACE-MP generations (MP-0, MPA-0, OMAT-0);
``aimnet2_foundation_molecules.ipynb`` loads the published AIMNet2 models
with ``AIMNet2.from_foundation()`` and follows the paper's demonstrations:
the net charge as an input, a charged hydrogen bond (chloride-water), a
torsion profile with the four-member ensemble, geometry optimization,
dipoles from the predicted charges, the open-shell and palladium families,
and a periodic CO2 box with the damped shifted-force Coulomb sum.

.. example-toctree::
   :caption: NequIP, MACE, Allegro, CACE, AIMNet2 (gnn)

   Training & testing NequIP on Argon MD data: `xnn` vs the original NequIP, step by step <nb/gnn/nequip/nequip_argon_train_test>
   Argon density from MD: `xnn` vs the original NequIP <nb/gnn/nequip/nequip_argon_density_md>
   Training & testing MACE on Argon MD data: `xnn` vs the original MACE, step by step <nb/gnn/mace/mace_argon_train_test>
   Argon density from MD: `xnn` vs the original MACE <nb/gnn/mace/mace_argon_density_md>
   04 · Recreating the MACE architecture, block by block: original MACE **and** `xnn` <nb/gnn/mace/recreate_mace_architecture>
   MACE-OFF23 in xnn: organic chemistry with a pretrained foundation model <nb/gnn/mace/mace_foundation_molecules>
   Fine-tuning strategies for a foundation model: naive, LoRA and multi-head replay <nb/gnn/mace/mace_finetuning_strategies>
   MACE-MP foundation models in xnn: materials properties across generations <nb/gnn/mace/mace_foundation_materials>
   Training & testing Allegro on Argon MD data: `xnn` vs the original Allegro, step by step <nb/gnn/allegro/allegro_argon_train_test>
   Argon density from MD: `xnn` vs the original Allegro <nb/gnn/allegro/allegro_argon_density_md>
   Training & testing CACE on Argon MD data: `xnn` vs the original CACE, step by step <nb/gnn/cace/cace_argon_train_test>
   Argon density from MD: `xnn` vs the original CACE <nb/gnn/cace/cace_argon_density_md>
   AIMNet2 in xnn: neutral, charged and open-shell molecules with a pretrained foundation model <nb/gnn/aimnet2/aimnet2_foundation_molecules>

Long-range: Latent Ewald Summation (gnn)
========================================

.. example-toctree::
   :caption: Long-range: Latent Ewald Summation (gnn)

   Why long-range matters: molecular-dimer binding curves (`xnn` LES vs a short-range model) <nb/gnn/les/les_molecular_dimers>

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
D4 correcting a short-range MLIP through every deploy channel. ``d4_large_scale_ethanol.ipynb`` pushes the large-system regime to the
memory limit of one GPU: a MACE trained on rMD17 ethanol, corrected with D4,
evaluated on liquid-ethanol boxes packed from rMD17 conformers up to about
20 000 atoms, where the MACE itself fills an 80 GB GPU (time and memory scaling, dense vs. large regime, the cost of
a training step, the EEQ solvers, and NVE dynamics through the ASE calculator).

.. example-toctree::
   :caption: Dispersion: DFT-D4 (common)

   DFT-D4 in xnn: reproducing examples of the D4 paper <nb/common/d4/d4_paper_examples>
   DFT-D4 in xnn vs the reference `dftd4`: accuracy, speed, and deployment with an MLIP <nb/common/d4/d4_benchmark>
   Large-scale DFT-D4: liquid ethanol from rMD17 conformers, up to the memory limit <nb/common/d4/d4_large_scale_ethanol>

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

.. example-toctree::
   :caption: Dispersion: DFT-D3 (common)

   DFT-D3 in xnn: reproducing examples of the two Grimme papers <nb/common/d3/d3_paper_examples>
   DFT-D3 in xnn vs the reference `simple-dftd3`: accuracy, speed, and deployment with an MLIP <nb/common/d3/d3_benchmark>

BAMBOO (hybrid)
===============

.. example-toctree::
   :caption: BAMBOO (hybrid)

   BAMBOO charges, energy decomposition, and deployment <nb/hybrid/bamboo/bamboo_charge_analysis>
   BAMBOO on charged / polar molecular dimers: why the charge-equilibrium term matters <nb/hybrid/bamboo/bamboo_dimer_electrostatics>

ReaxFF / ReaxFF-nn (ffnn)
=========================

Training a reactive force field by gradient descent: a generic seed library
is fit to rMD17 malonaldehyde energies and forces through the standard
pipeline, then exported as a portable ``ffield.json``. The companion
notebook runs ASE molecular dynamics with the trained library and analyses
the reactive descriptors -- per-pair bond orders, geometry-dependent EEM
charges, and a smooth bond-dissociation scan -- including an honest look at
what equilibrium-only training data cannot constrain.

.. example-toctree::
   :caption: ReaxFF / ReaxFF-nn (ffnn)

   Training ReaxFF-nn on rMD17 (malonaldehyde) <nb/ffnn/reaxff/reaxff_rmd17_train_test>
   ReaxFF in action: bond orders, charges, MD and dissociation <nb/ffnn/reaxff/reaxff_md_bond_orders>

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

.. example-toctree::
   :caption: OPLS / L-OPLS (ffnn)

   OPLS-AA conformational energetics: reproducing Table 1 of Jorgensen et al. (1996) <nb/ffnn/opls/opls_conformational_energetics>
   L-OPLS, and refitting OPLS torsions by gradient descent <nb/ffnn/opls/opls_lopls_torsion_refit>

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

.. example-toctree::
   :caption: DREIDING (ffnn)

   DREIDING conformational energetics: reproducing Tables XI and XII of the 1990 paper <nb/ffnn/dreiding/dreiding_conformational_energetics>
   Retraining DREIDING: refitting the generators, and testing whether they transfer <nb/ffnn/dreiding/dreiding_refit_aromatics>

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

.. example-toctree::
   :caption: Deployment: MDI (common)

   Deploying a trained model as an MDI engine: liquid argon over the MolSSI Driver Interface <nb/deploy/mdi_argon_md>
   Driving the xnn MDI engine from LAMMPS: liquid-argon NVE and $g(r)$ <nb/deploy/mdi_argon_lammps>

Fidelity checks
===============

Block-by-block numerical verification of each xnn implementation against its
upstream reference (see :ref:`fidelity` for the summary of what matches and to
what precision). SchNet is the exception that proves the rule: a clean-room
build verified against the manuscripts' equations instead of a reference code.

.. example-toctree::
   :caption: Fidelity checks

   SchNet, block by block: verifying the `xnn` implementation against the manuscripts <nb/fidelity_checks/schnet_verification>
   ANI-1, block by block: reproducing the original implementation with `xnn` <nb/fidelity_checks/ani_verification>
   PhysNet, block by block: reproducing the original implementation with `xnn` <nb/fidelity_checks/physnet_verification>
   NequIP, block by block: reproducing the original implementation with `xnn` <nb/fidelity_checks/nequip_verification>
   MACE, block by block: reproducing the original implementation with `xnn` <nb/fidelity_checks/mace_verification>
   MACE foundation models: verifying `MACE.from_foundation()` against `mace-torch` <nb/fidelity_checks/mace_foundation_verification>
   AIMNet2 foundation models: verifying `AIMNet2.from_foundation()` against the `aimnet` package <nb/fidelity_checks/aimnet2_verification>
   Allegro, block by block: reproducing the original implementation with `xnn` <nb/fidelity_checks/allegro_verification>
   CACE, block by block: reproducing the original implementation with `xnn` <nb/fidelity_checks/cace_verification>
   Latent Ewald Summation (LES), block by block: reproducing the original implementation with `xnn` <nb/fidelity_checks/les_verification>
   DFT-D4 dispersion, block by block: reproducing the reference `dftd4` with `xnn` <nb/fidelity_checks/d4_verification>
   DFT-D3 dispersion, block by block: reproducing the reference `simple-dftd3` with `xnn` <nb/fidelity_checks/d3_verification>
   BAMBOO fidelity check: `xnn` vs the original `bytedance/bamboo`, block by block <nb/fidelity_checks/bamboo_verification>
   OPLS: verifying the `xnn` implementation against OpenMM and the 1996 paper <nb/fidelity_checks/opls_verification>
   DREIDING: verifying the `xnn` implementation against LAMMPS and the 1990 paper <nb/fidelity_checks/dreiding_verification>

Parity checks
=============

The fast paths (fused GPU kernels behind ``use_fast``, see
:ref:`howto-fast-paths`) against the reference implementations they replace,
with the same weights: energies, forces, stress, charges and training
gradients, and the time of each, on the systems the models are served on.

.. example-toctree::
   :caption: Parity checks

   MACE and NequIP on cuEquivariance: parity with the reference <nb/parity_checks/mace_cueq_parity>
   Dispersion fast paths: parity with the reference (DFT-D3, DFT-D4) <nb/parity_checks/dispersion_parity>
