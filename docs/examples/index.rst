.. _examples:

********
Examples
********

Every notebook of the repository's `examples/
<https://github.com/molssi-ai/xnn/tree/main/examples>`_ directory, rendered
with its executed outputs. To run one yourself, see :ref:`howto-examples`
for the environments. Entries marked *coming soon* are being reviewed and
become links as they are published.

Data
====

The data hub: ``load_dataset()``, splits, units, caching.

.. example-toctree::
   :caption: Data

   Loading datasets with `xnn`: the data hub & `load_dataset()` <nb/data/load_dataset_tutorial>

Models
======

The model hub: foundation models, xnn-trained models and Zenodo uploads
under one ``from_pretrained()`` call, the cache layout and sharing your own.

.. example-toctree::
   :caption: Models

   Pre-trained models with `xnn`: the model hub & `from_pretrained()` <nb/models/pretrained_models_tutorial>

ANI (dnn)
=========

ANI from scratch on rMD17, then the four published ANI datasets, each paired
with its model preset.

.. example-toctree::
   :caption: ANI (dnn)

   Training ANI from scratch on rMD17 (paracetamol) <nb/dnn/ani/ani_rmd17_train>
   The ANI-1 dataset in one line, and a paper-style correlation test <nb/dnn/ani/ani1_dataset>
   The ANI-1x dataset in one line: forces, active learning, and the `ani-1x` preset <nb/dnn/ani/ani1x_dataset>
   The ANI-1ccx dataset: coupled-cluster labels and transfer learning with the `ani-1ccx` preset <nb/dnn/ani/ani1ccx_dataset>
   The ANI-2x dataset: seven elements (S, F, Cl) and the `ani-2x` preset <nb/dnn/ani/ani2x_dataset>

PhysNet (dnn)
=============

Training and MD against the original TensorFlow PhysNet.

.. example-toctree::
   :caption: PhysNet (dnn)

   Training & testing PhysNet on Argon MD data: `xnn` vs the original PhysNet, step by step <nb/dnn/physnet/physnet_argon_train_test>
   Argon MD with PhysNet: `xnn` NPT density + lock-step NVE against the original <nb/dnn/physnet/physnet_argon_density_md>

SchNet (cnn)
============

The paper architecture trained on rMD17 ethanol, then thermostat-free NVE
dynamics to show energy conservation by construction.

.. example-toctree::
   :caption: SchNet (cnn)

   Training SchNet on rMD17 (ethanol) <nb/cnn/schnet/schnet_rmd17_train>
   SchNet-driven NVE dynamics: energy conservation by construction <nb/cnn/schnet/schnet_ethanol_md>

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

Reproductions of the D4 paper's examples, a benchmark against the reference
``dftd4`` with deployment through every channel, and the large-system regime
pushed to the memory limit of one GPU.

.. example-toctree::
   :caption: Dispersion: DFT-D4 (common)

   DFT-D4 in xnn: reproducing examples of the D4 paper <nb/common/d4/d4_paper_examples>
   DFT-D4 in xnn vs the reference `dftd4`: accuracy, speed, and deployment with an MLIP <nb/common/d4/d4_benchmark>
   Large-scale DFT-D4: liquid ethanol from rMD17 conformers, up to the memory limit <nb/common/d4/d4_large_scale_ethanol>

Dispersion: DFT-D3 (common)
===========================

Reproductions of the two Grimme papers and a benchmark against
``simple-dftd3`` for every damping function.

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

A reactive force field fit by gradient descent to rMD17 malonaldehyde, then
MD with the trained library: bond orders, EEM charges and a dissociation
scan.

.. example-toctree::
   :caption: ReaxFF / ReaxFF-nn (ffnn)

   Training ReaxFF-nn on rMD17 (malonaldehyde) <nb/ffnn/reaxff/reaxff_rmd17_train_test>
   ReaxFF in action: bond orders, charges, MD and dissociation <nb/ffnn/reaxff/reaxff_md_bond_orders>

OPLS / L-OPLS (ffnn)
====================

Table 1 of the 1996 paper from relaxed dihedral scans, then the L-OPLS
hexane torsion re-derived by gradient descent to the published coefficients.

.. example-toctree::
   :caption: OPLS / L-OPLS (ffnn)

   OPLS-AA conformational energetics: reproducing Table 1 of Jorgensen et al. (1996) <nb/ffnn/opls/opls_conformational_energetics>
   L-OPLS, and refitting OPLS torsions by gradient descent <nb/ffnn/opls/opls_lopls_torsion_refit>

DREIDING (ffnn)
===============

Tables XI and XII of the 1990 paper from relaxed scans, then the generators
refit on benzene and tested for transfer to naphthalene and toluene.

.. example-toctree::
   :caption: DREIDING (ffnn)

   DREIDING conformational energetics: reproducing Tables XI and XII of the 1990 paper <nb/ffnn/dreiding/dreiding_conformational_energetics>
   Retraining DREIDING: refitting the generators, and testing whether they transfer <nb/ffnn/dreiding/dreiding_refit_aromatics>

Deployment: MDI (common)
========================

A checkpoint served as an MDI engine, driven first from a minimal Python
driver and then from LAMMPS ``fix mdi/qm``.

.. example-toctree::
   :caption: Deployment: MDI (common)

   Deploying a trained model as an MDI engine: liquid argon over the MolSSI Driver Interface <nb/deploy/mdi_argon_md>
   Driving the xnn MDI engine from LAMMPS: liquid-argon NVE and $g(r)$ <nb/deploy/mdi_argon_lammps>

Fidelity checks
===============

Block-by-block verification of each implementation against its upstream
reference (:ref:`fidelity`), ending with a weight transplant.

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

The fast paths (:ref:`howto-fast-paths`) against the reference
implementations they replace: energies, forces, stress, charges, training
gradients and timings.

.. example-toctree::
   :caption: Parity checks

   MACE and NequIP on cuEquivariance: parity with the reference <nb/parity_checks/mace_cueq_parity>
   Dispersion fast paths: parity with the reference (DFT-D3, DFT-D4) <nb/parity_checks/dispersion_parity>
