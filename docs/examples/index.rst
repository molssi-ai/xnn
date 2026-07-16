.. _examples:

********
Examples
********

Every example notebook in the repository's `examples/
<https://github.com/molssi-ai/xnns/tree/main/examples>`_ directory, rendered
with its executed outputs (training curves, parity plots, MD observables, and
the fidelity tables) so you can read them without running anything. To run
one yourself, see :ref:`howto-examples` for the required environments and
kernels; the notebooks below are the committed, fully executed versions.

Data
====

.. toctree::
   :maxdepth: 1

   nb/data/load_dataset_tutorial

ANI (dnn)
=========

Training ANI from scratch on rMD17, then the four published training sets
(ANI-1, ANI-1x, the coupled-cluster ANI-1ccx with its transfer-learning recipe,
and the seven-element ANI-2x set), each paired with its matching model preset.

.. toctree::
   :maxdepth: 1

   nb/dnn/ani/ani_rmd17_train
   nb/dnn/ani/ani1_dataset
   nb/dnn/ani/ani1x_dataset
   nb/dnn/ani/ani1ccx_dataset
   nb/dnn/ani/ani2x_dataset

PhysNet (dnn)
=============

.. toctree::
   :maxdepth: 1

   nb/dnn/physnet/physnet_argon_train_test
   nb/dnn/physnet/physnet_argon_density_md

NequIP, MACE, Allegro, CACE (gnn)
=================================

The shared Argon train/evaluate/deploy series (one pair of notebooks per
model), plus a block-by-block walkthrough of the MACE architecture.

.. toctree::
   :maxdepth: 1

   nb/gnn/nequip/nequip_argon_train_test
   nb/gnn/nequip/nequip_argon_density_md
   nb/gnn/mace/mace_argon_train_test
   nb/gnn/mace/mace_argon_density_md
   nb/gnn/mace/recreate_mace_architecture
   nb/gnn/allegro/allegro_argon_train_test
   nb/gnn/allegro/allegro_argon_density_md
   nb/gnn/cace/cace_argon_train_test
   nb/gnn/cace/cace_argon_density_md

Long-range: Latent Ewald Summation (gnn)
========================================

.. toctree::
   :maxdepth: 1

   nb/gnn/les/les_molecular_dimers

BAMBOO (hybrid)
===============

.. toctree::
   :maxdepth: 1

   nb/hybrid/bamboo/bamboo_charge_analysis
   nb/hybrid/bamboo/bamboo_dimer_electrostatics

Fidelity checks
===============

Block-by-block numerical verification of each xnns implementation against its
upstream reference (see :ref:`fidelity` for the summary of what matches and to
what precision).

.. toctree::
   :maxdepth: 1

   nb/fidelity_checks/ani_verification
   nb/fidelity_checks/physnet_verification
   nb/fidelity_checks/nequip_verification
   nb/fidelity_checks/mace_verification
   nb/fidelity_checks/allegro_verification
   nb/fidelity_checks/cace_verification
   nb/fidelity_checks/les_verification
   nb/fidelity_checks/bamboo_verification
