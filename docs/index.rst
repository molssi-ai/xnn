.. _xnn-main:

****
xnn
****

Machine-Learning Interatomic Potentials in PyTorch
==================================================
**xnn** is a library of machine-learning interatomic potentials (MLIPs) for
molecular and periodic systems. Every model, from the equivariant graph
networks MACE, NequIP, Allegro and CACE to the classical force fields ReaxFF,
OPLS and DREIDING, sits behind the same ``nn.Module`` interface, so one data
object, one trainer and one deployment path serve all of them.

- **Faithful implementations.** Each model reproduces its reference code or
  paper to round-off, verified block by block (:ref:`fidelity`).
- **One pipeline.** Shared data object, config schema, trainer and
  deployment to ASE, LAMMPS and the MolSSI Driver Interface.
- **Pre-trained and benchmark data in one line.** ``from_pretrained()``
  loads the MACE and AIMNet2 foundation models and your own checkpoints;
  ``load_dataset()`` downloads and converts standard datasets.
- **Fine-tuning, benchmarking and multi-GPU training** from a config file
  and the ``xnn`` command.
- **Physics add-ons for any model.** Latent Ewald long-range electrostatics
  and DFT-D3 / DFT-D4 dispersion wrap every potential.
- **Fast paths.** Fused GPU kernels for the expensive blocks, switched on
  per run without changing the model.

.. grid:: 1 1 2 2

   .. grid-item-card:: Getting Started
      :margin: 0 3 0 0

      Installing xnn and a first training run

      .. button-link:: ./getting_started/index.html
         :color: primary
         :expand:

         To the Getting Started Guide

   .. grid-item-card:: How-To Guides
      :margin: 0 3 0 0

      Recipes for accomplishing common tasks

      .. button-link:: ./how_tos/index.html
         :color: primary
         :expand:

         To the How-To Guides

   .. grid-item-card:: Examples
      :margin: 0 3 0 0

      Executed example notebooks, rendered with their outputs

      .. button-link:: ./examples/index.html
         :color: primary
         :expand:

         To the Examples

   .. grid-item-card:: User Guide
      :margin: 0 3 0 0

      Reference information for using xnn

      .. button-link:: ./user_guide/index.html
         :color: primary
         :expand:

         To the User Guide

   .. grid-item-card:: Developer Guide
      :margin: 0 3 0 0

      Extending xnn with new models and featurizers

      .. button-link:: ./developer_guide/index.html
         :color: primary
         :expand:

         To the Developer Guide

   .. grid-item-card:: Background Information
      :margin: 0 3 0 0

      The design of xnn and the models it implements

      .. button-link:: ./background/index.html
         :color: primary
         :expand:

         To the Background Information

   .. grid-item-card:: API Reference
      :margin: 0 3 0 0

      Documentation of the xnn Python API

      .. button-link:: ./api/index.html
         :color: primary
         :expand:

         To the API Reference

   .. grid-item-card:: Equivariant GNNs with e3nn
      :margin: 0 3 0 0

      A hands-on course for developing equivariant GNN interatomic potentials

      .. button-link:: https://github.com/molssi-ai/e3nn-course
         :color: primary
         :expand:

         To the Equivariant GNN Course

Models at a glance
==================

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Model
     - Family
     - Featurizer
     - State
   * - SchNet
     - gnn
     - Gaussian RBF
     - Complete: training, evaluation, deployment (TorchScript, LAMMPS, ASE);
       matches the `NIPS 2017 manuscript
       <https://proceedings.neurips.cc/paper/2017/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html>`_.
   * - 3D steerable CNN
     - cnn
     - voxelized environments (species density grids)
     - Complete: training, evaluation, deployment (ASE only); matches the
       `NeurIPS 2018 manuscript <https://arxiv.org/abs/1807.02547>`_ (Weiler
       et al.), with its Tetris experiment reproduced
   * - 3D CNN
     - cnn
     - voxelized environments (species density grids)
     - Complete: training, evaluation, deployment (ASE only); the
       non-equivariant control of the same paper
   * - PhysNet
     - dnn
     - exp-Gaussian RBF + attention masks
     - Complete: training, evaluation, deployment (ASE only); matches
       `MMunibas/PhysNet <https://github.com/MMunibas/PhysNet>`_
   * - HDNNP
     - dnn
     - radial symmetry functions (G2)
     - Under development
   * - ANI
     - dnn
     - AEV (radial + angular symmetry functions)
     - Complete: training, evaluation, deployment (ASE only); matches
       `aiqm/torchani <https://github.com/aiqm/torchani>`_
   * - NequIP
     - gnn
     - spherical-harmonic edges
     - Complete: training, evaluation, deployment (TorchScript, LAMMPS, ASE);
       matches `mir-group/nequip <https://github.com/mir-group/nequip>`_
   * - MACE
     - gnn
     - `ACEsuit/mace <https://github.com/ACEsuit/mace>`_; loads the MACE-MP /
       MACE-OFF foundation models
   * - NequIP
     - gnn
     - `mir-group/nequip <https://github.com/mir-group/nequip>`_
   * - Allegro
     - gnn
     - `mir-group/allegro <https://github.com/mir-group/allegro>`_
   * - CACE
     - gnn
     - `BingqingCheng/cace <https://github.com/BingqingCheng/cace>`_
   * - AIMNet2
     - gnn
     - `isayevlab/aimnetcentral <https://github.com/isayevlab/aimnetcentral>`_;
       loads the published AIMNet2 models
   * - SchNet
     - gnn
     - Schütt *et al.*, NIPS 2017 (clean-room build from the paper)
   * - 3D steerable CNN
     - cnn
     - Weiler *et al.*, NeurIPS 2018; with a conventional 3D CNN baseline
   * - ANI
     - dnn
     - `aiqm/torchani <https://github.com/aiqm/torchani>`_; ANI-1, ANI-1x,
       ANI-1ccx and ANI-2x presets
   * - PhysNet
     - dnn
     - `MMunibas/PhysNet <https://github.com/MMunibas/PhysNet>`_
   * - HDNNP
     - dnn
     - Behler and Parrinello 2007 (under development)
   * - BAMBOO
     - hybrid
     - `bytedance/bamboo <https://github.com/bytedance/bamboo>`_
   * - ReaxFF / ReaxFF-nn
     - ffnn
     - the published equations, cross-checked against LAMMPS ``pair_style reaxff``
   * - OPLS / L-OPLS
     - ffnn
     - Jorgensen *et al.* 1996; matches `OpenMM <https://openmm.org>`_
   * - DREIDING / X6
     - ffnn
     - Mayo *et al.* 1990; matches `LAMMPS <https://lammps.org>`_

Every model trains, evaluates and runs under ASE. SchNet, NequIP, MACE and
Allegro also export to TorchScript for LAMMPS, and any checkpoint serves as
an MDI engine (:ref:`deployment`). The add-ons LES, DFT-D3 and DFT-D4 wrap
any of them (:ref:`models`).

xnn is developed by `The Molecular Sciences Software Institute (MolSSI)
<https://molssi.org>`_. Visit the `GitHub repository
<https://github.com/molssi-ai/xnn>`_ for the latest updates.

.. toctree::
   :maxdepth: 5
   :titlesonly:
   :hidden:

   getting_started/index
   how_tos/index
   examples/index
   user_guide/index
   developer_guide/index
   background/index
   api/index
