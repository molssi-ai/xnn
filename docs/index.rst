.. _xnn-main:

****
xnn
****

Machine-Learning Interatomic Potentials in PyTorch
==================================================
**xnn** is a library of machine-learning interatomic potentials (MLIPs) for
molecular and periodic systems. 

The *x* in **xnn** stands for the architecture family: graph neural networks
(``gnn``), dense neural networks (``dnn``), convolutional neural networks on
voxel grids (``cnn``), classical force field neural networks (``ffnn``) and
hybrid models (``hybrid``). Every model is implemented using the same PyTorch
``nn.Module`` interface. Thus, one data object, one trainer and one deployment
path serve all model families.

xnn offers several benefits out-of-the-box:

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
   :widths: 34 66

   * - Family
     - Models
   * - Graph neural networks (``gnn``)
     - MACE, NequIP, Allegro, CACE, AIMNet2, SchNet, DimeNet, DimeNet++, PaiNN
   * - Dense neural networks (``dnn``)
     - ANI (ANI-1, ANI-1x, ANI-1ccx, ANI-2x), PhysNet, HDNNP (generations 1 to 4)
   * - Convolutional neural networks (``cnn``)
     - SE(3) steerable CNN, 3D CNN, spherical CNN
   * - Classical force fields (``ffnn``)
     - ReaxFF / ReaxFF-nn, OPLS-AA / L-OPLS, DREIDING / X6
   * - Hybrid neural networks (``hybrid``)
     - BAMBOO, SpookyNet
   * - Add-ons for any model (``common``)
     - LES long-range electrostatics, DFT-D3 and DFT-D4 dispersion

Each model matches its reference code or paper to round-off
(:ref:`fidelity`); the MACE and AIMNet2 foundation models load through the
hub. Every model trains, evaluates and runs under ASE. SchNet, DimeNet, PaiNN, SpookyNet,
NequIP, MACE, Allegro and AIMNet2 also export to TorchScript for LAMMPS, and any
checkpoint serves as an MDI engine (:ref:`deployment`).

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
