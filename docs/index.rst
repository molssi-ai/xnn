.. _xnns-main:

****
xnns
****

Machine-Learning Interatomic Potentials in PyTorch
==================================================
**xnns** is a library of machine-learning interatomic potentials for molecular
and periodic systems, implemented in PyTorch behind a single coherent
``nn.Module`` interface. It provides faithful, self-contained implementations
of state-of-the-art equivariant models — NequIP, MACE, and Allegro — alongside
SchNet, HDNNP, and ANI, all sharing one data object, one training loop, and one
deployment path to ASE and LAMMPS.

.. grid:: 1 1 2 2

   .. grid-item-card:: Getting Started
      :margin: 0 3 0 0

      Installing xnns and a first training run

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

   .. grid-item-card:: Background Information
      :margin: 0 3 0 0

      The design of xnns and the models it implements

      .. button-link:: ./background/index.html
         :color: primary
         :expand:

         To the Background Information

   .. grid-item-card:: User Guide
      :margin: 0 3 0 0

      Reference information for using xnns

      .. button-link:: ./user_guide/index.html
         :color: primary
         :expand:

         To the User Guide

   .. grid-item-card:: Developer Guide
      :margin: 0 3 0 0

      Extending xnns with new models and featurizers

      .. button-link:: ./developer_guide/index.html
         :color: primary
         :expand:

         To the Developer Guide

   .. grid-item-card:: API Reference
      :margin: 0 3 0 0

      Documentation of the xnns Python API

      .. button-link:: ./api/index.html
         :color: primary
         :expand:

         To the API Reference

Models at a glance
==================

.. list-table::
   :header-rows: 1
   :widths: 12 10 34 44

   * - Model
     - Family
     - Featurizer
     - State
   * - SchNet
     - cnn
     - Gaussian RBF
     - full; trainable; TorchScript/LAMMPS-deployable
   * - HDNNP
     - dnn
     - radial symmetry functions (G2)
     - full; trainable
   * - ANI
     - dnn
     - AEV (radial + angular)
     - full; trainable
   * - NequIP
     - gnn
     - spherical-harmonic edges
     - faithful; matches `mir-group/nequip <https://github.com/mir-group/nequip>`_; deployable
   * - MACE
     - gnn
     - spherical-harmonic edges
     - faithful; matches `ACEsuit/mace <https://github.com/ACEsuit/mace>`_; deployable
   * - Allegro
     - gnn
     - spherical-harmonic edges
     - faithful; matches `mir-group/allegro <https://github.com/mir-group/allegro>`_; deployable

xnns is developed by `The Molecular Sciences Software Institute (MolSSI)
<https://molssi.org>`_. Visit the `GitHub repository
<https://github.com/molssi-ai/xnns>`_ for the latest updates, and check out
the MolSSI `Guidelines and Best Practices
<https://molssi-ai.github.io/molssi-ai-guidelines/index.html>`_ for data
science, machine learning and high-performance computing.

.. toctree::
   :maxdepth: 5
   :titlesonly:
   :hidden:

   getting_started/index
   how_tos/index
   background/index
   user_guide/index
   developer_guide/index
   api/index
