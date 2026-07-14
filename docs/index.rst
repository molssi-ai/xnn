.. _xnns-main:

****
xnns
****

Machine-Learning Interatomic Potentials in PyTorch
==================================================
**xnns** is a library of machine-learning interatomic potentials (MLIPs) for
molecular and periodic systems, implemented in PyTorch behind a single coherent
``nn.Module`` interface. It provides faithful and self-contained implementations
of state-of-the-art equivariant open-source models such as NequIP, MACE,
Allegro, and CACE, alongside SchNet, HDNNP, ANI, PhysNet, and the BAMBOO graph
equivariant transformer. The key strengths
of xnns are

- all models share one data object, module interface, training loop, and
  deployment path to popular molecular dynamics packages such as ASE and LAMMPS

- distributed training and evaluation on multiple GPUs is supported out of the box

- the library is designed to be easily extensible with new models and featurizers

- a single configuration file and command-line command enables benchmarking of a
  wide range of models on a variety of datasets

- upstream benchmark datasets download and preprocess in one line with a
  HuggingFace-style ``load_dataset()``, ready to train

- the library is accompanied by extensive documentation, tutorials, examples and
  a complete hands-on course focusing on developing and training MLIPs (see the
  `Equivariant Graph Neural Networks with e3nn: A Hands-On Course
  <https://github.com/molssi-ai/e3nn-course>`_ repository)


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

   .. grid-item-card:: Background Information
      :margin: 0 3 0 0

      The design of xnns and the models it implements

      .. button-link:: ./background/index.html
         :color: primary
         :expand:

         To the Background Information

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
     - Under development
   * - PhysNet
     - dnn
     - exp-Gaussian RBF + attention masks
     - Complete: training, evaluation, deployment (ASE only); matches `MMunibas/PhysNet <https://github.com/MMunibas/PhysNet>`_
   * - HDNNP
     - dnn
     - radial symmetry functions (G2)
     - Under development
   * - ANI
     - dnn
     - AEV (radial + angular)
     - Under development
   * - NequIP
     - gnn
     - spherical-harmonic edges
     - Complete: training, evaluation, deployment (TorchScript, LAMMPS, ASE); matches `mir-group/nequip <https://github.com/mir-group/nequip>`_
   * - MACE
     - gnn
     - spherical-harmonic edges
     - Complete: training, evaluation, deployment (TorchScript, LAMMPS, ASE); matches `ACEsuit/mace <https://github.com/ACEsuit/mace>`_
   * - CACE
     - gnn
     - Cartesian monomial edges
     - Complete: training, evaluation, deployment (ASE only); matches `BingqingCheng/cace <https://github.com/BingqingCheng/cace>`_
   * - Allegro
     - gnn
     - spherical-harmonic edges
     - Complete: training, evaluation, deployment (TorchScript, LAMMPS, ASE); matches `mir-group/allegro <https://github.com/mir-group/allegro>`_
   * - BAMBOO
     - hybrid
     - exp-normal RBF + edge attention
     - Complete: training, evaluation, deployment (ASE only); matches `bytedance/bamboo <https://github.com/bytedance/bamboo>`_

xnns is developed by `The Molecular Sciences Software Institute (MolSSI)
<https://molssi.org>`_. Visit the `GitHub repository
<https://github.com/molssi-ai/xnns>`_ for the latest updates.

.. toctree::
   :maxdepth: 5
   :titlesonly:
   :hidden:

   getting_started/index
   how_tos/index
   user_guide/index
   developer_guide/index
   background/index
   api/index
