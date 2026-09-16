.. _xnn-main:

****
xnn
****

Machine-Learning Interatomic Potentials in PyTorch
==================================================
**xnn** is a library of machine-learning interatomic potentials (MLIPs) for
molecular and periodic systems, implemented in PyTorch behind a single coherent
``nn.Module`` interface. It provides faithful and self-contained implementations
of state-of-the-art equivariant open-source models such as NequIP, MACE,
Allegro, and CACE, alongside SchNet, HDNNP, ANI, PhysNet, the BAMBOO graph
equivariant transformer, and the learnable classical force fields ReaxFF /
ReaxFF-nn (reactive) and OPLS / L-OPLS (fixed topology). The key strengths
of xnn are

- all models share one data object, module interface, training loop, and
  deployment path to popular molecular dynamics packages such as ASE and LAMMPS

- distributed training and evaluation on multiple GPUs is supported out of the box

- the library is designed to be easily extensible with new models and featurizers

- a single configuration file and command-line interface enable benchmarking of
  a wide range of models on a variety of datasets

- upstream benchmark datasets download and preprocess in one line with a
  HuggingFace-style ``load_dataset()``, ready to train

- the library is accompanied by extensive documentation, tutorials, examples and
  a complete hands-on course focusing on developing and training MLIPs (see the
  `Equivariant Graph Neural Networks with e3nn: A Hands-On Course
  <https://github.com/molssi-ai/e3nn-course>`_ repository)


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
     - Complete: training, evaluation, deployment (TorchScript, LAMMPS, ASE);
       matches the `NIPS 2017 manuscript
       <https://proceedings.neurips.cc/paper/2017/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html>`_.
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
     - spherical-harmonic edges
     - Complete: training, evaluation, deployment (TorchScript, LAMMPS, ASE);
       matches `ACEsuit/mace <https://github.com/ACEsuit/mace>`_
   * - CACE
     - gnn
     - Cartesian monomial edges
     - Complete: training, evaluation, deployment (ASE only); matches
       `BingqingCheng/cace <https://github.com/BingqingCheng/cace>`_
   * - Allegro
     - gnn
     - spherical-harmonic edges
     - Complete: training, evaluation, deployment (TorchScript, LAMMPS, ASE);
       matches `mir-group/allegro <https://github.com/mir-group/allegro>`_
   * - BAMBOO
     - hybrid
     - exp-normal RBF + edge attention
     - Complete: training, evaluation, deployment (ASE only); matches
       `bytedance/bamboo <https://github.com/bytedance/bamboo>`_
   * - ReaxFF / ReaxFF-nn
     - ffnn
     - bond orders from distances (reactive)
     - Complete: training, evaluation, deployment (ASE only); implements the
       published equations, cross-checked against LAMMPS ``pair_style
       reaxff`` (see the fidelity notes)
   * - OPLS / L-OPLS
     - ffnn
     - fixed valence topology
     - Complete: training, evaluation, deployment (ASE only); matches
       `OpenMM <https://openmm.org>`_ to ~1e-7 kJ/mol and Table 1 of
       Jorgensen et al. (1996)

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
