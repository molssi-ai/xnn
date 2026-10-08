.. _howto-examples:

*************************
Run the Example Notebooks
*************************

The notebooks under ``examples/`` train, deploy and verify every model. They
are committed with their outputs and rendered in :ref:`examples`, so read
them there first; this page is about running them yourself.

Install and launch
==================

.. code-block:: bash

   pip install -e ".[examples]"
   jupyter lab examples/

Every training and MD notebook loads its data through the data hub
(:ref:`howto-load-datasets`). The argon set is bundled; the others download
on first use.

What is where
=============
- ``examples/data``, ``examples/models``: the data hub and the model hub
  tutorials.
- ``examples/gnn/<model>``: for MACE, NequIP, Allegro and CACE a train/test
  notebook and an argon NPT density notebook, each against the reference
  code; MACE and AIMNet2 add foundation-model and fine-tuning notebooks;
  SchNet trains on rMD17, then runs NVE dynamics; DimeNet checks its paper and
  trains on rMD17 against SchNet; LES shows long-range binding curves.
- ``examples/cnn/se3cnn``: the Tetris experiment of the SE(3) steerable CNN
  paper, then steerable against conventional 3D CNN potentials on rMD17
  under rotations.
- ``examples/cnn/s2cnn``: the QM7 atomization-energy experiment of the
  spherical CNN paper.
- ``examples/dnn``: ANI (rMD17 and the four published ANI datasets) and
  PhysNet (against the original).
- ``examples/hybrid``, ``examples/ffnn``: BAMBOO charges and electrostatics;
  ReaxFF, OPLS and DREIDING training, refits and conformational energetics.
- ``examples/common``: DFT-D3 and DFT-D4 paper reproductions, benchmarks and
  a large-system study.
- ``examples/benchmark``, ``examples/deploy``: one config scoring four
  models on argon; a checkpoint served as an MDI engine, driven from Python
  and from LAMMPS.
- ``examples/fidelity_checks``, ``examples/parity_checks``: one notebook per
  model against its upstream code, and the fast paths against the
  reference.

Extra requirements
==================
Most notebooks run with the ``examples`` extra. The exceptions:

- ``allegro_verification``: ``pip install "git+https://github.com/mir-group/allegro@v0.3.0"``
- ``cace_verification`` and ``gnn/cace/*``: ``pip install git+https://github.com/BingqingCheng/cace``
- ``ani_verification``: the ``ani`` extra
- ``aimnet2_verification``: ``pip install aimnet`` in a separate environment
  (it needs torch 2.8 or newer)
- ``dnn/physnet/*``, ``physnet_verification``: an environment with both
  TensorFlow and torch; the original PhysNet is cloned on demand
- ``opls_verification``: the ``opls`` extra (OpenMM)
- ``common/d4/*``, ``d4_verification``: the ``d4`` extra; the paper examples
  also need ``rdkit``, the large-scale notebook the ``vesin`` extra and a GPU
  with tens of GB of memory
- ``common/d3/*``, ``d3_verification``: the ``d3`` extra
- ``deploy/mdi_argon_md``: the ``mdi`` extra; ``mdi_argon_lammps`` also needs
  a LAMMPS built with the MDI package, found as ``lmp`` on ``PATH`` or
  through ``XNN_LMP``

Good to know
============
- ``schnet_ethanol_md`` loads the checkpoint written by ``schnet_rmd17_train``.
- The MDI library initializes once per process: restart the kernel before
  re-running ``mdi_argon_md``.
- The D4 notebooks import ``dftd4`` before ``torch``; the wheel's OpenMP
  runtime returns wrong charges once torch's thread pool is active.
