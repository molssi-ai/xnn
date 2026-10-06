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

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Directory
     - Contents
   * - ``examples/data``, ``examples/models``
     - The dataset hub and the model hub (``load_dataset()``,
       ``from_pretrained()``).
   * - ``examples/gnn/<model>``
     - For MACE, NequIP, Allegro and CACE: a train/test notebook and an NPT
       argon-density MD notebook, each run with xnn and with the reference
       code side by side. MACE and AIMNet2 add foundation-model notebooks;
       SchNet trains on rMD17, then runs NVE dynamics.
   * - ``examples/dnn``
     - ANI (rMD17 training, the four published ANI data sets) and PhysNet
       (argon train/test and MD against the original).
   * - ``examples/cnn``
     - The 3D steerable CNN: the paper's Tetris experiment (steerable
       against conventional kernels under rotations) and the same
       comparison for the potentials on rMD17.
   * - ``examples/hybrid``, ``examples/ffnn``
     - BAMBOO charges and electrostatics; ReaxFF, OPLS and DREIDING
       force-field training, refits and conformational energetics.
   * - ``examples/common``
     - The DFT-D3 and DFT-D4 dispersion add-ons: paper reproductions,
       benchmarks against the reference codes, a large-system study.
   * - ``examples/deploy``
     - Serving a checkpoint as an MDI engine, driven from Python and from
       LAMMPS.
   * - ``examples/fidelity_checks``
     - One notebook per model that rebuilds it block by block against the
       upstream code and ends with a weight transplant
       (:ref:`howto-transplant`).
   * - ``examples/parity_checks``
     - The fused GPU fast paths against the reference implementations
       (:ref:`howto-fast-paths`).

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
