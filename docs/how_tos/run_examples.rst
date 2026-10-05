.. _howto-examples:

*************************
Run the Example Notebooks
*************************

The notebooks under ``examples/`` train, deploy and verify every xnn model.
They are committed with their outputs and rendered in :ref:`examples`, so
read them there first. This page is about running them yourself.

Install and launch
==================
The ``examples`` extra brings the reference packages the notebooks compare
against (``mace-torch``, ``nequip``), ASE, matplotlib and Jupyter:

.. code-block:: bash

   pip install -e ".[examples]"
   jupyter lab examples/

Every training and MD notebook loads its data through the dataset hub
(``load_dataset(...)``, see :ref:`howto-load-datasets`). The argon MD set is
bundled with the repository; the others download on first use.

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
       code side by side. MACE and AIMNet2 add foundation-model notebooks.
   * - ``examples/dnn``, ``examples/cnn``
     - ANI (rMD17 training, the four published ANI data sets), PhysNet
       (argon train/test and MD against the original) and SchNet (rMD17
       training, then NVE dynamics).
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
Most notebooks run with the ``examples`` extra alone. These need more:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Notebook
     - Needs
   * - ``fidelity_checks/allegro_verification``
     - ``pip install "git+https://github.com/mir-group/allegro@v0.3.0"``
       (the reference is not on PyPI).
   * - ``fidelity_checks/cace_verification``, ``gnn/cace/*``
     - ``pip install git+https://github.com/BingqingCheng/cace``
   * - ``fidelity_checks/ani_verification``
     - The ``ani`` extra (``torchani``).
   * - ``fidelity_checks/aimnet2_verification``
     - ``pip install aimnet``, which needs torch 2.8 or newer; use a separate
       environment.
   * - ``dnn/physnet/*``, ``fidelity_checks/physnet_verification``
     - An environment with both ``tensorflow`` and ``torch``; the notebooks
       clone the original TensorFlow PhysNet on demand.
   * - ``fidelity_checks/bamboo_verification``
     - Clones ``bytedance/bamboo`` on demand; nothing to install.
   * - ``fidelity_checks/opls_verification``
     - The ``opls`` extra (OpenMM).
   * - ``common/d4/*``, ``fidelity_checks/d4_verification``
     - The ``d4`` extra (``dftd4``). ``d4_paper_examples`` also needs
       ``rdkit``. ``d4_large_scale_ethanol`` needs the ``vesin`` extra and a
       GPU with tens of GB of memory (about 20 minutes on 80 GB).
   * - ``common/d3/*``, ``fidelity_checks/d3_verification``
     - The ``d3`` extra (``simple-dftd3``).
   * - ``deploy/mdi_argon_md``
     - The ``mdi`` extra (``pymdi``).
   * - ``deploy/mdi_argon_lammps``
     - In addition, a LAMMPS executable built with the MDI package
       (``cmake -D PKG_MDI=yes``), found as ``lmp`` on ``PATH`` or through
       ``XNN_LMP``.

Good to know
============
- ``schnet_ethanol_md`` loads the checkpoint written by
  ``schnet_rmd17_train``: run the training notebook first.
- The MDI library initializes once per process. Restart the kernel before
  re-running ``mdi_argon_md``. The LAMMPS notebook runs both codes as
  subprocesses and does not need a restart.
- The D4 notebooks import ``dftd4`` before ``torch``. The wheel's bundled
  OpenMP runtime returns wrong EEQ charges once torch's thread pool is
  active. The D3 package has no such constraint.
- ``xnn mdi`` serves a checkpoint in its own dtype unless ``--dtype`` says
  otherwise, and ``--dispersion d3|d4`` adds a correction to a checkpoint
  trained without one. See the deploy notebooks for the full command lines.
