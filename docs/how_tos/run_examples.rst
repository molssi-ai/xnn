.. _howto-examples:

*************************
Run the Example Notebooks
*************************

The example notebooks live under ``examples/`` folder and cover training,
fine-tuning and deployment and verification of every xnn model. The notebooks
are committed with their outputs and rendered in :ref:`examples`, so you can
read them there first. The details on this page is about running them on your
own machine.

Launching the examples
======================
You can run the notebooks in a Python environment with xnn and the ``examples``
extra dependency installed. The following command installs the extra dependency
and launches a Jupyter Lab server in the ``examples/`` folder. The notebooks
will run in the environment where xnn is installed, so you can use your own
Python environment or a virtual one.

.. code-block:: bash

   pip install -e ".[examples]"
   jupyter lab examples/

Every training and MD notebook loads its data through the data hub
(:ref:`howto-load-datasets`). The argon dataset is bundled and shipped with the
package. Other datasets are downloaded and cached on their first use.

What is where?
==============
- The ``examples/data`` and ``examples/models`` folders focus on the data hub
  and the model hub tutorials.
- The ``examples/gnn/<model>`` folders:
    - MACE, NequIP, Allegro and CACE offer a train/test notebook and an argon
      NPT density notebook, each compares the performance of the model against
      the reference code. 
    - MACE and AIMNet2 add their foundation-models to the experiments and
      provide a fine-tuning notebook.
    - SchNet trains on rMD17, then runs a NVE dynamics. 
    - DimeNet checks its performance against the tests in its original paper and
      trains on rMD17 while comparing its results against those of SchNet. 
    - GemNet attempts to reproduce some of the experiments in its original paper and trains GemNet-T, -Q and -dT
      on the rMD17 data.
    - PaiNN checks its paper, reproduces its MD17 tables, the ferrocene rotation
      profiles and the ethanol IR and Raman spectra. 
    - TensorNet checks its paper, trains on rMD17 aspirin, and compares its 0, 1
      and 2 layer models with ET on a QM9 subset (TorchMD-Net 2.0 Table 1).
    - LES creates several long-range binding curves for dimers.
- The ``examples/trfnn/et`` folder checks the ET paper (sizes, attention
  weights, the cutoff fix of TorchMD-Net 2.0) and trains ET on rMD17 aspirin.
- The ``examples/cnn/se3cnn`` folder provides the Tetris experiment of the SE(3)
  steerable CNN paper, then compares its performance against that of
  conventional 3D CNN potentials, trained on rMD17 under rotations.
- The ``examples/cnn/s2cnn`` folder focuses on the QM7 atomization-energy
  experiment of the spherical CNN paper.
- The ``examples/dnn`` folder contains:
    - the ANI experiments (training on rMD17 and the four published ANI
      datasets),
    - the PhysNet experiments and comparing the performance of the trained model
      against that in the original manuscript, and
    - the HDNNP generations (paper experiment checks, comparing the performance of 1G
      against 2G on rMD17, and attempting to reproduce the 2G / 3G / 4G
      benchmarks of the fourth-generation paper).
- The ``examples/hybnn`` folder focuses on:
    - experiments involving BAMBOO charges and electrostatics,
    - reproducing experiments in the SpookyNet paper and reproducing its
      electronic-state, nonlocal and charge-transfer examples as well as MD17
      dataset training.
- The ``examples/ffnn`` folder involves ReaxFF, OPLS and DREIDING trainings,
  refits and conformational energetics.
- The ``examples/common`` folder involves examples focusing on the DFT-D3 and
  DFT-D4 paper experiment reproductions, benchmarks and a large-system studies.
- The ``examples/benchmark``, and ``examples/deploy`` folders use one config
  file for scoring four models on the argon dataset. Here, a checkpoint served
  as an MDI engine, driven from Python and from LAMMPS.
- The ``examples/fidelity_checks``, and ``examples/parity_checks`` folders
  provide one notebook per model and compare them against their upstream
  counterparts, and use the fast paths against the reference implementation in
  xnn.

Extra requirements
==================
Most notebooks run with the ``examples`` extra. The exceptions are:

- The ``allegro_verification`` notebook. You need to run:

  .. code-block:: bash

    pip install "git+https://github.com/mir-group/allegro@v0.3.0"

  to get the Allegro reference code installed.

- The ``cace_verification`` and ``gnn/cace/*`` notebooks. You need to run:

  .. code-block:: bash

    pip install pip install git+https://github.com/BingqingCheng/cace

  to have the CACE reference code installed.

- The ``ani_verification`` notebook which requires the installation of ``ani``
  extra dependency.
- The ``aimnet2_verification`` notebook. You should run

  .. code-block:: bash

    pip install aimnet 

  in a separate environment (it needs torch 2.8 or newer).
- The ``dnn/physnet/*`` and ``physnet_verification`` notebooks require an
  environment with both TensorFlow and torch installed. The original PhysNet is
  cloned on demand.
- The ``opls_verification`` notebooks uses the ``opls`` extra dependency
  (and OpenMM).
- The ``common/d4/*`` and ``d4_verification`` notebooks use the ``d4`` extra
  dependency. The paper examples also need the ``rdkit`` extra dependency. The
  large-scale experiment notebook requires the ``vesin`` extra dependency to be
  able to calculate the neighborlists efficiently and a GPU with tens of GBs of
  memory.
- The ``common/d3/*`` and ``d3_verification`` notebooks use the ``d3`` extra
  dependency.
- The ``deploy/mdi_argon_md`` notebook uses the ``mdi`` extra dependency. The
  ``mdi_argon_lammps`` notebook also needs a LAMMPS installation, built with the
  MDI package, found as ``lmp`` on ``PATH`` or through ``XNN_LMP`` environment
  variables.

Important notes
===============
- The ``schnet_ethanol_md`` example notebook loads the checkpoint written by the
  ``schnet_rmd17_train`` notebook. You need to run the latter notebook first to
  generate the checkpoint.
- The MDI library initializes once per process. So, you should restart the
  kernel before re-running ``mdi_argon_md``.
- The D4 notebooks import ``dftd4`` before ``torch``. The wheel's OpenMP
  runtime returns wrong charges once torch's thread pool is active.
