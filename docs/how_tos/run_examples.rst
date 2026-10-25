.. _howto-examples:

*************************
Run the Example Notebooks
*************************

The notebooks live under the ``notebooks/`` folder and cover training,
fine-tuning, deployment and verification of every xnn model. The notebooks
are committed with their outputs and rendered in :ref:`examples`, so you can
read them there first. This page is about running them on your own machine.

Launching the examples
======================
You can run the notebooks in a Python environment with xnn and the ``examples``
extra dependency installed. The following command installs the extra dependency
and launches a Jupyter Lab server in the ``notebooks/`` folder. The notebooks
will run in the environment where xnn is installed, so you can use your own
Python environment or a virtual one.

.. code-block:: bash

   pip install -e ".[examples]"
   jupyter lab notebooks/

Every training and MD notebook loads its data through the data hub
(:ref:`howto-load-datasets`). The argon dataset is bundled and shipped with the
package. Other datasets are downloaded and cached on their first use.

What is where?
==============
The three folders under ``notebooks/`` follow the package layout, one
``<family>/<model>/`` folder per model, so the notebooks of a model sit at
matching paths:

.. code-block:: text

   notebooks/
     examples/           training, MD, paper reproductions, tutorials
       common/           data, models (hubs), deploy, benchmark, d3, d4, les
       gnn/  dnn/  cnn/  ffnn/  hybnn/  sqmnn/  trfnn/
     fidelity_checks/    each model against its reference code, block by block
       common/  gnn/  dnn/  cnn/  ffnn/  hybnn/  sqmnn/  trfnn/
     parity_checks/      the fast paths against the reference implementation
       common/dispersion/  gnn/mace/

For example, MACE has ``examples/gnn/mace/``, ``fidelity_checks/gnn/mace/`` and
``parity_checks/gnn/mace/``. In ``examples/``:

- ``common/data`` and ``common/models`` hold the data hub and model hub
  tutorials.
- ``gnn/<model>``:
    - MACE, NequIP, Allegro and CACE offer a train/test notebook and an argon
      NPT density notebook, each compares the performance of the model against
      the reference code.
    - MACE and AIMNet2 add their foundation-models to the experiments and
      provide a fine-tuning notebook.
    - SchNet trains on rMD17, then runs a NVE dynamics.
    - DimeNet checks its performance against the tests in its original paper and
      trains on rMD17 while comparing its results against those of SchNet.
    - GemNet attempts to reproduce some of the experiments in its original paper
      and trains GemNet-T, -Q and -dT on the rMD17 data.
    - PaiNN checks its paper, reproduces its MD17 tables, the ferrocene rotation
      profiles and the ethanol IR and Raman spectra.
    - TensorNet checks its paper, trains on rMD17 aspirin, and compares its 0, 1
      and 2 layer models with ET on a QM9 subset (TorchMD-Net 2.0 Table 1).
- ``trfnn/et`` checks the ET paper (sizes, attention weights, the cutoff fix of
  TorchMD-Net 2.0) and trains ET on rMD17 aspirin.
- ``trfnn/equiformer`` checks the Equiformer paper (sizes, reflections, the
  attention variants) and trains Equiformer on rMD17 aspirin.
- ``trfnn/equiformer_v2`` checks the EquiformerV2 paper (sizes, edge frames,
  sphere sampling, liquid argon), compares Equiformer and EquiformerV2 on a QM9
  subset, and trains EquiformerV2 on OC20 S2EF.
- ``cnn/se3cnn`` provides the Tetris experiment of the SE(3) steerable CNN
  paper, then compares its performance against that of conventional 3D CNN
  potentials, trained on rMD17 under rotations.
- ``cnn/s2cnn`` focuses on the QM7 atomization-energy experiment of the
  spherical CNN paper.
- ``dnn`` contains:
    - the ANI experiments (training on rMD17 and the four published ANI
      datasets),
    - the PhysNet experiments and comparing the performance of the trained model
      against that in the original manuscript, and
    - the HDNNP generations (paper experiment checks, comparing the performance
      of 1G against 2G on rMD17, and attempting to reproduce the 2G / 3G / 4G
      benchmarks of the fourth-generation paper).
- ``hybnn`` focuses on:
    - experiments involving BAMBOO charges and electrostatics,
    - reproducing experiments in the SpookyNet paper and reproducing its
      electronic-state, nonlocal and charge-transfer examples as well as MD17
      dataset training.
- ``ffnn`` involves ReaxFF, OPLS, DREIDING, AMBER, CHARMM, AMOEBA and HIPPO
  trainings, refits, conformational energetics, water cluster energetics and
  normal modes.
- ``sqmnn/nddo`` reproduces the PM3, PM6, PM6-ORG and PM7 papers and refits the
  NDDO parameters.
- ``common/d3`` and ``common/d4`` hold the D3 and D4 paper experiment
  reproductions, benchmarks and a large-system study.
- ``common/les`` creates long-range binding curves for charged and polar
  dimers.
- ``common/benchmark`` scores four models on the argon dataset with one config
  file, and ``common/deploy`` serves a checkpoint as an MDI engine, driven from
  Python and from LAMMPS.

Extra requirements
==================
Most notebooks run with the ``examples`` extra. The exceptions are:

- The ``allegro_verification`` notebook. You need to run:

  .. code-block:: bash

    pip install "git+https://github.com/mir-group/allegro@v0.3.0"

  to get the Allegro reference code installed.

- The ``cace_verification`` and ``examples/gnn/cace/*`` notebooks. You need to run:

  .. code-block:: bash

    pip install pip install git+https://github.com/BingqingCheng/cace

  to have the CACE reference code installed.

- The ``ani_verification`` notebook which requires the installation of ``ani``
  extra dependency.
- The ``aimnet2_verification`` notebook. You should run

  .. code-block:: bash

    pip install aimnet 

  in a separate environment (it needs torch 2.8 or newer).
- The ``examples/dnn/physnet/*`` and ``physnet_verification`` notebooks require an
  environment with both TensorFlow and torch installed. The original PhysNet is
  cloned on demand.
- The ``opls_verification`` notebooks uses the ``opls`` extra dependency
  (and OpenMM).
- The ``amber_verification`` notebook needs the LAMMPS Python module
  (``import lammps``, with the MOLECULE and EXTRA-MOLECULE packages); its
  comparison with the AMBER programs runs when ``tleap`` and ``sander`` are
  on the path (or reachable through a wrapper script named by
  ``XNN_AMBER_WRAPPER``) and is skipped otherwise.
- The ``charmm_verification`` notebook needs the LAMMPS Python module too;
  its comparison with ``psfgen`` and NAMD runs when VMD (text mode) and NAMD
  are reachable through the wrapper scripts named by ``XNN_VMD_WRAPPER`` and
  ``XNN_NAMD_WRAPPER`` and the CHARMM files by ``XNN_CHARMM_TOPPAR``, and is
  skipped otherwise. The ``charmm_backbone_energetics`` notebook uses scipy.
- The ``amoeba_verification`` and ``hippo_verification`` notebooks compare
  with Tinker when its ``analyze`` and ``testgrad`` programs are on the path,
  and with the LAMMPS AMOEBA package when the ``lammps`` Python module (with
  the AMOEBA package) is importable and ``XNN_TINKER2LMP`` names the
  ``tinker2lmp.py`` converter of the LAMMPS distribution; each comparison is
  skipped otherwise. The AMOEBA protein example uses RDKit.
- The ``examples/common/d4/*`` and ``d4_verification`` notebooks use the ``d4`` extra
  dependency. The paper examples also need the ``rdkit`` extra dependency. The
  large-scale experiment notebook requires the ``vesin`` extra dependency to be
  able to calculate the neighborlists efficiently and a GPU with tens of GBs of
  memory.
- The ``examples/common/d3/*`` and ``d3_verification`` notebooks use the ``d3`` extra
  dependency.
- The ``examples/common/deploy/mdi_argon_md`` notebook uses the ``mdi`` extra dependency. The
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
