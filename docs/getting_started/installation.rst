.. _installation:

************
Installation
************

xnn needs Python 3.10 or later. The core package depends on ``torch`` (2.0 or
newer), ``numpy``, ``pyyaml`` and ``tqdm``. Everything else can be added based
on your need as an extra optional dependency.

From PyPI
=========
The distribution currently uses the label, ``xnns`` (we have an open `PEP 541
<https://github.com/pypi/support/issues/12316>`_ issue to be able to claim the
stale PyPI repo, ``xnn``). Regardless, the import package and the command line
use the ``xnn`` label:

.. code-block:: bash

   pip install xnns                  # core
   pip install "xnns[gnn,ase]"       # + e3nn (NequIP / MACE / Allegro) and ASE

From source
===========

To get the latest development version, clone the GitHub repository and install
with

.. code-block:: bash

   git clone https://github.com/molssi-ai/xnn.git
   cd xnn
   pip install -e ".[gnn,ase]"

Extras
======

xnn gives you the ability to install only what you need:

.. code-block:: bash

   pip install "xnns[gnn]"          # e3nn: NequIP, MACE, Allegro
   pip install "xnns[ase]"          # ASE calculator, reading structure files
   pip install "xnns[vesin]"        # cell-list neighbor lists for large systems
   pip install "xnns[hub]"          # h5py, for the ANI datasets of the data hub
   pip install "xnns[ffnn]"         # RDKit, for typing molecules with OPLS / DREIDING / AMBER
   pip install "xnns[mdi]"          # pymdi, for the MDI engine (xnn mdi)
   pip install "xnns[hydra]"        # Hydra / OmegaConf config frontend
   pip install "xnns[tensorboard]"  # TensorBoard training logs
   pip install "xnns[dev]"          # pytest
   pip install "xnns[docs]"         # Sphinx and the docs theme
   pip install "xnns[examples]"     # everything the example notebooks need
   pip install "xnns[all]"          # gnn, ase, vesin, hydra, tensorboard, dev and examples

The parity notebooks and tests compare against reference packages, which have
their own extra optional dependencies: ``ani`` (torchani), ``d3``
(simple-dftd3), ``d4`` (dftd4) and ``opls`` (OpenMM). The ``examples`` extra
pins ``mace-torch`` and ``nequip``, which pin ``e3nn==0.4.4``. xnn runs on that
version.

The fused GPU kernels (:ref:`howto-fast-paths`) are not considered extra
optional dependencies. They need a CUDA 12.6 or newer PyTorch build and are
installed manually by the user:

.. code-block:: bash

   pip install cuequivariance==0.6.1 cuequivariance-torch==0.6.1 cuequivariance-ops-torch-cu12==0.6.1

GPU environment with uv
=======================
The ``pyproject.toml`` carries a `uv <https://docs.astral.sh/uv/>`_
configuration that reproduces the environment the example notebooks were built
in (``torch 2.5.1+cu121``):

.. code-block:: bash

   uv sync --extra all

Check the installation
======================

You can check the installation by running the following Python snippet. It
prints the xnn version and the list of available models. 

.. code-block:: python

   import xnn
   from xnn.common.models import available_models

   print(xnn.__version__)
   print(available_models())
   # ['aimnet2', 'allegro', 'amber', 'amoeba', 'ani', 'bamboo', 'cace', 'charmm', 'cnn3d', 'd3', 'd4', 'dimenet', 'dimenet++', 'dreiding',
   #  'hdnnp', 'mace', 'nequip', 'opls', 'physnet', 'reaxff', 'schnet', 'se3cnn']

The e3nn-based models (NequIP, MACE, Allegro, the SE(3) steerable CNN) only
appear when the ``gnn`` extra is installed. The test suite runs with ``pytest
tests/`` (``dev`` extra).
