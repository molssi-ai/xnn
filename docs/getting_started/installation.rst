.. _installation:

************
Installation
************

xnn needs Python 3.11 or later. The core package depends on ``torch`` (2.0 or
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
   pip install "xnns[gnn,ase]"       # + e3nn (the equivariant models) and ASE

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

   pip install "xnns[gnn]"          # e3nn: NequIP, MACE, Allegro, CACE, AIMNet2, Equiformer, EquiformerV2
   pip install "xnns[ase]"          # ASE calculator, reading structure files
   pip install "xnns[vesin]"        # cell-list neighbor lists for large systems
   pip install "xnns[cpu-kernels]"  # ninja, for the optional C++ CPU kernels (D3, D4, HDNNP)
   pip install "xnns[hub]"          # h5py, for the ANI datasets of the data hub
   pip install "xnns[ffnn]"         # RDKit, for typing molecules with OPLS / DREIDING / AMBER
   pip install "xnns[mdi]"          # pymdi, for the MDI engine (xnn mdi)
   pip install "xnns[hydra]"        # Hydra / OmegaConf config frontend
   pip install "xnns[tensorboard]"  # TensorBoard training logs
   pip install "xnns[dev]"          # pytest
   pip install "xnns[docs]"         # Sphinx and the docs theme
   pip install "xnns[examples]"     # everything the example notebooks need
   pip install "xnns[all]"          # gnn, ase, vesin, cpu-kernels, hydra, tensorboard, dev and most of examples

The fidelity notebooks and tests compare against reference packages, which have
their own extra optional dependencies: ``ani`` (torchani), ``d3``
(simple-dftd3), ``d4`` (dftd4) and ``opls`` (OpenMM). The ``examples`` extra
pins the MACE, NequIP and Allegro references (``mace-torch``, ``nequip`` and
``nequip-allegro``), which pin ``e3nn==0.4.4``. xnn runs on that version.

Most fast paths (:ref:`howto-fast-paths`) run Triton kernels, which ship with
CUDA builds of PyTorch, or ``torch.compile`` (Python 3.11 or 3.12). On CPUs,
the kernels of the force fields and of AMOEBA and HIPPO also need a C++
compiler with OpenMP and the ``triton`` package. Only the MACE and NequIP
kernels need extra packages. They are not extra optional dependencies of xnn.
They need a CUDA 12.6 or newer PyTorch build and are installed manually:

.. code-block:: bash

   pip install cuequivariance==0.6.1 cuequivariance-torch==0.6.1 cuequivariance-ops-torch-cu12==0.6.1

GPU environment with uv
=======================
The ``pyproject.toml`` carries a `uv <https://docs.astral.sh/uv/>`_
configuration for a GPU development environment (``torch 2.14.1+cu126``, any
CUDA 12 driver):

.. code-block:: bash

   uv sync --extra all

torchani 2.9.0 (the ``ani`` extra) declares ``torch<=2.13``, so pip installs
an older torch with it. It runs on torch 2.14 as well, and the uv environment
uses 2.14.1.

Check the installation
======================

You can check the installation by running the following Python snippet. It
prints the xnn version and the list of available models. 

.. code-block:: python

   import xnn
   from xnn.common.models import available_models

   print(xnn.__version__)
   print(available_models())
   # ['aimnet2', 'allegro', 'amber', 'amoeba', 'ani', 'bamboo', 'byteff', 'byteff_pol', 'cace', 'charmm',
   #  'cnn3d', 'd3', 'd4', 'dimenet', 'dimenet++', 'dreiding', 'equiformer', 'equiformer_v2', 'et', 'gdml', 'gemnet', 'hdnnp',
   #  'hippo', 'mace', 'megnet', 'nequip', 'nnp1g', 'opls', 'painn', 'physnet', 'pm3', 'pm6', 'pm6_org', 'pm7', 'qet', 'qrnn',
   #  'reaxff', 's2cnn', 'schnet', 'se3cnn', 'sgdml', 'spookynet', 'tensornet']

NequIP, MACE, Allegro, CACE, AIMNet2, Equiformer and EquiformerV2 only appear
when the ``gnn`` extra is installed. The SE(3) steerable CNN needs it to build
a model. The test suite runs with ``pytest
tests/`` (``dev`` extra).
