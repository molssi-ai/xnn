.. _installation:

************
Installation
************

xnn needs Python 3.10 or later. The core package depends on ``torch``
(2.0 or newer), ``numpy``, ``pyyaml`` and ``tqdm``; everything else is an
optional extra.

From PyPI
=========
The distribution is called ``xnns``; the import package and the command line
are ``xnn``:

.. code-block:: bash

   pip install xnns                  # core
   pip install "xnns[gnn,ase]"       # + e3nn (NequIP / MACE / Allegro) and ASE

From source
===========

.. code-block:: bash

   git clone https://github.com/molssi-ai/xnn.git
   cd xnn
   pip install -e ".[gnn,ase]"

Extras
======
Install only what you need:

.. code-block:: bash

   pip install "xnns[gnn]"        # e3nn: NequIP, MACE, Allegro
   pip install "xnns[ase]"        # ASE calculator, reading structure files
   pip install "xnns[vesin]"      # cell-list neighbor lists for large systems
   pip install "xnns[hub]"        # h5py, for the ANI datasets of the data hub
   pip install "xnns[ffnn]"       # RDKit, for typing molecules with OPLS / DREIDING
   pip install "xnns[mdi]"        # pymdi, for the MDI engine (xnn mdi)
   pip install "xnns[hydra]"      # Hydra / OmegaConf config frontend
   pip install "xnns[dev]"        # pytest
   pip install "xnns[docs]"       # Sphinx and the docs theme
   pip install "xnns[examples]"   # everything the example notebooks need
   pip install "xnns[all]"        # gnn, ase, vesin, hydra, dev and examples

The parity notebooks and tests compare against reference packages, which have
their own extras: ``ani`` (torchani), ``d3`` (simple-dftd3), ``d4`` (dftd4)
and ``opls`` (OpenMM). The ``examples`` extra pins ``mace-torch`` and
``nequip``, which pin ``e3nn==0.4.4``; xnn runs on that version.

The fused GPU kernels (:ref:`howto-fast-paths`) are not an extra. They need a
CUDA 12.6 or newer PyTorch build and are installed by hand:

.. code-block:: bash

   pip install cuequivariance==0.6.1 cuequivariance-torch==0.6.1 cuequivariance-ops-torch-cu12==0.6.1

GPU environment with uv
=======================
``pyproject.toml`` carries a `uv <https://docs.astral.sh/uv/>`_ configuration
that reproduces the environment the example notebooks were built in
(``torch 2.5.1+cu121``):

.. code-block:: bash

   uv sync --extra all

Check the installation
======================

.. code-block:: python

   import xnn
   from xnn.common.models import available_models

   print(xnn.__version__)
   print(available_models())
   # ['aimnet2', 'allegro', 'ani', 'bamboo', 'cace', 'cnn3d', 'd3', 'd4', 'dimenet', 'dimenet++', 'dreiding',
   #  'hdnnp', 'mace', 'nequip', 'opls', 'physnet', 'reaxff', 'schnet', 'se3cnn']

The e3nn-based models (NequIP, MACE, Allegro, the SE(3) steerable CNN) only
appear when the ``gnn`` extra is installed. The test suite runs with ``pytest tests/`` (``dev`` extra).
