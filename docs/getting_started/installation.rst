.. _installation:

************
Installation
************

Requirements
============
xnn requires Python 3.10 or later. The core package depends only on
``torch`` (>= 2.0), ``numpy`` and ``pyyaml``; everything else is optional.

Installing from source
======================
Clone the repository and install with ``pip``:

.. code-block:: bash

   git clone https://github.com/molssi-ai/xnn.git
   cd xnn
   pip install -e .

Optional extras
===============
The optional dependencies are organized as extras, so you install only what
you need:

.. code-block:: bash

   pip install -e .              # core (torch, numpy, pyyaml)
   pip install -e ".[gnn]"      # + e3nn, for NequIP / MACE / Allegro
   pip install -e ".[ase]"      # + ASE calculator support
   pip install -e ".[hydra]"    # + Hydra / OmegaConf config frontend
   pip install -e ".[dev]"      # + pytest, for running the test suite
   pip install -e ".[examples]" # everything the example notebooks need
   pip install -e ".[all]"      # all of the above

.. list-table::
   :header-rows: 1
   :widths: 15 30 55

   * - Extra
     - Adds
     - Needed for
   * - ``gnn``
     - ``e3nn >= 0.4.4``
     - the E(3)-equivariant models: NequIP, MACE, Allegro
   * - ``ase``
     - ``ase >= 3.22``
     - :class:`~xnn.common.deploy.ase_calculator.XNNCalculator` and reading
       structure files with the ``xnn`` command line
   * - ``hydra``
     - ``hydra-core``, ``omegaconf``
     - the :func:`~xnn.common.config.loaders.from_hydra` config frontend
   * - ``dev``
     - ``pytest``
     - running ``pytest tests/``
   * - ``examples``
     - ase, e3nn, ``mace-torch``, ``nequip``, ``allegro``, matplotlib, jupyter
     - the validation notebooks in ``examples/gnn/``, which benchmark xnn
       against the reference implementations

.. note::

   The ``examples`` extra pins ``mace-torch==0.3.16`` and ``nequip==0.6.2``,
   which in turn pin ``e3nn==0.4.4``. xnn itself runs fine on that pin (all
   tests pass), so the extras can coexist in one environment.

GPU installation with uv
========================
``pyproject.toml`` carries a `uv <https://docs.astral.sh/uv/>`_ configuration
that reproduces the GPU environment the example notebooks were built in
(``torch 2.5.1+cu121`` from the PyTorch cu121 wheel index, for CUDA 12.x
drivers):

.. code-block:: bash

   uv sync --extra all

Verifying the installation
==========================
Run the test suite (requires the ``dev`` extra, and ``gnn`` for the
equivariant-model tests):

.. code-block:: bash

   pytest tests/

or check quickly from Python:

.. code-block:: python

   import xnn
   from xnn.common.models import available_models

   print(xnn.__version__)
   print(available_models())   # ['allegro', 'ani', 'bamboo', 'cace', 'hdnnp', 'mace', 'nequip', 'opls', 'physnet', 'reaxff', 'schnet']

The GNN models (NequIP, MACE, Allegro, etc.) only appear in the registry when
``e3nn`` is installed.
