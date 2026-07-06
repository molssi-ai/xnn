.. _howto-examples:

*************************
Run the Example Notebooks
*************************

The repository ships validation notebooks that benchmark the xnns
implementations against the reference codes on real Argon MD data. Install
the ``examples`` extra first — it pulls in the reference packages
(``mace-torch``, ``nequip``, ``allegro``), ASE, matplotlib, and Jupyter:

.. code-block:: bash

   pip install -e ".[examples]"
   jupyter lab examples/

The trilogies
=============
Each equivariant model has the same three-notebook validation series under
``examples/gnn/<model>/``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Notebook
     - What it shows
   * - ``01_*_block_by_block_vs_original.ipynb``
     - Reproduces every architectural block (embedding, radial basis,
       spherical harmonics, interaction, readout, scale/shift) and checks
       each numerically against the reference implementation, ending with a
       whole-model weight transplant.
   * - ``02_*_argon_train_test.ipynb``
     - A full train/test pipeline on Argon MD data, run twice — xnns vs. the
       original code — and compared at every stage (losses, parity plots,
       errors).
   * - ``03_*_argon_density_md.ipynb``
     - Liquid-argon mass density from NPT molecular dynamics through ASE,
       comparing xnns against the reference (identical weights → ~zero
       difference, plus independently trained models).

MACE additionally has ``04_recreate_mace_architecture.ipynb`` — a
step-by-step tutorial that rebuilds the MACE architecture block by block in
*both* ``mace-torch`` and xnns, with the defining equations and architecture
figures.

The Argon dataset lives in ``examples/gnn/mace/data/`` and is shared by the
NequIP and Allegro notebooks.
