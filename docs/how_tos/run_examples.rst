.. _howto-examples:

*************************
Run the Example Notebooks
*************************

The repository ships validation notebooks that check the xnns
implementations against the reference codes on real Argon MD data. Install
the ``examples`` extra first — it pulls in the reference packages
(``mace-torch``, ``nequip``, ``allegro``), ASE, matplotlib, and Jupyter:

.. code-block:: bash

   pip install -e ".[examples]"
   jupyter lab examples/

The trilogies
=============
Each GNN model (MACE, NequIP, Allegro, CACE) has the same validation series.
The block-by-block fidelity check is collected under
``examples/fidelity_checks/<model>_verification.ipynb`` (across all families);
the train/test and MD notebooks stay under ``examples/gnn/<model>/``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Notebook
     - What it shows
   * - ``fidelity_checks/<model>_verification.ipynb``
     - Reproduces every architectural block (embedding, radial basis,
       spherical harmonics, interaction, readout, scale/shift) and checks
       each numerically against the reference implementation, ending with a
       whole-model weight transplant.
   * - ``gnn/<model>/<model>_argon_train_test.ipynb``
     - A full train/test pipeline on Argon MD data, run twice — xnns vs. the
       original code — and compared at every stage (losses, parity plots,
       errors).
   * - ``gnn/<model>/<model>_argon_density_md.ipynb``
     - Liquid-argon mass density from NPT molecular dynamics through ASE,
       comparing xnns against the reference (identical weights → ~zero
       difference, plus independently trained models).

MACE additionally has ``recreate_mace_architecture.ipynb`` — a
step-by-step tutorial that rebuilds the MACE architecture block by block in
*both* ``mace-torch`` and xnns, with the defining equations and architecture
figures.

The Argon dataset lives in ``datasets/argon_md/`` (at the repository root) and
is shared by the MACE, NequIP, Allegro, CACE, and PhysNet notebooks. The CACE series compares against
the original ``cace`` package
(``pip install git+https://github.com/BingqingCheng/cace``); the PhysNet
series (``examples/dnn/physnet/``) compares against the original
**TensorFlow** implementation and needs a venv with both ``tensorflow`` and
``torch`` (the notebooks clone MMunibas/PhysNet on demand).

BAMBOO (the ``hybrid`` family) has its block-by-block fidelity check in
``examples/fidelity_checks/bamboo_verification.ipynb`` (it clones
bytedance/bamboo on demand and transplants the weights, matching every GET
layer and the charge/energy outputs to machine precision). Its usage examples
live in ``examples/hybrid/``:
``bamboo_charge_analysis.ipynb`` tours the predicted partial charges, the
``energy_nn``/``energy_elec`` split, the dipole, the symmetries, and ASE
deployment; ``bamboo_dimer_electrostatics.ipynb`` shows BAMBOO's built-in
charge-equilibrium electrostatics binding the charged/polar dimers beyond the
GET cutoff (electrostatics on vs. off), reusing the same CC/CP/PP dimer set as
the LES example.

The Latent Ewald Summation long-range add-on is validated against the original
``cace`` ``EwaldPotential`` in ``examples/fidelity_checks/les_verification.ipynb``;
``examples/gnn/les/les_molecular_dimers.ipynb`` reproduces
the LES paper's central experiment: extrapolating the binding
curves of charged/polar molecular dimers, where short-range models fail
qualitatively -- a CC/CP/PP subset of the BioFragment dimer set ships with
the example. (Neutral homogeneous systems like the Argon set carry no
long-range tail, so they are deliberately *not* used here.)
