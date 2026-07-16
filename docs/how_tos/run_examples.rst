.. _howto-examples:

*************************
Run the Example Notebooks
*************************

The repository ships validation notebooks that check the xnns
implementations against the reference codes on real Argon MD data.

.. tip::

   Every notebook mentioned on this page is also **rendered in these docs with
   its executed outputs**; see :ref:`examples` to read them without running
   anything. This page is about running them yourself.

Install the ``examples`` extra first: it pulls in the reference packages
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
     - A full train/test pipeline on Argon MD data, run twice (xnns vs. the
       original code) and compared at every stage (losses, parity plots,
       errors).
   * - ``gnn/<model>/<model>_argon_density_md.ipynb``
     - Liquid-argon mass density from NPT molecular dynamics through ASE,
       comparing xnns against the reference (identical weights → ~zero
       difference, plus independently trained models).

MACE additionally has ``recreate_mace_architecture.ipynb``, a
step-by-step tutorial that rebuilds the MACE architecture block by block in
*both* ``mace-torch`` and xnns, with the defining equations and architecture
figures.

Every training / MD notebook loads its data through the dataset hub
(:func:`~xnns.common.data.hub.base.load_dataset`) rather than reading files by
hand. The Argon set (shared by the MACE, NequIP, Allegro, CACE, and PhysNet
notebooks) is bundled in the repository and loaded with
``load_dataset("argon_md", split=...)`` (no download; see :ref:`data`). The CACE
series compares against the original ``cace`` package
(``pip install git+https://github.com/BingqingCheng/cace``); the PhysNet
series (``examples/dnn/physnet/``) compares against the original
**TensorFlow** implementation and needs a venv with both ``tensorflow`` and
``torch`` (the notebooks clone MMunibas/PhysNet on demand).

ANI (``examples/dnn/ani/``) has ``ani_rmd17_train.ipynb``, which trains ANI
from scratch on rMD17 paracetamol (``load_dataset("rmd17", ...)``) with an
energy/force parity plot and a smooth potential-energy scan, and
``ani1_dataset.ipynb``, which loads a subset of the original 20 M-conformation
ANI-1 training set (``load_dataset("ani1", ...)``) and reproduces the paper's
energy-correlation result. Its ANI-1x companion ``ani1x_dataset.ipynb`` loads
the active-learning ANI-1x set (``load_dataset("ani1x", ...)``) and trains the
``ani-1x`` preset on energies **and** forces; read the two side by side to see
what separates the two models. ``ani1ccx_dataset.ipynb`` completes the series:
it loads the coupled-cluster ANI-1ccx subset (``load_dataset("ani1ccx", ...)``),
contrasts the CCSD(T)*/CBS energies with the DFT values for the same
conformations, and mimics the paper's **transfer learning**: pre-training the
``ani-1ccx`` preset on DFT, then retraining on the coupled-cluster energies
with the paper's exact 65,280 network weights held fixed, against the paper's
CC-only ANI-1ccx-R control. ``ani2x_dataset.ipynb`` extends the series to seven
elements: it loads the ANI-2x set (``load_dataset("ani2x", ...)``) and
trains/evaluates the ``ani-2x`` preset on a subset containing S/F/Cl.
Its block-by-block fidelity check against
``aiqm/torchani`` is ``examples/fidelity_checks/ani_verification.ipynb`` (needs
the ``ani`` extra: ``pip install -e ".[ani]"``).

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
the LES example (``load_dataset("lode_dimers", subset="bio_scan")``).

The Latent Ewald Summation long-range add-on is validated against the original
``cace`` ``EwaldPotential`` in ``examples/fidelity_checks/les_verification.ipynb``;
``examples/gnn/les/les_molecular_dimers.ipynb`` reproduces
the LES paper's central experiment: extrapolating the binding
curves of charged/polar molecular dimers, where short-range models fail
qualitatively -- a CC/CP/PP subset of the BioFragment dimer set, loaded with
``load_dataset("lode_dimers", subset="bio_scan")`` (bundled with the repo).
(Neutral homogeneous systems like the Argon set carry no long-range tail, so
they are deliberately *not* used here.)
