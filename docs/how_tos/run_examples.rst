.. _howto-examples:

*************************
Run the Example Notebooks
*************************

The repository ships validation notebooks that check the xnn
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
     - A full train/test pipeline on Argon MD data, run twice (xnn vs. the
       original code) and compared at every stage (losses, parity plots,
       errors).
   * - ``gnn/<model>/<model>_argon_density_md.ipynb``
     - Liquid-argon mass density from NPT molecular dynamics through ASE,
       comparing xnn against the reference (identical weights → ~zero
       difference, plus independently trained models).

MACE additionally has ``recreate_mace_architecture.ipynb``, a
step-by-step tutorial that rebuilds the MACE architecture block by block in
*both* ``mace-torch`` and xnn, with the defining equations and architecture
figures, and two **foundation-model** notebooks built on
``MACE.from_foundation()``: ``mace_foundation_molecules.ipynb`` (MACE-OFF23
on the butane torsion vs the built-in OPLS-AA, the water dimer vs the
CCSD(T)/CBS benchmark, and a ``Trainer`` fine-tune to rMD17 malonaldehyde)
and ``mace_foundation_materials.ipynb`` (equations of state of Si / Al /
NaCl across the MACE-MP generations). The conversion of every published
checkpoint is itself verified in
``examples/fidelity_checks/mace_foundation_verification.ipynb``.

Every training / MD notebook loads its data through the dataset hub
(:func:`~xnn.common.data.hub.base.load_dataset`) rather than reading files by
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

SchNet (``examples/cnn/schnet/``) is validated differently from the rest: it
is a clean-room build from the manuscripts, so
``examples/fidelity_checks/schnet_verification.ipynb`` checks every block
(embedding, Gaussian RBF, shifted softplus, cfconv, interaction blocks,
readout/standardization) against an independent NumPy implementation of the
papers' equations — nothing from schnetpack is used, and no extra dependency
is needed (it runs with the plain ``xnn`` kernel).
``schnet_rmd17_train.ipynb`` trains the paper architecture on rMD17 ethanol
(``load_dataset("rmd17", ...)``) with the paper's energy+force loss
weighting, and ``schnet_ethanol_md.ipynb`` loads that checkpoint and runs
thermostat-free NVE dynamics through the ASE calculator to demonstrate the
paper's energy-conservation-by-construction claim (run the training notebook
first).

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

ReaxFF (``examples/ffnn/reaxff/``) has ``reaxff_rmd17_train_test.ipynb``,
which trains the ReaxFF-nn reactive force field from a generic seed library
on rMD17 malonaldehyde (``load_dataset("rmd17", ...)``, energies **and**
forces) and exports the result as a portable ``ffield.json``, and
``reaxff_md_bond_orders.ipynb``, which runs ASE molecular dynamics with the
trained library and analyses the reactive descriptors (bond orders, EEM
charges, a smooth bond-dissociation scan). Both notebooks also benchmark the
**original classical ReaxFF**: the published C/H/O combustion field
(``ReaxFF("CHO_cho_2008")``, Chenoweth *et al.* 2008, shipped in the SEAMM
``.frc`` format) runs in the same ``ReaxFF`` class, giving the
published-parameter baseline for the test-set metrics and the
bond-dissociation scan. No third-party ReaxFF code is required (see the
fidelity notes in the documentation).

OPLS (``examples/ffnn/opls/``) has
``opls_conformational_energetics.ipynb``, which reproduces the relaxed
torsional energies of Table 1 of the OPLS-AA paper (Jorgensen *et al.* 1996)
with an ASE dihedral driver on the built-in parameter libraries, and
``opls_lopls_torsion_refit.ipynb``, which compares OPLS-AA against the
L-OPLS long-hydrocarbon refit (Siu *et al.* 2012) on the hexane torsion
profile and then recovers the published L-OPLS torsion coefficients by
gradient descent (``trainable=("dihedral_v",)``), exporting the trained
library and finishing with an NVE energy-conservation check. The
implementation itself is cross-validated against OpenMM in
``examples/fidelity_checks/opls_verification.ipynb`` (optional ``openmm``
dependency).

The Latent Ewald Summation long-range add-on is validated against the original
``cace`` ``EwaldPotential`` in ``examples/fidelity_checks/les_verification.ipynb``;
``examples/gnn/les/les_molecular_dimers.ipynb`` reproduces
the LES paper's central experiment: extrapolating the binding
curves of charged/polar molecular dimers, where short-range models fail
qualitatively -- a CC/CP/PP subset of the BioFragment dimer set, loaded with
``load_dataset("lode_dimers", subset="bio_scan")`` (bundled with the repo).
(Neutral homogeneous systems like the Argon set carry no long-range tail, so
they are deliberately *not* used here.)

The DFT-D4 dispersion add-on (``examples/common/d4/``) is validated against
the reference ``dftd4`` Python package in
``examples/fidelity_checks/d4_verification.ipynb`` (install it with the
``d4`` extra, ``pip install -e ".[d4]"``; the notebook imports ``dftd4``
*before* ``torch``, because the wheel's bundled OpenMP runtime returns wrong
EEQ charges once torch's thread pool is active). ``d4_paper_examples.ipynb``
reproduces figures and numbers of the D4 paper (needs ``rdkit`` for the
hexenyne geometry and ``ase`` for the g2 and S22 sets) and
``d4_benchmark.ipynb`` compares accuracy and timing with ``dftd4`` and runs
a D4-corrected MLIP through the ASE, TorchScript and LAMMPS-ABI channels.
The DFT-D3 counterpart (``examples/common/d3/``, fidelity notebook
``examples/fidelity_checks/d3_verification.ipynb``) compares against the
reference ``simple-dftd3`` Python package (``pip install -e ".[d3]"``; no
import-order caveat for this one).

Deployment over MDI (``examples/deploy/mdi_argon_md.ipynb``) trains a small
MACE on the bundled Argon set, serves the checkpoint with the ``xnn mdi``
command as a `MolSSI Driver Interface
<https://github.com/MolSSI-MDI/MDI_Library>`_ engine, and drives NVE molecular
dynamics from a minimal Python driver over TCP. It needs the ``mdi`` extra
(``pip install -e ".[mdi]"``, i.e. ``pymdi``); because the MDI library can
only be initialized once per process, restart the kernel before re-running it.
``xnn mdi`` serves in the checkpoint's own dtype unless ``--dtype`` says
otherwise (the model is built in float64 and cast once, so a float64 run sees
the exact D3 / D4 / LES tables), takes the neighbor-list radius from the built
model (a dispersion wrapper widens it beyond the config's cutoff), and can add
a D3 / D4 correction to a checkpoint trained without one (``--dispersion d4``
or a YAML mapping as in ``extra.dispersion``; refused when the checkpoint
already carries dispersion). The system's net charge, which D4's EEQ charges
and charge-aware models use, is set with ``--total-charge`` and can be changed
by the driver at run time through ``>TOTCHARGE``.
The engine is model agnostic: the same command serves any family's
``best.pt``. Its companion ``mdi_argon_lammps.ipynb`` drives the identical
engine from **LAMMPS** (``fix mdi/qm``) instead: NVE plus a LAMMPS-side radial
distribution function, with the step-0 energy and pressure validated against
direct evaluation. It additionally needs a LAMMPS executable built with the
MDI package (``cmake -D PKG_MDI=yes``; serial is fine) available as ``lmp`` on
``PATH`` or via the ``XNN_LMP`` environment variable; it runs both codes as
subprocesses, so no kernel restart is needed.
