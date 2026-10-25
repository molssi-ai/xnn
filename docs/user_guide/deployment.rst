.. _deployment:

**********
Deployment
**********

A trained model can be served three ways. All three require the same inputs: a
checkpoint, a model directory, a model hub name, a URL or a DOI. The three
deployment channels are:

- **ASE calculator**, which enables molecular dynamics, relaxations and analysis in Python,
- **TorchScript file**, which is compatible with LAMMPS and other engines, with
  no xnn installed, and 
- **MDI engine** which works with any MD driver that can communicate with the
  `MolSSI Driver Interface <https://github.com/MolSSI-MDI/MDI_Library>`_.

ASE calculator
==============
The :class:`~xnn.common.deploy.ase_calculator.XNNCalculator` (requires ``ase``
extra dependencies installed) provides energy, forces and stress, plus charges
and dipole for models that can predict them:

.. code-block:: python

   from xnn.common.deploy import XNNCalculator

   atoms.calc = XNNCalculator.from_pretrained("mace-mp-0-medium", device="cuda")
   atoms.calc = XNNCalculator(model, cutoff=5.0)       #  A model object you hold

The net charge and spin multiplicity can come from ``atoms.info`` entries in an
ASE ``*.xyz`` file. See the :ref:`howto-ase` guide for more information.

TorchScript
===========
The ``xnn export`` command writes one ``.pt`` file that carries its own neighbor
list:

.. code-block:: bash

   xnn export --ckpt runs/exp/best.pt --out deployed.pt

You can load it in Python and call ``forward`` with the same arguments as the
original model:

.. code-block:: python

   model = torch.jit.load("deployed.pt")
   out = model(pos, atomic_numbers, cell, pbc)      # energy, forces, stress, ...

The ``forward_lammps(pos, edge_index, cell_shifts, atomic_numbers, cell)`` takes
the engine's own neighbor list instead, as the ``pair_nequip`` /
``pair_allegro`` / ``pair_mace`` pair styles expect. Note that:

- There should be one structure per call here. Using ``torch.no_grad()`` will
  work but ``torch.inference_mode()`` will not.
- The net charge and spin multiplicity are fixed at export time
  (by setting ``--total-charge`` and ``--spin-multiplicity`` flags).
- A long-range (LES) model needs the whole atomistic system on one rank, so
  drive it through ``forward``, ``fix external`` or the MDI engine rather than a
  domain-decomposed pair style.

For more technical details and options, see the :ref:`howto-lammps`.

MDI engine
==========
The ``xnn mdi`` (requires ``mdi`` extra dependency installed) serves a
checkpoint to LAMMPS ``fix mdi/qm`` or any other `MDI
<https://github.com/MolSSI-MDI/MDI_Library>`_ compatible MD driver:

.. code-block:: bash

   xnn mdi --ckpt runs/exp/best.pt --device cuda:0 \
           -mdi "-role ENGINE -name xnn -method TCP -port 8021 -hostname localhost"

You can use the ``--dispersion d4`` flag to add a dispersion term to a plain
model. Set the ``--eeq-reuse`` to speed up D4 in an MD run. For AMOEBA and HIPPO
the engine predicts each step's induced dipoles from the last steps
(``--dipole-predictor``, ASPC by default). On a GPU it also evaluates their
steps on a Verlet pair list (``--skin``, 1 angstrom by default) and replays each
step as one CUDA graph. The ``--step-graph off`` flag disables this behavior.
The :ref:`cli` guide lists every option available to you. 

There are example notebooks in ``notebooks/examples/common/deploy`` which illustrate how to drive
the engine from Python and LAMMPS deployment interfaces.

Dispersion removed from the labels
==================================
A model trained on ``E_ref - E_D4`` needs the D4 term added back wherever it is
adopted for inference. If you record the ``subtracted_dispersion`` section in
the training config, every deployment channel will automatically add the
subtracted dispersion term back to the predicted energies at the inference time:

.. code-block:: yaml

   subtracted_dispersion:
     name: d4                   # Or d3, with the damping parameters of the labels
     s8: 1.20065498
     a1: 0.40085597
     a2: 5.02928789
     cutoff_pair: 12.0          # Settings for periodic runs
     switch_width_pair: 2.0

You can use the ``--no-dispersion`` flag to serve or export your model without
it.

Which model deploys how
=======================
- **ASE and MDI**: every model is supported, with or without a D3 / D4 term.
- **TorchScript / LAMMPS**: SchNet, DimeNet, GemNet, PaiNN, TensorNet, ET, Equiformer, EquiformerV2,
  SpookyNet, NequIP, MACE, Allegro and AIMNet2, with or without a D3 / D4 term.
