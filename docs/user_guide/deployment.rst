.. _deployment:

**********
Deployment
**********

A trained model deploys three ways: as an ASE calculator, as a self-contained
TorchScript file for LAMMPS and other engines, and as an MDI engine. All
three load from the same sources (a checkpoint, a model directory, a hub
name, a URL or a DOI).

ASE calculator
==============
:class:`~xnn.common.deploy.ase_calculator.XNNCalculator` (``ase`` extra)
makes any model a standard calculator with energy, forces, stress and, for
charge-predicting models, charges and dipole:

.. code-block:: python

   from xnn.common.deploy import XNNCalculator

   atoms.calc = XNNCalculator.from_pretrained("mace-mp-0-medium", device="cuda")
   atoms.calc = XNNCalculator(model, cutoff=5.0)       # a model object you hold

The net charge and spin multiplicity are read from ``atoms.info["charge"]``
and ``atoms.info["spin_multiplicity"]``. ``eeq_reuse=True`` carries the D4
charge solve from one MD step to the next. See :ref:`howto-ase`.

TorchScript
===========
:func:`~xnn.common.deploy.export_torchscript_potential` (``xnn export``)
writes a ``.pt`` file that carries its own neighbor list and needs only
``libtorch`` or ``torch.jit.load``:

.. code-block:: python

   model = torch.jit.load("deployed.pt")
   out = model(pos, atomic_numbers, cell, pbc)                            # whole system
   out = model.forward_lammps(pos, edge_index, cell_shifts, atomic_numbers, cell)   # pair style

``forward`` returns ``energy``, ``node_energy``, ``forces``, ``stress`` and
``virial`` (plus ``energy_sr`` / ``energy_lr`` / ``latent_charges`` for LES
models and ``charges`` for AIMNet2). ``forward_lammps`` follows the
``pair_nequip`` / ``pair_allegro`` / ``pair_mace`` pattern: the engine
supplies its neighbor list. Rules of the artifact:

- One structure per call, no batch dimension.
- Positions may be float32 or float64; the module computes in its own dtype.
- ``torch.no_grad()`` is fine (grad is re-enabled inside for the forces);
  ``torch.inference_mode()`` is not supported.
- The net charge and spin multiplicity are fixed at export
  (``--total-charge``, ``--spin-multiplicity``).
- The cutoff and a ``long_range`` flag are stored as extra files:
  ``torch.jit.load(path, _extra_files={"cutoff": "", "long_range": ""})``.
- The built-in neighbor list is the brute-force reference; for large cells
  pass the engine's own list through ``forward_lammps``.

A **long-range (LES) model** adds a global Ewald sum, which does not
decompose over MPI subdomains. Drive it with the whole system on one rank:
``forward``, ``fix external``, or the MDI engine. The ``long_range`` flag
records whether this applies. Exports always script the reference
implementation of every block, never the fused kernels.

MDI engine
==========
``xnn mdi`` (``mdi`` extra) serves any checkpoint over the `MolSSI Driver
Interface <https://github.com/MolSSI-MDI/MDI_Library>`_, for LAMMPS
``fix mdi/qm`` or any other driver, and :class:`~xnn.common.deploy.mdi_engine.MDIEngine`
is the Python side:

.. code-block:: bash

   xnn mdi --ckpt runs/exp/best.pt --device cuda:0 \
           -mdi "-role ENGINE -name xnn -method TCP -port 8021 -hostname localhost"

The engine builds graphs at the model's own cutoff, serves in the
checkpoint's dtype unless ``--dtype`` says otherwise, understands
``>TOTCHARGE`` and ``--total-charge``, adds D3 / D4 to a plain checkpoint
with ``--dispersion``, carries the D4 charge solve between steps with
``--eeq-reuse``, and selects the fused kernels with ``--fast``. The
``examples/deploy`` notebooks drive it from Python and from LAMMPS.

Dispersion subtracted from the labels
=====================================
A model trained on labels with a dispersion correction removed
(``E_ref - E_D4``) must have that term added back wherever it is served. The
config records it, the record travels in the checkpoint, and ``xnn mdi``,
``xnn export`` and ``from_pretrained`` add it without any option:

.. code-block:: yaml

   subtracted_dispersion:
     name: d4                   # the functional parameters of the labels
     s6: 1.0
     s8: 1.20065498
     a1: 0.40085597
     a2: 5.02928789
     cutoff_pair: 12.0          # and the settings to add it back in a periodic run
     switch_width_pair: 2.0
     cutoff_triple: 10.0
     switch_width_triple: 1.0
     dataset: water_clusters_minusD4

The keys are the constructor options of
:class:`~xnn.common.models.d4.DFTD4` (or ``DFTD3``) plus the annotations
``dataset`` and ``note``; a model that already includes the term is
refused. ``--dispersion`` on ``xnn mdi`` overrides the recorded keys it
names, ``--no-dispersion`` serves or exports the checkpoint as is, and
``tools/record_subtracted_dispersion.py`` adds the record to an older
checkpoint.

Which model deploys how
=======================
- **ASE and MDI**: every model.
- **TorchScript / LAMMPS**: SchNet, DimeNet, PaiNN, NequIP, MACE, Allegro and AIMNet2, with
  or without a D3 / D4 term. CACE, PhysNet and BAMBOO deploy through ASE
  and MDI, like their reference codes. The classical force fields do too;
  use their customary timesteps (about 0.1 fs for ReaxFF, 0.5 to 1 fs for
  OPLS and DREIDING with explicit hydrogens).
