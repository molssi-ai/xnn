.. _deployment:

**********
Deployment
**********

Trained xnn models deploy in two ways: as an ASE calculator for Python
workflows, and as TorchScript for production LAMMPS runs.

ASE calculator
==============
:class:`~xnn.common.deploy.ase_calculator.XNNCalculator` (requires the
``ase`` extra) makes any xnn model a standard ASE calculator:

.. code-block:: python

   from xnn.common.deploy import XNNCalculator

   atoms.calc = XNNCalculator(model, cutoff=5.0)
   atoms.get_potential_energy()
   atoms.get_forces()

It builds the same PBC-aware neighbor list used in training on every call,
so molecular and periodic systems both work. See :ref:`howto-ase` for a full
molecular-dynamics recipe.

TorchScript and LAMMPS export
=============================
:func:`~xnn.common.deploy.torchscript.export_torchscript_potential` writes a
**self-contained** ``.pt``: it is driven purely by tensors and carries its own
neighbor list, so a consumer needs nothing but ``libtorch`` /
``torch.jit.load`` -- no ``xnn`` import, no Python model code, no config file.

.. code-block:: console

   $ xnn export --ckpt runs/exp/best.pt --out deployed.pt

``--config`` is optional: xnn-trained checkpoints embed their own
:class:`~xnn.common.config.schema.Config`, so the architecture is recovered
from the checkpoint itself. The net charge (and, for the two-channel
AIMNet2 models, the spin multiplicity) of the deployed system is fixed in
the artifact: ``--total-charge`` / ``--spin-multiplicity``
(``total_charge=`` / ``spin_multiplicity=`` in Python). The equivalent
Python call is

.. code-block:: python

   from xnn.common.deploy import export_torchscript_potential

   export_torchscript_potential(model, cutoff=5.0, path="deployed.pt")

The artifact exposes two entry points:

``forward(pos, atomic_numbers, cell, pbc)``
   The whole-system ABI. Builds its own neighbor list from the cutoff baked in
   at export time, and returns ``energy``, ``node_energy``, ``forces``,
   ``stress``, ``virial`` (plus ``energy_sr`` / ``energy_lr`` /
   ``latent_charges`` for long-range models, and ``charges`` for a
   charge-predicting model such as AIMNet2, whose Coulomb sum is part of the
   artifact). This is the general-purpose entry point.

``forward_lammps(pos, edge_index, cell_shifts, atomic_numbers, cell)``
   The pair-style ABI, matching
   :class:`~xnn.common.deploy.lammps.LAMMPSWrapper` and hence the
   ``pair_nequip`` / ``pair_mace`` / ``pair_allegro`` pattern: the MD engine
   supplies the neighbor list it already has.

Loading it from any MD package is then:

.. code-block:: python

   import torch

   model = torch.jit.load("deployed.pt")          # no xnn needed
   out = model(pos, atomic_numbers, cell, pbc)
   energy, forces = out["energy"], out["forces"]

Calling conventions
-------------------
The artifact behaves like an ordinary scripted module: ``.parameters()``,
``.state_dict()``, ``.eval()``, ``.to(device)`` and ``.double()`` all work, and
it runs on GPU via ``torch.jit.load(path, map_location="cuda")``. Positions may
be float32 or float64 regardless of the weights' dtype -- the module computes
in its own dtype and returns results in the caller's -- and ``cell`` / ``pbc``
may be omitted for a molecular system. Three things differ from a typical
inference model:

* **One structure per call.** Inputs are ``(N, 3)``, not ``(B, N, 3)``; there
  is no batch dimension. Loop over structures.
* **Wrapping the call in** ``torch.no_grad()`` **is fine** -- the module
  re-enables grad internally, because its forces come from autograd, and
  restores the caller's grad mode afterwards.
* ``torch.inference_mode()`` **is not supported** and raises. Tensors created
  under inference mode can never participate in autograd, so the force
  gradient cannot be taken; this cannot be worked around from inside the
  module. Use ``torch.no_grad()``, or no context manager at all.

The cutoff and a ``long_range`` flag are embedded as extra files in the
archive, so a consumer can introspect the artifact without xnn::

   extra = {"cutoff": "", "long_range": ""}
   torch.jit.load("deployed.pt", _extra_files=extra)

.. warning::

   **Long-range (LES) models must be driven with the whole system on one
   rank.** :class:`~xnn.common.models.les.LatentEwald` adds an Ewald energy
   over latent charges, which is a *global* sum -- every atom's latent charge
   enters, with no cutoff -- so it does not decompose into a local, per-domain
   neighbor list. An MPI-decomposed pair style that only ever sees its own
   subdomain plus ghosts cannot reproduce the trained energy. Use ``forward``
   (or ``forward_lammps`` with a full-system neighbor list) via a
   single-rank run, ``fix external``, or the
   :class:`~xnn.common.deploy.mdi_engine.MDIEngine`. The
   ``long_range`` metadata key records whether this applies.

   The MDI engine (``xnn mdi``) also understands ``>TOTCHARGE`` and the
   ``--total-charge`` option for the system's net charge, adds D3 / D4 on top
   of a plain checkpoint with ``--dispersion`` (refused if the checkpoint
   already includes it), builds graphs at the served model's own cutoff, and
   serves in the checkpoint's dtype unless ``--dtype`` is given. For molecular
   dynamics and optimizations, ``--eeq-reuse`` carries the large-regime D4 EEQ
   solve over from one step to the next
   (:class:`~xnn.common.models.eeq.EEQReuse`); the ASE calculator offers the
   same as ``XNNCalculator(model, cutoff, eeq_reuse=True)``.

.. note::

   The built-in neighbor list is the brute-force ``O(S N^2)`` reference
   algorithm, matching :func:`~xnn.common.data.build_neighbor_list`. It is
   fine for molecular and modest periodic systems; for large cells, supply the
   engine's own neighbor list through ``forward_lammps``.

Route B: labels with the dispersion removed
-------------------------------------------

A model trained on labels from which a dispersion correction was subtracted
(``E_ref - E_D4``, so the network learns only the rest) must have that term
added back wherever it is served. The training config records what was
subtracted, and the record travels in the checkpoint::

   model:
     name: mace
     ...
   subtracted_dispersion:
     name: d4                     # the exact functional parameters of the labels
     s6: 1.0
     s8: 1.20065498
     a1: 0.40085597
     a2: 5.02928789
     s9: 1.0
     alp: 16.0
     cutoff_pair: 12.0            # and the settings to add it back in a periodic run
     switch_width_pair: 2.0
     cutoff_triple: 10.0
     switch_width_triple: 1.0
     cutoff_eeq: 16.0
     regime: auto
     dataset: water_clusters_minusD4

The keys are the constructor options of :class:`~xnn.common.models.d4.DFTD4`
(or ``DFTD3``) plus the annotations ``dataset`` and ``note``; a misspelled
option or a model that already includes the term (``model.extra.dispersion``)
is refused when the config is built. ``xnn mdi`` and ``xnn export`` then add
the recorded term without any option and log that they did, so a route-B
checkpoint is correct whichever launcher starts it. ``--dispersion`` on
``xnn mdi`` overrides the recorded keys it names (``"{cutoff_triple: 8.0}"``
keeps the recorded functional parameters), a different term name is an
error, and ``--no-dispersion`` serves or exports the checkpoint as is. The
same rules apply to :meth:`MDIEngine.from_checkpoint(path, dispersion=...)
<xnn.common.deploy.mdi_engine.MDIEngine.from_checkpoint>`.

The values above are the settings recommended for periodic production runs
(the labels themselves are usually computed on clusters, uncut and
unswitched): a 12 Å pair cutoff with a 2 Å switch, a 10 Å three-body cutoff
with a 1 Å switch (8 Å leaves about 24 atm of pressure error against 12 Å,
10 Å about 7 atm, at little cost for large systems), the 16 Å default EEQ
range and the automatic regime. Precision and the EEQ reuse remain run
options of the engine (``--dtype``, ``--eeq-reuse``): float32 with the reuse
is as accurate as float64 (~1e-6 eV/Å) and the fastest.

For a checkpoint trained before the field existed,
``tools/record_subtracted_dispersion.py best.pt --spec "{name: d4, ...}"``
adds the record in place (keeping a ``.bak`` copy); checkpoints without a
record serve as they always did.

The older :func:`~xnn.common.deploy.lammps.export_to_lammps` remains for the
short-range-only pair-style wrapper. A model is exportable when it provides
the scriptable core
``node_energy(atomic_numbers, edge_index, edge_vec)`` and, for the
self-contained export, ``node_features_energy(...)``; SchNet, NequIP,
MACE, and Allegro all do, and the scripted models reproduce the eager ones
to ~1e-15 (verified in the test suite). CACE is the exception: like the
original ``cace`` package (which has no LAMMPS interface) it deploys via the
ASE calculator only. ReaxFF likewise deploys via the ASE calculator only
(its per-structure EEM linear solve and valence enumeration have no
scriptable per-edge core); when running molecular dynamics with it, use the
customary ReaxFF timestep of about 0.1 fs. OPLS also deploys via the ASE
calculator (its energy is defined relative to a bound molecular topology,
not per edge); with explicit hydrogens the customary timestep is 0.5-1 fs.

See :ref:`howto-lammps` for the step-by-step guide, including the CLI form
(``xnn export``).
