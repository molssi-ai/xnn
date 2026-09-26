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
from the checkpoint itself. The equivalent Python call is

.. code-block:: python

   from xnn.common.deploy import export_torchscript_potential

   export_torchscript_potential(model, cutoff=5.0, path="deployed.pt")

The artifact exposes two entry points:

``forward(pos, atomic_numbers, cell, pbc)``
   The whole-system ABI. Builds its own neighbor list from the cutoff baked in
   at export time, and returns ``energy``, ``node_energy``, ``forces``,
   ``stress``, ``virial`` (plus ``energy_sr`` / ``energy_lr`` /
   ``latent_charges`` for long-range models). This is the general-purpose
   entry point.

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
