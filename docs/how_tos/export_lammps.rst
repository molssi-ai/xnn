.. _howto-lammps:

**********************************
Export to LAMMPS and TorchScript
**********************************

SchNet, NequIP, MACE and Allegro export to a self-contained TorchScript
file that needs nothing but ``libtorch`` or ``torch.jit.load``: no xnn, no
Python model code, no config. The scripted models match the eager ones to
about 1e-15.

Export
======

.. code-block:: bash

   xnn export --ckpt runs/exp/best.pt --out deployed.pt
   xnn export --ckpt mace-off23-small --out mace_off.pt        # any hub name works

.. code-block:: python

   from xnn.common.deploy import export_torchscript_potential

   export_torchscript_potential(model, cutoff=5.0, path="deployed.pt")

Options: ``--head`` picks the head of a multi-head checkpoint (LoRA adapters
are always folded in), ``--total-charge`` and ``--spin-multiplicity`` fix
the charge state of the deployed system, and ``--no-dispersion`` leaves out
a dispersion term the checkpoint records as subtracted from its labels
(:ref:`deployment`).

Use it
======
The file has two entry points. ``forward`` takes the whole system and builds
its own neighbor list:

.. code-block:: python

   import torch

   model = torch.jit.load("deployed.pt")
   out = model(pos, atomic_numbers, cell, pbc)      # cell / pbc optional for molecules
   energy, forces, stress = out["energy"], out["forces"], out["stress"]

``forward_lammps(pos, edge_index, cell_shifts, atomic_numbers, cell)`` is
the pair-style interface of the ``pair_nequip`` / ``pair_allegro`` /
``pair_mace`` pattern: the MD engine supplies the neighbor list. One
structure per call, no batch dimension; ``torch.no_grad()`` is fine,
``torch.inference_mode()`` is not (forces need autograd).

Which models export
===================
SchNet, NequIP, MACE, Allegro and AIMNet2, with or without a D3 / D4 term.
CACE, PhysNet, BAMBOO and the classical force fields deploy through ASE or
the MDI engine instead. A long-range LES model exports, but its Ewald sum is
global, so drive it with the whole system on one rank (``forward``, ``fix
external`` or the :ref:`MDI engine <deployment>`), never through a
domain-decomposed pair style.
