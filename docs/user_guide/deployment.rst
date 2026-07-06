.. _deployment:

**********
Deployment
**********

Trained xnns models deploy in two ways: as an ASE calculator for Python
workflows, and as TorchScript for production LAMMPS runs.

ASE calculator
==============
:class:`~xnns.common.deploy.ase_calculator.XNNSCalculator` (requires the
``ase`` extra) makes any xnns model a standard ASE calculator:

.. code-block:: python

   from xnns.common.deploy import XNNSCalculator

   atoms.calc = XNNSCalculator(model, cutoff=5.0)
   atoms.get_potential_energy()
   atoms.get_forces()

It builds the same PBC-aware neighbor list used in training on every call,
so molecular and periodic systems both work. See :ref:`howto-ase` for a full
molecular-dynamics recipe.

TorchScript and LAMMPS export
=============================
.. code-block:: python

   from xnns.common.deploy import export_torchscript, export_to_lammps

   export_torchscript(model, path="model_ts.pt")
   export_to_lammps(model, cutoff=5.0, path="deployed.pt")

``export_to_lammps`` wraps the model in
:class:`~xnns.common.deploy.lammps.LAMMPSWrapper`, which defines the tensor
ABI for the LAMMPS side, and compiles the result with TorchScript. Pair the
exported ``.pt`` with the matching C++ pair style (the ``pair_nequip`` /
``pair_mace`` / ``pair_allegro`` pattern).

A model is exportable when it provides the scriptable core
``node_energy(atomic_numbers, edge_index, edge_vec)`` — SchNet, NequIP,
MACE, and Allegro all do, and the scripted models reproduce the eager ones
to ~1e-15 (verified in the test suite).

See :ref:`howto-lammps` for the step-by-step guide, including the CLI form
(``xnns export``).
