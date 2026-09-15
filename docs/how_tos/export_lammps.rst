.. _howto-lammps:

**********************************
Export to LAMMPS and TorchScript
**********************************

SchNet, NequIP, MACE, and Allegro can be compiled to TorchScript and deployed
in `LAMMPS <https://www.lammps.org>`_. A model is exportable when it provides
the scriptable core

.. code-block:: python

   node_energy(atomic_numbers, edge_index, edge_vec)

which all four deployable models do. The scripted models reproduce their eager
counterparts up to ~1e-15 (verified in ``tests/test_mace.py``,
``tests/test_nequip.py``, and ``tests/test_allegro.py``).

From Python
===========

.. code-block:: python

   from xnns.common.deploy import export_to_lammps, export_torchscript

   # LAMMPS wrapper
   export_to_lammps(model, cutoff=5.0, path="deployed.pt")

   # plain TorchScript
   export_torchscript(model, path="model_ts.pt")

``export_to_lammps`` wraps the model in
:class:`~xnns.common.deploy.lammps.LAMMPSWrapper`, which defines the tensor
application binary interface (ABI) expected by the LAMMPS pair styles
(positions, atomic numbers, edge index, and edge vectors in; per-atom and total
energies out).

From the command line
=====================

.. code-block:: bash

   xnns export --config configs/train.yaml --ckpt runs/exp/best.pt --to lammps
   xnns export --config configs/train.yaml --ckpt runs/exp/best.pt --to torchscript

Using the exported model in LAMMPS
==================================
Pair the exported ``.pt`` file with the matching C++ pair style, following
the `pair_nequip <https://github.com/mir-group/pair_nequip>`_ /
`pair_allegro <https://github.com/mir-group/pair_allegro>`_ /
pair_mace pattern. The tensor interface is defined in one place
(``src/xnns/common/deploy/lammps.py``), so a single pair style covers every
exportable xnns model.

.. note::

   For NequIP, TorchScript export required a scriptable, bit-exact stand-in
   for e3nn's ``Gate`` (``xnns.gnn.models.nequip._Gate``); the e3nn 0.4.4
   original cannot be scripted on torch 2.x. This is transparent to users:
   the substitution is numerically identical.
