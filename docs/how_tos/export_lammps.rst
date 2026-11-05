.. _howto-lammps:

**********************************
Export to LAMMPS and TorchScript
**********************************

SchNet, DimeNet, DimeNet++, GemNet, PaiNN, TensorNet, NequIP, MACE, Allegro,
CACE, AIMNet2, ANI, PhysNet, HDNNP, ET, Equiformer, EquiformerV2, BAMBOO,
SpookyNet and PM3 / PM6 / PM6-ORG / PM7 export to a self-contained
TorchScript file that needs nothing but ``libtorch`` or ``torch.jit.load``: no
xnn, no Python model code, no config. The scripted models match their eager
variants to about 1e-15.

Export
======

You can export a model checkpoint to its TorchScript format using the command
line interface:

.. code-block:: bash

   xnn export --ckpt runs/exp/best.pt --out deployed.pt
   xnn export --ckpt mace-off23-small --out mace_off.pt     # Any valid model hub name works

or using the Python API:

.. code-block:: python

   from xnn.common.deploy import export_torchscript_potential

   export_torchscript_potential(model, cutoff=5.0, path="deployed.pt")

Options: ``--head`` picks the head of a multi-head checkpoint (LoRA adapters
are always folded in), ``--total-charge`` and ``--spin-multiplicity`` fix
the charge state of the deployed system, and ``--no-dispersion`` leaves out
a dispersion term the checkpoint records as subtracted from its labels
(:ref:`deployment`).

Using the exported model
========================
The export framework offers two entry points for inference. The ``forward``
function, which takes the whole system and builds its own neighbor list from it:

.. code-block:: python

   import torch

   model = torch.jit.load("deployed.pt")

   # cell / pbc optional for molecules
   out = model(pos, atomic_numbers, cell, pbc)

   energy, forces, stress = out["energy"], out["forces"], out["stress"]

The ``forward_lammps(pos, edge_index, cell_shifts, atomic_numbers, cell)``
function, which implements the pair-style interface of the ``pair_nequip`` /
``pair_allegro`` / ``pair_mace`` pattern where the MD engine supplies the
neighbor list. One structure is processed per call with no batch dimension.
Using ``torch.no_grad()`` is fine but using ``torch.inference_mode()`` is not
(because forces are computed with autograd).

Semiempirical methods
=====================
PM3, PM6, PM6-ORG and PM7 export the same way, from the Python API:

.. code-block:: python

   from xnn.common.deploy import export_torchscript_potential
   from xnn.sqmnn.models import PM7

   export_torchscript_potential(PM7(), cutoff=6.0, path="pm7.pt", total_charge=0.0, spin_multiplicity=1.0)

The file holds the tables of every element of the parameter set and runs the
SCF of the reference implementation, with its corrections. It evaluates one
molecule per call and uses every atom pair, so the whole molecule must be on
one rank. Periodic cells are refused. The neighbor list and the cutoff are not
used by the energy. Each call starts its SCF from the previous density when the
atoms are the same and have moved by less than 0.3 Angstrom, as in a
molecular dynamics run. The energy is the heat of formation in eV, and the
charges are the Mulliken charges.

The graph executor's optimizations take several seconds on the first calls
and do not make later calls faster. In Python, call the model inside
``torch.jit.optimized_execution(False)`` to skip them.

Classical force fields
======================
OPLS, AMBER, CHARMM, DREIDING and classical ReaxFF models export to LAMMPS's
own styles instead of a TorchScript file. :func:`~xnn.ffnn.models.write_lammps_input`
writes a data file with the bound topology and an input script with the
model's parameters, trained ones included:

.. code-block:: python

   from xnn.ffnn.models import write_lammps_input

   script = write_lammps_input(model, pos, cell, "lammps_run", name="box")
   # lammps_run/box.data and lammps_run/in.box; run with: lmp -in in.box

LAMMPS then evaluates the same energy function to round-off. The charges
are scaled so that LAMMPS's Coulomb constant reproduces the model's. The
cell must be orthorhombic, and a structure without a cell becomes a
shrink-wrapped box no smaller than its extent plus the cutoff. A switched cutoff (``switch_width > 0``) has no LAMMPS
style and is refused. OPLS's torsion constants ``V0`` have no LAMMPS term:
the script states their sum as an energy offset. A ReaxFF model with
``conventions="lammps"`` also writes its field as ``<name>.ffield`` and needs
the atomic numbers (``atomic_numbers=z``). LAMMPS solves its charges with
``fix qeq/reaxff`` for a neutral system.

AMOEBA and HIPPO
================
AMOEBA and HIPPO models export to LAMMPS's AMOEBA package with the same
function:

.. code-block:: python

   script = write_lammps_input(model, pos, cell, "lammps_run", name="box")

It writes the parameters (``box.prm``), the coordinates (``box.xyz``), the
settings (``box.key``) and the input script (``in.box``). Make the data file
first with LAMMPS's ``tools/tinker/tinker2lmp.py``. The header of ``in.box``
gives the command.

LAMMPS then reproduces the model's energy, with two limits:

- LAMMPS has no charge flux. The export leaves it out and warns.
- In a cell, the forces of LAMMPS 29Aug2024 do not match its energy.
