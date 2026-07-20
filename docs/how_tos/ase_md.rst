.. _howto-ase:

*********************************
Run Molecular Dynamics with ASE
*********************************

Any trained xnns model can drive `ASE <https://wiki.fysik.dtu.dk/ase/>`_ through
:class:`~xnns.common.deploy.ase_calculator.XNNSCalculator` (requires the ``ase``
extra installed).

Attach the calculator
=====================

.. code-block:: python

   import torch
   from ase.io import read
   from xnns.common.deploy import XNNSCalculator
   from xnns.common.models import build_model

   ckpt = torch.load("runs/argon_mace/best.pt", weights_only=False)
   model = build_model(ckpt["cfg"].model)
   model.load_state_dict(ckpt["model"])

   atoms = read("liquid_argon.xyz")
   atoms.calc = XNNSCalculator(model, cutoff=ckpt["cfg"].model.cutoff)

   print(atoms.get_potential_energy(), atoms.get_forces().shape)

The calculator handles molecular and periodic cells alike: it builds the
same PBC-aware neighbor list used in training, so energies, forces, and
stresses are consistent with the training setup.

Run NPT dynamics
================
With the calculator attached, the model works in any ASE dynamics driver.
For example, liquid-argon NPT (the workflow of the
``*_argon_density_md.ipynb`` example notebooks):

.. code-block:: python

   from ase import units
   from ase.md.npt import NPT
   from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

   MaxwellBoltzmannDistribution(atoms, temperature_K=94.4)

   dyn = NPT(
       atoms,
       timestep=2.0 * units.fs,
       temperature_K=94.4,
       externalstress=1.0 * units.bar,
       ttime=25 * units.fs,
       pfactor=(75 * units.fs) ** 2 * units.GPa,
   )
   dyn.run(10_000)

.. tip::

   The notebooks ``examples/gnn/{mace,nequip,allegro}/*_argon_density_md.ipynb``
   run this exact pipeline with both xnns and the corresponding reference
   implementation and compare the resulting mass densities; with identical
   weights the difference is essentially zero.
