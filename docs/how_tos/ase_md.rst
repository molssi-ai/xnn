.. _howto-ase:

*********************************
Run Molecular Dynamics with ASE
*********************************

Any xnn model drives `ASE <https://wiki.fysik.dtu.dk/ase/>`_ through
:class:`~xnn.common.deploy.ase_calculator.XNNCalculator` (``ase`` extra).

Attach the calculator
=====================

.. code-block:: python

   from ase.io import read
   from xnn.common.deploy import XNNCalculator

   atoms = read("liquid_argon.xyz")
   atoms.calc = XNNCalculator.from_pretrained("runs/argon_mace/best.pt")   # or a hub name

   print(atoms.get_potential_energy(), atoms.get_forces().shape, atoms.get_stress())

``from_pretrained`` takes any source the model hub accepts and the options
of :func:`~xnn.common.models.hub.from_pretrained` (``device``, ``dtype``,
``dispersion``, ``use_fast``). A model object you already hold goes in
directly: ``XNNCalculator(model, cutoff=5.0)``. Molecular and periodic
systems both work; the calculator builds the same neighbor list the trainer
used.

Run dynamics
============
With the calculator attached, every ASE driver works. Liquid-argon NPT, as in
the ``*_argon_density_md`` example notebooks:

.. code-block:: python

   from ase import units
   from ase.md.npt import NPT
   from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

   MaxwellBoltzmannDistribution(atoms, temperature_K=94.4)
   dyn = NPT(atoms, timestep=2.0 * units.fs, temperature_K=94.4,
             externalstress=1.0 * units.bar, ttime=25 * units.fs,
             pfactor=(75 * units.fs) ** 2 * units.GPa)
   dyn.run(10_000)

Good to know
============
- A model with a D4 term runs faster in MD with ``eeq_reuse=True``, which
  carries the charge solve from one step to the next.
- Classical force fields need their customary timesteps: about 0.1 fs for
  ReaxFF, 0.5 to 1 fs for OPLS and DREIDING with explicit hydrogens.
- Models that predict charges (AIMNet2, PhysNet, BAMBOO, ReaxFF, OPLS,
  DREIDING) expose them as ``atoms.calc.results["charges"]``.
