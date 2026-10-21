.. _howto-ase:

*********************************
Run Molecular Dynamics with ASE
*********************************

Any xnn model can drive a molecular dynamics simulation with `ASE
<https://wiki.fysik.dtu.dk/ase/>`_ through
:class:`~xnn.common.deploy.ase_calculator.XNNCalculator` (requires the``ase``
extra optional dependency installed).

Attach the calculator
=====================

.. code-block:: python

   from ase.io import read
   from xnn.common.deploy import XNNCalculator

   atoms = read("liquid_argon.xyz")
   atoms.calc = XNNCalculator.from_pretrained("runs/argon_mace/best.pt")   # Or a model hub name

   print(atoms.get_potential_energy(), atoms.get_forces().shape, atoms.get_stress())

The ``from_pretrained`` function takes a valid model hub name and any additional
arguments (e.g., ``device``, ``dtype``, ``dispersion``, ``use_fast``). Your
model object can be passed directly to the xnn calculator:
``XNNCalculator(model, cutoff=5.0)``. Both molecular and periodic systems are
supported.

Run dynamics
============
With the calculator attached to the atomic frames, every ASE MD driver becomes
available. Here is an example of a NPT simulation with a MACE model trained on
liquid argon:

.. code-block:: python

   from ase import units
   from ase.md.npt import NPT
   from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

   # Initialize the velocities to a temperature of 94.4 K
   MaxwellBoltzmannDistribution(atoms, temperature_K=94.4)

   # Set up the NPT dynamics
   dyn = NPT(atoms, timestep=2.0 * units.fs, temperature_K=94.4,
             externalstress=1.0 * units.bar, ttime=25 * units.fs,
             pfactor=(75 * units.fs) ** 2 * units.GPa)
   
   # Run the dynamics for 10,000 steps
   dyn.run(10_000)

Things to keep in mind
======================
- A model with a D4 term runs faster in MD with ``eeq_reuse=True``, which
  carries the charge solve from one step to the next.
- Classical force fields need their customary timesteps: about 0.1 fs for
  ReaxFF, 0.5 to 1 fs for OPLS, DREIDING, AMBER and CHARMM with explicit hydrogens.
- Models that predict charges (AIMNet2, PhysNet, BAMBOO, ReaxFF, OPLS,
  DREIDING, AMBER, CHARMM) expose them as ``atoms.calc.results["charges"]``.
