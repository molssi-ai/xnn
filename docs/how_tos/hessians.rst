.. _howto-hessians:

*****************************************
Compute Hessians and Harmonic Frequencies
*****************************************

Any xnn model gives the Hessian of its energy with respect to the positions.
The harmonic analysis then turns it into frequencies and a zero-point energy.

.. code-block:: python

   import torch
   from ase.build import molecule
   from xnn.common.data import structure_to_graph
   from xnn.common.models import harmonic_analysis, hessian
   from xnn.sqmnn.models import PM7

   atoms = molecule("CH3CHO")
   graph = structure_to_graph({"pos": torch.tensor(atoms.positions),
                               "atomic_numbers": atoms.numbers}, 0.1)
   model = PM7()

   (H,) = hessian(model, graph)          # (3n, 3n) in eV / Angstrom^2
   out = harmonic_analysis(H, graph.pos, atomic_numbers=graph.atomic_numbers)
   print(out["frequencies"])             # cm^-1, imaginary ones negative
   print(out["zero_point_energy"])       # eV

- :func:`~xnn.common.models.hessian` returns one matrix per structure of a batch.
- The harmonic analysis projects out translations and rotations. It treats a
  molecule as linear when its smallest moment of inertia is close to zero.
- Pass ``masses=`` to use other masses than the standard atomic weights.
- Frequencies are only meaningful at a minimum, so optimize the geometry first.

Curvatures along given directions
=================================
:func:`~xnn.common.models.hessian_vector_product` gives ``H v`` without forming
the whole matrix. Several directions share one evaluation of the model.

.. code-block:: python

   from xnn.common.models import hessian_vector_product

   modes = out["displacements"]                    # (K, n, 3)
   hv = hessian_vector_product(model, graph, modes)
   curvatures = (modes * hv).sum((1, 2))           # the eigenvalues of the modes

Training on curvatures
======================
With ``create_graph=True`` the products stay differentiable. A loss built on
them trains the model's parameters, for example on reference curvatures along
fixed normal modes.

.. code-block:: python

   model = PM7(trainable=["USS", "UPP", "BETAS", "BETAP"])
   hv = hessian_vector_product(model, graph, modes, create_graph=True)
   loss = ((modes * hv).sum((1, 2)) - reference).pow(2).sum()
   loss.backward()

The semiempirical methods (PM3, PM6, PM6-ORG, PM7) give exact Hessians and
exact parameter gradients of such losses. Their self-consistent density then
follows the positions as well. These functions switch this on for their own
evaluations, and ``position_response=True`` keeps it on for every call.

The acetylenic C-C term
=======================
PM6, PM7 and PM6-ORG add 12 kcal/mol to every C-C bond shorter than 1.21
Angstrom and switch this off by 1.33 Angstrom. The switch is steep, so a C=C
bond near 1.33 Angstrom has a stretching frequency that changes quickly with
its length. For fits to curvatures, a smooth and wider switch helps.

.. code-block:: python

   model = PM7(yne_switch="smooth",
               correction_constants={"YNE": [12.0, 1.21, 1.45]},
               trainable=["YNE", "USS", "BETAS"])

- ``yne_switch="smooth"`` makes the Hessian change smoothly with the bond length.
- ``correction_constants`` sets the energy and the start and end of the switch.
  With ``"YNE"`` in ``trainable``, the fit refines them.
- To check a Hessian by finite differences where a bond is inside the switch,
  use small steps of about 1e-4 Angstrom.

The PM7 hydrogen-bond term
==========================
PM7's hydrogen-bond term jumps where its search ends at 7 Angstrom and where it
switches between choices of reference atoms. Its short O...O term also has a
hump in its curvature. So, frequencies of strong O-H...O bonds can jump. Where
a hydrogen bond is planar or linear, its forces have kinks as well.

.. code-block:: python

   model = PM7(hbond_switch="smooth", hbond_torsion="smooth",
               trainable=["HBOND", "USS", "BETAS"])

- ``hbond_switch="smooth"`` replaces each jump with a narrow switch, so the
  energy is continuous.
- ``hbond_torsion="smooth"`` also rounds the kinks of planar and linear
  hydrogen bonds, so forces and Hessians are continuous too.
- Away from the switches the energy is PM7's.
- The energy fades out between 7 and 8 Angstrom.

Bond-order and resonance terms
==============================
The acetylenic term switches on with the bond length, which barely differs
between triple bonds and some double bonds. A switch on the bond order of the
SCF density separates them clearly and keeps Hessians smooth.

.. code-block:: python

   model = PM7(yne_switch="off",
               bond_order_terms={"C-C": [11.09, 2.20, 2.65]},
               trainable=["BOND_ORDER"])

- Each entry is an energy in kcal/mol and the range of bond orders over which
  it switches on.
- ``yne_switch="off"`` removes the acetylenic term that this one replaces.
- Forces, Hessians and parameter gradients stay exact.
- ``resonance_overlap`` is a second option: it rescales the resonance
  integrals of p orbitals per element pair.

Tight-binding methods
=====================
GFN0-xTB, GFN1-xTB and GFN2-xTB also give exact Hessians and exact parameter
gradients of curvature losses, with the same functions.

- The Hessian is as accurate as the converged charges. Use a tight
  ``accuracy``, for example ``GFN2(accuracy=1e-4)``.
- With ``create_graph=True`` the evaluation runs without the fast path's
  kernels, which have no third derivatives.
