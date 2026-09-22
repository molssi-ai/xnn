.. _developer-guide-testing:

*******
Testing
*******

The test suite lives in a flat ``tests/`` directory and runs with pytest:

.. code-block:: bash

   pip install -e ".[dev,gnn]"
   pytest tests/

What is covered
===============
.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - File
     - Coverage
   * - ``test_smoke.py``
     - registry, molecular forces, periodic stress, a batched train step
   * - ``test_neighborlist.py``
     - periodic edge lengths within cutoff, agreement with ASE's neighbor
       list, zero shifts for molecular systems
   * - ``test_gnn.py``
     - featurizer shapes and signal, triplets, rotation invariance,
       equivariance of every GNN model, periodic stress, upstream key
       translation
   * - ``test_mace.py``
     - U-matrix shapes, symmetric-contraction equivariance, flexible
       ``num_interactions`` (including T=0), pair repulsion,
       TorchScript/LAMMPS export, key translation; the ScaleShiftMACE
       energy expression, hand-recomputed Agnesi/Soft distance transforms,
       density interaction blocks, and **foundation-checkpoint conversion
       parity with mace-torch** (every architecture flavor plus multi-head
       slicing, built in-process; a cached-checkpoint test when one is on
       disk -- tests never download)
   * - ``test_nequip.py``
     - hidden-irreps order, ``_Gate`` vs. e3nn's ``Gate``, equivariance,
       per-species scale/shift, export, key translation, **parity with
       upstream** ``nequip`` given identical weights
   * - ``test_allegro.py``
     - equivariance, periodic stress, export, key translation, **parity
       with upstream** ``allegro`` given identical weights
   * - ``test_schnet.py``
     - **parity with an equation-by-equation reference forward** built from
       the NIPS 2017 manuscript (SchNet is a clean-room build, so the paper
       — not schnetpack — is the reference), invariances, forces vs. finite
       differences, cosine-cutoff continuity, size extensivity, batching,
       TorchScript/LAMMPS export, schnetpack key translation
   * - ``test_cace.py`` / ``test_physnet.py`` / ``test_ani.py`` /
       ``test_bamboo.py`` / ``test_les.py``
     - the same pattern for the other faithful implementations:
       invariance/equivariance and **parity with the upstream code** given
       identical weights (upstream packages required where applicable),
       plus each model's specific conventions
   * - ``test_reaxff.py``
     - **equation-by-equation references from the ReaxFF papers** (bond
       orders and corrections, bond energy, analytic two-atom EEM, van der
       Waals dimer, the water valence angle, hydrogen bonds, brute-force
       angle/torsion enumeration), invariances, forces vs. finite
       differences, size extensivity, batching, ``total_charge`` handling,
       trainable-group selection and training steps, ``ffield`` text and
       JSON library round-trips, key translation (self-contained: no
       third-party ReaxFF code is used -- see the fidelity notes)
   * - ``test_opls.py``
     - **hand-recomputed OPLS equations** (harmonic bond/angle, Fourier
       torsion with exact 1,4 scaling, ``V2`` impropers, the
       Lennard-Jones/Coulomb dimer), **parity with OpenMM** on randomized
       conformations (skipped without ``openmm``), the relaxed ethane
       barrier of the 1996 paper, topology derivation counts, invariances,
       forces vs. finite differences, minimum-image bonded terms and
       periodic stress, batching, size extensivity, trainable-group
       selection and training steps, shared-force-field gradients,
       native-JSON and SEAMM ``.frc`` library readers, key translation
   * - ``test_dreiding.py``
     - **hand-recomputed DREIDING equations** (additive bond radii and
       bond-order scaling, the harmonic-cosine and linear angle forms, the
       Morse bond, the torsion of eq 13 with its per-bond barrier
       splitting, spectroscopic inversions, the Lennard-Jones and
       exponential-6 nonbonds with their combination rules, the Coulomb
       constant of eq 37 and the 12-10 hydrogen bond of eq 38), **the nine
       torsion rules of eqs 14-23** (each branch, and symmetry under
       reversal), the published Tables I/II/III/V, the exact eclipsed-ethane
       barrier and hydrogen-bond minimum, invariances, forces vs. finite
       differences, minimum-image bonded terms and periodic stress,
       batching, size extensivity, SMARTS typing with bond-order perception,
       Gasteiger charges, trainable-generator selection and training steps,
       shared-force-field gradients, library JSON round-trips, key
       translation (self-contained: the LAMMPS cross-check lives in the
       fidelity notebook)
   * - ``test_hub.py`` / ``test_ase_io.py`` / ``test_benchmark.py`` /
       ``test_trainer_distributed.py``
     - dataset hub builders and caching, ASE file I/O, the benchmark
       runner/config, and ``torchrun`` DDP training

The equivariance tests rotate the inputs and check that energies are
invariant and forces co-rotate (errors ~1e-7). The parity tests require the
reference packages (the ``examples`` extra) and reproduce upstream outputs
to ~1e-15/1e-16 with transplanted weights.

Conventions
===========
- New models should get, at minimum: a registration test, an equivariance
  or invariance test, a periodic-stress test, and, if deployable, a
  script-vs-eager parity test.
- Faithful re-implementations should additionally pin down parity with the
  upstream code under transplanted weights, guarded by an import check so
  the suite still runs without the reference package installed. For a
  clean-room build whose reference is a manuscript rather than a code base
  (SchNet), pin down parity with an independent implementation of the
  paper's equations instead.
