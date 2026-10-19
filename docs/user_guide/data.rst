.. _data:

*************
Data Pipeline
*************

AtomicGraph: the data interface
================================
Every model consumes an :class:`~xnn.common.data.atomic_data.AtomicGraph`, a
dataclass object holding one structure or a batch of structures.

.. code-block:: python

   graph.pos              # (N, 3) positions
   graph.atomic_numbers   # (N,)
   graph.edge_index       # (2, E) neighbor pairs [src, dst] within the cutoff
   graph.cell_shifts      # (E, 3) integer periodic image shift per edge
   graph.batch            # (N,) structure index per atom
   graph.n_atoms          # (B,)
   graph.cell, graph.pbc  # (B, 3, 3), (B, 3); None for molecules
   graph.energy, graph.forces, graph.stress        # Targets, optional
   graph.dipole, graph.polarizability              # (B, 3), (B, 3, 3) targets, optional
   graph.charges                                   # (N,) partial-charge targets, optional
   graph.total_charge, graph.spin_multiplicity     # (B,), optional; neutral / closed shell if None
   graph.weight, graph.head                        # (B,) loss weight and readout head, optional
   graph.energy_weight, graph.force_weight, graph.stress_weight   # (B,) term weights, optional
   graph.atom_force_weight                         # (N,) per-atom force weight, optional

The :meth:`~xnn.common.data.atomic_data.AtomicGraph.edge_vectors` method
computes ``pos[dst] - pos[src] + cell_shift @ cell`` differentiably, which is
where periodicity is realized. From models' point of view, there is no
difference between a molecule and a crystal, especially when forces and stresses
are computed by autograd.

Structure dictionaries
======================
.. _structure-dicts:

The input format is a plain dictionary representing each **structure**:

.. code-block:: python

   {
       "pos": ...,                # (N, 3), required
       "atomic_numbers": ...,     # (N,),   required
       "cell": ..., "pbc": ...,   # (3, 3), (3,): periodic systems
       "energy": ...,             # Scalar target
       "forces": ...,             # (N, 3) target
       "stress": ...,             # (3, 3) target
       "dipole": ...,             # (3,) target, e Angstrom; optional
       "polarizability": ...,     # (3, 3) target, Angstrom^3; optional
       "charges": ...,            # (N,) partial charges, e; optional (HDNNP 3G / 4G)
       "total_charge": 0.0,       # Or "charge"; optional
       "spin_multiplicity": 1,    # Optional
       "weight": 1.0,             # Per-structure loss weight, optional
       "energy_weight": 1.0,      # Or force_weight / stress_weight: one term only, optional
       "atom_force_weight": ...,  # (N,) weight of each atom's force error, optional
   }

The weights are explained in :ref:`howto-weight-force-labels`.

Datasets and batching
=====================
The :class:`~xnn.common.data.dataset.AtomicDataset` class converts each
structure dictionary to a graph with a cutoff radius and caches it. The
:func:`~xnn.common.data.dataset.collate` function concatenates these graphs into
one batched graph structure, which the trainer can use as its ``collate_fn``:

.. code-block:: python

   from xnn.common.data import AtomicDataset, collate

   ds = AtomicDataset(structures, cutoff=5.0)
   batch = collate([ds[0], ds[1], ds[2]])     # One AtomicGraph holding 3 structures

A graph batch may mix molecules and cells, and structures with and without
energy, force, stress, dipole or charge labels. The missing labels are masked
out of the loss, automatically. 

Data files
==========
Any ASE-native file format can be loaded into xnn in one line (the ``ase`` extra
dependency is required). Energies, forces and stresses are picked up from the
frames if present. The stress is converted from Voigt to a full matrix form

.. code-block:: python

   ds = AtomicDataset.from_file("trajectory.extxyz", cutoff=5.0)
   ds = AtomicDataset.from_file("dft.extxyz", cutoff=5.0,
                                energy_key="REF_energy", 
                                forces_key="REF_forces")   # Non-default key names
   ds = AtomicDataset.from_atoms(list_of_atoms, cutoff=5.0)

The atomic positions do need not be wrapped into a cell. The converters
:func:`~xnn.common.data.ase_io.load_structures` and
:func:`~xnn.common.data.ase_io.atoms_to_structure` are public and can be used.
Per-atom ``charges`` are read from ``atoms.arrays["charges"]``. The loss weights
``weight``, ``energy_weight``, ``force_weight``, ``stress_weight`` can be read
from the ``atoms.info`` and the ``atom_force_weight`` from ``atoms.arrays`` from
the atomic frames in the same way.

RuNNer ``input.data`` files (bohr, hartree) are read into structure dictionaries
in Angstrom and eV, with reference charges and the total charge:

.. code-block:: python

   from xnn.common.data.runner_io import read_runner_data, write_runner_data

   structures = read_runner_data("input.data")
   write_runner_data(structures, "copy.data")

Neighbor lists
==============
The :func:`~xnn.common.data.neighborlist.build_neighbor_list` builds the edges
and image shifts for molecular and periodic structures alike:

.. code-block:: python

   from xnn.common.data import build_neighbor_list

   edge_index, cell_shifts = build_neighbor_list(pos, 5.0, cell=cell, pbc=pbc)

The built-in implementation is brute-force and quadratic in the atom count. With
the ``vesin`` extra dependency installed, the cell list functionality is used
automatically, with the same edges and conventions. On a system of 5000 atoms at
a 6 Å cutoff radius, this implementation becomes a thousandfold faster. The
geometry is kept in float64 regardless of the model dtype is to avoid numerical
issues.

The data hub
============
Some of the popular standard benchmark datasets can be downloaded, converted to
xnn-native formats and cached in one function call:

.. code-block:: python

   from xnn.common.data import load_dataset, list_datasets

   # List the available datasets and their options
   list_datasets()

   # Load the RMD17 aspirin dataset, with the default split and cutoff
   # All splits returned: {"train": [...], "test": [...]}
   splits = load_dataset("rmd17", molecule="aspirin")

   # Load the training split as an AtomicDataset with a 5 Å cutoff
   train = load_dataset("rmd17", molecule="aspirin", split="train", cutoff=5.0)

Currently, the following datasets are available:

- ``rmd17``: revised MD17, ten molecules with PBE energies and forces.

  - Available arguments: ``molecule``, ``fold`` (1 to 5), ``split``, ``units``
    (``eV`` or ``kcal/mol``), ``n_train`` / ``n_test``.

- ``qm7``: the 7165 QM7 molecules with PBE0 atomization energies and the
  stratified five-fold split (needs the scipy installed).

- ``qm9``: the 133,885 QM9 molecules with 15 B3LYP properties, the 3054
  uncharacterized ones excluded, in random 110,000 / 10,000 / rest splits.

  - Available arguments: ``label`` (``U0`` by default), ``n_train`` / ``n_val`` / ``n_test``,
    ``seed``, ``units`` (``eV`` or ``hartree``), ``exclude_uncharacterized``,
    ``return_info``.

- ``ethanol_response``: 10,000 ethanol conformations at PBE0/def2-TZVP with
  energies, forces, dipole moments and polarizability tensors. This is the data
  of the PaiNN spectra (needs ase installed).
  
  - Available arguments: ``n_train`` / ``n_val`` / ``n_test`` (8000 / 1000 / 1000), ``seed``,
    ``units``, ``shielding``.

- ``ani1``: the ANI-1 set, 20 M conformations of H/C/N/O molecules in one
  4.8 GB archive.

  - Available arguments: ``heavy_atoms`` (1 to 8), ``max_molecules``, ``max_conformations``,
    ``split``, ``units``.

- ``ani1x``: the ANI-1x set, 5 M conformations with energies and forces
  (5.6 GB). 
  
  - Available arguments: ``level`` (``wb97x_dz``, ``wb97x_tz``, ``ccsd(t)_cbs``),
    ``forces``, and the caps and splits of ``ani1``.

- ``ani1ccx``: the coupled-cluster subset of ANI-1x (energies only), from
  the same file.
- ``ani2x``: the seven-element ANI-2x set (H/C/N/O/S/F/Cl) with energies
  and forces (3.7 GB). 
  
  - Available arguments: ``n_atoms``, ``forces``, ``max_groups``, ``max_conformations``,
    ``split``, ``units``.

- ``hdnnp4g``: the four charge-transfer benchmarks of the 4G-HDNNP paper
  with energies, forces and Hirshfeld charges, as one ``all`` split (needs
  ase).
  
  - Available arguments: ``system``: ``carbon_chain``, ``ag_clusters``,
    ``nacl_clusters``, ``au2_mgo`` (periodic) or ``au2_mgo_minima`` (its four
    DFT-optimized structures). The molecular sets and the minima are CC BY-NC
    4.0.

- ``spookynet_states``: the electronic-state sets of the SpookyNet paper,
  2200 Ag3+ / Ag3- clusters or singlet / triplet CH2 molecules at GFN2-xTB
  with total charges, spin multiplicities, energies, forces and dipoles.
  
  - Available arguments: ``system`` (``ag3`` or ``carbene``), ``n_train`` /
    ``n_val`` (1000 / 100, the rest is ``test``), ``seed``.

- ``argon_md``: periodic argon with energies, forces and stress, bundled
  with the repository. This dataset is used by several example notebooks.
- ``lode_dimers``: the LODE non-bonded sets (biomolecular dimers, monomers,
  point charges, xenon).

  - Available arguments: ``subset``, ``label``, ``return_info``. The bundled
    ``bio_scan`` subset loads offline.

- ``s22`` / ``s66``: the S22 and S66 noncovalent dimers at their reference
  geometries with per-atom ``fragment`` labels and CCSD(T)/CBS
  ``interaction_energy`` (kcal/mol).
  
  - Available arguments: ``reference`` (``original`` or, for S22, ``revised``).

- ``gmtkn55``: the GMTKN55 database (Goerigk 2017), 55 sets of main-group
  reaction energies, barrier heights and noncovalent interactions with
  CCSD(T)-quality references in kcal/mol. Returns ``structures`` (species
  with ``total_charge``, ``spin_multiplicity`` and the maintainers' PBEh-3c
  energy) and ``reactions`` (species, stoichiometry, reference). The helpers
  :func:`~xnn.common.data.hub.gmtkn55.reaction_energies` and
  :func:`~xnn.common.data.hub.gmtkn55.wtmad2` score a model.

  - Available arguments: ``version`` (``v2``, the maintainers' updated
    references, or ``v1``, the paper's), ``subsets``, ``elements``,
    ``max_abs_charge``, ``closed_shell``.

Files are cached and MD5-verified under ``datasets/<name>/``. The directory can
also be set by ``cache_dir`` argument or by setting the environment variables
``XNN_DATASETS`` or ``XNN_CACHE``. The ANI sets need the ``hub`` extra
dependency installed. The :ref:`howto-load-datasets` has examples per dataset
and the :ref:`developer-guide-extending` guide shows how to register a new
dataset in the data hub.
