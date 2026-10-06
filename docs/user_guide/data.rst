.. _data:

*************
Data Pipeline
*************

AtomicGraph: the one data object
================================
Every model consumes an :class:`~xnn.common.data.atomic_data.AtomicGraph`,
a dataclass holding one structure or a batch:

.. code-block:: python

   graph.pos              # (N, 3) positions
   graph.atomic_numbers   # (N,)
   graph.edge_index       # (2, E) neighbor pairs [src, dst] within the cutoff
   graph.cell_shifts      # (E, 3) integer periodic image shift per edge
   graph.batch            # (N,)   structure index per atom
   graph.n_atoms          # (B,)
   graph.cell, graph.pbc  # (B, 3, 3), (B, 3); None for molecules
   graph.energy, graph.forces, graph.stress        # targets, optional
   graph.total_charge, graph.spin_multiplicity     # (B,), optional; neutral / closed shell if None
   graph.weight, graph.head                        # (B,) loss weight and readout head, optional

:meth:`~xnn.common.data.atomic_data.AtomicGraph.edge_vectors` computes
``pos[dst] - pos[src] + cell_shift @ cell`` differentiably, which is where
periodicity lives: models never see the difference between a molecule and a
crystal, and forces and stress follow by autograd.

Structure dictionaries
======================
.. _structure-dicts:

The input format is a plain dictionary per structure:

.. code-block:: python

   {
       "pos": ...,                # (N, 3), required
       "atomic_numbers": ...,     # (N,),   required
       "cell": ..., "pbc": ...,   # (3, 3), (3,): periodic systems
       "energy": ...,             # scalar target
       "forces": ...,             # (N, 3) target
       "stress": ...,             # (3, 3) target
       "total_charge": 0.0,       # or "charge"; optional
       "spin_multiplicity": 1,    # optional
       "weight": 1.0,             # per-structure loss weight, optional
   }

Datasets and batching
=====================
:class:`~xnn.common.data.dataset.AtomicDataset` converts each dictionary to
a graph at a cutoff and caches it. :func:`~xnn.common.data.dataset.collate`
concatenates graphs into one batched graph, which the trainer uses as its
``collate_fn``:

.. code-block:: python

   from xnn.common.data import AtomicDataset, collate

   ds = AtomicDataset(structures, cutoff=5.0)
   batch = collate([ds[0], ds[1], ds[2]])     # one AtomicGraph holding 3 structures

A batch may mix molecules and cells, and structures with and without force
or stress labels: the missing labels are masked out of the loss. Energy
labels are all or nothing.

Files
=====
Any format ASE can read loads in one line (``ase`` extra). Energies, forces
and stress are picked up from the frames; stress is converted from Voigt to
a full matrix:

.. code-block:: python

   ds = AtomicDataset.from_file("trajectory.extxyz", cutoff=5.0)
   ds = AtomicDataset.from_file("dft.extxyz", cutoff=5.0,
                                energy_key="REF_energy", forces_key="REF_forces")   # other key names
   ds = AtomicDataset.from_atoms(list_of_atoms, cutoff=5.0)

Positions need not be wrapped into the cell. The converters
:func:`~xnn.common.data.ase_io.load_structures` and
:func:`~xnn.common.data.ase_io.atoms_to_structure` are public as well.

Neighbor lists
==============
:func:`~xnn.common.data.neighborlist.build_neighbor_list` builds the edges
and image shifts for molecular and periodic structures alike:

.. code-block:: python

   from xnn.common.data import build_neighbor_list

   edge_index, cell_shifts = build_neighbor_list(pos, 5.0, cell=cell, pbc=pbc)

The built-in implementation is brute force and quadratic in the atom count.
With the ``vesin`` extra installed its cell list is used automatically, with
the same edges and conventions; on 5000 atoms at a 6 Å cutoff that is a
thousandfold faster. Geometry is kept in float64 whatever the model dtype.

The data hub
============
Standard benchmark datasets download, convert and cache in one call:

.. code-block:: python

   from xnn.common.data import load_dataset, list_datasets

   list_datasets()
   splits = load_dataset("rmd17", molecule="aspirin")                      # {"train": [...], "test": [...]}
   train = load_dataset("rmd17", molecule="aspirin", split="train", cutoff=5.0)   # AtomicDataset

- ``rmd17``: revised MD17, ten molecules with PBE energies and forces.
  Options ``molecule``, ``fold`` (1 to 5), ``split``, ``units`` (``eV`` or
  ``kcal/mol``), ``n_train`` / ``n_test``.
- ``ani1``: the ANI-1 set, 20 M conformations of H/C/N/O molecules in one
  4.8 GB archive. Options ``heavy_atoms`` (1 to 8), ``max_molecules``,
  ``max_conformations``, ``split``, ``units``.
- ``ani1x``: the ANI-1x set, 5 M conformations with energies and forces
  (5.6 GB). Options ``level`` (``wb97x_dz``, ``wb97x_tz``, ``ccsd(t)_cbs``),
  ``forces``, and the caps and splits of ``ani1``.
- ``ani1ccx``: the coupled-cluster subset of ANI-1x (energies only), from
  the same file.
- ``ani2x``: the seven-element ANI-2x set (H/C/N/O/S/F/Cl) with energies
  and forces (3.7 GB). Options ``n_atoms``, ``forces``, ``max_groups``,
  ``max_conformations``, ``split``, ``units``.
- ``argon_md``: periodic argon with energies, forces and stress, bundled
  with the repository; used by the argon example notebooks.
- ``lode_dimers``: the LODE non-bonded sets (biomolecular dimers, monomers,
  point charges, xenon). Options ``subset``, ``label``, ``return_info``; the
  bundled ``bio_scan`` subset loads offline.

Files are cached and MD5-verified under ``datasets/<name>/`` (``cache_dir``,
``XNN_DATASETS`` or ``XNN_CACHE`` change that). The ANI sets need the ``hub``
extra. :ref:`howto-load-datasets` has examples per dataset;
:ref:`developer-guide-extending` shows how to register a new one.
