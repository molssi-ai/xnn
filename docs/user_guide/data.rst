.. _data:

*************
Data Pipeline
*************

AtomicGraph — the one data object
=================================
Every xnns model consumes an
:class:`~xnns.common.data.atomic_data.AtomicGraph`, a dataclass holding one
structure or a batch of structures:

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Field
     - Shape
     - Meaning
   * - ``pos``
     - ``(N, 3)``
     - atomic positions
   * - ``atomic_numbers``
     - ``(N,)``
     - element of each atom
   * - ``edge_index``
     - ``(2, E)``
     - neighbor pairs ``[src, dst]`` within the cutoff
   * - ``cell_shifts``
     - ``(E, 3)``
     - integer periodic-image shift of each edge
   * - ``batch``
     - ``(N,)``
     - which structure each atom belongs to
   * - ``n_atoms``
     - ``(B,)``
     - atoms per structure
   * - ``cell``
     - ``(B, 3, 3)``
     - lattice vectors (optional; periodic systems)
   * - ``pbc``
     - ``(B, 3)``
     - periodic flags (optional)
   * - ``energy`` / ``forces`` / ``stress``
     - ``(B,)`` / ``(N, 3)`` / ``(B, 3, 3)``
     - training targets (optional)

Useful members: ``num_graphs``, ``num_nodes``, ``num_edges``, ``.to(device)``,
and :meth:`~xnns.common.data.atomic_data.AtomicGraph.edge_vectors`, which
computes ``pos[dst] - pos[src] + cell_shift @ cell`` — differentiably, so
forces and stress can be obtained by autograd.

Neighbor lists
==============
:func:`~xnns.common.data.neighborlist.build_neighbor_list` builds the edges:

.. code-block:: python

   from xnns.common.data import build_neighbor_list

   edge_index, cell_shifts = build_neighbor_list(pos, cutoff=5.0)               # molecular
   edge_index, cell_shifts = build_neighbor_list(pos, 5.0, cell=cell, pbc=pbc)  # periodic

It is PBC-aware and validated against ASE's neighbor list
(``tests/test_neighborlist.py``). The reference implementation is correct but
brute-force; for very large periodic systems you can swap in a cell-list or
``matscipy`` builder — the ``edge_index`` / ``cell_shifts`` interface is all
a model sees.

Datasets and batching
=====================
:class:`~xnns.common.data.dataset.AtomicDataset` is a
``torch.utils.data.Dataset`` over a list of structure dictionaries; it
converts each to an ``AtomicGraph`` (via
:func:`~xnns.common.data.dataset.structure_to_graph`) and caches the graphs:

.. code-block:: python

   from xnns.common.data import AtomicDataset, collate

   ds = AtomicDataset(structures, cutoff=5.0)
   graph = ds[0]

   batch = collate([ds[0], ds[1], ds[2]])   # one AtomicGraph holding 3 structures

:func:`~xnns.common.data.dataset.collate` batches graphs by concatenation,
offsetting ``edge_index`` — the standard disconnected-graph trick, so a batch
is itself just an ``AtomicGraph``. The
:class:`~xnns.common.train.trainer.Trainer` uses it as the ``collate_fn`` of
its data loaders automatically.

Loading ASE-native files (extxyz, CIF, VASP, ...)
=================================================
Any file format ASE can read loads in one line (requires the ``ase`` extra):

.. code-block:: python

   from xnns.common.data import AtomicDataset

   ds = AtomicDataset.from_file("trajectory.extxyz", cutoff=5.0)
   ds = AtomicDataset.from_file("crystal.cif", cutoff=5.0)

Energy, forces and stress targets are picked up automatically when the file
carries them (from the frame's calculator, with ``atoms.info`` /
``atoms.arrays`` as a fallback); stress is converted from Voigt to a full
``(3, 3)`` matrix. Datasets that store targets under other names — e.g. the
MACE convention ``REF_energy`` / ``REF_forces`` / ``REF_stress`` — pass the
key names explicitly (the CLI equivalents live in ``data.energy_key`` etc.):

.. code-block:: python

   ds = AtomicDataset.from_file("dft.extxyz", cutoff=5.0,
                                energy_key="REF_energy",
                                forces_key="REF_forces",
                                stress_key="REF_stress")

``Atoms`` objects already in memory go through
:meth:`~xnns.common.data.dataset.AtomicDataset.from_atoms`:

.. code-block:: python

   from ase.io import read
   ds = AtomicDataset.from_atoms(read("relaxed.cif"), cutoff=5.0)

The underlying converters are public too —
:func:`~xnns.common.data.ase_io.load_structures` (file → list of structure
dicts) and :func:`~xnns.common.data.ase_io.atoms_to_structure` (one ``Atoms``
→ one dict) — and the :ref:`command line <cli>` reads ``data.train_path`` /
``data.val_path`` through the same path. Positions do *not* need to be
wrapped into the cell first: the neighbor-list builder handles unwrapped
(e.g. MD trajectory) coordinates.

Downloading upstream datasets — ``load_dataset``
================================================
The dataset *hub* downloads and preprocesses standard benchmark datasets in one
call, HuggingFace ``load_dataset()``-style — no manual downloading, unpacking, or
unit conversion:

.. code-block:: python

   from xnns.common.data import load_dataset, list_datasets

   list_datasets()                          # ['ani1', 'argon_md', 'lode_dimers', 'rmd17']

   # all splits, as lists of structure dictionaries
   splits = load_dataset("rmd17", molecule="aspirin")     # {"train": [...], "test": [...]}

   # one split, wrapped as a ready-to-train AtomicDataset
   train = load_dataset("rmd17", molecule="aspirin", split="train", cutoff=5.0)

:func:`~xnns.common.data.hub.base.load_dataset` returns lists of
:ref:`structure dictionaries <structure-dicts>` — a mapping of splits when
``split`` is omitted, a single list otherwise. Pass ``cutoff=`` to get
:class:`~xnns.common.data.dataset.AtomicDataset` objects instead, ready for a
``DataLoader``. Downloaded files are cached and MD5-verified under
``datasets/<name>/`` in the repository by default (override with ``cache_dir=``
or the ``XNNS_DATASETS`` / ``XNNS_CACHE`` environment variable), and a tqdm
progress bar tracks both downloading and preprocessing.
:func:`~xnns.common.data.hub.base.list_datasets` names what is registered:

.. list-table::
   :header-rows: 1
   :widths: 16 22 62

   * - Name
     - Key options
     - Contents
   * - ``rmd17``
     - ``molecule``, ``fold`` (1–5), ``split`` (``train`` / ``test`` /
       ``all``), ``units`` (``eV`` / ``kcal/mol``), ``n_train`` / ``n_test``
     - Revised MD17: ten small molecules with PBE/def2-SVP energies and forces
       and five official 1000-structure train/test splits (converted to eV by
       default).
   * - ``ani1``
     - ``heavy_atoms`` (1–8), ``max_molecules``, ``max_conformations``,
       ``split`` (``train`` / ``val`` / ``test``), ``units`` (``eV`` /
       ``hartree``)
     - The ANI-1 training set (Smith *et al.* 2017): ~20 M off-equilibrium
       conformations and wB97X energies for H/C/N/O organic molecules from
       GDB-11 (pyanitools HDF5). One 4.8 GB archive is downloaded once; select
       heavy-atom subsets and cap the amount materialised.
   * - ``argon_md``
     - ``split`` (``train`` / ``test`` / ``all``)
     - Periodic argon configurations with reference energies, forces and stress
       (bundled with the repository, MACE convention). Used by the
       ``*_argon_*`` example notebooks; no download.
   * - ``lode_dimers``
     - ``subset`` (``bio`` / ``bio_scan`` / ``monomers`` /
       ``point_charges_coulomb`` / ``point_charges_dispersion`` / ``xenon``),
       ``label`` (``CC`` / ``CP`` / ``PP`` / …), ``return_info``
     - LODE non-bonded interactions: biomolecular sidechain dimers (energies and
       forces, tagged by fragment polarity) plus monomers, point-charge toy
       systems, and Xe clusters. ``return_info=True`` attaches per-frame
       metadata (labels, distances, monomer energies) — enough to build
       binding-energy curves. The bundled ``bio_scan`` subset (a curated
       charged/polar dimer distance scan, no download) drives the long-range
       example notebooks.

A runnable, end-to-end walkthrough lives in
``examples/data/load_dataset_tutorial.ipynb``. To add your own dataset, register
a :class:`~xnns.common.data.hub.base.DatasetBuilder` — see
:ref:`developer-guide-extending`.

.. _structure-dicts:

Structure dictionaries
======================
Underneath, the input format is a plain dictionary per structure — use it
directly for data that does not come from ASE:

.. code-block:: python

   {
       "pos": ...,               # (N, 3), required
       "atomic_numbers": ...,    # (N,),   required
       "cell": ...,              # (3, 3), optional
       "pbc": ...,               # (3,),   optional
       "energy": ...,            # scalar, optional target
       "forces": ...,            # (N, 3), optional target
       "stress": ...,            # (3, 3), optional target
   }
