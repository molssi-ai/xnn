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

Structure dictionaries
======================
The input format is a plain dictionary per structure:

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

The command line reads structure files (extxyz, VASP, and anything else ASE
understands) into this form with ``ase.io.read``.

.. note::

   For periodic datasets, positions should lie inside the cell (wrapped);
   apply ASE's ``atoms.wrap()`` before extracting positions if your
   trajectory stores unwrapped coordinates.
