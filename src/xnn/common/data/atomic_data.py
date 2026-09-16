"""The single data abstraction that flows through the whole library.

`AtomicGraph` represents one *or* a batch of atomic systems (molecular or
periodic) as a graph. Every model in `xnn.common.models` consumes this object and
nothing else, which is what keeps the model interface coherent.

Design notes
------------
* Positions, cell and the integer ``cell_shifts`` are kept *separately* from
  edge displacement vectors. The displacement ``r_ij`` is recomputed inside the
  model as ``pos[dst] - pos[src] + cell_shift @ cell`` so that autograd can flow
  back to ``pos`` (forces) and ``cell`` (stress). See ``models.outputs``.
* A batch is just several graphs concatenated along the node/edge axes with a
  ``batch`` vector mapping each node to its structure index. This is the same
  convention PyTorch Geometric uses, so interop is easy if you later want it.
* This is a plain dataclass of tensors -- no heavyweight dependency. ``.to()``
  moves everything to a device in one call.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import torch
from torch import Tensor


@dataclass
class AtomicGraph:
    """Graph representation of one or a batch of atomic systems.

    ``AtomicGraph`` is the single data abstraction that flows through the whole
    library: every model in ``xnn.common.models`` consumes this object and
    nothing else. It represents molecular or periodic systems as a graph where
    nodes are atoms and edges connect neighboring atoms within a cutoff. A batch
    is several graphs concatenated along the node/edge axes, with a ``batch``
    vector mapping each node to its structure index (the PyTorch Geometric
    convention).

    Positions, cell and the integer ``cell_shifts`` are stored separately from
    edge displacement vectors; the displacement ``r_ij`` is recomputed inside
    the model (see :meth:`edge_vectors`) so that autograd can flow back to
    ``pos`` (forces) and ``cell`` (stress).

    Parameters
    ----------
    pos : Tensor
        Cartesian positions of shape ``(N, 3)``.
    atomic_numbers : Tensor
        Integer atomic number ``Z`` per atom, of shape ``(N,)``.
    edge_index : Tensor
        Edge list of shape ``(2, E)`` holding ``[src, dst]`` node indices; the
        energy of ``dst`` depends on ``src``.
    cell_shifts : Tensor
        Integer periodic image shift for each edge, of shape ``(E, 3)``.
    batch : Tensor
        Structure index per atom, of shape ``(N,)`` (all zero for a single
        structure).
    n_atoms : Tensor
        Number of atoms per structure, of shape ``(B,)``.
    cell : Tensor, optional
        Lattice vectors as rows, of shape ``(B, 3, 3)``. ``None`` for
        molecular systems.
    pbc : Tensor, optional
        Boolean periodicity flags, of shape ``(B, 3)``. ``None`` for molecular
        systems.
    energy : Tensor, optional
        Target energy per structure, of shape ``(B,)``. Present during
        training.
    forces : Tensor, optional
        Target forces, of shape ``(N, 3)``. Present during training.
    stress : Tensor, optional
        Target stress, of shape ``(B, 3, 3)``. Present during training.

    Attributes
    ----------
    pos : Tensor
        Cartesian positions ``(N, 3)``.
    atomic_numbers : Tensor
        Integer atomic number ``Z`` per atom ``(N,)``.
    edge_index : Tensor
        Edge list ``(2, E)`` as ``[src, dst]``.
    cell_shifts : Tensor
        Integer periodic image shift per edge ``(E, 3)``.
    batch : Tensor
        Structure index per atom ``(N,)``.
    n_atoms : Tensor
        Atoms per structure ``(B,)``.
    cell : Tensor or None
        Lattice vectors as rows ``(B, 3, 3)``.
    pbc : Tensor or None
        Boolean periodicity flags ``(B, 3)``.
    energy : Tensor or None
        Target energy ``(B,)``.
    forces : Tensor or None
        Target forces ``(N, 3)``.
    stress : Tensor or None
        Target stress ``(B, 3, 3)``.
    """

    # --- structure ---
    pos: Tensor              # (N, 3) cartesian positions
    atomic_numbers: Tensor   # (N,)   integer Z per atom
    edge_index: Tensor       # (2, E) [src, dst]; energy of dst depends on src
    cell_shifts: Tensor      # (E, 3) integer periodic image shift for each edge
    batch: Tensor            # (N,)   structure index per atom (all 0 if single)
    n_atoms: Tensor          # (B,)   atoms per structure
    cell: Optional[Tensor] = None     # (B, 3, 3) lattice vectors as rows
    pbc: Optional[Tensor] = None      # (B, 3) bool periodicity flags

    # --- targets / labels (optional; present during training) ---
    energy: Optional[Tensor] = None   # (B,)
    forces: Optional[Tensor] = None   # (N, 3)
    stress: Optional[Tensor] = None   # (B, 3, 3)

    @property
    def num_graphs(self) -> int:
        """int : Number of structures in the batch (``B``)."""
        return int(self.n_atoms.shape[0])

    @property
    def num_nodes(self) -> int:
        """int : Total number of atoms (nodes) across the batch (``N``)."""
        return int(self.pos.shape[0])

    @property
    def num_edges(self) -> int:
        """int : Total number of edges across the batch (``E``)."""
        return int(self.edge_index.shape[1])

    def to(self, device: torch.device | str) -> "AtomicGraph":
        """Return a copy of the graph with all tensors moved to ``device``.

        Parameters
        ----------
        device : torch.device or str
            Target device to move every tensor field to.

        Returns
        -------
        AtomicGraph
            A new ``AtomicGraph`` whose tensor fields live on ``device``.
            Non-tensor fields (e.g. ``None``) are copied unchanged.
        """
        kwargs = {}
        for f in fields(self):
            v = getattr(self, f.name)
            kwargs[f.name] = v.to(device) if isinstance(v, Tensor) else v
        return AtomicGraph(**kwargs)

    def edge_vectors(self) -> Tensor:
        """Compute the displacement vector ``r_ij`` for every edge.

        The displacement is ``pos[dst] - pos[src]`` plus, for periodic systems,
        the contribution ``cell_shift @ cell`` of the periodic image. The result
        is differentiable with respect to ``pos`` (forces) and ``cell``
        (stress). Works for molecular (``cell`` is ``None``) and periodic
        systems alike.

        Returns
        -------
        Tensor
            Edge displacement vectors ``r_ij`` of shape ``(E, 3)``.
        """
        src, dst = self.edge_index[0], self.edge_index[1]
        vec = self.pos[dst] - self.pos[src]
        if self.cell is not None:
            # cell of the structure each edge belongs to (via its src node)
            cell_per_edge = self.cell[self.batch[src]]              # (E, 3, 3)
            shift = torch.einsum("ei,eij->ej",
                                 self.cell_shifts.to(vec.dtype), cell_per_edge)
            vec = vec + shift
        return vec
