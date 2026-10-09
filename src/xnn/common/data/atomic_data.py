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
        molecular systems. In a batch mixing molecular and periodic
        structures, a molecular structure has a zero cell and no periodic
        flag (its edges carry no image shift).
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
    forces_mask : Tensor, optional
        Which structures carry force labels, bool ``(B,)``: set when a batch
        mixes structures with and without them (the missing ones are zeros
        in ``forces`` and excluded from the loss). ``None`` means all do.
    stress_mask : Tensor, optional
        The same for ``stress``.
    dipole : Tensor, optional
        Target dipole moment per structure, of shape ``(B, 3)`` (e Angstrom
        in the xnn units). Present during training of the models with a
        dipole head (PaiNN, PhysNet, AIMNet2).
    polarizability : Tensor, optional
        Target polarizability tensor per structure, of shape ``(B, 3, 3)``
        (Angstrom^3). Present during training of PaiNN's polarizability head.
    dipole_mask, polarizability_mask : Tensor, optional
        The same as ``forces_mask`` for ``dipole`` / ``polarizability``.
    fragment_charges : Tensor, optional
        Net charge, in e, of the fragment each atom belongs to, shape ``(N,)``
        (for the fragment constraints of the LES charge solve); ``None`` when
        unlabeled.
    fragment_charges_mask : Tensor, optional
        Which structures carry ``fragment_charges`` when a batch mixes labeled
        and unlabeled ones, shape ``(B,)``; ``None`` means all of them.
    total_charge : Tensor, optional
        Net charge per structure, of shape ``(B,)``. ``None`` means neutral.
        Read by the charge-aware models (D4 dispersion, PhysNet, ReaxFF,
        AIMNet2).
    spin_multiplicity : Tensor, optional
        Spin multiplicity ``2S + 1`` per structure, of shape ``(B,)``.
        ``None`` means closed shell (1). Read by the open-shell AIMNet2
        models (``aimnet2-nse``).
    weight : Tensor, optional
        Per-structure loss weight, of shape ``(B,)``. ``None`` means every
        structure counts equally, which is the default and reproduces the
        unweighted loss exactly. Read only by
        :func:`~xnn.common.train.losses.weighted_loss`; no model sees it.
    head : Tensor, optional
        Index of the readout head each structure belongs to, of shape
        ``(B,)``. ``None`` means the first head. Read by
        :class:`~xnn.common.finetune.MultiHead` (multi-head fine-tuning) and
        by the loss, which weights every head's terms separately.

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
    forces_mask, stress_mask : Tensor or None
        Bool ``(B,)``: the structures that carry force / stress labels.
    dipole : Tensor or None
        Target dipole moment ``(B, 3)``.
    polarizability : Tensor or None
        Target polarizability tensor ``(B, 3, 3)``.
    dipole_mask, polarizability_mask : Tensor or None
        Bool ``(B,)``: the structures that carry dipole / polarizability labels.
    total_charge : Tensor or None
        Net charge per structure ``(B,)``.
    spin_multiplicity : Tensor or None
        Spin multiplicity per structure ``(B,)``.
    weight : Tensor or None
        Per-structure loss weight ``(B,)``.
    head : Tensor or None
        Readout head per structure ``(B,)``.
    """

    # structure
    pos: Tensor              # (N, 3) cartesian positions
    atomic_numbers: Tensor   # (N,)   integer Z per atom
    edge_index: Tensor       # (2, E) [src, dst]; energy of dst depends on src
    cell_shifts: Tensor      # (E, 3) integer periodic image shift for each edge
    batch: Tensor            # (N,)   structure index per atom (all 0 if single)
    n_atoms: Tensor          # (B,)   atoms per structure
    cell: Optional[Tensor] = None     # (B, 3, 3) lattice vectors as rows
    pbc: Optional[Tensor] = None      # (B, 3) bool periodicity flags

    # targets / labels (optional; present during training)
    energy: Optional[Tensor] = None   # (B,)
    forces: Optional[Tensor] = None   # (N, 3)
    stress: Optional[Tensor] = None   # (B, 3, 3)
    # which structures carry forces / stress when a batch mixes labelled and
    # unlabelled ones; None = all of them
    forces_mask: Optional[Tensor] = None   # (B,) bool
    stress_mask: Optional[Tensor] = None   # (B,) bool
    # tensorial labels of molecules: the dipole moment (e A) and the
    # polarizability tensor (A^3), with the same masks as forces / stress
    dipole: Optional[Tensor] = None            # (B, 3)
    polarizability: Optional[Tensor] = None    # (B, 3, 3)
    dipole_mask: Optional[Tensor] = None           # (B,) bool
    polarizability_mask: Optional[Tensor] = None   # (B,) bool
    # the fragment charge of every atom (LES charge solve); None = unlabeled
    fragment_charges: Optional[Tensor] = None        # (N,) e
    fragment_charges_mask: Optional[Tensor] = None   # (B,) bool; None = all labeled
    # optional per-structure metadata
    total_charge: Optional[Tensor] = None   # (B,) net charge; None = neutral
    spin_multiplicity: Optional[Tensor] = None   # (B,) 2S+1; None = closed shell
    weight: Optional[Tensor] = None         # (B,) loss weight; None = all equal
    head: Optional[Tensor] = None           # (B,) readout head; None = the first
    # dtype the model computes in, when the geometry is kept in a wider one
    # (float64 positions for a float32 model); None = the positions' dtype
    compute_dtype: Optional[torch.dtype] = None

    @property
    def model_dtype(self) -> torch.dtype:
        """torch.dtype : The dtype the model computes in (``compute_dtype``, or
        the positions' dtype when it is not set)."""
        return self.compute_dtype if self.compute_dtype is not None else self.pos.dtype

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

    def subset(self, keep: Tensor) -> "AtomicGraph":
        """The structures flagged in ``keep`` as a batch of their own.

        Nodes and edges of the other structures are dropped and the indices
        renumbered; the kept structures stay in their original order. The
        positions of the result are a view-like gather of ``pos``, so forces
        and stress computed on the subset flow back to the full graph.

        Parameters
        ----------
        keep : Tensor
            Bool tensor of shape ``(B,)``, ``True`` for the structures to keep.

        Returns
        -------
        AtomicGraph
            The sub-batch, with every optional field restricted the same way.
        """
        keep = keep.to(device=self.batch.device, dtype=torch.bool)
        node_keep = keep[self.batch]
        edge_keep = node_keep[self.edge_index[0]]
        new_node = torch.cumsum(node_keep.long(), 0) - 1
        new_graph = torch.cumsum(keep.long(), 0) - 1

        def rows(v: Optional[Tensor], mask: Tensor) -> Optional[Tensor]:
            return None if v is None else v[mask.to(v.device)]

        return AtomicGraph(
            pos=self.pos[node_keep],
            atomic_numbers=self.atomic_numbers[node_keep],
            edge_index=new_node[self.edge_index[:, edge_keep]],
            cell_shifts=self.cell_shifts[edge_keep],
            batch=new_graph[self.batch[node_keep]],
            n_atoms=self.n_atoms[keep],
            cell=rows(self.cell, keep),
            pbc=rows(self.pbc, keep),
            energy=rows(self.energy, keep),
            forces=rows(self.forces, node_keep),
            stress=rows(self.stress, keep),
            forces_mask=rows(self.forces_mask, keep),
            stress_mask=rows(self.stress_mask, keep),
            dipole=rows(self.dipole, keep),
            polarizability=rows(self.polarizability, keep),
            dipole_mask=rows(self.dipole_mask, keep),
            polarizability_mask=rows(self.polarizability_mask, keep),
            fragment_charges=rows(self.fragment_charges, node_keep),
            fragment_charges_mask=rows(self.fragment_charges_mask, keep),
            total_charge=rows(self.total_charge, keep),
            spin_multiplicity=rows(self.spin_multiplicity, keep),
            weight=rows(self.weight, keep),
            head=rows(self.head, keep),
            compute_dtype=self.compute_dtype,
        )

    def edge_vectors(self) -> Tensor:
        """Compute the displacement vector ``r_ij`` for every edge.

        The displacement is ``pos[dst] - pos[src]`` plus, for periodic systems,
        the contribution ``cell_shift @ cell`` of the periodic image. The result
        is differentiable with respect to ``pos`` (forces) and ``cell``
        (stress). Works for molecular (``cell`` is ``None``) and periodic
        systems alike.

        The difference is formed in the dtype of the geometry and returned in
        :attr:`model_dtype`: absolute coordinates of size ``L`` in float32 leave
        every vector with an error of about ``6e-8 L``, so the deploy paths keep
        the positions and cell in float64 for a float32 model and only the
        (short) vectors are rounded.

        Returns
        -------
        Tensor
            Edge displacement vectors ``r_ij`` of shape ``(E, 3)``.
        """
        if self.cell is None or self.cell.shape[0] == 1:
            # molecules and single structures: one function that forms the
            # vectors block by block in the geometry dtype, so a float64
            # geometry never holds (E, 3) float64 temporaries for every edge
            # (4.8 GB at 71 million edges), and whose backward keeps no float
            # copy of the shifts. A single cell is a plain product: gathering
            # it onto every edge makes the stress backward accumulate 9 E
            # values into nine entries (1.5 s of a D4 step on 5000 atoms).
            cell = (self.cell[0].to(self.pos.dtype) if self.cell is not None
                    else self.pos.new_zeros(0))
            return _EdgeVectors.apply(self.pos, cell, self.edge_index, self.cell_shifts,
                                      self.model_dtype)
        src, dst = self.edge_index[0], self.edge_index[1]
        vec = self.pos[dst] - self.pos[src]
        # cell of the structure each edge belongs to (via its src node)
        cell_per_edge = self.cell[self.batch[src]]              # (E, 3, 3)
        vec = vec + torch.einsum("ei,eij->ej", self.cell_shifts.to(vec.dtype), cell_per_edge)
        return vec.to(self.model_dtype)


#: Edges per block of :class:`_EdgeVectors` (its temporaries: three ``(block, 3)``
#: tensors in the geometry dtype, 100 MB each in float64).
EDGE_BLOCK = 1 << 22


class _EdgeVectors(torch.autograd.Function):
    """``pos[dst] - pos[src] + cell_shifts @ cell``, in blocks of edges.

    Each block is formed in the geometry dtype exactly as the whole would be and
    written into the result in the model dtype. The backward accumulates the
    position and cell gradients block by block from the integer shifts; it is
    made of differentiable operations, so force training (``create_graph``)
    differentiates through it.
    """

    @staticmethod
    def forward(ctx, pos: Tensor, cell: Tensor, edge_index: Tensor, cell_shifts: Tensor,
                out_dtype: torch.dtype) -> Tensor:
        ctx.save_for_backward(edge_index, cell_shifts)
        ctx.n_atoms, ctx.dtype, ctx.periodic = pos.shape[0], pos.dtype, cell.numel() > 0
        n_edges = edge_index.shape[1]
        out = torch.empty((n_edges, 3), dtype=out_dtype, device=pos.device)
        for e0 in range(0, n_edges, EDGE_BLOCK):
            e1 = min(n_edges, e0 + EDGE_BLOCK)
            vec = pos[edge_index[1, e0:e1]] - pos[edge_index[0, e0:e1]]
            if ctx.periodic:
                vec = vec + cell_shifts[e0:e1].to(pos.dtype) @ cell
            out[e0:e1] = vec
        return out

    @staticmethod
    def backward(ctx, grad: Tensor):
        edge_index, cell_shifts = ctx.saved_tensors
        need_pos = ctx.needs_input_grad[0]
        need_cell = ctx.needs_input_grad[1] and ctx.periodic
        grad_pos = grad.new_zeros((ctx.n_atoms, 3), dtype=ctx.dtype) if need_pos else None
        grad_cell = grad.new_zeros((3, 3), dtype=ctx.dtype) if need_cell else None
        n_edges = edge_index.shape[1]
        for e0 in range(0, n_edges, EDGE_BLOCK):
            e1 = min(n_edges, e0 + EDGE_BLOCK)
            g = grad[e0:e1].to(ctx.dtype)
            if grad_pos is not None:
                grad_pos = (grad_pos.index_add(0, edge_index[1, e0:e1], g)
                            .index_add(0, edge_index[0, e0:e1], g, alpha=-1))
            if grad_cell is not None:
                grad_cell = grad_cell + cell_shifts[e0:e1].to(ctx.dtype).t() @ g
        return grad_pos, grad_cell, None, None, None
