"""Datasets and graph batching.

`AtomicDataset` turns a list of plain structure dicts into `AtomicGraph`
single-structure objects (computing the neighbor list once, cached). `collate`
concatenates several `AtomicGraph`s into one batched `AtomicGraph` -- this is
what enables optional *batch training*: set ``batch_size=1`` to disable it.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .atomic_data import AtomicGraph
from .neighborlist import build_neighbor_list


def structure_to_graph(s: dict, cutoff: float) -> AtomicGraph:
    """Build a single-structure :class:`AtomicGraph` from a dict of arrays.

    Values may be array-likes or tensors; they are coerced to tensors. The
    neighbor list is computed via :func:`build_neighbor_list`. When ``cell`` is
    provided but ``pbc`` is not, full periodicity is assumed.

    Parameters
    ----------
    s : dict
        Structure data. Required keys: ``pos`` ``(N, 3)`` and
        ``atomic_numbers`` ``(N,)``. Optional keys: ``cell`` ``(3, 3)``,
        ``pbc`` ``(3,)``, ``energy`` (scalar), ``forces`` ``(N, 3)`` and
        ``stress`` ``(3, 3)``.
    cutoff : float
        Neighbor cutoff radius passed to the neighbor list builder.

    Returns
    -------
    AtomicGraph
        A single-structure graph (``batch`` all zeros, ``n_atoms`` of length
        one), with optional target fields populated when present in ``s``.
    """
    def t(x, dtype=torch.get_default_dtype()):
        """Coerce ``x`` to a tensor, leaving existing tensors untouched.

        Parameters
        ----------
        x : Tensor or array_like
            Value to convert.
        dtype : torch.dtype, optional
            Target dtype used only when ``x`` is not already a tensor;
            defaults to the current PyTorch default dtype.

        Returns
        -------
        Tensor
            ``x`` as a tensor.
        """
        return x if isinstance(x, Tensor) else torch.as_tensor(x, dtype=dtype)

    pos = t(s["pos"])
    z = t(s["atomic_numbers"], dtype=torch.long)
    cell = t(s["cell"]).reshape(3, 3) if s.get("cell") is not None else None
    pbc = (torch.as_tensor(s["pbc"], dtype=torch.bool)
           if s.get("pbc") is not None else
           (torch.ones(3, dtype=torch.bool) if cell is not None else None))

    edge_index, cell_shifts = build_neighbor_list(pos, cutoff, cell, pbc)

    n = pos.shape[0]
    return AtomicGraph(
        pos=pos,
        atomic_numbers=z,
        edge_index=edge_index,
        cell_shifts=cell_shifts,
        batch=torch.zeros(n, dtype=torch.long),
        n_atoms=torch.tensor([n], dtype=torch.long),
        cell=cell[None] if cell is not None else None,
        pbc=pbc[None] if pbc is not None else None,
        energy=t([s["energy"]]).reshape(1) if s.get("energy") is not None else None,
        forces=t(s["forces"]) if s.get("forces") is not None else None,
        stress=t(s["stress"]).reshape(1, 3, 3) if s.get("stress") is not None else None,
    )


class AtomicDataset(Dataset):
    """PyTorch ``Dataset`` wrapping a list of structure dicts.

    Each structure is converted to a single-structure :class:`AtomicGraph` via
    :func:`structure_to_graph` on first access and cached, so the neighbor list
    for a given index is only computed once.

    Parameters
    ----------
    structures : list of dict
        Structure dicts, each in the format accepted by
        :func:`structure_to_graph`.
    cutoff : float
        Neighbor cutoff radius used when building each graph.

    Attributes
    ----------
    structures : list of dict
        The wrapped structure dicts.
    cutoff : float
        Neighbor cutoff radius.
    """

    def __init__(self, structures: list[dict], cutoff: float):
        self.structures = structures
        self.cutoff = cutoff
        self._cache: dict[int, AtomicGraph] = {}

    def __len__(self) -> int:
        """Return the number of structures in the dataset.

        Returns
        -------
        int
            Number of wrapped structures.
        """
        return len(self.structures)

    def __getitem__(self, idx: int) -> AtomicGraph:
        """Return the :class:`AtomicGraph` for structure ``idx``.

        The graph is built on first access and cached for subsequent calls.

        Parameters
        ----------
        idx : int
            Index of the structure.

        Returns
        -------
        AtomicGraph
            The single-structure graph at ``idx``.
        """
        if idx not in self._cache:
            self._cache[idx] = structure_to_graph(self.structures[idx], self.cutoff)
        return self._cache[idx]


def collate(graphs: list[AtomicGraph]) -> AtomicGraph:
    """Concatenate single-structure graphs into one batched graph.

    Nodes and edges are concatenated along their respective axes, ``edge_index``
    entries are offset by the running node count, and a ``batch`` vector mapping
    each node to its structure index is built. Optional fields (``cell``/``pbc``,
    ``energy``, ``forces``, ``stress``) are only included when present in every
    input graph. Using ``batch_size=1`` effectively disables batching.

    Parameters
    ----------
    graphs : list of AtomicGraph
        Single-structure graphs to concatenate.

    Returns
    -------
    AtomicGraph
        One batched graph with ``num_graphs`` equal to ``len(graphs)``.
    """
    pos, z, batch, n_atoms = [], [], [], []
    edge_index, cell_shifts = [], []
    cells, pbcs = [], []
    energies, forces, stresses = [], [], []

    node_offset = 0
    has_cell = all(g.cell is not None for g in graphs)
    has_e = all(g.energy is not None for g in graphs)
    has_f = all(g.forces is not None for g in graphs)
    has_s = all(g.stress is not None for g in graphs)

    for b, g in enumerate(graphs):
        pos.append(g.pos)
        z.append(g.atomic_numbers)
        batch.append(torch.full((g.num_nodes,), b, dtype=torch.long))
        n_atoms.append(g.n_atoms)
        edge_index.append(g.edge_index + node_offset)
        cell_shifts.append(g.cell_shifts)
        node_offset += g.num_nodes
        if has_cell:
            cells.append(g.cell)
            pbcs.append(g.pbc)
        if has_e:
            energies.append(g.energy)
        if has_f:
            forces.append(g.forces)
        if has_s:
            stresses.append(g.stress)

    return AtomicGraph(
        pos=torch.cat(pos, 0),
        atomic_numbers=torch.cat(z, 0),
        edge_index=torch.cat(edge_index, 1),
        cell_shifts=torch.cat(cell_shifts, 0),
        batch=torch.cat(batch, 0),
        n_atoms=torch.cat(n_atoms, 0),
        cell=torch.cat(cells, 0) if has_cell else None,
        pbc=torch.cat(pbcs, 0) if has_cell else None,
        energy=torch.cat(energies, 0) if has_e else None,
        forces=torch.cat(forces, 0) if has_f else None,
        stress=torch.cat(stresses, 0) if has_s else None,
    )
