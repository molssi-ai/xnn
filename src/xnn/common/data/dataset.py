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


def structure_to_graph(
    s: dict, cutoff: float, device: Optional[torch.device] = None
) -> AtomicGraph:
    """Build a single-structure :class:`AtomicGraph` from a dict of arrays.

    Values may be array-likes or tensors; they are coerced to tensors. The
    neighbor list is computed via :func:`build_neighbor_list`. When ``cell`` is
    provided but ``pbc`` is not, full periodicity is assumed.

    Parameters
    ----------
    s : dict
        Structure data. Required keys: ``pos`` ``(N, 3)`` and
        ``atomic_numbers`` ``(N,)``. Optional keys: ``cell`` ``(3, 3)``,
        ``pbc`` ``(3,)``, ``energy`` (scalar), ``forces`` ``(N, 3)``,
        ``stress`` ``(3, 3)``, ``total_charge`` (scalar net charge; the
        key ``charge`` is accepted as a synonym) and ``weight`` (scalar
        per-structure loss weight; see
        :func:`~xnn.common.train.losses.weighted_loss`).
    cutoff : float
        Neighbor cutoff radius passed to the neighbor list builder.
    device : torch.device, optional
        Device to build the graph on. The neighbor list is the expensive part
        of this function, so building it where the model already is avoids
        paying for it on the CPU and copying the result across. Defaults to
        ``None``, which builds on the CPU as before -- the right choice when
        the graph is being cached by a dataset rather than fed straight to a
        model.

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
        out = x if isinstance(x, Tensor) else torch.as_tensor(x, dtype=dtype)
        return out if device is None else out.to(device)

    pos = t(s["pos"])
    z = t(s["atomic_numbers"], dtype=torch.long)
    cell = t(s["cell"]).reshape(3, 3) if s.get("cell") is not None else None
    pbc = (torch.as_tensor(s["pbc"], dtype=torch.bool)
           if s.get("pbc") is not None else
           (torch.ones(3, dtype=torch.bool) if cell is not None else None))

    edge_index, cell_shifts = build_neighbor_list(pos, cutoff, cell, pbc)

    n = pos.shape[0]
    charge = s.get("total_charge", s.get("charge"))
    return AtomicGraph(
        pos=pos,
        atomic_numbers=z,
        edge_index=edge_index,
        cell_shifts=cell_shifts,
        batch=torch.zeros(n, dtype=torch.long, device=device),
        n_atoms=torch.tensor([n], dtype=torch.long, device=device),
        cell=cell[None] if cell is not None else None,
        pbc=pbc[None] if pbc is not None else None,
        energy=t([s["energy"]]).reshape(1) if s.get("energy") is not None else None,
        forces=t(s["forces"]) if s.get("forces") is not None else None,
        stress=t(s["stress"]).reshape(1, 3, 3) if s.get("stress") is not None else None,
        total_charge=(t([charge]).reshape(1) if charge is not None else None),
        weight=(t([s["weight"]]).reshape(1) if s.get("weight") is not None else None),
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

    @classmethod
    def from_file(cls, path: str, cutoff: float, index: str = ":",
                  **target_keys) -> "AtomicDataset":
        """Build a dataset from a structure file readable by ASE.

        Any ASE-readable format works (``.xyz`` / ``.extxyz`` / ``.cif`` /
        VASP / ...); frames are converted via
        :func:`~xnn.common.data.ase_io.load_structures`, picking up energy /
        forces / stress targets when the file carries them. Requires the
        ``ase`` extra.

        Parameters
        ----------
        path : str
            Path to the structure file.
        cutoff : float
            Neighbor cutoff radius used when building each graph.
        index : str, optional
            Frame selection passed to :func:`ase.io.read`; the default ``":"``
            loads all frames.
        **target_keys
            ``energy_key`` / ``forces_key`` / ``stress_key`` overrides for
            files that store targets under non-standard names (e.g.
            ``energy_key="REF_energy"``); see
            :func:`~xnn.common.data.ase_io.atoms_to_structure`.

        Returns
        -------
        AtomicDataset
            Dataset over all selected frames.
        """
        from .ase_io import load_structures
        return cls(load_structures(path, index, **target_keys), cutoff)

    @classmethod
    def from_atoms(cls, atoms, cutoff: float, **target_keys) -> "AtomicDataset":
        """Build a dataset from ASE ``Atoms`` object(s) already in memory.

        Parameters
        ----------
        atoms : ase.Atoms or list of ase.Atoms
            Structure(s) to convert, via
            :func:`~xnn.common.data.ase_io.atoms_to_structure`.
        cutoff : float
            Neighbor cutoff radius used when building each graph.
        **target_keys
            ``energy_key`` / ``forces_key`` / ``stress_key`` overrides; see
            :func:`~xnn.common.data.ase_io.atoms_to_structure`.

        Returns
        -------
        AtomicDataset
            Dataset over the given structure(s).
        """
        from .ase_io import atoms_to_structure
        if not isinstance(atoms, (list, tuple)):
            atoms = [atoms]
        return cls([atoms_to_structure(a, **target_keys) for a in atoms], cutoff)

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
    """Concatenate graphs into one batched graph.

    Nodes and edges are concatenated along their respective axes, ``edge_index``
    entries are offset by the running node count, and a ``batch`` vector mapping
    each node to its structure index is built. Using ``batch_size=1``
    effectively disables batching.

    A batch may mix any structures:

    * molecular and periodic ones: when any structure has a cell, the batch
      keeps ``cell`` / ``pbc`` and a molecular structure gets a zero cell and no
      periodic flag (its edges carry no image shift, so every edge keeps its
      length, and its stress is zero);
    * structures with and without force or stress labels: the missing ones are
      zeros, and ``forces_mask`` / ``stress_mask`` (per structure) mark the
      labelled ones for the loss and the metrics;
    * structures with and without ``total_charge`` (missing = neutral) or
      ``weight`` (missing = 1).

    ``energy`` is kept only when every structure has one.

    Parameters
    ----------
    graphs : list of AtomicGraph
        Graphs (single structures or batches) to concatenate.

    Returns
    -------
    AtomicGraph
        One batched graph with ``num_graphs`` equal to the total number of
        structures.
    """
    pos, z, batch, n_atoms = [], [], [], []
    edge_index, cell_shifts = [], []
    cells, pbcs = [], []
    energies, forces, stresses, charges, weights = [], [], [], [], []
    f_masks, s_masks = [], []

    dtype, device = graphs[0].pos.dtype, graphs[0].pos.device
    has_cell = any(g.cell is not None for g in graphs)
    has_e = all(g.energy is not None for g in graphs)
    has_f = any(g.forces is not None for g in graphs)
    has_s = any(g.stress is not None for g in graphs)
    has_q = any(g.total_charge is not None for g in graphs)
    has_w = any(g.weight is not None for g in graphs)

    def label_mask(g: AtomicGraph, value, mask) -> Tensor:
        if mask is not None:
            return mask
        return torch.full((g.num_graphs,), value is not None, dtype=torch.bool, device=device)

    node_offset = 0
    graph_offset = 0
    for g in graphs:
        b = g.num_graphs
        pos.append(g.pos)
        z.append(g.atomic_numbers)
        batch.append(g.batch + graph_offset)
        n_atoms.append(g.n_atoms)
        edge_index.append(g.edge_index + node_offset)
        cell_shifts.append(g.cell_shifts)
        node_offset += g.num_nodes
        graph_offset += b
        if has_cell:
            if g.cell is not None:
                cells.append(g.cell)
                # graphs built on a GPU keep pbc on the CPU (structure_to_graph)
                pbcs.append(g.pbc.to(device) if g.pbc is not None
                            else torch.ones((b, 3), dtype=torch.bool, device=device))
            else:
                cells.append(torch.zeros((b, 3, 3), dtype=dtype, device=device))
                pbcs.append(torch.zeros((b, 3), dtype=torch.bool, device=device))
        if has_e:
            energies.append(g.energy)
        if has_f:
            forces.append(g.forces if g.forces is not None else torch.zeros_like(g.pos))
            f_masks.append(label_mask(g, g.forces, g.forces_mask))
        if has_s:
            stresses.append(g.stress if g.stress is not None
                            else torch.zeros((b, 3, 3), dtype=dtype, device=device))
            s_masks.append(label_mask(g, g.stress, g.stress_mask))
        if has_q:
            charges.append(g.total_charge if g.total_charge is not None
                           else torch.zeros(b, dtype=dtype, device=device))
        if has_w:
            weights.append(g.weight if g.weight is not None
                           else torch.ones(b, dtype=dtype, device=device))

    f_mask = torch.cat(f_masks, 0) if has_f else None
    s_mask = torch.cat(s_masks, 0) if has_s else None
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
        # a mask only where the batch mixes labelled and unlabelled structures
        forces_mask=None if f_mask is None or bool(f_mask.all()) else f_mask,
        stress_mask=None if s_mask is None or bool(s_mask.all()) else s_mask,
        total_charge=torch.cat(charges, 0) if has_q else None,
        weight=torch.cat(weights, 0) if has_w else None,
    )
