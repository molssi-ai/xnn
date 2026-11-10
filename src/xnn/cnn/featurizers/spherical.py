"""Spherical signals of atomic environments: the input of the spherical CNN.

The molecular representation of Cohen *et al.* (ICLR 2018, Sec. 5.4): a
sphere of radius ``r`` is placed around every atom ``i`` and, for every
species ``z``, the potential of the atoms of that species is sampled on it,

``U_z(x) = sum_{j != i, Z_j = z} Z_i Z_j / |x - p_j|^exponent``,

on the Driscoll-Healy grid of bandwidth ``b`` (``2b x 2b`` points, axes
``beta, alpha``). The signals are invariant to translations, rotate with the
molecule, and are differentiable in the positions, so autograd forces work
unchanged. The paper's potential has ``exponent = 1``; its data script
uses the inverse square distance (``exponent = 2``).
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor

from xnn.common.data import AtomicGraph
from xnn.common.featurizers import CosineCutoff, Featurizer

_MAX_Z = 120

#: grid points per block of edges (bounds the ``(edges, points)`` temporaries)
POINT_BLOCK = 1 << 22


def _grid_directions(b: int) -> Tensor:
    """Unit vectors of the ``S^2`` grid, ``(2b, 2b, 3)`` indexed ``[beta, alpha]``."""
    import math
    beta = (torch.arange(2 * b, dtype=torch.float64) + 0.5) * math.pi / (2 * b)
    alpha = torch.arange(2 * b, dtype=torch.float64) * (2 * math.pi / (2 * b))
    bb, aa = torch.meshgrid(beta, alpha, indexing="ij")
    return torch.stack([torch.sin(bb) * torch.cos(aa), torch.sin(bb) * torch.sin(aa), torch.cos(bb)], dim=-1)


class SphericalGrid(Featurizer):
    """Per-atom spherical signals of the species-resolved neighbor potential.

    Parameters
    ----------
    species : sequence of int
        Atomic numbers with a potential channel, in channel order.
    cutoff : float, optional
        Neighbor-list radius of the sums (Angstrom), by default 10.0.
    radius : float, optional
        Radius of the sphere around each atom, by default 0.48 (half the
        shortest interatomic distance of QM7).
    bandwidth : int, optional
        Bandwidth ``b`` of the grid (``2b`` samples per axis), by default 10.
    exponent : float, optional
        Power of the inverse distance, by default 1.0.
    cutoff_fn : str or None, optional
        ``"cosine"`` multiplies each neighbor's potential by the
        :class:`~xnn.common.featurizers.CosineCutoff` envelope at ``cutoff``;
        ``None`` (default) keeps the untruncated potential of the paper.

    Attributes
    ----------
    species : list of int
        The channel species.
    directions : Tensor
        Unit vectors of the grid points, ``(2b, 2b, 3)``.
    """

    def __init__(self, species: Sequence[int], cutoff: float = 10.0, radius: float = 0.48,
                 bandwidth: int = 10, exponent: float = 1.0, cutoff_fn: Optional[str] = None):
        super().__init__()
        if cutoff_fn not in (None, "cosine"):
            raise ValueError(f"cutoff_fn must be None or 'cosine', got {cutoff_fn!r}")
        if bandwidth < 1 or radius <= 0:
            raise ValueError("bandwidth must be positive and radius larger than zero")
        self.species = [int(z) for z in species]
        if len(set(self.species)) != len(self.species):
            raise ValueError("species must not repeat")
        self.cutoff, self.radius = float(cutoff), float(radius)
        self.bandwidth, self.exponent = int(bandwidth), float(exponent)
        self.envelope = CosineCutoff(self.cutoff) if cutoff_fn == "cosine" else None
        index = torch.full((_MAX_Z,), -1, dtype=torch.long)
        for c, z in enumerate(self.species):
            index[z] = c
        self.register_buffer("species_index", index)
        self.register_buffer("directions", _grid_directions(self.bandwidth).to(torch.get_default_dtype()))

    @property
    def output_dim(self) -> int:
        """int : Number of channels (one per species)."""
        return len(self.species)

    @property
    def n_channels(self) -> int:
        """int : Number of channels (one per species)."""
        return len(self.species)

    @property
    def grid_size(self) -> int:
        """int : Samples per axis of the grid, ``2b``."""
        return 2 * self.bandwidth

    def channel_index(self, atomic_numbers: Tensor) -> Tensor:
        """Channel of every atom, raising for species without a channel."""
        idx = self.species_index[atomic_numbers]
        if bool((idx < 0).any()):
            missing = sorted(set(atomic_numbers[idx < 0].tolist()))
            raise ValueError(f"atomic numbers {missing} have no potential channel (species={self.species})")
        return idx

    def potential(self, atomic_numbers: Tensor, edge_index: Tensor, edge_vec: Tensor) -> Tensor:
        """Build the signals from tensors (the core reused by :meth:`forward`).

        Parameters
        ----------
        atomic_numbers : Tensor
            Per-atom atomic numbers, shape ``(N,)``.
        edge_index : Tensor
            Edge index ``(2, E)``; row 0 the neighbor, row 1 the central atom.
        edge_vec : Tensor
            ``pos[dst] - pos[src]`` per edge, shape ``(E, 3)``; the neighbor sits
            at ``-edge_vec`` relative to the center.

        Returns
        -------
        Tensor
            Shape ``(N, n_channels, 2b, 2b)`` in the dtype of ``edge_vec``.
        """
        n_atoms, n_ch, s = atomic_numbers.shape[0], self.n_channels, self.grid_size
        channel = self.channel_index(atomic_numbers)
        dtype, device = edge_vec.dtype, edge_vec.device
        points = (self.radius * self.directions.to(dtype)).reshape(-1, 3)            # (P, 3)
        grid = torch.zeros(n_atoms * n_ch, points.shape[0], dtype=dtype, device=device)
        src, dst = edge_index[0], edge_index[1]
        rel = -edge_vec
        charge = (atomic_numbers[dst] * atomic_numbers[src]).to(dtype)
        if self.envelope is not None:
            charge = charge * self.envelope(torch.linalg.norm(rel, dim=-1))
        # every signal sums its neighbors in one fixed order, so that the grid is the same in every run (the
        # atomic additions of a scatter on a GPU would not be, and a ReLU downstream turns those round-off
        # differences into different forces): the edges sorted by signal, padded to the longest signal
        # (a padding slot is a neighbor of charge zero) and summed along the padding
        n_pts, n_edges = points.shape[0], rel.shape[0]
        target_sorted, order = torch.sort(dst * n_ch + channel[src], stable=True)
        rows, counts = torch.unique_consecutive(target_sorted, return_counts=True)
        n_rows = rows.shape[0]
        width = int(counts.max()) if n_rows > 0 else 0
        rank = (torch.arange(n_edges, device=device)
                - torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts))
        slots = torch.full((n_rows * width,), n_edges, dtype=torch.long, device=device)
        slots[torch.repeat_interleave(torch.arange(n_rows, device=device), counts) * width + rank] = order
        rel_ext = torch.cat([rel, rel.new_zeros(1, 3)])
        charge_ext = torch.cat([charge, charge.new_zeros(1)])
        block = max(1, POINT_BLOCK // max(1, width * n_pts))
        for r0 in range(0, n_rows, block):
            idx = slots[r0 * width:(r0 + block) * width]
            r = rel_ext.index_select(0, idx)
            dist = torch.linalg.norm(points[None, :, :] - r[:, None, :], dim=-1)   # (rows x width, P)
            val = charge_ext.index_select(0, idx)[:, None] / dist ** self.exponent
            grid = grid.index_copy(0, rows[r0:r0 + block], val.view(-1, width, n_pts).sum(1))
        return grid.view(n_atoms, n_ch, s, s)

    def forward(self, data: AtomicGraph) -> Tensor:
        """Signals of every atom of a batch, ``(N, n_channels, 2b, 2b)``."""
        return self.potential(data.atomic_numbers, data.edge_index, data.edge_vectors())

    def extra_repr(self) -> str:
        return (f"species={self.species}, cutoff={self.cutoff}, radius={self.radius}, "
                f"bandwidth={self.bandwidth}, exponent={self.exponent}, "
                f"envelope={'cosine' if self.envelope is not None else None}")
