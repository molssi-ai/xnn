"""Voxelized atomic environments: the input fields of the 3D convolution models.

The volumetric models of the ``cnn`` family (:class:`~xnn.cnn.models.cnn3d.CNN3D`
and the 3D steerable :class:`~xnn.cnn.models.steerable.SteerableCNN` of Weiler
*et al.*, NeurIPS 2018) read scalar fields sampled on a cubic grid. This
featurizer builds one such grid per atom: a cube of side ``2 * cutoff``
centered on the atom, with one channel per chemical species holding the
density that the neighbors of that species deposit on the voxels. The
density of a neighbor is a Gaussian of width ``sigma`` placed at its
position (the construction of the paper's amino-acid and CATH experiments,
where "the values of the voxels were set to the densities arising from
placing a Gaussian at each atom position"), multiplied by a smooth envelope
that vanishes at the cutoff so the fields, and with them the energy, stay
continuous when a neighbor enters or leaves the cutoff sphere.

The grids are differentiable with respect to the positions, so the autograd
forces and stress of :class:`~xnn.common.models.outputs.ForceStressOutput`
work unchanged, and periodic images enter only through the edge vectors.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import Tensor

from xnn.common.data import AtomicGraph
from xnn.common.featurizers import CosineCutoff, Featurizer

_MAX_Z = 120

#: voxels per block of edges splatted at once (bounds the ``(edges, s, s, s)``
#: temporaries to about 64 MB in float32)
VOXEL_BLOCK = 1 << 24


class VoxelGrid(Featurizer):
    """Per-atom voxel grids of the species-resolved neighbor density.

    For atom ``i`` the grid covers the cube ``[-cutoff, cutoff]^3`` around its
    position with ``grid_size`` voxels per axis (voxel width ``2 * cutoff /
    grid_size``, voxel centers at ``(k - (grid_size - 1) / 2)`` widths). Channel
    ``c`` holds the density of the neighbors of species ``species[c]``,

    ``rho_c(x) = sum_{j in N(i), Z_j = species[c]} w(|r_ij|) exp(-|x - r_ij|^2 / (2 sigma^2))``,

    with ``r_ij`` the position of ``j`` relative to ``i`` and ``w`` the cutoff
    envelope. The atom itself is deposited at the origin of its own species
    channel when ``include_center`` is set, so the grid also encodes the
    species of the central atom.

    Parameters
    ----------
    species : sequence of int
        Atomic numbers with a density channel, in channel order.
    cutoff : float, optional
        Half the side of the cube (the neighbor-list radius), by default 4.0.
    grid_size : int, optional
        Voxels per axis ``s``, by default 17 (0.47 Angstrom voxels at 4.0).
    sigma : float or None, optional
        Standard deviation of the Gaussian density of one atom. ``None``
        (default) uses the voxel width, which keeps the density sampled
        without aliasing so a rotation of the structure rotates the grids
        to the interpolation error; the paper's CATH data set uses half the
        voxel width (``sigma = cutoff / grid_size``), a sharper but aliased
        choice.
    cutoff_fn : str or None, optional
        ``"cosine"`` (default) multiplies each neighbor's density by the
        :class:`~xnn.common.featurizers.CosineCutoff` envelope so the fields
        are continuous at the cutoff; ``None`` deposits every neighbor with
        unit weight.
    include_center : bool, optional
        Deposit the central atom at the origin of its species channel, by
        default ``True``.

    Attributes
    ----------
    species : list of int
        The channel species.
    axis : Tensor
        The ``grid_size`` voxel-center coordinates along one axis.
    """

    def __init__(self, species: Sequence[int], cutoff: float = 4.0,
                 grid_size: int = 17, sigma: Optional[float] = None,
                 cutoff_fn: Optional[str] = "cosine", include_center: bool = True):
        super().__init__()
        if cutoff_fn not in (None, "cosine"):
            raise ValueError(f"cutoff_fn must be None or 'cosine', got {cutoff_fn!r}")
        if grid_size < 1:
            raise ValueError("grid_size must be positive")
        self.species = [int(z) for z in species]
        if len(set(self.species)) != len(self.species):
            raise ValueError("species must not repeat")
        self.cutoff = float(cutoff)
        self.grid_size = int(grid_size)
        self.spacing = 2.0 * self.cutoff / self.grid_size
        self.sigma = float(sigma) if sigma is not None else self.spacing
        self.include_center = bool(include_center)
        self.envelope = CosineCutoff(self.cutoff) if cutoff_fn == "cosine" else None
        index = torch.full((_MAX_Z,), -1, dtype=torch.long)
        for c, z in enumerate(self.species):
            index[z] = c
        self.register_buffer("species_index", index)
        k = torch.arange(self.grid_size, dtype=torch.get_default_dtype())
        self.register_buffer("axis", (k - (self.grid_size - 1) / 2.0) * self.spacing)

    @property
    def output_dim(self) -> int:
        """int : Number of channels (one per species)."""
        return len(self.species)

    @property
    def n_channels(self) -> int:
        """int : Number of channels (one per species)."""
        return len(self.species)

    def channel_index(self, atomic_numbers: Tensor) -> Tensor:
        """Channel of every atom, raising for species without a channel."""
        idx = self.species_index[atomic_numbers]
        if bool((idx < 0).any()):
            missing = sorted(set(atomic_numbers[idx < 0].tolist()))
            raise ValueError(f"atomic numbers {missing} have no density channel "
                             f"(species={self.species})")
        return idx

    def _profile(self, coord: Tensor) -> Tensor:
        """Gaussian density along one axis, ``(E, grid_size)`` for ``(E,)`` coordinates."""
        axis = self.axis.to(coord.dtype)
        return torch.exp(-((axis[None, :] - coord[:, None]) ** 2) / (2.0 * self.sigma ** 2))

    def voxelize(self, atomic_numbers: Tensor, edge_index: Tensor, edge_vec: Tensor) -> Tensor:
        """Build the grids from tensors (the core reused by :meth:`forward`).

        Parameters
        ----------
        atomic_numbers : Tensor
            Per-atom atomic numbers, shape ``(N,)``.
        edge_index : Tensor
            Edge index ``(2, E)``; row 0 the neighbor, row 1 the central atom.
        edge_vec : Tensor
            ``pos[dst] - pos[src]`` per edge (with the periodic image shift),
            shape ``(E, 3)``; the neighbor sits at ``-edge_vec`` relative to
            the center.

        Returns
        -------
        Tensor
            The grids, shape ``(N, n_channels, s, s, s)`` in the dtype of
            ``edge_vec``.
        """
        n_atoms = atomic_numbers.shape[0]
        n_ch, s = self.n_channels, self.grid_size
        channel = self.channel_index(atomic_numbers)
        dtype, device = edge_vec.dtype, edge_vec.device
        grid = torch.zeros(n_atoms * n_ch, s, s, s, dtype=dtype, device=device)

        if self.include_center and n_atoms > 0:
            zero = torch.zeros(1, dtype=dtype, device=device)
            g = self._profile(zero)[0]                                   # (s,)
            blob = (g[:, None, None] * g[None, :, None] * g[None, None, :])
            own = torch.arange(n_atoms, device=device) * n_ch + channel
            grid = grid.index_add(0, own, blob.expand(n_atoms, s, s, s))

        src, dst = edge_index[0], edge_index[1]
        rel = -edge_vec                                                  # neighbor relative to center
        if self.envelope is not None:
            weight = self.envelope(torch.linalg.norm(rel, dim=-1))
        else:
            weight = torch.ones(rel.shape[0], dtype=dtype, device=device)
        target = dst * n_ch + channel[src]
        block = max(1, VOXEL_BLOCK // (s * s * s))
        for e0 in range(0, rel.shape[0], block):
            r = rel[e0:e0 + block]
            gx, gy, gz = self._profile(r[:, 0]), self._profile(r[:, 1]), self._profile(r[:, 2])
            dens = (weight[e0:e0 + block, None, None, None]
                    * gx[:, :, None, None] * gy[:, None, :, None] * gz[:, None, None, :])
            grid = grid.index_add(0, target[e0:e0 + block], dens)
        return grid.view(n_atoms, n_ch, s, s, s)

    def forward(self, data: AtomicGraph) -> Tensor:
        """Voxelize the environment of every atom of a batch.

        Parameters
        ----------
        data : AtomicGraph
            Batched atomic graph (atomic numbers, edge index, geometry).

        Returns
        -------
        Tensor
            Density grids of shape ``(N, n_channels, s, s, s)``.
        """
        return self.voxelize(data.atomic_numbers, data.edge_index, data.edge_vectors())

    def extra_repr(self) -> str:
        return (f"species={self.species}, cutoff={self.cutoff}, grid_size={self.grid_size}, "
                f"sigma={self.sigma:.4g}, envelope={'cosine' if self.envelope is not None else None}, "
                f"include_center={self.include_center}")
