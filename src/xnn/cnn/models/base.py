"""Shared body of the volumetric (voxel) models of the ``cnn`` family.

:class:`VoxelPotential` composes the :class:`~xnn.cnn.featurizers.VoxelGrid`
featurizer with a convolutional trunk that a subclass supplies (a
conventional 3D CNN in :mod:`~xnn.cnn.models.cnn3d`, the SE(3)-equivariant
3D steerable CNN of Weiler *et al.*, NeurIPS 2018, in
:mod:`~xnn.cnn.models.steerable`) and the per-atom energy readout shared
with SchNet: a two-layer atom-wise network on the pooled features, the
per-atom energy standardization ``E_i = energy_scale * E^hat_i +
energy_shift`` and the per-element reference energy ``atom_ref``.

The module also holds the grid operations both trunks use: the Gaussian
low-pass filter the paper applies before every strided downsampling
(Sec. 4.4.2), the global average pooling of its architectures, the
bookkeeping of field multiplicities, and the exact rotation of a voxel
grid by a symmetry of the cube (what the equivariance tests and notebooks
use).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from xnn.common.data import AtomicGraph
from xnn.common.models.base import InteratomicPotential
from xnn.common.models.ops import make_activation
from ..featurizers import VoxelGrid

_MAX_Z = 120


def field_dim(fields: Sequence[int]) -> int:
    """Number of channels of a stack of fields.

    Parameters
    ----------
    fields : sequence of int
        Multiplicities ``(m_0, m_1, m_2, ...)``: ``m_l`` fields transforming
        under the irreducible representation of order ``l`` (dimension
        ``2l + 1``). A conventional CNN has only ``m_0`` scalar channels.

    Returns
    -------
    int
        ``sum_l m_l (2l + 1)``.
    """
    return int(sum(m * (2 * l + 1) for l, m in enumerate(fields)))


def default_fields(n_features: int, n_blocks: int, l_max: int = 2) -> list[tuple[int, ...]]:
    """The default field multiplicities of the voxel potentials.

    Hidden block ``t`` has ``n_features // 4 * 2**t`` scalar fields and half
    as many fields for each higher order up to ``l_max`` (the doubling per
    block of the paper's architectures, Table 1 and 3); the last block has
    ``n_features`` scalar fields only, since the readout pools scalars.

    Parameters
    ----------
    n_features : int
        Scalar fields of the last block (the readout width).
    n_blocks : int
        Number of convolution blocks.
    l_max : int, optional
        Highest field order of the hidden blocks, by default 2.

    Returns
    -------
    list of tuple of int
        One multiplicity tuple per block.
    """
    if n_blocks < 1:
        raise ValueError("n_blocks must be at least 1")
    out = []
    for t in range(n_blocks - 1):
        m0 = max(1, n_features // 4) * 2 ** t
        out.append(tuple(max(1, m0 >> l) for l in range(l_max + 1)))
    out.append((int(n_features),))
    return out


def default_strides(n_blocks: int) -> list[int]:
    """Stride 2 on every block but the first and the last (the paper's
    Tetris network downsamples between layers 1-2 and 2-3)."""
    return [1 if t in (0, n_blocks - 1) else 2 for t in range(n_blocks)]


def low_pass_filter(x: Tensor, scale: float, stride: int = 1) -> Tensor:
    """Gaussian blur of the spatial axes, optionally followed by a stride.

    The anti-aliasing filter of Sec. 4.4.2: a downsampling by ``scale`` is
    preceded by a Gaussian of standard deviation ``0.5 sqrt(scale^2 - 1)``
    voxels (the width that turns the voxel-sized sampling kernel into one of
    ``scale`` voxels), truncated at 2.5 standard deviations and normalized to
    unit sum. Applied channel by channel, so it commutes with every fiber
    transformation and preserves equivariance.

    Parameters
    ----------
    x : Tensor
        Fields of shape ``(..., X, Y, Z)``.
    scale : float
        Downsampling factor the blur prepares for; ``scale <= 1`` returns
        ``x`` unchanged (then ``stride`` must be 1).
    stride : int, optional
        Stride of the subsampling applied together with the blur, by
        default 1.

    Returns
    -------
    Tensor
        The filtered fields, ``(..., X', Y', Z')``.
    """
    if scale <= 1:
        if stride != 1:
            raise ValueError("a stride needs a low-pass scale larger than one")
        return x
    sigma = 0.5 * (scale ** 2 - 1) ** 0.5
    size = int(1 + 2 * 2.5 * sigma)
    if size % 2 == 0:
        size += 1
    r = torch.arange(size, dtype=x.dtype, device=x.device) - size // 2
    g = torch.exp(-r ** 2 / (2 * sigma ** 2))
    kernel = g[:, None, None] * g[None, :, None] * g[None, None, :]
    kernel = (kernel / kernel.sum()).view(1, 1, size, size, size)
    lead = x.shape[:-3]
    out = F.conv3d(x.reshape(-1, 1, *x.shape[-3:]), kernel, padding=size // 2, stride=stride)
    return out.view(*lead, *out.shape[-3:])


def global_average_pool(x: Tensor) -> Tensor:
    """Average the spatial axes of ``(B, C, X, Y, Z)`` fields into ``(B, C)``."""
    return x.flatten(2).mean(-1)


def grid_coordinates(size: int, dtype: torch.dtype = torch.float64,
                     device: Optional[torch.device] = None) -> Tensor:
    """Integer-centered coordinates of a cubic grid.

    Parameters
    ----------
    size : int
        Voxels per axis.
    dtype, device
        Of the returned tensor.

    Returns
    -------
    Tensor
        Shape ``(size, size, size, 3)``; axis ``a`` of voxel ``(i, j, k)`` is
        ``(i, j, k)[a] - (size - 1) / 2`` (the origin at the center).
    """
    r = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2.0
    return torch.stack(torch.meshgrid(r, r, r, indexing="ij"), dim=-1)


def rotate_voxels(x: Tensor, R: Tensor) -> Tensor:
    """Rotate scalar voxel grids by a symmetry of the cube, exactly.

    Returns the grids of ``f(R^{-1} x)`` for ``R`` one of the 48 signed
    permutation matrices (the rotations and reflections mapping the cubic
    grid onto itself), which needs no interpolation. The fiber (channel)
    transformation of non-scalar fields is the caller's job.

    Parameters
    ----------
    x : Tensor
        Grids of shape ``(..., s, s, s)``.
    R : Tensor
        A ``(3, 3)`` signed permutation matrix.

    Returns
    -------
    Tensor
        The rotated grids, same shape as ``x``.

    Raises
    ------
    ValueError
        If ``R`` is not a signed permutation matrix.
    """
    R = torch.as_tensor(R, dtype=torch.float64)
    if R.shape != (3, 3) or not bool(((R.abs() == 1).sum(0) == 1).all()) \
            or not bool(((R.abs() == 1).sum(1) == 1).all()) \
            or not bool(((R.abs() == 0) | (R.abs() == 1)).all()):
        raise ValueError("R must be a signed permutation matrix (a symmetry of the cube)")
    s = x.shape[-1]
    coords = grid_coordinates(s).reshape(-1, 3)              # target voxel positions
    source = coords @ R                                      # R^{-1} x = R^T x for orthogonal R
    idx = (source + (s - 1) / 2.0).round().long()
    flat = (idx[:, 0] * s + idx[:, 1]) * s + idx[:, 2]
    out = x.reshape(*x.shape[:-3], s * s * s)[..., flat.to(x.device)]
    return out.reshape(x.shape)


class VoxelPotential(InteratomicPotential):
    """Shared body of the voxel models: grid featurizer, trunk, readout.

    A subclass implements :meth:`trunk`, mapping the ``(N, C, s, s, s)``
    density grids to ``(N, n_features)`` invariant per-atom features; this
    class voxelizes the environments, reads the features out into per-atom
    energies (atom-wise ``n_features -> n_features/2 -> 1`` network with a
    zero-initialized last layer, the DTNN standardization and the per-element
    reference energies, as in :class:`~xnn.cnn.models.schnet.SchNet`) and
    sum-pools them per structure.

    Parameters
    ----------
    species : sequence of int
        Atomic numbers with a density channel.
    cutoff : float, optional
        Radius of the voxelized environment (Angstrom), by default 4.0.
    grid_size : int, optional
        Voxels per axis, by default 17.
    n_features : int, optional
        Width of the pooled features (input of the readout), by default 32.
    sigma : float or None, optional
        Width of the atomic Gaussians (``None``: one voxel).
    cutoff_fn : str or None, optional
        Envelope of the neighbor densities (``"cosine"`` or ``None``).
    include_center : bool, optional
        Deposit the central atom in its own channel, by default ``True``.
    readout_activation : str, optional
        Activation of the readout network, by default ``"ssp"`` (the exact
        shifted softplus, smooth).
    energy_shift, energy_scale : float, optional
        Per-atom energy standardization (buffers), by default 0 and 1.
    atomic_energies : array-like or None, optional
        Per-species reference energies loaded into ``atom_ref``.

    Attributes
    ----------
    voxelizer : VoxelGrid
        The grid featurizer.
    readout : torch.nn.Sequential
        The atom-wise readout.
    atom_ref : torch.nn.Embedding
        Learnable per-element energy reference, initialized to zero.
    """

    head_modules = ("readout", "energy_scale", "energy_shift", "atom_ref")

    def __init__(self, species: Sequence[int] = (1, 6, 8), cutoff: float = 4.0,
                 grid_size: int = 17, n_features: int = 32,
                 sigma: Optional[float] = None, cutoff_fn: Optional[str] = "cosine",
                 include_center: bool = True, readout_activation: str = "ssp",
                 energy_shift: float = 0.0, energy_scale: float = 1.0,
                 atomic_energies=None):
        super().__init__()
        self.species = [int(z) for z in species]
        self.cutoff = float(cutoff)
        self.node_feature_dim = int(n_features)
        self.voxelizer = VoxelGrid(self.species, cutoff, grid_size, sigma, cutoff_fn,
                                   include_center)
        self.readout = nn.Sequential(
            nn.Linear(n_features, max(1, n_features // 2)), make_activation(readout_activation),
            nn.Linear(max(1, n_features // 2), 1),
        )
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)
        self.register_buffer("energy_scale", torch.tensor(float(energy_scale)))
        self.register_buffer("energy_shift", torch.tensor(float(energy_shift)))
        self.atom_ref = nn.Embedding(_MAX_Z, 1)
        nn.init.zeros_(self.atom_ref.weight)
        if atomic_energies is not None:
            self.set_atomic_energies(atomic_energies)

    @property
    def grid_size(self) -> int:
        """int : Voxels per axis of the environment grids."""
        return self.voxelizer.grid_size

    @property
    def n_channels(self) -> int:
        """int : Input channels of the trunk (one per species)."""
        return self.voxelizer.n_channels

    def set_energy_scale_shift(self, scale: float, shift: float) -> None:
        """Set the per-atom energy standardization from training statistics.

        Parameters
        ----------
        scale : float
            Standard deviation of the training-set energy per atom.
        shift : float
            Mean training-set energy per atom.
        """
        with torch.no_grad():
            self.energy_scale.fill_(float(scale))
            self.energy_shift.fill_(float(shift))

    def set_atomic_energies(self, values) -> None:
        """Initialize the per-element reference energies ``atom_ref``.

        Parameters
        ----------
        values : array-like
            One reference energy per entry of ``species``, in order.

        Raises
        ------
        ValueError
            If the number of values does not match the number of species.
        """
        ae = torch.as_tensor(values, dtype=self.atom_ref.weight.dtype).flatten()
        if ae.numel() != len(self.species):
            raise ValueError(f"got {ae.numel()} atomic energies for {len(self.species)} species")
        with torch.no_grad():
            self.atom_ref.weight[torch.tensor(self.species), 0] = ae

    def trunk(self, grid: Tensor) -> Tensor:
        """Map density grids to invariant per-atom features.

        Parameters
        ----------
        grid : Tensor
            Shape ``(N, n_channels, s, s, s)``.

        Returns
        -------
        Tensor
            Shape ``(N, n_features)``.
        """
        raise NotImplementedError

    def node_features_energy(self, grid: Tensor, atomic_numbers: Tensor) -> tuple[Tensor, Tensor]:
        """Features and per-atom energies of already voxelized environments."""
        features = self.trunk(grid)
        node_energy = (self.energy_scale * self.readout(features).squeeze(-1)
                       + self.energy_shift + self.atom_ref(atomic_numbers).squeeze(-1))
        return features, node_energy

    def forward(self, data: AtomicGraph) -> Dict[str, Tensor]:
        """Compute per-atom and total energies for a batch of structures.

        Parameters
        ----------
        data : AtomicGraph
            Batched atomic graph.

        Returns
        -------
        dict[str, Tensor]
            ``"node_energy"`` ``(N,)``, ``"energy"`` ``(B,)`` and the pooled
            invariant ``"node_features"`` ``(N, n_features)``.
        """
        grid = self.voxelizer(data)
        features, node_energy = self.node_features_energy(grid, data.atomic_numbers)
        return {"node_energy": node_energy, "energy": self.aggregate_energy(node_energy, data),
                "node_features": features}

    @staticmethod
    def _common_config_options(cfg) -> dict:
        """The constructor options every voxel model reads from a config."""
        from xnn.common.config.coerce import coerce_per_species, coerce_species

        extra = dict(cfg.extra or {})
        species = coerce_species(extra.get("species"), default=[1, 6, 8])
        sigma = extra.get("sigma")
        return dict(
            species=species,
            cutoff=cfg.cutoff,
            grid_size=int(extra.get("grid_size", 17)),
            sigma=None if sigma is None else float(sigma),
            cutoff_fn=extra.get("cutoff_fn", "cosine"),
            include_center=bool(extra.get("include_center", True)),
            readout_activation=extra.get("readout_activation", "ssp"),
            energy_shift=float(extra.get("energy_shift", 0.0)),
            energy_scale=float(extra.get("energy_scale", 1.0)),
            atomic_energies=coerce_per_species(extra.get("atomic_energies"), species,
                                               "atomic_energies"),
        )
