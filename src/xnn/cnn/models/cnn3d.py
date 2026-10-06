"""Conventional 3D CNN over voxelized atomic environments.

The baseline the 3D steerable CNN of Weiler *et al.* (NeurIPS 2018) is
compared with: the same architecture "simply without the constraint of
being equivariant for rotation" (supplement Sec. 3). Each block is a
``Conv3d`` with an unconstrained kernel, the Gaussian low-pass filter before
any strided downsampling (Sec. 4.4.2, the ``smooth_stride`` of the paper's
experiments), optional batch normalization and an element-wise activation;
the last feature map is pooled by a global average into the per-atom
features of the :class:`~xnn.cnn.models.base.VoxelPotential` readout.

Because its kernels are unconstrained, the energy this model predicts is
*not* invariant under rotations of the structure (only under its
translations and the permutations of its atoms); it is the control that
shows what the steerable kernels buy.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor, nn

from xnn.common.models.ops import make_activation
from xnn.common.models.registry import register_model
from .base import (VoxelPotential, default_fields, default_strides, field_dim,
                   global_average_pool, low_pass_filter)


class Conv3dBlock(nn.Module):
    """One block of the conventional CNN: convolution, low-pass stride,
    normalization, activation.

    Parameters
    ----------
    in_channels, out_channels : int
        Channels in and out.
    kernel_size : int, optional
        Side of the cubic kernel, by default 5.
    stride : int, optional
        Downsampling factor, by default 1.
    padding : int or None, optional
        Zero padding; ``None`` keeps the grid size (``kernel_size // 2``).
    activation : str or None, optional
        Element-wise activation (:func:`~xnn.common.models.ops.make_activation`),
        by default ``"relu"``; ``None`` for a linear block.
    normalization : str or None, optional
        ``"batch"`` inserts a ``BatchNorm3d`` after the convolution.
    smooth_stride : bool, optional
        Low-pass filter before subsampling (default) instead of a strided
        convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5,
                 stride: int = 1, padding: Optional[int] = None,
                 activation: Optional[str] = "relu", normalization: Optional[str] = None,
                 smooth_stride: bool = True):
        super().__init__()
        if normalization not in (None, "batch"):
            raise ValueError(f"normalization must be None or 'batch', got {normalization!r}")
        padding = kernel_size // 2 if padding is None else int(padding)
        self.smooth_stride = bool(smooth_stride) and stride > 1
        self.stride = int(stride)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding,
                              stride=1 if self.smooth_stride else self.stride)
        self.norm = nn.BatchNorm3d(out_channels) if normalization == "batch" else None
        self.act = make_activation(activation) if activation is not None else None

    def forward(self, x: Tensor) -> Tensor:
        """Apply the block to ``(B, C_in, X, Y, Z)`` fields."""
        x = self.conv(x)
        if self.smooth_stride:
            x = low_pass_filter(x, self.stride, self.stride)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


@register_model("cnn3d")
class CNN3D(VoxelPotential):
    """Conventional 3D convolutional potential over voxelized environments.

    ``len(channels)`` :class:`Conv3dBlock` blocks map the species density
    grids of :class:`~xnn.cnn.featurizers.VoxelGrid` to ``channels[-1]``
    feature maps, which a global average pool turns into the per-atom
    features of the shared readout. The defaults mirror the steerable model
    (:func:`~xnn.cnn.models.base.default_fields`): the channel count of
    each block equals the number of components of the steerable fields of
    the same block, the paper's recipe for its CNN baselines.

    Parameters
    ----------
    species : sequence of int, optional
        Atomic numbers with a density channel, by default ``(1, 6, 8)``.
    cutoff : float, optional
        Radius of the voxelized environment (Angstrom), by default 4.0.
    grid_size : int, optional
        Voxels per axis, by default 17.
    channels : sequence of int or None, optional
        Output channels of each block; ``None`` derives them from
        ``n_features`` and ``n_blocks``.
    n_features : int, optional
        Channels of the last block when ``channels`` is ``None`` (and the
        readout width), by default 32.
    n_blocks : int, optional
        Number of blocks when ``channels`` is ``None``, by default 3.
    kernel_size : int, optional
        Side of the cubic kernels, by default 5.
    strides : sequence of int or None, optional
        Downsampling factor per block; ``None`` downsamples by 2 in every
        block but the first and the last.
    padding : int or None, optional
        Zero padding of every convolution; ``None`` keeps the grid size.
    activation : str, optional
        Activation of the blocks, by default ``"ssp"`` (smooth, for a smooth
        energy surface; the paper's networks use ``"relu"``).
    normalization : str or None, optional
        ``"batch"`` for batch normalization in every block, by default
        ``None``.
    smooth_stride : bool, optional
        Low-pass filter before every downsampling (default), see
        :func:`~xnn.cnn.models.base.low_pass_filter`.
    sigma, cutoff_fn, include_center, readout_activation, energy_shift, energy_scale, atomic_energies
        See :class:`~xnn.cnn.models.base.VoxelPotential`.

    Attributes
    ----------
    blocks : torch.nn.Sequential
        The convolution blocks.
    """

    def __init__(self, species: Sequence[int] = (1, 6, 8), cutoff: float = 4.0,
                 grid_size: int = 17, channels: Optional[Sequence[int]] = None,
                 n_features: int = 32, n_blocks: int = 3, kernel_size: int = 5,
                 strides: Optional[Sequence[int]] = None, padding: Optional[int] = None,
                 activation: str = "ssp", normalization: Optional[str] = None,
                 smooth_stride: bool = True, sigma: Optional[float] = None,
                 cutoff_fn: Optional[str] = "cosine", include_center: bool = True,
                 readout_activation: str = "ssp", energy_shift: float = 0.0,
                 energy_scale: float = 1.0, atomic_energies=None):
        if channels is None:
            channels = [field_dim(f) for f in default_fields(n_features, n_blocks)]
        channels = [int(c) for c in channels]
        if strides is None:
            strides = default_strides(len(channels))
        if len(strides) != len(channels):
            raise ValueError("strides must have one entry per block")
        super().__init__(species, cutoff, grid_size, channels[-1], sigma, cutoff_fn,
                         include_center, readout_activation, energy_shift, energy_scale,
                         atomic_energies)
        self.channels = channels
        blocks, c_in = [], self.n_channels
        for c_out, stride in zip(channels, strides):
            blocks.append(Conv3dBlock(c_in, c_out, kernel_size, stride, padding, activation,
                                      normalization, smooth_stride))
            c_in = c_out
        self.blocks = nn.Sequential(*blocks)

    def trunk(self, grid: Tensor) -> Tensor:
        """Convolution blocks followed by global average pooling."""
        return global_average_pool(self.blocks(grid))

    @classmethod
    def from_config(cls, cfg) -> "CNN3D":
        """Build a :class:`CNN3D` from a configuration object.

        ``cfg.cutoff``, ``cfg.n_features`` (channels of the last block) and
        ``cfg.n_interactions`` (number of blocks) are the core fields;
        ``cfg.extra`` may set ``species``, ``grid_size``, ``sigma``,
        ``cutoff_fn``, ``include_center``, ``channels``, ``kernel_size``,
        ``strides``, ``padding``, ``activation``, ``normalization``,
        ``smooth_stride``, ``readout_activation``, ``energy_shift``,
        ``energy_scale`` and ``atomic_energies``.

        Parameters
        ----------
        cfg : xnn.common.config.schema.ModelConfig
            The model config.

        Returns
        -------
        CNN3D
            The model.
        """
        extra = dict(cfg.extra or {})
        opts = cls._common_config_options(cfg)
        channels = extra.get("channels")
        strides = extra.get("strides")
        padding = extra.get("padding")
        return cls(
            channels=None if channels is None else [int(c) for c in channels],
            n_features=cfg.n_features,
            n_blocks=cfg.n_interactions,
            kernel_size=int(extra.get("kernel_size", 5)),
            strides=None if strides is None else [int(s) for s in strides],
            padding=None if padding is None else int(padding),
            activation=extra.get("activation", "ssp"),
            normalization=extra.get("normalization"),
            smooth_stride=bool(extra.get("smooth_stride", True)),
            **opts,
        )
