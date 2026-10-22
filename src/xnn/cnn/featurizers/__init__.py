"""Featurizers of the cnn family: voxelized environments and spherical signals."""
from .spherical import SphericalGrid
from .voxel import VoxelGrid

__all__ = ["VoxelGrid", "SphericalGrid"]
