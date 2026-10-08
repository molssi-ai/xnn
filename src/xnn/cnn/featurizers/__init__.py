"""Featurizers of the cnn family: voxelized environments and spherical signals."""
from .voxel import VoxelGrid
from .spherical import SphericalGrid

__all__ = ["VoxelGrid", "SphericalGrid"]
