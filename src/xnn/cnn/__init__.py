"""Convolutional family: 3D CNNs over voxelized atomic environments (the
conventional CNN3D and the SE(3)-equivariant 3D steerable CNN of Weiler et
al., NeurIPS 2018) and the spherical CNN of Cohen et al. (ICLR 2018) over
spherical signals of the environments."""
from . import featurizers, models

__all__ = ["featurizers", "models"]
