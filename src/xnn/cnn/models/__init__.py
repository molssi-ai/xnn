from .base import (VoxelPotential, default_fields, default_strides, field_dim,
                   global_average_pool, grid_coordinates, low_pass_filter, rotate_voxels)
from .cnn3d import CNN3D, Conv3dBlock

__all__ = ["VoxelPotential", "CNN3D", "Conv3dBlock", "default_fields",
           "default_strides", "field_dim", "global_average_pool", "grid_coordinates",
           "low_pass_filter", "rotate_voxels"]

# the 3D steerable CNN needs e3nn (spherical harmonics, Clebsch-Gordan
# coefficients) at construction time; the module itself imports without it
from .steerable import (SteerableCNN, SteerableConv3d, SteerableBatchNorm, GatedBlock,
                        steerable_kernel_basis, angular_kernel_basis, n_basis_kernels,
                        rotate_fields, shell_bandlimits, BANDLIMITS)

__all__ += ["SteerableCNN", "SteerableConv3d", "SteerableBatchNorm", "GatedBlock",
            "steerable_kernel_basis", "angular_kernel_basis", "n_basis_kernels",
            "rotate_fields", "shell_bandlimits", "BANDLIMITS"]
