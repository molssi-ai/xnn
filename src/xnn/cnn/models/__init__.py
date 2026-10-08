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

# the spherical CNN (S^2 / SO(3) correlations by generalized FFTs; no e3nn)
from .spherical import (SphericalCNN, SphericalResBlock, S2Convolution, SO3Convolution,
                        S2Transform, SO3Transform, so3_integrate, so3_rotate, s2_rotate,
                        wigner_d, wigner_D, quadrature_weights, s2_near_identity_grid,
                        s2_equatorial_grid, so3_near_identity_grid, so3_equatorial_grid,
                        s2_grid_points, euler_to_matrix, matrix_to_euler)

__all__ += ["SphericalCNN", "SphericalResBlock", "S2Convolution", "SO3Convolution",
            "S2Transform", "SO3Transform", "so3_integrate", "so3_rotate", "s2_rotate",
            "wigner_d", "wigner_D", "quadrature_weights", "s2_near_identity_grid",
            "s2_equatorial_grid", "so3_near_identity_grid", "so3_equatorial_grid",
            "s2_grid_points", "euler_to_matrix", "matrix_to_euler"]
