"""Fast paths of the voxel models (SteerableCNN, CNN3D).

The 3D convolutions of their blocks have a second implementation,
:func:`~xnn.cnn.fast.conv.conv3d`, which runs cuDNN's channels-last kernels with its algorithm search;
:class:`~xnn.common.models.fast.FastPathModule` explains the contract and
``use_fast`` the choice. :data:`AUTO_POLICY` and
:data:`AUTO_POLICY_TRAINING` hold where ``use_fast="auto"`` takes it.

The correlations of the SphericalCNN have Triton kernels instead, in
:mod:`xnn.cnn.fast.spherical` with their own policies.
"""
from xnn.common.models.fast import AutoPolicy

from .conv import conv3d, supported

#: Edge counts above which ``use_fast="auto"`` takes the fast convolutions in
#: evaluation mode (energy and forces). Above it the channels-last kernels are
#: faster and, run in batch chunks, need less memory than the reference; on a
#: few molecules their workspace per grid exceeds the reference's.
AUTO_POLICY = AutoPolicy(default_min_edges=1000, default_min_edges_float64=1000)

#: The same for training mode (a step with a force loss, second derivatives
#: included). cuDNN's algorithm search runs once per input shape, so a training
#: run pays that once for each batch size it meets.
AUTO_POLICY_TRAINING = AutoPolicy(default_min_edges=1000, default_min_edges_float64=1000)

__all__ = ["AUTO_POLICY", "AUTO_POLICY_TRAINING", "conv3d", "supported"]
