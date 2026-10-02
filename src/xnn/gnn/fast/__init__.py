"""Fast paths of the equivariant GNN blocks (fused cuEquivariance kernels).

:class:`ConvTensorProduct` (the message step of MACE and NequIP) and the
symmetric contraction of MACE have a fused GPU implementation next to their
reference one; :class:`~xnn.common.models.fast.FastPathModule` explains the
contract and ``use_fast`` the choice. :data:`AUTO_POLICY` holds the measured
size above which ``use_fast="auto"`` takes the kernels.
"""
from xnn.common.models.fast import AutoPolicy

from ._cueq import available
from .conv import ConvTensorProduct

#: Edge counts above which ``use_fast="auto"`` uses the kernels. From the MD-step
#: crossover of the kernels' fixed per-step cost (about 26-38 ms on an A100,
#: 14-21 ms on an H200 and L40S) with the reference's cost per edge, measured
#: for the production water model, MACE-OFF23 small/medium and MACE-MP-0
#: medium: the lighter the model, the later the crossover (float32 A100:
#: 19k edges for MP-0 medium, 90k for the light water models), and the
#: thresholds sit between the two. In float64 the reference is slower and the
#: kernels win earlier; on an L40S (weak float64) always.
AUTO_POLICY = AutoPolicy(
    min_edges={"A100": 50_000, "H200": 60_000, "H100": 60_000, "L40S": 30_000},
    min_edges_float64={"A100": 25_000, "H200": 30_000, "H100": 30_000, "L40S": 0},
    default_min_edges=50_000,
    default_min_edges_float64=25_000,
)

__all__ = ["AUTO_POLICY", "ConvTensorProduct", "available"]
