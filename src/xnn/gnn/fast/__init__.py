"""Fast paths of the equivariant GNN blocks (fused cuEquivariance kernels).

:class:`ConvTensorProduct` (the message step of MACE and NequIP) and the
symmetric contraction of MACE have a fused GPU implementation next to their
reference one; :class:`~xnn.common.models.fast.FastPathModule` explains the
contract and ``use_fast`` the choice. :data:`AUTO_POLICY` holds the measured
size above which ``use_fast="auto"`` takes the kernels. The fused continuous-filter
convolution of SchNet and the fused message block of PaiNN (Triton kernels) live in
:mod:`xnn.gnn.fast.schnet` and :mod:`xnn.gnn.fast.painn`. The equivariant linear maps and
the element self-connection of MACE have plain-PyTorch fast paths in :mod:`xnn.gnn.fast.linear`.
"""
from xnn.common.models.fast import AutoPolicy

# the equivariant blocks need e3nn; without it the fast paths of the plain-PyTorch models
# (SchNet, PaiNN, TensorNet, MEGNet) still import from this package
try:
    from ._cueq import available
    from .conv import ConvTensorProduct
    from .linear import EquivariantLinear, SpeciesTensorProduct
    HAS_EQUIVARIANT = True
except ImportError:
    HAS_EQUIVARIANT = False

#: Edge counts above which ``use_fast="auto"`` uses the kernels. The kernels
#: carry a fixed per-step cost and the reference a cost per edge, so the
#: crossover moves with the model's width: early for foundation-size models,
#: late for light ones. The thresholds sit near the wide models' crossover,
#: since the reference's loss past it grows with the system while the kernels'
#: loss below a light model's crossover stays a fixed fraction of a step. In
#: float64 the reference is slower and the kernels win earlier; on an L40S
#: (weak float64) always.
AUTO_POLICY = AutoPolicy(
    min_edges={"A100": 15_000, "B200": 40_000, "H200": 60_000, "H100": 60_000, "L40S": 30_000,
               "T4": 6_000, "V100": 30_000, "A30": 20_000},
    min_edges_float64={"A100": 10_000, "B200": 30_000, "H200": 30_000, "H100": 30_000, "L40S": 0,
                       "T4": 2_000, "V100": 12_000, "A30": 8_000},
    default_min_edges=50_000,
    default_min_edges_float64=25_000,
)

#: The same for MACE in training mode (a step with a force loss): the kernels
#: pay off earlier than in evaluation, as the backward of a training step
#: repeats the work the fixed cost is spread over.
AUTO_POLICY_TRAINING = AutoPolicy(
    min_edges={"A100": 10_000, "B200": 30_000, "T4": 5_000, "V100": 20_000, "A30": 8_000},
    min_edges_float64={"A100": 10_000, "B200": 25_000, "T4": 1_000, "V100": 6_000, "A30": 6_000},
    default_min_edges=50_000,
    default_min_edges_float64=25_000,
)

#: NequIP's convolution in evaluation and training: on an A100, A30, V100 and T4 the
#: kernels were faster than the reference from a single small molecule on, in
#: both precisions and both modes.
NEQUIP_AUTO_POLICY = AutoPolicy(
    min_edges={"A100": 0, "A30": 0, "V100": 0, "T4": 0},
    min_edges_float64={"A100": 0, "A30": 0, "V100": 0, "T4": 0},
    default_min_edges=50_000,
    default_min_edges_float64=25_000,
)
NEQUIP_AUTO_POLICY_TRAINING = NEQUIP_AUTO_POLICY

__all__ = ["AUTO_POLICY", "AUTO_POLICY_TRAINING", "NEQUIP_AUTO_POLICY", "NEQUIP_AUTO_POLICY_TRAINING",
           "ConvTensorProduct", "EquivariantLinear", "SpeciesTensorProduct", "available"]
