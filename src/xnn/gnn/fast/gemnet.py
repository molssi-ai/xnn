"""Fast path of the GemNet efficient bilinear layer (Triton kernels).

The reference gathers the down-projected features onto every triplet (or
quadruplet), scatters them into a zero-padded ``(E, K_max, F)`` neighbor
matrix and multiplies it with the padded angular basis. Here one kernel walks
the items of every output edge, gathers the features of their input rows and
accumulates ``sum_t s_t (x) x[in_t]`` in registers: neither the per-item
features nor the padded matrices are formed, and autograd keeps the edge-sized
features and the per-item basis only. The radial contraction and the bilinear
weights stay matrix products. The sums are bilinear in the features and the
basis: their gradients are one kernel over the items of every input row, and
the derivatives of that kernel are the two kernels again, so forces and the
weight gradients of a force loss take the fused path at any derivative order.
"""
from __future__ import annotations

import torch

from xnn.common.models.fast import AutoPolicy

#: A threshold no graph reaches: ``"auto"`` keeps the reference.
NEVER = 1 << 62

#: Edge counts above which ``use_fast="auto"`` takes the triplet kernels in
#: evaluation mode (energy and forces). In float32 they were faster than the
#: reference from about 10k edges on (A100); in float64, where Triton has no
#: tile product and the reference's padded batched products run on the GPU's
#: float64 matrix units, only on the largest graphs. A GPU in neither table
#: takes the largest measured thresholds.
AUTO_POLICY = AutoPolicy(
    min_edges={"A100": 10_000},
    min_edges_float64={"A100": 60_000},
    default_min_edges=10_000,
    default_min_edges_float64=60_000,
)

#: The same for training mode (a step with a force loss, second derivatives
#: included): in float32 the kernels were faster at every size measured, in
#: float64 at none.
AUTO_POLICY_TRAINING = AutoPolicy(
    min_edges={"A100": 0},
    min_edges_float64={"A100": NEVER},
    default_min_edges=0,
    default_min_edges_float64=NEVER,
)

#: The quadruplet kernels in evaluation and training: in float32 faster than the
#: reference from a few thousand edges (the long rows of quadruplets split over
#: many programs), in float64 at no size measured.
QUADRUPLET_AUTO_POLICY = AutoPolicy(
    min_edges={"A100": 5_000},
    min_edges_float64={"A100": NEVER},
    default_min_edges=5_000,
    default_min_edges_float64=NEVER,
)
QUADRUPLET_AUTO_POLICY_TRAINING = QUADRUPLET_AUTO_POLICY


def supported(device: torch.device, dtype: torch.dtype) -> bool:
    """Triton kernels on a CUDA device in float32 or float64."""
    from xnn.common.models.dispersion_fast._triton import supported as triton_supported
    return triton_supported(device, dtype)
