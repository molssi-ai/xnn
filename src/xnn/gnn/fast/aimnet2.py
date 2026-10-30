"""Fast path of the AIMNet2 shell convolution (Triton kernels).

The reference gathers the convolved per-atom quantity onto every edge, forms
its products with the radial basis and with the radial basis times each
Cartesian component of the unit bond vector, and sums each product onto the
centers: per pass a ``(E, C, S)`` gather and four products of that size, all
kept by autograd. Here one kernel walks the edges of every center and
accumulates the four shell sums ``sum_j a_jcs g_ijsk`` directly from the
per-atom quantity, so nothing edge-sized beyond the basis is formed or saved.
The sums are bilinear in the quantity and the basis: their gradients are one
kernel over the edges of every neighbor, and the derivatives of that kernel
are the two kernels again, so forces, stress and the weight gradients of a
force loss take the fused path at any derivative order.
"""
from __future__ import annotations

import torch

from xnn.common.models.fast import AutoPolicy

#: Edge counts above which ``use_fast="auto"`` takes the kernels in evaluation
#: mode (energy and forces). On an A100, A30, L40S, V100 and T4 the
#: kernels were faster than the reference from a single small molecule on, in
#: both precisions; a GPU in neither table takes the largest measured thresholds.
AUTO_POLICY = AutoPolicy(
    min_edges={"A100": 0, "A30": 0, "L40S": 0, "V100": 0, "T4": 0},
    min_edges_float64={"A100": 0, "A30": 0, "L40S": 0, "V100": 0, "T4": 0},
    default_min_edges=0,
    default_min_edges_float64=0,
)

#: The same for training mode (a step with a force loss, second derivatives included).
AUTO_POLICY_TRAINING = AUTO_POLICY


def supported(device: torch.device, dtype: torch.dtype) -> bool:
    """Triton kernels on a CUDA device in float32 or float64."""
    from xnn.common.models.dispersion_fast._triton import supported as triton_supported
    return triton_supported(device, dtype)
