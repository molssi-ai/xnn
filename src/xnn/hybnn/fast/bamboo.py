"""Fast path of the BAMBOO graph-equivariant-transformer messages (Triton kernels).

A GET layer scores every edge with ``sum_(f in h) q_i k_j`` per head and sums onto the center
``v_j o d_ij o a_ij`` (the scalar message) and ``v_j o e_ij`` (the vector message, three
components). The scores are the fused attention logits of ET (:mod:`xnn.trfnn.fast.et`), and
both messages together are SchNet's continuous-filter convolution over four feature blocks (the
values repeated, the filter ``[d o a | e]``; :mod:`xnn.gnn.fast.schnet`): one program per atom, no
edge-sized value tensor. Both kernels are twice differentiable, so force training takes them as
well.
"""
from __future__ import annotations

import torch

from xnn.common.models.fast import AutoPolicy

#: Edge counts above which ``use_fast="auto"`` takes the kernels in evaluation mode (energy and
#: forces). Below them the kernels' fixed cost loses (1.1-1.2 of the reference); above them they take
#: 0.80-0.96 of its time (to 136k edges on an A100). An unmeasured GPU takes 60k.
AUTO_POLICY = AutoPolicy(
    min_edges={"A100": 60_000, "A30": 20_000, "T4": 20_000, "V100": 40_000, "L40S": 60_000},
    min_edges_float64={"A100": 60_000, "A30": 12_000, "T4": 5_000, "V100": 30_000, "L40S": 20_000},
    default_min_edges=60_000, default_min_edges_float64=60_000,
)

#: The same for training mode (a step with a force loss, second derivatives included).
AUTO_POLICY_TRAINING = AutoPolicy(
    min_edges={"A100": 60_000, "A30": 20_000, "T4": 20_000, "V100": 60_000, "L40S": 60_000},
    min_edges_float64={"A100": 60_000, "A30": 12_000, "T4": 5_000, "V100": 30_000, "L40S": 20_000},
    default_min_edges=60_000, default_min_edges_float64=60_000,
)


def supported(device: torch.device, dtype: torch.dtype) -> bool:
    """Triton kernels on a CUDA device in float32 or float64."""
    from xnn.common.models.dispersion_fast._triton import supported as triton_supported
    return triton_supported(device, dtype)
