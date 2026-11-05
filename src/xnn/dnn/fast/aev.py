"""Fast path of the ANI atomic environment vectors (Triton kernels).

The reference radial and angular terms evaluate every edge's or triplet's
``(E, P)`` / ``(T, P)`` block of symmetry functions as separate tensor operations,
each written to memory and kept for the backward pass, and add them onto the
centers. Here the neighbor list, the triplets and their geometry (distances and
angles) stay as they are, and one kernel per term evaluates the functions and sums
them onto the centers in registers (one program per center, so no atomic adds and
the same result in every run): no edge- or triplet-sized block is written. The
gradient with respect to the distances and angles is one kernel per term, and
that gradient is itself differentiable through two more kernels of the same
forms, so a force loss trains through the kernels as well.
"""
from __future__ import annotations

import torch

from xnn.common.models.fast import AutoPolicy

#: A threshold no graph reaches: the reference.
NEVER = 1 << 62

#: When ``use_fast="auto"`` takes the kernels in evaluation mode (energy and forces). In float32
#: they win at every size on every GPU measured (0.80-0.98 of the reference). In float64 an A100
#: wins at every size; elsewhere they win on small graphs and lose on large ones, so they are taken
#: below an edge count (A30 25k, V100 60k); a T4 or L40S, weak in float64, keeps the reference; an
#: unmeasured GPU takes 25k.
AUTO_POLICY = AutoPolicy(
    min_edges_float64={"T4": NEVER, "L40S": NEVER},
    default_min_edges=0, default_min_edges_float64=0,
    max_edges_float64={"A100": NEVER, "A30": 25_000, "V100": 60_000},
    default_max_edges_float64=25_000,
)

#: The same for training mode (a step with a force loss, second derivatives included).
AUTO_POLICY_TRAINING = AutoPolicy(
    min_edges_float64={"T4": NEVER, "L40S": NEVER},
    default_min_edges=0, default_min_edges_float64=0,
    max_edges_float64={"A100": NEVER, "A30": 25_000, "V100": 60_000},
    default_max_edges_float64=25_000,
)


def supported(device: torch.device, dtype: torch.dtype) -> bool:
    """Triton kernels on a CUDA device in float32 or float64."""
    from xnn.common.models.dispersion_fast._triton import supported as triton_supported
    return triton_supported(device, dtype)
