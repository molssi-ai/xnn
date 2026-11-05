"""Fast path of the PhysNet message sums (Triton kernels).

PhysNet's interaction layer sums ``G g(r_ij) * (W_J act(x_j) + b_J)`` over the
neighbors of every atom: the attention-masked features of the neighbors with a
filter linear in the radial basis. That is SchNet's continuous-filter
convolution without the filter activation and envelope, so the fast path is the
same fused kernel (:mod:`xnn.gnn.fast.schnet`): it gathers, multiplies and sums
in registers, one program per atom, without writing an edge-sized tensor. The
kernel is twice differentiable, so force training takes it as well.
"""
from __future__ import annotations

import torch

from xnn.common.models.fast import AutoPolicy

#: Edge counts above which ``use_fast="auto"`` takes the kernels in evaluation mode (energy and
#: forces). Below them the fused sums and the reference tie within a few percent; above them the
#: kernels take 0.63-0.83 of the reference's time. A T4 wins at every size in float64; an
#: unmeasured GPU takes 150k.
AUTO_POLICY = AutoPolicy(
    min_edges={"A100": 300_000, "A30": 50_000, "T4": 50_000, "V100": 200_000, "L40S": 120_000},
    min_edges_float64={"A100": 200_000, "A30": 50_000, "T4": 0, "V100": 100_000, "L40S": 20_000},
    default_min_edges=150_000, default_min_edges_float64=150_000,
)

#: The same for training mode (a step with a force loss, second derivatives included).
AUTO_POLICY_TRAINING = AutoPolicy(
    min_edges={"A100": 300_000, "A30": 50_000, "T4": 50_000, "V100": 200_000, "L40S": 120_000},
    min_edges_float64={"A100": 200_000, "A30": 50_000, "T4": 0, "V100": 50_000, "L40S": 20_000},
    default_min_edges=150_000, default_min_edges_float64=150_000,
)


def supported(device: torch.device, dtype: torch.dtype) -> bool:
    """Triton kernels on a CUDA device in float32 or float64."""
    from xnn.common.models.dispersion_fast._triton import supported as triton_supported
    return triton_supported(device, dtype)
