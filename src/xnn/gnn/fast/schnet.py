"""Fast path of the SchNet continuous-filter convolution (Triton kernels).

The reference block evaluates the filter-generating network, the envelope,
the gather of the neighbor features, their product with the filters and the
sum onto the centers as separate tensor operations, each writing an ``(E, F)``
tensor that autograd keeps for the backward pass. Here the filter network stays
as it is (two dense layers and the shifted softplus between them), and one
kernel applies the second layer's shifted softplus (when the model has one) and
the envelope, gathers ``h[src]``, multiplies and adds the products onto
``out[dst]``: the filters and messages are never stored, only the second
layer's output is saved. The backward pass is one kernel for the feature
gradient, the gradient of the second layer's output and that of the envelope.

The gradient reaches the features, the filter network (its weights included)
and the envelope, so forces, stress and the weight gradients of a force loss:
the backward kernel is itself differentiable, through a kernel of the same
form, so training takes the fused convolution as well.
"""
from __future__ import annotations

import torch

from xnn.common.models.fast import AutoPolicy

#: Edge counts above which ``use_fast="auto"`` takes the kernels in evaluation mode (energy
#: and forces). Below them a call is bound by kernel launches and the two paths tie, the
#: kernels up to a few percent behind; above them the kernels win on every GPU measured. The
#: crossover comes earlier in float64, where the reference's elementwise passes cost more, and
#: on GPUs with little float64 throughput (T4) the kernels win at every size. A GPU in neither
#: table takes the largest measured thresholds.
AUTO_POLICY = AutoPolicy(
    min_edges={"T4": 6_000, "V100": 15_000, "A30": 6_000, "A100": 30_000},
    min_edges_float64={"T4": 0, "V100": 8_000, "A30": 4_000, "A100": 25_000},
    default_min_edges=30_000,
    default_min_edges_float64=25_000,
)

#: The same for training mode (a step with a force loss, second derivatives included).
AUTO_POLICY_TRAINING = AutoPolicy(
    min_edges={"T4": 6_000, "V100": 15_000, "A30": 6_000, "A100": 30_000},
    min_edges_float64={"T4": 0, "V100": 4_000, "A30": 2_000, "A100": 15_000},
    default_min_edges=30_000,
    default_min_edges_float64=15_000,
)


def supported(device: torch.device, dtype: torch.dtype) -> bool:
    """Triton kernels on a CUDA device in float32 or float64."""
    from xnn.common.models.dispersion_fast._triton import supported as triton_supported
    return triton_supported(device, dtype)
