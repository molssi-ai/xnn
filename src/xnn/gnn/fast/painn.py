"""Fast path of the PaiNN message block (Triton kernels).

The reference message block gathers ``phi(s_j)`` and ``v_j`` to the edges,
multiplies them with the filters and directions and sums the products onto the
centers as separate tensor operations, each writing an ``(E, F)`` or
``(E, 3, F)`` tensor that autograd keeps for the backward pass. Here the network
``phi`` and the filters stay as they are (atom-wise layers and one matrix product
for all blocks), and one kernel per block gathers, multiplies and adds the
messages onto the centers: no edge-sized tensor is written. The backward pass is
one kernel that adds the gradients of ``phi(s)`` and ``v`` onto the neighbors and
writes those of the filters and directions.

The gradient reaches the features, the filters and the directions (so forces,
stress and, through ``phi`` and the filters, the weights). The backward kernel
is itself differentiable, through one more kernel of the same form, so the
second derivatives of a force loss run fused as well and training takes the
kernels too.
"""
from __future__ import annotations

import torch

from xnn.common.models.fast import AutoPolicy

#: Edge counts above which ``use_fast="auto"`` takes the kernels in evaluation mode (energy
#: and forces): every size, since the kernels were faster than the reference from a single
#: small molecule on, on every GPU measured (T4, V100, A30) in both precisions, and use less
#: memory.
AUTO_POLICY = AutoPolicy(default_min_edges=0, default_min_edges_float64=0)

#: The same for training mode (a step with a force loss, second derivatives included): every
#: size as well, on the same GPUs.
AUTO_POLICY_TRAINING = AutoPolicy(default_min_edges=0, default_min_edges_float64=0)


def supported(device: torch.device, dtype: torch.dtype) -> bool:
    """Triton kernels on a CUDA device in float32 or float64."""
    from xnn.common.models.dispersion_fast._triton import supported as triton_supported
    return triton_supported(device, dtype)
