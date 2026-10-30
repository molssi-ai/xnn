"""Fast path of the Allegro tensor product (Triton kernels).

The reference contracts the two per-channel operands of every edge through
the block Wigner-3j tensor as one matrix product: it forms their full outer
product, ``d1 * d2`` entries per edge and channel, and multiplies it with the
dense coupling. The coupling has few nonzero entries (83 of 2835 for
``l_max = 2``). Here one kernel per coupling pattern sums exactly those
products per row, with the pattern unrolled and the coefficients read from
the module's own coupling tensor: the outer product is never formed. The
contraction is bilinear in the two operands; its gradients are the same
contraction with the axes of the coupling permuted, so forces and the weight
gradients of a force loss take the kernels at any derivative order.
"""
from __future__ import annotations

import torch

from xnn.common.models.fast import AutoPolicy

#: Edge counts above which ``use_fast="auto"`` takes the kernels in evaluation
#: mode (energy and forces). A GPU in neither table takes the default.
AUTO_POLICY = AutoPolicy(default_min_edges=0, default_min_edges_float64=0)

#: The same for training mode (a step with a force loss, second derivatives included).
AUTO_POLICY_TRAINING = AUTO_POLICY


def supported(device: torch.device, dtype: torch.dtype) -> bool:
    """Triton kernels on a CUDA device in float32 or float64."""
    from xnn.common.models.dispersion_fast._triton import supported as triton_supported
    return triton_supported(device, dtype)
