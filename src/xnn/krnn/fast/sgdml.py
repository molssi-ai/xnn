"""Fast path of the GDML / sGDML kernel expansion (Triton kernels).

The reference forms, chunk by chunk, the differences between the descriptors of
the batch and every permuted training descriptor, and makes several passes over
them. The fast path reads each training term once per tile of structures: one
kernel forms the kernel factors of every pair of a structure and a training term
(and the energies), a second contracts them with the training terms into the
gradient. A third pass of the same two kernels gives Hessian-vector products, so
forces stay differentiable (Hessians) as in the reference. The arithmetic is
float64, as in the reference, whatever the model's dtype.
"""
from __future__ import annotations

import torch

from xnn.common.models.fast import AutoPolicy

#: Sizes (structures times training terms, ``B M Q``) above which ``use_fast="auto"`` takes the
#: kernels in evaluation mode (energy and forces). On an A100, A30 or V100 they win from a single
#: structure (0.9 of the reference's time) to a thousand (0.2), with a fraction of its memory. A T4,
#: whose float64 throughput is low, loses for a single structure and wins from 12k.
AUTO_POLICY = AutoPolicy(min_edges_float64={"T4": 12_000}, default_min_edges=0, default_min_edges_float64=0)

#: The same for training mode (Hessians through the forces): 0.7 of the reference's time for one
#: structure, 0.1 for a thousand; a T4 from 6k.
AUTO_POLICY_TRAINING = AutoPolicy(min_edges_float64={"T4": 6_000}, default_min_edges=0, default_min_edges_float64=0)


def supported(device: torch.device, dtype: torch.dtype) -> bool:
    """Triton kernels on a CUDA device (the expansion is evaluated in float64)."""
    from xnn.common.models.dispersion_fast._triton import supported as triton_supported
    return triton_supported(device, dtype)
