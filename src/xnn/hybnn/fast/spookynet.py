"""Fast path of the SpookyNet local message sums (Triton kernels).

The local interaction sums ``G_s rho(r_ij) o resmlp_s(x_j)`` and, per angular component
``k``, ``G_p rho(r_ij) Y_1k(r_ij) o resmlp_p(x_j)`` and ``G_d rho(r_ij) Y_2k(r_ij) o
resmlp_d(x_j)`` over the neighbors of every atom. Each is SchNet's continuous-filter
convolution without the filter activation and envelope, the p and d sums over 3 and 5
feature blocks (the features repeated, the filter times the angular function), so the fast
path is the same fused kernel (:mod:`xnn.gnn.fast.schnet`): one program per atom, no
edge-sized message tensor. The kernel is twice differentiable, so force training takes it as
well.
"""
from __future__ import annotations

import torch

from xnn.common.models.fast import AutoPolicy

#: A threshold no graph reaches: the reference.
NEVER = 1 << 62

#: Edge counts above which ``use_fast="auto"`` takes the kernels in evaluation mode (energy and
#: forces). The local sums are a small part of SpookyNet's time (the residual MLPs, the attention and
#: the all-pairs terms dominate): measured on four GPUs they tie with the reference at best, so
#: ``"auto"`` keeps the reference; ``use_fast=True`` still takes them.
AUTO_POLICY = AutoPolicy(default_min_edges=NEVER, default_min_edges_float64=NEVER)

#: The same for training mode (a step with a force loss, second derivatives included).
AUTO_POLICY_TRAINING = AutoPolicy(default_min_edges=NEVER, default_min_edges_float64=NEVER)


def supported(device: torch.device, dtype: torch.dtype) -> bool:
    """Triton kernels on a CUDA device in float32 or float64."""
    from xnn.common.models.dispersion_fast._triton import supported as triton_supported
    return triton_supported(device, dtype)
