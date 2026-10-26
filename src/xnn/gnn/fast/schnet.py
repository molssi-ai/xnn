"""Fast path of the SchNet continuous-filter convolution (Triton kernels).

The reference block evaluates the filter-generating network, the envelope,
the gather of the neighbor features, their product with the filters and the
sum onto the centers as separate tensor operations, each writing an ``(E, F)``
tensor that autograd keeps for the backward pass. Here the two dense layers
stay matrix products (the ``-ln 2`` of the first shifted softplus is folded
into the second bias), and one kernel applies the last shifted softplus and
the envelope, gathers ``h[src]``, multiplies and adds the products onto
``out[dst]``: the filters and messages are never stored. The backward pass is
one kernel for the feature gradient and the gradient of the second layer's
output, then two matrix products back to the radial expansion. Only the two
pre-activations are saved.

The gradient reaches the features, the radial expansion and the envelope (so
forces and stress), not the weights, and only once: evaluations that train
(:attr:`torch.nn.Module.training`) use the reference block.
"""
from __future__ import annotations

import torch

from xnn.common.models.fast import AutoPolicy

#: Edge counts above which ``use_fast="auto"`` takes the kernels. Below a few
#: thousand edges the launch cost of the kernels outweighs the elementwise passes
#: they remove, and the paper's filter network (many radial functions) keeps the
#: reference ahead up to about ten thousand; above the threshold the kernels win
#: on every GPU measured (T4, V100, A30) in both precisions, and use about a third
#: of the memory at any size.
AUTO_POLICY = AutoPolicy(default_min_edges=20_000, default_min_edges_float64=20_000)


def supported(device: torch.device, dtype: torch.dtype) -> bool:
    """Triton kernels on a CUDA device in float32 or float64."""
    from xnn.common.models.dispersion_fast._triton import supported as triton_supported
    return triton_supported(device, dtype)
