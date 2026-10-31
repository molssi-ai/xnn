"""Fast path of the DimeNet and DimeNet++ triplet interactions (Triton kernels).

The reference gathers per-edge quantities onto every triplet ``k -> j -> i``
and sums the triplet messages onto the edges ``j -> i``: DimeNet contracts
the triplet basis with a projection of the edge features per triplet (a
``(T, N_bilinear, F)`` gather and one small matrix product per triplet),
DimeNet++ multiplies the down-projected features with the mapped basis per
triplet. Both are the neighbor sums of GemNet's efficient bilinear layer,
``S[e, b, f] = sum_{t: ji_t = e} s[t, b] x[kj_t, f]``, followed by a contraction
with the weights per edge: the bilinear tensor for DimeNet, the second
(bias-free) basis layer for DimeNet++. The sums run in the kernels of
:mod:`xnn.gnn.fast.gemnet_kernels`, so nothing triplet-sized beyond the basis is
formed, and they are differentiable at any order, so forces and the weight
gradients of a force loss take the fused path as well.
"""
from __future__ import annotations

import torch

from xnn.common.models.fast import AutoPolicy

#: A threshold no graph reaches: ``"auto"`` keeps the reference.
NEVER = 1 << 62

#: Edge counts above which ``use_fast="auto"`` takes the DimeNet kernels in
#: evaluation mode (energy and forces). On an A100 they were faster than the
#: reference from about 3k edges on in float32 and 1k in float64 (the
#: per-triplet products of the reference grow with the triplets); on smaller
#: graphs the step is bound by kernel launches and the reference wins or ties.
#: A GPU in neither table takes the same thresholds.
AUTO_POLICY = AutoPolicy(
    min_edges={"A100": 3_000},
    min_edges_float64={"A100": 1_000},
    default_min_edges=3_000,
    default_min_edges_float64=1_000,
)

#: The same for training mode (a step with a force loss, second derivatives included).
AUTO_POLICY_TRAINING = AutoPolicy(
    min_edges={"A100": 3_000},
    min_edges_float64={"A100": 1_000},
    default_min_edges=3_000,
    default_min_edges_float64=1_000,
)

#: The DimeNet++ thresholds. Its triplet products are elementwise and cheap: on
#: an A100 the kernels win only on large graphs in float32 and, in float64,
#: where Triton has no tile product and the sums run as multiply-adds, at no
#: size measured (they still need half the memory); on the A30, T4, V100 and
#: L40S they win from about 4k edges in both precisions.
PP_AUTO_POLICY = AutoPolicy(
    min_edges={"A100": 10_000, "L40S": 4_000, "V100": 4_000, "A30": 4_000, "T4": 4_000},
    min_edges_float64={"A100": NEVER, "L40S": 4_000, "V100": 4_000, "A30": 4_000, "T4": 4_000},
    default_min_edges=10_000,
    default_min_edges_float64=NEVER,
)

#: The same for training mode.
PP_AUTO_POLICY_TRAINING = AutoPolicy(
    min_edges={"A100": 10_000, "L40S": 4_000, "V100": 4_000, "A30": 4_000, "T4": 4_000},
    min_edges_float64={"A100": NEVER, "L40S": 4_000, "V100": 4_000, "A30": 4_000, "T4": 4_000},
    default_min_edges=10_000,
    default_min_edges_float64=NEVER,
)


def supported(device: torch.device, dtype: torch.dtype) -> bool:
    """Triton kernels on a CUDA device in float32 or float64."""
    from xnn.common.models.dispersion_fast._triton import supported as triton_supported
    return triton_supported(device, dtype)
