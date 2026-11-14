"""Fast path of QRNN: the fused AEV of ANI and the grouped element networks.

QRNN evaluates its per-element networks three times (two charge cycles and the
energy), one small matrix product per element and layer. The fast path stacks
the weights of all elements and evaluates every layer as one batched product
over the atoms of each element, padded to the largest element
(:meth:`~xnn.dnn.models.base._ElementNetworks.groupable`), and computes the AEV
with the kernels of :mod:`xnn.dnn.fast.aev`. Both are plain differentiable
operations, so forces and force training take them as well.
"""
from __future__ import annotations

from xnn.common.models.fast import AutoPolicy

#: When ``use_fast="auto"`` takes the fast path in evaluation mode (energy and forces). It saves
#: kernel launches, so it wins from the smallest molecules (0.63-0.75 of the reference's time up to
#: 1000 atoms on an A100) until the padded products cost as much as they save (a tie near 500k edges
#: in float32, 300k in float64). An unmeasured GPU takes the A100's bounds.
AUTO_POLICY = AutoPolicy(
    max_edges={"A100": 500_000}, max_edges_float64={"A100": 300_000},
    default_max_edges=500_000, default_max_edges_float64=300_000,
)

#: The same for training mode (a step with a force loss, second derivatives included): 0.43-0.55 of
#: the reference's time up to 1000 atoms, a tie near 300k edges in both precisions.
AUTO_POLICY_TRAINING = AutoPolicy(
    max_edges={"A100": 300_000}, max_edges_float64={"A100": 300_000},
    default_max_edges=300_000, default_max_edges_float64=300_000,
)
