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
#: 1000 atoms on an A100, 0.66-0.72 on an A30, V100, L40S or T4 in float32) until the padded products
#: cost as much as they save: near 500k edges on an A100 or L40S, 100k on an A30 or T4, 250k on a V100
#: in float32. GPUs with little float64 throughput tie earlier in float64 (an L40S near 20k edges) or
#: never win (a T4). An unmeasured GPU takes the A100's bounds.
AUTO_POLICY = AutoPolicy(
    max_edges={"A100": 500_000, "L40S": 500_000, "A30": 100_000, "T4": 100_000, "V100": 250_000},
    max_edges_float64={"A100": 300_000, "L40S": 20_000, "A30": 50_000, "T4": 0, "V100": 100_000},
    default_max_edges=500_000, default_max_edges_float64=300_000,
)

#: The same for training mode (a step with a force loss, second derivatives included): 0.43-0.59 of
#: the reference's time for small molecules on every GPU measured in float32, a tie near 300k edges on
#: an A100 or L40S, 200k on a V100 and 100k on an A30 or T4; in float64 earlier on GPUs with little
#: float64 throughput (an L40S near 20k edges, a T4 near 5k).
AUTO_POLICY_TRAINING = AutoPolicy(
    max_edges={"A100": 300_000, "L40S": 300_000, "A30": 100_000, "T4": 100_000, "V100": 200_000},
    max_edges_float64={"A100": 300_000, "L40S": 20_000, "A30": 80_000, "T4": 5_000, "V100": 100_000},
    default_max_edges=300_000, default_max_edges_float64=300_000,
)
