"""Fast path of EquiformerV2: each transformer block compiled into fused kernels.

The blocks are compiled (:mod:`xnn.trfnn.fast.compiled`) with compiled first and
second derivatives in training mode. In compiled code the rotations of the
attention are the plain batched products, differentiated by the compiler,
instead of the reference's multilinear blocks.
"""
from xnn.common.models.fast import AutoPolicy
from xnn.trfnn.fast.compiled import supported  # noqa: F401

#: Edge counts below which ``use_fast="auto"`` takes the compiled blocks in evaluation mode
#: (energy and forces): they save kernel launches, so they win on small graphs. On the A100
#: they won on a single small molecule (0.39-0.41 of upstream's time against 0.71-0.74)
#: and lost or tied from 4.5k edges.
AUTO_POLICY = AutoPolicy(default_min_edges=0, default_min_edges_float64=0, default_max_edges=1000,
                         default_max_edges_float64=1000)

#: The same for training mode (a step with a force loss, second derivatives included).
AUTO_POLICY_TRAINING = AutoPolicy(default_min_edges=0, default_min_edges_float64=0)
