"""Fast path of EquiformerV2: each transformer block compiled into fused kernels.

The blocks are compiled (:mod:`xnn.trfnn.fast.compiled`) with compiled first and
second derivatives in training mode. In compiled code the rotations of the
attention are the plain batched products, differentiated by the compiler,
instead of the reference's multilinear blocks.
"""
from xnn.common.models.fast import AutoPolicy
from xnn.trfnn.fast.compiled import supported  # noqa: F401

#: Edge counts below which ``use_fast="auto"`` takes the compiled blocks in evaluation mode
#: (energy and forces): they save kernel launches, so they win on small graphs (a single
#: molecule) and lose or tie on larger ones.
AUTO_POLICY = AutoPolicy(default_min_edges=0, default_min_edges_float64=0, default_max_edges=1000,
                         default_max_edges_float64=1000)

#: The same for training mode (a step with a force loss, second derivatives included), where
#: the compiled derivatives also keep a fraction of the reference's memory.
AUTO_POLICY_TRAINING = AutoPolicy(default_min_edges=0, default_min_edges_float64=0, default_max_edges=1000,
                                  default_max_edges_float64=1000)
