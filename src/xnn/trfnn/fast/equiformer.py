"""Fast path of Equiformer: each transformer block compiled into fused kernels.

The reference blocks run as many small operations per irreps block (the
depth-wise tensor products, the irreps linear maps, the gates, the equivariant
layer norms and the heads), and their backward passes add zero fills and
accumulations for every slice: thousands of kernels per evaluation, most of the
time elementwise. Here every block is compiled (:mod:`xnn.common.models.compiled`;
its tensor products are plain ``fx`` code, so the whole block is one graph), with
compiled first and second derivatives in training mode.
"""
from xnn.common.models.compiled import supported  # noqa: F401
from xnn.common.models.fast import AutoPolicy

#: Edge counts below which ``use_fast="auto"`` takes the compiled blocks in evaluation mode
#: (energy and forces): they save kernel launches, so they win on small graphs, and lose on
#: large ones (generic kernels on large tensors against cuBLAS).
AUTO_POLICY = AutoPolicy(default_min_edges=0, default_min_edges_float64=0, default_max_edges=12000,
                         default_max_edges_float64=12000)

#: The same for training mode (a step with a force loss, second derivatives included), where
#: the compiled derivatives also keep less memory. The float64 bound is lower: a dense graph
#: of a few thousand edges already loses there.
AUTO_POLICY_TRAINING = AutoPolicy(default_min_edges=0, default_min_edges_float64=0, default_max_edges=10000,
                                  default_max_edges_float64=4000)
