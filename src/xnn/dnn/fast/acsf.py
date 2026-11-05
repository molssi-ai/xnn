"""Fast path of the atom-centered symmetry functions (HDNNP).

RuNNer evaluates the symmetry functions in fused Fortran loops; the reference
(:class:`~xnn.dnn.featurizers.acsf.AtomCenteredSymmetryFunctions`) evaluates the
radial functions of every edge and the angular functions of every triplet as a
sequence of tensor operations over ``(edges, functions)`` and ``(triplets,
functions)`` blocks, each read and written in full.

On a GPU, the Triton kernels of :mod:`xnn.dnn.fast.acsf_kernels` evaluate the
functions and sum them onto the centers in registers, for the function types 1,
2, 3 and 9 with at most two cosine or tanh cutoffs. Their gradient is
differentiable once, so they serve evaluation and training. On a CPU, and for
other function sets, the reference's two blocks run compiled by ``torch.compile`` with dynamic sizes
(C++ / OpenMP loops on a CPU, Triton kernels on a GPU); those are differentiable
once only, so they serve evaluation and a training step keeps the reference.
With the optional C++ kernels turned on (:mod:`xnn.common.models.cpu_extension`),
:mod:`xnn.dnn.fast.acsf_cpu` evaluates the kernels' function sets on a CPU in loops
over the atoms, as RuNNer does (evaluation).
"""
from __future__ import annotations

from xnn.common.models.dispersion_fast.atm_cpu import available
from xnn.common.models.fast import AutoPolicy

#: A threshold no graph reaches: the reference.
NEVER = 1 << 62

#: ``use_fast="auto"`` in evaluation: on a GPU the kernels and compiled blocks are faster than the
#: reference at every size measured (A100, A30, T4, V100, L40S in float32); float64 on an L40S or a T4 (little
#: float64 throughput, 1.1-1.5 of the reference from 192 atoms on an L40S, 1.1-1.3 at every size on a T4), an
#: unmeasured GPU and a CPU keep the reference.
AUTO_POLICY = AutoPolicy(
    min_edges={"A100": 0, "A30": 0, "T4": 0, "V100": 0, "L40S": 0},
    min_edges_float64={"A100": 0, "A30": 0, "T4": NEVER, "V100": 0, "L40S": NEVER},
    default_min_edges=NEVER, default_min_edges_float64=NEVER,
)

#: ``AUTO_POLICY`` where the optional C++ kernels are on, build and cover the function set: on a CPU
#: every size takes them (0.43-0.57 of RuNNer's time on one and eight threads, 81 to 375 atoms).
AUTO_POLICY_CPU = AutoPolicy(
    min_edges=AUTO_POLICY.min_edges, min_edges_float64=AUTO_POLICY.min_edges_float64,
    default_min_edges=NEVER, default_min_edges_float64=NEVER,
    cpu_min_edges=0, cpu_min_edges_float64=0,
)

#: Training mode: the Triton kernels (differentiable twice) on the GPUs measured (0.60-0.78 of the
#: reference's time on an A100, 0.55-0.84 on an A30, 0.55-0.73 on a V100, at every size; on an L40S 0.43-0.73
#: in float32, 1.3-1.8 in float64; on a T4 0.49-0.73 in float32, 1.3-1.8 in float64); elsewhere the reference
#: (the compiled blocks are differentiable once).
AUTO_POLICY_TRAINING = AutoPolicy(
    min_edges={"A100": 0, "A30": 0, "T4": 0, "V100": 0, "L40S": 0},
    min_edges_float64={"A100": 0, "A30": 0, "T4": NEVER, "V100": 0, "L40S": NEVER},
    default_min_edges=NEVER, default_min_edges_float64=NEVER,
)

__all__ = ["AUTO_POLICY", "AUTO_POLICY_TRAINING", "available"]
