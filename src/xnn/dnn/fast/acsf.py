"""Fast path of the atom-centered symmetry functions (HDNNP).

RuNNer evaluates the symmetry functions in fused Fortran loops; the reference
(:class:`~xnn.dnn.featurizers.acsf.AtomCenteredSymmetryFunctions`) evaluates the
radial functions of every edge and the angular functions of every triplet as a
sequence of tensor operations over ``(edges, functions)`` and ``(triplets,
functions)`` blocks, each read and written in full.

On a GPU, the Triton kernels of :mod:`xnn.dnn.fast.acsf_kernels` evaluate the
functions and sum them onto the centers in registers, for the function types 1,
2, 3 and 9 with at most two cosine or tanh cutoffs. Their gradient is
differentiable once, so they serve evaluation and training. On a CPU, the C++ /
OpenMP kernels of :mod:`xnn.dnn.fast.acsf_cpu` evaluate the same function sets
in loops over the atoms, as RuNNer does (evaluation). Other function sets run
the reference's two blocks compiled by ``torch.compile`` with dynamic sizes
(C++ / OpenMP loops on a CPU, Triton kernels on a GPU); those are differentiable
once only, so they serve evaluation and a training step keeps the reference.
"""
from __future__ import annotations

from xnn.common.models.dispersion_fast.atm_cpu import available
from xnn.common.models.fast import AutoPolicy

#: A threshold no graph reaches: the reference.
NEVER = 1 << 62

#: ``use_fast="auto"`` in evaluation: on a GPU the kernels and compiled blocks are faster than the
#: reference at every size measured (A100, A30, T4, V100); an unmeasured GPU and a CPU keep the reference.
AUTO_POLICY = AutoPolicy(
    min_edges={"A100": 0, "A30": 0, "T4": 0, "V100": 0},
    min_edges_float64={"A100": 0, "A30": 0, "T4": 0, "V100": 0},
    default_min_edges=NEVER, default_min_edges_float64=NEVER,
)

#: ``AUTO_POLICY`` where the C++ kernels build and cover the function set: on a CPU every size takes
#: them (0.43-0.57 of RuNNer's time on one and eight threads, 81 to 375 atoms).
AUTO_POLICY_CPU = AutoPolicy(
    min_edges=AUTO_POLICY.min_edges, min_edges_float64=AUTO_POLICY.min_edges_float64,
    default_min_edges=NEVER, default_min_edges_float64=NEVER,
    cpu_min_edges=0, cpu_min_edges_float64=0,
)

#: Training mode: the Triton kernels (differentiable twice) on the GPUs measured (0.60-0.78 of the
#: reference's time on an A100, 0.55-0.84 on an A30, at every size); elsewhere the reference (the
#: compiled blocks are differentiable once).
AUTO_POLICY_TRAINING = AutoPolicy(
    min_edges={"A100": 0, "A30": 0}, min_edges_float64={"A100": 0, "A30": 0},
    default_min_edges=NEVER, default_min_edges_float64=NEVER,
)

__all__ = ["AUTO_POLICY", "AUTO_POLICY_TRAINING", "available"]
