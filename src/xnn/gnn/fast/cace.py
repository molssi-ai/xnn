"""Fast path of the CACE edge-basis sums (Triton kernels).

The reference forms the edge basis ``chi_e = R(r_e) (x) L(u_e) (x) T_e``, one
``(n_rbf, L, C)`` tensor per edge (``C`` the products of the sender and receiver
element embeddings), and sums it onto the centers for the atom-centred basis
``A`` and, weighted by the sender's ``H(B)``, for the ``Bchi`` messages; autograd
keeps the edge-sized tensors. The receiver's embedding is constant over the
edges of a center, so the sums are ``theta_i (x) sum_{e -> i} p_e (x) x_e`` with
``p_e = R(r_e) (x) L(u_e)`` and ``x_e`` the sender's (weighted) embedding: the
neighbor sums of :mod:`xnn.gnn.fast.gemnet`, the same kernels, differentiable at
any order. ``chi`` is never formed. The ``Ar`` messages, the senders' ``A``
features times a decay per angular order, radial and edge channel, are summed
onto the receivers by kernels of their own (:mod:`xnn.gnn.fast.cace_kernels`):
the gathered features of every edge are never formed either.
"""
from __future__ import annotations

import torch

from xnn.common.models.fast import AutoPolicy

#: Edge counts above which ``use_fast="auto"`` takes the kernels in evaluation
#: mode (energy and forces). On an A100 they were faster than the reference
#: from about 15k edges (a few hundred water atoms) on in float32 and 8k in
#: float64, under half its time and memory from 100k edges. The A30, T4, V100
#: and L40S were measured the same way; a GPU in neither table takes the A100's
#: thresholds.
AUTO_POLICY = AutoPolicy(
    min_edges={"A100": 15_000, "L40S": 12_000, "V100": 12_000, "A30": 5_000, "T4": 5_000},
    min_edges_float64={"A100": 8_000, "L40S": 5_000, "V100": 5_000, "A30": 5_000, "T4": 2_000},
    default_min_edges=15_000,
    default_min_edges_float64=8_000,
)

#: The same for training mode (a step with a force loss, second derivatives
#: included): faster from about 12k edges in float32 and 5k in float64.
AUTO_POLICY_TRAINING = AutoPolicy(
    min_edges={"A100": 12_000, "L40S": 5_000, "V100": 5_000, "A30": 5_000, "T4": 2_000},
    min_edges_float64={"A100": 5_000, "L40S": 5_000, "V100": 5_000, "A30": 2_000, "T4": 2_000},
    default_min_edges=12_000,
    default_min_edges_float64=5_000,
)


def supported(device: torch.device, dtype: torch.dtype) -> bool:
    """Triton kernels on a CUDA device in float32 or float64."""
    from xnn.common.models.dispersion_fast._triton import supported as triton_supported
    return triton_supported(device, dtype)
