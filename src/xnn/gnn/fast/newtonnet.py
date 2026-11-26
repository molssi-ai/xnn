"""Fast path of the NewtonNet interaction blocks (Triton kernels).

The reference forms the edge terms of the force and displacement features as separate tensor operations,
each writing an ``(E, 3, F)`` tensor (the directional term, the gathered neighbor features and their products)
that autograd keeps for the backward pass, and then sums them onto the atoms. Here the edge networks stay as
they are, and one kernel per sum gathers, multiplies and adds the terms onto the atoms: no ``(E, 3, F)``
tensor is written. The backward passes are kernels of the same two kinds (see
:mod:`xnn.gnn.fast.newtonnet_kernels`), themselves differentiable, so forces, stress, the second derivatives
of force training and Hessians run fused, with the reference's numbers to round-off.
"""
from __future__ import annotations

import torch

from xnn.common.models.fast import AutoPolicy

#: Edge counts above which ``use_fast="auto"`` takes the kernels in evaluation mode (energy and forces), per GPU
#: model (a substring of the device name) and precision; other GPUs use the defaults. Small graphs are bound by
#: kernel launches, where the reference's fewer operations win; on slower GPUs the kernels win earlier.
AUTO_POLICY = AutoPolicy(
    min_edges={"T4": 4_000, "V100": 25_000, "A30": 5_000, "L40S": 18_000},
    min_edges_float64={"T4": 0, "V100": 9_000, "A30": 5_000, "L40S": 5_000},
    default_min_edges=30_000, default_min_edges_float64=30_000)

#: The same for training mode (a step with a force loss, second derivatives included).
AUTO_POLICY_TRAINING = AutoPolicy(
    min_edges={"T4": 4_000, "V100": 18_000, "A30": 5_000, "L40S": 18_000},
    min_edges_float64={"T4": 0, "V100": 7_000, "A30": 5_000, "L40S": 0},
    default_min_edges=30_000, default_min_edges_float64=15_000)


def supported(device: torch.device, dtype: torch.dtype) -> bool:
    """Triton kernels on a CUDA device in float32 or float64."""
    from xnn.common.models.dispersion_fast._triton import supported as triton_supported
    return triton_supported(device, dtype)
