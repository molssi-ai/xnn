"""Fast path of the ET update layers (Triton kernels).

The reference layer gathers queries, keys, values and vector features onto the
pairs and multiplies them with the distance filters as separate tensor
operations, each writing a ``(P, F)`` or ``(P, 3, F)`` tensor that autograd keeps
for the backward pass (``P`` pairs, the self pairs included). Here the
atom-wise projections and the filter networks stay as they are, and two kernels
per layer do the pair work: one forms the logits of every pair and head (a
``(P, H)`` tensor), the other gathers, weighs and adds the messages onto the
centers, writing no pair-sized tensor. The attention nonlinearity runs between
them on the ``(P, H)`` logits, so all its derivatives are the reference's.

Both kernels are linear in each input, so their backward kernels are of the same
form and differentiable once more (:mod:`xnn.common.models.multilinear`): a
force loss trains through the fused layers. Third derivatives are not
supported.
"""
import torch

from xnn.common.models.fast import AutoPolicy

#: Edge counts above which ``use_fast="auto"`` takes the kernels in evaluation mode (energy
#: and forces): every size, since the fused layers were faster than the reference from a
#: single small molecule on, in both precisions, and use about a third of its memory.
AUTO_POLICY = AutoPolicy(default_min_edges=0, default_min_edges_float64=0)

#: The same for training mode (a step with a force loss, second derivatives included): from
#: about ten thousand edges on (A100). Below, a training step is bound by kernel launches and
#: the second derivatives' kernels cost slightly more than they save.
AUTO_POLICY_TRAINING = AutoPolicy(min_edges={"A100": 12000}, min_edges_float64={"A100": 10000},
                                  default_min_edges=12000, default_min_edges_float64=10000)


def supported(device: torch.device, dtype: torch.dtype) -> bool:
    """Triton kernels on a CUDA device in float32 or float64."""
    from xnn.common.models.dispersion_fast._triton import supported as triton_supported
    return triton_supported(device, dtype)
