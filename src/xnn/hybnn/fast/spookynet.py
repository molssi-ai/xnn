"""Fast paths of SpookyNet: a compiled block for evaluation, fused MLPs for training.

Evaluation: the interaction modules, the nonlocal attention included, run as one block compiled with
:func:`torch.compile` (see :mod:`xnn.common.models.compiled`). The sizes of the attention's padding (the
structures and the largest structure) enter as the shape of an input, so a new batch does not compile
the block again. The compiled code fuses the elementwise work of all modules, which is what a SpookyNet
evaluation spends its launches on.

Training: every residual MLP (the grouped ones of the local and nonlocal interactions, the single ones
and the residual blocks) runs as one fused function per pass, its second derivatives included
(:mod:`xnn.hybnn.fast.mlp_kernels`). They run a quarter fewer kernels than the reference, which counts
in captured steps; in ordinary steps on small systems their orchestration on the host costs as much.
"""
from typing import List

import torch
from torch import Tensor, nn

from xnn.common.models.fast import AutoPolicy

#: A threshold no graph reaches: the reference.
NEVER = 1 << 62

#: When ``use_fast="auto"`` takes the compiled block in evaluation mode (energy and forces): below an
#: edge count, since it saves kernel launches. Larger graphs, and large batches of small molecules,
#: are bound by the GPU's work, where the compiled code is not faster. A T4 gains in float32 at every
#: size and never in float64 (its float64 throughput is low).
AUTO_POLICY = AutoPolicy(min_edges_float64={"T4": NEVER}, max_edges={"T4": NEVER}, default_min_edges=0,
                         default_min_edges_float64=0, default_max_edges=30_000, default_max_edges_float64=30_000)

#: Edge counts above which ``use_fast="auto"`` takes the fused MLPs in an ordinary training step (a force
#: loss, second derivatives included), per GPU model and precision. Ordinary steps on small systems are
#: bound by the host, where the fused MLPs' Python orchestration costs as much as the launches it saves.
AUTO_POLICY_TRAINING = AutoPolicy(min_edges={"A30": 30_000, "T4": 30_000, "V100": 70_000},
                                  min_edges_float64={"A100": 50_000, "A30": 16_000, "V100": 30_000},
                                  default_min_edges=NEVER, default_min_edges_float64=NEVER)

#: The same for a captured training step (a padded batch with an index plan, replayed as a CUDA graph),
#: where only the GPU's work counts: the fused MLPs run fewer kernels at every size (not in float64 on a
#: T4).
AUTO_POLICY_CAPTURED = AutoPolicy(min_edges_float64={"T4": NEVER}, default_min_edges=0, default_min_edges_float64=0)


def supported(device: torch.device, dtype: torch.dtype) -> bool:
    """torch.compile on a CUDA device in float32 or float64."""
    from xnn.common.models.compiled import supported as compiled_supported
    return compiled_supported(device, dtype)


def fused_supported(device: torch.device, dtype: torch.dtype) -> bool:
    """The fused MLP layers (:mod:`xnn.hybnn.fast.mlp_kernels`): a CUDA device with Triton, float32 or
    float64."""
    if device.type != "cuda" or dtype not in (torch.float32, torch.float64):
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def _params(acts: List[nn.Module], lins: List[nn.Module]) -> List[Tensor]:
    """The stacked parameters of one layer of ``G`` MLPs (views for one MLP): ``alpha``, ``beta``
    ``(G, F)``, the transposed weights ``(G, F, F')`` and the biases ``(G, F')`` if the layers have them."""
    if len(acts) == 1:
        lin = lins[0]
        out = [acts[0].alpha.unsqueeze(0), acts[0].beta.unsqueeze(0), lin.weight.t().unsqueeze(0)]
        return out if lin.bias is None else out + [lin.bias.unsqueeze(0)]
    out = [torch.stack([a.alpha for a in acts]), torch.stack([a.beta for a in acts]),
           torch.stack([lin.weight for lin in lins]).transpose(1, 2)]
    return out if lins[0].bias is None else out + [torch.stack([lin.bias for lin in lins])]


def residual_mlps(x: Tensor, mlps: List[nn.Module]) -> List[Tensor]:
    """``[m(x) for m in mlps]`` for residual MLPs of one structure (plain linear layers and Swish
    activations, all with biases or none), one fused evaluation for all of them."""
    from xnn.hybnn.fast.mlp_kernels import residual_mlps as fused
    params: List[Tensor] = []
    for i in range(len(mlps[0].residual)):
        blocks = [m.residual[i] for m in mlps]
        params += _params([b.activation1 for b in blocks], [b.linear1 for b in blocks])
        params += _params([b.activation2 for b in blocks], [b.linear2 for b in blocks])
    params += _params([m.activation for m in mlps], [m.linear for m in mlps])
    return list(fused(x, len(mlps[0].residual), True, params).unbind(0))


def residual_block(x: Tensor, block: nn.Module) -> Tensor:
    """``x + W2 swish(W1 swish(x))`` of one residual block (both layers with biases or none), fused."""
    from xnn.hybnn.fast.mlp_kernels import residual_mlps as fused
    params = _params([block.activation1], [block.linear1]) + _params([block.activation2], [block.linear2])
    return fused(x, 1, False, params)[0]
