"""3D convolutions on cuDNN's channels-last kernels, with its algorithm search.

For the 3D grids of the voxel models cuDNN's kernels for the channels-last
memory format are several times faster than those for the default one, and
for the double backward of force training its timed algorithm search
(``torch.backends.cudnn.benchmark``) beats its heuristic choice. Both settings
must reach the backward passes, which run outside any forward context, so
:func:`conv3d` evaluates the convolution as a
:mod:`multilinear <xnn.common.models.multilinear>` block whose value and
adjoints run cuDNN on channels-last copies of their operands (the outputs and
input gradients stay channels-last, as the next convolution takes them), in
batch chunks that bound the kernels' workspace, with the search on: the same convolution and the same
derivatives (to the round-off of the chosen algorithms), keeping only the
input and the kernel for the backward pass, as autograd does. The search runs
once per input shape and is cached by cuDNN for the process.
"""
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from xnn.common.models.multilinear import MultilinearBlock, multilinear


def supported(device: torch.device, dtype: torch.dtype) -> bool:
    """Whether the searched convolutions run: a CUDA device with cuDNN, float32 or float64."""
    return (device.type == "cuda" and torch.backends.cudnn.is_available()
            and dtype in (torch.float32, torch.float64))


#: Whether the convolutions use cuDNN's timed algorithm search (in addition to the
#: channels-last kernels).
SEARCH = True


def _searched():
    """cuDNN flags with the algorithm search on (see :data:`SEARCH`) and TF32 off.

    The channels-last kernels take TF32 when it is allowed (PyTorch's default for
    cuDNN), the default-format kernels of the reference do not: with it, float32
    results would differ from the reference by far more than round-off.
    """
    cudnn = torch.backends.cudnn
    return cudnn.flags(enabled=True, benchmark=SEARCH or cudnn.benchmark, benchmark_limit=cudnn.benchmark_limit,
                       deterministic=cudnn.deterministic, allow_tf32=False)


def _cl(t: Tensor) -> Tensor:
    return t.contiguous(memory_format=torch.channels_last_3d)


#: Output elements per cuDNN call. The workspace of the channels-last kernels grows
#: with the batch (several times the output), while their time per grid levels off
#: from a few dozen grids on: a large batch runs in batch chunks of this size, which
#: keeps the speed and bounds the workspace.
CHUNK_ELEMENTS = 1 << 24


def _chunks(n: int, per_item: int):
    size = max(1, CHUNK_ELEMENTS // max(per_item, 1))
    return [(i, min(n, i + size)) for i in range(0, n, size)]


class _Conv3d(MultilinearBlock):
    """``conv3d(x, weight)``, bilinear in the input and the kernel."""

    def __init__(self, stride: int, padding: int, dilation: int, groups: int):
        self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups

    def _out_size(self, n: int, k: int) -> int:
        return (n + 2 * self.padding - self.dilation * (k - 1) - 1) // self.stride + 1

    def value(self, ops, only: int = -1) -> Tensor:
        x, weight = ops
        spatial = [self._out_size(n, k) for n, k in zip(x.shape[2:], weight.shape[2:])]
        w = _cl(weight)
        out = None
        with _searched():
            for a, b in _chunks(x.shape[0], weight.shape[0] * math.prod(spatial)):
                y = F.conv3d(_cl(x[a:b]), w, None, self.stride, self.padding, self.dilation, self.groups)
                if a == 0 and b == x.shape[0]:
                    return y
                if out is None:
                    out = y.new_empty((x.shape[0], *y.shape[1:])).contiguous(memory_format=torch.channels_last_3d)
                out[a:b] = y
        return out

    def adjoints(self, g, ops, needs, only: int = -1):
        x, weight = ops
        w = _cl(weight)
        gx = torch.empty_like(x, memory_format=torch.channels_last_3d) if needs[0] else None
        gw = None
        with _searched():
            for a, b in _chunks(x.shape[0], g[0].numel()):
                dx, dw, _ = torch.ops.aten.convolution_backward(
                    _cl(g[a:b]), _cl(x[a:b]), w, None, [self.stride] * 3, [self.padding] * 3, [self.dilation] * 3,
                    False, [0, 0, 0], self.groups, [bool(needs[0]), bool(needs[1]), False])
                if needs[0]:
                    gx[a:b] = dx
                if needs[1]:
                    gw = dw if gw is None else gw + dw
        return [gx, gw.contiguous() if gw is not None else None]


def conv3d(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: int = 1, padding: int = 0,
           dilation: int = 1, groups: int = 1) -> Tensor:
    """``torch.nn.functional.conv3d`` on cuDNN's channels-last kernels, twice differentiable.

    Parameters
    ----------
    x : Tensor
        Input ``(B, C_in, X, Y, Z)`` on a CUDA device.
    weight : Tensor
        Kernel ``(C_out, C_in / groups, s, s, s)``.
    bias : Tensor or None, optional
        Per-channel bias, added outside the searched convolution.
    stride, padding, dilation, groups : int, optional
        As in ``conv3d`` (the same value on every axis).

    Returns
    -------
    Tensor
        ``(B, C_out, X', Y', Z')``.
    """
    out = multilinear(_Conv3d(int(stride), int(padding), int(dilation), int(groups)), x, weight)
    return out if bias is None else out + bias.view(1, -1, 1, 1, 1)
