"""Building blocks shared by the Equiformer models (Equiformer and EquiformerV2).

* :func:`segment_softmax`, the attention normalization over the incoming edges of
  every atom, with an optional smooth-cutoff form;
* :class:`GraphDropPath`, stochastic depth per structure;
* :class:`RadialMLP`, the radial function (linear layers with layer norm and SiLU);
* :class:`SmoothLeakyReLU`, the activation of the attention logits.
"""
# NOTE: no `from __future__ import annotations` here -- PEP 563 stringifies
# the annotations that TorchScript needs to resolve.
from typing import List, Optional

import torch
from torch import Tensor, nn

from xnn.common.models.ops import scatter_sum


def segment_softmax(logits: Tensor, index: Tensor, num_segments: int,
                    weight: Optional[Tensor] = None) -> Tensor:
    """Softmax of edge logits over the edges that share a receiving atom.

    Without ``weight`` this is the attention normalization of the Equiformers,
    ``a_ij = exp(z_ij) / sum_k exp(z_ik)``. With a per-edge ``weight`` ``u_ij``
    (a cutoff envelope) it becomes ``a_ij = u_ij exp(z_ij) / (1 + sum_k u_ik
    exp(z_ik))``: a neighbor leaving the cutoff sphere fades out instead of
    dropping from the normalization, and the null term (a key of logit zero
    and zero value) lets the last neighbor fade out too, so the attention is
    continuous in the positions.

    Parameters
    ----------
    logits : Tensor
        Edge logits ``(E, H)``.
    index : Tensor
        Receiving atom of every edge ``(E,)``.
    num_segments : int
        Number of atoms.
    weight : Tensor, optional
        Per-edge weights ``(E,)`` in ``[0, 1]``.

    Returns
    -------
    Tensor
        The attention weights ``(E, H)``.
    """
    idx = index.unsqueeze(-1).expand_as(logits)
    zmax = torch.full((num_segments, logits.shape[1]), float("-inf"), dtype=logits.dtype,
                      device=logits.device)
    zmax = zmax.scatter_reduce(0, idx, logits.detach(), reduce="amax", include_self=True)
    if weight is None:
        ex = torch.exp(logits - zmax[index])
        return ex / (scatter_sum(ex, index, num_segments) + 1e-16)[index]
    zmax = zmax.clamp(min=0.0)
    ex = torch.exp(logits - zmax[index]) * weight.unsqueeze(-1)
    return ex / (scatter_sum(ex, index, num_segments) + torch.exp(-zmax))[index]


class GraphDropPath(nn.Module):
    """Stochastic depth per structure: drop a residual branch for whole structures.

    Parameters
    ----------
    drop_prob : float
        Probability of dropping the branch of a structure during training.
    """

    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor, batch: Tensor, num_graphs: int) -> Tensor:
        """Scale ``x`` (atoms on dim 0) by a kept-and-rescaled mask of its structure."""
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        mask = torch.floor(keep + torch.rand(num_graphs, dtype=x.dtype, device=x.device)) / keep
        shape: List[int] = [x.shape[0]] + [1] * (x.dim() - 1)
        return x * mask[batch].reshape(shape)


class RadialMLP(nn.Module):
    """The radial function: linear layers, each but the last followed by layer norm and SiLU.

    Parameters
    ----------
    channels : list of int
        Widths ``[n_in, hidden..., n_out]``.

    Attributes
    ----------
    net : torch.nn.Sequential
        ``Linear, LayerNorm, SiLU, ..., Linear``.
    """

    def __init__(self, channels: List[int]):
        super().__init__()
        layers: List[nn.Module] = []
        for k in range(1, len(channels)):
            layers.append(nn.Linear(channels[k - 1], channels[k]))
            if k < len(channels) - 1:
                layers.append(nn.LayerNorm(channels[k]))
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Map ``(E, n_in)`` edge scalars to ``(E, n_out)``."""
        return self.net(x)


class SmoothLeakyReLU(nn.Module):
    """``(1 + a) / 2 x + (1 - a) / 2 x (2 sigmoid(x) - 1)``, a smooth leaky ReLU of slope ``a``.

    Parameters
    ----------
    negative_slope : float, optional
        The slope ``a`` for large negative inputs, by default 0.2.
    """

    def __init__(self, negative_slope: float = 0.2):
        super().__init__()
        self.alpha = float(negative_slope)

    def forward(self, x: Tensor) -> Tensor:
        return (0.5 * (1.0 + self.alpha)) * x + (0.5 * (1.0 - self.alpha)) * x * (
            2.0 * torch.sigmoid(x) - 1.0)
