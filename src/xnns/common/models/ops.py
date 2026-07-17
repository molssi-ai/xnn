"""Tensor operations shared across model families (cnn / dnn / gnn).

Kept here, at the ``models`` level, because they are common to more than one
architecture type. Per-family helpers live under the family package instead
(e.g. ``models/gnn/base.py``), and graph-level reductions that every model needs
live on :class:`~xnns.common.models.base.InteratomicPotential` (``aggregate_energy``).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def shifted_softplus(x: Tensor) -> Tensor:
    """Shifted softplus ``ssp(x) = ln(0.5 e^x + 0.5)`` with ``ssp(0) = 0``.

    The activation of SchNet (Schuett et al., NIPS 2017) and PhysNet
    (Unke & Meuwly 2019); smooth with infinite order of continuity, which is
    what keeps the predicted potential-energy surface (and thus the autograd
    forces) smooth.

    Evaluated as ``max(x, 0) + log1p(exp(-|x|)) - ln 2`` -- exact for all
    ``x``, unlike :func:`torch.nn.functional.softplus`, which switches to the
    identity above its threshold and drops the ``log1p(exp(-x))`` tail
    (~1e-9 at the default threshold of 20). TorchScript-compatible.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of any shape.

    Returns
    -------
    torch.Tensor
        ``ln(0.5 e^x + 0.5)``, same shape as ``x``.
    """
    return F.relu(x) + torch.log1p(torch.exp(-x.abs())) - math.log(2.0)


def scatter_sum(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Sum rows of ``src`` into ``dim_size`` buckets given by ``index`` (dim 0).

    This is the canonical edge -> node aggregation for message passing,
    implemented with :meth:`torch.Tensor.index_add`. For each row ``e`` it
    accumulates ``out[index[e]] += src[e]``.

    Parameters
    ----------
    src : torch.Tensor
        Source tensor whose leading dimension is scattered. Shape
        ``(E, *feature_dims)`` (e.g. per-edge messages).
    index : torch.Tensor
        1-D integer tensor of length ``E`` giving, for each row of ``src``, the
        bucket (destination node) it is added into along dimension 0.
    dim_size : int
        Number of output buckets, i.e. the size of the leading dimension of the
        result (e.g. the number of nodes).

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(dim_size, *feature_dims)`` with the same dtype and
        device as ``src``, holding the per-bucket sums.

    Notes
    -----
    Written to stay compatible with :func:`torch.jit.script` so scriptable model
    cores (e.g. SchNet's ``node_energy``) can call it.
    """
    shape = [dim_size] + list(src.shape[1:])
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    return out.index_add(0, index, src)
