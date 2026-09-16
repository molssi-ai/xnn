"""Tensor operations shared across model families (cnn / dnn / gnn).

Kept here, at the ``models`` level, because they are common to more than one
architecture type. Per-family helpers live under the family package instead
(e.g. ``models/gnn/base.py``), and graph-level reductions that every model needs
live on :class:`~xnn.common.models.base.InteratomicPotential` (``aggregate_energy``).
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


def build_triplets(edge_index: Tensor, num_nodes: int) -> tuple[Tensor, Tensor, Tensor]:
    """Enumerate neighbour pairs ``(j, k)`` sharing a centre ``i``.

    For every centre atom, all unordered pairs of its incoming edges are
    returned. Fully vectorised (no Python loop over atoms): edges are grouped by
    their destination (centre) node and, within each group of ``c`` edges, the
    ``c*(c-1)/2`` pairs are enumerated with a shared lower-triangular index
    template -- the same scheme ``torchani`` uses. Used by the angular
    descriptors of the ``dnn`` family (ANI / HDNNP) and by the valence-angle
    enumeration of the ``ffnn`` family (ReaxFF).

    Parameters
    ----------
    edge_index : Tensor
        Edge index of shape ``(2, E)``; row 0 is the source (neighbour) and
        row 1 the destination (centre) node of each edge.
    num_nodes : int
        Number of atoms (nodes) in the graph.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(edge_jk_first, edge_jk_second, center)``, each of shape ``(T,)``
        where ``T`` is the number of triplets. The first two index into the
        edge dimension (the two edges forming a triplet, ``first < second`` in
        the per-centre ordering) and ``center`` is the shared centre node. All
        three are empty long tensors when no triplet exists.
    """
    device = edge_index.device
    dst = edge_index[1]
    # Group edges by their centre (destination) node.
    order = torch.argsort(dst, stable=True)          # (E,) edge ids, centre-grouped
    counts = torch.bincount(dst, minlength=num_nodes)  # (num_nodes,) edges per centre
    n_pairs = counts * (counts - 1) // 2               # pairs per centre
    total = int(n_pairs.sum())
    if total == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, empty

    # Start offset of each centre's block within `order`.
    offsets = torch.cumsum(counts, 0) - counts         # (num_nodes,)
    # Centre node of every emitted pair, and that centre's block offset/count.
    center = torch.repeat_interleave(torch.arange(num_nodes, device=device), n_pairs)
    base = torch.repeat_interleave(offsets, n_pairs)   # (T,) offset into `order`
    cnt = torch.repeat_interleave(counts, n_pairs)     # (T,) edges at this centre

    # Local pair (a, b) with 0 <= a < b < cnt, laid out from a shared template
    # sized to the largest centre, then masked down to each centre's count.
    m = int(counts.max())
    tri = torch.tril_indices(m, m, -1, device=device)  # (2, m*(m-1)/2): b > a
    # position of each emitted pair within its centre's pair-list
    pair_pos = torch.arange(total, device=device) - (
        torch.cumsum(n_pairs, 0) - n_pairs).repeat_interleave(n_pairs)
    a_local = tri[1][pair_pos]                          # smaller local edge index
    b_local = tri[0][pair_pos]                          # larger local edge index
    first = order[base + a_local]
    second = order[base + b_local]
    return first, second, center
