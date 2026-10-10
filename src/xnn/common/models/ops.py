"""Tensor operations shared across model families (gnn / cnn / dnn).

Kept here, at the ``models`` level, because they are common to more than one
architecture type. Per-family helpers live under the family package instead
(e.g. ``models/gnn/base.py``), and graph-level reductions that every model needs
live on :class:`~xnn.common.models.base.InteratomicPotential` (``aggregate_energy``).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
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


def glorot_orthogonal_(weight: Tensor, scale: float = 2.0) -> Tensor:
    """Initialize a weight matrix as a (semi-)orthogonal matrix with Glorot variance.

    The rows (or columns) are orthonormal, then the whole matrix is rescaled
    so that its elements have variance ``scale / (fan_in + fan_out)``: the
    initialization of the dense layers of DimeNet and SpookyNet.

    Parameters
    ----------
    weight : Tensor
        A 2-D weight tensor, modified in place.
    scale : float, optional
        Variance numerator, by default 2.0 (Glorot).

    Returns
    -------
    Tensor
        ``weight``.
    """
    with torch.no_grad():
        nn.init.orthogonal_(weight)
        fan_out, fan_in = weight.shape
        weight.mul_(math.sqrt(scale / ((fan_in + fan_out) * float(weight.var()))))
    return weight


def softplus_inverse(x):
    """Return ``y`` such that ``softplus(y) = x`` (``x > 0``, float or array).

    Evaluated as ``x + log(1 - exp(-x))`` (i.e. ``log(expm1(x))`` rearranged
    so the exponential never overflows for large ``x``); used to initialize
    parameters that a softplus keeps positive (PhysNet, SpookyNet).
    """
    import numpy as np
    return x + np.log(-np.expm1(-x))


class ShiftedSoftplus(nn.Module):
    """Module form of :func:`shifted_softplus`, for use inside ``nn.Sequential``."""

    def forward(self, x: Tensor) -> Tensor:
        return shifted_softplus(x)


class GaussianActivation(nn.Module):
    """Gaussian activation ``exp(-x^2)`` (the original ANI-1 hidden activation)."""

    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(-x * x)


class Softplus(nn.Module):
    """Softplus ``log(1 + e^x)`` evaluated exactly at every ``x``, as ``max(x, 0) + log1p(e^-|x|)``.

    :class:`torch.nn.Softplus` switches to ``x`` above its threshold and drops
    the ``log1p`` tail; this one keeps it (the softplus of the HDNNP networks).
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.relu(x) + torch.log1p(torch.exp(-x.abs()))


class ScaledTanh(nn.Module):
    """``1.59223 tanh(x)``, the scaled hyperbolic tangent of the RuNNer networks."""

    def forward(self, x: Tensor) -> Tensor:
        return 1.59223 * torch.tanh(x)


class Square(nn.Module):
    """``x^2``."""

    def forward(self, x: Tensor) -> Tensor:
        return x * x


def make_activation(activation) -> nn.Module:
    """Return a fresh activation module from a name or a module instance.

    The one activation factory of the model families (the per-element
    networks of ANI / HDNNP, the voxel convolution networks of the ``cnn``
    family).

    Parameters
    ----------
    activation : str or torch.nn.Module
        One of ``"silu"``, ``"celu"`` (ANI, alpha=0.1), ``"gaussian"``
        (original ANI-1), ``"tanh"``, ``"relu"``, ``"sigmoid"``, ``"ssp"``
        (the exact shifted softplus of SchNet / PhysNet), ``"softplus"``
        (:class:`Softplus`), ``"scaled_tanh"`` (:class:`ScaledTanh`),
        ``"square"`` or ``"linear"`` (the identity); or an ``nn.Module`` used
        as-is.

    Returns
    -------
    torch.nn.Module
        A fresh activation module.

    Raises
    ------
    ValueError
        If ``activation`` is an unknown name.
    """
    if isinstance(activation, nn.Module):
        return activation
    factories = {
        "silu": nn.SiLU,
        "celu": lambda: nn.CELU(alpha=0.1),   # ANI / torchani convention
        "gaussian": GaussianActivation,
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "ssp": ShiftedSoftplus,
        "softplus": Softplus,
        "scaled_tanh": ScaledTanh,
        "square": Square,
        "linear": nn.Identity,
    }
    key = str(activation).lower()
    if key not in factories:
        raise ValueError(f"unknown activation {activation!r}; "
                         f"choose one of {sorted(factories)}")
    return factories[key]()


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


def structure_sum(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Sum per-atom values into per-structure totals, accumulated in float64.

    A float32 total of tens of thousands of atoms is only good to its own
    rounding, about ``|E| * 6e-8`` (tenths of an eV for 1e4 to 1e5 water atoms),
    so per-structure energies are summed and returned in float64 whatever the
    model's dtype; the per-atom values keep theirs.

    Parameters
    ----------
    src : torch.Tensor
        Per-atom values of shape ``(N,)``.
    index : torch.Tensor
        Structure index of every atom, shape ``(N,)``.
    dim_size : int
        Number of structures.

    Returns
    -------
    torch.Tensor
        Float64 totals of shape ``(dim_size,)``.
    """
    out = torch.zeros(dim_size, dtype=torch.float64, device=src.device)
    return out.index_add(0, index, src.to(torch.float64))


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


def cell_volume(cell: Tensor) -> Tensor:
    """Volume of row-vector cells ``(..., 3, 3)`` as the scalar triple product.

    Used instead of ``torch.det`` for cell volumes: the backward of ``det``
    builds an LU solve in the *process default* dtype, so a float32 model
    served in a process whose default is float64 fails in the force pass
    (torch 2.5). The triple product differentiates in the cell's own dtype
    and is scriptable.
    """
    cross = torch.linalg.cross(cell[..., 1, :], cell[..., 2, :])
    return (cell[..., 0, :] * cross).sum(-1).abs()


class _SegmentSum(torch.autograd.Function):
    """Sum of consecutive segments of ``x`` (lengths given), differentiable to any order.

    ``torch.segment_reduce`` is a segmented reduction without atomics, but its
    built-in backward has no derivative of its own, which the recompute
    blocks need in training mode. Here the backward is the expansion
    (:class:`_SegmentExpand`) and that function's backward is this sum again,
    so the pair differentiates to any order.
    """

    @staticmethod
    def forward(ctx, x: Tensor, lengths: Tensor) -> Tensor:
        ctx.save_for_backward(lengths)
        return torch.segment_reduce(x, "sum", lengths=lengths, unsafe=True)

    @staticmethod
    def backward(ctx, g: Tensor):
        (lengths,) = ctx.saved_tensors
        return _SegmentExpand.apply(g, lengths), None


class _SegmentExpand(torch.autograd.Function):
    """Broadcast one value per segment to its elements (the adjoint of the segment sum)."""

    @staticmethod
    def forward(ctx, g: Tensor, lengths: Tensor) -> Tensor:
        ctx.save_for_backward(lengths)
        return torch.repeat_interleave(g, lengths, dim=0)

    @staticmethod
    def backward(ctx, h: Tensor):
        (lengths,) = ctx.saved_tensors
        return _SegmentSum.apply(h, lengths), None


def segment_sum(x: Tensor, lengths: Tensor) -> Tensor:
    """Sum of consecutive segments of ``x`` along dim 0; ``lengths`` (int64) sum to ``len(x)``.

    A segmented reduction (no atomics) for values that arrive grouped, such as
    the triplets of a center; differentiable to any order (see
    :class:`_SegmentSum`).
    """
    return _SegmentSum.apply(x, lengths)
