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


def exact_softplus(x: Tensor) -> Tensor:
    """Evaluate ``log(1 + exp(x))`` exactly to round-off, with exact derivatives.

    One fused :func:`torch.nn.functional.softplus` kernel per pass in eager
    mode (two selections around ``log1p(exp(-|x|))`` under TorchScript). The
    fused kernel returns ``x`` above its threshold and so drops the tail ``log(1 + exp(-x))``,
    which is below the rounding of the value and of the derivative
    ``sigmoid(x)`` once ``exp(-threshold)`` is: a threshold of 40 in float64
    (``exp(-40) = 4e-18``) and 20 otherwise (``exp(-20) = 2e-9``), against
    the default 20 that drops ``2e-9`` in float64.

    Parameters
    ----------
    x : Tensor
        Input values of any shape.

    Returns
    -------
    Tensor
        Softplus values with the same shape as ``x``.
    """
    if torch.jit.is_scripting():
        # TorchScript differentiates softplus symbolically through exp(x), which
        # overflows for large x; selections around exp(-|x|) stay finite (and
        # are fused by the TorchScript fuser)
        positive = x > 0
        exponent = torch.where(positive, -x, x)
        return torch.where(positive, x, torch.zeros_like(x)) + torch.log1p(torch.exp(exponent))
    threshold = 40.0 if x.dtype == torch.float64 else 20.0
    return nn.functional.softplus(x, 1.0, threshold)


def shifted_softplus(x: Tensor) -> Tensor:
    """Shifted softplus ``ssp(x) = ln(0.5 e^x + 0.5)`` with ``ssp(0) = 0``.

    The activation of SchNet (Schuett et al., NIPS 2017) and PhysNet
    (Unke & Meuwly 2019); smooth with infinite order of continuity, which is
    what keeps the predicted potential-energy surface (and thus the autograd
    forces) smooth.

    Evaluated with :func:`exact_softplus` minus ``ln 2``, exact to round-off
    for all ``x`` in float32 and float64. TorchScript-compatible.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of any shape.

    Returns
    -------
    torch.Tensor
        ``ln(0.5 e^x + 0.5)``, same shape as ``x``.
    """
    return exact_softplus(x) - math.log(2.0)


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


def he_orthogonal_(weight: Tensor) -> Tensor:
    """He-scaled orthogonal initialization with exact standardization (GemNet, section 6).

    A random (semi-)orthogonal matrix is standardized per output unit to zero
    mean and unit variance and scaled to variance ``1 / fan_in``.

    Parameters
    ----------
    weight : Tensor
        A ``(fan_out, fan_in)`` weight matrix, modified in place.

    Returns
    -------
    Tensor
        ``weight``.
    """
    with torch.no_grad():
        nn.init.orthogonal_(weight)
        var, mean = torch.var_mean(weight, dim=1, unbiased=True, keepdim=True)
        weight.sub_(mean).div_(torch.sqrt(var + 1e-6)).mul_(weight.shape[1] ** -0.5)
    return weight


#: Weight initializations by name (:func:`init_linear_`); ``"torch"`` keeps
#: PyTorch's default.
WEIGHT_INITS = {
    "xavier_uniform": nn.init.xavier_uniform_,
    "xavier_normal": nn.init.xavier_normal_,
    "glorot_orthogonal": glorot_orthogonal_,
    "he_orthogonal": he_orthogonal_,
    "zeros": nn.init.zeros_,
    "torch": None,
}


def init_linear_(linear: nn.Linear, weight: str = "xavier_uniform") -> nn.Linear:
    """Initialize a dense layer by name: its weight by ``weight``, its bias to zero.

    Parameters
    ----------
    linear : torch.nn.Linear
        The layer, modified in place.
    weight : str, optional
        A name of :data:`WEIGHT_INITS`, by default ``"xavier_uniform"``;
        ``"torch"`` keeps PyTorch's initialization of weight and bias.

    Returns
    -------
    torch.nn.Linear
        ``linear``.

    Raises
    ------
    ValueError
        For an unknown name.
    """
    if weight not in WEIGHT_INITS:
        raise ValueError(f"weight_init must be one of {sorted(WEIGHT_INITS)}, got {weight!r}")
    init = WEIGHT_INITS[weight]
    if init is not None:
        with torch.no_grad():
            init(linear.weight)
            if linear.bias is not None:
                linear.bias.zero_()
    return linear


def init_linears_(module: nn.Module, weight: str = "xavier_uniform") -> nn.Module:
    """Apply :func:`init_linear_` to every dense layer inside ``module``.

    Returns
    -------
    torch.nn.Module
        ``module``.
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            init_linear_(m, weight)
    return module


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
    """Softplus ``log(1 + e^x)`` evaluated exactly at every ``x`` (module form of :func:`exact_softplus`).

    :class:`torch.nn.Softplus` switches to ``x`` above its threshold and drops
    the ``log1p`` tail; this one keeps it (the softplus of the HDNNP networks).
    """

    def forward(self, x: Tensor) -> Tensor:
        return exact_softplus(x)


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
    if dim_size == 1:
        # one bucket: a reduction, not every row's atomic add into the same addresses
        return src.sum(0, keepdim=True)
    shape = [dim_size] + list(src.shape[1:])
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    return out.index_add(0, index, src)


def gather_rows(src: Tensor, index: Tensor) -> Tensor:
    """Rows ``src[index]`` (dim 0), with the faster backward of the device.

    The node -> edge gather of message passing, the counterpart of
    :func:`scatter_sum`. On CUDA it indexes (``src[index]``), whose backward
    sorts the indices and accumulates without atomics, and whose second
    derivative keeps less memory; elsewhere it uses
    :func:`torch.index_select`, whose backward (an ``index_add``) is the faster
    one on CPUs. The values are the same.

    Parameters
    ----------
    src : torch.Tensor
        Tensor of shape ``(N, *feature_dims)`` (e.g. per-node features).
    index : torch.Tensor
        1-D integer tensor of length ``E``, the row of ``src`` for each output
        row.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(E, *feature_dims)``.

    Notes
    -----
    Compatible with :func:`torch.jit.script`.
    """
    if src.is_cuda:
        return src[index]
    return src.index_select(0, index)


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
    if dim_size == 1:
        # one structure: a reduction, not every atom's atomic add into one address
        return src.to(torch.float64).sum(0, keepdim=True)
    out = torch.zeros(dim_size, dtype=torch.float64, device=src.device)
    return out.index_add(0, index, src.to(torch.float64))


def structure_sums(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """:func:`structure_sum` of several per-atom rows at once: ``src`` ``(T, N)``
    to float64 totals ``(T, dim_size)`` in one scatter (one cast, one zero
    fill and one index add for every row instead of three launches per row).
    """
    if dim_size == 1:
        # one structure: a reduction, not every atom's atomic adds into the same few addresses
        return src.to(torch.float64).sum(1, keepdim=True)
    out = torch.zeros(int(src.shape[0]), dim_size, dtype=torch.float64, device=src.device)
    return out.index_add(1, index, src.to(torch.float64))


#: Standard atomic weights (u) by atomic number, Z = 0 to 99 (the ASE table),
#: for :func:`center_of_mass`.
ATOMIC_MASSES = [
    1.0000, 1.0080, 4.0026, 6.9400, 9.0122, 10.8100, 12.0110, 14.0070, 15.9990, 18.9984,
    20.1797, 22.9898, 24.3050, 26.9815, 28.0850, 30.9738, 32.0600, 35.4500, 39.9480, 39.0983,
    40.0780, 44.9559, 47.8670, 50.9415, 51.9961, 54.9380, 55.8450, 58.9332, 58.6934, 63.5460,
    65.3800, 69.7230, 72.6300, 74.9216, 78.9710, 79.9040, 83.7980, 85.4678, 87.6200, 88.9058,
    91.2240, 92.9064, 95.9500, 97.9072, 101.0700, 102.9055, 106.4200, 107.8682, 112.4140, 114.8180,
    118.7100, 121.7600, 127.6000, 126.9045, 131.2930, 132.9055, 137.3270, 138.9055, 140.1160, 140.9077,
    144.2420, 144.9128, 150.3600, 151.9640, 157.2500, 158.9254, 162.5000, 164.9303, 167.2590, 168.9342,
    173.0540, 174.9668, 178.4900, 180.9479, 183.8400, 186.2070, 190.2300, 192.2170, 195.0840, 196.9666,
    200.5920, 204.3800, 207.2000, 208.9804, 208.9824, 209.9872, 222.0176, 223.0197, 226.0254, 227.0277,
    232.0377, 231.0359, 238.0289, 237.0482, 244.0642, 243.0614, 247.0703, 247.0703, 251.0796, 252.0830,
]


def center_of_mass(pos: Tensor, atomic_numbers: Tensor, batch: Tensor,
                   num_graphs: int, masses: Tensor) -> Tensor:
    """Center of mass of every structure of a batch.

    Parameters
    ----------
    pos : Tensor
        Positions ``(N, 3)``.
    atomic_numbers : Tensor
        Atomic numbers ``(N,)``.
    batch : Tensor
        Structure index of every atom ``(N,)``.
    num_graphs : int
        Number of structures ``B``.
    masses : Tensor
        Mass table indexed by atomic number.

    Returns
    -------
    Tensor
        The centers of mass ``(B, 3)``.
    """
    m = masses[atomic_numbers].to(pos.dtype).unsqueeze(-1)
    total = scatter_sum(m, batch, num_graphs)
    return scatter_sum(m * pos, batch, num_graphs) / total


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
