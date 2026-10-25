"""Equiformer, the equivariant graph attention transformer (Liao & Smidt, ICLR 2023).

An implementation of

* Y.-L. Liao, T. Smidt, "Equiformer: Equivariant Graph Attention Transformer for
  3D Atomistic Graphs", ICLR 2023 (arXiv:2206.11990),

written from the paper on e3nn irreps features and the xnn abstractions, and
consistent with the conventions of the authors' code (atomicarchitects/equiformer:
feature layouts, normalizations, initialization) without copying it.

Every atom carries an irreps feature ``x_i`` (``C_L`` type-L vectors per degree).
A Transformer block is pre-norm: equivariant layer norm, equivariant graph
attention, residual; layer norm, feed-forward network, residual. The graph
attention (section 4.2) mixes the two endpoints of an edge,
``x_ij = Linear_dst(x_i) + Linear_src(x_j)``, takes a depth-wise tensor product
(DTP) with the spherical harmonics of the edge weighted by a radial function,
``f_ij = Linear(x_ij (x)_DTP SH(r_ij))``, and derives

* MLP attention: ``a_ij = softmax_j(w^T LeakyReLU(f_ij^(0)))`` from the scalars;
* non-linear values: ``v_ij = Linear(Gate(f_ij) (x)_DTP SH(r_ij))`` (or the
  linear message ``v_ij = f_ij``);

summed as ``sum_j a_ij v_ij`` over ``h`` heads and projected. The dot-product
attention of the ablation (appendix C.3) is the alternative ``attention="dot_product"``.
The embedding adds a degree embedding (a DTP of a constant feature summed over
the neighbors and divided by ``sqrt(avg_degree)``) to the atom embedding; the
energy is a two-layer readout of the final scalars summed over the atoms and
divided by ``sqrt(avg_num_nodes)``.

Conventions worth knowing:

* The spherical harmonics are evaluated on ``r_j - r_i`` (center to neighbor),
  the negative of the xnn edge vector.
* The SE(3) models of the paper label every irrep even (``1e``, ``2e``), so
  tensor-product paths ignore parity; E(3) irreps (``1o``, ...) give the
  E(3)-Equiformer of appendix C.4.
* ``smooth_cutoff=True`` makes the energy continuous at the cutoff (not part of
  the paper): the degree-embedding messages are multiplied by a polynomial
  envelope and the attention becomes the envelope-weighted softmax of
  :func:`~xnn.trfnn.models.blocks.segment_softmax`.
"""
# NOTE: no `from __future__ import annotations` here -- PEP 563 stringifies
# the annotations that TorchScript needs to resolve.
import math
from typing import Dict, Final, List, Optional, Tuple

import torch
from e3nn import o3
from torch import Tensor, nn
from torch.nn import functional as F

from xnn.common.data import AtomicGraph
from xnn.common.featurizers import BesselRBF, ExpNormalSmearing, PolynomialCutoff
from xnn.common.models.ops import scatter_sum
from xnn.common.models.registry import register_model

from .base import TransformerPotential
from .blocks import GraphDropPath, RadialMLP, SmoothLeakyReLU, segment_softmax

# QM9 statistics at a 5 Angstrom cutoff, the reference code's defaults
_AVG_NUM_NODES = 18.03065905448718
_AVG_DEGREE = 15.57930850982666

# sqrt(2 pi) with the value of pi in the reference code's Gaussian basis
_SQRT_2PI = (2 * 3.14159) ** 0.5

# the paper's architectures by (l_max, E(3)): QM9 / MD17 (Tables 8 and 11) and OC20 (Table 14)
_PRESETS = {
    (1, False): ("256x0e+128x1e", "32x0e+16x1e", 8, "768x0e+384x1e"),
    (2, False): ("128x0e+64x1e+32x2e", "32x0e+16x1e+8x2e", 4, "384x0e+192x1e+96x2e"),
    (3, False): ("128x0e+64x1e+64x2e+32x3e", "32x0e+16x1e+16x2e+8x3e", 4,
                 "384x0e+192x1e+192x2e+96x3e"),
    (1, True): ("256x0e+64x0o+64x1e+64x1o", "32x0e+8x0o+8x1e+8x1o", 8,
                "768x0e+192x0o+192x1e+192x1o"),
    (2, True): ("128x0e+32x0o+32x1e+32x1o+16x2e+16x2o", "32x0e+8x0o+8x1e+8x1o+4x2e+4x2o", 4,
                "384x0e+96x0o+96x1e+96x1o+48x2e+48x2o"),
    (3, True): ("128x0e+64x0o+32x1e+32x1o+32x2e+32x2o+16x3e+16x3o",
                "32x0e+16x0o+8x1e+8x1o+8x2e+8x2o+4x3e+4x3o", 4,
                "384x0e+192x0o+96x1e+96x1o+96x2e+96x2o+48x3e+48x3o"),
}


def paper_irreps(l_max: int = 2, e3: bool = False, n_features: Optional[int] = None
                 ) -> Dict[str, object]:
    """The irreps of the paper's architectures, optionally rescaled.

    Parameters
    ----------
    l_max : int, optional
        1 (OC20), 2 (QM9, MD17) or 3 (MD17), by default 2.
    e3 : bool, optional
        The E(3) model (both parities) instead of the SE(3) one, by default False.
    n_features : int, optional
        Number of even scalars of the embedding; every multiplicity is scaled
        by ``n_features`` over the paper's value (128, 256 for ``l_max = 1``).

    Returns
    -------
    dict
        ``irreps_node_embedding``, ``irreps_head``, ``num_heads``, ``irreps_mlp_mid``.

    Raises
    ------
    ValueError
        For an ``l_max`` without a paper architecture.
    """
    if (l_max, e3) not in _PRESETS:
        raise ValueError(f"the paper has no architecture with l_max={l_max}")
    emb, head, heads, mid = _PRESETS[(l_max, e3)]
    if n_features is not None:
        base = o3.Irreps(emb)[0].mul

        def scale(irreps):
            out = o3.Irreps([(mul * n_features // base, ir) for mul, ir in o3.Irreps(irreps)])
            if any(mul == 0 for mul, _ in out):
                raise ValueError(f"n_features={n_features} leaves {irreps} with an empty irrep")
            return str(out)

        emb, head, mid = scale(emb), scale(head), scale(mid)
    return {"irreps_node_embedding": emb, "irreps_head": head, "num_heads": heads,
            "irreps_mlp_mid": mid}


def _moment_scale(fn) -> float:
    """The factor that normalizes the second moment of ``fn`` on a standard normal (e3nn's)."""
    from e3nn.math import normalize2mom
    norm = normalize2mom(fn)
    return 1.0 if norm._is_id else float(norm.cst)


def _sort_even_first(irreps: o3.Irreps) -> Tuple[o3.Irreps, List[int]]:
    """Sort by degree with the even irrep first; return the sorted irreps and ``old -> new`` indices."""
    keys = sorted((ir.l, -ir.p, i, mul) for i, (mul, ir) in enumerate(irreps))
    new_of_old = [0] * len(keys)
    for new, (_, _, old, _) in enumerate(keys):
        new_of_old[old] = new
    return o3.Irreps([(mul, (l, -negp)) for l, negp, _, mul in keys]), new_of_old


def _sort(irreps: o3.Irreps, even_first: bool) -> o3.Irreps:
    if even_first:
        return _sort_even_first(irreps)[0].simplify()
    return irreps.sort().irreps.simplify()


def _irreps_to_gate(irreps: o3.Irreps) -> Tuple[o3.Irreps, o3.Irreps, o3.Irreps]:
    """Split ``irreps`` into activated even scalars, gate scalars and gated irreps."""
    scalars = o3.Irreps([(mul, ir) for mul, ir in irreps if ir.l == 0 and ir.p == 1]).simplify()
    gated = o3.Irreps([(mul, ir) for mul, ir in irreps if not (ir.l == 0 and ir.p == 1)]).simplify()
    gates = o3.Irreps([(mul, "0e") for mul, _ in gated]).simplify()
    return scalars, gates, gated


def depthwise_tensor_product(irreps_in, irreps_sh, irreps_target,
                             internal_weights: bool) -> o3.TensorProduct:
    """The depth-wise tensor product: one weight per channel and path (``uvu``).

    Every channel of every input irrep couples with every spherical harmonic
    to the degrees present in ``irreps_target`` (and to even scalars); the
    outputs are sorted by degree with the even irrep first.

    Parameters
    ----------
    irreps_in, irreps_sh, irreps_target : e3nn.o3.Irreps or str
        The node irreps, the spherical-harmonic irreps and the wanted degrees.
    internal_weights : bool
        Weights shared by all edges (``True``) or given per edge (``False``,
        from a radial function).

    Returns
    -------
    e3nn.o3.TensorProduct
        The tensor product, component-normalized with no path normalization.
    """
    irreps_in, irreps_sh = o3.Irreps(irreps_in), o3.Irreps(irreps_sh)
    irreps_target = o3.Irreps(irreps_target)
    outs, instructions = [], []
    for i, (mul, ir_in) in enumerate(irreps_in):
        for j, (_, ir_sh) in enumerate(irreps_sh):
            for ir_out in ir_in * ir_sh:
                if ir_out in irreps_target or ir_out == o3.Irrep(0, 1):
                    instructions.append((i, j, len(outs), "uvu", True))
                    outs.append((mul, ir_out))
    irreps_out, new_of_old = _sort_even_first(o3.Irreps(outs))
    instructions = [(i, j, new_of_old[k], mode, train) for i, j, k, mode, train in instructions]
    tp = o3.TensorProduct(irreps_in, irreps_sh, irreps_out, instructions,
                          internal_weights=internal_weights, shared_weights=internal_weights,
                          irrep_normalization="component", path_normalization="none")
    if internal_weights:
        with torch.no_grad():
            for weight, ins in zip(tp.weight_views(), tp.instructions):
                fan_in = sum(irreps_sh[other.i_in2].mul for other in tp.instructions
                             if other.i_out == ins.i_out)
                weight.mul_(fan_in ** -0.5)
    return tp


class IrrepsLinear(nn.Module):
    """Equivariant linear map: one weight matrix per pair of equal irreps, a bias on the even scalars.

    ``y_w = sum_u W_uw x_u`` for every degree (no normalization of the
    weights in the forward), the weights stored flat in input-major order.

    Parameters
    ----------
    irreps_in, irreps_out : e3nn.o3.Irreps or str
        Input and output irreps (at most one even-scalar output entry).
    bias : bool, optional
        Add a bias to the even scalars, by default True.
    """

    n_out: Final[int]
    out_muls: Final[List[int]]
    out_dims: Final[List[int]]
    path_lo: Final[List[int]]
    path_hi: Final[List[int]]
    p_in_start: Final[List[int]]
    p_in_mul: Final[List[int]]
    p_w_start: Final[List[int]]
    bias_entry: Final[int]

    def __init__(self, irreps_in, irreps_out, bias: bool = True):
        super().__init__()
        self.irreps_in, self.irreps_out = o3.Irreps(irreps_in), o3.Irreps(irreps_out)
        starts = [0]
        for mul, ir in self.irreps_in:
            starts.append(starts[-1] + mul * ir.dim)
        paths, w = [], 0
        for i, (mul_i, ir_i) in enumerate(self.irreps_in):
            for o, (mul_o, ir_o) in enumerate(self.irreps_out):
                if ir_i == ir_o and mul_i > 0 and mul_o > 0:
                    paths.append((o, starts[i], mul_i, w))
                    w += mul_i * mul_o
        paths.sort(key=lambda p: p[0])
        self.n_out = len(self.irreps_out)
        self.out_muls = [mul for mul, _ in self.irreps_out]
        self.out_dims = [ir.dim for _, ir in self.irreps_out]
        self.path_lo = [sum(1 for p in paths if p[0] < o) for o in range(self.n_out)]
        self.path_hi = [sum(1 for p in paths if p[0] <= o) for o in range(self.n_out)]
        self.p_in_start = [p[1] for p in paths]
        self.p_in_mul = [p[2] for p in paths]
        self.p_w_start = [p[3] for p in paths]
        self.weight = nn.Parameter(torch.randn(w))
        with torch.no_grad():
            for o in range(self.n_out):
                fan_in = sum(self.p_in_mul[k] for k in range(self.path_lo[o], self.path_hi[o]))
                for k in range(self.path_lo[o], self.path_hi[o]):
                    self.weight.narrow(0, self.p_w_start[k], self.p_in_mul[k] * self.out_muls[o]).mul_(
                        fan_in ** -0.5)
        scalars = [o for o, (_, ir) in enumerate(self.irreps_out) if ir == o3.Irrep(0, 1)]
        if len(scalars) > 1:
            raise ValueError(f"expected one even-scalar output entry, got {self.irreps_out}")
        self.bias_entry = scalars[0] if bias and scalars else -1
        self.bias: Optional[nn.Parameter] = None
        if self.bias_entry >= 0:
            self.bias = nn.Parameter(torch.zeros(self.out_muls[self.bias_entry]))

    def forward(self, x: Tensor) -> Tensor:
        """Map ``(N, irreps_in.dim)`` to ``(N, irreps_out.dim)``."""
        n = x.shape[0]
        parts: List[Tensor] = []
        for o in range(self.n_out):
            mul_o, d = self.out_muls[o], self.out_dims[o]
            y = x.new_zeros((n, mul_o, d))
            for k in range(self.path_lo[o], self.path_hi[o]):
                mul_i = self.p_in_mul[k]
                xi = x.narrow(1, self.p_in_start[k], mul_i * d).reshape(n, mul_i, d)
                wk = self.weight.narrow(0, self.p_w_start[k], mul_i * mul_o).view(mul_i, mul_o)
                y = y + torch.einsum("nud,uw->nwd", xi, wk)
            y = y.reshape(n, mul_o * d)
            if o == self.bias_entry:
                bias = self.bias
                if bias is not None:
                    y = y + bias
            parts.append(y)
        return torch.cat(parts, dim=1)


class EquivariantLayerNorm(nn.Module):
    """Equivariant layer norm of an irreps feature (section 4.1).

    Per irreps entry: the even scalars are centered over the channels; every
    entry is divided by the root mean square of its component-normalized
    norms over the channels and scaled by a learnable weight per channel; the
    even scalars get a learnable bias.

    Parameters
    ----------
    irreps : e3nn.o3.Irreps or str
        The irreps of the feature.
    eps : float, optional
        Added to the mean square, by default 1e-5.
    """

    starts: Final[List[int]]
    muls: Final[List[int]]
    dims: Final[List[int]]
    scalar: Final[List[bool]]
    w_starts: Final[List[int]]
    b_starts: Final[List[int]]

    def __init__(self, irreps, eps: float = 1e-5):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.eps = float(eps)
        self.starts, self.muls, self.dims, self.scalar = [], [], [], []
        self.w_starts, self.b_starts = [], []
        x = w = b = 0
        for mul, ir in self.irreps:
            is_scalar = ir.l == 0 and ir.p == 1
            self.starts.append(x); self.muls.append(mul); self.dims.append(ir.dim)
            self.scalar.append(is_scalar); self.w_starts.append(w); self.b_starts.append(b)
            x += mul * ir.dim
            w += mul
            b += mul if is_scalar else 0
        self.affine_weight = nn.Parameter(torch.ones(w))
        self.affine_bias = nn.Parameter(torch.zeros(b))

    def forward(self, x: Tensor) -> Tensor:
        """Normalize ``(N, irreps.dim)``."""
        fields: List[Tensor] = []
        for k in range(len(self.muls)):
            mul, d = self.muls[k], self.dims[k]
            field = x.narrow(1, self.starts[k], mul * d).reshape(-1, mul, d)
            if self.scalar[k]:
                field = field - torch.mean(field, dim=1, keepdim=True)
            norm = torch.mean(field.pow(2).mean(-1), dim=1, keepdim=True)
            norm = (norm + self.eps).pow(-0.5)
            norm = norm * self.affine_weight.narrow(0, self.w_starts[k], mul)[None, :]
            field = field * norm.reshape(-1, mul, 1)
            if self.scalar[k]:
                field = field + self.affine_bias.narrow(0, self.b_starts[k], mul).reshape(mul, 1)
            fields.append(field.reshape(-1, mul * d))
        return torch.cat(fields, dim=-1)


class Gate(nn.Module):
    """The gate activation (section 4.1): SiLU on the scalars, sigmoid gates on the other irreps.

    The input is ``[scalars | gates | gated]``; both activations are
    normalized to unit second moment, as in e3nn.

    Parameters
    ----------
    irreps_out : e3nn.o3.Irreps or str
        The irreps after the activation; ``irreps_in`` adds one gate scalar per
        gated channel.
    """

    has_gates: Final[bool]
    n_scalars: Final[int]
    n_gates: Final[int]
    gated_muls: Final[List[int]]
    gated_dims: Final[List[int]]

    def __init__(self, irreps_out):
        super().__init__()
        scalars, gates, gated = _irreps_to_gate(o3.Irreps(irreps_out))
        self.has_gates = gated.num_irreps > 0
        self.irreps_out = (scalars + gated).simplify() if self.has_gates else o3.Irreps(irreps_out)
        self.irreps_in = (scalars + gates + gated).simplify() if self.has_gates else self.irreps_out
        self.n_scalars = scalars.dim if self.has_gates else self.irreps_out.dim
        self.n_gates = gates.dim
        self.gated_muls = [mul for mul, _ in gated]
        self.gated_dims = [ir.dim for _, ir in gated]
        self.silu_scale = _moment_scale(F.silu)
        self.sigmoid_scale = _moment_scale(torch.sigmoid)

    def forward(self, x: Tensor) -> Tensor:
        """Activate ``(N, irreps_in.dim)`` into ``(N, irreps_out.dim)``."""
        if not self.has_gates:
            return F.silu(x) * self.silu_scale
        scalars = F.silu(x.narrow(1, 0, self.n_scalars)) * self.silu_scale
        gates = torch.sigmoid(x.narrow(1, self.n_scalars, self.n_gates)) * self.sigmoid_scale
        parts: List[Tensor] = [scalars]
        start, g = self.n_scalars + self.n_gates, 0
        for mul, d in zip(self.gated_muls, self.gated_dims):
            field = x.narrow(1, start, mul * d).reshape(-1, mul, d)
            field = field * gates.narrow(1, g, mul).unsqueeze(-1)
            parts.append(field.reshape(-1, mul * d))
            start += mul * d
            g += mul
        return torch.cat(parts, dim=1)


class _ChannelDropout(nn.Module):
    """Dropout of whole irreps channels (every component of a type-L vector together)."""

    muls: Final[List[int]]
    dims: Final[List[int]]

    def __init__(self, irreps, p: float):
        super().__init__()
        irreps = o3.Irreps(irreps)
        self.p = float(p)
        self.muls = [mul for mul, _ in irreps]
        self.dims = [ir.dim for _, ir in irreps]

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        n = x.shape[0]
        mask = F.dropout(x.new_ones((n, sum(self.muls))), self.p, True)
        parts: List[Tensor] = []
        start, c = 0, 0
        for mul, d in zip(self.muls, self.dims):
            field = x.narrow(1, start, mul * d).reshape(n, mul, d) * mask.narrow(1, c, mul).unsqueeze(-1)
            parts.append(field.reshape(n, mul * d))
            start += mul * d
            c += mul
        return torch.cat(parts, dim=1)


class _Heads(nn.Module):
    """Split ``(E, H * irreps_head)`` into ``(E, H, irreps_head)`` per head, and back."""

    widths: Final[List[int]]
    num_heads: Final[int]

    def __init__(self, irreps_head, num_heads: int):
        super().__init__()
        self.widths = [mul * ir.dim for mul, ir in o3.Irreps(irreps_head)]
        self.num_heads = num_heads

    def split(self, x: Tensor) -> Tensor:
        n, h = x.shape[0], self.num_heads
        parts: List[Tensor] = []
        start = 0
        for width in self.widths:
            parts.append(x.narrow(1, start, width * h).reshape(n, h, width))
            start += width * h
        return torch.cat(parts, dim=2)

    def merge(self, x: Tensor) -> Tensor:
        n, h = x.shape[0], self.num_heads
        parts: List[Tensor] = []
        start = 0
        for width in self.widths:
            parts.append(x.narrow(2, start, width).reshape(n, h * width))
            start += width
        return torch.cat(parts, dim=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.split(x)


class _GaussianBasis(nn.Module):
    """Gaussians with learnable centers and widths on ``d / cutoff`` (the QM9 basis of the paper).

    Parameters
    ----------
    n_rbf : int
        Number of Gaussians.
    cutoff : float
        Distance scale.
    """

    def __init__(self, n_rbf: int, cutoff: float):
        super().__init__()
        self.n_rbf = n_rbf
        self.cutoff = float(cutoff)
        self.mean = nn.Parameter(torch.empty(1, n_rbf).uniform_(0.0, 1.0))
        self.std = nn.Parameter(torch.empty(1, n_rbf).uniform_(1.0 / n_rbf, 1.0))
        self.weight = nn.Parameter(torch.ones(1, 1))
        self.bias = nn.Parameter(torch.zeros(1, 1))
        self.sqrt_2pi = _SQRT_2PI

    def forward(self, r: Tensor) -> Tensor:
        x = (self.weight * (r / self.cutoff).unsqueeze(-1) + self.bias).expand(-1, self.n_rbf)
        std = self.std.abs() + 1e-5
        return torch.exp(-0.5 * (((x - self.mean) / std) ** 2)) / (self.sqrt_2pi * std)


class _BesselBasis(nn.Module):
    """Bessel functions with learnable frequencies times a polynomial envelope of exponent 5."""

    def __init__(self, n_rbf: int, cutoff: float):
        super().__init__()
        self.rbf = BesselRBF(n_rbf, cutoff, trainable=True)
        self.envelope = PolynomialCutoff(cutoff, p=5)

    def forward(self, r: Tensor) -> Tensor:
        return self.envelope(r).unsqueeze(-1) * self.rbf(r)


def make_radial_basis(name: str, n_rbf: int, cutoff: float) -> nn.Module:
    """The radial basis of the Equiformer models.

    Parameters
    ----------
    name : str
        ``"expnorm"`` (the PhysNet basis of the MD17 models), ``"gaussian"``
        (learnable Gaussians, the QM9 and OC20 models) or ``"bessel"``.
    n_rbf : int
        Number of basis functions.
    cutoff : float
        The cutoff radius.

    Returns
    -------
    torch.nn.Module
        Maps distances ``(E,)`` to ``(E, n_rbf)``.

    Raises
    ------
    ValueError
        For an unknown ``name``.
    """
    if name == "expnorm":
        return ExpNormalSmearing(n_rbf, cutoff, trainable=False)
    if name == "gaussian":
        return _GaussianBasis(n_rbf, cutoff)
    if name == "bessel":
        return _BesselBasis(n_rbf, cutoff)
    raise ValueError(f"radial_basis must be 'expnorm', 'gaussian' or 'bessel', got {name!r}")


class _SeparableTP(nn.Module):
    """A depth-wise tensor product with the edge harmonics followed by a linear map."""

    def __init__(self, irreps_in, irreps_sh, irreps_target, irreps_lin_out,
                 radial: Optional[List[int]]):
        super().__init__()
        self.dtp = depthwise_tensor_product(irreps_in, irreps_sh, irreps_target,
                                            internal_weights=radial is None)
        self.rad: Optional[RadialMLP] = None
        if radial is not None:
            self.rad = RadialMLP(radial + [self.dtp.weight_numel])
        self.lin = IrrepsLinear(self.dtp.irreps_out.simplify(), irreps_lin_out)

    def tp(self, x: Tensor, sh: Tensor, edge_scalars: Tensor) -> Tensor:
        rad = self.rad
        if rad is None:
            return self.dtp(x, sh)
        return self.dtp(x, sh, rad(edge_scalars))

    def forward(self, x: Tensor, sh: Tensor, edge_scalars: Tensor) -> Tensor:
        return self.lin(self.tp(x, sh, edge_scalars))


class _MLPAttentionScores(nn.Module):
    """``w^T LeakyReLU(f^(0))`` per head, with the activation normalized to unit second moment."""

    def __init__(self, num_heads: int, mul_alpha_head: int):
        super().__init__()
        self.alpha_act = SmoothLeakyReLU(0.2)
        self.alpha_scale = _moment_scale(SmoothLeakyReLU(0.2))
        # Glorot-uniform, as in GATv2
        bound = math.sqrt(6.0 / (num_heads + mul_alpha_head))
        self.alpha_dot = nn.Parameter(torch.empty(1, num_heads, mul_alpha_head).uniform_(-bound, bound))

    def forward(self, alpha: Tensor) -> Tensor:
        alpha = self.alpha_act(alpha) * self.alpha_scale
        return torch.einsum("ehk,hk->eh", alpha, self.alpha_dot[0])


class _NonlinearMLPAttention(nn.Module):
    """MLP attention with non-linear messages (eq 3 to 5): the logits and values of every edge."""

    num_heads: Final[int]
    mul_alpha_head: Final[int]

    def __init__(self, pre, irreps_sh, attn_heads, irreps_alpha, radial: List[int],
                 heads: _Heads, num_heads: int, mul_alpha_head: int):
        super().__init__()
        self.num_heads, self.mul_alpha_head = num_heads, mul_alpha_head
        gate = Gate(pre)
        self.sep_act = _SeparableTP(pre, irreps_sh, pre, gate.irreps_in, radial)
        self.gate = gate
        self.sep_alpha = IrrepsLinear(self.sep_act.dtp.irreps_out.simplify(), irreps_alpha)
        self.sep_value = _SeparableTP(pre, irreps_sh, attn_heads, attn_heads, None)
        self.scores = _MLPAttentionScores(num_heads, mul_alpha_head)
        self.heads = heads

    def forward(self, x: Tensor, msg: Tensor, dst: Tensor, sh: Tensor,
                edge_scalars: Tensor) -> Tuple[Tensor, Tensor]:
        f = self.sep_act.tp(msg, sh, edge_scalars)
        alpha = self.sep_alpha(f).reshape(f.shape[0], self.num_heads, self.mul_alpha_head)
        value = self.gate(self.sep_act.lin(f))
        value = self.heads.split(self.sep_value(value, sh, edge_scalars))
        return self.scores(alpha), value


class _LinearMLPAttention(nn.Module):
    """MLP attention with linear messages: one separable tensor product gives logits and values."""

    mul_alpha_head: Final[int]

    def __init__(self, pre, irreps_sh, attn_heads, irreps_alpha, irreps_head, radial: List[int],
                 num_heads: int, mul_alpha_head: int):
        super().__init__()
        self.mul_alpha_head = mul_alpha_head
        attn_all = (irreps_alpha + attn_heads).simplify()
        self.sep = _SeparableTP(pre, irreps_sh, attn_all, attn_all, radial)
        self.heads_all = _Heads((o3.Irreps(f"{mul_alpha_head}x0e") + irreps_head).simplify(),
                                num_heads)
        self.scores = _MLPAttentionScores(num_heads, mul_alpha_head)

    def forward(self, x: Tensor, msg: Tensor, dst: Tensor, sh: Tensor,
                edge_scalars: Tensor) -> Tuple[Tensor, Tensor]:
        f = self.heads_all.split(self.sep(msg, sh, edge_scalars))
        alpha = f.narrow(2, 0, self.mul_alpha_head)
        value = f.narrow(2, self.mul_alpha_head, f.shape[2] - self.mul_alpha_head)
        return self.scores(alpha), value


class _DotProductAttention(nn.Module):
    """Scaled dot-product attention with linear messages (appendix C.3)."""

    num_heads: Final[int]

    def __init__(self, irreps_in, pre, irreps_sh, attn_heads, irreps_head, radial: List[int],
                 heads: _Heads, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.query = IrrepsLinear(irreps_in, attn_heads)
        kv_heads = _sort(irreps_head * (2 * num_heads), even_first=False)
        self.key_value = _SeparableTP(pre, irreps_sh, kv_heads, kv_heads, radial)
        self.heads = heads
        self.heads_kv = _Heads(irreps_head, 2 * num_heads)
        # 1 / sqrt(number of irreps of a head x dimension of the irrep)
        scale = [1.0 / math.sqrt(irreps_head.num_irreps * ir.dim)
                 for mul, ir in irreps_head for _ in range(mul * ir.dim)]
        self.register_buffer("q_scale", torch.tensor(scale), persistent=False)

    def forward(self, x: Tensor, msg: Tensor, dst: Tensor, sh: Tensor,
                edge_scalars: Tensor) -> Tuple[Tensor, Tensor]:
        q = self.heads.split(self.query(x)) * self.q_scale
        kv = self.heads_kv.split(self.key_value(msg, sh, edge_scalars))
        key = kv.narrow(1, 0, self.num_heads)
        value = kv.narrow(1, self.num_heads, self.num_heads)
        return (q[dst] * key).sum(dim=-1), value


class GraphAttention(nn.Module):
    """Equivariant graph attention (section 4.2), or the dot-product attention of appendix C.3.

    Parameters
    ----------
    irreps_in, irreps_sh, irreps_out : e3nn.o3.Irreps or str
        Node, spherical-harmonic and output irreps.
    radial : list of int
        Widths of the radial function, ``[n_rbf, hidden...]``.
    irreps_head : e3nn.o3.Irreps or str
        The value irreps of one head.
    num_heads : int
        Number of heads.
    irreps_pre_attn : e3nn.o3.Irreps or str, optional
        Irreps of the merged edge feature, by default ``irreps_in``.
    attention : str, optional
        ``"mlp"`` (default) or ``"dot_product"``.
    nonlinear_message : bool, optional
        Non-linear values (MLP attention only), by default True.
    rescale_degree : bool, optional
        Multiply the aggregated messages by the number of neighbors.
    alpha_drop, proj_drop : float, optional
        Dropout of the attention weights and of the output channels.
    avg_degree : float, optional
        The degree normalization of ``rescale_degree`` with dot-product attention.

    Attributes
    ----------
    core : torch.nn.Module
        Computes the logits ``(E, H)`` and values ``(E, H, irreps_head.dim)`` of every edge.
    """

    rescale_degree: Final[bool]
    degree_divisor: Final[float]

    def __init__(self, irreps_in, irreps_sh, irreps_out, radial: List[int], irreps_head,
                 num_heads: int, irreps_pre_attn=None, attention: str = "mlp",
                 nonlinear_message: bool = True, rescale_degree: bool = False,
                 alpha_drop: float = 0.0, proj_drop: float = 0.0,
                 avg_degree: float = _AVG_DEGREE):
        super().__init__()
        if attention not in ("mlp", "dot_product"):
            raise ValueError(f"attention must be 'mlp' or 'dot_product', got {attention!r}")
        irreps_in, irreps_head = o3.Irreps(irreps_in), o3.Irreps(irreps_head)
        pre = irreps_in if irreps_pre_attn is None else o3.Irreps(irreps_pre_attn)
        mlp = attention == "mlp"
        self.rescale_degree = bool(rescale_degree)
        self.degree_divisor = 1.0 if mlp else float(avg_degree)
        attn_heads = _sort(irreps_head * num_heads, even_first=mlp)
        self.merge_src = IrrepsLinear(irreps_in, pre, bias=True)
        self.merge_dst = IrrepsLinear(irreps_in, pre, bias=False)
        self.heads = _Heads(irreps_head, num_heads)
        if mlp:
            mul_alpha = sum(mul for mul, ir in attn_heads if ir == o3.Irrep(0, 1))
            irreps_alpha = o3.Irreps(f"{mul_alpha}x0e")
            if nonlinear_message:
                self.core = _NonlinearMLPAttention(pre, irreps_sh, attn_heads, irreps_alpha, radial,
                                                   self.heads, num_heads, mul_alpha // num_heads)
            else:
                self.core = _LinearMLPAttention(pre, irreps_sh, attn_heads, irreps_alpha,
                                                irreps_head, radial, num_heads,
                                                mul_alpha // num_heads)
        else:
            self.core = _DotProductAttention(irreps_in, pre, irreps_sh, attn_heads, irreps_head,
                                             radial, self.heads, num_heads)
        self.alpha_dropout = nn.Dropout(alpha_drop) if alpha_drop > 0 else nn.Identity()
        self.proj = IrrepsLinear(attn_heads, irreps_out)
        self.proj_drop = _ChannelDropout(irreps_in, proj_drop)

    def forward(self, x: Tensor, src: Tensor, dst: Tensor, sh: Tensor, edge_scalars: Tensor,
                envelope: Optional[Tensor]) -> Tensor:
        """Attend over the incoming edges of every atom.

        Parameters
        ----------
        x : Tensor
            Node features ``(N, irreps_in.dim)``.
        src, dst : Tensor
            Sending (neighbor ``j``) and receiving (center ``i``) atom of every edge.
        sh : Tensor
            Spherical harmonics of ``r_j - r_i``, ``(E, irreps_sh.dim)``.
        edge_scalars : Tensor
            Radial basis of the edge lengths ``(E, n_rbf)``.
        envelope : Tensor, optional
            Smooth-cutoff weight per edge.

        Returns
        -------
        Tensor
            ``(N, irreps_out.dim)``.
        """
        n = x.shape[0]
        msg = self.merge_src(x)[src] + self.merge_dst(x)[dst]
        logits, value = self.core(x, msg, dst, sh, edge_scalars)
        attn = self.alpha_dropout(segment_softmax(logits, dst, n, envelope))
        out = self.heads.merge(scatter_sum(value * attn.unsqueeze(-1), dst, n))
        if self.rescale_degree:
            degree = scatter_sum(torch.ones_like(dst, dtype=out.dtype), dst, n).unsqueeze(-1)
            out = out * degree / self.degree_divisor
        return self.proj_drop(self.proj(out))



class FeedForward(nn.Module):
    """Linear, gate, linear (the feed-forward network of section 4.3)."""

    def __init__(self, irreps_in, irreps_mid, irreps_out, proj_drop: float = 0.0):
        super().__init__()
        self.gate = Gate(irreps_mid)
        self.lin_1 = IrrepsLinear(irreps_in, self.gate.irreps_in)
        self.lin_2 = IrrepsLinear(self.gate.irreps_out, irreps_out)
        self.proj_drop = _ChannelDropout(irreps_out, proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj_drop(self.lin_2(self.gate(self.lin_1(x))))


class TransformerBlock(nn.Module):
    """Pre-norm block: norm, graph attention, residual; norm, feed-forward, residual."""

    def __init__(self, irreps_in, irreps_sh, irreps_out, radial: List[int], irreps_head,
                 num_heads: int, irreps_pre_attn, irreps_mlp_mid, attention: str,
                 nonlinear_message: bool, rescale_degree: bool, alpha_drop: float,
                 proj_drop: float, drop_path_rate: float, avg_degree: float):
        super().__init__()
        irreps_in, irreps_out = o3.Irreps(irreps_in), o3.Irreps(irreps_out)
        self.norm_1 = EquivariantLayerNorm(irreps_in)
        self.ga = GraphAttention(irreps_in, irreps_sh, irreps_in, radial, irreps_head, num_heads,
                                 irreps_pre_attn, attention, nonlinear_message, rescale_degree,
                                 alpha_drop, proj_drop, avg_degree)
        self.drop_path = GraphDropPath(drop_path_rate)
        self.norm_2 = EquivariantLayerNorm(irreps_in)
        self.ffn = FeedForward(irreps_in, irreps_mlp_mid, irreps_out, proj_drop)
        self.ffn_shortcut: Optional[IrrepsLinear] = None
        if irreps_in != irreps_out:
            self.ffn_shortcut = IrrepsLinear(irreps_in, irreps_out)

    def forward(self, x: Tensor, src: Tensor, dst: Tensor, sh: Tensor, edge_scalars: Tensor,
                envelope: Optional[Tensor], batch: Tensor, num_graphs: int) -> Tensor:
        h = self.ga(self.norm_1(x), src, dst, sh, edge_scalars, envelope)
        x = x + self.drop_path(h, batch, num_graphs)
        h = self.ffn(self.norm_2(x))
        shortcut = self.ffn_shortcut
        if shortcut is not None:
            x = shortcut(x)
        return x + self.drop_path(h, batch, num_graphs)


class EdgeDegreeEmbedding(nn.Module):
    """The degree embedding of section 4.3: a DTP of a constant feature summed over the neighbors."""

    def __init__(self, irreps_node, irreps_sh, radial: List[int], avg_degree: float):
        super().__init__()
        self.exp = IrrepsLinear("1x0e", irreps_node)
        self.dw = depthwise_tensor_product(irreps_node, irreps_sh, irreps_node, internal_weights=False)
        self.rad = RadialMLP(radial + [self.dw.weight_numel])
        self.proj = IrrepsLinear(self.dw.irreps_out.simplify(), irreps_node)
        self.avg_degree = float(avg_degree)

    def forward(self, num_nodes: int, sh: Tensor, edge_scalars: Tensor, dst: Tensor,
                envelope: Optional[Tensor]) -> Tensor:
        const = self.exp(sh.new_ones((1, 1))).expand(sh.shape[0], -1)
        edge = self.proj(self.dw(const, sh, self.rad(edge_scalars)))
        if envelope is not None:
            edge = edge * envelope.unsqueeze(-1)
        return scatter_sum(edge, dst, num_nodes) / math.sqrt(self.avg_degree)


class _AtomEmbedding(nn.Module):
    """Linear embedding of the one-hot atomic number into the even scalars of the node irreps."""

    n_scalars: Final[int]
    n_rest: Final[int]

    def __init__(self, n_elements: int, irreps_node):
        super().__init__()
        irreps_node = o3.Irreps(irreps_node)
        if irreps_node[0][1] != o3.Irrep(0, 1):
            raise ValueError(f"the node irreps must start with even scalars, got {irreps_node}")
        self.n_scalars = irreps_node[0].mul
        self.n_rest = irreps_node.dim - self.n_scalars
        self.weight = nn.Parameter(torch.randn(n_elements, self.n_scalars))
        self.bias = nn.Parameter(torch.zeros(self.n_scalars))

    def forward(self, atomic_numbers: Tensor) -> Tensor:
        scalars = self.weight[atomic_numbers] + self.bias
        return torch.cat([scalars, scalars.new_zeros((scalars.shape[0], self.n_rest))], dim=1)


@register_model("equiformer")
class Equiformer(TransformerPotential):
    """Equiformer, the equivariant graph attention transformer.

    See the module docstring for the architecture. The defaults are the MD17
    model of the paper with ``l_max = 2`` (Table 11, 3.5 M parameters): six
    blocks of irreps ``128x0e+64x1e+32x2e``, four heads of
    ``32x0e+16x1e+8x2e``, a feed-forward width ``384x0e+192x1e+96x2e``, 32
    exponential-normal radial functions at a 5 Angstrom cutoff, MLP attention
    with non-linear messages. :func:`paper_irreps` gives the irreps of the other
    architectures of the paper.

    Parameters
    ----------
    irreps_node_embedding : str, optional
        Irreps of the node features (``d_embed``); starts with the even scalars.
    n_interactions : int, optional
        Number of Transformer blocks, by default 6.
    cutoff : float, optional
        Neighbor-list radius (Angstrom), by default 5.0.
    n_rbf : int, optional
        Number of radial basis functions, by default 32.
    radial_basis : str, optional
        ``"expnorm"`` (default), ``"gaussian"`` or ``"bessel"``.
    radial_hidden : list of int, optional
        Hidden widths of the radial functions, by default ``[64, 64]``.
    irreps_sh : str, optional
        Spherical-harmonic irreps; by default every degree of the embedding,
        with the parity convention of the embedding (all even for SE(3)).
    irreps_feature : str, optional
        Irreps after the last block, by default ``"512x0e"``.
    irreps_head : str, optional
        Value irreps of one attention head, by default ``"32x0e+16x1e+8x2e"``.
    num_heads : int, optional
        Number of attention heads, by default 4.
    irreps_pre_attn : str, optional
        Irreps of the merged edge feature, by default the embedding irreps.
    irreps_mlp_mid : str, optional
        Hidden irreps of the feed-forward networks.
    attention : str, optional
        ``"mlp"`` (default) or ``"dot_product"``.
    nonlinear_message : bool, optional
        Non-linear message passing, by default True.
    rescale_degree : bool, optional
        Multiply the aggregated messages by the number of neighbors, by default False.
    alpha_drop, proj_drop, out_drop, drop_path_rate : float, optional
        Dropout of the attention weights, of the block outputs, of the final
        features, and the stochastic depth rate; all 0 by default.
    avg_degree : float, optional
        Average number of neighbors (normalizes the degree embedding).
    avg_num_nodes : float, optional
        Average number of atoms (normalizes the energy).
    smooth_cutoff : bool, optional
        Make the energy continuous at the cutoff, by default False (the paper's model).
    n_elements : int, optional
        Size of the atom-type embedding, indexed by ``Z``, by default 100.
    energy_shift, energy_scale : float, optional
        Per-atom energy standardization, by default 0 and 1.
    species : list of int, optional
        Only used to interpret ``atomic_energies``.
    atomic_energies : array-like, optional
        Per-species reference energies loaded into ``atom_ref``.
    """

    head_modules = ("head", "energy_scale", "energy_shift", "atom_ref")
    exact_float64 = True
    smooth: Final[bool]

    def __init__(self, irreps_node_embedding: str = "128x0e+64x1e+32x2e", n_interactions: int = 6,
                 cutoff: float = 5.0, n_rbf: int = 32, radial_basis: str = "expnorm",
                 radial_hidden: Optional[List[int]] = None, irreps_sh: Optional[str] = None,
                 irreps_feature: str = "512x0e", irreps_head: str = "32x0e+16x1e+8x2e",
                 num_heads: int = 4, irreps_pre_attn: Optional[str] = None,
                 irreps_mlp_mid: str = "384x0e+192x1e+96x2e", attention: str = "mlp",
                 nonlinear_message: bool = True, rescale_degree: bool = False,
                 alpha_drop: float = 0.0, proj_drop: float = 0.0, out_drop: float = 0.0,
                 drop_path_rate: float = 0.0, avg_degree: float = _AVG_DEGREE,
                 avg_num_nodes: float = _AVG_NUM_NODES, smooth_cutoff: bool = False,
                 n_elements: int = 100, energy_shift: float = 0.0, energy_scale: float = 1.0,
                 species=None, atomic_energies=None):
        super().__init__()
        self.cutoff = float(cutoff)
        if species is not None:
            self.species = [int(z) for z in species]
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.irreps_feature = o3.Irreps(irreps_feature)
        if irreps_sh is None:
            lmax = self.irreps_node_embedding.lmax
            if all(ir.p == 1 for _, ir in self.irreps_node_embedding):
                irreps_sh = "+".join(f"1x{l}e" for l in range(lmax + 1))
            else:
                irreps_sh = str(o3.Irreps.spherical_harmonics(lmax))
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.n_interactions = n_interactions
        self.avg_num_nodes = float(avg_num_nodes)
        self.smooth = bool(smooth_cutoff)
        self.node_feature_dim = sum(mul for mul, ir in self.irreps_feature if ir == o3.Irrep(0, 1))
        radial = [n_rbf] + list(radial_hidden if radial_hidden is not None else [64, 64])

        self.sh = o3.SphericalHarmonics(self.irreps_sh, normalize=True, normalization="component")
        self.rbf = make_radial_basis(radial_basis, n_rbf, cutoff)
        self.envelope = PolynomialCutoff(cutoff, p=6)
        self.atom_embed = _AtomEmbedding(n_elements, self.irreps_node_embedding)
        self.edge_deg_embed = EdgeDegreeEmbedding(self.irreps_node_embedding, self.irreps_sh,
                                                  radial, avg_degree)
        self.blocks = nn.ModuleList([
            TransformerBlock(self.irreps_node_embedding, self.irreps_sh,
                             self.irreps_feature if k == n_interactions - 1
                             else self.irreps_node_embedding,
                             radial, irreps_head, num_heads, irreps_pre_attn, irreps_mlp_mid,
                             attention, nonlinear_message, rescale_degree, alpha_drop,
                             proj_drop, drop_path_rate, avg_degree)
            for k in range(n_interactions)])
        self.norm = EquivariantLayerNorm(self.irreps_feature)
        self.out_dropout = _ChannelDropout(self.irreps_feature, out_drop)
        gate = Gate(self.irreps_feature)
        self.head = nn.ModuleList([IrrepsLinear(self.irreps_feature, gate.irreps_in), gate,
                                   IrrepsLinear(gate.irreps_out, "1x0e")])
        self._init_energy_terms(n_elements, energy_scale, energy_shift, species, atomic_energies)

    def _evaluate(self, atomic_numbers: Tensor, edge_index: Tensor, edge_vec: Tensor,
                  batch: Tensor, num_graphs: int) -> Tuple[Tensor, Tensor]:
        """The final (normalized) features and the per-atom readout."""
        n = atomic_numbers.shape[0]
        src, dst = edge_index[0], edge_index[1]
        r = torch.linalg.norm(edge_vec, dim=-1)
        sh = self.sh(-edge_vec)
        edge_scalars = self.rbf(r)
        envelope: Optional[Tensor] = None
        if self.smooth:
            envelope = self.envelope(r)
        x = self.atom_embed(atomic_numbers) + self.edge_deg_embed(n, sh, edge_scalars, dst, envelope)
        for block in self.blocks:
            x = block(x, src, dst, sh, edge_scalars, envelope, batch, num_graphs)
        x = self.out_dropout(self.norm(x))
        y = x
        for layer in self.head:
            y = layer(y)
        return x, y.squeeze(-1) / math.sqrt(self.avg_num_nodes)

    def _energy(self, y: Tensor, atomic_numbers: Tensor) -> Tensor:
        return (self.energy_scale * y + self.energy_shift
                + self.atom_ref(atomic_numbers).squeeze(-1))

    @torch.jit.export
    def node_features_energy(self, atomic_numbers: Tensor, edge_index: Tensor,
                             edge_vec: Tensor) -> Tuple[Tensor, Tensor]:
        """TorchScript-compatible core: scalar features and per-atom energies.

        Parameters
        ----------
        atomic_numbers : Tensor
            Per-atom atomic numbers ``(N,)``.
        edge_index : Tensor
            Edge index ``(2, E)``; row 0 the neighbor ``j``, row 1 the center ``i``.
        edge_vec : Tensor
            Edge vectors ``r_i - r_j`` (with any periodic image shift), ``(E, 3)``.

        Returns
        -------
        tuple of Tensor
            The even scalars of the final features ``(N, node_feature_dim)``
            and the per-atom energies ``(N,)``.
        """
        batch = torch.zeros_like(atomic_numbers)
        x, y = self._evaluate(atomic_numbers, edge_index, edge_vec, batch, 1)
        return x.narrow(1, 0, self.node_feature_dim), self._energy(y, atomic_numbers)

    @torch.jit.export
    def node_energy(self, atomic_numbers: Tensor, edge_index: Tensor, edge_vec: Tensor) -> Tensor:
        """Per-atom energy ``(N,)`` (the LAMMPS / TorchScript entry point)."""
        return self.node_features_energy(atomic_numbers, edge_index, edge_vec)[1]

    @torch.jit.ignore
    def forward(self, data: AtomicGraph) -> Dict[str, Tensor]:
        """Energies of a batched graph.

        Parameters
        ----------
        data : AtomicGraph
            Batched atomic graph.

        Returns
        -------
        dict of str to Tensor
            ``"node_energy"`` ``(N,)``, ``"energy"`` ``(B,)`` and
            ``"node_features"`` (the even scalars of the final features).
        """
        x, y = self._evaluate(data.atomic_numbers, data.edge_index, data.edge_vectors(),
                              data.batch, data.num_graphs)
        node_energy = self._energy(y, data.atomic_numbers)
        return {"node_energy": node_energy,
                "energy": self.aggregate_energy(node_energy, data),
                "node_features": x.narrow(1, 0, self.node_feature_dim)}

    @classmethod
    def from_config(cls, cfg) -> "Equiformer":
        """Build an :class:`Equiformer` from a configuration object.

        Core fields: ``cfg.n_interactions``, ``cfg.n_rbf``, ``cfg.cutoff``, and
        ``cfg.n_features``, the number of even scalars of the paper
        architecture chosen by ``extra.l_max`` (1, 2 or 3, default 2) and
        ``extra.e3`` (default false), see :func:`paper_irreps`. Explicit
        irreps in ``extra`` (``irreps_node_embedding``, ``irreps_head``,
        ``num_heads``, ``irreps_mlp_mid``, ``irreps_sh``, ``irreps_feature``,
        ``irreps_pre_attn``) take precedence. The other options are the
        keyword arguments of the class, read from ``cfg.extra``.

        Parameters
        ----------
        cfg : xnn.common.config.schema.ModelConfig
            The core model config.

        Returns
        -------
        Equiformer
            Instantiated model.
        """
        from xnn.common.config.coerce import coerce_per_species, coerce_species

        extra = dict(cfg.extra or {})
        species = (coerce_species(extra.get("species"))
                   if extra.get("species") is not None else None)
        preset = paper_irreps(int(extra.get("l_max", 2)), bool(extra.get("e3", False)),
                              int(cfg.n_features))
        kwargs = {k: extra[k] if k in extra else preset[k] for k in preset}
        for key in ("irreps_sh", "irreps_feature", "irreps_pre_attn", "radial_hidden",
                    "radial_basis", "attention"):
            if extra.get(key) is not None:
                kwargs[key] = extra[key]
        for key, kind in (("nonlinear_message", bool), ("rescale_degree", bool),
                          ("smooth_cutoff", bool), ("alpha_drop", float), ("proj_drop", float),
                          ("out_drop", float), ("drop_path_rate", float), ("avg_degree", float),
                          ("avg_num_nodes", float), ("n_elements", int), ("energy_shift", float),
                          ("energy_scale", float)):
            if key in extra:
                kwargs[key] = kind(extra[key])
        kwargs["num_heads"] = int(kwargs["num_heads"])
        if kwargs.get("radial_basis") == "exp":     # the reference code's name of the PhysNet basis
            kwargs["radial_basis"] = "expnorm"
        return cls(n_interactions=cfg.n_interactions, cutoff=cfg.cutoff, n_rbf=cfg.n_rbf,
                   species=species,
                   atomic_energies=coerce_per_species(extra.get("atomic_energies"),
                                                      species or [], "atomic_energies"),
                   **kwargs)
