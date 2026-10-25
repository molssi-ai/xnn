"""EquiformerV2, the equivariant Transformer for higher-degree representations (Liao et al., ICLR 2024).

An implementation of

* Y.-L. Liao, B. Wood, A. Das, T. Smidt, "EquiformerV2: Improved Equivariant
  Transformer for Scaling to Higher-Degree Representations", ICLR 2024
  (arXiv:2306.12059),

written from the paper and consistent with the conventions of the reference code
(fairchem-core 1.x, ``models/equiformer_v2``: feature layout, normalizations,
initialization, parameter names) without copying it.

Every atom carries ``C`` channels of a full spherical-harmonic expansion,
``x_i`` of shape ``((L + 1)^2, C)``. The equivariant graph attention of
Equiformer is rebuilt on eSCN convolutions (section 3.1): the features of the
two endpoints of an edge are concatenated and rotated into a frame whose y axis
is the edge direction, where the tensor product with the edge harmonics reduces
to an SO(2) linear map over the orders ``|m| <= m_max``; the result is rotated
back. On top of that (sections 3.2 to 3.4):

* attention re-normalization: a layer norm on the scalars before the MLP attention;
* separable S2 activation: SiLU on scalars, a SiLU on sphere samples for the
  higher degrees (also in the feed-forward networks, as a grid MLP);
* separable layer norm: degree 0 and the degrees above 0 normalized separately.

The degree embedding rotates the m = 0 output of a radial function back from the
edge frames and sums it over the neighbors (divided by ``avg_degree``); the energy
is a feed-forward network of the final features summed over the atoms and divided
by ``avg_num_nodes``. With ``direct_forces`` a final attention block predicts the
forces from its degree-1 output (the OC20 S2EF models).

Conventions worth knowing:

* The edge frames rotate ``r_j - r_i`` (center to neighbor), the negative of the
  xnn edge vector, onto the y axis (the e3nn polar axis). The reference picks a
  random roll about the edge for every evaluation; here the frame is the minimal
  rotation, deterministic and differentiable. The model depends on the roll only
  through the sampling of the sphere in the S2 activation.
* The reference detaches the rotation matrices from the positions, which leaves
  the gradient of the energy incomplete; here the energy is differentiated through
  them, so forces and stress from autograd are the exact derivatives.
* ``smooth_cutoff=True`` makes the energy continuous at the cutoff (not part of
  the paper), as in :class:`~xnn.trfnn.models.equiformer.Equiformer`. The
  reference's truncation to the nearest ``max_neighbors`` atoms is not supported
  (it makes the energy discontinuous); use the radius instead.
"""
# NOTE: no `from __future__ import annotations` here -- PEP 563 stringifies
# the annotations that TorchScript needs to resolve.
import math
from functools import lru_cache
from typing import Dict, Final, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from xnn.common.data import AtomicGraph
from xnn.common.featurizers import GaussianRBF, PolynomialCutoff
from xnn.common.models.constants import register_constant
from xnn.common.models.ops import scatter_sum
from xnn.common.models.registry import register_model

from .base import TransformerPotential
from .blocks import GraphDropPath, RadialMLP, SmoothLeakyReLU, segment_softmax

# QM9 statistics at a 5 Angstrom cutoff
_AVG_NUM_NODES = 18.03065905448718
_AVG_DEGREE = 15.57930850982666
_ACTIVATIONS = ("sep_s2", "s2", "gate")
_NORMS = ("layer_norm_sh", "layer_norm", "rms_norm_sh")


def _orders(lmax: int, mmax: int) -> List[int]:
    """``min(l, mmax)`` for every degree: the orders kept in the edge frame."""
    return [min(l, mmax) for l in range(lmax + 1)]


def reduced_index(lmax: int, mmax: int) -> List[int]:
    """Positions of the coefficients with ``|m| <= mmax`` in the full ``(lmax + 1)^2`` layout."""
    return [l * l + l + m for l, mm in enumerate(_orders(lmax, mmax)) for m in range(-mm, mm + 1)]


def m_primary_order(lmax: int, mmax: int) -> List[int]:
    """The reduced coefficients reordered by order: ``m = 0`` for every degree, then ``+m`` and ``-m``."""
    offsets, start = [], 0
    for mm in _orders(lmax, mmax):
        offsets.append(start + mm)          # position of m = 0 in each degree block
        start += 2 * mm + 1
    order = [offsets[l] for l in range(lmax + 1)]
    for m in range(1, mmax + 1):
        order += [offsets[l] + m for l in range(m, lmax + 1)]
        order += [offsets[l] - m for l in range(m, lmax + 1)]
    return order


@lru_cache(maxsize=None)
def _s2_grid_float64(lmax: int, mmax: int, res_beta: int, res_alpha: int) -> Tuple[Tensor, Tensor]:
    """The matrices from the reduced coefficients to sphere samples and back, float64 ``(P, n)``."""
    from e3nn.o3 import FromS2Grid, ToS2Grid
    to_grid = ToS2Grid(lmax, (res_beta, res_alpha), normalization="component", dtype=torch.float64)
    to_mat = torch.einsum("mbi,am->bai", to_grid.shb, to_grid.sha)
    from_grid = FromS2Grid((res_beta, res_alpha), lmax, normalization="component",
                           dtype=torch.float64)
    from_mat = torch.einsum("am,mbi->bai", from_grid.sha, from_grid.shb)
    for l in range(mmax + 1, lmax + 1):
        # the degrees cut to |m| <= mmax keep the variance of the full degree
        scale = math.sqrt((2 * l + 1) / (2 * mmax + 1))
        to_mat[:, :, l * l:(l + 1) * (l + 1)] *= scale
        from_mat[:, :, l * l:(l + 1) * (l + 1)] *= scale
    idx = torch.tensor(reduced_index(lmax, mmax))
    to_mat = to_mat[:, :, idx].reshape(res_beta * res_alpha, -1)
    from_mat = from_mat[:, :, idx].reshape(res_beta * res_alpha, -1)
    return to_mat, from_mat


def grid_resolution(lmax: int, mmax: int, resolution: Optional[int]) -> Tuple[int, int]:
    """Samples in latitude and longitude: ``resolution`` for both, or the reference's minimum."""
    if resolution is not None:
        return int(resolution), int(resolution)
    return 2 * (lmax + 1), (2 * (mmax + 1) + 1 if lmax == mmax else 2 * mmax + 1)


class S2Grid(nn.Module):
    """Projection of ``(..., n, C)`` coefficients onto sphere samples ``(..., P, C)`` and back.

    Parameters
    ----------
    lmax, mmax : int
        The degrees and the orders kept (``mmax < lmax`` for edge-frame features).
    resolution : int, optional
        Samples in each angle; by default the smallest exact resolution.
    """

    def __init__(self, lmax: int, mmax: int, resolution: Optional[int] = None):
        super().__init__()
        to_mat, from_mat = _s2_grid_float64(lmax, mmax, *grid_resolution(lmax, mmax, resolution))
        register_constant(self, "to_grid_mat", to_mat, persistent=False)
        register_constant(self, "from_grid_mat", from_mat, persistent=False)

    def to_grid(self, x: Tensor) -> Tensor:
        return torch.matmul(self.to_grid_mat, x)

    def from_grid(self, x: Tensor) -> Tensor:
        return torch.matmul(self.from_grid_mat.transpose(0, 1), x)

    def forward(self, x: Tensor) -> Tensor:
        """SiLU applied pointwise on the sphere."""
        return self.from_grid(F.silu(self.to_grid(x)))


def edge_rotation(vec: Tensor) -> Tensor:
    """The minimal rotation taking each edge direction onto the y axis, ``(E, 3, 3)``.

    ``R = c I + [v]_x + v v^T / (1 + c)`` with ``c = n_y`` and ``v = n x e_y``,
    smooth everywhere except at ``n = -e_y``, where the rotation by pi about x is
    used (the model is invariant to the remaining roll up to the sphere sampling).

    Parameters
    ----------
    vec : Tensor
        Edge vectors ``(E, 3)``.

    Returns
    -------
    Tensor
        Rotation matrices with ``R n = e_y``.
    """
    n = vec / torch.linalg.norm(vec, dim=-1, keepdim=True)
    x, y, z = n[:, 0], n[:, 1], n[:, 2]
    s2 = x * x + z * z
    tiny = 1e-20
    pole = (y < 0) & (s2 < tiny)
    s2_safe = torch.where(s2 < tiny, torch.ones_like(s2), s2)
    # 1 / (1 + c), evaluated without cancellation on both hemispheres
    k = torch.where(y >= 0, 1.0 / (1.0 + y.clamp(min=0.0)), (1.0 - y) / s2_safe)
    rows = torch.stack([y + z * z * k, -x, -x * z * k,
                        x, y, z,
                        -x * z * k, -z, y + x * x * k], dim=-1).reshape(-1, 3, 3)
    flip = torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
                        dtype=vec.dtype, device=vec.device)
    return torch.where(pole[:, None, None], flip.expand_as(rows), rows)


class EdgeFrames(nn.Module):
    """Wigner-D blocks of the edge rotations, degree by degree, from Clebsch-Gordan recursion.

    ``D^1`` is the rotation in the e3nn basis of degree 1 and
    ``D^l = C_l (D^(l-1) (x) D^1) C_l^T`` with the coupling coefficients of
    ``(l - 1) x 1 -> l``: polynomial in the rotation, so it is differentiable
    everywhere the rotation is.

    Parameters
    ----------
    lmax : int
        Highest degree.
    """

    lmax: Final[int]
    cg_starts: Final[List[int]]

    def __init__(self, lmax: int):
        super().__init__()
        from e3nn import o3
        self.lmax = lmax
        axes = torch.eye(3, dtype=torch.float64)
        basis = o3.spherical_harmonics(1, axes, normalize=True, normalization="norm").T
        register_constant(self, "basis", basis)
        blocks, starts = [], [0, 0, 0]
        for l in range(2, lmax + 1):
            c = o3.wigner_3j(l - 1, 1, l, dtype=torch.float64).permute(2, 0, 1) * math.sqrt(2 * l + 1)
            blocks.append(c.reshape(-1))
            starts.append(starts[-1] + c.numel())
        self.cg_starts = starts
        flat = torch.cat(blocks) if blocks else torch.zeros(0, dtype=torch.float64)
        register_constant(self, "cg", flat)

    def forward(self, vec: Tensor) -> List[Tensor]:
        """The blocks ``D^l(R)`` of shape ``(E, 2l + 1, 2l + 1)``, ``l = 0 .. lmax``, with ``R n = e_y``."""
        rot = edge_rotation(vec)
        d1 = self.basis @ rot @ self.basis.transpose(0, 1)
        out: List[Tensor] = [torch.ones_like(rot[:, :1, :1])]
        if self.lmax >= 1:
            out.append(d1)
        prev = d1
        for l in range(2, self.lmax + 1):
            c = self.cg.narrow(0, self.cg_starts[l], self.cg_starts[l + 1] - self.cg_starts[l])
            c = c.view(2 * l + 1, 2 * l - 1, 3)
            t = torch.einsum("aij,eik->eajk", c, prev)
            t = torch.einsum("eajk,ejm->eakm", t, d1)
            prev = torch.einsum("eakm,bkm->eab", t, c)
            out.append(prev)
        return out


class SO3Linear(nn.Module):
    """Linear map per degree of ``(N, (L + 1)^2, C_in)`` features, with a bias on degree 0.

    Parameters
    ----------
    in_features, out_features : int
        Channels.
    lmax : int
        Highest degree.
    """

    def __init__(self, in_features: int, out_features: int, lmax: int):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        bound = 1.0 / math.sqrt(in_features)
        self.weight = nn.Parameter(torch.empty(lmax + 1, out_features, in_features).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.zeros(out_features))
        degree = [l for l in range(lmax + 1) for _ in range(2 * l + 1)]
        self.register_buffer("expand_index", torch.tensor(degree), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.einsum("bmi,moi->bmo", x, self.weight.index_select(0, self.expand_index))
        return torch.cat([out.narrow(1, 0, 1) + self.bias, out.narrow(1, 1, out.shape[1] - 1)], dim=1)


class _SO2mConvolution(nn.Module):
    """The SO(2) linear map of the ``+m`` and ``-m`` coefficients of one order ``m > 0``."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc = nn.Linear(in_features, 2 * out_features, bias=False)
        with torch.no_grad():
            self.fc.weight.mul_(1.0 / math.sqrt(2.0))


class SO2Convolution(nn.Module):
    """eSCN convolution in the edge frame: one linear map per order ``m``, mixing ``+m`` and ``-m``.

    ``m = 0`` is a plain linear map (with bias); for ``m > 0`` the real and
    imaginary parts ``x_+m``, ``x_-m`` map as ``(W_r x_+m - W_i x_-m, W_r x_-m +
    W_i x_+m)``, which commutes with rotations about the edge. With a radial
    function the inputs are first multiplied by per-edge weights.

    Parameters
    ----------
    in_channels, out_channels : int
        Channels per coefficient.
    lmax, mmax : int
        Degrees and orders of the edge-frame features.
    radial : list of int, optional
        Widths of the radial function ``[n_in, hidden...]``; ``None`` for no radial weights.
    extra_m0 : int, optional
        Extra outputs of the ``m = 0`` map returned on their own (attention scalars, gates).
    """

    lmax: Final[int]
    out_channels: Final[int]
    extra_m0: Final[int]

    def __init__(self, in_channels: int, out_channels: int, lmax: int, mmax: int,
                 radial: Optional[List[int]] = None, extra_m0: int = 0):
        super().__init__()
        self.lmax, self.out_channels, self.extra_m0 = lmax, out_channels, extra_m0
        n0 = (lmax + 1) * in_channels
        self.fc_m0 = nn.Linear(n0, out_channels * (lmax + 1) + extra_m0)
        self.so2_m_conv = nn.ModuleList([
            _SO2mConvolution((lmax - m + 1) * in_channels, (lmax - m + 1) * out_channels)
            for m in range(1, mmax + 1)])
        n_rad = n0 + sum((lmax - m + 1) * in_channels for m in range(1, mmax + 1))
        self.rad_func: Optional[RadialMLP] = None
        if radial is not None:
            self.rad_func = RadialMLP(list(radial) + [n_rad])
        order = m_primary_order(lmax, mmax)
        self.register_buffer("to_m", torch.tensor(order), persistent=False)
        self.register_buffer("from_m", torch.argsort(torch.tensor(order)), persistent=False)

    def forward(self, x: Tensor, x_edge: Tensor) -> Tuple[Tensor, Tensor]:
        """Map edge-frame features ``(E, n, C_in)`` to ``(E, n, C_out)`` and the extra ``m = 0`` outputs."""
        e = x.shape[0]
        x = x.index_select(1, self.to_m)
        w: Optional[Tensor] = None
        if self.rad_func is not None:
            w = self.rad_func(x_edge)
        n0 = self.fc_m0.in_features
        x0 = x.narrow(1, 0, self.lmax + 1).reshape(e, n0)
        if w is not None:
            x0 = x0 * w.narrow(1, 0, n0)
        y0 = self.fc_m0(x0)
        extra = y0.narrow(1, 0, self.extra_m0)
        outs: List[Tensor] = [y0.narrow(1, self.extra_m0, y0.shape[1] - self.extra_m0)
                              .reshape(e, self.lmax + 1, self.out_channels)]
        start, rstart = self.lmax + 1, n0
        for k, conv in enumerate(self.so2_m_conv):
            size = self.lmax - k
            n_in = conv.fc.in_features
            xm = x.narrow(1, start, 2 * size).reshape(e, 2, n_in)
            if w is not None:
                xm = xm * w.narrow(1, rstart, n_in).unsqueeze(1)
            ym = conv.fc(xm)
            half = ym.shape[2] // 2
            yr, yi = ym.narrow(2, 0, half), ym.narrow(2, half, half)
            real = yr.select(1, 0) - yi.select(1, 1)
            imag = yr.select(1, 1) + yi.select(1, 0)
            outs.append(torch.stack((real, imag), dim=1).reshape(e, 2 * size, self.out_channels))
            start += 2 * size
            rstart += n_in
        return torch.cat(outs, dim=1).index_select(1, self.from_m), extra


class _Rotations(nn.Module):
    """Rotation of features into the edge frames (keeping ``|m| <= mmax``) and back."""

    lmax: Final[int]
    orders: Final[List[int]]
    scales: Final[List[float]]

    def __init__(self, lmax: int, mmax: int):
        super().__init__()
        self.lmax = lmax
        self.orders = _orders(lmax, mmax)
        self.scales = [math.sqrt((2 * l + 1) / (2 * mmax + 1)) if l > mmax else 1.0
                       for l in range(lmax + 1)]

    def rotate(self, x: Tensor, wigner: List[Tensor]) -> Tensor:
        """``(E, (L + 1)^2, C)`` to the edge frames, ``(E, n_reduced, C)``."""
        parts: List[Tensor] = []
        for l, d in enumerate(wigner):
            mm = self.orders[l]
            parts.append(torch.bmm(d.narrow(1, l - mm, 2 * mm + 1), x.narrow(1, l * l, 2 * l + 1)))
        return torch.cat(parts, dim=1)

    def rotate_inv(self, x: Tensor, wigner: List[Tensor]) -> Tensor:
        """``(E, n_reduced, C)`` from the edge frames back to ``(E, (L + 1)^2, C)``."""
        parts: List[Tensor] = []
        start = 0
        for l, d in enumerate(wigner):
            mm = self.orders[l]
            block = torch.bmm(d.narrow(1, l - mm, 2 * mm + 1).transpose(1, 2),
                              x.narrow(1, start, 2 * mm + 1))
            parts.append(block * self.scales[l] if self.scales[l] != 1.0 else block)
            start += 2 * mm + 1
        return torch.cat(parts, dim=1)


class GateActivation(nn.Module):
    """SiLU on degree 0, the sigmoid of extra scalars (one per channel and degree) on the others."""

    lmax: Final[int]
    channels: Final[int]

    def __init__(self, lmax: int, mmax: int, channels: int):
        super().__init__()
        self.lmax, self.channels = lmax, channels
        index = [l - 1 for l in range(1, lmax + 1) for _ in range(min(2 * l + 1, 2 * mmax + 1))]
        self.register_buffer("expand_index", torch.tensor(index, dtype=torch.long), persistent=False)

    def forward(self, gates: Tensor, x: Tensor) -> Tensor:
        g = torch.sigmoid(gates).reshape(gates.shape[0], self.lmax, self.channels)
        g = g.index_select(1, self.expand_index)
        return torch.cat([F.silu(x.narrow(1, 0, 1)), x.narrow(1, 1, x.shape[1] - 1) * g], dim=1)


class SeparableS2Activation(nn.Module):
    """SiLU on separate degree-0 scalars, S2 activation (SiLU on the sphere) for the higher degrees."""

    def __init__(self, grid: S2Grid):
        super().__init__()
        self.grid = grid

    def forward(self, scalars: Tensor, x: Tensor) -> Tensor:
        out = self.grid(x)
        return torch.cat([F.silu(scalars).unsqueeze(1), out.narrow(1, 1, out.shape[1] - 1)], dim=1)


class SeparableLayerNorm(nn.Module):
    """Separable layer norm (section 3.4): a layer norm on degree 0, one RMS over all degrees above 0.

    The mean square of the higher degrees weights each degree equally
    (``1 / (2l + 1)`` per component, averaged over the degrees).

    Parameters
    ----------
    lmax : int
        Highest degree.
    channels : int
        Channels.
    eps : float, optional
        By default 1e-5.
    """

    lmax: Final[int]

    def __init__(self, lmax: int, channels: int, eps: float = 1e-5):
        super().__init__()
        self.lmax, self.eps = lmax, float(eps)
        self.norm_l0 = nn.LayerNorm(channels, eps=eps)
        self.affine_weight = nn.Parameter(torch.ones(lmax, channels))
        weights = [1.0 / ((2 * l + 1) * lmax) for l in range(1, lmax + 1) for _ in range(2 * l + 1)]
        register_constant(self, "balance_degree_weight", torch.tensor(weights).reshape(-1, 1),
                          persistent=False)
        degree = [l - 1 for l in range(1, lmax + 1) for _ in range(2 * l + 1)]
        self.register_buffer("expand_index", torch.tensor(degree, dtype=torch.long), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        out0 = self.norm_l0(x.narrow(1, 0, 1))
        if self.lmax == 0:
            return out0
        rest = x.narrow(1, 1, x.shape[1] - 1)
        norm = torch.einsum("nic,ia->nac", rest.pow(2), self.balance_degree_weight)
        norm = (torch.mean(norm, dim=2, keepdim=True) + self.eps).pow(-0.5)
        weight = self.affine_weight.index_select(0, self.expand_index)
        return torch.cat([out0, rest * (norm * weight)], dim=1)


class DegreeLayerNorm(nn.Module):
    """Equivariant layer norm of Equiformer on ``(N, (L + 1)^2, C)``: each degree normalized on its own."""

    lmax: Final[int]

    def __init__(self, lmax: int, channels: int, eps: float = 1e-5):
        super().__init__()
        self.lmax, self.eps = lmax, float(eps)
        self.affine_weight = nn.Parameter(torch.ones(lmax + 1, channels))
        self.affine_bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: Tensor) -> Tensor:
        parts: List[Tensor] = []
        for l in range(self.lmax + 1):
            field = x.narrow(1, l * l, 2 * l + 1)
            if l == 0:
                field = field - torch.mean(field, dim=2, keepdim=True)
            norm = torch.mean(field.pow(2).mean(dim=1, keepdim=True), dim=2, keepdim=True)
            norm = (norm + self.eps).pow(-0.5) * self.affine_weight.narrow(0, l, 1).view(1, 1, -1)
            field = field * norm
            if l == 0:
                field = field + self.affine_bias.view(1, 1, -1)
            parts.append(field)
        return torch.cat(parts, dim=1)


class RMSLayerNorm(nn.Module):
    """RMS norm over all degrees (degree-balanced), with degree 0 centered and given a bias."""

    lmax: Final[int]

    def __init__(self, lmax: int, channels: int, eps: float = 1e-5):
        super().__init__()
        self.lmax, self.eps = lmax, float(eps)
        self.affine_weight = nn.Parameter(torch.ones(lmax + 1, channels))
        self.affine_bias = nn.Parameter(torch.zeros(channels))
        weights = [1.0 / ((2 * l + 1) * (lmax + 1)) for l in range(lmax + 1) for _ in range(2 * l + 1)]
        register_constant(self, "balance_degree_weight", torch.tensor(weights).reshape(-1, 1),
                          persistent=False)
        degree = [l for l in range(lmax + 1) for _ in range(2 * l + 1)]
        self.register_buffer("expand_index", torch.tensor(degree, dtype=torch.long), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        x0 = x.narrow(1, 0, 1)
        x = torch.cat([x0 - x0.mean(dim=2, keepdim=True), x.narrow(1, 1, x.shape[1] - 1)], dim=1)
        norm = torch.einsum("nic,ia->nac", x.pow(2), self.balance_degree_weight)
        norm = (torch.mean(norm, dim=2, keepdim=True) + self.eps).pow(-0.5)
        out = x * (norm * self.affine_weight.index_select(0, self.expand_index).unsqueeze(0))
        return torch.cat([out.narrow(1, 0, 1) + self.affine_bias, out.narrow(1, 1, out.shape[1] - 1)],
                         dim=1)


def make_norm(kind: str, lmax: int, channels: int) -> nn.Module:
    """``"layer_norm_sh"`` (separable, the paper's), ``"layer_norm"`` (per degree) or ``"rms_norm_sh"``."""
    if kind == "layer_norm_sh":
        return SeparableLayerNorm(lmax, channels)
    if kind == "layer_norm":
        return DegreeLayerNorm(lmax, channels)
    if kind == "rms_norm_sh":
        return RMSLayerNorm(lmax, channels)
    raise ValueError(f"norm_type must be one of {_NORMS}, got {kind!r}")


class _EdgeScalars(nn.Module):
    """The invariant edge input of a radial function: the distance basis and, optionally, element embeddings."""

    use_elements: Final[bool]

    def __init__(self, n_elements: int, channels: int, use_elements: bool):
        super().__init__()
        self.use_elements = use_elements
        self.source_embedding: Optional[nn.Embedding] = None
        self.target_embedding: Optional[nn.Embedding] = None
        if use_elements:
            self.source_embedding = nn.Embedding(n_elements, channels)
            self.target_embedding = nn.Embedding(n_elements, channels)
            nn.init.uniform_(self.source_embedding.weight, -0.001, 0.001)
            nn.init.uniform_(self.target_embedding.weight, -0.001, 0.001)

    def forward(self, edge_rbf: Tensor, atomic_numbers: Tensor, src: Tensor, dst: Tensor) -> Tensor:
        if self.source_embedding is not None and self.target_embedding is not None:
            return torch.cat([edge_rbf, self.source_embedding(atomic_numbers[src]),
                              self.target_embedding(atomic_numbers[dst])], dim=1)
        return edge_rbf


class SO2GraphAttention(nn.Module):
    """Equivariant graph attention on eSCN convolutions (section 3.5, Figure 1b).

    Parameters
    ----------
    channels : int
        Node channels ``C``.
    hidden_channels : int
        Channels of the first SO(2) convolution (``d_attn_hidden``).
    num_heads, alpha_channels, value_channels : int
        Heads and per-head widths of the attention scalars and values.
    out_channels : int
        Output channels.
    lmax, mmax : int
        Degrees and the orders kept in the edge frames.
    radial : list of int
        Widths of the radial function ``[n_in, hidden...]``.
    n_elements : int
        Size of the element embeddings of the edges.
    use_atom_edge_embedding : bool
        Embed the elements of the two atoms into the radial input.
    use_m_share_rad : bool
        One radial weight per degree and channel, applied before the rotation.
    activation : str
        ``"sep_s2"`` (separable S2, the paper's), ``"s2"`` or ``"gate"``.
    use_attn_renorm : bool
        Layer norm on the attention scalars (attention re-normalization).
    resolution : int, optional
        Sphere samples of the S2 activation.
    alpha_drop : float, optional
        Dropout of the attention weights.
    """

    num_heads: Final[int]
    alpha_channels: Final[int]
    value_channels: Final[int]
    hidden_channels: Final[int]
    activation: Final[str]
    lmax: Final[int]

    def __init__(self, channels: int, hidden_channels: int, num_heads: int, alpha_channels: int,
                 value_channels: int, out_channels: int, lmax: int, mmax: int, radial: List[int],
                 n_elements: int, use_atom_edge_embedding: bool, use_m_share_rad: bool,
                 activation: str, use_attn_renorm: bool, resolution: Optional[int],
                 alpha_drop: float = 0.0):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {_ACTIVATIONS}, got {activation!r}")
        self.num_heads, self.alpha_channels = num_heads, alpha_channels
        self.value_channels, self.hidden_channels = value_channels, hidden_channels
        self.activation, self.lmax = activation, lmax
        radial = list(radial)
        self.edge_scalars = _EdgeScalars(n_elements, radial[-1], use_atom_edge_embedding)
        if use_atom_edge_embedding:
            radial[0] += 2 * radial[-1]
        extra = num_heads * alpha_channels
        if activation == "gate":
            extra += lmax * hidden_channels
        elif activation == "sep_s2":
            extra += hidden_channels
        self.rad_func: Optional[RadialMLP] = None
        if use_m_share_rad:
            self.rad_func = RadialMLP(radial + [2 * channels * (lmax + 1)])
            degree = [l for l in range(lmax + 1) for _ in range(2 * l + 1)]
            self.register_buffer("expand_index", torch.tensor(degree), persistent=False)
        self.rotations = _Rotations(lmax, mmax)
        self.so2_conv_1 = SO2Convolution(2 * channels, hidden_channels, lmax, mmax,
                                         None if use_m_share_rad else radial, extra)
        self.alpha_norm = nn.LayerNorm(alpha_channels) if use_attn_renorm else nn.Identity()
        self.alpha_act = SmoothLeakyReLU(0.2)
        bound = 1.0 / math.sqrt(alpha_channels)
        self.alpha_dot = nn.Parameter(torch.empty(num_heads, alpha_channels).uniform_(-bound, bound))
        self.alpha_dropout = nn.Dropout(alpha_drop) if alpha_drop > 0 else nn.Identity()
        self.gate_act: Optional[GateActivation] = None
        self.s2_act: Optional[SeparableS2Activation] = None
        self.grid: Optional[S2Grid] = None
        if activation == "gate":
            self.gate_act = GateActivation(lmax, mmax, hidden_channels)
        elif activation == "sep_s2":
            self.s2_act = SeparableS2Activation(S2Grid(lmax, mmax, resolution))
        else:
            self.grid = S2Grid(lmax, mmax, resolution)
        self.so2_conv_2 = SO2Convolution(hidden_channels, num_heads * value_channels, lmax, mmax)
        self.proj = SO3Linear(num_heads * value_channels, out_channels, lmax)

    def forward(self, x: Tensor, atomic_numbers: Tensor, edge_rbf: Tensor, src: Tensor, dst: Tensor,
                wigner: List[Tensor], envelope: Optional[Tensor]) -> Tensor:
        """Attend over the incoming edges of every atom.

        Parameters
        ----------
        x : Tensor
            Node features ``(N, (L + 1)^2, C)``.
        atomic_numbers : Tensor
            ``(N,)``.
        edge_rbf : Tensor
            Distance basis of the edges (with the shared element embeddings, if any).
        src, dst : Tensor
            Sending (neighbor ``j``) and receiving (center ``i``) atom of every edge.
        wigner : list of Tensor
            The Wigner blocks of the edge frames (:class:`EdgeFrames`).
        envelope : Tensor, optional
            Smooth-cutoff weight per edge.

        Returns
        -------
        Tensor
            ``(N, (L + 1)^2, out_channels)``.
        """
        n, e = x.shape[0], src.shape[0]
        x_edge = self.edge_scalars(edge_rbf, atomic_numbers, src, dst)
        msg = torch.cat([x[src], x[dst]], dim=2)
        if self.rad_func is not None:
            w = self.rad_func(x_edge).reshape(e, self.lmax + 1, msg.shape[2]).index_select(1, self.expand_index)
            msg = msg * w
        msg = self.rotations.rotate(msg, wigner)
        msg, extra = self.so2_conv_1(msg, x_edge)
        n_alpha = self.num_heads * self.alpha_channels
        alpha = extra.narrow(1, 0, n_alpha)
        if self.gate_act is not None:
            msg = self.gate_act(extra.narrow(1, n_alpha, extra.shape[1] - n_alpha), msg)
        elif self.s2_act is not None:
            msg = self.s2_act(extra.narrow(1, n_alpha, extra.shape[1] - n_alpha), msg)
        elif self.grid is not None:
            msg = self.grid(msg)
        msg, _ = self.so2_conv_2(msg, x_edge)
        alpha = self.alpha_act(self.alpha_norm(alpha.reshape(e, self.num_heads, self.alpha_channels)))
        logits = torch.einsum("ehk,hk->eh", alpha, self.alpha_dot)
        attn = self.alpha_dropout(segment_softmax(logits, dst, n, envelope))
        msg = msg.reshape(e, msg.shape[1], self.num_heads, self.value_channels) * attn[:, None, :, None]
        msg = self.rotations.rotate_inv(
            msg.reshape(e, msg.shape[1], self.num_heads * self.value_channels), wigner)
        return self.proj(scatter_sum(msg, dst, n))


class FeedForwardNetwork(nn.Module):
    """The feed-forward network (Figure 1d): SO(3) linear, separable S2 activation, SO(3) linear.

    Parameters
    ----------
    channels, hidden_channels, out_channels : int
        Input, hidden and output channels.
    lmax : int
        Highest degree.
    activation : str
        ``"sep_s2"``, ``"s2"`` or ``"gate"``.
    use_grid_mlp : bool
        A two-layer MLP on the sphere samples instead of a single SiLU (the paper's).
    resolution : int, optional
        Sphere samples.
    """

    def __init__(self, channels: int, hidden_channels: int, out_channels: int, lmax: int,
                 activation: str, use_grid_mlp: bool, resolution: Optional[int]):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {_ACTIVATIONS}, got {activation!r}")
        self.so3_linear_1 = SO3Linear(channels, hidden_channels, lmax)
        self.grid: Optional[S2Grid] = None
        self.scalar_mlp: Optional[nn.Sequential] = None
        self.grid_mlp: Optional[nn.Sequential] = None
        self.gating_linear: Optional[nn.Linear] = None
        self.gate_act: Optional[GateActivation] = None
        self.s2_act: Optional[SeparableS2Activation] = None
        if use_grid_mlp:
            self.grid = S2Grid(lmax, lmax, resolution)
            if activation == "sep_s2":
                self.scalar_mlp = nn.Sequential(nn.Linear(channels, hidden_channels), nn.SiLU())
            self.grid_mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels, bias=False), nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels, bias=False), nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels, bias=False))
        elif activation == "gate":
            self.gating_linear = nn.Linear(channels, lmax * hidden_channels)
            self.gate_act = GateActivation(lmax, lmax, hidden_channels)
        elif activation == "sep_s2":
            self.gating_linear = nn.Linear(channels, hidden_channels)
            self.s2_act = SeparableS2Activation(S2Grid(lmax, lmax, resolution))
        else:
            self.grid = S2Grid(lmax, lmax, resolution)
        self.so3_linear_2 = SO3Linear(hidden_channels, out_channels, lmax)

    def forward(self, x: Tensor) -> Tensor:
        x0 = x.select(1, 0)
        h = self.so3_linear_1(x)
        if self.grid is not None and self.grid_mlp is not None:
            h = self.grid.from_grid(self.grid_mlp(self.grid.to_grid(h)))
            if self.scalar_mlp is not None:
                h = torch.cat([self.scalar_mlp(x0).unsqueeze(1), h.narrow(1, 1, h.shape[1] - 1)], dim=1)
        elif self.gating_linear is not None and self.gate_act is not None:
            h = self.gate_act(self.gating_linear(x0), h)
        elif self.gating_linear is not None and self.s2_act is not None:
            h = self.s2_act(self.gating_linear(x0), h)
        elif self.grid is not None:
            h = self.grid(h)
        return self.so3_linear_2(h)


class _EquivariantDropout(nn.Module):
    """Dropout of whole channels of ``(N, (L + 1)^2, C)`` features."""

    def __init__(self, p: float):
        super().__init__()
        self.p = float(p)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        return x * F.dropout(x.new_ones((x.shape[0], 1, x.shape[2])), self.p, True)


class TransformerBlockV2(nn.Module):
    """Pre-norm block: norm, SO(2) graph attention, residual; norm, feed-forward, residual."""

    def __init__(self, channels: int, attn_hidden_channels: int, num_heads: int,
                 attn_alpha_channels: int, attn_value_channels: int, ffn_hidden_channels: int,
                 lmax: int, mmax: int, radial: List[int], n_elements: int,
                 use_atom_edge_embedding: bool, use_m_share_rad: bool, activation: str,
                 use_attn_renorm: bool, use_grid_mlp: bool, norm_type: str,
                 resolution: Optional[int], alpha_drop: float, drop_path_rate: float,
                 proj_drop: float):
        super().__init__()
        self.norm_1 = make_norm(norm_type, lmax, channels)
        self.ga = SO2GraphAttention(channels, attn_hidden_channels, num_heads, attn_alpha_channels,
                                    attn_value_channels, channels, lmax, mmax, radial, n_elements,
                                    use_atom_edge_embedding, use_m_share_rad, activation,
                                    use_attn_renorm, resolution, alpha_drop)
        self.drop_path = GraphDropPath(drop_path_rate)
        self.proj_drop = _EquivariantDropout(proj_drop)
        self.norm_2 = make_norm(norm_type, lmax, channels)
        self.ffn = FeedForwardNetwork(channels, ffn_hidden_channels, channels, lmax, activation,
                                      use_grid_mlp, resolution)

    def forward(self, x: Tensor, atomic_numbers: Tensor, edge_rbf: Tensor, src: Tensor, dst: Tensor,
                wigner: List[Tensor], envelope: Optional[Tensor], batch: Tensor,
                num_graphs: int) -> Tensor:
        h = self.ga(self.norm_1(x), atomic_numbers, edge_rbf, src, dst, wigner, envelope)
        x = x + self.proj_drop(self.drop_path(h, batch, num_graphs))
        h = self.ffn(self.norm_2(x))
        return x + self.proj_drop(self.drop_path(h, batch, num_graphs))


class EdgeDegreeEmbedding(nn.Module):
    """The degree embedding: radial ``m = 0`` coefficients rotated back from the edge frames and summed."""

    lmax: Final[int]
    channels: Final[int]
    scales: Final[List[float]]

    def __init__(self, channels: int, lmax: int, mmax: int, radial: List[int], n_elements: int,
                 use_atom_edge_embedding: bool, avg_degree: float):
        super().__init__()
        self.lmax, self.channels = lmax, channels
        radial = list(radial)
        self.edge_scalars = _EdgeScalars(n_elements, radial[-1], use_atom_edge_embedding)
        if use_atom_edge_embedding:
            radial[0] += 2 * radial[-1]
        self.rad_func = RadialMLP(radial + [(lmax + 1) * channels])
        self.scales = [math.sqrt((2 * l + 1) / (2 * mmax + 1)) if l > mmax else 1.0
                       for l in range(lmax + 1)]
        self.avg_degree = float(avg_degree)

    def forward(self, atomic_numbers: Tensor, edge_rbf: Tensor, src: Tensor, dst: Tensor,
                wigner: List[Tensor], num_nodes: int, envelope: Optional[Tensor]) -> Tensor:
        e = src.shape[0]
        coef = self.rad_func(self.edge_scalars(edge_rbf, atomic_numbers, src, dst))
        coef = coef.reshape(e, self.lmax + 1, self.channels)
        if envelope is not None:
            coef = coef * envelope[:, None, None]
        # only the m = 0 coefficient is nonzero in the edge frame: the rotation back
        # takes the m = 0 row of every Wigner block
        parts: List[Tensor] = []
        for l, d in enumerate(wigner):
            parts.append(d.select(1, l).unsqueeze(-1) * coef.narrow(1, l, 1) * self.scales[l])
        out = scatter_sum(torch.cat(parts, dim=1), dst, num_nodes)
        return out / self.avg_degree


def _init_weights(module: nn.Module, weight_init: str) -> None:
    """The reference initialization: normal or uniform ``1/sqrt(fan_in)`` linear weights, zero biases."""
    radial = set()
    for m in module.modules():
        if isinstance(m, RadialMLP):
            for lin in m.modules():
                if isinstance(lin, nn.Linear):
                    radial.add(id(lin))
                    bound = 1.0 / math.sqrt(lin.in_features)
                    nn.init.uniform_(lin.weight, -bound, bound)
                    if lin.bias is not None:
                        nn.init.zeros_(lin.bias)
    for m in module.modules():
        if isinstance(m, (nn.Linear, SO3Linear)) and id(m) not in radial:
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            if weight_init == "normal":
                nn.init.normal_(m.weight, 0.0, 1.0 / math.sqrt(m.in_features))
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


@register_model("equiformer_v2")
class EquiformerV2(TransformerPotential):
    """EquiformerV2: equivariant graph attention on eSCN convolutions.

    See the module docstring for the architecture. The defaults are the QM9
    model of the paper (Table 13): six blocks of ``C = 96`` channels up to
    degree 4, four heads (64 attention scalars and 24 value channels each),
    hidden widths 48 (attention) and 96 (feed-forward), separable S2
    activations on an 18 x 18 grid, separable layer norm, attention
    re-normalization, and a 5 Angstrom cutoff with 128 Gaussians and edge
    channels of width 64. The OC20 models are ``l_max = 6``, ``m_max = 2``
    (or 3), ``C = 128``, 12 to 20 blocks, eight heads.

    Parameters
    ----------
    n_features : int, optional
        Channels per coefficient ``C`` (``sphere_channels``), by default 96.
    n_interactions : int, optional
        Number of Transformer blocks, by default 6.
    cutoff : float, optional
        Neighbor-list radius (Angstrom), by default 5.0.
    n_rbf : int, optional
        Number of Gaussians of the distance basis, by default 128.
    l_max, m_max : int, optional
        Highest degree, and highest order kept in the edge frames, by default 4 and 4.
    num_heads : int, optional
        Attention heads, by default 4.
    attn_hidden_channels : int, optional
        Channels of the first SO(2) convolution, by default 48.
    attn_alpha_channels, attn_value_channels : int, optional
        Per-head widths of the attention scalars and values, by default 64 and 24.
    ffn_hidden_channels : int, optional
        Hidden channels of the feed-forward networks, by default 96.
    edge_channels : int, optional
        Hidden width of the radial functions, by default 64.
    norm_type : str, optional
        ``"layer_norm_sh"`` (default), ``"layer_norm"`` or ``"rms_norm_sh"``.
    activation : str, optional
        ``"sep_s2"`` (default), ``"s2"`` or ``"gate"``.
    use_grid_mlp : bool, optional
        Grid MLP in the feed-forward networks, by default True.
    use_attn_renorm : bool, optional
        Attention re-normalization, by default True.
    use_atom_edge_embedding : bool, optional
        Element embeddings in the radial inputs, by default True.
    share_atom_edge_embedding : bool, optional
        One pair of element embeddings for all blocks, by default False.
    use_m_share_rad : bool, optional
        Radial weights shared by the orders of a degree, by default False.
    grid_resolution : int, optional
        Sphere samples per angle of the S2 activations, by default 18.
    alpha_drop, drop_path_rate, proj_drop : float, optional
        Attention dropout, stochastic depth and channel dropout, all 0 by default.
    avg_degree : float, optional
        Average number of neighbors (normalizes the degree embedding).
    avg_num_nodes : float, optional
        Average number of atoms (normalizes the energy).
    direct_forces : bool, optional
        Predict the forces with a final attention block (not conservative), by default False.
    smooth_cutoff : bool, optional
        Make the energy continuous at the cutoff, by default False.
    weight_init : str, optional
        ``"normal"`` (default) or ``"uniform"`` initialization of the linear layers.
    n_elements : int, optional
        Size of the element embeddings, indexed by ``Z``, by default 100.
    energy_shift, energy_scale : float, optional
        Per-atom energy standardization, by default 0 and 1.
    species : list of int, optional
        Only used to interpret ``atomic_energies``.
    atomic_energies : array-like, optional
        Per-species reference energies loaded into ``atom_ref``.
    """

    head_modules = ("energy_block", "energy_scale", "energy_shift", "atom_ref")
    exact_float64 = True
    smooth: Final[bool]
    direct_forces: Final[bool]
    lmax: Final[int]
    n_features: Final[int]

    def __init__(self, n_features: int = 96, n_interactions: int = 6, cutoff: float = 5.0,
                 n_rbf: int = 128, l_max: int = 4, m_max: int = 4, num_heads: int = 4,
                 attn_hidden_channels: int = 48, attn_alpha_channels: int = 64,
                 attn_value_channels: int = 24, ffn_hidden_channels: int = 96,
                 edge_channels: int = 64, norm_type: str = "layer_norm_sh",
                 activation: str = "sep_s2", use_grid_mlp: bool = True,
                 use_attn_renorm: bool = True, use_atom_edge_embedding: bool = True,
                 share_atom_edge_embedding: bool = False, use_m_share_rad: bool = False,
                 grid_resolution: Optional[int] = 18, alpha_drop: float = 0.0,
                 drop_path_rate: float = 0.0, proj_drop: float = 0.0,
                 avg_degree: float = _AVG_DEGREE, avg_num_nodes: float = _AVG_NUM_NODES,
                 direct_forces: bool = False, smooth_cutoff: bool = False,
                 weight_init: str = "normal", n_elements: int = 100, energy_shift: float = 0.0,
                 energy_scale: float = 1.0, species=None, atomic_energies=None):
        super().__init__()
        if not 0 <= m_max <= l_max:
            raise ValueError(f"need 0 <= m_max <= l_max, got m_max={m_max}, l_max={l_max}")
        if norm_type not in _NORMS:
            raise ValueError(f"norm_type must be one of {_NORMS}, got {norm_type!r}")
        if weight_init not in ("normal", "uniform"):
            raise ValueError(f"weight_init must be 'normal' or 'uniform', got {weight_init!r}")
        self.cutoff = float(cutoff)
        if species is not None:
            self.species = [int(z) for z in species]
        self.n_features = n_features
        self.node_feature_dim = n_features
        self.lmax, self.mmax = l_max, m_max
        self.avg_num_nodes = float(avg_num_nodes)
        self.smooth = bool(smooth_cutoff)
        self.direct_forces = bool(direct_forces)
        spacing = cutoff / (n_rbf - 1) if n_rbf > 1 else 1.0
        # the Gaussians of the reference: centers on [0, cutoff], width twice the spacing
        self.distance_expansion = GaussianRBF(n_rbf, cutoff, gamma=0.5 / (2.0 * spacing) ** 2)
        self.envelope = PolynomialCutoff(cutoff, p=6)
        radial = [n_rbf, edge_channels, edge_channels]
        self.shared_edge_scalars = _EdgeScalars(n_elements, edge_channels, share_atom_edge_embedding)
        block_elements = use_atom_edge_embedding and not share_atom_edge_embedding
        if share_atom_edge_embedding:
            radial[0] += 2 * edge_channels
        self.sphere_embedding = nn.Embedding(n_elements, n_features)
        self.frames = EdgeFrames(l_max)
        self.edge_degree_embedding = EdgeDegreeEmbedding(n_features, l_max, m_max, radial, n_elements,
                                                         block_elements, avg_degree)
        self.blocks = nn.ModuleList([
            TransformerBlockV2(n_features, attn_hidden_channels, num_heads, attn_alpha_channels,
                               attn_value_channels, ffn_hidden_channels, l_max, m_max, radial,
                               n_elements, block_elements, use_m_share_rad, activation,
                               use_attn_renorm, use_grid_mlp, norm_type, grid_resolution,
                               alpha_drop, drop_path_rate, proj_drop)
            for _ in range(n_interactions)])
        self.norm = make_norm(norm_type, l_max, n_features)
        self.energy_block = FeedForwardNetwork(n_features, ffn_hidden_channels, 1, l_max, activation,
                                               use_grid_mlp, grid_resolution)
        self.force_block: Optional[SO2GraphAttention] = None
        if self.direct_forces:
            self.force_block = SO2GraphAttention(
                n_features, attn_hidden_channels, num_heads, attn_alpha_channels,
                attn_value_channels, 1, l_max, m_max, radial, n_elements, block_elements,
                use_m_share_rad, activation, use_attn_renorm, grid_resolution, 0.0)
        _init_weights(self, weight_init)
        self._init_energy_terms(n_elements, energy_scale, energy_shift, species, atomic_energies)

    def _evaluate(self, atomic_numbers: Tensor, edge_index: Tensor, edge_vec: Tensor,
                  batch: Tensor, num_graphs: int) -> Tuple[Tensor, Tensor, Tensor]:
        """The final features ``(N, (L + 1)^2, C)``, the per-atom readout and the direct forces."""
        n = atomic_numbers.shape[0]
        src, dst = edge_index[0], edge_index[1]
        vec = -edge_vec
        r = torch.linalg.norm(vec, dim=-1)
        wigner = self.frames(vec)
        edge_rbf = self.shared_edge_scalars(self.distance_expansion(r), atomic_numbers, src, dst)
        envelope: Optional[Tensor] = None
        if self.smooth:
            envelope = self.envelope(r)
        x0 = self.sphere_embedding(atomic_numbers).unsqueeze(1)
        x = torch.cat([x0, x0.new_zeros((n, (self.lmax + 1) * (self.lmax + 1) - 1, self.n_features))], dim=1)
        x = x + self.edge_degree_embedding(atomic_numbers, edge_rbf, src, dst, wigner, n, envelope)
        for block in self.blocks:
            x = block(x, atomic_numbers, edge_rbf, src, dst, wigner, envelope, batch, num_graphs)
        x = self.norm(x)
        y = self.energy_block(x)[:, 0, 0] / self.avg_num_nodes
        forces = x.new_zeros((0, 3))
        if self.force_block is not None:
            out = self.force_block(x, atomic_numbers, edge_rbf, src, dst, wigner, envelope)
            forces = out[:, 1:4, 0] @ self.frames.basis
        return x, y, forces

    def _energy(self, y: Tensor, atomic_numbers: Tensor) -> Tensor:
        return (self.energy_scale * y + self.energy_shift
                + self.atom_ref(atomic_numbers).squeeze(-1))

    @torch.jit.export
    def node_features_energy(self, atomic_numbers: Tensor, edge_index: Tensor,
                             edge_vec: Tensor) -> Tuple[Tensor, Tensor]:
        """TorchScript-compatible core: degree-0 features and per-atom energies.

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
            The degree-0 channels of the final features ``(N, C)`` and the
            per-atom energies ``(N,)``.

        Raises
        ------
        RuntimeError
            For a model with direct forces, whose forces are not the gradient
            of its energy.
        """
        if self.direct_forces:
            raise RuntimeError("an EquiformerV2 with direct_forces has no energy-only core "
                               "(its forces are not the gradient of the energy)")
        batch = torch.zeros_like(atomic_numbers)
        x, y, _ = self._evaluate(atomic_numbers, edge_index, edge_vec, batch, 1)
        return x.select(1, 0), self._energy(y, atomic_numbers)

    @torch.jit.export
    def node_energy(self, atomic_numbers: Tensor, edge_index: Tensor, edge_vec: Tensor) -> Tensor:
        """Per-atom energy ``(N,)`` (the LAMMPS / TorchScript entry point)."""
        return self.node_features_energy(atomic_numbers, edge_index, edge_vec)[1]

    @torch.jit.ignore
    def forward(self, data: AtomicGraph) -> Dict[str, Tensor]:
        """Energies (and, with ``direct_forces``, forces) of a batched graph.

        Parameters
        ----------
        data : AtomicGraph
            Batched atomic graph.

        Returns
        -------
        dict of str to Tensor
            ``"node_energy"`` ``(N,)``, ``"energy"`` ``(B,)``,
            ``"node_features"`` (the degree-0 channels of the final features)
            and, with ``direct_forces``, ``"forces"`` ``(N, 3)``.
        """
        x, y, forces = self._evaluate(data.atomic_numbers, data.edge_index, data.edge_vectors(),
                                      data.batch, data.num_graphs)
        node_energy = self._energy(y, data.atomic_numbers)
        out = {"node_energy": node_energy, "energy": self.aggregate_energy(node_energy, data),
               "node_features": x.select(1, 0)}
        if self.direct_forces:
            out["forces"] = forces.to(data.pos.dtype)
        return out

    @classmethod
    def from_config(cls, cfg) -> "EquiformerV2":
        """Build an :class:`EquiformerV2` from a configuration object.

        Core fields: ``cfg.n_features`` (``C``), ``cfg.n_interactions``,
        ``cfg.n_rbf``, ``cfg.cutoff``. Everything else is a keyword argument of
        the class read from ``cfg.extra``; the reference code's spellings
        (``sphere_channels``, ``lmax_list``, ``max_radius``, ...) are
        translated by :mod:`xnn.common.config.translate`.

        Parameters
        ----------
        cfg : xnn.common.config.schema.ModelConfig
            The core model config.

        Returns
        -------
        EquiformerV2
            Instantiated model.
        """
        from xnn.common.config.coerce import coerce_per_species, coerce_species

        extra = dict(cfg.extra or {})
        species = (coerce_species(extra.get("species"))
                   if extra.get("species") is not None else None)
        kwargs = {}
        for key in ("l_max", "m_max"):
            if key in extra:
                value = extra[key]
                if isinstance(value, (list, tuple)):
                    if len(value) != 1:
                        raise ValueError(f"{key}: one resolution is supported, got {value}")
                    value = value[0]
                kwargs[key] = int(value)
        for key, kind in (("num_heads", int), ("attn_hidden_channels", int),
                          ("attn_alpha_channels", int), ("attn_value_channels", int),
                          ("ffn_hidden_channels", int), ("edge_channels", int),
                          ("norm_type", str), ("activation", str), ("use_grid_mlp", bool),
                          ("use_attn_renorm", bool), ("use_atom_edge_embedding", bool),
                          ("share_atom_edge_embedding", bool), ("use_m_share_rad", bool),
                          ("alpha_drop", float), ("drop_path_rate", float), ("proj_drop", float),
                          ("avg_degree", float), ("avg_num_nodes", float),
                          ("direct_forces", bool), ("smooth_cutoff", bool), ("weight_init", str),
                          ("n_elements", int), ("energy_shift", float), ("energy_scale", float)):
            if key in extra and extra[key] is not None:
                kwargs[key] = kind(extra[key])
        if "grid_resolution" in extra:
            res = extra["grid_resolution"]
            kwargs["grid_resolution"] = None if res is None else int(res)
        return cls(n_features=cfg.n_features, n_interactions=cfg.n_interactions, cutoff=cfg.cutoff,
                   n_rbf=cfg.n_rbf, species=species,
                   atomic_energies=coerce_per_species(extra.get("atomic_energies"),
                                                      species or [], "atomic_energies"),
                   **kwargs)
