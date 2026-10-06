"""3D steerable CNN (Weiler, Geiger, Welling, Boomsma, Cohen, NeurIPS 2018).

An SE(3)-equivariant convolutional network over fields on a cubic grid,
built from the manuscript (arXiv:1807.02547v2) and the conventions of its
reference implementation, without using its code:

* feature spaces are stacks of fields transforming under the irreducible
  representations ``D^l`` of SO(3) (Sec. 3.2); a stack is described by its
  multiplicities ``(m_0, m_1, m_2, ...)`` and laid out channel-wise as the
  ``m_l`` fields of each order in turn, each with its ``2l + 1`` components;
* an equivariant linear map is a cross-correlation with a rotation-steerable
  kernel (Theorem 2), ``kappa(r x) = D^j(r) kappa(x) D^l(r)^{-1}`` for every
  block mapping order ``l`` to order ``j`` (eq 12);
* the solutions are, per angular frequency ``J`` from ``|j - l|`` to ``j + l``,
  the spherical harmonics ``Y^J`` of the direction, carried back from the
  coupled basis to the ``(2j + 1) x (2l + 1)`` block by the change of basis
  ``Q`` of eq 14, times a free radial profile (eqs 15-16). ``Q`` is made
  of Clebsch-Gordan coefficients (footnote 4; the real-basis ``wigner_3j``
  of e3nn), so no numerical solve is needed; the tests solve eq 14
  numerically as the paper does and compare;
* on the discrete grid (Sec. 4.4.1) the radial profiles are Gaussian shells
  of width 0.6 voxels at the integer radii ``0 .. s // 2``, and every shell
  only carries the frequencies ``J <= J_max`` of a radius-dependent bandlimit
  that suppresses aliasing; every basis kernel is scaled to unit norm. A
  convolution learns one weight per basis kernel and per (output field,
  input field) pair and stacks the kernel it spans for a standard
  ``conv3d`` (Sec. 4.4.3);
* nonlinearities are the gated kind (Sec. 4.3, supplement Fig. 5): scalar
  fields pass through an element-wise activation (ReLU in the paper), every
  non-scalar field is multiplied by the sigmoid of an extra scalar field the
  same convolution produces;
* downsampling is preceded by a Gaussian low-pass filter (Sec. 4.4.2) and
  batch normalization, when used, normalizes non-scalar fields by their
  norm (supplement eq 17).

:class:`SteerableCNN` is the interatomic potential built from these blocks
on the voxelized environments of :class:`~xnn.cnn.featurizers.VoxelGrid`:
the energy is a pooled scalar field, so it is exactly invariant under the
rotations of the grid onto itself, approximately (to the bandlimit) under
every other rotation, and the forces co-rotate. The paper's networks are
SE(3)- but not O(3)-equivariant (a reflection is not a rotation), and so is
this potential: mirror images are not constrained to the same energy.

Requires e3nn (``pip install "xnn[gnn]"``) for the spherical harmonics and
Clebsch-Gordan coefficients.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from xnn.common.models.ops import make_activation
from xnn.common.models.registry import register_model
from .base import (VoxelPotential, default_fields, default_strides, field_dim,
                   global_average_pool, grid_coordinates, low_pass_filter, rotate_voxels)



def _import_o3():
    """Import ``e3nn.o3``; e3nn 0.4.4 reads its constants with ``torch.load``,
    which from torch 2.6 rejects the slice objects of that file unless they
    are allowed (the guard of :mod:`xnn.gnn`)."""
    import contextlib
    try:
        ctx = torch.serialization.safe_globals([slice])
    except AttributeError:          # torch < 2.4
        ctx = contextlib.nullcontext()
    with ctx:
        from e3nn import o3
    return o3


try:
    o3 = _import_o3()
    _HAS_E3NN = True
except Exception:                   # noqa: BLE001 - optional dependency
    o3 = None
    _HAS_E3NN = False

#: radius-dependent angular bandlimits ``J_max`` of the shells at radii
#: ``0, 1, 2, ...`` voxels (Sec. 4.4.1); the names follow the reference
#: implementation: ``conservative`` keeps the equivariance of every shell
#: above 90 %, ``sfcnn`` is the choice of Weiler et al. CVPR 2018, and
#: ``compromise`` (the default) lies in between
BANDLIMITS: Dict[str, Tuple[int, ...]] = {
    "conservative": (0, 2, 4, 6, 8, 10, 12, 14),
    "compromise": (0, 3, 5, 7, 9, 11, 13, 15),
    "sfcnn": (0, 4, 6, 8, 10, 12, 14, 16),
}


def _require_e3nn() -> None:
    if not _HAS_E3NN:
        raise ImportError("the 3D steerable CNN needs e3nn: pip install \"xnn[gnn]\"")


def shell_bandlimits(size: int, bandlimit="compromise") -> Tuple[List[float], List[int]]:
    """Radii and angular bandlimits of the Gaussian shells of a kernel.

    Parameters
    ----------
    size : int
        Side of the cubic kernel.
    bandlimit : str or sequence of int, optional
        A key of :data:`BANDLIMITS` or an explicit ``J_max`` per shell.

    Returns
    -------
    (list of float, list of int)
        The ``size // 2 + 1`` integer shell radii and their ``J_max``.
    """
    n_shells = size // 2 + 1
    if isinstance(bandlimit, str):
        if bandlimit not in BANDLIMITS:
            raise ValueError(f"unknown bandlimit {bandlimit!r}; choose one of {sorted(BANDLIMITS)}")
        limits = BANDLIMITS[bandlimit]
    else:
        limits = tuple(int(j) for j in bandlimit)
    if len(limits) < n_shells:
        raise ValueError(f"need a bandlimit for each of the {n_shells} shells, got {len(limits)}")
    return [float(m) for m in range(n_shells)], list(limits[:n_shells])


def angular_kernel_basis(l_in: int, l_out: int, J: int, points: Tensor) -> Tensor:
    """The angular part of the steerable kernel basis (eqs 13-16).

    ``kappa^{jl,J}(x)[i, k] = sum_M C^{jlJ}_{ikM} Y^J_M(x / |x|)``, with the
    real spherical harmonics ``Y^J`` of e3nn and the real Clebsch-Gordan
    coefficients ``C`` (``o3.wigner_3j``), which form the change of basis
    ``Q`` of eq 14: ``[D^j (x) D^l](r) C = C D^J(r)``. The block therefore
    satisfies the steerability constraint ``kappa(r x) = D^j(r) kappa(x)
    D^l(r)^T`` of eq 12, which the tests verify. At the origin the value
    is the constant ``Y^0`` for ``J = 0`` and zero otherwise.

    Parameters
    ----------
    l_in, l_out : int
        Orders ``l`` and ``j`` of the input and output fields.
    J : int
        Angular frequency, ``|j - l| <= J <= j + l``.
    points : Tensor
        Sample points, shape ``(..., 3)``.

    Returns
    -------
    Tensor
        Shape ``(2 l_out + 1, 2 l_in + 1, ...)`` in float64.
    """
    _require_e3nn()
    if not abs(l_in - l_out) <= J <= l_in + l_out:
        raise ValueError(f"J={J} is not coupled by l_in={l_in}, l_out={l_out}")
    pts = points.to(torch.float64).reshape(-1, 3)
    norm = torch.linalg.norm(pts, dim=-1)
    unit = torch.where(norm[:, None] > 0, pts / norm.clamp(min=1e-300)[:, None], pts)
    Y = o3.spherical_harmonics(J, unit, normalize=False, normalization="component")
    if J > 0:
        Y = torch.where(norm[:, None] > 0, Y, torch.zeros_like(Y))
    C = o3.wigner_3j(l_out, l_in, J, dtype=torch.float64)             # (2j+1, 2l+1, 2J+1)
    block = torch.einsum("ikM,pM->ikp", C, Y)
    return block.reshape(2 * l_out + 1, 2 * l_in + 1, *points.shape[:-1])


def steerable_kernel_basis(l_in: int, l_out: int, size: int, bandlimit="compromise",
                           shell_width: float = 0.6) -> Optional[Tensor]:
    """Sampled basis of the steerable kernels between two field orders (Sec. 4.4.1).

    Basis kernel ``(m, J)`` is the angular block
    :func:`angular_kernel_basis` of frequency ``J`` times the Gaussian shell
    ``exp(-(|x| - m)^2 / (2 shell_width^2))`` at integer radius ``m``, kept
    only when ``J <= J_max(m)``, and scaled to unit norm. The basis is
    ordered shell by shell, frequency by frequency.

    Parameters
    ----------
    l_in, l_out : int
        Orders of the input and output fields.
    size : int
        Side ``s`` of the cubic kernel; the grid points are the integer
        offsets from its center.
    bandlimit : str or sequence of int, optional
        Shell bandlimits, see :func:`shell_bandlimits`.
    shell_width : float, optional
        Standard deviation of the radial Gaussians in voxels, by default 0.6.

    Returns
    -------
    Tensor or None
        Shape ``(B, 2 l_out + 1, 2 l_in + 1, s, s, s)`` in float64, or
        ``None`` when no frequency passes the bandlimits (``B = 0``).
    """
    _require_e3nn()
    points = grid_coordinates(size)
    radius = torch.linalg.norm(points, dim=-1)
    radii, limits = shell_bandlimits(size, bandlimit)
    angular = {J: angular_kernel_basis(l_in, l_out, J, points)
               for J in range(abs(l_in - l_out), l_in + l_out + 1)}
    basis = []
    for m, j_max in zip(radii, limits):
        window = torch.exp(-0.5 * ((radius - m) / shell_width) ** 2)
        for J, block in angular.items():
            if J <= j_max:
                kernel = block * window
                basis.append(kernel / kernel.norm())
    if not basis:
        return None
    return torch.stack(basis)


def n_basis_kernels(l_in: int, l_out: int, size: int, bandlimit="compromise") -> int:
    """Number of basis kernels ``B`` between two orders (Sec. 4.4.1)."""
    _, limits = shell_bandlimits(size, bandlimit)
    return sum(1 for j_max in limits
               for J in range(abs(l_in - l_out), l_in + l_out + 1) if J <= j_max)


def field_slices(fields: Sequence[int]) -> List[Tuple[int, int, int]]:
    """``(l, start, stop)`` channel ranges of the non-empty orders of a stack."""
    out, start = [], 0
    for l, m in enumerate(fields):
        if m > 0:
            out.append((l, start, start + m * (2 * l + 1)))
            start += m * (2 * l + 1)
    return out


def rotate_fields(x: Tensor, R: Tensor, fields: Sequence[int]) -> Tensor:
    """Rotate a stack of fields by a symmetry of the cube: grid and fibers.

    Applies :func:`~xnn.cnn.models.base.rotate_voxels` to every channel and
    ``D^l(R)`` to the components of each field of order ``l`` (the induced
    representation of eq 1, for a proper rotation ``R``).

    Parameters
    ----------
    x : Tensor
        Fields of shape ``(B, field_dim(fields), s, s, s)``.
    R : Tensor
        A proper rotation among the signed permutation matrices.
    fields : sequence of int
        Multiplicities of the stack.

    Returns
    -------
    Tensor
        The rotated stack, same shape.
    """
    _require_e3nn()
    R = torch.as_tensor(R, dtype=torch.float64)
    if abs(float(torch.det(R)) - 1.0) > 1e-8:
        raise ValueError("R must be a proper rotation (det +1)")
    out = rotate_voxels(x, R)
    pieces = []
    for l, start, stop in field_slices(fields):
        part = out[:, start:stop]
        if l > 0:
            D = o3.wigner_D(l, *o3.matrix_to_angles(R)).to(x.dtype).to(x.device)
            m = (stop - start) // (2 * l + 1)
            part = torch.einsum("ij,bmjxyz->bmixyz",
                                D, part.reshape(x.shape[0], m, 2 * l + 1, *x.shape[2:]))
            part = part.reshape(x.shape[0], stop - start, *x.shape[2:])
        pieces.append(part)
    return torch.cat(pieces, dim=1)


class SteerableConv3d(nn.Module):
    """SE(3)-equivariant 3D convolution between stacks of fields.

    For every pair of output order ``j`` and input order ``l`` the kernel
    block is a learned linear combination of the ``B_{jl}`` basis kernels of
    :func:`steerable_kernel_basis`, with one weight per (output field,
    input field, basis kernel); the blocks are stacked into one
    ``(K_out, K_in, s, s, s)`` filter bank for ``conv3d`` (Sec. 4.4.3). The
    basis of a block is scaled by ``sqrt((2j + 1) / (B_{jl} sum_l m_l))`` so
    that weights drawn from ``N(0, 1)`` give outputs of unit variance (the
    initialization of the reference implementation). There is no bias: the
    biases of scalar fields live in the gated nonlinearity.

    Parameters
    ----------
    fields_in, fields_out : sequence of int
        Multiplicities ``(m_0, m_1, ...)`` of the input and output stacks.
    kernel_size : int, optional
        Side of the cubic kernel, by default 5.
    n_gates : int, optional
        Extra scalar output fields appended after ``fields_out`` (the gates
        of :class:`GatedBlock`), by default 0.
    stride, padding, dilation : int, optional
        As in ``conv3d``; ``padding=None`` keeps the grid size.
    bandlimit : str or sequence of int, optional
        Shell bandlimits, see :func:`shell_bandlimits`.
    shell_width : float, optional
        Width of the radial Gaussian shells in voxels, by default 0.6.

    Attributes
    ----------
    weight : torch.nn.ParameterDict
        ``"{i}_{j}"`` -> ``(m_out, m_in, B)`` weights of output block ``i``
        and input block ``j``.
    """

    def __init__(self, fields_in: Sequence[int], fields_out: Sequence[int],
                 kernel_size: int = 5, n_gates: int = 0, stride: int = 1,
                 padding: Optional[int] = None, dilation: int = 1,
                 bandlimit="compromise", shell_width: float = 0.6):
        super().__init__()
        _require_e3nn()
        self.fields_in = tuple(int(m) for m in fields_in)
        self.fields_out = tuple(int(m) for m in fields_out)
        self.n_gates = int(n_gates)
        self.kernel_size = int(kernel_size)
        self.stride, self.dilation = int(stride), int(dilation)
        self.padding = self.kernel_size // 2 if padding is None else int(padding)
        self.blocks_in = [(m, l) for l, m in enumerate(self.fields_in) if m > 0]
        self.blocks_out = [(m, l) for l, m in enumerate(self.fields_out) if m > 0]
        if self.n_gates > 0:
            self.blocks_out.append((self.n_gates, 0))
        self.in_channels = field_dim(self.fields_in)
        self.out_channels = field_dim(self.fields_out) + self.n_gates
        total_in = sum(m for m, _ in self.blocks_in)
        self.weight = nn.ParameterDict()
        self._basis_keys: List[Tuple[int, int, Optional[str]]] = []
        for i, (m_out, l_out) in enumerate(self.blocks_out):
            for j, (m_in, l_in) in enumerate(self.blocks_in):
                basis = steerable_kernel_basis(l_in, l_out, self.kernel_size, bandlimit, shell_width)
                if basis is None:
                    self._basis_keys.append((i, j, None))
                    continue
                n_b = basis.shape[0]
                basis = basis * math.sqrt((2 * l_out + 1) / (n_b * total_in))
                name = f"basis_{i}_{j}"
                self.register_buffer(name, basis.to(torch.get_default_dtype()))
                self.weight[f"{i}_{j}"] = nn.Parameter(torch.randn(m_out, m_in, n_b))
                self._basis_keys.append((i, j, name))
        if len(self.weight) == 0:
            raise ValueError("no basis kernel couples these fields at this kernel size "
                             "and bandlimit; the layer would be identically zero")

    def kernel(self) -> Tensor:
        """The filter bank spanned by the current weights, ``(K_out, K_in, s, s, s)``."""
        s = self.kernel_size
        rows = []
        for i, (m_out, l_out) in enumerate(self.blocks_out):
            cols = []
            for j, (m_in, l_in) in enumerate(self.blocks_in):
                name = self._basis_keys[i * len(self.blocks_in) + j][2]
                d_out, d_in = m_out * (2 * l_out + 1), m_in * (2 * l_in + 1)
                if name is None:
                    ref = next(iter(self.weight.values()))
                    cols.append(torch.zeros(d_out, d_in, s, s, s, dtype=ref.dtype, device=ref.device))
                    continue
                w = self.weight[f"{i}_{j}"]
                block = torch.einsum("uvb,bikxyz->uivkxyz", w, getattr(self, name))
                cols.append(block.reshape(d_out, d_in, s, s, s))
            rows.append(torch.cat(cols, dim=1))
        return torch.cat(rows, dim=0)

    def forward(self, x: Tensor) -> Tensor:
        """Convolve ``(B, in_channels, X, Y, Z)`` fields into ``(B, out_channels, X', Y', Z')``."""
        return F.conv3d(x, self.kernel(), stride=self.stride, padding=self.padding,
                        dilation=self.dilation)

    def extra_repr(self) -> str:
        return (f"fields_in={self.fields_in}, fields_out={self.fields_out}, n_gates={self.n_gates}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}")


class SteerableBatchNorm(nn.Module):
    """Equivariant batch normalization (supplement Sec. 1.2, eq 17).

    Scalar fields get ordinary batch normalization (mean and variance over
    the batch and the grid). A non-scalar field is only rescaled, by the
    inverse root of the batch- and grid-averaged squared norm of its
    components, which commutes with every rotation. Running statistics are
    kept for evaluation; the affine transform has a gain per field and a
    bias for scalar fields only.

    Parameters
    ----------
    fields : sequence of int
        Multiplicities of the normalized stack.
    eps : float, optional
        Added to the variances, by default 1e-5.
    momentum : float, optional
        Running-statistics momentum, by default 0.1.
    affine : bool, optional
        Learn the gains and scalar biases, by default ``True``.
    """

    def __init__(self, fields: Sequence[int], eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True):
        super().__init__()
        self.fields = tuple(int(m) for m in fields)
        self.eps, self.momentum, self.affine = float(eps), float(momentum), bool(affine)
        n_scalar = self.fields[0] if self.fields else 0
        n_fields = sum(self.fields)
        self.register_buffer("running_mean", torch.zeros(n_scalar))
        self.register_buffer("running_var", torch.ones(n_fields))
        if affine:
            self.weight = nn.Parameter(torch.ones(n_fields))
            self.bias = nn.Parameter(torch.zeros(n_scalar))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        """Normalize ``(B, field_dim(fields), X, Y, Z)`` fields."""
        batch = x.shape[0]
        pieces, field0 = [], 0
        new_var = []
        for l, start, stop in field_slices(self.fields):
            dim = 2 * l + 1
            m = (stop - start) // dim
            f = x[:, start:stop].reshape(batch, m, dim, -1)         # (B, m, 2l+1, V)
            if l == 0:
                if self.training:
                    mean = f.mean(dim=(0, 2, 3))
                    with torch.no_grad():
                        self.running_mean.lerp_(mean, self.momentum)
                else:
                    mean = self.running_mean
                f = f - mean.view(1, m, 1, 1)
            if self.training:
                var = f.pow(2).sum(2).mean(dim=(0, 2))              # mean squared norm per field
                new_var.append(var.detach())
            else:
                var = self.running_var[field0:field0 + m]
            f = f / torch.sqrt(var + self.eps).view(1, m, 1, 1)
            if self.affine:
                f = f * self.weight[field0:field0 + m].view(1, m, 1, 1)
                if l == 0:
                    f = f + self.bias.view(1, m, 1, 1)
            pieces.append(f.reshape(batch, stop - start, *x.shape[2:]))
            field0 += m
        if self.training and new_var:
            with torch.no_grad():
                self.running_var.lerp_(torch.cat(new_var), self.momentum)
        return torch.cat(pieces, dim=1)


class GatedBlock(nn.Module):
    """Steerable convolution with the gated nonlinearity (Sec. 4.3, Fig. 5).

    The convolution produces the output stack plus one extra scalar field
    per non-scalar output field. Scalar outputs pass through ``activation``
    (with a bias), the extra scalars through ``gate_activation`` (with a
    bias) and multiply, as gates, the non-scalar fields they belong to.
    With a ``stride`` the result is low-pass filtered and subsampled
    (Sec. 4.4.2; ``smooth_stride=False`` strides the convolution instead).
    ``normalization="batch"`` normalizes the *input* stack with
    :class:`SteerableBatchNorm`, the "batch normalization merged with the
    convolution" of the supplement.

    Parameters
    ----------
    fields_in, fields_out : sequence of int
        Multiplicities of the input and output stacks.
    kernel_size : int, optional
        Side of the cubic kernels, by default 5.
    stride : int, optional
        Downsampling factor, by default 1.
    padding : int or None, optional
        Zero padding; ``None`` keeps the grid size.
    activation : str or None, optional
        Activation of the scalar fields, by default ``"relu"`` (the paper);
        ``None`` leaves them linear.
    gate_activation : str or None, optional
        Activation of the gates, by default ``"sigmoid"``; ``None`` leaves
        the non-scalar fields ungated (a linear block).
    normalization : str or None, optional
        ``"batch"`` or ``None``.
    smooth_stride : bool, optional
        Low-pass filter before subsampling, by default ``True``.
    bias : bool, optional
        Biases on the scalar fields and gates, by default ``True``.
    bandlimit, shell_width
        See :class:`SteerableConv3d`.

    Attributes
    ----------
    conv : SteerableConv3d
        The convolution (its ``n_gates`` extra scalar outputs are the gates).
    """

    def __init__(self, fields_in: Sequence[int], fields_out: Sequence[int],
                 kernel_size: int = 5, stride: int = 1, padding: Optional[int] = None,
                 activation: Optional[str] = "relu", gate_activation: Optional[str] = "sigmoid",
                 normalization: Optional[str] = None, smooth_stride: bool = True,
                 bias: bool = True, bandlimit="compromise", shell_width: float = 0.6):
        super().__init__()
        if normalization not in (None, "batch"):
            raise ValueError(f"normalization must be None or 'batch', got {normalization!r}")
        self.fields_in = tuple(int(m) for m in fields_in)
        self.fields_out = tuple(int(m) for m in fields_out)
        self.n_scalar = self.fields_out[0] if self.fields_out else 0
        n_non_scalar = sum(self.fields_out[1:])
        self.n_gates = n_non_scalar if gate_activation is not None else 0
        self.smooth_stride = bool(smooth_stride) and stride > 1
        self.stride = int(stride)
        self.norm = SteerableBatchNorm(self.fields_in) if normalization == "batch" else None
        self.conv = SteerableConv3d(self.fields_in, self.fields_out, kernel_size, self.n_gates,
                                    stride=1 if self.smooth_stride else self.stride,
                                    padding=padding, bandlimit=bandlimit, shell_width=shell_width)
        self.act = make_activation(activation) if activation is not None else None
        self.gate_act = make_activation(gate_activation) if gate_activation is not None else None
        n_bias = (self.n_scalar if self.act is not None else 0) + self.n_gates
        self.bias = nn.Parameter(torch.zeros(n_bias)) if bias and n_bias > 0 else None

    @property
    def out_channels(self) -> int:
        """int : Channels of the output stack (gates excluded)."""
        return field_dim(self.fields_out)

    def forward(self, x: Tensor) -> Tensor:
        """Map ``(B, field_dim(fields_in), X, Y, Z)`` to ``(B, field_dim(fields_out), X', Y', Z')``."""
        if self.norm is not None:
            x = self.norm(x)
        y = self.conv(x)
        n_out = self.out_channels
        pieces, b0 = [], 0
        scalars = y[:, :self.n_scalar]
        if self.act is not None and self.n_scalar > 0:
            if self.bias is not None:
                scalars = scalars + self.bias[:self.n_scalar].view(1, -1, 1, 1, 1)
                b0 = self.n_scalar
            scalars = self.act(scalars)
        pieces.append(scalars)
        if self.n_gates > 0:
            gates = y[:, n_out:]
            if self.bias is not None:
                gates = gates + self.bias[b0:b0 + self.n_gates].view(1, -1, 1, 1, 1)
            gates = self.gate_act(gates)
            g0 = 0
            for l, start, stop in field_slices(self.fields_out):
                if l == 0:
                    continue
                m = (stop - start) // (2 * l + 1)
                field = y[:, start:stop].reshape(y.shape[0], m, 2 * l + 1, *y.shape[2:])
                gate = gates[:, g0:g0 + m].unsqueeze(2)
                g0 += m
                pieces.append((field * gate).reshape(y.shape[0], stop - start, *y.shape[2:]))
        else:
            pieces.append(y[:, self.n_scalar:n_out])
        z = torch.cat(pieces, dim=1)
        if self.smooth_stride:
            z = low_pass_filter(z, self.stride, self.stride)
        return z


@register_model("se3cnn")
class SteerableCNN(VoxelPotential):
    """3D steerable CNN interatomic potential (Weiler *et al.*, NeurIPS 2018).

    The species density grids of :class:`~xnn.cnn.featurizers.VoxelGrid`
    (one scalar field per species) pass through ``len(fields)``
    :class:`GatedBlock` blocks; the last block outputs scalar fields only,
    which a global average pool turns into the invariant per-atom features
    of the shared readout. The defaults are the paper's recipe scaled to
    environments: hidden blocks carry fields of order 0, 1 and 2 with
    multiplicities doubling block by block, kernels of side 5, a stride of 2
    (low-pass filtered) in the inner blocks, gated nonlinearities, no batch
    normalization, the ``compromise`` bandlimits and shells of width 0.6
    voxels.

    Parameters
    ----------
    species : sequence of int, optional
        Atomic numbers with a density channel, by default ``(1, 6, 8)``.
    cutoff : float, optional
        Radius of the voxelized environment (Angstrom), by default 4.0.
    grid_size : int, optional
        Voxels per axis, by default 17.
    fields : sequence of sequence of int or None, optional
        Multiplicities ``(m_0, m_1, ...)`` of the output stack of each
        block; the last entry must be scalar only. ``None`` derives them
        from ``n_features``, ``n_blocks`` and ``l_max``
        (:func:`~xnn.cnn.models.base.default_fields`).
    n_features : int, optional
        Scalar fields of the last block when ``fields`` is ``None`` (and the
        readout width), by default 32.
    n_blocks : int, optional
        Number of blocks when ``fields`` is ``None``, by default 3.
    l_max : int, optional
        Highest field order of the default hidden blocks, by default 2.
    kernel_size : int, optional
        Side of the cubic kernels, by default 5.
    strides : sequence of int or None, optional
        Downsampling per block; ``None`` downsamples by 2 in every block but
        the first and the last.
    padding : int or None, optional
        Zero padding; ``None`` keeps the grid size.
    activation : str, optional
        Activation of the scalar fields, by default ``"ssp"`` (smooth, for
        a smooth energy surface; the paper's networks use ``"relu"``).
    gate_activation : str, optional
        Activation of the gates, by default ``"sigmoid"``.
    normalization : str or None, optional
        ``"batch"`` for :class:`SteerableBatchNorm` in every block, by
        default ``None``.
    smooth_stride : bool, optional
        Low-pass filter before every downsampling, by default ``True``.
    bandlimit : str or sequence of int, optional
        Shell bandlimits of the kernels, by default ``"compromise"``.
    shell_width : float, optional
        Width of the radial Gaussian shells in voxels, by default 0.6.
    sigma, cutoff_fn, include_center, readout_activation, energy_shift, energy_scale, atomic_energies
        See :class:`~xnn.cnn.models.base.VoxelPotential`.

    Attributes
    ----------
    blocks : torch.nn.Sequential
        The gated blocks.
    fields : list of tuple of int
        The output multiplicities of every block.
    """

    def __init__(self, species: Sequence[int] = (1, 6, 8), cutoff: float = 4.0,
                 grid_size: int = 17, fields: Optional[Sequence[Sequence[int]]] = None,
                 n_features: int = 32, n_blocks: int = 3, l_max: int = 2,
                 kernel_size: int = 5, strides: Optional[Sequence[int]] = None,
                 padding: Optional[int] = None, activation: str = "ssp",
                 gate_activation: str = "sigmoid", normalization: Optional[str] = None,
                 smooth_stride: bool = True, bandlimit="compromise", shell_width: float = 0.6,
                 sigma: Optional[float] = None, cutoff_fn: Optional[str] = "cosine",
                 include_center: bool = True, readout_activation: str = "ssp",
                 energy_shift: float = 0.0, energy_scale: float = 1.0, atomic_energies=None):
        _require_e3nn()
        if fields is None:
            fields = default_fields(n_features, n_blocks, l_max)
        fields = [tuple(int(m) for m in f) for f in fields]
        if not fields or any(m > 0 for m in fields[-1][1:]) or fields[-1][0] < 1:
            raise ValueError("the last block must output scalar fields only")
        if strides is None:
            strides = default_strides(len(fields))
        if len(strides) != len(fields):
            raise ValueError("strides must have one entry per block")
        super().__init__(species, cutoff, grid_size, fields[-1][0], sigma, cutoff_fn,
                         include_center, readout_activation, energy_shift, energy_scale,
                         atomic_energies)
        self.fields = fields
        blocks, f_in = [], (self.n_channels,)
        for f_out, stride in zip(fields, strides):
            blocks.append(GatedBlock(f_in, f_out, kernel_size, stride, padding, activation,
                                     gate_activation, normalization, smooth_stride, True,
                                     bandlimit, shell_width))
            f_in = f_out
        self.blocks = nn.Sequential(*blocks)

    def trunk(self, grid: Tensor) -> Tensor:
        """Gated blocks followed by global average pooling of the scalar fields."""
        return global_average_pool(self.blocks(grid))

    @classmethod
    def from_config(cls, cfg) -> "SteerableCNN":
        """Build a :class:`SteerableCNN` from a configuration object.

        ``cfg.cutoff``, ``cfg.n_features`` (scalar fields of the last block)
        and ``cfg.n_interactions`` (number of blocks) are the core fields;
        ``cfg.extra`` may set ``species``, ``grid_size``, ``sigma``,
        ``cutoff_fn``, ``include_center``, ``fields``, ``l_max``,
        ``kernel_size``, ``strides``, ``padding``, ``activation``,
        ``gate_activation``, ``normalization``, ``smooth_stride``,
        ``bandlimit``, ``shell_width``, ``readout_activation``,
        ``energy_shift``, ``energy_scale`` and ``atomic_energies``.

        Parameters
        ----------
        cfg : xnn.common.config.schema.ModelConfig
            The model config.

        Returns
        -------
        SteerableCNN
            The model.
        """
        extra = dict(cfg.extra or {})
        opts = cls._common_config_options(cfg)
        fields = extra.get("fields")
        strides = extra.get("strides")
        padding = extra.get("padding")
        bandlimit = extra.get("bandlimit", "compromise")
        return cls(
            fields=None if fields is None else [[int(m) for m in f] for f in fields],
            n_features=cfg.n_features,
            n_blocks=cfg.n_interactions,
            l_max=int(extra.get("l_max", 2)),
            kernel_size=int(extra.get("kernel_size", 5)),
            strides=None if strides is None else [int(s) for s in strides],
            padding=None if padding is None else int(padding),
            activation=extra.get("activation", "ssp"),
            gate_activation=extra.get("gate_activation", "sigmoid"),
            normalization=extra.get("normalization"),
            smooth_stride=bool(extra.get("smooth_stride", True)),
            bandlimit=bandlimit if isinstance(bandlimit, str) else [int(j) for j in bandlimit],
            shell_width=float(extra.get("shell_width", 0.6)),
            **opts,
        )
