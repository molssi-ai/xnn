"""Triton kernels of the SphericalCNN fast path (see :mod:`xnn.cnn.fast.spherical`).

Imported only when the fast path runs, so that ``triton`` is needed only there.

Spectra are compact and blocked by degree: the entries ``|m|, |n| <= l`` of the
signals of a batch ``B`` with ``C`` channels form, for each degree, one contiguous
block ``(B, 2l + 1, C, 2l + 1)`` indexed ``(b, m, c, n)`` (``SO(3)``) or ``(B, 2l + 1,
C)`` indexed ``(b, m, c)`` (``S^2``), and the blocks follow each other degree after
degree; complex numbers are (real, imaginary) pairs on a trailing axis of two. A
block is then the matrix of a correlation's spectral product of that degree, rows
``(b, m)`` and columns ``(c, n)``, as a view. For small batches the blocks may instead
be padded to ``2L - 1`` entries per index for every degree (centered, zeros outside
``|m|, |n| <= l``), so that the products of all degrees are one batched matrix
product. Two kernels:

* the contraction over ``beta`` of the forward transforms, ``F[(l, b, m, c, n)] =
  sum_j X[b, c, j, m, n] T[(m, n), j, l]`` for every ``l >= max(|m|, |n|)``, which
  reads the FFT of the grid signal at the band positions (frequency ``m`` at
  position ``m mod S``) and is a small matrix product per frequency;
* the contraction over the degree of the inverse transforms, ``X[b, c, j, m, n] =
  sum_l F[(l, b, m, c, n)] T[(m, n), j, l]``, written onto the whole FFT grid, zero
  where no frequency of the band sits.

For the same (real) table the two are adjoint to each other, so the backward of
each transform is the other kernel and differentiable again. float32 products run
on ``tl.dot`` in IEEE precision, float64 ones as broadcast sums.
"""
# no postponed annotations: Triton's interpreter (TRITON_INTERPRET=1) recognizes the
# tl.constexpr parameters of a kernel only from evaluated annotations

import math
from typing import List

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _mm(a, b, acc, USE_DOT: tl.constexpr):
    """``acc + a @ b``: ``tl.dot`` in IEEE float32, a broadcast sum otherwise (float64)."""
    if USE_DOT:
        out = tl.dot(a, b, acc, input_precision="ieee")
    else:
        out = acc + tl.sum(a[:, :, None] * b[None, :, :], axis=1)
    return out


@triton.jit
def _entry(p, l, m, n, P, C, L: tl.constexpr, SO3: tl.constexpr, PADDED: tl.constexpr):
    """Positions ``(rows, degrees)`` of ``f^l_{mn}`` of the signals ``p = b C + c`` (``P`` of
    them) in a degree-blocked spectrum (blocks of ``2l + 1`` or, padded, ``2L - 1``)."""
    p = p.to(tl.int64)[:, None]
    b = p // C
    c = p % C
    if PADDED:
        d = 2 * L - 1 + 0 * l
        u = m + L - 1 + 0 * l
        v = n + L - 1 + 0 * l
        if SO3:
            start = l * (d * d)
        else:
            start = l * d
    else:
        d = 2 * l + 1
        u = m + l
        v = n + l
        if SO3:
            start = (l * (4 * l * l - 1)) // 3
        else:
            start = l * l
    d = d[None, :]
    if SO3:
        out = ((b * d + u[None, :]) * C + c) * d + v[None, :]
    else:
        out = (b * d + u[None, :]) * C + c
    return start.to(tl.int64)[None, :] * P + out


@triton.jit
def _to_spectrum(x_ptr, t_ptr, out_ptr, P, C, J: tl.constexpr, L: tl.constexpr, S: tl.constexpr,
                 N_FREQ: tl.constexpr, PLANE: tl.constexpr, SO3: tl.constexpr,
                 PADDED: tl.constexpr, BP: tl.constexpr, BJ: tl.constexpr, BL: tl.constexpr,
                 USE_DOT: tl.constexpr):
    """``out[(l, b, m, c, n)] = sum_j x[p, j, m mod S, n mod S] t[(m, n), j, l]`` for one
    frequency ``(m, n)`` and a tile of signals ``p = b C + c``."""
    pid = tl.program_id(0)
    freq = pid % N_FREQ
    tile = pid // N_FREQ
    W = 2 * L - 1
    if SO3:
        m = freq // W - (L - 1)
        n = freq % W - (L - 1)
        pos = ((m + S) % S) * S + (n + S) % S
    else:
        m = freq - (L - 1)
        n = m * 0
        pos = (m + S) % S
    l_min = tl.maximum(tl.where(m < 0, -m, m), tl.where(n < 0, -n, n))
    p = tile * BP + tl.arange(0, BP)
    p_ok = p < P
    rows = p.to(tl.int64) * (J * PLANE) + pos
    l = tl.arange(0, BL)
    acc_r = tl.zeros([BP, BL], dtype=out_ptr.dtype.element_ty)
    acc_i = tl.zeros([BP, BL], dtype=out_ptr.dtype.element_ty)
    for j0 in range(0, J, BJ):
        j = j0 + tl.arange(0, BJ)
        j_ok = j < J
        mask = p_ok[:, None] & j_ok[None, :]
        ptrs = x_ptr + (rows[:, None] + (j * PLANE)[None, :]) * 2
        xr = tl.load(ptrs, mask=mask, other=0.0)
        xi = tl.load(ptrs + 1, mask=mask, other=0.0)
        t = tl.load(t_ptr + freq * (J * L) + j[:, None] * L + l[None, :],
                    mask=j_ok[:, None] & (l < L)[None, :], other=0.0)
        acc_r = _mm(xr, t, acc_r, USE_DOT)
        acc_i = _mm(xi, t, acc_i, USE_DOT)
    if PADDED:
        # the padding of the blocks too: the table is zero for l < l_min
        mask = p_ok[:, None] & (l < L)[None, :]
    else:
        mask = p_ok[:, None] & ((l >= l_min) & (l < L))[None, :]
    ptrs = out_ptr + _entry(p, l, m, n, P, C, L, SO3, PADDED) * 2
    tl.store(ptrs, acc_r, mask=mask)
    tl.store(ptrs + 1, acc_i, mask=mask)


@triton.jit
def _to_grid(z_ptr, t_ptr, out_ptr, P, C, J: tl.constexpr, L: tl.constexpr, S: tl.constexpr,
             PLANE: tl.constexpr, SO3: tl.constexpr, PADDED: tl.constexpr, BP: tl.constexpr,
             BJ: tl.constexpr, BL: tl.constexpr, USE_DOT: tl.constexpr):
    """``out[p, j, m mod S, n mod S] = sum_l z[(l, b, m, c, n)] t[(m, n), j, l]`` for one grid
    position and a tile of signals ``p = b C + c``; zero at the positions off the band."""
    pid = tl.program_id(0)
    pos = pid % PLANE
    tile = pid // PLANE
    W = 2 * L - 1
    if SO3:
        mp = pos // S
        np_ = pos % S
        m = tl.where(mp < L, mp, mp - S)
        n = tl.where(np_ < L, np_, np_ - S)
        band = ((mp < L) | (mp > S - L)) & ((np_ < L) | (np_ > S - L))
        freq = (m + L - 1) * W + n + L - 1
    else:
        m = tl.where(pos < L, pos, pos - S)
        n = m * 0
        band = (pos < L) | (pos > S - L)
        freq = m + L - 1
    p = tile * BP + tl.arange(0, BP)
    p_ok = p < P
    j = tl.arange(0, BJ)
    j_ok = j < J
    acc_r = tl.zeros([BP, BJ], dtype=out_ptr.dtype.element_ty)
    acc_i = tl.zeros([BP, BJ], dtype=out_ptr.dtype.element_ty)
    if band:
        l_min = tl.maximum(tl.where(m < 0, -m, m), tl.where(n < 0, -n, n))
        for l0 in range(0, L, BL):
            l = l0 + tl.arange(0, BL)
            l_ok = (l >= l_min) & (l < L)
            mask = p_ok[:, None] & l_ok[None, :]
            ptrs = z_ptr + _entry(p, l, m, n, P, C, L, SO3, PADDED) * 2
            zr = tl.load(ptrs, mask=mask, other=0.0)
            zi = tl.load(ptrs + 1, mask=mask, other=0.0)
            t = tl.load(t_ptr + freq * (J * L) + j[None, :] * L + l[:, None],
                        mask=l_ok[:, None] & j_ok[None, :], other=0.0)
            acc_r = _mm(zr, t, acc_r, USE_DOT)
            acc_i = _mm(zi, t, acc_i, USE_DOT)
    ptrs = out_ptr + ((p.to(tl.int64) * (J * PLANE) + pos)[:, None] + (j * PLANE)[None, :]) * 2
    mask = p_ok[:, None] & j_ok[None, :]
    tl.store(ptrs, acc_r, mask=mask)
    tl.store(ptrs + 1, acc_i, mask=mask)


_WARPS = 4


def _pow2(n: int, minimum: int = 1) -> int:
    p = minimum
    while p < n:
        p *= 2
    return p


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def compact_size(L: int, so3: bool) -> int:
    """Entries of a compact spectrum of bandwidth ``L``: ``L (4 L^2 - 1) / 3`` or ``L^2``."""
    return L * (4 * L * L - 1) // 3 if so3 else L * L


class SpectralPlan:
    """One transform between a grid and compact spectra, for the contraction kernels.

    Parameters
    ----------
    table : Tensor
        ``T[(m, n), j, l]``, shape ``((2L - 1)^2, J, L)`` (``SO(3)``) or ``(2L - 1, J, L)``
        (``S^2``), frequencies ordered ``m, n = -(L - 1) .. L - 1``; contiguous, in the
        precision of the signals.
    J : int
        Samples of ``beta``.
    L : int
        Bandwidth of the spectra.
    S : int
        Samples of each FFT axis.
    so3 : bool
        ``SO(3)`` grids (two FFT axes) or ``S^2`` grids (one).
    padded : bool, optional
        Blocks of ``2L - 1`` entries per index ``m`` and ``n`` for every degree, centered
        (``m + L - 1``) and zero outside ``|m|, |n| <= l``, instead of ``2l + 1``.
    """

    def __init__(self, table: Tensor, J: int, L: int, S: int, so3: bool, padded: bool = False):
        self.table, self.J, self.L, self.S, self.so3 = table, int(J), int(L), int(S), bool(so3)
        self.padded = bool(padded)
        if S < 2 * L:
            raise ValueError(f"an FFT axis of {S} samples cannot hold the band of bandwidth {L}")
        width = 2 * self.L - 1
        self.n_freq = width * width if self.so3 else width
        self.plane = self.S * self.S if self.so3 else self.S
        # entries per signal
        self.K = self.L * self.n_freq if self.padded else compact_size(self.L, self.so3)
        self.grid_dims = 4 if self.so3 else 3

    def grid_shape(self, lead) -> List[int]:
        """The real view of an FFT grid with leading dimensions ``lead``."""
        return [*lead, self.J, *([self.S] * (self.grid_dims - 2)), 2]


def _contraction_sizes(dtype: torch.dtype, J: int, L: int):
    """Tile sizes ``(BP, BJ, BL)`` of the two contractions and whether ``tl.dot`` runs."""
    if dtype == torch.float32:
        return dict(BP=32, BJ_SPEC=16 if J <= 16 else 32, BL_SPEC=_pow2(L, 16),
                    BJ_GRID=_pow2(J, 16), BL_GRID=16, USE_DOT=True)
    return dict(BP=32, BJ_SPEC=4, BL_SPEC=_pow2(L, 2), BJ_GRID=_pow2(J, 2), BL_GRID=4, USE_DOT=False)


def _rows(lead) -> tuple:
    """The signal count ``P`` and channel count ``C`` of leading dimensions ``(..., C)``."""
    return math.prod(lead), (lead[-1] if len(lead) else 1)


def _spectrum(x: Tensor, plan: SpectralPlan) -> Tensor:
    x = x.contiguous()
    lead = tuple(x.shape[:x.dim() - plan.grid_dims])
    n_rows, n_channels = _rows(lead)
    out = x.new_empty(n_rows * plan.K, 2)
    if n_rows:
        cfg = _contraction_sizes(x.dtype, plan.J, plan.L)
        grid = (plan.n_freq * _cdiv(n_rows, cfg["BP"]),)
        _to_spectrum[grid](x, plan.table, out, n_rows, n_channels, J=plan.J, L=plan.L, S=plan.S,
                           N_FREQ=plan.n_freq, PLANE=plan.plane, SO3=plan.so3,
                           PADDED=plan.padded, BP=cfg["BP"],
                           BJ=cfg["BJ_SPEC"], BL=cfg["BL_SPEC"], USE_DOT=cfg["USE_DOT"],
                           num_warps=_WARPS)
    return out


def _grid(z: Tensor, plan: SpectralPlan, lead) -> Tensor:
    z = z.contiguous()
    n_rows, n_channels = _rows(lead)
    out = z.new_empty(plan.grid_shape(lead))
    if n_rows:
        cfg = _contraction_sizes(z.dtype, plan.J, plan.L)
        grid = (plan.plane * _cdiv(n_rows, cfg["BP"]),)
        _to_grid[grid](z, plan.table, out, n_rows, n_channels, J=plan.J, L=plan.L, S=plan.S,
                       PLANE=plan.plane, SO3=plan.so3, PADDED=plan.padded, BP=cfg["BP"],
                       BJ=cfg["BJ_GRID"],
                       BL=cfg["BL_GRID"], USE_DOT=cfg["USE_DOT"], num_warps=_WARPS)
    return out


class _ToSpectrum(torch.autograd.Function):
    """The contraction of a forward transform; its backward is :class:`_ToGrid`."""

    @staticmethod
    def forward(ctx, x: Tensor, plan: SpectralPlan) -> Tensor:
        ctx.plan, ctx.lead = plan, tuple(x.shape[:x.dim() - plan.grid_dims])
        return _spectrum(x, plan)

    @staticmethod
    def backward(ctx, g: Tensor):
        return _ToGrid.apply(g, ctx.plan, ctx.lead), None


class _ToGrid(torch.autograd.Function):
    """The contraction of an inverse transform; its backward is :class:`_ToSpectrum`."""

    @staticmethod
    def forward(ctx, z: Tensor, plan: SpectralPlan, lead) -> Tensor:
        ctx.plan = plan
        return _grid(z, plan, lead)

    @staticmethod
    def backward(ctx, g: Tensor):
        return _ToSpectrum.apply(g, ctx.plan), None, None


def to_spectrum(x: Tensor, plan: SpectralPlan) -> Tensor:
    """Degree-blocked spectra from the FFTs of grid signals, differentiable to any order.

    Parameters
    ----------
    x : Tensor
        The real view of the FFT, ``(..., C, J, S, S, 2)`` (``SO(3)``) or ``(..., C, J, S, 2)``
        (``S^2``); the leading dimensions before the channels form the batch.
    plan : SpectralPlan
        The table (quadrature weights folded in) and sizes.

    Returns
    -------
    Tensor
        ``(B C K, 2)``.
    """
    return _ToSpectrum.apply(x, plan)


def to_grid(z: Tensor, plan: SpectralPlan, lead) -> Tensor:
    """The FFT grids of the inverse transform of degree-blocked spectra, differentiable to any order.

    Parameters
    ----------
    z : Tensor
        Degree-blocked spectra ``(B C K, 2)``.
    plan : SpectralPlan
        The table (degree factors folded in) and sizes.
    lead : tuple of int
        The leading dimensions ``(..., C)`` of the grids.

    Returns
    -------
    Tensor
        The real view of the FFT grid, ``(*lead, J, S, S, 2)`` or ``(*lead, J, S, 2)``, zero off
        the band.
    """
    return _ToGrid.apply(z, plan, tuple(int(v) for v in lead))
