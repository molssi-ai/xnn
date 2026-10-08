"""Spherical CNN (Cohen, Geiger, Koehler, Welling, ICLR 2018).

Rotation-equivariant convolutions for signals on the sphere ``S^2`` and on
the rotation group ``SO(3)``, built from the manuscript (arXiv:1801.10130)
and the conventions of its reference implementation
(jonas-koehler/s2cnn), without using its code:

* signals are sampled on the Driscoll-Healy grids of the paper: ``S^2`` at
  ``beta_j = pi (2j + 1) / 4b``, ``alpha_k = 2 pi k / 2b`` (``j, k < 2b``) and
  ``SO(3)`` at the ZYZ Euler angles ``(beta_j, alpha_k, gamma_l)`` for a
  bandwidth ``b`` (Appendix A; the axes of a signal tensor are ``beta, alpha``
  and ``beta, alpha, gamma``);
* the generalized Fourier transform (eq 8, 21) expands a signal in the Wigner
  D-functions ``D^l_{mn}(alpha, beta, gamma) = e^{i m alpha} d^l_{mn}(beta)
  e^{i n gamma}`` (``Y^l_m = D^l_{m0}`` on the sphere, footnote 3), with the
  Wigner d-matrices ``d^l(beta) = exp(-i beta J_y)`` evaluated by the
  eigendecomposition of the spin matrix ``J_y`` (exact in float64 at any
  degree) and the quadrature weights of the grid, and its inverse is eq 9;
  the transforms are a translational FFT over the ``alpha`` (and ``gamma``)
  axis followed by a contraction over ``beta`` with the sampled ``d``
  (Sec. 4.1);
* the correlations ``[psi * f](R) = <L_R psi, f>`` (eq 4 and 6) are computed
  in the spectrum by the Fourier theorems of Appendix D, ``f^l psi^l+`` (an
  outer product of ``S^2`` spectra, a matrix product of ``SO(3)`` spectra),
  with the filter ``psi`` parameterized by its values on a small set of grid
  points (a local patch around the north pole or identity, or a ring around
  the equator) and transformed with the same basis functions;
* :func:`so3_integrate` is the invariant integral over ``SO(3)`` (eq 11),
  :func:`so3_rotate` / :func:`s2_rotate` rotate a signal exactly in the
  spectrum (``L_R f`` of eq 1 and 5), which is what the equivariance checks
  of Sec. 5.1 use.

:class:`SphericalCNN` is the interatomic potential of Sec. 5.4: every atom
carries the spherical signals of :class:`~xnn.cnn.featurizers.SphericalGrid`
(one Coulomb-like potential channel per species on a sphere around the
atom), a stack of ResNet blocks of ``S^2`` / ``SO(3)`` correlations maps
them to features on ``SO(3)``, the integral over ``SO(3)`` gives the
invariant per-atom features, and the readout the energy; the paper's
DeepSet readout is available as an option. Needs no e3nn.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from xnn.common.data import AtomicGraph
from xnn.common.models.ops import make_activation
from xnn.common.models.registry import register_model
from ..featurizers import SphericalGrid
from .base import VoxelPotential


def so3_betas(b: int, dtype: torch.dtype = torch.float64) -> Tensor:
    """The ``2b`` polar angles ``beta_j = pi (2j + 1) / 4b`` of the grids."""
    return (torch.arange(2 * b, dtype=dtype) + 0.5) * math.pi / (2 * b)


def so3_alphas(b: int, dtype: torch.dtype = torch.float64) -> Tensor:
    """The ``2b`` azimuths ``alpha_k = 2 pi k / 2b`` (also the ``gamma`` grid)."""
    return torch.arange(2 * b, dtype=dtype) * (2 * math.pi / (2 * b))


@lru_cache(maxsize=None)
def quadrature_weights(b: int) -> Tensor:
    """Quadrature weights ``w_j`` of the ``SO(3)`` grid for the normalized Haar
    measure (eq 11): ``int f dR = sum_{j,k,l} w_j f(beta_j, alpha_k, gamma_l)``.

    The Driscoll-Healy weights ``(2/b) sin(beta_j) sum_{k<b} sin((2k+1)
    beta_j) / (2k+1)`` integrate every polynomial of degree below ``2b`` in
    ``cos beta`` exactly over ``sin(beta) d beta``; they are normalized by the
    ``2 (2b)^2`` of the measure and of the ``alpha`` / ``gamma`` sums.

    Parameters
    ----------
    b : int
        Bandwidth of the grid.

    Returns
    -------
    Tensor
        Shape ``(2b,)``, float64, summing to ``1 / (2b)^2``.
    """
    beta = so3_betas(b)
    k = torch.arange(b, dtype=torch.float64)
    series = (torch.sin((2 * k + 1)[None, :] * beta[:, None]) / (2 * k + 1)[None, :]).sum(1)
    return (2.0 / b) * torch.sin(beta) * series / (2.0 * (2 * b) ** 2)


@lru_cache(maxsize=None)
def _spin_y_eigen(l: int) -> Tuple[Tensor, Tensor]:
    """Eigenvalues (the integers ``-l .. l``) and eigenvectors of ``J_y`` in the
    ``|l m>`` basis ordered ``m = -l .. l``."""
    m = torch.arange(-l, l + 1, dtype=torch.float64)
    raise_amp = torch.sqrt(l * (l + 1) - m[:-1] * (m[:-1] + 1))      # <m+1| J_+ |m>
    j_plus = torch.zeros(2 * l + 1, 2 * l + 1, dtype=torch.complex128)
    j_plus[torch.arange(1, 2 * l + 1), torch.arange(0, 2 * l)] = raise_amp.to(torch.complex128)
    j_y = (j_plus - j_plus.conj().T) / 2j
    evals, evecs = torch.linalg.eigh(j_y)
    return evals.round(), evecs


def wigner_d(l: int, beta: Tensor) -> Tensor:
    """Wigner d-matrix ``d^l_{mn}(beta) = <l m| exp(-i beta J_y) |l n>``.

    Rows and columns run over ``m, n = -l .. l``; the matrix is real and
    orthogonal, ``d^l(beta_1) d^l(beta_2) = d^l(beta_1 + beta_2)``.

    Parameters
    ----------
    l : int
        Degree.
    beta : Tensor
        Angles of any shape.

    Returns
    -------
    Tensor
        Shape ``(*beta.shape, 2l + 1, 2l + 1)`` in float64.
    """
    evals, evecs = _spin_y_eigen(l)
    beta = torch.as_tensor(beta, dtype=torch.float64)
    phase = torch.exp(-1j * beta[..., None] * evals)                           # (..., 2l+1)
    d = torch.einsum("mk,...k,nk->...mn", evecs, phase, evecs.conj())
    return d.real


@lru_cache(maxsize=None)
def wigner_d_table(b_grid: int, b_spec: int) -> Tensor:
    """``d^l_{mn}(beta_j)`` on the grid of bandwidth ``b_grid`` for every
    ``l < b_spec``, as a zero-padded table ``[j, l, m + b_spec - 1, n + b_spec - 1]``
    of shape ``(2 b_grid, b_spec, 2 b_spec - 1, 2 b_spec - 1)`` (float64)."""
    table = torch.zeros(2 * b_grid, b_spec, 2 * b_spec - 1, 2 * b_spec - 1, dtype=torch.float64)
    betas = so3_betas(b_grid)
    for l in range(b_spec):
        o = b_spec - 1 - l
        table[:, l, o:o + 2 * l + 1, o:o + 2 * l + 1] = wigner_d(l, betas)
    return table


def _frequency_index(b_spec: int, size: int) -> Tensor:
    """Positions of the frequencies ``-(b_spec - 1) .. b_spec - 1`` on an FFT
    axis of ``size`` samples."""
    m = torch.arange(-(b_spec - 1), b_spec)
    return torch.remainder(m, size)


def _complex(real: Tensor, imag: Tensor) -> Tensor:
    return torch.complex(real, imag)


class S2Transform(nn.Module):
    """Spherical harmonic transform between the ``S^2`` grid of bandwidth
    ``b_grid`` and spectra of bandwidth ``b_spec`` (``b_spec <= b_grid``).

    A spectrum is a complex tensor ``(..., b_spec, 2 b_spec - 1)`` holding
    ``f^l_m`` at ``[l, m + b_spec - 1]`` (zero where ``|m| > l``).

    Parameters
    ----------
    b_grid : int
        Bandwidth of the grid (``2 b_grid`` samples per axis).
    b_spec : int
        Bandwidth of the spectrum (degrees ``l < b_spec``).
    """

    def __init__(self, b_grid: int, b_spec: int):
        super().__init__()
        if b_spec > b_grid:
            raise ValueError(f"the spectrum bandwidth {b_spec} exceeds the grid bandwidth {b_grid}")
        self.b_grid, self.b_spec = int(b_grid), int(b_spec)
        dtype = torch.get_default_dtype()
        self.register_buffer("d", wigner_d_table(b_grid, b_spec)[:, :, :, b_spec - 1].to(dtype))
        self.register_buffer("weights", (quadrature_weights(b_grid) * (2 * b_grid)).to(dtype))
        self.register_buffer("degree", (2 * torch.arange(b_spec) + 1).to(dtype))
        self.register_buffer("m_index", _frequency_index(b_spec, 2 * b_grid))

    def analyze(self, x: Tensor) -> Tensor:
        """Spectrum ``f^l_m = int f conj(Y^l_m) dx`` of grid signals ``(..., 2b, 2b)``."""
        spec = torch.fft.fft(x, dim=-1)[..., self.m_index]                # (..., beta, m)
        return torch.einsum("...jm,jlm,j->...lm", spec, self.d.to(spec.dtype), self.weights.to(spec.dtype))

    def synthesize(self, f_hat: Tensor, real: bool = True) -> Tensor:
        """Grid signal ``f = sum_l (2l + 1) sum_m f^l_m Y^l_m`` of spectra ``(..., b_spec, 2 b_spec - 1)``."""
        rows = torch.einsum("...lm,jlm,l->...jm", f_hat, self.d.to(f_hat.dtype), self.degree.to(f_hat.dtype))
        full = rows.new_zeros(*rows.shape[:-1], 2 * self.b_grid)
        full[..., self.m_index] = rows
        out = torch.fft.ifft(full, dim=-1) * (2 * self.b_grid)
        return out.real if real else out


class SO3Transform(nn.Module):
    """Fourier transform between the ``SO(3)`` grid of bandwidth ``b_grid`` and
    spectra of bandwidth ``b_spec`` (``b_spec <= b_grid``).

    A spectrum is a complex tensor ``(..., b_spec, 2 b_spec - 1, 2 b_spec - 1)``
    holding ``f^l_{mn}`` at ``[l, m + b_spec - 1, n + b_spec - 1]``.

    Parameters
    ----------
    b_grid : int
        Bandwidth of the grid.
    b_spec : int
        Bandwidth of the spectrum.
    """

    def __init__(self, b_grid: int, b_spec: int):
        super().__init__()
        if b_spec > b_grid:
            raise ValueError(f"the spectrum bandwidth {b_spec} exceeds the grid bandwidth {b_grid}")
        self.b_grid, self.b_spec = int(b_grid), int(b_spec)
        dtype = torch.get_default_dtype()
        self.register_buffer("d", wigner_d_table(b_grid, b_spec).to(dtype))
        self.register_buffer("weights", quadrature_weights(b_grid).to(dtype))
        self.register_buffer("degree", (2 * torch.arange(b_spec) + 1).to(dtype))
        self.register_buffer("m_index", _frequency_index(b_spec, 2 * b_grid))

    def analyze(self, x: Tensor) -> Tensor:
        """Spectrum ``f^l_{mn} = int f conj(D^l_{mn}) dR`` of grid signals ``(..., 2b, 2b, 2b)``."""
        spec = torch.fft.fft2(x, dim=(-2, -1))[..., self.m_index, :][..., :, self.m_index]
        return torch.einsum("...jmn,jlmn,j->...lmn", spec, self.d.to(spec.dtype), self.weights.to(spec.dtype))

    def synthesize(self, f_hat: Tensor, real: bool = True) -> Tensor:
        """Grid signal ``f = sum_l (2l + 1) sum_{mn} f^l_{mn} D^l_{mn}`` (eq 9)."""
        rows = torch.einsum("...lmn,jlmn,l->...jmn", f_hat, self.d.to(f_hat.dtype), self.degree.to(f_hat.dtype))
        full = rows.new_zeros(*rows.shape[:-2], 2 * self.b_grid, 2 * self.b_grid)
        full[..., self.m_index[:, None], self.m_index[None, :]] = rows
        out = torch.fft.ifft2(full, dim=(-2, -1)) * (2 * self.b_grid) ** 2
        return out.real if real else out


def so3_integrate(x: Tensor) -> Tensor:
    """Integral of ``SO(3)`` signals ``(..., 2b, 2b, 2b)`` over the normalized
    Haar measure (eq 11): the invariant pooling of the paper's networks."""
    b = x.shape[-1] // 2
    w = quadrature_weights(b).to(device=x.device, dtype=x.dtype)
    return torch.einsum("...jkl,j->...", x, w)


def s2_near_identity_grid(max_beta: float = math.pi / 8, n_alpha: int = 8, n_beta: int = 3) -> Tensor:
    """Points ``(beta, alpha)`` on ``n_beta`` rings around the north pole, the
    local filter support of the paper's networks.

    Returns
    -------
    Tensor
        Shape ``(n_alpha * n_beta, 2)``, float64.
    """
    beta = torch.arange(1, n_beta + 1, dtype=torch.float64) * max_beta / n_beta
    alpha = torch.arange(n_alpha, dtype=torch.float64) * (2 * math.pi / n_alpha)
    bb, aa = torch.meshgrid(beta, alpha, indexing="ij")
    return torch.stack([bb.flatten(), aa.flatten()], dim=1)


def s2_equatorial_grid(max_beta: float = 0.0, n_alpha: int = 32, n_beta: int = 1) -> Tensor:
    """Points ``(beta, alpha)`` on ``n_beta`` rings around the equator (the
    non-local filter support of the SHREC17 network)."""
    beta = torch.linspace(math.pi / 2 - max_beta, math.pi / 2 + max_beta, n_beta, dtype=torch.float64)
    alpha = torch.arange(n_alpha, dtype=torch.float64) * (2 * math.pi / n_alpha)
    bb, aa = torch.meshgrid(beta, alpha, indexing="ij")
    return torch.stack([bb.flatten(), aa.flatten()], dim=1)


def so3_near_identity_grid(max_beta: float = math.pi / 8, max_gamma: float = 2 * math.pi,
                           n_alpha: int = 8, n_beta: int = 3, n_gamma: Optional[int] = None) -> Tensor:
    """Rotations ``(beta, alpha, gamma)`` on rings around the identity: every
    rotation of a ring is at the same distance from the identity.

    Returns
    -------
    Tensor
        Shape ``(n_alpha * n_beta * n_gamma, 3)``, float64.
    """
    n_gamma = n_alpha if n_gamma is None else n_gamma
    beta = torch.arange(1, n_beta + 1, dtype=torch.float64) * max_beta / n_beta
    alpha = torch.arange(n_alpha, dtype=torch.float64) * (2 * math.pi / n_alpha)
    pre_gamma = torch.linspace(-max_gamma, max_gamma, n_gamma, dtype=torch.float64)
    bb, aa, cc = torch.meshgrid(beta, alpha, pre_gamma, indexing="ij")
    return torch.stack([bb.flatten(), aa.flatten(), (cc - aa).flatten()], dim=1)


def so3_equatorial_grid(max_beta: float = 0.0, max_gamma: float = math.pi / 8, n_alpha: int = 32,
                        n_beta: int = 1, n_gamma: int = 2) -> Tensor:
    """Rotations ``(beta, alpha, gamma)`` on rings around the equator."""
    beta = torch.linspace(math.pi / 2 - max_beta, math.pi / 2 + max_beta, n_beta, dtype=torch.float64)
    alpha = torch.arange(n_alpha, dtype=torch.float64) * (2 * math.pi / n_alpha)
    gamma = torch.linspace(-max_gamma, max_gamma, n_gamma, dtype=torch.float64)
    bb, aa, cc = torch.meshgrid(beta, alpha, gamma, indexing="ij")
    return torch.stack([bb.flatten(), aa.flatten(), cc.flatten()], dim=1)


def s2_kernel_basis(points: Tensor, b: int) -> Tuple[Tensor, Tensor]:
    """``2b conj(Y^l_m)`` at filter points ``(beta, alpha)``: the matrix that
    turns filter values into the filter spectrum of an ``S^2`` correlation.

    Returns
    -------
    (Tensor, Tensor)
        Real and imaginary parts, each ``(n_points, b, 2b - 1)`` in float64.
    """
    points = points.to(torch.float64)
    beta, alpha = points[:, 0], points[:, 1]
    m = torch.arange(-(b - 1), b, dtype=torch.float64)
    basis = torch.zeros(points.shape[0], b, 2 * b - 1, dtype=torch.float64)
    for l in range(b):
        o = b - 1 - l
        basis[:, l, o:o + 2 * l + 1] = wigner_d(l, beta)[:, :, l]          # d^l_{m0}(beta)
    phase = torch.exp(-1j * alpha[:, None] * m[None, :])                   # conj(e^{i m alpha})
    full = (2 * b) * basis * phase[:, None, :]
    return full.real.contiguous(), full.imag.contiguous()


def so3_kernel_basis(points: Tensor, b: int) -> Tuple[Tensor, Tensor]:
    """``conj(D^l_{mn})`` at filter rotations ``(beta, alpha, gamma)``: the
    matrix that turns filter values into the filter spectrum of an ``SO(3)``
    correlation.

    Returns
    -------
    (Tensor, Tensor)
        Real and imaginary parts, each ``(n_points, b, 2b - 1, 2b - 1)`` in float64.
    """
    points = points.to(torch.float64)
    beta, alpha, gamma = points[:, 0], points[:, 1], points[:, 2]
    m = torch.arange(-(b - 1), b, dtype=torch.float64)
    basis = torch.zeros(points.shape[0], b, 2 * b - 1, 2 * b - 1, dtype=torch.float64)
    for l in range(b):
        o = b - 1 - l
        basis[:, l, o:o + 2 * l + 1, o:o + 2 * l + 1] = wigner_d(l, beta)
    phase = torch.exp(-1j * (alpha[:, None, None] * m[None, :, None] + gamma[:, None, None] * m[None, None, :]))
    full = basis * phase[:, None, :, :]
    return full.real.contiguous(), full.imag.contiguous()


class S2Convolution(nn.Module):
    """``S^2`` correlation layer (eq 4, computed by the Fourier theorem of Appendix D).

    The filter of every (input, output) channel pair is a set of values on the
    ``points`` of the sphere; its spectrum is formed with
    :func:`s2_kernel_basis`, multiplied with the input spectrum as the outer
    product ``f^l psi^l+`` and transformed back onto the ``SO(3)`` grid of
    bandwidth ``b_out``. The filter values are drawn uniformly in ``[-1, 1]``
    and scaled by ``1 / sqrt(n_points n_in b_out^4 / b_in^2)``, the
    initialization of the reference implementation.

    Parameters
    ----------
    n_in, n_out : int
        Input and output channels.
    b_in, b_out : int
        Bandwidth of the input (``S^2`` grid of ``2 b_in`` samples per axis)
        and of the output (``SO(3)`` grid of ``2 b_out`` samples per axis;
        ``b_out <= b_in``).
    points : Tensor
        Filter support, ``(n_points, 2)`` pairs ``(beta, alpha)``.
    """

    def __init__(self, n_in: int, n_out: int, b_in: int, b_out: int, points: Tensor):
        super().__init__()
        self.n_in, self.n_out, self.b_in, self.b_out = int(n_in), int(n_out), int(b_in), int(b_out)
        self.analysis = S2Transform(b_in, b_out)
        self.synthesis = SO3Transform(b_out, b_out)
        real, imag = s2_kernel_basis(points, b_out)
        dtype = torch.get_default_dtype()
        self.register_buffer("basis_real", real.to(dtype))
        self.register_buffer("basis_imag", imag.to(dtype))
        self.register_buffer("points", points.to(dtype))
        self.kernel = nn.Parameter(torch.empty(n_in, n_out, points.shape[0]).uniform_(-1, 1))
        self.bias = nn.Parameter(torch.zeros(1, n_out, 1, 1, 1))
        self.scaling = 1.0 / math.sqrt(points.shape[0] * n_in * self.b_out ** 4 / self.b_in ** 2)

    def filter_spectrum(self) -> Tensor:
        """``psi^l_m`` of every channel pair, ``(n_in, n_out, b_out, 2 b_out - 1)``."""
        basis = _complex(self.basis_real, self.basis_imag)
        return torch.einsum("iop,plm->iolm", (self.kernel * self.scaling).to(basis.dtype), basis)

    def forward(self, x: Tensor) -> Tensor:
        """``(B, n_in, 2 b_in, 2 b_in)`` -> ``(B, n_out, 2 b_out, 2 b_out, 2 b_out)``."""
        f_hat = self.analysis.analyze(x)                                    # (B, i, l, m)
        z = torch.einsum("bilm,ioln->bolmn", f_hat, self.filter_spectrum().conj())
        return self.synthesis.synthesize(z) + self.bias

    def extra_repr(self) -> str:
        return (f"n_in={self.n_in}, n_out={self.n_out}, b_in={self.b_in}, b_out={self.b_out}, "
                f"n_points={self.points.shape[0]}")


class SO3Convolution(nn.Module):
    """``SO(3)`` correlation layer (eq 6, computed as the matrix product
    ``f^l psi^l+`` of Appendix D).

    Parameters
    ----------
    n_in, n_out : int
        Input and output channels.
    b_in, b_out : int
        Bandwidth of the input and output ``SO(3)`` grids (``b_out <= b_in``).
    points : Tensor
        Filter support, ``(n_points, 3)`` triples ``(beta, alpha, gamma)``.
    """

    def __init__(self, n_in: int, n_out: int, b_in: int, b_out: int, points: Tensor):
        super().__init__()
        self.n_in, self.n_out, self.b_in, self.b_out = int(n_in), int(n_out), int(b_in), int(b_out)
        self.analysis = SO3Transform(b_in, b_out)
        self.synthesis = SO3Transform(b_out, b_out)
        real, imag = so3_kernel_basis(points, b_out)
        dtype = torch.get_default_dtype()
        self.register_buffer("basis_real", real.to(dtype))
        self.register_buffer("basis_imag", imag.to(dtype))
        self.register_buffer("points", points.to(dtype))
        self.kernel = nn.Parameter(torch.empty(n_in, n_out, points.shape[0]).uniform_(-1, 1))
        self.bias = nn.Parameter(torch.zeros(1, n_out, 1, 1, 1))
        self.scaling = 1.0 / math.sqrt(points.shape[0] * n_in * self.b_out ** 3 / self.b_in ** 3)

    def filter_spectrum(self) -> Tensor:
        """``psi^l_{mn}`` of every channel pair, ``(n_in, n_out, b_out, 2 b_out - 1, 2 b_out - 1)``."""
        basis = _complex(self.basis_real, self.basis_imag)
        return torch.einsum("iop,plmn->iolmn", (self.kernel * self.scaling).to(basis.dtype), basis)

    def forward(self, x: Tensor) -> Tensor:
        """``(B, n_in, 2 b_in, 2 b_in, 2 b_in)`` -> ``(B, n_out, 2 b_out, 2 b_out, 2 b_out)``."""
        f_hat = self.analysis.analyze(x)                                    # (B, i, l, m, n)
        z = torch.einsum("bilmn,iolkn->bolmk", f_hat, self.filter_spectrum().conj())
        return self.synthesis.synthesize(z) + self.bias

    def extra_repr(self) -> str:
        return (f"n_in={self.n_in}, n_out={self.n_out}, b_in={self.b_in}, b_out={self.b_out}, "
                f"n_points={self.points.shape[0]}")


def wigner_D(l: int, alpha: Tensor, beta: Tensor, gamma: Tensor) -> Tensor:
    """``D^l_{mn}(alpha, beta, gamma) = e^{i m alpha} d^l_{mn}(beta) e^{i n gamma}``
    (complex128, ``(..., 2l + 1, 2l + 1)``), the basis functions of the
    ``SO(3)`` Fourier transform in the convention of the reference code."""
    m = torch.arange(-l, l + 1, dtype=torch.float64)
    d = wigner_d(l, beta).to(torch.complex128)
    left = torch.exp(1j * torch.as_tensor(alpha, dtype=torch.float64)[..., None] * m)
    right = torch.exp(1j * torch.as_tensor(gamma, dtype=torch.float64)[..., None] * m)
    return left[..., :, None] * d * right[..., None, :]


def _rotation_blocks(b: int, alpha, beta, gamma) -> Tensor:
    """``conj(D^l(alpha, beta, gamma))`` for ``l < b`` as a zero-padded ``(b, 2b-1, 2b-1)`` table."""
    out = torch.zeros(b, 2 * b - 1, 2 * b - 1, dtype=torch.complex128)
    a, be, g = (torch.as_tensor(v, dtype=torch.float64) for v in (alpha, beta, gamma))
    for l in range(b):
        o = b - 1 - l
        out[l, o:o + 2 * l + 1, o:o + 2 * l + 1] = wigner_D(l, a, be, g).conj()
    return out


def so3_rotate(x: Tensor, alpha: float, beta: float, gamma: float) -> Tensor:
    """``[L_R f](Q) = f(R^{-1} Q)`` (eq 5) of ``SO(3)`` signals ``(..., 2b, 2b, 2b)``,
    for ``R = Z(alpha) Y(beta) Z(gamma)``, computed exactly in the spectrum."""
    b = x.shape[-1] // 2
    tr = SO3Transform(b, b).to(x.device)
    f_hat = tr.analyze(x)
    blocks = _rotation_blocks(b, alpha, beta, gamma).to(device=x.device, dtype=f_hat.dtype)
    return tr.synthesize(torch.einsum("lmk,...lkn->...lmn", blocks, f_hat))


def s2_rotate(x: Tensor, alpha: float, beta: float, gamma: float) -> Tensor:
    """``[L_R f](x) = f(R^{-1} x)`` (eq 1) of spherical signals ``(..., 2b, 2b)``,
    computed exactly in the spectrum."""
    b = x.shape[-1] // 2
    tr = S2Transform(b, b).to(x.device)
    f_hat = tr.analyze(x)
    blocks = _rotation_blocks(b, alpha, beta, gamma).to(device=x.device, dtype=f_hat.dtype)
    return tr.synthesize(torch.einsum("lmk,...lk->...lm", blocks, f_hat))


def euler_to_matrix(alpha, beta, gamma) -> Tensor:
    """The rotation matrix ``Z(alpha) Y(beta) Z(gamma)`` (eq 10), float64."""
    a, b, g = (float(v) for v in (alpha, beta, gamma))

    def z(t):
        return torch.tensor([[math.cos(t), -math.sin(t), 0.0], [math.sin(t), math.cos(t), 0.0],
                             [0.0, 0.0, 1.0]], dtype=torch.float64)

    y = torch.tensor([[math.cos(b), 0.0, math.sin(b)], [0.0, 1.0, 0.0],
                      [-math.sin(b), 0.0, math.cos(b)]], dtype=torch.float64)
    return z(a) @ y @ z(g)


def matrix_to_euler(R: Tensor) -> Tuple[float, float, float]:
    """ZYZ Euler angles ``(alpha, beta, gamma)`` of a rotation matrix."""
    R = torch.as_tensor(R, dtype=torch.float64)
    beta = math.acos(max(-1.0, min(1.0, float(R[2, 2]))))
    if abs(math.sin(beta)) < 1e-9:
        # gimbal lock: Z(alpha) Y(0) Z(gamma) = Z(alpha + gamma) and
        # Z(alpha) Y(pi) Z(gamma) = Z(alpha - gamma) Y(pi); put the whole angle in alpha
        sign = 1.0 if float(R[2, 2]) > 0 else -1.0
        return math.atan2(sign * float(R[1, 0]), sign * float(R[0, 0])), beta, 0.0
    alpha = math.atan2(float(R[1, 2]), float(R[0, 2]))
    gamma = math.atan2(float(R[2, 1]), -float(R[2, 0]))
    return alpha, beta, gamma


def s2_grid_points(b: int, dtype: torch.dtype = torch.float64) -> Tensor:
    """Unit vectors ``x(alpha, beta) = Z(alpha) Y(beta) n`` of the ``S^2`` grid
    (eq 12), shape ``(2b, 2b, 3)`` indexed ``[beta, alpha]``."""
    beta, alpha = so3_betas(b, dtype), so3_alphas(b, dtype)
    bb, aa = torch.meshgrid(beta, alpha, indexing="ij")
    return torch.stack([torch.sin(bb) * torch.cos(aa), torch.sin(bb) * torch.sin(aa), torch.cos(bb)], dim=-1)


class SphericalResBlock(nn.Module):
    """ResNet block of the QM7 network (Sec. 5.4): correlation, batch norm,
    activation, ``SO(3)`` correlation, batch norm, plus a shortcut of the input.

    Parameters
    ----------
    n_in, n_out : int
        Channels in and out.
    b_in, b_out : int
        Bandwidths in and out.
    s2_input : bool
        Whether the input is a spherical signal (first block) or an ``SO(3)``
        signal.
    n_alpha, n_beta, n_gamma, max_beta
        Filter support (near-identity grids of ``n_beta`` rings at polar
        angles up to ``max_beta`` with ``n_alpha`` azimuths, and ``n_gamma``
        values of ``gamma`` for ``SO(3)`` filters); ``n_alpha=None`` uses
        ``2 b_in``.
    normalization : str or None
        ``"batch"`` (the paper) or ``None``.
    activation : str
        Activation between the two correlations.
    """

    def __init__(self, n_in: int, n_out: int, b_in: int, b_out: int, s2_input: bool,
                 n_alpha: Optional[int] = None, n_beta: int = 2, n_gamma: int = 2,
                 max_beta: float = math.pi / 8, normalization: Optional[str] = "batch",
                 activation: str = "relu"):
        super().__init__()
        if normalization not in (None, "batch"):
            raise ValueError(f"normalization must be None or 'batch', got {normalization!r}")
        n_alpha = 2 * b_in if n_alpha is None else int(n_alpha)
        so3_points = so3_near_identity_grid(max_beta, 2 * math.pi, n_alpha, n_beta, n_gamma)
        if s2_input:
            self.conv1 = S2Convolution(n_in, n_out, b_in, b_out, s2_near_identity_grid(max_beta, n_alpha, n_beta))
            self.shortcut = S2Convolution(n_in, n_out, b_in, b_out, torch.zeros(1, 2, dtype=torch.float64))
        else:
            self.conv1 = SO3Convolution(n_in, n_out, b_in, b_out, so3_points)
            self.shortcut = (SO3Convolution(n_in, n_out, b_in, b_out, torch.zeros(1, 3, dtype=torch.float64))
                             if (n_in != n_out or b_in != b_out) else None)
        self.conv2 = SO3Convolution(n_out, n_out, b_out, b_out,
                                    so3_near_identity_grid(max_beta, 2 * math.pi, 2 * b_out, n_beta, n_gamma))
        self.norm1 = nn.BatchNorm3d(n_out) if normalization == "batch" else None
        self.norm2 = nn.BatchNorm3d(n_out) if normalization == "batch" else None
        self.act = make_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv1(x)
        if self.norm1 is not None:
            y = self.norm1(y)
        y = self.conv2(self.act(y))
        if self.norm2 is not None:
            y = self.norm2(y)
        return y + (x if self.shortcut is None else self.shortcut(x))


def default_schedule(n_features: int, n_blocks: int, bandwidth: int) -> Tuple[list, list]:
    """Channels and bandwidths of ``n_blocks`` ResNet blocks: channels growing
    linearly to ``n_features``, bandwidths falling from ``bandwidth`` to 2
    (the shape of Table 3 of the paper)."""
    feats = [max(1, round(n_features * (t + 1) / n_blocks)) for t in range(n_blocks)]
    bws = [max(2, round(bandwidth - (bandwidth - 2) * t / max(1, n_blocks - 1))) for t in range(n_blocks)]
    return feats, bws


@register_model("s2cnn")
class SphericalCNN(VoxelPotential):
    """Spherical CNN interatomic potential (Cohen *et al.*, ICLR 2018, Sec. 5.4).

    Every atom's environment is the stack of spherical signals of
    :class:`~xnn.cnn.featurizers.SphericalGrid` (the potential of the
    neighbors of each species on a sphere around the atom, sampled on the
    Driscoll-Healy grid of bandwidth ``bandwidth``). ``len(features)``
    :class:`SphericalResBlock` blocks map it to ``SO(3)`` signals of
    decreasing bandwidth, :func:`so3_integrate` pools them into invariant
    per-atom features, and the shared readout gives the per-atom energies. The
    defaults are Table 3 of the paper: five blocks with 20, 40, 60, 80, 160
    channels at bandwidths 10, 8, 6, 4, 2, batch normalization and ReLU. With
    ``set_readout`` the paper's DeepSet readout is used instead: the per-atom
    features pass through an MLP, are summed over the structure and mapped to
    the energy by a second MLP (the per-atom energy is then the structure's
    energy divided by its atom count, plus the atom's reference energy).

    Parameters
    ----------
    species : sequence of int, optional
        Atomic numbers with a potential channel, by default ``(1, 6, 7, 8, 16)``
        (the elements of QM7).
    cutoff : float, optional
        Neighbor-list radius of the potential sums (Angstrom), by default 10.0.
    radius : float, optional
        Radius of the sphere around each atom (Angstrom), by default 0.48 (half
        the shortest interatomic distance of QM7, so no atom lies on a sphere).
    bandwidth : int, optional
        Bandwidth of the input signals, by default 10.
    exponent : float, optional
        Power of the inverse distance of the potential, by default 1 (the
        paper's ``1/|x - p|``; the reference data script uses 2).
    features, bandwidths : sequence of int or None, optional
        Channels and output bandwidth of every block; ``None`` derives them from
        ``n_features``, ``n_blocks`` and ``bandwidth`` (:func:`default_schedule`).
    n_features : int, optional
        Channels of the last block when ``features`` is ``None``, by default 160.
    n_blocks : int, optional
        Number of blocks when ``features`` is ``None``, by default 5.
    n_alpha, n_beta, n_gamma, max_beta
        Filter support of the correlations (see :class:`SphericalResBlock`).
    normalization : str or None, optional
        ``"batch"`` (default, the paper) or ``None``.
    activation : str, optional
        Activation inside the blocks, by default ``"relu"`` (the paper).
    set_readout : sequence of int or None, optional
        ``(phi_hidden, latent, psi_hidden)`` of the DeepSet readout, e.g. the
        paper's ``(150, 100, 50)``; ``None`` (default) reads every atom out
        separately with the shared atom-wise readout.
    cutoff_fn : str or None, optional
        ``"cosine"`` multiplies each neighbor's potential by the cosine
        envelope at ``cutoff`` (continuous energies for dynamics); ``None``
        (default) is the paper's untruncated potential.
    readout_activation, energy_shift, energy_scale, atomic_energies
        See :class:`~xnn.cnn.models.base.VoxelPotential`.

    Attributes
    ----------
    blocks : torch.nn.Sequential
        The ResNet blocks.
    features, bandwidths : list of int
        Channels and bandwidths of every block.
    """

    def __init__(self, species: Sequence[int] = (1, 6, 7, 8, 16), cutoff: float = 10.0,
                 radius: float = 0.48, bandwidth: int = 10, exponent: float = 1.0,
                 features: Optional[Sequence[int]] = None, bandwidths: Optional[Sequence[int]] = None,
                 n_features: int = 160, n_blocks: int = 5, n_alpha: Optional[int] = None,
                 n_beta: int = 2, n_gamma: int = 2, max_beta: float = math.pi / 8,
                 normalization: Optional[str] = "batch", activation: str = "relu",
                 set_readout: Optional[Sequence[int]] = None, cutoff_fn: Optional[str] = None,
                 readout_activation: str = "ssp", energy_shift: float = 0.0,
                 energy_scale: float = 1.0, atomic_energies=None):
        if features is None and bandwidths is None and n_features == 160 and n_blocks == 5 and bandwidth == 10:
            features, bandwidths = [20, 40, 60, 80, 160], [10, 8, 6, 4, 2]       # Table 3
        if features is None or bandwidths is None:
            f_default, b_default = default_schedule(n_features, n_blocks, bandwidth)
            features = f_default if features is None else features
            bandwidths = b_default if bandwidths is None else bandwidths
        features, bandwidths = [int(f) for f in features], [int(b) for b in bandwidths]
        if len(features) != len(bandwidths) or not features:
            raise ValueError("features and bandwidths must have one entry per block")
        if bandwidths[0] > bandwidth or any(b1 > b0 for b0, b1 in zip(bandwidths, bandwidths[1:])):
            raise ValueError("bandwidths must not increase from block to block")
        featurizer = SphericalGrid(species, cutoff, radius, bandwidth, exponent, cutoff_fn)
        super().__init__(species, cutoff, n_features=features[-1], readout_activation=readout_activation,
                         energy_shift=energy_shift, energy_scale=energy_scale,
                         atomic_energies=atomic_energies, featurizer=featurizer)
        self.features, self.bandwidths, self.bandwidth = features, bandwidths, int(bandwidth)
        blocks, n_in, b_in = [], self.n_channels, int(bandwidth)
        for t, (n_out, b_out) in enumerate(zip(features, bandwidths)):
            blocks.append(SphericalResBlock(n_in, n_out, b_in, b_out, s2_input=(t == 0), n_alpha=n_alpha,
                                            n_beta=n_beta, n_gamma=n_gamma, max_beta=max_beta,
                                            normalization=normalization, activation=activation))
            n_in, b_in = n_out, b_out
        self.blocks = nn.Sequential(*blocks)
        self.set_readout = None
        if set_readout is not None:
            phi_hidden, latent, psi_hidden = (int(v) for v in set_readout)
            act = make_activation(activation)
            self.set_readout = nn.ModuleDict({
                "phi": nn.Sequential(nn.Linear(features[-1], phi_hidden), make_activation(activation),
                                     nn.Linear(phi_hidden, latent), act),
                "psi": nn.Sequential(nn.Linear(latent, psi_hidden), make_activation(activation),
                                     nn.Linear(psi_hidden, 1)),
            })

    def trunk(self, grid: Tensor) -> Tensor:
        """ResNet blocks followed by the integral over ``SO(3)``."""
        return so3_integrate(self.blocks(grid))

    def forward(self, data: AtomicGraph) -> Dict[str, Tensor]:
        if self.set_readout is None:
            return super().forward(data)
        features = self.trunk(self.featurizer(data))
        latent = self.set_readout["phi"](features)
        pooled = torch.zeros(data.num_graphs, latent.shape[1], dtype=latent.dtype, device=latent.device)
        pooled = pooled.index_add(0, data.batch, latent)
        residual = self.energy_scale * self.set_readout["psi"](pooled).squeeze(-1) + self.energy_shift * data.n_atoms.to(latent.dtype)
        ref = self.atom_ref(data.atomic_numbers).squeeze(-1)
        node_energy = residual[data.batch] / data.n_atoms.to(latent.dtype)[data.batch] + ref
        return {"node_energy": node_energy, "energy": self.aggregate_energy(node_energy, data),
                "node_features": features}

    @classmethod
    def from_config(cls, cfg) -> "SphericalCNN":
        """Build a :class:`SphericalCNN` from a configuration object.

        ``cfg.cutoff``, ``cfg.n_features`` (channels of the last block) and
        ``cfg.n_interactions`` (number of blocks) are the core fields;
        ``cfg.extra`` may set ``species``, ``radius``, ``bandwidth``,
        ``exponent``, ``features``, ``bandwidths``, ``n_alpha``, ``n_beta``,
        ``n_gamma``, ``max_beta``, ``normalization``, ``activation``,
        ``set_readout``, ``cutoff_fn``, ``readout_activation``,
        ``energy_shift``, ``energy_scale`` and ``atomic_energies``.
        """
        from xnn.common.config.coerce import coerce_per_species, coerce_species

        extra = dict(cfg.extra or {})
        species = coerce_species(extra.get("species"), default=[1, 6, 7, 8, 16])
        features, bandwidths = extra.get("features"), extra.get("bandwidths")
        set_readout = extra.get("set_readout")
        n_alpha = extra.get("n_alpha")
        return cls(
            species=species, cutoff=cfg.cutoff,
            radius=float(extra.get("radius", 0.48)), bandwidth=int(extra.get("bandwidth", 10)),
            exponent=float(extra.get("exponent", 1.0)),
            features=None if features is None else [int(f) for f in features],
            bandwidths=None if bandwidths is None else [int(b) for b in bandwidths],
            n_features=cfg.n_features, n_blocks=cfg.n_interactions,
            n_alpha=None if n_alpha is None else int(n_alpha),
            n_beta=int(extra.get("n_beta", 2)), n_gamma=int(extra.get("n_gamma", 2)),
            max_beta=float(extra.get("max_beta", math.pi / 8)),
            normalization=extra.get("normalization", "batch"), activation=extra.get("activation", "relu"),
            set_readout=None if set_readout is None else [int(v) for v in set_readout],
            cutoff_fn=extra.get("cutoff_fn"),
            readout_activation=extra.get("readout_activation", "ssp"),
            energy_shift=float(extra.get("energy_shift", 0.0)),
            energy_scale=float(extra.get("energy_scale", 1.0)),
            atomic_energies=coerce_per_species(extra.get("atomic_energies"), species, "atomic_energies"),
        )
