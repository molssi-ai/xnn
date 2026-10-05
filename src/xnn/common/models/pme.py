"""Smooth particle-mesh Ewald reciprocal sums (Essmann and co-workers, 1995).

*J. Chem. Phys.* **103**, 8577 (1995).

The reciprocal-space part of an Ewald sum through a charge mesh: the point
charges are spread onto a regular grid of the unit cell with cardinal
B-splines of order ``p`` (eq 4.1 of the paper, the weights ``M_p(u - k)`` of
the scaled fractional coordinate ``u``), the grid is Fourier transformed, and
the structure factor is recovered from the transform through the Euler
exponential spline coefficients ``b_i(m_i)`` (eq 4.4), so that

    ``E_rec = 1 / (2 pi V) sum_{m != 0} exp(-pi^2 m^2 / alpha^2) / m^2 B(m) |F(Q)(m)|^2``

with ``B(m) = prod_i |b_i(m_i)|^2`` (eq 4.7). The per-atom energies come from
the reciprocal potential interpolated back onto the atoms with the same
splines, ``E_i = q_i phi(r_i) / 2``, which sum to ``E_rec``. Everything is
written in differentiable tensor operations, so forces (through the spline
weights and the charges) and the stress (through the cell) follow from
autograd, and the functions compile under TorchScript.

The mesh size follows the production heuristic ``K_i = 2 alpha L_i / (3
eps^(1/5))`` rounded up to a power of two (the same choice as the reference
AIMNet2 calculator), which covers both the Gaussian truncation and the
spline aliasing error at the target relative accuracy ``eps``. The B-spline
coefficients are the exact ones of the paper (the discrete Euler spline),
not the continuous ``sinc^p`` approximation some codes use; the two agree
to the aliasing error the mesh heuristic controls.
"""
import math
from typing import List

import torch
from torch import Tensor

from .ops import cell_volume


def bspline_weights(t: Tensor, order: int) -> Tensor:
    """Cardinal B-spline weights ``M_p(t + j)`` for ``j`` from ``0`` to ``p - 1``.

    For a scaled coordinate ``u`` with fractional part ``t = u - floor(u)``
    the charge lands on the grid points ``floor(u) - j`` with these weights
    (paper eq 4.1). Built by the recursion ``M_n(x) = x / (n - 1) M_{n-1}(x) +
    (n - x) / (n - 1) M_{n-1}(x - 1)`` from ``M_2``.

    Parameters
    ----------
    t : Tensor
        Fractional offsets in ``[0, 1)``, any shape ``(...)``.
    order : int
        The spline order ``p >= 2``.

    Returns
    -------
    Tensor
        Weights ``(..., order)`` summing to one along the last axis.
    """
    weights: List[Tensor] = [t, 1.0 - t]
    for n in range(3, order + 1):
        new: List[Tensor] = []
        for j in range(n):
            w = torch.zeros_like(t)
            if j < n - 1:
                w = w + (t + j) / (n - 1) * weights[j]
            if j >= 1:
                w = w + (n - t - j) / (n - 1) * weights[j - 1]
            new.append(w)
        weights = new
    return torch.stack(weights, dim=-1)


def bspline_moduli(size: int, order: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    """``|b(m)|^2`` of the Euler exponential spline along one mesh axis (paper eq 4.4).

    ``1 / |b(m)|^2 = | sum_{k=0}^{p-2} M_p(k + 1) exp(2 pi i m k / K) |^2`` for the
    ``K`` frequencies ``m`` in FFT order. A vanishing denominator (odd orders
    at the Nyquist frequency) is replaced by the mean of its neighbors, the
    usual safeguard.

    Parameters
    ----------
    size : int
        Mesh points ``K`` along the axis.
    order : int
        Spline order ``p``.
    dtype, device
        Of the returned tensor.

    Returns
    -------
    Tensor
        ``|b(m)|^2``, shape ``(K,)``.
    """
    m = torch.fft.fftfreq(size, d=1.0 / size, dtype=dtype, device=device)      # integers
    values = bspline_weights(torch.zeros(1, dtype=dtype, device=device), order)[0]   # M_p(0..p-1)
    k = torch.arange(order - 1, dtype=dtype, device=device)
    phase = 2.0 * math.pi * m[:, None] * k[None, :] / size
    coeff = values[1:]                                                    # M_p(k + 1)
    denominator = ((coeff * torch.cos(phase)).sum(-1).square()
                   + (coeff * torch.sin(phase)).sum(-1).square())
    small = denominator < 1e-7 * denominator.max()
    neighbors = 0.5 * (torch.roll(denominator, 1) + torch.roll(denominator, -1))
    denominator = torch.where(small, neighbors, denominator)
    return 1.0 / denominator


def pme_mesh(cell: Tensor, alpha: float, accuracy: float) -> List[int]:
    """Mesh dimensions for a target accuracy: ``2 alpha L_i / (3 eps^(1/5))``, up to a power of two.

    Parameters
    ----------
    cell : Tensor
        Lattice vectors as rows ``(3, 3)``.
    alpha : float
        The Ewald splitting parameter in 1/Angstrom.
    accuracy : float
        Target relative accuracy ``eps``.

    Returns
    -------
    list of int
        ``(K_1, K_2, K_3)``, each at least 2.
    """
    lengths = torch.linalg.norm(cell.detach(), dim=1)
    factor = 2.0 * alpha / (3.0 * accuracy ** 0.2)
    mesh: List[int] = []
    for i in range(3):
        n = factor * float(lengths[i])
        mesh.append(max(2, 1 << int(math.ceil(math.log(max(n, 1.0)) / math.log(2.0)))))
    return mesh


def pme_reciprocal(charges: Tensor, pos: Tensor, cell: Tensor, alpha: float,
                   mesh: List[int], order: int) -> Tensor:
    """Per-atom reciprocal-space energies of one periodic structure, ``q_i phi(r_i) / 2``.

    Parameters
    ----------
    charges : Tensor
        Charges ``(N,)``.
    pos : Tensor
        Positions ``(N, 3)``, same dtype as ``charges``.
    cell : Tensor
        Lattice vectors as rows ``(3, 3)``.
    alpha : float
        Splitting parameter.
    mesh : list of int
        Mesh dimensions ``(K_1, K_2, K_3)``.
    order : int
        B-spline order ``p``.

    Returns
    -------
    Tensor
        Energies ``(N,)`` in units of ``e^2 / Angstrom`` (multiply by the
        Coulomb constant), summing to ``E_rec`` of the module docstring.
    """
    dtype, device = pos.dtype, pos.device
    n = pos.shape[0]
    k1, k2, k3 = mesh[0], mesh[1], mesh[2]
    sizes = torch.tensor([k1, k2, k3], dtype=dtype, device=device)
    recip = torch.linalg.inv(cell)                                        # columns: a_i^*
    volume = cell_volume(cell)

    # charge assignment: p^3 grid points per atom, the outer product of the
    # one-dimensional spline weights (paper eq 4.6)
    u = (pos @ recip) * sizes                                             # (N, 3) scaled fractional
    base = torch.floor(u)
    weights = bspline_weights(u - base, order)                            # (N, 3, p)
    offsets = torch.arange(order, dtype=dtype, device=device)
    index = (base[:, :, None] - offsets).remainder(sizes[:, None]).to(torch.long)   # (N, 3, p)
    w3 = (weights[:, 0, :, None, None] * weights[:, 1, None, :, None]
          * weights[:, 2, None, None, :]).reshape(n, -1)                  # (N, p^3)
    flat = (index[:, 0, :, None, None] * (k2 * k3) + index[:, 1, None, :, None] * k3
            + index[:, 2, None, None, :]).reshape(n, -1)                   # (N, p^3)
    grid = torch.zeros(k1 * k2 * k3, dtype=dtype, device=device)
    grid = grid.index_add(0, flat.reshape(-1), (charges[:, None] * w3).reshape(-1))

    # the reciprocal potential on the mesh: F^-1[ G(m) F(Q)(m) ] with the
    # influence function G = K^3 / (pi V) exp(-pi^2 m^2 / alpha^2) / m^2 B(m)
    f1 = torch.fft.fftfreq(k1, d=1.0 / k1, dtype=dtype, device=device)
    f2 = torch.fft.fftfreq(k2, d=1.0 / k2, dtype=dtype, device=device)
    f3 = torch.fft.fftfreq(k3, d=1.0 / k3, dtype=dtype, device=device)
    mesh_axes = torch.meshgrid([f1, f2, f3], indexing="ij")
    m1, m2, m3 = mesh_axes[0], mesh_axes[1], mesh_axes[2]
    miller = torch.stack([m1, m2, m3], dim=-1)                             # (K1, K2, K3, 3)
    mvec = miller @ recip.t()                                              # rows m . a_i^*
    m2sq = (mvec * mvec).sum(-1)
    safe = torch.where(m2sq > 0, m2sq, torch.ones_like(m2sq))
    moduli = (bspline_moduli(k1, order, dtype, device)[:, None, None]
              * bspline_moduli(k2, order, dtype, device)[None, :, None]
              * bspline_moduli(k3, order, dtype, device)[None, None, :])
    influence = (k1 * k2 * k3) / (math.pi * volume) \
        * torch.exp(-(math.pi ** 2) * safe / (alpha * alpha)) / safe * moduli
    influence = torch.where(m2sq > 0, influence, torch.zeros_like(influence))
    transform = torch.fft.fftn(grid.view(k1, k2, k3))
    potential = torch.real(torch.fft.ifftn(transform * influence)).reshape(-1)

    # back-interpolation onto the atoms with the same weights
    phi = (potential[flat] * w3).sum(-1)
    return 0.5 * charges * phi
