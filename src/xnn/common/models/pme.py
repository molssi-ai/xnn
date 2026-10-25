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
from typing import List, Tuple

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


def mesh_wavevectors(cell: Tensor, mesh: List[int]) -> Tuple[Tensor, Tensor]:
    """The wave vectors of a mesh in FFT order: ``2 pi`` times the Miller
    indices dotted with the reciprocal vectors.

    Parameters
    ----------
    cell : Tensor
        Lattice vectors as rows ``(3, 3)``.
    mesh : list of int
        Mesh dimensions ``(K_1, K_2, K_3)``.

    Returns
    -------
    tuple of Tensor
        ``k`` of shape ``(K_1, K_2, K_3, 3)`` and ``|k|^2`` of shape
        ``(K_1, K_2, K_3)``, in 1/Angstrom.
    """
    dtype, device = cell.dtype, cell.device
    axes: List[Tensor] = []
    for size in mesh:
        axes.append(torch.fft.fftfreq(size, d=1.0 / size, dtype=dtype, device=device))
    grid = torch.meshgrid([axes[0], axes[1], axes[2]], indexing="ij")
    miller = torch.stack([grid[0], grid[1], grid[2]], dim=-1)
    kvec = 2.0 * math.pi * (miller @ torch.linalg.inv(cell).t())
    return kvec, (kvec * kvec).sum(-1)


def spread_charges(charges: Tensor, pos: Tensor, cell: Tensor, mesh: List[int],
                   order: int) -> Tuple[Tensor, Tensor, Tensor]:
    """Charges on the mesh by B-spline assignment (paper eq 4.6).

    Parameters
    ----------
    charges : Tensor
        ``(N,)`` or ``(N, C)`` for several channels.
    pos : Tensor
        Positions ``(N, 3)``.
    cell : Tensor
        Lattice vectors as rows ``(3, 3)``.
    mesh : list of int
        Mesh dimensions ``(K_1, K_2, K_3)``.
    order : int
        B-spline order ``p``.

    Returns
    -------
    tuple of Tensor
        The charge grid ``(C, K_1 K_2 K_3)`` (``C = 1`` for ``(N,)`` charges),
        the flat grid indices ``(N, p^3)`` and the weights ``(N, p^3)`` of
        every atom, for the back-interpolation.
    """
    dtype, device = pos.dtype, pos.device
    n = pos.shape[0]
    q = charges.reshape(n, -1)
    k1, k2, k3 = mesh[0], mesh[1], mesh[2]
    sizes = torch.tensor([k1, k2, k3], dtype=dtype, device=device)
    u = (pos @ torch.linalg.inv(cell)) * sizes                            # (N, 3) scaled fractional
    base = torch.floor(u)
    weights = bspline_weights(u - base, order)                            # (N, 3, p)
    offsets = torch.arange(order, dtype=dtype, device=device)
    index = (base[:, :, None] - offsets).remainder(sizes[:, None]).to(torch.long)   # (N, 3, p)
    w3 = (weights[:, 0, :, None, None] * weights[:, 1, None, :, None]
          * weights[:, 2, None, None, :]).reshape(n, -1)                  # (N, p^3)
    flat = (index[:, 0, :, None, None] * (k2 * k3) + index[:, 1, None, :, None] * k3
            + index[:, 2, None, None, :]).reshape(n, -1)                   # (N, p^3)
    grid = torch.zeros(q.shape[1], k1 * k2 * k3, dtype=dtype, device=device)
    grid = grid.index_add(1, flat.reshape(-1), (q.t()[:, :, None] * w3[None]).reshape(q.shape[1], -1))
    return grid, flat, w3


def least_squares_moduli(size: int, order: int, dtype: torch.dtype, device: torch.device,
                         n_alias: int = 50) -> Tensor:
    """``|b(m)|^2`` of the least-squares spline fit along one mesh axis.

    The particle-mesh sum replaces ``exp(2 pi i m u / K)`` by
    ``b(m) sum_k M_p(u - k) exp(2 pi i m k / K)``. The Euler exponential
    spline (:func:`bspline_moduli`) chooses ``b(m)`` to make the two equal
    at the mesh points; here ``b(m)`` minimises the mean squared difference
    over ``u``. With ``s_j = |sinc(m / K + j)|^p`` the amplitudes of the
    spline sum at the aliased frequencies ``m + j K``, that is
    ``|b(m)|^2 = s_0^2 / (sum_j s_j^2)^2``. Both choices agree at ``m = 0``
    and as the mesh is refined; at a coarse mesh the least-squares one
    leaves a far smaller mean self-interaction error in the energies.

    Parameters
    ----------
    size : int
        Mesh points ``K`` along the axis.
    order : int
        Spline order ``p``.
    dtype, device
        Of the returned tensor.
    n_alias : int, optional
        Aliases ``|j| <= n_alias`` in the sums.

    Returns
    -------
    Tensor
        ``|b(m)|^2``, shape ``(K,)``, in FFT order.
    """
    m = torch.fft.fftfreq(size, d=1.0 / size, dtype=dtype, device=device) / size
    j = torch.arange(-n_alias, n_alias + 1, dtype=dtype, device=device)
    s2 = torch.sinc(m[:, None] + j[None, :]).abs() ** (2 * order)
    return s2[:, n_alias] / s2.sum(1).square()


def spline_moduli(mesh: List[int], order: int, dtype: torch.dtype, device: torch.device,
                  least_squares: bool = False) -> Tensor:
    """``B(m) = prod_i |b_i(m_i)|^2`` over the mesh (paper eq 4.7), ``(K_1, K_2, K_3)``:
    the Euler exponential-spline moduli, or with ``least_squares`` those of
    :func:`least_squares_moduli`."""
    axis = least_squares_moduli if least_squares else bspline_moduli
    return (axis(mesh[0], order, dtype, device)[:, None, None]
            * axis(mesh[1], order, dtype, device)[None, :, None]
            * axis(mesh[2], order, dtype, device)[None, None, :])


def pme_weighted_sum(charges: Tensor, pos: Tensor, cell: Tensor, mesh: List[int], order: int,
                     weight: Tensor) -> Tensor:
    """``sum_c sum_m w(m) |S_c(m)|^2`` with the structure factors from the mesh.

    The structure factors of every charge channel come from the transform of
    the spread charges, ``|S(m)|^2 = B(m) |F(Q)(m)|^2``, so any reciprocal
    sum with per-vector weights ``w(m)`` (``0`` where a vector is excluded)
    runs on the mesh.

    Parameters
    ----------
    charges : Tensor
        ``(N,)`` or ``(N, C)``.
    pos, cell, mesh, order
        As in :func:`spread_charges`.
    weight : Tensor
        ``w(m)`` over the mesh in FFT order, ``(K_1, K_2, K_3)``.

    Returns
    -------
    Tensor
        The scalar sum.
    """
    grid, _, _ = spread_charges(charges, pos, cell, mesh, order)
    transform = torch.fft.fftn(grid.view(grid.shape[0], mesh[0], mesh[1], mesh[2]), dim=(1, 2, 3))
    power = (transform.real.square() + transform.imag.square()).sum(0)
    return (weight * spline_moduli(mesh, order, pos.dtype, pos.device) * power).sum()


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
    k1, k2, k3 = mesh[0], mesh[1], mesh[2]
    volume = cell_volume(cell)
    grid, flat, w3 = spread_charges(charges, pos, cell, mesh, order)

    # the reciprocal potential on the mesh: F^-1[ G(m) F(Q)(m) ] with the
    # influence function G = K^3 / (pi V) exp(-pi^2 m^2 / alpha^2) / m^2 B(m)
    kvec, k2sq = mesh_wavevectors(cell, mesh)
    m2sq = k2sq / (4.0 * math.pi * math.pi)
    safe = torch.where(m2sq > 0, m2sq, torch.ones_like(m2sq))
    influence = (k1 * k2 * k3) / (math.pi * volume) \
        * torch.exp(-(math.pi ** 2) * safe / (alpha * alpha)) / safe \
        * spline_moduli(mesh, order, dtype, device)
    influence = torch.where(m2sq > 0, influence, torch.zeros_like(influence))
    transform = torch.fft.fftn(grid.view(k1, k2, k3))
    potential = torch.real(torch.fft.ifftn(transform * influence)).reshape(-1)

    # back-interpolation onto the atoms with the same weights
    phi = (potential[flat] * w3).sum(-1)
    return 0.5 * charges * phi
