"""Classical energy terms evaluated with per-term parameters.

The library force fields of this family (OPLS, AMBER, ...) look parameters up by
atom type. A force field whose parameters are predicted per bond, angle, torsion
and atom (ByteFF and ByteFF-Pol in :mod:`xnn.hybnn`) evaluates the same functional
forms with one parameter row per term; these functions take those rows. They are
unit-agnostic: the energies come out in the units of the parameters.

Polarization follows the induced-point-dipole model with Thole damping of the
AMOEBA force field (Ren & Ponder, J. Comput. Chem. 23, 1497 (2002),
https://doi.org/10.1002/jcc.10127): permanent point charges polarize isotropic
atomic polarizabilities, the dipoles interact through the damped dipole field
tensor and are solved for directly (Cholesky factorization).
"""
from __future__ import annotations

from typing import List

import torch
from torch import Tensor


def bond_angle(u: Tensor, v: Tensor) -> Tensor:
    """The angle ``(M,)`` between the vectors ``u`` and ``v`` ``(M, 3)`` from both of its
    projections (well conditioned at 0 and 180 degrees)."""
    w = torch.linalg.cross(u, v, dim=-1)
    return torch.atan2(torch.sqrt((w * w).sum(-1)), (u * v).sum(-1))


def torsion_cos(b1: Tensor, b2: Tensor, b3: Tensor) -> Tensor:
    """``cos phi`` ``(M,)`` of the torsions along the bond vectors ``b1``, ``b2``, ``b3`` ``(M, 3)``
    (``phi = 0`` at cis; undefined for collinear atoms)."""
    n1 = torch.linalg.cross(b1, b2, dim=-1)
    n2 = torch.linalg.cross(b2, b3, dim=-1)
    return (n1 * n2).sum(-1) / (torch.sqrt((n1 * n1).sum(-1)) * torch.sqrt((n2 * n2).sum(-1)))


def harmonic(x: Tensor, k: Tensor, x0: Tensor) -> Tensor:
    """``k / 2 (x - x0)^2``."""
    return 0.5 * k * (x - x0) ** 2


def fourier_series(cos1: Tensor, k: Tensor, n: List[int], sign: Tensor) -> Tensor:
    """``sum_m k_m [1 + s_m cos(n_m phi)]`` per torsion, the phases zero (``s = 1``) or
    pi (``s = -1``).

    ``cos(n phi)`` is the Chebyshev polynomial ``T_n(cos phi)``
    (``T_{n+1} = 2 c T_n - T_{n-1}``), so the sine of the torsion is never needed.

    Parameters
    ----------
    cos1 : Tensor
        ``cos phi`` of the torsions ``(M,)``.
    k : Tensor
        Barrier parameters ``(M, K)``.
    n : list of int
        The multiplicity of each column of ``k``.
    sign : Tensor
        The sign ``s`` of each column, ``(K,)`` or ``(M, K)``.

    Returns
    -------
    Tensor
        Energies ``(M,)``.
    """
    cols = [torch.ones_like(cos1), cos1]
    for _ in range(1, max(n)):
        cols.append(2.0 * cos1 * cols[-1] - cols[-2])
    t = torch.stack([cols[m] for m in n], -1)
    s = sign.to(dtype=cos1.dtype, device=cos1.device)
    return (k * (1.0 + s * t)).sum(-1)


def lennard_jones(r2: Tensor, sigma: Tensor, epsilon: Tensor) -> Tensor:
    """``4 eps [(sigma/r)^12 - (sigma/r)^6]`` from the squared distances."""
    x = (sigma * sigma / r2) ** 3
    return 4.0 * epsilon * (x * x - x)


def exp_repulsion(r: Tensor, epsilon: Tensor, lam: Tensor, r0: Tensor, prefactor: float = 6.0,
                  s12: float = 0.0) -> Tensor:
    """``a eps exp[lam (1 - r/r0)] + (s12/r)^12``: Born-Mayer repulsion with a
    short-range wall (``s12 = 0`` drops the wall)."""
    e = prefactor * epsilon * torch.exp(lam * (1.0 - r / r0))
    if s12 > 0.0:
        e = e + (s12 / r) ** 12
    return e


def damped_dispersion(r: Tensor, c6: Tensor, r0: Tensor, damping: float) -> Tensor:
    """``-C6 / (s r0^6 + r^6)``: the rational damping of the C6 term."""
    return -c6 / (damping * r0 ** 6 + r ** 6)


def charge_transfer(r: Tensor, epsilon: Tensor, lam: Tensor, r0: Tensor) -> Tensor:
    """``-eps r^-4 exp[-(lam r / r0)^3]``."""
    return -epsilon / r ** 4 * torch.exp(-((lam * r / r0) ** 3))


def polarization_system(pos: Tensor, q: Tensor, alpha: Tensor, damp: Tensor, scale: Tensor, thole: float,
                        floor: float = 1.0e-6) -> tuple[Tensor, Tensor]:
    """The linear system of induced point dipoles ``A mu = E`` of isolated structures.

    ``A`` holds the inverse polarizabilities on its diagonal and the damped dipole field
    tensor between distinct atoms; ``E`` is the field of the permanent charges (see
    :func:`induced_dipoles` for the damping and the arguments).

    Parameters
    ----------
    pos, q, alpha, damp, scale, thole, floor
        As for :func:`induced_dipoles`.

    Returns
    -------
    tuple of Tensor
        ``A`` ``(B, 3 n, 3 n)`` and the field ``E`` ``(B, n, 3)``.
    """
    B, n, _ = pos.shape
    d = pos.unsqueeze(1) - pos.unsqueeze(2)                   # d[b, i, j] = r_j - r_i
    eye = torch.eye(n, dtype=torch.bool, device=pos.device)
    r2 = torch.where(eye, torch.ones((), dtype=pos.dtype, device=pos.device), (d * d).sum(-1))
    r = torch.sqrt(r2)
    off = (~eye).to(pos.dtype)
    u = thole * r * r2 / torch.sqrt(damp.unsqueeze(-1) * damp.unsqueeze(-2))
    ex = torch.exp(-u)
    d1 = (1.0 - ex) * off
    d2 = torch.clamp(1.0 - (1.0 + u) * ex, min=floor) * off
    inv3 = 1.0 / (r * r2)
    field = -((scale * d1 * inv3) * q.unsqueeze(-2)).unsqueeze(-1).mul(d).sum(-2)
    inv5 = inv3 / r2
    tensor = (r2 * d1 * inv5).unsqueeze(-1).unsqueeze(-1) * torch.eye(3, dtype=pos.dtype, device=pos.device) \
        - (3.0 * d2 * inv5).unsqueeze(-1).unsqueeze(-1) * d.unsqueeze(-1) * d.unsqueeze(-2)
    A = tensor.transpose(2, 3).reshape(B, 3 * n, 3 * n)
    A = A + torch.diag_embed((1.0 / alpha).expand(B, n).repeat_interleave(3, dim=-1))
    return A, field


def solve_dipoles(A: Tensor, field: Tensor) -> Tensor:
    """The induced dipoles ``(B, n, 3)`` of the system ``A mu = E`` of :func:`polarization_system`.

    The matrix is symmetric positive definite unless atoms overlap (a polarization catastrophe): a
    Cholesky factorization solves it, an LU factorization where that fails (not tested inside a CUDA-graph
    capture, which reads nothing from the device).
    """
    B, n = field.shape[0], field.shape[1]
    rhs = field.reshape(B, 3 * n, 1)
    L, info = torch.linalg.cholesky_ex(A)
    # a CUDA-graph capture reads nothing from the device: the factorization of the recorded step serves
    capturing = False if torch.jit.is_scripting() else _capturing(A)
    if not capturing and bool((info != 0).any()):
        return torch.linalg.solve(A, rhs).reshape(B, n, 3)
    # two triangular solves (cuBLAS) rather than cholesky_solve, whose batched GPU path (MAGMA) allocates
    # memory and so cannot be captured
    y = torch.linalg.solve_triangular(L, rhs, upper=False)
    return torch.linalg.solve_triangular(L.transpose(-2, -1), y, upper=True).reshape(B, n, 3)


@torch.jit.unused
def _capturing(A: Tensor) -> bool:
    """Whether ``A`` lives on a CUDA stream that records a graph (not in TorchScript, which never captures)."""
    return A.is_cuda and torch.cuda.is_current_stream_capturing()


def induced_dipoles(pos: Tensor, q: Tensor, alpha: Tensor, damp: Tensor, scale: Tensor, thole: float,
                    floor: float = 1.0e-6) -> tuple[Tensor, Tensor, Tensor]:
    """Induced point dipoles of isolated structures by a direct linear solve.

    The field of the permanent charges, damped as ``1 - exp(-u)`` with
    ``u = a r^3 / (d_i d_j)^(1/2)`` and scaled pairwise by ``scale``, polarizes
    ``alpha``; the dipoles interact through the damped dipole field tensor
    (``r^2 I (1 - e^-u) - 3 r r^T [1 - (1 + u) e^-u]``, the second factor floored at
    ``floor``) between every pair of distinct atoms, unscaled. The symmetric positive-definite system is
    solved by a Cholesky factorization (by LU where it is not positive definite).

    Parameters
    ----------
    pos : Tensor
        Positions ``(B, n, 3)``.
    q : Tensor
        Charges ``(B, n)`` or ``(n,)``.
    alpha : Tensor
        Polarizabilities ``(B, n)`` or ``(n,)``, in the cube of the length unit.
    damp : Tensor
        Damping volumes ``d`` ``(B, n)`` or ``(n,)``.
    scale : Tensor
        Field scale factors ``(n, n)`` of the charges' field (zero on the diagonal).
    thole : float
        The damping strength ``a``.
    floor : float, optional
        Lower bound of the tensor's second damping factor.

    Returns
    -------
    tuple of Tensor
        The per-atom polarization energies ``-mu_i . E_i / 2`` ``(B, n)`` (in charge
        squared over length; multiply by the Coulomb constant), the dipoles
        ``(B, n, 3)`` and the fields ``(B, n, 3)``.
    """
    A, field = polarization_system(pos, q, alpha, damp, scale, thole, floor)
    mu = solve_dipoles(A, field)
    return -0.5 * (mu * field).sum(-1), mu, field


__all__ = ["bond_angle", "torsion_cos", "harmonic", "fourier_series", "lennard_jones", "exp_repulsion",
           "damped_dispersion", "charge_transfer", "polarization_system", "solve_dipoles", "induced_dipoles"]
