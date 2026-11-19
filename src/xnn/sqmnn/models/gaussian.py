"""Overlap and multipole integrals over contracted Gaussian shells (``s``, ``p``, ``d``).

Integrals of Cartesian Gaussians factor into one-dimensional moments

    ``G[m, j] = int (x - A)^m (x - B)^j exp(-p (x - P)^2) dx``

with the Gaussian product centre ``P = (a A + b B) / p``, ``p = a + b``, times
``exp(-a b / p (A - B)^2)``. They follow from ``G[0, 0] = sqrt(pi / p)`` by the
recursions (Obara and Saika, J. Chem. Phys. 84, 3963 (1986))

    ``G[m, 0] = (P - A) G[m - 1, 0] + (m - 1) G[m - 2, 0] / (2 p)``,

    ``G[m, j] = (P - B) G[m, j - 1] + ((j - 1) G[m, j - 2] + m G[m - 1, j - 1]) / (2 p)``.

Dipole and quadrupole integrals with the origin on the centre ``A`` of the
bra raise ``m``. The ``d`` shells are real solid harmonics, built from the
normalized Cartesian components; every function is normalized on its own
centre.

All lengths are in bohr. The functions are differentiable in the positions.
"""
from __future__ import annotations

import functools
import math
from typing import Dict, List, Tuple

import torch
from torch import Tensor

#: Cartesian exponents of the components of each angular momentum.
CARTESIAN: Dict[int, List[Tuple[int, int, int]]] = {
    0: [(0, 0, 0)],
    1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    2: [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)],
}
#: Number of spherical functions per angular momentum.
N_SPHERICAL = {0: 1, 1: 3, 2: 5}
#: Second-moment components ``xx, yy, zz, xy, xz, yz``.
QUADRUPOLE = [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)]

_S3 = math.sqrt(3.0) / 2.0
# rows: d0 (3z^2 - r^2), d(xz), d(yz), d(x^2 - y^2), d(xy) in normalized Cartesian components
_D_TRANSFORM = ((-0.5, -0.5, 1.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                (_S3, -_S3, 0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 1.0, 0.0, 0.0))


def _double_factorial(n: int) -> int:
    return 1 if n <= 0 else n * _double_factorial(n - 2)


def primitive_norm(alpha: Tensor, l: int) -> Tensor:
    """Normalization of ``x^l exp(-alpha r^2)`` (the ``xx``-type component for ``d``)."""
    return (2.0 * alpha / math.pi) ** 0.75 * (4.0 * alpha) ** (0.5 * l) / math.sqrt(_double_factorial(2 * l - 1))


def _cartesian_norm_ratio(l: int) -> List[float]:
    """Norm of each Cartesian component relative to the ``x^l`` component."""
    out = []
    for e in CARTESIAN[l]:
        f = 1
        for k in e:
            f *= _double_factorial(2 * k - 1)
        out.append(math.sqrt(_double_factorial(2 * l - 1) / f))
    return out


def _moments_1d(pa: Tensor, pb: Tensor, p: Tensor, mmax: int, jmax: int) -> List[List[Tensor]]:
    """The moments ``G[m][j]`` of the recursions above (without the exponential prefactor)."""
    half = 0.5 / p
    g: List[List[Tensor]] = [[pa] * (jmax + 1) for _ in range(mmax + 1)]
    g[0][0] = torch.sqrt(math.pi / p).expand(pa.shape)
    for m in range(1, mmax + 1):
        g[m][0] = pa * g[m - 1][0]
        if m > 1:
            g[m][0] = g[m][0] + (m - 1) * half * g[m - 2][0]
    for j in range(1, jmax + 1):
        for m in range(mmax + 1):
            val = pb * g[m][j - 1]
            if j > 1:
                val = val + (j - 1) * half * g[m][j - 2]
            if m > 0:
                val = val + m * half * g[m - 1][j - 1]
            g[m][j] = val
    return g


@functools.lru_cache(maxsize=None)
def _component_selectors(la: int, lb: int, n_ops: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    """One-hot matrices ``(3, n_moments, n_ops * nca * ncb)`` that pick, along each axis, the moment
    ``G[m][j]`` (at row ``m (lb + 1) + j``) of every (operator, bra component, ket component)."""
    ops, ca, cb = _OPERATORS[:n_ops], CARTESIAN[la], CARTESIAN[lb]
    mmax = la + (2 if n_ops > 1 else 0)
    sel = torch.zeros(3, (mmax + 1) * (lb + 1), len(ops) * len(ca) * len(cb), dtype=dtype, device=device)
    for c, (o, ea, eb) in enumerate((o, ea, eb) for o in ops for ea in ca for eb in cb):
        for k in range(3):
            sel[k, (o[k] + ea[k]) * (lb + 1) + eb[k], c] = 1.0
    return sel


def shell_pair_integrals(ra: Tensor, rb: Tensor, alpha_a: Tensor, coef_a: Tensor, la: int,
                         alpha_b: Tensor, coef_b: Tensor, lb: int, multipoles: bool = True,
                         neglect: float = float("inf"), normalized: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
    """Overlap, dipole and second-moment integrals of a batch of shell pairs.

    Parameters
    ----------
    ra, rb : Tensor
        Shell centres, shape ``(P, 3)``, bohr.
    alpha_a, coef_a : Tensor
        Primitive exponents and contraction coefficients of the bra shells, shape
        ``(P, Ka)``; the coefficients multiply normalized ``x^l``-type primitives.
    la : int
        Angular momentum of the bra shells.
    alpha_b, coef_b, lb
        The same for the ket shells.
    multipoles : bool
        Also return the dipole and second-moment integrals.
    neglect : float
        Primitive pairs with ``a b / (a + b) R^2`` above this contribute nothing.
    normalized : bool
        The coefficients already include the norms of the primitives (:func:`primitive_norm`).

    Returns
    -------
    S : Tensor
        ``(P, na, nb)`` overlaps of the spherical functions.
    D : Tensor
        ``(P, 3, na, nb)`` integrals of ``r - A`` (origin on the bra centre), or empty.
    Q : Tensor
        ``(P, 6, na, nb)`` integrals of ``(r - A)_i (r - A)_j`` (``xx, yy, zz, xy, xz, yz``), or empty.
    """
    n_ops = len(_OPERATORS) if multipoles else 1
    if not normalized:
        coef_a = coef_a * primitive_norm(alpha_a, la)
        coef_b = coef_b * primitive_norm(alpha_b, lb)
    args = (ra, rb, alpha_a, coef_a, alpha_b, coef_b)
    if torch.is_grad_enabled() and (ra.requires_grad or rb.requires_grad) and not any(
            t.requires_grad for t in (alpha_a, coef_a, alpha_b, coef_b)):
        vals = _PairIntegrals.apply(*args, la, lb, n_ops, neglect)
    else:
        vals = _to_spherical(_cartesian_values(*args, la, lb, n_ops, neglect), la, lb)
    if not multipoles:
        empty = vals.new_zeros((0,))
        return vals[:, 0], empty, empty
    return vals[:, 0], vals[:, 1:4], vals[:, 4:10]


def _primitive_moments(ra: Tensor, rb: Tensor, alpha_a: Tensor, coef_a: Tensor, alpha_b: Tensor, coef_b: Tensor,
                       mmax: int, jmax: int, neglect: float):
    """The primitive pairs that survive the neglect threshold, ``R`` in all: the shell pair of
    each ``(R,)``, its bra exponent ``(R,)``, its weight ``(R,)`` (normalized coefficients and the
    exponential prefactor) and its moments ``G[m][j]``, each ``(3, R)``."""
    ab = rb - ra                                     # (P, 3)
    r2 = (ab * ab).sum(-1)
    reduced = alpha_a[:, :, None] * alpha_b[:, None, :] / (alpha_a[:, :, None] + alpha_b[:, None, :])
    row, ka, kb = torch.nonzero(reduced.detach() * r2.detach()[:, None, None] <= neglect, as_tuple=True)
    ia, ib = row * alpha_a.shape[1] + ka, row * alpha_b.shape[1] + kb
    a, b = alpha_a.reshape(-1).index_select(0, ia), alpha_b.reshape(-1).index_select(0, ib)
    p = a + b
    pref = (torch.exp(-a * b / p * r2.index_select(0, row)) * coef_a.reshape(-1).index_select(0, ia)
            * coef_b.reshape(-1).index_select(0, ib))
    d = ab.index_select(0, row).t()                  # (3, R)
    return row, a, pref, _moments_1d(b / p * d, -a / p * d, p, mmax, jmax)   # P - A and P - B


def _cartesian_values(ra: Tensor, rb: Tensor, alpha_a: Tensor, coef_a: Tensor, alpha_b: Tensor, coef_b: Tensor,
                      la: int, lb: int, n_ops: int, neglect: float) -> Tensor:
    """Integrals over the Cartesian components, ``(P, ops, nca, ncb)``; the coefficients include
    the norms of the primitives."""
    mmax = la + (2 if n_ops > 1 else 0)
    row, _, pref, g = _primitive_moments(ra, rb, alpha_a, coef_a, alpha_b, coef_b, mmax, lb, neglect)
    sel = _component_selectors(la, lb, n_ops, pref.dtype, pref.device)
    table = torch.stack([g[m][j] for m in range(mmax + 1) for j in range(lb + 1)], -1)   # (3, R, m j)
    # every (operator, bra component, ket component): one selection per axis, one sum over primitives
    prim = ((table[0] * pref[:, None]) @ sel[0]) * (table[1] @ sel[1]) * (table[2] @ sel[2])
    vals = prim.new_zeros(ra.shape[0], prim.shape[1]).index_add(0, row, prim)
    return vals.view(-1, n_ops, len(CARTESIAN[la]), len(CARTESIAN[lb]))


def _centre_gradient(ra: Tensor, rb: Tensor, alpha_a: Tensor, coef_a: Tensor, alpha_b: Tensor, coef_b: Tensor,
                     la: int, lb: int, n_ops: int, neglect: float, weight: Tensor) -> Tensor:
    """Derivative with respect to the bra centres, ``(P, 3)``, of the Cartesian integrals contracted
    with ``weight`` (``(P, ops, nca, ncb)``).

    The derivative of a one-dimensional moment follows from differentiating the bra Gaussian (whose
    centre is also the origin of the operator): ``dG[m, j] / dA = 2 a G[m + 1, j] - m G[m - 1, j]``.
    """
    mmax = la + (2 if n_ops > 1 else 0)
    row, a, pref, g = _primitive_moments(ra, rb, alpha_a, coef_a, alpha_b, coef_b, mmax + 1, lb, neglect)
    sel = _component_selectors(la, lb, n_ops, pref.dtype, pref.device)
    pairs = [(m, j) for m in range(mmax + 1) for j in range(lb + 1)]
    table = torch.stack([g[m][j] for m, j in pairs], -1)
    slope = torch.stack([2.0 * a * g[m + 1][j] - m * g[m - 1][j] if m else 2.0 * a * g[1][j] for m, j in pairs], -1)
    f = [table[k] @ sel[k] for k in range(3)]
    df = [slope[k] @ sel[k] for k in range(3)]
    w = pref[:, None] * weight.reshape(weight.shape[0], -1).index_select(0, row)
    wx = w * f[0]
    prim = torch.stack([(w * df[0] * f[1] * f[2]).sum(-1), (wx * df[1] * f[2]).sum(-1),
                        (wx * f[1] * df[2]).sum(-1)], -1)
    return prim.new_zeros(ra.shape[0], 3).index_add(0, row, prim)


def _to_spherical(vals: Tensor, la: int, lb: int) -> Tensor:
    """Normalized Cartesian components to the spherical functions (last two axes)."""
    ratio = (torch.tensor(_cartesian_norm_ratio(la), dtype=vals.dtype, device=vals.device)[:, None]
             * torch.tensor(_cartesian_norm_ratio(lb), dtype=vals.dtype, device=vals.device)[None, :])
    vals = vals * ratio
    if la == 2:
        vals = torch.einsum("si,...ij->...sj", _transform(vals), vals)
    if lb == 2:
        vals = torch.einsum("...ij,sj->...is", vals, _transform(vals))
    return vals


def _from_spherical(grad: Tensor, la: int, lb: int) -> Tensor:
    """Adjoint of :func:`_to_spherical`: gradients of the spherical values on the Cartesian ones."""
    if la == 2:
        grad = torch.einsum("si,...sj->...ij", _transform(grad), grad)
    if lb == 2:
        grad = torch.einsum("...is,sj->...ij", grad, _transform(grad))
    ratio = (torch.tensor(_cartesian_norm_ratio(la), dtype=grad.dtype, device=grad.device)[:, None]
             * torch.tensor(_cartesian_norm_ratio(lb), dtype=grad.dtype, device=grad.device)[None, :])
    return grad * ratio


class _PairIntegrals(torch.autograd.Function):
    """Shell-pair integrals with their analytic gradients in the centres (the exponents and
    coefficients are constants); the moving origin makes the ket gradient minus the bra one."""

    @staticmethod
    def forward(ctx, ra, rb, alpha_a, coef_a, alpha_b, coef_b, la, lb, n_ops, neglect):
        vals = _cartesian_values(ra, rb, alpha_a, coef_a, alpha_b, coef_b, la, lb, n_ops, neglect)
        ctx.save_for_backward(ra, rb, alpha_a, coef_a, alpha_b, coef_b)
        ctx.args = (la, lb, n_ops, neglect)
        return _to_spherical(vals, la, lb)

    @staticmethod
    def backward(ctx, grad):
        la, lb, n_ops, neglect = ctx.args
        g_ra = _centre_gradient(*ctx.saved_tensors, la, lb, n_ops, neglect, _from_spherical(grad, la, lb))
        return g_ra, -g_ra, None, None, None, None, None, None, None, None


#: Operators of the integrals: overlap, the dipole components and the second moments.
_OPERATORS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)] + QUADRUPOLE


def _transform(like: Tensor) -> Tensor:
    return torch.tensor(_D_TRANSFORM, dtype=like.dtype, device=like.device)


def self_overlap(alpha: Tensor, coef: Tensor, l: int) -> Tensor:
    """Overlap of a contracted shell function with itself (its ``x^l``-type component)."""
    a = alpha[..., :, None]
    b = alpha[..., None, :]
    p = a + b
    n = primitive_norm(alpha, l)
    w = (coef * n)[..., :, None] * (coef * n)[..., None, :]
    val = (math.pi / p) ** 1.5 * _double_factorial(2 * l - 1) / (2.0 * p) ** l
    return (w * val).sum((-1, -2))
