"""Iterative solve of the (s)GDML fit for training sets whose kernel matrix does not fit in memory.

The system ``(lam I - K) z = y`` (``alpha = -z``) is solved by preconditioned
conjugate gradients without forming ``K``: a product ``K v`` is the gradient of
the kernel expansion with the coefficients ``v``, evaluated at the training
geometries and taken through their descriptor Jacobians (the energies for the
energy rows), the same sums the model evaluates when it predicts. When memory
allows, the kernel factors of every training pair are formed once and a product
is three matrix products (:class:`PairFactors`). The
preconditioner is the inverse of a Nystrom approximation of ``lam I - K`` built
from the rows of ``k`` inducing geometries, drawn with probabilities that follow
their ridge leverage scores (estimated from a few random geometries).
"""
from __future__ import annotations

import math
import os
import warnings
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor

from ..models.sgdml import SGDML, kernel_energy
from .sgdml_fit import kernel_rows

#: geometries whose rows estimate the leverage scores that draw the inducing geometries
LEVERAGE_SAMPLE = 10


def device_memory(device: torch.device) -> float:
    """Total memory of ``device`` in GB (physical memory on a CPU)."""
    if device.type == "cuda":
        return torch.cuda.get_device_properties(device).total_memory / 2 ** 30
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30


def analytic_bytes(n: int) -> float:
    """Bytes of the analytic solve of an ``n``-dimensional system (the matrix and its factor)."""
    return 16.0 * n * n


def inducing_count(n_train: int, n_atoms: int, max_memory: float) -> int:
    """The number of inducing geometries for a preconditioner within ``max_memory`` GB.

    The largest ``k`` with ``8 (4 M k + 5 k^2) (3 N)^2`` bytes within the budget, for ``M``
    training geometries of ``N`` atoms.
    """
    dof = 8.0 * (3 * n_atoms) ** 2
    lin, quad = 4.0 * n_train * dof, 5.0 * dof
    k = (math.sqrt(lin * lin + 4.0 * quad * max_memory * 2 ** 30) - lin) / (2.0 * quad)
    return max(1, min(n_train, int(k)))


def kernel_product(model: SGDML, x: Tensor, u: Tensor, v: Tensor, energy_constraint: bool = False) -> Tensor:
    """``K v`` for the kernel matrix of :func:`~xnn.krnn.common.sgdml_fit.kernel_matrix`, without forming it.

    Parameters
    ----------
    model : SGDML
        Supplies the kernel and the permutations.
    x, u : Tensor
        ``(M, D)`` training descriptors and ``(M, D, 3)`` their pair gradients.
    v : Tensor
        ``(3 N M,)`` coefficients (``(3 N M + M,)`` with ``energy_constraint``).
    energy_constraint : bool, optional
        Whether the system has energy rows.

    Returns
    -------
    Tensor
        The product, shaped like ``v``.
    """
    m, n = x.shape[0], model.n_atoms
    a, b = model.descriptor.pairs
    vf = v[:m * 3 * n].reshape(m, n, 3)
    ec = v[m * 3 * n:] if energy_constraint else v[:0]
    w = (u * (vf.index_select(1, a) - vf.index_select(1, b))).sum(-1)       # J_j v_j, (M, D)
    pidx = model.perm_index
    xp, wp = x[:, pidx].contiguous(), w[:, pidx].contiguous()
    if model.uses_fast_path(x):
        from ..fast.sgdml_kernels import energy_gradient
        e, grad = energy_gradient(x, xp, wp, ec, model.kernel)
    else:
        e, grad = kernel_energy(x, xp, wp, ec, model.kernel, gradient=True)
    pair = grad.unsqueeze(-1) * u                                           # (M, D, 3)
    jt = torch.zeros_like(vf).index_add_(1, a, pair).index_add_(1, b, pair, alpha=-1)
    out = -jt.reshape(-1)
    return torch.cat([out, -e]) if energy_constraint else out


class PairFactors:
    """``K v`` from the kernel factors of every training pair, formed once.

    The geometry does not change during the solve, so ``g``, ``t`` (and ``k`` with
    energy rows) of every pair of a training geometry and a permuted training
    term are kept (from exact distances); a product then needs three matrix
    products over the ``(M, M Q)`` pairs: the dot products ``D . w`` as
    ``x_i . w - x_j . w`` and the gradient terms likewise. Those differences
    cancel to about 1e-12 relative, far below the solver's tolerance.

    Parameters
    ----------
    model : SGDML
        Supplies the kernel and the permutations.
    x, u : Tensor
        ``(M, D)`` training descriptors and ``(M, D, 3)`` their pair gradients.
    energy_constraint : bool, optional
        Whether the system has energy rows.
    """

    def __init__(self, model: SGDML, x: Tensor, u: Tensor, energy_constraint: bool = False):
        m, d = x.shape
        pidx = model.perm_index
        self.model, self.x, self.u, self.energy = model, x, u, energy_constraint
        self.xp = x[:, pidx].reshape(-1, d)                                   # (M Q, D)
        step = max(1, (1 << 24) // max(1, self.xp.numel()))
        r = torch.cat([torch.linalg.vector_norm(x[s:s + step, None, :] - self.xp[None], dim=-1)
                       for s in range(0, m, step)])                              # (M, M Q)
        self.g, self.t = model.kernel.g_and_t(r)
        self.k = model.kernel.k(r) if energy_constraint else None

    @staticmethod
    def nbytes(m: int, q: int, energy_constraint: bool) -> float:
        """Bytes of the factors and of the temporaries of a product."""
        return 8.0 * m * m * q * (5 if energy_constraint else 4)

    def __call__(self, v: Tensor) -> Tensor:
        model, x, u = self.model, self.x, self.u
        m, n = x.shape[0], model.n_atoms
        q = model.perm_index.shape[0]
        a, b = model.descriptor.pairs
        vf = v[:m * 3 * n].reshape(m, n, 3)
        w = (u * (vf.index_select(1, a) - vf.index_select(1, b))).sum(-1)    # J_j v_j, (M, D)
        wp = w[:, model.perm_index].reshape(-1, x.shape[1])                    # (M Q, D)
        s = x @ wp.T - (self.xp * wp).sum(-1)                                # D . P_q w_j, (M, M Q)
        e = (self.g * s).sum(1)
        c = self.t * s
        if self.energy:
            ec = v[m * 3 * n:].repeat_interleave(q)
            e = e + self.k @ ec
            c = c - self.g * ec
        grad = self.g @ wp + c.sum(1, keepdim=True) * x - c @ self.xp
        pair = grad.unsqueeze(-1) * u
        jt = torch.zeros_like(vf).index_add_(1, a, pair).index_add_(1, b, pair, alpha=-1)
        out = -jt.reshape(-1)
        return torch.cat([out, -e]) if self.energy else out


def _stable_cholesky(a: Tensor) -> Tensor:
    """Lower Cholesky factor of a positive semi-definite ``a``, with the smallest shift of the diagonal
    (relative to its largest entry, from 1e-14 up) that makes it definite."""
    scale = float(a.diagonal().abs().max())
    for eps in [0.0] + [10.0 ** e for e in range(-14, -3)]:
        shifted = a if eps == 0.0 else a + eps * scale * torch.eye(a.shape[0], dtype=a.dtype, device=a.device)
        chol, info = torch.linalg.cholesky_ex(shifted)
        if int(info) == 0:
            return chol
    raise RuntimeError("the inducing block of the kernel matrix is not positive semi-definite")


def _dofs(idx: Sequence[int], m: int, n3: int, energy_constraint: bool) -> Tensor:
    """Indices of the unknowns of the geometries ``idx`` (force rows first, then energy rows)."""
    rows = [i * n3 + c for i in idx for c in range(n3)]
    if energy_constraint:
        rows += [m * n3 + i for i in idx]
    return torch.tensor(rows, dtype=torch.long)


def _nystroem_factor(model: SGDML, x: Tensor, u: Tensor, idx: Sequence[int], lam: float,
                     energy_constraint: bool) -> Tuple[Tensor, Tensor]:
    """``B`` ``(r, n)`` and the lower Cholesky factor ``R`` of ``lam I + B B^T``, with ``B^T B`` the
    Nystrom approximation of ``G = -K`` on the rows of the geometries ``idx``, so that
    ``(lam I + B^T B)^-1 = (I - B^T (R R^T)^-1 B) / lam``."""
    m, n3 = x.shape[0], 3 * model.n_atoms
    rows = kernel_rows(model, x, u, idx, energy_constraint).neg_()          # rows of G, (r, n)
    dofs = _dofs(idx, m, n3, energy_constraint).to(x.device)
    chol = _stable_cholesky(rows[:, dofs])                                   # W = L L^T
    # B = L^-1 G_I (in place, in column blocks)
    step = max(1, (1 << 24) // max(1, rows.shape[0]))
    for s in range(0, rows.shape[1], step):
        rows[:, s:s + step] = torch.linalg.solve_triangular(chol, rows[:, s:s + step], upper=False)
    inner = rows @ rows.T
    inner.diagonal().add_(lam)
    return rows, torch.linalg.cholesky(inner)


def _leverage_draw(model: SGDML, x: Tensor, u: Tensor, k: int, lam: float, energy_constraint: bool,
                   generator: Optional[torch.Generator]) -> list:
    """``k`` inducing geometries drawn with probabilities that follow their ridge leverage scores."""
    m, n3 = x.shape[0], 3 * model.n_atoms
    if k >= m:
        return list(range(m))
    sample = torch.randperm(m, generator=generator)[:min(k, LEVERAGE_SAMPLE)].sort().values.tolist()
    b, r = _nystroem_factor(model, x, u, sample, lam, energy_constraint)
    e = torch.linalg.solve_triangular(r, b, upper=False)
    lev = (e * e).sum(0)                                                     # per unknown
    per = lev[:m * n3].reshape(m, n3).sum(1)
    if energy_constraint:
        per = per + lev[m * n3:]
    weights = per.clamp_min(0).cpu() + 1e-300
    return torch.multinomial(weights, k, replacement=False, generator=generator).sort().values.tolist()


def _pcg(apply: Callable[[Tensor], Tensor], b: Tensor, precondition: Callable[[Tensor], Tensor], tol: float,
         max_iter: int) -> Tuple[Tensor, int, float, bool]:
    """Preconditioned conjugate gradients for ``A z = b``: solution, iterations, relative residual, converged."""
    z = torch.zeros_like(b)
    r = b.clone()
    bnorm = float(torch.linalg.vector_norm(b))
    p = precondition(r)
    rho = torch.dot(r, p)
    for it in range(max_iter + 1):
        res = float(torch.linalg.vector_norm(r)) / bnorm
        if res <= tol:
            return z, it, res, True
        if it == max_iter:
            break
        ap = apply(p)
        step = rho / torch.dot(p, ap)
        z.add_(step * p)
        r.sub_(step * ap)
        s = precondition(r)
        rho_new = torch.dot(r, s)
        p = s + (rho_new / rho) * p
        rho = rho_new
    return z, max_iter, res, False


def solve_iterative(model: SGDML, x: Tensor, u: Tensor, y: Tensor, lam: float, energy_constraint: bool = False,
                    tol: float = 1e-4, n_inducing: Optional[int] = None, max_memory: Optional[float] = None,
                    max_iter: Optional[int] = None,
                    generator: Optional[torch.Generator] = None) -> Tuple[Tensor, Dict[str, object]]:
    """``alpha`` with ``(K - lam I) alpha = y`` by preconditioned conjugate gradients.

    Parameters
    ----------
    model : SGDML
        Supplies the kernel and the permutations (closed under inversion, so
        that the system is symmetric).
    x, u : Tensor
        ``(M, D)`` training descriptors and ``(M, D, 3)`` their pair gradients.
    y : Tensor
        Right-hand side (scaled labels).
    lam : float
        Regularization.
    energy_constraint : bool, optional
        Whether the system has energy rows.
    tol : float, optional
        Relative residual ``|r| / |y|`` at which the iterations stop, by default 1e-4.
    n_inducing : int, optional
        Inducing geometries of the preconditioner; by default as many as
        ``max_memory`` holds.
    max_memory : float, optional
        Memory in GB for the preconditioner and, when half of it holds them, the
        kernel factors of every training pair (:class:`PairFactors`); by default
        the device's.
    max_iter : int, optional
        Iteration limit, by default ten times the size of the system.
    generator : torch.Generator, optional
        Random numbers of the inducing geometries.

    Returns
    -------
    tuple of Tensor and dict
        ``alpha`` and the solver's record (``iterations``, ``residual``,
        ``converged``, ``n_inducing``).
    """
    m, n = x.shape[0], model.n_atoms
    memory = device_memory(x.device) if max_memory is None else float(max_memory)
    if n_inducing is None:
        n_inducing = inducing_count(m, n, memory)
    k = max(1, min(int(n_inducing), m))
    idx = _leverage_draw(model, x, u, k, lam, energy_constraint, generator)
    b, r = _nystroem_factor(model, x, u, idx, lam, energy_constraint)

    def precondition(v: Tensor) -> Tensor:
        return (v - b.T @ torch.cholesky_solve((b @ v).unsqueeze(1), r).squeeze(1)) / lam

    product = (PairFactors(model, x, u, energy_constraint)
               if PairFactors.nbytes(m, model.perm_index.shape[0], energy_constraint) <= memory * 2 ** 30 / 2
               else (lambda v: kernel_product(model, x, u, v, energy_constraint)))

    def apply(v: Tensor) -> Tensor:
        return (-product(v)).add_(v, alpha=lam)

    limit = 10 * y.numel() if max_iter is None else int(max_iter)
    z, iters, res, ok = _pcg(apply, y, precondition, tol, limit)
    if not ok:
        warnings.warn(f"the iterative solve stopped after {iters} iterations at a relative residual of {res:.2e} "
                      f"(tolerance {tol:.0e}); more inducing geometries or a larger sigma may help",
                      RuntimeWarning, stacklevel=3)
    return -z, {"iterations": iters, "residual": res, "converged": ok, "n_inducing": k}
