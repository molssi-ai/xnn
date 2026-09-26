"""Large-system EEQ charges: matrix-free operator, solvers, implicit differentiation.

The dense EEQ path of :class:`~xnn.common.models.d4.DFTD4` builds the
``(N, N)`` interaction matrix with autograd recording every pairwise
intermediate (about eight ``(N, N)`` tensors for a molecule, and for a periodic
cell the Wigner-Seitz image search plus 125 translations per pair, about
7 kB per pair). It reproduces ``dftd4`` bit for bit and is the right tool up to
a few thousand atoms. This module is the *large* regime: the same linear
system, evaluated so that memory stays ``O(E + N N_G + N c)`` (edges,
structure factors, one row block) and gradients cost one matrix-vector
product per backward.

Algorithm:

* **Operator.** Molecules: ``A_ij = erf(gamma_ij r_ij) / r_ij`` applied in
  row blocks that are never stored. Periodic cells: an Ewald split at a
  splitting parameter ``alpha`` chosen from the real-space cutoff, the
  screened kernel ``[erf(gamma r) - erf(alpha r)] / r`` summed over the
  neighbor list (all images within the cutoff, self-images included), and the
  reciprocal sum written through structure factors: with ``C = cos(r . G)``,
  ``S = sin(r . G)`` of shape ``(N, N_G)`` the reciprocal block is the low-rank
  product ``C diag(g) C^T + S diag(g) S^T``, so ``A v`` costs ``O(N N_G)``
  without forming ``(N, N)``. The diagonal is ``J + sqrt(2/pi) / a`` (minus
  ``2 alpha / sqrt(pi)`` in the periodic case), as in the reference code.
* **Solve.** The charge-constrained system ``[[A, 1], [1^T, 0]] [q; mu] =
  [x; Q]`` is solved with an LU factorization of the assembled matrix while
  ``(N, N)`` fits, and otherwise by conjugate gradients on the symmetric
  positive definite ``A`` (Jacobi preconditioner) with the constraint
  eliminated: ``q = y_1 - mu y_2``, ``A y_1 = x``, ``A y_2 = 1``.
* **Gradients.** Implicit differentiation with a constant Jacobian: the
  solution ``q_0`` is computed without autograd, then ``q = q_0 + J^{-1}
  [x - A(r) q_0 - mu_0; Q - 1^T q_0]`` with the *constant* operator ``J^{-1}``
  (a custom autograd function whose backward is another solve) and one
  differentiable application of ``A(r)`` in recomputed blocks. The correction is
  zero at convergence, its derivative is the exact first-order sensitivity
  ``dq/dr = -J^{-1} (dA/dr) q``, and force training (a second derivative with
  respect to the model parameters, on which ``A`` does not depend) is exact.

The large regime agrees with the dense one to the truncation and solver
tolerances (about 1e-10 hartree), not to the last bit, and it is eager-only:
none of this scripts, so TorchScript exports keep the dense path.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor
from .ops import cell_volume, scatter_sum
from .recompute import recompute

#: relative truncation of the real- and reciprocal-space Ewald tails
EWALD_EPS = 1e-10
#: conjugate-gradient stopping criterion, ``|r| <= tol |b|`` (raised to 100
#: machine epsilons in single precision)
CG_TOL = 1e-12
CG_MAXITER = 1000
#: atoms up to which the assembled matrix is LU-factorized (two ``(N, N)``
#: float64 tensors: 2.3 GB at 12000 atoms)
LU_MAX_ATOMS = 12000
#: rows per block of the molecular dense kernel (block: ``ROW_CHUNK x N``)
ROW_CHUNK = 2048
#: largest ``N x N_G`` structure-factor table kept in memory (float64: 3.2 GB);
#: beyond it the reciprocal sum is evaluated in chunks of G vectors
SF_BUDGET = 2 * 10 ** 8

_SQRT_PI = math.sqrt(math.pi)
_SQRT_2_PI = math.sqrt(2.0 / math.pi)


def _erfc_inverse(eps: float) -> float:
    """``x`` with ``erfc(x) = eps`` (Newton iterations from a log estimate)."""
    x = math.sqrt(-math.log(eps))
    for _ in range(20):
        f = math.erfc(x) - eps
        x -= f / (-2.0 / _SQRT_PI * math.exp(-x * x))
    return x


def ewald_alpha(cutoff: float, eps: float = EWALD_EPS) -> float:
    """Splitting parameter with ``erfc(alpha r_c) = eps`` at the real-space cutoff (bohr)."""
    return _erfc_inverse(eps) / cutoff


def reciprocal_vectors(cell: Tensor, alpha: float, eps: float = EWALD_EPS) -> tuple[Tensor, Tensor]:
    """Reciprocal vectors ``G != 0`` with ``exp(-G^2 / 4 alpha^2) >= eps`` and their weights.

    Returns the integer triples ``grid (N_G, 3)``, ``gvec (N_G, 3)`` and
    ``g (N_G,) = 4 pi / V exp(-G^2 / 4 alpha^2) / G^2`` (bohr units; ``cell``
    rows are lattice vectors in bohr).
    """
    vol = cell_volume(cell)
    recip = 2.0 * math.pi * torch.linalg.inv(cell).t()                  # rows: b_i
    g_max = 2.0 * alpha * math.sqrt(-math.log(eps))
    # |m_i| = |G . a_i| / 2 pi <= G_max |a_i| / 2 pi
    a_len = torch.linalg.norm(cell, dim=1)
    m_max = torch.ceil(g_max * a_len / (2.0 * math.pi)).to(torch.long)
    ranges = [torch.arange(-int(m), int(m) + 1, device=cell.device) for m in m_max]
    grid = torch.cartesian_prod(*ranges).to(cell.dtype).reshape(-1, 3)
    gvec = grid @ recip
    g2 = (gvec * gvec).sum(-1)
    keep = (g2 > 0) & (g2 <= g_max * g_max)
    return grid[keep], *reciprocal_weights(grid[keep], cell, alpha)


def reciprocal_weights(grid: Tensor, cell: Tensor, alpha: float) -> tuple[Tensor, Tensor]:
    """``G = m B`` and ``g(G)`` for integer triples ``grid`` and a (live) ``cell``."""
    vol = cell_volume(cell)
    recip = 2.0 * math.pi * torch.linalg.inv(cell).t()
    gvec = grid @ recip
    g2 = (gvec * gvec).sum(-1)
    g = 4.0 * math.pi / vol * torch.exp(-0.25 * g2 / (alpha * alpha)) / g2
    return gvec, g


class EEQSystem:
    """The EEQ linear system of one structure as a constant, matrix-free operator.

    Built from *detached* geometry: it is the fixed Jacobian of the implicit
    differentiation. :meth:`apply_differentiable` re-evaluates the same
    operator on live tensors for the residual.

    Parameters
    ----------
    diag : Tensor
        Diagonal ``J_A + sqrt(2/pi) / a_A`` (periodic: ``- 2 alpha / sqrt(pi)``), ``(N,)``.
    rad : Tensor
        EEQ charge widths ``a_A`` in bohr, ``(N,)``.
    pos : Tensor
        Positions in bohr, ``(N, 3)``; molecular kernel and structure factors.
    edge_index, edge_vec : Tensor or None
        Directed neighbor list within the real-space cutoff (periodic case),
        ``(2, E)`` and ``(E, 3)`` in bohr; ``None`` for a molecule.
    alpha : float
        Ewald splitting parameter (periodic case), 0 for a molecule.
    gvec, gfac, grid : Tensor or None
        Reciprocal vectors ``(N_G, 3)``, weights ``(N_G,)`` and their integer
        triples (periodic), from :func:`reciprocal_vectors`.
    solver : str
        ``"auto"`` (LU up to :data:`LU_MAX_ATOMS`, else CG), ``"lu"`` or ``"cg"``.
    reuse : EEQReuse or None
        Carry the solve over from the previous structure of an MD run or
        optimization (:class:`EEQReuse`); ``solver`` is then only the fallback.
    signature : tuple or None
        Hashable identity of the system (atom count and elements) that
        :class:`EEQReuse` compares before reusing its state.
    """

    def __init__(self, diag: Tensor, rad: Tensor, pos: Tensor,
                 edge_index: Optional[Tensor] = None, edge_vec: Optional[Tensor] = None,
                 alpha: float = 0.0, gvec: Optional[Tensor] = None,
                 gfac: Optional[Tensor] = None, grid: Optional[Tensor] = None,
                 solver: str = "auto", reuse: Optional["EEQReuse"] = None,
                 signature: Optional[tuple] = None):
        self.n = pos.shape[0]
        self.diag = diag.detach()
        self.rad = rad.detach()
        self.pos = pos.detach()
        self.periodic = edge_index is not None
        self.alpha = float(alpha)
        self.edge_index = edge_index
        self.edge_vec = edge_vec.detach() if edge_vec is not None else None
        self.gvec = gvec
        self.gfac = gfac
        self.grid = grid
        if solver not in ("auto", "lu", "cg"):
            raise ValueError(f"eeq_solver must be 'auto', 'lu' or 'cg', got {solver!r}")
        self.solver = ("lu" if self.n <= LU_MAX_ATOMS else "cg") if solver == "auto" else solver
        self.reuse = reuse
        self.signature = signature   # identifies the system for EEQReuse (atom count, elements)
        self._lu = None            # (LU, pivots) of the augmented matrix
        self._y_ones = None        # A^-1 1 (CG path)
        self._sf = None            # (C, S) structure factors when they fit
        if self.periodic:
            r = torch.linalg.norm(self.edge_vec, dim=-1)
            src, dst = edge_index[0], edge_index[1]
            gamma = torch.rsqrt(self.rad[src] ** 2 + self.rad[dst] ** 2)
            self.kernel_e = _real_kernel(r, gamma, self.alpha)
            n_g = int(gvec.shape[0])
            if self.n * n_g <= SF_BUDGET:
                phase = self.pos @ gvec.t()
                self._sf = (torch.cos(phase), torch.sin(phase))
            self.n_g = n_g
        else:
            self.kernel_e = None
            self.n_g = 0

    # the constant operator
    def matvec(self, v: Tensor) -> Tensor:
        """``A v`` for the detached geometry, ``(N,)`` (no autograd graph)."""
        out = self.diag * v
        if self.periodic:
            src, dst = self.edge_index[0], self.edge_index[1]
            out = out + scatter_sum(self.kernel_e * v[src], dst, self.n)
            out = out + self._reciprocal(self.pos, v, self._sf)
        else:
            for i0 in range(0, self.n, ROW_CHUNK):
                i1 = min(i0 + ROW_CHUNK, self.n)
                out[i0:i1] += _molecular_block(self.pos, self.rad, i0, i1) @ v
        return out

    def _reciprocal(self, pos: Tensor, v: Tensor, sf) -> Tensor:
        """``[C g C^T + S g S^T] v``, from stored or chunked structure factors."""
        if sf is not None:
            c, s = sf
            return c @ (self.gfac * (c.t() @ v)) + s @ (self.gfac * (s.t() @ v))
        out = torch.zeros_like(v)
        step = max(1, SF_BUDGET // max(self.n, 1))
        for g0 in range(0, self.n_g, step):
            g1 = min(g0 + step, self.n_g)
            phase = pos @ self.gvec[g0:g1].t()
            c, s = torch.cos(phase), torch.sin(phase)
            gf = self.gfac[g0:g1]
            out = out + c @ (gf * (c.t() @ v)) + s @ (gf * (s.t() @ v))
        return out

    def assemble(self) -> Tensor:
        """The dense ``(N, N)`` matrix of the operator (for the LU path and tests)."""
        n = self.n
        if self.periodic:
            amat = torch.zeros((n, n), dtype=self.pos.dtype, device=self.pos.device)
            amat.index_put_((self.edge_index[1], self.edge_index[0]), self.kernel_e, accumulate=True)
            if self._sf is not None:
                c, s = self._sf
                amat = amat + (c * self.gfac) @ c.t() + (s * self.gfac) @ s.t()
            else:
                step = max(1, SF_BUDGET // max(n, 1))
                for g0 in range(0, self.n_g, step):
                    phase = self.pos @ self.gvec[g0:g0 + step].t()
                    c, s = torch.cos(phase), torch.sin(phase)
                    gf = self.gfac[g0:g0 + step]
                    amat = amat + (c * gf) @ c.t() + (s * gf) @ s.t()
        else:
            amat = torch.cat([_molecular_block(self.pos, self.rad, i0, min(i0 + ROW_CHUNK, n))
                              for i0 in range(0, n, ROW_CHUNK)], dim=0)
        return amat + torch.diag(self.diag)

    # solves (no autograd)
    def solve_augmented(self, b_top: Tensor, b_bot: Tensor,
                        role: str = "charges") -> tuple[Tensor, Tensor]:
        """Solve ``[[A, 1], [1^T, 0]] [y; mu] = [b_top; b_bot]``; returns ``(y, mu)``.

        ``role`` names the solve of the step: ``"charges"``, ``"residual"``
        (the correction of :func:`eeq_charges_large`) or ``"adjoint"`` (the
        backward pass); :class:`EEQReuse` keeps a separate history per role.
        """
        with torch.no_grad():
            if self.reuse is not None:
                return self.reuse.solve(self, b_top, b_bot, role)
            if self.solver == "lu":
                return self._solve_lu(b_top, b_bot)
            return self._solve_cg(b_top, b_bot)

    def _solve_lu(self, b_top: Tensor, b_bot: Tensor) -> tuple[Tensor, Tensor]:
        n = self.n
        if self._lu is None:
            amat = self.assemble()
            ones = torch.ones((n, 1), dtype=amat.dtype, device=amat.device)
            zero = torch.zeros((1, 1), dtype=amat.dtype, device=amat.device)
            self._full = torch.cat([torch.cat([amat, ones], dim=1),
                                    torch.cat([ones.t(), zero], dim=1)], dim=0)
            self._lu = torch.linalg.lu_factor(self._full)
        rhs = torch.cat([b_top, b_bot.reshape(1)]).unsqueeze(1)
        sol = torch.linalg.lu_solve(*self._lu, rhs)
        # iterative refinement: a float32 factor leaves ~1e-2 e in the raw
        # solution and, through the adjoint, ~1e-3 eV/A in the forces; each
        # round with the residual in float64 recovers about four digits
        rounds = 2 if sol.dtype == torch.float32 else 1
        full64 = self._full.double() if rounds == 2 else self._full
        for _ in range(rounds):
            resid = (rhs.double() - full64 @ sol.double()).to(sol.dtype) if rounds == 2 \
                else rhs - full64 @ sol
            sol = sol + torch.linalg.lu_solve(*self._lu, resid)
        sol = sol.squeeze(1)
        return sol[:n], sol[n]

    def _solve_cg(self, b_top: Tensor, b_bot: Tensor) -> tuple[Tensor, Tensor]:
        if self._y_ones is None:
            self._y_ones = self.conjugate_gradient(torch.ones_like(b_top))
        y1 = self.conjugate_gradient(b_top)
        y2 = self._y_ones
        mu = (y1.sum() - b_bot) / y2.sum()
        return y1 - mu * y2, mu

    def conjugate_gradient(self, b: Tensor, tol: float = CG_TOL,
                           maxiter: int = CG_MAXITER) -> Tensor:
        """Jacobi-preconditioned CG for ``A y = b`` (``A`` symmetric positive definite)."""
        b_norm = torch.linalg.norm(b)
        if float(b_norm) == 0.0:
            return torch.zeros_like(b)
        # single precision cannot reach the double-precision tolerance
        tol = max(tol, 100.0 * torch.finfo(b.dtype).eps)
        inv_diag = 1.0 / self.diag
        x = b * inv_diag
        r = b - self.matvec(x)
        zvec = r * inv_diag
        p = zvec.clone()
        rz = torch.dot(r, zvec)
        for _ in range(maxiter):
            if float(torch.linalg.norm(r)) <= tol * float(b_norm):
                return x
            ap = self.matvec(p)
            a = rz / torch.dot(p, ap)
            x = x + a * p
            r = r - a * ap
            zvec = r * inv_diag
            rz_new = torch.dot(r, zvec)
            p = zvec + (rz_new / rz) * p
            rz = rz_new
        raise RuntimeError(f"EEQ conjugate gradient did not converge in {maxiter} iterations "
                           f"(residual {float(torch.linalg.norm(r) / b_norm):.2e})")

    # the differentiable operator (for the implicit-function residual)
    def apply_differentiable(self, pos: Tensor, edge_vec: Optional[Tensor], rad: Tensor,
                             diag: Tensor, q: Tensor, cell: Optional[Tensor] = None) -> Tensor:
        """``A(r) q`` on live tensors in recomputed blocks (memory ``O(E + N c)`` at any order).

        ``cell`` is the live lattice (periodic case): the reciprocal vectors
        and weights are rebuilt from it so strain derivatives (stress) flow.
        """
        out = diag * q
        if self.periodic:
            src, dst = self.edge_index[0], self.edge_index[1]
            r = torch.linalg.norm(edge_vec, dim=-1)
            gamma = torch.rsqrt(rad[src] ** 2 + rad[dst] ** 2)
            out = out + scatter_sum(_real_kernel(r, gamma, self.alpha) * q[src], dst, self.n)
            gvec, gfac = reciprocal_weights(self.grid, cell, self.alpha)
            step = self.n_g if self._sf is not None else max(1, SF_BUDGET // max(self.n, 1))
            for g0 in range(0, self.n_g, step):
                out = out + recompute(_reciprocal_block, pos, q, gvec[g0:g0 + step],
                                      gfac[g0:g0 + step])
        else:
            blocks = []
            for i0 in range(0, self.n, ROW_CHUNK):
                i1 = min(i0 + ROW_CHUNK, self.n)
                blocks.append(recompute(lambda p, a, v, i0=i0, i1=i1: _molecular_block_matvec(p, a, v, i0, i1),
                                        pos, rad, q))
            out = out + torch.cat(blocks)
        return out


class EEQReuse:
    """Carry the EEQ solve from one structure to the next of an MD run or optimization.

    Between the steps of molecular dynamics (or of a geometry optimization) the
    atoms move by hundredths of an Angstrom and the EEQ matrix hardly changes,
    so the previous step's solve is worth reusing instead of assembling and
    factorizing the matrix every step:

    * the preconditioner is the explicit inverse of an *earlier* step's matrix,
      formed once and re-formed only when a solve needs more than ``refresh``
      iterations; applying it is one matrix-vector product;
    * each of the step's solves is conjugate gradients from a good initial
      guess: the charges from a quadratic extrapolation of the last three
      steps, the adjoint (forces) from the previous step's adjoint, ``A^-1 1``
      (the charge constraint) from the previous step's; the residual
      correction of :func:`eeq_charges_large` uses an absolute tolerance
      scaled to the charge right-hand side, so it costs nothing when the first
      solve was already converged.

    The results do not depend on the reuse beyond the tolerance ``tol``
    (relative residual; by default 1e-9 in float64 and 1e-6 in float32): a
    poor preconditioner or guess only costs iterations. If a solve
    does not converge within ``maxiter`` the preconditioner is re-formed from
    the current matrix and the solve retried once, and failing that the step
    falls back to the LU path.

    Measured on 5001 water atoms at 0.997 g/cm^3 (A100, cutoff_eeq 16 A, a
    real 0.5 fs trajectory): the three solves of a step take 20 ms in float32
    and 59 ms in float64, against 0.13-0.31 s for assembly + LU, with one
    inverse formed for the whole run.

    One instance serves one sequence of structures of the same system (atom
    count, elements, dtype, device); it resets itself when any of these
    change, and :class:`~xnn.common.models.d4.DFTD4` uses it only for a
    structure evaluated on its own, never inside a batch. It holds an
    ``(N, N)`` matrix (0.1 GB at 5000 atoms in float32).

    Parameters
    ----------
    tol : float or None
        Relative residual of every solve; ``None`` picks it from the dtype.
    refresh : int
        Re-form the preconditioner for the next step when a step's solves need
        more than this many iterations in total.
    maxiter : int
        Iterations after which a solve is considered failed.
    """

    def __init__(self, tol: Optional[float] = None, refresh: int = 15, maxiter: int = 200):
        self.tol = tol
        self.refresh = int(refresh)
        self.maxiter = int(maxiter)
        self.stats = {"solves": 0, "iterations": 0, "preconditioners": 0, "fallbacks": 0}
        self.reset()

    def reset(self) -> None:
        """Forget the previous structures (a new run, or a different system)."""
        self._key = None
        self._inv: Optional[Tensor] = None
        self._stale = False
        self._history: list = []
        self._adjoint: Optional[Tensor] = None
        self._ones: Optional[Tensor] = None
        self._main_norm: Optional[float] = None

    def _tolerance(self, dtype: torch.dtype) -> float:
        if self.tol is not None:
            return float(self.tol)
        # float32: 1e-6, not 100 machine epsilons (1.2e-5), which left 2-5e-4 e
        # in the charges over a trajectory where the refined LU keeps 3e-6 e;
        # the tighter tolerance costs one or two iterations per solve
        return 1e-9 if dtype == torch.float64 else 1e-6

    def _precondition(self, system: EEQSystem) -> None:
        amat = system.assemble()
        try:
            self._inv = torch.cholesky_inverse(torch.linalg.cholesky(amat))
        except RuntimeError:          # not numerically positive definite
            self._inv = torch.linalg.inv(amat)
        self._stale = False
        self.stats["preconditioners"] += 1

    def _pcg(self, system: EEQSystem, b: Tensor, x0: Tensor, tol_abs: float):
        """PCG for ``A y = b``; returns ``(y, iterations)``, iterations ``-1`` if not converged."""
        x = x0.clone()
        r = b - system.matvec(x)
        if float(torch.linalg.norm(r)) <= tol_abs:
            return x, 0
        z = self._inv @ r
        p = z.clone()
        rz = torch.dot(r, z)
        for it in range(1, self.maxiter + 1):
            ap = system.matvec(p)
            a = rz / torch.dot(p, ap)
            x = x + a * p
            r = r - a * ap
            if float(torch.linalg.norm(r)) <= tol_abs:
                return x, it
            z = self._inv @ r
            rz_new = torch.dot(r, z)
            p = z + (rz_new / rz) * p
            rz = rz_new
        return x, -1

    def solve(self, system: EEQSystem, b_top: Tensor, b_bot: Tensor,
              role: str = "charges") -> tuple[Tensor, Tensor]:
        """The constrained solve of ``system`` for one ``role`` of the step
        (``"charges"``, ``"residual"``, ``"adjoint"``; see
        :meth:`EEQSystem.solve_augmented`)."""
        with torch.no_grad():
            key = (system.n, b_top.dtype, b_top.device, system.signature)
            if key != self._key:
                self.reset()
                self._key = key
            if self._inv is None or self._stale:
                self._precondition(system)
            call = {"charges": 0, "residual": 1}.get(role, 2)
            out = self._attempt(system, b_top, b_bot, call)
            if out is None:                # re-form from this matrix and retry once
                self._precondition(system)
                out = self._attempt(system, b_top, b_bot, call)
            if out is None:
                self.stats["fallbacks"] += 1
                self._stale = True
                return system._solve_lu(b_top, b_bot)
            return out

    def _attempt(self, system: EEQSystem, b_top: Tensor, b_bot: Tensor, call: int):
        tol = self._tolerance(b_top.dtype)
        its = 0
        y2 = system.__dict__.get("_reuse_y2")
        if y2 is None:
            ones = torch.ones_like(b_top)
            x0 = self._ones if self._ones is not None else ones / system.diag
            y2, i = self._pcg(system, ones, x0, tol * float(torch.linalg.norm(ones)))
            if i < 0:
                return None
            its += i
        b_norm = float(torch.linalg.norm(b_top))
        if call == 0:                     # the charges
            h = self._history
            if len(h) >= 3:
                x0 = 3.0 * h[-1] - 3.0 * h[-2] + h[-3]
            elif len(h) == 2:
                x0 = 2.0 * h[-1] - h[-2]
            elif h:
                x0 = h[-1]
            else:
                x0 = b_top / system.diag
            y1, i = self._pcg(system, b_top, x0, tol * b_norm)
        elif call == 1:                   # the residual correction: near zero
            scale = self._main_norm if self._main_norm is not None else b_norm
            y1, i = self._pcg(system, b_top, torch.zeros_like(b_top), tol * scale)
        else:                             # the adjoint (forces), and any higher order
            x0 = self._adjoint if self._adjoint is not None else b_top / system.diag
            y1, i = self._pcg(system, b_top, x0, tol * b_norm)
        if i < 0:
            return None
        its += i
        # commit the state only after success
        system._reuse_y2 = y2
        self._ones = y2
        if call == 0:
            self._main_norm = b_norm
            self._history = (self._history + [y1])[-3:]
        elif call >= 2:
            self._adjoint = y1
        self.stats["solves"] += 1
        self.stats["iterations"] += its
        if its > self.refresh:
            self._stale = True
        mu = (y1.sum() - b_bot) / y2.sum()
        return y1 - mu * y2, mu


def _real_kernel(r: Tensor, gamma: Tensor, alpha: float) -> Tensor:
    """Screened real-space kernel ``[erf(gamma r) - erf(alpha r)] / r`` (``alpha = 0``: bare erf)."""
    k = torch.erf(gamma * r)
    if alpha > 0.0:
        k = k - torch.erf(alpha * r)
    return k / r


def _molecular_block(pos: Tensor, rad: Tensor, i0: int, i1: int) -> Tensor:
    """Rows ``i0:i1`` of the molecular kernel ``erf(gamma r) / r`` (zero diagonal)."""
    d = pos[i0:i1, None, :] - pos[None, :, :]
    r2 = (d * d).sum(-1)
    idx = torch.arange(i0, i1, device=pos.device)
    own = torch.zeros_like(r2, dtype=torch.bool)
    own[idx - i0, idx] = True
    r = torch.sqrt(torch.where(own, torch.ones_like(r2), r2))
    gamma = torch.rsqrt(rad[i0:i1, None] ** 2 + rad[None, :] ** 2)
    return torch.where(own, torch.zeros_like(r), torch.erf(gamma * r) / r)


def _molecular_block_matvec(pos: Tensor, rad: Tensor, q: Tensor, i0: int, i1: int) -> Tensor:
    return _molecular_block(pos, rad, i0, i1) @ q


def _reciprocal_block(pos: Tensor, q: Tensor, gvec: Tensor, gfac: Tensor) -> Tensor:
    phase = pos @ gvec.t()
    c, s = torch.cos(phase), torch.sin(phase)
    return c @ (gfac * (c.t() @ q)) + s @ (gfac * (s.t() @ q))


class _AugmentedSolve(torch.autograd.Function):
    """``J^{-1} [b_top; b_bot]`` for the constant EEQ Jacobian ``J``.

    ``J`` is symmetric, so the backward is the same solve applied to the
    incoming gradients; it is expressed through ``apply`` again, which makes
    every order of differentiation another ``O(1)``-memory solve.
    """

    @staticmethod
    def forward(ctx, system: EEQSystem, b_top: Tensor, b_bot: Tensor, role: str = "residual"):
        ctx.system = system
        y, mu = system.solve_augmented(b_top.detach(), b_bot.detach(), role)
        return y, mu.reshape(())

    @staticmethod
    def backward(ctx, g_y: Tensor, g_mu: Optional[Tensor]):
        if g_mu is None:
            g_mu = torch.zeros((), dtype=g_y.dtype, device=g_y.device)
        lam, kappa = _AugmentedSolve.apply(ctx.system, g_y, g_mu, "adjoint")
        return None, lam, kappa, None


def eeq_charges_large(system: EEQSystem, pos: Tensor, edge_vec: Optional[Tensor],
                      rad: Tensor, diag: Tensor, x: Tensor, total_charge: Tensor,
                      cell: Optional[Tensor] = None) -> Tensor:
    """Constrained EEQ charges with implicit differentiation, ``(N,)``.

    Parameters
    ----------
    system : EEQSystem
        The constant operator built from the detached geometry.
    pos, edge_vec : Tensor
        Live positions ``(N, 3)`` and (periodic) edge vectors ``(E, 3)`` in
        bohr, through which gradients flow.
    rad, diag : Tensor
        Live charge widths and diagonal ``(N,)`` (element tables; constant).
    x : Tensor
        Right-hand side ``-chi + k_CN sqrt(CN)`` ``(N,)`` (depends on positions).
    total_charge : Tensor
        Scalar net charge.
    cell : Tensor or None
        Live lattice vectors ``(3, 3)`` in bohr (periodic case).
    """
    total = total_charge.to(x.dtype).reshape(())
    q0, mu0 = system.solve_augmented(x.detach(), total.detach(), "charges")
    res_top = x - system.apply_differentiable(pos, edge_vec, rad, diag, q0, cell) - mu0
    res_bot = total - q0.sum()
    dq, _ = _AugmentedSolve.apply(system, res_top, res_bot, "residual")
    return q0 + dq
