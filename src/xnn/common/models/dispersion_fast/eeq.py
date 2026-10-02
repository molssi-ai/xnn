"""The large-regime EEQ operator, evaluated without trigonometry in the solver loop.

:class:`~xnn.common.models.eeq.EEQSystem` applies the Ewald reciprocal sum
through structure factors ``cos(G . r_i)``, ``sin(G . r_i)`` of shape ``(N,
N_G)``. Above :data:`~xnn.common.models.eeq.SF_BUDGET` they are recomputed at
every matrix-vector product, which is 85-90% of a D4 step at 12-24k atoms (2e9
sines and cosines per product at 24k atoms). :class:`FastEEQSystem` computes the
same operator another way:

* **Factorized phases.** With ``G = m_1 b_1 + m_2 b_2 + m_3 b_3`` and ``theta_k =
  b_k . r``, ``exp(i G . r) = exp(i (m_1 theta_1 + m_2 theta_2)) exp(i m_3
  theta_3)``. The reciprocal vectors are grouped into columns of fixed ``(m_1,
  m_2)``, and the two factors are tabulated once per system: ``B (C, N)`` over
  the columns and ``P (N, 2 M_3 + 1)`` over ``m_3``. ``A_rec v`` is then two
  complex matrix products (``S = B (v P)``, the structure factor of ``v`` on
  every column, and ``B^H (g S)`` back onto the atoms) and no trigonometry.
  ``G`` and ``-G`` contribute equally, so only half of the set is kept, with
  twice the weight. The tables are formed in float64 from float64 phases, so
  in float32 they are more accurate than the reference's float32 phases.
* **Real space.** The screened kernel of the neighbor list becomes a sparse
  CSR matrix (images of the same pair merged), applied with one sparse
  matrix product.
* **Solves.** The charge solve and the constraint solve ``A y = 1`` run in
  lockstep as two right-hand sides of one conjugate-gradient loop. The
  residual correction of :func:`~xnn.common.models.eeq.eeq_charges_large`, which
  is zero at convergence, stops at the charge solve's tolerance instead of
  converging its own (tiny) right-hand side again.
* **Implicit differentiation.** The differentiable application of ``A(r)``
  uses the same factorization on live positions and cell, in column blocks
  that are recomputed in the backward pass.

The operator is the same matrix, so the charges agree with the reference
solve to the solver tolerance.
"""
from __future__ import annotations

from typing import Iterator, Optional, Tuple

import torch
from torch import Tensor

from ..eeq import CG_MAXITER, CG_TOL, EEQSystem, _real_kernel, reciprocal_weights
from ..ops import scatter_sum
from ..reciprocal import PhaseColumns, complex_dtype, half_space, structure_factors
from ..recompute import recompute
from . import _triton

#: atoms up to which the fast system factorizes the assembled matrix; conjugate
#: gradients above (the reference switches at 12000 atoms)
FAST_LU_MAX_ATOMS = 2000
#: complex entries of the ``(C, N)`` column table kept between products; above
#: it the table is rebuilt in blocks at every product
TABLE_BUDGET = 4 * 10 ** 8
#: complex entries per column block of the differentiable reciprocal sum
LIVE_BUDGET = 5 * 10 ** 7
#: relative residual of the float32 solves: the reference's 100 machine epsilons
#: (1.2e-5) leave about 3e-5 e in the charges of a 5k-atom box, where its refined
#: LU solve reaches 2e-6 e; 1e-6 matches that at a few more iterations
#: (EEQReuse uses the same value)
CG_TOL_FLOAT32 = 1e-6
#: reciprocal vectors (smallest |G|, largest weight) in the low-rank part of the
#: conjugate-gradient preconditioner; 0 for plain Jacobi. Measured on water boxes
#: (A100, 5k-24k atoms): 256 cuts the operator applications of a step 3-5x
#: (94 -> 31 at 5k, 154 -> 43 at 24k atoms in float32) and is the fastest or
#: within 1% of it; more vectors save few iterations and cost more set-up
LOW_K_VECTORS = 256


class FastEEQSystem(EEQSystem):
    """:class:`~xnn.common.models.eeq.EEQSystem` with the fast operator and solves.

    Takes the arguments of the reference class plus the lattice ``cell``
    ``(3, 3)`` in bohr (periodic case), from which the phase tables are built.
    Right-hand sides may be ``(N,)`` or ``(N, K)``.

    Preconditioner: the long-range Coulomb modes (the smallest reciprocal
    vectors, whose weights ``g ~ 1/G^2`` are the largest) are what a Jacobi
    preconditioner leaves untouched and what slows the conjugate gradients. The
    periodic system therefore uses ``M = D + U W U^T``: ``U = [cos, sin]`` of the
    :data:`LOW_K_VECTORS` smallest vectors, ``W`` their weights, and ``D`` the
    full diagonal of ``A`` without that part; ``M^-1`` is applied through the
    Woodbury identity (one ``2L x 2L`` Cholesky factor per structure, two
    ``N x 2L`` products per iteration). A preconditioner only changes how fast
    the iteration converges, not what it converges to.
    """

    keep_structure_factors = False

    def __init__(self, diag: Tensor, rad: Tensor, pos: Tensor,
                 edge_index: Optional[Tensor] = None, edge_vec: Optional[Tensor] = None,
                 alpha: float = 0.0, gvec: Optional[Tensor] = None,
                 gfac: Optional[Tensor] = None, grid: Optional[Tensor] = None,
                 solver: str = "auto", reuse=None, signature: Optional[tuple] = None,
                 cell: Optional[Tensor] = None):
        super().__init__(diag, rad, pos, edge_index, edge_vec, alpha, gvec, gfac, grid,
                         solver, reuse, signature)
        if solver == "auto":
            self.solver = "lu" if self.n <= FAST_LU_MAX_ATOMS else "cg"
        self._main_norm: Optional[float] = None
        self._cdtype = complex_dtype(self.pos.dtype)
        self._csr = None
        if self.periodic:
            if cell is None:
                raise ValueError("FastEEQSystem needs the cell of a periodic structure")
            self._cell = cell.detach()
            self._build_real_space()
            self._build_columns(grid, gfac)
        self._preconditioner = self._build_preconditioner()

    # set-up
    def _build_real_space(self) -> None:
        # the CSR matrix of the real-space kernel, rows = dst, built directly so
        # that only its values and 32-bit columns outlive the set-up (a COO
        # coalesce left the 64-bit index pairs alive: 1.1 GB at 71 million edges)
        src, dst = self.edge_index[0], self.edge_index[1]
        n = self.n
        key, order = torch.sort(dst.to(torch.int64) * n + src)
        values = self.kernel_e[order]
        del order
        key, inverse = torch.unique_consecutive(key, return_inverse=True)
        if key.numel() < values.numel():      # one pair at several images (small cells)
            values = values.new_zeros(key.numel()).index_add_(0, inverse, values)
        del inverse
        # 32-bit indices: half the index traffic of the sparse product
        index_dtype = torch.int32 if values.numel() < 2 ** 31 else torch.int64
        crow = torch.zeros(n + 1, dtype=index_dtype, device=key.device)
        crow[1:] = torch.cumsum(torch.bincount(torch.div(key, n, rounding_mode="floor"),
                                               minlength=n), 0)
        col = torch.remainder(key, n).to(index_dtype)
        del key
        try:
            self._csr = torch.sparse_csr_tensor(crow, col, values, (n, n))
        except RuntimeError:          # no sparse kernels for this device / dtype
            self._csr = None

    def _build_preconditioner(self):
        """The low-rank + diagonal preconditioner (periodic), or ``None`` (Jacobi)."""
        if not self.periodic or LOW_K_VECTORS <= 0 or self._grid_half.shape[0] == 0:
            return None
        f64 = torch.float64
        weights = self._w_half.to(f64)
        n_low = min(LOW_K_VECTORS, int(weights.shape[0]))
        low = torch.topk(weights, n_low).indices
        w_low = weights[low]
        phase = self._theta @ self._grid_half[low].to(f64).t()                  # (N, L)
        u = torch.cat([torch.cos(phase), torch.sin(phase)], dim=1)              # (N, 2L)
        # the rest of A's diagonal: the other reciprocal vectors and the
        # real-space self-images
        src, dst = self.edge_index[0], self.edge_index[1]
        own = src == dst
        d = (self.diag.to(f64) + (weights.sum() - w_low.sum())
             + scatter_sum(self.kernel_e[own].to(f64), dst[own], self.n))
        d_inv = 1.0 / d
        inner = torch.diag(1.0 / torch.cat([w_low, w_low])) + u.t() @ (d_inv[:, None] * u)
        return u, d_inv, torch.linalg.cholesky(inner)

    def precondition(self, r: Tensor) -> Tensor:
        """``M^-1 r`` for ``r`` of shape ``(N,)`` or ``(N, K)``: the matrix-free
        preconditioner :class:`~xnn.common.models.eeq.EEQReuse` uses for large
        systems in place of a dense inverse."""
        if r.dim() == 1:
            return self._precondition(r[:, None])[:, 0]
        return self._precondition(r)

    def _precondition(self, r: Tensor) -> Tensor:
        """``M^-1 r`` for ``r (N, K)``."""
        if self._preconditioner is None:
            return r / self.diag[:, None]
        u, d_inv, chol = self._preconditioner
        y = d_inv[:, None] * r.to(torch.float64)
        s = torch.cholesky_solve(u.t() @ y, chol)
        return (y - d_inv[:, None] * (u @ s)).to(r.dtype)

    def _build_columns(self, grid: Tensor, gfac: Tensor) -> None:
        """Group the half space of the reciprocal set into ``(m_1, m_2)`` columns."""
        half = half_space(grid)
        self._grid_half = grid[half]
        self._columns = PhaseColumns(self._grid_half)
        self._w_half = 2.0 * gfac[half]
        self._weight = self._columns.scatter(self._w_half)
        # float64 phases theta_k = b_k . r of the detached geometry
        theta = PhaseColumns.phases(self.pos, self._cell)
        self._theta = theta
        self._p3 = self._columns.axis_table(theta, self._cdtype)           # (N, n_m3)
        self._p3_conj = self._p3.conj().resolve_conj()
        n_cols = self._columns.n_cols
        self._table = (self._columns.column_table(theta, 0, n_cols, self._cdtype)
                       if n_cols * self.n <= TABLE_BUDGET else None)

    def _column_blocks(self) -> Iterator[Tuple[int, int, Tensor]]:
        if self._table is not None:
            yield 0, self._columns.n_cols, self._table
            return
        for c0, c1 in self._columns.blocks(self.n, TABLE_BUDGET // 4):
            yield c0, c1, self._columns.column_table(self._theta, c0, c1, self._cdtype)

    # the constant operator
    def matvec(self, v: Tensor) -> Tensor:
        """``A v`` for ``v`` of shape ``(N,)`` or ``(N, K)`` (no autograd graph)."""
        one = v.dim() == 1
        # contiguous: the CUDA sparse product misreads strided dense operands
        vv = (v[:, None] if one else v).contiguous()
        out = self.diag[:, None] * vv
        if self.periodic:
            out = out + self._real_space(vv) + self._reciprocal_fast(vv)
        else:
            out = out + self._molecular(vv)
        return out[:, 0] if one else out

    def _real_space(self, vv: Tensor) -> Tensor:
        if self._csr is not None:
            return torch.sparse.mm(self._csr, vv)
        src, dst = self.edge_index[0], self.edge_index[1]
        return scatter_sum(self.kernel_e[:, None] * vv[src], dst, self.n)

    def _reciprocal_fast(self, vv: Tensor) -> Tensor:
        n, k = vv.shape
        n_m3 = self._columns.n_m3
        y = torch.zeros((n, k * n_m3), dtype=self._cdtype, device=vv.device)
        for c0, c1, table in self._column_blocks():
            s = structure_factors(self._columns, table, self._p3, vv)
            s = (s * self._weight[c0:c1, None, :]).reshape(c1 - c0, k * n_m3)
            y = y + table.mH @ s
        return (y.view(n, k, n_m3) * self._p3_conj[:, None, :]).real.sum(-1).to(vv.dtype)

    def _molecular(self, vv: Tensor) -> Tensor:
        if _triton.supported(vv.device, vv.dtype):
            from .coulomb import coulomb_matvec
            return coulomb_matvec(self.pos, self.rad, vv)
        cols = [EEQSystem.matvec(self, vv[:, j]) - self.diag * vv[:, j]
                for j in range(vv.shape[1])]
        return torch.stack(cols, dim=1)

    # solves
    def solve_augmented(self, b_top: Tensor, b_bot: Tensor,
                        role: str = "charges") -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            if self.reuse is not None:
                return self.reuse.solve(self, b_top, b_bot, role)
            if self.solver == "lu":
                return self._solve_lu(b_top, b_bot)
            return self._solve_cg_lockstep(b_top, b_bot, role)

    def _solve_cg_lockstep(self, b_top: Tensor, b_bot: Tensor, role: str):
        tol = self._tolerance(b_top.dtype)
        if self._y_ones is None:
            both = self.conjugate_gradient(torch.stack([b_top, torch.ones_like(b_top)], dim=1))
            y1, self._y_ones = both[:, 0], both[:, 1]
        else:
            atol = None
            if role == "residual" and self._main_norm is not None:
                # the correction is zero at convergence: it needs the charge
                # solve's accuracy, not its own relative tolerance
                atol = tol * self._main_norm
            y1 = self.conjugate_gradient(b_top, atol=atol)
        if role == "charges":
            self._main_norm = float(torch.linalg.norm(b_top))
        y2 = self._y_ones
        mu = (y1.sum() - b_bot) / y2.sum()
        return y1 - mu * y2, mu

    @staticmethod
    def _tolerance(dtype: torch.dtype) -> float:
        return CG_TOL if dtype == torch.float64 else CG_TOL_FLOAT32

    def conjugate_gradient(self, b: Tensor, tol: Optional[float] = None,
                           maxiter: int = CG_MAXITER, atol: Optional[float] = None) -> Tensor:
        """Preconditioned CG for ``A Y = B``, the columns of ``B`` in lockstep.

        Each column stops at ``|r| <= tol |b|`` (or ``|r| <= atol``); a
        finished column takes no further steps while the others converge.
        ``tol`` defaults to :data:`~xnn.common.models.eeq.CG_TOL` in float64 and
        :data:`CG_TOL_FLOAT32` in float32.
        """
        one = b.dim() == 1
        bb = b[:, None] if one else b
        b_norm = torch.linalg.norm(bb, dim=0)
        tol = self._tolerance(bb.dtype) if tol is None else tol
        thresh = tol * b_norm if atol is None else torch.full_like(b_norm, float(atol))
        x = self._precondition(bb)
        r = bb - self.matvec(x)
        z = self._precondition(r)
        p = z.clone()
        rz = (r * z).sum(0)
        zero = torch.zeros_like(rz)
        for _ in range(maxiter):
            done = torch.linalg.norm(r, dim=0) <= thresh
            if bool(done.all()):
                return x[:, 0] if one else x
            ap = self.matvec(p)
            step = torch.where(done, zero, rz / (p * ap).sum(0))
            x = x + step * p
            r = r - step * ap
            z = self._precondition(r)
            rz_new = (r * z).sum(0)
            beta = torch.where(done, zero, rz_new / rz)
            p = z + beta * p
            rz = torch.where(done, rz, rz_new)
        resid = float((torch.linalg.norm(r, dim=0) / b_norm.clamp(min=1e-300)).max())
        raise RuntimeError(f"EEQ conjugate gradient did not converge in {maxiter} iterations "
                           f"(residual {resid:.2e})")

    # the differentiable operator
    def apply_differentiable(self, pos: Tensor, edge_vec: Optional[Tensor], rad: Tensor,
                             diag: Tensor, q: Tensor, cell: Optional[Tensor] = None) -> Tensor:
        if not self.periodic:
            return super().apply_differentiable(pos, edge_vec, rad, diag, q, cell)
        out = diag * q
        src, dst = self.edge_index[0], self.edge_index[1]
        r = torch.linalg.norm(edge_vec, dim=-1)
        gamma = torch.rsqrt(rad[src] ** 2 + rad[dst] ** 2)
        out = out + scatter_sum(_real_kernel(r, gamma, self.alpha) * q[src], dst, self.n)
        # the weights g(G) of the live cell (strain derivatives), then the
        # column blocks, recomputed in the backward pass
        _, g = reciprocal_weights(self._grid_half, cell, self.alpha)
        for c0, c1 in self._columns.blocks(self.n, LIVE_BUDGET):
            out = out + recompute(
                lambda p, v, lat, w, c0=c0, c1=c1: self._reciprocal_live(p, v, lat, w, c0, c1),
                pos, q, cell, 2.0 * g)
        return out

    def _reciprocal_live(self, pos: Tensor, q: Tensor, cell: Tensor, weights: Tensor,
                         c0: int, c1: int) -> Tensor:
        """Columns ``c0:c1`` of ``A_rec(r, cell) q`` from live tensors (differentiable)."""
        theta = PhaseColumns.phases(pos, cell)
        table = self._columns.column_table(theta, c0, c1, self._cdtype)   # (c, N)
        p3 = self._columns.axis_table(theta, self._cdtype)                 # (N, n_m3)
        s = structure_factors(self._columns, table, p3, q[:, None])[:, 0]  # (c, n_m3)
        y = table.mH @ (s * self._columns.scatter(weights, c0, c1))        # (N, n_m3)
        return (y * p3.conj()).real.sum(-1).to(q.dtype)
