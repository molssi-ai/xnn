"""Closed-form fit of a (s)GDML force field.

The forces of the training geometries are interpolated by the gradient of the
kernel expansion of :class:`~xnn.krnn.models.sgdml.SGDML`: with the descriptor
Jacobians ``J_i`` and ``H(D) = g(r) I + t(r) D D^T`` (``g = -k'/r``,
``t = g'/r``), the force-force block of the kernel matrix is

``K_ij = -J_i^T sum_q H(x_i - P_q x_j) P_q J_j``

and the coefficients solve ``(K - lam I) alpha = F / s`` (Chmiela et al., Sci. Adv.
3, e1603015, 2017, eq. 2), with ``s`` the standard deviation of the force
components. With ``energy_constraint`` the energies join the system as extra
rows (negated, which keeps the matrix symmetric). The system is solved by a
Cholesky factorization of the positive-definite ``lam I - K`` (an LU solve when
that fails). The integration constant is the mean deviation of the predicted
training energies from their labels.
"""
from __future__ import annotations

import math
import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from xnn.common.data import AtomicGraph

from ..kernels import safe_norm
from ..models.sgdml import SGDML


def _stack(model: SGDML, graphs: Sequence[AtomicGraph]):
    """Positions ``(M, N, 3)``, cells ``(M, 3, 3)`` or ``None``, energies and forces."""
    pos, cells, energies, forces = [], [], [], []
    for g in graphs:
        p, c = model.geometry(g)
        pos.append(p.detach().to(torch.float64))
        if c is not None:
            cells.append(c.detach().to(torch.float64))
        energies.append(None if g.energy is None else g.energy.reshape(-1).to(torch.float64))
        forces.append(None if g.forces is None else g.forces.reshape(-1, model.n_atoms, 3).to(torch.float64))
    pos = torch.cat(pos)
    cell = torch.cat(cells) if cells else None
    e = torch.cat(energies) if all(x is not None for x in energies) else None
    if any(x is None for x in forces):
        raise ValueError("every training structure needs force labels")
    return pos, cell, e, torch.cat(forces)


def _incidence(rows: Tensor, cols: Tensor, signs: Tensor, shape: Tuple[int, int]) -> Tensor:
    """A sparse (CSR) matrix with the entries ``signs`` at ``(rows, cols)``."""
    values = signs.to(torch.float64)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")
        coo = torch.sparse_coo_tensor(torch.stack([rows, cols]), values, shape, check_invariants=True)
        return coo.coalesce().to_sparse_csr()


def _incidences(model: SGDML, device: torch.device) -> Tuple[Tensor, Tensor, Tensor]:
    """The placements of the kernel-matrix sums as sparse matrices of +1 and -1.

    ``pair`` ``(N N, Q D)`` adds the outer product of rows ``d`` and ``p_q(d)`` to its four
    atom pairs; ``atom`` ``(N, D)`` adds row ``d`` to its atoms; ``perm`` ``(Q N, Q D)`` adds
    row ``p_q(d)`` to its atoms, for every ``q``.
    """
    n = model.n_atoms
    a, b = model.descriptor.pairs.to(device)
    pidx = model.perm_index.to(device)
    q, d = pidx.shape
    ap, bp = a[pidx], b[pidx]
    aq, bq = a.expand(q, d), b.expand(q, d)
    src = torch.arange(q * d, device=device)
    one = torch.ones(q * d, device=device)
    pair = _incidence(torch.cat([(aq * n + ap).reshape(-1), (aq * n + bp).reshape(-1), (bq * n + ap).reshape(-1),
                                 (bq * n + bp).reshape(-1)]), src.repeat(4), torch.cat([one, -one, -one, one]),
                      (n * n, q * d))
    rows = torch.arange(d, device=device)
    atom = _incidence(torch.cat([a, b]), rows.repeat(2), torch.cat([one[:d], -one[:d]]), (n, d))
    shift = (torch.arange(q, device=device)[:, None] * n)
    perm = _incidence(torch.cat([(shift + ap).reshape(-1), (shift + bp).reshape(-1)]), src.repeat(2),
                      torch.cat([one, -one]), (q * n, q * d))
    return pair, atom, perm


def closed_under_inversion(perms: Tensor) -> bool:
    """Whether the inverse of every permutation is in the set (the kernel matrix is then symmetric)."""
    rows = {tuple(p) for p in perms.tolist()}
    return all(tuple(torch.argsort(p).tolist()) in rows for p in perms)


def kernel_matrix(model: SGDML, x: Tensor, u: Tensor, energy_constraint: bool = False,
                  full: bool = True) -> Tensor:
    """The kernel matrix of the training set.

    The force block of training geometries ``i`` and ``j`` is
    ``-sum_q [g_q J_i^T P_q J_j + t_q (J_i^T D_q) (D_q^T P_q J_j)]`` with
    ``D_q = x_i - P_q x_j``. Every row of a descriptor Jacobian holds two atom
    blocks (``u_d`` on atom ``a_d``, ``-u_d`` on atom ``b_d``), so ``J_i^T P_q J_j``
    is a sum of 3x3 outer products ``u_i,d u_j,p(d)^T`` placed on the atom pairs,
    and ``J_i^T D_q`` and ``D_q^T P_q J_j`` are the rows scaled by ``D_q`` and summed
    onto atoms. Both placements are products with sparse matrices of +1 and -1.
    The matrix is symmetric when the permutations are closed under inversion; only
    the blocks ``j >= i`` are formed then.

    Parameters
    ----------
    model : SGDML
        Supplies the kernel and the permutations.
    x : Tensor
        ``(M, D)`` training descriptors.
    u : Tensor
        ``(M, D, 3)`` pair gradients of the descriptors
        (:meth:`~xnn.krnn.featurizers.inverse_distance.InverseDistances.pair_gradients`).
    energy_constraint : bool, optional
        Append the (negated) energy rows and columns.
    full : bool, optional
        Fill the whole matrix (the default). ``False`` leaves the strict upper
        triangle of a symmetric matrix unset, which a Cholesky factorization
        does not read.

    Returns
    -------
    Tensor
        ``(3NM, 3NM)``, or ``(3NM + M, 3NM + M)`` with energies.
    """
    m = x.shape[0]
    n3 = 3 * model.n_atoms
    rows = _BlockRows(model, x, u)
    off = m * n3
    k = x.new_zeros(off + (m if energy_constraint else 0), off + (m if energy_constraint else 0))
    symmetric = closed_under_inversion(model.perms)
    for i in range(m):
        j0 = i if symmetric else 0
        block, f_e, e_f, e_e = rows(i, j0, energy_constraint)
        rows_i = slice(i * n3, (i + 1) * n3)
        if full or not symmetric:
            k[rows_i, j0 * n3:off] = block.permute(1, 0, 2).reshape(n3, -1)
        if symmetric:
            k[j0 * n3:off, rows_i] = block.transpose(1, 2).reshape(-1, n3)
        if energy_constraint:
            k[off + i, j0 * n3:off] = e_f.reshape(-1)          # below the diagonal: always needed
            if full or not symmetric:
                k[rows_i, off + j0:] = f_e.transpose(0, 1)
                k[off + i, off + j0:] = e_e
            if symmetric:
                k[off + j0:, rows_i] = f_e
                k[j0 * n3:off, off + i] = e_f.reshape(-1)
                k[off + j0:, off + i] = e_e
    return k


def kernel_rows(model: SGDML, x: Tensor, u: Tensor, idx: Sequence[int], energy_constraint: bool = False) -> Tensor:
    """The rows of the kernel matrix (:func:`kernel_matrix`) of the training geometries ``idx``.

    Returns the ``3N`` force rows of every geometry in ``idx``, in that order,
    followed (with ``energy_constraint``) by their energy rows.
    """
    m, n3 = x.shape[0], 3 * model.n_atoms
    off = m * n3
    blocks = _BlockRows(model, x, u)
    out = x.new_zeros(len(idx) * (n3 + (1 if energy_constraint else 0)), off + (m if energy_constraint else 0))
    for r, i in enumerate(idx):
        block, f_e, e_f, e_e = blocks(int(i), 0, energy_constraint)
        out[r * n3:(r + 1) * n3, :off] = block.permute(1, 0, 2).reshape(n3, off)
        if energy_constraint:
            out[r * n3:(r + 1) * n3, off:] = f_e.transpose(0, 1)
            out[len(idx) * n3 + r, :off] = e_f.reshape(-1)
            out[len(idx) * n3 + r, off:] = e_e
    return out


class _BlockRows:
    """The blocks ``K_ij`` (``j >= j0``) of one training geometry ``i``, from :func:`kernel_matrix`."""

    def __init__(self, model: SGDML, x: Tensor, u: Tensor):
        self.n, self.d = model.n_atoms, x.shape[1]
        pidx = model.perm_index
        self.q = pidx.shape[0]
        self.pair, self.atom, self.perm = _incidences(model, x.device)
        self.x, self.u, self.kern = x, u, model.kernel
        self.xp = x[:, pidx]                                       # (M, Q, D)
        self.up = u[:, pidx].permute(1, 2, 0, 3).contiguous()      # (Q, D, M, 3)

    def __call__(self, i: int, j0: int, energy: bool):
        n, d, q, n3 = self.n, self.d, self.q, 3 * self.n
        mj = self.x.shape[0] - j0
        diff = (self.x[i] - self.xp[j0:]).permute(1, 2, 0)          # (Q, D, mj)
        r = safe_norm(diff.permute(2, 0, 1))                         # (mj, Q)
        g, t = self.kern.g_and_t(r)
        ui, uj = self.u[i], self.up[:, :, j0:]
        # g_q J_i^T P_q J_j: outer products of the pair gradients, placed on the atom pairs
        outer = (g.t()[:, None, :, None, None] * ui[None, :, None, :, None]) * uj[..., None, :]
        blocks = torch.sparse.mm(self.pair, outer.reshape(q * d, mj * 9))
        blocks = blocks.reshape(n, n, mj, 3, 3).permute(2, 0, 3, 1, 4).reshape(mj, n3, n3)
        # t_q (J_i^T D_q)(D_q^T P_q J_j): the rows scaled by D_q, summed onto atoms
        su = diff[..., None] * ui[None, :, None, :]                  # (Q, D, mj, 3)
        w = torch.sparse.mm(self.atom, su.transpose(0, 1).reshape(d, q * mj * 3))
        w = w.reshape(n, q, mj, 3).permute(2, 1, 0, 3).reshape(mj, q, n3)
        v = torch.sparse.mm(self.perm, (diff[..., None] * uj).reshape(q * d, mj * 3))
        v = v.reshape(q, n, mj, 3).permute(2, 0, 1, 3).reshape(mj, q, n3)
        block = -(blocks + torch.einsum("mq,mqa,mqb->mab", t, w, v))
        if not energy:
            return block, None, None, None
        f_e = (g[..., None] * w).sum(1)                  # force row i, energy column j: J_i^T sum_q g D
        e_f = -(g[..., None] * v).sum(1)                 # energy row i (negated): -sum_q g D^T P_q J_j
        e_e = -self.kern.k(r).sum(1)
        return block, f_e, e_f, e_e


def _solve(k: Tensor, y: Tensor, lam: float, lower_only: bool = False) -> Tensor:
    """``alpha`` with ``(K - lam I) alpha = y``, through the positive-definite ``lam I - K``.

    ``k`` is overwritten. With ``lower_only`` only its lower triangle is set (a
    symmetric matrix); the LU fallback then fills the upper triangle first.
    """
    a = k.neg_()
    a.diagonal().add_(lam)
    chol, info = torch.linalg.cholesky_ex(a)
    if int(info) == 0:
        return -torch.cholesky_solve(y.unsqueeze(1), chol).squeeze(1)
    if lower_only:
        a = torch.tril(a) + torch.tril(a, -1).transpose(0, 1)
    return -torch.linalg.solve(a, y)


def _use_iterative(solver: str, n: int, symmetric: bool, device: torch.device, max_memory: Optional[float]) -> bool:
    """Whether the fit takes the iterative solver: asked for, or (``"auto"``) when the analytic solve
    does not fit in ``max_memory`` GB (by default the device's memory)."""
    if solver not in ("auto", "analytic", "cg"):
        raise ValueError(f"solver must be 'auto', 'analytic' or 'cg', got {solver!r}")
    if solver == "analytic":
        return False
    if solver == "auto":
        from .sgdml_iterative import analytic_bytes, device_memory
        memory = device_memory(device) if max_memory is None else float(max_memory)
        if analytic_bytes(n) <= memory * 2 ** 30:
            return False
    if not symmetric:
        raise ValueError("the iterative solver needs a symmetric kernel matrix: permutations closed under inversion")
    return True


def _fit_one(model: SGDML, x: Tensor, u: Tensor, forces: Tensor, energies: Optional[Tensor],
             lam: float, use_energy: bool, energy_constraint: bool, solver: str = "auto",
             solver_options: Optional[dict] = None) -> None:
    m, n3 = x.shape[0], 3 * model.n_atoms
    y = forces.reshape(-1)
    mean_e = None
    if energy_constraint:
        if energies is None:
            raise ValueError("energy_constraint needs energy labels")
        mean_e = energies.mean()
        y = torch.cat([y, -energies + mean_e])
    std = y.std(unbiased=False)
    lower = closed_under_inversion(model.perms)
    options = dict(solver_options or {})
    if _use_iterative(solver, y.numel(), lower, x.device, options.get("max_memory")):
        from .sgdml_iterative import solve_iterative
        alpha, record = solve_iterative(model, x, u, y / std, lam, energy_constraint, **options)
        model.fit_info = {"solver": "cg", **record}
    else:
        alpha = _solve(kernel_matrix(model, x, u, energy_constraint, full=not lower), y / std, lam,
                       lower_only=lower)
        model.fit_info = {"solver": "analytic"}
    # w_j = J_j alpha_j: row d of J_j is u_d on atom a_d and -u_d on atom b_d
    a, b = model.descriptor.pairs
    al = alpha[:m * n3].reshape(m, model.n_atoms, 3)
    weights = (u * (al[:, a] - al[:, b])).sum(-1)
    model.train_desc = x.clone()
    model.train_weights = weights
    model.energy_coef = alpha[m * n3:].clone() if energy_constraint else x.new_zeros(0)
    model.std = std.clone()
    model.c = x.new_zeros(())
    if energy_constraint:
        model.c = mean_e.clone()
    elif use_energy and energies is not None:
        with torch.no_grad():
            pred = model.energy_of_descriptors(x)
        model.c = (energies - pred).mean()


def fit_sgdml(model: SGDML, graphs: Sequence[AtomicGraph], sigma: Union[None, float, Sequence[float]] = None,
              lam: float = 1e-10, perms=None, use_energy: bool = True, energy_constraint: bool = False,
              validation: Optional[Sequence[AtomicGraph]] = None,
              max_symmetry_samples: int = 1000, generator: Optional[torch.Generator] = None,
              solver: str = "auto", tol: float = 1e-4, n_inducing: Optional[int] = None,
              max_memory: Optional[float] = None, max_iter: Optional[int] = None) -> SGDML:
    """Fit a (s)GDML model to the forces (and energies) of its molecule.

    Parameters
    ----------
    model : SGDML
        The model (fitted in place, in float64).
    graphs : sequence of AtomicGraph
        Training structures with ``forces`` (and ``energy``) labels.
    sigma : float or sequence of float, optional
        Kernel length scale; several values are compared by the force RMSE on
        ``validation``. ``None`` keeps the model's.
    lam : float, optional
        Regularization, by default 1e-10.
    perms : array_like, optional
        ``(Q, N)`` atom permutations to use. When ``None`` they are found from
        the training geometries if the model uses symmetries
        (:func:`~xnn.krnn.common.permutations.find_permutations`), or the
        identity alone.
    use_energy : bool, optional
        Set the integration constant from the training energies, by default ``True``.
    energy_constraint : bool, optional
        Fit the energies together with the forces, by default ``False``.
    validation : sequence of AtomicGraph, optional
        Structures that choose among several ``sigma``.
    max_symmetry_samples : int, optional
        At most this many training geometries (a random subset) are matched to
        find the symmetries, by default 1000.
    generator : torch.Generator, optional
        Random numbers of that subset and of the inducing geometries.
    solver : str, optional
        ``"analytic"``: a Cholesky factorization of the kernel matrix (exact);
        ``"cg"``: preconditioned conjugate gradients without the matrix
        (:func:`~xnn.krnn.common.sgdml_iterative.solve_iterative`), accurate to
        ``tol``; ``"auto"`` (the default): analytic when its matrix fits in
        ``max_memory``, otherwise iterative.
    tol : float, optional
        Relative residual at which the iterative solve stops, by default 1e-4.
    n_inducing : int, optional
        Inducing geometries of the iterative solver's preconditioner; by default
        as many as ``max_memory`` holds.
    max_memory : float, optional
        Memory in GB for the choice of the solver and the preconditioner; by
        default the device's.
    max_iter : int, optional
        Iteration limit of the iterative solve, by default ten times the number of
        unknowns.

    Returns
    -------
    SGDML
        ``model``, fitted.
    """
    model.double()
    pos, cell, energies, forces = _stack(model, graphs)
    if perms is None:
        if model.use_symmetries:
            from .permutations import find_permutations
            sel = pos
            if pos.shape[0] > max_symmetry_samples:
                sel = pos[torch.randperm(pos.shape[0], generator=generator)[:max_symmetry_samples]]
            perms = find_permutations(sel.cpu().numpy(), model._species.cpu().numpy(),
                                      None if cell is None else cell[0].cpu().numpy())
        else:
            perms = torch.arange(model.n_atoms).unsqueeze(0)
    model.set_permutations(perms)
    if cell is not None:
        model.lattice = cell[0].clone()
    x, u = model.descriptor(pos, cell), model.descriptor.pair_gradients(pos, cell)
    sigmas = [model.sigma] if sigma is None else (
        [float(sigma)] if isinstance(sigma, (int, float)) else [float(s) for s in sigma])
    if len(sigmas) > 1 and not validation:
        raise ValueError("choosing among several sigma needs validation structures")
    best: Tuple[float, Optional[dict]] = (math.inf, None)
    for s in sigmas:
        model.set_kernel(s)
        _fit_one(model, x, u, forces, energies, lam, use_energy, energy_constraint, solver,
                 {"tol": tol, "n_inducing": n_inducing, "max_memory": max_memory, "max_iter": max_iter,
                  "generator": generator})
        if len(sigmas) == 1:
            return model
        err = force_rmse(model, validation)
        if err < best[0]:
            best = (err, {"sigma": s, "state": {k: v.clone() for k, v in model.state_dict().items()}})
    model.set_kernel(best[1]["sigma"])
    model.load_state_dict(best[1]["state"])
    return model


def force_rmse(model: SGDML, graphs: Sequence[AtomicGraph]) -> float:
    """Root-mean-square error of the predicted forces over ``graphs``."""
    total, count = 0.0, 0
    for g in graphs:
        pos, cell = model.geometry(g)
        pos = pos.detach().to(torch.float64).requires_grad_(True)
        e = model.energy_of_descriptors(model.descriptor(pos, cell))
        f = -torch.autograd.grad(e.sum(), pos)[0]
        ref = g.forces.reshape(f.shape).to(torch.float64)
        total += float(((f - ref) ** 2).sum())
        count += ref.numel()
    return math.sqrt(total / count)
