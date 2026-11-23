"""What the scriptable cores of exported tight-binding models share (see
:func:`~xnn.common.deploy.export_torchscript_potential`): Mulliken moments, Fermi occupations, Anderson mixing
and the self-consistent cycle of one molecule, with the charge potentials of a method supplied by
:meth:`TightBindingExport.potentials`.
"""
import math
from typing import List, Tuple

import torch
from torch import Tensor, nn


class TightBindingExport(nn.Module):
    """The self-consistent cycle of :class:`~xnn.sqmnn.models.tightbinding.TightBinding` for one molecule in
    TorchScript. Subclasses set ``multipoles``, ``kt``, ``threshold``, ``max_iter``, ``mixing``, ``history``
    and ``self_consistent`` and define :meth:`potentials`."""

    multipoles: bool
    kt: float
    threshold: float
    max_iter: int
    mixing: float
    history: int
    self_consistent: bool

    def potentials(self, q_shell: Tensor, mu: Tensor, theta: Tensor, shell_atom: Tensor,
                   ctx: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """Derivatives of the charge-dependent energy in the shell charges and atomic moments."""
        raise NotImplementedError

    def residual(self, r: Tensor) -> float:
        """The measure of the change of the mixed quantities that the convergence test compares with
        ``threshold`` (the root mean square)."""
        return float(torch.sqrt((r * r).mean()))

    def moments(self, P: Tensor, M: Tensor, n0: Tensor, ao_shell: Tensor, ao_atom: Tensor,
                n_atoms: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Mulliken shell charges, atomic dipoles and second moments."""
        q_shell = n0 - torch.zeros_like(n0).index_add(0, ao_shell, (P * M[0]).sum(-1))
        mu = torch.zeros((n_atoms, 3), dtype=P.dtype, device=P.device)
        theta = torch.zeros((n_atoms, 6), dtype=P.dtype, device=P.device)
        if self.multipoles:
            cols = (P[None] * M[1:]).sum(-2)                         # (9, nb): column sums
            mu = -mu.index_add(0, ao_atom, cols[:3].t())
            theta = -theta.index_add(0, ao_atom, cols[3:].t())
        return q_shell, mu, theta

    def fermi(self, eps: Tensor, n_alpha: float, n_beta: float) -> Tuple[Tensor, Tensor]:
        """Spin-summed Fermi occupations and ``T S_el``; the Fermi levels by Newton steps."""
        occ = torch.zeros_like(eps)
        ts = torch.zeros((), dtype=eps.dtype, device=eps.device)
        for nel in [n_alpha, n_beta]:
            if nel <= 0.0:
                continue
            if self.kt <= 0.0:
                occ = occ + (torch.arange(eps.shape[0], device=eps.device) < nel).to(eps.dtype)
                continue
            # Newton steps from between the highest occupied and the lowest empty level (exact at the middle of a
            # degenerate partly filled pair), bisecting where a step leaves the bracket, as the eager model
            n = eps.shape[0]
            top = min(max(int(math.ceil(nel)), 1), n)
            mid = 0.5 * (float(eps[top - 1]) + float(eps[min(top, n - 1)]))
            lo, hi = float(eps.min()) - 1.0, float(eps.max()) + 1.0
            target = 4.0 * 2.220446049250313e-16 * max(nel, 1.0)
            for _ in range(200):
                f = 0.5 * (1.0 - torch.tanh(0.5 * (eps - mid) / self.kt))
                res = float(f.sum()) - nel
                if abs(res) <= target:
                    break
                if res > 0.0:
                    hi = mid
                else:
                    lo = mid
                step = mid - res / max(float((f * (1.0 - f)).sum()) / self.kt, 1e-300)
                new = step if lo < step < hi else 0.5 * (lo + hi)
                if new == mid:
                    break
                mid = new
            f = torch.sigmoid(-(eps - mid) / self.kt)
            occ = occ + f
            ts = ts - self.kt * (torch.special.xlogy(f, f) + torch.special.xlogy(1.0 - f, 1.0 - f)).sum()
        return occ, ts

    def anderson(self, xs: List[Tensor], rs: List[Tensor], beta: float) -> Tensor:
        m = len(xs)
        if m == 1:
            return xs[0] + beta * rs[0]
        R = torch.stack(rs)
        A = R @ R.t()
        A = A + torch.eye(m, dtype=A.dtype, device=A.device) * 1e-14 * A.diagonal().max()
        w, info = torch.linalg.solve_ex(A, torch.ones(m, dtype=A.dtype, device=A.device))
        if int(info) != 0:
            return xs[m - 1] + beta * rs[m - 1]
        c = w / w.sum()
        return (c[:, None] * (torch.stack(xs) + beta * R)).sum(0)

    def scc(self, M: Tensor, H0: Tensor, ctx: List[Tensor], n0: Tensor, n_alpha: float, n_beta: float,
            shell_atom: Tensor, ao_shell: Tensor, ao_atom: Tensor, n_atoms: int,
            start: Tensor) -> Tuple[Tensor, Tensor, Tensor, bool]:
        """Self-consistent charges and moments as in the eager model: density, energy-weighted density,
        ``T S_el`` and whether the cycle converged."""
        n_shell = n0.shape[0]
        L = torch.linalg.cholesky(M[0])
        Lt = L.t()
        left = torch.linalg.solve_triangular(L, M, upper=False)
        h0 = torch.linalg.solve_triangular(Lt, torch.linalg.solve_triangular(L, H0, upper=False), upper=True,
                                           left=False)
        x = torch.zeros(n_shell + (9 * n_atoms if self.multipoles else 0), dtype=H0.dtype, device=H0.device)
        x[:n_shell] = start
        xs: List[Tensor] = []
        rs: List[Tensor] = []
        converged = False
        beta = self.mixing
        best_err, best_x, worse = float("inf"), x.clone(), 0
        checkpoint = float("inf")
        P, W = torch.zeros_like(H0), torch.zeros_like(H0)
        ts = torch.zeros((), dtype=H0.dtype, device=H0.device)
        for iteration in range(1, self.max_iter + 1):
            q_shell = x[:n_shell]
            mu = torch.zeros((n_atoms, 3), dtype=x.dtype, device=x.device)
            theta = torch.zeros((n_atoms, 6), dtype=x.dtype, device=x.device)
            if self.multipoles:
                mu = x[n_shell:n_shell + 3 * n_atoms].view(n_atoms, 3)
                theta = x[n_shell + 3 * n_atoms:].view(n_atoms, 6)
            v_shell, v_mu, v_theta = self.potentials(q_shell, mu, theta, shell_atom, ctx)
            V = v_shell[ao_shell][None, :]
            if self.multipoles:
                V = torch.cat([V, v_mu[ao_atom].t(), v_theta[ao_atom].t()], 0)
            K = torch.linalg.solve_triangular(Lt, (left * V[:, None, :]).sum(0), upper=True, left=False)
            eps, Co = torch.linalg.eigh(h0 - 0.5 * (K + K.t()))
            occ, ts = self.fermi(eps, n_alpha, n_beta)
            C = torch.linalg.solve_triangular(Lt, Co, upper=True)
            P = (C * occ[None, :]) @ C.t()
            W = (C * (occ * eps)[None, :]) @ C.t()
            qn, mun, thn = self.moments(P, M, n0, ao_shell, ao_atom, n_atoms)
            xn = torch.cat([qn, mun.reshape(-1), thn.reshape(-1)]) if self.multipoles else qn
            r = xn - x
            err = float(r.abs().max())
            if not self.self_consistent or self.residual(r) < self.threshold:
                converged = True
                break
            if err < best_err:
                best_err, best_x, worse = err, x.clone(), 0
            else:
                worse += 1
            stalled = iteration % 40 == 0 and best_err > 1e-2 * checkpoint
            if iteration % 40 == 0:
                checkpoint = best_err
            if (worse >= 8 and err > 2.0 * best_err) or stalled:
                beta, worse = 0.5 * beta, 0
                xs, rs = [], []
                x = best_x.clone()
                continue
            xs.append(x.clone())
            rs.append(r.clone())
            if len(xs) > self.history:
                xs, rs = xs[1:], rs[1:]
            x = self.anderson(xs, rs, beta)
        return P, W, ts, converged
