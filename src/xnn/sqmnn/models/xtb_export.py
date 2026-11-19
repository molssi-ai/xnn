"""The scriptable cores of exported GFN models (see :func:`~xnn.common.deploy.export_torchscript_potential`).

:class:`GFN0Export`, :class:`GFN1Export` and :class:`GFN2Export` hold the tables of every element of the
parameter set, made once from the model, and evaluate one molecule with the equations of
:mod:`~xnn.sqmnn.models.xtb`: the Gaussian integrals, the extended Hueckel matrix, the self-consistent charges
(and atomic multipoles) or, for GFN0-xTB, the charges of its charge model, the repulsion and the dispersion
terms. All atom pairs enter, so the exported model needs the whole molecule on one rank; periodic cells are
refused. Forces follow from automatic differentiation at the converged density.
"""
import copy
import math
from typing import List, Tuple

import torch
from torch import Tensor, nn

from xnn.common.models.d4 import charge_scaling_terms
from xnn.common.models.dispersion import three_body_energy

from ..common.xtb_tables import GFN0_SRB, GFN0_SRB_FACTOR, GFN2_AES_RADII
from .gaussian import _D_TRANSFORM, CARTESIAN, _cartesian_norm_ratio, _component_selectors
from .xtb import AUTOEV, BOHR, KB_AU, MAX_Z, _metal_pair_term, _origin_shift


class GFNExport(nn.Module):
    """What the two methods share: ``node_features_energy_charges`` of one molecule (energies in eV, the
    Mulliken charges), the interface of the exported potentials' charge-predicting cores."""

    bohr: float
    autoev: float
    multipoles: bool
    n_ops: int
    cutoff: float
    kt: float
    threshold: float
    max_iter: int
    mixing: float
    history: int
    eeq_guess: bool
    self_consistent: bool
    screening: float
    neglect: float
    budget: int
    enscale: float
    kexp: float
    kexplight: float
    k_values: List[int]

    def __init__(self, model):
        super().__init__()
        t = torch.float64
        device = model.table("LEV").device
        par, g = model.par, model.par.globals
        # the working precision the exported potential reads from the parameters (the tables are buffers)
        self.precision = nn.Parameter(torch.zeros(0, dtype=t, device=device), requires_grad=False)
        self.bohr, self.autoev = float(BOHR), float(AUTOEV)
        self.multipoles = bool(model.multipoles)
        self.n_ops = 10 if model.multipoles else 1
        self.cutoff = float(model.cutoff)
        self.kt = KB_AU * model.electronic_temperature
        self.threshold = model.charge_convergence * model.accuracy
        self.max_iter, self.mixing, self.history = int(model.max_iter), float(model.mixing), int(model.history)
        self.eeq_guess = model.guess == "eeq"
        self.self_consistent = True
        self.screening, self.neglect = float(model.integral_screening), float(model.integral_neglect)
        self.budget = int(model.integral_budget)
        self.enscale, self.kexp, self.kexplight = float(g.get("enscale", 0.0)), float(g["kexp"]), float(g["kexplight"])
        prims = {z: model.shell_primitives(z) for z in par.shells}
        kmax = max(len(a) for shells in prims.values() for a, _, _ in shells)
        self.k_values = sorted({len(a) for shells in prims.values() for a, _, _ in shells})
        nshell = torch.zeros(MAX_Z + 1, dtype=torch.long)
        sh_l = torch.zeros(MAX_Z + 1, 3, dtype=torch.long)
        sh_k = torch.ones(MAX_Z + 1, 3, dtype=torch.long)
        sh_alpha = torch.ones(MAX_Z + 1, 3, kmax, dtype=t)
        sh_coef = torch.zeros(MAX_Z + 1, 3, kmax, dtype=t)
        n0 = torch.zeros(MAX_Z + 1, 3, dtype=t)
        zeta = torch.ones(MAX_Z + 1, 3, dtype=t)
        for z, shells in prims.items():
            nshell[z] = len(shells)
            n0[z, :len(shells)] = torch.tensor(model.reference_occupations(z), dtype=t)
            for k, (alpha, coef, l) in enumerate(shells):
                sh_l[z, k], sh_k[z, k] = l, len(alpha)
                sh_alpha[z, k, :len(alpha)] = torch.tensor(alpha, dtype=t)
                sh_coef[z, k, :len(coef)] = torch.tensor(coef, dtype=t)
                zeta[z, k] = par.value(z, "EXP", k)
        norm = (2.0 * sh_alpha / math.pi) ** 0.75 * torch.stack(
            [(4.0 * sh_alpha[:, k]) ** (0.5 * sh_l[:, k, None].to(t)) for k in range(3)], 1)
        norm = norm / torch.sqrt(torch.tensor([1.0, 1.0, 3.0], dtype=t))[sh_l][:, :, None]
        tables = {"nshell": nshell, "sh_l": sh_l, "sh_k": sh_k, "sh_alpha": sh_alpha, "sh_weight": sh_coef * norm,
                  "n0": n0, "zeta": zeta, "lev": model.table("LEV").detach().cpu().clone(),
                  "kpoly": 0.01 * model.table("POLY").detach().cpu()[torch.arange(MAX_Z + 1)[:, None], sh_l],
                  "en": torch.tensor(model.en[:MAX_Z + 1], dtype=t),
                  "rcov": torch.tensor(model.rcov[:MAX_Z + 1], dtype=t),
                  "radii": torch.tensor(model.radii[:MAX_Z + 1], dtype=t),
                  "repa": model.table("REPA").detach().cpu().clone(),
                  "repb": model.table("REPB").detach().cpu().clone()}
        # selection of the one-dimensional moments of every (operator, bra component, ket component)
        extra = 2 if model.multipoles else 0
        selectors = torch.zeros(9, 3, (2 + extra + 1) * 3, self.n_ops * 36, dtype=t)
        for la in range(3):
            for lb in range(3):
                sel = _component_selectors(la, lb, self.n_ops, t, torch.device("cpu"))
                selectors[la * 3 + lb, :, :sel.shape[1], :sel.shape[2]] = sel
        # Cartesian components to the spherical functions (with the norms of the components)
        sph = torch.zeros(3, 5, 6, dtype=t)
        sph[0, 0, 0] = 1.0
        sph[1, :3, :3] = torch.eye(3, dtype=t)
        sph[2] = torch.tensor(_D_TRANSFORM, dtype=t) * torch.tensor(_cartesian_norm_ratio(2), dtype=t)
        first, second = _origin_shift(t, torch.device("cpu"))
        tables.update({"selectors": selectors, "sph": sph, "shift_first": first, "shift_second": second,
                       "n_cart": torch.tensor([len(CARTESIAN[l]) for l in range(3)])})
        for name, value in tables.items():
            self.register_buffer(name, value.to(device))
        self.eeq = copy.deepcopy(model._eeq)

    # what the methods define

    def begin(self, z: Tensor, x: Tensor, total_charge: float) -> List[Tensor]:
        """What a method prepares from the structure and its total charge before the Hamiltonian."""
        return [x[:0, 0]]

    def level_shift(self, sz: Tensor, sk: Tensor, shell_atom: Tensor, state: List[Tensor]) -> Tensor:
        """A shift of the shell levels (hartree) from the prepared state."""
        return torch.zeros(sz.shape[0], dtype=state[0].dtype, device=sz.device)

    def state_energy(self, z: Tensor, x: Tensor, state: List[Tensor]) -> Tensor:
        """Classical energy terms that need the prepared state."""
        return torch.zeros((), dtype=x.dtype, device=x.device)

    def reported_charges(self, charges: Tensor, state: List[Tensor]) -> Tensor:
        return charges

    def coordination(self, z: Tensor, x: Tensor) -> Tensor:
        raise NotImplementedError

    def levels(self, sz: Tensor, sk: Tensor, sl: Tensor, cn: Tensor) -> Tensor:
        raise NotImplementedError

    def pair_factor(self, sz: Tensor, sk: Tensor, sl: Tensor, shell_atom: Tensor, x: Tensor) -> Tensor:
        raise NotImplementedError

    def prepare(self, z: Tensor, x: Tensor, cn: Tensor, sz: Tensor, sk: Tensor, sl: Tensor,
                shell_atom: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def potentials(self, q_shell: Tensor, mu: Tensor, theta: Tensor, shell_atom: Tensor,
                   ctx: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError

    def charge_energy(self, q_shell: Tensor, mu: Tensor, theta: Tensor, shell_atom: Tensor, z: Tensor,
                      ctx: List[Tensor]) -> Tensor:
        raise NotImplementedError

    def classical(self, z: Tensor, x: Tensor) -> Tensor:
        raise NotImplementedError

    # shared pieces

    def pairs(self, n: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        tri = torch.triu_indices(n, n, 1, device=device)
        return tri[0], tri[1]

    def repulsion(self, z: Tensor, x: Tensor) -> Tensor:
        i, j = self.pairs(z.shape[0], x.device)
        alpha, zeff = self.repa[z], self.repb[z]
        light = z <= 2
        kexp = torch.where(light[i] & light[j], torch.full_like(alpha[i], self.kexplight),
                           torch.full_like(alpha[i], self.kexp))
        r = (x[j] - x[i]).norm(p=2, dim=-1)
        return (zeff[i] * zeff[j] / r * torch.exp(-torch.sqrt(alpha[i] * alpha[j]) * r ** kexp)).sum()

    def distance_polynomial(self, sz: Tensor, sl: Tensor, sk: Tensor, shell_atom: Tensor, x: Tensor) -> Tensor:
        p = x[shell_atom]
        diff = p[:, None, :] - p[None, :, :]
        same = shell_atom[:, None] == shell_atom[None, :]
        r = torch.sqrt((diff * diff).sum(-1) + same.to(x.dtype))
        rc = self.radii[sz]
        kpoly = self.kpoly[sz, sk]
        ratio = torch.sqrt(r / (rc[:, None] + rc[None, :]))
        return (1.0 + kpoly[:, None] * ratio) * (1.0 + kpoly[None, :] * ratio)

    def electronegativity_factor(self, sz: Tensor) -> Tensor:
        en = self.en[sz]
        return 1.0 + 0.01 * self.enscale * (en[:, None] - en[None, :]) ** 2

    def cartesian(self, ra: Tensor, rb: Tensor, aa: Tensor, wa: Tensor, ab: Tensor, wb: Tensor, la: int,
                  lb: int) -> Tensor:
        """Integrals of a batch of shell pairs over the Cartesian components, ``(P, ops, nca, ncb)``."""
        d = rb - ra
        r2 = (d * d).sum(-1)
        reduced = aa[:, :, None] * ab[:, None, :] / (aa[:, :, None] + ab[:, None, :])
        keep = (reduced * r2.detach()[:, None, None] <= self.neglect).nonzero()
        row, ka, kb = keep[:, 0], keep[:, 1], keep[:, 2]
        a, b = aa[row, ka], ab[row, kb]
        p = a + b
        pref = torch.exp(-a * b / p * r2[row]) * wa[row, ka] * wb[row, kb]
        dd = d[row].t()
        pa, pb = b / p * dd, -a / p * dd
        half = 0.5 / p
        mmax = la + (2 if self.multipoles else 0)
        # columns of the moments G[m][j] by the recursions of the integral module
        columns: List[List[Tensor]] = []
        column: List[Tensor] = [torch.sqrt(math.pi / p).expand(pa.shape)]
        for m in range(1, mmax + 1):
            value = pa * column[m - 1]
            if m > 1:
                value = value + (m - 1) * half * column[m - 2]
            column.append(value)
        columns.append(column)
        for j in range(1, lb + 1):
            column = []
            for m in range(mmax + 1):
                value = pb * columns[j - 1][m]
                if j > 1:
                    value = value + (j - 1) * half * columns[j - 2][m]
                if m > 0:
                    value = value + m * half * columns[j - 1][m - 1]
                column.append(value)
            columns.append(column)
        flat: List[Tensor] = []
        for m in range(mmax + 1):
            for j in range(lb + 1):
                flat.append(columns[j][m])
        table = torch.stack(flat, -1)
        nca, ncb = int(self.n_cart[la]), int(self.n_cart[lb])
        ncomb = self.n_ops * nca * ncb
        sel = self.selectors[la * 3 + lb, :, :(mmax + 1) * (lb + 1), :ncomb]
        prim = ((table[0] * pref[:, None]) @ sel[0]) * (table[1] @ sel[1]) * (table[2] @ sel[2])
        vals = torch.zeros((ra.shape[0], ncomb), dtype=prim.dtype, device=prim.device).index_add(0, row, prim)
        return vals.view(-1, self.n_ops, nca, ncb)

    def integrals(self, x: Tensor, shell_atom: Tensor, sl: Tensor, kcount: Tensor, alpha: Tensor, weight: Tensor,
                  shell_first: Tensor, nb: int) -> Tensor:
        """The stacked dense matrices ``(n_ops, nb, nb)``: overlap, then with multipoles the dipole and
        second-moment matrices with the origin on the centre of the column's orbital."""
        ns = sl.shape[0]
        tri = torch.triu_indices(ns, ns, 0, device=x.device)
        ti, tj = tri[0], tri[1]
        amin = alpha.masked_fill(weight == 0, float("inf")).min(-1).values
        dx = (x[shell_atom[ti]] - x[shell_atom[tj]]).detach()
        near = amin[ti] * amin[tj] / (amin[ti] + amin[tj]) * (dx * dx).sum(-1) <= self.screening
        ti, tj = ti[near], tj[near]
        index: List[Tensor] = []
        values: List[Tensor] = []
        ops = torch.arange(self.n_ops, device=x.device)[None, :, None, None]
        for la in range(3):
            for lb in range(3):
                na, nbf = 2 * la + 1, 2 * lb + 1
                nca, ncb = int(self.n_cart[la]), int(self.n_cart[lb])
                Ta, Tb = self.sph[la, :na, :nca], self.sph[lb, :nbf, :ncb]
                for ka in self.k_values:
                    for kb in self.k_values:
                        sel = ((sl[ti] == la) & (sl[tj] == lb) & (kcount[ti] == ka) & (kcount[tj] == kb)).nonzero()
                        sel = sel.squeeze(1)
                        chunk = max(64, self.budget // (ka * kb * self.n_ops * nca * ncb))
                        for start in range(0, sel.shape[0], chunk):
                            part = sel[start:start + chunk]
                            ii, jj = ti[part], tj[part]
                            ra, rb = x[shell_atom[ii]], x[shell_atom[jj]]
                            cart = self.cartesian(ra, rb, alpha[ii, :ka], weight[ii, :ka], alpha[jj, :kb],
                                                  weight[jj, :kb], la, lb)
                            on_a = torch.matmul(torch.matmul(Ta, cart), Tb.t())
                            on_b = on_a
                            if self.multipoles:
                                dvec = ra - rb
                                both = torch.stack([dvec[:, 0] * dvec[:, 0], dvec[:, 1] * dvec[:, 1],
                                                    dvec[:, 2] * dvec[:, 2], dvec[:, 0] * dvec[:, 1],
                                                    dvec[:, 0] * dvec[:, 2], dvec[:, 1] * dvec[:, 2]], 1)
                                shift = (torch.eye(10, dtype=x.dtype, device=x.device).view(100)
                                         + dvec @ self.shift_first + both @ self.shift_second)
                                on_b = torch.bmm(shift.view(-1, 10, 10), on_a.flatten(2)).view(on_a.shape)
                            ia = (shell_first[ii][:, None] + torch.arange(na, device=x.device))[:, None, :, None]
                            ib = (shell_first[jj][:, None] + torch.arange(nbf, device=x.device))[:, None, None, :]
                            index.append(((ops * nb + ia) * nb + ib).reshape(-1))
                            values.append(on_b.reshape(-1))
                            off = ii != jj
                            index.append(((ops * nb + ib) * nb + ia).expand(on_a.shape)[off].reshape(-1))
                            values.append(on_a[off].reshape(-1))
        M = torch.zeros(self.n_ops * nb * nb, dtype=x.dtype, device=x.device)
        return M.index_add(0, torch.cat(index), torch.cat(values)).view(self.n_ops, nb, nb)

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
        """Spin-summed Fermi occupations and ``T S_el``; the Fermi levels by bisection."""
        occ = torch.zeros_like(eps)
        ts = torch.zeros((), dtype=eps.dtype, device=eps.device)
        for nel in [n_alpha, n_beta]:
            if nel <= 0.0:
                continue
            if self.kt <= 0.0:
                occ = occ + (torch.arange(eps.shape[0], device=eps.device) < nel).to(eps.dtype)
                continue
            lo, hi = float(eps.min()) - 1.0, float(eps.max()) + 1.0
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if mid == lo or mid == hi:
                    break
                if float(torch.sigmoid(-(eps - mid) / self.kt).sum()) > nel:
                    hi = mid
                else:
                    lo = mid
            f = torch.sigmoid(-(eps - 0.5 * (lo + hi)) / self.kt)
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
            if not self.self_consistent or float(torch.sqrt((r * r).mean())) < self.threshold:
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

    def node_features_energy_charges(self, atomic_numbers: Tensor, edge_index: Tensor, edge_vec: Tensor,
                                     pos: Tensor, cell: Tensor, pbc: Tensor, total_charge: float,
                                     spin_multiplicity: float) -> Tuple[Tensor, Tensor, Tensor]:
        """No features ``(N, 0)``, the energies per atom (eV) and the Mulliken charges of one molecule; the
        neighbor list is not used (every atom pair enters)."""
        if bool(pbc.any()) and bool((cell.abs() > 0).any()):
            raise ValueError("a GFN model evaluates molecules: periodic cells are not supported")
        device = pos.device
        z = atomic_numbers.long()
        n = z.shape[0]
        x = pos.to(torch.float64) / self.bohr
        counts = self.nshell[z]
        if bool((counts == 0).any()):
            raise ValueError("the parameter set has no parameters for an element of the structure")
        shell_atom = torch.repeat_interleave(torch.arange(n, device=device), counts)
        sk = torch.arange(shell_atom.shape[0], device=device) - (torch.cumsum(counts, 0) - counts)[shell_atom]
        sz = z[shell_atom]
        sl = self.sh_l[sz, sk]
        width = 2 * sl + 1
        shell_first = torch.cumsum(width, 0) - width
        nb = int(width.sum())
        ao_shell = torch.repeat_interleave(torch.arange(sl.shape[0], device=device), width)
        ao_atom = shell_atom[ao_shell]
        M = self.integrals(x, shell_atom, sl, self.sh_k[sz, sk], self.sh_alpha[sz, sk], self.sh_weight[sz, sk],
                           shell_first, nb)
        S = M[0]
        state = self.begin(z, x, total_charge)
        cn = self.coordination(z, x)
        hb = (self.levels(sz, sk, sl, cn[shell_atom]) + self.level_shift(sz, sk, shell_atom, state))[ao_shell]
        factor = self.pair_factor(sz, sk, sl, shell_atom, x)[ao_shell][:, ao_shell]
        off = 0.5 * (hb[:, None] + hb[None, :]) * S * factor
        same = ao_atom[:, None] == ao_atom[None, :]
        H0 = torch.where(same, torch.zeros_like(off), off) + torch.diag(hb * S.diagonal())
        ctx = self.prepare(z, x, cn, sz, sk, sl, shell_atom)
        n0 = self.n0[sz, sk]
        electrons = int(round(float(n0.sum()) - total_charge))
        multiplicity = int(spin_multiplicity)
        unpaired = multiplicity - 1 if multiplicity > 0 else electrons % 2
        n_alpha, n_beta = 0.5 * (electrons + unpaired), 0.5 * (electrons - unpaired)
        with torch.no_grad():
            Md, Hd = M.detach(), H0.detach()
            ctx_d = [c.detach() for c in ctx]
            neutral = torch.zeros_like(n0)
            start = neutral
            if self.eeq_guess:
                i, j = self.pairs(n, device)
                ei = torch.cat([torch.stack([i, j]), torch.stack([j, i])], 1)
                xd = x.detach()
                dij = (xd[j] - xd[i]).norm(p=2, dim=-1)
                cn_eeq = self.eeq.coordination_numbers(z, ei, torch.cat([dij, dij]), n)[1]
                q = self.eeq.eeq_charges(z, xd, cn_eeq, torch.tensor(total_charge, dtype=x.dtype, device=device),
                                         torch.zeros((3, 3), dtype=x.dtype, device=device), False)
                total = torch.zeros_like(q).index_add(0, shell_atom, n0)
                start = q[shell_atom] * n0 / total[shell_atom]
            P, W, ts, converged = self.scc(Md, Hd, ctx_d, n0, n_alpha, n_beta, shell_atom, ao_shell, ao_atom, n, start)
            if self.eeq_guess and not converged:
                P2, W2, ts2, again = self.scc(Md, Hd, ctx_d, n0, n_alpha, n_beta, shell_atom, ao_shell, ao_atom, n,
                                              neutral)
                if again:
                    P, W, ts = P2, W2, ts2
        q_shell, mu, theta = self.moments(P, M, n0, ao_shell, ao_atom, n)
        energy = ((P * H0).sum() + self.charge_energy(q_shell, mu, theta, shell_atom, z, ctx) + self.classical(z, x)
                  + self.state_energy(z, x, state)
                  - ts - (W * (S - S.detach())).sum())
        charges = torch.zeros(n, dtype=x.dtype, device=device).index_add(0, shell_atom, q_shell)
        charges = self.reported_charges(charges, state)
        node_energy = (energy * self.autoev / n).expand(n)
        return torch.zeros((n, 0), dtype=x.dtype, device=device), node_energy, charges.detach()


class GFN2Export(GFNExport):
    """The scriptable core of GFN2-xTB."""

    aesrmax: float
    aesexp: float
    aesshift: float
    aesdmp3: float
    aesdmp5: float

    def __init__(self, model):
        super().__init__(model)
        t = torch.float64
        device = self.lev.device
        g = model.par.globals
        self.aesrmax, self.aesexp, self.aesshift = float(g["aesrmax"]), float(g["aesexp"]), float(g["aesshift"])
        self.aesdmp3, self.aesdmp5 = float(g["aesdmp3"]), float(g["aesdmp5"])
        kl = (g["ks"], g["kp"], g["kd"])
        K = torch.tensor([[kl[a] if a == b else 0.5 * (kl[a] + kl[b]) for b in range(3)] for a in range(3)], dtype=t)
        K[0, 2] = K[2, 0] = g["ksd"]
        K[1, 2] = K[2, 1] = g["kpd"]
        l_of = self.sh_l.cpu()
        rows = torch.arange(MAX_Z + 1)[:, None]
        lpar = model.table("LPAR").detach().cpu()[rows, l_of]
        kappa = torch.where(l_of > 0, 0.1 * lpar, torch.zeros_like(lpar))
        k3 = torch.tensor([g["gam3s"], g["gam3p"], g["gam3d1"]], dtype=t)
        tables = {"k_shell": K, "kcn": 0.1 * model.table("KCN").detach().cpu()[rows, l_of],
                  "eta": model.table("GAM").detach().cpu()[:, None] * (1.0 + kappa),
                  "gam3": 0.1 * model.table("GAM3").detach().cpu()[:, None] * k3[l_of],
                  "dpol": 0.01 * model.table("DPOL").detach().cpu(), "qpol": 0.01 * model.table("QPOL").detach().cpu(),
                  "aes_r0": torch.tensor([GFN2_AES_RADII.get(z, (g["aesrmax"], 0.0))[0] for z in range(MAX_Z + 1)],
                                         dtype=t),
                  "aes_nval": torch.tensor([GFN2_AES_RADII.get(z, (0.0, 0.0))[1] for z in range(MAX_Z + 1)], dtype=t)}
        for name, value in tables.items():
            self.register_buffer(name, value.to(device))
        self.d4 = copy.deepcopy(model.d4)

    def coordination(self, z: Tensor, x: Tensor) -> Tensor:
        i, j = self.pairs(z.shape[0], x.device)
        rc = self.rcov[z]
        r = (x[i] - x[j]).norm(p=2, dim=-1)
        r0 = 4.0 / 3.0 * (rc[i] + rc[j])
        f = torch.sigmoid(10.0 * (r0 / r - 1.0)) * torch.sigmoid(20.0 * ((r0 + 2.0) / r - 1.0))
        return torch.zeros(z.shape[0], dtype=x.dtype, device=x.device).index_add(0, i, f).index_add(0, j, f)

    def levels(self, sz: Tensor, sk: Tensor, sl: Tensor, cn: Tensor) -> Tensor:
        return (self.lev[sz, sk] - self.kcn[sz, sk] * cn) / self.autoev

    def pair_factor(self, sz: Tensor, sk: Tensor, sl: Tensor, shell_atom: Tensor, x: Tensor) -> Tensor:
        zeta = self.zeta[sz, sk]
        Y = torch.sqrt(2.0 * torch.sqrt(zeta[:, None] * zeta[None, :]) / (zeta[:, None] + zeta[None, :]))
        return (self.k_shell[sl[:, None], sl[None, :]] * self.electronegativity_factor(sz) * Y
                * self.distance_polynomial(sz, sl, sk, shell_atom, x))

    def prepare(self, z: Tensor, x: Tensor, cn: Tensor, sz: Tensor, sk: Tensor, sl: Tensor,
                shell_atom: Tensor) -> List[Tensor]:
        """Charge-independent tables of the cycle: the shell Coulomb matrix, the third-order factors, the ten
        matrices of the anisotropic electrostatics, the multipole kernels' factors and the D4 tables."""
        n = z.shape[0]
        d4 = self.d4
        eta = self.eta[sz, sk]
        ps = x[shell_atom]
        d = ps[:, None, :] - ps[None, :, :]
        gamma = 1.0 / torch.sqrt((d * d).sum(-1) + (0.5 * (eta[:, None] + eta[None, :])) ** -2)
        r0 = self.aes_r0[z]
        rad = r0 + (self.aesrmax - r0) * torch.sigmoid(self.aesexp * (cn - self.aes_nval[z] - self.aesshift))
        i, j = self.pairs(n, x.device)
        R = x[j] - x[i]
        r = R.norm(p=2, dim=-1)
        rab = 0.5 * (rad[i] + rad[j])
        f3 = 1.0 / (r ** 3 * (1.0 + 6.0 * (rab / r) ** self.aesdmp3))
        f5 = 1.0 / (r ** 5 * (1.0 + 6.0 * (rab / r) ** self.aesdmp5))
        rows = torch.stack([f5 * (R * R).sum(-1), f3 * R[:, 0], f3 * R[:, 1], f3 * R[:, 2], f5 * R[:, 0] * R[:, 0],
                            f5 * R[:, 1] * R[:, 1], f5 * R[:, 2] * R[:, 2], f5 * R[:, 0] * R[:, 1],
                            f5 * R[:, 0] * R[:, 2], f5 * R[:, 1] * R[:, 2]])
        sign = torch.ones((10, 1), dtype=x.dtype, device=x.device)
        sign[1:4] = -1.0
        T = torch.zeros((10, n, n), dtype=x.dtype, device=x.device)
        T[:, i, j] = rows
        T[:, j, i] = rows * sign
        ei = torch.cat([torch.stack([i, j]), torch.stack([j, i])], 1)
        rr = torch.cat([r, r])
        cn_d4 = d4.coordination_numbers(z, ei, rr, n)[0]
        inside = rr < d4.cutoff_pair / d4.bohr
        damping = torch.zeros((n, n), dtype=x.dtype, device=x.device).index_put(
            (ei[1], ei[0]), torch.where(inside, d4.pair_damping(z, ei, rr), torch.zeros_like(rr)))
        zeff = d4.zeff[z][:, None]
        return [gamma, self.gam3[sz, sk], T, f3, f5, R, damping, d4.gaussian_weights(z, cn_d4),
                d4.refalpha[:, :, z].permute(2, 1, 0), zeff, (d4.hardness[z] * d4.gc)[:, None],
                d4.refq[:, z].t() + zeff, self.dpol[z], self.qpol[z]]

    def potentials(self, q_shell: Tensor, mu: Tensor, theta: Tensor, shell_atom: Tensor,
                   ctx: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        gamma, gam3, T, damping, w, refalpha = ctx[0], ctx[1], ctx[2], ctx[6], ctx[7], ctx[8]
        n = mu.shape[0]
        q = torch.zeros(n, dtype=q_shell.dtype, device=q_shell.device).index_add(0, shell_atom, q_shell)
        v_shell = gamma @ q_shell + q_shell * q_shell * gam3
        tr = theta[:, :3].sum(-1, keepdim=True)
        th = torch.cat([1.5 * theta[:, :3] - 0.5 * tr, 1.5 * theta[:, 3:]], 1)
        out = torch.matmul(T, torch.cat([q[:, None], mu, th[:, :3], 2.0 * th[:, 3:]], 1))
        v_q = out[4:, :, 4:].diagonal(dim1=0, dim2=2).sum(-1) - out[1:4, :, 1:4].diagonal(dim1=0, dim2=2).sum(-1)
        t2mu = torch.stack([out[4, :, 1], out[7, :, 2], out[8, :, 3], out[7, :, 1], out[5, :, 2], out[9, :, 3],
                            out[8, :, 1], out[9, :, 2], out[6, :, 3]], -1).view(n, 3, 3)
        dpol, qpol = ctx[12], ctx[13]
        v_mu = out[1:4, :, 0].t() + out[0, :, 1:4] - 3.0 * t2mu.sum(-1) + 2.0 * dpol[:, None] * mu
        G = out[4:, :, 0].t() + 2.0 * qpol[:, None] * th
        v_theta = torch.cat([1.5 * G[:, :3] - 0.5 * G[:, :3].sum(-1, keepdim=True), 3.0 * G[:, 3:]], 1)
        zeta, slope = charge_scaling_terms(q, ctx[9], ctx[10], ctx[11], self.d4.ga)
        both = torch.bmm(torch.stack([w * zeta, w * slope], 1), refalpha)
        v_d4 = -(3.0 / math.pi) * (self.d4.cp_weights * both[:, 1] * (damping @ both[:, 0])).sum(-1)
        return v_shell + (v_q + v_d4)[shell_atom], v_mu, v_theta

    def charge_energy(self, q_shell: Tensor, mu: Tensor, theta: Tensor, shell_atom: Tensor, z: Tensor,
                      ctx: List[Tensor]) -> Tensor:
        gamma, gam3, f3, f5, R, damping, w, refalpha = ctx[0], ctx[1], ctx[3], ctx[4], ctx[5], ctx[6], ctx[7], ctx[8]
        n = z.shape[0]
        q = torch.zeros(n, dtype=q_shell.dtype, device=q_shell.device).index_add(0, shell_atom, q_shell)
        e = (0.5 * q_shell * (gamma @ q_shell)).sum() + (q_shell ** 3 * gam3 / 3.0).sum()
        i, j = self.pairs(n, q.device)
        tr = theta[:, :3].sum(-1)
        Th = 1.5 * torch.stack([theta[:, 0], theta[:, 3], theta[:, 4], theta[:, 3], theta[:, 1], theta[:, 5],
                                theta[:, 4], theta[:, 5], theta[:, 2]], -1).view(n, 3, 3)
        Th = Th - 0.5 * tr[:, None, None] * torch.eye(3, dtype=q.dtype, device=q.device)
        r2 = (R * R).sum(-1)
        mi_r, mj_r = (mu[i] * R).sum(-1), (mu[j] * R).sum(-1)
        rtr_i = ((Th[i] @ R[:, :, None]).squeeze(-1) * R).sum(-1)
        rtr_j = ((Th[j] @ R[:, :, None]).squeeze(-1) * R).sum(-1)
        e = e + (f3 * (q[j] * mi_r - q[i] * mj_r)
                 + f5 * (q[i] * rtr_j + q[j] * rtr_i + (mu[i] * mu[j]).sum(-1) * r2 - 3.0 * mi_r * mj_r)).sum()
        e = e + (self.dpol[z] * (mu * mu).sum(-1) + self.qpol[z] * (Th * Th).sum((-1, -2))).sum()
        alpha = ((w * self.d4.charge_scaling(z, q))[:, :, None] * refalpha).sum(1)
        return e - 0.5 * (3.0 / math.pi) * (alpha * self.d4.cp_weights * (damping @ alpha)).sum()

    def classical(self, z: Tensor, x: Tensor) -> Tensor:
        d4 = self.d4
        n = z.shape[0]
        i, j = self.pairs(n, x.device)
        ei = torch.cat([torch.stack([i, j]), torch.stack([j, i])], 1)
        dij = x[j] - x[i]
        vec = torch.cat([dij, -dij])
        rr = vec.norm(p=2, dim=-1)
        cn_d4 = d4.coordination_numbers(z, ei, rr, n)[0]
        a0 = d4.dynamic_polarizabilities(z, d4.reference_weights(z, cn_d4, torch.zeros_like(cn_d4)))
        c6 = (3.0 / math.pi) * (a0 * d4.cp_weights) @ a0.t()
        keep = (rr <= d4.cutoff_triple / d4.bohr).nonzero().squeeze(1)
        e3 = three_body_energy(z, ei[:, keep], vec[keep], rr[keep], c6, d4.pair_radius_table(), d4.s9, d4.alp / 3.0,
                               d4.cutoff_triple / d4.bohr, d4.switch_width_triple / d4.bohr, n)
        return self.repulsion(z, x) + e3.sum()


class GFN1Export(GFNExport):
    """The scriptable core of GFN1-xTB."""

    cn_cutoff: float
    kdiff: float
    xbrad: float
    xbdamp: float

    def __init__(self, model):
        super().__init__(model)
        t = torch.float64
        device = self.lev.device
        par, g = model.par, model.par.globals
        self.cn_cutoff, self.kdiff = float(model.cn_cutoff), float(g["kdiff"])
        self.xbrad, self.xbdamp = float(g["xbrad"]), float(g["xbdamp"])
        K = torch.tensor([[g["ks"], g["ksp"], 0.5 * (g["ks"] + g["kd"])],
                          [g["ksp"], g["kp"], 0.5 * (g["kp"] + g["kd"])],
                          [0.5 * (g["ks"] + g["kd"]), 0.5 * (g["kp"] + g["kd"]), g["kd"]]], dtype=t)
        l_of = self.sh_l.cpu()
        rows = torch.arange(MAX_Z + 1)[:, None]
        lpar = model.table("LPAR").detach().cpu()[rows, l_of]
        kappa = torch.where(l_of > 0, model.kappa_scale * lpar, torch.zeros_like(lpar))
        kcn = torch.zeros(MAX_Z + 1, 3, dtype=t)
        metal = torch.zeros(MAX_Z + 1, dtype=t)
        cxb = torch.zeros(MAX_Z + 1, dtype=t)
        acceptor = torch.zeros(MAX_Z + 1, dtype=torch.bool)
        for z, shells in par.shells.items():
            kcn[z, :len(shells)] = torch.tensor(model._kcn(z), dtype=t)
            metal[z] = _metal_pair_term(z) if shells[0][1] == 2 else 0.0
            cxb[z] = par.value(z, "CXB")
            acceptor[z] = z in model.xb_acceptors
        diffuse = torch.zeros(MAX_Z + 1, 3, dtype=torch.bool)
        diffuse[1, 1] = True
        tables = {"k_shell": K, "k_level": torch.tensor([g["ks"], g["kp"], g["kd"]], dtype=t), "kcn": kcn,
                  "eta": model.table("GAM").detach().cpu()[:, None] * (1.0 + kappa),
                  "gam3": model.gam3_scale * model.table("GAM3").detach().cpu(), "metal": metal, "cxb": cxb,
                  "acceptor": acceptor, "diffuse": diffuse, "pair": model._pair_table(t, torch.device("cpu"))}
        for name, value in tables.items():
            self.register_buffer(name, value.to(device))
        self.d3 = copy.deepcopy(model.d3)

    def coordination(self, z: Tensor, x: Tensor) -> Tensor:
        i, j = self.pairs(z.shape[0], x.device)
        rc = self.rcov[z]
        r = (x[i] - x[j]).norm(p=2, dim=-1)
        f = torch.sigmoid(16.0 * (4.0 / 3.0 * (rc[i] + rc[j]) / r - 1.0))
        f = torch.where(r <= self.cn_cutoff, f, torch.zeros_like(f))
        return torch.zeros(z.shape[0], dtype=x.dtype, device=x.device).index_add(0, i, f).index_add(0, j, f)

    def levels(self, sz: Tensor, sk: Tensor, sl: Tensor, cn: Tensor) -> Tensor:
        return self.lev[sz, sk] * (1.0 + self.kcn[sz, sk] * cn) / self.autoev

    def pair_factor(self, sz: Tensor, sk: Tensor, sl: Tensor, shell_atom: Tensor, x: Tensor) -> Tensor:
        K = self.k_shell[sl[:, None], sl[None, :]] * self.pair[sz[:, None], sz[None, :]]
        metal = self.metal[sz]
        both = (metal[:, None] > 0) & (metal[None, :] > 0)
        K = torch.where(both, K * (1.0 + 0.5 * (metal[:, None] + metal[None, :])), K)
        diffuse = self.diffuse[sz, sk]
        kl = self.k_level[sl]
        kd = torch.where(diffuse, torch.full_like(kl, self.kdiff), kl)
        either = diffuse[:, None] | diffuse[None, :]
        K = torch.where(either, 0.5 * (kd[:, None] + kd[None, :]), K)
        X = self.electronegativity_factor(sz)
        X = torch.where(either, torch.ones_like(X), X)
        return K * X * self.distance_polynomial(sz, sl, sk, shell_atom, x)

    def prepare(self, z: Tensor, x: Tensor, cn: Tensor, sz: Tensor, sk: Tensor, sl: Tensor,
                shell_atom: Tensor) -> List[Tensor]:
        eta = self.eta[sz, sk]
        ps = x[shell_atom]
        d = ps[:, None, :] - ps[None, :, :]
        gamma = 1.0 / torch.sqrt((d * d).sum(-1) + (2.0 / (1.0 / eta[:, None] + 1.0 / eta[None, :])) ** -2)
        return [gamma, self.gam3[z]]

    def potentials(self, q_shell: Tensor, mu: Tensor, theta: Tensor, shell_atom: Tensor,
                   ctx: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        q = torch.zeros(mu.shape[0], dtype=q_shell.dtype, device=q_shell.device).index_add(0, shell_atom, q_shell)
        return ctx[0] @ q_shell + (q * q * ctx[1])[shell_atom], torch.zeros_like(mu), torch.zeros_like(theta)

    def charge_energy(self, q_shell: Tensor, mu: Tensor, theta: Tensor, shell_atom: Tensor, z: Tensor,
                      ctx: List[Tensor]) -> Tensor:
        q = torch.zeros(z.shape[0], dtype=q_shell.dtype, device=q_shell.device).index_add(0, shell_atom, q_shell)
        return (0.5 * q_shell * (ctx[0] @ q_shell)).sum() + (q ** 3 * ctx[1] / 3.0).sum()

    def halogen_bond(self, z: Tensor, x: Tensor) -> Tensor:
        """The halogen-bond correction of the eager model, one halogen at a time."""
        out = torch.zeros((), dtype=x.dtype, device=x.device)
        n = z.shape[0]
        atoms = torch.arange(n, device=x.device)
        for h in (self.cxb[z] != 0).nonzero().squeeze(1):
            others = atoms[atoms != h]
            acceptors = others[self.acceptor[z[others]]]
            if others.shape[0] == 0 or acceptors.shape[0] == 0:
                continue
            nearest = others[torch.argmin((x[others] - x[h]).norm(p=2, dim=-1).detach())]
            rxa = x[acceptors] - x[h]
            rxb = x[nearest] - x[h]
            dxa = rxa.norm(p=2, dim=-1)
            cos = (rxa * rxb).sum(-1) / (dxa * rxb.norm(p=2))
            x6 = (self.xbrad * (self.radii[z[acceptors]] + self.radii[z[h]]) / dxa) ** 6
            e = 0.1 * self.cxb[z[h]] * (0.5 * (1.0 - cos)) ** 6 * (x6 * x6 - self.xbdamp * x6) / (x6 * x6 + 1.0)
            out = out + torch.where(acceptors == nearest, torch.zeros_like(e), e).sum()
        return out

    def classical(self, z: Tensor, x: Tensor) -> Tensor:
        d3 = self.d3
        n = z.shape[0]
        i, j = self.pairs(n, x.device)
        ei = torch.cat([torch.stack([i, j]), torch.stack([j, i])], 1)
        dij = x[j] - x[i]
        batch = torch.zeros(n, dtype=torch.long, device=x.device)
        e3 = d3.evaluate(z, x * self.bohr, ei, torch.cat([dij, -dij]) * self.bohr, batch, 1,
                         torch.zeros((1, 3, 3), dtype=x.dtype, device=x.device),
                         torch.zeros((1, 3), dtype=torch.bool, device=x.device),
                         torch.zeros(1, dtype=x.dtype, device=x.device))["node_energy"] / d3.hartree
        return self.repulsion(z, x) + e3.sum() + self.halogen_bond(z, x)


class GFN0Export(GFNExport):
    """The scriptable core of GFN0-xTB: the charges of its charge model, one diagonalization."""

    kdiff: float
    enscale4: float
    renscale: float
    srbshift: float
    srbpre: float
    srbexp: float
    srbken: float
    srb_k1: float
    srb_k2: float
    metal_pair: float
    coinage_pair: float

    def __init__(self, model):
        super().__init__(model)
        t = torch.float64
        device = self.lev.device
        par, g = model.par, model.par.globals
        self.self_consistent = False
        self.eeq_guess = False
        self.kdiff, self.enscale4, self.renscale = float(g["kdiff"]), float(g["enscale4"]), float(g["renscale"])
        self.srbshift, self.srbpre = float(g["srbshift"]), float(g["srbpre"])
        self.srbexp, self.srbken = float(g["srbexp"]), float(g["srbken"])
        self.srb_k1, self.srb_k2 = float(GFN0_SRB_FACTOR[0]), float(GFN0_SRB_FACTOR[1])
        self.metal_pair, self.coinage_pair = float(model.metal_pair), float(model.coinage_pair)
        l_of = self.sh_l.cpu()
        rows = torch.arange(MAX_Z + 1)[:, None]
        metal = torch.zeros(MAX_Z + 1, dtype=torch.bool)
        for z, shells in par.shells.items():
            metal[z] = shells[0][1] == 2
        coinage = torch.zeros(MAX_Z + 1, dtype=torch.bool)
        coinage[torch.tensor([29, 47, 79])] = True
        diffuse = torch.zeros(MAX_Z + 1, 3, dtype=torch.bool)
        diffuse[1, 1] = True
        srb = torch.zeros(MAX_Z + 1, 3, dtype=t)
        for z, values in GFN0_SRB.items():
            srb[z] = torch.tensor(values, dtype=t)
        tables = {"kcn": 0.1 * model.table("KCN").detach().cpu()[rows, l_of],
                  "kq": model.table("KQ").detach().cpu()[rows, l_of],
                  "kqat2": model.table("KQAT2").detach().cpu().clone(),
                  "k_level": torch.tensor([g["ks"], g["kp"], g["kd"]], dtype=t),
                  "ken": 0.01 * torch.tensor([g["ens"], g["enp"], g["end"]], dtype=t),
                  "en_method": torch.tensor(model.en_method[:MAX_Z + 1], dtype=t), "metal": metal, "coinage": coinage,
                  "diffuse": diffuse, "srb": srb}
        for name, value in tables.items():
            self.register_buffer(name, value.to(device))
        self.charge_model = copy.deepcopy(model.charge_model)
        self.d4 = copy.deepcopy(model.d4)

    def charge_cn(self, z: Tensor, x: Tensor) -> Tensor:
        """Coordination numbers of the charge model."""
        n = z.shape[0]
        i, j = self.pairs(n, x.device)
        r = (x[j] - x[i]).norm(p=2, dim=-1)
        ei = torch.cat([torch.stack([i, j]), torch.stack([j, i])], 1)
        return self.charge_model.coordination_numbers(z, ei, torch.cat([r, r]), n)[1]

    def begin(self, z: Tensor, x: Tensor, total_charge: float) -> List[Tensor]:
        """Charges, coordination numbers and electrostatic energy of the charge model, and the total charge."""
        model = self.charge_model
        cn = self.charge_cn(z, x)
        charge = torch.tensor([total_charge], dtype=x.dtype, device=x.device)
        q = model.eeq_charges(z, x, cn, charge[0], torch.zeros((3, 3), dtype=x.dtype, device=x.device), False)
        rhs = -model.eeq_chi[z] + model.eeq_kcnchi[z] * cn / torch.sqrt(cn + model.eeq_cn_reg)
        es = q @ (0.5 * model._coulomb_matrix_molecular(z, x) @ q - rhs)
        return [q, cn, es.reshape(1), charge]

    def coordination(self, z: Tensor, x: Tensor) -> Tensor:
        return self.charge_cn(z, x)

    def levels(self, sz: Tensor, sk: Tensor, sl: Tensor, cn: Tensor) -> Tensor:
        return (self.lev[sz, sk] - self.kcn[sz, sk] * cn) / self.autoev

    def level_shift(self, sz: Tensor, sk: Tensor, shell_atom: Tensor, state: List[Tensor]) -> Tensor:
        q = state[0][shell_atom]
        return -(self.kq[sz, sk] * q + self.kqat2[sz] * q * q) / self.autoev

    def pair_factor(self, sz: Tensor, sk: Tensor, sl: Tensor, shell_atom: Tensor, x: Tensor) -> Tensor:
        kl = self.k_level[sl]
        K = 0.5 * (kl[:, None] + kl[None, :])
        diffuse = self.diffuse[sz, sk]
        K = torch.where(diffuse[:, None] | diffuse[None, :], K * self.kdiff, K)
        K = torch.where(diffuse[:, None] & diffuse[None, :], torch.zeros_like(K), K)
        metal, coinage = self.metal[sz], self.coinage[sz]
        K = torch.where(metal[:, None] & metal[None, :], K * self.metal_pair, K)
        K = torch.where(coinage[:, None] & coinage[None, :], K * (self.coinage_pair / self.metal_pair), K)
        ken = self.ken[sl]
        ken = 0.5 * (ken[:, None] + ken[None, :])
        en = self.en_method[sz]
        d2 = (en[:, None] - en[None, :]) ** 2
        X = 1.0 + ken * d2 + ken * self.enscale4 * d2 * d2
        zeta = self.zeta[sz, sk]
        Y = 2.0 * torch.sqrt(zeta[:, None] * zeta[None, :]) / (zeta[:, None] + zeta[None, :])
        return K * X * Y * self.distance_polynomial(sz, sl, sk, shell_atom, x)

    def prepare(self, z: Tensor, x: Tensor, cn: Tensor, sz: Tensor, sk: Tensor, sl: Tensor,
                shell_atom: Tensor) -> List[Tensor]:
        return [x[:0, 0]]

    def potentials(self, q_shell: Tensor, mu: Tensor, theta: Tensor, shell_atom: Tensor,
                   ctx: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        return torch.zeros_like(q_shell), torch.zeros_like(mu), torch.zeros_like(theta)

    def charge_energy(self, q_shell: Tensor, mu: Tensor, theta: Tensor, shell_atom: Tensor, z: Tensor,
                      ctx: List[Tensor]) -> Tensor:
        return torch.zeros((), dtype=q_shell.dtype, device=q_shell.device)

    def classical(self, z: Tensor, x: Tensor) -> Tensor:
        """The repulsion, its exponent scaled with the electronegativity difference."""
        i, j = self.pairs(z.shape[0], x.device)
        en = self.en_method[z]
        d2 = (en[i] - en[j]) ** 2
        alpha, zeff = self.repa[z], self.repb[z]
        exponent = torch.sqrt(alpha[i] * alpha[j]) * (1.0 + 0.01 * self.renscale * (d2 + d2 * d2))
        r = (x[j] - x[i]).norm(p=2, dim=-1)
        return (zeff[i] * zeff[j] / r * torch.exp(-exponent * r ** self.kexp)).sum()

    def state_energy(self, z: Tensor, x: Tensor, state: List[Tensor]) -> Tensor:
        """Electrostatic energy of the charge model, two-body D4 dispersion and the short-range bond term."""
        d4 = self.d4
        n = z.shape[0]
        cn = state[1]
        i, j = self.pairs(n, x.device)
        dij = x[j] - x[i]
        r = dij.norm(p=2, dim=-1)
        ei = torch.cat([torch.stack([i, j]), torch.stack([j, i])], 1)
        batch = torch.zeros(n, dtype=torch.long, device=x.device)
        out = d4.evaluate(z, x * self.bohr, ei, torch.cat([dij, -dij]) * self.bohr, batch, 1,
                          torch.zeros((1, 3, 3), dtype=x.dtype, device=x.device),
                          torch.zeros((1, 3), dtype=torch.bool, device=x.device), state[3])
        disp = out["energy_2body"].sum() / d4.hartree
        inside = (z >= 5) & (z <= 9)
        sel = (inside[i] & inside[j] & (z[i] != z[j])).nonzero().squeeze(1)
        a, b = i[sel], j[sel]
        ta, tb = self.srb[z[a]], self.srb[z[b]]
        dd = (ta[:, 2] - tb[:, 2]).abs()
        r0 = ((1.0 - self.srb_k1 * dd - self.srb_k2 * dd * dd)
              * (ta[:, 0] + tb[:, 0] + 2.0 * self.srbshift + ta[:, 1] * cn[a] + tb[:, 1] * cn[b]))
        width = self.srbexp * (1.0 + self.srbken * (self.en[z[a]] - self.en[z[b]]) ** 2)
        srb = (self.srbpre * torch.exp(-width * (r[sel] - r0) ** 2)).sum()
        return state[2].sum() + disp + srb

    def reported_charges(self, charges: Tensor, state: List[Tensor]) -> Tensor:
        return state[0]

