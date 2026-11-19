"""Deployment of a bound GFN-FF model: the scriptable cores of its TorchScript export and the forces and strain
derivative that the ASE calculator and the MDI engine take from it.

A GFN-FF model is bound to the structure it was built from, so both routes serve that system with its atoms in
their order. The exported cores hold the rows of the terms as buffers and evaluate the eager terms of
:class:`~xnn.ffnn.models.gfnff.GFNFF`. For molecules and clusters (:class:`GFNFFExport`) the hydrogen- and
halogen-bond rows of each conformation are chosen inside the scripted module: every row the topology allows is
kept and the distance cuts are applied at each evaluation, which gives the rows of the model's lists. Periodic cells
(:class:`GFNFFCellExport`) keep the rows the model holds, which stay valid while every atom is within half the
model's ``skin`` of the positions they were chosen at. :func:`direct_outputs` serves molecules and periodic cells
through :meth:`~xnn.ffnn.models.gfnff.GFNFF.energy_and_gradient`, with the fused kernels where the model takes them.
"""
# no postponed annotations: TorchScript reads the class-level attribute annotations below
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from xnn.common.models.dispersion import HARTREE

from . import gfnfflib as lib

#: the buffers of the bound model that the export keeps, in this order (the first floating one sets its dtype)
_KEPT = ("rcov", "numbers", "b_ij", "b_sh", "b_fp", "b_al", "b_k", "b_ra", "b_ci", "b_cj", "a_idx", "a_t0", "a_k",
         "arad", "t_idx", "t_n", "t_p0", "t_k", "i_idx", "i_neg", "i_c0", "i_k", "r_ij", "r_c", "r_m", "r_cut",
         "m_idx", "m_c", "d_zz", "d_ref", "d_valid", "d_alpha", "d_w", "d_q", "d_r06", "d_r08", "e_diag", "e_g",
         "e_x0", "e_kappa", "e_con", "e_tot")
#: the per-row tables of the hydrogen bonds with an H bonded to the donor
_BB = ("h_rr", "h_p", "h_cnb", "h_nb", "h_nbm", "h_o1f", "h_o1x", "h_o1y", "h_o1ym", "h_n2f", "h_n2nb")


def _bb_tables(at: dict, rows: np.ndarray) -> Dict[str, Tensor]:
    """The tables of the rows ``(a, b, h)`` of the hydrogen bonds with an H bonded to the donor, as the model
    forms them (:meth:`~xnn.ffnn.models.gfnff.GFNFF._bind_hb`); the O1 and N2 rows are flagged per row."""
    a, b, h = np.asarray(rows, dtype=np.int64).reshape(-1, 3).T
    rr = at["rad"][a] + at["rad"][b]
    o1 = at["o1"][b]
    p = np.where(o1 > 0, 1.75, 1.34) * at["ac"][a] * at["bs"][b] * at["sq"][a] * at["sq"][b] * at["sh"][h]
    k, ky = at["nb"].shape[1], at["y1"].shape[1]
    return {"h_rr": torch.as_tensor(rr), "h_p": torch.as_tensor(p), "h_cnb": torch.as_tensor(at["cnb"][b] / rr),
            "h_nb": at["nb"][b].reshape(-1, k), "h_nbm": at["nbm"][b].reshape(-1, k),
            "h_o1f": torch.as_tensor(o1 > 0), "h_o1x": torch.as_tensor(at["x1"][b]),
            "h_o1y": at["y1"][b].reshape(-1, ky), "h_o1ym": at["y1m"][b].reshape(-1, ky),
            "h_n2f": torch.as_tensor(at["n2"][b]), "h_n2nb": at["nb"][b, :2].reshape(-1, min(k, 2))}


class _Terms(nn.Module):
    """The bonded terms shared by the exported cores; ``_d`` gives the distances of index pairs (Bohr)."""

    tors_damp: float
    angle_damp: float

    def _td(self, x: Tensor, dist: Tensor, rad: Tensor, p: Tensor, q: Tensor) -> Tensor:
        return 1.0 / (1.0 + self.tors_damp * (self._d(x, dist, p, q) / (rad[p] + rad[q])) ** 4)

    def _hb_damp(self, rr: Tensor, rab: Tensor) -> Tensor:
        return 1.0 / (1.0 + (22.0 * rr / rab ** 2) ** 6) / (1.0 + (rab ** 2 / 85.0) ** 6)

    def _bonded(self, x: Tensor, dist: Tensor, logcn: Tensor, rad: Tensor) -> List[Tensor]:
        """Bond, angle and torsion energies (Hartree) at the positions ``x`` (Bohr) with the coordination numbers
        ``logcn`` and angle radii ``rad`` of their atoms."""
        i, j = self.b_ij[:, 0], self.b_ij[:, 1]
        r0 = self.b_fp * (self.b_ra + self.b_ci * logcn[i] + self.b_cj * logcn[j] + self.b_sh)
        ib, h, b = self.b_acc[:, 0], self.b_acc[:, 1], self.b_acc[:, 2]
        rhb = self._d(x, dist, h, b)
        m = torch.zeros_like(r0).index_add(0, ib, 0.5 * (1.0 + torch.erf(-27.5 * (rhb / self.b_accd - 1.0))))
        e_bond = (self.b_k * torch.exp(-self.b_al * (1.0 - 0.1 * m) * (self._d(x, dist, i, j) - r0) ** 2)).sum()

        c, a, b = self.a_idx[:, 0], self.a_idx[:, 1], self.a_idx[:, 2]
        cos = ((x[a] - x[c]) * (x[b] - x[c])).sum(-1) / (self._d(x, dist, c, a) * self._d(x, dist, c, b))
        damp = 1.0 / (1.0 + self.angle_damp * (self._d(x, dist, c, a) / (rad[c] + rad[a])) ** 4) \
            / (1.0 + self.angle_damp * (self._d(x, dist, c, b) / (rad[c] + rad[b])) ** 4)
        lin = self.a_lin
        u, v = x[a[lin]] - x[c[lin]], x[b[lin]] - x[c[lin]]
        e_lin = torch.atan2(torch.linalg.cross(u, v).norm(2, -1), -(u * v).sum(-1)) ** 2
        e_ang = torch.where(lin, torch.zeros_like(cos), (cos - torch.cos(self.a_t0)) ** 2).masked_scatter(lin, e_lin)
        e_angle = (self.a_k * e_ang * damp).sum()

        a, b, c, l = self.t_idx[:, 0], self.t_idx[:, 1], self.t_idx[:, 2], self.t_idx[:, 3]
        b0, b1, b2 = x[c] - x[a], x[b] - x[c], x[l] - x[b]
        n1, n2 = torch.linalg.cross(b0, b1), torch.linalg.cross(b1, b2)
        mm = torch.linalg.cross(n1, b1 / b1.norm(2, -1, True))
        phi = torch.atan2((mm * n2).sum(-1), (n1 * n2).sum(-1))
        e_t = self.t_k * self._td(x, dist, rad, a, c) * self._td(x, dist, rad, c, b) * self._td(x, dist, rad, b, l) \
            * (1.0 + torch.cos(self.t_n * (phi - self.t_p0) + math.pi))
        c, j, k, l = self.i_idx[:, 0], self.i_idx[:, 1], self.i_idx[:, 2], self.i_idx[:, 3]
        nrm, v = torch.linalg.cross(x[j] - x[c], x[k] - x[c]), x[l] - x[c]
        sw = (nrm * v).sum(-1) / (nrm.norm(2, -1) * v.norm(2, -1))
        cw = torch.sqrt((1.0 - sw ** 2).clamp_min(0.0))
        e_i = self.i_k * self._td(x, dist, rad, c, j) * self._td(x, dist, rad, j, k) * self._td(x, dist, rad, j, l) \
            * torch.where(self.i_neg, (cw - self.i_c0) ** 2, 1.0 - cw)
        return [e_bond, e_angle, e_t.sum() + e_i.sum()]


class GFNFFExport(_Terms):
    """``node_features_energy_charges`` of a bound GFN-FF model: the energy of one conformation of a molecule or
    cluster as equal per-atom shares (eV), its EEQ charges, and no features.

    Parameters
    ----------
    model : GFNFF
        A model built without a cell.

    Raises
    ------
    ValueError
        If the model is periodic.
    """

    cutoff: float
    n_atoms: int
    bohr: float
    hartree: float
    log_cap: float
    angle_damp: float
    tors_damp: float
    disp_cut: float
    switch_cut: float
    hb_cut: float
    hb_cut3: float
    xb_cut: float
    loose_fixed: bool

    def __init__(self, model: nn.Module):
        super().__init__()
        if model.cell is not None:
            raise ValueError("a model with a cell exports with GFNFFCellExport")
        from .gfnff import DISPERSION_CUT

        dtype, device = model.rcov.dtype, model.rcov.device
        self.cutoff = float(model.cutoff)
        self.n_atoms = int(model.num_atoms)
        self.bohr, self.hartree = float(lib.BOHR), float(HARTREE)
        self.log_cap = math.log(1.0 + math.exp(4.4))
        self.angle_damp, self.tors_damp = float(lib.ANGLE_DAMP), float(lib.TORS_DAMP)
        self.disp_cut, self.switch_cut = float(DISPERSION_CUT), float(lib.HB_SWITCH_CUT)
        for name in _KEPT:
            self.register_buffer(name, getattr(model, name).detach().clone())
        self.register_buffer("a_lin", (self.a_t0 - math.pi).abs() < 1e-6)

        t = model.topology
        z, q, nb = t.z, t.qt, lib.hb_neighbours(t)
        at = model._hb_atoms()
        x = np.asarray(model._ref, dtype=float) / lib.BOHR
        flt = lambda v: torch.as_tensor(np.asarray(v, dtype=float).reshape(-1)).to(device, dtype)  # noqa: E731
        idx = lambda v, w: torch.as_tensor(np.asarray(v, dtype=np.int64).reshape(-1, w)).to(device)  # noqa: E731
        self.loose_fixed = bool(model.fixed_lists)
        if model.fixed_lists:
            # the rows of the build structure, every one of them evaluated (as the model does)
            self.hb_cut = self.hb_cut3 = self.xb_cut = 1e30
            bb, bl = model.h_bb.cpu().numpy(), model.h_bl.cpu().numpy()
            xb = model.x_idx.cpu().numpy()
            acc, accd = model.b_acc.cpu().numpy(), model.b_accd.cpu().numpy()
        else:
            # every row the topology allows; the cuts of each evaluation select them
            self.hb_cut, self.hb_cut3, self.xb_cut = float(lib.HB_CUT), float(lib.HB_CUT3), float(lib.XB_CUT)
            bb, _ = lib.hb_rows(z, nb, q, x, math.inf, -1.0)
            bl = np.zeros((0, 3), dtype=np.int64)
            xb = lib.xb_rows(t, x, math.inf)
            hbb = lib.hbb_list(t, x, math.inf)
            acc = [(ib, h, b) for ib, h in model._bond_h
                   for b in sorted({b_ for _, b_, h_ in hbb if h_ == h and z[b_] in (7, 8)})]
            accd = [1.78 * (4.0 / 3.0) * (lib.RCOV[1] + lib.RCOV[z[b]]) for _, _, b in acc]
        self.register_buffer("b_acc", idx(acc, 3))
        self.register_buffer("b_accd", flt(accd))
        self.register_buffer("h_bb", idx(bb, 3))
        for name, value in _bb_tables(at, bb).items():
            value = value.to(device)
            self.register_buffer(name, value.to(dtype) if value.is_floating_point() else value)
        self.register_buffer("x_idx", idx(xb, 3))
        self.register_buffer("x_rad", flt([lib.TOPO_RAD[z[d]] + lib.TOPO_RAD[z[y]] for d, y, xa in xb]))
        self.register_buffer("x_p", flt([lib.XB_C[z[xa]] / ((1.0 + 0.03 * math.exp(-15.0 * q[xa]))
                                                            * (1.0 + 0.03 * math.exp(15.0 * q[y])))
                                         for d, y, xa in xb]))
        # the loose hydrogen bonds: per-atom tables (float64, cast per row as the model casts its rows), the
        # fixed rows or the candidate pairs and H atoms of hb_rows
        f64 = lambda v: torch.as_tensor(np.asarray(v, dtype=float)).to(device)  # noqa: E731
        self.register_buffer("l_rad", f64(at["rad"]))
        self.register_buffer("l_ac", f64(at["ac"]))
        self.register_buffer("l_bs", f64(at["bs"] * at["sq"]))
        self.register_buffer("l_sh", f64(at["sh"]))
        self.register_buffer("h_bl", idx(bl, 3))
        n = len(z)
        act = [lib.hb_active(z, nb, q, i) for i in range(n)]
        # the H atoms of the loose rows of hb_rows, with the atom each is bonded to
        one = {h: next(iter(nb[h])) for h in range(n) if z[h] == 1 and len(nb[h]) == 1}
        hs = [h for h, a in one.items() if q[h] > lib.HB_H_CUT or z[a] > 10]
        loose = [h for h in hs if not (z[one[h]] == 6 and len(nb[one[h]]) >= 4 and q[h] <= 0.06)]
        on = [i for i in range(n) if act[i]]
        pairs = [(i, j, j in nb[i]) for i in on for j in on if j < i and act[i] + act[j] >= 3]
        self.register_buffer("p_i", idx([p[0] for p in pairs], 1).reshape(-1))
        self.register_buffer("p_j", idx([p[1] for p in pairs], 1).reshape(-1))
        self.register_buffer("p_bond", torch.as_tensor([bool(p[2]) for p in pairs], dtype=torch.bool).to(device))
        self.register_buffer("lh", idx(loose, 1).reshape(-1))
        self.register_buffer("dh", idx([one[h] for h in loose], 1).reshape(-1))

    def _d(self, x: Tensor, dist: Tensor, p: Tensor, q: Tensor) -> Tensor:
        return dist[p, q]

    def _hbb(self, x: Tensor, dist: Tensor) -> Tensor:
        """Hydrogen bonds of H atoms bonded to the donor, over the rows within the cut."""
        sel = torch.nonzero(dist[self.h_bb[:, 0], self.h_bb[:, 1]].detach() ** 2 < self.hb_cut).reshape(-1)
        a, b, h = self.h_bb[sel, 0], self.h_bb[sel, 1], self.h_bb[sel, 2]
        rr, h_p, h_cnb = self.h_rr[sel], self.h_p[sel], self.h_cnb[sel]
        h_nb, h_nbm = self.h_nb[sel], self.h_nbm[sel]
        rab, rah, rbh = dist[a, b], dist[a, h], dist[b, h]
        arg = 49.0 / rr * ((rah + rbh) / rab - 1.0)
        on = (arg < self.switch_cut) & (rab.detach() ** 2 < self.hb_cut)
        o = torch.where(on, 2.0 / (1.0 + torch.exp(arg.clamp_max(self.switch_cut))), torch.zeros_like(arg))
        arm = (dist[a.unsqueeze(1), h_nb] + dist[b.unsqueeze(1), h_nb]) / rab.unsqueeze(1)
        fac = 2.0 / (1.0 + torch.exp(-h_cnb.unsqueeze(1) * (arm - 1.0))) - 1.0
        o = o * torch.where(h_nbm, fac, torch.ones_like(fac)).prod(1)
        r = torch.nonzero(self.h_o1f[sel]).reshape(-1)
        if r.shape[0] > 0:
            xb = self.h_o1x[sel][r]
            o1y, o1ym = self.h_o1y[sel][r], self.h_o1ym[sel][r]
            u, v = x[xb] - x[b[r]], x[h[r]] - x[b[r]]
            cos = (u * v).sum(-1) / (u.norm(2, -1) * v.norm(2, -1))
            f1 = 1.0 - 16.0 / 45.0 * (cos + 0.5) ** 2
            b1 = (u / u.norm(2, -1, True)).unsqueeze(1)
            b2 = x[o1y] - x[xb].unsqueeze(1)
            vv = (v.unsqueeze(1) - (v.unsqueeze(1) * b1).sum(-1, keepdim=True) * b1).expand_as(b2)
            ww = b2 - (b2 * b1).sum(-1, keepdim=True) * b1
            s = (torch.linalg.cross(b1.expand_as(b2), vv) * ww).sum(-1)
            c = (vv * ww).sum(-1)
            den = s ** 2 + c ** 2
            sin2 = torch.where(den > 0, s ** 2 / torch.where(den > 0, den, torch.ones_like(den)),
                               torch.ones_like(den))
            f1 = f1 * torch.where(o1ym, 1.0 - 0.06 * sin2, torch.ones_like(sin2)).prod(1)
            o = o * torch.ones_like(o).index_copy(0, r, f1)
        r = torch.nonzero(self.h_n2f[sel]).reshape(-1)
        if r.shape[0] > 0:
            n2nb = self.h_n2nb[sel][r]
            u = x[b[r]] - 0.5 * (x[n2nb[:, 0]] + x[n2nb[:, 1]])
            lp = x[b[r]] + 0.41 * u / u.norm(2, -1, True)
            arg = 56.0 / rr[r] * ((0.41 + (x[a[r]] - lp).norm(2, -1)) / rab[r] - 1.0)
            o = o * torch.ones_like(o).index_copy(0, r, 2.0 / (1.0 + torch.exp(arg.clamp_max(300.0))))
        return (-h_p * o * self._hb_damp(rr, rab) * (1.8 / rbh ** 3 - 0.8 / rab ** 3)).sum()

    def _loose_rows(self, dist: Tensor) -> Tensor:
        """The loose hydrogen-bond rows ``(A, B, H)`` within the cuts, as :func:`~xnn.ffnn.models.gfnfflib.hb_rows`
        chooses them: pairs of active atoms within the cut, and the H atoms bonded to both or to neither with
        ``r_AB^2 + r_AH^2 + r_BH^2`` below the three-side cut."""
        if self.loose_fixed:
            return self.h_bl
        d = dist.detach()
        pd2 = d[self.p_i, self.p_j] ** 2
        ps = torch.nonzero(pd2 < self.hb_cut).reshape(-1)
        i, j, bond = self.p_i[ps], self.p_j[ps], self.p_bond[ps]
        d2 = pd2[ps].unsqueeze(1) + d[i][:, self.lh] ** 2 + d[j][:, self.lh] ** 2
        free = (self.dh.unsqueeze(0) != i.unsqueeze(1)) & (self.dh.unsqueeze(0) != j.unsqueeze(1))
        ok = (d2 < self.hb_cut3) & (bond.unsqueeze(1) | free)
        rc = torch.nonzero(ok)
        return torch.stack([i[rc[:, 0]], j[rc[:, 0]], self.lh[rc[:, 1]]], 1)

    def _hbl(self, dist: Tensor) -> Tensor:
        """Hydrogen bonds of H atoms bonded to neither or to both partners."""
        rows = self._loose_rows(dist)
        a, b, h = rows[:, 0], rows[:, 1], rows[:, 2]
        dtype = dist.dtype
        rr = (self.l_rad[a] + self.l_rad[b]).to(dtype)
        l_ca, l_cb = self.l_ac[a].to(dtype), self.l_ac[b].to(dtype)
        l_ba, l_bb, l_qh = self.l_bs[a].to(dtype), self.l_bs[b].to(dtype), self.l_sh[h].to(dtype)
        rab, rah, rbh = dist[a, b], dist[a, h], dist[b, h]
        arg = 49.0 / rr * ((rah + rbh) / rab - 1.0)
        r2 = rab.detach() ** 2
        on = (arg < self.switch_cut) & (r2 < self.hb_cut) \
            & (r2 + rah.detach() ** 2 + rbh.detach() ** 2 < self.hb_cut3)
        o = torch.where(on, 2.0 / (1.0 + torch.exp(arg.clamp_max(self.switch_cut))), torch.zeros_like(arg))
        wa, wb = rbh ** 4, rah ** 4
        acid = (l_ca * wa + l_cb * wb) / (wa + wb)
        base = (l_ba * wb + l_bb * wa) / (wa + wb)
        return (-5.0 * o * self._hb_damp(rr, rab) * acid * base * l_qh / rab ** 3).sum()

    def _xb(self, dist: Tensor) -> Tensor:
        """Halogen bonds over the rows within the cut."""
        sel = torch.nonzero(dist[self.x_idx[:, 0], self.x_idx[:, 1]].detach() ** 2 < self.xb_cut).reshape(-1)
        d, y, xa = self.x_idx[sel, 0], self.x_idx[sel, 1], self.x_idx[sel, 2]
        rdx, rxy, rdy = dist[d, xa], dist[xa, y], dist[d, y]
        arg = 70.0 * ((rdx + rxy) / rdy - 1.0)
        on = (arg < self.switch_cut) & (rdy.detach() ** 2 < self.xb_cut)
        o = torch.where(on, 2.0 / (1.0 + torch.exp(arg.clamp_max(self.switch_cut))), torch.zeros_like(arg))
        srt = 1.0 / (1.0 + (5.0 * self.x_rad[sel] / rxy ** 2) ** 6)
        return (-self.x_p[sel] * o * srt / (1.0 + (rxy ** 2 / 70.0) ** 6) / rxy ** 3).sum()

    def terms(self, pos: Tensor) -> Tuple[List[Tensor], Tensor]:
        """The energies (Hartree) in the order of :data:`~xnn.ffnn.models.gfnff.TERMS` and the EEQ charges of the
        conformation at ``pos`` (Angstrom, ``(N, 3)``)."""
        x = pos.to(self.rcov.dtype) / self.bohr
        n = self.n_atoms
        eye = torch.eye(n, dtype=x.dtype, device=x.device)
        dist = torch.sqrt((x[:, 0].unsqueeze(1) - x[:, 0].unsqueeze(0)) ** 2
                          + (x[:, 1].unsqueeze(1) - x[:, 1].unsqueeze(0)) ** 2
                          + (x[:, 2].unsqueeze(1) - x[:, 2].unsqueeze(0)) ** 2 + eye)
        rc = (4.0 / 3.0) * (self.rcov.unsqueeze(1) + self.rcov.unsqueeze(0))
        ij = torch.nonzero((dist.detach() < 2.0 * rc) & (eye == 0))
        i, j = ij[:, 0], ij[:, 1]
        count = torch.zeros(n, dtype=x.dtype, device=x.device).index_add(
            0, i, 0.5 * (1.0 + torch.erf(-7.5 * (dist[i, j] / rc[i, j] - 1.0))))
        logcn = self.log_cap - torch.log(1.0 + torch.exp(4.4 - count))
        out = self._bonded(x, dist, logcn, self.arad)
        i, j = self.r_ij[:, 0], self.r_ij[:, 1]
        keep = dist.detach()[i, j] < self.r_cut
        rr, rep_m = dist[i[keep], j[keep]], self.r_m[keep]
        out.append((self.r_c[keep] / rr * torch.exp(-rep_m * rr * torch.sqrt(rr))).sum())
        i, j, k = self.m_idx[:, 0], self.m_idx[:, 1], self.m_idx[:, 2]
        rij, rik, rjk = dist[i, j], dist[i, k], dist[j, k]
        ci = (rij ** 2 + rik ** 2 - rjk ** 2) / (2 * rij * rik)
        cj = (rij ** 2 + rjk ** 2 - rik ** 2) / (2 * rij * rjk)
        ck = (rik ** 2 + rjk ** 2 - rij ** 2) / (2 * rik * rjk)
        out.append((self.m_c * (3.0 * ci * cj * ck + 1.0) / (rij * rik * rjk) ** 3).sum())
        g = torch.exp(-4.0 * (logcn.unsqueeze(1) - self.d_ref) ** 2)
        g = torch.where(self.d_valid, g, torch.zeros_like(g))
        alpha = torch.einsum("nr,krn->nk", [g / g.sum(1, keepdim=True), self.d_alpha])
        c6 = (3.0 / math.pi) * (alpha * self.d_w) @ alpha.t()
        ij = torch.nonzero(torch.triu(dist < self.disp_cut, 1))
        i, j = ij[:, 0], ij[:, 1]
        r2 = dist[i, j] ** 2
        r6 = r2 * r2 * r2
        pair = 1.0 / (r6 + self.d_r06[i, j]) + 2.0 * self.d_q[i, j] / (r6 * r2 + self.d_r08[i, j])
        pair = self.d_zz[i, j] * c6[i, j] * pair
        out.append(-pair.sum())
        out.append(self._hbb(x, dist) + self._hbl(dist))
        out.append(self._xb(dist))
        amat = torch.erf(self.e_g * dist) / dist * (1.0 - eye) + torch.diag(self.e_diag)
        pos_cn = logcn > 0
        xx = self.e_x0 + self.e_kappa * torch.where(
            pos_cn, torch.sqrt(torch.where(pos_cn, logcn, torch.ones_like(logcn))), torch.zeros_like(logcn))
        nc = self.e_tot.shape[0]
        zero = torch.zeros((nc, nc), dtype=x.dtype, device=x.device)
        mat = torch.cat([torch.cat([amat, self.e_con.t()], 1), torch.cat([self.e_con, zero], 1)], 0)
        sol = torch.linalg.solve(mat, torch.cat([xx, self.e_tot]))
        q, lam = sol[:n], sol[n:]
        out.append(-(q * xx).sum() + 0.5 * q @ amat @ q + lam @ (self.e_con @ q - self.e_tot))
        return out, q

    def node_features_energy_charges(self, atomic_numbers: Tensor, edge_index: Tensor, edge_vec: Tensor,
                                     pos: Tensor, cell: Tensor, pbc: Tensor, total_charge: float,
                                     spin_multiplicity: float) -> Tuple[Tensor, Tensor, Tensor]:
        """The features ``(N, 0)``, the energy as equal per-atom shares ``(N,)`` in eV and the EEQ charges ``(N,)``
        of the bound system at ``pos``; the neighbor list is not used (the model evaluates every pair), and the
        charge is the bound one.

        Raises
        ------
        RuntimeError
            If the atoms are not those of the bound system in its order, or the structure is periodic.
        """
        if atomic_numbers.shape[0] != self.n_atoms or not bool(torch.equal(atomic_numbers, self.numbers)):
            raise RuntimeError("the atoms differ from those of the system the model was bound to")
        if bool(pbc.any()):
            raise RuntimeError("the exported GFN-FF model evaluates molecules and clusters, not cells")
        terms, q = self.terms(pos)
        energy = terms[0] * self.hartree
        for k in range(1, len(terms)):
            energy = energy + terms[k] * self.hartree
        node = (energy / self.n_atoms).reshape(1).repeat(self.n_atoms)
        return torch.zeros((self.n_atoms, 0), dtype=pos.dtype, device=pos.device), node, q


#: the buffers of a bound periodic model that the export keeps, in this order
_KEPT_CELL = ("rcov", "numbers", "b_ij", "b_sh", "b_fp", "b_al", "b_k", "b_ra", "b_ci", "b_cj", "a_idx", "a_t0",
              "a_k", "arad", "t_idx", "t_n", "t_p0", "t_k", "i_idx", "i_neg", "i_c0", "i_k", "r_ij", "r_sh", "r_c",
              "r_m", "r_cut", "m_idx", "m_c", "d_zz", "d_ref", "d_valid", "d_alpha", "d_w", "d_q", "d_r06", "d_r08",
              "e_diag", "e_g", "e_x0", "e_kappa", "x_base", "x_shift", "x_wrap", "cn_shift", "b_acc", "b_accd")
#: the hydrogen- and halogen-bond rows of a periodic model (kept in float64 and cast per evaluation, as the model
#: casts them)
_ROWS_CELL = ("bb_idx", "bb_sh", "bb_rr", "bb_p", "bb_cnb", "bb_nb", "bb_nbsh", "bb_nbm", "o1_rows", "o1_x",
              "o1_xsh", "o1_y", "o1_ysh", "o1_ym", "n2_rows", "n2_nb", "n2_nbsh", "bl_idx", "bl_sh", "bl_rr", "bl_ca",
              "bl_cb", "bl_ba", "bl_bb", "bl_qh", "xb_idx", "xb_sh", "xb_rad", "xb_p")


class GFNFFCellExport(_Terms):
    """``node_features_energy_charges`` of a bound periodic GFN-FF model: the energy of one structure of the bound
    cell as equal per-atom shares (eV), its EEQ charges, and no features.

    The terms are those of the model's periodic path: the bonded terms over the images of the topology, the
    coordination numbers, the dispersion over the translation list of each cell, the hydrogen and halogen bonds of
    the rows the model holds and the electrostatics by its Ewald sum. The positions are unwrapped against the
    positions the rows were chosen at, so an atom wrapped back into the cell keeps its bonds.

    Parameters
    ----------
    model : GFNFF
        A model built with a cell.

    Raises
    ------
    ValueError
        If the model has no cell.
    """

    cutoff: float
    n_atoms: int
    bohr: float
    hartree: float
    log_cap: float
    angle_damp: float
    tors_damp: float
    switch_cut: float
    hb_cut: float
    hb_cut3: float
    xb_cut: float
    disp_cut: float
    disp_list2: float
    near_image: float
    gamma: float
    skin2: float

    def __init__(self, model: nn.Module):
        super().__init__()
        if model.cell is None:
            raise ValueError("a model without a cell exports with GFNFFExport")
        from . import gfnff_periodic as per

        dtype, device = model.rcov.dtype, model.rcov.device
        self.cutoff = float(model.cutoff)
        self.n_atoms = int(model.num_atoms)
        self.bohr, self.hartree = float(lib.BOHR), float(HARTREE)
        self.log_cap = math.log(1.0 + math.exp(4.4))
        self.angle_damp, self.tors_damp = float(lib.ANGLE_DAMP), float(lib.TORS_DAMP)
        self.switch_cut = float(lib.HB_SWITCH_CUT)
        self.hb_cut, self.hb_cut3, self.xb_cut = float(lib.HB_CUT), float(lib.HB_CUT3), float(lib.XB_CUT)
        self.disp_cut, self.disp_list2, self.near_image = float(per.DISP_CUT), float(per.DISP_LIST2), per.NEAR_IMAGE
        self.gamma = per.mean_gamma(model)
        # the rows stay valid while every atom is within half the skin of the positions they were chosen at
        self.skin2 = math.inf if model.fixed_lists else float(model.skin) ** 2
        for name in _KEPT_CELL:
            self.register_buffer(name, getattr(model, name).detach().clone())
        self.register_buffer("a_lin", (self.a_t0 - math.pi).abs() < 1e-6)
        rows = model._hbxb_rows
        for name in _ROWS_CELL:
            value = torch.as_tensor(np.asarray(rows[name]))
            if value.is_floating_point():
                value = value.double()
            elif value.dtype != torch.bool:
                value = value.long()
            self.register_buffer(name, value.to(device))
        con, tot = per._constraints(model, dtype, device)
        self.register_buffer("q_con", con)
        self.register_buffer("q_tot", tot)
        self.register_buffer("ref", torch.as_tensor(np.asarray(model._ref, dtype=float)).to(device))
        f64 = lambda v: torch.as_tensor(np.asarray(v, dtype=float)).to(device)  # noqa: E731
        self.register_buffer("short", f64(per._SHORT))
        self.register_buffer("cube1", f64(per._CUBE1))
        self.register_buffer("cube2", f64(per._CUBE2))
        self.register_buffer("cube2_nz", f64(per._CUBE2[np.any(per._CUBE2 != 0, axis=1)]))

    def _d(self, x: Tensor, dist: Tensor, p: Tensor, q: Tensor) -> Tensor:
        return torch.sqrt(((x[q] - x[p]) ** 2).sum(-1))

    def _translations(self, cellb: Tensor) -> Tensor:
        """The lattice shifts of the capped translation list of the dispersion
        (:func:`~xnn.ffnn.models.gfnfflib.translation_list`) for the cell ``cellb`` (Bohr)."""
        cell = cellb.detach().double() * self.bohr / self.bohr
        r = math.sqrt(self.disp_list2)
        h = 1.0 / torch.linalg.norm(torch.linalg.inv(cell), dim=0)
        k = [max(1, int(math.ceil(r / float(h[0])))), max(1, int(math.ceil(r / float(h[1])))),
             max(1, int(math.ceil(r / float(h[2]))))]
        top = min(max(k), max(1, int(math.ceil(r / float(torch.linalg.norm(cell, dim=1).min())))))
        dev = cell.device
        c, b, a = torch.meshgrid([torch.arange(-k[2], k[2] + 1, device=dev), torch.arange(-k[1], k[1] + 1, device=dev),
                                  torch.arange(-k[0], k[0] + 1, device=dev)], indexing="ij")
        p = torch.stack([a.reshape(-1), b.reshape(-1), c.reshape(-1)], 1)
        shell = p.abs().max(1)[0]
        far = torch.linalg.norm(p.double() @ cell, dim=1) * (shell - 1).double() / shell.clamp_min(1).double()
        keep = (shell <= top) & ((shell < 2) | (far < r))
        p, shell = p[keep], shell[keep]
        order = torch.sort(shell, stable=True)[1]
        return p[order]

    def _eta(self, cellb: Tensor) -> float:
        """The Ewald splitting parameter (:func:`~xnn.ffnn.models.gfnff_periodic.ewald_eta`) of the cell ``cellb``
        (Bohr)."""
        c = cellb.detach().double()
        vol = abs(float(torch.det(c)))
        rmin = float(torch.linalg.norm(self.short @ c, dim=1).min())
        gmin = float(torch.linalg.norm(self.short @ (2.0 * math.pi * torch.linalg.inv(c).t()), dim=1).min())
        width = math.erfc(self.gamma * rmin) / rmin
        lo, hi = 1.0e-8, 2.0
        ratio = (1.0 + math.sqrt(2.0)) / 2.0
        while abs(hi - lo) >= 1.0e-4:
            left = hi - (hi - lo) / ratio
            right = lo + (hi - lo) / ratio
            fl = abs(math.erfc(left * rmin) / rmin - width
                     - 4.0 * math.pi / vol * math.exp(-gmin * gmin / (4.0 * left * left)) / gmin ** 2)
            fr = abs(math.erfc(right * rmin) / rmin - width
                     - 4.0 * math.pi / vol * math.exp(-gmin * gmin / (4.0 * right * right)) / gmin ** 2)
            if fl < fr:
                hi = right
            else:
                lo = left
        return 0.5 * (lo + hi)

    def _ewald(self, x: Tensor, cellb: Tensor, gam: Tensor, eta: float) -> Tensor:
        """The Ewald matrix of the charges without the hardness
        (:func:`~xnn.ffnn.models.gfnff_periodic._ewald_matrix`)."""
        n = x.shape[0]
        dt = x.dtype
        vol = torch.abs(torch.det(cellb))
        f = (x.unsqueeze(0) - x.unsqueeze(1)) @ torch.linalg.inv(cellb)
        v = (f.unsqueeze(2) + self.cube1.to(dt).unsqueeze(0).unsqueeze(0)) @ cellb
        ln = torch.sqrt((v.detach() ** 2).sum(-1))
        tied = ln <= ln.min(-1, True)[0] + self.near_image
        ijk = torch.nonzero(tied)
        i, j, k = ijk[:, 0], ijk[:, 1], ijk[:, 2]
        tr = self.cube2.to(dt) @ cellb
        d = v[i, j, k].unsqueeze(1) + tr.unsqueeze(0)
        r2 = (d ** 2).sum(-1)
        small = r2.detach() < 1e-20
        r = torch.sqrt(torch.where(small, torch.ones_like(r2), r2))
        pair = (torch.erfc(eta * r) - torch.erfc(gam[i, j].unsqueeze(1) * r)) / r
        per_pair = torch.where(small, torch.zeros_like(r), pair).sum(-1)
        at: List[Optional[Tensor]] = [i, j]
        real = torch.zeros((n, n), dtype=dt, device=x.device).index_put(at, per_pair, True)
        real = real / tied.sum(-1).to(dt)
        g = self.cube2_nz.to(dt) @ (2.0 * math.pi * torch.linalg.inv(cellb).t())
        k2 = (g ** 2).sum(-1)
        w = 4.0 * math.pi / vol * torch.exp(-k2 / (4.0 * eta ** 2)) / k2
        ph = x @ g.t()
        c, s = torch.cos(ph), torch.sin(ph)
        rec = (c * w) @ c.t() + (s * w) @ s.t()
        return real + rec - torch.diag(torch.full((n,), 2.0 * eta / math.sqrt(math.pi), dtype=dt, device=x.device))

    def _place(self, x: Tensor, cellb: Tensor, idx: Tensor, sh: Tensor) -> Tensor:
        return x[idx] + sh.to(x.dtype) @ cellb

    def _hb_xb(self, x: Tensor, cellb: Tensor) -> List[Tensor]:
        """Hydrogen- and halogen-bond energies (Hartree) of the rows (:func:`~xnn.ffnn.models.gfnff_periodic.hb_xb`)."""
        dt = x.dtype
        e = torch.zeros((), dtype=dt, device=x.device)
        if self.bb_idx.shape[0] > 0:
            idx, sh = self.bb_idx, self.bb_sh
            xa, xb, xh = (self._place(x, cellb, idx[:, 0], sh[:, 0]), self._place(x, cellb, idx[:, 1], sh[:, 1]),
                          self._place(x, cellb, idx[:, 2], sh[:, 2]))
            rr = self.bb_rr.to(dt)
            rab, rah, rbh = (xa - xb).norm(2, -1), (xa - xh).norm(2, -1), (xb - xh).norm(2, -1)
            arg = 49.0 / rr * ((rah + rbh) / rab - 1.0)
            on = (arg < self.switch_cut) & (rab.detach() ** 2 < self.hb_cut)
            o = torch.where(on, 2.0 / (1.0 + torch.exp(arg.clamp_max(self.switch_cut))), torch.zeros_like(arg))
            xn = x[self.bb_nb] + self.bb_nbsh.to(dt) @ cellb
            arm = ((xa.unsqueeze(1) - xn).norm(2, -1) + (xb.unsqueeze(1) - xn).norm(2, -1)) / rab.unsqueeze(1)
            fac = 2.0 / (1.0 + torch.exp(-self.bb_cnb.to(dt).unsqueeze(1) * (arm - 1.0))) - 1.0
            o = o * torch.where(self.bb_nbm, fac, torch.ones_like(fac)).prod(1)
            if self.o1_rows.shape[0] > 0:
                r = self.o1_rows
                xx1 = x[self.o1_x] + self.o1_xsh.to(dt) @ cellb
                u, v = xx1 - xb[r], xh[r] - xb[r]
                cos = (u * v).sum(-1) / (u.norm(2, -1) * v.norm(2, -1))
                f1 = 1.0 - 16.0 / 45.0 * (cos + 0.5) ** 2
                b1 = (u / u.norm(2, -1, True)).unsqueeze(1)
                b2 = x[self.o1_y] + self.o1_ysh.to(dt) @ cellb - xx1.unsqueeze(1)
                vv = (v.unsqueeze(1) - (v.unsqueeze(1) * b1).sum(-1, keepdim=True) * b1).expand_as(b2)
                ww = b2 - (b2 * b1).sum(-1, keepdim=True) * b1
                s = (torch.linalg.cross(b1.expand_as(b2), vv) * ww).sum(-1)
                c = (vv * ww).sum(-1)
                den = s ** 2 + c ** 2
                sin2 = torch.where(den > 0, s ** 2 / torch.where(den > 0, den, torch.ones_like(den)),
                                   torch.ones_like(den))
                f1 = f1 * torch.where(self.o1_ym, 1.0 - 0.06 * sin2, torch.ones_like(sin2)).prod(1)
                o = o * torch.ones_like(o).index_copy(0, r, f1)
            if self.n2_rows.shape[0] > 0:
                r = self.n2_rows
                nb2 = x[self.n2_nb] + self.n2_nbsh.to(dt) @ cellb
                u = xb[r] - 0.5 * (nb2[:, 0] + nb2[:, 1])
                lp = xb[r] + 0.41 * u / u.norm(2, -1, True)
                arg2 = 56.0 / rr[r] * ((0.41 + (xa[r] - lp).norm(2, -1)) / rab[r] - 1.0)
                o = o * torch.ones_like(o).index_copy(0, r, 2.0 / (1.0 + torch.exp(arg2.clamp_max(300.0))))
            damp = 1.0 / (1.0 + (22.0 * rr / rab ** 2) ** 6) / (1.0 + (rab ** 2 / 85.0) ** 6)
            e = e + (-self.bb_p.to(dt) * o * damp * (1.8 / rbh ** 3 - 0.8 / rab ** 3)).sum()
        if self.bl_idx.shape[0] > 0:
            idx, sh = self.bl_idx, self.bl_sh
            xa, xb, xh = (self._place(x, cellb, idx[:, 0], sh[:, 0]), self._place(x, cellb, idx[:, 1], sh[:, 1]),
                          self._place(x, cellb, idx[:, 2], sh[:, 2]))
            rr = self.bl_rr.to(dt)
            rab, rah, rbh = (xa - xb).norm(2, -1), (xa - xh).norm(2, -1), (xb - xh).norm(2, -1)
            arg = 49.0 / rr * ((rah + rbh) / rab - 1.0)
            r2 = rab.detach() ** 2
            on = (arg < self.switch_cut) & (r2 < self.hb_cut) \
                & (r2 + rah.detach() ** 2 + rbh.detach() ** 2 < self.hb_cut3)
            o = torch.where(on, 2.0 / (1.0 + torch.exp(arg.clamp_max(self.switch_cut))), torch.zeros_like(arg))
            wa, wb = rbh ** 4, rah ** 4
            acid = (self.bl_ca.to(dt) * wa + self.bl_cb.to(dt) * wb) / (wa + wb)
            base = (self.bl_ba.to(dt) * wb + self.bl_bb.to(dt) * wa) / (wa + wb)
            damp = 1.0 / (1.0 + (22.0 * rr / rab ** 2) ** 6) / (1.0 + (rab ** 2 / 85.0) ** 6)
            e = e + (-5.0 * o * damp * acid * base * self.bl_qh.to(dt) / rab ** 3).sum()
        ex = torch.zeros((), dtype=dt, device=x.device)
        if self.xb_idx.shape[0] > 0:
            idx, sh = self.xb_idx, self.xb_sh
            xd, xy, xx = (self._place(x, cellb, idx[:, 0], sh[:, 0]), self._place(x, cellb, idx[:, 1], sh[:, 1]),
                          self._place(x, cellb, idx[:, 2], sh[:, 2]))
            rdx, rxy, rdy = (xd - xx).norm(2, -1), (xx - xy).norm(2, -1), (xd - xy).norm(2, -1)
            arg = 70.0 * ((rdx + rxy) / rdy - 1.0)
            on = (arg < self.switch_cut) & (rdy.detach() ** 2 < self.xb_cut)
            o = torch.where(on, 2.0 / (1.0 + torch.exp(arg.clamp_max(self.switch_cut))), torch.zeros_like(arg))
            srt = 1.0 / (1.0 + (5.0 * self.xb_rad.to(dt) / rxy ** 2) ** 6)
            ex = (-self.xb_p.to(dt) * o * srt / (1.0 + (rxy ** 2 / 70.0) ** 6) / rxy ** 3).sum()
        return [e, ex]

    def terms(self, pos: Tensor, cell: Tensor) -> Tuple[List[Tensor], Tensor]:
        """The energies (Hartree) in the order of :data:`~xnn.ffnn.models.gfnff.TERMS` and the EEQ charges of the
        structure at ``pos`` (Angstrom, ``(N, 3)``, unwrapped) in the cell ``cell`` (Angstrom, rows)."""
        dtype = self.rcov.dtype
        cellb = cell.to(dtype) / self.bohr
        x = pos.to(dtype) / self.bohr - self.x_wrap @ cellb
        n = self.n_atoms
        eye = torch.eye(n, dtype=dtype, device=x.device)
        tr = self.cn_shift @ cellb
        dv = x.unsqueeze(0).unsqueeze(2) + tr.unsqueeze(0).unsqueeze(0) - x.unsqueeze(1).unsqueeze(2)
        own = (eye > 0).unsqueeze(2) & (self.cn_shift.abs().sum(-1) == 0).unsqueeze(0).unsqueeze(0)
        d = torch.sqrt((dv ** 2).sum(-1) + own.to(dtype))
        rc = ((4.0 / 3.0) * (self.rcov.unsqueeze(1) + self.rcov.unsqueeze(0))).unsqueeze(2)
        on = (d.detach() < 2.0 * rc) & ~own
        f = torch.where(on, 0.5 * (1.0 + torch.erf(-7.5 * (d / rc - 1.0))), torch.zeros_like(d))
        f = f * (1.0 + eye.unsqueeze(2))
        count = f.sum((1, 2))
        logcn = self.log_cap - torch.log(1.0 + torch.exp(4.4 - count))
        xe = torch.cat([x, x[self.x_base] + self.x_shift @ cellb])
        base = torch.cat([torch.arange(n, device=x.device), self.x_base])
        none = torch.zeros(0, dtype=dtype, device=x.device)
        out = self._bonded(xe, none, logcn[base], self.arad[base])
        i, j = self.r_ij[:, 0], self.r_ij[:, 1]
        rr = (x[j] + self.r_sh @ cellb - x[i]).norm(2, -1)
        keep = rr.detach() < self.r_cut
        rr, rep_m = rr[keep], self.r_m[keep]
        out.append((self.r_c[keep] / rr * torch.exp(-rep_m * rr * torch.sqrt(rr))).sum())
        i, j, k = self.m_idx[:, 0], self.m_idx[:, 1], self.m_idx[:, 2]
        rij, rik, rjk = self._d(xe, none, i, j), self._d(xe, none, i, k), self._d(xe, none, j, k)
        ci = (rij ** 2 + rik ** 2 - rjk ** 2) / (2 * rij * rik)
        cj = (rij ** 2 + rjk ** 2 - rik ** 2) / (2 * rij * rjk)
        ck = (rik ** 2 + rjk ** 2 - rij ** 2) / (2 * rik * rjk)
        out.append((self.m_c * (3.0 * ci * cj * ck + 1.0) / (rij * rik * rjk) ** 3).sum())
        # dispersion over the capped translation list
        g = torch.exp(-4.0 * (logcn.unsqueeze(1) - self.d_ref) ** 2)
        g = torch.where(self.d_valid, g, torch.zeros_like(g))
        alpha = torch.einsum("nr,krn->nk", [g / g.sum(1, keepdim=True), self.d_alpha])
        c6 = (3.0 / math.pi) * (alpha * self.d_w) @ alpha.t()
        trd = self._translations(cellb).to(dtype) @ cellb
        dd = x.unsqueeze(0).unsqueeze(2) - x.unsqueeze(1).unsqueeze(2) + trd.unsqueeze(0).unsqueeze(0)
        r2 = (dd ** 2).sum(-1)
        ok = (r2.detach() > 1e-20) & (r2.detach() < self.disp_cut ** 2)
        r2s = torch.where(ok, r2, torch.ones_like(r2))
        r6 = r2s ** 3
        e = 1.0 / (r6 + self.d_r06.unsqueeze(2)) + 2.0 * self.d_q.unsqueeze(2) / (r6 * r2s + self.d_r08.unsqueeze(2))
        e = torch.where(ok, (self.d_zz * c6).unsqueeze(2) * e, torch.zeros_like(e))
        w = torch.full((n, n), 0.5, dtype=dtype, device=x.device) + 0.5 * eye
        out.append(-(w.unsqueeze(2) * e).sum())
        out.extend(self._hb_xb(x, cellb))
        # electrostatics by the Ewald sum
        a = self._ewald(x, cellb, self.e_g, self._eta(cellb)) + torch.diag(self.e_diag)
        pos_cn = logcn > 0
        xx = self.e_x0 + self.e_kappa * torch.where(
            pos_cn, torch.sqrt(torch.where(pos_cn, logcn, torch.ones_like(logcn))), torch.zeros_like(logcn))
        nc = self.q_tot.shape[0]
        mat = torch.cat([torch.cat([a, self.q_con.t()], 1),
                         torch.cat([self.q_con, torch.zeros((nc, nc), dtype=dtype, device=x.device)], 1)], 0)
        sol = torch.linalg.solve(mat, torch.cat([xx, self.q_tot]))
        q, lam = sol[:n], sol[n:]
        out.append(-0.5 * (q * xx).sum() + 0.5 * (lam * self.q_tot).sum())
        return out, q

    def unwrap(self, pos: Tensor, cell: Tensor) -> Tensor:
        """``pos`` (Angstrom) moved by lattice vectors of ``cell`` to the images nearest the positions the rows were
        chosen at.

        Raises
        ------
        RuntimeError
            If an atom is then farther than half the model's skin from that position.
        """
        jumps = torch.round((pos.detach().double() - self.ref) @ torch.linalg.inv(cell.detach().double()))
        pos = pos - jumps.to(pos.dtype) @ cell
        d = pos.detach().double() - self.ref
        if 4.0 * float((d * d).sum(1).max()) > self.skin2:
            raise RuntimeError("an atom moved by more than half the skin from the positions of the exported "
                               "hydrogen- and halogen-bond rows: export the model with a larger skin")
        return pos

    def node_features_energy_charges(self, atomic_numbers: Tensor, edge_index: Tensor, edge_vec: Tensor,
                                     pos: Tensor, cell: Tensor, pbc: Tensor, total_charge: float,
                                     spin_multiplicity: float) -> Tuple[Tensor, Tensor, Tensor]:
        """The features ``(N, 0)``, the energy as equal per-atom shares ``(N,)`` in eV and the EEQ charges ``(N,)``
        of the bound cell at ``pos`` in ``cell``; the neighbor list is not used, and the charge is the bound one.

        Raises
        ------
        RuntimeError
            If the atoms are not those of the bound system in its order, the structure is not periodic in three
            directions, or an atom moved beyond the rows (see :meth:`unwrap`).
        """
        if atomic_numbers.shape[0] != self.n_atoms or not bool(torch.equal(atomic_numbers, self.numbers)):
            raise RuntimeError("the atoms differ from those of the system the model was bound to")
        if not bool(pbc.all()):
            raise RuntimeError("the exported periodic GFN-FF model evaluates cells periodic in three directions")
        terms, q = self.terms(self.unwrap(pos, cell), cell)
        energy = terms[0] * self.hartree
        for k in range(1, len(terms)):
            energy = energy + terms[k] * self.hartree
        node = (energy / self.n_atoms).reshape(1).repeat(self.n_atoms)
        return torch.zeros((self.n_atoms, 0), dtype=pos.dtype, device=pos.device), node, q


def direct_outputs(model: nn.Module, data) -> Dict[str, Tensor]:
    """The outputs of :meth:`~xnn.ffnn.models.gfnff.GFNFF.forward` with the forces and the strain derivative
    ``"virial"`` (eV) of every structure, from :meth:`~xnn.ffnn.models.gfnff.GFNFF.energy_and_gradient`. The
    positions of a periodic structure are first unwrapped against the positions of the model's hydrogen-bond
    lists (an atom a driver wraps back into the cell keeps its bonds). The outputs carry no autograd graph."""
    from .gfnff import TERMS

    n, nb = model.num_atoms, int(data.num_graphs)
    if data.pos.shape[0] != n * nb or not bool((data.atomic_numbers.reshape(nb, n) == model.numbers).all()):
        raise ValueError("every structure must be a conformation of the molecule the GFN-FF model was built for")
    out_dtype = data.pos.dtype
    energies, forces, virials, parts = [], [], [], []
    with torch.no_grad():
        for g in range(nb):
            pos = data.pos[g * n:(g + 1) * n].detach().double()
            cell = None
            if model.cell is not None:
                if data.cell is None:
                    raise ValueError("a periodic GFN-FF model needs the cell of every structure")
                cell = data.cell[g].detach().double()
                ref = torch.as_tensor(model._ref, dtype=torch.float64, device=pos.device)
                pos = pos - torch.round((pos - ref) @ torch.linalg.inv(cell)) @ cell
            res = model.energy_and_gradient(pos, cell)
            grad = res[1].double() * HARTREE
            virial = pos.T @ grad
            if cell is not None:
                virial = virial + cell.T @ (res[2].double() * HARTREE)
            term = res[0] * HARTREE
            energy = term[0]
            for k in range(1, len(TERMS)):
                energy = energy + term[k]
            energies.append(energy)
            parts.append(term)
            forces.append(-grad)
            virials.append(0.5 * (virial + virial.T))
    energy = torch.stack(energies)
    parts = torch.stack(parts)
    out = {"energy": energy, "node_energy": (energy / n).repeat_interleave(n)}
    out.update({f"e_{k}": parts[:, j] for j, k in enumerate(TERMS)})
    out["forces"] = torch.cat(forces).to(out_dtype)
    out["virial"] = torch.stack(virials).to(out_dtype)
    return out
