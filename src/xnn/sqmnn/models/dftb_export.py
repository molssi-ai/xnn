"""The scriptable core of exported DFTB models (see :func:`~xnn.common.deploy.export_torchscript_potential`).

:class:`DFTBExport` holds the Slater-Koster tables (with the corrections of their tail joins), the rotation
polynomials, the levels, Hubbard parameters and their derivatives, the cutoffs of the short-range gamma, the
repulsive potentials and the dispersion term of a model's elements, made once from the model, and evaluates one
molecule, or one cell periodic in three directions at the Gamma point, with the equations of
:mod:`~xnn.sqmnn.models.dftb`. Every atom pair and its images within reach enter (the neighbor list is not used).
Charges are per atom or per shell, with collinear spin, long-range exchange, the onsite correction and DFTB+U, and
cells may be sampled at k-points (without the density-dependent terms); shells up to f. Forces and stress follow
from automatic differentiation at the converged density.
"""
import copy
import math
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn

from xnn.common.models.eeq import ewald_alpha

from ..common.skf import COLUMNS, EXTENDED_COLUMNS
from .dftb import (
    BONDI_RADII,
    CLOSE_WIDTHS,
    EQUAL_WIDTHS,
    EWALD_SPLIT,
    HH_CUTOFF,
    HH_REPULSION,
    HYDROGEN_MASS,
    TAIL_LENGTH,
    _end_weights,
    _lagrange8,
    gamma_onsite,
    gamma_short,
    gamma_short_slope,
    gamma_yukawa,
    gamma_yukawa_onsite,
    short_range_cutoff,
)
from .slater_koster import monomial_powers, rotation_polynomials
from .tightbinding_export import TightBindingExport

_LETTERS = "spdf"


class _NoDispersion(nn.Module):
    """The dispersion term of a model without one."""

    @torch.jit.export
    def evaluate(self, atomic_numbers: Tensor, pos: Tensor, edge_index: Tensor, edge_vec: Tensor, batch: Tensor,
                 num_graphs: int, cell: Tensor, pbc: Tensor, total_charge: Tensor) -> Dict[str, Tensor]:
        return {"node_energy": torch.zeros_like(pos[:, 0])}


class DFTBExport(TightBindingExport):
    """``node_features_energy_charges`` of one molecule or cell (energies in eV, the Mulliken charges) for a
    :class:`~xnn.sqmnn.models.dftb.DFTB` model."""

    bohr: float
    autoev: float
    cutoff: float
    third: int
    has_damping: bool
    damping: float
    has_h5: bool
    r_scaling: float
    w_scaling: float
    hh: bool
    disp_bohr: float
    disp_hartree: float
    tail: float
    equal_widths: float
    close_widths: float
    shells: bool
    spin: bool
    density_mixing: bool
    has_omega: bool
    omega: float
    exchange_cutoff: float
    has_onsite: bool
    plus_u: int
    lmax_top: int
    has_k: bool
    copies: float
    sk_reach: float
    gamma_reach: float
    rep_reach: float
    disp_reach: float
    ewald_split: float
    ewald_real: float
    ewald_recip: float

    def __init__(self, model):
        super().__init__()
        self.density_mixing = bool(model.density_mixing)
        self.has_omega = model.omega is not None
        self.omega = float(model.omega) if model.omega is not None else 0.0
        self.exchange_cutoff = float(model.exchange_cutoff or 0.0)
        self.has_onsite = model.onsite is not None
        self.plus_u = 0 if model.plus_u is None else (1 if model.plus_u[0] == "fll" else 2)
        self.spin = bool(model.spin_polarized)
        self.shells = bool(model.shell_resolved)
        if self.shells and model.third_order == "full":
            raise NotImplementedError("the full third order needs one charge per atom")
        # k-points: the grid with -k merged into k, as the eager model (one Gamma point otherwise)
        self.has_k = model.kpoints is not None and math.prod(model.kpoints["grid"]) > 1
        if self.has_k and self.density_mixing:
            raise NotImplementedError("range separation, the onsite correction and DFTB+U need the Gamma point")
        kpts, kweight = torch.zeros(1, 3, dtype=torch.float64), torch.ones(1, dtype=torch.float64)
        if self.has_k:
            axes = [(torch.arange(n, dtype=torch.float64) + s) / n
                    for n, s in zip(model.kpoints["grid"], model.kpoints["shift"])]
            kpts, kweight = type(model)._time_reversal(torch.stack(torch.meshgrid(*axes, indexing="ij"), -1)
                                                       .reshape(-1, 3))
        self.copies = float(round(2.0 * float(kweight.sum()))) if self.has_k else 1.0
        t, cpu = torch.float64, torch.device("cpu")
        self.bohr, self.autoev = float(model.bohr), float(model.hartree)
        self.cutoff = float(model.cutoff)
        self.multipoles = False
        self.kt = max(model.boltzmann * model.electronic_temperature, model.minimum_kt)
        self.threshold = float(model.scc_tolerance)
        self.max_iter, self.mixing, self.history = int(model.max_iter), float(model.mixing), int(model.history)
        self.self_consistent = bool(model.scc)
        self.third = {None: 0, "onsite": 1, "full": 2}[model.third_order]
        self.has_damping = model.damping_exponent is not None
        self.damping = float(model.damping_exponent or 0.0)
        self.has_h5 = model.h5 is not None
        self.r_scaling = float(model.h5["r_scaling"]) if self.has_h5 else 0.0
        self.w_scaling = float(model.h5["w_scaling"]) if self.has_h5 else 0.0
        self.h5_width = 2.0 * math.sqrt(2.0 * math.log(2.0))
        self.hh = bool(model.hh_repulsion) if model.d3 is not None else False
        self.tail, self.equal_widths, self.close_widths = TAIL_LENGTH, EQUAL_WIDTHS, CLOSE_WIDTHS
        trained = set(model.trainable)
        elements = list(model.supported)
        size = max(elements) + 1
        lmax = torch.full((size,), -1, dtype=torch.long)
        n0, levels = torch.zeros(size, 4, dtype=t), torch.zeros(size, 4, dtype=t)
        hubbard, derivs = torch.zeros(size, 4, dtype=t), torch.zeros(size, dtype=t)
        hydrogen = torch.zeros(size, dtype=torch.bool)
        spin_w, has_spin = torch.zeros(size, 4, 4, dtype=t), torch.zeros(size, dtype=torch.bool)
        # the onsite correction: lambda of the shell pairs; DFTB+U: the block of each shell (-1: none) and its U - J
        lam = torch.zeros(size, 4, 4, dtype=t)
        u_block_id = torch.full((size, 4), -1, dtype=torch.long)
        u_value = torch.zeros(size, 4, dtype=t)
        for z in elements:
            if model.onsite is not None and z in model.onsite:
                n_l = model.lmax[z] + 1
                lam[z, :n_l, :n_l] = torch.as_tensor(model.onsite[z], dtype=t)
            if model.plus_u is not None:
                for k, (shells, uj) in enumerate(model.plus_u[1].get(z, [])):
                    for l in shells:
                        u_block_id[z, l] = k
                    u_value[z, k] = uj
        for z in elements:
            l = model.lmax[z]
            if l > 3:
                raise NotImplementedError("the DFTB export covers s, p, d and f shells")
            lmax[z] = l
            n0[z, :l + 1] = torch.tensor(model.reference_occupations(z), dtype=t)
            if "onsite" in trained:
                levels[z, :l + 1] = model.params["onsite"].detach().to(cpu, t)[z, :l + 1]
            else:
                levels[z, :l + 1] = torch.tensor([model._element(z).energies[k] for k in range(l + 1)], dtype=t)
            for k in range(l + 1):
                hubbard[z, k] = (float(model.params["hubbard"].detach()[z, k if self.shells else 0])
                                 if "hubbard" in trained else model.hubbard(z, k))
            derivs[z] = float(model.hubbard_derivatives.get(z, 0.0))
            hydrogen[z] = model._element(z).mass < HYDROGEN_MASS
            if model.spin_polarized and z in model.spin_constants:
                table = model.spin_constants[z]
                n_sh = l + 1
                has_spin[z] = True
                for a in range(n_sh):
                    for b in range(n_sh):
                        spin_w[z, a, b] = float(table[a * n_sh + b]) if table.size > 1 else float(table[0])
        # the Slater-Koster tables of every pair of elements, rows padded to the longest table
        names = EXTENDED_COLUMNS if any(model.skset.has(a, b) and model.skset.pair(a, b).extended
                                        for a in elements for b in elements) else COLUMNS
        pair_id = torch.full((size, size), -1, dtype=torch.long)
        gamma_cut = torch.zeros(size, 4, size, 4, dtype=t)          # per pair of charge sites (shells or atoms)
        tables_h, tables_s, ends_h, ends_s, grids, rows = [], [], [], [], [], []
        for za in elements:
            for zb in elements:
                for ka in range(model.lmax[za] + 1 if self.shells else 1):
                    for kb in range(model.lmax[zb] + 1 if self.shells else 1):
                        gamma_cut[za, ka, zb, kb] = short_range_cutoff(3.2 * float(hubbard[za, ka]),
                                                                       3.2 * float(hubbard[zb, kb]))
                if not model.skset.has(za, zb):
                    continue
                cols = model.skset.pair(za, zb).columns
                pick = [cols.index(c) if c in cols else -1 for c in names]
                pair_id[za, zb] = len(grids)
                for which, tabs, ends in (("H", tables_h, ends_h), ("S", tables_s, ends_s)):
                    table, grid, end = model._table(za, zb, which, t, cpu)
                    tabs.append(torch.stack([table.detach()[:, k] if k >= 0 else torch.zeros_like(table[:, 0])
                                             for k in pick], 1))
                    ends.append(torch.stack([end.detach()[:, k] if k >= 0 else torch.zeros_like(end[:, 0])
                                             for k in pick], 1))
                grids.append(grid)
                rows.append(tables_h[-1].shape[0])
        longest = max(rows)
        pad = lambda tabs: torch.stack([torch.nn.functional.pad(x, (0, 0, 0, longest - x.shape[0]))  # noqa: E731
                                        for x in tabs])
        n_rows = torch.tensor(rows, dtype=torch.long)
        grid_t = torch.tensor(grids, dtype=t)
        # the integral columns of a shell pair: index in the columns of (lower l, higher l, |m|)
        column = torch.zeros(4, 4, 4, dtype=torch.long)
        for lo in range(4):
            for hi in range(lo, 4):
                for k in range(lo + 1):
                    name = f"{_LETTERS[lo]}{_LETTERS[hi]}{k}"
                    if name in names:
                        column[lo, hi, k] = names.index(name)
        # rotation polynomials of every shell pair in the monomials of their degree
        coef = torch.zeros(4, 4, 28, 4, 7, 7, dtype=t)
        powers = torch.zeros(7, 28, 3, dtype=torch.long)
        n_mono = torch.zeros(7, dtype=torch.long)
        for n in range(7):
            p = torch.as_tensor(monomial_powers(n))
            powers[n, :p.shape[0]] = p
            n_mono[n] = p.shape[0]
        self.lmax_top = int(lmax.max())
        for la in range(self.lmax_top + 1):
            for lb in range(self.lmax_top + 1):
                c = torch.as_tensor(rotation_polynomials(la, lb), dtype=t)
                coef[la, lb, :c.shape[0], :c.shape[1], :c.shape[2], :c.shape[3]] = c
        # repulsive potentials per pair of elements: splines (padded segments) or polynomials
        rep_id = torch.full((size, size), -1, dtype=torch.long)
        keys = sorted({(min(a, b), max(a, b)) for a in elements for b in elements
                       if model.skset.has(min(a, b), max(a, b))})
        n_seg = max([len(model.skset.pair(*k).spline.coefficients) for k in keys
                     if model.skset.pair(*k).spline is not None] + [1])
        starts = torch.full((len(keys), n_seg), float("inf"), dtype=t)
        seg_coef = torch.zeros(len(keys), n_seg, 6, dtype=t)
        heads = torch.zeros(len(keys), 3, dtype=t)
        rep_cut = torch.zeros(len(keys), dtype=t)
        segments = torch.ones(len(keys), dtype=torch.long)
        is_spline = torch.zeros(len(keys), dtype=torch.bool)
        poly = torch.zeros(len(keys), 8, dtype=t)
        for k, (za, zb) in enumerate(keys):
            rep_id[za, zb] = rep_id[zb, za] = k
            f = model.skset.pair(za, zb)
            if f.spline is None:
                rep_cut[k] = f.polynomial_cutoff
                poly[k, :len(f.polynomial)] = torch.tensor(f.polynomial, dtype=t)
                continue
            key = f"{za}_{zb}"
            is_spline[k] = True
            rep_cut[k] = f.spline.cutoff
            m = len(f.spline.coefficients)
            segments[k] = m
            starts[k, :m] = torch.as_tensor(f.spline.knots[:-1], dtype=t)
            if "repulsive" in trained:
                seg_coef[k, :m] = model.params[f"spline_{key}"].detach().to(cpu, t)
                heads[k] = model.params[f"head_{key}"].detach().to(cpu, t)
            else:
                seg_coef[k, :m] = torch.as_tensor(f.spline.coefficients, dtype=t)
                heads[k] = torch.tensor(f.spline.head, dtype=t)
        h5_k, bondi = torch.zeros(size, dtype=t), torch.zeros(size, dtype=t)
        for z, r in BONDI_RADII.items():
            if z < size:
                bondi[z] = r
        if self.has_h5:
            for z, k in model.h5["scaling"].items():
                if z < size:
                    h5_k[z] = k
        buffers = {"lmax": lmax, "n0": n0, "levels": levels, "hubbard": hubbard, "derivs": derivs,
                   "spin_w": spin_w, "has_spin": has_spin,
                   "lam": lam, "u_block_id": u_block_id, "u_value": u_value,
                   "kpts": kpts, "kweight": kweight,
                   "hydrogen": hydrogen, "pair_id": pair_id, "gamma_cut": gamma_cut, "tables_h": pad(tables_h),
                   "tables_s": pad(tables_s), "ends_h": torch.stack(ends_h), "ends_s": torch.stack(ends_s),
                   "grid": grid_t, "n_rows": n_rows, "column": column, "coef": coef, "powers": powers,
                   "n_mono": n_mono, "end_weights": torch.as_tensor(_end_weights(), dtype=t), "rep_id": rep_id,
                   "starts": starts, "seg_coef": seg_coef, "heads": heads, "rep_cut": rep_cut, "segments": segments,
                   "is_spline": is_spline, "poly": poly, "h5_k": h5_k, "bondi": bondi}
        device = next(iter(model.parameters()), torch.zeros(0)).device
        for name, value in buffers.items():
            self.register_buffer(name, value.to(device))
        self.precision = nn.Parameter(torch.zeros(0, dtype=t, device=device), requires_grad=False)
        if model.d3 is not None:
            self.dispersion = copy.deepcopy(model.d3).to(device)
            self.disp_bohr, self.disp_hartree = float(model.d3.bohr), float(model.d3.hartree)
        else:
            self.dispersion = _NoDispersion()
            self.disp_bohr, self.disp_hartree = 1.0, 1.0
        self.hh_height, self.hh_steepness, self.hh_middle = HH_REPULSION
        self.hh_cutoff = float(HH_CUTOFF)
        # periodic cells: the reach of every pair term (bohr) and the Ewald settings of dftb.ewald_matrix
        self.sk_reach = float(max(n * g for n, g in zip(rows, grids))) + TAIL_LENGTH
        self.gamma_reach = float(gamma_cut.max())
        self.rep_reach = float(rep_cut.max())
        self.disp_reach = 0.0
        if model.d3 is not None:
            cuts = [getattr(model.d3, k, 0.0) for k in ("cutoff_pair", "cutoff_triple", "cutoff_cn", "cutoff_eeq_cn")]
            self.disp_reach = float(max(cuts)) / float(model.d3.bohr)
        self.ewald_split = float(EWALD_SPLIT)
        self.ewald_real = float(ewald_alpha(1.0, 1e-16))              # alpha times the real-space reach
        self.ewald_recip = 2.0 * math.sqrt(-math.log(1e-16))          # the reciprocal reach over alpha

    def residual(self, r: Tensor) -> float:
        return float(r.abs().max())

    def table(self, tables: Tensor, ends: Tensor, pid: Tensor, r: Tensor) -> Tensor:
        """Interpolated table columns ``(P, C)`` of the pairs ``pid`` at the distances ``r`` (bohr), as
        :func:`~xnn.sqmnn.models.dftb.table_values`."""
        grid, n = self.grid[pid], self.n_rows[pid]
        x = r / grid - 1.0
        i0 = torch.minimum((torch.floor(x.detach()) - 3).clamp(min=0).long(), n - 8)
        t = x - i0.to(x.dtype)
        rows = i0[:, None] + torch.arange(8, device=r.device)[None, :]
        inner = (_lagrange8(t)[..., None] * tables[pid[:, None], rows]).sum(1)
        last = (n - 8)[:, None] + torch.arange(8, device=r.device)[None, :]
        f = torch.matmul(self.end_weights[None], tables[pid[:, None], last])        # (P, 3, C) in rows
        scale = (self.tail / grid)[:, None]
        f0 = f[:, 0] + ends[pid, 0]
        f1 = f[:, 1] * scale + ends[pid, 1] * self.tail
        f2 = f[:, 2] * scale ** 2 + ends[pid, 2] * self.tail ** 2
        c3 = -10.0 * f0 - 6.0 * f1 - 1.5 * f2
        c4 = 15.0 * f0 + 8.0 * f1 + 1.5 * f2
        c5 = -6.0 * f0 - 3.0 * f1 - 0.5 * f2
        r0 = (n.to(r.dtype) * grid)[:, None]
        rr = r[:, None]
        s = ((rr - r0) / self.tail).clamp(0.0, 1.0)
        outer = f0 + s * (f1 + s * (0.5 * f2 + s * (c3 + s * (c4 + s * c5))))
        return torch.where(rr < r0, inner, torch.where(rr < r0 + self.tail, outer, torch.zeros_like(outer)))

    def block(self, u: Tensor, v: Tensor, la: int, lb: int) -> Tensor:
        """Two-center blocks ``(P, 2 la + 1, 2 lb + 1)`` from the unit vectors and bond-frame integrals, as
        :func:`~xnn.sqmnn.models.slater_koster.shell_block`."""
        n = la + lb
        m = int(self.n_mono[n])
        k = min(la, lb) + 1
        pw = [torch.ones_like(u)]
        for _ in range(n):
            pw.append(pw[-1] * u)
        stacked = torch.stack(pw, -2)
        p = self.powers[n, :m]
        if n == 0:
            mono = torch.ones(u.shape[0], 1, dtype=u.dtype, device=u.device)
        else:
            mono = stacked[:, p[:, 0], 0] * stacked[:, p[:, 1], 1] * stacked[:, p[:, 2], 2]
        coef = self.coef[la, lb, :m, :k, :2 * la + 1, :2 * lb + 1]
        angular = torch.matmul(mono, coef.reshape(m, -1)).reshape(-1, k, 2 * la + 1, 2 * lb + 1)
        return (angular * v[:, :, None, None]).sum(1)

    def two_center(self, z: Tensor, i: Tensor, j: Tensor, vec: Tensor, r: Tensor, atom_first: Tensor, once: Tensor,
                   nb: int, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Overlap and Hamiltonian ``(K, nb, nb)`` of the atom pairs ``i < j`` and their images (without the
        diagonal blocks), each entry weighted by the real and imaginary parts ``cos``, ``sin`` ``(E, K)`` of its
        phase at the ``K`` k-points (one k-point of phase one at the Gamma point): the real parts of ``S`` and
        ``H``, then their imaginary parts (zero when ``sin`` has no column). An atom's own images (``once``) fill its
        block once, the other pairs both blocks (the transposed ones with the conjugate phase)."""
        n_k = cos.shape[1]
        S = torch.zeros(n_k, nb, nb, dtype=vec.dtype, device=vec.device)
        H = torch.zeros(n_k, nb, nb, dtype=vec.dtype, device=vec.device)
        S_im, H_im = torch.zeros_like(S), torch.zeros_like(H)
        imaginary = sin.shape[1] > 0
        za, zb = z[i], z[j]
        ab, ba = self.pair_id[za, zb], self.pair_id[zb, za]
        inside = r < self.n_rows[ab].to(r.dtype) * self.grid[ab] + self.tail
        ii, jj, rr, ab, ba = i[inside], j[inside], r[inside], ab[inside], ba[inside]
        mirror = ~once[inside]
        c_in, s_in = cos[inside], sin[inside]
        u = vec[inside] / rr[:, None]
        h_ab, s_ab = self.table(self.tables_h, self.ends_h, ab, rr), self.table(self.tables_s, self.ends_s, ab, rr)
        h_ba, s_ba = self.table(self.tables_h, self.ends_h, ba, rr), self.table(self.tables_s, self.ends_s, ba, rr)
        la_max, lb_max = self.lmax[z[ii]], self.lmax[z[jj]]
        kk = torch.arange(n_k, device=u.device)
        for la in range(self.lmax_top + 1):
            for lb in range(self.lmax_top + 1):
                sel = (la_max >= la) & (lb_max >= lb)
                if not bool(sel.any()):
                    continue
                lo, hi = min(la, lb), max(la, lb)
                idx = self.column[lo, hi, :lo + 1]
                ra = (atom_first[ii[sel]] + la * la)[:, None, None] + torch.arange(
                    2 * la + 1, device=u.device)[None, :, None]
                cb = (atom_first[jj[sel]] + lb * lb)[:, None, None] + torch.arange(
                    2 * lb + 1, device=u.device)[None, None, :]
                ra, cb = torch.broadcast_tensors(ra, cb)
                shape = [n_k, ra.shape[0], ra.shape[1], ra.shape[2]]
                kr = kk[:, None, None, None].expand(shape).reshape(-1)
                rk, ck = ra[None].expand(shape).reshape(-1), cb[None].expand(shape).reshape(-1)
                mt = mirror[sel][None, :, None, None].expand(shape).reshape(-1)
                ph_c = c_in[sel].t()[:, :, None, None]
                for which in range(2):
                    if la <= lb:
                        v = (h_ab if which == 0 else s_ab)[sel][:, idx]
                    else:
                        v = (h_ba if which == 0 else s_ba)[sel][:, idx] * float((-1) ** (la + lb))
                    blk = self.block(u[sel], v, la, lb)[None]
                    re = (blk * ph_c).reshape(-1)
                    M = H if which == 0 else S
                    M = M.index_put((kr, rk, ck), re, accumulate=True)
                    M = M.index_put((kr[mt], ck[mt], rk[mt]), re[mt], accumulate=True)
                    if imaginary:
                        im = (blk * s_in[sel].t()[:, :, None, None]).reshape(-1)
                        Mi = H_im if which == 0 else S_im
                        Mi = Mi.index_put((kr, rk, ck), im, accumulate=True)
                        Mi = Mi.index_put((kr[mt], ck[mt], rk[mt]), -im[mt], accumulate=True)
                        if which == 0:
                            H_im = Mi
                        else:
                            S_im = Mi
                    if which == 0:
                        H = M
                    else:
                        S = M
        return S, S_im, H, H_im

    def gamma_terms(self, z: Tensor, x: Tensor, site_atom: Tensor, site_l: Tensor, shell_site: Tensor) -> List[Tensor]:
        """Gamma between the charge sites (atoms, or shells when resolved) of a molecule, their onsite gamma and the
        Gammas of the third order, as :meth:`~xnn.sqmnn.models.dftb.DFTB.scc_context`; the last entries are the
        site pairs and the site of every shell."""
        n_sites = site_atom.shape[0]
        iu = torch.triu_indices(n_sites, n_sites, 1, device=x.device)
        si, sj = iu[0], iu[1]
        ai, aj = site_atom[si], site_atom[sj]
        za, zb, la, lb = z[ai], z[aj], site_l[si], site_l[sj]
        ua, ub = self.hubbard[za, la], self.hubbard[zb, lb]
        same = ai == aj
        d = torch.where(same[:, None], torch.ones_like(x[ai]), x[aj] - x[ai])     # shells of one atom: no distance
        r = torch.linalg.norm(d, dim=-1)
        inside = r < self.gamma_cut[za, la, zb, lb]
        short = gamma_short(3.2 * ua, 3.2 * ub, r, self.equal_widths, self.close_widths)
        short = torch.where(inside, short, torch.zeros_like(short))
        h = torch.ones_like(r)
        damp = self.hydrogen[za] | self.hydrogen[zb]
        if self.has_damping:
            h = torch.where(damp, torch.exp(-(0.5 * (ua + ub)) ** self.damping * r * r), h)
        f = self.h5_factor(za, zb, r)
        gamma = torch.where(same, gamma_onsite(3.2 * ua, 3.2 * ub), (1.0 / r - short * h) * f)
        onsite = self.hubbard[z[site_atom], site_l]
        gab, gba = torch.zeros_like(r), torch.zeros_like(r)
        gons = 0.5 * self.derivs[z[site_atom]] if self.third > 0 else torch.zeros_like(onsite)
        if self.third == 2:
            # derivatives of S in the Hubbard parameters (tau = 3.2 U)
            sa = torch.where(inside, 3.2 * gamma_short_slope(3.2 * ua, 3.2 * ub, r, self.equal_widths),
                             torch.zeros_like(r))
            sb = torch.where(inside, 3.2 * gamma_short_slope(3.2 * ub, 3.2 * ua, r, self.equal_widths),
                             torch.zeros_like(r))
            both = torch.where(3.2 * (ua - ub).abs() < self.equal_widths, 2.0, 1.0).to(r.dtype)
            dh = torch.zeros_like(r)
            if self.has_damping:
                slope = -0.5 * self.damping * (0.5 * (ua + ub)) ** (self.damping - 1.0) * r * r
                dh = torch.where(damp, slope, torch.zeros_like(slope)) * h
            gab = -(both * sa * h + short * dh) * self.derivs[za]
            gba = -(both * sb * h + short * dh) * self.derivs[zb]
        return [gamma, onsite, gab, gba, gons, si, sj, shell_site]

    def potentials(self, q_shell: Tensor, mu: Tensor, theta: Tensor, shell_atom: Tensor,
                   ctx: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        if not self.self_consistent:
            return torch.zeros_like(q_shell), mu, theta
        gamma, onsite, gab, gba, gons, i, j, shell_site = ctx[0], ctx[1], ctx[2], ctx[3], ctx[4], ctx[5], ctx[6], ctx[7]
        q = torch.zeros_like(onsite).index_add(0, shell_site, q_shell)
        qi, qj = q[i], q[j]
        v = onsite * q
        v = v.index_add(0, i, gamma * qj).index_add(0, j, gamma * qi)
        if self.third > 0:
            v = v - gons * q * q
            if self.third == 2:
                v = v.index_add(0, i, -(2.0 * gab * qi * qj + gba * qj * qj) / 3.0)
                v = v.index_add(0, j, -(gab * qi * qi + 2.0 * gba * qi * qj) / 3.0)
        return v[shell_site], mu, theta

    def charge_energy(self, q_shell: Tensor, shell_atom: Tensor, ctx: List[Tensor]) -> Tensor:
        """The second- and third-order energies."""
        if not self.self_consistent:
            return torch.zeros((), dtype=q_shell.dtype, device=q_shell.device)
        gamma, onsite, gab, gba, gons, i, j, shell_site = ctx[0], ctx[1], ctx[2], ctx[3], ctx[4], ctx[5], ctx[6], ctx[7]
        q = torch.zeros_like(onsite).index_add(0, shell_site, q_shell)
        qi, qj = q[i], q[j]
        e = (gamma * qi * qj).sum() + (0.5 * onsite * q * q).sum()
        if self.third > 0:
            e = e + (-gons * q ** 3 / 3.0).sum()
            if self.third == 2:
                e = e + (-(gab * qi * qi * qj + gba * qj * qj * qi) / 3.0).sum()
        return e

    def repulsion(self, z: Tensor, i: Tensor, j: Tensor, r: Tensor, weight: Tensor) -> Tensor:
        """The repulsive energy (hartree) of the pairs, each with its ``weight``, as
        :meth:`~xnn.sqmnn.models.dftb.DFTB.repulsive`."""
        k = self.rep_id[z[i], z[j]]
        st = self.starts[k]
        seg = torch.searchsorted(st, r[:, None].contiguous(), right=True).squeeze(1) - 1
        seg = torch.minimum(seg.clamp(min=0), self.segments[k] - 1)
        t = r - st.gather(1, seg[:, None]).squeeze(1)
        c = self.seg_coef[k, seg]
        spline = c[:, 0] + t * (c[:, 1] + t * (c[:, 2] + t * (c[:, 3] + t * (c[:, 4] + t * c[:, 5]))))
        head = self.heads[k]
        spline = torch.where(r < st[:, 0], torch.exp(-head[:, 0] * r + head[:, 1]) + head[:, 2], spline)
        x = (self.rep_cut[k] - r).clamp_min(0.0)
        polynomial = torch.zeros_like(r)
        for p in range(8):
            polynomial = polynomial + self.poly[k, p] * x ** (p + 2)
        value = torch.where(self.is_spline[k], spline, polynomial)
        return (torch.where(r < self.rep_cut[k], value, torch.zeros_like(r)) * weight).sum()

    def h5_factor(self, za: Tensor, zb: Tensor, r: Tensor) -> Tensor:
        """The H5 factor of gamma (one without the correction)."""
        f = torch.ones_like(r)
        if self.has_h5:
            pair = (za == 1) ^ (zb == 1)
            other = torch.where(za == 1, zb, za)
            k = torch.where(pair, self.h5_k[other], torch.zeros_like(r))
            radius = torch.where(pair, self.bondi[other] + self.bondi[1], torch.zeros_like(r))
            w = self.w_scaling * radius / self.h5_width
            w = torch.where(k > 0.0, w, torch.ones_like(w))
            x = r * self.bohr - self.r_scaling * radius
            f = 1.0 + k * torch.exp(-x * x / (2.0 * w * w))
        return f

    def translations(self, cell: Tensor, reach: float) -> Tensor:
        """Integer translations whose lattice vectors cover every distance up to ``reach`` (bohr), as
        :func:`~xnn.sqmnn.models.dftb.lattice_translations` in three periodic directions."""
        c = cell.detach()
        volume = float(torch.det(c).abs())
        axes: List[Tensor] = []
        for k in range(3):
            height = volume / float(torch.linalg.norm(torch.linalg.cross(c[(k + 1) % 3], c[(k + 2) % 3])))
            n = int(math.ceil(reach / height))
            axes.append(torch.arange(-n, n + 1, device=c.device))
        grid = torch.meshgrid(axes, indexing="ij")
        return torch.stack([grid[0].reshape(-1), grid[1].reshape(-1), grid[2].reshape(-1)], 1)

    def image_pairs(self, x: Tensor, cell: Tensor, reach: float) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Atom pairs ``i < j`` with every lattice translation and an atom's own images (``i = j``, non-zero
        translations) within ``reach`` (bohr): ``i``, ``j``, the vectors from ``i`` to ``j`` and the flag of an atom's
        own images, as :meth:`~xnn.sqmnn.models.dftb.DFTB._pair_entries`."""
        n = x.shape[0]
        idx = torch.arange(n, device=x.device)
        ii, jj = idx.repeat_interleave(n), idx.repeat(n)
        d0 = x[jj] - x[ii]
        # the nearest image of every pair, so that few translations reach every image within reach
        d0 = d0 - torch.round(torch.linalg.solve(cell.detach().t(), d0.detach().t()).t()) @ cell
        extent = float(torch.linalg.norm(d0.detach(), dim=-1).max())
        n_t = self.translations(cell, reach + extent)
        d = d0[:, None, :] + (n_t.to(x.dtype) @ cell)[None, :, :]
        r = torch.linalg.norm(d.detach(), dim=-1)
        zero = (n_t == 0).all(-1)
        mask = (r < reach) & ((ii < jj)[:, None] | ((ii == jj)[:, None] & ~zero[None, :]))
        nz = torch.nonzero(mask)
        pair, t = nz[:, 0], nz[:, 1]
        return ii[pair], jj[pair], d[pair, t], (ii == jj)[pair]

    def ewald(self, x: Tensor, cell: Tensor) -> Tensor:
        """The lattice sum of ``1 / |r_j - r_i + T|`` in a neutralizing background ``(N, N)``, as
        :func:`~xnn.sqmnn.models.dftb.ewald_matrix`."""
        n = x.shape[0]
        volume = torch.det(cell).abs()
        alpha = self.ewald_split / float(volume.detach()) ** (1.0 / 3.0)
        reach = self.ewald_real / alpha
        d0 = x[None, :, :] - x[:, None, :]
        frac = torch.linalg.solve(cell.detach().t(), d0.detach().reshape(-1, 3).t()).t().reshape(n, n, 3)
        d0 = d0 - torch.round(frac) @ cell
        extent = float(torch.linalg.norm(d0.detach(), dim=-1).max()) if n > 1 else 0.0
        shifts = self.translations(cell, reach + extent).to(x.dtype) @ cell
        v = d0[:, :, None, :] + shifts[None, None, :, :]
        r = torch.linalg.norm(v.detach(), dim=-1)
        keep = (r > 0.0) & (r <= reach)
        rs = torch.linalg.norm(torch.where(keep[..., None], v, torch.ones_like(v)), dim=-1)
        real = torch.where(keep, torch.erfc(alpha * rs) / rs, torch.zeros_like(r)).sum(-1)
        # the reciprocal sum over G != 0 within its reach, by structure factors
        recip = 2.0 * math.pi * torch.linalg.inv(cell).t()
        g_max = alpha * self.ewald_recip
        lengths = torch.linalg.norm(cell.detach(), dim=1)
        axes: List[Tensor] = []
        for k in range(3):
            m = int(math.ceil(g_max * float(lengths[k]) / (2.0 * math.pi)))
            axes.append(torch.arange(-m, m + 1, device=x.device))
        grid = torch.meshgrid(axes, indexing="ij")
        mvec = torch.stack([grid[0].reshape(-1), grid[1].reshape(-1), grid[2].reshape(-1)], 1).to(x.dtype)
        gvec = mvec @ recip
        g2 = (gvec * gvec).sum(-1)
        inside = (g2.detach() > 0.0) & (g2.detach() <= g_max * g_max)
        gvec, g2 = gvec[inside], g2[inside]
        g = 4.0 * math.pi / volume * torch.exp(-0.25 * g2 / (alpha * alpha)) / g2
        phase = x @ gvec.t()
        c, s = torch.cos(phase), torch.sin(phase)
        rec = (c * g) @ c.t() + (s * g) @ s.t()
        eye = torch.eye(n, dtype=x.dtype, device=x.device)
        return real + rec - math.pi / (alpha * alpha * volume) - 2.0 * alpha / math.sqrt(math.pi) * eye

    def periodic_gamma(self, z: Tensor, x: Tensor, cell: Tensor, site_atom: Tensor, site_l: Tensor,
                       shell_site: Tensor) -> List[Tensor]:
        """The terms of :meth:`gamma_terms` in a periodic cell, as
        :meth:`~xnn.sqmnn.models.dftb.DFTB._periodic_gamma`: the Ewald sum of ``1 / R`` between the atoms of the sites
        plus the short-range rest summed over images, on every pair of sites, and the onsite terms with the images of
        a site's atom."""
        n, n_sites = z.shape[0], site_atom.shape[0]
        device = x.device
        G = self.ewald(x, cell)[site_atom][:, site_atom]
        i, j, vec, itself = self.image_pairs(x, cell, self.gamma_reach)
        r = torch.linalg.norm(vec, dim=-1)
        # every pair of sites of the two atoms of an entry
        count = torch.zeros(n, dtype=torch.long, device=device).index_add(0, site_atom, torch.ones_like(site_atom))
        width = int(count.max())
        first = torch.cumsum(count, 0) - count
        rank = torch.arange(n_sites, device=device) - first[site_atom]
        table = torch.full((n, width), -1, dtype=torch.long, device=device)
        table = table.index_put((site_atom, rank), torch.arange(n_sites, device=device))
        s = table[i][:, :, None].expand(-1, width, width)
        t = table[j][:, None, :].expand(-1, width, width)
        valid = (s >= 0) & (t >= 0)
        entry = torch.arange(i.shape[0], device=device)[:, None, None].expand(-1, width, width)[valid]
        s, t = s[valid], t[valid]
        ai, aj, own, rr = i[entry], j[entry], itself[entry], r[entry]
        za, zb, la, lb = z[ai], z[aj], site_l[s], site_l[t]
        within = rr < self.gamma_cut[za, la, zb, lb]
        s, t, own, rr = s[within], t[within], own[within], rr[within]
        za, zb, la, lb = za[within], zb[within], la[within], lb[within]
        ua, ub = self.hubbard[za, la], self.hubbard[zb, lb]
        short = gamma_short(3.2 * ua, 3.2 * ub, rr, self.equal_widths, self.close_widths)
        h = torch.ones_like(rr)
        damp = self.hydrogen[za] | self.hydrogen[zb]
        if self.has_damping:
            h = torch.where(damp, torch.exp(-(0.5 * (ua + ub)) ** self.damping * rr * rr), h)
        f = self.h5_factor(za, zb, rr)
        G = G.index_put((s, t), -short * h * f + (f - 1.0) / rr, accumulate=True)
        iu = torch.triu_indices(n_sites, n_sites, 1, device=device)
        si, sj = iu[0], iu[1]
        u = self.hubbard[z[site_atom], site_l]
        same = site_atom[si] == site_atom[sj]
        gamma = G[si, sj] + torch.where(same, gamma_onsite(3.2 * u[si], 3.2 * u[sj]), torch.zeros_like(G[si, sj]))
        onsite = u + torch.diagonal(G)
        gons = 0.5 * self.derivs[z[site_atom]] if self.third > 0 else torch.zeros_like(onsite)
        gab, gba = torch.zeros_like(gamma), torch.zeros_like(gamma)
        if self.third == 2:
            sa = 3.2 * gamma_short_slope(3.2 * ua, 3.2 * ub, rr, self.equal_widths)
            sb = 3.2 * gamma_short_slope(3.2 * ub, 3.2 * ua, rr, self.equal_widths)
            both = torch.where(3.2 * (ua - ub).abs() < self.equal_widths, 2.0, 1.0).to(rr.dtype)
            dh = torch.zeros_like(rr)
            if self.has_damping:
                slope = -0.5 * self.damping * (0.5 * (ua + ub)) ** (self.damping - 1.0) * rr * rr
                dh = torch.where(damp, slope, torch.zeros_like(slope)) * h
            ga = -(both * sa * h + short * dh) * self.derivs[za]
            gb = -(both * sb * h + short * dh) * self.derivs[zb]
            pair = ~own
            zeros = torch.zeros(n_sites, n_sites, dtype=x.dtype, device=device)
            gab = zeros.index_put((s[pair], t[pair]), ga[pair], accumulate=True)[si, sj]
            gba = zeros.index_put((s[pair], t[pair]), gb[pair], accumulate=True)[si, sj]
            gons = gons + torch.zeros_like(onsite).index_add(0, s[own], ga[own])
        return [gamma, onsite, gab, gba, gons, si, sj, shell_site]

    def density_context(self, z: Tensor, x: Tensor, cell: Tensor, periodic: bool, n0: Tensor, S: Tensor,
                        ao_atom: Tensor, ao_shell: Tensor, shell_atom: Tensor, sl: Tensor) -> List[Tensor]:
        """What the density-dependent terms need, as :meth:`~xnn.sqmnn.models.dftb.DFTB._exchange_context`,
        ``_onsite_context`` and ``_plus_u_context``: the reference density, the overlap, the long-range gamma
        between the orbitals, the onsite weights of the orbital pairs, the onsite coefficient and width of every
        shell, and ``U - J`` on the orbital pairs of the blocks and on their diagonals."""
        nb = ao_atom.shape[0]
        n = z.shape[0]
        device = x.device
        l_ao = sl[ao_shell]
        p0 = torch.diag(n0[ao_shell] / (2.0 * l_ao.to(x.dtype) + 1.0))
        z_ao = z[ao_atom]
        same_atom = ao_atom[:, None] == ao_atom[None, :]
        lr = torch.zeros(nb, nb, dtype=x.dtype, device=device)
        if self.has_omega:
            tau = 3.2 * self.hubbard[z, 0]
            if periodic:
                faces = torch.linalg.norm(torch.linalg.cross(cell.roll(-1, dims=0), cell.roll(-2, dims=0)), dim=-1)
                inscribed = 0.5 * float((torch.det(cell.detach()).abs() / faces.detach()).min())
                cut = self.exchange_cutoff if self.exchange_cutoff > 0.0 else inscribed
                i, j, vec, itself = self.image_pairs(x, cell, cut)
            else:
                iu = torch.triu_indices(n, n, 1, device=device)
                i, j = iu[0], iu[1]
                vec = x[j] - x[i]
                itself = torch.zeros_like(i, dtype=torch.bool)
            r = torch.linalg.norm(vec, dim=-1)
            ti, tj = tau[i], tau[j]
            pair = 1.0 / r - gamma_short(ti, tj, r, self.equal_widths, self.close_widths) \
                - gamma_yukawa(ti, tj, r, self.omega)
            other = ~itself
            G = torch.zeros(n, n, dtype=x.dtype, device=device)
            G = G.index_put((i[other], j[other]), pair[other], accumulate=True)
            G = G.index_put((j[other], i[other]), pair[other], accumulate=True)
            G = G.index_put((i[itself], i[itself]), pair[itself], accumulate=True)
            idx = torch.arange(n, device=device)
            G = G.index_put((idx, idx), 5.0 * tau / 16.0 - gamma_yukawa_onsite(tau, self.omega), accumulate=True)
            lr = G[ao_atom][:, ao_atom]
        w_ons = torch.zeros(nb, nb, dtype=x.dtype, device=device)
        c_shell = torch.zeros(sl.shape[0], dtype=x.dtype, device=device)
        if self.has_onsite:
            off = same_atom & ~torch.eye(nb, dtype=torch.bool, device=device)
            w_ons = torch.where(off, self.lam[z_ao[:, None], l_ao[:, None], l_ao[None, :]], w_ons)
            zs = z[shell_atom]
            c_shell = torch.where(sl > 0, self.lam[zs, sl, sl] / (2.0 * sl.to(x.dtype) + 1.0), c_shell)
        width = 2.0 * sl.to(x.dtype) + 1.0
        u_block = torch.zeros(nb, nb, dtype=x.dtype, device=device)
        u_diag = torch.zeros(nb, dtype=x.dtype, device=device)
        if self.plus_u > 0:
            bid = self.u_block_id[z_ao, l_ao]
            uj = torch.where(bid >= 0, self.u_value[z_ao, bid.clamp(min=0)], torch.zeros_like(u_diag))
            in_block = same_atom & (bid[:, None] == bid[None, :]) & (bid[:, None] >= 0)
            u_block = torch.where(in_block, uj[:, None].expand(nb, nb), u_block)
            u_diag = uj
        return [p0, S, lr, w_ons, c_shell, width, u_block, u_diag]

    def exchange_fock(self, dP: Tensor, S: Tensor, G: Tensor) -> Tensor:
        """Long-range exchange in matrix form, as :meth:`~xnn.sqmnn.models.dftb.DFTB._exchange_fock`."""
        SP, PS = S @ dP, dP @ S
        return -0.125 * ((SP * G) @ S + G * (SP @ S) + S @ (dP * G) @ S + S @ (PS * G))

    def onsite_terms(self, P: Tensor, S: Tensor, dens: List[Tensor], ao_shell: Tensor) -> Tuple[Tensor, Tensor]:
        """Energy and Fock term of the onsite correction, as :meth:`~xnn.sqmnn.models.dftb.DFTB._onsite`."""
        p0, w_ons, c_shell, width = dens[0], dens[3], dens[4], dens[5]
        Q = 0.5 * (P @ S + S @ P) - p0
        diag = torch.diagonal(Q)
        sum_x = torch.zeros_like(c_shell).index_add(0, ao_shell, diag)
        sum_x2 = torch.zeros_like(c_shell).index_add(0, ao_shell, diag * diag)
        energy = (w_ons * Q * Q).sum() + (c_shell * (width * sum_x2 - sum_x * sum_x)).sum()
        g_diag = 2.0 * c_shell[ao_shell] * (width[ao_shell] * diag - sum_x[ao_shell])
        G = 2.0 * w_ons * Q + torch.diag(g_diag)
        return energy, 0.5 * (G @ S + S @ G)

    def plus_u_terms(self, P: Tensor, S: Tensor, dens: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """Energy and Fock term of DFTB+U of one spin channel, as
        :meth:`~xnn.sqmnn.models.dftb.DFTB._orbital_potential`."""
        u_block, u_diag = dens[6], dens[7]
        n = 0.5 * (P @ S + S @ P)
        if self.plus_u == 1:
            energy = 0.5 * ((u_diag * torch.diagonal(n)).sum() - (u_block * n * n).sum())
            G = torch.diag(0.5 * u_diag) - u_block * n
        else:
            energy = -0.5 * (u_block * n * n).sum()
            G = -u_block * n
        return energy, 0.5 * (G @ S + S @ G)

    def density_fock(self, P: Tensor, dens: List[Tensor], ao_shell: Tensor) -> Tensor:
        """The density-dependent Fock terms of the channel densities ``P`` ``(C, nb, nb)``."""
        S = dens[1]
        out = torch.zeros_like(P)
        if self.plus_u > 0:
            for c in range(P.shape[0]):
                out[c] = self.plus_u_terms(P[c], S, dens)[1]
            return out
        if self.has_omega:
            out[0] = out[0] + self.exchange_fock(P[0] - dens[0], S, dens[2])
        if self.has_onsite:
            out[0] = out[0] + self.onsite_terms(P[0], S, dens, ao_shell)[1]
        return out

    def density_energy(self, P: Tensor, dens: List[Tensor], ao_shell: Tensor) -> Tensor:
        """The energy of the density-dependent terms at the channel densities ``P`` ``(C, nb, nb)``."""
        S = dens[1]
        energy = torch.zeros((), dtype=P.dtype, device=P.device)
        if self.plus_u > 0:
            for c in range(P.shape[0]):
                energy = energy + self.plus_u_terms(P[c], S, dens)[0]
            return energy
        if self.has_omega:
            dP = P[0] - dens[0]
            energy = energy + 0.5 * (dP * self.exchange_fock(dP, S, dens[2])).sum()
        if self.has_onsite:
            energy = energy + self.onsite_terms(P[0], S, dens, ao_shell)[0]
        return energy

    def scc_general(self, M: Tensor, H0: Tensor, ctx: List[Tensor], n0: Tensor, n_alpha: float, n_beta: float,
                    shell_atom: Tensor, ao_shell: Tensor, n_atoms: int, w_i: Tensor, w_j: Tensor, w_v: Tensor,
                    dens: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor, bool]:
        """The self-consistent cycle of the eager model with spin channels and density mixing: the mixed quantities
        are the shell charges, the shell magnetizations (spin) and the channel densities (density-dependent terms;
        each spin channel starts from half the reference density). Returns the channel densities ``(C, nb, nb)``,
        the energy-weighted density, ``T S_el`` and whether the cycle converged."""
        n_shell, nb = n0.shape[0], H0.shape[0]
        channels = 2 if self.spin else 1
        L = torch.linalg.cholesky(M[0])
        Lt = L.t()
        left = torch.linalg.solve_triangular(L, M, upper=False)
        h0 = torch.linalg.solve_triangular(Lt, torch.linalg.solve_triangular(L, H0, upper=False), upper=True,
                                           left=False)
        parts = [torch.zeros(n_shell * channels, dtype=H0.dtype, device=H0.device)]
        if self.density_mixing:
            start = 0.5 * dens[0] if self.spin else dens[0]
            parts.append(start[None].expand(channels, nb, nb).reshape(-1))
        x = torch.cat(parts)
        off = n_shell * channels
        xs: List[Tensor] = []
        rs: List[Tensor] = []
        converged = False
        beta = self.mixing
        best_err, best_x, worse = float("inf"), x.clone(), 0
        checkpoint = float("inf")
        zm = torch.zeros((n_atoms, 3), dtype=H0.dtype, device=H0.device)
        zt = torch.zeros((n_atoms, 6), dtype=H0.dtype, device=H0.device)
        out = torch.zeros(channels, nb, nb, dtype=H0.dtype, device=H0.device)
        W = torch.zeros_like(H0)
        ts = torch.zeros((), dtype=H0.dtype, device=H0.device)
        for iteration in range(1, self.max_iter + 1):
            v_q = self.potentials(x[:n_shell], zm, zt, shell_atom, ctx)[0]
            v_m = torch.zeros_like(v_q)
            if self.spin:
                v_m = v_m.index_add(0, w_i, w_v * x[n_shell:2 * n_shell][w_j])
            D = torch.zeros(channels, nb, nb, dtype=H0.dtype, device=H0.device)
            if self.density_mixing:
                D = self.density_fock(x[off:].view(channels, nb, nb), dens, ao_shell)
            pops: List[Tensor] = []
            dens_out: List[Tensor] = []
            W = torch.zeros_like(H0)
            ts = torch.zeros((), dtype=H0.dtype, device=H0.device)
            for c in range(channels):
                v = v_q - v_m if c == 0 else v_q + v_m
                K = torch.linalg.solve_triangular(Lt, (left * v[ao_shell][None, None, :]).sum(0), upper=True,
                                                  left=False)
                F = h0 - 0.5 * (K + K.t())
                if self.density_mixing:
                    F = F + torch.linalg.solve_triangular(Lt, torch.linalg.solve_triangular(L, D[c], upper=False),
                                                          upper=True, left=False)
                eps, Co = torch.linalg.eigh(F)
                if self.spin:
                    occ, t_c = self.fermi(eps, n_alpha if c == 0 else n_beta, 0.0)
                else:
                    occ, t_c = self.fermi(eps, n_alpha, n_beta)
                C = torch.linalg.solve_triangular(Lt, Co, upper=True)
                P = (C * occ[None, :]) @ C.t()
                W = W + (C * (occ * eps)[None, :]) @ C.t()
                ts = ts + t_c
                dens_out.append(P)
                pops.append(torch.zeros_like(n0).index_add(0, ao_shell, (P * M[0]).sum(-1)))
            out = torch.stack(dens_out)
            new = [n0 - pops[0] - pops[1]] if self.spin else [n0 - pops[0]]
            if self.spin:
                new.append(pops[0] - pops[1])
            if self.density_mixing:
                new.append(out.reshape(-1))
            r = torch.cat(new) - x
            err = float(r.abs().max())
            if self.residual(r) < self.threshold:
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
        return out, W, ts, converged

    def fermi_pooled(self, eps: Tensor, nel: float) -> Tuple[Tensor, Tensor]:
        """Occupations ``(K, n)`` of one spin channel at the k-points and ``T S_el`` per k-point, with one Fermi level
        for all of them: the levels of every k-point counted with its weight, ``nel`` electrons over all copies, by
        the Newton steps of :meth:`~xnn.sqmnn.models.tightbinding_export.TightBindingExport.fermi`."""
        if nel <= 0.0:
            return torch.zeros_like(eps), torch.zeros(eps.shape[0], dtype=eps.dtype, device=eps.device)
        pooled = torch.sort(torch.repeat_interleave(eps, torch.round(self.kweight).long(), dim=0).reshape(-1)).values
        count = pooled.shape[0]
        top = min(max(int(math.ceil(nel)), 1), count)
        mid = 0.5 * (float(pooled[top - 1]) + float(pooled[min(top, count - 1)]))
        lo, hi = float(pooled.min()) - 1.0, float(pooled.max()) + 1.0
        target = 4.0 * 2.220446049250313e-16 * max(nel, 1.0)
        for _ in range(200):
            f = 0.5 * (1.0 - torch.tanh(0.5 * (pooled - mid) / self.kt))
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
        occ = torch.sigmoid(-(eps - mid) / self.kt)
        ts = -self.kt * (torch.special.xlogy(occ, occ) + torch.special.xlogy(1.0 - occ, 1.0 - occ)).sum(-1)
        return occ, ts

    def scc_kpoints(self, S2: Tensor, H2: Tensor, ctx: List[Tensor], n0: Tensor, n_alpha: float, n_beta: float,
                    shell_atom: Tensor, ao_shell2: Tensor, n_atoms: int, w_i: Tensor, w_j: Tensor,
                    w_v: Tensor) -> Tuple[Tensor, Tensor, Tensor, bool]:
        """The self-consistent cycle of a cell sampled at k-points, as the eager model: the real forms ``(K, 2 nb,
        2 nb)`` of the matrices of the irreducible k-points, one Fermi level per spin channel for all of them, the
        populations summed with the weights of the k-points over the copies. Returns the channel densities
        ``(C, K, 2 nb, 2 nb)``, the energy-weighted densities ``(K, 2 nb, 2 nb)``, ``T S_el`` and whether the cycle
        converged."""
        n_shell = n0.shape[0]
        channels = 2 if self.spin else 1
        scale = self.kweight / self.copies
        L = torch.linalg.cholesky(S2)
        Lt = L.transpose(-1, -2)
        left = torch.linalg.solve_triangular(L, S2, upper=False)
        h0 = torch.linalg.solve_triangular(Lt, torch.linalg.solve_triangular(L, H2, upper=False), upper=True,
                                           left=False)
        x = torch.zeros(n_shell * channels, dtype=H2.dtype, device=H2.device)
        xs: List[Tensor] = []
        rs: List[Tensor] = []
        converged = False
        beta = self.mixing
        best_err, best_x, worse = float("inf"), x.clone(), 0
        checkpoint = float("inf")
        zm = torch.zeros((n_atoms, 3), dtype=H2.dtype, device=H2.device)
        zt = torch.zeros((n_atoms, 6), dtype=H2.dtype, device=H2.device)
        out = torch.zeros_like(H2).unsqueeze(0).repeat(channels, 1, 1, 1)
        W = torch.zeros_like(H2)
        ts = torch.zeros((), dtype=H2.dtype, device=H2.device)
        for iteration in range(1, self.max_iter + 1):
            v_q = self.potentials(x[:n_shell], zm, zt, shell_atom, ctx)[0]
            v_m = torch.zeros_like(v_q)
            if self.spin:
                v_m = v_m.index_add(0, w_i, w_v * x[n_shell:][w_j])
            pops: List[Tensor] = []
            dens: List[Tensor] = []
            W = torch.zeros_like(H2)
            ts = torch.zeros((), dtype=H2.dtype, device=H2.device)
            for c in range(channels):
                v = (v_q - v_m if c == 0 else v_q + v_m)[ao_shell2]
                K = torch.linalg.solve_triangular(Lt, left * v[None, None, :], upper=True, left=False)
                eps, Co = torch.linalg.eigh(h0 - 0.5 * (K + K.transpose(-1, -2)))
                if self.spin:
                    occ, t_k = self.fermi_pooled(eps, (n_alpha if c == 0 else n_beta) * self.copies)
                else:
                    f_a, t_a = self.fermi_pooled(eps, n_alpha * self.copies)
                    f_b, t_b = self.fermi_pooled(eps, n_beta * self.copies)
                    occ, t_k = f_a + f_b, t_a + t_b
                C = torch.linalg.solve_triangular(Lt, Co, upper=True)
                P = (C * occ[:, None, :]) @ C.transpose(-1, -2)
                W = W + (C * (occ * eps)[:, None, :]) @ C.transpose(-1, -2)
                ts = ts + (scale * t_k).sum()
                dens.append(P)
                rows = (scale[:, None] * (P * S2).sum(-1)).sum(0)
                pops.append(torch.zeros_like(n0).index_add(0, ao_shell2, rows))
            out = torch.stack(dens)
            new = [n0 - pops[0] - pops[1], pops[0] - pops[1]] if self.spin else [n0 - pops[0]]
            r = torch.cat(new) - x
            err = float(r.abs().max())
            if self.residual(r) < self.threshold:
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
        return out, W, ts, converged

    def node_features_energy_charges(self, atomic_numbers: Tensor, edge_index: Tensor, edge_vec: Tensor,
                                     pos: Tensor, cell: Tensor, pbc: Tensor, total_charge: float,
                                     spin_multiplicity: float) -> Tuple[Tensor, Tensor, Tensor]:
        """No features ``(N, 0)``, the energies per atom (eV) and the Mulliken charges of one molecule or one cell
        periodic in three directions (at the Gamma point); the neighbor list is not used (every atom pair and its
        images enter)."""
        periodic = bool(pbc.any()) and bool((cell.abs() > 0).any())
        if periodic and not bool(pbc.all()):
            raise ValueError("an exported DFTB model evaluates molecules and cells periodic in three directions")
        device = pos.device
        z = atomic_numbers.long()
        n = z.shape[0]
        if bool((z >= self.lmax.shape[0]).any()) or bool((self.lmax[z.clamp(max=self.lmax.shape[0] - 1)] < 0).any()):
            raise ValueError("the parameter set has no parameters for an element of the structure")
        kinds = torch.unique(z)
        if bool((self.pair_id[kinds[:, None], kinds[None, :]] < 0).any()):
            raise ValueError("the parameter set has no parameters for a pair of elements of the structure")
        x = pos.to(torch.float64) / self.bohr
        lat = cell.to(torch.float64) / self.bohr
        counts = self.lmax[z] + 1
        shell_atom = torch.repeat_interleave(torch.arange(n, device=device), counts)
        sl = torch.arange(shell_atom.shape[0], device=device) - (torch.cumsum(counts, 0) - counts)[shell_atom]
        width = 2 * sl + 1
        ao_shell = torch.repeat_interleave(torch.arange(sl.shape[0], device=device), width)
        ao_atom = shell_atom[ao_shell]
        atom_orbitals = (counts * counts)
        atom_first = torch.cumsum(atom_orbitals, 0) - atom_orbitals
        nb = int(width.sum())
        if periodic:
            i, j, vec, once = self.image_pairs(x, lat, max(self.sk_reach, self.rep_reach))
        else:
            iu = torch.triu_indices(n, n, 1, device=device)
            i, j = iu[0], iu[1]
            vec = x[j] - x[i]
            once = torch.zeros_like(i, dtype=torch.bool)
        r = torch.linalg.norm(vec, dim=-1)
        sampled = periodic and self.has_k
        if sampled:
            # the phase of every entry at the k-points, from its lattice translation
            shift = (vec - (x[j] - x[i])).detach()
            phase = 2.0 * math.pi * (torch.round(torch.linalg.solve(lat.detach().t(), shift.t()).t()) @ self.kpts.t())
            cos_k, sin_k = torch.cos(phase), torch.sin(phase)
        else:
            cos_k = torch.ones(r.shape[0], 1, dtype=x.dtype, device=device)
            sin_k = torch.zeros(r.shape[0], 0, dtype=x.dtype, device=device)
        S_re, S_im, H_re, H_im = self.two_center(z, i, j, vec, r, atom_first, once, nb, cos_k, sin_k)
        sz = z[shell_atom]
        eye = torch.eye(nb, dtype=x.dtype, device=device)
        levels = torch.diag(self.levels[sz, sl][ao_shell])
        S, H0 = S_re[0] + eye, H_re[0] + levels
        # at k-points: the real forms [[A, -B], [B, A]] of the complex matrices A + iB
        S2, H2 = S[None], H0[None]
        if sampled:
            A_s, A_h = S_re + eye, H_re + levels
            S2 = torch.cat([torch.cat([A_s, -S_im], -1), torch.cat([S_im, A_s], -1)], -2)
            H2 = torch.cat([torch.cat([A_h, -H_im], -1), torch.cat([H_im, A_h], -1)], -2)
        ao_shell2 = torch.cat([ao_shell, ao_shell])
        if self.shells:
            site_atom, site_l = shell_atom, sl
            shell_site = torch.arange(shell_atom.shape[0], device=device)
        else:
            site_atom, site_l = torch.arange(n, device=device), torch.zeros(n, dtype=torch.long, device=device)
            shell_site = shell_atom
        if periodic:
            ctx = self.periodic_gamma(z, x, lat, site_atom, site_l, shell_site)
        else:
            ctx = self.gamma_terms(z, x, site_atom, site_l, shell_site)
        n0 = self.n0[sz, sl]
        electrons = float(n0.sum()) - total_charge
        half = 0.5 * float(int(round(electrons)))
        n_up, n_dn = half, half
        if self.spin:
            # a given multiplicity m puts m - 1 more electrons in the first channel, otherwise an odd count one
            unpaired = int(round(spin_multiplicity)) - 1 if spin_multiplicity > 0.0 else int(round(electrons)) % 2
            n_up, n_dn = 0.5 * (electrons + unpaired), 0.5 * (electrons - unpaired)
        M = S[None]
        same = shell_atom[:, None] == shell_atom[None, :]
        pairs = torch.nonzero(same)
        w_i, w_j = pairs[:, 0], pairs[:, 1]
        w_v = self.spin_w[z[shell_atom[w_i]], sl[w_i], sl[w_j]]
        if self.spin and bool((~self.has_spin[z]).any()):
            raise ValueError("the model has no spin constants for an element of the structure")
        dens = self.density_context(z, x, lat, periodic, n0, S, ao_atom, ao_shell, shell_atom, sl)
        channels = torch.zeros((1, nb, nb), dtype=x.dtype, device=device)
        channels_k = torch.zeros_like(S2).unsqueeze(0)
        W_k = torch.zeros_like(S2)
        P, W = torch.zeros_like(H0), torch.zeros_like(H0)
        with torch.no_grad():
            ctx_d = [c.detach() for c in ctx]
            if sampled:
                channels_k, W_k, ts, converged = self.scc_kpoints(S2.detach(), H2.detach(), ctx_d, n0, n_up, n_dn,
                                                                  shell_atom, ao_shell2, n, w_i, w_j, w_v)
            elif self.spin or self.density_mixing:
                dens_d = [d.detach() for d in dens]
                channels, W, ts, converged = self.scc_general(M.detach(), H0.detach(), ctx_d, n0, n_up, n_dn,
                                                              shell_atom, ao_shell, n, w_i, w_j, w_v, dens_d)
                P = channels.sum(0)
            else:
                P, W, ts, converged = self.scc(M.detach(), H0.detach(), ctx_d, n0, half, half, shell_atom,
                                               ao_shell, ao_atom, n, torch.zeros_like(n0))
        weight = torch.where(once, torch.full_like(r, 0.5), torch.ones_like(r))
        if sampled:
            # every k-point with its weight over the copies of the electrons
            scale = self.kweight / self.copies
            P2 = channels_k.sum(0)
            q_shell = n0 - torch.zeros_like(n0).index_add(0, ao_shell2, (scale[:, None] * (P2 * S2).sum(-1)).sum(0))
            energy = (scale[:, None, None] * P2 * H2).sum() + self.charge_energy(q_shell, shell_atom, ctx) \
                + self.repulsion(z, i, j, r, weight) - ts - (scale[:, None, None] * W_k * (S2 - S2.detach())).sum()
            if self.spin:
                spin_rows = [(scale[:, None] * (channels_k[c] * S2).sum(-1)).sum(0) for c in range(2)]
                up = torch.zeros_like(n0).index_add(0, ao_shell2, spin_rows[0])
                dn = torch.zeros_like(n0).index_add(0, ao_shell2, spin_rows[1])
                m = up - dn
                energy = energy + 0.5 * (m[w_i] * w_v * m[w_j]).sum()
        else:
            q_shell = self.moments(P, M, n0, ao_shell, ao_atom, n)[0]
            energy = (P * H0).sum() + self.charge_energy(q_shell, shell_atom, ctx) \
                + self.repulsion(z, i, j, r, weight) - ts - (W * (S - S.detach())).sum()
            if self.spin:
                up = torch.zeros_like(n0).index_add(0, ao_shell, (channels[0] * S).sum(-1))
                dn = torch.zeros_like(n0).index_add(0, ao_shell, (channels[1] * S).sum(-1))
                m = up - dn
                energy = energy + 0.5 * (m[w_i] * w_v * m[w_j]).sum()
        if self.density_mixing:
            energy = energy + self.density_energy(channels, dens, ao_shell)
        zero = torch.zeros(n, dtype=torch.long, device=device)
        charge = torch.full((1,), total_charge, dtype=x.dtype, device=device)
        if periodic:
            if self.disp_reach > 0.0:
                di, dj, dvec, dself = self.image_pairs(x, lat, self.disp_reach)
                both = ~dself
                ei = torch.cat([torch.stack([di, dj]), torch.stack([dj[both], di[both]])], 1)
                node = self.dispersion.evaluate(z, x * self.disp_bohr, ei,
                                                torch.cat([dvec, -dvec[both]]) * self.disp_bohr, zero, 1,
                                                (lat * self.disp_bohr)[None],
                                                torch.ones((1, 3), dtype=torch.bool, device=device),
                                                charge)["node_energy"]
                energy = energy + node.sum() / self.disp_hartree
            if self.hh:
                hi, hj, hvec, hself = self.image_pairs(x, lat, self.hh_cutoff)
                rh = torch.linalg.norm(hvec, dim=-1)
                hh = (z[hi] == 1) & (z[hj] == 1)
                e_hh = self.hh_height / (1.0 + torch.exp(self.hh_steepness * (rh * self.bohr / self.hh_middle - 1.0)))
                e_hh = torch.where(hself, 0.5 * e_hh, e_hh)
                energy = energy + torch.where(hh, e_hh, torch.zeros_like(rh)).sum()
        else:
            ei = torch.cat([torch.stack([i, j]), torch.stack([j, i])], 1)
            node = self.dispersion.evaluate(z, x * self.disp_bohr, ei, torch.cat([vec, -vec]) * self.disp_bohr, zero,
                                            1, torch.zeros((1, 3, 3), dtype=x.dtype, device=device),
                                            torch.zeros((1, 3), dtype=torch.bool, device=device), charge)["node_energy"]
            energy = energy + node.sum() / self.disp_hartree
            if self.hh:
                hh = (z[i] == 1) & (z[j] == 1) & (r < self.hh_cutoff)
                e_hh = self.hh_height / (1.0 + torch.exp(self.hh_steepness * (r * self.bohr / self.hh_middle - 1.0)))
                energy = energy + torch.where(hh, e_hh, torch.zeros_like(r)).sum()
        charges = torch.zeros(n, dtype=x.dtype, device=device).index_add(0, shell_atom, q_shell)
        node_energy = (energy * self.autoev / n).expand(n)
        return torch.zeros((n, 0), dtype=x.dtype, device=device), node_energy, charges.detach()
