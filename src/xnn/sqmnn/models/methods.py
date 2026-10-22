"""PM3, PM6, PM6-ORG and PM7: the core-core terms and corrections of each method.

Common core-core form: ``E_AB = Z_A Z_B gamma_AB s(R) + G(R)`` with
``gamma_AB = (s_A s_A | s_B s_B)`` evaluated with the core additive terms.

* **PM3** (Stewart 1989): ``s = 1 + exp(-alpha_A R) + exp(-alpha_B R)``
  (``R exp(-alpha R)`` for N and O bonded to H), Gaussians
  ``G = Z_A Z_B / R sum_k K_k exp(-L_k (R - M_k)^2)`` of both atoms; pairs with
  diatomic parameters use ``s = 1 + 2 x_AB exp(-alpha_AB R)`` instead.
* **PM6** (Stewart 2007, eqs 6-11): ``s = 1 + 2 x_AB exp(-alpha_AB (R +
  0.0003 R^6))`` with the C-H, N-H, O-H (``exp(-alpha R^2)``), C-C (triple
  bonds) and Si-O special forms; pairs without diatomic parameters use
  ``|10 exp(-2.18 R) Z_A Z_B gamma| + Z_A Z_B gamma``; one Gaussian per atom;
  the repulsion ``1e-8 / x^12`` of the unpolarizable cores, ``x = R / (Z_A^(1/3)
  + Z_B^(1/3))``.
* **PM7** (Stewart 2013): PM6's form; missing diatomic parameters are the
  means of the like-pair values; a further O-H term.
* **PM6-ORG** (Stewart & Stewart 2023): PM6's form with Gaussian steric
  terms for the pairs of H, C, N, O and S.

Each threshold (``ax <= 25`` Gaussian cut-offs, ``x < 3``) and the numerical
constants follow the reference program MOPAC.
"""
from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import Tensor

from xnn.common.data import AtomicGraph
from xnn.common.models import register_model

from ..common import elements as el
from .corrections import (
    bond_topology,
    h4_hbond,
    hh_repulsion,
    nsp2_correction,
    peptide_torsion,
    pm7_dispersion,
    pm7_hbond,
    si_o_h_correction,
    yne_correction,
)
from .nddo import NDDO


def _pair_is(za: Tensor, zb: Tensor, a: int, b: int) -> Tensor:
    lo, hi = torch.minimum(za, zb), torch.maximum(za, zb)
    return (lo == min(a, b)) & (hi == max(a, b))


def _gaussians(model: NDDO, za: Tensor, zb: Tensor, r: Tensor, zz: Tensor, ks: Sequence[int],
               strict: bool, active: Tensor) -> Tensor:
    out = torch.zeros_like(r)
    for j in ks:
        for z in (za, zb):
            g1, g2, g3 = model._gaussian(z, j)
            ax = g2 * (r - g3) ** 2
            ok = (ax < 25.0) if strict else (ax <= 25.0)
            ok = ok & (g1.abs() > 0.0) & active
            out = out + torch.where(ok, zz / r * g1 * torch.exp(-torch.where(ok, ax, torch.zeros_like(ax))),
                                    torch.zeros_like(r))
    return out


class _PM6Family(NDDO):
    """Core-core terms shared by PM6, PM7 and PM6-ORG."""

    average_missing_pairs = False

    def _diatomic(self, za, zb):
        x = self.pair_parameter("XFAC")[za, zb]
        a = self.pair_parameter("ALPB")[za, zb]
        if self.average_missing_pairs:
            missing = x.abs() < 1e-5
            X, A = self.pair_parameter("XFAC"), self.pair_parameter("ALPB")
            x = torch.where(missing, 0.5 * (X[za, za] + X[zb, zb]), x)
            a = torch.where(missing, 0.5 * (A[za, za] + A[zb, zb]), a)
        return x, a

    def _scale(self, za, zb, r, x, a) -> Tensor:
        par = self.global_parameters()
        scale = 1.0 + 2.0 * x * torch.exp(-a * (r + 0.0003 * r ** 6))
        hx = 1.0 + 2.0 * x * torch.exp(-a * r * r)
        scale = torch.where(_pair_is(za, zb, 1, 6) | _pair_is(za, zb, 1, 7), hx, scale)
        scale = torch.where(_pair_is(za, zb, 1, 8), hx - par[3] * torch.exp(-par[4] * r * 2.0), scale)
        scale = torch.where(_pair_is(za, zb, 6, 6), scale + par[1] * torch.exp(-par[2] * r), scale)
        scale = torch.where(_pair_is(za, zb, 8, 14), scale - 0.7e-3 * torch.exp(-(r - 2.9) ** 2), scale)
        return scale

    def core_core(self, za, zb, r, gab, tore_a, tore_b):
        zz = tore_a * tore_b
        enuc = zz * gab
        x, a = self._diatomic(za, zb)
        has = x.abs() > 1e-5
        a = torch.where(a < 1e-6, torch.full_like(a, 1.2), a)
        e_pair = enuc * self._scale(za, zb, r, x, a)
        lanth = ((za > 56) & (za < 72)) | ((zb > 56) & (zb < 72))
        generic = torch.where(lanth, 10.0 * torch.exp(-3.0 * r), 10.0 * torch.exp(-2.18 * r))
        e_generic = (generic * enuc).abs() + enuc
        e = torch.where(has, e_pair, e_generic)
        e = e + _gaussians(self, za, zb, r, zz, (1,), True, torch.ones_like(has))
        e = e + _gaussians(self, za, zb, r, zz, (1, 2, 3, 4), False, ~has)
        ax = r / (za.to(r.dtype) ** 0.3333 + zb.to(r.dtype) ** 0.3333)
        core = torch.clamp(1e-8 / torch.where(ax < 3.0, ax, torch.ones_like(ax)) ** 12, max=1e5)
        return e + torch.where(ax < 3.0, core, torch.zeros_like(core))

    def _mm_terms(self, z, pos, bonds, barrier: float, nsp2: bool, si_o_h: bool) -> Tensor:
        out = peptide_torsion(z, pos, barrier) + yne_correction(z, pos, bonds)
        if nsp2:
            out = out + nsp2_correction(z, pos, bonds)
        if si_o_h:
            out = out + si_o_h_correction(z, pos, bonds)
        return out


def _per_structure(data: AtomicGraph, pos: Tensor, fn) -> Tensor:
    out = pos.new_zeros(pos.shape[0])
    z_all = data.atomic_numbers.tolist()
    batch = data.batch
    for b in range(data.num_graphs):
        idx = (batch == b).nonzero().squeeze(-1)
        if idx.numel() == 0:
            continue
        zb = [z_all[i] for i in idx.tolist()]
        out = out.index_add(0, idx, fn(zb, pos[idx]))
    return out


@register_model("pm3")
class PM3(NDDO):
    """PM3 (Stewart, J. Comput. Chem. 10, 209 and 221, 1989); see :class:`NDDO`."""

    default_parameters = "pm3"
    method = "PM3"
    peptide_barrier = 7.1853

    def core_core(self, za, zb, r, gab, tore_a, tore_b):
        zz = tore_a * tore_b
        enuc = zz * gab
        x = self.pair_parameter("XFAC")[za, zb]
        a = self.pair_parameter("ALPB")[za, zb]
        has = x.abs() > 1e-5
        a = torch.where(a < 1e-6, torch.full_like(a, 1.2), a)
        e_pair = enuc * (1.0 + 2.0 * x * torch.exp(-a * r))
        alp = self.element_parameter("ALP")
        eni, enj = torch.exp(-alp[za] * r), torch.exp(-alp[zb] * r)
        scale = eni + enj
        nt = za + zb
        nh = (nt == 8) | (nt == 9)
        scale = scale + torch.where(nh & ((za == 7) | (za == 8)), (r - 1.0) * eni, torch.zeros_like(r))
        scale = scale + torch.where(nh & ((zb == 7) | (zb == 8)), (r - 1.0) * enj, torch.zeros_like(r))
        e = torch.where(has, e_pair, (scale * enuc).abs() + enuc)
        return e + _gaussians(self, za, zb, r, zz, (1, 2, 3, 4), False, ~(x > 1e-4))

    def corrections(self, data, pos):
        return _per_structure(data, pos, lambda z, p: peptide_torsion(z, p, self.peptide_barrier))


@register_model("pm6")
class PM6(_PM6Family):
    """PM6 (Stewart, J. Mol. Model. 13, 1173, 2007); see :class:`NDDO`."""

    default_parameters = "pm6"
    method = "PM6"

    def corrections(self, data, pos):
        def fn(z, p):
            bonds = bond_topology(z, p)
            return self._mm_terms(z, p, bonds, 2.5, nsp2=True, si_o_h=False)
        return _per_structure(data, pos, fn)


def _pm7_balance(ints: Dict[str, Tensor], za: Tensor, zb: Tensor, n_orb_a: Tensor,
                 n_orb_b: Tensor) -> None:
    """PM7's balance of the two-center integrals of transition metals (in place).

    For an atom X with ``d`` electrons in its reference configuration the
    average of the two-center Coulomb integrals between its ``d`` shell and
    the ``s``, ``p`` and ``d`` shells of the partner, and of the attraction of
    its ``d`` (and, the partner having ``p`` orbitals, its ``p``) electrons
    by the partner's core, are set to the ``(ss|ss)`` value used in the
    core-core term.
    """
    if ints["W"].shape[-1] < 9:
        return
    W, VA, VB = ints["W"].clone(), ints["VA"].clone(), ints["VB"].clone()
    ints["W"], ints["VA"], ints["VB"] = W, VA, VB
    dt = W.dtype
    iod = torch.as_tensor(el.OCC_D, device=za.device)
    s, p, d = [0], [1, 2, 3], [4, 5, 6, 7, 8]
    ss = W[:, 0, 0, 0, 0].clone()

    def coul(rows, cols, left_on_a):
        """Sum of (rr|cc) over the diagonals of two shells, rows on A or on B."""
        tot = torch.zeros_like(ss)
        for i in rows:
            for j in cols:
                tot = tot + (W[:, i, i, j, j] if left_on_a else W[:, j, j, i, i])
        return tot

    def shift(rows_a, cols_b, value, mask):
        add = torch.where(mask, value, torch.zeros_like(value))
        for i in rows_a:
            for j in cols_b:
                W[:, i, i, j, j] = W[:, i, i, j, j] + add

    def core_shift(V, shell, mask):
        avg = sum(V[:, i, i] for i in shell) / len(shell)
        add = torch.where(mask, V[:, 0, 0] - avg, torch.zeros_like(avg))
        for i in shell:
            V[:, i, i] = V[:, i, i] + add

    # the partner listed second in the reference program's pair loop is atom B
    x_has = iod[zb] > 0                     # d electrons on B
    y_has = iod[za] > 0                     # d electrons on A
    p_on_a, p_on_b = n_orb_a > 1, n_orb_b > 1
    # B's d shell with A's p and s shells; B's p and d core attraction
    m1 = x_has & p_on_a
    shift(p, d, ss - coul(p, d, True) / 15.0, m1)
    core_shift(VB, p, m1)
    shift(s, d, ss - coul(s, d, True) / 5.0, x_has)
    core_shift(VB, d, x_has)
    # A's d shell with B's d, p and s shells; A's p and d core attraction
    m2 = y_has & x_has
    shift(d, d, ss - coul(d, d, True) / 25.0, m2)
    m3 = y_has & p_on_b
    shift(d, p, ss - coul(d, p, True) / 15.0, m3)
    core_shift(VA, p, m3)
    shift(d, s, ss - coul(d, s, True) / 5.0, y_has)
    core_shift(VA, d, y_has)
    del dt


@register_model("pm7")
class PM7(_PM6Family):
    """PM7 (Stewart, J. Mol. Model. 19, 1, 2013); see :class:`NDDO`.

    PM6's Hamiltonian with the PM7 parameters, the transition-metal integral
    balance, like-pair means for missing diatomic parameters, and the
    dispersion and hydrogen-bond terms of the method.
    """

    default_parameters = "pm7"
    method = "PM7"
    average_missing_pairs = True
    feather = True

    def adjust_integrals(self, ints, za, zb, n_orb_a, n_orb_b):
        _pm7_balance(ints, za, zb, n_orb_a, n_orb_b)

    def corrections(self, data, pos):
        def fn(z, p):
            bonds = bond_topology(z, p)
            return (self._mm_terms(z, p, bonds, 3.1595, nsp2=False, si_o_h=True)
                    + pm7_dispersion(z, p, bonds) + pm7_hbond(z, p))
        return _per_structure(data, pos, fn)


@register_model("pm6_org")
class PM6ORG(_PM6Family):
    """PM6-ORG (Stewart & Stewart, J. Mol. Model. 29, 284, 2023); see :class:`NDDO`.

    PM6's Hamiltonian reparameterized for proteins, with Gaussian steric
    terms in the core-core repulsion of the H, C, N, O and S pairs and the
    D3 dispersion, H4 hydrogen-bond and H-H repulsion corrections (D3H4).
    """

    default_parameters = "pm6_org"
    method = "PM6-ORG"
    feather = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from xnn.common.models.d3 import DFTD3
        par = self.global_table
        self.d3 = DFTD3(damping="zero", s6=float(par[7]), rs6=float(par[9]), alp=float(par[8]),
                        s8=float(par[10]), rs8=1.0, references="2010", cutoff_pair=15.0,
                        cutoff_triple=15.0, cutoff_cn=1.0e4)

    def _scale(self, za, zb, r, x, a):
        par = self.global_parameters()
        w = 0.01
        base = 1.0 + 2.0 * x * torch.exp(-a * (r + 0.0003 * r ** 6))
        hx = 1.0 + 2.0 * x * torch.exp(-a * r * r)

        def steric(n1, n2, n3):
            return w * par[n1] * torch.where(r - par[n3] > 0.0, torch.exp(-par[n2] * (r - par[n3]) ** 2),
                                             torch.ones_like(r))

        scale = base
        for (a1, a2), form in (
                ((1, 1), lambda: base + steric(16, 17, 18)),
                ((1, 6), lambda: hx + steric(19, 11, 12)),
                ((1, 7), lambda: hx + steric(38, 39, 40)),
                ((1, 8), lambda: hx + steric(3, 4, 5)),
                ((1, 16), lambda: base + steric(44, 45, 46)),
                ((6, 6), lambda: base + par[1] * torch.exp(-par[2] * r) + steric(13, 14, 15)),
                ((6, 7), lambda: base + steric(35, 36, 37)),
                ((6, 8), lambda: base + steric(20, 21, 22)),
                ((6, 16), lambda: base + steric(29, 30, 31)),
                ((7, 8), lambda: base + steric(26, 27, 28)),
                ((7, 16), lambda: base + steric(41, 42, 43)),
                ((8, 8), lambda: base + steric(32, 33, 34)),
                ((8, 14), lambda: base - 0.7e-3 * torch.exp(-(r - 2.9) ** 2)),
                ((8, 16), lambda: base + steric(23, 24, 25))):
            scale = torch.where(_pair_is(za, zb, a1, a2), form(), scale)
        return scale

    def corrections(self, data, pos):
        from xnn.common.models.electrostatics import all_pairs
        from xnn.common.models.ops import scatter_sum  # noqa: F401
        src, dst = all_pairs(data.batch)
        edge_index = torch.stack([src, dst])
        vec = pos[dst] - pos[src]
        b = data.num_graphs
        zero_cell = torch.zeros((b, 3, 3), dtype=pos.dtype, device=pos.device)
        no_pbc = torch.zeros((b, 3), dtype=torch.bool, device=pos.device)
        charge = torch.zeros(b, dtype=pos.dtype, device=pos.device)
        d3 = self.d3.to(pos.dtype).evaluate(data.atomic_numbers, pos, edge_index, vec, data.batch, b,
                                            zero_cell, no_pbc, charge)["node_energy"]

        def fn(z, p):
            bonds = bond_topology(z, p)
            return self._mm_terms(z, p, bonds, 0.0, nsp2=False, si_o_h=False) + h4_hbond(z, p) + hh_repulsion(z, p)
        return d3 * 23.060547830619029 + _per_structure(data, pos, fn)
