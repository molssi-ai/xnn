"""Per-element quantities derived from the NDDO parameters.

For the elements of a batch this module computes, differentiably:

* the classic ``sp`` multipole parameters of MNDO (Dewar & Thiel 1977): the
  charge separations ``dd`` (dipole) and ``qq`` (quadrupole) from the Slater
  exponents, and the additive terms ``am``, ``ad``, ``aq`` that make the
  two-center integrals reduce to the one-center ``GSS``, ``HSP`` and
  ``(GPP - GP2) / 2`` at zero separation;
* for ``spd`` atoms the charge separations ``ddp`` and additive terms ``po``
  of the Thiel-Voityuk multipole scheme, the one-center Slater-Condon
  integrals from the internal exponents, and (transition metals) the ``sp``
  one-center integrals from the same exponents;
* the full one-center two-electron integral tensor ``(ij|kl)`` in the
  molecular orbital order;
* the energy ``EISOL`` of the isolated atom in its reference configuration.

The additive terms are roots of nonlinear equations; they are found with the
iteration counts and tolerances of MOPAC (a five-step secant for ``ad``,
``aq`` and a golden-section search to ``1e-8`` bohr for ``po``), and their
derivatives with respect to the parameters follow from the implicit-function
theorem, so the values are MOPAC's and the gradients exact.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Callable, Dict, Sequence

import numpy as np
import torch
from torch import Tensor

from ..common import elements as el
from ..common.constants import HARTREE
from .angular import L_OF, MOLECULAR, pair_type, slater_condon_factors
from .integrals import slater_condon

# d-shell population terms of the isolated transition-metal atoms (number of
# F0sd, F0dd, G2sd, F2dd, F4dd terms in the configuration energy)
_D_CONFIG = {
    21: (2, 0, 1, 0, 0), 22: (4, 1, 2, 8, 1), 23: (6, 3, 3, 15, 8), 24: (5, 10, 5, 35, 35),
    25: (10, 10, 5, 35, 35), 26: (12, 15, 6, 35, 35), 27: (14, 21, 7, 43, 36),
    28: (16, 28, 8, 50, 43), 29: (10, 45, 5, 70, 70),
    39: (2, 0, 1, 0, 0), 40: (4, 1, 2, 8, 1), 41: (4, 6, 4, 21, 21), 42: (5, 10, 5, 35, 35),
    43: (10, 10, 5, 35, 35), 44: (7, 21, 5, 43, 36), 45: (8, 28, 5, 50, 43),
    46: (0, 45, 0, 70, 70), 47: (10, 45, 5, 70, 70),
    57: (2, 0, 1, 0, 0), 71: (2, 0, 1, 0, 0),
    72: (4, 1, 2, 8, 1), 73: (6, 3, 3, 15, 8), 74: (5, 10, 5, 35, 35), 75: (10, 10, 5, 35, 35),
    76: (12, 15, 6, 35, 35), 77: (14, 21, 7, 43, 36), 78: (9, 36, 5, 56, 56),
    79: (10, 45, 5, 70, 70),
}

# radial integrals of the one-center tensor, by index
_RADIAL = ("none", "gss", "gsp", "g1sp", "f0pp", "f2pp", "f0sd", "f0pd", "f0dd", "f2dd", "f4dd",
           "f2pd", "g1pd", "g3pd", "g2sd", "r1sppd", "r2sdpp", "r2sddd")
_RULES = {
    (frozenset(["ss"]), 0): "gss", (frozenset(["ss", "pp"]), 0): "gsp",
    (frozenset(["sp"]), 1): "g1sp", (frozenset(["pp"]), 0): "f0pp", (frozenset(["pp"]), 2): "f2pp",
    (frozenset(["ss", "dd"]), 0): "f0sd", (frozenset(["pp", "dd"]), 0): "f0pd",
    (frozenset(["pp", "dd"]), 2): "f2pd", (frozenset(["dd"]), 0): "f0dd",
    (frozenset(["dd"]), 2): "f2dd", (frozenset(["dd"]), 4): "f4dd",
    (frozenset(["pd"]), 1): "g1pd", (frozenset(["pd"]), 3): "g3pd",
    (frozenset(["sd"]), 2): "g2sd", (frozenset(["sp", "pd"]), 1): "r1sppd",
    (frozenset(["sd", "pp"]), 2): "r2sdpp", (frozenset(["sd", "dd"]), 2): "r2sddd",
}


@lru_cache(maxsize=None)
def _one_center_layout():
    """Angular factors ``(5, 9, 9, 9, 9)`` and the radial index of every entry."""
    A = slater_condon_factors()
    idx = np.zeros(A.shape, dtype=np.int64)
    ls = [L_OF[lab] for lab in MOLECULAR]
    for K in range(5):
        for i in range(9):
            for j in range(9):
                t1 = pair_type(ls[i], ls[j])
                for k in range(9):
                    for l in range(9):
                        if A[K, i, j, k, l] == 0.0:
                            continue
                        t2 = pair_type(ls[k], ls[l])
                        name = _RULES.get((frozenset([t1, t2]), K))
                        if name is None:
                            raise RuntimeError(f"no radial integral for {t1}{t2} K={K}")
                        idx[K, i, j, k, l] = _RADIAL.index(name)
    return A, idx


def _implicit(x0: Tensor, residual: Callable[[Tensor], Tensor]) -> Tensor:
    """The root ``x0`` (value unchanged) with the implicit-function derivative.

    ``residual(x)`` vanishes at the root and depends on the parameters; the
    returned tensor equals ``x0`` and differentiates as ``-r_theta / r_x``.
    """
    x0 = x0.detach()
    if not torch.is_grad_enabled():
        return x0
    xg = x0.clone().requires_grad_(True)
    with torch.enable_grad():
        r = residual(xg)
        if not r.requires_grad:
            return x0
        drdx = torch.autograd.grad(r.sum(), xg, retain_graph=True)[0]
    g = residual(x0)
    if not g.requires_grad:
        return x0
    safe = torch.where(drdx.abs() > 1e-30, drdx, torch.ones_like(drdx))
    return x0 - (g - g.detach()) / safe.detach()


def _poij(order: int, d: Tensor, fg: Tensor) -> Tensor:
    """Additive term reproducing the one-center integral ``fg`` (golden-section search)."""
    if order == 0:
        return 0.5 * HARTREE / fg

    def f(y):
        if order == 1:
            return 0.25 * HARTREE * (1.0 / y - 1.0 / torch.sqrt(y * y + d * d))
        return 0.125 * HARTREE * (1.0 / y - 2.0 / torch.sqrt(y * y + 0.5 * d * d)
                                  + 1.0 / torch.sqrt(y * y + d * d))

    with torch.no_grad():
        dd, target = d.detach(), fg.detach()

        def fd(y):
            if order == 1:
                return 0.25 * HARTREE * (1.0 / y - 1.0 / torch.sqrt(y * y + dd * dd))
            return 0.125 * HARTREE * (1.0 / y - 2.0 / torch.sqrt(y * y + 0.5 * dd * dd)
                                      + 1.0 / torch.sqrt(y * y + dd * dd))
        a1 = torch.full_like(dd, 0.1)
        a2 = torch.full_like(dd, 5.0)
        f1 = torch.zeros_like(dd)
        f2 = torch.zeros_like(dd)
        for _ in range(100):
            delta = a2 - a1
            active = delta >= 1e-8
            if not bool(active.any()):
                break
            y1 = a1 + delta * 0.382
            y2 = a1 + delta * 0.618
            e1 = (fd(y1) - target) ** 2
            e2 = (fd(y2) - target) ** 2
            left = e1 < e2
            a2 = torch.where(active & left, y2, a2)
            a1 = torch.where(active & ~left, y1, a1)
            f1 = torch.where(active, e1, f1)
            f2 = torch.where(active, e2, f2)
        root = torch.where(f1 >= f2, a2, a1)
    return _implicit(root, lambda y: f(y) - fg)


def _secant(residual: Callable[[Tensor], Tensor], x1: Tensor, x2: Tensor, steps: int = 5) -> Tensor:
    """MOPAC's secant iteration (fixed step count) for the classic additive terms."""
    with torch.no_grad():
        x1, x2 = x1.detach(), x2.detach()
        done = torch.zeros_like(x1, dtype=torch.bool)
        for _ in range(steps):
            r1, r2 = residual(x1), residual(x2)
            done = done | ((r2 - r1).abs() < 1e-25)
            safe = torch.where(done, torch.ones_like(r1), r2 - r1)
            x3 = x1 - (x2 - x1) * r1 / safe
            x1 = torch.where(done, x1, x2)
            x2 = torch.where(done, x2, x3)
    return x2


def _aij(z1: Tensor, z2: Tensor, n1: int, n2: int, l: int) -> Tensor:
    """Radial moment ``<r^l>`` of the product of two Slater orbitals (bohr^l)."""
    zz = z1 + z2 + 1e-20
    f = math.factorial
    return (f(n1 + n2 + l) / math.sqrt(f(2 * n1) * f(2 * n2))
            * (2 * z1 / zz) ** (n1 + 0.5) * (2 * z2 / zz) ** (n2 + 0.5) * 2 ** l / zz ** l)


def element_quantities(z: Sequence[int], p: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Derived per-element quantities of the elements ``z``.

    Parameters
    ----------
    z : sequence of int
        Distinct atomic numbers.
    p : dict
        Per-element parameter tensors ``(len(z),)`` keyed by the MOPAC names
        (``USS``, ``ZS``, ``GSS``, ...); absent parameters are zeros.

    Returns
    -------
    dict
        ``n_orb`` (int list), shell principal quantum numbers, exponents,
        ``uss``/``upp``/``udd``, ``beta*``, ``tore``, ``dd``/``qq``/``rho0``/
        ``rho1``/``rho2`` (classic), ``ddp`` ``(n, 7)`` and ``po`` ``(n, 10)``
        (Thiel-Voityuk, 1-based columns), ``one_center`` ``(n, 9, 9, 9, 9)``
        and ``eisol`` (eV).
    """
    z = [int(v) for v in z]
    n = len(z)
    ref = p["USS"]
    dtype, device = ref.dtype, ref.device

    def col(values):
        return torch.as_tensor(values, dtype=dtype, device=device)

    zero = torch.zeros(n, dtype=dtype, device=device)
    g = {k: v for k, v in p.items()}
    n_orb = [9 if float(g["ZD"][i]) > 1e-8 else (4 if float(g["ZP"][i]) > 1e-20 else 1)
             for i in range(n)]
    has_p = col([no > 1 for no in n_orb])
    row = [el.period(v) for v in z]
    nd_shell = [el.d_shell(v) for v in z]
    main = [el.MAIN_GROUP[v] for v in z]

    zs = g["ZS"]
    zp = torch.where(has_p > 0, torch.clamp(g["ZP"], min=0.3), g["ZP"])
    zd = g["ZD"]
    zsn, zpn, zdn = g["ZSN"], g["ZPN"], g["ZDN"]

    # one-center sp integrals of the transition metals from the internal exponents
    gss, gsp, gpp, gp2, hsp = g["GSS"], g["GSP"], g["GPP"], g["GP2"], g["HSP"]
    tm = [(not main[i]) and float(zsn[i]) >= 1e-4 and float(zpn[i]) >= 1e-4 and z[i] <= 80
          for i in range(n)]
    if any(tm):
        sel = col(tm) > 0
        es = torch.where(sel, zsn, torch.ones_like(zsn))
        ep = torch.where(sel, zpn, torch.ones_like(zpn))
        # the other elements' values are discarded; their shell is raised to 2 where the first row (hydrogen)
        # would leave R^2 undefined
        r_gss, r_gsp, r_hsp, r_f0, r_f2 = (torch.stack(v) for v in zip(*[
            (slater_condon(0, ns, es[i], ns, es[i], ns, es[i], ns, es[i]),
             slater_condon(0, ns, es[i], ns, es[i], ns, ep[i], ns, ep[i]),
             slater_condon(1, ns, es[i], ns, ep[i], ns, es[i], ns, ep[i]) / 3.0,
             slater_condon(0, ns, ep[i], ns, ep[i], ns, ep[i], ns, ep[i]),
             slater_condon(2, ns, ep[i], ns, ep[i], ns, ep[i], ns, ep[i]))
            for i, ns in enumerate(r if tm[j] else max(r, 2) for j, r in enumerate(row))]))
        gss = torch.where(sel, r_gss, gss)
        gsp = torch.where(sel, r_gsp, gsp)
        hsp = torch.where(sel, r_hsp, hsp)
        gpp = torch.where(sel, r_f0 + 0.16 * r_f2, gpp)
        gp2 = torch.where(sel, r_f0 - 0.08 * r_f2, gp2)
    hsp = torch.where(has_p > 0, torch.clamp(hsp, min=1e-7), hsp)

    # classic multipole parameters
    qn = col(row)
    dd = ((2.0 * qn + 1.0) * (4.0 * zs * zp) ** (qn + 0.5) / (zs + zp) ** (2.0 * qn + 2.0)
          / math.sqrt(3.0))
    zp_safe = torch.where(zp > 0, zp, torch.ones_like(zp))
    qq = torch.sqrt((4.0 * qn * qn + 6.0 * qn + 2.0) / 20.0) / zp_safe
    am = torch.where(gss > 1e-20, gss / HARTREE, torch.ones_like(gss))
    hpp = torch.clamp(0.5 * (gpp - gp2), min=0.1)
    dd_s = torch.where(has_p > 0, dd, torch.ones_like(dd))
    qq_s = torch.where(has_p > 0, qq, torch.ones_like(qq))
    hsp_s = torch.where(has_p > 0, hsp, torch.ones_like(hsp))

    def r_ad(x, dd=dd_s, h=hsp_s):
        return 0.5 * x - 0.5 / torch.sqrt(4.0 * dd * dd + 1.0 / (x * x)) - h / HARTREE

    def r_aq(x, qq=qq_s, h=hpp):
        return (0.25 * x - 0.5 / torch.sqrt(4.0 * qq * qq + 1.0 / (x * x))
                + 0.25 / torch.sqrt(8.0 * qq * qq + 1.0 / (x * x)) - h / HARTREE)

    d1 = (hsp_s / (HARTREE * dd_s ** 2)) ** (1.0 / 3.0)
    ad = _implicit(_secant(r_ad, d1, d1 + 0.04), r_ad)
    q1 = (16.0 * hpp / (HARTREE * 48.0 * qq_s ** 4)) ** 0.2
    aq = _implicit(_secant(r_aq, q1, q1 + 0.04), r_aq)
    ad = torch.where(has_p > 0, ad, am)
    aq = torch.where(has_p > 0, aq, am)

    # Thiel-Voityuk multipole parameters (columns 1..6 of ddp, 1..9 of po)
    ddp = [torch.zeros_like(zero) for _ in range(7)]
    po = [torch.zeros_like(zero) for _ in range(10)]
    is_d = col([no == 9 for no in n_orb]) > 0
    radial = {name: torch.zeros_like(zero) for name in _RADIAL[1:]}
    radial.update(gss=gss, gsp=gsp, g1sp=3.0 * hsp, f0pp=(gpp + 2.0 * gp2) / 3.0,
                  f2pp=25.0 / 6.0 * (gpp - gp2))
    if bool(is_d.any()):
        ones = torch.ones_like(zero)
        zs_d = torch.where(is_d, zs, ones)
        zp_d = torch.where(is_d, zp, ones)
        zd_d = torch.where(is_d, zd, ones)
        has_int = is_d & (zdn > 1e-4)
        es = torch.where(has_int, zsn, ones)
        ep = torch.where(has_int, zpn, ones)
        ed = torch.where(has_int, zdn, ones)
        rad = {k: [] for k in ("f0sd", "f0pd", "f0dd", "f2dd", "f4dd", "f2pd", "g1pd", "g3pd",
                               "g2sd", "r1sppd", "r2sdpp", "r2sddd")}
        aij = {k: [] for k in (2, 3, 4, 5, 6)}
        for i in range(n):
            ns, nd = max(row[i], 3), nd_shell[i]
            S, P_, D = es[i], ep[i], ed[i]
            rad["f0sd"].append(slater_condon(0, ns, S, ns, S, nd, D, nd, D))
            rad["f0pd"].append(slater_condon(0, ns, P_, ns, P_, nd, D, nd, D))
            rad["f0dd"].append(slater_condon(0, nd, D, nd, D, nd, D, nd, D))
            rad["f2dd"].append(slater_condon(2, nd, D, nd, D, nd, D, nd, D))
            rad["f4dd"].append(slater_condon(4, nd, D, nd, D, nd, D, nd, D))
            rad["f2pd"].append(slater_condon(2, ns, P_, ns, P_, nd, D, nd, D))
            rad["g1pd"].append(slater_condon(1, ns, P_, nd, D, ns, P_, nd, D))
            rad["g3pd"].append(slater_condon(3, ns, P_, nd, D, ns, P_, nd, D))
            rad["g2sd"].append(slater_condon(2, ns, S, nd, D, ns, S, nd, D))
            rad["r1sppd"].append(slater_condon(1, ns, S, ns, P_, ns, P_, nd, D))
            rad["r2sdpp"].append(slater_condon(2, ns, P_, ns, P_, ns, S, nd, D))
            rad["r2sddd"].append(slater_condon(2, ns, S, nd, D, nd, D, nd, D))
            aij[2].append(_aij(zs_d[i], zp_d[i], ns, ns, 1))
            aij[3].append(_aij(zp_d[i], zp_d[i], ns, ns, 2))
            aij[4].append(_aij(zs_d[i], zd_d[i], ns, nd, 2))
            aij[5].append(_aij(zp_d[i], zd_d[i], ns, nd, 1))
            aij[6].append(_aij(zd_d[i], zd_d[i], nd, nd, 2))
        for k, v in rad.items():
            radial[k] = torch.where(has_int, torch.stack(v), zero)
        radial["f0sd"] = torch.where(g["F0SD"] > 0.001, g["F0SD"], radial["f0sd"])
        radial["g2sd"] = torch.where(g["G2SD"] > 0.001, g["G2SD"], radial["g2sd"])
        aij = {k: torch.stack(v) for k, v in aij.items()}

        def sel(x, fill=1.0):
            return torch.where(is_d, x, torch.full_like(x, fill))

        po[1] = torch.where(is_d & (gss > 0.1), 0.5 * HARTREE / torch.where(gss > 0.1, gss, ones),
                            zero)
        ddp[2] = sel(aij[2] / math.sqrt(12.0))
        po[2] = torch.where(is_d, _poij(1, ddp[2], sel(hsp)), zero)
        ddp[3] = sel(torch.sqrt(aij[3] * 0.1))
        po[3] = torch.where(is_d, _poij(2, ddp[3], sel(0.5 * (gpp - gp2))), zero)
        po[7] = po[1]
        ddp[4] = sel(torch.sqrt(aij[4] * math.sqrt(1.0 / 60.0)))
        po[4] = torch.where(is_d, _poij(2, ddp[4], sel(radial["g2sd"] / 5.0)), zero)
        ddp[5] = sel(aij[5] / math.sqrt(20.0))
        g1, g3 = radial["g1pd"], radial["g3pd"]
        fg5 = 4.0 / 15.0 * g1 + 27.0 / 245.0 * g3 - 1.8 * (3.0 / 49.0 * g3)
        po[5] = torch.where(is_d, _poij(1, ddp[5], sel(fg5)), zero)
        f0, f2, f4 = radial["f0dd"], radial["f2dd"], radial["f4dd"]
        fg8 = 0.2 * ((f0 + 4.0 / 49.0 * f2 + 4.0 / 49.0 * f4)
                     + 2.0 * (f0 + 2.0 / 49.0 * f2 - 24.0 / 441.0 * f4)
                     + 2.0 * (f0 - 4.0 / 49.0 * f2 + 6.0 / 441.0 * f4))
        po[8] = torch.where(is_d & (fg8 > 1e-5), 0.5 * HARTREE / torch.where(fg8 > 1e-5, fg8, ones),
                            torch.where(is_d, torch.full_like(zero, 1e5), zero))
        ddp[6] = sel(torch.sqrt(aij[6] / 14.0))
        fg6 = (3.0 / 49.0 * f2 + 20.0 / 441.0 * f4) - 20.0 / 35.0 * (35.0 / 441.0 * f4)
        po[6] = torch.where(is_d, _poij(2, ddp[6], sel(fg6)), zero)

    # classic values for the main group and the sp atoms
    classic = col([no < 6 or m for no, m in zip(n_orb, main)]) > 0
    rho0 = 0.5 / am
    rho1 = torch.where(ad > 1e-5, 0.5 / ad, zero)
    rho2 = torch.where(aq > 1e-5, 0.5 / aq, zero)
    po[1] = torch.where(classic, rho0, po[1])
    po[2] = torch.where(classic, rho1, po[2])
    po[3] = torch.where(classic, rho2, po[3])
    po[7] = torch.where(classic, po[1], po[7])
    ddp[2] = torch.where(classic, dd, ddp[2])
    ddp[3] = torch.where(classic, qq * math.sqrt(2.0), ddp[3])
    po[9] = torch.where(g["POC"] > 1e-5, g["POC"], po[1])
    is_h = col([v == 1 for v in z]) > 0
    po[2] = torch.where(is_h, zero, po[2])
    po[3] = torch.where(is_h, zero, po[3])

    # one-center two-electron integrals
    A, idx = _one_center_layout()
    A_t = torch.as_tensor(A, dtype=dtype, device=device)
    idx_t = torch.as_tensor(idx, device=device)
    table = torch.stack([zero] + [radial[name] for name in _RADIAL[1:]], dim=-1)  # (n, R)
    one_center = (A_t.unsqueeze(0) * table[:, idx_t]).sum(1)                        # (n,9,9,9,9)

    # energy of the isolated atom
    ios = col([el.OCC_S[v] for v in z])
    iop = col([el.OCC_P[v] for v in z])
    iod = col([el.OCC_D[v] for v in z])
    gssc = torch.clamp(ios - 1.0, min=0.0)
    kk = iop
    gspc = ios * kk
    ll = torch.minimum(kk, 6.0 - kk)
    gp2c = kk * (kk - 1.0) / 2.0 + 0.5 * ll * (ll - 1.0) / 2.0
    gppc = -0.5 * ll * (ll - 1.0) / 2.0
    hspc = -kk * ios * 0.5
    eisol = (g["USS"] * ios + g["UPP"] * iop + g["UDD"] * iod + gss * gssc + gpp * gppc
             + gsp * gspc + gp2 * gp2c + hsp * hspc)
    conf = torch.as_tensor([_D_CONFIG.get(v, (0, 0, 0, 0, 0)) for v in z], dtype=dtype,
                           device=device)
    eisol = eisol + (conf[:, 0] * radial["f0sd"] + conf[:, 1] * radial["f0dd"]
                     - conf[:, 2] * radial["g2sd"] / 5.0 - conf[:, 3] * radial["f2dd"] / 49.0
                     - conf[:, 4] * radial["f4dd"] / 49.0)
    eisol = torch.where(is_h, g["USS"], eisol)

    pqn = [(max(el.PQN_S[v], 1), max(el.PQN_P[v], 2), max(el.PQN_D[v], 3)) for v in z]
    return dict(
        z=z, n_orb=n_orb, pqn=pqn,
        zs=zs, zp=zp, zd=zd, zd_overlap=torch.clamp(zd, min=0.3),
        uss=g["USS"], upp=g["UPP"], udd=g["UDD"],
        betas=g["BETAS"], betap=g["BETAP"], betad=g["BETAD"],
        tore=col([el.CORE_CHARGE[v] for v in z]),
        dd=dd, qq=qq, rho0=rho0, rho1=rho1, rho2=rho2,
        ddp=torch.stack(ddp, dim=-1), po=torch.stack(po, dim=-1),
        one_center=one_center, eisol=eisol,
        radial=radial,
    )
