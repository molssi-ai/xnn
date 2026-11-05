"""Two-center integrals of atom pairs: overlaps, electron repulsion, core attraction.

All integrals of a pair are formed in the diatomic frame (``z`` from atom A
to atom B) and rotated to the molecular frame:

* the ``sp``-``sp`` electron-repulsion integrals with the MNDO multipole
  model (monopole, dipole ``dd``, linear quadrupoles ``2 qq`` and square
  quadrupoles ``qq`` with the additive terms ``am``, ``ad``, ``aq``);
  ``(p_x p_y | p_x p_y)`` follows from rotational invariance, ``((xx|xx) -
  (xx|yy)) / 2``;
* every integral that involves a ``d`` orbital with the Thiel-Voityuk scheme
  (multipoles up to ``L = 2`` of each charge distribution, separations
  ``ddp`` and additive terms ``po``);
* the attraction between the electrons of one atom and the core of the
  other, the core being a monopole with additive term ``po9``.
"""
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from .angular import L_OF, LOCAL, M_OF, MULTIPOLES, ORDERS, multipole_coefficient, pair_type
from .integrals import (
    SPD_CONFIG,
    _a_integrals,
    _b_integrals,
    _config_pairs,
    _overlap_poly,
    config_kernel,
    constants,
    local_frames,
    overlap_local_batched,
    series_order,
    take0,
    take1,
)

# per-atom sources of the charge separations and additive terms
D_SOURCES = ("none", "dd", "2qq", "qq", "ddp2", "ddp3", "ddp4", "ddp5", "ddp6", "sq2ddp3")
RHO_SOURCES = ("rho0", "rho1", "rho2") + tuple(f"po{i}" for i in range(1, 10))
_DDP_OF = {"sp": "ddp2", "pp": "ddp3", "sd": "ddp4", "pd": "ddp5", "dd": "ddp6"}
_PO_OF = {"sp": "po2", "pp": "po3", "sd": "po4", "pd": "po5", "dd": "po6"}
_MONO_PO_OF = {"ss": "po1", "pp": "po7", "dd": "po8"}
_D_TYPES = ("sd", "pd", "dd")


def _coef(entries) -> np.ndarray:
    c = np.zeros((9, 9))
    for (i, j), v in entries:
        c[i, j] = v
    return c


@lru_cache(maxsize=None)
def slot_tables():
    """The multipole components ("slots") of an atom and the interacting slot pairs.

    Returns a dict of numpy tables: for each of ``ee_classic``, ``ee_general``,
    ``core_classic``, ``core_general`` the configuration names, the
    separation and additive-term sources and the ``(S, 9, 9)`` coefficients
    over the local orbital pairs; plus the ``(a, b)`` slot pairs whose
    interaction is needed.
    """
    s, z, x, y = 0, 1, 2, 3
    cls = [  # name, config, D source, rho source, entries
        ("Q", "Q", "none", "rho0", [((s, s), 1), ((z, z), 1), ((x, x), 1), ((y, y), 1)]),
        ("Dz", "Dz", "dd", "rho1", [((s, z), 1), ((z, s), 1)]),
        ("Dx", "Dx", "dd", "rho1", [((s, x), 1), ((x, s), 1)]),
        ("Dy", "Dy", "dd", "rho1", [((s, y), 1), ((y, s), 1)]),
        ("Qzz", "Qzz", "2qq", "rho2", [((z, z), 1)]),
        ("Qxx", "Qxx", "2qq", "rho2", [((x, x), 1)]),
        ("Qyy", "Qyy", "2qq", "rho2", [((y, y), 1)]),
        ("Qxz", "Qxz", "qq", "rho2", [((z, x), 1), ((x, z), 1)]),
        ("Qyz", "Qyz", "qq", "rho2", [((z, y), 1), ((y, z), 1)]),
    ]
    groups = ({"Q", "Dz", "Qzz", "Qxx", "Qyy"}, {"Dx", "Qxz"}, {"Dy", "Qyz"})
    cls_pairs = [(a, b) for a in range(len(cls)) for b in range(len(cls))
                 if any(cls[a][0] in g and cls[b][0] in g for g in groups)]

    gen = []   # config, D source, rho source, coefficients, m, type
    for t in ("ss", "sp", "pp", "sd", "pd", "dd"):
        for L in ORDERS[t]:
            for comp in MULTIPOLES[L]:
                entries = []
                for i, a in enumerate(LOCAL):
                    for j, b in enumerate(LOCAL):
                        if pair_type(L_OF[a], L_OF[b]) != t:
                            continue
                        v = multipole_coefficient(a, b, comp)
                        if v != 0.0:
                            entries.append(((i, j), v))
                d_src = "none" if L == 0 else _DDP_OF[t]
                rho_src = _MONO_PO_OF[t] if L == 0 else _PO_OF[t]
                gen.append((SPD_CONFIG[comp], d_src, rho_src, _coef(entries), M_OF[comp], t))
    gen_pairs = [(a, b) for a in range(len(gen)) for b in range(len(gen))
                 if gen[a][4] == gen[b][4] and (gen[a][5] in _D_TYPES or gen[b][5] in _D_TYPES)]

    core_cls = [
        ("Q", "none", "po1", [((s, s), 1)]),
        ("Q", "none", "po7", [((z, z), 1), ((x, x), 1), ((y, y), 1)]),
        ("Dz", "ddp2", "po2", [((s, z), 1), ((z, s), 1)]),
        ("Qzz", "sq2ddp3", "po3", [((z, z), 1)]),
        ("Qxx", "sq2ddp3", "po3", [((x, x), 1)]),
        ("Qyy", "sq2ddp3", "po3", [((y, y), 1)]),
    ]
    core_gen = [g for g in gen if g[4] == 0 and g[5] in _D_TYPES]

    def pack(rows, coef_index, cfg_index=0, d_index=1, rho_index=2):
        return dict(
            config=[r[cfg_index] for r in rows],
            d_src=np.array([D_SOURCES.index(r[d_index]) for r in rows]),
            rho_src=np.array([RHO_SOURCES.index(r[rho_index]) for r in rows]),
            coef=np.stack([r[coef_index] if isinstance(r[coef_index], np.ndarray)
                           else _coef(r[coef_index]) for r in rows]),
        )

    return dict(
        ee_classic=pack(cls, 4, 1, 2, 3), ee_classic_pairs=np.array(cls_pairs),
        ee_general=pack(gen, 3, 0, 1, 2), ee_general_pairs=np.array(gen_pairs),
        core_classic=pack(core_cls, 3, 0, 1, 2), core_general=pack(core_gen, 3, 0, 1, 2),
    )


def atom_sources(q: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
    """Per-element separation ``(n, len(D_SOURCES))`` and additive-term
    ``(n, len(RHO_SOURCES))`` tables from :func:`~.derived.element_quantities`."""
    zero = torch.zeros_like(q["dd"])
    ddp, po = q["ddp"], q["po"]
    d = torch.stack([zero, q["dd"], 2.0 * q["qq"], q["qq"], ddp[:, 2], ddp[:, 3], ddp[:, 4],
                     ddp[:, 5], ddp[:, 6], 1.4142135623730951 * ddp[:, 3]], dim=-1)          # sqrt(2)
    rho = torch.stack([q["rho0"], q["rho1"], q["rho2"]] + [po[:, i] for i in range(1, 10)], dim=-1)
    return d, rho


@lru_cache(maxsize=None)
def _slot_orbitals(kind: str) -> np.ndarray:
    """The fewest orbitals (1, 4 or 9) an atom needs for each slot of ``kind`` to have a coefficient (10: none)."""
    c = np.abs(slot_tables()[kind]["coef"]) > 0
    return np.array([next((m for m in (1, 4, 9) if c[s, :m, :m].any()), 10) for s in range(len(c))])


def _slot_groups(ka: int, kb: int, oa: Optional[Tensor], ob: Optional[Tensor]):
    """The pairs by the orbital counts of their atoms: ``(counts a, counts b)`` and pair indices per distinct
    combination; ``(ka, kb)`` with all pairs when the counts are not given."""
    combos: List[Tuple[int, int]] = []
    rows: List[Optional[Tensor]] = []
    if oa is None or ob is None:
        combos.append((ka, kb))
        rows.append(None)
    else:
        key = oa * 100 + ob
        codes = torch.jit.annotate(List[int], torch.unique(key).tolist())
        for code in codes:
            combos.append((code // 100, code % 100))
            rows.append((key == code).nonzero().squeeze(-1))
    return combos, rows


def _ee_kernel(kind: str, tab: Dict[str, Tensor], ka: int, kb: int, dA: Tensor, rA: Tensor, dB: Tensor,
               rB: Tensor, r: Tensor, oa: Optional[Tensor], ob: Optional[Tensor]) -> Tensor:
    """Electron-repulsion integrals of one kind (``ee_classic``, ``ee_general``) in the diatomic frame
    ``(P, ka, ka, kb, kb)``, from the tables of :func:`pair_tables`."""
    da = take1(dA, tab[kind + "/d_a"])
    db = take1(dB, tab[kind + "/d_b"])
    rho = take1(rA, tab[kind + "/rho_a"]) + take1(rB, tab[kind + "/rho_b"])
    K = r.new_zeros((r.shape[0], tab[kind + "/d_a"].shape[0]))
    # per group of pairs (the orbital counts ``oa``, ``ob`` of their atoms) only the slot pairs both atoms have
    combos, rows_list = _slot_groups(ka, kb, oa, ob)
    for i in range(len(combos)):
        ga, gb = combos[i]
        name = kind + "/" + str(ga) + "/" + str(gb)
        if name + "/cols" not in tab:
            continue
        cols = tab[name + "/cols"]
        ua, ub, slot, per_slot = tab[name + "/ua"], tab[name + "/ub"], tab[name + "/s"], tab[name + "/per_slot"]
        rows = rows_list[i]
        if rows is None:
            Kg = config_kernel(ua, ub, slot, per_slot, take1(da, cols), take1(db, cols), r, take1(rho, cols))
            K = K.index_copy(1, cols, Kg)
        else:
            Kg = config_kernel(ua, ub, slot, per_slot, da[rows][:, cols], db[rows][:, cols], r[rows],
                               rho[rows][:, cols])
            where: List[Optional[Tensor]] = [rows[:, None], cols[None, :]]
            K = K.index_put(where, Kg)
    coef = tab[kind + "/coef/" + str(ka) + "/" + str(kb)]
    return torch.matmul(K, coef).reshape(-1, ka, ka, kb, kb)


@lru_cache(maxsize=None)
def _pair_coefficients(kind: str, pairs_key: str, ka: int, kb: int) -> np.ndarray:
    """The products ``coef[a]_ij coef[b]_kl`` of every slot pair ``(a, b)``, ``(S, ka * ka * kb * kb)``."""
    t = slot_tables()[kind]
    pairs = slot_tables()[pairs_key]
    ca = t["coef"][pairs[:, 0]][:, :ka, :ka]
    cb = t["coef"][pairs[:, 1]][:, :kb, :kb]
    return (ca[:, :, :, None, None] * cb[:, None, None, :, :]).reshape(len(pairs), -1)


def _core_kernel(kind: str, tab: Dict[str, Tensor], k: int, d_e: Tensor, r_e: Tensor, rho_core: Tensor,
                 r: Tensor, electrons_on_a: bool, oe: Optional[Tensor]) -> Tensor:
    """Attraction of the electrons of one atom by the other's core in the diatomic frame ``(P, k, k)``."""
    de = take1(d_e, tab[kind + "/d_src"])
    rho = take1(r_e, tab[kind + "/rho_src"]) + rho_core.unsqueeze(-1)
    K = r.new_zeros((r.shape[0], tab[kind + "/d_src"].shape[0]))
    # per group of pairs (the orbital count ``oe`` of the atom with the electrons) only the slots it has
    combos, rows_list = _slot_groups(k, 0, oe, None if oe is None else torch.zeros_like(oe))
    side = "/A" if electrons_on_a else "/B"
    for i in range(len(combos)):
        ke = combos[i][0]
        name = kind + "/" + str(ke)
        if name + "/cols" not in tab:
            continue
        cols = tab[name + "/cols"]
        rows = rows_list[i]
        if rows is None:
            dg, rg, rr = take1(de, cols), take1(rho, cols), r
        else:
            dg, rg, rr = de[rows][:, cols], rho[rows][:, cols], r[rows]
        zero = torch.zeros_like(dg)
        ua, ub = tab[name + side + "/ua"], tab[name + side + "/ub"]
        slot, per_slot = tab[name + side + "/s"], tab[name + side + "/per_slot"]
        if electrons_on_a:
            Kg = config_kernel(ua, ub, slot, per_slot, dg, zero, rr, rg)
        else:
            Kg = config_kernel(ua, ub, slot, per_slot, zero, dg, rr, rg)
        if rows is None:
            K = K.index_copy(1, cols, Kg)
        else:
            where: List[Optional[Tensor]] = [rows[:, None], cols[None, :]]
            K = K.index_put(where, Kg)
    return torch.einsum("ps,sij->pij", K, tab[kind + "/coef/" + str(k)])


def _overlap_scheme(tab: Dict[str, Tensor], ia: Tensor, ib: Tensor) -> Tensor:
    """Series truncation scheme of the overlaps of each pair (see
    :func:`~.integrals.series_order`): pairs of ``sp`` elements H-Cl (except He
    and Ne) use the scheme of their row combination, all others 0."""
    row_class = tab["row_class"]
    ra, rb = row_class[ia], row_class[ib]
    lo, hi = torch.minimum(ra, rb), torch.maximum(ra, rb)
    # highest auxiliary index of the row combination: H-H 2, H-2nd 3, H-3rd 4,
    # 2nd-2nd 4, 2nd-3rd 5, 3rd-3rd 6
    return torch.where(lo > 0, tab["scheme_table"][lo, hi], torch.zeros_like(lo))


def _row_class(z, n_orb) -> List[int]:
    out = []
    for zz, no in zip(z, n_orb):
        if zz > 17 or no > 4 or zz in (2, 10):
            out.append(0)
        else:
            out.append(1 if zz == 1 else (2 if zz <= 10 else 3))
    return out


# the largest (pairs x shell combinations x polynomial size) evaluated in one batch, above it per combination
_BATCHED_OVERLAP_SIZE = 1 << 21


@lru_cache(maxsize=64)
def _overlap_tables(pqn_a: tuple, pqn_b: tuple, shells: tuple):
    """The overlap polynomials of every element pair and shell combination ``(la, lb, |m|)``: the distinct
    coefficient tables ``(Q, D, D)`` (zero-padded) with their constants and principal quantum numbers, and the
    table index ``(n_a * n_b, U)`` of each element pair (row ``a * n_b + b``) and combination."""
    key, polys, const, n_a, n_b = {}, [], [], [], []
    ids = np.zeros((len(pqn_a) * len(pqn_b), len(shells)), dtype=np.int64)
    for ea, qa in enumerate(pqn_a):
        for eb, qb in enumerate(pqn_b):
            for u, (la, lb, m) in enumerate(shells):
                spec = (qa[la], la, qb[lb], lb, m)
                if spec not in key:
                    c, k = _overlap_poly(*spec)
                    key[spec] = len(polys)
                    polys.append(c)
                    const.append(k)
                    n_a.append(qa[la])
                    n_b.append(qb[lb])
                ids[ea * len(pqn_b) + eb, u] = key[spec]
    D = max(c.shape[0] for c in polys)
    table = np.zeros((len(polys), D, D))
    for i, c in enumerate(polys):
        table[i, :c.shape[0], :c.shape[1]] = c
    return table, np.array(const), np.array(n_a), np.array(n_b), ids


def _poly_specs(pqn: tuple, shells: tuple):
    """The distinct polynomials of :func:`_overlap_tables` in table order, with their shell combinations."""
    seen, out = set(), []
    for qa in pqn:
        for qb in pqn:
            for la, lb, m in shells:
                spec = (qa[la], la, qb[lb], lb, m)
                if spec not in seen:
                    seen.add(spec)
                    out.append((len(out), (la, lb, m)))
    return out


def _overlap_entries(ka: int, kb: int):
    """The orbital pairs ``(i, j, (la, lb, |m|))`` with a nonzero local overlap and their distinct shell
    combinations: the overlap depends on ``(la, lb, |m|)`` only, the +m and -m orbital pairs share their values."""
    entries = [(i, j, (L_OF[a], L_OF[b], abs(M_OF[a])))
               for i, a in enumerate(LOCAL[:ka]) for j, b in enumerate(LOCAL[:kb]) if M_OF[a] == M_OF[b]]
    return entries, list(dict.fromkeys(e[2] for e in entries))


def _zeta(qt: Dict[str, Tensor]) -> Tensor:
    z = torch.stack([qt["zs"], qt["zp"], qt["zd_overlap"]], dim=-1)
    # shells an element does not have get a harmless exponent (masked later)
    return torch.where(z > 0, z, torch.ones_like(z))


def _overlap_kernel(qt: Dict[str, Tensor], tab: Dict[str, Tensor], c: Dict[str, Tensor], ia: Tensor,
                    ib: Tensor, r: Tensor, ka: int, kb: int) -> Tensor:
    """Local-frame overlaps ``(P, ka, kb)`` of the pairs (element indices ``ia``, ``ib``), every (pair, shell
    combination) a row with its own polynomial, in batches of pairs."""
    P = r.shape[0]
    S = r.new_zeros((P, ka, kb))
    if P == 0:
        return S
    zeta = _zeta(qt)
    za, zb = take0(zeta, ia), take0(zeta, ib)
    scheme = _overlap_scheme(tab, ia, ib)
    name = "ov/" + str(ka) + "/" + str(kb)
    ids_t, polys_t, const_t = tab[name + "/ids"], tab[name + "/polys"], tab[name + "/const"]
    na_t, nb_t, la, lb = tab[name + "/na"], tab[name + "/nb"], tab[name + "/la"], tab[name + "/lb"]
    U, D = la.shape[0], polys_t.shape[-1]
    rows_per_batch = max(1, 2097152 // (U * D * D))                      # _BATCHED_OVERLAP_SIZE
    code = ia * tab["n_elements"].shape[0] + ib
    parts: List[Tensor] = []
    for start in range(0, P, rows_per_batch):
        stop = min(start + rows_per_batch, P)
        rid = ids_t[code[start:stop]]                                    # (p, U)
        p = rid.shape[0]
        parts.append(overlap_local_batched(
            polys_t[rid].reshape(p * U, D, D), const_t[rid].reshape(-1),
            na_t[rid].reshape(-1), nb_t[rid].reshape(-1), take1(za[start:stop], la).reshape(-1),
            take1(zb[start:stop], lb).reshape(-1),
            r[start:stop].repeat_interleave(U), scheme[start:stop].repeat_interleave(U), c).reshape(p, U))
    vals = parts[0] if len(parts) == 1 else torch.cat(parts)
    return S.reshape(P, ka * kb).index_copy(1, tab[name + "/pos"], take1(vals, tab[name + "/col"])).reshape(
        P, ka, kb)


def _overlap_poly_local(tab: Dict[str, Tensor], c: Dict[str, Tensor], name: str, pid: int, za: Tensor, zb: Tensor,
                        r: Tensor, scheme: Tensor) -> Tensor:
    """The overlaps of one polynomial ``pid`` of the tables (as :func:`~.integrals.overlap_local` evaluates them)."""
    da, db = int(tab[name + "/da"][pid]), int(tab[name + "/db"][pid])
    na, nb = int(tab[name + "/na"][pid]), int(tab[name + "/nb"][pid])
    cc = tab[name + "/polys"][pid, :da, :db].contiguous()
    const = float(tab[name + "/const"][pid])
    fact = c["sqrt_factorial_2n"]
    half = 0.5 * r
    a = _a_integrals(half * (za + zb), da)
    t = half * (za - zb)
    b = _b_integrals(t, db, series_order(t, scheme), c)
    s = (torch.matmul(a, cc) * b).sum(-1)
    norm_a = (2.0 * za) ** (na + 0.5) / float(fact[na])
    norm_b = (2.0 * zb) ** (nb + 0.5) / float(fact[nb])
    return const * norm_a * norm_b * half ** (na + nb + 1) * s


def _overlaps_host(qt: Dict[str, Tensor], tab: Dict[str, Tensor], c: Dict[str, Tensor], ia: Tensor, ib: Tensor,
                   r: Tensor, ka: int, kb: int) -> Tensor:
    """:func:`_overlap_kernel` per shell combination and polynomial, the element pairs present read on the host
    (the form of a CPU above :data:`_BATCHED_OVERLAP_SIZE`)."""
    P = r.shape[0]
    S = r.new_zeros((P, ka, kb))
    zeta = _zeta(qt)
    za, zb = take0(zeta, ia), take0(zeta, ib)
    scheme = _overlap_scheme(tab, ia, ib)
    code = ia * tab["n_elements"].shape[0] + ib
    name = "ov/" + str(ka) + "/" + str(kb)
    present = torch.unique(code)
    codes = torch.jit.annotate(List[int], present.tolist())
    la_t, lb_t, ids_t = tab[name + "/la"], tab[name + "/lb"], tab[name + "/ids"]
    pos = torch.jit.annotate(List[int], tab[name + "/pos"].tolist())
    col = torch.jit.annotate(List[int], tab[name + "/col"].tolist())
    # the overlap depends on (la, lb, |m|) only: the +m and -m orbital pairs share their values
    shared: Dict[int, Tensor] = {}
    rows = torch.arange(P, device=r.device)
    for e in range(len(pos)):
        u = col[e]
        if u not in shared:
            la, lb = int(la_t[u]), int(lb_t[u])
            # the element pairs present by their polynomial (their principal quantum numbers), in order
            pids = torch.jit.annotate(List[int], ids_t[present, u].tolist())
            order: List[int] = []
            members: Dict[int, List[int]] = {}
            for i in range(len(codes)):
                if pids[i] not in members:
                    order.append(pids[i])
                    members[pids[i]] = torch.jit.annotate(List[int], [])
                members[pids[i]].append(codes[i])
            if len(order) == 1:
                val = _overlap_poly_local(tab, c, name, order[0], za[:, la], zb[:, lb], r, scheme)
            else:
                val = r.new_zeros(P)
                for pid in order:
                    idx = torch.isin(code, torch.tensor(members[pid], device=r.device)).nonzero().squeeze(-1)
                    where: List[Optional[Tensor]] = [idx]
                    val = val.index_put(where, _overlap_poly_local(tab, c, name, pid, za[idx, la], zb[idx, lb],
                                                                   r[idx], scheme[idx]))
            shared[u] = val
        at: List[Optional[Tensor]] = [rows, torch.full((P,), pos[e] // kb, device=r.device),
                                      torch.full((P,), pos[e] % kb, device=r.device)]
        S = S.index_put(at, shared[u])
    return S


def pair_tables(q: Dict, ka: int, kb: int, general: bool, device, dtype) -> Dict[str, Tensor]:
    """The fixed tables of :func:`pair_kernel` for pairs of ``ka`` and ``kb`` orbitals (``general``: with the
    Thiel-Voityuk terms of ``d`` orbitals) of the elements of ``q``, on ``device`` (created once)."""
    key = ("pair_tables", tuple(q["z"]), tuple(q["n_orb"]), tuple(map(tuple, q["pqn"])), ka, kb, general,
           str(device), dtype)
    tab = _PAIR_TABLES.get(key)
    if tab is not None:
        return tab
    tab = {}
    slots = slot_tables()

    def put(name, values, dt=None):
        tab[name] = torch.as_tensor(values, dtype=dt, device=device)

    kinds = [("ee_classic", "ee_classic_pairs")] + ([("ee_general", "ee_general_pairs")] if general else [])
    for kind, pairs_key in kinds:
        t, pairs = slots[kind], slots[pairs_key]
        a, b = pairs[:, 0], pairs[:, 1]
        put(kind + "/d_a", t["d_src"][a])
        put(kind + "/d_b", t["d_src"][b])
        put(kind + "/rho_a", t["rho_src"][a])
        put(kind + "/rho_b", t["rho_src"][b])
        need = _slot_orbitals(kind)
        for ga in (1, 4, 9):
            for gb in (1, 4, 9):
                sel = np.nonzero((need[a] <= ga) & (need[b] <= gb))[0]
                if not len(sel):
                    continue
                name = f"{kind}/{ga}/{gb}"
                put(name + "/cols", sel)
                ua, ub, slot, per_slot = _config_pairs(tuple(t["config"][x] for x in a[sel]),
                                                       tuple(t["config"][x] for x in b[sel]))
                put(name + "/ua", ua, dtype)
                put(name + "/ub", ub, dtype)
                put(name + "/s", slot)
                put(name + "/per_slot", per_slot, dtype)
        put(f"{kind}/coef/{ka}/{kb}", _pair_coefficients(kind, pairs_key, ka, kb), dtype)
    for kind in ("core_classic",) + (("core_general",) if general else ()):
        t = slots[kind]
        put(kind + "/d_src", t["d_src"])
        put(kind + "/rho_src", t["rho_src"])
        need = _slot_orbitals(kind)
        for ke in (1, 4, 9):
            sel = np.nonzero(need <= ke)[0]
            if not len(sel):
                continue
            name = f"{kind}/{ke}"
            put(name + "/cols", sel)
            electrons, core = tuple(t["config"][x] for x in sel), ("Q",) * len(sel)
            for side, (na_, nb_) in (("/A", (electrons, core)), ("/B", (core, electrons))):
                ua, ub, slot, per_slot = _config_pairs(na_, nb_)
                put(name + side + "/ua", ua, dtype)
                put(name + side + "/ub", ub, dtype)
                put(name + side + "/s", slot)
                put(name + side + "/per_slot", per_slot, dtype)
        for k in sorted({ka, kb}):
            put(f"{kind}/coef/{k}", t["coef"][:, :k, :k], dtype)
    put("row_class", _row_class(q["z"], q["n_orb"]))
    put("scheme_table", [[0, 0, 0, 0], [0, 2, 3, 4], [0, 3, 4, 5], [0, 4, 5, 6]])
    put("n_orb", q["n_orb"])
    put("n_elements", np.zeros(len(q["z"]), dtype=np.int64))
    entries, shells = _overlap_entries(ka, kb)
    pqn = tuple(map(tuple, q["pqn"]))
    polys, const, n_a_tab, n_b_tab, ids = _overlap_tables(pqn, pqn, tuple(shells))
    name = f"ov/{ka}/{kb}"
    put(name + "/ids", ids)
    put(name + "/polys", polys, dtype)
    put(name + "/const", const, dtype)
    put(name + "/na", n_a_tab)
    put(name + "/nb", n_b_tab)
    put(name + "/la", [sh[0] for sh in shells])
    put(name + "/lb", [sh[1] for sh in shells])
    sizes = [_overlap_poly(int(n_a_tab[i]), la, int(n_b_tab[i]), lb, m)[0].shape
             for i, (la, lb, m) in _poly_specs(pqn, tuple(shells))]
    put(name + "/da", [d[0] for d in sizes])
    put(name + "/db", [d[1] for d in sizes])
    put(name + "/pos", [i * kb + j for i, j, _ in entries])
    put(name + "/col", [shells.index(e[2]) for e in entries])
    _PAIR_TABLES[key] = tab
    return tab


_PAIR_TABLES: Dict[tuple, Dict[str, Tensor]] = {}


def pair_integrals(q: Dict[str, Tensor], ea: Tensor, eb: Tensor, vec: Tensor, k: int,
                   need_general: bool, k_b: Optional[int] = None) -> Dict[str, Tensor]:
    """Molecular-frame integrals of atom pairs.

    Parameters
    ----------
    q : dict
        Derived element quantities (:func:`~.derived.element_quantities`).
    ea, eb : Tensor
        Element index (into ``q``) of atoms A and B of every pair ``(P,)``.
    vec : Tensor
        ``R_B - R_A`` in Angstrom ``(P, 3)``.
    k : int
        Orbitals per atom of the batch (1, 4 or 9; padded), of atom A when
        ``k_b`` is given.
    need_general : bool
        Whether any atom has ``d`` orbitals.
    k_b : int, optional
        Orbitals of atom B. With it, ``k`` and ``k_b`` are the orbital counts
        of the atoms of every pair; without it, the pairs are padded to ``k``.

    Returns
    -------
    dict
        ``S`` overlaps ``(P, ka, kb)``; ``W`` electron repulsion ``(mu nu | la si)``
        ``(P, ka, ka, kb, kb)`` with ``mu nu`` on A; ``VA`` / ``VB`` attraction of
        the electrons of A by the core of B ``(P, ka, ka)`` (and vice versa ``(P, kb, kb)``);
        ``gab`` the ``(s_A s_A | s_B s_B)`` core integral ``(P,)``; ``r``
        the distances in Angstrom. Energies in eV.
    """
    ka, kb = k, (k if k_b is None else k_b)
    general = need_general and max(ka, kb) == 9
    tab = pair_tables(q, ka, kb, general, vec.device, vec.dtype)
    qt = {key: v for key, v in q.items() if isinstance(v, Tensor)}
    return pair_kernel(qt, tab, constants(vec.device, vec.dtype), ea, eb, vec, ka, kb, general, k_b is None, None)


def pair_kernel(qt: Dict[str, Tensor], tab: Dict[str, Tensor], c: Dict[str, Tensor], ea: Tensor, eb: Tensor,
                vec: Tensor, ka: int, kb: int, general: bool, padded: bool,
                s_local: Optional[Tensor]) -> Dict[str, Tensor]:
    """:func:`pair_integrals` from the derived quantities' tensors ``qt``, the tables of :func:`pair_tables` and
    the constants of :func:`~.integrals.constants`; ``padded``: every pair at ``ka == kb`` orbitals with the
    atoms' own counts from the tables; ``s_local``: local overlaps computed outside (else here)."""
    r_ang = vec.norm(p=2, dim=-1)
    r = r_ang / 0.529177210903                                           # BOHR
    d_src, rho_src = atom_sources(qt)
    dA, rA, dB, rB = take0(d_src, ea), take0(rho_src, ea), take0(d_src, eb), take0(rho_src, eb)
    T = local_frames(vec, max(ka, kb), c)
    TA, TB = T[:, :ka, :ka], T[:, :kb, :kb]
    oa: Optional[Tensor] = None
    ob: Optional[Tensor] = None
    if padded:
        n_orb = tab["n_orb"]
        oa, ob = n_orb[ea], n_orb[eb]
    W = _ee_kernel("ee_classic", tab, ka, kb, dA, rA, dB, rB, r, oa, ob)
    if ka >= 4 and kb >= 4:
        hpp = 0.5 * (W[:, 2, 2, 2, 2] - W[:, 2, 2, 3, 3])
        W[:, 2, 3, 2, 3] = hpp
        W[:, 2, 3, 3, 2] = hpp
        W[:, 3, 2, 2, 3] = hpp
        W[:, 3, 2, 3, 2] = hpp
    if general:
        W = W + _ee_kernel("ee_general", tab, ka, kb, dA, rA, dB, rB, r, oa, ob)
    po9 = qt["po"][:, 9]
    po9A, po9B = take0(po9, ea), take0(po9, eb)
    tore_a, tore_b = take0(qt["tore"], ea), take0(qt["tore"], eb)
    VA = _core_kernel("core_classic", tab, ka, dA, rA, po9B, r, True, oa)
    VB = _core_kernel("core_classic", tab, kb, dB, rB, po9A, r, False, ob)
    if general and ka == 9:
        VA = VA + _core_kernel("core_general", tab, ka, dA, rA, po9B, r, True, oa)
    if general and kb == 9:
        VB = VB + _core_kernel("core_general", tab, kb, dB, rB, po9A, r, False, ob)
    VA = -tore_b[:, None, None] * VA
    VB = -tore_a[:, None, None] * VB
    gab = 27.211386245988 / torch.sqrt(r * r + (po9A + po9B) ** 2)      # HARTREE
    if s_local is not None:
        S = s_local
    else:
        name = "ov/" + str(ka) + "/" + str(kb)
        U, D = tab[name + "/la"].shape[0], tab[name + "/polys"].shape[-1]
        if vec.is_cuda or vec.shape[0] <= max(1, 2097152 // (U * D * D)):       # _BATCHED_OVERLAP_SIZE
            S = _overlap_kernel(qt, tab, c, ea, eb, r, ka, kb)
        else:
            # above the batch size a CPU evaluates the overlaps per shell combination
            S = _overlaps_host(qt, tab, c, ea, eb, r, ka, kb)
    S = torch.where((r_ang * r_ang < 225.0)[:, None, None], S, torch.zeros_like(S))
    # rotate to the molecular frame
    TAt, TBt = TA.transpose(-1, -2), TB.transpose(-1, -2)
    S = torch.matmul(TAt, torch.matmul(S, TB))
    VA = torch.matmul(TAt, torch.matmul(VA, TA))
    VB = torch.matmul(TBt, torch.matmul(VB, TB))
    W = torch.einsum("pai,pabcd->pibcd", TA, W)
    W = torch.einsum("pbj,pibcd->pijcd", TA, W)
    W = torch.einsum("pck,pijcd->pijkd", TB, W)
    W = torch.einsum("pdl,pijkd->pijkl", TB, W)
    out: Dict[str, Tensor] = {"S": S, "W": W, "VA": VA, "VB": VB, "gab": gab, "r": r_ang}
    return out
