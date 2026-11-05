"""The geometry-dependent computation of an NDDO model in TorchScript-compatible form.

:class:`~.nddo.NDDO` and its exported form call these functions with the same tables: the plan of a batch, the
integral build and the core Hamiltonian, the core-core terms of each method, the starting density, the electron
counts and the energy outputs. Tables that depend only on the parameters and the elements (derived quantities,
pair tables, constants) are made outside.
"""
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .integrals import take0, take1
from .scf import Layout, electronic_energy, fock_matrices
from .twocenter import pair_kernel


def make_plan(z: Tensor, batch: Tensor, n_struct: int, lookup: Tensor, n_orb_el: Tensor,
              group_min_pairs: int) -> Tuple[Tensor, Tensor, int, Tensor, Tensor, Layout, Tensor, Tensor]:
    """Element indices (``lookup`` maps atomic numbers to them), orbital counts, the largest count ``k``, the
    pairs ``i < j`` of each structure, the basis layout, the orbitals each atom has and the real basis
    functions of a batch."""
    device = z.device
    ea = lookup[z]
    n_orb = n_orb_el[ea]
    k = int(n_orb.max())
    # pairs i < j of the same structure
    same = (batch.unsqueeze(0) == batch.unsqueeze(1)).triu(diagonal=1)
    nz = same.nonzero()
    pa, pb = nz[:, 0], nz[:, 1]
    # basis layout
    counts = torch.zeros(n_struct, dtype=torch.long, device=device).index_add(0, batch, n_orb)
    n_basis = int(counts.max())
    # atoms of a structure are contiguous (the collate convention)
    cum = torch.cumsum(n_orb, 0) - n_orb
    start = cum - (torch.cumsum(counts, 0) - counts)[batch]
    orb = torch.arange(k, device=device)
    has = orb.unsqueeze(0) < n_orb.unsqueeze(1)                          # (N, k)
    ao = torch.where(has, start.unsqueeze(1) + orb.unsqueeze(0), torch.full_like(has, n_basis, dtype=torch.long))
    valid = torch.arange(n_basis, device=device).unsqueeze(0) < counts.unsqueeze(1)
    layout = Layout(n_struct, n_basis, k, batch, ao, pa, pb, valid, group_min_pairs)
    layout.pair_groups()
    return ea, n_orb, k, pa, pb, layout, has, valid


def table_key(ka: int, kb: int, general: bool) -> str:
    """The key of the pair tables of ``ka`` and ``kb`` orbitals (``general``: with the d-orbital terms)."""
    return str(ka) + "/" + str(kb) + "/" + ("1" if general else "0")


def _pair(qt: Dict[str, Tensor], tables: Dict[str, Dict[str, Tensor]], c: Dict[str, Tensor], ea: Tensor,
          eb: Tensor, vec: Tensor, ka: int, kb: int, need_general: bool, padded: bool) -> Dict[str, Tensor]:
    general = need_general and max(ka, kb) == 9
    return pair_kernel(qt, tables[table_key(ka, kb, general)], c, ea, eb, vec, ka, kb, general, padded, None)


def padded_blocks(ints: Dict[str, Tensor], ka: int, kb: int, k: int) -> Dict[str, Tensor]:
    """Pair integrals of atoms with ``ka`` and ``kb`` orbitals zero-padded to ``k`` orbitals each."""
    da, db = k - ka, k - kb
    if da == 0 and db == 0:
        return ints
    out = dict(ints)
    out["S"] = torch.nn.functional.pad(ints["S"], [0, db, 0, da])
    out["VA"] = torch.nn.functional.pad(ints["VA"], [0, da, 0, da])
    out["VB"] = torch.nn.functional.pad(ints["VB"], [0, db, 0, db])
    out["W"] = torch.nn.functional.pad(ints["W"], [0, db, 0, db, 0, da, 0, da])
    return out


def feather(ints: Dict[str, Tensor], mask_a: Tensor, mask_b: Tensor, tore_a: Tensor, tore_b: Tensor,
            r_point: float = 7.0, width: float = 0.22) -> None:
    """Blend the two-center Coulomb integrals into point-charge values (in place).

    With ``c = 1 - exp(-width (R - r_point)^2)`` below ``r_point`` (Angstrom) and
    0 beyond, every integral becomes ``c`` times its NDDO value, and the
    Coulomb ones (``(mu mu|la la)``, the core attractions of the diagonal
    distributions and the core-core ``gamma``) gain ``(1 - c)`` times the
    point-charge value ``e^2 / R``: the integrals reach the exact
    point-charge limit at ``r_point``. Used by PM7 and PM6-ORG.
    """
    r = ints["r"]
    c = torch.where(r < r_point, 1.0 - torch.exp(-width * (r - r_point) ** 2), torch.zeros_like(r))
    point = 14.399645478456 / r                                          # COULOMB
    one = 1.0 - c
    eye_a = torch.diag_embed(mask_a)
    eye_b = torch.diag_embed(mask_b)
    ints["W"] = (c[:, None, None, None, None] * ints["W"]
                 + (one * point)[:, None, None, None, None] * eye_a[:, :, :, None, None]
                 * eye_b[:, None, None, :, :])
    ints["VA"] = c[:, None, None] * ints["VA"] - (one * point * tore_b)[:, None, None] * eye_a
    ints["VB"] = c[:, None, None] * ints["VB"] - (one * point * tore_a)[:, None, None] * eye_b
    ints["gab"] = c * ints["gab"] + one * point


def _diagonal_sum(W: Tensor, rows: List[int], cols: List[int], left_on_a: bool) -> Tensor:
    """Sum of (rr|cc) over the diagonals of two shells, rows on A or on B."""
    tot = torch.zeros_like(W[:, 0, 0, 0, 0])
    for i in rows:
        for j in cols:
            tot = tot + (W[:, i, i, j, j] if left_on_a else W[:, j, j, i, i])
    return tot


def _shift(W: Tensor, rows_a: List[int], cols_b: List[int], value: Tensor, mask: Tensor) -> None:
    add = torch.where(mask, value, torch.zeros_like(value))
    for i in rows_a:
        for j in cols_b:
            W[:, i, i, j, j] = W[:, i, i, j, j] + add


def _core_shift(V: Tensor, shell: List[int], mask: Tensor) -> None:
    tot = V[:, shell[0], shell[0]]
    for i in shell[1:]:
        tot = tot + V[:, i, i]
    avg = tot / len(shell)
    add = torch.where(mask, V[:, 0, 0] - avg, torch.zeros_like(avg))
    for i in shell:
        V[:, i, i] = V[:, i, i] + add


def pm7_balance(ints: Dict[str, Tensor], za: Tensor, zb: Tensor, n_orb_a: Tensor, n_orb_b: Tensor,
                occ_d: Tensor) -> None:
    """PM7's balance of the two-center integrals of transition metals (in place).

    For an atom X with ``d`` electrons in its reference configuration (``occ_d`` by atomic number) the average
    of the two-center Coulomb integrals between its ``d`` shell and the ``s``, ``p`` and ``d`` shells of the
    partner, and of the attraction of its ``d`` (and, the partner having ``p`` orbitals, its ``p``) electrons by
    the partner's core, are set to the ``(ss|ss)`` value used in the core-core term.
    """
    if ints["W"].shape[-1] < 9:
        return
    W, VA, VB = ints["W"].clone(), ints["VA"].clone(), ints["VB"].clone()
    ints["W"], ints["VA"], ints["VB"] = W, VA, VB
    s, p, d = [0], [1, 2, 3], [4, 5, 6, 7, 8]
    ss = W[:, 0, 0, 0, 0].clone()
    # the partner listed second in the reference program's pair loop is atom B
    x_has = occ_d[zb] > 0                   # d electrons on B
    y_has = occ_d[za] > 0                   # d electrons on A
    p_on_a, p_on_b = n_orb_a > 1, n_orb_b > 1
    # B's d shell with A's p and s shells; B's p and d core attraction
    m1 = x_has & p_on_a
    _shift(W, p, d, ss - _diagonal_sum(W, p, d, True) / 15.0, m1)
    _core_shift(VB, p, m1)
    _shift(W, s, d, ss - _diagonal_sum(W, s, d, True) / 5.0, x_has)
    _core_shift(VB, d, x_has)
    # A's d shell with B's d, p and s shells; A's p and d core attraction
    m2 = y_has & x_has
    _shift(W, d, d, ss - _diagonal_sum(W, d, d, True) / 25.0, m2)
    m3 = y_has & p_on_b
    _shift(W, d, p, ss - _diagonal_sum(W, d, p, True) / 15.0, m3)
    _core_shift(VA, p, m3)
    _shift(W, d, s, ss - _diagonal_sum(W, d, s, True) / 5.0, y_has)
    _core_shift(VA, d, y_has)


def build_integrals(qt: Dict[str, Tensor], tables: Dict[str, Dict[str, Tensor]], c: Dict[str, Tensor],
                    layout: Layout, positions: Tensor, ea: Tensor, z: Tensor, n_orb: Tensor, k: int, pa: Tensor,
                    pb: Tensor, has: Tensor, valid: Tensor, use_feather: bool, balance: bool, occ_d: Tensor,
                    shells: Tensor) -> Tuple[Dict[str, Tensor], Tensor, Tensor, Tensor, Tensor]:
    """The molecular-frame pair integrals, the core Hamiltonian ``(B, n, n)``, the one-center integrals
    ``(N, k, k, k, k)`` and the atom and pair blocks of the core Hamiltonian, from the positions."""
    dtype, device = positions.dtype, positions.device
    # gathers whose backward adds without synchronizing (capturable in a CUDA graph)
    vec = positions.gather(0, pb[:, None].expand(-1, 3)) - positions.gather(0, pa[:, None].expand(-1, 3))
    # each group of pairs at the orbital counts of its atoms, padded to k with zeros afterwards
    parts: List[Dict[str, Tensor]] = []
    order: List[Tensor] = []
    for ka, kb, rows, _, _ in layout.pair_groups():
        # in chunks: the multipole intermediates scale as pairs x components
        chunk = 20000 if max(ka, kb) < 9 else 1250                       # PAIR_CHUNK, PAIR_CHUNK // 16
        for s in range(0, int(rows.shape[0]), chunk):
            sub = rows[s:s + chunk]
            if layout.padded_pairs:
                # every block at k orbitals; the masks below clear the orbitals an atom does not have
                parts.append(_pair(qt, tables, c, ea[pa[sub]], ea[pb[sub]], vec[sub], k, k, k == 9, False))
            else:
                parts.append(padded_blocks(_pair(qt, tables, c, ea[pa[sub]], ea[pb[sub]], vec[sub], ka, kb,
                                                 k == 9, False), ka, kb, k))
            order.append(sub)
    if len(parts) == 0:
        parts = [padded_blocks(_pair(qt, tables, c, ea[:0], ea[:0], vec, k, k, k == 9, True), k, k, k)]
        order = [pa]
    inverse = torch.argsort(torch.cat(order))
    if len(order) == 1 and layout.padded_pairs:
        ints = parts[0]          # one group of all pairs in order
    else:
        ints: Dict[str, Tensor] = {}
        for key in parts[0].keys():
            ints[key] = torch.cat([p[key] for p in parts]).index_select(0, inverse)
    mask = has.to(dtype)
    if layout.padded_pairs:
        # the orbitals an atom does not have
        mA, mB = mask[pa], mask[pb]
        mAA, mBB = mA[:, :, None] * mA[:, None, :], mB[:, :, None] * mB[:, None, :]
        ints["S"] = ints["S"] * (mA[:, :, None] * mB[:, None, :])
        ints["VA"] = ints["VA"] * mAA
        ints["VB"] = ints["VB"] * mBB
        ints["W"] = ints["W"] * (mAA[:, :, :, None, None] * mBB[:, None, None, :, :])
    if use_feather:
        core = take0(qt["tore"], ea)
        feather(ints, mask[pa], mask[pb], take0(core, pa), take0(core, pb))
    if balance:
        pm7_balance(ints, z[pa], z[pb], n_orb[pa], n_orb[pb], occ_d)
    # core Hamiltonian
    u = take1(take0(torch.stack([qt["uss"], qt["upp"], qt["udd"]], -1), ea), shells) * mask
    beta = take1(take0(torch.stack([qt["betas"], qt["betap"], qt["betad"]], -1), ea), shells) * mask
    atom_blocks = torch.diag_embed(u)
    atom_blocks = atom_blocks.index_add(0, pa, ints["VA"]).index_add(0, pb, ints["VB"])
    pair_blocks = ints["S"] * 0.5 * (take0(beta, pa)[:, :, None] + take0(beta, pb)[:, None, :])
    H = layout.scatter(atom_blocks, pair_blocks)
    diag_pad = torch.where(valid, torch.zeros((), dtype=dtype, device=device),
                           torch.full((), 1.0e4, dtype=dtype, device=device))           # PAD_ENERGY
    H = H + torch.diag_embed(diag_pad)
    W1 = take0(qt["one_center"], ea)[:, :k, :k, :k, :k]
    W1 = (W1 * mask[:, :, None, None, None] * mask[:, None, :, None, None]
          * mask[:, None, None, :, None] * mask[:, None, None, None, :])
    return ints, H, W1, atom_blocks, pair_blocks


def _pair_is(za: Tensor, zb: Tensor, a: int, b: int) -> Tensor:
    lo, hi = torch.minimum(za, zb), torch.maximum(za, zb)
    return (lo == min(a, b)) & (hi == max(a, b))


def _pair_value(table: Tensor, za: Tensor, zb: Tensor) -> Tensor:
    """``table[za, zb]`` of a diatomic table (a row gather of the flattened table on a GPU)."""
    if table.is_cuda:
        return table.reshape(-1).index_select(0, za * table.shape[1] + zb)
    return table[za, zb]


def gaussians(params: Dict[str, Tensor], za: Tensor, zb: Tensor, r: Tensor, zz: Tensor, ks: List[int],
              strict: bool, active: Tensor) -> Tensor:
    """The core-core Gaussians ``FN1j``, ``FN2j``, ``FN3j`` (``j`` in ``ks``) of both atoms of each pair."""
    # all Gaussians of both atoms at once, columns (j, atom A), (j, atom B), ... in the order of the sum
    g: List[Tensor] = []
    for n in (1, 2, 3):
        t = torch.stack([params["FN" + str(n) + str(j)] for j in ks], -1)
        g.append(torch.stack([take0(t, za), take0(t, zb)], -1).reshape(r.shape[0], 2 * len(ks)))
    g1, g2, g3 = g[0], g[1], g[2]
    ax = g2 * (r.unsqueeze(-1) - g3) ** 2
    ok = (ax < 25.0) if strict else (ax <= 25.0)
    ok = ok & (g1.abs() > 0.0) & active.unsqueeze(-1)
    terms = torch.where(ok, (zz / r).unsqueeze(-1) * g1 * torch.exp(-torch.where(ok, ax, torch.zeros_like(ax))),
                        torch.zeros_like(ax))
    return terms.sum(-1)


def pm3_core_core(params: Dict[str, Tensor], za: Tensor, zb: Tensor, r: Tensor, gab: Tensor, tore_a: Tensor,
                  tore_b: Tensor) -> Tensor:
    """Core-core repulsion of each pair (eV) of PM3; ``r`` in Angstrom."""
    zz = tore_a * tore_b
    enuc = zz * gab
    x = _pair_value(params["XFAC"], za, zb)
    a = _pair_value(params["ALPB"], za, zb)
    has = x.abs() > 1e-5
    a = torch.where(a < 1e-6, torch.full_like(a, 1.2), a)
    e_pair = enuc * (1.0 + 2.0 * x * torch.exp(-a * r))
    alp = params["ALP"]
    eni, enj = torch.exp(-take0(alp, za) * r), torch.exp(-take0(alp, zb) * r)
    scale = eni + enj
    nt = za + zb
    nh = (nt == 8) | (nt == 9)
    scale = scale + torch.where(nh & ((za == 7) | (za == 8)), (r - 1.0) * eni, torch.zeros_like(r))
    scale = scale + torch.where(nh & ((zb == 7) | (zb == 8)), (r - 1.0) * enj, torch.zeros_like(r))
    e = torch.where(has, e_pair, (scale * enuc).abs() + enuc)
    return e + gaussians(params, za, zb, r, zz, [1, 2, 3, 4], False, ~(x > 1e-4))


def _pm6_scale(params: Dict[str, Tensor], za: Tensor, zb: Tensor, r: Tensor, x: Tensor, a: Tensor) -> Tensor:
    par = params["PAR"]
    scale = 1.0 + 2.0 * x * torch.exp(-a * (r + 0.0003 * r ** 6))
    hx = 1.0 + 2.0 * x * torch.exp(-a * r * r)
    scale = torch.where(_pair_is(za, zb, 1, 6) | _pair_is(za, zb, 1, 7), hx, scale)
    scale = torch.where(_pair_is(za, zb, 1, 8), hx - par[3] * torch.exp(-par[4] * r * 2.0), scale)
    scale = torch.where(_pair_is(za, zb, 6, 6), scale + par[1] * torch.exp(-par[2] * r), scale)
    scale = torch.where(_pair_is(za, zb, 8, 14), scale - 0.7e-3 * torch.exp(-(r - 2.9) ** 2), scale)
    return scale


def _steric(par: Tensor, r: Tensor, n1: int, n2: int, n3: int) -> Tensor:
    return 0.01 * par[n1] * torch.where(r - par[n3] > 0.0, torch.exp(-par[n2] * (r - par[n3]) ** 2),
                                        torch.ones_like(r))


def _org_scale(params: Dict[str, Tensor], za: Tensor, zb: Tensor, r: Tensor, x: Tensor, a: Tensor) -> Tensor:
    """PM6-ORG's scale factors with the Gaussian steric terms of its H, C, N, O and S pairs."""
    par = params["PAR"]
    base = 1.0 + 2.0 * x * torch.exp(-a * (r + 0.0003 * r ** 6))
    hx = 1.0 + 2.0 * x * torch.exp(-a * r * r)
    scale = base
    # only the forms of the element pairs present (one transfer)
    codes = torch.minimum(za, zb) * 256 + torch.maximum(za, zb)
    present = torch.jit.annotate(List[int], torch.unique(codes).tolist())
    # (code, form): the steric parameters (n1, n2, n3) of each pair, on base or on hx
    forms: List[Tuple[int, int, int, int, int]] = [
        (1 * 256 + 1, 0, 16, 17, 18), (1 * 256 + 6, 1, 19, 11, 12), (1 * 256 + 7, 1, 38, 39, 40),
        (1 * 256 + 8, 1, 3, 4, 5), (1 * 256 + 16, 0, 44, 45, 46), (6 * 256 + 6, 2, 13, 14, 15),
        (6 * 256 + 7, 0, 35, 36, 37), (6 * 256 + 8, 0, 20, 21, 22), (6 * 256 + 16, 0, 29, 30, 31),
        (7 * 256 + 8, 0, 26, 27, 28), (7 * 256 + 16, 0, 41, 42, 43), (8 * 256 + 8, 0, 32, 33, 34),
        (8 * 256 + 14, 3, 0, 0, 0), (8 * 256 + 16, 0, 23, 24, 25)]
    for code, kind, n1, n2, n3 in forms:
        if code not in present:
            continue
        if kind == 0:
            form = base + _steric(par, r, n1, n2, n3)
        elif kind == 1:
            form = hx + _steric(par, r, n1, n2, n3)
        elif kind == 2:
            form = base + par[1] * torch.exp(-par[2] * r) + _steric(par, r, n1, n2, n3)
        else:
            form = base - 0.7e-3 * torch.exp(-(r - 2.9) ** 2)
        scale = torch.where(codes == code, form, scale)
    return scale


def pm6_core_core(params: Dict[str, Tensor], za: Tensor, zb: Tensor, r: Tensor, gab: Tensor, tore_a: Tensor,
                  tore_b: Tensor, average_missing_pairs: bool, steric: bool) -> Tensor:
    """Core-core repulsion of each pair (eV) of PM6, PM7 (``average_missing_pairs``: like-pair means for
    missing diatomic parameters) and PM6-ORG (``steric``); ``r`` in Angstrom."""
    zz = tore_a * tore_b
    enuc = zz * gab
    x = _pair_value(params["XFAC"], za, zb)
    a = _pair_value(params["ALPB"], za, zb)
    if average_missing_pairs:
        missing = x.abs() < 1e-5
        X, A = params["XFAC"], params["ALPB"]
        x = torch.where(missing, 0.5 * (_pair_value(X, za, za) + _pair_value(X, zb, zb)), x)
        a = torch.where(missing, 0.5 * (_pair_value(A, za, za) + _pair_value(A, zb, zb)), a)
    has = x.abs() > 1e-5
    a = torch.where(a < 1e-6, torch.full_like(a, 1.2), a)
    scale = _org_scale(params, za, zb, r, x, a) if steric else _pm6_scale(params, za, zb, r, x, a)
    e_pair = enuc * scale
    lanth = ((za > 56) & (za < 72)) | ((zb > 56) & (zb < 72))
    generic = torch.where(lanth, 10.0 * torch.exp(-3.0 * r), 10.0 * torch.exp(-2.18 * r))
    e_generic = (generic * enuc).abs() + enuc
    e = torch.where(has, e_pair, e_generic)
    e = e + gaussians(params, za, zb, r, zz, [1], True, torch.ones_like(has))
    e = e + gaussians(params, za, zb, r, zz, [1, 2, 3, 4], False, ~has)
    ax = r / (za.to(r.dtype) ** 0.3333 + zb.to(r.dtype) ** 0.3333)
    core = torch.clamp(1e-8 / torch.where(ax < 3.0, ax, torch.ones_like(ax)) ** 12, max=1e5)
    return e + torch.where(ax < 3.0, core, torch.zeros_like(core))


def electron_counts(tore: Tensor, batch: Tensor, n_struct: int, charge: Tensor, multiplicity: Optional[Tensor],
                    unrestricted: bool) -> Tuple[Tensor, Tensor, Tensor, bool]:
    """The electrons of each structure and spin (``n_e``, ``n_alpha``, ``n_beta``) and whether the calculation
    is unrestricted (requested, or an open shell)."""
    dtype, device = tore.dtype, tore.device
    n_elec = torch.zeros(n_struct, dtype=dtype, device=device).index_add(0, batch, tore.detach()) - charge
    n_e = torch.round(n_elec).long()
    if multiplicity is not None:
        mult = torch.round(multiplicity.to(dtype)).long()
    else:
        mult = 1 + (n_e % 2)
    if bool(((n_e + mult - 1) % 2 != 0).any()):
        raise ValueError("spin multiplicity inconsistent with the electron count")
    n_alpha = (n_e + mult - 1) // 2
    n_beta = (n_e - mult + 1) // 2
    uhf = unrestricted or bool((mult != 1).any())
    return n_e, n_alpha, n_beta, uhf


def initial_guess(occ_ref: Tensor, ea: Tensor, shells: Tensor, has: Tensor, batch: Tensor, n_struct: int,
                  n_e: Tensor, layout: Layout, n_pairs: int, k: int) -> Tensor:
    """The starting density: reference occupations (``occ_ref`` per element and shell) spread over each shell
    and scaled to the electron count of each structure."""
    dtype, device = occ_ref.dtype, occ_ref.device
    diag = occ_ref[ea][:, shells] * has.to(dtype)
    tot = torch.zeros(n_struct, dtype=dtype, device=device).index_add(0, batch, diag.sum(-1))
    scale = torch.where(tot > 0, n_e.to(dtype) / tot.clamp(min=1e-12), torch.zeros_like(tot))
    diag = diag * scale[batch].unsqueeze(-1)
    return layout.scatter(torch.diag_embed(diag), torch.zeros((n_pairs, k, k), dtype=dtype, device=device))


def energy_outputs(layout: Layout, H: Tensor, W1: Tensor, W2: Tensor, P: Tensor, uhf: bool, e_core_pair: Tensor,
                   pa: Tensor, pb: Tensor, batch: Tensor, n_struct: int, eisol: Tensor, atom_heat: Tensor,
                   corr: Tensor) -> Dict[str, Tensor]:
    """Heats of formation and energies from the converged densities ``P`` (differentiable in the integrals):
    the electronic energy partitioned over the atoms, the core-core terms, the isolated-atom energies
    ``eisol`` (eV), the atomic heats of formation ``atom_heat`` and the corrections ``corr`` (kcal/mol)."""
    dtype, device = H.dtype, H.device
    F = fock_matrices(layout, H, W1, W2, P, uhf)
    e_el = electronic_energy(H, F, P, uhf)
    if uhf:
        e_mat = 0.5 * ((P[0] + P[1]) * H + (P * F).sum(0))
    else:
        e_mat = 0.5 * P[0] * (H + F[0])
    # per-atom partition of the electronic energy
    e_atom = layout.gather_atom(e_mat).sum((-1, -2))
    if pa.shape[0] > 0:
        e_pair = layout.gather_pair(e_mat).sum((-1, -2))
        e_atom = e_atom.index_add(0, pa, e_pair).index_add(0, pb, e_pair)
        e_atom = e_atom.index_add(0, pa, 0.5 * e_core_pair).index_add(0, pb, 0.5 * e_core_pair)
    ev_to_kcal = 23.06054783061903                                        # EV_TO_KCAL
    node_kcal = (e_atom - eisol) * ev_to_kcal + atom_heat + corr
    node_energy = node_kcal / ev_to_kcal
    hof = torch.zeros(n_struct, dtype=dtype, device=device).index_add(0, batch, node_kcal)
    e_core = torch.zeros(n_struct, dtype=dtype, device=device)
    if pa.shape[0] > 0:
        e_core = e_core.index_add(0, batch[pa], e_core_pair)
    out: Dict[str, Tensor] = {
        "node_energy": node_energy,
        "energy": hof / ev_to_kcal,
        "heat_of_formation": hof,
        "total_energy": e_el + e_core,
        "electronic_energy": e_el,
        "core_core_energy": e_core,
    }
    return out


def charges_and_dipole(layout: Layout, P: Tensor, uhf: bool, tore: Tensor, k: int, dfac: Tensor,
                       sp_mask: Tensor, pd_mask: Tensor, d_sp: Tensor, d_pd: Tensor, n_orb: Tensor, mass: Tensor,
                       batch: Tensor, n_struct: int, charge: Tensor, pos: Tensor) -> Tuple[Tensor, Tensor]:
    """Mulliken charges and the dipole moments (e Angstrom; about the center of mass of a charged structure):
    the point charges and the one-center hybrid dipoles (``dfac`` ``(3, 9, 9)``, the ``sp`` and ``pd`` pair
    masks ``(9, 9)``, the element separations ``d_sp`` and ``d_pd`` per atom)."""
    dtype, device = P.dtype, P.device
    P_tot = P.sum(0) if uhf else P[0]
    P_atom = layout.gather_atom(P_tot)
    charges = tore - P_atom.diagonal(dim1=-2, dim2=-1).sum(-1)
    fac = dfac[:, :k, :k].unsqueeze(0) * (sp_mask[:k, :k] * d_sp[:, None, None]
                                          + pd_mask[:k, :k] * d_pd[:, None, None]).unsqueeze(1)
    has_p = (n_orb > 1).to(dtype)
    hybrid = -0.529177210903 * torch.einsum("aij,acij->ac", P_atom, fac) * has_p[:, None]        # BOHR
    msum = torch.zeros(n_struct, dtype=dtype, device=device).index_add(0, batch, mass)
    com = torch.zeros(n_struct, 3, dtype=dtype, device=device).index_add(
        0, batch, mass[:, None] * pos) / msum[:, None]
    charged = (charge.abs() > 0.5).to(dtype)
    ref_point = com * charged[:, None]
    dip_atom = charges[:, None] * (pos - ref_point[batch]) + hybrid
    dipole = torch.zeros(n_struct, 3, dtype=dtype, device=device).index_add(0, batch, dip_atom)
    return charges, dipole
