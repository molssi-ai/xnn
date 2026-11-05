"""The gradient of the atom-centered symmetry functions, written out (HDNNP on CPUs).

The reference (:class:`~xnn.dnn.featurizers.acsf.AtomCenteredSymmetryFunctions`) gets the forces from an
autograd pass through its radial and angular blocks. Here, with the networks' gradient ``lam = dE / dG``
of every atom and symmetry function, the derivative of ``sum(lam * G)`` in the edge vectors is formed in one
pass over the edges and triplets: each function's slope in its distances and angle, as the reference's
formulas give them, contracted with ``lam`` of its center. The function types 1, 2, 3 and 9 (the factorized
angular functions) with any cutoff kind; evaluation only. :func:`forward_direct` serves a standalone 2G HDNNP
on a CPU (see :class:`~xnn.common.models.outputs.ForceStressOutput`).
"""
from __future__ import annotations

import functools
import math
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor

_TORCH = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])


def cutoff_slope(r: Tensor, kind: str, r_cut: float, r_inner: float, exponent: int) -> Tuple[Tensor, Tensor]:
    """:func:`~xnn.dnn.featurizers.acsf.cutoff_function` and its slope in ``r``."""
    inside = r < r_cut
    zero = torch.zeros_like(r)
    if kind == "hard":
        return inside.to(r.dtype), zero
    if kind == "polynomial":
        base = 1.0 - (r / r_cut) ** 2
        val = torch.ones_like(r)
        slope = torch.zeros_like(r)
        for _ in range(int(exponent)):
            slope = slope * base + val * (-2.0 * r / (r_cut * r_cut))
            val = val * base
        return torch.where(inside, val, zero), torch.where(inside, slope, zero)
    width = r_cut - r_inner
    x = ((r - r_inner) / width).clamp(min=0.0, max=1.0)
    # the clamp passes the slope where r_inner <= r <= r_cut, as its autograd does
    dx = torch.where((r >= r_inner) & (r <= r_cut), torch.full_like(r, 1.0 / width), zero)
    if kind == "cosine":
        val = 0.5 * (torch.cos(math.pi * x) + 1.0)
        slope = -0.5 * math.pi * torch.sin(math.pi * x) * dx
    elif kind == "tanh":
        t = torch.tanh(1.0 - x)
        val = t ** 3
        slope = -3.0 * t * t * (1.0 - t * t) * dx
    else:
        y = 1.0 - x
        num, den = y * (27.0 + y * y), 27.0 + 9.0 * y * y
        u = num / den
        du = ((27.0 + 3.0 * y * y) * den - num * 18.0 * y) / (den * den)
        val = u ** 3
        slope = -3.0 * u * u * du * dx
    return torch.where(inside, val, zero), torch.where(inside, slope, zero)


def _cutoffs(feat, r: Tensor) -> Tuple[Tensor, Tensor]:
    """Every declared cutoff function at ``r`` and its slope, stacked on a new last axis."""
    values: List[Tensor] = []
    slopes: List[Tensor] = []
    for i in range(len(feat._cut_kind)):
        v, s = cutoff_slope(r, feat._cut_kind[i], feat._cut_r[i], feat._cut_inner[i], feat._cut_exponent[i])
        values.append(v)
        slopes.append(s)
    return torch.stack(values, dim=-1), torch.stack(slopes, dim=-1)


def radial_slopes(feat, r: Tensor, sp: Tensor, src: Tensor, dst: Tensor, lam: Tensor) -> Tensor:
    """``d sum(lam G) / dr`` of the radial functions (types 1 and 2), per edge ``(E,)``; ``lam`` is
    ``(N D,)``."""
    dtype = r.dtype
    rows = feat._rad_lookup[sp[dst], sp[src]]                          # (E, K)
    ok = rows >= 0
    rows = rows.clamp(min=0)
    fc, dfc = _cutoffs(feat, r)
    if feat._single_cutoff:
        f, df = fc, dfc
    else:
        pick = feat._p_cutoff[rows]
        f, df = torch.gather(fc, 1, pick), torch.gather(dfc, 1, pick)
    power = feat._p_power.to(dtype)[rows]
    slope1 = power * torch.pow(f, power - 1.0) * df
    if feat._has_type2:
        eta = feat._p_eta.to(dtype)[rows]
        d = r.unsqueeze(-1) - feat._p_rs.to(dtype)[rows]
        slope = torch.exp(-eta * d * d) * (df - 2.0 * eta * d * f)
        if feat._has_type1:
            slope = torch.where(feat._p_type[rows] == 1, slope1, slope)
    else:
        slope = slope1
    w = lam[dst.unsqueeze(-1) * feat._dim + feat._p_column[rows]]
    return torch.where(ok, w * slope, torch.zeros_like(slope)).sum(-1)


def _lam_tables(feat, sp: Tensor, lam: Tensor) -> Tensor:
    """``lam`` of the factorized angular functions by atom, neighbor species pair, angular and radial factor,
    ``(N, S^2, nA + 1, nR)`` (the extra angular factor: the empty slots, zero)."""
    n, n_sp = sp.shape[0], feat._n_sp
    pairs = torch.arange(n_sp * n_sp, device=sp.device)
    bucket = sp.clamp(min=0).unsqueeze(-1) * (n_sp * n_sp) + pairs                         # (N, S^2)
    ia, ir, col = feat._ang_ia[bucket], feat._ang_ir[bucket], feat._ang_col[bucket]      # (N, S^2, K)
    atom = torch.arange(n, device=sp.device)[:, None, None].expand_as(ia)
    pair = pairs[None, :, None].expand_as(ia)
    values = lam.reshape(n, feat._dim)[atom, col]
    out = lam.new_zeros((n, n_sp * n_sp, feat._n_factor_a + 1, feat._n_factor_r))
    return out.index_put((atom, pair, ia, ir), values, accumulate=True)


def _constants(feat) -> Tuple:
    """The factorized angular functions' parameters as Python numbers (constants of the compiled code): per
    angular factor lambda, zeta and 2^(1 - zeta), per radial factor eta, its type and its cutoff."""
    cached = feat.__dict__.get("_direct_constants")
    if cached is None:
        cached = (tuple(float(x) for x in feat._a_lambda), tuple(float(x) for x in feat._a_zeta),
                  tuple(float(x) for x in feat._a_pref), tuple(float(x) for x in feat._r_eta),
                  tuple(bool(x) for x in feat._r_type3), tuple(int(x) for x in feat._r_cutoff))
        feat.__dict__["_direct_constants"] = cached
    return cached


def _power(base: Tensor, k: int) -> Tensor:
    """``base ** k`` for an integer ``k >= 0`` by repeated multiplication."""
    out = torch.ones_like(base)
    while k:
        if k & 1:
            out = out * base
        k >>= 1
        if k:
            base = base * base
    return out


def _edge_factors(feat, r: Tensor) -> Tuple[Tensor, Tensor]:
    """Per edge and radial factor ``exp(-eta r^2) fc(r)`` and its slope in ``r``, ``(E, nR)`` each."""
    _, _, _, etas, _, cuts = _constants(feat)
    fc, dfc = _cutoffs(feat, r)
    h, dh = [], []
    for eta, c in zip(etas, cuts):
        g = torch.exp(-eta * r * r)
        h.append(g * fc[:, c])
        dh.append(g * (dfc[:, c] - 2.0 * eta * r * fc[:, c]))
    return torch.stack(h, -1), torch.stack(dh, -1)


def _triplet_factors(feat, vec: Tensor, r: Tensor, he: Tensor, dhe: Tensor, e1: Tensor, e2: Tensor):
    """Per triplet the angular factors ``2^(1-zeta) (1 + lambda cos)^zeta`` (an extra zero factor for the empty
    slots) and their slopes in ``cos``, the radial factors (the edges' ``exp(-eta r^2) fc(r)``, times the third
    side's for type 3) and their slopes in the three sides, with the geometry they need."""
    lams, zetas, prefs, etas, type3, cuts = _constants(feat)
    v1, v2 = vec[e1], vec[e2]
    r1, r2 = r[e1], r[e2]
    d12 = v1 - v2
    r3 = torch.linalg.norm(d12, dim=-1)
    inv12 = 1.0 / (r1 * r2)
    cos = (v1 * v2).sum(-1) * inv12
    a, da = [], []
    for lam, zeta, pref in zip(lams, zetas, prefs):
        raw = 1.0 + lam * cos
        base = raw.clamp(min=0.0)
        if zeta == int(zeta) and 1 <= zeta <= 64:
            lower = _power(base, int(zeta) - 1)
            a.append(pref * lower * base)
            slope = (pref * zeta * lam) * lower
        else:
            value = pref * torch.pow(base, zeta)
            a.append(value)
            safe = torch.where(base > 0.0, base, torch.ones_like(base))
            slope = torch.where(base > 0.0, value * zeta / safe, torch.zeros_like(base)) * lam
        # zero below the clamp, as its autograd gives it
        da.append(torch.where(raw >= 0.0, slope, torch.zeros_like(slope)))
    zero = torch.zeros_like(cos)
    a_fac, a_slope = torch.stack(a + [zero], -1), torch.stack(da + [zero], -1)
    h1, h2, g1, g2 = he[e1], he[e2], dhe[e1], dhe[e2]
    if any(type3):
        fc3, dfc3 = _cutoffs(feat, r3)
        h3, g3 = [], []
        for eta, t3, c in zip(etas, type3, cuts):
            if t3:
                gauss = torch.exp(-eta * r3 * r3)
                h3.append(gauss * fc3[:, c])
                g3.append(gauss * (dfc3[:, c] - 2.0 * eta * r3 * fc3[:, c]))
            else:
                h3.append(torch.ones_like(r3))
                g3.append(torch.zeros_like(r3))
        h3, g3 = torch.stack(h3, -1), torch.stack(g3, -1)
    else:
        h3, g3 = torch.ones_like(h1), torch.zeros_like(h1)
    h12 = h1 * h2
    return (v1, v2, r1, r2, d12, r3, inv12, cos, a_fac, a_slope, h12 * h3, g1 * h2 * h3, h1 * g2 * h3, h12 * g3)


def angular_table(feat, vec: Tensor, r: Tensor, he: Tensor, dhe: Tensor, sp: Tensor, src: Tensor, e1: Tensor,
                  e2: Tensor, center: Tensor, n: int) -> Tensor:
    """Every triplet's products of angular and radial factors summed per center and neighbor species pair,
    ``(n S^2, (nA + 1) nR)``."""
    n_sp = feat._n_sp
    factors = _triplet_factors(feat, vec, r, he, dhe, e1, e2)
    a_fac, r_fac = factors[8], factors[10]
    pair = sp[src[e1]].clamp(min=0) * n_sp + sp[src[e2]].clamp(min=0)
    na, nr = a_fac.shape[-1], r_fac.shape[-1]
    return vec.new_zeros((n * n_sp * n_sp, na * nr)).index_add(
        0, center * (n_sp * n_sp) + pair, (a_fac.unsqueeze(-1) * r_fac.unsqueeze(1)).reshape(-1, na * nr))


def angular_columns(feat, table: Tensor, sp: Tensor, n: int) -> Tensor:
    """The factorized angular functions (types 3 and 9) of every atom ``(n D,)`` from :func:`angular_table`:
    each function reads its factors' product of its center and neighbor species pair."""
    n_sp = feat._n_sp
    table = table.reshape(n, n_sp * n_sp, feat._n_factor_a + 1, feat._n_factor_r)
    pairs = torch.arange(n_sp * n_sp, device=sp.device)
    bucket = sp.clamp(min=0).unsqueeze(-1) * (n_sp * n_sp) + pairs                         # (N, S^2)
    ia, ir, col = feat._ang_ia[bucket], feat._ang_ir[bucket], feat._ang_col[bucket]      # (N, S^2, K)
    atom = torch.arange(n, device=sp.device)[:, None, None].expand_as(ia)
    values = table[atom, pairs[None, :, None].expand_as(ia), ia, ir]
    return table.new_zeros(n * feat._dim).index_add(0, (atom * feat._dim + col).reshape(-1), values.reshape(-1))


def angular_gradients(feat, vec: Tensor, r: Tensor, he: Tensor, dhe: Tensor, sp: Tensor, src: Tensor, e1: Tensor,
                      e2: Tensor, center: Tensor, lam: Tensor) -> Tuple[Tensor, Tensor]:
    """``d sum(lam G) / d v`` of the factorized angular functions (types 3 and 9) for the first and second
    edge vector of every triplet, ``(T, 3)`` each."""
    (v1, v2, r1, r2, d12, r3, inv12, cos, a_fac, a_slope, r_fac, dr1, dr2, dr3) = _triplet_factors(
        feat, vec, r, he, dhe, e1, e2)
    s_1, s_2 = sp[src[e1]].clamp(min=0), sp[src[e2]].clamp(min=0)
    # lam of every function as a (angular factor, radial factor) table of its center and neighbor species pair
    # (the same for all triplets of a center and pair), one row read per triplet
    m = _lam_tables(feat, sp, lam)[center, s_1 * feat._n_sp + s_2]                      # (T, nA + 1, nR)
    s_cos = ((m * r_fac.unsqueeze(1)).sum(-1) * a_slope).sum(-1)
    wr = (m * a_fac.unsqueeze(-1)).sum(1)                                               # (T, nR)
    s_r1, s_r2, s_r3 = (wr * dr1).sum(-1), (wr * dr2).sum(-1), (wr * dr3).sum(-1)
    u12 = d12 / r3.unsqueeze(-1)
    k1 = (s_cos * inv12).unsqueeze(-1)
    grad1 = (k1 * v2 + (s_r1 / r1 - s_cos * cos / (r1 * r1)).unsqueeze(-1) * v1 + s_r3.unsqueeze(-1) * u12)
    grad2 = (k1 * v1 + (s_r2 / r2 - s_cos * cos / (r2 * r2)).unsqueeze(-1) * v2 - s_r3.unsqueeze(-1) * u12)
    return grad1, grad2


def edge_gradient(feat, vec: Tensor, sp: Tensor, src: Tensor, dst: Tensor, e1: Tensor, e2: Tensor,
                  center: Tensor, lam: Tensor, angular: bool) -> Tensor:
    """``d sum(lam G) / d vec`` per edge ``(E, 3)``: the radial functions' and, with ``angular``, the angular
    functions' of the triplets ``(e1, e2)``."""
    r = torch.linalg.norm(vec, dim=-1)
    grad = torch.zeros_like(vec)
    if feat._rad_lookup.shape[-1] > 0:
        grad = grad + (radial_slopes(feat, r, sp, src, dst, lam) / r).unsqueeze(-1) * vec
    if angular:
        he, dhe = _edge_factors(feat, r)
        g1, g2 = angular_gradients(feat, vec, r, he, dhe, sp, src, e1, e2, center, lam)
        grad = grad.index_add(0, e1, g1).index_add(0, e2, g2)
    return grad


def angular_raw(feat, vec: Tensor, sp: Tensor, src: Tensor, e1: Tensor, e2: Tensor, center: Tensor,
                n: int) -> Tensor:
    """:func:`angular_table` from the edge vectors."""
    r = torch.linalg.norm(vec, dim=-1)
    he, dhe = _edge_factors(feat, r)
    return angular_table(feat, vec, r, he, dhe, sp, src, e1, e2, center, n)


def _triplet_rows(order: Tensor, offsets: Tensor, pair_start: Tensor, pair_end: Tensor, total: int, vec: Tensor,
                  reach: float):
    """The triplets of :func:`~xnn.common.models.ops.build_triplets` (same order) from the edges grouped by
    center (``order``, each center's first row ``offsets``, its pairs from ``pair_start`` to ``pair_end``), with
    whether each triplet's third side is shorter than ``reach``: pair ``p`` of a center is ``(a, b)`` with
    ``p = b (b - 1) / 2 + a``, ``a < b``."""
    p = torch.arange(total, device=order.device)
    center = torch.bucketize(p, pair_end, right=True)
    local = p - pair_start[center]
    b = torch.floor((1.0 + torch.sqrt(1.0 + 8.0 * local.to(torch.float64))) * 0.5).to(torch.long)
    # exact integer b despite rounding: b (b - 1) / 2 <= local < b (b + 1) / 2
    b = b - (b * (b - 1) // 2 > local).to(torch.long)
    b = b + ((b + 1) * b // 2 <= local).to(torch.long)
    a = local - b * (b - 1) // 2
    first, second = order[offsets[center] + a], order[offsets[center] + b]
    d12 = vec[first] - vec[second]
    return first, second, center, (d12 * d12).sum(-1) < reach * reach


@functools.lru_cache(maxsize=None)
def _triplets_compiled() -> Callable:
    return torch.compile(_triplet_rows, dynamic=True)


def triplets(edge_index: Tensor, n: int, vec: Tensor, reach: Optional[float]) -> Tuple[Tensor, Tensor, Tensor]:
    """The triplets of the edges ``edge_index`` (rows of ``vec``), as :func:`~xnn.common.models.ops.build_triplets`
    gives them, generated in a compiled call; with ``reach``, only those whose third side is shorter."""
    dst = edge_index[1]
    order = torch.argsort(dst, stable=True)
    counts = torch.bincount(dst, minlength=n)
    pairs = counts * (counts - 1) // 2
    pair_end = torch.cumsum(pairs, 0)
    total = int(pair_end[-1]) if n > 0 else 0
    if total == 0:
        empty = torch.zeros(0, dtype=torch.long, device=dst.device)
        return empty, empty, empty
    offsets = torch.cumsum(counts, 0) - counts
    first, second, center, near = _triplets_compiled()(order, offsets, pair_end - pairs, pair_end, total, vec,
                                                       reach if reach is not None else float("inf"))
    if reach is None:
        return first, second, center
    keep = torch.nonzero(near).squeeze(-1)
    return first[keep], second[keep], center[keep]


def _third_side_reach(feat) -> Optional[float]:
    """The largest cutoff of the third side when every factorized angular function is of type 3 (cut off in all
    three sides), else ``None``."""
    _, _, _, _, type3, cuts = _constants(feat)
    if not type3 or not all(type3):
        return None
    return max(feat._cut_r[c] for c in cuts)


def applies(feat) -> bool:
    """Whether the written-out gradient covers ``feat`` (no type 8) and ``torch.compile`` produces CPU code."""
    from xnn.common.models.dispersion_fast.atm_cpu import available
    return feat._dim > 0 and not feat._has_type8 and available()


@functools.lru_cache(maxsize=None)
def _compiled(name: str) -> Callable:
    # 256-bit vectors (inductor's 512-bit code with its gathers ran 20-30% slower on AVX-512 AMD CPUs; torch 2.5
    # takes no simdlen option); the gradient's slot sums of every triplet in registers (inductor stores a
    # reduction's input of many reads)
    options = {"cpp.simdlen": 256} if _TORCH >= (2, 7) else {}
    if name == "gradient":
        return torch.compile(edge_gradient, dynamic=True, options={**options, "realize_acc_reads_threshold": 1 << 30})
    return torch.compile(angular_raw, dynamic=True, options=options)


def _run(name: str, feat, *args) -> Tensor:
    """:func:`edge_gradient` or :func:`angular_raw` compiled, else (where the compiler fails) eager."""
    if not feat.__dict__.get("_direct_eager", False):
        try:
            return _compiled(name)(feat, *args)
        except Exception:  # noqa: BLE001  (a compiler failure: the eager pass gives the same numbers)
            feat.__dict__["_direct_eager"] = True
    return {"gradient": edge_gradient, "angular": angular_raw}[name](feat, *args)


def forward_direct(model, data) -> Dict[str, Tensor]:
    """A 2G :class:`~xnn.dnn.models.hdnnp.HDNNP` on a CPU with its forces and ``"virial"`` (``dE / d(strain)``)
    from the written-out gradient: the symmetry functions on the compiled blocks without autograd, the networks'
    gradient ``dE / dG`` by autograd, and :func:`edge_gradient` compiled. Evaluation only."""
    feat = model.featurizer
    z, batch = data.atomic_numbers, data.batch
    n = int(z.shape[0])
    src, dst = data.edge_index[0], data.edge_index[1]
    with torch.no_grad():
        vec = data.edge_vectors()
        r = torch.linalg.norm(vec, dim=-1)
        sp = feat._z2i[z]
        raw = vec.new_zeros(n * feat._dim)
        if feat._rad_lookup.shape[-1] > 0:
            raw = raw + feat._radial_fast(r, sp, src, dst, n)
        empty = torch.zeros(0, dtype=torch.long, device=vec.device)
        e1, e2, center = empty, empty, empty
        if feat._ang_lookup.shape[-1] > 0:
            eidx = (r < feat._ang_cutoff).nonzero().squeeze(-1)
            # with every angular function cut off in the third side, the triplets beyond it add exact zeros
            e1, e2, center = triplets(data.edge_index[:, eidx], n, vec[eidx], _third_side_reach(feat))
            if center.numel() > 0:
                e1, e2 = eidx[e1], eidx[e2]
                raw = raw + angular_columns(feat, _run("angular", feat, vec, sp, src, e1, e2, center, n), sp, n)
    with torch.enable_grad():
        g = raw.reshape(n, feat._dim).requires_grad_(True)
        desc = feat.scale(g, z)
        short = model.energy_scale.to(desc.dtype) * model.element_nets(desc, z, None)
        short = short + model._self_energies_by_z.to(desc.dtype)[z]
        lam, = torch.autograd.grad(short.sum(), g)
    with torch.no_grad():
        grad = _run("gradient", feat, vec, sp, src, dst, e1, e2, center, lam.reshape(-1), bool(center.numel() > 0))
        forces = vec.new_zeros((n, 3)).index_add(0, dst, -grad).index_add(0, src, grad)
        virial = vec.new_zeros((data.num_graphs, 3, 3)).index_add(0, batch[dst], vec[:, :, None] * grad[:, None, :])
        virial = 0.5 * (virial + virial.transpose(1, 2))
        node = short.detach()
        energy = model.aggregate_energy(node, data)
    pdtype = data.pos.dtype
    return {"node_energy_short": node, "energy_short": energy, "node_energy": node, "energy": energy,
            "node_features": desc.detach(), "forces": forces.to(pdtype), "virial": virial.to(pdtype)}
