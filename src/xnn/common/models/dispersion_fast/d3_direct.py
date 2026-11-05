"""D3 on CPUs with its forces and stress formed without autograd, in PyTorch.

The reference (:class:`~xnn.common.models.d3.DFTD3`) gets its forces and stress from an autograd
pass through every tensor operation of the evaluation. Here the derivatives are written out: the
coordination numbers with their slope per edge, the Gaussian reference weights with their slope
in the coordination number, the pair C6 with their derivatives in both coordination numbers, the
pair energies with their slope in the distance, and the three-body term from the compiled tiles of
:mod:`~xnn.common.models.dispersion_fast.atm_cpu` with their gradient
(:func:`~xnn.common.models.dispersion_fast.atm_cpu.three_body_direct`). The derivatives through the
C6 collect on the coordination numbers, which carry them to the edges; the edge gradient gives the
forces and the strain derivative (the stress). Evaluation only: the outputs carry no autograd graph.

Molecules up to ``DENSE_ATOMS`` atoms take the dense path (:func:`d3_dense`): the same derivatives over
every pair ``i < j`` and every triangle of atoms, in one call compiled with ``torch.compile``, which leaves no
per-call work in Python beyond the call itself.
"""
from __future__ import annotations

import functools
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import torch
from torch import Tensor

from .atm_cpu import Plan, _atm_slopes, _switch_slope, available, three_body_direct

#: Neighbors per tile of the three-body term for molecules (fastest at 81 and 375 atoms on one and eight threads).
DIRECT_TILE = 16
#: Neighbors per tile for cells (their centers have many neighbors, which fill large tiles).
CELL_TILE = 64
#: Most atoms of a molecule for the dense path (every pair and triangle of atoms in one compiled call).
DENSE_ATOMS = 2048
#: Most atoms of a batch of several molecules for the dense paths.
DENSE_BATCH_ATOMS = 256
#: Most atoms for the dense paths' serial variant (code for one thread: thread start-up would cost more than it
#: saves).
SERIAL_ATOMS = 48
#: Inductor options of the dense paths' compiled calls.
DENSE_OPTIONS = {"realize_acc_reads_threshold": 1 << 30, "cpp_wrapper": True}


def _damped(term, r: Tensor, r0: Tensor, rr: Tensor) -> Tuple[Tensor, Tensor]:
    """``s6 t6 + s8 rr t8`` of the term's damping and its slope in ``r``."""
    s6, s8, alp, rs6, rs8, bet = term.s6, term.s8, term.alp, term.rs6, term.rs8, term.bet
    r2 = r * r
    r6, r8 = r2 * r2 * r2, r2 * r2 * r2 * r2
    if term.damping == "bj":
        q6 = r0 ** 6
        t6, t8 = 1.0 / (r6 + q6), 1.0 / (r8 + q6 * r0 * r0)
        d6, d8 = -6.0 * r6 / r * t6 * t6, -8.0 * r8 / r * t8 * t8
    elif term.damping == "zero":
        f6, f8 = 6.0 * (rs6 * r0 / r) ** alp, 6.0 * (rs8 * r0 / r) ** (alp + 2.0)
        t6, t8 = 1.0 / (1.0 + f6) / r6, 1.0 / (1.0 + f8) / r8
        d6 = t6 * (alp * f6 / (1.0 + f6) - 6.0) / r
        d8 = t8 * ((alp + 2.0) * f8 / (1.0 + f8) - 8.0) / r
    elif term.damping == "mzero":
        u6, u8 = r / (rs6 * r0) + bet * r0, r / (rs8 * r0) + bet * r0
        f6, f8 = 6.0 * u6 ** (-alp), 6.0 * u8 ** (-(alp + 2.0))
        t6, t8 = 1.0 / (1.0 + f6) / r6, 1.0 / (1.0 + f8) / r8
        d6 = t6 * (alp * f6 / (u6 * rs6 * r0 * (1.0 + f6)) - 6.0 / r)
        d8 = t8 * ((alp + 2.0) * f8 / (u8 * rs8 * r0 * (1.0 + f8)) - 8.0 / r)
    else:
        rb, ab = r ** bet, r0 ** bet
        drb = bet * rb / r
        q6 = r0 ** 6
        q8 = q6 * r0 * r0
        den6, den8 = rb * r6 + ab * q6, rb * r8 + ab * q8
        t6, t8 = rb / den6, rb / den8
        d6 = (drb * den6 - rb * (drb * r6 + rb * 6.0 * r6 / r)) / (den6 * den6)
        d8 = (drb * den8 - rb * (drb * r8 + rb * 8.0 * r8 / r)) / (den8 * den8)
    return s6 * t6 + s8 * rr * t8, s6 * d6 + s8 * rr * d8


def _weights(term, z: Tensor, cn: Tensor) -> Tuple[Tensor, Tensor]:
    """The Gaussian reference weights ``(N, 7)`` and their derivatives in the coordination number."""
    return _reference_weights(term.refcn, term.nref, float(term.wf), z, cn)


def _reference_weights(refcn: Tensor, nref: Tensor, wf: float, z: Tensor, cn: Tensor) -> Tuple[Tensor, Tensor]:
    """:func:`_weights` from the tables."""
    refcn = refcn[:, z].t()
    valid = torch.arange(refcn.shape[1], device=z.device)[None, :] < nref[z][:, None]
    d = cn[:, None] - refcn
    gw = torch.where(valid, torch.exp(-wf * d * d), torch.zeros_like(d))
    dgw = -2.0 * wf * d * gw
    norm = gw.sum(1, keepdim=True)
    ok = norm > 0.0
    safe = torch.where(ok, norm, torch.ones_like(norm))
    w = gw / safe
    dw = (dgw - w * dgw.sum(1, keepdim=True)) / safe
    top = torch.where(valid, refcn, torch.full_like(refcn, -float("inf"))).max(dim=1, keepdim=True).values
    fallback = (valid & ((refcn - top).abs() < 1e-12)).to(cn.dtype)
    return torch.where(ok, w, fallback), torch.where(ok, dw, torch.zeros_like(dw))


@torch.no_grad()
def d3_direct(term, data):
    """D3 of a batched graph with its forces, no autograd: the energies per atom and per structure ``(B, 3)``
    (total, two-body, three-body; eV), the features ``(N, 2)`` (coordination number, homoatomic C6), the
    forces ``(N, 3)`` (eV / A) and ``dE / d(strain)`` ``(B, 3, 3)`` (eV). Molecules take :func:`d3_dense` where
    :func:`dense_applies` (its outputs instead)."""
    if dense_applies(data, DENSE_ATOMS):
        return d3_dense(term, data)
    b, hartree = term.bohr, term.hartree
    z, batch = data.atomic_numbers, data.batch
    n = z.shape[0]
    src, dst = data.edge_index[0], data.edge_index[1]
    vec_a = data.edge_vectors().detach().to(term.c6ref.dtype)
    vec = vec_a / b
    r = torch.linalg.norm(vec, dim=-1)
    zi, zj = z[dst], z[src]
    # coordination numbers and their slope per edge
    rc = term.rcov[zi] + term.rcov[zj]
    ex = torch.exp(-term.kcn * (rc / r - 1.0))
    in_cn = r <= term.cutoff_cn / b
    count = torch.where(in_cn, 1.0 / (1.0 + ex), torch.zeros_like(r))
    dcount = -count * count * ex * term.kcn * rc / (r * r)
    cn = r.new_zeros(n).index_add(0, dst, count)
    w, dw = _weights(term, z, cn)
    # pair C6 and their derivatives in the first and second coordination number, (N, N): C6_ij = <f_i, g_j> with
    # f_i[s, b] = sum_a W_ia C6ref[a, b, Z_i, Z_s] over the species s present and g_j[s, b] = W_jb at Z_j only
    species, spec = torch.unique(z, return_inverse=True)
    ref = term.c6ref[:, :, z][:, :, :, species]                       # (7, 7, N, S)
    f = torch.einsum("ia,abis->isb", w, ref).reshape(n, -1)
    df = torch.einsum("ia,abis->isb", dw, ref).reshape(n, -1)
    own = torch.nn.functional.one_hot(spec, species.shape[0]).to(w.dtype)[:, :, None]
    g = (own * w[:, None, :]).reshape(n, -1)
    dg = (own * dw[:, None, :]).reshape(n, -1)
    c6m, dc6m, dc6n = f @ g.t(), df @ g.t(), f @ dg.t()
    # two-body sum: every directed edge gives half its pair energy to its center
    c6 = c6m[dst, src]
    rr = 3.0 * term.r4r2[zi] * term.r4r2[zj]
    r0 = term.pair_radius(zi, zj)
    g, dg = _damped(term, r, r0, rr)
    in_pair = r <= term.cutoff_pair / b
    width = term.switch_width_pair / b
    if width > 0.0:
        s, ds = _switch_slope(r, term.cutoff_pair / b, width)
    else:
        s, ds = torch.ones_like(r), torch.zeros_like(r)
    s, ds = torch.where(in_pair, s, torch.zeros_like(s)), torch.where(in_pair, ds, torch.zeros_like(ds))
    e2 = r.new_zeros(n).index_add(0, dst, -0.5 * c6 * s * g)
    de_dr = -0.5 * c6 * (ds * g + s * dg)
    de_dc6 = -0.5 * s * g
    dcn = (r.new_zeros(n).index_add(0, dst, de_dc6 * dc6m[dst, src])
           .index_add(0, src, de_dc6 * dc6n[dst, src]))
    grad = (de_dr / r)[:, None] * vec
    e3 = torch.zeros_like(e2)
    s9 = float(term.s9)
    if s9 != 0.0:
        cutoff = term.cutoff_triple / b
        sel = torch.nonzero(r <= cutoff).squeeze(1)
        # small tiles: a center pairs only its neighbors of larger index, which fill large tiles poorly
        plan = Plan(data.edge_index[:, sel], n, tile=DIRECT_TILE if molecules(data) else CELL_TILE)
        rows = sel[plan.order]
        e3, g3, dcn3 = three_body_direct(z, plan, vec[rows].contiguous(), r[rows].contiguous(), c6m, dc6m, dc6n,
                                         term.rs9 * term.rvdw, s9, (term.alp + 2.0) / 3.0, cutoff,
                                         term.switch_width_triple / b, n)
        grad.index_add_(0, rows, g3)
        dcn = dcn + dcn3
    # the coordination-number chain, then Angstrom and eV
    grad = grad + (dcn[dst] * dcount / r)[:, None] * vec
    grad = grad * (hartree / b)
    forces = vec.new_zeros((n, 3)).index_add(0, dst, -grad).index_add(0, src, grad)
    strain = vec.new_zeros((data.num_graphs, 3, 3)).index_add(0, batch[dst], vec_a[:, :, None] * grad[:, None, :])
    strain = 0.5 * (strain + strain.transpose(1, 2))
    node = (e2 + e3) * hartree
    energies = vec.new_zeros((data.num_graphs, 3)).index_add(
        0, batch, torch.stack([node, e2 * hartree, e3 * hartree], dim=1))
    features = torch.stack([cn, c6m.diagonal()], dim=1)
    dtype, pdtype = data.model_dtype, data.pos.dtype
    return (node.to(dtype), energies.to(dtype), features.to(dtype), forces.to(pdtype), strain.to(pdtype))


def molecules(data) -> bool:
    """Whether no structure of ``data`` is a cell."""
    return not (data.cell is not None and data.pbc is not None
                and bool(((data.cell.abs().sum(dim=(1, 2)) > 1e-8) & data.pbc.any(-1).to(data.cell.device)).any()))


def dense_applies(data, limit: int) -> bool:
    """Whether the dense paths serve ``data``: one molecule of at most ``limit`` atoms or a batch of molecules of
    at most ``DENSE_BATCH_ATOMS`` (the pairs and triangles across its molecules are computed and dropped), and
    ``torch.compile`` produces CPU code here."""
    n = int(data.pos.shape[0])
    return (n <= (limit if data.num_graphs == 1 else min(limit, DENSE_BATCH_ATOMS)) and available()
            and molecules(data))


class _Setting(NamedTuple):
    """The constants of a D3 term for the dense path (bohr and hartree)."""
    damping: str
    s6: float
    s8: float
    s9: float
    a1: float
    a2: float
    alp: float
    rs6: float
    rs8: float
    rs9: float
    bet: float
    kcn: float
    wf: float
    bohr: float
    hartree: float
    cutoff_cn: float
    cutoff_pair: float
    width_pair: float
    cutoff_triple: float
    width_triple: float


#: The term's tensors and plain attributes behind the dense path's setting and tables.
_TENSORS = ("s6", "s8", "s9", "a1", "a2", "rs6", "rs8", "bet", "c6ref")
_PLAIN = ("damping", "cutoff_cn", "cutoff_pair", "switch_width_pair", "cutoff_triple", "switch_width_triple")


def _setting(term) -> _Setting:
    b = term.bohr
    return _Setting(str(term.damping), *(float(getattr(term, k)) for k in (
        "s6", "s8", "s9", "a1", "a2", "alp", "rs6", "rs8", "rs9", "bet", "kcn", "wf", "bohr", "hartree")),
        term.cutoff_cn / b, term.cutoff_pair / b, term.switch_width_pair / b, term.cutoff_triple / b,
        term.switch_width_triple / b)


def _triangles(s9: float, alp3: float, switched: bool, n: int, upper: Tensor, triples: Optional[Tensor], r2: Tensor,
               r: Tensor, s6p: Tensor, qp: Tensor, s3: Tensor, ds3: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """The three-body sums of the dense paths over the pairs ``upper`` (``i < j``, ``(2, P)``): per pair the
    energies of its triangles and their slope in its squared length, and per atom its share of the energy.
    The pairs' squared lengths, ``sqrt(C6)`` and damping factors (zero off the cutoff), switch and slope come
    per pair. With ``triples`` (the pairs of every triangle ``a < b < c``, ``(ab, ac, bc)``, and its atoms,
    ``(6, T)``) each triangle is evaluated once with the slopes of its three sides; else once per side, over
    the third atom of every pair (padded to whole vectors of 8: no C9 there)."""
    if triples is not None:
        pab, pac, pbc = triples[0], triples[1], triples[2]
        x, y, w = r2[pab], r2[pac], r2[pbc]
        c9 = s9 * s6p[pab] * s6p[pac] * s6p[pbc]
        tt = 6.0 * qp[pab] * qp[pac] * qp[pbc]
        e, ex, ey, ew = _atm_slopes(x, y, w, c9, tt, torch.rsqrt(x * y * w), alp3)
        if switched:
            sa, sb, sc = s3[pab], s3[pac], s3[pbc]
            da, db, dc = ds3[pab] / (2.0 * r[pab]), ds3[pac] / (2.0 * r[pac]), ds3[pbc] / (2.0 * r[pbc])
            ex, ey, ew = ex * sa * sb * sc + e * sb * sc * da, ey * sa * sb * sc + e * sa * sc * db, \
                ew * sa * sb * sc + e * sa * sb * dc
            e = e * sa * sb * sc
        tri = torch.zeros_like(r2).index_add(0, pab, e).index_add(0, pac, e).index_add(0, pbc, e)
        slope = torch.zeros_like(r2).index_add(0, pab, ex).index_add(0, pac, ey).index_add(0, pbc, ew)
        e3 = (r2.new_zeros(n).index_add(0, triples[3], e).index_add(0, triples[4], e)
              .index_add(0, triples[5], e)) / 3.0
        return tri, slope, e3
    i, j = upper[0], upper[1]
    both, kp = (torch.cat([i, j]), torch.cat([j, i])), (n + 7) // 8 * 8
    r2m = torch.ones((n, kp), dtype=r2.dtype, device=r2.device).index_put(both, torch.cat([r2, r2]))
    s6m = torch.zeros((n, kp), dtype=r2.dtype, device=r2.device).index_put(both, torch.cat([s6p, s6p]))
    qm = torch.zeros((n, kp), dtype=r2.dtype, device=r2.device).index_put(both, torch.cat([qp, qp]))
    # sides x = r_ij^2, y = r_ik^2, w = r_jk^2
    x, y, w = r2[:, None], r2m[i], r2m[j]
    c9 = s9 * s6p[:, None] * s6m[i] * s6m[j]
    tt = 6.0 * qp[:, None] * qm[i] * qm[j]
    e, ex, _, _ = _atm_slopes(x, y, w, c9, tt, torch.rsqrt(x * y * w), alp3)
    if switched:
        s3m = torch.zeros((n, kp), dtype=r2.dtype, device=r2.device).index_put(both, torch.cat([s3, s3]))
        rest = s3m[i] * s3m[j]
        ex = ex * s3[:, None] * rest + e * rest * (ds3 / (2.0 * r))[:, None]
        e = e * s3[:, None] * rest
    tri, slope = e.sum(1), ex.sum(1)
    return tri, slope, (r2.new_zeros(n).index_add(0, i, tri).index_add(0, j, tri)) / 6.0


def _dense_terms(st: _Setting, pos: Tensor, edge_index: Tensor, z: Tensor, upper: Tensor, triples: Optional[Tensor],
                 batch: Tensor, num_graphs: int, refcn: Tensor, nref: Tensor, c6ref: Tensor, rcov: Tensor,
                 r4r2: Tensor, rvdw: Tensor) -> Tuple[Tensor, ...]:
    """D3 of a batch of molecules over every pair ``upper`` (``i < j``, ``(2, P)``) and every triangle of atoms
    (``triples``: see :func:`_triangles`), as the outputs of :meth:`~xnn.common.models.d3.DFTD3.evaluate_direct`
    (eV, A): the energies per atom and per structure (total, two-body, three-body), the coordination numbers,
    the features ``(N, 2)``, the forces ``(N, 3)`` and ``dE / d(strain)`` ``(B, 3, 3)``. Pairs that are no edge of
    the graph are left out. ``c6ref`` is ordered ``(Z_i, Z_j, b, a)``."""
    dt = c6ref.dtype
    pos = pos.to(dt) / st.bohr
    n = pos.shape[0]
    i, j = upper[0], upper[1]
    adj = torch.zeros((n, n), dtype=torch.bool, device=pos.device).index_put(
        (edge_index[1], edge_index[0]), torch.ones_like(edge_index[0], dtype=torch.bool))[i, j]
    v = pos[i] - pos[j]
    r2 = torch.where(adj, (v * v).sum(-1), torch.ones((), dtype=dt))
    r = torch.sqrt(r2)
    zi, zj = z[i], z[j]
    zero = torch.zeros((), dtype=dt)
    # coordination numbers and their slope per pair
    rc = rcov[zi] + rcov[zj]
    ex = torch.exp(-st.kcn * (rc / r - 1.0))
    count = torch.where(adj & (r <= st.cutoff_cn), 1.0 / (1.0 + ex), zero)
    dcount = -count * count * ex * st.kcn * rc / r2
    cn = pos.new_zeros(n).index_add(0, i, count).index_add(0, j, count)
    w, dw = _reference_weights(refcn, nref, st.wf, z, cn)
    # pair C6 and their derivatives in the coordination numbers of the first and second atom
    ref = c6ref[zi, zj]                                                # (P, 7 (b), 7 (a))
    half = (w[i][:, None, :] * ref).sum(2)
    dhalf = (dw[i][:, None, :] * ref).sum(2)
    c6 = (half * w[j]).sum(1)
    dc6i, dc6j = (dhalf * w[j]).sum(1), (half * dw[j]).sum(1)
    # two-body: the energy, its slope in the distance (h) and in the C6 (p) of every pair
    rr = 3.0 * r4r2[zi] * r4r2[zj]
    r0 = st.a1 * torch.sqrt(rr) + st.a2 if st.damping in ("bj", "op") else rvdw[zi, zj]
    g, dg = _damped(st, r, r0, rr)
    in_pair = adj & (r <= st.cutoff_pair)
    if st.width_pair > 0.0:
        sw, dsw = _switch_slope(r, st.cutoff_pair, st.width_pair)
    else:
        sw, dsw = torch.ones_like(r), torch.zeros_like(r)
    sw, dsw = torch.where(in_pair, sw, zero), torch.where(in_pair, dsw, zero)
    e_pair = -0.5 * c6 * sw * g
    e2 = pos.new_zeros(n).index_add(0, i, e_pair).index_add(0, j, e_pair)
    h = -c6 * (dsw * g + sw * dg)
    p = -sw * g
    e3 = torch.zeros_like(e2)
    if st.s9 != 0.0:
        # three-body: sqrt(C6), the damping factors and the switch of every pair (zero off the cutoff)
        in3 = adj & (r <= st.cutoff_triple)
        s6p = torch.where(in3, torch.sqrt(c6), zero)
        qp = torch.where(in3, (st.rs9 * rvdw[zi, zj] / r) ** ((st.alp + 2.0) / 3.0), zero)
        if st.width_triple > 0.0:
            s3, ds3 = _switch_slope(r, st.cutoff_triple, st.width_triple)
            s3 = torch.where(in3, s3, zero)
        else:
            s3, ds3 = torch.ones_like(r), torch.zeros_like(r)
        tri, slope, e3 = _triangles(st.s9, (st.alp + 2.0) / 3.0, st.width_triple > 0.0, n, upper, triples, r2, r,
                                    s6p, qp, s3, ds3)
        h = h + 2.0 * r * slope
        p = p + torch.where(in3, 0.5 * tri / torch.where(in3, c6, torch.ones_like(c6)), zero)
    # the C6 chain into the coordination numbers, then into the distances
    dcn = pos.new_zeros(n).index_add(0, i, p * dc6i).index_add(0, j, p * dc6j)
    h = h + (dcn[i] + dcn[j]) * dcount
    gv = torch.where(adj, h / r, zero)[:, None] * v
    grad = pos.new_zeros((n, 3)).index_add(0, i, gv).index_add(0, j, -gv)
    hartree = st.hartree
    node = (e2 + e3) * hartree
    strain = node.new_zeros((num_graphs, 3, 3)).index_add(0, batch[i], (gv * hartree)[:, :, None] * v[:, None, :])
    c6_own = (w[:, None, :] * c6ref[z, z]).sum(2)
    energies = node.new_zeros((num_graphs, 3)).index_add(
        0, batch, torch.stack([node, e2 * hartree, e3 * hartree], dim=1))
    # in the order of _OUTPUTS (a tuple: the C++ wrapper returns no dict)
    return (node, energies[:, 0], energies[:, 1], energies[:, 2], cn,
            torch.stack([cn, (c6_own * w).sum(1)], dim=1), -grad * (hartree / st.bohr),
            0.5 * (strain + strain.transpose(1, 2)))


#: The outputs of the dense path's compiled call.
_OUTPUTS = ("node_energy", "energy", "energy_2body", "energy_3body", "coordination_numbers", "node_features",
            "forces", "virial")
#: The settings of the dense path by their key: the compiled call looks its setting up, so that its numbers are
#: constants of the compiled code (not inputs of it) and each setting compiles on its own.
_SETTINGS: Dict[str, _Setting] = {}


def _dense_entry(name: str, *args):
    return _dense_terms(_SETTINGS[name], *args)


@functools.lru_cache(maxsize=None)
def _dense_compiled(serial: bool) -> Callable:
    # the triangle energies and slopes summed in registers (inductor stores a reduction's input of many reads),
    # the buffers allocated in C++; above SERIAL_ATOMS parallel loops whatever the size of the compiling call
    # (inductor sizes its threads at compile time), else code for one thread (no parallel loop, and plain sums
    # where several threads would need atomic ones)
    options = {**DENSE_OPTIONS, **({"cpp.threads": 1} if serial else {"cpp.min_chunk_size": 1})}
    return torch.compile(_dense_entry, options=options)


@functools.lru_cache(maxsize=8)
def _upper(n: int) -> Tensor:
    return torch.triu_indices(n, n, 1)


@functools.lru_cache(maxsize=8)
def _triples(n: int) -> Tensor:
    """For every triangle ``a < b < c`` of ``n`` atoms the indices of its pairs in :func:`_upper` and its atoms,
    ``(6, T)``."""
    a, b, c = torch.combinations(torch.arange(n), 3).t()

    def pair(x, y):
        return x * n - x * (x + 1) // 2 + (y - x - 1)
    return torch.stack([pair(a, b), pair(a, c), pair(b, c), a, b, c])


def prepared(term, setting: Callable, tables: Callable, tensors: Tuple[str, ...], plain: Tuple[str, ...],
             registry: Dict[str, Any]) -> Tuple[str, Any]:
    """The key of ``setting(term)`` in ``registry`` and ``tables(term)`` of the dense path, cached on the term
    while its buffers or parameters ``tensors`` stay the same tensors at the same versions and its attributes
    ``plain`` the same."""
    buffers, parameters, attributes = term._buffers, term._parameters, term.__dict__
    current = tuple(buffers[k] if k in buffers else parameters[k] for k in tensors)
    key = (tuple(t._version for t in current), tuple(attributes[k] for k in plain))
    cached = attributes.get("_dense_prepared")
    if cached is not None and cached[0] == key and all(a is b for a, b in zip(cached[1], current)):
        return cached[2], cached[3]
    st = setting(term)
    name = repr(st)
    registry[name] = st
    tab = tables(term)
    attributes["_dense_prepared"] = (key, current, name, tab)
    return name, tab


def _c6_table(term) -> Tensor:
    return term.c6ref.permute(2, 3, 1, 0).contiguous()


def _cast(out: Dict[str, Tensor], data) -> Dict[str, Tensor]:
    """The dense outputs in the model's dtype (the forces and the strain derivative in the positions')."""
    dtype, pdtype = data.model_dtype, data.pos.dtype
    if out["node_energy"].dtype == dtype and out["forces"].dtype == pdtype:
        return out
    return {k: v.to(pdtype if k in ("forces", "virial") else dtype) for k, v in out.items()}


def d3_dense(term, data) -> Dict[str, Tensor]:
    """:func:`d3_direct` for a batch of molecules over every pair and triangle of atoms, in one compiled call: the
    outputs of :meth:`~xnn.common.models.d3.DFTD3.evaluate_direct`."""
    name, c6ref = prepared(term, _setting, _c6_table, _TENSORS, _PLAIN, _SETTINGS)
    # a fresh view of the atomic numbers: the tile path marks the batch's own tensor dynamic for its compiled calls
    n = int(data.pos.shape[0])
    serial = n <= SERIAL_ATOMS
    values = _dense_compiled(serial)(
        name, data.pos, data.edge_index, data.atomic_numbers.view(-1), _upper(n), _triples(n) if serial else None,
        data.batch,
        int(data.num_graphs), term.refcn, term.nref, c6ref, term.rcov, term.r4r2, term.rvdw)
    out = _cast(dict(zip(_OUTPUTS, values)), data)
    out["c6_matrix"] = out["node_energy"].new_zeros((0, 0))
    return out
