"""D4 on CPUs with its forces and stress formed without autograd, in PyTorch.

The reference (:class:`~xnn.common.models.d4.DFTD4`) gets its forces from an autograd pass through
every tensor operation of the evaluation. Here the derivatives are written out, as dftd4 does: the
two coordination numbers with their slope per edge, the EEQ charges of each molecule from the dense
linear system and, for the gradient, its adjoint solve, the dynamic polarizabilities with their
derivatives in the coordination number and the charge, the two-body energies with their slope in
the distance and in the polarizabilities, and the three-body term from the compiled tiles of
:mod:`~xnn.common.models.dispersion_fast.atm_cpu` with their gradient (C6 of the neutral atoms).
For a cell the EEQ matrix is the reference's Ewald sum; its derivative in the positions and the
cell, contracted with the adjoint and the charges, is the one autograd pass left, over that matrix
alone (the tail correction likewise over its own formula). Evaluation only: the outputs carry no
autograd graph.

Molecules up to ``DENSE_ATOMS`` atoms in the dense EEQ regime take the dense path (:func:`d4_dense`):
one EEQ system for the batch, the pairs ``i < j`` and the triangles of atoms, in one compiled call.
"""
from __future__ import annotations

import functools
import math
from typing import Callable, Dict, NamedTuple, Optional, Tuple

import torch
from torch import Tensor

from .atm_cpu import Plan, _switch_slope, three_body_direct
from .cpu import MAX_DENSE_ATOMS
from .d3_direct import (
    CELL_TILE,
    DENSE_OPTIONS,
    DIRECT_TILE,
    SERIAL_ATOMS,
    _cast,
    _triangles,
    _triples,
    _upper,
    molecules,
    prepared,
)
from .d3_direct import dense_applies as _dense_applies


def _eeq_solve(lu: Tensor, piv: Tensor, full: Tensor, rhs: Tensor) -> Tensor:
    """A solve of the EEQ system ``full`` (factor ``lu, piv``) as the reference makes it: the LU solution in float64,
    and in float32 the solution refined with float64 residuals (:func:`~xnn.common.models.eeq.lu_solve_implicit`),
    without which the charges of a few hundred atoms drift by 1e-4 e. The system is symmetric, so the adjoint
    solves take the same factor."""
    if full.dtype == torch.float32:
        from ..eeq import lu_solve_implicit
        return lu_solve_implicit(lu, piv, full, rhs, 1)
    return torch.linalg.lu_solve(lu, piv, rhs)

#: Most atoms of a molecule for the dense path (every pair and triangle of atoms in one compiled call).
DENSE_ATOMS = 2048
_TORCH = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])



def dense_applies(term, data) -> bool:
    """Whether :func:`d4_dense` serves ``data``: molecules of at most ``DENSE_ATOMS`` atoms in the dense EEQ
    regime, where ``torch.compile`` produces CPU code."""
    return _dense_applies(data, DENSE_ATOMS) and term.select_regime(int(data.pos.shape[0]), False) == "dense"


def _alpha(term, z: Tensor, cn: Tensor, q: Tensor, neutral: bool) -> Tuple[Tensor, ...]:
    """The dynamic polarizabilities ``(N, 23)`` at the charges ``q`` with their derivatives in each atom's
    coordination number and charge; with ``neutral``, also those of the neutral atoms with their
    coordination-number derivative."""
    return _alpha_of(term.refcn, term.ngw, term.nref, term.zeff, term.hardness, term.refq, term.refalpha,
                     float(term.wf), float(term.ga), float(term.gc), int(term.ngw[:, z].max()), z, cn, q, neutral)


def _alpha_of(refcn: Tensor, ngw: Tensor, nref: Tensor, zeff: Tensor, hardness: Tensor, refq: Tensor,
              refalpha: Tensor, wf: float, ga: float, gc: float, max_ngw: int, z: Tensor, cn: Tensor, q: Tensor,
              neutral: bool) -> Tuple[Optional[Tensor], ...]:
    """:func:`_alpha` from the tables (``max_ngw``: at least the largest Gaussian count of the atoms)."""
    refcn = refcn[:, z].t()
    ngw = ngw[:, z].t()
    valid = torch.arange(refcn.shape[1], device=z.device)[None, :] < nref[z][:, None]
    d = cn[:, None] - refcn
    j = torch.arange(1, max_ngw + 1, device=z.device, dtype=cn.dtype)
    use = (j[None, None, :] <= ngw[:, :, None]) & valid[:, :, None]
    gauss = torch.where(use, torch.exp(-wf * j * (d * d)[:, :, None]), torch.zeros_like(d)[:, :, None])
    gw = gauss.sum(-1)
    dgw = (gauss * (-2.0 * wf * j * d[:, :, None])).sum(-1)
    eps_norm = 1.0842021724855044e-19 if cn.dtype == torch.float32 else 1.4916681462400413e-154
    norm = gw.sum(1, keepdim=True)
    ok = norm > eps_norm
    safe = torch.where(ok, norm, torch.ones_like(norm))
    w = gw / safe
    dw = (dgw - w * dgw.sum(1, keepdim=True)) / safe
    top = torch.where(valid, refcn, torch.full_like(refcn, -float("inf"))).max(dim=1, keepdim=True).values
    fallback = (valid & ((refcn - top).abs() < 1e-12)).to(cn.dtype)
    w, dw = torch.where(ok, w, fallback), torch.where(ok, dw, torch.zeros_like(dw))
    # charge scaling (paper eqs 2-4) and its slope in the charge
    zi = zeff[z][:, None]
    gi = (hardness[z] * gc)[:, None]
    qref = refq[:, z].t() + zi

    def zeta(qmod):
        negative = qmod < 0.0
        safe_q = torch.where(negative, torch.ones_like(qmod), qmod)
        eu = torch.exp(gi * (1.0 - qref / safe_q))
        value = torch.exp(ga * (1.0 - eu))
        slope = value * (-ga * eu) * gi * qref / (safe_q * safe_q)
        return (torch.where(negative, torch.full_like(value, math.exp(ga)), value),
                torch.where(negative, torch.zeros_like(slope), slope))

    zq, dzq = zeta(q[:, None] + zi)
    refalpha = refalpha[:, :, z]                                          # (23, 7, N)
    alpha = torch.einsum("nr,krn->nk", w * zq, refalpha)
    dacn = torch.einsum("nr,krn->nk", dw * zq, refalpha)
    daq = torch.einsum("nr,krn->nk", w * dzq, refalpha)
    if not neutral:
        return alpha, dacn, daq, None, None
    z0, _ = zeta(zi.expand_as(qref))
    return (alpha, dacn, daq, torch.einsum("nr,krn->nk", w * z0, refalpha),
            torch.einsum("nr,krn->nk", dw * z0, refalpha))


def _large_charges(term, z: Tensor, pos: Tensor, vec: Tensor, in_eeq: Tensor, edge_index: Tensor, batch: Tensor,
                   members: Tensor, g: int, cell, x: Tensor, total: Tensor, q: Tensor):
    """Charges of one structure in the large EEQ regime (the reference's matrix-free system and its two refinement
    rounds), written into ``q``; returns the system, the structure's EEQ-range edges (global rows and local
    numbering) and its charges."""
    from ..d4 import _MIN_CUTOFF_EEQ_AU
    from ..eeq import EEQSystem, ewald_alpha, reciprocal_vectors
    b = term.bohr
    zm, pm = z[members], pos[members]
    rad = term.eeq_rad[zm]
    diag = term.eeq_eta[zm] + math.sqrt(2.0 / math.pi) / rad
    if cell is not None:
        if term.cutoff_eeq < _MIN_CUTOFF_EEQ_AU * b:
            raise ValueError(
                f"the large EEQ regime needs cutoff_eeq >= {_MIN_CUTOFF_EEQ_AU:.0f} bohr "
                f"({_MIN_CUTOFF_EEQ_AU * b:.1f} A) of neighbor list, got "
                f"{term.cutoff_eeq:.2f} A; raise cutoff_eeq or use regime='dense'")
        local = torch.full((z.shape[0],), -1, dtype=torch.long, device=z.device)
        local[members] = torch.arange(members.shape[0], device=z.device)
        sel = torch.nonzero(in_eeq & (batch[edge_index[1]] == g)).squeeze(1)
        local_edges = local[edge_index[:, sel]]
        alpha = ewald_alpha(term.cutoff_eeq / b)
        grid, gvec, gfac = reciprocal_vectors(cell, alpha)
        diag = diag - 2.0 * alpha / math.sqrt(math.pi)
        system = EEQSystem(diag, rad, pm, local_edges, vec[sel], alpha, gvec, gfac, grid, solver=term.eeq_solver)
    else:
        sel, local_edges = None, None
        system = EEQSystem(diag, rad, pm, solver=term.eeq_solver)
    xm = x[members]
    q0, mu0 = system.solve_augmented(xm, total, "charges")
    dq, dmu = system.solve_augmented(xm - system.matvec(q0) - mu0, total - q0.sum(), "residual")
    q1, mu1 = q0 + dq, mu0 + dmu
    corr, _ = system.solve_augmented(xm - system.matvec(q1) - mu1, total - q1.sum(), "residual")
    q[members] = q1 + corr
    return system, sel, local_edges, q1 + corr


@torch.no_grad()
def d4_direct(term, data):
    """D4 of a batch (molecules and cells of the dense EEQ regime) with its forces and stress: the energies per
    atom and per structure ``(B, 4)`` (total, two-body, three-body, tail; eV), the D4 coordination numbers, the
    EEQ charges, the dynamic polarizabilities ``(N, 23)``, the forces ``(N, 3)`` (eV / A) and ``dE / d(strain)``
    ``(B, 3, 3)`` (eV). Molecules take :func:`d4_dense` where :func:`dense_applies` (its outputs instead)."""
    if dense_applies(term, data):
        return d4_dense(term, data)
    b, hartree = term.bohr, term.hartree
    dt = term.refalpha.dtype
    z, batch = data.atomic_numbers, data.batch
    n = z.shape[0]
    src, dst = data.edge_index[0], data.edge_index[1]
    pos = data.pos.detach().to(dt) / b
    vec = data.edge_vectors().detach().to(dt) / b
    r = torch.linalg.norm(vec, dim=-1)
    zi, zj = z[dst], z[src]
    # coordination numbers: D4's (electronegativity-weighted) and the EEQ's (softly capped), slopes per edge
    rc = term.rcov[zi] + term.rcov[zj]
    u = -term.kcn * (r - rc) / rc
    count = 0.5 * (1.0 + torch.erf(u))
    dcount = -term.kcn / (rc * math.sqrt(math.pi)) * torch.exp(-u * u)
    en = term.k4 * torch.exp(-((term.en[zi] - term.en[zj]).abs() + term.k5) ** 2 / term.k6)
    in_cn = r <= term.cutoff_cn / b
    in_eeq = r <= term.cutoff_eeq_cn / b
    zero = torch.zeros_like(r)
    cn = r.new_zeros(n).index_add(0, dst, torch.where(in_cn, en * count, zero))
    raw = r.new_zeros(n).index_add(0, dst, torch.where(in_eeq, count, zero))
    ex = torch.exp(term.eeq_cn_max - raw)
    cn_eeq = math.log(1.0 + math.exp(term.eeq_cn_max)) - torch.log(1.0 + ex)
    dcap = ex / (1.0 + ex)
    # EEQ charges of every structure, in the reference's regime: dense (the factors kept for the adjoint solve; a
    # cell's Ewald matrix with its autograd graph in the positions and the cell) or large (the matrix-free system)
    charge = getattr(data, "total_charge", None)
    qtot = charge.detach().to(dt) if charge is not None else r.new_zeros(data.num_graphs)
    x = -term.eeq_chi[z] + term.eeq_kcnchi[z] * cn_eeq / torch.sqrt(cn_eeq + term.eeq_cn_reg)
    q = torch.empty_like(cn)
    cells = data.cell.detach().to(dt) / b if data.cell is not None else None
    in_eeq_all = r <= term.cutoff_eeq / b
    systems = []
    for g in range(data.num_graphs):
        members = torch.nonzero(batch == g).squeeze(1)
        m = members.shape[0]
        periodic = (cells is not None and data.pbc is not None and bool(data.pbc[g].any())
                    and bool(cells[g].abs().sum() > 1e-8))
        if term.select_regime(m, periodic) == "large":
            systems.append((g, members, periodic, "large",
                            _large_charges(term, z, pos, vec, in_eeq_all, data.edge_index, batch, members, g,
                                           cells[g] if periodic else None, x, qtot[g], q)))
            continue
        if periodic:
            with torch.enable_grad():
                leaves = (pos[members].clone().requires_grad_(True), cells[g].clone().requires_grad_(True))
                a_graph = term._coulomb_matrix_periodic(z[members], leaves[0], leaves[1])
            a = a_graph.detach()
        else:
            a = term._coulomb_matrix_molecular(z[members], pos[members])
        full = a.new_zeros((m + 1, m + 1))
        full[:m, :m] = a
        full[:m, m] = 1.0
        full[m, :m] = 1.0
        lu, piv = torch.linalg.lu_factor(full)
        rhs = torch.cat([x[members], qtot[g].reshape(1)]).unsqueeze(1)
        q[members] = _eeq_solve(lu, piv, full, rhs)[:m, 0]
        systems.append((g, members, periodic, "dense", (lu, piv, full, leaves + (a_graph,) if periodic else None)))
    neutral = float(term.s9) != 0.0
    alpha, dacn, daq, alpha0, dalpha0 = _alpha(term, z, cn, q, neutral)
    # two-body sum: every directed edge gives half its pair energy to its center
    cp = term.cp_weights
    k6 = 3.0 / math.pi
    # the pair C6 and the polarizability slopes as (N, N) products, read per edge (no per-edge 23-vectors), up to
    # the size of the three-body term's dense C6
    dense = n <= MAX_DENSE_ATOMS
    c6 = (k6 * (alpha * cp) @ alpha.t())[dst, src] if dense else k6 * (alpha[dst] * alpha[src] * cp).sum(-1)
    rr = 3.0 * term.r4r2[zi] * term.r4r2[zj]
    r0 = term.a1 * torch.sqrt(rr) + term.a2
    r2 = r * r
    r6, r8 = r2 * r2 * r2, r2 * r2 * r2 * r2
    q6 = r0 ** 6
    t6, t8 = 1.0 / (r6 + q6), 1.0 / (r8 + q6 * r0 * r0)
    gp = term.s6 * t6 + term.s8 * rr * t8
    dgp = term.s6 * (-6.0 * r6 / r * t6 * t6) + term.s8 * rr * (-8.0 * r8 / r * t8 * t8)
    in_pair = r <= term.cutoff_pair / b
    width = term.switch_width_pair / b
    if width > 0.0:
        s, ds = _switch_slope(r, term.cutoff_pair / b, width)
    else:
        s, ds = torch.ones_like(r), torch.zeros_like(r)
    s, ds = torch.where(in_pair, s, zero), torch.where(in_pair, ds, zero)
    e2 = r.new_zeros(n).index_add(0, dst, -0.5 * c6 * s * gp)
    grad = (-0.5 * c6 * (ds * gp + s * dgp) / r)[:, None] * vec
    # in alpha of the center: this edge's energy and its reverse's (both directions are edges)
    if dense:
        galpha = (r.new_zeros((n, n)).index_put_((dst, src), -s * gp * k6, accumulate=True) @ alpha) * cp
    else:
        galpha = r.new_zeros((n, alpha.shape[1])).index_add(0, dst, (-s * gp * k6)[:, None] * cp * alpha[src])
    # tail correction of the cells (reference formula), its derivative in alpha and the cell
    e_tail = torch.zeros_like(cn)
    gcell = r.new_zeros((data.num_graphs, 3, 3))
    if term.tail_correction:
        for g, members, periodic, _, _ in systems:
            if not periodic:
                continue
            with torch.enable_grad():
                a_leaf = alpha[members].clone().requires_grad_(True)
                c_leaf = cells[g].clone().requires_grad_(True)
                tail = term.tail_energy(z[members], a_leaf, c_leaf)
                ga, gc = torch.autograd.grad(tail.sum(), (a_leaf, c_leaf))
            e_tail[members] = tail.detach()
            galpha[members] += ga
            gcell[g] += gc
    gcn = (galpha * dacn).sum(-1)
    gq = (galpha * daq).sum(-1)
    e3 = torch.zeros_like(e2)
    if neutral:
        an = alpha0 * cp
        c6m = k6 * (an @ alpha0.t())
        dc6m = k6 * ((dalpha0 * cp) @ alpha0.t())
        cutoff = term.cutoff_triple / b
        sel = torch.nonzero(r <= cutoff).squeeze(1)
        plan = Plan(data.edge_index[:, sel], n, tile=DIRECT_TILE if molecules(data) else CELL_TILE)
        rows = sel[plan.order]
        e3, g3, dcn3 = three_body_direct(z, plan, vec[rows].contiguous(), r[rows].contiguous(), c6m, dc6m,
                                         dc6m.t().contiguous(), term.pair_radius_table(), float(term.s9),
                                         term.alp / 3.0, cutoff, term.switch_width_triple / b, n)
        grad.index_add_(0, rows, g3)
        gcn = gcn + dcn3
    # the charges: adjoint solve, then the Coulomb matrix and the EEQ coordination numbers
    gpos = pos.new_zeros((n, 3))
    graw = torch.zeros_like(cn)
    for g, members, periodic, kind, payload in systems:
        m = members.shape[0]
        if kind == "large":
            system, sel, local_edges, q_g = payload
            with torch.no_grad():
                lam, _ = system.solve_augmented(gq[members], gq.new_zeros(()), "adjoint")
        else:
            lu, piv, full, leaves = payload
            lam = _eeq_solve(lu, piv, full, torch.cat([gq[members], gq.new_zeros(1)]).unsqueeze(1))[:m, 0]
        c = cn_eeq[members]
        sc = torch.sqrt(c + term.eeq_cn_reg)
        graw[members] = lam * term.eeq_kcnchi[z[members]] * (1.0 / sc - c / (2.0 * sc ** 3)) * dcap[members]
        if kind == "large":
            # dE = -lambda^T dA q over the operator's differentiable application (positions; for a cell also its
            # EEQ-range edge vectors and the lattice)
            with torch.enable_grad():
                p_leaf = pos[members].clone().requires_grad_(True)
                v_leaf = vec[sel].clone().requires_grad_(True) if periodic else None
                c_leaf = cells[g].clone().requires_grad_(True) if periodic else None
                applied = system.apply_differentiable(p_leaf, v_leaf, system.rad, system.diag, q_g, c_leaf)
                leaves_l = (p_leaf,) + ((v_leaf, c_leaf) if periodic else ())
                grads = torch.autograd.grad(-(lam * applied).sum(), leaves_l)
            gpos.index_add_(0, members, grads[0])
            if periodic:
                grad.index_add_(0, sel, grads[1])
                gcell[g] += grads[2]
            continue
        if leaves is not None:
            # dE = -lambda^T dA q: the Ewald matrix's derivative in the positions and the cell
            p_leaf, c_leaf, a_graph = leaves
            with torch.enable_grad():
                gp_eeq, gc_eeq = torch.autograd.grad(-(lam @ a_graph @ q[members]), (p_leaf, c_leaf))
            gpos.index_add_(0, members, gp_eeq)
            gcell[g] += gc_eeq
            continue
        p = pos[members]
        rad2 = term.eeq_rad[z[members]] ** 2
        gam = torch.rsqrt(rad2[:, None] + rad2[None, :])
        dvec = p[None, :, :] - p[:, None, :]                                # pos_j - pos_i
        rij = torch.linalg.norm(dvec, dim=-1)
        off = ~torch.eye(m, dtype=torch.bool, device=p.device)
        rs = torch.where(off, rij, torch.ones_like(rij))
        da = torch.where(off, (2.0 * gam / math.sqrt(math.pi) * torch.exp(-gam * gam * rs * rs)
                               - torch.erf(gam * rs) / rs) / rs, torch.zeros_like(rs))
        qm = q[members]
        coef = -(lam[:, None] * qm[None, :]) * da / rs                      # dE / dr_ij / r_ij, both orders
        gpos.index_add_(0, members, -(coef[:, :, None] * dvec).sum(1) + (coef[:, :, None] * dvec).sum(0))
    # strain derivative (pos and cell scale with the strain): the EEQ and tail parts in the positions and cells,
    # then the edges'
    strain = (r.new_zeros((data.num_graphs, 3, 3)).index_add(0, batch, pos[:, :, None] * gpos[:, None, :])
              + (cells.transpose(1, 2) @ gcell if cells is not None else 0.0))
    # the coordination-number chain into the edges, then the forces (eV / A)
    gr = torch.where(in_cn, gcn[dst] * en, zero) + torch.where(in_eeq, graw[dst], zero)
    grad = grad + (gr * dcount / r)[:, None] * vec
    gpos = gpos.index_add(0, dst, grad).index_add(0, src, -grad)
    strain = strain.index_add(0, batch[dst], vec[:, :, None] * grad[:, None, :])
    strain = 0.5 * (strain + strain.transpose(1, 2)) * hartree
    forces = -gpos * (hartree / b)
    node = (e2 + e3 + e_tail) * hartree
    energies = node.new_zeros((data.num_graphs, 4)).index_add(
        0, batch, torch.stack([node, e2 * hartree, e3 * hartree, e_tail * hartree], dim=1))
    dtype, pdtype = data.model_dtype, data.pos.dtype
    return (node.to(dtype), energies.to(dtype), cn.to(dtype), q.to(dtype), alpha.to(dtype), forces.to(pdtype),
            strain.to(pdtype))


class _Setting(NamedTuple):
    """The constants of a D4 term for the dense path (bohr and hartree)."""
    s6: float
    s8: float
    s9: float
    a1: float
    a2: float
    alp: float
    kcn: float
    k4: float
    k5: float
    k6: float
    eeq_cn_max: float
    eeq_cn_reg: float
    wf: float
    ga: float
    gc: float
    max_ngw: int
    bohr: float
    hartree: float
    cutoff_cn: float
    cutoff_eeq_cn: float
    cutoff_pair: float
    width_pair: float
    cutoff_triple: float
    width_triple: float


#: The term's tensors and plain attributes behind the dense path's setting and tables.
_TENSORS = ("s6", "s8", "s9", "a1", "a2", "rcov", "en", "r4r2", "eeq_chi", "eeq_kcnchi", "eeq_eta", "eeq_rad", "zeff",
            "hardness", "refcn", "refq")
_PLAIN = ("cutoff_cn", "cutoff_eeq_cn", "cutoff_pair", "switch_width_pair", "cutoff_triple", "switch_width_triple")


def _setting(term) -> _Setting:
    b = term.bohr
    return _Setting(*(float(getattr(term, k)) for k in ("s6", "s8", "s9", "a1", "a2", "alp", "kcn", "k4", "k5", "k6",
                                                        "eeq_cn_max", "eeq_cn_reg", "wf", "ga", "gc")),
                    int(term.max_ngw), float(b), float(term.hartree), term.cutoff_cn / b, term.cutoff_eeq_cn / b,
                    term.cutoff_pair / b, term.switch_width_pair / b, term.cutoff_triple / b,
                    term.switch_width_triple / b)


def _dense_terms(st: _Setting, pos: Tensor, edge_index: Tensor, z: Tensor, upper: Tensor, triples: Optional[Tensor],
                 batch: Tensor, num_graphs: int, qtot: Tensor, elements: Tensor, references: Tensor, ngw: Tensor,
                 nref: Tensor, refalpha: Tensor, cp: Tensor) -> Tuple[Tensor, ...]:
    """D4 of a batch of molecules over every pair ``upper`` (``i < j``, ``(2, P)``) and every triangle of atoms
    (``triples``: see :func:`~xnn.common.models.dispersion_fast.d3_direct._triangles`), as the outputs of
    :meth:`~xnn.common.models.d4.DFTD4.evaluate_direct` (eV, A): the energies, the D4 coordination numbers, the EEQ
    charges, the dynamic polarizabilities ``(N, 23)``, the forces ``(N, 3)`` and ``dE / d(strain)`` ``(B, 3, 3)``.
    The EEQ couples every pair of a molecule; the other terms leave out the pairs that are no edge of the graph.
    ``elements`` and ``references`` are the term's tables stacked by :func:`_tables`."""
    rcov, en, r4r2, eeq_chi, eeq_kcnchi, eeq_eta, eeq_rad, zeff, hardness = elements.unbind(0)
    refcn, refq = references.unbind(0)
    dt = refalpha.dtype
    dev = pos.device
    pos = pos.to(dt) / st.bohr
    n = pos.shape[0]
    i, j = upper[0], upper[1]
    adj = torch.zeros((n, n), dtype=torch.bool, device=dev).index_put(
        (edge_index[1], edge_index[0]), torch.ones_like(edge_index[0], dtype=torch.bool))[i, j]
    same = batch[i] == batch[j]
    v = pos[i] - pos[j]
    r2 = torch.where(same, (v * v).sum(-1), torch.ones((), dtype=dt))
    r = torch.sqrt(r2)
    zi, zj = z[i], z[j]
    zero = torch.zeros((), dtype=dt)
    # coordination numbers (D4's and the EEQ's) and their slopes per pair
    rc = rcov[zi] + rcov[zj]
    u = -st.kcn * (r - rc) / rc
    count = 0.5 * (1.0 + torch.erf(u))
    dcount = -st.kcn / (rc * math.sqrt(math.pi)) * torch.exp(-u * u)
    enf = st.k4 * torch.exp(-((en[zi] - en[zj]).abs() + st.k5) ** 2 / st.k6)
    in_cn = adj & (r <= st.cutoff_cn)
    in_eeq = adj & (r <= st.cutoff_eeq_cn)
    c_cn = torch.where(in_cn, enf * count, zero)
    c_eeq = torch.where(in_eeq, count, zero)
    cn = pos.new_zeros(n).index_add(0, i, c_cn).index_add(0, j, c_cn)
    raw = pos.new_zeros(n).index_add(0, i, c_eeq).index_add(0, j, c_eeq)
    ex = torch.exp(st.eeq_cn_max - raw)
    cn_eeq = math.log(1.0 + math.exp(st.eeq_cn_max)) - torch.log(1.0 + ex)
    dcap = ex / (1.0 + ex)
    # EEQ charges: one system for the batch, the molecules' blocks and a charge constraint per molecule
    sc = torch.sqrt(cn_eeq + st.eeq_cn_reg)
    x = -eeq_chi[z] + eeq_kcnchi[z] * cn_eeq / sc
    rad2 = eeq_rad[z] ** 2
    gam = torch.rsqrt(rad2[i] + rad2[j])
    erf_r = torch.erf(gam * r)
    a_pair = torch.where(same, erf_r / r, zero)
    m = n + num_graphs
    own = torch.arange(n, device=dev)
    full = (torch.zeros((m, m), dtype=dt, device=dev)
            .index_put((torch.cat([i, j]), torch.cat([j, i])), torch.cat([a_pair, a_pair]))
            .index_put((own, own), eeq_eta[z] + math.sqrt(2.0 / math.pi) / eeq_rad[z])
            .index_put((own, n + batch), torch.ones_like(x)).index_put((n + batch, own), torch.ones_like(x)))
    lu, piv = torch.linalg.lu_factor(full)
    q = _eeq_solve(lu, piv, full, torch.cat([x, qtot.to(dt)])[:, None])[:n, 0]
    alpha, dacn, daq, alpha0, dalpha0 = _alpha_of(refcn, ngw, nref, zeff, hardness, refq, refalpha, st.wf, st.ga,
                                                  st.gc, st.max_ngw, z, cn, q, st.s9 != 0.0)
    # two-body: the energy, its slope in the distance (h) and in each atom's polarizabilities
    k6 = 3.0 / math.pi
    ai, aj = alpha[i], alpha[j]
    c6 = k6 * (ai * aj * cp).sum(-1)
    rr = 3.0 * r4r2[zi] * r4r2[zj]
    r0 = st.a1 * torch.sqrt(rr) + st.a2
    r6, r8 = r2 * r2 * r2, r2 * r2 * r2 * r2
    q6 = r0 ** 6
    t6, t8 = 1.0 / (r6 + q6), 1.0 / (r8 + q6 * r0 * r0)
    gp = st.s6 * t6 + st.s8 * rr * t8
    dgp = st.s6 * (-6.0 * r6 / r * t6 * t6) + st.s8 * rr * (-8.0 * r8 / r * t8 * t8)
    in_pair = adj & (r <= st.cutoff_pair)
    if st.width_pair > 0.0:
        sw, dsw = _switch_slope(r, st.cutoff_pair, st.width_pair)
    else:
        sw, dsw = torch.ones_like(r), torch.zeros_like(r)
    sw, dsw = torch.where(in_pair, sw, zero), torch.where(in_pair, dsw, zero)
    e_pair = -0.5 * c6 * sw * gp
    e2 = pos.new_zeros(n).index_add(0, i, e_pair).index_add(0, j, e_pair)
    h = -c6 * (dsw * gp + sw * dgp)
    ga_pair = (-sw * gp * k6)[:, None] * cp
    galpha = pos.new_zeros((n, cp.shape[0])).index_add(0, i, ga_pair * aj).index_add(0, j, ga_pair * ai)
    gcn = (galpha * dacn).sum(-1)
    gq = (galpha * daq).sum(-1)
    e3 = torch.zeros_like(e2)
    if st.s9 != 0.0:
        assert alpha0 is not None and dalpha0 is not None
        # three-body: C6 of the neutral atoms; sqrt(C6), the damping factors and the switch of every pair (zero off
        # the cutoff)
        a0i, a0j = alpha0[i] * cp, alpha0[j]
        c6n = k6 * (a0i * a0j).sum(-1)
        dc6i = k6 * (dalpha0[i] * cp * a0j).sum(-1)
        dc6j = k6 * (a0i * dalpha0[j]).sum(-1)
        in3 = adj & (r <= st.cutoff_triple)
        s6p = torch.where(in3, torch.sqrt(c6n), zero)
        qp = torch.where(in3, (r0 / r) ** (st.alp / 3.0), zero)
        if st.width_triple > 0.0:
            s3, ds3 = _switch_slope(r, st.cutoff_triple, st.width_triple)
            s3 = torch.where(in3, s3, zero)
        else:
            s3, ds3 = torch.ones_like(r), torch.zeros_like(r)
        tri, slope, e3 = _triangles(st.s9, st.alp / 3.0, st.width_triple > 0.0, n, upper, triples, r2, r, s6p, qp,
                                    s3, ds3)
        h = h + 2.0 * r * slope
        p3 = torch.where(in3, 0.5 * tri / torch.where(in3, c6n, torch.ones_like(c6n)), zero)
        gcn = gcn.index_add(0, i, p3 * dc6i).index_add(0, j, p3 * dc6j)
    # the charges: adjoint solve, the EEQ coordination numbers and the Coulomb matrix (dE = -lambda^T dA q)
    lam = _eeq_solve(lu, piv, full, torch.cat([gq, gq.new_zeros(num_graphs)])[:, None])[:n, 0]
    graw = lam * eeq_kcnchi[z] * (1.0 / sc - cn_eeq / (2.0 * sc ** 3)) * dcap
    da = (2.0 * gam / math.sqrt(math.pi) * torch.exp(-gam * gam * r2) - erf_r / r) / r
    h = h - torch.where(same, (lam[i] * q[j] + lam[j] * q[i]) * da, zero)
    # the coordination-number chain, then the gradient
    h = h + (gcn[i] + gcn[j]) * torch.where(in_cn, enf * dcount, zero) + (graw[i] + graw[j]) * torch.where(
        in_eeq, dcount, zero)
    gv = (h / r)[:, None] * v
    grad = pos.new_zeros((n, 3)).index_add(0, i, gv).index_add(0, j, -gv)
    hartree = st.hartree
    node = (e2 + e3) * hartree
    strain = node.new_zeros((num_graphs, 3, 3)).index_add(0, batch[i], (gv * hartree)[:, :, None] * v[:, None, :])
    energies = node.new_zeros((num_graphs, 3)).index_add(
        0, batch, torch.stack([node, e2 * hartree, e3 * hartree], dim=1))
    alpha_0 = alpha[:, 0]
    # in the order of _OUTPUTS
    return (node, energies[:, 0], energies[:, 1], energies[:, 2], node.new_zeros(num_graphs), cn, q, alpha_0,
            alpha, torch.stack([cn, q, alpha_0], dim=1), -grad * (hartree / st.bohr),
            0.5 * (strain + strain.transpose(1, 2)))


#: The outputs of the dense path's compiled call.
_OUTPUTS = ("node_energy", "energy", "energy_2body", "energy_3body", "energy_tail", "coordination_numbers", "charges",
            "polarizabilities", "dynamic_polarizabilities", "node_features", "forces", "virial")
#: The settings of the dense path by their key (as for D3).
_SETTINGS: Dict[str, _Setting] = {}


def _dense_entry(name: str, *args):
    return _dense_terms(_SETTINGS[name], *args)


@functools.lru_cache(maxsize=None)
def _dense_compiled(serial: bool) -> Callable:
    # as the D3 dense path, with the wrapper in Python (the C++ wrapper reaches the LU factorization and solves
    # through a slow fallback); torch < 2.7 compiles the single-thread variant of this graph into wrong forces, so
    # there the serial variant keeps the default thread count
    one_thread = {"cpp.threads": 1} if _TORCH >= (2, 7) else {}
    options = {**DENSE_OPTIONS, "cpp_wrapper": False, **(one_thread if serial else {"cpp.min_chunk_size": 1})}
    return torch.compile(_dense_entry, options=options)


def _tables(term) -> Tuple[Tensor, Tensor]:
    """The per-element tables of the dense path stacked (fewer inputs to the compiled call): ``(9, Z)`` and
    ``(2, 7, Z)``."""
    elements = torch.stack([term.rcov, term.en, term.r4r2, term.eeq_chi, term.eeq_kcnchi, term.eeq_eta, term.eeq_rad,
                            term.zeff, term.hardness])
    return elements, torch.stack([term.refcn, term.refq])


def d4_dense(term, data) -> Dict[str, Tensor]:
    """:func:`d4_direct` for a batch of molecules over every pair and triangle of atoms, in one compiled call: the
    outputs of :meth:`~xnn.common.models.d4.DFTD4.evaluate_direct`."""
    charge = getattr(data, "total_charge", None)
    qtot = charge.detach() if charge is not None else data.pos.new_zeros(data.num_graphs)
    name, (elements, references) = prepared(term, _setting, _tables, _TENSORS, _PLAIN, _SETTINGS)
    n = int(data.pos.shape[0])
    serial = n <= SERIAL_ATOMS
    # a fresh view of the atomic numbers: the tile path marks the batch's own tensor dynamic for its compiled calls
    values = _dense_compiled(serial)(
        name, data.pos, data.edge_index, data.atomic_numbers.view(-1), _upper(n), _triples(n) if serial else None,
        data.batch,
        int(data.num_graphs), qtot.reshape(-1), elements, references, term.ngw, term.nref, term.refalpha,
        term.cp_weights)
    return _cast(dict(zip(_OUTPUTS, values)), data)
