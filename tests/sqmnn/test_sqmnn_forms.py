"""NDDO semiempirical methods (sqmnn): the alternative forms of one evaluation agree.

The TorchScript branches of the corrections (run eagerly) give the energies and gradients of the host forms
(NumPy searches, reverse-mode tape); the operations of the host tape give the values and gradients of their
PyTorch form; the core of the exported potential, called eagerly, gives the eager model's energies and charges;
the per-shell overlaps of a CPU equal the batched ones; and the correction formulas reach their published values.
"""
import math
import types

import numpy as np
import pytest
import torch

from xnn.common.data import structure_to_graph
from xnn.sqmnn.models import PM3, PM6, PM6ORG, PM7, twocenter
from xnn.sqmnn.models import corrections as cr
from xnn.sqmnn.models.core import _org_scale, _pm6_scale
from xnn.sqmnn.models.integrals import _b_integrals, constants
from xnn.sqmnn.models.tape import TORCH, host_term

WATER = (np.array([[0.0, 0.0, 0.12], [0.0, 0.76, -0.47], [0.05, -0.75, -0.49]]), [8, 1, 1])
FORMAMIDE = (np.array([[0, 0.42, 0], [1.2, 0.5, 0.02], [-0.72, -0.73, 0], [-0.55, 1.36, 0.01],
                       [-1.73, -0.69, 0.05], [-0.27, -1.63, -0.03]]), [6, 8, 7, 1, 1, 1])
# an imide (one nitrogen shared by two carbonyls), slightly twisted
DIFORMAMIDE = ([7, 1, 6, 8, 1, 6, 8, 1],
               [[0, 0, 0], [0, -1.0, 0.25], [1.2, 0.7, 0], [2.25, 0.1, 0.05], [1.2, 1.79, 0], [-1.2, 0.7, 0],
                [-2.25, 0.1, -0.05], [-1.2, 1.79, 0]])
SILANOL = ([14, 8, 1, 1, 1, 1], [[0, 0, 0], [1.65, 0, 0], [2.05, 0.87, 0], [-0.5, 1.39, 0], [-0.5, -0.69, 1.2],
                                 [-0.5, -0.69, -1.2]])
# an acetylene with its C-C bond inside the switching range of the acetylenic correction
ACETYLENE = ([1, 6, 6, 1], [[0, 0, -1.695], [0, 0, -0.635], [0, 0, 0.635], [0, 0, 1.695]])
WATER_H2S = ([8, 1, 1, 16, 1, 1], [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0], [3.3, 0.35, 0], [3.6, 1.64, 0],
                                   [3.6, -0.25, 1.15]])
AMMONIA_DIMER = ([7, 1, 1, 1, 7, 1, 1, 1],
                 [[0, 0, 0], [1.01, 0, 0], [-0.34, 0.95, 0], [-0.34, -0.47, 0.83], [3.2, 0, 0], [3.54, 0.95, 0],
                  [3.54, -0.47, 0.83], [3.54, -0.47, -0.83]])
# systems where the searches stop early: one atom, no heavy atom, no hydrogen, no partner, no wide angle,
# and candidates that the reference frames or the angles reject
EDGE_CASES = [
    ([8], [[0, 0, 0]]),
    ([1, 1], [[0, 0, 0], [0, 0, 0.74]]),
    ([8, 8], [[0, 0, 0], [0, 0, 1.21]]),
    ([8, 1, 1], [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]),
    ([8, 1, 1, 8], [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0], [0, -3.0, 0]]),
    ([7, 1, 6, 1, 7], [[2.344, 1.043, 1.595], [1.063, 0.908, 1.626], [1.885, 2.381, 0.695], [1.999, 1.988, 2.432],
                       [0.655, 2.593, 0.319]]),
    ([8, 7, 1, 1, 1], [[0.046, 2.277, 2.556], [1.699, 2.043, 1.522], [2.618, 2.284, 0.008], [2.401, 0.094, 2.043],
                       [0.492, 2.417, 1.516]]),
]


@pytest.fixture(autouse=True)
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _graph(pos, z, charge=0.0, mult=None):
    s = {"pos": torch.as_tensor(np.asarray(pos, dtype=np.float64)), "atomic_numbers": np.asarray(z),
         "total_charge": float(charge)}
    if mult is not None:
        s["spin_multiplicity"] = float(mult)
    return structure_to_graph(s, 0.1)


def _hbond_cluster():
    """Eight randomly placed waters, ammonium ions and formate ions (every kind of hydrogen-bond group)."""
    rng = np.random.default_rng(11)
    mols = [([8, 1, 1], [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]),
            ([7, 1, 1, 1, 1], [[0, 0, 0], [0.6, 0.6, 0.6], [-0.6, -0.6, 0.6], [-0.6, 0.6, -0.6], [0.6, -0.6, -0.6]]),
            ([6, 8, 8, 1], [[0, 0, 0], [1.25, 0, 0], [-0.62, 1.08, 0], [-0.55, -0.95, 0]])]
    z, pos = [], []
    for m in range(8):
        mz, mp = mols[m % 3]
        R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        z += mz
        pos += list(np.asarray(mp) @ R.T + rng.uniform(0, 6.0, 3))
    return z, np.asarray(pos)


def _scripted_branches(monkeypatch, module):
    """Make ``torch.jit.is_scripting()`` true inside ``module``: its TorchScript branches run eagerly."""
    jit = types.ModuleType("torch.jit")
    jit.__dict__.update(vars(torch.jit))
    jit.is_scripting = lambda: True
    fake = types.ModuleType("torch")
    fake.__dict__.update(vars(torch))
    fake.jit = jit
    monkeypatch.setattr(module, "torch", fake)


def _corrections(kind, z, pos, tables):
    p = torch.tensor(pos, requires_grad=True)
    e = cr.structure_corrections(kind, z, p, tables, 7.1853)
    g = torch.autograd.grad(e.sum(), p, allow_unused=True)[0] if e.requires_grad else None
    return e.detach(), torch.zeros_like(p) if g is None else g


def test_corrections_scripted_branches_match_host_forms(monkeypatch):
    """The TorchScript branches of every method's corrections (bond topology, PM7 dispersion, the PyTorch searches
    of the PM7 and H4 hydrogen bonds, H-H repulsion), run eagerly, give the energies and gradients of the host
    forms of an eager model, also where the searches find nothing or reject every candidate."""
    tables = cr.correction_tables(torch.device("cpu"))
    systems = [_hbond_cluster(), DIFORMAMIDE, SILANOL, ACETYLENE, WATER_H2S, AMMONIA_DIMER] + EDGE_CASES
    systems = [(list(z), np.asarray(pos, dtype=np.float64)) for z, pos in systems]
    host = [[_corrections(kind, z, pos, tables) for kind in range(4)] for z, pos in systems]
    # the last two systems have candidates that their reference frames (then their angles) all reject
    (z5, x5), (z6, x6) = systems[-2:]
    assert len(cr._np_hbond_triples(z5, x5, cr._distances(x5), 1.4, 7.0)) > 0
    assert cr._np_pm7_hbond_tables(z5, x5) is None
    assert cr._np_pm7_hbond_tables(z6, x6) is not None
    assert cr.pm7_hbond_energy(TORCH, z6, torch.tensor(x6), tables) is None
    _scripted_branches(monkeypatch, cr)
    for (z, pos), per_kind in zip(systems, host):
        for kind in range(4):
            e, g = _corrections(kind, z, pos, tables)
            assert torch.allclose(e, per_kind[kind][0], rtol=1e-12, atol=1e-12), (z, kind)
            assert torch.allclose(g, per_kind[kind][1], rtol=1e-10, atol=1e-10), (z, kind)
    # every method's terms are present in some system
    for kind in range(4):
        assert any(float(per_kind[kind][0].abs().sum()) > 0.0 for per_kind in host), kind


def test_correction_terms_reach_their_formulas():
    """The acetylenic term at 1.27 Angstrom, the planar-nitrogen term, the Si-O-H bend and an imide's single
    amide torsion equal their closed forms."""
    z, pos = ACETYLENE
    p = torch.tensor(pos, dtype=torch.float64)
    bonds = cr.bond_topology(z, p, cr.correction_tables(torch.device("cpu")))
    # x = 1/2 of the switching range: 12 (1 - 10 x^3 + 15 x^4 - 6 x^5 + (-5 + 25 x)(x^3 - 3 x^4 + 3 x^5 - x^6))
    assert float(cr.yne_correction(z, p, bonds).sum()) == pytest.approx(7.40625, abs=1e-9)

    def angle(x, i, j, k):
        a, b = x[i] - x[j], x[k] - x[j]
        return math.acos(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))

    z, pos = DIFORMAMIDE
    x = np.asarray(pos, dtype=np.float64)
    p = torch.tensor(x)
    bonds = cr.bond_topology(z, p, cr.correction_tables(torch.device("cpu")))
    assert sorted(bonds[0]) == [1, 2, 5]
    total = angle(x, 1, 0, 2) + angle(x, 2, 0, 5) + angle(x, 5, 0, 1)
    nsp2 = cr.nsp2_correction(z, p, bonds)
    assert float(nsp2[0]) == pytest.approx(-0.5 * math.exp(-10.0 * (2.0 * math.pi - total)), rel=1e-12)
    assert float(nsp2[1:].abs().sum()) == 0.0
    # the nitrogen shared by both carbonyls takes the torsions of the first one only
    torsion = cr.peptide_torsion(z, p, 1.0)
    first = cr._sin2_dihedral(p, 3, 2, 0, 5) + cr._sin2_dihedral(p, 3, 2, 0, 1)
    assert float(torsion[0]) == pytest.approx(float(first), rel=1e-12) and float(torsion[1:].abs().sum()) == 0.0
    z, pos = SILANOL
    x = np.asarray(pos, dtype=np.float64)
    p = torch.tensor(x)
    bonds = cr.bond_topology(z, p, cr.correction_tables(torch.device("cpu")))
    bend = cr.si_o_h_correction(z, p, bonds)
    assert float(bend[1]) == pytest.approx(15.0 * (angle(x, 0, 1, 2) - math.radians(125.0)) ** 2, rel=1e-12)
    assert float(bend.sum()) == pytest.approx(float(bend[1]), abs=0.0)


def test_batch_corrections_skip_empty_structures():
    """Per-structure corrections of a batch with an empty structure equal those of each molecule alone."""
    tables = cr.correction_tables(torch.device("cpu"))
    (za, xa), (zb, xb) = (DIFORMAMIDE[0], DIFORMAMIDE[1]), (WATER_H2S[0], WATER_H2S[1])
    pos = torch.tensor(np.vstack([xa, np.asarray(xb) + 20.0]), dtype=torch.float64)
    batch = torch.tensor([0] * len(za) + [2] * len(zb))
    out = cr.batch_corrections(2, za + zb, pos, batch, 3, tables, 3.1595)
    alone = torch.cat([cr.structure_corrections(2, za, pos[:len(za)], tables, 3.1595),
                       cr.structure_corrections(2, zb, pos[len(za):], tables, 3.1595)])
    assert torch.allclose(out, alone, rtol=0.0, atol=1e-12) and float(out.abs().sum()) > 0.0


def _ops_term(ia, ib, n):
    """A term with every operation of the namespaces (broadcasting, powers, sums, roots, cross products, stacks,
    clamps, inverse trigonometric functions and selections)."""
    def term(ops, p):
        a, b = ops.take(p, ops.array(ia, p)), ops.take(p, ops.array(ib, p))
        d = a - b
        r = ops.sqrt(ops.sum(d * d))
        c = ops.cross(a, b) + p[0]                                     # (M, 3) + (3,)
        s = ops.stack([r ** 1, r ** 2, r ** 3])
        t = ops.sum(s * c + r[:, None] * c)                            # (M, 1) * (M, 3)
        u = ops.clamp(t / (1.0 + ops.abs(t)), -0.6, 0.6)
        e = ops.acos(u) + ops.atan2(t, r)
        e = ops.where(ops.value(r) > 2.5, 0.5, e) + ops.where(ops.value(r) > 2.0, e, 1.5)
        return ops.index_add(n, ops.array(ia, p), e)
    return term


def test_tape_operations_match_their_torch_form():
    """Values and gradients of a term on the host tape equal those of its PyTorch form, with and without
    gradients requested; a selection between two numbers is refused."""
    rng = np.random.default_rng(7)
    n = 6
    x = rng.normal(size=(n, 3)) * 1.3
    ia, ib = torch.triu_indices(n, n, 1)
    term = _ops_term(ia, ib, n)
    p = torch.tensor(x, requires_grad=True)
    ref = term(TORCH, p)
    g_ref, = torch.autograd.grad((ref * torch.arange(n)).sum(), p)
    p2 = torch.tensor(x, requires_grad=True)
    tape = host_term(term, p2)
    g_tape, = torch.autograd.grad((tape * torch.arange(n)).sum(), p2)
    assert torch.allclose(tape, ref.detach(), rtol=1e-13, atol=1e-13)
    assert torch.allclose(g_tape, g_ref, rtol=1e-11, atol=1e-11)
    with torch.no_grad():
        assert torch.allclose(host_term(term, torch.tensor(x)), ref.detach(), rtol=1e-13, atol=1e-13)
    with pytest.raises(ValueError, match="tensor operand"):
        TORCH.where(torch.tensor([True]), 1.0, 2.0)


def test_hydrogen_bonds_at_a_straight_angle():
    """At an exactly straight O-H...O angle the H4 and PM7 hydrogen bonds, in both forms, have the central
    differences of their energy as gradient and the bent angles' limit as Hessian."""
    z = [8, 1, 8, 1, 1, 1]

    def dimer(bend):
        return torch.tensor([[0.0, 0, 0], [0.97, bend, 0], [2.9, 0, 0], [3.2, 0.7, 0.55], [-0.3, 0.9, 0.1],
                             [3.2, -0.75, 0.5]], dtype=torch.float64)
    tables = cr.correction_tables(torch.device("cpu"))
    v = torch.tensor(np.random.default_rng(3).normal(size=(6, 3)))
    terms = [lambda p: cr.h4_hbond(z, p, tables), lambda p: cr.h4_energy(TORCH, z, p, tables),
             lambda p: cr.pm7_hbond(z, p, tables), lambda p: cr.pm7_hbond_energy(TORCH, z, p, tables)]
    for f in terms:
        derivs = []
        for bend in (0.0, 1e-7, -1e-7):
            p = dimer(bend).requires_grad_(True)
            g, = torch.autograd.grad(f(p).sum(), p, create_graph=True)
            hv, = torch.autograd.grad((g * v).sum(), p)
            derivs.append((g.detach(), hv))
        (g, hv), (_, hv_up), (_, hv_down) = derivs
        hv_bent = 0.5 * (hv_up + hv_down)              # the bend's first-order change cancels
        assert torch.isfinite(g).all() and torch.isfinite(hv).all()
        x, h = dimer(0.0), 1e-5
        fd = torch.zeros_like(x)
        for i in range(6):
            for k in range(3):
                xp, xm = x.clone(), x.clone()
                xp[i, k] += h
                xm[i, k] -= h
                fd[i, k] = (f(xp).sum() - f(xm).sum()) / (2.0 * h)
        assert torch.allclose(g, fd, rtol=0, atol=1e-7)
        assert torch.allclose(hv, hv_bent, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("cls,z,pos,mult", [
    (PM3, FORMAMIDE[1], FORMAMIDE[0], 1.0),
    (PM6ORG, FORMAMIDE[1] + WATER[1], np.vstack([FORMAMIDE[0], WATER[0] + [3.0, 0.5, 0.2]]), 1.0),
    (PM7, [6, 1, 1, 1], [[0, 0, 0.02], [1.08, 0, 0], [-0.54, 0.94, 0.01], [-0.55, -0.93, 0]], 2.0)])
def test_export_core_matches_eager_model(cls, z, pos, mult):
    """The core of the exported potential, called eagerly, gives the eager model's energies per atom and charges,
    also from the warm start of a second geometry (closed and open shells, PM6-ORG's D3 term); it refuses
    periodic cells and elements without parameters."""
    core = cls().export_core()
    eager = cls(reuse_density=False)
    rng = np.random.default_rng(1)
    zt = torch.tensor(z)
    empty_edges, empty_vec = torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 3))
    cell, pbc = torch.zeros((3, 3)), torch.zeros(3, dtype=torch.bool)
    for step in range(2):
        x = np.asarray(pos, dtype=np.float64) + 0.01 * step * rng.normal(size=np.shape(pos))
        p = torch.tensor(x)
        feats, node_energy, charges = core.node_features_energy_charges(zt, empty_edges, empty_vec, p, cell, pbc,
                                                                        0.0, mult)
        ref = eager(_graph(x, z, mult=mult))
        assert feats.shape == (len(z), 0)
        # the warm start converges to the SCF tolerance: the total to round-off, its partition to 1e-7 eV
        tol = 1e-10 if step == 0 else 1e-7
        assert torch.allclose(node_energy, ref["node_energy"].detach(), rtol=0.0, atol=tol)
        assert abs(float(node_energy.sum() - ref["energy"].sum())) < 1e-9
        assert torch.allclose(charges, ref["charges"].detach(), rtol=0.0, atol=1e-8)
    with pytest.raises(ValueError, match="periodic"):
        core.node_features_energy_charges(zt, empty_edges, empty_vec, p, torch.eye(3) * 20.0,
                                          torch.ones(3, dtype=torch.bool), 0.0, mult)
    if cls is PM6ORG:
        with pytest.raises(ValueError, match="no parameters"):
            core.node_features_energy_charges(torch.tensor([22, 17]), empty_edges, empty_vec,
                                              torch.tensor([[0.0, 0, 0], [0, 0, 2.2]]), cell, pbc, 0.0, 2.0)


def test_overlaps_per_shell_match_batched_overlaps():
    """The overlaps a CPU evaluates per shell combination and polynomial (element pairs of several principal
    quantum numbers in one group) equal the batched kernel's, with their gradients; given local overlaps replace
    the computed ones in the pair kernel."""
    model = PM6()
    q = model._element_quantities([1, 8, 16, 22])
    qt = {key: v for key, v in q.items() if isinstance(v, torch.Tensor)}
    c = constants(torch.device("cpu"), torch.float64)
    rng = np.random.default_rng(2)
    for ka, kb, ea, eb in ((4, 4, [1, 2, 1, 2, 2], [2, 1, 1, 2, 1]), (9, 4, [3, 3, 3], [1, 2, 1]),
                           (4, 1, [1, 2, 2, 1], [0, 0, 0, 0]), (9, 9, [3, 3], [3, 3])):
        tab = twocenter.pair_tables(q, ka, kb, True, torch.device("cpu"), torch.float64)
        ia, ib = torch.tensor(ea), torch.tensor(eb)
        r = torch.tensor(rng.uniform(1.5, 6.0, size=len(ea)), requires_grad=True)
        batched = twocenter._overlap_kernel(qt, tab, c, ia, ib, r, ka, kb)
        per_shell = twocenter._overlaps_host(qt, tab, c, ia, ib, r, ka, kb)
        assert torch.allclose(per_shell, batched, rtol=1e-12, atol=1e-14), (ka, kb)
        assert float(batched.abs().sum()) > 0.0
        w = torch.tensor(rng.normal(size=batched.shape))
        g_b, = torch.autograd.grad((batched * w).sum(), r)
        g_s, = torch.autograd.grad((per_shell * w).sum(), r)
        assert torch.allclose(g_s, g_b, rtol=1e-11, atol=1e-13), (ka, kb)
        vec = torch.tensor(rng.normal(size=(len(ea), 3)) * 1.2)
        own = twocenter.pair_kernel(qt, tab, c, ia, ib, vec, ka, kb, ka == 9, False, None)
        r_bohr = vec.norm(dim=-1) / 0.529177210903
        given = twocenter.pair_kernel(qt, tab, c, ia, ib, vec, ka, kb, ka == 9, False,
                                      twocenter._overlaps_host(qt, tab, c, ia, ib, r_bohr, ka, kb))
        for key in own:
            assert torch.allclose(given[key], own[key], rtol=1e-12, atol=1e-13), key


def test_b_integrals_by_recursion_against_quadrature():
    """``B_k(t)`` of arguments all beyond the series range (the upward recursion alone) equals Gauss-Legendre
    quadrature."""
    t = torch.tensor([6.5, -7.25, 10.0, -12.5])
    got = _b_integrals(t, 6, None, constants(torch.device("cpu"), torch.float64))
    eta, w = np.polynomial.legendre.leggauss(80)
    want = np.array([[np.sum(w * eta ** k * np.exp(-tv * eta)) for k in range(6)] for tv in t.tolist()])
    assert np.allclose(got.numpy(), want, rtol=1e-12, atol=0.0)


def test_pm6_org_steric_scale_of_cc_and_osi_pairs():
    """PM6-ORG's scale of a C-C pair is PM6's plus the Gaussian steric term of its parameters 13-15; that of an
    O-Si pair is PM6's."""
    params = PM6ORG().core_parameters()
    par = params["PAR"]
    za, zb = torch.tensor([6, 6, 8, 14]), torch.tensor([6, 6, 14, 8])
    r = torch.tensor([1.3, float(par[15]) + 0.4, 1.7, 2.6])
    x, a = torch.tensor([0.8, 0.7, 0.9, 0.6]), torch.tensor([2.0, 1.6, 1.5, 1.1])
    steric = 0.01 * par[13] * torch.where(r > par[15], torch.exp(-par[14] * (r - par[15]) ** 2), torch.ones_like(r))
    want = _pm6_scale(params, za, zb, r, x, a) + torch.where(za == 6, steric, torch.zeros_like(r))
    assert torch.allclose(_org_scale(params, za, zb, r, x, a), want, rtol=1e-14, atol=1e-15)
