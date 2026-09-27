"""D4 scale regimes: the large-system EEQ operator (matrix-free, LU / CG,
implicit differentiation) against the dense reference path, and the
checkpointed three-body chunks against the plain loop."""
import numpy as np
import pytest
import torch

from xnn.common.config import from_dict
from xnn.common.data import structure_to_graph
from xnn.common.models import ForceStressOutput, build_model, eeq
from xnn.common.models.d3 import D3Dispersion
from xnn.common.models.d4 import AUTO_LARGE_MOLECULAR, AUTO_LARGE_PERIODIC, D4Dispersion, DFTD4
from xnn.common.models.dispersion import three_body_energy, three_body_energy_chunked

pytestmark = pytest.mark.usefixtures("_f64")


@pytest.fixture
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


MOL = dict(cutoff_pair=9.0, cutoff_triple=6.0, cutoff_cn=8.0, cutoff_eeq_cn=8.0)
# periodic: the EEQ range must reach 20 bohr, so the pair cutoff (its default) does
PER = dict(cutoff_pair=11.0, cutoff_triple=6.0, cutoff_cn=8.0, cutoff_eeq_cn=8.0)


def _graph(pos, z, cutoff, cell=None, charge=None):
    s = {"pos": torch.tensor(pos), "atomic_numbers": torch.tensor(z)}
    if cell is not None:
        s["cell"] = torch.tensor(cell)
        s["pbc"] = torch.tensor([True, True, True])
    if charge is not None:
        s["total_charge"] = charge
    return structure_to_graph(s, cutoff)


def _molecule(n=120, seed=0, box=9.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, box, (n, 3)), ([8, 1, 1, 6, 7] * (n // 5 + 1))[:n]


def _crystal(n=200, seed=1, L=12.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, L, (n, 3)), ([8, 1, 1, 6] * (n // 4 + 1))[:n], np.eye(3) * L


def _run(model, pos, z, cell=None, charge=None, stress=False):
    return ForceStressOutput(model, compute_stress=stress)(_graph(pos, z, model.cutoff, cell, charge))


# regime plumbing
def test_regime_options_and_auto_selection():
    d4 = DFTD4()
    assert d4.regime == "auto" and d4.checkpoint_triplets and d4.eeq_solver == "auto"
    assert d4.select_regime(AUTO_LARGE_PERIODIC, True) == "dense"
    assert d4.select_regime(AUTO_LARGE_PERIODIC + 1, True) == "large"
    assert d4.select_regime(AUTO_LARGE_MOLECULAR, False) == "dense"
    assert d4.select_regime(AUTO_LARGE_MOLECULAR + 1, False) == "large"
    assert DFTD4(regime="LARGE").select_regime(2, False) == "large"
    assert DFTD4(regime="dense").select_regime(10 ** 6, True) == "dense"
    with pytest.raises(ValueError, match="regime"):
        DFTD4(regime="fast")
    with pytest.raises(ValueError, match="eeq_solver"):
        DFTD4(eeq_solver="gmres")
    # cutoff_eeq defaults to 16 A (or the other cutoffs, if larger) unless the
    # regime is dense, which has no Ewald split; an explicit value always wins
    from xnn.common.models.d4 import DEFAULT_CUTOFF_EEQ
    assert DEFAULT_CUTOFF_EEQ == 16.0
    assert DFTD4(**MOL).cutoff == 16.0 and DFTD4(**MOL).cutoff_eeq == 16.0
    assert DFTD4(regime="large", **MOL).cutoff == 16.0
    assert DFTD4(regime="dense", **MOL).cutoff == 9.0 and DFTD4(regime="dense", **MOL).cutoff_eeq == 9.0
    assert DFTD4(cutoff_eeq=14.0, **MOL).cutoff == 14.0
    assert DFTD4(cutoff_eeq=9.0, **MOL).cutoff == 9.0
    wide = dict(MOL, cutoff_pair=18.0)
    assert DFTD4(**wide).cutoff_eeq == 18.0                       # never below the other cutoffs
    assert DFTD4().cutoff_eeq == DFTD4().cutoff                    # upstream 60 bohr pair cutoff
    # config hook
    cfg = from_dict({"model": {"name": "d4", "extra": {"regime": "large", "eeq_solver": "cg",
                                                        "checkpoint_triplets": False, "cutoff_eeq": 12.0}}})
    m = build_model(cfg.model)
    assert (m.term.regime, m.term.eeq_solver, m.term.checkpoint_triplets, m.term.cutoff_eeq) == ("large", "cg", False, 12.0)


def test_large_regime_needs_twenty_bohr_of_neighbor_list_for_crystals():
    pos, z, cell = _crystal(n=20)
    m = D4Dispersion(regime="large", cutoff_eeq=9.0, **MOL)          # 9 A = 17 bohr
    with pytest.raises(ValueError, match="20 bohr"):
        _run(m, pos, z, cell)
    _run(D4Dispersion(regime="large", cutoff_eeq=9.0, **MOL), *_molecule(20))   # molecules: no such limit


# EEQ: large vs dense
@pytest.mark.parametrize("solver", ["lu", "cg"])
def test_molecular_large_matches_dense_to_rounding(solver):
    pos, z = _molecule()
    dense = _run(D4Dispersion(regime="dense", **MOL), pos, z, charge=1.0)
    large = _run(D4Dispersion(regime="large", eeq_solver=solver, **MOL), pos, z, charge=1.0)
    # the molecular operator is the same matrix, so only the solver differs
    assert torch.allclose(dense["eeq_charges"], large["eeq_charges"], atol=1e-12, rtol=0)
    assert abs(float(dense["energy"] - large["energy"])) < 1e-11
    assert torch.allclose(dense["forces"], large["forces"], atol=1e-9, rtol=0)
    assert abs(float(large["eeq_charges"].sum()) - 1.0) < 1e-12


@pytest.mark.parametrize("solver", ["lu", "cg"])
def test_periodic_large_matches_dense_to_its_ewald_tolerance(solver):
    """dftd4's Ewald sum (which the dense path reproduces) is converged to
    sqrt(eps) ~ 1e-8 by its alpha bisection; the large path is converged to
    1e-10, so the two agree at the dense side's tolerance."""
    pos, z, cell = _crystal()
    dense = _run(D4Dispersion(regime="dense", **PER), pos, z, cell, stress=True)
    large = _run(D4Dispersion(regime="large", eeq_solver=solver, **PER), pos, z, cell, stress=True)
    assert torch.allclose(dense["eeq_charges"], large["eeq_charges"], atol=2e-7, rtol=0)
    assert abs(float(dense["energy"] - large["energy"])) < 1e-7
    assert torch.allclose(dense["forces"], large["forces"], atol=5e-5, rtol=0)
    assert torch.allclose(dense["stress"], large["stress"], atol=1e-8, rtol=0)


def test_periodic_large_is_independent_of_the_ewald_split():
    """Different real-space ranges (hence alpha, G set) must give the same
    charges up to the truncation tolerance: the Ewald identity holds."""
    pos, z, cell = _crystal()
    q = {}
    for ce in (11.0, 14.0, 18.0):
        m = D4Dispersion(regime="large", cutoff_eeq=ce, **PER)
        q[ce] = m(_graph(pos, z, m.cutoff, cell))["eeq_charges"].detach()
    assert torch.allclose(q[11.0], q[14.0], atol=1e-8, rtol=0)
    assert torch.allclose(q[11.0], q[18.0], atol=1e-8, rtol=0)


def test_large_regime_forces_match_finite_differences():
    rng = np.random.default_rng(3)
    n = 40
    pos, z, cell = rng.uniform(0, 9.0, (n, 3)), ([8, 1, 1, 6] * 10), np.eye(3) * 9.0
    m = D4Dispersion(regime="large", switch_width_pair=2.0, switch_width_triple=1.0, **PER)
    f = _run(m, pos, z, cell)["forces"].detach().numpy()

    def energy(p):
        return float(m(_graph(p, z, m.cutoff, cell))["energy"])

    h = 1e-4
    for i in (0, 7, 23):
        for k in range(3):
            pp, pm = pos.copy(), pos.copy()
            pp[i, k] += h
            pm[i, k] -= h
            assert abs(-(energy(pp) - energy(pm)) / (2 * h) - f[i, k]) < 1e-7


def test_large_regime_supports_force_training_double_backward():
    """The implicit-function gradient is exact to first order, which is all a
    force loss needs: d(loss)/d(theta) must match the dense path."""
    rng = np.random.default_rng(3)
    pos, z, cell = rng.uniform(0, 9.0, (40, 3)), ([8, 1, 1, 6] * 10), np.eye(3) * 9.0
    grads = {}
    for regime in ("dense", "large"):
        m = D4Dispersion(regime=regime, switch_width_pair=2.0, trainable=True, **PER)
        out = _run(m, pos, z, cell)
        (out["forces"] ** 2).sum().backward()
        grads[regime] = torch.stack([m.term.s8.grad, m.term.a1.grad, m.term.a2.grad])
    # the dense side carries dftd4's ~1e-8 Ewald truncation (see the periodic test)
    assert torch.allclose(grads["dense"], grads["large"], rtol=1e-4, atol=1e-9)


def test_auto_regime_switches_and_batches_mix_regimes():
    """A batch of one small and one large-regime structure: each gets its own
    path and matches its standalone evaluation."""
    pos_a, z_a = _molecule(30, seed=5)
    pos_b, z_b = _molecule(60, seed=6)
    m = D4Dispersion(**MOL)
    m.term.auto_large_molecular = 40              # force the switch between the two
    assert m.term.select_regime(30, False) == "dense" and m.term.select_regime(60, False) == "large"
    from xnn.common.data import collate
    batch = collate([_graph(pos_a, z_a, m.cutoff), _graph(pos_b, z_b, m.cutoff)])
    out = m(batch)
    ea = float(m(_graph(pos_a, z_a, m.cutoff))["energy"])
    eb = float(m(_graph(pos_b, z_b, m.cutoff))["energy"])
    assert torch.allclose(out["energy"], torch.tensor([ea, eb]), atol=1e-11, rtol=0)


# EEQ operator internals
def test_operator_matvec_matches_assembled_matrix_and_cg_matches_lu():
    pos, z, cell = _crystal(n=60, L=11.0)
    d4 = DFTD4(**PER)
    g = _graph(pos, z, d4.cutoff, cell)
    zt, p_au = g.atomic_numbers, g.pos / d4.bohr
    vec = g.edge_vectors() / d4.bohr
    r = torch.linalg.norm(vec, dim=-1)
    sel = r <= d4.cutoff_eeq / d4.bohr
    rad = d4.eeq_rad[zt]
    alpha = eeq.ewald_alpha(d4.cutoff_eeq / d4.bohr)
    grid, gvec, gfac = eeq.reciprocal_vectors(g.cell[0] / d4.bohr, alpha)
    diag = d4.eeq_eta[zt] + np.sqrt(2 / np.pi) / rad - 2 * alpha / np.sqrt(np.pi)
    system = eeq.EEQSystem(diag, rad, p_au, g.edge_index[:, sel], vec[sel], alpha, gvec, gfac, grid)
    amat = system.assemble()
    assert torch.allclose(amat, amat.t(), atol=1e-12, rtol=0)          # symmetric
    v = torch.randn(zt.shape[0])
    assert torch.allclose(system.matvec(v), amat @ v, atol=1e-10, rtol=0)
    # chunked structure factors give the same reciprocal sum
    system._sf = None
    assert torch.allclose(system.matvec(v), amat @ v, atol=1e-10, rtol=0)
    y_lu, mu_lu = system._solve_lu(v, torch.tensor(0.0))
    y_cg, mu_cg = system._solve_cg(v, torch.tensor(0.0))
    assert torch.allclose(y_lu, y_cg, atol=1e-9, rtol=0) and abs(float(mu_lu - mu_cg)) < 1e-9
    assert abs(float(y_cg.sum())) < 1e-9                                 # the constraint
    with pytest.raises(ValueError, match="eeq_solver"):
        eeq.EEQSystem(diag, rad, p_au, solver="qr")


def test_ewald_parameters():
    assert abs(eeq.ewald_alpha(20.0) * 20.0 - 4.572824967) < 1e-6      # erfc(x) = 1e-10
    cell = torch.eye(3) * 20.0
    grid, gvec, gfac = eeq.reciprocal_vectors(cell, eeq.ewald_alpha(20.0))
    assert grid.shape == gvec.shape and gfac.shape[0] == grid.shape[0]
    assert not (grid == 0).all(dim=1).any()                             # G = 0 excluded
    g2 = (gvec * gvec).sum(-1)
    alpha = eeq.ewald_alpha(20.0)
    assert float(torch.exp(-0.25 * g2 / alpha ** 2).min()) >= 1e-10 * (1 - 1e-9)


# ATM: checkpointed blocks vs the plain loop
def test_checkpointed_triplets_match_plain_loop_and_script():
    pos, z = _molecule(80, seed=2, box=7.0)
    plain = D4Dispersion(checkpoint_triplets=False, **MOL)
    chunked = D4Dispersion(checkpoint_triplets=True, **MOL)
    a, b = _run(plain, pos, z), _run(chunked, pos, z)
    assert abs(float(a["energy_3body"] - b["energy_3body"])) < 1e-13
    assert torch.allclose(a["forces"], b["forces"], atol=1e-12, rtol=0)
    # tiny blocks exercise the block boundaries; D3 shares the machinery
    g = _graph(pos, z, plain.cutoff)
    d4 = plain.term
    vec = g.edge_vectors() / d4.bohr
    r = torch.linalg.norm(vec, dim=-1)
    sel = r <= d4.cutoff_triple / d4.bohr
    ei, vs, rs = g.edge_index[:, sel], vec[sel], r[sel]
    zt = g.atomic_numbers
    cn, _ = d4.coordination_numbers(zt, g.edge_index, r, zt.shape[0])
    alpha = d4.dynamic_polarizabilities(zt, d4.reference_weights(zt, cn, torch.zeros_like(cn)))
    c6 = (3.0 / np.pi) * (alpha * d4.cp_weights) @ alpha.t()
    args = (d4.pair_radius_table(), d4.s9, d4.alp / 3.0, d4.cutoff_triple / d4.bohr, 0.0, zt.shape[0])
    ref = three_body_energy(zt, ei, vs, rs, c6, *args)
    for chunk in (1, 37, 1 << 20):
        e_mat = three_body_energy_chunked(zt, ei, vs, rs, *args, c6_mat=c6, chunk=chunk)
        e_alpha = three_body_energy_chunked(zt, ei, vs, rs, *args, alpha_a=(3.0 / np.pi) * alpha * d4.cp_weights,
                                            alpha_b=alpha, chunk=chunk)
        assert torch.allclose(ref, e_mat, atol=1e-15, rtol=0) and torch.allclose(ref, e_alpha, atol=1e-15, rtol=0)
    d3a = _run(D3Dispersion(s9=1.0, checkpoint_triplets=False, cutoff_pair=9.0, cutoff_triple=6.0, cutoff_cn=8.0), pos, z)
    d3b = _run(D3Dispersion(s9=1.0, checkpoint_triplets=True, cutoff_pair=9.0, cutoff_triple=6.0, cutoff_cn=8.0), pos, z)
    assert abs(float(d3a["energy_3body"] - d3b["energy_3body"])) < 1e-13
    assert torch.allclose(d3a["forces"], d3b["forces"], atol=1e-12, rtol=0)


def _saved_elements(model, graph):
    """Elements of the tensors autograd keeps for the backward pass.

    Non-reentrant checkpoints install their own saved-tensor hooks inside the
    block, so an outer pack hook only sees what is truly retained (reading
    ``grad_fn._saved_*`` would instead trigger the recompute).
    """
    packed = []

    def pack(t):
        packed.append(t.numel())
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        model(graph)["energy"].sum()
    return sum(packed)


def test_checkpointed_triplets_retain_no_block_state():
    """Only the neighbor list and per-atom sums survive the forward pass: the
    retained tensors must not scale with the triplet count."""
    pos, z = _molecule(80, seed=2, box=7.0)
    plain = D4Dispersion(checkpoint_triplets=False, **MOL)
    chunked = D4Dispersion(checkpoint_triplets=True, **MOL)
    g = _graph(pos, z, plain.cutoff)
    g.pos.requires_grad_(True)
    n_edges = int(g.edge_index.shape[1])
    counts = torch.bincount(g.edge_index[1], minlength=len(z))
    n_triplets = int((counts * (counts - 1) // 2).sum())
    assert n_triplets > 10 * n_edges                  # a regime where triplets dominate
    kept_plain, kept_chunked = _saved_elements(plain, g), _saved_elements(chunked, g)
    assert kept_plain > 10 * n_triplets               # the plain loop keeps the triplet state
    assert kept_chunked < 80 * n_edges + 200 * len(z) ** 2   # edge- and (dense EEQ) atom-sized only
    assert kept_chunked < kept_plain / 10


def test_exported_head_pins_the_dense_paths():
    from xnn.common.deploy.torchscript import _DispersionHead
    m = D4Dispersion(regime="large", **MOL)
    head = _DispersionHead(m.term)
    assert head.term.regime == "dense" and head.term.checkpoint_triplets is False
    assert m.term.regime == "large" and m.term.checkpoint_triplets is True      # untouched
    torch.jit.script(head)


# recompute blocks: first and second derivatives
def test_recompute_matches_plain_autograd_to_second_order():
    from xnn.common.models.recompute import recompute
    torch.manual_seed(0)
    x = torch.randn(30, 3, requires_grad=True)
    theta = torch.tensor(0.7, requires_grad=True)

    def block(p, th):
        d = p[:, None] - p[None]
        r = (d * d).sum(-1).add(torch.eye(30)).sqrt()
        return (th * torch.erf(0.9 * r) / r ** 3).sum()

    e_ref = block(x, theta)
    f_ref = torch.autograd.grad(e_ref, x, create_graph=True)[0]
    g_ref = torch.autograd.grad((f_ref ** 2).sum(), [x, theta])
    e = recompute(block, x, theta)
    assert torch.allclose(e, e_ref, atol=1e-14, rtol=0)
    f = torch.autograd.grad(e, x, create_graph=True)[0]
    assert torch.allclose(f, f_ref, atol=1e-13, rtol=0)
    g = torch.autograd.grad((f ** 2).sum(), [x, theta])
    assert torch.allclose(g[0], g_ref[0], atol=1e-11, rtol=0)
    assert torch.allclose(g[1], g_ref[1], atol=1e-11, rtol=0)
    # eval-style single backward and no-grad evaluation
    x2 = x.detach().requires_grad_(True)
    (fx,) = torch.autograd.grad(recompute(block, x2, theta.detach()), x2)
    assert torch.allclose(fx, f_ref.detach(), atol=1e-13, rtol=0)
    with torch.no_grad():
        assert float(recompute(block, x, theta)) == pytest.approx(float(e_ref), abs=1e-14)


def test_chunked_triplets_exact_in_training_mode():
    """create_graph=True path (force loss): chunked blocks equal the plain loop
    in energy, forces and the force-loss gradients of the damping parameters."""
    pos, z = _molecule(60, seed=4, box=7.0)
    grads = {}
    for chunked in (False, True):
        m = D4Dispersion(checkpoint_triplets=chunked, trainable=True, **MOL)
        fs = ForceStressOutput(m)
        fs.train()
        out = fs(_graph(pos, z, m.cutoff))
        (out["forces"] ** 2).sum().backward()
        grads[chunked] = (out["energy"].detach(), out["forces"].detach(),
                          torch.stack([m.term.s8.grad, m.term.a1.grad, m.term.a2.grad, m.term.s9.grad]))
    assert torch.allclose(grads[False][0], grads[True][0], atol=1e-12, rtol=0)
    assert torch.allclose(grads[False][1], grads[True][1], atol=1e-12, rtol=0)
    assert torch.allclose(grads[False][2], grads[True][2], rtol=1e-10, atol=1e-12)


def _retained_elements(t):
    """Elements of the distinct tensors saved by the autograd graph of ``t``."""
    seen, ptrs, total = set(), set(), 0
    stack = [t.grad_fn]
    while stack:
        fn = stack.pop()
        if fn is None or fn in seen:
            continue
        seen.add(fn)
        saved = []
        if hasattr(fn, "saved_tensors"):                 # custom autograd functions
            try:
                saved += [v for v in fn.saved_tensors if torch.is_tensor(v)]
            except RuntimeError:
                pass
        for attr in dir(fn):                             # built-in nodes
            if attr.startswith("_saved_"):
                v = getattr(fn, attr, None)
                if torch.is_tensor(v):
                    saved.append(v)
        for v in saved:
            key = (v.data_ptr(), v.numel())
            if key not in ptrs:
                ptrs.add(key)
                total += v.numel()
        stack.extend(nxt for nxt, _ in fn.next_functions)
    return total


def test_training_mode_retains_only_inputs():
    """With create_graph=True the plain loop keeps every triplet intermediate
    for the second backward; the recompute blocks keep their inputs only."""
    pos, z = _molecule(80, seed=2, box=7.0)
    kept, counts = {}, None
    for chunked in (False, True):
        m = ForceStressOutput(D4Dispersion(checkpoint_triplets=chunked, **MOL))
        m.train()
        g = _graph(pos, z, m.cutoff)
        out = m(g)                                       # forward + first backward, graph kept
        kept[chunked] = _retained_elements(out["forces"])
        counts = torch.bincount(g.edge_index[1], minlength=len(z))
    n_triplets = int((counts * (counts - 1) // 2).sum())
    n_edges = int(counts.sum())
    assert kept[False] > 10 * n_triplets
    assert kept[True] < 200 * n_edges + 300 * len(z) ** 2
    assert kept[True] < kept[False] / 10


def test_recompute_pairs_match_plain_two_body_in_training_mode():
    """Edge blocks of the two-body term equal the plain evaluation, including
    the force-loss gradients of the damping parameters."""
    pos, z = _molecule(90, seed=7, box=8.0)
    res = {}
    for chunked in (False, True):
        m = D4Dispersion(recompute_pairs=chunked, trainable=True, s9=0.0, **MOL)
        if chunked:                                   # small blocks on this instance only
            import functools
            m.term._two_body_chunked = functools.partial(m.term._two_body_chunked, chunk=257)
        fs = ForceStressOutput(m)
        fs.train()
        out = fs(_graph(pos, z, m.cutoff))
        (out["forces"] ** 2).sum().backward()
        res[chunked] = (out["energy"].detach(), out["forces"].detach(),
                        torch.stack([m.term.s6.grad, m.term.s8.grad, m.term.a1.grad, m.term.a2.grad]))
    assert torch.allclose(res[False][0], res[True][0], atol=1e-12, rtol=0)
    assert torch.allclose(res[False][1], res[True][1], atol=1e-12, rtol=0)
    assert torch.allclose(res[False][2], res[True][2], rtol=1e-10, atol=1e-12)
    # the exported head pins the plain path
    from xnn.common.deploy.torchscript import _DispersionHead
    assert _DispersionHead(D4Dispersion(**MOL).term).term.recompute_pairs is False


@pytest.mark.parametrize("L", [7.0, 11.0])
def test_per_edge_c6_third_side_lookup_on_periodic_cells(L):
    """The per-edge C6 path finds the third side of every kept triplet through
    the packed (dst, src, shift) keys, including cells shorter than twice the
    cutoff where a pair has several images; energies, forces and stress equal
    the plain loop to rounding."""
    from xnn.common.models.dispersion import edge_cell_shifts, pair_edge_keys
    pos, z, cell = _crystal(n=40, L=L)
    plain = _run(D4Dispersion(regime="dense", checkpoint_triplets=False, **PER), pos, z, cell, stress=True)
    new = _run(D4Dispersion(regime="dense", **PER), pos, z, cell, stress=True)
    assert abs(float(new["energy_3body"] - plain["energy_3body"])) < 1e-13
    assert torch.allclose(new["forces"], plain["forces"], atol=1e-13, rtol=0)
    assert torch.allclose(new["stress"], plain["stress"], atol=1e-15, rtol=0)
    # recovered shifts reproduce the graph's edge vectors and give unique keys
    g = _graph(pos, z, 11.0, cell)
    shifts = edge_cell_shifts(g.pos, g.edge_index, g.edge_vectors(), g.cell, g.batch)
    rebuilt = g.pos[g.edge_index[1]] - g.pos[g.edge_index[0]] + shifts.to(g.pos.dtype) @ g.cell[0]
    assert torch.allclose(rebuilt, g.edge_vectors(), atol=1e-12, rtol=0)
    keys = pair_edge_keys(g.edge_index, shifts, len(z))
    assert torch.unique(keys).numel() == keys.numel()


def test_analytic_block_gradient_matches_autograd():
    """The closed-form gradient of the per-edge block (forces, C6, radii, a1,
    a2, s9) equals autograd through the block, in eval mode and through the
    force-training double backward."""
    import functools
    from xnn.common.models import dispersion as disp
    rng = np.random.default_rng(11)
    pos, z, cell = rng.uniform(0, 9.0, (60, 3)), ([8, 1, 1, 6] * 15), np.eye(3) * 9.0
    res = {}
    for analytic in (True, False):
        m = D4Dispersion(regime="dense", trainable=True, switch_width_pair=2.0, switch_width_triple=1.5, **PER)
        chunked = functools.partial(disp.three_body_energy_chunked, analytic=analytic, chunk=4096)
        m.term._three_body_chunked = functools.partial(_three_body_with, m.term, chunked)
        fs = ForceStressOutput(m, compute_stress=True)
        fs.train()
        out = fs(_graph(pos, z, m.cutoff, cell))
        (out["forces"] ** 2).sum().backward()
        res[analytic] = (out["energy_3body"].detach(), out["forces"].detach(), out["stress"].detach(),
                         torch.stack([m.term.s9.grad, m.term.a1.grad, m.term.a2.grad, m.term.s8.grad]))
    assert torch.allclose(res[True][0], res[False][0], atol=1e-13, rtol=0)
    assert torch.allclose(res[True][1], res[False][1], atol=1e-11, rtol=0)
    assert torch.allclose(res[True][2], res[False][2], atol=1e-13, rtol=0)
    assert torch.allclose(res[True][3], res[False][3], rtol=1e-9, atol=1e-12)


def _three_body_with(term, chunked, z, pos, edge_index, edge_vec, r, alpha_neutral, cell, batch, n_atoms):
    """DFTD4._three_body_chunked with a chosen chunked driver (test helper)."""
    import math
    from xnn.common.models.dispersion import edge_cell_shifts
    alpha_a = (3.0 / math.pi) * alpha_neutral * term.cp_weights
    c6_edge = (alpha_a[edge_index[1]] * alpha_neutral[edge_index[0]]).sum(-1)
    shifts = edge_cell_shifts(pos, edge_index, edge_vec, cell, batch)
    r0_atom = (3.0 ** 0.25) * torch.sqrt(term.r4r2[z])
    return chunked(z, edge_index, edge_vec, r, None, term.s9, term.alp / 3.0, term.cutoff_triple / term.bohr,
                   term.switch_width_triple / term.bohr, n_atoms, r0_atom=r0_atom, a1=term.a1, a2=term.a2,
                   c6_edge=c6_edge, edge_shift=shifts)
