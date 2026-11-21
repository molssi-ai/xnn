"""NDDO semiempirical methods (sqmnn): the forms of the SCF iterations and of its density response agree.

The static (fixed-buffer) forms of the DIIS extrapolation, the SCF loop and the coupled-perturbed iterations,
which the CUDA graphs of the fast path capture, compute what the host-counted forms compute; the stage dispatch
of a graphed loop reproduces the launched loop and its buffers take each evaluation's electron counts, and the
response iterations in pairs of steps stop where step by step they would; the launched derivatives of a graphed
function are partial derivatives in each input; the safety nets of the iterations (several starts, the aufbau
check after orbital rotations) reach the ground state; and the caches of a model start empty in its copies.
"""
import copy
import pickle
import types
import warnings
from functools import partial

import numpy as np
import pytest
import torch

from xnn.common.data import collate, structure_to_graph
from xnn.common.models import ForceStressOutput
from xnn.sqmnn.models import PM6, PM7, graphed, nddo, scf

CCH = (np.array([[0.67291273451353, 0.0, 0.0], [-0.53840317956358, 0.0, 0.0], [-1.60293110389655, 0.0, 0.0]]),
       [6, 6, 1])
FCCH = (np.array([[0.0, 0.0, -0.09544399299461], [0.0, 0.0, -1.28414890887293], [0.0, 0.0, 1.18037991640086],
                  [0.0, 0.0, -2.34586283337243]]), [6, 6, 9, 1])
WATER = (np.array([[0.0, 0.0, 0.12], [0.0, 0.76, -0.47], [0.05, -0.75, -0.49]]), [8, 1, 1])
FORMAMIDE = (np.array([[0, 0.42, 0], [1.2, 0.5, 0.02], [-0.72, -0.73, 0], [-0.55, 1.36, 0.01],
                       [-1.73, -0.69, 0.05], [-0.27, -1.63, -0.03]]), [6, 8, 7, 1, 1, 1])
METHYL = (np.array([[0, 0, 0.02], [1.08, 0, 0], [-0.54, 0.94, 0.01], [-0.55, -0.93, 0]]), [6, 1, 1, 1])


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


def _scf_raw_inputs(monkeypatch, graph):
    """The inputs of a model's SCF of ``graph``: layout, core Hamiltonian, one- and two-center integrals,
    electrons of each spin, whether unrestricted, and the initial density."""
    calls = []

    def record(*args, **kwargs):
        calls.append(args)
        return scf.run_scf(*args, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(nddo, "run_scf", record)
        PM6(reuse_density=False)(graph)
    layout, H, W1, W2, n_alpha, n_beta, uhf, guess = calls[0][:8]
    return layout, H.detach(), W1.detach(), W2.detach(), n_alpha, n_beta, uhf, guess


def _scf_inputs(monkeypatch, graph):
    """:func:`_scf_raw_inputs` with the integrals packed."""
    layout, H, W1, W2, *rest = _scf_raw_inputs(monkeypatch, graph)
    return (layout, H, scf.pack_one_center(W1), scf.pack_pairs(layout, W2), *rest)


def test_diis_static_form_matches_host_count():
    """The static DIIS (device-side count, whole ring with empty slots decoupled) gives the extrapolations of the
    host-counted one, before and after the ring wraps and after a reset."""
    rng = np.random.default_rng(0)
    forms = [scf.DIIS(2, [1, 4, 4], 16, 5, torch.float64, torch.device("cpu"), static) for static in (False, True)]
    for sweep in range(2):
        for d in forms:
            d.reset()
        for step in range(8):
            F = torch.tensor(rng.normal(size=(2, 1, 4, 4)))
            err = torch.tensor(rng.normal(size=(2, 1, 4, 4))) * 0.5 ** step
            host, static = (d.extrapolate(F, err) for d in forms)
            assert torch.allclose(static, host, rtol=1e-9, atol=1e-10), (sweep, step)
            assert step < 1 or not torch.equal(host, F)


@pytest.mark.parametrize("ahead", [False, True])
@pytest.mark.parametrize("molecules", [[(FORMAMIDE, None)], [(FORMAMIDE, None), (METHYL, 2.0)]])
def test_static_loop_and_stage_dispatch_match_launched_loop(monkeypatch, molecules, ahead):
    """The SCF loop on fixed buffers, driven stage by stage through the dispatch of a graphed loop (smeared
    iterations, diagonalizations and orbital rotations; with the Fock step of the next iteration run ahead),
    converges to the launched loop's solution and keeps its buffers (closed shells and a batch with an open
    shell)."""
    graph = collate([_graph(*mol, mult=mult) for mol, mult in molecules])
    layout, H, W1, W2, n_alpha, n_beta, uhf, guess = _scf_inputs(monkeypatch, graph)
    assert uhf == (len(molecules) > 1)
    settings = (300, 1e-10, 1e-11)
    ref = scf.scf_solve(layout, H, W1, W2, n_alpha, n_beta, uhf, guess, *settings, 8, 2.0, 5, None, 0.05, 0)
    P, n_occ, weight, filled, blocks = scf.scf_start(guess, n_alpha, n_beta, uhf, None)
    loop = scf.SCFLoop(layout, H, W1, W2, n_occ, weight, uhf, 8, filled, blocks, static=True)
    buffers = (loop.P, loop.C, loop.Fx, loop.dP)
    loop.start(P)
    # the dispatch of SCFGraphs (the merged end of an iteration included) on launched stages; with host buffers
    # the next Fock step runs ahead of the changes' read
    host = tuple(torch.empty_like(x) for x in (loop.Fx, loop.eps, loop.C_new)) if ahead else None
    stand_in = types.SimpleNamespace(loop=loop, check=H.new_zeros(2), host=host, host_check=H.new_zeros(2),
                                     dE_kept=loop.dE.clone())
    stand_in.run = lambda stage: (scf.SCFGraphs._occupy_commit(stand_in) if stage == "occupy_commit"
                                  else getattr(loop, stage)())
    stand_in.diagonalize = lambda prefetched=False: scf.SCFGraphs.diagonalize(stand_in, prefetched)
    got = scf.solve_scf(scf._GraphedLoop(stand_in), layout, H, W1, W2, n_occ, weight, uhf, 8, filled, blocks,
                        *settings, 2.0, 5, 0.05)
    assert bool(got.converged.all()) and bool(ref.converged.all())
    assert torch.allclose(got.electronic, ref.electronic, rtol=0.0, atol=1e-8)
    assert torch.allclose(got.densities, ref.densities, rtol=0.0, atol=1e-6)
    assert all(a is b for a, b in zip(buffers, (loop.P, loop.C, loop.Fx, loop.dP)))


def test_graph_buffers_take_the_loaded_electron_counts(monkeypatch):
    """Loading a new evaluation into the buffers of SCF graphs sets its electron counts and the aufbau occupations
    that the occupy stage reads (cached at its first launch, before the recording), so that one bucket's graphs
    serve batches of other electron counts."""
    graph = collate([_graph(*FORMAMIDE)])
    layout, H, W1, W2, n_alpha, n_beta, uhf, guess = _scf_inputs(monkeypatch, graph)
    P, n_occ, weight, filled, blocks = scf.scf_start(guess, n_alpha, n_beta, uhf, None)
    loop = scf.SCFLoop(layout, H, W1, W2, n_occ.clone(), weight, uhf, 8, filled, blocks, static=True)
    loop.start(P)
    loop.fock_step()
    loop.diagonalize()
    loop.occupy()
    first = loop.aufbau.clone()
    graphs = object.__new__(scf.SCFGraphs)
    graphs.layout, graphs.loop, graphs.H, graphs.W1, graphs.W2 = None, loop, loop.H, loop.W1, loop.W2
    fewer = n_occ - 1
    graphs.load(H, W1, W2, P, n_occ=fewer)
    assert torch.equal(loop.n_occ, fewer)
    assert torch.equal(loop.aufbau, scf._occupations(loop.eps, fewer, weight))
    assert float(first.sum() - loop.aufbau.sum()) == pytest.approx(weight * float(n_occ.numel()))


def test_cphf_static_form_matches(monkeypatch):
    """The coupled-perturbed iterations on fixed buffers give the solution of the rebinding form (an open-shell
    batch, both spins coupled)."""
    graph = collate([_graph(*FORMAMIDE), _graph(*METHYL, mult=2.0)])
    layout, H, W1, W2, n_alpha, n_beta, uhf, guess = _scf_inputs(monkeypatch, graph)
    res = scf.scf_solve(layout, H, W1, W2, n_alpha, n_beta, uhf, guess, 300, 1e-10, 1e-11, 8, 2.0, 25, None,
                        0.0, 0)
    gP = torch.tensor(np.random.default_rng(3).normal(size=res.densities.shape))
    loops = [scf.CPHFLoop(layout, W1, W2, res.orbitals, res.energies, res.occupations, gP, uhf, static)
             for static in (False, True)]
    kept = loops[1].u
    for loop in loops:
        loop.start()
        for _ in range(40):
            loop.step()
    assert loops[1].u is kept
    assert float(loops[0].res) < 1e-8
    for name in ("u", "r", "rz", "res"):
        assert torch.allclose(getattr(loops[1], name), getattr(loops[0], name), rtol=1e-10, atol=1e-12), name


@pytest.mark.parametrize("tol", [1e-4, 1e-7, 1e-10])
def test_cphf_step_pairs_stop_where_single_steps_do(monkeypatch, tol):
    """Two steps per read of the residuals (the dispatch of the response graphs, on launched steps) give the
    solution and flags of one step per read, whether the first or the second step of a pair meets the tolerance."""
    graph = collate([_graph(*FORMAMIDE), _graph(*METHYL, mult=2.0)])
    layout, H, W1, W2, n_alpha, n_beta, uhf, guess = _scf_inputs(monkeypatch, graph)
    res = scf.scf_solve(layout, H, W1, W2, n_alpha, n_beta, uhf, guess, 300, 1e-10, 1e-11, 8, 2.0, 25, None,
                        0.0, 0)
    gP = torch.tensor(np.random.default_rng(3).normal(size=res.densities.shape))
    single = scf.CPHFLoop(layout, W1, W2, res.orbitals, res.energies, res.occupations, gP, uhf, True)
    single.start()
    for _ in range(200):
        single.step()
        if float(single.res) < tol:
            break
    loop = scf.CPHFLoop(layout, W1, W2, res.orbitals, res.energies, res.occupations, gP, uhf, True)
    stand_in = types.SimpleNamespace(loop=loop, kept_u=loop.u.clone(), kept_unstable=loop.unstable.clone(),
                                     kept_res=loop.res.clone(), load=lambda *args: loop)
    stand_in.run = lambda stage: scf.CPHFGraphs._step(stand_in) if stage == "step" else loop.start()
    stand_in.step_pair = lambda t: scf.CPHFGraphs.step_pair(stand_in, t)
    paired = scf._cphf(layout, W1, W2, res, gP, uhf, tol, stand_in)
    assert torch.equal(paired.u, single.u) and torch.equal(paired.unstable, single.unstable)


def test_open_shell_parameter_gradients_through_scf_response():
    """Parameter gradients of the energy, forces and charges of an unrestricted (doublet) calculation, through
    the density response of both spins, equal finite differences."""
    pos, z = METHYL

    def values(model):
        out = ForceStressOutput(model)(_graph(pos, z, mult=2.0))
        w = torch.linspace(0.3, 1.1, out["forces"].numel()).reshape(out["forces"].shape)
        return [out["energy"].sum(), (out["forces"] * w).sum(), (out["charges"] * torch.arange(len(z))).sum()]

    model = PM6(trainable=["USS"], tol_density=1e-12, tol_energy=1e-13)
    model.train()
    grads = []
    for v in values(model):
        model.zero_grad()
        v.backward(retain_graph=True)
        grads.append(float(model.params["USS"].grad[6]))
    h = 1e-5
    fd = []
    for s in (1, -1):
        m = PM6(trainable=["USS"], tol_density=1e-12, tol_energy=1e-13)
        with torch.no_grad():
            m.params["USS"][6] += s * h
        m.eval()
        fd.append([float(v) for v in values(m)])
    for g, vp, vm in zip(grads, *fd):
        assert g == pytest.approx((vp - vm) / (2 * h), rel=1e-5, abs=1e-8)


def test_open_shell_pseudo_diagonalization(monkeypatch):
    """Orbital rotations of a doublet (different occupied counts per spin) reach the solution of full
    diagonalizations."""
    graph = _graph(*METHYL, mult=2.0)
    monkeypatch.setattr(scf, "PSEUDO_MIN_BASIS", 0)
    calls = []
    rotate = scf._pseudo_diagonalize_blocks
    monkeypatch.setattr(scf, "_pseudo_diagonalize_blocks", lambda *a, **k: calls.append(a[2]) or rotate(*a, **k))
    model = ForceStressOutput(PM6(reuse_density=False))
    ref = model(graph)
    assert set(calls) == {4, 3}
    monkeypatch.setattr(nddo, "run_scf", partial(scf.run_scf, pseudo_diagonalization=0.0))
    full = model(graph)
    assert torch.allclose(ref["energy"], full["energy"], rtol=0.0, atol=1e-7)
    assert torch.allclose(ref["forces"], full["forces"], rtol=0.0, atol=1e-6)


def test_rotated_end_state_of_another_occupation_restarts(monkeypatch):
    """Iterations that end, after orbital rotations, on a determinant other than the aufbau occupation of its own
    Fock matrix (the deepest occupied and the highest virtual orbital swapped) are caught by the aufbau check: the
    occupied and virtual blocks do not separate, and full diagonalizations go on to the ground state."""
    layout, H, W1, W2, n_alpha, n_beta, uhf, guess = _scf_inputs(monkeypatch, _graph(*FORMAMIDE))
    settings = (300, 1e-10, 1e-11)
    ground = scf.scf_solve(layout, H, W1, W2, n_alpha, n_beta, uhf, guess, *settings, 8, 2.0, 0, None, 0.0, 0)
    n = H.shape[-1]
    C = ground.orbitals.clone()
    C[..., [0, n - 1]] = C[..., [n - 1, 0]]
    occ = ground.occupations
    P, n_occ, weight, filled, blocks = scf.scf_start(scf.density(C, occ)[0], n_alpha, n_beta, uhf, None)
    loop = scf.SCFLoop(layout, H, W1, W2, n_occ, weight, uhf, 8, filled, blocks)
    loop.start(P, C, occ)
    # the first iterations end where they start, on the rotated excited determinant
    iterate = scf._iterate
    monkeypatch.setattr(scf, "_iterate", lambda lp, *a, **k: (3, True) if lp is loop else iterate(lp, *a, **k))
    separated = []
    block_pairs = scf._block_eigenpairs
    monkeypatch.setattr(scf, "_block_eigenpairs",
                        lambda *a: (lambda out: separated.append(out is not None) or out)(block_pairs(*a)))
    got = scf.solve_scf(loop, layout, H, W1, W2, n_occ, weight, uhf, 8, filled, blocks, *settings, 2.0, 0, 0.05,
                        True)
    assert separated == [False]
    assert got.iterations > 3 and bool(got.converged.all())
    assert torch.allclose(got.electronic, ground.electronic, rtol=0.0, atol=1e-8)
    assert torch.allclose(got.densities, ground.densities, rtol=0.0, atol=1e-6)


def test_response_of_a_saddle_point_solution_warns(monkeypatch):
    """The density response of an SCF solution that is not a minimum (an excited aufbau state, reached by orbital
    rotations from the determinant with HOMO and LUMO swapped) warns that it is incomplete; that of the ground
    state does not."""
    layout, H, W1, W2, n_alpha, n_beta, uhf, guess = _scf_raw_inputs(monkeypatch, _graph(*FORMAMIDE))
    W1p, W2p = scf.pack_one_center(W1), scf.pack_pairs(layout, W2)
    ground = scf.scf_solve(layout, H, W1p, W2p, n_alpha, n_beta, uhf, guess, 300, 1e-10, 1e-11, 8, 2.0, 0, None,
                           0.0, 0)
    h = int(n_alpha[0])
    C = ground.orbitals.clone()
    C[..., [h - 1, h]] = C[..., [h, h - 1]]
    occ = ground.occupations
    rotate = scf._pseudo_diagonalize
    with monkeypatch.context() as m:
        m.setattr(scf, "_pseudo_diagonalize", lambda *a, **k: (rotate(*a, **k)[0], torch.zeros((), dtype=torch.bool)))
        saddle = scf.scf_solve(layout, H, W1p, W2p, n_alpha, n_beta, uhf, scf.density(C, occ)[0], 300, 1e-10,
                               1e-11, 8, 2.0, 0, None, 0.05, 0, C, occ)
    assert bool(saddle.converged.all()) and float(saddle.electronic[0] - ground.electronic[0]) > 1.0
    for result, warns in ((saddle, True), (ground, False)):
        Hg = H.clone().requires_grad_(True)
        P = scf.SCFDensity.apply(Hg, W1, W2, result, layout, uhf, 1e-10)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.autograd.grad((P * torch.linspace(-1.0, 1.0, P.numel()).reshape(P.shape)).sum(), Hg)
        assert any("not a minimum" in str(w.message) for w in caught) == warns


def test_lowest_takes_the_converged_lower_energy_per_structure():
    """Of several SCF results, each structure takes the converged one of lowest electronic energy."""
    def result(e, converged, fill):
        e = torch.tensor(e)
        return scf.SCFResult(torch.full((1, 2, 3, 3), fill), torch.full((1, 2, 3, 3), fill),
                             torch.full((1, 2, 3), fill), torch.full((1, 2, 3), fill), 7 + int(fill),
                             torch.tensor(converged), e)
    first = result([-1.0, -2.0], [True, True], 0.0)
    best = scf.lowest([first, result([-3.0, -1.0], [True, True], 1.0), result([-9.0, -9.0], [False, False], 2.0)])
    assert best.electronic.tolist() == [-3.0, -2.0] and best.converged.tolist() == [True, True]
    assert best.densities[0, :, 0, 0].tolist() == [1.0, 0.0] and best.iterations == 8
    assert scf.lowest([first, result([-1.0, -2.0], [True, True], 1.0)]) is first


def test_restart_rule():
    """A cold start restarts when unrestricted, not converged or of a small frontier gap, and otherwise not."""
    def result(gap, converged):
        z = torch.zeros(1, 1, 3, 3)
        return scf.SCFResult(z, z, torch.tensor([[[-10.0, -5.0, -5.0 + gap]]]), torch.tensor([[[2.0, 2.0, 0.0]]]),
                             5, torch.tensor([converged]), torch.tensor([-1.0]))
    assert not scf.needs_restart(result(8.0, True), False, 6.5)
    assert scf.needs_restart(result(8.0, True), True, 6.5)
    assert scf.needs_restart(result(8.0, False), False, 6.5)
    assert scf.needs_restart(result(4.0, True), False, 6.5)
    assert not scf.needs_restart(result(4.0, True), False, 3.0)


def test_excited_closed_shell_solution_restarts():
    """PM7 fluoroacetylene: the plain first run settles in an excited closed-shell solution (gap about 3.2 eV); the
    default rule restarts and reaches MOPAC's ground state, about 250 kcal/mol lower."""
    g = _graph(*FCCH)
    old = float(PM7(restart_gap=3.0, reuse_density=False)(g)["heat_of_formation"][0])
    new = float(PM7(reuse_density=False)(g)["heat_of_formation"][0])
    every = float(PM7(restart_gap=float("inf"), reuse_density=False)(g)["heat_of_formation"][0])
    assert old - new > 200.0
    assert abs(new - every) < 1e-8
    assert abs(new - 24.29451) < 1e-3                           # MOPAC's heat of formation


def test_unconverged_first_run_restarts(monkeypatch):
    """A first run that did not converge brings the further starts even at a large gap."""
    calls = []
    real = nddo.run_scf

    def counted(*args, fail_first=False, **kwargs):
        res = real(*args, **kwargs)
        calls.append(kwargs.get("smearing"))
        if fail_first and len(calls) == 1:
            res = res._replace(converged=torch.zeros_like(res.converged))
        return res
    g = _graph(*WATER)
    monkeypatch.setattr(nddo, "run_scf", counted)
    PM6(reuse_density=False)(g)
    assert len(calls) == 1
    calls.clear()
    monkeypatch.setattr(nddo, "run_scf", partial(counted, fail_first=True))
    out = PM6(reuse_density=False)(g)
    assert len(calls) == 3 and bool(out["scf_converged"].all())


def test_run_stopped_while_annealing_is_not_converged():
    """A run whose iteration limit ends inside the annealing schedule is not converged, whatever its last change."""
    out = PM6(reuse_density=False, max_iter=10, scf_starts=((1.0, 80),))(_graph(*WATER))
    assert out["scf_converged"].dtype == torch.bool and not bool(out["scf_converged"].any())


def _first_scf_inputs(monkeypatch, graph):
    """Layout, electron counts, unrestricted flag and initial guess of a model's first SCF run of ``graph``."""
    seen = []
    real = nddo.run_scf

    def spy(layout, H, W1, W2, n_alpha, n_beta, uhf, guess, *args, **kwargs):
        seen.append((layout, n_alpha, n_beta, uhf, guess))
        return real(layout, H, W1, W2, n_alpha, n_beta, uhf, guess, *args, **kwargs)
    monkeypatch.setattr(nddo, "run_scf", spy)
    PM6(reuse_density=False)(graph)
    monkeypatch.setattr(nddo, "run_scf", real)
    return seen[0]


@pytest.mark.parametrize("open_shell", (False, True))
def test_search_guesses_keep_electron_counts(monkeypatch, open_shell):
    """The perturbed starts are deterministic, symmetric, zero on the padding, and the occupation kicks hold each
    structure's electrons of each spin."""
    mols = ([_graph(*CCH, mult=2), _graph(*METHYL, mult=2)] if open_shell
            else [_graph(*WATER), _graph(*FORMAMIDE)])
    layout, na, nb, uhf, guess = _first_scf_inputs(monkeypatch, collate(mols))
    assert uhf == open_shell
    real = layout.valid.to(guess.dtype)
    pad = 1.0 - real[:, :, None] * real[:, None, :]
    for i in range(4):
        start, spin = scf.search_guess(guess, layout, na, nb, uhf, i)
        again, _ = scf.search_guess(guess, layout, na, nb, uhf, i)
        assert torch.equal(start, again) and not torch.allclose(start, guess)
        assert torch.allclose(start, start.transpose(-1, -2), rtol=0.0, atol=1e-15)
        assert float((start * pad).abs().max()) == 0.0
        trace = start.diagonal(dim1=-2, dim2=-1).sum(-1)
        if i % 2 == 0:
            assert torch.allclose(trace, (na + nb).to(trace.dtype), rtol=1e-12)
            if uhf:
                spins = spin.diagonal(dim1=-2, dim2=-1).sum(-1)
                assert torch.allclose(spins, torch.stack([na, nb]).to(trace.dtype), rtol=1e-12)
                assert torch.allclose(spin.sum(0), start)
            else:
                assert spin is None
        else:
            assert spin is None
        assert scf.search_schedule(i) == ((1.0, 80) if i % 2 == 0 else (0.0, 0))


def test_search_starts_reach_the_ground_state():
    """PM7 ethynyl radical: every default start puts the minority-spin electron in a sigma orbital; the search
    starts reach MOPAC's ground state, about 21 kcal/mol lower, alone and in a batch."""
    g = _graph(*CCH, mult=2)
    plain = float(PM7(reuse_density=False)(g)["heat_of_formation"][0])
    found = float(PM7(reuse_density=False, search_starts=6)(g)["heat_of_formation"][0])
    assert plain - found > 20.0
    assert abs(found - 141.952795) < 1e-3                          # MOPAC's heat of formation
    water = float(PM7(reuse_density=False)(_graph(*WATER))["heat_of_formation"][0])
    both = PM7(reuse_density=False, search_starts=6)(collate([g, _graph(*WATER)]))["heat_of_formation"]
    assert abs(float(both[0]) - found) < 1e-6 and abs(float(both[1]) - water) < 1e-6


def test_exported_potential_runs_the_search_starts(tmp_path):
    """The TorchScript export (and its core called eagerly) runs its model's search starts and gives the eager
    model's energy."""
    from xnn.common.deploy import export_torchscript_potential
    eager = ForceStressOutput(PM7(reuse_density=False, search_starts=6)).eval()
    ref = eager(_graph(*CCH, mult=2))
    assert abs(float(ref["heat_of_formation"][0]) - 141.952795) < 1e-3
    core = PM7(search_starts=6).export_core()
    _, node_energy, _ = core.node_features_energy_charges(
        torch.tensor(CCH[1]), torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 3)), torch.tensor(CCH[0]),
        torch.zeros((3, 3)), torch.zeros(3, dtype=torch.bool), 0.0, 2.0)
    assert abs(float(node_energy.sum()) - float(ref["energy"].sum())) < 1e-9
    path = export_torchscript_potential(PM7(search_starts=6).eval(), cutoff=10.0, path=str(tmp_path / "m.pt"),
                                        spin_multiplicity=2.0)
    scripted = torch.jit.load(path)
    with torch.jit.optimized_execution(False):
        out = scripted(torch.tensor(CCH[0]), torch.tensor(CCH[1]))
    assert abs(float(out["energy"]) - float(ref["energy"].sum())) < 1e-9


def test_caches_start_empty_in_copies():
    """The SCF graph cache captures a composition after it recurs and keeps the most recent ones; the caches of a
    model (plans, graphs) start empty in its copies and pickles."""
    cache = scf.SCFGraphCache(capacity=2)
    built = []

    def build(key):
        return lambda: built.append(key) or ("graphs", key)
    for _ in range(scf.CAPTURE_AFTER):
        assert cache.get("a", build("a")) is None
    assert cache.get("a", build("a")) == ("graphs", "a")
    assert cache.get("a", build("a")) == ("graphs", "a") and built == ["a"]
    for key in "bcdef":
        for _ in range(scf.CAPTURE_AFTER + 1):
            cache.get(key, build(key))
    assert list(cache.entries) == ["e", "f"] and len(cache.seen) <= 4
    clone = copy.deepcopy(cache)
    restored = pickle.loads(pickle.dumps(cache))
    for c in (clone, restored):
        assert c.capacity == 2 and not c.entries and not c.seen
    cache.clear()
    assert not cache.entries and not cache.seen
    model = PM6()
    model._plans["key"] = "plan"
    model._scf_graphs.entries["key"] = "graphs"
    assert model._plans.__getstate__() == {}
    for other in (copy.deepcopy(model), pickle.loads(pickle.dumps(model))):
        assert type(other._plans) is type(model._plans) and not other._plans
        assert not other._scf_graphs.entries
        out = other(_graph(*WATER))
        assert float(out["energy"][0]) == pytest.approx(float(PM6()(_graph(*WATER))["energy"][0]), abs=1e-9)


def test_derived_quantities_with_replaced_leaves():
    """Replacing the tensors of nested derived quantities by their key paths leaves the original untouched."""
    a, b, c = torch.zeros(2), torch.ones(3), torch.full((1,), 2.0)
    q = {"a": a, "inner": {"b": b, "deeper": {"c": c}, "n": 4}, "label": "x"}
    leaves = nddo._tensor_leaves(q)
    paths = [path for path, _ in leaves]
    assert paths == [("a",), ("inner", "b"), ("inner", "deeper", "c")]
    new = nddo._with_leaves(q, paths, [v + 1.0 for _, v in leaves])
    assert torch.equal(new["inner"]["deeper"]["c"], c + 1.0) and torch.equal(new["a"], a + 1.0)
    assert new["inner"]["n"] == 4 and new["label"] == "x"
    assert q["inner"]["deeper"]["c"] is c and q["inner"]["b"] is b and q["a"] is a


def test_launched_derivatives_of_a_graphed_function_are_partial():
    """The launched first and second derivatives of a graphed function are partial derivatives in each input,
    also where one input depends on another (``f(a, b) = a b`` with ``b = a^2``: ``b`` in ``a``, not the total
    ``3 a^2``)."""
    graph = types.SimpleNamespace(fn=lambda a, b: (a * b,), grad_index=[0, 1], diff_index=[0])
    a = torch.tensor([1.5, -0.5], requires_grad=True)
    b = a * a
    cot = torch.tensor([1.0, 2.0])
    ga, gb = graphed._launched_vjp(graph, [a, b], [cot], True)
    assert torch.allclose(ga, cot * b) and torch.allclose(gb, cot * a)
    # the total derivative ga + gb db/da = 3 cot a^2 (of f = a^3) and its own derivative in a, 6 cot a
    (total,) = torch.autograd.grad((ga + gb * 2.0 * a).sum(), a)
    assert torch.allclose(total, 6.0 * cot * a)
    cc = torch.tensor([0.5, -1.0])
    h = graphed._launched_hvp(graph, [a, b], [cot], [cc, torch.zeros(2)])
    # gradients of <cc, cot b> in a (partial: 0), b (cot cc) and cot (b cc)
    assert torch.allclose(h[0], torch.zeros(2)) and torch.allclose(h[1], cot * cc) and torch.allclose(h[2], b * cc)
