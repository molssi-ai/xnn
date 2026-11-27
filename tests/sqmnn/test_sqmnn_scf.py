"""NDDO semiempirical methods (sqmnn): the forms of the SCF iterations and of its density response agree.

The static (fixed-buffer) forms of the DIIS extrapolation, the SCF loop and the coupled-perturbed iterations,
which the CUDA graphs of the fast path capture, compute what the host-counted forms compute; the stage dispatch
of a graphed loop reproduces the launched loop and its buffers take each evaluation's electron counts, and the
response iterations in pairs of steps stop where step by step they would; the launched derivatives of a graphed
function are partial derivatives in each input; the safety nets of the iterations (several starts, the aufbau
check after orbital rotations) reach the ground state; and the caches of a model start empty in its copies.
"""
import copy
import gc
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


def _scf_raw_inputs(monkeypatch, graph, **options):
    """The inputs of a model's SCF of ``graph`` (its first run): layout, core Hamiltonian, one- and two-center
    integrals, electrons of each spin, whether unrestricted, and the initial density."""
    calls = []

    def record(*args, **kwargs):
        calls.append(args)
        return scf.run_scf(*args, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(nddo, "run_scf", record)
        PM6(reuse_density=False, **options)(graph)
    layout, H, W1, W2, n_alpha, n_beta, uhf, guess = calls[0][:8]
    return layout, H.detach(), W1.detach(), W2.detach(), n_alpha, n_beta, uhf, guess


def _scf_inputs(monkeypatch, graph, **options):
    """:func:`_scf_raw_inputs` with the integrals packed."""
    layout, H, W1, W2, *rest = _scf_raw_inputs(monkeypatch, graph, **options)
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
    # a batch with an open shell as one unrestricted run (a mixed batch otherwise runs its closed shells apart)
    layout, H, W1, W2, n_alpha, n_beta, uhf, guess = _scf_inputs(monkeypatch, graph,
                                                                 unrestricted=len(molecules) > 1)
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
    stand_in = types.SimpleNamespace(loop=loop, check=H.new_zeros(3), host=host, host_check=H.new_zeros(3),
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
    monkeypatch.setattr(scf, "_iterate", lambda lp, *a, **k: (3, True, False) if lp is loop else iterate(lp, *a, **k))
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
    assert len(calls) == 2 and bool(out["scf_converged"].all())          # the further starts in one run


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


class _Graphs:
    def __init__(self, key):
        self.key = key

    def __eq__(self, other):
        return isinstance(other, _Graphs) and other.key == self.key

    __hash__ = object.__hash__


def _fake_gpu(monkeypatch, total=1000):
    """The CUDA memory queries of the graph cache on a CPU: a device of ``total`` bytes, the allocated count in
    the returned list."""
    used = [0]
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: types.SimpleNamespace(total_memory=total))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *args: used[0])
    return used


def test_host_eigh_only_for_one_or_two_small_matrices():
    """A GPU hands one or two small eigenvalue problems to the host and solves batches itself."""
    assert scf._host_eigh(200) and scf._host_eigh(100, 2)
    assert not scf._host_eigh(201) and not scf._host_eigh(101, 2) and not scf._host_eigh(12, 3)


def test_graph_memory_budget(monkeypatch):
    """No capture starts while the graphs held take the cache's share of the device memory; it resumes once
    evicted graphs are released, and charges to one owner add up."""
    used = _fake_gpu(monkeypatch)
    cache = scf.SCFGraphCache(capacity=8, memory=0.25)

    def build(key, size):
        def make():
            used[0] += size
            return _Graphs(key)
        return make
    for _ in range(scf.CAPTURE_AFTER):
        assert cache.get("a", build("a", 300)) is None
    held = cache.get("a", build("a", 300))
    assert sum(cache.held.values()) == 300 and not cache.room()
    for _ in range(scf.CAPTURE_AFTER + 1):
        assert cache.get("b", build("b", 10)) is None
    cache.entries.clear()
    del held
    gc.collect()
    assert cache.room() and cache.get("b", build("b", 10)) == _Graphs("b")
    owner = _Graphs("build")

    def capture():
        used[0] += 40
        return True
    assert cache.charged(capture, owner) and cache.charged(capture, owner) and cache.held[owner] == 80
    restored = pickle.loads(pickle.dumps(cache))
    assert restored.memory == 0.25 and not restored.held and not restored.entries
    old = scf.SCFGraphCache.__new__(scf.SCFGraphCache)
    old.__setstate__({"capacity": 3, "entries": {}, "seen": {}})
    assert old.capacity == 3 and old.memory == scf.GRAPH_MEMORY


def test_second_order_captures_wait_for_memory(monkeypatch):
    """A build graph is captured, and records its differentiable derivatives, only where the model's graphs leave
    memory (asked again later otherwise)."""
    _fake_gpu(monkeypatch)
    model = PM6()
    calls = []

    class Graph:
        wants_second = True

        def capture_second(self):
            calls.append(1)
            return True
    graph = Graph()
    model._scf_graphs.memory = 0.0
    model._capture_second(graph)
    assert graph.wants_second is False and not calls
    model._capture_second(graph)
    graph.wants_second = True
    model._scf_graphs.memory = 1.0
    model._capture_second(graph)
    model._capture_second(None)
    assert calls == [1] and graph in model._scf_graphs.held
    # nor does a build graph start while the memory is taken
    holder = nddo._Bucket(("bucket",))
    holder.uses = scf.CAPTURE_AFTER + 1
    model._scf_graphs.memory = 0.0
    on_gpu = types.SimpleNamespace(is_cuda=True, requires_grad=False)
    assert model._build_graph(holder, "eval", None, {}, [on_gpu]) is None and not holder.graphs


def test_caches_start_empty_in_copies(monkeypatch):
    """The SCF graph cache captures a composition after it recurs and keeps the most recent ones; the caches of a
    model (plans, graphs) start empty in its copies and pickles."""
    _fake_gpu(monkeypatch)
    cache = scf.SCFGraphCache(capacity=2)
    built = []

    def build(key):
        return lambda: built.append(key) or _Graphs(key)
    for _ in range(scf.CAPTURE_AFTER):
        assert cache.get("a", build("a")) is None
    assert cache.get("a", build("a")) == _Graphs("a")
    assert cache.get("a", build("a")) == _Graphs("a") and built == ["a"]
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


def _recorded_runs(monkeypatch):
    """The SCF runs of NDDO models as (structures, unrestricted, smeared iterations)."""
    runs = []
    plain = nddo.run_scf

    def run(layout, H, *args, **kwargs):
        runs.append((int(H.shape[0]), bool(args[4]), kwargs.get("smear_steps")))
        return plain(layout, H, *args, **kwargs)
    monkeypatch.setattr(nddo, "run_scf", run)
    return runs


@pytest.mark.parametrize("split", [True, False])
def test_restarts_and_spin_forms_per_structure(monkeypatch, split):
    """A cold batch gives each structure its own result: only the structures whose first run calls for them
    (FCCH's excited solution, the methyl radical) get the further starts, with closed shells restricted apart
    from the open shells where the device splits them, else in one unrestricted run; with a bond-order term."""
    monkeypatch.setitem(nddo.SPLIT_SPINS, "cpu", split)
    settings = dict(reuse_density=False, tol_density=1e-11, tol_energy=1e-12,
                    yne_switch="off", bond_order_terms={"C-C": [11.09, 2.2, 2.65]})
    systems = [(FCCH, {}), (WATER, {}), (METHYL, {"mult": 2.0})]
    alone = [float(PM7(**settings)(_graph(*s, **kw))["heat_of_formation"][0]) for s, kw in systems]
    runs = _recorded_runs(monkeypatch)
    out = PM7(**settings)(collate([_graph(*s, **kw) for s, kw in systems]))
    assert out["heat_of_formation"].detach().numpy() == pytest.approx(alone, abs=1e-7)
    assert bool(out["scf_converged"].all())
    # the further starts of both schedules in one run (the structures once per schedule)
    if split:
        assert runs == [(2, False, 0), (2, False, 80), (1, True, 0), (2, True, 80)]
    else:
        assert runs == [(3, True, 0), (4, True, 80)]


def test_converged_structures_keep_their_state(monkeypatch):
    """In a batch, a structure stops at its own convergence: its density stays as it was while the others
    iterate, and its result is that of the structure alone."""
    settings = dict(reuse_density=False, scf_starts=((0.0, 0),), tol_density=1e-11, tol_energy=1e-12)
    states = []
    commit = scf.SCFLoop.commit

    def tracked(self):
        commit(self)
        states.append((bool(self.active[0]), self.P[:, 0].clone()))
    monkeypatch.setattr(scf.SCFLoop, "commit", tracked)
    out = PM7(**settings)(collate([_graph(*WATER), _graph(*FCCH)]))
    stopped = [P for active, P in states if not active]
    assert stopped and len(stopped) < len(states) and all(torch.equal(P, stopped[0]) for P in stopped)
    monkeypatch.setattr(scf.SCFLoop, "commit", commit)
    water = PM7(**settings)(_graph(*WATER))
    assert float(out["heat_of_formation"][0]) == pytest.approx(float(water["heat_of_formation"][0]), abs=1e-8)


def test_restart_schedules_in_one_run(monkeypatch):
    """Smearing schedules per structure in one batch reach the states of separate runs of each schedule."""
    layout, H, W1, W2, n_alpha, n_beta, uhf, guess = _scf_raw_inputs(monkeypatch, _graph(*FCCH))
    settings = (300, 1e-10, 1e-11)
    alone = [scf.run_scf(layout, H, W1, W2, n_alpha, n_beta, uhf, guess, *settings, smearing=kt, smear_steps=n)
             for kt, n in ((1.0, 80), (2.0, 25))]
    rep = torch.tensor([0, 0])
    sub, H2, W12, W22, _ = scf.restrict(layout, rep, H, W1, W2)
    both = scf.run_scf(sub, H2, W12, W22, n_alpha[rep], n_beta[rep], uhf, guess[rep], *settings, smearing=2.0,
                       smear_steps=80, schedule=(torch.tensor([1.0, 2.0]), torch.tensor([80, 25])))
    expected = torch.cat([r.electronic for r in alone])
    assert torch.allclose(both.electronic, expected, rtol=0.0, atol=1e-8) and bool(both.converged.all())


def test_diis_rows_restart_their_history():
    """The device form of DIIS forgets the history of the reset structures only: their extrapolations are those
    of a fresh DIIS, the others' go on."""
    rng = np.random.default_rng(1)
    shape = [1, 4, 4]
    data = [(torch.tensor(rng.normal(size=(2, *shape))), torch.tensor(rng.normal(size=(2, *shape))) * 0.5 ** k)
            for k in range(6)]
    kept = scf.DIIS(2, shape, 16, 4, torch.float64, torch.device("cpu"), True)
    fresh = scf.DIIS(2, shape, 16, 4, torch.float64, torch.device("cpu"), True)
    whole = scf.DIIS(2, shape, 16, 4, torch.float64, torch.device("cpu"), True)
    for F, err in data[:3]:
        kept.extrapolate(F, err)
        whole.extrapolate(F, err)
    kept.reset_rows(torch.tensor([True, False]))
    for F, err in data[3:]:
        out = kept.extrapolate(F, err)
        assert torch.allclose(out[0], fresh.extrapolate(F, err)[0], rtol=1e-12, atol=1e-12)
        assert torch.allclose(out[1], whole.extrapolate(F, err)[1], rtol=1e-12, atol=1e-12)


def test_stagnant_iterations_are_not_converged(monkeypatch):
    """A structure whose density and energy stop changing away from a stationary point is not converged: it
    stays active and takes a plain Fock step, and a final density with a large orbital gradient is flagged."""
    layout, H, W1, W2, n_alpha, n_beta, uhf, guess = _scf_inputs(monkeypatch, _graph(*FORMAMIDE))
    P, n_occ, weight, filled, blocks = scf.scf_start(guess, n_alpha, n_beta, uhf, None)
    loop = scf.SCFLoop(layout, H, W1, W2, n_occ, weight, uhf, 8, filled, blocks)
    loop.start(P)
    loop.settle(True, 1e-9, 1e-10)
    loop.fock_step()
    loop.diagonalize()
    loop.occupy()
    assert float(loop.dG.max()) > 1e-3                       # the guess is far from stationary
    loop.dP.zero_()                                          # changes that vanish (a stagnant extrapolation)
    loop.dE.zero_()
    loop.commit()
    assert bool(loop.stuck.all()) and bool(loop.active.all())
    loop.fock_step()
    assert torch.equal(loop.Fx, scf._fock_energy(layout, H, W1, W2, loop.P, uhf, None)[0])
    loop.dP.zero_()
    loop.dE.zero_()
    result, _ = scf._converged(loop, layout, H, W1, W2, n_occ, weight, uhf, filled, blocks, 1, False, 0, 300,
                               1e-9, 1e-10, 0.0)
    assert not bool(result.converged.any())
