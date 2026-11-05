"""NDDO semiempirical methods (sqmnn): the forms of the SCF iterations and of its density response agree.

The static (fixed-buffer) forms of the DIIS extrapolation, the SCF loop and the coupled-perturbed iterations,
which the CUDA graphs of the fast path capture, compute what the host-counted forms compute; the stage dispatch
of a graphed loop reproduces the launched loop; the safety nets of the iterations (several starts, the aufbau
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
from xnn.sqmnn.models import PM6, nddo, scf

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


@pytest.mark.parametrize("molecules", [[(FORMAMIDE, None)], [(FORMAMIDE, None), (METHYL, 2.0)]])
def test_static_loop_and_stage_dispatch_match_launched_loop(monkeypatch, molecules):
    """The SCF loop on fixed buffers, driven stage by stage through the dispatch of a graphed loop (smeared
    iterations, diagonalizations and orbital rotations), converges to the launched loop's solution and keeps its
    buffers (closed shells and a batch with an open shell)."""
    graph = collate([_graph(*mol, mult=mult) for mol, mult in molecules])
    layout, H, W1, W2, n_alpha, n_beta, uhf, guess = _scf_inputs(monkeypatch, graph)
    assert uhf == (len(molecules) > 1)
    settings = (300, 1e-10, 1e-11)
    ref = scf.scf_solve(layout, H, W1, W2, n_alpha, n_beta, uhf, guess, *settings, 8, 2.0, 5, None, 0.05, 0)
    P, n_occ, weight, filled, blocks = scf.scf_start(guess, n_alpha, n_beta, uhf, None)
    loop = scf.SCFLoop(layout, H, W1, W2, n_occ, weight, uhf, 8, filled, blocks, static=True)
    buffers = (loop.P, loop.C, loop.Fx, loop.dP)
    loop.start(P)
    stand_in = types.SimpleNamespace(loop=loop, run=lambda stage: getattr(loop, stage)())
    got = scf.solve_scf(scf._GraphedLoop(stand_in), layout, H, W1, W2, n_occ, weight, uhf, 8, filled, blocks,
                        *settings, 2.0, 5, 0.05)
    assert bool(got.converged.all()) and bool(ref.converged.all())
    assert torch.allclose(got.electronic, ref.electronic, rtol=0.0, atol=1e-8)
    assert torch.allclose(got.densities, ref.densities, rtol=0.0, atol=1e-6)
    assert all(a is b for a, b in zip(buffers, (loop.P, loop.C, loop.Fx, loop.dP)))


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
