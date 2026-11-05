"""The dispersion fast paths (``xnn.common.models.dispersion_fast``) against the reference.

The ATM kernels and the fast EEQ operator compute the same functions as the
reference code with another summation order, so energies, forces, stress and
charges must agree to rounding (float64) and to the solver tolerance (the
iterative EEQ solve). Systems: a small water cell (narrower than twice the
three-body cutoff, so it has self-image triangles), a larger cell (none) and a
water cluster. GPU tests skip without CUDA; the operator tests and the C++ kernels run on the CPU.
"""
import copy
import warnings

import numpy as np
import pytest
import torch

from xnn.common.data import structure_to_graph
from xnn.common.models import D3Dispersion, D4Dispersion, ForceStressOutput, fast
from xnn.common.models.dispersion_fast import supported

GPU = torch.cuda.is_available() and supported(torch.device("cuda"), torch.float64)
needs_gpu = pytest.mark.skipif(not GPU, reason="needs CUDA and Triton")

WATER = np.array([[0.0, 0.0, 0.119262], [0.0, 0.763239, -0.477047],
                  [0.0, -0.763239, -0.477047]])
SPACING = 3.104
# a 6 A three-body cutoff keeps the cells small (the reference's dense EEQ needs
# about 7 kB per atom pair): 9.3 A is narrower than twice the cutoff, 12.4 A is not
OPTIONS = dict(cutoff_pair=9.0, switch_width_pair=2.0, cutoff_triple=6.0,
               switch_width_triple=1.0, s9=1.0, cutoff_cn=9.0)
D4_ONLY = dict(cutoff_eeq_cn=9.0)


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _water(n_side, periodic, seed=3):
    rng = np.random.default_rng(seed)
    pos = []
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
                q *= np.sign(np.linalg.det(q))
                pos.extend((WATER - WATER[0]) @ q.T + np.array([i, j, k]) * SPACING
                           + rng.normal(scale=0.1, size=3))
    s = {"pos": np.array(pos), "atomic_numbers": [8, 1, 1] * n_side ** 3}
    if periodic:
        s["cell"] = np.eye(3) * SPACING * n_side
        s["pbc"] = [True, True, True]
    return s


SYSTEMS = {"small_cell": (3, True), "cell": (4, True), "cluster": (3, False)}


def _model(method, dtype=torch.float64, device="cuda", **extra):
    cls = D4Dispersion if method == "d4" else D3Dispersion
    opts = {**OPTIONS, **(D4_ONLY if method == "d4" else {}), **extra}
    # evaluation mode: create_graph=False, the kernels' first-order backward
    # (training mode asks for second derivatives: test_dispersion_fast_training.py)
    model = ForceStressOutput(cls(**opts), compute_stress=True).to(device=device, dtype=dtype)
    return model.eval()


def _graph(model, system, dtype=torch.float64, device="cuda"):
    n_side, periodic = SYSTEMS[system]
    s = _water(n_side, periodic)
    s = {k: (torch.tensor(v, dtype=dtype) if k in ("pos", "cell") else v) for k, v in s.items()}
    return structure_to_graph(s, model.model.cutoff, device=device)


def _pair(model, graph):
    fast.set_use_fast(model, False)
    ref = model(copy.copy(graph))
    fast.set_use_fast(model, True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", fast.FastPathPrecisionWarning)
        out = model(copy.copy(graph))
    return ref, out


def _cpu_compiles():
    from xnn.common.models.dispersion_fast.atm_cpu import available
    return available()


def _rel(a, b):
    scale = b.detach().abs().max().clamp(min=1e-30)
    return float((a.detach() - b.detach()).abs().max() / scale)


# the ATM kernels
@needs_gpu
@pytest.mark.parametrize("system", list(SYSTEMS))
@pytest.mark.parametrize("method", ["d4", "d3"])
@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-11), (torch.float32, 2e-5)])
def test_atm_kernels_match_the_reference(method, system, dtype, tol):
    model = _model(method, dtype)
    ref, out = _pair(model, _graph(model, system, dtype))
    assert model.model.term.fast_active
    assert float(ref["energy_3body"].abs().max()) > 0
    for key in ("energy_3body", "energy", "node_energy", "forces"):
        assert _rel(out[key], ref[key]) < tol, key
    if system != "cluster":
        assert _rel(out["stress"], ref["stress"]) < tol
    if method == "d3":
        assert out["c6_matrix"].numel() == 0 and ref["c6_matrix"].numel() > 0


@needs_gpu
def test_atm_self_image_triangles_are_covered():
    """The small cell has self-image triangles; dropping them must show."""
    from xnn.common.models.dispersion_fast.atm import plan_triplets
    model = _model("d4")
    g = _graph(model, "small_cell")
    r = g.edge_vectors().norm(dim=-1)
    ei = g.edge_index[:, r <= OPTIONS["cutoff_triple"]]
    modes = [launch.mode for launch in plan_triplets(ei, g.num_nodes).launches]
    assert modes == [1, 2]
    g_big = _graph(model, "cell")
    r = g_big.edge_vectors().norm(dim=-1)
    ei = g_big.edge_index[:, r <= OPTIONS["cutoff_triple"]]
    assert [launch.mode for launch in plan_triplets(ei, g_big.num_nodes).launches] == [1]


@needs_gpu
@pytest.mark.parametrize("method", ["d4", "d3"])
def test_second_derivatives_through_the_kernels(method):
    """A Hessian-vector product through the fast ATM term (a create_graph backward)."""
    disp = _model(method).model                       # evaluation mode: the kernels run
    hv = {}
    for mode in (False, True):
        fast.set_use_fast(disp, mode)
        g = _graph(_model(method), "cell")
        g.pos.requires_grad_(True)
        energy = disp(g)["energy"].sum()
        assert disp.term.fast_active is mode
        forces = -torch.autograd.grad(energy, g.pos, create_graph=True)[0]
        # a random direction: the forces sum to zero, so a uniform one gives Hv = 0
        v = torch.randn(forces.shape, generator=torch.Generator().manual_seed(0), dtype=forces.dtype)
        hv[mode] = torch.autograd.grad((forces * v.to(forces.device)).sum(), g.pos)[0]
    assert hv[False].abs().max() > 1e-6
    assert _rel(hv[True], hv[False]) < 1e-10


@needs_gpu
def test_training_mode_uses_the_fast_three_body():
    """A training step differentiates the kernels' gradient (the tangent kernels)."""
    model = _model("d4").train()
    fast.set_use_fast(model, True)
    calls = []
    import xnn.common.models.dispersion_fast.atm as atm
    original = atm._ATMGrad.apply
    atm._ATMGrad.apply = lambda *a: calls.append(1) or original(*a)
    try:
        model(_graph(model, "cell"))
    finally:
        atm._ATMGrad.apply = original
    assert model.model.term.fast_active and calls


@needs_gpu
def test_wrapped_model_trains_with_the_fast_dispersion():
    from xnn.common.config import from_dict
    from xnn.common.models import build_model
    cfg = from_dict({"model": {"name": "mace", "cutoff": 4.0, "n_interactions": 1, "n_rbf": 6,
                               "n_features": 8, "extra": {"species": [1, 8], "l_max": 1,
                                                          "dispersion": {**OPTIONS, **D4_ONLY}}}})
    torch.manual_seed(0)
    model = ForceStressOutput(build_model(cfg.model), compute_stress=True).cuda().train()
    grads = {}
    for mode in (False, True):
        fast.set_use_fast(model, mode)
        model.zero_grad(set_to_none=True)
        out = model(_graph(model, "small_cell"))
        (out["energy"].square().sum() + out["forces"].square().sum()).backward()
        grads[mode] = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    assert grads[True].keys() == grads[False].keys() and grads[True]
    for n, g0 in grads[False].items():
        assert _rel(grads[True][n], g0) < 1e-9, n


# the fast EEQ operator
def _eeq_system(periodic, device, fast_cls):
    from xnn.common.models.d4 import DFTD4
    from xnn.common.models.dispersion_fast.eeq import FastEEQSystem
    from xnn.common.models.eeq import EEQSystem, ewald_alpha, reciprocal_vectors
    d4 = DFTD4(**OPTIONS, **D4_ONLY, regime="large").to(device)
    s = _water(4, periodic)
    pos = torch.tensor(s["pos"], device=device) / d4.bohr
    z = torch.tensor(s["atomic_numbers"], device=device)
    rad = d4.eeq_rad[z]
    diag = d4.eeq_eta[z] + np.sqrt(2.0 / np.pi) / rad
    if not periodic:
        args = (diag, rad, pos)
        kw = {}
    else:
        cell = torch.tensor(s["cell"], device=device) / d4.bohr
        g = structure_to_graph({**s, "pos": torch.tensor(s["pos"], device=device),
                                "cell": torch.tensor(s["cell"], device=device)},
                               d4.cutoff_eeq, device=device)
        vec = g.edge_vectors() / d4.bohr
        alpha = ewald_alpha(d4.cutoff_eeq / d4.bohr)
        grid, gvec, gfac = reciprocal_vectors(cell, alpha)
        diag = diag - 2.0 * alpha / np.sqrt(np.pi)
        args = (diag, rad, pos, g.edge_index, vec, alpha, gvec, gfac, grid)
        kw = {"cell": cell}
    ref = EEQSystem(*args)
    fst = FastEEQSystem(*args, **kw) if fast_cls else None
    return ref, fst


@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=needs_gpu)])
def test_fast_eeq_operator_matches_the_reference(periodic, device):
    ref, fst = _eeq_system(periodic, device, True)
    v = torch.randn(ref.n, 2, device=device, generator=None)
    expected = torch.stack([ref.matvec(v[:, 0]), ref.matvec(v[:, 1])], dim=1)
    assert _rel(fst.matvec(v), expected) < 1e-12
    assert _rel(fst.matvec(v[:, 0]), expected[:, 0]) < 1e-12
    y = fst.conjugate_gradient(v)
    assert _rel(fst.matvec(y), v) < 1e-10


@needs_gpu
@pytest.mark.parametrize("system", ["cell", "cluster"])
@pytest.mark.parametrize("solver", ["cg", "lu"])
def test_fast_eeq_charges_match_the_reference(system, solver):
    model = _model("d4", regime="large", eeq_solver=solver, s9=0.0)
    ref, out = _pair(model, _graph(model, system))
    assert _rel(out["eeq_charges"], ref["eeq_charges"]) < 1e-9
    for key in ("energy", "forces"):
        assert _rel(out[key], ref[key]) < 1e-9, key
    if system == "cell":
        assert _rel(out["stress"], ref["stress"]) < 1e-9


# selection, CPU behaviour, export
def test_cpu_fast_request_runs_the_reference_elsewhere():
    """On a CPU a fast request takes the C++ kernels where they build, else the compiled three-body
    tiles where torch.compile produces code, else the reference; the results are the reference's."""
    from xnn.common.models.dispersion_fast import cpu
    model = _model("d4", device="cpu")
    g = _graph(model, "small_cell", device="cpu")
    ref, out = _pair(model, g)
    assert model.model.term.fast_active == (cpu.available() or _cpu_compiles())
    for key in ("energy", "forces", "stress"):
        assert _rel(out[key], ref[key]) < 1e-12, key


@pytest.mark.parametrize("method", ["d3", "d4"])
@pytest.mark.parametrize("system", ["small_cell", "cluster"])
@pytest.mark.parametrize("width", [0.0, 1.0])
def test_cpu_three_body_matches_the_reference(method, system, width, monkeypatch):
    """The compiled CPU three-body tiles give the reference's energies, atomic energies,
    forces and stress (repeated-atom triangles of the periodic cell included), with the
    forces from autograd (the written-out forces of the direct paths turned off)."""
    if not _cpu_compiles():
        pytest.skip("torch.compile produces no CPU code here")
    model = _model(method, device="cpu", switch_width_triple=width)
    monkeypatch.setattr(type(model.model.term), "direct_applies", lambda self, data, use_fast: False)
    g = _graph(model, system, device="cpu")
    fast.set_use_fast(model, False)
    ref = model(copy.copy(g))
    fast.set_use_fast(model, True)
    out = model(copy.copy(g))
    assert model.model.term.fast_active
    keys = ("energy", "node_energy", "forces") + (("stress",) if system != "cluster" else ())
    for key in keys:
        assert _rel(out[key], ref[key]) < 1e-12, key


@pytest.fixture
def cpu_kernels(monkeypatch):
    """The optional C++ CPU kernels, turned on for the test (skipped where they do not build)."""
    from xnn.common.models import cpu_extension
    from xnn.common.models.dispersion_fast import cpu
    monkeypatch.setattr(cpu_extension, "_ENABLED", True)
    if not cpu.available():
        pytest.skip("the C++ kernels do not build here")


_DAMPINGS = {"bj": {}, "zero": {}, "mzero": {"bet": 0.1}, "op": {"bet": 6.0}}


@pytest.mark.parametrize("method", ["d3", "d4"])
@pytest.mark.parametrize("direct", [False, True])
def test_cpu_three_body_small_inputs(method, direct, monkeypatch):
    """The compiled CPU tiles on the smallest inputs give the reference's energies and forces: a dimer
    (no triangle), a linear trimer (one tile job, padded to two) and a one-water cell (every atom's own
    images), unswitched cutoffs, through autograd and through the written-out forces of the dense and the
    tile paths."""
    from xnn.common.models.dispersion_fast import d3_direct, d4_direct
    if not _cpu_compiles():
        pytest.skip("torch.compile produces no CPU code here")
    model = _model(method, device="cpu", switch_width_pair=0.0, switch_width_triple=0.0)
    if not direct:
        monkeypatch.setattr(type(model.model.term), "direct_applies", lambda self, data, use_fast: False)
    one = {k: (torch.tensor(v) if k in ("pos", "cell") else v) for k, v in _water(1, True).items()}
    structures = [{"pos": torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), "atomic_numbers": [8, 1]},
                  {"pos": torch.tensor([[0.0, 0.0, 0.0], [3.5, 0.0, 0.0], [7.0, 0.0, 0.0]]),
                   "atomic_numbers": [8, 8, 8]}, one]
    for limit in ((2048, 0) if direct else (2048,)):
        monkeypatch.setattr(d3_direct, "DENSE_ATOMS", limit)
        monkeypatch.setattr(d4_direct, "DENSE_ATOMS", limit)
        for s in structures:
            g = structure_to_graph(s, model.model.cutoff)
            ref, out = _pair(model, g)
            for key in ("energy", "forces"):
                assert _rel(out[key], ref[key]) < 1e-11, key


def test_direct_paths_step_aside(monkeypatch):
    """The written-out forces decline where they do not apply: trained parameters, a structure beyond the
    tile path's dense C6 arrays; the optional C++ kernels decline for trained parameters, too many atoms with
    a three-body term, the large EEQ regime, a cell outside the dense regime and a cell with the tail
    correction (decisions only: nothing is built)."""
    from xnn.common.models import cpu_extension
    from xnn.common.models.dispersion_fast import cpu, d3_direct, d4_direct
    if not _cpu_compiles():
        pytest.skip("torch.compile produces no CPU code here")
    monkeypatch.setattr(d3_direct, "DENSE_ATOMS", 0)
    monkeypatch.setattr(d4_direct, "DENSE_ATOMS", 0)
    for method in ("d3", "d4"):
        term = _model(method, device="cpu").model.term
        trained = _model(method, device="cpu", trainable=True).model.term
        g = _graph(_model(method, device="cpu"), "small_cell", device="cpu")
        assert term.direct_applies(g, True) and not trained.direct_applies(g, True)
        with monkeypatch.context() as m:
            m.setattr(cpu, "MAX_DENSE_ATOMS", 4)
            assert not term.direct_applies(g, True)
        with monkeypatch.context() as m:
            m.setattr(cpu_extension, "enabled", lambda: True)
            m.setattr(cpu, "available", lambda: True)
            assert term.cpu_kernels_apply(g.num_nodes) and not trained.cpu_kernels_apply(g.num_nodes)
            m.setattr(cpu, "MAX_DENSE_ATOMS", 4)
            assert not term.cpu_kernels_apply(g.num_nodes)
            m.setattr(cpu, "MAX_DENSE_ATOMS", 20000)
            if method == "d4":
                assert term._cpp_direct(g)
                term.regime = "large"
                assert not term._cpp_direct(g)
                term.regime, term.auto_large_periodic = "auto", 1
                assert not term._cpp_direct(g)
                term.auto_large_periodic, term.tail_correction = 100000, True
                assert term._periodic(g) and not term._cpp_direct(g)


@pytest.mark.parametrize("damping", list(_DAMPINGS))
@pytest.mark.parametrize("system", ["small_cell", "cluster"])
@pytest.mark.parametrize("s9", [0.0, 1.0])
def test_d3_cpu_kernels_match_the_reference(damping, system, s9, cpu_kernels):
    """D3 on the C++ kernels gives the reference's energies (total, per atom, two- and three-body), forces,
    stress, coordination numbers and features for every damping, with and without the three-body term."""
    model = _model("d3", device="cpu", damping=damping, s9=s9, **_DAMPINGS[damping])
    g = _graph(model, system, device="cpu")
    ref, out = _pair(model, g)
    assert model.model.term.fast_active and out["c6_matrix"].numel() == 0
    keys = ["energy", "node_energy", "energy_2body", "forces", "coordination_numbers", "node_features"]
    keys += ["energy_3body"] * (s9 != 0.0) + ["stress"] * (system != "cluster")
    for key in keys:
        assert _rel(out[key], ref[key]) < 1e-12, key


@pytest.mark.parametrize("system", ["small_cell", "cluster"])
def test_d3_cpu_kernels_weighted_gradients(system, cpu_kernels):
    """Gradients other than those of summed energies (the three-body energy alone, weighted atomic energies,
    the coordination numbers, the homoatomic C6) evaluate again with their weights: the reference's."""
    model = _model("d3", device="cpu")
    g = _graph(model, system, device="cpu")
    grads = {}
    for mode in (False, True):
        fast.set_use_fast(model, mode)
        gg = copy.copy(g)
        vec = gg.edge_vectors().detach().requires_grad_(True)
        gg.edge_vectors = lambda v=vec: v
        out = model.model(gg)
        assert model.model.term.fast_active == mode
        w = torch.linspace(0.3, 1.7, out["node_energy"].shape[0])
        losses = [out["energy_3body"].sum(), (out["node_energy"] * w).sum() + 0.5 * out["energy_2body"].sum(),
                  (out["coordination_numbers"] * w).sum(), (out["node_features"][:, 1] * w).sum()]
        grads[mode] = [torch.autograd.grad(loss, vec, retain_graph=True)[0] for loss in losses]
    for a, b in zip(grads[True], grads[False]):
        assert _rel(a, b) < 1e-12


def test_d3_cpu_kernels_float32_is_at_least_as_accurate(cpu_kernels):
    """float32 inputs run in float64 inside the kernels: closer to the float64 result than the float32
    reference (whose periodic three-body sums cancel to a few digits)."""
    model = _model("d3", device="cpu")
    exact = model(copy.copy(_graph(model, "small_cell", device="cpu")))
    model32 = _model("d3", dtype=torch.float32, device="cpu")
    ref, out = _pair(model32, _graph(model32, "small_cell", dtype=torch.float32, device="cpu"))
    for key in ("energy_3body", "energy", "forces"):
        err_out = _rel(out[key].double(), exact[key])
        assert err_out < 1e-5 and err_out <= max(_rel(ref[key].double(), exact[key]), 1e-7), key


def test_d3_cpu_kernels_step_aside(cpu_kernels):
    """Trained parameters and training steps keep the reference (the kernels' gradient is differentiable
    once); elements without reference data raise as the reference does."""
    model = _model("d3", device="cpu", trainable=True)
    fast.set_use_fast(model, True)
    assert not model.model.term.cpu_kernels_apply(10)
    model = _model("d3", device="cpu").train()
    fast.set_use_fast(model, True)
    out = model(_graph(model, "cluster", device="cpu"))
    assert out["c6_matrix"].numel() > 0
    model.eval()
    g = _graph(model, "cluster", device="cpu")
    g.atomic_numbers = g.atomic_numbers.clone()
    g.atomic_numbers[0] = 104
    with pytest.raises(ValueError, match="atomic numbers"):
        model(g)


@pytest.mark.parametrize("system", ["small_cell", "cell", "cluster"])
@pytest.mark.parametrize("s9", [0.0, 1.0])
def test_d4_cpu_kernels_match_the_reference(system, s9, cpu_kernels):
    """D4's two- and three-body terms on the C++ kernels give the reference's energies, forces and stress
    (a standalone cell takes the PyTorch direct path, whose LU-factored EEQ solve differs from the reference's
    solve by a few 1e-13), and the gradients of other weightings (the three-body energy alone, weighted atomic
    energies)."""
    model = _model("d4", device="cpu", s9=s9)
    g = _graph(model, system, device="cpu")
    ref, out = _pair(model, g)
    assert model.model.term.fast_active
    keys = ["energy", "node_energy", "energy_2body", "energy_3body", "forces"] + ["stress"] * (system != "cluster")
    for key in keys:
        assert _rel(out[key], ref[key]) < 1e-11, key
    grads = {}
    for mode in (False, True):
        fast.set_use_fast(model, mode)
        gg = copy.copy(g)
        vec = gg.edge_vectors().detach().requires_grad_(True)
        gg.edge_vectors = lambda v=vec: v
        o = model.model(gg)
        w = torch.linspace(0.3, 1.7, o["node_energy"].shape[0])
        losses = [o["energy_3body"].sum() + o["energy_2body"].sum() * 0.25, (o["node_energy"] * w).sum()]
        grads[mode] = [torch.autograd.grad(loss, vec, retain_graph=True)[0] for loss in losses]
    for a, b in zip(grads[True], grads[False]):
        assert _rel(a, b) < 1e-12


@pytest.mark.parametrize("method", ["d3", "d4"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_cpu_kernels_form_forces_and_stress_directly(method, dtype, monkeypatch, cpu_kernels):
    """A standalone term on the C++ kernels in evaluation forms its forces and stress without autograd through
    the evaluation (molecules, a charged one included, and a cell; D4's Ewald matrix of the cell is
    differentiated over that matrix alone); the results equal the float64 reference to rounding (float32 inputs
    run in float64 inside the kernels)."""
    from xnn.common.data import collate
    from xnn.common.models.dispersion_fast import cpu
    calls = []
    name = f"{method}_direct"
    original = getattr(cpu, name)
    monkeypatch.setattr(cpu, name, lambda *a: calls.append(1) or original(*a))
    model = _model(method, device="cpu")
    model32 = _model(method, dtype=dtype, device="cpu")
    systems = [_water(2, False, seed=1), _water(2, False, seed=2), _water(3, True)]
    graphs = {}
    for dt, m in ((torch.float64, model), (dtype, model32)):
        structures = [{k: (torch.tensor(v, dtype=dt) if k in ("pos", "cell") else v) for k, v in s.items()}
                      for s in systems]
        g = collate([structure_to_graph(s, m.model.cutoff) for s in structures])
        if method == "d4":
            g.total_charge = torch.tensor([0.0, 1.0, 0.0], dtype=dt)
        graphs[dt] = g
    fast.set_use_fast(model, False)
    ref = model(copy.copy(graphs[torch.float64]))
    fast.set_use_fast(model32, True)
    out = model32(copy.copy(graphs[dtype]))
    assert calls and "virial" not in out and set(out) == set(ref)
    tol = 1e-12 if dtype == torch.float64 else 1e-5
    keys = ["energy", "node_energy", "energy_2body", "energy_3body", "forces", "node_features"]
    keys += ["stress"] + ["eeq_charges", "dynamic_polarizabilities"] * (method == "d4")
    for key in keys:
        assert _rel(out[key].double(), ref[key]) < tol, key


@pytest.mark.parametrize("damping", list(_DAMPINGS))
@pytest.mark.parametrize("s9", [0.0, 1.0])
def test_d3_pytorch_direct_path_matches_the_reference(damping, s9, monkeypatch):
    """With the C++ kernels off, a standalone D3 term forms its forces and stress in PyTorch without an
    autograd pass (dispersion_fast.d3_direct; with a three-body term on the compiled tiles of atm_cpu with
    their analytic gradient): the reference's energies, forces, stress and features for every damping, on
    molecules, a small cell with repeated-atom triangles, a one-water cell (every atom's own images inside the
    three-body cutoff) and a molecule-plus-cell batch."""
    from xnn.common.data import collate
    from xnn.common.models.dispersion_fast import d3_direct
    if s9 != 0.0 and not _cpu_compiles():
        pytest.skip("torch.compile produces no CPU code here")
    calls = []
    original = d3_direct.d3_direct
    monkeypatch.setattr(d3_direct, "d3_direct", lambda *a: calls.append(1) or original(*a))
    model = _model("d3", device="cpu", damping=damping, s9=s9, **_DAMPINGS[damping])
    graphs = [_graph(model, "cluster", device="cpu"), _graph(model, "small_cell", device="cpu")]
    one = {k: (torch.tensor(v) if k in ("pos", "cell") else v) for k, v in _water(1, True).items()}
    graphs.append(structure_to_graph(one, model.model.cutoff))
    small = {k: (torch.tensor(v) if k in ("pos", "cell") else v) for k, v in _water(2, True).items()}
    molecule = {k: (torch.tensor(v) if k == "pos" else v) for k, v in _water(2, False, seed=5).items()}
    graphs.append(collate([structure_to_graph(molecule, model.model.cutoff),
                           structure_to_graph(small, model.model.cutoff)]))
    for g in graphs:
        ref, out = _pair(model, g)
        keys = ["energy", "node_energy", "energy_2body", "energy_3body", "forces", "node_features"]
        keys += ["stress"] * (g.cell is not None)
        for key in keys:
            assert _rel(out[key], ref[key]) < 1e-12, key
    assert len(calls) == len(graphs)


@pytest.mark.parametrize("s9", [0.0, 1.0])
@pytest.mark.parametrize("tail", [False, True])
def test_d4_pytorch_direct_path_matches_the_reference(s9, tail, monkeypatch):
    """With the C++ kernels off, a standalone D4 term forms its forces and stress in PyTorch without an
    autograd pass through the evaluation (dispersion_fast.d4_direct: the EEQ adjoint, polarizability slopes,
    compiled three-body tiles with their gradient; for cells the Ewald matrix's derivative): the reference's
    energies, forces, stress, charges and polarizabilities, for a cluster, a small cell, a one-water cell
    (every atom's own images inside the three-body cutoff) and a batch of a charged molecule with a cell, with
    and without the tail correction."""
    from xnn.common.data import collate
    from xnn.common.models.dispersion_fast import d4_direct
    if s9 != 0.0 and not _cpu_compiles():
        pytest.skip("torch.compile produces no CPU code here")
    calls = []
    original = d4_direct.d4_direct
    monkeypatch.setattr(d4_direct, "d4_direct", lambda *a: calls.append(1) or original(*a))
    model = _model("d4", device="cpu", s9=s9, tail_correction=tail)
    molecule = {k: (torch.tensor(v) if k == "pos" else v) for k, v in _water(2, False, seed=1).items()}
    cell = {k: (torch.tensor(v) if k in ("pos", "cell") else v) for k, v in _water(2, True).items()}
    batch = collate([structure_to_graph(molecule, model.model.cutoff), structure_to_graph(cell, model.model.cutoff)])
    batch.total_charge = torch.tensor([1.0, 0.0])
    one = {k: (torch.tensor(v) if k in ("pos", "cell") else v) for k, v in _water(1, True).items()}
    graphs = [_graph(model, "cluster", device="cpu"), _graph(model, "small_cell", device="cpu"),
              structure_to_graph(one, model.model.cutoff), batch]
    for g in graphs:
        ref, out = _pair(model, g)
        keys = ["energy", "node_energy", "energy_2body", "energy_3body", "energy_tail", "forces", "eeq_charges",
                "dynamic_polarizabilities", "node_features"] + ["stress"] * (g.cell is not None)
        for key in keys:
            assert _rel(out[key], ref[key]) < 1e-10, key
    assert len(calls) == len(graphs)


@pytest.mark.parametrize("solver", ["lu", "cg"])
def test_d4_pytorch_direct_path_large_eeq_regime(solver, monkeypatch):
    """The direct path in the large EEQ regime (matrix-free system, its adjoint solve, the derivative of its
    differentiable application in the positions, the EEQ-range edges and the cell) gives the reference's
    energies, forces, stress and charges, for a cluster and a cell."""
    from xnn.common.models.dispersion_fast import d4_direct
    if not _cpu_compiles():
        pytest.skip("torch.compile produces no CPU code here")
    calls = []
    original = d4_direct.d4_direct
    monkeypatch.setattr(d4_direct, "d4_direct", lambda *a: calls.append(1) or original(*a))
    model = _model("d4", device="cpu", regime="large", cutoff_eeq=11.0, eeq_solver=solver)
    tol = 1e-12 if solver == "lu" else 1e-10
    for system in ("cluster", "cell"):
        g = _graph(model, system, device="cpu")
        ref, out = _pair(model, g)
        for key in ["energy", "node_energy", "energy_3body", "forces", "eeq_charges"] + ["stress"] * (system == "cell"):
            assert _rel(out[key], ref[key]) < tol, key
    assert len(calls) == 2


def _molecule_batch(model, charge=None):
    from xnn.common.data import collate
    first = {k: (torch.tensor(v) if k == "pos" else v) for k, v in _water(2, False, seed=5).items()}
    second = {k: (torch.tensor(v) if k == "pos" else v) for k, v in _water(1, False, seed=7).items()}
    batch = collate([structure_to_graph(first, model.model.cutoff), structure_to_graph(second, model.model.cutoff)])
    if charge is not None:
        batch.total_charge = torch.tensor(charge)
    return batch


@pytest.mark.parametrize("damping", list(_DAMPINGS))
@pytest.mark.parametrize("s9", [0.0, 1.0])
def test_d3_dense_path_matches_the_reference(damping, s9, monkeypatch):
    """Molecules up to DENSE_ATOMS take the dense path (every pair and triangle in one compiled call, the pairs
    outside the graph's cutoffs left out): the reference's energies, forces, stress and features for every
    damping, on a cluster and a batch of two molecules."""
    from xnn.common.models.dispersion_fast import d3_direct
    if not _cpu_compiles():
        pytest.skip("torch.compile produces no CPU code here")
    calls = []
    original = d3_direct.d3_dense
    monkeypatch.setattr(d3_direct, "d3_dense", lambda *a: calls.append(1) or original(*a))
    monkeypatch.setattr(d3_direct, "DENSE_ATOMS", 1000)
    model = _model("d3", device="cpu", damping=damping, s9=s9, **_DAMPINGS[damping])
    graphs = [_graph(model, "cluster", device="cpu"), _molecule_batch(model)]
    for g in graphs:
        ref, out = _pair(model, g)
        for key in ["energy", "node_energy", "energy_2body", "energy_3body", "forces", "node_features"]:
            assert _rel(out[key], ref[key]) < 1e-12, key
    assert len(calls) == len(graphs)


@pytest.mark.parametrize("s9", [0.0, 1.0])
def test_d4_dense_path_matches_the_reference(s9, monkeypatch):
    """Molecules up to DENSE_ATOMS take the dense path (one EEQ system for the batch with a charge constraint per
    molecule, its adjoint, every pair and triangle in one compiled call): the reference's energies, forces,
    charges and polarizabilities, on a cluster and a batch of a charged and a neutral molecule."""
    from xnn.common.models.dispersion_fast import d4_direct
    if not _cpu_compiles():
        pytest.skip("torch.compile produces no CPU code here")
    calls = []
    original = d4_direct.d4_dense
    monkeypatch.setattr(d4_direct, "d4_dense", lambda *a: calls.append(1) or original(*a))
    monkeypatch.setattr(d4_direct, "DENSE_ATOMS", 1000)
    model = _model("d4", device="cpu", s9=s9)
    graphs = [_graph(model, "cluster", device="cpu"), _molecule_batch(model, [1.0, 0.0])]
    for g in graphs:
        ref, out = _pair(model, g)
        for key in ["energy", "node_energy", "energy_2body", "energy_3body", "forces", "eeq_charges",
                    "dynamic_polarizabilities", "node_features"]:
            assert _rel(out[key], ref[key]) < 1e-10, key
    assert len(calls) == len(graphs)


def test_set_use_fast_reaches_the_dispersion_wrapper():
    model = _model("d3", device="cpu")
    assert model.model.use_fast == "auto"
    fast.set_use_fast(model, False)
    assert model.model.use_fast is False
    assert model.model.term in fast.fast_modules(model)


def test_torchscript_export_after_a_fast_request(tmp_path):
    from xnn.common.deploy import export_torchscript_potential
    model = _model("d4", device="cpu")
    fast.set_use_fast(model, True)
    model(_graph(model, "small_cell", device="cpu"))
    path = export_torchscript_potential(model.model, model.model.cutoff, str(tmp_path / "d4.pt"))
    assert torch.jit.load(path) is not None


@needs_gpu
def test_eeq_reuse_with_the_fast_system_forms_no_dense_matrix(monkeypatch):
    """Above DENSE_REUSE_MAX_ATOMS the reuse preconditions with the fast system's
    own low-rank preconditioner: no (N, N) matrix, charges as a fresh solve."""
    import xnn.common.models.eeq as eeq

    rng = np.random.default_rng(5)
    s = _water(4, True)
    frames = []
    for _ in range(4):
        s = {**s, "pos": s["pos"] + rng.normal(scale=0.01, size=s["pos"].shape)}
        frames.append(s)
    fresh = _model("d4", regime="large")
    fast.set_use_fast(fresh, True)
    graphs = [structure_to_graph({**f, "pos": torch.tensor(f["pos"]), "cell": torch.tensor(f["cell"])},
                                 fresh.model.cutoff, device="cuda") for f in frames]
    expected = [fresh(copy.copy(g))["eeq_charges"].detach() for g in graphs]

    monkeypatch.setattr(eeq, "DENSE_REUSE_MAX_ATOMS", 0)

    def no_dense(self):
        raise AssertionError("the reuse assembled the dense EEQ matrix")

    monkeypatch.setattr(eeq.EEQSystem, "assemble", no_dense)
    reused = _model("d4", regime="large")
    fast.set_use_fast(reused, True)
    reused.model.d4.enable_eeq_reuse()
    for g, q in zip(graphs, expected):
        out = reused(copy.copy(g))
        # two iterative solves, each to a 1e-9 relative residual
        assert float((out["eeq_charges"] - q).abs().max()) < 5e-8
    stats = reused.model.d4.__dict__["_eeq_reuse"].stats
    assert stats["preconditioners"] == 0 and stats["fallbacks"] == 0
