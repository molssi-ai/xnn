"""The dispersion fast paths (``xnn.common.models.dispersion_fast``) against the reference.

The ATM kernels and the fast EEQ operator compute the same functions as the
reference code with another summation order, so energies, forces, stress and
charges must agree to rounding (float64) and to the solver tolerance (the
iterative EEQ solve). Systems: a small water cell (narrower than twice the
three-body cutoff, so it has self-image triangles), a larger cell (none) and a
water cluster. GPU tests skip without CUDA; the operator tests also run on the CPU.
"""
import copy
import warnings

import numpy as np
import pytest
import torch

from xnn.common.data import structure_to_graph
from xnn.common.models import D3Dispersion, D4Dispersion, ForceStressOutput
from xnn.common.models import fast
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
    # evaluation mode: create_graph=False, so the kernels' own backward runs
    # (training mode asks for second derivatives, which use the reference)
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
def test_second_derivatives_use_the_reference(method):
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
def test_training_mode_uses_the_reference_three_body():
    model = _model("d4").train()
    fast.set_use_fast(model, True)
    calls = []
    import xnn.common.models.dispersion_fast.atm as atm
    original = atm._ATMEnergy.apply
    atm._ATMEnergy.apply = lambda *a: calls.append(1) or original(*a)
    try:
        model(_graph(model, "cell"))
    finally:
        atm._ATMEnergy.apply = original
    assert model.model.term.fast_active and not calls


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
    from xnn.common.models.eeq import EEQSystem, ewald_alpha, reciprocal_vectors
    from xnn.common.models.dispersion_fast.eeq import FastEEQSystem
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
def test_cpu_fast_request_runs_the_reference():
    model = _model("d4", device="cpu")
    g = _graph(model, "small_cell", device="cpu")
    ref, out = _pair(model, g)
    assert not model.model.term.fast_active
    # the same code path; only a threaded scatter's summation order can differ
    for key in ("energy", "forces", "stress"):
        assert torch.allclose(out[key], ref[key], rtol=0, atol=1e-12)


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
