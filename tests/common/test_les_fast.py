"""The LES fast path: the factorized reciprocal sum against the reference one.

``EwaldSummation.reciprocal_fast`` sums the same wave vectors with the same
weights as ``reciprocal``; only the structure factors are formed another way
(factorized phases, :mod:`xnn.common.models.reciprocal`). The energy and its
derivatives in positions, latent charges and cell (stress) must agree to
rounding, also through the recomputed column blocks and at second order
(force training). The method itself runs on any device, so the parity tests
run on the CPU; the model-level tests need CUDA.
"""
import copy
import warnings

import numpy as np
import pytest
import torch

import xnn.common.models.les as les
from xnn.common.config import from_dict
from xnn.common.data import collate, structure_to_graph
from xnn.common.models import ForceStressOutput, build_model, fast
from xnn.common.models.les import EwaldSummation

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _box(n=40, triclinic=True, seed=0):
    rng = np.random.default_rng(seed)
    cell = np.diag([9.0, 10.0, 11.0])
    if triclinic:
        cell = cell + np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [-0.7, 2.0, 0.0]])
    frac = rng.uniform(size=(n, 3))
    # some atoms outside the cell, as in an unwrapped trajectory
    pos = (frac + rng.integers(-2, 3, size=(n, 3)) * (rng.uniform(size=(n, 1)) < 0.3)) @ cell
    q = rng.normal(size=(n, 3))
    return (torch.tensor(pos, requires_grad=True), torch.tensor(q, requires_grad=True),
            torch.tensor(cell, requires_grad=True))


def _grads(fn, pos, q, cell):
    energy = fn(pos, q, cell)
    return (energy, *torch.autograd.grad(energy, (pos, q, cell)))


@pytest.mark.parametrize("exponent", [1, 6])
@pytest.mark.parametrize("remove_self", [False, True])
@pytest.mark.parametrize("triclinic", [False, True])
def test_reciprocal_fast_matches_the_reference(exponent, remove_self, triclinic):
    ew = EwaldSummation(dl=1.5, sigma=1.0, exponent=exponent, remove_self_interaction=remove_self)
    pos, q, cell = _box(triclinic=triclinic)
    ref = _grads(ew.reciprocal, pos, q, cell)
    out = _grads(ew.reciprocal_fast, pos, q, cell)
    for a, b in zip(out, ref):
        assert torch.allclose(a, b, rtol=1e-11, atol=1e-11 * float(b.abs().max()))


def test_reciprocal_fast_in_column_blocks(monkeypatch):
    """Several recomputed blocks: values, first and second derivatives."""
    ew = EwaldSummation(dl=1.0, sigma=1.0)
    pos, q, cell = _box()
    results = {}
    for blocks in ("one", "many"):
        if blocks == "many":
            monkeypatch.setattr(les, "FAST_BLOCK_ENTRIES", 5 * pos.shape[0])
        energy = ew.reciprocal_fast(pos, q, cell)
        g_pos, = torch.autograd.grad(energy, pos, create_graph=True)
        # second order: d/dq of |F|^2 (what a force loss differentiates)
        hq, = torch.autograd.grad(g_pos.square().sum(), q)
        results[blocks] = (energy.detach(), g_pos.detach(), hq)
    ref_e = ew.reciprocal(pos, q, cell)
    ref_g, = torch.autograd.grad(ref_e, pos, create_graph=True)
    ref_h, = torch.autograd.grad(ref_g.square().sum(), q)
    for e, g, h in results.values():
        assert torch.allclose(e, ref_e.detach(), rtol=1e-11)
        assert torch.allclose(g, ref_g.detach(), rtol=1e-10, atol=1e-11 * float(ref_g.abs().max()))
        assert torch.allclose(h, ref_h, rtol=1e-10, atol=1e-11 * float(ref_h.abs().max()))


def test_reciprocal_fast_float32_is_at_least_as_accurate():
    ew = EwaldSummation(dl=1.5, sigma=1.0)
    pos, q, cell = _box(n=200)
    exact = ew.reciprocal(pos, q, cell).detach()
    p32, q32, c32 = (t.detach().float() for t in (pos, q, cell))
    err_ref = abs(float(ew.reciprocal(p32, q32, c32)) - float(exact))
    err_fast = abs(float(ew.reciprocal_fast(p32, q32, c32)) - float(exact))
    assert err_fast <= 2.0 * err_ref + 1e-6 * abs(float(exact))


# models
def _les_model(dtype=torch.float64, device="cuda"):
    cfg = from_dict({"model": {"name": "mace", "cutoff": 4.0, "n_interactions": 2, "n_rbf": 6,
                               "n_features": 8, "extra": {"species": [1, 8], "l_max": 2,
                                                          "long_range": {"n_channels": 4,
                                                                         "sigma": 1.0, "dl": 2.0}}}})
    torch.manual_seed(0)
    model = ForceStressOutput(build_model(cfg.model), compute_stress=True)
    return model.to(device=device, dtype=dtype)


def _water_graph(n_side, periodic, dtype, device="cuda", seed=3):
    rng = np.random.default_rng(seed)
    unit = np.array([[0.0, 0.0, 0.119262], [0.0, 0.763239, -0.477047], [0.0, -0.763239, -0.477047]])
    pos = []
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                rot, _ = np.linalg.qr(rng.normal(size=(3, 3)))
                pos.extend((unit - unit[0]) @ rot.T + np.array([i, j, k]) * 3.104)
    s = {"pos": torch.tensor(np.array(pos), dtype=dtype), "atomic_numbers": [8, 1, 1] * n_side ** 3}
    if periodic:
        s["cell"] = torch.eye(3, dtype=dtype) * 3.104 * n_side
    return structure_to_graph(s, 4.0, device=device)


def _pair(model, graph):
    fast.set_use_fast(model, False)
    ref = model(copy.copy(graph))
    fast.set_use_fast(model, True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", fast.FastPathPrecisionWarning)
        out = model(copy.copy(graph))
    return ref, out


def _rel(a, b):
    return float((a.detach() - b.detach()).abs().max() / b.detach().abs().max().clamp(min=1e-30))


@needs_gpu
@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-10), (torch.float32, 1e-4)])
def test_les_model_matches_the_reference(dtype, tol):
    model = _les_model(dtype).eval()
    ref, out = _pair(model, _water_graph(6, True, dtype))
    assert model.model.ewald.fast_active
    assert float(ref["energy_lr"].abs().max()) > 0
    for key in ("energy_lr", "energy", "forces", "stress"):
        assert _rel(out[key], ref[key]) < tol, key


@needs_gpu
def test_les_mixed_batch_and_training_gradients():
    model = _les_model().train()
    batch = collate([_water_graph(4, True, torch.float64), _water_graph(2, False, torch.float64),
                     _water_graph(3, True, torch.float64)])
    grads = {}
    for mode in (False, True):
        fast.set_use_fast(model, mode)
        model.zero_grad(set_to_none=True)
        out = model(copy.copy(batch))
        (out["energy"].square().sum() + out["forces"].square().sum()
         + out["stress"].square().sum()).backward()
        grads[mode] = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    assert grads[True].keys() == grads[False].keys()
    for n, g0 in grads[False].items():
        assert _rel(grads[True][n], g0) < 1e-9, n


@needs_gpu
def test_auto_decides_per_structure():
    model = _les_model().eval()
    fast.set_use_fast(model, "auto")
    calls = []
    original = EwaldSummation.reciprocal_fast
    EwaldSummation.reciprocal_fast = lambda self, *a: calls.append(a[0].shape[0]) or original(self, *a)
    try:
        model(collate([_water_graph(4, True, torch.float64), _water_graph(10, True, torch.float64)]))
    finally:
        EwaldSummation.reciprocal_fast = original
    assert model.model.ewald.fast_min_atoms == les.AUTO_POLICY.threshold(torch.device("cuda"))
    assert calls == [3000]                      # the 3000-atom cell, not the 192-atom one


def test_set_use_fast_reaches_les():
    model = _les_model(device="cpu")
    assert model.model.use_fast == "auto"
    fast.set_use_fast(model, False)
    assert model.model.use_fast is False and model.model.model.use_fast is False
    fast.set_use_fast(model, True)
    model(_water_graph(2, True, torch.float64, device="cpu"))
    assert not model.model.ewald.fast_active                # CPU: the reference


def test_torchscript_export_after_a_fast_request(tmp_path):
    from xnn.common.deploy import export_torchscript_potential
    model = _les_model(device="cpu")
    fast.set_use_fast(model, True)
    model(_water_graph(2, True, torch.float64, device="cpu"))
    path = export_torchscript_potential(model.model, 4.0, str(tmp_path / "les.pt"))
    assert torch.jit.load(path) is not None
