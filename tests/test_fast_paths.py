"""Tests for the fast paths (``use_fast``) and the reference they must reproduce.

CPU tests cover the selection logic, the float32 warning, the fallback to the
reference off a GPU, the unchanged parameters and ``state_dict``, TorchScript
export, and the matrix-product evaluation of the symmetric contraction against
its defining einsum. The GPU tests (skipped without CUDA and cuEquivariance)
compare every fast block and whole models with the reference.
"""
import copy
import warnings

import numpy as np
import pytest
import torch

pytest.importorskip("e3nn")

from e3nn import o3  # noqa: E402

from xnn.common.config import from_dict  # noqa: E402
from xnn.common.data import structure_to_graph  # noqa: E402
from xnn.common.models import ForceStressOutput, build_model  # noqa: E402
from xnn.common.models import fast  # noqa: E402
from xnn.gnn.fast import AUTO_POLICY, ConvTensorProduct, available  # noqa: E402
from xnn.gnn.models.blocks import tp_out_irreps_with_instructions  # noqa: E402
from xnn.gnn.models.mace import SymmetricContraction  # noqa: E402

SPECIES = [1, 6, 8]
GPU = torch.cuda.is_available() and available()
needs_gpu = pytest.mark.skipif(not GPU, reason="needs CUDA and cuEquivariance")


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _graph(n=12, cutoff=4.0, periodic=True, device="cpu", dtype=torch.float64):
    rng = np.random.default_rng(1)
    s = {"pos": torch.tensor(rng.uniform(0, 6, (n, 3)), dtype=dtype),
         "atomic_numbers": torch.tensor(([1, 6, 8] * n)[:n])}
    if periodic:
        s["cell"] = torch.eye(3, dtype=dtype) * 6.0
        s["pbc"] = torch.ones(3, dtype=torch.bool)
    return structure_to_graph(s, cutoff, device=device)


def _mace(**extra):
    cfg = from_dict({"model": {
        "name": "mace", "cutoff": 4.0, "n_features": 16, "n_interactions": 2,
        "extra": {"species": SPECIES, "max_ell": 3, "correlation": 3, "num_channels": 16,
                  "max_L": 1, **extra}}})
    torch.manual_seed(0)
    return build_model(cfg.model)


def _nequip():
    cfg = from_dict({"model": {"name": "nequip", "cutoff": 4.0, "n_features": 16,
                               "n_interactions": 2, "l_max": 2, "species": SPECIES,
                               "avg_num_neighbors": 10.0}})
    torch.manual_seed(0)
    return build_model(cfg.model)


# selection
def test_resolve_use_fast():
    assert fast.resolve_use_fast("auto") == "auto"
    assert fast.resolve_use_fast("AUTO") == "auto"
    assert fast.resolve_use_fast(True) is True and fast.resolve_use_fast(False) is False
    with pytest.raises(ValueError):
        fast.resolve_use_fast("fast")
    with pytest.raises(TypeError):
        fast.resolve_use_fast(1)


def test_auto_policy_thresholds(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "NVIDIA A100-SXM4-80GB")
    cuda = torch.device("cuda")
    assert AUTO_POLICY.threshold(cuda, torch.float32) == 50_000
    assert AUTO_POLICY.threshold(cuda, torch.float64) == 25_000
    assert not AUTO_POLICY.decide(cuda, torch.float32, 49_999)
    assert AUTO_POLICY.decide(cuda, torch.float32, 50_000)
    assert not AUTO_POLICY.decide(torch.device("cpu"), torch.float32, 10**9)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "Some Future GPU")
    assert AUTO_POLICY.threshold(cuda, torch.float32) == AUTO_POLICY.default_min_edges


class _Dummy(torch.nn.Module, fast.FastPathModule):
    def __init__(self):
        super().__init__()
        self.fast_active = False

    def fast_supported(self, device, dtype):
        return True


def test_select_and_float32_warning(monkeypatch):
    monkeypatch.setattr(fast, "_warned_float32", False)
    m = _Dummy()
    cpu = torch.device("cpu")
    assert not fast.select([m], False, None, cpu, torch.float32, 10)
    assert not m.fast_active
    with pytest.warns(fast.FastPathPrecisionWarning, match="1e-6"):
        assert fast.select([m], True, None, cpu, torch.float32, 10)
    assert m.fast_active
    with warnings.catch_warnings():
        warnings.simplefilter("error")             # once per process
        fast.select([m], True, None, cpu, torch.float32, 10)
        fast.select([m], True, None, cpu, torch.float64, 10)
    # auto: the policy decides (a CPU device never takes the fast path)
    assert not fast.select([m], "auto", AUTO_POLICY, cpu, torch.float64, 10**9)


def test_set_use_fast_reaches_wrapped_models():
    model = ForceStressOutput(_mace())
    assert model.model.use_fast == "auto"
    fast.set_use_fast(model, False)
    assert model.model.use_fast is False
    with pytest.raises(ValueError):
        fast.set_use_fast(model, "sometimes")


# the reference stays the reference
@pytest.mark.parametrize("build", [_mace, _nequip])
def test_cpu_fast_request_runs_reference(build):
    core = build()
    model = ForceStressOutput(core, compute_stress=True)
    g = _graph()
    fast.set_use_fast(model, False)
    ref = model(copy.copy(g))
    fast.set_use_fast(model, True)
    out = model(copy.copy(g))
    assert not any(m.fast_active for m in fast.fast_modules(core))
    # the same code path; only the threaded scatter's summation order can differ
    for key in ("energy", "forces", "stress"):
        assert torch.allclose(ref[key], out[key], rtol=0, atol=1e-12)


def test_fast_blocks_keep_the_reference_state_dict():
    irreps_in, irreps_sh = o3.Irreps("8x0e+8x1o"), o3.Irreps.spherical_harmonics(2)
    mid, ins = tp_out_irreps_with_instructions(irreps_in, irreps_sh, o3.Irreps("8x0e+8x1o+8x2e"))
    kw = dict(shared_weights=False, internal_weights=False)
    a = ConvTensorProduct(irreps_in, irreps_sh, mid, ins, **kw)
    b = o3.TensorProduct(irreps_in, irreps_sh, mid, ins, **kw)
    assert a.state_dict().keys() == b.state_dict().keys()
    assert a.fast_eligible and a.weight_numel == b.weight_numel
    x, y = torch.randn(5, irreps_in.dim), torch.randn(5, irreps_sh.dim)
    w = torch.randn(5, a.weight_numel)
    assert torch.allclose(a(x, y, w), b(x, y, w), rtol=0, atol=1e-14)
    # MACE's parameter names are those of a model without fast paths
    names = set(_mace().state_dict())
    assert not any("_fast" in n for n in names)


def test_conv_matches_gather_tp_scatter():
    irreps_in, irreps_sh = o3.Irreps("4x0e+4x1o"), o3.Irreps.spherical_harmonics(2)
    mid, ins = tp_out_irreps_with_instructions(irreps_in, irreps_sh, o3.Irreps("4x0e+4x1o"))
    tp = ConvTensorProduct(irreps_in, irreps_sh, mid, ins, shared_weights=False, internal_weights=False)
    x = torch.randn(6, irreps_in.dim)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 0], [1, 2, 3, 4, 5, 0, 3]])
    y = torch.randn(7, irreps_sh.dim)
    w = torch.randn(7, tp.weight_numel)
    expected = torch.zeros(6, mid.dim).index_add_(0, edge_index[1], tp(x[edge_index[0]], y, w))
    assert torch.allclose(tp.conv(x, y, w, edge_index, 6), expected, atol=1e-13)
    assert torch.allclose(tp.conv(x, y, w, edge_index, 6, 2.0), expected / 2.0, atol=1e-13)


def _einsum_contraction(c, x, y):
    """The defining nested einsum of one MACE contraction (the previous implementation)."""
    axes = "mnopqrstuvwx"
    m_axis = 1 if c.lmax_out > 0 else 0
    corr = c.correlation
    lead = axes[: corr + m_axis - 1]
    acc = torch.einsum(f"{lead}ik,ekc,bci,be->bc{lead}", c._U(corr), c.weights[corr - 1], x, y)
    for order in range(corr - 1, 0, -1):
        aw, af = axes[: order + m_axis], axes[: order + m_axis - 1]
        weighted = torch.einsum(f"{aw}k,ekc,be->bc{aw}", c._U(order), c.weights[order - 1], y)
        acc = torch.einsum(f"bc{af}i,bci->bc{af}", weighted + acc, x)
    return acc.reshape(acc.shape[0], -1)


@pytest.mark.parametrize("correlation", [1, 2, 3, 4])
def test_symmetric_contraction_matches_its_einsum(correlation):
    irreps_in = o3.Irreps("6x0e+6x1o+6x2e")
    sc = SymmetricContraction(irreps_in, o3.Irreps("6x0e+6x1o"), correlation, num_elements=3)
    torch.manual_seed(1)
    x = torch.randn(9, 6, 9)
    y = torch.nn.functional.one_hot(torch.tensor([0, 1, 2, 2, 1, 0, 0, 1, 2]), 3).double()
    expected = torch.cat([_einsum_contraction(c, x, y) for c in sc.contractions], dim=-1)
    assert torch.allclose(sc(x, y), expected, rtol=1e-12, atol=1e-12)


def test_torchscript_export_after_fast_request(tmp_path):
    from xnn.common.deploy import export_torchscript_potential
    core = _mace()
    fast.set_use_fast(core, True)
    ForceStressOutput(core)(_graph())
    path = export_torchscript_potential(core, 4.0, str(tmp_path / "m.pt"))
    assert torch.jit.load(path) is not None


# GPU: every fast block and whole models against the reference
def _gpu_pair(build, dtype, built_in=torch.float64):
    old = torch.get_default_dtype()
    torch.set_default_dtype(built_in)
    try:
        core = build()
    finally:
        torch.set_default_dtype(old)
    core = core.to(device="cuda", dtype=dtype)
    return ForceStressOutput(core, compute_stress=True).eval(), core


@needs_gpu
@pytest.mark.parametrize("build", [_mace, _nequip])
# float64: summation order only (the random-weight MACE has energies near 1e5, so
# its relative differences reach ~1e-13); a model built in float32 and cast
# meets the same bound, its coupling constants made exact on the cast
@pytest.mark.parametrize("dtype,built_in,tol", [
    (torch.float64, torch.float64, 1e-12),
    (torch.float64, torch.float32, 1e-12),
    (torch.float32, torch.float32, 5e-5)])
def test_gpu_model_parity(build, dtype, built_in, tol):
    model, core = _gpu_pair(build, dtype, built_in)
    g = _graph(n=48, device="cuda", dtype=dtype)
    fast.set_use_fast(model, False)
    ref = model(copy.copy(g))
    fast.set_use_fast(model, True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", fast.FastPathPrecisionWarning)
        out = model(copy.copy(g))
    assert all(m.fast_active for m in fast.fast_modules(core))
    for key in ("energy", "forces", "stress"):
        scale = ref[key].abs().max().clamp(min=1.0)
        assert float((out[key] - ref[key]).abs().max() / scale) < tol, key


@needs_gpu
def test_gpu_training_gradients():
    model, core = _gpu_pair(_mace, torch.float64)
    model.train()
    g = _graph(n=24, device="cuda")
    grads = {}
    for mode in (False, True):
        fast.set_use_fast(model, mode)
        model.zero_grad(set_to_none=True)
        out = model(copy.copy(g))
        (out["energy"].square().sum() + out["forces"].square().sum()).backward()
        grads[mode] = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    assert grads[False].keys() == grads[True].keys()
    for n, g0 in grads[False].items():
        err = (grads[True][n] - g0).abs().max() / g0.abs().max().clamp(min=1e-30)
        assert float(err) < 1e-10, n
