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
from xnn.common.models import (  # noqa: E402
    ForceStressOutput,
    build_model,
    fast,  # noqa: E402
)
from xnn.gnn.fast import (  # noqa: E402
    AUTO_POLICY,
    ConvTensorProduct,
    EquivariantLinear,
    SpeciesTensorProduct,
    available,
)
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
    assert AUTO_POLICY.threshold(cuda, torch.float32) == 15_000
    assert AUTO_POLICY.threshold(cuda, torch.float64) == 10_000
    assert not AUTO_POLICY.decide(cuda, torch.float32, 14_999)
    assert AUTO_POLICY.decide(cuda, torch.float32, 15_000)
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
    # no kernel runs on the CPU; the linear maps' fast path is plain PyTorch and runs anywhere
    plain = (EquivariantLinear, SpeciesTensorProduct)
    assert not any(m.fast_active for m in fast.fast_modules(core) if not isinstance(m, plain))
    # the same numbers; only the summation order can differ
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


class _StandInKernel:
    """The parts of a fused convolution kernel, made of the reference: a layout change of the node features
    and of the node sums (column permutations) around the gathered, multiplied and summed messages."""

    weight = None

    def __init__(self, tp, dim_in, dim_out):
        self.tp = tp
        self.perm_in = torch.randperm(dim_in, generator=torch.Generator().manual_seed(5))
        self.perm_out = torch.randperm(dim_out, generator=torch.Generator().manual_seed(6))

    def transpose_in1(self, x):
        return x[:, self.perm_in]

    def transpose_in2(self, a):
        return a

    def transpose_out(self, y):
        return y[:, torch.argsort(self.perm_out)]

    def f(self, inputs, input_indices, output_shapes, output_indices):
        w, x, a = inputs
        if 0 in input_indices:
            w = w[input_indices[0]]
        messages = self.tp(x[:, torch.argsort(self.perm_in)][input_indices[1]], a, w)
        sums = torch.zeros(output_shapes[0].shape[0], messages.shape[1], dtype=messages.dtype)
        return [sums.index_add(0, output_indices[0], messages)[:, self.perm_out]]


def test_edge_pairs_finds_the_two_directions_of_every_bond():
    """Every edge of a full periodic list is paired with its reverse (sender and receiver swapped, the
    vector negated); a list missing a reverse is not paired."""
    from xnn.gnn.fast.conv import edge_pairs
    rng = np.random.default_rng(4)
    s = {"pos": rng.uniform(0, 4.0, (5, 3)), "atomic_numbers": [1, 6, 8, 1, 6],
         "cell": np.eye(3) * 4.0, "pbc": [True, True, True]}
    g = structure_to_graph(s, 3.5)
    vec = g.edge_vectors().detach()
    rows, weight_index = edge_pairs(g.edge_index, vec)
    E = g.edge_index.shape[1]
    assert rows.shape == (E // 2,) and torch.equal(torch.bincount(weight_index), torch.full((E // 2,), 2))
    assert torch.equal(weight_index[rows], torch.arange(E // 2))
    partner = torch.empty(E, dtype=torch.long)
    for b in range(E // 2):
        e1, e2 = (weight_index == b).nonzero().flatten().tolist()
        partner[e1], partner[e2] = e2, e1
    assert torch.equal(g.edge_index[0], g.edge_index[1, partner])
    assert torch.equal(vec, -vec[partner])
    keep = torch.ones(E, dtype=torch.bool)
    keep[[0, 1]] = False
    assert edge_pairs(g.edge_index[:, keep], vec[keep]) is None
    assert edge_pairs(g.edge_index[:, 1:], vec[1:]) is None


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("split", [False, True])
def test_conv_blocks_equal_one_conv(monkeypatch, recompute, split):
    """In blocks of edges, the weights made per block (kept, or made again in the backward), the message
    pass gives conv's node sums and gradients over all edges, by the reference or by a kernel whose layout
    changes once around the blocks."""
    irreps_in, irreps_sh = o3.Irreps("4x0e+4x1o"), o3.Irreps.spherical_harmonics(2)
    mid, ins = tp_out_irreps_with_instructions(irreps_in, irreps_sh, o3.Irreps("4x0e+4x1o"))
    tp = ConvTensorProduct(irreps_in, irreps_sh, mid, ins, shared_weights=False,
                           internal_weights=False).double()
    gen = torch.Generator().manual_seed(3)
    f64 = dict(dtype=torch.float64)
    x = torch.randn(6, irreps_in.dim, generator=gen, **f64).requires_grad_()
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 0], [1, 2, 3, 4, 5, 0, 3]])
    y = torch.randn(7, irreps_sh.dim, generator=gen, **f64).requires_grad_()
    feats = torch.randn(7, 3, generator=gen, **f64).requires_grad_()
    project = torch.randn(3, tp.weight_numel, generator=gen, **f64)

    def weights(e):
        return torch.tanh(e @ project)

    ref = tp.conv(x, y, weights(feats), edge_index, 6)
    # three weight rows read by the seven edges through an index
    rows = torch.tensor([2, 0, 1, 1, 0, 2, 0])
    assert torch.allclose(tp.conv(x, y, weights(feats[:3]), edge_index, 6, weight_index=rows),
                          tp.conv(x, y, weights(feats[:3])[rows], edge_index, 6), rtol=0, atol=1e-13)
    if split:
        kernel = {"op": _StandInKernel(tp, irreps_in.dim, mid.dim), "weight_index": torch.arange(tp.weight_numel),
                  "weight_scale": torch.ones(tp.weight_numel, **f64), "out_index": torch.arange(mid.dim)}
        monkeypatch.setattr(tp, "_split_kernel", lambda like: kernel)
        tp.fast_active = True
    out = tp.conv_blocks(x, y, feats, edge_index, 6, 3, weights, recompute=recompute)
    assert torch.allclose(out, ref, rtol=0, atol=1e-13)
    g_ref = torch.autograd.grad((ref ** 2).sum(), (x, y, feats))
    g_out = torch.autograd.grad((out ** 2).sum(), (x, y, feats))
    for a, b in zip(g_out, g_ref):
        assert torch.allclose(a, b, rtol=0, atol=1e-12)


def test_split_kernel_needs_a_kernel_with_its_parts(monkeypatch):
    """Blocks change the kernel's layout once only when the kernel builds and its parts can be called
    apart; otherwise every block calls conv."""
    irreps_in, irreps_sh = o3.Irreps("4x0e+4x1o"), o3.Irreps.spherical_harmonics(2)
    mid, ins = tp_out_irreps_with_instructions(irreps_in, irreps_sh, o3.Irreps("4x0e+4x1o"))
    tp = ConvTensorProduct(irreps_in, irreps_sh, mid, ins, shared_weights=False, internal_weights=False)

    def no_kernel(device, dtype):
        raise RuntimeError("no kernel")

    monkeypatch.setattr(tp, "_kernel", no_kernel)
    assert tp._split_kernel(torch.zeros(1)) is None
    monkeypatch.setattr(tp, "_kernel", lambda device, dtype: {"op": object()})
    assert tp._split_kernel(torch.zeros(1)) is None
    kernel = {"op": _StandInKernel(tp, irreps_in.dim, mid.dim)}
    monkeypatch.setattr(tp, "_kernel", lambda device, dtype: kernel)
    assert tp._split_kernel(torch.zeros(1)) is kernel


@pytest.mark.skipif(GPU, reason="the fused kernel builds where CUDA and cuEquivariance are present")
@pytest.mark.parametrize("mapped", [False, True])
def test_conv_without_kernel_falls_back_to_reference(monkeypatch, mapped):
    """Switched on where its kernel cannot build, the fused message pass maps no weights, warns,
    switches itself off and gives the gathered reference sums (the message divisor applied)."""
    monkeypatch.delenv("XNN_FAST_STRICT", raising=False)
    irreps_in, irreps_sh = o3.Irreps("4x0e+4x1o"), o3.Irreps.spherical_harmonics(2)
    mid, ins = tp_out_irreps_with_instructions(irreps_in, irreps_sh, o3.Irreps("4x0e+4x1o"))
    tp = ConvTensorProduct(irreps_in, irreps_sh, mid, ins, shared_weights=False, internal_weights=False)
    torch.manual_seed(2)
    x = torch.randn(6, irreps_in.dim)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 0], [1, 2, 3, 4, 5, 0, 3]])
    y = torch.randn(7, irreps_sh.dim)
    w = torch.randn(7, tp.weight_numel)
    expected = torch.zeros(6, mid.dim).index_add_(0, edge_index[1], tp(x[edge_index[0]], y, w))
    tp.fast_active = True
    assert tp.weight_map(w, 2.0) == (None, None)
    with pytest.warns(RuntimeWarning, match="reference implementation"):
        if mapped:
            # the edges read their rows of the weights through an index
            rows = torch.tensor([3, 0, 6, 1, 5, 2, 4])
            w_rows = torch.empty_like(w)
            w_rows[rows] = w
            out = tp.conv(x, y, w_rows, edge_index, 6, weights_mapped=True, weight_index=rows)
        else:
            out = tp.conv(x, y, w, edge_index, 6, 2.0)
    assert not tp.fast_active and not tp.fast_eligible
    assert torch.allclose(out, expected if mapped else expected / 2.0, rtol=0, atol=1e-13)


@pytest.mark.parametrize("parts", [False, True])
def test_kernel_call_reads_the_weights_through_an_index(parts):
    """With a weight index each edge reads its row of the weights: through the kernel's input indices,
    or gathered per edge for a kernel whose parts cannot be called apart."""
    from xnn.gnn.fast.conv import _run_channelwise
    irreps_in, irreps_sh = o3.Irreps("4x0e+4x1o"), o3.Irreps.spherical_harmonics(2)
    mid, ins = tp_out_irreps_with_instructions(irreps_in, irreps_sh, o3.Irreps("4x0e+4x1o"))
    tp = ConvTensorProduct(irreps_in, irreps_sh, mid, ins, shared_weights=False,
                           internal_weights=False).double()
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(6, irreps_in.dim, generator=gen, dtype=torch.float64)
    src, dst = torch.tensor([0, 1, 2, 3, 4, 5, 0]), torch.tensor([1, 2, 3, 4, 5, 0, 3])
    y = torch.randn(7, irreps_sh.dim, generator=gen, dtype=torch.float64)
    w = torch.randn(3, tp.weight_numel, generator=gen, dtype=torch.float64)
    rows = torch.tensor([2, 0, 1, 1, 0, 2, 0])
    expected = torch.zeros(6, mid.dim, dtype=torch.float64).index_add_(0, dst, tp(x[src], y, w[rows]))
    if parts:
        op = _StandInKernel(tp, irreps_in.dim, mid.dim)
    else:
        def op(x1, x2, weight, indices_1, indices_out, size_out):
            return torch.zeros(size_out, mid.dim, dtype=x1.dtype).index_add_(0, indices_out,
                                                                             tp(x1[indices_1], x2, weight))
    out = _run_channelwise(op, x, y, w, src, dst, 6, rows)
    assert torch.allclose(out, expected, rtol=0, atol=1e-13)


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
    # the blocks that can run here (the fused radial MLP needs float32 and its layer shapes)
    capable = list(fast.iter_fast_capable(core, torch.device("cuda"), dtype))
    assert capable and all(m.fast_active for m in capable)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_compile_twice_differentiable():
    """A compiled function with its compiled first and second derivatives (finite differences)."""
    from xnn.common.models.fast import compile_twice_differentiable

    def fn(a, b, c):
        n = (a * a).sum(dim=1, keepdim=True) + 1.0
        return a * b / n, (b * c).sum(dim=1) + c.pow(3).sum(dim=1)

    compiled = compile_twice_differentiable(fn, 3)
    gen = torch.Generator().manual_seed(0)
    args = [torch.randn(5, 4, dtype=torch.float64, generator=gen).cuda().requires_grad_(True)
            for _ in range(3)]
    for got, ref in zip(compiled(*args), fn(*args)):
        assert torch.allclose(got, ref, rtol=0, atol=1e-14)
    assert torch.autograd.gradcheck(compiled, args)
    assert torch.autograd.gradgradcheck(compiled, args)


def test_gnn_models_name_their_auto_policies():
    """Every equivariant GNN reads one policy per mode from xnn.gnn.fast; NequIP has its own."""
    import xnn.gnn.fast as fast
    from xnn.common.models.fast import AutoPolicy
    from xnn.gnn.models.base import EquivariantGNN
    from xnn.gnn.models.nequip import NequIP

    default = ("AUTO_POLICY", "AUTO_POLICY_TRAINING")
    assert getattr(EquivariantGNN, "fast_policies", default) == default
    for name in default + NequIP.fast_policies:
        assert isinstance(getattr(fast, name), AutoPolicy)
    assert NequIP.fast_policies != default


def test_auto_policy_upper_bounds(monkeypatch):
    """A fast path that wins only on small inputs is taken below the upper bound of its GPU (or the default)."""
    from xnn.common.models.fast import AutoPolicy
    policy = AutoPolicy(max_edges={"A100": 1000}, max_edges_float64={"A100": 500}, default_max_edges=2000)
    cuda = torch.device("cuda")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "NVIDIA A100-SXM4-80GB")
    assert policy.upper_bound(cuda, torch.float32) == 1000 and policy.upper_bound(cuda, torch.float64) == 500
    assert policy.decide(cuda, torch.float32, 999) and not policy.decide(cuda, torch.float32, 1000)
    assert not policy.decide(cuda, torch.float64, 500)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "Some Future GPU")
    assert policy.upper_bound(cuda, torch.float32) == 2000 and policy.upper_bound(cuda, torch.float64) is None
    assert policy.decide(cuda, torch.float64, 10**9)


def test_separate_shapes(monkeypatch):
    """Inside the context dynamic sizes get symbols of their own; without the compiler's config it does nothing."""
    import sys

    import torch.fx.experimental._config as fx_config
    with fast.separate_shapes():
        assert fx_config.use_duck_shape is False
    assert fx_config.use_duck_shape is True
    monkeypatch.setitem(sys.modules, "torch.fx.experimental._config", None)
    with fast.separate_shapes():
        assert fx_config.use_duck_shape is True
