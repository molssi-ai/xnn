"""SphericalCNN fast path: Triton kernels of the spectral transforms and products.

The kernels are checked against the reference: the transforms of degree-blocked
spectra against ``S2Transform`` / ``SO3Transform``, the spectral products (matrix
products of the blocks) against einsums of the padded spectra (float64 and
float32), and their first and second derivatives by ``gradcheck`` /
``gradgradcheck`` (float64), on the GPU and under Triton's interpreter. The
correlation layers and the whole model give the reference outputs, forces and
force-loss weight gradients with the fast path, in evaluation and in training
mode; off the GPU the fast path falls back to the reference.
"""
import math
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.autograd import gradcheck, gradgradcheck

from xnn.cnn.models import (
    S2Convolution,
    S2Transform,
    SO3Convolution,
    SO3Transform,
    SphericalCNN,
    s2_near_identity_grid,
    so3_near_identity_grid,
)
from xnn.cnn.models.spherical import _compact_index
from xnn.common.data import collate, structure_to_graph
from xnn.common.models import ForceStressOutput

CUDA = torch.cuda.is_available()
INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"
SPECIES = [1, 6, 8]


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _kernel_devices():
    """Where the kernels run: the GPU, or the CPU under Triton's interpreter."""
    return (["cuda"] if CUDA else []) + (["cpu"] if INTERPRETED else [])


def _rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


def _tol(dtype):
    return 1e-12 if dtype == torch.float64 else 1e-5


def _randn(*shape, seed=0, device="cpu", dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float64).to(device, dtype)


def _blocked(spec, so3):
    """The degree-blocked real view ``(B C K, 2)`` of padded complex spectra ``(B, C, b, ...)``."""
    n_tail = 3 if so3 else 2
    b = spec.shape[-n_tail]
    parts = []
    for l in range(b):
        o, d = b - 1 - l, 2 * l + 1
        if so3:
            block = spec[:, :, l, o:o + d, o:o + d].permute(0, 2, 1, 3)
        else:
            block = spec[:, :, l, o:o + d].permute(0, 2, 1)
        parts.append(block.reshape(-1))
    return torch.view_as_real(torch.cat(parts))


def _unblocked(z, batch, channels, b, so3):
    """The padded complex spectra ``(B, C, b, ...)`` of a degree-blocked real view (differentiable)."""
    parts, start = [], 0
    for l in range(b):
        o, d = b - 1 - l, 2 * l + 1
        size = batch * channels * (d * d if so3 else d)
        if so3:
            block = z[start:start + size].view(batch, d, channels, d, 2).permute(0, 2, 1, 3, 4)
            parts.append(F.pad(block, (0, 0, o, o, o, o)))
        else:
            block = z[start:start + size].view(batch, d, channels, 2).permute(0, 2, 1, 3)
            parts.append(F.pad(block, (0, 0, o, o)))
        start += size
    return torch.view_as_complex(torch.stack(parts, dim=2).contiguous())


def _padded(c, b, so3):
    """The padded complex spectrum of a compact real view (differentiable)."""
    w = 2 * b - 1
    tail = (b, w, w) if so3 else (b, w)
    cc = torch.view_as_complex(c)
    flat = torch.zeros(*c.shape[:-2], math.prod(tail), dtype=cc.dtype, device=c.device)
    flat = flat.index_copy(-1, _compact_index(b, so3).to(c.device), cc)
    return flat.reshape(*c.shape[:-2], *tail)


def _needs_triton():
    pytest.importorskip("triton")


def _layout(spec, so3, padded):
    """Padded complex spectra ``(B, C, b, ...)`` in the fast path's layout (real view)."""
    if not padded:
        return _blocked(spec, so3)
    order = (2, 0, 3, 1, 4) if so3 else (2, 0, 3, 1)
    return torch.view_as_real(spec.permute(*order).reshape(-1))


def _spectra(z, batch, channels, b, so3, padded):
    """The inverse of :func:`_layout` (differentiable)."""
    if not padded:
        return _unblocked(z, batch, channels, b, so3)
    w = 2 * b - 1
    if so3:
        return torch.view_as_complex(z).view(b, batch, w, channels, w).permute(1, 3, 0, 2, 4)
    return torch.view_as_complex(z).view(b, batch, w, channels).permute(1, 3, 0, 2)


def _entries(b, so3, padded):
    from xnn.cnn.fast.spherical_kernels import compact_size
    w = 2 * b - 1
    return (b * w * w if so3 else b * w) if padded else compact_size(b, so3)


@pytest.fixture(params=[False, True], ids=["blocked", "padded"])
def padded(request, monkeypatch):
    """Both layouts of the spectra; the layers and the model take the one chosen here."""
    import xnn.cnn.fast.spherical as fast
    monkeypatch.setattr(fast, "PADDED_MAX_BATCH", 10 ** 9 if request.param else 0)
    return request.param


# the kernels

@pytest.mark.parametrize("device", _kernel_devices())
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("so3", [False, True])
def test_transforms_match_reference(device, dtype, so3, padded):
    _needs_triton()
    from xnn.cnn.fast.spherical import analyze, synthesize
    b_grid, b_spec = 4, 3
    tr = (SO3Transform if so3 else S2Transform)(b_grid, b_spec).to(device=device, dtype=dtype)
    x = _randn(2, 3, *([2 * b_grid] * (3 if so3 else 2)), device=device, dtype=dtype)
    assert _rel(analyze(tr, x, padded), _layout(tr.analyze(x), so3, padded)) < _tol(dtype)
    # in the padded layout the entries outside |m|, |n| <= l are ignored
    z = _randn(2 * 3 * _entries(b_spec, so3, padded), 2, seed=1, device=device, dtype=dtype)
    ref = tr.synthesize(_spectra(z, 2, 3, b_spec, so3, padded))
    assert _rel(synthesize(tr, z, (2, 3), padded), ref) < _tol(dtype)


@pytest.mark.parametrize("device", _kernel_devices())
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("so3", [False, True])
def test_product_matches_reference(device, dtype, so3, padded):
    """The per-degree matrix products and their gradients against the einsum of the padded spectra."""
    from xnn.cnn.fast.spherical import degree_matrices, spectral_product
    from xnn.cnn.fast.spherical_kernels import compact_size
    b, nb, ni, no = 4, 5, 3, 2
    kf, kz = compact_size(b, so3), compact_size(b, True)
    spec = _unblocked(_randn(nb * ni * kf, 2, seed=2, device=device, dtype=dtype), nb, ni, b, so3)
    f = _layout(spec, so3, padded).detach().requires_grad_()
    w = _randn(ni, no, kf, 2, seed=3, device=device, dtype=dtype).requires_grad_()
    g = _layout(_unblocked(_randn(nb * no * kz, 2, seed=4, device=device, dtype=dtype), nb, no, b, True),
                True, padded)
    equation = "bilmn,iolkn->bolmk" if so3 else "bilm,ioln->bolmn"
    ref = _layout(torch.einsum(equation, _spectra(f, nb, ni, b, so3, padded), _padded(w, b, so3)), True, padded)
    ref_f, ref_w = torch.autograd.grad((ref * g).sum(), (f, w))
    got = spectral_product(f, degree_matrices(torch.view_as_complex(w), b, so3, padded), b, nb, so3, padded)
    got_f, got_w = torch.autograd.grad((got * g).sum(), (f, w))
    for a, r in ((got, ref), (got_f, ref_f), (got_w, ref_w)):
        assert _rel(a, r) < _tol(dtype)


@pytest.mark.parametrize("device", _kernel_devices())
@pytest.mark.parametrize("so3", [False, True])
def test_kernels_are_twice_differentiable(device, so3, padded):
    """First and second derivatives of the transforms and the product (finite differences)."""
    _needs_triton()
    from xnn.cnn.fast.spherical import analyze, degree_matrices, spectral_product, synthesize
    from xnn.cnn.fast.spherical_kernels import compact_size
    fast_mode = device == "cpu"
    b_grid, b_spec = 3, 2
    tr = (SO3Transform if so3 else S2Transform)(b_grid, b_spec).to(device)
    x = _randn(1, 2, *([2 * b_grid] * (3 if so3 else 2)), device=device).requires_grad_()
    z = _randn(2 * _entries(b_spec, so3, padded), 2, seed=1, device=device).requires_grad_()
    f = _randn(3 * 2 * _entries(b_spec, so3, padded), 2, seed=2, device=device).requires_grad_()
    w = _randn(2, 2, compact_size(b_spec, so3), 2, seed=3, device=device).requires_grad_()

    def product(a, c):
        return spectral_product(a, degree_matrices(torch.view_as_complex(c), b_spec, so3, padded),
                                b_spec, 3, so3, padded)

    checks = [(lambda a: analyze(tr, a, padded), (x,)), (lambda a: synthesize(tr, a, (1, 2), padded), (z,)),
              (product, (f, w))]
    for fn, inputs in checks:
        assert gradcheck(fn, inputs, fast_mode=fast_mode)
        assert gradgradcheck(fn, inputs, fast_mode=fast_mode)


@pytest.mark.parametrize("device", _kernel_devices())
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_layers_match_reference(device, dtype, padded):
    """Outputs and the gradients of the input and every parameter of both correlation layers."""
    _needs_triton()
    torch.manual_seed(0)
    layers = [(S2Convolution(2, 3, 4, 3, s2_near_identity_grid(math.pi / 6, 4, 2)), (2, 2, 8, 8)),
              (SO3Convolution(3, 2, 3, 2, so3_near_identity_grid(math.pi / 6, math.pi, 4, 2, 2)), (2, 3, 6, 6, 6))]
    for layer, shape in layers:
        layer = layer.to(device, dtype)
        with torch.no_grad():
            layer.bias.normal_()
        x = _randn(*shape, device=device, dtype=dtype).requires_grad_()
        results = []
        for active in (False, True):
            layer.fast_active = active
            out = layer(x)
            g = _randn(*out.shape, seed=5, device=device, dtype=dtype)
            grads = torch.autograd.grad((out * g).sum(), (x, layer.kernel, layer.bias))
            results.append((out.detach(), *grads))
        for got, ref in zip(results[1], results[0]):
            assert _rel(got, ref) < 10 * _tol(dtype)


# the model

def _structure(n=5, seed=0, box=3.0):
    rng = np.random.default_rng(seed)
    return {"pos": rng.uniform(0, box, (n, 3)), "atomic_numbers": ([1, 6, 8] * n)[:n]}


def _batch(device, dtype):
    batch = collate([structure_to_graph(_structure(5, 1), 10.0),
                     structure_to_graph(_structure(4, 2), 10.0)]).to(device)
    batch.pos = batch.pos.to(dtype)
    if getattr(batch, "cell", None) is not None:
        batch.cell = batch.cell.to(dtype)
    return batch


def _model(**kw):
    torch.manual_seed(0)
    kw.setdefault("species", SPECIES)
    kw.setdefault("cutoff", 10.0)
    kw.setdefault("radius", 0.4)
    kw.setdefault("bandwidth", 3)
    kw.setdefault("features", [3, 4])
    kw.setdefault("bandwidths", [3, 2])
    kw.setdefault("n_alpha", 4)
    kw.setdefault("normalization", None)
    model = SphericalCNN(**kw)
    torch.nn.init.normal_(model.readout[-1].weight)
    torch.nn.init.normal_(model.readout[-1].bias)
    # nonzero biases everywhere, so the zero-initialized ones hide nothing
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (S2Convolution, SO3Convolution)):
                m.bias.normal_(0.0, 0.1)
    return model


def _convs(model):
    return [m for m in model.modules() if isinstance(m, (S2Convolution, SO3Convolution))]


def _run(model, batch, use_fast, train):
    """Energies, forces and (in training mode) the weight gradients of a force loss."""
    model.set_use_fast(use_fast).train(train)
    model.zero_grad(set_to_none=True)
    out = ForceStressOutput(model).train(train)(batch)
    assert all(m.fast_active == use_fast for m in _convs(model))
    res = {"energy": out["energy"].detach().double(), "forces": out["forces"].detach().double()}
    if train:
        (out["energy"].pow(2).sum() + out["forces"].pow(2).sum()).backward()
        res.update({f"grad:{name}": p.grad.detach().double().clone()
                    for name, p in model.named_parameters() if p.grad is not None})
    return res


def parity_errors(device, dtype, train, **options):
    """Relative differences of the fast path from the reference, per output; the weight
    gradients relative to the largest one (some vanish)."""
    model = _model(**options).to(device, dtype)
    # a fresh batch per run: the force wrapper attaches its displacement to the batch
    ref = _run(model, _batch(device, dtype), False, train)
    fast = _run(model, _batch(device, dtype), True, train)
    assert ref.keys() == fast.keys()
    grads = [v.abs().max() for k, v in ref.items() if k.startswith("grad:")]
    grad_scale = max(grads) if grads else None
    errors = {}
    for key, value in ref.items():
        scale = grad_scale if key.startswith("grad:") else value.abs().max()
        errors[key] = float((fast[key] - value).abs().max() / scale.clamp_min(1e-30))
    return errors


def test_fast_support_needs_cuda():
    from xnn.cnn.fast.spherical import AUTO_POLICY, AUTO_POLICY_TRAINING, supported
    cpu = torch.device("cpu")
    assert not supported(cpu, torch.float64)
    assert not AUTO_POLICY.decide(cpu, torch.float64, 10 ** 6)
    assert not AUTO_POLICY_TRAINING.decide(cpu, torch.float32, 10 ** 6)
    assert SphericalCNN.fast_policy_module == "xnn.cnn.fast.spherical"


def test_fast_path_falls_back_on_cpu():
    """Off the GPU ``use_fast=True`` runs the reference: the same numbers, no kernels."""
    model = _model()
    ref = _run(model, _batch("cpu", torch.float64), False, False)
    out = ForceStressOutput(model.set_use_fast(True).eval())(_batch("cpu", torch.float64))
    assert not any(m.fast_active for m in _convs(model))
    assert torch.equal(out["energy"].detach(), ref["energy"])
    assert torch.equal(out["forces"].detach(), ref["forces"])


@pytest.mark.skipif(not INTERPRETED, reason="runs where Triton's interpreter is on (no CUDA device)")
@pytest.mark.parametrize("train", [False, True])
def test_fast_path_parity_interpreted(monkeypatch, train, padded):
    """The fused correlations under Triton's interpreter give the reference outputs and gradients."""
    _needs_triton()
    import xnn.cnn.fast.spherical as fast
    monkeypatch.setattr(fast, "supported", lambda device, dtype: True)
    keys = set(_model().state_dict())
    errors = parity_errors("cpu", torch.float64, train)
    assert max(errors.values()) < 1e-10, errors
    model = _model()
    _run(model, _batch("cpu", torch.float64), True, False)
    assert set(model.state_dict()) == keys


_FAST_OPTIONS = [{}, {"normalization": "batch"}, {"set_readout": [6, 5, 4]},
                 {"bandwidth": 6, "features": [5, 6, 8], "bandwidths": [6, 4, 2], "n_alpha": 6}]


@pytest.mark.skipif(not CUDA, reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("options", _FAST_OPTIONS)
@pytest.mark.parametrize("train", [False, True])
def test_fast_path_parity(dtype, options, train, padded):
    """The fused correlations give the reference energies, forces and force-loss weight gradients."""
    _needs_triton()
    errors = parity_errors("cuda", dtype, train, **options)
    tol = 1e-10 if dtype == torch.float64 else 1e-5
    assert max(errors.values()) < tol, errors


@pytest.mark.skipif(not CUDA, reason="needs a CUDA device")
def test_auto_crossover_per_mode(monkeypatch):
    """"auto" applies the evaluation crossover in evaluation mode and the training one in training mode."""
    _needs_triton()
    import xnn.cnn.fast.spherical as fast
    from xnn.common.models.fast import AutoPolicy
    never = AutoPolicy(default_min_edges=10 ** 9, default_min_edges_float64=10 ** 9)
    always = AutoPolicy(default_min_edges=0, default_min_edges_float64=0)
    model = _model().to("cuda").set_use_fast("auto")
    names = ("AUTO_POLICY", "AUTO_POLICY_TRAINING")
    for training in (False, True):
        for policy in names:
            with monkeypatch.context() as patch:
                for name in names:
                    patch.setattr(fast, name, never if name == policy else always)
                ForceStressOutput(model.train(training)).train(training)(_batch("cuda", torch.float64))
            expect = policy != ("AUTO_POLICY_TRAINING" if training else "AUTO_POLICY")
            assert all(m.fast_active == expect for m in _convs(model)), (training, policy)
