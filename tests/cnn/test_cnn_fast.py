"""Fast paths of the voxel models: the 3D convolutions with cuDNN's algorithm search.

The searched convolution is a multilinear block; its value and its first and
second derivatives are checked against ``conv3d`` (float64, on the CPU, where
the cuDNN flags do nothing, and on the GPU). On the GPU the whole models
(SteerableCNN with and without batch norm and strides, CNN3D) give the same
energies, forces and weight gradients of a force loss with ``use_fast=True``
as with the reference, in evaluation and in training mode; off the GPU the
fast path falls back to the reference.
"""
import copy

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("e3nn")
from test_cnn3d import SPECIES, _graph, _structure  # noqa: E402
from xnn.cnn.fast import AUTO_POLICY, AUTO_POLICY_TRAINING, conv3d, supported  # noqa: E402
from xnn.cnn.models import CNN3D, SteerableCNN  # noqa: E402
from xnn.common.data import collate  # noqa: E402
from xnn.common.models import ForceStressOutput  # noqa: E402

CUDA = torch.cuda.is_available()


@pytest.fixture(autouse=True)
def _no_tf32():
    """Round-off comparisons: PyTorch lets cuDNN take TF32 by default, which the reference's float32 convolutions
    would then use in some layers (the fast path never does)."""
    old = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = old


def _devices():
    return ["cpu", "cuda"] if CUDA else ["cpu"]


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("stride, padding, groups", [(1, 1, 1), (2, 2, 1), (1, 1, 2)])
def test_searched_conv_value_and_derivatives(device, stride, padding, groups):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 4, 6, 6, 6, generator=g, dtype=torch.float64).to(device).requires_grad_()
    w = torch.randn(3 * groups, 4 // groups, 3, 3, 3, generator=g, dtype=torch.float64).to(device).requires_grad_()
    ref = F.conv3d(x, w, stride=stride, padding=padding, groups=groups)
    out = conv3d(x, w, stride=stride, padding=padding, groups=groups)
    assert torch.allclose(out, ref, rtol=1e-12, atol=1e-12)

    def fn(a, b):
        return conv3d(a, b, stride=stride, padding=padding, groups=groups)
    assert torch.autograd.gradcheck(fn, (x, w))
    assert torch.autograd.gradgradcheck(fn, (x, w))


@pytest.mark.parametrize("device", _devices())
def test_searched_conv_in_batch_chunks(device, monkeypatch):
    import xnn.cnn.fast.conv as fc
    monkeypatch.setattr(fc, "CHUNK_ELEMENTS", 1)    # one grid per cuDNN call
    g = torch.Generator().manual_seed(2)
    x = torch.randn(3, 2, 5, 5, 5, generator=g, dtype=torch.float64).to(device).requires_grad_()
    w = torch.randn(4, 2, 3, 3, 3, generator=g, dtype=torch.float64).to(device).requires_grad_()
    assert torch.allclose(conv3d(x, w, padding=1), F.conv3d(x, w, padding=1), rtol=1e-12, atol=1e-12)

    def fn(a, b):
        return conv3d(a, b, padding=1, stride=2)
    assert torch.autograd.gradcheck(fn, (x, w))
    assert torch.autograd.gradgradcheck(fn, (x, w))


def test_searched_conv_bias():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 2, 5, 5, 5, generator=g, dtype=torch.float64)
    w = torch.randn(3, 2, 3, 3, 3, generator=g, dtype=torch.float64)
    b = torch.randn(3, generator=g, dtype=torch.float64)
    assert torch.allclose(conv3d(x, w, b, padding=1), F.conv3d(x, w, b, padding=1), rtol=1e-12, atol=1e-12)


def test_supported_and_policies():
    cpu = torch.device("cpu")
    assert not supported(cpu, torch.float64)
    assert not AUTO_POLICY.decide(cpu, torch.float64, 10**6)
    assert not AUTO_POLICY_TRAINING.decide(cpu, torch.float32, 10**6)


def _steerable(**kw):
    torch.manual_seed(0)
    kw.setdefault("fields", [(3, 2, 1), (4, 2), (4,)])
    m = SteerableCNN(species=SPECIES, cutoff=4.0, grid_size=9, kernel_size=3, **kw)
    torch.nn.init.normal_(m.readout[-1].weight)
    torch.nn.init.normal_(m.readout[-1].bias)
    return m


def _cnn3d(**kw):
    torch.manual_seed(0)
    m = CNN3D(species=SPECIES, cutoff=4.0, grid_size=9, channels=(6, 8, 8), kernel_size=3, **kw)
    torch.nn.init.normal_(m.readout[-1].weight)
    torch.nn.init.normal_(m.readout[-1].bias)
    return m


MODELS = {
    "se3cnn": lambda: _steerable(),
    "se3cnn_bn": lambda: _steerable(normalization="batch"),
    "cnn3d": lambda: _cnn3d(),
    "cnn3d_bn": lambda: _cnn3d(normalization="batch"),
}


def _batch(device, dtype):
    g = collate([_graph(_structure(n=5, seed=s)) for s in (3, 4)])
    g = g.to(device)
    g.pos = g.pos.to(dtype)
    return g


def _outputs(model, g, train):
    model.train(train)
    model.zero_grad(set_to_none=True)
    out = ForceStressOutput(model).train(train)(g)
    loss = out["energy"].pow(2).sum() + out["forces"].pow(2).sum()
    if train:
        loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    return out["energy"].detach(), out["forces"].detach(), grads


@pytest.mark.skipif(not CUDA, reason="the searched convolutions run on the GPU")
@pytest.mark.parametrize("name", sorted(MODELS))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("train", [False, True])
def test_fast_path_parity(name, dtype, train):
    ref = MODELS[name]().to("cuda", dtype).set_use_fast(False)
    fast = copy.deepcopy(ref).set_use_fast(True)
    g = _batch("cuda", dtype)
    e_r, f_r, g_r = _outputs(ref, g, train)
    e_f, f_f, g_f = _outputs(fast, g, train)
    assert any(m.fast_active for m in fast.__dict__["_fast_blocks"])
    tol = dict(rtol=1e-4, atol=1e-5) if dtype == torch.float32 else dict(rtol=1e-10, atol=1e-11)
    assert torch.allclose(e_f, e_r, **tol)
    assert torch.allclose(f_f, f_r, **tol)
    assert g_f.keys() == g_r.keys()
    # relative to the largest gradient of the model: some vanish exactly (a convolution bias before batch norm)
    scale = max((float(t.abs().max()) for t in g_r.values()), default=1.0)
    for k in g_r:
        assert float((g_f[k] - g_r[k]).abs().max()) / scale < (1e-4 if dtype == torch.float32 else 1e-10), k


@pytest.mark.parametrize("name", ["se3cnn", "cnn3d"])
def test_fast_path_falls_back_on_cpu(name):
    ref = MODELS[name]().set_use_fast(False)
    fast = copy.deepcopy(ref).set_use_fast(True)
    g = _batch("cpu", torch.float64)
    e_r, f_r, _ = _outputs(ref, g, False)
    e_f, f_f, _ = _outputs(fast, g, False)
    assert not any(m.fast_active for m in fast.__dict__["_fast_blocks"])
    assert torch.equal(e_f, e_r) and torch.equal(f_f, f_r)
