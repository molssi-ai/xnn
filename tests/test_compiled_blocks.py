"""Compiled blocks on the CPU (xnn.common.models.compiled and compile_twice_differentiable of
xnn.common.models.fast): values and first and second derivatives equal the reference forward's, blocks
of one structure share their compiled code and a block with acting dropout runs the reference.

The compiler runs with its eager backend here (the generated kernels are checked by the GPU tests of
the models); with TORCHDYNAMO_DISABLE=1 the compiled functions run as plain Python."""
import functools
import sys

import pytest
import torch

from xnn.common.models import compiled
from xnn.common.models.fast import compile_twice_differentiable


@pytest.fixture(autouse=True)
def _eager_compiler(monkeypatch):
    monkeypatch.setattr(torch, "compile", functools.partial(torch.compile, backend="eager"))


class _Block(torch.nn.Module):
    """A block whose reference sums transformed rows into buckets (an index and a size as constants)."""

    def __init__(self, p=0.0):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4, dtype=torch.float64)
        self.drop = torch.nn.Dropout(p)
        self.register_buffer("scale", torch.tensor(2.0, dtype=torch.float64))

    def _reference(self, x, index, n):
        h = torch.tanh(self.lin(self.drop(x))) * self.scale
        return torch.zeros(n, 4, dtype=h.dtype).index_add(0, index, h)


def _inputs(seed=0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(6, 4, dtype=torch.float64, generator=gen).requires_grad_(True)
    return x, torch.tensor([0, 1, 2, 0, 1, 2])


def test_supported_devices():
    """The compiled path is taken on a CUDA device in float32 or float64 only."""
    cuda = torch.device("cuda")
    assert not compiled.supported(torch.device("cpu"), torch.float64)
    assert compiled.supported(cuda, torch.float32) == (sys.version_info < (3, 13))
    assert not compiled.supported(cuda, torch.float16)


def test_compile_twice_differentiable_on_the_cpu():
    """A compiled function with its compiled first and second derivatives (finite differences), with a
    constant argument that is not differentiated."""
    def fn(a, b, c, k):
        n = (a * a).sum(dim=1, keepdim=True) + k
        return a * b / n, (b * c).sum(dim=1) + c.pow(3).sum(dim=1)

    wrapped = compile_twice_differentiable(fn, 3, 1)
    gen = torch.Generator().manual_seed(0)
    args = [torch.randn(3, 2, dtype=torch.float64, generator=gen).requires_grad_(True) for _ in range(3)]
    k = torch.ones(3, 1, dtype=torch.float64)
    for got, ref in zip(wrapped(*args, k), fn(*args, k)):
        assert torch.allclose(got, ref, rtol=0, atol=1e-14)
    assert torch.autograd.gradcheck(lambda *xs: wrapped(*xs, k), args)
    assert torch.autograd.gradgradcheck(lambda *xs: wrapped(*xs, k), args)
    single = compile_twice_differentiable(lambda a: (a * a).sum(dim=1), 1)
    assert torch.allclose(single(args[0]), (args[0] * args[0]).sum(dim=1), rtol=0, atol=1e-15)


@pytest.mark.parametrize("training", [False, True])
def test_a_compiled_block_equals_its_reference(training):
    """Values, forces (the derivative by the inputs) and, in training mode, the weight gradients of a force
    loss (second derivatives) equal the reference's."""
    torch.manual_seed(0)
    block = _Block().train(training)
    x, index = _inputs()
    out = compiled.run_compiled(block, x, index, 3)
    ref = block._reference(x, index, 3)
    assert torch.allclose(out, ref, rtol=0, atol=1e-14)
    grads = []
    for y in (out, ref):
        (g,) = torch.autograd.grad((y * y).sum(), x, create_graph=training)
        grads.append(g)
    assert torch.allclose(grads[0], grads[1], rtol=0, atol=1e-13)
    if training:
        weights = []
        for g in grads:
            weights.append(torch.autograd.grad((g * g).sum(), list(block.parameters())))
        for a, b in zip(*weights):
            assert torch.allclose(a, b, rtol=0, atol=1e-12)


def test_blocks_of_one_structure_share_their_compiled_code():
    """A second block of the same structure reuses the first one's compiled function with its own weights;
    the cached state follows a registration."""
    torch.manual_seed(1)
    a, b = _Block().eval(), _Block().eval()
    x, index = _inputs(1)
    compiled.run_compiled(a, x, index, 3)
    shared = len(compiled._SHARED)
    out = compiled.run_compiled(b, x, index, 3)
    assert len(compiled._SHARED) == shared
    assert torch.allclose(out, b._reference(x, index, 3), rtol=0, atol=1e-14)
    state = b.__dict__["_compiled_state"]
    compiled.run_compiled(b, x, index, 3)
    assert b.__dict__["_compiled_state"] is state
    b.register_buffer("extra", torch.zeros(1))
    compiled.run_compiled(b, x, index, 3)
    assert b.__dict__["_compiled_state"] is not state


def test_a_block_with_acting_dropout_runs_the_reference():
    """In training mode dropout draws as the reference does (the same draws for one seed); in evaluation
    mode it is off and the block is compiled."""
    block = _Block(p=0.5).train()
    x, index = _inputs(2)
    torch.manual_seed(5)
    out = compiled.run_compiled(block, x, index, 3)
    torch.manual_seed(5)
    assert torch.equal(out, block._reference(x, index, 3))
    block.eval()
    assert torch.allclose(compiled.run_compiled(block, x, index, 3), block._reference(x, index, 3), rtol=0,
                          atol=1e-14)
