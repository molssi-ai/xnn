"""Multilinear blocks (xnn.common.models.multilinear): the derivatives from the block itself, also where
torch cannot say which operands a backward pass reaches, and the interface a block implements."""
import pytest
import torch

from xnn.common.models import multilinear as ml


class _Product(ml.MultilinearBlock):
    """``a * b`` elementwise: one term, linear in each operand."""

    def value(self, ops, only=-1):
        return ops[0] * ops[1]

    def adjoints(self, g, ops, needs, only=-1):
        return [g * ops[1] if needs[0] else None, g * ops[0] if needs[1] else None]


@pytest.mark.parametrize("edges", [True, False])
def test_first_and_second_derivatives(monkeypatch, edges):
    """Values, gradients and second derivatives equal autograd's; without the autograd edges every
    requested gradient is still computed."""
    if not edges:
        def missing(_):
            raise RuntimeError("no gradient edge")
        monkeypatch.setattr(torch.autograd.graph, "get_gradient_edge", missing)
    gen = torch.Generator().manual_seed(0)
    a, b = (torch.randn(5, dtype=torch.float64, generator=gen).requires_grad_(True) for _ in range(2))
    assert ml._nodes([a, b.detach()])[1] is None
    assert (ml._nodes([a])[0] is None) != edges
    y = ml.multilinear(_Product(), a, b)
    assert torch.equal(y, a * b)
    (ga,) = torch.autograd.grad((y * y).sum(), a)
    assert torch.allclose(ga, 2 * a * b * b, rtol=1e-15)
    assert torch.autograd.gradcheck(lambda x, z: ml.multilinear(_Product(), x, z), (a, b))
    assert torch.autograd.gradgradcheck(lambda x, z: ml.multilinear(_Product(), x, z), (a, b))


def test_the_interface_is_abstract():
    """A block must implement its value and its adjoints."""
    block = ml.MultilinearBlock()
    x = torch.ones(2)
    with pytest.raises(NotImplementedError):
        block.value([x, x])
    with pytest.raises(NotImplementedError):
        block.adjoints(x, [x, x], [True, True])
