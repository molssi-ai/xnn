"""The folded radial network ``RadialNet`` against e3nn's ``FullyConnectedNet``.

``RadialNet`` keeps e3nn's parameters and evaluates the layers as folded matrix
products, optionally with an output column map (``out_index``, ``out_scale``) applied
to the last matrix; TorchScript applies the layers one by one, and a network whose
layers are not e3nn's (a replaced block) runs as written.
"""
import pytest
import torch

pytest.importorskip("e3nn")

from e3nn import nn as e3nn_nn  # noqa: E402

from xnn.gnn.models import RadialNet  # noqa: E402


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _pair(out_act=False, variance_out=1.0, seed=0):
    """A RadialNet and the e3nn network with the same weights."""
    hs = [5, 7, 6, 4]
    torch.manual_seed(seed)
    net = RadialNet(hs, torch.nn.functional.silu, variance_out=variance_out, out_act=out_act)
    ref = e3nn_nn.FullyConnectedNet(hs, torch.nn.functional.silu, variance_out=variance_out, out_act=out_act)
    ref.load_state_dict(net.state_dict())
    return net, ref


def _map():
    """A column map: a permutation with a repeat, and a scale per output column."""
    return torch.tensor([2, 0, 3, 1, 2]), torch.tensor([0.5, -1.25, 2.0, 0.75, 3.0])


def _mapped(y, index, scale):
    return y.index_select(1, index) * scale


@pytest.mark.parametrize("out_act,variance_out", [(False, 1.0), (True, 1.0), (True, 2.0)])
def test_column_map_matches_mapped_output(out_act, variance_out):
    """With and without an output activation, the folded map equals mapping e3nn's output."""
    net, ref = _pair(out_act, variance_out)
    x = torch.randn(9, 5)
    index, scale = _map()
    assert torch.allclose(net(x), ref(x), rtol=0, atol=1e-13)
    assert torch.allclose(net(x, index, scale), _mapped(ref(x), index, scale), rtol=0, atol=1e-13)
    assert torch.allclose(net(x, None, scale[:4]), ref(x) * scale[:4], rtol=0, atol=1e-13)


def test_scripted_network_applies_the_map():
    """TorchScript applies the layers one by one, then the column map: the eager values."""
    net, _ = _pair(out_act=True, variance_out=2.0)
    scripted = torch.jit.script(net)
    x = torch.randn(9, 5)
    index, scale = _map()
    assert torch.allclose(scripted(x, index, scale), net(x, index, scale), rtol=0, atol=1e-13)
    assert torch.allclose(scripted(x, None, scale[:4]), net(x, None, scale[:4]), rtol=0, atol=1e-13)
    assert torch.allclose(scripted(x), net(x), rtol=0, atol=1e-13)


def test_replaced_layer_runs_as_written():
    """A layer that is not e3nn's (no folding plan): the layers in turn, then the map."""
    net, ref = _pair()
    torch.manual_seed(3)
    plain = torch.nn.Linear(7, 6)
    net.layer1 = plain
    ref.layer1 = plain
    x = torch.randn(9, 5)
    index, scale = _map()
    assert net._plan() is None
    expect = ref.layer2(plain(ref.layer0(x)))
    assert torch.allclose(net(x), expect, rtol=0, atol=1e-13)
    assert torch.allclose(net(x, index, scale), _mapped(expect, index, scale), rtol=0, atol=1e-13)
    assert torch.allclose(net(x, None, scale[:4]), expect * scale[:4], rtol=0, atol=1e-13)
