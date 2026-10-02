"""Radial networks that recompute their hidden layers give the same results.

``RadialNet`` (MACE's ``conv_tp_weights``, NequIP's ``fc``) can run its hidden
layers again in the backward pass instead of keeping their activations, which
lowers the memory of large force / stress evaluations.
"""
import copy

import numpy as np
import pytest
import torch

pytest.importorskip("e3nn")

from xnn.common.config import from_dict  # noqa: E402
from xnn.common.data import structure_to_graph  # noqa: E402
from xnn.common.models import ForceStressOutput, build_model  # noqa: E402
from xnn.gnn.models import RadialNet, set_recompute_radial  # noqa: E402
from xnn.gnn.models import blocks  # noqa: E402

SPECIES = [1, 6, 8]
CONFIGS = {
    "nequip": {"name": "nequip", "cutoff": 4.0, "n_features": 16, "n_interactions": 2,
               "l_max": 2, "species": SPECIES, "avg_num_neighbors": 10.0},
    "mace": {"name": "mace", "cutoff": 4.0, "n_features": 16, "n_interactions": 2,
             "extra": {"species": SPECIES, "max_ell": 3, "correlation": 3, "num_channels": 16,
                       "max_L": 1}},
}


def _model(name):
    torch.manual_seed(0)
    core = build_model(from_dict({"model": CONFIGS[name]}).model).double()
    return ForceStressOutput(core, compute_stress=True)


def _graph():
    rng = np.random.default_rng(1)
    s = {"pos": torch.tensor(rng.uniform(0, 6, (12, 3))),
         "atomic_numbers": torch.tensor([1, 6, 8] * 4),
         "cell": torch.eye(3, dtype=torch.float64) * 6.0, "pbc": torch.ones(3, dtype=torch.bool)}
    return structure_to_graph(s, 4.0)


def _spy(monkeypatch):
    calls = []
    original = RadialNet._forward_recomputed

    def spy(self, x):
        calls.append(x.shape[0])
        return original(self, x)
    monkeypatch.setattr(RadialNet, "_forward_recomputed", spy)
    return calls


@pytest.mark.parametrize("name", list(CONFIGS))
def test_recomputed_forces_and_stress_are_identical(name, monkeypatch):
    model = _model(name).eval()
    assert any(isinstance(m, RadialNet) for m in model.modules())
    g = _graph()
    ref = model(copy.copy(g))
    calls = _spy(monkeypatch)
    out = set_recompute_radial(model, True)(copy.copy(g))
    assert calls
    for key in ("energy", "forces", "stress"):
        assert torch.equal(out[key], ref[key]), key


def test_auto_recomputes_large_evaluations_only(monkeypatch):
    model = _model("mace")
    calls = _spy(monkeypatch)
    g = _graph()
    model.eval()(copy.copy(g))
    assert not calls                              # below RECOMPUTE_MIN_EDGES
    monkeypatch.setattr(blocks, "RECOMPUTE_MIN_EDGES", 1)
    model(copy.copy(g))
    assert calls
    calls.clear()
    model.train()(copy.copy(g))                   # training keeps the activations
    assert not calls
    set_recompute_radial(model, False).eval()(copy.copy(g))
    assert not calls
    with pytest.raises(ValueError):
        set_recompute_radial(model, "sometimes")


def test_training_gradients_with_recomputation_are_identical():
    # a force loss differentiates through the forces (create_graph), so the
    # recomputed hidden layers take part in a second derivative
    g = _graph()
    grads = []
    for recompute in (False, True):
        model = set_recompute_radial(_model("mace").train(), recompute)
        out = model(copy.copy(g))
        loss = out["energy"].pow(2).sum() + out["forces"].pow(2).sum()
        params = [p for p in model.parameters() if p.requires_grad]
        grads.append(torch.autograd.grad(loss, params))
    for a, b in zip(*grads):
        assert torch.equal(a, b)


def test_state_dict_and_torchscript_are_unchanged(tmp_path):
    from e3nn.nn import FullyConnectedNet

    from xnn.common.deploy import export_torchscript_potential

    torch.manual_seed(0)
    plain = FullyConnectedNet([8, 16, 16, 24], torch.nn.functional.silu)
    torch.manual_seed(0)
    radial = RadialNet([8, 16, 16, 24], torch.nn.functional.silu)
    assert plain.state_dict().keys() == radial.state_dict().keys()
    x = torch.randn(5, 8)
    assert torch.equal(plain(x), radial(x))
    core = set_recompute_radial(_model("mace").model, True)
    path = export_torchscript_potential(core, 4.0, str(tmp_path / "model.pt"))
    assert torch.jit.load(path) is not None
