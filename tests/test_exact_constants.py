"""Float64 constants of the GNN potentials are exact however the model was made.

Wigner-3j tensors and the bases built from them (the MACE ``U``, Allegro's block
tensor), fixed Bessel frequencies and normalization constants are buffers
computed at construction in the default dtype. A model built in float32 and cast
to float64, or given a float32 checkpoint, used to keep them float32-rounded, so
its float64 results sat 1e-10 to 1e-7 from an exact evaluation.
"""
import copy

import numpy as np
import pytest
import torch

pytest.importorskip("e3nn")

from xnn.common.config import from_dict  # noqa: E402
from xnn.common.data import structure_to_graph  # noqa: E402
from xnn.common.models import ForceStressOutput, build_model  # noqa: E402
from xnn.gnn.models import exact_float64_constants  # noqa: E402

SPECIES = [1, 6, 8]
CONFIGS = {
    "nequip": {"name": "nequip", "cutoff": 4.0, "n_features": 16, "n_interactions": 2,
               "l_max": 2, "species": SPECIES, "avg_num_neighbors": 10.0},
    "mace": {"name": "mace", "cutoff": 4.0, "n_features": 16, "n_interactions": 2,
             "extra": {"species": SPECIES, "max_ell": 3, "correlation": 3, "num_channels": 16,
                       "max_L": 1}},
    # the Agnesi transform and ZBL repulsion register constants float32 cannot hold
    "mace-agnesi-zbl": {"name": "mace", "cutoff": 4.0, "n_features": 16, "n_interactions": 2,
                        "extra": {"species": SPECIES, "max_ell": 2, "correlation": 2,
                                  "num_channels": 16, "max_L": 1, "distance_transform": "Agnesi",
                                  "pair_repulsion": True}},
    "allegro": {"name": "allegro", "cutoff": 4.0, "n_features": 8, "n_interactions": 2,
                "extra": {"species": SPECIES, "l_max": 2, "avg_num_neighbors": 6.0,
                          "two_body_latent": [16, 32], "latent": [32], "edge_eng": [16]}},
}


def _build(name, default_dtype):
    old = torch.get_default_dtype()
    torch.set_default_dtype(default_dtype)
    try:
        torch.manual_seed(0)
        return build_model(from_dict({"model": CONFIGS[name]}).model)
    finally:
        torch.set_default_dtype(old)


def _constants(model):
    return {k: v for k, v in model.named_buffers() if v.dtype.is_floating_point}


def _graph(dtype=torch.float64):
    rng = np.random.default_rng(1)
    s = {"pos": torch.tensor(rng.uniform(0, 6, (12, 3)), dtype=dtype),
         "atomic_numbers": torch.tensor([1, 6, 8] * 4),
         "cell": torch.eye(3, dtype=dtype) * 6.0, "pbc": torch.ones(3, dtype=torch.bool)}
    return structure_to_graph(s, 4.0)


def _parameters_from(target, source):
    target.load_state_dict({k: v for k, v in source.state_dict().items()
                            if k in dict(source.named_parameters())}, strict=False)


@pytest.mark.parametrize("name", list(CONFIGS))
def test_cast_to_float64_makes_the_constants_exact(name):
    exact = _build(name, torch.float64)
    cast = _build(name, torch.float32)
    rounded = {k: v.clone() for k, v in _constants(cast).items()}
    cast = cast.to(torch.float64)
    _parameters_from(exact, cast)
    constants, reference = _constants(cast), _constants(exact)
    assert constants and constants.keys() == reference.keys()
    for k, v in constants.items():
        assert torch.equal(v, reference[k]), k
    # the float32 build really was rounded, so the test sees the repair
    assert any(not torch.equal(rounded[k].double(), v) for k, v in constants.items())
    # and the whole model then agrees with the float64 build bit for bit
    a = ForceStressOutput(cast, compute_stress=True).eval()(copy.copy(_graph()))
    b = ForceStressOutput(exact, compute_stress=True).eval()(copy.copy(_graph()))
    for key in ("energy", "forces", "stress"):
        assert torch.equal(a[key], b[key]), key


@pytest.mark.parametrize("name", list(CONFIGS))
def test_float32_checkpoint_keeps_float64_constants_exact(name):
    exact = _build(name, torch.float64)
    expected = {k: v.clone() for k, v in _constants(exact).items()}
    checkpoint = _build(name, torch.float32).state_dict()      # float32-rounded constants
    exact.load_state_dict(checkpoint)
    for k, v in _constants(exact).items():
        assert torch.equal(v, expected[k]), k


@pytest.mark.parametrize("name", list(CONFIGS))
def test_float32_models_are_left_as_built(name):
    model = _build(name, torch.float32)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    assert exact_float64_constants(model) == 0
    model.load_state_dict(before)
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), k


def test_torchscript_export_after_the_repair(tmp_path):
    from xnn.common.deploy import export_torchscript_potential

    model = _build("nequip", torch.float32).to(torch.float64)
    path = export_torchscript_potential(model, 4.0, str(tmp_path / "model.pt"))
    scripted = torch.jit.load(path)
    buffers = dict(scripted.named_buffers())                    # under the wrapper's prefix
    for k, v in _constants(model).items():
        match = [b for n, b in buffers.items() if n.endswith("." + k)]
        assert match and torch.equal(match[0], v), k


def test_bessel_frequencies_keep_learned_and_custom_values():
    from xnn.gnn.featurizers.radial import BesselRBF

    learned = BesselRBF(8, 5.0, trainable=True).double()
    with torch.no_grad():
        learned.freqs.add_(0.25)
    before = learned.freqs.detach().clone()
    assert exact_float64_constants(learned) == 0 and torch.equal(learned.freqs, before)
    custom = BesselRBF(8, 5.0).double()
    custom.freqs.mul_(1.5)                                       # not n * pi: a deliberate value
    before = custom.freqs.clone()
    assert exact_float64_constants(custom) == 0 and torch.equal(custom.freqs, before)
    rounded = BesselRBF(8, 5.0).float().double()
    assert exact_float64_constants(rounded) == 1
    assert torch.equal(rounded.freqs, torch.pi * torch.arange(1, 9, dtype=torch.float64))


@pytest.mark.parametrize("default", [torch.float32, torch.float64])
def test_values_set_on_purpose_are_kept(default):
    from xnn.gnn.constants import register_constant

    old = torch.get_default_dtype()
    torch.set_default_dtype(default)
    try:
        block = torch.nn.Module()
        register_constant(block, "scale", 1.0)       # its exact value must not alias the buffer
    finally:
        torch.set_default_dtype(old)
    block.double()
    block.scale.fill_(0.873)                                  # a fitted value, not 1.0 rounded
    assert exact_float64_constants(block) == 0 and float(block.scale) == 0.873
    block.scale.fill_(1.0 + 3e-8)                             # 1.0 after float32 rounding noise
    assert exact_float64_constants(block) == 1 and float(block.scale) == 1.0


def test_a_checkpoints_own_coupling_basis_is_kept():
    # foundation checkpoints carry U bases that differ from a fresh one by O(1)
    # (other e3nn versions); the weights are trained in that basis
    model = _build("mace", torch.float64)
    state = model.state_dict()
    key = next(k for k in state if k.endswith("U_matrix_3"))
    state[key] = state[key] + 0.25 * torch.randn_like(state[key])
    model.load_state_dict(state)
    assert torch.equal(dict(model.named_buffers())[key], state[key])
    model.float().double()
    assert torch.allclose(dict(model.named_buffers())[key], state[key], rtol=1e-6, atol=1e-6)
