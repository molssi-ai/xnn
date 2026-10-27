"""Config keys and options that xnn does not use are ignored, with a warning."""
import os
import warnings

import pytest
import yaml

from xnn.common.config import Config, apply_overrides, from_dict
from xnn.common.config.options import IgnoredOptionWarning, ModelOptions, model_options
from xnn.common.featurizers import make_radial_basis
from xnn.common.models import build_model

CONFIGS = os.path.join(os.path.dirname(__file__), "..", "configs", "model")


def _model(**extra):
    return from_dict({"model": {"name": "schnet", "cutoff": 5.0, "n_features": 16,
                                "n_interactions": 1, "n_rbf": 8, **extra}}).model


def test_unused_model_option_warns():
    with pytest.warns(IgnoredOptionWarning, match="n_filtrs"):
        build_model(_model(n_filtrs=16))


def test_used_options_are_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error", IgnoredOptionWarning)
        build_model(_model(energy_scale=2.0, species=[1, 8], dispersion={"name": "d4", "s9": 1.0}))


def test_unknown_dispersion_option_warns():
    with pytest.warns(IgnoredOptionWarning, match="s9x"):
        build_model(_model(dispersion={"name": "d4", "s9x": 1.0}))


def test_reads_are_shared_by_copies():
    tracked = ModelOptions({"a": 1, "b": 2, "c": 3})

    class Cfg:
        extra = tracked

    opts = model_options(Cfg)
    opts.get("a")
    assert "b" in opts.copy()
    assert tracked.unread() == ["c"]
    assert ModelOptions({"head": None}).unread() == []      # None: not set


def test_unknown_section_and_top_level_keys_warn():
    with pytest.warns(IgnoredOptionWarning, match="batch_sise"):
        cfg = from_dict({"data": {"batch_sise": 4, "batch_size": 8}})
    assert cfg.data.batch_size == 8
    with pytest.warns(IgnoredOptionWarning, match="sead"):
        from_dict({"sead": 1})


def test_overrides():
    cfg = apply_overrides(Config(), ["model.name='painn'", "model.n_rbf=5"])
    assert cfg.model.extra == {}             # the MACE defaults go with MACE
    apply_overrides(cfg, ["model.epsilon=1e-9"])
    assert cfg.model.extra == {"epsilon": 1e-9}
    with pytest.warns(IgnoredOptionWarning, match="lrr"):
        apply_overrides(cfg, ["optim.lrr=1"])
    with pytest.warns(IgnoredOptionWarning, match="model.cutoff"):
        apply_overrides(cfg, ["model.cutoff"])


def test_default_extra_is_mace_only():
    assert Config().model.extra["correlation"] == 3
    assert from_dict({"model": {"name": "painn"}}).model.extra == {}


def test_basis_options_of_another_basis_warn():
    with pytest.warns(IgnoredOptionWarning, match="trainable"):
        make_radial_basis("gaussian", 8, 5.0, trainable=True)
    with pytest.warns(IgnoredOptionWarning, match="gamma"):
        make_radial_basis("bessel", 8, 5.0, gamma=10.0)


@pytest.mark.parametrize("name", sorted(f[:-5] for f in os.listdir(CONFIGS) if f.endswith(".yaml")))
def test_shipped_model_configs_use_every_option(name):
    """Every shipped model config builds without ignoring any of its options."""
    with open(os.path.join(CONFIGS, f"{name}.yaml")) as f:
        cfg = from_dict({"model": yaml.safe_load(f)})
    with warnings.catch_warnings():
        warnings.simplefilter("error", IgnoredOptionWarning)
        try:
            build_model(cfg.model)
        except (OSError, RuntimeError, ValueError) as err:
            # parameter files fetched on first use; force fields built for a topology
            if any(word in str(err).lower() for word in ("download", "offline", "network", "topology")):
                pytest.skip(f"{name}: needs data the test does not have ({err})")
            raise


def test_resolved_options_record_the_defaults():
    """The config a build returns (and a checkpoint saves) holds the defaults the model used."""
    from xnn.common.models.registry import prepare_model
    model, cfg = prepare_model(from_dict({"model": {"name": "painn", "n_features": 8,
                                                    "dipole": True}}).model)
    assert cfg.extra["dipole"] is True                         # given
    assert cfg.extra["radial_basis"] == "gaussian"             # the default used, recorded
    assert cfg.extra["atomic_dipoles"] is False
    yaml.safe_dump(cfg.extra)                                  # a config file can hold it
    with warnings.catch_warnings():
        warnings.simplefilter("error", IgnoredOptionWarning)
        rebuilt = build_model(cfg)
    assert sorted(rebuilt.state_dict()) == sorted(model.state_dict())


def test_old_checkpoint_load_names_the_old_defaults():
    """A checkpoint saved under the old PaiNN defaults fails with the options that rebuild it."""
    from xnn.common.models import ForceStressOutput
    from xnn.common.models.hub.checkpoint import build_potential
    from xnn.gnn.models.painn import PaiNN

    old = ForceStressOutput(PaiNN(n_features=8, radial_basis="bessel")).state_dict()
    cfg = from_dict({"model": {"name": "painn", "n_features": 8, "n_interactions": 3, "n_rbf": 20,
                               "cutoff": 5.0}})
    with pytest.raises(RuntimeError, match="radial_basis: bessel"):
        build_potential(cfg, old)
    cfg.model.extra["radial_basis"] = "bessel"
    build_potential(cfg, old)


def test_pop_and_setdefault_are_reads_and_defaults_stay_plain():
    opts = ModelOptions({"a": 1, "b": 2})
    assert opts.pop("a") == 1 and opts.setdefault("c", 3) == 3
    assert opts.read == {"a", "c"} and opts.unread() == ["b"]
    opts.get("table", {"x": (1, 2)})                         # a dict default, its tuple as a list
    opts.get("fn", print)                                    # no config file can hold a function
    assert opts.resolved() == {"b": 2, "c": 3, "table": {"x": [1, 2]}}


def test_override_paths_into_dicts_and_unknown_sections():
    cfg = apply_overrides(Config(), ["model.extra.head.rank=2", "model.extra.scale=0.5"])
    assert cfg.model.extra["head"] == {"rank": 2} and cfg.model.extra["scale"] == 0.5
    with pytest.warns(IgnoredOptionWarning, match="nosuch.a"):
        apply_overrides(cfg, ["nosuch.a=1"])


def test_checkpoint_mismatch_without_old_defaults_raises_as_is():
    """A checkpoint that does not fit, for a model whose defaults never changed: the plain error."""
    from xnn.common.models import ForceStressOutput
    from xnn.common.models.hub.checkpoint import build_potential

    def cfg(n):
        return from_dict({"model": {"name": "tensornet", "cutoff": 4.5, "n_features": n,
                                    "n_interactions": 1, "n_rbf": 8}})

    state = ForceStressOutput(build_model(cfg(8).model)).state_dict()
    with pytest.raises(RuntimeError) as err:
        build_potential(cfg(16), state)
    assert "defaults were different" not in str(err.value)
