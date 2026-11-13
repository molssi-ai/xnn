"""MEGNet and multi-fidelity MEGNet: the model, its options, the MatGL conversion and training."""
import json
import os
import pickle
import sys
import types

import numpy as np
import pytest
import torch

from xnn.common.config import from_dict
from xnn.common.data import AtomicDataset, collate, structure_to_graph
from xnn.common.models import build_model
from xnn.gnn.models import MEGNet

MATGL_SPECIES = [z for z in range(1, 95) if not 84 <= z <= 88]


def _cell(seed=0, n=6, numbers=(14, 8), a=4.5):
    rng = np.random.default_rng(seed)
    cell = np.eye(3) * a + rng.uniform(-0.3, 0.3, (3, 3))
    pos = rng.uniform(0, 1, (n, 3)) @ cell
    return {"pos": pos, "atomic_numbers": np.resize(numbers, n), "cell": cell, "pbc": [True] * 3}


def _molecule(seed=1):
    rng = np.random.default_rng(seed)
    pos = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [-0.4, 1.0, 0.0], [-0.4, -0.5, 0.9]])
    return {"pos": pos + rng.normal(0, 0.05, pos.shape), "atomic_numbers": [6, 1, 1, 8]}


def _small(**kw):
    opts = dict(n_features=8, n_interactions=2, n_rbf=12, encoder_sizes=(10, 6), conv_sizes=(8, 6),
                readout_sizes=(5,), state_dim=3)
    opts.update(kw)
    torch.manual_seed(0)
    return MEGNet(**opts).double()


def _graph(s, model, **extra):
    torch.set_default_dtype(torch.float64)
    try:
        return structure_to_graph({**s, **extra}, model.cutoff)
    finally:
        torch.set_default_dtype(torch.float32)


def test_forward_batch_and_permutation():
    """One value per structure, the same alone and in a batch, invariant to the order of the atoms."""
    model = _small()
    graphs = [_graph(_cell(), model), _graph(_molecule(), model), _graph(_cell(2, 5), model)]
    alone = torch.stack([model(g)["properties"] for g in graphs])
    out = model(collate(graphs))
    assert out["properties"].shape == (3,) and out["structure_features"].shape == (3, 30)
    assert torch.allclose(out["properties"], alone.reshape(3), rtol=1e-12, atol=1e-13)
    s = _cell()
    perm = np.random.default_rng(3).permutation(len(s["pos"]))
    shuffled = {**s, "pos": s["pos"][perm], "atomic_numbers": np.asarray(s["atomic_numbers"])[perm]}
    assert torch.allclose(model(_graph(shuffled, model))["properties"], alone[0], rtol=1e-12)


def test_float_state_features():
    """Float state features enter the network; a missing state is the zero vector."""
    model = _small()
    s = _cell()
    zero = model(_graph(s, model))["properties"]
    assert torch.allclose(model(_graph(s, model, state=[0.0, 0.0, 0.0]))["properties"], zero)
    hot = model(_graph(s, model, state=[300.0, 1.0, 0.0]))["properties"]
    assert not torch.allclose(hot, zero)
    batch = collate([_graph(s, model), _graph(s, model, state=[300.0, 1.0, 0.0])])
    assert batch.state.shape == (2, 3)
    assert torch.allclose(model(batch)["properties"], torch.cat([zero, hot]))


def test_multi_fidelity_labels_and_names():
    """Integer labels select the fidelity embedding; a missing label is fidelity 0; names map to labels."""
    model = _small(n_state_types=3, state_names=["PBE", "HSE", "experiment"])
    s = _cell()
    g = [_graph(s, model, state=k) for k in range(3)]
    assert g[1].state.dtype == torch.long and g[1].state.shape == (1,)
    values = torch.cat([model(x)["properties"] for x in g])
    assert len(set(values.tolist())) == 3
    assert torch.equal(model(_graph(s, model))["properties"], values[:1])
    assert torch.allclose(model(collate(g))["properties"], values)
    assert model.state_label("HSE") == 1
    with pytest.raises(ValueError, match="unknown state"):
        model.state_label("SCAN")


def test_outputs_scale_shift_classification_and_several_targets():
    """Standardization, the sigmoid of a classifier and several outputs per structure."""
    model = _small(n_outputs=2)
    g = collate([_graph(_cell(), model), _graph(_molecule(), model)])
    y = model(g)["properties"]
    assert y.shape == (2, 2)
    model.set_property_scale_shift(2.0, -1.0)
    assert torch.allclose(model(g)["properties"], 2.0 * y - 1.0)
    clf = _small(classification=True, dropout=0.1).eval()
    p = clf(g)["properties"]
    assert ((p > 0) & (p < 1)).all()


def test_constructor_checks():
    """The residual widths and the state names are checked."""
    with pytest.raises(ValueError, match="residual"):
        MEGNet(encoder_sizes=(64, 16), conv_sizes=(64, 32))
    with pytest.raises(ValueError, match="state names"):
        MEGNet(n_state_types=2, state_names=["PBE"])


def test_float64_gaussian_centers_are_exact():
    """In float64 the Gaussian centers are the exact grid on [0, cutoff + 1]."""
    model = MEGNet().double()
    assert torch.equal(model.rbf.centers, torch.linspace(0.0, 5.0, 100, dtype=torch.float64))


def test_config_with_matgl_spellings_and_the_shipped_yaml():
    """MatGL's constructor spellings translate to the canonical options; the shipped config builds."""
    import yaml
    cfg = from_dict({"model": {"name": "megnet", "dim_node_embedding": 8, "nblocks": 2, "dim_edge_embedding": 20,
                               "cutoff": 5.0, "ntypes_state": 4, "dim_state_embedding": 6,
                               "niters_set2set": 3, "gauss_width": 4.0}})
    model = build_model(cfg.model)
    assert isinstance(model, MEGNet) and len(model.blocks) == 2 and model.n_state_types == 4
    assert model.node_set2set.steps == 3 and model.rbf.gamma == 4.0 and model.cutoff == 5.0
    assert model.state_embedding.weight.shape == (4, 6)
    path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "model", "megnet.yaml")
    with open(path) as f:
        shipped = build_model(from_dict({"model": yaml.safe_load(f)}).model)
    ref = MEGNet()
    assert sum(p.numel() for p in shipped.parameters()) == sum(p.numel() for p in ref.parameters())


def _normalizer_module(monkeypatch, names=("Normalizer",)):
    """A stand-in ``matgl.data.transformer`` module, so that ``model.pt`` pickles MatGL's class names."""
    mod = types.ModuleType("matgl.data.transformer")
    for name in names:
        setattr(mod, name, type(name, (), {"__module__": "matgl.data.transformer"}))
    for full in ("matgl", "matgl.data"):
        monkeypatch.setitem(sys.modules, full, types.ModuleType(full))
    monkeypatch.setitem(sys.modules, "matgl.data.transformer", mod)
    return mod


def _matgl_directory(path, model, monkeypatch, mean=1.5, std=2.0, transformer="Normalizer", **init):
    """Write ``model`` as a MatGL MEGNet directory with MatGL's layer names and layout."""
    path.mkdir(parents=True, exist_ok=True)
    own = model.state_dict()
    out = {"bond_expansion.rbf.centers": own["rbf.centers"].float(),
           "embedding.layer_node_embedding.weight": own["node_embedding.weight"][MATGL_SPECIES]}
    if model.state_embedding is not None:
        out["embedding.layer_state_embedding.weight"] = own["state_embedding.weight"]
    renames = [("node_set2set.", "node_s2s."), ("edge_set2set.", "edge_s2s."), ("readout.", "output_proj.")]
    for name, value in own.items():
        parts = name.split(".")
        if "layers" in parts and not name.startswith(("node_set2set", "edge_set2set")):
            i = parts.index("layers")
            parts[i + 1] = str(2 * int(parts[i + 1]))
        theirs = ".".join(parts)
        for mine, upstream in renames:
            theirs = theirs.replace(mine, upstream, 1) if theirs.startswith(mine) else theirs
        if theirs.startswith("blocks."):
            for kind in ("edge", "node", "state"):
                theirs = theirs.replace(f".{kind}_mlp.", f".conv.{kind}_func.").replace(f".{kind}_dense.",
                                                                                    f".{kind}_func.")
        if theirs.startswith(("blocks.", "edge_encoder.", "node_encoder.", "state_encoder.", "node_s2s.",
                              "edge_s2s.", "output_proj.")):
            out[theirs] = value
    torch.save({"model." + k: v.float() for k, v in out.items()}, path / "state.pt")
    args = {"dim_node_embedding": model.node_embedding.weight.shape[1], "dim_edge_embedding": model.rbf.centers.numel(),
            "dim_state_embedding": model.state_dim, "ntypes_state": model.n_state_types,
            "nblocks": len(model.blocks), "hidden_layer_sizes_input": [10, 6], "hidden_layer_sizes_conv": [8, 6],
            "hidden_layer_sizes_output": [5], "nlayers_set2set": 1, "niters_set2set": 2,
            "activation_type": "softplus2", "is_classification": False, "include_state": True,
            "dropout": None, "graph_transformations": None, "element_types": None, "bond_expansion": None,
            "cutoff": model.cutoff, "gauss_width": 0.5}
    args.update(init)
    meta = {"@class": "TransformedTargetModel", "kwargs": {"model": {"@class": "MEGNet", "init_args": args},
                                                           "target_transformer": "Normalizer"}}
    (path / "model.json").write_text(json.dumps(meta))
    mod = _normalizer_module(monkeypatch, ("Normalizer", "LogTransformer"))
    t = getattr(mod, transformer)()
    t.mean, t.std = mean, std
    torch.save({"model": {"@class": "MEGNet"}, "target_transformer": t}, path / "model.pt")
    for name in ("matgl", "matgl.data", "matgl.data.transformer"):
        monkeypatch.delitem(sys.modules, name)
    return path


def test_matgl_directory_converts_and_loads_through_the_hub(tmp_path, monkeypatch):
    """A MatGL MEGNet directory converts weight for weight with its standardization, and loads by the hub."""
    from xnn.common.models import from_pretrained
    from xnn.common.models.hub.formats import detect_format
    from xnn.gnn.models import matgl_format
    from xnn.gnn.models.megnet_foundation import convert_directory
    src = _small(n_state_types=4).float()
    folder = _matgl_directory(tmp_path / "ckpt", src, monkeypatch)
    assert matgl_format.matgl_class(folder) == "MEGNet" and detect_format(folder) == "matgl"
    model, cfg = convert_directory(folder / "state.pt", torch.float64, ["PBE", "GLLB-SC", "HSE", "SCAN"])
    assert cfg.model.extra["property_scale"] == 2.0 and cfg.model.extra["n_state_types"] == 4
    assert model.state_label("SCAN") == 3
    s = _cell(numbers=(31, 7))
    for label in range(4):
        g = _graph(s, model, state=label)
        expect = 2.0 * src.double()(g)["properties"] + 1.5
        assert torch.allclose(model(g)["properties"], expect, rtol=1e-12)
    hub = from_pretrained(folder, wrap=False, cache_dir=tmp_path / "cache")
    assert isinstance(hub, MEGNet)
    assert torch.equal(hub.node_embedding.weight[MATGL_SPECIES].double(),
                       model.node_embedding.weight[MATGL_SPECIES].float().double())


def test_matgl_refusals(tmp_path, monkeypatch):
    """Architectures and transformers outside the converted ones are refused."""
    from xnn.gnn.models import matgl_format
    from xnn.gnn.models.megnet_foundation import convert_directory, from_foundation
    src = _small().float()
    with pytest.raises(NotImplementedError, match="bond expansion"):
        convert_directory(_matgl_directory(tmp_path / "a", src, monkeypatch, bond_expansion={"@class": "B"}))
    with pytest.raises(NotImplementedError, match="LogTransformer"):
        convert_directory(_matgl_directory(tmp_path / "b", src, monkeypatch, transformer="LogTransformer"))
    folder = _matgl_directory(tmp_path / "c", src, monkeypatch)
    meta = json.loads((folder / "model.json").read_text())
    meta["kwargs"]["model"]["@class"] = "M3GNet"
    (folder / "model.json").write_text(json.dumps(meta))
    with pytest.raises(NotImplementedError, match="not a MEGNet"):
        convert_directory(folder)
    assert not matgl_format.is_matgl_dir(folder)
    with pytest.raises(NotImplementedError, match="not converted"):
        matgl_format.convert_directory(folder)
    from xnn.gnn.models.megnet_foundation import _standardization
    torch.save({"target_transformer": print}, folder / "model.pt")
    with pytest.raises(pickle.UnpicklingError, match="print"):
        _standardization(folder)
    (folder / "model.pt").write_bytes(pickle.dumps({"target_transformer": print}))
    with pytest.raises(Exception, match="print"):
        _standardization(folder)
    with pytest.raises(KeyError):
        from_foundation("megnet-unknown")
    for i, (key, value, match) in enumerate((("graph_transformations", ["t"], "graph transformations"),
                                             ("include_state", False, "without a state"),
                                             ("activation_type", "softexp", "activation"))):
        with pytest.raises(NotImplementedError, match=match):
            convert_directory(_matgl_directory(tmp_path / f"d{i}", src, monkeypatch, **{key: value}))
    folder = _matgl_directory(tmp_path / "e", src, monkeypatch)
    state = torch.load(folder / "state.pt")
    del state["model.output_proj.layers.0.weight"]
    torch.save(state, folder / "state.pt")
    with pytest.raises(KeyError, match="output_proj"):
        convert_directory(folder)


def test_standardization_with_tensor_statistics(tmp_path, monkeypatch):
    """A normalizer whose mean and std are tensors (MatGL's ``from_data``) converts too."""
    from xnn.gnn.models.megnet_foundation import _standardization
    folder = _matgl_directory(tmp_path / "ckpt", _small().float(), monkeypatch,
                              mean=torch.tensor(0.25, dtype=torch.float64), std=torch.tensor(3.0))
    assert _standardization(folder) == (3.0, 0.25)


def test_without_model_pt_the_output_is_unscaled(tmp_path, monkeypatch):
    """A directory without ``model.pt`` (or without a transformer in it) keeps the network's output."""
    from xnn.gnn.models.megnet_foundation import _standardization
    folder = _matgl_directory(tmp_path / "ckpt", _small().float(), monkeypatch)
    torch.save({"model": {}}, folder / "model.pt")
    assert _standardization(folder) == (1.0, 0.0)
    (folder / "model.pt").unlink()
    assert _standardization(folder) == (1.0, 0.0)


def test_from_foundation_reads_the_cache(tmp_path, monkeypatch):
    """A registered model already in the cache converts in the asked dtype, with its fidelity names."""
    import xnn.common.data.hub._download as download
    from xnn.gnn.models import megnet_foundation as mf
    name = "megnet-mp-2019.4.1-bandgap-mfi"
    revision = mf.FOUNDATION_MODELS[name][1]
    folder = tmp_path / "models" / "matgl" / name / revision
    src = _small(n_state_types=4).float()
    _matgl_directory(folder, src, monkeypatch)
    monkeypatch.setattr(download, "download_file", lambda url, dest, md5=None, **kw: dest)
    model = MEGNet.from_foundation(name, dtype="float64", cache_dir=tmp_path / "models")
    assert not model.training and model.node_embedding.weight.dtype == torch.float64
    assert model.state_label("HSE") == 2


@pytest.mark.parametrize("name, n_state_types", [("megnet-mp-2018.6.1-eform", None),
                                                 ("megnet-mp-2019.4.1-bandgap-mfi", 4)])
def test_pretrained_models_get_their_training_bond_direction(tmp_path, monkeypatch, name, n_state_types):
    """MatGL's pretrained checkpoints come with the two atom blocks of every bond update swapped; a plain
    conversion keeps MatGL's layout."""
    import xnn.common.data.hub._download as download
    from xnn.gnn.models import megnet_foundation as mf
    _, revision, _, _, reversed_bonds = mf.FOUNDATION_MODELS[name]
    assert reversed_bonds
    folder = tmp_path / "models" / "matgl" / name / revision
    _matgl_directory(folder, _small(n_state_types=n_state_types).float(), monkeypatch)
    monkeypatch.setattr(download, "download_file", lambda url, dest, md5=None, **kw: dest)
    model = MEGNet.from_foundation(name, dtype="float64", cache_dir=tmp_path / "models")
    plain, _ = mf.convert_directory(folder, torch.float64)
    for block, ref in zip(model.blocks, plain.blocks):
        w, r = block.edge_mlp.layers[0].weight, ref.edge_mlp.layers[0].weight
        n = r.shape[1] // 4
        assert torch.equal(w[:, :n], r[:, n:2 * n]) and torch.equal(w[:, n:2 * n], r[:, :n])
        assert torch.equal(w[:, 2 * n:], r[:, 2 * n:])
    s = _cell()
    assert not torch.equal(model(_graph(s, model))["properties"], plain(_graph(s, plain))["properties"])


def test_training_on_property_labels(tmp_path):
    """The trainer fits properties of mixed fidelities: no forces, a property loss and its MAE."""
    from xnn.common.config import Config
    from xnn.common.train import Trainer
    cfg = Config()
    cfg.model = from_dict({"model": {"name": "megnet", "n_features": 8, "n_interactions": 1, "n_rbf": 12,
                                     "cutoff": 4.0, "n_state_types": 2, "encoder_sizes": [8, 6],
                                     "conv_sizes": [8, 6], "readout_sizes": [4]}}).model
    cfg.optim.epochs = 2
    cfg.optim.property_weight = 1.0
    cfg.data.batch_size = 4
    cfg.data.val_fraction = 0.25
    cfg.output_dir = str(tmp_path)
    structures = []
    for k in range(8):
        s = _cell(k, 4 + k % 3)
        s.update(properties=float(k % 3), state=k % 2)
        structures.append(s)
    trainer = Trainer(cfg, AtomicDataset(structures, cfg.model.cutoff))
    assert not trainer.module.compute_forces
    metrics = trainer.fit()
    assert np.isfinite(metrics["val"]["loss"]) and "properties_mae" in metrics["val"]
    assert "property_mse" in metrics["train"]


def _parity_batch(model, device="cpu", dtype=torch.float64):
    """Two cells, a molecule and an isolated atom (no bonds), with fidelity labels when the model has them."""
    extra = {"state": 1} if model.n_state_types is not None else {}
    lone = {"pos": np.zeros((1, 3)), "atomic_numbers": [26]}
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        graphs = [structure_to_graph({**s, **extra}, model.cutoff) for s in (_cell(), _molecule(), _cell(4, 7), lone)]
    finally:
        torch.set_default_dtype(previous)
    return collate(graphs).to(device)


def _outputs_and_grads(model, batch, use_fast):
    """Properties, their gradients by weights and positions, and a second derivative by the weights."""
    model.set_use_fast(use_fast)
    batch.pos = batch.pos.detach().requires_grad_(True)
    y = model(batch)["properties"]
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad((y * torch.arange(1.0, 1.0 + y.numel(), dtype=y.dtype, device=y.device)).sum(),
                                [batch.pos, *params], create_graph=True, allow_unused=True)
    pos_grad = grads[0]
    second = torch.autograd.grad((pos_grad ** 2).sum(), params, allow_unused=True)
    zero = [torch.zeros_like(p) for p in params]
    return (y.detach(), pos_grad.detach(), [g.detach() if g is not None else z for g, z in zip(grads[1:], zero)],
            [g if g is not None else z for g, z in zip(second, zero)])


def _assert_parity(ref, fast, rtol, atol):
    for a, b in zip(ref[:2], fast[:2]):
        assert torch.allclose(a, b, rtol=rtol, atol=atol), (a, b)
    for group_a, group_b in zip(ref[2:], fast[2:]):
        for a, b in zip(group_a, group_b):
            assert torch.allclose(a, b, rtol=rtol, atol=atol), float((a - b).abs().max())


@pytest.mark.parametrize("n_state_types", [None, 3])
def test_fast_path_matches_the_reference_with_derivatives(n_state_types):
    """The split and staged fast path gives the reference's properties, gradients and second derivatives."""
    model = _small(n_state_types=n_state_types, set2set_layers=2)
    model.train()
    batch = _parity_batch(model)
    ref = _outputs_and_grads(model, batch, False)
    assert not model.blocks[0].fast_active
    fast = _outputs_and_grads(model, batch, True)
    assert model.fast_active and all(b.fast_active for b in model.blocks) and model.node_set2set.fast_active
    _assert_parity(ref, fast, 1e-11, 1e-12)
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(batch)["properties"], ref[0], rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("structure", ["cell", "lone atom"])
def test_fast_path_single_structure_matches_the_reference(structure):
    """One structure per batch (sums as matrix products) gives the reference's values and derivatives."""
    model = _small(n_state_types=2).train()
    s = _cell(3, 9) if structure == "cell" else {"pos": np.zeros((1, 3)), "atomic_numbers": [8]}
    batch = collate([_graph(s, model, state=1)])
    ref = _outputs_and_grads(model, batch, False)
    fast = _outputs_and_grads(model, batch, True)
    _assert_parity(ref, fast, 1e-11, 1e-12)


def test_fast_path_dropout_draws_the_reference_masks():
    """With dropout in training, both paths draw the same masks from the same seed."""
    model = _small(dropout=0.2).train()
    batch = _parity_batch(model)
    out = []
    for use_fast in (False, True):
        model.set_use_fast(use_fast)
        torch.manual_seed(5)
        out.append(model(batch)["properties"])
    assert torch.allclose(out[0], out[1], rtol=1e-11, atol=1e-12)


def test_fast_path_falls_back_for_wrapped_layers_and_auto_takes_it_on_a_cpu():
    """A wrapped dense layer (e.g. LoRA) sends every block to the reference; "auto" takes the fast path on a CPU."""
    class Wrapped(torch.nn.Linear):
        pass

    model = _small()
    batch = _parity_batch(model)
    ref = model.set_use_fast(False)(batch)["properties"]
    layer = model.blocks[1].edge_mlp.layers[0]
    wrapped = Wrapped(layer.in_features, layer.out_features).double()
    wrapped.load_state_dict(layer.state_dict())
    model.blocks[1].edge_mlp.layers[0] = wrapped
    model.__dict__.pop("_fast_blocks", None)
    out = model.set_use_fast(True)(batch)["properties"]
    assert not any(b.fast_active for b in model.blocks) and model.fast_active
    assert torch.allclose(out, ref, rtol=1e-12)
    model.set_use_fast("auto")(batch)
    assert model.fast_active and model.node_set2set.fast_active
    with pytest.raises(ValueError):
        model.set_use_fast("always")


def test_stages_compile_once_per_shape(monkeypatch):
    """On a GPU the stages are compiled once per layer layout and gradient need (compilers stubbed here)."""
    import xnn.gnn.fast.megnet as fm
    calls = []
    monkeypatch.setattr(fm, "_compiled", {})
    monkeypatch.setattr(fm, "compile_twice_differentiable", lambda fn, n, m: calls.append(("twice", n, m)) or fn)
    monkeypatch.setattr(fm.torch, "compile", lambda fn, dynamic: calls.append(("once", dynamic)) or fn)

    class Cuda:
        is_cuda = True
        dtype = torch.float64

    act = torch.nn.SiLU()
    a = fm.stage("block", (0, 2, True), act, 7, 7, Cuda(), True)
    b = fm.stage("block", (0, 2, True), act, 7, 7, Cuda(), True)
    fm.stage("set2set", (2, 1), None, 5, 2, Cuda(), False)
    fm.stage("encode", (2, 2, 2), act, 15, 2, Cuda(), False)
    assert a is b and calls == [("twice", 7, 7), ("once", True), ("once", True)]


def _alloy(x, seed=2):
    """A cell with two sites shared by Ga and Al (fraction ``x`` of Al) and nitrogen on the others."""
    s = _cell(seed, 4, numbers=(31, 7))
    s["occupancies"] = [{"Ga": 1 - x, "Al": x}, {"N": 1.0}, {"Ga": 1 - x, "Al": x}, {"N": 1.0}]
    return s


def test_disordered_sites_mix_element_embeddings():
    """A site's embedding is the occupancy-weighted sum of its species'; occupancy 1 is the ordered site."""
    model = _small(n_state_types=2)
    ordered = _cell(2, 4, numbers=(31, 7))
    pure = {**ordered, "occupancies": [{31: 1.0}, {7: 1.0}, {31: 1.0}, {7: 1.0}]}
    assert torch.equal(model(_graph(pure, model, state=1))["properties"],
                       model(_graph(ordered, model, state=1))["properties"])
    g = _graph(_alloy(0.3), model, state=1)
    w = model.node_embedding.weight
    assert torch.allclose(model._sites(g)[0], 0.7 * w[31] + 0.3 * w[13], rtol=1e-12)
    values = [model(_graph(_alloy(x), model, state=1))["properties"] for x in (0.0, 0.5, 1.0)]
    assert len({round(float(v), 10) for v in values}) == 3
    batch = collate([_graph(ordered, model, state=0), _graph(_alloy(0.5), model, state=1)])
    alone = torch.cat([model(_graph(ordered, model, state=0))["properties"], values[1]])
    assert torch.allclose(model(batch)["properties"], alone, rtol=1e-12)


def test_fast_path_matches_the_reference_on_disordered_sites():
    """The fast path reads the same mixed embeddings: same properties, gradients and second derivatives."""
    model = _small(n_state_types=2).train()
    batch = collate([_graph(_alloy(0.4), model, state=1), _graph(_cell(5, 6), model, state=0)])
    ref = _outputs_and_grads(model, batch, False)
    fast = _outputs_and_grads(model, batch, True)
    _assert_parity(ref, fast, 1e-11, 1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("single", [False, True])
def test_compiled_fast_path_on_a_gpu(dtype, single):
    """On a GPU the compiled stages give the reference's properties, gradients and second derivatives."""
    model = _small(n_state_types=3).to("cuda", dtype).train()
    if single:
        batch = collate([_graph(_cell(3, 9), model, state=1)]).to("cuda")
        batch.pos, batch.cell = batch.pos.to(dtype), batch.cell.to(dtype)
    else:
        batch = _parity_batch(model, "cuda", dtype)
    with torch.backends.cudnn.flags(enabled=False):
        # cuDNN's LSTM has no second derivative; the native one has
        ref = _outputs_and_grads(model, batch, False)
    fast = _outputs_and_grads(model, batch, True)
    tol = (1e-10, 1e-11) if dtype == torch.float64 else (2e-4, 2e-5)
    _assert_parity(ref, fast, *tol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_compiled_stages_are_not_recompiled_per_batch():
    """Training on batches of new sizes reuses the compiled stages (no guard on the edge count)."""
    from torch._dynamo.utils import counters
    model = _small().to("cuda").train().set_use_fast(True)

    def step(k):
        data = collate([_graph(_cell(10 * k + i, 5 + k % 4), model) for i in range(1 + k % 3)]).to("cuda")
        model(data)["properties"].sum().backward()

    for k in range(3):
        step(k)
    before = counters["stats"]["unique_graphs"]
    for k in range(3, 12):
        step(k)
    assert counters["stats"]["unique_graphs"] - before <= 2


def _matgl_fidelity(dtype):
    paths = os.environ.get("MATGL_MEGNET_PATHS")
    if not paths:
        pytest.skip("set MATGL_MEGNET_PATHS to MatGL MEGNet directories (separated by the path separator)")
    pytest.importorskip("matgl")
    import subprocess
    script = os.path.join(os.path.dirname(__file__), "megnet_matgl_fidelity.py")
    res = subprocess.run([sys.executable, script, *paths.split(os.pathsep), dtype], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    return float(res.stdout.strip().splitlines()[-1])


def test_fidelity_vs_matgl_float64():
    assert _matgl_fidelity("float64") < 1e-12


def test_fidelity_vs_matgl_float32():
    assert _matgl_fidelity("float32") < 1e-5
