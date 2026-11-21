"""CGCNN: the model, its options, the element tables, the checkpoint conversion and training."""
import argparse
import copy
import dataclasses
import itertools
import json
import os
import pickle
import sys

import numpy as np
import pytest
import torch

from xnn.common.config import from_dict
from xnn.common.data import AtomicDataset, collate, nearest_edges, structure_to_graph
from xnn.common.models import build_model
from xnn.gnn.models import CGCNN
from xnn.gnn.models import cgcnn_foundation as cf
from xnn.gnn.models.cgcnn import read_atom_features


def _cell(seed=0, n=6, numbers=(14, 8), a=4.5):
    rng = np.random.default_rng(seed)
    cell = np.eye(3) * a + rng.uniform(-0.3, 0.3, (3, 3))
    pos = rng.uniform(0, 1, (n, 3)) @ cell
    return {"pos": pos, "atomic_numbers": np.resize(numbers, n), "cell": cell, "pbc": [True] * 3}


def _molecule(seed=1):
    rng = np.random.default_rng(seed)
    pos = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [-0.4, 1.0, 0.0], [-0.4, -0.5, 0.9]])
    return {"pos": pos + rng.normal(0, 0.05, pos.shape), "atomic_numbers": [6, 1, 1, 8]}


def _chain(n=8):
    """A bent chain: inner atoms have four neighbors within 2.5 Angstrom, the ends two or three."""
    rng = np.random.default_rng(4)
    pos = np.stack([np.arange(n) * 1.0, 0.1 * (np.arange(n) % 2), np.zeros(n)], axis=1) + rng.normal(0, 0.02, (n, 3))
    return {"pos": pos, "atomic_numbers": np.resize([6, 7, 8], n)}


def _table(rows=20, width=7, seed=3):
    return torch.as_tensor(np.random.default_rng(seed).uniform(0, 1, (rows, width)))


def _small(**kw):
    opts = dict(n_features=8, n_interactions=2, n_rbf=10, cutoff=4.0, max_neighbors=6, readout_sizes=(6, 5),
                atom_features=_table())
    opts.update(kw)
    torch.manual_seed(0)
    model = CGCNN(**opts).double()
    gen = torch.Generator().manual_seed(1)
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.running_mean.copy_(0.3 * torch.randn(m.num_features, generator=gen, dtype=torch.float64))
            m.running_var.copy_(0.5 + torch.rand(m.num_features, generator=gen, dtype=torch.float64))
            with torch.no_grad():
                m.weight.copy_(0.5 + torch.rand(m.num_features, generator=gen, dtype=torch.float64))
                m.bias.copy_(0.1 * torch.randn(m.num_features, generator=gen, dtype=torch.float64))
    return model.eval()


def _graph(s, cutoff, **kw):
    torch.set_default_dtype(torch.float64)
    try:
        return structure_to_graph(s, cutoff, **kw)
    finally:
        torch.set_default_dtype(torch.float32)


def _neighbors(s, cutoff, k):
    """Brute force over periodic images: the ``k`` nearest ``(j, distance)`` of every atom."""
    pos = np.asarray(s["pos"], dtype=float)
    images = [np.zeros(3)]
    if "cell" in s:
        images = [np.array(n) @ np.asarray(s["cell"]) for n in itertools.product(range(-2, 3), repeat=3)]
    out = []
    for i in range(len(pos)):
        found = sorted((float(np.linalg.norm(pos[j] + t - pos[i])), j) for j in range(len(pos)) for t in images)
        found = [(j, d) for d, j in found if 1e-8 < d < cutoff]
        out.append(found if k is None else found[:k])
    return out


def _softplus(x):
    return torch.nn.functional.softplus(x, threshold=40.0)


def _bn(m, x):
    return x if m is None else (x - m.running_mean) / torch.sqrt(m.running_var + m.eps) * m.weight + m.bias


def _reference(model, s):
    """Eq. 5 of the paper with the authors' batch normalizations (evaluation mode), written per atom."""
    nbrs = _neighbors(s, model.cutoff, model.max_neighbors)
    centers, gamma = model.rbf.centers, model.rbf.gamma
    z = torch.as_tensor(np.asarray(s["atomic_numbers"]))
    v = model.embedding(model.atom_features[z])
    f = v.shape[1]
    for conv in model.convs:
        w, b = conv.fc_full.weight, conv.fc_full.bias
        rows = []
        for i, nb in enumerate(nbrs):
            total = torch.zeros(f, dtype=v.dtype)
            for j, d in nb:
                u = torch.exp(-gamma * (d - centers) ** 2)
                h = _bn(conv.bn1, torch.cat([v[i], v[j], u]) @ w.T + b)
                total = total + torch.sigmoid(h[:f]) * _softplus(h[f:])
            rows.append(v[i] + _bn(conv.bn2, total))
        v = torch.stack(rows)
        if conv.residual_activation:
            v = _softplus(v)
    h = _softplus(model.conv_to_fc(_softplus(v.mean(0))))
    for fc in model.fcs:
        h = _softplus(fc(h))
    return model.property_scale * model.fc_out(h) + model.property_shift


@pytest.mark.parametrize("paper", [False, True])
@pytest.mark.parametrize("which", ["cell", "molecule", "chain"])
def test_matches_eq5_written_per_atom(paper, which):
    """Eq. 5 per atom: a crystal (more than k neighbors each), a molecule (fewer, all equal) and a chain (more and
    fewer than k)."""
    opts = dict(normalization=None, residual_activation=False) if paper else {}
    if which == "chain":
        opts.update(cutoff=2.5, max_neighbors=3)
    model = _small(property_scale=1.7, property_shift=-0.4, **opts)
    s = {"cell": _cell, "molecule": _molecule, "chain": _chain}[which]()
    with torch.no_grad():
        mine = model(_graph(s, model.cutoff))["properties"]
        ref = _reference(model, s)
    assert torch.allclose(mine, ref.reshape(mine.shape), rtol=1e-12, atol=1e-12)


def test_batch_permutation_and_all_neighbors():
    """One value per crystal: alone and batched, in any atom order, on a graph of the 6 nearest neighbors built
    beforehand; max_neighbors=None keeps every neighbor within the cutoff."""
    model = _small()
    graphs = [_graph(_cell(), 4.0), _graph(_molecule(), 4.0), _graph(_cell(2, 5), 4.0)]
    with torch.no_grad():
        alone = torch.stack([model(g)["properties"] for g in graphs]).reshape(3)
        out = model(collate(graphs))
    assert out["properties"].shape == (3,) and out["structure_features"].shape == (3, 8)
    assert torch.allclose(out["properties"], alone, rtol=1e-12, atol=1e-13)
    s = _cell()
    perm = np.random.default_rng(3).permutation(len(s["pos"]))
    shuffled = {**s, "pos": s["pos"][perm], "atomic_numbers": np.asarray(s["atomic_numbers"])[perm]}
    with torch.no_grad():
        assert torch.allclose(model(_graph(shuffled, 4.0))["properties"], alone[0], rtol=1e-12)
        assert torch.allclose(model(_graph(s, 4.0, max_neighbors=6))["properties"], alone[0], rtol=1e-12)
        model.max_neighbors = None
        assert torch.allclose(model(graphs[0])["properties"], _reference(model, s).reshape(1), rtol=1e-12)


class _Subclassed(torch.nn.Linear):
    """A dense layer of another type: the convolution concatenates its input instead of splitting it."""


def test_split_and_concatenated_first_layer_agree():
    """The first dense layer split by input (on a CPU) and applied to the concatenation agree, with their gradients."""
    model = _small()
    g = _graph(_cell(), 4.0)
    params = dict(model.named_parameters())
    y = model(g)["properties"].sum()
    ref = dict(zip(params, torch.autograd.grad(y, list(params.values()))))
    for conv in model.convs:
        layer = _Subclassed(conv.fc_full.in_features, conv.fc_full.out_features).double()
        layer.load_state_dict(conv.fc_full.state_dict())
        conv.fc_full = layer
    params = dict(model.named_parameters())
    y2 = model(g)["properties"].sum()
    grads = dict(zip(params, torch.autograd.grad(y2, list(params.values()))))
    assert torch.allclose(y2, y, rtol=1e-12)
    assert all(torch.allclose(grads[k], ref[k], rtol=1e-10, atol=1e-12) for k in ref)


def _fast_pair(**kw):
    """A model on the reference and its copy on the fast path (as written on a CPU)."""
    ref = _small(**kw)
    fast = copy.deepcopy(ref)
    return ref.set_use_fast(False), fast.set_use_fast(True)


def _outputs_and_derivatives(model, batch, train, positions=True):
    """Properties, the gradients of a squared loss by the positions (with ``positions``) and the weights, a second
    derivative (through the first gradient) and the buffers after the evaluation."""
    model.train(train)
    pos = batch.pos.clone().requires_grad_(positions)
    y = model(dataclasses.replace(batch, pos=pos))["properties"]
    params = list(model.parameters())
    grads = torch.autograd.grad((y ** 2).sum(), [pos, *params] if positions else params, create_graph=True)
    second = torch.autograd.grad(grads[0].square().sum(), params[0])[0]
    return [y.detach(), *[g.detach() for g in grads], second, *[b.clone() for b in model.buffers()]]


@pytest.mark.parametrize("paper", [False, True])
def test_fast_path_matches_the_reference(paper):
    """The fast stage gives the reference's properties, position and weight gradients, second derivatives and
    running statistics, in evaluation and training mode, for a batch of cells, one cell and a molecule."""
    ref, fast = _fast_pair(**(dict(normalization=None, residual_activation=False) if paper else {}))
    batches = [collate([_graph(_cell(), 4.0), _graph(_cell(4, 7), 4.0)]), _graph(_cell(4, 7), 4.0),
               _graph(_molecule(), 4.0)]
    for batch in batches:
        for train, positions in ((False, True), (True, True), (True, False), (False, False)):
            mine = _outputs_and_derivatives(fast, batch, train, positions)
            theirs = _outputs_and_derivatives(ref, batch, train, positions)
            assert fast.fast_active and not ref.fast_active
            for a, b in zip(mine, theirs):
                assert torch.allclose(a.double(), b.double(), rtol=1e-10, atol=1e-12)


def test_fast_path_falls_back_on_irregular_graphs_wrapped_layers_cell_gradients_and_auto_on_a_cpu(monkeypatch):
    """Atoms with different neighbor counts, a wrapped dense layer and a cell that needs a gradient take the
    reference; "auto" never takes the stage on a CPU."""
    import xnn.gnn.fast.cgcnn as fc
    ref, fast = _fast_pair(cutoff=2.5, max_neighbors=3)
    chain = _graph(_chain(), 2.5)
    with torch.no_grad():
        assert torch.allclose(fast(chain)["properties"], ref(chain)["properties"], rtol=1e-12)
    ref, fast = _fast_pair()
    g = _graph(_cell(), 4.0)
    cell = g.cell.clone().requires_grad_(True)
    monkeypatch.setattr(fc, "stage", lambda *a: pytest.fail("the stage was taken"))
    out = fast(dataclasses.replace(g, cell=cell))["properties"]
    assert torch.allclose(out, ref(dataclasses.replace(g, cell=cell))["properties"], rtol=1e-12)
    monkeypatch.undo()
    layer = fast.convs[1].fc_full
    wrapped = _Subclassed(layer.in_features, layer.out_features).double()
    wrapped.load_state_dict(layer.state_dict())
    fast.convs[1].fc_full = wrapped
    with torch.no_grad():
        out = fast(g)["properties"]
    assert not fast.fast_active
    assert torch.allclose(out, ref(g)["properties"], rtol=1e-12)
    fast.convs[1].fc_full = layer
    fast.set_use_fast("auto")
    with torch.no_grad():
        fast(g)
    assert not fast.fast_active
    with pytest.raises(ValueError):
        fast.set_use_fast("always")


def test_stages_compile_once_per_layout(monkeypatch):
    """On a GPU the stage is compiled once per layout, mode and gradient need (compilers stubbed here)."""
    import xnn.gnn.fast.cgcnn as fc
    calls = []
    monkeypatch.setattr(fc, "_compiled", {})
    monkeypatch.setattr(fc, "compile_differentiable", lambda fn, n, m: calls.append(("first", n, m)) or fn)
    monkeypatch.setattr(fc.torch, "compile", lambda fn, dynamic: calls.append(("once", dynamic)) or fn)

    class Cuda:
        is_cuda = True
        dtype = torch.float64

    train = (3, True, True, True, 25.0, 1e-5, False, True, False)
    a = fc.stage(train, 384, 20, 9, Cuda(), True)
    b = fc.stage(train, 6000, 20, 9, Cuda(), True)
    fc.stage((3, True, False, True, 25.0, 1e-5, False, True, True), 384, 20, 21, Cuda(), False)
    c = fc.stage(train, 10 ** 5, 20, 9, Cuda(), True)
    assert a is b and c is not a and calls == [("first", 20, 9), ("once", True), ("first", 20, 9)]
    assert [fc.size_class(e) for e in (100, 8191, 8192, 65535, 65536)] == [0, 0, 1, 1, 2]


def test_mark_sizes_gives_the_hints_of_the_size_class():
    """The compiler is given the representative sizes of the class of a graph; sizes 0 and 1 are left alone."""
    import xnn.gnn.fast.cgcnn as fc
    edges, atoms = torch.zeros(10 ** 5, dtype=torch.long), torch.zeros(10 ** 4, dtype=torch.long)
    fc.mark_sizes([edges], [atoms])
    hints = getattr(edges, "_dynamo_hint_overrides", None)
    if hints is not None:       # a torch with size hints
        assert hints == {0: (2 ** 18 // 10) * 10} and atoms._dynamo_hint_overrides == {0: 2 ** 18 // 10}
    one = torch.zeros(1, dtype=torch.long)
    fc.mark_sizes([one], [one])
    assert getattr(one, "_dynamo_hint_overrides", None) is None


def test_mark_sizes_without_size_hints(monkeypatch):
    """A torch whose mark_dynamic takes no size hint leaves the sizes unmarked."""
    import xnn.gnn.fast.cgcnn as fc

    def no_hints(t, dim):
        raise AssertionError("not called without a hint")
    monkeypatch.setattr(torch._dynamo, "mark_dynamic", no_hints)
    edges, atoms = torch.zeros(120, dtype=torch.long), torch.zeros(10, dtype=torch.long)
    fc.mark_sizes([edges], [atoms])
    assert getattr(edges, "_dynamo_hint_overrides", None) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_compiled_fast_path_on_a_gpu(dtype):
    """The compiled stages agree with the reference on a GPU, with their derivatives, in both modes."""
    ref, fast = _fast_pair()
    ref, fast = ref.to("cuda", dtype), fast.to("cuda", dtype)
    batch = collate([_graph(_cell(), 4.0), _graph(_cell(4, 7), 4.0)]).to("cuda")
    batch = dataclasses.replace(batch, pos=batch.pos.to(dtype), cell=batch.cell.to(dtype))
    tol = 1e-10 if dtype == torch.float64 else 2e-4
    for train, positions in ((False, True), (True, True), (True, False)):
        mine = _outputs_and_derivatives(fast, batch, train, positions)
        theirs = _outputs_and_derivatives(ref, batch, train, positions)
        assert fast.fast_active
        for a, b in zip(mine, theirs):
            scale = float(b.abs().max()) if b.numel() and b.is_floating_point() else 1.0
            assert float((a.double() - b.double()).abs().max()) <= tol * max(scale, 1.0)


def test_nearest_edges_keeps_the_k_shortest_per_atom():
    """The kept edges are the k shortest of every receiver, by receiver then length, with the common count when every
    atom has the same; equal counts within k are grouped by receiver, unequal ones kept as they are."""
    g = _graph(_cell(5, 8), 5.0)
    src, dst = g.edge_index
    dist = g.edge_vectors().norm(dim=-1)
    keep, per_atom = nearest_edges(dist, dst, g.num_nodes, 5)
    assert per_atom == 5 and torch.equal(dst[keep], torch.arange(8).repeat_interleave(5))
    for i in range(8):
        mine = dist[keep][dst[keep] == i]
        assert torch.equal(mine, torch.sort(dist[dst == i]).values[:5])
    shuffled = keep[torch.randperm(keep.numel(), generator=torch.Generator().manual_seed(0))]
    again, per_atom = nearest_edges(dist[shuffled], dst[shuffled], g.num_nodes, 7)
    assert per_atom == 5 and torch.equal(dst[shuffled][again], torch.arange(8).repeat_interleave(5))
    assert nearest_edges(dist[keep], dst[keep], g.num_nodes, 7) == (None, 5)        # already grouped: kept as is
    assert nearest_edges(dist, dst, g.num_nodes, 10 ** 6) == (None, None)
    assert nearest_edges(dist[:0], dst[:0], g.num_nodes, 5) == (None, None)
    chain = _graph(_chain(), 2.5)
    keep, per_atom = nearest_edges(chain.edge_vectors().norm(dim=-1), chain.edge_index[1], chain.num_nodes, 3)
    assert per_atom is None and keep.numel() == 2 * 2 + 2 * 3 + 4 * 3


def test_classification_and_dropout():
    """A classifier returns two log-probabilities and the second class's probability; dropout defaults to
    0.5 for a classifier only."""
    model = _small(classification=True)
    assert model.dropout.p == 0.5 and _small().dropout is None and _small(dropout=0.2).dropout.p == 0.2
    batch = collate([_graph(_cell(), 4.0), _graph(_cell(1), 4.0)])
    with torch.no_grad():
        out = model(batch)
    assert out["log_probabilities"].shape == (2, 2)
    assert torch.allclose(out["log_probabilities"].exp().sum(-1), torch.ones(2, dtype=torch.float64))
    assert torch.allclose(out["properties"], out["log_probabilities"][:, 1].exp())


def test_several_outputs_scale_shift_and_constructor_checks():
    """Several properties per crystal, the standardization, and the refused options."""
    model = _small(n_outputs=3)
    model.set_property_scale_shift(2.0, 1.0)
    with torch.no_grad():
        out = model(collate([_graph(_cell(), 4.0)] * 2))["properties"]
    assert out.shape == (2, 3)
    with pytest.raises(ValueError, match="hidden layer"):
        CGCNN(readout_sizes=(), atom_features="onehot")
    with pytest.raises(ValueError, match="normalization"):
        CGCNN(normalization="layer", atom_features="onehot")


def test_defaults_and_exact_float64_centers():
    """The authors' grid: 41 Gaussians on [0, 8] with exponent 25; centers exact in float64 for any start."""
    model = CGCNN(atom_features="onehot")
    assert model.cutoff == 8.0 and model.max_neighbors == 12 and len(model.convs) == 3
    assert model.rbf.centers.numel() == 41 and abs(model.rbf.gamma - 25.0) < 1e-12
    assert model.embedding.weight.shape == (64, 101) and model.conv_to_fc.out_features == 128
    shifted = CGCNN(rbf_start=0.7, rbf_max=5.2, n_rbf=10, atom_features="onehot").double()
    assert torch.equal(shifted.rbf.centers, torch.linspace(0.7, 5.2, 10, dtype=torch.float64))
    assert CGCNN(n_rbf=1, atom_features="onehot").rbf.gamma == 1.0


def test_element_tables(tmp_path):
    """Element tables from a name, a file in the authors' format, a mapping or an array."""
    assert torch.equal(read_atom_features("onehot", 5), torch.eye(5, dtype=torch.float64))
    path = tmp_path / "atom_init.json"
    path.write_text(json.dumps({"1": [0, 1, 0], "8": [1, 0, 1]}))
    table = read_atom_features(path)
    assert table.shape == (9, 3) and table[8].tolist() == [1, 0, 1] and not table[2].any()
    assert torch.equal(read_atom_features(str(path)), table)
    assert torch.equal(read_atom_features(table.numpy()), table)
    with pytest.raises(ValueError, match="different lengths"):
        read_atom_features({1: [0, 1], 2: [1]})
    with pytest.raises(ValueError, match="two dimensions"):
        read_atom_features([1.0, 2.0])
    model = CGCNN(atom_features=str(path), n_features=4)
    assert model.embedding.in_features == 3 and torch.equal(model.atom_features, table.float())


def test_the_authors_table_is_fetched_when_first_needed(monkeypatch):
    """The default table is fetched at the first evaluation or save, never when a checkpoint brings it."""
    calls = []
    table = torch.arange(101 * 92, dtype=torch.float64).reshape(101, 92) % 3

    def fake(*args, **kwargs):
        calls.append(1)
        return table
    monkeypatch.setattr(cf, "atom_init_table", fake)
    model = CGCNN(n_features=4, n_interactions=1, readout_sizes=(3,))
    assert not model.atom_features.any() and not calls
    state = model.state_dict()
    assert len(calls) == 1 and torch.equal(state["atom_features"], table.float())
    loaded = CGCNN(n_features=4, n_interactions=1, readout_sizes=(3,))
    loaded.load_state_dict(state)
    loaded.eval()
    with torch.no_grad():
        loaded(structure_to_graph(_cell(), 8.0))
    assert len(calls) == 1 and torch.equal(loaded.atom_features, state["atom_features"])
    fresh = CGCNN(n_features=4, n_interactions=1, readout_sizes=(3,)).eval()
    with torch.no_grad():
        fresh(structure_to_graph(_cell(), 8.0))
    assert len(calls) == 2


def test_config_with_the_authors_spellings_and_the_shipped_yaml():
    """The authors' argument spellings translate; the shipped config builds the default model."""
    import yaml
    cfg = from_dict({"model": {"name": "cgcnn", "atom_fea_len": 16, "n_conv": 2, "radius": 6.0, "max_num_nbr": 8,
                               "dmin": 0.5, "n_rbf": 12, "atom_features": "onehot", "readout_sizes": [32, 16]}})
    model = build_model(cfg.model)
    assert isinstance(model, CGCNN) and len(model.convs) == 2 and model.cutoff == 6.0
    assert model.max_neighbors == 8 and float(model.rbf.centers[0]) == 0.5 and len(model.fcs) == 1
    assert model.embedding.out_features == 16
    path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "model", "cgcnn.yaml")
    with open(path) as f:
        shipped = build_model(from_dict({"model": yaml.safe_load(f)}).model)
    ref = CGCNN()
    assert sum(p.numel() for p in shipped.parameters()) == sum(p.numel() for p in ref.parameters())
    paper = build_model(from_dict({"model": {"name": "cgcnn", "normalization": None, "residual_activation": False,
                                             "max_neighbors": None, "atom_features": "onehot"}}).model)
    assert paper.convs[0].bn1 is None and not paper.convs[0].residual_activation and paper.max_neighbors is None


def _checkpoint(path, model, classification=False, mean=-1.2, std=0.8):
    """Write ``model`` in the authors' checkpoint layout."""
    state = {k: v.float() for k, v in model.state_dict().items()
             if k not in ("atom_features", "rbf.centers", "property_scale", "property_shift")}
    args = {"task": "classification" if classification else "regression", "atom_fea_len": 8, "h_fea_len": 6,
            "n_h": 2, "n_conv": 2, "data_options": ["data"]}
    torch.save({"epoch": 3, "state_dict": state, "best_mae_error": np.float64(0.1), "optimizer": {},
                "normalizer": {"mean": torch.tensor(mean), "std": torch.tensor(std)}, "args": args},
               path, pickle_protocol=2, _use_new_zipfile_serialization=False)
    return path


@pytest.mark.parametrize("classification", [False, True])
def test_checkpoints_convert_layer_for_layer(tmp_path, classification):
    """A checkpoint in the authors' layout converts to the same predictions, standardization included."""
    table = _table(101, 92)
    src = CGCNN(n_features=8, n_interactions=2, n_rbf=41, readout_sizes=(6, 6), classification=classification,
                atom_features=table).float().eval()
    path = _checkpoint(tmp_path / "m.pth.tar", src, classification)
    model = cf.convert_checkpoint(path, torch.float64, atom_features=table)
    assert model.atom_features.dtype == torch.float64 and not model.training
    batch = collate([_graph(_cell(), 8.0), _graph(_cell(4, 7), 8.0)])
    with torch.no_grad():
        mine = model(batch)["properties"]
        ref = src.double()(batch)["properties"]
    if not classification:
        ref = 0.8 * ref - 1.2
    assert torch.allclose(mine, ref.to(mine.dtype), rtol=1e-6)
    bad = torch.load(path, weights_only=False)
    bad["state_dict"]["extra.weight"] = torch.zeros(1)
    torch.save(bad, tmp_path / "bad.pth.tar")
    with pytest.raises(ValueError, match="does not map"):
        cf.convert_checkpoint(tmp_path / "bad.pth.tar", atom_features=table)
    torch.save({"args": argparse.Namespace(task="regression")}, tmp_path / "pickled.pth.tar")
    with pytest.raises(pickle.UnpicklingError, match="argparse.Namespace"):
        cf.convert_checkpoint(tmp_path / "pickled.pth.tar", atom_features=table)


def test_from_foundation_reads_the_cache(tmp_path, monkeypatch):
    """A registered checkpoint and the element table come from the cache, with the license note beside them."""
    import xnn.common.data.hub._download as download
    from xnn.common.models.hub.provenance import ParameterLicenseWarning
    folder = tmp_path / "models" / "cgcnn" / cf._REVISION
    folder.mkdir(parents=True)
    table = _table(101, 92)
    (folder / "atom_init.json").write_text(json.dumps({str(z): table[z].tolist() for z in range(1, 101)}))
    src = CGCNN(n_features=8, n_interactions=2, readout_sizes=(6, 6), atom_features=table).eval()
    _checkpoint(folder / "band-gap.pth.tar", src)
    monkeypatch.setattr(download, "download_file", lambda url, dest, md5=None, **kw: dest)
    with pytest.warns(ParameterLicenseWarning, match="MIT"):
        model = CGCNN.from_foundation("cgcnn-band-gap", dtype="float64", cache_dir=tmp_path / "models")
    assert (folder / "LICENSE.txt").exists() and model.atom_features.dtype == torch.float64
    assert torch.allclose(model.atom_features[1:], table[1:]) and not model.atom_features[0].any()
    with pytest.raises(KeyError, match="unknown CGCNN"):
        CGCNN.from_foundation("cgcnn-nothing")
    (folder / "atom_init.json").write_text(json.dumps({"1": [0.0, 1.0]}))
    with pytest.raises(ValueError, match="shape"):
        cf.atom_init_table(tmp_path / "models")


def test_gradients_flow_to_positions_and_weights():
    """The property is differentiable by the positions (through the bond lengths) and by every weight."""
    model = _small()
    g = _graph(_cell(), 4.0)
    g.pos.requires_grad_(True)
    y = model(g)["properties"].sum()
    grads = torch.autograd.grad(y, [g.pos, *model.parameters()], allow_unused=True)
    assert grads[0].abs().max() > 0 and all(gr is not None for gr in grads[1:])


def test_training_on_property_labels(tmp_path):
    """The trainer fits crystal properties with batch normalization in training mode."""
    from xnn.common.config import Config
    from xnn.common.train import Trainer
    cfg = Config()
    cfg.model = from_dict({"model": {"name": "cgcnn", "n_features": 8, "n_interactions": 2, "n_rbf": 10,
                                     "cutoff": 4.0, "max_neighbors": 6, "readout_sizes": [6],
                                     "atom_features": "onehot"}}).model
    cfg.optim.epochs = 2
    cfg.optim.property_weight = 1.0
    cfg.data.batch_size = 4
    cfg.data.val_fraction = 0.25
    cfg.output_dir = str(tmp_path)
    structures = []
    for k in range(8):
        s = _cell(k, 4 + k % 3)
        s.update(properties=float(k % 3))
        structures.append(s)
    dataset = AtomicDataset(structures, cfg.model.cutoff)
    trainer = Trainer(cfg, dataset)
    assert not trainer.module.compute_forces and dataset.max_neighbors == 6      # chosen once, in the graphs
    metrics = trainer.fit()
    assert np.isfinite(metrics["val"]["loss"]) and "properties_mae" in metrics["val"]
    assert int(trainer.module.model.convs[0].bn1.num_batches_tracked) > 0


def _cgcnn_fidelity(dtype):
    path = os.environ.get("CGCNN_UPSTREAM_PATH")
    if not path:
        pytest.skip("set CGCNN_UPSTREAM_PATH to a checkout of txie-93/cgcnn")
    pytest.importorskip("pymatgen")
    import subprocess
    script = os.path.join(os.path.dirname(__file__), "cgcnn_fidelity.py")
    res = subprocess.run([sys.executable, script, path, dtype], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    return float(res.stdout.strip().splitlines()[-1])


def test_fidelity_vs_cgcnn_float64():
    assert _cgcnn_fidelity("float64") < 1e-12


def test_fidelity_vs_cgcnn_float32():
    assert _cgcnn_fidelity("float32") < 1e-5
