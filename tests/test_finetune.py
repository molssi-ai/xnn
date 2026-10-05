"""Fine-tuning tools: multi-head potentials, LoRA, reference energies, replay.

Covers the generic machinery on every model family that supports it, the
MACE fidelity of the multi-head routing and of the LoRA composition against
the upstream ``mace-torch`` implementations (when installed), and the
trainer's fine-tuning paths (pretrained sources, estimated reference
energies, replay sets, EMA, freezing).
"""
import copy

import numpy as np
import pytest
import torch

from xnn.common.config import Config, from_dict
from xnn.common.data import AtomicDataset, collate, structure_to_graph
from xnn.common.finetune import (MultiHead, LoRAAdapter, LoRAEquivariantLinear,
                                 average_atomic_energies, count_parameters,
                                 element_filter, estimate_atomic_energies,
                                 freeze_parameters, get_atomic_energies, has_lora,
                                 inject_lora, label_head, lora_modules, lora_parameters,
                                 merge_lora, pseudolabel, select_replay,
                                 set_atomic_energies)
from xnn.common.models import ForceStressOutput, build_model, from_pretrained, save_pretrained
from xnn.common.models.hub import build_potential, load_checkpoint
from xnn.common.models.registry import prepare_model
from xnn.common.train import Trainer, weighted_loss

SPECIES = [1, 6, 8]


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _structs(n=6, seed=0, periodic=False, labels=True, n_atoms=(4, 5, 6)):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        k = n_atoms[i % len(n_atoms)]
        s = {"pos": rng.uniform(0, 4, (k, 3)),
             "atomic_numbers": [SPECIES[j % 3] for j in range(k)]}
        if periodic and i % 2 == 0:
            s["cell"] = np.eye(3) * 6.0
            s["pbc"] = [True, True, True]
        if labels:
            s["energy"] = float(rng.normal())
            s["forces"] = rng.normal(0, 1, (k, 3))
            if "cell" in s:
                s["stress"] = rng.normal(0, 0.1, (3, 3))
        out.append(s)
    return out


def _batch(structs, cutoff=4.0, heads=None):
    graphs = [structure_to_graph(s, cutoff) for s in structs]
    if heads is not None:
        for g, h in zip(graphs, heads):
            g.head = torch.tensor([h])
    return collate(graphs)


# tiny configs of every model with readout heads
def _model_cfg(name):
    e3nn = pytest.importorskip("e3nn") if name in ("mace", "nequip", "allegro", "cace", "aimnet2") else None
    del e3nn
    base = {"cutoff": 4.0, "extra": {"species": SPECIES}}
    extras = {
        "mace": dict(n_features=8, n_interactions=2, extra={"max_ell": 2, "max_L": 1, "correlation": 2}),
        "nequip": dict(n_features=8, n_interactions=1, extra={"l_max": 1, "avg_num_neighbors": 6.0}),
        "allegro": dict(n_features=4, n_interactions=1,
                        extra={"l_max": 1, "avg_num_neighbors": 6.0, "two_body_latent": [8, 8],
                               "latent": [8], "edge_eng": [8]}),
        "cace": dict(n_interactions=1, n_rbf=4,
                     extra={"n_atom_basis": 2, "n_radial_basis": 4, "max_l": 2, "max_nu": 2,
                            "avg_num_neighbors": 6.0}),
        "aimnet2": dict(n_features=4, n_rbf=6, n_interactions=2,
                        extra={"hidden": [[12, 8], [12, 8, 8]], "aim_size": 8,
                               "readout_hidden": [8], "n_vector_combinations": 2}),
        "schnet": dict(n_features=16, n_interactions=1, n_rbf=10),
        "physnet": dict(n_features=12, n_rbf=8, n_interactions=1,
                        extra={"use_electrostatics": False, "use_dispersion": False}),
        "hdnnp": dict(extra={"hidden": [8]}),
        "bamboo": dict(n_features=16, n_rbf=8, n_interactions=2,
                       extra={"num_heads": 4, "use_dispersion": False}),
    }
    spec = dict(base)
    spec.update({k: v for k, v in extras[name].items() if k != "extra"})
    spec["extra"] = {**base["extra"], **extras[name].get("extra", {})}
    return from_dict({"model": {"name": name, **spec}})


def _randomize(model, seed=0, scale=0.3):
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(scale * torch.randn_like(p))


HEADED = ["mace", "nequip", "allegro", "cace", "aimnet2", "schnet", "physnet", "hdnnp", "bamboo"]


# AtomicGraph.subset
def test_subset_matches_collating_the_kept_structures():
    structs = _structs(5, periodic=True)
    graphs = [structure_to_graph(s, 4.0) for s in structs]
    for g, h in zip(graphs, [0, 1, 1, 0, 1]):
        g.head = torch.tensor([h])
        g.weight = torch.tensor([1.0 + h])
    full = collate(graphs)
    keep = full.head == 1
    sub = full.subset(keep)
    ref = collate([g for g, k in zip(graphs, keep.tolist()) if k])
    for name in ("pos", "atomic_numbers", "edge_index", "cell_shifts", "batch", "n_atoms",
                 "cell", "pbc", "energy", "forces", "stress", "weight", "head"):
        a, b = getattr(sub, name), getattr(ref, name)
        assert (a is None) == (b is None), name
        if a is not None:
            assert torch.equal(a, b), name
    assert sub.num_edges == ref.num_edges and sub.num_graphs == 3


# MultiHead
@pytest.mark.parametrize("name", HEADED)
def test_multihead_routes_each_structure_to_its_head(name):
    base = build_model(_model_cfg(name).model)
    _randomize(base)
    multi = MultiHead(base, ["a", "b"])
    assert multi.heads == ["a", "b"] and multi.index("b") == 1
    _randomize(multi.extra_heads[0], seed=1)        # make head b differ
    with torch.no_grad():
        for buf in multi.extra_heads[0].buffers():
            if buf.is_floating_point() and buf.numel() < 100:
                buf.mul_(1.1).add_(0.05)
    structs = _structs(4)
    heads = [0, 1, 1, 0]
    out = ForceStressOutput(multi)(_batch(structs, heads=heads))

    a = ForceStressOutput(multi.select("a"))
    b = ForceStressOutput(multi.select("b"))
    assert type(multi.select("a")) is type(base)
    n0 = 0
    n_seen = 0
    for s, h in zip(structs, heads):
        ref = (a if h == 0 else b)(_batch([s]))
        n = len(s["atomic_numbers"])
        i = n_seen
        n_seen += 1
        assert abs(float(out["energy"][i]) - float(ref["energy"][0])) < 1e-10
        assert (out["forces"][n0:n0 + n] - ref["forces"]).abs().max() < 1e-9
        assert (out["node_energy"][n0:n0 + n] - ref["node_energy"]).abs().max() < 1e-10
        n0 += n
    # the two heads disagree, and a batch without head labels is head 0
    e_a = a(_batch(structs))["energy"]
    e_b = b(_batch(structs))["energy"]
    assert (e_a - e_b).abs().max() > 1e-6
    assert (ForceStressOutput(multi)(_batch(structs))["energy"] - e_a).abs().max() < 1e-10
    assert "node_features" in out and out["node_features"].shape[0] == sum(
        len(s["atomic_numbers"]) for s in structs)


def test_multihead_gradients_reach_only_the_heads_present():
    pytest.importorskip("e3nn")
    base = build_model(_model_cfg("mace").model)
    multi = MultiHead(base, ["a", "b"])
    model = ForceStressOutput(multi)
    structs = _structs(3)
    data = _batch(structs, heads=[0, 0, 0])
    loss, _ = weighted_loss(model(data), data, 1.0, 10.0, 0.0)
    loss.backward()
    head_b = [p for p in multi.extra_heads[0].parameters()]
    assert all(p.grad is None for p in head_b)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in base.named_parameters()
               if n.startswith("interactions"))
    model.zero_grad()
    data = _batch(structs, heads=[1, 0, 1])
    loss, logs = weighted_loss(model(data), data, 1.0, 10.0, 0.0, head_names=["a", "b"])
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head_b)
    assert "a/energy_mse" in logs and "b/force_mse" in logs


def test_multihead_needs_head_modules_and_distinct_names():
    from xnn.common.models.base import InteratomicPotential

    class NoHeads(InteratomicPotential):
        def forward(self, data):
            raise NotImplementedError

        @classmethod
        def from_config(cls, cfg):
            return cls()

    with pytest.raises(TypeError, match="head_modules"):
        MultiHead(NoHeads(), ["a", "b"])
    model = build_model(_model_cfg("schnet").model)
    with pytest.raises(ValueError, match="distinct"):
        MultiHead(model, ["a", "a"])
    multi = MultiHead(model, ["a", "b"])
    with pytest.raises(KeyError):
        multi.index("c")


def test_multihead_config_roundtrip_and_head_selection(tmp_path):
    pytest.importorskip("e3nn")
    cfg = _model_cfg("mace")
    cfg.model.extra["heads"] = {"pt_head": None, "Default": {"atomic_energies": {1: -1.0, 6: -2.0, 8: -3.0}}}
    model = build_model(cfg.model)
    multi = model
    assert isinstance(multi, MultiHead) and multi.heads == ["pt_head", "Default"]
    assert multi.atomic_energies("Default") == {1: -1.0, 6: -2.0, 8: -3.0}
    assert multi.atomic_energies("pt_head") == {1: 0.0, 6: 0.0, 8: 0.0}
    _randomize(multi)
    structs = _structs(3)
    out = ForceStressOutput(multi)(_batch(structs, heads=[1, 0, 1]))

    rebuilt = build_model(cfg.model)
    rebuilt.load_state_dict(multi.state_dict())
    out2 = ForceStressOutput(rebuilt)(_batch(structs, heads=[1, 0, 1]))
    assert torch.allclose(out["energy"], out2["energy"], atol=1e-12)

    # the hub serves one head
    save_pretrained(multi, tmp_path / "two-heads", config=cfg)
    with pytest.raises(ValueError, match="pass head="):
        from_pretrained(tmp_path / "two-heads")
    served = from_pretrained(tmp_path / "two-heads", head="Default")
    ref = ForceStressOutput(multi.select("Default"))(_batch(structs))
    assert torch.allclose(served(_batch(structs))["energy"], ref["energy"], atol=1e-12)
    assert type(served.model).__name__ == "MACE"
    ck = load_checkpoint(tmp_path / "two-heads")
    assert ck.card.heads == ["pt_head", "Default"]


def test_multihead_mace_matches_upstream_multihead_model():
    """The xnn routing of a two-head MACE reproduces mace-torch's `head` per structure."""
    pytest.importorskip("mace")
    pytest.importorskip("ase")
    from test_mace import _upstream_eval, _upstream_scale_shift_mace
    from xnn.gnn.models.mace_foundation import from_mace_torch

    up = _upstream_scale_shift_mace("0b2", heads=["ha", "hb"])
    head_a = from_mace_torch(up, head="ha")
    head_b = from_mace_torch(up, head="hb")
    multi = MultiHead(head_a, ["ha", "hb"])
    for name in multi.head_module_names:
        getattr(multi.extra_heads[0], name).load_state_dict(getattr(head_b, name).state_dict())
    model = ForceStressOutput(multi)
    structs = _structs(4, seed=3)
    for heads in ([0, 1, 1, 0], [1, 1, 0, 1]):
        out = model(_batch(structs, cutoff=4.0, heads=heads))
        n0 = 0
        for i, (s, h) in enumerate(zip(structs, heads)):
            g = structure_to_graph(s, 4.0)
            e_u, f_u = _upstream_eval(up, g, head=h)
            n = g.num_nodes
            assert abs(float(out["energy"][i]) - e_u) < 1e-10
            assert (out["forces"][n0:n0 + n] - f_u).abs().max() < 1e-10
            n0 += n


# LoRA
@pytest.mark.parametrize("name", HEADED)
def test_lora_starts_at_the_pretrained_model_and_merges_back(name):
    base = build_model(_model_cfg(name).model)
    _randomize(base)
    structs = _structs(3)
    ref = ForceStressOutput(base)(_batch(structs))
    model = copy.deepcopy(base)
    inject_lora(model, rank=2, alpha=1.0)
    n_adapters = len(list(lora_modules(model)))
    assert n_adapters > 0 and has_lora(model)
    out = ForceStressOutput(model)(_batch(structs))
    assert torch.allclose(out["energy"], ref["energy"], atol=1e-11)
    assert torch.allclose(out["forces"], ref["forces"], atol=1e-10)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_in" in n or "lora_out" in n for n in trainable)
    assert count_parameters(model, trainable_only=True) < 0.5 * count_parameters(model)

    # a trained adapter differs from the base and merges exactly
    torch.manual_seed(1)
    with torch.no_grad():
        for p in lora_parameters(model):
            p.add_(0.2 * torch.randn_like(p))
    adapted = ForceStressOutput(model)(_batch(structs))
    assert (adapted["energy"] - ref["energy"]).abs().max() > 1e-6
    merged = merge_lora(copy.deepcopy(model))
    assert not has_lora(merged)
    assert set(merged.state_dict()) == set(base.state_dict())
    out_m = ForceStressOutput(merged)(_batch(structs))
    assert torch.allclose(out_m["energy"], adapted["energy"], atol=1e-10)
    assert torch.allclose(out_m["forces"], adapted["forces"], atol=1e-9)
    assert all(p.requires_grad for p in merged.parameters())


@pytest.mark.parametrize("name", ["mace", "nequip"])
def test_lora_keeps_equivariance(name):
    pytest.importorskip("e3nn")
    model = build_model(_model_cfg(name).model)
    _randomize(model)
    inject_lora(model, rank=3)
    torch.manual_seed(2)
    with torch.no_grad():
        for p in lora_parameters(model):
            p.add_(0.3 * torch.randn_like(p))
    fmodel = ForceStressOutput(model)
    rng = np.random.default_rng(3)
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    s = _structs(1, seed=4)[0]
    out = fmodel(_batch([s]))
    rotated = dict(s, pos=s["pos"] @ R.T)
    out_r = fmodel(_batch([rotated]))
    assert abs(float(out["energy"][0]) - float(out_r["energy"][0])) < 1e-9
    assert (out["forces"] @ torch.tensor(R).T - out_r["forces"]).abs().max() < 1e-8


def test_lora_config_roundtrip_and_deployment_merge(tmp_path):
    pytest.importorskip("e3nn")
    cfg = _model_cfg("nequip")
    cfg.model.extra["lora"] = {"rank": 2, "trainable": ["*atom_ref*"]}
    model = build_model(cfg.model)
    assert has_lora(model)
    assert model.atom_ref.weight.requires_grad
    torch.manual_seed(5)
    with torch.no_grad():
        for p in lora_parameters(model):
            p.add_(0.2 * torch.randn_like(p))
    structs = _structs(2)
    out = ForceStressOutput(model)(_batch(structs))
    rebuilt = build_model(cfg.model)
    rebuilt.load_state_dict(model.state_dict())
    assert torch.allclose(ForceStressOutput(rebuilt)(_batch(structs))["energy"], out["energy"])
    served = build_potential(cfg, ForceStressOutput(model).state_dict())
    assert not has_lora(served)
    assert torch.allclose(served(_batch(structs))["energy"], out["energy"], atol=1e-10)
    with pytest.raises(ValueError, match="unknown option"):
        build_model(from_dict({"model": {"name": "nequip", "cutoff": 4.0,
                                         "extra": {"species": SPECIES, "lora": {"rnk": 2}}}}).model)


def test_lora_equivariant_linear_matches_upstream_composition():
    """Per-block LoRA on an o3.Linear equals mace-torch's LoRAO3Linear once the factors are mapped."""
    pytest.importorskip("mace")
    from e3nn import o3
    from mace.modules.lora import LoRAO3Linear

    torch.manual_seed(0)
    # (no biases: the upstream fused path cannot handle bias instructions)
    lin = o3.Linear("6x0e+6x0e+4x1o+2x2e", "5x0e+3x1o+2x2e")
    up = LoRAO3Linear(copy.deepcopy(lin), rank=3, alpha=2.0)
    with torch.no_grad():
        for p in list(up.lora_A.parameters()) + list(up.lora_B.parameters()):
            p.normal_()
    mine = LoRAEquivariantLinear(copy.deepcopy(lin), rank=3, alpha=2.0)
    # map upstream's (A: i_in -> mid, B: mid -> i_out) onto my per-path factors
    a_blocks = {ins.i_in: (idx, ins) for idx, ins in enumerate(up.lora_A.instructions)}
    b_blocks = {(ins.i_in, ins.i_out): (idx, ins) for idx, ins in enumerate(up.lora_B.instructions)}
    weight_paths = [ins for ins in lin.instructions if ins.i_in >= 0]
    with torch.no_grad():
        for k, ins in enumerate(weight_paths):
            ia, ins_a = a_blocks[ins.i_in]
            ib, ins_b = b_blocks[(ins_a.i_out, ins.i_out)]
            a = up.lora_A.weight_view_for_instruction(ia)
            b = up.lora_B.weight_view_for_instruction(ib)
            mine.lora_in[k].copy_(a)
            mine.lora_out[k].copy_(ins_a.path_weight * ins_b.path_weight * b)
    x = torch.randn(7, lin.irreps_in.dim)
    with torch.enable_grad():                     # upstream: activation-space path
        assert torch.allclose(mine(x), up(x), atol=1e-12)
    with torch.no_grad():                         # upstream: fused weight-space path
        assert torch.allclose(mine(x), up(x), atol=1e-12)
        assert torch.allclose(mine.adapted_weight(), up.compute_merged_weight(), atol=1e-12)


def test_lora_dense_adapters_match_upstream_composition():
    pytest.importorskip("mace")
    from e3nn.nn import FullyConnectedNet
    from mace.modules.lora import LoRADenseLinear, LoRAFCLayer

    torch.manual_seed(0)
    lin = torch.nn.Linear(9, 5)
    up = LoRADenseLinear(copy.deepcopy(lin), rank=2, alpha=3.0)
    mine = LoRAAdapter(copy.deepcopy(lin), rank=2, alpha=3.0, in_axis=1)
    with torch.no_grad():
        up.lora_A.weight.normal_()
        up.lora_B.weight.normal_()
        mine.lora_in.copy_(up.lora_A.weight.t())
        mine.lora_out.copy_(up.lora_B.weight.t())
    x = torch.randn(4, 9)
    assert torch.allclose(mine(x), up(x), atol=1e-12)

    net = FullyConnectedNet([6, 8, 3], torch.nn.functional.silu)
    layer = net.layer0
    up_fc = LoRAFCLayer(copy.deepcopy(layer), rank=2, alpha=1.5)
    mine_fc = LoRAAdapter(copy.deepcopy(layer), rank=2, alpha=1.5, in_axis=0)
    with torch.no_grad():
        up_fc.lora_A.normal_()
        up_fc.lora_B.normal_()
        mine_fc.lora_in.copy_(up_fc.lora_A)
        mine_fc.lora_out.copy_(up_fc.lora_B)
    x = torch.randn(4, 6)
    assert torch.allclose(mine_fc(x), up_fc(x), atol=1e-12)


# reference energies
@pytest.mark.parametrize("name", ["mace", "schnet", "hdnnp", "physnet"])
def test_reference_energies_get_set_and_estimate(name):
    model = build_model(_model_cfg(name).model)
    _randomize(model)
    set_atomic_energies(model, {1: -0.5, "C": -1.5, 8: -2.5})
    assert get_atomic_energies(model) == {1: -0.5, 6: -1.5, 8: -2.5}
    structs = _structs(8, seed=7, n_atoms=(4, 5, 6, 7))
    wrapped = ForceStressOutput(model)
    shift = {1: 0.3, 6: -1.2, 8: 2.0}
    with torch.no_grad():
        for s in structs:
            e = float(wrapped(_batch([s]))["energy"][0])
            s["energy"] = e + sum(shift[int(z)] for z in s["atomic_numbers"])
    e0 = estimate_atomic_energies(model, structs)
    for z in SPECIES:
        assert abs(e0[z] - (get_atomic_energies(model)[z] + shift[z])) < 1e-7
    # averaged references recover a strictly additive energy
    avg = {1: -13.6, 6: -1030.0, 8: -2040.0}
    for s in structs:
        s["energy"] = sum(avg[int(z)] for z in s["atomic_numbers"])
    est = average_atomic_energies(structs)
    for z in SPECIES:
        assert abs(est[z] - avg[z]) < 1e-6


def test_multihead_reference_energies_per_head():
    model = build_model(_model_cfg("schnet").model)
    multi = MultiHead(model, ["pt_head", "Default"])
    multi.set_atomic_energies("Default", [-1.0, -2.0, -3.0])
    assert multi.atomic_energies("Default") == {1: -1.0, 6: -2.0, 8: -3.0}
    assert multi.atomic_energies("pt_head") == {1: 0.0, 6: 0.0, 8: 0.0}
    with multi.using("Default") as single:
        assert get_atomic_energies(single)[8] == -3.0
    assert get_atomic_energies(multi.model)[8] == 0.0


# replay
def test_replay_selection_and_pseudolabels():
    structs = _structs(6, periodic=True, seed=2)
    structs.append({"pos": np.zeros((2, 3)) + [[0, 0, 0], [1.1, 0, 0]], "atomic_numbers": [7, 1]})
    assert len(element_filter(structs, SPECIES, "subset")) == 6
    assert len(element_filter(structs, SPECIES, "exact")) == 6
    assert len(element_filter(structs, [7], "superset")) == 1
    assert len(element_filter(structs, SPECIES, "none")) == 7
    with pytest.raises(ValueError, match="replay filter"):
        element_filter(structs, SPECIES, "all")
    picked = select_replay(structs, SPECIES, n=3, seed=4)
    assert len(picked) == 3 and picked == select_replay(structs, SPECIES, n=3, seed=4)

    model = build_model(_model_cfg("schnet").model)
    _randomize(model)
    labelled = pseudolabel(model, picked, batch_size=2)
    wrapped = ForceStressOutput(model, compute_stress=True)
    for s, new in zip(picked, labelled):
        ref = wrapped(_batch([s]))
        assert abs(new["energy"] - float(ref["energy"][0])) < 1e-10
        assert np.abs(new["forces"] - ref["forces"].detach().numpy()).max() < 1e-9
        assert ("stress" in new) == ("cell" in s)
        if "cell" in s:
            assert np.abs(new["stress"] - ref["stress"][0].detach().numpy()).max() < 1e-9
    multi = MultiHead(model, ["pt_head", "Default"])
    _randomize(multi.extra_heads[0], seed=9)
    via_head = pseudolabel(multi, picked[:1], head="Default")
    ref = ForceStressOutput(multi.select("Default"))(_batch(picked[:1]))
    assert abs(via_head[0]["energy"] - float(ref["energy"][0])) < 1e-10
    assert label_head(picked, 1)[0]["head"] == 1


# the loss
def test_loss_sums_per_head_means_with_their_own_weights():
    model = ForceStressOutput(build_model(_model_cfg("schnet").model))
    structs = _structs(5, seed=5)
    heads = [0, 1, 1, 0, 1]
    data = _batch(structs, heads=heads)
    pred = model(data)
    hw = {1: (1.0, 10.0, 0.0)}
    loss, logs = weighted_loss(pred, data, 10.0, 10.0, 0.0, head_weights=hw, head_names=["t", "r"])
    total = 0.0
    for h, (we, wf, ws) in ((0, (10.0, 10.0, 0.0)), (1, hw[1])):
        sub = [s for s, k in zip(structs, heads) if k == h]
        d = _batch(sub)
        part, part_logs = weighted_loss(model(d), d, we, wf, ws)
        total += float(part)
        name = ["t", "r"][h]
        assert abs(logs[f"{name}/energy_mse"] - part_logs["energy_mse"]) < 1e-10
    assert abs(float(loss) - total) < 1e-9
    # one head everywhere is the plain loss
    plain = _batch(structs)
    one = _batch(structs, heads=[0] * 5)
    a, _ = weighted_loss(model(plain), plain, 1.0, 10.0, 0.0)
    b, _ = weighted_loss(model(one), one, 1.0, 10.0, 0.0)
    assert abs(float(a) - float(b)) < 1e-12


# the trainer
def _train_cfg(tmp_path, name="schnet", **optim):
    cfg = _model_cfg(name)
    cfg.device = "cpu"
    cfg.optim.epochs = 2
    cfg.optim.lr = 1e-3
    cfg.data.batch_size = 3
    cfg.data.val_fraction = 0.25
    cfg.output_dir = str(tmp_path / "run")
    for k, v in optim.items():
        setattr(cfg.optim, k, v)
    cfg.__post_init__()
    return cfg


def test_trainer_multihead_replay_with_pseudolabels(tmp_path):
    cfg = _train_cfg(tmp_path)
    cfg.model.extra["heads"] = ["pt_head", "Default"]
    cfg.optim.head_weights = {"pt_head": {"energy_weight": 1.0, "force_weight": 10.0}}
    cfg.optim.energy_weight = 10.0
    cfg.data.replay_pseudolabel = True
    cfg.data.replay_samples = 4
    train = AtomicDataset(_structs(8, seed=1), 4.0)
    replay = AtomicDataset(_structs(6, seed=2, labels=False), 4.0)
    trainer = Trainer(cfg, train, replay_set=replay)
    assert trainer.heads == ["pt_head", "Default"]
    assert all(s["head"] == 1 for s in train.structures)
    assert len(trainer.train_loader.dataset) == 6 + 3            # 8 - 2 val, 4 - 1 val
    metrics = trainer.fit()
    assert "Default/loss" in metrics["train"] and "pt_head/loss" in metrics["train"]
    ck = torch.load(tmp_path / "run" / "last.pt", weights_only=False)
    assert ck["cfg"].model.extra["heads"] == ["pt_head", "Default"]
    served = from_pretrained(tmp_path / "run" / "last.pt", head="Default")
    assert type(served.model).__name__ == "SchNet"
    with pytest.raises(ValueError, match="pass head="):
        from_pretrained(tmp_path / "run" / "last.pt")
    with pytest.raises(ValueError, match="multi-head"):
        Trainer(_train_cfg(tmp_path), AtomicDataset(_structs(4), 4.0), replay_set=replay)


def test_trainer_pretrained_lora_estimated_references_ema(tmp_path):
    # a "foundation" checkpoint of the tiny model
    pre = _train_cfg(tmp_path / "pre")
    structs = _structs(8, seed=1)
    Trainer(pre, AtomicDataset(structs, 4.0)).fit()
    source = str(tmp_path / "pre" / "run" / "last.pt")
    pretrained = from_pretrained(source)

    cfg = _train_cfg(tmp_path, optimizer="adamw", clip_grad=1.0, ema_decay=0.9)
    cfg.model = from_dict({"model": {"name": "schnet", "cutoff": 4.0,
                                     "extra": {"pretrained": source, "lora": {"rank": 2},
                                               "atomic_energies": "estimated"}}}).model
    cfg.__post_init__()
    # labels = the pretrained predictions plus 2 eV per atom: the reestimated
    # references must come out exactly 2 eV above the pretrained ones
    shifted = [dict(s, energy=float(pretrained(_batch([s]))["energy"][0]) + 2.0 * len(s["atomic_numbers"]))
               for s in structs]
    e0_pre = get_atomic_energies(pretrained.model)
    trainer = Trainer(cfg, AtomicDataset(shifted, 4.0))
    core = trainer.module.model
    assert has_lora(core) and trainer.ema is not None
    assert "pretrained" not in trainer.cfg.model.extra
    e0 = trainer.cfg.model.extra["atomic_energies"]
    assert set(e0) == set(SPECIES) and all(abs(e0[z] - e0_pre[z] - 2.0) < 1e-6 for z in SPECIES)
    assert isinstance(trainer.opt, torch.optim.AdamW)
    trainer.fit()
    ck = torch.load(tmp_path / "run" / "last.pt", weights_only=False)
    assert "lora" in ck["cfg"].model.extra and any("lora_in" in k for k in ck["model"])
    served = from_pretrained(tmp_path / "run" / "last.pt")
    merged = merge_lora(copy.deepcopy(trainer.eval_module))
    batch = _batch(structs[:2])
    assert torch.allclose(served(batch)["energy"], merged(batch)["energy"], atol=1e-10)


def test_trainer_train_only_freezes_the_rest(tmp_path):
    cfg = _train_cfg(tmp_path, train_only=["*readout*", "*atom_ref*"])
    trainer = Trainer(cfg, AtomicDataset(_structs(6), 4.0))
    frozen = {n: p.detach().clone() for n, p in trainer.module.named_parameters()
              if not p.requires_grad}
    assert frozen and any("interactions" in n for n in frozen)
    trainer.fit()
    for n, p in trainer.module.named_parameters():
        if n in frozen:
            assert torch.equal(p, frozen[n]), n


def test_prepare_model_pretrained_checks(tmp_path):
    pre = _train_cfg(tmp_path / "pre")
    Trainer(pre, AtomicDataset(_structs(4), 4.0)).fit()
    source = str(tmp_path / "pre" / "run" / "last.pt")
    with pytest.raises(ValueError, match="cutoff"):
        prepare_model(from_dict({"model": {"name": "schnet", "cutoff": 5.0,
                                           "extra": {"pretrained": source}}}).model)
    with pytest.raises(ValueError, match="architecture comes from the checkpoint"):
        prepare_model(from_dict({"model": {"name": "schnet", "cutoff": 4.0,
                                           "extra": {"pretrained": source, "gamma": 3.0}}}).model)
    with pytest.raises(ValueError, match="holds a 'schnet'"):
        prepare_model(from_dict({"model": {"name": "physnet", "cutoff": 4.0,
                                           "extra": {"pretrained": source}}}).model)
    with pytest.raises(ValueError, match="estimated"):
        build_model(from_dict({"model": {"name": "schnet", "cutoff": 4.0,
                                         "extra": {"species": SPECIES,
                                                   "atomic_energies": "estimated"}}}).model)
    model, resolved = prepare_model(from_dict({"model": {"name": "schnet", "cutoff": 4.0,
                                                         "extra": {"pretrained": source,
                                                                   "heads": ["pt_head", "Default"]}}}).model)
    assert isinstance(model, MultiHead) and resolved.extra["heads"] == ["pt_head", "Default"]
    assert "pretrained" not in resolved.extra and resolved.n_features == pre.model.n_features


def test_freeze_parameters_patterns():
    model = build_model(_model_cfg("schnet").model)
    n_train, n_total = freeze_parameters(model, freeze=["embedding*"])
    assert not model.embedding.weight.requires_grad and n_train < n_total
    n_train, _ = freeze_parameters(model, train_only=["atom_ref*"])
    assert n_train == model.atom_ref.weight.numel()
