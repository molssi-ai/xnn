"""Captured evaluations (xnn.common.deploy.GraphedEvaluation): the replayed results equal ordinary ones."""
import types

import numpy as np
import pytest
import torch

from xnn.common.data import collate, structure_to_graph
from xnn.common.deploy import GraphedEvaluation
from xnn.common.deploy import graphed_eval as graphed_eval_module
from xnn.common.models import ForceStressOutput


def _physnet():
    from xnn.dnn.models.physnet import PhysNet
    torch.manual_seed(0)
    model = PhysNet(cutoff=5.0, n_features=16, n_rbf=8, num_blocks=2)
    with torch.no_grad():
        for ob in model.output_blocks:
            ob.dense.weight.normal_(0, 0.1)
        for ib in model.interaction_blocks:
            ib.interaction.k2f.weight.normal_(0, 0.1)
    return model


def _molecule(n, seed, step=0):
    """A random molecule, displaced by ``0.02 * step`` per coordinate (steps of a trajectory)."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 4.0, (n, 3)) + 0.02 * step * np.random.default_rng(seed + 100 + step).standard_normal((n, 3))
    return {"pos": pos, "atomic_numbers": ([8, 1, 1, 6, 7] * n)[:n]}


def test_a_model_without_an_index_planner_is_rejected():
    from xnn.gnn.models.schnet import SchNet
    with pytest.raises(ValueError, match="index_planner"):
        GraphedEvaluation(ForceStressOutput(SchNet()))


def _aimnet2(coulomb=None, n_rbf=8):
    pytest.importorskip("e3nn")
    from xnn.gnn.models.aimnet2 import AIMNet2
    torch.manual_seed(2)
    return AIMNet2([1, 6, 7, 8], cutoff=3.5, n_features=4, n_rbf=n_rbf, hidden=[[16, 12], [16, 12], [16, 12, 12]],
                   aim_size=10, readout_hidden=[8], n_vector_combinations=3, coulomb=coulomb)


def test_aimnet2_checks_its_species_in_the_planner_and_needs_a_capturable_coulomb():
    """AIMNet2's planner rejects elements outside its species before the padding; the all-pairs and lattice
    Coulomb sums and the reference shell convolutions size their work on the host and are refused."""
    planner = _aimnet2().index_planner()
    good = structure_to_graph(_molecule(6, 0), 3.5)
    assert planner(good) == {}
    bad = structure_to_graph({"pos": _molecule(3, 1)["pos"], "atomic_numbers": [8, 1, 9]}, 3.5)
    with pytest.raises(ValueError, match=r"elements \[9\]"):
        planner(bad)
    assert _aimnet2("dsf").index_planner() is not None
    with pytest.raises(ValueError, match="coulomb"):
        GraphedEvaluation(ForceStressOutput(_aimnet2("simple")))
    with pytest.raises(ValueError, match="n_rbf"):
        _aimnet2(n_rbf=6).index_planner()
    with pytest.raises(ValueError, match="use_fast"):
        _aimnet2().set_use_fast(False).index_planner()


def _model(name):
    if name == "physnet":
        return _physnet()
    if name.startswith("aimnet2"):
        return _aimnet2("dsf" if name == "aimnet2-dsf" else None)
    if name == "mace":
        pytest.importorskip("e3nn")
        from test_trainer_cuda_graph import _small as small
        return small("mace")
    from test_trainer_cuda_graph import _small
    return _small(name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("name", ["physnet", "dimenet", "dimenet++", "gemnet", "spookynet", "spookynet-lr", "mace",
                                  "aimnet2", "aimnet2-dsf"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_replays_equal_ordinary_evaluations(name, dtype):
    """Energies and forces (charges and dipoles of PhysNet) along a sequence of inputs (the steps of a
    trajectory, a larger structure, which captures again, and a two-structure batch) equal the
    ordinary evaluation's."""
    model = _model(name).to("cuda", dtype).eval()
    wrapped = ForceStressOutput(model).eval()
    graphed = GraphedEvaluation(wrapped, trial=0)
    inputs = ([[_molecule(8, 0, k)] for k in range(5)] + [[_molecule(14, 4, k)] for k in range(4)]
              + [[_molecule(6, 6), _molecule(5, 7)]])
    tol = 1e-12 if dtype == torch.float64 else 1e-5
    captured = 0
    for structures in inputs:
        g = collate([structure_to_graph(s, model.cutoff) for s in structures]).to("cuda")
        g.pos = g.pos.to(dtype)
        ref = wrapped(g)
        out = graphed(g)
        captured += graphed.graph is not None
        for key in ("energy", "forces", "charges", "dipole"):
            if key not in ref:
                continue
            r = ref[key].detach()
            assert out[key].shape == r.shape, key
            assert (out[key] - r).abs().max() <= tol * max(float(r.abs().max()), 1.0), key
    assert captured > 0


class _ScriptedTerm(torch.nn.Module):
    """A model plus a per-atom energy term from a TorchScript function of the positions."""

    def __init__(self, model, term):
        super().__init__()
        self.model, self.term, self.cutoff = model, term, model.cutoff

    def forward(self, data):
        out = dict(self.model(data))
        atom = self.term(data.pos).sum(-1).to(out["energy"].dtype)
        out["energy"] = out["energy"] + torch.zeros_like(out["energy"]).index_add(0, data.batch, atom)
        return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_a_torchscript_block_first_run_in_the_warmup_is_captured():
    """A TorchScript block never run before the wrapper is captured and replays the ordinary results."""
    def term(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x) * x + torch.exp(-x * x)

    model = _ScriptedTerm(_physnet(), torch.jit.script(term)).to("cuda").eval()
    wrapped = ForceStressOutput(model).eval()
    graphed = GraphedEvaluation(wrapped, trial=0)
    graphs = [structure_to_graph(_molecule(8, 0, k), model.cutoff).to("cuda") for k in range(5)]
    outs = [graphed(g) for g in graphs]
    assert graphed.graph is not None
    for g, out in zip(graphs, outs):
        ref = wrapped(g)
        for key in ("energy", "forces"):
            r = ref[key].detach()
            assert (out[key] - r).abs().max() <= 1e-5 * max(float(r.abs().max()), 1.0), key


def test_padded_inputs_give_the_ordinary_results_and_keep_their_counts():
    """On the CPU (no capture): a padded input evaluates to the ordinary energies and forces of its real
    atoms; the counts grow for a larger input or another number of structures and hold for a smaller one;
    later inputs are copied into the static one; positions with autograd history are detached."""
    model = _physnet().double().eval()
    wrapped = ForceStressOutput(model).eval()
    graphed = GraphedEvaluation(wrapped, slack=8)
    graphs = [structure_to_graph(s, model.cutoff) for s in (_molecule(8, 0), _molecule(8, 0, 1), _molecule(6, 2),
                                                             _molecule(20, 4))]
    for g in graphs:
        g.pos = g.pos.double()
    caps = []
    for g in graphs:
        g.pos.requires_grad_(True)
        padded = graphed._pad(g)
        assert not padded.pos.requires_grad and padded.num_graphs == 2
        caps.append(graphed.cap)
        ref, out = wrapped(g), wrapped(padded)
        n = g.num_nodes
        assert torch.allclose(out["energy"][:1], ref["energy"], rtol=0, atol=1e-12)
        assert torch.allclose(out["forces"][:n], ref["forces"], rtol=0, atol=1e-12)
    assert caps[0] is caps[1] is caps[2] and caps[3] is not caps[2]
    assert caps[3].atoms >= 20 + 2 and caps[3].edges % 2 == 0
    two = graphed._pad(collate(graphs[:2]))
    assert graphed.n_graphs == 2 and two.num_graphs == 3
    first = graphed._pad(graphs[0])
    graphed._reset()
    graphed._load(first)
    pos = graphed.static.pos
    second = graphed._pad(graphs[1])
    graphed._load(second)
    assert graphed.static.pos is pos and torch.equal(pos, second.pos)
    third = graphed._pad(graphs[2])
    third.cell = torch.eye(3, dtype=torch.float64)[None] * 20.0
    with pytest.raises(RuntimeError, match="same fields"):
        graphed._load(third)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("winner", ["replay", "eager"])
def test_the_trial_keeps_the_faster_evaluation(monkeypatch, winner):
    """After a capture the faster of replays and ordinary evaluations is kept; an ordinary winner
    releases the graph. The results equal the ordinary evaluation's either way."""
    model = _physnet().to("cuda").eval()
    wrapped = ForceStressOutput(model).eval()
    graphed = GraphedEvaluation(wrapped)
    monkeypatch.setattr(graphed, "_choose", lambda graph: winner)
    for k in range(5):
        g = structure_to_graph(_molecule(8, 0, k), model.cutoff).to("cuda")
        ref, out = wrapped(g), graphed(g)
        assert torch.allclose(out["forces"], ref["forces"].detach(), atol=1e-5)
        assert torch.allclose(out["energy"], ref["energy"].detach(), atol=1e-5)
    assert graphed.mode == winner and (graphed.graph is None) == (winner == "eager")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_the_trial_times_both_evaluations():
    """The trial itself runs and picks one of the two."""
    model = _physnet().to("cuda").eval()
    graphed = GraphedEvaluation(ForceStressOutput(model).eval(), trial=2)
    for k in range(4):
        graphed(structure_to_graph(_molecule(8, 0, k), model.cutoff).to("cuda"))
    assert graphed.mode in ("replay", "eager")


def _cpu_graphed(monkeypatch, **options):
    """A captured evaluation of a float64 PhysNet on the CPU, with CUDA's synchronization and cache release
    recorded instead of run."""
    model = _physnet().double().eval()
    wrapped = ForceStressOutput(model).eval()
    graphed = GraphedEvaluation(wrapped, **options)
    calls = {"synchronize": 0, "empty_cache": 0}

    def synchronize(*args, **kwargs):
        calls["synchronize"] += 1

    def empty_cache():
        calls["empty_cache"] += 1

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    graphs = [structure_to_graph(_molecule(8, 0, k), model.cutoff) for k in range(4)]
    for g in graphs:
        g.pos = g.pos.double()
    return wrapped, graphed, graphs, calls


def _stand_in_replay(graphed, replays):
    """A replay that evaluates ordinarily and counts as captured after the warmup."""
    def replayed(graph):
        replays.append(graph)
        graphed.steps += 1
        if graphed.steps > graphed.warmup:
            graphed.graph = "captured"
        return graphed._ordinary(graph)
    return replayed


def _same(out, ref):
    return all(torch.allclose(out[k], ref[k].detach(), rtol=0, atol=1e-12) for k in ("energy", "forces"))


@pytest.mark.parametrize("ordinary_time, expected", [(2.0, "replay"), (0.5, "eager")])
def test_the_trial_picks_the_faster_by_the_clock(monkeypatch, ordinary_time, expected):
    """The trial times the replays and the ordinary evaluations between synchronizations and keeps the
    faster; ordinary evaluations out of memory keep the replays and release the cache."""
    wrapped, graphed, graphs, calls = _cpu_graphed(monkeypatch, trial=3)
    clock = [0.0]
    monkeypatch.setattr(graphed_eval_module, "time", types.SimpleNamespace(perf_counter=lambda: clock[0]))
    runs = {"replay": 0, "ordinary": 0}

    def replayed(graph):
        runs["replay"] += 1
        clock[0] += 1.0

    def ordinary(graph):
        runs["ordinary"] += 1
        clock[0] += ordinary_time
        return graphed_eval_module.GraphedEvaluation._ordinary(graphed, graph)

    monkeypatch.setattr(graphed, "_replayed", replayed)
    monkeypatch.setattr(graphed, "_ordinary", ordinary)
    assert graphed._choose(graphs[0]) == expected
    assert runs == {"replay": 3, "ordinary": 3} and calls["synchronize"] == 4

    def out_of_memory(graph):
        raise torch.OutOfMemoryError("out of memory")

    monkeypatch.setattr(graphed, "_ordinary", out_of_memory)
    assert graphed._choose(graphs[0]) == "replay" and calls["empty_cache"] == 1


@pytest.mark.parametrize("trial, winner", [(0, None), (3, "replay"), (3, "eager")])
def test_the_mode_follows_the_first_capture(monkeypatch, trial, winner):
    """No mode before the capture; then replays without a trial, or the trial's winner, an ordinary winner
    releasing the graph and evaluating ordinarily from then on. The results are the ordinary ones."""
    wrapped, graphed, graphs, calls = _cpu_graphed(monkeypatch, trial=trial, warmup=2)
    replays = []
    monkeypatch.setattr(graphed, "_replayed", _stand_in_replay(graphed, replays))
    monkeypatch.setattr(graphed, "_choose", lambda graph: winner)
    for k, g in enumerate(graphs):
        assert _same(graphed(g), wrapped(g))
        if k < 2:
            assert graphed.mode is None
    expected = "replay" if trial == 0 else winner
    assert graphed.mode == expected
    if expected == "eager":
        assert graphed.graph is None and graphed.static is None and graphed.inputs == {}
        assert len(replays) == 3 and calls["empty_cache"] == 1
    else:
        assert graphed.graph == "captured" and len(replays) == 4 and calls["empty_cache"] == 0


def test_a_replay_out_of_memory_falls_back_to_ordinary_evaluations(monkeypatch):
    """A replay out of memory releases the graph and evaluates this input and the later ones ordinarily."""
    wrapped, graphed, graphs, calls = _cpu_graphed(monkeypatch)
    replays = []

    def out_of_memory(graph):
        replays.append(graph)
        raise torch.OutOfMemoryError("out of memory")

    monkeypatch.setattr(graphed, "_replayed", out_of_memory)
    graphed.graph, graphed.out = "captured", {}
    for g in graphs[:2]:
        assert _same(graphed(g), wrapped(g))
    assert graphed.mode == "eager" and graphed.graph is None and graphed.out is None
    assert len(replays) == 1 and calls["empty_cache"] == 1
