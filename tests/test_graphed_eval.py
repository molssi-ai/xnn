"""Captured evaluations (xnn.common.deploy.GraphedEvaluation): the replayed results equal ordinary ones."""
import numpy as np
import pytest
import torch

from xnn.common.data import collate, structure_to_graph
from xnn.common.deploy import GraphedEvaluation
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
