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


def _model(name):
    if name == "physnet":
        return _physnet()
    from test_trainer_cuda_graph import _small
    return _small(name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("name", ["physnet", "dimenet", "dimenet++", "gemnet"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_replays_equal_ordinary_evaluations(name, dtype):
    """Energies and forces (charges and dipoles of PhysNet) along a sequence of inputs (the steps of a
    trajectory, a larger structure, which captures again, and a two-structure batch) equal the
    ordinary evaluation's."""
    model = _model(name).to("cuda", dtype).eval()
    wrapped = ForceStressOutput(model).eval()
    graphed = GraphedEvaluation(wrapped)
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
