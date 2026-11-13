"""Per-structure property labels and global states: graphs, batches, subsets, loss, metrics, outputs."""
import numpy as np
import pytest
import torch
from torch import nn

from xnn.common.benchmark.metrics import prediction_pairs
from xnn.common.data import collate, structure_to_graph
from xnn.common.models import ForceStressOutput
from xnn.common.train.losses import weighted_loss


def _s(n=3, seed=0, **extra):
    rng = np.random.default_rng(seed)
    return {"pos": rng.uniform(0, 3, (n, 3)), "atomic_numbers": [6] * n, **extra}


def test_graph_state_labels_features_and_properties():
    """An integer state stays a long label, a float state becomes a feature row; properties keep their shape."""
    g = structure_to_graph(_s(state=2, properties=1.5), 3.0)
    assert g.state.dtype == torch.long and g.state.tolist() == [2] and g.properties.shape == (1,)
    g = structure_to_graph(_s(state=torch.tensor(1), properties=[1.0, 2.0]), 3.0)
    assert g.state.dtype == torch.long and g.properties.shape == (1, 2)
    g = structure_to_graph(_s(state=[300.0, 1.0]), 3.0)
    assert g.state.dtype == torch.get_default_dtype() and g.state.shape == (1, 2)
    assert g.properties is None


def test_collate_fills_missing_labels_and_states():
    """A batch mixing labelled and unlabelled structures gets zeros and a mask; a missing state is zero."""
    graphs = [structure_to_graph(_s(properties=2.0, state=1), 3.0),
              structure_to_graph(_s(seed=1), 3.0),
              structure_to_graph(_s(seed=2, properties=-1.0, state=3), 3.0)]
    b = collate(graphs)
    assert b.properties.tolist() == [2.0, 0.0, -1.0] and b.properties_mask.tolist() == [True, False, True]
    assert b.state.tolist() == [1, 0, 3] and b.state.dtype == torch.long
    full = collate([graphs[0], graphs[2]])
    assert full.properties_mask is None
    sub = b.subset(torch.tensor([False, True, True]))
    assert sub.properties.tolist() == [0.0, -1.0] and sub.state.tolist() == [0, 3]
    assert sub.properties_mask.tolist() == [False, True]
    feats = collate([structure_to_graph(_s(state=[1.0, 2.0]), 3.0), structure_to_graph(_s(seed=1), 3.0)])
    assert feats.state.tolist() == [[1.0, 2.0], [0.0, 0.0]]
    assert collate(graphs[1:2]).properties is None and collate(graphs[1:2]).state is None


def test_property_loss_is_a_plain_mean_over_labelled_structures():
    """The property term averages over the labelled structures, without division by the atom count."""
    b = collate([structure_to_graph(_s(n=2, properties=1.0), 3.0), structure_to_graph(_s(n=5, seed=1), 3.0),
                 structure_to_graph(_s(n=4, seed=2, properties=3.0), 3.0)])
    pred = {"properties": torch.tensor([2.0, 7.0, 1.0], requires_grad=True)}
    counts = {}
    loss, logs = weighted_loss(pred, b, 1.0, 10.0, 0.0, property_weight=2.0, counts=counts)
    assert logs["property_mse"] == pytest.approx((1.0 + 4.0) / 2)
    assert float(loss) == pytest.approx(5.0) and counts["property_mse"] == (2.0, 2.0)
    loss.backward()
    assert pred["properties"].grad[1] == 0.0
    unsynced, logs = weighted_loss(pred, b, 1.0, 0.0, 0.0, property_weight=2.0, sync=False)
    assert float(unsynced) == pytest.approx(5.0) and torch.is_tensor(logs["property_mse"])
    none, logs = weighted_loss(pred, b, 1.0, 0.0, 0.0)
    assert float(none) == 0.0 and "property_mse" not in logs


def test_property_loss_per_head_and_vector_properties():
    """Several heads sum their own property terms; vector properties average over their components."""
    graphs = []
    for k, h in enumerate((0, 1, 1)):
        g = structure_to_graph(_s(seed=k, properties=[float(k), 0.0], head=h), 3.0)
        graphs.append(g)
    b = collate(graphs)
    pred = {"properties": torch.zeros(3, 2)}
    loss, logs = weighted_loss(pred, b, 1.0, 0.0, 0.0, property_weight=1.0)
    assert logs["0/property_mse"] == pytest.approx(0.0)
    assert logs["1/property_mse"] == pytest.approx((1.0 + 4.0) / 4)
    assert float(loss) == pytest.approx(1.25)


def test_prediction_pairs_score_labelled_properties():
    """The metrics pair the labelled properties only."""
    b = collate([structure_to_graph(_s(properties=1.0), 3.0), structure_to_graph(_s(seed=1), 3.0)])
    pairs = prediction_pairs({"properties": torch.tensor([1.5, 9.0])}, b, ["properties"])
    p, r = pairs["properties"]
    assert p.tolist() == [1.5] and r.tolist() == [1.0]
    full = collate([structure_to_graph(_s(properties=1.0), 3.0)])
    p, _ = prediction_pairs({"properties": torch.tensor([2.0])}, full, ["properties"])["properties"]
    assert p.tolist() == [2.0]


class _PropertyOnly(nn.Module):
    """A model with a property output and no energy."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(()))

    def forward(self, data):
        return {"properties": self.w * data.pos.sum(-1)[: data.num_graphs]}


def test_force_wrapper_passes_a_model_without_energy_through():
    """ForceStressOutput returns the outputs of a property model as they are, even when forces are asked."""
    g = structure_to_graph(_s(properties=1.0), 3.0)
    out = ForceStressOutput(_PropertyOnly(), compute_forces=True, compute_stress=True)(g)
    assert set(out) == {"properties"}


def test_disordered_sites_in_graphs_batches_and_subsets():
    """Per-site occupancies become padded species and occupancy rows; ordered structures join as occupancy 1."""
    s = _s(n=3, occupancies=[{"Al": 0.25, "Ga": 0.75}, {7: 1.0}, {"O": 0.9}])
    del s["atomic_numbers"]
    g = structure_to_graph(s, 3.0)
    assert g.atomic_numbers.tolist() == [31, 7, 8]
    assert g.site_species.tolist() == [[13, 31], [7, 0], [8, 0]]
    expected = torch.tensor([[0.25, 0.75], [1.0, 0.0], [0.9, 0.0]], dtype=g.site_occupancy.dtype)
    assert torch.equal(g.site_occupancy, expected)
    ordered = structure_to_graph(_s(n=2, seed=1), 3.0)
    b = collate([ordered, g])
    assert b.site_species.shape == (5, 2) and b.site_species[:2, 0].tolist() == [6, 6]
    assert b.site_occupancy[:2].tolist() == [[1.0, 0.0], [1.0, 0.0]]
    sub = b.subset(torch.tensor([False, True]))
    assert torch.equal(sub.site_species, g.site_species)
    assert collate([ordered]).site_species is None
    with pytest.raises(ValueError, match="sites"):
        structure_to_graph(_s(n=3, occupancies=[{"Al": 1.0}]), 3.0)
    with pytest.raises(ValueError, match="at most 1"):
        structure_to_graph(_s(n=1, occupancies=[{"Al": 0.6, "Ga": 0.6}]), 3.0)
    with pytest.raises(ValueError, match="non-negative"):
        structure_to_graph(_s(n=1, occupancies=[{"Al": -0.1}]), 3.0)
    bare = _s(n=1)
    del bare["atomic_numbers"]
    with pytest.raises(KeyError, match="atomic_numbers"):
        structure_to_graph(bare, 3.0)
