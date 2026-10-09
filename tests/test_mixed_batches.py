"""Batches that mix molecules and periodic cells, and labelled and unlabelled structures.

``collate`` used to keep a batch's cells only when every structure had one, so a
batch mixing clusters and cells silently lost the periodic images (wrong edge
lengths across the boundary) and the stress term. These tests evaluate mixed
batches of uneven size against the same structures one at a time, for a bare
model and for its long-range (LES) and dispersion (D4) wrappers, and check that
partially labelled batches train and score on their labelled structures only.
"""
import numpy as np
import pytest
import torch

from xnn.common.benchmark.metrics import collect_predictions
from xnn.common.config import from_dict
from xnn.common.data import collate, structure_to_graph
from xnn.common.models import ForceStressOutput, build_model
from xnn.common.train.losses import weighted_loss

BOX = 6.0


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _structure(n, seed, periodic, labels=("energy", "forces", "stress")):
    rng = np.random.default_rng(seed)
    s = {"pos": rng.uniform(0.0, BOX - 0.5, (n, 3)), "atomic_numbers": ([1, 8] * n)[:n]}
    if periodic:
        s["cell"] = np.eye(3) * BOX + rng.normal(scale=0.1, size=(3, 3))
        s["pbc"] = [True, True, True]
    if "energy" in labels:
        s["energy"] = float(rng.normal())
    if "forces" in labels:
        s["forces"] = rng.normal(size=(n, 3))
    if "stress" in labels and periodic:
        s["stress"] = rng.normal(size=(3, 3)) * 1e-2
    return s


# uneven sizes, clusters and cells interleaved
MIXED = [(7, 0, False), (10, 1, True), (6, 2, False), (9, 3, True), (5, 4, False)]


def _graphs(cutoff, spec=MIXED, **kw):
    return [structure_to_graph(_structure(n, seed, per, **kw), cutoff) for n, seed, per in spec]


def _model(extra):
    cfg = from_dict({"model": {"name": "mace", "cutoff": 4.0, "n_interactions": 2, "n_rbf": 6,
                               "n_features": 8, "extra": {"species": [1, 8], "l_max": 2, **extra}}})
    torch.manual_seed(0)
    return ForceStressOutput(build_model(cfg.model), compute_stress=True).eval()


def test_collate_keeps_cells_and_image_shifts():
    graphs = _graphs(4.0)
    batch = collate(graphs)
    assert batch.cell is not None and batch.cell.shape == (len(graphs), 3, 3)
    periodic = torch.tensor([per for _, _, per in MIXED])
    assert torch.equal(batch.pbc.any(-1), periodic)
    assert torch.equal(batch.cell[~periodic], torch.zeros_like(batch.cell[~periodic]))
    # every edge keeps the length it has in its own structure
    lengths = torch.cat([g.edge_vectors().norm(dim=-1) for g in graphs])
    assert torch.allclose(batch.edge_vectors().norm(dim=-1), lengths, rtol=0, atol=1e-12)
    # the batch does cross the boundary, so the test means something
    assert any(bool((g.cell_shifts != 0).any()) for g in graphs if g.cell is not None)


@pytest.mark.parametrize("extra", [
    {},
    {"long_range": {"n_channels": 2, "sigma": 1.0, "dl": 2.0}},
    {"dispersion": dict(cutoff_pair=7.0, cutoff_triple=5.0, cutoff_cn=6.0,
                        cutoff_eeq_cn=6.0, cutoff_eeq=7.0)},
], ids=["mace", "mace+les", "mace+d4"])
def test_mixed_batch_matches_structures_one_at_a_time(extra):
    model = _model(extra)
    cutoff = float(model.model.cutoff)
    graphs = _graphs(cutoff)
    out = model(collate(graphs))
    offset = 0
    for b, g in enumerate(graphs):
        one = model(g)
        n = g.num_nodes
        assert torch.allclose(out["energy"][b], one["energy"][0], rtol=0, atol=1e-10)
        assert torch.allclose(out["forces"][offset:offset + n], one["forces"], rtol=0, atol=1e-10)
        if g.cell is not None:
            assert torch.allclose(out["stress"][b], one["stress"][0], rtol=0, atol=1e-10)
            assert float(one["stress"].abs().max()) > 0
        else:
            assert torch.equal(out["stress"][b], torch.zeros(3, 3))
        offset += n


def test_partial_labels_get_masks_and_fillers():
    specs = [_structure(6, 0, False, labels=("energy",)),            # no forces
             _structure(8, 1, True),                                  # forces + stress
             _structure(5, 2, False),                                 # forces, no stress
             {**_structure(7, 3, True, labels=("energy", "forces")), "total_charge": -1.0,
              "weight": 2.0}]                                         # cell, no stress
    batch = collate([structure_to_graph(s, 4.0) for s in specs])
    assert batch.forces_mask.tolist() == [False, True, True, True]
    assert batch.stress_mask.tolist() == [False, True, False, False]
    assert torch.equal(batch.forces[:6], torch.zeros(6, 3))
    assert batch.total_charge.tolist() == [0.0, 0.0, 0.0, -1.0]
    assert batch.weight.tolist() == [1.0, 1.0, 1.0, 2.0]
    # a fully labelled batch carries no masks
    full = collate([structure_to_graph(_structure(6, s, True), 4.0) for s in range(3)])
    assert full.forces_mask is None and full.stress_mask is None
    # tensorial labels of some structures: fillers, masks and the loss on the labelled ones
    one = {**_structure(6, 0, False), "dipole": [0.1, 0.2, 0.3],
           "polarizability": np.eye(3) * 2.0}
    mixed = collate([structure_to_graph(one, 4.0), structure_to_graph(_structure(5, 1, False), 4.0)])
    assert mixed.dipole_mask.tolist() == [True, False]
    assert mixed.polarizability_mask.tolist() == [True, False]
    assert torch.equal(mixed.dipole[1], torch.zeros(3))
    pred = {"energy": mixed.energy, "forces": mixed.forces,
            "dipole": torch.ones(2, 3), "polarizability": torch.ones(2, 3, 3)}
    _, logs = weighted_loss(pred, mixed, 1.0, 0.0, 0.0, dipole_weight=1.0, polarizability_weight=1.0)
    assert logs["dipole_mse"] == pytest.approx(float(((1 - mixed.dipole[0]) ** 2).mean()), rel=1e-12)
    assert logs["polarizability_mse"] == pytest.approx(float(((1 - mixed.polarizability[0]) ** 2).mean()), rel=1e-12)
    sub = mixed.subset(torch.tensor([True, False]))
    assert sub.dipole.shape == (1, 3) and sub.dipole_mask.tolist() == [True]


def test_loss_uses_only_the_labelled_structures():
    model = _model({})
    specs = [_structure(6, 0, False, labels=("energy",)), _structure(8, 1, True),
             _structure(5, 2, False), _structure(7, 3, True, labels=("energy", "forces"))]
    graphs = [structure_to_graph(s, 4.0) for s in specs]
    batch = collate(graphs)
    pred = model(batch)
    _, logs = weighted_loss(pred, batch, 1.0, 1.0, 1.0)
    # forces: atoms of structures 1-3; stress: structure 1
    atoms = batch.forces_mask[batch.batch]
    f_ref = float(((pred["forces"][atoms] - batch.forces[atoms]) ** 2).mean())
    s_ref = float(((pred["stress"][1] - batch.stress[1]) ** 2).mean())
    assert logs["force_mse"] == pytest.approx(f_ref, rel=1e-12)
    assert logs["stress_mse"] == pytest.approx(s_ref, rel=1e-12)
    # the same as a batch of only the labelled structures
    _, only = weighted_loss(model(collate(graphs[1:2])), collate(graphs[1:2]), 1.0, 1.0, 1.0)
    assert logs["stress_mse"] == pytest.approx(only["stress_mse"], rel=1e-10)
    # a batch with no stress label at all skips the term
    none = collate([graphs[0], graphs[2]])
    _, logs_none = weighted_loss(model(none), none, 1.0, 1.0, 1.0)
    assert "stress_mse" not in logs_none and "force_mse" in logs_none


def test_metrics_score_only_the_labelled_structures():
    model = _model({})
    specs = [_structure(6, 0, False, labels=("energy",)), _structure(8, 1, True)]
    batch = collate([structure_to_graph(s, 4.0) for s in specs])
    pairs = collect_predictions(model, [batch], "cpu", ["energy", "forces", "stress"])
    assert pairs["forces"][0].numel() == 8 * 3
    assert pairs["stress"][0].numel() == 9
    assert pairs["energy"][0].numel() == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_mixed_batch_built_on_the_gpu():
    model = _model({}).cuda()
    cutoff = float(model.model.cutoff)
    graphs = [structure_to_graph(_structure(n, seed, per), cutoff, device="cuda")
              for n, seed, per in MIXED]
    out = model(collate(graphs))
    for b, g in enumerate(graphs):
        one = model(g)
        assert torch.allclose(out["energy"][b], one["energy"][0], rtol=0, atol=1e-10)
        if g.cell is not None:
            assert torch.allclose(out["stress"][b], one["stress"][0], rtol=0, atol=1e-10)
        else:
            assert torch.equal(out["stress"][b].cpu(), torch.zeros(3, 3))
