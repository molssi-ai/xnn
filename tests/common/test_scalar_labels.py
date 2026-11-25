"""Dipole-magnitude and ionization-potential labels: graphs, batches, subsets, the ASE reader, the loss terms
and the metrics."""
import numpy as np
import pytest
import torch

from xnn.common.benchmark.metrics import prediction_pairs
from xnn.common.data import collate, structure_to_graph
from xnn.common.train.losses import LOSS_TERMS, weighted_loss


@pytest.fixture(autouse=True)
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _s(n=3, seed=0, **extra):
    rng = np.random.default_rng(seed)
    return {"pos": rng.uniform(0, 3, (n, 3)), "atomic_numbers": [8] + [1] * (n - 1), **extra}


def _batch():
    """Three structures: a dipole magnitude only, an ionization potential only, both."""
    return collate([structure_to_graph(_s(seed=0, dipole_magnitude=1.8), 3.0),
                    structure_to_graph(_s(seed=1, ionization_potential=12.6), 3.0),
                    structure_to_graph(_s(seed=2, dipole_magnitude=0.5, ionization_potential=9.2), 3.0)])


def test_labels_in_graphs_batches_and_subsets():
    """The scalar labels become ``(1,)`` per graph, zeros with a mask where a batch lacks them, and keep their
    rows in a subset."""
    g = structure_to_graph(_s(dipole_magnitude=1.8, ionization_potential=12.6), 3.0)
    assert g.dipole_magnitude.tolist() == [1.8] and g.ionization_potential.tolist() == [12.6]
    b = _batch()
    assert b.dipole_magnitude.tolist() == [1.8, 0.0, 0.5] and b.ionization_potential.tolist() == [0.0, 12.6, 9.2]
    assert b.dipole_magnitude_mask.tolist() == [True, False, True]
    assert b.ionization_potential_mask.tolist() == [False, True, True]
    sub = b.subset(torch.tensor([False, True, True]))
    assert sub.ionization_potential.tolist() == [12.6, 9.2] and sub.dipole_magnitude_mask.tolist() == [False, True]
    both = collate([structure_to_graph(_s(ionization_potential=10.0), 3.0)] * 2)
    assert both.ionization_potential_mask is None and both.dipole_magnitude is None


def test_ase_reader_takes_the_labels():
    """``atoms.info`` keys of the same names become the labels."""
    from ase import Atoms

    from xnn.common.data.ase_io import atoms_to_structure
    atoms = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]],
                  info={"dipole_magnitude": 1.85, "ionization_potential": 12.62})
    d = atoms_to_structure(atoms)
    assert d["dipole_magnitude"] == 1.85 and d["ionization_potential"] == 12.62


def test_dipole_magnitude_and_ionization_terms():
    """Each term is the mean squared error over its labelled structures: the length of the predicted dipole
    against the magnitudes, the predicted ionization potential against the labels."""
    b = _batch()
    dipole = torch.tensor([[1.0, 1.0, 0.5], [0.2, 0.1, 0.0], [0.3, 0.0, 0.4]], requires_grad=True)
    ip = torch.tensor([11.0, 12.0, 10.0], requires_grad=True)
    pred = {"energy": torch.zeros(3), "dipole": dipole, "ionization_potential": ip}
    _, plain = weighted_loss(pred, b, 0.0, 0.0, 0.0)
    assert "dipole_magnitude_mse" not in plain and "ionization_mse" not in plain
    loss, logs = weighted_loss(pred, b, 0.0, 0.0, 0.0, dipole_magnitude_weight=2.0, ionization_weight=0.5)
    mu = dipole.detach().norm(dim=-1).tolist()
    assert logs["dipole_magnitude_mse"] == pytest.approx(0.5 * ((mu[0] - 1.8) ** 2 + (mu[2] - 0.5) ** 2), rel=1e-12)
    assert logs["ionization_mse"] == pytest.approx(0.5 * ((12.0 - 12.6) ** 2 + (10.0 - 9.2) ** 2), rel=1e-12)
    assert float(loss) == pytest.approx(2.0 * logs["dipole_magnitude_mse"] + 0.5 * logs["ionization_mse"], rel=1e-12)
    loss.backward()
    assert float(ip.grad[0]) == 0.0 and float(dipole.grad[1].abs().sum()) == 0.0
    counts, parts = {}, {}
    weighted_loss(pred, b, 0.0, 0.0, 0.0, dipole_magnitude_weight=2.0, ionization_weight=0.5, counts=counts,
                  parts=parts)
    assert counts["dipole_magnitude_mse"][0] == 2.0 and counts["ionization_mse"] == (2.0, 0.5)
    assert float(parts["ionization_mse"]) == pytest.approx(0.5 * logs["ionization_mse"], rel=1e-12)
    assert LOSS_TERMS["dipole_magnitude_mse"] == "dipole_magnitude_weight"
    assert LOSS_TERMS["ionization_mse"] == "ionization_weight"


def test_zero_dipole_has_a_finite_gradient():
    """A zero predicted dipole has length 0 and a zero (not NaN) gradient."""
    b = collate([structure_to_graph(_s(dipole_magnitude=1.0), 3.0)])
    dipole = torch.zeros(1, 3, requires_grad=True)
    loss, logs = weighted_loss({"energy": torch.zeros(1), "dipole": dipole}, b, 0.0, 0.0, 0.0,
                               dipole_magnitude_weight=1.0)
    loss.backward()
    assert logs["dipole_magnitude_mse"] == pytest.approx(1.0) and torch.equal(dipole.grad, torch.zeros(1, 3))


def test_metrics_score_the_labelled_structures():
    """The metrics pair the dipole's length and the ionization potential of the labelled structures."""
    b = _batch()
    out = {"dipole": torch.tensor([[1.0, 0, 0], [0, 2.0, 0], [0, 0, 3.0]]),
           "ionization_potential": torch.tensor([1.0, 2.0, 3.0])}
    pairs = prediction_pairs(out, b, ["dipole_magnitude", "ionization_potential"])
    assert pairs["dipole_magnitude"][0].tolist() == [1.0, 3.0] and pairs["dipole_magnitude"][1].tolist() == [1.8, 0.5]
    assert pairs["ionization_potential"][0].tolist() == [2.0, 3.0]
    assert pairs["ionization_potential"][1].tolist() == [12.6, 9.2]
