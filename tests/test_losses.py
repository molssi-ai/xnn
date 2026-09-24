"""Per-structure loss weights and Huber tails in :func:`weighted_loss`."""
import numpy as np
import pytest
import torch

from xnn.common.data import AtomicDataset, collate
from xnn.common.train import weighted_loss


@pytest.fixture(autouse=True)
def _pinned_dtype():
    """Pin the default dtype, and restore whatever the session had.

    Other test modules set a global default dtype (the D4 parity tests run in
    float64), and the values compared here are dtype-sensitive, so a module
    that inherits the ambient dtype passes alone and fails in a full run.
    """
    before = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(before)


def _structs(weights=None):
    """Two structures of different size, with targets and optional weights."""
    rng = np.random.default_rng(0)
    out = []
    for i, n in enumerate((3, 6)):
        s = {"atomic_numbers": np.array([8] + [1] * (n - 1)),
             "pos": rng.normal(scale=1.5, size=(n, 3)),
             "cell": None, "pbc": np.zeros(3, bool),
             "energy": float(rng.normal()),
             "forces": rng.normal(size=(n, 3))}
        if weights is not None:
            s["weight"] = float(weights[i])
        out.append(s)
    return out


def _batch(weights=None, cutoff=5.0):
    ds = AtomicDataset(_structs(weights), cutoff)
    return collate([ds[0], ds[1]])


def _pred(batch, seed=1):
    """Predictions offset from the targets, so every term is non-zero."""
    g = torch.Generator().manual_seed(seed)
    return {"energy": batch.energy + torch.randn(batch.energy.shape, generator=g),
            "forces": batch.forces + torch.randn(batch.forces.shape, generator=g)}


def test_weight_plumbs_through_graph_and_collate():
    """The per-structure weight survives to_graph and collate; absent by default."""
    assert _batch().weight is None
    b = _batch([2.0, 5.0])
    assert b.weight is not None
    torch.testing.assert_close(b.weight, torch.tensor([2.0, 5.0]))
    assert b.weight.shape == (b.num_graphs,)


def test_no_weights_is_bitwise_the_original_loss():
    """A dataset without weights must take the old code path exactly.

    This is the backward-compatibility guarantee: every fit trained before
    weights existed stays reproducible bit for bit.
    """
    b = _batch()
    assert b.weight is None
    pred = _pred(b)
    _, logs = weighted_loss(pred, b, 1.0, 10.0, 0.0)

    e_ref = float((((pred["energy"] - b.energy) / b.n_atoms) ** 2).mean())
    f_ref = float(((pred["forces"] - b.forces) ** 2).mean())
    assert logs["energy_mse"] == e_ref            # bitwise, not approx
    assert logs["force_mse"] == f_ref


def test_uniform_weights_reproduce_the_unweighted_loss():
    """Equal weights are a no-op whatever their common value.

    Only to round-off, not bitwise: the weighted path sums and divides where
    the plain path calls ``mean()``, and floating-point addition is not
    associative.
    """
    plain = _batch()
    ref, ref_logs = weighted_loss(_pred(plain), plain, 1.0, 10.0, 0.0)
    for value in (1.0, 0.25, 7.0):
        w = _batch([value, value])
        got, logs = weighted_loss(_pred(w), w, 1.0, 10.0, 0.0)
        torch.testing.assert_close(got, ref, rtol=1e-12, atol=0)
        assert logs["energy_mse"] == pytest.approx(ref_logs["energy_mse"], rel=1e-12)
        assert logs["force_mse"] == pytest.approx(ref_logs["force_mse"], rel=1e-12)


def test_weights_reweight_each_term_as_hand_computed():
    """Check the weighted means against the arithmetic, term by term."""
    b = _batch([1.0, 3.0])
    pred = _pred(b)
    _, logs = weighted_loss(pred, b, 1.0, 1.0, 0.0)

    w = b.weight
    e_err = ((pred["energy"] - b.energy) / b.n_atoms) ** 2
    expect_e = float((w * e_err).sum() / w.sum())
    assert logs["energy_mse"] == pytest.approx(expect_e, rel=1e-6)

    # the force term weights atoms, so the 6-atom structure carries its weight
    # over twice as many rows as the 3-atom one
    w_atom = w[b.batch]
    f_err = (pred["forces"] - b.forces) ** 2
    expect_f = float((w_atom[:, None] * f_err).sum() / (w_atom.sum() * 3))
    assert logs["force_mse"] == pytest.approx(expect_f, rel=1e-6)


def test_a_zero_weight_removes_a_structure_entirely():
    """Weighting a structure to zero equals not having it in the batch."""
    b = _batch([0.0, 1.0])
    pred = _pred(b)
    _, logs = weighted_loss(pred, b, 1.0, 1.0, 0.0)

    second = b.batch == 1
    e_err = ((pred["energy"][1] - b.energy[1]) / b.n_atoms[1]) ** 2
    f_err = ((pred["forces"][second] - b.forces[second]) ** 2).mean()
    assert logs["energy_mse"] == pytest.approx(float(e_err), rel=1e-6)
    assert logs["force_mse"] == pytest.approx(float(f_err), rel=1e-6)


def test_huber_matches_the_squared_error_below_delta():
    """Below the crossover the two coincide, so weights keep their meaning."""
    b = _batch()
    pred = _pred(b)
    _, mse = weighted_loss(pred, b, 1.0, 1.0, 0.0)
    _, huber = weighted_loss(pred, b, 1.0, 1.0, 0.0, huber_delta=1e6)
    assert huber["energy_mse"] == pytest.approx(mse["energy_mse"], rel=1e-9)
    assert huber["force_mse"] == pytest.approx(mse["force_mse"], rel=1e-9)


def test_huber_clips_large_residuals():
    """Beyond delta the term grows linearly, so it must fall below the square."""
    b = _batch()
    pred = _pred(b)
    _, mse = weighted_loss(pred, b, 1.0, 1.0, 0.0)
    _, huber = weighted_loss(pred, b, 1.0, 1.0, 0.0, huber_delta=0.05)
    assert huber["force_mse"] < mse["force_mse"]

    # exact form: delta**2 + 2*delta*(|d| - delta) for |d| > delta
    delta = 0.05
    d = (pred["forces"] - b.forces).abs()
    expect = torch.where(d <= delta, d ** 2, 2 * delta * d - delta ** 2).mean()
    assert huber["force_mse"] == pytest.approx(float(expect), rel=1e-6)


def test_per_term_delta_overrides_the_global_one():
    """Energies and forces differ in scale, so each term takes its own delta."""
    b = _batch()
    pred = _pred(b)
    _, both = weighted_loss(pred, b, 1.0, 1.0, 0.0, huber_delta=0.05)
    _, only_f = weighted_loss(pred, b, 1.0, 1.0, 0.0, huber_delta=0.05,
                              huber_delta_energy=0.0)
    _, plain = weighted_loss(pred, b, 1.0, 1.0, 0.0)
    assert only_f["force_mse"] == pytest.approx(both["force_mse"], rel=1e-9)
    assert only_f["energy_mse"] == pytest.approx(plain["energy_mse"], rel=1e-9)
    assert both["energy_mse"] != pytest.approx(plain["energy_mse"], rel=1e-9)


def test_weights_and_huber_compose():
    """The two features are independent and can be used together."""
    b = _batch([1.0, 3.0])
    pred = _pred(b)
    _, logs = weighted_loss(pred, b, 1.0, 1.0, 0.0, huber_delta=0.05)

    delta, w_atom = 0.05, b.weight[b.batch]
    d = (pred["forces"] - b.forces).abs()
    err = torch.where(d <= delta, d ** 2, 2 * delta * d - delta ** 2)
    expect = float((w_atom[:, None] * err).sum() / (w_atom.sum() * 3))
    assert logs["force_mse"] == pytest.approx(expect, rel=1e-6)


def test_gradients_still_flow_through_both_paths():
    """The loss must stay differentiable with weights and clipped tails on."""
    b = _batch([1.0, 3.0])
    pred = _pred(b)
    pred = {k: v.clone().requires_grad_(True) for k, v in pred.items()}
    loss, _ = weighted_loss(pred, b, 1.0, 10.0, 0.0, huber_delta=0.05)
    loss.backward()
    assert torch.isfinite(pred["energy"].grad).all()
    assert torch.isfinite(pred["forces"].grad).all()
    assert pred["forces"].grad.abs().sum() > 0
