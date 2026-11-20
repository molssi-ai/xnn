"""Reaction-energy loss (xnn.common.train.reactions) and its term in the trainer.

Reaction sets resolve their species by set and name, keep references in eV, give the stoichiometry of any subset
with only the species it needs, draw shuffled mini-batches that cover every reaction once, and score a weighted,
scaled (WTMAD-2-like) squared or Huber error; the trainer adds the term to its training loss and validation
metrics, also with learned loss weights, and refuses settings it cannot run.
"""
import numpy as np
import pytest
import torch

from xnn.common.config import from_dict
from xnn.common.data import AtomicDataset
from xnn.common.data.hub.units import KCAL_MOL_TO_EV
from xnn.common.train import ReactionSet, Trainer, reaction_energy_loss

CUTOFF = 3.0


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _species(name, z, seed, subset=None):
    rng = np.random.default_rng(seed)
    s = {"name": name, "pos": rng.uniform(0.0, 2.5, (len(z), 3)), "atomic_numbers": z}
    if subset is not None:
        s["subset"] = subset
    return s


STRUCTURES = [_species("A", [1, 1], 0), _species("B", [8], 1), _species("C", [8, 1, 1], 2),
              _species("A", [1, 8], 3, subset="X")]
RECORDS = [{"species": ["C", "A", "B"], "stoichiometry": [1, -1, -1], "reference": -2.0},
           {"subset": "X", "species": ["A", "B"], "stoichiometry": [1, -1], "reference": 0.5, "weight": 2.0},
           {"species": ["C", "B"], "stoichiometry": [1, -2], "reference": 1.0, "scale": 3.0},
           {"species": ["A"], "stoichiometry": [2], "reference": -1.0}]


def test_species_keys_units_and_bad_records():
    """Species resolve under the reaction's set first (else the set-less ones), the references are kept in eV,
    and missing species, coefficient counts and units are reported."""
    rs = ReactionSet(STRUCTURES, RECORDS, units="kcal/mol")
    assert len(rs) == 4
    assert rs.members[0] == ([2, 0, 1], [1.0, -1.0, -1.0])
    assert rs.members[1][0] == [3, 1]
    assert torch.allclose(rs.reference, torch.tensor([-2.0, 0.5, 1.0, -1.0]) * KCAL_MOL_TO_EV)
    assert rs.labels[1] == ("X", 1) and rs.weight.tolist() == [1.0, 2.0, 1.0, 1.0]
    S = rs.stoichiometry()
    assert S[3].tolist() == [2.0, 0.0, 0.0, 0.0] and S[2].tolist() == [0.0, -2.0, 1.0, 0.0]
    with pytest.raises(KeyError, match="'D'"):
        ReactionSet(STRUCTURES, [{"species": ["D"], "stoichiometry": [1], "reference": 0.0}])
    with pytest.raises(ValueError, match="coefficients"):
        ReactionSet(STRUCTURES, [{"species": ["A"], "stoichiometry": [1, 1], "reference": 0.0}])
    with pytest.raises(ValueError, match="units"):
        ReactionSet(STRUCTURES, RECORDS, units="kJ/mol")


def test_loss_is_the_weighted_mean_of_scaled_squared_errors():
    """The loss equals the weighted mean of the squared (or Huber) scaled errors; the errors are unscaled."""
    rs = ReactionSet(STRUCTURES, RECORDS)
    e = torch.tensor([0.3, -1.2, 0.7, 2.0], requires_grad=True)
    dE = torch.stack([e[2] - e[0] - e[1], e[3] - e[1], e[2] - 2 * e[1], 2 * e[0]])
    err = dE - torch.tensor([-2.0, 0.5, 1.0, -1.0])
    scaled = err * torch.tensor([1.0, 1.0, 3.0, 1.0])
    w = torch.tensor([1.0, 2.0, 1.0, 1.0])
    assert torch.allclose(rs.loss(e), (w * scaled ** 2).sum() / w.sum())
    assert torch.allclose(rs.errors(e), err.detach())
    huber = torch.where(scaled.abs() <= 1.0, scaled ** 2, 2.0 * scaled.abs() - 1.0)
    assert torch.allclose(rs.loss(e, huber_delta=1.0), (w * huber).sum() / w.sum())
    plain = reaction_energy_loss(e, rs.stoichiometry(), rs.reference)
    assert torch.allclose(plain, (err ** 2).mean())
    (g,) = torch.autograd.grad(rs.loss(e), e)
    assert g.shape == e.shape and bool(g.abs().sum() > 0)


def test_subsets_and_batches():
    """A subset keeps the species its reactions need in the order met (and their graphs); the batches of one
    pass cover every reaction once, in a seeded order."""
    rs = ReactionSet(STRUCTURES, RECORDS)
    graphs = rs.graphs(CUTOFF)
    assert rs.graphs(CUTOFF) is graphs and len(graphs) == 4
    sub = rs.subset([2, 1])
    assert [s["name"] for s in sub.structures] == ["C", "B", "A"]
    assert sub.members == [([0, 1], [1.0, -2.0]), ([2, 1], [1.0, -1.0])]
    assert sub.graphs(CUTOFF)[1] is graphs[1]
    assert sub.labels == [rs.labels[2], rs.labels[1]] and torch.equal(sub.scale, rs.scale[[2, 1]])
    seen = [lab for b in rs.batches(3, seed=5) for lab in b.labels]
    assert sorted(seen, key=str) == sorted(rs.labels, key=str)
    assert [b.labels for b in rs.batches(3, seed=5)] == [b.labels for b in rs.batches(3, seed=5)]
    assert [lab for b in rs.batches(3, shuffle=False) for lab in b.labels] == rs.labels


def test_gmtkn55_records_and_wtmad2_scale():
    """The records of the GMTKN55 dataset (kcal/mol) load; WTMAD-2 scaling divides 56.84 by each set's mean
    absolute reference."""
    data = {"structures": [dict(s, subset="S1") for s in STRUCTURES[:3]] + [dict(STRUCTURES[0], subset="S2")],
            "reactions": [{"subset": "S1", "index": 1, "species": ["C", "A"], "stoichiometry": [1, -1],
                           "reference": 2.0},
                          {"subset": "S1", "index": 2, "species": ["B"], "stoichiometry": [1], "reference": -6.0},
                          {"subset": "S2", "index": 1, "species": ["A"], "stoichiometry": [1], "reference": 10.0}]}
    plain = ReactionSet.from_gmtkn55(data)
    assert torch.allclose(plain.reference, torch.tensor([2.0, -6.0, 10.0]) * KCAL_MOL_TO_EV)
    assert plain.scale.tolist() == [1.0, 1.0, 1.0]
    scaled = ReactionSet.from_gmtkn55(data, wtmad2=True)
    assert torch.allclose(scaled.scale, torch.tensor([56.84 / 4.0, 56.84 / 4.0, 56.84 / 10.0]))
    assert plain.members[2][0] == [3]


def _structures(n, seed):
    rng = np.random.default_rng(seed)
    return [{"pos": rng.uniform(0.0, 3.0, (3, 3)), "atomic_numbers": [1, 8, 1], "energy": float(rng.normal()),
             "forces": rng.normal(size=(3, 3))} for _ in range(n)]


def _cfg(tmp_path, **optim):
    return from_dict({"model": {"name": "schnet", "n_features": 8, "n_interactions": 1, "cutoff": CUTOFF},
                      "optim": {"epochs": 2, "lr": 2e-2, "scheduler": "none", **optim},
                      "data": {"batch_size": 2, "val_fraction": 0.0},
                      "trainer": {"progress_bar": False}, "device": "cpu", "output_dir": str(tmp_path)})


def test_trainer_adds_the_reaction_term(tmp_path):
    """A training step adds the weighted reaction loss of the next reaction batch to its loss (the dataset loss
    is rebuilt with it); validation scores the validation reactions and adds the term to its loss; learned loss
    weights take the term as well."""
    rs = ReactionSet(STRUCTURES, RECORDS)
    train, val = AtomicDataset(_structures(4, 0), CUTOFF), AtomicDataset(_structures(2, 1), CUTOFF)
    trainer = Trainer(_cfg(tmp_path, reaction_weight=0.5, reaction_batch_size=3), train, val, logger=False,
                      reactions=rs, val_reactions=rs)
    data = next(iter(trainer.train_loader))
    logs, counts = trainer._step(data, True)
    assert counts["reaction_mse"] == (3.0, 0.5)
    rebuilt = sum(w * logs[k] for k, (_, w) in counts.items() if w is not None and k != "prior")
    assert logs["loss"] == pytest.approx(rebuilt, rel=1e-10)
    out = trainer.fit()
    assert "reaction_mse" in out["train"]
    val_metrics = out["val"]
    assert {"reaction_mse", "reaction_mae", "reaction_rmse"} <= set(val_metrics)
    with torch.no_grad():
        energies = rs.species_energies(trainer._potential(False), CUTOFF)
    assert val_metrics["reaction_mae"] == pytest.approx(float(rs.errors(energies).abs().mean()), rel=1e-10)

    learned = Trainer(_cfg(tmp_path / "u", reaction_weight=1.0, loss_weighting="uncertainty"), train,
                      logger=False, reactions=rs)
    assert "reaction_mse" in learned.loss_weights.sigmas()
    logs, _ = learned._step(next(iter(learned.train_loader)), True)
    assert "sigma_reaction_mse" in logs

    scored = Trainer(_cfg(tmp_path / "v"), train, logger=False, val_reactions=rs)
    metrics = scored._reaction_metrics({})
    assert "loss" not in metrics and metrics["reaction_rmse"] >= metrics["reaction_mae"]


def test_trainer_refuses_what_it_cannot_run(tmp_path):
    """A reaction weight needs reactions and a positive batch size, and runs only in ordinary single-process
    steps."""
    train = AtomicDataset(_structures(2, 0), CUTOFF)
    rs = ReactionSet(STRUCTURES, RECORDS)
    with pytest.raises(ValueError, match="needs reactions"):
        Trainer(_cfg(tmp_path, reaction_weight=1.0), train, logger=False)
    with pytest.raises(ValueError, match="reaction_batch_size"):
        Trainer(_cfg(tmp_path, reaction_weight=1.0, reaction_batch_size=0), train, logger=False, reactions=rs)
    trainer = Trainer(_cfg(tmp_path, reaction_weight=1.0), train, logger=False, reactions=rs)
    trainer._graph_mode = True
    with pytest.raises(ValueError, match="single-process"):
        trainer._setup_reactions(rs, None)
