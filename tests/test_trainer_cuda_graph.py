"""Training steps replayed as one CUDA graph (``trainer.cuda_graph``): the padding, the sync-free
loss and the captured steps against ordinary ones."""
import numpy as np
import pytest
import torch

from xnn.common.config import from_dict
from xnn.common.data import AtomicDataset, collate, structure_to_graph
from xnn.common.models import ForceStressOutput
from xnn.common.train import Trainer
from xnn.common.train.graphed import PaddedCollate, PlannedDataset, capacity, strip_padding
from xnn.common.train.losses import weighted_loss
from xnn.gnn.models.dimenet import DimeNet, DimeNetPP
from xnn.gnn.models.gemnet import GemNet


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _structures(n=8, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        k = 4 + i % 3
        pos = rng.uniform(0.0, 2.5, (k, 3))
        out.append({"pos": pos, "atomic_numbers": ([1, 6, 8] * k)[:k],
                    "energy": float(rng.normal()), "forces": rng.normal(size=(k, 3))})
    return out


#: GemNet sizes (small), as in the GemNet tests
GEMNET = dict(n_features=12, n_edge_features=10, n_interactions=2, n_rbf=3, n_spherical=3,
              cutoff=3.0, int_cutoff=4.0, n_triplet_features=6, n_quadruplet_features=5,
              n_rbf_features=4, n_cbf_features=4, n_sbf_features=5, n_bilinear_triplet=7,
              n_bilinear_quadruplet=6, n_atom_residual=1)

MODELS = ["dimenet", "dimenet++", "gemnet-t", "gemnet-q"]


def _small(name="dimenet"):
    torch.manual_seed(0)
    if name.startswith("gemnet"):
        return GemNet(quadruplets=name == "gemnet-q", **GEMNET)
    common = dict(n_features=16, n_interactions=2, n_rbf=4, n_spherical=3, cutoff=5.0, n_output_layers=2)
    if name == "dimenet++":
        return DimeNetPP(n_triplet_features=8, n_basis_features=4, n_output_features=12, **common)
    return DimeNet(n_bilinear=4, **common)


@pytest.mark.parametrize("name", MODELS)
def test_padded_batch_is_inert(name):
    """The padding structure changes neither the real energies and forces nor the loss."""
    model = _small(name)
    graphs = [structure_to_graph(s, model.cutoff) for s in _structures()]
    planner = model.index_planner()
    planned = PlannedDataset(graphs, planner)
    cap = capacity(planner, planned, 3)
    # the batch plan concatenated from the structures' own plans
    padded = PaddedCollate(planner, cap)([planned[i] for i in range(3)])
    assert padded.num_nodes == cap.atoms and padded.edge_index.shape[1] == cap.edges
    triplets = padded.index_plan["t_out" if name.startswith("gemnet") else "kj"]
    assert triplets.shape[0] == cap.plan["triplets"]
    plain = collate(graphs[:3])
    wrapped = ForceStressOutput(model).train()
    out_pad, out = wrapped(padded), wrapped(plain)
    n = plain.num_nodes
    assert torch.allclose(out_pad["energy"][:3], out["energy"], rtol=0, atol=1e-12)
    assert torch.allclose(out_pad["forces"][:n], out["forces"], rtol=0, atol=1e-11)
    assert torch.isfinite(out_pad["forces"]).all()
    # an ordinary step takes the real structures back
    stripped = strip_padding(padded)
    assert stripped.num_nodes == n and stripped.edge_index.shape[1] == plain.edge_index.shape[1]
    out_s = wrapped(stripped)
    assert torch.equal(out_s["energy"], out["energy"]) and torch.equal(out_s["forces"], out["forces"])
    loss_pad, logs_pad = weighted_loss(out_pad, padded, 1.0, 10.0, 0.0, sync=False)
    loss, logs = weighted_loss(out, plain, 1.0, 10.0, 0.0)
    assert torch.allclose(loss_pad, loss, rtol=1e-13, atol=0)
    assert {k: float(v) for k, v in logs_pad.items()}.keys() == logs.keys()


def test_sync_free_loss_of_an_unlabelled_term_is_zero():
    """A term whose structures are all masked out is 0 without the host check (not 0 / 0)."""
    model = _small()
    g = structure_to_graph(_structures(1)[0], 5.0)
    batch = collate([g])
    batch.forces_mask = torch.zeros(1, dtype=torch.bool)
    out = ForceStressOutput(model).train()(batch)
    loss, logs = weighted_loss(out, batch, 0.0, 1.0, 0.0, sync=False)
    assert float(loss) == 0.0 and float(logs["force_mse"]) == 0.0


def _cfg(tmp_path, cuda_graph, model="dimenet"):
    if model.startswith("gemnet"):
        extra = {k: v for k, v in GEMNET.items()
                 if k not in ("n_features", "n_interactions", "n_rbf", "cutoff")}
        extra["quadruplets"] = model == "gemnet-q"
        spec = {"name": "gemnet", "cutoff": GEMNET["cutoff"], "n_features": GEMNET["n_features"],
                "n_interactions": 2, "n_rbf": GEMNET["n_rbf"], "extra": extra}
    else:
        extra = {"n_spherical": 3, "n_output_layers": 2}
        extra.update({"n_triplet_features": 8, "n_basis_features": 4, "n_output_features": 12}
                     if model == "dimenet++" else {"n_bilinear": 4})
        spec = {"name": model, "cutoff": 5.0, "n_features": 16, "n_interactions": 2, "n_rbf": 4,
                "extra": extra}
    return from_dict({
        "model": spec,
        "optim": {"epochs": 4, "lr": 1e-3, "energy_weight": 1.0, "force_weight": 10.0,
                  "scheduler": "cosine", "ema_decay": 0.9, "clip_grad": 10.0},
        "data": {"batch_size": 2, "val_fraction": 0.0},
        "trainer": {"loggers": [], "cuda_graph": cuda_graph},
        "device": "cuda", "seed": 3, "output_dir": str(tmp_path)})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("model", MODELS)
def test_captured_steps_match_ordinary_steps(tmp_path, model):
    """Training with captured steps (warm-up, capture, replays) ends at the ordinary run's weights."""
    ds = AtomicDataset(_structures(), _small(model).cutoff)
    weights = {}
    for cuda_graph in (False, True):
        torch.manual_seed(1)
        trainer = Trainer(_cfg(tmp_path / str(cuda_graph), cuda_graph, model), ds)
        trainer.fit()
        if cuda_graph:
            assert trainer._graphed is not None and trainer._graphed.graph is not None
        weights[cuda_graph] = {n: p.detach().clone() for n, p in trainer.module.named_parameters()}
    for name, p in weights[False].items():
        err = (weights[True][name] - p).abs().max() / p.abs().max().clamp_min(1e-30)
        assert err < 1e-10, (name, float(err))


def test_cuda_graph_needs_an_index_plan(tmp_path):
    """A model without an index plan (or a CPU run) cannot use captured steps."""
    cfg = _cfg(tmp_path, True)
    cfg.device = "cpu"
    with pytest.raises(ValueError, match="CUDA"):
        Trainer(cfg, AtomicDataset(_structures(), 5.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_trial_against_fast_kernels(tmp_path, monkeypatch):
    """Where the fast kernels take the batches, the first steps time ordinary steps on them
    against captured ones and training goes on with the faster; the weights are those of an
    ordinary run either way."""
    pytest.importorskip("triton")
    from xnn.common.models.fast import AutoPolicy
    from xnn.gnn.fast import dimenet as fast_dimenet
    monkeypatch.setattr(fast_dimenet, "AUTO_POLICY_TRAINING", AutoPolicy())
    ds = AtomicDataset(_structures(), 5.0)
    weights = {}
    for cuda_graph in (False, True):
        torch.manual_seed(1)
        trainer = Trainer(_cfg(tmp_path / str(cuda_graph), cuda_graph), ds)
        trainer.fit()
        if cuda_graph:
            assert trainer._trial is None                     # decided
        weights[cuda_graph] = {n: p.detach().clone() for n, p in trainer.module.named_parameters()}
    for name, p in weights[False].items():
        err = (weights[True][name] - p).abs().max() / p.abs().max().clamp_min(1e-30)
        assert err < 1e-9, (name, float(err))
