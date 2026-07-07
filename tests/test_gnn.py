"""Tests for featurizers and the equivariant GNN models."""
import numpy as np
import pytest
import torch

from xnns.common.data import structure_to_graph
from xnns.dnn.featurizers import (
    RadialSymmetryFunctions, AngularSymmetryFunctions, AEV, build_triplets,
)
from xnns.common.models import build_model, ForceStressOutput, available_models
from xnns.common.config import from_dict

e3nn = pytest.importorskip("e3nn")  # GNN tests need e3nn

SPECIES = [1, 6, 8]


def _graph(n=6, cutoff=5.2, periodic=False):
    rng = np.random.default_rng(0)
    s = {"pos": rng.uniform(0, 4, (n, 3)),
         "atomic_numbers": ([1, 6, 8] * n)[:n]}
    if periodic:
        s["cell"] = np.eye(3) * 6.0
        s["pbc"] = [True, True, True]
    return structure_to_graph(s, cutoff)


def _proper_rotation(seed=1):
    rng = np.random.default_rng(seed)
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    return R


# --- featurizers (independently usable) ---

def test_radial_sf_shape():
    sf = RadialSymmetryFunctions(SPECIES, cutoff=5.0)
    out = sf(_graph())
    assert out.shape == (6, sf.output_dim)


def test_aev_has_angular_signal():
    aev = AEV(SPECIES)
    out = aev(_graph())
    angular = out[:, aev.radial.output_dim:]
    assert out.shape[1] == aev.output_dim
    assert float(angular.abs().sum()) > 0  # angular term is actually populated


def test_triplets():
    g = _graph(n=5)
    a, b, c = build_triplets(g.edge_index, g.num_nodes)
    assert a.shape == b.shape == c.shape


def test_radial_sf_rotation_invariant():
    sf = RadialSymmetryFunctions(SPECIES, cutoff=5.0)
    R = _proper_rotation()
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 4, (6, 3))
    z = [1, 6, 8, 1, 6, 8]
    d0 = sf(structure_to_graph({"pos": pos, "atomic_numbers": z}, 5.0))
    d1 = sf(structure_to_graph({"pos": pos @ R.T, "atomic_numbers": z}, 5.0))
    assert torch.allclose(d0, d1, atol=1e-4)


# --- equivariant GNNs ---

@pytest.mark.parametrize("name", ["nequip", "mace", "allegro", "cace"])
def test_gnn_equivariance(name):
    cfg = from_dict({"model": {"name": name, "n_features": 32, "n_interactions": 2,
                               "n_rbf": 8, "cutoff": 5.0,
                               "extra": {"species": SPECIES, "l_max": 2}}})
    model = ForceStressOutput(build_model(cfg.model))
    rng = np.random.default_rng(1)
    pos = rng.uniform(0, 4, (6, 3))
    z = [1, 6, 8, 1, 6, 8]
    R = _proper_rotation()
    Rt = torch.tensor(R, dtype=torch.get_default_dtype())

    o0 = model(structure_to_graph({"pos": pos, "atomic_numbers": z}, 5.0))
    o1 = model(structure_to_graph({"pos": pos @ R.T, "atomic_numbers": z}, 5.0))
    # energy invariant, forces co-rotate
    assert abs(float(o0["energy"].detach()) - float(o1["energy"].detach())) < 1e-4
    assert torch.allclose(o1["forces"].detach(), o0["forces"].detach() @ Rt.T, atol=1e-4)


@pytest.mark.parametrize("name", ["nequip", "mace", "cace"])
def test_gnn_periodic_stress(name):
    cfg = from_dict({"model": {"name": name, "n_features": 16, "n_interactions": 1,
                               "n_rbf": 8, "cutoff": 5.0,
                               "extra": {"species": SPECIES, "l_max": 1}}})
    model = ForceStressOutput(build_model(cfg.model), compute_stress=True)
    out = model(_graph(n=4, cutoff=5.0, periodic=True))
    assert out["stress"].shape == (1, 3, 3)


@pytest.mark.parametrize("name", ["nequip", "allegro"])
def test_upstream_key_translation(name):
    """Upstream NequIP/Allegro yaml spellings map onto the xnns core fields."""
    cfg = from_dict({"model": {"name": name, "r_max": 4.5, "num_layers": 3,
                               "species": SPECIES}})
    assert cfg.model.cutoff == 4.5
    assert cfg.model.n_interactions == 3
    assert cfg.data.cutoff == 4.5


def test_ani_registered_and_runs():
    assert "ani" in available_models()
    cfg = from_dict({"model": {"name": "ani", "cutoff": 5.2,
                               "extra": {"species": SPECIES}}})
    model = ForceStressOutput(build_model(cfg.model))
    out = model(_graph(cutoff=5.2))
    assert out["forces"].shape == (6, 3)
