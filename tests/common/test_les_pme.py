"""The particle-mesh backend of the LES reciprocal sum."""
import numpy as np
import pytest
import torch

from xnn.common.config import from_dict
from xnn.common.data import structure_to_graph
from xnn.common.models import EwaldSummation, ForceStressOutput, LatentEwald, build_model
from xnn.common.models.pme import mesh_wavevectors, pme_reciprocal, pme_weighted_sum


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _system(n=30, triclinic=False, seed=0):
    rng = np.random.default_rng(seed)
    pos = torch.tensor(rng.uniform(0, 8, (n, 3)))
    q = torch.tensor(rng.normal(size=(n, 2)))
    cell = (torch.tensor([[8.0, 0, 0], [1.5, 7.6, 0], [-0.9, 1.1, 8.4]])
            if triclinic else torch.eye(3) * 8.0)
    return pos, q, cell


def test_mesh_wavevectors_match_the_cell():
    cell = torch.tensor([[8.0, 0, 0], [1.5, 7.6, 0], [-0.9, 1.1, 8.4]])
    kvec, k2 = mesh_wavevectors(cell, [8, 4, 16])
    assert kvec.shape == (8, 4, 16, 3) and k2.shape == (8, 4, 16)
    # k . a_j = 2 pi m_j: the (1, 0, 0) vector dotted with the first lattice vector
    assert float(kvec[1, 0, 0] @ cell[0]) == pytest.approx(2 * np.pi, abs=1e-12)
    assert float(kvec[1, 0, 0] @ cell[1]) == pytest.approx(0.0, abs=1e-12)
    assert torch.allclose(k2, (kvec * kvec).sum(-1))


def test_weighted_mesh_sum_reproduces_the_structure_factors():
    """With a unit weight on one vector the mesh sum is |S(k)|^2 of that vector."""
    pos, q, cell = _system(12)
    mesh = [32, 32, 32]
    kvec, _ = mesh_wavevectors(cell, mesh)
    weight = torch.zeros(mesh)
    weight[2, 1, 0] = 1.0
    k = kvec[2, 1, 0]
    phase = pos @ k
    exact = ((torch.cos(phase)[:, None] * q).sum(0).square() + (torch.sin(phase)[:, None] * q).sum(0).square()).sum()
    got = pme_weighted_sum(q, pos, cell, mesh, 6, weight)
    assert float(got) == pytest.approx(float(exact), rel=1e-6)


def test_pme_reciprocal_is_unchanged_by_the_refactor():
    """The Coulomb PME of the charge models: per-atom energies sum to the
    direct reciprocal sum of the same wave vectors."""
    pos, q, cell = _system(20)
    q = q[:, 0]
    q = q - q.mean()
    alpha = 0.35
    node = pme_reciprocal(q, pos, cell, alpha, [32, 32, 32], 5)
    assert node.shape == (20,)
    kvec, k2 = mesh_wavevectors(cell, [32, 32, 32])
    inside = k2 > 0
    phase = pos @ kvec[inside].T
    s2 = (torch.cos(phase).T @ q).square() + (torch.sin(phase).T @ q).square()
    vol = float(torch.linalg.det(cell))
    exact = (2 * np.pi / vol) * (torch.exp(-k2[inside] / (4 * alpha ** 2)) / k2[inside] * s2).sum()
    assert float(node.sum()) == pytest.approx(float(exact), rel=1e-5)


@pytest.mark.parametrize("exponent", [1, 6])
@pytest.mark.parametrize("triclinic", [False, True])
def test_pme_matches_the_reciprocal_sum(exponent, triclinic):
    pos, q, cell = _system(40, triclinic)
    ref = EwaldSummation(dl=1.5, sigma=1.0, exponent=exponent)
    pme = EwaldSummation(dl=1.5, sigma=1.0, exponent=exponent, method="pme", pme_order=8, pme_oversample=2.0)

    def energy_and_grads(fn):
        p, c = pos.clone().requires_grad_(True), cell.clone().requires_grad_(True)
        e = fn(p, q, c)
        return (e.detach(),) + tuple(torch.autograd.grad(e, [p, c]))

    e_ref, f_ref, s_ref = energy_and_grads(ref.reciprocal)
    e_pme, f_pme, s_pme = energy_and_grads(pme.reciprocal_pme)
    assert float(e_pme) == pytest.approx(float(e_ref), rel=1e-6)
    assert torch.allclose(f_pme, f_ref, atol=1e-5 * float(f_ref.abs().max()))
    assert torch.allclose(s_pme, s_ref, atol=1e-5 * float(s_ref.abs().max()))


def test_pme_order_and_mesh_control_the_error():
    pos, q, cell = _system(40)
    e_ref = float(EwaldSummation(dl=1.5).reciprocal(pos, q, cell))
    coarse = EwaldSummation(dl=1.5, method="pme", pme_order=4, pme_oversample=1.0)
    fine = EwaldSummation(dl=1.5, method="pme", pme_order=8, pme_oversample=2.0)
    err_coarse = abs(float(coarse.reciprocal_pme(pos, q, cell)) - e_ref)
    err_fine = abs(float(fine.reciprocal_pme(pos, q, cell)) - e_ref)
    assert err_fine < 1e-3 * err_coarse
    assert fine.pme_mesh(cell) == [24, 24, 24] and coarse.pme_mesh(cell) == [12, 12, 12]
    from xnn.common.models.les import smooth_size
    assert [smooth_size(x) for x in (1, 8.0, 8.1, 17, 31, 49, 97)] == [8, 8, 9, 18, 32, 50, 100]


def test_model_option_and_config_hook():
    cfg = from_dict({"model": {"name": "schnet", "cutoff": 4.5, "n_interactions": 1, "n_rbf": 6,
                               "n_features": 8, "extra": {"long_range": {"n_channels": 2, "dl": 1.5,
                                                                          "method": "pme"}}}})
    torch.manual_seed(0)
    model = build_model(cfg.model)
    assert isinstance(model, LatentEwald) and model.ewald.method == "pme"
    kspace = build_model(from_dict({"model": {**cfg.model.__dict__,
                                              "extra": {"long_range": {"n_channels": 2, "dl": 1.5}}}}).model)
    kspace.load_state_dict(model.state_dict())
    rng = np.random.default_rng(1)
    s = {"pos": rng.uniform(0, 7, (24, 3)), "atomic_numbers": [1, 8] * 12, "cell": np.eye(3) * 7.0, "pbc": [True] * 3}
    a = ForceStressOutput(model, compute_stress=True)(structure_to_graph(s, 4.5))
    b = ForceStressOutput(kspace, compute_stress=True)(structure_to_graph(s, 4.5))
    assert float(a["energy_lr"]) == pytest.approx(float(b["energy_lr"]), rel=1e-6)
    assert torch.allclose(a["forces"], b["forces"], atol=1e-5 * float(b["forces"].abs().max()))
    assert torch.allclose(a["stress"], b["stress"], atol=1e-5 * float(b["stress"].abs().max()))
    # molecules never use a reciprocal sum, so the option is inert there
    m = {"pos": rng.uniform(0, 5, (10, 3)), "atomic_numbers": [1, 8] * 5}
    e_mesh = float(model(structure_to_graph(m, 4.5))["energy"])
    e_sum = float(kspace(structure_to_graph(m, 4.5))["energy"])
    assert e_mesh == pytest.approx(e_sum, abs=1e-12)
    with pytest.raises(ValueError, match="method"):
        EwaldSummation(method="fft")


def test_les_is_refused_on_a_model_with_its_own_electrostatics():
    base = {"name": "physnet", "cutoff": 4.5, "n_interactions": 1, "n_rbf": 6, "n_features": 8}
    with pytest.raises(ValueError, match="count them twice"):
        build_model(from_dict({"model": {**base, "extra": {"use_electrostatics": True, "long_range": True}}}).model)
    model = build_model(from_dict({"model": {**base,
                                             "extra": {"use_electrostatics": False, "long_range": True}}}).model)
    assert isinstance(model, LatentEwald)
