"""The LES global charge solve: charge equilibration with the latent Ewald kernel."""
import math

import numpy as np
import pytest
import torch

from xnn.common.config import from_dict
from xnn.common.data import collate, structure_to_graph
from xnn.common.models import EwaldSummation, ForceStressOutput, LatentEwald, build_model
from xnn.common.models.charge_solve import charge_energy, coulomb_matrix, solve_charges
from xnn.common.models.les import LATENT_CHARGE_PER_E

SCHNET = {"name": "schnet", "cutoff": 4.5, "n_interactions": 1, "n_rbf": 6, "n_features": 8}


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _cluster(seed, n, charge=0.0, periodic=False, cutoff=4.5):
    rng = np.random.default_rng(seed)
    s = {"pos": rng.uniform(0, 5, (n, 3)), "atomic_numbers": ([1, 8, 11, 17] * n)[:n],
         "total_charge": charge}
    if periodic:
        s.update(cell=np.eye(3) * 6.0, pbc=[True] * 3)
    return structure_to_graph(s, cutoff)


def _energy_lr(ewald, pos, q, cell):
    return ewald.realspace(pos, q[:, None]) if cell is None else ewald.reciprocal(pos, q[:, None], cell)


# the Coulomb matrix and the solve against the Ewald energy itself

@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("remove_self", [False, True])
def test_coulomb_matrix_is_the_hessian_of_the_ewald_energy(periodic, remove_self):
    ewald = EwaldSummation(dl=1.5, sigma=1.0, remove_self_interaction=remove_self)
    g = _cluster(0, 7, periodic=periodic)
    pos, cell = g.pos, (g.cell[0] if periodic else None)
    gamma = coulomb_matrix(ewald, pos, cell)
    hess = torch.autograd.functional.hessian(lambda q: _energy_lr(ewald, pos, q, cell), torch.zeros(7))
    assert torch.allclose(gamma, hess, atol=1e-12)
    q = torch.randn(7)
    assert float(0.5 * q @ gamma @ q) == pytest.approx(float(_energy_lr(ewald, pos, q, cell)), abs=1e-12)


@pytest.mark.parametrize("periodic", [False, True])
def test_solve_is_stationary_and_conserves_charge(periodic):
    ewald = EwaldSummation(dl=1.5, sigma=1.0)
    g = _cluster(1, 9, periodic=periodic)
    gamma = coulomb_matrix(ewald, g.pos, g.cell[0] if periodic else None)
    chi, J = torch.randn(9), torch.rand(9) + 0.05
    target = torch.tensor(-1.0) * LATENT_CHARGE_PER_E
    q, mu = solve_charges(chi, J, gamma, target)
    assert float(q.sum()) == pytest.approx(float(target), abs=1e-12)
    grad = chi + J * q + gamma @ q               # dE/dq
    assert torch.allclose(grad, torch.full_like(grad, float(mu)), atol=1e-11)
    # the minimum: any neutral perturbation raises the energy
    def energy(x):
        return float((chi * x).sum() + 0.5 * (J * x * x).sum() + 0.5 * x @ gamma @ x)
    d = torch.randn(9)
    d = d - d.mean()
    assert energy(q + 1e-3 * d) > energy(q) and energy(q - 1e-3 * d) > energy(q)


def test_coupling_free_limit_is_the_net_charge_constraint():
    """gamma = 0 gives the constraint shift with weights 1/J (the inverse hardness)."""
    chi, J = torch.randn(12), torch.rand(12) + 0.1
    q, _ = solve_charges(chi, J, torch.zeros(12, 12), torch.tensor(2.5))
    q0, w = -chi / J, 1.0 / J
    expect = q0 - w * (q0.sum() - 2.5) / w.sum()
    assert torch.allclose(q, expect, atol=1e-12)


@pytest.mark.parametrize("periodic", [False, True])
def test_solve_matches_a_plain_dense_solve_with_its_forces_and_strain(periodic):
    """Charges, energy, forces and strain derivatives against a plain dense
    solve with the Hessian of the Ewald energy as the Coulomb matrix."""
    torch.manual_seed(2)
    ewald = EwaldSummation(dl=1.5, sigma=1.0)
    g = _cluster(8, 11, 0.0 if periodic else -1.0, periodic)
    chi, J = torch.randn(11), torch.rand(11) + 0.1
    target = torch.tensor(0.0 if periodic else -1.0) * LATENT_CHARGE_PER_E

    def oracle(pos, cell):
        gamma = torch.autograd.functional.hessian(lambda q: _energy_lr(ewald, pos, q, cell), torch.zeros(11),
                                                  create_graph=True)
        full = torch.cat([torch.cat([gamma + torch.diag(J), torch.ones(11, 1)], 1),
                          torch.cat([torch.ones(1, 11), torch.zeros(1, 1)], 1)], 0)
        sol = torch.linalg.solve(full, torch.cat([-chi, target.reshape(1)]))
        q = sol[:11]
        return q, charge_energy(chi, J, q).sum() + _energy_lr(ewald, pos, q, cell)

    def mine(pos, cell):
        q, _ = solve_charges(chi, J, coulomb_matrix(ewald, pos, cell), target)
        return q, charge_energy(chi, J, q).sum() + _energy_lr(ewald, pos, q, cell)

    results = {}
    for name, fn in (("oracle", oracle), ("mine", mine)):
        pos = g.pos.clone().requires_grad_(True)
        strain = torch.zeros(3, 3, requires_grad=True)
        sym = torch.eye(3) + 0.5 * (strain + strain.T)
        cell = g.cell[0] @ sym.T if periodic else None
        q, energy = fn(pos @ sym.T, cell)
        grads = torch.autograd.grad(energy, [pos, strain])
        results[name] = (q.detach(), float(energy), grads[0], grads[1])
    for mine_, ref in zip(results["mine"][:1] + results["mine"][2:], results["oracle"][:1] + results["oracle"][2:]):
        assert torch.allclose(mine_, ref, atol=1e-9)
    assert results["mine"][1] == pytest.approx(results["oracle"][1], abs=1e-10)
    # the oracle's own forces with the charges frozen: the charge response drops out
    pos = g.pos.clone().requires_grad_(True)
    q = results["oracle"][0]
    hf = -torch.autograd.grad(charge_energy(chi, J, q).sum() + _energy_lr(ewald, pos, q, g.cell[0] if periodic else None), pos)[0]
    assert torch.allclose(hf, -results["oracle"][2], atol=1e-9)


# the LatentEwald integration

def _model(seed=3, **options):
    torch.manual_seed(seed)
    base = build_model(from_dict({"model": SCHNET}).model)
    # dl = 1.9 keeps every k shell of a 6 A cell off the cutoff (finite strains)
    return LatentEwald(base, n_channels=3, charge_channel=1, dl=1.9, sigma=1.0, **options)


@pytest.mark.parametrize("periodic", [False, True])
def test_model_solve_conserves_charge_and_matches_single_structures(periodic):
    model = _model(charge_solve=True)
    batch = collate([_cluster(1, 7, -1.0, periodic), _cluster(2, 5, 0.0, periodic)])
    out = model(batch)
    q = out["latent_charges"]
    sums = torch.zeros(2, 3).index_add_(0, batch.batch, q)
    assert torch.allclose(sums[:, 1], torch.tensor([-1.0, 0.0]) * LATENT_CHARGE_PER_E, atol=1e-10)
    assert out["hardness"].shape == (12,) and bool((out["hardness"] > 0).all())
    assert out["chemical_potential"].shape == (2,) and out["energy_charge"].shape == (2,)
    assert torch.allclose(out["energy"], out["energy_sr"] + out["energy_lr"] + out["energy_charge"], atol=1e-12)
    single = model(_cluster(1, 7, -1.0, periodic))
    assert torch.allclose(single["latent_charges"], q[:7], atol=1e-10)
    assert float(single["energy"]) == pytest.approx(float(out["energy"][0]), abs=1e-10)
    # the per-atom energies add up to the structure energies
    node = torch.zeros(2).index_add_(0, batch.batch, out["node_energy"])
    assert torch.allclose(node, out["energy"], atol=1e-10)
    # off: the reference head, with the same weights, differs only in channel 1
    plain = _model()(_cluster(1, 7, -1.0, periodic))["latent_charges"]
    assert torch.allclose(plain[:, [0, 2]], single["latent_charges"][:, [0, 2]], atol=1e-12)
    assert not torch.allclose(plain[:, 1], single["latent_charges"][:, 1])


def test_model_forces_and_stress_are_the_energy_gradient():
    """Forces and stress equal central differences, cluster and cell."""
    for periodic, hardness in ((False, "element"), (True, "features")):
        model = _model(charge_solve=True, hardness=hardness)
        g = _cluster(4, 6, 1.0 if not periodic else 0.0, periodic)
        s = {"pos": g.pos.numpy(), "atomic_numbers": g.atomic_numbers.tolist(),
             "total_charge": 1.0 if not periodic else 0.0}
        if periodic:
            s.update(cell=g.cell[0].numpy(), pbc=[True] * 3)
        out = ForceStressOutput(model, compute_stress=periodic)(structure_to_graph(s, 4.5))
        h = 1e-5
        for atom, comp in ((0, 0), (3, 2)):
            plus, minus = np.array(s["pos"]), np.array(s["pos"])
            plus[atom, comp] += h
            minus[atom, comp] -= h
            e_p = float(model(structure_to_graph(dict(s, pos=plus), 4.5))["energy"])
            e_m = float(model(structure_to_graph(dict(s, pos=minus), 4.5))["energy"])
            assert float(out["forces"][atom, comp]) == pytest.approx(-(e_p - e_m) / (2 * h), abs=1e-6)
        if periodic:
            vol = float(torch.linalg.det(g.cell[0]))
            for a, b in ((0, 0), (1, 2)):
                eps = np.zeros((3, 3))
                eps[a, b] = eps[b, a] = 1e-4

                def energy(sign):
                    m = np.eye(3) + sign * eps
                    return float(model(structure_to_graph(dict(s, pos=np.array(s["pos"]) @ m, cell=s["cell"] @ m), 4.5))["energy"])

                fd = (energy(1) - energy(-1)) / 2e-4 / vol / (1 if a == b else 2)
                assert float(out["stress"][0, a, b]) == pytest.approx(fd, abs=1e-6)


def test_force_training_gradients_match_finite_differences():
    """The force-loss gradient in the solve's and the head's parameters equals
    finite differences."""
    model = _model(charge_solve=True)
    g = _cluster(5, 6, -1.0)
    fso = ForceStressOutput(model)

    def loss():
        return (fso(g)["forces"] ** 2).sum()

    params = [model.hardness_table.weight, model.q_net[-1].weight]
    grads = torch.autograd.grad(loss(), params)
    for p, grad in zip(params, grads):
        flat = p.detach().reshape(-1)
        for k in (0, min(8, flat.numel() - 1), 11 if flat.numel() > 11 else 0):
            h = 1e-5
            with torch.no_grad():
                flat[k] += h
            lp = float(loss())
            with torch.no_grad():
                flat[k] -= 2 * h
            lm = float(loss())
            with torch.no_grad():
                flat[k] += h
            assert float(grad.reshape(-1)[k]) == pytest.approx((lp - lm) / (2 * h), abs=1e-5, rel=1e-5)


def test_charge_solve_options_and_config_hook():
    with pytest.raises(ValueError, match="hardness"):
        _model(charge_solve=True, hardness="table")
    with pytest.raises(ValueError, match="exponent"):
        _model(charge_solve=True, exponent=6)
    cfg = from_dict({"model": {**SCHNET, "extra": {"long_range": {
        "n_channels": 2, "charge_solve": True, "hardness": "features", "hardness_init": 5.0}}}})
    model = build_model(cfg.model)
    assert isinstance(model, LatentEwald) and model.charge_solve and model.hardness_net is not None
    expect = 5.0 / LATENT_CHARGE_PER_E ** 2
    g = _cluster(6, 4)
    assert torch.allclose(model(g)["hardness"], torch.full((4,), expect), atol=1e-9)
    plain = _model()
    assert not plain.charge_solve and plain.hardness_table is None


def test_float32_solve_is_refined():
    """The float32 path (one float64-residual round) stays close to float64."""
    g64 = _cluster(7, 10, -1.0)
    model = _model(charge_solve=True)
    ref = model(g64)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        s = {"pos": g64.pos.numpy().astype(np.float32), "atomic_numbers": g64.atomic_numbers.tolist(),
             "total_charge": -1.0}
        out = model.float()(structure_to_graph(s, 4.5))     # the same weights, cast
    finally:
        torch.set_default_dtype(prev)
    assert torch.allclose(out["latent_charges"].double(), ref["latent_charges"], atol=2e-5, rtol=0)
    assert float(out["energy"]) == pytest.approx(float(ref["energy"]), abs=1e-4)


# fragments: one constraint row per molecule or ion

from xnn.common.models.charge_solve import (
    ION_CHARGES, bonded_fragments, covalent_radii, fragment_targets, ion_charge_table,
)

WATER = np.array([[0.0, 0.0, 0.119262], [0.0, 0.763239, -0.477047], [0.0, -0.763239, -0.477047]])


def _ion_cluster(charge=0.0, label=None, periodic=False):
    """Two waters, Na and Cl: atoms 0-2, 3-5 water, 6 Na, 7 Cl."""
    pos = np.concatenate([WATER, WATER + [3.0, 0.0, 0.0], [[0.0, 3.2, 0.0]], [[0.0, -3.2, 0.0]]])
    s = {"pos": pos, "atomic_numbers": [8, 1, 1, 8, 1, 1, 11, 17], "total_charge": charge}
    if label is not None:
        s["fragment_charges"] = np.array(label, dtype=float)
    if periodic:
        s.update(cell=np.eye(3) * 8.0, pbc=[True] * 3)
    return structure_to_graph(s, 4.5)


def test_bonded_fragments_and_their_targets():
    g = _ion_cluster()
    radii, ions = covalent_radii(), ion_charge_table()
    frag = bonded_fragments(g.atomic_numbers, g.edge_index, g.edge_vectors(), radii, ions)
    assert frag.tolist() == [0, 0, 0, 1, 1, 1, 2, 3]       # ions on their own, numbered by first atom
    assert fragment_targets(frag, g.atomic_numbers, ions).tolist() == [0.0, 0.0, 1.0, -1.0]
    label = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0])
    assert fragment_targets(frag, g.atomic_numbers, ions, label, torch.tensor(0.0)).tolist() == [0.0, 0.0, 1.0, -1.0]
    with pytest.raises(ValueError, match="differ between atoms"):
        fragment_targets(frag, g.atomic_numbers, ions, torch.tensor([0.0, 1.0, 0, 0, 0, 0, 1, -1]))
    with pytest.raises(ValueError, match="add up to"):
        fragment_targets(frag, g.atomic_numbers, ions, None, torch.tensor(1.0))
    # an ion never bonds; an element left out of the table does (Na 2.3 A from the O)
    close = structure_to_graph({"pos": np.concatenate([WATER, [[2.3, 0.0, 0.119262]]]),
                                "atomic_numbers": [8, 1, 1, 11]}, 4.5)
    args = (close.atomic_numbers, close.edge_index, close.edge_vectors(), radii)
    assert bonded_fragments(*args, ion_charge_table()).tolist() == [0, 0, 0, 1]
    assert bonded_fragments(*args, ion_charge_table({"Cl": -1})).tolist() == [0, 0, 0, 0]
    assert ion_charge_table()[11] == 1.0 and ION_CHARGES["Cl"] == -1


def test_fragments_follow_molecules_across_a_cell_boundary():
    pos = np.concatenate([WATER + [5.9, 3.0, 3.0], WATER + [3.0, 3.0, 3.0]])   # one O at x = 5.9 of a 6 A cell
    pos[1, 0] = 0.3                                                              # its H wrapped to the far side
    s = {"pos": pos, "atomic_numbers": [8, 1, 1, 8, 1, 1], "cell": np.eye(3) * 6.0, "pbc": [True] * 3}
    g = structure_to_graph(s, 4.5)
    frag = bonded_fragments(g.atomic_numbers, g.edge_index, g.edge_vectors(), covalent_radii(), ion_charge_table())
    assert frag.tolist() == [0, 0, 0, 1, 1, 1]


@pytest.mark.parametrize("periodic", [False, True])
def test_fragment_solve_keeps_each_fragment_at_its_charge(periodic):
    model = _model(charge_solve=True, fragments=True)
    out = model(_ion_cluster(0.0, periodic=periodic))
    q = out["latent_charges"][:, 1] / LATENT_CHARGE_PER_E
    assert out["fragments"].tolist() == [0, 0, 0, 1, 1, 1, 2, 3]
    assert float(q[:3].sum()) == pytest.approx(0.0, abs=1e-10)
    assert float(q[3:6].sum()) == pytest.approx(0.0, abs=1e-10)
    assert float(q[6]) == pytest.approx(1.0, abs=1e-10) and float(q[7]) == pytest.approx(-1.0, abs=1e-10)
    assert out["chemical_potential"].shape == (4,)
    # without fragments the ions share their charge with the waters (plain charge equilibration)
    plain = _model(charge_solve=True)(_ion_cluster(0.0, periodic=periodic))["latent_charges"][:, 1] / LATENT_CHARGE_PER_E
    assert abs(float(plain[6]) - 1.0) > 1e-3
    # a label overrides the ion table: Na counted neutral and Cl as -1 within a -1 structure
    labeled = model(_ion_cluster(-1.0, label=[0, 0, 0, 0, 0, 0, 0, -1], periodic=periodic))
    ql = labeled["latent_charges"][:, 1] / LATENT_CHARGE_PER_E
    assert float(ql[6]) == pytest.approx(0.0, abs=1e-10) and float(ql[7]) == pytest.approx(-1.0, abs=1e-10)
    with pytest.raises(ValueError, match="add up to"):
        model(_ion_cluster(1.0, periodic=periodic))


def test_fragment_label_travels_through_batches_and_subsets():
    a = _ion_cluster(0.0, label=[0, 0, 0, 0, 0, 0, 1, -1])
    b = _ion_cluster(0.0)
    batch = collate([a, b])
    assert batch.fragment_charges.shape == (16,) and batch.fragment_charges_mask.tolist() == [True, False]
    assert torch.equal(batch.subset(torch.tensor([True, False])).fragment_charges, a.fragment_charges)
    assert batch.subset(torch.tensor([False, True])).fragment_charges_mask.tolist() == [False]
    model = _model(charge_solve=True, fragments=True)
    out = model(batch)
    singles = [model(a), model(b)]
    assert torch.allclose(out["latent_charges"], torch.cat([s["latent_charges"] for s in singles]), atol=1e-10)
    assert torch.allclose(out["energy"], torch.cat([s["energy"] for s in singles]), atol=1e-10)


def test_fragment_solve_forces_are_the_energy_gradient():
    model = _model(charge_solve=True, fragments=True)
    s = {"pos": _ion_cluster().pos.numpy(), "atomic_numbers": [8, 1, 1, 8, 1, 1, 11, 17], "total_charge": 0.0}
    out = ForceStressOutput(model)(structure_to_graph(s, 4.5))
    h = 1e-5
    for atom, comp in ((6, 1), (1, 0)):
        plus, minus = np.array(s["pos"]), np.array(s["pos"])
        plus[atom, comp] += h
        minus[atom, comp] -= h
        e_p = float(model(structure_to_graph(dict(s, pos=plus), 4.5))["energy"])
        e_m = float(model(structure_to_graph(dict(s, pos=minus), 4.5))["energy"])
        assert float(out["forces"][atom, comp]) == pytest.approx(-(e_p - e_m) / (2 * h), abs=1e-6)


def test_trainer_warns_without_fragment_labels(tmp_path):
    import warnings
    from xnn.common.data import AtomicDataset
    from xnn.common.train import Trainer
    rng = np.random.default_rng(0)

    def structures(fragment_label):
        out = []
        for _ in range(3):
            s = {"pos": rng.uniform(0, 4, (4, 3)), "atomic_numbers": [1, 8, 1, 8], "energy": 0.0,
                 "forces": np.zeros((4, 3)), "total_charge": 0.0}
            if fragment_label:
                s["fragment_charges"] = np.zeros(4)
            out.append(s)
        return out

    def trainer(fragments, labeled):
        cfg = from_dict({"model": {**SCHNET, "extra": {"long_range": {
            "n_channels": 2, "charge_solve": True, "fragments": fragments}}},
            "data": {"batch_size": 3, "val_fraction": 0.34},
            "optim": {"epochs": 1}, "device": "cpu", "output_dir": str(tmp_path / "run")})
        return Trainer(cfg, AtomicDataset(structures(labeled), cfg.model.cutoff))

    with pytest.warns(UserWarning, match="no training structure carries a 'fragment_charges'"):
        trainer(True, False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        trainer(True, True)
        trainer(False, False)
