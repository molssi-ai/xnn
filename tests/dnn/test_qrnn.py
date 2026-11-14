"""QRNN tests: the charge recursion, energy, Coulomb and dispersion terms of the paper's equations
rebuilt atom by atom, QeqNN's charge equilibration, physics checks, TorchScript deployment, MDI,
training, configs and the fused AEV.

The AEV is ANI's (tested in ``test_ani.py``); the references here take it from the model and
rebuild everything that follows from the equations with the model's networks.
"""
import math
import os

import numpy as np
import pytest
import torch

from xnn.common.config import from_dict
from xnn.common.data import collate, structure_to_graph
from xnn.common.models import ForceStressOutput, available_models, build_model
from xnn.common.models.d4 import _load_reference
from xnn.common.models.dispersion import HARTREE
from xnn.common.models.electrostatics import COULOMB_CONSTANT, coulomb_direct
from xnn.dnn.models.base import _ElementNetworks
from xnn.dnn.models.qrnn import (
    C6_UNIT,
    D2_C6,
    D2_R0,
    QRNN,
    QRNN_SPECIES,
    chai_head_gordon_dispersion,
    eeq_hardness,
)

SPECIES = [1, 6, 7, 8, 16, 17]
CUTOFF = 5.2


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _structure(n=7, seed=0, elements=(8, 1, 6, 1, 7, 16, 17, 1, 6)):
    """Atoms on a jittered 1.4 Angstrom grid (no close contacts)."""
    rng = np.random.default_rng(seed)
    grid = np.array([[i, j, k] for i in range(3) for j in range(3) for k in range(3)], dtype=float) * 1.4
    pos = grid[rng.permutation(len(grid))[:n]] + rng.uniform(-0.2, 0.2, (n, 3))
    return {"pos": pos, "atomic_numbers": [elements[i % len(elements)] for i in range(n)]}


def _graph(n=7, seed=0, total_charge=None, cutoff=CUTOFF, **kw):
    s = _structure(n, seed)
    s.update(kw)
    if total_charge is not None:
        s["total_charge"] = float(total_charge)
    return structure_to_graph(s, cutoff)


def _small(seed=0, **kw):
    torch.manual_seed(seed)
    kw.setdefault("species", SPECIES)
    kw.setdefault("hidden", (8, 6))
    model = QRNN(**kw)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("bias"):
                p.add_(0.1 * torch.randn_like(p))
    return model


def _pairs(pos):
    n = len(pos)
    return [(i, j, float(np.linalg.norm(pos[i] - pos[j]))) for i in range(n) for j in range(n) if i != j]


def _net(nets, z, x):
    return nets.nets[str(int(z))](x.unsqueeze(0))[0]


def _reference(model, s, total):
    """The paper's equations atom by atom (eq 7, 12, 13, 8, 9 and the wB97X-D dispersion)."""
    g = structure_to_graph({**s, "total_charge": total}, model.cutoff)
    desc = model.featurizer(g).detach()
    z = list(s["atomic_numbers"])
    pos = np.asarray(s["pos"], dtype=float)
    n = len(z)
    radial = model.featurizer.radial
    etas, shifts, rc, pref = radial.etas.tolist(), radial.rs.tolist(), radial.cutoff, radial.prefactor
    pairs = _pairs(pos)

    def charge_vector(q):
        out = torch.zeros(n, len(etas) * len(shifts))
        for i, j, r in pairs:
            if r < rc:
                fc = 0.5 * (math.cos(math.pi * r / rc) + 1.0)
                terms = [pref * float(q[j]) * math.exp(-eta * (r - rs) ** 2) * fc for eta in etas for rs in shifts]
                out[i] += torch.tensor(terms)
        return out

    ref = _load_reference()
    hard = [(ref["eeq_eta"][zi] + math.sqrt(2 / math.pi) / ref["eeq_rad"][zi]) * HARTREE for zi in z]
    q = torch.full((n,), total / n)
    with torch.no_grad():
        for _ in range(model.n_iterations):
            gq = charge_vector(q)
            chi = [float(_net(model.charge_nets, z[i], torch.cat([desc[i], q[i:i + 1], gq[i]]))) for i in range(n)]
            lam = (total + sum(c / h for c, h in zip(chi, hard))) / sum(1 / h for h in hard)
            q = torch.tensor([-(c - lam) / h for c, h in zip(chi, hard)])
        gq = charge_vector(q)
        energy = sum(float(_net(model.element_nets, z[i], torch.cat([desc[i], q[i:i + 1], gq[i]])))
                     for i in range(n))
    energy += sum(float(model._self_energies_by_z[zi]) for zi in z)
    coul = 0.5 * sum(COULOMB_CONSTANT * float(q[i] * q[j]) / r / (1 + math.exp(8.5 * (2.2 - r))) for i, j, r in pairs)
    disp = 0.0
    for i, j, r in pairs:
        c6 = math.sqrt(D2_C6[z[i] - 1] * D2_C6[z[j] - 1]) * 1e6 / 96485.33212331001
        r0 = D2_R0[z[i] - 1] + D2_R0[z[j] - 1]
        disp -= 0.5 * c6 / r ** 6 / (1 + 6.0 * (r / r0) ** -12)
    return energy + coul + disp, q, coul, disp


def _augmented_minimum(amat, chi, total):
    """Minimize chi.q + 1/2 q^T A q at a fixed sum of q by the dense stationarity system."""
    n = len(chi)
    full = np.zeros((n + 1, n + 1))
    full[:n, :n] = amat
    full[:n, n] = full[n, :n] = 1.0
    sol = np.linalg.solve(full, np.concatenate([-np.asarray(chi), [total]]))
    return sol[:n]


# definitions

def test_registered_and_defaults():
    assert "qrnn" in available_models()
    model = QRNN()
    assert model.species == QRNN_SPECIES
    assert model.featurizer.output_dim == 16 * 8 + 32 * 36       # the ANI-1x grids over eight elements
    assert model.n_charge_features == 16 and model.cutoff == 5.2
    assert model.charge_model == "recursive" and model.n_iterations == 2
    assert model.initial_charges == "uniform" and model.receptive_field() is None
    first = model.element_nets.nets["6"][0]
    assert first.in_features == model.featurizer.output_dim + 1 + 16 and first.out_features == 128
    assert type(model.element_nets.nets["6"][1]).__name__ == "GELU"
    assert model.charge_nets.nets["1"][-1].out_features == 1


def test_option_errors():
    with pytest.raises(ValueError, match="charge_model"):
        QRNN(SPECIES, charge_model="ewald")
    with pytest.raises(ValueError, match="initial_charges"):
        QRNN(SPECIES, initial_charges="random")
    with pytest.raises(ValueError, match="n_iterations"):
        QRNN(SPECIES, n_iterations=0)
    with pytest.raises(ValueError, match="H to Xe"):
        QRNN([1, 6, 79])
    QRNN([1, 6, 79], dispersion=False)
    with pytest.raises(ValueError, match="hardness: got"):
        QRNN(SPECIES, hardness=[1.0])
    with pytest.raises(ValueError, match="positive"):
        QRNN([1, 6], hardness=[1.0, -1.0])


def test_hardness_is_the_eeq_diagonal():
    ref = _load_reference()
    values = eeq_hardness(QRNN_SPECIES)
    for z, h in zip(QRNN_SPECIES, values):
        assert h == pytest.approx((ref["eeq_eta"][z] + math.sqrt(2 / math.pi) / ref["eeq_rad"][z]) * HARTREE,
                                  rel=1e-15)
        assert h > 0
    torch.set_default_dtype(torch.float32)
    model = QRNN(QRNN_SPECIES).double()                          # built in float32, the tables exact
    for z, h in zip(QRNN_SPECIES, values):
        assert float(model._inverse_hardness_by_z[z]) == pytest.approx(1 / h, rel=1e-15)
    assert float(model._c6_by_z[17]) == pytest.approx(5.07 * C6_UNIT, rel=1e-15)


def test_dispersion_table_and_unit():
    assert len(D2_C6) == len(D2_R0) == 54
    assert C6_UNIT == pytest.approx(10.364269656262175, rel=1e-15)
    model = QRNN([1, 6])
    assert float(model._c6_by_z[6]) == pytest.approx(1.75 * C6_UNIT, rel=1e-15)
    assert float(model._r0_by_z[1]) == 1.001


# the paper's equations

@pytest.mark.parametrize("total", [0.0, 1.0, -2.0])
def test_matches_the_paper_equations(total):
    model = _small().eval()
    s = _structure(8, seed=1)
    out = model(structure_to_graph({**s, "total_charge": total}, CUTOFF))
    energy, q, coul, disp = _reference(model, s, total)
    assert float(out["energy"]) == pytest.approx(energy, rel=1e-12)
    assert torch.allclose(out["charges"], q, atol=1e-13)
    assert float(out["energy_elec"]) == pytest.approx(coul, rel=1e-12)
    assert float(out["energy_dispersion"]) == pytest.approx(disp, rel=1e-12)
    assert float(out["charges"].sum()) == pytest.approx(total, abs=1e-12)


def test_closed_form_is_the_constrained_minimum():
    """Eq 12 minimizes eq 11 (diagonal hardness only) at the fixed net charge."""
    model = _small().eval()
    g = _graph(7, seed=2, total_charge=1.0)
    out = model(g)
    hard = 1.0 / model._inverse_hardness_by_z[g.atomic_numbers]
    q = _augmented_minimum(np.diag(hard.numpy()), out["electronegativities"].detach().numpy(), 1.0)
    assert np.allclose(out["charges"].detach().numpy(), q, atol=1e-13)


def test_qeq_charges_minimize_the_charge_equilibration():
    """QeqNN: chi = X^2, sigma = sigma_0 + S^2 and the Gaussian Coulomb matrix of eq 5."""
    model = _small(charge_model="qeq").eval()
    s = _structure(7, seed=3)
    g = structure_to_graph({**s, "total_charge": -1.0}, CUTOFF)
    out = model(g)
    with torch.no_grad():
        xs = model.charge_nets(model.featurizer(g), g.atomic_numbers)
    assert xs.shape == (7, 2)
    chi = (xs[:, 0] ** 2).numpy()
    sigma = (0.05 + xs[:, 1] ** 2).numpy()
    pos = np.asarray(s["pos"])
    amat = np.zeros((7, 7))
    for i in range(7):
        amat[i, i] = 1 / (math.sqrt(math.pi) * sigma[i])
        for j in range(7):
            if i != j:
                r = np.linalg.norm(pos[i] - pos[j])
                amat[i, j] = math.erf(r / math.sqrt(2 * (sigma[i] ** 2 + sigma[j] ** 2))) / r
    q = _augmented_minimum(COULOMB_CONSTANT * amat, chi, -1.0)
    assert np.allclose(out["charges"].detach().numpy(), q, atol=1e-12)
    assert np.allclose(out["electronegativities"].detach().numpy(), chi, atol=1e-15)


def test_recursion_options():
    """The starting charges and the number of cycles change the charges; one cycle from zero
    charges ignores the charge inputs of the first networks."""
    g = _graph(7, seed=4, total_charge=1.0)
    charges = {}
    for kw in ({}, {"initial_charges": "zero"}, {"n_iterations": 3}, {"n_iterations": 1}):
        charges[str(kw)] = _small(**kw).eval()(g)["charges"]
    for kw in ({"initial_charges": "zero"}, {"n_iterations": 3}, {"n_iterations": 1}):
        assert float((charges["{}"] - charges[str(kw)]).abs().max()) > 1e-10
    for q in charges.values():
        assert float(q.sum()) == pytest.approx(1.0, abs=1e-12)


def test_custom_hardness_and_terms_off():
    model = _small(hardness=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], coulomb=False, dispersion=False).eval()
    g = _graph(7, seed=5)
    out = model(g)
    assert float(model._inverse_hardness_by_z[8]) == 0.25
    assert float(out["energy_elec"]) == 0.0 and float(out["energy_dispersion"]) == 0.0
    assert float(out["energy"]) == pytest.approx(float(out["energy_short"]), rel=1e-15)
    assert out["node_energy"].dtype == torch.float64


def test_coulomb_damping_both_paths():
    """The attenuated pair sum of eq 9, for one structure (dense) and a batch (pair list)."""
    rng = np.random.default_rng(0)
    pos = torch.tensor(rng.uniform(0, 4, (5, 3)))
    q = torch.tensor(rng.normal(size=5))
    ref = np.zeros(5)
    for i, j, r in _pairs(pos.numpy()):
        ref[i] += 0.5 * COULOMB_CONSTANT * float(q[i] * q[j]) / r / (1 + math.exp(8.5 * (2.2 - r)))
    batch = torch.zeros(5, dtype=torch.long)
    for num_graphs in (1, 0):
        node = coulomb_direct(q, pos, batch, COULOMB_CONSTANT, num_graphs, 2.2, 8.5)
        assert np.allclose(node.numpy(), ref, rtol=1e-14, atol=0)
    # two structures: the masked pair matrix equals the pair list
    batch = torch.tensor([0, 0, 1, 1, 1])
    pairs = coulomb_direct(q, pos, batch, COULOMB_CONSTANT, 2, 2.2, 8.5)
    dense = coulomb_direct(q, pos, batch, COULOMB_CONSTANT, 2, 2.2, 8.5, dense=True)
    assert torch.allclose(dense, pairs, rtol=1e-14, atol=0)


def test_dispersion_function_per_pair():
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 3.0, 0.4]])
    z = torch.tensor([6, 1, 17])
    model = QRNN([1, 6, 17])
    node = chai_head_gordon_dispersion(z, pos, torch.zeros(3, dtype=torch.long), model._c6_by_z, model._r0_by_z,
                                       1.0, 6.0)
    expected = 0.0
    for i, j, r in _pairs(pos.numpy()):
        a, b = int(z[i]), int(z[j])
        c6 = math.sqrt(D2_C6[a - 1] * D2_C6[b - 1]) * C6_UNIT
        expected -= 0.5 * c6 / r ** 6 / (1 + 6.0 * (r / (D2_R0[a - 1] + D2_R0[b - 1])) ** -12)
    assert float(node.sum()) == pytest.approx(expected, rel=1e-14)


# physics

def test_total_charge_changes_energy_and_dipole():
    model = _small().eval()
    s = _structure(7, seed=6)
    neutral = model(structure_to_graph(s, CUTOFF))
    cation = model(structure_to_graph({**s, "total_charge": 1.0}, CUTOFF))
    assert abs(float(neutral["energy"] - cation["energy"])) > 1e-6
    expected = (cation["charges"].unsqueeze(-1) * torch.as_tensor(s["pos"])).sum(0)
    assert torch.allclose(cation["dipole"][0], expected, atol=1e-14)


@pytest.mark.parametrize("charge_model", ["recursive", "qeq"])
def test_forces_are_minus_gradient(charge_model):
    model = ForceStressOutput(_small(charge_model=charge_model)).eval()
    s = {**_structure(6, seed=7), "total_charge": 1.0}
    out = model(structure_to_graph(s, CUTOFF))
    pos = np.asarray(s["pos"], dtype=float)
    h = 1e-5
    for (i, k) in ((0, 0), (3, 1), (5, 2)):
        p = pos.copy(); p[i, k] += h
        e_plus = float(model(structure_to_graph({**s, "pos": p}, CUTOFF))["energy"])
        p = pos.copy(); p[i, k] -= h
        e_minus = float(model(structure_to_graph({**s, "pos": p}, CUTOFF))["energy"])
        assert float(out["forces"][i, k]) == pytest.approx(-(e_plus - e_minus) / (2 * h), abs=1e-7)


def test_force_training_second_derivatives():
    """The weight gradient of a force loss equals its finite difference along one weight."""
    model = _small()
    wrapped = ForceStressOutput(model)
    g = _graph(6, seed=8, total_charge=-1.0)
    weight = model.charge_nets.nets["6"][0].weight

    def loss():
        return wrapped(g)["forces"].pow(2).sum()

    (grad,) = torch.autograd.grad(loss(), weight)
    h = 1e-6
    with torch.no_grad():
        weight[0, 0] += h
    up = float(loss())
    with torch.no_grad():
        weight[0, 0] -= 2 * h
    down = float(loss())
    with torch.no_grad():
        weight[0, 0] += h
    assert float(grad[0, 0]) == pytest.approx((up - down) / (2 * h), rel=1e-6, abs=1e-8)


def test_invariance_and_permutation():
    model = _small().eval()
    s = {**_structure(7, seed=9), "total_charge": 1.0}
    out = model(structure_to_graph(s, CUTOFF))
    rot, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))
    turned = model(structure_to_graph({**s, "pos": s["pos"] @ rot.T + 1.3}, CUTOFF))
    assert abs(float(out["energy"] - turned["energy"])) < 1e-11
    assert torch.allclose(out["charges"], turned["charges"], atol=1e-12)
    perm = np.random.default_rng(1).permutation(7)
    swapped = model(structure_to_graph({**s, "pos": s["pos"][perm],
                                        "atomic_numbers": np.array(s["atomic_numbers"])[perm]}, CUTOFF))
    assert abs(float(out["energy"] - swapped["energy"])) < 1e-11
    assert torch.allclose(out["charges"][perm], swapped["charges"], atol=1e-12)


@pytest.mark.parametrize("charge_model", ["recursive", "qeq"])
def test_batching(charge_model):
    model = _small(charge_model=charge_model).eval()
    graphs = [_graph(n, seed=n, total_charge=q) for n, q in ((4, 0.0), (6, 1.0), (8, -1.0))]
    out = model(collate(graphs))
    offset = 0
    for b, g in enumerate(graphs):
        one = model(g)
        assert abs(float(out["energy"][b] - one["energy"][0])) < 1e-11
        assert torch.allclose(out["charges"][offset:offset + g.num_nodes], one["charges"], atol=1e-12)
        assert torch.allclose(out["dipole"][b], one["dipole"][0], atol=1e-12)
        offset += g.num_nodes


def test_charge_flows_beyond_the_cutoff():
    """A substitution far beyond the cutoff moves the charges (global charges, no receptive field)."""
    model = _small().eval()
    s = _structure(5, seed=10)
    far = {"pos": np.concatenate([s["pos"], [[20.0, 0.0, 0.0]]]), "atomic_numbers": list(s["atomic_numbers"]) + [1],
           "total_charge": 0.0}
    out = model(structure_to_graph(far, CUTOFF))
    other = model(structure_to_graph({**far, "atomic_numbers": list(s["atomic_numbers"]) + [17]}, CUTOFF))
    assert float((out["charges"][:5] - other["charges"][:5]).abs().max()) > 1e-6


def test_float32_forward_matches_float64():
    model = _small().eval()
    g = _graph(8, seed=11, total_charge=1.0)
    out64 = model(g)
    out32 = model.float()(structure_to_graph({"pos": g.pos.float(), "atomic_numbers": g.atomic_numbers,
                                              "total_charge": 1.0}, CUTOFF))
    scale = max(1.0, abs(float(out64["energy"])))
    assert abs(float(out64["energy"] - out32["energy"].double())) < 1e-4 * scale
    assert torch.allclose(out64["charges"], out32["charges"].double(), atol=1e-5)


def test_periodic_structures():
    """Periodic cells take the charges and the networks (QRNN and QeqNN) without the all-pair terms."""
    cell = {"cell": np.eye(3) * 6.0, "pbc": [True, True, True]}
    g = _graph(6, seed=12, total_charge=1.0, **cell)
    with pytest.raises(ValueError, match="periodic"):
        _small().eval()(g)
    for charge_model in ("recursive", "qeq"):
        out = _small(charge_model=charge_model, coulomb=False, dispersion=False).eval()(g)
        assert float(out["charges"].sum()) == pytest.approx(1.0, abs=1e-12)
        assert "dipole" not in out


def test_element_networks_with_several_outputs():
    """Atoms of elements without a network give zero rows."""
    nets = _ElementNetworks([1, 6], 3, (4,), "gelu", n_outputs=2)
    desc = torch.randn(4, 3)
    out = nets(desc, torch.tensor([1, 6, 8, 1]))
    assert out.shape == (4, 2) and torch.equal(out[2], torch.zeros(2))
    assert torch.allclose(out[0], nets.nets["1"](desc[0:1])[0])


# data, deployment and training

def test_ase_calculator_reports_charges():
    ase = pytest.importorskip("ase")
    from xnn.common.deploy import XNNCalculator
    model = ForceStressOutput(_small()).eval()
    atoms = ase.Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    atoms.info["total_charge"] = 1.0
    atoms.calc = XNNCalculator(model, cutoff=CUTOFF)
    assert atoms.get_charges().sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("charge_model", ["recursive", "qeq"])
def test_scriptable_and_exports(charge_model, tmp_path):
    """The TorchScript core gives the eager charges and energies; the exported file carries a fixed
    total charge and serves the whole-system and the pair-style entries; the edge-only LAMMPS wrapper
    evaluates a periodic cell without the all-pair terms."""
    from xnn.common.deploy import export_to_lammps, export_torchscript_potential

    model = _small(charge_model=charge_model).eval()
    g = _graph(6, seed=13)
    scripted = torch.jit.script(model)
    args = (g.atomic_numbers, g.edge_index, g.edge_vectors(), g.pos, torch.zeros(3, 3),
            torch.zeros(3, dtype=torch.bool), 1.0, 1.0)
    for a, b in zip(scripted.node_features_energy_charges(*args), model.node_features_energy_charges(*args)):
        assert (a - b).abs().max() < 1e-12
    with pytest.raises(Exception, match="needs the positions"):
        scripted.node_features_energy(g.atomic_numbers, g.edge_index, g.edge_vectors())
    with pytest.raises(RuntimeError, match="needs the positions"):
        model.node_features_energy(g.atomic_numbers, g.edge_index, g.edge_vectors())

    s = _structure(6, seed=14)
    path = str(tmp_path / "qrnn_q.pt")
    export_torchscript_potential(model, model.cutoff, path, total_charge=1.0)
    out = torch.jit.load(path)(torch.as_tensor(s["pos"]), torch.as_tensor(s["atomic_numbers"]))
    ref = ForceStressOutput(model)(structure_to_graph({**s, "total_charge": 1.0}, CUTOFF))
    assert abs(float(out["energy"]) - float(ref["energy"])) < 1e-8
    assert (out["forces"] - ref["forces"].detach()).abs().max() < 1e-7
    assert (out["charges"] - ref["charges"].detach()).abs().max() < 1e-10

    # the pair-style entry of the same file (the MD engine's neighbor list, a molecule: zero cell) gives
    # the eager energies, forces and charges with the all-pair terms on
    g = structure_to_graph({**s, "total_charge": 1.0}, CUTOFF)
    lammps = torch.jit.load(path).forward_lammps(g.pos, g.edge_index, g.cell_shifts, g.atomic_numbers,
                                                 torch.zeros(3, 3))
    assert abs(float(lammps["energy"]) - float(ref["energy"])) < 1e-8
    assert (lammps["forces"] - ref["forces"].detach()).abs().max() < 1e-7
    assert (lammps["charges"] - ref["charges"].detach()).abs().max() < 1e-10

    # a periodic cell without the all-pair terms, through the pair-style entry
    periodic = _small(charge_model=charge_model, coulomb=False, dispersion=False).eval()
    g = _graph(6, seed=16, total_charge=-1.0, cell=np.eye(3) * 6.0, pbc=[True, True, True])
    path = str(tmp_path / "qrnn_cell.pt")
    export_torchscript_potential(periodic, periodic.cutoff, path, total_charge=-1.0)
    lammps = torch.jit.load(path).forward_lammps(g.pos, g.edge_index, g.cell_shifts, g.atomic_numbers, g.cell[0])
    ref = ForceStressOutput(periodic)(g)
    assert abs(float(lammps["energy"]) - float(ref["energy"])) < 1e-8
    assert (lammps["forces"] - ref["forces"].detach()).abs().max() < 1e-7

    if charge_model == "recursive":
        model = _small(coulomb=False, dispersion=False).eval()
        g = _graph(6, seed=15, cell=np.eye(3) * 6.0, pbc=[True, True, True])
        desc, energy = model.node_features_energy(g.atomic_numbers, g.edge_index, g.edge_vectors())
        assert abs(float(energy.sum()) - float(model(g)["energy"])) < 1e-12
        path = str(tmp_path / "qrnn_lammps.pt")
        export_to_lammps(model, CUTOFF, path)
        out = torch.jit.load(path)(g.pos, g.edge_index, g.cell_shifts, g.atomic_numbers, g.cell[0])
        ref = ForceStressOutput(model)(g)
        assert abs(float(out["total_energy"]) - float(ref["energy"])) < 1e-8
        assert (out["forces"] - ref["forces"].detach()).abs().max() < 1e-8


def _config(**extra):
    return {"name": "qrnn", "cutoff": CUTOFF, "n_features": 0, "n_interactions": 0, "n_rbf": 0,
            "extra": {"species": [1, 6, 8], "hidden": [8], **extra}}


def test_mdi_engine_serves_charged_systems(tmp_path):
    from xnn.common.deploy.mdi_engine import BOHR_TO_ANGSTROM, HARTREE_TO_EV, MDIEngine
    cfg = from_dict({"model": _config()})
    torch.manual_seed(0)
    model = ForceStressOutput(build_model(cfg.model))
    path = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg}, path)
    pos = np.array([[0.0, 0.0, 0.119262], [0.0, 0.763239, -0.477047], [0.0, -0.763239, -0.477047]])
    energies = {}
    for charge in (0.0, 1.0):
        engine = MDIEngine.from_checkpoint(str(path), total_charge=charge)
        engine.natoms, engine.atomic_numbers = 3, np.array([8, 1, 1])
        engine.coords_bohr = pos / BOHR_TO_ANGSTROM
        engine.calculate()
        energies[charge] = engine.energy * HARTREE_TO_EV
    ref = model(structure_to_graph({"pos": pos, "atomic_numbers": [8, 1, 1], "total_charge": 1.0}, CUTOFF))
    assert energies[1.0] == pytest.approx(float(ref["energy"]), abs=1e-5)
    assert abs(energies[1.0] - energies[0.0]) > 1e-6


def test_train_step_with_dipoles_and_charges(tmp_path):
    """Energy, force, dipole and charge losses train the energy and charge networks."""
    from xnn.common.data import AtomicDataset
    from xnn.common.train import Trainer

    rng = np.random.default_rng(0)
    structures = []
    for i in range(6):
        s = _structure(5, seed=i, elements=(8, 1, 6))
        q = rng.normal(scale=0.2, size=5)
        s.update(energy=float(rng.normal()), forces=rng.normal(size=(5, 3)), charges=q - q.mean() + 0.2,
                 dipole=rng.normal(size=3), total_charge=1.0)
        structures.append(s)
    cfg = from_dict({"model": _config(),
                     "optim": {"epochs": 2, "charge_weight": 1.0, "dipole_weight": 1.0},
                     "data": {"batch_size": 3, "val_fraction": 0.0},
                     "output_dir": str(tmp_path)})
    logs = Trainer(cfg, AtomicDataset(structures, CUTOFF)).fit()["train"]
    assert {"energy_mse", "force_mse", "charge_mse", "dipole_mse"} <= set(logs)


def test_shipped_config_and_unknown_options():
    import yaml

    from xnn.common.config.options import IgnoredOptionWarning
    path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "model", "qrnn.yaml")
    with open(path) as f:
        shipped = build_model(from_dict({"model": yaml.safe_load(f)}).model)
    assert sum(p.numel() for p in shipped.parameters()) == sum(p.numel() for p in QRNN().parameters())
    with pytest.warns(IgnoredOptionWarning, match="charge_cycles"):
        model = build_model(from_dict({"model": _config(charge_cycles=3, charge_model="qeq",
                                                        hardness=[10.0, 11.0, 12.0])}).model)
    assert model.charge_model == "qeq" and float(model._inverse_hardness_by_z[6]) == pytest.approx(1 / 11.0)


# fused AEV

def test_fast_path_matches_the_reference(monkeypatch):
    """The fused AEV kernels give the reference energies, charges, forces and force-loss gradients."""
    pytest.importorskip("triton")
    device = "cuda" if torch.cuda.is_available() else "cpu"     # the interpreter on a CPU
    if device == "cpu":
        if os.environ.get("TRITON_INTERPRET") != "1":
            pytest.skip("needs a CUDA device or the Triton interpreter")
        monkeypatch.setattr("xnn.dnn.fast.aev.supported", lambda device, dtype: True)
    model = _small().to(device)
    wrapped = ForceStressOutput(model)
    g = _graph(7, seed=16, total_charge=1.0).to(device)
    out = {}
    for use_fast in (False, True):
        model.set_use_fast(use_fast)
        model.zero_grad(set_to_none=True)
        res = wrapped(g)
        assert model.featurizer.radial.fast_active == use_fast
        res["forces"].pow(2).sum().backward()
        grad = model.charge_nets.nets["6"][0].weight.grad.detach().clone()
        out[use_fast] = (res["energy"].detach(), res["charges"].detach(), res["forces"].detach(), grad)
    model.set_use_fast(False)
    for ref, fast in zip(out[False], out[True]):
        assert (fast - ref).abs().max() <= 1e-12 * ref.abs().max().clamp_min(1e-30)


# grouped element networks

def _set_grouped(model, on):
    for nets in (model.element_nets, model.charge_nets):
        nets.fast_active = on


@pytest.mark.parametrize("charge_model", ["recursive", "qeq"])
def test_grouped_networks_match_the_reference(charge_model):
    """One batched product per layer gives the reference energies, charges, forces and force-loss
    gradients (second derivatives)."""
    model = _small(charge_model=charge_model)
    assert model.element_nets.grouping and model.charge_nets.grouping
    wrapped = ForceStressOutput(model)
    g = collate([_graph(7, seed=17, total_charge=1.0), _graph(5, seed=18)])
    out = {}
    for on in (False, True):
        _set_grouped(model, on)
        model.zero_grad(set_to_none=True)
        res = wrapped(g)
        res["forces"].pow(2).sum().backward()
        grads = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
        out[on] = [res["energy"].detach(), res["charges"].detach(), res["forces"].detach()] + grads
    _set_grouped(model, False)
    assert len(out[False]) == len(out[True])
    for ref, fast in zip(out[False], out[True]):
        assert (fast - ref).abs().max() <= 1e-12 * ref.abs().max().clamp_min(1e-30)


def test_grouped_evaluation_edge_cases():
    """Elements without a network, outputs of several values, no bias and descriptors wider than the
    networks read; no atom of a known element gives zeros."""
    torch.manual_seed(0)
    for bias in (True, False):
        nets = _ElementNetworks([1, 6, 8], 3, (4, 5), "celu", bias=bias, extra_inputs=1, n_outputs=2)
        assert nets.groupable()
        desc, extra = torch.randn(6, 5), torch.randn(6, 1)
        z = torch.tensor([6, 1, 9, 1, 8, 1])
        ref = nets(desc, z, extra)
        assert torch.allclose(nets.forward_padded(desc, z, extra, 6), ref, atol=1e-14)
        nets.fast_active = True
        assert torch.allclose(nets(desc, z, extra), ref, atol=1e-14)
        assert torch.equal(nets(desc, torch.tensor([9, 9, 9, 9, 9, 9]), extra), torch.zeros(6, 2))
        nets.fast_active = False
    scalar = _ElementNetworks([1, 6], 3, (4,), "gelu")
    scalar.fast_active = True
    assert scalar(torch.randn(2, 3), torch.tensor([17, 17])).shape == (2,)


def test_groupable_and_fast_support():
    assert not _ElementNetworks([1, 6], 3, {1: (4,), 6: (5,)}).groupable()
    assert not _ElementNetworks([1, 6], 3, {1: (4,), 6: (4, 4)}).groupable()
    assert not _ElementNetworks([1, 6], 3, (4,), {1: "gelu", 6: "silu"}).groupable()
    assert not _ElementNetworks([1, 6], 3, (4,), {1: "celu", 6: torch.nn.CELU(0.5)}).groupable()
    assert not _ElementNetworks([1, 6], {1: 3, 6: 2}, (4,)).groupable()
    nets = _ElementNetworks([1, 6], 3, (4,))
    assert not nets.fast_supported(torch.device("cuda"), torch.float64)
    nets.grouping = True
    assert nets.fast_supported(torch.device("cuda"), torch.float64)
    assert not nets.fast_supported(torch.device("cpu"), torch.float64)
    from xnn.dnn.fast.qrnn import AUTO_POLICY, AUTO_POLICY_TRAINING
    assert QRNN([1, 6]).fast_policies() == (AUTO_POLICY, AUTO_POLICY_TRAINING)
    from xnn.dnn.featurizers import AEV
    from xnn.dnn.models.base import DescriptorPotential

    class Plain(torch.nn.Module):
        cutoff, output_dim = 3.0, 4

    assert DescriptorPotential(Plain(), [1, 6]).fast_policies() == (None, None)
    assert AEV([1, 6]).fast_policies() == DescriptorPotential(AEV([1, 6]), [1, 6]).fast_policies()


def test_dispersion_dense_and_pair_paths_agree():
    model = QRNN([1, 6, 17])
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 3.0, 0.4], [2.0, 2.0, 2.0]], requires_grad=True)
    z = torch.tensor([6, 1, 17, 1])
    batch = torch.zeros(4, dtype=torch.long)
    nodes = [chai_head_gordon_dispersion(z, pos, batch, model._c6_by_z, model._r0_by_z, 1.0, 6.0, k)
             for k in (1, 0)]
    assert torch.allclose(nodes[0], nodes[1], rtol=1e-14, atol=0)
    grads = [torch.autograd.grad(e.sum(), pos)[0] for e in nodes]
    assert torch.allclose(grads[0], grads[1], rtol=1e-13, atol=1e-16)
    two = torch.tensor([0, 0, 1, 1])
    pairs = chai_head_gordon_dispersion(z, pos, two, model._c6_by_z, model._r0_by_z, 1.0, 6.0, 2)
    dense = chai_head_gordon_dispersion(z, pos, two, model._c6_by_z, model._r0_by_z, 1.0, 6.0, 2, dense=True)
    assert torch.allclose(dense, pairs, rtol=1e-14, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fast_path_parity_on_the_gpu(dtype):
    """use_fast=True (fused AEV and grouped networks) gives the reference energies, forces and
    force-loss gradients on a GPU."""
    pytest.importorskip("triton")
    model = _small().to("cuda", dtype)
    wrapped = ForceStressOutput(model)
    g = collate([_graph(12, seed=19, total_charge=1.0), _graph(9, seed=20, total_charge=-1.0)]).to("cuda")
    g.pos = g.pos.to(dtype)
    out = {}
    for use_fast in (False, True):
        model.set_use_fast(use_fast).train()
        model.zero_grad(set_to_none=True)
        res = wrapped(g)
        assert model.element_nets.fast_active == use_fast
        res["forces"].pow(2).sum().backward()
        grads = [p.grad.detach().double() for p in model.parameters() if p.grad is not None]
        out[use_fast] = [res["energy"].detach().double(), res["forces"].detach().double()] + grads
    tol = 1e-11 if dtype == torch.float64 else 2e-4
    for ref, fast in zip(out[False], out[True]):
        assert float((fast - ref).abs().max() / ref.abs().max().clamp_min(1e-30)) < tol


# captured steps

def _padded(model, graphs):
    from xnn.common.train.graphed import PaddedCollate, PlannedDataset, capacity
    planner = model.index_planner()
    planned = PlannedDataset(graphs, planner)
    cap = capacity(planner, planned, len(graphs))
    batch = PaddedCollate(planner, cap)([planned[i] for i in range(len(graphs))])
    assert batch.index_plan["aev_e1"].shape[0] == cap.plan["triplets"]
    return batch


def test_padded_batch_matches_the_reference():
    """A padded batch with enumerated triplets (a captured step's input, evaluated without host reads)
    gives the real structures' energies, charges, dipoles, forces and force-loss gradients."""
    model = _small()
    graphs = [_graph(n, seed=n, total_charge=q) for n, q in ((5, 0.0), (7, 1.0), (6, -1.0))]
    padded, plain = _padded(model, graphs), collate(graphs)
    wrapped = ForceStressOutput(model).train()
    results = []
    for batch in (plain, padded):
        model.zero_grad(set_to_none=True)
        out = wrapped(batch)
        n = plain.num_nodes
        out["forces"][:n].pow(2).sum().backward()
        grads = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
        results.append([out["energy"][:3].detach(), out["charges"][:n].detach(), out["dipole"][:3].detach(),
                        out["forces"][:n].detach()] + grads)
    assert len(results[0]) == len(results[1])
    for ref, pad in zip(*results):
        assert (pad - ref).abs().max() <= 1e-11 * max(float(ref.abs().max()), 1.0)


def test_captured_steps_need_molecules_and_recursive_charges():
    model = _small()
    planner = model.index_planner()
    with pytest.raises(ValueError, match="periodic"):
        planner(_graph(5, seed=1, cell=np.eye(3) * 6.0, pbc=[True, True, True]))
    short = _small(coulomb=False, dispersion=False).index_planner()
    assert set(short(_graph(5, seed=1, cell=np.eye(3) * 6.0, pbc=[True, True, True]))) == {"aev_e1", "aev_e2"}
    qeq = _small(charge_model="qeq")
    with pytest.raises(ValueError, match="recursive"):
        qeq.index_planner()
    g = _graph(5, seed=2)
    with pytest.raises(ValueError, match="captured"):
        qeq.compute(g.atomic_numbers, g.edge_index, g.edge_vectors(), g.pos, g.batch, 1,
                    torch.zeros(1), None, None, None, g.num_nodes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_replayed_evaluations_equal_ordinary_ones(dtype):
    """Captured evaluations along a trajectory, a larger molecule (captured again) and a two-molecule
    batch equal the ordinary ones."""
    from xnn.common.deploy.graphed_eval import GraphedEvaluation
    model = _small().to("cuda", dtype).eval()
    wrapped = ForceStressOutput(model).eval()
    graphed = GraphedEvaluation(wrapped, trial=0)
    rng = np.random.default_rng(0)
    base = _structure(8, seed=3)
    inputs = ([[{**base, "pos": base["pos"] + 0.02 * rng.normal(size=(8, 3)), "total_charge": 1.0}]
               for _ in range(4)] + [[{**_structure(12, seed=4), "total_charge": -1.0}]]
              + [[_structure(6, seed=5), {**_structure(7, seed=6), "total_charge": 1.0}]])
    tol = 1e-12 if dtype == torch.float64 else 1e-5
    captured = 0
    for structures in inputs:
        g = collate([structure_to_graph(s, CUTOFF) for s in structures]).to("cuda")
        g.pos = g.pos.to(dtype)
        ref = wrapped(g)
        out = graphed(g)
        captured += graphed.graph is not None
        for key in ("energy", "forces", "charges", "dipole"):
            r = ref[key].detach()
            assert out[key].shape == r.shape, key
            assert (out[key] - r).abs().max() <= tol * max(float(r.abs().max()), 1.0), key
    assert captured > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_captured_training_matches_ordinary_training(tmp_path):
    """Training with captured steps ends at the ordinary run's weights."""
    from xnn.common.data import AtomicDataset
    from xnn.common.train import Trainer

    rng = np.random.default_rng(0)
    structures = []
    for i in range(6):
        s = _structure(6, seed=i, elements=(8, 1, 6))
        q = rng.normal(scale=0.2, size=6)
        s.update(energy=float(rng.normal()), forces=rng.normal(size=(6, 3)), charges=q - q.mean() + 1 / 6,
                 total_charge=1.0)
        structures.append(s)
    ds = AtomicDataset(structures, CUTOFF)
    weights = {}
    for cuda_graph in (False, True):
        torch.manual_seed(1)
        cfg = from_dict({"model": _config(), "optim": {"epochs": 4, "lr": 1e-3, "charge_weight": 1.0,
                                                      "scheduler": "cosine", "ema_decay": 0.9},
                         "data": {"batch_size": 2, "val_fraction": 0.0},
                         "trainer": {"loggers": [], "cuda_graph": cuda_graph},
                         "device": "cuda", "seed": 3, "output_dir": str(tmp_path / str(cuda_graph))})
        trainer = Trainer(cfg, ds)
        trainer.fit()
        if cuda_graph:
            assert trainer._graphed is not None and trainer._graphed.graph is not None
        weights[cuda_graph] = {n: p.detach().clone() for n, p in trainer.module.named_parameters()}
    for name, p in weights[False].items():
        err = (weights[True][name] - p).abs().max() / p.abs().max().clamp_min(1e-30)
        assert err < 1e-9, (name, float(err))
