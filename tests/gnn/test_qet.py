"""QET tests: the charge equilibration and Coulomb potential of the paper's equations, the
total charge and the applied potential, physics checks, TorchScript deployment, configs, the
MatGL conversion, the fused trunk and (optionally) fidelity to MatGL.

The trunk is TensorNet's (tested in ``test_tensornet.py``); the references here rebuild the
charge head from its equations with the model's parameters.
"""
import json
import math
import os
import sys

import numpy as np
import pytest
import torch

from xnn.common.config import from_dict
from xnn.common.data import collate, structure_to_graph
from xnn.common.models import ForceStressOutput, available_models, build_model
from xnn.gnn.models.qet import COULOMB_CONSTANT, COVALENT_RADII, QET


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _structure(n=6, seed=0, spread=3.0):
    rng = np.random.default_rng(seed)
    return {"pos": rng.uniform(0, spread, (n, 3)), "atomic_numbers": ([1, 6, 8] * n)[:n]}


def _graph(n=6, cutoff=4.5, seed=0, periodic=False, cell=6.0, total_charge=None, potential=None):
    s = _structure(n, seed)
    if periodic:
        s["cell"] = np.eye(3) * cell
        s["pbc"] = [True, True, True]
    if total_charge is not None:
        s["total_charge"] = float(total_charge)
    if potential is not None:
        s["external_potential"] = potential
    return structure_to_graph(s, cutoff)


def _small(seed=0, **kw):
    torch.manual_seed(seed)
    kw.setdefault("n_features", 8)
    kw.setdefault("n_interactions", 2)
    kw.setdefault("n_rbf", 6)
    kw.setdefault("cutoff", 4.5)
    model = QET(**kw)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("bias") or name.startswith("readout_norm"):
                p.add_(0.1 * torch.randn_like(p))
            elif name in ("hardness_readout.weight", "sigma"):
                p.copy_(torch.empty_like(p).uniform_(0.5, 2.0))
    return model


def _head_inputs(model, g):
    """The trunk's features and the electronegativities and hardnesses the model equilibrates."""
    x = model._features(model.tensors(g.atomic_numbers, g.edge_index, g.edge_vectors(),
                                      torch.zeros(g.num_nodes)))
    chi = model.chi_readout(x).squeeze(-1)
    if g.external_potential is not None:
        chi = chi + g.external_potential
    return x, chi, model.hardness_readout(x, g.atomic_numbers)


def _reference_charges(chi, eta, batch, total):
    """Minimize sum(chi q + eta q^2 / 2) at a fixed total charge per structure: the linear system
    of the stationarity conditions and the constraint, solved densely."""
    q = torch.empty_like(chi)
    for b in range(len(total)):
        idx = (batch == b).nonzero().squeeze(-1)
        n = len(idx)
        a = torch.zeros(n + 1, n + 1)
        a[:n, :n] = torch.diag(eta[idx])
        a[:n, n] = -1.0
        a[n, :n] = 1.0
        rhs = torch.cat([-chi[idx], total[b:b + 1]])
        q[idx] = torch.linalg.solve(a, rhs)[:n]
    return q


def _reference_potential(model, g, q):
    """Equation 3 of the paper, pair by pair, with the degree-3 polynomial envelope written out."""
    sigma = model.sigma[g.atomic_numbers]
    v = torch.zeros(g.num_nodes)
    vec = g.edge_vectors()
    for e in range(g.edge_index.shape[1]):
        j, i = int(g.edge_index[0, e]), int(g.edge_index[1, e])
        r = float(torch.linalg.norm(vec[e]))
        x = r / model.cutoff
        f = 1 - 10 * x ** 3 + 15 * x ** 4 - 6 * x ** 5 if r < model.cutoff else 0.0
        gamma = math.sqrt(float(sigma[i]) ** 2 + float(sigma[j]) ** 2)
        v[i] += COULOMB_CONSTANT * float(q[j]) * math.erf(r / (math.sqrt(2) * gamma)) * f / r
    return v


# registry / defaults

def test_registered_and_defaults():
    """The defaults of the authors' code: 32 Gaussians on [0, cutoff + 1] with exponent 0.5, no self
    pairs, a gated energy network on F + 2 inputs, one hardness per element, covalent-radius widths."""
    assert "qet" in available_models()
    m = QET()
    assert m.cutoff == 5.0 and m.n_features == 64 and len(m.layers) == 2 and not m.self_pairs
    assert float(m.rbf.centers[-1]) == 6.0 and m.rbf.gamma == 0.5 and m.rbf.centers.numel() == 32
    sizes = [(layer.in_features, layer.out_features) for layer in m.readout.mlp.layers]
    assert sizes == [(66, 66), (66, 64), (64, 64), (64, 1)]
    assert m.hardness_mode == "element" and torch.equal(m.hardness_readout.weight, torch.ones(128))
    assert float(m.sigma[8]) == 0.66 and not m.sigma.requires_grad
    assert m.receptive_field() is None and not m.refreshes_ghosts


def test_published_architecture_size():
    """The MatPES QET checkpoints (F = 128, 3 smooth Bessel functions, environment hardness) have
    905,492 parameters over 89 elements; xnn's tables have 128 rows and a reference energy each."""
    m = QET(n_features=128, radial_basis="smooth_bessel", n_rbf=3, hardness="environment")
    n = sum(p.numel() for p in m.parameters())
    assert n - (128 - 89) * 128 - 128 == 905_492


def test_options():
    """The hardness and width options, and the Gaussian options of other bases."""
    with pytest.raises(ValueError, match="hardness"):
        QET(hardness="global")
    m = QET(n_features=8, hardness="environment", trainable_widths=True)
    assert isinstance(m.sigma, torch.nn.Parameter) and torch.equal(m.sigma.detach(), torch.ones(128))
    assert m.hardness_readout.mlp.activate_last
    with pytest.warns(UserWarning, match="rbf_max"):
        QET(n_features=8, radial_basis="smooth_bessel", n_rbf=3, rbf_max=7.0)
    bessel = QET(n_features=8, radial_basis="smooth_bessel", n_rbf=3)
    assert bessel.rbf.__class__.__name__ == "SmoothBesselRBF"


def test_covalent_radii_match_ase():
    """The widths are the covalent radii ASE tabulates (Cordero et al. 2008)."""
    data = pytest.importorskip("ase.data")
    for z, r in enumerate(COVALENT_RADII):
        assert r == pytest.approx(float(data.covalent_radii[z]), abs=1e-12), z
    assert float(QET(n_elements=128).sigma[110]) == 0.2


# the charge head

def test_charges_follow_the_closed_form():
    """Equation 2: the charges minimize the equilibration energy at each structure's total charge,
    with an applied potential shifting the electronegativities."""
    model = _small().eval()
    graphs = [_graph(n=7, seed=1, total_charge=0.0), _graph(n=5, seed=2, total_charge=1.0),
              _graph(n=6, seed=3, total_charge=-2.0, potential=np.linspace(-0.3, 0.4, 6))]
    batch = collate(graphs)
    out = model(batch)
    _, chi, eta = _head_inputs(model, batch)
    expect = _reference_charges(chi, eta, batch.batch, torch.tensor([0.0, 1.0, -2.0]))
    assert torch.allclose(out["charges"], expect, rtol=0, atol=1e-12)
    sums = torch.zeros(3).index_add_(0, batch.batch, out["charges"])
    assert torch.allclose(sums, torch.tensor([0.0, 1.0, -2.0]), atol=1e-12)


@pytest.mark.parametrize("periodic", [False, True])
def test_potential_follows_equation_3(periodic):
    model = _small().eval()
    g = _graph(n=7, seed=4, periodic=periodic, cell=5.0, total_charge=1.0)
    out = model(g)
    expect = _reference_potential(model, g, out["charges"].detach())
    assert torch.allclose(out["electrostatic_potential"], expect, rtol=0, atol=1e-11)


def test_energy_network_reads_the_charges_and_the_potential():
    """The atomic energy is the gated network of the layer-normed [features, charge, potential]."""
    model = _small().eval()
    g = _graph(n=6, seed=5, total_charge=1.0)
    out = model(g)
    h = torch.cat([out["node_features"], out["charges"][:, None], out["electrostatic_potential"][:, None]], -1)
    expect = (model.energy_scale * model.readout(model.readout_norm(h)).squeeze(-1) + model.energy_shift
              + model.atom_ref(g.atomic_numbers).squeeze(-1))
    assert torch.allclose(out["node_energy"], expect, rtol=0, atol=1e-13)


def test_total_charge_and_potential_enter_the_equilibration_alone():
    """The trunk's features ignore the total charge; a uniform applied potential is absorbed by the
    Lagrange multiplier; a potential on some atoms moves charge between them."""
    model = _small().eval()
    neutral, charged = model(_graph(n=6, seed=6)), model(_graph(n=6, seed=6, total_charge=1.0))
    assert torch.equal(neutral["node_features"], charged["node_features"])
    assert not torch.allclose(neutral["charges"], charged["charges"])
    uniform = model(_graph(n=6, seed=6, potential=np.full(6, 0.7)))
    assert torch.allclose(uniform["charges"], neutral["charges"], atol=1e-13)
    assert torch.allclose(uniform["energy"], neutral["energy"], atol=1e-12)
    biased = model(_graph(n=6, seed=6, potential=np.array([1.0, 0, 0, 0, 0, 0])))
    assert float(biased["charges"][0]) < float(neutral["charges"][0])
    assert float(biased["charges"].sum()) == pytest.approx(0.0, abs=1e-12)


# physics

def test_forces_are_minus_gradient():
    model = ForceStressOutput(_small()).eval()
    s = {**_structure(6, seed=4), "total_charge": 1.0, "external_potential": np.linspace(-0.2, 0.2, 6)}
    out = model(structure_to_graph(s, 4.5))
    pos = np.asarray(s["pos"], dtype=float)
    h = 1e-5
    for (i, k) in ((0, 0), (3, 1), (5, 2)):
        p = pos.copy(); p[i, k] += h
        e_plus = float(model(structure_to_graph({**s, "pos": p}, 4.5))["energy"])
        p = pos.copy(); p[i, k] -= h
        e_minus = float(model(structure_to_graph({**s, "pos": p}, 4.5))["energy"])
        assert float(out["forces"][i, k]) == pytest.approx(-(e_plus - e_minus) / (2 * h), abs=1e-7)


def test_stress_is_the_strain_derivative():
    model = ForceStressOutput(_small(), compute_stress=True).eval()
    s = {**_structure(6, seed=8), "cell": np.eye(3) * 4.5 + 0.1 * np.ones((3, 3)), "pbc": [True] * 3,
         "total_charge": -1.0}
    out = model(structure_to_graph(s, 4.5))
    h = 1e-6
    volume = abs(np.linalg.det(s["cell"]))
    for a, b in ((0, 0), (1, 2), (2, 1)):
        energies = []
        for sign in (1, -1):
            eps = np.zeros((3, 3)); eps[a, b] += 0.5 * sign * h; eps[b, a] += 0.5 * sign * h
            strained = {**s, "pos": s["pos"] @ (np.eye(3) + eps), "cell": s["cell"] @ (np.eye(3) + eps)}
            energies.append(float(model(structure_to_graph(strained, 4.5))["energy"]))
        assert float(out["stress"][0, a, b]) == pytest.approx((energies[0] - energies[1]) / (2 * h) / volume,
                                                              abs=1e-7)


def test_invariance_and_permutation():
    model = _small().eval()
    s = {**_structure(7, seed=9), "total_charge": 1.0}
    out = model(structure_to_graph(s, 4.5))
    rot, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))
    turned = model(structure_to_graph({**s, "pos": s["pos"] @ rot.T + 1.3}, 4.5))
    assert abs(float(out["energy"] - turned["energy"])) < 1e-12
    assert torch.allclose(out["charges"], turned["charges"], atol=1e-12)
    perm = np.random.default_rng(1).permutation(7)
    swapped = model(structure_to_graph({**s, "pos": s["pos"][perm],
                                        "atomic_numbers": np.array(s["atomic_numbers"])[perm]}, 4.5))
    assert abs(float(out["energy"] - swapped["energy"])) < 1e-12
    assert torch.allclose(out["charges"][perm], swapped["charges"], atol=1e-12)


def test_batching_and_separated_copies():
    """A batch gives each structure's own energies and charges; two neutral copies far apart keep
    their charges and double the energy (the charge flows within the structure only through the
    shared multiplier, which is the same for both)."""
    model = _small().eval()
    graphs = [_graph(n, seed=n, total_charge=q) for n, q in ((4, 0.0), (6, 1.0), (9, -1.0))]
    out = model(collate(graphs))
    offset = 0
    for b, g in enumerate(graphs):
        one = model(g)
        assert abs(float(out["energy"][b] - one["energy"][0])) < 1e-12
        assert torch.allclose(out["charges"][offset:offset + g.num_nodes], one["charges"], atol=1e-12)
        offset += g.num_nodes
    s = _structure(5, seed=7)
    pair = {"pos": np.concatenate([s["pos"], s["pos"] + np.array([30.0, 0, 0])]),
            "atomic_numbers": list(s["atomic_numbers"]) * 2}
    one, two = model(structure_to_graph(s, 4.5)), model(structure_to_graph(pair, 4.5))
    assert float(two["energy"]) == pytest.approx(2 * float(one["energy"]), abs=1e-10)
    assert torch.allclose(two["charges"][:5], one["charges"], atol=1e-12)


def test_pes_smooth_across_cutoff():
    """The trunk's cosine cutoff and the potential's polynomial envelope make the energy and the
    forces continuous when a neighbor leaves the cutoff sphere."""
    model = ForceStressOutput(_small(cutoff=4.0)).eval()

    def evaluate(d):
        pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.9, 0.0], [math.sqrt(d ** 2 - 0.45 ** 2), 0.45, 0.0]])
        out = model(structure_to_graph({"pos": pos, "atomic_numbers": [8, 1, 1], "total_charge": 1.0}, 4.0))
        return float(out["energy"]), out["forces"].detach().clone()

    eps = 1e-7
    (e_in, f_in), (e_out, f_out) = evaluate(4.0 - eps), evaluate(4.0 + eps)
    assert abs(e_in - e_out) < 1e-10
    assert float((f_in - f_out).abs().max()) < 1e-5


def test_float32_forward_matches_float64():
    model = _small().eval()
    g = _graph(n=8, total_charge=1.0)
    out64 = model(g)
    out32 = model.float()(structure_to_graph({"pos": g.pos.float(), "atomic_numbers": g.atomic_numbers,
                                              "total_charge": 1.0}, 4.5))
    scale = max(1.0, abs(float(out64["energy"])))
    assert abs(float(out64["energy"] - out32["energy"].double())) < 1e-4 * scale
    assert torch.allclose(out64["charges"], out32["charges"].double(), atol=1e-5)


# data, deployment and training

def test_external_potential_through_the_data_layer():
    """The applied potential is a per-atom field: built from a structure, zero for the structures of
    a batch that lack it, kept by sub-batches, read from ASE arrays."""
    g1, g2 = _graph(n=4, seed=1, potential=[0.1, 0.2, 0.3, 0.4]), _graph(n=3, seed=2)
    batch = collate([g1, g2])
    assert torch.equal(batch.external_potential, torch.tensor([0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0]))
    sub = batch.subset(torch.tensor([True, False]))
    assert torch.equal(sub.external_potential, g1.external_potential)
    assert collate([g2, g2]).external_potential is None
    ase = pytest.importorskip("ase")
    from xnn.common.data.ase_io import atoms_to_structure
    atoms = ase.Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    atoms.set_array("external_potential", np.array([0.5, 0.0, -0.5]))
    assert np.array_equal(atoms_to_structure(atoms)["external_potential"], [0.5, 0.0, -0.5])


def test_ase_calculator_reports_charges_and_reads_the_potential():
    ase = pytest.importorskip("ase")
    from xnn.common.deploy import XNNCalculator
    model = ForceStressOutput(_small()).eval()
    atoms = ase.Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    atoms.info["total_charge"] = 1.0
    atoms.calc = XNNCalculator(model, cutoff=4.5)
    q = atoms.get_charges()
    assert q.sum() == pytest.approx(1.0, abs=1e-12)
    atoms.set_array("external_potential", np.array([0.5, 0.0, -0.5]))
    atoms.calc = XNNCalculator(model, cutoff=4.5)
    assert not np.allclose(atoms.get_charges(), q)


def test_scriptable_and_exports(tmp_path):
    """The TorchScript core gives the eager charges and energies; the whole-system export carries a
    fixed total charge; the LAMMPS export evaluates the whole neutral system."""
    from xnn.common.deploy import export_to_lammps, export_torchscript_potential

    model = _small().eval()
    g = _graph(n=6, periodic=True)
    scripted = torch.jit.script(model)
    args = (g.atomic_numbers, g.edge_index, g.edge_vectors(), g.pos, g.cell[0], g.pbc[0], 1.0, 1.0)
    for a, b in zip(scripted.node_features_energy_charges(*args), model.node_features_energy_charges(*args)):
        assert (a - b).abs().max() < 1e-12

    s = _structure(6, seed=5)
    path = str(tmp_path / "qet_q.pt")
    export_torchscript_potential(model, model.cutoff, path, total_charge=1.0)
    out = torch.jit.load(path)(torch.as_tensor(s["pos"]), torch.as_tensor(s["atomic_numbers"]))
    ref = ForceStressOutput(model)(structure_to_graph({**s, "total_charge": 1.0}, 4.5))
    assert abs(float(out["energy"]) - float(ref["energy"])) < 1e-8
    assert (out["forces"] - ref["forces"].detach()).abs().max() < 1e-7
    assert (out["charges"] - ref["charges"].detach()).abs().max() < 1e-10

    path = str(tmp_path / "qet_lammps.pt")
    export_to_lammps(model, 4.5, path)
    out = torch.jit.load(path)(g.pos, g.edge_index, g.cell_shifts, g.atomic_numbers, g.cell[0])
    ref = ForceStressOutput(model)(g)
    assert abs(float(out["total_energy"]) - float(ref["energy"])) < 1e-8


def test_mdi_engine_serves_charged_systems(tmp_path):
    """The MDI engine passes the net charge to the charge equilibration."""
    from xnn.common.deploy.mdi_engine import BOHR_TO_ANGSTROM, HARTREE_TO_EV, MDIEngine
    cfg = from_dict({"model": {"name": "qet", "n_features": 8, "n_interactions": 1, "n_rbf": 4, "cutoff": 4.5}})
    torch.manual_seed(0)
    model = ForceStressOutput(build_model(cfg.model), compute_stress=True)
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
    ref = model(structure_to_graph({"pos": pos, "atomic_numbers": [8, 1, 1], "total_charge": 1.0}, 4.5))
    assert energies[1.0] == pytest.approx(float(ref["energy"]), abs=1e-5)
    assert abs(energies[1.0] - energies[0.0]) > 1e-6


def test_train_step_with_charges(tmp_path):
    """A force and charge loss trains every part of the model (second derivatives included)."""
    from xnn.common.data import AtomicDataset
    from xnn.common.train import Trainer

    rng = np.random.default_rng(0)
    structures = []
    for i in range(6):
        s = _structure(5, seed=i)
        q = rng.normal(scale=0.2, size=5)
        s.update(energy=float(rng.normal()), forces=rng.normal(size=(5, 3)), charges=q - q.mean(),
                 total_charge=0.0)
        structures.append(s)
    cfg = from_dict({"model": {"name": "qet", "n_features": 8, "n_interactions": 1, "n_rbf": 4,
                               "cutoff": 4.5, "extra": {"hardness": "environment"}},
                     "optim": {"epochs": 2, "charge_weight": 1.0},
                     "data": {"batch_size": 3, "val_fraction": 0.0},
                     "output_dir": str(tmp_path)})
    logs = Trainer(cfg, AtomicDataset(structures, 4.5)).fit()["train"]
    assert {"energy_mse", "force_mse", "charge_mse"} <= set(logs)


def test_shipped_config_and_matgl_spellings():
    import yaml
    path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "model", "qet.yaml")
    with open(path) as f:
        shipped = build_model(from_dict({"model": yaml.safe_load(f)}).model)
    assert sum(p.numel() for p in shipped.parameters()) == sum(p.numel() for p in QET().parameters())
    cfg = from_dict({"model": {"name": "qet", "units": 16, "nblocks": 1, "num_rbf": 10, "cutoff": 4.0,
                               "width": 0.8, "is_sigma_train": True,
                               "equivariance_invariance_group": "SO(3)"}})
    m = build_model(cfg.model)
    assert m.n_features == 16 and len(m.layers) == 1 and m.rbf.centers.numel() == 10
    assert m.rbf.gamma == 0.8 and float(m.rbf.centers[-1]) == 5.0 and m.equivariance_group == "SO(3)"
    assert isinstance(m.sigma, torch.nn.Parameter)


# MatGL conversion

def _matgl_qet_directory(path, environment=False, trainable_widths=False, gaussian=False, seed=0,
                         widths=(0.31, 0.76, 0.66, 1.11)):
    """A MatGL QET checkpoint directory with random weights over H, C, O and Si."""
    from test_tensornet import _matgl_directory
    sd = _matgl_directory(path, units=8, n_layers=2, seed=seed)
    gen = torch.Generator().manual_seed(seed + 100)

    def rand(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    F = 8
    for branch in ("layers", "gates"):
        for k, (a, b) in enumerate([(F + 2, F + 2), (F + 2, F), (F, F), (F, 1)]):
            sd[f"model.final_layer.gated.{branch}.{2 * k}.weight"] = rand(b, a)
            sd[f"model.final_layer.gated.{branch}.{2 * k}.bias"] = rand(b)
    heads = ["chi_readout"] + (["hardness_readout"] if environment else [])
    for head in heads:
        for k, (a, b) in enumerate([(F, F), (F, F), (F, 1)]):
            sd[f"model.{head}.layers.{2 * k}.weight"] = rand(b, a)
            sd[f"model.{head}.layers.{2 * k}.bias"] = rand(b)
    if not environment:
        sd["model.hardness_readout"] = rand(4).abs() + 0.5
    sd["model.sigma"] = rand(4).abs() + 0.5 if trainable_widths else torch.tensor(widths, dtype=torch.float32)
    sd["model.norm.weight"], sd["model.norm.bias"] = rand(F + 2), rand(F + 2)
    torch.save(sd, path / "state.pt")
    meta = json.loads((path / "model.json").read_text())
    args = meta["kwargs"]["model"]["init_args"]
    for key in ("is_intensive",):
        args.pop(key)
    args.update(is_hardness_envs=environment, is_sigma_train=trainable_widths, include_magmom=False)
    if gaussian:
        args.update(rbf_type="Gaussian", num_rbf=3, width=0.7)
    meta["kwargs"]["model"]["@class"] = "QET"
    meta["kwargs"]["calc_charge"] = True
    (path / "model.json").write_text(json.dumps(meta))
    return sd


@pytest.mark.parametrize("environment, trainable_widths, gaussian",
                         [(False, False, False), (True, True, False), (False, False, True)])
def test_matgl_checkpoint_converts(tmp_path, environment, trainable_widths, gaussian):
    """A MatGL QET directory converts weight for weight: the trunk as for TensorNet, the charge
    networks layer by layer, the per-element tables by atomic number, the radial basis."""
    from xnn.gnn.models import qet_foundation as qf
    from xnn.gnn.models.matgl_format import convert_directory, is_matgl_dir
    sd = _matgl_qet_directory(tmp_path, environment, trainable_widths, gaussian)
    assert is_matgl_dir(tmp_path)
    model, cfg = qf.convert_directory(tmp_path)
    assert isinstance(model, QET) and not model.self_pairs and model.cutoff == 4.0
    assert model.hardness_mode == ("environment" if environment else "element")
    dt = model.linear.weight.dtype
    assert torch.equal(model.chi_readout.layers[2].weight, sd["model.chi_readout.layers.4.weight"].to(dt))
    assert torch.equal(model.readout.mlp.layers[0].weight, sd["model.final_layer.gated.layers.0.weight"].to(dt))
    assert torch.equal(model.readout_norm.bias, sd["model.norm.bias"].to(dt))
    if environment:
        assert torch.equal(model.hardness_readout.mlp.layers[1].weight,
                           sd["model.hardness_readout.layers.2.weight"].to(dt))
    else:
        table = model.hardness_readout.weight
        assert torch.equal(table[[1, 6, 8, 14]], sd["model.hardness_readout"].to(dt)) and float(table[2]) == 1.0
    if trainable_widths:
        assert torch.equal(model.sigma[[1, 6, 8, 14]], sd["model.sigma"].to(dt))
    else:
        assert float(model.sigma[14]) == pytest.approx(1.11)
    if gaussian:
        assert cfg.model.extra["radial_basis"] == "gaussian" and model.rbf.gamma == pytest.approx(0.7)
        assert float(model.rbf.centers[-1]) == 5.0
    model = model.double()
    rebuilt = build_model(cfg.model).double()
    rebuilt.load_state_dict(model.state_dict())
    g = _graph(n=8, cutoff=4.0, seed=5, periodic=True, total_charge=1.0)
    g.atomic_numbers = torch.tensor([1, 6, 8, 14, 1, 6, 8, 14])
    assert torch.allclose(rebuilt(g)["energy"], model(g)["energy"])
    hub, _ = convert_directory(tmp_path)
    assert isinstance(hub, QET)


def test_matgl_checkpoint_refusals(tmp_path):
    """A magnetic-moment head, the SO(3) group (MatGL's edge vectors point the other way), widths other than the
    covalent radii and another class are refused."""
    from xnn.gnn.models import qet_foundation as qf
    _matgl_qet_directory(tmp_path, widths=(0.31, 0.76, 0.66, 1.2))
    with pytest.raises(ValueError, match="covalent radii"):
        qf.convert_directory(tmp_path)
    meta = json.loads((tmp_path / "model.json").read_text())
    meta["kwargs"]["model"]["init_args"].update(include_magmom=True, equivariance_invariance_group="SO(3)")
    (tmp_path / "model.json").write_text(json.dumps(meta))
    with pytest.raises(NotImplementedError) as err:
        qf.convert_directory(tmp_path)
    assert "magnetic-moment head" in str(err.value) and "SO(3) group" in str(err.value)
    meta["kwargs"]["model"]["@class"] = "TensorNet"
    (tmp_path / "model.json").write_text(json.dumps(meta))
    with pytest.raises(NotImplementedError, match="not a QET"):
        qf.convert_directory(tmp_path)
    with pytest.raises(KeyError):
        qf.from_foundation("qet-unknown")


def test_matgl_checkpoint_loads_from_the_foundation_cache(tmp_path):
    """A registered model already in the cache loads with QET.from_foundation in the asked dtype."""
    from xnn.gnn.models import qet_foundation as qf
    name = "qet-matpes-r2scan-2025.2"
    _, revision = qf.FOUNDATION_MODELS[name]
    folder = tmp_path / "models" / "matgl" / name / revision
    folder.mkdir(parents=True)
    sd = _matgl_qet_directory(folder, environment=True)
    model = QET.from_foundation(name, dtype="float64", cache_dir=tmp_path / "models")
    assert not model.training and model.chi_readout.layers[0].weight.dtype == torch.float64
    assert torch.equal(model.chi_readout.layers[0].weight, sd["model.chi_readout.layers.0.weight"].double())


# fused trunk (TensorNet's kernels)

def _needs_interpreter():
    if os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("runs where Triton's interpreter is on (no CUDA device)")
    triton = pytest.importorskip("triton")
    if tuple(int(v) for v in triton.__version__.split(".")[:2]) < (3, 2):
        pytest.skip("Triton's interpreter before 3.2 mis-evaluates these kernels")


def _fast_batch(dtype, device):
    graphs = [_graph(n=14, seed=1, periodic=True, total_charge=1.0),
              _graph(n=9, seed=2, total_charge=-1.0, potential=np.linspace(-0.2, 0.2, 9))]
    batch = collate(graphs).to(device)
    batch.pos = batch.pos.to(dtype)
    batch.cell = batch.cell.to(dtype)
    return batch


def _fast_vs_reference(model, batch, keys=("energy", "forces", "charges")):
    wrapped = ForceStressOutput(model).eval()
    out = {}
    for use_fast in (False, True):
        model.set_use_fast(use_fast)
        res = wrapped(batch)
        out[use_fast] = {k: res[k].detach() for k in keys}
    return max(float((out[True][k] - ref).abs().max() / ref.abs().max()) for k, ref in out[False].items())


def test_fast_path_parity_interpreted(monkeypatch):
    """The fused trunk under Triton's interpreter gives the reference energies, forces and charges."""
    _needs_interpreter()
    import xnn.gnn.fast.tensornet as fast
    monkeypatch.setattr(fast, "supported", lambda device, dtype: True)
    model = _small(n_features=16).eval()
    assert _fast_vs_reference(model, _fast_batch(torch.float64, "cpu")) < 1e-12


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype, tol", [(torch.float32, 1e-5), (torch.float64, 1e-12)])
def test_fast_path_parity(dtype, tol):
    pytest.importorskip("triton")
    model = _small(n_features=16, hardness="environment").to("cuda", dtype).eval()
    assert _fast_vs_reference(model, _fast_batch(dtype, "cuda")) < tol


# fidelity to MatGL (optional)

def _matgl_fidelity(dtype):
    pytest.importorskip("matgl")
    import subprocess
    paths = [p for p in os.environ.get("MATGL_QET_PATHS", "").split(os.pathsep) if p]
    script = os.path.join(os.path.dirname(__file__), "qet_matgl_fidelity.py")
    res = subprocess.run([sys.executable, script, *paths, dtype], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    return float(res.stdout.strip().splitlines()[-1])


def test_fidelity_vs_matgl_float64():
    assert _matgl_fidelity("float64") < 1e-10


def test_fidelity_vs_matgl_float32():
    assert _matgl_fidelity("float32") < 1e-4
