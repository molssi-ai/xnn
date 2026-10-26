"""Equiformer tests: paper sizes, the building blocks, invariances and equivariances,
forces and stress, smoothness at the cutoff, TorchScript deployment, configs and
(optionally) fidelity to the reference code."""
import copy
import importlib.util
import math
import os

import numpy as np
import pytest
import torch

e3nn = pytest.importorskip("e3nn")
from e3nn import o3  # noqa: E402

from xnn.common.config import from_dict  # noqa: E402
from xnn.common.data import collate, structure_to_graph  # noqa: E402
from xnn.common.models import ForceStressOutput, available_models, build_model  # noqa: E402
from xnn.trfnn.models.blocks import segment_softmax  # noqa: E402
from xnn.trfnn.models.equiformer import (  # noqa: E402
    Equiformer,
    EquivariantLayerNorm,
    Gate,
    IrrepsLinear,
    paper_irreps,
)


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _structure(n=6, seed=0, spread=3.0):
    rng = np.random.default_rng(seed)
    return {"pos": rng.uniform(0, spread, (n, 3)), "atomic_numbers": ([1, 6, 8] * n)[:n]}


def _graph(n=6, cutoff=4.0, seed=0, periodic=False, cell=6.0):
    s = _structure(n, seed)
    if periodic:
        s["cell"] = np.eye(3) * cell + np.array([[0.0, 0.3, 0.0], [0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
        s["pbc"] = [True, True, True]
    return structure_to_graph(s, cutoff)


def _small(seed=0, l_max=2, e3=False, **kw):
    torch.manual_seed(seed)
    kw = {**paper_irreps(l_max, e3, n_features=32), **kw}
    kw.setdefault("n_interactions", 2)
    kw.setdefault("cutoff", 4.0)
    kw.setdefault("n_rbf", 8)
    kw.setdefault("radial_hidden", [12, 12])
    kw.setdefault("irreps_feature", "24x0e")
    model = Equiformer(**kw)
    # random biases and norm weights, so the zero-initialized ones hide nothing
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("bias") or "affine" in name or "norm" in name:
                p.add_(0.1 * torch.randn_like(p))
    return model


def _n_params(model):
    return sum(p.numel() for name, p in model.named_parameters() if not name.startswith("atom_ref"))


def _random_rotation(seed, proper=True):
    Q = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(seed)))[0]
    if (torch.det(Q) > 0) != proper:
        Q = Q @ torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
    return Q


def _transform(s, R, t):
    pos = torch.as_tensor(s["pos"]) @ R.T + t
    return {**s, "pos": pos}


# registry / paper sizes

def test_registered():
    assert "equiformer" in available_models()


@pytest.mark.parametrize("attention, nonlinear, expected", [
    ("mlp", True, 3.53e6), ("mlp", False, 3.01e6), ("dot_product", False, 3.35e6)])
def test_qm9_sizes_of_table_6(attention, nonlinear, expected):
    """QM9 models: Gaussian basis of 128, five atom types, l_max = 2."""
    model = Equiformer(**paper_irreps(2), n_rbf=128, radial_basis="gaussian", n_elements=5,
                       attention=attention, nonlinear_message=nonlinear)
    assert _n_params(model) == pytest.approx(expected, rel=2e-3)


@pytest.mark.parametrize("l_max, expected", [(2, 3.50e6), (3, 5.50e6)])
def test_md17_sizes_of_section_e1(l_max, expected):
    model = Equiformer(**paper_irreps(l_max), n_rbf=32, n_elements=64)
    assert _n_params(model) == pytest.approx(expected, rel=1e-2)


def test_paper_irreps_presets_and_scaling():
    assert paper_irreps(2)["irreps_node_embedding"] == "128x0e+64x1e+32x2e"
    assert paper_irreps(1)["num_heads"] == 8
    half = paper_irreps(3, n_features=64)
    assert half["irreps_node_embedding"] == "64x0e+32x1e+32x2e+16x3e"
    assert half["irreps_head"] == "16x0e+8x1e+8x2e+4x3e"
    assert paper_irreps(2, e3=True)["irreps_node_embedding"].count("o") == 3
    with pytest.raises(ValueError):
        paper_irreps(5)


# building blocks

def test_irreps_linear_is_equivariant_with_scalar_bias():
    lin = IrrepsLinear("4x0e+3x1o+2x2e", "5x0e+2x1o+3x2e+1x3o")
    with torch.no_grad():
        lin.bias.normal_()
    x = o3.Irreps("4x0e+3x1o+2x2e").randn(7, -1)
    R = _random_rotation(1)
    D_in = o3.Irreps("4x0e+3x1o+2x2e").D_from_matrix(R)
    D_out = o3.Irreps("5x0e+2x1o+3x2e+1x3o").D_from_matrix(R)
    assert torch.allclose(lin(x @ D_in.T), lin(x) @ D_out.T, atol=1e-12)
    assert lin(x)[:, -7:].abs().max() == 0          # no input of degree 3


def test_layer_norm_statistics():
    irreps = o3.Irreps("6x0e+4x1e+3x2e")
    norm = EquivariantLayerNorm(irreps)
    x = 3.0 * irreps.randn(5, -1) + 1.0
    y = norm(x)
    scalars = y[:, :6]
    assert torch.allclose(scalars.mean(dim=1), torch.zeros(5), atol=1e-12)
    assert torch.allclose(scalars.pow(2).mean(dim=1), torch.ones(5), atol=1e-4)
    vec = y[:, 6:18].reshape(5, 4, 3)
    assert torch.allclose(vec.pow(2).mean(dim=(1, 2)), torch.ones(5), atol=1e-4)


def test_gate_layout_and_equivariance():
    gate = Gate("8x0e+4x1e+2x2e")
    assert str(gate.irreps_in) == "14x0e+4x1e+2x2e" and str(gate.irreps_out) == "8x0e+4x1e+2x2e"
    x = gate.irreps_in.randn(6, -1)
    R = _random_rotation(2)
    y1 = gate(x @ gate.irreps_in.D_from_matrix(R).T)
    y2 = gate(x) @ gate.irreps_out.D_from_matrix(R).T
    assert torch.allclose(y1, y2, atol=1e-12)
    # the activations are normalized to unit second moment on a standard normal
    z = torch.randn(200_000, generator=torch.Generator().manual_seed(0))
    assert float((torch.nn.functional.silu(z) * gate.silu_scale).pow(2).mean()) == pytest.approx(1.0, rel=2e-2)


def test_segment_softmax_plain_and_smooth():
    logits = torch.randn(7, 3)
    index = torch.tensor([0, 0, 1, 1, 1, 3, 3])
    a = segment_softmax(logits, index, 4)
    sums = torch.zeros(4, 3).index_add(0, index, a)
    assert torch.allclose(sums[[0, 1, 3]], torch.ones(3, 3), atol=1e-12)
    w = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    s = segment_softmax(logits, index, 4, w)
    assert (torch.zeros(4, 3).index_add(0, index, s)[[0, 1, 3]] < 1.0).all()
    w[5:] = 0.0                                      # both neighbors of atom 3 at the cutoff
    s = segment_softmax(logits, index, 4, w)
    assert s[5:].abs().max() == 0


def test_depthwise_tensor_product_matches_e3nn_paths():
    from xnn.trfnn.models.equiformer import depthwise_tensor_product
    tp = depthwise_tensor_product("8x0e+4x1e+2x2e", "1x0e+1x1e+1x2e", "8x0e+4x1e+2x2e", False)
    assert all(ins.connection_mode == "uvu" for ins in tp.instructions)
    # sorted by degree, every output path reaching 0e .. 2e
    assert [ir.l for _, ir in tp.irreps_out] == sorted(ir.l for _, ir in tp.irreps_out)
    assert tp.irreps_out.lmax == 2


# invariances and equivariances

@pytest.mark.parametrize("e3", [False, True])
def test_rotation_invariance_and_force_equivariance(e3):
    model = ForceStressOutput(_small(e3=e3)).eval()
    s = _structure(7, seed=1)
    R, t = _random_rotation(4), torch.tensor([1.3, -0.7, 2.1])
    out = model(structure_to_graph(s, 4.0))
    out2 = model(structure_to_graph(_transform(s, R, t), 4.0))
    assert abs(float(out["energy"] - out2["energy"])) < 1e-10
    assert torch.allclose(out["forces"] @ R.T, out2["forces"], atol=1e-10)
    assert torch.allclose(out["node_features"], out2["node_features"], atol=1e-10)


def test_reflections_distinguish_se3_from_e3():
    """The SE(3) model of the paper labels every irrep even, so it tells mirror images
    apart; the E(3) model is invariant under reflections."""
    s = _structure(7, seed=3)
    R = _random_rotation(5, proper=False)
    for e3, invariant in ((False, False), (True, True)):
        model = _small(e3=e3).eval()
        e1 = float(model(structure_to_graph(s, 4.0))["energy"])
        e2 = float(model(structure_to_graph(_transform(s, R, torch.zeros(3)), 4.0))["energy"])
        assert (abs(e1 - e2) < 1e-10) == invariant


def test_permutation_invariance():
    model = _small().eval()
    s = _structure(7, seed=2)
    perm = np.random.default_rng(1).permutation(7)
    out = model(structure_to_graph(s, 4.0))
    out2 = model(structure_to_graph({"pos": s["pos"][perm],
                                     "atomic_numbers": np.array(s["atomic_numbers"])[perm]}, 4.0))
    assert abs(float(out["energy"] - out2["energy"])) < 1e-12
    assert torch.allclose(out["node_energy"][perm], out2["node_energy"], atol=1e-12)


@pytest.mark.parametrize("options", [{}, {"attention": "dot_product", "nonlinear_message": False},
                                     {"nonlinear_message": False}, {"radial_basis": "gaussian"},
                                     {"radial_basis": "bessel", "l_max": 3}])
def test_forces_are_minus_gradient(options):
    model = ForceStressOutput(_small(**options)).eval()
    s = _structure(6, seed=4)
    out = model(structure_to_graph(s, 4.0))
    pos = np.asarray(s["pos"], dtype=float)
    h = 1e-5
    for (i, k) in ((0, 0), (3, 1), (5, 2)):
        p = pos.copy(); p[i, k] += h
        e_plus = float(model(structure_to_graph({**s, "pos": p}, 4.0))["energy"])
        p = pos.copy(); p[i, k] -= h
        e_minus = float(model(structure_to_graph({**s, "pos": p}, 4.0))["energy"])
        assert float(out["forces"][i, k]) == pytest.approx(-(e_plus - e_minus) / (2 * h), abs=1e-7)


def test_periodic_stress_is_the_strain_derivative():
    model = ForceStressOutput(_small(), compute_stress=True).eval()
    g = _graph(n=6, periodic=True, cell=4.5)
    out = model(g)
    s = {"pos": g.pos.detach().numpy(), "atomic_numbers": g.atomic_numbers.numpy(),
         "cell": g.cell[0].detach().numpy(), "pbc": [True, True, True]}
    volume = abs(np.linalg.det(s["cell"]))
    h = 1e-6
    for a, b in ((0, 0), (1, 2), (2, 1)):
        energies = []
        for sign in (1, -1):
            eps = np.zeros((3, 3)); eps[a, b] += 0.5 * sign * h; eps[b, a] += 0.5 * sign * h
            strained = {**s, "pos": s["pos"] @ (np.eye(3) + eps), "cell": s["cell"] @ (np.eye(3) + eps)}
            energies.append(float(model(structure_to_graph(strained, 4.0))["energy"]))
        assert float(out["stress"][0, a, b]) == pytest.approx((energies[0] - energies[1]) / (2 * h) / volume,
                                                              abs=1e-7)


def test_size_extensivity_and_batching():
    model = _small().eval()
    graphs = [structure_to_graph(_structure(n, seed=n), 4.0) for n in (4, 6, 9)]
    singles = [model(g) for g in graphs]
    out = model(collate(graphs))
    for b, one in enumerate(singles):
        assert abs(float(out["energy"][b] - one["energy"][0])) < 1e-12
    s = _structure(5, seed=7)
    pair = {"pos": np.concatenate([s["pos"], s["pos"] + np.array([30.0, 0, 0])]),
            "atomic_numbers": list(s["atomic_numbers"]) * 2}
    e1 = float(model(structure_to_graph(s, 4.0))["energy"])
    e2 = float(model(structure_to_graph(pair, 4.0))["energy"])
    assert e2 == pytest.approx(2 * e1, abs=1e-10)


def test_supercell_energy_is_extensive():
    model = _small().eval()
    s = {"pos": np.random.default_rng(3).uniform(0, 5.0, (5, 3)), "atomic_numbers": [1, 6, 8, 6, 1],
         "cell": np.diag([5.0, 5.2, 5.4]), "pbc": [True, True, True]}
    shift = np.array([5.0, 0.0, 0.0])
    double = {"pos": np.concatenate([s["pos"], s["pos"] + shift]),
              "atomic_numbers": s["atomic_numbers"] * 2,
              "cell": np.diag([10.0, 5.2, 5.4]), "pbc": [True, True, True]}
    e1 = float(model(structure_to_graph(s, 4.0))["energy"])
    e2 = float(model(structure_to_graph(double, 4.0))["energy"])
    assert e2 == pytest.approx(2 * e1, rel=1e-10)


def _crossing(model, d):
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.9, 0.0], [math.sqrt(d ** 2 - 0.45 ** 2), 0.45, 0.0]])
    out = model(structure_to_graph({"pos": pos, "atomic_numbers": [8, 1, 1]}, 4.0))
    return float(out["energy"]), out["forces"].detach().clone()


@pytest.mark.parametrize("basis", ["expnorm", "gaussian"])
def test_smooth_cutoff_makes_the_energy_continuous(basis):
    eps = 1e-7
    smooth = ForceStressOutput(_small(radial_basis=basis, smooth_cutoff=True)).eval()
    (e_in, f_in), (e_out, f_out) = _crossing(smooth, 4.0 - eps), _crossing(smooth, 4.0 + eps)
    assert abs(e_in - e_out) < 1e-10
    assert float((f_in - f_out).abs().max()) < 1e-5
    paper = ForceStressOutput(_small(radial_basis=basis)).eval()
    (e_in, _), (e_out, _) = _crossing(paper, 4.0 - eps), _crossing(paper, 4.0 + eps)
    assert abs(e_in - e_out) > 1e-6


def test_last_neighbor_fades_out_with_smooth_cutoff():
    model = _small(smooth_cutoff=True).eval()

    def energy(d):
        pos = np.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]])
        return float(model(structure_to_graph({"pos": pos, "atomic_numbers": [8, 1]}, 4.0))["energy"])

    assert abs(energy(4.0 - 1e-7) - energy(4.0 + 1e-7)) < 1e-10


def test_isolated_atoms_have_finite_gradients():
    model = ForceStressOutput(_small())
    g = structure_to_graph({"pos": [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]], "atomic_numbers": [8, 1]}, 4.0)
    out = model(g)
    out["energy"].sum().backward()
    assert torch.isfinite(out["forces"]).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_float32_matches_float64_and_exact_constants():
    model = _small().eval()
    g = _graph(n=8)
    e64 = float(model(g)["energy"])
    model32 = copy.deepcopy(model).float()
    g32 = structure_to_graph({"pos": g.pos.float(), "atomic_numbers": g.atomic_numbers}, 4.0)
    assert abs(float(model32(g32)["energy"]) - e64) < 1e-4 * max(1.0, abs(e64))
    # a float32 round trip leaves the coupling coefficients exact
    back = model32.double()
    for name, buf in back.named_buffers():
        if "_w3j_" in name:
            l1, l2, l3 = (int(v) for v in name.rsplit("_w3j_", 1)[1].split("_"))
            assert torch.equal(buf, o3.wigner_3j(l1, l2, l3, dtype=torch.float64))


def test_dropout_and_drop_path_only_in_training():
    model = _small(alpha_drop=0.3, proj_drop=0.2, out_drop=0.2, drop_path_rate=0.3)
    g = collate([_graph(n=5, seed=1), _graph(n=6, seed=2)])
    model.eval()
    assert torch.equal(model(g)["energy"], model(g)["energy"])
    model.train()
    torch.manual_seed(0)
    e1 = model(g)["energy"]
    torch.manual_seed(1)
    e2 = model(g)["energy"]
    assert not torch.allclose(e1, e2)


# deployment

def test_scriptable_and_lammps_export(tmp_path):
    from xnn.common.deploy import export_to_lammps

    model = _small().eval()
    g = _graph(n=6, periodic=True)
    scripted = torch.jit.script(model)
    d = (scripted.node_energy(g.atomic_numbers, g.edge_index, g.edge_vectors())
         - model.node_energy(g.atomic_numbers, g.edge_index, g.edge_vectors()))
    assert d.abs().max() < 1e-12
    path = str(tmp_path / "equiformer_lammps.pt")
    export_to_lammps(model, 4.0, path)
    loaded = torch.jit.load(path)
    out = loaded(g.pos, g.edge_index, g.cell_shifts, g.atomic_numbers, g.cell[0])
    ref = ForceStressOutput(model)(g)
    assert abs(float(out["total_energy"]) - float(ref["energy"])) < 1e-8
    assert (out["forces"] - ref["forces"].detach()).abs().max() < 1e-8


def test_scriptable_dot_product_and_e3():
    for options in ({"attention": "dot_product", "nonlinear_message": False}, {"e3": True},
                    {"nonlinear_message": False, "smooth_cutoff": True}):
        model = _small(**options).eval()
        g = _graph(n=5)
        scripted = torch.jit.script(model)
        d = scripted.node_energy(g.atomic_numbers, g.edge_index, g.edge_vectors()) - model(g)["node_energy"]
        assert d.abs().max() < 1e-12


def test_les_wrapper_reads_node_features():
    from xnn.common.models.les import LatentEwald

    model = LatentEwald(_small()).double()
    out = model(_graph())
    assert out["energy"].shape == (1,) and torch.isfinite(out["energy"]).all()


def test_multihead_and_reference_energies():
    from xnn.common.finetune import MultiHead, get_atomic_energies, set_atomic_energies

    model = _small()
    set_atomic_energies(model, {1: -0.5, 6: -1.0, 8: -2.0})
    assert get_atomic_energies(model) == {1: -0.5, 6: -1.0, 8: -2.0}
    multi = MultiHead(model, ["a", "b"])
    assert multi(_graph())["energy"].shape == (1,)


# training and config

def test_train_step(tmp_path):
    from xnn.common.config import Config
    from xnn.common.data import AtomicDataset
    from xnn.common.train import Trainer

    rng = np.random.default_rng(0)
    structures = []
    for i in range(6):
        s = _structure(5, seed=i)
        s.update(energy=float(rng.normal()), forces=rng.normal(size=(5, 3)))
        structures.append(s)
    cfg = from_dict({"model": {"name": "equiformer", "n_features": 16, "n_interactions": 1,
                               "n_rbf": 4, "cutoff": 4.0,
                               "extra": {"irreps_feature": "16x0e", "radial_hidden": [8]}},
                     "optim": {"epochs": 2},
                     "data": {"batch_size": 3, "val_fraction": 0.0},
                     "output_dir": str(tmp_path)})
    assert isinstance(cfg, Config)
    logs = Trainer(cfg, AtomicDataset(structures, 4.0)).fit()["train"]
    assert {"energy_mse", "force_mse"} <= set(logs)


def test_from_config_presets_and_extras():
    cfg = from_dict({"model": {
        "name": "equiformer", "cutoff": 4.5, "n_features": 64, "n_interactions": 2, "n_rbf": 16,
        "extra": {"l_max": 3, "radial_basis": "gaussian", "attention": "dot_product",
                  "nonlinear_message": False, "smooth_cutoff": True, "avg_degree": 12.0,
                  "avg_num_nodes": 20.0, "species": [1, 6, 8],
                  "atomic_energies": [-13.6, -1030.0, -2043.0]}}})
    m = build_model(cfg.model)
    assert str(m.irreps_node_embedding) == "64x0e+32x1e+32x2e+16x3e"
    assert m.cutoff == 4.5 and m.smooth and m.avg_num_nodes == 20.0
    assert m.edge_deg_embed.avg_degree == 12.0
    assert type(m.blocks[0].ga.core).__name__ == "_DotProductAttention"
    assert float(m.atom_ref.weight[6, 0]) == pytest.approx(-1030.0)
    assert m(_graph())["energy"].shape == (1,)


def test_reference_key_translation():
    cfg = from_dict({"model": {"name": "equiformer", "n_features": 32, "num_layers": 3,
                               "max_radius": 4.5, "number_of_basis": 20, "basis_type": "exp",
                               "fc_neurons": [16, 16]}})
    assert cfg.model.n_interactions == 3 and cfg.model.cutoff == 4.5 and cfg.model.n_rbf == 20
    m = build_model(cfg.model)
    assert m.rbf.n_rbf == 20 and m.edge_deg_embed.rad.net[0].out_features == 16


# fidelity to the reference implementation (optional)

def _fidelity(dtype):
    path = os.environ.get("EQUIFORMER_UPSTREAM_PATH")
    if not path:
        pytest.skip("set EQUIFORMER_UPSTREAM_PATH to a checkout of atomicarchitects/equiformer")
    if importlib.util.find_spec("fairchem") is None:
        pytest.skip("fairchem-core 1.x is not installed")
    pytest.importorskip("torch_geometric")
    import subprocess
    import sys
    script = os.path.join(os.path.dirname(__file__), "equiformer_fidelity.py")
    res = subprocess.run([sys.executable, script, "v1", path, dtype], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    return float(res.stdout.strip().splitlines()[-1])


def test_fidelity_vs_reference_float64():
    assert _fidelity("float64") < 1e-11


def test_fidelity_vs_reference_float32():
    assert _fidelity("float32") < 2e-4
