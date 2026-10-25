"""EquiformerV2 tests: paper sizes, edge frames and Wigner matrices, SO(2) convolutions,
sphere grids, invariances, exact forces and stress, smoothness at the cutoff, direct
forces, TorchScript deployment, configs and (optionally) fidelity to the reference code."""
import copy
import importlib.util
import math

import numpy as np
import pytest
import torch

e3nn = pytest.importorskip("e3nn")
from e3nn import o3  # noqa: E402

import xnn.trfnn.models.equiformer_v2 as eqv2  # noqa: E402
from xnn.common.config import from_dict  # noqa: E402
from xnn.common.data import collate, structure_to_graph  # noqa: E402
from xnn.common.models import ForceStressOutput, available_models, build_model  # noqa: E402
from xnn.trfnn.models.equiformer_v2 import (  # noqa: E402
    EdgeFrames,
    EquiformerV2,
    S2Grid,
    SO2Convolution,
    edge_rotation,
    m_primary_order,
    reduced_index,
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


def _small(seed=0, **kw):
    torch.manual_seed(seed)
    defaults = dict(n_features=8, n_interactions=2, cutoff=4.0, n_rbf=16, l_max=3, m_max=2,
                    num_heads=2, attn_hidden_channels=8, attn_alpha_channels=4,
                    attn_value_channels=4, ffn_hidden_channels=12, edge_channels=8,
                    grid_resolution=14)
    model = EquiformerV2(**{**defaults, **kw})
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
    return {**s, "pos": torch.as_tensor(s["pos"]) @ R.T + t}


def _energy(model, s, cutoff=4.0):
    return float(model(structure_to_graph(s, cutoff))["energy"])


# registry / paper sizes

def test_registered():
    assert "equiformer_v2" in available_models()


OC20 = dict(n_features=128, cutoff=12.0, n_rbf=600, l_max=6, m_max=2, n_interactions=12, num_heads=8,
            attn_hidden_channels=64, attn_alpha_channels=64, attn_value_channels=16,
            ffn_hidden_channels=128, edge_channels=128, n_elements=90, direct_forces=True)


@pytest.mark.parametrize("options, expected", [
    ({}, 83.16e6),                                                   # Table 8, S2EF-2M base
    ({"l_max": 4}, 44.83e6),
    ({"m_max": 3}, 95.11e6),
    ({"m_max": 4}, 102.14e6),
    ({"l_max": 4, "n_interactions": 8}, 31.06e6),                    # S2EF-All+MD, lambda_E = 4
])
def test_oc20_sizes_of_table_8(options, expected):
    model = EquiformerV2(**{**OC20, **options})
    assert _n_params(model) == pytest.approx(expected, rel=2e-3)


@pytest.mark.parametrize("blocks, expected", [(6, 11.20e6), (5, 9.35e6)])
def test_qm9_sizes_of_section_f2(blocks, expected):
    # the counts of the paper correspond to radial inputs without element embeddings
    model = EquiformerV2(n_interactions=blocks, use_atom_edge_embedding=False)
    assert _n_params(model) == pytest.approx(expected, rel=3e-3)


# edge frames and Wigner matrices

def test_edge_rotation_maps_edges_onto_y():
    vec = torch.randn(200, 3, generator=torch.Generator().manual_seed(0))
    vec = torch.cat([vec, torch.tensor([[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [1e-12, -1.0, 0.0],
                                        [0.0, -1.0, 1e-9], [3.0, 0.0, 0.0]])])
    R = edge_rotation(vec)
    n = vec / vec.norm(dim=1, keepdim=True)
    assert torch.allclose(torch.einsum("eij,ej->ei", R, n), torch.tensor([0.0, 1.0, 0.0]).expand_as(n),
                          atol=1e-12)
    assert torch.allclose(R @ R.transpose(1, 2), torch.eye(3).expand_as(R), atol=1e-12)
    assert torch.allclose(torch.linalg.det(R), torch.ones(len(vec)), atol=1e-12)


def test_edge_rotation_has_finite_gradients_at_the_poles():
    vec = torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [1e-13, -1.0, 0.0], [0.3, -1.0, 0.1]],
                       requires_grad=True)
    edge_rotation(vec).pow(2).sum().backward()
    assert torch.isfinite(vec.grad).all()


@pytest.mark.parametrize("lmax", [1, 3, 6])
def test_wigner_blocks_match_e3nn(lmax):
    frames = EdgeFrames(lmax)
    vec = torch.randn(40, 3, generator=torch.Generator().manual_seed(lmax))
    R = edge_rotation(vec)
    blocks = frames(vec)
    alpha, beta, gamma = o3.matrix_to_angles(R)
    for l, d in enumerate(blocks):
        ref = o3.wigner_D(l, alpha, beta, gamma)
        assert torch.allclose(d, ref, atol=1e-11), l


@pytest.mark.parametrize("lmax", [2, 4])
def test_rotated_harmonics_point_along_y(lmax):
    """The frame takes the harmonics of the edge onto those of the y axis."""
    frames = EdgeFrames(lmax)
    vec = torch.randn(30, 3, generator=torch.Generator().manual_seed(1))
    blocks = frames(vec)
    y = o3.spherical_harmonics(list(range(lmax + 1)), torch.tensor([0.0, 1.0, 0.0]), True)
    sh = o3.spherical_harmonics(list(range(lmax + 1)), vec, True)
    for l, d in enumerate(blocks):
        rot = torch.einsum("eab,eb->ea", d, sh[:, l * l:(l + 1) ** 2])
        assert torch.allclose(rot, y[l * l:(l + 1) ** 2].expand_as(rot), atol=1e-11)


def test_coefficient_layouts():
    assert reduced_index(3, 1) == [0, 1, 2, 3, 5, 6, 7, 11, 12, 13]
    order = m_primary_order(3, 1)
    assert sorted(order) == list(range(10)) and order[:4] == [0, 2, 5, 8]


# SO(2) convolution and sphere grids

def _roll(lmax, mmax, theta):
    """The reduced-layout matrix of a rotation by theta about the y axis."""
    D = o3.Irreps([(1, (l, 1)) for l in range(lmax + 1)]).D_from_angles(
        torch.tensor(theta), torch.tensor(0.0), torch.tensor(0.0))
    idx = torch.tensor(reduced_index(lmax, mmax))
    return D[idx][:, idx]


@pytest.mark.parametrize("lmax, mmax", [(3, 1), (4, 2), (3, 3)])
def test_so2_convolution_commutes_with_rolls(lmax, mmax):
    torch.manual_seed(0)
    conv = SO2Convolution(5, 3, lmax, mmax, radial=[4, 6], extra_m0=2)
    n = len(reduced_index(lmax, mmax))
    x, edge = torch.randn(7, n, 5), torch.randn(7, 4)
    roll = _roll(lmax, mmax, 0.83)
    y, extra = conv(x, edge)
    y2, extra2 = conv(torch.einsum("ab,ebc->eac", roll, x), edge)
    assert torch.allclose(torch.einsum("ab,ebc->eac", roll, y), y2, atol=1e-12)
    assert torch.allclose(extra, extra2, atol=1e-12)


def test_s2_grid_round_trip():
    full = S2Grid(4, 4, 18)
    x = torch.randn(3, 25, 2)
    assert torch.allclose(full.from_grid(full.to_grid(x)), x, atol=1e-10)
    cut = S2Grid(4, 2, 18)
    idx = reduced_index(4, 2)
    x = torch.randn(3, len(idx), 2)
    scale = torch.tensor([((2 * l + 1) / 5.0 if l > 2 else 1.0) for l in range(5)
                          for _ in range(2 * min(l, 2) + 1)])
    assert torch.allclose(cut.from_grid(cut.to_grid(x)), x * scale[:, None], atol=1e-10)


# invariances and equivariances

def test_gate_model_is_exactly_invariant():
    model = ForceStressOutput(_small(activation="gate", use_grid_mlp=False)).eval()
    s = _structure(7, seed=1)
    R, t = _random_rotation(4), torch.tensor([1.3, -0.7, 2.1])
    out = model(structure_to_graph(s, 4.0))
    out2 = model(structure_to_graph(_transform(s, R, t), 4.0))
    assert abs(float(out["energy"] - out2["energy"])) < 1e-10
    assert torch.allclose(out["forces"] @ R.T, out2["forces"], atol=1e-10)


def test_s2_models_are_invariant_up_to_the_sphere_sampling():
    s = _structure(7, seed=1)
    R = _random_rotation(4)
    errors = []
    for res in (10, 30):
        model = _small(grid_resolution=res).eval()
        e1, e2 = _energy(model, s), _energy(model, _transform(s, R, torch.zeros(3)))
        errors.append(abs(e1 - e2) / max(1.0, abs(e1)))
    assert errors[1] < 1e-3 and errors[1] < errors[0]


def test_roll_of_the_frames_only_matters_through_the_sampling(monkeypatch):
    s = _structure(7, seed=5)
    exact = _small(activation="gate", use_grid_mlp=False).eval()
    sampled = _small().eval()
    base = [_energy(exact, s), _energy(sampled, s)]
    frame = eqv2.edge_rotation
    c, sn = math.cos(0.7), math.sin(0.7)

    def rolled(vec):
        roll = torch.tensor([[c, 0.0, sn], [0.0, 1.0, 0.0], [-sn, 0.0, c]], dtype=vec.dtype)
        return roll @ frame(vec)

    monkeypatch.setattr(eqv2, "edge_rotation", rolled)
    assert abs(_energy(exact, s) - base[0]) < 1e-10
    assert 0.0 < abs(_energy(sampled, s) - base[1]) < 1e-3 * max(1.0, abs(base[1]))


def test_permutation_and_translation_invariance():
    model = _small().eval()
    s = _structure(7, seed=2)
    perm = np.random.default_rng(1).permutation(7)
    out = model(structure_to_graph(s, 4.0))
    out2 = model(structure_to_graph({"pos": s["pos"][perm],
                                     "atomic_numbers": np.array(s["atomic_numbers"])[perm]}, 4.0))
    assert abs(float(out["energy"] - out2["energy"])) < 1e-12
    assert torch.allclose(out["node_energy"][perm], out2["node_energy"], atol=1e-12)
    shifted = _energy(model, _transform(s, torch.eye(3), torch.tensor([3.0, -2.0, 1.0])))
    assert abs(shifted - float(out["energy"])) < 1e-12


# forces and stress

@pytest.mark.parametrize("options", [{}, {"norm_type": "rms_norm_sh", "activation": "gate",
                                          "use_grid_mlp": False},
                                     {"norm_type": "layer_norm", "activation": "s2", "use_grid_mlp": False,
                                      "m_max": 3, "use_m_share_rad": True}])
def test_forces_are_minus_gradient(options):
    model = ForceStressOutput(_small(**options)).eval()
    s = _structure(6, seed=4)
    out = model(structure_to_graph(s, 4.0))
    pos = np.asarray(s["pos"], dtype=float)
    h = 1e-5
    for (i, k) in ((0, 0), (3, 1), (5, 2)):
        p = pos.copy(); p[i, k] += h
        e_plus = _energy(model, {**s, "pos": p})
        p = pos.copy(); p[i, k] -= h
        e_minus = _energy(model, {**s, "pos": p})
        assert float(out["forces"][i, k]) == pytest.approx(-(e_plus - e_minus) / (2 * h), abs=1e-7)


def test_detached_frames_give_wrong_forces(monkeypatch):
    """Differentiating through the edge frames is required for the exact gradient."""
    model = ForceStressOutput(_small()).eval()
    g = _graph(n=7, seed=6)
    full = model(g)["forces"].detach()
    original = EdgeFrames.forward
    monkeypatch.setattr(EdgeFrames, "forward", lambda self, vec: [d.detach() for d in original(self, vec)])
    detached = model(_graph(n=7, seed=6))["forces"].detach()
    assert float((full - detached).abs().max()) > 1e-4 * float(full.abs().max())


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
            energies.append(_energy(model, strained))
        assert float(out["stress"][0, a, b]) == pytest.approx((energies[0] - energies[1]) / (2 * h) / volume,
                                                              abs=1e-7)


def test_size_extensivity_batching_and_supercells():
    model = _small().eval()
    graphs = [structure_to_graph(_structure(n, seed=n), 4.0) for n in (4, 6, 9)]
    singles = [model(g) for g in graphs]
    out = model(collate(graphs))
    for b, one in enumerate(singles):
        assert abs(float(out["energy"][b] - one["energy"][0])) < 1e-12
    s = {"pos": np.random.default_rng(3).uniform(0, 5.0, (5, 3)), "atomic_numbers": [1, 6, 8, 6, 1],
         "cell": np.diag([5.0, 5.2, 5.4]), "pbc": [True, True, True]}
    double = {"pos": np.concatenate([s["pos"], s["pos"] + np.array([5.0, 0.0, 0.0])]),
              "atomic_numbers": s["atomic_numbers"] * 2, "cell": np.diag([10.0, 5.2, 5.4]),
              "pbc": [True, True, True]}
    assert _energy(model, double) == pytest.approx(2 * _energy(model, s), rel=1e-10)


def _crossing(model, d):
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.9, 0.0], [math.sqrt(d ** 2 - 0.45 ** 2), 0.45, 0.0]])
    out = model(structure_to_graph({"pos": pos, "atomic_numbers": [8, 1, 1]}, 4.0))
    return float(out["energy"]), out["forces"].detach().clone()


def test_smooth_cutoff_makes_the_energy_continuous():
    eps = 1e-7
    smooth = ForceStressOutput(_small(smooth_cutoff=True)).eval()
    (e_in, f_in), (e_out, f_out) = _crossing(smooth, 4.0 - eps), _crossing(smooth, 4.0 + eps)
    assert abs(e_in - e_out) < 1e-10
    assert float((f_in - f_out).abs().max()) < 1e-5
    paper = ForceStressOutput(_small()).eval()
    (e_in, _), (e_out, _) = _crossing(paper, 4.0 - eps), _crossing(paper, 4.0 + eps)
    assert abs(e_in - e_out) > 1e-6


def test_isolated_atoms_have_finite_gradients():
    model = ForceStressOutput(_small())
    g = structure_to_graph({"pos": [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]], "atomic_numbers": [8, 1]}, 4.0)
    out = model(g)
    out["energy"].sum().backward()
    assert torch.isfinite(out["forces"]).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_axis_aligned_crystal_has_finite_forces():
    """Edges along +y and -y (the poles of the frames) are common in crystals."""
    s = {"pos": [[0.0, 0.0, 0.0], [1.8, 1.8, 0.0], [1.8, 0.0, 1.8], [0.0, 1.8, 1.8]],
         "atomic_numbers": [29, 29, 29, 29], "cell": np.eye(3) * 3.6, "pbc": [True, True, True]}
    out = ForceStressOutput(_small(), compute_stress=True)(structure_to_graph(s, 4.0))
    assert torch.isfinite(out["forces"]).all() and torch.isfinite(out["stress"]).all()
    # every atom is a center of symmetry: no force on an exactly equivariant model
    exact = ForceStressOutput(_small(activation="gate", use_grid_mlp=False))
    assert float(exact(structure_to_graph(s, 4.0))["forces"].abs().max()) < 1e-8


# direct forces

def test_direct_forces_are_equivariant():
    model = ForceStressOutput(_small(direct_forces=True, activation="gate", use_grid_mlp=False)).eval()
    s = _structure(7, seed=8)
    R = _random_rotation(9)
    f1 = model(structure_to_graph(s, 4.0))["forces"]
    f2 = model(structure_to_graph(_transform(s, R, torch.zeros(3)), 4.0))["forces"]
    assert f1.shape == (7, 3) and torch.allclose(f1 @ R.T, f2, atol=1e-10)
    with pytest.raises(RuntimeError):
        model.model.node_energy(torch.tensor([1, 6]), torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 3))


def test_float32_matches_float64_and_exact_constants():
    model = _small().eval()
    g = _graph(n=8)
    e64 = float(model(g)["energy"])
    model32 = copy.deepcopy(model).float()
    g32 = structure_to_graph({"pos": g.pos.float(), "atomic_numbers": g.atomic_numbers}, 4.0)
    assert abs(float(model32(g32)["energy"]) - e64) < 1e-4 * max(1.0, abs(e64))
    back = model32.double()
    fresh = EdgeFrames(3)
    assert torch.equal(back.frames.cg, fresh.cg) and torch.equal(back.frames.basis, fresh.basis)
    grid = back.blocks[0].ga.s2_act.grid
    assert torch.equal(grid.to_grid_mat, S2Grid(3, 2, 14).to_grid_mat)


def test_dropout_and_drop_path_only_in_training():
    model = _small(alpha_drop=0.3, proj_drop=0.2, drop_path_rate=0.3)
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
    path = str(tmp_path / "equiformer_v2_lammps.pt")
    export_to_lammps(model, 4.0, path)
    loaded = torch.jit.load(path)
    out = loaded(g.pos, g.edge_index, g.cell_shifts, g.atomic_numbers, g.cell[0])
    ref = ForceStressOutput(model)(g)
    assert abs(float(out["total_energy"]) - float(ref["energy"])) < 1e-8
    assert (out["forces"] - ref["forces"].detach()).abs().max() < 1e-8


@pytest.mark.parametrize("options", [{"activation": "gate", "use_grid_mlp": False, "norm_type": "rms_norm_sh"},
                                     {"activation": "s2", "use_grid_mlp": False, "norm_type": "layer_norm",
                                      "use_m_share_rad": True, "share_atom_edge_embedding": True},
                                     {"smooth_cutoff": True, "use_attn_renorm": False}])
def test_scriptable_variants(options):
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
    from xnn.common.data import AtomicDataset
    from xnn.common.train import Trainer

    rng = np.random.default_rng(0)
    structures = []
    for i in range(6):
        s = _structure(5, seed=i)
        s.update(energy=float(rng.normal()), forces=rng.normal(size=(5, 3)))
        structures.append(s)
    cfg = from_dict({"model": {"name": "equiformer_v2", "n_features": 8, "n_interactions": 1,
                               "n_rbf": 8, "cutoff": 4.0,
                               "extra": {"l_max": 2, "m_max": 1, "num_heads": 2,
                                         "attn_hidden_channels": 8, "attn_alpha_channels": 4,
                                         "attn_value_channels": 4, "ffn_hidden_channels": 8,
                                         "edge_channels": 8, "grid_resolution": 10}},
                     "optim": {"epochs": 2},
                     "data": {"batch_size": 3, "val_fraction": 0.0},
                     "output_dir": str(tmp_path)})
    logs = Trainer(cfg, AtomicDataset(structures, 4.0)).fit()["train"]
    assert {"energy_mse", "force_mse"} <= set(logs)


def test_reference_key_translation():
    cfg = from_dict({"model": {"name": "equiformer_v2", "sphere_channels": 8, "num_layers": 2,
                               "max_radius": 4.5, "lmax_list": [3], "mmax_list": [1],
                               "max_num_elements": 40, "n_rbf": 10,
                               "extra": {"num_heads": 2, "attn_hidden_channels": 8,
                                         "attn_alpha_channels": 4, "attn_value_channels": 4,
                                         "ffn_hidden_channels": 8, "edge_channels": 8,
                                         "grid_resolution": None}}})
    assert cfg.model.n_features == 8 and cfg.model.n_interactions == 2 and cfg.model.cutoff == 4.5
    m = build_model(cfg.model)
    assert m.lmax == 3 and m.mmax == 1 and m.sphere_embedding.num_embeddings == 40
    assert m(_graph())["energy"].shape == (1,)
    with pytest.raises(ValueError):
        build_model(from_dict({"model": {"name": "equiformer_v2", "lmax_list": [4, 6]}}).model)


# fidelity to the reference implementation (optional)

def _fidelity(dtype):
    if importlib.util.find_spec("fairchem") is None:
        pytest.skip("fairchem-core 1.x is not installed")
    pytest.importorskip("torch_geometric")
    import os
    import subprocess
    import sys
    script = os.path.join(os.path.dirname(__file__), "equiformer_fidelity.py")
    res = subprocess.run([sys.executable, script, "v2", dtype], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    return float(res.stdout.strip().splitlines()[-1])


def test_fidelity_vs_reference_float64():
    assert _fidelity("float64") < 1e-10


def test_fidelity_vs_reference_float32():
    assert _fidelity("float32") < 5e-4
