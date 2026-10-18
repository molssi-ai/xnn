"""TensorNet tests: manuscript fidelity (the tensor embedding, interaction and readout
of the NeurIPS 2023 paper as released in TorchMD-Net 2.0), the O(3) / SO(3) variants,
the total-charge coupling, smoothness, TorchScript deployment, config handling and
(optionally) parity with the reference code.

``_reference`` is a loop-based implementation of the equations (explicit 3 x 3
matrices per atom and feature, the self term of the reference neighbor list) that
reuses only the model's *parameters*; the model must reproduce it to float64 precision.
"""
import math
import os

import numpy as np
import pytest
import torch

from xnn.common.config import from_dict
from xnn.common.data import collate, structure_to_graph
from xnn.common.models import ForceStressOutput, available_models, build_model
from xnn.gnn.models.tensornet import (TensorNet, axial_vector, compose, decompose,
                                      frobenius, skew_tensor)

SPECIES = [1, 6, 8]


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _structure(n=6, seed=0, spread=3.0):
    rng = np.random.default_rng(seed)
    return {"pos": rng.uniform(0, spread, (n, 3)),
            "atomic_numbers": ([1, 6, 8] * n)[:n]}


def _graph(n=6, cutoff=4.5, seed=0, periodic=False, cell=6.0, total_charge=None):
    s = _structure(n, seed)
    if periodic:
        s["cell"] = np.eye(3) * cell
        s["pbc"] = [True, True, True]
    if total_charge is not None:
        s["total_charge"] = float(total_charge)
    return structure_to_graph(s, cutoff)


def _small(seed=0, **kw):
    torch.manual_seed(seed)
    kw.setdefault("n_features", 8)
    kw.setdefault("n_interactions", 2)
    kw.setdefault("n_rbf", 6)
    kw.setdefault("cutoff", 4.5)
    model = TensorNet(**kw)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("bias"):
                p.add_(0.1 * torch.randn_like(p))
    return model


def _silu(x):
    return x * torch.sigmoid(x)


def _lin(layer, x):
    y = x @ layer.weight.T
    return y if layer.bias is None else y + layer.bias


def _layer_norm(layer, x):
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(var + layer.eps) * layer.weight + layer.bias


# registry / defaults

def test_registered():
    assert "tensornet" in available_models()


def test_reference_defaults_and_size():
    """TensorNet() is the reference architecture (F = 128, 2 layers, 32 fixed radial
    functions, 4.5 A, O(3)); the F = 256, 3-layer QM9 model of TorchMD-Net 2.0 Table 1
    has about 4.0 M parameters."""
    m = TensorNet()
    assert m.tensor_embedding.emb.weight.shape == (128, 128) and len(m.layers) == 2
    assert m.cutoff == 4.5 and m.equivariance_group == "O(3)"
    assert not m.rbf.means.requires_grad and m.rbf.means.shape == (32,)
    n = sum(p.numel() for p in TensorNet(n_features=256, n_interactions=3).parameters()) - 128
    assert abs(n - 4.0e6) / 4.0e6 < 0.03


def test_invalid_options():
    with pytest.raises(ValueError):
        TensorNet(n_features=8, equivariance_group="SU(2)")
    with pytest.raises(ValueError):
        TensorNet(n_features=8, atomic_energies=[1.0])


# the tensor algebra

def test_irreducible_decomposition():
    """compose(decompose(X)) = X, the parts are orthogonal, and the skew part is the
    cross-product matrix of its axial vector."""
    X = torch.randn(5, 3, 3, 4)
    I, A, S = decompose(X)
    assert torch.allclose(compose(I, A, S), X, atol=1e-14)
    assert torch.allclose(A, -A.transpose(1, 2), atol=1e-14)
    assert torch.allclose(S, S.transpose(1, 2), atol=1e-14)
    assert torch.allclose(S.diagonal(dim1=1, dim2=2).sum(-1), torch.zeros(5, 4), atol=1e-14)
    assert torch.allclose((A * S).sum((1, 2)), torch.zeros(5, 4), atol=1e-14)
    assert torch.allclose(frobenius(X), 3 * I ** 2 + frobenius(A) + frobenius(S), atol=1e-13)
    v = torch.randn(5, 3, 4)
    assert torch.allclose(axial_vector(skew_tensor(v)), v, atol=1e-14)
    w = torch.randn(5, 3, 4)
    cross = torch.linalg.cross(v.movedim(1, -1), w.movedim(1, -1)).movedim(-1, 1)
    assert torch.allclose(torch.einsum("nijf,njf->nif", skew_tensor(v), w), cross, atol=1e-14)


# the paper's equations, atom by atom

def _reference(model, Z, pos, total_charge=0.0):
    """Loop implementation of the embedding, the interaction layers and the readout."""
    F, N, rc = model.n_features, len(Z), model.cutoff
    Z = torch.as_tensor(Z)
    pos = torch.as_tensor(pos, dtype=torch.float64)
    eye = torch.eye(3)
    emb = model.tensor_embedding

    def basis(dist):
        return model.rbf(torch.tensor([dist]))[0]

    def fcut(dist):
        return 0.5 * (math.cos(math.pi * dist / rc) + 1.0) if dist < rc else 0.0

    def pairs(i):
        for j in range(N):
            dist = 0.0 if i == j else float((pos[i] - pos[j]).norm())
            if i != j and dist >= rc:
                continue
            unit = (pos[i] - pos[j]) / dist if i != j else torch.zeros(3)
            yield j, dist, (1.0 if i == j else fcut(dist)), unit

    def split(X):
        A = 0.5 * (X - X.transpose(0, 1))
        I = X.diagonal(dim1=0, dim2=1).mean(-1)
        return I, A, X - A - I * eye[..., None]

    def join(I, A, S):
        return I * eye[..., None] + A + S

    def norm2(X):
        return (X * X).sum((0, 1))

    def matmul(a, b):
        return torch.einsum("ijf,jkf->ikf", a, b)

    def skew(v):
        return torch.stack([torch.stack([0 * v[0], -v[2], v[1]]),
                            torch.stack([v[2], 0 * v[0], -v[0]]),
                            torch.stack([-v[1], v[0], 0 * v[0]])])

    def axial(A):
        return 0.5 * torch.stack([A[2, 1] - A[1, 2], A[0, 2] - A[2, 0], A[1, 0] - A[0, 1]])

    tensors = []
    for i in range(N):
        I, Av, Sf = torch.zeros(F), torch.zeros(3, F), torch.zeros(3, 3, F)
        for j, dist, c, unit in pairs(i):
            zij = _lin(emb.emb2, torch.cat([emb.emb.weight[Z[i]], emb.emb.weight[Z[j]]]))
            w, e = c * zij, basis(dist)
            I = I + w * _lin(emb.distance_proj1, e)
            Av = Av + (w * _lin(emb.distance_proj2, e))[None] * unit[:, None]
            Sf = Sf + (w * _lin(emb.distance_proj3, e))[None, None] * torch.outer(unit, unit)[..., None]
        A = skew(Av)
        S = Sf - Sf.diagonal(dim1=0, dim2=1).mean(-1) * eye[..., None]
        n = _layer_norm(emb.init_norm, norm2(join(I, A, S)))
        n = _silu(_lin(emb.linears_scalar[1], _silu(_lin(emb.linears_scalar[0], n)))).reshape(3, F)
        I = (I @ emb.linears_tensor[0].weight.T) * n[0]
        A = (A @ emb.linears_tensor[1].weight.T) * n[1]
        S = (S @ emb.linears_tensor[2].weight.T) * n[2]
        tensors.append(join(I, A, S))
    scale = 1.0 + 0.1 * total_charge
    for layer in model.layers:
        W = layer.linears_tensor
        normed = [X / (norm2(X) + 1.0) for X in tensors]
        parts, Ys = [], []
        for X in normed:
            I, A, S = split(X)
            I, A, S = I @ W[0].weight.T, A @ W[1].weight.T, S @ W[2].weight.T
            parts.append((I, A, S))
            Ys.append(join(I, A, S))
        new = []
        for i in range(N):
            Im, Avm, Sm = torch.zeros(F), torch.zeros(3, F), torch.zeros(3, 3, F)
            for j, dist, c, _ in pairs(i):
                f = basis(dist)
                for lin in layer.linears_scalar:
                    f = _silu(_lin(lin, f))
                f = (f * c).reshape(3, F)
                Ij, Aj, Sj = parts[j]
                Im = Im + f[0] * Ij
                Avm = Avm + f[1][None] * axial(Aj)
                Sm = Sm + f[2][None, None] * Sj
            M = join(Im, skew(Avm), Sm)
            if layer.o3:
                C = scale * (matmul(Ys[i], M) + matmul(M, Ys[i]))
            else:
                C = 2.0 * matmul(Ys[i], M)
            I, A, S = split(C)
            n1 = norm2(C) + 1.0
            I, A, S = (I / n1) @ W[3].weight.T, (A / n1) @ W[4].weight.T, (S / n1) @ W[5].weight.T
            dX = join(I, A, S)
            new.append(normed[i] + dX + scale * matmul(dX, dX))
        tensors = new
    feats, energies = [], []
    for i in range(N):
        I, A, S = split(tensors[i])
        x = torch.cat([3 * I * I, norm2(A), norm2(S)])
        x = _silu(_lin(model.linear, _layer_norm(model.out_norm, x)))
        feats.append(x)
        e = _lin(model.readout[2], _silu(_lin(model.readout[0], x)))[0]
        energies.append(float(model.energy_scale) * e + float(model.energy_shift)
                        + model.atom_ref.weight[Z[i], 0])
    return {"node_energy": torch.stack(energies), "node_features": torch.stack(feats),
            "node_tensors": torch.stack(tensors)}


@pytest.mark.parametrize("group", ["O(3)", "SO(3)"])
@pytest.mark.parametrize("n_interactions", [0, 1, 2])
def test_forward_matches_paper_equations(group, n_interactions):
    model = _small(n_interactions=n_interactions, equivariance_group=group,
                   energy_shift=-0.3, energy_scale=1.7)
    model.set_atomic_energies(SPECIES, [-0.5, -1.0, -2.0])
    s = _structure(7, seed=3)
    out = model(structure_to_graph(s, 4.5))
    ref = _reference(model, s["atomic_numbers"], s["pos"])
    assert torch.allclose(out["node_energy"], ref["node_energy"], atol=1e-12)
    assert torch.allclose(out["node_features"], ref["node_features"], atol=1e-12)
    assert torch.allclose(out["node_tensors"], ref["node_tensors"], atol=1e-12)


def test_total_charge_enters_the_products():
    """The total charge scales the tensor products by 1 + 0.1 q: a charged structure
    has another energy, a zero charge the same as none, and the loop reference agrees."""
    model = _small().eval()
    s = _structure(6, seed=5)
    e0 = float(model(structure_to_graph(s, 4.5))["energy"])
    assert float(model(_graph(n=6, seed=5, total_charge=0.0))["energy"]) == pytest.approx(e0, abs=1e-12)
    g = structure_to_graph({**s, "total_charge": 1.0}, 4.5)
    out = model(g)
    assert abs(float(out["energy"]) - e0) > 1e-6
    ref = _reference(model, s["atomic_numbers"], s["pos"], total_charge=1.0)
    assert torch.allclose(out["node_energy"], ref["node_energy"], atol=1e-12)
    # a batch mixing charges gives every structure its own
    batch = collate([structure_to_graph(s, 4.5), g])
    assert torch.allclose(model(batch)["energy"], torch.stack([torch.tensor(e0), out["energy"][0]]),
                          atol=1e-12)


# physical invariances and equivariances

def _random_rotation(seed, proper=True):
    Q = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(seed)))[0]
    if (torch.det(Q) > 0) != proper:
        Q = Q @ torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
    return Q


@pytest.mark.parametrize("proper", [True, False])
def test_o3_invariance_and_tensor_equivariance(proper):
    """O(3): energies invariant under rotations and reflections, forces equivariant,
    and the tensors transform as X -> R X R^T up to the sign det(R) of the skew part
    (an axial vector)."""
    model = ForceStressOutput(_small()).eval()
    s = _structure(7, seed=1)
    R, t = _random_rotation(4, proper), torch.tensor([1.3, -0.7, 2.1])
    out = model(structure_to_graph(s, 4.5))
    pos2 = torch.as_tensor(s["pos"]) @ R.T + t
    out2 = model(structure_to_graph({"pos": pos2, "atomic_numbers": s["atomic_numbers"]}, 4.5))
    assert abs(float(out["energy"] - out2["energy"])) < 1e-10
    assert torch.allclose(out["forces"] @ R.T, out2["forces"], atol=1e-10)
    assert torch.allclose(out["node_features"], out2["node_features"], atol=1e-10)
    I, A, S = decompose(out["node_tensors"])
    expected = compose(I, torch.det(R) * A, S)
    rotated = torch.einsum("ab,nbcf,dc->nadf", R, expected, R)
    assert torch.allclose(rotated, out2["node_tensors"], atol=1e-10)


def test_so3_distinguishes_enantiomers():
    """SO(3): invariant under proper rotations but not under reflections, so a chiral
    structure and its mirror image get different energies."""
    model = ForceStressOutput(_small(equivariance_group="SO(3)")).eval()
    s = _structure(7, seed=1)
    out = model(structure_to_graph(s, 4.5))
    pos_rot = torch.as_tensor(s["pos"]) @ _random_rotation(4, True).T
    e_rot = float(model(structure_to_graph({"pos": pos_rot, "atomic_numbers": s["atomic_numbers"]}, 4.5))["energy"])
    assert abs(float(out["energy"]) - e_rot) < 1e-10
    pos_mir = torch.as_tensor(s["pos"]) @ _random_rotation(4, False).T
    e_mir = float(model(structure_to_graph({"pos": pos_mir, "atomic_numbers": s["atomic_numbers"]}, 4.5))["energy"])
    assert abs(float(out["energy"]) - e_mir) > 1e-6


def test_permutation_invariance():
    model = _small().eval()
    s = _structure(7, seed=2)
    perm = np.random.default_rng(1).permutation(7)
    out = model(structure_to_graph(s, 4.5))
    out2 = model(structure_to_graph({"pos": s["pos"][perm],
                                     "atomic_numbers": np.array(s["atomic_numbers"])[perm]}, 4.5))
    assert abs(float(out["energy"] - out2["energy"])) < 1e-12
    assert torch.allclose(out["node_energy"][perm], out2["node_energy"], atol=1e-12)


def test_forces_are_minus_gradient():
    model = ForceStressOutput(_small()).eval()
    s = _structure(6, seed=4)
    out = model(structure_to_graph(s, 4.5))
    pos = np.asarray(s["pos"], dtype=float)
    h = 1e-5
    for (i, k) in ((0, 0), (3, 1), (5, 2)):
        p = pos.copy(); p[i, k] += h
        e_plus = float(model(structure_to_graph({"pos": p, "atomic_numbers": s["atomic_numbers"]}, 4.5))["energy"])
        p = pos.copy(); p[i, k] -= h
        e_minus = float(model(structure_to_graph({"pos": p, "atomic_numbers": s["atomic_numbers"]}, 4.5))["energy"])
        assert float(out["forces"][i, k]) == pytest.approx(-(e_plus - e_minus) / (2 * h), abs=1e-7)


def test_pes_smooth_across_cutoff():
    """Every edge term carries the cosine cutoff, so the energy and the forces are
    continuous when a neighbor leaves the cutoff sphere."""
    model = ForceStressOutput(_small(cutoff=4.0)).eval()

    def evaluate(d):
        pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.9, 0.0],
                        [math.sqrt(d ** 2 - 0.45 ** 2), 0.45, 0.0]])
        out = model(structure_to_graph({"pos": pos, "atomic_numbers": [8, 1, 1]}, 4.0))
        return float(out["energy"]), out["forces"].detach().clone()

    eps = 1e-7
    (e_in, f_in), (e_out, f_out) = evaluate(4.0 - eps), evaluate(4.0 + eps)
    assert abs(e_in - e_out) < 1e-10
    assert float((f_in - f_out).abs().max()) < 1e-5


def test_isolated_atoms_have_finite_gradients():
    model = ForceStressOutput(_small())
    g = structure_to_graph({"pos": [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
                            "atomic_numbers": [8, 1]}, 4.5)
    out = model(g)
    out["energy"].sum().backward()
    assert torch.isfinite(out["forces"]).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_size_extensivity_and_batching():
    model = _small().eval()
    graphs = [structure_to_graph(_structure(n, seed=n), 4.5) for n in (4, 6, 9)]
    singles = [model(g) for g in graphs]
    out = model(collate(graphs))
    for b, one in enumerate(singles):
        assert abs(float(out["energy"][b] - one["energy"][0])) < 1e-12
    s = _structure(5, seed=7)
    pair = {"pos": np.concatenate([s["pos"], s["pos"] + np.array([30.0, 0, 0])]),
            "atomic_numbers": list(s["atomic_numbers"]) * 2}
    e1 = float(model(structure_to_graph(s, 4.5))["energy"])
    e2 = float(model(structure_to_graph(pair, 4.5))["energy"])
    assert e2 == pytest.approx(2 * e1, abs=1e-10)


def test_periodic_stress_and_images():
    model = ForceStressOutput(_small(), compute_stress=True).eval()
    g = _graph(n=6, periodic=True, cell=4.0)
    out = model(g)
    assert out["stress"].shape == (1, 3, 3) and torch.isfinite(out["stress"]).all()
    assert abs(float(out["energy"] - model(_graph(n=6, periodic=False))["energy"])) > 1e-6


def test_float32_forward_matches_float64():
    model = _small().eval()
    g = _graph(n=8)
    out64 = model(g)
    model32 = model.float()
    g32 = structure_to_graph({"pos": g.pos.float(), "atomic_numbers": g.atomic_numbers}, 4.5)
    out32 = model32(g32)
    scale = max(1.0, abs(float(out64["energy"])))
    assert abs(float(out64["energy"] - out32["energy"].double())) < 1e-4 * scale
    assert torch.allclose(out64["node_features"], out32["node_features"].double(), atol=1e-4)


# deployment

def test_scriptable_and_lammps_export(tmp_path):
    from xnn.common.deploy import export_to_lammps

    model = _small().eval()
    g = _graph(n=6, periodic=True)
    scripted = torch.jit.script(model)
    d = (scripted.node_energy(g.atomic_numbers, g.edge_index, g.edge_vectors())
         - model.node_energy(g.atomic_numbers, g.edge_index, g.edge_vectors()))
    assert d.abs().max() < 1e-12

    path = str(tmp_path / "tensornet_lammps.pt")
    export_to_lammps(model, 4.5, path)
    loaded = torch.jit.load(path)
    out = loaded(g.pos, g.edge_index, g.cell_shifts, g.atomic_numbers, g.cell[0])
    ref = ForceStressOutput(model)(g)
    assert abs(float(out["total_energy"]) - float(ref["energy"])) < 1e-5
    assert (out["forces"] - ref["forces"].detach()).abs().max() < 1e-5


def test_torchscript_export_carries_the_total_charge(tmp_path):
    from xnn.common.deploy import export_torchscript_potential

    model = _small().eval()
    s = _structure(6, seed=5)
    path = str(tmp_path / "tensornet_q.pt")
    export_torchscript_potential(model, model.cutoff, path, total_charge=1.0)
    loaded = torch.jit.load(path)
    out = loaded(torch.as_tensor(s["pos"]), torch.as_tensor(s["atomic_numbers"]))
    ref = ForceStressOutput(model)(structure_to_graph({**s, "total_charge": 1.0}, 4.5))
    assert abs(float(out["energy"]) - float(ref["energy"])) < 1e-8
    assert (out["forces"] - ref["forces"].detach()).abs().max() < 1e-7


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


# training

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
    cfg = from_dict({"model": {"name": "tensornet", "n_features": 8, "n_interactions": 1,
                               "n_rbf": 4, "cutoff": 4.5},
                     "optim": {"epochs": 2},
                     "data": {"batch_size": 3, "val_fraction": 0.0},
                     "output_dir": str(tmp_path)})
    assert isinstance(cfg, Config)
    logs = Trainer(cfg, AtomicDataset(structures, 4.5)).fit()["train"]
    assert {"energy_mse", "force_mse"} <= set(logs)


# config

def test_from_config_extras():
    cfg = from_dict({"model": {
        "name": "tensornet", "cutoff": 4.0, "n_features": 8, "n_interactions": 1, "n_rbf": 6,
        "extra": {"equivariance_group": "SO(3)", "trainable_rbf": True,
                  "energy_shift": -1.5, "energy_scale": 0.2, "species": SPECIES,
                  "atomic_energies": [-13.6, -1030.0, -2043.0]}}})
    m = build_model(cfg.model)
    assert m.cutoff == 4.0 and m.rbf.means.requires_grad and not m.layers[0].o3
    assert float(m.energy_shift) == -1.5 and float(m.energy_scale) == pytest.approx(0.2)
    assert float(m.atom_ref.weight[6, 0]) == pytest.approx(-1030.0)
    assert m(_graph(cutoff=4.0))["energy"].shape == (1,)


def test_reference_key_translation():
    cfg = from_dict({"model": {"name": "tensornet", "embedding_dimension": 16, "num_layers": 1,
                               "num_rbf": 10, "cutoff_upper": 4.0, "max_z": 50,
                               "equivariance_invariance_group": "SO(3)"}})
    assert cfg.model.n_features == 16 and cfg.model.n_interactions == 1
    assert cfg.model.n_rbf == 10 and cfg.model.cutoff == 4.0 and cfg.data.cutoff == 4.0
    m = build_model(cfg.model)
    assert m.tensor_embedding.emb.weight.shape == (50, 16) and m.equivariance_group == "SO(3)"


# parity with the reference implementation (optional)

def _parity(dtype):
    path = os.environ.get("TORCHMDNET_UPSTREAM_PATH")
    if not path:
        pytest.skip("set TORCHMDNET_UPSTREAM_PATH to a checkout of torchmd/torchmd-net")
    import subprocess
    import sys
    script = os.path.join(os.path.dirname(__file__), "torchmdnet_parity.py")
    res = subprocess.run([sys.executable, script, path, "tensornet", dtype],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    return float(res.stdout.strip().splitlines()[-1])


def test_parity_vs_reference_float64():
    assert _parity("float64") < 1e-12


def test_parity_vs_reference_float32():
    assert _parity("float32") < 1e-4
