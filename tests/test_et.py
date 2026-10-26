"""ET tests: manuscript fidelity (ICLR 2022 equations and sizes), invariances and
equivariances, smoothness at the cutoff, TorchScript deployment, config handling and
(optionally) fidelity to the reference code.

The centerpiece is ``_reference``: a loop-based implementation of the paper's
equations (the embedding of eq 1 to 4, the attention of eq 5 to 9 and the update of
eq 10 and 11, with the self term of the reference neighbor list, then the gated
equivariant output network) that reuses only the model's *parameters*. Given the same
weights the model must reproduce it to float64 precision.
"""
import math
import os

import numpy as np
import pytest
import torch

from xnn.common.config import from_dict
from xnn.common.data import collate, structure_to_graph
from xnn.common.models import ForceStressOutput, available_models, build_model
from xnn.trfnn.models.et import EquivariantTransformer

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


def _graph(n=6, cutoff=5.0, seed=0, periodic=False, cell=6.0):
    s = _structure(n, seed)
    if periodic:
        s["cell"] = np.eye(3) * cell
        s["pbc"] = [True, True, True]
    return structure_to_graph(s, cutoff)


def _small(seed=0, **kw):
    torch.manual_seed(seed)
    kw.setdefault("n_features", 16)
    kw.setdefault("n_interactions", 2)
    kw.setdefault("n_rbf", 8)
    kw.setdefault("num_heads", 4)
    kw.setdefault("cutoff", 5.0)
    model = EquivariantTransformer(**kw)
    # random biases everywhere, so the zero-initialized ones hide nothing
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
    assert "et" in available_models()


def test_paper_sizes():
    """ET() is the MD17 / ANI-1 architecture of Table 4 (1.34 M parameters); the QM9
    architecture (F = 256, 8 layers, 64 radial functions) has 6.87 M."""
    m = EquivariantTransformer()
    assert m.embedding.weight.shape == (100, 128) and len(m.layers) == 6
    assert m.layers[0].num_heads == 8 and m.rbf.means.shape == (32,) and m.cutoff == 5.0
    assert m.layers[0].v_proj.weight.shape == (384, 128)
    n = sum(p.numel() for p in m.parameters()) - 100          # without atom_ref
    assert abs(n - 1.34e6) / 1.34e6 < 0.003
    big = EquivariantTransformer(n_features=256, n_interactions=8, n_rbf=64)
    n = sum(p.numel() for p in big.parameters()) - 100
    assert abs(n - 6.87e6) / 6.87e6 < 0.003


def test_invalid_options():
    with pytest.raises(ValueError):
        EquivariantTransformer(n_features=12, num_heads=8)
    with pytest.raises(ValueError):
        EquivariantTransformer(n_features=16, num_heads=4, distance_influence="edges")
    with pytest.raises(ValueError):
        EquivariantTransformer(n_features=16, num_heads=4, radial_basis="bessel")
    with pytest.raises(ValueError):
        EquivariantTransformer(n_features=16, num_heads=4, atomic_energies=[1.0])


def test_radial_basis_is_eq_1():
    """exp(-beta (exp(-d) - mu)^2) times the cosine cutoff, mu from exp(-r_cut) to 1
    and beta = (2 (1 - exp(-r_cut)) / K)^-2."""
    m = _small(n_rbf=5)
    r = torch.linspace(0.3, 4.9, 25)
    mu = torch.linspace(math.exp(-5.0), 1.0, 5)
    beta = (2.0 / 5 * (1.0 - math.exp(-5.0))) ** -2
    ref = 0.5 * (torch.cos(math.pi * r / 5.0) + 1)[:, None] * torch.exp(
        -beta * (torch.exp(-r)[:, None] - mu) ** 2)
    assert torch.allclose(m.rbf(r), ref, atol=1e-14)
    assert float(m.rbf(torch.tensor([5.0, 6.0])).abs().max()) == 0.0


# the paper's equations, atom by atom

def _reference(model, Z, pos):
    """Loop implementation of eq 1 to 11 and the output network."""
    F, H, N, rc = model.n_features, model.num_heads, len(Z), model.cutoff
    d = F // H
    Z = torch.as_tensor(Z)
    pos = torch.as_tensor(pos, dtype=torch.float64)

    def basis(dist):
        return model.rbf(torch.tensor([dist]))[0]

    def fcut(dist):
        return 0.5 * (math.cos(math.pi * dist / rc) + 1.0) if dist < rc else 0.0

    x = model.embedding.weight[Z].clone()
    ne = model.neighbor_embedding
    if ne is not None:                                            # eq 3 and 4
        nbh = torch.zeros(N, F)
        for i in range(N):
            for j in range(N):
                dist = float((pos[j] - pos[i]).norm())
                if i == j or dist >= rc:
                    continue
                nbh[i] += (_lin(ne.distance_proj, basis(dist)) * fcut(dist)
                           * ne.embedding.weight[Z[j]])
        x = torch.stack([_lin(ne.combine, torch.cat([x[i], nbh[i]])) for i in range(N)])
    vec = torch.zeros(N, 3, F)
    for layer in model.layers:
        xn = _layer_norm(layer.layernorm, x)
        q = _lin(layer.q_proj, xn).reshape(N, H, d)
        k = _lin(layer.k_proj, xn).reshape(N, H, d)
        v = _lin(layer.v_proj, xn).reshape(N, H, 3 * d)
        mixed = vec @ layer.vec_proj.weight.T                     # U_1 v, U_2 v, U_3 v
        vec1, vec2, vec3 = mixed[..., :F], mixed[..., F:2 * F], mixed[..., 2 * F:]
        vec_dot = (vec1 * vec2).sum(1)
        dx = torch.zeros(N, H, d)
        dvec = torch.zeros(N, 3, H, d)
        for i in range(N):
            for j in range(N):                                    # j = i included
                dist = 0.0 if i == j else float((pos[j] - pos[i]).norm())
                if i != j and dist >= rc:
                    continue
                e = basis(dist)
                c = 1.0 if i == j else fcut(dist)
                logits = q[i] * k[j]
                if layer.dk_proj is not None:
                    logits = logits * _silu(_lin(layer.dk_proj, e)).reshape(H, d)
                a = _silu(logits.sum(-1))                         # eq 7 and 8
                vj = v[j]
                if layer.vector_cutoff:
                    vj = vj * c
                else:
                    a = a * c
                if layer.dv_proj is not None:
                    vj = vj * _silu(_lin(layer.dv_proj, e)).reshape(H, 3 * d)
                s3, s1, s2 = vj[:, :d], vj[:, d:2 * d], vj[:, 2 * d:]   # eq 9
                dx[i] += s3 * a[:, None]
                direction = (pos[j] - pos[i]) / dist if i != j else torch.zeros(3)
                dvec[i] += (vec[j].reshape(3, H, d) * s1[None]
                            + s2[None] * direction[:, None, None])    # eq 11
        o = _lin(layer.o_proj, dx.reshape(N, F))
        o1, o2, o3 = o[:, :F], o[:, F:2 * F], o[:, 2 * F:]
        x = x + vec_dot * o2 + o3                                 # eq 10
        vec = vec + vec3 * o1[:, None] + dvec.reshape(N, 3, F)
    x = _layer_norm(model.out_norm, x)
    feats = x.clone()
    for blk in model.readout.blocks:                              # gated equivariant blocks
        w = blk.lin_v.weight
        v1, v2 = vec @ w[:blk.n_norm].T, vec @ w[blk.n_norm:].T
        h = _lin(blk.net[2], _silu(_lin(blk.net[0], torch.cat([x, v1.norm(dim=1)], -1))))
        x, gate = h[:, :blk.n_out_s], h[:, blk.n_out_s:]
        vec = gate[:, None] * v2
        if blk.scalar_activation:
            x = _silu(x)
    node_energy = (float(model.energy_scale) * x[:, 0] + float(model.energy_shift)
                   + model.atom_ref.weight[Z, 0])
    return {"node_energy": node_energy, "node_features": feats}


@pytest.mark.parametrize("vector_cutoff", [True, False])
@pytest.mark.parametrize("options", [{}, {"neighbor_embedding": False},
                                     {"distance_influence": "keys"},
                                     {"distance_influence": "values"},
                                     {"distance_influence": "none"}])
def test_forward_matches_paper_equations(vector_cutoff, options):
    model = _small(vector_cutoff=vector_cutoff, energy_shift=-0.3, energy_scale=1.7,
                   **options)
    model.set_atomic_energies(SPECIES, [-0.5, -1.0, -2.0])
    s = _structure(7, seed=3)
    out = model(structure_to_graph(s, 5.0))
    ref = _reference(model, s["atomic_numbers"], s["pos"])
    assert torch.allclose(out["node_energy"], ref["node_energy"], atol=1e-12)
    assert torch.allclose(out["node_features"], ref["node_features"], atol=1e-12)


def test_edge_direction_is_the_paper_convention():
    """r_ij = r_j - r_i (the negative of the xnn edge vector): the vector feature of an
    atom with one neighbor points along the bond toward that neighbor when the
    directional value s^2 is positive."""
    model = _small(n_interactions=1, neighbor_embedding=False)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        layer = model.layers[0]
        # the value projection is laid out per head as (s^3, s^1, s^2): s^2 = 1 in every head
        layer.v_proj.bias.view(4, 3, 4)[:, 2, :].fill_(1.0)
        layer.dv_proj.bias.fill_(10.0)              # D^V ~ 10 (times the cutoff)
    g = structure_to_graph({"pos": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                            "atomic_numbers": [1, 1]}, 5.0)
    v = model(g)["node_vectors"]
    assert v[0, 0, 0] > 0 and v[1, 0, 0] < 0 and abs(float(v[0, 1, 0])) < 1e-12
    assert torch.allclose(v[0], -v[1])


# physical invariances and equivariances

def _random_rotation(seed, proper=True):
    Q = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(seed)))[0]
    if (torch.det(Q) > 0) != proper:
        Q = Q @ torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
    return Q


@pytest.mark.parametrize("proper", [True, False])
def test_energy_invariance_forces_equivariance(proper):
    """E(3) invariant energies (the vector features are polar vectors, so reflections
    are covered too) and equivariant forces and vector features."""
    model = ForceStressOutput(_small()).eval()
    s = _structure(7, seed=1)
    R, t = _random_rotation(4, proper), torch.tensor([1.3, -0.7, 2.1])
    out = model(structure_to_graph(s, 5.0))
    pos2 = torch.as_tensor(s["pos"]) @ R.T + t
    out2 = model(structure_to_graph({"pos": pos2, "atomic_numbers": s["atomic_numbers"]}, 5.0))
    assert abs(float(out["energy"] - out2["energy"])) < 1e-10
    assert torch.allclose(out["forces"] @ R.T, out2["forces"], atol=1e-10)
    assert torch.allclose(out["node_features"], out2["node_features"], atol=1e-10)
    assert torch.allclose(torch.einsum("ab,nbf->naf", R, out["node_vectors"]),
                          out2["node_vectors"], atol=1e-10)


def test_permutation_invariance():
    model = _small().eval()
    s = _structure(7, seed=2)
    perm = np.random.default_rng(1).permutation(7)
    out = model(structure_to_graph(s, 5.0))
    out2 = model(structure_to_graph({"pos": s["pos"][perm],
                                     "atomic_numbers": np.array(s["atomic_numbers"])[perm]}, 5.0))
    assert abs(float(out["energy"] - out2["energy"])) < 1e-12
    assert torch.allclose(out["node_energy"][perm], out2["node_energy"], atol=1e-12)


def test_forces_are_minus_gradient():
    model = ForceStressOutput(_small()).eval()
    s = _structure(6, seed=4)
    out = model(structure_to_graph(s, 5.0))
    pos = np.asarray(s["pos"], dtype=float)
    h = 1e-5
    for (i, k) in ((0, 0), (3, 1), (5, 2)):
        p = pos.copy(); p[i, k] += h
        e_plus = float(model(structure_to_graph({"pos": p, "atomic_numbers": s["atomic_numbers"]}, 5.0))["energy"])
        p = pos.copy(); p[i, k] -= h
        e_minus = float(model(structure_to_graph({"pos": p, "atomic_numbers": s["atomic_numbers"]}, 5.0))["energy"])
        assert float(out["forces"][i, k]) == pytest.approx(-(e_plus - e_minus) / (2 * h), abs=1e-7)


def _crossing(model, d):
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.9, 0.0],
                    [math.sqrt(d ** 2 - 0.45 ** 2), 0.45, 0.0]])
    out = model(structure_to_graph({"pos": pos, "atomic_numbers": [8, 1, 1]}, 4.0))
    return float(out["energy"]), out["forces"].detach().clone()


def test_pes_smooth_across_cutoff_with_vector_cutoff():
    """With the TorchMD-Net 2.0 cutoff on the value pathway, the energy and the forces
    are continuous when a neighbor leaves the cutoff sphere."""
    model = ForceStressOutput(_small(cutoff=4.0, vector_cutoff=True)).eval()
    eps = 1e-7
    (e_in, f_in), (e_out, f_out) = _crossing(model, 4.0 - eps), _crossing(model, 4.0 + eps)
    assert abs(e_in - e_out) < 1e-10
    assert float((f_in - f_out).abs().max()) < 1e-5


def test_original_et_is_discontinuous_at_the_cutoff():
    """The ICLR 2022 model weights only the attention by the cutoff: the vector
    residual s^1 v_j + s^2 r_ij / |r_ij| keeps the bias of D^V at the cutoff, so the
    energy jumps there (the discontinuity removed in TorchMD-Net 2.0)."""
    model = ForceStressOutput(_small(cutoff=4.0, vector_cutoff=False)).eval()
    eps = 1e-7
    (e_in, _), (e_out, _) = _crossing(model, 4.0 - eps), _crossing(model, 4.0 + eps)
    assert abs(e_in - e_out) > 1e-6


def test_isolated_atoms_have_finite_gradients():
    model = ForceStressOutput(_small())
    g = structure_to_graph({"pos": [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
                            "atomic_numbers": [8, 1]}, 5.0)
    out = model(g)
    out["energy"].sum().backward()
    assert torch.isfinite(out["forces"]).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_size_extensivity_and_batching():
    model = _small().eval()
    graphs = [structure_to_graph(_structure(n, seed=n), 5.0) for n in (4, 6, 9)]
    singles = [model(g) for g in graphs]
    out = model(collate(graphs))
    for b, one in enumerate(singles):
        assert abs(float(out["energy"][b] - one["energy"][0])) < 1e-12
    s = _structure(5, seed=7)
    pair = {"pos": np.concatenate([s["pos"], s["pos"] + np.array([30.0, 0, 0])]),
            "atomic_numbers": list(s["atomic_numbers"]) * 2}
    e1 = float(model(structure_to_graph(s, 5.0))["energy"])
    e2 = float(model(structure_to_graph(pair, 5.0))["energy"])
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
    g32 = structure_to_graph({"pos": g.pos.float(), "atomic_numbers": g.atomic_numbers}, 5.0)
    out32 = model32(g32)
    scale = max(1.0, abs(float(out64["energy"])))
    assert abs(float(out64["energy"] - out32["energy"].double())) < 1e-4 * scale
    assert torch.allclose(out64["node_vectors"], out32["node_vectors"].double(), atol=1e-4)


def test_attention_weights():
    model = _small(n_interactions=3).eval()
    g = _graph(n=6)
    weights = model.attention_weights(g)
    assert len(weights) == 3 and all(w.shape == (g.num_edges, 4) for w in weights)
    # the weights carry the cutoff: an edge at the cutoff has none
    assert torch.isfinite(torch.stack(weights)).all()


# deployment

def test_scriptable_and_lammps_export(tmp_path):
    from xnn.common.deploy import export_to_lammps

    model = _small().eval()
    g = _graph(n=6, periodic=True)
    scripted = torch.jit.script(model)
    d = (scripted.node_energy(g.atomic_numbers, g.edge_index, g.edge_vectors())
         - model.node_energy(g.atomic_numbers, g.edge_index, g.edge_vectors()))
    assert d.abs().max() < 1e-12
    x, v = scripted.representation(g.atomic_numbers, g.edge_index, g.edge_vectors())
    assert x.shape == (6, 16) and v.shape == (6, 3, 16)

    path = str(tmp_path / "et_lammps.pt")
    export_to_lammps(model, 5.0, path)
    loaded = torch.jit.load(path)
    out = loaded(g.pos, g.edge_index, g.cell_shifts, g.atomic_numbers, g.cell[0])
    ref = ForceStressOutput(model)(g)
    assert abs(float(out["total_energy"]) - float(ref["energy"])) < 1e-5
    assert (out["forces"] - ref["forces"].detach()).abs().max() < 1e-5


def test_scriptable_without_neighbor_embedding():
    model = _small(neighbor_embedding=False, distance_influence="keys").eval()
    g = _graph(n=5)
    scripted = torch.jit.script(model)
    d = (scripted.node_energy(g.atomic_numbers, g.edge_index, g.edge_vectors())
         - model(g)["node_energy"])
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
    cfg = from_dict({"model": {"name": "et", "n_features": 8, "n_interactions": 1,
                               "n_rbf": 4, "cutoff": 5.0, "extra": {"num_heads": 2}},
                     "optim": {"epochs": 2},
                     "data": {"batch_size": 3, "val_fraction": 0.0},
                     "output_dir": str(tmp_path)})
    assert isinstance(cfg, Config)
    logs = Trainer(cfg, AtomicDataset(structures, 5.0)).fit()["train"]
    assert {"energy_mse", "force_mse"} <= set(logs)


# config

def test_from_config_extras():
    cfg = from_dict({"model": {
        "name": "et", "cutoff": 4.0, "n_features": 16, "n_interactions": 2, "n_rbf": 6,
        "extra": {"num_heads": 2, "radial_basis": "gaussian", "neighbor_embedding": False,
                  "distance_influence": "values", "vector_cutoff": False,
                  "energy_shift": -1.5, "energy_scale": 0.2, "species": SPECIES,
                  "atomic_energies": [-13.6, -1030.0, -2043.0]}}})
    m = build_model(cfg.model)
    assert m.rbf.centers.shape == (6,) and m.cutoff == 4.0 and m.neighbor_embedding is None
    assert m.layers[0].dk_proj is None and m.layers[0].dv_proj is not None
    assert m.layers[0].num_heads == 2 and not m.layers[0].vector_cutoff
    assert float(m.energy_shift) == -1.5 and float(m.energy_scale) == pytest.approx(0.2)
    assert float(m.atom_ref.weight[6, 0]) == pytest.approx(-1030.0)
    assert m(_graph())["energy"].shape == (1,)


def test_reference_key_translation():
    cfg = from_dict({"model": {"name": "et", "embedding_dimension": 48, "num_layers": 3,
                               "num_rbf": 20, "cutoff_upper": 4.5, "rbf_type": "expnorm",
                               "max_z": 50}})
    assert cfg.model.n_features == 48 and cfg.model.n_interactions == 3
    assert cfg.model.n_rbf == 20 and cfg.model.cutoff == 4.5 and cfg.data.cutoff == 4.5
    m = build_model(cfg.model)
    assert m.embedding.weight.shape == (50, 48)


# fidelity to the reference implementation (optional)

def _fidelity(model_name, dtype):
    path = os.environ.get("TORCHMDNET_UPSTREAM_PATH")
    if not path:
        pytest.skip("set TORCHMDNET_UPSTREAM_PATH to a checkout of torchmd/torchmd-net")
    import subprocess
    import sys
    script = os.path.join(os.path.dirname(__file__), "torchmdnet_fidelity.py")
    res = subprocess.run([sys.executable, script, path, model_name, dtype],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    return float(res.stdout.strip().splitlines()[-1])


def test_fidelity_vs_reference_float64():
    assert _fidelity("et", "float64") < 1e-12


def test_fidelity_vs_reference_float32():
    assert _fidelity("et", "float32") < 1e-4
