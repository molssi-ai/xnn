"""TensorNet tests: manuscript fidelity (the tensor embedding, interaction and readout
of the NeurIPS 2023 paper as released in TorchMD-Net 2.0), the O(3) / SO(3) variants,
the total-charge coupling, smoothness, TorchScript deployment, config handling and
(optionally) fidelity to the reference code.

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
from xnn.gnn.models.tensornet import TensorNet, axial_vector, compose, decompose, frobenius, skew_tensor

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
            if i != j and dist >= rc or i == j and not model.self_pairs:
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
        x = _lin(model.linear, _layer_norm(model.out_norm, x))
        if model.activate_features:
            x = _silu(x)
            feats.append(x)
            e = _lin(model.readout[2], _silu(_lin(model.readout[0], x)))[0]
        else:
            feats.append(x)
            h, g = x, x
            last = len(model.readout.mlp.layers) - 1
            for k, (a, b) in enumerate(zip(model.readout.mlp.layers, model.readout.gate.layers)):
                h, g = _lin(a, h), _lin(b, g)
                if k < last:
                    h, g = _silu(h), _silu(g)
            e = (h * torch.sigmoid(g))[0]
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


@pytest.mark.parametrize("options", [
    {"radial_basis": "smooth_bessel", "n_rbf": 3},
    {"readout": "gated"},
    {"self_pairs": False},
    {"radial_basis": "smooth_bessel", "n_rbf": 3, "readout": "gated", "self_pairs": False,
     "readout_hidden": [8, 6]},
])
def test_options_match_paper_equations(options):
    """The smooth spherical Bessel basis, the gated readout and the pair lists without self
    pairs give the per-atom loop of the equations."""
    model = _small(**options).eval()
    with torch.no_grad():
        for p in model.readout.parameters():
            p.add_(0.1 * torch.randn_like(p))
    s = _structure(7, seed=3)
    out = model(structure_to_graph(s, 4.5))
    ref = _reference(model, s["atomic_numbers"], s["pos"])
    assert torch.allclose(out["node_energy"], ref["node_energy"], atol=1e-11)
    assert torch.allclose(out["node_features"], ref["node_features"], atol=1e-11)


def test_self_pairs_change_the_model():
    """Without self pairs an atom no longer sees itself: the energies differ."""
    g = _graph(n=6, seed=4)
    e_with = _small(self_pairs=True)(g)["energy"]
    e_without = _small(self_pairs=False)(g)["energy"]
    assert not torch.allclose(e_with, e_without)


def test_smooth_bessel_basis():
    """The basis is the paper's recursion, vanishes with its slope at the cutoff and keeps exact
    float64 coefficients through a float32 build."""
    from xnn.common.featurizers import SmoothBesselRBF
    rc, n_rbf = 4.5, 4
    basis = SmoothBesselRBF(n_rbf, rc)
    r = torch.linspace(0.0, rc, 41, dtype=torch.float64, requires_grad=True)

    def j0(x):
        return torch.where(x == 0, torch.ones_like(x), torch.sin(x) / torch.where(x == 0, 1.0, x))

    f = [(-1) ** n * math.sqrt(2) * math.pi / rc ** 1.5 * (n + 1) * (n + 2)
         / math.sqrt((n + 1) ** 2 + (n + 2) ** 2)
         * (j0((n + 1) * math.pi * r / rc) + j0((n + 2) * math.pi * r / rc)) for n in range(n_rbf)]
    g, d = [f[0]], 1.0
    for n in range(1, n_rbf):
        e = n ** 2 * (n + 2) ** 2 / (4 * (n + 1) ** 4 + 1)
        d_new = 1 - e / d
        g.append((f[n] + math.sqrt(e / d) * g[-1]) / math.sqrt(d_new))
        d = d_new
    values = basis(r)
    assert torch.allclose(values, torch.stack(g, -1), rtol=0, atol=1e-13)
    slope = torch.autograd.grad(values[-1].sum(), r)[0][-1]
    assert values[-1].abs().max() < 1e-12 and abs(float(slope)) < 1e-10
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        model = _small(radial_basis="smooth_bessel", n_rbf=3)
    finally:
        torch.set_default_dtype(old)
    model = model.double()
    exact = SmoothBesselRBF(3, model.cutoff).coefficients
    assert torch.equal(model.rbf.coefficients, exact)


def test_default_readout_keeps_its_parameter_names():
    """The default readout is still the indexed dense stack (state dicts load unchanged)."""
    keys = {k for k in _small().state_dict() if k.startswith("readout.")}
    assert keys == {"readout.0.weight", "readout.0.bias", "readout.2.weight", "readout.2.bias"}
    with pytest.raises(ValueError):
        TensorNet(n_features=8, readout="set2set")


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

@pytest.mark.parametrize("options", [{}, {"radial_basis": "smooth_bessel", "n_rbf": 3, "readout": "gated",
                                         "self_pairs": False}])
def test_scriptable_and_lammps_export(tmp_path, options):
    from xnn.common.deploy import export_to_lammps

    model = _small(**options).eval()
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


# fidelity to the reference implementation (optional)

def _fidelity(dtype):
    path = os.environ.get("TORCHMDNET_UPSTREAM_PATH")
    if not path:
        pytest.skip("set TORCHMDNET_UPSTREAM_PATH to a checkout of torchmd/torchmd-net")
    import subprocess
    import sys
    script = os.path.join(os.path.dirname(__file__), "tensornet_fidelity.py")
    res = subprocess.run([sys.executable, script, path, dtype],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    return float(res.stdout.strip().splitlines()[-1])


def test_fidelity_vs_reference_float64():
    assert _fidelity("float64") < 1e-12


def test_fidelity_vs_reference_float32():
    assert _fidelity("float32") < 1e-4


# adapters on the pair embedding

def test_lora_adapter_reaches_the_pair_embedding():
    """An adapted model gives the outputs of the model with the adapters merged."""
    from xnn.common.finetune import inject_lora, merge_lora
    model = _small(n_features=16)
    inject_lora(model, rank=2)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_out" in name:
                p.normal_()
    graph = _graph(n=7, seed=3, total_charge=1.0)
    adapted = ForceStressOutput(model)(graph)
    merged = ForceStressOutput(merge_lora(model))(_graph(n=7, seed=3, total_charge=1.0))
    for key in ("energy", "forces"):
        err = (adapted[key] - merged[key]).abs().max() / merged[key].abs().max()
        assert err < 1e-12, (key, float(err))


# fast path (fused Triton kernels)

def test_fast_path_eligibility(monkeypatch):
    """The fused layers need a power-of-two width; adapters stay outside the kernels."""
    import xnn.gnn.fast.tensornet as fast
    from xnn.common.finetune import inject_lora
    monkeypatch.setattr(fast, "supported", lambda device, dtype: True)
    cuda = torch.device("cuda")
    m = _small(n_features=16)
    assert m.tensor_embedding.fast_supported(cuda, torch.float32)
    assert all(layer.fast_supported(cuda, torch.float32) for layer in m.layers)
    assert not _small(n_features=24).layers[0].fast_supported(cuda, torch.float32)
    inject_lora(m, rank=2)
    assert m.layers[0].fast_supported(cuda, torch.float32)


def test_fast_support_needs_cuda():
    from xnn.gnn.fast.tensornet import supported
    assert not supported(torch.device("cpu"), torch.float64)


def _fused_inputs(device, seed):
    """Random inputs of both fused functions: ten rows per atom, pairs with self pairs."""
    gen = torch.Generator().manual_seed(seed)
    n, n_pairs, F = 5, 14, 16

    def rand(*shape):
        return torch.randn(*shape, dtype=torch.float64, generator=gen).to(device).requires_grad_(True)

    src = torch.cat([torch.randint(0, n, (n_pairs - n,), generator=gen), torch.arange(n)]).to(device)
    dst = torch.cat([torch.randint(0, n, (n_pairs - n,), generator=gen), torch.arange(n)]).to(device)
    u = torch.nn.functional.normalize(torch.randn(n_pairs, 3, dtype=torch.float64, generator=gen), dim=1)
    u[n_pairs - n:] = 0.0                                   # self pairs carry no direction
    return dict(rows=rand(n, 10, F), a=rand(n_pairs, 3 * F), cut=rand(n_pairs),
                left=rand(n, F), right=rand(n, F), d=[rand(n_pairs, F) for _ in range(3)],
                u=u.to(device).requires_grad_(True), src=src, dst=dst)


def _check_fused_gradients(kernels, device, fast_mode=False):
    t = _fused_inputs(device, seed=0)
    src, dst = t["src"], t["dst"]

    def messages(rows, a, cut):
        return kernels.MessageFunction.apply(rows, a, cut, src, dst)

    def embedding(left, right, cut, d1, d2, d3, u):
        return kernels.EmbeddingFunction.apply(left, right, cut, d1, d2, d3, u, src, dst)

    # the embedding's neighbor term adds atomically, in a run-dependent order
    assert torch.autograd.gradcheck(messages, (t["rows"], t["a"], t["cut"]), nondet_tol=1e-12,
                                    fast_mode=fast_mode)
    assert torch.autograd.gradcheck(embedding, (t["left"], t["right"], t["cut"], *t["d"], t["u"]),
                                    nondet_tol=1e-12, fast_mode=fast_mode)
    # second derivatives (force training): the backward functions, differentiated once more
    assert torch.autograd.gradgradcheck(messages, (t["rows"], t["a"], t["cut"]), nondet_tol=1e-12,
                                        fast_mode=fast_mode)
    assert torch.autograd.gradgradcheck(embedding, (t["left"], t["right"], t["cut"], *t["d"], t["u"]),
                                        nondet_tol=1e-12, fast_mode=fast_mode)
    # and the values are the reference sums
    with torch.no_grad():
        F = t["left"].shape[1]
        a = (t["a"] * t["cut"][:, None]).reshape(-1, 3, F)
        part = torch.tensor([0, 1, 1, 1, 2, 2, 2, 2, 2, 2], device=device)
        expect = torch.zeros_like(t["rows"]).index_add_(0, dst, a[:, part] * t["rows"][src])
        assert torch.allclose(messages(t["rows"], t["a"], t["cut"]), expect, rtol=0, atol=1e-13)
        w = t["cut"][:, None] * (t["left"][dst] + t["right"][src])
        u = t["u"]
        uu = torch.stack([u[:, 0] * u[:, 0], u[:, 0] * u[:, 1], u[:, 0] * u[:, 2],
                          u[:, 1] * u[:, 1], u[:, 1] * u[:, 2], u[:, 2] * u[:, 2]], dim=1)
        per_pair = torch.cat([(w * t["d"][0])[:, None], (w * t["d"][1])[:, None] * u[:, :, None],
                              (w * t["d"][2])[:, None] * uu[:, :, None]], dim=1)
        expect = torch.zeros_like(t["rows"]).index_add_(0, dst, per_pair)
        got = embedding(t["left"], t["right"], t["cut"], *t["d"], t["u"])
        assert torch.allclose(got, expect, rtol=0, atol=1e-13)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_fast_kernels_gradients():
    """Values and every first and second derivative of the fused functions (finite differences)."""
    pytest.importorskip("triton")
    from xnn.gnn.fast import tensornet_kernels
    _check_fused_gradients(tensornet_kernels, "cuda")


def _fast_batch(dtype, device="cuda"):
    graphs = [_graph(n=14, seed=1, periodic=True, total_charge=1.0),
              _graph(n=9, seed=2, total_charge=-1.0)]
    batch = collate(graphs).to(device)
    batch.pos = batch.pos.to(dtype)
    batch.cell = batch.cell.to(dtype)
    return batch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("group", ["O(3)", "SO(3)"])
@pytest.mark.parametrize("self_pairs", [True, False])
def test_fast_path_parity(dtype, group, self_pairs):
    """The fused layers give the reference energies, forces and stress."""
    pytest.importorskip("triton")
    model = _small(n_features=32, n_rbf=8, equivariance_group=group,
                   self_pairs=self_pairs).to("cuda", dtype).eval()
    wrapped = ForceStressOutput(model, compute_stress=True).eval()
    keys = ("energy", "forces", "stress")
    out = {}
    for use_fast in (False, True):
        model.set_use_fast(use_fast)
        res = wrapped(_fast_batch(dtype))
        assert model.tensor_embedding.fast_active == use_fast
        assert all(layer.fast_active == use_fast for layer in model.layers)
        out[use_fast] = {k: res[k].detach().double() for k in keys}
    tol = 1e-12 if dtype == torch.float64 else 1e-5
    for key, ref in out[False].items():
        err = (out[True][key] - ref).abs().max() / ref.abs().max()
        assert err < tol, (key, float(err))


def _training_gradients(model, batch, use_fast):
    """The weight gradients of an energy and force loss (forces with create_graph)."""
    model.set_use_fast(use_fast).train()
    model.zero_grad(set_to_none=True)
    out = ForceStressOutput(model, compute_stress=True).train()(batch)
    assert model.tensor_embedding.fast_active == use_fast
    assert all(layer.fast_active == use_fast for layer in model.layers)
    (out["energy"].pow(2).sum() + out["forces"].pow(2).sum() + out["stress"].pow(2).sum()).backward()
    return {name: p.grad.detach().double().clone() for name, p in model.named_parameters()
            if p.grad is not None}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("group", ["O(3)", "SO(3)"])
def test_fast_path_training(dtype, group):
    """Force training through the fused layers gives the reference's weight gradients."""
    pytest.importorskip("triton")
    model = _small(n_features=32, n_rbf=8, equivariance_group=group).to("cuda", dtype)
    ref = _training_gradients(model, _fast_batch(dtype), False)
    fast = _training_gradients(model, _fast_batch(dtype), True)
    assert ref.keys() == fast.keys()
    tol = 1e-11 if dtype == torch.float64 else 1e-4
    for name, g in ref.items():
        err = (fast[name] - g).abs().max() / g.abs().max().clamp_min(1e-30)
        assert err < tol, (name, float(err))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_auto_crossover_per_mode(monkeypatch):
    """"auto" applies the evaluation crossover in evaluation mode and the training one in
    training mode."""
    pytest.importorskip("triton")
    import xnn.gnn.fast.tensornet as fast
    from xnn.common.models.fast import AutoPolicy
    never = AutoPolicy(default_min_edges=10 ** 9, default_min_edges_float64=10 ** 9)
    always = AutoPolicy(default_min_edges=0, default_min_edges_float64=0)
    model = _small(n_features=16).to("cuda", torch.float64).set_use_fast("auto")
    names = ("AUTO_POLICY", "AUTO_POLICY_TRAINING")
    for training in (False, True):
        for policy in names:
            with monkeypatch.context() as patch:
                for name in names:
                    patch.setattr(fast, name, never if name == policy else always)
                ForceStressOutput(model.train(training)).train(training)(_fast_batch(torch.float64))
            expect = policy != ("AUTO_POLICY_TRAINING" if training else "AUTO_POLICY")
            assert model.tensor_embedding.fast_active == expect, (training, policy)
            assert all(layer.fast_active == expect for layer in model.layers)


def _needs_interpreter():
    """Skip unless Triton's interpreter is on (no CUDA device) and recent enough."""
    if os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("runs where Triton's interpreter is on (no CUDA device)")
    triton = pytest.importorskip("triton")
    if tuple(int(v) for v in triton.__version__.split(".")[:2]) < (3, 2):
        pytest.skip("Triton's interpreter before 3.2 mis-evaluates these kernels")


def test_fast_kernels_gradients_interpreted():
    """The fused functions and their first and second derivatives, run by the Triton interpreter."""
    _needs_interpreter()
    from xnn.gnn.fast import tensornet_kernels
    # random projections of the Jacobians: the interpreter runs every kernel call in Python
    _check_fused_gradients(tensornet_kernels, "cpu", fast_mode=True)


def test_fast_messages_strided_filters_interpreted():
    """A filter output with strided rows (a transposed view) gives the reference messages."""
    _needs_interpreter()
    from xnn.gnn.fast import tensornet_kernels
    t = _fused_inputs("cpu", seed=1)
    src, dst = t["src"], t["dst"]
    rows, cut = t["rows"].detach(), t["cut"].detach()
    F = rows.shape[2]
    a = t["a"].detach().t().contiguous().t()                 # (P, 3F) with stride (1, P)
    assert a.stride(1) != 1
    part = torch.tensor([0, 1, 1, 1, 2, 2, 2, 2, 2, 2])
    scaled = (a * cut[:, None]).reshape(-1, 3, F)
    expect = torch.zeros_like(rows).index_add_(0, dst, scaled[:, part] * rows[src])
    got = tensornet_kernels.messages(rows, a, cut, src, dst)
    assert torch.allclose(got, expect, rtol=0, atol=1e-13)


@pytest.mark.parametrize("group", ["O(3)", "SO(3)"])
def test_fast_interaction_product_stage(group):
    """The fast path's product stage against the reference's tensor algebra: the parts of
    ``(1 + q/10)(YM + MY)`` (O(3)) or ``2YM`` (SO(3)), divided by ``|C|^2 + 1``."""
    from xnn.gnn.fast.tensornet import stage
    from xnn.gnn.models.tensornet import (
        axial_vector,
        compose,
        decompose,
        frobenius,
        skew_tensor,
        tensor_matmul,
    )
    gen = torch.Generator().manual_seed(2)
    n, F = 4, 3

    def rand(*shape):
        return torch.randn(*shape, dtype=torch.float64, generator=gen)

    def symmetric(entries):
        """``(N, 6, F)`` entries xx, xy, xz, yy, yz, zz as ``(N, 3, 3, F)`` tensors."""
        return entries[:, torch.tensor([0, 1, 2, 1, 3, 4, 2, 4, 5])].reshape(n, 3, 3, F)

    I, v, S, m, q = rand(n, F), rand(n, 3, F), rand(n, 6, F), rand(n, 10, F), rand(n)
    Y = compose(I, skew_tensor(v), symmetric(S))
    M = compose(m[:, 0], skew_tensor(m[:, 1:4]), symmetric(m[:, 4:]))
    if group == "O(3)":
        C = (1.0 + 0.1 * q)[:, None, None, None] * (tensor_matmul(Y, M) + tensor_matmul(M, Y))
    else:
        C = 2.0 * tensor_matmul(Y, M)
    scalar, skew, sym = decompose(C)
    norm = frobenius(C) + 1.0
    entries = sym.reshape(n, 9, F)[:, torch.tensor([0, 1, 2, 4, 5, 8])]
    expect = (scalar / norm, axial_vector(skew) / norm[:, None], entries / norm[:, None])
    name = "interaction_product_o3" if group == "O(3)" else "interaction_product_so3"
    got = stage(name, I)(I, v, S, m, q)
    for x, y in zip(got, expect):
        assert torch.allclose(x, y, rtol=0, atol=1e-14)


def test_fast_path_parity_interpreted(monkeypatch):
    """The fused layers under Triton's interpreter give the reference energies and forces."""
    _needs_interpreter()
    import xnn.gnn.fast.tensornet as fast
    monkeypatch.setattr(fast, "supported", lambda device, dtype: True)
    model = _small(n_features=16).eval()
    wrapped = ForceStressOutput(model).eval()
    out = {}
    for use_fast in (False, True):
        model.set_use_fast(use_fast)
        res = wrapped(_fast_batch(torch.float64, device="cpu"))
        out[use_fast] = {k: res[k].detach() for k in ("energy", "forces")}
    for key, ref in out[False].items():
        assert (out[True][key] - ref).abs().max() / ref.abs().max() < 1e-12, key


# recomputing the filter networks in the backward pass

def test_filter_networks_recompute(monkeypatch):
    """Recomputed filter networks give the same energies and forces; "auto" takes them in
    evaluation mode from the threshold on, "off" never."""
    import torch.utils.checkpoint

    import xnn.gnn.models.tensornet as tn
    from xnn.gnn.models.blocks import set_recompute_radial
    real, calls = torch.utils.checkpoint.checkpoint, []

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", counting)
    model = _small(n_features=16).eval()

    def run():
        return ForceStressOutput(model)(_graph(n=9, seed=4, total_charge=1.0))

    kept = run()
    assert not calls                                   # below RECOMPUTE_MIN_PAIRS
    set_recompute_radial(model, True)
    assert all(layer.recompute == "on" for layer in model.layers)
    recomputed = run()
    assert len(calls) == len(model.layers)
    for key in ("energy", "forces"):
        assert torch.allclose(recomputed[key], kept[key], rtol=0, atol=1e-13), key
    calls.clear()
    set_recompute_radial(model, "auto")
    monkeypatch.setattr(tn, "RECOMPUTE_MIN_PAIRS", 1)
    run()
    assert len(calls) == len(model.layers)
    calls.clear()
    model.train()
    run()
    assert not calls                                   # "auto" keeps them when training
    set_recompute_radial(model.eval(), False)
    run()
    assert not calls


def _matgl_directory(path, units=8, n_layers=2, elements=("H", "C", "O", "Si"), seed=0):
    """A MatGL TensorNet checkpoint directory with random weights."""
    import json
    gen = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    F, n_rbf = units, 3
    sd = {"data_mean": torch.tensor(0.0), "data_std": torch.tensor(2.5),
          "element_refs.property_offset": rand(len(elements)).double(),
          "model.bond_expansion.rbf.cutoff": torch.tensor(4.0),
          "model.tensor_embedding.emb.weight": rand(len(elements), F),
          "model.tensor_embedding.emb2.weight": rand(F, 2 * F), "model.tensor_embedding.emb2.bias": rand(F),
          "model.tensor_embedding.init_norm.weight": rand(F), "model.tensor_embedding.init_norm.bias": rand(F),
          "model.out_norm.weight": rand(3 * F), "model.out_norm.bias": rand(3 * F),
          "model.linear.weight": rand(F, 3 * F), "model.linear.bias": rand(F)}
    for k in (1, 2, 3):
        sd[f"model.tensor_embedding.distance_proj{k}.weight"] = rand(F, n_rbf)
        sd[f"model.tensor_embedding.distance_proj{k}.bias"] = rand(F)
    for k in range(3):
        sd[f"model.tensor_embedding.linears_tensor.{k}.weight"] = rand(F, F)
    for k, (a, b) in enumerate([(F, 2 * F), (2 * F, 3 * F)]):
        sd[f"model.tensor_embedding.linears_scalar.{k}.weight"] = rand(b, a)
        sd[f"model.tensor_embedding.linears_scalar.{k}.bias"] = rand(b)
    for layer in range(n_layers):
        for k, (a, b) in enumerate([(n_rbf, F), (F, 2 * F), (2 * F, 3 * F)]):
            sd[f"model.layers.{layer}.linears_scalar.{k}.weight"] = rand(b, a)
            sd[f"model.layers.{layer}.linears_scalar.{k}.bias"] = rand(b)
        for k in range(6):
            sd[f"model.layers.{layer}.linears_tensor.{k}.weight"] = rand(F, F)
    for branch in ("layers", "gates"):
        for k, (a, b) in enumerate([(F, F), (F, F), (F, F), (F, 1)]):
            sd[f"model.final_layer.gated.{branch}.{2 * k}.weight"] = rand(b, a)
            sd[f"model.final_layer.gated.{branch}.{2 * k}.bias"] = rand(b)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(sd, path / "state.pt")
    args = {"element_types": list(elements), "units": F, "nblocks": n_layers, "num_rbf": 32, "max_n": n_rbf,
            "max_l": 3, "rbf_type": "SphericalBessel", "use_smooth": True, "activation_type": "swish",
            "cutoff": 4.0, "equivariance_invariance_group": "O(3)", "is_intensive": False,
            "include_state": False, "ntargets": 1}
    meta = {"@class": "Potential", "kwargs": {"model": {"@class": "TensorNet", "init_args": args},
                                              "calc_repuls": False}}
    (path / "model.json").write_text(json.dumps(meta))
    return sd


def test_matgl_checkpoint_converts(tmp_path):
    """A MatGL TensorNet directory converts weight for weight: embedding rows by Z, the gated
    network's dense layers, the energy scale and the element references."""
    from xnn.common.models.hub.formats import detect_format
    from xnn.gnn.models import tensornet_foundation as tf
    from xnn.gnn.models.matgl_format import is_matgl_dir
    sd = _matgl_directory(tmp_path / "ckpt")
    assert is_matgl_dir(tmp_path / "ckpt") and detect_format(tmp_path / "ckpt") == "matgl"
    assert not is_matgl_dir(tmp_path)
    model, cfg = tf.convert_directory(tmp_path / "ckpt")
    assert model.self_pairs is False and not model.activate_features
    assert cfg.model.extra["radial_basis"] == "smooth_bessel" and model.cutoff == 4.0
    emb = model.tensor_embedding.emb.weight
    for row, z in enumerate((1, 6, 8, 14)):
        assert torch.equal(emb[z], sd["model.tensor_embedding.emb.weight"][row].to(emb.dtype))
        assert float(model.atom_ref.weight[z, 0]) == pytest.approx(float(sd["element_refs.property_offset"][row]))
    assert torch.equal(model.readout.gate.layers[2].weight,
                       sd["model.final_layer.gated.gates.4.weight"].to(emb.dtype))
    assert torch.equal(model.layers[1].linears_tensor[5].weight,
                       sd["model.layers.1.linears_tensor.5.weight"].to(emb.dtype))
    assert float(model.energy_scale) == pytest.approx(2.5)
    # the 3F filter outputs: part k of feature f is MatGL's row 3 f + k, xnn's row k F + f
    theirs = sd["model.layers.0.linears_scalar.2.weight"]
    ours = model.layers[0].linears_scalar[2].weight
    assert torch.equal(ours[2 * 8 + 5], theirs[3 * 5 + 2].to(ours.dtype))
    model = model.double()
    rebuilt = build_model(cfg.model).double()
    rebuilt.load_state_dict(model.state_dict())
    g = _graph(n=8, cutoff=4.0, seed=5, periodic=True)
    g.atomic_numbers = torch.tensor([1, 6, 8, 14, 1, 6, 8, 14])
    assert torch.allclose(rebuilt(g)["energy"], model(g)["energy"])


def test_matgl_checkpoint_refusals(tmp_path):
    """Architectures outside the converted ones are refused."""
    import json

    from xnn.gnn.models import tensornet_foundation as tf
    _matgl_directory(tmp_path)
    meta = json.loads((tmp_path / "model.json").read_text())
    meta["kwargs"]["model"]["init_args"]["use_smooth"] = False
    (tmp_path / "model.json").write_text(json.dumps(meta))
    with pytest.raises(NotImplementedError, match="radial basis"):
        tf.convert_directory(tmp_path)
    with pytest.raises(KeyError):
        tf.from_foundation("tensornet-unknown")


@pytest.mark.parametrize("width, gamma", [(0.7, 0.7), (None, 2 / 5.0)])
def test_matgl_gaussian_basis_converts(tmp_path, width, gamma):
    """MatGL's Gaussians (centers on [0, cutoff + 1], its width the exponent; no width: the inverse
    center spacing) convert to the Gaussian basis with rbf_max and gamma."""
    import json

    from xnn.gnn.models import tensornet_foundation as tf
    _matgl_directory(tmp_path)
    meta = json.loads((tmp_path / "model.json").read_text())
    meta["kwargs"]["model"]["init_args"].update(rbf_type="Gaussian", num_rbf=3, width=width)
    (tmp_path / "model.json").write_text(json.dumps(meta))
    model, cfg = tf.convert_directory(tmp_path, torch.float64)
    assert cfg.model.extra["radial_basis"] == "gaussian" and cfg.model.extra["rbf_max"] == 5.0
    assert model.rbf.gamma == pytest.approx(gamma)
    assert torch.equal(model.rbf.centers, torch.tensor([0.0, 2.5, 5.0]))


def test_gaussian_basis_options():
    """rbf_max moves the last Gaussian and gamma sets their exponent; other bases ignore rbf_max."""
    m = TensorNet(n_features=8, n_rbf=5, cutoff=4.0, radial_basis="gaussian", rbf_max=6.0, gamma=0.5)
    assert float(m.rbf.centers[-1]) == 6.0 and m.rbf.gamma == 0.5
    assert float(TensorNet(n_features=8, n_rbf=5, cutoff=4.0, radial_basis="gaussian").rbf.centers[-1]) == 4.0
    with pytest.warns(UserWarning, match="rbf_max"):
        TensorNet(n_features=8, radial_basis="expnorm", rbf_max=6.0)


def test_matgl_directory_detection(tmp_path):
    """Either file of a MatGL TensorNet directory is detected; an unreadable or foreign model.json is not."""
    from xnn.gnn.models.matgl_format import is_matgl_dir
    _matgl_directory(tmp_path)
    assert is_matgl_dir(tmp_path / "model.json") and is_matgl_dir(tmp_path / "state.pt")
    (tmp_path / "model.json").write_text("{not json")
    assert not is_matgl_dir(tmp_path)
    (tmp_path / "model.json").write_text('{"kwargs": {}}')
    assert not is_matgl_dir(tmp_path)


def test_matgl_checkpoint_refusals_name_every_unconverted_piece(tmp_path):
    """Another model class, every unconverted option at once, a per-structure energy offset (also in a
    wrapped state dict) and a missing weight are refused."""
    import json

    from xnn.gnn.models import tensornet_foundation as tf
    sd = _matgl_directory(tmp_path)
    text = (tmp_path / "model.json").read_text()

    meta = json.loads(text)
    meta["kwargs"]["model"]["@class"] = "M3GNet"
    (tmp_path / "model.json").write_text(json.dumps(meta))
    with pytest.raises(NotImplementedError, match="M3GNet, not a TensorNet"):
        tf.convert_directory(tmp_path / "model.json")

    meta = json.loads(text)
    meta["kwargs"]["model"]["init_args"].update(is_intensive=True, include_state=True, ntargets=2,
                                                activation_type="tanh", equivariance_invariance_group="SO(3)")
    meta["kwargs"]["calc_repuls"] = True
    (tmp_path / "model.json").write_text(json.dumps(meta))
    with pytest.raises(NotImplementedError) as err:
        tf.convert_directory(tmp_path)
    for piece in ("intensive target", "state features", "several targets", "repulsion term", "activation tanh",
                  "SO(3) group"):
        assert piece in str(err.value)

    (tmp_path / "model.json").write_text(text)
    torch.save({"state_dict": {**sd, "data_mean": torch.tensor(0.5)}}, tmp_path / "state.pt")
    with pytest.raises(NotImplementedError, match="energy offset"):
        tf.convert_directory(tmp_path)

    torch.save({k: v for k, v in sd.items() if k != "model.layers.1.linears_tensor.5.weight"},
               tmp_path / "state.pt")
    with pytest.raises(KeyError, match="layers.1.linears_tensor.5.weight"):
        tf.convert_directory(tmp_path / "state.pt")


def test_matgl_checkpoint_loads_through_the_hub_and_the_foundation_cache(tmp_path):
    """A local MatGL directory loads with from_pretrained, and a registered model already in the cache
    loads with TensorNet.from_foundation in the asked dtype: the weights of the direct conversion."""
    from xnn.common.models import from_pretrained
    from xnn.gnn.models import tensornet_foundation as tf
    _matgl_directory(tmp_path / "ckpt")
    direct, _ = tf.convert_directory(tmp_path / "ckpt")
    hub = from_pretrained(tmp_path / "ckpt", wrap=False, cache_dir=tmp_path / "cache")
    assert isinstance(hub, TensorNet)
    _same_converted_weights(hub, direct)

    name = "tensornet-matpes-r2scan-2025.2"
    _, revision = tf.FOUNDATION_MODELS[name]
    folder = tmp_path / "models" / "matgl" / name / revision
    _matgl_directory(folder, seed=3)
    model = TensorNet.from_foundation(name, dtype="float64", cache_dir=tmp_path / "models")
    assert not model.training and model.tensor_embedding.emb.weight.dtype == torch.float64
    _same_converted_weights(model, tf.convert_directory(folder, torch.float64)[0])


def _same_converted_weights(model, ref, species=(1, 6, 8, 14)):
    """Every converted parameter equal; of the embedding, the rows of the checkpoint's elements (the
    others keep their initial values)."""
    theirs = dict(model.named_parameters())
    for name, p in ref.named_parameters():
        a, b = theirs[name].double(), p.double()
        if name == "tensor_embedding.emb.weight":
            a, b = a[list(species)], b[list(species)]
        assert torch.equal(a, b), name


def _matgl_fidelity(dtype):
    path = os.environ.get("MATGL_TENSORNET_PATH")
    if not path:
        pytest.skip("set MATGL_TENSORNET_PATH to a MatGL TensorNet directory (model.json, state.pt)")
    pytest.importorskip("matgl")
    import subprocess
    import sys
    script = os.path.join(os.path.dirname(__file__), "tensornet_matgl_fidelity.py")
    res = subprocess.run([sys.executable, script, path, dtype], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    return float(res.stdout.strip().splitlines()[-1])


def test_fidelity_vs_matgl_float64():
    assert _matgl_fidelity("float64") < 1e-10


def test_fidelity_vs_matgl_float32():
    assert _matgl_fidelity("float32") < 1e-4
