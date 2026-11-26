"""NewtonNet tests: the paper's equations (Digital Discovery 2022) and the authors' current code, physical
invariances and equivariances, smoothness, TorchScript deployment, configs, the latent-force loss and
(optionally) fidelity to the authors' code.

The centerpiece is ``_reference``: an independent, loop-based implementation of the equations (the message
of eq 5 to 7, the force features of eq 8 to 12, the displacement and energy update of eq 13 to 16 or the
contraction of the current code, the readout of eq 20 and 21 and the direct forces) that reuses only the
model's parameters. Given the same weights the model must reproduce it to float64 precision.
"""
import math
import os

import numpy as np
import pytest
import torch

from xnn.common.config import from_dict
from xnn.common.data import collate, structure_to_graph
from xnn.common.models import ForceStressOutput, available_models, build_model
from xnn.gnn.models.newtonnet import NewtonNet

PAPER = dict(latent_force=True, displacement=True, rbf_bias=True, rbf_prefactor=math.sqrt(2 / 5.0),
             readout_widths=[128, 64])
VARIANTS = [{}, {"latent_force": True}, {"displacement": True}, {"latent_force": True, "displacement": True},
            {"layer_norm": True}, {"direct_forces": True}, {"cutoff_function": "cosine"},
            {"rbf_bias": True, "rbf_prefactor": 0.7, "readout_widths": [6, 5, 4]}]


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _structure(n=6, seed=0, spread=3.0):
    rng = np.random.default_rng(seed)
    pos = []
    while len(pos) < n:
        p = rng.uniform(0, spread, 3)
        if all(np.linalg.norm(p - q) > 0.8 for q in pos):
            pos.append(p)
    return {"pos": np.array(pos), "atomic_numbers": ([1, 6, 8, 7] * n)[:n]}


def _graph(n=6, cutoff=5.0, seed=0, periodic=False, cell=6.0):
    s = _structure(n, seed)
    if periodic:
        s["cell"] = np.eye(3) * cell
        s["pbc"] = [True, True, True]
    return structure_to_graph(s, cutoff)


def _small(seed=0, **kw):
    """A small NewtonNet with every parameter nonzero."""
    torch.manual_seed(seed)
    kw.setdefault("n_features", 8)
    kw.setdefault("n_interactions", 2)
    kw.setdefault("n_rbf", 6)
    kw.setdefault("cutoff", 5.0)
    model = NewtonNet(**kw)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("bias") or "norm" in name:
                p.uniform_(-0.2, 0.2)
        model.atom_scale.weight.uniform_(0.5, 1.5)
        model.atom_ref.weight.uniform_(-0.5, 0.5)
        if model.force_scale is not None:
            model.force_scale.weight.uniform_(0.5, 1.5)
    return model


def _silu(x):
    return x * torch.sigmoid(x)


def _lin(layer, x):
    y = x @ layer.weight.T
    return y if layer.bias is None else y + layer.bias


def _mlp(seq, x):
    return _lin(seq[2], _silu(_lin(seq[0], x)))


def _reference(model: NewtonNet, g):
    """Energy ``(1,)``, direct forces (or None) and latent forces from loops over atoms and edges."""
    Z = g.atomic_numbers.tolist()
    n = len(Z)
    vecs = g.edge_vectors()
    src, dst = g.edge_index[0].tolist(), g.edge_index[1].tolist()
    rc, R = model.cutoff, model.rbf.freqs.numel()
    edges = []
    for e in range(len(src)):
        rij = -vecs[e]                                    # r_j - r_i (with the image shift)
        d = rij.norm()
        x = d / rc
        k = torch.arange(1, R + 1)
        basis = model.rbf.norm * torch.sin(k * math.pi * x) / d
        if hasattr(model.cutoff_fn, "p"):
            p = model.cutoff_fn.p
            cut = 1 - (p + 1) * (p + 2) / 2 * x ** p + p * (p + 2) * x ** (p + 1) - p * (p + 1) / 2 * x ** (p + 2)
        else:
            cut = 0.5 * (torch.cos(math.pi * x) + 1)
        edges.append((src[e], dst[e], rij / d, basis, cut))
    F = model.n_features
    a = [model.embedding.weight[z] for z in Z]
    f = [torch.zeros(3, F) for _ in range(n)]
    dr = [torch.zeros(3, F) for _ in range(n)]
    lat = [torch.zeros(3) for _ in range(n)]
    for blk in model.interactions:
        x = [_mlp(blk.phi_a, a[i]) for i in range(n)]
        msg = [x[i] * x[j] * _lin(blk.phi_rbf, basis) * cut for (j, i, u, basis, cut) in edges]
        a_new = [a[i].clone() for i in range(n)]
        f_new = [f[i].clone() for i in range(n)]
        for (j, i, u, _, _), m in zip(edges, msg):
            a_new[i] = a_new[i] + m
            if model.latent_force:
                force = _lin(blk.phi_F, m) * u
                lat[i] = lat[i] + force
            else:
                force = u
            f_new[i] = f_new[i] + _mlp(blk.phi_f, m)[None, :] * force[:, None]
            if not model.displacement:
                f_new[i] = f_new[i] + _mlp(blk.phi_p, m)[None, :] * f[j]
        a_out = []
        if model.displacement:
            dr_new = [_mlp(blk.phi_r, a_new[i])[None, :] * f_new[i] for i in range(n)]
            for (j, i, _, _, _), m in zip(edges, msg):
                dr_new[i] = dr_new[i] + _mlp(blk.phi_rj, m)[None, :] * dr[j]
            for i in range(n):
                a_out.append(a_new[i] - _mlp(blk.phi_u, a_new[i]) * (f_new[i] * dr_new[i]).sum(0))
            dr = dr_new
        else:
            for i in range(n):
                a_out.append(a_new[i] + (f_new[i] * (f_new[i] @ blk.mix.weight.T)).sum(0))
        if isinstance(blk.norm, torch.nn.LayerNorm):
            a_out = [(v - v.mean()) / torch.sqrt(v.var(unbiased=False) + blk.norm.eps) * blk.norm.weight
                     + blk.norm.bias for v in a_out]
        a, f = a_out, f_new
    energy = 0.0
    forces = []
    for i in range(n):
        h = a[i]
        lins = [m for m in model.readout if isinstance(m, torch.nn.Linear)]
        for k, lin in enumerate(lins):
            h = _lin(lin, h)
            if k < len(lins) - 1:
                h = _silu(h)
        y = h[0] * model.atom_scale.weight[Z[i], 0]
        energy = energy + model.energy_scale * y + model.energy_shift + model.atom_ref.weight[Z[i], 0]
        if model.force_head is not None:
            head = model.force_head
            w = _lin(head[4], _silu(_lin(head[2], _silu(_lin(head[0], a[i])))))
            forces.append((f[i] * w[None, :]).sum(-1) * model.force_scale.weight[Z[i], 0])
    direct = torch.stack(forces) if forces else None
    return torch.as_tensor(energy).reshape(1), direct, torch.stack(lat)


def _rotation(seed=0, proper=True):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    if (np.linalg.det(q) > 0) != proper:
        q[:, 0] = -q[:, 0]
    return q


# registry and architecture

def test_registered():
    assert "newtonnet" in available_models()


def test_defaults_are_the_current_code():
    """NewtonNet() has the authors' current architecture: F = 128, T = 3, 20 Bessel functions with the
    degree-9 envelope at 5 A, and 401,135 parameters (the authors' count without its 20 frozen frequencies)."""
    m = NewtonNet()
    assert m.embedding.weight.shape == (119, 128) and m.embedding.padding_idx == 0
    assert len(m.interactions) == 3 and m.cutoff == 5.0
    assert m.rbf.freqs.shape == (20,) and m.rbf.norm == 5.0
    assert m.cutoff_fn.p == 9
    blk = m.interactions[0]
    assert blk.phi_rbf.bias is None and blk.phi_f[0].bias is None and blk.phi_p[0].bias is None
    assert isinstance(blk.phi_F, torch.nn.Identity) and isinstance(blk.phi_r, torch.nn.Identity)
    assert [lin.weight.shape for lin in m.readout[::2]] == [(128, 128), (128, 128), (1, 128)]
    assert sum(p.numel() for p in m.parameters()) == 401_135
    assert not m.latent_force and not m.displacement and m.force_head is None


def test_paper_architecture():
    """The paper's model: force magnitudes (phi_F, no bias), displacement networks (phi_r' without bias),
    a 128 -> 64 -> 1 readout and about 500k parameters in the interaction blocks."""
    m = NewtonNet(**PAPER)
    blk = m.interactions[0]
    assert blk.phi_F.weight.shape == (1, 128) and blk.phi_F.bias is None
    assert blk.phi_f[0].bias is not None and blk.phi_rj[0].bias is None and blk.phi_r[0].bias is not None
    assert blk.phi_rbf.bias is not None and isinstance(blk.phi_p, torch.nn.Identity)
    assert [lin.weight.shape for lin in m.readout[::2]] == [(128, 128), (64, 128), (1, 64)]
    n = sum(p.numel() for p in m.interactions.parameters())
    assert abs(n - 500_000) / 500_000 < 0.01
    assert m.rbf.norm == pytest.approx(math.sqrt(2 / 5.0))


def test_options_carry_only_their_parameters():
    base = sum(p.numel() for p in _small().parameters())
    latent = sum(p.numel() for p in _small(latent_force=True).parameters())
    assert latent - base == 2 * (8 + 16)       # per block: phi_F (8 weights), phi_f gains its biases (8 + 8)
    disp = _small(displacement=True)
    assert not any("phi_p" in k or "mix" in k for k in disp.state_dict())
    assert _small(direct_forces=True).force_scale.weight.shape == (119, 1)


def test_invalid_options():
    with pytest.raises(ValueError, match="cutoff_function"):
        NewtonNet(cutoff_function="smooth")
    with pytest.raises(ValueError, match="species"):
        NewtonNet(atomic_energies=[1.0])
    m = _small()
    with pytest.raises(ValueError, match="atomic energies"):
        m.set_atomic_energies([1, 6], [1.0])


def test_radial_basis_and_envelope():
    """The default basis is sin(n pi x) / x with x = r / r_c; the envelope is the degree-9 polynomial."""
    m = NewtonNet(n_rbf=4)
    r = torch.linspace(0.5, 4.9, 7)
    x = r / 5.0
    n = torch.arange(1, 5)
    assert torch.allclose(m.rbf(r), torch.sin(n * math.pi * x[:, None]) / x[:, None], atol=1e-13)
    env = 1 - 55 * x ** 9 + 99 * x ** 10 - 45 * x ** 11
    assert torch.allclose(m.cutoff_fn(r), env, atol=1e-13)
    g = NewtonNet(n_rbf=4, radial_basis="gaussian")
    assert g.rbf(r).shape == (7, 4)


def test_float64_basis_exact_after_cast():
    m = NewtonNet(n_rbf=5).float().double()
    assert torch.equal(m.rbf.freqs, math.pi * torch.arange(1, 6, dtype=torch.float64))


# the equations

@pytest.mark.parametrize("kw", VARIANTS)
def test_forward_matches_equations(kw):
    model = _small(**kw).eval()
    for g in (_graph(5, seed=1), _graph(4, seed=2, periodic=True, cell=4.0)):
        out = model(g)
        energy, direct, lat = _reference(model, g)
        assert torch.allclose(out["energy"], energy, atol=1e-11, rtol=1e-11)
        if direct is not None:
            assert torch.allclose(out["forces"], direct, atol=1e-11)
        if model.latent_force:
            assert torch.allclose(out["latent_forces"], lat, atol=1e-12)
        else:
            assert "latent_forces" not in out


def test_edge_direction_is_the_paper_convention():
    """The latent forces sum phi_F(m_ij) (r_j - r_i) / |r_j - r_i| (eq 8 and 9): for two atoms they point
    along the bond, from i towards j times the magnitude."""
    model = _small(latent_force=True, n_interactions=1).eval()
    g = structure_to_graph({"pos": [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], "atomic_numbers": [6, 8]}, 5.0)
    lat = model(g)["latent_forces"]
    blk = model.interactions[0]
    a = model.embedding.weight[g.atomic_numbers]
    x = blk.phi_a(a)
    m = x[0] * x[1] * blk.phi_rbf(model.rbf(torch.tensor([1.2])))[0] * model.cutoff_fn(torch.tensor([1.2]))
    mag = blk.phi_F(m)
    assert torch.allclose(lat[0], mag * torch.tensor([1.0, 0, 0]), atol=1e-12)
    assert torch.allclose(lat[1], -lat[0], atol=1e-12)


@pytest.mark.parametrize("kw", [{}, PAPER | {"n_features": 8, "readout_widths": None}, {"direct_forces": True}])
@pytest.mark.parametrize("proper", [True, False])
def test_invariance_and_equivariance(kw, proper):
    model = ForceStressOutput(_small(**kw)).eval()
    s = _structure(7, seed=3)
    q = _rotation(1, proper)
    shift = np.array([0.3, -1.2, 2.0])
    out = model(structure_to_graph(s, 5.0))
    out2 = model(structure_to_graph({"pos": s["pos"] @ q.T + shift, "atomic_numbers": s["atomic_numbers"]}, 5.0))
    qt = torch.as_tensor(q)
    assert abs(float(out["energy"] - out2["energy"])) < 1e-11
    assert torch.allclose(out["forces"] @ qt.T, out2["forces"], atol=1e-10)
    assert torch.allclose(torch.einsum("ij,njf->nif", qt, out["node_vectors"]), out2["node_vectors"], atol=1e-10)
    if "latent_forces" in out:
        assert torch.allclose(out["latent_forces"] @ qt.T, out2["latent_forces"], atol=1e-10)


def test_permutation_invariance():
    model = _small(**PAPER | {"n_features": 8, "readout_widths": None}).eval()
    s = _structure(7, seed=2)
    perm = np.random.default_rng(1).permutation(7)
    out = model(structure_to_graph(s, 5.0))
    out2 = model(structure_to_graph({"pos": s["pos"][perm],
                                     "atomic_numbers": np.array(s["atomic_numbers"])[perm]}, 5.0))
    assert abs(float(out["energy"] - out2["energy"])) < 1e-12
    assert torch.allclose(out["node_energy"][perm], out2["node_energy"], atol=1e-12)
    assert torch.allclose(out["latent_forces"][perm], out2["latent_forces"], atol=1e-12)


@pytest.mark.parametrize("kw", [{}, {"latent_force": True, "displacement": True}])
def test_forces_are_minus_gradient(kw):
    model = ForceStressOutput(_small(**kw)).eval()
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


@pytest.mark.parametrize("kw", [{}, {"latent_force": True, "displacement": True}, {"direct_forces": True}])
def test_pes_smooth_across_cutoff(kw):
    """Every edge term carries the envelope (no bias where a neighbor quantity is multiplied), so energies
    and forces are continuous when a neighbor leaves the cutoff sphere."""
    model = ForceStressOutput(_small(cutoff=4.0, **kw)).eval()

    def evaluate(d):
        pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.9, 0.0], [math.sqrt(d ** 2 - 0.45 ** 2), 0.45, 0.0]])
        out = model(structure_to_graph({"pos": pos, "atomic_numbers": [8, 1, 1]}, 4.0))
        return float(out["energy"]), out["forces"].detach().clone()

    eps = 1e-6
    (e_in, f_in), (e_out, f_out) = evaluate(4.0 - eps), evaluate(4.0 + eps)
    assert abs(e_in - e_out) < 1e-10
    assert float((f_in - f_out).abs().max()) < 1e-8


def test_isolated_atoms_have_finite_gradients():
    model = ForceStressOutput(_small(**PAPER | {"n_features": 8, "readout_widths": None}))
    g = structure_to_graph({"pos": [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]], "atomic_numbers": [8, 1]}, 5.0)
    out = model(g)
    (out["energy"].sum() + out["latent_forces"].sum()).backward()
    assert torch.isfinite(out["forces"]).all() and float(out["forces"].abs().max()) == 0.0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize("kw", [{}, {"latent_force": True, "displacement": True}])
def test_directional_information_beyond_distances(kw):
    """A hexagon and two triangles with the same bond length and a cutoff below the second-neighbor
    distance give every atom the same neighbor distances but different angles: the force features
    resolve them in one block."""
    bond, cutoff = 1.5, 2.0
    hexagon = np.array([[bond * math.cos(k * math.pi / 3), bond * math.sin(k * math.pi / 3), 0.0]
                        for k in range(6)])
    tri = np.array([[0.0, 0.0, 0.0], [bond, 0.0, 0.0], [bond / 2, bond * math.sqrt(3) / 2, 0.0]])
    triangles = np.concatenate([tri, tri + np.array([10.0, 0.0, 0.0])])
    model = _small(n_interactions=1, cutoff=cutoff, **kw).eval()
    e_hex = float(model(structure_to_graph({"pos": hexagon, "atomic_numbers": [6] * 6}, cutoff))["energy"])
    e_tri = float(model(structure_to_graph({"pos": triangles, "atomic_numbers": [6] * 6}, cutoff))["energy"])
    assert abs(e_hex - e_tri) > 1e-8


def test_size_extensivity_and_batching():
    model = _small(latent_force=True, displacement=True).eval()
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
    assert abs(float(out["energy"] - model(_graph(n=6))["energy"])) > 1e-6


def test_float32_forward_matches_float64():
    model = _small().eval()
    g = _graph(n=8)
    out64 = model(g)
    g32 = structure_to_graph({"pos": g.pos.float(), "atomic_numbers": g.atomic_numbers}, 5.0)
    out32 = model.float()(g32)
    assert abs(float(out64["energy"] - out32["energy"].double())) < 1e-4 * max(1.0, abs(float(out64["energy"])))


# deployment

@pytest.mark.parametrize("kw", [{}, {"latent_force": True, "displacement": True, "layer_norm": True}])
def test_scriptable_and_lammps_export(tmp_path, kw):
    from xnn.common.deploy import export_to_lammps

    model = _small(**kw).eval()
    g = _graph(n=6, periodic=True)
    scripted = torch.jit.script(model)
    args = (g.atomic_numbers, g.edge_index, g.edge_vectors())
    assert (scripted.node_energy(*args) - model.node_energy(*args)).abs().max() < 1e-12
    a, f, lat = scripted.representation(*args)
    assert a.shape == (6, 8) and f.shape == (6, 3, 8) and lat.shape == (6, 3)
    path = str(tmp_path / "newtonnet_lammps.pt")
    export_to_lammps(model, 5.0, path)
    out = torch.jit.load(path)(g.pos, g.edge_index, g.cell_shifts, g.atomic_numbers, g.cell[0])
    ref = ForceStressOutput(model)(g)
    assert abs(float(out["total_energy"]) - float(ref["energy"])) < 1e-5
    assert (out["forces"] - ref["forces"].detach()).abs().max() < 1e-5


def test_direct_forces_have_no_energy_core():
    model = _small(direct_forces=True).eval()
    g = _graph()
    with pytest.raises(RuntimeError, match="direct_forces"):
        model.node_energy(g.atomic_numbers, g.edge_index, g.edge_vectors())
    out = ForceStressOutput(model)(g)
    assert out["forces"].shape == (6, 3)


def test_les_wrapper_reads_node_features():
    from xnn.common.models.les import LatentEwald

    model = LatentEwald(_small()).double()
    out = model(_graph())
    assert out["energy"].shape == (1,) and torch.isfinite(out["energy"]).all()


def test_multihead_and_reference_energies():
    from xnn.common.finetune import MultiHead, get_atomic_energies, set_atomic_energies

    model = _small()
    with torch.no_grad():
        model.atom_ref.weight.zero_()
    set_atomic_energies(model, {1: -0.5, 6: -1.0, 8: -2.0})
    assert get_atomic_energies(model) == {1: -0.5, 6: -1.0, 8: -2.0}
    model.set_energy_scale_shift(2.0, -0.25)
    assert float(model.energy_scale) == 2.0 and float(model.energy_shift) == -0.25
    out = MultiHead(model, ["a", "b"])(_graph())
    assert out["energy"].shape == (1,)


# the force-direction loss (eq 23)

def test_latent_force_loss_term():
    from xnn.common.train.losses import weighted_loss

    g = collate([_graph(5, seed=1), _graph(4, seed=2)])
    g.forces = torch.randn(9, 3)
    lat = g.forces * torch.rand(9, 1) + 0.0
    pred = {"energy": torch.zeros(2), "latent_forces": lat.clone().requires_grad_(True)}
    loss, logs = weighted_loss(pred, g, 0.0, 0.0, 0.0, latent_force_weight=2.0)
    assert logs["latent_force_dir"] == pytest.approx(0.0, abs=1e-12)        # parallel: cos = 1
    pred["latent_forces"] = (-lat).clone().requires_grad_(True)
    counts, parts = {}, {}
    loss, logs = weighted_loss(pred, g, 0.0, 0.0, 0.0, latent_force_weight=2.0, counts=counts, parts=parts)
    assert logs["latent_force_dir"] == pytest.approx(2.0) and float(loss) == pytest.approx(4.0)
    assert counts["latent_force_dir"][1] == 2.0 and float(parts["latent_force_dir"]) == pytest.approx(4.0)
    # a zero latent force counts as 1 and has a finite gradient
    pred["latent_forces"] = torch.zeros(9, 3, requires_grad=True)
    loss, logs = weighted_loss(pred, g, 0.0, 0.0, 0.0, latent_force_weight=1.0)
    loss.backward()
    assert logs["latent_force_dir"] == pytest.approx(1.0) and torch.isfinite(pred["latent_forces"].grad).all()
    loss, logs = weighted_loss(pred, g, 0.0, 0.0, 0.0, latent_force_weight=1.0, sync=False)
    assert float(logs["latent_force_dir"]) == pytest.approx(1.0)


def test_train_with_latent_force_loss(tmp_path):
    from xnn.common.data import AtomicDataset
    from xnn.common.train import Trainer

    rng = np.random.default_rng(0)
    structures = []
    for i in range(6):
        s = _structure(5, seed=i)
        s.update(energy=float(rng.normal()), forces=rng.normal(size=(5, 3)))
        structures.append(s)
    cfg = from_dict({"model": {"name": "newtonnet", "n_features": 8, "n_interactions": 1, "n_rbf": 4,
                               "cutoff": 5.0, "extra": {"latent_force": True, "displacement": True}},
                     "optim": {"epochs": 2, "force_weight": 50.0, "latent_force_weight": 1.0},
                     "data": {"batch_size": 3, "val_fraction": 0.0},
                     "output_dir": str(tmp_path)})
    logs = Trainer(cfg, AtomicDataset(structures, 5.0)).fit()["train"]
    assert {"energy_mse", "force_mse", "latent_force_dir"} <= set(logs)


# config

def test_from_config_extras():
    cfg = from_dict({"model": {
        "name": "newtonnet", "cutoff": 4.0, "n_features": 16, "n_interactions": 2, "n_rbf": 6,
        "extra": {"activation": "swish", "cutoff_function": "poly", "num_polynomial_cutoff": 7,
                  "rbf_bias": True, "rbf_prefactor": 0.5, "latent_force": True, "displacement": True,
                  "layer_norm": True, "readout_widths": [12, 6], "weight_init": "xavier_uniform",
                  "energy_shift": -1.5, "energy_scale": 0.2, "species": [1, 6, 8],
                  "atomic_energies": [-13.6, -1030.0, -2043.0]}}})
    m = build_model(cfg.model)
    assert m.rbf.freqs.shape == (6,) and m.cutoff == 4.0 and m.rbf.norm == 0.5 and m.cutoff_fn.p == 7
    assert m.latent_force and m.displacement and isinstance(m.interactions[0].norm, torch.nn.LayerNorm)
    assert m.readout[0].weight.shape == (12, 16) and m.readout[2].weight.shape == (6, 12)
    assert float(m.energy_shift) == -1.5 and float(m.energy_scale) == pytest.approx(0.2)
    assert float(m.atom_ref.weight[6, 0]) == pytest.approx(-1030.0)
    assert m(_graph())["latent_forces"].shape == (6, 3)
    cfg = from_dict({"model": {"name": "newtonnet", "n_features": 8, "n_rbf": 5,
                               "extra": {"radial_basis": "gaussian", "cutoff_function": "cosine",
                                         "direct_forces": True}}})
    m = build_model(cfg.model)
    assert m.force_head is not None and not hasattr(m.cutoff_fn, "p")


def test_shipped_config_is_the_defaults():
    """configs/model/newtonnet.yaml lists every option at the constructor's default."""
    import yaml

    path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "model", "newtonnet.yaml")
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    m = build_model(from_dict({"model": raw}).model)
    d = NewtonNet()
    assert set(m.state_dict()) == set(d.state_dict())
    assert all(m.state_dict()[k].shape == v.shape for k, v in d.state_dict().items())
    assert m.rbf.norm == d.rbf.norm and m.cutoff_fn.p == d.cutoff_fn.p


def test_reference_key_translation():
    cfg = from_dict({"model": {"name": "newtonnet", "cutoff": 4.5, "n_features": 16, "n_basis": 12,
                               "cutoff_network": "cosine", "layer_norm": True}})
    assert cfg.model.n_rbf == 12 and cfg.model.cutoff == 4.5
    m = build_model(cfg.model)
    assert not hasattr(m.cutoff_fn, "p") and isinstance(m.interactions[0].norm, torch.nn.LayerNorm)
    cfg = from_dict({"model": {"name": "newtonnet", "n_features": 8, "resolution": 7}})
    assert cfg.model.n_rbf == 7


# fidelity to the authors' code (optional)

def _fidelity(dtype):
    import subprocess
    import sys

    from newtonnet_fidelity import reference_version

    if not reference_version():
        pytest.skip("the NewtonNet reference code is not installed")
    script = os.path.join(os.path.dirname(__file__), "newtonnet_fidelity.py")
    res = subprocess.run([sys.executable, script, dtype], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    return float(res.stdout.strip().splitlines()[-1])


def test_fidelity_vs_reference_float64():
    """Transplanted weights give the reference energies, forces, stress and latent forces at the float64
    floor (a layout or sign convention error would show as O(1))."""
    assert _fidelity("float64") < 1e-12


def test_fidelity_vs_reference_float32():
    """The same in float32, within 1e-5 or a few times the reference's own float32 error."""
    assert _fidelity("float32") <= 1.0


# fast path (fused edge sums)

FAST_VARIANTS = [{}, {"latent_force": True, "displacement": True, "layer_norm": True}, {"direct_forces": True}]


def _check_kernel_gradients(device, fast_mode=False):
    """Every first and second derivative of the fused edge sums (finite differences), each term alone and
    together, and an empty edge list."""
    from xnn.gnn.fast.newtonnet_kernels import edge_sum

    gen = torch.Generator().manual_seed(0)
    n, n_edges, F = 5, 12, 6           # a width that is not a power of two (padded lanes)

    def rand(*shape):
        return torch.randn(*shape, dtype=torch.float64, generator=gen).to(device).requires_grad_(True)

    c1, u, c2, v = rand(n_edges, F), rand(n_edges, 3), rand(n_edges, F), rand(n, 3, F)
    src, dst = torch.randint(0, n, (2, n_edges), generator=gen).to(device)
    cases = {"both": (lambda c1, u, c2, v: edge_sum(n, src, dst, c1, u, c2, v), (c1, u, c2, v)),
             "outer": (lambda c1, u: edge_sum(n, src, dst, c1, u), (c1, u)),
             "neighbors": (lambda c2, v: edge_sum(n, src, dst, None, None, c2, v), (c2, v))}
    for fn, inputs in cases.values():
        # atomic adds sum in a run-dependent order
        assert torch.autograd.gradcheck(fn, inputs, nondet_tol=1e-12, fast_mode=fast_mode)
        assert torch.autograd.gradgradcheck(fn, inputs, nondet_tol=1e-12, fast_mode=fast_mode)
    ref = (c1.unsqueeze(1) * u.unsqueeze(-1) + c2.unsqueeze(1) * v[src])
    ref = torch.zeros(n, 3, F, dtype=torch.float64, device=device).index_add(0, dst, ref)
    assert torch.allclose(edge_sum(n, src, dst, c1, u, c2, v), ref, atol=1e-13)
    assert edge_sum(n, src[:0], dst[:0], c1[:0], u[:0], c2[:0], v).abs().max() == 0


def _fast_outputs(model, graph, use_fast, training=False):
    model.set_use_fast(use_fast)
    wrapped = ForceStressOutput(model, compute_stress=graph.cell is not None).train(training)
    out = wrapped(graph)
    assert all(b.fast_active == use_fast for b in model.interactions)
    keys = [k for k in ("energy", "forces", "stress", "latent_forces") if out.get(k) is not None]
    return {k: out[k].detach().double() for k in keys}


def _training_gradients(model, graph, use_fast):
    """The weight gradients of an energy and force loss (forces with create_graph)."""
    model.set_use_fast(use_fast)
    model.zero_grad(set_to_none=True)
    out = ForceStressOutput(model).train()(graph)
    assert all(b.fast_active == use_fast for b in model.interactions)
    loss = out["energy"].pow(2).sum() + out["forces"].pow(2).sum()
    if "latent_forces" in out:
        loss = loss + out["latent_forces"].pow(2).sum()
    loss.backward()
    return {name: p.grad.detach().double().clone() for name, p in model.named_parameters() if p.grad is not None}


def _compare(ref, fast, tol):
    assert ref.keys() == fast.keys()
    for key, r in ref.items():
        err = (fast[key] - r).abs().max() / r.abs().max().clamp_min(1e-30)
        assert err < tol, (key, float(err))


@pytest.fixture
def interpreted(monkeypatch):
    """The fused kernels under Triton's interpreter (chosen by conftest.py on machines without CUDA), on CPU
    tensors, with the fast path offered on CPUs."""
    if os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("runs where Triton's interpreter is on (no CUDA device)")
    pytest.importorskip("triton")
    import xnn.gnn.fast.newtonnet as fast
    monkeypatch.setattr(fast, "supported", lambda device, dtype: True)


def test_fast_support_needs_cuda():
    from xnn.gnn.fast.newtonnet import supported
    assert not supported(torch.device("cpu"), torch.float64)
    m = _small()
    m(_graph())
    assert not any(b.fast_active for b in m.interactions)
    with pytest.raises(ValueError):
        m.set_use_fast("sometimes")


def test_fast_kernel_gradients_interpreted(interpreted):
    """The fused edge sums and their first and second derivatives under the Triton interpreter."""
    _check_kernel_gradients("cpu", fast_mode=True)


@pytest.mark.parametrize("kw", FAST_VARIANTS)
def test_fast_path_parity_interpreted(interpreted, kw):
    """The whole model with the fused edge sums on a CPU (the interpreter) and the reference."""
    model = _small(**kw).eval()
    graph = collate([_graph(n=6, seed=1, periodic=True, cell=4.5), _graph(n=5, seed=2, periodic=True)])
    _compare(_fast_outputs(model, graph, False), _fast_outputs(model, graph, True), 1e-12)


def test_fast_path_training_interpreted(interpreted):
    """Force training through the fused edge sums gives the reference's weight gradients (interpreter)."""
    model = _small(latent_force=True, displacement=True, n_features=4)
    graph = _graph(n=4, seed=3)
    _compare(_training_gradients(model, graph, False), _training_gradients(model, graph, True), 1e-11)
    model = _small(n_features=4)
    _compare(_training_gradients(model, graph, False), _training_gradients(model, graph, True), 1e-11)


def _cuda_batch(dtype):
    batch = collate([_graph(n=14, seed=1, periodic=True, cell=5.0), _graph(n=9, seed=2)]).to("cuda")
    batch.pos = batch.pos.to(dtype)
    batch.cell = batch.cell.to(dtype)
    return batch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_fast_kernel_gradients():
    pytest.importorskip("triton")
    _check_kernel_gradients("cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kw", FAST_VARIANTS)
def test_fast_path_parity(dtype, kw):
    """The fused edge sums give the reference energies, forces, stress and latent forces."""
    pytest.importorskip("triton")
    model = _small(n_features=32, n_interactions=3, **kw).to("cuda", dtype).eval()
    batch = _cuda_batch(dtype)
    _compare(_fast_outputs(model, batch, False), _fast_outputs(model, batch, True),
             1e-12 if dtype == torch.float64 else 1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kw", FAST_VARIANTS[:2])
def test_fast_path_training(dtype, kw):
    """Force training through the fused edge sums gives the reference's weight gradients."""
    pytest.importorskip("triton")
    model = _small(n_features=32, n_interactions=3, **kw).to("cuda", dtype)
    ref = _training_gradients(model, _cuda_batch(dtype), False)
    fast = _training_gradients(model, _cuda_batch(dtype), True)
    _compare(ref, fast, 1e-11 if dtype == torch.float64 else 1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_fast_path_hessian():
    """The position Hessian through the fused edge sums equals the reference's."""
    pytest.importorskip("triton")
    model = _small(latent_force=True, displacement=True).to("cuda").eval()
    s = _structure(5, seed=4)
    hess = {}
    for use_fast in (False, True):
        model.set_use_fast(use_fast)
        pos = torch.as_tensor(s["pos"], device="cuda").requires_grad_(True)

        def energy(p):
            g = structure_to_graph({"pos": p.detach().cpu().numpy(), "atomic_numbers": s["atomic_numbers"]},
                                   5.0).to("cuda")
            g.pos = p
            return model(g)["energy"].sum()

        hess[use_fast] = torch.autograd.functional.hessian(energy, pos)
    assert torch.allclose(hess[True], hess[False], atol=1e-11 * hess[False].abs().max())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_auto_crossover_per_mode(monkeypatch):
    """"auto" applies the evaluation crossover in evaluation mode and the training one in training mode."""
    pytest.importorskip("triton")
    import xnn.gnn.fast.newtonnet as fast
    from xnn.common.models.fast import AutoPolicy

    never = AutoPolicy(default_min_edges=10 ** 9, default_min_edges_float64=10 ** 9)
    always = AutoPolicy(default_min_edges=0, default_min_edges_float64=0)
    model = _small().to("cuda").set_use_fast("auto")
    batch = _cuda_batch(torch.float64)
    for training in (False, True):
        for name in ("AUTO_POLICY", "AUTO_POLICY_TRAINING"):
            with monkeypatch.context() as patch:
                patch.setattr(fast, "AUTO_POLICY", always if name == "AUTO_POLICY" else never)
                patch.setattr(fast, "AUTO_POLICY_TRAINING", always if name == "AUTO_POLICY_TRAINING" else never)
                model.train(training)
                model(batch)
                expected = (name == "AUTO_POLICY_TRAINING") == training
                assert all(b.fast_active == expected for b in model.interactions)


def _adapter_parity(device, dtype, tol):
    """With LoRA adapters on the networks the fast path still
    gives the reference numbers."""
    from xnn.common.finetune import inject_lora

    for kw in ({}, {"latent_force": True, "displacement": True}):
        model = _small(**kw)
        inject_lora(model, rank=2)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if "lora" in name:
                    p.normal_(std=0.1)
        model = model.to(device, dtype).eval()
        graph = _graph(n=7, seed=5, periodic=True).to(device)
        graph.pos, graph.cell = graph.pos.to(dtype), graph.cell.to(dtype)
        _compare(_fast_outputs(model, graph, False), _fast_outputs(model, graph, True), tol)


def test_fast_path_with_adapters_interpreted(interpreted):
    _adapter_parity("cpu", torch.float64, 1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_fast_path_with_adapters():
    pytest.importorskip("triton")
    _adapter_parity("cuda", torch.float64, 1e-12)


@pytest.mark.parametrize("direct", [False, True])
def test_hessians(direct):
    """``hessian`` is the energy Hessian, or with direct forces the symmetrized force constants -dF/dx of the
    predicted forces; ``hessian_vector_product`` multiplies by the same matrix, differentiably."""
    from xnn.common.models import hessian, hessian_vector_product

    model = _small(direct_forces=direct).eval()
    s = _structure(4, seed=6)
    g = structure_to_graph(s, 5.0)
    pos0 = torch.as_tensor(s["pos"]).reshape(-1)

    def graph_at(p):
        gg = structure_to_graph(s, 5.0)
        gg.pos = p.reshape(-1, 3)
        return gg

    if direct:
        def minus_forces(p):
            return -ForceStressOutput(model)(graph_at(p))["forces"].reshape(-1)
        jac = torch.autograd.functional.jacobian(minus_forces, pos0)
        ref = 0.5 * (jac + jac.T)
        assert (jac - jac.T).abs().max() > 1e-6            # direct forces are not conservative
    else:
        ref = torch.autograd.functional.hessian(lambda p: model(graph_at(p))["energy"].sum(), pos0)
    H = hessian(ForceStressOutput(model), g)[0]
    assert torch.allclose(H, ref, atol=1e-10)
    v = torch.randn(4, 3)
    hv = hessian_vector_product(model, g, v)
    assert torch.allclose(hv.reshape(-1), ref @ v.reshape(-1), atol=1e-10)
    hv2 = hessian_vector_product(model, g, torch.stack([v, 2 * v]), create_graph=True)
    assert torch.allclose(hv2[1], 2 * hv, atol=1e-10)
    hv2.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_direct_force_hessians_without_neighbors():
    """Direct forces that do not depend on the positions (isolated atoms) give zero force constants, with
    trainable and with frozen weights."""
    from xnn.common.models import hessian, hessian_vector_product

    model = _small(direct_forces=True).eval()
    g = structure_to_graph({"pos": [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]], "atomic_numbers": [8, 1]}, 5.0)
    for trainable in (True, False):
        model.requires_grad_(trainable)
        assert float(hessian(model, g)[0].abs().max()) == 0.0
        assert float(hessian_vector_product(model, g, torch.ones(2, 3)).abs().max()) == 0.0
