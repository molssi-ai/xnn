"""PaiNN tests: manuscript fidelity (ICML 2021), physical invariances and
equivariances of the tensorial heads, smoothness, TorchScript deployment,
config handling and (optionally) fidelity to the reference code.

The centerpiece is ``_reference``: an independent, loop-based implementation
of the paper's equations (the message block of eq 7 and 8, the update block of
eq 9 and 10, the readout of Fig. 2a, the gated equivariant blocks of Fig. 3
and the tensorial outputs of eq 13 and 14) that reuses only the model's
*parameters*. Given the same weights the model must reproduce it to float64
precision.
"""
import math
import os

import numpy as np
import pytest
import torch

from xnn.common.config import from_dict
from xnn.common.data import collate, structure_to_graph
from xnn.common.models import ForceStressOutput, available_models, build_model
from xnn.gnn.models.painn import ATOMIC_MASSES, GatedEquivariantBlock, PaiNN

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
    kw.setdefault("cutoff", 5.0)
    return PaiNN(**kw)


def _silu(x):
    return x * torch.sigmoid(x)


def _lin(layer, x):
    y = x @ layer.weight.T
    return y if layer.bias is None else y + layer.bias


# registry / defaults

def test_registered():
    assert "painn" in available_models()


def test_defaults():
    """PaiNN() has F = 128, T = 3 and 20 Gaussians with a cosine cutoff at 5 A."""
    m = PaiNN()
    assert m.embedding.weight.shape == (100, 128)
    assert len(m.interactions) == 3 and m.cutoff == 5.0
    assert m.rbf.centers.tolist() == pytest.approx(torch.linspace(0, 5, 20).tolist())
    assert m.rbf.gamma == pytest.approx(0.5 * (19 / 5.0) ** 2)
    assert not m.atomic_dipoles and not m.shared_filters and not m.shared_interactions


def test_paper_architecture():
    """With the paper's options, PaiNN is the ICML 2021 architecture: F = 128,
    T = 3, 20 Bessel functions with a cosine cutoff at 5 A, a 128 -> 64 -> 1
    readout, and about 588k parameters (Table IV: 588.3k)."""
    m = PaiNN(radial_basis="bessel")
    assert m.rbf.freqs.shape == (20,) and m.rbf.norm == 1.0
    assert m.filters.weight.shape == (3 * 384, 20)              # three blocks of (s, vv, vs)
    assert m.interactions[0].update.lin_v.weight.shape == (256, 128)
    assert m.readout[0].weight.shape == (64, 128) and m.readout[2].weight.shape == (1, 64)
    assert m.dipole_head is None and m.polarizability_head is None
    n = sum(p.numel() for p in m.parameters())
    assert abs(n - 588_300) / 588_300 < 0.01
    # the tensorial heads: two gated blocks 128 -> 64 -> 1
    t = PaiNN(radial_basis="bessel", dipole=True, atomic_dipoles=True, polarizability=True)
    assert [b.lin_v.weight.shape for b in t.dipole_head.blocks] == [(128, 128), (2, 64)]
    assert t.polarizability_head.blocks[1].net[2].weight.shape == (2, 64)
    # latent charges alone (eq 12): an atom-wise network 128 -> 64 -> 1
    q = PaiNN(dipole=True)
    assert [lin.weight.shape for lin in q.dipole_head[::2]] == [(64, 128), (1, 64)]


def test_weight_sharing():
    """Shared filters: one block's filters for all; shared interactions: one set of
    message and update networks, filters per block."""
    m = PaiNN(n_features=8, shared_filters=True)
    assert m.filters.weight.shape[0] == m.filter_width == 24
    assert m.interactions[0].update is not m.interactions[1].update
    m = PaiNN(n_features=8, shared_interactions=True)
    assert m.interactions[0].message.phi is m.interactions[2].message.phi
    assert m.interactions[0].update is m.interactions[2].update
    assert m.filters.weight.shape[0] == 3 * m.filter_width
    both = PaiNN(n_features=8, shared_interactions=True, shared_filters=True)
    assert _count(both) < _count(m) < _count(PaiNN(n_features=8))


def _count(module):
    return sum(p.numel() for p in module.parameters())


def _n_params(**kw):
    return sum(p.numel() for p in PaiNN(**kw).parameters()) - 100   # without atom_ref


def _readout(F):
    return F * (F // 2) + 2 * (F // 2) + 1


@pytest.mark.parametrize("F,kw,per_block", [
    (128, {}, lambda F: 11 * F * F + 71 * F),
    (134, {"scalar_product": False}, lambda F: 10 * F * F + 70 * F),
    (135, {"vector_propagation": False}, lambda F: 10 * F * F + 49 * F),
    (142, {"scalar_product": False, "vector_propagation": False},
     lambda F: 9 * F * F + 48 * F),
    (174, {"vector_features": False}, lambda F: 4 * F * F + 25 * F),
])
def test_ablations_drop_their_parameters(F, kw, per_block):
    """Ablated terms carry no parameters: the counts are the closed forms of
    the remaining layers (embedding, 3 blocks of 20 radial functions, the
    F -> F/2 -> 1 readout)."""
    assert _n_params(n_features=F, **kw) == 100 * F + 3 * per_block(F) + _readout(F)


def test_invalid_options():
    with pytest.raises(ValueError, match="radial_basis"):
        PaiNN(n_features=8, radial_basis="chebyshev")
    with pytest.raises(ValueError):
        PaiNN(n_features=8, atomic_energies=[1.0])


def test_radial_basis_is_the_paper_form():
    """The filters are built from sin(n pi r / r_cut) / r, 1 <= n <= n_rbf,
    times the cosine cutoff."""
    m = _small(n_rbf=5, radial_basis="bessel")
    r = torch.linspace(0.3, 4.9, 25)
    ref = torch.stack([torch.sin(n * math.pi * r / 5.0) / r for n in range(1, 6)], -1)
    assert torch.allclose(m.rbf(r), ref, atol=1e-14)
    assert torch.allclose(m.cutoff_fn(r), 0.5 * (torch.cos(math.pi * r / 5.0) + 1))
    assert float(m.cutoff_fn(torch.tensor([5.0, 6.0])).abs().max()) == 0.0


@pytest.mark.parametrize("shared_filters", [False, True])
def test_per_block_filter_state_dict_loads(shared_filters):
    """A state dict with one filter layer per block (xnn 0.22 and earlier) loads."""
    model = _small(shared_filters=shared_filters, dipole=True).eval()
    old = dict(model.state_dict())
    w, b = old.pop("filters.weight"), old.pop("filters.bias")
    for k in range(len(model.interactions)):
        part = slice(0, None) if shared_filters else slice(k * model.filter_width,
                                                           (k + 1) * model.filter_width)
        old[f"interactions.{k}.message.filter.weight"] = w[part].clone()
        old[f"interactions.{k}.message.filter.bias"] = b[part].clone()
    fresh = _small(seed=1, shared_filters=shared_filters, dipole=True).eval()
    fresh.load_state_dict(old)
    g = _graph()
    assert torch.equal(fresh(g)["energy"], model(g)["energy"])


@pytest.mark.parametrize("radial_basis", ["gaussian", "bessel"])
def test_float64_basis_exact_after_cast_and_load(radial_basis):
    """A float32 build cast to float64, or a float32 checkpoint, keeps the exact
    Gaussian centers and Bessel frequencies n pi."""
    name, exact = (("centers", torch.linspace(0.0, 5.0, 20, dtype=torch.float64))
                   if radial_basis == "gaussian"
                   else ("freqs", math.pi * torch.arange(1, 21, dtype=torch.float64)))
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        m32 = PaiNN(radial_basis=radial_basis)
    finally:
        torch.set_default_dtype(old)
    assert torch.equal(getattr(m32.double().rbf, name), exact)
    m64 = PaiNN(radial_basis=radial_basis)
    m64.load_state_dict(PaiNN(radial_basis=radial_basis).float().state_dict())
    assert torch.equal(getattr(m64.rbf, name), exact)


# the paper's equations, atom by atom

def _reference(model, Z, pos, total_charge=0.0):
    """Loop implementation of eq 7-10, the readout, Fig. 3 and eq 13-14."""
    F, N, rc = model.n_features, len(Z), model.cutoff
    Z = torch.as_tensor(Z)
    pos = torch.as_tensor(pos, dtype=torch.float64)
    s = model.embedding.weight[Z].clone()
    v = torch.zeros(N, 3, F)
    for k, block in enumerate(model.interactions):
        msg, upd = block.message, block.update
        first = (0 if model.shared_filters else k) * model.filter_width
        ds, dv = torch.zeros(N, F), torch.zeros(N, 3, F)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                rij = pos[j] - pos[i]
                d = float(rij.norm())
                if d >= rc:
                    continue
                e = rij / d
                rbf = model.rbf(torch.tensor([d]))[0]
                fcut = 0.5 * (math.cos(math.pi * d / rc) + 1.0)
                W = _lin(model.filters, rbf)[first:first + model.filter_width] * fcut
                phi = _lin(msg.phi[2], _silu(_lin(msg.phi[0], s[j])))
                x = phi * W
                ds[i] += x[:F]                                   # eq 7
                if msg.vectors:                                  # eq 8
                    x_vs = x[-F:]
                    dv[i] += x_vs[None, :] * e[:, None]
                    if msg.vector_propagation:
                        dv[i] += v[j] * x[F:2 * F][None, :]
        s, v = s + ds, v + dv
        ds, dv = torch.zeros(N, F), torch.zeros(N, 3, F)
        for i in range(N):
            if not upd.vectors:
                ds[i] = _lin(upd.net[2], _silu(_lin(upd.net[0], s[i])))
                continue
            Wu, Wv = upd.lin_v.weight[:F], upd.lin_v.weight[F:]
            uv, vv = v[i] @ Wu.T, v[i] @ Wv.T                 # (3, F)
            norm = torch.sqrt((vv ** 2).sum(0) + upd.epsilon)
            a = _lin(upd.net[2], _silu(_lin(upd.net[0], torch.cat([s[i], norm]))))
            dv[i] = a[:F][None, :] * uv                          # eq 10
            ds[i] = a[-F:]                                       # eq 9
            if upd.scalar_product:
                ds[i] += a[F:2 * F] * (uv * vv).sum(0)
        s, v = s + ds, v + dv
    node_energy = torch.stack([
        float(model.energy_scale) * _lin(model.readout[2], _silu(_lin(model.readout[0], s[i])))[0]
        + float(model.energy_shift) + model.atom_ref.weight[Z[i], 0] for i in range(N)])
    out = {"energy": node_energy.sum(), "node_energy": node_energy, "s": s, "v": v}

    def gated(head, si, vi):
        for blk in head.blocks:
            w = blk.lin_v.weight
            nv = blk.n_out_v
            v1, v2 = vi @ w[:nv].T, vi @ w[nv:].T
            x = _lin(blk.net[2], _silu(_lin(blk.net[0], torch.cat([si, v1.norm(dim=0)]))))
            si, gate = x[:blk.n_out_s], x[blk.n_out_s:]
            vi = gate[None, :] * v2
            if blk.scalar_activation:
                si = _silu(si)
        return si[0], vi[:, 0]

    m = torch.tensor(ATOMIC_MASSES)[Z][:, None]
    r = pos - (m * pos).sum(0) / m.sum()
    def atomwise(net, si):
        for layer in net:
            si = _lin(layer, si) if isinstance(layer, torch.nn.Linear) else _silu(si)
        return si[0]

    if model.dipole_head is not None:
        if model.atomic_dipoles:                                 # eq 13
            q, mu = zip(*[gated(model.dipole_head, s[i], v[i]) for i in range(N)])
            q, mu = torch.stack(q), torch.stack(mu)
        else:                                                    # eq 12
            q, mu = torch.stack([atomwise(model.dipole_head, s[i]) for i in range(N)]), 0.0
        if model.correct_charges:
            q = q + (total_charge - q.sum()) / N
        out["charges"] = q
        out["dipole"] = (mu + q[:, None] * r).sum(0)
    if model.polarizability_head is not None:
        a0, nu = zip(*[gated(model.polarizability_head, s[i], v[i]) for i in range(N)])
        a0, nu = torch.stack(a0), torch.stack(nu)
        alpha = torch.zeros(3, 3)
        for i in range(N):
            alpha += a0[i] * torch.eye(3) + torch.outer(nu[i], r[i]) + torch.outer(r[i], nu[i])
        out["polarizability"] = alpha
    return out


@pytest.mark.parametrize("ablation", [{}, {"scalar_product": False},
                                      {"vector_propagation": False},
                                      {"scalar_product": False, "vector_propagation": False},
                                      {"vector_features": False},
                                      {"atomic_dipoles": True},
                                      {"radial_basis": "bessel"}, {"radial_basis": "expnorm"},
                                      {"shared_filters": True}, {"shared_interactions": True},
                                      {"shared_filters": True, "shared_interactions": True}])
def test_forward_matches_paper_equations(ablation):
    model = _small(dipole=True, polarizability=True,
                   energy_shift=-0.3, energy_scale=1.7, **ablation)
    model.set_atomic_energies(SPECIES, [-0.5, -1.0, -2.0])
    s = _structure(7, seed=3)
    out = model(structure_to_graph(s, 5.0))
    ref = _reference(model, s["atomic_numbers"], s["pos"])
    assert torch.allclose(out["node_energy"], ref["node_energy"], atol=1e-12)
    assert torch.allclose(out["node_features"], ref["s"], atol=1e-12)
    assert torch.allclose(out["node_vectors"], ref["v"], atol=1e-12)
    assert torch.allclose(out["charges"], ref["charges"], atol=1e-12)
    assert torch.allclose(out["dipole"][0], ref["dipole"], atol=1e-12)
    assert torch.allclose(out["polarizability"][0], ref["polarizability"], atol=1e-12)


def test_edge_direction_is_the_paper_convention():
    """The message uses r_ij = r_j - r_i (the negative of the xnn edge
    vector): the vector feature of an atom with one neighbor points along
    the bond toward that neighbor for a filter that is positive there."""
    model = _small(n_interactions=1, n_rbf=4)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        msg = model.interactions[0].message
        model.filters.bias[2 * 16:].fill_(1.0)   # W_vs = 1 (times the cutoff)
        msg.phi[2].bias[2 * 16:].fill_(1.0)      # phi_vs = 1
        model.interactions[0].update.lin_v.weight.zero_()
    g = structure_to_graph({"pos": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                            "atomic_numbers": [1, 1]}, 5.0)
    v = model(g)["node_vectors"]
    fcut = 0.5 * (math.cos(math.pi / 5.0) + 1)
    assert torch.allclose(v[0, :, 0], torch.tensor([fcut, 0.0, 0.0]))
    assert torch.allclose(v[1, :, 0], torch.tensor([-fcut, 0.0, 0.0]))


def test_gated_block_equivariance():
    torch.manual_seed(0)
    block = GatedEquivariantBlock(8, 8, 4, 4)
    s, v = torch.randn(5, 8), torch.randn(5, 3, 8)
    R = torch.linalg.qr(torch.randn(3, 3))[0]
    s1, v1 = block(s, v)
    s2, v2 = block(s, torch.einsum("ab,nbf->naf", R, v))
    assert torch.allclose(s1, s2, atol=1e-12)
    assert torch.allclose(torch.einsum("ab,nbf->naf", R, v1), v2, atol=1e-12)


# physical invariances and equivariances

def _random_rotation(seed, proper=True):
    Q = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(seed)))[0]
    if (torch.det(Q) > 0) != proper:
        Q = Q @ torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
    return Q


@pytest.mark.parametrize("proper", [True, False])
def test_energy_invariance_forces_equivariance(proper):
    """E(3) invariant energies (the vector features are polar vectors, so
    reflections are covered too) and equivariant forces."""
    model = ForceStressOutput(_small()).eval()
    s = _structure(7, seed=1)
    R, t = _random_rotation(4, proper), torch.tensor([1.3, -0.7, 2.1])
    out = model(structure_to_graph(s, 5.0))
    pos2 = torch.as_tensor(s["pos"]) @ R.T + t
    out2 = model(structure_to_graph({"pos": pos2, "atomic_numbers": s["atomic_numbers"]}, 5.0))
    assert abs(float(out["energy"] - out2["energy"])) < 1e-10
    assert torch.allclose(out["forces"] @ R.T, out2["forces"], atol=1e-10)


@pytest.mark.parametrize("proper", [True, False])
def test_dipole_and_polarizability_equivariance(proper):
    """mu(R x + t) = R mu(x), alpha(R x + t) = R alpha R^T (eq 13, 14):
    translation invariance comes from the center-of-mass frame and the
    charge correction, the rotation law from the vector features."""
    model = _small(dipole=True, polarizability=True).eval()
    s = _structure(7, seed=2)
    R, t = _random_rotation(5, proper), torch.tensor([-2.0, 0.4, 1.1])
    out = model(structure_to_graph(s, 5.0))
    pos2 = torch.as_tensor(s["pos"]) @ R.T + t
    out2 = model(structure_to_graph({"pos": pos2, "atomic_numbers": s["atomic_numbers"]}, 5.0))
    assert torch.allclose(out["charges"], out2["charges"], atol=1e-10)
    assert torch.allclose(out["dipole"][0] @ R.T, out2["dipole"][0], atol=1e-10)
    assert torch.allclose(R @ out["polarizability"][0] @ R.T, out2["polarizability"][0], atol=1e-10)
    a = out["polarizability"][0]
    assert torch.allclose(a, a.T, atol=1e-12)
    assert abs(float(out["charges"].sum())) < 1e-12
    assert abs(float(out["dipole"].norm())) > 1e-6


def test_charges_sum_to_the_total_charge():
    model = _small(dipole=True).eval()
    s = {**_structure(5, seed=6), "total_charge": -1.0}
    out = model(structure_to_graph(s, 5.0))
    assert float(out["charges"].sum()) == pytest.approx(-1.0, abs=1e-12)
    free = _small(dipole=True, correct_charges=False).eval()
    assert abs(float(free(structure_to_graph(s, 5.0))["charges"].sum()) + 1.0) > 1e-6


def test_permutation_invariance():
    model = _small(dipole=True, polarizability=True).eval()
    s = _structure(7, seed=2)
    perm = np.random.default_rng(1).permutation(7)
    out = model(structure_to_graph(s, 5.0))
    out2 = model(structure_to_graph({"pos": s["pos"][perm],
                                     "atomic_numbers": np.array(s["atomic_numbers"])[perm]}, 5.0))
    assert abs(float(out["energy"] - out2["energy"])) < 1e-12
    assert torch.allclose(out["node_energy"][perm], out2["node_energy"], atol=1e-12)
    assert torch.allclose(out["dipole"], out2["dipole"], atol=1e-12)
    assert torch.allclose(out["polarizability"], out2["polarizability"], atol=1e-12)


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


def test_pes_smooth_across_cutoff():
    """The cosine cutoff is C^1 at r_cut, so the energy and the forces are
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
    """An atom without neighbors keeps v = 0; the stabilized norm keeps the
    gradient finite (a plain norm has none at zero)."""
    model = ForceStressOutput(_small(dipole=True, polarizability=True))
    g = structure_to_graph({"pos": [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
                            "atomic_numbers": [8, 1]}, 5.0)
    out = model(g)
    loss = out["energy"].sum() + out["dipole"].sum() + out["polarizability"].sum()
    loss.backward()
    assert torch.isfinite(out["forces"]).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_directional_information_beyond_distances():
    """A hexagon and two triangles with the same bond length and a cutoff
    below the second-neighbor distance give every atom the same neighbor
    distances, but different angles: the vector features of PaiNN resolve
    them in a single message pass (paper section III.C, Table I)."""
    bond, cutoff = 1.5, 2.0
    hexagon = np.array([[bond * math.cos(k * math.pi / 3), bond * math.sin(k * math.pi / 3), 0.0]
                        for k in range(6)])
    tri = np.array([[0.0, 0.0, 0.0], [bond, 0.0, 0.0],
                    [bond / 2, bond * math.sqrt(3) / 2, 0.0]])
    triangles = np.concatenate([tri, tri + np.array([10.0, 0.0, 0.0])])
    model = _small(n_interactions=1, cutoff=cutoff).eval()
    e_hex = float(model(structure_to_graph({"pos": hexagon, "atomic_numbers": [6] * 6}, cutoff))["energy"])
    e_tri = float(model(structure_to_graph({"pos": triangles, "atomic_numbers": [6] * 6}, cutoff))["energy"])
    assert abs(e_hex - e_tri) > 1e-8          # orders of magnitude above float64 round-off
    # an invariant model of the same neighbor distances cannot
    from xnn.gnn.models.schnet import SchNet
    torch.manual_seed(0)
    inv = SchNet(n_features=16, n_interactions=2, n_rbf=10, cutoff=cutoff, cutoff_fn="cosine").eval()
    e_hex = float(inv(structure_to_graph({"pos": hexagon, "atomic_numbers": [6] * 6}, cutoff))["energy"])
    e_tri = float(inv(structure_to_graph({"pos": triangles, "atomic_numbers": [6] * 6}, cutoff))["energy"])
    assert abs(e_hex - e_tri) < 1e-12


def test_size_extensivity_and_batching():
    model = _small(dipole=True, polarizability=True).eval()
    graphs = [structure_to_graph(_structure(n, seed=n), 5.0) for n in (4, 6, 9)]
    singles = [model(g) for g in graphs]
    out = model(collate(graphs))
    for b, one in enumerate(singles):
        assert abs(float(out["energy"][b] - one["energy"][0])) < 1e-12
        assert torch.allclose(out["dipole"][b], one["dipole"][0], atol=1e-12)
        assert torch.allclose(out["polarizability"][b], one["polarizability"][0], atol=1e-12)
    # two far-apart copies: twice the energy
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
    # with a cell shorter than the cutoff, periodic images are distinct neighbors
    g_iso = _graph(n=6, periodic=False)
    assert abs(float(out["energy"] - model(g_iso)["energy"])) > 1e-6


def test_float32_forward_matches_float64():
    model = _small(dipole=True, polarizability=True).eval()
    g = _graph(n=8)
    out64 = model(g)
    model32 = model.float()
    g32 = structure_to_graph({"pos": g.pos.float(), "atomic_numbers": g.atomic_numbers}, 5.0)
    out32 = model32(g32)
    assert abs(float(out64["energy"] - out32["energy"].double())) < 1e-4 * max(1.0, abs(float(out64["energy"])))
    assert torch.allclose(out64["dipole"], out32["dipole"].double(), atol=1e-4)


# deployment

def test_scriptable_and_lammps_export(tmp_path):
    from xnn.common.deploy import export_to_lammps

    model = _small().eval()
    g = _graph(n=6, periodic=True)
    scripted = torch.jit.script(model)
    d = (scripted.node_energy(g.atomic_numbers, g.edge_index, g.edge_vectors())
         - model.node_energy(g.atomic_numbers, g.edge_index, g.edge_vectors()))
    assert d.abs().max() < 1e-12
    s, v = scripted.representation(g.atomic_numbers, g.edge_index, g.edge_vectors())
    assert s.shape == (6, 16) and v.shape == (6, 3, 16)

    path = str(tmp_path / "painn_lammps.pt")
    export_to_lammps(model, 5.0, path)
    loaded = torch.jit.load(path)
    out = loaded(g.pos, g.edge_index, g.cell_shifts, g.atomic_numbers, g.cell[0])
    ref = ForceStressOutput(model)(g)
    assert abs(float(out["total_energy"]) - float(ref["energy"])) < 1e-5
    assert (out["forces"] - ref["forces"].detach()).abs().max() < 1e-5


def test_les_wrapper_reads_node_features():
    from xnn.common.models.les import LatentEwald

    model = LatentEwald(_small()).double()
    out = model(_graph())
    assert out["energy"].shape == (1,) and torch.isfinite(out["energy"]).all()


def test_multihead_and_reference_energies():
    from xnn.common.finetune import MultiHead, get_atomic_energies, set_atomic_energies

    model = _small(dipole=True, polarizability=True)
    set_atomic_energies(model, {1: -0.5, 6: -1.0, 8: -2.0})
    assert get_atomic_energies(model) == {1: -0.5, 6: -1.0, 8: -2.0}
    multi = MultiHead(model, ["a", "b"])
    out = multi(_graph())
    assert out["energy"].shape == (1,) and out["polarizability"].shape == (1, 3, 3)


def test_ase_calculator_reports_dipole():
    pytest.importorskip("ase")
    from ase import Atoms

    from xnn.common.deploy import XNNCalculator

    model = ForceStressOutput(_small(dipole=True)).eval()
    atoms = Atoms(numbers=[8, 1, 1], positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    atoms.calc = XNNCalculator(model, cutoff=5.0)
    mu = atoms.get_dipole_moment()
    assert mu.shape == (3,) and np.isfinite(atoms.get_potential_energy())


# training on tensorial labels

def test_train_step_with_dipole_and_polarizability_targets(tmp_path):
    from xnn.common.config import Config
    from xnn.common.data import AtomicDataset
    from xnn.common.train import Trainer

    rng = np.random.default_rng(0)
    structures = []
    for i in range(6):
        s = _structure(5, seed=i)
        s.update(energy=float(rng.normal()), forces=rng.normal(size=(5, 3)),
                 dipole=rng.normal(size=3), polarizability=rng.normal(size=(3, 3)))
        structures.append(s)
    cfg = from_dict({"model": {"name": "painn", "n_features": 8, "n_interactions": 1,
                               "n_rbf": 4, "cutoff": 5.0,
                               "extra": {"dipole": True, "polarizability": True}},
                     "optim": {"epochs": 2, "dipole_weight": 1.0, "polarizability_weight": 1.0},
                     "data": {"batch_size": 3, "val_fraction": 0.0},
                     "output_dir": str(tmp_path)})
    assert isinstance(cfg, Config)
    trainer = Trainer(cfg, AtomicDataset(structures, 5.0))
    logs = trainer.fit()["train"]
    assert {"energy_mse", "force_mse", "dipole_mse", "polarizability_mse"} <= set(logs)


# config

def test_from_config_extras():
    cfg = from_dict({"model": {
        "name": "painn", "cutoff": 4.0, "n_features": 16, "n_interactions": 2, "n_rbf": 6,
        "extra": {"radial_basis": "bessel", "shared_filters": True, "atomic_dipoles": True,
                  "dipole": True, "polarizability": True, "scalar_product": False,
                  "energy_shift": -1.5, "energy_scale": 0.2, "species": SPECIES,
                  "atomic_energies": [-13.6, -1030.0, -2043.0]}}})
    m = build_model(cfg.model)
    assert m.rbf.freqs.shape == (6,) and m.cutoff == 4.0
    assert m.filters.weight.shape[0] == m.filter_width
    assert m.atomic_dipoles
    assert m.dipole_head is not None and m.polarizability_head is not None
    assert not m.interactions[0].update.scalar_product
    assert float(m.energy_shift) == -1.5 and float(m.energy_scale) == pytest.approx(0.2)
    assert float(m.atom_ref.weight[6, 0]) == pytest.approx(-1030.0)
    out = m(_graph())
    assert out["dipole"].shape == (1, 3)


def test_model_option_override():
    """A dotted override of a model option lands in the model's options."""
    from xnn.common.config import apply_overrides
    cfg = apply_overrides(from_dict({"model": {"name": "painn", "n_features": 8}}),
                          ["model.radial_basis='bessel'"])
    assert cfg.model.extra["radial_basis"] == "bessel"
    assert hasattr(build_model(cfg.model).rbf, "freqs")


def test_reference_key_translation():
    cfg = from_dict({"model": {"name": "painn", "cutoff": 4.5, "n_atom_basis": 48,
                               "n_radial_basis": 30}})
    assert cfg.model.n_features == 48 and cfg.model.n_rbf == 30
    assert cfg.model.cutoff == 4.5 and cfg.data.cutoff == 4.5


# fidelity to the reference implementation (optional)

def _fidelity(dtype):
    pytest.importorskip("schnetpack")
    import subprocess
    import sys
    script = os.path.join(os.path.dirname(__file__), "painn_fidelity.py")
    res = subprocess.run([sys.executable, script, dtype], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    return float(res.stdout.strip().splitlines()[-1])


def test_fidelity_vs_reference_float64():
    """Transplanted weights give the reference energies, forces, charges,
    dipoles and polarizabilities at the float64 floor (a layout or sign
    convention error would show as O(1))."""
    assert _fidelity("float64") < 1e-12


def test_fidelity_vs_reference_float32():
    """The same in float32, at float32 round-off."""
    assert _fidelity("float32") < 1e-5


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gather_rows_is_indexing(device):
    """The message gather: the values and both derivatives of ``src[index]``, scriptable."""
    from xnn.common.models.ops import gather_rows

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    gen = torch.Generator().manual_seed(0)
    src = torch.randn(7, 3, 4, dtype=torch.float64, generator=gen).to(device).requires_grad_(True)
    index = torch.randint(0, 7, (20,), generator=gen).to(device)
    assert torch.equal(gather_rows(src, index), src[index])
    probe = torch.randn(20, 3, 4, dtype=torch.float64, generator=gen).to(device)
    a, = torch.autograd.grad((gather_rows(src, index) * probe).sum(), src)
    b, = torch.autograd.grad((src[index] * probe).sum(), src)
    assert torch.allclose(a, b, rtol=0, atol=1e-14)
    assert torch.autograd.gradgradcheck(lambda x: gather_rows(x, index) ** 2, (src,))
    assert torch.equal(torch.jit.script(gather_rows)(src, index), src[index])


def test_fast_path_eligibility(monkeypatch):
    """The fused message block needs all three terms and a power-of-two width."""
    import xnn.gnn.fast.painn as fast
    from xnn.common.finetune import inject_lora
    monkeypatch.setattr(fast, "supported", lambda device, dtype: True)
    cuda = torch.device("cuda")
    m = _small()
    assert m.interactions[0].message.fast_supported(cuda, torch.float32)
    assert not _small(n_features=24).interactions[0].message.fast_supported(cuda, torch.float32)
    assert not _small(vector_propagation=False).interactions[0].message.fast_supported(
        cuda, torch.float32)
    # phi and the filters stay outside the kernels, adapters included
    inject_lora(m, rank=2)
    assert m.interactions[0].message.fast_supported(cuda, torch.float32)


def _fast_batch(dtype):
    graphs = [_graph(n=14, seed=1, periodic=True), _graph(n=9, seed=2)]
    batch = collate(graphs).to("cuda")
    batch.pos = batch.pos.to(dtype)
    batch.cell = batch.cell.to(dtype)
    return batch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_fast_message_gradients():
    """Every first derivative of the fused message block (finite differences)."""
    pytest.importorskip("triton")
    from xnn.gnn.fast.painn_kernels import MessageFunction
    gen = torch.Generator().manual_seed(0)
    n, n_edges, F = 5, 12, 16

    def rand(*shape):
        return torch.randn(*shape, dtype=torch.float64, generator=gen).cuda().requires_grad_(True)

    x, v, direction = rand(n, 3 * F), rand(n, 3, F), rand(n_edges, 3)
    stacked = rand(n_edges, 2 * 3 * F)                  # one block's columns of stacked filters
    edge_index = torch.randint(0, n, (2, n_edges), generator=gen).cuda()

    def f(x, stacked, direction, v):
        W = stacked[:, 3 * F:]
        return MessageFunction.apply(x, W, direction, v, edge_index[0], edge_index[1])

    # atomic adds sum in a run-dependent order
    assert torch.autograd.gradcheck(f, (x, stacked, direction, v), nondet_tol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shared", [False, True])
def test_fast_path_parity(dtype, shared):
    """The fused message blocks give the reference energies, forces, stress,
    charges, dipoles and polarizabilities."""
    pytest.importorskip("triton")
    model = _small(n_features=32, n_interactions=3, shared_interactions=shared,
                   shared_filters=shared, dipole=True, polarizability=True, atomic_dipoles=True)
    model = model.to("cuda", dtype).eval()
    wrapped = ForceStressOutput(model, compute_stress=True).eval()
    keys = ("energy", "forces", "stress", "charges", "dipole", "polarizability")
    out = {}
    for use_fast in (False, True):
        model.set_use_fast(use_fast)
        res = wrapped(_fast_batch(dtype))
        assert all(b.message.fast_active == use_fast for b in model.interactions)
        out[use_fast] = {k: res[k].detach().double() for k in keys}
    tol = 1e-12 if dtype == torch.float64 else 1e-5
    for key, ref in out[False].items():
        err = (out[True][key] - ref).abs().max() / ref.abs().max()
        assert err < tol, (key, float(err))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_fast_path_off_in_training():
    """A training-mode evaluation (force training) runs the reference message blocks."""
    pytest.importorskip("triton")
    model = _small().to("cuda").train()
    model.set_use_fast(True)
    out = ForceStressOutput(model).train()(_fast_batch(torch.float64))
    assert not any(b.message.fast_active for b in model.interactions)
    out["forces"].pow(2).sum().backward()
    assert model.filters.weight.grad is not None


@pytest.fixture
def interpreted(monkeypatch):
    """The fused kernels under Triton's interpreter (chosen by conftest.py on machines without
    CUDA), on CPU tensors, with the fast path offered on CPUs."""
    if os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("runs where Triton's interpreter is on (no CUDA device)")
    pytest.importorskip("triton")
    import xnn.gnn.fast.painn as fast
    from xnn.gnn.fast import painn_kernels
    monkeypatch.setattr(fast, "supported", lambda device, dtype: True)
    return painn_kernels


def test_fast_support_needs_cuda():
    from xnn.gnn.fast.painn import supported
    assert not supported(torch.device("cpu"), torch.float64)


def test_fast_message_gradients_interpreted(interpreted):
    """The fused message block and its first derivatives, run by the Triton interpreter."""
    gen = torch.Generator().manual_seed(1)
    n, n_edges, F = 4, 9, 16

    def rand(*shape):
        return torch.randn(*shape, dtype=torch.float64, generator=gen).requires_grad_(True)

    x, v, direction = rand(n, 3 * F), rand(n, 3, F), rand(n_edges, 3)
    stacked = rand(n_edges, 2 * 3 * F)
    edge_index = torch.randint(0, n, (2, n_edges), generator=gen)

    def f(x, stacked, direction, v):
        return interpreted.message(x, stacked[:, 3 * F:], direction, v, edge_index)

    assert torch.autograd.gradcheck(f, (x, stacked, direction, v), nondet_tol=1e-12, fast_mode=True)
    assert interpreted.message(x, stacked[:0, 3 * F:], direction[:0], v,
                               edge_index[:, :0])[1].abs().max() == 0


def test_fast_path_parity_interpreted(interpreted):
    """The whole model with the fused message blocks on a CPU (the interpreter) and the reference."""
    model = _small(dipole=True, polarizability=True, atomic_dipoles=True).eval()
    wrapped = ForceStressOutput(model, compute_stress=True).eval()
    graph = collate([_graph(n=8, seed=1, periodic=True), _graph(n=5, seed=2)])
    keys = ("energy", "forces", "stress", "charges", "dipole", "polarizability")
    out = {}
    for use_fast in (False, True):
        model.set_use_fast(use_fast)
        res = wrapped(graph)
        assert all(b.message.fast_active == use_fast for b in model.interactions)
        out[use_fast] = {k: res[k].detach() for k in keys}
    for key, ref in out[False].items():
        assert (out[True][key] - ref).abs().max() <= 1e-12 * ref.abs().max(), key
