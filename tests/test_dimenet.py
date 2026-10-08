"""DimeNet / DimeNet++ tests: manuscript fidelity (ICLR 2020, NeurIPS-W 2020),
the bases, the directed triplets, physical invariances, smoothness,
TorchScript deployment, config handling and (optionally) parity with the
authors' TensorFlow code.

The centerpiece is ``_reference_energy``: an independent, loop-based
implementation of the papers' equations (eq 6-9 and the blocks of Fig. 4 /
Fig. 1) with closed-form spherical Bessel functions and Legendre
polynomials, reusing only the model's *parameters*. Given the same weights
the model must reproduce it to float64 precision.
"""
import math
import os

import numpy as np
import pytest
import torch

from xnn.common.config import from_dict
from xnn.common.data import collate, structure_to_graph
from xnn.common.models import ForceStressOutput, available_models, build_model
from xnn.gnn.featurizers import (BesselRBF, PolynomialCutoff, SphericalBesselBasis,
                                 spherical_bessel_jn, spherical_bessel_zeros,
                                 zonal_harmonics)
from xnn.gnn.models.dimenet import DimeNet, DimeNetPP, directed_triplets

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


def _small(pp=False, seed=0, **kw):
    torch.manual_seed(seed)
    kw.setdefault("n_features", 16)
    kw.setdefault("n_interactions", 2)
    kw.setdefault("n_rbf", 4)
    kw.setdefault("n_spherical", 3)
    kw.setdefault("cutoff", 5.0)
    kw.setdefault("n_output_layers", 2)
    if pp:
        kw.setdefault("n_triplet_features", 8)
        kw.setdefault("n_basis_features", 4)
        kw.setdefault("n_output_features", 12)
        return DimeNetPP(**kw)
    kw.setdefault("n_bilinear", 4)
    return DimeNet(**kw)


# registry / defaults

def test_registered():
    assert "dimenet" in available_models()
    assert "dimenet++" in available_models()


def test_paper_defaults():
    """DimeNet() is the ICLR 2020 architecture (Appendix B) and DimeNetPP()
    the NeurIPS-W 2020 one (Table 1)."""
    m = DimeNet()
    assert m.embedding.embedding.weight.shape[1] == 128
    assert len(m.interactions) == 6 and len(m.outputs) == 7
    assert m.rbf.freqs.shape == (6,) and isinstance(m.rbf.freqs, torch.nn.Parameter)
    assert m.sbf.zeros.shape == (7, 6) and m.n_sbf == 42
    assert m.interactions[0].bilinear.shape == (8, 128, 128)
    assert m.envelope.p == 6 and m.cutoff == 5.0
    assert len(m.interactions[0].before_skip) == 1
    assert len(m.interactions[0].after_skip) == 2
    assert len(m.outputs[0].dense) == 3 and m.outputs[0].up is None
    assert isinstance(m.embedding.act, torch.nn.SiLU)
    pp = DimeNetPP()
    assert len(pp.interactions) == 4 and len(pp.outputs) == 5
    assert pp.interactions[0].lin_down.weight.shape == (64, 128)
    assert pp.interactions[0].lin_rbf1.weight.shape == (8, 6)
    assert pp.interactions[0].lin_sbf2.weight.shape == (64, 8)
    assert pp.outputs[0].up.weight.shape == (256, 128)
    assert pp.node_feature_dim == 256
    assert sum(p.numel() for p in pp.parameters()) < sum(p.numel() for p in m.parameters())


def test_invalid_options():
    with pytest.raises(ValueError):
        DimeNet(n_features=8, output_init="ones")
    with pytest.raises(ValueError):
        DimeNet(n_features=8, atomic_energies=[1.0])


# the bases (paper section 5)

def _jl_closed(l, x):
    """j_0, j_1, j_2 in closed form (numpy)."""
    if l == 0:
        return np.sin(x) / x
    if l == 1:
        return np.sin(x) / x ** 2 - np.cos(x) / x
    return (3 / x ** 3 - 1 / x) * np.sin(x) - 3 * np.cos(x) / x ** 2


def test_spherical_bessel_functions():
    """j_l from the series/recurrence against the closed forms (l <= 2) and
    scipy (higher orders), including the small-argument regime."""
    x = torch.linspace(0.05, 30.0, 601, dtype=torch.float64)
    j = spherical_bessel_jn(8, x)
    # the closed forms cancel catastrophically at small x, compare them beyond 2
    large = x.numpy() >= 2.0
    for l in range(3):
        assert np.allclose(j[large, l].numpy(), _jl_closed(l, x.numpy()[large]),
                           atol=1e-15, rtol=1e-12)
    scipy_special = pytest.importorskip("scipy.special")
    for l in range(9):
        ref = scipy_special.spherical_jn(l, x.numpy())
        assert np.allclose(j[:, l].numpy(), ref, atol=1e-16, rtol=1e-11), l
    # the origin: j_0(0) = 1, the others vanish, nothing is NaN
    j0 = spherical_bessel_jn(6, torch.zeros(1, dtype=torch.float64))[0]
    assert torch.allclose(j0, torch.tensor([1.0] + [0.0] * 6, dtype=torch.float64))
    # float32 input: computed in float64, returned correctly rounded
    j32 = spherical_bessel_jn(6, x.float())
    assert j32.dtype == torch.float32
    assert torch.allclose(j32.double(), spherical_bessel_jn(6, x)[:, :7], atol=1e-6)


def test_spherical_bessel_zeros():
    """z_0n = n pi, j_l(z_ln) = 0, zeros increase along n and interlace in l."""
    z = spherical_bessel_zeros(6, 5)
    assert z.shape == (7, 5)
    assert np.allclose(z[0], np.arange(1, 6) * np.pi, atol=1e-14)
    assert abs(z[1, 0] - 4.493409457909064) < 1e-12        # first zero of j_1
    assert abs(z[2, 0] - 5.763459196894550) < 1e-12        # first zero of j_2
    for l in range(7):
        assert np.all(np.diff(z[l]) > 0)
        residual = spherical_bessel_jn(l, torch.tensor(z[l]))[:, l].abs().max()
        assert float(residual) < 1e-13
        if l > 0:
            assert np.all(z[l - 1] < z[l]) and np.all(z[l, :-1] < z[l - 1, 1:])


def test_zonal_harmonics():
    """Y_l^0 = sqrt((2l+1)/4pi) P_l(cos alpha), Legendre polynomials explicit."""
    c = torch.linspace(-1, 1, 41, dtype=torch.float64)
    y = zonal_harmonics(3, c)
    p = [torch.ones_like(c), c, (3 * c ** 2 - 1) / 2, (5 * c ** 3 - 3 * c) / 2]
    for l in range(4):
        ref = math.sqrt((2 * l + 1) / (4 * math.pi)) * p[l]
        assert torch.allclose(y[:, l], ref, atol=1e-14)
    # parity: Y_l(pi - alpha) = (-1)^l Y_l(alpha) (the sign that relates the
    # paper's angle to the reference code's)
    assert torch.allclose(zonal_harmonics(5, -c), zonal_harmonics(5, c)
                          * torch.tensor([1.0, -1, 1, -1, 1, -1], dtype=torch.float64), atol=1e-14)


def test_2d_basis_orthonormal_on_cutoff_sphere():
    """eq 6: a_ln are orthonormal on the sphere of radius c:
    int_0^c int_0^pi int_0^2pi a_ln a_l'n' d^2 sin(alpha) dphi dalpha dd = delta."""
    c = 4.0
    basis = SphericalBesselBasis(n_spherical=4, n_radial=3, cutoff=c)
    xr, wr = np.polynomial.legendre.leggauss(200)        # radial nodes on [0, c]
    d = 0.5 * c * (xr + 1)
    wd = 0.5 * c * wr * d ** 2
    xa, wa = np.polynomial.legendre.leggauss(60)         # cos(alpha) on [-1, 1]
    R = basis.radial(torch.tensor(d)).numpy()             # (200, L, N)
    Y = basis.angular(torch.tensor(xa)).numpy()           # (60, L)
    gram_r = np.einsum("d,dln,dmk->lnmk", wd, R, R)       # radial overlaps
    gram_a = 2 * math.pi * np.einsum("a,al,am->lm", wa, Y, Y)   # angular overlaps
    gram = gram_r * gram_a[:, None, :, None]
    eye = np.eye(12).reshape(4, 3, 4, 3)
    assert np.allclose(gram, eye, atol=1e-6)
    # the flattened layout runs degree-major
    r = torch.tensor([1.3, 2.2])
    ca = torch.tensor([0.4, -0.7])
    full = basis(r, ca)
    assert full.shape == (2, 12)
    assert torch.allclose(full[:, 3 * 2 + 1], basis.radial(r)[:, 2, 1] * basis.angular(ca)[:, 2])


def test_radial_basis_matches_eq7_and_is_orthonormal():
    """e_n(d) = sqrt(2/c) sin(n pi d / c) / d, orthonormal on [0, c] with d^2."""
    c = 5.0
    rbf = BesselRBF(6, c, trainable=True)
    d = torch.tensor([0.5, 1.7, 4.9], dtype=torch.float64)
    n = torch.arange(1, 7, dtype=torch.float64)
    ref = math.sqrt(2 / c) * torch.sin(n * math.pi * d[:, None] / c) / d[:, None]
    assert torch.allclose(rbf(d), ref, atol=1e-14)
    xr, wr = np.polynomial.legendre.leggauss(300)
    dd = 0.5 * c * (xr + 1)
    e = rbf(torch.tensor(dd)).detach().numpy()
    gram = np.einsum("d,dn,dm->nm", 0.5 * c * wr * dd ** 2, e, e)
    assert np.allclose(gram, np.eye(6), atol=1e-8)


def test_envelope_eq8_has_triple_root_at_cutoff():
    """u(d) of eq 8 with p = 6: value, first and second derivative vanish at c
    (twice continuous differentiability), and u(0) = 1."""
    c, p = 5.0, 6
    env = PolynomialCutoff(c, p)
    x = torch.tensor([0.0, 0.3, 0.9, 0.999], dtype=torch.float64) * c
    xs = x / c
    ref = 1 - (p + 1) * (p + 2) / 2 * xs ** p + p * (p + 2) * xs ** (p + 1) - p * (p + 1) / 2 * xs ** (p + 2)
    assert torch.allclose(env(x), ref, atol=1e-14)
    d = torch.tensor(c - 1e-9, dtype=torch.float64, requires_grad=True)
    u = env(d)
    (du,) = torch.autograd.grad(u, d, create_graph=True)
    (d2u,) = torch.autograd.grad(du, d)
    assert abs(float(u)) < 1e-24 and abs(float(du)) < 1e-14 and abs(float(d2u)) < 1e-5
    assert float(env(torch.tensor(c + 1e-9, dtype=torch.float64))) == 0.0


# the directed triplets (paper section 4)

def _brute_triplets(edge_index, edge_vec):
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    vec = edge_vec.numpy()
    pairs = set()
    for e1 in range(len(src)):                       # j -> i
        for e2 in range(len(src)):                   # k -> j
            if dst[e2] != src[e1]:
                continue
            same_image_reverse = (src[e2] == dst[e1]
                                  and np.array_equal(vec[e2], -vec[e1]))
            if not same_image_reverse:
                pairs.add((e2, e1))
    return pairs


@pytest.mark.parametrize("periodic", [False, True])
def test_directed_triplets_vs_brute_force(periodic):
    g = _graph(n=5, cutoff=4.0, periodic=periodic, cell=3.0)
    vec = g.edge_vectors()
    kj, ji = directed_triplets(g.edge_index, vec, g.num_nodes)
    found = set(zip(kj.tolist(), ji.tolist()))
    assert len(found) == len(kj)                     # no duplicates
    assert found == _brute_triplets(g.edge_index, vec)
    if periodic:                                     # self-images exist in a 3 A cell at 4 A
        assert bool((g.edge_index[0] == g.edge_index[1]).any())
    # every k -> j pair is excluded exactly once per message
    assert all(g.edge_index[1][e2] == g.edge_index[0][e1] for e2, e1 in found)


def test_directed_triplets_empty():
    g = structure_to_graph({"pos": [[0, 0, 0], [10, 0, 0]], "atomic_numbers": [1, 1]}, 5.0)
    kj, ji = directed_triplets(g.edge_index, g.edge_vectors(), 2)
    assert kj.numel() == 0 and ji.numel() == 0
    m = _small()
    out = m(g)
    assert out["energy"].shape == (1,) and torch.isfinite(out["energy"]).all()


# manuscript equations

def _reference_energy(model, Z, pos):
    """Loop-based re-implementation of the papers' forward pass (float64).

    Closed-form j_0..j_2 and Y_0^0..Y_2^0 (so the model under test must have
    ``n_spherical = 3``), the model's zeros z_ln and Bessel frequencies, eq 7
    and eq 6 with the envelope of eq 8, eq 9, the interaction block of Fig. 4
    (DimeNet, bilinear) or Fig. 1 of the NeurIPS-W paper (DimeNet++), and the
    output blocks; per-atom energies summed over blocks plus ``atom_ref``.
    """
    act = torch.nn.functional.silu

    def dense(lin, t):
        out = t @ lin.weight.T
        return out + lin.bias if lin.bias is not None else out

    c, p = model.cutoff, model.envelope.p
    freqs = model.rbf.freqs.detach()
    zeros = model.sbf.zeros.detach().numpy()
    pp = isinstance(model, DimeNetPP)
    n = len(Z)
    F = model.embedding.embedding.weight.shape[1]

    def envelope(d):
        x = d / c
        return 1 - (p + 1) * (p + 2) / 2 * x ** p + p * (p + 2) * x ** (p + 1) - p * (p + 1) / 2 * x ** (p + 2)

    def e_rbf(d):                                                   # eq 7 x eq 8
        return math.sqrt(2 / c) * torch.sin(freqs * d / c) / d * envelope(d)

    def a_sbf(d, cos_alpha):                                        # eq 6 x eq 8
        vals = []
        for l in range(3):
            y = math.sqrt((2 * l + 1) / (4 * math.pi)) * [1.0, cos_alpha, (3 * cos_alpha ** 2 - 1) / 2][l]
            for z in zeros[l]:
                jl1 = _jl_closed(l + 1, z) if l < 2 else float(
                    spherical_bessel_jn(3, torch.tensor(z, dtype=torch.float64))[3])
                norm = math.sqrt(2 / (c ** 3 * jl1 ** 2))
                radial = norm * _jl_closed(l, z * float(d) / c)
                if model.reference_basis:
                    vals.append(radial * y * float(envelope(d)) * c / float(d))
                else:
                    vals.append(radial * y * float(envelope(d)))
        return torch.tensor(vals, dtype=torch.float64)

    edges = [(j, i) for i in range(n) for j in range(n) if i != j]  # all pairs: complete graph
    dist = {(j, i): torch.linalg.norm(pos[i] - pos[j]) for (j, i) in edges}
    emb = model.embedding
    h = emb.embedding.weight[Z]
    m = {}
    for (j, i) in edges:                                            # eq 9
        e = act(dense(emb.lin_rbf, e_rbf(dist[(j, i)])))
        m[(j, i)] = act(dense(emb.lin, torch.cat([h[j], h[i], e])))

    def output(block, m):
        hi = torch.zeros(n, F, dtype=torch.float64)
        for (j, i) in edges:
            hi[i] = hi[i] + m[(j, i)] * dense(block.lin_rbf, e_rbf(dist[(j, i)]))
        if block.up is not None:
            hi = dense(block.up, hi)
        for lin in block.dense:
            hi = act(dense(lin, hi))
        return dense(block.final, hi).squeeze(-1)

    node_e = model.atom_ref.weight[Z, 0] + output(model.outputs[0], m)
    for block, out_block in zip(model.interactions, model.outputs[1:]):
        new_m = {}
        for (j, i) in edges:
            mji = m[(j, i)]
            x_ji = act(dense(block.lin_ji, mji))
            agg = torch.zeros(block.lin_down.out_features if pp else F, dtype=torch.float64)
            for k in range(n):                                      # k in N_j \ {i}
                if k == j or k == i:
                    continue
                v_ji, v_jk = pos[i] - pos[j], pos[k] - pos[j]
                cos_alpha = float(v_ji @ v_jk / (torch.linalg.norm(v_ji) * torch.linalg.norm(v_jk)))
                sbf = a_sbf(dist[(k, j)], cos_alpha)
                # the radial gate: e_RBF(d_kj) (reference code) or e_RBF(d_ji) (eq 4)
                rbf_gate = e_rbf(dist[(k, j)] if model.radial_gate == "kj" else dist[(j, i)])
                if pp:
                    x_kj = act(dense(block.lin_kj, m[(k, j)])) * dense(
                        block.lin_rbf2, dense(block.lin_rbf1, rbf_gate))
                    x_kj = act(dense(block.lin_down, x_kj))
                    agg = agg + x_kj * dense(block.lin_sbf2, dense(block.lin_sbf1, sbf))
                else:
                    x_kj = act(dense(block.lin_kj, m[(k, j)])) * dense(block.lin_rbf, rbf_gate)
                    s = dense(block.lin_sbf, sbf)
                    agg = agg + torch.einsum("b,f,bfg->g", s, x_kj, block.bilinear)
            if pp:
                agg = act(dense(block.lin_up, agg))
            x = x_ji + agg
            for res in block.before_skip:
                x = x + act(dense(res.lin2, act(dense(res.lin1, x))))
            x = mji + act(dense(block.lin_skip, x))
            for res in block.after_skip:
                x = x + act(dense(res.lin2, act(dense(res.lin1, x))))
            new_m[(j, i)] = x
        m = new_m
        node_e = node_e + output(out_block, m)
    return node_e.sum()


@pytest.mark.parametrize("pp", [False, True])
@pytest.mark.parametrize("reference_basis, radial_gate",
                         [(False, "kj"), (False, "ji"), (True, "kj")])
def test_forward_matches_paper_equations(pp, reference_basis, radial_gate):
    """Same weights -> the model reproduces the independent equation-by-
    equation implementation to float64 precision."""
    model = _small(pp=pp, seed=3, reference_basis=reference_basis, radial_gate=radial_gate,
                   cutoff=6.0).double()
    model.set_atomic_energies(SPECIES, [-0.5, -1.0, -2.0])
    s = _structure(n=5)
    g = structure_to_graph(s, 6.0)                  # 6 A > any pair distance: complete graph
    with torch.no_grad():
        e_model = model(g)["energy"]
        e_ref = _reference_energy(model, torch.tensor(s["atomic_numbers"]),
                                  torch.tensor(s["pos"], dtype=torch.float64))
    assert abs(float(e_model) - float(e_ref)) < 1e-10


def test_zero_init_output_predicts_atom_ref():
    """With output_init='zeros' the final layers vanish, so E = sum_i atom_ref[Z_i]."""
    m = _small(output_init="zeros")
    m.set_atomic_energies(SPECIES, [-1.0, -2.0, -3.0])
    e = float(m(_graph())["energy"])
    assert abs(e - 2 * (-1.0 - 2.0 - 3.0)) < 1e-12   # Z = [1, 6, 8, 1, 6, 8]


def test_initialization_conventions():
    """Glorot-orthogonal dense layers with zero biases, N(0, 2/F) bilinear
    tensor, uniform(-sqrt 3, sqrt 3) embeddings (the reference code)."""
    torch.manual_seed(0)
    m = DimeNet(n_features=64, n_interactions=1, n_spherical=3)
    w = m.interactions[0].lin_ji.weight.detach()
    gram = w @ w.T                                     # a scaled orthogonal matrix
    assert torch.allclose(gram, gram.diagonal().mean() * torch.eye(64), atol=1e-10)
    assert float(w.var()) == pytest.approx(2 / 128, rel=1e-6)
    assert float(m.interactions[0].lin_ji.bias.abs().max()) == 0.0
    assert float(m.interactions[0].bilinear.std()) == pytest.approx(2 / 64, rel=0.05)
    emb = m.embedding.embedding.weight
    assert float(emb.abs().max()) <= math.sqrt(3) and float(emb.std()) == pytest.approx(1.0, rel=0.05)


# physical properties

@pytest.mark.parametrize("pp", [False, True])
def test_energy_invariance_forces_equivariance(pp):
    """Invariant under rotation, translation and inversion; forces rotate."""
    model = ForceStressOutput(_small(pp=pp, seed=1)).double()
    rng = np.random.default_rng(1)
    pos = rng.uniform(0, 3, (6, 3))
    z = [1, 6, 8, 1, 6, 8]
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    t = rng.normal(size=(1, 3))
    o0 = model(structure_to_graph({"pos": pos, "atomic_numbers": z}, 5.0))
    o1 = model(structure_to_graph({"pos": pos @ R.T + t, "atomic_numbers": z}, 5.0))
    o2 = model(structure_to_graph({"pos": -pos, "atomic_numbers": z}, 5.0))
    Rt = torch.tensor(R, dtype=torch.float64)
    assert abs(float(o0["energy"]) - float(o1["energy"])) < 1e-10
    assert abs(float(o0["energy"]) - float(o2["energy"])) < 1e-10
    assert torch.allclose(o1["forces"].detach(), o0["forces"].detach() @ Rt.T, atol=1e-10)


def test_permutation_invariance():
    model = _small(seed=2).double()
    s = _structure()
    perm = np.array([3, 1, 5, 0, 4, 2])
    e0 = model(structure_to_graph(s, 5.0))["energy"]
    e1 = model(structure_to_graph({"pos": s["pos"][perm],
                                   "atomic_numbers": list(np.array(s["atomic_numbers"])[perm])},
                                  5.0))["energy"]
    assert abs(float(e0) - float(e1)) < 1e-11


@pytest.mark.parametrize("pp", [False, True])
def test_forces_are_minus_gradient(pp):
    """Autograd forces match central finite differences of the energy."""
    model = ForceStressOutput(_small(pp=pp, seed=4)).double()
    s = _structure()
    f = model(structure_to_graph(s, 5.0))["forces"].detach().numpy()

    def energy(pos):
        return float(model(structure_to_graph(
            {"pos": pos, "atomic_numbers": s["atomic_numbers"]}, 5.0))["energy"])

    h = 1e-5
    for (i, k) in [(0, 0), (2, 1), (5, 2)]:
        p_plus, p_minus = s["pos"].copy(), s["pos"].copy()
        p_plus[i, k] += h
        p_minus[i, k] -= h
        assert abs(-(energy(p_plus) - energy(p_minus)) / (2 * h) - f[i, k]) < 1e-7


def test_pes_smooth_across_cutoff():
    """The envelope of eq 8 makes energy and forces continuous when a third
    atom crosses the cutoff (both bases go to zero with their derivatives)."""
    model = ForceStressOutput(_small(seed=5, cutoff=4.0)).double()
    z = [8, 1, 1]
    eps = 1e-7
    outs = []
    # the third atom sits at the same distance from both others, so both of its
    # edges (and every triplet) cross the cutoff together
    for d in (4.0 - eps, 4.0 + eps):
        x = math.sqrt(d ** 2 - 0.45 ** 2)
        pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.9, 0.0], [x, 0.45, 0.0]])
        out = model(structure_to_graph({"pos": pos, "atomic_numbers": z}, 4.0))
        outs.append((float(out["energy"]), out["forces"].detach()))
    assert abs(outs[0][0] - outs[1][0]) < 1e-9
    assert (outs[0][1] - outs[1][1]).abs().max() < 1e-6
    # beyond the cutoff the energy is exactly additive: the pair plus the lone atom
    pair = float(model(structure_to_graph({"pos": pos[:2], "atomic_numbers": z[:2]}, 4.0))["energy"])
    lone = float(model(structure_to_graph({"pos": pos[2:], "atomic_numbers": z[2:]}, 4.0))["energy"])
    assert abs(outs[1][0] - (pair + lone)) < 1e-12


def test_distinguishes_hexagon_from_two_triangles():
    """ICLR paper Appendix A, Fig. 6: with a cutoff below the second-neighbor
    distance a distance-only GNN sees identical atom environments in a
    hexagon and in two triangles of the same bond length; directional
    message passing tells them apart through the 120 vs 60 degree angles."""
    from xnn.gnn.models.schnet import SchNet

    bond, cutoff = 1.5, 2.0
    hexagon = np.array([[math.cos(a), math.sin(a), 0.0]
                        for a in np.arange(6) * math.pi / 3]) * bond   # side = radius = bond
    tri = np.array([[math.cos(a), math.sin(a), 0.0]
                    for a in np.arange(3) * 2 * math.pi / 3]) * bond / math.sqrt(3)
    triangles = np.vstack([tri, tri + np.array([50.0, 0.0, 0.0])])
    assert abs(np.linalg.norm(tri[0] - tri[1]) - bond) < 1e-12
    z = [6] * 6
    g_hex = structure_to_graph({"pos": hexagon, "atomic_numbers": z}, cutoff)
    g_tri = structure_to_graph({"pos": triangles, "atomic_numbers": z}, cutoff)
    assert g_hex.num_edges == g_tri.num_edges == 12
    torch.manual_seed(0)
    schnet = SchNet(n_features=16, n_interactions=2, n_rbf=20, cutoff=cutoff).double()
    torch.nn.init.normal_(schnet.readout[-1].weight)
    assert abs(float(schnet(g_hex)["energy"]) - float(schnet(g_tri)["energy"])) < 1e-10
    dimenet = _small(seed=7, cutoff=cutoff).double()
    assert abs(float(dimenet(g_hex)["energy"]) - float(dimenet(g_tri)["energy"])) > 1e-6


def test_size_extensivity_and_batching():
    model = _small(seed=6).double()
    s = _structure()
    far = {"pos": s["pos"] + 100.0, "atomic_numbers": s["atomic_numbers"]}
    both = {"pos": np.vstack([s["pos"], far["pos"]]), "atomic_numbers": list(s["atomic_numbers"]) * 2}
    e1 = float(model(structure_to_graph(s, 5.0))["energy"])
    e2 = float(model(structure_to_graph(far, 5.0))["energy"])
    e12 = float(model(structure_to_graph(both, 5.0))["energy"])
    assert abs(e12 - (e1 + e2)) < 1e-10
    batch = collate([structure_to_graph(s, 5.0), structure_to_graph(far, 5.0)])
    out = model(batch)
    assert torch.allclose(out["energy"], torch.tensor([e1, e2], dtype=torch.float64), atol=1e-10)
    assert out["node_features"].shape == (12, model.node_feature_dim)
    assert out["node_energy"].shape == (12,)


def test_periodic_images_are_distinct_neighbors():
    """In a small cell the periodic images of an atom count as neighbors k of
    the message j -> i (only the exact reverse image is excluded), and the
    energy is invariant under a lattice translation of one atom."""
    model = ForceStressOutput(_small(seed=8, cutoff=4.0), compute_stress=True).double()
    s = _structure(n=3, spread=2.5)
    s["cell"], s["pbc"] = np.eye(3) * 3.0, [True, True, True]
    out = model(structure_to_graph(s, 4.0))
    assert out["stress"].shape == (1, 3, 3)
    shifted = dict(s, pos=s["pos"] + np.array([[3.0, 0.0, 0.0], [0, 0, 0], [0, 0, 0]]))
    out2 = model(structure_to_graph(shifted, 4.0))
    assert abs(float(out["energy"]) - float(out2["energy"])) < 1e-9
    assert torch.allclose(out["forces"].detach(), out2["forces"].detach(), atol=1e-9)


def test_float32_forward_matches_float64():
    model = _small(seed=9)
    e64 = float(model.double()(_graph())["energy"])
    torch.set_default_dtype(torch.float32)             # a float32 graph for the float32 model
    e32 = float(model.float()(_graph())["energy"])
    torch.set_default_dtype(torch.float64)
    assert abs(e64 - e32) < 1e-4 * max(1.0, abs(e64))


# deployment

@pytest.mark.parametrize("pp", [False, True])
def test_scriptable_and_lammps_export(tmp_path, pp):
    from xnn.common.deploy import export_to_lammps

    model = _small(pp=pp).eval()
    g = _graph(n=6, cutoff=5.0, periodic=True)
    scripted = torch.jit.script(model)
    d = (scripted.node_energy(g.atomic_numbers, g.edge_index, g.edge_vectors())
         - model.node_energy(g.atomic_numbers, g.edge_index, g.edge_vectors()))
    assert d.abs().max() < 1e-12
    path = str(tmp_path / "dimenet_lammps.pt")
    export_to_lammps(model, 5.0, path)
    loaded = torch.jit.load(path)
    out = loaded(g.pos, g.edge_index, g.cell_shifts, g.atomic_numbers, g.cell[0])
    ref = ForceStressOutput(model)(g)
    assert abs(float(out["total_energy"]) - float(ref["energy"])) < 1e-5
    assert (out["forces"] - ref["forces"].detach()).abs().max() < 1e-5


def test_les_wrapper_reads_node_features():
    from xnn.common.models.les import LatentEwald

    model = LatentEwald(_small(pp=True)).double()
    out = model(_graph())
    assert out["energy"].shape == (1,) and torch.isfinite(out["energy"]).all()


def test_multihead_and_reference_energies():
    from xnn.common.finetune import MultiHead, get_atomic_energies, set_atomic_energies

    model = _small(pp=True)
    set_atomic_energies(model, {1: -0.5, 6: -1.0, 8: -2.0})
    assert get_atomic_energies(model) == {1: -0.5, 6: -1.0, 8: -2.0}
    multi = MultiHead(model, ["a", "b"])
    out = multi(_graph())
    assert out["energy"].shape == (1,)


# config

def test_from_config_extras():
    cfg = from_dict({"model": {
        "name": "dimenet", "cutoff": 4.5, "n_features": 16, "n_interactions": 1, "n_rbf": 5,
        "extra": {"n_spherical": 4, "n_bilinear": 3, "p": 5, "n_before_skip": 2,
                  "n_after_skip": 1, "n_output_layers": 1, "output_init": "zeros",
                  "reference_basis": True, "species": SPECIES,
                  "atomic_energies": [-13.6, -1030.0, -2043.0]}}})
    m = build_model(cfg.model)
    assert isinstance(m, DimeNet) and not isinstance(m, DimeNetPP)
    assert m.sbf.zeros.shape == (4, 5) and m.interactions[0].bilinear.shape == (3, 16, 16)
    assert m.envelope.p == 5 and m.reference_basis
    assert len(m.interactions[0].before_skip) == 2 and len(m.interactions[0].after_skip) == 1
    assert len(m.outputs[0].dense) == 1
    assert float(m.atom_ref.weight[6, 0]) == pytest.approx(-1030.0)
    assert float(m.outputs[0].final.weight.abs().max()) == 0.0


def test_from_config_pp_and_reference_key_translation():
    """The reference code's config spellings (config_pp.yaml) build the paper's
    DimeNet++; envelope_exponent maps to p = exponent + 1."""
    cfg = from_dict({"model": {
        "name": "dimenet++", "cutoff": 5.0, "emb_size": 32, "out_emb_size": 24,
        "int_emb_size": 8, "basis_emb_size": 4, "num_blocks": 2, "num_spherical": 3,
        "num_radial": 4, "envelope_exponent": 5, "num_before_skip": 1, "num_after_skip": 2,
        "num_dense_output": 3}})
    assert cfg.model.n_features == 32 and cfg.model.n_interactions == 2 and cfg.model.n_rbf == 4
    m = build_model(cfg.model)
    assert isinstance(m, DimeNetPP)
    assert m.outputs[0].up.weight.shape == (24, 32)
    assert m.interactions[0].lin_down.weight.shape == (8, 32)
    assert m.interactions[0].lin_rbf1.weight.shape == (4, 4)
    assert m.envelope.p == 6 and m.n_sbf == 12
    cfg2 = from_dict({"model": {"name": "dimenet", "emb_size": 16, "num_bilinear": 2,
                                "num_blocks": 1, "num_radial": 3, "num_spherical": 2}})
    m2 = build_model(cfg2.model)
    assert m2.interactions[0].bilinear.shape == (2, 16, 16) and m2.n_sbf == 6


# parity with the authors' TensorFlow implementation (optional)

def _parity(mode):
    pytest.importorskip("tensorflow")
    path = os.environ.get("DIMENET_UPSTREAM_PATH")
    if not path or not os.path.isdir(os.path.join(path, "dimenet")):
        pytest.skip("set DIMENET_UPSTREAM_PATH to a checkout of gasteigerjo/dimenet")
    if mode == "pretrained" and not os.path.isdir(os.path.join(path, "pretrained", "dimenet_pp", "U0")):
        pytest.skip("the checkout has no pretrained/dimenet_pp/U0")
    import subprocess
    import sys
    script = os.path.join(os.path.dirname(__file__), "dimenet_tf_parity.py")
    res = subprocess.run([sys.executable, script, path, mode], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    return float(res.stdout.strip().splitlines()[-1])


def test_parity_vs_reference_random_weights():
    """Weight transplant from the TF DimeNet and DimeNet++ (float64, random
    weights) gives identical energies and forces: the worst relative error
    (energy relative to ``|E|``, forces relative to ``max|F|``) is at the float64
    floor, measured at 7e-14 over both models; a convention error (angle,
    envelope, basis order) would show as O(1)."""
    assert _parity("random") < 1e-12


def test_parity_vs_reference_pretrained():
    """The published DimeNet++ U0 weights give the same energies and forces
    in xnn. The published models exist in float32 only; measured floors over
    four molecules: energies 1e-7 relative (float32 epsilon), forces 1.5e-5
    of the largest force component (gradients of a 4-block float32 net)."""
    assert _parity("pretrained") < 1e-4
