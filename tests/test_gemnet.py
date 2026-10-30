"""GemNet tests: manuscript fidelity (NeurIPS 2021), the bases, the triplet and
quadruplet enumeration, invariances, smoothness, scaling factors, direct
forces, deployment, configs and (optionally) fidelity to the reference code.

``_reference`` re-implements the paper's equations (eq 8-11, the blocks of
Fig. 4) with loops over atoms, closed-form spherical Bessel functions and
spherical harmonics of the geometric angles, reusing only the model's
parameters; the model must reproduce it to float64 precision.
"""
import math
import os

import numpy as np
import pytest
import torch

from xnn.common.config import from_dict
from xnn.common.data import collate, structure_to_graph
from xnn.common.models import ForceStressOutput, available_models, build_model
from xnn.gnn.featurizers import real_spherical_harmonics, zonal_harmonics
from xnn.gnn.models.gemnet import GemNet, reverse_edges, triplets

SPECIES = [1, 6, 8]


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _structure(n=5, seed=0, spread=3.0):
    rng = np.random.default_rng(seed)
    return {"pos": rng.uniform(0, spread, (n, 3)), "atomic_numbers": ([1, 6, 8] * n)[:n]}


def _small(seed=0, scales=True, **kw):
    torch.manual_seed(seed)
    opts = dict(n_features=12, n_edge_features=10, n_interactions=2, n_rbf=3, n_spherical=3,
                cutoff=3.0, int_cutoff=6.0, n_triplet_features=6, n_quadruplet_features=5,
                n_rbf_features=4, n_cbf_features=4, n_sbf_features=5, n_bilinear_triplet=7,
                n_bilinear_quadruplet=6, n_atom_residual=1)
    opts.update(kw)
    model = GemNet(**opts)
    if scales:                                   # non-trivial factors, to test their placement
        gen = torch.Generator().manual_seed(seed + 100)
        with torch.no_grad():
            for factor in model.scale_factors().values():
                factor.scale_factor.fill_(float(0.5 + 1.5 * torch.rand(1, generator=gen)))
    return model


def _graph(s, cutoff):
    return structure_to_graph(s, cutoff)


# registry / defaults

def test_registered():
    assert "gemnet" in available_models()


def test_paper_defaults():
    """GemNet() is the paper's GemNet-Q (App. F, Table 6) with 2.2M parameters;
    GemNet-T has 1.9M (section 7)."""
    m = GemNet()
    assert m.emb_cutoff == 5.0 and m.int_cutoff == 10.0 and m.cutoff == 10.0
    assert m.envelope.p == 5 and m.int_envelope.cutoff == 10.0
    assert len(m.interactions) == 4 and len(m.outputs) == 5
    assert m.cbf.zeros.shape == (7, 6) and m.rbf.freqs.shape == (6,)
    assert isinstance(m.rbf.freqs, torch.nn.Parameter)
    block = m.interactions[0]
    assert block.trip.bilinear.shape == (16, 64, 64)
    assert block.quad.bilinear.shape == (32, 32, 32)
    assert m.lin_sbf4.weight.shape == (32, 49 * 6) and m.lin_cbf3.weight.shape == (16, 42)
    assert all(lin.bias is None for lin in m.modules() if isinstance(lin, torch.nn.Linear))
    n_q = sum(p.numel() for p in m.parameters())
    t = GemNet(quadruplets=False)
    n_t = sum(p.numel() for p in t.parameters())
    assert round(n_q / 1e5) == 22 and round(n_t / 1e5) == 19
    assert t.cutoff == 5.0 and t.interactions[0].quad is None and t.lin_sbf4 is None


def test_invalid_options():
    with pytest.raises(ValueError):
        GemNet(n_features=8, output_init="ones")
    with pytest.raises(ValueError):
        GemNet(n_features=8, atomic_energies=[1.0])


def test_initialization_conventions():
    """He-orthogonal dense layers: exactly zero mean and variance 1/fan_in per
    output unit; atom embeddings uniform in [-sqrt 3, sqrt 3]."""
    torch.manual_seed(0)
    m = GemNet(n_features=64, n_edge_features=64, n_interactions=1, n_spherical=3)
    w = m.interactions[0].dense_ca.weight.detach()
    assert torch.allclose(w.mean(1), torch.zeros(64), atol=1e-12)
    assert torch.allclose(w.var(1), torch.full((64,), 1 / 64), rtol=1e-4)
    bil = m.interactions[0].trip.bilinear.detach()           # (J, F_in, F_out), fan-in J F_in
    per_out = bil.reshape(-1, bil.shape[2])
    # (the standardization's 1e-6 regularizer lowers small variances slightly)
    assert torch.allclose(per_out.var(0), torch.full((bil.shape[2],), 1 / per_out.shape[0]), rtol=2e-3)
    emb = m.atom_embedding.weight
    assert float(emb.abs().max()) <= math.sqrt(3)


# the bases (paper section 5)

def test_real_spherical_harmonics():
    """Closed forms for l <= 1, m = 0 equals the zonal harmonics, the addition
    theorem sum_m Y_lm(u) Y_lm(v) = (2l+1)/(4 pi) P_l(u.v), orthonormality."""
    rng = np.random.default_rng(0)
    u = torch.tensor(rng.normal(size=(50, 3)))
    u = u / u.norm(dim=1, keepdim=True)
    v = torch.tensor(rng.normal(size=(50, 3)))
    v = v / v.norm(dim=1, keepdim=True)
    yu = real_spherical_harmonics(4, u[:, 0], u[:, 1], u[:, 2])
    yv = real_spherical_harmonics(4, v[:, 0], v[:, 1], v[:, 2])
    k = math.sqrt(3 / (4 * math.pi))
    assert torch.allclose(yu[:, 1:4], k * u[:, [1, 2, 0]], atol=1e-14)   # m = -1, 0, 1: y, z, x
    for l in range(5):
        assert torch.allclose(yu[:, l * l + l], zonal_harmonics(l, u[:, 2])[:, l], atol=1e-14)
        dot = (yu[:, l * l:(l + 1) ** 2] * yv[:, l * l:(l + 1) ** 2]).sum(1)
        legendre = zonal_harmonics(l, (u * v).sum(1))[:, l] * math.sqrt((2 * l + 1) / (4 * math.pi))
        assert torch.allclose(dot, legendre, atol=1e-13), l
    # orthonormal on the sphere (Gauss-Legendre in cos(theta), uniform in phi)
    xc, wc = np.polynomial.legendre.leggauss(20)
    phi = np.arange(40) * 2 * np.pi / 40
    ct, ph = np.meshgrid(xc, phi, indexing="ij")
    st = np.sqrt(1 - ct ** 2)
    y = real_spherical_harmonics(4, torch.tensor(st * np.cos(ph)), torch.tensor(st * np.sin(ph)),
                                 torch.tensor(ct)).reshape(-1, 25).numpy()
    w = (wc[:, None] * np.full(40, 2 * np.pi / 40)[None, :]).reshape(-1)
    assert np.allclose(np.einsum("p,pi,pj->ij", w, y, y), np.eye(25), atol=1e-12)


def test_3d_basis_orthonormal_in_cutoff_ball():
    """eq 10: the spherical Fourier-Bessel functions e_SBF,lmn are orthonormal
    in the ball of radius c_emb (before the envelope)."""
    m = GemNet(n_spherical=3, n_rbf=3, cutoff=4.0, n_interactions=1)
    c = 4.0
    xr, wr = np.polynomial.legendre.leggauss(120)
    d = 0.5 * c * (xr + 1)
    wd = 0.5 * c * wr * d ** 2
    rad = m.cbf.radial(torch.tensor(d)).numpy()                       # (D, L, N)
    xc, wc = np.polynomial.legendre.leggauss(16)
    phi = np.arange(24) * 2 * np.pi / 24
    ct, ph = np.meshgrid(xc, phi, indexing="ij")
    st = np.sqrt(1 - ct ** 2)
    y = real_spherical_harmonics(2, torch.tensor(st * np.cos(ph)), torch.tensor(st * np.sin(ph)),
                                 torch.tensor(ct)).reshape(-1, 9).numpy()
    wa = (wc[:, None] * np.full(24, 2 * np.pi / 24)[None, :]).reshape(-1)
    deg = m.sbf_degree.numpy()
    radial = rad[:, deg, :]                                           # (D, 9, N)
    gram = (np.einsum("d,dan,dbk->anbk", wd, radial, radial)
            * np.einsum("p,pa,pb->ab", wa, y, y)[:, None, :, None])
    assert np.allclose(gram.reshape(27, 27), np.eye(27), atol=1e-6)


# triplets, quadruplets, reverse edges

def _periodic(n=4, cell=3.0, cutoff=4.0, seed=1):
    s = _structure(n, seed=seed, spread=cell)
    s["cell"], s["pbc"] = np.eye(3) * cell, [True, True, True]
    return structure_to_graph(s, cutoff)


@pytest.mark.parametrize("periodic", [False, True])
def test_reverse_edges(periodic):
    g = _periodic() if periodic else _graph(_structure(6), 5.0)
    vec = g.edge_vectors()
    rev = reverse_edges(g.edge_index, vec)
    assert torch.equal(rev[rev], torch.arange(len(rev)))
    assert torch.equal(g.edge_index[0][rev], g.edge_index[1])
    assert torch.equal(vec[rev], -vec)
    if periodic:                                           # images: one atom pair, several edges
        assert bool((g.edge_index[0] == g.edge_index[1]).any())
    with pytest.raises(RuntimeError):
        reverse_edges(g.edge_index[:, 1:], vec[1:])


@pytest.mark.parametrize("periodic", [False, True])
def test_triplets_vs_brute_force(periodic):
    """Every pair of distinct edges into the same atom, grouped by c -> a."""
    g = _periodic() if periodic else _graph(_structure(6), 5.0)
    ca, ba, slot = triplets(g.edge_index, g.num_nodes)
    dst = g.edge_index[1].tolist()
    brute = {(e1, e2) for e1 in range(len(dst)) for e2 in range(len(dst))
             if e1 != e2 and dst[e1] == dst[e2]}
    found = list(zip(ca.tolist(), ba.tolist()))
    assert set(found) == brute and len(found) == len(brute)
    assert bool((ca[1:] >= ca[:-1]).all())
    for e in set(ca.tolist()):
        assert sorted(slot[ca == e].tolist()) == list(range(int((ca == e).sum())))


def _geometry(model, g):
    """The model's quadruplets as (emb edge c->a, graph edge b->a, emb edge d->b)."""
    vec = g.edge_vectors()
    r_all = vec.norm(dim=-1)
    rev_all = reverse_edges(g.edge_index, vec)
    emb_ids = torch.nonzero(r_all <= model.emb_cutoff).squeeze(-1)
    int_ids = torch.nonzero(r_all <= model.int_cutoff).squeeze(-1)
    src, dst = g.edge_index[0][emb_ids], g.edge_index[1][emb_ids]
    out = model._quadruplet_geometry(src, dst, vec[emb_ids], r_all[emb_ids], emb_ids,
                                     g.edge_index, vec, r_all, rev_all, g.num_nodes)
    intm_db, intm_q, _, _, ca, q_intm, q_k, sph = out
    quads = list(zip(emb_ids[ca].tolist(), int_ids[intm_q[q_intm]].tolist(),
                     emb_ids[intm_db[q_intm]].tolist()))
    return quads, ca, q_k, sph


def _brute_quadruplets(g, emb_cutoff, int_cutoff):
    """c -> a - b <- d with four distinct atoms (periodic images are distinct)."""
    src, dst = g.edge_index[0].tolist(), g.edge_index[1].tolist()
    vec = g.edge_vectors().numpy()
    r = np.linalg.norm(vec, axis=1)
    out = set()
    for ca in np.nonzero(r <= emb_cutoff)[0]:
        for ba in np.nonzero(r <= int_cutoff)[0]:
            if dst[ba] != dst[ca]:
                continue
            for db in np.nonzero(r <= emb_cutoff)[0]:
                if dst[db] != src[ba]:
                    continue
                # positions relative to a: c, b, d
                pos = {"c": (src[ca], -vec[ca]), "b": (src[ba], -vec[ba]),
                       "d": (src[db], -vec[ba] - vec[db]), "a": (dst[ca], np.zeros(3))}
                atoms = list(pos.values())
                distinct = all(not (atoms[i][0] == atoms[j][0]
                                    and np.allclose(atoms[i][1], atoms[j][1], atol=1e-6))
                               for i in range(4) for j in range(i + 1, 4))
                if distinct:
                    out.add((int(ca), int(ba), int(db)))
    return out


@pytest.mark.parametrize("periodic", [False, True])
def test_quadruplets_vs_brute_force(periodic):
    model = _small(cutoff=2.5, int_cutoff=4.0)
    s = _structure(6, seed=3)
    g = (_periodic(n=3, cell=3.2, cutoff=4.0, seed=2) if periodic
         else structure_to_graph(s, 4.0))
    quads, ca, q_k, sph = _geometry(model, g)
    assert len(set(quads)) == len(quads)
    assert set(quads) == _brute_quadruplets(g, 2.5, 4.0)
    assert len(quads) > 0 and bool((ca[1:] >= ca[:-1]).all())
    assert sph.shape == (len(quads), 9)


# manuscript equations

def _act(x):
    return torch.nn.functional.silu(x) / 0.6


def _lin(layer, x, act=False):
    y = x @ layer.weight.T
    return _act(y) if act else y


def _jl(l, x):
    s, c = math.sin(x), math.cos(x)
    if l == 0:
        return s / x
    if l == 1:
        return s / x ** 2 - c / x
    if l == 2:
        return (3 / x ** 3 - 1 / x) * s - 3 * c / x ** 2
    return (15 / x ** 4 - 6 / x ** 2) * s - (15 / x ** 3 - 1 / x) * c


def _ylm(l, m, polar, azim):
    """Real spherical harmonics (no Condon-Shortley phase), l <= 2."""
    st, ct = math.sin(polar), math.cos(polar)
    table = {
        (0, 0): 0.5 * math.sqrt(1 / math.pi),
        (1, -1): math.sqrt(3 / (4 * math.pi)) * st * math.sin(azim),
        (1, 0): math.sqrt(3 / (4 * math.pi)) * ct,
        (1, 1): math.sqrt(3 / (4 * math.pi)) * st * math.cos(azim),
        (2, -2): 0.25 * math.sqrt(15 / math.pi) * st ** 2 * math.sin(2 * azim),
        (2, -1): 0.5 * math.sqrt(15 / math.pi) * st * ct * math.sin(azim),
        (2, 0): 0.25 * math.sqrt(5 / math.pi) * (3 * ct ** 2 - 1),
        (2, 1): 0.5 * math.sqrt(15 / math.pi) * st * ct * math.cos(azim),
        (2, 2): 0.25 * math.sqrt(15 / math.pi) * st ** 2 * math.cos(2 * azim),
    }
    return table[(l, m)]


def _angle(u, v):
    return math.acos(max(-1.0, min(1.0, float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v))))))


def _reject(u, axis):
    return u - (u @ axis) / (axis @ axis) * axis


def _reference(model, Z, pos):
    """Loop-based forward pass of the paper's equations (float64): energy and,
    for the direct variants, forces."""
    rc, ri, p = model.emb_cutoff, model.int_cutoff, model.envelope.p
    assert model.n_spherical == 3
    N = model.n_rbf
    zeros = model.cbf.zeros.numpy()
    freqs = model.rbf.freqs.detach()
    x = pos
    n = len(Z)
    dist = lambda i, j: float(np.linalg.norm(x[i] - x[j]))     # noqa: E731

    def env(d, c):
        t = d / c
        return 1 - (p + 1) * (p + 2) / 2 * t ** p + p * (p + 2) * t ** (p + 1) - p * (p + 1) / 2 * t ** (p + 2)

    def e_rbf(d):                                                    # eq 8
        return math.sqrt(2 / rc) * torch.sin(freqs * d / rc) / d * env(d, rc)

    def radial(l, k, d, c):
        z = zeros[l, k]
        return math.sqrt(2 / (c ** 3 * _jl(l + 1, z) ** 2)) * _jl(l, z * d / c)

    def e_cbf(d, angle, c):                                          # eq 9
        return torch.tensor([radial(l, k, d, c) * _ylm(l, 0, angle, 0.0) * env(d, c)
                             for l in range(3) for k in range(N)])

    def e_sbf(d, polar, azim):                                       # eq 10
        return torch.tensor([radial(l, k, d, rc) * _ylm(l, mm, polar, azim) * env(d, rc)
                             for l in range(3) for mm in range(-l, l + 1) for k in range(N)])

    def mlp(dense, residuals, v):
        v = _lin(dense, v, True)
        for res in residuals:
            v = (v + _lin(res.lin2, _lin(res.lin1, v, True), True)) / math.sqrt(2)
        return v

    emb = [(c, a) for a in range(n) for c in range(n) if c != a and dist(c, a) <= rc]
    rbf = {e: e_rbf(dist(*e)) for e in emb}
    h = model.atom_embedding.weight[Z]
    m = {(c, a): _lin(model.edge_embedding.dense, torch.cat([h[c], h[a], rbf[(c, a)]]), True)
         for (c, a) in emb}
    rbf3 = {e: _lin(model.lin_rbf3, rbf[e]) for e in emb}
    rbf_h = {e: _lin(model.lin_rbf_h, rbf[e]) for e in emb}
    rbf_out = {e: _lin(model.lin_rbf_out, rbf[e]) for e in emb}
    rbf4 = {e: _lin(model.lin_rbf4, rbf[e]) for e in emb} if model.quadruplets else None

    def output(block, m):
        hsum = torch.zeros(n, m[emb[0]].shape[0])
        gated = {}
        for (c, a) in emb:
            gated[(c, a)] = m[(c, a)] * _lin(block.atom.lin_rbf, rbf_out[(c, a)])
            hsum[a] = hsum[a] + gated[(c, a)]
        hh = mlp(block.atom.dense, block.atom.residuals, hsum * block.atom.scale_sum.scale_factor)
        t = _lin(block.final, hh)[:, 0]
        force = {}
        if block.forces is not None:
            fh = block.forces
            for e in emb:
                force[e] = _lin(fh.final, mlp(fh.dense, fh.residuals,
                                              gated[e] * fh.scale_rbf.scale_factor))[0]
        return t, force

    def symmetric(inter, y):
        return {(c, a): (_lin(inter.up_ca, y[(c, a)], True) + _lin(inter.up_ac, y[(a, c)], True))
                / math.sqrt(2) for (c, a) in emb}

    def interaction(block, h, m):
        tr = block.trip
        x_ba = {e: _lin(tr.down, _lin(tr.dense_ba, m[e], True) * _lin(tr.lin_rbf, rbf3[e])
                        * tr.scale_rbf.scale_factor, True) for e in emb}
        y3 = {}
        for (c, a) in emb:                                           # one-hop: b in N_a \ {c}
            acc = torch.zeros(tr.bilinear.shape[2])
            for (b, a2) in emb:
                if a2 != a or b == c:
                    continue
                basis = _lin(model.lin_cbf3, e_cbf(dist(c, a), _angle(x[c] - x[a], x[b] - x[a]), rc))
                acc = acc + torch.einsum("j,f,jfo->o", basis, x_ba[(b, a)], tr.bilinear)
            y3[(c, a)] = acc * tr.scale_cbf_sum.scale_factor
        x3 = symmetric(tr, y3)
        merged = {e: _lin(block.dense_ca, m[e], True) + x3[e] for e in emb}
        if block.quad is not None:                                   # eq 11
            q = block.quad
            x_db = {e: _lin(q.down, _lin(q.dense_db, m[e], True) * _lin(q.lin_rbf, rbf4[e])
                            * q.scale_rbf.scale_factor, True) for e in emb}
            y4 = {}
            for (c, a) in emb:
                acc = torch.zeros(q.bilinear.shape[2])
                for b in range(n):
                    if b in (a, c) or dist(b, a) > ri:
                        continue
                    for d in range(n):
                        if d in (a, b, c) or (d, b) not in m:
                            continue
                        cbf = e_cbf(dist(b, a), _angle(x[a] - x[b], x[d] - x[b]), ri)
                        z = (x_db[(d, b)] * _lin(q.lin_cbf, _lin(model.lin_cbf4, cbf))
                             * q.scale_cbf.scale_factor)
                        polar = _angle(x[c] - x[a], x[b] - x[a])
                        azim = _angle(_reject(x[c] - x[a], x[b] - x[a]), _reject(x[d] - x[b], x[a] - x[b]))
                        basis = _lin(model.lin_sbf4, e_sbf(dist(c, a), polar, azim))
                        acc = acc + torch.einsum("j,f,jfo->o", basis, z, q.bilinear)
                y4[(c, a)] = acc * q.scale_sbf_sum.scale_factor
            x4 = symmetric(q, y4)
            merged = {e: (merged[e] + x4[e]) / math.sqrt(3) for e in emb}
        else:
            merged = {e: merged[e] / math.sqrt(2) for e in emb}
        new = {}
        for e in emb:
            v = merged[e]
            for res in block.before_skip:
                v = (v + _lin(res.lin2, _lin(res.lin1, v, True), True)) / math.sqrt(2)
            v = (m[e] + v) / math.sqrt(2)
            for res in block.after_skip:
                v = (v + _lin(res.lin2, _lin(res.lin1, v, True), True)) / math.sqrt(2)
            new[e] = v
        au = block.atom_update
        hsum = torch.zeros(n, new[emb[0]].shape[0])
        for (c, a) in emb:
            hsum[a] = hsum[a] + new[(c, a)] * _lin(au.lin_rbf, rbf_h[(c, a)])
        h = (h + mlp(au.dense, au.residuals, hsum * au.scale_sum.scale_factor)) / math.sqrt(2)
        out = {}
        for (c, a) in emb:
            v = _lin(block.concat.dense, torch.cat([h[c], h[a], new[(c, a)]]), True)
            for res in block.residual_m:
                v = (v + _lin(res.lin2, _lin(res.lin1, v, True), True)) / math.sqrt(2)
            out[(c, a)] = (new[(c, a)] + v) / math.sqrt(2)
        return h, out

    energy = model.atom_ref.weight[Z, 0].sum()
    edge_force = {e: 0.0 for e in emb}
    t, f = output(model.outputs[0], m)
    energy = energy + t.sum()
    for e in f:
        edge_force[e] = edge_force[e] + f[e]
    for block, out_block in zip(model.interactions, model.outputs[1:]):
        h, m = interaction(block, h, m)
        t, f = output(out_block, m)
        energy = energy + t.sum()
        for e in f:
            edge_force[e] = edge_force[e] + f[e]
    forces = torch.zeros(n, 3)
    if model.direct_forces:
        for (c, a) in emb:
            fe = edge_force[(c, a)]
            if model.forces_coupled:
                fe = 0.5 * (fe + edge_force[(a, c)])
            forces[a] = forces[a] + fe * torch.tensor(x[a] - x[c]) / dist(c, a)
    return energy, forces


@pytest.mark.parametrize("quadruplets, direct, coupled",
                         [(True, False, False), (False, False, False),
                          (True, True, False), (False, True, True)])
def test_forward_matches_paper_equations(quadruplets, direct, coupled):
    """Same weights -> the model reproduces the loop-based implementation of the
    paper's equations (energies, and direct forces) to float64 precision."""
    model = _small(seed=3, quadruplets=quadruplets, direct_forces=direct,
                   forces_coupled=coupled).double()
    model.set_atomic_energies(SPECIES, [-0.5, -1.0, -2.0])
    # two of the ten pairs (0-4, 2-4) lie between c_emb = 3 and c_int = 6
    s = {"pos": np.array([[0.0, 0.0, 0.0], [1.1, 0.3, -0.2], [0.4, 1.5, 0.6],
                          [2.2, 1.4, 0.9], [3.4, 0.2, 1.3]]),
         "atomic_numbers": [1, 6, 8, 1, 6]}
    g = structure_to_graph(s, 6.0)                   # GemNet-T keeps the edges within c_emb
    with torch.no_grad():
        out = model(g)
        e_ref, f_ref = _reference(model, torch.tensor(s["atomic_numbers"]), s["pos"])
    r = g.edge_vectors().norm(dim=-1)
    assert bool((r > model.emb_cutoff).any())          # both kinds of pairs are present
    assert abs(float(out["energy"]) - float(e_ref)) < 1e-10 * max(1.0, abs(float(e_ref)))
    if direct:
        assert torch.allclose(out["forces"], f_ref, atol=1e-10)
        assert float(f_ref.abs().max()) > 1e-3


def test_zero_init_output_predicts_atom_ref():
    m = _small(output_init="zeros")
    m.set_atomic_energies(SPECIES, [-1.0, -2.0, -3.0])
    e = float(m(_graph(_structure(6), m.cutoff))["energy"])
    assert abs(e - 2 * (-1.0 - 2.0 - 3.0)) < 1e-12


# physical properties

@pytest.mark.parametrize("quadruplets", [True, False])
def test_energy_invariance_forces_equivariance(quadruplets):
    """Invariant under rotation, translation, reflection and permutation;
    autograd forces rotate."""
    model = ForceStressOutput(_small(seed=1, quadruplets=quadruplets)).double()
    rng = np.random.default_rng(1)
    s = _structure(6, seed=1)
    pos, z = s["pos"], s["atomic_numbers"]
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    cut = model.cutoff
    o0 = model(structure_to_graph({"pos": pos, "atomic_numbers": z}, cut))
    o1 = model(structure_to_graph({"pos": pos @ R.T + 1.3, "atomic_numbers": z}, cut))
    o2 = model(structure_to_graph({"pos": pos * np.array([1, 1, -1]), "atomic_numbers": z}, cut))
    perm = np.array([3, 1, 5, 0, 4, 2])
    o3 = model(structure_to_graph({"pos": pos[perm], "atomic_numbers": list(np.array(z)[perm])}, cut))
    e0 = float(o0["energy"])
    for o in (o1, o2, o3):
        assert abs(float(o["energy"]) - e0) < 1e-10
    Rt = torch.tensor(R)
    assert torch.allclose(o1["forces"].detach(), o0["forces"].detach() @ Rt.T, atol=1e-10)
    assert torch.allclose(o3["forces"].detach(), o0["forces"].detach()[perm], atol=1e-10)


@pytest.mark.parametrize("quadruplets", [True, False])
def test_direct_forces_are_equivariant(quadruplets):
    model = ForceStressOutput(_small(seed=2, quadruplets=quadruplets, direct_forces=True)).double()
    s = _structure(6, seed=2)
    R, _ = np.linalg.qr(np.random.default_rng(2).normal(size=(3, 3)))
    o0 = model(structure_to_graph(s, model.cutoff))
    o1 = model(structure_to_graph(dict(s, pos=s["pos"] @ R.T), model.cutoff))
    assert torch.allclose(o1["forces"], o0["forces"] @ torch.tensor(R).T, atol=1e-10)
    assert abs(float(o0["energy"]) - float(o1["energy"])) < 1e-10
    # the forces are the model's own prediction, not the energy gradient
    grad = ForceStressOutput(_small(seed=2, quadruplets=quadruplets)).double()
    assert not torch.allclose(o0["forces"], grad(structure_to_graph(s, model.cutoff))["forces"])


@pytest.mark.parametrize("quadruplets", [True, False])
def test_forces_are_minus_gradient(quadruplets):
    model = ForceStressOutput(_small(seed=4, quadruplets=quadruplets)).double()
    s = _structure(5, seed=4)
    f = model(structure_to_graph(s, model.cutoff))["forces"].detach().numpy()

    def energy(pos):
        return float(model(structure_to_graph(dict(s, pos=pos), model.cutoff))["energy"])

    h = 1e-5
    for (i, k) in [(0, 0), (2, 1), (4, 2)]:
        p_plus, p_minus = s["pos"].copy(), s["pos"].copy()
        p_plus[i, k] += h
        p_minus[i, k] -= h
        assert abs(-(energy(p_plus) - energy(p_minus)) / (2 * h) - f[i, k]) < 1e-7


def test_pes_smooth_across_embedding_cutoff():
    """An atom leaving c_emb (both of its edges at once) changes energy and
    forces continuously; beyond it the energy is additive (GemNet-T)."""
    model = ForceStressOutput(_small(seed=5, quadruplets=False, cutoff=3.0)).double()
    z = [8, 1, 1]
    outs = []
    for d in (3.0 - 1e-7, 3.0 + 1e-7):
        x = math.sqrt(d ** 2 - 0.45 ** 2)
        pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.9, 0.0], [x, 0.45, 0.0]])
        out = model(structure_to_graph({"pos": pos, "atomic_numbers": z}, 3.0))
        outs.append((float(out["energy"]), out["forces"].detach()))
    assert abs(outs[0][0] - outs[1][0]) < 1e-9
    assert (outs[0][1] - outs[1][1]).abs().max() < 1e-6
    pair = float(model(structure_to_graph({"pos": pos[:2], "atomic_numbers": z[:2]}, 3.0))["energy"])
    lone = float(model(structure_to_graph({"pos": pos[2:], "atomic_numbers": z[2:]}, 3.0))["energy"])
    assert abs(outs[1][0] - (pair + lone)) < 1e-12


def test_pes_smooth_across_interaction_cutoff():
    """Two dimers beyond c_emb of each other interact only through quadruplets;
    the interaction vanishes smoothly when the closest pair leaves c_int."""
    model = ForceStressOutput(_small(seed=6, cutoff=1.5, int_cutoff=4.0)).double()
    z = [6, 8, 6, 1]
    dimer = np.array([[0.0, 0.0, 0.0], [0.3, 1.1, 0.2]])

    def place(sep):
        return np.vstack([dimer, dimer * np.array([1, -1, 1]) + np.array([sep, 0.1, 0.0])])

    def run(sep):
        out = model(structure_to_graph({"pos": place(sep), "atomic_numbers": z}, 4.0))
        return float(out["energy"]), out["forces"].detach()

    def closest(sep):
        pos = place(sep)
        return min(np.linalg.norm(pos[i] - pos[j]) for i in (0, 1) for j in (2, 3))

    lo, hi = 3.0, 6.0                                   # the closest pair at exactly c_int
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if closest(mid) < 4.0 else (lo, mid)
    sep = 0.5 * (lo + hi)
    e_in, f_in = run(sep - 1e-7)
    e_out, f_out = run(sep + 1e-7)
    e_near, _ = run(sep - 1.5)
    assert abs(e_in - e_out) < 1e-9 and (f_in - f_out).abs().max() < 1e-6
    e_far, _ = run(20.0)
    assert abs(e_near - e_far) > 1e-8                   # the dimers do interact inside c_int
    sep_mono = model(structure_to_graph({"pos": dimer, "atomic_numbers": z[:2]}, 4.0))["energy"]
    other = dimer * np.array([1, -1, 1]) + np.array([20.0, 0.1, 0.0])
    sep_other = model(structure_to_graph({"pos": other, "atomic_numbers": z[2:]}, 4.0))["energy"]
    assert abs(e_far - float(sep_mono + sep_other)) < 1e-12


def test_dihedral_resolution():
    """With c_emb below the 1-3 distance of a chain A-B-C-D, GemNet-T sees only
    bonds and bond angles, so all torsions look alike; GemNet-Q resolves the
    dihedral through quadruplets (mirror images stay equivalent)."""
    bond, angle = 1.5, math.radians(110.0)

    def chain(torsion):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([bond, 0.0, 0.0])
        c = b + bond * np.array([-math.cos(angle), math.sin(angle), 0.0])
        bc = (c - b) / bond
        n = np.cross(b - a, c - b)
        n /= np.linalg.norm(n)
        m = np.cross(n, bc)
        d = c + bond * (-math.cos(angle) * bc + math.sin(angle) * (math.cos(torsion) * m
                                                                    + math.sin(torsion) * n))
        return {"pos": np.array([a, b, c, d]), "atomic_numbers": [6, 6, 6, 6]}

    assert np.linalg.norm(chain(0.0)["pos"][0] - chain(0.0)["pos"][2]) > 2.0
    q = _small(seed=7, cutoff=2.0, int_cutoff=6.0)
    t = _small(seed=7, cutoff=2.0, quadruplets=False)
    e = {name: [float(model(structure_to_graph(chain(tor), model.cutoff))["energy"])
                for tor in (math.radians(60), math.radians(180), math.radians(-60))]
         for name, model in (("q", q), ("t", t))}
    assert abs(e["t"][0] - e["t"][1]) < 1e-12
    assert abs(e["q"][0] - e["q"][1]) > 1e-6
    assert abs(e["q"][0] - e["q"][2]) < 1e-10


def test_size_extensivity_and_batching():
    model = _small(seed=6).double()
    s = _structure(5)
    far = {"pos": s["pos"] + 100.0, "atomic_numbers": s["atomic_numbers"]}
    both = {"pos": np.vstack([s["pos"], far["pos"]]), "atomic_numbers": list(s["atomic_numbers"]) * 2}
    cut = model.cutoff
    e1 = float(model(structure_to_graph(s, cut))["energy"])
    e2 = float(model(structure_to_graph(far, cut))["energy"])
    assert abs(float(model(structure_to_graph(both, cut))["energy"]) - (e1 + e2)) < 1e-10
    out = model(collate([structure_to_graph(s, cut), structure_to_graph(far, cut)]))
    assert torch.allclose(out["energy"], torch.tensor([e1, e2]), atol=1e-10)
    assert out["node_features"].shape == (10, model.node_feature_dim)
    lone = structure_to_graph({"pos": [[0.0, 0.0, 0.0]], "atomic_numbers": [8]}, cut)
    model.set_atomic_energies([8], [-3.0])
    assert float(model(lone)["energy"]) == -3.0


def test_periodic_lattice_translation_and_stress():
    model = ForceStressOutput(_small(seed=8, cutoff=2.5, int_cutoff=4.0),
                              compute_stress=True).double()
    s = _structure(n=3, spread=3.2)
    s["cell"], s["pbc"] = np.eye(3) * 3.2, [True, True, True]
    out = model(structure_to_graph(s, 4.0))
    assert out["stress"].shape == (1, 3, 3) and torch.isfinite(out["stress"]).all()
    shifted = dict(s, pos=s["pos"] + np.array([[3.2, 0.0, 0.0], [0, 0, 0], [0, 0, 0]]))
    out2 = model(structure_to_graph(shifted, 4.0))
    assert abs(float(out["energy"]) - float(out2["energy"])) < 1e-9
    assert torch.allclose(out["forces"].detach(), out2["forces"].detach(), atol=1e-9)


def test_float32_forward_matches_float64():
    model = _small(seed=9)
    s = _structure(5)
    e64 = float(model.double()(structure_to_graph(s, model.cutoff))["energy"])
    torch.set_default_dtype(torch.float32)
    e32 = float(model.float()(structure_to_graph(s, model.cutoff))["energy"])
    torch.set_default_dtype(torch.float64)
    assert abs(e64 - e32) < 1e-4 * max(1.0, abs(e64))


# scaling factors (paper section 6)

def test_fit_scaling_factors_preserve_variance():
    """After fitting, every factor maps its input variance onto its output on
    the fitting data (factors are fitted in forward order)."""
    model = _small(seed=10, scales=False, direct_forces=True)
    batches = [collate([structure_to_graph(_structure(6, seed=10 * b + i), model.cutoff)
                        for i in range(4)]) for b in range(2)]
    assert not bool(model.scaling_fitted)
    fitted = model.fit_scaling_factors(batches)
    assert bool(model.scaling_fitted) and set(fitted) == set(model.scale_factors())
    assert any(abs(v - 1.0) > 0.05 for v in fitted.values())
    for name, factor in model.scale_factors().items():
        factor.stats.zero_()
        factor.fitting = True
        with torch.no_grad():
            for g in batches:
                model(g)
        factor.fitting = False
        var_ref, var_y, _ = factor.stats.tolist()
        assert var_y == pytest.approx(var_ref, rel=1e-9), name


def test_load_scaling_factors_reference_names(tmp_path):
    import json

    q = _small(scales=False)
    names = list(q.scale_factors())
    assert names[:2] == ["OutBlock_0_sum", "QuadInteraction_1_had_rbf"]
    values = {name: 1.0 + i / 10 for i, name in enumerate(names)}
    path = tmp_path / "scaling_factors.json"
    path.write_text(json.dumps(dict(values, comment="GemNet")))
    q.load_scaling_factors(str(path))
    assert bool(q.scaling_fitted)
    assert float(q.interactions[1].trip.scale_cbf_sum.scale_factor) == values["TripInteraction_2_sum_cbf"]
    t = _small(scales=False, quadruplets=False)
    t.load_scaling_factors(values)                     # the quadruplet entries are ignored
    with pytest.raises(KeyError):
        _small(scales=False, direct_forces=True).load_scaling_factors(values)


def test_trainer_fits_scaling_factors(tmp_path):
    from xnn.common.data import AtomicDataset
    from xnn.common.train import Trainer

    rng = np.random.default_rng(0)
    structures = [dict(_structure(5, seed=i), energy=float(rng.normal()),
                       forces=rng.normal(size=(5, 3))) for i in range(6)]
    cfg = from_dict({"model": {"name": "gemnet", "n_features": 8, "n_interactions": 1,
                               "n_rbf": 3, "cutoff": 3.0,
                               "extra": {"int_cutoff": 4.0, "n_spherical": 2,
                                         "n_edge_features": 8, "n_triplet_features": 4,
                                         "n_quadruplet_features": 4, "n_bilinear_triplet": 4,
                                         "n_bilinear_quadruplet": 4}},
                     "optim": {"epochs": 1},
                     "data": {"batch_size": 3, "val_fraction": 0.0},
                     "output_dir": str(tmp_path)})
    dataset = AtomicDataset(structures, 3.0)
    trainer = Trainer(cfg, dataset)
    core = trainer.module.model
    assert isinstance(core, GemNet) and bool(core.scaling_fitted)
    assert dataset.cutoff == 4.0                        # graphs reach the interaction cutoff
    assert float(core.outputs[0].atom.scale_sum.scale_factor) != 1.0
    trainer.fit()


# deployment and add-ons

@pytest.mark.parametrize("quadruplets", [True, False])
def test_scriptable_and_lammps_export(tmp_path, quadruplets):
    from xnn.common.deploy import export_to_lammps

    model = _small(quadruplets=quadruplets, cutoff=2.5, int_cutoff=4.0).eval()
    s = _structure(n=4, spread=3.2)
    s["cell"], s["pbc"] = np.eye(3) * 3.2, [True, True, True]
    g = structure_to_graph(s, model.cutoff)
    scripted = torch.jit.script(model)
    d = (scripted.node_energy(g.atomic_numbers, g.edge_index, g.edge_vectors())
         - model.node_energy(g.atomic_numbers, g.edge_index, g.edge_vectors()))
    assert d.abs().max() < 1e-12
    path = str(tmp_path / "gemnet_lammps.pt")
    export_to_lammps(model, model.cutoff, path)
    loaded = torch.jit.load(path)
    out = loaded(g.pos, g.edge_index, g.cell_shifts, g.atomic_numbers, g.cell[0])
    ref = ForceStressOutput(model)(g)
    assert abs(float(out["total_energy"]) - float(ref["energy"])) < 1e-8
    assert (out["forces"] - ref["forces"].detach()).abs().max() < 1e-8


def test_direct_forces_refuse_energy_only_paths():
    from xnn.common.models.les import LatentEwald

    model = _small(quadruplets=False, direct_forces=True)
    g = structure_to_graph(_structure(5), model.cutoff)
    with pytest.raises(RuntimeError):
        model.node_energy(g.atomic_numbers, g.edge_index, g.edge_vectors())
    with pytest.raises(ValueError):
        ForceStressOutput(LatentEwald(model))(g)


def test_les_wrapper_and_multihead():
    from xnn.common.finetune import MultiHead, get_atomic_energies, set_atomic_energies
    from xnn.common.models.les import LatentEwald

    g = structure_to_graph(_structure(5), 6.0)
    out = ForceStressOutput(LatentEwald(_small())).double()(g)
    assert torch.isfinite(out["energy"]).all() and out["forces"].shape == (5, 3)
    model = _small(quadruplets=False)
    set_atomic_energies(model, {1: -0.5, 6: -1.0, 8: -2.0})
    assert get_atomic_energies(model) == {1: -0.5, 6: -1.0, 8: -2.0}
    assert MultiHead(model, ["a", "b"])(g)["energy"].shape == (1,)


def test_lora_adapts_the_dense_layers():
    """LoRA wraps the activated dense layers without changing the model until
    trained, and merges back exactly."""
    from xnn.common.finetune import inject_lora, merge_lora

    model = _small(quadruplets=True)
    g = structure_to_graph(_structure(5), model.cutoff)
    e0 = float(model(g)["energy"])
    inject_lora(model, rank=2)
    assert abs(float(model(g)["energy"]) - e0) < 1e-12
    with torch.no_grad():
        for _name, p in model.named_parameters():
            if p.requires_grad:
                p.add_(0.01 * torch.randn_like(p))
    e1 = float(model(g)["energy"])
    assert abs(e1 - e0) > 1e-8
    merge_lora(model)
    assert abs(float(model(g)["energy"]) - e1) < 1e-10


# config

def test_from_config_and_yaml_presets():
    import yaml

    root = os.path.join(os.path.dirname(__file__), "..", "configs", "model")
    for name, quad in (("gemnet.yaml", True), ("gemnet_t.yaml", False)):
        with open(os.path.join(root, name)) as f:
            m = build_model(from_dict({"model": yaml.safe_load(f)}).model)
        assert isinstance(m, GemNet) and m.quadruplets == quad
    cfg = from_dict({"model": {
        "name": "gemnet", "cutoff": 4.0, "n_features": 16, "n_interactions": 1, "n_rbf": 5,
        "extra": {"int_cutoff": 8.0, "n_spherical": 4, "n_edge_features": 12, "p": 6,
                  "direct_forces": True, "forces_coupled": True, "output_init": "zeros",
                  "species": SPECIES, "atomic_energies": [-13.6, -1030.0, -2043.0]}}})
    m = build_model(cfg.model)
    assert m.cutoff == 8.0 and m.envelope.p == 6 and m.direct_forces and m.forces_coupled
    assert m.cbf.zeros.shape == (4, 5) and m.edge_embedding.dense.weight.shape == (12, 2 * 16 + 5)
    assert float(m.atom_ref.weight[6, 0]) == pytest.approx(-1030.0)
    assert float(m.outputs[0].final.weight.abs().max()) == 0.0


def test_reference_key_translation():
    """The reference code's model_kwargs spellings build the same model."""
    kwargs = {"num_spherical": 3, "num_radial": 4, "num_blocks": 2, "emb_size_atom": 16,
              "emb_size_edge": 12, "emb_size_trip": 8, "emb_size_quad": 6, "emb_size_rbf": 5,
              "emb_size_cbf": 5, "emb_size_sbf": 7, "emb_size_bil_trip": 9,
              "emb_size_bil_quad": 10, "num_before_skip": 2, "num_after_skip": 3,
              "num_concat": 1, "num_atom": 1, "triplets_only": False, "direct_forces": False,
              "cutoff": 5.0, "int_cutoff": 10.0, "envelope_exponent": 5, "num_targets": 1,
              "extensive": True, "output_init": "HeOrthogonal", "activation": "swish"}
    m = build_model(from_dict({"model": dict(kwargs, name="gemnet")}).model)
    block = m.interactions[0]
    assert m.n_spherical == 3 and m.n_rbf == 4 and len(m.interactions) == 2
    assert block.trip.bilinear.shape == (5, 8, 9) and block.quad.bilinear.shape == (7, 6, 10)
    assert len(block.before_skip) == 2 and len(block.after_skip) == 3
    assert len(m.outputs[0].atom.residuals) == 1 and m.envelope.p == 5 and m.quadruplets
    t = build_model(from_dict({"model": dict(kwargs, name="gemnet", triplets_only=True)}).model)
    assert not t.quadruplets and t.cutoff == 5.0


# fidelity to the reference implementation (optional)

def _fidelity(mode):
    path = os.environ.get("GEMNET_UPSTREAM_PATH")
    if not path or not os.path.isdir(os.path.join(path, "gemnet")):
        pytest.skip("set GEMNET_UPSTREAM_PATH to a checkout of TUM-DAML/gemnet_pytorch")
    pytest.importorskip("sympy")
    import subprocess
    import sys
    script = os.path.join(os.path.dirname(__file__), "gemnet_fidelity.py")
    res = subprocess.run([sys.executable, script, path, mode], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    return float(res.stdout.strip().splitlines()[-1])


def test_fidelity_vs_reference_random_weights():
    """Transplanted random weights (all four variants, float64): energies and
    forces agree with the reference code to float64 round-off."""
    assert _fidelity("random") < 1e-10


def test_fidelity_vs_reference_pretrained():
    """The published GemNet-Q and GemNet-T weights give the same energies and
    forces in xnn (float32)."""
    assert _fidelity("pretrained") < 1e-3


def _check_neighbor_sums(kernels, device):
    """The fused neighbor sums against the dense sum, and every first and second derivative."""
    gen = torch.Generator().manual_seed(7)
    n_rows, n_items, n_out, a, f = 6, 17, 5, 3, 4

    def rand(*shape):
        return torch.randn(*shape, dtype=torch.float64, generator=gen).to(device).requires_grad_(True)

    x, s = rand(n_rows, f), rand(n_items, a)
    in_idx = torch.randint(0, n_rows, (n_items,), generator=gen).to(device)
    out_idx = torch.sort(torch.randint(0, n_out, (n_items,), generator=gen))[0].to(device)

    def fn(x, s):
        return kernels.neighbor_sums(x, s, in_idx, out_idx, n_out)

    with torch.no_grad():
        expect = torch.zeros(n_out, a, f, dtype=torch.float64, device=device).index_add_(
            0, out_idx, s[:, :, None] * x[in_idx][:, None, :])
        assert torch.allclose(fn(x, s), expect, rtol=0, atol=1e-13)
    assert torch.autograd.gradcheck(fn, (x, s))
    assert torch.autograd.gradgradcheck(fn, (x, s))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_fast_neighbor_sums_gradients():
    """Values and every first and second derivative of the fused neighbor sums."""
    pytest.importorskip("triton")
    from xnn.gnn.fast import gemnet_kernels
    _check_neighbor_sums(gemnet_kernels, "cuda")


def test_fast_neighbor_sums_gradients_interpreted():
    """The same under Triton's interpreter (chosen by conftest.py on machines without CUDA)."""
    if os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("runs where Triton's interpreter is on (no CUDA device)")
    pytest.importorskip("triton")
    from xnn.gnn.fast import gemnet_kernels
    _check_neighbor_sums(gemnet_kernels, "cpu")


def test_fast_support_needs_cuda():
    from xnn.gnn.fast.gemnet import supported
    assert not supported(torch.device("cpu"), torch.float64)


def _fast_batch(dtype, cutoff):
    graphs = [_graph(_structure(n=9, seed=3), cutoff), _graph(_structure(n=6, seed=4), cutoff)]
    batch = collate(graphs).to("cuda")
    batch.pos = batch.pos.to(dtype)
    return batch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("quadruplets", [False, True])
def test_fast_path_parity(dtype, quadruplets):
    """The fused bilinear layers give the reference energies and forces (GemNet-T and -Q)."""
    pytest.importorskip("triton")
    model = _small(seed=5, quadruplets=quadruplets).to("cuda", dtype).eval()
    wrapped = ForceStressOutput(model).eval()
    out = {}
    for use_fast in (False, True):
        model.set_use_fast(use_fast)
        res = wrapped(_fast_batch(dtype, model.cutoff))
        assert model.interactions[0].trip.fast_active == use_fast
        out[use_fast] = {k: res[k].detach().double() for k in ("energy", "forces")}
    tol = 1e-12 if dtype == torch.float64 else 1e-5
    for key, ref in out[False].items():
        err = (out[True][key] - ref).abs().max() / ref.abs().max().clamp(min=1.0)
        assert err < tol, (key, float(err))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("quadruplets", [False, True])
def test_fast_path_training(dtype, quadruplets):
    """Force training through the fused bilinear layers gives the reference's weight gradients."""
    pytest.importorskip("triton")
    model = _small(seed=6, quadruplets=quadruplets).to("cuda", dtype)
    grads = {}
    for use_fast in (False, True):
        model.set_use_fast(use_fast).train()
        model.zero_grad(set_to_none=True)
        out = ForceStressOutput(model).train()(_fast_batch(dtype, model.cutoff))
        assert model.interactions[0].trip.fast_active == use_fast
        (out["energy"].pow(2).sum() + out["forces"].pow(2).sum()).backward()
        grads[use_fast] = {name: p.grad.detach().double().clone() for name, p in model.named_parameters()
                           if p.grad is not None}
    assert grads[False].keys() == grads[True].keys()
    tol = 1e-11 if dtype == torch.float64 else 1e-4
    for name, g in grads[False].items():
        err = (grads[True][name] - g).abs().max() / g.abs().max().clamp_min(1e-30)
        assert err < tol, (name, float(err))
