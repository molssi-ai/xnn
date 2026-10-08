"""Spherical CNN (Cohen et al., ICLR 2018): manuscript fidelity.

The harmonic analysis is checked against its defining properties (Wigner
d-matrices, the representation property of the D-functions for the ZYZ
Euler composition, exact quadrature of the grids, round trips of the
generalized Fourier transforms), the correlation layers against the direct
evaluation of eq 4 and eq 6 for point filters, the rotation operators
against exact grid rotations, and the equivariance of every layer and of
the potential; the spherical-signal featurizer against a loop reference.
``tests/s2cnn_parity.py`` compares with the reference implementation when
it is importable.
"""
import math

import numpy as np
import pytest
import torch

from xnn.common.config import from_dict
from xnn.common.data import collate, structure_to_graph
from xnn.common.models import ForceStressOutput, available_models, build_model
from xnn.cnn.featurizers import SphericalGrid
from xnn.cnn.models import (S2Convolution, S2Transform, SO3Convolution, SO3Transform,
                            SphericalCNN, SphericalResBlock, euler_to_matrix, matrix_to_euler,
                            quadrature_weights, s2_grid_points, s2_near_identity_grid,
                            s2_equatorial_grid, s2_rotate, so3_integrate, so3_near_identity_grid,
                            so3_equatorial_grid, so3_rotate, wigner_D, wigner_d)
from xnn.cnn.models.spherical import default_schedule, so3_alphas, so3_betas

SPECIES = [1, 6, 8]


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _random_angles(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(n, generator=g, dtype=torch.float64) * 2 * math.pi
    b = torch.acos(torch.rand(n, generator=g, dtype=torch.float64) * 2 - 1)
    c = torch.rand(n, generator=g, dtype=torch.float64) * 2 * math.pi
    return a, b, c


# harmonic analysis

def test_wigner_d_properties():
    beta1, beta2 = torch.tensor(0.7), torch.tensor(1.9)
    for l in range(5):
        d0 = wigner_d(l, torch.tensor(0.0))
        assert torch.allclose(d0, torch.eye(2 * l + 1, dtype=torch.float64), atol=1e-14)
        d = wigner_d(l, beta1)
        assert torch.allclose(d @ d.T, torch.eye(2 * l + 1, dtype=torch.float64), atol=1e-13)
        assert torch.allclose(d @ wigner_d(l, beta2), wigner_d(l, beta1 + beta2), atol=1e-13)
    c, s = math.cos(0.7), math.sin(0.7)
    ref = torch.tensor([[(1 + c) / 2, s / math.sqrt(2), (1 - c) / 2],
                        [-s / math.sqrt(2), c, s / math.sqrt(2)],
                        [(1 - c) / 2, -s / math.sqrt(2), (1 + c) / 2]], dtype=torch.float64)
    assert torch.allclose(wigner_d(1, beta1), ref, atol=1e-14)         # m, n = -1, 0, 1


def test_wigner_D_is_a_representation_of_the_euler_composition():
    """D(R1) D(R2) = D(R1 R2) with R = Z(alpha) Y(beta) Z(gamma); D unitary."""
    a1, b1, c1 = _random_angles(3, 1)
    a2, b2, c2 = _random_angles(3, 2)
    for i in range(3):
        R = euler_to_matrix(a1[i], b1[i], c1[i]) @ euler_to_matrix(a2[i], b2[i], c2[i])
        a, b, c = matrix_to_euler(R)
        assert torch.allclose(euler_to_matrix(a, b, c), R, atol=1e-12)
        for l in range(4):
            D1, D2 = wigner_D(l, a1[i], b1[i], c1[i]), wigner_D(l, a2[i], b2[i], c2[i])
            assert torch.allclose(D1 @ D2, wigner_D(l, a, b, c), atol=1e-12)
            eye = torch.eye(2 * l + 1, dtype=torch.complex128)
            assert torch.allclose(D1 @ D1.conj().T, eye, atol=1e-12)
    for ag, bg, cg in [(0.4, 0.0, 1.3), (0.4, math.pi, 1.3), (2.5, 0.0, -0.7)]:   # gimbal lock
        Rg = euler_to_matrix(ag, bg, cg)
        assert torch.allclose(euler_to_matrix(*matrix_to_euler(Rg)), Rg, atol=1e-12)


def test_quadrature_integrates_the_basis_exactly():
    """int D^l_{mn} conj(D^l'_{m'n'}) dR = delta / (2l + 1) on the grid (eq 19)."""
    b = 4
    w = quadrature_weights(b)
    assert float(w.sum() * (2 * b) ** 2) == pytest.approx(1.0, abs=1e-14)
    beta, alpha = so3_betas(b), so3_alphas(b)
    bb, aa, gg = torch.meshgrid(beta, alpha, alpha, indexing="ij")
    blocks = {l: wigner_D(l, aa, bb, gg) for l in range(b)}                 # (2b, 2b, 2b, 2l+1, 2l+1)
    for l in range(b):
        for l2 in range(b):
            gram = torch.einsum("jkl,jklmn,jklpq->mnpq", w[:, None, None].expand(2 * b, 2 * b, 2 * b).to(torch.complex128),
                                blocks[l], blocks[l2].conj())
            if l == l2:
                ref = torch.einsum("mp,nq->mnpq", torch.eye(2 * l + 1), torch.eye(2 * l + 1)) / (2 * l + 1)
                assert torch.allclose(gram, ref.to(gram.dtype), atol=1e-13)
            else:
                assert float(gram.abs().max()) < 1e-13


def test_s2_transform_round_trip_and_harmonics():
    b_grid, b_spec = 6, 4
    tr = S2Transform(b_grid, b_spec)
    g = torch.Generator().manual_seed(0)
    spec = torch.randn(2, b_spec, 2 * b_spec - 1, generator=g, dtype=torch.complex128)
    mask = torch.zeros(b_spec, 2 * b_spec - 1, dtype=torch.bool)
    for l in range(b_spec):
        mask[l, b_spec - 1 - l:b_spec + l] = True
    spec = spec * mask
    x = tr.synthesize(spec, real=False)
    assert x.shape == (2, 2 * b_grid, 2 * b_grid)
    assert torch.allclose(tr.analyze(x), spec, atol=1e-12)
    # a sampled harmonic Y^l_m = D^l_{m0} transforms to 1 / (2l + 1) at (l, m)
    beta, alpha = so3_betas(b_grid), so3_alphas(b_grid)
    bb, aa = torch.meshgrid(beta, alpha, indexing="ij")
    for l, m in [(0, 0), (2, -1), (3, 3)]:
        y = wigner_D(l, aa, bb, torch.zeros_like(aa))[..., m + l, l]
        f_hat = tr.analyze(y)
        ref = torch.zeros_like(f_hat)
        ref[l, m + b_spec - 1] = 1.0 / (2 * l + 1)
        assert torch.allclose(f_hat, ref, atol=1e-12)
    with pytest.raises(ValueError):
        S2Transform(3, 4)


def test_so3_transform_round_trip():
    b_grid, b_spec = 5, 3
    tr = SO3Transform(b_grid, b_spec)
    g = torch.Generator().manual_seed(1)
    spec = torch.randn(3, b_spec, 2 * b_spec - 1, 2 * b_spec - 1, generator=g, dtype=torch.complex128)
    for l in range(b_spec):
        o = b_spec - 1 - l
        keep = torch.zeros(2 * b_spec - 1, 2 * b_spec - 1, dtype=torch.bool)
        keep[o:o + 2 * l + 1, o:o + 2 * l + 1] = True
        spec[:, l] = spec[:, l] * keep
    x = tr.synthesize(spec, real=False)
    assert x.shape == (3, 2 * b_grid, 2 * b_grid, 2 * b_grid)
    assert torch.allclose(tr.analyze(x), spec, atol=1e-12)
    # a real signal synthesized from a spectrum with the reality symmetry stays real
    sym = spec.clone()
    for l in range(b_spec):
        o = b_spec - 1 - l
        block = spec[:, l, o:o + 2 * l + 1, o:o + 2 * l + 1]
        sign = torch.tensor([(-1) ** (m - n) for m in range(-l, l + 1) for n in range(-l, l + 1)],
                            dtype=torch.float64).reshape(2 * l + 1, 2 * l + 1)
        sym[:, l, o:o + 2 * l + 1, o:o + 2 * l + 1] = 0.5 * (block + sign * block.flip(-1).flip(-2).conj())
    out = tr.synthesize(sym, real=False)
    assert float(out.imag.abs().max()) < 1e-12


def _random_bandlimited_s2(b_grid, b_spec, batch, channels, seed=0):
    """Real spherical signals on the b_grid grid with no content above b_spec."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, channels, 2 * b_grid, 2 * b_grid, generator=g, dtype=torch.float64)
    tr = S2Transform(b_grid, b_spec)
    return tr.synthesize(tr.analyze(x))


def _random_bandlimited_so3(b_grid, b_spec, batch, channels, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, channels, 2 * b_grid, 2 * b_grid, 2 * b_grid, generator=g, dtype=torch.float64)
    tr = SO3Transform(b_grid, b_spec)
    return tr.synthesize(tr.analyze(x))


def _eval_s2(f_hat, points):
    """f(x) = sum_l (2l+1) sum_m f^l_m D^l_{m0}(x) at unit vectors (..., 3)."""
    b = f_hat.shape[-2]
    beta = torch.acos(points[..., 2].clamp(-1, 1))
    alpha = torch.atan2(points[..., 1], points[..., 0])
    out = torch.zeros(f_hat.shape[:-2] + points.shape[:-1], dtype=torch.complex128)
    for l in range(b):
        D = wigner_D(l, alpha, beta, torch.zeros_like(alpha))[..., :, l]   # (..., 2l+1): D_{m0}
        out = out + (2 * l + 1) * torch.einsum("...m,pm->...p", f_hat[..., l, b - 1 - l:b + l], D.reshape(-1, 2 * l + 1)).reshape(out.shape)
    return out


def _eval_so3(f_hat, matrices):
    """f(R) = sum_l (2l+1) sum_{mn} f^l_{mn} D^l_{mn}(R) at rotation matrices (P, 3, 3)."""
    b = f_hat.shape[-3]
    angles = [matrix_to_euler(R) for R in matrices]
    a = torch.tensor([v[0] for v in angles])
    be = torch.tensor([v[1] for v in angles])
    c = torch.tensor([v[2] for v in angles])
    out = torch.zeros(f_hat.shape[:-3] + (len(matrices),), dtype=torch.complex128)
    for l in range(b):
        D = wigner_D(l, a, be, c)                                              # (P, 2l+1, 2l+1)
        o = b - 1 - l
        out = out + (2 * l + 1) * torch.einsum("...mn,pmn->...p", f_hat[..., l, o:o + 2 * l + 1, o:o + 2 * l + 1], D)
    return out


def test_s2_correlation_matches_direct_evaluation():
    """eq 4 for a point filter: [psi * f](R) = 2b sum_i K_i f(R x_i), at every
    grid rotation of the output, for bandlimited f (exact)."""
    torch.manual_seed(0)
    b_in, b_out, n_in, n_out = 5, 3, 2, 3
    conv = S2Convolution(n_in, n_out, b_in, b_out, s2_near_identity_grid(math.pi / 6, 4, 2))
    with torch.no_grad():
        conv.bias.normal_()
    f = _random_bandlimited_s2(b_in, b_out, 2, n_in)
    y = conv(f)
    f_hat = S2Transform(b_in, b_out).analyze(f)                              # (B, i, l, m)
    beta, alpha = so3_betas(b_out), so3_alphas(b_out)
    bb, aa, gg = torch.meshgrid(beta, alpha, alpha, indexing="ij")
    Rs = torch.stack([euler_to_matrix(a, be, c) for a, be, c in zip(aa.flatten(), bb.flatten(), gg.flatten())])
    pts = conv.points
    x_i = torch.stack([torch.sin(pts[:, 0]) * torch.cos(pts[:, 1]), torch.sin(pts[:, 0]) * torch.sin(pts[:, 1]),
                       torch.cos(pts[:, 0])], dim=1)                          # (P, 3)
    moved = torch.einsum("rab,pb->rpa", Rs, x_i)                               # (R, P, 3)
    values = _eval_s2(f_hat, moved).real                                       # (B, i, R, P)
    k = conv.kernel * conv.scaling
    ref = 2 * b_out * torch.einsum("birp,iop->bor", values, k) + conv.bias.reshape(1, n_out, 1)
    assert torch.allclose(y.reshape(2, n_out, -1), ref, atol=1e-11)


def test_so3_correlation_matches_direct_evaluation():
    """eq 6 for a point filter: [psi * f](R) = sum_i K_i f(R Q_i)."""
    torch.manual_seed(0)
    b_in, b_out, n_in, n_out = 4, 3, 2, 2
    conv = SO3Convolution(n_in, n_out, b_in, b_out, so3_near_identity_grid(math.pi / 6, math.pi, 3, 2, 2))
    with torch.no_grad():
        conv.bias.normal_()
    f = _random_bandlimited_so3(b_in, b_out, 2, n_in)
    y = conv(f)
    f_hat = SO3Transform(b_in, b_out).analyze(f)
    beta, alpha = so3_betas(b_out), so3_alphas(b_out)
    bb, aa, gg = torch.meshgrid(beta, alpha, alpha, indexing="ij")
    Rs = [euler_to_matrix(a, be, c) for a, be, c in zip(aa.flatten(), bb.flatten(), gg.flatten())]
    Qs = [euler_to_matrix(p[1], p[0], p[2]) for p in conv.points]
    values = torch.stack([_eval_so3(f_hat, torch.stack([R @ Q for R in Rs])).real for Q in Qs], dim=-1)  # (B, i, R, P)
    k = conv.kernel * conv.scaling
    ref = torch.einsum("birp,iop->bor", values, k) + conv.bias.reshape(1, n_out, 1)
    assert torch.allclose(y.reshape(2, n_out, -1), ref, atol=1e-11)


def test_rotation_operators():
    """L_R f for R = Z(phi) with phi a grid angle is an exact roll along alpha;
    rotations compose; the integral is invariant."""
    b = 5
    f = _random_bandlimited_s2(b, b, 2, 3)
    phi = 3 * 2 * math.pi / (2 * b)
    assert torch.allclose(s2_rotate(f, phi, 0.0, 0.0), torch.roll(f, 3, dims=-1), atol=1e-12)
    g = _random_bandlimited_so3(b, b, 2, 2)
    assert torch.allclose(so3_rotate(g, phi, 0.0, 0.0), torch.roll(g, 3, dims=-2), atol=1e-12)
    a1, b1, c1 = (float(v) for v in torch.tensor([0.4, 1.1, 2.3]))
    a2, b2, c2 = (float(v) for v in torch.tensor([2.9, 0.6, 5.0]))
    a, be, c = matrix_to_euler(euler_to_matrix(a1, b1, c1) @ euler_to_matrix(a2, b2, c2))
    assert torch.allclose(so3_rotate(so3_rotate(g, a2, b2, c2), a1, b1, c1), so3_rotate(g, a, be, c), atol=1e-11)
    assert torch.allclose(s2_rotate(s2_rotate(f, a2, b2, c2), a1, b1, c1), s2_rotate(f, a, be, c), atol=1e-11)
    assert torch.allclose(so3_integrate(so3_rotate(g, a1, b1, c1)), so3_integrate(g), atol=1e-12)
    assert float(so3_integrate(torch.ones(2 * b, 2 * b, 2 * b))) == pytest.approx(1.0, abs=1e-13)


def test_layers_are_equivariant():
    """conv(L_R f) = L_R conv(f) exactly for bandlimited inputs (eq 7)."""
    torch.manual_seed(0)
    b_in, b_out = 5, 3
    s2 = S2Convolution(2, 3, b_in, b_out, s2_near_identity_grid(math.pi / 6, 4, 2))
    so3 = SO3Convolution(3, 2, b_out, b_out, so3_near_identity_grid(math.pi / 6, math.pi, 4, 1, 2))
    block = SphericalResBlock(2, 3, b_in, b_out, s2_input=True, n_alpha=4, normalization=None).eval()
    f = _random_bandlimited_s2(b_in, b_out, 2, 2)
    a, be, c = 1.3, 0.8, 4.2
    y = s2(f)
    assert torch.allclose(s2(s2_rotate(f, a, be, c)), so3_rotate(y, a, be, c), atol=1e-11)
    z = so3(torch.relu(y))
    y_rel = torch.relu(y)
    # relu breaks the bandlimit; compare the layer on a bandlimited version of its input
    tr = SO3Transform(b_out, b_out)
    y_rel = tr.synthesize(tr.analyze(y_rel))
    assert torch.allclose(so3(so3_rotate(y_rel, a, be, c)), so3_rotate(so3(y_rel), a, be, c), atol=1e-11)
    assert torch.allclose(so3_integrate(so3(so3_rotate(y_rel, a, be, c))), so3_integrate(so3(y_rel)), atol=1e-11)
    assert block(f).shape == (2, 3, 2 * b_out, 2 * b_out, 2 * b_out)
    del z


def test_grids_and_schedule():
    assert s2_near_identity_grid(math.pi / 8, 8, 3).shape == (24, 2)
    assert so3_near_identity_grid(math.pi / 8, 2 * math.pi, 8, 3, 2).shape == (48, 3)
    assert s2_equatorial_grid(0.0, 32, 1).shape == (32, 2)
    assert torch.allclose(s2_equatorial_grid(0.0, 32, 1)[:, 0], torch.full((32,), math.pi / 2, dtype=torch.float64))
    assert so3_equatorial_grid(0.0, math.pi / 8, 32, 1, 2).shape == (64, 3)
    pts = s2_grid_points(3)
    assert pts.shape == (6, 6, 3) and torch.allclose(pts.norm(dim=-1), torch.ones(6, 6, dtype=torch.float64))
    assert default_schedule(160, 5, 10) == ([32, 64, 96, 128, 160], [10, 8, 6, 4, 2])


# the featurizer

def _structure(n=6, seed=0, box=3.0):
    rng = np.random.default_rng(seed)
    return {"pos": rng.uniform(0, box, (n, 3)), "atomic_numbers": ([1, 6, 8] * n)[:n]}


def _graph(s, cutoff=10.0):
    return structure_to_graph(s, cutoff)


def _reference_signals(feat, s):
    pos = torch.as_tensor(s["pos"], dtype=torch.float64)
    Z = list(s["atomic_numbers"])
    n, b = len(Z), feat.bandwidth
    pts = s2_grid_points(b) * feat.radius
    out = torch.zeros(n, feat.n_channels, 2 * b, 2 * b, dtype=torch.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rel = pos[j] - pos[i]
            d = float(rel.norm())
            if d >= feat.cutoff:
                continue
            w = 0.5 * (math.cos(math.pi * d / feat.cutoff) + 1.0) if feat.envelope is not None else 1.0
            out[i, feat.species.index(Z[j])] += w * Z[i] * Z[j] / (pts - rel).norm(dim=-1) ** feat.exponent
    return out


@pytest.mark.parametrize("exponent,cutoff_fn", [(1.0, None), (2.0, "cosine")])
def test_spherical_grid_matches_definition(exponent, cutoff_fn):
    feat = SphericalGrid(SPECIES, cutoff=4.0, radius=0.4, bandwidth=4, exponent=exponent, cutoff_fn=cutoff_fn)
    s = _structure(7, seed=1)
    sig = feat(_graph(s, 4.0))
    assert sig.shape == (7, 3, 8, 8)
    assert torch.allclose(sig, _reference_signals(feat, s), atol=1e-12)


def test_spherical_grid_rotation_and_invariances():
    feat = SphericalGrid(SPECIES, cutoff=10.0, radius=0.4, bandwidth=4)
    s = _structure(6, seed=2)
    ref = feat(_graph(s))
    # a rotation about z by a grid angle permutes the alpha samples
    k = 3
    R = euler_to_matrix(k * 2 * math.pi / 8, 0.0, 0.0).numpy()
    assert torch.allclose(feat(_graph(dict(s, pos=np.asarray(s["pos"]) @ R.T))), torch.roll(ref, k, dims=-1), atol=1e-12)
    shifted = dict(s, pos=np.asarray(s["pos"]) + np.array([1.0, -2.0, 0.5]))
    assert torch.allclose(feat(_graph(shifted)), ref, atol=1e-12)
    with pytest.raises(ValueError, match="no potential channel"):
        SphericalGrid([1, 6], bandwidth=3)(_graph(_structure()))
    assert feat.output_dim == 3 and feat.grid_size == 8


# the potential

def _small_model(**kw):
    torch.manual_seed(0)
    kw.setdefault("species", SPECIES)
    kw.setdefault("cutoff", 10.0)
    kw.setdefault("radius", 0.4)
    kw.setdefault("bandwidth", 4)
    kw.setdefault("features", [4, 6])
    kw.setdefault("bandwidths", [4, 2])
    kw.setdefault("n_alpha", 4)
    kw.setdefault("normalization", None)
    m = SphericalCNN(**kw)
    torch.nn.init.normal_(m.readout[-1].weight)
    torch.nn.init.normal_(m.readout[-1].bias)
    return m.eval()


def test_registered_and_paper_defaults():
    assert "s2cnn" in available_models()
    m = SphericalCNN()
    assert m.features == [20, 40, 60, 80, 160] and m.bandwidths == [10, 8, 6, 4, 2]
    assert m.species == [1, 6, 7, 8, 16] and m.featurizer.radius == 0.48
    assert m.blocks[0].norm1 is not None and isinstance(m.blocks[0].conv1, S2Convolution)
    assert all(isinstance(b.conv1, SO3Convolution) for b in m.blocks[1:])
    cfg = from_dict({"model": {"name": "s2cnn", "cutoff": 6.0, "n_features": 8, "n_interactions": 2,
                               "extra": {"species": ["H", "C", "O"], "bandwidth": 4, "radius": 0.4,
                                         "n_alpha": 4, "set_readout": [6, 5, 4],
                                         "atomic_energies": [-0.5, -1.0, -2.0]}}})
    m = build_model(cfg.model)
    assert isinstance(m, SphericalCNN) and m.features == [4, 8] and m.bandwidths == [4, 2]
    assert m.set_readout is not None and float(m.atom_ref.weight[8, 0]) == -2.0
    out = m.eval()(_graph(_structure(), 6.0))
    assert out["energy"].shape == (1,) and out["node_features"].shape == (6, 8)
    assert torch.allclose(out["node_energy"].sum(), out["energy"].to(out["node_energy"].dtype))
    with pytest.raises(ValueError):
        SphericalCNN(SPECIES, bandwidth=4, features=[4, 4], bandwidths=[2, 4])


def test_reference_spellings_translate():
    cfg = from_dict({"model": {"name": "s2cnn", "cutoff": 6.0, "n_interactions": 2,
                               "extra": {"species": [1, 6, 8], "b_in": 4, "nfeature_out": 6, "n_alpha": 4}}})
    m = build_model(cfg.model)
    assert m.bandwidth == 4 and m.features[-1] == 6


def test_fresh_model_predicts_shift():
    torch.manual_seed(0)
    m = SphericalCNN(SPECIES, bandwidth=4, radius=0.4, features=[4, 4], bandwidths=[4, 2], n_alpha=4,
                     normalization=None, energy_shift=-1.5, atomic_energies=[-0.5, -1.0, -2.0]).eval()
    s = _structure()
    ref = -1.5 * 6 + sum({1: -0.5, 6: -1.0, 8: -2.0}[z] for z in s["atomic_numbers"])
    assert float(m(_graph(s))["energy"]) == pytest.approx(ref)


def test_energy_invariant_under_grid_rotations():
    """Rotations about z by multiples of 2 pi / 2b of the coarsest grid permute
    every grid of the network: exact invariance, co-rotating forces. (A
    rotation of the input grid alone is not exact once a block reduces the
    bandwidth and applies ReLU on the coarser grid: that is the discretization
    error of Sec. 5.1 of the paper.)"""
    m = ForceStressOutput(_small_model())
    s = _structure(6, seed=3)
    out = m(_graph(s))
    e0, f0 = out["energy"].detach(), out["forces"].detach()
    assert float(f0.abs().max()) > 1e-6
    b_min = min(m.model.bandwidths)
    for k in range(1, 2 * b_min):
        R = euler_to_matrix(k * 2 * math.pi / (2 * b_min), 0.0, 0.0).numpy()
        rot = m(_graph(dict(s, pos=np.asarray(s["pos"]) @ R.T)))
        assert torch.allclose(rot["energy"].detach(), e0, rtol=0.0, atol=1e-10 * float(e0.abs()))
        assert torch.allclose(rot["forces"].detach(), f0 @ torch.tensor(R).T, rtol=0.0,
                              atol=1e-10 * float(f0.abs().max()))
    # with every block at the input bandwidth every grid rotation is exact
    full = ForceStressOutput(_small_model(bandwidths=[4, 4]))
    e_full = full(_graph(s))["energy"].detach()
    for k in range(1, 8):
        R = euler_to_matrix(k * 2 * math.pi / 8, 0.0, 0.0).numpy()
        e_rot = full(_graph(dict(s, pos=np.asarray(s["pos"]) @ R.T)))["energy"].detach()
        assert torch.allclose(e_rot, e_full, rtol=0.0, atol=1e-10 * float(e_full.abs()))


def test_approximate_invariance_under_arbitrary_rotations():
    m = _small_model()
    s = _structure(6, seed=4)
    e0 = float(m(_graph(s))["energy"])
    a, b, c = _random_angles(6, 5)
    es = [float(m(_graph(dict(s, pos=np.asarray(s["pos"]) @ euler_to_matrix(a[i], b[i], c[i]).numpy().T)))["energy"])
          for i in range(6)]
    spread = float(np.std(es + [e0]))
    scale = float(np.std([float(m(_graph(_structure(6, seed=10 + k)))["energy"]) for k in range(6)]))
    assert spread < 0.1 * scale


def test_translation_permutation_batch():
    m = _small_model()
    s = _structure(6, seed=5)
    e0 = m(_graph(s))["energy"]
    shifted = dict(s, pos=np.asarray(s["pos"]) + np.array([1.3, -0.7, 2.1]))
    assert torch.allclose(m(_graph(shifted))["energy"], e0, atol=1e-10)
    perm = np.random.default_rng(0).permutation(6)
    permuted = {"pos": np.asarray(s["pos"])[perm], "atomic_numbers": [s["atomic_numbers"][i] for i in perm]}
    assert torch.allclose(m(_graph(permuted))["energy"], e0, atol=1e-10)
    graphs = [_graph(_structure(n, seed)) for n, seed in [(5, 1), (7, 2), (4, 3)]]
    e_single = torch.cat([m(g)["energy"] for g in graphs])
    assert torch.allclose(m(collate(graphs))["energy"], e_single, atol=1e-10)


def test_forces_match_finite_differences():
    m = ForceStressOutput(_small_model())
    s = _structure(5, seed=7)
    forces = m(_graph(s))["forces"].detach()
    pos = np.asarray(s["pos"])
    h = 1e-5
    for i, a in [(0, 0), (2, 1), (4, 2)]:
        plus, minus = pos.copy(), pos.copy()
        plus[i, a] += h
        minus[i, a] -= h
        e_plus = float(m(_graph(dict(s, pos=plus)))["energy"])
        e_minus = float(m(_graph(dict(s, pos=minus)))["energy"])
        assert float(forces[i, a]) == pytest.approx(-(e_plus - e_minus) / (2 * h), abs=1e-6)


def test_train_step_energy_only():
    """Energy-only training (the QM7 setting: no forces) runs through the trainer."""
    from xnn.common.config import Config
    from xnn.common.data import AtomicDataset
    from xnn.common.train import Trainer

    rng = np.random.default_rng(0)
    structs = []
    for _ in range(8):
        s = _structure(n=4, seed=int(rng.integers(1 << 30)))
        s["energy"] = float(rng.normal())
        structs.append(s)
    cfg = Config()
    cfg.model.name = "s2cnn"
    cfg.model.cutoff = 6.0
    cfg.model.n_features = 4
    cfg.model.n_interactions = 2
    cfg.model.extra = {"species": SPECIES, "bandwidth": 3, "radius": 0.4, "n_alpha": 4,
                       "set_readout": [4, 4, 4]}
    cfg.optim.epochs = 2
    cfg.optim.force_weight = 0.0
    cfg.data.batch_size = 4
    cfg.data.val_fraction = 0.25
    cfg.device = "cpu"
    Trainer(cfg, AtomicDataset(structs, cfg.model.cutoff)).fit()


# the reference implementation

def test_reference_parity():
    pytest.importorskip("s2cnn")
    pytest.importorskip("lie_learn")
    from s2cnn_parity import compare_all
    for dtype, tol in ((torch.float32, 1e-5), (torch.float64, 1e-11)):
        worst = compare_all(dtype)
        assert max(worst.values()) < tol, worst
