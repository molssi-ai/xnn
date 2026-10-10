"""SpookyNet tests: manuscript fidelity (Nat. Commun. 2021), electronic states,
nonlocality, invariances, smoothness, batching, TorchScript, config handling
and (optionally) parity with the reference code.

The centerpiece is ``_reference``: an independent, loop-based implementation
of the paper's equations (eqs 1-27, with the dispersion term taken from the
:class:`~xnn.common.models.d4.DFTD4` building blocks) that reuses only the
model's *parameters*.
"""
import math
import os

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from xnn.common.config import from_dict
from xnn.common.data import collate, structure_to_graph
from xnn.common.models import ForceStressOutput, available_models, build_model
from xnn.common.models.d4 import DFTD4
from xnn.common.models.dispersion import BOHR, HARTREE
from xnn.common.models.electrostatics import COULOMB_CONSTANT
from xnn.hybrid.featurizers import ExponentialBernsteinRBF
from xnn.hybrid.models.spookynet import (SpookyNet, electron_configurations,
                                      orthogonal_random_features, smooth_switch)

CUTOFF = 4.0


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _structure(n=7, seed=0, spread=3.2, min_dist=0.9, z=(1, 6, 7, 8)):
    rng = np.random.default_rng(seed)
    pos = []
    while len(pos) < n:
        p = rng.uniform(0, spread, 3)
        if all(np.linalg.norm(p - q) > min_dist for q in pos):
            pos.append(p)
    return {"pos": np.array(pos), "atomic_numbers": [z[i % len(z)] for i in range(n)]}


def _graph(s, cutoff=CUTOFF, **labels):
    return structure_to_graph({**s, **labels}, cutoff)


def _randomize(model, seed=0):
    """Random values for every learned parameter (many start at zero)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.startswith(("repulsion.", "dispersion.")) or name.endswith("gamma_raw"):
                continue
            noise = torch.randn(p.shape, generator=g, dtype=p.dtype)
            if name.endswith(".alpha"):
                p.copy_(1.0 + 0.1 * noise)
            elif name.endswith(".beta"):
                p.copy_(1.702 + 0.1 * noise)
            elif p.dim() == 2 and "embedding.element" not in name and "_ref" not in name:
                p.copy_(noise / math.sqrt(p.shape[1]))
            else:
                p.copy_(0.3 * noise)
    return model


def _small(seed=0, **kw):
    kw.setdefault("n_features", 8)
    kw.setdefault("n_interactions", 2)
    kw.setdefault("n_rbf", 6)
    kw.setdefault("cutoff", CUTOFF)
    kw.setdefault("seed", seed)
    return _randomize(SpookyNet(**kw), seed)


# the independent reference of the paper's equations

def _lin(layer, x):
    y = x @ layer.weight.T
    return y if layer.bias is None else y + layer.bias


def _silu(act, x):
    return act.alpha * x / (1.0 + torch.exp(-act.beta * x))                  # eq 6


def _residual(block, x):
    return x + _lin(block.linear2, _silu(block.activation2,
                                         _lin(block.linear1, _silu(block.activation1, x))))


def _resmlp(m, x):
    for block in m.residual:
        x = _residual(block, x)
    return _lin(m.linear, _silu(m.activation, x))                              # eq 8


def _real_harmonic(l, m, v):
    """Eq 17 for a unit vector: real spherical harmonics without normalization."""
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    am = abs(m)
    pi_lm = sum((-1) ** p * 2.0 ** -l * math.comb(l, p) * math.comb(2 * l - 2 * p, l)
                * math.factorial(l - 2 * p) / math.factorial(l - 2 * p - am)
                * z ** (l - 2 * p - am) for p in range((l - am) // 2 + 1))
    pi_lm *= math.sqrt(math.factorial(l - am) / math.factorial(l + am))
    if m == 0:
        return pi_lm
    trig = math.sin if m < 0 else math.cos
    ab = sum(math.comb(am, p) * x ** p * y ** (am - p) * trig((am - p) * math.pi / 2)
             for p in range(am + 1))
    return math.sqrt(2.0) * pi_lm * ab


def _reference(model, Z, pos, Q=0.0, S=0.0):
    """Total energy and charges of one structure from the paper's equations."""
    pos = torch.as_tensor(pos, dtype=torch.float64)
    n, nf, rc = len(Z), model.n_features, model.local_cutoff
    ne = model.nuclear_embedding
    d = torch.tensor(electron_configurations())
    e_z = torch.stack([ne.element_embedding[z] + ne.config_linear.weight @ d[z] for z in Z])  # eq 9
    x = e_z.clone()
    for emb, psi, charge in ((model.charge_embedding, Q, True),
                             (model.spin_embedding, S, False)):
        if emb is None:
            continue
        col = 0 if psi >= 0 or not charge else 1
        k = emb.linear_k.weight[:, col] * float(psi != 0)
        # the reference code's value carries |psi|; the paper's v_- is its negative
        v = emb.linear_v.weight[:, col] * (-1.0 if col == 1 else 1.0)
        dots = torch.stack([_lin(emb.linear_q, e_z[i]) @ k / math.sqrt(nf) for i in range(n)])
        soft = F.softplus(dots, threshold=1e9)
        a = psi * soft / (soft.sum() + 1e-8)                                     # eq 10
        x = x + torch.stack([_resmlp(emb.resmlp, a[i] * v) for i in range(n)])

    rbf = model.radial_basis
    gamma = F.softplus(rbf.gamma_raw)
    K = rbf.n_rbf

    def rho(r):                                                                 # eqs 14-16
        xr = math.exp(-float(gamma) * r)
        fc = math.exp(-r * r / ((rc - r) * (rc + r)))
        return torch.tensor([math.comb(K - 1, k) * xr ** k * (1 - xr) ** (K - 1 - k) * fc
                             for k in range(K)])

    pairs = [(i, j) for i in range(n) for j in range(n)
             if i != j and float(torch.linalg.norm(pos[j] - pos[i])) < rc]
    basis = {}
    for i, j in pairs:
        rij = pos[j] - pos[i]
        r = float(torch.linalg.norm(rij))
        u = rij / r
        basis[(i, j)] = (rho(r), torch.tensor([_real_harmonic(1, m, u) for m in (-1, 0, 1)]),
                         torch.tensor([_real_harmonic(2, m, u) for m in range(-2, 3)]))

    f = torch.zeros(n, nf)
    for module in model.interactions:
        xt = x
        for block in module.residual_pre:
            xt = _residual(block, xt)
        li = module.local_interaction
        loc = []
        for i in range(n):                                                      # eq 12
            s = torch.zeros(nf)
            p = torch.zeros(3, nf)
            dd = torch.zeros(5, nf)
            for (a, j), (rho_ij, y1, y2) in basis.items():
                if a != i:
                    continue
                s = s + _resmlp(li.resmlp_s, xt[j]) * (li.radial_s.weight @ rho_ij)
                p = p + y1[:, None] * (_resmlp(li.resmlp_p, xt[j]) * (li.radial_p.weight @ rho_ij))
                dd = dd + y2[:, None] * (_resmlp(li.resmlp_d, xt[j]) * (li.radial_d.weight @ rho_ij))
            P1, P2 = li.projection_p.weight[:nf], li.projection_p.weight[nf:]
            D1, D2 = li.projection_d.weight[:nf], li.projection_d.weight[nf:]
            loc.append(_resmlp(li.resmlp_l, _resmlp(li.resmlp_c, xt[i]) + s
                               + ((p @ P1.T) * (p @ P2.T)).sum(0)
                               + ((dd @ D1.T) * (dd @ D2.T)).sum(0)))
        h = xt + torch.stack(loc)
        nl = module.nonlocal_interaction
        if nl is not None:                                                      # eqs 18-20
            q = torch.stack([_resmlp(nl.resmlp_q, xt[i]) for i in range(n)])
            k = torch.stack([_resmlp(nl.resmlp_k, xt[i]) for i in range(n)])
            v = torch.stack([_resmlp(nl.resmlp_v, xt[i]) for i in range(n)])
            if nl.exact:
                dot = q @ k.T
                w = torch.exp((dot - dot.max()) / math.sqrt(nf))
            else:
                om = nl.omega
                m = om.shape[1]
                uq, uk = (q / nf ** 0.25) @ om, (k / nf ** 0.25) @ om
                hq = (q * q).sum(1, keepdim=True) / (2 * math.sqrt(nf))
                hk = (k * k).sum(1, keepdim=True) / (2 * math.sqrt(nf))
                # FAVOR+ features with the reference code's stabilizers
                phq = (torch.exp(uq - hq - uq.max(1, keepdim=True).values) + 1e-4) / math.sqrt(m)
                phk = (torch.exp(uk - hk - uk.max()) + 1e-4) / math.sqrt(m)
                w = phq @ phk.T
            h = h + (w @ v) / (w.sum(1, keepdim=True) + 1e-8)
        for block in module.residual_post:
            h = _residual(block, h)
        x = h
        f = f + torch.stack([_resmlp(module.resmlp_y, x[i]) for i in range(n)])  # eqs 3, 11

    w_e, w_q = model.output.weight[0], model.output.weight[1]
    e_atoms = torch.stack([w_e @ f[i] + model.atom_ref.weight[Z[i], 0] for i in range(n)])  # eq 21
    q_raw = torch.stack([w_q @ f[i] + model.charge_ref.weight[Z[i], 0] for i in range(n)])
    charges = q_raw + (Q - q_raw.sum()) / n                                     # eq 24
    energy = e_atoms.sum()

    rep = model.repulsion
    if rep is not None:                                                         # eq 22
        c = F.softplus(rep._c)
        c = c / c.sum()
        a_k = F.softplus(rep._a)
        p_exp = F.softplus(rep._apow)
        inv_d = F.softplus(rep._adiv)
        for i in range(n):
            for j in range(i + 1, n):
                r = float(torch.linalg.norm(pos[j] - pos[i]))
                if r >= rc:
                    continue
                fc = math.exp(-r * r / ((rc - r) * (rc + r)))
                screen = (c * torch.exp(-a_k * r * (Z[i] ** p_exp + Z[j] ** p_exp) * inv_d)).sum()
                energy = energy + COULOMB_CONSTANT * Z[i] * Z[j] / r * fc * screen

    if model.electrostatics is not None:                                        # eqs 23, 25
        r_on, r_off = 0.25 * rc, 0.75 * rc

        def sigma(t):
            return math.exp(-1.0 / t) if t > 0 else 0.0

        for i in range(n):
            for j in range(i + 1, n):
                r = float(torch.linalg.norm(pos[j] - pos[i]))
                t = (r - r_on) / (r_off - r_on)
                fs = 1.0 if t <= 0 else 0.0 if t >= 1 else sigma(1 - t) / (sigma(1 - t) + sigma(t))
                energy = energy + COULOMB_CONSTANT * charges[i] * charges[j] * (
                    fs / math.sqrt(r * r + 1) + (1 - fs) / r)

    if model.dispersion is not None:                                            # eq 27
        disp = model.dispersion
        d4 = DFTD4(s6=1.0, s8=float(F.softplus(disp._s8)), a1=float(F.softplus(disp._a1)),
                   a2=float(F.softplus(disp._a2)), s9=0.0, cutoff_pair=1e3, cutoff_cn=1e3,
                   regime="dense")
        z = torch.tensor(Z)
        ei = torch.tensor([(j, i) for i in range(n) for j in range(n) if i != j]).T
        r_au = torch.linalg.norm(pos[ei[1]] - pos[ei[0]], dim=-1) / BOHR
        cn, _ = d4.coordination_numbers(z, ei, r_au, n)
        alpha = d4.dynamic_polarizabilities(z, d4.reference_weights(z, cn, charges))
        energy = energy + d4.two_body_energy(z, ei, r_au, alpha, n).sum() * HARTREE
    return energy, charges


def _check_reference(model, s, Q=0.0, S=0.0, tol=1e-10):
    g = _graph(s, total_charge=Q, spin_multiplicity=S + 1)
    out = model(g)
    e_ref, q_ref = _reference(model, s["atomic_numbers"], s["pos"], Q, S)
    assert abs(float(out["energy"]) - float(e_ref)) < tol * max(1.0, abs(float(e_ref)))
    assert torch.allclose(out["charges"], q_ref, atol=tol)


# registry / defaults

def test_registered():
    assert "spookynet" in available_models()


def test_paper_defaults():
    """SpookyNet() is the paper architecture (F 128, T 6, K 16, 10 bohr, FAVOR+, all terms)."""
    m = SpookyNet()
    assert m.n_features == 128 and len(m.interactions) == 6
    assert m.radial_basis.n_rbf == 16 and m.local_cutoff == pytest.approx(10 * BOHR)
    nl = m.interactions[0].nonlocal_interaction
    assert not nl.exact and nl.omega.shape == (128, 128)
    assert m.repulsion is not None and m.electrostatics is not None and m.dispersion is not None
    assert m.lr_cutoff is None and m.cutoff == m.local_cutoff
    assert float(F.softplus(m.radial_basis.gamma_raw)) == pytest.approx(0.5 / BOHR)
    assert float(F.softplus(m.dispersion._s8)) == pytest.approx(1.61679827)
    assert m.output.weight.shape == (2, 128)


def test_invalid_options():
    with pytest.raises(ValueError):
        SpookyNet(n_features=8, attention="linear")
    with pytest.raises(ValueError):
        SpookyNet(n_features=8, ewald=True)
    with pytest.raises(ValueError):
        SpookyNet(n_features=8, atomic_energies=[1.0])


# building blocks

def test_electron_configurations():
    """Occupations add up to Z, Madelung exceptions are applied, columns are scaled to [0, 1]."""
    table = electron_configurations()
    assert table.shape == (87, 20) and table.min() == 0.0 and table.max(axis=0).max() == 1.0
    maxima = np.array([86, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 6, 10, 14.0])
    raw = table * maxima
    assert np.allclose(raw[:, 1:16].sum(1), np.arange(87))
    assert np.allclose(raw[6], [6, 2, 2, 2] + [0] * 12 + [2, 2, 0, 0])
    assert np.allclose(raw[29, [6, 7]], [1, 10])                       # Cu 4s1 3d10
    assert np.allclose(raw[46, [9, 10, 16]], [0, 10, 0])               # Pd 5s0 4d10
    assert np.allclose(raw[79, 16:], [1, 0, 10, 14])                   # Au valence


def test_exponential_bernstein_basis():
    """Eqs 14-15: binom(K-1, k) x^k (1-x)^(K-1-k) of x = exp(-gamma r), a partition of unity."""
    basis = ExponentialBernsteinRBF(7, gamma=0.8)
    r = torch.linspace(0.3, 6.0, 11)
    rbf = basis(r)
    x = torch.exp(-0.8 * r)
    closed = torch.stack([math.comb(6, k) * x ** k * (1 - x) ** (6 - k) for k in range(7)], -1)
    assert torch.allclose(rbf, closed, atol=1e-14)
    assert torch.allclose(rbf.sum(-1), torch.ones_like(r), atol=1e-14)
    weighted = ExponentialBernsteinRBF(7, gamma=0.8, exp_weighting=True)(r)
    assert torch.allclose(weighted, closed * x[:, None], atol=1e-14)


def test_smooth_switch():
    r = torch.linspace(0.0, 4.0, 401)
    f = smooth_switch(r, 1.0, 3.0)
    assert torch.all(f[r <= 1.0] == 1.0) and torch.all(f[r >= 3.0] == 0.0)
    assert torch.allclose(f + smooth_switch(4.0 - r, 1.0, 3.0), torch.ones_like(r), atol=1e-14)
    rr = r.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(smooth_switch(rr, 1.0, 3.0).sum(), rr)
    assert torch.isfinite(g).all() and g[r <= 1.0].abs().max() == 0.0


def test_orthogonal_random_features():
    omega = orthogonal_random_features(20, 8, torch.Generator().manual_seed(0))
    assert omega.shape == (8, 20)
    block = omega[:, :8] / omega[:, :8].norm(dim=0)
    assert torch.allclose(block.T @ block, torch.eye(8), atol=1e-12)


def test_favor_approximates_softmax_attention():
    """With many random features FAVOR+ approaches the exact attention of eq 19."""
    torch.manual_seed(0)
    from xnn.hybrid.models.spookynet import _NonlocalInteraction
    g = torch.Generator().manual_seed(1)
    fav = _NonlocalInteraction(8, 1, 20000, g)
    ex = _NonlocalInteraction(8, 1, None)
    ex.load_state_dict({k: v for k, v in fav.state_dict().items() if k != "omega"}, strict=False)
    for mod in (fav, ex):
        _randomize(mod, 3)
    batch = torch.zeros(6, dtype=torch.long)
    x = 0.5 * torch.randn(6, 8)
    slot = torch.arange(6)
    a = fav(x, batch, 1, slot, 6)
    b = ex(x, batch, 1, slot, 6)
    assert float((a - b).abs().max()) < 0.05 * float(b.abs().max())


# fidelity to the equations

@pytest.mark.parametrize("kw", [{}, {"attention": "exact"}, {"n_residual": 2},
                                {"nonlocal_interactions": False},
                                {"charge_embedding": False, "spin_embedding": False}])
def test_forward_matches_paper_equations(kw):
    model = _small(**kw).eval()
    _check_reference(model, _structure(6, seed=1))


@pytest.mark.parametrize("Q,S", [(1.0, 0.0), (-1.0, 1.0), (2.0, 2.0), (0.0, 2.0)])
def test_electronic_states_match_paper_equations(Q, S):
    model = _small(seed=2).eval()
    _check_reference(model, _structure(6, seed=3), Q, S)


def test_charge_and_spin_change_the_energy():
    """Charge and spin change the energy with the embeddings (eq 1) and not without them."""
    s = _structure(5, seed=4)
    model = _small(seed=4).eval()
    e = {(q, m): float(model(_graph(s, total_charge=q, spin_multiplicity=m))["energy"])
         for q in (0, 1, -1) for m in (1, 3)}
    assert len({round(v, 8) for v in e.values()}) == 6
    blind = _small(seed=4, charge_embedding=False, spin_embedding=False,
                   electrostatics=False, d4_dispersion=False).eval()
    e = {float(blind(_graph(s, total_charge=q, spin_multiplicity=m))["energy"])
         for q in (0, 1, -1) for m in (1, 3)}
    assert max(e) - min(e) < 1e-12


def test_charges_sum_to_the_total_charge():
    model = _small().eval()
    graphs = [_graph(_structure(n, seed=n), total_charge=q) for n, q in ((4, 1), (6, -2), (5, 0))]
    out = model(collate(graphs))
    sums = torch.zeros(3).index_add(0, collate(graphs).batch, out["charges"])
    assert torch.allclose(sums, torch.tensor([1.0, -2.0, 0.0]), atol=1e-12)


def test_electronic_embedding_vanishes_for_neutral_singlets():
    """Without bias terms e_Psi = 0 when Psi = 0 (eq 10)."""
    model = _small()
    emb = model.charge_embedding
    e_z = torch.randn(5, 8)
    out = emb(e_z, torch.zeros(1), torch.zeros(5, dtype=torch.long), 1)
    assert out.abs().max() == 0.0


def test_zbl_reaches_the_bare_coulomb_repulsion():
    """Eq 22 with normalized c_k: E_rep r / (k_e Z_i Z_j) -> 1 as r -> 0."""
    model = SpookyNet(n_features=8, n_interactions=1, n_rbf=4, electrostatics=False,
                      d4_dispersion=False)
    rep = model.repulsion
    zf = torch.tensor([8.0, 6.0])
    r = torch.tensor([1e-4, 1e-4])
    e = rep(zf, r, torch.ones(2), torch.tensor([1, 0]), torch.tensor([0, 1])).sum()
    assert float(e) * 1e-4 / (COULOMB_CONSTANT * 48.0) == pytest.approx(1.0, rel=1e-3)


def test_electrostatics_kernel_limits():
    """Eq 23 kernel limits and the zero value and slope of the truncated kernel."""
    model = SpookyNet(n_features=8, n_interactions=1, n_rbf=4, cutoff=4.0)
    ele = model.electrostatics
    r = torch.tensor([0.5, 0.9, 3.1, 7.0])
    k = ele.kernel(r)
    assert torch.allclose(k[:2], 1.0 / torch.sqrt(r[:2] ** 2 + 1))
    assert torch.allclose(k[2:], 1.0 / r[2:])
    trunc = SpookyNet(n_features=8, n_interactions=1, n_rbf=4, cutoff=4.0,
                      lr_cutoff=8.0).electrostatics
    rr = torch.tensor([8.0 - 1e-6], requires_grad=True)
    v = trunc.kernel(rr)
    (g,) = torch.autograd.grad(v.sum(), rr)
    assert abs(float(v)) < 1e-10 and abs(float(g)) < 1e-5


def test_d4_tables_are_owned_by_each_model():
    """Changing one model's D4 tables leaves a second model untouched."""
    a = SpookyNet(n_features=8, n_interactions=1, n_rbf=4)
    with torch.no_grad():
        a.dispersion.r4r2.zero_()
    b = SpookyNet(n_features=8, n_interactions=1, n_rbf=4)
    assert float(b.dispersion.r4r2.abs().max()) > 0.0


def test_d4_reference_polarizabilities_match_dftd4():
    """With s_q = 1 the recomputed reference polarizabilities are those of DFTD4."""
    model = SpookyNet(n_features=8, n_interactions=1, n_rbf=4)
    alpha = model.dispersion.reference_polarizabilities(torch.tensor(1.0))
    assert torch.allclose(alpha, DFTD4(regime="dense").refalpha, rtol=1e-13, atol=1e-13)


# invariances, smoothness, batching

def _rotation(seed, proper=True):
    Q = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(seed)))[0]
    if (torch.det(Q) > 0) != proper:
        Q = Q @ torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
    return Q


@pytest.mark.parametrize("proper", [True, False])
def test_energy_invariance_forces_equivariance(proper):
    model = ForceStressOutput(_small(seed=5)).eval()
    s = _structure(7, seed=5)
    R, t = _rotation(4, proper), torch.tensor([1.3, -0.7, 2.1])
    out = model(_graph(s, total_charge=1.0))
    s2 = {"pos": torch.as_tensor(s["pos"]) @ R.T + t, "atomic_numbers": s["atomic_numbers"]}
    out2 = model(_graph(s2, total_charge=1.0))
    assert abs(float(out["energy"] - out2["energy"])) < 1e-10
    assert torch.allclose(out["forces"] @ R.T, out2["forces"], atol=1e-10)
    assert torch.allclose(out["charges"], out2["charges"], atol=1e-12)


def test_permutation_invariance():
    model = _small(seed=6).eval()
    s = _structure(6, seed=6)
    perm = np.random.default_rng(0).permutation(6)
    s2 = {"pos": s["pos"][perm], "atomic_numbers": [s["atomic_numbers"][i] for i in perm]}
    out, out2 = model(_graph(s)), model(_graph(s2))
    assert abs(float(out["energy"] - out2["energy"])) < 1e-10
    assert torch.allclose(out["charges"][perm], out2["charges"], atol=1e-12)


def test_forces_are_minus_gradient():
    model = ForceStressOutput(_small(seed=7)).eval()
    s = _structure(6, seed=7)
    out = model(_graph(s, total_charge=-1.0, spin_multiplicity=2))
    h = 1e-5
    for (i, k) in ((0, 0), (3, 1), (5, 2)):
        e = []
        for sign in (1, -1):
            p = s["pos"].copy()
            p[i, k] += sign * h
            g = _graph({"pos": p, "atomic_numbers": s["atomic_numbers"]}, total_charge=-1.0,
                       spin_multiplicity=2)
            e.append(float(model(g)["energy"]))
        assert float(out["forces"][i, k]) == pytest.approx(-(e[0] - e[1]) / (2 * h), abs=1e-6)


def test_pes_smooth_across_cutoff():
    """Energy and forces are continuous when a neighbor crosses the cutoff (eq 16)."""
    model = ForceStressOutput(_small(seed=8, electrostatics=False, d4_dispersion=False)).eval()

    def evaluate(d):
        pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.9, 0.0],
                        [math.sqrt(d ** 2 - 0.45 ** 2), 0.45, 0.0]])
        out = model(_graph({"pos": pos, "atomic_numbers": [8, 1, 1]}))
        return float(out["energy"]), out["forces"].detach().clone()

    (e_in, f_in), (e_out, f_out) = evaluate(CUTOFF - 1e-7), evaluate(CUTOFF + 1e-7)
    assert abs(e_in - e_out) < 1e-10
    assert float((f_in - f_out).abs().max()) < 1e-8


def test_batching_matches_single_structures():
    """A structure's energy does not depend on the other structures of a batch."""
    for kw in ({}, {"attention": "exact"}):
        model = _small(seed=9, **kw).eval()
        structures = [(_structure(n, seed=n), q, m) for n, q, m in ((4, 0, 1), (7, 1, 2), (5, -1, 3))]
        graphs = [_graph(s, total_charge=q, spin_multiplicity=m) for s, q, m in structures]
        out = model(collate(graphs))
        start = 0
        for b, g in enumerate(graphs):
            one = model(g)
            assert abs(float(out["energy"][b] - one["energy"][0])) < 1e-10
            n = g.atomic_numbers.shape[0]
            assert torch.allclose(out["charges"][start:start + n], one["charges"], atol=1e-12)
            assert torch.allclose(out["dipole"][b], one["dipole"][0], atol=1e-12)
            start += n


def test_nonlocal_interactions_see_distant_atoms():
    """A distant atom changes a fragment energy only through the attention (Fig. 5)."""
    a = _structure(4, seed=10)
    far = {"pos": np.concatenate([a["pos"], [[40.0, 0.0, 0.0]]]),
           "atomic_numbers": list(a["atomic_numbers"]) + [9]}
    lone = {"pos": np.array([[40.0, 0.0, 0.0]]), "atomic_numbers": [9]}
    kw = dict(electrostatics=False, d4_dispersion=False, charge_embedding=False,
              spin_embedding=False)
    model = _small(seed=10, **kw).eval()

    def gap(m):
        return float(m(_graph(far))["energy"] - m(_graph(a))["energy"]
                     - m(_graph(lone))["energy"])

    assert abs(gap(model)) > 1e-6
    model.use_nonlocal = False
    assert abs(gap(model)) < 1e-12
    assert abs(gap(_small(seed=10, nonlocal_interactions=False, **kw).eval())) < 1e-12


def test_orbital_switches_change_the_energy():
    model = _small(seed=11).eval()
    g = _graph(_structure(6, seed=11))
    e_all = float(model(g)["energy"])
    model.use_d_orbitals = False
    e_sp = float(model(g)["energy"])
    model.use_p_orbitals = False
    e_s = float(model(g)["energy"])
    assert len({round(e_all, 9), round(e_sp, 9), round(e_s, 9)}) == 3


def test_periodic_structures_need_lr_cutoff():
    s = _structure(5, seed=12)
    s.update(cell=np.eye(3) * 6.0, pbc=[True] * 3)
    with pytest.raises(ValueError):
        _small()(_graph(s))
    model = ForceStressOutput(_small(lr_cutoff=6.0), compute_stress=True).eval()
    out = model(_graph(s, cutoff=6.0))
    assert torch.isfinite(out["stress"]).all() and torch.isfinite(out["forces"]).all()


def test_ewald_matches_the_molecular_sum_in_a_large_cell():
    """Ewald sum plus the tin-foil surface term equals the molecular sum in a large box."""
    s = _structure(5, seed=13)
    kw = dict(seed=13, d4_dispersion=False)
    molecular = _small(**kw).eval()
    periodic = _small(lr_cutoff=CUTOFF, ewald=True, ewald_accuracy=1e-8, **kw).eval()
    length = 40.0
    box = dict(s, cell=np.eye(3) * length, pbc=[True] * 3)
    out = molecular(_graph(s))
    e_box = float(periodic(_graph(box))["energy"])
    mu2 = float((out["dipole"] ** 2).sum())
    surface = 2.0 * math.pi * COULOMB_CONSTANT * mu2 / (3.0 * length ** 3)
    assert e_box + surface == pytest.approx(float(out["energy"]), abs=3e-5)


def test_float32_forward_matches_float64():
    model = _small(seed=14).eval()
    s = _structure(7, seed=14)
    out64 = model(_graph(s, total_charge=1.0))
    model32 = model.float()
    g32 = structure_to_graph({"pos": torch.as_tensor(s["pos"]).float(),
                              "atomic_numbers": s["atomic_numbers"], "total_charge": 1.0}, CUTOFF)
    out32 = model32(g32)
    assert abs(float(out64["energy"]) - float(out32["energy"])) < 1e-4 * max(1.0, abs(float(out64["energy"])))
    assert torch.allclose(out64["charges"], out32["charges"].double(), atol=1e-4)


def test_float64_cast_restores_exact_tables():
    torch.set_default_dtype(torch.float32)
    model = SpookyNet(n_features=8, n_interactions=1, n_rbf=5).double()
    torch.set_default_dtype(torch.float64)
    reference = SpookyNet(n_features=8, n_interactions=1, n_rbf=5)
    assert torch.equal(model.nuclear_embedding.electron_config,
                       reference.nuclear_embedding.electron_config)
    assert torch.equal(model.radial_basis.log_binom, reference.radial_basis.log_binom)
    for name in ("refalpha", "alphaiw", "refcn", "r4r2", "cp_weights"):
        if hasattr(reference.dispersion, name):
            assert torch.equal(getattr(model.dispersion, name), getattr(reference.dispersion, name))


# deployment, training, config

def test_scripted_core_and_torchscript_export(tmp_path):
    from xnn.common.deploy import export_torchscript_potential

    model = _small(seed=15).eval()
    s = _structure(6, seed=15)
    g = _graph(s)
    scripted = torch.jit.script(model)
    args = (g.atomic_numbers, g.edge_index, g.edge_vectors(), g.pos, torch.zeros(3, 3),
            torch.zeros(3, dtype=torch.bool), 1.0, 3.0)
    f1, e1, q1 = scripted.node_features_energy_charges(*args)
    f2, e2, q2 = model.node_features_energy_charges(*args)
    assert (e1 - e2).abs().max() < 1e-12 and (q1 - q2).abs().max() < 1e-12

    path = str(tmp_path / "spookynet.pt")
    export_torchscript_potential(model, model.cutoff, path, total_charge=1.0, spin_multiplicity=3.0)
    loaded = torch.jit.load(path)
    out = loaded(g.pos, g.atomic_numbers)
    ref = ForceStressOutput(model)(_graph(s, total_charge=1.0, spin_multiplicity=3))
    assert abs(float(out["energy"]) - float(ref["energy"])) < 1e-8
    assert (out["forces"] - ref["forces"].detach()).abs().max() < 1e-8
    assert torch.allclose(out["charges"], ref["charges"].detach(), atol=1e-10)


def test_ase_calculator_reads_charge_and_spin():
    pytest.importorskip("ase")
    from ase import Atoms
    from xnn.common.deploy import XNNCalculator

    model = ForceStressOutput(_small(seed=16)).eval()
    energies = []
    for mult in (1, 3):
        atoms = Atoms("CH2", positions=[[0, 0, 0], [-0.865, -0.584, 0], [0.865, -0.584, 0]])
        atoms.info["spin_multiplicity"] = mult
        atoms.calc = XNNCalculator(model, cutoff=model.model.cutoff)
        energies.append(atoms.get_potential_energy())
    assert abs(energies[0] - energies[1]) > 1e-6


def test_multihead_and_reference_energies():
    from xnn.common.finetune import MultiHead, get_atomic_energies, set_atomic_energies

    model = _small()
    set_atomic_energies(model, {1: -0.5, 6: -1.0, 8: -2.0})
    assert get_atomic_energies(model, [1, 6, 8]) == {1: -0.5, 6: -1.0, 8: -2.0}
    out = MultiHead(model, ["a", "b"])(_graph(_structure(5)))
    assert out["energy"].shape == (1,) and out["charges"].shape == (5,)


def test_train_step_with_charge_states_and_dipoles(tmp_path):
    from xnn.common.data import AtomicDataset
    from xnn.common.train import Trainer

    rng = np.random.default_rng(0)
    structures = []
    for i in range(6):
        s = _structure(4, seed=20 + i)
        s.update(energy=float(rng.normal()), forces=rng.normal(size=(4, 3)),
                 dipole=rng.normal(size=3), total_charge=float(i % 2),
                 spin_multiplicity=1 + 2 * (i % 2))
        structures.append(s)
    cfg = from_dict({"model": {"name": "spookynet", "n_features": 8, "n_interactions": 1,
                               "n_rbf": 4, "cutoff": CUTOFF},
                     "optim": {"epochs": 2, "dipole_weight": 1.0},
                     "data": {"batch_size": 3, "val_fraction": 0.0},
                     "output_dir": str(tmp_path)})
    logs = Trainer(cfg, AtomicDataset(structures, CUTOFF)).fit()["train"]
    assert {"energy_mse", "force_mse", "dipole_mse"} <= set(logs)


def test_from_config_and_reference_key_translation():
    cfg = from_dict({"model": {
        "name": "spookynet", "cutoff": 4.0, "num_features": 16, "num_modules": 2,
        "num_basis_functions": 6,
        "extra": {"use_d4_dispersion": False, "lr_cutoff": 8.0, "attention": "exact",
                  "n_residual": 2, "species": [1, 6], "atomic_energies": [-13.6, -1030.0]}}})
    m = build_model(cfg.model)
    assert m.n_features == 16 and len(m.interactions) == 2 and m.radial_basis.n_rbf == 6
    assert m.dispersion is None and m.lr_cutoff == 8.0 and m.cutoff == 8.0
    assert m.interactions[0].nonlocal_interaction.exact
    assert len(m.interactions[0].residual_pre) == 2
    assert float(m.atom_ref.weight[6, 0]) == pytest.approx(-1030.0)


def test_paper_config_file():
    import yaml
    path = os.path.join(os.path.dirname(__file__), "..", "configs", "model", "spookynet.yaml")
    with open(path) as fh:
        cfg = from_dict({"model": yaml.safe_load(fh)})
    m = build_model(cfg.model)
    assert m.n_features == 128 and len(m.interactions) == 6
    assert m.local_cutoff == pytest.approx(10 * BOHR)


def test_states_dataset_reader(tmp_path):
    """The spookynet_states hub reads charge, spin, energy, forces and dipole rows."""
    import sqlite3
    from xnn.common.data import load_dataset
    from xnn.common.data.hub.spookynet_states import _read

    path = tmp_path / "states.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE data (id INTEGER PRIMARY KEY, Q FLOAT, S FLOAT, Z BLOB, "
                   "R BLOB, E FLOAT, F BLOB, D BLOB)")
        for i, (q, s) in enumerate(((1.0, 0.0), (-1.0, 2.0))):
            db.execute("INSERT INTO data VALUES (?,?,?,?,?,?,?,?)",
                       (i, q, s, np.array([6, 1, 1], "<i4").tobytes(),
                        np.arange(9, dtype="<f4").tobytes(), -10.0 - i,
                        np.ones(9, "<f4").tobytes(), np.array([0.1, 0, 0], "<f4").tobytes()))
    rows = _read(path)
    assert [r["total_charge"] for r in rows] == [1.0, -1.0]
    assert [r["spin_multiplicity"] for r in rows] == [1.0, 3.0]
    assert rows[1]["pos"].shape == (3, 3) and rows[1]["energy"] == -11.0
    assert np.allclose(rows[0]["dipole"], [0.1, 0, 0])
    g = structure_to_graph(rows[1], CUTOFF)
    assert float(g.total_charge) == -1.0 and float(g.spin_multiplicity) == 3.0
    with pytest.raises(ValueError):
        load_dataset("spookynet_states", system="water", cache_dir=tmp_path)


# parity with the reference implementation (optional)

def _parity(mode):
    pytest.importorskip("spookynet")
    import subprocess
    import sys
    script = os.path.join(os.path.dirname(__file__), "spookynet_parity.py")
    res = subprocess.run([sys.executable, script, mode], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    return float(res.stdout.strip().splitlines()[-1])


def test_parity_vs_reference_float64():
    """Transplanted weights reproduce the reference energies, forces, charges and dipoles."""
    assert _parity("float64") < 1e-12


def test_parity_vs_reference_float32():
    assert _parity("float32") < 1e-5


def test_parity_published_checkpoint():
    """The reference code's published example model gives the same predictions."""
    path = os.environ.get("SPOOKYNET_REFERENCE_CHECKPOINT")
    if not path:
        pytest.skip("SPOOKYNET_REFERENCE_CHECKPOINT is not set")
    pytest.importorskip("spookynet")
    import subprocess
    import sys
    script = os.path.join(os.path.dirname(__file__), "spookynet_parity.py")
    res = subprocess.run([sys.executable, script, "float64", path], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    assert float(res.stdout.strip().splitlines()[-1]) < 1e-12
