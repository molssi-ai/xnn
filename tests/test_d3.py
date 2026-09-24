"""Tests for the DFT-D3 dispersion model (:mod:`xnn.common.models.d3`).

Covers the :class:`~xnn.common.models.d3.DFTD3` core (defaults of the two
Grimme papers, exact invariances, the four damping functions against their
closed forms, the three-body term, switching windows, forces and stress by
finite differences, batching, TorchScript), the
:class:`~xnn.common.models.d3.D3Dispersion` wrapper around **every**
registered model, the ``extra["dispersion"] = {"name": "d3", ...}`` config
hook, the deploy channels, the untouched legacy PhysNet/BAMBOO API, and --
when the upstream ``dftd3`` (simple-dftd3) Python package is installed --
floating-point parity of energies, gradients and virials for every damping
function, molecular and periodic.
"""
import numpy as np
import pytest
import torch

from xnn.common.config import from_dict
from xnn.common.data import collate, structure_to_graph
from xnn.common.models import (
    D3Dispersion,
    D4Dispersion,
    DFTD3,
    ForceStressOutput,
    LatentEwald,
    build_model,
)
from xnn.common.models import d3 as legacy
from xnn.common.models.d3 import PBE0_D3BJ, PBE0_D3ZERO
from xnn.common.models.dispersion import BOHR, HARTREE


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


WATER = (np.array([[0.0, 0.0, 0.119262], [0.0, 0.763239, -0.477047],
                   [0.0, -0.763239, -0.477047]]), [8, 1, 1])
BENZENE_R = 1.397
FAST = dict(cutoff_pair=9.0, cutoff_triple=7.0, cutoff_cn=8.0)


def _benzene():
    ang = np.arange(6) * np.pi / 3
    c = np.stack([BENZENE_R * np.cos(ang), BENZENE_R * np.sin(ang), np.zeros(6)], 1)
    h = c * (BENZENE_R + 1.08) / BENZENE_R
    return np.concatenate([c, h]), [6] * 6 + [1] * 6


def _graph(pos, z, cutoff, cell=None, R=None, shift=0.0):
    pos = np.asarray(pos, dtype=float) + shift
    s = {"pos": pos if R is None else pos @ R.T, "atomic_numbers": z}
    if cell is not None:
        cell = np.asarray(cell, dtype=float)
        s["cell"] = cell if R is None else cell @ R.T
        s["pbc"] = [True] * 3
    return structure_to_graph(s, cutoff)


def _rotation(seed=3):
    rng = np.random.default_rng(seed)
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    return R


def _cluster(n=10, seed=0, box=6.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, box - 1, (n, 3)), [1, 8, 6, 7] * (n // 4) + [1] * (n % 4)


def _energy(model, pos, z, **kw):
    return float(model(_graph(pos, z, model.cutoff, **kw))["energy"])


# --------------------------------------------------------------------------
# DFTD3 core
# --------------------------------------------------------------------------

def test_defaults_are_pbe0_of_the_papers():
    bj = DFTD3()
    assert bj.damping == "bj"
    for k in ("s6", "s8", "a1", "a2", "s9"):
        assert float(getattr(bj, k)) == PBE0_D3BJ[k]
    zero = DFTD3(damping="zero")
    for k in ("s6", "s8", "rs6", "rs8", "s9"):
        assert float(getattr(zero, k)) == PBE0_D3ZERO[k]
    assert bj.alp == 14.0 and bj.wf == 4.0 and bj.kcn == 16.0
    assert abs(bj.cutoff_pair - 60 * BOHR) < 1e-12
    assert abs(bj.cutoff_triple - 40 * BOHR) < 1e-12
    assert abs(bj.cutoff_cn - 40 * BOHR) < 1e-12
    with pytest.raises(ValueError, match="damping"):
        DFTD3(damping="nonsense")


def test_reference_data_sanity():
    d3 = DFTD3()
    assert d3.c6ref.shape == (7, 7, 104, 104)
    assert int(d3.nref[6]) == 5 and int(d3.nref[1]) == 2 and int(d3.nref[89]) == 7
    # the 2010 paper's table II (TD-DFT column, printed to 2-3 digits): C6 of
    # the free rare-gas atoms
    for z, paper in [(2, 1.54), (18, 64.2), (54, 288.6), (86, 410.5)]:   # He, Ar, Xe, Rn
        assert abs(float(d3.c6ref[0, 0, z, z]) / paper - 1) < 0.02
    # symmetric under swapping both index pairs
    assert torch.allclose(d3.c6ref, d3.c6ref.permute(1, 0, 3, 2))
    # the pair cutoff radii of the zero damping: C-C 2.9103 A (sec II.D)
    assert abs(float(d3.rvdw[6, 6]) * BOHR - 2.9103) < 1e-6


def test_matches_reference_values_for_water():
    """PBE0-D3(BJ) water: value generated with simple-dftd3 1.6.0."""
    d3 = D3Dispersion()
    out = d3(_graph(*WATER, d3.cutoff))
    assert abs(float(out["energy"]) / HARTREE - (-2.768885028766e-04)) < 1e-14
    zero = D3Dispersion(damping="zero")
    assert abs(float(zero(_graph(*WATER, zero.cutoff))["energy"]) / HARTREE
               - (-4.644598055357e-06)) < 1e-16


@pytest.mark.parametrize("damping", ["bj", "zero", "mzero", "op"])
def test_two_body_damping_closed_forms_on_a_dimer(damping):
    """Each damping function reproduces its formula on an Ar-Ar pair."""
    kw = dict(damping=damping, s9=0.0)
    if damping == "mzero":
        kw.update(bet=0.1)
    if damping == "op":
        kw.update(bet=6.0)
    d3 = DFTD3(**kw)
    r_a = 5.0
    z = torch.tensor([18, 18])
    pos = torch.tensor([[0.0, 0, 0], [r_a, 0, 0]])
    g = _graph(pos.numpy(), [18, 18], d3.cutoff)
    out = d3(g)
    c6 = float(out["c6_matrix"][0, 1])
    r = r_a / BOHR
    q = float(d3.r4r2[18])
    rr = 3 * q * q
    s6, s8, alp = float(d3.s6), float(d3.s8), d3.alp
    if damping == "bj":
        r0 = float(d3.a1) * (rr ** 0.5) + float(d3.a2)
        e = -c6 * (s6 / (r ** 6 + r0 ** 6) + s8 * rr / (r ** 8 + r0 ** 8))
    elif damping == "zero":
        r0 = float(d3.rvdw[18, 18])
        e = -c6 * (s6 / (1 + 6 * (float(d3.rs6) * r0 / r) ** alp) / r ** 6
                   + s8 * rr / (1 + 6 * (float(d3.rs8) * r0 / r) ** (alp + 2)) / r ** 8)
    elif damping == "mzero":
        r0 = float(d3.rvdw[18, 18])
        b = float(d3.bet) * r0
        e = -c6 * (s6 / (1 + 6 * (r / (float(d3.rs6) * r0) + b) ** (-alp)) / r ** 6
                   + s8 * rr / (1 + 6 * (r / (float(d3.rs8) * r0) + b) ** (-alp - 2)) / r ** 8)
    else:
        r0 = float(d3.a1) * (rr ** 0.5) + float(d3.a2)
        b = float(d3.bet)
        e = -c6 * (s6 * r ** b / (r ** (6 + b) + r0 ** (6 + b))
                   + s8 * rr * r ** b / (r ** (8 + b) + r0 ** (8 + b)))
    assert abs(float(out["energy"]) / HARTREE - e) < 1e-15 * max(1.0, abs(e) * 1e3)


def test_c6_interpolation_follows_the_coordination_number():
    """Fig 5 of the 2010 paper: C6(C-C) drops from the free atom (~49) to the
    sp3 value (~18) as the coordination number grows, monotonically."""
    d3 = DFTD3()
    cn = torch.linspace(0.0, 5.0, 26)
    z = torch.full((26,), 6, dtype=torch.long)
    w = d3.reference_weights(z, cn)
    v = d3._species_vectors(z, w)
    c6 = d3.c6_matrix(z, w, v).diagonal()
    assert abs(float(c6[0]) - 49.0) < 0.3 and abs(float(c6[-1]) - 18.1) < 0.3
    assert bool((c6[1:] <= c6[:-1] + 1e-9).all())
    # at a reference CN the neighboring references still carry exp(-4) weight,
    # so the interpolated value sits within a few tenths of the reference
    # (table II: sp2 C 25.78)
    w_ref = d3.reference_weights(torch.tensor([6]), torch.tensor([float(d3.refcn[3, 6])]))
    assert abs(float(d3.c6_matrix(torch.tensor([6]), w_ref, d3._species_vectors(
        torch.tensor([6]), w_ref))[0, 0]) - 25.7809) < 0.2


@pytest.mark.parametrize("periodic", [False, True])
def test_exact_invariances(periodic):
    d3 = D3Dispersion(s9=1.0, **FAST)
    pos, z = _cluster()
    cell = np.eye(3) * 6.0 if periodic else None
    e0 = _energy(d3, pos, z, cell=cell)
    assert abs(e0 - _energy(d3, pos, z, cell=cell, R=_rotation())) < 1e-11
    assert abs(e0 - _energy(d3, pos, z, cell=cell, shift=1.7)) < 1e-11
    perm = np.random.default_rng(1).permutation(len(z))
    assert abs(e0 - _energy(d3, pos[perm], [z[i] for i in perm], cell=cell)) < 1e-11
    if periodic:
        pos2 = pos.copy()
        pos2[3] += cell[1]
        assert abs(e0 - _energy(d3, pos2, z, cell=cell)) < 1e-11


def test_three_body_repulsive_and_switchable():
    pos, z = _benzene()
    on = D3Dispersion(s9=1.0)
    off = D3Dispersion()
    e_on = on(_graph(pos, z, on.cutoff))
    e_off = off(_graph(pos, z, off.cutoff))
    assert float(e_on["energy_3body"]) > 0
    assert float(e_off["energy_3body"]) == 0.0
    assert abs(float(e_on["energy_2body"]) - float(e_off["energy"])) < 1e-14
    # the 2010 paper: E(3) is a few percent of E(2) for small molecules
    assert float(e_on["energy_3body"]) < 0.05 * abs(float(e_on["energy_2body"]))


def test_switching_window_makes_energy_continuous():
    z = [18, 18]
    sharp = D3Dispersion(cutoff_pair=8.0, cutoff_cn=8.0, cutoff_triple=8.0)
    smooth = D3Dispersion(cutoff_pair=8.0, switch_width_pair=2.0, cutoff_cn=8.0,
                          cutoff_triple=8.0)
    e_in = [_energy(m, np.array([[0.0, 0, 0], [7.999, 0, 0]]), z) for m in (sharp, smooth)]
    e_out = [_energy(m, np.array([[0.0, 0, 0], [8.001, 0, 0]]), z) for m in (sharp, smooth)]
    assert abs(e_in[0] - e_out[0]) > 1e-8
    assert abs(e_in[1] - e_out[1]) < 1e-12


def _cluster_clear_of_cutoffs(cell, cutoffs=(7.0, 8.0, 9.0), margin=0.003):
    """A random cluster with no (image) pair within ``margin`` of a cutoff.

    The sharp upstream cutoffs make the energy discontinuous there, which a
    finite-difference derivative across the cutoff would expose (the same
    holds for the reference code); the test geometry keeps every pair farther
    from a cutoff than the finite-difference displacements move it.
    """
    from xnn.common.data import build_neighbor_list
    for seed in range(2, 50):
        pos, z = _cluster(8, seed=seed)
        g = structure_to_graph({"pos": pos, "atomic_numbers": z, "cell": cell,
                                "pbc": [True] * 3}, 9.5)
        r = torch.linalg.norm(g.edge_vectors(), dim=-1)
        if all(bool((r - c).abs().min() > margin) for c in cutoffs):
            return pos, z
    raise RuntimeError("no cutoff-clear cluster found")


def test_forces_and_stress_match_finite_differences():
    cell = np.array([[6.0, 0, 0], [0.8, 5.7, 0], [-0.4, 0.6, 6.2]])
    pos, z = _cluster_clear_of_cutoffs(cell)
    model = ForceStressOutput(D3Dispersion(s9=1.0, **FAST), compute_stress=True)
    out = model(_graph(pos, z, 9.0, cell=cell))
    h = 1e-4
    for i, a in [(0, 0), (3, 2), (5, 1)]:
        pp, pm = pos.copy(), pos.copy()
        pp[i, a] += h
        pm[i, a] -= h
        fd = -(_energy(model.model, pp, z, cell=cell) - _energy(model.model, pm, z, cell=cell)) / (2 * h)
        assert abs(float(out["forces"][i, a]) - fd) < 1e-7
    vol = abs(np.linalg.det(cell))
    for (a, b) in [(0, 0), (1, 2)]:
        eps = np.zeros((3, 3))
        eps[a, b] = eps[b, a] = 1e-4
        e_p = _energy(model.model, pos @ (np.eye(3) + eps), z, cell=cell @ (np.eye(3) + eps))
        e_m = _energy(model.model, pos @ (np.eye(3) - eps), z, cell=cell @ (np.eye(3) - eps))
        fd = (e_p - e_m) / (2e-4) / vol / (1 if a == b else 2)
        assert abs(float(out["stress"][0, a, b]) - fd) < 1e-7


@pytest.mark.parametrize("periodic", [False, True])
def test_batching_matches_single_structures(periodic):
    d3 = D3Dispersion(s9=1.0, **FAST)
    cell = np.eye(3) * 6.0 if periodic else None
    graphs = [_graph(*_cluster(10, seed=1), 9.0, cell=cell),
              _graph(*_cluster(7, seed=2), 9.0, cell=cell)]
    singles = [float(d3(g)["energy"]) for g in graphs]
    batch = d3(collate(graphs))["energy"]
    for b in range(2):
        assert abs(float(batch[b]) - singles[b]) < 1e-12


def test_scripted_core_matches_eager():
    d3 = DFTD3(s9=1.0, **FAST)
    scripted = torch.jit.script(d3)
    g = _graph(*_cluster(10), 9.0, cell=np.eye(3) * 6.0)
    args = (g.atomic_numbers, g.pos, g.edge_index, g.edge_vectors(), g.batch, 1,
            g.cell, g.pbc, torch.zeros(1))
    assert torch.equal(d3.evaluate(*args)["node_energy"], scripted.evaluate(*args)["node_energy"])


def test_trainable_parameters_and_gradients():
    model = D3Dispersion(trainable=True, s9=1.0, **FAST)
    names = {n for n, _ in model.named_parameters()}
    assert names == {"term.s6", "term.s8", "term.s9", "term.a1", "term.a2",
                     "term.rs6", "term.rs8", "term.bet"}
    e = model(_graph(*_cluster(8), 9.0))["energy"].sum()
    e.backward()
    assert float(model.term.s8.grad) < 0            # more s8, more binding
    assert float(model.term.s9.grad) > 0            # the ATM term is repulsive
    assert model.term.rs6.grad is None or float(model.term.rs6.grad) == 0.0  # unused by BJ


def test_rejects_unsupported_elements():
    with pytest.raises(ValueError, match="1..103"):
        D3Dispersion()(_graph(np.array([[0.0, 0, 0], [2.0, 0, 0]]), [104, 1], 5.0))


def test_legacy_physnet_api_is_unchanged():
    """The functional API PhysNet / BAMBOO use keeps its tables and values."""
    rng = np.random.default_rng(5)
    pos = rng.uniform(0, 4.0, (6, 3))
    Z = torch.tensor([8, 1, 1, 6, 7, 16])
    ii = torch.tensor(np.repeat(np.arange(6), 5))
    jj = torch.tensor(np.concatenate([[j for j in range(6) if j != i] for i in range(6)]))
    r = torch.tensor(np.linalg.norm(pos[ii] - pos[jj], axis=-1)) / legacy.d3_autoang
    e = legacy.edisp(Z, r, ii, jj)
    assert abs(float(e[0]) - (-0.00624853170553548)) < 1e-14   # MMunibas/PhysNet value
    # Grimme's original table (PhysNet layout) is derived from d3_reference.npz
    assert legacy.d3_c6ab.shape == (95, 95, 5, 5, 3)
    assert float(legacy.d3_c6ab[6, 8, 0, 0, 0]) == float(legacy._REFERENCE["c6"][0, 0, 6, 8])
    assert float(legacy.d3_c6ab[92, 8, 0, 0, 0]) == 162.7926        # 2010 U-O value
    assert float(legacy.d3_c6ab[92, 8, 1, 0, 1]) == 2.8878          # 2010 U reference CN
    assert torch.all(legacy.d3_c6ab[92, 8, 2:, :, :] == -1) and torch.all(legacy.d3_c6ab[0] == 0)
    assert legacy.d3_rcov.shape == (95,) and legacy.d3_r2r4.shape == (95,)
    # both reference tables agree wherever the original 2010 data apply
    core = DFTD3()
    c6_legacy = legacy._getc6(torch.tensor([6]), torch.tensor([8]), torch.tensor([3.0]),
                              torch.tensor([1.0]), legacy.d3_c6ab)
    w = core.reference_weights(torch.tensor([6, 8]), torch.tensor([3.0, 1.0]))
    c6_new = core.c6_matrix(torch.tensor([6, 8]), w, core._species_vectors(torch.tensor([6, 8]), w))
    assert abs(float(c6_legacy[0]) - float(c6_new[0, 1])) < 1e-9


def test_reference_sets():
    """2010 (Grimme / PhysNet) vs 2024 (simple-dftd3 >= 1.1.0) reference systems."""
    t10, t24 = legacy.reference_tables("2010"), legacy.reference_tables("2024")
    # identical up to radon, different for the actinides, 2010 ends at Pu
    assert np.array_equal(t10["c6"][:, :, :87, :87], t24["c6"][:, :, :87, :87])
    assert np.array_equal(t10["nref"][:87], t24["nref"][:87])
    assert t10["nref"][92] == 2 and t24["nref"][92] == 6
    assert t10["c6"][0, 0, 92, 8] == 162.7926 and t24["c6"][0, 0, 92, 8] == 163.3259
    assert t10["c6"][0, 0, 8, 92] == 162.7926                        # symmetric patch
    assert t10["nref"][95:].sum() == 0 and t24["nref"][95:].min() > 0
    # legacy layout of the 2024 set widens to seven slots
    t = legacy.legacy_c6_table("2024")
    assert t.shape == (95, 95, 7, 7, 3) and float(t[92, 8, 0, 0, 0]) == 163.3259
    assert torch.equal(t[:87, :87, :5, :5], legacy.d3_c6ab[:87, :87])
    with pytest.raises(ValueError):
        legacy.reference_tables("2015")
    # the model: same energy for Z <= 86, different for UF6, 2010 rejects Am
    a, b = D3Dispersion(references="2010", **FAST), D3Dispersion(references=2024, **FAST)
    assert a.term.references == "2010" and b.term.references == "2024"
    pos, z = _cluster()
    assert _energy(a, pos, z) == _energy(b, pos, z)
    pos_uf6 = np.array([[0.0, 0, 0], [2.0, 0, 0], [-2.0, 0, 0], [0, 2.0, 0],
                        [0, -2.0, 0], [0, 0, 2.0], [0, 0, -2.0]])
    z_uf6 = [92, 9, 9, 9, 9, 9, 9]
    assert abs(_energy(a, pos_uf6, z_uf6) - _energy(b, pos_uf6, z_uf6)) > 1e-6
    with pytest.raises(ValueError, match="reference systems"):
        _energy(a, np.array([[0.0, 0, 0], [2.0, 0, 0]]), [95, 9])
    with pytest.raises(ValueError, match="references"):
        D3Dispersion(references="2015")
    # config hook (YAML delivers the year as an int)
    m = build_model(from_dict({"model": {"name": "d3", "extra": {"references": 2010}}}).model)
    assert m.term.references == "2010"


# --------------------------------------------------------------------------
# wrapper, config hook, nesting
# --------------------------------------------------------------------------

MODEL_CONFIGS = {
    "cace": {"extra": {"species": [1, 8], "n_atom_basis": 2, "max_l": 2, "max_nu": 2}},
    "mace": {"n_features": 8, "extra": {"species": [1, 8], "l_max": 2}},
    "nequip": {"n_features": 8, "extra": {"species": [1, 8], "l_max": 1}},
    "allegro": {"n_features": 8, "extra": {"species": [1, 8], "l_max": 1,
                "two_body_latent": [8, 16], "latent": [16], "edge_eng": [8],
                "avg_num_neighbors": 9.0}},
    "schnet": {"n_features": 16},
    "physnet": {"n_features": 16, "extra": {"use_dispersion": False}},
    "hdnnp": {"extra": {"species": [1, 8]}},
    "ani": {"extra": {"species": [1, 8]}},
}


def _config(name, dispersion=None, **more):
    over = dict(MODEL_CONFIGS[name])
    extra = dict(over.pop("extra", {}))
    extra["dispersion"] = {"name": "d3", "s9": 1.0, **FAST} if dispersion is None else dispersion
    extra.update(more)
    return from_dict({"model": {"name": name, "cutoff": 4.5, "n_interactions": 1,
                               "n_rbf": over.pop("n_rbf", 6),
                               "n_features": over.pop("n_features", 8),
                               "extra": extra}})


@pytest.mark.parametrize("name", sorted(MODEL_CONFIGS))
def test_wraps_every_model(name):
    if name in ("mace", "nequip", "allegro", "cace"):
        pytest.importorskip("e3nn")
    torch.manual_seed(3)
    model = build_model(_config(name).model)
    assert isinstance(model, D3Dispersion) and isinstance(model.term, DFTD3)
    base = model.model
    assert model.cutoff == max(base.cutoff, 9.0)
    pos, _ = _cluster(10)
    z = [1, 8] * 5
    cell = np.eye(3) * 6.0
    out = ForceStressOutput(model, compute_stress=True)(_graph(pos, z, model.cutoff, cell=cell))
    e_base = float(base(_graph(pos, z, base.cutoff, cell=cell))["energy"])
    assert abs(float(out["energy_sr"]) - e_base) < 1e-10
    assert abs(float(out["energy"]) - float(out["energy_sr"]) - float(out["energy_disp"])) < 1e-12
    assert abs(float(out["node_energy"].sum()) - float(out["energy"])) < 1e-10
    assert torch.isfinite(out["forces"]).all() and out["stress"].shape == (1, 3, 3)
    assert out["coordination_numbers"].shape == (10,) and out["c6_matrix"].shape == (10, 10)
    assert "eeq_charges" not in out


def test_config_hook_selects_d3_or_d4():
    base = {"name": "schnet", "cutoff": 4.5, "n_interactions": 1, "n_rbf": 6, "n_features": 8}
    m3 = build_model(from_dict({"model": {**base, "extra": {"dispersion": {"name": "d3", "damping": "zero", "s8": 0.5}}}}).model)
    assert isinstance(m3, D3Dispersion) and m3.term.damping == "zero" and float(m3.term.s8) == 0.5
    m4 = build_model(from_dict({"model": {**base, "extra": {"dispersion": {"name": "d4"}}}}).model)
    assert isinstance(m4, D4Dispersion)
    m_default = build_model(from_dict({"model": {**base, "extra": {"dispersion": True}}}).model)
    assert isinstance(m_default, D4Dispersion)
    with pytest.raises(KeyError, match="unknown dispersion"):
        build_model(from_dict({"model": {**base, "extra": {"dispersion": {"name": "d5"}}}}).model)
    alone = build_model(from_dict({"model": {"name": "d3", "extra": {"s9": 1.0}}}).model)
    assert isinstance(alone, D3Dispersion) and alone.model is None and alone.node_feature_dim == 2


def test_nests_with_les_and_standalone_features():
    pytest.importorskip("e3nn")
    torch.manual_seed(0)
    both = build_model(_config("mace", long_range={"n_channels": 2, "dl": 3.0}).model)
    assert isinstance(both, LatentEwald) and isinstance(both.model, D3Dispersion)
    pos, _ = _cluster(10)
    z = [1, 8] * 5
    out = ForceStressOutput(both)(_graph(pos, z, both.cutoff, cell=np.eye(3) * 6.0))
    assert torch.isfinite(out["forces"]).all() and "energy_disp" in out
    les_d3 = LatentEwald(D3Dispersion(**FAST), n_channels=1)
    assert torch.isfinite(les_d3(_graph(pos, z, 9.0))["energy"]).all()


# --------------------------------------------------------------------------
# deploy channels
# --------------------------------------------------------------------------

@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("kind", ["standalone", "schnet"])
def test_torchscript_export_matches_eager(kind, periodic):
    from xnn.common.data import build_neighbor_list
    from xnn.common.deploy import LAMMPSWrapper, TorchScriptPotential
    torch.manual_seed(0)
    model = (D3Dispersion(s9=1.0, **FAST) if kind == "standalone"
             else build_model(_config("schnet").model))
    pos, _ = _cluster(10)
    z = [1, 8] * 5
    cell = np.eye(3) * 6.0 if periodic else None
    ref = ForceStressOutput(model, compute_stress=True)(_graph(pos, z, model.cutoff, cell=cell))
    scripted = torch.jit.script(TorchScriptPotential(model, model.cutoff).eval())
    cell_t = torch.tensor(cell) if periodic else None
    pbc_t = torch.tensor([periodic] * 3)
    out = scripted(torch.tensor(pos), torch.tensor(z), cell_t, pbc_t)
    assert abs(float(out["energy"]) - float(ref["energy"])) < 1e-12
    assert torch.allclose(out["forces"], ref["forces"], atol=1e-12)
    assert out["eeq_charges"].shape == (10, 0)                  # D3 has no charges
    if periodic:
        assert torch.allclose(out["stress"], ref["stress"][0], atol=1e-12)
    ei, cs = build_neighbor_list(torch.tensor(pos), model.cutoff, cell_t, pbc_t if periodic else None)
    cell_l = cell_t if periodic else torch.zeros(3, 3)
    out_l = scripted.forward_lammps(torch.tensor(pos), ei, cs, torch.tensor(z), cell_l)
    assert abs(float(out_l["energy"]) - float(ref["energy"])) < 1e-12
    lammps = torch.jit.script(LAMMPSWrapper(model, model.cutoff).eval())
    out_w = lammps(torch.tensor(pos), ei, cs, torch.tensor(z), cell_l)
    assert abs(float(out_w["total_energy"]) - float(ref["energy"])) < 1e-12


def test_ase_calculator_matches_eager():
    pytest.importorskip("ase")
    from ase import Atoms
    from xnn.common.deploy import XNNCalculator
    model = D3Dispersion(s9=1.0, **FAST)
    pos, z = _cluster(9, seed=7)
    atoms = Atoms(numbers=z, positions=pos, cell=np.eye(3) * 6.0, pbc=True)
    atoms.calc = XNNCalculator(ForceStressOutput(model, compute_stress=True), cutoff=model.cutoff)
    ref = ForceStressOutput(model, compute_stress=True)(_graph(pos, z, model.cutoff, cell=np.eye(3) * 6.0))
    assert abs(atoms.get_potential_energy() - float(ref["energy"])) < 1e-10
    assert np.abs(atoms.get_forces() - ref["forces"].detach().numpy()).max() < 1e-10


# --------------------------------------------------------------------------
# parity with the upstream simple-dftd3 package
# --------------------------------------------------------------------------

def _upstream_param(damping, **kw):
    from dftd3 import interface as di
    cls = {"bj": di.RationalDampingParam, "zero": di.ZeroDampingParam,
           "mzero": di.ModifiedZeroDampingParam, "op": di.OptimizedPowerDampingParam}[damping]
    return cls(**kw)


def _upstream(pos, z, param, cell=None, cutoffs=None):
    from dftd3.interface import DispersionModel
    model = DispersionModel(np.asarray(z), np.asarray(pos) / BOHR,
                            lattice=None if cell is None else np.asarray(cell) / BOHR,
                            periodic=None if cell is None else np.array([True] * 3))
    if cutoffs:
        model.set_realspace_cutoff(**cutoffs)
    return model.get_dispersion(param, grad=True)


def _compare(pos, z, damping, upstream_kw, cell=None, **opts):
    res = _upstream(pos, z, _upstream_param(damping, **upstream_kw), cell)
    model = D3Dispersion(damping=damping, **opts)
    g = _graph(pos, z, model.cutoff, cell=cell)
    out = ForceStressOutput(model, compute_stress=cell is not None)(g)
    energy = float(out["energy"]) / HARTREE
    assert abs(energy - float(res["energy"])) < 1e-14 * max(1.0, abs(energy) * 1e3)
    grad = -out["forces"].detach().numpy() / HARTREE * BOHR
    assert np.abs(grad - res["gradient"]).max() < 1e-13
    if cell is not None:
        vol = abs(np.linalg.det(np.asarray(cell))) / BOHR ** 3
        virial = out["stress"][0].detach().numpy() / HARTREE * BOHR ** 3 * vol
        assert np.abs(virial - res["virial"]).max() < 1e-13


VARIANTS = {
    "bj": (dict(s6=1.0, s8=1.2177, s9=1.0, a1=0.4145, a2=4.8593, alp=14.0),
           dict(s9=1.0)),
    "zero": (dict(s6=1.0, s8=0.928, s9=1.0, rs6=1.287, rs8=1.0, alp=14.0),
             dict(s9=1.0)),
    "mzero": (dict(s6=1.0, s8=0.000081, s9=1.0, rs6=2.077949, rs8=1.0, alp=14.0, bet=0.116755),
              dict(s8=0.000081, rs6=2.077949, bet=0.116755, s9=1.0)),
    "op": (dict(s6=0.8829, s8=0.0, s9=1.0, a1=0.150, a2=4.750, alp=14.0, bet=6.0),
           dict(s6=0.8829, s8=0.0, a1=0.150, a2=4.750, bet=6.0, s9=1.0)),
}


@pytest.mark.parametrize("damping", sorted(VARIANTS))
def test_parity_vs_sdftd3_molecules(damping):
    pytest.importorskip("dftd3")
    up, mine = VARIANTS[damping]
    _compare(*WATER, damping, up, **mine)
    _compare(*_benzene(), damping, up, **mine)
    pos, z = _cluster(12, seed=9)
    _compare(pos, z, damping, up, **mine)


@pytest.mark.parametrize("damping", ["bj", "zero"])
def test_parity_vs_sdftd3_crystals(damping):
    pytest.importorskip("dftd3")
    up, mine = VARIANTS[damping]
    a = 5.64
    _compare(np.array([[0.0, 0, 0], [a / 2, 0, 0]]), [11, 17], damping, up,
             cell=[[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]], **mine)
    a = 5.43
    cell = np.array([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]])
    cell[1, 0] += 0.5
    _compare(np.array([[0.0, 0, 0], [a / 4, a / 4, a / 4]]), [14, 14], damping, up, cell=cell, **mine)
    pos = np.concatenate([WATER[0] + shift for shift in
                          ([0.0, 0.0, 0.0], [3.1, 0.4, 0.2], [0.3, 3.0, 3.2])])
    _compare(pos, [8, 1, 1] * 3, damping, up, cell=np.eye(3) * 6.2, **mine)


def test_parity_vs_sdftd3_heavy_elements_and_cutoffs():
    """Actinide references (Z > 86, the re-parametrized part of the table) and
    matching non-default real-space cutoffs."""
    pytest.importorskip("dftd3")
    up, mine = VARIANTS["bj"]
    pos = np.array([[0.0, 0, 0], [2.0, 0, 0], [-2.0, 0, 0], [0, 2.0, 0], [0, -2.0, 0],
                    [0, 0, 2.0], [0, 0, -2.0]])
    _compare(pos, [92, 9, 9, 9, 9, 9, 9], "bj", up, **mine)
    pos, z = _cluster(12, seed=3)
    res = _upstream(pos, z, _upstream_param("bj", **up),
                    cutoffs=dict(disp2=12.0, disp3=9.0, cn=10.0))
    model = D3Dispersion(s9=1.0, cutoff_pair=12.0 * BOHR, cutoff_triple=9.0 * BOHR,
                         cutoff_cn=10.0 * BOHR)
    e = float(model(_graph(pos, z, model.cutoff))["energy"]) / HARTREE
    assert abs(e - float(res["energy"])) < 1e-15
