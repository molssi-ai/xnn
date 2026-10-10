"""Tests of the HDNNP family: 1G-4G models, atom-centered symmetry functions, the
Gaussian-charge Coulomb matrix, the RuNNer file bridge and, with ``XNN_RUNNER``
set, parity with the RuNNer executable."""
import itertools
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from xnn.common.config import from_dict
from xnn.common.data import collate, structure_to_graph
from xnn.common.data.runner_io import read_runner_data, write_runner_data
from xnn.common.models import ForceStressOutput, available_models, build_model
from xnn.common.models.charge_solve import gaussian_coulomb_matrix
from xnn.dnn.common.runner import load_runner_model, parse_input_nn, runner_feature_key
from xnn.dnn.featurizers import AtomCenteredSymmetryFunctions, cutoff_function
from xnn.dnn.models.hdnnp import HDNNP, NNP1G, default_symmetry_functions

HERE = Path(__file__).parent


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _molecule(n=7, seed=0, z=(8, 1, 1, 6, 1, 8, 1), box=3.0, charge=None):
    rng = np.random.default_rng(seed)
    pos = []
    while len(pos) < n:
        p = rng.uniform(0, box, 3)
        if all(np.linalg.norm(p - q) > 0.9 for q in pos):
            pos.append(p)
    s = {"pos": np.array(pos), "atomic_numbers": np.array([z[i % len(z)] for i in range(n)])}
    if charge is not None:
        s["total_charge"] = charge
    return s


def _cell(n=8, seed=0, z=(11, 17), length=5.5):
    rng = np.random.default_rng(seed)
    cell = np.diag([length, length * 1.1, length * 0.95]) + rng.normal(scale=0.1, size=(3, 3))
    frac = []
    while len(frac) < n:
        f = rng.uniform(0, 1, 3)
        if all(np.linalg.norm(((f - g) - np.round(f - g)) @ cell) > 1.3 for g in frac):
            frac.append(f)
    return {"pos": np.array(frac) @ cell, "atomic_numbers": np.array([z[i % len(z)] for i in range(n)]),
            "cell": cell, "pbc": np.ones(3, bool)}


def _acsf(species=(1, 6, 8), rc=4.0, mode="none"):
    cutoffs = [{"kind": "cosine", "r_cut": rc, "r_inner": 0.3}, {"kind": "tanh", "r_cut": rc - 0.5},
               {"kind": "polynomial", "r_cut": rc, "exponent": 3}, {"kind": "tanh_approx", "r_cut": rc}]
    funcs = []
    for zc in species:
        for zn in species:
            funcs += [{"element": zc, "type": 2, "neighbors": [zn], "eta": 0.5, "rs": 0.8, "cutoff": 1},
                      {"element": zc, "type": 2, "neighbors": [zn], "eta": 2.0, "rs": 0.0, "cutoff": 0}]
        funcs.append({"element": zc, "type": 1, "neighbors": [species[0]], "n_features": 2, "cutoff": 2})
        for za, zb in itertools.combinations_with_replacement(species, 2):
            funcs += [{"element": zc, "type": 3, "neighbors": [zb, za], "eta": 0.1, "lambda": -1.0,
                       "zeta": 2.0, "cutoff": 0},
                      {"element": zc, "type": 9, "neighbors": [za, zb], "eta": 0.05, "lambda": 1.0,
                       "zeta": 1.0, "cutoff": 3},
                      {"element": zc, "type": 8, "neighbors": [za, zb], "theta_s": 100.0, "eta": 0.001,
                       "cutoff": 2}]
    return AtomCenteredSymmetryFunctions(list(species), cutoffs, funcs, mode=mode)


# cutoff functions
def test_cutoff_functions_values_and_support():
    """Values at the inner radius, the midpoint and the cutoff for every kind."""
    r = torch.tensor([0.1, 0.5, 2.75, 5.0, 6.0])
    cos = cutoff_function(r, "cosine", 5.0, 0.5)
    assert cos[0] == 1.0 and cos[1] == 1.0 and abs(float(cos[2]) - 0.5) < 1e-15 and cos[3] == 0
    tanh = cutoff_function(r, "tanh", 5.0, 0.5)
    assert abs(float(tanh[0]) - math.tanh(1.0) ** 3) < 1e-15 and tanh[3] == 0
    approx = cutoff_function(r, "tanh_approx", 5.0, 0.5)
    assert abs(float(approx[0]) - (7 / 9) ** 3) < 1e-15
    poly = cutoff_function(r, "polynomial", 5.0, exponent=3)
    assert abs(float(poly[1]) - (1 - 0.01) ** 3) < 1e-15 and poly[4] == 0
    assert cutoff_function(r, "hard", 5.0).tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]


def test_cutoff_functions_vanish_smoothly():
    """The smooth kinds reach zero with zero slope at the cutoff."""
    r = torch.tensor([4.999999], requires_grad=True)
    for kind in ("cosine", "tanh", "tanh_approx", "polynomial"):
        f = cutoff_function(r, kind, 5.0, 0.0, 2)
        (g,) = torch.autograd.grad(f.sum(), r)
        assert float(f) < 1e-9 and abs(float(g)) < 1e-4


# symmetry functions against a direct evaluation of the formulas
def _brute_acsf(acsf, s):
    pos = np.asarray(s["pos"])
    z = np.asarray(s["atomic_numbers"])
    n = len(z)
    images = [np.zeros(3)]
    if s.get("cell") is not None:
        cell = np.asarray(s["cell"])
        images = [np.array(t) @ cell for t in itertools.product(range(-2, 3), repeat=3)]
    out = np.zeros((n, acsf.output_dim))

    def fc(c, r):
        return float(cutoff_function(torch.tensor([r]), c["kind"], c["r_cut"], c["r_inner"], c["exponent"]))

    for i in range(n):
        nb = []
        for j in range(n):
            for t in images:
                d = pos[i] - (pos[j] + t)
                r = np.linalg.norm(d)
                if r > 1e-9 and r < acsf.cutoff:
                    nb.append((z[j], d, r))
        for f in acsf.functions:
            if f["element"] != z[i]:
                continue
            c = acsf.cutoffs[f["cutoff"]]
            val = 0.0
            if f["type"] in (1, 2):
                for zj, d, r in nb:
                    if zj != f["neighbors"][0]:
                        continue
                    val += fc(c, r) ** f["power"] if f["type"] == 1 else \
                        math.exp(-f["eta"] * (r - f["rs"]) ** 2) * fc(c, r)
            else:
                for a in range(len(nb)):
                    for b in range(len(nb)):
                        if a == b:
                            continue
                        (zj, dj, rj), (zk, dk, rk) = nb[a], nb[b]
                        if tuple(sorted((zj, zk))) != f["neighbors"]:
                            continue
                        rjk = np.linalg.norm(dj - dk)
                        cos = dj @ dk / (rj * rk)
                        if f["type"] == 8:
                            th = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
                            ts, eta = f["theta_s"], f["eta"]
                            g = sum(math.exp(-eta * (th - t) ** 2)
                                    for t in (ts, 360 - ts, -ts, 360 + ts))
                            val += 0.5 * g * fc(c, rj) * fc(c, rk) * fc(c, rjk)
                            continue
                        ang = max(0.0, 1 + f["lambda"] * cos) ** f["zeta"]
                        if f["type"] == 3:
                            rad = math.exp(-f["eta"] * (rj * rj + rk * rk + rjk * rjk)) * fc(c, rjk)
                        else:
                            rad = math.exp(-f["eta"] * (rj * rj + rk * rk))
                        # the ordered double sum with 2^-zeta == the unordered one with 2^(1-zeta)
                        val += 2.0 ** (-f["zeta"]) * ang * rad * fc(c, rj) * fc(c, rk)
            out[i, f["column"]] = val
    return out


@pytest.mark.parametrize("periodic", [False, True])
def test_symmetry_functions_match_the_formulas(periodic):
    """Every type, cutoff kind and element combination equals a direct double loop."""
    acsf = _acsf(species=(11, 17) if periodic else (1, 6, 8), rc=3.6)
    s = _cell(seed=3) if periodic else _molecule(seed=2)
    g = acsf(structure_to_graph(s, acsf.cutoff)).detach().numpy()
    want = _brute_acsf(acsf, s)
    np.testing.assert_allclose(g, want, rtol=1e-12, atol=1e-13)
    assert np.abs(want).sum() > 1.0


def test_descriptor_layout_and_padding():
    """Columns follow the list order per element; rows beyond an element's count are zero."""
    acsf = AtomCenteredSymmetryFunctions(
        [1, 8], [{"kind": "cosine", "r_cut": 4.0}],
        [{"element": 8, "type": 2, "neighbors": [1], "eta": 1.0},
         {"element": 1, "type": 2, "neighbors": [8], "eta": 1.0},
         {"element": 8, "type": 2, "neighbors": [8], "eta": 0.5},
         {"element": 8, "type": 3, "neighbors": [1, 1], "eta": 0.1, "lambda": 1.0, "zeta": 1.0}])
    assert acsf.n_features == {1: 1, 8: 3} and acsf.output_dim == 3
    s = _molecule(n=3, z=(8, 1, 1), seed=1)
    g = acsf(structure_to_graph(s, 4.0))
    assert torch.all(g[1:, 1:] == 0)
    with pytest.raises(ValueError):
        AtomCenteredSymmetryFunctions([1], [{"kind": "cosine", "r_cut": 4.0}],
                                      [{"element": 1, "type": 3, "neighbors": [1, 1], "lambda": 0.5}])


def test_scaling_modes():
    """fit_scaling statistics and every scaling formula."""
    acsf = _acsf()
    graphs = [structure_to_graph(_molecule(seed=k), acsf.cutoff) for k in range(4)]
    raw = [acsf.raw(g) for g in graphs]
    acsf.fit_scaling(graphs)
    sp = acsf._z2i[graphs[0].atomic_numbers]
    rows = torch.cat([r[g.atomic_numbers == 8] for r, g in zip(raw, graphs)])
    i_o = acsf.species.index(8)
    n_o = acsf.n_features[8]
    torch.testing.assert_close(acsf.stat_min[i_o, :n_o], rows.min(0).values[:n_o])
    torch.testing.assert_close(acsf.stat_avg[i_o, :n_o], rows.mean(0)[:n_o])
    lo, hi, avg = (t[sp] for t in (acsf.stat_min, acsf.stat_max, acsf.stat_avg))
    span = torch.where(hi > lo, hi - lo, torch.ones_like(hi))
    valid = acsf._valid[sp]
    want = {"scale": (raw[0] - lo) / span, "center": raw[0] - avg,
            "center_scale": (raw[0] - avg) / span, "range": -1.0 + 2.0 * (raw[0] - lo) / span}
    acsf.scale_range = (-1.0, 1.0)
    for mode, ref in want.items():
        acsf.mode = mode
        torch.testing.assert_close(acsf(graphs[0]), torch.where(valid, ref, torch.zeros_like(ref)))


# Gaussian-charge electrostatics
def test_gaussian_coulomb_matrix_molecule():
    """Pair kernel erf(r / sqrt(2) gamma) / r, Gaussian self term, point-charge limit."""
    pos = torch.tensor([[0.0, 0, 0], [1.5, 0, 0], [0, 2.0, 0.5]])
    sigma = torch.tensor([0.4, 0.6, 0.0])
    a = gaussian_coulomb_matrix(pos, sigma)
    r01 = 1.5
    assert abs(float(a[0, 1]) - math.erf(r01 / math.sqrt(2 * (0.16 + 0.36))) / r01) < 1e-15
    assert abs(float(a[0, 0]) - 1 / (0.4 * math.sqrt(math.pi))) < 1e-15
    assert float(a[2, 2]) == 0.0
    r12 = float(torch.linalg.norm(pos[1] - pos[2]))
    assert abs(float(a[1, 2]) - math.erf(r12 / (math.sqrt(2) * 0.6)) / r12) < 1e-15
    torch.testing.assert_close(a, a.t())
    pts = gaussian_coulomb_matrix(pos, torch.zeros(3))
    assert abs(float(pts[0, 1]) - 1 / r01) < 1e-15


def test_ewald_madelung_and_extensivity():
    """Rocksalt point charges give the Madelung constant; Gaussian charges are extensive."""
    a0 = 5.64
    base = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    pos = np.concatenate([base, base + 0.5]) * a0
    q = torch.tensor([1.0] * 4 + [-1.0] * 4)
    cell = torch.eye(3) * a0
    amat = gaussian_coulomb_matrix(torch.tensor(pos), torch.zeros(8), cell)
    energy = 0.5 * q @ amat @ q
    assert abs(float(energy) / 4 * (a0 / 2) + 1.747564594633182) < 1e-8
    # Gaussian charges: a 2x1x1 supercell carries twice the energy, also for a charged cell
    s = _cell(n=5, seed=4)
    sig = torch.tensor([0.3, 0.5, 0.3, 0.5, 0.3])
    qs = torch.tensor([0.7, -0.4, 0.2, -0.6, 0.3])
    c = torch.tensor(s["cell"])
    p = torch.tensor(s["pos"])
    e1 = 0.5 * qs @ gaussian_coulomb_matrix(p, sig, c) @ qs
    p2 = torch.cat([p, p + c[0]])
    c2 = torch.stack([2 * c[0], c[1], c[2]])
    q2 = torch.cat([qs, qs])
    e2 = 0.5 * q2 @ gaussian_coulomb_matrix(p2, torch.cat([sig, sig]), c2) @ q2
    assert abs(float(e2) - 2 * float(e1)) < 1e-9 * abs(float(e1))


def test_ewald_tends_to_the_molecule_in_a_large_box():
    """A neutral cluster in a growing box approaches the molecular sum."""
    pos = torch.tensor([[0.0, 0, 0], [1.0, 0.2, 0], [0.3, 1.1, 0.4]])
    sig = torch.tensor([0.3, 0.4, 0.5])
    q = torch.tensor([0.5, -0.2, -0.3])
    e_mol = float(0.5 * q @ gaussian_coulomb_matrix(pos, sig) @ q)
    errs = []
    for length in (20.0, 40.0):
        e = float(0.5 * q @ gaussian_coulomb_matrix(pos + 5, sig, torch.eye(3) * length) @ q)
        errs.append(abs(e - e_mol))
    assert errs[1] < errs[0] / 6 and errs[1] < 1e-4


# models
def _model(generation, species=(1, 6, 8), seed=0, **kw):
    torch.manual_seed(seed)
    cutoffs, funcs = default_symmetry_functions(list(species), 4.0, n_radial=4)
    acsf = AtomCenteredSymmetryFunctions(list(species), cutoffs, funcs)
    return HDNNP(list(species), acsf, generation=generation, hidden=(8, 6),
                 activation=["tanh", "softplus", "linear"], **kw).double()


def _graph(s, model):
    return collate([structure_to_graph(s, model.cutoff)])


def _rotate(s, seed=0):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    out = dict(s, pos=np.asarray(s["pos"]) @ q.T + 0.7)
    if s.get("cell") is not None:
        out["cell"] = np.asarray(s["cell"]) @ q.T
    return out


@pytest.mark.parametrize("generation", [2, 3, 4])
@pytest.mark.parametrize("periodic", [False, True])
def test_invariance_and_force_equivariance(generation, periodic):
    """Energies are invariant under rotation, translation and permutation; forces rotate."""
    species = (11, 17) if periodic else (1, 6, 8)
    model = ForceStressOutput(_model(generation, species))
    s = _cell(seed=1) if periodic else _molecule(seed=5)
    rng = np.random.default_rng(1)
    a = model(_graph(s, model.model))
    b = model(_graph(dict(_rotate(s)), model.model))
    assert abs(float(a["energy"] - b["energy"])) < 1e-10 * max(1.0, abs(float(a["energy"])))
    np.testing.assert_allclose(b["forces"].detach().numpy(),
                               a["forces"].detach().numpy() @ np.linalg.qr(
                                   np.random.default_rng(0).normal(size=(3, 3)))[0].T, atol=1e-9)
    perm = rng.permutation(len(s["atomic_numbers"]))
    c = model(_graph(dict(s, pos=np.asarray(s["pos"])[perm],
                          atomic_numbers=np.asarray(s["atomic_numbers"])[perm]), model.model))
    assert abs(float(a["energy"] - c["energy"])) < 1e-10 * max(1.0, abs(float(a["energy"])))
    if generation >= 3:
        np.testing.assert_allclose(c["charges"].detach().numpy(),
                                   a["charges"].detach().numpy()[perm], atol=1e-10)


@pytest.mark.parametrize("generation,periodic,screened", [(3, False, True), (3, True, False),
                                                          (4, False, True), (4, True, True)])
def test_forces_are_the_energy_gradient(generation, periodic, screened):
    """Analytic forces equal central finite differences of the energy."""
    species = (11, 17) if periodic else (1, 6, 8)
    screening = {"kind": "cosine", "r_cut": 3.0, "r_inner": 0.5} if screened else None
    model = ForceStressOutput(_model(generation, species, screening=screening))
    s = _cell(n=6, seed=2) if periodic else _molecule(n=5, seed=6, charge=1.0)
    f = model(_graph(s, model.model))["forces"].detach().numpy()
    h = 1e-5
    for atom, axis in ((0, 0), (2, 1), (4, 2)):
        e = []
        for sign in (1, -1):
            pos = np.array(s["pos"], dtype=float)
            pos[atom, axis] += sign * h
            e.append(float(model.model(_graph(dict(s, pos=pos), model.model))["energy"]))
        assert abs(-(e[0] - e[1]) / (2 * h) - f[atom, axis]) < 1e-7


def test_3g_charges_sum_to_the_total_charge():
    """3G charges are the network outputs shifted uniformly to the total charge."""
    model = _model(3)
    s = _molecule(seed=3, charge=-1.0)
    out = model(_graph(s, model))
    assert abs(float(out["charges"].sum()) + 1.0) < 1e-12
    shift = out["charges_raw"] - out["charges"]
    assert float(shift.max() - shift.min()) < 1e-12
    raw = _model(3, constrain_charges=False)(_graph(s, model))
    torch.testing.assert_close(raw["charges"], raw["charges_raw"])


def test_3g_electrostatics_and_screening():
    """E_elec is the Gaussian-charge energy; screening scales pairs by 1 - fc and drops self terms."""
    screen = {"kind": "cosine", "r_cut": 2.5}
    plain, screened = _model(3), _model(3, screening=screen)
    s = _molecule(seed=4)
    q = plain(_graph(s, plain))["charges"].detach()
    pos = torch.tensor(s["pos"])
    z = torch.tensor(s["atomic_numbers"])
    sigma = plain.gaussian_widths[z]
    amat = gaussian_coulomb_matrix(pos, sigma)
    e_ref = 0.5 * q @ amat @ q * plain.coulomb_constant
    assert abs(float(plain(_graph(s, plain))["energy_elec"]) - float(e_ref)) < 1e-10
    want = 0.0
    for i in range(len(q)):
        for j in range(i + 1, len(q)):
            r = float(torch.linalg.norm(pos[i] - pos[j]))
            g = math.sqrt(float(sigma[i]) ** 2 + float(sigma[j]) ** 2)
            fc = float(cutoff_function(torch.tensor([r]), "cosine", 2.5))
            want += float(q[i] * q[j]) * math.erf(r / (math.sqrt(2) * g)) / r * (1 - fc)
    got = float(screened(_graph(s, screened))["energy_elec"])
    assert abs(got - want * plain.coulomb_constant) < 1e-10


def test_4g_charge_equilibration():
    """4G charges solve chi + J q + A q = mu with sum q = Q and respond to distant changes."""
    model = _model(4, species=(1, 6, 8))
    s = _molecule(n=8, seed=7, charge=1.0, box=5.0)
    out = model(_graph(s, model))
    q = out["charges"].detach()
    assert abs(float(q.sum()) - 1.0) < 1e-12
    pos = torch.tensor(s["pos"])
    amat = model.coulomb_constant * gaussian_coulomb_matrix(pos, model.gaussian_widths[torch.tensor(s["atomic_numbers"])])
    resid = out["electronegativities"] + out["hardness"] * q + amat @ q
    torch.testing.assert_close(resid.detach(), out["chemical_potential"].expand_as(resid).detach(),
                               rtol=0, atol=1e-10)
    # moving the farthest atom changes the charge of every atom (non-local response)
    far = int(np.argmax(np.linalg.norm(s["pos"] - s["pos"][0], axis=1)))
    pos2 = np.array(s["pos"])
    pos2[far] += [0.0, 0.0, 0.3]
    q2 = model(_graph(dict(s, pos=pos2), model))["charges"].detach()
    assert abs(float(q2[0] - q[0])) > 1e-8
    # the total charge enters the energy
    e_neutral = float(model(_graph(dict(s, total_charge=0.0), model))["energy"])
    assert abs(e_neutral - float(out["energy"])) > 1e-6


def test_4g_charge_neuron_and_hardness_network():
    """The charge is the last short-range input, scaled; the hardness can be a network."""
    model = _model(4, hardness="network")
    assert model.hardness_nets is not None and model.hardness_raw is None
    net = model.element_nets.nets["8"][0]
    assert net.in_features == model.featurizer.n_features[8] + 1
    s = _molecule(seed=8)
    e0 = float(model(_graph(s, model))["energy"])
    model.charge_input_factor[8] = 3.0
    assert abs(float(model(_graph(s, model))["energy"]) - e0) > 1e-8
    off = _model(4, charge_neuron=False)
    assert off.element_nets.nets["8"][0].in_features == off.featurizer.n_features[8]


def test_mixed_batch_matches_single_structures():
    """A batch of clusters and cells gives the energies of the structures one by one."""
    model = _model(4, species=(11, 17))
    structs = [_cell(seed=1), _molecule(n=4, z=(11, 17), seed=2, charge=0.0), _cell(n=6, seed=3)]
    batch = collate([structure_to_graph(s, model.cutoff) for s in structs])
    e = model(batch)["energy"].detach()
    for k, s in enumerate(structs):
        assert abs(float(model(_graph(s, model))["energy"]) - float(e[k])) < 1e-9


# 1G
def test_nnp1g_invariances_and_symmetrization():
    """1G energies are rotation invariant; symmetrization makes them permutation invariant."""
    s = _molecule(n=4, z=(8, 1, 1, 1), seed=9)
    torch.manual_seed(0)
    plain = NNP1G([8, 1, 1, 1]).double()
    torch.manual_seed(0)
    sym = NNP1G([8, 1, 1, 1], symmetrize=True).double()
    g = _graph(s, plain)
    assert abs(float(plain(g)["energy"] - plain(_graph(_rotate(s), plain))["energy"])) < 1e-12
    swapped = dict(s, pos=np.asarray(s["pos"])[[0, 2, 1, 3]])
    assert abs(float(plain(g)["energy"] - plain(_graph(swapped, plain))["energy"])) > 1e-8
    assert abs(float(sym(g)["energy"] - sym(_graph(swapped, sym))["energy"])) < 1e-12
    assert sym._perm_pairs.shape[0] == 6
    sym.fit_scaling([g, _graph(swapped, sym)])
    assert abs(float(sym(g)["energy"] - sym(_graph(swapped, sym))["energy"])) < 1e-12
    with pytest.raises(ValueError):
        plain(_graph(_molecule(n=3, z=(8, 1, 1)), plain))


# configs and registry
def test_registered_and_built_from_config():
    """hdnnp / nnp1g are registered; configs build 1G-4G models and translate RuNNer keys."""
    assert "hdnnp" in available_models() and "nnp1g" in available_models()
    m = build_model(from_dict({"model": {"name": "hdnnp", "extra": {"species": [1, 8]}}}).model)
    assert isinstance(m, HDNNP) and m.generation == 2 and m.cutoff == 4.0
    m1 = build_model(from_dict({"model": {"name": "hdnnp", "extra": {
        "generation": 1, "species": [8, 1, 1]}}}).model)
    assert isinstance(m1, NNP1G)
    m4 = build_model(from_dict({"model": {"name": "hdnnp", "cutoff": 5.0, "nnp_generation": 4,
                                          "elements": ["H", "O"], "screening": {"r_cut": 4.0},
                                          "atomic_energies": {"H": -13.6, "O": -2041.0}}}).model)
    assert m4.generation == 4 and m4.species == [1, 8] and float(m4._self_energies_by_z[1]) == -13.6
    legacy = build_model(from_dict({"model": {"name": "hdnnp", "extra": {
        "species": [1, 8], "etas": [0.5, 1.0], "rs": [0.0]}}}).model)
    assert legacy.featurizer.n_features == {1: 4, 8: 4}
    with pytest.raises(ValueError):
        build_model(from_dict({"model": {"name": "hdnnp", "cutoff": 3.0, "extra": {
            "species": [1], "cutoffs": [{"r_cut": 5.0}],
            "symmetry_functions": [{"element": 1, "type": 2, "neighbors": [1]}]}}}).model)


def test_trains_charges_and_energies():
    """A few optimizer steps on energies, forces and charges lower the 4G loss."""
    from xnn.common.train.losses import weighted_loss
    model = ForceStressOutput(_model(4))
    rng = np.random.default_rng(0)
    structs = []
    for k in range(4):
        s = _molecule(n=5, seed=10 + k, charge=0.0)
        s["energy"], s["forces"] = float(rng.normal()), rng.normal(size=(5, 3)) * 0.1
        s["charges"] = rng.normal(scale=0.2, size=5)
        s["charges"] -= s["charges"].mean()
        structs.append(s)
    batch = collate([structure_to_graph(s, model.model.cutoff) for s in structs])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(60):
        loss, _ = weighted_loss(model(batch), batch, 1.0, 1.0, 0.0, charge_weight=10.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < 0.6 * losses[0]
    assert model.model.hardness_raw.grad is not None


# RuNNer files
def test_runner_data_round_trip(tmp_path):
    """input.data written and read back, with unit conversion and periodic cells."""
    structs = [dict(_molecule(seed=1, charge=1.0), energy=-3.5, forces=np.ones((7, 3)),
                    charges=np.linspace(-0.3, 0.3, 7)), _cell(seed=2)]
    write_runner_data(structs, tmp_path / "input.data")
    back = read_runner_data(tmp_path / "input.data")
    np.testing.assert_allclose(back[0]["pos"], structs[0]["pos"], rtol=1e-14)
    np.testing.assert_allclose(back[0]["forces"], 1.0, rtol=1e-14)
    np.testing.assert_allclose(back[0]["charges"], structs[0]["charges"], atol=1e-15)
    assert back[0]["total_charge"] == 1.0 and abs(back[0]["energy"] + 3.5) < 1e-12
    np.testing.assert_allclose(back[1]["cell"], structs[1]["cell"], rtol=1e-14)
    custom = tmp_path / "custom.data"
    custom.write_text("begin position(3) element hirshfeld_volume charges forces(3)\n"
                      "atom 0 0 0 H 3.0 0.25 0.1 0.2 0.3\natom 1.4 0 0 H 3.0 -0.25 0 0 0\n"
                      "energy -1.0\ncharge 0.0\nend\n")
    s = read_runner_data(custom, units="atomic")[0]
    assert s["charges"].tolist() == [0.25, -0.25] and s["forces"][0].tolist() == [0.1, 0.2, 0.3]


def test_hdnnp4g_dataset_reader(tmp_path):
    """The hdnnp4g hub reads energies, forces, cells and Hirshfeld charges from extended XYZ."""
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import write

    from xnn.common.data import load_dataset
    from xnn.common.data.hub.hdnnp4g import _read_extxyz

    c = _cell(seed=4)
    atoms = Atoms(numbers=c["atomic_numbers"], positions=c["pos"], cell=c["cell"], pbc=True)
    q = np.linspace(-0.5, 0.5, len(atoms))
    atoms.set_initial_charges(q)
    f = np.arange(3 * len(atoms), dtype=float).reshape(-1, 3)
    atoms.calc = SinglePointCalculator(atoms, energy=-7.25, forces=f)
    write(tmp_path / "frames.xyz", [atoms, atoms], format="extxyz")
    back = _read_extxyz(tmp_path / "frames.xyz")
    assert len(back) == 2 and back[0]["total_charge"] == 0.0 and back[0]["energy"] == -7.25
    np.testing.assert_allclose(back[0]["charges"], q, atol=1e-12)
    np.testing.assert_allclose(back[0]["forces"], f, atol=1e-12)
    np.testing.assert_allclose(back[0]["cell"], c["cell"], atol=1e-12)
    mol = Atoms("NaCl", positions=[[0, 0, 0], [2.4, 0, 0]])
    mol.arrays["charge"] = np.array([0.9, 0.1])
    mol.calc = SinglePointCalculator(mol, energy=-1.0, forces=np.zeros((2, 3)))
    write(tmp_path / "mol.xyz", mol, format="extxyz")
    m = _read_extxyz(tmp_path / "mol.xyz")[0]
    assert m["total_charge"] == 1.0 and m["charges"].tolist() == [0.9, 0.1]
    with pytest.raises(ValueError):
        load_dataset("hdnnp4g", system="water", cache_dir=tmp_path)
    with pytest.raises(ValueError):
        load_dataset("hdnnp4g", split="train", cache_dir=tmp_path)


def test_runner_feature_order():
    """Type, cutoff radius and kind, parameters, then elements (fluorine sorts last)."""
    cos, tanh = {"kind": "cosine", "r_cut": 8.0}, {"kind": "tanh", "r_cut": 8.0}
    rows = [({"type": 3, "neighbors": [1, 1], "eta": 0.0, "lambda": 1.0, "zeta": 1.0}, cos),
            ({"type": 2, "neighbors": [9], "eta": 0.1, "rs": 0.0}, cos),
            ({"type": 2, "neighbors": [8], "eta": 0.1, "rs": 0.0}, cos),
            ({"type": 2, "neighbors": [1], "eta": 0.1, "rs": 0.0}, tanh),
            ({"type": 2, "neighbors": [1], "eta": 0.05, "rs": 1.0}, cos)]
    order = sorted(range(len(rows)), key=lambda k: runner_feature_key(*rows[k]))
    assert order == [4, 2, 1, 3, 0]


def test_runner_model_files_load(tmp_path):
    """A generated RuNNer model loads with its feature order, scaling and weights."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("parity", HERE / "hdnnp_runner_parity.py")
    parity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parity)
    for case in ("2g_molecules", "3g_screened", "4g_molecules", "4g_hardness_nn"):
        d = tmp_path / case
        d.mkdir()
        parity.make_case(case, d)
        s = parse_input_nn(d / "input.nn")
        model = load_runner_model(d)
        assert model.generation == s["generation"]
        out = parity.run_xnn(d)
        assert np.isfinite(out["energy"]).all() and np.isfinite(out["forces"]).all()


@pytest.mark.skipif(not os.environ.get("XNN_RUNNER"), reason="needs the RuNNer executable (XNN_RUNNER)")
@pytest.mark.parametrize("case", ["2g_molecules", "2g_cells", "2g_range", "3g_molecules", "3g_screened",
                                  "3g_cells", "3g_point", "3g_point_cells", "4g_molecules", "4g_cells",
                                  "4g_hardness_nn", "4g_hardness_nn_cells", "3g_separate", "4g_separate",
                                  "au2mgo"])
def test_runner_parity(case):
    """Energies, forces and charges equal the RuNNer executable's in float64."""
    import importlib.util
    if case == "au2mgo" and not os.environ.get("XNN_RUNNER_SRC"):
        pytest.skip("needs the RuNNer source tree (XNN_RUNNER_SRC)")
    spec = importlib.util.spec_from_file_location("parity", HERE / "hdnnp_runner_parity.py")
    parity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parity)
    assert parity.run_case(case) < 1e-9
