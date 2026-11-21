"""Tests of the GFN tight-binding methods (GFN1-xTB, GFN2-xTB).

Integrals, derivatives, invariances and the classical terms are checked
without the reference program. Fidelity to the xtb program runs when
``XNN_XTB`` names its binary (see ``sqmnn_xtb_fidelity.py``).
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from xnn.common.data import collate, structure_to_graph
from xnn.common.models import ForceStressOutput
from xnn.sqmnn.common import xtb_published
from xnn.sqmnn.common.xtb_published import parse_parameters
from xnn.sqmnn.common.xtb_tables import GFN2_RADII, STO_NG, n_gaussians
from xnn.sqmnn.models import GFN0, GFN1, GFN2
from xnn.sqmnn.models.gaussian import N_SPHERICAL, primitive_norm, self_overlap, shell_pair_integrals
from xnn.sqmnn.models.xtb import BOHR, gfn1_reference_occupations, gfn2_reference_occupations, valence_electrons

#: Convergence of the self-consistent charges in the tests (xtb's default accuracy is 1).
TIGHT = 1e-6
WATER = ([8, 1, 1], [[0, 0, 0.12], [0, 0.76, -0.47], [0.05, -0.75, -0.49]])
FORMAMIDE = ([6, 8, 7, 1, 1, 1], [[0, 0.42, 0], [1.2, 0.5, 0.02], [-0.72, -0.73, 0], [-0.55, 1.36, 0.01],
                                  [-1.73, -0.69, 0.05], [-0.27, -1.63, -0.03]])
HALOGEN_BOND = ([7, 1, 1, 1, 35, 9], [[0, 0, 0], [0.94, 0, -0.33], [-0.47, 0.814, -0.33], [-0.47, -0.814, -0.33],
                                      [0.05, 0, 2.9], [0.1, 0, 4.66]])
TICL = ([22, 17], [[0, 0, 0], [0.02, 0, 2.2]])


@pytest.fixture(scope="module")
def gfn1():
    return GFN1(accuracy=TIGHT)


@pytest.fixture(scope="module")
def gfn2():
    return GFN2(accuracy=TIGHT)


def _graph(z, pos, charge=0.0, mult=None):
    s = {"pos": torch.tensor(np.asarray(pos, dtype=np.float64)), "atomic_numbers": np.asarray(z),
         "total_charge": float(charge)}
    if mult is not None:
        s["spin_multiplicity"] = float(mult)
    return structure_to_graph(s, 0.1)


def _rotation(seed=0):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def test_registration_and_config():
    """Both methods are registered and build from a model config."""
    from xnn.common.config.schema import ModelConfig
    from xnn.common.models.registry import build_model
    for name, cls in (("gfn0_xtb", GFN0), ("gfn1_xtb", GFN1), ("gfn2_xtb", GFN2)):
        extra = {"electronic_temperature": 500.0, "accuracy": 1e-5}
        model = build_model(ModelConfig(name=name, cutoff=0.1, extra=extra))
        assert isinstance(model, cls)
        assert model.electronic_temperature == 500.0 and model.accuracy == 1e-5


def test_model_config_files():
    """The config files list every option of the constructors at its default."""
    import inspect

    import yaml

    from xnn.sqmnn.models import GFN
    root = Path(__file__).resolve().parents[2] / "configs" / "model"
    sig = inspect.signature(GFN.__init__)
    for name in ("gfn0_xtb", "gfn1_xtb", "gfn2_xtb"):
        cfg = yaml.safe_load((root / f"{name}.yaml").read_text())
        for key, par in sig.parameters.items():
            if key == "self":
                continue
            assert key in cfg, (name, key)
            if key != "parameters":
                assert cfg[key] == par.default, (name, key)


def test_parameter_file_parsing():
    """The xtb parameter format: globals, pair factors and element blocks."""
    text = """$info
name test
$end
$globpar
ks          1.85000
kexp        1.50000
$end
$pairpar
  1  6   0.9500
$end
$Z= 6 a comment
 ao=2s2p
 lev=  -13.0   -10.0
 exp=    2.0    1.8
 GAM=    0.5
$end
"""
    par = parse_parameters(text, "test")
    assert par.globals == {"ks": 1.85, "kexp": 1.5}
    assert par.pairs == {(6, 1): 0.95}
    assert par.shells[6] == [(2, 0), (2, 1)]
    assert par.value(6, "LEV", 1) == -10.0 and par.value(6, "GAM") == 0.5 and par.value(6, "POLYS") == 0.0


def test_download_provenance_note(tmp_path):
    """Downloaded parameter files get a note with their source and license beside them."""
    import shutil
    import warnings

    from xnn.common.models.hub.provenance import NOTE, ParameterLicenseWarning, write_license_note
    cached = xtb_published.parameter_file("gfn2")
    assert "LGPL-3.0" in (cached.parent / NOTE).read_text()
    folder = tmp_path / "sqmnn" / f"xtb-{xtb_published.RELEASE.lstrip('v')}"
    folder.mkdir(parents=True)
    shutil.copy(cached, folder / cached.name)
    with pytest.warns(ParameterLicenseWarning, match="accept its terms"):
        again = xtb_published.parameter_file("gfn2", cache_dir=tmp_path)
    text = (again.parent / NOTE).read_text()
    assert again.parent == folder and "grimme-lab/xtb" in text and "not part of xnn" in text
    with warnings.catch_warnings():
        warnings.simplefilter("error", ParameterLicenseWarning)
        xtb_published.parameter_file("gfn2", cache_dir=tmp_path)        # the note is there: no second warning
    blocked = tmp_path / "file"
    blocked.write_text("")
    assert not write_license_note(blocked / "sub", "program", "v1", "source", "license", ("url",)).exists()


def test_published_parameters_offline(tmp_path, monkeypatch):
    """Without network access an empty cache raises with a hint."""
    monkeypatch.setenv("XNN_OFFLINE", "1")
    with pytest.raises(FileNotFoundError, match="XNN_OFFLINE"):
        xtb_published.parameter_file("gfn2", cache_dir=tmp_path)


def test_published_parameter_sets(gfn1, gfn2):
    """The published sets cover Z = 1-86 with the expected shells."""
    assert gfn1.supported == tuple(range(1, 87)) and gfn2.supported == tuple(range(1, 87))
    assert gfn1.par.shells[1] == [(1, 0), (2, 0)] and gfn2.par.shells[1] == [(1, 0)]
    assert gfn2.par.shells[26] == [(3, 2), (4, 0), (4, 1)]


def test_reference_occupations_and_valence(gfn1, gfn2):
    """Reference occupations sum to the valence electrons."""
    for z in range(1, 87):
        nval = valence_electrons(z)
        assert sum(gfn2_reference_occupations(gfn2.par, z)) == pytest.approx(nval)
        assert sum(gfn1_reference_occupations(gfn1.par, z)) == pytest.approx(nval)
    assert gfn2_reference_occupations(gfn2.par, 6) == [1.0, 3.0]
    assert gfn1_reference_occupations(gfn1.par, 6) == [2.0, 2.0]
    assert gfn2_reference_occupations(gfn2.par, 26) == [6.0, 1.0, 1.0]
    assert gfn1_reference_occupations(gfn1.par, 26) == [6.0, 2.0, 0.0]
    assert valence_electrons(30) == 2 and valence_electrons(64) == 3 and valence_electrons(79) == 11
    with pytest.raises(ValueError):
        valence_electrons(87)


def test_sto_tables_are_normalized():
    """The STO-nG expansions are normalized to the precision of their published digits."""
    for (n, l, k), (a, c) in STO_NG.items():
        norm = float(self_overlap(torch.tensor(a, dtype=torch.float64), torch.tensor(c, dtype=torch.float64), l))
        assert abs(norm - 1.0) < 3e-7, (n, l, k)
    assert n_gaussians("gfn1", 2, 0, second_s=True) == 3
    assert n_gaussians("gfn2", 6, 1) == 6 and n_gaussians("gfn2", 4, 2) == 3 and n_gaussians("gfn1", 1, 0) == 4


def _grid(center, n=64, half=12.0):
    x, w = np.polynomial.legendre.leggauss(n)
    x, w = x * half, w * half
    g = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3) + center
    return g, (w[:, None, None] * w[None, :, None] * w[None, None, :]).reshape(-1)


def _shell_values(g, center, alpha, coef, l):
    """Spherical components of a contracted shell on the grid points ``g``."""
    d = g - center
    r2 = (d * d).sum(-1)
    rad = sum(c * float(primitive_norm(torch.tensor(a, dtype=torch.float64), l)) * np.exp(-a * r2)
              for a, c in zip(alpha, coef))
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    s3 = math.sqrt(3.0)
    ang = {0: [np.ones_like(x)], 1: [x, y, z],
           2: [(2 * z * z - x * x - y * y) / 2.0, s3 * x * z, s3 * y * z, s3 / 2.0 * (x * x - y * y), s3 * x * y]}
    return [rad * a for a in ang[l]]


@pytest.mark.parametrize("la,lb", [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)])
def test_shell_integrals_against_quadrature(la, lb):
    """Overlap, dipole and second-moment integrals of contracted shells match numerical quadrature."""
    ra, rb = np.array([0.1, -0.2, 0.0]), np.array([0.6, 0.9, 1.3])
    aa, ca = [0.5, 0.2], [0.4, 0.7]
    ab, cb = [0.4, 0.15], [0.5, 0.6]
    t = torch.float64
    S, D, Q = shell_pair_integrals(torch.tensor([ra], dtype=t), torch.tensor([rb], dtype=t),
                                   torch.tensor([aa], dtype=t), torch.tensor([ca], dtype=t), la,
                                   torch.tensor([ab], dtype=t), torch.tensor([cb], dtype=t), lb)
    g, w = _grid(0.5 * (ra + rb))
    fa, fb = _shell_values(g, ra, aa, ca, la), _shell_values(g, rb, ab, cb, lb)
    d = g - ra
    ops = [np.ones(len(g))] + [d[:, k] for k in range(3)]
    ops += [d[:, i] * d[:, j] for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))]
    for i in range(N_SPHERICAL[la]):
        for j in range(N_SPHERICAL[lb]):
            ref = [float((fa[i] * fb[j] * op * w).sum()) for op in ops]
            got = ([float(S[0, i, j])] + [float(D[0, k, i, j]) for k in range(3)]
                   + [float(Q[0, k, i, j]) for k in range(6)])
            np.testing.assert_allclose(got, ref, atol=1e-10)


@pytest.mark.parametrize("name", ["gfn0", "gfn1", "gfn2"])
def test_forces_match_finite_differences(name, gfn1, gfn2):
    """Forces are the negative gradient of the energy (central differences)."""
    model = {"gfn0": GFN0(), "gfn1": gfn1, "gfn2": gfn2}[name]
    for z, pos in (FORMAMIDE, HALOGEN_BOND, TICL):
        pos = np.asarray(pos, dtype=np.float64)
        out = ForceStressOutput(model)(collate([_graph(z, pos)]))
        forces = out["forces"].detach().numpy()
        h = 1e-4
        for atom, axis in ((0, 0), (1, 2), (len(z) - 1, 1)):
            e = []
            for sign in (1, -1):
                p = pos.copy()
                p[atom, axis] += sign * h
                e.append(float(model(collate([_graph(z, p)]))["energy"][0]))
            assert -(e[0] - e[1]) / (2 * h) == pytest.approx(forces[atom, axis], abs=2e-6)


def test_rotation_translation_and_batching(gfn2):
    """Energies are invariant, forces rotate with the molecule, a batch equals separate runs."""
    z, pos = FORMAMIDE
    pos = np.asarray(pos, dtype=np.float64)
    rot = _rotation(3)
    a = ForceStressOutput(gfn2)(collate([_graph(z, pos)]))
    b = ForceStressOutput(gfn2)(collate([_graph(z, pos @ rot.T + [1.0, -2.0, 0.5])]))
    assert float(a["energy"][0]) == pytest.approx(float(b["energy"][0]), abs=1e-9)
    np.testing.assert_allclose(a["forces"].detach().numpy() @ rot.T, b["forces"].detach().numpy(), atol=1e-8)
    batch = gfn2(collate([_graph(*WATER), _graph(*FORMAMIDE), _graph(*WATER, charge=1.0, mult=2.0)]))
    sep = [gfn2(collate([g]))["energy"][0] for g in (_graph(*WATER), _graph(*FORMAMIDE),
                                                       _graph(*WATER, charge=1.0, mult=2.0))]
    np.testing.assert_allclose(batch["energy"].numpy(), torch.stack(sep).numpy(), atol=1e-9)


def test_charges_and_electronic_entropy(gfn1, gfn2):
    """Charges sum to the total charge; the free energy of stretched H2 falls with the temperature."""
    for model in (gfn1, gfn2):
        out = model(collate([_graph(*WATER, charge=-1.0, mult=2.0)]))
        assert float(out["charges"].sum()) == pytest.approx(-1.0, abs=1e-10)
        assert bool(out["scf_converged"])
    h2 = ([1, 1], [[0, 0, 0], [0, 0, 3.5]])
    cold = GFN2(accuracy=TIGHT, electronic_temperature=1.0)
    hot = GFN2(accuracy=TIGHT, electronic_temperature=3000.0)
    assert float(hot(collate([_graph(*h2)]))["energy"][0]) < float(cold(collate([_graph(*h2)]))["energy"][0])


def test_halogen_bond_term(gfn1):
    """The GFN1 halogen bond acts on Br with an N acceptor, with the angular damping ((1 - cos)/2)^6."""
    z, pos = HALOGEN_BOND
    pos = np.asarray(pos, dtype=np.float64)
    linear = pos.copy()
    linear[4:, 0] = 0.0
    bent = linear.copy()
    bent[5] = linear[4] + 1.76 * np.array([math.sin(math.radians(30)), 0.0, math.cos(math.radians(30))])
    t = torch.float64
    zt, batch = torch.tensor(z), torch.zeros(len(z), dtype=torch.long)
    e_lin = float(gfn1.halogen_bond_energy(zt, torch.tensor(linear, dtype=t) / BOHR, batch, 1))
    e_bent = float(gfn1.halogen_bond_energy(zt, torch.tensor(bent, dtype=t) / BOHR, batch, 1))
    assert e_lin != 0.0
    assert e_bent / e_lin == pytest.approx(((1 + math.cos(math.radians(30))) / 2) ** 6, rel=1e-12)
    # carbon is no acceptor
    zc = torch.tensor([6, 1, 1, 1, 35, 9])
    assert float(gfn1.halogen_bond_energy(zc, torch.tensor(linear, dtype=t) / BOHR, batch, 1)) == 0.0
    # the radii of the distance polynomial set the minimum: x = 1.3 (R_N + R_Br) / R
    r = 1.3 * (GFN2_RADII[7] + GFN2_RADII[35])
    lin_far = linear.copy()
    lin_far[4:, 2] += r - linear[4, 2]
    e0 = float(gfn1.halogen_bond_energy(zt, torch.tensor(lin_far, dtype=t) / BOHR, batch, 1))
    assert e0 == pytest.approx(0.1 * gfn1.par.value(35, "CXB") * (1 - 0.44) / 2, rel=1e-12)


def test_unsupported_element_and_options():
    """Missing parameters raise; parameters come as a file, its text or a published name."""
    path = xtb_published.parameter_file("gfn2")
    by_text = GFN2(accuracy=TIGHT, parameters=path.read_text())
    with pytest.raises(ValueError, match="no parameters"):
        by_text(collate([_graph([87, 1], [[0, 0, 0], [0, 0, 2.0]])]))
    by_path, by_name = GFN2(accuracy=TIGHT, parameters=str(path)), GFN2(accuracy=TIGHT, parameters="gfn2")
    energies = [float(m(collate([_graph(*WATER)]))["energy"][0]) for m in (by_text, by_path, by_name)]
    assert energies[0] == energies[1] == energies[2]


def test_zero_temperature_and_empty_spin_channel():
    """Integer occupations at 0 K; a one-electron cation has an empty beta channel."""
    cold = GFN2(accuracy=TIGHT, electronic_temperature=0.0)
    e_cold = float(cold(collate([_graph(*WATER)]))["energy"][0])
    e_warm = float(GFN2(accuracy=TIGHT)(collate([_graph(*WATER)]))["energy"][0])
    assert e_cold == pytest.approx(e_warm, abs=1e-8)
    out = GFN1(accuracy=TIGHT)(collate([_graph([1, 1], [[0, 0, 0], [0, 0, 1.06]], charge=1.0, mult=2.0)]))
    assert float(out["charges"].sum()) == pytest.approx(1.0, abs=1e-12) and bool(out["scf_converged"])


def test_overlaps_without_multipoles():
    """The overlap-only evaluation equals the overlap block of the full one."""
    t = torch.float64
    args = (torch.tensor([[0.0, 0.0, 0.0]], dtype=t), torch.tensor([[0.3, 0.2, 1.1]], dtype=t),
            torch.tensor([[0.8, 0.3]], dtype=t), torch.tensor([[0.5, 0.6]], dtype=t), 2,
            torch.tensor([[0.7, 0.2]], dtype=t), torch.tensor([[0.4, 0.7]], dtype=t), 1)
    full = shell_pair_integrals(*args)
    only, d, q = shell_pair_integrals(*args, multipoles=False)
    torch.testing.assert_close(full[0], only, rtol=1e-14, atol=1e-16)
    assert d.numel() == 0 and q.numel() == 0


@pytest.mark.skipif("XNN_XTB" not in os.environ, reason="the xtb program is not available (XNN_XTB)")
@pytest.mark.parametrize("method,dispersion", [("gfn0", True), ("gfn1", True), ("gfn2", True)])
def test_fidelity_vs_xtb(method, dispersion):
    """Energies, gradients and charges equal xtb's to its convergence."""
    import sqmnn_xtb_fidelity as fid
    res = fid.compare(Path(os.environ["XNN_XTB"]), method, dispersion)
    w = res["worst"]
    assert w["energy"] < 1e-9 and w["gradient"] < 1e-8 and w["charges"] < 1e-7, res["rows"]


@pytest.mark.parametrize("name", ["gfn1", "gfn2"])
def test_closed_form_potentials(name, gfn1, gfn2):
    """The closed-form potentials of the cycle equal the derivatives of the charge-dependent energy."""
    from xnn.sqmnn.models.xtb import GFN, _Basis
    model = {"gfn1": gfn1, "gfn2": gfn2}[name]
    z, pos = HALOGEN_BOND
    g = collate([_graph(z, pos)])
    zt, p = g.atomic_numbers.long(), g.pos.to(torch.float64) / BOHR
    basis = _Basis(model, zt.tolist(), g.batch.tolist(), 1, torch.float64, p.device)
    cn = model.coordination_numbers(zt, p, g.batch)
    ctx = model.scc_context(zt, p, basis, cn, g.batch, 1)
    rng = np.random.default_rng(1)
    q = torch.tensor(rng.normal(scale=0.2, size=len(basis.shell_l)))
    mu = torch.tensor(rng.normal(scale=0.1, size=(len(z), 3)))
    th = torch.tensor(rng.normal(scale=0.1, size=(len(z), 6)))
    closed = model.potentials(q, mu, th, ctx)
    auto = GFN.potentials(model, q, mu, th, ctx)
    torch.testing.assert_close(closed[0], auto[0], rtol=1e-12, atol=1e-12)
    if name == "gfn2":
        torch.testing.assert_close(closed[1], auto[1], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(closed[2], auto[2], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("la,lb", [(0, 0), (0, 2), (1, 1), (2, 1), (2, 2)])
def test_analytic_integral_gradients(la, lb):
    """The analytic centre gradients of the integrals equal automatic differentiation."""
    from xnn.sqmnn.models.gaussian import _cartesian_values, _to_spherical, primitive_norm
    t = torch.float64
    rng = np.random.default_rng(la * 3 + lb)
    ra = torch.tensor(rng.normal(size=(3, 3)), dtype=t, requires_grad=True)
    rb = torch.tensor(rng.normal(size=(3, 3)) + 1.0, dtype=t, requires_grad=True)
    aa, ca = torch.tensor([[0.9, 0.3]] * 3, dtype=t), torch.tensor([[0.4, 0.7]] * 3, dtype=t)
    ab, cb = torch.tensor([[0.8, 0.25]] * 3, dtype=t), torch.tensor([[0.5, 0.6]] * 3, dtype=t)
    weights = torch.tensor(rng.normal(size=(3, 10, 2 * la + 1, 2 * lb + 1)), dtype=t)
    s, d, q = shell_pair_integrals(ra, rb, aa, ca, la, ab, cb, lb)
    got = torch.autograd.grad((torch.cat([s[:, None], d, q], 1) * weights).sum(), [ra, rb])
    wa, wb = ca * primitive_norm(aa, la), cb * primitive_norm(ab, lb)
    ref_vals = _to_spherical(_cartesian_values(ra, rb, aa, wa, ab, wb, la, lb, 10, float("inf")), la, lb)
    ref = torch.autograd.grad((ref_vals * weights).sum(), [ra, rb])
    for x, y in zip(got, ref):
        torch.testing.assert_close(x, y, rtol=1e-11, atol=1e-13)


def test_fast_path_three_body_matches_reference():
    """The compiled three-body dispersion gives the reference's energy and forces."""
    from xnn.common.models.dispersion_fast import atm_cpu
    if not atm_cpu.available():
        pytest.skip("torch.compile produces no CPU code here")
    ref, fast = GFN2(accuracy=TIGHT, use_fast=False).eval(), GFN2(accuracy=TIGHT, use_fast=True).eval()
    assert not ref._compiled_three_body(torch.zeros(1)) and fast._compiled_three_body(torch.zeros(1))
    # twice differentiated (a training step): the same second derivatives in the positions
    second = []
    for model in (ref.train(), fast.train()):
        g = collate([_graph(*FORMAMIDE)])
        g.pos.requires_grad_(True)
        (f,) = torch.autograd.grad(model(g)["energy"].sum(), g.pos, create_graph=True)
        second.append(torch.autograd.grad((f * torch.linspace(-1.0, 1.0, f.numel()).view_as(f)).sum(), g.pos)[0])
    torch.testing.assert_close(second[0], second[1], rtol=0.0, atol=1e-8)
    # the same with structures of different weights
    second = []
    for model in (ref, fast):
        g = collate([_graph(*WATER), _graph(*FORMAMIDE)])
        g.pos.requires_grad_(True)
        e = model(g)["energy"]
        (f,) = torch.autograd.grad(e[0] + 2.0 * e[1], g.pos, create_graph=True)
        second.append(torch.autograd.grad(f.pow(2).sum(), g.pos)[0])
    torch.testing.assert_close(second[0], second[1], rtol=0.0, atol=1e-8)
    ref.eval(), fast.eval()
    for z, pos in (FORMAMIDE, HALOGEN_BOND):
        a = ForceStressOutput(ref).eval()(collate([_graph(z, pos)]))
        b = ForceStressOutput(fast).eval()(collate([_graph(z, pos)]))
        assert float(a["energy"][0]) == pytest.approx(float(b["energy"][0]), abs=1e-10)
        np.testing.assert_allclose(a["forces"].detach().numpy(), b["forces"].detach().numpy(), atol=1e-10)
        with torch.no_grad():
            plain = float(fast(collate([_graph(z, pos)]))["energy"][0])
        assert plain == pytest.approx(float(a["energy"][0]), abs=1e-10)
    # structures with different weights: the gradient of the weighted energies
    grads = []
    for model in (ref, fast):
        g = collate([_graph(*FORMAMIDE), _graph(*HALOGEN_BOND)])
        g.pos.requires_grad_(True)
        e = model(g)["energy"]
        grads.append(torch.autograd.grad(e[0] + 2.0 * e[1], g.pos)[0])
    torch.testing.assert_close(grads[0], grads[1], rtol=0.0, atol=1e-10)


SULFUR = ([16, 16, 8, 1, 1], [[0, 0, 0], [0.1, 0.2, 2.05], [1.5, 0.3, -0.4], [-0.6, 1.1, -0.5], [0.3, -1.0, 2.9]])


def _kernel_inputs(model, z, pos):
    """Positions (bohr), basis and classified shell pairs of one molecule."""
    from xnn.sqmnn.models.xtb import _Basis
    p = torch.tensor(pos, dtype=torch.float64) / BOHR
    basis = _Basis(model, list(z), [0] * len(z), 1, torch.float64, p.device)
    ti, tj, kind = model._shell_pairs(basis, p)
    return p, basis, ti, tj, kind


@pytest.mark.parametrize("cls", [GFN0, GFN1, GFN2])
def test_integral_kernels_match_reference(cls):
    """The integral kernels give the reference's matrices, their gradient, and its energies and forces."""
    from xnn.sqmnn.models import xtb_kernels
    if not xtb_kernels.available(torch.device("cpu")):
        pytest.skip("no C++ compiler for the CPU kernels here")
    ref, fast = cls(accuracy=TIGHT, use_fast=False).eval(), cls(accuracy=TIGHT, use_fast=True).eval()
    assert fast._fused_integrals(torch.zeros(1)) and not ref._fused_integrals(torch.zeros(1))
    p, basis, ti, tj, kind = _kernel_inputs(ref, *SULFUR)
    n_ops = 10 if ref.multipoles else 1
    classes = ref._pair_classes(ti, tj, kind)
    assert {(c[0], c[1]) for c in classes} >= {(0, 0), (1, 2), (2, 2)}
    q = p.clone().requires_grad_(True)
    want = ref._integral_stack(basis, q, 1, ti, tj, kind)
    got = xtb_kernels.integral_matrices(p, basis.kernel_tables, classes, 1, n_ops, basis.nb, ref.integral_neglect)
    torch.testing.assert_close(got, want.detach(), rtol=0.0, atol=1e-13)
    weight = torch.randn(want.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    grad = xtb_kernels.integral_gradient(p, basis.kernel_tables, classes, n_ops, basis.nb, ref.integral_neglect, weight)
    torch.testing.assert_close(grad, torch.autograd.grad((want * weight).sum(), q)[0], rtol=0.0, atol=1e-11)
    # twice differentiated: in the weight (the tangent kernels) and in the positions (the reference's functions)
    direction = torch.randn(p.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    second = []
    for model in (ref, fast):
        x, w = p.clone().requires_grad_(True), weight.clone().requires_grad_(True)
        S, D, Q = model._integrals(basis, x, 1)
        M = torch.cat([S[:, None], D, Q], 1) if model.multipoles else S[:, None]
        (g,) = torch.autograd.grad((M * w).sum(), x, create_graph=True)
        second.append(torch.autograd.grad((g * direction).sum(), (w, x)))
    torch.testing.assert_close(second[0][0], second[1][0], rtol=0.0, atol=1e-11)
    torch.testing.assert_close(second[0][1], second[1][1], rtol=0.0, atol=1e-10)
    a = ForceStressOutput(ref).eval()(collate([_graph(*SULFUR), _graph(*WATER)]))
    b = ForceStressOutput(fast).eval()(collate([_graph(*SULFUR), _graph(*WATER)]))
    torch.testing.assert_close(a["energy"], b["energy"], rtol=0.0, atol=1e-9)
    torch.testing.assert_close(a["forces"], b["forces"], rtol=0.0, atol=1e-8)
    with torch.no_grad():
        torch.testing.assert_close(fast(collate([_graph(*WATER)]))["energy"], a["energy"][1:], rtol=0.0, atol=1e-9)


@pytest.mark.parametrize("n_ops", [1, 10])
def test_integral_kernels_interpreted(n_ops):
    """The generated kernels, run by Triton's interpreter, give the reference's matrices and gradient."""
    if os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("runs where Triton's interpreter is on (no CUDA device)")
    from xnn.sqmnn.models import xtb_kernels
    model = (GFN2 if n_ops == 10 else GFN1)(use_fast=False)
    z, pos = SULFUR[0][:3], SULFUR[1][:3]
    p, basis, ti, tj, kind = _kernel_inputs(model, z, pos)
    q = p.clone().requires_grad_(True)
    want = model._integral_stack(basis, q, 1, ti, tj, kind)
    weight = torch.randn(want.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    w = weight.clone().requires_grad_(True)
    (wgrad,) = torch.autograd.grad((want * w).sum(), q, create_graph=True)
    direction = torch.randn(p.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    (wtangent,) = torch.autograd.grad((wgrad * direction).sum(), w)
    wgrad = wgrad.detach()
    got, grad, tangent = torch.zeros_like(want), torch.zeros_like(p), torch.zeros_like(want)
    c = xtb_kernels._constants(model.integral_neglect, p)
    kmax, block = basis.alpha.shape[1], 4
    for la, lb, ka, kb, si, sj in model._pair_classes(ti, tj, kind):
        n = si.shape[0]
        head = (p, *basis.kernel_tables, si, sj, c)
        grid = ((n + block - 1) // block,)
        xtb_kernels._kernel("values", la, lb, n_ops)[grid](*head, got, n, ka, kb, kmax, basis.nb, BLOCK=block)
        g = torch.zeros(n, 3, dtype=torch.float64)
        xtb_kernels._kernel("gradient", la, lb, n_ops)[grid](*head, weight, g, n, ka, kb, kmax, basis.nb, BLOCK=block)
        xtb_kernels._kernel("tangent", la, lb, n_ops)[grid](*head, direction, tangent, n, ka, kb, kmax, basis.nb,
                                                            BLOCK=block)
        grad.index_add_(0, basis.shell_atom[si], g)
        grad.index_add_(0, basis.shell_atom[sj], -g)
    torch.testing.assert_close(got, want.detach(), rtol=0.0, atol=1e-13)
    torch.testing.assert_close(grad, wgrad, rtol=0.0, atol=1e-11)
    torch.testing.assert_close(tangent, wtangent, rtol=0.0, atol=1e-11)


def test_cycle_safeguards(monkeypatch):
    """A long cycle is run a second time and each structure keeps the lower free energy; a cycle that
    fails is tried again from a warm start."""
    g = collate([_graph(*WATER), _graph(*FORMAMIDE)])
    plain = GFN1(accuracy=TIGHT)
    with torch.no_grad():
        want = plain(g)["energy"]
    model = GFN1(accuracy=TIGHT)
    model.long_cycle = 0
    with torch.no_grad():
        got = model(g)["energy"]
    torch.testing.assert_close(got, want, rtol=0.0, atol=1e-8)
    assert model.last_scf["iterations"] > plain.last_scf["iterations"]
    # a first result with a raised free energy of one structure: that structure takes the second run
    seen = {}
    real = model._safeguarded

    def raised(res, *args):
        bad = dict(res)
        bad["ts"] = res["ts"] - torch.tensor([1.0, 0.0], dtype=res["ts"].dtype)
        seen["first"], seen["kept"] = bad["ts"], real(bad, *args)["ts"]
        return real(bad, *args)

    monkeypatch.setattr(model, "_safeguarded", raised)
    with torch.no_grad():
        got = model(g)["energy"]
    torch.testing.assert_close(got, want, rtol=0.0, atol=1e-8)
    assert float(seen["kept"][0] - seen["first"][0]) == pytest.approx(1.0, abs=1e-8)
    assert float(seen["kept"][1]) == float(seen["first"][1])
    short = GFN1(accuracy=TIGHT, max_iter=2)
    with torch.no_grad():
        assert not bool(short(g)["scf_converged"])
    # with atomic moments among the mixed quantities
    multi = GFN2(accuracy=TIGHT)
    multi.long_cycle = 0
    with torch.no_grad():
        torch.testing.assert_close(multi(g)["energy"], GFN2(accuracy=TIGHT)(g)["energy"], rtol=0.0, atol=1e-8)


def test_singular_overlap_route(monkeypatch):
    """The eigenvector orthogonalization (a failed Cholesky factorization) gives the same energy."""
    g = collate([_graph(*FORMAMIDE)])
    ref = float(GFN2(accuracy=TIGHT)(g)["energy"][0])
    monkeypatch.setattr(torch.linalg, "cholesky_ex", lambda S: (S, torch.ones(S.shape[0], dtype=torch.int32)))
    assert float(GFN2(accuracy=TIGHT)(g)["energy"][0]) == pytest.approx(ref, abs=1e-9)


def test_host_route_of_small_gpu_batches():
    """The fast path sends small batches on a GPU to a host copy of the model, with the same results."""
    from types import SimpleNamespace
    model = GFN2(accuracy=TIGHT)
    g = collate([_graph(*FORMAMIDE), _graph(*WATER)])
    on_gpu = SimpleNamespace(pos=SimpleNamespace(device=torch.device("cuda")), num_graphs=2, batch=g.batch,
                             atomic_numbers=g.atomic_numbers)
    assert not model._on_host(g) and model._on_host(on_gpu)
    model.host_basis = 16                       # formamide alone has 18 orbitals
    assert not model._on_host(on_gpu)
    model.host_basis = GFN2.host_basis
    assert not GFN2(accuracy=TIGHT, use_fast=False)._on_host(on_gpu)
    twin = model._host_model()
    assert twin is model._host_model() and twin is not model and "_twin" not in dict(model.named_modules())
    g.pos.requires_grad_(True)
    ref = model(g)["energy"]
    ref_grad = torch.autograd.grad(ref.sum(), g.pos)[0]
    model._on_host = lambda data: True          # as on a GPU: the evaluation goes through the host copy
    got = model(g)["energy"]
    torch.testing.assert_close(got, ref, rtol=0.0, atol=1e-10)
    torch.testing.assert_close(torch.autograd.grad(got.sum(), g.pos)[0], ref_grad, rtol=0.0, atol=1e-10)
    assert model.last_scf is twin.last_scf
    model.accuracy = 1.0
    assert model._host_model() is not twin and model._host_model().accuracy == 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("cls", [GFN0, GFN1, GFN2])
def test_gpu_matches_cpu(cls):
    """On a GPU the host route and the device route give the CPU's energies and forces."""
    graphs = [_graph(*FORMAMIDE), _graph(*HALOGEN_BOND)]
    cpu = ForceStressOutput(cls(accuracy=TIGHT)).eval()(collate(graphs))
    for limit, host in ((cls.host_basis, True), (0, False)):
        model = cls(accuracy=TIGHT).to("cuda")
        model.host_basis = limit
        data = collate(graphs).to("cuda")
        assert model._on_host(data) is host
        gpu = ForceStressOutput(model).eval()(data)
        assert gpu["energy"].device.type == "cuda"
        torch.testing.assert_close(gpu["energy"].cpu(), cpu["energy"], rtol=0.0, atol=1e-8)
        torch.testing.assert_close(gpu["forces"].cpu(), cpu["forces"], rtol=0.0, atol=1e-7)


@pytest.mark.parametrize("cls", [GFN0, GFN1, GFN2])
def test_parameter_gradients_of_the_energy(cls):
    """The energy gradient in every trained element parameter equals finite differences."""
    g = collate([_graph(*FORMAMIDE), _graph(*WATER)])
    model = cls(accuracy=TIGHT, trainable=True)
    assert model.trainable == cls.trainable_names and len(list(model.parameters())) == len(cls.trainable_names)
    weights = torch.tensor([1.0, -0.7], dtype=torch.float64)
    grads = torch.autograd.grad((model(g)["energy"] * weights).sum(), list(model.parameters()), allow_unused=True)
    rng = np.random.default_rng(1)
    for (name, par), grad in zip(model.named_parameters(), grads):
        entries = [(z, k) for z in (1, 6, 7, 8) for k in range(par.shape[1] if par.dim() == 2 else 1)]
        for z, k in [entries[i] for i in rng.choice(len(entries), 3, replace=False)]:
            at = (z, k) if par.dim() == 2 else (z,)
            h = 1e-5 * max(1.0, abs(float(par[at])))
            e = []
            for sign in (1.0, -1.0):
                with torch.no_grad():
                    par[at] += sign * h
                    e.append(float((model(g)["energy"] * weights).sum()))
                    par[at] -= sign * h
            fd = (e[0] - e[1]) / (2 * h)
            got = 0.0 if grad is None else float(grad[at])
            assert got == pytest.approx(fd, rel=2e-5, abs=2e-7), (name, at)
    with pytest.raises(ValueError, match="not trainable"):
        cls(trainable=["ZETA"])
    assert cls(trainable=["REPA", "LEV"]).trainable == ("LEV", "REPA")


@pytest.mark.parametrize("cls", [GFN0, GFN1, GFN2])
def test_parameter_gradients_of_a_force_loss(cls):
    """In training mode the gradient of a loss on energies and forces follows the density response:
    it equals finite differences of the loss in the parameters (degenerate levels and a radical included)."""
    tetra = np.array([[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]]) * 1.09 / math.sqrt(3.0)
    graphs = [_graph(*WATER), _graph([6, 1, 1, 1, 1], np.vstack([[0.0, 0.0, 0.0], tetra])),
              _graph([6, 1, 1, 1], [[0, 0, 0.02], [1.08, 0, 0], [-0.54, 0.94, 0.01], [-0.55, -0.93, 0]], mult=2.0)]
    g = collate(graphs)
    rng = np.random.default_rng(3)
    wf = torch.tensor(rng.normal(size=tuple(g.pos.shape)), dtype=torch.float64)
    we = torch.tensor([0.3, -0.2, 0.5], dtype=torch.float64)
    model = cls(accuracy=TIGHT, trainable=True)

    def loss(training):
        out = ForceStressOutput(model).train(training)(collate(graphs))
        return (out["forces"] * wf).sum() + (out["energy"] * we).sum()

    reference = float(loss(False))
    value = loss(True)
    assert float(value) == pytest.approx(reference, abs=1e-9)
    grads = torch.autograd.grad(value, list(model.parameters()), allow_unused=True)
    for (name, par), grad in zip(model.named_parameters(), grads):
        entries = [(z, k) for z in (1, 6, 8) for k in range(par.shape[1] if par.dim() == 2 else 1)]
        for z, k in [entries[i] for i in rng.choice(len(entries), 2, replace=False)]:
            at = (z, k) if par.dim() == 2 else (z,)
            h = 1e-4 * max(1.0, abs(float(par[at])))
            values = []
            for sign in (1.0, -1.0):
                with torch.no_grad():
                    par[at] += sign * h
                values.append(float(loss(False)))
                with torch.no_grad():
                    par[at] -= sign * h
            fd = (values[0] - values[1]) / (2 * h)
            got = 0.0 if grad is None else float(grad[at])
            assert got == pytest.approx(fd, rel=1e-4, abs=1e-6), (name, at)


@pytest.mark.parametrize("cls", [GFN0, GFN1, GFN2])
def test_exponents_global_and_pair_parameters_train(cls):
    """Slater exponents, global and pair parameters are trained on request: the gradients of the
    energy and of a force loss equal finite differences, and changed values reach the parameter
    file and the export."""
    graphs = [_graph(*WATER),
              _graph([6, 1, 1, 1], [[0, 0, 0.02], [1.08, 0, 0], [-0.54, 0.94, 0.01], [-0.55, -0.93, 0]], mult=2.0)]
    model = cls(accuracy=TIGHT, trainable="all")
    assert model.trainable == cls.trainable_names + cls.extra_trainable
    model = cls(accuracy=TIGHT, trainable=cls.extra_trainable)
    assert model.trainable == cls.extra_trainable and cls(trainable="GLOBAL").trainable == ("GLOBAL",)
    wanted = [n for n in ("ks", "kp", "kdiff", "enscale", "ens", "kexp", "gam3s", "aesshift", "srbpre", "cns")
              if n in model.global_names][:5]
    entries = {"EXP": [(1, 0), (8, 1), (6, 0)] + ([(1, 1)] if len(model.par.shells[1]) > 1 else []),
               "GLOBAL": [(model.global_names.index(n),) for n in wanted], "KPAIR": [(8, 1), (6, 1)]}
    weights = torch.tensor([1.0, -0.7], dtype=torch.float64)

    def energy():
        return (model.eval()(collate(graphs))["energy"] * weights).sum()

    def loss():
        out = ForceStressOutput(model).train()(collate(graphs))
        return out["forces"].pow(2).mean() + 1e-3 * (out["energy"] * weights).sum()

    for function, step, rel, tol in ((energy, 1e-5, 2e-5, 2e-7), (loss, 1e-4, 2e-4, 1e-6)):
        grads = torch.autograd.grad(function(), list(model.parameters()), allow_unused=True)
        for (name, par), grad in zip(model.named_parameters(), grads):
            for at in entries.get(name[len("table_"):], [(1,), (6,), (8,)]):
                h = step * max(1.0, abs(float(par[at])))
                values = []
                for sign in (1.0, -1.0):
                    with torch.no_grad():
                        par[at] += sign * h
                    values.append(float(function().detach()))
                    with torch.no_grad():
                        par[at] -= sign * h
                got = 0.0 if grad is None else float(grad[at])
                assert got == pytest.approx((values[0] - values[1]) / (2 * h), rel=rel, abs=tol), (name, at)
    with torch.no_grad():
        model.table_EXP[8, 1] *= 1.02
        model.table_GLOBAL[model.global_names.index("ks")] += 0.03
        if "KPAIR" in model.trainable:
            model.table_KPAIR[8, 1] += 0.02
            model.table_KPAIR[6, 1] -= 0.01
    g = collate([_graph(*WATER)])
    with torch.no_grad():
        changed = float(model.eval()(g)["energy"][0])
        assert abs(changed - float(cls(accuracy=TIGHT)(g)["energy"][0])) > 1e-3
        loaded = float(cls(accuracy=TIGHT, parameters=model.parameter_text())(g)["energy"][0])
    assert loaded == pytest.approx(changed, abs=1e-6)
    zero, no = torch.zeros(3, 3, dtype=torch.float64), torch.zeros(3, dtype=torch.bool)
    p = torch.tensor(np.asarray(WATER[1], dtype=np.float64))
    _, node, _ = model.export_core().node_features_energy_charges(
        torch.tensor(WATER[0]), torch.zeros(2, 0, dtype=torch.long), p[:0], p, zero, no, 0.0, 0.0)
    assert float(node.sum()) == pytest.approx(changed, abs=1e-8)


def test_trained_parameters_round_trip():
    """Changed element parameters are written in the xtb format, load again and, with the xtb
    program at hand, give its energy and gradient."""
    model = GFN2(accuracy=TIGHT, trainable=["LEV", "GAM", "REPA", "POLY", "KCN"])
    with torch.no_grad():
        model.table_LEV[8, 1] += 0.2
        model.table_GAM[1] *= 1.03
        model.table_REPA[6] -= 0.05
        model.table_POLY[7, 1] += 0.1
        model.table_KCN[8, 0] += 0.01
    text = model.parameter_text()
    again = GFN2(accuracy=TIGHT, parameters=text)
    assert float(again.table("LEV")[8, 1]) == pytest.approx(float(model.table("LEV")[8, 1]), abs=1e-9)
    g = collate([_graph(*FORMAMIDE)])
    changed, loaded = float(model(g)["energy"][0]), float(again(g)["energy"][0])
    assert loaded == pytest.approx(changed, abs=1e-7)
    assert abs(changed - float(GFN2(accuracy=TIGHT)(g)["energy"][0])) > 1e-3
    if os.environ.get("XNN_XTB"):
        import importlib.util
        spec = importlib.util.spec_from_file_location("xtb_fidelity", Path(__file__).with_name("sqmnn_xtb_fidelity.py"))
        fidelity = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fidelity)
        z, pos = FORMAMIDE
        ref = fidelity.run_xtb(Path(os.environ["XNN_XTB"]), "gfn2", z, pos, 0, 0, text)
        got = fidelity.run_xnn(GFN2(accuracy=1e-5, parameters=text), z, pos, 0, 0)
        assert abs(got["energy"] - ref["energy"]) < 1e-8
        assert np.abs(got["gradient"] - ref["gradient"]).max() < 1e-7


@pytest.mark.parametrize("cls", [GFN0, GFN1, GFN2])
def test_torchscript_export_matches_eager(cls, tmp_path):
    """The exported TorchScript potential and its core called as plain Python give the eager model's
    energy, forces and charges, for a neutral molecule with a halogen bond and for an ion."""
    from xnn.common.deploy import export_torchscript_potential
    eager = ForceStressOutput(cls(accuracy=TIGHT)).eval()
    core = cls(accuracy=TIGHT).export_core()
    zero, no = torch.zeros(3, 3, dtype=torch.float64), torch.zeros(3, dtype=torch.bool)
    empty = torch.zeros(2, 0, dtype=torch.long)
    for (z, pos), charge in ((HALOGEN_BOND, 0.0), (FORMAMIDE, 0.0), (([8, 1], [[0, 0, 0], [0.02, 0, 0.97]]), -1.0)):
        ref = eager(collate([_graph(z, pos, charge=charge)]))
        p = torch.tensor(np.asarray(pos, dtype=np.float64))
        _, node, q = core.node_features_energy_charges(torch.tensor(z), empty, p[:0], p, zero, no, charge, 0.0)
        assert float(node.sum()) == pytest.approx(float(ref["energy"][0]), abs=1e-8)
        torch.testing.assert_close(q, ref["charges"], rtol=0.0, atol=1e-8)
        path = export_torchscript_potential(cls(accuracy=TIGHT).eval(), cutoff=10.0, path=str(tmp_path / "m.pt"),
                                            total_charge=charge)
        scripted = torch.jit.load(path)
        with torch.jit.optimized_execution(False):
            out = scripted(p, torch.tensor(z))
            assert float(out["energy"]) == pytest.approx(float(ref["energy"][0]), abs=1e-8)
            torch.testing.assert_close(out["forces"], ref["forces"], rtol=0.0, atol=1e-7)
            torch.testing.assert_close(out["charges"], ref["charges"], rtol=0.0, atol=1e-8)
            with pytest.raises(Exception, match="periodic"):
                scripted(p, torch.tensor(z), torch.eye(3, dtype=torch.float64) * 20.0, torch.ones(3, dtype=torch.bool))
    from xnn.common.deploy.torchscript import build_neighbor_list_ts
    with torch.jit.optimized_execution(False):
        ei, cs = build_neighbor_list_ts(p, 10.0, zero, no)
        pair = scripted.forward_lammps(p, ei, cs, torch.tensor(z), zero)
    assert float(pair["energy"]) == pytest.approx(float(out["energy"]), abs=1e-9)
    torch.testing.assert_close(pair["forces"], out["forces"], rtol=0.0, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_torchscript_export_on_gpu(tmp_path):
    """The same export from a model on a GPU."""
    from xnn.common.deploy import export_torchscript_potential
    z, pos = FORMAMIDE
    ref = ForceStressOutput(GFN2(accuracy=TIGHT)).eval()(collate([_graph(z, pos)]))
    model = GFN2(accuracy=TIGHT).to("cuda").eval()
    path = export_torchscript_potential(model, cutoff=10.0, path=str(tmp_path / "g.pt"))
    with torch.jit.optimized_execution(False):
        out = torch.jit.load(path)(torch.tensor(np.asarray(pos, dtype=np.float64), device="cuda"),
                                   torch.tensor(z, device="cuda"))
    assert float(out["energy"]) == pytest.approx(float(ref["energy"][0]), abs=1e-8)
    torch.testing.assert_close(out["forces"].cpu(), ref["forces"], rtol=0.0, atol=1e-7)


def test_gfn0_terms_and_options():
    """GFN0-xTB: one diagonalization, the charges of its charge model, its energy terms, batches, an ion, changed
    parameters through a parameter file."""
    model = GFN0()
    graphs = [_graph(*FORMAMIDE), _graph([8, 1], [[0, 0, 0], [0.02, 0, 0.97]], charge=-1.0), _graph(*TICL)]
    batch = model(collate(graphs))
    assert model.last_scf["iterations"] == 1 and bool(batch["scf_converged"])
    for k, g in enumerate(graphs):
        single = model(collate([g]))
        assert float(single["energy"][0]) == pytest.approx(float(batch["energy"][k]), abs=1e-9)
        terms = sum(float(single[f"{name}_energy"][0]) for name in ("electronic", "repulsion", "electrostatic",
                                                                     "dispersion", "short_range"))
        assert terms == pytest.approx(float(single["energy"][0]), abs=1e-9)
    assert float(batch["charges"][:6].sum()) == pytest.approx(0.0, abs=1e-10)
    assert float(batch["charges"][6:8].sum()) == pytest.approx(-1.0, abs=1e-10)
    assert float(batch["short_range_energy"][0]) < -1e-3 and float(batch["short_range_energy"][2]) == 0.0
    assert n_gaussians("gfn0", 2, 0, second_s=True) == 2 and n_gaussians("gfn0", 2, 1) == 3
    assert n_gaussians("gfn0", 6, 1) == 6 and n_gaussians("gfn0", 3, 2) == 4 and n_gaussians("gfn0", 1, 0) == 3
    changed = GFN0(trainable=["KQ", "KQAT2"])
    with torch.no_grad():
        changed.table_KQ[8, 1] += 0.05
        changed.table_KQAT2[1] -= 0.02
    again = GFN0(parameters=changed.parameter_text())
    g = collate([_graph(*FORMAMIDE)])
    assert float(again(g)["energy"][0]) == pytest.approx(float(changed(g)["energy"][0]), abs=1e-7)
    assert abs(float(again(g)["energy"][0]) - float(model(g)["energy"][0])) > 1e-4


def test_accuracy_option():
    """The default accuracy ends the cycle earlier, within its energy threshold of the tight result."""
    g = collate([_graph(*FORMAMIDE)])
    loose, tight = GFN2(accuracy=1.0), GFN2(accuracy=TIGHT)
    e_loose, e_tight = float(loose(g)["energy"][0]), float(tight(g)["energy"][0])
    assert GFN1(accuracy=1.0).accuracy == loose.accuracy
    assert loose.last_scf["iterations"] < tight.last_scf["iterations"]
    assert e_loose == pytest.approx(e_tight, abs=3e-5)
    with pytest.raises(ValueError, match="accuracy"):
        GFN2(accuracy=0.0)


def test_starting_charges():
    """Both starts reach the same solution for a closed shell; a failed cycle restarts from neutral atoms."""
    eeq, neutral = GFN2(accuracy=TIGHT), GFN2(accuracy=TIGHT, guess="neutral")
    g = collate([_graph(*FORMAMIDE)])
    assert float(eeq(g)["energy"][0]) == pytest.approx(float(neutral(g)["energy"][0]), abs=1e-8)
    with pytest.raises(ValueError, match="guess"):
        GFN2(accuracy=TIGHT, guess="core")
    short = GFN2(accuracy=TIGHT, max_iter=2)
    assert not bool(short(g)["scf_converged"])
