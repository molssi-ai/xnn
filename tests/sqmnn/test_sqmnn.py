"""NDDO semiempirical methods (sqmnn): integrals, SCF, energies, derivatives, training.

Self-contained checks of the published approximations (Dewar & Thiel 1977;
Thiel & Voityuk 1992; Stewart 1989, 2007, 2013, 2023): the one-center
integrals reproduce the parameters they are built from, the two-center
integrals reach their one-center and point-charge limits, overlaps agree
with numerical quadrature, the energy is invariant and its derivatives match
finite differences, and parameter gradients through the SCF response are
exact. Fidelity to the MOPAC program runs when ``XNN_MOPAC`` names its
executable (``sqmnn_mopac_fidelity.py``).
"""
import math
import os
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch

from xnn.common.config.schema import ModelConfig
from xnn.common.data import collate, structure_to_graph
from xnn.common.models import ForceStressOutput, build_model
from xnn.sqmnn.common import elements as el
from xnn.sqmnn.common import list_parameter_sets, read_parameters, write_parameters
from xnn.sqmnn.common.constants import COULOMB, EV_TO_KCAL, HARTREE
from xnn.sqmnn.models import PM3, PM6, PM6ORG, PM7, nddo, scf, twocenter
from xnn.sqmnn.models.angular import LOCAL, gaunt, multipole_coefficient
from xnn.sqmnn.models.corrections import correction_tables, h4_hbond, hh_repulsion, yne_correction
from xnn.sqmnn.models.integrals import (
    config_interaction,
    config_tensors,
    constants,
    local_frames,
    overlap_local,
    slater_condon,
)
from xnn.sqmnn.models.twocenter import pair_integrals

HERE = Path(__file__).resolve().parent
METHODS = (PM3, PM6, PM6ORG, PM7)
WATER = (np.array([[0.0, 0.0, 0.12], [0.0, 0.76, -0.47], [0.05, -0.75, -0.49]]), [8, 1, 1])
FORMAMIDE = (np.array([[0, 0.42, 0], [1.2, 0.5, 0.02], [-0.72, -0.73, 0], [-0.55, 1.36, 0.01],
                       [-1.73, -0.69, 0.05], [-0.27, -1.63, -0.03]]), [6, 8, 7, 1, 1, 1])


@pytest.fixture(autouse=True)
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _graph(pos, z, charge=0.0, mult=None):
    s = {"pos": torch.as_tensor(np.asarray(pos, dtype=np.float64)), "atomic_numbers": np.asarray(z),
         "total_charge": float(charge)}
    if mult is not None:
        s["spin_multiplicity"] = float(mult)
    return structure_to_graph(s, 0.1)


def _rotation(seed=0):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def test_registration_and_config():
    """The four methods are registered and build from a model config."""
    for name, cls in (("pm3", PM3), ("pm6", PM6), ("pm6_org", PM6ORG), ("pm7", PM7)):
        model = build_model(ModelConfig(name=name, cutoff=0.1, extra={"trainable": ["BETAS"]}))
        assert isinstance(model, cls) and "BETAS" in model.params


def test_model_config_files():
    """The shipped model configs build the published methods."""
    import yaml

    from xnn.common.config.loaders import from_dict
    root = HERE.parents[1] / "configs" / "model"
    for name, cls in (("pm3", PM3), ("pm6", PM6), ("pm6_org", PM6ORG), ("pm7", PM7)):
        cfg = from_dict({"model": yaml.safe_load((root / f"{name}.yaml").read_text())})
        model = build_model(cfg.model)
        assert type(model) is cls and not model.trainable_groups


def test_published_parameter_sets():
    """The published sets read from MOPAC's sources cover the published elements and values."""
    assert set(list_parameter_sets()) == {"pm3", "pm6", "pm6-org", "pm7"}
    counts = {"pm3": 42, "pm6": 70, "pm7": 70, "pm6-org": 18}
    for name, n in counts.items():
        assert len(read_parameters(name).atomic_numbers) == n
    pm3, pm6 = read_parameters("pm3"), read_parameters("pm6")
    assert (pm3.get(1, "USS"), pm3.get(1, "BETAS"), pm3.get(1, "ZS")) == (-13.073321, -5.626512, 0.967807)
    assert pm6.get(1, "USS") == -11.246958 and pm6.pair(8, 1, "ALPB") > 0.0


def test_published_parameters_offline(tmp_path, monkeypatch):
    """Offline, an empty cache is reported instead of downloading."""
    from xnn.sqmnn.common import published_parameters
    monkeypatch.setenv("XNN_OFFLINE", "1")
    with pytest.raises(FileNotFoundError, match="XNN_OFFLINE"):
        published_parameters("pm3", cache_dir=tmp_path)


def test_parameter_file_roundtrip(tmp_path):
    """write_parameters / read_parameters is exact, also after refitting."""
    model = PM6(trainable=["USS", "ALPB"])
    with torch.no_grad():
        model.params["USS"][8] += 0.123456789
    pset = model.parameter_set()
    path = write_parameters(pset, tmp_path / "pm6_refit.txt", header="refit")
    back = read_parameters(path)
    assert back.get(8, "USS") == pset.get(8, "USS")
    assert back.pair(1, 6, "ALPB") == pset.pair(6, 1, "ALPB")
    assert back.par(1) == pset.par(1)
    assert all(len(line) <= 60 for line in path.read_text().splitlines() if not line.startswith("*"))
    assert torch.allclose(PM6(parameters=back).element_parameter("USS"), model.element_parameter("USS"))


def test_gaunt_orthonormality_and_multipole_coefficients():
    """Real harmonics are orthonormal; product multipoles follow Thiel & Voityuk's table."""
    for a in LOCAL:
        for b in LOCAL:
            assert gaunt(a, b, "s") == pytest.approx((a == b) / math.sqrt(4 * math.pi), abs=1e-12)
    assert multipole_coefficient("z", "z", "z2") == pytest.approx(4.0 / 3.0)
    assert multipole_coefficient("x", "x", "z2") == pytest.approx(-2.0 / 3.0)
    assert multipole_coefficient("x", "x", "x2y2") == pytest.approx(1.0)
    assert multipole_coefficient("z2", "z", "z") == pytest.approx(2.0 / math.sqrt(3.0))
    assert multipole_coefficient("z2", "x", "x") == pytest.approx(-1.0 / math.sqrt(3.0))
    assert multipole_coefficient("xz", "z2", "xz") == pytest.approx(1.0 / math.sqrt(3.0))
    assert multipole_coefficient("x2y2", "z2", "x2y2") == pytest.approx(-2.0 / math.sqrt(3.0))
    assert multipole_coefficient("x2y2", "z", "z") == 0.0


def test_one_center_integrals_reproduce_parameters():
    """(ss|ss), (ss|pp), (sp|sp), (pp|pp), (pp|p'p'), (pp'|pp') equal GSS..HSP and (GPP - GP2) / 2."""
    model = PM6()
    q = model._element_quantities([8])
    w = q["one_center"][0]
    p = read_parameters("pm6")
    g = {k: p.get(8, k) for k in ("GSS", "GSP", "HSP", "GPP", "GP2")}
    assert float(w[0, 0, 0, 0]) == pytest.approx(g["GSS"], abs=1e-12)
    assert float(w[0, 0, 1, 1]) == pytest.approx(g["GSP"], abs=1e-12)
    assert float(w[0, 1, 0, 1]) == pytest.approx(g["HSP"], abs=1e-12)
    assert float(w[1, 1, 1, 1]) == pytest.approx(g["GPP"], abs=1e-12)
    assert float(w[1, 1, 2, 2]) == pytest.approx(g["GP2"], abs=1e-12)
    assert float(w[1, 2, 1, 2]) == pytest.approx(0.5 * (g["GPP"] - g["GP2"]), abs=1e-12)
    assert float(w[0, 1, 2, 3]) == 0.0


def test_one_center_spd_tensor_is_rotation_invariant():
    """The spd one-center tensor of a transition metal transforms as a tensor under rotations."""
    model = PM6()
    q = model._element_quantities([26])
    w = q["one_center"][0]
    vec = torch.tensor([[0.3, -0.5, 0.8], [-0.9, 0.2, 0.1]])
    frames = local_frames(vec, 9, constants(vec.device, vec.dtype))
    T = frames[0].T @ frames[1]          # a rotation in the molecular orbital basis
    rotated = torch.einsum("ai,bj,abcd,ck,dl->ijkl", T, T, w, T, T)
    assert torch.allclose(rotated, w, atol=1e-10)


def test_slater_condon_against_quadrature():
    """R^k of Slater orbitals equals a numerical double integral."""
    za, zb = torch.tensor(1.7), torch.tensor(2.3)
    value = float(slater_condon(2, 3, za, 3, zb, 3, zb, 3, za))
    r = np.linspace(0.0, 40.0, 400001)[1:]

    def radial(n, zeta):
        return (2 * zeta) ** (n + 0.5) / math.sqrt(math.factorial(2 * n)) * r ** (n - 1) * np.exp(-zeta * r)

    rho = radial(3, 1.7) * radial(3, 2.3) * r * r
    dr = r[1] - r[0]
    inner = np.cumsum(rho * r ** 2) * dr            # int_0^r rho r2^2
    outer = (np.sum(rho / r ** 3) - np.cumsum(rho / r ** 3)) * dr
    ref = np.sum(rho * (inner / r ** 3 + r ** 2 * outer)) * dr * 27.211386245988
    assert value == pytest.approx(ref, rel=1e-4)


@pytest.mark.parametrize("na,la,nb,lb,m", [(1, 0, 2, 0, 0), (2, 1, 2, 1, 0), (2, 1, 2, 1, 1),
                                           (3, 2, 2, 1, 1), (3, 2, 3, 2, 2), (4, 0, 3, 2, 0)])
def test_overlap_against_quadrature(na, la, nb, lb, m):
    """Local-frame overlaps of real Slater orbitals equal a numerical 3D integral."""
    za, zb, R = 1.3, 1.9, 2.4
    value = float(overlap_local(na, la, torch.tensor(za), nb, lb, torch.tensor(zb), torch.tensor(R), m))
    # direct quadrature in prolate spheroidal coordinates (the orbitals are
    # evaluated in Cartesian form, independent of the analytic expansion)
    u, wu = np.polynomial.legendre.leggauss(200)
    u, wu = 0.5 * (u + 1.0), 0.5 * wu
    xi, dxi = 1.0 + u / (1.0 - u), 1.0 / (1.0 - u) ** 2
    eta, weta = np.polynomial.legendre.leggauss(80)
    phi = np.linspace(0.0, 2.0 * math.pi, 32, endpoint=False)
    XI, ETA, PHI = np.meshgrid(xi, eta, phi, indexing="ij")
    W = (wu * dxi)[:, None, None] * weta[None, :, None] * (2.0 * math.pi / 32)
    h = R / 2.0
    rho = h * np.sqrt(np.clip((XI ** 2 - 1.0) * (1.0 - ETA ** 2), 0.0, None))
    x, y, z = rho * np.cos(PHI), rho * np.sin(PHI), h * (1.0 + XI * ETA)
    harm = {(0, 0): lambda x, y, z, r: 1 / math.sqrt(4 * math.pi) + 0 * x,
            (1, 0): lambda x, y, z, r: math.sqrt(3 / (4 * math.pi)) * z / r,
            (1, 1): lambda x, y, z, r: math.sqrt(3 / (4 * math.pi)) * x / r,
            (2, 0): lambda x, y, z, r: math.sqrt(5 / (16 * math.pi)) * (3 * z * z - r * r) / r ** 2,
            (2, 1): lambda x, y, z, r: math.sqrt(15 / (4 * math.pi)) * x * z / r ** 2,
            (2, 2): lambda x, y, z, r: math.sqrt(15 / (16 * math.pi)) * (x * x - y * y) / r ** 2}

    def sto(n, l, zeta, x, y, z):
        r = np.sqrt(x * x + y * y + z * z)
        rad = (2 * zeta) ** (n + 0.5) / math.sqrt(math.factorial(2 * n)) * r ** (n - 1) * np.exp(-zeta * r)
        return rad * harm[(l, m)](x, y, z, r)

    f = sto(na, la, za, x, y, z) * sto(nb, lb, zb, x, y, z - R)
    ref = np.sum(W * f * h ** 3 * (XI ** 2 - ETA ** 2))
    assert value == pytest.approx(ref, abs=1e-9)


def test_two_center_limits():
    """Two-center integrals tend to the one-center values at R -> 0 and to e^2/R far apart."""
    model = PM6()
    q = model._element_quantities([8])
    zero = torch.tensor([0])
    near = pair_integrals(q, zero, zero, torch.tensor([[0.0, 0.0, 1e-7]]), 4, False)
    w = near["W"][0]
    p = read_parameters("pm6")
    assert float(w[0, 0, 0, 0]) == pytest.approx(p.get(8, "GSS"), rel=1e-6)
    assert float(w[0, 1, 0, 1]) == pytest.approx(p.get(8, "HSP"), rel=1e-6)
    far = pair_integrals(q, zero, zero, torch.tensor([[0.0, 0.0, 1000.0]]), 4, False)
    assert float(far["W"][0, 0, 0, 0, 0]) == pytest.approx(COULOMB / 1000.0, rel=1e-5)
    assert float(far["W"][0, 3, 3, 1, 1]) == pytest.approx(COULOMB / 1000.0, rel=1e-5)


def test_config_interaction_against_charge_sums():
    """The point-charge interactions over the nonzero charges equal the plain double sum over all charges."""
    names = ["Q", "Dz", "Qxx", "Qxz", "z2", "xy"]
    q, u = config_tensors(names, torch.float64, "cpu")
    rng = np.random.default_rng(3)
    P, S = 5, len(names)
    a = rng.integers(0, S, size=S)
    b = rng.integers(0, S, size=S)
    da, db = (torch.as_tensor(rng.uniform(0.1, 1.5, size=(P, S))) for _ in range(2))
    r = torch.as_tensor(rng.uniform(0.5, 6.0, size=P))
    rho = torch.as_tensor(rng.uniform(0.0, 2.0, size=(P, S)))
    got = config_interaction([names[x] for x in a], da, [names[x] for x in b], db, r, rho)
    want = torch.zeros(P, S)
    for p in range(P):
        for s in range(S):
            for qi, ui in zip(q[a[s]], u[a[s]]):
                for qj, uj in zip(q[b[s]], u[b[s]]):
                    diff = uj * db[p, s] + torch.tensor([0.0, 0.0, float(r[p])]) - ui * da[p, s]
                    want[p, s] += qi * qj / torch.sqrt((diff * diff).sum() + rho[p, s] ** 2)
    assert torch.allclose(got, HARTREE * want, rtol=1e-12, atol=1e-12)


def test_pair_integrals_at_the_atoms_orbital_counts():
    """Integrals of a pair evaluated at its atoms' own orbital counts are the blocks of the padded ones."""
    model = PM6()
    q = model._element_quantities([1, 8, 22])
    n_orb = q["n_orb"]
    rng = np.random.default_rng(5)
    for ea, eb in ((1, 0), (0, 1), (2, 1), (0, 2), (2, 2), (0, 0)):
        vec = torch.as_tensor(rng.normal(size=(3, 3)) * 1.5)
        e_a, e_b = torch.full((3,), ea), torch.full((3,), eb)
        ka, kb = n_orb[ea], n_orb[eb]
        own = pair_integrals(q, e_a, e_b, vec, ka, True, k_b=kb)
        full = pair_integrals(q, e_a, e_b, vec, 9, True)
        assert torch.allclose(own["S"], full["S"][:, :ka, :kb], rtol=1e-12, atol=1e-14)
        assert torch.allclose(own["VA"], full["VA"][:, :ka, :ka], rtol=1e-12, atol=1e-12)
        assert torch.allclose(own["VB"], full["VB"][:, :kb, :kb], rtol=1e-12, atol=1e-12)
        assert torch.allclose(own["W"], full["W"][:, :ka, :ka, :kb, :kb], rtol=1e-12, atol=1e-12)
        assert torch.equal(own["gab"], full["gab"])


def test_pair_kernel_scripts_and_matches_eager():
    """The two-center kernel compiles with TorchScript and gives the eager integrals and their gradients, per
    orbital-count group and padded (H, O, Ti: s, sp and spd atoms)."""
    model = PM6()
    q = model._element_quantities([1, 8, 22])
    qt = {key: v for key, v in q.items() if isinstance(v, torch.Tensor)}
    scripted = torch.jit.script(twocenter.pair_kernel)
    rng = np.random.default_rng(11)
    c = constants(torch.device("cpu"), torch.float64)
    for ka, kb, ea, eb, padded in ((1, 1, 0, 0, False), (4, 1, 1, 0, False), (9, 4, 2, 1, False),
                                   (9, 9, 2, 2, False), (9, 9, 0, 2, True)):
        vec = torch.tensor(rng.normal(size=(4, 3)) * 1.5, requires_grad=True)
        e_a, e_b = torch.full((4,), ea), torch.full((4,), eb)
        if padded:
            e_a = torch.tensor([0, 1, 2, 2])
        tab = twocenter.pair_tables(q, ka, kb, True, vec.device, vec.dtype)
        eager = twocenter.pair_kernel(qt, tab, c, e_a, e_b, vec, ka, kb, ka == 9, padded, None)
        compiled = scripted(qt, tab, c, e_a, e_b, vec, ka, kb, ka == 9, padded, None)
        for key in eager:
            assert torch.allclose(compiled[key], eager[key], rtol=1e-14, atol=1e-14), key
        g_e, = torch.autograd.grad(sum(v.sum() for v in eager.values()), vec)
        g_c, = torch.autograd.grad(sum(v.sum() for v in compiled.values()), vec)
        assert torch.allclose(g_c, g_e, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("cls", (PM6, PM7))
def test_pair_groups_match_one_padded_group(cls, monkeypatch):
    """Pairs grouped by their atoms' orbital counts give the energies and forces of one padded group."""
    graph = collate([_graph(*FORMAMIDE), _graph(*WATER), _graph([[0, 0, 0], [2.1, 0, 0], [0, 1.6, 0.1]],
                                                                 [22, 17, 1])])
    model = ForceStressOutput(cls(reuse_density=False))
    monkeypatch.setattr(scf, "GROUP_MIN_PAIRS", 0)
    grouped = model(graph)
    monkeypatch.setattr(scf, "GROUP_MIN_PAIRS", 10 ** 9)
    padded = model(graph)
    assert torch.allclose(grouped["energy"], padded["energy"], rtol=0.0, atol=1e-8)
    assert torch.allclose(grouped["forces"], padded["forces"], rtol=0.0, atol=1e-7)


def test_host_terms_match_their_torch_form():
    """The hydrogen-bond, H-H and dispersion corrections on the host tape give the energies, gradients and
    second derivatives of their PyTorch form."""
    from xnn.sqmnn.models import corrections as cr
    from xnn.sqmnn.models.tape import TORCH
    rng = np.random.default_rng(11)
    mols = [([8, 1, 1], [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]),
            ([7, 1, 1, 1, 1], [[0, 0, 0], [0.6, 0.6, 0.6], [-0.6, -0.6, 0.6], [-0.6, 0.6, -0.6], [0.6, -0.6, -0.6]]),
            ([6, 8, 8, 1], [[0, 0, 0], [1.25, 0, 0], [-0.62, 1.08, 0], [-0.55, -0.95, 0]])]
    z, pos = [], []
    for m in range(8):
        mz, mp = mols[m % 3]
        R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        z += mz
        pos += list(np.asarray(mp) @ R.T + rng.uniform(0, 6.0, 3))
    pos = np.asarray(pos)
    tables = cr.correction_tables(torch.device("cpu"))
    bonds = cr.bond_topology(z, torch.tensor(pos), tables)
    w, v = torch.tensor(rng.normal(size=len(z))), torch.tensor(rng.normal(size=(len(z), 3)))
    terms = {"pm7_hbond": partial(cr.pm7_hbond_energy, TORCH, z, tables=tables),
             "h4_hbond": partial(cr.h4_energy, TORCH, z, tables=tables),
             "hh_repulsion": partial(cr.hh_energy, TORCH, z)}
    for name, term in terms.items():
        results = []
        for f in (partial(getattr(cr, name), z, tables=tables), term):
            p = torch.tensor(pos, requires_grad=True)
            e = f(p)
            g, = torch.autograd.grad((e * w).sum(), p, create_graph=True)
            hv, = torch.autograd.grad((g * v).sum(), p)
            results.append((e.detach(), g.detach(), hv))
        assert float(results[0][0].abs().sum()) > 0.0, name
        for a, b in zip(*results):
            assert torch.allclose(a, b, rtol=1e-12, atol=1e-12), name
    # the dispersion through the host tape against the plain sum over pairs
    p = torch.tensor(pos, requires_grad=True)
    e = cr.pm7_dispersion(z, p, bonds, tables)
    assert float(e.sum()) < 0.0
    g, = torch.autograd.grad(e.sum(), p)
    assert torch.allclose(g.sum(0), torch.zeros(3), atol=1e-12)     # translation invariance
    # every method's corrections compiled with TorchScript (the torch form) against the eager ones (the tape)
    scripted = torch.jit.script(cr.structure_corrections)
    for kind in range(4):
        out = []
        for f in (cr.structure_corrections, scripted):
            p = torch.tensor(pos, requires_grad=True)
            e = f(kind, z, p, tables, 7.1853)
            g, = torch.autograd.grad((e * w).sum(), p, allow_unused=True) if e.requires_grad else (None,)
            out.append((e.detach(), g if g is not None else torch.zeros_like(p)))
        # this system has PM7's and PM6-ORG's corrections (no amide or planar amine for PM3 and PM6)
        assert kind < 2 or float(out[0][0].abs().sum()) > 0.0, kind
        for a, b in zip(*out):
            assert torch.allclose(a, b, rtol=1e-12, atol=1e-12), kind


def test_pm7_hbond_search_forms_agree():
    """The host (NumPy) and the PyTorch searches of the PM7 hydrogen bonds give the same tables on crowded
    random boxes (centers with five or more bonded atoms, every kind of reference frame)."""
    import xnn.sqmnn.models.corrections as cr
    tables = correction_tables(torch.device("cpu"))
    seen = {"capped": 0, "one bond": 0, "rows": 0}
    for seed in range(12):
        rng = np.random.default_rng(seed)
        n = 30 + 5 * seed
        z = [int(v) for v in rng.choice([1, 1, 1, 6, 7, 8], size=n)]
        x = rng.uniform(0.0, (n / 0.25) ** (1.0 / 3.0), size=(n, 3))
        host = cr._np_pm7_hbond_tables(z, x)
        torch_form = cr._pm7_hbond_tables(z, torch.tensor(x), tables)
        assert (host is None) == (torch_form is None)
        if host is None:
            continue
        assert host.keys() == torch_form.keys()
        for key, value in host.items():
            assert np.array_equal(value, torch_form[key].numpy()), key
        _, count, _, k = cr._np_bonded_lists(cr._distances(x), cr._NP_HB_RADIUS[np.asarray(z)], 4)
        seen["capped"] += int((count >= 5).sum())
        seen["one bond"] += int((k[host["atoms"][..., 0]] == 1).sum())
        seen["rows"] += len(host["ahb"])
    assert min(seen.values()) > 0, seen


def _build_graphs(model, kind):
    """The captured build graphs of one kind of a model on the fast path."""
    return [g for plan in model._plans.values() for (k, _), g in plan.graphs.items() if k == kind and g]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("cls", (PM6, PM7))
def test_fast_path_matches_reference_on_gpu(cls):
    """The graphs of the fast path (SCF stages, the integral build and its derivatives to second order, replayed
    once a composition recurs) give the energies, forces, second derivatives and force-loss parameter gradients
    of the reference."""
    pos = np.vstack([FORMAMIDE[0], np.asarray(WATER[0]) + [3.0, 0.5, 0.2]])
    z = FORMAMIDE[1] + WATER[1]
    v = torch.tensor(np.random.default_rng(2).normal(size=pos.shape), device="cuda")
    results = []
    for fast in (False, True):
        m = cls(reuse_density=False).cuda().eval()
        m.set_use_fast(fast)
        for _ in range(5):
            p = torch.tensor(pos, device="cuda", requires_grad=True)
            g = structure_to_graph({"pos": p, "atomic_numbers": torch.tensor(z)}, 0.1, device="cuda")
            g.pos = p
            e = m(g)["energy"].sum()
            f, = torch.autograd.grad(e, p, create_graph=True)
            hv, = torch.autograd.grad((f * v).sum(), p)
        if fast:
            assert m._scf_graphs.entries
            assert any(g.second for g in _build_graphs(m, "eval"))
        t = cls(trainable=("USS", "BETAS"), reuse_density=False).cuda()
        t.set_use_fast(fast)
        for _ in range(6):
            t.zero_grad()
            p = torch.tensor(pos, device="cuda", requires_grad=True)
            g = structure_to_graph({"pos": p, "atomic_numbers": torch.tensor(z)}, 0.1, device="cuda")
            g.pos = p
            et = t(g)["energy"].sum()
            ft, = torch.autograd.grad(et, p, create_graph=True)
            (et + (ft * ft).sum()).backward()
        grads = torch.cat([q.grad.reshape(-1) for q in t.parameters() if q.grad is not None])
        if fast:
            assert any(g.second for g in _build_graphs(t, "train")) and _build_graphs(t, "response")
            assert any(key[0] == "cphf" for key in t._scf_graphs.entries)     # the density response
        results.append((e.detach(), f.detach(), hv, grads))
    for a, b in zip(*results):
        assert torch.allclose(a, b, rtol=1e-9, atol=1e-9 * float(b.abs().max()))


def test_pseudo_diagonalization_matches_full_diagonalization(monkeypatch):
    """The SCF with orbital rotations near convergence reaches the solution of full diagonalizations."""
    graph = collate([_graph(*FORMAMIDE), _graph(*WATER, charge=1.0, mult=2.0)])
    monkeypatch.setattr(scf, "PSEUDO_MIN_BASIS", 0)
    calls = []
    rotate = scf._pseudo_diagonalize
    monkeypatch.setattr(scf, "_pseudo_diagonalize", lambda *a, **k: calls.append(1) or rotate(*a, **k))
    model = ForceStressOutput(PM6(reuse_density=False))
    ref = model(graph)
    assert calls
    monkeypatch.setattr(nddo, "run_scf", partial(scf.run_scf, pseudo_diagonalization=0.0))
    full = model(graph)
    assert torch.allclose(ref["energy"], full["energy"], rtol=0.0, atol=1e-7)
    assert torch.allclose(ref["forces"], full["forces"], rtol=0.0, atol=1e-6)


def test_warm_start_from_previous_orbitals(monkeypatch):
    """A warm start that rotates the previous geometry's orbitals from its first iteration (WARM_ORBITALS) needs
    one full diagonalization fewer and reaches the solution of the warm start from the density alone."""
    rng = np.random.default_rng(4)
    pos = np.vstack([np.asarray(WATER[0]) + 2.9 * np.array(s) for s in np.ndindex(2, 2, 2)])
    pos += 0.05 * rng.normal(size=pos.shape)
    step = pos + 0.01 * rng.normal(size=pos.shape)
    z = WATER[1] * 8
    n = 6 * len(z) // 3                                                # basis functions
    monkeypatch.setattr(scf, "PSEUDO_MIN_BASIS", 0)
    calls = []
    full = scf.eigh
    monkeypatch.setattr(scf, "eigh", lambda A: calls.append(A.shape[-1]) or full(A))
    results = []
    for seed in (False, True):
        monkeypatch.setattr(scf, "WARM_ORBITALS", {"cpu": seed, "cuda": seed})
        model = ForceStressOutput(PM6())
        model(_graph(pos, z))
        calls.clear()
        results.append(model(_graph(step, z)))
        results[-1]["full diagonalizations"] = sum(1 for m in calls if m == n)
    plain, seeded = results
    assert seeded["full diagonalizations"] == plain["full diagonalizations"] - 1
    assert torch.allclose(plain["energy"], seeded["energy"], rtol=0.0, atol=1e-7)
    assert torch.allclose(plain["forces"], seeded["forces"], rtol=0.0, atol=1e-6)


@pytest.mark.parametrize("cls", METHODS)
def test_hydrogen_atom_heat_of_formation(cls):
    """An isolated hydrogen atom has the experimental heat of formation, 52.102 kcal/mol."""
    out = cls()(_graph([[0.0, 0.0, 0.0]], [1], mult=2))
    assert float(out["heat_of_formation"][0]) == pytest.approx(el.ATOM_HEAT[1], abs=1e-9)


@pytest.mark.parametrize("cls", METHODS)
def test_invariance_and_partition(cls):
    """Energies are invariant to rotation and translation; per-atom energies sum to the total."""
    pos, z = FORMAMIDE
    model = cls()
    a = model(_graph(pos, z))
    b = model(_graph(pos @ _rotation(1).T + np.array([0.3, -2.0, 1.1]), z))
    assert float(a["energy"][0]) == pytest.approx(float(b["energy"][0]), abs=1e-9)
    assert float(a["node_energy"].sum()) == pytest.approx(float(a["energy"][0]), abs=1e-10)
    assert float(a["heat_of_formation"][0]) == pytest.approx(float(a["energy"][0]) * EV_TO_KCAL, abs=1e-9)


@pytest.mark.parametrize("cls", METHODS)
def test_forces_match_finite_differences(cls):
    """Forces are the negative gradient of the heat of formation."""
    pos, z = FORMAMIDE
    model = ForceStressOutput(cls(tol_density=1e-11, tol_energy=1e-13))
    forces = model(_graph(pos, z))["forces"].detach().numpy()
    h = 1e-4
    for atom, c in ((0, 0), (2, 1), (5, 2)):
        e = []
        for s in (1, -1):
            p = pos.copy()
            p[atom, c] += s * h
            with torch.no_grad():
                e.append(float(model.model(_graph(p, z))["energy"][0]))
        assert forces[atom, c] == pytest.approx(-(e[0] - e[1]) / (2 * h), abs=2e-6)


def test_transition_metal_rotation_invariance():
    """spd integrals and the PM7 d-shell balance are invariant to rotation."""
    t = np.array([[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]]) / math.sqrt(3.0) * 2.17
    pos = np.vstack([[0, 0, 0], t + np.array([[0.02, 0, 0], [0, 0, 0], [0, 0.01, 0], [0, 0, 0]])])
    for cls in (PM6, PM7):
        model = cls()
        a = float(model(_graph(pos, [22, 17, 17, 17, 17]))["energy"][0])
        b = float(model(_graph(pos @ _rotation(3).T, [22, 17, 17, 17, 17]))["energy"][0])
        assert a == pytest.approx(b, abs=1e-8)


def test_batch_matches_individual():
    """A batch of molecules gives the energies of the molecules evaluated one by one."""
    model = PM6()
    graphs = [_graph(*WATER), _graph(*FORMAMIDE), _graph([[0, 0, 0], [0, 0, 0.74]], [1, 1])]
    single = [float(model(g)["energy"][0]) for g in graphs]
    batched = model(collate(graphs))["energy"].tolist()
    assert np.allclose(single, batched, atol=1e-9)


def test_open_shell_and_charge_handling():
    """Odd electron counts default to doublets; inconsistent spins and missing elements raise."""
    model = PM6()
    ch3 = [[0, 0, 0.02], [1.08, 0, 0], [-0.54, 0.94, 0.01], [-0.55, -0.93, 0]]
    a = model(_graph(ch3, [6, 1, 1, 1]))
    b = model(_graph(ch3, [6, 1, 1, 1], mult=2))
    assert float(a["energy"][0]) == pytest.approx(float(b["energy"][0]), abs=1e-9)
    with pytest.raises(ValueError):
        model(_graph(ch3, [6, 1, 1, 1], mult=1))
    with pytest.raises(ValueError):
        PM6ORG()(_graph([[0, 0, 0], [0, 0, 2.2]], [22, 17]))
    ion = model(_graph([[0, 0, 0], [0.02, 0, 0.97]], [8, 1], charge=-1))
    assert float(ion["charges"].sum()) == pytest.approx(-1.0, abs=1e-10)


def test_density_reuse_gives_same_energy():
    """Restarting from the previous density converges to the same state."""
    pos, z = FORMAMIDE
    model = PM7()
    a = float(model(_graph(pos, z))["energy"][0])
    cold = model.last_scf.iterations
    b = float(model(_graph(pos + 1e-3, z))["energy"][0])
    c = float(PM7(reuse_density=False)(_graph(pos + 1e-3, z))["energy"][0])
    assert b == pytest.approx(c, abs=1e-9) and model.last_scf.iterations < cold
    assert a != b
    # a geometry beyond warm_start_distance (another sample, not a step) starts cold
    moved = pos + np.array([[0.0, 0.0, 0.0]] * (len(z) - 1) + [[0.5, 0.0, 0.0]])
    model(_graph(moved, z))
    fresh = PM7(reuse_density=False)
    fresh(_graph(moved, z))
    assert model.last_scf.iterations == fresh.last_scf.iterations


@pytest.mark.parametrize("cls,group,z", [(PM3, "BETAP", 8), (PM6, "ZS", 1), (PM7, "GSS", 7),
                                         (PM6ORG, "PAR", 3)])
def test_parameter_gradients_through_scf_response(cls, group, z):
    """Parameter gradients of energy, forces, dipole and charges equal finite differences."""
    pos, zs = FORMAMIDE

    def values(model):
        out = ForceStressOutput(model)(_graph(pos, zs))
        w = torch.linspace(0.3, 1.1, out["forces"].numel()).reshape(out["forces"].shape)
        return [out["energy"].sum(), (out["forces"] * w).sum(),
                (out["dipole"] * torch.tensor([0.7, -0.4, 1.3])).sum(),
                (out["charges"] * torch.arange(len(zs))).sum()]

    model = cls(trainable=[group], tol_density=1e-12, tol_energy=1e-13)
    model.train()
    grads = []
    for v in values(model):
        model.zero_grad()
        if v.requires_grad:
            v.backward(retain_graph=True)
            g = model.params[group].grad
            grads.append(0.0 if g is None else float(g[z]))
        else:
            grads.append(0.0)
    h = 1e-5
    fd = []
    for s in (1, -1):
        m = cls(trainable=[group], tol_density=1e-12, tol_energy=1e-13)
        with torch.no_grad():
            m.params[group][z] += s * h
        m.eval()
        fd.append([float(v) for v in values(m)])
    for g, vp, vm in zip(grads, *fd):
        assert g == pytest.approx((vp - vm) / (2 * h), rel=1e-5, abs=1e-8)


def test_training_step_reduces_loss():
    """A few optimizer steps on trainable parameters lower an energy + force loss."""
    pos, z = WATER
    model = PM3(trainable=["USS", "BETAS", "BETAP"])
    wrapped = ForceStressOutput(model)
    model.train()
    target_e = torch.tensor([-2.5])
    target_f = torch.zeros(3, 3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(5):
        out = wrapped(_graph(pos, z))
        loss = (out["energy"] - target_e).pow(2).sum() + out["forces"].sub(target_f).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0]


def test_yne_and_hh_corrections():
    """The acetylenic correction is 12 kcal/mol per bond below 1.21 Angstrom and zero above 1.33."""
    bonds = [[1], [0]]
    short = yne_correction([6, 6], torch.tensor([[0.0, 0, 0], [0, 0, 1.2]]), bonds).sum()
    long = yne_correction([6, 6], torch.tensor([[0.0, 0, 0], [0, 0, 1.4]]), bonds).sum()
    assert float(short) == pytest.approx(12.0) and float(long) == 0.0
    hh = hh_repulsion([1, 1], torch.tensor([[0.0, 0, 0], [0, 0, 0.9]]), correction_tables(torch.device("cpu"))).sum()
    assert float(hh) == pytest.approx(25.46293603147693)


def test_h4_hydrogen_bond_sign_and_range():
    """H4 stabilizes a linear O-H...O bond and vanishes beyond its 5.5 Angstrom range."""
    def dimer(d):
        return torch.tensor([[0.0, 0, 0], [0.97, 0, 0], [d, 0, 0], [d + 0.3, 0.9, 0], [-0.3, 0.9, 0],
                             [d + 0.3, -0.9, 0]])
    z = [8, 1, 8, 1, 1, 1]
    tables = correction_tables(torch.device("cpu"))
    assert float(h4_hbond(z, dimer(2.9), tables).sum()) < -0.5
    assert float(h4_hbond(z, dimer(5.6), tables).sum()) == 0.0


@pytest.mark.skipif(not os.environ.get("XNN_MOPAC"), reason="needs the MOPAC executable (XNN_MOPAC)")
@pytest.mark.parametrize("method", ["PM3", "PM6", "PM6-ORG", "PM7"])
def test_mopac_fidelity(method):
    """Heats of formation, gradients, dipoles and charges equal MOPAC's."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("fidelity", HERE / "sqmnn_mopac_fidelity.py")
    fidelity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fidelity)
    worst = fidelity.compare(method, verbose=False)
    assert worst["hof"] < 1e-5
    assert worst["gradients"] < 2e-3
    assert worst["dipole"] < 1e-3 and worst["charges"] < 1e-4


def _export_checks(cls, tmp_path, device):
    from xnn.common.deploy import export_torchscript_potential
    pos = np.vstack([FORMAMIDE[0], np.asarray(WATER[0]) + [3.0, 0.5, 0.2]])
    z = FORMAMIDE[1] + WATER[1]
    eager = cls(reuse_density=False).to(device).eval()
    path = export_torchscript_potential(cls().to(device).eval(), cutoff=10.0, path=str(tmp_path / "m.pt"))
    scripted = torch.jit.load(path)
    # the graph executor's optimizations take seconds on the first calls and leave the time per call as it is
    with torch.jit.optimized_execution(False):
        _trajectory_checks(scripted, eager, pos, z, device)


def _trajectory_checks(scripted, eager, pos, z, device):
    from xnn.common.deploy.torchscript import build_neighbor_list_ts
    rng = np.random.default_rng(3)
    for step in range(3):
        # a short trajectory: after the first call the exported model starts each SCF from the last density
        p = torch.tensor(pos + 0.02 * step * rng.normal(size=pos.shape), device=device)
        out = scripted(p, torch.tensor(z, device=device))
        g = structure_to_graph({"pos": p, "atomic_numbers": torch.tensor(z)}, 10.0, device=device)
        ref = ForceStressOutput(eager).eval()(g)
        e_tol, f_tol = (1e-10, 1e-10) if step == 0 else (1e-9, 1e-7)       # warm starts: the SCF tolerance
        assert abs(float(out["energy"]) - float(ref["energy"].sum())) < e_tol
        assert torch.allclose(out["forces"], ref["forces"], atol=f_tol, rtol=0.0)
        assert torch.allclose(out["charges"], ref["charges"], atol=1e-8, rtol=0.0)
    zero = torch.zeros((3, 3), dtype=torch.float64, device=device)
    ei, cs = build_neighbor_list_ts(p, 10.0, zero, torch.zeros(3, dtype=torch.bool, device=device))
    pair = scripted.forward_lammps(p, ei, cs, torch.tensor(z, device=device), zero)
    assert abs(float(pair["energy"]) - float(out["energy"])) < 1e-9
    assert torch.allclose(pair["forces"], out["forces"], atol=1e-7, rtol=0.0)
    with pytest.raises(Exception, match="periodic"):
        scripted(p, torch.tensor(z, device=device), torch.eye(3, dtype=torch.float64, device=device) * 20.0,
                 torch.ones(3, dtype=torch.bool, device=device))


@pytest.mark.parametrize("cls", (PM3, PM6, PM7, PM6ORG))
def test_torchscript_export_matches_eager(cls, tmp_path):
    """The exported TorchScript potential (``forward`` and ``forward_lammps``) gives the eager model's heat of
    formation, forces and charges, carries the SCF warm start of an MD run and refuses periodic cells."""
    _export_checks(cls, tmp_path, "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_torchscript_export_matches_eager_on_gpu(tmp_path):
    """The same, exported from a model on a GPU."""
    _export_checks(PM7, tmp_path, "cuda")
