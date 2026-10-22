"""NDDO semiempirical methods (sqmnn): integrals, SCF, energies, derivatives, training.

Self-contained checks of the published approximations (Dewar & Thiel 1977;
Thiel & Voityuk 1992; Stewart 1989, 2007, 2013, 2023): the one-center
integrals reproduce the parameters they are built from, the two-center
integrals reach their one-center and point-charge limits, overlaps agree
with numerical quadrature, the energy is invariant and its derivatives match
finite differences, and parameter gradients through the SCF response are
exact. Parity with the MOPAC program runs when ``XNN_MOPAC`` names its
executable (``sqmnn_mopac_parity.py``).
"""
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from xnn.common.config.schema import ModelConfig
from xnn.common.data import collate, structure_to_graph
from xnn.common.models import ForceStressOutput, build_model
from xnn.sqmnn.common import elements as el
from xnn.sqmnn.common import list_parameter_sets, read_parameters, write_parameters
from xnn.sqmnn.common.constants import COULOMB, EV_TO_KCAL
from xnn.sqmnn.models import PM3, PM6, PM6ORG, PM7
from xnn.sqmnn.models.angular import LOCAL, gaunt, multipole_coefficient
from xnn.sqmnn.models.corrections import h4_hbond, hh_repulsion, yne_correction
from xnn.sqmnn.models.integrals import local_frames, overlap_local, slater_condon
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
    root = HERE.parent / "configs" / "model"
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
    frames = local_frames(torch.tensor([[0.3, -0.5, 0.8], [-0.9, 0.2, 0.1]]), 9)
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
    hh = hh_repulsion([1, 1], torch.tensor([[0.0, 0, 0], [0, 0, 0.9]])).sum()
    assert float(hh) == pytest.approx(25.46293603147693)


def test_h4_hydrogen_bond_sign_and_range():
    """H4 stabilizes a linear O-H...O bond and vanishes beyond its 5.5 Angstrom range."""
    def dimer(d):
        return torch.tensor([[0.0, 0, 0], [0.97, 0, 0], [d, 0, 0], [d + 0.3, 0.9, 0], [-0.3, 0.9, 0],
                             [d + 0.3, -0.9, 0]])
    z = [8, 1, 8, 1, 1, 1]
    assert float(h4_hbond(z, dimer(2.9)).sum()) < -0.5
    assert float(h4_hbond(z, dimer(5.6)).sum()) == 0.0


@pytest.mark.skipif(not os.environ.get("XNN_MOPAC"), reason="needs the MOPAC executable (XNN_MOPAC)")
@pytest.mark.parametrize("method", ["PM3", "PM6", "PM6-ORG", "PM7"])
def test_mopac_parity(method):
    """Heats of formation, gradients, dipoles and charges equal MOPAC's."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("parity", HERE / "sqmnn_mopac_parity.py")
    parity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parity)
    worst = parity.compare(method, verbose=False)
    assert worst["hof"] < 1e-5
    assert worst["gradients"] < 2e-3
    assert worst["dipole"] < 1e-3 and worst["charges"] < 1e-4
