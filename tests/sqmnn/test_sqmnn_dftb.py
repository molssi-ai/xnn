"""DFTB1, DFTB2 and DFTB3: Slater-Koster files, tables, rotations, gamma, energies and derivatives."""
from __future__ import annotations

import math
import os
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch

from xnn.common.data import collate, structure_to_graph
from xnn.common.models import ForceStressOutput, hessian
from xnn.sqmnn.common import skf
from xnn.sqmnn.models import DFTB, DFTB1, DFTB2, DFTB3
from xnn.sqmnn.models import dftb as D
from xnn.sqmnn.models.slater_koster import monomials, real_harmonics, rotation_polynomials, shell_block

LMAX = {"H": 0, "C": 1, "S": 2}
LEVELS = {"H": (-0.24,), "C": (-0.50, -0.19), "S": (-0.63, -0.26, 0.32)}
HUBBARD = {"H": (0.42,), "C": (0.36, 0.39), "S": (0.33, 0.30, 0.31)}
OCCUPATIONS = {"H": (1.0,), "C": (2.0, 2.0), "S": (2.0, 4.0, 0.0)}
MASS = {"H": 1.008, "C": 12.01, "S": 32.06}
DERIVS = {"H": -0.19, "C": -0.15, "S": -0.11}
GRID, NGRID = 0.02, 600
THIOFORMALDEHYDE = (["C", "S", "H", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.61], [0.0, 0.93, -0.55],
                                                    [0.05, -0.92, -0.56]]))
METHANE = (["C", "H", "H", "H", "H"], np.array([[0.0, 0.0, 0.0], [0.63, 0.63, 0.63], [-0.63, -0.6, 0.66],
                                                [-0.62, 0.64, -0.63], [0.64, -0.63, -0.61]]))
HS = (["S", "H"], np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 1.33]]))


@pytest.fixture(autouse=True)
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _column(a: str, b: str, name: str, overlap: bool) -> np.ndarray:
    """A smooth integral column of the synthetic set (decaying, distinct per pair and column)."""
    r = GRID * np.arange(1, NGRID + 1)
    seed = sum(map(ord, a + b + name)) % 17
    if overlap:
        return (0.25 + 0.01 * seed) * np.exp(-(0.75 + 0.02 * seed) * r) * (1.0 + 0.1 * r)
    return -(0.35 + 0.01 * seed) * np.exp(-(0.6 + 0.02 * seed) * r) * (1.0 + 0.2 * r - 0.01 * r * r)


def _skf_text(a: str, b: str) -> str:
    """A synthetic Slater-Koster file in the simple format, with Fortran repeat counts and a spline."""
    lines = [f"{GRID}, {NGRID}"]
    if a == b:
        lv = list(LEVELS[a]) + [0.0] * (3 - len(LEVELS[a]))
        hu = list(HUBBARD[a]) + [HUBBARD[a][0]] * (3 - len(HUBBARD[a]))
        oc = list(OCCUPATIONS[a]) + [0.0] * (3 - len(OCCUPATIONS[a]))
        lines.append(" ".join(f"{x:.10f}" for x in lv[::-1]) + ", 0.0, " + " ".join(f"{x}" for x in hu[::-1])
                     + " " + " ".join(f"{x}" for x in oc[::-1]))
    lines.append(f"{MASS[a] if a == b else 0.0}, 19*0.0")
    table = np.stack([_column(a, b, c, False) for c in skf.COLUMNS] + [_column(a, b, c, True) for c in skf.COLUMNS], 1)
    lines += [" ".join(f"{x:.15e}" for x in row) for row in table]
    start, cut, amp = 1.0, 4.0, 0.02
    length = cut - start
    a1 = 3.0 / length
    a2 = math.log(amp * length ** 3) + a1 * start
    lines += ["Spline", f"1 {cut}", f"{a1:.15f} {a2:.15f} 0.0",
              f"{start} {cut} {amp * length ** 3:.15e} {-3 * amp * length ** 2:.15e} {3 * amp * length:.15e} "
              f"{-amp:.15e} 0.0 0.0", "", "<Documentation>", "</Documentation>"]
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="module")
def skset(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("skf")
    for a in LMAX:
        for b in LMAX:
            (folder / f"{a}-{b}.skf").write_text(_skf_text(a, b))
    return folder


def _graph(symbols, pos, charge=0.0):
    from xnn.common.data.elements import CHEMICAL_SYMBOLS
    z = np.array([CHEMICAL_SYMBOLS.index(s) for s in symbols])
    return structure_to_graph({"pos": torch.as_tensor(np.asarray(pos, dtype=np.float64)), "atomic_numbers": z,
                               "total_charge": charge}, 0.1)


def _model(cls, folder, **kw):
    kw.setdefault("max_angular_momentum", {"H": "s", "C": "p", "S": "d"})
    if cls is DFTB3:
        kw.setdefault("hubbard_derivatives", DERIVS)
        kw.setdefault("damping_exponent", 4.0)
    if cls is not DFTB1:
        kw.setdefault("scc_tolerance", 1e-12)
        kw.setdefault("max_iter", 500)
    return cls(str(folder), **kw)


def test_parse_simple_and_extended_files(skset):
    """The reader takes the atomic line, the repulsive line, the table rows and the spline of a file."""
    f = skf.parse_skf((skset / "S-S.skf").read_text(), True)
    assert f.grid == GRID and f.n_grid == NGRID and f.hamiltonian.shape == (NGRID, 10)
    assert f.energies == LEVELS["S"] and f.hubbard == HUBBARD["S"] and f.occupations == OCCUPATIONS["S"]
    assert f.mass == MASS["S"] and f.polynomial == (0.0,) * 8 and f.spline.cutoff == 4.0
    assert np.allclose(f.overlap[:, f.columns.index("sp0")], _column("S", "S", "sp0", True))
    g = skf.parse_skf((skset / "C-H.skf").read_text(), False)
    assert g.energies is None and g.n_shells is None
    ext = "@ extended\n0.1 9 4\n" + "0.0 " * 4 + "0.0 " + "0.5 " * 4 + "0.0 0.0 2.0 2.0\n" + "1.0 19*0.0\n" + \
        "\n".join(" ".join(["1.0"] * 40) for _ in range(9)) + "\n"
    e = skf.parse_skf(ext, True)
    assert e.extended and e.n_shells == 4 and e.hamiltonian.shape == (9, 20) and e.columns == skf.EXTENDED_COLUMNS
    assert e.occupations == (2.0, 2.0, 0.0, 0.0) and e.spline is None


def test_set_lookup_and_readme_recommendations(skset, tmp_path):
    """A folder of files is a set; its README gives angular momenta, Hubbard derivatives and zeta."""
    s = skf.load_set(str(skset))
    assert s.elements() == (1, 6, 16) and s.has(6, 16) and not s.has(6, 7)
    with pytest.raises(KeyError):
        s.pair(6, 7)
    assert skf.load_set(s) is s
    folder = tmp_path / "set" / "skfiles"
    folder.mkdir(parents=True)
    (folder.parent / "README").write_text("zeta = 4.00 (exponent)\n\nHubbard derivatives (au):\n C = -0.1492\n"
                                          " H = -0.1857\n\n---------\nMaximum angular momenta:\n\n C = \"p\"\n"
                                          "Cl = \"d\"\n\n---------\n")
    rec = skf.SlaterKosterSet(folder).recommended()
    assert rec["max_angular_momentum"] == {6: "p", 17: "d"}
    assert rec["hubbard_derivatives"] == {6: -0.1492, 1: -0.1857} and rec["damping_exponent"] == {0: 4.0}
    assert skf.SlaterKosterSet(tmp_path).readme() == ""


def test_published_sets_download_and_check(skset, tmp_path, monkeypatch):
    """A published set is fetched once, checked against its digest and read offline afterwards."""
    from xnn.common.data.hub import _download

    archive = tmp_path / "src.tar.gz"
    with tarfile.open(archive, "w:gz") as t:
        for p in sorted(skset.glob("*.skf")):
            t.add(p, arcname=f"top/skfiles/{p.name}")
    digest = skf.content_digest(skset)
    monkeypatch.setitem(skf.PUBLISHED, "toy-1-0", skf._Published("toy", "0" * 40, digest))
    def fake_download(url, dest, *args, **kwargs):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(archive.read_bytes())
        return Path(dest)

    monkeypatch.setattr(_download, "download_file", fake_download)
    monkeypatch.delenv("XNN_OFFLINE", raising=False)
    cache = tmp_path / "cache"
    folder = skf.published_folder("toy-1-0", cache)
    from xnn.common.models.hub.provenance import NOTE
    assert skf.content_digest(folder) == digest and (folder.parent / NOTE).exists()
    monkeypatch.setenv("XNN_OFFLINE", "1")
    assert skf.published_set("toy-1-0", cache).elements() == (1, 6, 16)
    with pytest.raises(FileNotFoundError):
        skf.published_folder("toy-1-0", tmp_path / "empty")
    with pytest.raises(KeyError):
        skf.published_folder("nope")
    with pytest.raises(KeyError):
        skf.published_folder("mio/split", cache)
    with pytest.raises(KeyError):
        skf.published_folder("ob2/wrong", cache)
    monkeypatch.delenv("XNN_OFFLINE")
    monkeypatch.setitem(skf.PUBLISHED, "toy-1-0", skf._Published("toy", "0" * 40, "f" * 32))
    with pytest.raises(ValueError):
        skf.published_folder("toy-1-0", tmp_path / "other")


def _lagrange_reference(rows, grid, r):
    x = r / grid - 1.0
    i0 = min(max(int(np.floor(x)) - 3, 0), len(rows) - 8)
    c = np.polyfit(np.arange(i0, i0 + 8, dtype=float), rows[i0:i0 + 8], 7)
    return np.polyval(c, x)


def test_table_interpolation_and_tail():
    """Inside the table an 8-point Lagrange polynomial; past it a quintic joining smoothly to zero."""
    rows = np.exp(-0.7 * GRID * np.arange(1, 101))[:, None] * np.array([1.0, -0.3])
    table = torch.as_tensor(rows)
    r = torch.tensor([0.05, 0.37, 1.0, 1.234, 1.9])
    got = D.table_values(table, GRID, r)
    for k in range(2):
        ref = [_lagrange_reference(rows[:, k], GRID, float(x)) for x in r]
        assert np.allclose(got[:, k].numpy(), ref, rtol=1e-12, atol=1e-14)
    r0 = 100 * GRID
    x = torch.tensor([r0 - 1e-9, r0 + 1e-9, r0 + 1.0 - 1e-9, r0 + 1.0 + 1e-9, r0 + 3.0], requires_grad=True)
    v = D.table_values(table, GRID, x)[:, 0]
    (d1,) = torch.autograd.grad(v.sum(), x, create_graph=True)
    (d2,) = torch.autograd.grad(d1.sum(), x)
    assert torch.allclose(v[0], v[1], atol=1e-12) and torch.allclose(d1[0], d1[1], atol=1e-9)
    assert torch.allclose(d2[0], d2[1], rtol=1e-6)
    assert float(v[2].abs()) < 1e-15 and float(v[3]) == 0.0 and float(v[4]) == 0.0


def test_tail_join():
    """The tail joins the last row and central differences (step 1e-5 bohr) of the interpolant there."""
    rows = np.exp(-0.7 * GRID * np.arange(1, 101))[:, None] * np.array([1.0, -0.3])
    ends = D.tail_join(rows, GRID)
    exact = D._end_weights() @ rows[-8:] * np.array([1.0, 1.0 / GRID, 1.0 / GRID ** 2])[:, None]
    assert np.array_equal(ends[0], rows[-1])
    assert np.allclose(ends[1], exact[1], rtol=1e-8) and np.allclose(ends[2], exact[2], rtol=1e-4)
    corr = torch.as_tensor(D.tail_correction(rows, GRID))
    r0 = 100 * GRID
    v = D.table_values(torch.as_tensor(rows), GRID, torch.tensor([r0 - 1e-9, r0 + 1e-9, r0 + 0.5]), ends=corr)
    assert torch.allclose(v[0], v[1], atol=1e-12)
    s = 0.5
    f0, f1, f2 = ends
    quintic = (f0 + s * f1 + s * s * f2 / 2 + s ** 3 * (-10 * f0 - 6 * f1 - 1.5 * f2)
               + s ** 4 * (15 * f0 + 8 * f1 + 1.5 * f2) + s ** 5 * (-6 * f0 - 3 * f1 - 0.5 * f2))
    assert np.allclose(v[2].numpy(), quintic, rtol=1e-13)
    # inside the table the tableau gives the polynomial through the eight rows around the point
    for x in (0.731, 1.2345, r0 - 1e-5):
        ref = [_lagrange_reference(rows[:, k], GRID, x) for k in range(2)]
        assert np.allclose(D._point_value(rows, GRID, x), ref, rtol=1e-12, atol=1e-15)


def test_short_range_cutoff(skset):
    """Short-range gamma is dropped from the cutoff of its pair on, found by DFTB+'s bisection."""
    cutoffs = {(0.4954, 0.4954): 17.00625, (0.4195, 0.4954): 18.515625, (0.4309, 0.4954): 19.01875,
               (0.135642, 0.135642): 58.08125, (0.09, 0.09): 86.034375, (0.06, 0.06): 128.1}
    for (ua, ub), rc in cutoffs.items():
        assert abs(D.short_range_cutoff(3.2 * ua, 3.2 * ub) - rc) < 1e-9
    model = _model(DFTB2, skset)
    u = torch.tensor([0.4195, 0.4195], dtype=torch.float64)
    rc = D.short_range_cutoff(3.2 * 0.4195, 3.2 * 0.4195)
    pos = torch.tensor([[0.0, 0.0, 0.0], [rc - 1e-6, 0.0, 0.0], [rc + 1e-6, 0.0, 0.0]], dtype=torch.float64)
    short, *_ = model._gamma_parts(u, u, torch.tensor([0, 0]), torch.tensor([1, 2]), pos,
                                   torch.zeros(3, dtype=torch.bool))
    assert float(short[0]) > 1e-11 and float(short[1]) == 0.0


def test_rotation_polynomials():
    """The bond-frame products are those of Slater and Koster and sum to the identity over |m|."""
    rng = np.random.default_rng(3)
    u = torch.as_tensor(rng.normal(size=(5, 3)))
    u = u / u.norm(dim=-1, keepdim=True)
    sp = shell_block(u, torch.ones(5, 1), 0, 1)
    assert torch.allclose(sp[:, 0], u[:, [1, 2, 0]], atol=1e-13)
    vs, vp = 0.7, -0.2
    pp = shell_block(u, torch.tensor([[vs, vp]]).expand(5, 2), 1, 1)
    up = u[:, [1, 2, 0]]
    ref = vp * torch.eye(3) + (vs - vp) * up[:, :, None] * up[:, None, :]
    assert torch.allclose(pp, ref, atol=1e-13)
    for l in range(4):
        total = shell_block(u, torch.ones(5, l + 1), l, l)
        assert torch.allclose(total, torch.eye(2 * l + 1).expand_as(total), atol=1e-12)
        coef = rotation_polynomials(l, l)
        along = np.einsum("m,mkab->kab", monomials(torch.tensor([0.0, 0.0, 1.0]), 2 * l).numpy(), coef)
        for k in range(l + 1):
            diag = np.zeros(2 * l + 1)
            diag[[l - k, l + k]] = 1.0
            assert np.allclose(along[k], np.diag(diag), atol=1e-12)
    with pytest.raises(ValueError):
        real_harmonics(4, np.zeros((1, 3)))
    assert monomials(u, 0).shape == (5, 1)


def test_gamma_limits():
    """Gamma tends to U on an atom and to 1 / R far away; unequal widths tend to the equal form."""
    ta, tb = torch.tensor([1.3]), torch.tensor([1.3 * (1.0 + 1e-3)])
    r = torch.tensor([2.0])
    mid = D.gamma_short(0.5 * (ta + tb), 0.5 * (ta + tb), r)
    assert torch.allclose(D.gamma_short(ta, tb, r), mid, rtol=1e-5)
    small = torch.tensor([1e-4])
    assert abs(float(1.0 / small - D.gamma_short(ta, ta, small)) - 5.0 * 1.3 / 16.0) < 1e-6
    assert float(D.gamma_short(ta, 1.7 * ta, torch.tensor([40.0])).abs()) < 1e-15
    assert torch.allclose(D.gamma_onsite(ta, ta), 5.0 * ta / 16.0)
    t = torch.tensor([1.3], dtype=torch.float64)
    close = D.gamma_short(t, t + 0.5 * D.EQUAL_WIDTHS, r.double())
    mean = 0.5 * (t + (t + 0.5 * D.EQUAL_WIDTHS))
    assert torch.allclose(close, D.gamma_short(mean, mean, r.double()), rtol=1e-15, atol=0.0)
    for a, b in ((1.3, 1.7), (1.3, 1.3)):
        ta = torch.tensor([a], dtype=torch.float64, requires_grad=True)
        tb = torch.tensor([b], dtype=torch.float64, requires_grad=True)
        (ga,) = torch.autograd.grad(D.gamma_short(ta, tb, r.double()).sum(), ta)
        assert torch.allclose(D.gamma_short_slope(ta, tb, r.double()), ga, rtol=1e-12)


@pytest.mark.parametrize("cls", [DFTB1, DFTB2, DFTB3])
def test_energy_invariance_and_forces(cls, skset):
    """Energies are invariant under rotations and translations; forces are their negative gradient."""
    model = _model(cls, skset)
    symbols, pos = THIOFORMALDEHYDE
    wrapped = ForceStressOutput(model)
    out = wrapped(collate([_graph(symbols, pos)]))
    assert bool(out["scf_converged"])
    q, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))
    moved = model(collate([_graph(symbols, pos @ q.T + 0.3)]))["energy"]
    assert torch.allclose(out["energy"], moved, rtol=0.0, atol=1e-11)
    h = 1e-5
    fd = np.zeros_like(pos)
    for a in range(len(symbols)):
        for c in range(3):
            e = []
            for s in (1, -1):
                p = pos.copy()
                p[a, c] += s * h
                e.append(float(model(collate([_graph(symbols, p)]))["energy"][0]))
            fd[a, c] = -(e[0] - e[1]) / (2 * h)
    assert np.allclose(out["forces"].detach().numpy(), fd, atol=1e-7)


@pytest.mark.parametrize("cls", [DFTB2, DFTB3])
def test_exact_hessian(cls, skset):
    """The Hessian of the self-consistent energy (a closed shell) equals central differences of the forces."""
    model = _model(cls, skset)
    symbols, pos = HS
    (H,) = hessian(model, _graph(symbols, pos, charge=-1.0))
    wrapped = ForceStressOutput(model).eval()
    n, step = 3 * len(symbols), 1e-4
    fd = np.zeros((n, n))
    for k in range(n):
        cols = []
        for s in (1, -1):
            p = pos.copy()
            p.reshape(-1)[k] += s * step
            cols.append(wrapped(collate([_graph(symbols, p, charge=-1.0)]))["forces"].detach().numpy().reshape(-1))
        fd[:, k] = -(cols[0] - cols[1]) / (2 * step)
    assert torch.allclose(H, H.T, atol=1e-9)
    assert np.allclose(H.numpy(), fd, atol=1e-6)


def test_batches_charges_and_options(skset):
    """A batch equals its structures; charged and odd-electron molecules converge; options take effect."""
    model = _model(DFTB3, skset)
    graphs = [_graph(*THIOFORMALDEHYDE), _graph(*METHANE), _graph(*HS, charge=-1.0)]
    batch = model(collate(graphs))
    alone = [float(model(collate([g]))["energy"][0]) for g in graphs]
    assert np.allclose(batch["energy"].detach().numpy(), alone, rtol=0.0, atol=1e-10)
    assert abs(float(batch["charges"][-2:].sum()) + 1.0) < 1e-10
    radical = model(collate([_graph(["C", "H", "H", "H"], METHANE[1][:4])]))
    assert bool(radical["scf_converged"])
    warm = _model(DFTB2, skset, electronic_temperature=3000.0)
    cold = _model(DFTB2, skset)
    g = collate([_graph(*THIOFORMALDEHYDE)])
    assert float((warm(g)["energy"] - cold(g)["energy"]).abs()) > 0.0
    shell = _model(DFTB2, skset, shell_resolved=True)
    assert float((shell(g)["energy"] - cold(g)["energy"]).abs()) > 1e-6
    onsite = _model(DFTB, skset, third_order="onsite", hubbard_derivatives=DERIVS)
    assert float(onsite(g)["third_order_energy"].abs()) > 0.0
    with pytest.raises(NotImplementedError):
        _model(DFTB3, skset, shell_resolved=True)(g)
    with pytest.raises(ValueError):
        DFTB(str(skset), third_order="half")
    with pytest.raises(ValueError):
        DFTB(str(skset), scc=False, third_order="full")
    one = _model(DFTB1, skset)
    assert float(one(g)["second_order_energy"].abs()) == 0.0 and one.last_scf["iterations"] == 1
    assert _model(DFTB2, skset, max_angular_momentum={1: 0, 6: 1, 16: 1}).lmax[16] == 1
    assert DFTB(str(skset)).lmax == {1: 0, 6: 1, 16: 2}


def test_from_config(skset):
    """A model config builds the model with its options."""
    from xnn.common.config.schema import ModelConfig
    cfg = ModelConfig(name="dftb3", extra={"parameters": str(skset), "hubbard_derivatives": DERIVS,
                                           "damping_exponent": 4.0, "scc_tolerance": 1e-9})
    model = DFTB3.from_config(cfg)
    assert model.scc_tolerance == 1e-9 and model.damping_exponent == 4.0 and model.third_order == "full"


def test_fidelity_vs_dftbplus():
    """Energies, charges and forces equal those of the DFTB+ program (when it is available)."""
    binary = os.environ.get("XNN_DFTB")
    if not binary:
        pytest.skip("XNN_DFTB is not set")
    from sqmnn_dftb_fidelity import compare, compare_periodic
    try:
        results = compare(Path(binary))
        periodic = compare_periodic(Path(binary))
    except FileNotFoundError as exc:
        pytest.skip(f"Slater-Koster sets not available: {exc}")
    for row in results:
        assert row["converged"], row
        assert abs(row["energy"]) < 5e-13 * max(1.0, abs(row["reference"])), row
        assert row["charges"] < 1e-9, row
        assert row["forces"] < 1e-6, row
    for row in periodic:
        assert row["converged"] and abs(row["energy"]) < 5e-13 * abs(row["reference"]), row
        assert row["charges"] < 1e-7 and row["forces"] < 1e-6 and row["stress"] < 1e-9, row


def test_dispersion_polynomial_repulsion_and_no_grad(skset, tmp_path):
    """The D3 term adds to the energy; files without a spline use their polynomial; no-grad runs agree."""
    g = collate([_graph(*THIOFORMALDEHYDE)])
    plain = _model(DFTB3, skset)
    disp = _model(DFTB3, skset, dispersion={"model": "d3", "damping": "bj", "s8": 3.209, "a1": 0.746, "a2": 4.191})
    e0, out = plain(g)["energy"], disp(g)
    assert float(out["dispersion_energy"]) < 0.0
    assert torch.allclose(out["energy"] - e0, out["dispersion_energy"], atol=1e-9)
    with pytest.raises(ValueError):
        _model(DFTB2, skset, dispersion={"model": "d5"})
    with torch.no_grad():
        assert torch.allclose(plain(g)["energy"], e0, rtol=0.0, atol=1e-10)
    free = _model(DFTB, skset, third_order="full", hubbard_derivatives=DERIVS)
    assert bool(free(g)["scf_converged"]) and free.damping_exponent is None
    folder = tmp_path / "poly"
    folder.mkdir()
    for p in skset.glob("*.skf"):
        text = p.read_text()
        if p.name == "H-H.skf":
            head, _, _ = text.partition("Spline")
            lines = head.splitlines()
            lines[2] = "1.008, 0.5, 7*0.0, 3.0, 10*0.0"
            text = "\n".join(lines) + "\n"
        (folder / p.name).write_text(text)
    model = _model(DFTB2, folder)
    r = torch.tensor([1.0, 2.5, 3.5])
    assert torch.allclose(model.repulsive(1, 1, r), torch.tensor([2.0, 0.125, 0.0]))


def test_h5_correction(skset):
    """H5 scales gamma of H and N, O, S by its Gaussian; it excludes the damping and needs vdW radii."""
    model = _model(DFTB2, skset, h5=True)
    r0 = 0.714 * (1.80 + 1.20) / model.bohr
    f = model._h5_factor(torch.tensor([0, 0]), torch.tensor([1, 2]), torch.tensor([r0, r0]), [16, 1, 6])
    assert torch.allclose(f, torch.tensor([1.21, 1.0]))
    g = collate([_graph(*HS)])
    assert float((model(g)["energy"] - _model(DFTB2, skset)(g)["energy"]).abs()) > 1e-8
    custom = _model(DFTB2, skset, h5={"scaling": {"S": -1.0, "C": 0.1}, "r_scaling": 0.7})
    assert custom.h5["scaling"] == {7: 0.18, 8: 0.06, 6: 0.1} and custom.h5["r_scaling"] == 0.7
    third = _model(DFTB3, skset, h5=True, damping_exponent=None)
    assert third.damping_exponent is None and bool(third(g)["scf_converged"])
    with pytest.raises(ValueError):
        _model(DFTB2, skset, h5=True, damping_exponent=4.0)
    with pytest.raises(ValueError):
        _model(DFTB2, skset, h5={"scaling": {"Fe": 0.1}})


SPIN_W = {"H": [-0.07], "C": [-0.023], "S": [-0.02]}
SPIN_SHELL_W = {"H": [-0.07], "C": [-0.03, -0.025, -0.025, -0.023],
                "S": [-0.021, -0.017, -0.017, -0.017, -0.015, -0.015, -0.017, -0.015, -0.015]}


def _radical(symbols, pos, mult):
    from xnn.common.data.elements import CHEMICAL_SYMBOLS
    z = np.array([CHEMICAL_SYMBOLS.index(s) for s in symbols])
    return structure_to_graph({"pos": torch.as_tensor(np.asarray(pos, dtype=np.float64)), "atomic_numbers": z,
                               "total_charge": 0.0, "spin_multiplicity": float(mult)}, 0.1)


@pytest.mark.parametrize("table", [SPIN_W, SPIN_SHELL_W])
def test_spin_polarization(skset, table):
    """Collinear spin: the unpaired electron magnetizes the radical and lowers the energy; batches agree."""
    methyl = _radical(["C", "H", "H", "H"], METHANE[1][:4], 2)
    spin = _model(DFTB3, skset, spin_constants=table)
    out = ForceStressOutput(spin)(collate([methyl]))
    assert bool(out["scf_converged"]) and abs(float(out["magnetizations"].sum()) - 1.0) < 1e-10
    assert float(out["spin_energy"]) < 0.0
    plain = _model(DFTB3, skset)(collate([methyl]))
    assert float(out["energy"]) < float(plain["energy"])
    pair = spin(collate([methyl, _radical(*HS, 2)]))
    alone = float(spin(collate([_radical(*HS, 2)]))["energy"][0])
    assert abs(float(pair["energy"][1]) - alone) < 1e-10
    warm = _model(DFTB2, skset, spin_constants=table, electronic_temperature=2000.0)
    assert bool(warm(collate([methyl]))["scf_converged"])
    singlet = spin(collate([_graph(*THIOFORMALDEHYDE)]))
    assert float(singlet["magnetizations"].abs().max()) < 1e-8


def test_spin_hessian_and_errors(skset):
    """The spin-polarized Hessian equals differences of the forces; bad spin options are refused."""
    model = _model(DFTB2, skset, spin_constants=SPIN_W)
    symbols, pos = HS
    (H,) = hessian(model, _radical(symbols, pos, 2))
    wrapped = ForceStressOutput(model).eval()
    n, step = 3 * len(symbols), 1e-4
    fd = np.zeros((n, n))
    for k in range(n):
        cols = []
        for s in (1, -1):
            p = pos.copy()
            p.reshape(-1)[k] += s * step
            cols.append(wrapped(collate([_radical(symbols, p, 2)]))["forces"].detach().numpy().reshape(-1))
        fd[:, k] = -(cols[0] - cols[1]) / (2 * step)
    assert np.allclose(H.numpy(), fd, atol=1e-6)
    with pytest.raises(ValueError):
        _model(DFTB2, skset, spin_constants={"C": [0.1, 0.2]})
    with pytest.raises(ValueError):
        DFTB(str(skset), scc=False, spin_constants=SPIN_W)
    with pytest.raises(ValueError):
        _model(DFTB2, skset, spin_constants={"H": [-0.07]})(collate([_radical(*HS, 2)]))


def test_format_round_trip(skset):
    """A written file reads back to the same values."""
    for name, homo in (("S-S.skf", True), ("C-H.skf", False)):
        f = skf.parse_skf((skset / name).read_text(), homo)
        g = skf.parse_skf(skf.format_skf(f), homo)
        assert np.array_equal(f.hamiltonian, g.hamiltonian) and np.array_equal(f.overlap, g.overlap)
        assert f.energies == g.energies and f.hubbard == g.hubbard and f.mass == g.mass
        assert np.array_equal(f.spline.coefficients, g.spline.coefficients) and f.spline.head == g.spline.head
    ext = skf.parse_skf("@ x\n0.1 9 4\n" + "0.0 " * 4 + "0.0 " + "0.5 " * 4 + "0.0 0.0 2.0 2.0\n1.0 19*0.0\n"
                        + "\n".join(" ".join(["1.0"] * 40) for _ in range(9)) + "\n", True)
    assert skf.parse_skf(skf.format_skf(ext), True).extended


def _energy(model, graph):
    return float(model(graph)["energy"].sum())


def test_trainable_parameter_gradients(skset):
    """Gradients of the energy in the trained tables, splines, levels and Hubbard parameters match differences."""
    model = _model(DFTB3, skset, trainable=True).train()
    g = collate([_graph(*THIOFORMALDEHYDE)])
    energy = model(g)["energy"].sum()
    names = [("hubbard", (6, 0)), ("onsite", (16, 2)), ("H_6_16", (60, 9)), ("S_1_6", (70, 9)), ("spline_1_6", (0, 1))]
    grads = torch.autograd.grad(energy, [model.params[n] for n, _ in names])
    model.eval()
    for (name, idx), grad in zip(names, grads):
        p = model.params[name]
        h = 1e-5
        with torch.no_grad():
            p[idx] += h
        up = _energy(model, g)
        with torch.no_grad():
            p[idx] -= 2 * h
        down = _energy(model, g)
        with torch.no_grad():
            p[idx] += h
        assert abs(float(grad[idx]) - (up - down) / (2 * h)) < 1e-6 * max(1.0, abs(float(grad[idx]))), name
    with pytest.raises(ValueError):
        _model(DFTB2, skset, trainable=["levels"])


def test_force_loss_gradient(skset):
    """The gradient of a force loss in a trained parameter matches finite differences of the loss."""
    model = _model(DFTB2, skset, trainable=("hubbard", "repulsive")).train()
    g = collate([_graph(*THIOFORMALDEHYDE)])
    wrapped = ForceStressOutput(model)
    loss = (wrapped(g)["forces"] ** 2).sum()
    (grad,) = torch.autograd.grad(loss, model.params["hubbard"])
    h = 1e-5

    def value():
        return float((wrapped(collate([_graph(*THIOFORMALDEHYDE)]))["forces"] ** 2).sum())

    for idx in ((6, 0), (16, 0)):
        p = model.params["hubbard"]
        with torch.no_grad():
            p[idx] += h
        up = value()
        with torch.no_grad():
            p[idx] -= 2 * h
        down = value()
        with torch.no_grad():
            p[idx] += h
        assert abs(float(grad[idx]) - (up - down) / (2 * h)) < 1e-5 * max(1.0, abs(float(grad[idx])))


def test_write_trained_set(skset, tmp_path):
    """A model read from the written set has the trained model's energy."""
    model = _model(DFTB3, skset, trainable=True)
    with torch.no_grad():
        model.params["H_6_16"][40:60] *= 1.01
        model.params["hubbard"][16, 0] += 0.01
        model.params["onsite"][6, 1] -= 0.002
        model.params["spline_1_6"][0, 0] += 0.001
    model.write_set(tmp_path / "trained")
    again = _model(DFTB3, tmp_path / "trained")
    g = collate([_graph(*THIOFORMALDEHYDE)])
    assert abs(_energy(model.eval(), g) - _energy(again, g)) < 1e-10


def test_parse_flags_comments_and_range_separation(skset):
    """Logical flags after numbers, comments and extra blocks after the table, and RangeSep blocks are read."""
    lines = (skset / "S-S.skf").read_text().splitlines()
    lines[1] += "  T  2  0.0 0.0"
    lines[2] += " T"
    at = lines.index("Spline")
    lines[at:at] = ["# a comment", "Potential in Slaters # fitting data", "16 2 1"]
    lines.insert(lines.index("<Documentation>"), "RangeSep")
    lines.insert(lines.index("<Documentation>"), "LC 0.300000")
    f = skf.parse_skf("\n".join(lines) + "\n", True)
    assert f.occupations == OCCUPATIONS["S"] and f.hamiltonian.shape == (NGRID, 10)
    assert f.spline is not None and f.range_separation == ("LC", 0.3)


def test_long_range_gamma():
    """The Yukawa gamma: equal widths continue the unequal form, it vanishes at zero screening length and
    reduces to the Coulomb gamma without screening."""
    ta, r = torch.tensor([1.3]), torch.tensor([2.2])
    near = D.gamma_yukawa(ta, ta * (1.0 + 1e-4), r, 0.3)
    assert torch.allclose(near, D.gamma_yukawa(ta, ta, r, 0.3), rtol=1e-4)
    coulomb = 1.0 / r - D.gamma_short(ta, 1.7 * ta, r)
    assert torch.allclose(D.gamma_yukawa(ta, 1.7 * ta, r, 1e-7), coulomb, rtol=1e-6)
    assert torch.allclose(D.gamma_yukawa(ta, ta, r, 1e-7), 1.0 / r - D.gamma_short(ta, ta, r), rtol=1e-6)
    assert torch.allclose(D.gamma_yukawa_onsite(ta, 1e-7), 5.0 * ta / 16.0, rtol=1e-6)
    assert float(D.gamma_yukawa(ta, ta, torch.tensor([60.0]), 0.3)) < 1e-8


def test_long_range_exchange(skset):
    """LC-DFTB: the exchange term enters the energy; forces and the Hessian are exact; bad combinations fail."""
    model = _model(DFTB2, skset, range_separation=0.3)
    assert model.density_mixing and model.omega == 0.3
    g = collate([_graph(*THIOFORMALDEHYDE)])
    out = ForceStressOutput(model)(g)
    assert bool(out["scf_converged"]) and float(out["exchange_energy"].abs()) > 1e-6
    symbols, pos = THIOFORMALDEHYDE
    h = 1e-5
    for a, c in ((1, 2), (2, 0)):
        e = []
        for s in (1, -1):
            p = pos.copy()
            p[a, c] += s * h
            e.append(float(model(collate([_graph(symbols, p)]))["energy"][0]))
        assert abs(float(out["forces"][a, c]) + (e[0] - e[1]) / (2 * h)) < 1e-7
    (H,) = hessian(model, _graph(*HS, charge=-1.0))
    wrapped = ForceStressOutput(model).eval()
    k, step = 5, 1e-4
    cols = []
    for s in (1, -1):
        p = HS[1].copy()
        p.reshape(-1)[k] += s * step
        cols.append(wrapped(collate([_graph(HS[0], p, charge=-1.0)]))["forces"].detach().numpy().reshape(-1))
    assert np.allclose(H[:, k].numpy(), -(cols[0] - cols[1]) / (2 * step), atol=1e-6)
    assert not _model(DFTB2, skset).density_mixing and not _model(DFTB2, skset, range_separation=False).density_mixing
    with pytest.raises(ValueError):
        _model(DFTB2, skset, range_separation=0.3, spin_constants=SPIN_W)


def test_ase_calculator(skset):
    """The model runs as an ASE calculator: energies and forces equal the direct evaluation."""
    from ase import Atoms

    from xnn.common.deploy import XNNCalculator

    model = _model(DFTB2, skset)
    symbols, pos = THIOFORMALDEHYDE
    atoms = Atoms(symbols=symbols, positions=pos)
    atoms.calc = XNNCalculator(ForceStressOutput(model).eval(), cutoff=model.cutoff)
    out = ForceStressOutput(model)(collate([_graph(symbols, pos)]))
    assert abs(atoms.get_potential_energy() - float(out["energy"])) < 1e-8
    assert np.allclose(atoms.get_forces(), out["forces"].detach().numpy(), atol=1e-8)


ONSITE = {"H": {"uu": [0.0], "ud": [0.0]},
          "C": {"uu": [0.0, 0.05, 0.05, -0.012], "ud": [0.0, 0.105, 0.105, 0.026]},
          "S": {"uu": [0.0, 0.075, 0.004, 0.075, 0.003, 0.011, 0.004, 0.011, -0.018],
                "ud": [0.0, 0.117, 0.039, 0.117, 0.031, 0.050, 0.039, 0.050, 0.016]}}


def test_onsite_correction(skset):
    """The onsite correction raises the energy of polarized atoms; forces and the Hessian stay exact."""
    # smearing: the synthetic set gives this molecule nearly degenerate frontier levels
    model = _model(DFTB2, skset, onsite_correction=ONSITE, electronic_temperature=1000.0)
    plain = _model(DFTB2, skset, electronic_temperature=1000.0)
    g = collate([_graph(*THIOFORMALDEHYDE)])
    out = ForceStressOutput(model)(g)
    assert bool(out["scf_converged"]) and float(out["onsite_energy"]) > 0.0
    assert float((out["energy"] - plain(g)["energy"]).abs()) > 1e-4
    symbols, pos = THIOFORMALDEHYDE
    h = 1e-5
    for a, c in ((1, 2), (3, 1)):
        e = []
        for s in (1, -1):
            p = pos.copy()
            p[a, c] += s * h
            e.append(float(model(collate([_graph(symbols, p)]))["energy"][0]))
        assert abs(float(out["forces"][a, c]) + (e[0] - e[1]) / (2 * h)) < 1e-7
    (H,) = hessian(model, _graph(*HS, charge=-1.0))
    wrapped = ForceStressOutput(model).eval()
    k, step = 4, 1e-4
    cols = []
    for s in (1, -1):
        p = HS[1].copy()
        p.reshape(-1)[k] += s * step
        cols.append(wrapped(collate([_graph(HS[0], p, charge=-1.0)]))["forces"].detach().numpy().reshape(-1))
    assert np.allclose(H[:, k].numpy(), -(cols[0] - cols[1]) / (2 * step), atol=1e-6)
    both = _model(DFTB2, skset, onsite_correction=ONSITE, range_separation=0.3, electronic_temperature=1000.0)
    assert bool(both(g)["scf_converged"]) and float(both(g)["exchange_energy"]) != 0.0
    with pytest.raises(ValueError):
        _model(DFTB2, skset, onsite_correction={"C": {"uu": [0.0, 0.1], "ud": [0.0, 0.1]}})
    with pytest.raises(ValueError):
        _model(DFTB2, skset, onsite_correction=ONSITE, spin_constants=SPIN_W)


@pytest.mark.parametrize("functional", ["fll", "psic"])
def test_orbital_potential(skset, functional):
    """DFTB+U on spin-polarized runs: FLL raises the energy of fractional occupations, pSIC lowers it; forces and
    the Hessian are exact; it needs spin polarization."""
    op = {"functional": functional, "shells": {"C": [{"shells": [1], "uj": 0.2}], "S": [{"shells": [1, 2], "uj": 0.1}]}}
    model = _model(DFTB2, skset, spin_constants=SPIN_W, orbital_potential=op)
    plain = _model(DFTB2, skset, spin_constants=SPIN_W)
    g = collate([_radical(*THIOFORMALDEHYDE, 1)])
    out = ForceStressOutput(model)(g)
    assert bool(out["scf_converged"])
    assert (float(out["orbital_energy"]) > 0.0) == (functional == "fll")
    assert float((out["energy"] - plain(g)["energy"]).abs()) > 1e-4
    symbols, pos = THIOFORMALDEHYDE
    h = 1e-5
    e = []
    for s in (1, -1):
        p = pos.copy()
        p[1, 2] += s * h
        e.append(float(model(collate([_radical(symbols, p, 1)]))["energy"][0]))
    assert abs(float(out["forces"][1, 2]) + (e[0] - e[1]) / (2 * h)) < 1e-7
    (H,) = hessian(model, _radical(*HS, 2))
    wrapped = ForceStressOutput(model).eval()
    cols = []
    for s in (1, -1):
        p = HS[1].copy()
        p.reshape(-1)[5] += s * 1e-4
        cols.append(wrapped(collate([_radical(HS[0], p, 2)]))["forces"].detach().numpy().reshape(-1))
    assert np.allclose(H[:, 5].numpy(), -(cols[0] - cols[1]) / 2e-4, atol=1e-6)
    with pytest.raises(ValueError):
        _model(DFTB2, skset, orbital_potential=op)
    with pytest.raises(ValueError):
        _model(DFTB2, skset, spin_constants=SPIN_W, orbital_potential={"functional": "amf"})
    with pytest.raises(ValueError):
        _model(DFTB2, skset, spin_constants=SPIN_W, range_separation=0.3)


DIAMOND = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0], [.25, .25, .25], [.25, .75, .75],
                    [.75, .25, .75], [.75, .75, .25]])


def _crystal(symbols, pos, cell, charge=0.0):
    from xnn.common.data.elements import CHEMICAL_SYMBOLS
    z = np.array([CHEMICAL_SYMBOLS.index(s) for s in symbols])
    return structure_to_graph({"pos": torch.as_tensor(np.asarray(pos, dtype=np.float64)), "atomic_numbers": z,
                               "total_charge": charge, "cell": torch.as_tensor(np.asarray(cell, dtype=np.float64)),
                               "pbc": torch.ones(3, dtype=torch.bool)}, 0.1)


@pytest.mark.parametrize("cls", [DFTB2, DFTB3])
def test_periodic_gamma_point(skset, cls):
    """Periodic structures at the Gamma point: invariant under rotations and lattice translations; forces and stress
    are derivatives of the energy; a batch mixes crystals and molecules."""
    model = _model(cls, skset)
    rng = np.random.default_rng(5)
    a = 5.6
    cell = np.eye(3) * a + 0.1 * rng.normal(size=(3, 3))
    symbols = ["C", "S", "C", "S", "H", "H", "H", "H"]
    pos = DIAMOND @ cell + 0.1 * rng.normal(size=(8, 3))
    wrapped = ForceStressOutput(model, compute_stress=True)
    out = wrapped(collate([_crystal(symbols, pos, cell)]))
    assert bool(out["scf_converged"])
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    shifted = pos.copy()
    shifted[2] += cell[1]
    for p, c in ((pos @ q.T, cell @ q.T), (shifted, cell)):
        assert abs(float(model(collate([_crystal(symbols, p, c)]))["energy"]) - float(out["energy"])) < 1e-9
    h = 1e-5
    e = []
    for s in (1, -1):
        p = pos.copy()
        p[1, 0] += s * h
        e.append(float(model(collate([_crystal(symbols, p, cell)]))["energy"]))
    assert abs(float(out["forces"][1, 0]) + (e[0] - e[1]) / (2 * h)) < 1e-6
    volume = abs(np.linalg.det(cell))
    e = []
    for s in (1, -1):
        strain = np.eye(3)
        strain[0, 1] = strain[1, 0] = s * h
        e.append(float(model(collate([_crystal(symbols, pos @ strain, cell @ strain)]))["energy"]))
    assert abs(float(out["stress"][0, 0, 1]) - (e[0] - e[1]) / (2 * h) / (2 * volume)) < 1e-6
    batch = model(collate([_crystal(symbols, pos, cell), _graph(*THIOFORMALDEHYDE)]))
    alone = float(model(collate([_graph(*THIOFORMALDEHYDE)]))["energy"])
    assert abs(float(batch["energy"][1]) - alone) < 1e-9 and abs(float(batch["energy"][0] - out["energy"][0])) < 1e-9


@pytest.mark.parametrize("options", [
    {"dispersion": {"model": "d3", "damping": "zero", "rs6": 1.25, "alp": 29.61, "s8": 0.49, "hh_repulsion": True}},
    {"dispersion": {"model": "d4", "s8": 0.66, "s9": 0.0, "a1": 0.55, "a2": 4.35}},
    {"shell_resolved": True}, {"range_separation": 0.3}])
def test_periodic_dispersion_and_shells(skset, options):
    """Dispersion, the hydrogen repulsion, shell-resolved charges and long-range exchange in a cell: invariant under
    lattice translations, with forces and stress the derivatives of the energy."""
    model = _model(DFTB2, skset, **options)
    rng = np.random.default_rng(7)
    cell = np.eye(3) * 5.6 + 0.1 * rng.normal(size=(3, 3))
    symbols = ["C", "S", "C", "S", "H", "H", "H", "H"]
    pos = DIAMOND @ cell + 0.1 * rng.normal(size=(8, 3))
    if "range_separation" in options:                # a molecule in a box: the exchange of a dense cell is stiff
        symbols, mol = THIOFORMALDEHYDE
        cell = np.eye(3) * 9.0 + 0.1 * rng.normal(size=(3, 3))
        pos = np.asarray(mol) + 2.0
    last = len(symbols) - 1
    out = ForceStressOutput(model, compute_stress=True)(collate([_crystal(symbols, pos, cell)]))
    assert bool(out["scf_converged"])
    shifted = pos.copy()
    shifted[last] -= cell[2]
    assert abs(float(model(collate([_crystal(symbols, shifted, cell)]))["energy"]) - float(out["energy"])) < 1e-9
    if "dispersion" in options:
        assert float(out["dispersion_energy"]) < 0.0
    h = 1e-5
    e = []
    for s in (1, -1):
        p = pos.copy()
        p[last, 2] += s * h
        e.append(float(model(collate([_crystal(symbols, p, cell)]))["energy"]))
    assert abs(float(out["forces"][last, 2]) + (e[0] - e[1]) / (2 * h)) < 1e-6
    if options.get("dispersion", {}).get("model") == "d4":
        return                                  # hard D4 cutoffs make strained differences of the energy noisy
    e = []
    for s in (1, -1):
        strain = np.eye(3)
        strain[1, 2] = strain[2, 1] = s * h
        e.append(float(model(collate([_crystal(symbols, pos @ strain, cell @ strain)]))["energy"]))
    volume = abs(np.linalg.det(cell))
    assert abs(float(out["stress"][0, 1, 2]) - (e[0] - e[1]) / (2 * h) / (2 * volume)) < 1e-6


def test_periodic_errors(skset):
    """Partly periodic cells are refused; the Ewald sum of a point charge gives the Madelung constant."""
    model = _model(DFTB2, skset)
    g = _crystal(*HS, np.eye(3) * 8.0)
    g.pbc = torch.tensor([[True, True, False]])
    with pytest.raises(NotImplementedError):
        model(collate([g]))
    dense = D.ewald_matrix(torch.zeros(1, 3), torch.eye(3) * 10.0)
    assert abs(float(dense[0, 0]) + 2.837297479 / 10.0) < 1e-8        # Madelung constant of the cubic lattice


def test_d4_and_hydrogen_repulsion(skset):
    """D4 adds a dispersion energy; the D3H5 hydrogen repulsion adds the Fermi-shaped term of every H-H pair."""
    g = collate([_graph(*METHANE)])
    d4 = _model(DFTB3, skset, dispersion={"model": "d4", "s8": 0.66, "s9": 1.0, "a1": 0.55, "a2": 4.35})
    assert float(d4(g)["dispersion_energy"]) < 0.0
    d3 = {"model": "d3", "damping": "zero", "rs6": 1.25, "alp": 29.61, "s8": 0.49}
    plain = _model(DFTB3, skset, dispersion=d3)
    hh = _model(DFTB3, skset, dispersion=dict(d3, hh_repulsion=True))
    out = hh(g)
    _, pos = METHANE
    r = [np.linalg.norm(pos[i] - pos[j]) * hh.bohr / 0.529177249 for i in range(1, 5) for j in range(i + 1, 5)]
    k, e, r0 = D.HH_REPULSION
    expected = sum(k / (1.0 + np.exp(e * (x / r0 - 1.0))) for x in r) * hh.hartree
    assert abs(float(out["hh_repulsion_energy"]) - expected) < 1e-12
    assert abs(float(out["energy"] - plain(g)["energy"]) - expected) < 1e-9


def test_kpoints(skset):
    """A Monkhorst-Pack grid of a cell equals the Gamma point of the matching supercell; forces follow."""
    rng = np.random.default_rng(7)
    cell = np.eye(3) * 5.6 + 0.1 * rng.normal(size=(3, 3))
    symbols = ["C", "S", "C", "S", "H", "H", "H", "H"]
    pos = DIAMOND @ cell + 0.1 * rng.normal(size=(8, 3))
    sampled = ForceStressOutput(_model(DFTB2, skset, kpoints=(2, 1, 1)))(collate([_crystal(symbols, pos, cell)]))
    big = np.array([2.0, 1.0, 1.0])[:, None] * cell
    super_pos = np.concatenate([pos, pos + cell[0]])
    whole = ForceStressOutput(_model(DFTB2, skset))(collate([_crystal(symbols * 2, super_pos, big)]))
    assert bool(sampled["scf_converged"]) and bool(whole["scf_converged"])
    assert abs(2.0 * float(sampled["energy"]) - float(whole["energy"])) < 1e-8
    assert torch.allclose(sampled["forces"], whole["forces"][:8], atol=1e-8)
    assert torch.allclose(sampled["charges"], whole["charges"][:8], atol=1e-9)
    single = _model(DFTB2, skset, kpoints={"grid": [1, 1, 1], "shift": [0.0, 0.0, 0.0]})
    plain = _model(DFTB2, skset)
    g = collate([_crystal(symbols, pos, cell)])
    assert abs(float(single(g)["energy"]) - float(plain(g)["energy"])) < 1e-12
    with pytest.raises(NotImplementedError):
        _model(DFTB2, skset, kpoints=(2, 1, 1), range_separation=0.3)(g)
    with pytest.raises(NotImplementedError):
        hessian(_model(DFTB2, skset, kpoints=(2, 1, 1)), _crystal(symbols, pos, cell))
    with pytest.raises(ValueError):
        _model(DFTB2, skset, kpoints=(2, 0, 1))


def test_kpoints_mixed_batch(skset):
    """A batch of a molecule and a crystal on a k-point grid equals its structures."""
    rng = np.random.default_rng(7)
    cell = np.eye(3) * 5.6 + 0.1 * rng.normal(size=(3, 3))
    symbols = ["C", "S", "C", "S", "H", "H", "H", "H"]
    pos = DIAMOND @ cell + 0.1 * rng.normal(size=(8, 3))
    model = _model(DFTB2, skset, kpoints=(2, 1, 1))
    graphs = [_graph(*THIOFORMALDEHYDE), _crystal(symbols, pos, cell)]
    batch = model(collate(graphs))
    alone = [float(model(collate([g]))["energy"][0]) for g in graphs]
    assert np.allclose(batch["energy"].detach().numpy(), alone, rtol=0.0, atol=1e-9)


@pytest.mark.parametrize("cls, options", [
    (DFTB1, {}), (DFTB2, {}), (DFTB3, {}), (DFTB2, {"shell_resolved": True}),
    (DFTB2, {"h5": True, "dispersion": {"model": "d3", "damping": "zero", "rs6": 1.25, "alp": 29.61, "s8": 0.49,
                                         "hh_repulsion": True}}),
    (DFTB3, {"damping_exponent": None, "dispersion": {"model": "d4", "s8": 0.66, "s9": 1.0, "a1": 0.55, "a2": 4.35}})])
def test_torchscript_export_matches_eager(skset, tmp_path, cls, options):
    """The exported TorchScript potential and its core called as plain Python give the eager model's energy,
    forces and charges, for neutral molecules and an anion."""
    from xnn.common.data.elements import CHEMICAL_SYMBOLS
    from xnn.common.deploy import export_torchscript_potential
    from xnn.common.deploy.torchscript import build_neighbor_list_ts
    model = _model(cls, skset, **options)
    eager = ForceStressOutput(model).eval()
    core = model.export_core()
    zero, no = torch.zeros(3, 3, dtype=torch.float64), torch.zeros(3, dtype=torch.bool)
    empty = torch.zeros(2, 0, dtype=torch.long)
    for (symbols, pos), charge in ((THIOFORMALDEHYDE, 0.0), (METHANE, 0.0), (HS, -1.0)):
        z = torch.tensor([CHEMICAL_SYMBOLS.index(s) for s in symbols])
        ref = eager(collate([_graph(symbols, pos, charge=charge)]))
        p = torch.as_tensor(np.asarray(pos, dtype=np.float64))
        _, node, q = core.node_features_energy_charges(z, empty, p[:0], p, zero, no, charge, 0.0)
        assert float(node.sum()) == pytest.approx(float(ref["energy"][0]), abs=1e-9)
        torch.testing.assert_close(q, ref["charges"], rtol=0.0, atol=1e-9)
        path = export_torchscript_potential(model, cutoff=10.0, path=str(tmp_path / "m.pt"), total_charge=charge)
        scripted = torch.jit.load(path)
        with torch.jit.optimized_execution(False):
            out = scripted(p, z)
            assert float(out["energy"]) == pytest.approx(float(ref["energy"][0]), abs=1e-9)
            torch.testing.assert_close(out["forces"], ref["forces"], rtol=0.0, atol=1e-8)
            torch.testing.assert_close(out["charges"], ref["charges"], rtol=0.0, atol=1e-9)
    with torch.jit.optimized_execution(False):
        with pytest.raises(Exception, match="periodic"):
            scripted(p, z, torch.eye(3, dtype=torch.float64) * 20.0, torch.tensor([True, True, False]))
        ei, cs = build_neighbor_list_ts(p, 10.0, zero, no)
        pair = scripted.forward_lammps(p, ei, cs, z, zero)
    assert float(pair["energy"]) == pytest.approx(float(out["energy"]), abs=1e-9)




@pytest.mark.parametrize("cls, options", [(DFTB3, {}), (DFTB2, {"h5": True}), (DFTB2, {"shell_resolved": True}),
                                          (DFTB2, {"dispersion": {"model": "d3", "damping": "bj", "s8": 0.5,
                                                                  "a1": 0.5, "a2": 4.0, "hh_repulsion": True}})])
def test_torchscript_export_periodic(skset, tmp_path, cls, options):
    """A cell exported to TorchScript gives the eager model's energy, forces, stress and charges at the Gamma
    point."""
    from xnn.common.data.elements import CHEMICAL_SYMBOLS
    from xnn.common.deploy import export_torchscript_potential
    rng = np.random.default_rng(5)
    cell = np.eye(3) * 5.6 + 0.1 * rng.normal(size=(3, 3))
    symbols = ["C", "S", "C", "S", "H", "H", "H", "H"]
    pos = DIAMOND @ cell + 0.1 * rng.normal(size=(8, 3))
    model = _model(cls, skset, **options)
    ref = ForceStressOutput(model, compute_stress=True).eval()(collate([_crystal(symbols, pos, cell)]))
    z = torch.tensor([CHEMICAL_SYMBOLS.index(s) for s in symbols])
    p, c = torch.as_tensor(pos), torch.as_tensor(cell)
    _, node, q = model.export_core().node_features_energy_charges(
        z, torch.zeros(2, 0, dtype=torch.long), p[:0], p, c, torch.ones(3, dtype=torch.bool), 0.0, 0.0)
    assert float(node.sum()) == pytest.approx(float(ref["energy"][0]), abs=1e-9)
    torch.testing.assert_close(q, ref["charges"], rtol=0.0, atol=1e-9)
    path = export_torchscript_potential(model, cutoff=10.0, path=str(tmp_path / "cell.pt"))
    with torch.jit.optimized_execution(False):
        out = torch.jit.load(path)(p, z, c, torch.ones(3, dtype=torch.bool))
    assert float(out["energy"]) == pytest.approx(float(ref["energy"][0]), abs=1e-9)
    torch.testing.assert_close(out["forces"], ref["forces"], rtol=0.0, atol=1e-8)
    torch.testing.assert_close(out["stress"], ref["stress"][0], rtol=0.0, atol=1e-8)
    torch.testing.assert_close(out["charges"], ref["charges"], rtol=0.0, atol=1e-9)




@pytest.mark.parametrize("table", [SPIN_W, SPIN_SHELL_W])
def test_torchscript_export_spin(skset, tmp_path, table):
    """A spin-polarized model exported to TorchScript gives the eager energies, forces and charges of radicals of a
    given multiplicity and of a closed shell."""
    from xnn.common.data.elements import CHEMICAL_SYMBOLS
    from xnn.common.deploy import export_torchscript_potential
    model = _model(DFTB3, skset, spin_constants=table)
    eager = ForceStressOutput(model).eval()
    core = model.export_core()
    zero, no = torch.zeros(3, 3, dtype=torch.float64), torch.zeros(3, dtype=torch.bool)
    for (symbols, pos), mult in (((["C", "H", "H", "H"], METHANE[1][:4]), 2), (HS, 2), (THIOFORMALDEHYDE, 1)):
        ref = eager(collate([_radical(symbols, pos, mult)]))
        z = torch.tensor([CHEMICAL_SYMBOLS.index(s) for s in symbols])
        p = torch.as_tensor(np.asarray(pos, dtype=np.float64))
        _, node, q = core.node_features_energy_charges(z, torch.zeros(2, 0, dtype=torch.long), p[:0], p, zero, no,
                                                        0.0, float(mult))
        assert float(node.sum()) == pytest.approx(float(ref["energy"][0]), abs=1e-9)
        torch.testing.assert_close(q, ref["charges"], rtol=0.0, atol=1e-9)
        path = export_torchscript_potential(model, cutoff=10.0, path=str(tmp_path / "s.pt"),
                                            spin_multiplicity=float(mult))
        with torch.jit.optimized_execution(False):
            out = torch.jit.load(path)(p, z)
        assert float(out["energy"]) == pytest.approx(float(ref["energy"][0]), abs=1e-9)
        torch.testing.assert_close(out["forces"], ref["forces"], rtol=0.0, atol=1e-8)
    with pytest.raises(ValueError, match="spin constants"):
        _model(DFTB2, skset, spin_constants={"H": [-0.07], "C": [-0.02]}).export_core().node_features_energy_charges(
            torch.tensor([16, 1]), torch.zeros(2, 0, dtype=torch.long), p[:0],
            torch.as_tensor(np.asarray(HS[1], dtype=np.float64)), zero, no, 0.0, 2.0)


_PLUS_U = {"shells": {"C": [{"shells": [1], "uj": 0.2}], "S": [{"shells": [1, 2], "uj": 0.1}]}}


@pytest.mark.parametrize("options", [
    {"range_separation": 0.3},
    {"onsite_correction": ONSITE, "electronic_temperature": 1000.0},
    {"onsite_correction": ONSITE, "range_separation": 0.3, "electronic_temperature": 1000.0},
    {"spin_constants": SPIN_W, "orbital_potential": dict(_PLUS_U, functional="fll")},
    {"spin_constants": SPIN_W, "orbital_potential": dict(_PLUS_U, functional="psic")}])
def test_torchscript_export_density_terms(skset, tmp_path, options):
    """Long-range exchange, the onsite correction and DFTB+U exported to TorchScript give the eager energies, forces,
    stress and charges of molecules and of a molecule in a cell."""
    from xnn.common.data.elements import CHEMICAL_SYMBOLS
    from xnn.common.deploy import export_torchscript_potential
    model = _model(DFTB2, skset, **options)
    core = model.export_core()
    zero, no = torch.zeros(3, 3, dtype=torch.float64), torch.zeros(3, dtype=torch.bool)
    mult = 1 if "spin_constants" in options else 0
    for symbols, pos in (THIOFORMALDEHYDE, METHANE):
        ref = ForceStressOutput(model).eval()(collate([_radical(symbols, pos, mult) if mult else _graph(symbols, pos)]))
        z = torch.tensor([CHEMICAL_SYMBOLS.index(s) for s in symbols])
        p = torch.as_tensor(np.asarray(pos, dtype=np.float64))
        _, node, q = core.node_features_energy_charges(z, torch.zeros(2, 0, dtype=torch.long), p[:0], p, zero, no,
                                                        0.0, float(mult))
        assert float(node.sum()) == pytest.approx(float(ref["energy"][0]), abs=1e-9)
        torch.testing.assert_close(q, ref["charges"], rtol=0.0, atol=1e-9)
    symbols, pos = THIOFORMALDEHYDE
    cell = np.array([[7.0, 0.0, 0.0], [0.5, 7.5, 0.0], [0.0, 0.3, 8.0]])
    z = torch.tensor([CHEMICAL_SYMBOLS.index(s) for s in symbols])
    data = {"pos": torch.as_tensor(pos + 2.5), "atomic_numbers": z.numpy(), "total_charge": 0.0,
            "cell": torch.as_tensor(cell), "pbc": torch.ones(3, dtype=torch.bool)}
    if mult:
        data["spin_multiplicity"] = float(mult)
    ref = ForceStressOutput(model, compute_stress=True).eval()(collate([structure_to_graph(data, 0.1)]))
    _, node, _ = core.node_features_energy_charges(z, torch.zeros(2, 0, dtype=torch.long), data["pos"][:0],
                                                   data["pos"], data["cell"], data["pbc"], 0.0, float(mult))
    assert float(node.sum()) == pytest.approx(float(ref["energy"][0]), abs=1e-9)
    path = export_torchscript_potential(model, cutoff=10.0, path=str(tmp_path / "d.pt"), spin_multiplicity=float(mult))
    with torch.jit.optimized_execution(False):
        out = torch.jit.load(path)(torch.as_tensor(pos + 2.5), z, torch.as_tensor(cell),
                                   torch.ones(3, dtype=torch.bool))
    assert float(out["energy"]) == pytest.approx(float(ref["energy"][0]), abs=1e-9)
    torch.testing.assert_close(out["forces"], ref["forces"], rtol=0.0, atol=1e-8)
    torch.testing.assert_close(out["stress"], ref["stress"][0], rtol=0.0, atol=1e-8)


@pytest.mark.parametrize("options, mult", [({"kpoints": (2, 1, 1)}, 1),
                                           ({"kpoints": {"grid": [2, 2, 1], "shift": [0.5, 0.5, 0.0]}}, 1),
                                           ({"kpoints": (2, 1, 2), "shell_resolved": True}, 1),
                                           ({"kpoints": (2, 1, 1), "spin_constants": SPIN_W}, 2)])
def test_torchscript_export_kpoints(skset, tmp_path, options, mult):
    """A cell sampled at k-points exported to TorchScript gives the eager energy, forces, stress and charges."""
    from xnn.common.data.elements import CHEMICAL_SYMBOLS
    from xnn.common.deploy import export_torchscript_potential
    rng = np.random.default_rng(7)
    cell = np.eye(3) * 5.6 + 0.1 * rng.normal(size=(3, 3))
    symbols = ["C", "S", "C", "S", "H", "H", "H", "H"]
    pos = DIAMOND @ cell + 0.1 * rng.normal(size=(8, 3))
    if mult > 1:                                # a radical: the methyl radical in a cell
        cell = np.array([[6.0, 0.0, 0.0], [0.4, 6.5, 0.0], [0.0, 0.3, 7.0]])
        symbols, pos = ["C", "H", "H", "H"], METHANE[1][:4] + 2.5
    model = _model(DFTB2, skset, **options)
    z = torch.tensor([CHEMICAL_SYMBOLS.index(s) for s in symbols])
    data = {"pos": torch.as_tensor(pos), "atomic_numbers": z.numpy(), "total_charge": 0.0,
            "cell": torch.as_tensor(cell), "pbc": torch.ones(3, dtype=torch.bool), "spin_multiplicity": float(mult)}
    ref = ForceStressOutput(model, compute_stress=True).eval()(collate([structure_to_graph(data, 0.1)]))
    assert bool(ref["scf_converged"])
    _, node, q = model.export_core().node_features_energy_charges(
        z, torch.zeros(2, 0, dtype=torch.long), data["pos"][:0], data["pos"], data["cell"], data["pbc"], 0.0,
        float(mult))
    assert float(node.sum()) == pytest.approx(float(ref["energy"][0]), abs=1e-9)
    torch.testing.assert_close(q, ref["charges"], rtol=0.0, atol=1e-9)
    path = export_torchscript_potential(model, cutoff=10.0, path=str(tmp_path / "k.pt"), spin_multiplicity=float(mult))
    with torch.jit.optimized_execution(False):
        out = torch.jit.load(path)(data["pos"], z, data["cell"], data["pbc"])
    assert float(out["energy"]) == pytest.approx(float(ref["energy"][0]), abs=1e-9)
    torch.testing.assert_close(out["forces"], ref["forces"], rtol=0.0, atol=1e-8)
    torch.testing.assert_close(out["stress"], ref["stress"][0], rtol=0.0, atol=1e-8)
def test_torchscript_export_refusals(skset):
    """Methods the export does not cover are refused when the core is made."""
    for options in ({"kpoints": (2, 1, 1), "range_separation": 0.3},):
        with pytest.raises(NotImplementedError):
            _model(DFTB2, skset, **options).export_core()
    with pytest.raises(NotImplementedError, match="one charge per atom"):
        _model(DFTB3, skset, shell_resolved=True).export_core()
    f_shells = _model(DFTB2, skset)
    f_shells.lmax[16] = 4
    with pytest.raises(NotImplementedError, match="s, p, d and f"):
        f_shells.export_core()


def test_torchscript_export_trained_and_polynomial(skset, tmp_path):
    """The export reads trained values and polynomial repulsions, and refuses partly periodic cells and unknown
    elements."""
    from xnn.common.data.elements import CHEMICAL_SYMBOLS
    symbols, pos = THIOFORMALDEHYDE
    z = torch.tensor([CHEMICAL_SYMBOLS.index(s) for s in symbols])
    p = torch.as_tensor(np.asarray(pos, dtype=np.float64))
    zero, no = torch.zeros(3, 3, dtype=torch.float64), torch.zeros(3, dtype=torch.bool)
    empty = torch.zeros(2, 0, dtype=torch.long)
    trained = _model(DFTB3, skset, trainable=True)
    with torch.no_grad():
        for name in ("onsite", "hubbard", "spline_1_6", "head_1_6"):
            trained.params[name].mul_(1.01)
    folder = tmp_path / "poly"
    folder.mkdir()
    for f in skset.glob("*.skf"):
        text = f.read_text()
        if f.name == "H-H.skf":
            head, _, _ = text.partition("Spline")
            lines = head.splitlines()
            lines[2] = "1.008, 0.5, 7*0.0, 3.0, 10*0.0"
            text = "\n".join(lines) + "\n"
        (folder / f.name).write_text(text)
    for model in (trained.eval(), _model(DFTB2, folder)):
        ref = model(collate([_graph(symbols, pos)]))
        core = model.export_core()
        _, node, q = core.node_features_energy_charges(z, empty, p[:0], p, zero, no, 0.0, 0.0)
        assert float(node.sum()) == pytest.approx(float(ref["energy"][0]), abs=1e-9)
        torch.testing.assert_close(q, ref["charges"], rtol=0.0, atol=1e-9)
    with pytest.raises(ValueError, match="periodic"):
        core.node_features_energy_charges(z, empty, p[:0], p, torch.eye(3, dtype=torch.float64) * 9.0,
                                          torch.tensor([True, False, True]), 0.0, 0.0)
    with pytest.raises(ValueError, match="no parameters"):
        core.node_features_energy_charges(torch.tensor([8, 1, 1, 1]), empty, p[:0], p, zero, no, 0.0, 0.0)


def test_stacked_tables_match_single_tables():
    """Interpolating several tables at once gives each table's own values (inside, in the tail, past it)."""
    rows = [np.exp(-0.7 * g * np.arange(1, n + 1))[:, None] * np.array([1.0, -0.3]) for g, n in ((0.02, 100),
                                                                                                 (0.05, 60))]
    grids = (0.02, 0.05)
    tables = [torch.as_tensor(t) for t in rows]
    coef = [D.tail_polynomial(t, g) for t, g in zip(tables, grids)]
    r = torch.tensor([0.05, 0.37, 1.234, 1.99, 2.5, 3.5, 2.95, 3.0, 3.9, 5.0])
    which = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    start, count = torch.tensor([0, 100]), torch.tensor([100, 60])
    got = D.stacked_values(torch.cat(tables), start[which], count[which], torch.tensor(grids)[which],
                           torch.stack(coef)[which], r)
    for k in range(2):
        sel = which == k
        torch.testing.assert_close(got[sel], D.table_values(tables[k], grids[k], r[sel]), rtol=0.0, atol=0.0)


def test_set_without_a_pair_file(skset, tmp_path):
    """Elements without a file between them may share a batch (and an exported model) when they never meet in a
    structure."""
    for f in skset.iterdir():
        if f.name not in ("C-S.skf", "S-C.skf"):
            (tmp_path / f.name).write_text(f.read_text())
    graphs = [_graph(*METHANE), _graph(*HS)]
    full = _model(DFTB2, skset)(collate(graphs))["energy"]
    torch.testing.assert_close(_model(DFTB2, tmp_path)(collate(graphs))["energy"], full, rtol=0.0, atol=1e-12)
    from xnn.common.data.elements import CHEMICAL_SYMBOLS
    core = _model(DFTB2, tmp_path).export_core()
    zero, no = torch.zeros(3, 3, dtype=torch.float64), torch.zeros(3, dtype=torch.bool)
    for (symbols, pos), ok in ((METHANE, True), (THIOFORMALDEHYDE, False)):
        z = torch.tensor([CHEMICAL_SYMBOLS.index(s) for s in symbols])
        p = torch.as_tensor(np.asarray(pos, dtype=np.float64))
        args = (z, torch.zeros(2, 0, dtype=torch.long), p[:0], p, zero, no, 0.0, 0.0)
        if ok:
            ref = _model(DFTB2, tmp_path)(collate([_graph(symbols, pos)]))["energy"]
            assert float(core.node_features_energy_charges(*args)[1].sum()) == pytest.approx(float(ref[0]), abs=1e-9)
        else:
            with pytest.raises(ValueError, match="pair"):
                core.node_features_energy_charges(*args)


def test_host_route_of_small_gpu_batches(skset):
    """The fast path sends small batches on a GPU to a host copy of the model, with the same results."""
    from types import SimpleNamespace
    model = _model(DFTB3, skset)
    g = collate([_graph(*THIOFORMALDEHYDE), _graph(*METHANE)])
    on_gpu = SimpleNamespace(pos=SimpleNamespace(device=torch.device("cuda")), num_graphs=2, batch=g.batch,
                             atomic_numbers=g.atomic_numbers)
    assert model.use_fast == "auto" and not model._on_host(g) and model._on_host(on_gpu)
    model.host_basis = 12                       # thioformaldehyde alone has 15 orbitals
    assert not model._on_host(on_gpu)
    model.host_basis = DFTB.host_basis
    assert not _model(DFTB3, skset, use_fast=False)._on_host(on_gpu)
    assert not _model(DFTB3, skset, trainable="hubbard")._on_host(on_gpu)
    with pytest.raises(ValueError):
        _model(DFTB3, skset, use_fast="sometimes")
    model._tables[("probe",)] = (torch.zeros(1),)
    twin = model._host_model()
    assert twin is model._host_model() and twin is not model and "_twin" not in dict(model.named_modules())
    assert ("probe",) not in twin._tables and ("probe",) in model._tables
    g.pos.requires_grad_(True)
    ref = model(g)["energy"]
    ref_grad = torch.autograd.grad(ref.sum(), g.pos)[0]
    model._on_host = lambda data: True          # as on a GPU: the evaluation goes through the host copy
    got = model(g)["energy"]
    torch.testing.assert_close(got, ref, rtol=0.0, atol=1e-10)
    torch.testing.assert_close(torch.autograd.grad(got.sum(), g.pos)[0], ref_grad, rtol=0.0, atol=1e-10)
    assert model.last_scf is twin.last_scf
    model.scc_tolerance = 1e-9
    assert model._host_model() is not twin and model._host_model().scc_tolerance == 1e-9


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gpu_matches_cpu(skset):
    """On a GPU the host route and the device route give the CPU's energies and forces."""
    graphs = [_graph(*THIOFORMALDEHYDE), _graph(*METHANE)]
    cpu = ForceStressOutput(_model(DFTB3, skset)).eval()(collate(graphs))
    for limit, host in ((DFTB.host_basis, True), (0, False)):
        model = _model(DFTB3, skset).to("cuda")
        model.host_basis = limit
        data = collate(graphs).to("cuda")
        assert model._on_host(data) is host
        gpu = ForceStressOutput(model).eval()(data)
        assert gpu["energy"].device.type == "cuda"
        torch.testing.assert_close(gpu["energy"].cpu(), cpu["energy"], rtol=0.0, atol=1e-10)
        torch.testing.assert_close(gpu["forces"].cpu(), cpu["forces"], rtol=0.0, atol=1e-9)


def test_two_center_kernels_interpreted(skset, monkeypatch):
    """The fast path's kernels, run by Triton's interpreter, give the reference's energies, forces and stress for
    molecules and a crystal with periodic images; they are taken only where they apply."""
    if os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("runs where Triton's interpreter is on (no CUDA device)")
    from types import SimpleNamespace

    from xnn.sqmnn.models import dftb_kernels
    rng = np.random.default_rng(5)
    cell = np.eye(3) * 5.6 + 0.1 * rng.normal(size=(3, 3))
    crystal = _crystal(["C", "S", "C", "S", "H", "H", "H", "H"], DIAMOND @ cell + 0.1 * rng.normal(size=(8, 3)), cell)
    for graphs in ([_graph(*THIOFORMALDEHYDE), _graph(*METHANE), _graph(*HS, charge=-1.0)], [crystal]):
        ref = ForceStressOutput(_model(DFTB3, skset), compute_stress=True).eval()(collate(graphs))
        fast = _model(DFTB3, skset)
        fast._fused_integrals = lambda basis, pos: True
        got = ForceStressOutput(fast, compute_stress=True).eval()(collate(graphs))
        for key, tol in ((k, t) for k, t in (("energy", 1e-10), ("forces", 1e-9), ("stress", 1e-9)) if k in ref):
            torch.testing.assert_close(got[key], ref[key], rtol=0.0, atol=tol)
    model = _model(DFTB3, skset)
    basis = SimpleNamespace(copies=None)
    gpu = SimpleNamespace(is_cuda=True, device=torch.device("cuda"))
    monkeypatch.setattr(dftb_kernels, "available", lambda device: True)
    assert model._fused_integrals(basis, gpu) and not model._fused_integrals(basis, torch.zeros(1))
    assert not model._fused_integrals(SimpleNamespace(copies=[2]), gpu)
    assert not _model(DFTB3, skset, use_fast=False)._fused_integrals(basis, gpu)
    assert _model(DFTB3, skset, trainable="tables")._fused_integrals(basis, gpu)
    assert _model(DFTB3, skset, trainable="hubbard")._fused_integrals(basis, gpu)
    monkeypatch.undo()
    assert not dftb_kernels.available(torch.device("cpu"))


def test_two_center_kernel_second_derivatives(skset):
    """The kernels' gradient is differentiable once more: derivatives in the positions and in the weights of the
    matrices equal those of the reference; Hessians and force-loss gradients of trained parameters agree."""
    if os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("runs where Triton's interpreter is on (no CUDA device)")
    g = collate([_graph(*THIOFORMALDEHYDE), _graph(*HS)])
    gen = torch.Generator().manual_seed(3)
    results = []
    for fused in (False, True):
        model = _model(DFTB3, skset)
        model._fused_integrals = (lambda basis, pos: True) if fused else (lambda basis, pos: False)
        model.__dict__["_lattice"], model.__dict__["_kgrid"] = None, None
        basis = model._make_basis(g.atomic_numbers.tolist(), g.batch.tolist(), 2, torch.float64, g.pos.device)
        pos = (g.pos / model.bohr).detach().requires_grad_(True)
        S, H = model._two_center(basis, pos, 2)
        wS, wH = torch.randn(S.shape, generator=gen).requires_grad_(True), torch.randn(H.shape, generator=gen)
        wH.requires_grad_(True)
        (grad,) = torch.autograd.grad((S * wS).sum() + (H * wH).sum(), pos, create_graph=True)
        direction = torch.randn(grad.shape, generator=gen)
        second = torch.autograd.grad((grad * direction).sum(), (pos, wS, wH))
        results.append((S.detach(), H.detach(), grad.detach()) + tuple(t.detach() for t in second))
        gen = torch.Generator().manual_seed(3)
    for ref, got in zip(*results):
        torch.testing.assert_close(got, ref, rtol=0.0, atol=1e-11)
    hess = []
    for fused in (False, True):
        model = _model(DFTB3, skset)
        model._fused_integrals = (lambda basis, pos: True) if fused else (lambda basis, pos: False)
        hess.append(hessian(model, collate([_graph(*THIOFORMALDEHYDE)]))[0])
    torch.testing.assert_close(hess[1], hess[0], rtol=0.0, atol=1e-8)
    for trained in ("hubbard", "tables"):
        grads = []
        for fused in (False, True):
            model = _model(DFTB3, skset, trainable=trained)
            model._fused_integrals = (lambda basis, pos: True) if fused else (lambda basis, pos: False)
            out = ForceStressOutput(model).train()(collate([_graph(*THIOFORMALDEHYDE)]))
            loss = (out["forces"] ** 2).sum() + out["energy"].sum()
            grads.append(torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad]))
        for ref, got in zip(*grads):
            torch.testing.assert_close(got, ref, rtol=1e-11, atol=1e-9)
    model = _model(DFTB2, skset, trainable="tables")
    model._fused_integrals = lambda basis, pos: True
    energy = model(collate([_graph(*HS)]))["energy"].sum()
    table = model.params["H_16_1"]
    (g_table,) = torch.autograd.grad(energy, table, create_graph=True)
    with pytest.raises(NotImplementedError):
        torch.autograd.grad(g_table.sum(), table)


def test_periodic_hessian(skset):
    """The Hessian of a crystal at the Gamma point is finite, symmetric and equals central differences of the
    forces."""
    rng = np.random.default_rng(5)
    cell = np.eye(3) * 5.6 + 0.1 * rng.normal(size=(3, 3))
    symbols = ["C", "S", "C", "S", "H", "H", "H", "H"]
    pos = DIAMOND @ cell + 0.1 * rng.normal(size=(8, 3))
    model = _model(DFTB2, skset)
    (H,) = hessian(model, _crystal(symbols, pos, cell))
    assert bool(torch.isfinite(H).all()) and torch.allclose(H, H.T, atol=1e-8)
    wrapped = ForceStressOutput(model).eval()
    step = 1e-4
    for k in (0, 4, 13):
        cols = []
        for s in (1, -1):
            p = pos.copy()
            p.reshape(-1)[k] += s * step
            cols.append(wrapped(collate([_crystal(symbols, p, cell)]))["forces"].detach().numpy().reshape(-1))
        assert np.allclose(H[:, k].numpy(), -(cols[0] - cols[1]) / (2 * step), atol=1e-6)


@pytest.fixture(scope="module")
def fset(skset, tmp_path_factory) -> Path:
    """The synthetic set with an f shell on sulfur: extended files (with smooth f columns) for every pair with
    sulfur."""
    import dataclasses
    folder = tmp_path_factory.mktemp("skf_f")
    for path in skset.glob("*.skf"):
        a, b = path.stem.split("-")
        if "S" not in (a, b):
            (folder / path.name).write_text(path.read_text())
            continue
        g = skf.parse_skf(path.read_text(), a == b)
        n = g.hamiltonian.shape[0]

        def wide(tab, overlap, a=a, b=b, n=n):
            out = np.zeros((n, len(skf.EXTENDED_COLUMNS)))
            for k, name in enumerate(skf.EXTENDED_COLUMNS):
                out[:, k] = (tab[:, skf.COLUMNS.index(name)] if name in skf.COLUMNS
                             else 0.3 * _column(a, b, name, overlap)[:n])
            return out
        extra = {}
        if a == b:
            extra = {"energies": g.energies + (0.45,), "hubbard": g.hubbard + (0.28,),
                     "occupations": g.occupations + (0.0,), "n_shells": 4}
        f = dataclasses.replace(g, extended=True, hamiltonian=wide(g.hamiltonian, False),
                                overlap=wide(g.overlap, True), **extra)
        (folder / path.name).write_text(skf.format_skf(f))
    return folder


def test_f_shells_and_their_export(fset, tmp_path):
    """f shells: the energy is invariant under rotations, forces are its derivatives, and the TorchScript export
    gives the eager energy, forces and charges of a molecule and a cell."""
    from xnn.common.data.elements import CHEMICAL_SYMBOLS
    from xnn.common.deploy import export_torchscript_potential
    model = _model(DFTB2, fset, max_angular_momentum={"H": "s", "C": "p", "S": "f"})
    assert model.lmax[16] == 3
    symbols, pos = THIOFORMALDEHYDE
    wrapped = ForceStressOutput(model).eval()
    ref = wrapped(collate([_graph(symbols, pos)]))
    assert bool(ref["scf_converged"])
    q, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))
    assert float(model(collate([_graph(symbols, pos @ q.T)]))["energy"][0]) == pytest.approx(
        float(ref["energy"][0]), abs=1e-9)
    e = []
    for s in (1, -1):
        p = pos.copy()
        p[1, 2] += s * 1e-5
        e.append(float(model(collate([_graph(symbols, p)]))["energy"][0]))
    assert abs(float(ref["forces"][1, 2]) + (e[0] - e[1]) / 2e-5) < 1e-6
    z = torch.tensor([CHEMICAL_SYMBOLS.index(s) for s in symbols])
    p = torch.as_tensor(np.asarray(pos, dtype=np.float64))
    zero, no = torch.zeros(3, 3, dtype=torch.float64), torch.zeros(3, dtype=torch.bool)
    _, node, charges = model.export_core().node_features_energy_charges(z, torch.zeros(2, 0, dtype=torch.long), p[:0],
                                                                        p, zero, no, 0.0, 0.0)
    assert float(node.sum()) == pytest.approx(float(ref["energy"][0]), abs=1e-9)
    torch.testing.assert_close(charges, ref["charges"], rtol=0.0, atol=1e-9)
    path = export_torchscript_potential(model, cutoff=10.0, path=str(tmp_path / "f.pt"))
    cell = np.array([[7.0, 0.0, 0.0], [0.5, 7.5, 0.0], [0.0, 0.3, 8.0]])
    ref_cell = ForceStressOutput(model, compute_stress=True).eval()(collate([_crystal(symbols, pos + 2.5, cell)]))
    with torch.jit.optimized_execution(False):
        scripted = torch.jit.load(path)
        out = scripted(p, z)
        out_cell = scripted(p + 2.5, z, torch.as_tensor(cell), torch.ones(3, dtype=torch.bool))
    assert float(out["energy"]) == pytest.approx(float(ref["energy"][0]), abs=1e-9)
    torch.testing.assert_close(out["forces"], ref["forces"], rtol=0.0, atol=1e-8)
    assert float(out_cell["energy"]) == pytest.approx(float(ref_cell["energy"][0]), abs=1e-9)
    torch.testing.assert_close(out_cell["forces"], ref_cell["forces"], rtol=0.0, atol=1e-8)
    torch.testing.assert_close(out_cell["stress"], ref_cell["stress"][0], rtol=0.0, atol=1e-8)
