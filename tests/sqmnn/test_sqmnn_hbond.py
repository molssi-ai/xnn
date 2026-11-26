"""PM7 hydrogen bonds (sqmnn): the smooth form of the term.

Away from its switches the smooth form gives PM7's energies; the enumerated choices of reference atoms reduce to
PM7's where no comparison is inside its switch; the O...O switch, the taper and the blend of tied orders have
their defined values; energy, slope and curvature are continuous across every switch where PM7 jumps; gradients
and Hessians equal central differences inside the switches; the energy does not depend on the numbering of the
atoms; the host, PyTorch and TorchScript forms agree; and the model option reaches the energy and the export.
The smooth torsions leave the energy unchanged outside their windows and at planar and linear hydrogen bonds,
remove the kinks there and keep exact derivatives.
"""
import math

import numpy as np
import pytest
import torch

from test_sqmnn_forms import EDGE_CASES, _graph, _hbond_cluster, _scripted_branches
from xnn.sqmnn.models import PM7
from xnn.sqmnn.models import corrections as cr

TABLES = cr.correction_tables(torch.device("cpu"))
# the acceptor hydrogens of a water (directions from its oxygen, mirror images in y)
_ACC = np.array([0.35, 0.79, 0.5]) / math.sqrt(0.9966)


@pytest.fixture(autouse=True)
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _water_dimer(r_oo, d_oh=0.96, tilt=0.0):
    """A water dimer with a straight O-H...O along x: donor O at the origin, its bonded H at ``d_oh``, the acceptor
    O at ``r_oo``; ``tilt`` turns the acceptor's hydrogens about z (0: mirror images, an exact tie)."""
    up, down = _ACC.copy(), _ACC * np.array([1.0, -1.0, 1.0])
    c, s = math.cos(tilt), math.sin(tilt)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    o2 = np.array([r_oo, 0.0, 0.0])
    x = np.array([[0.0, 0, 0], [d_oh, 0, 0], [-0.24, 0.93, 0.0], o2, o2 + 0.96 * rot @ up, o2 + 0.96 * rot @ down])
    return [8, 1, 1, 8, 1, 1], x


def _tied_amine(shift=0.0):
    """Water donating to an amine nitrogen whose two hydrogens are exactly equally far from the donor's hydrogen and
    whose third neighbor (a carbon) is off their mirror plane, so that the two orders give different energies;
    ``shift`` moves one of the hydrogens away from the donor's hydrogen."""
    u = np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0)
    n = np.array([1.0, -1.0, 0.0]) / math.sqrt(2.0)
    ez = np.array([0.0, 0.0, 1.0])
    nitrogen = 2.05 * u

    def bond(phi):
        return 0.334 * u + 0.943 * (math.cos(phi) * n + math.sin(phi) * ez)
    h3 = nitrogen + 1.01 * bond(math.radians(30.0))
    h4 = h3[[1, 0, 2]] + shift * u
    carbon = nitrogen + 1.47 * bond(math.radians(250.0))
    donor = -0.96 * u
    x = np.array([donor, [0.0, 0, 0], donor + 0.96 * (-0.25 * u + 0.968 * ez), nitrogen, h3, h4, carbon])
    return [8, 1, 1, 7, 1, 1, 6], x


def _formaldehyde_acceptor(theta):
    """Water donating to formaldehyde's oxygen at the C=O...H angle ``theta`` (degrees), H...O 1.9 Angstrom: the
    oxygen end has one bond, and its zero-torsion flag switches where H...O equals H...C."""
    t = math.radians(theta)
    o, c = np.zeros(3), np.array([-1.21, 0.0, 0.0])
    h = 1.9 * np.array([-math.cos(t), math.sin(t), 0.05])
    h = 1.9 * h / np.linalg.norm(h)
    away = h / np.linalg.norm(h)
    od = h + 0.96 * away
    x = np.array([c, o, c + [-0.55, 0.94, 0.0], c + [-0.55, -0.94, 0.0], od, h, od + [0.3, 0.2, 0.9]])
    return [6, 8, 1, 1, 8, 1, 1], x


def _energy(z, x, smooth=True, constants=None, torsion=False):
    c = None if constants is None else torch.tensor(constants)
    return float(cr.pm7_hbond(list(z), torch.as_tensor(np.asarray(x, dtype=np.float64)), TABLES, c, smooth,
                              torsion).sum())


def _derivatives(z, x, v, smooth=True, torsion=False):
    """Energy, slope and curvature along the direction ``v`` (the PyTorch form for the curvature)."""
    p = torch.tensor(np.asarray(x, dtype=np.float64), requires_grad=True)
    vt = torch.as_tensor(v)
    e = cr.pm7_hbond(list(z), p, TABLES, None, smooth, torsion).sum()
    if not e.requires_grad:
        return float(e), 0.0, 0.0
    g, = torch.autograd.grad(e, p, create_graph=True)
    slope = (g * vt).sum()
    if not slope.requires_grad:
        return float(e), float(slope), 0.0
    curv, = torch.autograd.grad(slope, p, allow_unused=True)
    return float(e), float(slope), 0.0 if curv is None else float((curv * vt).sum().detach())


def _jumps(path, s0, smooth=True, eps=1e-7, torsion=False):
    """The changes of energy, slope and curvature across ``s0`` of ``path(s) -> (z, x, dx/ds)``: the values
    ``eps`` to either side, carried to ``s0`` by their Taylor series."""
    (e_l, s_l, c_l), (e_r, s_r, c_r) = [_derivatives(*path(s), smooth=smooth, torsion=torsion)
                                        for s in (s0 - eps, s0 + eps)]
    return [abs((e_r - eps * s_r + 0.5 * eps * eps * c_r) - (e_l + eps * s_l + 0.5 * eps * eps * c_l)),
            abs((s_r - eps * c_r) - (s_l + eps * c_l)), abs(c_r - c_l)]


def _move(z, x, atoms, direction, s):
    """``x`` with ``atoms`` moved by ``s`` along ``direction``, and the derivative of the positions in ``s``."""
    d = np.zeros_like(x)
    d[atoms] = np.asarray(direction, dtype=np.float64) / np.linalg.norm(direction)
    return z, x + s * d, d


def test_unchanged_away_from_the_switches():
    """Structures with no comparison inside a switch and every bond below 7 Angstrom have PM7's energies, also
    with every candidate sent through the enumerated choices; an exact tie of mirror images changes nothing."""
    systems = [_water_dimer(2.9, tilt=0.3), _water_dimer(3.3), _water_dimer(5.0, tilt=-0.4),
               _formaldehyde_acceptor(120.0), _tied_amine(0.2)]
    for z, x in systems:
        sharp = _energy(z, x, smooth=False)
        assert sharp < -0.01
        assert _energy(z, x) == pytest.approx(sharp, abs=1e-12)


def test_enumerated_choices_reduce_to_pm7(monkeypatch):
    """Sending every candidate through the enumeration of choices gives the energies of the vectorized choices,
    with and without comparisons inside their switches (random clusters of every kind of group)."""
    systems = [_hbond_cluster(), _water_dimer(2.9, tilt=0.3), _tied_amine(0.0), _tied_amine(0.02),
               _formaldehyde_acceptor(71.5), _water_dimer(2.91, d_oh=1.27)]
    rng = np.random.default_rng(5)
    for _ in range(4):
        z, x = _hbond_cluster()
        systems.append((z, x + 0.3 * rng.normal(size=x.shape)))
    plain = [_energy(z, x) for z, x in systems]
    monkeypatch.setattr(cr, "_hb_open", lambda D, rz, zz, t, L, k, full: torch.ones(t.shape[0], dtype=torch.bool))
    for (z, x), e in zip(systems, plain):
        assert _energy(z, x) == pytest.approx(e, abs=1e-11)


def test_oo_switch_and_taper_values():
    """With only the O...O constant a straight O-H...O bond's energy is ``-G(R)``, the C2 switch of
    ``exp(-80 (R - 2.67)^2)`` within 0.05 Angstrom of 2.67 (PM7: the clamped Gaussian); the whole energy goes to
    zero between 7 and 8 Angstrom by the quintic smoothstep, PM7 below."""
    for r in np.linspace(2.57, 2.77, 41):
        z, x = _water_dimer(r)
        xo = r - 2.67
        s = min(max((xo + 0.05) / 0.1, 0.0), 1.0)
        step = s ** 3 * (10.0 - 15.0 * s + 6.0 * s * s)
        assert _energy(z, x, constants=[0.0, 0.0, 1.0]) == pytest.approx(
            -(1.0 - step * (1.0 - math.exp(-80.0 * xo * xo))), abs=1e-12)
        assert _energy(z, x, False, [0.0, 0.0, 1.0]) == pytest.approx(-math.exp(-80.0 * max(xo, 0.0) ** 2),
                                                                      abs=1e-12)
    below = _energy(*_water_dimer(7.0 - 1e-9), smooth=False)
    assert below < -0.05
    assert _energy(*_water_dimer(7.0 + 1e-9), smooth=False) == 0.0
    assert _energy(*_water_dimer(7.0 + 1e-9)) == pytest.approx(below, abs=1e-7)
    assert _energy(*_water_dimer(8.0)) == 0.0
    assert -0.1 < _energy(*_water_dimer(7.5)) < 0.0


def test_tied_orders_blend_and_numbering_does_not_matter():
    """At an exact tie of two neighbors' distances from the hydrogen the energy is the mean of PM7's two orders
    (which PM7 picks by numbering); the smooth energy is the same for every numbering of the atoms."""
    z, x = _tied_amine()
    swap = [0, 1, 2, 3, 5, 4, 6]
    first, second = _energy(z, x, smooth=False), _energy(np.asarray(z)[swap], x[swap], smooth=False)
    assert abs(first - second) > 1e-4
    assert _energy(z, x) == pytest.approx(0.5 * (first + second), abs=1e-12)
    rng = np.random.default_rng(2)
    systems = [(z, x), _tied_amine(0.01), _formaldehyde_acceptor(71.5), _water_dimer(2.91, d_oh=1.25),
               _water_dimer(7.4, tilt=0.2)]
    for zs, xs in systems:
        ref = _energy(zs, xs)
        assert ref < 0.0
        for _ in range(3):
            perm = rng.permutation(len(zs))
            assert _energy(np.asarray(zs)[perm], xs[perm]) == pytest.approx(ref, abs=1e-10)


def _locate(path, fn, lo, hi):
    """The parameter in ``[lo, hi]`` where ``fn`` of the positions of ``path`` changes sign (bisection)."""
    below = fn(path(lo)[1]) > 0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if (fn(path(mid)[1]) > 0) == below:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _switch_paths():
    """Paths ``(name, path, location, a point inside the switch)`` across every switch of the smooth term."""
    def oo(s):
        z, x = _water_dimer(2.9, tilt=0.3)
        return _move(z, x, [3, 4, 5], [1, 0, 0], s - 2.9)

    def proton(s):
        z, x = _water_dimer(2.91, tilt=0.3)
        return _move(z, x, [1], [1, 0, 0], s - 0.96)

    def order(s):
        z, x = _tied_amine()
        return _move(z, x, [5], x[5] - x[1], s)

    def zero_t(s):
        z, x = _formaldehyde_acceptor(71.5)
        return _move(z, x, [4, 5, 6], [1, 0, 0], s)

    def nitrogen(s):
        # a nitrile nitrogen accepting from a stretched O-H: the donor's hydrogen enters its bonded list
        z = [8, 1, 1, 7, 6, 1]
        x = np.array([[-1.1, 0, 0], [0.0, 0, 0], [-1.34, 0.93, 0.1], [1.6, 0.05, 0], [2.75, 0.17, 0.05],
                      [3.81, 0.32, 0.15]])
        return _move(z, x, [3, 4, 5], [-1, 0, 0], s)

    rho_hn = float(cr._NP_HB_RADIUS[1] + cr._NP_HB_RADIUS[7])
    s_zero = _locate(zero_t, lambda x: np.linalg.norm(x[5] - x[1]) - np.linalg.norm(x[5] - x[0]), -0.3, 0.3)
    s_n = _locate(nitrogen, lambda x: np.linalg.norm(x[3] - x[1]) - rho_hn, 0.0, 0.4)
    return [("O...O", oo, 2.67, 2.683), ("O...O edge", oo, 2.72, 2.70), ("taper", oo, 7.0, 7.3),
            ("taper end", oo, 8.0, 7.7), ("damping", proton, 1.205, 1.218), ("H-O membership", proton, 1.267, 1.29),
            ("order", order, 0.0, 0.013), ("order edge", order, 0.05, 0.04), ("zero torsion", zero_t, s_zero,
                                                                             s_zero + 0.013),
            ("H...N membership", nitrogen, s_n, s_n + 0.03)]


def test_energy_slope_and_curvature_continuous_across_switches():
    """Across every switch the smooth form's energy, slope and curvature are continuous (one-sided values
    1e-7 Angstrom apart), where PM7 jumps in energy or curvature at its switch."""
    for name, path, s0, _ in _switch_paths():
        de, ds, dc = _jumps(path, s0)
        assert de < 1e-8 and ds < 1e-5 and dc < 1.0, (name, de, ds, dc)
    sharp = {name: _jumps(path, s0, smooth=False) for name, path, s0, _ in _switch_paths()
             if name in ("O...O", "taper", "zero torsion", "H...N membership", "damping", "order")}
    assert sharp["O...O"][2] > 10.0
    for name in ("taper", "zero torsion", "H...N membership", "damping", "order"):
        assert sharp[name][0] > 1e-3, (name, sharp[name])


@pytest.mark.parametrize("index", range(10))
def test_derivatives_inside_switches(index):
    """Inside each switch the forces equal central differences of the energy and the Hessian equals central
    differences of the forces."""
    name, path, _, inside = _switch_paths()[index]
    z, x, _ = path(inside)
    p = torch.tensor(x, requires_grad=True)
    e = cr.pm7_hbond(z, p, TABLES, None, True).sum()
    g, = torch.autograd.grad(e, p, create_graph=True)
    v = torch.as_tensor(np.random.default_rng(index).normal(size=x.shape))
    hv, = torch.autograd.grad((g * v).sum(), p)
    h = 5e-6
    fd = np.zeros_like(x)
    for i in range(len(z)):
        for k in range(3):
            xp, xm = x.copy(), x.copy()
            xp[i, k] += h
            xm[i, k] -= h
            fd[i, k] = (_energy(z, xp) - _energy(z, xm)) / (2.0 * h)
    assert np.allclose(g.detach().numpy(), fd, rtol=0.0, atol=2e-6), name

    def grad(xx):
        q = torch.tensor(xx, requires_grad=True)
        return torch.autograd.grad(cr.pm7_hbond(z, q, TABLES, None, True).sum(), q)[0].numpy()
    hd = 2e-6
    fd_hv = (grad(x + hd * v.numpy()) - grad(x - hd * v.numpy())) / (2.0 * hd)
    assert np.allclose(hv.numpy(), fd_hv, rtol=0.0, atol=1e-3), name


def _crowded(seed):
    rng = np.random.default_rng(seed)
    n = 30 + 4 * seed
    z = [int(v) for v in rng.choice([1, 1, 1, 6, 7, 8], size=n)]
    return z, rng.uniform(0.0, (n / 0.12) ** (1.0 / 3.0), size=(n, 3))


def test_smooth_forms_agree(monkeypatch):
    """The host (NumPy) and PyTorch tables agree, and the TorchScript branches (run eagerly) and the compiled
    corrections give the host forms' energies and gradients, on structures with choices inside their switches."""
    systems = [_crowded(s) for s in range(6)] + [_tied_amine(0.01), _formaldehyde_acceptor(71.5),
                                                  _water_dimer(2.91, d_oh=1.25), _hbond_cluster()]
    opened = 0
    for z, x in systems:
        host = cr._np_pm7_hbond_tables(z, x, True)
        torch_form = cr._pm7_hbond_tables(z, torch.tensor(x), TABLES, True)
        assert (host is None) == (torch_form is None)
        if host is None:
            continue
        opened += int(host["fac_atoms"].shape[1] > 0)
        for key, value in host.items():
            assert np.array_equal(value, torch_form[key].numpy()), key

    def terms(fn):
        out = []
        for z, x in systems:
            for torsion in (False, True):
                p = torch.tensor(x, requires_grad=True)
                e = fn(2, z, p, TABLES, 3.1595, None, None, None, False, True, torsion)
                out.append((e.detach(), torch.autograd.grad(e.sum(), p)[0]))
        return out
    host = terms(cr.structure_corrections)
    scripted = terms(torch.jit.script(cr.structure_corrections))
    with monkeypatch.context() as m:
        _scripted_branches(m, cr)
        branches = terms(cr.structure_corrections)
    for (e, g), (es, gs), (eb, gb) in zip(host, scripted, branches):
        assert torch.isfinite(g).all()
        assert torch.allclose(es, e, rtol=1e-12, atol=1e-12) and torch.allclose(eb, e, rtol=1e-12, atol=1e-12)
        assert torch.allclose(gs, g, rtol=1e-10, atol=1e-10) and torch.allclose(gb, g, rtol=1e-10, atol=1e-10)
    assert opened >= 4
    # candidates that every configuration drops
    z5, x5 = EDGE_CASES[-2]
    assert cr._np_pm7_hbond_tables(z5, np.asarray(x5, dtype=np.float64), True) is None


def test_trained_constants_with_the_smooth_form():
    """The smooth energy stays linear in the constants: trainable constants give the plain evaluation's energy
    and the gradient of a constant is its unit evaluation."""
    z, x = _tied_amine(0.01)
    c = torch.tensor(cr.HBOND, requires_grad=True)
    e = cr.pm7_hbond(z, torch.tensor(x), TABLES, c, True).sum()
    assert float(e) == pytest.approx(_energy(z, x), abs=1e-12)
    g, = torch.autograd.grad(e, c)
    assert float(g[0]) == pytest.approx(_energy(z, x, constants=[1.0, 0.0, 0.0]), abs=1e-12)


def test_too_many_open_bonding_radii():
    """A crowd of atoms at the bonding radius of one hydrogen-bond end is refused."""
    z = [8] + [1] * 11 + [8]
    rng = np.random.default_rng(0)
    d = rng.normal(size=(11, 3))
    x = np.vstack([[0.0, 0, 0], 1.267 * d / np.linalg.norm(d, axis=1)[:, None], [[3.0, 0.0, 0.0]]])
    D = torch.as_tensor(cr._distances(x))
    rz = cr._NP_HB_RADIUS[np.asarray(z)].tolist()
    with pytest.raises(RuntimeError, match="more than 10"):
        cr._hb_configurations(0, 1, 12, z, rz, D, {}, cr._hb_near_lists(D, torch.tensor(rz)))


def test_hbond_switch_option():
    """``hbond_switch`` changes the heat of formation by the difference of the two forms, builds from a config,
    reaches the exported core and rejects unknown values."""
    from xnn.common.config import from_dict
    from xnn.common.models import build_model

    z, x = _tied_amine(0.01)
    sharp = PM7(reuse_density=False)(_graph(x, z, mult=2.0))["heat_of_formation"]
    smooth = PM7(reuse_density=False, hbond_switch="smooth")(_graph(x, z, mult=2.0))["heat_of_formation"]
    assert float(smooth - sharp) == pytest.approx(_energy(z, x) - _energy(z, x, smooth=False), abs=1e-8)
    assert abs(float(smooth - sharp)) > 1e-5
    built = build_model(from_dict({"model": {"name": "pm7", "hbond_switch": "smooth"}}).model)
    assert built.hbond_switch == "smooth"
    core = built.export_core()
    edges, vec = torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 3))
    _, node_energy, _ = core.node_features_energy_charges(torch.tensor(z), edges, vec, torch.tensor(x),
                                                          torch.zeros((3, 3)), torch.zeros(3, dtype=torch.bool),
                                                          0.0, 2.0)
    built.reuse_density = False
    assert torch.allclose(node_energy, built(_graph(x, z, mult=2.0))["node_energy"].detach(), rtol=0.0, atol=1e-10)
    with pytest.raises(ValueError, match="hbond_switch"):
        PM7(hbond_switch="soft")


def _planar_water(s):
    """An acceptor water in the xy plane with the donor water in that plane at s = 0, moved along z by s."""
    z = [8, 1, 1, 8, 1, 1]
    x = np.array([[0.0, 0, 0], [0.59, 0.76, 0], [0.59, -0.76, 0], [-2.9, 0.25, 0], [-1.95, 0.2, 0],
                  [-3.2, 1.15, 0]])
    return _move(z, x, [3, 4, 5], [0, 0, 1], s)


def _planar_amine(s):
    """A planar three-coordinate nitrogen donating to water, its hydrogen moved out of the plane by s."""
    z = [7, 6, 6, 1, 8, 1, 1]
    x = np.array([[0.0, 0, 0], [-0.73, 1.25, 0], [-0.73, -1.25, 0], [1.01, 0, 0], [2.95, 0.05, 0.1],
                  [3.3, 0.85, 0.45], [3.3, -0.7, 0.55]])
    return _move(z, x, [3], [0, 0, 1], s)


def _linear_nitrile(s):
    """A water donating along the C-N axis of a bent HCN at s = 0, moved across the axis by s."""
    z = [8, 1, 1, 7, 6, 1]
    x = np.array([[-0.96, 0, 0], [0.0, 0, 0], [-1.2, 0.93, 0.1], [2.0, 0, 0], [3.16, 0, 0], [4.21, 0.15, 0.05]])
    return _move(z, x, [0, 1, 2], [0, 1, 0], s)


TORSION_PATHS = [_planar_water, _planar_amine, _linear_nitrile]


def test_smooth_torsion_unchanged_outside_its_windows():
    """The smooth torsions give the smooth form's energy away from planar and linear hydrogen bonds, and at the
    planar or linear points themselves; inside their windows they change it."""
    for z, x in (_water_dimer(2.9, tilt=0.3), _water_dimer(3.3, tilt=0.4), _tied_amine(0.2),
                 _formaldehyde_acceptor(120.0)):
        assert _energy(z, x, torsion=True) == pytest.approx(_energy(z, x), abs=1e-12)
    for path in TORSION_PATHS:
        z, x, _ = path(0.0)
        assert _energy(z, x, torsion=True) == pytest.approx(_energy(z, x), abs=1e-12)
        z, x, _ = path(0.05)
        assert abs(_energy(z, x, torsion=True) - _energy(z, x)) > 1e-4
        z, x, _ = path(0.2)
        assert _energy(z, x, torsion=True) == pytest.approx(_energy(z, x), abs=1e-12)


@pytest.mark.parametrize("path", TORSION_PATHS)
def test_smooth_torsion_removes_the_kinks(path):
    """Through a planar water acceptor, a planar nitrogen donor and a nitrile axis the smooth form's slope jumps;
    with smooth torsions energy, slope and curvature are continuous, and the gradient and Hessian equal central
    differences at the planar or linear point and inside the window (on an axis the Hessian is checked off it:
    the lever weight is twice differentiable there, so differences of the gradient converge slowly)."""
    assert _jumps(path, 0.0)[1] > 0.1
    de, ds, dc = _jumps(path, 0.0, torsion=True)
    assert de < 1e-8 and ds < 1e-5 and dc < 1.0
    for s0 in (0.0, 0.03):
        z, x, _ = path(s0)
        p = torch.tensor(x, requires_grad=True)
        e = cr.pm7_hbond(z, p, TABLES, None, True, True).sum()
        g, = torch.autograd.grad(e, p, create_graph=True)
        v = torch.as_tensor(np.random.default_rng(1).normal(size=x.shape))
        hv, = torch.autograd.grad((g * v).sum(), p)
        h = 2e-6
        fd = np.zeros_like(x)
        for i in range(len(z)):
            for k in range(3):
                xp, xm = x.copy(), x.copy()
                xp[i, k] += h
                xm[i, k] -= h
                fd[i, k] = (_energy(z, xp, torsion=True) - _energy(z, xm, torsion=True)) / (2.0 * h)
        assert np.allclose(g.detach().numpy(), fd, rtol=0.0, atol=1e-6)
        if s0 == 0.0 and path is _linear_nitrile:
            continue

        def grad(y, z=z):
            q = torch.tensor(y, requires_grad=True)
            return torch.autograd.grad(cr.pm7_hbond(z, q, TABLES, None, True, True).sum(), q)[0].numpy()
        fd_hv = (grad(x + h * v.numpy()) - grad(x - h * v.numpy())) / (2.0 * h)
        assert np.allclose(hv.numpy(), fd_hv, rtol=0.0, atol=1e-4)


def test_hbond_torsion_option():
    """``hbond_torsion`` changes the heat of formation by the difference of the two forms, builds from a config,
    reaches the exported core and rejects unknown values."""
    from xnn.common.config import from_dict
    from xnn.common.models import build_model

    z, x, _ = _planar_amine(0.05)
    smooth = PM7(reuse_density=False, hbond_switch="smooth")(_graph(x, z))["heat_of_formation"]
    both = PM7(reuse_density=False, hbond_switch="smooth", hbond_torsion="smooth")(_graph(x, z))["heat_of_formation"]
    assert float(both - smooth) == pytest.approx(_energy(z, x, torsion=True) - _energy(z, x), abs=1e-8)
    built = build_model(from_dict({"model": {"name": "pm7", "hbond_switch": "smooth",
                                             "hbond_torsion": "smooth"}}).model)
    assert built.hbond_torsion == "smooth"
    core = built.export_core()
    edges, vec = torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 3))
    _, node_energy, _ = core.node_features_energy_charges(torch.tensor(z), edges, vec, torch.tensor(x),
                                                          torch.zeros((3, 3)), torch.zeros(3, dtype=torch.bool),
                                                          0.0, 1.0)
    built.reuse_density = False
    assert torch.allclose(node_energy, built(_graph(x, z))["node_energy"].detach(), rtol=0.0, atol=1e-10)
    with pytest.raises(ValueError, match="hbond_torsion"):
        PM7(hbond_torsion="soft")
