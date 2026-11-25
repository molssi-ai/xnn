"""Semiempirical methods (sqmnn): fit targets beyond energies and forces.

The energy of the highest occupied orbital (and Koopmans' ionization potential) equals the SCF orbital energy,
with and without a bond-order term, and its parameter gradients equal finite differences for closed and open
shells; the short-range H-H term has its defined form, switch and derivatives, is off unless its constants are
given or trained, and reaches the heat of formation and the exported core.
"""
import math

import numpy as np
import pytest
import torch

from xnn.common.data import structure_to_graph
from xnn.sqmnn.models import PM3, PM7
from xnn.sqmnn.models import corrections as cr
from xnn.sqmnn.models.scf import SCFResult, homo_orbitals

WATER = ([8, 1, 1], [[0.0, 0.0, 0.12], [0.0, 0.76, -0.47], [0.05, -0.75, -0.49]])
METHYL = ([6, 1, 1, 1], [[0, 0, 0.02], [1.08, 0, 0], [-0.54, 0.94, 0.01], [-0.55, -0.93, 0]])
ACETYLENE = ([1, 6, 6, 1], [[0, 0, -1.695], [0, 0, -0.635], [0, 0, 0.635], [0, 0, 1.695]])
TIGHT = dict(tol_density=1e-12, tol_energy=1e-13, reuse_density=False)


def _graph(pos, z, charge=0.0, mult=None, device="cpu"):
    s = {"pos": torch.as_tensor(np.asarray(pos, dtype=np.float64)), "atomic_numbers": np.asarray(z),
         "total_charge": float(charge)}
    if mult is not None:
        s["spin_multiplicity"] = float(mult)
    return structure_to_graph(s, 0.1, device=device)


@pytest.fixture(autouse=True)
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _highest_occupied(model):
    r = model.last_scf
    eps = torch.where(r.occupations > 0.0, r.energies, torch.full_like(r.energies, -1e30))
    return eps.amax(-1).amax(0)


@pytest.mark.parametrize("system,mult,kwargs", [
    (WATER, None, {}), (METHYL, 2.0, {}),
    (ACETYLENE, None, {"yne_switch": "off", "bond_order_terms": {"C-C": [11.09, 2.20, 2.65]}})])
def test_homo_energy_is_the_orbital_energy(system, mult, kwargs):
    """``homo_energy`` is the energy of the highest occupied orbital (of either spin), also with a bond-order
    term in the Fock matrix, and ``ionization_potential`` its negative."""
    z, pos = system
    model = PM7(**TIGHT, **kwargs)
    out = model(_graph(pos, z, mult=mult))
    assert torch.allclose(out["homo_energy"], _highest_occupied(model), rtol=0.0, atol=1e-8)
    assert torch.equal(out["ionization_potential"], -out["homo_energy"])
    assert 5.0 < float(out["ionization_potential"]) < 20.0


@pytest.mark.parametrize("system,mult,group,z_par", [(WATER, None, "UPP", 8), (METHYL, 2.0, "USS", 6),
                                                     (WATER, None, "BETAP", 8)])
def test_homo_energy_parameter_gradients(system, mult, group, z_par):
    """The parameter gradient of the HOMO energy (the SCF density's response included) equals central
    differences, for closed and open shells."""
    z, pos = system
    model = PM7(trainable=[group], **TIGHT).train()
    g = _graph(pos, z, mult=mult)
    e = model(g)["homo_energy"].sum()
    grad = torch.autograd.grad(e, model.params[group])[0][z_par]
    h, values = 1e-4, []
    with torch.no_grad():
        for s in (1.0, -1.0):
            model.params[group][z_par] += s * h
            values.append(float(model(g)["homo_energy"].sum()))
            model.params[group][z_par] -= s * h
    assert float(grad) == pytest.approx((values[0] - values[1]) / (2 * h), rel=1e-6, abs=1e-8)
    assert abs(float(grad)) > 1e-3


def test_highest_occupied_orbital_selection():
    """Of an unrestricted result the higher spin's highest occupied orbital is taken; a structure without
    electrons gets zeros."""
    C = torch.eye(3).expand(2, 2, 3, 3).clone()
    C[1, 0] = torch.tensor([[0.0, 1, 0], [1, 0, 0], [0, 0, 1]])
    energies = torch.tensor([[[-10.0, -5.0, 2.0], [-9.0, -1.0, 3.0]], [[-11.0, -4.0, 1.0], [-9.0, -2.0, 4.0]]])
    occ = torch.tensor([[[1.0, 1, 0], [0, 0, 0]], [[1.0, 1, 0], [0, 0, 0]]])
    r = SCFResult(occ, C, energies, occ, 1, torch.ones(2, dtype=torch.bool), torch.zeros(2))
    vec, spin = homo_orbitals(r)
    assert spin.tolist() == [1, 0]
    assert vec[0].tolist() == [1.0, 0.0, 0.0] and vec[1].tolist() == [0.0, 0.0, 0.0]


def _hh(r, constants=cr.HH_SHORT):
    pos = torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
    return float(cr.hh_short_correction([1, 1], pos, torch.tensor(constants)).sum())


def test_hh_short_form_and_switch():
    """The term is ``A exp(-b (R - R_c)^2)`` below 1.1 Angstrom, zero from 1.4 Angstrom on, the quintic switch
    between, and only for H-H pairs; energy, slope and curvature are continuous at both ends of the switch."""
    a, b, rc = cr.HH_SHORT
    for r in (0.6, 0.741, 1.0, 1.1):
        assert _hh(r) == pytest.approx(a * math.exp(-b * (r - rc) ** 2), rel=1e-14)
    t = (1.25 - 1.1) / 0.3
    assert _hh(1.25) == pytest.approx(a * math.exp(-b * (1.25 - rc) ** 2) * (1 - t ** 3 * (10 - 15 * t + 6 * t * t)))
    assert _hh(1.4) == 0.0 and _hh(2.5) == 0.0
    assert float(cr.hh_short_correction([1, 8], torch.tensor([[0.0, 0, 0], [0.9, 0, 0]]),
                                        torch.tensor(cr.HH_SHORT)).abs().sum()) == 0.0
    assert float(cr.hh_short_correction([1], torch.zeros(1, 3), torch.tensor(cr.HH_SHORT)).sum()) == 0.0

    def derivs(r):
        x = torch.tensor(r, requires_grad=True)
        pos = torch.stack([torch.zeros(3), torch.stack([x, torch.zeros(()), torch.zeros(())])])
        e = cr.hh_short_correction([1, 1], pos, torch.tensor(cr.HH_SHORT)).sum()
        if not e.requires_grad:
            return float(e), 0.0, 0.0
        g, = torch.autograd.grad(e, x, create_graph=True)
        c, = torch.autograd.grad(g, x)
        return float(e), float(g), float(c)
    for edge in (1.1, 1.4):
        (e0, s0, c0), (e1, s1, c1) = derivs(edge - 1e-9), derivs(edge + 1e-9)
        assert abs(e1 - e0) < 1e-6 and abs(s1 - s0) < 1e-5 and abs(c1 - c0) < 1e-3


def test_hh_short_derivatives():
    """Forces, Hessian-vector products and constant gradients of the term equal central differences."""
    z = [1, 1, 8, 1, 1]
    x = np.array([[0.0, 0, 0], [0.78, 0.05, 0.02], [3.0, 0, 0], [3.5, 0.8, 0], [2.0, 0.3, 0.9]])
    c = torch.tensor(cr.HH_SHORT, requires_grad=True)
    p = torch.tensor(x, requires_grad=True)
    e = cr.hh_short_correction(z, p, c).sum()
    g, = torch.autograd.grad(e, p, create_graph=True)
    v = torch.as_tensor(np.random.default_rng(0).normal(size=x.shape))
    hv, gc = torch.autograd.grad((g * v).sum(), (p, c), retain_graph=True)
    gconst, = torch.autograd.grad(e, c)

    def energy(y, consts=cr.HH_SHORT):
        return float(cr.hh_short_correction(z, torch.tensor(y), torch.tensor(consts)).sum())

    def grad(y):
        q = torch.tensor(y, requires_grad=True)
        return torch.autograd.grad(cr.hh_short_correction(z, q, torch.tensor(cr.HH_SHORT)).sum(), q)[0].numpy()
    h = 1e-6
    fd = np.zeros_like(x)
    for i in range(len(z)):
        for k in range(3):
            xp, xm = x.copy(), x.copy()
            xp[i, k] += h
            xm[i, k] -= h
            fd[i, k] = (energy(xp) - energy(xm)) / (2 * h)
    assert np.allclose(g.detach().numpy(), fd, rtol=0.0, atol=1e-6)
    assert np.allclose(hv.numpy(), (grad(x + h * v.numpy()) - grad(x - h * v.numpy())) / (2 * h), atol=1e-5)
    for j in range(3):
        cp, cm = list(cr.HH_SHORT), list(cr.HH_SHORT)
        hc = 1e-3 * cr.HH_SHORT[j]
        cp[j] += hc
        cm[j] -= hc
        assert float(gconst[j]) == pytest.approx((energy(x, cp) - energy(x, cm)) / (2 * hc), rel=1e-5)
    assert torch.isfinite(gc).all()


def test_hh_short_option():
    """The term is off by default and under ``trainable=["all"]``; given constants or training it switch it on,
    adding the term to the heat of formation and to the exported core."""
    from xnn.common.config import from_dict
    from xnn.common.models import build_model

    z, pos = [1, 1], [[0.0, 0, 0], [0.75, 0, 0]]
    plain = PM7(**TIGHT)
    assert not plain.hh_short and not PM7(trainable=["all"]).hh_short
    assert "HH_SHORT" not in PM7(trainable=["all"]).params
    on = PM7(correction_constants={"HH_SHORT": list(cr.HH_SHORT)}, **TIGHT)
    diff = on(_graph(pos, z))["heat_of_formation"] - plain(_graph(pos, z))["heat_of_formation"]
    assert float(diff) == pytest.approx(_hh(0.75), abs=1e-9)
    trained = PM3(trainable=["HH_SHORT"], **TIGHT)
    assert trained.hh_short and trained.params["HH_SHORT"].tolist() == list(cr.HH_SHORT)
    g, = torch.autograd.grad(trained(_graph(pos, z))["energy"].sum(), trained.params["HH_SHORT"])
    assert float(g[0]) == pytest.approx(_hh(0.75) / cr.HH_SHORT[0] / 23.060547830619029, rel=1e-9)
    built = build_model(from_dict({"model": {"name": "pm7", "correction_constants":
                                             {"HH_SHORT": [30.0, 10.0, 0.77]}}}).model)
    assert built.hh_short
    core = built.export_core()
    zc, xc = [1, 1, 8, 1, 1], torch.tensor([[0.0, 0, 0], [0.76, 0, 0], [3.0, 0, 0], [3.96, 0, 0], [2.76, 0.93, 0]])
    edges, vec = torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 3))
    _, node_energy, _ = core.node_features_energy_charges(torch.tensor(zc), edges, vec, xc, torch.zeros((3, 3)),
                                                          torch.zeros(3, dtype=torch.bool), 0.0, 1.0)
    built.reuse_density = False
    assert torch.allclose(node_energy, built(_graph(xc.numpy(), zc))["node_energy"].detach(), rtol=0.0, atol=1e-10)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_homo_energy_on_the_fast_path():
    """On a GPU the fast path's HOMO energies and their parameter gradients equal the CPU reference."""
    z, pos = WATER
    cpu = PM7(trainable=["UPP"], use_fast=False, **TIGHT).train()
    fast = PM7(trainable=["UPP"], **TIGHT).cuda().train()
    ref = cpu(_graph(pos, z))["homo_energy"]
    g_ref, = torch.autograd.grad(ref.sum(), cpu.params["UPP"])
    for _ in range(3):
        out = fast(_graph(pos, z, device="cuda"))["homo_energy"]
        g, = torch.autograd.grad(out.sum(), fast.params["UPP"])
        assert torch.allclose(out.cpu(), ref.detach(), rtol=0.0, atol=1e-9)
        assert torch.allclose(g.cpu(), g_ref, rtol=0.0, atol=1e-8)
