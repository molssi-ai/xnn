"""Semiempirical methods (sqmnn): exact second derivatives and trainable correction constants.

With the density's response to the positions, autograd Hessians equal central differences of the analytic
forces (closed and open shells), the Hessian-vector products of a batch equal those of its structures, and the
parameter gradients of a loss on mode curvatures equal finite differences in the parameters. The constants of
the acetylenic, dispersion and hydrogen-bond corrections train, their scripted forms agree with the host forms,
and the exported core carries their values. The tight-binding methods give exact Hessians (not differentiable
once more).
"""
import numpy as np
import pytest
import torch

from test_sqmnn_forms import ACETYLENE, _hbond_cluster, _scripted_branches
from xnn.common.data import collate, structure_to_graph
from xnn.common.models import ForceStressOutput, harmonic_analysis, hessian, hessian_vector_product
from xnn.sqmnn.models import GFN0, GFN1, GFN2, PM3, PM6, PM6ORG, PM7
from xnn.sqmnn.models import corrections as cr

WATER = (np.array([[0.0, 0.0, 0.12], [0.0, 0.76, -0.47], [0.05, -0.75, -0.49]]), [8, 1, 1])
FORMAMIDE = (np.array([[0, 0.42, 0], [1.2, 0.5, 0.02], [-0.72, -0.73, 0], [-0.55, 1.36, 0.01],
                       [-1.73, -0.69, 0.05], [-0.27, -1.63, -0.03]]), [6, 8, 7, 1, 1, 1])
METHYL = (np.array([[0, 0, 0.02], [1.08, 0, 0], [-0.54, 0.94, 0.01], [-0.55, -0.93, 0]]), [6, 1, 1, 1])
CO2 = (np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.17], [0.0, 0.0, -1.17]]), [6, 8, 8])
# an acetylene in the switching range of the acetylenic term next to a hydrogen-bonded water dimer
CLUSTER = ([1, 6, 6, 1, 8, 1, 1, 8, 1, 1],
           np.array([[0, 0, -1.695], [0, 0, -0.635], [0, 0, 0.635], [0, 0, 1.695], [4.0, 0, 0], [4.96, 0, 0.1],
                     [3.76, 0.93, 0], [6.85, 0.1, 0.2], [7.2, 0.95, 0.4], [7.25, -0.6, 0.75]]))


@pytest.fixture(autouse=True)
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _graph(pos, z, mult=None, device="cpu", charge=0.0):
    s = {"pos": torch.as_tensor(np.asarray(pos, dtype=np.float64)), "atomic_numbers": np.asarray(z),
         "total_charge": charge}
    if mult is not None:
        s["spin_multiplicity"] = float(mult)
    return structure_to_graph(s, 0.1, device=device)


def _fd_hessian(model, pos, z, mult, h):
    """Central differences of the analytic forces."""
    wrapped = ForceStressOutput(model).eval()
    n = 3 * len(z)
    out = np.zeros((n, n))
    for k in range(n):
        cols = []
        for s in (1, -1):
            p = np.array(pos, dtype=np.float64)
            p.reshape(-1)[k] += s * h
            cols.append(wrapped(_graph(p, z, mult))["forces"].detach().numpy().reshape(-1))
        out[:, k] = -(cols[0] - cols[1]) / (2 * h)
    return torch.as_tensor(out)


def _plain_hessian(model, graph):
    """The autograd Hessian of a density that does not follow the positions."""
    pos = graph.pos.detach().requires_grad_(True)
    graph.pos = pos
    (g,) = torch.autograd.grad(model(graph)["energy"].sum(), pos, create_graph=True)
    cols = [torch.autograd.grad(g.reshape(-1)[k], pos, retain_graph=True)[0].reshape(-1) for k in range(g.numel())]
    return torch.stack(cols, 1)


@pytest.mark.parametrize("cls,system,mult", [(PM7, FORMAMIDE, None), (PM6, METHYL, 2.0), (PM3, WATER, None)])
def test_exact_hessian_matches_finite_differences(cls, system, mult):
    """The Hessian equals Richardson-extrapolated central differences of the forces; without the density's
    response to the positions it does not."""
    pos, z = system
    model = cls(tol_density=1e-12, tol_energy=1e-13)
    (H,) = hessian(model, _graph(pos, z, mult))
    h = 2e-4
    fd = (4.0 * _fd_hessian(model, pos, z, mult, h / 2) - _fd_hessian(model, pos, z, mult, h)) / 3.0
    assert not model.position_response
    assert torch.allclose(H, H.T, rtol=0.0, atol=1e-9)
    assert torch.allclose(H, fd, rtol=0.0, atol=2e-7)
    plain = _plain_hessian(cls(tol_density=1e-12, tol_energy=1e-13), _graph(pos, z, mult))
    assert float((plain - H).abs().max()) > 1.0


def test_hessian_vector_products_of_a_batch():
    """The Hessians and Hessian-vector products of a batch (several directions at once) are those of its
    structures, from a model wrapped for forces as well."""
    model = PM7(tol_density=1e-12, tol_energy=1e-13, reuse_density=False)
    systems = [FORMAMIDE, WATER]
    batch = collate([_graph(*s) for s in systems])
    alone = [hessian(model, _graph(*s))[0] for s in systems]
    for H, ref in zip(hessian(ForceStressOutput(model), batch), alone):
        assert torch.allclose(H, ref, rtol=0.0, atol=1e-9)
    v = torch.linspace(-1.0, 1.0, 2 * 27).reshape(2, 9, 3)
    hv = hessian_vector_product(model, batch, v)
    assert hv.shape == v.shape
    for k in range(2):
        expect = torch.cat([alone[0] @ v[k, :6].reshape(-1), alone[1] @ v[k, 6:].reshape(-1)])
        assert torch.allclose(hv[k].reshape(-1), expect, rtol=0.0, atol=1e-9)
        single = hessian_vector_product(model, batch, v[k])
        assert torch.allclose(single, hv[k], rtol=0.0, atol=1e-12)


def _curvature_loss(model, pos, z, mult, U, create_graph):
    hu = hessian_vector_product(model, _graph(pos, z, mult), U, create_graph=create_graph)
    curv = (U * hu).sum((1, 2))
    w = torch.linspace(0.5, 1.5, curv.numel())
    return (w * curv).sum(), curv


@pytest.mark.parametrize("cls,system,mult,group,index", [(PM7, WATER, None, "GSS", 8), (PM7, WATER, None, "BETAS", 1),
                                                         (PM6, METHYL, 2.0, "USS", 6)])
def test_curvature_loss_parameter_gradients(cls, system, mult, group, index):
    """The parameter gradient of a loss on the curvatures along fixed normal modes equals finite differences;
    at the reference parameters the curvatures are the eigenvalues of the harmonic analysis."""
    pos, z = system
    settings = dict(tol_density=1e-12, tol_energy=1e-13)
    (H,) = hessian(cls(**settings), _graph(pos, z, mult))
    modes = harmonic_analysis(H, torch.as_tensor(pos), atomic_numbers=torch.as_tensor(z))
    U = modes["displacements"]
    model = cls(trainable=[group], **settings)
    loss, curv = _curvature_loss(model, pos, z, mult, U, True)
    assert torch.allclose(curv.detach(), modes["eigenvalues"], rtol=0.0, atol=1e-9)
    loss.backward()
    grad = float(model.params[group].grad[index])
    # the step keeps the differences above the noise of the linear-response solves (1e-10)
    h, fd = 3e-4, []
    for s in (1, -1):
        m = cls(trainable=[group], **settings)
        with torch.no_grad():
            m.params[group][index] += s * h
        fd.append(float(_curvature_loss(m, pos, z, mult, U, False)[0]))
    assert grad == pytest.approx((fd[0] - fd[1]) / (2 * h), rel=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_exact_hessian_on_gpu():
    """The Hessian of a model on a GPU (its SCF graphs captured after repeated evaluations) is the CPU one."""
    pos, z = FORMAMIDE
    (ref,) = hessian(PM7(tol_density=1e-12, tol_energy=1e-13), _graph(pos, z))
    model = PM7(tol_density=1e-12, tol_energy=1e-13).cuda()
    for _ in range(4):
        ForceStressOutput(model)(_graph(pos, z, device="cuda"))
        (H,) = hessian(model, _graph(pos, z, device="cuda"))
        assert torch.allclose(H.cpu(), ref, rtol=0.0, atol=1e-8)


def _all_terms():
    """The hydrogen-bond cluster (N and O ends, O...O pairs; charge +1) with an acetylene far from it."""
    z, pos = _hbond_cluster()
    return list(z) + ACETYLENE[0], np.vstack([pos, np.asarray(ACETYLENE[1]) + [0, 0, 14.0]])


def test_correction_constants_train():
    """The default constants of the acetylenic, dispersion and hydrogen-bond terms give the published energy;
    the gradients of an energy and force loss in every constant equal finite differences."""
    z, pos = _all_terms()
    groups = ["YNE", "DISP", "HBOND"]
    ref = ForceStressOutput(PM7(tol_density=1e-12, tol_energy=1e-13))(_graph(pos, z, charge=1.0))

    def loss(model):
        out = ForceStressOutput(model)(_graph(pos, z, charge=1.0))
        w = torch.linspace(0.3, 1.1, out["forces"].numel()).reshape(out["forces"].shape)
        return out, out["energy"].sum() + (out["forces"] * w).sum()

    model = PM7(trainable=groups, tol_density=1e-12, tol_energy=1e-13).train()
    out, value = loss(model)
    assert torch.allclose(out["energy"], ref["energy"], rtol=0.0, atol=1e-12)
    value.backward()
    for name in groups:
        grad = model.params[name].grad
        for i in range(3):
            h, fd = 1e-6, []
            for s in (1, -1):
                m = PM7(trainable=groups, tol_density=1e-12, tol_energy=1e-13)
                with torch.no_grad():
                    m.params[name][i] += s * h
                fd.append(float(loss(m)[1]))
            expect = (fd[0] - fd[1]) / (2 * h)
            assert abs(expect) > 1e-6, (name, i)
            assert float(grad[i]) == pytest.approx(expect, rel=1e-5, abs=1e-7), (name, i)
    assert {"YNE", "DISP", "HBOND"} <= set(PM7(trainable=["all"]).trainable_groups)
    assert set(PM6ORG(trainable=["all"]).trainable_groups) & {"DISP", "HBOND"} == set()
    assert "YNE" not in PM3(trainable=["all"]).trainable_groups
    with pytest.raises(ValueError, match="unknown trainable"):
        PM3(trainable=["YNE"])
    assert not any(k.startswith("constants_") for k in PM7().state_dict())


def _changed_constants(model):
    with torch.no_grad():
        model.constants_YNE.copy_(torch.tensor([10.0, 1.15, 1.40]))
        model.constants_DISP.mul_(torch.tensor([1.2, 0.9, 1.05]))
        model.constants_HBOND.mul_(torch.tensor([0.8, 1.3, 1.5]))
    return model


def test_changed_constants_reach_the_scripted_forms_and_the_export(monkeypatch):
    """Changed constants enter the energies of the corrections: the TorchScript branches (run eagerly) give the
    host forms' energies and gradients, and the exported core gives the eager model's energies."""
    z, pos = _all_terms()
    model = _changed_constants(PM7())
    consts = [model.correction_constants(n) for n in ("YNE", "DISP", "HBOND")]
    tables = cr.correction_tables(torch.device("cpu"))

    def terms():
        p = torch.tensor(pos, requires_grad=True)
        e = cr.structure_corrections(2, z, p, tables, 3.1595, *consts)
        return e.detach(), torch.autograd.grad(e.sum(), p)[0]

    host = terms()
    default = cr.structure_corrections(2, z, torch.tensor(pos), tables, 3.1595)
    assert float((host[0] - default).abs().max()) > 1e-3
    with monkeypatch.context() as m:
        _scripted_branches(m, cr)
        scripted = terms()
    assert torch.allclose(scripted[0], host[0], rtol=1e-12, atol=1e-12)
    assert torch.allclose(scripted[1], host[1], rtol=1e-10, atol=1e-10)

    zc, xc = CLUSTER
    core = model.export_core()
    edges, vec = torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 3))
    _, node_energy, _ = core.node_features_energy_charges(torch.tensor(zc), edges, vec, torch.tensor(xc),
                                                          torch.zeros((3, 3)), torch.zeros(3, dtype=torch.bool),
                                                          0.0, 1.0)
    eager = _changed_constants(PM7(reuse_density=False))(_graph(xc, zc))
    assert float((eager["energy"] - PM7(reuse_density=False)(_graph(xc, zc))["energy"]).abs()) > 1e-4
    assert torch.allclose(node_energy, eager["node_energy"].detach(), rtol=0.0, atol=1e-10)


def test_trained_dispersion_and_hydrogen_bonds_are_twice_differentiable():
    """With trainable constants the dispersion and hydrogen-bond terms keep their position Hessian (the force
    loss differentiates them twice) and their constant gradients are linear where the energy is."""
    z, pos = _hbond_cluster()
    tables = cr.correction_tables(torch.device("cpu"))
    bonds = cr.bond_topology(z, torch.tensor(pos), tables)
    for fn, default in ((lambda p, c: cr.pm7_dispersion(z, p, bonds, tables, constants=c), cr.DISPERSION),
                        (lambda p, c: cr.pm7_hbond(z, p, tables, c), cr.HBOND)):
        c = torch.tensor(default, requires_grad=True)
        p = torch.tensor(pos, requires_grad=True)
        e = fn(p, c).sum()
        assert float(e) == pytest.approx(float(fn(torch.tensor(pos), None).sum()), abs=1e-10)
        (g,) = torch.autograd.grad(e, p, create_graph=True)
        (hc,) = torch.autograd.grad((g * g).sum(), c)
        frozen = torch.tensor(default)
        h, fd = 1e-6, []
        for s in (1, -1):
            pp = torch.tensor(pos, requires_grad=True)
            cc = frozen.clone()
            cc[0] += s * h
            (gg,) = torch.autograd.grad(fn(pp, cc).sum(), pp)
            fd.append(float((gg * gg).sum()))
        assert float(hc[0]) == pytest.approx((fd[0] - fd[1]) / (2 * h), rel=1e-6)
    # a molecule without dispersion pairs
    lone = cr.pm7_dispersion([8], torch.zeros(1, 3), [[]], tables, constants=torch.tensor(cr.DISPERSION,
                                                                                          requires_grad=True))
    assert float(lone.abs().sum()) == 0.0


def _tight(cls, **options):
    """A tight-binding model whose cycle converges far below the default (differences of its forces)."""
    model = cls(accuracy=1e-4, **options)
    model.charge_convergence = 1e-8
    return model


@pytest.mark.parametrize("cls,system,mult", [(GFN0, FORMAMIDE, None), (GFN1, CO2, None), (GFN2, WATER, None),
                                             (GFN2, METHYL, 2.0)])
def test_tight_binding_hessian_matches_finite_differences(cls, system, mult):
    """The Hessian of the tight-binding methods (the density follows the positions through the self-consistent
    charges and moments) equals Richardson-extrapolated central differences of the forces, for a linear molecule
    and an open shell too; without the density response it does not."""
    pos, z = system
    model = _tight(cls)
    (H,) = hessian(model, _graph(pos, z, mult))
    h = 2e-4
    fd = (4.0 * _fd_hessian(model, pos, z, mult, h / 2) - _fd_hessian(model, pos, z, mult, h)) / 3.0
    assert torch.allclose(H, H.T, rtol=0.0, atol=1e-8)
    assert torch.allclose(H, fd, rtol=0.0, atol=1e-6)
    assert float((_plain_hessian(_tight(cls), _graph(pos, z, mult)) - H).abs().max()) > 1.0


def test_tight_binding_energy_gradient_and_limits():
    """The gradient of ``energy_gradient`` is the negative force and leaves no state behind; a wrapped model
    takes the plain autograd Hessian with a warning, and the Hessians cannot be differentiated once more."""
    pos, z = WATER
    model = _tight(GFN0)
    x, g = model.energy_gradient(_graph(pos, z))
    forces = ForceStressOutput(model)(_graph(pos, z))["forces"]
    assert torch.allclose(g.detach(), -forces, rtol=0.0, atol=1e-10)
    assert "_density_positions" not in model.__dict__ and "_state" not in model.__dict__

    class Wrapped(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, data):
            return self.inner(data)

    with pytest.warns(RuntimeWarning, match="approximate"):
        hessian(Wrapped(_tight(GFN1)), _graph(pos, z))
    with pytest.raises(NotImplementedError, match="create_graph=False"):
        hessian_vector_product(model, _graph(pos, z), torch.ones(3, 3), create_graph=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_tight_binding_hessian_on_gpu():
    """The Hessian of GFN2-xTB on a GPU (small batches otherwise run on the host) is the CPU one."""
    pos, z = FORMAMIDE
    (ref,) = hessian(_tight(GFN2), _graph(pos, z))
    (H,) = hessian(_tight(GFN2).cuda(), _graph(pos, z, device="cuda"))
    assert torch.allclose(H.cpu(), ref, rtol=0.0, atol=1e-8)
