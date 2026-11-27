"""NDDO methods (sqmnn): the bond-order and resonance-overlap terms.

The bond-order term's Fock matrices and response kernel are the derivatives of its energy; its energy gradient in
the pair energies is the switch times the cutoff; forces, Hessians, open shells and the parameter gradients of
energy, force and curvature losses equal finite differences; batches, buckets, the static SCF and response forms
and the exported core give the reference's numbers. The resonance-overlap term follows its formula. The options
build from configs, refuse bad input, and ``yne_switch="off"`` removes the acetylenic term.
"""
import math
import types

import numpy as np
import pytest
import torch

from xnn.common.data import collate, structure_to_graph
from xnn.common.models import ForceStressOutput, harmonic_analysis, hessian, hessian_vector_product
from xnn.sqmnn.models import PM3, PM6, PM7, bond_order, core, nddo, scf

KCAL = 23.06054783061903
# H-C#C-C#N, slightly bent
CYANOACETYLENE = ([1, 6, 6, 6, 7], np.array([[0.03, 0.0, -1.062], [0.0, 0.0, 0.0], [0.0, 0.01, 1.205],
                                             [0.0, 0.015, 2.58], [-0.02, 0.0, 3.74]]))
# the vinyl radical (doublet)
VINYL = ([6, 6, 1, 1, 1], np.array([[0.0, 0.0, 0.0], [1.315, 0.0, 0.01], [-0.56, 0.93, 0.0], [-0.55, -0.94, 0.02],
                                    [2.02, 0.86, -0.01]]))
WATER = ([8, 1, 1], np.array([[0.0, 0.0, 0.12], [0.0, 0.76, -0.47], [0.05, -0.75, -0.49]]))
ACETYLENE = ([1, 6, 6, 1], np.array([[0, 0, -1.695], [0, 0, -0.635], [0, 0, 0.635], [0, 0, 1.695]]))
TIGHT = dict(tol_density=1e-12, tol_energy=1e-13)


@pytest.fixture(autouse=True)
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _graph(system, mult=None, device="cpu"):
    z, pos = system
    s = {"pos": torch.as_tensor(np.asarray(pos, dtype=np.float64)), "atomic_numbers": np.asarray(z),
         "total_charge": 0.0}
    if mult is not None:
        s["spin_multiplicity"] = float(mult)
    return structure_to_graph(s, 0.1, device=device)


def _wiberg(model, z):
    """Wiberg indices of every atom pair of the last single-molecule evaluation (atoms' orbitals in order)."""
    P = model.last_scf.densities.sum(0)[0].detach()
    off = np.cumsum([0] + [1 if a == 1 else 4 for a in z])
    W = np.zeros((len(z), len(z)))
    for i in range(len(z)):
        for j in range(len(z)):
            if i != j:
                W[i, j] = float((P[off[i]:off[i + 1], off[j]:off[j + 1]] ** 2).sum())
    return W


def _window_terms(system, pairs, mult=None, ramp=(1.0, 1.5)):
    """Bond-order terms with windows around the plain method's bond indices of the atom ``pairs`` (one per
    element pair: energy, the index inside the window, the distance inside the cutoff's ramp)."""
    z, _ = system
    plain = PM7(**TIGHT)
    plain(_graph(system, mult))
    W = _wiberg(plain, z)
    symbol = {1: "H", 6: "C", 7: "N", 8: "O"}
    terms = {}
    for (i, j), c in pairs.items():
        w = float(W[i, j])
        terms[f"{symbol[z[i]]}-{symbol[z[j]]}"] = [float(c), w - 0.35, w + 0.3, *[float(r) for r in ramp]]
    return terms


@pytest.fixture(scope="module")
def cyano_terms():
    return _window_terms(CYANOACETYLENE, {(1, 2): 11.0, (3, 4): -8.0})


RESONANCE = {"C-C": [-0.1, 0.05], "C-N": -0.02}


def _model(terms, **options):
    return PM7(yne_switch="off", bond_order_terms=terms, resonance_overlap=RESONANCE, **TIGHT, **options)


def _fd_forces(model, system, mult, h=1e-4):
    z, pos = system
    out = np.zeros(pos.size)
    for k in range(pos.size):
        e = []
        for s in (1, -1):
            p = pos.copy()
            p.reshape(-1)[k] += s * h
            e.append(float(model(_graph((z, p), mult))["energy"].sum()))
        out[k] = -(e[0] - e[1]) / (2 * h)
    return torch.as_tensor(out.reshape(pos.shape))


def _fd_hessian(model, system, mult, h):
    """Central differences of the analytic forces."""
    z, pos = system
    wrapped = ForceStressOutput(model).eval()
    n = pos.size
    out = np.zeros((n, n))
    for k in range(n):
        cols = []
        for s in (1, -1):
            p = pos.copy()
            p.reshape(-1)[k] += s * h
            cols.append(wrapped(_graph((z, p), mult))["forces"].detach().numpy().reshape(-1))
        out[:, k] = -(cols[0] - cols[1]) / (2 * h)
    return torch.as_tensor(out)


def _recorded(monkeypatch, model, system, mult=None):
    """The bond-order term, layout and SCF result of a model's evaluation."""
    calls = []

    def record(*args, **kwargs):
        result = scf.run_scf(*args, **kwargs)
        calls.append((args, kwargs, result))
        return result
    with monkeypatch.context() as m:
        m.setattr(nddo, "run_scf", record)
        model(_graph(system, mult))
    args, kwargs, result = calls[0]
    return kwargs["bond_order"], args, result


def test_smooth_step_and_cutoff():
    """The quintic step is 0 and 1 outside its range with vanishing first and second derivatives at the ends,
    and its derivatives are those of autograd; the cutoff is its mirror image."""
    t = torch.tensor([-0.5, 0.0, 0.3, 0.5, 1.0, 1.7], requires_grad=True)
    S, S1, S2 = bond_order.smooth_step(t)
    assert S.tolist()[:2] == [0.0, 0.0] and S.tolist()[4:] == [1.0, 1.0]
    assert float(S[3]) == pytest.approx(0.5, abs=1e-15)
    assert S1[[0, 1, 4, 5]].abs().max() == 0.0 and S2[[0, 1, 4, 5]].abs().max() == 0.0
    (g,) = torch.autograd.grad(S.sum(), t, create_graph=True)
    assert torch.allclose(g, S1, rtol=0.0, atol=1e-14)
    (g2,) = torch.autograd.grad(g.sum(), t)
    assert torch.allclose(g2, S2, rtol=0.0, atol=1e-13)
    r = torch.tensor([1.3, 1.4, 1.525, 1.65, 1.8])
    g = bond_order.cutoff(r, torch.tensor(1.40), torch.tensor(0.25))
    assert g.tolist()[:2] == [1.0, 1.0] and g.tolist()[3:] == [0.0, 0.0]
    assert float(g[2]) == pytest.approx(0.5, abs=1e-15)


def test_fock_matrices_and_kernel_are_derivatives_of_the_energy(monkeypatch, cyano_terms):
    """At the converged density the term's Fock matrices are the symmetrized gradient of its energy and its
    kernel the derivative of the Fock matrices along a density change; the pairs sit inside the window."""
    bo, args, result = _recorded(monkeypatch, _model(cyano_terms), CYANOACETYLENE)
    P = bond_order.total_density(result.densities).clone().requires_grad_(True)
    _, S, S1 = bond_order.switch(bo, P)
    _, a1, a2 = bond_order.linearized(bo, P.detach())
    assert bool(((S1 > 0) & (a2.abs() > 0) & (bo.coef.abs() > 0)).any())
    F, E = bond_order.fock(bo, P)
    (g,) = torch.autograd.grad(E.sum(), P)
    assert torch.allclose(F, 0.5 * (g + g.transpose(-1, -2)), rtol=0.0, atol=1e-13)
    assert float(E.sum()) == pytest.approx(float(bond_order.pair_energies(bo, P).sum()), abs=1e-14)
    X = torch.tensor(np.random.default_rng(0).normal(size=P.shape))
    X = X + X.transpose(-1, -2)
    _, jvp = torch.autograd.functional.jvp(lambda p: bond_order.fock(bo, p)[0], P.detach(), X)
    assert torch.allclose(bond_order.kernel(bo, P.detach(), X), jvp, rtol=0.0, atol=1e-12)
    n = len(CYANOACETYLENE[0])
    atoms = bond_order.atom_energies(bo, P.detach(), n)
    assert float(atoms.sum()) == pytest.approx(float(E.sum()), abs=1e-14)
    # the NumPy form of the iterations on the CPU gives the same numbers
    F_h, E_h = bond_order.fock_host(bo, P.detach())
    assert torch.allclose(F_h, F.detach(), rtol=0.0, atol=1e-14) and torch.allclose(E_h, E.detach(), atol=1e-14)
    # the SCF's Fock matrices and energies: the NumPy form of the CPU iterations and the PyTorch form agree
    layout, H, W1, W2 = args[:4]
    W1p, W2p = scf.pack_one_center(W1.detach()), scf.pack_pairs(layout, W2.detach())
    dens = result.densities
    host = scf._fock_energy(layout, H.detach(), W1p, W2p, dens, False, bo)
    torch_form = scf._fock_energy(layout, H.detach(), W1p, W2p, dens.clone().requires_grad_(True), False, bo)
    assert all(torch.allclose(a, b.detach(), rtol=0.0, atol=1e-12) for a, b in zip(host, torch_form))
    plateau = bond_order.with_values(bo, bo.coef, bo.lower - 10.0, bo.width)       # every index above its window
    F_p, E_p = bond_order.fock_host(plateau, P.detach())
    assert F_p is None and torch.allclose(E_p, bond_order.fock(plateau, P.detach())[1], rtol=0.0, atol=1e-14)
    assert float(bond_order.fock(plateau, P.detach())[0].abs().max()) == 0.0
    lin = bond_order.linearized_host(bo, P.detach())
    assert torch.allclose(bond_order.kernel_host(bo, *lin, X), bond_order.kernel_at(bo, *bond_order.linearized(
        bo, P.detach()), X), rtol=0.0, atol=1e-13)


def test_energy_gradient_in_the_pair_energies_is_switch_times_cutoff(cyano_terms):
    """The term is variational: the energy's gradient in each element pair's energy is the sum of S(t) g(R)
    over its atom pairs (in eV per kcal/mol), with the bond index of the converged density."""
    model = _model(cyano_terms, trainable=["BOND_ORDER"]).eval()
    z, pos = CYANOACETYLENE
    model(_graph(CYANOACETYLENE))["energy"].sum().backward()
    W = _wiberg(model, z)

    def step(t):
        t = min(max(t, 0.0), 1.0)
        return t ** 3 * (10 - 15 * t + 6 * t * t)

    expect = []
    for key, (_, lo, hi, r_on, r_off) in cyano_terms.items():
        pair = (6, 6) if key == "C-C" else (6, 7)
        total = 0.0
        for i in range(len(z)):
            for j in range(i + 1, len(z)):
                if sorted((z[i], z[j])) == list(pair):
                    r = float(np.linalg.norm(pos[i] - pos[j]))
                    total += step((W[i, j] - lo) / (hi - lo)) * (1.0 - step((r - r_on) / (r_off - r_on)))
        expect.append(total / KCAL)
    assert min(abs(e) for e in expect) > 1e-3
    assert model.params["BOND_ORDER"].grad.tolist() == pytest.approx(expect, abs=1e-12)


@pytest.mark.parametrize("closed", [True, False])
def test_forces_and_hessian_match_finite_differences(closed, cyano_terms):
    """Forces equal central differences of the energy and the exact Hessian equals Richardson-extrapolated
    central differences of the forces, with pairs inside the windows and the cutoff ramps (closed shell with both
    terms; the vinyl radical, unrestricted)."""
    if closed:
        system, mult, model = CYANOACETYLENE, None, _model(cyano_terms)
    else:
        system, mult = VINYL, 2.0
        model = PM7(yne_switch="off", bond_order_terms=_window_terms(VINYL, {(0, 1): 9.0}, 2.0, (1.1, 1.5)),
                    **TIGHT)
    forces = ForceStressOutput(model).eval()(_graph(system, mult))["forces"].detach()
    assert torch.allclose(forces, _fd_forces(model, system, mult), rtol=0.0, atol=2e-6)
    (H,) = hessian(model, _graph(system, mult))
    h = 1e-4
    fd = (4.0 * _fd_hessian(model, system, mult, h / 2) - _fd_hessian(model, system, mult, h)) / 3.0
    assert torch.allclose(H, H.T, rtol=0.0, atol=1e-9)
    assert torch.allclose(H, fd, rtol=0.0, atol=1e-5)


@pytest.mark.parametrize("closed", [True, False])
def test_energy_and_force_loss_parameter_gradients(closed, cyano_terms):
    """The gradients of an energy and force loss in every trained value of the terms (through the density's
    response) equal finite differences."""
    if closed:
        system, mult = CYANOACETYLENE, None
        groups = ["BOND_ORDER", "BOND_ORDER_WINDOW", "BOND_ORDER_CUTOFF", "RESONANCE_OVERLAP"]

        def make():
            return _model(cyano_terms, trainable=groups)
    else:
        system, mult = VINYL, 2.0
        groups = ["BOND_ORDER", "BOND_ORDER_WINDOW", "BOND_ORDER_CUTOFF"]
        terms = _window_terms(VINYL, {(0, 1): 9.0}, 2.0, (1.1, 1.5))

        def make():
            return PM7(yne_switch="off", bond_order_terms=terms, trainable=groups, **TIGHT)

    def loss(model):
        out = ForceStressOutput(model)(_graph(system, mult))
        w = torch.linspace(0.3, 1.1, out["forces"].numel()).reshape(out["forces"].shape)
        return out["energy"].sum() + (out["forces"] * w).sum()

    model = make().train()
    loss(model).backward()
    for name in groups:
        grad = model.params[name].grad.reshape(-1)
        for i in range(grad.numel()):
            h, fd = 1e-6, []
            for s in (1, -1):
                m = make()
                with torch.no_grad():
                    m.params[name].view(-1)[i] += s * h
                fd.append(float(loss(m)))
            expect = (fd[0] - fd[1]) / (2 * h)
            assert float(grad[i]) == pytest.approx(expect, rel=1e-5, abs=1e-7), (name, i)


def test_curvature_loss_parameter_gradients(cyano_terms):
    """The gradient of a loss on mode curvatures in the pair energies and the windows equals finite
    differences."""
    settings = dict(yne_switch="off", bond_order_terms=cyano_terms, **TIGHT)
    pos = torch.as_tensor(CYANOACETYLENE[1])
    (H,) = hessian(PM7(**settings), _graph(CYANOACETYLENE))
    U = harmonic_analysis(H, pos, atomic_numbers=torch.as_tensor(CYANOACETYLENE[0]))["displacements"]
    w = torch.linspace(0.5, 1.5, U.shape[0])

    def loss(model, create_graph):
        hu = hessian_vector_product(model, _graph(CYANOACETYLENE), U, create_graph=create_graph)
        return (w * (U * hu).sum((1, 2))).sum()

    groups = ["BOND_ORDER", "BOND_ORDER_WINDOW"]
    model = PM7(trainable=groups, **settings)
    loss(model, True).backward()
    for name, index in (("BOND_ORDER", 0), ("BOND_ORDER", 1), ("BOND_ORDER_WINDOW", 0)):
        grad = float(model.params[name].grad.reshape(-1)[index])
        h, fd = 3e-4, []
        for s in (1, -1):
            m = PM7(trainable=groups, **settings)
            with torch.no_grad():
                m.params[name].view(-1)[index] += s * h
            fd.append(float(loss(m, False)))
        expect = (fd[0] - fd[1]) / (2 * h)
        assert abs(expect) > 1e-6, (name, index)
        assert grad == pytest.approx(expect, rel=1e-5), (name, index)


def test_resonance_overlap_formula():
    """The resonance change of a pair along z with diagonal p-p overlaps is the pi part on x and y and the sigma
    part on z; without a sigma factor only the pi part."""
    e = torch.tensor([[0.0, 0.0, 1.0]])
    S_pp = torch.diag(torch.tensor([0.2, 0.2, -0.3])).unsqueeze(0)
    d = core.resonance_overlap(S_pp, torch.tensor([-8.0]), torch.tensor([-6.0]), e, torch.tensor([[0.5, 0.25]]))
    assert torch.allclose(d[0], torch.diag(torch.tensor([-7.0 * 0.5 * 0.04, -7.0 * 0.5 * 0.04, -7.0 * 0.25 * 0.09])),
                          rtol=0.0, atol=1e-15)
    d = core.resonance_overlap(S_pp, torch.tensor([-8.0]), torch.tensor([-6.0]), e, torch.tensor([[0.5]]))
    assert float(d[0, 2, 2]) == 0.0


def test_terms_cost_nothing_without_pairs_and_zero_factors_change_nothing(cyano_terms):
    """A molecule without pairs of the terms gives the plain method's numbers exactly; zero factors of the
    resonance term give them to round-off."""
    plain = PM7(yne_switch="off", **TIGHT)
    for system in (WATER,):
        a, b = (ForceStressOutput(m).eval()(_graph(system)) for m in (_model(cyano_terms), plain))
        assert torch.equal(a["energy"], b["energy"]) and torch.equal(a["forces"], b["forces"])
    zero = PM7(yne_switch="off", resonance_overlap={"C-C": [0.0, 0.0], "C-N": 0.0}, **TIGHT)
    a, b = (ForceStressOutput(m).eval()(_graph(CYANOACETYLENE)) for m in (zero, plain))
    assert torch.allclose(a["energy"], b["energy"], rtol=0.0, atol=1e-12)
    assert torch.allclose(a["forces"], b["forces"], rtol=0.0, atol=1e-10)
    changed = ForceStressOutput(_model(cyano_terms)).eval()(_graph(CYANOACETYLENE))
    assert float((changed["energy"] - b["energy"]).abs()) > 1e-3


@pytest.fixture
def buckets(monkeypatch):
    """Buckets on the CPU too (without graphs there)."""
    monkeypatch.setitem(nddo.BUCKETS, "cpu", True)


def _outputs(model, data, train=False):
    model.train(train)
    out = ForceStressOutput(model).train(train)(data)
    values = [out[k].detach() for k in ("energy", "node_energy", "forces", "charges", "electronic_energy")]
    if train:
        w = torch.linspace(0.3, 1.1, out["forces"].numel(), device=data.pos.device).reshape(out["forces"].shape)
        model.zero_grad()
        (1e-3 * out["energy"].pow(2).sum() + 1e-2 * (out["forces"] * w).pow(2).sum()).backward()
        values += [p.grad.detach().clone() for p in model.parameters()]
    return values


def _close(a, b, tol):
    for x, y in zip(a, b):
        assert x.shape == y.shape
        assert torch.allclose(x.cpu(), y.cpu(), rtol=0.0, atol=tol * max(1.0, float(y.abs().max())))


def test_batches_and_buckets_match_single_molecules(cyano_terms, buckets, monkeypatch):
    """A batch (with a molecule without pairs) gives each molecule's energies and forces; padded to a bucket
    (every pair of the bucket carrying the terms) it gives the batch's outputs and parameter gradients."""
    systems = [CYANOACETYLENE, WATER, ACETYLENE]
    groups = ["BOND_ORDER", "BOND_ORDER_WINDOW", "RESONANCE_OVERLAP"]
    ref = _model(cyano_terms, trainable=groups, use_fast=False)
    batch = _outputs(ref, collate([_graph(s) for s in systems]), True)
    for k, system in enumerate(systems):
        one = _outputs(_model(cyano_terms, use_fast=False), _graph(system))
        assert float(one[0]) == pytest.approx(float(batch[0][k]), abs=1e-9)
    padded = _model(cyano_terms, trainable=groups)
    _close(_outputs(padded, collate([_graph(s) for s in systems]), True), batch, 1e-9)
    # the pairs kept per composition: the oldest leaves when the cache is full
    monkeypatch.setattr(nddo, "PADDED_CAPACITY", 1)
    model = _model(cyano_terms, use_fast=False)
    for system in (CYANOACETYLENE, ACETYLENE):
        model(_graph(system))
    assert len(model._bo_pairs) == 1


def test_static_scf_and_response_forms_carry_the_term(monkeypatch, cyano_terms):
    """The SCF loop and the coupled-perturbed iterations on fixed buffers, and the loads of graph buffers, carry
    the term: the static forms give the launched forms' solutions."""
    bo, args, result = _recorded(monkeypatch, _model(cyano_terms), CYANOACETYLENE)
    layout, H, W1, W2, n_alpha, n_beta, uhf, guess = args[:8]
    W1p, W2p = scf.pack_one_center(W1.detach()), scf.pack_pairs(layout, W2.detach())
    gP = torch.tensor(np.random.default_rng(3).normal(size=result.densities.shape))
    loops = [scf.CPHFLoop(layout, W1p, W2p, result.orbitals, result.energies, result.occupations, gP, uhf, static,
                          bo) for static in (False, True)]
    for loop in loops:
        loop.start()
        for _ in range(60):
            loop.step()
    assert float(loops[0].res) < 1e-9
    assert torch.allclose(loops[1].u, loops[0].u, rtol=0.0, atol=1e-12)
    plain = scf.CPHFLoop(layout, W1p, W2p, result.orbitals, result.energies, result.occupations, gP, uhf)
    plain.start()
    for _ in range(60):
        plain.step()
    assert float((plain.u - loops[0].u).abs().max()) > 1e-6

    copy = bond_order.cloned(bo)
    with torch.no_grad():
        copy.coef.zero_()
    graphs = object.__new__(scf.SCFGraphs)
    P, n_occ, weight, filled, blocks = scf.scf_start(guess, n_alpha, n_beta, uhf, None)
    loop = scf.SCFLoop(layout, H.detach(), W1p, W2p, n_occ, weight, uhf, 8, filled, blocks, True, copy)
    graphs.layout, graphs.loop, graphs.H, graphs.W1, graphs.W2 = None, loop, loop.H, loop.W1, loop.W2
    graphs.bond_order = copy
    graphs.load(H.detach(), W1p, W2p, P, bond_order=bo)
    assert torch.equal(copy.coef, bo.coef.detach())
    cphf = object.__new__(scf.CPHFGraphs)
    cphf.layout, cphf.W1, cphf.W2, cphf.loop = None, loops[1].W1, loops[1].W2, loops[1]
    cphf.bond_order = bond_order.cloned(bo)
    with torch.no_grad():
        cphf.bond_order.lower.add_(1.0)
    cphf.load(W1p, W2p, result.orbitals, result.energies, result.occupations, gP, bond_order=bo)
    assert torch.equal(cphf.bond_order.lower, bo.lower)


def test_torchscript_export_matches_eager(cyano_terms, tmp_path):
    """The exported TorchScript potential with both terms gives the eager model's energy, forces and charges."""
    from xnn.common.deploy import export_torchscript_potential
    z = CYANOACETYLENE[0] + WATER[0]
    pos = np.vstack([CYANOACETYLENE[1], WATER[1] + [4.0, 0.5, 0.2]])
    eager = _model(cyano_terms, reuse_density=False).eval()
    path = export_torchscript_potential(_model(cyano_terms).eval(), cutoff=10.0, path=str(tmp_path / "m.pt"))
    scripted = torch.jit.load(path)
    with torch.jit.optimized_execution(False):
        for step in range(2):
            p = torch.tensor(pos + 0.01 * step)
            out = scripted(p, torch.tensor(z))
            ref = ForceStressOutput(eager).eval()(_graph((z, p.numpy())))
            assert abs(float(out["energy"]) - float(ref["energy"].sum())) < 1e-9
            assert torch.allclose(out["forces"], ref["forces"], atol=1e-7, rtol=0.0)
            assert torch.allclose(out["charges"], ref["charges"], atol=1e-8, rtol=0.0)
    plain = ForceStressOutput(PM7(yne_switch="off", reuse_density=False)).eval()(_graph((z, pos)))
    assert abs(float(plain["energy"].sum()) - float(ref["energy"].sum())) > 1e-4
    # the core run eagerly (the lines TorchScript hides from coverage) gives the eager model's energies
    core = _model(cyano_terms).export_core()
    _, node_energy, charges = core.node_features_energy_charges(
        torch.tensor(z), torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 3)), torch.tensor(pos),
        torch.zeros((3, 3)), torch.zeros(3, dtype=torch.bool), 0.0, 1.0)
    eager_out = eager(_graph((z, pos)))
    assert torch.allclose(node_energy, eager_out["node_energy"].detach(), rtol=0.0, atol=1e-10)
    assert torch.allclose(charges, eager_out["charges"].detach(), rtol=0.0, atol=1e-8)


def test_options_configs_and_yne_off(cyano_terms):
    """The options build from a config, offer their groups for training (also through "all"), keep the state
    dict of an untrained model, reject bad input, and ``yne_switch="off"`` removes the acetylenic term."""
    from xnn.common.config import from_dict
    from xnn.common.models import build_model

    built = build_model(from_dict({"model": {"name": "pm7", "yne_switch": "off", "bond_order_terms": cyano_terms,
                                             "bond_order_cutoff": [1.3, 1.7],
                                             "resonance_overlap": RESONANCE}}).model)
    assert built.bond_order_terms == cyano_terms and built.bond_order_cutoff == [1.3, 1.7]
    assert built.term_values("BOND_ORDER").tolist() == [11.0, -8.0]
    lo, hi = cyano_terms["C-C"][1:3]
    assert built.term_values("BOND_ORDER_WINDOW")[0].tolist() == pytest.approx([lo, math.log(hi - lo)])
    assert built.term_values("RESONANCE_OVERLAP").tolist() == [[-0.1, 0.05], [-0.02, 0.0]]
    assert not any(k.startswith("terms_") for k in built.state_dict())
    everything = PM7(trainable=["all"], yne_switch="off", bond_order_terms={"C-C": [1.0, 2.2, 2.6]},
                     resonance_overlap={"C-N": 0.1})
    assert {"BOND_ORDER", "BOND_ORDER_WINDOW", "BOND_ORDER_CUTOFF", "RESONANCE_OVERLAP"} <= set(everything.params)
    assert "YNE" not in everything.params
    assert everything.term_values("BOND_ORDER_CUTOFF")[0].tolist() == pytest.approx([1.4, math.log(0.25)])
    assert everything.term_values("RESONANCE_OVERLAP").shape == (1, 1)

    graph = _graph(ACETYLENE)                                # C-C 1.27 Angstrom: 7.40625 kcal/mol of the switch
    hof = PM6(reuse_density=False)(graph)["heat_of_formation"]
    off = PM6(reuse_density=False, yne_switch="off")(graph)["heat_of_formation"]
    assert float(hof - off) == pytest.approx(7.40625, abs=1e-9)
    assert PM6(yne_switch="off").correction_constants("YNE").tolist() == [0.0, 1.21, 1.33]

    bad = [(dict(bond_order_terms={"C-C": [1.0, 2.6, 2.2]}), "W_hi"),
           (dict(bond_order_terms={"C-C": [1.0, 2.2, 2.6], "C-N": [1.0, 2.2, 2.6, 1.4, 1.6]}), "same form"),
           (dict(bond_order_terms={"C-C": [1.0, 2.2]}), "same form"),
           (dict(bond_order_terms={"C-C": [1.0, 2.2, 2.6, 1.6, 1.4]}), "R_off"),
           (dict(bond_order_terms={"CC": [1.0, 2.2, 2.6]}), "element pair"),
           (dict(bond_order_terms={"C-N": [1.0, 2.2, 2.6], "N-C": [1.0, 2.2, 2.6]}), "given twice"),
           (dict(resonance_overlap={"C-N": 0.1, "N-C": 0.2}), "given twice"),
           (dict(resonance_overlap={"C-C": [0.1, 0.2, 0.3]}), "kappa_pi"),
           (dict(bond_order_cutoff=[1.6, 1.4]), "bond_order_cutoff"),
           (dict(trainable=["BOND_ORDER"]), "need the terms"),
           (dict(trainable=["YNE"], yne_switch="off"), "yne_switch='off'"),
           (dict(yne_switch="steep"), "yne_switch")]
    for options, match in bad:
        with pytest.raises(ValueError, match=match):
            PM7(**options)
    assert PM3(bond_order_terms={"C-C": [1.0, 2.2, 2.6]}).bond_order_terms


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_bucket_graphs_with_the_terms_on_gpu(cyano_terms):
    """On a GPU, shuffled batches replay their buckets' graphs (build, SCF, response, outputs) with the terms and
    give the reference's energies, forces and parameter gradients."""
    systems = [(CYANOACETYLENE, None), (WATER, None), (ACETYLENE, None), (VINYL, 2.0)]
    groups = ["BOND_ORDER", "BOND_ORDER_WINDOW", "RESONANCE_OVERLAP", "BETAP"]
    fast = _model(cyano_terms, trainable=groups).cuda()
    ref = _model(cyano_terms, trainable=groups, use_fast=False).cuda()
    rng = np.random.default_rng(0)
    for _ in range(10):
        idx = rng.permutation(len(systems))[:2].tolist()

        def data(idx=idx):
            return collate([_graph(systems[i][0], systems[i][1], device="cuda") for i in idx])
        _close(_outputs(fast, data(), True), _outputs(ref, data(), True), 1e-8)
        _close(_outputs(fast, data()), _outputs(ref, data()), 1e-9)
    assert any(g for b in fast._plans.values() for g in b.graphs.values())
    (H,) = hessian(fast, _graph(CYANOACETYLENE, device="cuda"))
    (H_ref,) = hessian(ref, _graph(CYANOACETYLENE, device="cuda"))
    assert torch.allclose(H, H_ref, rtol=0.0, atol=1e-8)


def test_stand_in_graph_dispatch(monkeypatch, cyano_terms):
    """The launched SCF of a model with the term converges through the dispatch of graphed stages to the same
    solution."""
    bo, args, result = _recorded(monkeypatch, _model(cyano_terms), CYANOACETYLENE)
    layout, H, W1, W2, n_alpha, n_beta, uhf, guess = args[:8]
    W1p, W2p = scf.pack_one_center(W1.detach()), scf.pack_pairs(layout, W2.detach())
    P, n_occ, weight, filled, blocks = scf.scf_start(guess, n_alpha, n_beta, uhf, None)
    loop = scf.SCFLoop(layout, H.detach(), W1p, W2p, n_occ, weight, uhf, 8, filled, blocks, True, bo)
    loop.start(P)
    stand_in = types.SimpleNamespace(loop=loop, check=H.new_zeros(3), host=None, host_check=H.new_zeros(3),
                                     dE_kept=loop.dE.clone())
    stand_in.run = lambda stage: (scf.SCFGraphs._occupy_commit(stand_in) if stage == "occupy_commit"
                                  else getattr(loop, stage)())
    stand_in.diagonalize = lambda prefetched=False: scf.SCFGraphs.diagonalize(stand_in, prefetched)
    got = scf.solve_scf(scf._GraphedLoop(stand_in), layout, H.detach(), W1p, W2p, n_occ, weight, uhf, 8, filled,
                        blocks, 300, 1e-11, 1e-12, 2.0, 0, 0.05)
    assert bool(got.converged.all())
    assert torch.allclose(got.electronic, result.electronic, rtol=0.0, atol=1e-8)
    assert torch.allclose(got.densities, result.densities, rtol=0.0, atol=1e-6)


def _stand_in_loop(changes):
    """A loop whose iterations report the given largest density changes (no energy change, no orbital gradient)."""
    seq = iter(changes)
    loop = types.SimpleNamespace(diis=types.SimpleNamespace(reset=lambda: None), schedule=None)
    for name in ("fock_step", "diagonalize", "occupy", "commit", "settle"):
        setattr(loop, name, lambda *a: None)
    loop.changes = lambda: [next(seq), 0.0, 0.0]
    return loop


def test_stalled_iterations_are_detected():
    """A run without a new smallest density change for ``patience`` iterations stops as stalled; a slowly
    improving run does not, and patience 0 never stops."""
    wandering = [1e-1, 1e-2, 1e-3] + [2e-3, 5e-3, 1.5e-3] * 20
    it, _, stalled = scf._iterate(_stand_in_loop(wandering), 60, 0, 0.0, 0.0, 1e-9, 1e-10, False, 10)
    assert stalled and it == 13
    slow = [0.9 ** k for k in range(60)]
    assert not scf._iterate(_stand_in_loop(slow), 60, 0, 0.0, 0.0, 1e-9, 1e-10, False, 10)[2]
    assert scf._iterate(_stand_in_loop(wandering), 60, 0, 0.0, 0.0, 1e-9, 1e-10, False, 0) == (60, False, False)


def test_stalled_run_goes_on_level_shifted_to_the_same_solution(monkeypatch, cyano_terms):
    """A run stopped as stalled goes on from its density with level-shifted iterations (and the bond-order term)
    and reaches the solution of an ordinary run; energy changes within round-off count as none, so a tolerance
    below it still converges."""
    bo, args, result = _recorded(monkeypatch, _model(cyano_terms), CYANOACETYLENE)
    layout, H, W1, W2, n_alpha, n_beta, uhf, guess = args[:8]
    W1p, W2p = scf.pack_one_center(W1.detach()), scf.pack_pairs(layout, W2.detach())
    P, n_occ, weight, filled, blocks = scf.scf_start(guess, n_alpha, n_beta, uhf, None)
    loop = scf.SCFLoop(layout, H.detach(), W1p, W2p, n_occ, weight, uhf, 8, filled, blocks, False, bo)
    loop.start(P)
    real, shifts = scf._iterate, []

    def iterate(lp, max_iter, *a, **k):
        shifts.append(lp.shift)
        if lp is loop:
            it, rotated, _ = real(lp, 4, *a, **k)
            return it, rotated, True
        return real(lp, max_iter, *a, **k)
    monkeypatch.setattr(scf, "_iterate", iterate)
    got = scf.solve_scf(loop, layout, H.detach(), W1p, W2p, n_occ, weight, uhf, 8, filled, blocks, 300, 1e-11,
                        1e-16, 0.0, 0, 0.0, False, 40, 1.0)
    assert shifts == [0.0, 1.0]
    assert bool(got.converged.all()) and got.iterations < 300
    assert torch.allclose(got.electronic, result.electronic, rtol=0.0, atol=1e-8)
    assert torch.allclose(got.densities, result.densities, rtol=0.0, atol=1e-6)
