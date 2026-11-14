"""Tests of the (s)GDML force fields (xnn.krnn)."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from xnn.common.config import from_dict
from xnn.common.data import collate, structure_to_graph
from xnn.common.models import ForceStressOutput, build_model
from xnn.krnn.common import permutations as P
from xnn.krnn.common.sgdml_fit import fit_sgdml, force_rmse, kernel_matrix
from xnn.krnn.common.sgdml_io import load_sgdml_model
from xnn.krnn.featurizers import InverseDistances
from xnn.krnn.kernels import Matern, safe_norm
from xnn.krnn.models import GDML, SGDML

HERE = Path(__file__).parent


def _fidelity():
    spec = importlib.util.spec_from_file_location("sgdml_fidelity", HERE / "sgdml_fidelity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FID = _fidelity()


def _graphs(data, idx, cutoff=1.0, dtype=torch.float64, labels=True):
    out = []
    for k in idx:
        s = {"pos": torch.tensor(data["R"][k], dtype=dtype), "atomic_numbers": torch.tensor(data["z"])}
        if labels:
            s["energy"] = float(data["E"][k, 0])
            s["forces"] = torch.tensor(data["F"][k], dtype=dtype)
        out.append(structure_to_graph(s, cutoff))
    return out


@pytest.fixture(scope="module")
def data():
    return FID.dataset(80, 5)


@pytest.fixture(scope="module")
def fitted(data):
    model = SGDML(data["z"].tolist(), sigma=4.0)
    fit_sgdml(model, _graphs(data, range(30)))
    return model


def rel_err(a, b):
    """Largest deviation relative to the largest entry."""
    a, b = a.detach(), b.detach()
    return float((a - b).abs().max() / b.abs().max())


def reorder_tolerance(x, xp, wp, ec, kernel):
    """Tolerances of the energy and the gradient of the expansion: ten times the change when the
    training terms are summed in reverse order (the coefficients cancel to many digits), at
    least 1e-13."""
    from xnn.krnn.models.sgdml import kernel_energy
    e1, g1 = kernel_energy(x, xp, wp, ec, kernel, gradient=True)
    e2, g2 = kernel_energy(x, xp.flip(0), wp.flip(0), ec.flip(0), kernel, gradient=True)
    return max(10 * rel_err(e2, e1), 1e-13), max(10 * rel_err(g2, g1), 1e-13)


def test_matern_closed_forms():
    """The order-5/2 kernel and its radial derivatives have their closed forms; order 3/2 is refused."""
    sig = 1.7
    m = Matern(sig)
    r = torch.linspace(0, 3, 13, dtype=torch.float64)
    a = math.sqrt(5) / sig
    k = (1 + a * r + a * a * r * r / 3) * torch.exp(-a * r)
    assert torch.allclose(m.k(r), k, rtol=1e-15)
    assert torch.allclose(m.g(r), a * a / 3 * (1 + a * r) * torch.exp(-a * r), rtol=1e-15)
    assert torch.allclose(m.t_of(r), -a ** 4 / 3 * torch.exp(-a * r), rtol=1e-14)
    rr = r.clone().requires_grad_(True)
    m3 = Matern(sig, n=3)
    dk = torch.autograd.grad(m3.k(rr).sum(), rr, create_graph=True)[0]
    assert torch.allclose(-dk[1:] / rr[1:], m3.g(rr)[1:], rtol=1e-12)
    dg = torch.autograd.grad(m3.g(rr).sum(), rr)[0]
    assert torch.allclose(dg[1:] / r[1:], m3.t_of(r)[1:], rtol=1e-12)
    with pytest.raises(ValueError, match="5/2"):
        Matern(1.0, n=1)


def test_safe_norm_has_zero_gradient_at_the_origin():
    """The norm is exact and its gradient vanishes at zero."""
    x = torch.tensor([[0.0, 0.0], [3.0, 4.0]], dtype=torch.float64, requires_grad=True)
    n = safe_norm(x)
    assert n.tolist() == [0.0, 5.0]
    g = torch.autograd.grad(n.sum(), x)[0]
    assert g[0].tolist() == [0.0, 0.0] and g[1].tolist() == pytest.approx([0.6, 0.8])


def test_descriptor_order_jacobian_and_permutations():
    """Pairs follow the lower triangle; the Jacobian is the derivative; permuted indices permute atoms."""
    desc = InverseDistances(4)
    assert desc.pairs.t().tolist() == [[1, 0], [2, 0], [2, 1], [3, 0], [3, 1], [3, 2]]
    pos = torch.randn(2, 4, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    x = desc(pos)
    assert x[0, 2].item() == pytest.approx(1 / (pos[0, 2] - pos[0, 1]).norm().item())
    jac = desc.jacobian(pos)
    auto = torch.autograd.functional.jacobian(lambda p: desc(p.reshape(1, 4, 3))[0], pos[1].reshape(-1))
    assert torch.allclose(jac[1], auto, rtol=1e-13)
    perm = [2, 0, 3, 1]
    idx = desc.permutation_index([perm])[0]
    assert torch.allclose(x[0][idx], desc(pos[:, perm])[0], rtol=1e-15)


def test_descriptor_minimum_image():
    """In a cell every pair takes its nearest image."""
    desc = InverseDistances(2)
    cell = torch.eye(3, dtype=torch.float64).unsqueeze(0) * 4.0
    pos = torch.tensor([[[0.2, 0.0, 0.0], [3.9, 0.0, 0.0]]], dtype=torch.float64)
    assert desc(pos, cell).item() == pytest.approx(1 / 0.3)
    jac = desc.jacobian(pos, cell)
    auto = torch.autograd.functional.jacobian(lambda p: desc(p.reshape(1, 2, 3), cell)[0], pos.reshape(-1))
    assert torch.allclose(jac[0], auto, rtol=1e-13)


def test_energy_is_the_kernel_expansion(fitted, data):
    """The energy equals an explicit loop over training points and permutations."""
    model = fitted
    pos = torch.tensor(data["R"][50:52], dtype=torch.float64)
    x = model.descriptor(pos)
    expect = []
    for b in range(2):
        e = 0.0
        for j in range(model.train_desc.shape[0]):
            for q in range(model.perm_index.shape[0]):
                pi = model.perm_index[q]
                d = x[b] - model.train_desc[j][pi]
                r = d.norm()
                e += float(model.kernel.g(r) * (d * model.train_weights[j][pi]).sum())
        expect.append(float(model.c) + float(model.std) * e)
    got = model.energy_of_descriptors(x)
    assert got.tolist() == pytest.approx(expect, rel=1e-12)


def test_chunked_sums_equal_one_pass(monkeypatch, fitted):
    """Energies and gradients summed chunk by chunk equal a single pass; GPUs take larger chunks."""
    from xnn.krnn.models import sgdml as S
    x = fitted.descriptor(torch.rand(3, fitted.n_atoms, 3, dtype=torch.float64) * 3)
    xp, wp = fitted.permuted_training_set()
    ec = torch.rand(xp.shape[0], dtype=torch.float64)
    args = (x, xp, wp, ec, fitted.kernel)
    e1, g1 = S.kernel_energy(*args, gradient=True)
    h1 = S.kernel_energy_gradient(*args)
    monkeypatch.setattr(S, "CHUNK", 3 * xp.shape[1] * xp.shape[2])
    monkeypatch.setattr(S.torch, "get_num_threads", lambda: 1)
    assert len(S._chunks(xp.shape[0], 3 * xp.shape[1] * xp.shape[2])) == xp.shape[0]
    assert len(S._chunks(xp.shape[0], 3 * xp.shape[1] * xp.shape[2], torch.device("meta"))) == 1
    e2, g2 = S.kernel_energy(*args, gradient=True)
    tol_e, tol_g = reorder_tolerance(*args)
    assert rel_err(e2, e1) <= tol_e and rel_err(g2, g1) <= tol_g
    assert rel_err(S.kernel_energy_gradient(*args), h1) <= tol_g
    assert rel_err(g1, h1) <= tol_g


def test_forces_are_conservative_and_symmetric(fitted, data):
    """Forces are minus the gradient (finite differences) and the energy is invariant under the group."""
    model = fitted
    assert model.perms.shape[0] > 1
    g = collate(_graphs(data, [60], labels=False))
    out = ForceStressOutput(model)(g)
    pos = torch.tensor(data["R"][60], dtype=torch.float64)

    def energy(p):
        return model(collate([structure_to_graph({"pos": p, "atomic_numbers": torch.tensor(data["z"])}, 1.0)]))[
            "energy"].item()

    h = 1e-5
    for a, c in [(0, 0), (4, 1), (8, 2)]:
        pp, pm = pos.clone(), pos.clone()
        pp[a, c] += h
        pm[a, c] -= h
        assert out["forces"][a, c].item() == pytest.approx(-(energy(pp) - energy(pm)) / (2 * h), rel=1e-6, abs=1e-8)
    for perm in model.perms.tolist():
        assert energy(pos[perm]) == pytest.approx(energy(pos), rel=1e-12)


def _dense_kernel(model, x, jac, energy_constraint):
    """The kernel matrix from dense Jacobians (an independent form of the same equations)."""
    m, d, n3 = jac.shape
    pidx = model.perm_index
    xp, jp = x[:, pidx], jac[:, pidx, :]
    k = x.new_zeros(m * n3 + (m if energy_constraint else 0), m * n3 + (m if energy_constraint else 0))
    for i in range(m):
        diff = x[i] - xp
        r = diff.norm(dim=-1)
        g, t = model.kernel.g(r), model.kernel.t_of(r)
        proj = torch.einsum("mqd,mqdk->mqk", diff, jp)
        h = g[..., None, None] * jp + t[..., None, None] * diff[..., None] * proj[..., None, :]
        k[i * n3:(i + 1) * n3, :m * n3] = -torch.einsum("di,mdk->imk", jac[i], h.sum(1)).reshape(n3, -1)
        if energy_constraint:
            off = m * n3
            k[i * n3:(i + 1) * n3, off:] = torch.einsum("di,md->im", jac[i], (g[..., None] * diff).sum(1))
            k[off + i, :off] = -(g[..., None] * proj).sum(1).reshape(-1)
            k[off + i, off:] = -model.kernel.k(r).sum(1)
    return k


@pytest.mark.parametrize("group", [True, False])
def test_kernel_matrix_matches_the_dense_form(data, group):
    """The sparse assembly equals the dense one; with a symmetry group it is symmetric and definite."""
    model = SGDML(data["z"].tolist(), sigma=3.0)
    perms = [list(range(9)), [0, 1, 2, 4, 3, 5, 6, 7, 8]]
    model.set_permutations(perms if group else perms + [[0, 1, 2, 3, 4, 6, 7, 5, 8]])
    pos = torch.tensor(data["R"][:6], dtype=torch.float64)
    x, u, jac = model.descriptor(pos), model.descriptor.pair_gradients(pos), model.descriptor.jacobian(pos)
    for ec in (False, True):
        k = kernel_matrix(model, x, u, ec)
        dense = _dense_kernel(model, x, jac, ec)
        assert torch.allclose(k, dense, rtol=1e-12, atol=1e-14 * dense.abs().max())
        lower = kernel_matrix(model, x, u, ec, full=False)
        assert torch.allclose(torch.tril(lower), torch.tril(dense), rtol=1e-12, atol=1e-14 * dense.abs().max())
        if group:
            assert torch.allclose(k, k.T, rtol=1e-12, atol=1e-14 * k.abs().max())
            assert torch.linalg.eigvalsh(-k).min() > -1e-8 * k.abs().max()


def test_energy_function_derivatives(fitted):
    """The analytic gradient of the kernel energy and its own derivative pass gradcheck."""
    from xnn.krnn.models.sgdml import _KernelEnergy

    xp, wp = fitted.permuted_training_set()
    xp, wp = xp[:4], wp[:4]
    x = (xp[0, 0] + 0.01 * torch.randn(2, xp.shape[-1], dtype=torch.float64,
                                       generator=torch.Generator().manual_seed(2))).requires_grad_(True)
    for ec in (torch.zeros(0, dtype=torch.float64), torch.randn(4, dtype=torch.float64)):
        def f(z, ec=ec):
            return _KernelEnergy.apply(z, xp, wp, ec, fitted.kernel)
        assert torch.autograd.gradcheck(f, (x,))
        assert torch.autograd.gradgradcheck(f, (x,))


def test_hessian_matches_finite_differences(fitted, data):
    """Second derivatives of the energy by the positions match differences of the forces."""
    pos = torch.tensor(data["R"][62], dtype=torch.float64)

    def energy(p):
        return fitted.energy_of_descriptors(fitted.descriptor(p.reshape(1, 9, 3)))[0]

    hess = torch.autograd.functional.hessian(energy, pos.reshape(-1))
    h = 1e-5
    for k in (0, 7, 20):
        dp = torch.zeros(27, dtype=torch.float64)
        dp[k] = h
        gp = torch.autograd.functional.jacobian(energy, pos.reshape(-1) + dp)
        gm = torch.autograd.functional.jacobian(energy, pos.reshape(-1) - dp)
        assert torch.allclose(hess[k], (gp - gm) / (2 * h), rtol=1e-5, atol=1e-6 * hess.abs().max())


@pytest.mark.parametrize("energy_constraint", [False, True])
def test_fit_interpolates_training_forces(data, energy_constraint):
    """The fit reproduces the training forces and energies; validation errors are small."""
    model = GDML(data["z"].tolist(), sigma=4.0)
    train = _graphs(data, range(25))
    fit_sgdml(model, train, energy_constraint=energy_constraint)
    assert model.perms.shape[0] == 1
    assert force_rmse(model, train) < 1e-4 * float(model.std)
    out = model(collate(train))
    ref = torch.tensor(data["E"][:25, 0])
    assert (out["energy"] - ref).abs().max() < 0.05 * (ref.max() - ref.min())
    if energy_constraint:
        assert model.energy_coef.numel() == 25 and model.c.item() == pytest.approx(ref.mean().item())
    assert force_rmse(model, _graphs(data, range(60, 70))) < 0.5 * float(model.std)


def test_lu_fallback_and_fit_method(monkeypatch, data):
    """A failed Cholesky factorization falls back to LU; the model's fit method and kernel order work."""
    ref = GDML(data["z"].tolist(), sigma=4.0).fit(_graphs(data, range(10)))
    real = torch.linalg.cholesky_ex
    monkeypatch.setattr(torch.linalg, "cholesky_ex", lambda a: (real(a)[0], torch.tensor(1)))
    lu = GDML(data["z"].tolist(), sigma=4.0).fit(_graphs(data, range(10)))
    assert torch.allclose(lu.train_weights, ref.train_weights, rtol=1e-6, atol=1e-8 * ref.train_weights.abs().max())
    m = GDML(data["z"].tolist())
    m.set_kernel(3.0, kernel_order=3)
    assert m.kernel.n == 3 and m.sigma == 3.0


def test_sigma_selection_and_errors(data):
    """Several sigma are compared on validation structures; bad inputs raise."""
    model = SGDML(data["z"].tolist())
    fit_sgdml(model, _graphs(data, range(20)), sigma=[0.5, 4.0], validation=_graphs(data, range(60, 66)),
              perms=np.arange(9)[None])
    assert model.sigma == 4.0
    with pytest.raises(ValueError, match="validation"):
        fit_sgdml(model, _graphs(data, range(5)), sigma=[1.0, 2.0])
    with pytest.raises(ValueError, match="force labels"):
        fit_sgdml(model, _graphs(data, range(5), labels=False))
    g = _graphs(data, range(5), labels=False)
    for x in g:
        x.forces = torch.zeros(9, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="energy labels"):
        fit_sgdml(model, g, energy_constraint=True)
    with pytest.raises(ValueError, match="atoms"):
        model(collate([structure_to_graph({"pos": torch.zeros(3, 3), "atomic_numbers": torch.tensor([6, 6, 8])}, 1.0)]))
    with pytest.raises(ValueError, match="elements"):
        model(collate([structure_to_graph({"pos": torch.tensor(data["R"][0]),
                                           "atomic_numbers": torch.tensor(data["z"][::-1].copy())}, 1.0)]))
    with pytest.raises(ValueError, match="two atoms"):
        SGDML([1])


def test_symmetry_search_and_group_helpers():
    """The search finds the exchange of equivalent atoms; closure and the cycle subgroup behave."""
    rng = np.random.default_rng(0)
    base = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    r = []
    for _ in range(12):
        x = base + 0.03 * rng.standard_normal((3, 3))
        r.append(x[[0, 2, 1]] if rng.random() < 0.5 else x)
    perms = P.find_permutations(np.array(r), np.array([8, 1, 1]))
    assert {tuple(p) for p in perms.tolist()} == {(0, 1, 2), (0, 2, 1)}
    cyc = np.array([[1, 2, 0, 3], [0, 1, 2, 3]])
    assert P.close_group(cyc).shape[0] == 3
    assert P.close_group(np.array([[1, 2, 3, 4, 0], [1, 0, 2, 3, 4]]), max_size=10) is None
    sub = P.largest_cycle_subgroup(np.array([[1, 2, 0, 3], [1, 0, 2, 3], [0, 1, 3, 2]]))
    assert sub.tolist() == [[1, 2, 0, 3]]
    cell = np.eye(3) * 3.0
    d = P._distances(np.array([[0.1, 0, 0], [2.9, 0, 0]]), cell)
    assert d[0, 1] == pytest.approx(0.2)


def _match_pairs_one_by_one(r, z):
    """One assignment problem per pair of geometries: the matching written out pair by pair."""
    from scipy.optimize import linear_sum_assignment
    m = len(r)
    other = (z[:, None] != z[None, :]).astype(float)
    adj = [P._distances(x, None) for x in r]
    vec = []
    for a in adj:
        w, v = np.linalg.eig(a)
        vec.append(np.abs(v[:, np.argsort(w)[::-1]].real))
    perms, cost = {}, np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            c = -vec[i] @ vec[j].T
            c = c + other * np.max(np.abs(c))
            p = linear_sum_assignment(c)[1]
            before = np.linalg.norm(adj[i] - adj[j])
            after = np.linalg.norm(adj[i][np.ix_(p, p)] - adj[j])
            cost[i, j] = min(after, before)
            if after < before and not np.isclose(before, after):
                perms[i, j] = p
    cost = cost + cost.T
    np.fill_diagonal(cost, np.inf)
    return perms, cost


def test_batched_pair_matching_equals_one_assignment_per_pair(monkeypatch, data):
    """Batched pairs, with plain assignments read off the costs, match solving every pair."""
    exact = FID.ETHANOL_R
    r = np.concatenate([data["R"][:10], exact[None], exact[None][:, FID.SYMMETRIES[1]], exact[None]])
    z = np.asarray(data["z"])
    monkeypatch.setattr(P, "PAIR_BLOCK", 3 * len(z) ** 2)
    ref_perms, ref_cost = _match_pairs_one_by_one(r, z)
    perms, cost = P.match_pairs(r, z)
    assert set(perms) == set(ref_perms) and len(perms) > 0
    assert all(np.array_equal(perms[k], ref_perms[k]) for k in perms)
    np.testing.assert_allclose(cost, ref_cost, rtol=1e-12, atol=1e-14)


def test_symmetry_fallback_to_the_cycle_subgroup(monkeypatch, data):
    """A closure that outgrows the limit falls back to the unshadowed cycles."""
    big = np.array([np.roll(np.arange(9), k) for k in range(2)])
    monkeypatch.setattr(P, "synchronize", lambda perms, cost, n: big)
    out = P.find_permutations(data["R"][:3], data["z"], max_size=5)
    assert out.tolist() == [list(range(9))]
    s3 = np.array([np.arange(9), [1, 2, 0, 3, 4, 5, 6, 7, 8], [1, 0, 2, 3, 4, 5, 6, 7, 8]])
    monkeypatch.setattr(P, "synchronize", lambda perms, cost, n: s3)
    out = P.find_permutations(data["R"][:3], data["z"], max_size=4)
    assert out.shape[0] == 3


def test_symmetry_subset_and_periodic_fit(data):
    """Large training sets are subsampled for the search; periodic models keep their cell."""
    model = SGDML(data["z"].tolist(), sigma=4.0)
    fit_sgdml(model, _graphs(data, range(12)), max_symmetry_samples=6, generator=torch.Generator().manual_seed(1))
    assert model.perms.shape[0] >= 1
    cell = torch.eye(3, dtype=torch.float64) * 12.0
    graphs = []
    for k in range(10):
        s = {"pos": torch.tensor(data["R"][k]) + 3.0, "atomic_numbers": torch.tensor(data["z"]),
             "cell": cell, "pbc": torch.tensor([True, True, True]), "energy": float(data["E"][k, 0]),
             "forces": torch.tensor(data["F"][k])}
        graphs.append(structure_to_graph(s, 1.0))
    periodic = GDML(data["z"].tolist(), sigma=4.0)
    fit_sgdml(periodic, graphs)
    assert periodic.lattice.shape == (3, 3)
    assert force_rmse(periodic, graphs) < 1e-3 * float(periodic.std)
    _check_direct_forces(periodic, graphs[:3])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_fit_on_a_gpu_searches_symmetries_on_the_host(data):
    """Fits of structures on the GPU find the CPU fit's symmetries, for molecules and cells."""
    graphs = _graphs(data, range(12))
    cpu = SGDML(data["z"].tolist(), sigma=4.0)
    fit_sgdml(cpu, graphs, max_symmetry_samples=6, generator=torch.Generator().manual_seed(1))
    gpu = SGDML(data["z"].tolist(), sigma=4.0).cuda()
    on_gpu = [g.to("cuda") for g in graphs]
    fit_sgdml(gpu, on_gpu, max_symmetry_samples=6, generator=torch.Generator().manual_seed(1))
    assert torch.equal(gpu.perms.cpu(), cpu.perms)
    assert force_rmse(gpu, on_gpu) < 1e-3 * float(gpu.std)
    cell = torch.eye(3, dtype=torch.float64) * 12.0
    periodic_graphs = [structure_to_graph({"pos": torch.tensor(data["R"][k]) + 3.0,
                                           "atomic_numbers": torch.tensor(data["z"]), "cell": cell,
                                           "pbc": torch.tensor([True, True, True]),
                                           "energy": float(data["E"][k, 0]),
                                           "forces": torch.tensor(data["F"][k])}, 1.0).to("cuda")
                       for k in range(10)]
    periodic = SGDML(data["z"].tolist(), sigma=4.0).cuda()
    fit_sgdml(periodic, periodic_graphs)
    assert periodic.lattice.device.type == "cuda" and periodic.perms.shape[0] >= 1


def _check_direct_forces(model, graphs):
    """Evaluation forms forces and stress from the paper's force predictor; they equal the autograd ones."""
    wrapped = ForceStressOutput(model, compute_stress=True).eval()
    batch = collate([g for g in graphs])
    assert model.forms_forces(batch)
    direct = wrapped(collate([g for g in graphs]))
    graph = collate([g for g in graphs])
    graph.pos.requires_grad_(True)
    assert not model.forms_forces(graph)
    auto = wrapped(graph)
    assert ("stress" in direct) == ("stress" in auto)
    for key in [k for k in ("energy", "forces", "stress") if k in auto]:
        assert not direct[key].requires_grad
        assert rel_err(direct[key], auto[key].detach()) < 1e-11, key
    model.train()
    assert not model.forms_forces(batch)
    model.eval()


def test_direct_forces_of_a_molecule(fitted, data):
    """Molecules take the direct forces in evaluation as well."""
    _check_direct_forces(fitted, _graphs(data, [73, 74], labels=False))


def test_float32_matches_float64(fitted, data):
    """A float32 model keeps float64 coefficients and arithmetic; only its inputs are rounded."""
    out = {}
    for dtype in (torch.float64, torch.float32):
        m = SGDML(data["z"].tolist(), sigma=fitted.sigma)
        m.load_state_dict(fitted.state_dict())
        m.set_kernel(fitted.sigma)
        m = m.to(dtype)
        out[dtype] = ForceStressOutput(m)(collate(_graphs(data, [70, 71], dtype=dtype, labels=False)))
    e64, e32 = out[torch.float64], out[torch.float32]
    assert m.train_weights.dtype == torch.float64 and e32["forces"].dtype == torch.float32
    assert (e32["energy"].double() - e64["energy"]).abs().max() < 1e-6 * e64["energy"].abs().max()
    assert (e32["forces"].double() - e64["forces"]).abs().max() < 1e-4 * e64["forces"].abs().max()


def test_state_dict_round_trip_resizes_buffers(fitted, data):
    """A fresh model takes the sizes of a saved state dict."""
    fresh = SGDML(data["z"].tolist(), sigma=fitted.sigma)
    fresh.load_state_dict(fitted.state_dict())
    g = collate(_graphs(data, [72], labels=False))
    assert fresh(g)["energy"].item() == pytest.approx(fitted(g)["energy"].item(), rel=1e-14)


def test_model_file_and_config(tmp_path, fitted, data):
    """sGDML model files load (with energy coefficients and a lattice); the registry builds both models."""
    m = {"z": data["z"], "R_desc": fitted.train_desc.numpy().T, "R_d_desc_alpha": fitted.train_weights.numpy(),
         "sig": fitted.sigma, "std": float(fitted.std), "c": float(fitted.c), "perms": fitted.perms.numpy()}
    path = tmp_path / "m.npz"
    np.savez(path, **m)
    loaded = load_sgdml_model(path)
    g = collate(_graphs(data, [73], labels=False))
    assert loaded(g)["energy"].item() == pytest.approx(fitted(g)["energy"].item(), rel=1e-14)
    np.savez(path, **{**m, "alphas_E": np.zeros(fitted.train_desc.shape[0]), "lattice": np.eye(3) * 30.0})
    loaded = load_sgdml_model(path)
    assert loaded.energy_coef.numel() == fitted.train_desc.shape[0] and loaded.lattice[0, 0].item() == 30.0
    assert loaded(g)["energy"].item() == pytest.approx(fitted(g)["energy"].item(), rel=1e-12)
    cfg = from_dict({"model": {"name": "sgdml", "cutoff": 2.0, "extra": {"species": ["C", "C", "O"] + ["H"] * 6,
                                                                          "sigma": 5.0}}})
    model = build_model(cfg.model)
    assert isinstance(model, SGDML) and model.use_symmetries and model.sigma == 5.0
    gd = build_model(from_dict({"model": {"name": "gdml", "extra": {"species": [8, 1, 1]}}}).model)
    assert isinstance(gd, GDML) and not gd.use_symmetries and gd.receptive_field() is None
    fromfile = build_model(from_dict({"model": {"name": "sgdml", "extra": {"model_file": str(path)}}}).model)
    assert fromfile.train_desc.shape == fitted.train_desc.shape


def check_fidelity(rows, fast=False):
    """Both precisions of the predictions and of the refits against the sGDML package.

    The kernels sum in another order than the reference, so in float64 they may differ from
    the package by up to ten times the reference's own change under reordering.
    """
    eps32 = torch.finfo(torch.float32).eps
    tol64 = max(1e-11, 10 * max(rows["f64_reorder"].values())) if fast else 1e-11
    assert max(rows["f64"].values()) < tol64
    # the package computes in float64 only: on float32 positions it is the float32 oracle, to one rounding
    assert max(rows["f32_vs_upstream"].values()) < 2 * eps32
    if "upstream_f32" in rows:
        assert max(rows["f32_vs_upstream_f32"].values()) < 2 * eps32
    assert max(rows["f32"].values()) < 1e-4
    assert rows["refit"]["energy"] < 1e-8 and rows["refit"]["forces"] < 1e-8
    assert rows["refit_f32"]["energy"] < 2 * eps32 and rows["refit_f32"]["forces"] < 2 * eps32
    assert rows["refit"]["weights"] < 1e-5 and rows["refit_f32"]["weights"] < 1e-5
    if "perms" in rows:
        assert rows["perms"]["same_group"] == 1.0


@pytest.mark.parametrize("case", ["sym", "nosym", "ecstr"])
def test_fidelity_vs_sgdml(case):
    """Transplanted models, refits and symmetry groups match the sGDML package in float64 and float32."""
    if not FID.have_sgdml():
        pytest.skip("the sGDML package is not installed")
    check_fidelity(FID.run_case(case))


def test_ase_calculator(fitted, data):
    """The ASE calculator gives the model's energy and forces."""
    pytest.importorskip("ase")
    from ase import Atoms

    from xnn.common.deploy import XNNCalculator

    atoms = Atoms(numbers=data["z"], positions=data["R"][74])
    atoms.calc = XNNCalculator(ForceStressOutput(fitted), cutoff=fitted.cutoff)
    out = ForceStressOutput(fitted)(collate(_graphs(data, [74], labels=False)))
    assert atoms.get_potential_energy() == pytest.approx(out["energy"].item(), rel=1e-12)
    assert np.allclose(atoms.get_forces(), out["forces"].detach().numpy(), rtol=1e-10, atol=1e-12)


def test_trainer_fits_in_closed_form_and_checkpoints(tmp_path, data):
    """The trainer fits the model once, picks sigma on the validation set and saves a reloadable checkpoint."""
    from xnn.common.data import AtomicDataset
    from xnn.common.models.hub import load_pretrained
    from xnn.common.models.hub.checkpoint import core_model
    from xnn.common.train import Trainer

    structs = [{"pos": data["R"][k], "atomic_numbers": data["z"], "energy": float(data["E"][k, 0]),
                "forces": data["F"][k]} for k in range(30)]
    cfg = from_dict({"model": {"name": "sgdml", "cutoff": 1.0,
                               "extra": {"species": data["z"].tolist(), "sigma": [0.5, 4.0]}},
                     "data": {"batch_size": 8, "val_fraction": 0.2, "test_fraction": 0.1},
                     "device": "cpu", "output_dir": str(tmp_path)})
    trainer = Trainer(cfg, AtomicDataset(structs, 1.0))
    assert trainer.opt is None
    out = trainer.fit()
    assert out["val"] and out["test"] and out["train"]["force_mse"] < out["val"]["force_mse"]
    fitted = core_model(trainer.module)
    assert fitted.sigma == 4.0
    loaded = core_model(load_pretrained(tmp_path / "best.pt").model)
    assert loaded.sigma == 4.0
    g = collate(_graphs(data, [75], labels=False))
    assert loaded(g)["energy"].item() == pytest.approx(fitted(g)["energy"].item(), rel=1e-13)


def test_trainer_refuses_distributed_closed_form_fits(monkeypatch, tmp_path, data):
    """A closed-form fit runs in one process."""
    from xnn.common.data import AtomicDataset
    from xnn.common.train import Trainer
    from xnn.common.train import trainer as trainer_module

    monkeypatch.setattr(trainer_module.distributed, "is_distributed", lambda: True)
    monkeypatch.setattr(Trainer, "_init_distributed", lambda self, dev: torch.device("cpu"))
    structs = [{"pos": data["R"][k], "atomic_numbers": data["z"], "energy": float(data["E"][k, 0]),
                "forces": data["F"][k]} for k in range(4)]
    cfg = from_dict({"model": {"name": "gdml", "extra": {"species": data["z"].tolist(), "sigma": 4.0}},
                     "device": "cpu", "output_dir": str(tmp_path)})
    with pytest.raises(ValueError, match="one process"):
        Trainer(cfg, AtomicDataset(structs, 1.0))
