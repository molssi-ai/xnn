"""The iterative (conjugate-gradient) solve of the sGDML fit against the analytic one."""
import os

import pytest
import torch

from test_sgdml import FID, _graphs, rel_err
from xnn.common.config import from_dict
from xnn.common.data import collate
from xnn.common.models import ForceStressOutput, build_model
from xnn.krnn.common import sgdml_iterative as it
from xnn.krnn.common.sgdml_fit import fit_sgdml, kernel_matrix, kernel_rows
from xnn.krnn.models import SGDML

GROUP = [list(range(9)), [0, 1, 2, 4, 3, 5, 6, 7, 8]]


@pytest.fixture(scope="module")
def data():
    return FID.dataset(40, 11)


def _model(data, perms=GROUP, sigma=4.0):
    model = SGDML(data["z"].tolist(), sigma=sigma)
    model.set_permutations(perms)
    return model


def _inputs(model, data, m=8):
    pos = torch.tensor(data["R"][:m], dtype=torch.float64)
    return model.descriptor(pos), model.descriptor.pair_gradients(pos)


@pytest.mark.parametrize("energy", [False, True])
def test_kernel_product_and_rows_match_the_matrix(data, energy):
    """``K v`` without the matrix and the rows of chosen geometries equal the assembled matrix."""
    model = _model(data)
    x, u = _inputs(model, data)
    k = kernel_matrix(model, x, u, energy)
    v = torch.randn(k.shape[0], dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    assert rel_err(it.kernel_product(model, x, u, v, energy), k @ v) < 1e-12
    assert rel_err(it.PairFactors(model, x, u, energy)(v), k @ v) < 1e-9
    idx = [1, 5]
    dofs = it._dofs(idx, x.shape[0], 27, energy)
    assert rel_err(kernel_rows(model, x, u, idx, energy), k[dofs]) < 1e-13


@pytest.mark.parametrize("energy", [False, True])
def test_cg_fit_matches_the_analytic_fit(data, energy):
    """The iterative fit reaches the analytic coefficients to its tolerance."""
    graphs = _graphs(data, range(10))
    ref = _model(data)
    fit_sgdml(ref, graphs, lam=1e-6, perms=GROUP, energy_constraint=energy, solver="analytic")
    assert ref.fit_info == {"solver": "analytic"}
    cg = _model(data)
    fit_sgdml(cg, graphs, lam=1e-6, perms=GROUP, energy_constraint=energy, solver="cg", tol=1e-10, n_inducing=3,
              generator=torch.Generator().manual_seed(0))
    assert cg.fit_info["solver"] == "cg" and cg.fit_info["converged"] and cg.fit_info["n_inducing"] == 3
    assert rel_err(cg.train_weights, ref.train_weights) < 1e-6
    if energy:
        assert rel_err(cg.energy_coef, ref.energy_coef) < 1e-6
    a = ForceStressOutput(ref).eval()(collate(_graphs(data, range(30, 34), labels=False)))
    b = ForceStressOutput(cg).eval()(collate(_graphs(data, range(30, 34), labels=False)))
    assert rel_err(b["forces"], a["forces"]) < 1e-6 and rel_err(b["energy"], a["energy"]) < 1e-6


def test_full_preconditioner_converges_at_once(data):
    """With every geometry inducing, the preconditioner is the inverse of the system."""
    model = _model(data)
    x, u = _inputs(model, data, 6)
    y = torch.randn(6 * 27, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    alpha, record = it.solve_iterative(model, x, u, y, 1e-4, tol=1e-9, n_inducing=50)
    assert record["n_inducing"] == 6 and record["iterations"] <= 3 and record["converged"]
    k = kernel_matrix(model, x, u)
    k.diagonal().sub_(1e-4)
    assert rel_err(k @ alpha, y) < 1e-8


def test_auto_choice_and_memory_budget(data, monkeypatch):
    """``"auto"`` solves analytically when the matrix fits and iteratively when it does not."""
    graphs = _graphs(data, range(8))
    model = _model(data)
    fit_sgdml(model, graphs, lam=1e-6, perms=GROUP)
    assert model.fit_info["solver"] == "analytic"
    fit_sgdml(model, graphs, lam=1e-6, perms=GROUP, max_memory=1e-6, tol=1e-8)
    assert model.fit_info["solver"] == "cg" and model.fit_info["n_inducing"] == 1
    assert it.inducing_count(100, 9, 1e3) == 100 and it.inducing_count(100, 9, 1e-9) == 1
    assert it.inducing_count(1000, 9, 4) == 154
    assert it.device_memory(torch.device("cpu")) > 0

    class Props:
        total_memory = 2 ** 33

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: Props())
    assert it.device_memory(torch.device("cuda")) == 8.0
    monkeypatch.setattr(it, "device_memory", lambda device: 1e-6)
    fit_sgdml(model, graphs, lam=1e-6, perms=GROUP, tol=1e-8)
    assert model.fit_info["solver"] == "cg"


def test_solver_errors_and_warnings(data):
    """Unknown solvers and asymmetric kernel matrices are refused; an unconverged solve warns."""
    graphs = _graphs(data, range(6))
    model = _model(data)
    with pytest.raises(ValueError, match="solver"):
        fit_sgdml(model, graphs, perms=GROUP, solver="lu")
    with pytest.raises(ValueError, match="symmetric"):
        fit_sgdml(model, graphs, perms=GROUP + [[0, 1, 2, 3, 4, 6, 7, 5, 8]], solver="cg")
    with pytest.warns(RuntimeWarning, match="stopped after 1 iterations"):
        fit_sgdml(model, graphs, lam=1e-8, perms=GROUP, solver="cg", tol=1e-14, n_inducing=1, max_iter=1)
    assert not model.fit_info["converged"] and model.fit_info["iterations"] == 1


def test_stable_cholesky_and_leverage_draw(data):
    """Singular blocks get the smallest working shift, indefinite ones are refused; draws are distinct."""
    a = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    chol = it._stable_cholesky(a)
    assert torch.allclose(chol @ chol.T, a, atol=1e-10)
    with pytest.raises(RuntimeError, match="semi-definite"):
        it._stable_cholesky(-torch.eye(2, dtype=torch.float64))
    model = _model(data)
    x, u = _inputs(model, data, 12)
    gen = torch.Generator().manual_seed(3)
    idx = it._leverage_draw(model, x, u, 4, 1e-6, True, gen)
    assert len(set(idx)) == 4 and idx == sorted(idx)
    assert it._leverage_draw(model, x, u, 20, 1e-6, False, gen) == list(range(12))


def test_kernel_product_on_the_fused_kernels(data, monkeypatch):
    """The product takes the fused kernels when the model does (Triton's interpreter on a CPU)."""
    if os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("runs where Triton's interpreter is on (no CUDA device)")
    pytest.importorskip("triton")
    import xnn.krnn.fast.sgdml as fast
    monkeypatch.setattr(fast, "supported", lambda device, dtype: True)
    model = _model(data)
    x, u = _inputs(model, data, 5)
    v = torch.randn(5 * 27 + 5, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    ref = it.kernel_product(model.set_use_fast(False), x, u, v, True)
    fast_out = it.kernel_product(model.set_use_fast(True), x, u, v, True)
    assert rel_err(fast_out, ref) < 1e-10


def test_config_solver_options():
    """The config's solver keys reach the trainer's fit."""
    cfg = from_dict({"model": {"name": "sgdml", "extra": {"species": ["O", "H", "H"], "sigma": 2.0, "solver": "cg",
                                                          "solver_tol": 1e-6, "n_inducing": 4, "max_memory": 2}}})
    model = build_model(cfg.model)
    assert model.fit_options["solver"] == "cg" and model.fit_options["tol"] == 1e-6
    assert model.fit_options["n_inducing"] == 4 and model.fit_options["max_memory"] == 2.0
