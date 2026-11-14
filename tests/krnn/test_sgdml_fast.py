"""The sGDML fast path (fused Triton kernels) against the reference expansion."""
import os

import pytest
import torch

from test_sgdml import FID, _graphs, check_fidelity, rel_err, reorder_tolerance
from xnn.common.data import collate
from xnn.common.models import ForceStressOutput
from xnn.krnn.common.sgdml_fit import fit_sgdml
from xnn.krnn.models import SGDML
from xnn.krnn.models.sgdml import _KernelEnergy, kernel_energy, kernel_energy_gradient


@pytest.fixture(scope="module")
def data():
    return FID.dataset(60, 7)


@pytest.fixture(scope="module")
def fitted(data):
    model = SGDML(data["z"].tolist(), sigma=4.0)
    fit_sgdml(model, _graphs(data, range(12)), energy_constraint=True)
    return model


@pytest.fixture
def interpreted(monkeypatch):
    """The fused kernels under Triton's interpreter (chosen by conftest.py on machines without
    CUDA), on CPU tensors, with the fast path offered on CPUs."""
    if os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("runs where Triton's interpreter is on (no CUDA device)")
    pytest.importorskip("triton")
    import xnn.krnn.fast.sgdml as fast
    from xnn.krnn.fast import sgdml_kernels
    monkeypatch.setattr(fast, "supported", lambda device, dtype: True)
    return sgdml_kernels


def _descriptors(model, data, idx):
    pos = torch.tensor(data["R"][idx], dtype=torch.float64)
    return model.descriptor(pos)


def _close(a, b, tol):
    assert rel_err(a, b) <= tol, (rel_err(a, b), tol)


def test_fast_support_needs_cuda():
    """Without a CUDA device the model keeps the reference whatever ``use_fast`` says."""
    from xnn.krnn.fast import supported
    assert not supported(torch.device("cpu"), torch.float64)
    model = SGDML([1, 8], sigma=1.0)
    x = torch.zeros(1, 1, dtype=torch.float64)
    assert not model.set_use_fast(True).uses_fast_path(x)
    with pytest.raises(ValueError, match="use_fast"):
        model.set_use_fast("always")


@pytest.mark.parametrize("energy_terms", [False, True])
def test_fast_energy_and_gradient(interpreted, monkeypatch, fitted, data, energy_terms):
    """Energies and gradients of the kernels equal the reference sums, also over chunks of the batch."""
    xp, wp = fitted.permuted_training_set()
    ec = fitted.energy_coef if energy_terms else fitted.energy_coef[:0]
    x = torch.cat([_descriptors(fitted, data, [40, 41, 42]), fitted.train_desc[:1]])   # one pair at r = 0
    e_ref, g_ref = kernel_energy(x, xp, wp, ec, fitted.kernel, gradient=True)
    tol_e, tol_g = reorder_tolerance(x, xp, wp, ec, fitted.kernel)
    e, g = interpreted.energy_gradient(x, xp, wp, ec, fitted.kernel)
    _close(e, e_ref, tol_e)
    _close(g, g_ref, tol_g)
    e_only, none = interpreted.energy_gradient(x, xp, wp, ec, fitted.kernel, gradient=False)
    assert none is None
    _close(e_only, e_ref, tol_e)
    monkeypatch.setattr(interpreted, "PAIR_CHUNK", 2 * xp.shape[0] * xp.shape[1])
    monkeypatch.setattr(interpreted, "PROGRAMS_PER_SM", 10_000)
    e2, g2 = interpreted.energy_gradient(x, xp, wp, ec, fitted.kernel)
    _close(e2, e_ref, tol_e)
    _close(g2, g_ref, tol_g)


@pytest.mark.parametrize("energy_terms", [False, True])
def test_fast_hessian_vector_products(interpreted, fitted, data, energy_terms):
    """Hessian-vector products of the kernels equal the derivative of the reference gradient."""
    xp, wp = fitted.permuted_training_set()
    ec = fitted.energy_coef if energy_terms else fitted.energy_coef[:0]
    x = torch.cat([_descriptors(fitted, data, [43, 44]), fitted.train_desc[1:2]]).requires_grad_(True)
    v = torch.randn(x.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    grad = kernel_energy_gradient(x, xp, wp, ec, fitted.kernel)
    ref = torch.autograd.grad(grad, x, v)[0]
    flipped = kernel_energy_gradient(x, xp.flip(0), wp.flip(0), ec.flip(0), fitted.kernel)
    tol = max(10 * rel_err(torch.autograd.grad(flipped, x, v)[0], ref), 1e-13)
    _close(interpreted.hessian_vector(x.detach(), v, xp, wp, ec, fitted.kernel), ref, tol)


def test_target_programs_follow_the_device(interpreted, monkeypatch):
    """The contraction aims for a fixed number of programs per multiprocessor of the device."""
    class Props:
        multi_processor_count = 108

    interpreted._multiprocessors.cache_clear()
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: Props())
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    try:
        per_sm = interpreted.PROGRAMS_PER_SM
        assert interpreted.target_programs(torch.device("cuda", 1)) == 108 * per_sm
        assert interpreted.target_programs(torch.device("cuda")) == 108 * per_sm
        assert interpreted.target_programs(torch.device("cpu")) == per_sm
    finally:
        interpreted._multiprocessors.cache_clear()


def test_fast_autograd_function(interpreted, fitted):
    """The fast energy function and its gradient pass gradcheck and gradgradcheck."""
    xp, wp = fitted.permuted_training_set()
    xp, wp = xp[:3], wp[:3]
    x = (xp[0, 0] + 0.01 * torch.randn(2, xp.shape[-1], dtype=torch.float64,
                                       generator=torch.Generator().manual_seed(3))).requires_grad_(True)
    ec = fitted.energy_coef[:3]

    def f(z):
        return interpreted.KernelEnergy.apply(z, xp, wp, ec, fitted.kernel)

    assert torch.autograd.gradcheck(f, (x,), fast_mode=True)
    assert torch.autograd.gradgradcheck(f, (x,), fast_mode=True)
    tol = reorder_tolerance(x, xp, wp, ec, fitted.kernel)[0]
    _close(f(x), _KernelEnergy.apply(x, xp, wp, ec, fitted.kernel), tol)
    _close(f(x.detach()), _KernelEnergy.apply(x.detach(), xp, wp, ec, fitted.kernel), tol)


def test_fast_path_parity_interpreted(interpreted, monkeypatch, fitted, data):
    """Energies, forces and Hessians of the model with the kernels equal the reference ones."""
    graph = collate(_graphs(data, [50, 51], labels=False))
    wrapped = ForceStressOutput(fitted).eval()
    pos = torch.tensor(data["R"][52], dtype=torch.float64).reshape(-1)

    def energy(p):
        return fitted.energy_of_descriptors(fitted.descriptor(p.reshape(1, -1, 3)))[0]

    out = {}
    ec = fitted.energy_coef
    try:
        for use_fast in (False, True):
            fitted.set_use_fast(use_fast)
            assert fitted.uses_fast_path(pos.reshape(1, -1)) is use_fast
            res = wrapped(graph)
            out[use_fast] = (res["energy"].detach(), res["forces"].detach(),
                             torch.autograd.functional.hessian(energy, pos))
        # the reference with its training terms summed in reverse order sets the tolerances
        fitted.set_use_fast(False)
        xp, wp = fitted.permuted_training_set()
        # a buffer is swapped by hand: monkeypatch reads old values from the instance dict and would delete it
        fitted.energy_coef = ec.flip(0)
        with monkeypatch.context() as m:
            m.setattr(fitted, "permuted_training_set", lambda: (xp.flip(0), wp.flip(0)))
            res = wrapped(graph)
            flipped = (res["energy"].detach(), res["forces"].detach(), torch.autograd.functional.hessian(energy, pos))
    finally:
        fitted.energy_coef = ec
        fitted.set_use_fast("auto")
    for fast, ref, other in zip(out[True], out[False], flipped):
        _close(fast, ref, max(10 * rel_err(other, ref), 1e-13))


def test_fast_path_parity_float32_interpreted(interpreted, data):
    """A float32 model gives the reference's float32 energies and forces with the kernels."""
    model = SGDML(data["z"].tolist(), sigma=4.0)
    fit_sgdml(model, _graphs(data, range(12)), energy_constraint=True)
    model = model.to(torch.float32)
    assert model.energy_coef.numel() > 0
    wrapped = ForceStressOutput(model).eval()
    graph = collate(_graphs(data, [50, 51], dtype=torch.float32, labels=False))
    out = {}
    for use_fast in (False, True):
        model.set_use_fast(use_fast)
        assert model.uses_fast_path(torch.zeros(1, 1, dtype=torch.float64)) is use_fast
        out[use_fast] = wrapped(graph)
    model.set_use_fast("auto")
    assert out[True]["forces"].dtype == torch.float32 and model.train_weights.dtype == torch.float64
    # both sum in float64 and round once to float32, so they differ by at most one unit in the last place
    for key in ("energy", "forces"):
        _close(out[True][key], out[False][key], 2 * torch.finfo(torch.float32).eps)


@pytest.mark.parametrize("case", ["sym", "nosym", "ecstr"])
def test_fast_path_fidelity_vs_sgdml(interpreted, case):
    """With the kernels, both precisions of the predictions and the refits match the sGDML package."""
    if not FID.have_sgdml():
        pytest.skip("the sGDML package is not installed")
    check_fidelity(FID.run_case(case, use_fast=True), fast=True)


def test_auto_policy_choice(interpreted, monkeypatch, fitted):
    """``"auto"`` follows the policy of the mode: never on a CPU, above the crossover on a GPU."""
    import xnn.krnn.fast.sgdml as fast
    x = torch.zeros(4, fitted.descriptor.dim, dtype=torch.float64)
    fitted.set_use_fast("auto")
    assert not fitted.uses_fast_path(x)
    seen = []

    class Policy:
        def __init__(self, name):
            self.name = name

        def decide(self, device, dtype, size):
            seen.append((self.name, size))
            return True

    monkeypatch.setattr(fast, "AUTO_POLICY", Policy("eval"))
    monkeypatch.setattr(fast, "AUTO_POLICY_TRAINING", Policy("train"))
    assert fitted.uses_fast_path(x)
    fitted.train()
    try:
        assert fitted.uses_fast_path(x)
    finally:
        fitted.eval()
    size = 4 * fitted.train_desc.shape[0] * fitted.perm_index.shape[0]
    assert seen == [("eval", size), ("train", size)]
