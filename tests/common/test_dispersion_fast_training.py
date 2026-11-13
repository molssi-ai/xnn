"""Second derivatives of the fast D3 / D4 three-body term (``dispersion_fast.atm``).

The gradient of the ATM kernels is itself differentiable: its backward runs the
generated tangent kernels (``atm_tangents``). Checked here: ``gradgradcheck`` of
:func:`~xnn.common.models.dispersion_fast.atm.atm_energy` with the factor shapes
of D3 (two matrices) and D4 (one shared), Hessian-vector products and the
parameter gradients of a force loss against the reference three-body term, and
the force-loss gradients of a small model with a D3 or D4 term between the fast
path and the reference in training mode. All need a GPU: the ATM kernels loop
over a runtime factor count, which Triton's interpreter does not run.
"""
import copy

import numpy as np
import pytest
import torch

from xnn.common.models import ForceStressOutput, fast
from xnn.common.models.dispersion import DispersionCorrection, three_body_energy_chunked
from xnn.common.models.dispersion_fast import supported

GPU = torch.cuda.is_available() and supported(torch.device("cuda"), torch.float64)
DEVICE = "cuda"
needs_gpu = pytest.mark.skipif(not GPU, reason="needs CUDA and Triton")

CUTOFF, WIDTH = 6.0, 2.5         # bohr: most pairs of the systems below lie in the switching window
ALP3 = 16.0 / 3.0
D4_K, D3_REFS = 23, 3


def _rel(a, b):
    """Largest deviation relative to the largest entry of the reference."""
    scale = b.detach().abs().max().clamp(min=1e-30)
    return float((a.detach() - b.detach()).abs().max() / scale)


def _system(periodic, dtype, device):
    """Two species on a jittered grid (bohr): a cluster of six atoms, or two atoms in a
    cell narrower than twice the cutoff (self-image triangles). Returns the
    positions, species, directed edges ``[src, dst]`` within the cutoff and their
    constant cell offsets."""
    rng = np.random.default_rng(7)
    if periodic:
        pos = np.array([[0.4, 0.3, 0.2], [2.9, 2.6, 3.0]]) + rng.normal(scale=0.2, size=(2, 3))
        box = 5.5
        rng_shift = range(-2, 3)
        shifts = np.array([[a, b, c] for a in rng_shift for b in rng_shift for c in rng_shift]) * box
    else:
        grid = np.array([[i, j, 0] for i in range(3) for j in range(2)], dtype=float)
        pos = grid * 2.8 + rng.normal(scale=0.25, size=(6, 3))
        pos[:, 2] += rng.normal(scale=0.6, size=6)
        shifts = np.zeros((1, 3))
    n = pos.shape[0]
    src, dst, off = [], [], []
    for i in range(n):
        for j in range(n):
            for s in shifts:
                d = np.linalg.norm(pos[j] + s - pos[i])
                if 1e-8 < d <= CUTOFF:
                    src.append(j)
                    dst.append(i)
                    off.append(s)
    edge_index = torch.tensor([src, dst], device=device)
    z = torch.tensor([0, 1] * (n // 2), device=device)
    return (torch.tensor(pos, dtype=dtype, device=device), z, edge_index,
            torch.tensor(np.array(off), dtype=dtype, device=device))


def _consts(z, dtype, sharp=False):
    from xnn.common.models.dispersion_fast.atm import ATMConstants
    table = torch.tensor([[1.55, 1.62], [1.62, 1.70]], dtype=dtype, device=z.device)
    return ATMConstants(z.to(torch.int32).contiguous(), table, 1.0, ALP3, CUTOFF,
                        0.0 if sharp else WIDTH)


def _features(method, pos, z, edge_index, scale):
    """Pair-C6 factors as smooth functions of the positions (through a coordination
    number, like the CN dependence of D3 and D4) and of a parameter ``scale``:
    D4's ``f = g`` ``(N, 23)``, or D3's species-projected rows ``f`` and the
    one-hot weights ``g`` ``(N, 2 R)``, whose ``<f_x, g_y>`` is symmetric."""
    n = pos.shape[0]
    src, dst = edge_index
    gen = torch.Generator().manual_seed(11)
    r2 = (pos[src] - pos[dst]).square().sum(-1)
    cn = torch.zeros(n, dtype=pos.dtype, device=pos.device).index_add(0, dst, torch.exp(-r2 / 8.0))
    if method == "d4":
        base = (1.0 + torch.rand(n, D4_K, generator=gen, dtype=pos.dtype)).to(pos.device)
        f = scale * base * (1.0 + 0.2 * torch.tanh(cn))[:, None]
        return f, f
    refcn = torch.tensor([0.0, 1.0, 2.0], dtype=pos.dtype, device=pos.device)
    w = torch.softmax(-(cn[:, None] - refcn[None, :]).square(), dim=1)  # (N, R)
    c6ref = (5.0 + 20.0 * torch.rand(2, 2, D3_REFS, D3_REFS, generator=gen, dtype=pos.dtype)).to(pos.device)
    c6ref = 0.5 * (c6ref + c6ref.permute(1, 0, 3, 2)) * scale           # C6ref[s, t, a, b] = C6ref[t, s, b, a]
    vectors = torch.einsum("xa,xtab->xbt", w, c6ref[z])                 # (N, R, S)
    f = vectors.transpose(1, 2).reshape(n, -1)
    own = torch.nn.functional.one_hot(z, 2).to(pos.dtype)
    g = (own[:, :, None] * w[:, None, :]).reshape(n, -1)
    return f, g


def _energies(use_fast, method, periodic, dtype, device, sharp=False):
    """Per-atom ATM energies of the system above, its positions, the parameter and
    the per-atom weights (all differentiable)."""
    from xnn.common.models.dispersion_fast.atm import atm_energy
    pos, z, edge_index, off = _system(periodic, dtype, device)
    pos.requires_grad_(True)
    scale = torch.tensor(1.0, dtype=dtype, device=device, requires_grad=True)
    weights = (1.0 + 0.1 * torch.arange(pos.shape[0], dtype=dtype, device=device)).requires_grad_(True)
    vec = pos[edge_index[0]] - pos[edge_index[1]] + off
    fa, fb = _features(method, pos, z, edge_index, scale)
    consts = _consts(z, dtype, sharp)
    if use_fast:
        e = atm_energy(edge_index, vec, fa, fb, consts, pos.shape[0])
    else:
        e = three_body_energy_chunked(z, edge_index, vec, torch.linalg.norm(vec, dim=-1), consts.table,
                                      torch.tensor(1.0, dtype=dtype, device=device), ALP3, CUTOFF,
                                      0.0 if sharp else WIDTH, pos.shape[0], alpha_a=fa, alpha_b=fb)
    return e, pos, scale, weights


# the kernels' second derivatives
@needs_gpu
@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("method", ["d4", "d3"])
def test_atm_gradgradcheck(method, periodic):
    """Second derivatives of the kernels against finite differences of their gradient
    (edge vectors, factors and the per-atom cotangents), float64."""
    from xnn.common.models.dispersion_fast.atm import atm_energy
    pos, z, edge_index, off = _system(periodic, torch.float64, DEVICE)
    vec = (pos[edge_index[0]] - pos[edge_index[1]] + off).detach().requires_grad_(True)
    fa, fb = _features(method, pos, z, edge_index, torch.tensor(1.0, dtype=torch.float64, device=DEVICE))
    fa, fb = fa.detach().requires_grad_(True), fb.detach().requires_grad_(True)
    consts = _consts(z, torch.float64)
    n = pos.shape[0]
    if method == "d4":
        def fn(v, f):
            return atm_energy(edge_index, v, f, f, consts, n)
        inputs = (vec, fa)
    else:
        def fn(v, f, g):
            return atm_energy(edge_index, v, f, g, consts, n)
        inputs = (vec, fa, fb)
    assert float(fn(*inputs).abs().max()) > 0
    assert torch.autograd.gradgradcheck(fn, inputs, eps=1e-6, atol=1e-9, rtol=1e-6, nondet_tol=1e-12)


@needs_gpu
@pytest.mark.parametrize("sharp", [False, True])
@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("method", ["d4", "d3"])
def test_atm_second_derivatives_match_the_reference(method, periodic, sharp):
    """Energies, forces, a Hessian-vector product (random probe: the forces sum to
    zero, so a uniform one gives Hv = 0) and the parameter gradients of a force
    loss, kernels against the reference three-body term, float64."""
    out = {}
    for use_fast in (False, True):
        e, pos, scale, weights = _energies(use_fast, method, periodic, torch.float64, DEVICE, sharp)
        energy = (e * weights).sum()
        forces = -torch.autograd.grad(energy, pos, create_graph=True)[0]
        probe = torch.randn(forces.shape, generator=torch.Generator().manual_seed(0),
                            dtype=forces.dtype).to(forces.device)
        hv = torch.autograd.grad((forces * probe).sum(), pos, retain_graph=True)[0]
        g_scale, g_weights = torch.autograd.grad(forces.square().sum() + energy.square(), (scale, weights))
        out[use_fast] = {"energy": e, "forces": forces, "hv": hv, "scale": g_scale, "weights": g_weights}
    assert float(out[False]["hv"].abs().max()) > 1e-8
    for key, ref in out[False].items():
        err = _rel(out[True][key], ref)
        print(f"atm {method} periodic={periodic} sharp={sharp} {key}: {err:.2e}")
        assert err < 1e-10, key


# a model with a dispersion term, training mode
WATER = np.array([[0.0, 0.0, 0.119262], [0.0, 0.763239, -0.477047], [0.0, -0.763239, -0.477047]])
OPTIONS = dict(cutoff_pair=9.0, switch_width_pair=2.0, cutoff_triple=6.0, switch_width_triple=1.0, s9=1.0,
               cutoff_cn=9.0)
D4_ONLY = dict(cutoff_eeq_cn=9.0)


def _water_cell(dtype, n_side=3, spacing=3.104, seed=3):
    rng = np.random.default_rng(seed)
    pos = []
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
                q *= np.sign(np.linalg.det(q))
                pos.extend((WATER - WATER[0]) @ q.T + np.array([i, j, k]) * spacing + rng.normal(scale=0.1, size=3))
    return {"pos": torch.tensor(np.array(pos), dtype=dtype), "atomic_numbers": [8, 1, 1] * n_side ** 3,
            "cell": torch.eye(3, dtype=dtype) * spacing * n_side, "pbc": [True, True, True]}


def _set_dispersion_fast(model, use_fast):
    for m in model.modules():
        if isinstance(m, DispersionCorrection):
            m.set_use_fast(use_fast)


def _count_grad_kernels(monkeypatch):
    import xnn.common.models.dispersion_fast.atm as atm
    calls = []
    original = atm._ATMGrad.apply
    monkeypatch.setattr(atm._ATMGrad, "apply", lambda *a: calls.append(1) or original(*a))
    return calls


@needs_gpu
@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-10), (torch.float32, 5e-5)])
@pytest.mark.parametrize("method", ["d4", "d3"])
def test_model_force_loss_gradients(method, dtype, tol, monkeypatch):
    """A small MACE with a D3 or D4 term in training mode: the parameter gradients of
    an energy and force loss, the fast three-body term against the reference,
    relative to the largest gradient."""
    from xnn.common.config import from_dict
    from xnn.common.data import structure_to_graph
    from xnn.common.models import build_model
    disp = {**OPTIONS, **(D4_ONLY if method == "d4" else {"name": "d3"})}
    cfg = from_dict({"model": {"name": "mace", "cutoff": 4.0, "n_interactions": 1, "n_rbf": 6,
                               "n_features": 8, "extra": {"species": [1, 8], "l_max": 1, "dispersion": disp}}})
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(0)
        model = ForceStressOutput(build_model(cfg.model), compute_stress=True).to(device="cuda", dtype=dtype)
    finally:
        torch.set_default_dtype(old)
    model.train()
    graph = structure_to_graph(_water_cell(dtype), model.model.cutoff, device="cuda")
    calls = _count_grad_kernels(monkeypatch)
    grads = {}
    for use_fast in (False, True):
        _set_dispersion_fast(model, use_fast)
        model.zero_grad(set_to_none=True)
        out = model(copy.copy(graph))
        (out["energy"].square().sum() + out["forces"].square().sum() + out["stress"].square().sum()).backward()
        grads[use_fast] = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        assert bool(calls) is use_fast
    assert grads[True].keys() == grads[False].keys() and grads[True]
    scale = max(float(g.abs().max()) for g in grads[False].values())
    worst = max(float((grads[True][n] - g).abs().max()) / scale for n, g in grads[False].items())
    print(f"model {method} {dtype}: worst parameter gradient {worst:.2e} of the largest")
    assert worst < tol


@needs_gpu
@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-10), (torch.float32, 5e-5)])
@pytest.mark.parametrize("method", ["d4", "d3"])
def test_training_mode_hessian_vector_products(method, dtype, tol, monkeypatch):
    """A D3 / D4 term alone in training mode: forces and a Hessian-vector product with
    a random probe, the fast path against the reference."""
    from xnn.common.data import structure_to_graph
    from xnn.common.models import D3Dispersion, D4Dispersion
    cls = D4Dispersion if method == "d4" else D3Dispersion
    disp = cls(**OPTIONS, **(D4_ONLY if method == "d4" else {})).to(device="cuda", dtype=dtype).train()
    calls = _count_grad_kernels(monkeypatch)
    out = {}
    for use_fast in (False, True):
        fast.set_use_fast(disp, use_fast)
        graph = structure_to_graph(_water_cell(dtype), disp.cutoff, device="cuda")
        graph.pos.requires_grad_(True)
        res = disp(graph)
        assert disp.term.fast_active is use_fast
        forces = -torch.autograd.grad(res["energy"].sum(), graph.pos, create_graph=True)[0]
        probe = torch.randn(forces.shape, generator=torch.Generator().manual_seed(0), dtype=forces.dtype)
        hv = torch.autograd.grad((forces * probe.to(forces.device)).sum(), graph.pos)[0]
        out[use_fast] = {"energy_3body": res["energy_3body"].detach(), "forces": forces.detach(), "hv": hv}
        assert bool(calls) is use_fast
    assert float(out[False]["hv"].abs().max()) > 1e-6
    for key, ref in out[False].items():
        err = _rel(out[True][key], ref)
        print(f"hvp {method} {dtype} {key}: {err:.2e}")
        assert err < tol, key
