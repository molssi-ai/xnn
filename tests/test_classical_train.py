"""The kernels of the classical force fields differentiable twice (training) against their eager reference:
the gradients of every trained parameter from an energy loss and from a force loss, and the Hessian-vector
product in the positions, on periodic boxes and clusters of the test molecules of test_classical_fast,
float64 and float32. GPU only.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from xnn.common.data import structure_to_graph

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernels run on a GPU")

_spec = importlib.util.spec_from_file_location("_tcf", Path(__file__).with_name("test_classical_fast.py"))
tcf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tcf)
SYSTEMS = tcf.SYSTEMS


def _losses(model, s, cutoff, dtype, fast, w):
    """Energy, force-loss parameter gradients and the Hessian-vector product along ``w``."""
    model.use_fast = fast
    g = structure_to_graph(s, cutoff, device="cuda")
    g.compute_dtype = dtype
    pos = g.pos.detach().clone().requires_grad_(True)
    g.pos = pos
    energy = model(g)["energy"].sum()
    params = [p for p in model.parameters() if p.requires_grad]
    (forces,) = torch.autograd.grad(-energy, pos, create_graph=True)
    e_grads = torch.autograd.grad(energy, params, retain_graph=True, allow_unused=True)
    f_loss = (forces * w).sum()
    f_grads = torch.autograd.grad(f_loss, params + [pos], allow_unused=True)
    return energy.detach(), forces.detach(), e_grads, f_grads[:-1], f_grads[-1]


@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("case", SYSTEMS, ids=[c[0] for c in SYSTEMS])
def test_training_kernels_match_the_eager_derivatives(case, periodic):
    name, cls, lib, top1, z1, pos1, kw = case
    n1, n_side, cutoff = len(z1), 2, 8.0
    pos, L = tcf._box(pos1, n_side, 9.0)
    n_mol = n_side ** 3
    kw = dict(kw)
    if "charges" in kw:
        kw["charges"] = list(kw["charges"]) * n_mol
    model = cls(lib, tcf._replicate(top1, n_mol, n1), cutoff=cutoff, trainable="all", **kw)
    s = {"pos": torch.tensor(pos), "atomic_numbers": torch.tensor(list(z1) * n_mol)}
    if periodic:
        s.update(cell=torch.eye(3, dtype=torch.float64) * L, pbc=torch.ones(3, dtype=torch.bool))
    else:
        s["pos"] = torch.tensor(tcf._box(pos1, n_side, 9.0, unwrap=True)[0])
    w = torch.tensor(np.random.default_rng(2).normal(size=(n1 * n_mol, 3)), device="cuda")
    for dtype, tol in ((torch.float64, 1e-9), (torch.float32, 2e-4)):
        m = model.to(device="cuda", dtype=dtype).train()
        ref = _losses(m, s, cutoff, dtype, False, w.to(dtype))
        got = _losses(m, s, cutoff, dtype, True, w.to(dtype))
        pairs = [("energy", ref[0], got[0]), ("forces", ref[1], got[1]), ("hvp", ref[4], got[4])]
        pairs += [(f"energy grad {j}", a, b) for j, (a, b) in enumerate(zip(ref[2], got[2]))]
        pairs += [(f"force grad {j}", a, b) for j, (a, b) in enumerate(zip(ref[3], got[3]))]
        for label, a, b in pairs:
            if a is None or b is None:
                assert a is None or float(a.abs().max()) == 0.0, (name, label)
                assert b is None or float(b.abs().max()) == 0.0, (name, label)
                continue
            a, b = a.double(), b.double()
            scale = float(a.abs().max()) or 1.0
            assert float((a - b).abs().max()) <= tol * scale, (name, str(dtype), label,
                                                                 float((a - b).abs().max()), scale)


def test_the_training_kernels_refuse_a_stress_loss():
    """With a cell gradient (stress) and training, the eager terms run: the cell is a constant of the kernels."""
    from xnn.ffnn.models.classical_fast import wants_fast
    name, cls, lib, top1, z1, pos1, kw = SYSTEMS[0]
    pos, L = tcf._box(pos1, 1, 9.0)
    model = cls(lib, top1, cutoff=8.0, trainable="all", **kw).to("cuda").train()
    s = {"pos": torch.tensor(pos), "atomic_numbers": torch.tensor(list(z1)),
         "cell": torch.eye(3, dtype=torch.float64) * L, "pbc": torch.ones(3, dtype=torch.bool)}
    g = structure_to_graph(s, 8.0, device="cuda")
    assert wants_fast(model, True, g)
    g.cell = g.cell.clone().requires_grad_(True)
    assert not wants_fast(model, True, g)


@pytest.mark.parametrize("case", SYSTEMS, ids=[c[0] for c in SYSTEMS])
def test_evaluation_hessian_vector_products_match_the_eager_terms(case):
    """In evaluation mode (no trained parameter) the kernels are differentiable twice as well."""
    name, cls, lib, top1, z1, pos1, kw = case
    n1, n_side, cutoff = len(z1), 2, 8.0
    pos, L = tcf._box(pos1, n_side, 9.0)
    n_mol = n_side ** 3
    kw = dict(kw)
    if "charges" in kw:
        kw["charges"] = list(kw["charges"]) * n_mol
    model = cls(lib, tcf._replicate(top1, n_mol, n1), cutoff=cutoff, **kw).to("cuda").double().eval()
    s = {"pos": torch.tensor(pos), "atomic_numbers": torch.tensor(list(z1) * n_mol),
         "cell": torch.eye(3, dtype=torch.float64) * L, "pbc": torch.ones(3, dtype=torch.bool)}
    w = torch.tensor(np.random.default_rng(3).normal(size=pos.shape), device="cuda")
    hvp = []
    for fast in (False, True):
        model.use_fast = fast
        g = structure_to_graph(s, cutoff, device="cuda")
        p = g.pos.detach().clone().requires_grad_(True)
        g.pos = p
        (f,) = torch.autograd.grad(-model(g)["energy"].sum(), p, create_graph=True)
        hvp.append(torch.autograd.grad((f * w).sum(), p)[0])
    scale = float(hvp[0].abs().max())
    assert float((hvp[0] - hvp[1]).abs().max()) <= 1e-9 * scale, (name, float((hvp[0] - hvp[1]).abs().max()))


def test_a_second_derivative_with_a_cell_gradient_is_refused():
    """With a cell gradient (stress) the kernels are differentiable once: asking for more raises."""
    name, cls, lib, top1, z1, pos1, kw = SYSTEMS[0]
    pos, L = tcf._box(pos1, 1, 9.0)
    model = cls(lib, top1, cutoff=8.0, **kw).to("cuda").double().eval()
    s = {"pos": torch.tensor(pos), "atomic_numbers": torch.tensor(list(z1)),
         "cell": torch.eye(3, dtype=torch.float64) * L, "pbc": torch.ones(3, dtype=torch.bool)}
    g = structure_to_graph(s, 8.0, device="cuda")
    p = g.pos.detach().clone().requires_grad_(True)
    g.pos, g.cell = p, g.cell.clone().requires_grad_(True)
    energy = model(g)["energy"].sum()
    # first derivatives with a graph recorded pass; differentiating them again raises
    f, s = torch.autograd.grad(energy, (p, g.cell), create_graph=True)
    with pytest.raises(RuntimeError, match="differentiable once"):
        torch.autograd.grad(f.sum(), p)
