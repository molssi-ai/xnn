"""ByteFF-Pol's fused pair kernels under Triton's interpreter and the batches of a bound model's captured
evaluation, on the CPU."""
import os
import types

import numpy as np
import pytest
import torch

pytest.importorskip("rdkit")
from test_byteff import _box_graph, _byteff, _coords, _graphs, _mol, _pol_box, _z  # noqa: E402
from xnn.common.models import ForceStressOutput  # noqa: E402


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


@pytest.fixture
def interpreted(monkeypatch):
    """The pair kernels under Triton's interpreter (chosen by conftest.py on machines without CUDA), with
    stand-ins built from Triton's own math for the libdevice functions the interpreter lacks."""
    if os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("runs where Triton's interpreter is on (no CUDA device)")
    pytest.importorskip("triton")
    import triton.language as tl

    from xnn.hybnn.models import byteff_kernels
    math = types.SimpleNamespace(sqrt=lambda x: tl.sqrt(x), exp=lambda x: tl.exp(x),
                                 erfc=lambda x: 1.0 - tl.erf(x), pow=lambda x, y: tl.exp(y * tl.log(x)))
    monkeypatch.setattr(byteff_kernels, "libdevice", math)
    return byteff_kernels


def test_pol_pair_kernels_match_reference(interpreted):
    """The fused pair kernels give the reference pair block's energies, field, ladders and vector-Jacobian
    products on a step plan's pairs."""
    from xnn.ffnn.models.amoeba_kernels import pair_rows
    from xnn.hybnn.models.byteff import _pol_real_space
    model, pos, z, L = _pol_box()
    model = model.eval()
    graph = _box_graph(model, pos, z, L)
    plan = model.plan_step(graph, 0.3, capacity=None)
    src, dst, shift = plan.pairs
    r = (graph.pos[dst] - graph.pos[src] + shift.to(graph.pos.dtype) @ graph.cell[0]).detach()
    rows = pair_rows(src, dst, graph.num_nodes, int(plan.count), sorted_i=True)
    R = model._rows(model._parameters_cached(), 1, r.dtype)
    atoms = torch.stack([R["charge"], R["damp"]] + [R["atom_" + k] for k in ("eps", "lam", "r0", "c6", "ct_eps",
                                                                             "ct_lam")], 1).detach()
    consts = (model.cutoff ** 2, plan.ewald["alpha"], model.thole, model.coulomb_constant, model.repulsion_prefactor,
              model.wall, model.dispersion_damping)
    rr, rk = r.clone().requires_grad_(True), r.clone().requires_grad_(True)
    ref = _pol_real_space(rr, plan.valid, plan.scales, src, dst, atoms, graph.num_nodes, *consts)
    got = interpreted.pol_real_space(rk, src, dst, plan.scales, atoms, rows, plan.count, *consts)
    assert len(got) == len(ref) == 4
    for a, b in zip(got, ref):
        assert torch.allclose(a, b.detach(), rtol=1e-10, atol=1e-12 * float(b.abs().max()))
    gen = torch.Generator().manual_seed(0)
    cot = [torch.randn(t.shape, generator=gen) for t in ref]
    g_ref = torch.autograd.grad(sum((c * t).sum() for c, t in zip(cot, ref)), rr)[0]
    g_got = torch.autograd.grad(sum((c * t).sum() for c, t in zip(cot, got)), rk)[0]
    assert torch.allclose(g_got, g_ref, rtol=1e-10, atol=1e-12 * float(g_ref.abs().max()))


def test_graphed_evaluation_takes_bound_batches_unpadded():
    """A captured evaluation of a bound model takes each batch as it comes (detached, no index plan) and
    starts over when the batch's counts change."""
    from xnn.common.deploy import GraphedEvaluation
    mols = [_mol("CCO"), _mol("O")]
    model = _byteff(mols).eval()
    graphed = GraphedEvaluation(ForceStressOutput(model).eval())
    coords = np.concatenate([_coords(mols[0], n=2), _coords(mols[1], n=2) + 3.0], 1)
    z = _z(mols)
    one = _graphs(z, coords[:1], model)
    one.pos.requires_grad_(True)
    batch = graphed._pad(one)
    assert batch.num_nodes == one.num_nodes and not batch.pos.requires_grad and one.pos.requires_grad
    assert batch.index_plan == {} and batch.index_sizes == {}
    assert (graphed.cap.atoms, graphed.n_graphs) == (one.num_nodes, 1)
    graphed.steps = 3
    graphed._pad(_graphs(z, coords[1:], model))
    assert graphed.steps == 3
    two = _graphs(z, coords, model)
    graphed._pad(two)
    assert graphed.steps == 0 and (graphed.cap.atoms, graphed.n_graphs) == (two.num_nodes, 2)
