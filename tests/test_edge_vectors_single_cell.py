"""The single-structure fast path of AtomicGraph.edge_vectors matches the batched one."""
import numpy as np
import torch

from xnn.common.data import collate, structure_to_graph


def test_single_cell_edge_vectors_and_gradients_match_batched():
    torch.manual_seed(0)
    rng = np.random.default_rng(4)
    cell = np.array([[6.1, 0.0, 0.0], [0.7, 5.8, 0.0], [0.3, -0.4, 6.4]])
    pos = rng.uniform(0.0, 6.0, (12, 3))
    s = {"pos": pos, "atomic_numbers": [8, 1, 1] * 4, "cell": cell, "pbc": [True] * 3}
    single = structure_to_graph(s, 4.0)
    # the same structure twice in one batch takes the gather path
    batched = collate([structure_to_graph(s, 4.0), structure_to_graph(s, 4.0)])
    for g in (single, batched):
        g.pos = g.pos.detach().double().requires_grad_(True)
        g.cell = g.cell.detach().double().requires_grad_(True)
    v1 = single.edge_vectors()
    v2 = batched.edge_vectors()
    e = v1.shape[0]
    assert torch.equal(v1, v2[:e])
    w = torch.randn(e, 3, dtype=torch.float64)
    g1 = torch.autograd.grad((v1 * w).sum(), (single.pos, single.cell))
    g2 = torch.autograd.grad((v2[:e] * w).sum(), (batched.pos, batched.cell))
    n = single.pos.shape[0]
    torch.testing.assert_close(g1[0], g2[0][:n])
    torch.testing.assert_close(g1[1][0], g2[1][0])
