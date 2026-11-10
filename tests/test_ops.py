"""Shared tensor operations (xnn.common.models.ops) on the CPU: the gather and scatter of message passing
against plain indexing, and the triplets of an empty graph."""
import torch

from xnn.common.models import ops


def test_gather_and_scatter_on_the_cpu():
    """The CPU forms equal indexing and an indexed sum; the size limit of the small forms is a positive power
    of two; nothing is being compiled here."""
    gen = torch.Generator().manual_seed(0)
    src = torch.randn(7, 3, dtype=torch.float64, generator=gen)
    index = torch.randint(0, 7, (20,), generator=gen)
    assert torch.equal(ops.gather_rows(src, index), src[index])
    rows = ops.gather_rows(src, index)
    want = torch.zeros(7, 3, dtype=torch.float64)
    for e, i in enumerate(index.tolist()):
        want[i] += rows[e]
    assert torch.allclose(ops.scatter_sum(rows, index, 7), want, rtol=1e-15, atol=1e-15)
    limit = ops.small_gather_elements()
    assert limit > 0 and limit & (limit - 1) == 0
    assert ops._compiling() is False


def test_whole_structure_sum_under_a_domain_reduction(monkeypatch):
    """The eager sum takes an active domain reduction; without one, and in scripted code, it is the plain
    per-structure sum."""
    gen = torch.Generator().manual_seed(1)
    src = torch.randn(6, 2, dtype=torch.float64, generator=gen)
    index = torch.tensor([0, 0, 1, 1, 1, 0])
    plain = ops.scatter_sum(src, index, 2)
    assert torch.equal(ops.whole_structure_sum(src, index, 2), plain)
    monkeypatch.setattr(ops, "_DOMAIN_REDUCTION", [lambda s, i, n: 2.0 * ops.scatter_sum(s, i, n)])
    assert torch.equal(ops.whole_structure_sum(src, index, 2), 2.0 * plain)
    monkeypatch.setattr(torch.jit, "is_scripting", lambda: True)
    assert torch.equal(ops.whole_structure_sum(src, index, 2), plain)


def test_triplets_of_an_empty_graph():
    """A graph without atoms has no triplets."""
    first, second, center = ops.build_triplets(torch.zeros(2, 0, dtype=torch.long), 0)
    assert first.numel() == second.numel() == center.numel() == 0 and first.dtype == torch.long
