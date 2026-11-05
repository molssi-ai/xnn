"""Pair searches of orthorhombic cells on the CPU kernels (xnn.common.data.grid_pairs) and the choice of the
search (xnn.common.data.neighborlist.cell_grid, unique_pairs) against the reference neighbor list."""
import sys

import pytest
import torch

from xnn.common import cpu_kernels
from xnn.common.data import neighborlist as nl

needs_kernels = pytest.mark.skipif(not cpu_kernels.available(), reason="the CPU kernels need a C++ compiler")


def _box(n=60, lengths=(12.0, 12.0, 12.0), seed=0):
    g = torch.Generator().manual_seed(seed)
    lengths = torch.tensor(lengths, dtype=torch.float64)
    return torch.rand((n, 3), generator=g, dtype=torch.float64) * lengths, torch.diag(lengths)


def _reference(monkeypatch, pos, cutoff, cell, pbc):
    """The pairs ``i < k`` of the reference neighbor list, as ``(i, k, shift)``."""
    with monkeypatch.context() as m:
        m.setattr(nl, "_HAS_VESIN", False)
        edges, shifts = nl.build_neighbor_list(pos, cutoff, cell, pbc)
    keep = edges[0] < edges[1]
    return {(int(a), int(b), *(int(v) for v in s)) for a, b, s in zip(edges[0][keep], edges[1][keep], shifts[keep])}


def _as_set(i, k, shifts):
    return {(int(a), int(b), *(int(v) for v in s)) for a, b, s in zip(i, k, shifts)}


@needs_kernels
def test_the_grid_search_into_buffers_of_a_fixed_capacity(monkeypatch):
    """On the CPU kernels the grid search finds the reference pairs; with a capacity the pairs fill the
    first entries, their number comes back as a tensor and the rest repeat the first pair."""
    from xnn.common.data.grid_pairs import grid_pairs, grid_shape
    pos, cell = _box()
    pbc = torch.ones(3, dtype=torch.bool)
    want = _reference(monkeypatch, pos, 3.0, cell, pbc)
    lengths = torch.diagonal(cell)
    shape = grid_shape(lengths.tolist(), 3.0)
    i, k, s = grid_pairs(pos, 3.0, lengths, shape)
    assert _as_set(i, k, s) == want and i.shape[0] == len(want)
    M = i.shape[0]
    i2, k2, s2, total = grid_pairs(pos, 3.0, lengths, shape, capacity=M + 5)
    assert int(total) == M and i2.shape[0] == M + 5
    assert torch.equal(i2[:M], i) and torch.equal(k2[:M], k) and torch.equal(s2[:M], s)
    assert bool((i2[M:] == i[0]).all() and (k2[M:] == k[0]).all() and (s2[M:] == s[0]).all())


@needs_kernels
def test_the_staged_grid_search(monkeypatch):
    """Each atom's staged partners within the cutoff and their image indices, from the grid or from every
    pair, give the reference pairs; an atom with more partners than its slots gives ``None``."""
    from xnn.common.data.grid_pairs import all_pairs_staged, grid_pairs_staged, grid_shape
    pos, cell = _box()
    pbc = torch.ones(3, dtype=torch.bool)
    lengths = torch.diagonal(cell)
    shape = grid_shape(lengths.tolist(), 3.0, 2)
    cap = 64
    counts, k, code = grid_pairs_staged(pos, 3.0, lengths, shape, 2, cap)
    found = set()
    for i in range(pos.shape[0]):
        for slot in range(int(counts[i])):
            c = int(code[i * (cap + 1) + slot])
            found.add((i, int(k[i * (cap + 1) + slot]), c // 9 - 1, c // 3 % 3 - 1, c % 3 - 1))
    want = _reference(monkeypatch, pos, 3.0, cell, pbc)
    assert found == want
    assert grid_pairs_staged(pos, 3.0, lengths, shape, 2, 1) is None
    counts, k, code = all_pairs_staged(pos, 3.0, lengths, cap)
    found = set()
    for i in range(pos.shape[0]):
        for slot in range(int(counts[i])):
            c = int(code[i * (cap + 1) + slot])
            found.add((i, int(k[i * (cap + 1) + slot]), c // 9 - 1, c // 3 % 3 - 1, c % 3 - 1))
    assert found == want
    assert all_pairs_staged(pos, 3.0, lengths, 1) is None


@needs_kernels
def test_cells_the_cpu_grid_does_not_search(monkeypatch):
    """On the CPU, a skewed cell, one too small for a grid, or a missing grid search go to vesin; a large
    orthorhombic box gets the grid of half-cutoff cells."""
    pos, cell = _box(n=nl.GRID_CPU_ATOMS, lengths=(40.0, 40.0, 40.0))
    pbc = torch.ones(3, dtype=torch.bool)
    grid = nl.cell_grid(pos, 4.0, cell, pbc)
    assert grid[1] == (20, 20, 20) and grid[3] == 2
    assert nl.cell_grid(pos[:100], 4.0, cell, pbc) is None
    skewed = cell.clone()
    skewed[1, 0] = 2.0
    assert nl.cell_grid(pos, 4.0, skewed, pbc) is None
    assert nl.cell_grid(pos, 4.0, torch.eye(3, dtype=torch.float64) * 8.0, pbc) is None
    monkeypatch.setitem(sys.modules, "xnn.common.data.grid_pairs", None)
    assert nl.cell_grid(pos, 4.0, cell, pbc) is None


def test_the_unique_pairs_of_a_slab(monkeypatch):
    """A cell periodic on two axes has no half list from vesin: the unique pairs come from the full list."""
    pos, cell = _box()
    pbc = torch.tensor([True, True, False])
    assert nl.half_list(pos, 3.0, cell, pbc) is None
    i, k, s = nl.unique_pairs(pos, 3.0, cell, pbc)
    assert _as_set(i, k, s) == _reference(monkeypatch, pos, 3.0, cell, pbc) and i.shape[0] == len(set(zip(
        i.tolist(), k.tolist(), map(tuple, s.tolist()))))
    assert bool((i[1:] >= i[:-1]).all())
