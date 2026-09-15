"""Neighbor-list correctness, especially the periodic (cell-shift) convention.

Regression guard: the edge displacement reconstructed by
``AtomicGraph.edge_vectors`` (``pos[dst] - pos[src] + cell_shift @ cell``) must be
the true minimum-image displacement, so every edge length is <= cutoff. A wrong
shift sign leaves cross-boundary edges with lengths far beyond the cutoff, which
the smooth envelope then silently zeros -- i.e. all periodic neighbours dropped.
"""
import numpy as np
import torch

from xnns.common.data import build_neighbor_list, structure_to_graph


def _rng_crystal(n=40, L=8.0, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, L, (n, 3)), np.eye(3) * L


def test_periodic_edge_lengths_within_cutoff():
    """Every periodic edge's reconstructed length must be <= cutoff."""
    torch.set_default_dtype(torch.float64)
    pos, cell = _rng_crystal()
    cutoff = 4.0
    g = structure_to_graph(
        {"pos": pos, "atomic_numbers": [18] * len(pos), "cell": cell, "pbc": [True] * 3},
        cutoff,
    )
    lengths = g.edge_vectors().norm(dim=1)
    assert g.num_edges > 0
    # some edges must actually cross the boundary (nonzero shift), else the test is vacuous
    assert int((g.cell_shifts.abs().sum(1) > 0).sum()) > 0
    assert float(lengths.max()) <= cutoff + 1e-8


def test_matches_ase_neighbor_list():
    """Edge count and the multiset of edge lengths must match ASE's neighbor_list."""
    ase = __import__("importlib").import_module("ase")
    from ase import Atoms
    from ase.neighborlist import neighbor_list

    torch.set_default_dtype(torch.float64)
    pos, cell = _rng_crystal(n=60, L=10.0, seed=3)
    cutoff = 5.0
    atoms = Atoms(numbers=[18] * len(pos), positions=pos, cell=cell, pbc=True)
    _, _, d_ase = neighbor_list("ijd", atoms, cutoff)

    g = structure_to_graph(
        {"pos": pos, "atomic_numbers": [18] * len(pos), "cell": cell, "pbc": [True] * 3},
        cutoff,
    )
    d_xnns = g.edge_vectors().norm(dim=1).numpy()
    assert g.num_edges == len(d_ase)
    assert np.allclose(np.sort(d_xnns), np.sort(d_ase), atol=1e-8)


def test_unwrapped_positions_match_ase():
    """Positions outside the cell must give the same neighbors as ASE (which wraps)."""
    from ase import Atoms
    from ase.neighborlist import neighbor_list

    torch.set_default_dtype(torch.float64)
    pos, cell = _rng_crystal(n=40, L=8.0, seed=5)
    rng = np.random.default_rng(7)
    pos = pos + rng.integers(-3, 4, (len(pos), 3)) @ cell   # whole-lattice displacements
    cutoff = 4.0

    atoms = Atoms(numbers=[18] * len(pos), positions=pos, cell=cell, pbc=True)
    _, _, d_ase = neighbor_list("ijd", atoms, cutoff)

    g = structure_to_graph(
        {"pos": pos, "atomic_numbers": [18] * len(pos), "cell": cell, "pbc": [True] * 3},
        cutoff,
    )
    d_xnns = g.edge_vectors().norm(dim=1).numpy()
    assert g.num_edges == len(d_ase)
    assert np.allclose(np.sort(d_xnns), np.sort(d_ase), atol=1e-8)


def test_unwrapped_mixed_pbc_match_ase():
    """Unwrapped coords with periodicity on only two axes still match ASE."""
    from ase import Atoms
    from ase.neighborlist import neighbor_list

    torch.set_default_dtype(torch.float64)
    pos, cell = _rng_crystal(n=30, L=7.0, seed=9)
    pbc = [True, False, True]
    rng = np.random.default_rng(13)
    disp = rng.integers(-2, 3, (len(pos), 3))
    disp[:, 1] = 0                       # only displace along periodic axes
    pos = pos + disp @ cell
    cutoff = 3.5

    atoms = Atoms(numbers=[18] * len(pos), positions=pos, cell=cell, pbc=pbc)
    _, _, d_ase = neighbor_list("ijd", atoms, cutoff)

    g = structure_to_graph(
        {"pos": pos, "atomic_numbers": [18] * len(pos), "cell": cell, "pbc": pbc},
        cutoff,
    )
    d_xnns = g.edge_vectors().norm(dim=1).numpy()
    assert g.num_edges == len(d_ase)
    assert np.allclose(np.sort(d_xnns), np.sort(d_ase), atol=1e-8)


def test_molecular_shifts_are_zero():
    """Non-periodic systems have all-zero shifts and correct displacements."""
    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(1)
    pos = rng.uniform(0, 4, (8, 3))
    edge_index, shifts = build_neighbor_list(torch.tensor(pos), 3.0, None, None)
    assert torch.all(shifts == 0)
    assert edge_index.shape[0] == 2
