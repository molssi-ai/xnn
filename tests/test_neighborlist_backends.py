"""The vesin cell list must be a drop-in for the reference implementation.

Same edges, same [src, dst] order, same shift sign -- otherwise periodic edges
get the wrong displacement and are silently dropped by the cutoff envelope,
which is a wrong answer rather than an error.
"""

import pytest
import torch

from xnn.common.data import build_neighbor_list
from xnn.common.data.neighborlist import _vesin_neighbor_list

pytest.importorskip("vesin_torch", reason="vesin is optional")


def edge_set(edge_index, cell_shifts):
    return {
        (int(a), int(b), int(s[0]), int(s[1]), int(s[2]))
        for a, b, s in zip(edge_index[0], edge_index[1], cell_shifts)
    }


def reference(pos, cutoff, cell, pbc, self_interaction=False):
    """build_neighbor_list with the fast path disabled."""
    import xnn.common.data.neighborlist as nl

    saved = nl._HAS_VESIN
    nl._HAS_VESIN = False
    try:
        return nl.build_neighbor_list(pos, cutoff, cell, pbc, self_interaction)
    finally:
        nl._HAS_VESIN = saved


def water_like(n=60, seed=0):
    g = torch.Generator().manual_seed(seed)
    L = 12.0
    return torch.rand((n, 3), generator=g, dtype=torch.float64) * L, torch.eye(3, dtype=torch.float64) * L


@pytest.mark.parametrize("cutoff", [3.0, 6.0])
def test_periodic_matches_the_reference(cutoff):
    pos, cell = water_like()
    pbc = torch.ones(3, dtype=torch.bool)
    assert edge_set(*build_neighbor_list(pos, cutoff, cell, pbc)) == edge_set(
        *reference(pos, cutoff, cell, pbc)
    )


def test_molecular_matches_the_reference():
    pos, _ = water_like()
    assert edge_set(*build_neighbor_list(pos, 5.0, None, None)) == edge_set(
        *reference(pos, 5.0, None, None)
    )


def test_unwrapped_positions_match():
    """Shifts must stay relative to the positions as given, not to wrapped
    ones -- MD trajectories arrive unwrapped."""
    pos, cell = water_like()
    pbc = torch.ones(3, dtype=torch.bool)
    moved = pos + 2.5 * torch.diag(cell)
    assert edge_set(*build_neighbor_list(moved, 6.0, cell, pbc)) == edge_set(
        *reference(moved, 6.0, cell, pbc)
    )


def test_mixed_periodicity_falls_back():
    """vesin has one periodic flag, not three, so a slab must not use it."""
    pos, cell = water_like()
    pbc = torch.tensor([True, True, False])
    assert _vesin_neighbor_list(pos, 6.0, cell, pbc, False) is None
    # and the result is still correct
    assert edge_set(*build_neighbor_list(pos, 6.0, cell, pbc)) == edge_set(
        *reference(pos, 6.0, cell, pbc)
    )


def test_self_interaction_falls_back():
    """vesin cannot emit a zero-shift self-edge."""
    pos, cell = water_like()
    pbc = torch.ones(3, dtype=torch.bool)
    assert _vesin_neighbor_list(pos, 6.0, cell, pbc, True) is None
    ei, _ = build_neighbor_list(pos, 6.0, cell, pbc, self_interaction=True)
    assert (ei[0] == ei[1]).any(), "self-edges were requested but none appear"


def test_float32_is_accepted():
    pos, cell = water_like()
    pbc = torch.ones(3, dtype=torch.bool)
    ei, cs = build_neighbor_list(pos.float(), 6.0, cell.float(), pbc)
    assert ei.shape[0] == 2 and cs.shape[1] == 3


def test_shifts_are_integers():
    pos, cell = water_like()
    _, cs = build_neighbor_list(pos, 6.0, cell, torch.ones(3, dtype=torch.bool))
    assert cs.dtype == torch.long


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="needs an Apple mps device"
)
@pytest.mark.parametrize("periodic", [False, True])
def test_mps_positions_use_the_cpu_cell_list(periodic):
    """vesin insists on float64, which mps lacks: the cell list must be built
    on the CPU and the edges returned on the mps device, matching the reference."""
    pos64, cell64 = water_like()
    pos = pos64.to(torch.float32).to("mps")
    cell = cell64.to(torch.float32).to("mps") if periodic else None
    pbc = torch.ones(3, dtype=torch.bool, device="mps") if periodic else None
    edge_index, shifts = _vesin_neighbor_list(pos, 5.0, cell, pbc, False)
    assert edge_index.device.type == "mps" and shifts.device.type == "mps"
    ref = reference(pos64.to(torch.float32), 5.0,
                    None if cell is None else cell64.to(torch.float32),
                    None if pbc is None else pbc.cpu())
    assert edge_set(edge_index.cpu(), shifts.cpu()) == edge_set(*ref)
