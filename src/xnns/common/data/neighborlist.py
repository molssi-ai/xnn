"""Periodic-boundary-aware neighbor list.

Produces, for a single structure, the edge list and the *integer* periodic
image shift per edge within a cutoff. The same routine handles molecular
systems (``cell=None`` / no pbc) by simply skipping image enumeration.

This is intentionally a clear reference implementation (per-structure, brute
force over the minimal set of image shifts). It is correct for both small
molecules and dense crystals. For very large systems swap in a cell-list /
linked-cell algorithm or `ase.neighborlist` / `matscipy` -- the interface
(``edge_index``, ``cell_shifts``) stays identical.
"""
from __future__ import annotations

import itertools
from typing import Optional

import torch
from torch import Tensor


def _n_repeats(cell: Tensor, cutoff: float, pbc: Tensor) -> list[int]:
    """Compute how many lattice repeats per axis are needed to cover ``cutoff``.

    The number of repeats along axis ``i`` is derived from the interplanar
    spacing (``1 / |reciprocal_i|``); axes with periodicity disabled get zero
    repeats.

    Parameters
    ----------
    cell : Tensor
        Lattice vectors as rows, of shape ``(3, 3)``.
    cutoff : float
        Neighbor cutoff radius.
    pbc : Tensor
        Boolean periodicity flags of shape ``(3,)``.

    Returns
    -------
    list of int
        Number of image repeats required along each of the three axes.
    """
    # Interplanar spacing along axis i is 1 / |reciprocal_i|.
    recip = torch.linalg.inv(cell).T            # rows are reciprocal vectors
    spacing = 1.0 / torch.linalg.norm(recip, dim=1)
    reps = []
    for i in range(3):
        if bool(pbc[i]):
            reps.append(int(torch.ceil(cutoff / spacing[i]).item()))
        else:
            reps.append(0)
    return reps


def build_neighbor_list(
    pos: Tensor,                 # (N, 3)
    cutoff: float,
    cell: Optional[Tensor] = None,   # (3, 3) rows = lattice vectors
    pbc: Optional[Tensor] = None,    # (3,) bool
    self_interaction: bool = False,
) -> tuple[Tensor, Tensor]:
    """Build a periodic-boundary-aware neighbor list for a single structure.

    Enumerates the minimal set of periodic image shifts covering ``cutoff`` and
    brute-forces all pairwise distances to select edges within the cutoff. For
    molecular systems (``cell`` or ``pbc`` is ``None``, or no periodicity) image
    enumeration is skipped. Edges follow the convention ``dst = i`` (receiver)
    and ``src = j`` (sender), and the returned ``cell_shifts`` are negated so
    that :meth:`AtomicGraph.edge_vectors` reproduces the selecting displacement.

    Positions need not lie inside the cell: along periodic axes they are
    wrapped internally (image enumeration assumes in-cell positions) and the
    removed integer offsets are folded back into the returned ``cell_shifts``,
    so the shifts remain consistent with the *original* positions. Unwrapped
    trajectories (e.g. from MD) therefore work as-is.

    Parameters
    ----------
    pos : Tensor
        Cartesian positions of shape ``(N, 3)``.
    cutoff : float
        Neighbor cutoff radius; pairs with distance below this are kept.
    cell : Tensor, optional
        Lattice vectors as rows, of shape ``(3, 3)``. ``None`` for molecular
        systems.
    pbc : Tensor, optional
        Boolean periodicity flags of shape ``(3,)``. ``None`` for molecular
        systems.
    self_interaction : bool, default False
        If ``False``, drop self-edges (an atom to itself in the zero-shift
        image).

    Returns
    -------
    edge_index : Tensor
        Edge list of shape ``(2, E)`` holding ``[src, dst]`` node indices.
    cell_shifts : Tensor
        Integer periodic image shift per edge, of shape ``(E, 3)``.
    """
    device, dtype = pos.device, pos.dtype
    n = pos.shape[0]

    offsets = None
    if cell is None or pbc is None or not bool(pbc.any()):
        shifts = torch.zeros((1, 3), device=device, dtype=dtype)
        shift_idx = torch.zeros((1, 3), device=device, dtype=torch.long)
    else:
        # Wrap positions into the cell along periodic axes; the integer image
        # offsets removed here are added back to the returned shifts below.
        frac = pos @ torch.linalg.inv(cell)
        offsets = torch.floor(frac).to(torch.long)
        offsets[:, ~pbc.to(torch.bool)] = 0
        pos = pos - offsets.to(dtype) @ cell
        reps = _n_repeats(cell, cutoff, pbc)
        ranges = [range(-r, r + 1) for r in reps]
        combos = list(itertools.product(*ranges))
        shift_idx = torch.tensor(combos, device=device, dtype=torch.long)  # (S,3)
        shifts = shift_idx.to(dtype) @ cell                                # (S,3)

    # pairwise displacement for every image: (S, N, N, 3)
    rij = (pos[None, None, :, :] - pos[None, :, None, :]) + shifts[:, None, None, :]
    dist = torch.linalg.norm(rij, dim=-1)                                  # (S,N,N)

    within = dist < cutoff
    if not self_interaction:
        eye = torch.eye(n, dtype=torch.bool, device=device)
        zero_shift = (shift_idx == 0).all(dim=1)                           # (S,)
        within = within & ~(eye[None] & zero_shift[:, None, None])

    s_idx, i_idx, j_idx = torch.nonzero(within, as_tuple=True)
    # convention: dst = i (receiver), src = j (sender). The selecting displacement
    # was rij = pos[j] - pos[i] + shift@cell, so the min-image vector consumed by
    # `AtomicGraph.edge_vectors` (pos[dst] - pos[src] + cell_shift@cell
    #  = pos[i] - pos[j] + cell_shift@cell) equals that displacement only with the
    # negated shift. Without this, periodic (nonzero-shift) edges get the wrong
    # displacement (length far beyond the cutoff) and are silently killed by the
    # cutoff envelope -- i.e. all cross-boundary neighbours would be dropped.
    edge_index = torch.stack([j_idx, i_idx], dim=0)                        # (2, E)
    cell_shifts = -shift_idx[s_idx]                                        # (E, 3)
    if offsets is not None:
        # Re-express the shifts relative to the original (unwrapped) positions:
        # pos = wrapped + offset @ cell, so shift_unwrapped = shift + o_src - o_dst.
        cell_shifts = cell_shifts + offsets[j_idx] - offsets[i_idx]
    return edge_index, cell_shifts
