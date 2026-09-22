"""Shared bonded-term geometry for fixed-topology force fields.

Minimum-image displacement vectors, the dihedral cosine and a
differentiable-at-zero geometric mean, used by every fixed-topology model
(OPLS, DREIDING). All functions operate on a batched
:class:`~xnn.common.data.AtomicGraph` plus explicit atom-index tensors, so
bonded terms are correct for molecules wrapped across periodic boundaries
while gradients still flow to positions and cell.
"""
from __future__ import annotations

import torch
from torch import Tensor

from xnn.common.data import AtomicGraph

# Small numerical guard applied under square roots.
TINY = 1.0e-12


def geometric_mean(x: Tensor, y: Tensor) -> Tensor:
    """Geometric mean with a differentiable zero (for zero-LJ hydrogens).

    Parameters
    ----------
    x, y : Tensor
        Non-negative parameter values, gathered per pair.

    Returns
    -------
    Tensor
        ``sqrt(x * y)``, exactly zero (with zero gradient) where the
        product vanishes.
    """
    prod = x * y
    safe = torch.sqrt(torch.clamp(prod, min=TINY))
    return torch.where(prod > 0.0, safe, torch.zeros_like(prod))


def pair_vectors(data: AtomicGraph, a: Tensor, b: Tensor) -> Tensor:
    """Minimum-image displacement vectors ``pos[b] - pos[a]``.

    For periodic structures the integer image shift is recomputed from the
    fractional displacement (rounded, detached), so bonded terms are correct
    for molecules wrapped across the boundary while gradients still flow to
    positions and cell.

    Parameters
    ----------
    data : AtomicGraph
        The batched graph.
    a, b : Tensor
        Atom indices of shape ``(M,)``.

    Returns
    -------
    Tensor
        Displacements of shape ``(M, 3)``.
    """
    vec = data.pos[b] - data.pos[a]
    if data.cell is None or vec.shape[0] == 0:
        return vec
    cell = data.cell[data.batch[a]]                       # (M, 3, 3)
    inv = torch.linalg.inv(data.cell)[data.batch[a]]
    frac = torch.einsum("mi,mij->mj", vec, inv)
    shift = -torch.round(frac).detach()
    if data.pbc is not None:
        shift = shift * data.pbc[data.batch[a]].to(shift.dtype)
    return vec + torch.einsum("mi,mij->mj", shift, cell)


def dihedral_cos(data: AtomicGraph, idx: Tensor) -> Tensor:
    """Cosine of the dihedral angle over each atom quadruple.

    Uses the plane-normal formula with ``phi = 0`` at *cis*. Only
    ``cos phi`` is returned; periodic torsion terms are even in ``phi``, so
    the multiple angles come from Chebyshev identities and no
    ``arccos``/``atan2`` (whose derivatives are singular at planar
    geometries) enters the graph.

    Parameters
    ----------
    data : AtomicGraph
        The batched graph.
    idx : Tensor
        Atom indices of shape ``(4, M)``.

    Returns
    -------
    Tensor
        ``cos phi`` of shape ``(M,)``.
    """
    b1 = pair_vectors(data, idx[0], idx[1])
    b2 = pair_vectors(data, idx[1], idx[2])
    b3 = pair_vectors(data, idx[2], idx[3])
    n1 = torch.cross(b1, b2, dim=1)
    n2 = torch.cross(b2, b3, dim=1)
    denom = torch.sqrt((n1 * n1).sum(1) * (n2 * n2).sum(1) + TINY)
    return (n1 * n2).sum(1) / denom
