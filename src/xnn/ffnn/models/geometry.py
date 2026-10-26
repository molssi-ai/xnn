"""Shared bonded-term geometry for fixed-topology force fields.

Minimum-image displacement vectors, the dihedral cosine and a
differentiable-at-zero geometric mean, used by every fixed-topology model
(OPLS, DREIDING). All functions operate on a batched
:class:`~xnn.common.data.AtomicGraph` plus explicit atom-index tensors, so
bonded terms are correct for molecules wrapped across periodic boundaries
while gradients still flow to positions and cell.

The minimum image of a displacement is found in two steps: round its
fractional coordinates to the nearest lattice point, and, for a cell with
non-orthogonal axes, pick the shortest of the 27 images around that point.
Rounding alone is exact for vectors shorter than half the smallest spacing
between lattice planes; the 27-image search is exact for every vector on a
reduced cell (the Wigner-Seitz cell of a Minkowski-reduced lattice lies
inside the +-1 shell of its parallelepiped). Cells must therefore be reduced
once at the data level (``ase.geometry.minkowski_reduce``), the convention
of every molecular-dynamics code, rather than searched in every evaluation.
"""
from __future__ import annotations

import warnings
from typing import Optional

import torch
from torch import Tensor

from xnn.common.data import AtomicGraph

# Small numerical guard applied under square roots.
TINY = 1.0e-12
# Relative size of the off-diagonal Gram-matrix entries below which a cell
# counts as orthogonal (rounding is exact, no image search).
ORTHOGONAL_TOL = 1.0e-10
UNREDUCED_CELL_MESSAGE = (
    "a cell is not reduced (|a_i . a_j| > min(|a_i|^2, |a_j|^2) / 2 for some pair of "
    "lattice vectors), so bonded minimum images are not guaranteed to be the shortest; "
    "reduce it once, e.g. with ase.geometry.minkowski_reduce")


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


def _periodic_lattice(cell: Tensor, pbc: Optional[Tensor]) -> Tensor:
    """``cell`` with the rows of non-periodic axes zeroed: the lattice that images live on."""
    return cell if pbc is None else cell * pbc.to(cell.dtype)[..., None]


def _lattice_inverse(lattice: Tensor, pbc: Optional[Tensor]) -> Tensor:
    """The inverse of fully periodic lattices; otherwise the pseudo-inverse,
    which maps a vector to the fractional coordinates of its projection onto
    the periodic sublattice (and is zero for the zero cell of a molecule)."""
    if pbc is None or bool(pbc.all()):
        return torch.linalg.inv_ex(lattice)[0]
    return torch.linalg.pinv(lattice)


def _image_shifts(vec: Tensor, lattice: Tensor, inv: Tensor, pbc: Optional[Tensor]) -> Tensor:
    """Integer image shifts ``(M, 3)`` that minimize ``|vec + shift @ lattice|``.

    ``vec`` ``(M, 3)``; ``lattice`` ``(M, 3, 3)`` the periodic lattice vectors
    per displacement (:func:`_periodic_lattice`) and ``inv`` its
    pseudo-inverse, which maps a vector to the fractional coordinates of its
    projection onto the periodic sublattice; ``pbc`` ``(M, 3)`` bool or
    ``None``. The rounded image is returned for orthogonal lattices; otherwise
    the best of its 27 neighbours. Nothing here carries gradients.
    """
    with torch.no_grad():
        shift = -torch.round(torch.einsum("mi,mij->mj", vec, inv))
        mask = None if pbc is None else pbc.to(vec.dtype)
        if mask is not None:
            shift = shift * mask
        gram = lattice @ lattice.transpose(1, 2)
        diag = torch.diagonal(gram, dim1=1, dim2=2)
        off = gram - torch.diag_embed(diag)
        # one host sync for both flags: any non-orthogonal lattice, any unreduced lattice
        half = 0.5 * torch.minimum(diag[:, :, None], diag[:, None, :]) * (1.0 + 1.0e-8)
        skewed, unreduced = torch.stack([(off.abs() > ORTHOGONAL_TOL * gram.abs().amax()).any(),
                                         (off.abs() > half).any()]).tolist()
        if unreduced:
            warnings.warn(UNREDUCED_CELL_MESSAGE, stacklevel=3)
        if skewed:
            one = torch.arange(-1, 2, device=vec.device)
            offsets = torch.cartesian_prod(one, one, one).to(vec.dtype)          # (27, 3)
            offsets = offsets[None].expand(vec.shape[0], 27, 3)
            if mask is not None:
                offsets = offsets * mask[:, None, :]
            cand = vec[:, None, :] + torch.einsum("mki,mij->mkj", shift[:, None, :] + offsets, lattice)
            k = (cand * cand).sum(-1).argmin(1)
            shift = shift + offsets[torch.arange(vec.shape[0], device=vec.device), k]
    return shift


def minimum_image_vectors(vec: Tensor, cell: Tensor, pbc: Optional[Tensor] = None) -> Tensor:
    """Minimum-image displacements in one, possibly triclinic, cell.

    Parameters
    ----------
    vec : Tensor
        Displacement vectors ``(M, 3)``.
    cell : Tensor
        Lattice vectors as rows ``(3, 3)``; a reduced cell for the result to
        be the true minimum image at every distance (see the module notes).
    pbc : Tensor or None, optional
        Periodic axes ``(3,)``; ``None`` means all three.

    Returns
    -------
    Tensor
        Shortest images ``(M, 3)``, differentiable in ``vec`` and ``cell``
        through fixed integer image shifts.
    """
    if vec.shape[0] == 0 or (pbc is not None and not bool(pbc.any())):
        return vec
    m = vec.shape[0]
    flags = None if pbc is None else pbc.to(device=vec.device, dtype=torch.bool)
    lattice = _periodic_lattice(cell.to(vec.dtype), flags)
    inv = _lattice_inverse(lattice.detach(), flags)
    shift = _image_shifts(vec.detach(), lattice.detach()[None].expand(m, 3, 3), inv[None].expand(m, 3, 3),
                          None if flags is None else flags[None].expand(m, 3))
    return vec + shift @ cell.to(vec.dtype)


def pair_image_shifts(data: AtomicGraph, a: Tensor, b: Tensor) -> Optional[Tensor]:
    """The integer image shifts ``(M, 3)`` that :func:`pair_vectors` adds to
    ``pos[b] - pos[a]``, or ``None`` for a batch without cells.

    Parameters
    ----------
    data : AtomicGraph
        The batched graph.
    a, b : Tensor
        Atom indices of shape ``(M,)``.

    Returns
    -------
    Tensor or None
        Shifts in lattice vectors, detached.
    """
    if data.cell is None or a.shape[0] == 0:
        return None
    vec = data.pos[b] - data.pos[a]
    idx = data.batch[a]
    lattice = _periodic_lattice(data.cell.to(vec.dtype).detach(), data.pbc)
    inv = _lattice_inverse(lattice, data.pbc)
    pbc = None if data.pbc is None else data.pbc[idx]
    return _image_shifts(vec.detach(), lattice[idx], inv[idx], pbc)


def pair_vectors(data: AtomicGraph, a: Tensor, b: Tensor, shift: Optional[Tensor] = None) -> Tensor:
    """Minimum-image displacement vectors ``pos[b] - pos[a]``.

    For periodic structures the integer image shift is recomputed from the
    fractional displacement (rounded, then the shortest of the 27 images
    around it for non-orthogonal cells, detached), so bonded terms are
    correct for molecules wrapped across the boundary while gradients still
    flow to positions and cell. Only periodic axes shift: a slab's images are
    searched on its two-dimensional lattice, and molecular structures of a
    mixed batch (zero cell, no periodic axis) keep their plain displacement.

    Parameters
    ----------
    data : AtomicGraph
        The batched graph.
    a, b : Tensor
        Atom indices of shape ``(M,)``.
    shift : Tensor, optional
        Image shifts from :func:`pair_image_shifts` to reuse instead of
        searching them again (they stay valid while the atoms move by less
        than half the distance to the next image).

    Returns
    -------
    Tensor
        Displacements of shape ``(M, 3)``.
    """
    # formed in the geometry's dtype, returned in the model's (see
    # AtomicGraph.edge_vectors)
    vec = data.pos[b] - data.pos[a]
    if data.cell is None or vec.shape[0] == 0:
        return vec.to(data.model_dtype)
    if shift is None:
        shift = pair_image_shifts(data, a, b)
    cells = data.cell.to(vec.dtype)[data.batch[a]]                        # (M, 3, 3)
    return (vec + (shift[:, :, None] * cells).sum(1)).to(data.model_dtype)


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


def dihedral_cos_sin(data: AtomicGraph, idx: Tensor, vectors=None) -> tuple[Tensor, Tensor]:
    """Cosine and sine of the signed dihedral angle over each atom quadruple.

    The same plane-normal construction as :func:`dihedral_cos` (``phi = 0``
    at *cis*), plus the sine from the triple product of the two normals
    with the central bond, so terms with a phase (``cos(n phi - gamma)``,
    the AMBER torsion) can be formed from ``cos phi`` and ``sin phi`` by
    multiple-angle recurrences without ``arccos`` / ``atan2``.

    Parameters
    ----------
    data : AtomicGraph
        The batched graph.
    idx : Tensor
        Atom indices of shape ``(4, M)``.
    vectors : callable, optional
        ``vectors(a, b)`` returns the bond vectors; :func:`pair_vectors` by
        default.

    Returns
    -------
    tuple of Tensor
        ``(cos phi, sin phi)``, each of shape ``(M,)``; the sign of ``phi``
        follows the IUPAC convention (positive for a clockwise rotation of
        the far bond when looking along the central bond).
    """
    if vectors is None:
        def vectors(a, b):
            return pair_vectors(data, a, b)
    b1 = vectors(idx[0], idx[1])
    b2 = vectors(idx[1], idx[2])
    b3 = vectors(idx[2], idx[3])
    n1 = torch.cross(b1, b2, dim=1)
    n2 = torch.cross(b2, b3, dim=1)
    norm = torch.sqrt((n1 * n1).sum(1) * (n2 * n2).sum(1) + TINY)
    cos_phi = (n1 * n2).sum(1) / norm
    b2_len = torch.sqrt((b2 * b2).sum(1) + TINY)
    # (n1 x b2_hat) . n2 = -(n1 x n2) . b2_hat gives the IUPAC sign
    sin_phi = -(torch.cross(n1, n2, dim=1) * b2).sum(1) / (norm * b2_len)
    return cos_phi, sin_phi
