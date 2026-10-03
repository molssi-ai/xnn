"""Point-charge electrostatics shared by the charge-predicting models.

Per-atom Coulomb energies of partial charges, in eV for charges in units of
``e`` and distances in Angstrom, in the forms the models need:

* :func:`coulomb_direct`: the plain ``1/r`` sum over every pair of a
  molecular structure (no cutoff; the pairs come from :func:`all_pairs`);
* :func:`coulomb_dsf`: the damped shifted-force truncation of Fennell and
  Gezelter (*J. Chem. Phys.* 124, 234104, 2006) over a neighbor list, which
  gives a continuous energy and force at a finite cutoff and so serves
  periodic structures (images in the neighbor list) and large molecules;
* :func:`coulomb_ewald`: the Ewald sum of a periodic structure, the
  ``erfc``-screened real-space part over the neighbor list plus the
  reciprocal sum over the lattice vectors within a spherical cutoff
  (:func:`reciprocal_lattice`), with the self and neutralizing-background
  corrections;
* :func:`coulomb_pme`: the same with the reciprocal part on a charge mesh
  (smooth particle-mesh Ewald, :mod:`~xnn.common.models.pme`).

The Ewald parameters come from :func:`ewald_parameters`: for a real-space
cutoff ``r_c`` and a target relative accuracy ``eps`` the splitting parameter
is ``alpha = sqrt(-ln eps) / r_c`` (the screened pair term is ``eps`` of its
bare value at the cutoff) and the reciprocal cutoff ``k_c = 2 alpha sqrt(-ln
eps)`` (the Gaussian factor of the reciprocal sum is ``eps`` at ``k_c``), the
balance of Kolafa and Perram that the reference AIMNet2 calculator and
``nvalchemiops`` use as well; xnn fixes ``r_c`` to the model's neighbor-list
radius instead of choosing it per structure, so the graph the data pipeline
builds is the one the sum needs.

Every function returns the energy as a per-atom share (half of every pair
term on each of its atoms, the reciprocal energy as ``q_i phi_i / 2``) so
the callers' ``node_energy`` stays additive, and accumulates the pair and
reciprocal sums in float64 whatever the dtype of the charges. Used by
:class:`~xnn.gnn.models.aimnet2.AIMNet2`; the all-pairs enumeration also by
:class:`~xnn.hybrid.models.bamboo.BAMBOO`, the reciprocal lattice also by the
periodic EEQ charges of D4 (:mod:`~xnn.common.models.eeq`).
"""
import math
from typing import List, Tuple

import torch
from torch import Tensor

from .dispersion import BOHR, HARTREE
from .ops import cell_volume, scatter_sum
from .pme import pme_mesh, pme_reciprocal

#: Coulomb constant ``k_e e^2`` in eV Angstrom (one hartree times one bohr).
COULOMB_CONSTANT = HARTREE * BOHR
#: Largest ``(N, N_k)`` phase table of the reciprocal sum formed at once (float64:
#: 1.6 GB); longer lattice sums run in blocks of vectors. The table scales with
#: the atoms times the lattice vectors, so large cells are better served by PME.
RECIPROCAL_BUDGET = 10 ** 8


def all_pairs(batch: Tensor) -> Tuple[Tensor, Tensor]:
    """All ordered intra-structure atom pairs ``(i, j)``, ``i != j``.

    Parameters
    ----------
    batch : Tensor
        Structure index per atom ``(N,)``.

    Returns
    -------
    tuple of Tensor
        The center indices ``row`` and partner indices ``col`` of every
        ordered same-structure pair, ``(P,)`` each.
    """
    same = batch.unsqueeze(0) == batch.unsqueeze(1)
    same = same & ~torch.eye(batch.shape[0], dtype=torch.bool, device=batch.device)
    pairs = same.nonzero()
    return pairs[:, 0], pairs[:, 1]


def coulomb_direct(charges: Tensor, pos: Tensor, batch: Tensor,
                   k_e: float = COULOMB_CONSTANT) -> Tensor:
    """Per-atom share of the all-pairs Coulomb energy of molecular structures.

    ``E_i = (k_e / 2) sum_{j != i} q_i q_j / r_ij`` over the atoms of the same
    structure, summing to ``k_e sum_{i<j} q_i q_j / r_ij``. The pair terms are
    accumulated in float64 whatever their dtype.

    Parameters
    ----------
    charges : Tensor
        Partial charges ``(N,)``.
    pos : Tensor
        Positions ``(N, 3)``; the pair distances are formed in the charges'
        dtype.
    batch : Tensor
        Structure index per atom ``(N,)``.
    k_e : float, optional
        The Coulomb constant (an argument so the function compiles under
        TorchScript), by default :data:`COULOMB_CONSTANT`.

    Returns
    -------
    Tensor
        Per-atom energies ``(N,)`` in eV, float64.
    """
    row, col = all_pairs(batch)
    r = torch.linalg.norm((pos[row] - pos[col]).to(charges.dtype), dim=-1)
    pair = 0.5 * k_e * charges[row] * charges[col] / r
    return scatter_sum(pair.to(torch.float64), row, charges.shape[0])


def coulomb_dsf(charges: Tensor, edge_index: Tensor, r: Tensor, alpha: float,
                cutoff: float, k_e: float = COULOMB_CONSTANT) -> Tensor:
    """Per-atom share of the damped shifted-force Coulomb energy over a neighbor list.

    Each pair within ``cutoff`` contributes ``q_i q_j phi(r)`` with the
    force-shifted screened kernel

        ``phi(r) = erfc(alpha r) / r - erfc(alpha r_c) / r_c
        + (r - r_c) [erfc(alpha r_c) / r_c^2 + 2 alpha / sqrt(pi) exp(-alpha^2 r_c^2) / r_c]``

    (zero value and slope at ``r_c``), and every atom carries the self term
    ``-[erfc(alpha r_c) / (2 r_c) + alpha / sqrt(pi)] q_i^2``. The pair terms
    are accumulated in float64.

    Parameters
    ----------
    charges : Tensor
        Partial charges ``(N,)``.
    edge_index : Tensor
        Ordered pairs ``(2, E)`` as ``[src, dst]``; both directions of a pair
        are expected, as a neighbor list provides them.
    r : Tensor
        The pair distances ``(E,)``.
    alpha : float
        Damping parameter in 1/Angstrom.
    cutoff : float
        The real-space cutoff ``r_c`` in Angstrom; pairs beyond it are
        ignored, so the neighbor list may be longer.
    k_e : float, optional
        The Coulomb constant, by default :data:`COULOMB_CONSTANT`.

    Returns
    -------
    Tensor
        Per-atom energies ``(N,)`` in eV, float64.
    """
    src, dst = edge_index[0], edge_index[1]
    erfc_rc = math.erfc(alpha * cutoff)
    shift = erfc_rc / cutoff
    sqrt_pi = math.sqrt(math.pi)
    slope = erfc_rc / cutoff ** 2 + 2.0 * alpha / sqrt_pi \
        * math.exp(-(alpha * cutoff) ** 2) / cutoff
    phi = torch.erfc(alpha * r) / r - shift + (r - cutoff) * slope
    phi = torch.where(r < cutoff, phi, torch.zeros_like(phi))
    pair = 0.5 * k_e * charges[src] * charges[dst] * phi
    node = scatter_sum(pair.to(torch.float64), dst, charges.shape[0])
    self_coefficient = -(0.5 * shift + alpha / sqrt_pi)
    return node + k_e * self_coefficient * charges.to(torch.float64) ** 2


def ewald_parameters(cutoff: float, accuracy: float) -> Tuple[float, float]:
    """Splitting parameter and reciprocal cutoff for a real-space cutoff and an accuracy.

    ``alpha = sqrt(-ln eps) / r_c`` and ``k_c = 2 alpha sqrt(-ln eps)``: the
    real-space screening ``exp(-alpha^2 r_c^2)`` and the reciprocal Gaussian
    ``exp(-k_c^2 / 4 alpha^2)`` both equal ``eps`` at their cutoffs (the
    Kolafa-Perram balance for a given ``r_c``).

    Parameters
    ----------
    cutoff : float
        Real-space cutoff ``r_c`` in Angstrom.
    accuracy : float
        Target relative accuracy ``eps`` in ``(0, 1)``.

    Returns
    -------
    tuple of float
        ``(alpha, k_cutoff)`` in 1/Angstrom.
    """
    root = math.sqrt(-math.log(accuracy))
    alpha = root / cutoff
    return alpha, 2.0 * alpha * root


def reciprocal_lattice(cell: Tensor, k_cutoff: float) -> Tuple[Tensor, Tensor, Tensor]:
    """The reciprocal lattice vectors ``k != 0`` with ``|k| <= k_cutoff``.

    ``k = m_1 b_1 + m_2 b_2 + m_3 b_3`` over the integer triples ``m`` of
    the enclosing box ``|m_i| <= ceil(k_c |a_i| / 2 pi)``, both ``k`` and
    ``-k`` (the full space).

    Parameters
    ----------
    cell : Tensor
        Lattice vectors as rows ``(3, 3)``.
    k_cutoff : float
        Spherical cutoff in 1/Angstrom.

    Returns
    -------
    tuple of Tensor
        ``grid`` the integer triples ``(N_k, 3)`` (long), ``kvec`` the
        vectors ``(N_k, 3)`` and ``k2`` their squared norms ``(N_k,)`` (both
        differentiable in ``cell``).
    """
    device = cell.device
    recip = 2.0 * math.pi * torch.linalg.inv(cell).t()                  # rows: b_i
    lengths = torch.linalg.norm(cell.detach(), dim=1)
    bounds = torch.ceil(k_cutoff * lengths / (2.0 * math.pi)).to(torch.long)
    r1 = torch.arange(-int(bounds[0]), int(bounds[0]) + 1, device=device)
    r2 = torch.arange(-int(bounds[1]), int(bounds[1]) + 1, device=device)
    r3 = torch.arange(-int(bounds[2]), int(bounds[2]) + 1, device=device)
    grid = torch.cartesian_prod(r1, r2, r3).reshape(-1, 3)
    kvec = grid.to(cell.dtype) @ recip
    k2 = (kvec * kvec).sum(-1)
    keep = (k2 > 0) & (k2 <= k_cutoff * k_cutoff)
    return grid[keep], kvec[keep], k2[keep]


def ewald_real(charges: Tensor, edge_index: Tensor, r: Tensor, alpha: float,
               cutoff: float) -> Tensor:
    """Per-atom share of the screened real-space sum ``sum q_i q_j erfc(alpha r) / r``.

    Parameters
    ----------
    charges : Tensor
        Charges ``(N,)``.
    edge_index : Tensor
        Ordered pairs ``(2, E)`` as ``[src, dst]``, both directions present.
    r : Tensor
        Pair distances ``(E,)``.
    alpha : float
        Splitting parameter.
    cutoff : float
        Pairs at or beyond it are ignored.

    Returns
    -------
    Tensor
        ``(N,)`` in ``e^2 / Angstrom``, float64.
    """
    src, dst = edge_index[0], edge_index[1]
    phi = torch.erfc(alpha * r) / r
    phi = torch.where(r < cutoff, phi, torch.zeros_like(phi))
    pair = 0.5 * charges[src] * charges[dst] * phi
    return scatter_sum(pair.to(torch.float64), dst, charges.shape[0])


def ewald_corrections(charges: Tensor, alpha: float, volume: Tensor) -> Tensor:
    """Self and neutralizing-background terms per atom, float64.

    ``-alpha / sqrt(pi) q_i^2`` removes the interaction of each Gaussian with
    its own point charge, and ``-pi / (2 alpha^2 V) q_i Q`` (``Q`` the net
    charge, summing to ``-pi Q^2 / (2 alpha^2 V)``) the energy of the uniform
    background that neutralizes a charged cell.
    """
    q = charges.to(torch.float64)
    total = q.sum()
    v = volume.to(torch.float64)
    return -(alpha / math.sqrt(math.pi)) * q * q - (math.pi / (2.0 * alpha * alpha)) / v * q * total


def ewald_reciprocal(charges: Tensor, pos: Tensor, cell: Tensor, alpha: float,
                     k_cutoff: float, budget: int = RECIPROCAL_BUDGET) -> Tensor:
    """Per-atom reciprocal-space energies ``q_i phi_i / 2`` of one periodic structure.

    ``phi_i = (4 pi / V) sum_{k != 0} exp(-k^2 / 4 alpha^2) / k^2 [cos(k r_i) Re S(k)
    + sin(k r_i) Im S(k)]`` with the structure factor ``S(k) = sum_j q_j
    exp(i k r_j)``; the sum over the atoms is ``(2 pi / V) sum_k ... |S(k)|^2``.
    Phases and sums are formed in float64, in blocks of lattice vectors when
    the phase table exceeds :data:`RECIPROCAL_BUDGET`.

    Parameters
    ----------
    charges : Tensor
        Charges ``(N,)``.
    pos : Tensor
        Positions ``(N, 3)``.
    cell : Tensor
        Lattice vectors as rows ``(3, 3)``.
    alpha : float
        Splitting parameter.
    k_cutoff : float
        Spherical reciprocal cutoff.
    budget : int, optional
        Entries of the phase table formed at once, by default
        :data:`RECIPROCAL_BUDGET`.

    Returns
    -------
    Tensor
        ``(N,)`` in ``e^2 / Angstrom``, float64.
    """
    q = charges.to(torch.float64)
    p = pos.to(torch.float64)
    c = cell.to(torch.float64)
    _, kvec, k2 = reciprocal_lattice(c, k_cutoff)
    weight = 4.0 * math.pi / cell_volume(c) * torch.exp(-0.25 * k2 / (alpha * alpha)) / k2
    n_atoms, n_k = p.shape[0], kvec.shape[0]
    step = max(1, budget // max(n_atoms, 1))
    phi = torch.zeros_like(q)
    for k0 in range(0, n_k, step):
        k1 = min(k0 + step, n_k)
        phi = phi + _reciprocal_potential(p, q, kvec[k0:k1], weight[k0:k1])
    return 0.5 * q * phi


def _reciprocal_potential(pos: Tensor, q: Tensor, kvec: Tensor, weight: Tensor) -> Tensor:
    phase = pos @ kvec.t()                                                # (N, N_k)
    cos, sin = torch.cos(phase), torch.sin(phase)
    re = weight * (q @ cos)
    im = weight * (q @ sin)
    return cos @ re + sin @ im


def coulomb_ewald(charges: Tensor, pos: Tensor, cell: Tensor, edge_index: Tensor, r: Tensor,
                  alpha: float, cutoff: float, k_cutoff: float,
                  k_e: float = COULOMB_CONSTANT) -> Tensor:
    """Per-atom Ewald Coulomb energies of one periodic structure, in eV.

    The real-space sum over the neighbor list within ``cutoff``, the
    reciprocal sum within ``k_cutoff`` and the self and background
    corrections (:func:`ewald_real`, :func:`ewald_reciprocal`,
    :func:`ewald_corrections`), times the Coulomb constant.

    Parameters
    ----------
    charges : Tensor
        Charges ``(N,)``.
    pos : Tensor
        Positions ``(N, 3)``.
    cell : Tensor
        Lattice vectors as rows ``(3, 3)``.
    edge_index : Tensor
        Neighbor pairs ``(2, E)`` of this structure (both directions, with
        the periodic images the neighbor list provides).
    r : Tensor
        Their distances ``(E,)``.
    alpha, cutoff, k_cutoff : float
        See :func:`ewald_parameters`.
    k_e : float, optional
        The Coulomb constant, by default :data:`COULOMB_CONSTANT`.

    Returns
    -------
    Tensor
        ``(N,)`` in eV, float64.
    """
    node = (ewald_real(charges, edge_index, r, alpha, cutoff)
            + ewald_reciprocal(charges, pos, cell, alpha, k_cutoff)
            + ewald_corrections(charges, alpha, cell_volume(cell)))
    return k_e * node


def coulomb_pme(charges: Tensor, pos: Tensor, cell: Tensor, edge_index: Tensor, r: Tensor,
                alpha: float, cutoff: float, accuracy: float, order: int,
                k_e: float = COULOMB_CONSTANT) -> Tensor:
    """Per-atom particle-mesh Ewald Coulomb energies of one periodic structure, in eV.

    As :func:`coulomb_ewald` with the reciprocal sum on a B-spline charge
    mesh sized for ``accuracy`` (:func:`~xnn.common.models.pme.pme_mesh`,
    :func:`~xnn.common.models.pme.pme_reciprocal`), in float64.

    Parameters
    ----------
    charges, pos, cell, edge_index, r, alpha, cutoff
        As in :func:`coulomb_ewald`.
    accuracy : float
        Target relative accuracy of the mesh.
    order : int
        B-spline order (4: cubic).
    k_e : float, optional
        The Coulomb constant, by default :data:`COULOMB_CONSTANT`.

    Returns
    -------
    Tensor
        ``(N,)`` in eV, float64.
    """
    q = charges.to(torch.float64)
    c = cell.to(torch.float64)
    mesh = pme_mesh(c, alpha, accuracy)
    node = (ewald_real(charges, edge_index, r, alpha, cutoff)
            + pme_reciprocal(q, pos.to(torch.float64), c, alpha, mesh, order)
            + ewald_corrections(charges, alpha, cell_volume(c)))
    return k_e * node


def ewald_mesh_sizes(cell: Tensor, alpha: float, accuracy: float) -> List[int]:
    """The PME mesh :func:`coulomb_pme` uses for a cell (for reporting and tests)."""
    return pme_mesh(cell.to(torch.float64), alpha, accuracy)
