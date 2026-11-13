"""Permutational symmetries of a molecule recovered from its training geometries (sGDML).

Chmiela et al., Nat. Commun. 9, 3887 (2018) and Comput. Phys. Commun. 240, 38
(2019), eq. 3. Every pair of geometries is matched by an assignment of the
atoms of one onto the other that aligns the magnitudes of the eigenvectors of
their distance matrices (elements may only map onto themselves). A matching is
kept when it brings the two distance matrices closer. The matchings are made
consistent by keeping those on a minimum spanning tree of the matching costs,
and the result is closed under composition into a group.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

#: the largest symmetry group accepted before the closure falls back to a subgroup
MAX_PERMUTATIONS = 100


def _distances(r: np.ndarray, cell: Optional[np.ndarray]) -> np.ndarray:
    """``(N, N)`` distance matrix (minimum images when ``cell``, rows as vectors, is given)."""
    d = r[:, None, :] - r[None, :, :]
    if cell is not None:
        d = d - np.rint(d @ np.linalg.inv(cell)) @ cell
    return np.sqrt((d * d).sum(-1))


def match_pairs(positions: np.ndarray, atomic_numbers: np.ndarray, cell: Optional[np.ndarray] = None):
    """Match every pair of geometries.

    Parameters
    ----------
    positions : ndarray
        ``(M, N, 3)`` geometries.
    atomic_numbers : ndarray
        ``(N,)`` elements.
    cell : ndarray, optional
        ``(3, 3)`` periodic cell.

    Returns
    -------
    tuple
        ``{(i, j): perm}`` for the pairs ``i < j`` whose matching improved the
        agreement of their distance matrices, and the ``(M, M)`` matching costs.
    """
    from scipy.optimize import linear_sum_assignment

    m, n = positions.shape[:2]
    z = np.asarray(atomic_numbers)
    other_element = (z[:, None] != z[None, :]).astype(float)
    adj, vec = [], []
    for k in range(m):
        a = _distances(positions[k], cell)
        w, v = np.linalg.eig(a)
        adj.append(a)
        vec.append(np.abs(v[:, np.argsort(w)[::-1]].real))
    perms = {}
    cost = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            c = -vec[i] @ vec[j].T
            c = c + other_element * np.max(np.abs(c))
            _, p = linear_sum_assignment(c)
            before = np.linalg.norm(adj[i] - adj[j])
            after = np.linalg.norm(adj[i][np.ix_(p, p)] - adj[j])
            cost[i, j] = min(after, before)
            if after < before and not np.isclose(before, after):
                perms[i, j] = p
    cost = cost + cost.T
    np.fill_diagonal(cost, np.inf)
    return perms, cost


def synchronize(perms: dict, cost: np.ndarray, n_atoms: int) -> np.ndarray:
    """The matchings on a minimum spanning tree of ``cost`` (and the identity), as unique rows."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree

    tree = minimum_spanning_tree(csr_matrix(cost))
    out = [np.arange(n_atoms)]
    for edge in zip(*tree.nonzero()):
        p = perms.get((int(edge[0]), int(edge[1])))
        if p is not None:
            out.append(p)
    return np.unique(np.vstack(out), axis=0)


def close_group(perms: np.ndarray, max_size: int = MAX_PERMUTATIONS) -> Optional[np.ndarray]:
    """Close ``perms`` under composition; ``None`` when the group outgrows ``max_size``."""
    perms = np.asarray(perms)
    added = True
    while added:
        added = False
        n = perms.shape[0]
        for i in range(n):
            for j in range(n):
                new = perms[i, perms[j, :]]
                if not (new == perms).all(axis=1).any():
                    added = True
                    perms = np.vstack((perms, new))
                    if perms.shape[0] == max_size:
                        return None
    return perms


def _cycles(perm) -> list:
    """The cycles of a permutation."""
    left = {i: int(perm[i]) for i in range(len(perm))}
    out = []
    while left:
        start = next(iter(left))
        cycle, k = [], start
        while k in left:
            cycle.append(k)
            k = left.pop(k)
        out.append(cycle)
    return out


def largest_cycle_subgroup(perms: np.ndarray) -> np.ndarray:
    """The permutations none of whose cycles overlaps a longer cycle of another permutation."""
    long = [c for p in perms for c in _cycles(p) if len(c) > 1]

    def shadowed(c):
        return any(len(c) < len(o) and not set(c).isdisjoint(o) for o in long)

    keep = [k for k, p in enumerate(perms) if not any(shadowed(c) for c in _cycles(p) if len(c) > 1)]
    return perms[keep]


def find_permutations(positions, atomic_numbers, cell=None, max_size: int = MAX_PERMUTATIONS) -> np.ndarray:
    """The permutational symmetry group of a molecule, from its geometries.

    Parameters
    ----------
    positions : array_like
        ``(M, N, 3)`` geometries (the training set).
    atomic_numbers : array_like
        ``(N,)`` elements.
    cell : array_like, optional
        ``(3, 3)`` periodic cell, rows as vectors.
    max_size : int, optional
        Largest group accepted; above it, the group of the permutations whose
        cycles are not shadowed by longer ones is used, and the identity alone
        when that group is too large as well. By default 100.

    Returns
    -------
    ndarray
        ``(Q, N)`` permutations, the identity among them.
    """
    r = np.asarray(positions, dtype=float)
    cell = None if cell is None else np.asarray(cell, dtype=float)
    perms, cost = match_pairs(r, np.asarray(atomic_numbers), cell)
    synced = synchronize(perms, cost, r.shape[1])
    group = close_group(synced, max_size)
    if group is None:
        group = close_group(largest_cycle_subgroup(synced), max_size)
    if group is None:
        group = np.arange(r.shape[1])[None]
    return group
