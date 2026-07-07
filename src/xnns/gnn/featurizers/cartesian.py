"""Cartesian angular edge basis for the CACE potential (Cheng 2024).

Encodes an edge direction with the Cartesian monomials

    L_lxlylz(r_hat) = x^lx * y^ly * z^lz,   lx + ly + lz = l <= l_max

(paper eq. 2) instead of spherical harmonics -- the two bases span the same
space per total angular momentum ``l`` and are related by a fixed linear map,
so all symmetrization can be done with simple multinomial combination rules in
Cartesian coordinates (no Clebsch-Gordan contraction, no e3nn).

Faithful to the reference implementation
(https://github.com/BingqingCheng/cace, ``cace.modules.AngularComponent``):
the monomials are built by the same recursion ``L[lxlylz] = L[parent] *
component`` (autograd-safe -- no ``pow`` with zero exponents, which produces
NaN gradients for axis-aligned edges), grouped contiguously by total ``l`` in
ascending order.
"""
from __future__ import annotations

from math import factorial

import torch
from torch import Tensor, nn


def lxlylz_list(l_max: int) -> list[tuple[int, int, int]]:
    """Enumerate the Cartesian angular momentum triples up to ``l_max``.

    Parameters
    ----------
    l_max : int
        Maximum total angular momentum ``l = lx + ly + lz``.

    Returns
    -------
    list of tuple of int
        All ``(lx, ly, lz)`` triples, grouped contiguously by total ``l`` in
        ascending order (the CACE convention); ``(0, 0, 0)`` comes first. The
        length is ``(l_max+1)(l_max+2)(l_max+3)/6``.
    """
    out: list[tuple[int, int, int]] = []
    for l in range(l_max + 1):
        for lx in range(l + 1):
            for ly in range(l + 1 - lx):
                out.append((lx, ly, l - lx - ly))
    return out


def n_lxlylz(l_max: int) -> int:
    """Number of Cartesian monomials with total angular momentum ``<= l_max``.

    Parameters
    ----------
    l_max : int
        Maximum total angular momentum.

    Returns
    -------
    int
        ``(l_max+1)(l_max+2)(l_max+3)/6``, the width of the angular basis.
    """
    return (l_max + 1) * (l_max + 2) * (l_max + 3) // 6


def multinomial_coefficient(lxlylz) -> int:
    """Multinomial coefficient ``l! / (lx! ly! lz!)`` of one angular triple.

    This is the combinatorial prefactor ``C(l)`` of paper eq. 4, used when
    contracting pairs of Cartesian angular features into rotational
    invariants.

    Parameters
    ----------
    lxlylz : sequence of int
        The ``(lx, ly, lz)`` triple.

    Returns
    -------
    int
        ``(lx+ly+lz)! / (lx! ly! lz!)``.
    """
    lx, ly, lz = (int(v) for v in lxlylz)
    return factorial(lx + ly + lz) // (factorial(lx) * factorial(ly) * factorial(lz))


class CartesianAngularBasis(nn.Module):
    """Cartesian monomial angular basis ``x^lx y^ly z^lz`` (CACE, paper eq. 2).

    A drop-in Cartesian alternative to spherical harmonics for encoding edge
    directions: evaluates every monomial with ``lx + ly + lz <= l_max`` on the
    (normalized) edge vectors. Entries are grouped contiguously by total
    ``l`` in ascending order, matching :func:`lxlylz_list`.

    Parameters
    ----------
    l_max : int
        Maximum total angular momentum of the basis.

    Attributes
    ----------
    l_max : int
        The maximum total angular momentum.
    lxlylz : list of tuple of int
        The ``(lx, ly, lz)`` triple of every output entry, in order.
    l_of_entry : Tensor
        Long buffer of shape ``(n_angular,)`` giving the total ``l`` of each
        entry (used by the CACE blocks that share weights per ``l`` group).
    n_angular : int
        Output width, ``(l_max+1)(l_max+2)(l_max+3)/6``.
    """

    def __init__(self, l_max: int):
        super().__init__()
        self.l_max = l_max
        self.lxlylz = lxlylz_list(l_max)
        self.n_angular = len(self.lxlylz)
        index = {c: i for i, c in enumerate(self.lxlylz)}
        # recursion plan: entry i (> 0) is entry `parent[i]` times one vector
        # component `component[i]` (decrement the largest exponent, as upstream)
        parents: list[int] = []
        components: list[int] = []
        for c in self.lxlylz[1:]:
            k = max(range(3), key=lambda i: c[i])
            parent = list(c)
            parent[k] -= 1
            parents.append(index[tuple(parent)])
            components.append(k)
        self._parents = parents
        self._components = components
        self.register_buffer(
            "l_of_entry", torch.tensor([sum(c) for c in self.lxlylz]),
            persistent=False)

    def forward(self, unit_vec: Tensor) -> Tensor:
        """Evaluate the monomials on (unit) edge vectors.

        Parameters
        ----------
        unit_vec : Tensor
            Normalized edge vectors of shape ``(E, 3)``.

        Returns
        -------
        Tensor
            The angular basis of shape ``(E, n_angular)``; column ``i`` is the
            monomial of ``self.lxlylz[i]``.
        """
        feats = [torch.ones_like(unit_vec[:, 0])]
        for p, k in zip(self._parents, self._components):
            feats.append(feats[p] * unit_vec[:, k])
        return torch.stack(feats, dim=1)
