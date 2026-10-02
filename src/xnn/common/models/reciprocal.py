"""Reciprocal-space sums through factorized phases (the Ewald fast paths).

An Ewald reciprocal sum needs the structure factors ``S(G) = sum_i v_i exp(i G .
r_i)`` over a set of reciprocal vectors ``G = m_1 b_1 + m_2 b_2 + m_3 b_3``
(integer ``m``, ``b_k`` the rows of the reciprocal cell). Written directly they
cost one sine and one cosine per atom and vector, ``(N, N_G)`` of each, which is
what dominates (and, kept for the backward pass, fills the memory of) the
large-system EEQ operator of D4 and the latent Ewald sum of LES. With ``theta_k
= b_k . r`` the phase factorizes,

    exp(i G . r) = exp(i (m_1 theta_1 + m_2 theta_2)) * exp(i m_3 theta_3),

so the vectors are grouped into columns of fixed ``(m_1, m_2)`` and the two
factors tabulated, ``B (C, N)`` over the columns and ``P (N, 2 M_3 + 1)`` over
``m_3``: the structure factors of every column are one complex matrix product
``B (v P)``, and the trigonometry drops from ``N N_G`` to ``N (C + 2 M_3 + 1)``
values. The phases are formed in float64, so float32 runs keep accurate
phases even at large ``|G . r|``.

:class:`PhaseColumns` holds the grouping; the callers keep their own vector
sets and weights, so the sums are the reference sums term by term.
"""
from __future__ import annotations

import math
from typing import Iterator, Tuple

import torch
from torch import Tensor

_TWO_PI = 2.0 * math.pi


def complex_dtype(dtype: torch.dtype) -> torch.dtype:
    """The complex type of a real working precision."""
    return torch.complex64 if dtype == torch.float32 else torch.complex128


def unit(phase: Tensor) -> Tensor:
    """``exp(i phase)`` (differentiable in ``phase``)."""
    return torch.complex(torch.cos(phase), torch.sin(phase))


def half_space(grid: Tensor) -> Tensor:
    """Mask of the integer triples whose leading nonzero entry is positive.

    One of every ``{m, -m}`` pair, never ``m = 0``; ``G`` and ``-G`` contribute
    equally to an Ewald sum of a real variable, so the half space carries
    weight two.
    """
    m1, m2, m3 = torch.round(grid).to(torch.long).unbind(-1)
    return (m1 > 0) | ((m1 == 0) & (m2 > 0)) | ((m1 == 0) & (m2 == 0) & (m3 > 0))


class PhaseColumns:
    """Integer reciprocal-lattice triples grouped into columns of fixed ``(m_1, m_2)``.

    Parameters
    ----------
    grid : Tensor
        The triples ``m``, ``(G, 3)``, integer-valued (any dtype). The caller
        chooses the set (a half space, a spherical shell); the order of the
        values passed to :meth:`scatter` follows this order.
    """

    def __init__(self, grid: Tensor):
        g = torch.round(grid).to(torch.long)
        self.m3_max = int(g[:, 2].abs().max()) if g.shape[0] else 0
        cols, col_of = torch.unique(g[:, :2], dim=0, return_inverse=True)
        self.cols = cols                        # (C, 2)
        self.col_of = col_of                    # (G,) column of each triple
        self.m3_of = g[:, 2] + self.m3_max      # (G,) position along m_3
        self.n_cols = int(cols.shape[0])
        self.n_m3 = 2 * self.m3_max + 1

    @staticmethod
    def phases(pos: Tensor, cell: Tensor) -> Tensor:
        """``theta (N, 3)``, ``theta_ik = b_k . r_i``, in float64 (differentiable).

        ``cell`` rows are the lattice vectors; ``b_k`` satisfy ``b_k . a_l =
        2 pi delta_kl``.
        """
        recip = _TWO_PI * torch.linalg.inv(cell.to(torch.float64)).t()
        return pos.to(torch.float64) @ recip.t()

    def column_table(self, theta: Tensor, c0: int, c1: int, cdtype: torch.dtype) -> Tensor:
        """``B[c, i] = exp(i (m_1 theta_1 + m_2 theta_2))`` for columns ``c0:c1``, ``(c, N)``."""
        cols = self.cols[c0:c1].to(torch.float64)
        phase = cols[:, 0:1] * theta[:, 0][None, :] + cols[:, 1:2] * theta[:, 1][None, :]
        return unit(phase).to(cdtype)

    def axis_table(self, theta: Tensor, cdtype: torch.dtype) -> Tensor:
        """``P[i, m] = exp(i m theta_3)`` for ``m = -M_3 .. M_3``, ``(N, 2 M_3 + 1)``."""
        m3 = torch.arange(-self.m3_max, self.m3_max + 1, device=theta.device, dtype=torch.float64)
        return unit(theta[:, 2:3] * m3[None, :]).to(cdtype)

    def scatter(self, values: Tensor, c0: int = 0, c1: int = -1) -> Tensor:
        """Per-triple ``values (G,)`` laid out as ``(c, 2 M_3 + 1)`` for columns ``c0:c1``
        (zero where a column has no triple); differentiable in ``values``."""
        c1 = self.n_cols if c1 < 0 else c1
        sel = (self.col_of >= c0) & (self.col_of < c1)
        out = torch.zeros((c1 - c0, self.n_m3), dtype=values.dtype, device=values.device)
        return out.index_put((self.col_of[sel] - c0, self.m3_of[sel]), values[sel])

    def blocks(self, n_atoms: int, budget: int) -> Iterator[Tuple[int, int]]:
        """Column ranges whose ``(c, N)`` table stays within ``budget`` entries."""
        step = max(1, int(budget) // max(int(n_atoms), 1))
        for c0 in range(0, self.n_cols, step):
            yield c0, min(c0 + step, self.n_cols)


def structure_factors(columns: PhaseColumns, table: Tensor, axis: Tensor, v: Tensor) -> Tensor:
    """``S[c, k, m] = sum_i B[c, i] v[i, k] P[i, m]`` for one column block, ``(c, K, 2 M_3 + 1)``."""
    n, k = v.shape
    x = (v.to(table.dtype)[:, :, None] * axis[:, None, :]).reshape(n, k * columns.n_m3)
    return (table @ x).view(table.shape[0], k, columns.n_m3)
