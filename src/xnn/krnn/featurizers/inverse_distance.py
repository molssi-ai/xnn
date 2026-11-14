"""Global descriptor of a molecule: the inverse distances of all atom pairs."""
from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor


class InverseDistances(torch.nn.Module):
    """Inverse interatomic distances ``1 / |r_a - r_b|`` of all pairs ``a > b``.

    The pairs are ordered row by row through the strictly lower triangle of the
    distance matrix: ``(1, 0), (2, 0), (2, 1), (3, 0), ...``. In a periodic cell
    every pair takes its minimum image.

    Parameters
    ----------
    n_atoms : int
        Number of atoms of the molecule.
    """

    def __init__(self, n_atoms: int):
        super().__init__()
        self.n_atoms = int(n_atoms)
        a, b = torch.tril_indices(self.n_atoms, self.n_atoms, -1)
        self.register_buffer("pairs", torch.stack([a, b]), persistent=False)
        self.register_buffer("_first", a, persistent=False)
        self.register_buffer("_second", b, persistent=False)
        self.dim = int(a.numel())

    def differences(self, pos: Tensor, cell: Optional[Tensor] = None) -> Tensor:
        """``(B, D, 3)`` vectors ``r_a - r_b`` of every pair (minimum images in a cell)."""
        d = pos.index_select(1, self._first) - pos.index_select(1, self._second)
        if cell is not None:
            frac = torch.einsum("bdi,bij->bdj", d, torch.linalg.inv(cell))
            d = d - torch.einsum("bdi,bij->bdj", torch.round(frac), cell)
        return d

    def forward(self, pos: Tensor, cell: Optional[Tensor] = None) -> Tensor:
        """``(B, D)`` descriptors of ``(B, N, 3)`` positions (and ``(B, 3, 3)`` cells, rows as vectors)."""
        return 1.0 / self.differences(pos, cell).norm(dim=-1)

    def pair_gradients(self, pos: Tensor, cell: Optional[Tensor] = None) -> Tensor:
        """``(B, D, 3)`` derivatives ``u = d (1 / r_ab) / d r_a`` (the derivative by ``r_b`` is ``-u``).

        They are the only nonzero blocks of the Jacobian: row ``d`` of it is ``u_d``
        on atom ``a_d`` and ``-u_d`` on atom ``b_d``.
        """
        d = self.differences(pos, cell)
        return -d / d.norm(dim=-1, keepdim=True) ** 3

    def jacobian(self, pos: Tensor, cell: Optional[Tensor] = None) -> Tensor:
        """``(B, D, 3N)`` derivatives of the descriptors by the Cartesian positions."""
        a, b = self.pairs
        g = self.pair_gradients(pos, cell)
        bsz, n = pos.shape[0], self.n_atoms
        jac = pos.new_zeros(bsz, self.dim, n, 3)
        rows = torch.arange(self.dim, device=pos.device)
        jac[:, rows, a] = g
        jac[:, rows, b] = -g
        return jac.reshape(bsz, self.dim, 3 * n)

    def permutation_index(self, perms: Sequence[Sequence[int]]) -> Tensor:
        """``(Q, D)`` index of the descriptor of each atom permutation.

        For a permutation ``p`` of the atoms, ``x[index[q]]`` is the descriptor
        with atom ``a`` replaced by atom ``p[a]``.
        """
        a, b = self.pairs.tolist()
        lookup = {}
        for k, (i, j) in enumerate(zip(a, b)):
            lookup[(i, j)] = k
            lookup[(j, i)] = k
        out = [[lookup[(int(p[i]), int(p[j]))] for i, j in zip(a, b)] for p in perms]
        return torch.tensor(out, dtype=torch.long).reshape(len(out), self.dim)

