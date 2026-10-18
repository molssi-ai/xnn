"""Network blocks shared by several model families (PaiNN in ``gnn``, ET in ``trfnn``)."""
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from .ops import make_activation


class GatedEquivariantBlock(nn.Module):
    """Gated equivariant block (Weiler *et al.* 2018; Schuett *et al.* 2021, Fig. 3).

    Two linear maps of the vector features give ``W_1 v`` and ``W_2 v``; the
    norm of the first is stacked with the scalars and fed through a two-layer
    network whose output is split into new scalars and a gate that scales
    ``W_2 v``. Scalars pass through a nonlinearity, vectors are only scaled,
    so the block is equivariant.

    Parameters
    ----------
    n_in_s, n_in_v : int
        Input scalar and vector widths.
    n_out_s, n_out_v : int
        Output scalar and vector widths.
    n_hidden : int or None, optional
        Width of the hidden layer of the scalar network, by default
        ``n_in_s``.
    scalar_activation : bool, optional
        Whether the activation is applied to the output scalars (the hidden
        blocks of a stack); off for the last block. Default ``True``.
    n_norm : int or None, optional
        Width of the normed vector branch ``W_1 v``: ``n_out_v`` (default,
        the PaiNN reference code) or ``n_in_v`` (the TorchMD-Net output
        network).
    activation : str or torch.nn.Module, optional
        The nonlinearity of the scalar network, by default ``"silu"``.
    """

    def __init__(self, n_in_s: int, n_in_v: int, n_out_s: int, n_out_v: int,
                 n_hidden: Optional[int] = None, scalar_activation: bool = True,
                 n_norm: Optional[int] = None, activation="silu"):
        super().__init__()
        n_hidden = n_in_s if n_hidden is None else int(n_hidden)
        n_norm = n_out_v if n_norm is None else int(n_norm)
        self.n_out_s, self.n_out_v, self.n_norm = n_out_s, n_out_v, n_norm
        self.lin_v = nn.Linear(n_in_v, n_norm + n_out_v, bias=False)
        self.net = nn.Sequential(
            nn.Linear(n_in_s + n_norm, n_hidden), make_activation(activation),
            nn.Linear(n_hidden, n_out_s + n_out_v))
        self.scalar_activation = scalar_activation
        self.act = make_activation(activation)

    def forward(self, s: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        """Transform scalar and vector features.

        Parameters
        ----------
        s : Tensor
            Scalar features ``(N, n_in_s)``.
        v : Tensor
            Vector features ``(N, 3, n_in_v)``.

        Returns
        -------
        tuple of Tensor
            New scalars ``(N, n_out_s)`` and vectors ``(N, 3, n_out_v)``.
        """
        parts = torch.split(self.lin_v(v), [self.n_norm, self.n_out_v], dim=-1)
        v1, v2 = parts[0], parts[1]
        x = self.net(torch.cat([s, torch.linalg.norm(v1, dim=1)], dim=-1))
        s_out, gate = x[:, :self.n_out_s], x[:, self.n_out_s:]
        v_out = gate.unsqueeze(1) * v2
        if self.scalar_activation:
            s_out = self.act(s_out)
        return s_out, v_out
