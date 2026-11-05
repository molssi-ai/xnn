"""Edge-augmented graph transformer over the bonds of a molecule.

The edge-augmented graph transformer (EGT; Hussain, Zaki & Subramanian, KDD 2022,
https://doi.org/10.1145/3534678.3539296) lets edge channels steer and gate the
attention between nodes and updates the edges from the attention logits. Here the
attention is local, between bonded atoms only, as in ByteFF (Zheng et al., Chem.
Sci. 16, 2730 (2025)), where the network reads the two-dimensional molecular graph
and its outputs parameterize a classical force field.

:class:`MolecularGraphTransformer` embeds the integer atom and bond features of
:mod:`xnn.gnn.featurizers.molgraph`, averages the embeddings over classes of
equivalent atoms and bonds, runs the attention layers and returns per-atom and
per-directed-edge hidden states (the two directions of a bond averaged).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

from xnn.common.models.blocks import MLP
from xnn.common.models.ops import make_activation, scatter_sum

#: Embedding table sizes of the atom features: elements (atomic number - 1),
#: neighbors, formal charges (-2..2), ring bonds, smallest ring size.
N_ELEMENTS, MAX_DEGREE, MAX_FORMAL_CHARGE, MAX_RING = 86, 6, 2, 8
#: Bond feature table sizes: in ring (0/1), bond order (single, double, triple, aromatic).
N_BOND_RING, N_BOND_ORDER = 2, 4


def class_mean(x: Tensor, cls: Tensor) -> Tensor:
    """Each row of ``x`` replaced by the mean over the rows of its class ``cls``
    (class labels index rows, the first member of each class)."""
    n = x.shape[0]
    total = scatter_sum(x, cls, n)
    count = scatter_sum(torch.ones_like(cls, dtype=x.dtype), cls, n)
    return (total / count.clamp(min=1.0).unsqueeze(-1)).index_select(0, cls)


def _mlp(n_in: int, hidden: int, n_out: int, layers: int, activation, activate_last: bool = False) -> MLP:
    return MLP([n_in] + [hidden] * (layers - 1) + [n_out], activation, activate_last=activate_last)


class GraphEmbedding(nn.Module):
    """Embeddings of the atom and bond features, mixed by MLPs and averaged over
    equivalent atoms and bonds.

    Parameters
    ----------
    atom_dim : int
        Width of the element embedding.
    feature_dim : int
        Width of every other feature embedding.
    node_mlp, edge_mlp : tuple of int
        ``(hidden, out, layers)`` of the node and edge MLPs (activated after the
        last layer too).
    activation : str
        Activation of the MLPs.
    """

    def __init__(self, atom_dim: int = 32, feature_dim: int = 8, node_mlp=(64, 256, 2), edge_mlp=(64, 256, 2),
                 activation: str = "gelu"):
        super().__init__()
        self.element = nn.Embedding(N_ELEMENTS, atom_dim)
        self.degree = nn.Embedding(MAX_DEGREE + 1, feature_dim)
        self.formal_charge = nn.Embedding(2 * MAX_FORMAL_CHARGE + 1, feature_dim)
        self.ring_degree = nn.Embedding(MAX_DEGREE + 1, feature_dim)
        self.min_ring = nn.Embedding(MAX_RING + 1, feature_dim)
        self.bond_ring = nn.Embedding(N_BOND_RING, feature_dim)
        self.bond_order = nn.Embedding(N_BOND_ORDER, feature_dim)
        self.node_mlp = _mlp(atom_dim + 4 * feature_dim, node_mlp[0], node_mlp[1], node_mlp[2], activation, True)
        self.edge_mlp = _mlp(2 * feature_dim, edge_mlp[0], edge_mlp[1], edge_mlp[2], activation, True)
        self.node_dim, self.edge_dim = int(node_mlp[1]), int(edge_mlp[1])

    def forward(self, node_features: Tensor, edge_features: Tensor, atom_class: Tensor,
                edge_class: Tensor) -> tuple[Tensor, Tensor]:
        """Node ``(n, node_dim)`` and directed-edge ``(2 m, edge_dim)`` embeddings from the features
        ``(n, 5)`` (element index, neighbors, formal charge, ring bonds, smallest ring) and ``(2 m, 2)``
        (in ring, bond order)."""
        f = node_features
        charge = torch.clamp(f[:, 2] + MAX_FORMAL_CHARGE, 0, 2 * MAX_FORMAL_CHARGE)
        x = torch.cat([self.element(f[:, 0]), self.degree(f[:, 1]), self.formal_charge(charge),
                       self.ring_degree(f[:, 3]), self.min_ring(f[:, 4])], -1)
        e = torch.cat([self.bond_ring(edge_features[:, 0]), self.bond_order(edge_features[:, 1])], -1)
        return (class_mean(self.node_mlp(x), atom_class), class_mean(self.edge_mlp(e), edge_class))


class EGTLayer(nn.Module):
    """One local EGT layer: edge-biased, edge-gated multi-head attention between
    bonded atoms, pre-norm residual updates of nodes and edges.

    Parameters
    ----------
    dim : int
        Width of the node and edge channels.
    heads : int
        Attention heads.
    head_dim : int
        Width of each head's queries, keys and values.
    ffn : tuple of int
        ``(hidden, layers)`` of the node and edge feed-forward MLPs.
    activation : str
        Activation of the feed-forward MLPs.
    scale_by_degree : bool
        Multiply the attention weights by the receiving atom's number of bonds.
    """

    def __init__(self, dim: int, heads: int = 16, head_dim: int = 16, ffn=(256, 2), activation: str = "gelu",
                 scale_by_degree: bool = False):
        super().__init__()
        self.heads, self.head_dim = int(heads), int(head_dim)
        self.scale_by_degree = bool(scale_by_degree)
        width = self.heads * self.head_dim
        self.query, self.key, self.value = (nn.Linear(dim, width) for _ in range(3))
        self.proj = nn.Linear(width, dim)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.gate = nn.Linear(dim, self.heads)
        self.edge_bias = nn.Linear(dim, self.heads)
        self.edge_proj = nn.Linear(self.heads, dim)
        self.edge_norm1, self.edge_norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn = _mlp(dim, ffn[0], dim, ffn[1], activation)
        self.edge_ffn = _mlp(dim, ffn[0], dim, ffn[1], activation)

    def forward(self, x: Tensor, e: Tensor, src: Tensor, dst: Tensor) -> tuple[Tensor, Tensor]:
        """Updated nodes ``(n, dim)`` and edges ``(E, dim)``; edge ``k`` carries attention from atom
        ``src[k]`` to atom ``dst[k]``."""
        n, H, C = x.shape[0], self.heads, self.head_dim
        h = self.norm1(x)
        q = self.query(h).view(n, H, C)
        k = self.key(h).view(n, H, C)
        v = self.value(h).view(n, H, C)
        g = self.edge_norm1(e)
        gate = torch.sigmoid(self.gate(g))
        logits = (q.index_select(0, dst) * k.index_select(0, src)).sum(-1) / math.sqrt(C) + self.edge_bias(g)
        peak = torch.full((n, H), float("-inf"), dtype=logits.dtype, device=logits.device)
        peak = peak.scatter_reduce(0, dst.unsqueeze(-1).expand(-1, H), logits.detach(), "amax")
        w = torch.exp(logits - peak.index_select(0, dst))
        w = w / scatter_sum(w, dst, n).index_select(0, dst)
        if self.scale_by_degree:
            deg = scatter_sum(torch.ones_like(dst, dtype=w.dtype), dst, n)
            w = w * deg.index_select(0, dst).unsqueeze(-1)
        msg = v.index_select(0, src) * (gate * w).unsqueeze(-1)
        x = self.proj(scatter_sum(msg, dst, n).reshape(n, H * C)) + x
        x = self.ffn(self.norm2(x)) + x
        e = e + self.edge_proj(logits)
        e = self.edge_ffn(self.edge_norm2(e)) + e
        return x, e


class MolecularGraphTransformer(nn.Module):
    """Atom and bond hidden states of molecules from their two-dimensional graphs.

    Parameters
    ----------
    atom_dim, feature_dim : int
        Embedding widths (see :class:`GraphEmbedding`).
    node_mlp, edge_mlp : tuple of int
        ``(hidden, out, layers)`` of the embedding MLPs.
    hidden : int
        Width of the attention layers.
    layers : int
        Number of attention layers.
    heads, head_dim : int
        Attention heads and their width.
    ffn : tuple of int
        ``(hidden, layers)`` of the feed-forward MLPs.
    out_dim : int
        Width of the atom hidden states returned.
    jumping_knowledge : str or None
        ``"cat"``: the atom states are a linear map of every layer's activated output,
        concatenated; ``"last"``: of the last one; ``None``: the last layer's output
        as is (``hidden`` must then equal ``out_dim``).
    activation : str
        Activation of every MLP and between the layers.
    scale_by_degree : bool
        See :class:`EGTLayer`.

    Attributes
    ----------
    node_dim, edge_dim : int
        Widths of the returned atom and edge states.
    """

    def __init__(self, atom_dim: int = 32, feature_dim: int = 8, node_mlp=(64, 256, 2), edge_mlp=(64, 256, 2),
                 hidden: int = 256, layers: int = 4, heads: int = 16, head_dim: int = 16, ffn=(256, 2),
                 out_dim: int = 256, jumping_knowledge: Optional[str] = "cat", activation: str = "gelu",
                 scale_by_degree: bool = False):
        super().__init__()
        if jumping_knowledge not in ("cat", "last", None):
            raise ValueError(f"jumping_knowledge must be 'cat', 'last' or None, got {jumping_knowledge!r}")
        self.embedding = GraphEmbedding(atom_dim, feature_dim, node_mlp, edge_mlp, activation)
        node_in, edge_in = self.embedding.node_dim, self.embedding.edge_dim
        if hidden != node_in:
            raise ValueError("the attention width must equal the node embedding width "
                             f"({hidden} != {node_in})")
        if jumping_knowledge is None and out_dim != hidden:
            raise ValueError("without jumping knowledge out_dim must equal hidden")
        self.jumping_knowledge = jumping_knowledge
        self.edge_in = nn.Linear(edge_in, hidden)
        self.layers = nn.ModuleList(EGTLayer(hidden, heads, head_dim, ffn, activation, scale_by_degree)
                                    for _ in range(int(layers)))
        self.act = make_activation(activation)
        self.edge_out = nn.Linear(hidden, edge_in)
        self.out = None
        if jumping_knowledge is not None:
            width = hidden * int(layers) if jumping_knowledge == "cat" else hidden
            self.out = nn.Linear(width, out_dim)
        self.node_dim, self.edge_dim = int(out_dim), int(edge_in)

    def forward(self, node_features: Tensor, edge_features: Tensor, edge_index: Tensor, atom_class: Tensor,
                edge_class: Tensor) -> tuple[Tensor, Tensor]:
        """Atom states ``(n, node_dim)`` and directed-edge states ``(2 m, edge_dim)`` (both directions
        of a bond equal).

        Parameters
        ----------
        node_features : Tensor
            ``(n, 5)`` integer atom features.
        edge_features : Tensor
            ``(2 m, 2)`` integer bond features, one row per directed edge.
        edge_index : Tensor
            ``(2, 2 m)`` source and target atoms; edges ``2 k`` and ``2 k + 1`` are
            the two directions of bond ``k``.
        atom_class, edge_class : Tensor
            Equivalence classes ``(n,)`` and ``(2 m,)`` (the first member's index).

        Returns
        -------
        tuple of Tensor
            The atom and edge states.
        """
        x, e = self.embedding(node_features, edge_features, atom_class, edge_class)
        e = self.edge_in(e)
        src, dst = edge_index[0], edge_index[1]
        outs = []
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            x, e = layer(x, e, src, dst)
            if i < last or self.jumping_knowledge is not None:
                x = self.act(x)
            outs.append(x)
        if self.jumping_knowledge == "cat":
            x = self.out(torch.cat(outs, -1))
        elif self.jumping_knowledge == "last":
            x = self.out(x)
        e = self.edge_out(e)
        m = e.shape[0] // 2
        e = (0.5 * (e[0::2] + e[1::2])).unsqueeze(1).expand(m, 2, e.shape[1]).reshape(2 * m, e.shape[1])
        return x, e


__all__ = ["MolecularGraphTransformer", "EGTLayer", "GraphEmbedding", "class_mean"]
