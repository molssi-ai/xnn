"""Multi-head QKV attention on graph edges.

A single, reusable attention primitive for graph-transformer potentials.
Unlike the global self-attention of a sequence transformer, the attention here
is *sparse*: it acts only along the edges of the neighbour graph, so its cost
is ``O(E)`` (linear in the number of atoms for a fixed cutoff) rather than
``O(N^2)``.

The module projects each node's scalar feature to per-head queries, keys and
values, gathers the query from the *centre* atom and the key/value from the
*neighbour* atom of each edge, and forms a scalar attention weight per edge and
head::

    a_e = act( sum_h( q[centre_e] . k[neighbour_e] ) ) * envelope_e

where ``envelope_e`` is a smooth radial cutoff that makes the attention vanish
as a neighbour leaves the cutoff sphere. The caller decides how to combine the
returned values and weights into messages (BAMBOO couples them with a radial
edge feature and an equivariant edge vector -- see
:class:`xnns.hybrid.models.bamboo.GETLayer`).
"""
from typing import Tuple

from torch import Tensor, nn


class EdgeMultiheadAttention(nn.Module):
    """Multi-head QKV attention evaluated on graph edges.

    Layer-normalises the node scalar features, projects them to per-head
    queries/keys/values with a single ``Linear(dim, 3 * dim)``, and returns,
    for every edge, the neighbour's value vectors and the (radial-cutoff
    weighted) scalar attention weight per head. This is the shared attention
    core of the BAMBOO graph-equivariant transformer, factored out so it can be
    reused by future graph-transformer models.

    Parameters
    ----------
    dim : int
        Width of the node scalar features (must be divisible by ``num_heads``).
    num_heads : int
        Number of attention heads.
    act_fn : torch.nn.Module, optional
        Non-linearity applied to the raw attention logits ``q . k``. Defaults
        to :class:`torch.nn.GELU` (the BAMBOO choice).

    Attributes
    ----------
    qkv_proj : torch.nn.Linear
        The fused query/key/value projection ``dim -> 3 * dim``.
    layer_norm : torch.nn.LayerNorm
        Pre-attention layer norm on the node features.
    dim_per_head : int
        ``dim // num_heads``.

    Raises
    ------
    ValueError
        If ``dim`` is not divisible by ``num_heads``.
    """

    def __init__(self, dim: int, num_heads: int, act_fn: nn.Module | None = None):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )
        self.dim = dim
        self.num_heads = num_heads
        self.dim_per_head = dim // num_heads
        self.layer_norm = nn.LayerNorm(dim)
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.attn_act = act_fn if act_fn is not None else nn.GELU()

    def forward(self, node_feat: Tensor, center_index: Tensor,
                neighbor_index: Tensor, envelope: Tensor
                ) -> Tuple[Tensor, Tensor]:
        """Compute per-edge neighbour values and attention weights.

        Parameters
        ----------
        node_feat : Tensor
            Node scalar features of shape ``(N, dim)``.
        center_index : Tensor
            For each edge, the index of the centre (receiving) atom whose query
            is used; shape ``(E,)``.
        neighbor_index : Tensor
            For each edge, the index of the neighbour (sending) atom whose key
            and value are used; shape ``(E,)``.
        envelope : Tensor
            Smooth radial cutoff weight per edge, shape ``(E,)``.

        Returns
        -------
        tuple of Tensor
            ``(value, attn)`` where ``value`` are the neighbour value vectors
            of shape ``(E, num_heads, dim_per_head)`` and ``attn`` the scalar
            attention weight per edge and head, shape ``(E, num_heads)``.
        """
        normed = self.layer_norm(node_feat)
        packed = self.qkv_proj(normed)
        # the fused projection emits, per head, a [q | k | v] block of
        # 3 * dim_per_head entries; unfold it and peel off the three roles
        q, k, v = packed.unflatten(
            -1, (self.num_heads, 3, self.dim_per_head)).unbind(dim=-2)

        score = (q[center_index] * k[neighbor_index]).sum(dim=-1)
        attn = self.attn_act(score) * envelope.unsqueeze(-1)
        return v[neighbor_index], attn
