"""ET, the equivariant transformer of TorchMD-Net (Thoelke & De Fabritiis, ICLR 2022).

An implementation of

* P. Thoelke, G. De Fabritiis, "TorchMD-NET: Equivariant Transformers for
  Neural Network based Molecular Potentials", ICLR 2022 (arXiv:2202.02541),
* with the cutoff fix of TorchMD-Net 2.0 (Pelaez *et al.*, *J. Chem. Theory
  Comput.* **20**, 4076 (2024), section 2.1.2),

written from the papers on the xnn abstractions
(:class:`~xnn.common.featurizers.ExpNormalSmearing`,
:class:`~xnn.common.featurizers.CosineCutoff`,
:class:`~xnn.common.models.blocks.GatedEquivariantBlock`,
:func:`~xnn.common.models.ops.scatter_sum`) and consistent with the
conventions of the authors' code (torchmd/torchmd-net: feature layout,
initialization, self-interaction, the ``vector_cutoff`` option), without
copying it. Parameter names follow the reference so that weights transplant.

Representation. Every atom carries ``F`` scalar features ``x_i`` and ``F``
Cartesian vector features ``v_i`` (zero at the start). The embedding layer
(eq 1 to 4) combines an intrinsic embedding of ``Z_i`` with a neighborhood
embedding, a distance-filtered sum of the neighbors' type embeddings::

    x_i = W^C [embed(Z_i), sum_j embed_nbh(Z_j) o (W^F e_rbf(d_ij) + b) phi(d_ij)] + b^C

with the exponential-normal basis ``e_rbf`` of eq 1 and the cosine cutoff
``phi`` of eq 2. ``T`` residual update layers follow (eq 5 to 11): a
layer-normed multi-head attention whose logits include a distance filter
``D^K`` and whose values are filtered by ``D^V``::

    A_ij = SiLU(sum_k Q_k(x_i) K_k(x_j) D^K_k(d_ij)) phi(d_ij)
    s^1_ij, s^2_ij, s^3_ij = split(V(x_j) o D^V(d_ij))
    y_i = O(sum_j A_ij s^3_ij),  split into q^1_i, q^2_i, q^3_i
    dx_i = q^1_i + q^2_i o <U_1 v_i, U_2 v_i>
    dv_i = sum_j (s^1_ij o v_j + s^2_ij o r_ij / |r_ij|) + q^3_i o U_3 v_i

The sums over ``j`` include ``j = i`` (the reference neighbor list carries
self pairs at ``d_ii = 0``), which the model adds as a per-atom term. A final
layer norm and two gated equivariant blocks (``F -> F/2 -> 1``) give the
per-atom energy, standardized by ``energy_scale`` / ``energy_shift`` and
shifted by the per-element reference ``atom_ref``.

Conventions worth knowing:

* ``r_ij = r_j - r_i`` points from the center atom to its neighbor, the
  negative of the xnn edge vector ``pos[dst] - pos[src]``; the sign is only
  visible in ``node_vectors``, not in the energy.
* ``vector_cutoff=True`` (the default) applies the cutoff to the whole value
  pathway, ``split(phi(d_ij) V(x_j) o D^V(d_ij))``, as TorchMD-Net 2.0 does,
  so that the energy is continuous at the cutoff. ``False`` is the original
  ICLR 2022 model, whose vector residuals are not weighted by the cutoff
  (the ``ET_old`` of the 2024 paper).
* The reference standardizes with the dataset mean per *structure* and the
  standard deviation per atom; xnn applies both per atom.
"""
# NOTE: no `from __future__ import annotations` here -- PEP 563 stringifies
# the annotations that TorchScript needs to resolve.
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from xnn.common.data import AtomicGraph
from xnn.common.featurizers import CosineCutoff, make_radial_basis
from xnn.common.models.blocks import GatedEquivariantBlock
from xnn.common.models.ops import make_activation, scatter_sum
from xnn.common.models.registry import register_model

from .base import TransformerPotential

_INFLUENCES = ("both", "keys", "values", "none")


def _xavier_zero_bias_(module: nn.Module) -> None:
    """Xavier-uniform weights and zero biases for every linear layer (the reference init)."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class _NeighborEmbedding(nn.Module):
    """The neighborhood embedding of eq 3 and 4.

    Parameters
    ----------
    n_features : int
        Feature width ``F``.
    n_rbf : int
        Number of radial basis functions.
    cutoff : float
        Cosine cutoff radius.
    n_elements : int
        Size of the type embedding table (indexed by ``Z``).
    """

    def __init__(self, n_features: int, n_rbf: int, cutoff: float, n_elements: int):
        super().__init__()
        self.embedding = nn.Embedding(n_elements, n_features)
        self.distance_proj = nn.Linear(n_rbf, n_features)
        self.combine = nn.Linear(2 * n_features, n_features)
        self.cutoff_fn = CosineCutoff(cutoff)
        _xavier_zero_bias_(self)

    def forward(self, atomic_numbers: Tensor, x: Tensor, src: Tensor, dst: Tensor,
                r: Tensor, rbf: Tensor) -> Tensor:
        """Combine the intrinsic embedding ``x`` with the filtered neighbor types."""
        w = self.distance_proj(rbf) * self.cutoff_fn(r).unsqueeze(-1)
        msg = w * self.embedding(atomic_numbers)[src]
        neighborhood = scatter_sum(msg, dst, x.shape[0])
        return self.combine(torch.cat([x, neighborhood], dim=-1))


class _AttentionLayer(nn.Module):
    """One update layer: the modified attention of eq 5 to 9 and the update of eq 10 and 11.

    Parameters
    ----------
    n_features : int
        Feature width ``F`` (divisible by ``num_heads``).
    n_rbf : int
        Number of radial basis functions.
    num_heads : int
        Attention heads.
    activation : str
        Nonlinearity of the distance filters ``D^K`` and ``D^V``.
    attn_activation : str
        Nonlinearity of the attention logits (SiLU in the paper).
    cutoff : float
        Cosine cutoff radius.
    distance_influence : str
        Where the distance filters enter: ``"both"``, ``"keys"``,
        ``"values"`` or ``"none"``.
    vector_cutoff : bool
        Weight the value pathway (instead of the attention weight alone) by
        the cutoff.
    """

    def __init__(self, n_features: int, n_rbf: int, num_heads: int, activation: str,
                 attn_activation: str, cutoff: float, distance_influence: str,
                 vector_cutoff: bool):
        super().__init__()
        self.n_features = n_features
        self.num_heads = num_heads
        self.head_dim = n_features // num_heads
        self.vector_cutoff = vector_cutoff
        self.layernorm = nn.LayerNorm(n_features)
        self.act = make_activation(activation)
        self.attn_act = make_activation(attn_activation)
        self.cutoff_fn = CosineCutoff(cutoff)
        self.q_proj = nn.Linear(n_features, n_features)
        self.k_proj = nn.Linear(n_features, n_features)
        self.v_proj = nn.Linear(n_features, 3 * n_features)
        self.o_proj = nn.Linear(n_features, 3 * n_features)
        self.vec_proj = nn.Linear(n_features, 3 * n_features, bias=False)
        self.dk_proj: Optional[nn.Linear] = None
        if distance_influence in ("keys", "both"):
            self.dk_proj = nn.Linear(n_rbf, n_features)
        self.dv_proj: Optional[nn.Linear] = None
        if distance_influence in ("values", "both"):
            self.dv_proj = nn.Linear(n_rbf, 3 * n_features)
        _xavier_zero_bias_(self)

    def forward(self, x: Tensor, vec: Tensor, src: Tensor, dst: Tensor, r: Tensor,
                rbf: Tensor, direction: Tensor, rbf_self: Tensor
                ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute the residuals ``dx``, ``dv`` and the attention weights.

        Parameters
        ----------
        x : Tensor
            Scalar features ``(N, F)``.
        vec : Tensor
            Vector features ``(N, 3, F)``.
        src, dst : Tensor
            Neighbor ``j`` and center ``i`` of every edge, ``(E,)`` each.
        r : Tensor
            Edge lengths ``(E,)``.
        rbf : Tensor
            Radial basis of the edge lengths ``(E, n_rbf)``.
        direction : Tensor
            Unit vectors ``r_ij / |r_ij|`` with ``r_ij = r_j - r_i``, ``(E, 3)``.
        rbf_self : Tensor
            The radial basis at zero distance, ``(1, n_rbf)`` (the
            self-interaction term).

        Returns
        -------
        tuple of Tensor
            ``dx`` of shape ``(N, F)``, ``dv`` of shape ``(N, 3, F)`` and the
            attention weights ``A_ij`` of shape ``(E, num_heads)``.
        """
        n, F, H, d = x.shape[0], self.n_features, self.num_heads, self.head_dim
        x = self.layernorm(x)
        q = self.q_proj(x).reshape(n, H, d)
        k = self.k_proj(x).reshape(n, H, d)
        v = self.v_proj(x).reshape(n, H, 3 * d)
        parts = torch.split(self.vec_proj(vec), F, dim=-1)
        vec1, vec2, vec3 = parts[0], parts[1], parts[2]
        vec_dot = (vec1 * vec2).sum(dim=1)                         # <U_1 v, U_2 v>
        vec_h = vec.reshape(n, 3, H, d)
        envelope = self.cutoff_fn(r)

        # eq 7 and 8: the distance-filtered dot product of queries and keys
        logits = q[dst] * k[src]
        if self.dk_proj is not None:
            logits = logits * self.act(self.dk_proj(rbf)).reshape(-1, H, d)
        attn = self.attn_act(logits.sum(dim=-1))                  # (E, H)
        values = v[src]
        if self.vector_cutoff:
            values = values * envelope[:, None, None]
        else:
            attn = attn * envelope[:, None]
        if self.dv_proj is not None:
            values = values * self.act(self.dv_proj(rbf)).reshape(-1, H, 3 * d)
        val = torch.split(values, d, dim=-1)                       # s^3, s^1, s^2
        msg_x = val[0] * attn.unsqueeze(-1)
        msg_vec = vec_h[src] * val[1].unsqueeze(1) + val[2].unsqueeze(1) * direction[:, :, None, None]
        dx = scatter_sum(msg_x, dst, n)
        dvec = scatter_sum(msg_vec, dst, n)

        # the atom's own term (j = i): zero distance, so the cutoff is one and
        # the directional part vanishes
        logits0 = q * k
        if self.dk_proj is not None:
            logits0 = logits0 * self.act(self.dk_proj(rbf_self)).reshape(1, H, d)
        attn0 = self.attn_act(logits0.sum(dim=-1))
        values0 = v
        if self.dv_proj is not None:
            values0 = values0 * self.act(self.dv_proj(rbf_self)).reshape(1, H, 3 * d)
        val0 = torch.split(values0, d, dim=-1)
        dx = dx + val0[0] * attn0.unsqueeze(-1)
        dvec = dvec + vec_h * val0[1].unsqueeze(1)

        # eq 10 and 11
        outputs = torch.split(self.o_proj(dx.reshape(n, F)), F, dim=-1)
        o1, o2, o3 = outputs[0], outputs[1], outputs[2]
        dx_out = vec_dot * o2 + o3
        dvec_out = vec3 * o1.unsqueeze(1) + dvec.reshape(n, 3, F)
        return dx_out, dvec_out, attn


class _EquivariantScalar(nn.Module):
    """The output network: two gated equivariant blocks ``F -> F/2 -> 1``.

    Parameters
    ----------
    n_features : int
        Input width ``F``.
    activation : str
        Nonlinearity of the blocks.
    """

    def __init__(self, n_features: int, activation: str):
        super().__init__()
        half = n_features // 2
        self.blocks = nn.ModuleList([
            GatedEquivariantBlock(n_features, n_features, half, half, n_hidden=n_features,
                                  scalar_activation=True, n_norm=n_features,
                                  activation=activation),
            GatedEquivariantBlock(half, half, 1, 1, n_hidden=half, scalar_activation=False,
                                  n_norm=half, activation=activation)])
        _xavier_zero_bias_(self)

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        """Map the representation to one scalar per atom, shape ``(N,)``."""
        for block in self.blocks:
            x, vec = block(x, vec)
        return x.squeeze(-1)


@register_model("et")
class EquivariantTransformer(TransformerPotential):
    """The TorchMD-Net equivariant transformer (ET).

    See the module docstring for the equations. The defaults are the MD17 /
    ANI-1 architecture of the paper (Table 4): ``F = 128``, six update layers,
    eight heads, 32 exponential-normal radial functions with a cosine cutoff
    at 5 Angstrom (1.34 M parameters); the QM9 model has ``F = 256``, eight
    layers and 64 radial functions.

    Parameters
    ----------
    n_features : int, optional
        Width ``F`` of the scalar and vector features, by default 128.
    n_interactions : int, optional
        Number of update layers, by default 6.
    n_rbf : int, optional
        Number of radial basis functions, by default 32.
    cutoff : float, optional
        Neighbor-list radius and cosine cutoff (Angstrom), by default 5.0.
    num_heads : int, optional
        Attention heads (``n_features`` must be divisible by it), by default 8.
    radial_basis : str, optional
        Any basis of :func:`~xnn.common.featurizers.make_radial_basis`:
        ``"expnorm"`` (default, eq 1), ``"gaussian"`` or ``"bessel"``.
    trainable_rbf : bool, optional
        Learn the parameters of the basis (the exponential-normal centers and
        widths, the Bessel frequencies), by default ``False``.
    activation : str, optional
        Nonlinearity of the distance filters and the output network, by
        default ``"silu"``.
    attn_activation : str, optional
        Nonlinearity of the attention logits (eq 8), by default ``"silu"``.
    neighbor_embedding : bool, optional
        Use the neighborhood embedding of eq 3 and 4 (the ablation of
        section 3.2 drops it), by default ``True``.
    distance_influence : str, optional
        Where the distance filters enter the attention: ``"both"`` (default),
        ``"keys"``, ``"values"`` or ``"none"``.
    vector_cutoff : bool, optional
        Apply the cutoff to the whole value pathway (TorchMD-Net 2.0, a
        continuous energy at the cutoff), by default ``True``; ``False`` is
        the original ICLR 2022 model.
    n_elements : int, optional
        Size of the embedding tables, indexed by ``Z``, by default 100.
    energy_shift : float, optional
        Additive per-atom energy standardization, by default 0.0.
    energy_scale : float, optional
        Multiplicative per-atom energy standardization, by default 1.0.
    species : list of int or None, optional
        Only used to interpret ``atomic_energies``; the model handles every
        element below ``n_elements``.
    atomic_energies : array-like or None, optional
        Per-species reference energies loaded into ``atom_ref``.

    Attributes
    ----------
    cutoff : float
        Neighbor-list cutoff radius.
    embedding : torch.nn.Embedding
        The intrinsic type embedding.
    rbf : torch.nn.Module
        The radial basis.
    neighbor_embedding : torch.nn.Module or None
        The neighborhood embedding of eq 3 and 4.
    layers : torch.nn.ModuleList
        The update layers.
    out_norm : torch.nn.LayerNorm
        The layer norm before the output network.
    readout : torch.nn.Module
        The gated equivariant output network.
    atom_ref : torch.nn.Embedding
        Learnable per-element energy reference, initialized to zero.
    """

    head_modules = ("readout", "energy_scale", "energy_shift", "atom_ref")

    def __init__(self, n_features: int = 128, n_interactions: int = 6, n_rbf: int = 32,
                 cutoff: float = 5.0, num_heads: int = 8, radial_basis: str = "expnorm",
                 trainable_rbf: bool = False, activation: str = "silu",
                 attn_activation: str = "silu", neighbor_embedding: bool = True,
                 distance_influence: str = "both", vector_cutoff: bool = True,
                 n_elements: int = 100, energy_shift: float = 0.0,
                 energy_scale: float = 1.0, species=None, atomic_energies=None):
        super().__init__()
        if n_features % num_heads != 0:
            raise ValueError(f"n_features ({n_features}) must be divisible by "
                             f"num_heads ({num_heads})")
        if distance_influence not in _INFLUENCES:
            raise ValueError(f"distance_influence must be one of {_INFLUENCES}, "
                             f"got {distance_influence!r}")
        self.cutoff = float(cutoff)
        if species is not None:
            self.species = [int(z) for z in species]
        self.n_features = n_features
        self.node_feature_dim = n_features
        self.n_interactions = n_interactions
        self.num_heads = num_heads
        self.distance_influence = distance_influence
        self.vector_cutoff = vector_cutoff
        self.embedding = nn.Embedding(n_elements, n_features)
        self.rbf = make_radial_basis(radial_basis, n_rbf, cutoff, trainable_rbf)
        self.neighbor_embedding: Optional[_NeighborEmbedding] = None
        if neighbor_embedding:
            self.neighbor_embedding = _NeighborEmbedding(n_features, n_rbf, cutoff, n_elements)
        self.layers = nn.ModuleList([
            _AttentionLayer(n_features, n_rbf, num_heads, activation, attn_activation,
                            cutoff, distance_influence, vector_cutoff)
            for _ in range(n_interactions)])
        self.out_norm = nn.LayerNorm(n_features)
        self.readout = _EquivariantScalar(n_features, activation)
        self._init_energy_terms(n_elements, energy_scale, energy_shift, species,
                                atomic_energies)

    def _edge_terms(self, edge_vec: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Edge lengths, the unit vectors ``r_ij / |r_ij|``, their basis and the basis at zero."""
        r = torch.linalg.norm(edge_vec, dim=-1)
        # the paper's r_ij = r_j - r_i is the negative of the xnn edge vector
        direction = -edge_vec / r.clamp(min=1e-12).unsqueeze(-1)
        rbf_self = self.rbf(torch.zeros(1, dtype=r.dtype, device=r.device))
        return r, direction, self.rbf(r), rbf_self

    @torch.jit.export
    def representation(self, atomic_numbers: Tensor, edge_index: Tensor,
                       edge_vec: Tensor) -> Tuple[Tensor, Tensor]:
        """TorchScript-compatible trunk: the scalar and vector features of every atom.

        Parameters
        ----------
        atomic_numbers : Tensor
            Per-atom atomic numbers ``(N,)``.
        edge_index : Tensor
            Edge index ``(2, E)``; row 0 the neighbor ``j``, row 1 the
            center ``i``.
        edge_vec : Tensor
            Edge vectors ``pos[i] - pos[j]`` of shape ``(E, 3)`` (with any
            periodic image shift applied).

        Returns
        -------
        tuple of Tensor
            The layer-normed scalars ``x`` of shape ``(N, F)`` and the
            vectors ``v`` of shape ``(N, 3, F)``.
        """
        src, dst = edge_index[0], edge_index[1]
        r, direction, rbf, rbf_self = self._edge_terms(edge_vec)
        x = self.embedding(atomic_numbers)
        if self.neighbor_embedding is not None:
            x = self.neighbor_embedding(atomic_numbers, x, src, dst, r, rbf)
        vec = torch.zeros((x.shape[0], 3, x.shape[1]), dtype=x.dtype, device=x.device)
        for layer in self.layers:
            dx, dvec, _ = layer(x, vec, src, dst, r, rbf, direction, rbf_self)
            x = x + dx
            vec = vec + dvec
        return self.out_norm(x), vec

    @torch.jit.ignore
    def attention_weights(self, data: AtomicGraph) -> List[Tensor]:
        """The attention weights ``A_ij`` of every update layer.

        Parameters
        ----------
        data : AtomicGraph
            The graph (the weights refer to its ``edge_index``; the
            self-interaction terms are not included).

        Returns
        -------
        list of Tensor
            One ``(E, num_heads)`` tensor per layer.
        """
        src, dst = data.edge_index[0], data.edge_index[1]
        r, direction, rbf, rbf_self = self._edge_terms(data.edge_vectors())
        x = self.embedding(data.atomic_numbers)
        if self.neighbor_embedding is not None:
            x = self.neighbor_embedding(data.atomic_numbers, x, src, dst, r, rbf)
        vec = torch.zeros((x.shape[0], 3, x.shape[1]), dtype=x.dtype, device=x.device)
        weights = []
        for layer in self.layers:
            dx, dvec, attn = layer(x, vec, src, dst, r, rbf, direction, rbf_self)
            x, vec = x + dx, vec + dvec
            weights.append(attn)
        return weights

    def _energy(self, x: Tensor, vec: Tensor, atomic_numbers: Tensor) -> Tensor:
        """Per-atom energies from the representation."""
        return (self.energy_scale * self.readout(x, vec) + self.energy_shift
                + self.atom_ref(atomic_numbers).squeeze(-1))

    @torch.jit.export
    def node_features_energy(self, atomic_numbers: Tensor, edge_index: Tensor,
                             edge_vec: Tensor) -> Tuple[Tensor, Tensor]:
        """TorchScript-compatible core: scalar features and per-atom energies.

        Parameters
        ----------
        atomic_numbers : Tensor
            Per-atom atomic numbers ``(N,)``.
        edge_index : Tensor
            Edge index ``(2, E)``.
        edge_vec : Tensor
            Edge vectors ``(E, 3)``.

        Returns
        -------
        tuple of Tensor
            The scalar features ``(N, F)`` and the per-atom energy ``(N,)``.
        """
        x, vec = self.representation(atomic_numbers, edge_index, edge_vec)
        return x, self._energy(x, vec, atomic_numbers)

    @torch.jit.export
    def node_energy(self, atomic_numbers: Tensor, edge_index: Tensor,
                    edge_vec: Tensor) -> Tensor:
        """Per-atom energy ``(N,)`` (the LAMMPS / TorchScript entry point)."""
        return self.node_features_energy(atomic_numbers, edge_index, edge_vec)[1]

    @torch.jit.ignore
    def forward(self, data: AtomicGraph) -> Dict[str, Tensor]:
        """Energies of a batched graph.

        Parameters
        ----------
        data : AtomicGraph
            Batched atomic graph.

        Returns
        -------
        dict of str to Tensor
            ``"node_energy"`` ``(N,)``, ``"energy"`` ``(B,)``,
            ``"node_features"`` (the layer-normed scalars, ``(N, F)``) and
            ``"node_vectors"`` (``(N, 3, F)``).
        """
        x, vec = self.representation(data.atomic_numbers, data.edge_index,
                                     data.edge_vectors())
        node_energy = self._energy(x, vec, data.atomic_numbers)
        return {"node_energy": node_energy,
                "energy": self.aggregate_energy(node_energy, data),
                "node_features": x, "node_vectors": vec}

    @classmethod
    def from_config(cls, cfg) -> "EquivariantTransformer":
        """Build an :class:`EquivariantTransformer` from a configuration object.

        Core fields: ``cfg.n_features``, ``cfg.n_interactions``, ``cfg.n_rbf``,
        ``cfg.cutoff``. Everything else is read from ``cfg.extra``
        (``num_heads``, ``radial_basis``, ``trainable_rbf``, ``activation``,
        ``attn_activation``, ``neighbor_embedding``, ``distance_influence``,
        ``vector_cutoff``, ``n_elements``, ``energy_shift``, ``energy_scale``,
        ``species``, ``atomic_energies``); the reference code's key spellings
        are translated by :mod:`xnn.common.config.translate`.

        Parameters
        ----------
        cfg : xnn.common.config.schema.ModelConfig
            The core model config.

        Returns
        -------
        EquivariantTransformer
            Instantiated model.
        """
        from xnn.common.config.coerce import coerce_per_species, coerce_species
        from xnn.common.config.options import model_options

        extra = model_options(cfg)
        species = (coerce_species(extra.get("species"))
                   if extra.get("species") is not None else None)
        return cls(
            n_features=cfg.n_features,
            n_interactions=cfg.n_interactions,
            n_rbf=cfg.n_rbf,
            cutoff=cfg.cutoff,
            num_heads=int(extra.get("num_heads", 8)),
            radial_basis=str(extra.get("radial_basis", "expnorm")),
            trainable_rbf=bool(extra.get("trainable_rbf", False)),
            activation=str(extra.get("activation", "silu")),
            attn_activation=str(extra.get("attn_activation", "silu")),
            neighbor_embedding=bool(extra.get("neighbor_embedding", True)),
            distance_influence=str(extra.get("distance_influence", "both")),
            vector_cutoff=bool(extra.get("vector_cutoff", True)),
            n_elements=int(extra.get("n_elements", 100)),
            energy_shift=float(extra.get("energy_shift", 0.0)),
            energy_scale=float(extra.get("energy_scale", 1.0)),
            species=species,
            atomic_energies=coerce_per_species(
                extra.get("atomic_energies"), species or [], "atomic_energies"),
        )
