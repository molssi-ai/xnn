"""Allegro (Musaelian et al. 2023): strictly local equivariant potential.

Allegro keeps features on *edges* and never aggregates across neighbours, so it
has no message passing -- every edge energy depends only on the local
environment defined by the cutoff. Here each edge carries equivariant features
updated by per-edge tensor products with its own spherical harmonics, gated by a
scalar latent MLP; a final readout gives a per-edge energy assigned to the
centre atom.

Fidelity note: this captures Allegro's defining property (locality, no message
passing, latent-MLP-driven equivariant edge updates). The original adds a
two-body bootstrap and a specific normalization scheme; the data/forces/training
/deploy pipeline is unchanged.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from e3nn import o3
from e3nn.nn import FullyConnectedNet

from xnns.common.data import AtomicGraph
from xnns.common.models.ops import scatter_sum
from xnns.common.models.registry import register_model
from .base import EquivariantGNN
from .blocks import hidden_irreps


class _AllegroLayer(nn.Module):
    """Per-edge equivariant update (no neighbour aggregation -> strictly local).

    Updates the equivariant features carried on a single edge via a tensor
    product with that edge's spherical harmonics, weighted by an MLP driven by
    the edge's scalar latent. The invariants (norms) of the updated equivariant
    features feed a residual update of the latent. Because everything is
    computed per edge with no cross-neighbour aggregation, the layer preserves
    Allegro's strict locality.

    Parameters
    ----------
    irreps_edge : o3.Irreps
        Irreps of the input equivariant edge features.
    irreps_sh : o3.Irreps
        Irreps of the spherical-harmonic edge attributes.
    irreps_out : o3.Irreps
        Desired irreps of the output equivariant edge features.
    latent_dim : int
        Width of the scalar latent that drives the tensor-product weights and
        is updated residually.

    Attributes
    ----------
    irreps_mid : o3.Irreps
        Irreps of the tensor-product output.
    tp : o3.TensorProduct
        Path-restricted (``uvu``) tensor product edge (x) sh with external,
        non-shared weights.
    weight_mlp : e3nn.nn.FullyConnectedNet
        MLP mapping the latent to the tensor-product path weights.
    linear : o3.Linear
        Equivariant linear projecting the tensor-product output to
        ``irreps_out``.
    norm : o3.Norm
        Computes the invariant norms of the equivariant output channels.
    latent_update : e3nn.nn.FullyConnectedNet
        MLP producing the residual latent update from the concatenated latent
        and invariants.
    irreps_out : o3.Irreps
        The output irreps of the layer.
    """

    def __init__(self, irreps_edge: o3.Irreps, irreps_sh: o3.Irreps,
                 irreps_out: o3.Irreps, latent_dim: int):
        super().__init__()
        irreps_mid, instr = [], []
        for i, (mul, ir_in) in enumerate(irreps_edge):
            for j, (_, ir_sh) in enumerate(irreps_sh):
                for ir_out in ir_in * ir_sh:
                    if ir_out in [ir for _, ir in irreps_out]:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instr.append((i, j, k, "uvu", True))
        self.irreps_mid = o3.Irreps(irreps_mid)
        self.tp = o3.TensorProduct(irreps_edge, irreps_sh, self.irreps_mid, instr,
                                   shared_weights=False, internal_weights=False)
        self.weight_mlp = FullyConnectedNet(
            [latent_dim, latent_dim, self.tp.weight_numel], F.silu)
        self.linear = o3.Linear(self.irreps_mid, irreps_out)
        # latent update from invariants (norms of equivariant channels)
        self.norm = o3.Norm(irreps_out)
        self.latent_update = FullyConnectedNet(
            [latent_dim + self.norm.irreps_out.dim, latent_dim, latent_dim], F.silu)
        self.irreps_out = irreps_out

    def forward(self, edge_feat: Tensor, edge_sh: Tensor, latent: Tensor):
        """Apply one strictly-local per-edge equivariant update.

        Parameters
        ----------
        edge_feat : Tensor
            Equivariant edge features of shape ``(E, irreps_edge.dim)``.
        edge_sh : Tensor
            Spherical-harmonic edge attributes of shape ``(E, irreps_sh.dim)``.
        latent : Tensor
            Scalar edge latent of shape ``(E, latent_dim)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            The updated equivariant edge features of shape
            ``(E, irreps_out.dim)`` and the residually updated latent of shape
            ``(E, latent_dim)``.
        """
        w = self.weight_mlp(latent)
        v = self.linear(self.tp(edge_feat, edge_sh, w))
        inv = self.norm(v)
        latent = latent + self.latent_update(torch.cat([latent, inv], dim=-1))
        return v, latent


@register_model("allegro")
class Allegro(EquivariantGNN):
    """Allegro (Musaelian et al. 2023) strictly local equivariant potential.

    Built on the shared :class:`~xnns.gnn.models.base.EquivariantGNN`, Allegro
    keeps features on *edges* and never aggregates across neighbours, so it has
    no message passing: every edge energy depends only on the local environment
    within the cutoff. A two-body MLP initializes a scalar latent per edge from
    the pair of species and the radial embedding, the spherical harmonics are
    embedded into equivariant edge features, and ``n_layers``
    :class:`_AllegroLayer` updates refine both. A readout maps the final latent
    to a per-edge energy, which is summed onto the centre atom (plus the
    per-element reference energy).

    Parameters
    ----------
    species : list[int]
        Atomic numbers of the supported elements, in channel order.
    cutoff : float, optional
        Neighbour cutoff radius. Default is 6.0.
    l_max : int, optional
        Maximum spherical-harmonic degree. Default is 2.
    n_rbf : int, optional
        Number of radial basis functions in the edge featurizer. Default is 8.
    n_layers : int, optional
        Number of :class:`_AllegroLayer` updates. Default is 2.
    mul : int, optional
        Channel multiplicity of the hidden equivariant edge-feature irreps.
        Default is 32.
    latent_dim : int, optional
        Width of the per-edge scalar latent. Default is 64.

    Attributes
    ----------
    two_body : e3nn.nn.FullyConnectedNet
        MLP mapping ``[species_i, species_j, radial]`` to the initial latent.
    embed_edge : o3.Linear
        Equivariant linear embedding the spherical harmonics into the initial
        equivariant edge features.
    layers : torch.nn.ModuleList
        The stack of :class:`_AllegroLayer` per-edge update layers.
    readout : e3nn.nn.FullyConnectedNet
        Invariant MLP mapping the final latent to a per-edge energy.
    """

    def __init__(self, species: list[int], cutoff: float = 6.0, l_max: int = 2,
                 n_rbf: int = 8, n_layers: int = 2, mul: int = 32,
                 latent_dim: int = 64):
        super().__init__(species, cutoff, l_max, n_rbf)
        irreps_hidden = hidden_irreps(mul, l_max)
        n_species = len(species)

        # two-body latent: [species_i, species_j, radial] -> latent scalars
        self.two_body = FullyConnectedNet(
            [2 * n_species + self.edge_feat.output_dim, latent_dim, latent_dim], F.silu)
        # initial equivariant edge features: embed sh into mul channels
        self.embed_edge = o3.Linear(self.irreps_sh, irreps_hidden)

        self.layers = nn.ModuleList()
        irreps_edge = irreps_hidden
        for _ in range(n_layers):
            layer = _AllegroLayer(irreps_edge, self.irreps_sh, irreps_hidden, latent_dim)
            self.layers.append(layer)
            irreps_edge = layer.irreps_out
        self.readout = FullyConnectedNet([latent_dim, latent_dim // 2, 1], F.silu)

    def forward(self, data: AtomicGraph) -> dict[str, Tensor]:
        """Predict per-atom and total energy for an atomic graph.

        Parameters
        ----------
        data : AtomicGraph
            Input graph providing atomic numbers, ``edge_index`` of shape
            ``(2, E)`` (source/destination per edge) and the geometry used by
            the edge featurizer.

        Returns
        -------
        dict[str, Tensor]
            Mapping with ``"node_energy"`` (per-atom energy, shape ``(N,)``,
            the sum of the incident edge energies plus the per-element
            reference energy) and ``"energy"`` (total energy per structure,
            aggregated from the node energies).
        """
        node_attr = self.node_attr(data.atomic_numbers)
        edge = self.edge_feat(data)
        src, dst = data.edge_index[0], data.edge_index[1]

        two_body_in = torch.cat([node_attr[src], node_attr[dst], edge["edge_radial"]], dim=-1)
        latent = self.two_body(two_body_in)
        edge_feat = self.embed_edge(edge["edge_sh"])

        for layer in self.layers:
            edge_feat, latent = layer(edge_feat, edge["edge_sh"], latent)

        edge_energy = self.readout(latent).squeeze(-1)        # (E,)
        node_energy = scatter_sum(edge_energy, dst, data.num_nodes)   # assign to centre
        node_energy = node_energy + self.atom_ref(data.atomic_numbers).squeeze(-1)
        energy = self.aggregate_energy(node_energy, data)
        return {"node_energy": node_energy, "energy": energy}

    @classmethod
    def from_config(cls, cfg) -> "Allegro":
        """Build an :class:`Allegro` from a model configuration.

        Parameters
        ----------
        cfg : ModelConfig
            Model configuration. Uses ``cfg.cutoff``, ``cfg.n_rbf``,
            ``cfg.n_interactions``, ``cfg.n_features`` and the optional
            ``cfg.extra`` dict (``species``, ``l_max``, ``latent_dim``).

        Returns
        -------
        Allegro
            The constructed model instance.
        """
        extra = cfg.extra or {}
        return cls(
            species=extra.get("species", [1, 6, 8]),
            cutoff=cfg.cutoff,
            l_max=extra.get("l_max", 2),
            n_rbf=cfg.n_rbf,
            n_layers=cfg.n_interactions,
            mul=cfg.n_features // 4 if cfg.n_features >= 4 else 8,
            latent_dim=extra.get("latent_dim", 64),
        )
