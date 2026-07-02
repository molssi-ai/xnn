"""NequIP (Batzner et al. 2022): E(3)-equivariant message-passing potential.

Faithful architecture: species embedding -> N equivariant convolutions
(node features (x) spherical harmonics, radially weighted, with self-connection
and gated nonlinearity) -> invariant readout to per-atom energy.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from e3nn import o3
from e3nn.nn import FullyConnectedNet

from xnns.common.data import AtomicGraph
from xnns.common.models.registry import register_model
from .base import EquivariantGNN
from .blocks import EquivariantConv, species_irreps, hidden_irreps


@register_model("nequip")
class NequIP(EquivariantGNN):
    """NequIP (Batzner et al. 2022) E(3)-equivariant message-passing potential.

    A faithful NequIP architecture built on the shared
    :class:`~xnns.gnn.models.base.EquivariantGNN`: a species embedding is
    refined by ``n_layers`` :class:`~xnns.gnn.models.blocks.EquivariantConv`
    layers (node features tensor-multiplied with spherical harmonics, radially
    weighted, with an equivariant self-connection and gated nonlinearity),
    followed by an invariant MLP readout over the final scalar channels to a
    per-atom energy (plus the per-element reference energy).

    Parameters
    ----------
    species : list[int]
        Atomic numbers of the supported elements, in channel order.
    cutoff : float, optional
        Neighbour cutoff radius. Default is 5.0.
    l_max : int, optional
        Maximum spherical-harmonic degree. Default is 2.
    n_rbf : int, optional
        Number of radial basis functions in the edge featurizer. Default is 8.
    n_layers : int, optional
        Number of equivariant convolution layers. Default is 3.
    mul : int, optional
        Channel multiplicity of the hidden equivariant feature irreps. Default
        is 32.

    Attributes
    ----------
    embed : o3.Linear
        Equivariant linear embedding the one-hot species node attributes into
        the initial scalar node features.
    convs : torch.nn.ModuleList
        The stack of :class:`~xnns.gnn.models.blocks.EquivariantConv` layers.
    readout : e3nn.nn.FullyConnectedNet
        Invariant MLP mapping the final scalar channels to a per-atom energy.
    irreps_final : o3.Irreps
        Irreps of the node features after the last convolution.
    """

    def __init__(self, species: list[int], cutoff: float = 5.0, l_max: int = 2,
                 n_rbf: int = 8, n_layers: int = 3, mul: int = 32):
        super().__init__(species, cutoff, l_max, n_rbf)
        irreps_node = species_irreps(len(species))           # initial: scalars
        irreps_hidden = hidden_irreps(mul, l_max)
        n_radial = self.edge_feat.output_dim

        self.embed = o3.Linear(self.node_attr_irreps, irreps_node)
        self.convs = nn.ModuleList()
        irreps_in = irreps_node
        for _ in range(n_layers):
            conv = EquivariantConv(irreps_in, self.irreps_sh, irreps_hidden,
                                   self.node_attr_irreps, n_radial)
            self.convs.append(conv)
            irreps_in = conv.irreps_out
        # invariant readout from final scalar channels
        n_scalars = sum(m for m, ir in irreps_in if ir.l == 0)
        self.readout = FullyConnectedNet([n_scalars, n_scalars // 2, 1], F.silu)
        self.irreps_final = irreps_in

    def forward(self, data: AtomicGraph) -> dict[str, Tensor]:
        """Predict per-atom and total energy for an atomic graph.

        Parameters
        ----------
        data : AtomicGraph
            Input graph providing atomic numbers, ``edge_index`` of shape
            ``(2, E)`` and the geometry used by the edge featurizer.

        Returns
        -------
        dict[str, Tensor]
            Mapping with ``"node_energy"`` (per-atom energy, shape ``(N,)``,
            including the per-element reference energy) and ``"energy"`` (total
            energy per structure, aggregated from the node energies).
        """
        node_attr = self.node_attr(data.atomic_numbers)
        edge = self.edge_feat(data)
        x = self.embed(node_attr)
        for conv in self.convs:
            x = conv(x, node_attr, data.edge_index, edge["edge_sh"], edge["edge_radial"])
        # take l=0 (scalar) channels for the energy readout
        n_scalars = sum(m for m, ir in self.irreps_final if ir.l == 0)
        scalars = x[:, :n_scalars]
        node_energy = (self.readout(scalars).squeeze(-1)
                       + self.atom_ref(data.atomic_numbers).squeeze(-1))
        energy = self.aggregate_energy(node_energy, data)
        return {"node_energy": node_energy, "energy": energy}

    @classmethod
    def from_config(cls, cfg) -> "NequIP":
        """Build a :class:`NequIP` from a model configuration.

        Parameters
        ----------
        cfg : ModelConfig
            Model configuration. Uses ``cfg.cutoff``, ``cfg.n_rbf``,
            ``cfg.n_interactions``, ``cfg.n_features`` and the optional
            ``cfg.extra`` dict (``species``, ``l_max``).

        Returns
        -------
        NequIP
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
        )
