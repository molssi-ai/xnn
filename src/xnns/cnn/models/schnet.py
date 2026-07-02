"""SchNet (Schütt et al. 2017) -- continuous-filter convolutional network.

A complete, working reference implementation that exercises the entire library
(data -> model -> autograd forces -> training -> deploy). Use it as the template
for the message-passing pattern; the GNN/DNN models follow the same skeleton.

Works for molecules and periodic solids unchanged: periodicity enters only
through ``data.edge_vectors()``, which already accounts for cell shifts.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from xnns.common.data import AtomicGraph
from xnns.common.featurizers import GaussianRBF, CosineCutoff
from xnns.common.models.base import InteratomicPotential
from xnns.common.models.ops import scatter_sum
from xnns.common.models.registry import register_model

_MAX_Z = 100


class _CFConv(nn.Module):
    """Continuous-filter convolution block.

    Implements a single SchNet interaction: neighbour features are gated by a
    distance-dependent filter (derived from the radial basis expansion and the
    cosine cutoff) and summed onto each central atom.

    Parameters
    ----------
    n_features : int
        Dimension of the per-atom feature vectors.
    n_rbf : int
        Number of radial basis functions used to expand interatomic distances.
    cutoff : float
        Cutoff radius (in the same length units as the edge vectors) beyond
        which the cosine cutoff smoothly zeroes contributions.
    """

    def __init__(self, n_features: int, n_rbf: int, cutoff: float):
        super().__init__()
        self.lin_in = nn.Linear(n_features, n_features)
        self.lin_out = nn.Sequential(
            nn.Linear(n_features, n_features), nn.SiLU(),
            nn.Linear(n_features, n_features),
        )
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, n_features), nn.SiLU(),
            nn.Linear(n_features, n_features),
        )
        self.cutoff_fn = CosineCutoff(cutoff)

    def forward(self, x: Tensor, edge_index: Tensor, r: Tensor,
                rbf: Tensor) -> Tensor:
        """Apply one continuous-filter convolution.

        Parameters
        ----------
        x : Tensor
            Per-atom features, shape ``(N, F)``.
        edge_index : Tensor
            Edge index of shape ``(2, E)``; row 0 is the source (neighbour) and
            row 1 the destination (centre) node of each edge.
        r : Tensor
            Interatomic distances per edge, shape ``(E,)``.
        rbf : Tensor
            Radial basis expansion of ``r``, shape ``(E, n_rbf)``.

        Returns
        -------
        Tensor
            Updated per-atom features, shape ``(N, F)``.
        """
        src, dst = edge_index[0], edge_index[1]
        W = self.filter_net(rbf) * self.cutoff_fn(r)[:, None]   # (E, F)
        messages = self.lin_in(x)[src] * W                      # (E, F)
        agg = scatter_sum(messages, dst, x.shape[0])
        return self.lin_out(agg)


@register_model("schnet")
class SchNet(InteratomicPotential):
    """SchNet continuous-filter convolutional interatomic potential.

    Embeds atoms by species, refines their features through a stack of
    continuous-filter convolution (:class:`_CFConv`) interaction blocks, and
    reads out a per-atom energy plus a learnable per-element reference shift.
    Total energy is obtained by aggregating the per-atom energies over each
    structure. Works unchanged for molecules and periodic solids, since
    periodicity enters only through the edge vectors.

    Parameters
    ----------
    n_features : int, optional
        Dimension of the per-atom feature vectors, by default 128.
    n_interactions : int, optional
        Number of stacked continuous-filter convolution blocks, by default 3.
    n_rbf : int, optional
        Number of Gaussian radial basis functions used to expand distances,
        by default 50.
    cutoff : float, optional
        Cutoff radius for the radial basis and cosine cutoff, by default 5.0.

    Attributes
    ----------
    cutoff : float
        Neighbour-list cutoff radius.
    embedding : torch.nn.Embedding
        Species-to-feature embedding.
    rbf : GaussianRBF
        Gaussian radial basis expansion of interatomic distances.
    interactions : torch.nn.ModuleList
        Stack of :class:`_CFConv` interaction blocks.
    readout : torch.nn.Sequential
        MLP mapping final per-atom features to a scalar energy.
    atom_ref : torch.nn.Embedding
        Learnable per-element energy reference (shift), key for transferability.
    """

    def __init__(self, n_features: int = 128, n_interactions: int = 3,
                 n_rbf: int = 50, cutoff: float = 5.0):
        super().__init__()
        self.cutoff = cutoff
        self.embedding = nn.Embedding(_MAX_Z, n_features)
        self.rbf = GaussianRBF(n_rbf, cutoff)
        self.interactions = nn.ModuleList(
            [_CFConv(n_features, n_rbf, cutoff) for _ in range(n_interactions)]
        )
        self.readout = nn.Sequential(
            nn.Linear(n_features, n_features // 2), nn.SiLU(),
            nn.Linear(n_features // 2, 1),
        )
        # per-element energy reference (learnable shift), key for transferability
        self.atom_ref = nn.Embedding(_MAX_Z, 1)
        nn.init.zeros_(self.atom_ref.weight)

    @torch.jit.export
    def node_energy(self, atomic_numbers: Tensor, edge_index: Tensor,
                    edge_vec: Tensor) -> Tensor:
        """TorchScript-compatible core: tensors in, per-atom energy out.

        This is what the deploy wrappers (LAMMPS/TorchScript) call, so it must
        avoid the AtomicGraph dataclass and any Python-only constructs.

        Parameters
        ----------
        atomic_numbers : Tensor
            Per-atom atomic numbers, shape ``(N,)``.
        edge_index : Tensor
            Edge index of shape ``(2, E)``; row 0 is the source (neighbour) and
            row 1 the destination (centre) node of each edge.
        edge_vec : Tensor
            Edge displacement vectors, shape ``(E, 3)`` (already accounting for
            any periodic cell shifts).

        Returns
        -------
        Tensor
            Per-atom energy, shape ``(N,)``, including the learnable per-element
            reference shift.
        """
        x = self.embedding(atomic_numbers)
        r = torch.linalg.norm(edge_vec, dim=-1)
        rbf = self.rbf(r)
        for block in self.interactions:
            x = x + block(x, edge_index, r, rbf)
        return (self.readout(x).squeeze(-1)
                + self.atom_ref(atomic_numbers).squeeze(-1))

    @torch.jit.ignore
    def forward(self, data: AtomicGraph) -> dict[str, Tensor]:
        """Compute per-atom and total energies for a batch of structures.

        Parameters
        ----------
        data : AtomicGraph
            Batched atomic graph providing atomic numbers, edge index and
            edge vectors.

        Returns
        -------
        dict[str, Tensor]
            Dictionary with ``"node_energy"`` (per-atom energy, shape ``(N,)``)
            and ``"energy"`` (per-structure total energy).
        """
        node_energy = self.node_energy(
            data.atomic_numbers, data.edge_index, data.edge_vectors())
        energy = self.aggregate_energy(node_energy, data)
        return {"node_energy": node_energy, "energy": energy}

    @classmethod
    def from_config(cls, cfg) -> "SchNet":
        """Build a :class:`SchNet` from a configuration object.

        Parameters
        ----------
        cfg : object
            Configuration exposing ``n_features``, ``n_interactions``,
            ``n_rbf`` and ``cutoff`` attributes.

        Returns
        -------
        SchNet
            Instantiated model.
        """
        return cls(
            n_features=cfg.n_features,
            n_interactions=cfg.n_interactions,
            n_rbf=cfg.n_rbf,
            cutoff=cfg.cutoff,
        )
