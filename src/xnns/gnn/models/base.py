"""Shared base for the E(3)-equivariant GNN potentials (NequIP / MACE / Allegro).

Holds the pieces every equivariant GNN in this package needs: species
bookkeeping (atomic-number <-> element-index table, one-hot node attributes),
the per-element reference energy ``atom_ref``, and the shared
:class:`~xnns.gnn.featurizers.SphericalHarmonicEdgeEmbedding` edge featurizer.

Factored out of the individual model modules so ``nequip.py``, ``mace.py`` and
``allegro.py`` can all subclass it (see the package-structure convention: code
common to several models of one family lives in a shared module of that family).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from e3nn import o3

from xnns.common.models.base import InteratomicPotential
from ..featurizers import SphericalHarmonicEdgeEmbedding
from .blocks import species_irreps


class EquivariantGNN(InteratomicPotential):
    """Shared base for the E(3)-equivariant GNN potentials.

    Provides the pieces every equivariant GNN in this package needs and is
    subclassed by :class:`~xnns.gnn.models.nequip.NequIP`, MACE and
    :class:`~xnns.gnn.models.allegro.Allegro`. It handles species bookkeeping
    (an atomic-number -> element-index lookup table and one-hot node
    attributes), the per-element reference energy ``atom_ref``, and constructs
    the shared
    :class:`~xnns.gnn.featurizers.SphericalHarmonicEdgeEmbedding` edge
    featurizer plus the spherical-harmonic irreps used by the interaction
    blocks.

    Parameters
    ----------
    species : list[int]
        Atomic numbers of the elements the model supports, in order. The list
        index defines the element-index (channel) of each species.
    cutoff : float
        Cutoff radius passed to the edge featurizer.
    l_max : int
        Maximum spherical-harmonic degree; sets ``irreps_sh`` via
        ``o3.Irreps.spherical_harmonics(l_max)``.
    n_rbf : int
        Number of radial basis functions in the edge featurizer.

    Attributes
    ----------
    species : list[int]
        The supported atomic numbers.
    z_to_index : Tensor
        Registered long buffer of shape ``(200,)`` mapping atomic number to
        element index (``-1`` for unsupported elements).
    node_attr_irreps : o3.Irreps
        Irreps of the one-hot species node attributes (``n_species`` scalars).
    irreps_sh : o3.Irreps
        Spherical-harmonic irreps of the equivariant edge attributes.
    edge_feat : SphericalHarmonicEdgeEmbedding
        The shared edge featurizer.
    atom_ref : torch.nn.Embedding
        Per-element reference-energy embedding (200 entries, one scalar each),
        initialized to zero.
    """

    def __init__(self, species: list[int], cutoff: float, l_max: int, n_rbf: int):
        super().__init__()
        self.species = list(species)
        self.cutoff = cutoff
        self.l_max = l_max
        z_to_index = torch.full((200,), -1, dtype=torch.long)
        for i, z in enumerate(self.species):
            z_to_index[z] = i
        self.register_buffer("z_to_index", z_to_index)
        self.node_attr_irreps = species_irreps(len(self.species))
        self.irreps_sh = o3.Irreps.spherical_harmonics(l_max)
        self.edge_feat = SphericalHarmonicEdgeEmbedding(l_max, n_rbf, cutoff)
        self.atom_ref = nn.Embedding(200, 1)
        nn.init.zeros_(self.atom_ref.weight)

    def node_attr(self, atomic_numbers: Tensor) -> Tensor:
        """Build the one-hot species node attributes.

        Parameters
        ----------
        atomic_numbers : Tensor
            Long tensor of shape ``(N,)`` giving the atomic number of each of
            the ``N`` nodes.

        Returns
        -------
        Tensor
            One-hot node attributes of shape ``(N, n_species)``, cast to the
            dtype of ``atom_ref``. These are the invariant ``node_attr`` inputs
            consumed by the interaction blocks' self-connections.
        """
        idx = self.z_to_index[atomic_numbers]
        return F.one_hot(idx.clamp(min=0), len(self.species)).to(self.atom_ref.weight.dtype)
