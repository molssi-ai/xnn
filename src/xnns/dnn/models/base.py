"""Shared base for descriptor + per-element-network potentials (HDNNP / ANI).

A ``DescriptorPotential`` composes an invariant per-atom :class:`Featurizer`
(symmetry functions, AEV) with per-element atomic MLPs. HDNNP and ANI differ
only in the featurizer they pass in, so the model body lives here and is reused
by both ``hdnnp.py`` and ``ani.py``.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from xnns.common.data import AtomicGraph
from xnns.common.featurizers import Featurizer
from xnns.common.models.base import InteratomicPotential


class _ElementNetworks(nn.Module):
    """One atomic MLP per element; dispatches atoms by species.

    Holds a separate SiLU-activated MLP (mapping a descriptor to a scalar
    energy) for each chemical species, and routes each atom's descriptor to the
    network for its element.

    Parameters
    ----------
    species : sequence of int
        Atomic numbers to build a per-element network for.
    input_dim : int
        Dimension of the input per-atom descriptor (the featurizer output).
    hidden : sequence of int, optional
        Hidden-layer widths of each atomic MLP, by default ``(64, 64)``.

    Attributes
    ----------
    species : list[int]
        The elements handled.
    nets : torch.nn.ModuleDict
        Per-element MLPs keyed by ``str(atomic_number)``.
    """

    def __init__(self, species, input_dim, hidden=(64, 64)):
        super().__init__()
        self.species = list(species)
        self.nets = nn.ModuleDict()
        for z in self.species:
            layers, d = [], input_dim
            for h in hidden:
                layers += [nn.Linear(d, h), nn.SiLU()]
                d = h
            layers += [nn.Linear(d, 1)]
            self.nets[str(z)] = nn.Sequential(*layers)

    def forward(self, desc: Tensor, atomic_numbers: Tensor) -> Tensor:
        """Map per-atom descriptors to per-atom energies via element networks.

        Parameters
        ----------
        desc : Tensor
            Per-atom descriptors, shape ``(N, input_dim)``.
        atomic_numbers : Tensor
            Per-atom atomic numbers, shape ``(N,)``, used to select each atom's
            element network.

        Returns
        -------
        Tensor
            Per-atom energy, shape ``(N,)``.
        """
        node_energy = torch.zeros(desc.shape[0], device=desc.device, dtype=desc.dtype)
        for z in self.species:
            mask = atomic_numbers == z
            if mask.any():
                node_energy = node_energy.clone()
                node_energy[mask] = self.nets[str(z)](desc[mask]).squeeze(-1)
        return node_energy


class DescriptorPotential(InteratomicPotential):
    """Shared body for descriptor-based potentials (HDNNP, ANI).

    Composition: featurizer (``AtomicGraph -> per-atom descriptor``) +
    per-element atomic networks. Reuse by passing any invariant
    :class:`Featurizer`; HDNNP and ANI are thin subclasses that differ only in
    the featurizer supplied.

    Parameters
    ----------
    featurizer : Featurizer
        Invariant featurizer mapping an :class:`AtomicGraph` to a per-atom
        descriptor of shape ``(N, featurizer.output_dim)``. Its ``cutoff`` sets
        the model's neighbour-list cutoff.
    species : sequence of int
        Atomic numbers to build per-element networks for.
    hidden : sequence of int, optional
        Hidden-layer widths of each per-element MLP, by default ``(64, 64)``.

    Attributes
    ----------
    featurizer : Featurizer
        The composed featurizer.
    cutoff : float
        Neighbour-list cutoff, taken from ``featurizer.cutoff``.
    species : list[int]
        The elements handled.
    element_nets : _ElementNetworks
        Per-element atomic MLPs.
    """

    def __init__(self, featurizer: Featurizer, species, hidden=(64, 64)):
        super().__init__()
        self.featurizer = featurizer
        self.cutoff = featurizer.cutoff
        self.node_feature_dim = featurizer.output_dim  # for e.g. LES
        self.species = list(species)
        self.element_nets = _ElementNetworks(species, featurizer.output_dim, hidden)

    def forward(self, data: AtomicGraph):
        """Compute per-atom and total energies for a batch of structures.

        Parameters
        ----------
        data : AtomicGraph
            Batched atomic graph passed to the featurizer.

        Returns
        -------
        dict[str, Tensor]
            Dictionary with ``"node_energy"`` (per-atom energy, shape ``(N,)``)
            and ``"energy"`` (per-structure total energy from
            ``aggregate_energy``).
        """
        desc = self.featurizer(data)
        node_energy = self.element_nets(desc, data.atomic_numbers)
        energy = self.aggregate_energy(node_energy, data)
        return {"node_energy": node_energy, "energy": energy,
                "node_features": desc}
