"""Shared base for descriptor + per-element-network potentials (HDNNP / ANI).

A ``DescriptorPotential`` composes an invariant per-atom :class:`Featurizer`
(symmetry functions, AEV) with per-element atomic MLPs. HDNNP and ANI differ
mainly in the featurizer they pass in, so the model body lives here and is
reused by both ``hdnnp.py`` and ``ani.py``.

The per-element networks support the flexibility ANI needs: a distinct hidden
architecture per element, a choice of activation (ANI uses ``CELU``; the paper's
ANI-1 used a Gaussian; SchNet-style models use ``SiLU``), and optional
per-species *self atomic energies* added to each atom's contribution (ANI's
energy shift, in the same spirit as MACE/NequIP ``atomic_energies``).
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
from torch import Tensor, nn

from xnns.common.data import AtomicGraph
from xnns.common.featurizers import Featurizer
from xnns.common.models.base import InteratomicPotential


class _Gaussian(nn.Module):
    """Gaussian activation ``exp(-x^2)`` (the original ANI-1 hidden activation)."""

    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(-x * x)


def _make_activation(activation: Union[str, nn.Module]) -> nn.Module:
    """Return an activation ``nn.Module`` from a name or a module instance.

    Parameters
    ----------
    activation : str or torch.nn.Module
        One of ``"silu"``, ``"celu"`` (ANI, alpha=0.1), ``"gaussian"`` (original
        ANI-1), ``"tanh"``, ``"relu"``; or an ``nn.Module`` used as-is.

    Returns
    -------
    torch.nn.Module
        A fresh activation module.

    Raises
    ------
    ValueError
        If ``activation`` is an unknown name.
    """
    if isinstance(activation, nn.Module):
        return activation
    factories = {
        "silu": nn.SiLU,
        "celu": lambda: nn.CELU(alpha=0.1),   # ANI / torchani convention
        "gaussian": _Gaussian,
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
    }
    key = activation.lower()
    if key not in factories:
        raise ValueError(f"unknown activation {activation!r}; "
                         f"choose one of {sorted(factories)}")
    return factories[key]()


class _ElementNetworks(nn.Module):
    """One atomic MLP per element; dispatches atoms by species.

    Holds a separate MLP (mapping a descriptor to a scalar energy) for each
    chemical species, and routes each atom's descriptor to the network for its
    element.

    Parameters
    ----------
    species : sequence of int
        Atomic numbers to build a per-element network for.
    input_dim : int
        Dimension of the input per-atom descriptor (the featurizer output).
    hidden : sequence of int or dict[int, sequence of int], optional
        Hidden-layer widths of each atomic MLP. A single sequence is shared by
        every element; a ``{Z: widths}`` dict gives each element its own
        architecture (as ANI-1x does). Defaults to ``(64, 64)``.
    activation : str or torch.nn.Module, optional
        Hidden-layer activation, by default ``"silu"``. See
        :func:`_make_activation`.
    bias : bool, optional
        Whether the linear layers carry a bias, by default ``True``.

    Attributes
    ----------
    species : list[int]
        The elements handled.
    nets : torch.nn.ModuleDict
        Per-element MLPs keyed by ``str(atomic_number)``.
    """

    def __init__(self, species: Sequence[int], input_dim: int,
                 hidden: Union[Sequence[int], dict] = (64, 64),
                 activation: Union[str, nn.Module] = "silu", bias: bool = True):
        super().__init__()
        self.species = list(species)
        self.nets = nn.ModuleDict()
        for z in self.species:
            widths = hidden[z] if isinstance(hidden, dict) else hidden
            layers, d = [], input_dim
            for h in widths:
                layers += [nn.Linear(d, h, bias=bias), _make_activation(activation)]
                d = h
            layers += [nn.Linear(d, 1, bias=bias)]
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
    per-element atomic networks (+ optional per-species self atomic energies).
    Reuse by passing any invariant :class:`Featurizer`; HDNNP and ANI are thin
    subclasses that differ in the featurizer and per-element architecture.

    Parameters
    ----------
    featurizer : Featurizer
        Invariant featurizer mapping an :class:`AtomicGraph` to a per-atom
        descriptor of shape ``(N, featurizer.output_dim)``. Its ``cutoff`` sets
        the model's neighbour-list cutoff.
    species : sequence of int
        Atomic numbers to build per-element networks for.
    hidden : sequence of int or dict[int, sequence of int], optional
        Hidden-layer widths of each per-element MLP (shared sequence or per-Z
        dict), by default ``(64, 64)``.
    activation : str or torch.nn.Module, optional
        Hidden-layer activation, by default ``"silu"``.
    bias : bool, optional
        Whether the linear layers carry a bias, by default ``True``.
    atomic_energies : sequence of float or None, optional
        Per-species self atomic energy added to each atom's contribution
        (aligned with ``species``). ``None`` (default) adds nothing.

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

    def __init__(self, featurizer: Featurizer, species: Sequence[int],
                 hidden: Union[Sequence[int], dict] = (64, 64),
                 activation: Union[str, nn.Module] = "silu", bias: bool = True,
                 atomic_energies: Optional[Sequence[float]] = None):
        super().__init__()
        self.featurizer = featurizer
        self.cutoff = featurizer.cutoff
        self.node_feature_dim = featurizer.output_dim  # for e.g. LES
        self.species = list(species)
        self.element_nets = _ElementNetworks(
            species, featurizer.output_dim, hidden, activation, bias)

        # Per-species self energies, indexed directly by atomic number Z so the
        # forward pass can gather them without a Python dict lookup.
        max_z = max(self.species)
        sae = torch.zeros(max_z + 1)
        if atomic_energies is not None:
            ae = torch.as_tensor(atomic_energies, dtype=sae.dtype)
            if ae.numel() != len(self.species):
                raise ValueError(
                    f"atomic_energies: got {ae.numel()} values for "
                    f"{len(self.species)} species")
            for z, e in zip(self.species, ae.tolist()):
                sae[z] = e
        self.register_buffer("_self_energies_by_z", sae)

    def forward(self, data: AtomicGraph):
        """Compute per-atom and total energies for a batch of structures.

        Parameters
        ----------
        data : AtomicGraph
            Batched atomic graph passed to the featurizer.

        Returns
        -------
        dict[str, Tensor]
            Dictionary with ``"node_energy"`` (per-atom energy including the
            self-energy shift, shape ``(N,)``), ``"energy"`` (per-structure total
            from ``aggregate_energy``) and ``"node_features"`` (the descriptor).
        """
        desc = self.featurizer(data)
        node_energy = self.element_nets(desc, data.atomic_numbers)
        node_energy = node_energy + self._self_energies_by_z[data.atomic_numbers]
        energy = self.aggregate_energy(node_energy, data)
        return {"node_energy": node_energy, "energy": energy,
                "node_features": desc}
