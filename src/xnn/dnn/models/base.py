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

from xnn.common.data import AtomicGraph
from xnn.common.featurizers import Featurizer
from xnn.common.models.base import InteratomicPotential
from xnn.common.models.ops import make_activation


class _ElementNetworks(nn.Module):
    """One atomic MLP per element; dispatches atoms by species.

    Holds a separate MLP (mapping a descriptor to a scalar) for each chemical
    species, and routes each atom's descriptor to the network for its element.

    Parameters
    ----------
    species : sequence of int
        Atomic numbers to build a per-element network for.
    input_dim : int or dict[int, int]
        Width of the per-atom descriptor, or a ``{Z: width}`` dict when the
        elements have descriptors of different lengths (the HDNNP); an element
        reads the first ``width`` columns of the (zero-padded) descriptor.
    hidden : sequence of int or dict[int, sequence of int], optional
        Hidden-layer widths of each atomic MLP. A single sequence is shared by
        every element; a ``{Z: widths}`` dict gives each element its own
        architecture (as ANI-1x does). Defaults to ``(64, 64)``.
    activation : str, torch.nn.Module, sequence or dict, optional
        Activation of the hidden layers, by default ``"silu"``, with a linear
        output. A sequence gives one activation per layer *including* the
        output layer (``len(hidden) + 1`` entries, e.g. ``["tanh", "tanh",
        "linear"]``); a ``{Z: ...}`` dict sets either form per element. See
        :func:`~xnn.common.models.ops.make_activation`.
    bias : bool, optional
        Whether the linear layers carry a bias, by default ``True``.
    extra_inputs : int, optional
        Number of per-atom inputs appended after the descriptor columns (the
        atomic charge of the 4G-HDNNP), by default 0.

    Attributes
    ----------
    species : list[int]
        The elements handled.
    input_dims : dict[int, int]
        Descriptor columns read by every element.
    nets : torch.nn.ModuleDict
        Per-element MLPs keyed by ``str(atomic_number)``.
    """

    def __init__(self, species: Sequence[int], input_dim: Union[int, dict],
                 hidden: Union[Sequence[int], dict] = (64, 64),
                 activation: Union[str, nn.Module, Sequence, dict] = "silu",
                 bias: bool = True, extra_inputs: int = 0):
        super().__init__()
        self.species = list(species)
        self.input_dims = {z: int(input_dim[z]) if isinstance(input_dim, dict) else int(input_dim)
                           for z in self.species}
        self.extra_inputs = int(extra_inputs)
        self.nets = nn.ModuleDict()
        for z in self.species:
            widths = list(hidden[z] if isinstance(hidden, dict) else hidden)
            acts = activation[z] if isinstance(activation, dict) else activation
            if isinstance(acts, (str, nn.Module)):
                acts = [acts] * len(widths) + [None]
            acts = list(acts)
            if len(acts) != len(widths) + 1:
                raise ValueError(f"element {z}: {len(acts)} activations for {len(widths)} hidden "
                                 f"layers and the output layer (need {len(widths) + 1})")
            layers, d = [], self.input_dims[z] + self.extra_inputs
            for h, act in zip(widths + [1], acts):
                layers.append(nn.Linear(d, h, bias=bias))
                if act is not None and not (isinstance(act, str) and act.lower() == "linear"
                                            and h == 1):
                    layers.append(make_activation(act))
                d = h
            self.nets[str(z)] = nn.Sequential(*layers)

    def forward(self, desc: Tensor, atomic_numbers: Tensor,
                extra: Optional[Tensor] = None) -> Tensor:
        """Map per-atom descriptors to per-atom outputs via element networks.

        Parameters
        ----------
        desc : Tensor
            Per-atom descriptors, shape ``(N, D)``.
        atomic_numbers : Tensor
            Per-atom atomic numbers, shape ``(N,)``, used to select each atom's
            element network.
        extra : Tensor, optional
            The ``(N, extra_inputs)`` inputs appended to the descriptor.

        Returns
        -------
        Tensor
            Per-atom output, shape ``(N,)``.
        """
        out = torch.zeros(desc.shape[0], device=desc.device, dtype=desc.dtype)
        for z in self.species:
            mask = atomic_numbers == z
            if mask.any():
                x = desc[mask][:, :self.input_dims[z]]
                if self.extra_inputs:
                    x = torch.cat([x, extra[mask].reshape(-1, self.extra_inputs).to(x.dtype)], dim=-1)
                out = out.clone()
                out[mask] = self.nets[str(z)](x).squeeze(-1)
        return out


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
        the model's neighbour-list cutoff. A featurizer with an ``n_features``
        ``{Z: width}`` dict (the atom-centered symmetry functions) gives every
        element a network on its own leading columns.
    species : sequence of int
        Atomic numbers to build per-element networks for.
    hidden : sequence of int or dict[int, sequence of int], optional
        Hidden-layer widths of each per-element MLP (shared sequence or per-Z
        dict), by default ``(64, 64)``.
    activation : str, torch.nn.Module, sequence or dict, optional
        Hidden-layer activation, by default ``"silu"``, or the per-layer list
        of :class:`_ElementNetworks`.
    bias : bool, optional
        Whether the linear layers carry a bias, by default ``True``.
    atomic_energies : sequence of float or None, optional
        Per-species self atomic energy added to each atom's contribution
        (aligned with ``species``). ``None`` (default) adds nothing.
    extra_inputs : int, optional
        Per-atom inputs appended to the descriptor of the element networks,
        by default 0.

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

    # one readout head = the per-element networks and self energies (the
    # descriptor is the shared trunk; see MultiHead)
    head_modules = ("element_nets", "_self_energies_by_z")

    def __init__(self, featurizer: Featurizer, species: Sequence[int],
                 hidden: Union[Sequence[int], dict] = (64, 64),
                 activation: Union[str, nn.Module, Sequence, dict] = "silu", bias: bool = True,
                 atomic_energies: Optional[Sequence[float]] = None, extra_inputs: int = 0):
        super().__init__()
        self.featurizer = featurizer
        self.cutoff = featurizer.cutoff
        self.node_feature_dim = featurizer.output_dim  # for e.g. LES
        self.species = list(species)
        dims = getattr(featurizer, "n_features", None) or featurizer.output_dim
        self.element_nets = _ElementNetworks(
            species, dims, hidden, activation, bias, extra_inputs)

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
