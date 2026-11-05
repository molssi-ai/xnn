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

from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from xnn.common.data import AtomicGraph
from xnn.common.featurizers import Featurizer
from xnn.common.models.base import InteratomicPotential
from xnn.common.models.ops import gather_rows, make_activation


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
        self._z_of_name = {str(z): int(z) for z in self.species}
        # position of every element in ``species`` by atomic number (``len(species)``: no network)
        code = torch.full((max(self.species) + 1,), len(self.species), dtype=torch.long)
        for k, z in enumerate(self.species):
            code[z] = k
        self.register_buffer("_code_of_z", code, persistent=False)
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
        n_species = len(self.species)
        code = self._code_of_z[atomic_numbers.clamp(max=self._code_of_z.shape[0] - 1)]
        code = torch.where(atomic_numbers < self._code_of_z.shape[0], code, torch.full_like(code, n_species))
        # the atoms sorted by element: one gather, a contiguous block per network, one scatter
        # back, and a single device-to-host copy (the block sizes)
        order = torch.argsort(code, stable=True)
        counts = torch.bincount(code, minlength=n_species + 1).cpu()
        x_all = gather_rows(desc, order)
        e_all = gather_rows(extra.reshape(desc.shape[0], -1), order) if (
            self.extra_inputs > 0 and extra is not None) else None
        # contiguous blocks by split (its backward is one concatenation)
        sizes: List[int] = counts.tolist()
        blocks = x_all.split(sizes)
        e_blocks = e_all.split(sizes) if e_all is not None else None
        pieces: List[Tensor] = []
        k = 0
        # over the networks by item (TorchScript takes no computed module keys)
        for name, net in self.nets.items():
            if sizes[k] > 0:
                x = blocks[k]
                width = self.input_dims[self._z_of_name[name]]
                if width < x.shape[1]:
                    x = x[:, :width]
                if e_blocks is not None:
                    x = torch.cat([x, e_blocks[k].to(x.dtype)], dim=-1)
                pieces.append(net(x).squeeze(-1))
            k += 1
        rest = sizes[n_species]
        if rest > 0:
            pieces.append(desc.new_zeros(rest))
        values = torch.cat(pieces) if len(pieces) > 0 else desc.new_zeros(0)
        return desc.new_zeros(desc.shape[0]).scatter(0, order, values)


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
        # the fused featurizer kernels where it has them (see set_use_fast)
        self.use_fast = "auto"

    @torch.jit.unused
    def set_use_fast(self, use_fast) -> "DescriptorPotential":
        """Choose the implementation of the featurizer.

        Parameters
        ----------
        use_fast : bool or str
            ``"auto"`` (the default): the fused GPU kernels of the featurizer
            (the AEV of :mod:`xnn.dnn.fast.aev`) where they are available and
            faster for the input (a CUDA device, Triton, a graph above the size
            crossover measured for evaluation, ``AUTO_POLICY``, or for training,
            ``AUTO_POLICY_TRAINING``); ``True``: wherever they are available;
            ``False``: always the reference implementation. The kernels are
            twice differentiable, so force training takes them as well. The
            HDNNP symmetry functions have GPU kernels and, in evaluation, a
            compiled CPU path and optional C++ kernels (:mod:`xnn.dnn.fast.acsf`).

        Returns
        -------
        DescriptorPotential
            ``self``.
        """
        from xnn.common.models.fast import resolve_use_fast
        self.use_fast = resolve_use_fast(use_fast)
        return self

    @torch.jit.unused
    def _select_fast(self, num_edges: int) -> None:
        """Switch the fused featurizer kernels on or off for one evaluation."""
        from xnn.common.models.fast import fast_modules, select
        blocks = self.__dict__.get("_fast_blocks")
        if blocks is None:
            blocks = fast_modules(self)
            self.__dict__["_fast_blocks"] = blocks
        if not blocks:
            return
        policies = self.featurizer.fast_policies()
        like = self._self_energies_by_z
        policy = policies[1] if self.training else policies[0]
        dtype = next(self.element_nets.parameters()).dtype
        select(blocks, self.use_fast, policy, like.device, dtype, num_edges)

    @torch.jit.export
    def node_features_energy(self, atomic_numbers: Tensor, edge_index: Tensor,
                             edge_vec: Tensor) -> Tuple[Tensor, Tensor]:
        """Tensor core (TorchScript-compatible): the descriptor ``(N, D)`` and the per-atom
        energies ``(N,)`` including the self energies, from atomic numbers ``(N,)``, edges
        ``(2, E)`` and edge vectors ``(E, 3)``. The featurizer must offer ``compute``."""
        desc = self.featurizer.compute(atomic_numbers, edge_index, edge_vec, atomic_numbers.shape[0])
        node_energy = self.element_nets(desc, atomic_numbers)
        return desc, node_energy + self._self_energies_by_z[atomic_numbers]

    @torch.jit.export
    def node_energy(self, atomic_numbers: Tensor, edge_index: Tensor, edge_vec: Tensor) -> Tensor:
        """Per-atom energies ``(N,)`` (the deploy entry point, :meth:`node_features_energy`)."""
        return self.node_features_energy(atomic_numbers, edge_index, edge_vec)[1]

    @torch.jit.ignore
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
        self._select_fast(int(data.edge_index.shape[1]))
        desc = self.featurizer(data)
        node_energy = self.element_nets(desc, data.atomic_numbers)
        node_energy = node_energy + self._self_energies_by_z[data.atomic_numbers]
        energy = self.aggregate_energy(node_energy, data)
        return {"node_energy": node_energy, "energy": energy,
                "node_features": desc}
