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
from xnn.common.models.fast import FastPathModule
from xnn.common.models.ops import gather_rows, make_activation


class _ElementNetworks(FastPathModule, nn.Module):
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
    n_outputs : int, optional
        Width of every network's output layer, by default 1 (a scalar per
        atom).

    Attributes
    ----------
    species : list[int]
        The elements handled.
    input_dims : dict[int, int]
        Descriptor columns read by every element.
    nets : torch.nn.ModuleDict
        Per-element MLPs keyed by ``str(atomic_number)``.
    grouping : bool
        Whether the fast path may evaluate the networks together (set by the
        owning model, only when :meth:`groupable`): one batched matrix product
        per layer over the atoms of every element, padded to the largest
        element, instead of one product per element.
    """

    def __init__(self, species: Sequence[int], input_dim: Union[int, dict],
                 hidden: Union[Sequence[int], dict] = (64, 64),
                 activation: Union[str, nn.Module, Sequence, dict] = "silu",
                 bias: bool = True, extra_inputs: int = 0, n_outputs: int = 1):
        super().__init__()
        self.species = list(species)
        self.n_outputs = int(n_outputs)
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
            for h, act in zip(widths + [self.n_outputs], acts):
                layers.append(nn.Linear(d, h, bias=bias))
                if act is not None and not (isinstance(act, str) and act.lower() == "linear"
                                            and h == self.n_outputs):
                    layers.append(make_activation(act))
                d = h
            self.nets[str(z)] = nn.Sequential(*layers)
        self.grouping = False
        self.fast_active = False

    @torch.jit.unused
    def groupable(self) -> bool:
        """Whether every element has the same architecture (layer shapes, activations, inputs)."""
        nets = list(self.nets.values())
        if len(set(self.input_dims.values())) != 1:
            return False
        for net in nets[1:]:
            if len(net) != len(nets[0]):
                return False
            for a, b in zip(nets[0], net):
                if type(a) is not type(b):
                    return False
                if isinstance(a, nn.Linear):
                    if a.weight.shape != b.weight.shape or (a.bias is None) != (b.bias is None):
                        return False
                elif repr(a) != repr(b):
                    return False
        return True

    def fast_supported(self, device: torch.device, dtype: torch.dtype) -> bool:
        """The grouped evaluation, where the owning model allows it, on a CUDA device."""
        return self.grouping and device.type == "cuda"

    def plan(self, atomic_numbers: Tensor) -> Tuple[Tensor, Tensor, List[int]]:
        """The atoms sorted by element, which several calls on the same atoms can share.

        Parameters
        ----------
        atomic_numbers : Tensor
            Per-atom atomic numbers ``(N,)``.

        Returns
        -------
        tuple of (Tensor, Tensor, list of int)
            The position of every atom's element in ``species`` ``(N,)``
            (``len(species)``: no network), the stable sorting order ``(N,)``
            and the block size of every element and of the atoms without a
            network (one device-to-host copy).
        """
        n_species = len(self.species)
        code = self._element_code(atomic_numbers)
        order = torch.argsort(code, stable=True)
        sizes: List[int] = torch.bincount(code, minlength=n_species + 1).cpu().tolist()
        return code, order, sizes

    def _element_code(self, atomic_numbers: Tensor) -> Tensor:
        """The position of every atom's element in ``species`` ``(N,)`` (``len(species)``: no network)."""
        n_species = len(self.species)
        code = self._code_of_z[atomic_numbers.clamp(max=self._code_of_z.shape[0] - 1)]
        return torch.where(atomic_numbers < self._code_of_z.shape[0], code, torch.full_like(code, n_species))

    @torch.jit.unused
    def _forward_grouped(self, desc: Tensor, extra: Optional[Tensor], code: Tensor, order: Tensor,
                         sizes: List[int]) -> Tensor:
        """:meth:`forward` with one batched product per layer over the padded element blocks."""
        n_species, n = len(self.species), desc.shape[0]
        counts = sizes[:n_species]
        width = max(counts)
        if width == 0:
            out = desc.new_zeros(n, self.n_outputs)
            return out.squeeze(-1) if self.n_outputs == 1 else out
        kept = sum(counts)
        rows = order[:kept]
        # the slot of every atom in its element's padded block: block start plus rank in the block
        offsets = [0]
        for c in counts[:-1]:
            offsets.append(offsets[-1] + c)
        element = code[rows]
        start = torch.tensor(offsets, dtype=torch.long, device=desc.device)
        slot = element * width + (torch.arange(kept, device=desc.device) - start[element])
        return self._grouped_layers(desc, extra, rows, slot, width)

    @torch.jit.unused
    def forward_padded(self, desc: Tensor, atomic_numbers: Tensor, extra: Optional[Tensor],
                       width: int) -> Tensor:
        """:meth:`forward` as the grouped evaluation with element blocks of ``width`` rows, without
        reading a value back from the device (a captured step).

        Parameters
        ----------
        desc, atomic_numbers, extra
            As for :meth:`forward`.
        width : int
            Rows per element block, at least the largest element count (the atom count always is).

        Returns
        -------
        Tensor
            As :meth:`forward`.
        """
        n_species, n = len(self.species), desc.shape[0]
        code = self._element_code(atomic_numbers)
        order = torch.argsort(code, stable=True)
        # counted by an index_add into a fixed length: bincount sizes its output on the host
        counts = torch.zeros(n_species + 1, dtype=torch.long, device=desc.device).index_add_(
            0, code, torch.ones_like(code))
        start = torch.cumsum(counts, 0) - counts
        element = code[order]
        rank = torch.arange(n, device=desc.device) - start[element]
        # atoms without a network go to one spare row past the blocks, whose output is zero
        slot = torch.where(element < n_species, element * width + rank, torch.full_like(rank, n_species * width))
        return self._grouped_layers(desc, extra, order, slot, width)

    @torch.jit.unused
    def _grouped_layers(self, desc: Tensor, extra: Optional[Tensor], rows: Tensor, slot: Tensor,
                        width: int) -> Tensor:
        """The networks of every element as batched products: atom ``rows[i]`` sits at ``slot[i]`` of a
        table of ``len(species) * width`` rows plus a spare one, and gets its output from there."""
        n_species, n = len(self.species), desc.shape[0]
        columns = self.input_dims[self.species[0]]
        x = desc if columns == desc.shape[1] else desc[:, :columns]
        if self.extra_inputs > 0 and extra is not None:
            x = torch.cat([x, extra.reshape(n, -1).to(x.dtype)], dim=-1)
        table = x.new_zeros(n_species * width + 1, x.shape[1]).index_copy(0, slot, gather_rows(x, rows))
        h = table[:n_species * width].view(n_species, width, -1)
        nets = list(self.nets.values())
        for i, layer in enumerate(nets[0]):
            if isinstance(layer, nn.Linear):
                weight = torch.stack([net[i].weight for net in nets]).transpose(1, 2)
                if layer.bias is not None:
                    h = torch.baddbmm(torch.stack([net[i].bias for net in nets]).unsqueeze(1), h, weight)
                else:
                    h = torch.bmm(h, weight)
            else:
                h = layer(h)
        flat = torch.cat([h.reshape(n_species * width, self.n_outputs), h.new_zeros(1, self.n_outputs)])
        out = x.new_zeros(n, self.n_outputs).index_copy(0, rows, gather_rows(flat, slot))
        return out.squeeze(-1) if self.n_outputs == 1 else out

    def forward(self, desc: Tensor, atomic_numbers: Tensor,
                extra: Optional[Tensor] = None,
                plan: Optional[Tuple[Tensor, Tensor, List[int]]] = None) -> Tensor:
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
        plan : tuple, optional
            The :meth:`plan` of ``atomic_numbers``, when the caller has it.

        Returns
        -------
        Tensor
            Per-atom output, shape ``(N,)``, or ``(N, n_outputs)`` for
            networks with several outputs.
        """
        n_species = len(self.species)
        # the atoms sorted by element: one gather, a contiguous block per network, one scatter
        # back, and a single device-to-host copy (the block sizes)
        if plan is None:
            code, order, sizes = self.plan(atomic_numbers)
        else:
            code, order, sizes = plan
        if self.fast_active and not torch.jit.is_scripting():
            return self._forward_grouped(desc, extra, code, order, sizes)
        x_all = gather_rows(desc, order)
        e_all = gather_rows(extra.reshape(desc.shape[0], -1), order) if (
            self.extra_inputs > 0 and extra is not None) else None
        # contiguous blocks by split (its backward is one concatenation)
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
                pieces.append(net(x).squeeze(-1) if self.n_outputs == 1 else net(x))
            k += 1
        rest = sizes[n_species]
        if self.n_outputs != 1:
            if rest > 0:
                pieces.append(desc.new_zeros(rest, self.n_outputs))
            values = torch.cat(pieces) if len(pieces) > 0 else desc.new_zeros(0, self.n_outputs)
            return desc.new_zeros(desc.shape[0], self.n_outputs).index_copy(0, order, values)
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
    def fast_policies(self):
        """The ``"auto"`` policies (evaluation, training): the featurizer's (``None``: always fast)."""
        policies = getattr(self.featurizer, "fast_policies", None)
        return (None, None) if policies is None else policies()

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
        policies = self.fast_policies()
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
