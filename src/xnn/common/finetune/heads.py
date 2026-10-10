"""Multi-head potentials: one shared trunk, one readout and reference per head.

Multi-head replay fine-tuning (Batatia *et al.*, arXiv:2401.00096; Tompa
*et al.*, arXiv:2606.12704) trains a pretrained model on the target data and
on a replay set of the pretraining distribution at the same time. The two
sets share the embedding and message-passing layers and differ only in the
readout and in the per-element reference energies, so the replay head keeps
the model close to its pretraining behaviour while the target head adapts.

:class:`MultiHead` gives any xnn potential that declares its readout modules
(:attr:`~xnn.common.models.base.InteratomicPotential.head_modules`) several
such heads. Head 0 **is** the wrapped model's own readout; every further head
is an independent copy of those modules. A structure is routed to its head by
the ``head`` field of the :class:`~xnn.common.data.AtomicGraph` (set with
:func:`label_head`); a batch may mix heads, each sub-batch runs through the
trunk once with its own head's parameters substituted, and the outputs are
reassembled in the original order. A deployed model has one head:
:meth:`MultiHead.select` returns the plain potential of a head.

Enable from a config with ``model.extra["heads"]``::

    model:
      pretrained: mace-mp-0-small
      heads: [pt_head, Default]          # head 0 keeps the pretrained readout and E0s
      # or, with per-head reference energies:
      heads: {pt_head: null, Default: {atomic_energies: estimated}}
"""
from __future__ import annotations

import contextlib
import copy
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import torch
from torch import Tensor, nn

from ..data import AtomicGraph
from ..models.base import InteratomicPotential

# output keys that are per structure even when a batch has as many structures
# as atoms (every other key with one row per atom is taken as per atom)
_STRUCTURE_KEYS = frozenset({"energy", "stress", "dipole", "polarizability", "energy_nn", "energy_elec",
                             "energy_2body", "energy_3body", "energy_dispersion",
                             "energy_long_range", "energy_short_range"})


def head_module_names(model: nn.Module) -> tuple[str, ...]:
    """The attributes of ``model`` that make up one readout head.

    Parameters
    ----------
    model : torch.nn.Module
        A potential.

    Returns
    -------
    tuple of str
        The model's :attr:`~xnn.common.models.base.InteratomicPotential.head_modules`.

    Raises
    ------
    TypeError
        If the model declares none (a classical force field, say).
    """
    names = tuple(getattr(model, "head_modules", ()) or ())
    if not names:
        raise TypeError(
            f"{type(model).__name__} declares no head modules (head_modules is empty), "
            "so it cannot be given several readout heads")
    for name in names:
        if not hasattr(model, name):
            raise AttributeError(f"{type(model).__name__}.head_modules names {name!r}, "
                                 "which the model does not have")
    return names


class _HeadCopy(nn.Module):
    """The modules, parameters and buffers of one extra head.

    A deep copy of each head attribute of the trunk model, registered under
    the same name so the ``state_dict`` reads ``extra_heads.<i>.<name>...``.
    """

    def __init__(self, model: nn.Module, names: Sequence[str]):
        super().__init__()
        self.names = list(names)
        self.kinds: dict[str, str] = {}
        for name in self.names:
            attr = getattr(model, name)
            if isinstance(attr, nn.Module):
                self.add_module(name, copy.deepcopy(attr))
                self.kinds[name] = "module"
            elif isinstance(attr, nn.Parameter):
                self.register_parameter(name, nn.Parameter(attr.detach().clone(),
                                                           requires_grad=attr.requires_grad))
                self.kinds[name] = "parameter"
            elif torch.is_tensor(attr):
                self.register_buffer(name, attr.detach().clone())
                self.kinds[name] = "buffer"
            else:
                raise TypeError(f"head module {name!r} is a {type(attr).__name__}, not a "
                                "module, parameter or buffer")

    def overrides(self) -> dict[str, Tensor]:
        """Tensors of this head keyed by their name inside the trunk model."""
        out: dict[str, Tensor] = {}
        for name in self.names:
            attr = getattr(self, name)
            if isinstance(attr, nn.Module):
                for key, value in attr.named_parameters():
                    out[f"{name}.{key}"] = value
                for key, value in attr.named_buffers():
                    out[f"{name}.{key}"] = value
            else:
                out[name] = attr
        return out


def _swap(model: nn.Module, source: nn.Module, names: Iterable[str]) -> dict[str, Any]:
    """Move the head attributes of ``source`` onto ``model``; return the old ones."""
    previous: dict[str, Any] = {}
    for name in names:
        previous[name] = getattr(model, name)
        attr = getattr(source, name)
        if isinstance(attr, nn.Module):
            model._modules[name] = attr
        elif isinstance(attr, nn.Parameter):
            model._parameters[name] = attr
        else:
            model._buffers[name] = attr
    return previous


class MultiHead(InteratomicPotential):
    """A potential with several readout heads over one shared trunk.

    Head 0 is the wrapped model's own readout (and reference energies); the
    other heads are copies of the modules the model names in its
    ``head_modules``. Every head sees the same trunk, so a forward pass on a
    batch mixing heads runs the trunk once per head present (on that head's
    structures) with the head's parameters substituted, and nothing is
    computed twice.

    Parameters
    ----------
    model : InteratomicPotential
        The potential to give several heads. Its current readout becomes
        head 0.
    heads : sequence of str
        The head names, in index order (``["pt_head", "Default"]`` for the
        MACE replay convention: the pretrained readout serves the replay data,
        the copy adapts to the target data).

    Attributes
    ----------
    model : InteratomicPotential
        The trunk, carrying head 0.
    heads : list of str
        The head names.
    extra_heads : torch.nn.ModuleList
        One :class:`_HeadCopy` per head after the first.
    cutoff : float
        The trunk's radial cutoff.

    Raises
    ------
    TypeError
        If the model declares no ``head_modules``.
    ValueError
        For an empty or repeated head list.
    """

    def __init__(self, model: InteratomicPotential, heads: Sequence[str]):
        super().__init__()
        heads = [str(h) for h in heads]
        if not heads or len(set(heads)) != len(heads):
            raise ValueError(f"heads must be distinct names, got {heads}")
        names = head_module_names(model)
        self.model = model
        self.heads = heads
        self.head_module_names = names
        self.cutoff = getattr(model, "cutoff", None)
        self.node_feature_dim = getattr(model, "node_feature_dim", None)
        self.extra_heads = nn.ModuleList([_HeadCopy(model, names) for _ in heads[1:]])

    @property
    def species(self):
        """The trunk's supported atomic numbers, if it lists them."""
        return getattr(self.model, "species", None)

    def index(self, head: Union[int, str]) -> int:
        """The index of a head given by name or index.

        Raises
        ------
        KeyError
            For an unknown head name or an index out of range.
        """
        if isinstance(head, str):
            if head not in self.heads:
                raise KeyError(f"unknown head {head!r}; this model has {self.heads}")
            return self.heads.index(head)
        idx = int(head)
        if not 0 <= idx < len(self.heads):
            raise KeyError(f"head index {idx} out of range for {self.heads}")
        return idx

    def _run(self, head: int, data: AtomicGraph) -> dict[str, Tensor]:
        """Evaluate the trunk with the parameters of one head."""
        if head == 0:
            return self.model(data)
        overrides = self.extra_heads[head - 1].overrides()
        return torch.func.functional_call(self.model, overrides, (data,))

    def forward(self, data: AtomicGraph) -> dict[str, Tensor]:
        """Evaluate every structure with the head it is labelled with.

        Parameters
        ----------
        data : AtomicGraph
            The batch; ``data.head`` holds the head index per structure
            (``None`` means head 0 throughout).

        Returns
        -------
        dict of str to torch.Tensor
            The trunk's outputs (``node_energy``, ``energy``,
            ``node_features``, ...) in the batch's order.
        """
        if data.head is None or len(self.heads) == 1:
            return self._run(0, data)
        present = [int(h) for h in torch.unique(data.head).tolist()]
        for h in present:
            if h >= len(self.heads):
                raise IndexError(f"a structure asks for head {h}; this model has {self.heads}")
        if len(present) == 1:
            return self._run(present[0], data)

        node_head = data.head[data.batch]
        outs, graph_idx, node_idx = [], [], []
        for h in present:
            keep = data.head == h
            outs.append(self._run(h, data.subset(keep)))
            graph_idx.append(keep.nonzero().squeeze(-1))
            node_idx.append((node_head == h).nonzero().squeeze(-1))
        graph_order = torch.cat(graph_idx)
        node_order = torch.cat(node_idx)
        inv_graph = torch.empty_like(graph_order)
        inv_graph[graph_order] = torch.arange(graph_order.numel(), device=graph_order.device)
        inv_node = torch.empty_like(node_order)
        inv_node[node_order] = torch.arange(node_order.numel(), device=node_order.device)

        n_nodes, n_graphs = data.num_nodes, data.num_graphs
        merged: dict[str, Tensor] = {}
        for key in outs[0]:
            values = [o[key] for o in outs if key in o]
            if len(values) != len(outs) or not all(torch.is_tensor(v) for v in values):
                continue
            if values[0].dim() == 0:
                merged[key] = torch.stack(values).sum()        # a scalar regularizer
                continue
            cat = torch.cat(values, 0)
            if cat.shape[0] == n_nodes and (n_nodes != n_graphs or key not in _STRUCTURE_KEYS):
                merged[key] = cat[inv_node]
            elif cat.shape[0] == n_graphs:
                merged[key] = cat[inv_graph]
            else:
                merged[key] = cat
        return merged

    @contextlib.contextmanager
    def using(self, head: Union[int, str]):
        """Make the trunk carry the modules of ``head`` for the duration of a block.

        Lets any single-head routine (reference-energy estimation,
        pseudolabelling, a deployment export) act on one head through
        :attr:`model`; the trunk's own head is restored afterwards.

        Parameters
        ----------
        head : int or str
            The head to swap in.
        """
        idx = self.index(head)
        if idx == 0:
            yield self.model
            return
        previous = _swap(self.model, self.extra_heads[idx - 1], self.head_module_names)
        try:
            yield self.model
        finally:
            _swap(self.model, _Holder(previous), self.head_module_names)

    def select(self, head: Union[int, str]) -> InteratomicPotential:
        """A standalone single-head potential: the trunk with one head's modules.

        Parameters
        ----------
        head : int or str
            The head to keep.

        Returns
        -------
        InteratomicPotential
            A deep copy of the trunk carrying that head; the same class as the
            wrapped model, so every deployment path (TorchScript export, ASE
            calculator, MDI) applies unchanged.
        """
        with self.using(head) as model:
            return copy.deepcopy(model)

    def set_atomic_energies(self, head: Union[int, str], values) -> None:
        """Set the per-element reference energies of one head.

        Parameters
        ----------
        head : int or str
            The head.
        values : mapping or sequence
            ``{Z: E0}`` (atomic numbers or symbols as keys) or one value per
            entry of the trunk's ``species``; see
            :func:`~xnn.common.finetune.set_atomic_energies`.
        """
        from .reference import set_atomic_energies
        with self.using(head) as model:
            set_atomic_energies(model, values)

    def atomic_energies(self, head: Union[int, str]) -> dict[int, float]:
        """The per-element reference energies of one head, ``{Z: E0}``."""
        from .reference import get_atomic_energies
        with self.using(head) as model:
            return get_atomic_energies(model)

    def label(self, structures, head: Union[int, str]):
        """Tag structures with one of this model's heads (see :func:`label_head`)."""
        return label_head(structures, self.index(head))

    def _apply(self, fn, recurse=True):
        out = super()._apply(fn, recurse)
        # a model that keeps part of its head in a fixed dtype (AIMNet2's float64
        # reference energies) keeps the copies in that dtype too
        base = dict(self.model.named_parameters())
        base.update(dict(self.model.named_buffers()))
        for head in self.extra_heads:
            for name, tensor in head.overrides().items():
                ref = base.get(name)
                if ref is not None and ref.is_floating_point() and tensor.dtype != ref.dtype:
                    tensor.data = tensor.data.to(ref.dtype)
        return out

    @torch.jit.unused
    def set_use_fast(self, use_fast) -> "MultiHead":
        """Forward the fast-path setting to the trunk (see :func:`xnn.common.models.fast.set_use_fast`)."""
        if callable(getattr(self.model, "set_use_fast", None)):
            self.model.set_use_fast(use_fast)
        return self

    @classmethod
    def from_config(cls, cfg) -> "MultiHead":
        """Build through :func:`~xnn.common.models.registry.build_model` (``extra["heads"]``)."""
        from ..models.registry import build_model
        return build_model(cfg)


class _Holder:
    """Attribute bag so :func:`_swap` can restore a saved set of head attributes."""

    def __init__(self, attrs: Mapping[str, Any]):
        self.__dict__.update(attrs)


def head_names(spec: Any) -> list[str]:
    """The head names of a ``model.extra["heads"]`` entry (a list or a dict)."""
    if isinstance(spec, Mapping):
        return [str(k) for k in spec]
    if isinstance(spec, (list, tuple)):
        return [str(h) for h in spec]
    if isinstance(spec, str):
        return [s.strip() for s in spec.split(",") if s.strip()]
    raise TypeError(f"heads must be a list of names or a mapping, got {spec!r}")


def head_options(spec: Any, head: str) -> dict[str, Any]:
    """The per-head options of ``head`` in a ``model.extra["heads"]`` mapping (``{}`` for a list)."""
    if isinstance(spec, Mapping):
        return dict(spec.get(head) or {})
    return {}


def label_head(structures, head: int):
    """Tag structure dicts (or a dataset of them) with a readout head.

    Sets ``s["head"] = head`` on every structure, which
    :func:`~xnn.common.data.dataset.structure_to_graph` turns into the
    graph's ``head`` field. An :class:`~xnn.common.data.AtomicDataset` has its
    graph cache cleared so the label reaches graphs already built.

    Parameters
    ----------
    structures : iterable of dict or AtomicDataset
        The structures.
    head : int
        The head index.

    Returns
    -------
    The same object, labelled in place.
    """
    from ..data import AtomicDataset
    if isinstance(structures, AtomicDataset):
        for s in structures.structures:
            s["head"] = int(head)
        structures._cache.clear()
        return structures
    if isinstance(structures, Mapping):
        structures["head"] = int(head)
        return structures
    for s in structures:
        s["head"] = int(head)
    return structures


def find_multihead(model: nn.Module) -> Optional[MultiHead]:
    """The :class:`MultiHead` inside a (possibly wrapped) model, if any."""
    for m in model.modules():
        if isinstance(m, MultiHead):
            return m
    return None
