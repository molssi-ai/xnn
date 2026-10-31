"""Training steps replayed as one CUDA graph (``trainer.cuda_graph``).

A training step on small molecules is bound by kernel launches: the model,
the force derivative and the parameter gradients of a force loss take
thousands of small kernels, each launched from the host. A captured step
replays all of them as one graph. Capture needs fixed shapes and no host
synchronization inside the step:

* every batch is padded to fixed counts of atoms, edges and model indices by
  one inert structure (:class:`PaddedCollate`): its labels are masked out, so
  it adds exactly nothing to the loss and the gradients;
* the index work whose sizes depend on the data (the triplets of DimeNet,
  the embedding edges, triplets and quadruplets of GemNet) runs before the
  step, in the data-loader workers, through the model's
  ``index_planner()``, and the batch carries the result as ``index_plan``;
* the loss runs without reading values back (``weighted_loss(sync=False)``),
  the optimizer is capturable, and the logged values are read after the
  replay.

The first batches run as ordinary steps (they warm up the libraries'
workspaces), the next is captured and every later one replayed.
"""
from __future__ import annotations

import copy
import dataclasses
from typing import Callable, Dict, List, Optional

import torch
from torch import Tensor

from ..data import AtomicGraph, collate

#: The atomic number of the padding atoms.
PADDING_Z = 1

#: The spacing of the padding atoms along x (the two carrying the padding edges sit
#: within every cutoff; no edge links them to a real atom).
PADDING_SPACING = 1.0

#: Ordinary steps before the capture.
WARMUP_STEPS = 3


def index_planner(model: torch.nn.Module) -> Optional[Callable]:
    """The index planner of the first submodule of ``model`` that has one, or ``None``."""
    for module in model.modules():
        make = getattr(module, "index_planner", None)
        if callable(make):
            return make()
    return None


@dataclasses.dataclass
class Capacity:
    """The fixed counts of a padded batch.

    Attributes
    ----------
    atoms, edges : int
        Atoms and edges, real and padding.
    plan : dict of str to int
        The sizes of the index plan (e.g. ``{"triplets": ...}``).
    """

    atoms: int
    edges: int
    plan: Dict[str, int]


class PlannedDataset:
    """A dataset whose structures carry their index plans, built once.

    A structure's plan does not change during training, so it is built on the
    first access (the capacity pass touches every structure) and kept as
    32-bit integers; a batch then only concatenates the plans of its
    structures (:func:`combine`). Items are the dataset's graphs with the plan
    attached as ``index_plan``.

    Parameters
    ----------
    dataset : sequence of AtomicGraph
        The training structures.
    planner : callable
        The model's index planner.
    """

    def __init__(self, dataset, planner: Callable):
        self.dataset = dataset
        self.planner = planner
        self.plans: Dict[int, Dict[str, Tensor]] = {}

    def __len__(self) -> int:
        return len(self.dataset)

    def plan(self, i: int, graph: Optional[AtomicGraph] = None) -> Dict[str, Tensor]:
        """The index plan of structure ``i``."""
        if i not in self.plans:
            graph = self.dataset[i] if graph is None else graph
            self.plans[i] = {k: v.to(torch.int32) for k, v in self.planner(graph).items()}
        return self.plans[i]

    def __getitem__(self, i: int) -> AtomicGraph:
        graph = copy.copy(self.dataset[i])        # the dataset's own graph stays as it is
        graph.index_plan = self.plan(i, graph)
        return graph


def combine(planner: Callable, graphs: List[AtomicGraph]) -> Dict[str, Tensor]:
    """The index plan of a batch from the plans of its structures (``index_plan`` of each).

    Every plan entry indexes one count of its structure (``planner.offsets``:
    ``"edges"`` for the edges, a plan size such as ``"emb_edges"``, or
    ``None`` for a slot that needs no offset), so the plans concatenate with
    the running counts added.
    """
    plans = [g.index_plan for g in graphs]
    running: Dict[str, int] = {}
    out: Dict[str, List[Tensor]] = {k: [] for k in plans[0]}
    for g, plan in zip(graphs, plans):
        counts = dict(planner.sizes(plan), edges=int(g.edge_index.shape[1]))
        for k, v in plan.items():
            base = planner.offsets[k]
            out[k].append(v.long() + running.get(base, 0) if base is not None else v.long())
        for base, n in counts.items():
            running[base] = running.get(base, 0) + n
    return {k: torch.cat(v) for k, v in out.items()}


def capacity(planner: Callable, dataset: PlannedDataset, batch_size: int) -> Capacity:
    """Counts that fit every batch of ``batch_size`` structures of ``dataset``.

    Each count is the sum over the ``batch_size`` largest structures, plus
    room for the padding structure (two atoms, two edges as one reverse
    pair, one plan entry); the plan sizes the planner lists in ``maxima``
    (widths per edge) are the largest over the structures. Builds the index
    plan of every structure once (kept by ``dataset``).

    Parameters
    ----------
    planner : callable
        The model's index planner.
    dataset : PlannedDataset
        The training structures.
    batch_size : int
        Structures per batch.

    Returns
    -------
    Capacity
    """
    atoms, edges, plans = [], [], []
    for i in range(len(dataset)):
        g = dataset.dataset[i]
        atoms.append(g.num_nodes)
        edges.append(int(g.edge_index.shape[1]))
        plans.append(planner.sizes(dataset.plan(i, g)))

    def top(values: List[int]) -> int:
        return sum(sorted(values, reverse=True)[:batch_size])
    maxima = getattr(planner, "maxima", ())
    plan = {k: max(p[k] for p in plans) if k in maxima else top([p[k] for p in plans]) + 1
            for k in (plans[0] if plans else {})}
    # full neighbor lists hold every edge with its reverse: the padding edges, too, come in pairs
    return Capacity(atoms=top(atoms) + 2, edges=top(edges) + 2, plan=plan)


def padding_structure(n_atoms: int, n_edges: int, like: AtomicGraph) -> AtomicGraph:
    """The inert structure that fills a batch: ``n_atoms >= 2`` atoms on a line,
    ``n_edges`` edges between the first two (alternating in direction, so every edge has
    its reverse: ``n_edges`` is even), no labels."""
    if n_edges % 2:
        raise ValueError(f"the padding edges come in reverse pairs, got {n_edges}")
    pos = torch.zeros((n_atoms, 3), dtype=like.pos.dtype)
    pos[:, 0] = PADDING_SPACING * torch.arange(n_atoms, dtype=like.pos.dtype)
    src = torch.arange(n_edges) % 2
    return AtomicGraph(
        pos=pos,
        atomic_numbers=torch.full((n_atoms,), PADDING_Z, dtype=like.atomic_numbers.dtype),
        edge_index=torch.stack([src, 1 - src]).to(like.edge_index.dtype),
        cell_shifts=torch.zeros((n_edges, 3), dtype=like.cell_shifts.dtype),
        batch=torch.zeros(n_atoms, dtype=like.batch.dtype),
        n_atoms=torch.tensor([n_atoms], dtype=like.n_atoms.dtype),
        compute_dtype=like.compute_dtype,
    )


class PaddedCollate:
    """Collate structures into a batch of fixed counts with its index plan.

    The real structures come first, as in :func:`~xnn.common.data.collate`,
    followed by one padding structure (:func:`padding_structure`); the plan
    of the batch (from the structures' own plans, :class:`PlannedDataset`) is
    padded and attached as ``index_plan``, its fixed sizes (Python
    integers, so that the model reads no value from the device) as
    ``index_sizes``, and the real counts as ``real_atoms``, ``real_edges``
    and ``real_graphs`` (:func:`strip_padding`). Runs in the data-loader
    workers.

    Parameters
    ----------
    planner : callable
        The model's index planner.
    cap : Capacity
        The fixed counts.
    """

    def __init__(self, planner: Callable, cap: Capacity):
        self.planner = planner
        self.cap = cap
        #: Off: plain batches (training went on with ordinary steps).
        self.padded = True

    def __call__(self, graphs: List[AtomicGraph]) -> AtomicGraph:
        real = collate(graphs)
        if not self.padded:
            return real
        # the structures' own plans (PlannedDataset), else the batch planned whole
        plan = (combine(self.planner, graphs) if all(hasattr(g, "index_plan") for g in graphs)
                else self.planner(real))
        n, e = real.num_nodes, int(real.edge_index.shape[1])
        sizes = self.planner.sizes(plan)
        maxima = getattr(self.planner, "maxima", ())
        if (n + 2 > self.cap.atoms or e + 2 > self.cap.edges
                or any(sizes[k] + (k not in maxima) > self.cap.plan[k] for k in sizes)):
            raise RuntimeError(f"a batch of {n} atoms, {e} edges and {sizes} does not fit the "
                               f"captured step's counts {self.cap}")
        batch = collate(list(graphs) + [padding_structure(self.cap.atoms - n, self.cap.edges - e, real)])
        batch.index_plan = self.planner.pad(plan, self.cap.plan, e)
        batch.index_sizes = dict(self.cap.plan)
        batch.real_atoms, batch.real_edges, batch.real_graphs = n, e, len(graphs)
        return batch


#: The per-atom fields of a graph (the others are per edge: edge_index, cell_shifts; or per structure).
_ATOM_FIELDS = ("pos", "atomic_numbers", "batch", "forces", "charges", "fragment_charges", "atom_force_weight")


def strip_padding(batch: AtomicGraph) -> AtomicGraph:
    """The real structures of a padded batch (:class:`PaddedCollate`), without the padding
    structure and the index plan: what an ordinary step takes."""
    if not hasattr(batch, "real_atoms"):
        return batch
    n, e, b = batch.real_atoms, batch.real_edges, batch.real_graphs
    kwargs = {}
    for f in dataclasses.fields(batch):
        v = getattr(batch, f.name)
        if not isinstance(v, Tensor):
            kwargs[f.name] = v
        elif f.name == "edge_index":
            kwargs[f.name] = v[:, :e]
        elif f.name == "cell_shifts":
            kwargs[f.name] = v[:e]
        elif f.name in _ATOM_FIELDS:
            kwargs[f.name] = v[:n]
        else:
            kwargs[f.name] = v[:b]
    return AtomicGraph(**kwargs)


def _inputs(batch: AtomicGraph) -> Dict[str, Tensor]:
    """The tensors of a padded batch, by name (the plan's under ``plan/<key>``)."""
    out = {f.name: getattr(batch, f.name) for f in dataclasses.fields(batch)
           if isinstance(getattr(batch, f.name), Tensor)}
    out.update({f"plan/{k}": v for k, v in batch.index_plan.items()})
    return out


class GraphedStep:
    """A training step on padded batches, captured once and replayed.

    Parameters
    ----------
    body : callable
        ``body(batch) -> (pred, logs, counts)``: the whole training step on a
        padded batch (forward, ``weighted_loss(sync=False)``, backward,
        gradient clipping, the optimizer and the weight average), with tensors
        in ``pred``, ``logs`` and ``counts``.
    device : torch.device
        The training device.
    warmup : int, optional
        Ordinary steps before the capture. Defaults to :data:`WARMUP_STEPS`.
    """

    def __init__(self, body: Callable, device: torch.device, warmup: int = WARMUP_STEPS):
        self.body = body
        self.device = device
        self.warmup = warmup
        self.steps = 0
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.static: Optional[AtomicGraph] = None
        self.inputs: Dict[str, Tensor] = {}
        self.out = None

    def _load(self, batch: AtomicGraph) -> None:
        """Copy ``batch`` into the static batch (allocated from the first one)."""
        if self.static is None:
            self.static = batch.to(self.device)
            self.static.index_plan = {k: v.to(self.device) for k, v in batch.index_plan.items()}
            self.static.index_sizes = batch.index_sizes
            self.inputs = _inputs(self.static)
            return
        new = _inputs(batch)
        if new.keys() != self.inputs.keys():
            raise RuntimeError("a captured training step needs the same labels in every batch: "
                               f"{sorted(new.keys() ^ self.inputs.keys())} differ")
        with torch.no_grad():
            for k, t in new.items():
                self.inputs[k].copy_(t, non_blocking=True)
        # the step may rebind attributes (a strained copy of the positions): back to the inputs
        for k, t in self.inputs.items():
            if not k.startswith("plan/"):
                setattr(self.static, k, t)

    def __call__(self, batch: AtomicGraph):
        """Run one step on ``batch``; returns ``(pred, logs, counts, static_batch)`` with the
        tensors of the step (read them before the next call)."""
        self._load(batch)
        if self.graph is not None:
            self.graph.replay()
        elif self.steps < self.warmup:
            # the ordinary steps run on a side stream, as a capture needs
            side = torch.cuda.Stream(self.device)
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side):
                self.out = self.body(self.static)
            torch.cuda.current_stream(self.device).wait_stream(side)
        else:
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = self.body(self.static)
            # recording does not run the step: replay it for this batch
            self.graph.replay()
        self.steps += 1
        pred, logs, counts = self.out
        return pred, logs, counts, self.static
