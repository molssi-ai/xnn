"""Evaluations of a model replayed as one CUDA graph (molecular dynamics, optimization, screening).

On small structures an evaluation (the energy and its derivatives) is bound by kernel launches:
the model and the force derivative take a thousand small kernels, each launched from the host. A
captured evaluation records them once and replays them. Capture needs fixed shapes and no host
synchronization inside the evaluation, as the captured training step of
:mod:`xnn.common.train.graphed`, whose pieces this reuses:

* every input is padded to fixed counts of atoms, edges and model indices by one inert structure
  (:func:`~xnn.common.train.graphed.padding_structure`), whose results are dropped;
* the index work whose sizes depend on the input runs before the replay, through the model's
  ``index_planner()``;
* the counts carry a margin, and an input that outgrows them (or has another number of structures)
  is captured again; a model bound to one system (its planner's ``pads`` false) is replayed on the
  batch as it comes, and captured again when the counts change.

The replay runs the same reference code as an ordinary evaluation, so the results agree with it to
round-off.
"""
from __future__ import annotations

import copy
import dataclasses
import math
from typing import Dict, Optional

import torch
from torch import Tensor, nn

from ..data import AtomicGraph, collate
from ..train.graphed import Capacity, _inputs, index_planner, padding_structure


class GraphedEvaluation:
    """A model's evaluations on padded graphs, captured once per size class and replayed.

    Parameters
    ----------
    model : torch.nn.Module
        The evaluation, ``model(graph) -> dict`` (for example a
        :class:`~xnn.common.models.ForceStressOutput`), on a CUDA device; it or one of its
        submodules must provide ``index_planner()``.
    margin : float, optional
        Relative headroom of the atom, edge and plan counts over the first input (and over an
        input that outgrows them), by default 1.05: in molecular dynamics the edge count drifts by
        a few percent, and every padding edge costs as much as a real one.
    slack : int, optional
        Absolute headroom added to the counts, by default 64.
    warmup : int, optional
        Ordinary evaluations before the capture, by default 2.

    Raises
    ------
    ValueError
        If the model has no index planner.
    """

    def __init__(self, model: nn.Module, margin: float = 1.05, slack: int = 64, warmup: int = 2):
        self.model = model
        self.planner = index_planner(model)
        if self.planner is None:
            raise ValueError("a captured evaluation needs a model with index_planner() "
                             "(xnn.common.train.graphed)")
        self.margin = float(margin)
        self.slack = int(slack)
        self.warmup = int(warmup)
        self.cap: Optional[Capacity] = None
        self.n_graphs = 0
        self._reset()

    def _reset(self) -> None:
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.static: Optional[AtomicGraph] = None
        self.inputs: Dict[str, Tensor] = {}
        self.out: Optional[Dict[str, Tensor]] = None
        self.steps = 0

    def _fits(self, n: int, e: int, b: int, sizes: Dict[str, int]) -> bool:
        cap = self.cap
        maxima = getattr(self.planner, "maxima", ())
        return (cap is not None and b == self.n_graphs and n + 2 <= cap.atoms and e + 2 <= cap.edges
                and all(sizes[k] + (k not in maxima) <= cap.plan[k] for k in sizes))

    def _grow(self, n: int, e: int, b: int, sizes: Dict[str, int]) -> None:
        """Counts with the margin over ``n`` atoms, ``e`` edges and the plan ``sizes`` for ``b`` structures
        (the padding edges come in reverse pairs: an even count)."""
        self.n_graphs = b
        edges = int(math.ceil(e * self.margin)) + self.slack
        self.cap = Capacity(atoms=int(math.ceil(n * self.margin)) + self.slack + 2, edges=edges + edges % 2 + 2,
                            plan={k: int(math.ceil(v * self.margin)) + self.slack for k, v in sizes.items()})
        self._reset()

    def _pad(self, graph: AtomicGraph) -> AtomicGraph:
        # the input's own tensors may carry autograd history (positions that an earlier
        # evaluation marked for a force derivative): the captured evaluation starts from
        # detached copies, or its backward would reach into tensors outside the capture
        graph = copy.copy(graph)
        for f in dataclasses.fields(graph):
            v = getattr(graph, f.name)
            if isinstance(v, Tensor) and v.requires_grad:
                setattr(graph, f.name, v.detach())
        n, e = graph.num_nodes, int(graph.edge_index.shape[1])
        if not getattr(self.planner, "pads", True):
            # a model that takes no padding structure: the batch is replayed as it comes, and recorded
            # again when its counts change
            if self.cap is None or (self.cap.atoms, self.cap.edges, self.n_graphs) != (n, e, graph.num_graphs):
                self.n_graphs = graph.num_graphs
                self.cap = Capacity(atoms=n, edges=e, plan={})
                self._reset()
            graph.index_plan, graph.index_sizes = {}, {}
            return graph
        # planners work on CPU graphs (in training, the data-loader workers); one that plans nothing
        # (NoIndexPlanner) needs no copy
        plan = self.planner(graph.to("cpu") if self.planner.offsets else graph)
        sizes = self.planner.sizes(plan)
        b = graph.num_graphs
        if not self._fits(n, e, b, sizes):
            self._grow(n, e, b, sizes)
        pad = padding_structure(self.cap.atoms - n, self.cap.edges - e, graph).to(graph.pos.device)
        batch = collate([graph, pad])
        batch.compute_dtype = graph.compute_dtype
        batch.index_plan = {k: v.to(graph.pos.device) for k, v in self.planner.pad(plan, self.cap.plan, e).items()}
        batch.index_sizes = dict(self.cap.plan)
        return batch

    def _load(self, batch: AtomicGraph) -> None:
        if self.static is None:
            self.static = batch
            self.inputs = _inputs(batch)
            return
        new = _inputs(batch)
        if new.keys() != self.inputs.keys():
            raise RuntimeError("a captured evaluation needs the same fields in every input: "
                               f"{sorted(new.keys() ^ self.inputs.keys())} differ")
        with torch.no_grad():
            for k, t in new.items():
                self.inputs[k].copy_(t, non_blocking=True)
        for f in dataclasses.fields(self.static):
            t = self.inputs.get(f.name)
            if t is not None:
                setattr(self.static, f.name, t)

    def __call__(self, graph: AtomicGraph) -> Dict[str, Tensor]:
        """The model's outputs for ``graph`` (one structure or a batch), without the padding.

        Parameters
        ----------
        graph : AtomicGraph
            The input, on the model's CUDA device.

        Returns
        -------
        dict of str to Tensor
            The outputs of ``model(graph)``, per atom for the real atoms and per structure for the
            real structures (copies: the next call overwrites the captured ones).
        """
        n, b = graph.num_nodes, graph.num_graphs
        self._load(self._pad(graph))
        device = self.static.pos.device
        if self.graph is not None:
            self.graph.replay()
        elif self.steps < self.warmup:
            side = torch.cuda.Stream(device)
            side.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(side):
                self.out = self.model(self.static)
            torch.cuda.current_stream(device).wait_stream(side)
        else:
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = self.model(self.static)
            # recording does not run the evaluation: replay it for this input
            self.graph.replay()
        self.steps += 1
        # per-atom outputs have the padded atom count, per-structure ones one structure more
        n_pad, b_pad = self.static.num_nodes, self.static.num_graphs
        out = {}
        for k, v in self.out.items():
            if isinstance(v, Tensor) and v.dim() > 0 and v.shape[0] in (n_pad, b_pad):
                out[k] = (v[:n] if v.shape[0] == n_pad else v[:b]).detach().clone()
        return out
