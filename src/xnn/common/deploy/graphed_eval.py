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

The replay runs the same code as an ordinary evaluation, so the results agree with it to round-off.
On large structures the launches are a small part of an evaluation and the padding costs more than
they do: the first input is evaluated ordinarily and timed, an evaluation slower than ``max_ms`` is
not captured at all, and after each capture a few replays are timed against ordinary evaluations,
and the faster is kept (the graph and its memory are released when the ordinary evaluation wins).
"""
from __future__ import annotations

import copy
import dataclasses
import math
import time
from typing import Dict, List, Optional

import torch
from torch import Tensor, nn

from ..data import AtomicGraph, collate
from ..models.fast import capture_preparation
from ..train.graphed import Capacity, _inputs, index_planner, padding_structure

#: Timed ordinary evaluations of the first input, at most: the capture goes ahead at the first one no
#: slower than ``max_ms``; when all are slower the evaluations stay ordinary (the first calls of a new
#: model compile and are slow, at times the third again).
PROBES = 4


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
        Ordinary evaluations before the capture, by default 2. They and the capture run with
        TorchScript fusion off, so that no TorchScript block compiles a kernel while it is recorded.
    trial : int, optional
        Replays and ordinary evaluations timed against each other after a capture, by default
        3; the faster is used until the input outgrows the counts. ``0`` always replays.
    max_ms : float or None, optional
        Wall time in milliseconds of an ordinary evaluation of the first input above which nothing
        is captured and evaluations stay ordinary, by default 100: the launches are a small part
        of such an evaluation, and a graph's memory would come on top of its own. The first calls
        are ordinary evaluations, timed, until one takes at most ``max_ms`` or ``PROBES`` of them
        took longer. ``None`` captures whatever the time, from the first call.

    Raises
    ------
    ValueError
        If the model has no index planner.
    """

    def __init__(self, model: nn.Module, margin: float = 1.05, slack: int = 64, warmup: int = 2,
                 trial: int = 3, max_ms: Optional[float] = 100.0):
        self.model = model
        self.trial = int(trial)
        self.max_ms = None if max_ms is None else float(max_ms)
        self.planner = index_planner(model)
        if self.planner is None:
            raise ValueError("a captured evaluation needs a model with index_planner() "
                             "(xnn.common.train.graphed)")
        self.margin = float(margin)
        self.slack = int(slack)
        self.warmup = int(warmup)
        self.cap: Optional[Capacity] = None
        self.n_graphs = 0
        self.probe_ms: List[float] = []
        # whether the timed ordinary evaluations have let the capture go ahead
        self.probed = self.max_ms is None
        self._reset()

    def _reset(self) -> None:
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.static: Optional[AtomicGraph] = None
        self.inputs: Dict[str, Tensor] = {}
        self.out: Optional[Dict[str, Tensor]] = None
        self.steps = 0
        # "replay" or "eager" once the trial after a capture has run
        self.mode: Optional[str] = None

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

    def _ordinary(self, graph: AtomicGraph) -> Dict[str, Tensor]:
        """An ordinary evaluation of ``graph`` (no padding, no graph)."""
        out = self.model(graph)
        return {k: v.detach() for k, v in out.items() if isinstance(v, Tensor)}

    def _choose(self, graph: AtomicGraph) -> str:
        """Time ``trial`` replays against as many ordinary evaluations of ``graph``."""
        def timed(run):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(self.trial):
                run(graph)
            torch.cuda.synchronize()
            return time.perf_counter() - t0
        t_replay = timed(self._replayed)
        try:
            # untimed first: the first ordinary evaluation after the capture can be slower (TorchScript
            # blocks run fused again)
            self._ordinary(graph)
            t_ordinary = timed(self._ordinary)
        except torch.OutOfMemoryError:
            # an ordinary evaluation does not fit beside the graph's memory: the input is large, and
            # without the graph it fits (the warmup evaluations ran)
            return "eager"
        return "replay" if t_replay <= t_ordinary else "eager"

    def _release(self) -> None:
        """Drop the graph and its static inputs (and with them its memory pool)."""
        self.graph, self.static, self.inputs, self.out = None, None, {}, None
        torch.cuda.empty_cache()

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
        if self.mode == "eager":
            return self._ordinary(graph)
        if not self.probed:
            return self._probe(graph)
        try:
            out = self._replayed(graph)
        except torch.OutOfMemoryError:
            out = None
        if out is None:
            # the graph's memory pool comes on top of an ordinary evaluation's: evaluate ordinarily,
            # outside the handler, whose traceback holds the failed evaluation's tensors
            self._release()
            self.mode = "eager"
            return self._ordinary(graph)
        if self.graph is not None and self.mode is None:
            self.mode = "replay" if self.trial <= 0 else self._choose(graph)
            if self.mode == "eager":
                self._release()
        return out

    def _probe(self, graph: AtomicGraph) -> Dict[str, Tensor]:
        """An ordinary evaluation of ``graph``, timed: the capture goes ahead once one takes at most
        ``max_ms``, and evaluations stay ordinary when ``PROBES`` of them took longer."""
        torch.cuda.synchronize(graph.pos.device)
        t0 = time.perf_counter()
        out = self._ordinary(graph)
        torch.cuda.synchronize(graph.pos.device)
        self.probe_ms.append((time.perf_counter() - t0) * 1e3)
        if self.probe_ms[-1] <= self.max_ms:
            self.probed = True
        elif len(self.probe_ms) >= PROBES:
            # launches are a small part of so long an evaluation: no capture
            self._release()
            self.mode = "eager"
        return out

    def _replayed(self, graph: AtomicGraph) -> Dict[str, Tensor]:
        """The evaluation of ``graph`` padded, through the captured graph once it is recorded."""
        n, b = graph.num_nodes, graph.num_graphs
        self._load(self._pad(graph))
        device = self.static.pos.device
        if self.graph is not None:
            self.graph.replay()
        elif self.steps < self.warmup:
            side = torch.cuda.Stream(device)
            side.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(side), torch.jit.fuser("none"), capture_preparation():
                self.out = self.model(self.static)
            torch.cuda.current_stream(device).wait_stream(side)
        else:
            self.graph = torch.cuda.CUDAGraph()
            # TorchScript blocks (e3nn's spherical harmonics) compile fused kernels after their
            # profiling runs, and a capture cannot load a kernel: they run unfused here
            with torch.cuda.graph(self.graph), torch.jit.fuser("none"), capture_preparation():
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
