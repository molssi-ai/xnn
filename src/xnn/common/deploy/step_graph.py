"""Molecular dynamics steps on a step plan, replayed as one CUDA graph.

A model whose core records step plans (AMOEBA and HIPPO, see
:meth:`~xnn.ffnn.models.Amoeba.plan_step`) evaluates a moving structure
without deciding anything on the host: the plan holds the decisions, a
Verlet pair list and solves of fixed length. :class:`StepGraph` keeps such a
plan over a run and, on a GPU, records the whole step (energy, forces and
stress) once and replays it.
"""
from __future__ import annotations

import collections
import math
from typing import Optional

import torch
from torch import Tensor, nn

from ..data import structure_to_graph


def plan_core(model: nn.Module) -> Optional[nn.Module]:
    """The submodule of ``model`` that records step plans, or ``None``.

    Parameters
    ----------
    model : torch.nn.Module
        A model, possibly wrapped.

    Returns
    -------
    torch.nn.Module or None
        The first submodule with a ``plan_step`` method.
    """
    for module in model.modules():
        if callable(getattr(module, "plan_step", None)):
            return module
    return None


class StepGraph:
    """The steps of a molecular dynamics run, evaluated on a step plan.

    Each call takes the structure of a step and returns the energy, the
    forces and, for a cell, the stress. The pair list is built with
    ``cutoff + skin`` and rebuilt when an atom has moved by more than half the
    skin. The plan holds ``margin`` times the pairs of its first list, so a
    rebuild refreshes it in place. The first ``warmup`` steps run with
    converged solves; then the solves run the largest count of the last two
    steps plus a quarter, and on a GPU the step is recorded as a CUDA graph
    and replayed from then on. A step whose solves did not converge in that
    length is evaluated again with converged solves, and the length raised.

    Parameters
    ----------
    model : torch.nn.Module
        A :class:`~xnn.common.models.ForceStressOutput` around a model with
        step plans; it is put in evaluation mode.
    cutoff : float
        The model's neighbor-list cutoff in Angstrom.
    skin : float, optional
        The margin of the pair list in Angstrom, by default 1.
    margin : float, optional
        The capacity of the plan's pair list relative to its first list, by
        default 1.2.
    warmup : int, optional
        Steps with converged solves before the lengths are fixed, by default
        8 (it also fills the history of the dipole predictor).
    capture : bool, optional
        Record the step as a CUDA graph on a GPU, by default ``True``.
    compute_dtype : torch.dtype, optional
        The model's dtype for a geometry kept in float64.

    Raises
    ------
    ValueError
        If the model has no step plans.
    """

    def __init__(self, model: nn.Module, cutoff: float, skin: float = 1.0, margin: float = 1.2, warmup: int = 8,
                 capture: bool = True, compute_dtype: Optional[torch.dtype] = None):
        # in training mode the forces would carry a graph for a force loss, and the
        # backward pass would solve the dipoles again
        self.model = model.eval()
        self.core = plan_core(model)
        if self.core is None:
            raise ValueError("the model records no step plans")
        self.cutoff, self.skin, self.margin = float(cutoff), float(skin), float(margin)
        self.warmup, self.capture, self.compute_dtype = int(warmup), bool(capture), compute_dtype
        self.rebuilds = 0
        self.full_rebuilds = 0
        self.recordings = 0
        self.replays = 0
        self.reset()

    def reset(self) -> None:
        """Forget the plan and the recorded graph (a new system)."""
        self.plan = None
        self.graph = None
        self.pos: Optional[Tensor] = None
        self.cell: Optional[Tensor] = None
        self.cuda_graph = None
        self.outputs: Optional[dict] = None
        self.flag: Optional[Tensor] = None
        self.stream = None
        self.steps = 0
        self.recent: collections.deque = collections.deque(maxlen=2)
        self.floor: dict = {}

    def __call__(self, struct: dict) -> dict:
        """Evaluate one step.

        Parameters
        ----------
        struct : dict
            The structure, as for :func:`~xnn.common.data.structure_to_graph`,
            with its tensors on the model's device.

        Returns
        -------
        dict of str to Tensor
            ``energy``, ``forces`` and, when the model computes it, ``stress``.
        """
        pos, cell = struct["pos"], struct.get("cell")
        if self.plan is None or pos.shape != self.pos.shape or (cell is None) != (self.cell is None):
            self.reset()
            self._new_plan(struct)
        else:
            with torch.no_grad():
                self.pos.copy_(pos)
                if cell is not None:
                    self.cell.copy_(cell.reshape(self.cell.shape))
            if self.cuda_graph is not None:
                out = self._replay(struct)
                if out is not None:
                    return out
            elif bool(self.plan.outdated(self.pos)):
                self._refresh(struct)
        out = self._eager()
        self._record()
        return out

    def _graph(self, struct: dict):
        graph = structure_to_graph(struct, self.cutoff + self.skin, device=struct["pos"].device)
        graph.compute_dtype = self.compute_dtype
        # the positions and the cell are this object's buffers, refreshed in place every step
        graph.pos = graph.pos.detach().clone()
        if graph.cell is not None:
            graph.cell = graph.cell.detach().clone()
        if graph.pbc is not None:
            # read on the device: a copy from the host cannot be recorded
            graph.pbc = graph.pbc.to(graph.pos.device)
        return graph

    def _new_plan(self, struct: dict, lengths: Optional[dict] = None) -> None:
        graph = self._graph(struct)
        src, dst = graph.edge_index
        pairs = int((src < dst).sum())
        # a new plan outgrew the old one by more than the margin: the next one gets half as much again
        margin = self.margin if self.plan is None else max(self.margin, 1.5)
        # a multiple of 64, so the compiled pair functions keep the alignment they were compiled for
        capacity = 64 * math.ceil(max(margin * pairs, pairs + 1) / 64)
        self.plan = self.core.plan_step(graph, self.skin, capacity)
        self.plan.solve_iterations = lengths
        self.graph, self.pos, self.cell = graph, graph.pos, graph.cell
        self.cuda_graph = None

    def _refresh(self, struct: dict) -> bool:
        """Rebuild the pair list at the current positions: in place when it fits
        the plan (the recorded graph stays valid), else as a new plan. While the
        atoms move continuously only the pairs are found again; an atom wrapped
        back into the cell (a jump of half a cell height or more) also changes
        the image shifts of its bonded vectors, and the whole plan is recorded."""
        self.rebuilds += 1
        struct = dict(struct, pos=self.pos.detach().clone(),
                      cell=None if self.cell is None else self.cell.detach().clone())
        graph = self._graph(struct)
        src, dst = graph.edge_index
        if int((src < dst).sum()) <= self.plan.capacity:
            if not self._jumped() and self.core.plan_pairs(graph, self.plan):
                return True
            self.full_rebuilds += 1
            new = self.core.plan_step(graph, self.skin, self.plan.capacity)
            if self.plan.update(new):
                return True
        self._new_plan(struct, self.plan.solve_iterations)
        return False

    def _jumped(self) -> bool:
        """Whether an atom moved by half the smallest height of its cell or more
        since the pairs were found."""
        if self.cell is None:
            return False
        cell = self.cell.detach().reshape(-1, 3, 3)
        volume = torch.linalg.det(cell).abs()
        cross = torch.stack([torch.linalg.cross(cell[:, 1], cell[:, 2]), torch.linalg.cross(cell[:, 2], cell[:, 0]),
                             torch.linalg.cross(cell[:, 0], cell[:, 1])], 1)
        height = (volume[:, None] / torch.linalg.norm(cross, dim=-1)).min()
        step = self.pos.detach() - self.plan.reference.to(self.pos.dtype)
        return bool((step * step).sum(-1).max() >= (0.5 * height) ** 2)

    def _run(self) -> dict:
        graph = self.graph
        graph.pos, graph.cell = self.pos, self.cell
        out = self.model(graph, self.plan)
        return {key: out[key].detach() for key in ("energy", "forces", "stress") if key in out}

    def _replay(self, struct: dict) -> Optional[dict]:
        self.cuda_graph.replay()
        outdated, converged = torch.stack([self.flag, self.plan.converged]).tolist()
        if outdated:
            if self._refresh(struct):
                return self._replay(struct)
            return None
        if not converged:
            self._lengthen()
            return None
        self.replays += 1
        return {key: value.clone() for key, value in self.outputs.items()}

    def _lengthen(self) -> None:
        """The fixed length did not suffice: the next steps converge, and the length
        is raised from their counts."""
        self.floor, self.plan.solve_iterations = dict(self.plan.solve_iterations), None
        self.cuda_graph = None

    def _eager(self) -> dict:
        if self.capture and self.pos.is_cuda and self.stream is None:
            self.stream = torch.cuda.Stream(self.pos.device)
        if self.stream is not None:
            # the steps run on the stream the graph is recorded on, so its handles exist before
            self.stream.wait_stream(torch.cuda.current_stream(self.pos.device))
            with torch.cuda.stream(self.stream):
                out = self._step()
            torch.cuda.current_stream(self.pos.device).wait_stream(self.stream)
        else:
            out = self._step()
        self.steps += 1
        if self.plan.solve_iterations is None and self.steps >= self.warmup:
            counts: dict = {}
            for seen in self.recent:
                for phase, n in seen.items():
                    counts[phase] = max(counts.get(phase, 0), n)
            self.plan.solve_iterations = {phase: max(int(math.ceil(1.25 * n)) + 1, self.floor.get(phase, 0) + 2)
                                          for phase, n in counts.items()}
        return out

    def _step(self) -> dict:
        out = self._run()
        if self.plan.solve_iterations is not None and not bool(self.plan.converged):
            self._lengthen()
            out = self._run()
        if self.plan.solve_iterations is None:
            stats = self.core.solver_stats
            self.recent.append({phase: int(stats[phase]) for phase in ("forward", "adjoint") if phase in stats})
        return out

    def _record(self) -> None:
        if (not self.capture or not self.pos.is_cuda or self.cuda_graph is not None
                or self.plan.solve_iterations is None):
            return
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.stream, capture_error_mode="thread_local"):
            outputs = self._run()
            flag = self.plan.outdated(self.pos)
        self.cuda_graph, self.outputs, self.flag = graph, outputs, flag
        self.recordings += 1
