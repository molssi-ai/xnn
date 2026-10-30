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
from ..data.neighborlist import cell_grid, half_list, unique_pairs

# the phases of the fixed-length solves, in the order of a plan's ``needed`` counts
PHASES = ("forward", "adjoint")


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
    length is evaluated again with converged solves, and the length raised
    above the one that failed. Every ``shorten`` steps the lengths are
    lowered to the largest count the solves needed in them plus a quarter,
    when that is shorter, and the step is recorded again: the dipole
    predictor keeps improving after the warm-up, and a length set then would
    be paid at every step. A failed length stops being a floor after four
    times ``shorten`` converged steps. The replayed step also tells whether
    an atom will pass half the skin within two more steps like the last; the
    pair list then still holds for the next step, and is built again at that
    step's positions while it replays, on a second stream: in an orthorhombic
    cell the whole of it (a grid search), otherwise all but vesin's neighbor
    list, found before the replay starts. A rebuild costs no second replay
    unless an atom moves further than predicted; a step evaluated again first
    takes its dipoles back out of the predictors.

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
    shorten : int, optional
        Steps between the checks that lower the lengths, by default 25; 0
        keeps the lengths once fixed.
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
                 capture: bool = True, compute_dtype: Optional[torch.dtype] = None, shorten: int = 25):
        # in training mode the forces would carry a graph for a force loss, and the
        # backward pass would solve the dipoles again
        self.model = model.eval()
        self.core = plan_core(model)
        if self.core is None:
            raise ValueError("the model records no step plans")
        self.cutoff, self.skin, self.margin = float(cutoff), float(skin), float(margin)
        self.warmup, self.capture, self.compute_dtype = int(warmup), bool(capture), compute_dtype
        self.shorten = int(shorten)
        self.rebuilds = 0
        # rebuilds found only after a replay, which then replays again
        self.late_rebuilds = 0
        # rebuilds done behind a replay
        self.overlapped = 0
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
        # the rebuild ahead of time: its flag, and the positions of the step before (graph buffers)
        self.due: Optional[Tensor] = None
        self.last: Optional[Tensor] = None
        # a rebuild flagged by the last step, done during the next replay on the stream ``side``
        self.pending = False
        # the pairs are found by the grid search, all of it on the second stream
        self.gridded = False
        self.side = None
        self.staged = None
        self.stream = None
        self.steps = 0
        self.recent: collections.deque = collections.deque(maxlen=2)
        self.floor: dict = {}
        # the iterations the fixed solves needed, per step since the last check, and the converged
        # steps of fixed length since the last failed length
        self.needed: list = []
        self.settled = 0

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
            elif self.pending or bool(self.plan.outdated(self.pos)):
                self.pending = False
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
        self.gridded = cell_grid(self.pos, self.cutoff + self.skin, *self._box()) is not None
        self.cuda_graph = None

    def _refresh(self, struct: dict) -> bool:
        """Rebuild the pair list at the current positions: in place when it fits
        the plan (the recorded graph stays valid), else as a new plan. While the
        atoms move continuously only the pairs are found again; an atom wrapped
        back into the cell (a jump of half a cell height or more) also changes
        the image shifts of its bonded vectors, and the whole plan is recorded."""
        self.rebuilds += 1
        if (hasattr(self.core, "refresh_pairs") and self.graph.num_graphs == 1 and not self._jumped()
                and self.core.refresh_pairs(self.graph, self.plan, self.cutoff + self.skin)):
            # the pairs found once each at the plan's positions, without a graph
            return True
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
        return self.cell is not None and bool(self._jump())

    def _jump(self) -> Tensor:
        """:meth:`_jumped` as a boolean on the device, for a cell."""
        cell = self.cell.detach().reshape(-1, 3, 3)
        cross = torch.stack([torch.linalg.cross(cell[:, 1], cell[:, 2]), torch.linalg.cross(cell[:, 2], cell[:, 0]),
                             torch.linalg.cross(cell[:, 0], cell[:, 1])], 1)
        # the volume as a triple product: a determinant would factorize the cell
        volume = (cell[:, 0] * cross[:, 0]).sum(-1).abs()
        height = (volume[:, None] / torch.linalg.norm(cross, dim=-1)).min()
        step = self.pos.detach() - self.plan.reference.to(self.pos.dtype)
        return (step * step).sum(-1).max() >= (0.5 * height) ** 2

    def _box(self) -> tuple:
        """The cell and periodicity of the one structure, as the neighbor lists take them."""
        cell = None if self.cell is None else self.cell.detach().reshape(-1, 3, 3)[0]
        pbc = None if self.graph.pbc is None else self.graph.pbc.reshape(-1, 3)[0]
        return cell, pbc

    def _half(self) -> Optional[tuple]:
        """What :meth:`_stage` needs before the replay is queued: vesin's half
        neighbor list at the current positions, empty when the grid search finds
        the pairs on the second stream, ``None`` when the pairs are found otherwise."""
        if not (hasattr(self.core, "stage_pairs") and self.graph.num_graphs == 1 and self.pos.is_cuda):
            return None
        half = () if self.gridded else half_list(self.pos.detach(), self.cutoff + self.skin, *self._box())
        if half is not None:
            if self.side is None:
                self.side = torch.cuda.Stream(self.pos.device)
            # the positions and the list are ready before the replay is queued
            self.side.wait_stream(torch.cuda.current_stream(self.pos.device))
        return half

    def _stage(self, half: tuple) -> tuple:
        """The new pair list from ``half`` and whether an atom jumped, queued on the
        second stream while the replay runs; the current stream waits for them."""
        with torch.cuda.stream(self.side):
            cut = self.cutoff + self.skin
            grid = None if half else cell_grid(self.pos, cut, *self._box())
            if grid is not None:
                from ..data.grid_pairs import grid_pairs
                # the grid's cell is three cutoffs wide: one image of every pair
                fresh = self.core.stage_pairs(self.graph, self.plan, grid_pairs(self.pos.detach(), cut, *grid),
                                              single_image=True)
            else:
                fresh = self.core.stage_pairs(self.graph, self.plan,
                                              unique_pairs(self.pos.detach(), cut, *self._box(), half=half or None))
            jump = None if self.cell is None else self._jump()
        torch.cuda.current_stream(self.pos.device).wait_stream(self.side)
        # kept until the next stage: the copies into the plan read it on the current stream
        self.staged = fresh
        return fresh, jump

    def _rewind(self) -> None:
        """Take the dipoles of a step evaluated again out of the predictors."""
        if callable(getattr(self.core, "rewind_predictors", None)):
            self.core.rewind_predictors()

    def _rebuild(self, struct: dict, fresh) -> bool:
        """Take the staged pair list ``fresh``, or rebuild as :meth:`_refresh`."""
        if fresh is not None and self.plan.update(fresh, bonded=False):
            self.rebuilds += 1
            self.overlapped += 1
            return True
        return self._refresh(struct)

    def _run(self) -> dict:
        graph = self.graph
        graph.pos, graph.cell = self.pos, self.cell
        out = self.model(graph, self.plan)
        return {key: out[key].detach() for key in ("energy", "forces", "stress") if key in out}

    def _replay(self, struct: dict) -> Optional[dict]:
        rebuild, self.pending = self.pending, False
        half = self._half() if rebuild else None
        self.cuda_graph.replay()
        fresh = jump = None
        if half is not None:
            fresh, jump = self._stage(half)
        # one read from the device: the rebuild flags, the convergence, a jump and the iterations needed
        flags = [self.flag, self.due, self.plan.converged] + ([] if jump is None else [jump])
        status = torch.cat([torch.stack(flags).to(torch.float64), self.plan.needed]).tolist()
        (outdated, due, converged), needed = status[:3], status[len(flags):]
        if jump is not None and status[3]:
            # an atom wrapped back into the cell: its bonded image shifts change too
            fresh = None
        if outdated or not converged:
            # the step is evaluated again at the same positions
            self._rewind()
        if outdated:
            self.late_rebuilds += 1
            if self._rebuild(struct, fresh) if rebuild else self._refresh(struct):
                return self._replay(struct)
            return None
        if not converged:
            if rebuild:
                self._rebuild(struct, fresh)
            self._lengthen()
            return None
        self.replays += 1
        out = {key: value.clone() for key, value in self.outputs.items()}
        # a shortened length drops the graph: the next step records it again
        self._shorten(needed)
        if rebuild:
            # these results are valid on the old pairs; the next step takes the new ones
            self._rebuild(struct, fresh)
        elif due:
            # the pairs hold for one more step: it builds them again while it replays
            self.pending = True
        return out

    def _length(self, phase: str, count: int) -> int:
        """The fixed length for solves that needed ``count`` iterations: a quarter
        more plus one, and above a length that failed."""
        return max(int(math.ceil(1.25 * count)) + 1, self.floor[phase] + 2 if phase in self.floor else 0)

    def _lengthen(self) -> None:
        """The fixed length did not suffice: the next steps converge, and the length
        is raised from their counts."""
        self.floor, self.plan.solve_iterations = dict(self.plan.solve_iterations), None
        # the counts of the steps before the failure no longer set the length
        self.recent.clear()
        self.needed.clear()
        self.settled = 0
        self.cuda_graph = None

    def _shorten(self, needed: list) -> None:
        """Note the iterations the fixed solves needed; every ``shorten`` steps lower
        the lengths to what they needed, when that is shorter."""
        if not self.shorten or self.plan.solve_iterations is None:
            return
        self.needed.append(needed)
        self.settled += 1
        if self.settled > 4 * self.shorten:
            self.floor = {}
        if len(self.needed) < self.shorten:
            return
        lengths = dict(self.plan.solve_iterations)
        for slot, phase in enumerate(PHASES):
            if phase in lengths:
                lengths[phase] = min(lengths[phase], self._length(phase, max(int(row[slot]) for row in self.needed)))
        self.needed.clear()
        if lengths != self.plan.solve_iterations:
            self.plan.solve_iterations = lengths
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
        if self.plan.solve_iterations is None and self.steps >= self.warmup and len(self.recent) == 2:
            counts: dict = {}
            for seen in self.recent:
                for phase, n in seen.items():
                    counts[phase] = max(counts.get(phase, 0), n)
            self.plan.solve_iterations = {phase: self._length(phase, n) for phase, n in counts.items()}
        return out

    def _step(self) -> dict:
        out = self._run()
        if self.plan.solve_iterations is not None and not bool(self.plan.converged):
            self._lengthen()
            self._rewind()
            out = self._run()
        elif self.plan.solve_iterations is not None:
            self._shorten(self.plan.needed.tolist())
        if self.plan.solve_iterations is None:
            stats = self.core.solver_stats
            self.recent.append({phase: int(stats[phase]) for phase in ("forward", "adjoint") if phase in stats})
        return out

    def _record(self) -> None:
        if (not self.capture or not self.pos.is_cuda or self.cuda_graph is not None
                or self.plan.solve_iterations is None):
            return
        graph = torch.cuda.CUDAGraph()
        self.last = self.pos.detach().clone()
        with torch.cuda.graph(graph, stream=self.stream, capture_error_mode="thread_local"):
            outputs = self._run()
            flag = self.plan.outdated(self.pos)
            due = self.plan.due(self.pos, self.last, steps=2)
            self.last.copy_(self.pos.detach())
        self.cuda_graph, self.outputs, self.flag, self.due = graph, outputs, flag, due
        self.recordings += 1
