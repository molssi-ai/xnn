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
import warnings
from typing import Optional

import torch
from torch import Tensor, nn

from ..data import structure_to_graph
from ..data.neighborlist import cell_grid, half_list, small_cell, unique_pairs

# the phases of the fixed-length solves, in the order of a plan's ``needed`` counts
PHASES = ("forward", "adjoint")
# the term subsets of a step (multiple time steps): the valence terms, the rest, or both
TERMS = ("all", "valence", "nonbonded")


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

    The cell may change from step to step (constant-pressure dynamics). A
    changed cell brings the terms the plan keeps for it (the mesh influence
    functions, inverse cells and volumes) up to date in place before the step
    runs, so the recorded step follows it, and until the next pair list the
    rebuild test also counts how far the cell has moved the periodic images.
    The mesh and the Ewald coefficients stay those the plan was made with.

    For multiple time step integrators (RESPA) a step evaluates a subset of
    the terms: ``terms="valence"`` the bonded terms of an inner step,
    ``terms="nonbonded"`` the rest (pairs, meshes, induction) of an outer
    step, which add up to the whole step. Both run on the one plan. The
    outer steps make every decision above (pair list, solve lengths,
    dipole predictor). An inner step decides nothing and reads nothing from
    the device: it replays its own recorded graph from its second call on,
    and its bonded vectors follow atoms wrapped back into the cell since the
    pair list was built. The first call of a run is an outer step.

    With ``late_checks`` a replayed step returns before its status (pair
    list, convergence, iterations needed) reaches the host. The status is
    read during the next step, after that step's replay is queued, so the
    device is not left idle while the host decides. The pair list is then
    built one step earlier. A step whose list turns out outdated or whose
    solves did not converge has already returned its results: it is counted
    in ``late_failures`` with a warning, and the step that reads it is
    evaluated as without late checks (rebuilt, or with longer solves). A step
    that rebuilds the pair list takes the new list behind its replay and
    leaves its status to the next step too, when the list is found on the
    device alone: a grid search, or with late checks every pair of a smaller
    orthorhombic cell (each edge over the list's cutoff). A new list that does
    not hold (an atom wrapped back into the cell, more pairs than the plan
    holds) is built again in full by the next step. A step at which an atom
    is wrapped back into the cell is only seen to fail after it returned, so
    late checks suit integrators that do not wrap the positions. :meth:`settle`
    reads the status of the last step of a run.

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
    late_checks : bool, optional
        Read the status of a replayed step during the next step, by default
        ``False``.

    Raises
    ------
    ValueError
        If the model has no step plans.
    """

    def __init__(self, model: nn.Module, cutoff: float, skin: float = 1.0, margin: float = 1.2, warmup: int = 8,
                 capture: bool = True, compute_dtype: Optional[torch.dtype] = None, shorten: int = 25,
                 late_checks: bool = False):
        # in training mode the forces would carry a graph for a force loss, and the
        # backward pass would solve the dipoles again
        self.model = model.eval()
        self.core = plan_core(model)
        if self.core is None:
            raise ValueError("the model records no step plans")
        self.cutoff, self.skin, self.margin = float(cutoff), float(skin), float(margin)
        self.warmup, self.capture, self.compute_dtype = int(warmup), bool(capture), compute_dtype
        self.shorten = int(shorten)
        self.late_checks = bool(late_checks)
        self.rebuilds = 0
        # rebuilds found only after a replay, which then replays again
        self.late_rebuilds = 0
        # rebuilds done behind a replay
        self.overlapped = 0
        # steps under late checks found outdated or unconverged after their results were returned
        self.late_failures = 0
        self.full_rebuilds = 0
        self.recordings = 0
        self.replays = 0
        self.valence_replays = 0
        self.reset()

    def reset(self) -> None:
        """Forget the plan and the recorded graph (a new system)."""
        self.plan = None
        self.graph = None
        # the term subset of the recorded step that makes the decisions ("all" or "nonbonded"), and the
        # recorded valence step of an inner time step with its outputs and the evaluations before it
        self.terms = "all"
        self.valence_graph = None
        self.valence_outputs: Optional[dict] = None
        self.valence_steps = 0
        self.pos: Optional[Tensor] = None
        self.cell: Optional[Tensor] = None
        self.cuda_graph = None
        self.outputs: Optional[dict] = None
        self.flag: Optional[Tensor] = None
        # the rebuild ahead of time: its flag, and the positions of the step before (graph buffers)
        self.due: Optional[Tensor] = None
        self.last: Optional[Tensor] = None
        # with a plan that has them, both flags, the convergence and the iterations needed in one buffer
        self.status: Optional[Tensor] = None
        # a rebuild flagged by the last step, done during the next replay on the stream ``side``
        self.pending = False
        # the pairs are found by the grid search, all of it on the second stream
        self.gridded = False
        self._cell_key = None
        # the caller's cell tensor of the last step with its version, held so that the same tensor, unchanged,
        # skips the comparison with the buffer (a read of the device)
        self._seen_cell: Optional[tuple] = None
        self._grid_cache: Optional[tuple] = None
        # the cell the pair list was built in, and the bound on the image shifts of its pairs (per axis)
        # once the cell has changed since then (constant-pressure dynamics), else None
        self.list_cell: Optional[Tensor] = None
        self.shift_bound: Optional[Tensor] = None
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
        # under late checks: the status of the last replayed step on its way to the host (an event, the
        # pinned host buffer, the number of counts, whether it ends with the cell test), and the buffer
        self.previous: Optional[tuple] = None
        self.host: Optional[Tensor] = None
        self.event = None
        # a pair list taken late that did not hold: the next rebuild is a whole plan
        self.stale = False

    def __call__(self, struct: dict, terms: str = "all") -> dict:
        """Evaluate one step.

        Parameters
        ----------
        struct : dict
            The structure, as for :func:`~xnn.common.data.structure_to_graph`,
            with its tensors on the model's device.
        terms : str, optional
            The terms evaluated: ``"all"`` (default), ``"valence"`` (an
            inner step of a multiple time step integrator) or
            ``"nonbonded"`` (an outer step).

        Returns
        -------
        dict of str to Tensor
            ``energy``, ``forces`` and, when the model computes it, ``stress``
            of the terms evaluated.

        Raises
        ------
        ValueError
            If ``terms`` is none of these, or an inner step comes before an
            outer step of the structure.
        """
        if terms not in TERMS:
            raise ValueError(f"terms must be one of {TERMS} (got {terms!r})")
        pos, cell = struct["pos"], struct.get("cell")
        # the caller's cell as given: while it stays the same tensor, unchanged, the grid of the
        # staged pair search is not decided again (that reads the cell on the host)
        self._cell_key = None if cell is None else (id(cell), getattr(cell, "_version", None))
        if self.plan is None or pos.shape != self.pos.shape or (cell is None) != (self.cell is None):
            if terms == "valence":
                raise ValueError("an inner (valence) step needs an outer step of the structure first")
            self.reset()
            self.terms = terms
            self._new_plan(struct)
        else:
            with torch.no_grad():
                self.pos.copy_(pos)
                if cell is not None:
                    seen = self._seen_cell
                    if seen is None or seen[0] is not cell or seen[1] != self._cell_key[1] or seen[1] is None:
                        self._new_cell(cell.reshape(self.cell.shape))
                        self._seen_cell = (cell, self._cell_key[1])
            if terms == "valence":
                return self._valence()
            if terms != self.terms:
                self.terms, self.cuda_graph = terms, None
            if self.cuda_graph is None:
                self.settle()
            if self.cuda_graph is not None:
                out = self._replay(struct)
                if out is not None:
                    return out
            elif self.pending or bool(self.plan.outdated(self.pos)) or (
                    self.shift_bound is not None and bool(self._cell_outdated())):
                self.pending = False
                self._refresh(struct)
        out = self._eager()
        if self.late_checks and self._recordable() and not self._refresh(struct):
            # the first replays read their status a step late, and the eager steps before them rebuild only
            # an outdated list: they start from a list built at these positions (a new plan is evaluated
            # once before it is recorded)
            return out
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
        self.gridded = self._search() is not None
        self.cuda_graph = None
        self.valence_graph, self.valence_steps = None, 0
        self._listed()

    def _listed(self) -> None:
        """Note that the pair list was built at the current positions and cell."""
        self.list_cell = None if self.cell is None else self.cell.detach().clone()
        self.shift_bound = None

    def _new_cell(self, cell: Tensor) -> None:
        """Take the cell of a step into the buffer. A changed cell brings the cell terms the plan keeps
        (the model's ``refresh_cell``) up to date in place, so the recorded step follows it, and from then
        on until the next pair list the rebuild test also counts how far the cell moved the images."""
        if torch.equal(self.cell, cell.to(self.cell.dtype)):
            return
        self.cell.copy_(cell)
        self._grid_cache = None
        refresh = getattr(self.core, "refresh_cell", None)
        if callable(refresh):
            refresh(self.plan, self.cell)
        if self.shift_bound is None:
            self.shift_bound = self._shift_bound()

    def _shift_bound(self) -> Tensor:
        """The largest image shift along each axis of a pair within ``cutoff + skin`` in the cell the pair
        list was built in, ``(B, 3)`` (zero along an axis that is not periodic)."""
        cell = self.list_cell.detach().reshape(-1, 3, 3)
        cross = torch.stack([torch.linalg.cross(cell[:, 1], cell[:, 2]), torch.linalg.cross(cell[:, 2], cell[:, 0]),
                             torch.linalg.cross(cell[:, 0], cell[:, 1])], 1)
        volume = (cell[:, 0] * cross[:, 0]).sum(-1).abs()
        heights = volume[:, None] / torch.linalg.norm(cross, dim=-1)
        bound = torch.floor((self.cutoff + self.skin) / heights) + 1.0
        pbc = self.graph.pbc
        if pbc is not None:
            bound = bound * pbc.reshape(-1, 3).to(bound.dtype)
        return bound

    def _cell_outdated(self) -> Tensor:
        """Whether a pair may have come within the cutoff unlisted since the pair list was built, counting
        the cell's change: twice the largest step of an atom plus the largest move of an image of a pair
        (its shifts times the change of the lattice vectors) above the skin, as a boolean on the device."""
        cell = self.cell.detach().reshape(-1, 3, 3)
        drift = (torch.linalg.norm(cell - self.list_cell.reshape(-1, 3, 3), dim=-1) * self.shift_bound).sum(-1).max()
        step = self.pos.detach() - self.plan.reference.to(self.pos.dtype)
        moved = (step * step).sum(-1).max().sqrt()
        return 2.0 * moved + drift > self.skin

    def _refresh(self, struct: dict) -> bool:
        """Rebuild the pair list at the current positions: in place when it fits
        the plan (the recorded graph stays valid), else as a new plan. While the
        atoms move continuously only the pairs are found again; an atom wrapped
        back into the cell (a jump of half a cell height or more) also changes
        the image shifts of its bonded vectors, and the whole plan is recorded."""
        self.rebuilds += 1
        self._listed()
        # a list taken late that did not hold may hide an atom wrapped back into the cell
        jumped, self.stale = self.stale, False
        if (hasattr(self.core, "refresh_pairs") and self.graph.num_graphs == 1 and not jumped and not self._jumped()
                and self.core.refresh_pairs(self.graph, self.plan, self.cutoff + self.skin)):
            # the pairs found once each at the plan's positions, without a graph
            return True
        struct = dict(struct, pos=self.pos.detach().clone(),
                      cell=None if self.cell is None else self.cell.detach().clone())
        graph = self._graph(struct)
        src, dst = graph.edge_index
        if int((src < dst).sum()) <= self.plan.capacity:
            if not jumped and not self._jumped() and self.core.plan_pairs(graph, self.plan):
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

    def _search(self) -> Optional[tuple]:
        """The edges and cells per axis of the staged search on the device: :func:`cell_grid`, or with late
        checks the edges of a smaller cell searched without a grid (no cells, :func:`small_cell`); ``None``
        when vesin finds the pairs."""
        cut = self.cutoff + self.skin
        grid = cell_grid(self.pos, cut, *self._box())
        if grid is None and self.late_checks:
            edges = small_cell(self.pos, cut, *self._box())
            grid = None if edges is None else (edges, None)
        return grid

    def _grid(self):
        """:meth:`_search`, kept while the caller's cell is unchanged."""
        key = (self._cell_key, self.cutoff + self.skin)
        if self._grid_cache is None or self._grid_cache[0] != key or self._cell_key is None:
            self._grid_cache = (key, self._search())
        return self._grid_cache[1]

    def _stage(self, half: tuple) -> tuple:
        """The new pair list from ``half`` and whether an atom jumped, queued on the
        second stream while the replay runs; the current stream waits for them."""
        with torch.cuda.stream(self.side):
            cut = self.cutoff + self.skin
            grid = None if half else self._grid()
            over = None
            if grid is not None:
                from ..data.grid_pairs import grid_pairs
                # the grid's cell is three cutoffs wide: one image of every pair; the pairs go into
                # buffers of the plan's capacity with their number on the device, so the host reads
                # nothing while the replay runs (more pairs than fit count as a jump: a full rebuild)
                pairs = grid_pairs(self.pos.detach(), cut, *grid, capacity=self.plan.capacity)
                if len(pairs) == 4:
                    over = pairs[3][0] > int(self.plan.capacity)
                fresh = self.core.stage_pairs(self.graph, self.plan, pairs, single_image=grid[1] is not None)
            else:
                fresh = self.core.stage_pairs(self.graph, self.plan,
                                              unique_pairs(self.pos.detach(), cut, *self._box(), half=half or None))
            jump = None if self.cell is None else self._jump()
            if over is not None:
                jump = over if jump is None else torch.logical_or(jump, over)
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
            self._listed()
            return True
        return self._refresh(struct)

    def _run(self, terms: Optional[str] = None) -> dict:
        graph = self.graph
        graph.pos, graph.cell = self.pos, self.cell
        self.plan.terms = self.terms if terms is None else terms
        out = self.model(graph, self.plan)
        return {key: out[key].detach() for key in ("energy", "forces", "stress") if key in out}

    def _valence(self) -> dict:
        """An inner step: the valence terms on the plan of the last outer step, replayed from the second
        evaluation on (the first runs the compiled blocks once eagerly)."""
        if self.valence_graph is not None:
            self.valence_graph.replay()
            self.valence_replays += 1
            return {key: value.clone() for key, value in self.valence_outputs.items()}
        if self.capture and self.pos.is_cuda and self.stream is None:
            self.stream = torch.cuda.Stream(self.pos.device)
        if self.stream is None:
            out = self._run("valence")
        else:
            self.stream.wait_stream(torch.cuda.current_stream(self.pos.device))
            with torch.cuda.stream(self.stream):
                out = self._run("valence")
            torch.cuda.current_stream(self.pos.device).wait_stream(self.stream)
        self.valence_steps += 1
        if self.stream is not None and self.valence_steps >= 2:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.stream, capture_error_mode="thread_local"):
                outputs = self._run("valence")
            self.valence_graph, self.valence_outputs = graph, outputs
            self.recordings += 1
        return out

    def _replay(self, struct: dict) -> Optional[dict]:
        rebuild, self.pending = self.pending, False
        half = self._half() if rebuild else None
        self.cuda_graph.replay()
        fresh = jump = None
        if half is not None:
            fresh, jump = self._stage(half)
        status = self._status(jump)
        # under late checks the step before is read while this one runs
        seen = self._previous()
        if rebuild:
            # this step rebuilds: a rebuild the step before asked for is this one
            self.pending = False
        if self.late_checks and seen in (None, (False, True)) and not self.stale and (
                not rebuild or (half == () and fresh is not None)):
            self._queue(status, jump is not None)
            if rebuild and self.plan.update(fresh, bonded=False):
                # the new pairs, found on the device alone, are taken behind this replay; the next step reads
                # whether they hold
                self.rebuilds += 1
                self.overlapped += 1
                self._listed()
            elif rebuild:
                self.pending = True
            self.replays += 1
            return {key: value.clone() for key, value in self.outputs.items()}
        status = status.tolist()
        n = int(self.plan.needed.numel())
        (outdated, due, converged), needed = status[:3], status[3:3 + n]
        if jump is not None and status[3 + n]:
            # an atom wrapped back into the cell: its bonded image shifts change too
            fresh = None
        if self.shift_bound is not None and status[-1]:
            outdated = True
        if self.stale:
            # the pair list the step before took late does not hold: built again in full
            outdated, rebuild = True, False
        if outdated or not converged:
            # the step is evaluated again at the same positions
            self._rewind()
        if outdated:
            self.late_rebuilds += 1
            if (self._rebuild(struct, fresh) if rebuild else self._refresh(struct)) and self.cuda_graph is not None:
                return self._replay(struct)
            return None
        if not converged:
            if rebuild:
                self._rebuild(struct, fresh)
            self._lengthen()
            return None
        self.replays += 1
        out = {key: value.clone() for key, value in self.outputs.items()}
        if seen is not None and not seen[1] and self.plan.solve_iterations is not None:
            # the step before did not converge in the fixed length (late checks): it is raised
            self._lengthen()
        else:
            # a shortened length drops the graph: the next step records it again
            self._shorten(needed)
        if rebuild:
            # these results are valid on the old pairs; the next step takes the new ones
            self._rebuild(struct, fresh)
        elif due:
            # the pairs hold for one more step: it builds them again while it replays
            self.pending = True
        return out

    def _status(self, jump: Optional[Tensor]) -> Tensor:
        """The status of the replayed step as one float64 tensor on the device: the rebuild flags, the
        convergence, the iterations needed, a jump when the pairs were staged, and the cell test."""
        if self.status is not None:
            status = self.status if jump is None else torch.cat([self.status, jump.reshape(1).to(torch.float64)])
        else:
            flags = [self.flag, self.due, self.plan.converged]
            status = torch.cat([torch.stack(flags).to(torch.float64), self.plan.needed]
                               + ([] if jump is None else [jump.reshape(1).to(torch.float64)]))
        if self.shift_bound is not None:
            # the cell has changed since the pair list: the test that counts the images' move too
            status = torch.cat([status, self._cell_outdated().reshape(1).to(torch.float64)])
        return status

    def _queue(self, status: Tensor, jumps: bool = False) -> None:
        """Copy ``status`` to the pinned host buffer behind the replay, to be read by the next step;
        ``jumps``: it holds the jump test of a pair list staged during the replay."""
        if self.host is None or self.host.shape != status.shape:
            self.host = torch.empty(status.shape, dtype=status.dtype, pin_memory=True)
            self.event = torch.cuda.Event()
        self.host.copy_(status, non_blocking=True)
        self.event.record()
        self.previous = (self.event, self.host, int(self.plan.needed.numel()), self.shift_bound is not None, jumps)

    def _previous(self) -> Optional[tuple]:
        """Read the status of the last step returned under late checks: ``None`` without one, else whether
        its pair list was outdated and whether its solves converged. A step that held passes its counts
        and its rebuild flag on as it would have when read at once; one that failed is counted and warned
        about. A new pair list it took that does not hold marks the list stale (:attr:`stale`)."""
        if self.previous is None:
            return None
        event, host, n, celled, jumps = self.previous
        self.previous = None
        event.synchronize()
        status = host.tolist()
        (outdated, due, converged), needed = status[:3], status[3:3 + n]
        outdated = bool(outdated or (celled and status[-1]))
        if jumps and status[3 + n]:
            # the pairs it found while it ran: an atom wrapped back into the cell, or more pairs than fit
            self.stale = True
            warnings.warn("StepGraph: a pair list taken behind a replay did not hold (an atom wrapped back into "
                          "the cell, or more pairs than the plan holds); the next step builds it in full",
                          RuntimeWarning, stacklevel=4)
        if outdated or not converged:
            self.late_failures += 1
            what = "its pair list was outdated" if outdated else "its solves did not converge in the fixed length"
            warnings.warn(f"StepGraph: a step returned before its late check failed ({what}); "
                          f"{self.late_failures} such steps so far", RuntimeWarning, stacklevel=4)
            return outdated, bool(converged)
        self._shorten(needed)
        if due and not jumps:
            # (a step that found new pairs flags the rebuild of the list before them)
            self.pending = True
        return False, True

    def settle(self) -> bool:
        """Read the status of the last step under late checks, as the next step would (a failed step is
        counted and warned about, and the next step rebuilds or solves longer).

        Returns
        -------
        bool
            Whether the last step held (``True`` without a step to read).
        """
        seen = self._previous()
        if seen is None:
            return True
        outdated, converged = seen
        if outdated or self.stale:
            self.pending = True
        if not converged and self.plan.solve_iterations is not None:
            self._lengthen()
        return not outdated and converged and not self.stale

    def _length(self, phase: str, count: int) -> int:
        """The fixed length for solves whose largest count over the last ``shorten`` steps was
        ``count``: one more, and above a length that failed."""
        return max(count + 1, self.floor[phase] + 2 if phase in self.floor else 0)

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

    def _ahead(self) -> int:
        """The steps the pair list must still hold when a replayed step flags a rebuild: the next one,
        whose replay builds it, and with late checks the one after, which reads the flag."""
        return 3 if self.late_checks else 2

    def _recordable(self) -> bool:
        """Whether the step just evaluated is recorded now."""
        return (self.capture and self.pos.is_cuda and self.cuda_graph is None
                and self.plan.solve_iterations is not None)

    def _record(self) -> None:
        if not self._recordable():
            return
        graph = torch.cuda.CUDAGraph()
        self.last = self.pos.detach().clone()
        checks = getattr(self.plan, "checks", None)
        if callable(checks):
            # its kernel is built before the capture (no kernel is loaded during one); last stays pos
            checks(self.pos, self.last, steps=self._ahead())
        with torch.cuda.graph(graph, stream=self.stream, capture_error_mode="thread_local"):
            outputs = self._run()
            flag = due = status = None
            if callable(checks):
                status = checks(self.pos, self.last, steps=self._ahead())
            else:
                flag = self.plan.outdated(self.pos)
                due = self.plan.due(self.pos, self.last, steps=self._ahead())
                self.last.copy_(self.pos.detach())
        self.cuda_graph, self.outputs, self.flag, self.due, self.status = graph, outputs, flag, due, status
        self.recordings += 1
