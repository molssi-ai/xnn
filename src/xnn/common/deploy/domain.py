"""Spatial domain decomposition: one large structure evaluated on several GPUs.

The box is cut into a grid of sub-domains, one per process. A process evaluates
the atoms of its sub-domain (its owned atoms) and copies of the atoms around it
(the ghosts), and keeps the energies of its owned atoms only. The ghost shell
takes one of two forms:

- ``halo="layer"``: one cutoff of ghosts. Before every layer that reads its
  neighbors' features, the model replaces the ghosts' features by the values
  their owners computed (:func:`~xnn.common.models.ops.refresh_ghosts`, one
  exchange between neighboring processes per layer). For the models that set
  :attr:`~xnn.common.models.base.InteratomicPotential.refreshes_ghosts`
  (MACE, TensorNet, AIMNet2). The ghosts receive no messages: only the edges
  into owned atoms are evaluated.
- ``halo="field"``: ghosts out to the model's receptive field and no exchange
  inside the model. The ghosts at the outer edge of the shell are evaluated
  wrongly, but they are too far away to reach an owned atom.

The positions of the ghosts come from their owners through the same exchange,
so the derivatives of a process's energy with respect to its ghosts return to
their owners: every process ends with the forces of its own atoms, and only the
energy and the virial are summed over the processes.

A plan (the owner of every atom, the ghosts and the neighbor list) is built
with a ``skin``: it stays valid while the atoms have moved, and the cell has
deformed, less than the skin allows, so a simulation rebuilds it only now and
then.

Periodic boxes and molecules are both supported. Along a periodic axis that is
not cut, the sub-domains keep the box's periodicity; along a cut axis, the
ghosts are explicit periodic images.

Launch one process per GPU with ``torchrun``. With the whole structure on
every process::

    import torch, torch.distributed as dist
    from xnn.common.deploy import DomainParallel

    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    dp = DomainParallel(model.cuda().eval(), skin=1.0)
    out = dp(graph.to("cuda"))     # every process gets energy, forces and stress

With the positions of its own atoms only on each process (the forces are not
gathered)::

    plan = dp.plan(graph)                          # once, from the whole structure
    pos, vel = graph.pos[plan.owned], vel[plan.owned]
    for step in range(n_steps):
        if not dp.holds(plan, pos):
            plan, pos, vel = dp.reshard(plan, pos, vel)
        out = dp.evaluate(plan, pos)               # out["forces"]: the owned atoms
        ...                                        # move the owned atoms
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.distributed as dist
from torch import Tensor, nn

from ..data import AtomicGraph
from ..data.neighborlist import build_neighbor_list
from ..models import ops
from ..models.outputs import ForceStressOutput

#: distance added to the shell and the neighbor list, so that rounding never drops an atom on their edge
HALO_MARGIN = 1e-6

HALOS = ("auto", "layer", "field")


def _process(group) -> tuple:
    """The rank of this process and the number of processes."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(group), dist.get_world_size(group)
    return 0, 1


def _all_to_all(x: Tensor, out_counts: Sequence[int], in_counts: Sequence[int], group) -> Tensor:
    """Rows of ``x`` sent to every process (``in_counts`` to each, in rank order), the rows
    received from every process concatenated (``out_counts`` from each); differentiable."""
    from torch.distributed.nn.functional import all_to_all_single
    out = x.new_empty((int(sum(out_counts)),) + tuple(x.shape[1:]))
    return all_to_all_single(out, x.contiguous(), list(out_counts), list(in_counts),
                             group=group if group is not None else dist.group.WORLD)


def _all_reduce(x: Tensor, group, op: str = "sum") -> Tensor:
    """``x`` summed (or its maximum taken) over the processes; differentiable."""
    from torch.distributed.nn.functional import all_reduce
    return all_reduce(x, op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM,
                      group=group if group is not None else dist.group.WORLD)


def _all_gather(x: Tensor, group) -> list:
    """``x`` of every process, in rank order."""
    _, world = _process(group)
    out = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(out, x.contiguous(), group=group)
    return out


@dataclass
class DomainPlan:
    """The atoms one process evaluates, its exchanges and its neighbor list.

    Local atoms are the owned atoms followed by the ghosts.

    Attributes
    ----------
    grid : tuple of int
        Sub-domains along the three axes.
    rank, world : int
        The process (sub-domain) the plan belongs to and the number of processes.
    owned : Tensor
        Owned atoms ``(n_owned,)``, in ascending order.
    ghosts : Tensor
        Atom behind every ghost ``(n_ghost,)``, grouped by owner.
    ghost_shifts : Tensor or None
        Periodic image of every ghost relative to its owner's copy ``(n_ghost, 3)``;
        ``None`` for a molecule.
    send_rows : Tensor
        Owned atoms (rows of ``owned``) sent to the other processes, grouped by
        receiving process.
    send_counts, recv_counts : list of int
        Rows sent to and received from every process.
    edge_index, cell_shifts : Tensor
        Neighbor list of the local atoms, as in :class:`~xnn.common.data.AtomicGraph`,
        out to the cutoff plus the skin; with the one-cutoff shell, only the edges
        into owned atoms.
    pbc : Tensor or None
        Periodicity of the local atoms ``(3,)``: the periodic axes that are not cut.
        ``None`` when no axis stays periodic.
    atomic_numbers : Tensor
        Atomic numbers of every atom of the structure ``(N,)``.
    total_charge, spin_multiplicity : Tensor or None
        Of the structure, as in its graph.
    reference : Tensor
        Positions of the owned atoms when the plan was built ``(n_owned, 3)``,
        float64: fractional and wrapped into the cell for a periodic structure,
        Cartesian for a molecule.
    cell : Tensor or None
        The cell when the plan was built ``(3, 3)``, float64; ``None`` for a molecule.
    counts : list of int
        Owned atoms of every process.
    order : Tensor
        The owned atoms of every process, process by process ``(N,)``.
    need : float
        Thickness of the ghost shell the model needs (one cutoff or the receptive field).
    skin : float
        Thickness added to the shell and the neighbor list.
    """

    grid: tuple
    rank: int
    world: int
    owned: Tensor
    ghosts: Tensor
    ghost_shifts: Optional[Tensor]
    send_rows: Tensor
    send_counts: list
    recv_counts: list
    edge_index: Tensor
    cell_shifts: Tensor
    pbc: Optional[Tensor]
    atomic_numbers: Tensor
    total_charge: Optional[Tensor]
    spin_multiplicity: Optional[Tensor]
    reference: Tensor
    cell: Optional[Tensor]
    counts: list
    order: Tensor
    need: float
    skin: float

    @property
    def n_owned(self) -> int:
        """int : Number of owned atoms (the first ``n_owned`` local atoms)."""
        return int(self.owned.shape[0])

    @property
    def n_ghost(self) -> int:
        """int : Number of ghosts."""
        return int(self.ghosts.shape[0])

    @property
    def n_local(self) -> int:
        """int : Number of local atoms (owned and ghosts)."""
        return self.n_owned + self.n_ghost

    @property
    def n_atoms(self) -> int:
        """int : Number of atoms of the structure."""
        return int(self.atomic_numbers.shape[0])

    @property
    def index(self) -> Tensor:
        """Tensor : Atom behind every local atom ``(n_local,)``."""
        return torch.cat([self.owned, self.ghosts])


def choose_grid(n_domains: int, widths: Sequence[float], halo: float) -> tuple:
    """The grid of ``n_domains`` sub-domains that evaluates the fewest ghosts.

    Every cut axis adds a shell of thickness ``halo`` on both sides of a
    sub-domain; the grid minimizing the evaluated volume relative to the owned
    volume is returned.

    Parameters
    ----------
    n_domains : int
        Number of sub-domains (processes).
    widths : sequence of float
        Width of the box along each axis (the distance between opposite faces).
    halo : float
        Thickness of the ghost shell.

    Returns
    -------
    tuple of int
        Sub-domains along the three axes, with product ``n_domains``.
    """
    best, best_cost = None, math.inf
    for gx in range(1, n_domains + 1):
        if n_domains % gx:
            continue
        for gy in range(1, n_domains // gx + 1):
            if (n_domains // gx) % gy:
                continue
            grid = (gx, gy, n_domains // gx // gy)
            cost = 1.0
            for g, w in zip(grid, widths):
                if g > 1:
                    cost *= 1.0 + 2.0 * halo * g / max(float(w), 1e-12)
            if cost < best_cost - 1e-12:
                best, best_cost = grid, cost
    return best


def _model_dtype(model: nn.Module) -> Optional[torch.dtype]:
    for t in model.parameters():
        if t.is_floating_point():
            return t.dtype
    for t in model.buffers():
        if t.is_floating_point():
            return t.dtype
    return None


def _tie(x: Tensor) -> Tensor:
    """An exact zero that depends on ``x``.

    Every collective of an evaluation adds the tie of the previous one to its input, and the
    energy adds the ties of all of them: the backward pass of every process then runs the
    adjoint collectives exactly in the reverse order of the forward ones, whatever the shape of
    its own autograd graph (a process with few atoms, frozen parameters or another fast path).
    """
    return x.reshape(-1)[:0].sum(dtype=torch.float64)


class DomainParallel(nn.Module):
    """Evaluate one structure split over the processes of a distributed launch.

    Each process evaluates only its sub-domain and a shell of ghost atoms, so
    the memory per process falls with the number of processes, and the results
    equal a single-process evaluation to rounding. The model runs as it would
    on one device, fast paths included. Evaluation only: no second derivatives.
    Every method is collective: all processes call it together.

    Parameters
    ----------
    model : torch.nn.Module
        The potential (an :class:`~xnn.common.models.base.InteratomicPotential`,
        or a :class:`~xnn.common.models.outputs.ForceStressOutput` around one).
        Its energy must be local: it needs a :meth:`receptive_field` (models with
        long-range terms have none), unless ``receptive_field`` is given.
    grid : sequence of int, optional
        Sub-domains along the three lattice vectors (Cartesian axes for a
        molecule), with product equal to the number of processes. By default
        the grid with the fewest ghosts (:func:`choose_grid`).
    receptive_field : float, optional
        Distance in Angstrom beyond which atoms no longer change an atom's
        energy, by default the model's :meth:`receptive_field`.
    compute_stress : bool, optional
        Whether to compute the stress of a periodic structure, by default ``True``.
    group : torch.distributed.ProcessGroup, optional
        The processes that share the structure, by default all of them.
    halo : {"auto", "layer", "field"}, optional
        The ghost shell: one cutoff, with the ghosts' features refreshed before
        every layer (``"layer"``, for models that set ``refreshes_ghosts``), or
        the whole receptive field (``"field"``). ``"auto"`` (default) takes
        ``"layer"`` when the model supports it.
    skin : float, optional
        Thickness in Angstrom added to the ghost shell and the neighbor list, so
        that a plan holds while the atoms move, by default 0 (a plan holds for
        the positions it was built from only).

    Raises
    ------
    ValueError
        For a model without a receptive field, an unknown ``halo``, ``"layer"``
        for a model that does not refresh its ghosts, or a negative ``skin``.
    """

    def __init__(self, model: nn.Module, grid: Optional[Sequence[int]] = None,
                 receptive_field: Optional[float] = None, compute_stress: bool = True,
                 group=None, halo: str = "auto", skin: float = 0.0):
        super().__init__()
        if isinstance(model, ForceStressOutput):
            model = model.model
        if receptive_field is None:
            field = getattr(model, "receptive_field", None)
            receptive_field = field() if callable(field) else None
        if receptive_field is None:
            raise ValueError(f"{type(model).__name__} has no bounded receptive field (its energy "
                             "has long-range terms, or the model does not state it); pass "
                             "receptive_field= to decompose it anyway")
        if halo not in HALOS:
            raise ValueError(f"halo={halo!r} is not supported; options: {list(HALOS)}")
        refreshes = bool(getattr(model, "refreshes_ghosts", False))
        if halo == "layer" and not refreshes:
            raise ValueError(f"{type(model).__name__} does not refresh its ghosts between layers; "
                             "use halo='field'")
        if skin < 0:
            raise ValueError(f"skin must not be negative, got {skin}")
        self.model = model
        self.cutoff = float(model.cutoff)
        self.receptive_field = float(receptive_field)
        self.layer = halo == "layer" or (halo == "auto" and refreshes)
        self.need = self.cutoff if self.layer else self.receptive_field
        self.skin = float(skin)
        self.grid = tuple(int(g) for g in grid) if grid is not None else None
        self.compute_stress = compute_stress
        self.group = group
        self._plan: Optional[DomainPlan] = None

    @property
    def halo(self) -> str:
        """str : The ghost shell in use, ``"layer"`` or ``"field"``."""
        return "layer" if self.layer else "field"

    @torch.no_grad()
    def plan(self, data: AtomicGraph) -> DomainPlan:
        """Partition the structure and build this process's ghosts and neighbor list.

        Parameters
        ----------
        data : AtomicGraph
            One structure, periodic along all three axes or along none, the
            same on every process.

        Returns
        -------
        DomainPlan
            The plan of this process.

        Raises
        ------
        ValueError
            For a batch of several structures, a partially periodic structure or
            a grid that does not match the number of processes.
        """
        if data.num_graphs != 1:
            raise ValueError("DomainParallel evaluates one structure, not a batch")
        periodic = data.cell is not None and data.pbc is not None and bool(data.pbc.any())
        if periodic and not bool(data.pbc.all()):
            raise ValueError("DomainParallel takes structures periodic along all axes or none")
        return self._build(data.pos, data.atomic_numbers, data.cell[0] if periodic else None,
                           data.total_charge, data.spin_multiplicity)

    def _build(self, positions: Tensor, atomic_numbers: Tensor, cell: Optional[Tensor],
               total_charge: Optional[Tensor], spin_multiplicity: Optional[Tensor]) -> DomainPlan:
        rank, world = _process(self.group)
        pos = positions.detach().to(torch.float64)
        device, n = pos.device, int(pos.shape[0])
        halo = self.need + self.skin + HALO_MARGIN
        periodic = cell is not None
        if periodic:
            cell = cell.detach().to(torch.float64)
            inverse = torch.linalg.inv(cell)
            raw = pos @ inverse
            frac = raw - torch.floor(raw)
            widths = 1.0 / torch.linalg.norm(inverse, dim=0)
        else:
            low = pos.min(0).values if n else pos.new_zeros(3)
            widths = (pos.max(0).values - low if n else pos.new_zeros(3)) + 1e-6
            frac = (pos - low) / widths
        widths = [float(w) for w in widths]
        grid = self.grid if self.grid is not None else choose_grid(world, widths, halo)
        if len(grid) != 3 or math.prod(grid) != world:
            raise ValueError(f"grid {tuple(grid)} does not hold {world} processes")
        coords = (rank // (grid[1] * grid[2]), (rank // grid[2]) % grid[1], rank % grid[2])

        g = torch.tensor(grid, device=device)
        cells = torch.minimum(torch.floor(frac * g).long().clamp(min=0), g - 1)
        owner = (cells[:, 0] * grid[1] + cells[:, 1]) * grid[2] + cells[:, 2]
        owned = (owner == rank).nonzero().flatten()
        counts = torch.bincount(owner, minlength=world).tolist()
        order = torch.sort(owner, stable=True).indices

        masks, steps = [], []
        for a in range(3):
            if grid[a] == 1:
                masks.append(torch.ones(n, 1, dtype=torch.bool, device=device))
                steps.append(torch.zeros(1, dtype=torch.long, device=device))
                continue
            shell = halo / widths[a]
            lo, hi = coords[a] / grid[a] - shell, (coords[a] + 1) / grid[a] + shell
            m = math.floor(shell) + 1 if periodic else 0
            image = torch.arange(-m, m + 1, device=device)
            x = frac[:, a, None] + image.to(torch.float64)
            masks.append((x >= lo) & (x < hi))
            steps.append(image)
        mask = masks[0][:, :, None, None] & masks[1][:, None, :, None] & masks[2][:, None, None, :]
        atom, i0, i1, i2 = mask.nonzero(as_tuple=True)
        image = torch.stack([steps[0][i0], steps[1][i1], steps[2][i2]], dim=1)
        ghost = ~((owner[atom] == rank) & (image == 0).all(1))
        atom, image = atom[ghost], image[ghost]
        by_owner = torch.sort(owner[atom], stable=True).indices
        ghosts, ghost_image = atom[by_owner], image[by_owner]
        recv_counts = torch.bincount(owner[ghosts], minlength=world)

        # every process asks the owners of its ghosts for them
        if world > 1:
            send_counts = _all_to_all(recv_counts, [1] * world, [1] * world, self.group).tolist()
            requested = _all_to_all(ghosts, send_counts, recv_counts.tolist(), self.group)
        else:
            send_counts, requested = recv_counts.tolist(), ghosts
        recv_counts = recv_counts.tolist()
        send_rows = torch.searchsorted(owned, requested)
        if len(requested) and not (len(owned) and bool(
                (owned[send_rows.clamp(max=len(owned) - 1)] == requested).all())):
            raise RuntimeError("the processes disagree on the owners of the atoms")

        if periodic:
            local = torch.cat([frac[owned], frac[ghosts] + ghost_image.to(torch.float64)]) @ cell
            reference = frac[owned]
        else:
            local = torch.cat([pos[owned], pos[ghosts]])
            reference = pos[owned]
        reach = self.cutoff + self.skin + HALO_MARGIN
        native = [periodic and grid[a] == 1 for a in range(3)]
        if not len(local):
            # a sub-domain without atoms near it
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            cell_shifts = torch.zeros((0, 3), dtype=torch.long, device=device)
            pbc = torch.tensor(native) if any(native) else None
        elif any(native):
            # the cut axes stretched far beyond the shell, so that only the explicit
            # images are neighbors along them, while the others keep the box's periodicity
            search = cell.clone()
            for a in range(3):
                if not native[a]:
                    search[a] *= 1.0 + 1.0 / grid[a] + 2.0 * (halo + reach) / widths[a]
            edge_index, cell_shifts = build_neighbor_list(local, reach, search,
                                                          torch.ones(3, dtype=torch.bool))
            cut = torch.tensor([not x for x in native], device=cell_shifts.device)
            if bool((cell_shifts[:, cut] != 0).any()):
                raise RuntimeError("a neighbor across a cut axis was found through the box's periodicity")
            pbc = torch.tensor(native)
        else:
            edge_index, cell_shifts = build_neighbor_list(local, reach)
            pbc = None
        if self.layer:
            # the ghosts' features are replaced by their owners' before they are read: messages
            # into the ghosts are never needed
            keep = edge_index[1] < len(owned)
            edge_index, cell_shifts = edge_index[:, keep], cell_shifts[keep.to(cell_shifts.device)]
        return DomainPlan(
            grid=tuple(grid), rank=rank, world=world, owned=owned, ghosts=ghosts,
            ghost_shifts=ghost_image if periodic else None, send_rows=send_rows,
            send_counts=send_counts, recv_counts=recv_counts, edge_index=edge_index.to(device),
            cell_shifts=cell_shifts.to(device), pbc=pbc, atomic_numbers=atomic_numbers,
            total_charge=total_charge, spin_multiplicity=spin_multiplicity, reference=reference,
            cell=cell, counts=counts, order=order, need=self.need, skin=self.skin)

    def _cell(self, plan: DomainPlan, cell: Optional[Tensor]) -> Optional[Tensor]:
        if plan.cell is None:
            return None
        if cell is None:
            return plan.cell
        return cell[0] if cell.dim() == 3 else cell

    @torch.no_grad()
    def holds(self, plan: DomainPlan, pos: Tensor, cell: Optional[Tensor] = None) -> bool:
        """Whether the plan still holds for new positions and cell.

        It holds while ``need + 2 d <= (1 - e) (need + skin)``, with ``d`` the
        largest displacement of an atom since the plan was built and ``e`` the
        largest relative change of a distance through the deformation of the cell.

        Parameters
        ----------
        pos : Tensor
            Positions of the owned atoms ``(n_owned, 3)``, wrapped into the cell or not.
        cell : Tensor, optional
            The cell ``(3, 3)``, by default the plan's.

        Returns
        -------
        bool
            The same on every process.
        """
        cell = self._cell(plan, cell)
        strain = 0.0
        if cell is None:
            moved = pos.detach().to(torch.float64) - plan.reference
        else:
            cell = cell.detach().to(torch.float64)
            frac = pos.detach().to(torch.float64) @ torch.linalg.inv(cell)
            frac = frac - plan.reference
            moved = (frac - torch.round(frac)) @ cell
            strain = float(torch.linalg.matrix_norm(torch.linalg.inv(plan.cell) @ (cell - plan.cell), ord=2))
        largest = moved.norm(dim=-1).max() if len(moved) else moved.new_zeros(())
        if plan.world > 1:
            largest = _all_reduce(largest.reshape(1), self.group, op="max")[0]
        need = plan.need
        return need + 2.0 * float(largest) <= (1.0 - strain) * (need + plan.skin + HALO_MARGIN)

    @torch.no_grad()
    def reshard(self, plan: DomainPlan, pos: Tensor, *fields: Tensor,
                cell: Optional[Tensor] = None) -> tuple:
        """A new plan for the current positions, with the per-atom rows moved to their new owners.

        Parameters
        ----------
        plan : DomainPlan
            The plan the rows belong to.
        pos : Tensor
            Positions of the owned atoms ``(n_owned, 3)``.
        *fields : Tensor
            Other rows of the owned atoms (velocities, masses, ...), ``(n_owned, ...)``.
        cell : Tensor, optional
            The cell ``(3, 3)``, by default the plan's.

        Returns
        -------
        tuple
            The new plan, then ``pos`` and every field for the atoms it owns.
        """
        cell = self._cell(plan, cell)
        whole = [self._gather(plan, x) for x in (pos,) + fields]
        new = self._build(whole[0], plan.atomic_numbers, cell, plan.total_charge,
                          plan.spin_multiplicity)
        return (new,) + tuple(x[new.owned] for x in whole)

    def _gather(self, plan: DomainPlan, rows: Tensor) -> Tensor:
        """The rows of every process's owned atoms, in the order of the atoms of the structure."""
        if plan.world == 1:
            whole = rows.new_empty((plan.n_atoms,) + tuple(rows.shape[1:]))
            whole[plan.owned] = rows
            return whole
        size = max(plan.counts)
        padded = rows.new_zeros((size,) + tuple(rows.shape[1:]))
        padded[:len(rows)] = rows
        parts = _all_gather(padded, self.group)
        whole = rows.new_empty((plan.n_atoms,) + tuple(rows.shape[1:]))
        whole[plan.order] = torch.cat([p[:c] for p, c in zip(parts, plan.counts)])
        return whole

    def _exchange(self, plan: DomainPlan, rows: Tensor, ties: list) -> Tensor:
        """The rows of the ghosts, sent by their owners from ``rows`` (the owned atoms)."""
        send = rows.index_select(0, plan.send_rows)
        if plan.world == 1:
            return send
        if ties:
            send = send + _tie(ties[-1]).to(send.dtype)
        out = _all_to_all(send, plan.recv_counts, plan.send_counts, self.group)
        ties.append(out)
        return out

    @contextlib.contextmanager
    def _hooks(self, plan: DomainPlan, ties: list):
        """Point the model's whole-structure sums and ghost refreshes at this plan."""
        n_owned = plan.n_owned
        rows = max(plan.n_local, 1)

        def reduction(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
            part = ops.scatter_sum(src[:n_owned], index[:n_owned], dim_size)
            if plan.world > 1:
                part = part + _tie(ties[-1]).to(part.dtype)
                part = _all_reduce(part, self.group)
                ties.append(part)
            return part

        def refresh(x: Tensor) -> Tensor:
            if x.shape[0] != rows:
                raise ValueError(f"refresh_ghosts got {x.shape[0]} rows for {rows} atoms")
            return torch.cat([x[:n_owned], self._exchange(plan, x[:n_owned], ties), x[plan.n_local:]])

        state = ops._DOMAIN
        previous = (getattr(state, "reduction", None), getattr(state, "refresh", None))
        state.reduction, state.refresh = reduction, refresh if self.layer else None
        try:
            yield
        finally:
            state.reduction, state.refresh = previous

    def evaluate(self, plan: DomainPlan, pos: Tensor, cell: Optional[Tensor] = None) -> dict:
        """Energy, forces of the owned atoms and stress.

        Parameters
        ----------
        plan : DomainPlan
            This process's plan, valid for the positions (:meth:`holds`).
        pos : Tensor
            Positions of the owned atoms ``(n_owned, 3)``, wrapped into the cell or not.
        cell : Tensor, optional
            The cell ``(3, 3)``, by default the plan's.

        Returns
        -------
        dict of str to Tensor
            ``"energy"`` ``(1,)`` float64 (the whole structure), ``"forces"``
            ``(n_owned, 3)`` in the dtype of the positions and, for a periodic
            structure with ``compute_stress``, ``"stress"`` ``(1, 3, 3)``.
        """
        if pos.shape[0] != plan.n_owned:
            raise ValueError(f"{pos.shape[0]} positions for the {plan.n_owned} atoms of the plan")
        cell = self._cell(plan, cell)
        p = pos.detach().requires_grad_(True)
        dtype, device = p.dtype, p.device
        strain = None
        own = p
        volume = None
        if cell is not None:
            cell = cell.detach().to(dtype)
            volume = torch.linalg.det(cell.to(torch.float64)).abs().reshape(1)
            jumps = torch.round(p.detach().to(torch.float64) @ torch.linalg.inv(cell.to(torch.float64))
                                - plan.reference).to(dtype)
            if self.compute_stress:
                strain = torch.zeros((3, 3), dtype=dtype, device=device, requires_grad=True)
                sym = 0.5 * (strain + strain.transpose(0, 1))
                own = own + own @ sym
                cell = cell + cell @ sym
            # the owned atoms where the plan saw them, give or take their displacement
            own = own - jumps @ cell
        ties: list = []
        ghosts = self._exchange(plan, own, ties)
        if cell is not None:
            ghosts = ghosts + plan.ghost_shifts.to(dtype) @ cell
        local = torch.cat([own, ghosts])
        numbers = plan.atomic_numbers[plan.index]
        if plan.n_local == 0:
            # a process without atoms evaluates one isolated stand-in, whose energy it drops, so
            # that it runs the model and takes part in every exchange of the others
            local = local.new_zeros((1, 3))
            numbers = plan.atomic_numbers[:1]
        edge_index, cell_shifts = plan.edge_index, plan.cell_shifts
        with torch.no_grad():
            vec = local[edge_index[1]] - local[edge_index[0]]
            if plan.pbc is not None:
                vec = vec + cell_shifts.to(dtype) @ cell
            keep = (vec * vec).sum(-1) < self.cutoff ** 2
        edge_index, cell_shifts = edge_index[:, keep], cell_shifts[keep]
        model_dtype = _model_dtype(self.model)
        n_local = int(local.shape[0])
        graph = AtomicGraph(
            pos=local, atomic_numbers=numbers,
            edge_index=edge_index, cell_shifts=cell_shifts,
            batch=torch.zeros(n_local, dtype=torch.long, device=device),
            n_atoms=torch.tensor([n_local], dtype=torch.long, device=device),
            cell=cell[None] if plan.pbc is not None else None,
            pbc=plan.pbc[None] if plan.pbc is not None else None,
            total_charge=plan.total_charge, spin_multiplicity=plan.spin_multiplicity,
            compute_dtype=model_dtype if model_dtype is not None and model_dtype != dtype else None)
        with self._hooks(plan, ties):
            out = self.model(graph)
        energy = out["node_energy"][:plan.n_owned].to(torch.float64).sum()
        for t in ties:
            energy = energy + _tie(t)
        inputs = [p] + ([strain] if strain is not None else [])
        grads = [None] * len(inputs)
        if energy.requires_grad:
            grads = list(torch.autograd.grad([energy], inputs, allow_unused=True))
        g_pos = grads[0] if grads[0] is not None else torch.zeros_like(p)
        g_strain = (grads[1] if strain is not None and grads[1] is not None
                    else torch.zeros((3, 3), dtype=torch.float64, device=device))
        total = torch.cat([energy.detach().reshape(1), g_strain.to(torch.float64).reshape(9)])
        if plan.world > 1:
            total = _all_reduce(total, self.group)
        result = {"energy": total[:1], "forces": -g_pos.to(dtype)}
        if strain is not None:
            result["stress"] = (total[1:10].view(1, 3, 3) / volume).to(dtype)
        return result

    def forward(self, data: AtomicGraph) -> dict:
        """Energy, forces and stress of the whole structure, given on every process.

        The plan of the previous call is reused while it holds (:meth:`holds`).

        Parameters
        ----------
        data : AtomicGraph
            One structure, the same on every process.

        Returns
        -------
        dict of str to Tensor
            ``"energy"`` ``(1,)`` float64, ``"forces"`` ``(N, 3)`` in the
            dtype of the positions and, for a periodic structure with
            ``compute_stress``, ``"stress"`` ``(1, 3, 3)``.
        """
        periodic = data.cell is not None and data.pbc is not None and bool(data.pbc.any())
        cell = data.cell[0] if periodic else None
        plan = self._plan
        if (plan is None or plan.n_atoms != data.num_nodes or (plan.cell is None) == periodic
                or not torch.equal(plan.atomic_numbers, data.atomic_numbers)
                or not self.holds(plan, data.pos[plan.owned], cell)):
            plan = self._plan = self.plan(data)
        plan.total_charge, plan.spin_multiplicity = data.total_charge, data.spin_multiplicity
        out = self.evaluate(plan, data.pos[plan.owned], cell)
        out["forces"] = self._gather(plan, out["forces"])
        return out
