"""Spatial domain decomposition: one large structure evaluated on several GPUs.

The box is cut into a grid of sub-domains, one per process. Every process holds
all positions, evaluates the atoms of its own sub-domain (the owned atoms) and
copies of every atom within the model's receptive field of the sub-domain (the
ghosts), and keeps the energies of its owned atoms only. Those energies are
exact: an atom's energy depends on atoms within the receptive field only, and
the ghosts that are themselves evaluated wrongly (at the outer edge of the
shell) are too far away to reach an owned atom. Forces and stress are the
derivatives of the owned energies with respect to the shared positions and
strain, summed over the processes.

Periodic boxes and molecules are both supported. Along a periodic axis that is
not cut, the sub-domains keep the box's periodicity; along a cut axis, the
ghosts are explicit periodic images.

Launch one process per GPU with ``torchrun``::

    import torch, torch.distributed as dist
    from xnn.common.deploy import DomainParallel

    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    dp = DomainParallel(model.cuda().eval())
    out = dp(graph.to("cuda"))     # every process gets energy, forces, stress
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
from ..models.outputs import ForceStressOutput, _stress

#: distance added to the receptive field, so that rounding never drops an atom on its edge
HALO_MARGIN = 1e-6


@dataclass
class DomainPlan:
    """The atoms one process evaluates and their neighbor list.

    Attributes
    ----------
    grid : tuple of int
        Sub-domains along the three axes.
    rank : int
        The process (sub-domain) the plan belongs to.
    index : Tensor
        Atom of the structure behind every local atom ``(L,)``, owned atoms first.
    shifts : Tensor or None
        Integer periodic image of every local atom ``(L, 3)`` relative to the given
        positions; ``None`` for a molecule.
    n_owned : int
        Number of owned atoms (the first ``n_owned`` local atoms).
    edge_index, cell_shifts : Tensor
        Neighbor list of the local atoms, as in :class:`~xnn.common.data.AtomicGraph`.
    pbc : Tensor or None
        Periodicity of the local atoms ``(3,)``: the periodic axes that are not cut.
        ``None`` when no axis stays periodic.
    n_atoms : int
        Number of atoms of the structure.
    """

    grid: tuple
    rank: int
    index: Tensor
    shifts: Optional[Tensor]
    n_owned: int
    edge_index: Tensor
    cell_shifts: Tensor
    pbc: Optional[Tensor]
    n_atoms: int

    @property
    def n_local(self) -> int:
        """int : Number of local atoms (owned and ghosts)."""
        return int(self.index.shape[0])


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


@contextlib.contextmanager
def _whole_structure_sums(n_owned: int, world: int, group):
    """Make :func:`~xnn.common.models.ops.whole_structure_sum` sum the owned atoms of every process."""
    def reduce(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
        part = ops.scatter_sum(src[:n_owned], index[:n_owned], dim_size)
        if world > 1:
            from torch.distributed.nn.functional import all_reduce
            part = all_reduce(part, group=group if group is not None else dist.group.WORLD)
        return part
    previous = ops._DOMAIN_REDUCTION[0]
    ops._DOMAIN_REDUCTION[0] = reduce
    try:
        yield
    finally:
        ops._DOMAIN_REDUCTION[0] = previous


class DomainParallel(nn.Module):
    """Evaluate one structure split over the processes of a distributed launch.

    Every process calls the module with the whole structure (an
    :class:`~xnn.common.data.AtomicGraph` of one structure; its neighbor list is
    not used) and gets the energy, the forces of all atoms and the stress, the
    same on every process and equal to a single-process evaluation to rounding.
    Each process evaluates only its sub-domain and a shell of ghost atoms as thick
    as the model's receptive field, so the memory per process falls with the
    number of processes. The model runs as it would on one device, fast paths
    included. Evaluation only: no second derivatives.

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
        Thickness of the ghost shell in Angstrom, by default the model's
        :meth:`receptive_field`.
    compute_stress : bool, optional
        Whether to compute the stress of a periodic structure, by default ``True``.
    group : torch.distributed.ProcessGroup, optional
        The processes that share the structure, by default all of them.

    Raises
    ------
    ValueError
        For a model without a receptive field.
    """

    def __init__(self, model: nn.Module, grid: Optional[Sequence[int]] = None,
                 receptive_field: Optional[float] = None, compute_stress: bool = True,
                 group=None):
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
        self.model = model
        self.halo = float(receptive_field) + HALO_MARGIN
        self.cutoff = float(model.cutoff)
        self.grid = tuple(int(g) for g in grid) if grid is not None else None
        self.compute_stress = compute_stress
        self.group = group

    def _world(self) -> tuple:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(self.group), dist.get_world_size(self.group)
        return 0, 1

    @torch.no_grad()
    def plan(self, data: AtomicGraph, rank: Optional[int] = None,
             world_size: Optional[int] = None) -> DomainPlan:
        """Partition the structure and build the neighbor list of one process.

        A plan holds for the positions it was built from. :meth:`forward` builds
        a new one when it is given none.

        Parameters
        ----------
        data : AtomicGraph
            One structure, periodic along all three axes or along none.
        rank, world_size : int, optional
            The process and the number of processes, by default those of the
            launch (0 and 1 without one).

        Returns
        -------
        DomainPlan
            The local atoms and their neighbor list.

        Raises
        ------
        ValueError
            For a batch of several structures, a partially periodic structure or
            a grid that does not match the number of processes.
        """
        own_rank, own_world = self._world()
        rank = own_rank if rank is None else int(rank)
        world = own_world if world_size is None else int(world_size)
        if data.num_graphs != 1:
            raise ValueError("DomainParallel evaluates one structure, not a batch")
        periodic = data.cell is not None and data.pbc is not None and bool(data.pbc.any())
        if periodic and not bool(data.pbc.all()):
            raise ValueError("DomainParallel takes structures periodic along all axes or none")
        pos = data.pos.detach().to(torch.float64)
        device, n = pos.device, int(pos.shape[0])
        cell = None
        if periodic:
            cell = data.cell[0].detach().to(torch.float64)
            inverse = torch.linalg.inv(cell)
            raw = pos @ inverse
            wrap = torch.floor(raw)
            frac = raw - wrap
            widths = 1.0 / torch.linalg.norm(inverse, dim=0)
        else:
            low = pos.min(0).values
            widths = pos.max(0).values - low + 1e-6
            frac = (pos - low) / widths
        widths = [float(w) for w in widths]
        grid = self.grid if self.grid is not None else choose_grid(world, widths, self.halo)
        if len(grid) != 3 or math.prod(grid) != world:
            raise ValueError(f"grid {tuple(grid)} does not hold {world} processes")
        coords = (rank // (grid[1] * grid[2]), (rank // grid[2]) % grid[1], rank % grid[2])

        g = torch.tensor(grid, device=device)
        owner = torch.minimum(torch.floor(frac * g).long().clamp(min=0), g - 1)
        owned = (owner == torch.tensor(coords, device=device)).all(1)
        masks, steps = [], []
        for a in range(3):
            if grid[a] == 1:
                masks.append(torch.ones(n, 1, dtype=torch.bool, device=device))
                steps.append(torch.zeros(1, dtype=torch.long, device=device))
                continue
            shell = self.halo / widths[a]
            low, high = coords[a] / grid[a] - shell, (coords[a] + 1) / grid[a] + shell
            m = math.floor(shell) + 1 if periodic else 0
            image = torch.arange(-m, m + 1, device=device)
            x = frac[:, a, None] + image.to(torch.float64)
            masks.append((x >= low) & (x < high))
            steps.append(image)
        mask = masks[0][:, :, None, None] & masks[1][:, None, :, None] & masks[2][:, None, None, :]
        atom, i0, i1, i2 = mask.nonzero(as_tuple=True)
        image = torch.stack([steps[0][i0], steps[1][i1], steps[2][i2]], dim=1)
        own = owned[atom] & (image == 0).all(1)
        order = torch.cat([own.nonzero().flatten(), (~own).nonzero().flatten()])
        atom, image, n_owned = atom[order], image[order], int(own.sum())

        shifts = None
        local = pos[atom]
        if periodic:
            shifts = image - wrap[atom].long()
            local = local + shifts.to(torch.float64) @ cell
        native = [periodic and grid[a] == 1 for a in range(3)]
        if any(native):
            # the cut axes stretched far beyond the shell, so that only the explicit
            # images are neighbors along them, while the others keep the box's periodicity
            search = cell.clone()
            for a in range(3):
                if not native[a]:
                    search[a] *= 1.0 + 1.0 / grid[a] + 2.0 * (self.halo + self.cutoff) / widths[a]
            edge_index, cell_shifts = build_neighbor_list(local, self.cutoff, search,
                                                          torch.ones(3, dtype=torch.bool))
            cut = torch.tensor([not x for x in native], device=cell_shifts.device)
            if bool((cell_shifts[:, cut] != 0).any()):
                raise RuntimeError("a neighbor across a cut axis was found through the box's periodicity")
            pbc = torch.tensor(native)
        else:
            edge_index, cell_shifts = build_neighbor_list(local, self.cutoff)
            pbc = None
        return DomainPlan(grid=tuple(grid), rank=rank, index=atom, shifts=shifts, n_owned=n_owned,
                          edge_index=edge_index.to(device), cell_shifts=cell_shifts.to(device),
                          pbc=pbc, n_atoms=n)

    def local_terms(self, data: AtomicGraph, plan: DomainPlan) -> tuple:
        """The owned energy of one process and its derivatives, before the sum over processes.

        Parameters
        ----------
        data : AtomicGraph
            The structure the plan was built for.
        plan : DomainPlan
            The process's plan.

        Returns
        -------
        tuple of Tensor
            The owned energy ``()``, its gradient with respect to the positions
            ``(N, 3)`` and with respect to the strain ``(3, 3)`` (zero without
            stress), all float64.
        """
        if plan.n_atoms != data.num_nodes:
            raise ValueError("the plan was built for another structure")
        pos = data.pos.detach().requires_grad_(True)
        dtype, device = pos.dtype, pos.device
        cell = data.cell[0].detach().to(dtype) if plan.shifts is not None else None
        strain = None
        p = pos
        if self.compute_stress and cell is not None:
            strain = torch.zeros((3, 3), dtype=dtype, device=device, requires_grad=True)
            sym = 0.5 * (strain + strain.transpose(0, 1))
            p = pos + pos @ sym
            cell = cell + cell @ sym
        local = p[plan.index]
        if plan.shifts is not None:
            local = local + plan.shifts.to(dtype) @ cell
        n_local = plan.n_local
        model_dtype = _model_dtype(self.model)
        graph = AtomicGraph(
            pos=local, atomic_numbers=data.atomic_numbers[plan.index],
            edge_index=plan.edge_index, cell_shifts=plan.cell_shifts,
            batch=torch.zeros(n_local, dtype=torch.long, device=device),
            n_atoms=torch.tensor([n_local], dtype=torch.long, device=device),
            cell=cell[None] if plan.pbc is not None else None,
            pbc=plan.pbc[None] if plan.pbc is not None else None,
            total_charge=data.total_charge, spin_multiplicity=data.spin_multiplicity,
            compute_dtype=model_dtype if model_dtype is not None and model_dtype != dtype else None)
        _, world = self._world()
        with _whole_structure_sums(plan.n_owned, world, self.group):
            out = self.model(graph)
        energy = out["node_energy"][:plan.n_owned].to(torch.float64).sum()
        inputs = [pos] + ([strain] if strain is not None else [])
        grads = [None] * len(inputs)
        if energy.requires_grad:
            grads = list(torch.autograd.grad([energy], inputs, allow_unused=True))
        g_pos = grads[0] if grads[0] is not None else torch.zeros_like(pos)
        g_strain = (grads[1] if strain is not None and grads[1] is not None
                    else torch.zeros((3, 3), dtype=dtype, device=device))
        return energy.detach(), g_pos.to(torch.float64), g_strain.to(torch.float64)

    def forward(self, data: AtomicGraph, plan: Optional[DomainPlan] = None) -> dict:
        """Energy, forces and stress of the whole structure.

        Parameters
        ----------
        data : AtomicGraph
            One structure, the same on every process.
        plan : DomainPlan, optional
            This process's plan for these positions (:meth:`plan`), built when
            not given.

        Returns
        -------
        dict of str to Tensor
            ``"energy"`` ``(1,)`` float64, ``"forces"`` ``(N, 3)`` in the
            dtype of the positions and, for a periodic structure with
            ``compute_stress``, ``"stress"`` ``(1, 3, 3)``.
        """
        if plan is None:
            plan = self.plan(data)
        energy, g_pos, g_strain = self.local_terms(data, plan)
        n = data.num_nodes
        total = torch.cat([energy.reshape(1), g_strain.reshape(9), g_pos.reshape(-1)])
        _, world = self._world()
        if world > 1:
            dist.all_reduce(total, group=self.group)
        out = {"energy": total[:1], "forces": (-total[10:]).view(n, 3).to(data.pos.dtype)}
        if self.compute_stress and plan.shifts is not None:
            out["stress"] = _stress(total[1:10].view(1, 3, 3).to(data.pos.dtype), data)
        return out
