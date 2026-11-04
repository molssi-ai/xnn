"""Molecular-dynamics steps of the classical force fields (OPLS, AMBER, CHARMM, DREIDING) on their kernels.

A :class:`ClassicalStepper` evaluates the energy and forces of one structure at a sequence of positions. It
keeps a neighbor list built with ``cutoff + skin`` while no atom has moved by half the skin, and with it
the model's kernel terms (:func:`~xnn.ffnn.models.classical_fast.evaluate_spec`: parameter tables, term
atoms) and the image shifts of the bonded vectors; a step then runs the kernels alone, as a
CUDA graph replay once recorded. The list keeps each pair once and no excluded pair; the pairs beyond the
cutoff are skipped by the pair kernel. The numbers are those of the model's fast evaluation (and of its eager
reference to round-off).
"""
import dataclasses
from typing import Optional

import torch
from torch import Tensor

from xnn.common.data import structure_to_graph
from xnn.common.data.neighborlist import cell_grid, unique_pairs

from .classical_fast import exclusion_rows, image_shifts
from .geometry import TINY, pair_image_shifts


def _charges(model, B: int) -> Tensor:
    """The per-atom charges a model evaluates ``(B n,)``."""
    from .dreiding import Dreiding
    from .opls import OPLS
    if isinstance(model, OPLS):
        return model.ff.charges[model.top_type.repeat(B)]
    if isinstance(model, Dreiding):
        return model.charge.repeat(B)
    return model.charges.repeat(B)


def _spec(model, B: int, n: int, q: Tensor) -> dict:
    from .charmm import Charmm
    from .charmmlib import KCAL_TO_EV
    if isinstance(model, Charmm):
        return model._fast_spec(B, n, q, model.coulomb_constant * KCAL_TO_EV, model.ff.pair_tables())
    return model._fast_spec(B, n, q)


def _shortest_width(g) -> float:
    """The smallest distance between opposite faces of the cell over its periodic axes (``inf`` without
    one): no lattice vector is shorter."""
    if g.cell is None or g.pbc is None or not bool(g.pbc.any()):
        return float("inf")
    h = g.cell[0].double()
    volume = torch.det(h).abs()
    widths = [float(volume / torch.linalg.norm(torch.cross(h[(a + 1) % 3], h[(a + 2) % 3], dim=0)))
              for a in range(3) if bool(g.pbc[0, a])]
    return min(widths)


def _unsupported(model) -> Optional[str]:
    """Why ``model`` has terms outside the kernels, or ``None``."""
    if not hasattr(model, "_fast_spec"):
        return f"{type(model).__name__} has no kernel terms"
    return None


class ClassicalStepper:
    """Energy and forces of one structure along a trajectory from the kernels of a classical force field.

    Parameters
    ----------
    model : nn.Module
        An :class:`~xnn.ffnn.models.OPLS`, :class:`~xnn.ffnn.models.Amber`, :class:`~xnn.ffnn.models.Charmm`
        or :class:`~xnn.ffnn.models.Dreiding` model on a CUDA device (or a wrapper with a ``model``
        attribute), in evaluation mode with no trained parameter.
    skin : float, optional
        Margin of the neighbor list beyond the model's cutoff in Angstrom; by default 1.0.
    capture : bool, optional
        Replay the steps as a CUDA graph after ``warmup`` plain steps; by default ``True``.
    warmup : int, optional
        Plain steps before the graph is recorded (the kernels are compiled in the first); by default 2.

    Attributes
    ----------
    rebuilds : int
        Neighbor lists built.
    refills : int
        Of those, the lists copied into the buffers of the recorded graph (no new recording).
    replays : int
        Steps replayed from the CUDA graph.
    """

    def __init__(self, model, skin: float = 1.0, capture: bool = True, warmup: int = 2):
        self.model = getattr(model, "model", model)
        reason = _unsupported(self.model)
        if reason is not None:
            raise ValueError(f"the stepper runs on the kernels only: {reason}")
        self.skin = float(skin)
        self.capture, self.warmup = bool(capture), int(warmup)
        self.rebuilds = self.replays = self.refills = 0
        self.graph = None
        self.cuda_graph = None
        self.stream = None
        self.spec = None
        self.base = None
        self.plan = None
        self.moves = None
        self.images = False
        self.bonded_images = None
        self.ptr = None
        self.work = {}

    def _graph(self, struct: dict, device: torch.device, dtype: torch.dtype, rows: Optional[tuple] = None):
        """The structure with the pairs within ``cutoff + skin`` once each, ``i < k``, sorted by ``i``
        (:func:`~xnn.common.data.neighborlist.unique_pairs`: an atom's own images left out), and whether the
        excluded pairs of ``rows`` are already left out. On a CPU a grid search gives them as 32-bit atoms, in one
        pass with the exclusions and image indices for shifts (:func:`.classical_cpu.stepper_pairs`, ``images``
        set; every pair tested in a cell too small for a grid) where it can, else with 8-bit shifts; that list is
        read by rows (``self.staged``, ``self.rows``: no first atoms, a pair in the row of either of its atoms)."""
        rc = float(self.model.cutoff) + self.skin
        if self.base is None:
            self.base = structure_to_graph(struct, rc, device=device)
            pos = self.base.pos
            width = _shortest_width(self.base)
            if width <= rc:
                raise ValueError(f"the stepper's list has no periodic images of an atom itself: the cell's "
                                 f"width {width:.3f} A must exceed cutoff + skin = {rc:.3f} A (evaluate the "
                                 f"model instead)")
        else:
            pos = torch.as_tensor(struct["pos"]).to(device=device, dtype=self.base.pos.dtype)
        base = self.base
        cell, pbc = None if base.cell is None else base.cell[0], None if base.pbc is None else base.pbc[0]
        grid = None
        if device.type == "cpu":
            # the grid of the cell, the same from one list to the next while the cell stays
            key = (None if cell is None else tuple(cell.reshape(-1).tolist()), rc, int(pos.shape[0]))
            if getattr(self, "grid_key", None) != key:
                self.grid_key, self.grid = key, cell_grid(pos, rc, cell, pbc)
            grid = self.grid
        self.images = False
        self.ptr = None
        self.rows = self.staged = None
        filtered = False
        found = None
        search = grid
        if grid is None and device.type == "cpu":
            # a cell with no room for a grid: every pair tested
            from .classical_cpu import small_cell
            search = small_cell(pos, rc, cell, pbc)
        if search is not None and (not isinstance(search, tuple) or len(search) == 4):
            from .classical_cpu import stepper_pairs
            if rows is not None and int(rows[0].shape[0]) != int(pos.shape[0]) + 1:
                rows = None
            # each row's pairs within the cutoff first
            # (the partners left in the search's slots, read by rows)
            found = stepper_pairs(pos, rc, search, rows, inner=float(self.model.cutoff), work=self.work,
                                  compact=False, half=True)
        if found is not None:
            _, k, shifts, self.ptr, self.rows = found
            self.staged = (k, shifts)
            self.images, filtered = True, rows is not None
        elif grid is not None:
            from xnn.common.data.grid_pairs import grid_pairs
            i, k, shifts = grid_pairs(pos, rc, *grid, index_dtype=torch.int32, shift_dtype=torch.int8)
        else:
            i, k, shifts = unique_pairs(pos, rc, cell, pbc)
        if found is not None:
            # no list of pairs: the stepper reads its rows (self.staged, self.rows)
            g = dataclasses.replace(base, pos=pos, edge_index=torch.zeros(2, 0, dtype=torch.int32), cell_shifts=None)
        else:
            g = dataclasses.replace(base, pos=pos, edge_index=torch.stack([i, k]), cell_shifts=shifts)
        g.compute_dtype = dtype
        # the base of the next list (not the first, directed list)
        self.base = g
        return g, filtered

    def _build(self, struct: dict) -> None:
        """A neighbor list at the current positions, and the image shifts of the bonded vectors. The model's
        kernel terms are made once; while the new list fits the buffers of the recorded graph (their capacity
        is 1.2 times the first list), it is copied into them (the padding pairs ``0, 0`` are inactive) and the
        graph replays on; otherwise the buffers are made again and the graph is recorded anew."""
        model = self.model
        device = next(model.buffers()).device
        dtype = model.ff.params[next(iter(model.ff.params))].dtype
        if device.type != "cpu":
            self.plan = self.moves = None
        if self.spec is None:
            n = int(model.top_type.shape[0])
            with torch.no_grad():
                q = _charges(model, 1).to(dtype)
                self.spec = _spec(model, 1, n, q)
                ref = getattr(model, "atom_ref", None)
                z = torch.as_tensor(struct["atomic_numbers"]).to(device)
                self.const = 0.0 if ref is None else float(ref(z).sum())
        bonded, p14, nb = self.spec["bonded"], self.spec.get("pairs14"), self.spec["nonbonded"]
        _, compact, qn, table, ke, excl, n_local, cutoff, width, form = nb
        rows = exclusion_rows(model, excl, n_local) if excl.shape[1] > 0 and device.type == "cpu" else None
        g, filtered = self._graph(struct, device, dtype, rows)
        with torch.no_grad():
            terms = [(t[2], t[1]) for t in bonded] + ([(p14[1], "bond")] if p14 is not None else [])
            if self.images and terms:
                # one orthorhombic cell (the grid search's): the shifts from one launch
                from .classical_cpu import BondedImages
                if self.bonded_images is None:
                    self.bonded_images = BondedImages(g.pos, terms)
                found = self.bonded_images(g.pos, torch.diagonal(g.cell[0]).to(g.pos.dtype))
            else:
                found = image_shifts(g, terms)
            # the list's pairs once each (i < k), the excluded ones dropped: a step tests neither
            src, dst = g.edge_index[0], g.edge_index[1]
            if self.staged is not None:
                src, dst = None, self.staged[0]
            out = None
            if excl.shape[1] > 0 and not filtered:
                from .classical_kernels import excluded
                out = excluded(src, dst, exclusion_rows(model, excl, n_local), n_local)
                keep = (~out).nonzero()[:, 0]
                src, dst = src[keep], dst[keep]
            shifts = None if g.cell is None else (g.cell_shifts if out is None else g.cell_shifts[keep])
            if self.staged is not None:
                shifts = self.staged[1]
            if shifts is not None and device.type != "cpu":
                shifts = shifts.to(g.pos.dtype)
            K = int(self.ptr[-1]) if self.staged is not None else int(src.shape[0])
            hb = self._triplets(g, out)
            H = 0 if hb is None else (hb[2] if len(hb) == 3 else int(hb[0].shape[1]))
            if self.cuda_graph is not None and K <= self.capacity and (hb is None or H <= self.hb_capacity):
                if hb is not None:
                    self._fill_triplets(hb)
                self.src[:K].copy_(src)
                self.src[K:].zero_()
                self.dst[:K].copy_(dst)
                self.dst[K:].zero_()
                if shifts is not None:
                    self.shifts[:K].copy_(shifts)
                    self.shifts[K:].zero_()
                for (_, _, _, _, mine), new in zip(self.bonded, found):
                    if mine is not None:
                        mine.copy_(new)
                if self.p14 is not None and self.p14[1] is not None:
                    self.p14[1].copy_(found[-1][0])
                self.reference.copy_(g.pos)
                self.rebuilds += 1
                self.refills += 1
                return
        if device.type == "cpu":
            # no graph to replay: the list itself, as 32-bit atoms and 8-bit shifts (what the CPU kernels stream)
            self.capacity = K
            small = shifts is not None and (shifts.dtype == torch.int8 or K == 0 or float(shifts.abs().max()) <= 127)
            self.src, self.dst = None if src is None else src.to(torch.int32), dst.to(torch.int32)
            self.n_pairs = K
            self.shifts = None if shifts is None else shifts.to(torch.int8 if small or self.images else shifts.dtype)
        else:
            cap = int(1.2 * K) + 64
            self.capacity = cap
            self.src = torch.zeros(cap, dtype=src.dtype, device=device)
            self.dst = torch.zeros(cap, dtype=dst.dtype, device=device)
            self.src[:K], self.dst[:K] = src, dst
            self.shifts = None
            if shifts is not None:
                self.shifts = torch.zeros(cap, 3, dtype=shifts.dtype, device=device)
                self.shifts[:K] = shifts
        old = getattr(self, "pos", None)
        if device.type == "cpu" and old is not None and old.shape == g.pos.shape and old.dtype == g.pos.dtype:
            # the same buffers from one list to the next (a step's prepared launches read them)
            with torch.no_grad():
                self.pos.copy_(g.pos)
                self.reference.copy_(g.pos)
        else:
            self.pos = g.pos.detach().clone().requires_grad_(True)
            self.reference = self.pos.detach().clone()
            self.moves = None
        self.cell = None if g.cell is None else g.cell.detach().clone()
        self.batch = g.batch
        # the CPU's bonded shifts stay in the buffer of their kernel (BondedImages)
        keep = device.type == "cpu" and self.images
        self.bonded = [(idx, kind, tuple(p.detach() for p in params), options or {},
                        None if found_j is None else (found_j if keep else found_j.clone()))
                       for (_, kind, idx, params, options), found_j in zip(bonded, found)]
        self.p14 = None
        if p14 is not None:
            _, pairs, compact14, q14, table14, ke14 = p14
            s14 = None if found[-1] is None else (found[-1][0] if keep else found[-1][0].clone())
            self.p14 = (pairs, s14, compact14, q14.detach(), table14.detach(), ke14)
        self.nb = (n_local, compact, qn.detach(), table.detach(), ke, cutoff, width, form)
        old, self.hb = getattr(self, "hb", None), None
        if hb is not None and device.type == "cpu":
            # the triplets in buffers kept from one list to the next (5 % room, the padding inert: D0 = 0), so the
            # plan's launches run on after a rebuild
            if old is not None and self.hb_capacity >= H and (old[3] is None) == (hb[1] is None) \
                    and (hb[0] is old[0] or old[0].dtype == hb[0].dtype):
                self.hb = old
            else:
                self.hb_capacity = cap_h = int(1.05 * H) + 64
                self.hb = (torch.zeros(3, cap_h, dtype=torch.long, device=device),
                           torch.zeros(cap_h, dtype=dtype, device=device),
                           torch.zeros(cap_h, dtype=dtype, device=device),
                           None if hb[1] is None else torch.zeros(2, cap_h, 3, dtype=hb[1].dtype, device=device),
                           (model.hbond_cutoff ** 2, model.hbond_cos))
            self._fill_triplets(hb)
        elif hb is not None:
            self.hb_capacity = int(1.2 * H) + 64
            cap_h = self.hb_capacity
            self.hb = (torch.zeros(3, cap_h, dtype=torch.long, device=device),
                       torch.zeros(cap_h, dtype=dtype, device=device), torch.zeros(cap_h, dtype=dtype, device=device),
                       None if hb[1] is None else torch.zeros(2, cap_h, 3, dtype=hb[1].dtype, device=device),
                       (model.hbond_cutoff ** 2, model.hbond_cos))
            self._fill_triplets(hb)
        self.dtype = dtype
        self.graph = g
        self.cuda_graph = None
        self.steps = 0
        if device.type == "cpu":
            # the plan of the previous list runs on as long as it reads the same buffers
            key = self._plan_key()
            if key != getattr(self, "plan_key", None):
                self.plan = None
            self.plan_key = key
        self.rebuilds += 1

    def _plan_key(self) -> tuple:
        """What a CPU plan's launches depend on beyond the contents of their buffers: the buffers themselves, the
        cell and whether there are pairs."""
        def ptr(t):
            return None if t is None else (t.data_ptr(), tuple(t.shape), t.dtype)

        def at(t):
            # a list's view of a kept buffer: its length changes with the list, the launches do not
            return None if t is None else (t.data_ptr(), t.dtype)
        cell = None if self.cell is None else tuple(self.cell.reshape(-1).tolist())
        rows = None if self.rows is None else tuple(ptr(r) for r in self.rows)
        return (ptr(self.pos), cell, at(self.dst), at(self.shifts), ptr(self.ptr), rows, self.n_pairs > 0,
                tuple(ptr(b[4]) for b in self.bonded), None if self.p14 is None else ptr(self.p14[1]),
                None if self.hb is None else tuple(ptr(x) for x in self.hb[:4]), self.dtype)

    def _triplets(self, g, out: Optional[Tensor]) -> Optional[tuple]:
        """The DREIDING hydrogen-bond triplets ``(donor, H, acceptor)`` of the list (its pairs ``i < k`` in
        both directions within the hydrogen-bond cutoff plus the skin, not excluded (``out``: the excluded
        ones), once per hydrogen of the donor) with the image shifts of their vectors, or ``None`` for a model
        without them."""
        model = self.model
        hyd = getattr(model, "hbond_hydrogens", None)
        if hyd is None or not model.use_hbond or not bool((hyd >= 0).any()):
            return None
        # the pairs of a donor (an atom with hydrogens) and an acceptor in either direction first, their vectors
        # after: the directed list of both directions, forward pairs then reversed ones
        i, k = g.edge_index[0], g.edge_index[1]
        if self.images and out is None:
            # the rows of the CPU grid search's list (one orthorhombic cell): two launches
            from .classical_cpu import hbond_triplets
            kept = getattr(self, "hb", None)
            # written into the kept buffers where they fit
            k, img = self.staged
            return hbond_triplets(self.rows, k, img, g.pos.detach(), torch.diagonal(g.cell[0]), hyd,
                                  model.hbond_acceptor, model.hbond_cutoff + self.skin,
                                  None if kept is None else (kept[0], kept[3]))
        E = int(i.shape[0])
        donor, acceptor = hyd[:, 0] >= 0, model.hbond_acceptor
        fwd, bwd = donor[i] & acceptor[k], donor[k] & acceptor[i]
        sel = (fwd | bwd).nonzero()[:, 0]
        def shifts(rows):
            # the integer shifts of these pairs (decoded from image indices where the list holds those)
            if self.images:
                from .classical_cpu import decode_images
                return decode_images(g.cell_shifts[rows])
            return g.cell_shifts[rows]
        vec = g.pos[k[sel]] - g.pos[i[sel]]
        if g.cell is not None:
            vec = vec + shifts(sel).to(vec.dtype) @ g.cell[0].to(vec.dtype)
        near = (vec * vec).sum(1) < (model.hbond_cutoff + self.skin) ** 2
        if out is not None:
            near = near & ~out[sel]
        pick = torch.cat([sel[near & fwd[sel]], sel[near & bwd[sel]] + E])
        back = pick >= E
        e = torch.where(back, pick - E, pick)
        src, dst = torch.where(back, k[e], i[e]), torch.where(back, i[e], k[e])
        h_all = hyd[src].reshape(-1)
        keep = h_all >= 0
        rep = int(hyd.shape[1])
        idx = torch.stack([src.repeat_interleave(rep)[keep], h_all[keep], dst.repeat_interleave(rep)[keep]])
        if g.cell is None:
            return idx, None
        s_hd = pair_image_shifts(g, idx[1], idx[0])
        if s_hd is None:
            s_hd = torch.zeros(0, 3, dtype=g.pos.dtype, device=g.pos.device)
        # the donor-acceptor vector takes the list's image (a cell under twice the cutoff has several)
        sh = shifts(e).to(s_hd.dtype)
        s_da = torch.where(back[:, None], -sh, sh).repeat_interleave(rep, 0)[keep]
        return idx, torch.stack([s_da, s_hd])

    def _fill_triplets(self, hb: tuple) -> None:
        """Copy the triplets into the buffers (the padding gets ``D0 = 0``: no energy, no force); triplets
        written into the buffers already ``(idx, shifts, H)`` only set the ``D0`` of their columns."""
        idx, shifts = hb[0], hb[1]
        H = hb[2] if len(hb) == 3 else int(idx.shape[1])
        buf, d0, r0, sh, _ = self.hb
        P = self.model.ff.params
        d0[:H].fill_(float(P["hbond_d0"]))
        d0[H:].zero_()
        r0.fill_(float(P["hbond_r0"]))
        if idx is buf:
            return
        buf[:, :H].copy_(idx[:, :H])
        buf[:, H:].zero_()
        if sh is not None:
            sh[:, :H].copy_(shifts[:, :H])
            sh[:, H:].zero_()

    def _run(self) -> tuple:
        from . import classical_kernels as kk
        dtype = self.dtype
        bound = min(0.9999999999, 1.0 - 4.0 * torch.finfo(dtype).eps)
        pos, cell, batch = self.pos, self.cell, self.batch
        total = torch.zeros((), dtype=torch.float64, device=pos.device)
        for idx, kind, params, options, shifts in self.bonded:
            total = total + kk.bonded_terms(pos, cell, batch, idx, shifts, kind, params, dtype, TINY, bound,
                                            options).double().sum()
        if self.p14 is not None:
            pairs, shifts, compact, q, table, ke = self.p14
            e_v, e_c = kk.pair_terms(pos, cell, batch, pairs[0], pairs[1], shifts, None, 1, compact, q, table, dtype,
                                     ke, TINY, 1.0e15, 0.0, 0)
            total = total + e_v.double().sum() + e_c.double().sum()
        if self.hb is not None:
            idx, d0, r0, sh, extra = self.hb
            total = total + kk.bonded_terms(pos, cell, batch, idx, sh, "hbond", (d0, r0), dtype, TINY, bound, {},
                                            extra).double().sum()
        n_local, compact, q, table, ke, cutoff, width, form = self.nb
        e_v, e_c = kk.pair_terms(pos, cell, batch, self.src, self.dst, self.shifts, None, n_local, compact, q, table,
                                 dtype, ke, TINY, cutoff, width, form)
        total = total + e_v.double().sum() + e_c.double().sum()
        forces = -torch.autograd.grad(total, pos)[0]
        return total + self.const, forces

    def _cpu_plan(self):
        """The launches of :meth:`_run` on the CPU, prepared once per list (:class:`.classical_cpu.Plan`)."""
        from .classical_cpu import Plan
        dtype = self.dtype
        bound = min(0.9999999999, 1.0 - 4.0 * torch.finfo(dtype).eps)
        plan = Plan(self.pos.detach(), self.cell, self.batch, dtype)
        for idx, kind, params, options, shifts in self.bonded:
            plan.bonded(idx, shifts, kind, params, TINY, bound, options)
        if self.p14 is not None:
            pairs, shifts, compact, q, table, ke = self.p14
            plan.pairs(pairs[0], pairs[1], shifts, 1, compact, q, table, ke, TINY, 1.0e15, 0.0, 0, live=True)
        if self.hb is not None:
            idx, d0, r0, sh, extra = self.hb
            plan.bonded(idx, sh, "hbond", (d0, r0), TINY, bound, {}, extra)
        n_local, compact, q, table, ke, cutoff, width, form = self.nb
        plan.pairs(self.src, self.dst, self.shifts, n_local, compact, q, table, ke, TINY, cutoff, width, form,
                   by_rows=True, images=self.images, ptr=self.ptr, rows=self.rows)
        return plan

    def _cpu_step(self, struct: dict, pos: Tensor, energy: bool = True) -> dict:
        """A step on the CPU: the positions copied and tested by one kernel (:class:`.classical_cpu.Moves`), then
        the plan's launches."""
        from .classical_cpu import Moves
        new = pos.to(dtype=self.pos.dtype).contiguous()
        if self.moves is None:
            self.moves = Moves(new, self.pos.detach(), self.reference, 0.5 * self.skin)
        if self.moves(new):
            self._build(struct)
            with torch.no_grad():
                self.pos.copy_(new)
        if self.plan is None:
            self.plan = self._cpu_plan()
        e, forces = self.plan(energy)
        self.steps += 1
        if e is None:
            return {"forces": forces}
        return {"energy": (e + self.const).reshape(1), "forces": forces}

    def _outdated(self, pos: Tensor) -> bool:
        step = pos.to(self.reference.dtype) - self.reference
        return bool((step * step).sum(-1).max() > (0.5 * self.skin) ** 2)

    def __call__(self, struct: dict, energy: bool = True) -> dict:
        """The energy ``(1,)`` (float64) and forces ``(N, 3)`` at the positions of ``struct``.

        Parameters
        ----------
        struct : dict
            ``pos``, ``atomic_numbers`` and, for a cell, ``cell`` and ``pbc`` (arrays or tensors); the cell
            and the atoms are those of the first call.
        energy : bool, optional
            ``False``: the forces alone where that saves work (on a CPU the kernels then sum no energy; the
            result has no ``"energy"``). By default ``True``.
        """
        pos = torch.as_tensor(struct["pos"])
        if self.graph is not None and self.pos.device.type == "cpu":
            return self._cpu_step(struct, pos, energy)
        if self.graph is None or self._outdated(pos.to(self.reference.device)):
            self._build(struct)
        with torch.no_grad():
            self.pos.copy_(pos.to(device=self.pos.device, dtype=self.pos.dtype))
        if self.cuda_graph is not None:
            self.cuda_graph.replay()
            self.replays += 1
            return {"energy": self.energy.clone().reshape(1), "forces": self.forces.clone()}
        device = self.pos.device
        if device.type == "cpu":
            if self.plan is None:
                self.plan = self._cpu_plan()
            energy, forces = self.plan()
            self.steps += 1
            return {"energy": (energy + self.const).reshape(1), "forces": forces}
        if self.capture and self.stream is None:
            self.stream = torch.cuda.Stream(device)
        current = torch.cuda.current_stream(device)
        if self.stream is not None:
            # the plain steps run on the stream the graph is recorded on, so its handles exist before
            self.stream.wait_stream(current)
            with torch.cuda.stream(self.stream):
                energy, forces = self._run()
            current.wait_stream(self.stream)
        else:
            energy, forces = self._run()
        self.steps += 1
        if self.capture and self.steps >= self.warmup:
            graph = torch.cuda.CUDAGraph()
            self.stream.wait_stream(current)
            # the backward pass runs in autograd's worker thread: errors are checked per thread
            with torch.cuda.graph(graph, stream=self.stream, capture_error_mode="thread_local"):
                self.energy, self.forces = self._run()
            current.wait_stream(self.stream)
            self.cuda_graph = graph
        return {"energy": energy.detach().reshape(1), "forces": forces.detach()}


__all__ = ["ClassicalStepper"]
