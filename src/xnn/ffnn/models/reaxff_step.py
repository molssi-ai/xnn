"""Molecular-dynamics steps of ReaxFF (LAMMPS conventions) on its kernels.

A :class:`ReaxFFStepper` evaluates the energy and forces of one structure at a sequence of positions. At a
list rebuild (when an atom has moved by half the skin) it keeps the candidates of every term: the pairs within
the cutoff plus the skin, the bonds whose bond order can reach ``botol`` within the skin, and the angles,
torsions and hydrogen bonds of those bonds. A step solves for the charges by conjugate gradients from an
extrapolation of the previous steps' charges, flags the candidates that take part (LAMMPS's bond-order and
distance conditions) and runs the kernels of :mod:`.reaxff_kernels` on the flagged terms; once recorded, the
charge iterations and the energy with its forces are CUDA graph replays. The numbers are those of the model's
evaluation, with the charges converged to ``tol``.
"""
from typing import Optional

import torch
from torch import Tensor

from xnn.common.data import structure_to_graph
from xnn.common.data.neighborlist import unique_pairs
from xnn.common.models.ops import build_triplets, gather_rows, scatter_sum

from .classical_step import _shortest_width
from .reaxff import (
    EEM_CG_TOL,
    KCAL_TO_EV,
    LAMMPS_BOND_CUT,
    LAMMPS_EV_TO_KCAL,
    LAMMPS_HB_THRESHOLD,
    LAMMPS_QEQ_KE,
    LAMMPS_THB_CUT,
    LAMMPS_THB_CUTSQ,
    SAFETY,
    ReaxFF,
    _group_cross,
)

# the lists of a step, padded to a capacity, with ``valid``: per list its index arrays and, for vectors, the
# image shifts (``shift``, of the pair kernel) or their Cartesian offsets ``shift @ cell`` (``off``, zero without
# a cell); the charge matrix's rows hold the pair of each entry
_LISTS = {"pairs": ("src", "dst", "shift", "off"), "rows": ("col", "row", "pair"), "bonds": ("src", "dst", "off"),
          "angles": ("e1", "e2", "c", "i", "k"), "torsions": ("ei", "ec", "el", "i", "j", "k", "l"),
          "hbonds": ("d", "x", "h", "z", "off")}
_FLOATS = ("shift", "off")


def _bond_reach(model: ReaxFF, device, dtype, level: Optional[float] = None) -> Tensor:
    """The largest distance ``(S, S)`` at which a pair's uncorrected bond order reaches ``level`` (by default
    ``botol``; at most LAMMPS's 5 Angstrom), on a 0.005 Angstrom grid plus one step."""
    S = len(model.species)
    r = torch.arange(0.0, LAMMPS_BOND_CUT + 0.005, 0.005, dtype=dtype, device=device)
    a = torch.arange(S, device=device).repeat_interleave(S)
    b = torch.arange(S, device=device).repeat(S)
    with torch.no_grad():
        e1, e2, e3 = model._lammps_channels(r[None, :].expand(S * S, -1).reshape(-1),
                                            a.repeat_interleave(r.numel()), b.repeat_interleave(r.numel()))
        on = ((e1 + e2 + e3) >= (model.botol if level is None else level)).reshape(S * S, -1) \
            & model.has_bond[a, b][:, None]
        reach = torch.where(on, r[None, :], torch.zeros_like(r)[None, :]).amax(1) + 0.005
    reach = reach.reshape(S, S)
    return torch.maximum(reach, reach.T)


def _stage_tables(model: ReaxFF, dtype) -> tuple:
    """The constant tables of a stepper's bond-order stage (:func:`~.reaxff_fast.bond_orders`,
    :func:`~.reaxff_fast.atom_terms`): the channels per species pair, ``(botol, bond cutoff)``, the atom terms per
    species and per species pair, and the general parameters."""
    P, pr = model.params, model._pair
    with torch.no_grad():
        ch = torch.stack([pr("rosi"), pr("bo1"), pr("bo2"), pr("ropi"), pr("bo3"), pr("bo4"), pr("ropp"), pr("bo5"),
                          pr("bo6")] + [x.to(dtype) for x in (model.si_on, model.pi_on, model.pp_on)], -1).to(dtype)
        sc = torch.tensor([model.botol, LAMMPS_BOND_CUT], dtype=dtype, device=ch.device)
        at = torch.stack([P["val"], P["vale"], (model.mass <= 21.0).to(dtype), model.is_carbon.to(dtype), P["lp2"],
                          P["ovun2"], P["ovun5"]], -1).to(dtype)
        triple = torch.ones_like(model.co_pair) if model.triple_all else model.co_pair
        ap = torch.stack([pr("ovun1") * pr("Desi"), triple.to(dtype)], -1).to(dtype)
        kc2 = P["kc2"].reshape(())
        gen = torch.stack([P[k].reshape(()).to(dtype) for k in ("lp1", "ovun3", "ovun4", "ovun6", "ovun7", "ovun8")]
                          + [kc2.to(dtype), (kc2 > 0.001 * KCAL_TO_EV).to(dtype)]
                          + [P[k].reshape(()).to(dtype) for k in ("trip1", "trip2", "trip3", "trip4")])
    return ch, sc, at, ap, gen


class ReaxFFStepper:
    """Energy and forces of one structure along a trajectory from the kernels of a ReaxFF model.

    Parameters
    ----------
    model : ReaxFF
        A classical field in the LAMMPS conventions on a CUDA device (or a wrapper with a ``model``
        attribute), in evaluation mode with no trained parameter.
    skin : float, optional
        Margin of the candidate lists beyond the cutoffs in Angstrom; by default 0.3 (a larger one rebuilds
        less often but grows the torsion and hydrogen-bond lists faster).
    tol : float, optional
        Relative residual of the charges' conjugate gradients; by default the model's (1e-13 in float64,
        1e-6 in float32). LAMMPS's ``fix qeq/reaxff`` takes its own, often 1e-6.
    chunk : int, optional
        Conjugate-gradient iterations of one replay of the recorded charge solve; a step replays it until
        the charges converge. By default 16.
    capture : bool, optional
        Replay the steps as CUDA graphs after ``warmup`` plain steps; by default ``True``.
    warmup : int, optional
        Plain steps before the graphs are recorded; by default 2.

    Attributes
    ----------
    rebuilds : int
        Candidate lists built.
    refills : int
        Of those, the lists copied into the buffers of the recorded graphs.
    replays : int
        Steps replayed from the graphs.
    chunks : int
        Replays of the charge solve.
    """

    def __init__(self, model, skin: float = 0.3, tol: Optional[float] = None, chunk: int = 16, capture: bool = True,
                 warmup: int = 2):
        self.model = getattr(model, "model", model)
        m = self.model
        if not isinstance(m, ReaxFF) or m.conventions != "lammps" or m.nn:
            raise ValueError("the stepper runs classical ReaxFF fields in the LAMMPS conventions")
        if any(p.requires_grad for p in m.params.values()):
            raise ValueError("the stepper evaluates fixed parameters: none may require a gradient")
        self.dtype = m.params["val"].dtype
        # the shielding exponent as a number (no device read inside a recorded step)
        self.vdw1 = float(m.params["vdw1"])
        self.tol = float(EEM_CG_TOL.get(self.dtype, 1.0e-6) if tol is None else tol)
        self.skin = float(skin)
        self.chunk = int(chunk)
        self.capture, self.warmup = bool(capture), int(warmup)
        self.rebuilds = self.replays = self.refills = self.chunks = 0
        self.base = None
        self.stage = None
        self.lists = None
        self.graphs = None
        self.stream = None
        self.hist = None

    def _candidates(self, struct: dict) -> dict:
        """The unpadded candidate lists at the positions of ``struct``."""
        m = self.model
        device = m.params["val"].device
        rc = float(m.cutoff) + self.skin
        if self.base is None:
            self.base = structure_to_graph(struct, rc, device=device)
            if _shortest_width(self.base) <= rc:
                raise ValueError(f"the stepper's lists have no periodic images of an atom itself: the cell's width "
                                 f"must exceed cutoff + skin = {rc:.3f} A")
            self.s = m.z_to_index[self.base.atomic_numbers]
            self.reach = _bond_reach(m, device, self.dtype)
            # a donor's corrected bond order (the sum of the channels less botol, times corrections of at most
            # one) reaches the hydrogen-bond threshold only within this distance
            self.hb_reach = _bond_reach(m, device, self.dtype, (LAMMPS_HB_THRESHOLD + m.botol) * (1.0 - 1.0e-6))
            self.N = int(self.base.num_nodes)
        g = self.base
        pos = torch.as_tensor(struct["pos"]).to(device=device, dtype=g.pos.dtype)
        cell = None if g.cell is None else g.cell[0]
        pbc = None if g.pbc is None else g.pbc[0]
        found = None
        if device.type == "cpu":
            # one search by the CPU kernels (a grid, or every pair in a small cell)
            from .classical_cpu import cpu_pairs
            found = cpu_pairs(pos, rc, cell, pbc)
        if found is not None:
            # the shifts of the image indices from their table
            one = torch.arange(-1, 2, dtype=pos.dtype)
            i, k, sh = found[0].long(), found[1].long(), torch.cartesian_prod(one, one, one)[found[2].long()]
        else:
            i, k, sh = unique_pairs(pos, rc, cell, pbc)
        s, N = self.s, self.N
        sh = torch.zeros(i.numel(), 3, dtype=pos.dtype, device=device) if sh is None or cell is None \
            else sh.to(pos.dtype)
        offset = (lambda x: torch.zeros_like(x)) if cell is None else (lambda x: x @ cell.to(pos.dtype))
        from .reaxff_kernels import pair_geometry
        # the distances of the half list, the same in both directions, and the pairs' Cartesian offsets
        r, off_pairs = pair_geometry(i, k, sh, pos, cell)
        out = {"pairs": (i, k, sh, off_pairs)}
        E = int(i.numel())

        def directed(e):
            # the directed list is the pairs then their reverses: entry e < E is (i, k), e >= E (k, i)
            back = e >= E
            h = torch.where(back, e - E, e)
            return (torch.where(back, k[h], i[h]), torch.where(back, i[h], k[h]),
                    torch.where(back[:, None], -sh[h], sh[h]))
        if device.type == "cpu":
            # a CPU solves for the charges on the half list (reaxff_kernels.CGSolver): no rows
            none = torch.zeros(0, dtype=torch.int32, device=device)
            out["rows"] = (none, none, none)
            self.found_ptr = torch.zeros(N + 1, dtype=torch.int32, device=device)
        else:
            # the charge matrix by rows: both directions of every pair, sorted by row, with the row offsets and
            # each entry's pair
            # (the order within a row only orders the row's sum: an unstable sort of 32-bit keys)
            src, dst = torch.cat([i, k]), torch.cat([k, i])
            dst32 = dst.int()
            order = torch.argsort(dst32)
            out["rows"] = (src[order].int(), dst32[order], (order % max(i.numel(), 1)).int())
            # the row offsets from the sorted rows (no host read of a bin count)
            self.found_ptr = torch.searchsorted(dst32[order], torch.arange(N + 1, dtype=torch.int32, device=device))
        # the candidates of a symmetric test found on the half list and taken in both directions (forward pairs,
        # then reversed ones)
        reach = torch.clamp(self.reach + self.skin, max=LAMMPS_BOND_CUT + self.skin)
        hb_on = m._h_index >= 0
        if found is not None:
            # the bond and hydrogen-bond candidates of the half list from its rows, in one pass
            from .reaxff_kernels import pair_candidates
            half, a_fwd, a_bwd = pair_candidates(found[3], found[1], r, s, torch.where(m.has_bond, reach, -1.0),
                                                 m.hb_flag if hb_on else None, m.hb_long + self.skin)
        else:
            si, sj = s[k], s[i]
            half = ((r <= reach[si, sj]) & m.has_bond[si, sj] & (i != k)).nonzero()[:, 0]
        be = torch.cat([half, half + E])
        b_src, b_dst, b_shift = directed(be)
        out["bonds"] = (b_src, b_dst, offset(b_shift))
        # angles: pairs of bonds around a center whose type has a valence term
        e1, e2, c = build_triplets(torch.stack([b_src, b_dst]), N)
        ai, ak = b_src[e1], b_src[e2]
        val1 = m._ang("val1")[s[ai], s[c], s[ak]]
        keep = (val1.abs() > 0.001 * KCAL_TO_EV).nonzero()[:, 0]
        out["angles"] = tuple(x[keep] for x in (e1, e2, c, ai, ak))
        # torsions around each bond once (its first direction in the list), as the model enumerates them; a
        # bond's reverse is the directed entry one pair list away (the candidates hold both or neither)
        n_dir = 2 * E
        where = torch.full((n_dir,), -1, dtype=torch.long, device=device)
        where[be] = torch.arange(be.numel(), device=device)
        rev = where[(be + E) % max(n_dir, 1)]
        central = (torch.arange(be.numel(), device=device) < rev).nonzero()[:, 0]
        jc, kc = b_dst[central], b_src[central]
        c_idx, i_loc = _group_cross(jc, b_dst, N)
        ok = (i_loc != central[c_idx]).nonzero()[:, 0]
        c_idx, i_loc = c_idx[ok], i_loc[ok]
        pair_idx, l_loc = _group_cross(kc[c_idx], b_dst, N)
        c2, i2 = c_idx[pair_idx], i_loc[pair_idx]
        ok = (l_loc != rev[central[c2]]).nonzero()[:, 0]
        e_c, e_i, e_l = central[c2[ok]], i2[ok], l_loc[ok]
        ti, tj, tk, tl = b_src[e_i], b_dst[e_i], b_src[e_c], b_src[e_l]
        keep = m.tor_listed[s[ti], s[tj], s[tk], s[tl]].nonzero()[:, 0]
        out["torsions"] = tuple(x[keep] for x in (e_i, e_c, e_l, ti, tj, tk, tl))
        # hydrogen bonds: an X-H bond and an H...Z pair within the hydrogen-bond cutoff plus the skin
        empty = torch.zeros(0, dtype=torch.long, device=device)
        out["hbonds"] = (empty,) * 4 + (torch.zeros(0, 3, dtype=pos.dtype, device=device),)
        if m._h_index >= 0:
            flag = m.hb_flag[s]
            # the X-H bonds whose order can reach the threshold within the skin
            r_b = r[half].repeat(2)
            d_ids = ((flag[b_dst] == 1) & (flag[b_src] == 2)
                     & (r_b <= self.hb_reach[s[b_src], s[b_dst]] + self.skin)).nonzero()[:, 0]
            if found is not None:
                a_ids = torch.cat([a_fwd, a_bwd + E])
            else:
                near = r <= m.hb_long + self.skin
                a_ids = torch.cat([((flag[k] == 1) & (flag[i] == 2) & near).nonzero()[:, 0],
                                   ((flag[i] == 1) & (flag[k] == 2) & near).nonzero()[:, 0] + E])
            if d_ids.numel() and a_ids.numel() and device.type == "cpu":
                # the cross product by hydrogen in two launches
                from .reaxff_kernels import hbond_cross
                if found is not None:
                    # the pairs' own atoms and offsets (the reverse ones negated)
                    a_src, a_dst = torch.cat([i[a_fwd], k[a_bwd]]), torch.cat([k[a_fwd], i[a_bwd]])
                    a_off = torch.cat([off_pairs[a_fwd], -off_pairs[a_bwd]])
                else:
                    a_src, a_dst, a_sh = directed(a_ids)
                    a_off = offset(a_sh)
                out["hbonds"] = hbond_cross(d_ids, b_dst[d_ids], b_src, a_src, a_dst, a_off, N)
            elif d_ids.numel() and a_ids.numel():
                a_src, a_dst, _ = directed(a_ids)
                di, al = _group_cross(b_dst[d_ids], a_dst, N)
                d_e, a_e = d_ids[di], a_ids[al]
                a_from = a_src[al]
                ok = (a_from != b_src[d_e]).nonzero()[:, 0]
                d_e, a_e, a_from = d_e[ok], a_e[ok], a_from[ok]
                # the H...Z vector is x_H - x_Z + shift @ cell
                out["hbonds"] = (d_e, b_src[d_e], b_dst[d_e], a_from, offset(directed(a_e)[2]))
        self.reference = pos.clone()
        return out

    def _fill(self, found: dict) -> bool:
        """Copy the candidate lists into the buffers when they fit (the padding invalid); ``False`` when a
        list outgrew its buffer."""
        L = self.lists
        if L is None or any(found[n][0].numel() > L[n]["cap"] for n in _LISTS):
            return False
        for name, fields in _LISTS.items():
            got, buf = found[name], L[name]
            n = int(got[0].numel())
            # the padding past the previous list was zeroed then: only the entries between the two lists
            end = max(n, buf.get("n", buf["cap"]))
            for f, x in zip(fields, got):
                buf[f][:n].copy_(x)
                buf[f][n:end].zero_()
            buf["n"] = n
            if name == "bonds":
                # the padding bonds (inactive, values zero) spread over the atoms: the per-atom sums over the
                # bonds would otherwise add every one of them to atom 0
                spread = torch.arange(n, buf["cap"], device=buf["src"].device) % self.N
                buf["src"][n:].copy_(spread)
                buf["dst"][n:].copy_((spread + 1) % self.N)
            buf["valid"][:n] = True
            buf["valid"][n:end] = False
            if name == "pairs":
                self.n_pairs = n
            if name == "torsions":
                # in place: a recorded graph reads these buffers
                self.valid_t.copy_(buf["valid"])
            if name == "hbonds":
                self.valid_h.copy_(buf["valid"])
        self.rowptr.copy_(self.found_ptr)
        self._statics()
        return True

    def _allocate(self, found: dict) -> None:
        """Buffers of 1.5 times the lists in multiples of 1024 (a new size compiles no kernel variant), and the
        arrays a step reads."""
        device = self.s.device
        self.lists = {}
        for name, fields in _LISTS.items():
            n = int(found[name][0].numel())
            cap = 1024 * (int(1.5 * n) // 1024 + 1)
            buf = {"cap": cap, "valid": torch.zeros(cap, dtype=torch.bool, device=device)}
            for f in fields:
                # the charge rows (the largest list, read in every product) with 32-bit indices
                buf[f] = (torch.zeros(cap, 3, dtype=self.base.pos.dtype, device=device) if f in _FLOATS
                          else torch.zeros(cap, dtype=torch.int32 if name == "rows" else torch.long, device=device))
            self.lists[name] = buf
        g = self.base
        self.pos = g.pos.detach().clone().requires_grad_(True)
        self.cell = None if g.cell is None else g.cell.detach().clone()
        P = self.model.params
        s = self.s
        self.diag = 2.0 * P["mu"][s]
        self.rhs = torch.stack([-P["chi"][s], torch.ones_like(P["chi"][s])], 1)
        self.bnorm = torch.linalg.vector_norm(self.rhs, dim=0)
        total = 0.0 if self.total_charge is None else float(self.total_charge)
        self.total = torch.full((), total, dtype=self.dtype, device=device)
        self.res = torch.zeros(2, dtype=self.dtype, device=device)
        self.rowptr = torch.zeros(self.N + 1, dtype=torch.long, device=device)
        self.g_pairs = torch.zeros(self.lists["pairs"]["cap"], dtype=self.dtype, device=device)
        self.valid_t = torch.zeros(self.lists["torsions"]["cap"], dtype=torch.int32, device=device)
        self.valid_h = torch.zeros(self.lists["hbonds"]["cap"], dtype=torch.int32, device=device)
        if self.hist is None:
            # the last four solutions (s, t) of the charge system, newest first
            self.x = self.rhs / self.diag[:, None]
            self.hist = self.x[None].repeat(4, 1, 1)
        self._fill(found)

    def _statics(self) -> None:
        """The per-pair constants of the charge matrix (the shielding ``gamma^-3``)."""
        E = self.lists["pairs"]
        s = self.s
        gm3 = (1.0 / self.model._comb("gamma")[s[E["dst"]], s[E["src"]]]) ** 3
        # in place: a recorded graph reads this buffer
        if getattr(self, "gm3", None) is not None and self.gm3.shape == gm3.shape:
            self.gm3.copy_(gm3)
        else:
            self.gm3 = gm3

    def _vectors(self, a: Tensor, b: Tensor, off: Tensor) -> Tensor:
        """``x_b - x_a + off`` in the model's precision, ``off`` the Cartesian image offset (gathers whose
        backward adds atomically)."""
        return (gather_rows(self.pos, b) - gather_rows(self.pos, a) + off).to(self.dtype)

    def _matrix(self) -> None:
        """The off-diagonal coefficients of the charge matrix on the pairs within the cutoff, once per step,
        and in the order of its rows."""
        from .reaxff_kernels import eem_matrix
        E = self.lists["pairs"]
        eem_matrix(E["src"], E["dst"], E["off"], E["valid"], self.gm3, self.pos, self.g_pairs, LAMMPS_QEQ_KE,
                   float(self.model.cutoff), SAFETY)

    def _solve(self, iterations: Optional[int]) -> int:
        """Conjugate-gradient iterations of the charge system on the pairs within the cutoff (by rows) from
        the current solution ``x``: ``iterations`` of them, or (``None``) until converged; their count. The
        relative residuals go to ``res``."""
        from .reaxff_kernels import CGSolver, eem_rows
        with torch.no_grad():
            product = eem_rows(self.rowptr, self.lists["rows"]["col"], self.g_pairs, self.lists["rows"]["pair"])
            if self.pos.device.type == "cpu" and iterations is None:
                # on a CPU the iterations as kernel launches on the half list (each pair once)
                if self.cg is None:
                    E = self.lists["pairs"]
                    src = E["src"][:self.n_pairs]
                    ptr = torch.searchsorted(src, torch.arange(self.N + 1, dtype=src.dtype))
                    self.cg = CGSolver(ptr, E["dst"][:self.n_pairs].to(torch.int32), self.g_pairs, self.diag,
                                       self.rhs)
                x, res, it = self.cg.solve(self.x, self.tol)
                self.x.copy_(x)
                self.res.copy_(res)
                return it
            diag, inv, b = self.diag, 1.0 / self.diag, self.rhs

            def matvec(v):
                return diag[:, None] * v + product(v)
            x = self.x.clone()
            res = b - matvec(x)
            z = res * inv[:, None]
            p = z.clone()
            rz = (res * z).sum(0)
            it = 0
            while iterations is None or it < iterations:
                if iterations is None and bool((torch.linalg.vector_norm(res, dim=0)
                                                <= self.tol * self.bnorm).all()):
                    break
                ap = matvec(p)
                pap = (p * ap).sum(0)
                alpha = torch.where(pap != 0, rz / torch.where(pap != 0, pap, 1.0), 0.0)
                x = x + alpha * p
                res = res - alpha * ap
                z = res * inv[:, None]
                rz_new = (res * z).sum(0)
                beta = torch.where(rz != 0, rz_new / torch.where(rz != 0, rz, 1.0), 0.0)
                p = z + beta * p
                rz = rz_new
                it += 1
            self.x.copy_(x)
            self.res.copy_(torch.linalg.vector_norm(res, dim=0) / self.bnorm)
        return it

    def _energy(self) -> tuple:
        """The energy, forces and charges at the current positions with the charges of ``x``."""
        m, L, s, N, dtype = self.model, self.lists, self.s, self.N, self.dtype
        P = m.params
        with torch.no_grad():
            s_, t_ = self.x[:, 0], self.x[:, 1]
            q = s_ - (s_.sum() - self.total) / t_.sum() * t_
        # bonds: a candidate takes part when its uncorrected bond order reaches botol within 5 Angstrom
        B = L["bonds"]
        b_src, b_dst = B["src"], B["dst"]
        b_vec = self._vectors(b_src, b_dst, B["off"])
        b_r = torch.sqrt((b_vec * b_vec).sum(1) + SAFETY)
        # the bond orders, the coordinations and the atom terms on kernels (no PyTorch operation per term)
        from .reaxff_fast import atom_terms, bond_orders
        if self.stage is None:
            self.stage = _stage_tables(m, dtype)
        ch, sc, at, ap, gen = self.stage
        bop_si, bop_pi, bop_pp, deltap, act_b = bond_orders(b_src, b_dst, b_r, B["valid"], s, ch, sc, N, dtype)
        bo0, bosi, bopi, bopp, e_bond = m._fast_bond_corrections(s, b_src, b_dst, bop_si, bop_pi, bop_pp, deltap,
                                                                 dtype, act=act_b)
        delta = scatter_sum(bo0, b_dst, N)
        dv = delta - P["val"][s]
        dpi = scatter_sum(bopi + bopp, b_dst, N)
        e_atoms, nlp = atom_terms(b_src, b_dst, bo0, bopi, bopp, delta, s, at, ap, gen, dtype)
        thb = LAMMPS_THB_CUT
        A = L["angles"]
        b1, b2 = bo0[A["e1"]], bo0[A["e2"]]
        act_a = A["valid"] & (b1 > thb) & (b2 > thb) & (b1 * b2 > LAMMPS_THB_CUTSQ)
        e_ang, e_pen, e_tcon = m._fast_angles(s, N, b_dst, b_vec, b_r, bo0, None, delta, dv, dpi, nlp, A["e1"],
                                              A["e2"], A["c"], A["i"], A["k"], act=act_a.to(torch.int32))
        T = L["torsions"]
        # the torsions that take part tested in the kernel (their bond orders, a nonzero 1-4 vector)
        e_tor, e_fcon = m._fast_torsions(s, N, b_vec, b_r, bo0, bopi, delta,
                                         (T["ei"], T["ec"], T["el"], T["i"], T["j"], T["k"], T["l"]),
                                         act=self.valid_t, gate=thb)
        H = L["hbonds"]
        # the H...Z vectors formed by the kernel from the positions (its adjoint adds into them directly), the
        # hydrogen bonds that take part tested there (the donor's bond order, the H...Z distance)
        e_hb = m._fast_hbonds(s, N, b_vec, b_r, bo0, self.pos, self.zero_r,
                              (H["d"], self.hb_index, H["x"], H["h"], H["z"]), act=self.valid_h, off=H["off"],
                              gate=(LAMMPS_HB_THRESHOLD, m.hb_long ** 2))
        Q = L["pairs"]
        e_self = LAMMPS_EV_TO_KCAL * KCAL_TO_EV * q * (P["chi"][s] + q * P["mu"][s])
        terms = [e_bond, e_atoms, e_ang, e_pen, e_tcon, e_tor, e_fcon, e_hb, e_self]
        shifts = None if self.cell is None else Q["shift"]
        if self.pos.device.type == "cpu":
            # on a CPU the pairs' energies and forces in one pass, outside the graph (the charges are constants)
            e_pairs, g_pairs = m._fast_pairs(self.pos.detach(), self.cell, self.base.batch, Q["src"], Q["dst"], shifts,
                                             s, q, dtype, vdw1=self.vdw1, forces=True)
            total = sum(x.double().sum() for x in terms) - P["atomic"][s].double().sum()
            forces = -torch.autograd.grad(total, self.pos)[0] - g_pairs.to(self.pos.dtype)
            return total.detach() + e_pairs.sum(), forces, q
        e_vdw, e_coul = m._fast_pairs(self.pos, self.cell, self.base.batch, Q["src"], Q["dst"], shifts, s, q, dtype,
                                      vdw1=self.vdw1)
        total = sum(x.double().sum() for x in terms + [e_vdw, e_coul]) - P["atomic"][s].double().sum()
        forces = -torch.autograd.grad(total, self.pos)[0]
        return total, forces, q

    def _outdated(self, pos: Tensor) -> bool:
        step = pos.to(self.reference.dtype) - self.reference
        return bool((step * step).sum(-1).max() > (0.5 * self.skin) ** 2)

    def _build(self, struct: dict) -> None:
        self.cg = None
        found = self._candidates(struct)
        self.rebuilds += 1
        if self.s.device.type == "cpu" and self.lists is not None:
            self._take(found)
            self.refills += 1
            return
        if self.lists is not None and self.graphs is not None and self._fill(found):
            self.refills += 1
            return
        self.total_charge = struct.get("total_charge")
        self._allocate(found)
        self.hb_index = torch.arange(self.lists["hbonds"]["cap"], device=self.s.device)
        self.zero_r = torch.zeros(1, dtype=self.dtype, device=self.s.device)
        self.graphs = None
        self.steps = 0
        if self.s.device.type == "cpu":
            self._take(found)

    def _take(self, found: dict) -> None:
        """On a CPU (no recorded graph to replay) the candidate lists themselves, every entry taking part, in
        place of the buffers: no copy, and a step's kernels run over the lists' own lengths."""
        device = self.s.device
        self.lists = {}
        for name, fields in _LISTS.items():
            got = found[name]
            n = int(got[0].numel())
            buf = {"cap": n, "n": n, "valid": torch.ones(n, dtype=torch.bool, device=device)}
            for f, x in zip(fields, got):
                buf[f] = (x.to(self.base.pos.dtype) if f in _FLOATS
                          else x.to(torch.int32 if name == "rows" else torch.long)).contiguous()
            self.lists[name] = buf
        self.n_pairs = int(found["pairs"][0].numel())
        self.g_pairs = torch.zeros(self.n_pairs, dtype=self.dtype, device=device)
        self.valid_t = torch.ones(self.lists["torsions"]["cap"], dtype=torch.int32, device=device)
        self.valid_h = torch.ones(self.lists["hbonds"]["cap"], dtype=torch.int32, device=device)
        self.hb_index = torch.arange(self.lists["hbonds"]["cap"], device=device)
        self.rowptr = self.found_ptr.to(torch.long)
        self.gm3 = None
        self._statics()

    def _guess(self) -> None:
        """Start the charge solve from the cubic extrapolation of the last four solutions."""
        with torch.no_grad():
            h = self.hist
            self.x.copy_(4.0 * (h[0] + h[2]) - 6.0 * h[1] - h[3])

    def _keep(self) -> None:
        """Push the converged solution onto the history."""
        with torch.no_grad():
            self.hist[1:].copy_(self.hist[:-1].clone())
            self.hist[0].copy_(self.x)

    def __call__(self, struct: dict) -> dict:
        """The energy ``(1,)`` (float64), forces ``(N, 3)`` and charges ``(N,)`` at the positions of
        ``struct``.

        Parameters
        ----------
        struct : dict
            ``pos``, ``atomic_numbers`` and, for a cell, ``cell`` and ``pbc`` (arrays or tensors); the cell
            and the atoms are those of the first call.
        """
        pos = torch.as_tensor(struct["pos"])
        if self.lists is None or self._outdated(pos.to(self.reference.device)):
            self._build(struct)
        with torch.no_grad():
            self.pos.copy_(pos.to(device=self.pos.device, dtype=self.pos.dtype))
        self._guess()
        if self.graphs is not None:
            matrix, solve, energy = self.graphs
            matrix.replay()
            solve.replay()
            self.chunks += 1
            while bool((self.res > self.tol).any()):
                solve.replay()
                self.chunks += 1
            self._keep()
            energy.replay()
            self.replays += 1
            return {"energy": self.energy.clone().reshape(1), "forces": self.forces.clone(),
                    "charges": self.q.clone()}
        device = self.pos.device
        # on a CPU the steps run as they are (the kernels as C++ loops), without graphs
        capture = self.capture and device.type == "cuda"
        if capture and self.stream is None:
            self.stream = torch.cuda.Stream(device)
        current = torch.cuda.current_stream(device) if device.type == "cuda" else None
        if self.stream is not None:
            self.stream.wait_stream(current)
            with torch.cuda.stream(self.stream):
                self._matrix()
                self._solve(None)
                self._keep()
                energy, forces, q = self._energy()
            current.wait_stream(self.stream)
        else:
            self._matrix()
            self._solve(None)
            self._keep()
            energy, forces, q = self._energy()
        self.steps += 1
        if capture and self.steps >= self.warmup:
            matrix, solve, step = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
            self.stream.wait_stream(current)
            with torch.cuda.graph(matrix, stream=self.stream):
                self._matrix()
            with torch.cuda.graph(solve, stream=self.stream):
                self._solve(self.chunk)
            # the backward pass runs in autograd's worker thread: errors are checked per thread
            with torch.cuda.graph(step, stream=self.stream, capture_error_mode="thread_local"):
                self.energy, self.forces, self.q = self._energy()
            current.wait_stream(self.stream)
            self.graphs = (matrix, solve, step)
        return {"energy": energy.detach().reshape(1), "forces": forces.detach(), "charges": q.detach()}


__all__ = ["ReaxFFStepper"]
