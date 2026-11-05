"""The kernels of :mod:`.classical_kernels` differentiable twice, for training: the per-atom energies of the
pair and bonded terms with the gradients of their parameter arrays, and the position gradients as a function
that is itself differentiable (force losses, Hessian-vector products). The second derivatives run the
generated kernels of :mod:`.classical_tangents`: the backward kernel's tangent along the incoming direction
(the Hessian-vector product), the forward kernel's tangent (the gradient of the incoming energy gradients) and
the tangent of the parameter adjoint (the parameter part of a force loss). The cell enters as a constant: a
stress loss takes the reference terms.
"""
from typing import Optional

import torch
import triton
from torch import Tensor
from torch.autograd.function import once_differentiable

from . import classical_kernels as kk
from . import classical_tangents as kt

# the largest compile-time loop bounds the generated kernels unroll (series terms per torsion, multiplicity)
MAX_SERIES = 6

# per kind of bonded term: the kernel stem and the positions of the float parameter arrays in its params
_KINDS = {"bond": ("_bonds", (0, 1)), "morse": ("_morses", (0, 1, 2)), "angle": ("_angles", (1, 2)),
          "torsion": ("_torsions", (1, 3, 4, 5)), "inversion": ("_inversions", (0, 2)), "hbond": ("_hbonds", (0, 1))}


def _grid(n: int) -> tuple:
    return (triton.cdiv(max(n, 1), kk.BLOCK),)


class _PairsGrad(torch.autograd.Function):
    """The position gradient of :class:`PairTermsTwice` from the incoming gradients ``g`` of the per-atom
    energies, differentiable in the positions, the charges, the table and ``g``."""

    @staticmethod
    def forward(ctx, pos, q, table, g, c):
        args = c["args"](pos, q, table)
        gpos = torch.zeros(pos.shape[0], 3, dtype=torch.float64, device=pos.device)
        if c["E"]:
            kk._go(kk._pairs_backward, _grid(c["E"]), *args, g.contiguous(), gpos, gpos, c["E"], c["n_local"], c["T"],
                   **c["flags"], CELL_GRAD=False, BLOCK=kk.BLOCK, num_warps=kk.WARPS)
        ctx.save_for_backward(pos, q, table, g)
        ctx.c = c
        return gpos.to(pos.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, u):
        pos, q, table, g = ctx.saved_tensors
        c = ctx.c
        args = c["args"](pos, q, table)
        u = u.contiguous().to(pos.dtype)
        N = pos.shape[0]
        hvp = torch.zeros(N, 3, dtype=torch.float64, device=pos.device)
        g_g = g_q = g_t = None
        if c["E"]:
            grid, common = _grid(c["E"]), (c["E"], c["n_local"], c["T"])
            scratch = torch.zeros(N, 3, dtype=torch.float64, device=pos.device)
            kk._go(kt._pairs_backward_t, grid, *args, g.contiguous(), scratch, scratch, *common, pos_ptr_t=u,
                   gpos_ptr_t=hvp, **c["flags"], CELL_GRAD=False, BLOCK=kk.BLOCK, num_warps=kk.WARPS)
            if ctx.needs_input_grad[3]:
                out = torch.zeros(N, 2, dtype=torch.float64, device=pos.device)
                g_g = torch.zeros_like(out)
                kk._go(kt._pairs_forward_t, grid, *args, out, *common, pos_ptr_t=u, e_ptr_t=g_g, **c["flags"],
                       BLOCK=kk.BLOCK, num_warps=kk.WARPS)
                g_g = g_g.to(g.dtype)
            if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
                qa, ta = args[9], args[10]
                out = torch.zeros(N, 2, dtype=torch.float64, device=pos.device)
                g_q, g_t = torch.zeros_like(qa), torch.zeros_like(ta)
                kk._go(kt._pairs_forward_r_t, grid, *args, out, *common, g_ptr=g.contiguous(),
                       q_ptr_g=torch.zeros_like(qa), tab_ptr_g=torch.zeros_like(ta), pos_ptr_t=u, q_ptr_g_t=g_q,
                       tab_ptr_g_t=g_t, **c["flags"], BLOCK=kk.BLOCK, num_warps=kk.WARPS)
                g_q, g_t = g_q.to(q.dtype), g_t.reshape(table.shape).to(table.dtype)
        return hvp.to(pos.dtype), g_q, g_t, g_g, None


class PairTermsTwice(torch.autograd.Function):
    """:class:`~.classical_kernels.PairTerms` differentiable twice in the positions and once more in the
    charges ``q`` and the table (the cell a constant)."""

    @staticmethod
    def forward(ctx, pos, q, table, c):
        out = torch.zeros(pos.shape[0], 2, dtype=torch.float64, device=pos.device)
        if c["E"]:
            kk._go(kk._pairs_forward, _grid(c["E"]), *c["args"](pos, q, table), out, c["E"], c["n_local"], c["T"],
                   **c["flags"], BLOCK=kk.BLOCK, num_warps=kk.WARPS)
        ctx.save_for_backward(pos, q, table)
        ctx.c = c
        return out

    @staticmethod
    def backward(ctx, g):
        pos, q, table = ctx.saved_tensors
        c = ctx.c
        g = g.contiguous()
        gpos = _PairsGrad.apply(pos, q, table, g, c)
        g_q = g_t = None
        if (ctx.needs_input_grad[1] or ctx.needs_input_grad[2]) and c["E"]:
            with torch.no_grad():
                args = c["args"](pos, q, table)
                qa, ta = args[9], args[10]
                g_q, g_t = torch.zeros_like(qa), torch.zeros_like(ta)
                out = torch.zeros(pos.shape[0], 2, dtype=torch.float64, device=pos.device)
                kk._go(kt._pairs_forward_r, _grid(c["E"]), *args, out, c["E"], c["n_local"], c["T"], g_ptr=g.detach(),
                       q_ptr_g=g_q, tab_ptr_g=g_t, **c["flags"], BLOCK=kk.BLOCK, num_warps=kk.WARPS)
            g_q, g_t = g_q.to(q.dtype), g_t.reshape(table.shape).to(table.dtype)
        return gpos, g_q, g_t, None


def pair_terms_twice(pos: Tensor, cell: Optional[Tensor], batch: Tensor, src: Tensor, dst: Tensor,
                     shifts: Optional[Tensor], exclusions: Optional[tuple], n_local: int, t: Tensor, q: Tensor,
                     table: Tensor, dtype: torch.dtype, ke: float, tiny: float, cutoff: float, width: float,
                     form: int) -> tuple:
    """:func:`~.classical_kernels.pair_terms` through :class:`PairTermsTwice`."""
    sc = kk._scalars((float(ke), float(tiny), float(cutoff) ** 2, float(cutoff - width),
                      1.0 / width if width > 0 else 0.0), pos.device)
    cell_t, bat_t, s_t, has = kk._cell_args(pos, None if cell is None else cell.detach(), batch, shifts)
    xp, xk, width_e = exclusions if exclusions is not None else (pos, pos, 0)
    dt = triton.language.float64 if dtype == torch.float64 else triton.language.float32
    src_c, dst_c, t_c = src.contiguous(), dst.contiguous(), t.contiguous()

    def args(p, qq, tab):
        return (src_c, dst_c, s_t, p.detach().contiguous(), cell_t, bat_t, xp, xk, t_c,
                qq.detach().contiguous().to(dtype), tab.detach().contiguous().to(dtype), sc)
    c = {"args": args, "E": int(src.shape[0]), "n_local": int(n_local), "T": int(table.shape[0]),
         "flags": dict(DT=dt, HAS_CELL=has, EXCL=exclusions is not None, MAXE=int(width_e), FORM=int(form),
                       SWITCH=width > 0)}
    e = PairTermsTwice.apply(pos, q, table, c)
    return e[:, 0].to(dtype), e[:, 1].to(dtype)


class _BondedGrad(torch.autograd.Function):
    """The position gradient of :class:`BondedTermsTwice`, differentiable in the positions, the float
    parameter arrays and the incoming gradients."""

    @staticmethod
    def forward(ctx, pos, g, c, *params):
        args = c["args"](pos, params)
        gpos = torch.zeros(pos.shape[0], 3, dtype=torch.float64, device=pos.device)
        if c["M"]:
            kk._go(getattr(kk, c["stem"] + "_backward"), _grid(c["M"]), *args, g.contiguous(), gpos, gpos, c["M"],
                   **c["flags"], CELL_GRAD=False, BLOCK=kk.BLOCK, num_warps=kk.WARPS)
        ctx.save_for_backward(pos, g, *params)
        ctx.c = c
        return gpos.to(pos.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, u):
        pos, g, *params = ctx.saved_tensors
        c = ctx.c
        args = c["args"](pos, params)
        u = u.contiguous().to(pos.dtype)
        N = pos.shape[0]
        hvp = torch.zeros(N, 3, dtype=torch.float64, device=pos.device)
        g_g = None
        g_par = [None] * len(params)
        if c["M"]:
            grid, stem, opts = _grid(c["M"]), c["stem"], dict(**c["flags"], BLOCK=kk.BLOCK, num_warps=kk.WARPS)
            scratch = torch.zeros(N, 3, dtype=torch.float64, device=pos.device)
            kk._go(getattr(kt, stem + "_backward_t"), grid, *args, g.contiguous(), scratch, scratch, c["M"],
                   pos_ptr_t=u, gpos_ptr_t=hvp, CELL_GRAD=False, **opts)
            if ctx.needs_input_grad[1]:
                out = torch.zeros(N, dtype=torch.float64, device=pos.device)
                g_g = torch.zeros_like(out)
                kk._go(getattr(kt, stem + "_forward_t"), grid, *args, out, c["M"], pos_ptr_t=u, out_ptr_t=g_g, **opts)
                g_g = g_g.to(g.dtype)
            wanted = [j for j in c["floats"] if ctx.needs_input_grad[3 + j]]
            if wanted:
                out = torch.zeros(N, dtype=torch.float64, device=pos.device)
                kw = {}
                for j in c["floats"]:
                    name = c["names"][j]
                    kw[name + "_g"] = torch.zeros_like(args[5 + j])
                    kw[name + "_g_t"] = torch.zeros_like(args[5 + j])
                kk._go(getattr(kt, stem + "_forward_r_t"), grid, *args, out, c["M"], g_ptr=g.contiguous(), pos_ptr_t=u,
                       **kw, **opts)
                for j in wanted:
                    g_par[j] = kw[c["names"][j] + "_g_t"].reshape(params[j].shape).to(params[j].dtype)
        return (hvp.to(pos.dtype), g_g, None, *g_par)


class BondedTermsTwice(torch.autograd.Function):
    """:class:`~.classical_kernels.BondedTerms` differentiable twice in the positions and once more in the
    float parameter arrays (the cell a constant)."""

    @staticmethod
    def forward(ctx, pos, c, *params):
        out = torch.zeros(pos.shape[0], dtype=torch.float64, device=pos.device)
        if c["M"]:
            kk._go(getattr(kk, c["stem"] + "_forward"), _grid(c["M"]), *c["args"](pos, params), out, c["M"],
                   **c["flags"], BLOCK=kk.BLOCK, num_warps=kk.WARPS)
        ctx.save_for_backward(pos, *params)
        ctx.c = c
        return out

    @staticmethod
    def backward(ctx, g):
        pos, *params = ctx.saved_tensors
        c = ctx.c
        g = g.contiguous()
        gpos = _BondedGrad.apply(pos, g, c, *params)
        g_par = [None] * len(params)
        wanted = [j for j in c["floats"] if ctx.needs_input_grad[2 + j]]
        if wanted and c["M"]:
            with torch.no_grad():
                args = c["args"](pos, params)
                out = torch.zeros(pos.shape[0], dtype=torch.float64, device=pos.device)
                kw = {c["names"][j] + "_g": torch.zeros_like(args[5 + j]) for j in c["floats"]}
                kk._go(getattr(kt, c["stem"] + "_forward_r"), _grid(c["M"]), *args, out, c["M"], g_ptr=g.detach(), **kw,
                       **c["flags"], BLOCK=kk.BLOCK, num_warps=kk.WARPS)
            for j in wanted:
                g_par[j] = kw[c["names"][j] + "_g"].reshape(params[j].shape).to(params[j].dtype)
        return (gpos, None, *g_par)


def bonded_terms_twice(pos: Tensor, cell: Optional[Tensor], batch: Tensor, idx: Tensor, shifts: Optional[Tensor],
                       kind: str, params: tuple, dtype: torch.dtype, tiny: float, bound: float = 1.0,
                       options: Optional[dict] = None, extra: tuple = (0.0, 0.0)) -> Tensor:
    """:func:`~.classical_kernels.bonded_terms` through :class:`BondedTermsTwice`."""
    options = dict(options or {})
    if options.get("S", 1) > MAX_SERIES or options.get("NMAX", 1) > MAX_SERIES:
        raise ValueError(f"the training kernels take at most {MAX_SERIES} torsion terms and multiplicity "
                         f"{MAX_SERIES} (got S={options.get('S')}, NMAX={options.get('NMAX')})")
    stem, floats = _KINDS[kind]
    sc = kk._scalars((float(tiny), float(bound)) + tuple(float(x) for x in extra), pos.device)
    cell_t, bat_t, s_t, has = kk._cell_args(pos, None if cell is None else cell.detach(), batch, shifts)
    dt = triton.language.float64 if dtype == torch.float64 else triton.language.float32
    idx_c = idx.contiguous()
    fwd = getattr(kk, stem + "_forward")
    names = fwd.arg_names[5:5 + len(params)]

    def args(p, pars):
        return ((p.detach().contiguous(), cell_t, bat_t, s_t, idx_c)
                + tuple(x.detach().contiguous() for x in pars) + (sc,))
    c = {"args": args, "M": int(idx.shape[1]), "stem": stem, "floats": floats, "names": names,
         "flags": dict(DT=dt, HAS_CELL=has, **options)}
    return BondedTermsTwice.apply(pos, c, *params).to(dtype)


__all__ = ["PairTermsTwice", "pair_terms_twice", "BondedTermsTwice", "bonded_terms_twice", "MAX_SERIES"]
