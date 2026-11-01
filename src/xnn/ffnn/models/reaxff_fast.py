"""The fast path of :class:`~xnn.ffnn.models.ReaxFF`: the terms on the kernels of :mod:`.reaxff_kernels`, their
backward from the generated adjoints of :mod:`.reaxff_tangents`. The numbers are those of the eager terms to
round-off. Differentiable once, in the positions, the cells and the inputs (charges, parameter tables).
"""
from typing import Optional

import torch
import triton
from torch import Tensor

from .classical_kernels import _cell_args, _delayed, _scalars


def fast_supported(device: torch.device, dtype: torch.dtype) -> bool:
    """Whether the ReaxFF kernels run on ``device`` in ``dtype`` (CUDA and Triton)."""
    if device.type != "cuda" or dtype not in (torch.float32, torch.float64):
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


class NonbondedTerms(torch.autograd.Function):
    """The van der Waals and Coulomb energies per atom ``(N, 2)`` (float64) of the pairs ``i <= k`` of a
    directed pair list (:func:`~.reaxff_kernels._nonbonded_forward`), differentiable in the positions, the
    cells (``(B, 3, 3)``, or the positions as a stand-in without cells), the charges ``q`` and the species-pair
    table."""

    @staticmethod
    def forward(ctx, pos, cell, q, table, c):
        from . import reaxff_kernels as rk
        out = torch.zeros(pos.shape[0], 2, dtype=torch.float64, device=pos.device)
        if c["E"]:
            rk._nonbonded_forward[(triton.cdiv(c["E"], rk.BLOCK),)](*c["args"](pos, cell, q, table), out, c["E"],
                                                                   c["T"], **c["flags"], BLOCK=rk.BLOCK,
                                                                   num_warps=rk.WARPS)
        ctx.save_for_backward(pos, cell, q, table)
        ctx.c = c
        return out

    @staticmethod
    def backward(ctx, g):
        from . import reaxff_kernels as rk
        from . import reaxff_tangents as rt
        pos, cell, q, table = ctx.saved_tensors
        c = ctx.c
        args = c["args"](pos, cell, q, table)
        g_pos = torch.zeros(pos.shape[0], 3, dtype=args[3].dtype, device=pos.device)
        g_cell, g_q, g_t = torch.zeros_like(args[4]), torch.zeros_like(args[7]), torch.zeros_like(args[8])
        if c["E"]:
            out = torch.zeros(pos.shape[0], 2, dtype=torch.float64, device=pos.device)
            rt._nonbonded_forward_r[(triton.cdiv(c["E"], rk.BLOCK),)](
                *args, out, c["E"], c["T"], g_ptr=g.detach().contiguous(), pos_ptr_g=g_pos, cell_ptr_g=g_cell,
                q_ptr_g=g_q, tab_ptr_g=g_t, **c["flags"], BLOCK=rk.BLOCK, num_warps=rk.WARPS)
        g_cell = g_cell.reshape(cell.shape).to(cell.dtype) if c["flags"]["HAS_CELL"] else None
        return _delayed((pos, cell, q, table), g_pos.to(pos.dtype), g_cell, g_q.to(q.dtype),
                        g_t.reshape(table.shape).to(table.dtype), None)


def nonbonded(pos: Tensor, cell: Optional[Tensor], batch: Tensor, src: Tensor, dst: Tensor,
              shifts: Optional[Tensor], species: Tensor, q: Tensor, table: Tensor, dtype: torch.dtype, ke: float,
              cutoff: float, vdw1: float, safety: float, shield: bool, wall: bool) -> tuple:
    """The per-atom van der Waals and Coulomb energies ``(N,)`` each in ``dtype`` of the directed pair list
    ``(src, dst)`` with image shifts ``shifts`` (or ``None``), the species index per atom, the charges and the
    ``(S, S, NB_COLUMNS)`` table of :func:`~.reaxff_kernels.nonbonded_table`."""
    sc = _scalars((float(ke), float(cutoff), float(vdw1), float(safety), 1.0 / 3.0), pos.device)
    _, bat_t, s_t, has = _cell_args(pos, cell, batch, shifts)
    dt = triton.language.float64 if dtype == torch.float64 else triton.language.float32
    src_c, dst_c, t_c = src.contiguous(), dst.contiguous(), species.contiguous()

    def args(p, cl, qq, tab):
        cell_t = _cell_args(p, cl.detach(), batch, shifts)[0] if has else p.detach()
        return (src_c, dst_c, s_t, p.detach().contiguous(), cell_t, bat_t, t_c, qq.detach().contiguous().to(dtype),
                tab.detach().contiguous().to(dtype), sc)
    c = {"args": args, "E": int(src.shape[0]), "T": int(table.shape[0]),
         "flags": dict(DT=dt, HAS_CELL=has, SHIELD=bool(shield), WALL=bool(wall))}
    e = NonbondedTerms.apply(pos, cell if has else pos, q, table, c)
    return e[:, 0].to(dtype), e[:, 1].to(dtype)


class BondCorrections(torch.autograd.Function):
    """The corrected bond orders per bond and the per-atom bond energy of a bond list
    (:func:`~.reaxff_kernels._bond_corrections_forward`), differentiable in the uncorrected bond orders, the
    uncorrected coordination and the parameter tables."""

    @staticmethod
    def forward(ctx, bop_si, bop_pi, bop_pp, deltap, spt, tab, gen, c):
        from . import reaxff_kernels as rk
        outs = [torch.zeros(c["M"], dtype=c["dtype"], device=bop_si.device) for _ in range(4)]
        e = torch.zeros(c["N"], dtype=torch.float64, device=bop_si.device)
        if c["M"]:
            rk._bond_corrections_forward[(triton.cdiv(c["M"], rk.BLOCK),)](
                *c["args"](bop_si, bop_pi, bop_pp, deltap, spt, tab, gen), *outs, e, c["M"], c["T"], DT=c["dt"],
                BLOCK=rk.BLOCK, num_warps=rk.WARPS)
        ctx.save_for_backward(bop_si, bop_pi, bop_pp, deltap, spt, tab, gen)
        ctx.c = c
        return (*outs, e)

    @staticmethod
    def backward(ctx, g0, gsi, gpi, gpp, ge):
        from . import reaxff_kernels as rk
        from . import reaxff_tangents as rt
        inputs = ctx.saved_tensors
        c = ctx.c
        args = c["args"](*inputs)
        # the float inputs among the kernel's arguments (the species index at 6 is not one)
        grads = [torch.zeros_like(args[j]) for j in (2, 3, 4, 5, 7, 8, 9)]
        dev = inputs[0].device

        def given(t, n, dtype):
            return t.detach().contiguous() if t is not None else torch.zeros(n, dtype=dtype, device=dev)
        if c["M"]:
            scratch = [torch.zeros(c["M"], dtype=c["dtype"], device=dev) for _ in range(4)]
            rt._bond_corrections_forward_r[(triton.cdiv(c["M"], rk.BLOCK),)](
                *args, *scratch, torch.zeros(c["N"], dtype=torch.float64, device=dev), c["M"], c["T"],
                g_bo0_ptr=given(g0, c["M"], c["dtype"]), g_bosi_ptr=given(gsi, c["M"], c["dtype"]),
                g_bopi_ptr=given(gpi, c["M"], c["dtype"]), g_bopp_ptr=given(gpp, c["M"], c["dtype"]),
                g_ptr=given(ge, c["N"], torch.float64), bop_ptr_g=grads[0], bpi_ptr_g=grads[1], bpp_ptr_g=grads[2],
                dp_ptr_g=grads[3], spt_ptr_g=grads[4], tab_ptr_g=grads[5], gen_ptr_g=grads[6], DT=c["dt"],
                BLOCK=rk.BLOCK, num_warps=rk.WARPS)
        out = [gr.reshape(x.shape).to(x.dtype) for gr, x in zip(grads, inputs)]
        return _delayed(inputs, *out, None)


def bond_corrections(b_src: Tensor, b_dst: Tensor, bop_si: Tensor, bop_pi: Tensor, bop_pp: Tensor, deltap: Tensor,
                     species: Tensor, spt: Tensor, tab: Tensor, gen: Tensor, dtype: torch.dtype) -> tuple:
    """``(bo0, bosi, bopi, bopp)`` per bond and the per-atom bond energy ``(N,)`` in ``dtype`` of the bond list
    ``(b_src, b_dst)`` (LAMMPS conventions), from the uncorrected bond orders, the uncorrected coordination per
    atom, the per-species table ``(S, 2)`` (valency, correction valency), the per-species-pair table
    ``(S, S, BC_COLUMNS)`` and the general ``(boc1, boc2)``."""
    dt = triton.language.float64 if dtype == torch.float64 else triton.language.float32
    src_c, dst_c, t_c = b_src.contiguous(), b_dst.contiguous(), species.contiguous()

    def args(bsi, bpi, bpp, dp, sp, tb, gn):
        return (src_c, dst_c) + tuple(x.detach().contiguous().to(dtype) for x in (bsi, bpi, bpp, dp)) + (
            t_c, sp.detach().contiguous().to(dtype), tb.detach().contiguous().to(dtype),
            gn.detach().contiguous().to(dtype))
    c = {"args": args, "M": int(b_src.shape[0]), "N": int(deltap.shape[0]), "T": int(tab.shape[0]), "dt": dt,
         "dtype": dtype}
    bo0, bosi, bopi, bopp, e = BondCorrections.apply(bop_si, bop_pi, bop_pp, deltap, spt, tab, gen, c)
    return bo0, bosi, bopi, bopp, e.to(dtype)


class KernelTerms(torch.autograd.Function):
    """Per-atom energies ``(N, K)`` (float64) of a list of terms from a forward kernel of :mod:`.reaxff_kernels`
    whose arguments are index arrays (constants) followed by input arrays (the float ones differentiable), with
    the backward from the kernel's generated adjoint (its gradients into ``<name>_g`` for every float input)."""

    @staticmethod
    def forward(ctx, c, *inputs):
        from . import reaxff_kernels as rk
        out = torch.zeros(c["N"], c["K"], dtype=torch.float64, device=inputs[0].device)
        if c["M"]:
            getattr(rk, c["kernel"])[(triton.cdiv(c["M"], rk.BLOCK),)](
                *c["index"], *c["cast"](inputs), out, c["M"], *c["extra"], DT=c["dt"], **c["consts"],
                BLOCK=rk.BLOCK, num_warps=rk.WARPS)
        ctx.save_for_backward(*inputs)
        ctx.c = c
        return out

    @staticmethod
    def backward(ctx, g):
        from . import reaxff_kernels as rk
        from . import reaxff_tangents as rt
        inputs = ctx.saved_tensors
        c = ctx.c
        cast = c["cast"](inputs)
        # gradients of the float inputs (an integer input, such as the species index, has none)
        grads = {f"{name}_g": torch.zeros_like(x) for name, x in zip(c["names"], cast) if x.is_floating_point()}
        if c["M"]:
            scratch = torch.zeros(c["N"], c["K"], dtype=torch.float64, device=inputs[0].device)
            getattr(rt, c["kernel"] + "_r")[(triton.cdiv(c["M"], rk.BLOCK),)](
                *c["index"], *cast, scratch, c["M"], *c["extra"], g_ptr=g.detach().contiguous(), **grads,
                DT=c["dt"], **c["consts"], BLOCK=rk.BLOCK, num_warps=rk.WARPS)
        out = [grads[f"{name}_g"].reshape(x.shape).to(x.dtype) if f"{name}_g" in grads else None
               for name, x in zip(c["names"], inputs)]
        return _delayed(inputs, None, *out)


def kernel_terms(kernel: str, index: tuple, inputs: tuple, n_atoms: int, k: int, extra: tuple,
                 dtype: torch.dtype, consts: dict = None) -> Tensor:
    """The per-atom energies ``(N, k)`` in ``dtype`` of :class:`KernelTerms` for ``kernel`` of
    :mod:`.reaxff_kernels`; ``extra`` its integer arguments after the term count, ``consts`` its compile-time
    switches."""
    from . import reaxff_kernels as rk
    fn = getattr(rk, kernel)
    names = fn.arg_names[len(index):len(index) + len(inputs)]
    dt = triton.language.float64 if dtype == torch.float64 else triton.language.float32

    def cast(xs):
        return tuple(x.detach().contiguous().to(dtype) if x.is_floating_point() else x.contiguous() for x in xs)
    c = {"kernel": kernel, "index": tuple(i.contiguous() for i in index), "cast": cast, "names": names,
         "M": int(index[0].shape[0]), "N": int(n_atoms), "K": int(k), "extra": tuple(extra), "dt": dt,
         "consts": dict(consts or {})}
    return KernelTerms.apply(c, *inputs).to(dtype)


__all__ = ["fast_supported", "NonbondedTerms", "nonbonded", "BondCorrections", "bond_corrections", "KernelTerms",
           "kernel_terms"]
