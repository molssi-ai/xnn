"""The fast path of :class:`~xnn.ffnn.models.ReaxFF`: the terms on the kernels of :mod:`.reaxff_kernels`, their
backward from the generated adjoints of :mod:`.reaxff_tangents`. The numbers are those of the eager terms to
round-off. Differentiable once, in the positions, the cells and the inputs (charges, parameter tables).
"""
from typing import Optional

import torch
import triton
from torch import Tensor

from .classical_fast import fast_supported
from .classical_kernels import _cell_args, _delayed, _scalars


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
            rk._go(rk._nonbonded_forward, (triton.cdiv(c["E"], rk.BLOCK),), *c["args"](pos, cell, q, table), out,
                   c["E"], c["T"], **c["flags"], BLOCK=rk.BLOCK, num_warps=rk.WARPS)
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
        g_q = torch.zeros_like(args[7])
        # the cell and table gradients only where asked for (every pair adds into their few entries)
        full = (ctx.needs_input_grad[1] and c["flags"]["HAS_CELL"]) or ctx.needs_input_grad[3]
        # the charges' gradient only where asked for (the LAMMPS conventions' charges carry none)
        with_q = full or ctx.needs_input_grad[2]
        grads = {"pos_ptr_g": g_pos}
        if with_q:
            grads["q_ptr_g"] = g_q
        if full:
            grads.update(cell_ptr_g=torch.zeros_like(args[4]), tab_ptr_g=torch.zeros_like(args[8]))
        if c["E"] and not with_q:
            # the positions alone: each pair's force in closed form
            rk._go(rk._nonbonded_forces, (triton.cdiv(c["E"], rk.BLOCK),), *args, g.detach().contiguous(), g_pos,
                   c["E"], c["T"], **c["flags"], BLOCK=rk.BLOCK, num_warps=rk.WARPS)
        elif c["E"]:
            out = torch.zeros(pos.shape[0], 2, dtype=torch.float64, device=pos.device)
            kernel = rt._nonbonded_forward_r if full else rt._nonbonded_forward_rx
            rk._go(kernel, (triton.cdiv(c["E"], rk.BLOCK),), *args, out, c["E"], c["T"], g_ptr=g.detach().contiguous(),
                   **grads, **c["flags"], BLOCK=rk.BLOCK, num_warps=rk.WARPS)
        g_cell = g_t = None
        if full:
            g_t = grads["tab_ptr_g"].reshape(table.shape).to(table.dtype)
            if c["flags"]["HAS_CELL"]:
                g_cell = grads["cell_ptr_g"].reshape(cell.shape).to(cell.dtype)
        return _delayed((pos, cell, q, table), g_pos.to(pos.dtype), g_cell, g_q.to(q.dtype) if with_q else None, g_t,
                        None)


def nonbonded(pos: Tensor, cell: Optional[Tensor], batch: Tensor, src: Tensor, dst: Tensor,
              shifts: Optional[Tensor], species: Tensor, q: Tensor, table: Tensor, dtype: torch.dtype, ke: float,
              cutoff: float, vdw1: float, safety: float, shield: bool, wall: bool, twice: bool = False,
              forces: bool = False) -> tuple:
    """The per-atom van der Waals and Coulomb energies ``(N,)`` each in ``dtype`` of the directed pair list
    ``(src, dst)`` with image shifts ``shifts`` (or ``None``), the species index per atom, the charges and the
    ``(S, S, NB_COLUMNS)`` table of :func:`~.reaxff_kernels.nonbonded_table`; ``twice`` differentiable twice in
    the positions, charges and table (:mod:`.reaxff_train`, the cell a constant); ``forces`` instead the two
    total energies ``(2,)`` and their position gradient ``(N, 3)`` (float64) in one pass, not differentiable."""
    sc = _scalars((float(ke), float(cutoff), float(vdw1), float(safety), 1.0 / 3.0), pos.device)
    _, bat_t, s_t, has = _cell_args(pos, cell, batch, shifts)
    dt = triton.language.float64 if dtype == torch.float64 else triton.language.float32
    src_c, dst_c, t_c = src.contiguous(), dst.contiguous(), species.contiguous()

    def args(p, cl, qq, tab):
        cell_t = _cell_args(p, cl.detach(), batch, shifts)[0] if has else p.detach()
        return (src_c, dst_c, s_t, p.detach().contiguous(), cell_t, bat_t, t_c, qq.detach().contiguous().to(dtype),
                tab.detach().contiguous().to(dtype), sc)
    if twice:
        from .reaxff_train import kernel_terms_twice
        cell_t = _cell_args(pos, cell.detach(), batch, shifts)[0] if has else pos.detach()

        def cast(xs):
            sh, p, cl, bt, tt, qq, tab, scl = xs
            return (sh, p.detach().contiguous(), cl, bt, tt, qq.detach().contiguous().to(dtype),
                    tab.detach().contiguous().to(dtype), scl)
        from . import reaxff_kernels as rk
        names = rk._nonbonded_forward.arg_names[2:10]
        lc = {"kernel": "_nonbonded_forward", "index": (src_c, dst_c), "cast": cast, "names": names,
              "M": int(src.shape[0]), "N": int(pos.shape[0]), "K": 2, "extra": (int(table.shape[0]),), "dt": dt,
              "consts": dict(HAS_CELL=has, SHIELD=bool(shield), WALL=bool(wall))}
        e = kernel_terms_twice(lc, (s_t, pos, cell_t, bat_t, t_c, q, table, sc))
        return e[:, 0].to(dtype), e[:, 1].to(dtype)
    c = {"args": args, "E": int(src.shape[0]), "T": int(table.shape[0]),
         "flags": dict(DT=dt, HAS_CELL=has, SHIELD=bool(shield), WALL=bool(wall))}
    if forces:
        from . import reaxff_kernels as rk
        e = torch.zeros(2, dtype=torch.float64, device=pos.device)
        g = torch.zeros(pos.shape[0], 3, dtype=torch.float64, device=pos.device)
        if c["E"]:
            rk._go(rk._nonbonded_energy_forces, (triton.cdiv(c["E"], rk.BLOCK),), *args(pos, cell if has else pos, q,
                   table), e, g, c["E"], c["T"], **c["flags"], BLOCK=rk.BLOCK, num_warps=rk.WARPS)
        return e, g
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
            rk._go(rk._bond_corrections_forward, (triton.cdiv(c["M"], rk.BLOCK),), *c["args"](bop_si, bop_pi, bop_pp,
                   deltap, spt, tab, gen), *outs, e, c["M"], c["T"], DT=c["dt"], ACT=c["ACT"], BLOCK=rk.BLOCK,
                   num_warps=rk.WARPS)
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
        # the float inputs among the kernel's arguments (the species index at 6 is not one); the tables' gradients
        # only where asked for
        full = any(ctx.needs_input_grad[4:7])
        grads = [torch.zeros_like(args[j]) for j in ((2, 3, 4, 5, 7, 8, 9) if full else (2, 3, 4, 5))]
        dev = inputs[0].device

        def given(t, n, dtype):
            return t.detach().contiguous() if t is not None else torch.zeros(n, dtype=dtype, device=dev)
        if c["M"]:
            scratch = [torch.zeros(c["M"], dtype=c["dtype"], device=dev) for _ in range(4)]
            names = ("bop_ptr_g", "bpi_ptr_g", "bpp_ptr_g", "dp_ptr_g", "spt_ptr_g", "tab_ptr_g", "gen_ptr_g")
            kernel = rt._bond_corrections_forward_r if full else rt._bond_corrections_forward_rx
            rk._go(kernel, (triton.cdiv(c["M"], rk.BLOCK),), *args, *scratch, torch.zeros(c["N"], dtype=torch.float64,
                   device=dev), c["M"], c["T"], g_bo0_ptr=given(g0, c["M"], c["dtype"]), g_bosi_ptr=given(gsi, c["M"],
                   c["dtype"]), g_bopi_ptr=given(gpi, c["M"], c["dtype"]), g_bopp_ptr=given(gpp, c["M"], c["dtype"]),
                   g_ptr=given(ge, c["N"], torch.float64), **dict(zip(names, grads)), DT=c["dt"], ACT=c["ACT"],
                   BLOCK=rk.BLOCK, num_warps=rk.WARPS)
        out = [gr.reshape(x.shape).to(x.dtype) for gr, x in zip(grads, inputs)] + [None] * (7 - len(grads))
        return _delayed(inputs, *out, None)


def bond_corrections(b_src: Tensor, b_dst: Tensor, bop_si: Tensor, bop_pi: Tensor, bop_pp: Tensor, deltap: Tensor,
                     species: Tensor, spt: Tensor, tab: Tensor, gen: Tensor, dtype: torch.dtype,
                     act: Optional[Tensor] = None) -> tuple:
    """``(bo0, bosi, bopi, bopp)`` per bond and the per-atom bond energy ``(N,)`` in ``dtype`` of the bond list
    ``(b_src, b_dst)`` (LAMMPS conventions), from the uncorrected bond orders, the uncorrected coordination per
    atom, the per-species table ``(S, 2)`` (valency, correction valency), the per-species-pair table
    ``(S, S, BC_COLUMNS)`` and the general ``(boc1, boc2)``; ``act`` the integer flags of the bonds that take
    part (the others get zero bond orders; all without)."""
    dt = triton.language.float64 if dtype == torch.float64 else triton.language.float32
    src_c, dst_c, t_c = b_src.contiguous(), b_dst.contiguous(), species.contiguous()

    def args(bsi, bpi, bpp, dp, sp, tb, gn):
        return (src_c, dst_c) + tuple(x.detach().contiguous().to(dtype) for x in (bsi, bpi, bpp, dp)) + (
            t_c, sp.detach().contiguous().to(dtype), tb.detach().contiguous().to(dtype),
            gn.detach().contiguous().to(dtype), src_c if act is None else act)
    c = {"args": args, "M": int(b_src.shape[0]), "N": int(deltap.shape[0]), "T": int(tab.shape[0]), "dt": dt,
         "dtype": dtype, "ACT": act is not None}
    bo0, bosi, bopi, bopp, e = BondCorrections.apply(bop_si, bop_pi, bop_pp, deltap, spt, tab, gen, c)
    return bo0, bosi, bopi, bopp, e.to(dtype)


class BondOrders(torch.autograd.Function):
    """A stepper's uncorrected bond orders (sigma less botol, pi, double pi) per bond candidate and the uncorrected
    coordination per atom from the candidates' distances (:func:`~.reaxff_kernels._bond_orders_forward`),
    differentiable in the distances; the flags of the candidates that take part (int32) last."""

    @staticmethod
    def forward(ctx, br, c):
        from . import reaxff_kernels as rk
        outs = [torch.zeros(c["M"], dtype=c["dtype"], device=br.device) for _ in range(3)]
        act = torch.zeros(c["M"], dtype=torch.int32, device=br.device)
        dp = torch.zeros(c["N"], dtype=c["dtype"], device=br.device)
        if c["M"]:
            rk._go(rk._bond_orders_forward, (triton.cdiv(c["M"], rk.BLOCK),), *c["pre"], br.detach().contiguous(),
                   *c["post"], *outs, act, dp, c["M"], c["T"], DT=c["dt"], BLOCK=rk.BLOCK, num_warps=rk.WARPS)
        ctx.save_for_backward(br)
        ctx.c = c
        ctx.mark_non_differentiable(act)
        return (*outs, dp, act)

    @staticmethod
    def backward(ctx, gsi, gpi, gpp, gdp, _):
        from . import reaxff_kernels as rk
        from . import reaxff_tangents as rt
        (br,) = ctx.saved_tensors
        c = ctx.c
        dev = br.device
        br_g = torch.zeros(c["M"], dtype=c["dtype"], device=dev)

        def given(t, n):
            return t.detach().contiguous().to(c["dtype"]) if t is not None else torch.zeros(n, dtype=c["dtype"],
                                                                                            device=dev)
        if c["M"]:
            scratch = [torch.zeros(c["M"], dtype=c["dtype"], device=dev) for _ in range(3)]
            rk._go(rt._bond_orders_forward_rx, (triton.cdiv(c["M"], rk.BLOCK),), *c["pre"], br.detach().contiguous(),
                   *c["post"], *scratch, torch.zeros(c["M"], dtype=torch.int32, device=dev),
                   torch.zeros(c["N"], dtype=c["dtype"], device=dev), c["M"], c["T"], g_bsi_ptr=given(gsi, c["M"]),
                   g_bpi_ptr=given(gpi, c["M"]), g_bpp_ptr=given(gpp, c["M"]), g_dp_ptr=given(gdp, c["N"]),
                   br_ptr_g=br_g, DT=c["dt"], BLOCK=rk.BLOCK, num_warps=rk.WARPS)
        return br_g.to(br.dtype), None


def bond_orders(b_src: Tensor, b_dst: Tensor, b_r: Tensor, valid: Tensor, species: Tensor, ch: Tensor, sc: Tensor,
                n: int, dtype: torch.dtype) -> tuple:
    """The uncorrected bond orders ``(bop_si, bop_pi, bop_pp)`` of a stepper's bond candidates at the distances
    ``b_r`` (LAMMPS conventions, constant parameters: the pair table ``ch`` ``(S, S, BO_COLUMNS)`` of
    :mod:`.reaxff_kernels` and ``sc`` (botol, the bond cutoff)), the uncorrected coordination ``(n,)`` and the
    flags of the candidates that take part (``valid`` set, within the cutoff, at least botol)."""
    dt = triton.language.float64 if dtype == torch.float64 else triton.language.float32
    c = {"pre": (b_src.contiguous(), b_dst.contiguous()),
         "post": (valid.contiguous(), species.contiguous(), ch.detach().contiguous().to(dtype),
                  sc.detach().contiguous().to(dtype)),
         "M": int(b_src.shape[0]), "N": int(n), "T": int(ch.shape[0]), "dt": dt, "dtype": dtype}
    return BondOrders.apply(b_r.to(dtype), c)


class AtomTerms(torch.autograd.Function):
    """A stepper's lone-pair, over- and under-coordination, C-C over-coordination and triple-bond stabilization
    energies per atom (float64) and the lone pairs per atom from the corrected bond orders and the coordinations
    (:func:`~.reaxff_kernels._bond_atom_sums_forward`, :func:`~.reaxff_kernels._atom_terms_forward`),
    differentiable in both."""

    @staticmethod
    def forward(ctx, bo0, bopi, bopp, delta, c):
        from . import reaxff_kernels as rk
        dev, N, M = bo0.device, c["N"], c["M"]
        ins = [x.detach().contiguous().to(c["dtype"]) for x in (bo0, bopi, bopp, delta)]
        so, dpil = (torch.zeros(N, dtype=c["dtype"], device=dev) for _ in range(2))
        e = torch.zeros(N, dtype=torch.float64, device=dev)
        nlp = torch.zeros(N, dtype=c["dtype"], device=dev)
        if M:
            rk._go(rk._bond_atom_sums_forward, (triton.cdiv(M, rk.BLOCK),), *c["bonds"], *ins, *c["tables"], so, dpil,
                   e, M, c["T"], DT=c["dt"], BLOCK=rk.BLOCK, num_warps=rk.WARPS)
        if N:
            rk._go(rk._atom_terms_forward, (triton.cdiv(N, rk.BLOCK),), ins[3], so, dpil, *c["atoms"], nlp, e, N,
                   DT=c["dt"], BLOCK=rk.BLOCK, num_warps=rk.WARPS)
        ctx.save_for_backward(*ins, so, dpil)
        ctx.c = c
        return e, nlp

    @staticmethod
    def backward(ctx, ge, gnlp):
        from . import reaxff_kernels as rk
        from . import reaxff_tangents as rt
        bo0, bopi, bopp, delta, so, dpil = ctx.saved_tensors
        c = ctx.c
        dev, N, M, dtype = bo0.device, c["N"], c["M"], c["dtype"]
        ge = torch.zeros(N, dtype=torch.float64, device=dev) if ge is None else ge.detach().contiguous()
        gnlp = torch.zeros(N, dtype=dtype, device=dev) if gnlp is None else gnlp.detach().contiguous().to(dtype)
        del3, dpil_g, so_g = (torch.zeros(N, dtype=dtype, device=dev) for _ in range(3))
        if N:
            rk._go(rt._atom_terms_forward_rx, (triton.cdiv(N, rk.BLOCK),), delta, so, dpil, *c["atoms"],
                   torch.zeros(N, dtype=dtype, device=dev), torch.zeros(N, dtype=torch.float64, device=dev), N,
                   g_nlp_ptr=gnlp, g_ptr=ge, del_ptr_g=del3, dpil_ptr_g=dpil_g, so_ptr_g=so_g, DT=c["dt"],
                   BLOCK=rk.BLOCK, num_warps=rk.WARPS)
        grads = [torch.zeros(M, dtype=dtype, device=dev) for _ in range(3)] + [torch.zeros(N, dtype=dtype, device=dev)]
        if M:
            rk._go(rt._bond_atom_sums_forward_rx, (triton.cdiv(M, rk.BLOCK),), *c["bonds"], bo0, bopi, bopp, delta,
                   *c["tables"], torch.zeros(N, dtype=dtype, device=dev), torch.zeros(N, dtype=dtype, device=dev),
                   torch.zeros(N, dtype=torch.float64, device=dev), M, c["T"], g_so_ptr=so_g, g_dpil_ptr=dpil_g,
                   g_ptr=ge, bo0_ptr_g=grads[0], bpi_ptr_g=grads[1], bpp_ptr_g=grads[2], del_ptr_g=grads[3],
                   DT=c["dt"], BLOCK=rk.BLOCK, num_warps=rk.WARPS)
        return grads[0], grads[1], grads[2], grads[3] + del3, None


def atom_terms(b_src: Tensor, b_dst: Tensor, bo0: Tensor, bopi: Tensor, bopp: Tensor, delta: Tensor, species: Tensor,
               at: Tensor, ap: Tensor, gen: Tensor, dtype: torch.dtype) -> tuple:
    """The per-atom energies (float64) of the lone pairs, the over- and under-coordination, the C-C
    over-coordination and the triple-bond stabilization of a stepper's bonds and the lone pairs per atom (LAMMPS
    conventions, constant parameters: the tables ``at``, ``ap`` and ``gen`` of :mod:`.reaxff_kernels`)."""
    dt = triton.language.float64 if dtype == torch.float64 else triton.language.float32
    t_c = species.contiguous()
    tables = (t_c, at.detach().contiguous().to(dtype), ap.detach().contiguous().to(dtype),
              gen.detach().contiguous().to(dtype))
    c = {"bonds": (b_src.contiguous(), b_dst.contiguous()), "tables": tables,
         "atoms": (t_c, tables[1], tables[3]), "M": int(b_src.shape[0]), "N": int(delta.shape[0]),
         "T": int(at.shape[0]), "dt": dt, "dtype": dtype}
    return AtomTerms.apply(bo0, bopi, bopp, delta, c)


class KernelTerms(torch.autograd.Function):
    """Per-atom energies ``(N, K)`` (float64) of a list of terms from a forward kernel of :mod:`.reaxff_kernels`
    whose arguments are index arrays (constants) followed by input arrays (the float ones differentiable), with
    the backward from the kernel's generated adjoint (its gradients into ``<name>_g`` for every float input)."""

    @staticmethod
    def forward(ctx, c, *inputs):
        from . import reaxff_kernels as rk
        out = torch.zeros(c["N"], c["K"], dtype=torch.float64, device=inputs[0].device)
        if c["M"]:
            rk._go(getattr(rk, c["kernel"]), (triton.cdiv(c["M"], rk.BLOCK),), *c["index"], *c["cast"](inputs), out,
                   c["M"], *c["extra"], DT=c["dt"], **c["consts"], BLOCK=rk.BLOCK, num_warps=rk.WARPS)
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
        # gradients of the float inputs (an integer input, such as the species index, has none); the adjoint in
        # the per-atom and per-bond inputs only (_rx) where no parameter table asks for one
        grads = {f"{name}_g": torch.zeros_like(x) for name, x in zip(c["names"], cast) if x.is_floating_point()}
        kernel = getattr(rt, c["kernel"] + "_r")
        lean = getattr(rt, c["kernel"] + "_rx", None)
        if lean is not None:
            wanted = {f"{name}_g" for name, x, need in zip(c["names"], cast, ctx.needs_input_grad[1:])
                      if x.is_floating_point() and need}
            if wanted <= set(lean.arg_names):
                kernel = lean
        # the gradients of the arrays the adjoint takes (an input it reads as a constant, such as an offset, has none)
        grads = {k: v for k, v in grads.items() if k in kernel.arg_names}
        if c["M"]:
            scratch = torch.zeros(c["N"], c["K"], dtype=torch.float64, device=inputs[0].device)
            rk._go(kernel, (triton.cdiv(c["M"], rk.BLOCK),), *c["index"], *cast, scratch, c["M"], *c["extra"],
                   g_ptr=g.detach().contiguous(), **grads, DT=c["dt"], **c["consts"], BLOCK=rk.BLOCK,
                   num_warps=rk.WARPS)
        out = [grads[f"{name}_g"].reshape(x.shape).to(x.dtype) if f"{name}_g" in grads else None
               for name, x in zip(c["names"], inputs)]
        return _delayed(inputs, None, *out)


def kernel_terms(kernel: str, index: tuple, inputs: tuple, n_atoms: int, k: int, extra: tuple,
                 dtype: torch.dtype, consts: dict = None, twice: bool = False, act: Optional[Tensor] = None,
                 keep: tuple = ()) -> Tensor:
    """The per-atom energies ``(N, k)`` in ``dtype`` of :class:`KernelTerms` for ``kernel`` of
    :mod:`.reaxff_kernels`; ``extra`` its integer arguments after the term count, ``consts`` its compile-time
    switches; ``twice`` differentiable twice (:mod:`.reaxff_train`); ``act`` the integer flags of the terms that
    take part (all without); ``keep`` the arguments passed in their own precision (positions)."""
    from . import reaxff_kernels as rk
    fn = getattr(rk, kernel)
    consts = dict(consts or {})
    if "act_ptr" in fn.arg_names:
        inputs = tuple(inputs) + (index[0] if act is None else act,)
        consts["ACT"] = consts.get("ACT", True) if act is not None else False
    names = fn.arg_names[len(index):len(index) + len(inputs)]
    dt = triton.language.float64 if dtype == torch.float64 else triton.language.float32

    def cast(xs):
        return tuple(x.detach().contiguous().to(dtype) if x.is_floating_point() and name not in keep
                     else x.detach().contiguous() for name, x in zip(names, xs))
    c = {"kernel": kernel, "index": tuple(i.contiguous() for i in index), "cast": cast, "names": names,
         "M": int(index[0].shape[0]), "N": int(n_atoms), "K": int(k), "extra": tuple(extra), "dt": dt,
         "consts": dict(consts or {})}
    if twice:
        from .reaxff_train import kernel_terms_twice
        return kernel_terms_twice(c, inputs).to(dtype)
    return KernelTerms.apply(c, *inputs).to(dtype)


__all__ = ["fast_supported", "NonbondedTerms", "nonbonded", "BondCorrections", "bond_corrections", "KernelTerms",
           "kernel_terms"]
