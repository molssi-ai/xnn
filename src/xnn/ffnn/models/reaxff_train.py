"""The kernels of :mod:`.reaxff_kernels` differentiable twice, for training: the per-atom energies of a term
kernel, its gradients in the inputs (positions or bond vectors, per-atom and per-bond arrays, parameter tables)
as a function that is itself differentiable. The second derivatives run the generated kernels of
:mod:`.reaxff_tangents`: the adjoint's tangent along the incoming direction (the Hessian-vector product, into
every input) and the forward kernel's tangent (the gradient of the incoming energy gradients). The cell enters
as a constant: a stress loss takes the reference terms.
"""
import torch
import triton
from torch import Tensor
from torch.autograd.function import once_differentiable


def _launch(c: dict, kernel, cast: tuple, out: Tensor, **kw):
    from . import reaxff_kernels as rk
    rk._go(kernel, (triton.cdiv(c["M"], rk.BLOCK),), *c["index"], *cast, out, c["M"], *c["extra"], **kw, DT=c["dt"],
           **c["consts"], BLOCK=rk.BLOCK, num_warps=rk.WARPS)


class _KernelGrad(torch.autograd.Function):
    """The gradients of :class:`KernelTermsTwice` in its differentiable inputs from the incoming gradients ``g``
    of the per-atom energies, differentiable in the inputs and ``g``."""

    @staticmethod
    def forward(ctx, c, g, *inputs):
        from . import reaxff_tangents as rt
        cast = c["cast"](inputs)
        adjoint = getattr(rt, c["kernel"] + "_r")
        grads = {f"{name}_g": torch.zeros_like(x) for name, x in zip(c["names"], cast)
                 if f"{name}_g" in adjoint.arg_names}
        if c["M"]:
            out = torch.zeros(c["N"], c["K"], dtype=torch.float64, device=g.device)
            _launch(c, adjoint, cast, out, g_ptr=g, **grads)
        ctx.save_for_backward(g, *inputs)
        ctx.c = c
        return tuple(grads[f"{name}_g"].reshape(inputs[j].shape).to(inputs[j].dtype) for j, name in c["diff"])

    @staticmethod
    @once_differentiable
    def backward(ctx, *u):
        from . import reaxff_tangents as rt
        g, *inputs = ctx.saved_tensors
        c = ctx.c
        cast = c["cast"](inputs)
        tangents = {f"{name}_t": (torch.zeros_like(cast[j]) if v is None else v.contiguous().to(cast[j].dtype))
                    for (j, name), v in zip(c["diff"], u)}
        g_g = None
        g_in = [None] * len(inputs)
        if c["M"]:
            if ctx.needs_input_grad[1]:
                out = torch.zeros(c["N"], c["K"], dtype=torch.float64, device=g.device)
                g_g = torch.zeros_like(out)
                _launch(c, getattr(rt, c["kernel"] + "_t"), cast, out, **tangents, out_ptr_t=g_g)
                g_g = g_g.to(g.dtype)
            wanted = [(j, name) for j, name in c["diff"] if ctx.needs_input_grad[2 + j]]
            if wanted:
                kernel = getattr(rt, c["kernel"] + "_r_t")
                grads = {f"{name}_g": torch.zeros_like(x) for name, x in zip(c["names"], cast)
                         if f"{name}_g" in kernel.arg_names}
                hvp = {f"{name}_g_t": torch.zeros_like(cast[j]) for j, name in c["diff"]}
                out = torch.zeros(c["N"], c["K"], dtype=torch.float64, device=g.device)
                _launch(c, kernel, cast, out, g_ptr=g, **grads, **tangents, **hvp)
                for j, name in wanted:
                    g_in[j] = hvp[f"{name}_g_t"].reshape(inputs[j].shape).to(inputs[j].dtype)
        return (None, g_g, *g_in)


class KernelTermsTwice(torch.autograd.Function):
    """:class:`~.reaxff_fast.KernelTerms` differentiable twice: the per-atom energies ``(N, K)`` (float64) of a
    term kernel, the backward through :class:`_KernelGrad`."""

    @staticmethod
    def forward(ctx, c, *inputs):
        from . import reaxff_kernels as rk
        out = torch.zeros(c["N"], c["K"], dtype=torch.float64, device=inputs[0].device)
        if c["M"]:
            _launch(c, getattr(rk, c["kernel"]), c["cast"](inputs), out)
        ctx.save_for_backward(*inputs)
        ctx.c = c
        return out

    @staticmethod
    def backward(ctx, g):
        inputs = ctx.saved_tensors
        c = ctx.c
        grads = _KernelGrad.apply(c, g.contiguous(), *inputs)
        out = [None] * len(inputs)
        for (j, _), gr in zip(c["diff"], grads):
            out[j] = gr
        return (None, *out)


def kernel_terms_twice(c: dict, inputs: tuple) -> Tensor:
    """The per-atom energies ``(N, K)`` (float64) of :class:`KernelTermsTwice` for the launch ``c`` of
    :func:`~.reaxff_fast.kernel_terms`; the differentiable inputs those with a tangent in the generated
    kernels."""
    from . import reaxff_tangents as rt
    tangent = getattr(rt, c["kernel"] + "_t").arg_names
    c = dict(c, diff=tuple((j, name) for j, name in enumerate(c["names"]) if f"{name}_t" in tangent))
    return KernelTermsTwice.apply(c, *inputs)


__all__ = ["KernelTermsTwice", "kernel_terms_twice"]
