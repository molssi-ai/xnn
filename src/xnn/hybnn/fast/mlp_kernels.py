"""Triton kernels and autograd functions of SpookyNet's fused residual MLPs (see :mod:`xnn.hybnn.fast.spookynet`).

A residual MLP of SpookyNet is a stack of residual blocks ``h + L_B(L_A(h))``, optionally followed by an
output layer, where every layer is ``L(h) = swish(h) W + b`` with the learned activation
``swish(h) = alpha h sigmoid(beta h)`` (per-feature ``alpha``, ``beta``). Here ``G`` such MLPs of one
structure run at once on one input (the grouped evaluation; ``G = 1`` for a single one), and each of the
three passes a force loss needs is one autograd function for the whole MLP:

* the forward;
* the backward: the layers in reverse, each a product, the activation's backward and, when the pass
  needs them, the weight gradients. The force derivative needs no weight gradients and computes none,
  as autograd does for its own operations;
* the backward of the backward (second derivatives): the incoming gradient's tangent pushed forward
  through the MLP, then one reverse sweep of the adjoints of the values and of the tangents.

The activation's elementwise work is one Triton kernel per layer and pass; the sums over rows for
``alpha`` and ``beta`` are partial sums per program, added in a fixed order afterwards (deterministic). The
products run on cuBLAS. Third derivatives, and second derivatives through the weight gradients, are not
supported. Imported only when the fast path runs, so that ``triton`` is needed only there.
"""
# no postponed annotations: Triton's interpreter (TRITON_INTERPRET=1) recognizes the
# tl.constexpr parameters of a kernel only from evaluated annotations

import functools
import os
from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import once_differentiable

#: Elements per program (rows times padded features).
TILE = 1024

#: Triton's interpreter (CPU tests): launches through Triton's own path.
_INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"

#: Compiled kernels by kernel and argument specialization.
_COMPILED: dict = {}

#: Whether a compiled kernel takes the constexpr arguments too (Triton 3.3 on; earlier versions take only
#: the others).
_ALL_ARGUMENTS = tuple(int(v) for v in triton.__version__.split(".")[:2]) >= (3, 3)


def _launch_hooks(compiled):
    """The launch enter and exit hooks Triton passes to a compiled kernel's launcher: the runtime knobs'
    (Triton 3.4 on) or the compiled kernel's."""
    knobs = getattr(triton, "knobs", None)
    if knobs is None:
        return compiled.launch_enter_hook, compiled.launch_exit_hook
    return knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook


def _specialization(arg, constexpr: bool):
    """What Triton compiles a kernel argument into: the value of a constexpr, the dtype and 16-byte
    alignment of a tensor, whether an integer is 1, divisible by 16 or beyond 32 bits."""
    if constexpr:
        return arg
    if isinstance(arg, Tensor):
        return arg.dtype, arg.data_ptr() % 16 == 0
    if isinstance(arg, int):
        return arg == 1, arg % 16 == 0, -2 ** 31 <= arg < 2 ** 31
    return arg


def _go(kernel, grid: Tuple[int, int], *args) -> None:
    """Launch ``kernel`` with all its arguments in order (constexprs included). The first launch of a
    specialization goes through Triton's launcher, which compiles; later ones launch the compiled
    kernel directly, without the launcher's per-argument work on the host (most of a small kernel's
    launch time)."""
    if _INTERPRETED:
        kernel[grid](*args)
        return
    constexpr = _CONSTEXPR.get(kernel)
    if constexpr is None:
        constexpr = _CONSTEXPR[kernel] = tuple(p.is_constexpr for p in kernel.params)
    key = (kernel, tuple(_specialization(a, c) for a, c in zip(args, constexpr)))
    compiled = _COMPILED.get(key)
    if compiled is None:
        _COMPILED[key] = kernel[grid](*args)
        return
    if not _ALL_ARGUMENTS:
        args = tuple(a for a, c in zip(args, constexpr) if not c)
    stream = torch.cuda.current_stream(args[0].device).cuda_stream
    compiled.run(grid[0], grid[1], 1, stream, compiled.function, compiled.packed_metadata,
                 compiled.launch_metadata(grid, stream, *args), *_launch_hooks(compiled), *args)


#: Which arguments of a kernel are constexprs.
_CONSTEXPR: dict = {}


@triton.jit
def _activation_kernel(h_ptr, alpha_ptr, beta_ptr, t_ptr, out_ptr, n_rows, F: tl.constexpr, FP: tl.constexpr,
                       BLOCK_N: tl.constexpr, TANGENT: tl.constexpr):
    """``a = alpha h s`` or, with ``TANGENT``, its tangent ``alpha (s + u s') t`` for the tangent ``t`` of
    ``h`` (``u = beta h``, ``s = sigmoid(u)``, ``s' = s (1 - s)``), for one block of rows of one MLP."""
    grp = tl.program_id(1).to(tl.int64)
    rows = (tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)[:, None]
    f = tl.arange(0, FP)[None, :]
    m = (rows < n_rows) & (f < F)
    offs = grp * n_rows * F + rows * F + f
    h = tl.load(h_ptr + offs, mask=m, other=0.0)
    alpha = tl.load(alpha_ptr + grp * F + f, mask=f < F, other=0.0)
    beta = tl.load(beta_ptr + grp * F + f, mask=f < F, other=0.0)
    u = beta * h
    s = tl.sigmoid(u)
    if TANGENT:
        t = tl.load(t_ptr + offs, mask=m, other=0.0)
        out = alpha * (s + u * s * (1.0 - s)) * t
    else:
        out = alpha * h * s
    tl.store(out_ptr + offs, out, mask=m)


@triton.jit
def _activation_backward_kernel(h_ptr, alpha_ptr, beta_ptr, ga_ptr, gh_ptr, part_ptr, n_rows, n_blocks,
                                F: tl.constexpr, FP: tl.constexpr, BLOCK_N: tl.constexpr, PARAMS: tl.constexpr):
    """``gh = ga alpha (s + u s')`` and, with ``PARAMS``, this program's sums over rows of ``ga h s`` (for
    ``alpha``) and ``ga alpha h^2 s'`` (for ``beta``)."""
    grp = tl.program_id(1).to(tl.int64)
    pid = tl.program_id(0)
    rows = (pid * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)[:, None]
    f = tl.arange(0, FP)[None, :]
    mf = f < F
    m = (rows < n_rows) & mf
    offs = grp * n_rows * F + rows * F + f
    h = tl.load(h_ptr + offs, mask=m, other=0.0)
    ga = tl.load(ga_ptr + offs, mask=m, other=0.0)
    alpha = tl.load(alpha_ptr + grp * F + f, mask=mf, other=0.0)
    beta = tl.load(beta_ptr + grp * F + f, mask=mf, other=0.0)
    u = beta * h
    s = tl.sigmoid(u)
    s1 = s * (1.0 - s)
    tl.store(gh_ptr + offs, ga * alpha * (s + u * s1), mask=m)
    if PARAMS:
        fp = tl.arange(0, FP)
        base = part_ptr + ((grp * n_blocks + pid) * 2) * F
        tl.store(base + fp, tl.sum(ga * h * s, axis=0), mask=fp < F)
        tl.store(base + F + fp, tl.sum(ga * alpha * h * h * s1, axis=0), mask=fp < F)


@triton.jit
def _activation_double_kernel(h_ptr, th_ptr, alpha_ptr, beta_ptr, mu_ptr, lam_ptr, mu_h_ptr, lam_h_ptr, part_ptr,
                              n_rows, n_blocks, F: tl.constexpr, FP: tl.constexpr, BLOCK_N: tl.constexpr,
                              HAS_LAMBDA: tl.constexpr):
    """One layer's activation in the reverse sweep of the second derivatives. ``mu`` is the adjoint of the
    activation's tangent ``alpha (s + u s') th``, ``lam`` that of the activation (``HAS_LAMBDA``). Writes
    the adjoint of ``h``'s tangent, ``mu_h = mu alpha (s + u s')``, the adjoint of ``h``,
    ``lam_h = mu th alpha beta (2 s' + u s'') + lam alpha (s + u s')``, and this program's sums over rows
    of the derivatives with respect to ``alpha`` and ``beta`` (``s'' = s' (1 - 2 s)``)."""
    grp = tl.program_id(1).to(tl.int64)
    pid = tl.program_id(0)
    rows = (pid * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)[:, None]
    f = tl.arange(0, FP)[None, :]
    mf = f < F
    m = (rows < n_rows) & mf
    offs = grp * n_rows * F + rows * F + f
    h = tl.load(h_ptr + offs, mask=m, other=0.0)
    th = tl.load(th_ptr + offs, mask=m, other=0.0)
    mu = tl.load(mu_ptr + offs, mask=m, other=0.0)
    alpha = tl.load(alpha_ptr + grp * F + f, mask=mf, other=0.0)
    beta = tl.load(beta_ptr + grp * F + f, mask=mf, other=0.0)
    u = beta * h
    s = tl.sigmoid(u)
    s1 = s * (1.0 - s)
    s2 = s1 * (1.0 - 2.0 * s)
    d1 = s + u * s1
    e = 2.0 * s1 + u * s2
    mu_th = mu * th
    lam_h = mu_th * alpha * beta * e
    dalpha = mu_th * d1
    dbeta = mu_th * alpha * h * e
    if HAS_LAMBDA:
        lam = tl.load(lam_ptr + offs, mask=m, other=0.0)
        lam_h += lam * alpha * d1
        dalpha += lam * h * s
        dbeta += lam * alpha * h * h * s1
    tl.store(mu_h_ptr + offs, mu * alpha * d1, mask=m)
    tl.store(lam_h_ptr + offs, lam_h, mask=m)
    fp = tl.arange(0, FP)
    base = part_ptr + ((grp * n_blocks + pid) * 2) * F
    tl.store(base + fp, tl.sum(dalpha, axis=0), mask=fp < F)
    tl.store(base + F + fp, tl.sum(dbeta, axis=0), mask=fp < F)


@functools.lru_cache(maxsize=None)
def _launch(n_rows: int, n_features: int) -> Tuple[int, int, int]:
    """Padded feature count, rows per program and programs per MLP."""
    fp = triton.next_power_of_2(n_features)
    block = max(1, min(triton.next_power_of_2(max(n_rows, 1)), max(1, TILE // fp)))
    return fp, block, max(1, triton.cdiv(n_rows, block))


def _activation(h: Tensor, alpha: Tensor, beta: Tensor, tangent: Optional[Tensor] = None) -> Tensor:
    g, n, f = h.shape
    out = torch.empty_like(h)
    if n:
        fp, block, nb = _launch(n, f)
        _go(_activation_kernel, (nb, g), h, alpha, beta, h if tangent is None else tangent, out, n, f, fp, block,
            tangent is not None)
    return out


def _activation_backward(h: Tensor, alpha: Tensor, beta: Tensor, ga: Tensor, part: Optional[Tensor]) -> Tensor:
    g, n, f = h.shape
    gh = torch.empty_like(h)
    if n:
        fp, block, nb = _launch(n, f)
        _go(_activation_backward_kernel, (nb, g), h, alpha, beta, ga, gh, h if part is None else part, n, nb, f, fp,
            block, part is not None)
    return gh


def _activation_double(h: Tensor, th: Tensor, alpha: Tensor, beta: Tensor, mu: Tensor, lam: Optional[Tensor],
                       part: Tensor) -> Tuple[Tensor, Tensor]:
    g, n, f = h.shape
    mu_h = torch.empty_like(h)
    lam_h = torch.empty_like(h)
    if n:
        fp, block, nb = _launch(n, f)
        _go(_activation_double_kernel, (nb, g), h, th, alpha, beta, mu, mu if lam is None else lam, mu_h, lam_h,
            part, n, nb, f, fp, block, lam is not None)
    return mu_h, lam_h


def _linear(a: Tensor, w: Tensor, b: Optional[Tensor]) -> Tensor:
    return torch.bmm(a, w) if b is None else torch.baddbmm(b.unsqueeze(1), a, w)


def _needed(ctx) -> List[bool]:
    """Which tensor inputs of an autograd function the running backward pass needs gradients for."""
    out = []
    for fn, _ in ctx.next_functions:
        if fn is None:
            out.append(False)
            continue
        try:
            out.append(bool(torch._C._will_engine_execute_node(fn)))
        except RuntimeError:            # a leaf in autograd.grad: the engine cannot tell
            out.append(True)
    return out


class _Spec:
    """The structure of the MLPs: residual blocks, whether an output layer follows, biases. The layers are
    ``A_0, B_0, ..., A_{R-1}, B_{R-1}`` and the output layer ``O``; each has the parameters
    ``(alpha, beta, w[, b])`` in the flat parameter list."""

    def __init__(self, n_residual: int, output: bool, bias: bool):
        self.n_residual = n_residual
        self.output = output
        self.bias = bias
        self.per_layer = 4 if bias else 3
        self.n_layers = 2 * n_residual + (1 if output else 0)

    def layer(self, params, i: int):
        p = params[i * self.per_layer:(i + 1) * self.per_layer]
        return p[0], p[1], p[2], (p[3] if self.bias else None)


class MLPFunction(torch.autograd.Function):
    """``G`` residual MLPs on one input ``x`` ``(N, F)``: outputs ``(G, N, F')``."""

    @staticmethod
    def forward(ctx, spec: _Spec, x: Tensor, *params: Tensor) -> Tensor:
        g = params[0].shape[0]
        h = x.unsqueeze(0).expand(g, x.shape[0], x.shape[1]).contiguous()
        saved = []
        for r in range(spec.n_residual):
            alpha, beta, w, b = spec.layer(params, 2 * r)
            a = _activation(h, alpha, beta)
            z = _linear(a, w, b)
            alpha2, beta2, w2, b2 = spec.layer(params, 2 * r + 1)
            a2 = _activation(z, alpha2, beta2)
            saved += [h, a, z, a2]
            h = h + _linear(a2, w2, b2)
        out = h
        if spec.output:
            alpha, beta, w, b = spec.layer(params, 2 * spec.n_residual)
            a = _activation(h, alpha, beta)
            saved += [h, a]
            out = _linear(a, w, b)
        ctx.spec = spec
        ctx.save_for_backward(x, *params, *saved)
        return out

    @staticmethod
    def backward(ctx, gout):
        spec = ctx.spec
        tensors = ctx.saved_tensors
        need = _needed(ctx)
        out = MLPGradFunction.apply(spec, tuple(need), tensors[0], gout.contiguous(), *tensors[1:])
        return (None, *(o if n else None for o, n in zip(out, need)))


class MLPGradFunction(torch.autograd.Function):
    """The backward of :class:`MLPFunction` (the gradients of ``x`` and of the parameters the pass needs, for
    the incoming gradient ``gout``), differentiable once more with respect to ``x``, ``gout`` and the
    parameters through the gradient of ``x``."""

    @staticmethod
    def forward(ctx, spec: _Spec, need: Tuple[bool, ...], x: Tensor, gout: Tensor, *rest: Tensor):
        ctx.set_materialize_grads(False)
        n_params = spec.n_layers * spec.per_layer
        params, saved = rest[:n_params], rest[n_params:]
        need_x, need_p = need[0], need[1:]
        g, n, f = saved[0].shape if saved else (params[0].shape[0], x.shape[0], x.shape[1])
        _, _, nb = _launch(n, f)
        layers = [i for i in range(spec.n_layers) if any(need_p[i * spec.per_layer:(i + 1) * spec.per_layer])]
        part = x.new_zeros((spec.n_layers, g, nb, 2, f)) if layers else None
        grads: List[Optional[Tensor]] = [None] * n_params

        def layer_backward(i: int, h: Tensor, a: Tensor, gy: Tensor) -> Tensor:
            alpha, beta, w, b = spec.layer(params, i)
            base = i * spec.per_layer
            gh = _activation_backward(h, alpha, beta, torch.bmm(gy, w.transpose(1, 2)),
                                      part[i] if i in layers else None)
            if need_p[base + 2]:
                grads[base + 2] = torch.bmm(a.transpose(1, 2), gy)
            if spec.bias and need_p[base + 3]:
                grads[base + 3] = gy.sum(1)
            return gh

        gh = gout
        if spec.output:
            gh = layer_backward(2 * spec.n_residual, saved[-2], saved[-1], gout)
        for r in reversed(range(spec.n_residual)):
            h, a, z, a2 = saved[4 * r:4 * r + 4]
            gz = layer_backward(2 * r + 1, z, a2, gh)
            gh = gh + layer_backward(2 * r, h, a, gz)
        if layers:
            sums = part.sum(2)
            for i in layers:
                grads[i * spec.per_layer], grads[i * spec.per_layer + 1] = sums[i, :, 0], sums[i, :, 1]
        gx = gh.sum(0) if need_x else x.new_zeros(0)
        ctx.spec = spec
        ctx.save_for_backward(x, gout, *params, *saved)
        return (gx, *(gr if gr is not None else x.new_zeros(0) for gr in grads))

    @staticmethod
    @once_differentiable
    def backward(ctx, t_gx, *t_params):
        spec = ctx.spec
        tensors = ctx.saved_tensors
        n_params = spec.n_layers * spec.per_layer
        x, gout = tensors[0], tensors[1]
        params, saved = tensors[2:2 + n_params], tensors[2 + n_params:]
        if any(t is not None for t in t_params):
            raise NotImplementedError("SpookyNet's fused MLPs have no second derivatives through their weight "
                                      "gradients")
        nothing = (None, None) + (None,) * (2 + n_params + len(saved))
        if t_gx is None:
            return nothing
        need = _needed(ctx)
        need_x, need_gout, need_p = need[0], need[1], need[2:2 + n_params]
        g, n, f = gout.shape[0], x.shape[0], x.shape[1]

        # the tangent of the gradient's input pushed forward through the MLP
        th = t_gx.unsqueeze(0).expand(g, n, f).contiguous()
        tangents = []
        for r in range(spec.n_residual):
            h, a, z, a2 = saved[4 * r:4 * r + 4]
            alpha, beta, w, _ = spec.layer(params, 2 * r)
            ta = _activation(h, alpha, beta, th)
            tz = torch.bmm(ta, w)
            alpha2, beta2, w2, _ = spec.layer(params, 2 * r + 1)
            ta2 = _activation(z, alpha2, beta2, tz)
            tangents += [th, ta, tz, ta2]
            th = th + torch.bmm(ta2, w2)
        tout = th
        if spec.output:
            alpha, beta, w, _ = spec.layer(params, 2 * spec.n_residual)
            ta = _activation(saved[-2], alpha, beta, th)
            tangents += [th, ta]
            tout = torch.bmm(ta, w)

        # the reverse sweep: mu the adjoints of the tangents (gout for the output's), lam those of the values
        _, _, nb = _launch(n, f)
        part = x.new_zeros((spec.n_layers, g, nb, 2, f))
        grads: List[Optional[Tensor]] = [None] * n_params

        def layer_double(i: int, h: Tensor, th_in: Tensor, a: Tensor, ta: Tensor, mu: Tensor,
                         lam: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
            alpha, beta, w, b = spec.layer(params, i)
            base = i * spec.per_layer
            wt = w.transpose(1, 2)
            if need_p[base + 2]:
                dw = torch.bmm(ta.transpose(1, 2), mu)
                grads[base + 2] = dw if lam is None else torch.baddbmm(dw, a.transpose(1, 2), lam)
            if spec.bias and need_p[base + 3]:
                grads[base + 3] = lam.sum(1) if lam is not None else torch.zeros_like(b)
            return _activation_double(h, th_in, alpha, beta, torch.bmm(mu, wt),
                                      None if lam is None else torch.bmm(lam, wt), part[i])

        mu, lam = gout, None
        if spec.output:
            mu, lam = layer_double(2 * spec.n_residual, saved[-2], tangents[-2], saved[-1], tangents[-1], gout, None)
        for r in reversed(range(spec.n_residual)):
            h, a, z, a2 = saved[4 * r:4 * r + 4]
            th_in, ta, tz, ta2 = tangents[4 * r:4 * r + 4]
            mu_z, lam_z = layer_double(2 * r + 1, z, tz, a2, ta2, mu, lam)
            mu_a, lam_a = layer_double(2 * r, h, th_in, a, ta, mu_z, lam_z)
            mu = mu + mu_a
            lam = lam_a if lam is None else lam + lam_a
        sums = part.sum(2)
        for i in range(spec.n_layers):
            base = i * spec.per_layer
            if need_p[base]:
                grads[base] = sums[i, :, 0]
            if need_p[base + 1]:
                grads[base + 1] = sums[i, :, 1]
        dx = lam.sum(0) if need_x and lam is not None else None
        return (None, None, dx, tout if need_gout else None, *grads) + (None,) * len(saved)


def residual_mlps(x: Tensor, n_residual: int, output: bool, params: List[Tensor]) -> Tensor:
    """``G`` residual MLPs of one structure on ``x``, fused.

    Parameters
    ----------
    x : Tensor
        Input, shape ``(N, F)``.
    n_residual : int
        Residual blocks per MLP.
    output : bool
        Whether an output layer follows the blocks.
    params : list of Tensor
        Per layer (``A_0, B_0, ..., O``): ``alpha``, ``beta`` ``(G, F)``, the transposed weights
        ``(G, F, F')`` (``y = a w``) and, when the layers have them, the biases ``(G, F')``. All layers
        have biases or none has.

    Returns
    -------
    Tensor
        Shape ``(G, N, F')``.
    """
    bias = len(params) == 4 * (2 * n_residual + (1 if output else 0))
    spec = _Spec(n_residual, output, bias)
    return MLPFunction.apply(spec, x.contiguous(), *(p.contiguous() if p.dim() < 3 else p for p in params))
