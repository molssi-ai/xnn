"""Triton kernels and autograd function of the Allegro fast path (see :mod:`xnn.gnn.fast.allegro`).

Imported only when the fast path runs, so that ``triton`` is needed only there.

``out[r, k] = sum_t c_t x[r, i_t] y[r, j_t]`` over the nonzero entries
``t = (k_t, i_t, j_t)`` of a sparse coupling. One kernel per pattern and per
role of its three axes (output, first and second operand), generated from the
pattern: a program takes a block of rows, one row per thread, loads the
operand columns once and writes every output column from its unrolled sum.
The coefficients are a tensor argument, so the kernels run the model's own
constants in either precision. The gradients of the contraction are the same
contraction with the roles of the axes exchanged.
"""
# no postponed annotations: Triton's interpreter (TRITON_INTERPRET=1) recognizes the
# tl.constexpr parameters of a kernel only from evaluated annotations

import hashlib
import linecache
from typing import Dict, Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor

#: Rows per program (one per thread) and warps per program.
BLOCK = 128
WARPS = 4

#: The nonzero entries ``(k, i, j)`` of a coupling and its shape ``(dim_out, dim1, dim2)``.
Pattern = Tuple[Tuple[Tuple[int, int, int], ...], Tuple[int, int, int]]

_KERNELS: Dict[tuple, object] = {}


def _source(name: str, pattern: Pattern, roles: Tuple[int, int, int]) -> str:
    """The kernel source for the axes ``roles`` = (output, first operand, second operand)."""
    entries, dims = pattern
    a0, a1, a2 = roles
    d_out, d1, d2 = dims[a0], dims[a1], dims[a2]
    terms = [[] for _ in range(d_out)]
    for t, idx in enumerate(entries):
        terms[idx[a0]].append((t, idx[a1], idx[a2]))
    used1 = sorted({i for row in terms for _, i, _ in row})
    used2 = sorted({j for row in terms for _, _, j in row})
    lines = [
        f"def {name}(x_ptr, y_ptr, c_ptr, out_ptr, R, BLOCK: tl.constexpr):",
        "    r = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)",
        "    m = r < R",
        f"    xr = x_ptr + r * {d1}",
        f"    yr = y_ptr + r * {d2}",
        f"    outr = out_ptr + r * {d_out}",
    ]
    lines += [f"    x{i} = tl.load(xr + {i}, mask=m, other=0.0)" for i in used1]
    lines += [f"    y{j} = tl.load(yr + {j}, mask=m, other=0.0)" for j in used2]
    lines += [f"    c{t} = tl.load(c_ptr + {t})" for t in range(len(entries))]
    for k, row in enumerate(terms):
        if row:
            expr = " + ".join(f"c{t} * (x{i} * y{j})" for t, i, j in row)
        else:
            expr = "tl.zeros([BLOCK], dtype=out_ptr.dtype.element_ty)"
        lines.append(f"    tl.store(outr + {k}, {expr}, mask=m)")
    return "\n".join(lines) + "\n"


def _kernel(pattern: Pattern, roles: Tuple[int, int, int]):
    """The compiled kernel of ``pattern`` for ``roles`` (generated once per process)."""
    key = (pattern, roles)
    fn = _KERNELS.get(key)
    if fn is None:
        digest = hashlib.sha1(repr(key).encode()).hexdigest()[:16]
        name = f"_bilinear_{digest}"
        src = _source(name, pattern, roles)
        # Triton reads a kernel's source through inspect, which finds it in linecache
        filename = f"<xnn-allegro-{digest}>"
        linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
        namespace = {"__name__": __name__, "tl": tl, "triton": triton}
        exec(compile(src, filename, "exec"), namespace)
        fn = triton.jit(namespace[name])
        _KERNELS[key] = fn
    return fn


def _run(x: Tensor, y: Tensor, c: Tensor, pattern: Pattern, roles: Tuple[int, int, int]) -> Tensor:
    rows = x.shape[0]
    out = torch.empty((rows, pattern[1][roles[0]]), dtype=x.dtype, device=x.device)
    if rows:
        _kernel(pattern, roles)[(triton.cdiv(rows, BLOCK),)](x, y, c, out, rows, BLOCK=BLOCK,
                                                              num_warps=WARPS)
    return out


class BilinearFunction(torch.autograd.Function):
    """``out[r, a0] = sum c x[r, a1] y[r, a2]`` over the entries of ``pattern``, the
    axes in the order ``roles``; differentiable at any order in ``x`` and ``y``."""

    @staticmethod
    def forward(ctx, x: Tensor, y: Tensor, c: Tensor, pattern: Pattern,
                roles: Tuple[int, int, int]) -> Tensor:
        ctx.pattern, ctx.roles = pattern, roles
        ctx.save_for_backward(x, y, c)
        return _run(x, y, c, pattern, roles)

    @staticmethod
    def backward(ctx, g: Tensor):
        x, y, c = ctx.saved_tensors
        a0, a1, a2 = ctx.roles
        g = g.contiguous()
        gx = gy = None
        if ctx.needs_input_grad[0]:
            gx = BilinearFunction.apply(g, y, c, ctx.pattern, (a1, a0, a2))
        if ctx.needs_input_grad[1]:
            gy = BilinearFunction.apply(g, x, c, ctx.pattern, (a2, a0, a1))
        return gx, gy, None, None, None


def contract(x: Tensor, y: Tensor, c: Tensor, pattern: Pattern) -> Tensor:
    """The sparse contraction of the Allegro tensor product (see :mod:`xnn.gnn.fast.allegro`).

    Parameters
    ----------
    x, y : Tensor
        The two operands, one row per edge and channel, shapes ``(R, dim1)`` and ``(R, dim2)``.
    c : Tensor
        The coefficients of the nonzero entries, in the order of ``pattern``.
    pattern : tuple
        The nonzero entries ``(k, i, j)`` of the coupling and its shape ``(dim_out, dim1, dim2)``.

    Returns
    -------
    Tensor
        ``out[r, k] = sum_t c_t x[r, i_t] y[r, j_t]``, shape ``(R, dim_out)``.
    """
    return BilinearFunction.apply(x.contiguous(), y.contiguous(), c.contiguous(), pattern, (0, 1, 2))
