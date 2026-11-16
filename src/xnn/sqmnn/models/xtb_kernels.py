"""Triton kernels of the overlap, dipole and second-moment matrices of the GFN methods.

One kernel per pair of angular momenta and operator count, generated from the
recursions of :mod:`xnn.sqmnn.models.gaussian`: a lane takes one shell pair, loops
over its primitive pairs, sums every Cartesian component in a register, forms the
spherical functions and writes both blocks of the pair into the dense matrices (the
element ``[mu, nu]`` with the origin of the operator on the centre of ``nu``). A
second kernel per class reads the gradient of both blocks and gives the derivative
in the bra centre, with the dependence of the moved origin on the centres (the ket
derivative is its negative). On a CPU the kernels run as C++ loops
(:func:`xnn.common.cpu_kernels.launch`), on a GPU as compiled Triton programs.

The weights given to the kernels are the contraction coefficients times the norms
of their primitives and ``pi^(3/4)``, so that the kernels hold no inexact literal.
"""
# no postponed annotations: Triton's interpreter (TRITON_INTERPRET=1) recognizes the
# tl.constexpr parameters of a kernel only from evaluated annotations

import linecache
import math
from typing import Dict, List, Sequence, Tuple

import torch
from torch import Tensor

from .gaussian import _D_TRANSFORM, _OPERATORS, CARTESIAN, N_SPHERICAL

#: Shell pairs per program on a GPU (one per thread) and warps per program.
BLOCK = 64
WARPS = 2
#: The second moments ``xx, yy, zz, xy, xz, yz`` as pairs of axes.
_SECOND = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))

_KERNELS: Dict[tuple, object] = {}
_CONSTANTS: Dict[tuple, Tensor] = {}
_HALF_SQRT3 = math.sqrt(3.0) / 2.0

#: A class of shell pairs: ``(la, lb, ka, kb, bra shells, ket shells)``.
PairClass = Tuple[int, int, int, int, Tensor, Tensor]


def available(device: torch.device) -> bool:
    """Whether the kernels run on ``device``: Triton on a GPU, a C++ compiler on a CPU."""
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    if device.type == "cuda":
        return True
    from xnn.common import cpu_kernels
    return cpu_kernels.available()


def _spherical_rows(l: int) -> List[List[Tuple[int, str]]]:
    """Each spherical function of a shell as a sum of its normalized Cartesian components:
    ``(component, coefficient)`` with the coefficient as kernel source (``s3`` is ``sqrt(3)``)."""
    if l < 2:
        return [[(c, "")] for c in range(len(CARTESIAN[l]))]
    rows = []
    for row in _D_TRANSFORM:
        terms = []
        for c, t in enumerate(row):
            if t == 0.0:
                continue
            # the cross components carry sqrt(3) relative to the squares
            if c >= 3:
                terms.append((c, "s3 * "))
            elif abs(abs(t) - _HALF_SQRT3) < 1e-12:
                terms.append((c, ("0.5 * s3 * " if t > 0 else "-0.5 * s3 * ")))
            else:
                terms.append((c, f"{t!r} * "))
        rows.append(terms)
    return rows


def _moment_lines(axis: str, mmax: int, jmax: int) -> List[str]:
    """The moments ``G[m][j]`` of one axis as assignments ``g<axis>_<m>_<j> = ...``."""
    def g(m: int, j: int) -> str:
        return f"g{axis}_{m}_{j}"
    lines = [f"{g(0, 0)} = s"]
    for m in range(1, mmax + 1):
        expr = f"pa{axis} * {g(m - 1, 0)}"
        if m > 1:
            expr += f" + {m - 1}.0 * half * {g(m - 2, 0)}"
        lines.append(f"{g(m, 0)} = {expr}")
    for j in range(1, jmax + 1):
        for m in range(mmax + 1):
            expr = f"pb{axis} * {g(m, j - 1)}"
            if j > 1:
                expr += f" + {j - 1}.0 * half * {g(m, j - 2)}"
            if m > 0:
                expr += f" + {m}.0 * half * {g(m - 1, j - 1)}"
            lines.append(f"{g(m, j)} = {expr}")
    return lines


def _head(name: str, extra: str, n_ops: int) -> List[str]:
    """The shells, centres and matrix positions of a lane's pair; ``e = A - B``."""
    lines = [
        f"def {name}(pos_ptr, sat_ptr, sfirst_ptr, sstruct_ptr, alpha_ptr, weight_ptr, si_ptr, sj_ptr, c_ptr,",
        f"        {extra}, P, KA, KB, KMAX, NB, BLOCK: tl.constexpr):",
        "    r = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)",
        "    m = r < P",
        "    si = tl.load(si_ptr + r, mask=m, other=0)",
        "    sj = tl.load(sj_ptr + r, mask=m, other=0)",
        "    mo = m & (si != sj)",
        "    ia = tl.load(sat_ptr + si, mask=m, other=0)",
        "    ib = tl.load(sat_ptr + sj, mask=m, other=0)",
        "    fa = tl.load(sfirst_ptr + si, mask=m, other=0)",
        "    fb = tl.load(sfirst_ptr + sj, mask=m, other=0)",
        f"    base = tl.load(sstruct_ptr + si, mask=m, other=0) * NB * NB * {n_ops}",
    ]
    for k, axis in enumerate("xyz"):
        lines.append(f"    e{axis} = tl.load(pos_ptr + ia * 3 + {k}, mask=m, other=0.0)"
                     f" - tl.load(pos_ptr + ib * 3 + {k}, mask=m, other=0.0)")
    lines += ["    r2 = ex * ex + ey * ey + ez * ez", "    neglect = tl.load(c_ptr)", "    s3 = tl.load(c_ptr + 1)"]
    return lines


def _primitive(mmax: int, jmax: int) -> List[str]:
    """The loop head over the primitive pairs: exponents, weight and moments of one pair."""
    lines = [
        "    k = tl.program_id(0) * 0",
        "    while k < KA * KB:",
        "        ka = k // KB",
        "        kb = k - ka * KB",
        "        a = tl.load(alpha_ptr + si * KMAX + ka, mask=m, other=1.0)",
        "        b = tl.load(alpha_ptr + sj * KMAX + kb, mask=m, other=1.0)",
        "        w = tl.load(weight_ptr + si * KMAX + ka, mask=m, other=0.0)",
        "        w = w * tl.load(weight_ptr + sj * KMAX + kb, mask=m, other=0.0)",
        "        ip = 1.0 / (a + b)",
        "        x = a * b * ip * r2",
        "        pref = tl.where(x <= neglect, tl.exp(-x) * w, 0.0)",
        "        half = 0.5 * ip",
        "        s = tl.sqrt(ip)",
    ]
    for axis in "xyz":
        # the product centre from the bra and from the ket centre
        lines.append(f"        pa{axis} = -b * ip * e{axis}")
        lines.append(f"        pb{axis} = a * ip * e{axis}")
        lines += ["        " + line for line in _moment_lines(axis, mmax, jmax)]
    return lines


def _components(la: int, lb: int, n_ops: int) -> List[Tuple[int, Tuple[int, int, int], Tuple[int, int, int]]]:
    """``(flat index, bra moments, ket moments)`` of every Cartesian component of every operator."""
    out = []
    for c, (o, ea, eb) in enumerate((o, ea, eb) for o in range(n_ops) for ea in CARTESIAN[la] for eb in CARTESIAN[lb]):
        op = _OPERATORS[o]
        out.append((c, (op[0] + ea[0], op[1] + ea[1], op[2] + ea[2]), eb))
    return out


def _accumulate(comps) -> List[str]:
    """The sums of the Cartesian components over the primitive pairs (inside the loop)."""
    used = sorted({(ma[0], mb[0]) for _, ma, mb in comps})
    lines = [f"        fx_{i}_{j} = pref * gx_{i}_{j}" for i, j in used]
    for c, ma, mb in comps:
        lines.append(f"        acc{c} = acc{c} + fx_{ma[0]}_{mb[0]} * gy_{ma[1]}_{mb[1]} * gz_{ma[2]}_{mb[2]}")
    return lines


def _to_spherical(la: int, lb: int, n_ops: int, src: str = "acc", dst: str = "a") -> List[str]:
    """``<dst><o>_<sa>_<sb>``: the spherical functions from the sums ``<src>`` (bra, then ket)."""
    nca, ncb = len(CARTESIAN[la]), len(CARTESIAN[lb])
    rows_a, rows_b = _spherical_rows(la), _spherical_rows(lb)
    lines = []
    for o in range(n_ops):
        base = o * nca * ncb
        for sa, row in enumerate(rows_a):
            for cb in range(ncb):
                expr = " + ".join(f"{coef}{src}{base + ca * ncb + cb}" for ca, coef in row)
                lines.append(f"    h{o}_{sa}_{cb} = {expr}")
        for sa in range(len(rows_a)):
            for sb, row in enumerate(rows_b):
                expr = " + ".join(f"{coef}h{o}_{sa}_{cb}" for cb, coef in row)
                lines.append(f"    {dst}{o}_{sa}_{sb} = {expr}")
    return lines


def _at(o: int, sa: int, sb: int, transposed: bool) -> str:
    """Address offset of an element of a pair's block in the stack of dense matrices."""
    if transposed:
        return f"base + ({o} * NB + fb + {sb}) * NB + fa + {sa}"
    return f"base + ({o} * NB + fa + {sa}) * NB + fb + {sb}"


def _values_source(name: str, la: int, lb: int, n_ops: int) -> str:
    mmax = la + (2 if n_ops > 1 else 0)
    na, nb = N_SPHERICAL[la], N_SPHERICAL[lb]
    comps = _components(la, lb, n_ops)
    lines = _head(name, "out_ptr", n_ops)
    lines += [f"    acc{c} = tl.zeros([BLOCK], dtype=out_ptr.dtype.element_ty)" for c, _, _ in comps]
    lines += _primitive(mmax, lb) + _accumulate(comps) + ["        k = k + 1"]
    lines += _to_spherical(la, lb, n_ops)
    e = "xyz"
    for sa in range(na):
        for sb in range(nb):
            t = f"_{sa}_{sb}"
            for o in range(n_ops):
                # the origin on the ket centre: <r - B> = <r - A> + (A - B) <1>, and the second moments likewise
                if o == 0:
                    expr = f"a0{t}"
                elif o < 4:
                    expr = f"a{o}{t} + e{e[o - 1]} * a0{t}"
                else:
                    i, j = _SECOND[o - 4]
                    expr = (f"a{o}{t} + e{e[i]} * a{1 + j}{t} + e{e[j]} * a{1 + i}{t}"
                            f" + e{e[i]} * e{e[j]} * a0{t}")
                lines.append(f"    tl.store(out_ptr + {_at(o, sa, sb, False)}, {expr}, mask=m)")
                lines.append(f"    tl.store(out_ptr + {_at(o, sa, sb, True)}, a{o}{t}, mask=mo)")
    return "\n".join(lines) + "\n"


def _gradient_source(name: str, la: int, lb: int, n_ops: int) -> str:
    mmax = la + (2 if n_ops > 1 else 0)
    nca, ncb, na, nb = len(CARTESIAN[la]), len(CARTESIAN[lb]), N_SPHERICAL[la], N_SPHERICAL[lb]
    comps = _components(la, lb, n_ops)
    rows_a, rows_b = _spherical_rows(la), _spherical_rows(lb)
    e = "xyz"
    lines = _head(name, "w_ptr, out_ptr", n_ops)
    for sa in range(na):
        for sb in range(nb):
            t = f"_{sa}_{sb}"
            for o in range(n_ops):
                lines.append(f"    ub{o}{t} = tl.load(w_ptr + {_at(o, sa, sb, False)}, mask=m, other=0.0)")
                lines.append(f"    ua{o}{t} = tl.load(w_ptr + {_at(o, sa, sb, True)}, mask=mo, other=0.0)")
            # weights of the integrals with the origin on the bra centre: the adjoint of the moved origin
            for o in range(n_ops):
                expr = f"ua{o}{t} + ub{o}{t}"
                if n_ops > 1 and o == 0:
                    expr += "".join(f" + e{e[i]} * ub{1 + i}{t}" for i in range(3))
                    expr += "".join(f" + e{e[i]} * e{e[j]} * ub{4 + q}{t}" for q, (i, j) in enumerate(_SECOND))
                elif n_ops > 1 and o < 4:
                    for q, (i, j) in enumerate(_SECOND):
                        if i == o - 1:
                            expr += f" + e{e[j]} * ub{4 + q}{t}"
                        if j == o - 1:
                            expr += f" + e{e[i]} * ub{4 + q}{t}"
                lines.append(f"    u{o}{t} = {expr}")
    # onto the Cartesian components (the adjoint of the spherical transform)
    for o in range(n_ops):
        for sa in range(na):
            for cb in range(ncb):
                terms = [f"{coef}u{o}_{sa}_{sb}" for sb, row in enumerate(rows_b) for c, coef in row if c == cb]
                lines.append(f"    v{o}_{sa}_{cb} = {' + '.join(terms)}")
        for ca in range(nca):
            for cb in range(ncb):
                terms = [f"{coef}v{o}_{sa}_{cb}" for sa, row in enumerate(rows_a) for c, coef in row if c == ca]
                lines.append(f"    w{o * nca * ncb + ca * ncb + cb} = {' + '.join(terms)}")
    lines += [f"    out{axis} = tl.zeros([BLOCK], dtype=out_ptr.dtype.element_ty)" for axis in e]
    # the overlap and dipole integrals themselves enter through the moved origin
    low = [c for c in comps if c[0] < min(n_ops, 4) * nca * ncb] if n_ops > 1 else []
    lines += [f"    acc{c} = tl.zeros([BLOCK], dtype=out_ptr.dtype.element_ty)" for c, _, _ in low]
    lines += _primitive(mmax + 1, lb)
    lines += _slopes(comps)
    for k, axis in enumerate(e):
        terms = []
        for c, ma, mb in comps:
            f = [f"g{ax}_{ma[q]}_{mb[q]}" for q, ax in enumerate(e)]
            f[k] = f"d{axis}_{ma[k]}_{mb[k]}"
            terms.append(f"w{c} * {f[0]} * {f[1]} * {f[2]}")
        # sums of a few terms each: one long expression is deep for the compilers
        parts = [" + ".join(terms[q:q + 6]) for q in range(0, len(terms), 6)]
        lines.append(f"        sum{axis} = {parts[0]}")
        lines += [f"        sum{axis} = sum{axis} + {part}" for part in parts[1:]]
        lines.append(f"        out{axis} = out{axis} + pref * sum{axis}")
    lines += _accumulate(low) + ["        k = k + 1"]
    if n_ops > 1:
        lines += _to_spherical(la, lb, 4)
        for sa in range(na):
            for sb in range(nb):
                t = f"_{sa}_{sb}"
                for k, axis in enumerate(e):
                    expr = f"ub{1 + k}{t} * a0{t}"
                    for q, (i, j) in enumerate(_SECOND):
                        if i == k and j == k:
                            expr += f" + 2.0 * ub{4 + q}{t} * (a{1 + k}{t} + e{axis} * a0{t})"
                        elif i == k:
                            expr += f" + ub{4 + q}{t} * (a{1 + j}{t} + e{e[j]} * a0{t})"
                        elif j == k:
                            expr += f" + ub{4 + q}{t} * (a{1 + i}{t} + e{e[i]} * a0{t})"
                    lines.append(f"    out{axis} = out{axis} + {expr}")
    lines += [f"    tl.store(out_ptr + r * 3 + {k}, out{axis}, mask=m)" for k, axis in enumerate(e)]
    return "\n".join(lines) + "\n"


def _slopes(comps, indent: str = "        ") -> List[str]:
    """``d<axis>_<m>_<j>``: dG[m, j] / dA = 2 a G[m + 1, j] - m G[m - 1, j] of the moments in use."""
    lines = []
    for k, axis in enumerate("xyz"):
        for i, j in sorted({(ma[k], mb[k]) for _, ma, mb in comps}):
            expr = f"2.0 * a * g{axis}_{i + 1}_{j}"
            if i:
                expr += f" - {i}.0 * g{axis}_{i - 1}_{j}"
            lines.append(f"{indent}d{axis}_{i}_{j} = {expr}")
    return lines


def _tangent_source(name: str, la: int, lb: int, n_ops: int) -> str:
    """The matrices of :func:`_values_source` differentiated along a displacement of the atoms."""
    mmax = la + (2 if n_ops > 1 else 0)
    nca, ncb, na, nb = len(CARTESIAN[la]), len(CARTESIAN[lb]), N_SPHERICAL[la], N_SPHERICAL[lb]
    comps = _components(la, lb, n_ops)
    e = "xyz"
    lines = _head(name, "v_ptr, out_ptr", n_ops)
    for k, axis in enumerate(e):
        lines.append(f"    t{axis} = tl.load(v_ptr + ia * 3 + {k}, mask=m, other=0.0)"
                     f" - tl.load(v_ptr + ib * 3 + {k}, mask=m, other=0.0)")
    low = [c for c in comps if c[0] < 4 * nca * ncb] if n_ops > 1 else []
    lines += [f"    acc{c} = tl.zeros([BLOCK], dtype=out_ptr.dtype.element_ty)" for c, _, _ in low]
    lines += [f"    tac{c} = tl.zeros([BLOCK], dtype=out_ptr.dtype.element_ty)" for c, _, _ in comps]
    lines += _primitive(mmax + 1, lb) + _slopes(comps) + _accumulate(low)
    for c, ma, mb in comps:
        g = [f"g{ax}_{ma[q]}_{mb[q]}" for q, ax in enumerate(e)]
        d = [f"d{ax}_{ma[q]}_{mb[q]}" for q, ax in enumerate(e)]
        lines.append(f"        tac{c} = tac{c} + pref * (tx * {d[0]} * {g[1]} * {g[2]} + ty * {g[0]} * {d[1]} * {g[2]}"
                     f" + tz * {g[0]} * {g[1]} * {d[2]})")
    lines.append("        k = k + 1")
    if n_ops > 1:
        lines += _to_spherical(la, lb, 4)
    lines += _to_spherical(la, lb, n_ops, "tac", "b")
    for sa in range(na):
        for sb in range(nb):
            t = f"_{sa}_{sb}"
            for o in range(n_ops):
                if o == 0:
                    expr = f"b0{t}"
                elif o < 4:
                    i = o - 1
                    expr = f"b{o}{t} + e{e[i]} * b0{t} + t{e[i]} * a0{t}"
                else:
                    i, j = _SECOND[o - 4]
                    expr = (f"b{o}{t} + e{e[i]} * b{1 + j}{t} + t{e[i]} * a{1 + j}{t} + e{e[j]} * b{1 + i}{t}"
                            f" + t{e[j]} * a{1 + i}{t} + (t{e[i]} * e{e[j]} + e{e[i]} * t{e[j]}) * a0{t}"
                            f" + e{e[i]} * e{e[j]} * b0{t}")
                lines.append(f"    tl.store(out_ptr + {_at(o, sa, sb, False)}, {expr}, mask=m)")
                lines.append(f"    tl.store(out_ptr + {_at(o, sa, sb, True)}, b{o}{t}, mask=mo)")
    return "\n".join(lines) + "\n"


_SOURCES = {"values": _values_source, "gradient": _gradient_source, "tangent": _tangent_source}


def _kernel(kind: str, la: int, lb: int, n_ops: int):
    """The kernel of a class (generated once per process)."""
    key = (kind, la, lb, n_ops)
    fn = _KERNELS.get(key)
    if fn is None:
        import triton
        import triton.language as tl
        name = f"_pair_{kind}_{la}{lb}_{n_ops}"
        src = _SOURCES[kind](name, la, lb, n_ops)
        # Triton reads a kernel's source through inspect, which finds it in linecache
        filename = f"<xnn-xtb-{name}>"
        linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
        namespace = {"__name__": __name__, "tl": tl, "triton": triton}
        exec(compile(src, filename, "exec"), namespace)
        fn = triton.jit(namespace[name])
        _KERNELS[key] = fn
    return fn


def _constants(neglect: float, like: Tensor) -> Tensor:
    key = (neglect, like.dtype, like.device)
    c = _CONSTANTS.get(key)
    if c is None:
        c = torch.tensor([min(neglect, 1e300), math.sqrt(3.0)], dtype=torch.float64).to(like)
        _CONSTANTS[key] = c
    return c


def _launch(kind: str, la: int, lb: int, n_ops: int, tensors: tuple, n: int, ints: tuple) -> None:
    kernel = _kernel(kind, la, lb, n_ops)
    if tensors[0].is_cuda:
        import triton
        kernel[(triton.cdiv(n, BLOCK),)](*tensors, n, *ints, BLOCK=BLOCK, num_warps=WARPS)
    else:
        from xnn.common import cpu_kernels
        cpu_kernels.launch(kernel, (n,), *tensors, n, *ints, BLOCK=1)


def integral_matrices(pos: Tensor, tables: Sequence[Tensor], classes: Sequence[PairClass], n_struct: int, n_ops: int,
                      nb: int, neglect: float) -> Tensor:
    """The dense matrices ``(n_struct, n_ops, nb, nb)`` of the shell pairs of ``classes``.

    ``tables`` are the atom, first function (within its structure) and structure of every shell,
    and the exponents and weights of its primitives ``(n_shell, kmax)``."""
    out = pos.new_zeros(n_struct, n_ops, nb, nb)
    head = (pos.detach().contiguous(),) + tuple(tables)
    c = _constants(neglect, pos)
    for la, lb, ka, kb, si, sj in classes:
        _launch("values", la, lb, n_ops, head + (si, sj, c, out), si.shape[0], (ka, kb, tables[3].shape[1], nb))
    return out


def integral_tangent(pos: Tensor, tables: Sequence[Tensor], classes: Sequence[PairClass], n_struct: int, n_ops: int,
                     nb: int, neglect: float, direction: Tensor) -> Tensor:
    """The matrices of :func:`integral_matrices` differentiated along the displacement ``direction``
    (``(N, 3)``) of the atoms; also the gradient, in its weight, of :func:`integral_gradient`
    contracted with ``direction``."""
    out = pos.new_zeros(n_struct, n_ops, nb, nb)
    head = (pos.detach().contiguous(),) + tuple(tables)
    c, direction = _constants(neglect, pos), direction.detach().contiguous()
    for la, lb, ka, kb, si, sj in classes:
        _launch("tangent", la, lb, n_ops, head + (si, sj, c, direction, out), si.shape[0],
                (ka, kb, tables[3].shape[1], nb))
    return out


def integral_gradient(pos: Tensor, tables: Sequence[Tensor], classes: Sequence[PairClass], n_ops: int, nb: int,
                      neglect: float, weight: Tensor) -> Tensor:
    """Derivative in the positions, ``(N, 3)``, of the matrices of :func:`integral_matrices`
    contracted with ``weight``."""
    out = torch.zeros_like(pos)
    head = (pos.detach().contiguous(),) + tuple(tables)
    c, weight = _constants(neglect, pos), weight.detach().contiguous()
    for la, lb, ka, kb, si, sj in classes:
        g = pos.new_empty(si.shape[0], 3)
        _launch("gradient", la, lb, n_ops, head + (si, sj, c, weight, g), si.shape[0],
                (ka, kb, tables[3].shape[1], nb))
        out.index_add_(0, tables[0].index_select(0, si), g)
        out.index_add_(0, tables[0].index_select(0, sj), -g)
    return out
