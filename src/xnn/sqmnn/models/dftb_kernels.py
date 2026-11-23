"""Triton kernels of the DFTB overlap and Hamiltonian matrices.

One kernel per pair of shell angular momenta ``(la, lb)``: a lane takes one atom pair, interpolates the
Hamiltonian and overlap columns of the class from the stacked tables of :meth:`~xnn.sqmnn.models.dftb.DFTB._stacked`
(the 8-point Lagrange polynomial inside a table, its tail polynomial past it), forms the Slater-Koster rotation of
the bond direction (:func:`~xnn.sqmnn.models.slater_koster.rotation_polynomials`) and adds the block, and its
transpose for a pair of different atoms, into the dense matrices. A second kernel per class reads the gradient of
both matrices and gives the derivative in the pair's vector; a third differentiates that along given vectors (the
tangent matrices and the Hessian-vector product), so that Hessians and force losses run on the kernels too. The
rotation coefficients come from a float64 table, so the kernels hold no inexact literal.
"""
# no postponed annotations: Triton's interpreter (TRITON_INTERPRET=1) recognizes the
# tl.constexpr parameters of a kernel only from evaluated annotations

import linecache
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from .slater_koster import monomial_powers, rotation_polynomials

#: Pairs per program on a GPU (one per thread) and warps per program.
BLOCK = 64
WARPS = 2
LETTERS = "spdf"
#: The independent second derivatives in the direction, as pairs of axes.
SECOND = (("x", "x"), ("y", "y"), ("z", "z"), ("x", "y"), ("x", "z"), ("y", "z"))

_KERNELS: Dict[tuple, object] = {}
_ROTATIONS: Dict[tuple, Tensor] = {}


def available(device: torch.device) -> bool:
    """Whether the kernels run on ``device`` (Triton on a GPU)."""
    if device.type != "cuda":
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def _terms(la: int, lb: int) -> List[List[List[Tuple[int, int]]]]:
    """``terms[a][b][k]``: ``(monomial, coefficient index)`` of the nonzero rotation coefficients of element
    ``(a, b)`` of the block and bond integral ``k``, the coefficient index into :func:`_rotation_table`."""
    coef = rotation_polynomials(la, lb)
    n_mono, n_k, na, nb = coef.shape
    out, q = [], 0
    for a in range(na):
        row = []
        for b in range(nb):
            cell = []
            for k in range(n_k):
                items = []
                for i in range(n_mono):
                    if coef[i, k, a, b] != 0.0:
                        items.append((i, q))
                        q += 1
                cell.append(items)
            row.append(cell)
        out.append(row)
    return out


def _rotation_table(la: int, lb: int, like: Tensor) -> Tensor:
    """The nonzero rotation coefficients of the class in the order of :func:`_terms`."""
    key = (la, lb, like.dtype, like.device)
    t = _ROTATIONS.get(key)
    if t is None:
        coef = rotation_polynomials(la, lb)
        n_mono, n_k, na, nb = coef.shape
        values = [coef[i, k, a, b] for a in range(na) for b in range(nb) for k in range(n_k) for i in range(n_mono)
                  if coef[i, k, a, b] != 0.0]
        t = torch.tensor(np.array(values + [0.0]), dtype=like.dtype, device=like.device)
        _ROTATIONS[key] = t
    return t


def _columns(la: int, lb: int, names: Sequence[str]) -> List[int]:
    lo, hi = min(la, lb), max(la, lb)
    return [list(names).index(f"{LETTERS[lo]}{LETTERS[hi]}{k}") for k in range(lo + 1)]


def _head(name: str, extra: str) -> List[str]:
    """A lane's pair: vector, distance, direction, block offsets and the lookup of both tables."""
    return [
        f"def {name}(vec_ptr, oa_ptr, ob_ptr, row_ptr, once_ptr, lma_ptr, lmb_ptr, tab_ptr, rows_ptr, start_ptr,",
        f"        count_ptr, grid_ptr, coef_ptr, rot_ptr, {extra}, P, NTAB, C, NB, LA, LB, BLOCK: tl.constexpr):",
        "    e = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)",
        "    m = e < P",
        "    m = m & (tl.load(lma_ptr + e, mask=m, other=0) >= LA) & (tl.load(lmb_ptr + e, mask=m, other=0) >= LB)",
        "    vx = tl.load(vec_ptr + e * 3, mask=m, other=1.0)",
        "    vy = tl.load(vec_ptr + e * 3 + 1, mask=m, other=0.0)",
        "    vz = tl.load(vec_ptr + e * 3 + 2, mask=m, other=0.0)",
        "    r = tl.sqrt(vx * vx + vy * vy + vz * vz)",
        "    ux = vx / r",
        "    uy = vy / r",
        "    uz = vz / r",
        "    oa = tl.load(oa_ptr + e, mask=m, other=0) + LA * LA",
        "    ob = tl.load(ob_ptr + e, mask=m, other=0) + LB * LB",
        "    base = tl.load(row_ptr + e, mask=m, other=0) * NB * NB",
        "    mo = m & (tl.load(once_ptr + e, mask=m, other=1) == 0)",
        "    tab = tl.load(tab_ptr + e, mask=m, other=0)",
    ]


def _interpolation(which: str, cols: List[int], order: int) -> List[str]:
    """Values ``<which><k>`` of the class's columns of one table, with their first (``d<which><k>``) and second
    (``dd<which><k>``) derivatives in the distance up to ``order``."""
    tid = "tab" if which == "H" else "tab + NTAB"
    lines = [
        f"    t_{which} = {tid}",
        f"    g_{which} = tl.load(grid_ptr + t_{which}, mask=m, other=1.0)",
        f"    n_{which} = tl.load(count_ptr + t_{which}, mask=m, other=8)",
        f"    s0_{which} = tl.load(start_ptr + t_{which}, mask=m, other=0)",
        f"    x_{which} = r / g_{which} - 1.0",
        f"    i0_{which} = tl.minimum(tl.maximum(tl.floor(x_{which}) - 3.0, 0.0), (n_{which} - 8).to(x_{which}.dtype))",
        f"    t8_{which} = x_{which} - i0_{which}",
        f"    first_{which} = s0_{which} + i0_{which}.to(tl.int64)",
        f"    r0_{which} = n_{which}.to(r.dtype) * g_{which}",
        f"    sv_{which} = tl.minimum(tl.maximum(r - r0_{which}, 0.0), 1.0)",
    ]
    # Lagrange weights of the nodes 0..7 and their derivatives in t: prefix and suffix products of the factors
    # t - m, each carried with its first and second derivative, divided by the constant products of (j - m)
    w = which
    lines += [f"    q{mm}_{w} = t8_{w} - {float(mm)}" for mm in range(8)]
    lines += [f"    P0_{w} = t8_{w} * 0.0 + 1.0", f"    S8_{w} = t8_{w} * 0.0 + 1.0"]
    if order >= 1:
        lines += [f"    dP0_{w} = t8_{w} * 0.0", f"    dS8_{w} = t8_{w} * 0.0"]
    if order >= 2:
        lines += [f"    ddP0_{w} = t8_{w} * 0.0", f"    ddS8_{w} = t8_{w} * 0.0"]
    for mm in range(8):
        a, b = mm, mm + 1
        if order >= 2:
            lines.append(f"    ddP{b}_{w} = ddP{a}_{w} * q{mm}_{w} + 2.0 * dP{a}_{w}")
        if order >= 1:
            lines.append(f"    dP{b}_{w} = dP{a}_{w} * q{mm}_{w} + P{a}_{w}")
        lines.append(f"    P{b}_{w} = P{a}_{w} * q{mm}_{w}")
    for mm in range(7, -1, -1):
        a, b = mm + 1, mm
        if order >= 2:
            lines.append(f"    ddS{b}_{w} = ddS{a}_{w} * q{mm}_{w} + 2.0 * dS{a}_{w}")
        if order >= 1:
            lines.append(f"    dS{b}_{w} = dS{a}_{w} * q{mm}_{w} + S{a}_{w}")
        lines.append(f"    S{b}_{w} = S{a}_{w} * q{mm}_{w}")
    for j in range(8):
        den = float(math.prod(j - mm for mm in range(8) if mm != j))
        lp, rs = f"P{j}_{w}", f"S{j + 1}_{w}"
        lines.append(f"    w{j}_{w} = {lp} * {rs} / {den}")
        if order >= 1:
            lines.append(f"    dw{j}_{w} = (d{lp} * {rs} + {lp} * d{rs}) / {den}")
        if order >= 2:
            lines.append(f"    d2w{j}_{w} = (dd{lp} * {rs} + 2.0 * d{lp} * d{rs} + {lp} * dd{rs}) / {den}")
    for k, col in enumerate(cols):
        acc = " + ".join(f"w{j}_{which} * tl.load(rows_ptr + (first_{which} + {j}) * C + {col}, mask=m, other=0.0)"
                         for j in range(8))
        lines.append(f"    in{k}_{which} = {acc}")
        c = [f"tl.load(coef_ptr + (t_{which} * 6 + {q}) * C + {col}, mask=m, other=0.0)" for q in range(6)]
        lines.append(f"    c0_{k}_{which} = {c[0]}")
        lines.append(f"    c1_{k}_{which} = {c[1]}")
        lines.append(f"    c2_{k}_{which} = {c[2]}")
        lines.append(f"    c3_{k}_{which} = {c[3]}")
        lines.append(f"    c4_{k}_{which} = {c[4]}")
        lines.append(f"    c5_{k}_{which} = {c[5]}")
        s = f"sv_{which}"
        lines.append(f"    out{k}_{which} = c0_{k}_{which} + {s} * (c1_{k}_{which} + {s} * (c2_{k}_{which} + {s} * "
                     f"(c3_{k}_{which} + {s} * (c4_{k}_{which} + {s} * c5_{k}_{which}))))")
        lines.append(f"    {which}{k} = tl.where(r < r0_{which}, in{k}_{which}, tl.where(r < r0_{which} + 1.0, "
                     f"out{k}_{which}, 0.0))")
        if order >= 1:
            dacc = " + ".join(f"dw{j}_{which} * tl.load(rows_ptr + (first_{which} + {j}) * C + {col}, mask=m, "
                              f"other=0.0)" for j in range(8))
            lines.append(f"    din{k}_{which} = ({dacc}) / g_{which}")
            lines.append(f"    dout{k}_{which} = c1_{k}_{which} + {s} * (2.0 * c2_{k}_{which} + {s} * (3.0 * "
                         f"c3_{k}_{which} + {s} * (4.0 * c4_{k}_{which} + {s} * 5.0 * c5_{k}_{which})))")
            lines.append(f"    d{which}{k} = tl.where(r < r0_{which}, din{k}_{which}, tl.where(r < r0_{which} + 1.0, "
                         f"dout{k}_{which}, 0.0))")
        if order >= 2:
            dacc = " + ".join(f"d2w{j}_{which} * tl.load(rows_ptr + (first_{which} + {j}) * C + {col}, mask=m, "
                              f"other=0.0)" for j in range(8))
            lines.append(f"    d2in{k}_{which} = ({dacc}) / (g_{which} * g_{which})")
            lines.append(f"    d2out{k}_{which} = 2.0 * c2_{k}_{which} + {s} * (6.0 * c3_{k}_{which} + {s} * (12.0 * "
                         f"c4_{k}_{which} + {s} * 20.0 * c5_{k}_{which}))")
            lines.append(f"    dd{which}{k} = tl.where(r < r0_{which}, d2in{k}_{which}, tl.where(r < r0_{which} + 1.0, "
                         f"d2out{k}_{which}, 0.0))")
    return lines


def _power(axis: str, p: int, times: int) -> Tuple[float, str]:
    """``(coefficient, name)`` of the ``times``-th derivative of ``u<axis>^p``."""
    coef = 1.0
    for q in range(times):
        coef *= p - q
    return coef, f"p{axis}{max(p - times, 0)}"


def _monomials(n: int, order: int) -> List[str]:
    """``mono<i>`` of degree ``n``, its first (``dmono<i>_<axis>``) and second (``d2mono<i>_<axes>``) derivatives up
    to ``order``."""
    lines = []
    for axis in "xyz":
        lines.append(f"    p{axis}0 = u{axis} * 0.0 + 1.0")
        for p in range(1, n + 1):
            lines.append(f"    p{axis}{p} = p{axis}{p - 1} * u{axis}")
    for i, (a, b, c) in enumerate(monomial_powers(n)):
        lines.append(f"    mono{i} = px{a} * py{b} * pz{c}")
        if order >= 1:
            lines.append(f"    dmono{i}_x = {float(a)} * px{max(a - 1, 0)} * py{b} * pz{c}" if a else
                         f"    dmono{i}_x = px0 * 0.0")
            lines.append(f"    dmono{i}_y = {float(b)} * px{a} * py{max(b - 1, 0)} * pz{c}" if b else
                         f"    dmono{i}_y = px0 * 0.0")
            lines.append(f"    dmono{i}_z = {float(c)} * px{a} * py{b} * pz{max(c - 1, 0)}" if c else
                         f"    dmono{i}_z = px0 * 0.0")
        if order >= 2:
            powers = {"x": a, "y": b, "z": c}
            for first, second in SECOND:
                times = {ax: (ax == first) + (ax == second) for ax in "xyz"}
                coef, names = 1.0, []
                for ax in "xyz":
                    k, name = _power(ax, powers[ax], times[ax])
                    coef *= k
                    names.append(name)
                expr = f"{coef} * " + " * ".join(names) if coef != 0.0 else "px0 * 0.0"
                lines.append(f"    d2mono{i}_{first}{second} = {expr}")
    return lines


def _sign(la: int, lb: int) -> str:
    return "1.0" if la <= lb or (la + lb) % 2 == 0 else "-1.0"


def _coefficients(la: int, lb: int) -> List[str]:
    """``rq<q>``: the rotation coefficients of the class, each loaded once."""
    n = sum(len(items) for row in _terms(la, lb) for cell in row for items in cell)
    return [f"    rq{q} = tl.load(rot_ptr + {q})" for q in range(n)]


def _used(la: int, lb: int) -> List[Tuple[int, int]]:
    """The ``(bond integral, monomial)`` pairs with a nonzero coefficient somewhere in the block."""
    return sorted({(k, i) for row in _terms(la, lb) for cell in row for k, items in enumerate(cell) for i, _ in items})


def _blocks(la: int, lb: int, src: str) -> List[Tuple[int, int, str]]:
    """``(a, b, expression)`` of every block element as the sum of its coefficients times ``<src><k>_<i>``."""
    out = []
    for a, row in enumerate(_terms(la, lb)):
        for b, cell in enumerate(row):
            expr = " + ".join(f"rq{q} * {src}{k}_{i}" for k, items in enumerate(cell) for i, q in items) or "r * 0.0"
            out.append((a, b, expr))
    return out


def _contracted(la: int, lb: int, which: str, gptr: str, sign: str) -> List[str]:
    """``b<which><k>_<i>``: the gradient of one matrix (both blocks of the pair, with the class's sign) contracted
    with the rotation coefficients, so that the block's derivatives follow from the monomials."""
    lines = [f"    b{which}{k}_{i} = r * 0.0" for k, i in _used(la, lb)]
    for a, row in enumerate(_terms(la, lb)):
        for b, cell in enumerate(row):
            lines.append(f"    gw = tl.load({gptr} + base + (oa + {a}) * NB + ob + {b}, mask=m, other=0.0)")
            lines.append(f"    gw = (gw + tl.load({gptr} + base + (ob + {b}) * NB + oa + {a}, mask=mo, other=0.0))"
                         f" * {sign}")
            for k, items in enumerate(cell):
                for i, q in items:
                    lines.append(f"    b{which}{k}_{i} = b{which}{k}_{i} + gw * rq{q}")
    return lines


def _values_source(name: str, la: int, lb: int, names: Sequence[str]) -> str:
    cols = _columns(la, lb, names)
    lines = _head(name, "s_ptr, h_ptr") + _interpolation("H", cols, 0) + _interpolation("S", cols, 0)
    lines += _monomials(la + lb, 0) + _coefficients(la, lb)
    sign = _sign(la, lb)
    for which, ptr in (("S", "s_ptr"), ("H", "h_ptr")):
        lines += [f"    y{which}{k}_{i} = {which}{k} * mono{i}" for k, i in _used(la, lb)]
        for a, b, expr in _blocks(la, lb, f"y{which}"):
            lines.append(f"    v = ({expr}) * {sign}")
            lines.append(f"    tl.atomic_add({ptr} + base + (oa + {a}) * NB + ob + {b}, v, mask=m)")
            lines.append(f"    tl.atomic_add({ptr} + base + (ob + {b}) * NB + oa + {a}, v, mask=mo)")
    return "\n".join(lines) + "\n"


def _table_lines(which: str, k: int, col: int, amount: str, slope: str = "") -> List[str]:
    """Atomic adds of the derivative in the table rows (inside the table) and tail coefficients (past it) of column
    ``k`` of one table: ``amount`` times the Lagrange weights and powers of the tail coordinate, and with ``slope``
    also ``slope`` times their derivatives in the distance."""
    w = which
    lines = [f"    mi = m & (r < r0_{w})", f"    mt = m & (r >= r0_{w}) & (r < r0_{w} + 1.0)"]
    for j in range(8):
        v = f"{amount} * w{j}_{w}" + (f" + {slope} * dw{j}_{w} / g_{w}" if slope else "")
        lines.append(f"    tl.atomic_add(tr_ptr + (first_{w} + {j}) * C + {col}, {v}, mask=mi)")
    lines.append("    sp0 = r * 0.0 + 1.0")
    for q in range(1, 6):
        lines.append(f"    sp{q} = sp{q - 1} * sv_{w}")
    for q in range(6):
        v = f"{amount} * sp{q}" + (f" + {slope} * {float(q)} * sp{max(q - 1, 0)}" if slope and q else "")
        lines.append(f"    tl.atomic_add(tc_ptr + (t_{w} * 6 + {q}) * C + {col}, {v}, mask=mt)")
    return lines


def _gradient_source(name: str, la: int, lb: int, names: Sequence[str], tables: bool = False) -> str:
    cols = _columns(la, lb, names)
    extra = "gs_ptr, gh_ptr, out_ptr" + (", tr_ptr, tc_ptr" if tables else "")
    lines = _head(name, extra) + _interpolation("H", cols, 1) + _interpolation("S", cols, 1)
    lines += _monomials(la + lb, 1) + _coefficients(la, lb)
    sign = _sign(la, lb)
    lines += ["    fr = r * 0.0"] + [f"    w{c} = r * 0.0" for c in "xyz"]
    for which, gptr in (("S", "gs_ptr"), ("H", "gh_ptr")):
        lines += _contracted(la, lb, which, gptr, sign)
        for k in range(len(cols)):
            mono = [i for kk, i in _used(la, lb) if kk == k]
            lines.append("    al = " + " + ".join(f"b{which}{k}_{i} * mono{i}" for i in mono))
            lines.append(f"    fr = fr + d{which}{k} * al")
            if tables:
                lines += _table_lines(which, k, cols[k], "al")
            for c in "xyz":
                lines.append(f"    w{c} = w{c} + {which}{k} * (" +
                             " + ".join(f"b{which}{k}_{i} * dmono{i}_{c}" for i in mono) + ")")
    lines += [
        "    wu = wx * ux + wy * uy + wz * uz",
        "    tl.store(out_ptr + e * 3, fr * ux + (wx - wu * ux) / r, mask=m)",
        "    tl.store(out_ptr + e * 3 + 1, fr * uy + (wy - wu * uy) / r, mask=m)",
        "    tl.store(out_ptr + e * 3 + 2, fr * uz + (wz - wu * uz) / r, mask=m)",
    ]
    return "\n".join(lines) + "\n"


def _second_source(name: str, la: int, lb: int, names: Sequence[str], tables: bool = False) -> str:
    """The derivative of the gradient kernel along the vectors ``delta``: the tangent matrices (the derivative of
    both matrices along ``delta``, the gradient in ``gS`` and ``gH``) and the Hessian of the contracted matrices
    applied to ``delta``."""
    cols = _columns(la, lb, names)
    used = _used(la, lb)
    lines = _head(name, "gs_ptr, gh_ptr, dl_ptr, ts_ptr, th_ptr, out_ptr" + (", tr_ptr, tc_ptr" if tables else ""))
    lines += _interpolation("H", cols, 2) + _interpolation("S", cols, 2) + _monomials(la + lb, 2)
    lines += _coefficients(la, lb)
    lines += [
        "    dx = tl.load(dl_ptr + e * 3, mask=m, other=0.0)",
        "    dy = tl.load(dl_ptr + e * 3 + 1, mask=m, other=0.0)",
        "    dz = tl.load(dl_ptr + e * 3 + 2, mask=m, other=0.0)",
        "    dr = ux * dx + uy * dy + uz * dz",
        "    dux = (dx - dr * ux) / r",
        "    duy = (dy - dr * uy) / r",
        "    duz = (dz - dr * uz) / r",
    ]
    monos = sorted({i for _, i in used})
    lines += [f"    s{i} = dmono{i}_x * dux + dmono{i}_y * duy + dmono{i}_z * duz" for i in monos]
    sign = _sign(la, lb)
    lines += ["    fr = r * 0.0", "    frr = r * 0.0"]
    lines += [f"    gr{c} = r * 0.0" for c in "xyz"] + [f"    w{c} = r * 0.0" for c in "xyz"]
    lines += [f"    h{c}{d} = r * 0.0" for c, d in SECOND]
    for which, gptr, tptr in (("S", "gs_ptr", "ts_ptr"), ("H", "gh_ptr", "th_ptr")):
        # the tangent: the derivative of every block element along delta
        lines += [f"    z{which}{k}_{i} = d{which}{k} * dr * mono{i} + {which}{k} * s{i}" for k, i in used]
        for a, b, expr in _blocks(la, lb, f"z{which}"):
            lines.append(f"    tv = ({expr}) * {sign}")
            lines.append(f"    tl.atomic_add({tptr} + base + (oa + {a}) * NB + ob + {b}, tv, mask=m)")
            lines.append(f"    tl.atomic_add({tptr} + base + (ob + {b}) * NB + oa + {a}, tv, mask=mo)")
        # the contracted angular parts, their gradient and Hessian in u, with the radial parts
        lines += _contracted(la, lb, which, gptr, sign)
        for k in range(len(cols)):
            mono = [i for kk, i in used if kk == k]
            lines.append("    al = " + " + ".join(f"b{which}{k}_{i} * mono{i}" for i in mono))
            lines.append(f"    fr = fr + d{which}{k} * al")
            lines.append(f"    frr = frr + dd{which}{k} * al")
            lines.append("    bb = r * 0.0")
            for c in "xyz":
                lines.append("    ga = " + " + ".join(f"b{which}{k}_{i} * dmono{i}_{c}" for i in mono))
                lines.append(f"    gr{c} = gr{c} + d{which}{k} * ga")
                lines.append(f"    w{c} = w{c} + {which}{k} * ga")
                lines.append(f"    bb = bb + ga * du{c}")
            if tables:
                # the derivative of delta . gradient in this column's value and slope: dr * al and bb
                lines += _table_lines(which, k, cols[k], "bb", "(al * dr)")
            for c, d in SECOND:
                lines.append("    ha = " + " + ".join(f"b{which}{k}_{i} * d2mono{i}_{c}{d}" for i in mono))
                lines.append(f"    h{c}{d} = h{c}{d} + {which}{k} * ha")
    lines += [
        "    hyx = hxy",
        "    hzx = hxz",
        "    hzy = hyz",
        "    da = frr * dr + grx * dux + gry * duy + grz * duz",
        "    wu = wx * ux + wy * uy + wz * uz",
    ]
    for c in "xyz":
        lines.append(f"    dw{c} = gr{c} * dr + h{c}x * dux + h{c}y * duy + h{c}z * duz")
    lines.append("    dwu = dwx * ux + dwy * uy + dwz * uz + wx * dux + wy * duy + wz * duz")
    for k, c in enumerate("xyz"):
        lines.append(f"    o{c} = da * u{c} + fr * du{c} + (dw{c} - dwu * u{c} - wu * du{c}) / r"
                     f" - (w{c} - wu * u{c}) * dr / (r * r)")
        lines.append(f"    tl.store(out_ptr + e * 3 + {k}, o{c}, mask=m)")
    return "\n".join(lines) + "\n"


_SOURCES = {"values": _values_source, "gradient": _gradient_source, "second": _second_source,
            "gradient_tables": lambda *a: _gradient_source(*a, tables=True),
            "second_tables": lambda *a: _second_source(*a, tables=True)}


def _kernel(kind: str, la: int, lb: int, names: Sequence[str]):
    """The kernel of a class (generated once per process)."""
    key = (kind, la, lb, len(names))
    fn = _KERNELS.get(key)
    if fn is None:
        import triton
        import triton.language as tl
        name = f"_sk_{kind}_{la}{lb}_{len(names)}"
        src = _SOURCES[kind](name, la, lb, names)
        # Triton reads a kernel's source through inspect, which finds it in linecache
        filename = f"<xnn-dftb-{name}>"
        linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
        namespace = {"__name__": __name__, "tl": tl, "triton": triton}
        exec(compile(src, filename, "exec"), namespace)
        fn = triton.jit(namespace[name])
        _KERNELS[key] = fn
    return fn


def _launch(kind: str, la: int, lb: int, names: Sequence[str], tensors: tuple, n: int, ints: tuple) -> None:
    kernel = _kernel(kind, la, lb, names)
    import triton
    block = BLOCK if tensors[0].is_cuda else 4                       # a CPU runs them in Triton's interpreter
    extra = {"num_warps": WARPS} if tensors[0].is_cuda else {}
    kernel[(triton.cdiv(max(n, 1), block),)](*tensors, n, *ints, BLOCK=block, **extra)


class PairTables:
    """The per-pair inputs of the kernels: vectors ``(E, 3)``, first functions of both atoms within their
    structure, matrix row, the flag of an atom's own images, the atoms' highest shells and the table (of the
    side the class reads: ``A-B`` for ``la <= lb``, ``B-A`` otherwise)."""

    def __init__(self, oa: Tensor, ob: Tensor, row: Tensor, once: Tensor, lma: Tensor, lmb: Tensor, ab: Tensor,
                 ba: Tensor):
        self.oa, self.ob, self.row = oa.contiguous(), ob.contiguous(), row.contiguous()
        self.once = once.to(torch.int32).contiguous()
        self.lma, self.lmb, self.ab, self.ba = lma.contiguous(), lmb.contiguous(), ab.contiguous(), ba.contiguous()

    def args(self, vec: Tensor, la: int, lb: int) -> tuple:
        return (vec, self.oa, self.ob, self.row, self.once, self.lma, self.lmb, self.ab if la <= lb else self.ba)


def matrices(vec: Tensor, pairs: PairTables, stacked: tuple, top: int, n_rows: int, nb: int) -> Tuple[Tensor, Tensor]:
    """Overlap and Hamiltonian ``(n_rows, nb, nb)`` (zero diagonal blocks) of the pairs of vectors ``vec``,
    from the stacked tables ``stacked`` (rows, starts, counts, grids, tail coefficients, column names)."""
    rows, start, count, grid, coef, names = stacked
    vec = vec.detach().contiguous()
    S = vec.new_zeros(n_rows, nb, nb)
    H = vec.new_zeros(n_rows, nb, nb)
    n_tab = start.shape[0] // 2
    for la in range(top + 1):
        for lb in range(top + 1):
            tensors = pairs.args(vec, la, lb) + (rows, start, count, grid, coef, _rotation_table(la, lb, vec), S, H)
            _launch("values", la, lb, names, tensors, vec.shape[0], (n_tab, rows.shape[1], nb, la, lb))
    return S, H


def gradient(vec: Tensor, pairs: PairTables, stacked: tuple, top: int, gS: Tensor, gH: Tensor,
             tables: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
    """Derivative ``(E, 3)`` in the pair vectors of the matrices of :func:`matrices` contracted with ``gS`` and
    ``gH``; with ``tables`` also the derivatives in the table rows and tail coefficients (empty otherwise)."""
    rows, start, count, grid, coef, names = stacked
    vec = vec.detach().contiguous()
    gS, gH = gS.detach().contiguous(), gH.detach().contiguous()
    out = torch.zeros_like(vec)
    t_rows = torch.zeros_like(rows) if tables else rows.new_zeros(0)
    t_coef = torch.zeros_like(coef) if tables else coef.new_zeros(0)
    n_tab, nb = start.shape[0] // 2, gS.shape[-1]
    for la in range(top + 1):
        for lb in range(top + 1):
            g = torch.zeros_like(vec)
            tensors = pairs.args(vec, la, lb) + (rows.detach(), start, count, grid, coef.detach(),
                                                 _rotation_table(la, lb, vec), gS, gH, g)
            tensors += (t_rows, t_coef) if tables else ()
            _launch("gradient_tables" if tables else "gradient", la, lb, names, tensors, vec.shape[0],
                    (n_tab, rows.shape[1], nb, la, lb))
            out = out + g
    return out, t_rows, t_coef


def second(vec: Tensor, pairs: PairTables, stacked: tuple, top: int, gS: Tensor, gH: Tensor, delta: Tensor,
           tables: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """The derivative of :func:`gradient` along the vectors ``delta`` ``(E, 3)``: its derivative in the vectors
    ``(E, 3)``, the tangent overlap and Hamiltonian (its derivatives in ``gS`` and ``gH``) and, with ``tables``, its
    derivatives in the table rows and tail coefficients (empty otherwise)."""
    rows, start, count, grid, coef, names = stacked
    vec, delta = vec.detach().contiguous(), delta.detach().contiguous()
    gS, gH = gS.detach().contiguous(), gH.detach().contiguous()
    out, tS, tH = torch.zeros_like(vec), torch.zeros_like(gS), torch.zeros_like(gH)
    t_rows = torch.zeros_like(rows) if tables else rows.new_zeros(0)
    t_coef = torch.zeros_like(coef) if tables else coef.new_zeros(0)
    n_tab, nb = start.shape[0] // 2, gS.shape[-1]
    for la in range(top + 1):
        for lb in range(top + 1):
            g = torch.zeros_like(vec)
            tensors = pairs.args(vec, la, lb) + (rows.detach(), start, count, grid, coef.detach(),
                                                 _rotation_table(la, lb, vec), gS, gH, delta, tS, tH, g)
            tensors += (t_rows, t_coef) if tables else ()
            _launch("second_tables" if tables else "second", la, lb, names, tensors, vec.shape[0],
                    (n_tab, rows.shape[1], nb, la, lb))
            out = out + g
    return out, tS, tH, t_rows, t_coef


class TwoCenter(torch.autograd.Function):
    """:func:`matrices` of the vectors ``vec`` and the table rows and tail coefficients (the stacked tables are
    ``(rows, *meta[:3], coef, meta[3])``), with :func:`gradient` as its backward, which is differentiable once more
    (see :class:`TwoCenterGradient`)."""

    @staticmethod
    def forward(ctx, vec: Tensor, rows: Tensor, coef: Tensor, pairs: PairTables, meta: tuple, top: int, n_rows: int,
                nb: int):
        ctx.pairs, ctx.meta, ctx.top, ctx.shape = pairs, meta, top, (n_rows, nb, nb)
        ctx.save_for_backward(vec, rows, coef)
        start, count, grid, names = meta
        return matrices(vec, pairs, (rows.detach(), start, count, grid, coef.detach(), names), top, n_rows, nb)

    @staticmethod
    def backward(ctx, gS: Tensor, gH: Tensor):
        vec, rows, coef = ctx.saved_tensors
        gS = vec.new_zeros(ctx.shape) if gS is None else gS
        gH = vec.new_zeros(ctx.shape) if gH is None else gH
        tables = bool(ctx.needs_input_grad[1] or ctx.needs_input_grad[2])
        g, g_rows, g_coef = TwoCenterGradient.apply(vec, gS, gH, rows, coef, ctx.pairs, ctx.meta, ctx.top, tables)
        return g, (g_rows if tables else None), (g_coef if tables else None), None, None, None, None, None


class TwoCenterGradient(torch.autograd.Function):
    """:func:`gradient` (with the table derivatives when ``tables``) and :func:`second` as its backward (once
    differentiable: third derivatives take the reference)."""

    @staticmethod
    def forward(ctx, vec: Tensor, gS: Tensor, gH: Tensor, rows: Tensor, coef: Tensor, pairs: PairTables, meta: tuple,
                top: int, tables: bool):
        ctx.set_materialize_grads(False)
        ctx.pairs, ctx.meta, ctx.top, ctx.tables = pairs, meta, top, tables
        ctx.save_for_backward(vec, gS, gH, rows, coef)
        start, count, grid, names = meta
        out = gradient(vec, pairs, (rows, start, count, grid, coef, names), top, gS, gH, tables)
        if not tables:
            ctx.mark_non_differentiable(out[1], out[2])
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, delta: Tensor, d_rows: Tensor, d_coef: Tensor):
        if d_rows is not None or d_coef is not None:
            raise NotImplementedError("derivatives of the table gradients take the reference (use_fast=False)")
        vec, gS, gH, rows, coef = ctx.saved_tensors
        start, count, grid, names = ctx.meta
        dvec, tS, tH, t_rows, t_coef = second(vec, ctx.pairs, (rows, start, count, grid, coef, names), ctx.top, gS,
                                              gH, delta, ctx.tables)
        if not ctx.tables:
            t_rows = t_coef = None
        return dvec, tS, tH, t_rows, t_coef, None, None, None, None
