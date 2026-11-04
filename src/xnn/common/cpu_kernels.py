"""Triton kernels on the CPU.

:func:`launch` runs a ``@triton.jit`` kernel on CPU tensors as ``kernel[grid](*args, **meta)`` runs it on a GPU.
The kernel's source is translated into C++ once per specialization (the constexpr values and the argument
dtypes, as Triton specializes), compiled with the system C++ compiler and cached on disk. Every lane of every
program becomes one iteration of a loop that OpenMP threads share; the helpers a kernel calls are inlined.

Lane semantics:

- a masked load reads only where its mask holds;
- an atomic add goes to a buffer of the calling thread and the buffers are summed into the tensor after all
  lanes ran (each thread sums a share of the touched blocks, so a run needs no atomics and its sums do not
  depend on the scheduling);
- a block reduction ``tl.sum`` is allowed only as the value of an atomic add, where each lane adds its own term;
- a value that is the same in every lane of a program (a program id, a scalar argument, a load at such an
  address) stays uniform, and an atomic add or a store of a uniform value at a uniform address runs once per
  program.

Literals follow Triton's typing: a Python float with a float32 value is float32, with an integer value float32,
two literals stay a literal. A kernel outside these rules (another block reduction, a second grid axis) raises
:class:`Unsupported`; its helpers can be given lane-wise twins with :func:`override`.
"""
import ast
import ctypes
import functools
import hashlib
import inspect
import math
import operator
import os
import platform
import re
import subprocess
import tempfile
import textwrap
import threading
from pathlib import Path
from typing import Optional

import torch

__all__ = ["launch", "prepare", "Launch", "Sequence", "Deferred", "available", "override", "Unsupported",
           "translate"]


class Unsupported(Exception):
    """A construct the lane translation does not cover."""


_FLOATS = ("f32", "f64")
_INTS = ("b", "i8", "u8", "i16", "i32", "i64")
_RANK = {"b": 0, "i8": 1, "u8": 1, "i16": 2, "i32": 3, "i64": 4}
_C = {"f64": "double", "f32": "float", "i64": "int64_t", "i32": "int32_t", "i16": "int16_t", "i8": "int8_t",
      "u8": "uint8_t", "b": "bool", "wf": "double", "wi": "int64_t"}
_TORCH = {torch.float64: "f64", torch.float32: "f32", torch.int64: "i64", torch.int32: "i32", torch.int16: "i16",
          torch.int8: "i8", torch.uint8: "u8", torch.bool: "b"}
_TL = {"fp64": "f64", "fp32": "f32", "int64": "i64", "int32": "i32", "int16": "i16", "int8": "i8", "uint8": "u8",
       "int1": "b"}
_MATH = {"sqrt": "std::sqrt", "exp": "std::exp", "log": "std::log", "abs": "std::abs", "sin": "std::sin",
         "cos": "std::cos", "floor": "std::floor", "ceil": "std::ceil", "rsqrt": None, "erf": "std::erf",
         "tanh": "std::tanh", "exp2": "std::exp2", "log2": "std::log2", "sigmoid": None, "fma": "std::fma",
         "acos": "std::acos", "asin": "std::asin", "atan": "std::atan", "atan2": "std::atan2", "pow": "std::pow",
         "log1p": "std::log1p", "expm1": "std::expm1", "cbrt": "std::cbrt", "sinh": "std::sinh", "cosh": "std::cosh",
         "fabs": "std::fabs", "erfc": "std::erfc", "rint": "std::rint", "trunc": "std::trunc"}
_UNROLL = 256
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_BLOCK_SHIFT = 8

_OVERRIDES: dict = {}


def override(fn, source: str):
    """Translate the jit helper ``fn`` from ``source`` (a function of the same name and arguments in Triton
    syntax, resolved in ``fn``'s module) instead of its own source: the lane-wise twin of a helper with a
    block reduction."""
    f = getattr(fn, "fn", fn)
    _OVERRIDES[(f.__module__, f.__name__)] = textwrap.dedent(source)


def _tl_type(dtype) -> str:
    s = str(dtype)
    if s not in _TL:
        raise Unsupported(f"dtype {s}")
    return _TL[s]


def _tl_dtype(ty: str):
    """The Triton dtype of a scalar type."""
    import triton.language as tl
    return tl.dtype(next(k for k, v in _TL.items() if v == ty))


def _promote(a: str, b: str) -> str:
    """The type of a binary operation (Triton's rules with weak literals ``wf`` / ``wi``)."""
    if a == b:
        return a
    weak = ("wf", "wi")
    if a in weak and b in weak:
        return "wf"
    if a in weak:
        a, b = b, a
    if b == "wi":
        return "i32" if a == "b" else a
    if b == "wf":
        return a if a in _FLOATS else "f32"
    if a in _FLOATS or b in _FLOATS:
        return "f64" if "f64" in (a, b) else "f32"
    return a if _RANK[a] >= _RANK[b] else b


def _strong(t: str) -> str:
    """A literal's type where it becomes a value of its own (``tl.where`` of two literals)."""
    return {"wf": "f32", "wi": "i32"}.get(t, t)


class V:
    """A translated value: C++ code, a type (a scalar type or ``("ptr", element type, base argument)``),
    whether it is uniform across the lanes of a program, and the lane terms of a ``tl.sum`` or the condition
    and value of a ``tl.where(c, v, 0)``."""

    __slots__ = ("code", "ty", "uni", "sumof", "where0", "conj")

    def __init__(self, code, ty, uni, sumof=None, where0=None, conj=frozenset()):
        self.code, self.ty, self.uni, self.sumof, self.where0 = code, ty, uni, sumof, where0
        # a boolean's conjuncts that are variables (a lane where one is false has the whole value false)
        self.conj = conj

    @property
    def ptr(self) -> bool:
        return isinstance(self.ty, tuple)


def _lit(value) -> V:
    if isinstance(value, bool):
        return V("true" if value else "false", "b", True)
    if isinstance(value, int):
        return V(f"INT64_C({value})", "wi", True)
    if isinstance(value, float):
        if math.isnan(value):
            return V("std::numeric_limits<double>::quiet_NaN()", "wf", True)
        if value in (float("inf"), float("-inf")):
            return V(("-" if value < 0 else "") + "std::numeric_limits<double>::infinity()", "wf", True)
        return V(repr(value), "wf", True)
    raise Unsupported(f"constant {value!r}")


def _cast(v: V, ty: str) -> str:
    if v.sumof is not None:
        raise Unsupported("a block reduction used other than as the value of an atomic add")
    if v.ptr:
        raise Unsupported("a pointer used as a value")
    if v.ty == ty:
        return v.code
    return f"static_cast<{_C[ty]}>({v.code})"


_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow, ast.BitAnd: operator.and_,
           ast.BitOr: operator.or_, ast.BitXor: operator.xor, ast.LShift: operator.lshift, ast.RShift: operator.rshift}
_COMPARE = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Eq: operator.eq,
            ast.NotEq: operator.ne, ast.Is: operator.is_, ast.IsNot: operator.is_not, ast.In: lambda a, b: a in b,
            ast.NotIn: lambda a, b: a not in b}


class _Const:
    """A compile-time value in the environment (a constexpr argument, a module constant, a dtype)."""

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


def _jit_fn(obj):
    """The Python function of a ``@triton.jit`` object (compiled, or under ``TRITON_INTERPRET``), or ``None``."""
    try:
        from triton.runtime.jit import JITFunction
    except ImportError:
        return None
    kinds = (JITFunction,)
    try:
        from triton.runtime.interpreter import InterpretedFunction
        kinds += (InterpretedFunction,)
    except ImportError:
        pass
    return obj.fn if isinstance(obj, kinds) else None


_DEFS: dict = {}


def _function_def(fn) -> ast.FunctionDef:
    key = (fn.__module__, fn.__name__)
    src = _OVERRIDES.get(key)
    cached = _DEFS.get(key)
    if cached is not None and cached[0] == src:
        return cached[1]
    for node in ast.parse(src or textwrap.dedent(inspect.getsource(fn))).body:
        if isinstance(node, ast.FunctionDef) and node.name == fn.__name__:
            _DEFS[key] = (src, node)
            return node
    raise Unsupported(f"no source for {fn.__name__}")


def _dotted_name(node) -> str:
    if isinstance(node, ast.Attribute):
        return _dotted_name(node.value) + "." + node.attr
    return getattr(node, "id", "")


def _assigned(stmts) -> set:
    out = set()
    for st in stmts:
        for node in ast.walk(st):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                out.add(node.id)
    return out


class _Translator:
    """The C++ body of one kernel specialization."""

    def __init__(self):
        self.lines = []
        self.indent = 4
        self.prologue = []     # declarations uniform across a program's lanes, before its lane loop
        self.epilogue = []     # the stores of a program's block sums, after its lane loop
        self.known = {"j", "pid"}
        self.at = {}           # a lane-loop declaration -> the line after it
        self.at_any = {}       # any declaration -> the line after it and its indentation
        self.effects = []      # the conjuncts guarding each store and add
        self.hoisted = {"pid"}
        self.count = 0
        self.targets = {}      # base argument -> element type of the atomic adds into it
        self.stored = set()
        self.loaded = set()
        self.mutable = set()   # the variables of carried names (a name bound to one takes a copy)
        self.loops = 0         # the runtime loops around the statement being translated
        self.lane_return = False
        self.sum_stores = False  # a block sum stored (not added) after the lane loop

    def emit(self, line: str):
        self.lines.append("  " * self.indent + line)

    def fresh(self, name: str) -> str:
        self.count += 1
        var = f"{name}_{self.count}"
        self.known.add(var)
        return var

    def insert(self, at: int, line: str):
        """A line inserted at ``at``; the recorded positions after it move down."""
        self.lines.insert(at, line)
        self.at = {k: (i + 1 if i >= at else i) for k, i in self.at.items()}
        self.at_any = {k: ((i + 1, d) if i >= at else (i, d)) for k, (i, d) in self.at_any.items()}

    def place(self, line: str, v: V, var: str):
        """A declaration: before the lane loop when it is uniform, outside any runtime block and reads only
        names declared there."""
        names = set(_IDENT.findall(line.split("=", 1)[1])) & self.known
        if v.uni and self.indent == 4 and names <= self.hoisted:
            self.prologue.append("      " + line)
            self.hoisted.add(var)
        else:
            self.emit(line)
            if self.indent == 4:
                self.at[var] = len(self.lines)
            self.at_any[var] = (len(self.lines), self.indent)

    def declare(self, name: str, v: V) -> V:
        """A new C++ variable holding ``v``."""
        if v.ptr:
            var = self.fresh(name)
            self.place(f"{_C[v.ty[1]]}* {var} = {v.code};", v, var)
            return V(var, v.ty, v.uni)
        if v.sumof is not None:
            return v
        ty = v.ty
        var = self.fresh(name)
        self.place(f"const {_C[ty]} {var} = {v.code};", v, var)
        return V(var, ty, v.uni, where0=v.where0, conj=(v.conj | {var}) if ty == "b" else frozenset())


class _Scope:
    """The names of one inlined function: compile-time values and translated values."""

    def __init__(self, tr: _Translator, globs: dict, env: dict):
        self.tr, self.globs, self.env = tr, globs, env
        self.returned = None
        self.kernel = False

    # expressions

    def const(self, node):
        """The compile-time value of ``node`` (raises KeyError if it is not one)."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in self.env:
                b = self.env[node.id]
                if isinstance(b, _Const):
                    return b.value
                raise KeyError(node.id)
            if node.id in self.globs:
                return self.globs[node.id]
            import builtins
            if hasattr(builtins, node.id):
                return getattr(builtins, node.id)
            raise KeyError(node.id)
        if isinstance(node, ast.Attribute):
            inner = node.value
            if (node.attr == "element_ty" and isinstance(inner, ast.Attribute) and inner.attr == "dtype"
                    and isinstance(inner.value, ast.Name) and isinstance(self.env.get(inner.value.id), V)
                    and self.env[inner.value.id].ptr):
                # the element type of a pointer argument is fixed by the specialization
                return _tl_dtype(self.env[inner.value.id].ty[1])
            if (node.attr == "dtype" and isinstance(inner, ast.Name) and isinstance(self.env.get(inner.id), V)
                    and not self.env[inner.id].ptr):
                # so is the type of a value
                ty = self.env[inner.id].ty
                return _tl_dtype(_strong(ty) if ty in ("wf", "wi") else ty)
            base = self.const(inner)
            return getattr(base, node.attr)
        if isinstance(node, ast.UnaryOp):
            v = self.const(node.operand)
            return {ast.USub: lambda x: -x, ast.UAdd: lambda x: +x, ast.Not: lambda x: not x,
                    ast.Invert: lambda x: ~x}[type(node.op)](v)
        if isinstance(node, ast.BinOp):
            return _BINOPS[type(node.op)](self.const(node.left), self.const(node.right))
        if isinstance(node, ast.BoolOp):
            vals = [self.const(v) for v in node.values]
            out = vals[0]
            for v in vals[1:]:
                out = (out and v) if isinstance(node.op, ast.And) else (out or v)
            return out
        if isinstance(node, ast.Compare):
            left = self.const(node.left)
            for op, right in zip(node.ops, node.comparators):
                r = self.const(right)
                if not _COMPARE[type(op)](left, r):
                    return False
                left = r
            return True
        if isinstance(node, ast.IfExp):
            return self.const(node.body) if self.const(node.test) else self.const(node.orelse)
        if isinstance(node, ast.Call):
            f = self.const(node.func)
            if f in (min, max, abs, int, float, bool, len, round):
                return f(*[self.const(a) for a in node.args])
            raise KeyError("call")
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(self.const(e) for e in node.elts)
        raise KeyError(type(node).__name__)

    def is_const(self, node):
        try:
            self.const(node)
            return True
        except Exception:
            return False

    def expr(self, node) -> V:
        if not isinstance(node, ast.Call) or not self._is_tl_call(node):
            try:
                c = self.const(node)
                if isinstance(c, (bool, int, float)):
                    return _lit(c)
            except Exception:
                pass
        if isinstance(node, ast.Name):
            b = self.env.get(node.id)
            if isinstance(b, V):
                return b
            raise Unsupported(f"name {node.id}")
        if isinstance(node, ast.BinOp):
            return self.binop(self.expr(node.left), node.op, self.expr(node.right))
        if isinstance(node, ast.UnaryOp):
            v = self.expr(node.operand)
            if isinstance(node.op, ast.USub):
                if v.where0 is not None:
                    c, w = v.where0
                    return V(f"(-{v.code})", v.ty, v.uni, where0=(c, V(f"(-{w.code})", w.ty, w.uni)))
                return V(f"(-{v.code})", v.ty, v.uni)
            if isinstance(node.op, ast.UAdd):
                return v
            if isinstance(node.op, (ast.Invert, ast.Not)):
                if v.ty == "b" or isinstance(node.op, ast.Not):
                    return V(f"(!{_cast(v, 'b')})", "b", v.uni)
                return V(f"(~{v.code})", v.ty, v.uni)
        if isinstance(node, ast.Compare):
            out = None
            left = self.expr(node.left)
            for op, right in zip(node.ops, node.comparators):
                r = self.expr(right)
                t = _promote(left.ty, r.ty)
                sym = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}[type(op)]
                tc = _strong(t) if t in ("wf", "wi") else t
                c = V(f"({_cast(left, tc)} {sym} {_cast(r, tc)})", "b", left.uni and r.uni)
                out = c if out is None else V(f"({out.code} & {c.code})", "b", out.uni and c.uni)
                left = r
            return out
        if isinstance(node, ast.BoolOp):
            vals = [self.expr(v) for v in node.values]
            sym = " && " if isinstance(node.op, ast.And) else " || "
            conj = frozenset().union(*[v.conj for v in vals]) if isinstance(node.op, ast.And) else frozenset()
            return V("(" + sym.join(_cast(v, "b") for v in vals) + ")", "b", all(v.uni for v in vals), conj=conj)
        if isinstance(node, ast.IfExp):
            if self.is_const(node.test):
                return self.expr(node.body if self.const(node.test) else node.orelse)
            return self.where(self.expr(node.test), self.expr(node.body), self.expr(node.orelse))
        if isinstance(node, ast.Call):
            return self.call(node)
        raise Unsupported(f"expression {ast.dump(node)[:80]}")

    def reduce_store(self, p: V, term: V, m: Optional[V]):
        """``tl.store(p, tl.sum(term), mask=m)`` at an address uniform across the program: the lanes add their
        terms into one accumulator of the program, stored after its lanes ran."""
        tr = self.tr
        uses = set(_IDENT.findall(p.code + ("" if m is None else m.code))) & tr.known
        if not (p.uni and (m is None or m.uni) and tr.indent == 4 and uses <= tr.hoisted):
            raise Unsupported("a block sum stored other than once per program at a uniform address")
        tr.sum_stores = True
        ty = p.ty[1]
        acc = tr.fresh("sum")
        tr.prologue.append(f"      {_C[ty]} {acc} = 0;")
        if term.where0 is not None:
            c, w = term.where0
            tr.emit(f"if ({_cast(c, 'b')}) {acc} += {_cast(w, ty)};")
            tr.effects.append(c.conj)
        else:
            tr.emit(f"{acc} += {_cast(term, ty)};")
            tr.effects.append(frozenset())
        stmt = f"*{p.code} = {acc};"
        tr.epilogue.append("      " + (f"if ({_cast(m, 'b')}) {stmt}" if m is not None else stmt))

    def dtype_of(self, node) -> str:
        """The scalar type a ``dtype`` argument names: a Triton dtype or ``x.dtype`` of a value."""
        if isinstance(node, ast.Attribute) and node.attr == "dtype" and not self.is_const(node):
            v = self.expr(node.value)
            return _strong(v.ty) if v.ty in ("wf", "wi") else v.ty
        return _tl_type(self.const(node))

    def _is_tl_call(self, node) -> bool:
        try:
            f = self.const(node.func)
        except Exception:
            return False
        mod = getattr(f, "__module__", "") or ""
        return mod.startswith("triton") or _jit_fn(f) is not None

    def binop(self, a: V, op, b: V) -> V:
        if a.ptr or b.ptr:
            if isinstance(op, ast.Add) and a.ptr != b.ptr:
                p, i = (a, b) if a.ptr else (b, a)
                return V(f"({p.code} + {_cast(i, _strong(i.ty) if i.ty == 'wi' else i.ty)})", p.ty, p.uni and i.uni)
            if isinstance(op, ast.Sub) and a.ptr and not b.ptr:
                return V(f"({a.code} - {b.code})", a.ty, a.uni and b.uni)
            raise Unsupported("pointer arithmetic")
        uni = a.uni and b.uni
        t = _promote(a.ty, b.ty)
        if isinstance(op, ast.Div):
            if t in _INTS or t == "wi":
                t = "f32"
            return V(f"({_cast(a, t)} / {_cast(b, t)})", t, uni)
        if isinstance(op, ast.FloorDiv):
            if t in _FLOATS or t == "wf":
                return V(f"std::floor({_cast(a, t)} / {_cast(b, t)})", t, uni)
            return V(f"({_cast(a, t)} / {_cast(b, t)})", t, uni)
        if isinstance(op, ast.Mod):
            if t in _FLOATS or t == "wf":
                return V(f"std::fmod({_cast(a, t)}, {_cast(b, t)})", t, uni)
            return V(f"({_cast(a, t)} % {_cast(b, t)})", t, uni)
        if isinstance(op, ast.Pow):
            return V(f"std::pow({_cast(a, t)}, {_cast(b, t)})", t, uni)
        if isinstance(op, (ast.BitAnd, ast.BitOr, ast.BitXor)):
            sym = {ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^"}[type(op)]
            if t == "b":
                conj = a.conj | b.conj if isinstance(op, ast.BitAnd) else frozenset()
                return V(f"static_cast<bool>({a.code} {sym} {b.code})", "b", uni, conj=conj)
            return V(f"({_cast(a, t)} {sym} {_cast(b, t)})", t, uni)
        if isinstance(op, (ast.LShift, ast.RShift)):
            sym = "<<" if isinstance(op, ast.LShift) else ">>"
            return V(f"({_cast(a, t)} {sym} {_cast(b, t)})", t, uni)
        sym = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*"}.get(type(op))
        if sym is None:
            raise Unsupported(f"operator {type(op).__name__}")
        if t == "b":
            t = "i32"
        return V(f"({_cast(a, t)} {sym} {_cast(b, t)})", t, uni)

    def where(self, c: V, a: V, b: V) -> V:
        t = _promote(a.ty, b.ty)
        t = _strong(t) if t in ("wf", "wi") else t
        out = V(f"({_cast(c, 'b')} ? {_cast(a, t)} : {_cast(b, t)})", t, c.uni and a.uni and b.uni)
        if b.code in ("0.0", "INT64_C(0)", "0") and b.ty in ("wf", "wi"):
            out.where0 = (c, V(_cast(a, t), t, a.uni))
        return out

    def kw(self, node, names, defaults=None):
        """The arguments of a call by name."""
        out = dict(defaults or {})
        for n, a in zip(names, node.args):
            out[n] = a
        for k in node.keywords:
            out[k.arg] = k.value
        return out

    def mask_of(self, node) -> Optional[V]:
        if node is None:
            return None
        if self.is_const(node) and self.const(node) is None:
            return None
        return self.expr(node)

    def call(self, node) -> V:
        func = node.func
        # x.to(dtype)
        if isinstance(func, ast.Attribute) and func.attr == "to" and not self.is_const(func.value):
            v = self.expr(func.value)
            if any(k.arg == "bitcast" for k in node.keywords):
                raise Unsupported("bitcast")
            ty = self.dtype_of(node.args[0])
            return V(_cast(v, ty), ty, v.uni)
        try:
            f = self.const(func)
        except Exception as err:
            raise Unsupported(f"call {ast.dump(func)[:60]}") from err
        fn = _jit_fn(f)
        mod = getattr(fn or f, "__module__", "") or ""
        if fn is not None and not mod.startswith("triton."):
            return self.inline(fn, node)
        name = getattr(fn or f, "__name__", "")
        if name == "<lambda>" and isinstance(func, ast.Attribute):
            # Triton's interpreter, once it ran a kernel in this process, replaces triton.language's functions
            # with lambdas: the call's own name says which one it was
            name = func.attr
        if not mod.startswith("triton"):
            raise Unsupported(f"call {name}")
        a = node.args
        if name == "load":
            k = self.kw(node, ["pointer", "mask", "other"])
            p = self.expr(k["pointer"])
            if not p.ptr:
                raise Unsupported("load of a non-pointer")
            m = self.mask_of(k.get("mask"))
            if not (p.uni and (m is None or m.uni)):
                self.tr.loaded.add(p.ty[2])
            elem = p.ty[1]
            if m is None:
                return V(f"(*{p.code})", elem, p.uni)
            o = self.expr(k["other"]) if k.get("other") is not None and not (
                self.is_const(k["other"]) and self.const(k["other"]) is None) else _lit(0)
            return V(f"({_cast(m, 'b')} ? *{p.code} : {_cast(o, elem)})", elem, p.uni and m.uni and o.uni)
        if name == "store":
            k = self.kw(node, ["pointer", "value", "mask"])
            p = self.expr(k["pointer"])
            v = self.expr(k["value"])
            m = self.mask_of(k.get("mask"))
            self.tr.stored.add(p.ty[2])
            if v.sumof is not None:
                self.reduce_store(p, v.sumof, m)
                return None
            guard = []
            if m is not None:
                guard.append(_cast(m, "b"))
            if p.uni and v.uni:
                guard.append("j == 0")
            self.tr.effects.append(frozenset() if m is None else m.conj)
            stmt = f"*{p.code} = {_cast(v, p.ty[1])};"
            self.tr.emit(f"if ({' && '.join(guard)}) {stmt}" if guard else stmt)
            return None
        if name == "atomic_add":
            k = self.kw(node, ["pointer", "val", "mask", "sem", "scope"])
            p = self.expr(k["pointer"])
            v = self.expr(k["val"])
            m = self.mask_of(k.get("mask"))
            base, elem = p.ty[2], p.ty[1]
            self.tr.targets[base] = elem
            guard = [] if m is None else [_cast(m, "b")]
            if v.sumof is not None:
                v = v.sumof
            elif p.uni and v.uni:
                guard.append("j == 0")
            conj = frozenset() if m is None else m.conj
            if v.where0 is not None:
                c, v = v.where0
                guard.append(_cast(c, "b"))
                conj = conj | c.conj
            self.tr.effects.append(conj)
            tr = self.tr
            uses = set(_IDENT.findall(p.code)) & tr.known
            if p.uni and not v.uni and "j == 0" not in guard and uses <= tr.hoisted:
                # one address for the whole program: the lanes add into a register, the buffer once after them
                acc = tr.fresh("acc")
                tr.prologue.append(f"      {_C[elem]} {acc} = 0;")
                tr.emit(f"if ({' && '.join(guard)}) {acc} += {_cast(v, elem)};" if guard
                        else f"{acc} += {_cast(v, elem)};")
                off = tr.fresh("o")
                tr.epilogue.append(f"      {{ const int64_t {off} = {p.code} - {base}; buf_{base}[{off}] += {acc}; "
                                   f"mk_{base}[{off} >> {_BLOCK_SHIFT}] = 1; }}")
                return None
            off = self.tr.fresh("o")
            body = (f"{{ const int64_t {off} = {p.code} - {base}; buf_{base}[{off}] += {_cast(v, elem)}; "
                    f"mk_{base}[{off} >> {_BLOCK_SHIFT}] = 1; }}")
            self.tr.emit(f"if ({' && '.join(guard)}) {body}" if guard else body)
            return None
        if name == "where":
            return self.where(self.expr(a[0]), self.expr(a[1]), self.expr(a[2]))
        if name in ("maximum", "minimum"):
            x, y = self.expr(a[0]), self.expr(a[1])
            t = _promote(x.ty, y.ty)
            t = _strong(t) if t in ("wf", "wi") else t
            op = "std::max" if name == "maximum" else "std::min"
            return V(f"{op}({_cast(x, t)}, {_cast(y, t)})", t, x.uni and y.uni)
        if name in _MATH:
            xs = [self.expr(x) for x in a]
            t = xs[0].ty
            for x in xs[1:]:
                t = _promote(t, x.ty)
            t = "f32" if t in ("wf", "wi") or t in _INTS else t
            if name == "abs" and xs[0].ty in _INTS:
                t = xs[0].ty
            args = ", ".join(_cast(x, t) for x in xs)
            uni = all(x.uni for x in xs)
            if name == "rsqrt":
                return V(f"({_C[t]}(1) / std::sqrt({args}))", t, uni)
            if name == "sigmoid":
                return V(f"({_C[t]}(1) / ({_C[t]}(1) + std::exp(-{args})))", t, uni)
            return V(f"{_MATH[name]}({args})", t, uni)
        if name == "arange":
            start = self.const(a[0])
            return V(f"(j + {start})" if start else "j", "i32", False)
        if name == "program_id":
            if self.const(a[0]) != 0:
                raise Unsupported("a grid axis other than 0")
            return V("pid", "i32", True)
        if name in ("zeros", "full"):
            k = self.kw(node, ["shape", "value", "dtype"] if name == "full" else ["shape", "dtype"])
            ty = self.dtype_of(k["dtype"])
            val = self.expr(k["value"]) if name == "full" else _lit(0)
            return V(_cast(val, ty), ty, val.uni)
        if name in ("sum", "max", "min", "cumsum") and self.tr.block == 1:
            # a block of one lane: its reductions and scans are the lane's value
            return self.expr(a[0])
        if name == "sum":
            v = self.expr(a[0])
            return V(None, v.ty, True, sumof=v)
        if name == "cdiv":
            x, y = self.expr(a[0]), self.expr(a[1])
            t = _promote(x.ty, y.ty)
            return V(f"(({_cast(x, t)} + {_cast(y, t)} - 1) / {_cast(y, t)})", t, x.uni and y.uni)
        if name in ("static_assert", "device_assert", "debug_barrier", "multiple_of", "max_contiguous"):
            return self.expr(a[0]) if name in ("multiple_of", "max_contiguous") else None
        raise Unsupported(f"tl.{name}")

    def inline(self, fn, node) -> V:
        fdef = _function_def(fn)
        params = [p.arg for p in fdef.args.args]
        defaults = dict(zip(params[len(params) - len(fdef.args.defaults):], fdef.args.defaults))
        given = self.kw(node, params)
        env = {}
        for p in params:
            src = given.get(p)
            if src is None:
                if p not in defaults:
                    raise Unsupported(f"{fn.__name__}: argument {p}")
                env[p] = _Const(_Scope(self.tr, fn.__globals__, {}).const(defaults[p]))
                continue
            if self.is_const(src):
                c = self.const(src)
                env[p] = _Const(c)
                continue
            v = self.expr(src)
            env[p] = v
        scope = _Scope(self.tr, fn.__globals__, env)
        scope.body(fdef.body)
        return scope.returned

    # statements

    def bind(self, name: str, v):
        if name == "_":
            return
        if v is None:
            raise Unsupported(f"{name} = a call without a value")
        if isinstance(v, tuple):
            raise Unsupported(f"{name} = a tuple")
        keep = v.sumof is not None or (_simple(v.code) and v.code not in self.tr.mutable)
        self.env[name] = v if keep else self.tr.declare(name, v)

    def assign(self, targets, value):
        if isinstance(targets, ast.Name):
            if isinstance(value, ast.AST):
                try:
                    self.env[targets.id] = _Const(self.const(value))
                    return
                except Exception:
                    pass
                value = self.expr(value)
            self.bind(targets.id, value)
            return
        if isinstance(targets, ast.Tuple):
            if isinstance(value, ast.Tuple):
                vals = [self.expr(v) for v in value.elts]
            else:
                vals = self.expr(value) if isinstance(value, ast.AST) else value
                if not isinstance(vals, tuple):
                    raise Unsupported("unpacking a non-tuple")
            if len(vals) != len(targets.elts):
                raise Unsupported("unpacking length")
            for t, v in zip(targets.elts, vals):
                self.assign(t, v)
            return
        raise Unsupported(f"assignment to {type(targets).__name__}")

    def body(self, stmts):
        for st in stmts:
            if self.returned is not None:
                raise Unsupported("statements after a return")
            self.stmt(st)

    def stmt(self, st):
        if isinstance(st, ast.Expr):
            if isinstance(st.value, ast.Constant):
                return
            if isinstance(st.value, ast.Call):
                self.call(st.value)
                return
            raise Unsupported("expression statement")
        if isinstance(st, ast.Assign):
            if len(st.targets) != 1:
                raise Unsupported("chained assignment")
            if isinstance(st.targets[0], ast.Tuple) and isinstance(st.value, ast.Call):
                self.assign(st.targets[0], self.call(st.value))
            else:
                self.assign(st.targets[0], st.value)
            return
        if isinstance(st, ast.AnnAssign):
            self.assign(st.target, st.value)
            return
        if isinstance(st, ast.AugAssign):
            cur = self.expr(ast.Name(id=st.target.id, ctx=ast.Load()))
            self.bind(st.target.id, self.binop(cur, st.op, self.expr(st.value)))
            return
        if isinstance(st, ast.Return):
            if st.value is None and self.kernel and self.tr.indent > 4 and not self.tr.loops:
                # the kernel's own return inside a runtime branch: the lane is done
                self.tr.emit("continue;")
                self.tr.lane_return = True
                return
            if st.value is None:
                self.returned = ()
            elif isinstance(st.value, ast.Tuple):
                self.returned = tuple(self.expr(e) for e in st.value.elts)
            else:
                self.returned = self.expr(st.value)
            return
        if isinstance(st, ast.If):
            if self.is_const(st.test):
                self.body(st.body if self.const(st.test) else st.orelse)
                return
            c = self.expr(st.test)
            self.runtime_block(lambda: self.tr.emit(f"if ({_cast(c, 'b')}) {{"), [st.body, st.orelse], c.uni)
            return
        if isinstance(st, ast.For):
            self.loop_for(st)
            return
        if isinstance(st, ast.While):
            self.loop_while(st)
            return
        if isinstance(st, (ast.Pass, ast.Assert)):
            return
        raise Unsupported(f"statement {type(st).__name__}")

    def carried(self, stmts, uni: bool) -> dict:
        """Mutable C++ variables for the names a runtime block reassigns (their types fixed)."""
        out = {}
        for n in sorted(_assigned(stmts)):
            cur = self.env.get(n)
            if isinstance(cur, _Const) and isinstance(cur.value, (bool, int, float)):
                # a Python number reassigned at run time: a variable of the widest type of its kind
                cur = _lit(cur.value)
                cur.ty = {"wf": "f64", "wi": "i64"}.get(cur.ty, cur.ty)
            if isinstance(cur, V) and not cur.ptr and cur.sumof is None:
                ty = _strong(cur.ty) if cur.ty in ("wf", "wi") else cur.ty
                var = self.tr.fresh(n)
                self.tr.emit(f"{_C[ty]} {var} = {_cast(cur, ty)};")
                self.tr.mutable.add(var)
                out[n] = V(var, ty, uni and cur.uni)
        return out

    def runtime_block(self, opener, branches, uni: bool):
        stmts = [s for b in branches for s in b]
        carried = self.carried(stmts, uni)
        before = dict(self.env)
        self.env.update(carried)
        extra = {}
        for j, branch in enumerate(branches):
            if j == 0:
                extra = opener() or {}
            elif branch:
                self.tr.emit("} else {")
                extra = {}
            else:
                continue
            self.tr.indent += 1
            self.env = dict(before, **carried, **extra)
            self.sub_body(branch, carried)
            self.tr.indent -= 1
        self.tr.emit("}")
        self.env = dict(before, **carried)

    def sub_body(self, stmts, carried):
        """A branch or loop body: values bound to a carried name are written into its variable."""
        for st in stmts:
            self.stmt(st)
            for n, var in carried.items():
                cur = self.env.get(n)
                if cur is not var:
                    if not isinstance(cur, V) or cur.ptr:
                        raise Unsupported(f"{n} changes kind inside a runtime block")
                    self.tr.emit(f"{var.code} = {_cast(cur, var.ty)};")
                    if not cur.uni:
                        var.uni = False
                    self.env[n] = var
        if self.returned is not None:
            raise Unsupported("a return inside a runtime block")

    def loop_for(self, st):
        it = st.iter
        if not (isinstance(it, ast.Call) and isinstance(st.target, ast.Name)) or st.orelse:
            raise Unsupported("for loop")
        f = self.const(it.func)
        name = getattr(f, "__name__", "")
        if isinstance(it.func, ast.Attribute) and name not in ("static_range", "range"):
            # tl.static_range as Triton's interpreter replaces it (see call)
            name = it.func.attr
        if name not in ("static_range", "range"):
            raise Unsupported("for loop over another iterator")
        args = list(it.args)
        if len(args) == 1:
            args = [ast.Constant(0)] + args
        if len(args) == 2:
            args.append(ast.Constant(1))
        if all(self.is_const(a) for a in args):
            lo, hi, step = (self.const(a) for a in args)
            if len(range(lo, hi, step)) <= _UNROLL:
                for j in range(lo, hi, step):
                    self.env[st.target.id] = _Const(j)
                    self.body(st.body)
                return
        lo, hi, step = (self.expr(a) for a in args)
        uni = lo.uni and hi.uni and step.uni
        var = self.tr.fresh(st.target.id)
        t = _promote(_promote(lo.ty, hi.ty), step.ty)
        t = "i64" if t in ("wi", "wf") else t

        def opener():
            self.tr.emit(f"for ({_C[t]} {var} = {_cast(lo, t)}; {var} < {_cast(hi, t)}; {var} += {_cast(step, t)}) {{")
            return {st.target.id: V(var, t, uni)}

        tr = self.tr
        n0, line0, depth = len(tr.effects), len(tr.lines), tr.indent + 1
        carries = any(isinstance(self.env.get(n), (V, _Const)) for n in _assigned(st.body) - {st.target.id})
        tr.loops += 1
        self.runtime_block(opener, [st.body], uni)
        tr.loops -= 1
        # a mask every store and add of the body needs (and no value carried from one iteration to the next):
        # an iteration where it is false goes on to the next past its declaration
        effects = tr.effects[n0:]
        if effects and not carries:
            common = frozenset.intersection(*effects)
            found = [v for v in common if v in tr.at_any and tr.at_any[v][1] == depth and tr.at_any[v][0] > line0]
            if found:
                g = max(found, key=lambda v: tr.at_any[v][0])
                tr.insert(tr.at_any[g][0], "  " * depth + f"if (!{g}) continue;")

    def loop_while(self, st):
        if st.orelse:
            raise Unsupported("while-else")
        carried = self.carried(st.body, True)
        before = dict(self.env)
        self.env.update(carried)
        self.tr.emit("while (true) {")
        self.tr.indent += 1
        c = self.expr(st.test)
        self.tr.emit(f"if (!{_cast(c, 'b')}) break;")
        self.tr.loops += 1
        self.sub_body(st.body, carried)
        self.tr.loops -= 1
        self.tr.indent -= 1
        self.tr.emit("}")
        if not c.uni:
            for v in carried.values():
                v.uni = False
        self.env = dict(before, **carried)


def _simple(code: str) -> bool:
    """A C++ expression cheap to repeat (a name or a literal)."""
    return code.replace("_", "").replace(".", "").replace("-", "").replace("e", "").isalnum() or \
        code.startswith("INT64_C(")


def _arg_types(fdef, args, meta):
    """The kernel's runtime arguments in order: (name, kind, type), kind ``tensor`` / ``int`` / ``float``."""
    out = []
    for p, a in zip([p.arg for p in fdef.args.args], args):
        if isinstance(a, torch.Tensor):
            if a.dtype not in _TORCH:
                raise Unsupported(f"tensor dtype {a.dtype}")
            out.append((p, "tensor", _TORCH[a.dtype]))
        elif isinstance(a, bool):
            out.append((p, "int", "b"))
        elif isinstance(a, int):
            out.append((p, "int", "i64"))
        elif isinstance(a, float):
            out.append((p, "float", "f32"))
        else:
            raise Unsupported(f"argument {p} of type {type(a).__name__}")
    return out


def translate(kernel, args: tuple, meta: dict, balanced: bool = False) -> tuple:
    """The C++ source of a kernel for these arguments and constexpr values, and the names of its tensor
    arguments that receive atomic adds; ``balanced`` the programs split by a work prefix passed after the
    tensor arguments (:func:`prepare`)."""
    fn = _jit_fn(kernel)
    if fn is None:
        raise Unsupported("not a triton.jit kernel")
    fdef = _function_def(fn)
    params = [p.arg for p in fdef.args.args]
    n_runtime = len(args)
    runtime = _arg_types(fdef, args, meta)
    env = {}
    defaults = dict(zip(params[len(params) - len(fdef.args.defaults):], fdef.args.defaults))
    for name, kind, ty in runtime:
        if kind == "tensor":
            env[name] = V(name, ("ptr", ty, name), True)
        else:
            env[name] = V(name, ty, True)
    block = None
    for p in params[n_runtime:]:
        if p in meta:
            env[p] = _Const(meta[p])
        elif p in defaults:
            env[p] = _Const(_Scope(None, fn.__globals__, {}).const(defaults[p]))
        else:
            raise Unsupported(f"missing constexpr {p}")
        if p == "BLOCK" or p.startswith("BLOCK"):
            block = block or env[p].value
    if "BLOCK" in env and isinstance(env["BLOCK"], _Const):
        block = env["BLOCK"].value
    if block is None:
        raise Unsupported("no BLOCK constexpr")
    tr = _Translator()
    tr.block = int(block)
    for name, _, _ in runtime:
        tr.known.add(name)
        tr.hoisted.add(name)
    scope = _Scope(tr, fn.__globals__, env)
    scope.kernel = True
    scope.body(fdef.body)
    # a returned lane adds nothing to a block sum, as a returned program on a GPU; it would still store one
    if tr.lane_return and tr.sum_stores:
        raise Unsupported("a return before the store of a block sum")
    # a mask every store and add of a lane needs: past its declaration, a lane where it is false stops
    if tr.effects:
        common = frozenset.intersection(*tr.effects)
        lanes = [v for v in common if v in tr.at]
        if lanes:
            g = max(lanes, key=lambda v: tr.at[v])
            tr.insert(tr.at[g], "  " * 4 + f"if (!{g}) continue;")
        elif any(v in tr.hoisted for v in common):
            g = next(v for v in common if v in tr.hoisted)
            tr.insert(0, "  " * 4 + f"if (!{g}) continue;")
    # loads at addresses uniform across a program may read a tensor that receives adds: they see its values from
    # before the launch (the adds land after the lanes ran), as a kernel reading entries no lane adds to expects
    both = set(tr.targets) & (tr.loaded | tr.stored)
    if both:
        raise Unsupported(f"atomic adds into {sorted(both)}, also loaded or stored")
    targets = sorted(tr.targets)
    sig = "(void** tp, const double* fp, const int64_t* ip, const int64_t* np, int64_t n_programs, int n_threads)"
    lines = ["#include <cstdint>", "#include <cmath>", "#include <limits>", "#include <algorithm>",
             "#include <vector>", "#include <omp.h>", ""]
    ti = 0
    index = {}
    for name, kind, _ in runtime:
        if kind == "tensor":
            index[name] = ti
            ti += 1
    # the threads' buffers of the atomic adds, kept between launches: each thread's buffer and marks on cache
    # lines of their own (adds into a few entries, the energy totals, would otherwise share a line between all
    # threads), allocated by setup and zeroed by the threads that use them (their pages on those threads' memory)
    sizes = []
    for name in targets:
        ty = _C[tr.targets[name]]
        lines += [f"static {ty}* all_{name} = nullptr;", f"static uint8_t* allmk_{name} = nullptr;",
                  f"static int64_t size_{name} = -1;", f"static int64_t msize_{name} = -1;",
                  f"static bool fresh_{name} = false;"]
        sizes += [f"  const int64_t n_{name} = np[{index[name]}];",
                  f"  const int64_t nb_{name} = (n_{name} + {(1 << _BLOCK_SHIFT) - 1}) >> {_BLOCK_SHIFT};",
                  f"  const int64_t sn_{name} = ((n_{name} + 15) / 16) * 16 + 16;",
                  f"  const int64_t smk_{name} = ((nb_{name} + 63) / 64) * 64 + 64;"]
    lines += ["", f'extern "C" void setup{sig} {{', "  const int T = n_threads;", "  (void)T;"] + sizes
    for name in targets:
        ty = _C[tr.targets[name]]
        # the buffers only grow: a launch leaves them zero, so a smaller layout reuses them (a kernel called on
        # tensors of alternating sizes would otherwise free and fault in its buffers on every call)
        lines += [f"  fresh_{name} = false;",
                  f"  if (size_{name} < (int64_t)T * sn_{name}) {{",
                  f"    delete[] all_{name};",
                  f"    all_{name} = new {ty}[(size_t)T * sn_{name}];",
                  f"    size_{name} = (int64_t)T * sn_{name}; fresh_{name} = true;",
                  "  }",
                  f"  if (msize_{name} < (int64_t)T * smk_{name}) {{",
                  f"    delete[] allmk_{name};",
                  f"    allmk_{name} = new uint8_t[(size_t)T * smk_{name}];",
                  f"    msize_{name} = (int64_t)T * smk_{name}; fresh_{name} = true;",
                  "  }"]
    lines.append("}")
    # body: one thread's share inside a parallel region of n_threads threads, its adds summed into the tensors
    # (a barrier, then the thread's share of the touched blocks); body_deferred: the adds left in the buffers bp,
    # bm of the caller (one per target, in the sorted order of the targets), summed by it once for a sequence
    for variant in ("body", "body_deferred"):
        deferred = variant == "body_deferred"
        extra = ", void** bp, void** bm" if deferred else ""
        lines += ["", f'extern "C" void {variant}{sig[:-1]}{extra}) {{']
        ti = fi = ii = 0
        for name, kind, ty in runtime:
            if kind == "tensor":
                lines.append(f"  {_C[ty]}* __restrict__ {name} = static_cast<{_C[ty]}*>(tp[{ti}]);")
                ti += 1
            elif kind == "int":
                lines.append(f"  const {_C[ty]} {name} = static_cast<{_C[ty]}>(ip[{ii}]);")
                ii += 1
            else:
                lines.append(f"  const float {name} = static_cast<float>(fp[{fi}]);")
                fi += 1
        lines += ["  const int T = n_threads;", "  (void)T;"] + sizes
        lines += ["  {", "    const int tid = omp_get_thread_num();"]
        for j, name in enumerate(targets):
            ty = _C[tr.targets[name]]
            if deferred:
                lines += [f"    {ty}* __restrict__ buf_{name} = static_cast<{ty}*>(bp[{j}]) + (size_t)tid * sn_{name};",
                          f"    uint8_t* __restrict__ mk_{name} = static_cast<uint8_t*>(bm[{j}])"
                          f" + (size_t)tid * smk_{name};"]
            else:
                lines += [f"    {ty}* __restrict__ buf_{name} = all_{name} + (size_t)tid * sn_{name};",
                          f"    uint8_t* __restrict__ mk_{name} = allmk_{name} + (size_t)tid * smk_{name};",
                          f"    if (fresh_{name}) {{ std::fill(buf_{name}, buf_{name} + sn_{name}, ({ty})0); "
                          f"std::fill(mk_{name}, mk_{name} + smk_{name}, (uint8_t)0); }}"]
        # the programs in chunks taken by the threads as they finish (programs of uneven work, rows of a half
        # list, stay balanced; a chunk keeps neighboring programs on one thread)
        if balanced:
            # contiguous programs of an equal share of the work (a prefix of n_programs + 1 entries)
            lines += [f"    const int64_t* work = static_cast<const int64_t*>(tp[{len(index)}]);",
                      "    const int64_t W = work[n_programs];",
                      "    const int64_t p0 = tid == 0 ? 0"
                      " : std::lower_bound(work, work + n_programs, W * tid / T) - work;",
                      "    const int64_t p1 = tid == T - 1 ? n_programs"
                      " : std::lower_bound(work, work + n_programs, W * (tid + 1) / T) - work;",
                      "    for (int64_t p = p0; p < p1; ++p) {"]
        else:
            lines += ["    const int64_t chunk = std::max<int64_t>(1, n_programs / (16 * (int64_t)T));",
                      "    #pragma omp for schedule(dynamic, chunk) nowait",
                      "    for (int64_t p = 0; p < n_programs; ++p) {"]
        lines += ["      const int32_t pid = static_cast<int32_t>(p);", "      (void)pid;"]
        lines += tr.prologue
        lines += [f"      for (int32_t j = 0; j < {int(block)}; ++j) {{"]
        lines += tr.lines
        lines += ["      }"] + tr.epilogue + ["    }"]
        if targets and not deferred:
            lines.append("    #pragma omp barrier")
        for name in targets if not deferred else []:
            lines += [f"    for (int64_t b = nb_{name} * tid / T; b < nb_{name} * (tid + 1) / T; ++b) {{",
                      f"      const int64_t e0 = b << {_BLOCK_SHIFT};",
                      f"      const int64_t e1 = std::min(n_{name}, (b + 1) << {_BLOCK_SHIFT});",
                      "      for (int u = 0; u < T; ++u) {",
                      f"        uint8_t& m = allmk_{name}[(size_t)u * smk_{name} + b];",
                      "        if (!m) continue;",
                      f"        {_C[tr.targets[name]]}* s = all_{name} + (size_t)u * sn_{name};",
                      f"        for (int64_t e = e0; e < e1; ++e) {{ {name}[e] += s[e]; s[e] = 0; }}",
                      "        m = 0;", "      }", "    }"]
        lines += ["  }", "}"]
    lines += ["", f'extern "C" void run{sig} {{',
              "  setup(tp, fp, ip, np, n_programs, n_threads);",
              "  #pragma omp parallel num_threads(n_threads)",
              "  body(tp, fp, ip, np, n_programs, n_threads);", "}", ""]
    return "\n".join(lines), targets


_LOCK = threading.Lock()
_LIBS: dict = {}
_SIGS: dict = {}
_TARGETS: dict = {}


@functools.lru_cache(maxsize=None)
def _openmp_library() -> Optional[str]:
    """The OpenMP runtime torch loaded (linking against it keeps one thread pool in the process). Another
    runtime may be mapped too (a cluster module's ``libgomp.so.1`` on the library path): torch's own, in its
    package directory, comes first."""
    try:
        maps = Path("/proc/self/maps").read_text().split("\n")
    except OSError:
        return None
    found = []
    for ln in maps:
        path = ln.split(" ")[-1]
        name = os.path.basename(path)
        if name.startswith(("libgomp", "libomp", "libiomp5")) and ".so" in name and path not in found:
            found.append(path)
    if not found:
        return None
    root = os.path.dirname(os.path.realpath(torch.__file__))
    own = [p for p in found if os.path.realpath(p).startswith(root + os.sep)]
    return (own or found)[0]


@functools.lru_cache(maxsize=None)
def _cpu_tag() -> str:
    model = platform.processor()
    try:
        for ln in Path("/proc/cpuinfo").read_text().split("\n"):
            if ln.startswith(("model name", "flags")):
                model += ln
                if ln.startswith("flags"):
                    break
    except OSError:
        pass
    return hashlib.sha1(model.encode()).hexdigest()[:8]


def _cache_dir() -> Path:
    root = os.environ.get("XNN_CPU_KERNELS_DIR")
    if root:
        return Path(root)
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "xnn" / "cpu_kernels"


def _compiler() -> Optional[str]:
    for c in (os.environ.get("CXX"), "g++", "c++", "clang++"):
        if c and any(os.access(os.path.join(d, c), os.X_OK) for d in os.environ.get("PATH", "").split(os.pathsep)):
            return c
    return None


@functools.lru_cache(maxsize=None)
def available() -> bool:
    """Whether CPU kernels can be compiled here (Linux, a C++ compiler on the path, Triton importable); checked
    once per process (every launch asks)."""
    if platform.system() != "Linux" or _compiler() is None:
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def _build(src: str, symbol: str = "run") -> ctypes.CDLL:
    if src in _BY_SOURCE:
        return _BY_SOURCE[src]
    lib = _compile(src, symbol)
    _BY_SOURCE[src] = lib
    return lib


_BY_SOURCE: dict = {}


def _compile(src: str, symbol: str) -> ctypes.CDLL:
    omp = _openmp_library()
    key = hashlib.sha1((src + (omp or "") + _cpu_tag()).encode()).hexdigest()[:20]
    if key in _LIBS:
        return _LIBS[key]
    out = _cache_dir()
    out.mkdir(parents=True, exist_ok=True)
    so = out / f"k_{key}.so"
    if not so.exists():
        with tempfile.TemporaryDirectory(dir=out) as tmp:
            cpp, obj, lib = Path(tmp) / "k.cpp", Path(tmp) / "k.o", Path(tmp) / "k.so"
            cpp.write_text(src)
            cxx = _compiler()
            # linked against torch's OpenMP runtime where it has one: one thread pool in the process
            # no errno from the math functions (a square root is one instruction); the results are unchanged
            # with line information (no cost at run time) for a profiler, the source kept beside the library
            flags = ["-O3", "-g", "-march=native", "-fno-math-errno", "-fPIC", "-fopenmp", "-std=c++17"]
            steps = [[cxx, *flags, "-c", str(cpp), "-o", str(obj)],
                     [cxx, "-shared", str(obj), "-o", str(lib)]
                     + ([omp, f"-Wl,-rpath,{os.path.dirname(omp)}"] if omp else ["-fopenmp"])]
            for cmd in steps:
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode != 0:
                    raise RuntimeError(f"compiling a CPU kernel failed:\n{res.stderr[-4000:]}")
            os.replace(lib, so)
            os.replace(cpp, so.with_suffix(".cpp"))
    lib = ctypes.CDLL(str(so))
    if symbol == "run":
        lib.run.restype = None
        lib.run.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_double),
                            ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64,
                            ctypes.c_int]
    else:
        getattr(lib, symbol).restype = None
    _LIBS[key] = lib
    return lib


def _signature(kernel, args, meta) -> tuple:
    sig = []
    for a in args:
        if isinstance(a, torch.Tensor):
            sig.append(("t", a.dtype))
        elif isinstance(a, bool):
            sig.append(("b",))
        elif isinstance(a, int):
            sig.append(("i",))
        elif isinstance(a, float):
            sig.append(("f",))
        else:
            sig.append(("?", type(a).__name__))
    consts = tuple(sorted((k, str(v)) for k, v in meta.items() if k not in ("num_warps", "num_stages")))
    return (id(kernel), tuple(sig), consts)


class Launch:
    """A kernel launch with its arguments packed once: calling it runs the kernel on the tensors' current
    contents (they must stay allocated and in place)."""

    def __init__(self, lib, tensors, fs, ints, n, targets=()):
        self.lib, self.tensors, self.n = lib, tensors, int(n)
        # the tensor arguments (indices) that receive atomic adds, in the order of the kernel's buffers
        self.targets = list(targets)
        k = max(len(tensors), 1)
        self.tp = (ctypes.c_void_p * k)(*[t.data_ptr() for t in tensors])
        self.np = (ctypes.c_int64 * k)(*[t.numel() for t in tensors])
        self.fp = (ctypes.c_double * max(len(fs), 1))(*fs)
        self.ip = (ctypes.c_int64 * max(len(ints), 1))(*ints)

    def bind(self, j: int, tensor: torch.Tensor):
        """Point the ``j``-th tensor argument at ``tensor`` (contiguous, of the same dtype and size)."""
        old = self.tensors[j]
        if tensor.dtype != old.dtype or tensor.numel() != old.numel() or not tensor.is_contiguous() \
                or tensor.device.type != "cpu":
            raise ValueError("a rebound argument keeps its dtype and size")
        self.tensors[j] = tensor
        self.tp[j] = tensor.data_ptr()

    def __call__(self):
        if self.n > 0:
            self.lib.run(self.tp, self.fp, self.ip, self.np, self.n, torch.get_num_threads())


def prepare(kernel, grid, *args, balance: Optional[torch.Tensor] = None, **meta) -> Launch:
    """The launch ``kernel[grid](*args, **meta)`` on CPU tensors, compiled and packed, to be called once or
    many times (``grid`` an int or a 1-tuple; ``num_warps`` and ``num_stages`` are ignored). ``balance`` an
    int64 prefix ``(n_programs + 1,)`` of the programs' work (a compressed-row pointer): each thread takes
    contiguous programs of an equal share of it, instead of chunks taken as threads finish."""
    n = grid[0] if isinstance(grid, (tuple, list)) else grid
    if isinstance(grid, (tuple, list)) and len(grid) > 1 and any(g != 1 for g in grid[1:]):
        raise Unsupported("a grid of more than one axis")
    meta = {k: v for k, v in meta.items() if k not in ("num_warps", "num_stages")}
    # runtime arguments given by name (the generated kernels' tangent and gradient pointers) in their place
    fdef = _function_def(_jit_fn(kernel))
    names = [a.arg for a in fdef.args.args]
    is_const = [a.annotation is not None and _dotted_name(a.annotation).endswith("constexpr") for a in fdef.args.args]
    args = list(args)
    for name, c in zip(names[len(args):], is_const[len(args):]):
        if c:
            break
        if name not in meta:
            raise TypeError(f"{fdef.name}: missing argument {name}")
        args.append(meta.pop(name))
    args = tuple(args)
    sig = _signature(kernel, args, meta) + (balance is not None,)
    with _LOCK:
        lib = _SIGS.get(sig)
        if lib is None:
            src, targets = translate(kernel, args, meta, balanced=balance is not None)
            lib = _build(src)
            fdef = _function_def(_jit_fn(kernel))
            names = [n for n, kind, _ in _arg_types(fdef, args, meta) if kind == "tensor"]
            _TARGETS[sig] = [names.index(t) for t in targets]
            _SIGS[sig] = lib
    tensors = [a for a in args if isinstance(a, torch.Tensor)]
    targets = _TARGETS[sig]
    ptrs = [tensors[j].data_ptr() for j in targets]
    if len(set(ptrs)) < len(ptrs):
        raise ValueError("two tensors receiving atomic adds share their memory")
    for t in tensors:
        if t.device.type != "cpu" or not t.is_contiguous():
            raise ValueError("CPU kernels take contiguous CPU tensors")
    fs = [float(a) for a in args if isinstance(a, float)]
    ints = [int(a) for a in args if isinstance(a, int)]
    if balance is not None:
        if balance.dtype != torch.int64 or balance.numel() != int(n) + 1 or not balance.is_contiguous():
            raise ValueError("balance is a contiguous int64 prefix of n_programs + 1 entries")
        tensors = tensors + [balance]
    return Launch(lib, tensors, fs, ints, n, targets)


_DRIVER = r"""
#include <cstdint>
#include <omp.h>
typedef void (*kernel_t)(void**, const double*, const int64_t*, const int64_t*, int64_t, int);
// the launches of a sequence in one parallel region: each kernel's share per thread, a barrier after the kernels
// whose outputs the next one reads (wait[s])
extern "C" void run_sequence(int n, void** setups, void** bodies, void*** tps, const double** fps,
                             const int64_t** ips, const int64_t** nps, const int64_t* programs, const uint8_t* wait,
                             int n_threads) {
  for (int s = 0; s < n; ++s)
    ((kernel_t)setups[s])(tps[s], fps[s], ips[s], nps[s], programs[s], n_threads);
  #pragma omp parallel num_threads(n_threads)
  {
    for (int s = 0; s < n; ++s) {
      ((kernel_t)bodies[s])(tps[s], fps[s], ips[s], nps[s], programs[s], n_threads);
      if (wait[s]) {
        #pragma omp barrier
      }
    }
  }
}
"""


_DEFERRED = r"""
#include <cstdint>
#include <algorithm>
#include <omp.h>
typedef void (*deferred_t)(void**, const double*, const int64_t*, const int64_t*, int64_t, int, void**, void**);
typedef void (*kernel_t)(void**, const double*, const int64_t*, const int64_t*, int64_t, int);
// kernels adding into shared per-thread buffers with no barrier between them, one barrier, each target's touched
// blocks summed into it once (a thread's share of the blocks), then the kernels that read the sums
extern "C" void run_deferred(int n, void** bodies, void*** tps, const double** fps, const int64_t** ips,
                             const int64_t** nps, const int64_t* programs, void*** bps, void*** bms, int nt,
                             double** tensors, double** bufs, uint8_t** marks, const int64_t* sizes, int na,
                             void** after, void*** atps, const double** afps, const int64_t** aips,
                             const int64_t** anps, const int64_t* aprograms, int n_threads) {
  const int T = n_threads;
  #pragma omp parallel num_threads(T)
  {
    const int tid = omp_get_thread_num();
    for (int s = 0; s < n; ++s)
      ((deferred_t)bodies[s])(tps[s], fps[s], ips[s], nps[s], programs[s], T, bps[s], bms[s]);
    #pragma omp barrier
    for (int t = 0; t < nt; ++t) {
      const int64_t N = sizes[t], nb = (N + 255) >> 8;
      const int64_t sn = ((N + 15) / 16) * 16 + 16, smk = ((nb + 63) / 64) * 64 + 64;
      for (int64_t b = nb * tid / T; b < nb * (tid + 1) / T; ++b) {
        const int64_t e0 = b << 8, e1 = std::min(N, (b + 1) << 8);
        for (int u = 0; u < T; ++u) {
          uint8_t& m = marks[t][(size_t)u * smk + b];
          if (!m) continue;
          double* src = bufs[t] + (size_t)u * sn;
          for (int64_t e = e0; e < e1; ++e) { tensors[t][e] += src[e]; src[e] = 0; }
          m = 0;
        }
      }
    }
    for (int s = 0; s < na; ++s) {
      #pragma omp barrier
      ((kernel_t)after[s])(atps[s], afps[s], aips[s], anps[s], aprograms[s], T);
    }
  }
}
"""


_POOL = threading.local()


def _zeroed(n: int, nm: int, slot: int) -> tuple:
    """Per-thread buffers of ``n`` doubles and ``nm`` marks, zeroed, kept for the next deferred launches of this
    Python thread (they leave them zeroed; two threads never share them, the launches run without the GIL)."""
    pool = getattr(_POOL, "buffers", None)
    if pool is None:
        pool = _POOL.buffers = {}
    key = (n, nm, slot)
    if key not in pool:
        pool[key] = (torch.zeros(n, dtype=torch.float64), torch.zeros(nm, dtype=torch.uint8))
    return pool[key]


class Deferred:
    """Prepared launches whose atomic adds go to per-thread buffers shared by the launches with the same target
    tensor (float64), in one parallel region with no barrier between them; each target is summed once after
    them, then the ``after`` launches (which may read the sums) run."""

    def __init__(self, launches: list, after: list = ()):
        self.launches = [run for run in launches if run.n > 0]
        self.after = [run for run in after if run.n > 0]
        self.lib = _build(_DEFERRED, symbol="run_deferred")
        self.threads = None
        self.prepared = False

    def _pack(self, T: int):
        P = ctypes.c_void_p
        shared, order = {}, []
        for run in self.launches:
            for j in run.targets:
                t = run.tensors[j]
                if t.dtype != torch.float64:
                    raise ValueError("deferred launches add into float64 tensors")
                if t.data_ptr() not in shared:
                    n = t.numel()
                    nb = (n + 255) >> 8
                    sn, smk = ((n + 15) // 16) * 16 + 16, ((nb + 63) // 64) * 64 + 64
                    shared[t.data_ptr()] = (t, *_zeroed(T * sn, T * smk, len(order)))
                    order.append(t.data_ptr())
        self.shared = [shared[k] for k in order]
        n = len(self.launches)
        self.bodies = (P * max(n, 1))(*[ctypes.cast(r.lib.body_deferred, P).value for r in self.launches])
        self.tps = (P * max(n, 1))(*[ctypes.cast(r.tp, P).value for r in self.launches])
        self.fps = (P * max(n, 1))(*[ctypes.cast(r.fp, P).value for r in self.launches])
        self.ips = (P * max(n, 1))(*[ctypes.cast(r.ip, P).value for r in self.launches])
        self.nps = (P * max(n, 1))(*[ctypes.cast(r.np, P).value for r in self.launches])
        self.programs = (ctypes.c_int64 * max(n, 1))(*[r.n for r in self.launches])
        self.keep = []
        bps, bms = [], []
        for run in self.launches:
            b = (P * max(len(run.targets), 1))(*[shared[run.tensors[j].data_ptr()][1].data_ptr() for j in run.targets])
            m = (P * max(len(run.targets), 1))(*[shared[run.tensors[j].data_ptr()][2].data_ptr() for j in run.targets])
            self.keep += [b, m]
            bps.append(ctypes.cast(b, P).value)
            bms.append(ctypes.cast(m, P).value)
        self.bps, self.bms = (P * max(n, 1))(*bps), (P * max(n, 1))(*bms)
        nt = len(self.shared)
        self.nt = nt
        self.tensors = (P * max(nt, 1))(*[x[0].data_ptr() for x in self.shared])
        self.bufs = (P * max(nt, 1))(*[x[1].data_ptr() for x in self.shared])
        self.marks = (P * max(nt, 1))(*[x[2].data_ptr() for x in self.shared])
        self.sizes = (ctypes.c_int64 * max(nt, 1))(*[x[0].numel() for x in self.shared])
        na = len(self.after)
        self.na = na
        self.abodies = (P * max(na, 1))(*[ctypes.cast(r.lib.body, P).value for r in self.after])
        self.atps = (P * max(na, 1))(*[ctypes.cast(r.tp, P).value for r in self.after])
        self.afps = (P * max(na, 1))(*[ctypes.cast(r.fp, P).value for r in self.after])
        self.aips = (P * max(na, 1))(*[ctypes.cast(r.ip, P).value for r in self.after])
        self.anps = (P * max(na, 1))(*[ctypes.cast(r.np, P).value for r in self.after])
        self.aprograms = (ctypes.c_int64 * max(na, 1))(*[r.n for r in self.after])
        self.threads = T

    def __call__(self):
        T = torch.get_num_threads()
        if T != self.threads:
            self._pack(T)
        for run in self.after:
            run.lib.setup(run.tp, run.fp, run.ip, run.np, ctypes.c_int64(run.n), T)
        self.lib.run_deferred(len(self.launches), self.bodies, self.tps, self.fps, self.ips, self.nps, self.programs,
                              self.bps, self.bms, self.nt, self.tensors, self.bufs, self.marks, self.sizes, self.na,
                              self.abodies, self.atps, self.afps, self.aips, self.anps, self.aprograms, T)


class Sequence:
    """Prepared launches run in order in one parallel region: no thread start-up per kernel. A barrier follows
    each launch, or, with ``independent``, only the last of a run of launches that read none of each other's
    outputs (their atomic adds go to their own buffers, and a thread sums the same blocks in each)."""

    def __init__(self, launches: list, independent: Optional[list] = None):
        keep = [j for j, run in enumerate(launches) if run.n > 0]
        self.launches = [launches[j] for j in keep]
        free = [bool(independent[j]) if independent is not None else False for j in keep]
        # a barrier after a launch unless it and the next are both independent
        wait = [not (free[j] and j + 1 < len(free) and free[j + 1]) for j in range(len(free))]
        n = len(self.launches)
        self.lib = _build(_DRIVER, symbol="run_sequence")
        P = ctypes.c_void_p

        def address(fn):
            return ctypes.cast(fn, P).value
        self.setups = (P * max(n, 1))(*[address(r.lib.setup) for r in self.launches])
        self.bodies = (P * max(n, 1))(*[address(r.lib.body) for r in self.launches])
        self.tps = (P * max(n, 1))(*[ctypes.cast(r.tp, P).value for r in self.launches])
        self.fps = (P * max(n, 1))(*[ctypes.cast(r.fp, P).value for r in self.launches])
        self.ips = (P * max(n, 1))(*[ctypes.cast(r.ip, P).value for r in self.launches])
        self.nps = (P * max(n, 1))(*[ctypes.cast(r.np, P).value for r in self.launches])
        self.programs = (ctypes.c_int64 * max(n, 1))(*[r.n for r in self.launches])
        self.wait = (ctypes.c_uint8 * max(n, 1))(*[int(w) for w in wait])

    def __call__(self):
        if self.launches:
            self.lib.run_sequence(len(self.launches), self.setups, self.bodies, self.tps, self.fps, self.ips,
                                  self.nps, self.programs, self.wait, torch.get_num_threads())


def launch(kernel, grid, *args, **meta):
    """Run ``kernel[grid](*args, **meta)`` on CPU tensors (see :func:`prepare`)."""
    prepare(kernel, grid, *args, **meta)()
