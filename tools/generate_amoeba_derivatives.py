#!/usr/bin/env python
"""Generate forward-mode derivatives (tangents) of the Triton device functions of the AMOEBA and HIPPO
kernels: ``xnn/ffnn/models/amoeba_tangents.py``.

The pair kernels of :mod:`xnn.ffnn.models.amoeba_kernels` and the ladders of
:mod:`xnn.ffnn.models.amoeba_ladders` are straight-line arithmetic on the registers of one block of pairs.
Their derivatives along a direction (a tangent) follow from the chain rule statement by statement, so this
script writes them by source transformation instead of by hand: every assignment ``a = e`` is followed by
``a_t = d(e)``, tangents that are identically zero are tracked symbolically and produce no code, and a call of
another device function becomes a call of its tangent twin, specialized to the arguments that carry
tangents. Inexact constants of the derivative rules come from the scope of the function (the derivative of
``erfc`` takes ``inv_sqrt_pi``), never from float32 literals.

    python tools/generate_amoeba_derivatives.py [--out src/xnn/ffnn/models/amoeba_tangents.py]
"""
import argparse
import ast
import copy
import textwrap
from pathlib import Path

ZERO = None                      # the symbolic zero tangent

# functions whose calls carry no derivative: their results are indices, masks, constants
_FLAT = {"program_id", "arange", "zeros", "full", "cdiv", "num_programs", "multiple_of", "max_contiguous"}


def _name(x: str) -> ast.Name:
    return ast.Name(id=x, ctx=ast.Load())


def _num(v: float) -> ast.Constant:
    return ast.Constant(value=v)


def _bin(a, op, b):
    return ast.BinOp(left=a, op=op, right=b)


def _add(a, b):
    if a is ZERO:
        return b
    if b is ZERO:
        return a
    return _bin(a, ast.Add(), b)


def _neg(x):
    if isinstance(x, ast.UnaryOp) and isinstance(x.op, ast.USub):
        return x.operand
    return ast.UnaryOp(op=ast.USub(), operand=x)


def _sub(a, b):
    if b is ZERO:
        return a
    if a is ZERO:
        return _neg(b)
    return _bin(a, ast.Sub(), b)


def _is_zero_literal(x) -> bool:
    return isinstance(x, ast.Constant) and x.value == 0.0


def _mul(a, b):
    if a is ZERO or b is ZERO or _is_zero_literal(a) or _is_zero_literal(b):
        return ZERO
    return _bin(a, ast.Mult(), b)


def _call(func: str, *args, **kw):
    parts = func.split(".")
    f = _name(parts[0])
    for p in parts[1:]:
        f = ast.Attribute(value=f, attr=p, ctx=ast.Load())
    return ast.Call(func=f, args=list(args), keywords=[ast.keyword(arg=k, value=v) for k, v in kw.items()])


def _dotted(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _dotted(node.value) + "." + node.attr
    return ""


def _base_pointer(node):
    """The pointer argument a load or store addresses (``p + offset`` -> ``p``), or None."""
    while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        node = node.left
    return node.id if isinstance(node, ast.Name) else None


def _with_pointer(node, new: str):
    """``p + offset`` with the base pointer ``p`` replaced by ``new``."""
    node = copy.deepcopy(node)
    if isinstance(node, ast.Name):
        return _name(new)
    cur = node
    while isinstance(cur.left, ast.BinOp) and isinstance(cur.left.op, ast.Add):
        cur = cur.left
    cur.left = _name(new)
    return node


class Registry:
    """The device functions known to the transformation and their generated tangent twins."""

    def __init__(self, sources: dict):
        self.defs = {}            # name -> FunctionDef (as written)
        self.module = {}          # name -> the module prefix its callers use (e.g. "lad.")
        for prefix, src in sources.items():
            tree = ast.parse(textwrap.dedent(src))
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    self.defs[node.name] = node
                    self.module[node.name] = prefix
        self.out = {}             # (name, mask) -> generated FunctionDef
        self.order = []

    def known(self, call: ast.Call):
        name = _dotted(call.func)
        short = name.split(".")[-1]
        if short in self.defs and (name == short or name == self.module[short].rstrip(".") + "." + short):
            return short
        return None

    def tangent_kernel(self, name: str, pointers: tuple) -> str:
        """The tangent of a kernel: loads through ``pointers`` carry tangents (read from ``<p>_t``), and
        stores and atomic adds through them write their tangents to ``<p>_t``."""
        key = (name, ("kernel",) + tuple(pointers))
        if key not in self.out:
            fn = self.defs[name]
            gen = Tangent(self, fn, (), f"{name}_t", pointers).build()
            self.out[key] = gen
            self.order.append(key)
        return self.out[key].name

    def reverse(self, name: str, mask: tuple) -> str:
        """The adjoint twin of a device function for a mask (1 a varied value, 2 a parameter pointer)."""
        key = (name, ("reverse",) + tuple(mask))
        if key not in self.out:
            tname = f"{name}_r_" + "".join(str(int(m)) for m in mask)
            self.out[key] = None
            gen = Reverse(self, self.defs[name], tname, (), mask=tuple(mask)).build()
            self.out[key] = gen
            self.order.append(key)
        return self.out[key].name

    def reverse_kernel(self, name: str, params: tuple, cotangents: dict) -> str:
        """The adjoint of a kernel with respect to the parameter tables ``params``."""
        key = (name, ("reverse_kernel",) + tuple(params))
        if key not in self.out:
            gen = Reverse(self, self.defs[name], f"{name}_r", params, cotangents).build()
            self.out[key] = gen
            self.order.append(key)
        return self.out[key].name

    def tangent(self, name: str, mask: tuple) -> str:
        key = (name, mask)
        if key not in self.out:
            fn = self.defs[name]
            tname = f"{name}_t_" + "".join(str(int(m)) for m in mask)
            self.out[key] = None        # recursion guard
            # 1: a value with a tangent, 2: a pointer whose loads and stores carry tangents
            pointers = [a.arg for a, m in zip(fn.args.args, mask) if m == 2]
            gen = Tangent(self, fn, tuple(m == 1 for m in mask), tname, pointers, extra=mask).build()
            self.out[key] = gen
            self.order.append(key)
        return self.out[key].name


class Tangent:
    """The tangent twin of one function for one set of arguments with tangents."""

    def __init__(self, reg: Registry, fn: ast.FunctionDef, mask: tuple, name: str, pointers=(), extra=None):
        self.reg, self.fn, self.mask, self.name = reg, fn, mask, name
        # the arguments that get a tangent parameter, values (1) and pointers (2), in order; None for a kernel
        self.extra = extra
        self.diff = set()          # variables with a tangent variable
        self.assigned = set()      # variables defined so far
        # the variables that carry a tangent anywhere: each of their assignments defines it (zero where the
        # expression has none), so a tangent exists wherever its value does, on every compiled path
        self.force = set()
        self.ever = set()
        self.pointers = set(pointers)   # pointer arguments with tangent twins (kernels)
        self.counter = 0

    # the tangents of expressions
    def d(self, e):
        if isinstance(e, ast.Constant):
            return ZERO
        if isinstance(e, ast.Name):
            return _name(e.id + "_t") if e.id in self.diff else ZERO
        if isinstance(e, ast.Attribute):
            return ZERO
        if isinstance(e, ast.UnaryOp):
            if isinstance(e.op, ast.USub):
                t = self.d(e.operand)
                return ZERO if t is ZERO else _neg(t)
            if isinstance(e.op, ast.UAdd):
                return self.d(e.operand)
            return ZERO
        if isinstance(e, ast.BinOp):
            a, b = e.left, e.right
            ta, tb = self.d(a), self.d(b)
            if isinstance(e.op, ast.Add):
                return _add(ta, tb)
            if isinstance(e.op, ast.Sub):
                return _sub(ta, tb)
            if isinstance(e.op, ast.Mult):
                return _add(_mul(ta, copy.deepcopy(b)), _mul(copy.deepcopy(a), tb))
            if isinstance(e.op, ast.Div):
                # d(a / b) = a_t / b - (a / b) b_t / b
                first = ZERO if ta is ZERO else _bin(ta, ast.Div(), copy.deepcopy(b))
                second = ZERO if tb is ZERO else _bin(_mul(copy.deepcopy(e), tb), ast.Div(), copy.deepcopy(b))
                return _sub(first, second)
            if isinstance(e.op, ast.Pow):
                if tb is not ZERO or not isinstance(b, ast.Constant):
                    raise NotImplementedError("only constant powers")
                n = b.value
                if ta is ZERO:
                    return ZERO
                return _mul(_bin(_num(float(n)), ast.Mult(), _bin(copy.deepcopy(a), ast.Pow(), _num(n - 1))), ta)
            if isinstance(e.op, (ast.BitAnd, ast.BitOr, ast.Mod, ast.FloorDiv, ast.LShift, ast.RShift)):
                return ZERO
            raise NotImplementedError(ast.dump(e.op))
        if isinstance(e, (ast.Compare, ast.BoolOp)):
            return ZERO
        if isinstance(e, ast.IfExp):
            ta, tb = self.d(e.body), self.d(e.orelse)
            if ta is ZERO and tb is ZERO:
                return ZERO
            return ast.IfExp(test=copy.deepcopy(e.test), body=ta if ta is not ZERO else _num(0.0),
                             orelse=tb if tb is not ZERO else _num(0.0))
        if isinstance(e, ast.Call):
            return self.d_call(e)
        raise NotImplementedError(ast.dump(e))

    def d_call(self, e: ast.Call):
        name = _dotted(e.func)
        short = name.split(".")[-1]
        if isinstance(e.func, ast.Attribute) and e.func.attr == "to":
            t = self.d(e.func.value)
            if t is ZERO:
                return ZERO
            return ast.Call(func=ast.Attribute(value=t, attr="to", ctx=ast.Load()), args=e.args, keywords=e.keywords)
        if short in _FLAT:
            return ZERO
        if short == "load":
            base = _base_pointer(e.args[0])
            if base is None or base not in self.pointers:
                return ZERO
            kw = {k.arg: k.value for k in e.keywords}
            new = {k: copy.deepcopy(v) for k, v in kw.items() if k != "other"}
            if "other" in kw:
                new["other"] = _num(0.0)
            return _call("tl.load", _with_pointer(e.args[0], base + "_t"), *[copy.deepcopy(a) for a in e.args[1:]],
                         **new)
        if short == "where":
            c, a, b = e.args
            ta, tb = self.d(a), self.d(b)
            if ta is ZERO and tb is ZERO:
                return ZERO
            return _call("tl.where", copy.deepcopy(c), ta if ta is not ZERO else _num(0.0),
                         tb if tb is not ZERO else _num(0.0))
        if short in ("maximum", "minimum"):
            a, b = e.args
            ta, tb = self.d(a), self.d(b)
            if ta is ZERO and tb is ZERO:
                return ZERO
            # the larger (smaller) argument's tangent, and at a tie half of each (autograd's convention)
            ta = ta if ta is not ZERO else _num(0.0)
            tb = tb if tb is not ZERO else _num(0.0)
            cmp = ast.Gt() if short == "maximum" else ast.Lt()
            first = ast.Compare(left=copy.deepcopy(a), ops=[cmp], comparators=[copy.deepcopy(b)])
            second = ast.Compare(left=copy.deepcopy(b), ops=[cmp], comparators=[copy.deepcopy(a)])
            tie = _mul(_num(0.5), _add(copy.deepcopy(ta), copy.deepcopy(tb)))
            return _call("tl.where", first, ta, _call("tl.where", second, tb, tie))
        if short == "sum":
            t = self.d(e.args[0])
            if t is ZERO:
                return ZERO
            return ast.Call(func=copy.deepcopy(e.func), args=[t, *[copy.deepcopy(a) for a in e.args[1:]]],
                            keywords=copy.deepcopy(e.keywords))
        if short == "sqrt":
            t = self.d(e.args[0])
            return ZERO if t is ZERO else _bin(_mul(_num(0.5), t), ast.Div(), copy.deepcopy(e))
        if short == "exp":
            t = self.d(e.args[0])
            return ZERO if t is ZERO else _mul(copy.deepcopy(e), t)
        if short == "erfc":
            u = e.args[0]
            t = self.d(u)
            if t is ZERO:
                return ZERO
            if not self._in_scope("inv_sqrt_pi"):
                raise NotImplementedError("erfc needs inv_sqrt_pi in scope")
            gauss = _call("libdevice.exp", ast.UnaryOp(op=ast.USub(), operand=_bin(copy.deepcopy(u), ast.Mult(),
                                                                                    copy.deepcopy(u))))
            return _mul(_mul(_mul(_num(-2.0), _name("inv_sqrt_pi")), gauss), t)
        if short == "pow":
            a, b = e.args
            ta, tb = self.d(a), self.d(b)
            if tb is not ZERO:
                raise NotImplementedError("pow with a varying exponent")
            if ta is ZERO:
                return ZERO
            return _mul(_bin(_mul(copy.deepcopy(e), copy.deepcopy(b)), ast.Div(), copy.deepcopy(a)), ta)
        if self.reg.known(e):
            raise NotImplementedError(f"call of {name} inside an expression (hoist it)")
        if any(self.d(a) is not ZERO for a in e.args):
            raise NotImplementedError(f"no derivative rule for {name}")
        return ZERO

    def _in_scope(self, var: str) -> bool:
        args = {a.arg for a in self.fn.args.args}
        return var in args or any(isinstance(n, ast.Name) and n.id == var and isinstance(n.ctx, ast.Store)
                                  for n in ast.walk(self.fn))

    # the tangents of statements
    def hoist(self, stmt):
        """Calls of known device functions nested in expressions become assignments of their own."""
        before = []

        class H(ast.NodeTransformer):
            def visit_Call(h, node):
                h.generic_visit(node)
                if self.reg.known(node) and node is not top:
                    self.counter += 1
                    tmp = f"_h{self.counter}"
                    before.append(ast.Assign(targets=[ast.Name(id=tmp, ctx=ast.Store())], value=node, lineno=0))
                    return _name(tmp)
                return node

        stmt = copy.deepcopy(stmt)        # the function's own statements stay as written (several passes)
        top = stmt.value if isinstance(stmt, (ast.Assign, ast.Expr)) else None
        stmt = H().visit(stmt)
        return before + [stmt]

    def call_known(self, targets, call: ast.Call):
        """``targets = f(args)`` for a known device function: its tangent twin, or the call alone."""
        short = self.reg.known(call)
        mask = tuple(2 if isinstance(a, ast.Name) and a.id in self.pointers else int(self.d(a) is not ZERO)
                     for a in call.args)
        names = [t.id for t in targets]
        if not any(mask):
            for n in names:
                self.diff.discard(n)
            if _dotted(call.func) == short and self.reg.module[short]:
                # an untransformed ladder called from the generated module needs its module prefix
                call = copy.deepcopy(call)
                call.func = ast.Attribute(value=_name(self.reg.module[short].rstrip(".")), attr=short, ctx=ast.Load())
            return [ast.Assign(targets=[ast.Tuple(elts=[ast.Name(id=n, ctx=ast.Store()) for n in names],
                                                  ctx=ast.Store()) if len(names) > 1 else ast.Name(id=names[0],
                                                                                                   ctx=ast.Store())],
                               value=call, lineno=0)] + self.zeros(names)
        tname = self.reg.tangent(short, mask)
        # every tangent twin lives in the generated module
        args = [copy.deepcopy(a) for a in call.args] + [self.d(a) if m == 1 else _name(a.id + "_t")
                                                        for a, m in zip(call.args, mask) if m]
        new = _call(tname, *args)
        new.keywords = copy.deepcopy(call.keywords)
        gen = self.reg.out[(short, mask)]
        outs = gen.n_out
        if outs != len(names):
            raise NotImplementedError(f"{short}: {outs} outputs, {len(names)} targets")
        tnames = [n + "_t" for n in names]
        nonzero = gen.nonzero
        store = [ast.Name(id=n, ctx=ast.Store()) for n in names] + [ast.Name(id=t, ctx=ast.Store())
                                                                    for t, nz in zip(tnames, nonzero) if nz]
        for n, nz in zip(names, nonzero):
            (self.diff.add if nz else self.diff.discard)(n)
        return ([ast.Assign(targets=[ast.Tuple(elts=store, ctx=ast.Store())], value=new, lineno=0)]
                + self.zeros([n for n, nz in zip(names, nonzero) if not nz]))

    def stmt(self, s):
        out = []
        for h in self.hoist(s) if isinstance(s, (ast.Assign, ast.AugAssign, ast.Expr, ast.Return)) else [s]:
            out.extend(self.stmt1(h))
            self.ever |= self.diff
        return out

    def zeros(self, names):
        """Zero tangents of the forced variables among ``names``, which got none from their assignment."""
        out = []
        for n in names:
            if n in self.force and n not in self.diff:
                out.append(ast.Assign(targets=[ast.Name(id=n + "_t", ctx=ast.Store())],
                                      value=_bin(_name(n), ast.Mult(), _num(0.0)), lineno=0))
                self.diff.add(n)
        return out

    def stmt1(self, s):
        if not isinstance(s, ast.If):
            for node in ast.walk(s):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    self.assigned.add(node.id)
        if isinstance(s, ast.Assign):
            tgt = s.targets[0]
            if isinstance(s.value, ast.Call) and self.reg.known(s.value):
                elts = tgt.elts if isinstance(tgt, ast.Tuple) else [tgt]
                return self.call_known(elts, s.value)
            if not isinstance(tgt, ast.Name):
                raise NotImplementedError(ast.dump(s))
            if isinstance(s.value, ast.BinOp) and _base_pointer(s.value) in self.pointers:
                # a pointer into a tangent-carrying buffer: its tangent twin addresses the tangent buffer
                self.pointers.add(tgt.id)
                return [s, ast.Assign(targets=[ast.Name(id=tgt.id + "_t", ctx=ast.Store())],
                                      value=_with_pointer(s.value, _base_pointer(s.value) + "_t"), lineno=0)]
            t = self.d(s.value)
            out = [s]
            if t is ZERO:
                self.diff.discard(tgt.id)
                out += self.zeros([tgt.id])
            else:
                out.append(ast.Assign(targets=[ast.Name(id=tgt.id + "_t", ctx=ast.Store())], value=t, lineno=0))
                self.diff.add(tgt.id)
            return out
        if isinstance(s, ast.AugAssign):
            name = s.target.id
            t = self.d(s.value)
            if isinstance(s.op, ast.Add) or isinstance(s.op, ast.Sub):
                if t is ZERO:
                    return [s]
                if name in self.diff:
                    return [s, ast.AugAssign(target=ast.Name(id=name + "_t", ctx=ast.Store()), op=s.op, value=t)]
                self.diff.add(name)
                value = t if isinstance(s.op, ast.Add) else ast.UnaryOp(op=ast.USub(), operand=t)
                return [s, ast.Assign(targets=[ast.Name(id=name + "_t", ctx=ast.Store())], value=value, lineno=0)]
            raise NotImplementedError(ast.dump(s))
        if isinstance(s, ast.Expr):
            call = s.value
            # the atomic-add helper is a store (its tangent goes to the tangent buffer: side_effect)
            if isinstance(call, ast.Call) and self.reg.known(call) and self.reg.known(call) != "_add_sorted":
                short = self.reg.known(call)
                mask = tuple(2 if isinstance(a, ast.Name) and a.id in self.pointers else int(self.d(a) is not ZERO)
                             for a in call.args)
                if not any(mask):
                    return [s]
                args = [copy.deepcopy(a) for a in call.args] + [self.d(a) if m == 1 else _name(a.id + "_t")
                                                                for a, m in zip(call.args, mask) if m]
                new = _call(self.reg.tangent(short, mask), *args)
                new.keywords = copy.deepcopy(call.keywords)
                return [ast.Expr(value=new)]
            return [s] + self.side_effect(s.value)
        if isinstance(s, ast.If):
            before = set(self.diff)
            known = set(self.assigned)
            body = [x for b in s.body for x in self.stmt(b)]
            after_body, assigned_body = set(self.diff), set(self.assigned)
            self.diff, self.assigned = set(before), set(known)
            orelse = [x for b in s.orelse for x in self.stmt(b)]
            after_else, assigned_else = set(self.diff), set(self.assigned)
            self.diff = after_body | after_else
            self.assigned = assigned_body | assigned_else
            return [ast.If(test=s.test, body=body or [ast.Pass()], orelse=orelse)]
        if isinstance(s, ast.Return):
            vals = s.value.elts if isinstance(s.value, ast.Tuple) else ([s.value] if s.value is not None else [])
            if not vals:
                return [s]
            tans = [self.d(v) for v in vals]
            self.n_out = len(vals)
            self.nonzero = [t is not ZERO for t in tans]
            elts = [copy.deepcopy(v) for v in vals] + [t for t in tans if t is not ZERO]
            return [ast.Return(value=ast.Tuple(elts=elts, ctx=ast.Load()))]
        if isinstance(s, ast.Pass):
            return [s]
        raise NotImplementedError(ast.dump(s))

    def side_effect(self, call):
        """Stores and atomic adds of values with tangents: the same into the tangent outputs."""
        if not isinstance(call, ast.Call):
            return []
        short = _dotted(call.func).split(".")[-1]
        if short in ("store", "atomic_add"):
            base = _base_pointer(call.args[0])
            t = self.d(call.args[1])
            if t is ZERO or base not in self.pointers:
                return []
            new = copy.deepcopy(call)
            new.args[0] = _with_pointer(call.args[0], base + "_t")
            new.args[1] = t
            return [ast.Expr(value=new)]
        if short == "_add_sorted":
            base = _base_pointer(call.args[0])
            t = self.d(call.args[4])
            if t is ZERO or base not in self.pointers:
                return []
            new = copy.deepcopy(call)
            new.args[0] = _with_pointer(call.args[0], base + "_t")
            new.args[4] = t
            return [ast.Expr(value=new)]
        return []

    def build(self) -> ast.FunctionDef:
        args = [a.arg for a in self.fn.args.args]
        statements = [s for s in self.fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
        for _ in range(8):
            self.assigned = set(args)
            self.diff = {a for a, m in zip(args, self.mask) if m}
            self.ever, self.counter = set(self.diff), 0
            self.n_out, self.nonzero = 0, []
            body = self.zeros(args) + [x for s in statements for x in self.stmt(s)]
            if self.ever <= self.force:
                break
            self.force |= self.ever
        doc = f"Tangent of ``{self.fn.name}`` along the arguments {', '.join(a for a, m in zip(args, self.mask) if m)}."
        new_args = copy.deepcopy(self.fn.args)
        if self.extra is None:
            # a kernel: the tangent pointers go before the compile-time constants
            first = next((i for i, a in enumerate(new_args.args) if a.annotation is not None), len(new_args.args))
            extra = [ast.arg(arg=a + "_t") for a in args if a in self.pointers]
            new_args.args = new_args.args[:first] + extra + new_args.args[first:]
            doc = f"Tangent of the kernel ``{self.fn.name}``: tangents read and written through " + \
                ", ".join(f"``{a}_t``" for a in args if a in self.pointers) + "."
        else:
            new_args.args = new_args.args + [ast.arg(arg=a + "_t") for a, m in zip(args, self.extra) if m]
            along = ", ".join(a for a, m in zip(args, self.extra) if m)
            doc = f"Tangent of ``{self.fn.name}`` along the arguments {along}."
        fn = ast.FunctionDef(name=self.name, args=new_args, body=[ast.Expr(value=ast.Constant(value=doc))] + body,
                             decorator_list=copy.deepcopy(self.fn.decorator_list), returns=None, type_params=[],
                             lineno=0)
        fn.n_out, fn.nonzero = self.n_out, self.nonzero
        return fn


def _loads(e):
    return {n.id for n in ast.walk(e) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


class _Swap(ast.NodeTransformer):
    """Names replaced: by other names (``env``) or by expressions (``exprs``)."""

    def __init__(self, env=None, exprs=None):
        self.env, self.exprs = env or {}, exprs or {}

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            if node.id in self.exprs:
                return copy.deepcopy(self.exprs[node.id])
            if node.id in self.env:
                return ast.Name(id=self.env[node.id], ctx=ast.Load())
        return node


def _returns(fn) -> int:
    for node in ast.walk(fn):
        if isinstance(node, ast.Return) and node.value is not None:
            return len(node.value.elts) if isinstance(node.value, ast.Tuple) else 1
    return 0


class Reverse:
    """The adjoint (reverse mode) of a kernel or device function with respect to the per-atom parameters it
    loads: its statements in SSA form (a copy joins the versions of a variable where branches meet), each
    followed by the zero adjoint of a varied value (so an adjoint exists wherever its value does), then the
    adjoint sweep in reverse order. A statement's partials are its tangents along one operand (the tangent
    rules of :class:`Tangent`), a parameter load adds its adjoint atomically to the gradient table, and an
    output store of a kernel seeds the adjoints from the incoming gradients (``cotangents``: output pointer
    -> incoming-gradient pointer). A device function takes the adjoints of its outputs and returns those of
    its varied arguments (``mask``: 1 a varied value, 2 a parameter pointer)."""

    def __init__(self, reg, fn, name, params, cotangents=None, mask=None):
        self.reg, self.fn, self.name = reg, fn, name
        self.params = set(params)
        self.cot = dict(cotangents or {})
        self.mask = mask
        self.k = 0
        self.alias = {}           # SSA name -> the parameter pointer it addresses
        self.varied = set()

    def fresh(self, name):
        self.k += 1
        return f"{name}__{self.k}"

    def param_base(self, node):
        base = _base_pointer(node)
        if base in self.params:
            return base
        return self.alias.get(base)

    def is_param_load(self, e):
        return (isinstance(e, ast.Call) and _dotted(e.func).split(".")[-1] == "load"
                and self.param_base(e.args[0]) is not None)

    def zero_adjoint(self, name):
        return ast.Assign(targets=[ast.Name(id=name + "_b", ctx=ast.Store())],
                          value=_bin(_name(name), ast.Mult(), _num(0.0)), lineno=0)

    # the kernel bodies in single static assignment form, for the adjoint sweep
    def hoist(self, e, out):
        """Known calls and parameter loads nested in ``e`` become SSA assignments of their own."""
        rev = self

        class H(ast.NodeTransformer):
            def visit_Call(h, node):
                h.generic_visit(node)
                if node is top:
                    return node
                if rev.reg.known(node):
                    tmp = rev.fresh("_h")
                    out.extend(rev.call_assign([tmp], node, False))
                    return _name(tmp)
                if rev.is_param_load(node):
                    tmp = rev.fresh("_h")
                    out.extend(rev.assign(tmp, node))
                    return _name(tmp)
                return node

        top = e
        return H().visit(e)

    def assign(self, new, e):
        """``new = e`` (``e`` renamed and hoisted): the statement, its alias or variation, its zero adjoint."""
        out = [ast.Assign(targets=[ast.Name(id=new, ctx=ast.Store())], value=e, lineno=0)]
        if isinstance(e, ast.BinOp) and self.param_base(e) is not None and not isinstance(e, ast.Call):
            self.alias[new] = self.param_base(e)
            out.append(ast.Assign(targets=[ast.Name(id=new + "_g", ctx=ast.Store())],
                                  value=_with_pointer(e, self.grad_of(self.param_base(e))), lineno=0))
            return out
        if self.is_param_load(e) or (_loads(e) & self.varied):
            self.varied.add(new)
            out.append(self.zero_adjoint(new))
        return out

    def call_assign(self, news, value, as_tuple):
        """``news = f(args)`` for a known device function, with zero adjoints when an argument is varied."""
        varied = any((_loads(a) & self.varied)
                     or (isinstance(a, ast.Name) and (a.id in self.params or a.id in self.alias)) for a in value.args)
        short = self.reg.known(value)
        if _dotted(value.func) == short and self.reg.module[short]:
            # an untransformed ladder replayed in the generated module needs its module prefix
            value = copy.deepcopy(value)
            value.func = ast.Attribute(value=_name(self.reg.module[short].rstrip(".")), attr=short, ctx=ast.Load())
        store = (ast.Tuple(elts=[ast.Name(id=n, ctx=ast.Store()) for n in news], ctx=ast.Store())
                 if as_tuple else ast.Name(id=news[0], ctx=ast.Store()))
        node = ast.Assign(targets=[store], value=value, lineno=0)
        node.call_outputs = news
        out = [node]
        if varied:
            for n in news:
                self.varied.add(n)
                out.append(self.zero_adjoint(n))
        return out

    def grad_of(self, base):
        return base + "_g"

    def ssa(self, stmts, env, after=()):
        out = []
        for j, s in enumerate(stmts):
            out.extend(self.ssa1(s, env, [stmts[j + 1:], *after]))
        return out

    @staticmethod
    def first_use(stmts, name):
        """'read' if ``name`` is read in ``stmts`` before it is written on every path, 'write' if every path
        writes it first, None if neither."""
        for s in stmts:
            if isinstance(s, ast.If):
                if name in _loads(s.test):
                    return "read"
                b, e = Reverse.first_use(s.body, name), Reverse.first_use(s.orelse, name)
                if "read" in (b, e):
                    return "read"
                if b == "write" and e == "write":
                    return "write"
                continue
            if isinstance(s, ast.AugAssign) and isinstance(s.target, ast.Name) and s.target.id == name:
                return "read"
            if name in _loads(s):
                return "read"
            if isinstance(s, ast.Assign) and any(isinstance(n, ast.Name) and n.id == name
                                                 for tgt in s.targets for n in ast.walk(tgt)):
                return "write"
        return None

    def live(self, name, rest):
        for stmts in rest:
            use = self.first_use(stmts, name)
            if use is not None:
                return use == "read"
        return False

    def ssa1(self, s, env, rest=()):
        if isinstance(s, ast.Assign):
            pre = []
            value = self.hoist(_Swap(env).visit(copy.deepcopy(s.value)), pre)
            tgt = s.targets[0]
            if isinstance(tgt, ast.Name) and not (isinstance(value, ast.Call) and self.reg.known(value)):
                new = self.fresh(tgt.id)
                env[tgt.id] = new
                return pre + self.assign(new, value)
            elts = tgt.elts if isinstance(tgt, ast.Tuple) else [tgt]
            news = [self.fresh(x.id) for x in elts]
            for x, n in zip(elts, news):
                env[x.id] = n
            return pre + self.call_assign(news, value, isinstance(tgt, ast.Tuple))
        if isinstance(s, ast.AugAssign):
            pre = []
            value = self.hoist(_Swap(env).visit(copy.deepcopy(s.value)), pre)
            old = env.get(s.target.id, s.target.id)
            new = self.fresh(s.target.id)
            env[s.target.id] = new
            return pre + self.assign(new, _bin(_name(old), s.op, value))
        if isinstance(s, ast.If):
            test = _Swap(env).visit(copy.deepcopy(s.test))
            env_b, env_e = dict(env), dict(env)
            body = self.ssa(s.body, env_b, rest)
            orelse = self.ssa(s.orelse, env_e, rest)
            for name in sorted(set(env_b) | set(env_e)):
                vb, ve, before = env_b.get(name), env_e.get(name), env.get(name)
                if vb == ve:
                    env[name] = vb
                elif vb is None or ve is None or not self.live(name, rest):
                    # defined on one side only, or not read again before it is redefined: no join (a copy
                    # would read a version that some compiled paths never define)
                    env[name] = vb if vb != before else ve
                elif vb is not None and ve is not None:
                    new = self.fresh(name)
                    body.append(ast.Assign(targets=[ast.Name(id=new, ctx=ast.Store())], value=_name(vb), lineno=0))
                    orelse.append(ast.Assign(targets=[ast.Name(id=new, ctx=ast.Store())], value=_name(ve), lineno=0))
                    if vb in self.varied or ve in self.varied:
                        # the joined version has an adjoint on both paths
                        self.varied.add(new)
                        body.append(self.zero_adjoint(new))
                        orelse.append(self.zero_adjoint(new))
                    env[name] = new
                else:
                    env[name] = vb if vb is not None else ve
            return [ast.If(test=test, body=body or [ast.Pass()], orelse=orelse)]
        if isinstance(s, ast.Expr):
            call = _Swap(env).visit(copy.deepcopy(s.value))
            if isinstance(call, ast.Call):
                short = _dotted(call.func).split(".")[-1]
                if short in ("store", "atomic_add", "_add_sorted"):
                    base = _base_pointer(call.args[0])
                    if base in self.cot or base in ("epair_ptr",):
                        seed = ast.Pass()
                        seed.seed = (call, self.cot.get(base))
                        return [seed]
            return [ast.Expr(value=call)]
        if isinstance(s, ast.Return):
            if s.value is None:
                return [s]
            node = ast.Pass()
            node.returned = _Swap(env).visit(copy.deepcopy(s.value))
            return [node]
        if isinstance(s, ast.Pass):
            return []
        raise NotImplementedError(ast.dump(s))

    # the adjoint sweep
    def partial(self, e, a, bar):
        """``d e / d a`` times ``bar``: the tangent of ``e`` along ``a`` with ``bar`` as ``a``'s tangent."""
        tg = Tangent(self.reg, self.fn, (), "_")
        tg.diff = {a}
        d = tg.d(copy.deepcopy(e))
        return ZERO if d is ZERO else _Swap(exprs={a + "_t": bar}).visit(d)

    def accumulate(self, e, bar):
        out = []
        for a in sorted(_loads(e) & self.varied):
            c = self.partial(e, a, bar)
            if c is ZERO:
                continue
            c = ast.Call(func=ast.Attribute(value=c, attr="to", ctx=ast.Load()),
                         args=[ast.Attribute(value=_name(a), attr="dtype", ctx=ast.Load())], keywords=[])
            out.append(ast.Assign(targets=[ast.Name(id=a + "_b", ctx=ast.Store())],
                                  value=_bin(_name(a + "_b"), ast.Add(), c), lineno=0))
        return out

    def sweep(self, stmts):
        out = []
        for s in reversed(stmts):
            out.extend(self.sweep1(s))
        return out

    def sweep1(self, s):
        if isinstance(s, ast.If):
            body, orelse = self.sweep(s.body), self.sweep(s.orelse)
            if not body and not orelse:
                return []
            return [ast.If(test=s.test, body=body or [ast.Pass()], orelse=orelse)]
        if getattr(s, "seed", None) is not None:
            call, cot = s.seed
            if cot is None:
                return []
            short = _dotted(call.func).split(".")[-1]
            if short == "_add_sorted":
                addr = _bin(copy.deepcopy(call.args[0]), ast.Add(), copy.deepcopy(call.args[3]))
                mask = _bin(_bin(copy.deepcopy(call.args[5]), ast.BitOr(), copy.deepcopy(call.args[6])),
                            ast.BitOr(), copy.deepcopy(call.args[7]))
                value = call.args[4]
            else:
                addr = copy.deepcopy(call.args[0])
                kw = {k.arg: k.value for k in call.keywords}
                mask = copy.deepcopy(kw.get("mask")) if "mask" in kw else None
                value = call.args[1]
            if not (_loads(value) & self.varied):
                return []
            load = _call("tl.load", _with_pointer(addr, cot), **({"mask": mask, "other": _num(0.0)} if mask else {}))
            tmp = self.fresh("_cot")
            return [ast.Assign(targets=[ast.Name(id=tmp, ctx=ast.Store())], value=load, lineno=0)] + \
                self.accumulate(value, _name(tmp))
        if getattr(s, "returned", None) is not None:
            vals = s.returned.elts if isinstance(s.returned, ast.Tuple) else [s.returned]
            out = []
            for j, v in enumerate(vals):
                out.extend(self.accumulate(v, _name(f"out{j}_b")))
            return out
        if isinstance(s, ast.Assign) and getattr(s, "call_outputs", None) is not None:
            news = s.call_outputs
            if not any(n in self.varied for n in news):
                return []
            call = s.value
            short = self.reg.known(call)
            mask = tuple(2 if isinstance(a, ast.Name) and (a.id in self.params or a.id in self.alias)
                         else int(bool(_loads(a) & self.varied)) for a in call.args)
            twin = self.reg.reverse(short, mask)
            grads = [_name(a.id + "_g") if a.id in self.alias else _name(self.grad_of(a.id))
                     for a, m in zip(call.args, mask) if m == 2]
            bars = [_name(n + "_b") for n in news]
            res = [self.fresh("_adj") for a, m in zip(call.args, mask) if m == 1]
            new = _call(twin, *[copy.deepcopy(a) for a in call.args], *grads, *bars)
            new.keywords = copy.deepcopy(call.keywords)
            out = []
            if res:
                store = (ast.Tuple(elts=[ast.Name(id=r, ctx=ast.Store()) for r in res], ctx=ast.Store())
                         if len(res) > 1 else ast.Name(id=res[0], ctx=ast.Store()))
                out.append(ast.Assign(targets=[store], value=new, lineno=0))
                for r, a in zip(res, [a for a, m in zip(call.args, mask) if m == 1]):
                    out.extend(self.accumulate(a, _name(r)))
            else:
                out.append(ast.Expr(value=new))
            return out
        if isinstance(s, ast.Assign):
            tgt = s.targets[0]
            if not isinstance(tgt, ast.Name) or tgt.id not in self.varied:
                return []
            if self.is_param_load(s.value):
                base = _base_pointer(s.value.args[0])
                gbase = base + "_g" if base in self.alias else self.grad_of(base)
                kw = {k.arg: k.value for k in s.value.keywords}
                mask = {"mask": copy.deepcopy(kw["mask"])} if "mask" in kw else {}
                return [ast.Expr(value=_call("tl.atomic_add", _with_pointer(s.value.args[0], gbase),
                                             _name(tgt.id + "_b"), **mask, sem=ast.Constant(value="relaxed")))]
            return self.accumulate(s.value, _name(tgt.id + "_b"))
        return []

    def build(self) -> ast.FunctionDef:
        args = [a.arg for a in self.fn.args.args]
        env = {}
        prelude = []
        if self.mask is not None:
            for a, m in zip(args, self.mask):
                if m == 1:
                    self.varied.add(a)
                    prelude.append(self.zero_adjoint(a))
                elif m == 2:
                    self.params.add(a)
        statements = [s for s in self.fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
        forward = prelude + self.ssa(statements, env)
        body = forward + self.sweep(forward)
        new_args = copy.deepcopy(self.fn.args)
        if self.mask is None:
            # a kernel: the incoming-gradient pointers and the gradient table before the constants
            first = next((i for i, a in enumerate(new_args.args) if a.annotation is not None), len(new_args.args))
            extra = [ast.arg(arg=c) for c in dict.fromkeys(v for v in self.cot.values() if v)]
            extra += [ast.arg(arg=self.grad_of(p)) for p in sorted(self.params)]
            new_args.args = new_args.args[:first] + extra + new_args.args[first:]
            doc = (f"Adjoint of the kernel ``{self.fn.name}`` with respect to the parameters it loads: their "
                   f"gradients added to {', '.join('``' + self.grad_of(p) + '``' for p in sorted(self.params))}.")
        else:
            new_args.args = (new_args.args + [ast.arg(arg=self.grad_of(a)) for a, m in zip(args, self.mask) if m == 2]
                             + [ast.arg(arg=f"out{j}_b") for j in range(_returns(self.fn))])
            varied_args = [a for a, m in zip(args, self.mask) if m == 1]
            if len(varied_args) == 1:
                body.append(ast.Return(value=_name(varied_args[0] + "_b")))
            elif varied_args:
                body.append(ast.Return(value=ast.Tuple(elts=[_name(a + "_b") for a in varied_args],
                                                       ctx=ast.Load())))
            doc = f"Adjoint of ``{self.fn.name}``: the output adjoints in, those of the varied arguments out."
        fn = ast.FunctionDef(name=self.name, args=new_args, body=[ast.Expr(value=ast.Constant(value=doc))] + body,
                             decorator_list=copy.deepcopy(self.fn.decorator_list), returns=None, type_params=[],
                             lineno=0)
        fn.n_out, fn.nonzero = 0, []
        return fn


HEADER = '''"""Tangents (forward-mode derivatives) of the Triton device functions of the AMOEBA and HIPPO kernels.

Generated by ``tools/generate_amoeba_derivatives.py`` (do not edit by hand; regenerate). Each
``<name>_t_<mask>`` takes the arguments of ``<name>`` followed by the tangents of those marked in the mask
and returns the values of ``<name>`` followed by the tangents of the outputs that have one.
"""
# ruff: noqa: E501, F841
import triton
import triton.language as tl
from triton.language.extra import libdevice

'''


def generate(requests, kernels=(), reverse=(), reverse_tangents=()) -> str:
    """The module source for the tangent twins ``requests`` ((function name, mask) pairs) and the tangent
    kernels ``kernels`` ((kernel name, pointers with tangents) pairs)."""
    models = Path(__file__).resolve().parents[1] / "src/xnn/ffnn/models"
    sources = {"": (models / "amoeba_kernels.py").read_text(), "lad.": (models / "amoeba_ladders.py").read_text()}
    reg = Registry(sources)
    for name, mask in requests:
        reg.tangent(name, tuple(mask))
    for name, pointers in kernels:
        reg.tangent_kernel(name, tuple(pointers))
    for name, params, cotangents in reverse:
        reg.reverse_kernel(name, tuple(params), cotangents)
    # the generated adjoints become sources themselves: their tangents give the mixed second derivatives
    for key in list(reg.order):
        fn = reg.out[key]
        if "_r" in fn.name and fn.name not in reg.defs:
            reg.defs[fn.name] = fn
            reg.module[fn.name] = ""
    for name, pointers in reverse_tangents:
        reg.tangent_kernel(name, tuple(pointers))
    parts, defined, used = [], set(), set()
    for key in reg.order:
        fn = reg.out[key]
        defined.add(fn.name)
        used |= {n.func.id for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        mod = ast.Module(body=[fn], type_ignores=[])
        ast.fix_missing_locations(mod)
        parts.append(ast.unparse(mod).strip("\n"))
    # the untransformed kernel helpers the tangents call
    helpers = sorted(n for n in used - defined if n in reg.defs and reg.module[n] == "")
    imports = f"from .amoeba_kernels import {', '.join(helpers)}\n" if helpers else ""
    if any(isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "lad"
           for key in reg.order for n in ast.walk(reg.out[key])):
        imports = "from . import amoeba_ladders as lad\n" + imports

    # the header ends with a blank line after the third-party imports
    return HEADER + (imports + "\n\n" if imports else "\n") + "\n\n\n".join(parts) + "\n"


# the tangent twins the kernels need (grown with the kernels that use them)
REQUESTS = [
    ("_coulomb6", (1,)),
    ("_ewald6", (1, 1, 0, 0)),
    ("core_valence_deficit4", (0, 1, 1, 0, 0)),
    ("overlap_deficit5", (0, 1, 1, 0, 0)),
    ("overlap_deficit3", (0, 1, 1, 0, 0)),
    ("squared_overlap5", (0, 1, 1, 0, 0)),
]

# the tangent kernels: along the pair vectors, the multipoles (and AMOEBA's site offsets), into the outputs
_MULTIPOLES = ("r_ptr", "q_ptr", "d_ptr", "Q_ptr")
KERNELS = [
    ("_pairs_forward", _MULTIPOLES + ("enode_ptr", "epair_ptr", "fd_ptr", "fp_ptr", "du_ptr", "xp_ptr")),
    ("_pairs_backward", _MULTIPOLES + ("gr_ptr", "gq_ptr", "gd_ptr", "gQ_ptr")),
    ("_amoeba_forward", _MULTIPOLES + ("off_ptr", "enode_ptr", "epair_ptr", "fd_ptr", "fp_ptr", "du_ptr")),
    ("_amoeba_backward", _MULTIPOLES + ("off_ptr", "gr_ptr", "gq_ptr", "gd_ptr", "gQ_ptr", "goff_ptr")),
]

# the parameter adjoints of the forward kernels: the per-atom table's gradient from the incoming gradients of the
# energies, fields, mutual ladders and exchange-polarization tensors
REVERSE = [
    ("_pairs_forward", ("par_ptr",), {"enode_ptr": "ge_ptr", "fd_ptr": "gfd_ptr", "fp_ptr": "gfp_ptr",
                                      "du_ptr": "gdu_ptr", "xp_ptr": "gxp_ptr"}),
    ("_amoeba_forward", ("par_ptr",), {"enode_ptr": "ge_ptr", "fd_ptr": "gfd_ptr", "fp_ptr": "gfp_ptr",
                                       "du_ptr": "gdu_ptr"}),
]

# their tangents along the pair vectors and multipoles: the parameter part of a force loss's gradient
REVERSE_TANGENTS = [
    ("_pairs_forward_r", _MULTIPOLES + ("par_ptr_g",)),
    ("_amoeba_forward_r", _MULTIPOLES + ("off_ptr", "par_ptr_g")),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    default = Path(__file__).resolve().parents[1] / "src/xnn/ffnn/models/amoeba_tangents.py"
    ap.add_argument("--out", default=str(default))
    args = ap.parse_args()
    Path(args.out).write_text(generate(REQUESTS, KERNELS, REVERSE, REVERSE_TANGENTS))
    print("wrote", args.out)
