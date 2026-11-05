"""Small-array terms evaluated once in two forms: as PyTorch tensors, or on the host with a reverse-mode tape.

Terms with a few dozen small operations on a handful of atoms (the hydrogen-bond corrections) spend their time
in PyTorch's per-operation overhead. Their formulas are written once against an operations namespace:
:data:`TORCH` runs them on tensors (differentiable to any order, any device) and :data:`TAPE` on NumPy arrays,
recording each operation so that one reverse sweep gives the gradient. :func:`host_term` evaluates a term on the
host and returns the result as a differentiable tensor on the input's device: the first derivative comes from the
tape, higher derivatives (``create_graph``) from the PyTorch form.
"""
import math
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from torch import Tensor


def _unbroadcast(g: np.ndarray, shape) -> np.ndarray:
    """The gradient ``g`` of a broadcast result summed back to an operand of ``shape``."""
    if g.shape == shape:
        return g
    while g.ndim > len(shape):
        g = g.sum(0)
    axes = tuple(i for i, n in enumerate(shape) if n == 1 and g.shape[i] != 1)
    return g.sum(axes, keepdims=True) if axes else g


class Node:
    """A NumPy value on a :class:`Tape`; arithmetic with nodes, arrays and numbers records new nodes."""

    __slots__ = ("v", "g", "tape", "back")
    # NumPy arrays on the left defer to the reflected operators below
    __array_ufunc__ = None

    def __init__(self, tape: "Tape", v: np.ndarray, back: Optional[Callable] = None):
        self.v, self.g, self.tape, self.back = v, None, tape, back
        tape.nodes.append(self)

    def _acc(self, g):
        self.g = g if self.g is None else self.g + g

    def __add__(self, o):
        if isinstance(o, Node):
            return self.tape.op(self.v + o.v, lambda g: (self._acc(_unbroadcast(g, self.v.shape)),
                                                           o._acc(_unbroadcast(g, o.v.shape))))
        return self.tape.op(self.v + o, lambda g: self._acc(_unbroadcast(g, self.v.shape)))

    __radd__ = __add__

    def __sub__(self, o):
        if isinstance(o, Node):
            return self.tape.op(self.v - o.v, lambda g: (self._acc(_unbroadcast(g, self.v.shape)),
                                                           o._acc(_unbroadcast(-g, o.v.shape))))
        return self.tape.op(self.v - o, lambda g: self._acc(_unbroadcast(g, self.v.shape)))

    def __rsub__(self, o):
        return self.tape.op(o - self.v, lambda g: self._acc(_unbroadcast(-g, self.v.shape)))

    def __neg__(self):
        return self.tape.op(-self.v, lambda g: self._acc(-g))

    def __mul__(self, o):
        if isinstance(o, Node):
            return self.tape.op(self.v * o.v, lambda g: (self._acc(_unbroadcast(g * o.v, self.v.shape)),
                                                           o._acc(_unbroadcast(g * self.v, o.v.shape))))
        return self.tape.op(self.v * o, lambda g: self._acc(_unbroadcast(g * o, self.v.shape)))

    __rmul__ = __mul__

    def __truediv__(self, o):
        if isinstance(o, Node):
            out = self.v / o.v
            return self.tape.op(out, lambda g: (self._acc(_unbroadcast(g / o.v, self.v.shape)),
                                                  o._acc(_unbroadcast(-g * out / o.v, o.v.shape))))
        return self.tape.op(self.v / o, lambda g: self._acc(_unbroadcast(g / o, self.v.shape)))

    def __rtruediv__(self, o):
        out = o / self.v
        return self.tape.op(out, lambda g: self._acc(_unbroadcast(-g * out / self.v, self.v.shape)))

    def __pow__(self, k: int):
        if k == 1:
            return self
        if k == 2:
            return self * self
        x = self.v
        return self.tape.op(x ** k, lambda g: self._acc(g * k * x ** (k - 1)))

    def __getitem__(self, key):
        def back(g):
            full = np.zeros_like(self.v)
            full[key] = g
            self._acc(full)
        return self.tape.op(self.v[key], back)


class Tape:
    """The recorded operations of one evaluation."""

    def __init__(self):
        self.nodes: List[Node] = []

    def leaf(self, v: np.ndarray) -> Node:
        return Node(self, v)

    def op(self, v, back) -> Node:
        return Node(self, v, back)

    def backward(self, out: Node, seed: np.ndarray) -> None:
        """Accumulate the gradients of ``sum(seed * out)`` into the ``g`` of every node it depends on."""
        for n in self.nodes:
            n.g = None
        out.g = seed
        for n in reversed(self.nodes):
            if n.g is not None and n.back is not None:
                n.back(n.g)

    def release(self) -> None:
        """Drop the recorded operations (their closures refer to the nodes, a cycle only the collector frees)."""
        for n in self.nodes:
            n.back = None
        self.nodes.clear()


class _TapeOps:
    """The operations namespace on :class:`Node` values (NumPy arrays for masks and indices)."""

    pi = math.pi
    # tables from the host topology searches come as NumPy arrays
    host_arrays = True

    @staticmethod
    def value(x):
        return x.v if isinstance(x, Node) else x

    @staticmethod
    def take(x: Node, idx: np.ndarray) -> Node:
        def back(g):
            full = np.zeros_like(x.v)
            np.add.at(full, idx, g)
            x._acc(full)
        return x.tape.op(x.v[idx], back)

    @staticmethod
    def sum(x: Node) -> Node:
        return x.tape.op(x.v.sum(-1), lambda g: x._acc(np.broadcast_to(g[..., None], x.v.shape)))

    @staticmethod
    def _scatter(pos: Node, idx, grads):
        """Add the gradients ``grads`` of the gathered rows ``idx`` (lists) to ``pos``."""
        full = np.zeros_like(pos.v)
        np.add.at(full, np.concatenate([i.reshape(-1) for i in idx]),
                  np.concatenate([g.reshape(-1, g.shape[-1]) for g in grads]))
        pos._acc(full)

    @staticmethod
    def wrap(a: Node) -> Node:
        """Angles into ``(-pi, pi]`` (a shift by 2 pi, derivative 1)."""
        v = a.v
        out = np.where(v <= -math.pi, v + 2.0 * math.pi, np.where(v > math.pi, v - 2.0 * math.pi, v))
        return a.tape.op(out, a._acc)

    @staticmethod
    def reflect(a: Node) -> Node:
        """``-pi - a`` for negative ``a``, else ``pi - a`` (derivative -1)."""
        v = a.v
        return a.tape.op(np.where(v < 0, -math.pi - v, math.pi - v), lambda g: a._acc(-g))

    @staticmethod
    def norm(v: Node) -> Node:
        out = np.sqrt((v.v * v.v).sum(-1))
        return v.tape.op(out, lambda g: v._acc((g / out)[..., None] * v.v))

    @classmethod
    def angle(cls, pos: Node, i, j, k) -> Node:
        """Angles ``i-j-k``: ``acos`` of the cosine from the three squared distances (clamped to [-1, 1])."""
        x = pos.v
        pi, pj, pk = x[i], x[j], x[k]
        dij, djk, dik = pi - pj, pj - pk, pi - pk
        a, b, d = (dij * dij).sum(-1), (djk * djk).sum(-1), (dik * dik).sum(-1)
        s = np.sqrt(a * b)
        c = 0.5 * (a + b - d) / s
        cc = np.clip(c, -1.0, 1.0)

        def back(g):
            with np.errstate(divide="ignore", invalid="ignore"):
                gc = np.where((c >= -1.0) & (c <= 1.0), -g / np.sqrt(1.0 - cc * cc), 0.0)
            ga, gb, gd = gc * (0.5 / s - c / (2.0 * a)), gc * (0.5 / s - c / (2.0 * b)), -gc * 0.5 / s
            ta, tb, td = (2.0 * ga)[..., None] * dij, (2.0 * gb)[..., None] * djk, (2.0 * gd)[..., None] * dik
            cls._scatter(pos, (i, j, j, k, i, k), (ta, -ta, tb, -tb, td, -td))
        return pos.tape.op(np.arccos(cc), back)

    @classmethod
    def dihedral(cls, pos: Node, i, j, k, l) -> Node:
        """Dihedrals ``i-j-k-l`` in ``[0, 2 pi)`` from the projections on the ``j-k`` axis; 0 for projections
        shorter than 1e-6 and for angles within 4e-5 of zero."""
        x = pos.v
        pj, pk = x[j], x[k]
        b0, b1, b2 = x[i] - pj, pk - pj, x[l] - pk
        n1 = np.sqrt((b1 * b1).sum(-1))[..., None]
        axis = b1 / n1
        v = b0 - (b0 * axis).sum(-1)[..., None] * axis
        w = b2 - (b2 * axis).sum(-1)[..., None] * axis
        flat = (np.sqrt((v * v).sum(-1)) < 1e-6) | (np.sqrt((w * w).sum(-1)) < 1e-6)
        xx = (v * w).sum(-1)
        yy = (np.cross(axis, v) * w).sum(-1)
        phi = np.arctan2(np.where(flat, 0.0, yy), np.where(flat, 1.0, xx))
        zero = flat | (np.abs(phi) < 4e-5)
        out = np.where(zero, 0.0, np.where(phi < 0, phi + 2.0 * math.pi, phi))

        def back(g):
            # the closed form with F = b0, G = -b1, H = b2, A = F x G, B = H x G (0 for the zeroed angles)
            g = np.where(zero, 0.0, g)
            G = -b1
            nG = n1[..., 0]
            A, B = np.cross(b0, G), np.cross(b2, G)
            AA = np.where(flat, 1.0, (A * A).sum(-1))
            BB = np.where(flat, 1.0, (B * B).sum(-1))
            FG, HG = (b0 * G).sum(-1), (b2 * G).sum(-1)
            ta, tb = (g * nG / AA)[..., None] * A, (g * nG / BB)[..., None] * B
            ua, ub = (g * FG / (AA * nG))[..., None] * A, (g * HG / (BB * nG))[..., None] * B
            cls._scatter(pos, (i, j, k, l), (-ta, ta + ua - ub, -tb - ua + ub, tb))
        return pos.tape.op(out, back)

    @staticmethod
    def sqrt(x: Node) -> Node:
        out = np.sqrt(x.v)
        return x.tape.op(out, lambda g: x._acc(g / (2.0 * out)))

    @staticmethod
    def exp(x: Node) -> Node:
        out = np.exp(x.v)
        return x.tape.op(out, lambda g: x._acc(g * out))

    @staticmethod
    def cos(x: Node) -> Node:
        return x.tape.op(np.cos(x.v), lambda g: x._acc(-g * np.sin(x.v)))

    @staticmethod
    def acos(x: Node) -> Node:
        return x.tape.op(np.arccos(x.v), lambda g: x._acc(-g / np.sqrt(1.0 - x.v * x.v)))

    @staticmethod
    def atan2(y: Node, x: Node) -> Node:
        def back(g):
            r2 = x.v * x.v + y.v * y.v
            y._acc(g * x.v / r2)
            x._acc(-g * y.v / r2)
        return y.tape.op(np.arctan2(y.v, x.v), back)

    @staticmethod
    def abs(x: Node) -> Node:
        return x.tape.op(np.abs(x.v), lambda g: x._acc(g * np.sign(x.v)))

    @staticmethod
    def clamp(x: Node, lo=None, hi=None) -> Node:
        out = np.clip(x.v, lo, hi)
        inside = np.ones(x.v.shape, dtype=bool)
        if lo is not None:
            inside &= x.v >= lo
        if hi is not None:
            inside &= x.v <= hi
        return x.tape.op(out, lambda g: x._acc(np.where(inside, g, 0.0)))

    @staticmethod
    def _pick(mask, a, b, va, vb, tape):
        def back(g):
            if isinstance(a, Node):
                a._acc(_unbroadcast(np.where(mask, g, 0.0), a.v.shape))
            if isinstance(b, Node):
                b._acc(_unbroadcast(np.where(mask, 0.0, g), b.v.shape))
        return tape.op(np.where(mask, va, vb), back)

    @classmethod
    def where(cls, mask, a, b):
        tape = (a if isinstance(a, Node) else b).tape
        return cls._pick(np.asarray(mask), a, b, cls.value(a), cls.value(b), tape)

    @classmethod
    def maximum(cls, a: Node, b: Node) -> Node:
        # ties share the gradient, as PyTorch's
        va, vb = a.v, b.v

        def back(g):
            a._acc(_unbroadcast(np.where(va > vb, g, np.where(va == vb, 0.5 * g, 0.0)), va.shape))
            b._acc(_unbroadcast(np.where(vb > va, g, np.where(va == vb, 0.5 * g, 0.0)), vb.shape))
        return a.tape.op(np.maximum(va, vb), back)

    @classmethod
    def minimum(cls, a: Node, b: Node) -> Node:
        va, vb = a.v, b.v

        def back(g):
            a._acc(_unbroadcast(np.where(va < vb, g, np.where(va == vb, 0.5 * g, 0.0)), va.shape))
            b._acc(_unbroadcast(np.where(vb < va, g, np.where(va == vb, 0.5 * g, 0.0)), vb.shape))
        return a.tape.op(np.minimum(va, vb), back)

    @staticmethod
    def cross(a: Node, b: Node) -> Node:
        va, vb = a.v, b.v
        return a.tape.op(np.cross(va, vb), lambda g: (a._acc(np.cross(vb, g)), b._acc(np.cross(g, va))))

    @staticmethod
    def stack(xs: List[Node]) -> Node:
        def back(g):
            for i, x in enumerate(xs):
                x._acc(g[..., i])
        return xs[0].tape.op(np.stack([x.v for x in xs], -1), back)

    @staticmethod
    def index_add(n: int, idx: np.ndarray, e: Node) -> Node:
        out = np.zeros(n)
        np.add.at(out, idx, e.v)
        return e.tape.op(out, lambda g: e._acc(g[idx]))

    @staticmethod
    def host(x) -> "torch.Tensor":
        """The positions as a float64 CPU tensor (the input of the topology searches)."""
        return torch.from_numpy(np.ascontiguousarray(x.v if isinstance(x, Node) else x, dtype=np.float64))

    @staticmethod
    def vabs(x) -> np.ndarray:
        return np.abs(x)

    @staticmethod
    def longs(values, like=None) -> np.ndarray:
        return np.asarray(values, dtype=np.int64)

    @staticmethod
    def bools(values, like=None) -> np.ndarray:
        return np.asarray(values, dtype=bool)

    @staticmethod
    def array(a, like=None) -> np.ndarray:
        """A host array (indices, masks or constants) in this namespace."""
        return np.asarray(a)

    @staticmethod
    def vwhere(mask, a, b):
        """``where`` on values (no derivative)."""
        return np.where(mask, a, b)

    @staticmethod
    def nonzero(mask: np.ndarray) -> np.ndarray:
        return np.nonzero(mask)[0]


class TorchOps:
    """The same operations on PyTorch tensors (masks and indices are tensors); a TorchScript class, so that the
    terms written against it compile."""

    def __init__(self):
        self.pi = 3.141592653589793
        # tables from the host topology searches come as tensors
        self.host_arrays = False

    def value(self, x: Tensor) -> Tensor:
        return x.detach()

    def host(self, x: Tensor) -> Tensor:
        """The detached positions as a float64 tensor (the input of the topology searches)."""
        return x.detach().to(torch.float64)

    def take(self, x: Tensor, idx: Tensor) -> Tensor:
        return x[idx]

    def sum(self, x: Tensor) -> Tensor:
        return x.sum(-1)

    def norm(self, v: Tensor) -> Tensor:
        return torch.sqrt((v * v).sum(-1))

    def wrap(self, a: Tensor) -> Tensor:
        """Angles into ``(-pi, pi]``."""
        pi = 3.141592653589793
        return torch.where(a <= -pi, a + 2.0 * pi, torch.where(a > pi, a - 2.0 * pi, a))

    def reflect(self, a: Tensor) -> Tensor:
        """``-pi - a`` for negative ``a``, else ``pi - a``."""
        pi = 3.141592653589793
        return torch.where(a < 0, -pi - a, pi - a)

    def angle(self, pos: Tensor, i: Tensor, j: Tensor, k: Tensor) -> Tensor:
        """Angles ``i-j-k``: ``acos`` of the cosine from the three squared distances (clamped to [-1, 1])."""
        pi_, pj, pk = pos[i], pos[j], pos[k]
        dij, djk, dik = pi_ - pj, pj - pk, pi_ - pk
        a, b, d = (dij * dij).sum(-1), (djk * djk).sum(-1), (dik * dik).sum(-1)
        c = 0.5 * (a + b - d) / torch.sqrt(a * b)
        return torch.acos(c.clamp(-1.0, 1.0))

    def dihedral(self, pos: Tensor, i: Tensor, j: Tensor, k: Tensor, l: Tensor) -> Tensor:
        """Dihedrals ``i-j-k-l`` in ``[0, 2 pi)`` from the projections on the ``j-k`` axis; 0 for projections
        shorter than 1e-6 and for angles within 4e-5 of zero."""
        pj, pk = pos[j], pos[k]
        b0, b1, b2 = pos[i] - pj, pk - pj, pos[l] - pk
        axis = b1 / torch.sqrt((b1 * b1).sum(-1))[..., None]
        v = b0 - (b0 * axis).sum(-1)[..., None] * axis
        w = b2 - (b2 * axis).sum(-1)[..., None] * axis
        vv, wv = v.detach(), w.detach()
        flat = (torch.sqrt((vv * vv).sum(-1)) < 1e-6) | (torch.sqrt((wv * wv).sum(-1)) < 1e-6)
        x = (v * w).sum(-1)
        y = (torch.cross(axis, v, dim=-1) * w).sum(-1)
        # degenerate frames take a safe angle that the mask then zeroes
        phi = torch.atan2(torch.where(flat, torch.zeros_like(y), y), torch.where(flat, torch.ones_like(x), x))
        zero = flat | (phi.detach().abs() < 4e-5)
        phi = torch.where(phi < 0, phi + 2.0 * 3.141592653589793, phi)
        return torch.where(zero, torch.zeros_like(phi), phi)

    def sqrt(self, x: Tensor) -> Tensor:
        return torch.sqrt(x)

    def exp(self, x: Tensor) -> Tensor:
        return torch.exp(x)

    def cos(self, x: Tensor) -> Tensor:
        return torch.cos(x)

    def acos(self, x: Tensor) -> Tensor:
        return torch.acos(x)

    def atan2(self, y: Tensor, x: Tensor) -> Tensor:
        return torch.atan2(y, x)

    def abs(self, x: Tensor) -> Tensor:
        return torch.abs(x)

    def vabs(self, x: Tensor) -> Tensor:
        """``abs`` of values (no derivative)."""
        return torch.abs(x)

    def clamp(self, x: Tensor, lo: Optional[float] = None, hi: Optional[float] = None) -> Tensor:
        return x.clamp(lo, hi)

    def where(self, mask: Tensor, a: Union[float, Tensor], b: Union[float, Tensor]) -> Tensor:
        if isinstance(a, Tensor):
            if isinstance(b, Tensor):
                return torch.where(mask, a, b)
            return torch.where(mask, a, torch.full((), b, dtype=a.dtype, device=a.device))
        if isinstance(b, Tensor):
            return torch.where(mask, torch.full((), a, dtype=b.dtype, device=b.device), b)
        raise ValueError("where needs a tensor operand")

    def maximum(self, a: Tensor, b: Tensor) -> Tensor:
        return torch.maximum(a, b)

    def minimum(self, a: Tensor, b: Tensor) -> Tensor:
        return torch.minimum(a, b)

    def cross(self, a: Tensor, b: Tensor) -> Tensor:
        return torch.cross(a, b, dim=-1)

    def stack(self, xs: List[Tensor]) -> Tensor:
        return torch.stack(xs, -1)

    def index_add(self, n: int, idx: Tensor, e: Tensor) -> Tensor:
        return e.new_zeros(n).index_add(0, idx, e)

    def array(self, a: Tensor, like: Tensor) -> Tensor:
        """A table (indices, masks or constants) on the device of ``like`` (floats in its dtype)."""
        if a.is_floating_point():
            return a.to(device=like.device, dtype=like.dtype)
        return a.to(device=like.device)

    def longs(self, values: List[int], like: Tensor) -> Tensor:
        return torch.tensor(values, dtype=torch.long, device=like.device)

    def bools(self, values: List[bool], like: Tensor) -> Tensor:
        return torch.tensor(values, dtype=torch.bool, device=like.device)

    def vwhere(self, mask: Tensor, a: Union[float, Tensor], b: Union[float, Tensor]) -> Tensor:
        """``where`` on values (no derivative)."""
        return self.where(mask, a, b).detach()

    def nonzero(self, mask: Tensor) -> Tensor:
        return mask.nonzero().squeeze(-1)


TAPE = _TapeOps()
TORCH = TorchOps()


# below this many atoms the differentiable (PyTorch) form of a GPU tensor's term runs on the host, faster than
# the device's launches of its small operations (A100: 48 atoms 1.5 times, 122 level, 192 slower)
HOST_FORM_ATOMS = 100


class _HostTerm(torch.autograd.Function):
    """A term ``f(ops, positions) -> per-atom values`` evaluated on the tape, differentiable as a tensor."""

    @staticmethod
    def forward(ctx, pos: Tensor, term: Callable):
        tape = Tape()
        x = tape.leaf(pos.detach().to("cpu", torch.float64).numpy())
        out = term(TAPE, x)
        ctx.save_for_backward(pos)
        ctx.term, ctx.tape, ctx.x, ctx.out = term, tape, x, out
        value = out.v if isinstance(out, Node) else np.zeros(pos.shape[0])
        return torch.as_tensor(value, dtype=pos.dtype, device=pos.device)

    @staticmethod
    def backward(ctx, g: Tensor):
        pos, = ctx.saved_tensors
        if torch.is_grad_enabled():
            # a differentiable gradient (second derivatives): the PyTorch form of the term, for molecules up to
            # HOST_FORM_ATOMS atoms on the host (a few hundred operations on a handful of atoms), through
            # differentiable copies
            device = torch.device("cpu") if pos.shape[0] <= HOST_FORM_ATOMS else pos.device
            with torch.enable_grad():
                p = pos.to(device)
                out = ctx.term(TORCH, p)
                if not isinstance(out, Tensor) or not out.requires_grad:
                    return torch.zeros_like(pos), None
                return torch.autograd.grad(out, p, g.to(device), create_graph=True)[0].to(pos.device), None
        if not isinstance(ctx.out, Node):
            return torch.zeros_like(pos), None
        ctx.tape.backward(ctx.out, g.detach().to("cpu", torch.float64).numpy())
        grad = ctx.x.g if ctx.x.g is not None else np.zeros(pos.shape)
        ctx.tape.release()
        return torch.as_tensor(grad, dtype=pos.dtype, device=pos.device), None


def host_term(term: Callable, pos: Tensor) -> Tensor:
    """``term(ops, pos)`` (per-atom values ``(N,)``) on the host tape, for tensors on any device (a GPU tensor
    is copied once each way: these terms are a few hundred operations on a handful of atoms); higher derivatives
    from the PyTorch form (on the host below :data:`HOST_FORM_ATOMS` atoms)."""
    if not (torch.is_grad_enabled() and pos.requires_grad):
        tape = Tape()
        out = term(TAPE, tape.leaf(pos.detach().to("cpu", torch.float64).numpy()))
        value = out.v if isinstance(out, Node) else np.zeros(pos.shape[0])
        tape.release()
        return torch.as_tensor(value, dtype=pos.dtype, device=pos.device)
    return _HostTerm.apply(pos, term)
