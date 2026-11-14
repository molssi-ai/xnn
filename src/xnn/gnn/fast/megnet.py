"""Fast path of MEGNet: the encoders, the blocks and the Set2Set readouts as compiled stages.

The reference block concatenates ``[v_i, v_j, e_k, u]`` for every bond before the first
dense layer of the bond update. That layer is linear, so here its weight is split into
the four blocks of the concatenation: the atom and state parts are multiplied once per
atom and per structure and gathered to the bonds, and only the bond part is multiplied
per bond. The atom update is split the same way (its state part once per structure).
The results are the same to rounding, without the bond-sized concatenation and with a
quarter of the first layer's work on the bonds.

The Set2Set readouts write the LSTM cell out (the gate equations of
:class:`torch.nn.LSTM`), so that all their steps are one stage. On a GPU every stage is
compiled by :func:`torch.compile`, which fuses the activations, gathers and means into
few kernels. When a gradient is needed the stages are compiled with their first and
second derivatives (:func:`~xnn.common.models.fast.compile_twice_differentiable`); on a
CPU they run as written.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from xnn.common.models.fast import AutoPolicy, compile_twice_differentiable
from xnn.common.models.ops import gather_rows, scatter_sum, segment_softmax

#: Edge counts above which ``use_fast="auto"`` takes the stages in evaluation mode: every size, since
#: on an A100 the compiled stages and on a CPU the split layers (as written) were faster than the
#: reference from one 20-atom cell to one of 16384 atoms, in both precisions.
AUTO_POLICY = AutoPolicy(default_min_edges=0, default_min_edges_float64=0,
                         cpu_min_edges=0, cpu_min_edges_float64=0)

#: The same for training mode (a step with a property loss): every size, on an A100 with a third to
#: two thirds of the reference's memory.
AUTO_POLICY_TRAINING = AutoPolicy(default_min_edges=0, default_min_edges_float64=0,
                                  cpu_min_edges=0, cpu_min_edges_float64=0)


def _dense(x: Tensor, weights: Sequence[Tensor], biases: Sequence[Tensor], act: Callable,
           last: bool = True) -> Tensor:
    """Dense layers with ``act`` between them (and after the last with ``last``)."""
    n = len(weights)
    for i in range(n):
        x = F.linear(x, weights[i], biases[i])
        if i < n - 1 or last:
            x = act(x)
    return x


def _sum(src: Tensor, index: Tensor, n: int) -> Tensor:
    """Sum of the rows of ``src`` per bucket of ``index``.

    One bucket (a single structure) is a matrix product with a row of ones: compiled code would reduce
    the long row axis to a few outputs, which runs on few threads.
    """
    if n == 1:
        return src.new_ones((1, src.shape[0])) @ src
    return scatter_sum(src, index, n)


def _mean(src: Tensor, index: Tensor, count: Tensor) -> Tensor:
    """Mean of the rows of ``src`` per bucket; ``count`` is the clamped bucket size ``(n, 1)``."""
    return _sum(src, index, count.shape[0]) / count


def _max_rows(z: Tensor) -> Tensor:
    """The largest row of ``z`` ``(R, 1)`` as ``(1, 1)``, in blocks of 1024 rows and then over the blocks."""
    if z.shape[0] == 0:
        return z.new_zeros((1, 1))
    flat = F.pad(z.reshape(-1), (0, (-z.shape[0]) % 1024), value=float("-inf"))
    return flat.reshape(-1, 1024).amax(1).amax(0).reshape(1, 1)


def encode_function(n_edge: int, n_node: int, n_state: int, act: Callable) -> Callable:
    """``f(dist, x, u, *weights, centers, gamma)``: the Gaussian bond expansion and the three encoders.

    The weights are those of the edge, node and state encoders in order (weight, bias per layer);
    ``n_edge``, ``n_node`` and ``n_state`` are their numbers of layers.
    """
    def fn(dist: Tensor, x: Tensor, u: Tensor, *rest):
        k = 2 * (n_edge + n_node + n_state)
        p, (centers, gamma) = rest[:k], rest[k:]
        ew, nw, sw = p[:2 * n_edge], p[2 * n_edge:2 * (n_edge + n_node)], p[2 * (n_edge + n_node):]
        rbf = torch.exp(-gamma * (dist[:, None] - centers) ** 2)
        return (_dense(rbf, ew[0::2], ew[1::2], act), _dense(x, nw[0::2], nw[1::2], act),
                _dense(u, sw[0::2], sw[1::2], act))
    return fn


def block_function(n_dense: int, n_conv: int, residual: bool, act: Callable) -> Callable:
    """``f(e, x, u, *weights, dst, src, batch, edge_batch, per_atom, per_structure_bonds,
    per_structure_atoms)``: one MEGNet block with the first layers of the bond and atom updates split.

    The weights are the dense layers (bond, atom, state; ``n_dense`` layers each, none for the
    first block) and the update networks (bond, atom, state; ``n_conv`` layers each), as
    (weight, bias) per layer. The counts are clamped ``(n, 1)`` columns. With ``residual`` the
    block's inputs are added to its outputs.
    """
    def fn(e: Tensor, x: Tensor, u: Tensor, *rest):
        k = 6 * (n_dense + n_conv)
        p, (dst, src, batch, edge_batch, per_atom, per_bonds, per_atoms) = rest[:k], rest[k:]
        d, c = 2 * n_dense, 2 * n_conv
        e0, x0, u0 = e, x, u
        if n_dense:
            e = _dense(e, p[0:d:2], p[1:d:2], act)
            x = _dense(x, p[d:2 * d:2], p[d + 1:2 * d:2], act)
            u = _dense(u, p[2 * d:3 * d:2], p[2 * d + 1:3 * d:2], act)
        conv = p[3 * d:]
        ew, nw, sw = conv[:c], conv[c:2 * c], conv[2 * c:]
        w = x.shape[1]
        # bond update: [x_i, x_j, e, u] times W is x_i W_i + x_j W_j + e W_e + u W_u
        W = ew[0]
        h = (F.linear(e, W[:, 2 * w:3 * w]) + gather_rows(F.linear(x, W[:, :w]), dst)
             + gather_rows(F.linear(x, W[:, w:2 * w]), src) + gather_rows(F.linear(u, W[:, 3 * w:], ew[1]), edge_batch))
        e = _dense(act(h), ew[2::2], ew[3::2], act)
        o = e.shape[1]
        # atom update: [x, mean e, u]
        W = nw[0]
        h = (F.linear(x, W[:, :w]) + F.linear(_mean(e, dst, per_atom), W[:, w:w + o])
             + gather_rows(F.linear(u, W[:, w + o:], nw[1]), batch))
        x = _dense(act(h), nw[2::2], nw[3::2], act)
        u = _dense(torch.cat([u, _mean(e, edge_batch, per_bonds), _mean(x, batch, per_atoms)], dim=-1),
                   sw[0::2], sw[1::2], act)
        if residual:
            return e + e0, x + x0, u + u0
        return e, x, u
    return fn


def set2set_function(steps: int, layers: int) -> Callable:
    """``f(x, *weights, index, per_structure)``: Set2Set with the LSTM cell written out.

    The weights are those of :class:`torch.nn.LSTM` per layer (``weight_ih``, ``weight_hh``,
    ``bias_ih``, ``bias_hh``); ``per_structure`` is any ``(B, 1)`` column (its rows give ``B``).
    """
    def fn(x: Tensor, *rest):
        p, (index, per_structure) = rest[:4 * layers], rest[4 * layers:]
        b, f = per_structure.shape[0], x.shape[1]
        h = [x.new_zeros((b, f)) for _ in range(layers)]
        c = [x.new_zeros((b, f)) for _ in range(layers)]
        q_star = x.new_zeros((b, 2 * f))
        for _ in range(steps):
            inp = q_star
            for i in range(layers):
                w_ih, w_hh, b_ih, b_hh = p[4 * i:4 * i + 4]
                gate_i, gate_f, gate_g, gate_o = (F.linear(inp, w_ih, b_ih) + F.linear(h[i], w_hh, b_hh)).chunk(4, 1)
                c[i] = torch.sigmoid(gate_f) * c[i] + torch.sigmoid(gate_i) * torch.tanh(gate_g)
                h[i] = torch.sigmoid(gate_o) * torch.tanh(c[i])
                inp = h[i]
            z = (x * gather_rows(inp, index)).sum(-1, keepdim=True)
            if b == 1:
                # one structure: the same softmax with the reductions as matrix products (see _sum)
                ex = torch.exp(z - _max_rows(z.detach()))
                a = ex / (_sum(ex, index, 1) + 1e-16)
                r = a.transpose(0, 1) @ x
            else:
                a = segment_softmax(z, index, b)
                r = scatter_sum(a * x, index, b)
            q_star = torch.cat([inp, r], dim=-1)
        return q_star
    return fn


_compiled: Dict[tuple, Callable] = {}


def stage(kind: str, shape: tuple, act: Callable, n_inputs: int, n_constants: int, like: Tensor,
          grad: bool) -> Callable:
    """The stage ``kind`` (``"encode"``, ``"block"``, ``"set2set"``) of the given layer counts.

    For CUDA tensors compiled, with first and second derivatives when ``grad``; as written
    otherwise. ``n_inputs`` arguments are differentiated, ``n_constants`` follow.
    """
    if kind == "encode":
        fn = encode_function(*shape, act)
    elif kind == "block":
        fn = block_function(*shape, act)
    else:
        fn = set2set_function(*shape)
    if not like.is_cuda:
        return fn
    key = (kind, shape, type(act).__name__, like.dtype, grad)
    out = _compiled.get(key)
    if out is None:
        out = _compiled[key] = (compile_twice_differentiable(fn, n_inputs, n_constants) if grad
                                else torch.compile(fn, dynamic=True))
    return out


def linear_params(mlps: Sequence) -> List[Tensor]:
    """The (weight, bias) of every layer of the given MLPs, in order."""
    out: List[Tensor] = []
    for mlp in mlps:
        for layer in mlp.layers:
            out += [layer.weight, layer.bias]
    return out


def plain_mlp(mlp) -> bool:
    """Whether every layer of ``mlp`` is a plain :class:`torch.nn.Linear` with a bias (not wrapped, e.g.
    by LoRA)."""
    return all(type(layer) is torch.nn.Linear and layer.bias is not None for layer in mlp.layers)


__all__ = ["AUTO_POLICY", "AUTO_POLICY_TRAINING", "encode_function", "block_function", "set2set_function",
           "stage", "linear_params", "plain_mlp"]
