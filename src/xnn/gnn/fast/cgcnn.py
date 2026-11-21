"""Fast path of CGCNN: the network from the positions to the pooled atom features as one compiled stage.

The stage forms the bond vectors and lengths from the positions and the cell, their Gaussian
expansion, the element rows and their dense layer, every convolution and the mean over each
crystal. In a convolution the reference concatenates ``[v_i, v_j, u_ij]`` for every bond before
the first dense layer. That layer is linear, so here its weight is split into the three blocks of
the concatenation: the parts of the atom and of its neighbor are multiplied once per atom (the
neighbor's then gathered to the bonds), and only the Gaussian bond features are multiplied per
bond. The batch normalizations, the gate, the neighbor sum, the residual and the softplus follow.
The results are the same to rounding, with a quarter of the first layer's work on the bonds and
one compiled call in place of some fifty operations.

The stage serves graphs in which every atom keeps the same number of neighbors, grouped by atom
(the neighbor table of the authors' layout, as the selection of the model and
:func:`~xnn.common.data.structure_to_graph` with ``max_neighbors`` produce); other graphs take the
reference. In training mode the stage normalizes with the statistics of the batch and returns
them, and the running statistics are updated outside it, as :class:`torch.nn.BatchNorm1d` updates
them. The dense layers after the pooling stay outside the stage (their dropout draws random masks).
On a GPU the stage is compiled by :func:`torch.compile`; when a gradient is needed its compiled
backward gives the first derivatives and a backward that is itself differentiated gives the second
(:func:`~xnn.common.models.fast.compile_differentiable`). On a CPU it runs as written.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from xnn.common.models.fast import AutoPolicy, compile_differentiable
from xnn.common.models.ops import exact_softplus

#: Edge counts above which ``use_fast="auto"`` takes the stage in evaluation mode (never on a CPU, where
#: it runs as written). A GPU without an entry takes the A100's values.
AUTO_POLICY = AutoPolicy(min_edges={"A100": 0}, min_edges_float64={"A100": 0}, default_min_edges=0,
                         default_min_edges_float64=0)

#: The same for training mode (a step with a property loss).
AUTO_POLICY_TRAINING = AutoPolicy(min_edges={"A100": 16_384}, min_edges_float64={"A100": 16_384},
                                  default_min_edges=16_384, default_min_edges_float64=16_384)


def _normalize(h: Tensor, weight: Tensor, bias: Tensor, eps: float):
    """Batch normalization of the rows of ``h`` with their own statistics; also returns the mean and the
    (biased) variance."""
    # the column sums as products with a row of ones: compiled code reduces many rows to few outputs slowly
    ones = h.new_ones((1, h.shape[0]))
    mean = (ones @ h).squeeze(0) / h.shape[0]
    var = (ones @ (h - mean).square()).squeeze(0) / h.shape[0]
    return (h - mean) * torch.rsqrt(var + eps) * weight + bias, mean, var


def network_function(n_conv: int, normalization: bool, training: bool, residual_activation: bool,
                     gamma: float, eps: float, pos_input: bool, periodic: bool, single: bool) -> Callable:
    """The network up to the pooling as one function of its tensors.

    ``f([pos,] emb_w, emb_b, *conv_params, [pos,] cell, shifts, src, table, z, centers, batch, n_atoms,
    *running)``. The differentiated inputs come first: the positions (with ``pos_input``), the element
    dense layer and, per convolution, its first dense layer and (with ``normalization``) the affine
    parameters of its two batch normalizations. The constants follow: the positions (without
    ``pos_input``), the cells ``(B, 3, 3)`` (used with ``periodic``), the integer image shifts of the
    bonds, the neighbor of every bond (``m`` per atom, grouped by atom), the element table and atomic
    numbers, the Gaussian centers, the crystal of every atom, the atoms per crystal and, in evaluation
    mode with ``normalization``, four running statistics per convolution. Returns the mean atom
    features of every crystal, and in training mode with ``normalization`` the batch means and
    variances of every batch normalization.
    """
    per_conv = 6 if normalization else 2

    def fn(*args):
        i = 1 if pos_input else 0
        emb_w, emb_b = args[i], args[i + 1]
        i += 2
        params = [args[i + k * per_conv:i + (k + 1) * per_conv] for k in range(n_conv)]
        i += n_conv * per_conv
        pos = args[0] if pos_input else args[i]
        i += 0 if pos_input else 1
        cell, shifts, src, table, z, centers, batch, n_atoms = args[i:i + 8]
        running = args[i + 8:]
        n = pos.shape[0]
        vec = pos.unsqueeze(1) - pos.index_select(0, src).view(n, -1, 3)
        if periodic:
            vec = vec + shifts.view(n, -1, 3).to(pos.dtype) @ cell.index_select(0, batch).to(pos.dtype)
        dist = vec.to(emb_w.dtype).norm(dim=-1).reshape(-1)
        u = torch.exp(-gamma * (dist.unsqueeze(-1) - centers) ** 2)
        x = F.linear(table.index_select(0, z), emb_w, emb_b)
        f = x.shape[1]
        stats: List[Tensor] = []
        for k in range(n_conv):
            w, b = params[k][0], params[k][1]
            atoms = F.linear(x, torch.cat([w[:, :f], w[:, f:2 * f]]))
            h = (F.linear(u, w[:, 2 * f:], b).view(n, -1, 2 * f) + atoms[:, :2 * f].unsqueeze(1)
                 + atoms[:, 2 * f:].index_select(0, src).view(n, -1, 2 * f)).view(-1, 2 * f)
            if normalization:
                w1, b1, w2, b2 = params[k][2:]
                if training:
                    h, mean1, var1 = _normalize(h, w1, b1, eps)
                else:
                    h = (h - running[4 * k]) * torch.rsqrt(running[4 * k + 1] + eps) * w1 + b1
            gate, core = h.chunk(2, dim=-1)
            s = (torch.sigmoid(gate) * exact_softplus(core)).view(n, -1, f).sum(1)
            if normalization:
                if training:
                    s, mean2, var2 = _normalize(s, w2, b2, eps)
                    stats += [mean1, var1, mean2, var2]
                else:
                    s = (s - running[4 * k + 2]) * torch.rsqrt(running[4 * k + 3] + eps) * w2 + b2
            x = x + s
            if residual_activation:
                x = exact_softplus(x)
        if single:
            pooled = x.mean(0, keepdim=True)
        else:
            pooled = (x.new_zeros((n_atoms.shape[0], f)).index_add(0, batch, x)
                      / n_atoms.clamp(min=1).unsqueeze(-1).to(x.dtype))
        return (pooled, *stats) if stats else pooled
    return fn


_compiled: Dict[tuple, Callable] = {}


def size_class(num_edges: int) -> int:
    """The size class of a graph: 0 below 2^13 edges, 1 below 2^16, 2 above."""
    return 0 if num_edges < 2 ** 13 else (1 if num_edges < 2 ** 16 else 2)


def mark_sizes(edges: Sequence[Tensor], atoms: Sequence[Tensor]) -> None:
    """Give the compiler the representative sizes of the size class of a graph, on the first dimension of
    every edge-sized and every atom-sized input of the stage: it tunes the reductions of a stage for these sizes
    (and keeps the versions of the classes apart in its cache), not for those of its first call, so that a
    stage first called on a few hundred bonds still reduces hundreds of thousands quickly."""
    e, n = edges[0].shape[0], atoms[0].shape[0]
    if e < 2 or n < 2:      # sizes 0 and 1 are specialized, not marked
        return
    # the hints keep the edges per atom of the graph exactly: the compiler traces the shapes at them
    per_atom = max(1, e // n)
    atom_hint = max(2, 2 ** (12 + 3 * size_class(e)) // per_atom)
    try:
        for t in edges:
            torch._dynamo.mark_dynamic(t, 0, hint_override=atom_hint * per_atom)
        for t in atoms:
            torch._dynamo.mark_dynamic(t, 0, hint_override=atom_hint)
    except TypeError:       # a torch without size hints
        pass


def stage(key: tuple, num_edges: int, n_inputs: int, n_constants: int, like: Tensor, grad: bool) -> Callable:
    """The network stage of ``key`` (the arguments of :func:`network_function`) for a graph of ``num_edges``.

    For CUDA tensors compiled, with a fast first derivative and a second one on request when ``grad``
    (:func:`~xnn.common.models.fast.compile_differentiable`); as written otherwise. ``n_inputs``
    arguments are differentiated, ``n_constants`` follow. Each :func:`size_class` is compiled on its
    own, for the sizes :func:`mark_sizes` gives the compiler.
    """
    fn = network_function(*key)
    if not like.is_cuda:
        return fn
    full = (*key, size_class(num_edges), like.dtype, grad)
    out = _compiled.get(full)
    if out is None:
        # a code object of its own, so that the compiler keeps its versions apart
        name = f"cgcnn_stage_{len(_compiled)}"
        fn.__code__ = fn.__code__.replace(co_name=name)
        fn.__name__ = name
        out = _compiled[full] = (compile_differentiable(fn, n_inputs, n_constants) if grad
                                 else torch.compile(fn, dynamic=True))
    return out


@torch.no_grad()
def update_running_statistics(norm: torch.nn.BatchNorm1d, mean: Tensor, var: Tensor, rows: int) -> None:
    """Update the running statistics of ``norm`` from the batch ``mean`` and biased ``var`` of ``rows`` rows,
    as :class:`torch.nn.BatchNorm1d` does in training mode."""
    norm.num_batches_tracked.add_(1)
    m = norm.momentum
    norm.running_mean.mul_(1 - m).add_(mean, alpha=m)
    norm.running_var.mul_(1 - m).add_(var * (rows / max(rows - 1, 1)), alpha=m)


__all__ = ["AUTO_POLICY", "AUTO_POLICY_TRAINING", "mark_sizes", "network_function", "size_class", "stage",
           "update_running_statistics"]
