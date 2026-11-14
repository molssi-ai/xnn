"""The message step of an equivariant convolution, with a fused fast path.

Equivariant message passing (MACE, NequIP and their relatives) spends most of
its time in one step: gather the sender features onto the edges, take the
``uvu`` tensor product with the edge spherical harmonics weighted per edge by a
radial network, and sum the messages onto the receivers. The reference does the
three as separate operations and keeps the per-edge intermediates for the
backward pass; the fast path runs them as one cuEquivariance kernel that reads
the node features and writes the node sums directly.
"""
# NOTE: no `from __future__ import annotations` here: ConvTensorProduct.conv is
# compiled by torch.jit.script, which cannot resolve stringified annotations.
from typing import Callable, Dict, List, Optional, Tuple

import torch
from e3nn import o3
from torch import Tensor

from xnn.common.models.fast import FastPathModule
from xnn.common.models.ops import gather_rows, scatter_sum

from . import _cueq


class ConvTensorProduct(o3.TensorProduct, FastPathModule):
    """``uvu`` tensor product with per-edge weights and a fused message pass.

    An :class:`e3nn.o3.TensorProduct` (same arguments, parameters and
    ``state_dict``) with one more method, :meth:`conv`, which evaluates the
    whole message step. Called as a module it is the plain e3nn tensor product.

    The fast path applies when every instruction is ``uvu``, the second input
    has multiplicity one throughout (spherical harmonics) and the weights are
    per edge (``shared_weights=False``), which is the convolution of MACE and
    NequIP. It is exact up to the summation order.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fast_active = False
        self.fast_eligible = (
            not self.shared_weights and not self.internal_weights
            and all(ins.connection_mode == "uvu" for ins in self.instructions)
            and all(mul == 1 for mul, _ in self.irreps_in2)
            and len({ins.i_out for ins in self.instructions}) == len(self.instructions)
            and len(self.instructions) == len(self.irreps_out)
        )
        # built on first use per (device, dtype); kept out of the state_dict
        self.__dict__["_fast_kernels"] = {}

    def fast_supported(self, device: torch.device, dtype: torch.dtype) -> bool:
        return self.fast_eligible and _cueq.supported(device, dtype)

    def weight_map(self, like: Tensor, message_divisor: Optional[float] = None
                   ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """The column map of the per-edge weights on the fast path.

        The fused kernel takes the weights in its own column order and scale
        (and the message divisor folded in). Passed to the radial network
        (:class:`~xnn.gnn.models.blocks.RadialNet`), the map is applied once to
        its last layer instead of every edge, and :meth:`conv` is then called
        with ``weights_mapped=True``.

        Parameters
        ----------
        like : torch.Tensor
            Any tensor on the device and in the dtype of the evaluation.
        message_divisor : float, optional
            As in :meth:`conv`.

        Returns
        -------
        tuple of (torch.Tensor or None, torch.Tensor or None)
            ``(index, scale)`` on the fast path; ``(None, None)`` on the
            reference path, where the weights are used as they are.
        """
        if self.fast_active and not torch.jit.is_scripting():
            return self._fast_weight_map(like.device, like.dtype, message_divisor)
        return None, None

    @torch.jit.unused
    def _fast_weight_map(self, device: torch.device, dtype: torch.dtype,
                         message_divisor: Optional[float]) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        try:
            kernel = self._kernel(device, dtype)
        except Exception:  # conv falls back to the reference and says so
            return None, None
        scale = kernel["weight_scale"]
        if message_divisor is not None:
            scale = scale / message_divisor
        return kernel["weight_index"], scale

    def conv(self, node_feats: Tensor, edge_attrs: Tensor, edge_weights: Tensor,
             edge_index: Tensor, num_nodes: int,
             message_divisor: Optional[float] = None, weights_mapped: bool = False,
             weight_index: Optional[Tensor] = None) -> Tensor:
        """Gather, tensor product and sum onto the receivers in one call.

        Parameters
        ----------
        node_feats : torch.Tensor
            Node features, ``(N, irreps_in1.dim)``.
        edge_attrs : torch.Tensor
            Edge attributes (spherical harmonics), ``(E, irreps_in2.dim)``.
        edge_weights : torch.Tensor
            Per-edge path weights, ``(E, weight_numel)``.
        edge_index : torch.Tensor
            ``(2, E)``: row 0 the senders, row 1 the receivers.
        num_nodes : int
            ``N``.
        message_divisor : float, optional
            Divide the summed messages by this (NequIP's
            ``sqrt(avg_num_neighbors)``); the reference divides the node sums,
            one pass over the nodes instead of the edges, and the fast path
            folds it into the weights.
        weights_mapped : bool, optional
            ``edge_weights`` already carry the column map of
            :meth:`weight_map` (divisor included). Default ``False``.
        weight_index : torch.Tensor, optional
            ``(E,)``: edge ``e`` takes the weights of row ``weight_index[e]`` (the two directions of a
            bond share theirs, :func:`edge_pairs`); by default row ``e``.

        Returns
        -------
        torch.Tensor
            The summed messages, ``(N, irreps_out.dim)``.
        """
        if self.fast_active:
            if weights_mapped:
                return self._conv_fast(node_feats, edge_attrs, edge_weights, edge_index, num_nodes, True,
                                       weight_index)
            if message_divisor is not None:
                edge_weights = edge_weights / message_divisor
            # every argument given: an unused method keeps no defaults under TorchScript
            return self._conv_fast(node_feats, edge_attrs, edge_weights, edge_index, num_nodes, False,
                                   weight_index)
        if weight_index is not None:
            edge_weights = edge_weights.index_select(0, weight_index)
        messages = self(gather_rows(node_feats, edge_index[0]), edge_attrs, edge_weights)
        out = scatter_sum(messages, edge_index[1], num_nodes)
        if message_divisor is not None:
            out = out / message_divisor
        return out

    @torch.jit.unused
    def conv_blocks(self, node_feats: Tensor, edge_attrs: Tensor, edge_feats: Tensor, edge_index: Tensor,
                    num_nodes: int, block: int, weights: Callable[[Tensor], Tensor],
                    weights_mapped: bool = False, recompute: bool = False) -> Tensor:
        """:meth:`conv` over blocks of edges, the per-edge weights made per block from ``edge_feats``.

        The weights of one block exist at a time. On the fast path the node features are brought to
        the kernel's layout once and the blocks' node sums are added in that layout, then brought
        back once, and the edge inputs are split once, so that a block costs work over its edges
        only, besides one sum over the nodes.

        Parameters
        ----------
        node_feats, edge_attrs, edge_index, num_nodes
            As in :meth:`conv`.
        edge_feats : torch.Tensor
            The per-edge input of ``weights``, ``(E, ...)``.
        block : int
            Edges per block.
        weights : callable
            ``weights(edge_feats[a:b])`` is the per-edge weights of edges ``a:b``.
        weights_mapped : bool, optional
            As in :meth:`conv`.
        recompute : bool, optional
            Keep no block's weights for the backward: each block runs again there.

        Returns
        -------
        torch.Tensor
            The summed messages, ``(N, irreps_out.dim)``, equal to :meth:`conv` on all edges up to
            the order of the sums.
        """
        from torch.utils.checkpoint import checkpoint

        kernel = self._split_kernel(node_feats) if self.fast_active else None
        if kernel is None:
            def part(x: Tensor, a: Tensor, e: Tensor, idx: Tensor) -> Tensor:
                return self.conv(x, a, weights(e), idx, num_nodes, weights_mapped=weights_mapped)
            x = node_feats
        else:
            op = kernel["op"]

            def part(x: Tensor, a: Tensor, e: Tensor, idx: Tensor) -> Tensor:
                w = weights(e)
                if not weights_mapped:
                    if kernel["weight_index"] is not None:
                        w = w.index_select(1, kernel["weight_index"])
                    w = w * kernel["weight_scale"]
                return op.f([w, x, op.transpose_in2(a)], input_indices={1: idx[0]},
                            output_shapes={0: x.new_empty(num_nodes, 1)}, output_indices={0: idx[1]})[0]
            x = op.transpose_in1(node_feats)
        out: Optional[Tensor] = None
        # split once: the gradients of the parts are joined once, where slices would each fill and add
        # a gradient as large as all edges
        for a, e, idx in zip(edge_attrs.split(block), edge_feats.split(block), edge_index.split(block, dim=1)):
            y = checkpoint(part, x, a, e, idx, use_reentrant=False) if recompute else part(x, a, e, idx)
            out = y if out is None else out + y
        if kernel is not None:
            out = kernel["op"].transpose_out(out)
            if kernel["out_index"] is not None:
                out = out.index_select(1, kernel["out_index"])
        return out

    @torch.jit.unused
    def _split_kernel(self, like: Tensor) -> Optional[Dict[str, Optional[Tensor]]]:
        """The fast kernel when its layout changes and its product can be called apart, else ``None``."""
        try:
            kernel = self._kernel(like.device, like.dtype)
        except Exception:  # conv falls back to the reference and says so
            return None
        op = kernel["op"]
        parts = ("transpose_in1", "transpose_in2", "transpose_out", "f")
        if getattr(op, "weight", None) is not None or not all(hasattr(op, a) for a in parts):
            return None
        return kernel

    @torch.jit.unused
    def indexes_weights(self, like: Tensor) -> bool:
        """Whether :meth:`conv` reads ``weight_index`` inside the fast kernel; otherwise it gathers
        the weights per edge first."""
        return self.fast_active and self._split_kernel(like) is not None

    @torch.jit.unused
    def _conv_fast(self, node_feats: Tensor, edge_attrs: Tensor, edge_weights: Tensor,
                   edge_index: Tensor, num_nodes: int, mapped: bool = False,
                   weight_index: Optional[Tensor] = None) -> Tensor:
        try:
            kernel = self._kernel(node_feats.device, node_feats.dtype)
        except Exception as exc:  # an irreps layout the kernel does not cover
            self.fast_eligible = False
            self.fast_active = False
            _cueq.warn_fallback(type(self).__name__, exc)
            if weight_index is not None:
                edge_weights = edge_weights.index_select(0, weight_index)
            messages = self(gather_rows(node_feats, edge_index[0]), edge_attrs, edge_weights)
            return scatter_sum(messages, edge_index[1], num_nodes)
        w = edge_weights
        if not mapped:
            if kernel["weight_index"] is not None:
                w = w.index_select(1, kernel["weight_index"])
            w = w * kernel["weight_scale"]
        out = _run_channelwise(kernel["op"], node_feats, edge_attrs, w, edge_index[0], edge_index[1], num_nodes,
                               weight_index)
        if kernel["out_index"] is not None:
            out = out.index_select(1, kernel["out_index"])
        return out

    @torch.jit.unused
    def _kernel(self, device: torch.device, dtype: torch.dtype) -> Dict[str, Optional[Tensor]]:
        key = f"{device}|{dtype}"
        cache = self.__dict__["_fast_kernels"]
        if key not in cache:
            maps = self._weight_maps(device)
            op = _channelwise(self.irreps_in1, self.irreps_in2, self.irreps_out, device, dtype)
            cache[key] = {"op": op, **{k: (None if v is None else v.to(device=device, dtype=(
                dtype if k == "weight_scale" else v.dtype))) for k, v in maps.items()}}
        return cache[key]

    @torch.jit.unused
    def _weight_maps(self, device: torch.device) -> Dict[str, Optional[Tensor]]:
        """The maps from these weights and outputs to the kernel's, in float64.

        The kernel orders its paths by (first input, second input, output
        irrep) and its outputs sorted by irrep, and normalizes each path on its
        own: path ``k`` here is path ``j`` there up to a constant factor. The
        correspondence comes from the irreps, the factors from one evaluation
        of both on random inputs (exact: one path per output segment), on
        ``device`` (the kernel's layout transposes are CUDA-only).
        """
        cache = self.__dict__["_fast_kernels"]
        if "maps" in cache:
            return cache["maps"]
        import cuequivariance as cue

        poly = cue.descriptors.channelwise_tensor_product(
            _cueq.cue_irreps(self.irreps_in1), _cueq.cue_irreps(self.irreps_in2),
            str(self.irreps_out))
        stp = poly.polynomial.operations[0][1]
        fast_out = [(mul, str(ir), ir.dim) for mul, ir in poly.outputs[0].irreps]
        # the kernel's paths: weight segment, first and second input, output segment
        fast_paths = [tuple(path.indices) for path in stp.paths]
        w_sizes = [int(torch.tensor(seg).prod()) for seg in stp.operands[0].segments]
        w_offsets = [0]
        for size in w_sizes:
            w_offsets.append(w_offsets[-1] + size)
        out_dims = [mul * dim for mul, _, dim in fast_out]
        out_offsets = [0]
        for size in out_dims:
            out_offsets.append(out_offsets[-1] + size)
        by_inputs: Dict[tuple, List[tuple]] = {}
        for p in fast_paths:
            by_inputs.setdefault((p[1], p[2], fast_out[p[3]][1]), []).append(p)

        ref_offsets = [0]
        for ins in self.instructions:
            ref_offsets.append(ref_offsets[-1] + int(torch.tensor(ins.path_shape).prod()))
        weight_index = torch.zeros(w_offsets[-1], dtype=torch.long)
        out_index = torch.zeros(self.irreps_out.dim, dtype=torch.long)
        path_of_fast: Dict[int, int] = {}
        ref_out_offsets = [0]
        for mul, ir in self.irreps_out:
            ref_out_offsets.append(ref_out_offsets[-1] + mul * ir.dim)
        for k, ins in enumerate(self.instructions):
            key = (ins.i_in1, ins.i_in2, str(self.irreps_out[ins.i_out].ir))
            if not by_inputs.get(key):
                raise RuntimeError(f"no kernel path for instruction {key}")
            w_seg, _, _, o_seg = by_inputs[key].pop(0)
            size = ref_offsets[k + 1] - ref_offsets[k]
            weight_index[w_offsets[w_seg]:w_offsets[w_seg] + size] = torch.arange(
                ref_offsets[k], ref_offsets[k + 1])
            o_ref = ref_out_offsets[ins.i_out]
            o_dim = ref_out_offsets[ins.i_out + 1] - o_ref
            out_index[o_ref:o_ref + o_dim] = torch.arange(out_offsets[o_seg], out_offsets[o_seg] + o_dim)
            path_of_fast[w_seg] = k
        if any(by_inputs.values()):
            raise RuntimeError("the kernel has paths the reference does not")

        # the per-path factors: both evaluated on the same random inputs in float64
        # an exact float64 reference: the same paths built afresh (a copy of a
        # float32 module would carry float32-rounded coupling coefficients),
        # with this module's own path normalization carried over below
        default = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            ref = o3.TensorProduct(
                self.irreps_in1, self.irreps_in2, self.irreps_out,
                [(i.i_in1, i.i_in2, i.i_out, i.connection_mode, i.has_weight)
                 for i in self.instructions],
                shared_weights=False, internal_weights=False).to(device)
        finally:
            torch.set_default_dtype(default)
        path_ratio = [mine.path_weight / fresh.path_weight
                      for mine, fresh in zip(self.instructions, ref.instructions)]
        gen = torch.Generator().manual_seed(0)
        n = 7
        f64 = dict(dtype=torch.float64)
        x1 = torch.randn(n, self.irreps_in1.dim, generator=gen, **f64).to(device)
        x2 = torch.randn(n, self.irreps_in2.dim, generator=gen, **f64).to(device)
        w_ref = torch.randn(n, self.weight_numel, generator=gen, **f64).to(device)
        out_ref = ref(x1, x2, w_ref)
        op = _channelwise(self.irreps_in1, self.irreps_in2, self.irreps_out, device, torch.float64)
        weight_index, out_index = weight_index.to(device), out_index.to(device)
        out_fast = op(x1, x2, w_ref.index_select(1, weight_index)).index_select(1, out_index)
        weight_scale = torch.ones(w_offsets[-1], dtype=torch.float64, device=device)
        for w_seg, k in path_of_fast.items():
            o = self.instructions[k].i_out
            sl = slice(ref_out_offsets[o], ref_out_offsets[o + 1])
            r, f = out_ref[:, sl], out_fast[:, sl]
            weight_scale[w_offsets[w_seg]:w_offsets[w_seg + 1]] = float((r * f).sum() / (f * f).sum())
        out_check = op(x1, x2, w_ref.index_select(1, weight_index) * weight_scale).index_select(1, out_index)
        _cueq.check_close(out_ref, out_check, "convolution tensor product")
        for w_seg, k in path_of_fast.items():
            weight_scale[w_offsets[w_seg]:w_offsets[w_seg + 1]] *= path_ratio[k]

        identity_w = bool(torch.equal(weight_index.cpu(), torch.arange(len(weight_index))))
        identity_o = bool(torch.equal(out_index.cpu(), torch.arange(len(out_index))))
        maps = {"weight_index": None if identity_w else weight_index,
                "weight_scale": weight_scale,
                "out_index": None if identity_o else out_index}
        cache["maps"] = maps
        return maps


def edge_pairs(edge_index: Tensor, edge_vec: Tensor) -> Optional[Tuple[Tensor, Tensor]]:
    """The two directions of every bond, when each edge has its reverse.

    Edge ``s -> d`` with vector ``r`` and edge ``d -> s`` with vector ``-r`` (the same periodic
    image seen from the other atom) are one bond: a radial network gives both the same weights.

    Parameters
    ----------
    edge_index : torch.Tensor
        ``(2, E)``: senders and receivers.
    edge_vec : torch.Tensor
        ``(E, 3)``: the edge vectors.

    Returns
    -------
    tuple of (torch.Tensor, torch.Tensor) or None
        ``(rows, weight_index)``: one edge of each bond ``(E / 2,)``, and the bond of every edge
        ``(E,)``; ``None`` when some edge has no reverse (a half or an asymmetric list).
    """
    n = int(edge_index.shape[1])
    if n == 0 or n % 2:
        return None
    src, dst = edge_index[0], edge_index[1]
    v = edge_vec.detach()
    lead = torch.where(v[:, 0] != 0, v[:, 0], torch.where(v[:, 1] != 0, v[:, 1], v[:, 2]))
    first = (src < dst) | ((src == dst) & (lead > 0))
    lo, hi = torch.minimum(src, dst), torch.maximum(src, dst)
    canonical = torch.where(first[:, None], v, -v)
    # the bonds in order of (lo, hi, canonical vector): stable sorts, least significant key first
    order = torch.arange(n, device=v.device)
    for key in (canonical[:, 2], canonical[:, 1], canonical[:, 0], hi, lo):
        order = order.index_select(0, torch.sort(key.index_select(0, order), stable=True).indices)
    a, b = order[0::2], order[1::2]
    paired = ((lo[a] == lo[b]) & (hi[a] == hi[b]) & (canonical[a] == canonical[b]).all(1)
              & (first[a] != first[b]))
    if not bool(paired.all()):
        return None
    bond = torch.arange(n // 2, device=v.device)
    weight_index = torch.empty(n, dtype=torch.long, device=v.device)
    weight_index[a] = bond
    weight_index[b] = bond
    return torch.where(first[a], a, b), weight_index


def _run_channelwise(op, x1: Tensor, x2: Tensor, weight: Tensor, indices_1: Tensor,
                     indices_out: Tensor, size_out: int, weight_index: Optional[Tensor] = None) -> Tensor:
    """``op(x1, x2, weight, indices_1=..., indices_out=..., size_out=...)`` without a host sync.

    The channel-wise operation of cuEquivariance builds its output shape as a
    CPU tensor copied to the device on every call, which waits for the GPU;
    here the shape tensor is made on the device. With ``weight_index`` edge ``e`` reads
    the weights of row ``weight_index[e]``. Operations without the parts used (other
    cuEquivariance versions) are called as they are, on the weights gathered per edge.
    """
    parts = ("transpose_in1", "transpose_in2", "transpose_out", "f")
    if getattr(op, "weight", None) is not None or not all(hasattr(op, a) for a in parts):
        if weight_index is not None:
            weight = weight.index_select(0, weight_index)
        return op(x1, x2, weight, indices_1=indices_1, indices_out=indices_out, size_out=size_out)
    indices = {1: indices_1} if weight_index is None else {0: weight_index, 1: indices_1}
    out = op.f([weight, op.transpose_in1(x1), op.transpose_in2(x2)], input_indices=indices,
               output_shapes={0: x1.new_empty(size_out, 1)}, output_indices={0: indices_out})
    return op.transpose_out(out[0])


def _channelwise(irreps_in1, irreps_in2, irreps_out, device, dtype, method=None):
    """The cuEquivariance channelwise tensor product in e3nn's (mul, ir) layout."""
    import cuequivariance as cue
    import cuequivariance_torch as cuet

    return cuet.ChannelWiseTensorProduct(
        _cueq.cue_irreps(irreps_in1), _cueq.cue_irreps(irreps_in2), str(o3.Irreps(irreps_out)),
        layout=cue.mul_ir, shared_weights=False, internal_weights=False,
        device=device, dtype=dtype, math_dtype=dtype, method=method)
