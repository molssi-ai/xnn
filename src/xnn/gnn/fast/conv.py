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
from typing import Dict, List, Optional

import torch
from torch import Tensor
from e3nn import o3

from xnn.common.models.fast import FastPathModule
from xnn.common.models.ops import scatter_sum

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

    def conv(self, node_feats: Tensor, edge_attrs: Tensor, edge_weights: Tensor,
             edge_index: Tensor, num_nodes: int,
             message_divisor: Optional[float] = None) -> Tensor:
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
            Divide every message by this before summing (NequIP's
            ``sqrt(avg_num_neighbors)``); the fast path folds it into the
            weights.

        Returns
        -------
        torch.Tensor
            The summed messages, ``(N, irreps_out.dim)``.
        """
        if self.fast_active:
            if message_divisor is not None:
                edge_weights = edge_weights / message_divisor
            return self._conv_fast(node_feats, edge_attrs, edge_weights, edge_index, num_nodes)
        messages = self(node_feats[edge_index[0]], edge_attrs, edge_weights)
        if message_divisor is not None:
            messages = messages / message_divisor
        return scatter_sum(messages, edge_index[1], num_nodes)

    @torch.jit.unused
    def _conv_fast(self, node_feats: Tensor, edge_attrs: Tensor, edge_weights: Tensor,
                   edge_index: Tensor, num_nodes: int) -> Tensor:
        try:
            kernel = self._kernel(node_feats.device, node_feats.dtype)
        except Exception as exc:  # an irreps layout the kernel does not cover
            self.fast_eligible = False
            self.fast_active = False
            _cueq.warn_fallback(type(self).__name__, exc)
            messages = self(node_feats[edge_index[0]], edge_attrs, edge_weights)
            return scatter_sum(messages, edge_index[1], num_nodes)
        w = edge_weights
        if kernel["weight_index"] is not None:
            w = w.index_select(1, kernel["weight_index"])
        w = w * kernel["weight_scale"]
        out = kernel["op"](node_feats, edge_attrs, w, indices_1=edge_index[0],
                           indices_out=edge_index[1], size_out=num_nodes)
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


def _channelwise(irreps_in1, irreps_in2, irreps_out, device, dtype, method=None):
    """The cuEquivariance channelwise tensor product in e3nn's (mul, ir) layout."""
    import cuequivariance as cue
    import cuequivariance_torch as cuet

    return cuet.ChannelWiseTensorProduct(
        _cueq.cue_irreps(irreps_in1), _cueq.cue_irreps(irreps_in2), str(o3.Irreps(irreps_out)),
        layout=cue.mul_ir, shared_weights=False, internal_weights=False,
        device=device, dtype=dtype, math_dtype=dtype, method=method)
