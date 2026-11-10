"""Equivariant linear maps of the MACE blocks, with a fast path in plain PyTorch.

Two maps spend most of their time on work whose result is known in advance. An
:class:`e3nn.o3.Linear` reads its input irrep block by irrep block from slices,
and the backward of every slice is a zero-filled gradient of the full input
width, summed over the blocks. The self-connection of an interaction, a tensor
product with the one-hot element of each atom, multiplies by the weights of
every element of the model, all but one of them with a zero.

The fast paths split the input once (whose backward is one concatenation) and
multiply the atoms of each element by that element's weights only. They give
the reference's numbers up to the summation order, and their derivatives of
any order are those of plain PyTorch operations.
"""
# NOTE: no `from __future__ import annotations` here: the forward methods are
# compiled by torch.jit.script, which cannot resolve stringified annotations.
import math
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
from e3nn import o3
from torch import Tensor

from xnn.common.models.fast import FastPathModule


class _Path(NamedTuple):
    """One weighted path from an input irrep block into an output block."""

    k: int              # position in the list of path weights
    i_in: int           # input irrep block
    offset: int         # start in the flat weight
    mul_in: int
    mul_out: int
    coefficient: float  # the normalization of the path


class _BlockPlan:
    """The irrep blocks of a map from ``irreps_in`` to ``irreps_out`` and the paths into each output block."""

    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps, paths: List[_Path],
                 i_out: List[int]):
        self.widths = [mul * ir.dim for mul, ir in irreps_in]
        self.muls = [mul for mul, _ in irreps_in]
        self.dims = [ir.dim for _, ir in irreps_in]
        self.paths = paths
        self.outs: List[Tuple[int, int, List[_Path]]] = [
            (mul, ir.dim, [p for p, j in zip(paths, i_out) if j == i])
            for i, (mul, ir) in enumerate(irreps_out) if mul > 0
        ]

    def apply(self, x: Tensor, weights: List[Tensor], in_ir_mul: bool = False,
              out_ir_mul: bool = False) -> Tensor:
        """``x`` of shape ``(n, irreps_in.dim)`` through the paths, with ``weights[k]`` of shape
        ``(mul_in, mul_out)`` (normalization included) on path ``k``. Each irrep block is laid
        out multiplicity-major (e3nn's ``(mul, 2l + 1)``) or, with ``in_ir_mul`` / ``out_ir_mul``,
        component-major ``(2l + 1, mul)`` (cuEquivariance's ``ir_mul``)."""
        n = x.shape[0]
        blocks = torch.split(x, self.widths, dim=1) if len(self.widths) > 1 else (x,)
        rows: Dict[int, Tensor] = {}

        def rows_of(i: int) -> Tensor:
            # (n * dim, mul): the components of one atom's irreps on consecutive rows
            if i not in rows:
                d, u, b = self.dims[i], self.muls[i], blocks[i]
                if d == 1:
                    rows[i] = b
                elif in_ir_mul:
                    rows[i] = b.reshape(n * d, u)
                else:
                    rows[i] = b.reshape(n, u, d).transpose(1, 2).reshape(n * d, u)
            return rows[i]

        outs: List[Tensor] = []
        for mul_out, dim, paths in self.outs:
            if not paths:
                outs.append(x.new_zeros(n, mul_out * dim))
                continue
            if len(paths) == 1:
                y = rows_of(paths[0].i_in) @ weights[paths[0].k]
            else:
                y = (torch.cat([rows_of(p.i_in) for p in paths], dim=1)
                     @ torch.cat([weights[p.k] for p in paths], dim=0))
            if dim > 1:
                y = (y.reshape(n, dim * mul_out) if out_ir_mul
                     else y.reshape(n, dim, mul_out).transpose(1, 2).reshape(n, mul_out * dim))
            outs.append(y)
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)

    def to_ir_mul(self, x: Tensor, irreps: List[Tuple[int, int]], inverse: bool = False) -> Tensor:
        """``x`` with irrep blocks ``(mul, dim)`` from multiplicity-major to component-major
        (``inverse``: back)."""
        n = x.shape[0]
        parts = torch.split(x, [u * d for u, d in irreps], dim=1)
        out = [p if d == 1 else
               (p.reshape(n, d, u) if inverse else p.reshape(n, u, d)).transpose(1, 2).reshape(n, u * d)
               for p, (u, d) in zip(parts, irreps)]
        return out[0] if len(out) == 1 else torch.cat(out, dim=1)


class EquivariantLinear(o3.Linear, FastPathModule):
    """:class:`e3nn.o3.Linear` (same arguments, parameters and ``state_dict``) with a fast path.

    The fast path applies to shared weights without biases on inputs of shape
    ``(n, irreps_in.dim)``: one matrix product per output irrep block on the
    input split once, in place of e3nn's slices of the input whose backward
    fills a zero gradient of the full width per block.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fast_active = False
        self.fast_eligible = self.shared_weights and self.bias_numel == 0
        paths: List[_Path] = []
        i_out: List[int] = []
        offset = 0
        for ins in self.instructions:
            if ins.i_in < 0:
                continue
            mul_in, mul_out = ins.path_shape
            paths.append(_Path(len(paths), ins.i_in, offset, mul_in, mul_out, float(ins.path_weight)))
            i_out.append(ins.i_out)
            offset += mul_in * mul_out
        # kept out of the state_dict and of TorchScript
        self.__dict__["_plan"] = _BlockPlan(self.irreps_in, self.irreps_out, paths, i_out)

    def fast_supported(self, device: torch.device, dtype: torch.dtype) -> bool:
        return self.fast_eligible

    def forward(self, features: Tensor, weight: Optional[Tensor] = None,
                bias: Optional[Tensor] = None) -> Tensor:
        """The linear map of ``features`` (shape ``(..., irreps_in.dim)``); ``weight`` and ``bias``
        replace the module's own as in :class:`e3nn.o3.Linear`."""
        if weight is None:
            if self.weight_numel > 0 and not self.internal_weights:
                raise RuntimeError("Weights must be provided when internal_weights = False")
            weight = self.weight
        if bias is None:
            if self.bias_numel > 0 and not self.internal_weights:
                raise RuntimeError("Biases must be provided when internal_weights = False")
            bias = self.bias
        if self.fast_active and features.dim() == 2 and weight.dim() == 1:
            # every argument given: an unused method keeps no defaults under TorchScript
            return self._forward_fast(features, weight, 1.0, False)
        return self._compiled_main(features, weight, bias)

    @torch.jit.unused
    def _forward_fast(self, features: Tensor, weight: Tensor, scale: float = 1.0,
                      out_ir_mul: bool = False) -> Tensor:
        plan = self.__dict__["_plan"]
        weights = [weight.narrow(0, p.offset, p.mul_in * p.mul_out).view(p.mul_in, p.mul_out)
                   * (p.coefficient * scale) for p in plan.paths]
        return plan.apply(features, weights, out_ir_mul=out_ir_mul)

    @torch.jit.unused
    def scaled_ir_mul(self, features: Tensor, scale: float) -> Tensor:
        """``self(features) * scale`` with every output irrep block laid out ``(2l + 1, mul)``,
        the scale folded into the weights (fast path only)."""
        return self._forward_fast(features, self.weight, scale, out_ir_mul=True)


class SpeciesTensorProduct(o3.FullyConnectedTensorProduct, FastPathModule):
    """Tensor product with the one-hot chemical element of each atom, with a fast path.

    An :class:`e3nn.o3.FullyConnectedTensorProduct` (same arguments, parameters
    and ``state_dict``) whose second input is one scalar irrep per element.
    With one-hot rows in that input, the product is a linear map of the first
    input chosen by each atom's element. The fast path groups the atoms by
    element and applies each element's map to its atoms only, instead of the
    reference's product with every element. It runs outside CUDA-graph capture
    (grouping reads the element counts on the host) and when no gradient with
    respect to the element input is needed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fast_active = False
        in2 = self.irreps_in2
        self.fast_eligible = (
            len(in2) == 1 and in2[0].ir == o3.Irrep("0e")
            and self.internal_weights and self.shared_weights
            and all(ins.connection_mode == "uvw" and ins.has_weight for ins in self.instructions)
        )
        self.num_species = in2[0].mul if len(in2) == 1 else 0
        paths: List[_Path] = []
        i_out: List[int] = []
        offset = 0
        for ins in self.instructions:
            mul_in, n_species, mul_out = ins.path_shape
            dim = self.irreps_out[ins.i_out].ir.dim
            # e3nn's normalization of the path; the Clebsch-Gordan factor of l x 0 -> l is 1 / sqrt(2l + 1)
            paths.append(_Path(len(paths), ins.i_in1, offset, mul_in, mul_out,
                               float(ins.path_weight) / math.sqrt(dim)))
            i_out.append(ins.i_out)
            offset += mul_in * n_species * mul_out
        self.__dict__["_plan"] = _BlockPlan(self.irreps_in1, self.irreps_out, paths, i_out)

    def fast_supported(self, device: torch.device, dtype: torch.dtype) -> bool:
        return self.fast_eligible

    def forward(self, x: Tensor, y: Tensor, weight: Optional[Tensor] = None) -> Tensor:
        """The tensor product of ``x`` (shape ``(..., irreps_in1.dim)``) with the element
        attributes ``y`` (``(..., num_species)``, one-hot rows on the fast path)."""
        if self.fast_active and weight is None:
            return self._forward_fast(x, y, False)
        return self._compiled_main_left_right(x, y, self._get_weights(weight))

    @torch.jit.unused
    def _forward_fast(self, x: Tensor, y: Tensor, ir_mul: bool = False) -> Tensor:
        plan = self.__dict__["_plan"]
        if (x.dim() != 2 or y.requires_grad
                or (x.is_cuda and torch.cuda.is_current_stream_capturing())):
            if not ir_mul:
                return self._compiled_main_left_right(x, y, self.weight)
            in1 = [(mul, ir.dim) for mul, ir in self.irreps_in1]
            out = [(mul, ir.dim) for mul, ir in self.irreps_out]
            return plan.to_ir_mul(self._compiled_main_left_right(plan.to_ir_mul(x, in1, inverse=True), y,
                                                                 self.weight), out)
        species = y.argmax(dim=1)
        counts = torch.bincount(species, minlength=self.num_species).tolist()
        present = [s for s, c in enumerate(counts) if c > 0] or [0]
        if len(present) == 1:
            return plan.apply(x, self._species_weights(plan, present[0]), ir_mul, ir_mul)
        order = torch.argsort(species)
        parts = torch.split(x.index_select(0, order), [counts[s] for s in present])
        out = torch.cat([plan.apply(part, self._species_weights(plan, s), ir_mul, ir_mul)
                         for part, s in zip(parts, present)])
        # every row is written: the rows of each element back to their atoms
        return out.new_empty(out.shape).index_copy(0, order, out)

    @torch.jit.unused
    def ir_mul(self, x: Tensor, y: Tensor) -> Tensor:
        """The product with every irrep block of ``x`` and of the result laid out ``(2l + 1, mul)``
        (fast path only)."""
        return self._forward_fast(x, y, ir_mul=True)

    @torch.jit.unused
    def _species_weights(self, plan: _BlockPlan, s: int) -> List[Tensor]:
        """The ``(mul_in, mul_out)`` map of element ``s`` on every path, normalization included."""
        S = self.num_species
        return [self.weight.narrow(0, p.offset, p.mul_in * S * p.mul_out).view(p.mul_in, S, p.mul_out)[:, s]
                * p.coefficient for p in plan.paths]


__all__ = ["EquivariantLinear", "SpeciesTensorProduct"]
