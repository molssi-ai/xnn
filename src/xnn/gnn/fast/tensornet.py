"""Fast path of the TensorNet embedding and interaction layers (Triton kernels).

The reference layers gather the neighbors' tensor parts to the pairs, multiply
them with the filters and sum the products onto the centers as separate tensor
operations, each writing a ``(P, F)``, ``(P, 3, F)`` or ``(P, 3, 3, F)`` tensor
that autograd keeps for the backward pass (``P`` pairs, the self pairs
included); the embedding does the same with the radial projections and the
direction products. Here the networks stay as they are (the pair embedding per
atom, the radial projections and the filter networks), and one kernel per layer
gathers, multiplies and adds onto the centers, applying the cutoff on the way:
no pair-sized tensor is written. The symmetric parts travel as their six
distinct entries. The backward pass is one kernel per layer that adds the
gradients of the atom-wise inputs onto the neighbors (and centers) and writes
those of the filter networks' outputs, the cutoff and the directions.

The gradient reaches positions, cell and the networks' weights, and the
backward kernels are differentiable once more, by kernels of the same form (the
pair sums are linear in each input), so that a force loss trains through the
fused layers. Third derivatives are not supported. The atom-wise stages are
compiled, in training mode with their first and second derivatives compiled as
well.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch
from torch import Tensor

from xnn.common.models.fast import AutoPolicy, compile_twice_differentiable

#: Edge counts above which ``use_fast="auto"`` takes the kernels in evaluation mode (energy
#: and forces): every size, since the fused layers were faster than the reference from a
#: single small molecule on, in both precisions, and use less memory.
AUTO_POLICY = AutoPolicy(default_min_edges=0, default_min_edges_float64=0)

#: The same for training mode (a step with a force loss, second derivatives included): every
#: size on an A100, where the fused layers were faster than the reference from a single small
#: molecule on; on a T4 from a few thousand edges, below which a step is bound by kernel
#: launches.
AUTO_POLICY_TRAINING = AutoPolicy(
    min_edges={"T4": 5_000},
    min_edges_float64={"T4": 2_000},
    default_min_edges=0,
    default_min_edges_float64=0,
)


def supported(device: torch.device, dtype: torch.dtype) -> bool:
    """Triton kernels on a CUDA device in float32 or float64."""
    from xnn.common.models.dispersion_fast._triton import supported as triton_supported
    return triton_supported(device, dtype)


# The atom-wise stages of the fused layers, on the compact rows of a tensor: the scalar
# I (N, F), the axial vector v (N, 3, F) of the skew part and the six distinct entries S
# (N, 6, F) of the symmetric part; the full tensors are (N, 3, 3, F). Plain torch functions
# on whole tensors (a matrix product is one broadcast product and one sum, as in the
# reference); on a GPU each is compiled into fused kernels by torch.compile (see stage).

def _matrix(I: Tensor, v: Tensor, S: Tensor) -> Tensor:
    """The tensors ``I 1 + [v]_x + S``, ``(N, 3, 3, F)``."""
    vx, vy, vz = v.unbind(1)
    sxx, sxy, sxz, syy, syz, szz = S.unbind(1)
    return torch.stack([I + sxx, sxy - vz, sxz + vy,
                        sxy + vz, I + syy, syz - vx,
                        sxz - vy, syz + vx, I + szz], dim=1).reshape(I.shape[0], 3, 3, I.shape[1])


def _parts(C: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """The scalar, axial vector and symmetric traceless entries of the tensors ``C``."""
    I = C.diagonal(dim1=1, dim2=2).sum(dim=-1) / 3.0
    A = C - C.transpose(1, 2)
    v = 0.5 * torch.stack([A[:, 2, 1], A[:, 0, 2], A[:, 1, 0]], dim=1)
    B = C + C.transpose(1, 2)
    S = torch.stack([C[:, 0, 0] - I, 0.5 * B[:, 0, 1], 0.5 * B[:, 0, 2], C[:, 1, 1] - I,
                     0.5 * B[:, 1, 2], C[:, 2, 2] - I], dim=1)
    return I, v, S


def _product(a: Tensor, b: Tensor) -> Tensor:
    """The feature-wise matrix products ``a b`` of ``(N, 3, 3, F)`` tensors."""
    return (a.unsqueeze(3) * b.unsqueeze(1)).sum(dim=2)


def _squared_norm(C: Tensor) -> Tensor:
    """``|C|^2`` per atom and feature, ``(N, F)``."""
    return (C * C).sum(dim=(1, 2))


def embedding_parts(rows: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """The embedding's parts from the fused pair sums ``(N, 10, F)``: ``I``, ``v``, the
    traceless ``S`` and the squared Frobenius norm of ``I 1 + [v]_x + S``."""
    I, v, full = rows[:, 0], rows[:, 1:4], rows[:, 4:]
    third = (full[:, 0] + full[:, 3] + full[:, 5]) / 3.0
    S = torch.stack([full[:, 0] - third, full[:, 1], full[:, 2], full[:, 3] - third, full[:, 4],
                     full[:, 5] - third], dim=1)
    return I, v, S, _squared_norm(_matrix(I, v, S))


def compose_rows(I: Tensor, v: Tensor, S: Tensor) -> Tensor:
    """The ``(N, 3, 3, F)`` tensors ``I 1 + [v]_x + S``."""
    return _matrix(I, v, S)


def interaction_parts(X: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """``X / (|X|^2 + 1)`` and its parts ``I``, ``v``, ``S``."""
    Xn = X / (_squared_norm(X) + 1.0)[:, None, None, :]
    return (Xn, *_parts(Xn))


def interaction_product(I: Tensor, v: Tensor, S: Tensor, m: Tensor, q: Tensor,
                        o3: bool) -> Tuple[Tensor, Tensor, Tensor]:
    """The parts of ``C = (1 + q / 10)(Y M + M Y)`` (``O(3)``) or ``2 Y M`` (``SO(3)``), each
    divided by ``|C|^2 + 1``; ``Y`` from its parts, ``M`` from the message rows ``(N, 10, F)``."""
    y = _matrix(I, v, S)
    msg = _matrix(m[:, 0], m[:, 1:4], m[:, 4:])
    if o3:
        C = (1.0 + 0.1 * q)[:, None, None, None] * (_product(y, msg) + _product(msg, y))
    else:
        C = 2.0 * _product(y, msg)
    norm = _squared_norm(C) + 1.0
    cI, cv, cS = _parts(C)
    return cI / norm, cv / norm[:, None], cS / norm[:, None]


def interaction_update(Xn: Tensor, I: Tensor, v: Tensor, S: Tensor, q: Tensor) -> Tensor:
    """``X + dX + (1 + q / 10) dX^2`` with ``dX`` from its parts."""
    d = _matrix(I, v, S)
    return Xn + d + (1.0 + 0.1 * q)[:, None, None, None] * _product(d, d)


def _interaction_product_o3(I: Tensor, v: Tensor, S: Tensor, m: Tensor,
                            q: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    return interaction_product(I, v, S, m, q, True)


def _interaction_product_so3(I: Tensor, v: Tensor, S: Tensor, m: Tensor,
                             q: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    return interaction_product(I, v, S, m, q, False)


#: the stages (plain functions of tensors, which torch.compile guards cheaply) and the number of
#: their arguments
_STAGES = {"embedding_parts": (embedding_parts, 1), "compose_rows": (compose_rows, 3),
           "interaction_parts": (interaction_parts, 1),
           "interaction_product_o3": (_interaction_product_o3, 5),
           "interaction_product_so3": (_interaction_product_so3, 5),
           "interaction_update": (interaction_update, 5)}
_compiled: Dict[tuple, Callable] = {}


def stage(name: str, like: Tensor, training: bool = False) -> Callable:
    """The atom-wise stage ``name``: for CUDA tensors compiled, in training mode twice
    differentiable (see :func:`~xnn.common.models.fast.compile_twice_differentiable`); as
    written otherwise."""
    fn, n_inputs = _STAGES[name]
    if not like.is_cuda:
        return fn
    key = (name, training)
    out = _compiled.get(key)
    if out is None:
        out = _compiled[key] = (compile_twice_differentiable(fn, n_inputs) if training
                                else torch.compile(fn, dynamic=True))
    return out
