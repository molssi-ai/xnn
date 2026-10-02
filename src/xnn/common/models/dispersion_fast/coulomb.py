"""The molecular EEQ interaction ``erf(gamma_ij r_ij) / r_ij`` applied without storing it.

The reference applies the dense ``(N, N)`` kernel in row blocks that are
materialized (positions, distances, ``erf``) at every matrix-vector product. The
Triton kernel here evaluates each pair in registers and accumulates the product
directly: memory ``O(N)``, one pass over the pairs per product.
"""
from __future__ import annotations

import torch
from torch import Tensor

import triton
import triton.language as tl

BLOCK_I = 64
BLOCK_J = 64


@triton.jit
def _coulomb_kernel(pos_ptr, rad_ptr, v_ptr, out_ptr, n, ld_v,
                    TWO: tl.constexpr, BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr):
    i = tl.program_id(0) * BLOCK_I + tl.arange(0, BLOCK_I)
    mi = i < n
    xi = tl.load(pos_ptr + i * 3 + 0, mask=mi, other=0.0)
    yi = tl.load(pos_ptr + i * 3 + 1, mask=mi, other=0.0)
    zi = tl.load(pos_ptr + i * 3 + 2, mask=mi, other=0.0)
    ri = tl.load(rad_ptr + i, mask=mi, other=1.0)
    acc0 = tl.zeros([BLOCK_I], dtype=xi.dtype)
    acc1 = tl.zeros([BLOCK_I], dtype=xi.dtype)
    for j0 in range(0, n, BLOCK_J):
        j = j0 + tl.arange(0, BLOCK_J)
        mj = j < n
        xj = tl.load(pos_ptr + j * 3 + 0, mask=mj, other=0.0)
        yj = tl.load(pos_ptr + j * 3 + 1, mask=mj, other=0.0)
        zj = tl.load(pos_ptr + j * 3 + 2, mask=mj, other=0.0)
        rj = tl.load(rad_ptr + j, mask=mj, other=1.0)
        dx = xi[:, None] - xj[None, :]
        dy = yi[:, None] - yj[None, :]
        dz = zi[:, None] - zj[None, :]
        r2 = dx * dx + dy * dy + dz * dz
        pair = mi[:, None] & mj[None, :] & (i[:, None] != j[None, :])
        r = tl.sqrt(tl.where(pair, r2, 1.0))
        gamma = 1.0 / tl.sqrt(ri[:, None] * ri[:, None] + rj[None, :] * rj[None, :])
        kern = tl.where(pair, tl.math.erf(gamma * r) / r, 0.0)
        v0 = tl.load(v_ptr + j * ld_v, mask=mj, other=0.0)
        acc0 += tl.sum(kern * v0[None, :], axis=1)
        if TWO:
            v1 = tl.load(v_ptr + j * ld_v + 1, mask=mj, other=0.0)
            acc1 += tl.sum(kern * v1[None, :], axis=1)
    tl.store(out_ptr + i * ld_v, acc0, mask=mi)
    if TWO:
        tl.store(out_ptr + i * ld_v + 1, acc1, mask=mi)


def coulomb_matvec(pos: Tensor, rad: Tensor, v: Tensor) -> Tensor:
    """``sum_{j != i} erf(gamma_ij r_ij) / r_ij v_j`` for ``v`` of shape ``(N, K)``.

    ``gamma_ij = (a_i^2 + a_j^2)^(-1/2)``; ``pos (N, 3)`` and ``rad (N,)`` in bohr.
    No autograd (the operator of the solver loop).
    """
    n, k = v.shape
    pos = pos.detach().contiguous()
    rad = rad.detach().to(pos.dtype).contiguous()
    out = torch.empty_like(v, memory_format=torch.contiguous_format)
    grid = (triton.cdiv(n, BLOCK_I),)
    for j0 in range(0, k, 2):
        cols = v[:, j0:j0 + 2].contiguous()
        res = torch.empty_like(cols)
        _coulomb_kernel[grid](pos, rad, cols, res, n, cols.shape[1], TWO=cols.shape[1] == 2,
                              BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J)
        out[:, j0:j0 + 2] = res
    return out
