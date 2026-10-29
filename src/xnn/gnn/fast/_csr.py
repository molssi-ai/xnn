"""Compressed rows of a pair list: the pairs sorted by center or by neighbor.

The fused kernels of SchNet and TensorNet run one program per atom over its
pairs in one of these orders and accumulate in registers, so their sums need no
atomic adds and come out the same in every run.
"""
from typing import Optional, Tuple

import torch
from torch import Tensor


def compressed(index: Tensor, n: int) -> Tuple[Tensor, Tensor]:
    """The pairs sorted by ``index`` (stable) and the row pointers ``(n + 1,)``."""
    order = torch.argsort(index, stable=True)
    ptr = torch.zeros(n + 1, dtype=torch.int64, device=index.device)
    ptr[1:] = torch.cumsum(torch.bincount(index, minlength=n), 0)
    return order, ptr


#: the orderings of the last pair list: the layers of one evaluation share them
_last = [None]


def orderings(src: Tensor, dst: Tensor, n: int,
              key: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """The pairs by center and by neighbor: ``(order_dst, ptr_dst, order_src, ptr_src)``.

    Computed once per pair list: the layers of an evaluation pass the same
    ``src`` and ``dst`` tensors, or the same ``key`` (for example the edge index
    the two are rows of). The cache holds them, so an identical object is the
    same pair list.
    """
    keys = (src, dst) if key is None else (key,)
    hit = _last[0]
    if (hit is not None and len(hit[0]) == len(keys) and all(a is b for a, b in zip(hit[0], keys))
            and hit[1] == n):
        return hit[2]
    out = (*compressed(dst, n), *compressed(src, n))
    _last[0] = (keys, n, out)
    return out
