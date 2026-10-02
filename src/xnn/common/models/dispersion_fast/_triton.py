"""Whether the Triton kernels of the dispersion fast paths can run."""
from __future__ import annotations

import functools

import torch


@functools.lru_cache(maxsize=None)
def available() -> bool:
    """Triton is importable (it ships with the CUDA builds of PyTorch)."""
    try:
        import triton  # noqa: F401
        import triton.language  # noqa: F401
    except Exception:  # noqa: BLE001  (any import failure means no kernels)
        return False
    return True


def supported(device: torch.device, dtype: torch.dtype) -> bool:
    """The kernels run on CUDA devices in float32 and float64."""
    return device.type == "cuda" and dtype in (torch.float32, torch.float64) and available()


def next_power_of_two(n: int, minimum: int = 16) -> int:
    """The smallest power of two ``>= max(n, minimum)`` (Triton block shapes)."""
    p = minimum
    while p < n:
        p *= 2
    return p
