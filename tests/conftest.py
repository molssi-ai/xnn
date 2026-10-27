"""Shared test set-up."""
import os

import torch

# without a CUDA device the fused Triton kernels of the fast paths run under Triton's
# interpreter (tests marked so); it must be chosen before triton is first imported
if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

# xnn.gnn imports e3nn so that e3nn 0.4.4 also loads under torch >= 2.6; test
# modules import e3nn directly, so do it before any of them
try:
    import xnn.gnn  # noqa: F401
except ImportError:
    pass
