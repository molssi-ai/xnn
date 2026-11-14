"""Shared test set-up."""
import os
import sys

import torch

# without a CUDA device the fused Triton kernels of the fast paths run under Triton's
# interpreter (tests marked so); it must be chosen before triton is first imported
if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

# test modules share helpers and fidelity harnesses across the family folders, so
# every folder is importable whichever part of the suite is collected
_HERE = os.path.dirname(os.path.abspath(__file__))
for _name in sorted(os.listdir(_HERE)):
    _folder = os.path.join(_HERE, _name)
    if os.path.isdir(_folder) and not _name.startswith(("_", ".")) and _folder not in sys.path:
        sys.path.insert(0, _folder)

# xnn.gnn imports e3nn so that e3nn 0.4.4 also loads under torch >= 2.6; test
# modules import e3nn directly, so do it before any of them
try:
    import xnn.gnn  # noqa: F401
except ImportError:
    pass
