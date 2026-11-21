"""Shared test set-up."""
import os
import re
import sys

import pytest
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


# CI runs on CPUs, where the tests that need a CUDA device skip; they carry the marker "gpu", so that
# ``pytest -m gpu tests/`` runs them all on a GPU machine. A test needs one where a skip names the GPU or CUDA
# in its reason and holds exactly without a device (tests that run on CPUs only skip the other way round)
_GPU_REASON = re.compile(r"\b(GPU|CUDA)\b")
_CUDA = torch.cuda.is_available()


def _needs_gpu(mark) -> bool:
    if not _GPU_REASON.search(str(mark.kwargs.get("reason", ""))):
        return False
    conditions = mark.args or (mark.kwargs.get("condition", True),)
    return all(bool(c) != _CUDA for c in conditions if not isinstance(c, str))


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA device (skipped on CPUs; pytest -m gpu runs them all)")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if any(_needs_gpu(mark) for mark in item.iter_markers("skipif")):
            item.add_marker(pytest.mark.gpu)
