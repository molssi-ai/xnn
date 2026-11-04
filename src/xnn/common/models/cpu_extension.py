"""Building the C++ / OpenMP CPU kernels of the fast paths on first use.

The kernels ship as C++ source and are compiled with
:func:`torch.utils.cpp_extension.load_inline` the first time they run (a C++ compiler with OpenMP
and ``ninja`` are needed, ``pip install "xnns[cpu-kernels]"``); PyTorch keeps the build in its
extension cache, so later processes load it. Where the build fails the callers keep their other
paths.
"""
from __future__ import annotations

import os
import shutil
import sysconfig
from typing import Dict, Optional, Sequence

_MODULES: Dict[str, object] = {}
_FAILED: set = set()


def load(name: str, source: str, functions: Sequence[str]) -> Optional[object]:
    """The extension ``name`` built from ``source`` with ``functions`` bound, or ``None`` where it
    does not build (tried once per process)."""
    if name in _MODULES:
        return _MODULES[name]
    if name in _FAILED:
        return None
    scripts = sysconfig.get_path("scripts")
    if shutil.which("ninja") is None and shutil.which("ninja", path=scripts):
        # pip's ninja in this environment, not on PATH (an environment used without activation)
        os.environ["PATH"] = scripts + os.pathsep + os.environ.get("PATH", "")
    try:
        from torch.utils.cpp_extension import load_inline
        module = load_inline(name=name, cpp_sources=source, functions=list(functions),
                             extra_cflags=["-O3", "-fopenmp"], extra_ldflags=["-fopenmp"], verbose=False)
    except Exception:  # noqa: BLE001  (no compiler, no OpenMP or no ninja: the other paths run)
        _FAILED.add(name)
        return None
    _MODULES[name] = module
    return module
