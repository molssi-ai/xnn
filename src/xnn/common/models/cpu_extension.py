"""Optional C++ / OpenMP CPU kernels of some fast paths (D3, D4, the HDNNP symmetry functions).

xnn's CPU paths are PyTorch code. For D3, D4 and the HDNNP symmetry functions an optional
backend of hand-written C++ / OpenMP kernels is available as well, with the loops of the Fortran
reference codes. It is off by default; turn it on with :func:`enable` or the environment variable
``XNN_CPP_KERNELS=1``. The kernels ship as C++ source and are compiled with
:func:`torch.utils.cpp_extension.load_inline` the first time they run (a C++ compiler with OpenMP
and ``ninja`` are needed, ``pip install "xnns[cpu-kernels]"``); PyTorch keeps the build in its
extension cache, so later processes load it. Where the build fails, or while the backend is off,
the callers run their PyTorch paths.
"""
from __future__ import annotations

import os
import shutil
import sysconfig
from typing import Dict, Optional, Sequence

_MODULES: Dict[str, object] = {}
_FAILED: set = set()
_ENABLED = os.environ.get("XNN_CPP_KERNELS", "0").strip().lower() in ("1", "true", "yes", "on")


def enable(flag: bool = True) -> None:
    """Turn the optional C++ CPU kernels on (or off) for this process."""
    global _ENABLED
    _ENABLED = bool(flag)


def enabled() -> bool:
    """Whether the optional C++ CPU kernels are turned on."""
    return _ENABLED


def load(name: str, source: str, functions: Sequence[str]) -> Optional[object]:
    """The extension ``name`` built from ``source`` with ``functions`` bound, or ``None`` while the
    backend is off or where it does not build (tried once per process)."""
    if not _ENABLED:
        return None
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
    except Exception:  # noqa: BLE001  (no compiler, no OpenMP or no ninja: the PyTorch paths run)
        _FAILED.add(name)
        return None
    _MODULES[name] = module
    return module
