"""Importing xnns must work on PyTorch >= 2.6.

PyTorch 2.6 flipped the default of ``torch.load`` to ``weights_only=True``.
e3nn 0.4.4 -- pinned by mace-torch 0.3.16, so not a free choice -- loads its
Wigner constants at import time without passing the argument, and that file
contains a ``slice``. Unfixed, ``import xnns`` dies with an UnpicklingError.
"""

import subprocess
import sys

import torch


def test_slice_is_allow_listed():
    from xnns.common._torch_compat import allow_e3nn_constants

    allow_e3nn_constants()
    allowed = torch.serialization.get_safe_globals()
    assert any(g is slice or getattr(g, "__name__", "") == "slice" for g in allowed)


def test_calling_twice_is_harmless():
    from xnns.common._torch_compat import allow_e3nn_constants

    allow_e3nn_constants()
    allow_e3nn_constants()


def test_import_without_the_env_var_escape_hatch():
    """The env var disables weights_only for the whole process; xnns must not
    need it. Run in a subprocess so the parent's already-imported state and
    environment cannot mask the problem."""
    code = "import xnns; from xnns.common.deploy import MDIEngine; print('ok')"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"},
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "ok" in result.stdout


def test_gnn_failure_does_not_break_the_package(monkeypatch):
    """An optional family must degrade, not take the package down."""
    import importlib

    import xnns

    monkeypatch.setitem(sys.modules, "e3nn", None)
    for name in [m for m in sys.modules if m.startswith("xnns")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    # re-importing with a broken e3nn must still yield a usable package
    mod = importlib.import_module("xnns")
    assert hasattr(mod, "common")
