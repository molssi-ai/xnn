"""Compatibility shims for differences between supported PyTorch versions.

Imported for its side effects before any model family is loaded, so the
adjustments are in place by the time a dependency runs its own ``torch.load``
at import time.
"""

import torch


def allow_e3nn_constants() -> None:
    """Let e3nn load its Wigner constants under PyTorch's ``weights_only``.

    PyTorch 2.6 changed the default of ``torch.load`` from
    ``weights_only=False`` to ``True``. e3nn 0.4.4 loads ``o3/constants.pt``
    at import time without passing the argument, and that file contains a
    ``slice`` object, so on PyTorch >= 2.6 the import fails with::

        _pickle.UnpicklingError: Weights only load failed ...
        Unsupported global: GLOBAL slice was not an allowed global by default

    which takes down ``import xnn`` with it. e3nn 0.4.4 is not a free choice
    -- mace-torch 0.3.16 pins it -- so the fix belongs here.

    Allow-list just ``slice``, which is all that file needs. The blunter
    alternative, setting ``TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1``, would turn
    the safety default off for every ``torch.load`` in the process,
    including ones reading checkpoints from elsewhere.

    Safe to call more than once, and a no-op on PyTorch versions without
    ``add_safe_globals``.
    """
    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is None:  # PyTorch < 2.3
        return
    try:
        add_safe_globals([slice])
    except Exception:  # pragma: no cover - never worth failing an import over
        pass
