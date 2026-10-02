"""Constants of the GNN potentials that stay exact in float64.

Some values of a model are fixed by its configuration but computed when a block
is built, in the default dtype of that moment, and kept as buffers (so most are
in the state_dict): Clebsch-Gordan (Wigner-3j) tensors and the bases built from
them, radial frequencies, distance-transform and screening constants, the
energy scale and shift. A model built in float32 and cast to float64, or loaded
from a float32 checkpoint, would carry them float32-rounded, which caps its
float64 agreement with an exact evaluation (near 1e-10 for the coupling tensors,
1e-7 for the radial frequencies). :func:`exact_float64_constants` rebuilds every
float64 one exactly as a float64 build would make it, and
:class:`~xnn.gnn.models.base.GNNPotential` calls it after every cast and every
``load_state_dict``. Float32 values are left as built: their rounding is below
float32 arithmetic.
"""
from __future__ import annotations

import re

import torch
from torch import nn

_W3J_BUFFER = re.compile(r"_w3j_(\d+)_(\d+)_(\d+)")
# a stored value within this relative distance of its exact form is that value
# rounded (float32 rounding is 6e-8); farther away it was set on purpose
ROUNDING = 1e-6


def register_constant(module: nn.Module, name: str, value, persistent: bool = True) -> None:
    """Register ``value`` as a floating buffer that stays exact in float64.

    The buffer is made in the default dtype, as a plain ``register_buffer``
    would; the value itself is kept in float64, so
    :func:`exact_float64_constants` can restore it once the module is float64.

    Parameters
    ----------
    module : torch.nn.Module
        The module that owns the buffer.
    name : str
        The buffer name.
    value : float, sequence of float or torch.Tensor
        The constant, as exact as the caller has it.
    persistent : bool, optional
        Whether the buffer is part of the ``state_dict``, by default True.
    """
    exact = torch.as_tensor(value, dtype=torch.float64).detach().clone()
    # a copy even in a float64 build: the buffer is written in place (loads), the
    # exact value must not be
    module.register_buffer(name, exact.to(torch.get_default_dtype(), copy=True),
                           persistent=persistent)
    module.__dict__.setdefault("_exact_constants", {})[name] = exact


def restore_exact(buf: torch.Tensor, exact: torch.Tensor) -> bool:
    """Copy ``exact`` into the float64 ``buf`` when they differ by rounding only.

    A larger difference means the stored value is not this constant (a fitted
    scale, or a checkpoint's own coupling basis, which can differ from a fresh
    one across e3nn versions), and it is kept. Returns whether ``buf`` changed.
    """
    if buf.dtype != torch.float64 or buf.shape != exact.shape:
        return False
    exact = exact.to(buf.device)
    if torch.equal(buf, exact):
        return False
    scale = float(exact.abs().max()) if exact.numel() else 0.0
    if float((buf - exact).abs().max()) > ROUNDING * max(scale, 1e-300):
        return False
    buf.copy_(exact)
    return True


@torch.no_grad()
def exact_float64_constants(module: nn.Module) -> int:
    """Make the float64 constants of ``module`` exact.

    Covers the buffers made with :func:`register_constant`, e3nn's tensor
    products (their ``_w3j_<l1>_<l2>_<l3>`` buffers) and any block with an
    ``exact_constants()`` method (the MACE ``U`` basis, Allegro's block
    Wigner-3j, fixed Bessel frequencies). A stored value that is not its
    constant up to rounding (a fitted scale, a custom frequency, a checkpoint's
    own coupling basis) is kept. When
    a value changes, the fast-path kernels cached on the blocks are dropped,
    since they may have been fitted to it.

    Parameters
    ----------
    module : torch.nn.Module
        The model (or any part of it).

    Returns
    -------
    int
        The number of tensors whose values changed.
    """
    from e3nn import o3

    changed = 0
    for mod in module.modules():
        if isinstance(mod, torch.jit.ScriptModule):
            named = mod.named_buffers(recurse=False)
        else:
            own = getattr(mod, "exact_constants", None)
            if callable(own):
                changed += int(own())
            for name, exact in mod.__dict__.get("_exact_constants", {}).items():
                buf = mod._buffers.get(name)
                if buf is not None and restore_exact(buf, exact):
                    changed += 1
            named = mod.named_buffers(recurse=False)
        for name, buf in named:
            match = _W3J_BUFFER.fullmatch(name)
            if match is None or buf.dtype != torch.float64:
                continue
            exact = o3.wigner_3j(*(int(l) for l in match.groups()), dtype=torch.float64)
            if restore_exact(buf, exact):
                changed += 1
    if changed:
        for mod in module.modules():
            kernels = mod.__dict__.get("_fast_kernels")
            if isinstance(kernels, dict):
                kernels.clear()
    return changed
