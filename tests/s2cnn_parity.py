"""Parity of the xnn spherical CNN with the reference implementation
(jonas-koehler/s2cnn), as a harness for the tests and the fidelity notebook.

``setup()`` makes the reference importable on a current Python / numpy / torch
(``collections.Iterable`` for its ``lie_learn`` dependency, ``np.float`` in its
grids) and, for the float64 run, routes its float32 constant tables through
the default dtype. ``compare_all(dtype)`` returns the worst relative error of
every compared quantity.

    python tests/s2cnn_parity.py            # prints both precisions
"""
from __future__ import annotations

import collections
import collections.abc
import contextlib
import io
import math

import numpy as np
import torch

_PATCHED = False


def setup() -> None:
    """Import patches for the reference packages (idempotent)."""
    global _PATCHED
    if _PATCHED:
        return
    if not hasattr(collections, "Iterable"):
        collections.Iterable = collections.abc.Iterable
    if not hasattr(np, "float"):
        np.float = float
    import importlib
    import s2cnn  # noqa: F401
    # the package re-exports functions under their module names, so take the modules by path
    s2_ft = importlib.import_module("s2cnn.s2_ft")
    so3_ft = importlib.import_module("s2cnn.so3_ft")
    s2_fft = importlib.import_module("s2cnn.soft.s2_fft")
    so3_fft = importlib.import_module("s2cnn.soft.so3_fft")
    so3_integrate = importlib.import_module("s2cnn.soft.so3_integrate")
    so3_rotation = importlib.import_module("s2cnn.soft.so3_rotation")

    import lie_learn.spaces.S3 as S3
    from lie_learn.representations.SO3.wigner_d import wigner_D_matrix

    # the reference's CPU path of the real SO(3) transform calls rfftn on a complex
    # tensor (a torch < 1.8 idiom); route it through its complex transform instead
    def so3_rfft(x, for_grad=False, b_out=None):
        return so3_fft.so3_fft(torch.stack((x, torch.zeros_like(x)), -1), for_grad=for_grad, b_out=b_out)

    so3_fft.so3_rfft = so3_rfft

    # the reference builds its constant tables in float64 (lie_learn) and stores
    # them in float32; take the float64 arrays in the working precision instead
    def wigner_s2(b, nl, weighted, device):
        return torch.tensor(s2_fft._setup_s2_fft(b, nl, weighted), dtype=torch.get_default_dtype(), device=device).contiguous()

    def wigner_so3(b, nl, weighted, device):
        return torch.tensor(so3_fft._setup_so3_fft(b, nl, weighted), dtype=torch.get_default_dtype(), device=device).contiguous()

    def setup_s2_ft(b, grid, device_type, device_index):
        return torch.tensor(getattr(s2_ft, "__setup_s2_ft")(b, grid), dtype=torch.get_default_dtype(),
                            device=torch.device(device_type, device_index))

    def setup_so3_ft(b, grid, device_type, device_index):
        return torch.tensor(getattr(so3_ft, "__setup_so3_ft")(b, grid), dtype=torch.get_default_dtype(),
                            device=torch.device(device_type, device_index))

    def setup_integrate(b, device_type, device_index):
        return torch.tensor(S3.quadrature_weights(b), dtype=torch.get_default_dtype(),
                            device=torch.device(device_type, device_index))

    def setup_rotation(b, alpha, beta, gamma, device_type, device_index):
        out = []
        for l in range(b):
            U = wigner_D_matrix(l, alpha, beta, gamma, field="complex", normalization="quantum",
                                order="centered", condon_shortley="cs")
            U = np.ascontiguousarray(U.astype(np.complex128)).view(np.float64).reshape(2 * l + 1, 2 * l + 1, 2)
            out.append(torch.tensor(U, dtype=torch.get_default_dtype(), device=torch.device(device_type, device_index)))
        return out

    s2_fft._setup_wigner = wigner_s2
    so3_fft._setup_wigner = wigner_so3
    s2_ft._setup_s2_ft = setup_s2_ft
    so3_ft._setup_so3_ft = setup_so3_ft
    so3_integrate._setup_so3_integrate = setup_integrate
    so3_rotation._setup_so3_rotation = setup_rotation
    _PATCHED = True


def _relative(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max().clamp(min=1e-300))


def _to_padded_s2(flat: torch.Tensor, b: int) -> torch.Tensor:
    """Reference S2 spectrum [l * m, ..., complex] -> xnn layout (..., b, 2b - 1)."""
    z = torch.view_as_complex(flat.contiguous())                          # (nspec, ...)
    out = torch.zeros(z.shape[1:] + (b, 2 * b - 1), dtype=z.dtype)
    for l in range(b):
        out[..., l, b - 1 - l:b + l] = z[l * l:(l + 1) ** 2].movedim(0, -1)
    return out


def _to_padded_so3(flat: torch.Tensor, b: int) -> torch.Tensor:
    """Reference SO(3) spectrum [l * m * n, ..., complex] -> xnn layout (..., b, 2b - 1, 2b - 1)."""
    z = torch.view_as_complex(flat.contiguous())
    out = torch.zeros(z.shape[1:] + (b, 2 * b - 1, 2 * b - 1), dtype=z.dtype)
    for l in range(b):
        o = b - 1 - l
        start = l * (4 * l * l - 1) // 3
        block = z[start:start + (2 * l + 1) ** 2].movedim(0, -1).reshape(*z.shape[1:], 2 * l + 1, 2 * l + 1)
        out[..., l, o:o + 2 * l + 1, o:o + 2 * l + 1] = block
    return out


def _from_padded_s2(spec: torch.Tensor) -> torch.Tensor:
    """xnn S2 spectrum (..., b, 2b - 1) -> reference [l * m, ..., complex]."""
    b = spec.shape[-2]
    parts = [spec[..., l, b - 1 - l:b + l].movedim(-1, 0) for l in range(b)]
    return torch.view_as_real(torch.cat(parts, 0).contiguous())


def _from_padded_so3(spec: torch.Tensor) -> torch.Tensor:
    b = spec.shape[-3]
    parts = []
    for l in range(b):
        o = b - 1 - l
        block = spec[..., l, o:o + 2 * l + 1, o:o + 2 * l + 1]
        parts.append(block.reshape(*block.shape[:-2], (2 * l + 1) ** 2).movedim(-1, 0))
    return torch.view_as_real(torch.cat(parts, 0).contiguous())


def compare_all(dtype: torch.dtype = torch.float64, seed: int = 0) -> dict[str, float]:
    """Worst relative errors of every compared quantity, in ``dtype``."""
    setup()
    import importlib
    from s2cnn import (S2Convolution as RefS2Conv, SO3Convolution as RefSO3Conv, s2_near_identity_grid as ref_s2_grid,
                       so3_near_identity_grid as ref_so3_grid)
    from s2cnn.soft.s2_fft import s2_fft as ref_s2_fft, s2_ifft as ref_s2_ifft
    from s2cnn.soft.so3_fft import so3_rfft as ref_so3_rfft, so3_rifft as ref_so3_rifft
    ref_rotation = importlib.import_module("s2cnn.soft.so3_rotation").so3_rotation
    ref_integrate = importlib.import_module("s2cnn.soft.so3_integrate").so3_integrate
    from xnn.cnn.models import (S2Convolution, S2Transform, SO3Convolution, SO3Transform,
                                s2_near_identity_grid, so3_integrate, so3_near_identity_grid, so3_rotate)

    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    errors = {}
    try:
        g = torch.Generator().manual_seed(seed)
        b_in, b_out = 6, 4
        quiet = contextlib.redirect_stdout(io.StringIO())
        # S2 transform and inverse
        x = torch.randn(2, 3, 2 * b_in, 2 * b_in, generator=g, dtype=dtype)
        with quiet:
            ref = ref_s2_fft(torch.stack([x, torch.zeros_like(x)], -1), b_out=b_out)
        tr = S2Transform(b_in, b_out)
        errors["s2_fft"] = _relative(tr.analyze(x), _to_padded_s2(ref, b_out))
        spec = tr.analyze(x)
        with quiet:
            ref_sig = ref_s2_ifft(_from_padded_s2(spec), b_out=b_in)[..., 0]
        errors["s2_ifft"] = _relative(tr.synthesize(spec), ref_sig)
        # SO(3) transform and inverse
        y = torch.randn(2, 3, 2 * b_in, 2 * b_in, 2 * b_in, generator=g, dtype=dtype)
        with quiet:
            ref = ref_so3_rfft(y, b_out=b_out)
        tr3 = SO3Transform(b_in, b_out)
        errors["so3_fft"] = _relative(tr3.analyze(y), _to_padded_so3(ref, b_out))
        spec3 = tr3.analyze(y)
        with quiet:
            ref_sig = ref_so3_rifft(_from_padded_so3(spec3), b_out=b_in)
        errors["so3_ifft"] = _relative(tr3.synthesize(spec3), ref_sig)
        # integration and rotation
        with quiet:
            errors["so3_integrate"] = _relative(so3_integrate(y), ref_integrate(y))
            errors["so3_rotate"] = _relative(so3_rotate(y, 0.7, 1.2, 2.5), ref_rotation(y, 0.7, 1.2, 2.5))
        # the correlation layers with transplanted filters
        s2_pts = s2_near_identity_grid(math.pi / 8, 2 * b_in, 2)
        so3_pts = so3_near_identity_grid(math.pi / 8, 2 * math.pi, 2 * b_out, 2, 2)
        with quiet:
            ref_s2 = RefS2Conv(3, 4, b_in, b_out, ref_s2_grid(math.pi / 8, 2 * b_in, 2))
            ref_so3 = RefSO3Conv(4, 2, b_out, b_out, ref_so3_grid(math.pi / 8, 2 * math.pi, 2 * b_out, 2, 2))
        s2 = S2Convolution(3, 4, b_in, b_out, s2_pts)
        so3 = SO3Convolution(4, 2, b_out, b_out, so3_pts)
        errors["s2_grid"] = _relative(s2_pts, torch.tensor(ref_s2_grid(math.pi / 8, 2 * b_in, 2), dtype=dtype))
        errors["so3_grid"] = _relative(so3_pts, torch.tensor(ref_so3_grid(math.pi / 8, 2 * math.pi, 2 * b_out, 2, 2), dtype=dtype))
        with torch.no_grad():
            for mine, theirs in ((s2, ref_s2), (so3, ref_so3)):
                theirs.kernel.copy_(mine.kernel)
                theirs.bias.normal_(generator=g)
                mine.bias.copy_(theirs.bias)
            assert abs(s2.scaling - ref_s2.scaling) < 1e-15 and abs(so3.scaling - ref_so3.scaling) < 1e-15
            with quiet:
                z_ref = ref_s2(x)
            z = s2(x)
            errors["s2_conv"] = _relative(z, z_ref)
            with quiet:
                errors["so3_conv"] = _relative(so3(z), ref_so3(z_ref))
    finally:
        torch.set_default_dtype(old)
    return errors


if __name__ == "__main__":
    for dtype in (torch.float32, torch.float64):
        out = compare_all(dtype)
        print(dtype, {k: f"{v:.2e}" for k, v in out.items()})
        print("worst", max(out.values()))
