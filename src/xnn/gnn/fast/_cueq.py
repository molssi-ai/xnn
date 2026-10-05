"""cuEquivariance backend of the gnn fast paths.

`cuEquivariance <https://github.com/NVIDIA/cuEquivariance>`_ provides fused CUDA
kernels for equivariant tensor products. Its Python packages are Apache-2.0; its
kernels (``cuequivariance-ops-torch-cu12``) are binaries distributed by NVIDIA
under NVIDIA's license, so it is an optional install (the three packages of one
release, ``pip install cuequivariance==0.6.1 cuequivariance-torch==0.6.1
cuequivariance-ops-torch-cu12==0.6.1``)
and xnn never needs it: without it, or off a GPU, every model runs its
reference implementation.

xnn feeds the kernels its own parameters. Weight layouts and path
normalizations differ between the two implementations, so each kernel carries a
fixed linear map from xnn's weights to the kernel's, determined once (in
float64) when the kernel is first built and checked to reproduce the reference
to rounding.
"""
from __future__ import annotations

import functools
import logging
import os
import warnings

import torch
from e3nn import o3

logger = logging.getLogger(__name__)

_broken: dict = {"reason": None}


@functools.lru_cache(maxsize=None)
def _importable() -> bool:
    try:
        import cuequivariance  # noqa: F401
        import cuequivariance_torch  # noqa: F401
        import cuequivariance_ops_torch  # noqa: F401
    except Exception as exc:  # ImportError, or a CUDA library the kernels cannot load
        logger.debug("cuEquivariance is not available: %s", exc)
        return False
    return True


def available() -> bool:
    """Whether cuEquivariance and its CUDA kernels can run here."""
    return _broken["reason"] is None and torch.cuda.is_available() and _importable()


def supported(device: torch.device, dtype: torch.dtype) -> bool:
    """Whether the kernels run for tensors on ``device`` in ``dtype``."""
    return device.type == "cuda" and dtype in (torch.float32, torch.float64) and available()


def disable(reason: str) -> None:
    """Turn the backend off for this process (a kernel failed to build or run)."""
    if _broken["reason"] is None:
        _broken["reason"] = reason
        warnings.warn(f"cuEquivariance fast paths are disabled for this process: {reason}. "
                      "The reference implementation is used instead.", RuntimeWarning, stacklevel=3)


def warn_fallback(block: str, exc: Exception) -> None:
    """Warn that one block keeps its reference implementation (its kernel failed to build).

    With the environment variable ``XNN_FAST_STRICT=1`` (parity checks) the
    failure is raised instead.
    """
    if os.environ.get("XNN_FAST_STRICT", "0") not in ("", "0"):
        raise RuntimeError(f"{block}: the cuEquivariance kernel failed to build") from exc
    warnings.warn(f"{block}: no cuEquivariance kernel ({type(exc).__name__}: {exc}); "
                  "this block uses the reference implementation", RuntimeWarning, stacklevel=3)


@functools.lru_cache(maxsize=None)
def e3nn_group():
    """O(3) in e3nn's real basis, as a cuEquivariance irrep class.

    cuEquivariance's own ``O3`` uses a different real basis (and coupling
    signs) than e3nn, whose features xnn's models carry. This subclass keeps
    its labels, ordering and parity rules and takes the coupling coefficients
    (e3nn's Wigner 3j symbols) and the Lie-algebra generators (derivatives of
    e3nn's Wigner D matrices) from e3nn, so every descriptor built on it acts
    on e3nn-layout features. Products of its irreps stay in the class.
    """
    import dataclasses
    import itertools

    import numpy as np
    import cuequivariance as cue

    # repr=False keeps O3's '1o' spelling (a generated repr would replace it)
    @dataclasses.dataclass(frozen=True, repr=False)
    class O3E3nn(cue.O3):
        def __mul__(rep1, rep2):
            rep2 = rep1._from(rep2)
            return [type(rep1)(l=ell, p=rep1.p * rep2.p)
                    for ell in range(abs(rep1.l - rep2.l), rep1.l + rep2.l + 1)]

        @classmethod
        def clebsch_gordan(cls, rep1, rep2, rep3) -> np.ndarray:
            rep1, rep2, rep3 = cls._from(rep1), cls._from(rep2), cls._from(rep3)
            if rep1.p * rep2.p == rep3.p and abs(rep1.l - rep2.l) <= rep3.l <= rep1.l + rep2.l:
                return o3.wigner_3j(rep1.l, rep2.l, rep3.l, dtype=torch.float64).numpy()[None]
            return np.zeros((0, rep1.dim, rep2.dim, rep3.dim))

        @classmethod
        def iterator(cls):
            for ell in itertools.count(0):
                yield cls(l=ell, p=1 * (-1) ** ell)
                yield cls(l=ell, p=-1 * (-1) ** ell)

        def continuous_generators(rep) -> np.ndarray:
            return _e3nn_generators(rep.l)

        def rotation(rep, axis: np.ndarray, angle: float) -> np.ndarray:
            import scipy.linalg
            axis = np.asarray(axis, dtype=np.float64)
            axis = axis / np.linalg.norm(axis)
            return scipy.linalg.expm(angle * np.einsum("i,ijk->jk", axis, _e3nn_generators(rep.l)))

    O3E3nn.__name__ = O3E3nn.__qualname__ = "O3E3nn"
    return O3E3nn


@functools.lru_cache(maxsize=None)
def _e3nn_generators(ell: int):
    """The (x, y, z) rotation generators of e3nn's degree-``ell`` irrep, ``(3, d, d)``.

    e3nn writes rotations as ``D = Y(alpha) X(beta) Y(gamma)``, so the x and y
    generators are the derivatives of ``wigner_D`` at the identity and z is
    their commutator (``[X, Y] = Z`` for the so(3) structure constants
    ``epsilon_ijk``, the algebra of cuEquivariance's O(3)).
    """
    import numpy as np

    zero = torch.zeros((), dtype=torch.float64)

    def d_beta(b):
        return o3.wigner_D(ell, zero, b, zero)

    def d_alpha(a):
        return o3.wigner_D(ell, a, zero, zero)

    gx = torch.autograd.functional.jacobian(d_beta, zero.clone())
    gy = torch.autograd.functional.jacobian(d_alpha, zero.clone())
    gz = gx @ gy - gy @ gx
    return np.stack([gx.numpy(), gy.numpy(), gz.numpy()])


def cue_irreps(irreps) -> "cuequivariance.Irreps":  # noqa: F821
    """e3nn irreps as cuEquivariance irreps of :func:`e3nn_group`."""
    import cuequivariance as cue
    return cue.Irreps(e3nn_group(), str(o3.Irreps(irreps)))


def check_close(ref: torch.Tensor, fast: torch.Tensor, what: str, tol: float = 1e-10) -> None:
    """Raise if a float64 calibration of ``what`` does not reproduce the reference."""
    scale = float(ref.abs().max())
    err = float((ref - fast).abs().max()) / (scale if scale > 0 else 1.0)
    if not err <= tol:
        raise RuntimeError(f"{what}: the cuEquivariance kernel differs from the reference "
                           f"by {err:.1e} (relative) after calibration")
