"""Transformer models: the TorchMD-Net equivariant transformer (ET), Equiformer and EquiformerV2."""
import warnings

from .et import EquivariantTransformer

__all__ = ["EquivariantTransformer"]

# the Equiformers need e3nn (irreps, tensor products, coupling coefficients); without it
# ET alone is registered. Catch Exception, not just ImportError: e3nn does real work
# at import time
try:
    import torch

    try:
        with torch.serialization.safe_globals([slice]):
            import e3nn.o3  # noqa: F401
    except AttributeError:    # torch < 2.4 has no safe_globals
        import e3nn.o3  # noqa: F401
    from .equiformer import Equiformer
    from .equiformer_v2 import EquiformerV2
    __all__ += ["Equiformer", "EquiformerV2"]
    HAS_EQUIFORMER = True
except Exception as _error:  # noqa: BLE001 - optional dependency, degrade quietly
    warnings.warn(f"xnn: the Equiformer models are unavailable: {type(_error).__name__}: {_error}",
                  stacklevel=2)
    HAS_EQUIFORMER = False
