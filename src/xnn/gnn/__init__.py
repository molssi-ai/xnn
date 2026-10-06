"""Graph-network family: SchNet, and NequIP / MACE / Allegro / CACE / AIMNet2 (these need e3nn)."""
import torch

# e3nn 0.4.4 (the version mace-torch pins) reads its constants with torch.load
# when imported; from torch 2.6 that load is weights-only by default and rejects
# the slice objects the file holds, so allow exactly those for the import
try:
    with torch.serialization.safe_globals([slice]):
        import e3nn.o3  # noqa: F401
except (AttributeError, ImportError):    # torch < 2.4 has no safe_globals
    pass

from . import featurizers, models  # noqa: E402

__all__ = ["featurizers", "models"]
