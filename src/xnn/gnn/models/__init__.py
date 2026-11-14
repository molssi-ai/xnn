"""Graph-network models: SchNet, DimeNet, DimeNet++, GemNet, PaiNN, TensorNet, QET and MEGNet
(plain PyTorch) and the E(3)-equivariant NequIP / MACE / Allegro / CACE / AIMNet2 (e3nn)."""
import warnings

from .dimenet import DimeNet, DimeNetPP, directed_triplets
from .gemnet import GemNet
from .megnet import MEGNet
from .megnet_foundation import FOUNDATION_MODELS as MEGNET_FOUNDATION_MODELS
from .painn import GatedEquivariantBlock, PaiNN
from .qet import QET
from .qet_foundation import FOUNDATION_MODELS as QET_FOUNDATION_MODELS
from .schnet import SchNet
from .tensornet import TensorNet
from .tensornet_foundation import FOUNDATION_MODELS as TENSORNET_FOUNDATION_MODELS

__all__ = ["SchNet", "DimeNet", "DimeNetPP", "directed_triplets", "GemNet", "PaiNN",
           "GatedEquivariantBlock", "TensorNet", "TENSORNET_FOUNDATION_MODELS", "QET",
           "QET_FOUNDATION_MODELS", "MEGNet", "MEGNET_FOUNDATION_MODELS"]

# the equivariant models need e3nn; without it SchNet, DimeNet, GemNet, PaiNN, MEGNet,
# TensorNet and QET alone are registered. Catch Exception, not just ImportError: e3nn does real work at
# import time (it loads its Wigner constants with torch.load) and can fail in
# other ways
try:
    from xnn.common.models.constants import exact_float64_constants

    from .aimnet2 import AIMNet2
    from .aimnet2_foundation import FOUNDATION_MODELS as AIMNET2_FOUNDATION_MODELS
    from .aimnet2_foundation import from_aimnet_artifact
    from .allegro import Allegro
    from .base import EquivariantGNN, GNNPotential
    from .blocks import RadialNet, set_recompute_radial
    from .cace import CACE
    from .mace import MACE, set_message_blocks
    from .mace_foundation import FOUNDATION_MODELS, from_mace_torch
    from .nequip import NequIP
    __all__ += ["EquivariantGNN", "GNNPotential", "NequIP", "MACE", "Allegro",
                "CACE", "AIMNet2", "FOUNDATION_MODELS", "AIMNET2_FOUNDATION_MODELS",
                "from_mace_torch", "from_aimnet_artifact", "exact_float64_constants",
                "RadialNet", "set_recompute_radial", "set_message_blocks"]
    HAS_EQUIVARIANT = True
except Exception as _error:  # noqa: BLE001 - optional dependency, degrade quietly
    warnings.warn(
        f"xnn: the equivariant GNN models (NequIP/MACE/Allegro/CACE/AIMNet2) are "
        f"unavailable: {type(_error).__name__}: {_error}", stacklevel=2)
    HAS_EQUIVARIANT = False
