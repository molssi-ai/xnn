"""xnns: machine-learning interatomic potentials in PyTorch.

Organized by model family, with everything shared factored into ``common``:

    common/  abstractions used across all families
        data        -- AtomicGraph, neighbor lists, AtomicDataset, batching
        featurizers -- Featurizer base + shared basis functions (GaussianRBF, CosineCutoff)
        config      -- one schema, loaders for yaml/toml/argparse/hydra
        models      -- InteratomicPotential contract + registry + ForceStressOutput + ops
        train       -- Trainer, losses (batch + device aware)
        deploy      -- ASE calculator, LAMMPS/TorchScript export
        cli         -- the `xnns` command
    gnn/     E(3)-equivariant GNNs (NequIP / MACE / Allegro); needs e3nn
    cnn/     continuous-filter conv net (SchNet)
    dnn/     descriptor + per-element networks (HDNNP / ANI)

Importing a family package registers its models, e.g.:
    from xnns.common.data import AtomicDataset
    from xnns.common.models import build_model
    from xnns.gnn.models.mace import MACE
"""
from . import common  # noqa: F401  (data, config, models, train, deploy, cli)

# importing the family packages registers their models by name
from . import cnn, dnn  # noqa: F401

# the GNN family (NequIP/MACE/Allegro) requires e3nn; register only if available
try:
    from . import gnn  # noqa: F401
    _HAS_GNN = True
except ImportError:
    _HAS_GNN = False

__version__ = "0.1.0"
__all__ = ["common", "cnn", "dnn", "__version__"]
