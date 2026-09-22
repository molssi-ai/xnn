"""xnn: machine-learning interatomic potentials in PyTorch.

Organized by model family, with everything shared factored into ``common``:

    common/  abstractions used across all families
        data        -- AtomicGraph, neighbor lists, AtomicDataset, batching
        featurizers -- Featurizer base + shared basis functions (GaussianRBF, CosineCutoff)
        config      -- one schema, loaders for yaml/argparse/hydra
        models      -- InteratomicPotential contract + registry + ForceStressOutput + ops
        train       -- Trainer, losses (batch + device aware)
        deploy      -- ASE calculator, LAMMPS/TorchScript export
        cli         -- the `xnn` command
    gnn/     E(3)-equivariant GNNs (NequIP / MACE / Allegro / CACE); needs e3nn
    cnn/     continuous-filter conv net (SchNet)
    dnn/     descriptor + per-element networks (HDNNP / ANI / PhysNet)
    ffnn/    learnable classical force fields (ReaxFF / ReaxFF-nn / OPLS)
    transformer/ shared graph-transformer building blocks (attention, radial basis)
    hybrid/  GNN + transformer potentials with a physics energy split (BAMBOO)

Importing a family package registers its models, e.g.:
    from xnn.common.data import AtomicDataset
    from xnn.common.models import build_model
    from xnn.gnn.models.mace import MACE
"""
from . import common  # noqa: F401  (data, config, models, train, deploy, cli)

# importing the family packages registers their models by name; the hybrid
# family (BAMBOO), the classical force fields (ffnn) and the shared
# transformer building blocks need no e3nn
from . import cnn, dnn, ffnn, hybrid, transformer  # noqa: F401

# the GNN family (NequIP/MACE/Allegro) requires e3nn; register only if available.
# Catch Exception, not just ImportError: e3nn does real work at import time --
# it loads its Wigner constants with torch.load -- so it can fail in ways that
# are not import errors, and an optional family must not take the whole package
# with it when it does.
try:
    from . import gnn  # noqa: F401
    _HAS_GNN = True
except Exception as _gnn_error:  # noqa: BLE001 - optional family, degrade quietly
    import warnings as _warnings

    _warnings.warn(
        f"xnn: the GNN family (NequIP/MACE/Allegro/CACE) is unavailable: "
        f"{type(_gnn_error).__name__}: {_gnn_error}",
        stacklevel=2,
    )
    _HAS_GNN = False

__version__ = "0.2.1"
__all__ = ["common", "cnn", "dnn", "ffnn", "hybrid", "transformer",
           "__version__"]
