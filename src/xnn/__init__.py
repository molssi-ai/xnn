"""xnn: machine-learning interatomic potentials in PyTorch.

Organized by model family, with everything shared factored into ``common``:

    common/  abstractions used across all families
        data        -- AtomicGraph, neighbor lists, AtomicDataset, batching
        featurizers -- Featurizer base + shared basis functions (GaussianRBF, CosineCutoff)
        config      -- one schema, loaders for yaml/argparse/hydra
        models      -- InteratomicPotential contract + registry + ForceStressOutput + ops
        finetune    -- multi-head replay, LoRA, reference energies for pretrained models
        train       -- Trainer, losses (batch + device aware)
        deploy      -- ASE calculator, LAMMPS/TorchScript export
        cli         -- the `xnn` command
    gnn/     graph networks: SchNet, and NequIP / MACE / Allegro / CACE / AIMNet2 (need e3nn)
    cnn/     volumetric 3D CNNs over voxelized environments (CNN3D, 3D steerable CNN)
    dnn/     dense neural networks (HDNNP / ANI / PhysNet)
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

# the GNN family registers SchNet unconditionally and its equivariant models
# (NequIP/MACE/Allegro/CACE/AIMNet2) only when e3nn is available (it warns
# otherwise, see gnn/models/__init__.py)
from . import gnn  # noqa: F401

__version__ = "0.9.0"
__all__ = ["common", "gnn", "cnn", "dnn", "ffnn", "hybrid", "transformer",
           "__version__"]
