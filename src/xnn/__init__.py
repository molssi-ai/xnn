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
    gnn/     graph networks: SchNet / DimeNet / GemNet / PaiNN / TensorNet, and
             NequIP / MACE / Allegro / CACE / AIMNet2 (need e3nn)
    cnn/     volumetric 3D CNNs over voxelized environments (CNN3D, 3D steerable CNN)
    dnn/     dense neural networks (HDNNP / ANI / PhysNet)
    ffnn/    learnable classical force fields (ReaxFF / ReaxFF-nn / OPLS / DREIDING / AMBER)
    sqmnn/   semiempirical quantum chemistry (PM3 / PM6 / PM6-ORG / PM7)
    trfnn/   graph transformers (ET) and the shared edge attention
    hybnn/   GNN + transformer potentials with a physics energy split (BAMBOO, SpookyNet)

Importing a family package registers its models, e.g.:
    from xnn.common.data import AtomicDataset
    from xnn.common.models import build_model
    from xnn.gnn.models.mace import MACE
"""
from . import common  # noqa: F401  (data, config, models, train, deploy, cli)

# importing the family packages registers their models by name; the hybrid
# family (BAMBOO, SpookyNet), the classical force fields (ffnn), the semiempirical
# methods (sqmnn) and the trfnn family (ET) need no e3nn
from . import cnn, dnn, ffnn, hybnn, sqmnn, trfnn  # noqa: F401

# the GNN family registers SchNet, DimeNet, GemNet, PaiNN and TensorNet unconditionally and its
# equivariant models (NequIP/MACE/Allegro/CACE/AIMNet2) only when e3nn
# is available (it warns otherwise, see gnn/models/__init__.py)
from . import gnn  # noqa: F401

__version__ = "0.16.0"
__all__ = ["common", "gnn", "cnn", "dnn", "ffnn", "hybnn", "sqmnn", "trfnn",
           "__version__"]
