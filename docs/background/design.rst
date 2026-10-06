.. _design:

***************
Design of xnn
***************

xnn is organized by model family, with everything shared in ``common``:

.. code-block:: text

   src/xnn/
     common/
       data/          AtomicGraph, neighbor lists, AtomicDataset, hub/ (load_dataset)
       featurizers/   Featurizer base, GaussianRBF, CosineCutoff
       config/        Config schema, YAML / argparse / Hydra loaders, key translation
       models/        InteratomicPotential, registry, ForceStressOutput, les, d3, d4,
                      hub/ (from_pretrained), dispersion_fast/ (Triton kernels)
       finetune/      LoRA, multi-head replay, freezing, reference energies
       train/         Trainer, losses
       benchmark/     scoring pre-trained models
       deploy/        ASE calculator, TorchScript export, MDI engine
       cli/           the xnn command
     gnn/
       featurizers/   spherical harmonics, Cartesian monomials, Bessel, polynomial cutoff
       models/        schnet, nequip, mace, allegro, cace, aimnet2 (+ foundation loaders)
       fast/          cuEquivariance fast paths
     cnn/
       featurizers/   voxel grid of the atomic environment
       models/        3D steerable CNN, conventional 3D CNN
     dnn/
       featurizers/   symmetry functions, AEV
       models/        hdnnp, ani, physnet
     ffnn/
       common/        .frc force-field reader, SMARTS atom typing
       data/          the shipped .frc files
       models/        reaxff, opls, dreiding, topology
     transformer/
       featurizers/   exponential-normal radial basis
       attention.py   edge multi-head attention
     hybrid/
       models/        bamboo

Four ideas hold it together.

1. **One data object.** Every model takes an
   :class:`~xnn.common.data.atomic_data.AtomicGraph` and returns
   ``{"node_energy", "energy"}``. Periodicity lives only in the edge
   vectors, :math:`\mathbf{r}_{ij} = \mathbf{r}_j - \mathbf{r}_i +
   \mathbf{s}_{ij} \cdot \mathbf{h}`, so molecules and crystals look the same
   to a model and forces and stress stay differentiable.
2. **Featurizers are first-class.** Descriptors and equivariant edge features
   are standalone :class:`~xnn.common.featurizers.base.Featurizer` modules
   that models compose and you can inspect on their own.
3. **Physics in wrappers.** :class:`~xnn.common.models.outputs.ForceStressOutput`
   differentiates any model's energy for forces and stress; models never
   implement them. Long-range electrostatics (LES), dispersion (D3, D4) and
   the fast paths wrap any model the same way.
4. **One config, one registry.** ``@register_model`` plus ``from_config``
   expose a model to the YAML, argparse and Hydra frontends, which all fill
   one :class:`~xnn.common.config.schema.Config`. Datasets and pre-trained
   models register the same way behind ``load_dataset()`` and
   ``from_pretrained()``.
