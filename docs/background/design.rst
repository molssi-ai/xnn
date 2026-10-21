.. _design:

***************
Design of xnn
***************

xnn is organized by model family, with everything shared in ``common``:

.. code-block:: text

   src/xnn/
     common/
       data/          AtomicGraph, neighbor lists, AtomicDataset, hub/ (load_dataset)
       featurizers/   Featurizer base, GaussianRBF, ExpNormalSmearing, CosineCutoff
       config/        Config schema, YAML / argparse / Hydra loaders, key translation
       models/        InteratomicPotential, registry, ForceStressOutput, les, d3, d4,
                      blocks (gated equivariant block), hub/ (from_pretrained),
                      dispersion_fast/ (Triton kernels)
       finetune/      LoRA, multi-head replay, freezing, reference energies
       train/         Trainer, losses
       benchmark/     scoring pre-trained models
       deploy/        ASE calculator, TorchScript export, MDI engine
       cli/           the xnn command
     gnn/
       featurizers/   spherical harmonics, Cartesian monomials, Bessel, spherical Bessel, polynomial cutoff
       models/        schnet, dimenet, gemnet, painn, tensornet, nequip, mace, allegro, cace,
                      aimnet2 (+ foundation loaders)
       fast/          cuEquivariance fast paths
     cnn/
       featurizers/   voxel grids and spherical signals of the atomic environment
       models/        steerable (SE(3) CNN), cnn3d, spherical (S^2 / SO(3) correlations)
     dnn/
       common/        RuNNer model files (input.nn, weights, scaling)
       featurizers/   atom-centered symmetry functions, AEV
       models/        hdnnp (1G to 4G), ani, physnet
     ffnn/
       common/        .frc force-field reader, SMARTS atom typing
       data/          the shipped .frc files
       models/        reaxff, opls, dreiding, amber, charmm, topology
     sqmnn/
       common/        MOPAC EXTERNAL reader/writer, published values (downloaded), element data
       models/        nddo (SCF, integrals), pm3, pm6, pm6-org, pm7
     trfnn/
       models/        et (the TorchMD-Net equivariant transformer)
       attention.py   edge multi-head attention
     hybnn/
       featurizers/   exponential Bernstein radial basis
       models/        bamboo, spookynet

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
