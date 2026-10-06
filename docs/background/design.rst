.. _design:

***************
Design of xnn
***************

xnn is organized by model family, with everything shared in ``common``:

.. code-block:: text

   src/xnn/
     common/        data (AtomicGraph, neighbor lists, datasets, data hub), config,
                    models (registry, ForceStressOutput, model hub, LES, D3, D4),
                    finetune, train, benchmark, deploy (ASE, TorchScript, MDI), cli
     gnn/           NequIP, MACE, Allegro, CACE, AIMNet2, fused-kernel fast paths
     cnn/           SchNet
     dnn/           HDNNP, ANI, PhysNet
     ffnn/          ReaxFF, OPLS, DREIDING, the .frc force-field reader
     transformer/   edge attention and radial basis shared by attention models
     hybrid/        BAMBOO

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
