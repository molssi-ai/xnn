.. _design:

***************
Design of xnn
***************

xnn is organized by model family, with everything shared factored into
``common``. A component lives with the family that uses it, or in ``common``
when more than one family needs it, and each layer imports on its own.

.. code-block:: text

   src/xnn/
     common/        shared across families
       data/          AtomicGraph, neighbor lists, AtomicDataset, batching, the data hub
       featurizers/   Featurizer base, GaussianRBF, CosineCutoff
       config/        one dataclass schema; YAML / argparse / Hydra loaders; key translation
       models/        InteratomicPotential, registry, ForceStressOutput, the model hub,
                      the add-ons LES (les), DFT-D4 (d4), DFT-D3 (d3), the fast-path switch
       finetune/      LoRA, multi-head replay, freezing, reference-energy estimation
       train/         Trainer (batched, DDP-aware), losses
       benchmark/     scoring pre-trained models on a dataset
       deploy/        ASE calculator, TorchScript export, MDI engine
       cli/           the xnn command
     gnn/           NequIP, MACE, Allegro, CACE, AIMNet2; spherical and Cartesian featurizers;
                    fused-kernel fast paths
     cnn/           SchNet
     dnn/           HDNNP, ANI, PhysNet; symmetry functions, AEV
     ffnn/          ReaxFF, OPLS, DREIDING; the .frc force-field reader and SMARTS typing
     transformer/   edge attention and the exponential-normal basis
     hybrid/        BAMBOO

Four ideas hold the package together.

1. One data object
==================
Every model consumes an :class:`~xnn.common.data.atomic_data.AtomicGraph`
and returns ``{"node_energy", "energy"}``. Periodicity lives only in the
edge vectors,

.. math::

   \mathbf{r}_{ij} = \mathbf{r}_j - \mathbf{r}_i + \mathbf{s}_{ij} \cdot \mathbf{h},

with :math:`\mathbf{s}_{ij}` the integer cell shift of the edge and
:math:`\mathbf{h}` the cell. Molecules and crystals look the same to a
model, and forces and stress stay differentiable end to end.

2. Featurizers are first-class
==============================
A :class:`~xnn.common.featurizers.base.Featurizer` turns a graph into
invariant descriptors or equivariant edge attributes. Models are thin
compositions over featurizers, so the featurization is reusable and
inspectable on its own:

.. code-block:: python

   from xnn.dnn.featurizers import AEV
   from xnn.gnn.featurizers import SphericalHarmonicEdgeEmbedding

   descriptor = AEV(species=[1, 6, 8])(graph)               # (N, D) invariant
   edges = SphericalHarmonicEdgeEmbedding(l_max=2)(graph)   # equivariant

3. Physics in wrappers
======================
:class:`~xnn.common.models.outputs.ForceStressOutput` differentiates any
model's energy for forces and stress; models never implement them. The same
pattern adds long-range electrostatics
(:class:`~xnn.common.models.les.LatentEwald`) and dispersion
(:class:`~xnn.common.models.d4.D4Dispersion`,
:class:`~xnn.common.models.d3.D3Dispersion`) to any model, and the fast
paths swap a block's implementation without touching its parameters.

4. One config, one registry
===========================
``@register_model("name")`` plus a ``from_config`` classmethod make a model
available to every frontend: YAML, argparse and Hydra all fill one
:class:`~xnn.common.config.schema.Config`. Upstream spellings are handled by
a loader-level translation table, never by per-model aliases. Datasets and
pre-trained models register the same way, so ``load_dataset()`` and
``from_pretrained()`` grow by adding entries.

Everything downstream, from training to the deploy channels, is the same for
every model.
