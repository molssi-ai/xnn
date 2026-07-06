.. _design:

***************
Design of xnns
***************

xnns is organized **by model family** — ``gnn`` (E(3)-equivariant graph
networks), ``cnn`` (continuous-filter convolutions), ``dnn`` (descriptor +
per-element networks) — with everything shared across families factored into
``common``. A component lives with the family that uses it, or in ``common``
when more than one family needs it; each layer still stands alone and can be
imported on its own.

.. code-block:: text

   src/xnns/
     common/       shared across all families
       data/         AtomicGraph (the one data object), PBC neighbor list,
                     AtomicDataset, batching
       featurizers/  Featurizer base + shared basis functions
                     (GaussianRBF, CosineCutoff)
       config/       one dataclass schema; loaders for yaml/toml/argparse/hydra
       models/       InteratomicPotential interface + registry +
                     ForceStressOutput + ops (scatter_sum)
       train/        Trainer (batch + device aware), weighted
                     energy/force/stress loss
       deploy/       ASE Calculator, LAMMPS/TorchScript export
       cli/          the `xnns` command
     gnn/          E(3)-equivariant GNNs (need e3nn)
       featurizers/  SphericalHarmonicEdgeEmbedding, BesselRBF, PolynomialCutoff
       models/       base (EquivariantGNN), blocks, nequip, mace, allegro
     cnn/          continuous-filter conv net
       models/       schnet
     dnn/          descriptor + per-element networks
       featurizers/  symmetry functions, AEV
       models/       base (DescriptorPotential), hdnnp, ani

Four ideas hold the package together.

1. One data object
==================
Every model consumes an :class:`~xnns.common.data.atomic_data.AtomicGraph`
and returns ``{"node_energy", "energy"}``. Molecular vs. periodic is
invisible to models — periodicity lives only in
:meth:`~xnns.common.data.atomic_data.AtomicGraph.edge_vectors`:

.. math::

   \mathbf{r}_{ij} = \mathbf{r}_j - \mathbf{r}_i + \mathbf{s}_{ij} \cdot \mathbf{h}

where :math:`\mathbf{s}_{ij}` is the integer cell shift of the edge and
:math:`\mathbf{h}` the cell matrix. Because the edge vectors are computed
from positions and cell inside the graph, forces and stress stay
differentiable end to end.

2. Featurizers are first-class
==============================
A :class:`~xnns.common.featurizers.base.Featurizer` (a subclass of
``nn.Module``) turns a graph into invariant descriptors (symmetry functions,
AEV) or equivariant edge attributes (spherical harmonics). Descriptor models
(HDNNP, ANI) and GNNs (NequIP, MACE, Allegro) are thin compositions over
featurizers, so the featurization is reusable and inspectable on its own:

.. code-block:: python

   from xnns.dnn.featurizers import AEV
   from xnns.gnn.featurizers import SphericalHarmonicEdgeEmbedding

   descriptor = AEV(species=[1, 6, 8])(graph)               # (N, D) invariant
   edges = SphericalHarmonicEdgeEmbedding(l_max=2)(graph)   # equivariant

3. Forces and stress in one place
=================================
:class:`~xnns.common.models.outputs.ForceStressOutput` wraps any model and
differentiates the predicted energy with respect to positions (forces,
:math:`\mathbf{F}_i = -\partial E / \partial \mathbf{r}_i`) and a symmetric
strain (stress). Models never implement forces themselves — a model is just
an energy function, and the physics of differentiation is written once.

4. Extensibility via registry + one config, four frontends
==========================================================
A model becomes available everywhere with two ingredients: the
``@register_model("name")`` decorator and a ``from_config`` classmethod. The
registry (:func:`~xnns.common.models.registry.build_model`) dispatches on
``cfg.model.name``, and the single :class:`~xnns.common.config.schema.Config`
dataclass is filled from any of four frontends — YAML, TOML, argparse, or
Hydra — which all funnel into the same place. Upstream config spellings are
handled by a loader-level key-translation registry, never by per-model
aliases (see :ref:`howto-upstream-configs`).

Everything downstream — data, featurizers, autograd forces/stress, training,
ASE/LAMMPS deployment — is identical across all models.
