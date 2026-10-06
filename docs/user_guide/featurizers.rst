.. _featurizers:

***********
Featurizers
***********

A featurizer turns an :class:`~xnn.common.data.atomic_data.AtomicGraph` into
model inputs. All subclass :class:`~xnn.common.featurizers.base.Featurizer`,
an ``nn.Module`` with an ``output_dim`` property and ``forward(data)``, so
they train, script and compose like any module and can be used on their
own for analysis.

Shared basis functions (``xnn.common.featurizers``)
===================================================
- :class:`~xnn.common.featurizers.radial.GaussianRBF`: Gaussian expansion of
  distances (SchNet), with an optional explicit width ``gamma``.
- :class:`~xnn.common.featurizers.cutoff.CosineCutoff`: smooth cosine
  envelope.

Descriptors (``xnn.dnn.featurizers``)
=====================================
Invariant per-atom descriptors for the HDNNP and ANI models:

- :class:`~xnn.dnn.featurizers.symmetry_functions.RadialSymmetryFunctions`
  and :class:`~xnn.dnn.featurizers.symmetry_functions.AngularSymmetryFunctions`:
  Behler-Parrinello G2 and angular symmetry functions.
- :class:`~xnn.dnn.featurizers.aev.AEV`: the ANI atomic environment vector.

.. code-block:: python

   from xnn.dnn.featurizers import AEV

   descriptor = AEV(species=[1, 6, 8])(graph)      # (N, output_dim), rotation invariant

Equivariant edge features (``xnn.gnn.featurizers``)
===================================================
- :class:`~xnn.gnn.featurizers.spherical.SphericalHarmonicEdgeEmbedding`:
  edge lengths, real spherical harmonics up to ``l_max`` and a radial
  expansion (NequIP, MACE, Allegro).
- :class:`~xnn.gnn.featurizers.cartesian.CartesianAngularBasis`: the
  Cartesian monomials of CACE, spanning the same space without e3nn.
- :class:`~xnn.gnn.featurizers.radial.BesselRBF` and
  :class:`~xnn.gnn.featurizers.cutoff.PolynomialCutoff`: the trainable
  Bessel basis and polynomial envelope of NequIP and MACE.

.. code-block:: python

   from xnn.gnn.featurizers import SphericalHarmonicEdgeEmbedding, CartesianAngularBasis

   lengths, edge_sh, edge_radial = SphericalHarmonicEdgeEmbedding(l_max=2).embed(graph.edge_vectors())

   unit = graph.edge_vectors()
   angular = CartesianAngularBasis(l_max=3)(unit / unit.norm(dim=-1, keepdim=True))   # (E, 20)

Transformer pieces (``xnn.transformer``)
========================================
:class:`~xnn.transformer.featurizers.ExpNormalSmearing` (the exponential-normal
radial basis) and :class:`~xnn.transformer.attention.EdgeMultiheadAttention`
are the building blocks of BAMBOO, kept separate for other attention-based
models.

Writing a featurizer is a small task; see :ref:`developer-guide-extending`.
