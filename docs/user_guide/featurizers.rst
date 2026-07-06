.. _featurizers:

***********
Featurizers
***********

A featurizer turns an :class:`~xnns.common.data.atomic_data.AtomicGraph` into
model inputs. All featurizers subclass
:class:`~xnns.common.featurizers.base.Featurizer` — an ``nn.Module`` with an
``output_dim`` property and a ``forward(data)`` method — so they can be
trained, scripted, and composed like any other module, and used standalone
for analysis.

Shared basis functions (``xnns.common.featurizers``)
====================================================
- :class:`~xnns.common.featurizers.radial.GaussianRBF` — Gaussian radial
  basis expansion of distances (used by SchNet).
- :class:`~xnns.common.featurizers.cutoff.CosineCutoff` — smooth cosine
  cutoff envelope.

Descriptor featurizers (``xnns.dnn.featurizers``)
=================================================
Invariant per-atom descriptors for HDNNP/ANI-style models:

- :class:`~xnns.dnn.featurizers.symmetry_functions.RadialSymmetryFunctions` —
  Behler–Parrinello G2 radial symmetry functions.
- :class:`~xnns.dnn.featurizers.symmetry_functions.AngularSymmetryFunctions` —
  angular symmetry functions over atomic triplets (built with
  :func:`~xnns.dnn.featurizers.symmetry_functions.build_triplets`).
- :class:`~xnns.dnn.featurizers.aev.AEV` — the ANI atomic environment vector
  (radial + angular parts, per species pair).

.. code-block:: python

   from xnns.dnn.featurizers import AEV

   aev = AEV(species=[1, 6, 8])
   descriptor = aev(graph)      # (N, aev.output_dim), rotation invariant

Equivariant featurizers (``xnns.gnn.featurizers``)
==================================================
Edge attributes for the E(3)-equivariant models:

- :class:`~xnns.gnn.featurizers.spherical.SphericalHarmonicEdgeEmbedding` —
  the standard NequIP/MACE/Allegro edge embedding: edge lengths, real
  spherical harmonics :math:`Y_{lm}(\hat r_{ij})` up to ``l_max``, and a
  radial expansion.
- :class:`~xnns.gnn.featurizers.radial.BesselRBF` — (trainable) Bessel radial
  basis.
- :class:`~xnns.gnn.featurizers.cutoff.PolynomialCutoff` — the polynomial
  cutoff envelope of NequIP/MACE.

.. code-block:: python

   from xnns.gnn.featurizers import SphericalHarmonicEdgeEmbedding

   embed = SphericalHarmonicEdgeEmbedding(l_max=2)
   lengths, edge_sh, edge_radial = embed.embed(graph.edge_vectors())

Writing your own featurizer is a small task — see
:ref:`developer-guide-extending`.
