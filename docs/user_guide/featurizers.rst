.. _featurizers:

***********
Featurizers
***********

A featurizer turns an :class:`~xnn.common.data.atomic_data.AtomicGraph` into
model inputs. All featurizers subclass
:class:`~xnn.common.featurizers.base.Featurizer` (an ``nn.Module`` with an
``output_dim`` property and a ``forward(data)`` method), so they can be
trained, scripted, and composed like any other module, and used standalone
for analysis.

Shared basis functions (``xnn.common.featurizers``)
====================================================
- :class:`~xnn.common.featurizers.radial.GaussianRBF`: Gaussian radial
  basis expansion of distances (used by SchNet); the optional ``gamma``
  fixes the width explicitly (SchNet's ``gamma = 10`` Å\ :sup:`-2`) instead
  of tying it to the center spacing.
- :class:`~xnn.common.featurizers.cutoff.CosineCutoff`: smooth cosine
  cutoff envelope.

Descriptor featurizers (``xnn.dnn.featurizers``)
=================================================
Invariant per-atom descriptors for HDNNP/ANI-style models:

- :class:`~xnn.dnn.featurizers.symmetry_functions.RadialSymmetryFunctions`:
  Behler–Parrinello G2 radial symmetry functions.
- :class:`~xnn.dnn.featurizers.symmetry_functions.AngularSymmetryFunctions`:
  angular symmetry functions over atomic triplets (built with
  :func:`~xnn.dnn.featurizers.symmetry_functions.build_triplets`).
- :class:`~xnn.dnn.featurizers.aev.AEV`: the ANI atomic environment vector
  (radial + angular parts, per species pair).

.. code-block:: python

   from xnn.dnn.featurizers import AEV

   aev = AEV(species=[1, 6, 8])
   descriptor = aev(graph)      # (N, aev.output_dim), rotation invariant

Equivariant featurizers (``xnn.gnn.featurizers``)
==================================================
Edge attributes for the GNN models:

- :class:`~xnn.gnn.featurizers.spherical.SphericalHarmonicEdgeEmbedding`:
  the standard NequIP/MACE/Allegro edge embedding, with edge lengths, real
  spherical harmonics :math:`Y_{lm}(\hat r_{ij})` up to ``l_max``, and a
  radial expansion.
- :class:`~xnn.gnn.featurizers.cartesian.CartesianAngularBasis`: the CACE
  angular basis, i.e. the Cartesian monomials
  :math:`x^{l_x} y^{l_y} z^{l_z}` up to ``l_max``, spanning the same space
  as the spherical harmonics per total :math:`l` without e3nn.
- :class:`~xnn.gnn.featurizers.radial.BesselRBF`: (trainable) Bessel radial
  basis.
- :class:`~xnn.gnn.featurizers.cutoff.PolynomialCutoff`: the polynomial
  cutoff envelope of NequIP/MACE.

.. code-block:: python

   from xnn.gnn.featurizers import SphericalHarmonicEdgeEmbedding

   embed = SphericalHarmonicEdgeEmbedding(l_max=2)
   lengths, edge_sh, edge_radial = embed.embed(graph.edge_vectors())

.. code-block:: python

   from xnn.gnn.featurizers import CartesianAngularBasis

   basis = CartesianAngularBasis(l_max=3)
   unit = graph.edge_vectors()
   angular = basis(unit / unit.norm(dim=-1, keepdim=True))   # (E, 20)

Writing your own featurizer is a small task; see
:ref:`developer-guide-extending`.
