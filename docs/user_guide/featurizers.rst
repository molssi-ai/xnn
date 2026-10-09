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
- :class:`~xnn.dnn.featurizers.acsf.AtomCenteredSymmetryFunctions`: the
  HDNNP descriptor, a list of radial (types 1, 2) and angular (types 3, 8, 9)
  functions per element, each with its own cutoff function (cosine,
  hyperbolic tangent, polynomial, hard), with RuNNer's conventions and
  feature scaling.

.. code-block:: python

   from xnn.dnn.featurizers import AEV

   aev = AEV(species=[1, 6, 8])
   descriptor = aev(graph)      # (N, aev.output_dim), rotation invariant

Grid featurizers (``xnn.cnn.featurizers``)
===========================================
- :class:`~xnn.cnn.featurizers.voxel.VoxelGrid`: one cubic grid per atom
  (side ``2 * cutoff``, ``grid_size`` voxels per axis) with a density
  channel per species, each neighbor deposited as a Gaussian under a
  cosine envelope; differentiable in the positions, the input of the 3D
  steerable CNN and the conventional 3D CNN.

.. code-block:: python

   from xnn.cnn.featurizers import VoxelGrid

   vox = VoxelGrid(species=[1, 6, 8], cutoff=4.0, grid_size=17)
   grids = vox(graph)           # (N, 3, 17, 17, 17) density fields

- :class:`~xnn.cnn.featurizers.spherical.SphericalGrid`: one sphere per
  atom (radius ``radius``, Driscoll-Healy grid of bandwidth ``bandwidth``)
  with a channel per species holding the potential of the neighbors of that
  species, the molecular representation of the spherical CNN paper;
  differentiable in the positions.

.. code-block:: python

   from xnn.cnn.featurizers import SphericalGrid

   sph = SphericalGrid(species=[1, 6, 7, 8, 16], cutoff=10.0, radius=0.48, bandwidth=10)
   signals = sph(graph)         # (N, 5, 20, 20) spherical signals

Equivariant edge features (``xnn.gnn.featurizers``)
===================================================
- :class:`~xnn.gnn.featurizers.spherical.SphericalHarmonicEdgeEmbedding`:
  edge lengths, real spherical harmonics up to ``l_max`` and a radial
  expansion (NequIP, MACE, Allegro).
- :class:`~xnn.gnn.featurizers.cartesian.CartesianAngularBasis`: the
  Cartesian monomials of CACE, spanning the same space without e3nn.
- :class:`~xnn.gnn.featurizers.radial.BesselRBF` and
  :class:`~xnn.gnn.featurizers.cutoff.PolynomialCutoff`: the trainable
  Bessel basis and polynomial envelope of NequIP, MACE and DimeNet.
- :class:`~xnn.gnn.featurizers.spherical_bessel.SphericalBesselBasis`: the
  2D spherical Fourier-Bessel basis of DimeNet, ``n_spherical`` degrees
  times ``n_radial`` Bessel roots over a distance and an angle; plain
  PyTorch, no scipy.

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
