.. _featurizers:

***********
Featurizers
***********

A featurizer turns an :class:`~xnn.common.data.atomic_data.AtomicGraph` into a
model input. All featurizers subclass
:class:`~xnn.common.featurizers.base.Featurizer`, which is derived from
``nn.Module`` with an ``output_dim`` property and ``forward(data)`` method. As
such, xnn features are trainable, scriptable and composable components like any
other PyTorch module and can be used independently for any type of study and
analysis.

Shared basis functions
======================
A few featurizers are shared across multiple model categories and are
implemented in the ``xnn.common.featurizers`` namespace:

- :class:`~xnn.common.featurizers.radial.GaussianRBF` represents a Gaussian
  expansion of distances (e.g., in SchNet), with an optional explicit width
  ``gamma``.
- :class:`~xnn.common.featurizers.radial.ExpNormalSmearing` is the
  exponential-normal basis of the TorchMD-Net models (ET, and TensorNet) and
  BAMBOO: Gaussians in exponential distance space under a cosine cutoff.
- :class:`~xnn.common.featurizers.cutoff.CosineCutoff` represents a smooth
  cosine envelope function.
- :class:`~xnn.common.featurizers.cutoff.MollifierCutoff` denotes a
  bump-function envelope, ``exp(-r^2 / (r_c^2 - r^2))``, which is smooth and
  differentiable to infinite order (e.g., it is employed in AIMNet2, SpookyNet).
- :class:`~xnn.common.featurizers.radial.BesselRBF` and
  :class:`~xnn.common.featurizers.cutoff.PolynomialCutoff` offer trainable
  Bessel basis functions and polynomial envelopes (NequIP, MACE, DimeNet and
  the Equiformers).

DNN atomic descriptors
======================
Dense neural networks such as HDNNP and ANI often use per-atom invariant
descriptors for featurization. The following featurizers are implemented in the
``xnn.dnn.featurizers`` namespace:

- :class:`~xnn.dnn.featurizers.symmetry_functions.RadialSymmetryFunctions` and
  :class:`~xnn.dnn.featurizers.symmetry_functions.AngularSymmetryFunctions`
  refer to the Behler-Parrinello G2's radial and angular symmetry functions,
  respectively.
- :class:`~xnn.dnn.featurizers.aev.AEV` is the ANI atomic environment vector.
- :class:`~xnn.dnn.featurizers.acsf.AtomCenteredSymmetryFunctions`  the HDNNP's
  original descriptors, a list of radial (types 1, 2) and angular (types 3, 8,
  9) functions per element, each with its own cutoff function (cosine,
  hyperbolic tangent, polynomial, hard), with `RuNNer's
  <https://gitlab.com/runner-suite/runner2>`_ conventions and feature scaling.

As an example of using these descriptors independently, we can use the ANI AEV
featurizer to featurize a graph of carbon, hydrogen and oxygen atoms:

.. code-block:: python

   from xnn.dnn.featurizers import AEV

   # Create an AEV featurizer for hydrogen, carbon and oxygen
   aev = AEV(species=[1, 6, 8])

   # A rotation invariant descriptor of shape (N, aev.output_dim) 
   descriptor = aev(graph)

CNN grid featurizers
====================
The following featurizers are implemented in the ``xnn.cnn.featurizers``
namespace and are used to create 3D grids for convolutional neural networks
(CNNs):

- :class:`~xnn.cnn.featurizers.voxel.VoxelGrid` uses one cubic grid per atom
  (with a side of ``2 * cutoff`` radius in size and ``grid_size`` voxels per
  axis) with a density channel per species. Here, each neighbor is deposited as
  a Gaussian under a cosine envelope function. The grids are differentiable with
  respect to the positions, which are the input to the 3D steerable and the
  conventional CNNs. Let's create a voxel grid for a graph of hydrogen, carbon
  and oxygen atoms with a cutoff of 4.0 Å and a grid size of 17:

  .. code-block:: python

     from xnn.cnn.featurizers import VoxelGrid

     # Create a VoxelGrid featurizer for hydrogen, carbon and oxygen atoms
     # with a cutoff of 4.0 Å and a grid size of 17
     vox = VoxelGrid(species=[1, 6, 8], cutoff=4.0, grid_size=17)

     # (N, 3, 17, 17, 17) density fields
     grids = vox(graph)           

- :class:`~xnn.cnn.featurizers.spherical.SphericalGrid` creates one sphere per
  atom (with a radius of ``radius``, Driscoll-Healy grid of bandwidth
  ``bandwidth``) with a channel per species holding the potential of the
  neighbors of that species. The molecular representation of the spherical CNN
  is differentiable with respect to the atomic positions. As an example, we can
  create a spherical grid for a graph of hydrogen, carbon, nitrogen, oxygen and
  sulfur atoms with a cutoff of 10.0 Å, a radius of 0.48 Å and a bandwidth of 10:

  .. code-block:: python
  
     from xnn.cnn.featurizers import SphericalGrid
  
     # Create a SphericalGrid featurizer for hydrogen, carbon, nitrogen, oxygen and sulfur atoms
     # with a cutoff of 10.0 Å, a radius of 0.48 Å and a bandwidth of 10
     sph = SphericalGrid(species=[1, 6, 7, 8, 16], cutoff=10.0, radius=0.48, bandwidth=10)

     # (N, 5, 20, 20) spherical signals 
     signals = sph(graph)

Equivariant edge features for GNNs
==================================
The following featurizers are implemented in the ``xnn.gnn.featurizers``
namespace and are often used to create edge features for graph neural networks
(GNNs):

- :class:`~xnn.gnn.featurizers.spherical.SphericalHarmonicEdgeEmbedding` encode
  edge lengths, unit direction vectors or angles using real spherical harmonics
  with degrees up to ``l_max`` and a radial expansion (e.g., in NequIP, MACE,
  and Allegro).
- :class:`~xnn.gnn.featurizers.cartesian.CartesianAngularBasis` featurizes edges
  using the Cartesian monomials of CACE, spanning the latent space without using
  e3nn.
- :class:`~xnn.gnn.featurizers.spherical_bessel.SphericalBesselBasis` provides
  the 2D spherical Fourier-Bessel basis of DimeNet and GemNet with
  ``n_spherical`` degrees times ``n_radial`` Bessel roots over a distance and an
  angle. These featurizes are implemented in plain PyTorch with no dependency
  on scipy.
- :func:`~xnn.gnn.featurizers.spherical_bessel.real_spherical_harmonics`
  represents real spherical harmonics as Cartesian polynomials which are smooth
  at the poles. Using the radial featurizers mentioned above, they can form
  GemNet's 3D basis over a distance and two angles.

  .. code-block:: python

     from xnn.gnn.featurizers import SphericalHarmonicEdgeEmbedding, CartesianAngularBasis

     # Compute edge features from a graph's edge vectors using spherical
     harmonics lengths, edge_sh, edge_radial =
     SphericalHarmonicEdgeEmbedding(l_max=2).embed(graph.edge_vectors())

     unit = graph.edge_vectors()

     # Compute Cartesian angular features from graph's unit edge vectors
     angular = CartesianAngularBasis(l_max=3)(unit / unit.norm(dim=-1, keepdim=True))

Transformer building blocks
===========================
:class:`~xnn.trfnn.attention.EdgeMultiheadAttention` is the multi-head
QKV attention on graph edges that BAMBOO builds on. Tt is implemented in the
``xnn.trfnn`` namespace so that other attention-based models can employ
it as well.

Hybrid radial basis featurizers
===============================
The following featurizers are implemented in the ``xnn.hybnn.featurizers``
namespace and are often used to create radial basis functions for hybrid models
such as SpookyNet:

- :class:`~xnn.hybnn.featurizers.radial.ExponentialBernsteinRBF` provides the
  Bernstein polynomials of ``exp(-gamma r)`` with a learned decay rate. These
  featurizers construct the radial basis of SpookyNet.

Writing a new featurizer is relatively straightforward. See the
:ref:`developer-guide-extending` guide for more details.
