.. _models:

******
Models
******

Every model takes an :class:`~xnn.common.data.atomic_data.AtomicGraph` and
returns predicted properties such as ``node_energy`` and ``energy``. The
:class:`~xnn.common.models.outputs.ForceStressOutput` class enables the models
to compute atomic forces and stress using autograd. Models are built by name
from a config, or constructed directly with the same keys:

.. code-block:: python

   from xnn.common.models import available_models, build_model, ForceStressOutput

   available_models()               # ['aimnet2', 'allegro', 'ani', ...]
   model = build_model(cfg.model)   # Dispatches to <Model>.from_config(cfg.model)

   # Wrap the model to compute forces and stress
   model = ForceStressOutput(build_model(cfg.model), compute_stress=True)
   out = model(graph)               # energy, node_energy, forces, stress

Pre-trained models can be fetched with ``from_pretrained()`` from the model hub
(see :ref:`howto-pretrained-models` for more details). NequIP, MACE, Allegro and
the SE(3) steerable CNN require the ``gnn`` extra dependencies installed
(alongside e3nn).


MACE
====
The :class:`~xnn.gnn.models.mace.MACE` model (`ACEsuit/mace
<https://github.com/ACEsuit/mace>`_) offers the higher-order equivariant message
passing framework of `Batatia et al., Adv. Neural Inf. Process. Syst. 35, 11423
(2022) <https://doi.org/10.52202/068431-0830>`_.
Every published MACE-MP, MACE-OFF, etc. checkpoint can be fetched through the
xnn's model hub.

.. code-block:: yaml

   model:
     name: mace
     cutoff: 5.0
     species: [1, 6, 8]
     num_channels: 64
     num_interactions: 2         # Any depth from 0
     max_ell: 3
     correlation: 3
     avg_num_neighbors: 20.0
     atomic_energies: {1: -13.6, 6: -1029.9, 8: -2042.8}
     # foundation: mace-off23-small   # Uncomment to start from a pretrained checkpoint instead

Other options such as ``max_L``, ``hidden_irreps``, ``MLP_irreps``,
``radial_type``, ``distance_transform`` (Agnesi / Soft), ``pair_repulsion``
(ZBL), and ``scale`` / ``shift`` can also be set.

NequIP
======
:class:`~xnn.gnn.models.nequip.NequIP` (`mir-group/nequip
<https://github.com/mir-group/nequip>`_) is an E(3)-equivariant message passing
model with gated nonlinearities (`Batzner et al., Nat. Commun. 13, 2453 (2022)
<https://doi.org/10.1038/s41467-022-29939-5>`_).

.. code-block:: yaml

   model:
     name: nequip
     cutoff: 5.0
     species: [1, 6, 8]
     num_features: 32
     n_layers: 3
     l_max: 2
     parity: true
     avg_num_neighbors: 20.0
     atomic_energies: {1: -13.6, 6: -1029.9, 8: -2042.8}

Other options, including ``invariant_layers`` / ``invariant_neurons``,
``resnet``, ``use_sc``, ``trainable_rbf``, and ``atomic_scales`` are also
available.

Allegro
=======
:class:`~xnn.gnn.models.allegro.Allegro` (`mir-group/allegro
<https://github.com/mir-group/allegro>`_ v0.3.0) is a strictly local equivariant
many-body potential without message passing (`Musaelian et al., Nat. Commun. 14,
579 (2023) <https://doi.org/10.1038/s41467-023-36329-y>`_).

.. code-block:: yaml

   model:
     name: allegro
     cutoff: 5.0
     species: [1, 6, 8]
     num_layers: 2
     num_tensor_features: 16
     l_max: 1
     two_body_latent: [32, 64]
     latent: [64]
     avg_num_neighbors: 20.0
     atomic_energies: {1: -13.6, 6: -1029.9, 8: -2042.8}

Other options, including ``parity``, ``env_embed``, ``edge_eng``,
``latent_resnet``, and ``atomic_scales`` can also be used.

CACE
====
:class:`~xnn.gnn.models.cace.CACE` (`BingqingCheng/cace
<https://github.com/BingqingCheng/cace>`_) is the Cartesian atomic cluster
expansion of `Cheng, npj Comput. Mater. 10, 157 (2024)
<https://doi.org/10.1038/s41524-024-01332-4>`_. It builds
body-ordered invariants from Cartesian monomials, with optional message passing
(it does not required e3nn).

.. code-block:: yaml

   model:
     name: cace
     cutoff: 5.5
     species: [1, 6, 8]
     n_atom_basis: 3
     max_l: 3
     max_nu: 3
     num_message_passing: 1      # 0 gives plain Cartesian ACE
     avg_num_neighbors: 10.0

Other options, including ``message_types`` (M / Ar / Bchi), ``n_rbf``,
``readout_hidden`` and ``atomic_energies``, are also available.

AIMNet2
=======
:class:`~xnn.gnn.models.aimnet2.AIMNet2` (`isayevlab/aimnetcentral
<https://github.com/isayevlab/aimnetcentral>`_) is the atoms-in-molecules
network of `Anstine et al., Chem. Sci. 16, 10228 (2025)
<https://doi.org/10.1039/D4SC08572H>`_ for neutral,
charged and open-shell molecules. It predicts partial charges that sum exactly
to the net charge and adds their Coulomb energy. The six published families can
be fetched through the model hub.

.. code-block:: python

   from xnn.common.models import from_pretrained

   model = from_pretrained("aimnet2")                                    # wB97M-D3, with its D3 term
   model = from_pretrained("aimnet2", model_options={"coulomb": "pme"})  # Periodic cells: dsf / ewald / pme

The net charge and spin multiplicity come from the structure. To train from
scratch, set ``n_features``, ``n_rbf``, ``n_interactions``, ``hidden``,
``aim_size``, ``charge_channels`` (2 for open shells), ``coulomb`` and
``lr_cutoff``. The outputs also include ``charges`` and ``dipole``.

SchNet
======
:class:`~xnn.gnn.models.schnet.SchNet` is the continuous-filter convolutional
network of `Schütt et al., Adv. Neural Inf. Process. Syst. 30 (2017)
<https://proceedings.neurips.cc/paper/2017/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html>`_,
built from the paper with its architecture as the defaults.

.. code-block:: yaml

   model:
     name: schnet
     cutoff: 5.0
     cutoff_fn: cosine           # Smooth finite cutoff; the paper trains cutoff-free
     n_features: 64
     n_interactions: 3
     n_rbf: 50
     species: [1, 6, 8]
     atomic_energies: {1: -13.6, 6: -1029.9, 8: -2042.8}

Other options such as ``gamma`` and ``energy_shift`` / ``energy_scale`` can also
be set.

DimeNet and DimeNet++
=====================
:class:`~xnn.gnn.models.dimenet.DimeNet` and
:class:`~xnn.gnn.models.dimenet.DimeNetPP` (`gasteigerjo/dimenet
<https://github.com/gasteigerjo/dimenet>`_) are the directional message passing
networks of `Gasteiger et al., Int. Conf. Learn. Represent. (2020)
<https://arxiv.org/abs/2003.03123>`_ and
`Gasteiger et al., NeurIPS Workshop on Machine Learning for Molecules (2020)
<https://arxiv.org/abs/2011.14115>`_. Messages live on directed
edges and are updated from the distances and angles of neighboring edges.
DimeNet++ replaces the bilinear interaction with a cheaper one.

.. code-block:: yaml

   model:
     name: dimenet++             # Or dimenet
     cutoff: 5.0
     n_features: 128
     n_interactions: 4           # 6 for dimenet
     n_rbf: 6
     n_spherical: 7
     species: [1, 6, 8]
     atomic_energies: {1: -13.6, 6: -1029.9, 8: -2042.8}

Other options, including ``n_bilinear`` (DimeNet), ``n_triplet_features`` /
``n_basis_features`` / ``n_output_features`` (DimeNet++), ``p`` and
``trainable_rbf``, are also available.

GemNet
======
:class:`~xnn.gnn.models.gemnet.GemNet` (`TUM-DAML/gemnet_pytorch
<https://github.com/TUM-DAML/gemnet_pytorch>`_) is the geometric message
passing network of `Gasteiger et al., Adv. Neural Inf. Process. Syst. 34, 6790
(2021)
<https://proceedings.neurips.cc/paper/2021/hash/35cf8659cfcb13224cbd47863a34fc58-Abstract.html>`_.
Edge
embeddings are updated from neighboring edges (distances and angles) and, with
``quadruplets``, from edges two hops away (adding dihedral angles). The trainer
fits its scaling factors on the training structures before the first epoch.

.. code-block:: yaml

   model:
     name: gemnet
     cutoff: 5.0                 # Embedding cutoff
     n_features: 128
     n_interactions: 4
     n_rbf: 6
     int_cutoff: 10.0            # Graphs are built to this radius
     n_spherical: 7
     quadruplets: true           # false gives GemNet-T
     direct_forces: false        # true gives GemNet-dQ / -dT
     species: [1, 6, 8]
     atomic_energies: {1: -13.6, 6: -1029.9, 8: -2042.8}

Other options, including the feature sizes ``n_edge_features``,
``n_triplet_features`` and ``n_quadruplet_features``, ``scaling_factors`` and
``fit_scaling``, can also be set. Direct forces are not conservative, so they
cannot be combined with LES or dispersion terms or exported to TorchScript.

PaiNN
=====
:class:`~xnn.gnn.models.painn.PaiNN` is the equivariant message passing network
of `Schütt et al., Proc. Mach. Learn. Res. 139, 9377 (2021)
<https://proceedings.mlr.press/v139/schutt21a.html>`_ with scalar and vector
features, matching ``schnetpack``. Optional heads predict dipole moments and
polarizability tensors, trained with ``dipole_weight`` /
``polarizability_weight`` on data that has ``dipole`` / ``polarizability``
labels.

.. code-block:: yaml

   model:
     name: painn
     cutoff: 5.0
     n_features: 128
     n_interactions: 3
     n_rbf: 20
     dipole: true                # Outputs the dipole and latent charges
     polarizability: true
     species: [1, 6, 8]
     atomic_energies: {1: -13.6, 6: -1029.9, 8: -2042.8}

Other options, including ``radial_basis`` (bessel / gaussian),
``shared_filters``, ``atomic_dipoles`` and ``energy_shift`` /
``energy_scale``, are also available.

TensorNet
=========
:class:`~xnn.gnn.models.tensornet.TensorNet` (`torchmd/torchmd-net
<https://github.com/torchmd/torchmd-net>`_) is the Cartesian tensor network of
`Simeon and De Fabritiis, Adv. Neural Inf. Process. Syst. 36, 37334 (2023)
<https://doi.org/10.52202/075280-1623>`_, the model of
TorchMD-Net 2.0 (`Pelaez et al., J. Chem. Theory Comput. 20, 4076 (2024)
<https://doi.org/10.1021/acs.jctc.4c00253>`_). Here, every atom carries rank-2
tensors whose scalar, vector
and traceless parts are mixed by matrix products. A ``total_charge`` of the
structure enters the products.

.. code-block:: yaml

   model:
     name: tensornet
     cutoff: 4.5
     n_features: 128
     n_interactions: 2
     n_rbf: 32
     equivariance_group: O(3)    # SO(3) distinguishes enantiomers
     species: [1, 6, 8]
     atomic_energies: {1: -13.6, 6: -1029.9, 8: -2042.8}

Other options, including ``radial_basis`` (expnorm / gaussian),
``trainable_rbf``, ``activation`` and ``energy_shift`` / ``energy_scale``, are
also available.

SE(3) steerable CNN and 3D CNN
==============================
:class:`~xnn.cnn.models.steerable.SteerableCNN` is the rotation-equivariant
convolutional network of `Weiler et al., Adv. Neural Inf. Process. Syst. 31
(2018)
<https://proceedings.neurips.cc/paper/2018/hash/488e4104520c6aab692863cc1dba45af-Abstract.html>`_,
applied to a voxel grid around each atom (one density channel per species).
:class:`~xnn.cnn.models.cnn3d.CNN3D` is the conventional 3D CNN baseline with the
same grids and readout.

.. code-block:: yaml

   model:
     name: se3cnn                # Or cnn3d
     cutoff: 4.0                 # Half the side of the voxel cube
     species: [1, 6, 8]
     grid_size: 17               # Voxels per axis
     n_features: 32              # Channels of the last block
     n_interactions: 3           # Number of blocks
     l_max: 2                    # Highest field order (se3cnn only)
     kernel_size: 5
     atomic_energies: {1: -13.6, 6: -1029.9, 8: -2042.8}

Other options, including ``strides``, ``activation``, ``normalization``,
``bandlimit`` and ``sigma``, can also be set. Because the grid is discrete, the
steerable model is exactly invariant only under rotations that map the grid
onto itself, and approximately invariant under all others.

Spherical CNN
=============
:class:`~xnn.cnn.models.spherical.SphericalCNN` (`jonas-koehler/s2cnn
<https://github.com/jonas-koehler/s2cnn>`_) is the rotation-equivariant network
of `Cohen et al., Int. Conf. Learn. Represent. (2018)
<https://arxiv.org/abs/1801.10130>`_. Each atom sees its
neighbors as signals on a sphere around it, one channel per species, which
spherical convolutions map to rotation-invariant per-atom energies.

.. code-block:: yaml

   model:
     name: s2cnn
     cutoff: 10.0                # Neighbor radius of the potential sums
     species: [1, 6, 7, 8, 16]
     radius: 0.48                # Sphere radius in Angstrom
     bandwidth: 10               # 2b x 2b samples per sphere
     n_features: 160             # Channels of the last block
     n_interactions: 5           # ResNet blocks, as in the paper
     atomic_energies: {1: -13.6, 6: -1029.9, 7: -1485.3, 8: -2042.8, 16: -10831.3}

Other options, including ``features`` / ``bandwidths`` per block,
``set_readout`` (the paper's DeepSet readout), ``normalization`` and
``activation``, are also available.

ANI
===
:class:`~xnn.dnn.models.ani.ANI` (`aiqm/torchani
<https://github.com/aiqm/torchani>`_) is the per-element network over atomic
environment vectors of `Smith et al., Chem. Sci. 8, 3192 (2017)
<https://doi.org/10.1039/C6SC05720A>`_.
The published parameterizations are presets, and each has its training set in
the data hub:

.. code-block:: python

   from xnn.dnn.models import ANI

   ANI.ani1(species=[1, 6, 7, 8])      # ANI-1
   ANI.ani1x(species=[1, 6, 7, 8])     # ANI-1x
   ANI.ani1ccx(species=[1, 6, 7, 8])   # ANI-1ccx, coupled-cluster self energies
   ANI.ani2x()                         # ANI-2x, seven elements (H, C, N, O, S, F, Cl)

In a config, set ``preset: ani-2x`` (or ``ani-1``, ``ani-1x``, ``ani-1ccx``).
Without a preset, ``species``, ``radial_cutoff``, ``angular_cutoff``,
``hidden``, ``activation`` and ``atomic_energies`` define the model.

PhysNet
=======
:class:`~xnn.dnn.models.physnet.PhysNet` (`MMunibas/PhysNet
<https://github.com/MMunibas/PhysNet>`_) is the message passing network of
`Unke and Meuwly, J. Chem. Theory Comput. 15, 3678 (2019)
<https://doi.org/10.1021/acs.jctc.9b00181>`_ with explicit
electrostatics and D3 dispersion, translated from TensorFlow to PyTorch. It
embeds elements directly, so it needs no species list, and also returns
``charges`` and ``dipole``.

.. code-block:: yaml

   model:
     name: physnet
     cutoff: 10.0
     n_features: 128
     n_rbf: 64
     n_interactions: 5
     use_electrostatics: true
     use_dispersion: true

HDNNP
=====
High-dimensional neural network potentials cover the four generations of
the classification of `Behler, Chem. Rev. 121, 10037 (2021)
<https://doi.org/10.1021/acs.chemrev.0c00868>`_ and
match the RuNNer code:

- **1G** (``nnp1g``): one network maps the interatomic distances of a fixed
  system to its energy.
- **2G** (``hdnnp``): a sum of atomic energies from one network per element on
  atom-centered symmetry functions.
- **3G** (``generation: 3``): a second set of networks predicts charges, whose
  electrostatic energy is added.
- **4G** (``generation: 4``): a charge equilibration over the whole structure
  gives the charges, which allows non-local charge transfer.

.. code-block:: python

   from xnn.dnn.common.runner import load_runner_model

   model = load_runner_model("runner_model/")   # input.nn, weights and scaling files
   model.fit_scaling(train_graphs)              # Fit the symmetry-function scaling

The 3G and 4G models also return ``charges``, ``energy_elec`` and ``dipole``.
You can also train the charges with ``charge_weight`` on ``charges`` labels. The
``configs/model/hdnnp.yaml`` config file lists every available option.

Equivariant Transformer (ET)
============================
:class:`~xnn.trfnn.models.et.EquivariantTransformer` (`torchmd/torchmd-net
<https://github.com/torchmd/torchmd-net>`_) is the equivariant transformer of
`Thölke and De Fabritiis, Int. Conf. Learn. Represent. (2022)
<https://arxiv.org/abs/2202.02541>`_: scalar and vector
features updated by a distance-filtered multi-head attention. The default
``vector_cutoff`` applies the TorchMD-Net 2.0 fix that makes the energy
continuous at the cutoff. Setting it to ``false`` corresponds to the original
model.

.. code-block:: yaml

   model:
     name: et
     cutoff: 5.0
     n_features: 128
     n_interactions: 6
     n_rbf: 32
     num_heads: 8
     vector_cutoff: true
     species: [1, 6, 8]
     atomic_energies: {1: -13.6, 6: -1029.9, 8: -2042.8}

Other options, including ``neighbor_embedding``, ``distance_influence`` (both /
keys / values / none), ``attn_activation``, ``trainable_rbf`` and
``energy_shift`` / ``energy_scale``, are also available. The attention weights
of a structure are returned by ``model.attention_weights(graph)``.

Equiformer
==========
:class:`~xnn.trfnn.models.equiformer.Equiformer` (`atomicarchitects/equiformer
<https://github.com/atomicarchitects/equiformer>`_) is the equivariant graph
attention transformer of `Liao and Smidt, Int. Conf. Learn. Represent. (2023)
<https://arxiv.org/abs/2206.11990>`_. Irreps features pass through pre-norm
blocks of graph attention and feed-forward networks. The attention combines MLP
attention weights with non-linear messages built from depth-wise tensor
products.

.. code-block:: yaml

   model:
     name: equiformer
     cutoff: 5.0
     n_features: 128      # even scalars of the paper architecture
     n_interactions: 6
     n_rbf: 32
     l_max: 2             # 1 (OC20), 2 (QM9, MD17) or 3 (MD17)
     e3: false            # true: the E(3) model with both parities
     radial_basis: expnorm
     smooth_cutoff: true  # continuous energy at the cutoff (not in the paper)

``n_features`` scales the irreps of the paper architecture. Explicit irreps
(``irreps_node_embedding``, ``irreps_head``, ``irreps_mlp_mid``) override it.
Other options include ``attention: dot_product`` (the ablation baseline),
``nonlinear_message``, ``avg_degree`` / ``avg_num_nodes`` and the dropout rates.

EquiformerV2
============
:class:`~xnn.trfnn.models.equiformer_v2.EquiformerV2` (`fairchem
<https://github.com/facebookresearch/fairchem>`_) is the eSCN-based successor of
`Liao et al., Int. Conf. Learn. Represent. (2024)
<https://arxiv.org/abs/2306.12059>`_. The tensor products become SO(2) linear
maps in edge-aligned frames, which makes degrees up to 6 or 8 affordable. The
attention adds a layer norm on its scalars, the activations act on samples of
the sphere, and the layer norm treats degree 0 separately.

.. code-block:: yaml

   model:
     name: equiformer_v2
     cutoff: 5.0
     n_features: 96       # channels per coefficient
     n_interactions: 6
     n_rbf: 128
     l_max: 4
     m_max: 4             # orders kept in the edge frames (2 in the OC20 models)
     num_heads: 4
     smooth_cutoff: true
     # direct_forces: true  # the OC20 force head (not conservative)

Forces and stress from autograd include the dependence of the edge frames on the
positions, so they are exact derivatives of the energy. The reference code
detaches the frames. ``norm_type`` (``layer_norm_sh`` / ``layer_norm`` /
``rms_norm_sh``), ``activation`` (``sep_s2`` / ``s2`` / ``gate``),
``use_grid_mlp`` and ``grid_resolution`` select the variants of the paper's
ablations. The reference's truncation to the nearest ``max_neighbors`` atoms is
not supported, because it makes the energy discontinuous. Use a smaller cutoff
instead.

BAMBOO
======
:class:`~xnn.hybnn.models.bamboo.BAMBOO` (`bytedance/bamboo
<https://github.com/bytedance/bamboo>`_) is the graph equivariant transformer
of `Gong et al., Nat. Mach. Intell. 7, 543 (2025)
<https://doi.org/10.1038/s42256-025-01009-7>`_ with a charge-equilibrium
electrostatic term. It works in kcal/mol and Angstrom and also returns
``charges``, ``dipole``, ``energy_nn`` and ``energy_elec``.

.. code-block:: yaml

   model:
     name: bamboo
     cutoff: 5.0
     n_features: 64
     n_interactions: 3
     num_heads: 16
     use_electrostatics: true
     use_dispersion: false       # Optional D3(CSO) term

SpookyNet
=========
:class:`~xnn.hybnn.models.spookynet.SpookyNet` (`OUnke/SpookyNet
<https://github.com/OUnke/SpookyNet>`_) is the message passing network of
`Unke et al., Nat. Commun. 12, 7273 (2021)
<https://doi.org/10.1038/s41467-021-27504-0>`_ with electronic degrees of
freedom and nonlocal effects. The total charge and spin enter as embeddings, and
self-attention over all atoms carries nonlocal effects. The saved reference
models can be loaded with ``SpookyNet.from_reference_checkpoint(path)``.

.. code-block:: yaml

   model:
     name: spookynet
     cutoff: 5.29                # 10 bohr
     n_features: 128
     n_interactions: 6
     n_rbf: 16
     lr_cutoff: 10.0             # Long-range cutoff, needed for periodic cells
     electrostatics: false       # Enable when you have dipole or charge labels
     d4_dispersion: false

Other options, including ``nonlocal_interactions``, ``attention`` (favor /
exact), ``zbl_repulsion`` and ``ewald``, are also available. The outputs also
include ``charges`` and ``dipole``.

ReaxFF / ReaxFF-nn
==================
:class:`~xnn.ffnn.models.reaxff.ReaxFF` is the bond-order reactive force field
of `van Duin et al., J. Phys. Chem. A 105, 9396 (2001)
<https://doi.org/10.1021/jp004368u>`_ and its neural
variant. Bond orders are created from distances. Therefore, bonds break and form
smoothly, and any parameter group can be refit by gradient descent.

.. code-block:: python

   from xnn.ffnn.models import ReaxFF
   from xnn.ffnn.models.ffield import to_forcefield

   model = ReaxFF("CHO_cho_2008", trainable=("Desi", "ang_val1"))
   to_forcefield(model.export_library()).write("refit.frc")

The parameters come from a shipped ``.frc`` field by name, a ``.frc`` path or a
ReaxFF-nn JSON library (see :ref:`howto-forcefield-files`). For classical
fields, the default ``conventions: lammps`` reproduces LAMMPS ``pair_style
reaxff``. So, a field refit in xnn runs unchanged in LAMMPS.

OPLS / L-OPLS
=============
:class:`~xnn.ffnn.models.opls.OPLS` is the fixed-topology force field of
`Jorgensen et al., J. Am. Chem. Soc. 118, 11225 (1996)
<https://doi.org/10.1021/ja9621760>`_, matching OpenMM.
``from_atoms`` assigns the atom types from the SMARTS templates of the
parameter file, so no type names are needed.

.. code-block:: python

   from xnn.ffnn.models import OPLS

   model = OPLS.from_atoms(atoms, "oplsaa", cutoff=10.0)    # Or "lopls", "oplsaa-1996", "CL&P"
   model = OPLS.from_atoms(atoms, "oplsaa", trainable=("dihedral_v",))

Other options such as ``switch_width`` and ``fudge_lj`` / ``fudge_qq`` can also
be set.

A refit keeps the molecule's total charge: the charges the model uses are the
trainable per-type values projected onto that constraint (``constrain_charge``,
on by default). Labels on another energy zero than the force field's, such as
formation energies, are taken up by per-element reference energies: set
``atomic_energies: estimated`` in the config (or call
:func:`~xnn.common.finetune.reference.estimate_atomic_energies`) and add
``atom_ref`` to ``trainable`` to let them move with the fit. A pull of the
parameters toward their starting values, ``optim.prior_weight``, keeps a fit
of many parameters against few data well posed (:ref:`training`).

DREIDING / DREIDING-X6
======================
:class:`~xnn.ffnn.models.dreiding.Dreiding` is the generic force field of
`Mayo et al., J. Phys. Chem. 94, 8897 (1990)
<https://doi.org/10.1021/j100389a010>`_, matching the LAMMPS
DREIDING styles. Its parameters are generated from per-atom rules, so it covers
element combinations nobody tabulated, and those rules are what you train.

.. code-block:: python

   from xnn.ffnn.models import Dreiding

   model = Dreiding.from_atoms(atoms, "dreiding", cutoff=10.0, charges="gasteiger")
   model = Dreiding.from_atoms(atoms, "dreiding/X6")      # Exponential-6 van der Waals

Other options, including ``bond_style`` (harmonic / morse), ``angle_style`` and
``hbond``, are also available. Set resonance types, such as an amide
``C_R``-``N_R``, explicitly.

AMBER (ff94)
============
:class:`~xnn.ffnn.models.amber.Amber` is the force field of
`Cornell et al., J. Am. Chem. Soc. 117, 5179 (1995)
<https://doi.org/10.1021/ja00124a002>`_, matching LAMMPS and the AMBER
programs. Its charges belong to residue templates rather than to atom types:
``from_atoms`` matches the residues of the parameter file (amino acids with
their terminal forms, nucleotides, capping groups, water, ions) to assign
types and RESP charges, places the impropers by the AMBER builder's rules,
and types other organic molecules from SMARTS templates (give those their
charges with ``charges=``).

.. code-block:: python

   from xnn.ffnn.models import Amber

   model = Amber.from_atoms(atoms, "ff94", cutoff=10.0)          # a peptide, nucleic acid, water, ions
   model = Amber.from_atoms(atoms, "ff94", charges=resp_charges)  # a small molecule with its own charges
   model = Amber.from_atoms(atoms, "ff94", trainable=("dihedral_k", "charge"))

The options ``cutoff``, ``switch_width``, ``fudge_lj`` and ``fudge_qq`` work
as for OPLS. A config may give the AMBER divisors ``scnb`` and ``scee``
instead of the fudge factors. The trainable groups are the barrier heights
``dihedral_k`` and ``improper_k``, the bond and angle constants, the van der
Waals ``rstar`` and ``epsilon``, the per-atom ``charge`` and ``atom_ref``.
``constrain_charge`` holds the charges to the system's total charge. The
periodicities and phases of the Fourier terms stay fixed. Later AMBER sets
with the same functional form are further ``.frc`` libraries for this model.

CHARMM22
========
:class:`~xnn.ffnn.models.charmm.Charmm` is the all-atom protein force field of
`MacKerell et al., J. Phys. Chem. B 102, 3586 (1998)
<https://doi.org/10.1021/jp973084f>`_, matching LAMMPS and NAMD. On top
of the AMBER terms it has Urey-Bradley 1,3 terms, harmonic impropers,
Lennard-Jones parameters for 1,4 pairs and pair-specific overrides (NBFIX).
The 1,4 electrostatics are not scaled. ``from_atoms`` assigns types, charges
and impropers from the residue templates of the parameter file: amino acids
with their terminal, capped and patched forms, TIP3P water and ions. Small
model compounds are typed from SMARTS templates.

.. code-block:: python

   from xnn.ffnn.models import Charmm

   model = Charmm.from_atoms(atoms, "charmm22", cutoff=12.0)   # a peptide, water, ions
   model = Charmm.from_atoms(atoms, "charmm22", trainable=("dihedral_k", "ub_k", "charge"))

The options are ``cutoff``, ``switch_width``, ``fudge_qq`` (CHARMM's
``e14fac``) and ``coulomb_constant``. The constant defaults to 332.0716 of the
CHARMM program. NAMD uses 332.0636 and LAMMPS 332.06371. The trainable groups
are the force constants (``bond_k``, ``angle_k``, ``ub_k``, ``dihedral_k``,
``improper_k``), the equilibrium values (``bond_r0``, ``angle_theta0``,
``ub_s0``), the Lennard-Jones ``rstar`` and ``epsilon`` with their 1,4 and
pair-override counterparts, the per-atom ``charge`` and ``atom_ref``.
Periodicities, phases and improper reference angles stay fixed. The CMAP
correction of later CHARMM sets is not implemented, so ``charmm22`` is the
1998 force field.

.. note::

   **1,4 interactions in small rings.** CHARMM ties its 1,4 pairs to the
   dihedrals: every dihedral i-j-k-l implies a Lennard-Jones and Coulomb pair
   between i and l. In a six-membered ring two dihedrals reach the same para
   pair. In a five-membered ring a "1,4" pair is 1,3 the other way round, and
   in a four-membered ring it is bonded. The CHARMM rule is that the shortest
   bond path wins and that a pair counts once. xnn builds its 1,4 list that
   way from the topology, as CHARMM, NAMD, GROMACS and OpenMM do.

   LAMMPS is the exception. Its ``dihedral_style charmm`` evaluates the 1,4
   pair once per dihedral with a weight from the data file, so a weight of 1
   everywhere double counts the para pairs of six-membered rings and adds
   1,2 and 1,3 pairs across smaller rings. When comparing with or exporting
   to LAMMPS, use 1/2 in six-membered rings, 0 in four- and five-membered
   rings, and 0 on all but one of the terms that share a quadruple. The
   fidelity notebook derives those weights from xnn's pair list.

AMOEBA and HIPPO
================
:class:`~xnn.ffnn.models.amoeba.Amoeba` is the polarizable multipole force
field of `Ren and Ponder, J. Phys. Chem. B 107, 5933 (2003)
<https://doi.org/10.1021/jp027815+>`_ and `Ponder et al., J. Phys. Chem. B
114, 2549 (2010) <https://doi.org/10.1021/jp910674d>`_, with the protein set
of `Shi et al., J. Chem. Theory Comput. 9, 4046 (2013)
<https://doi.org/10.1021/ct4003702>`_. :class:`~xnn.ffnn.models.amoeba.Hippo`
is the HIPPO force field of `Rackers et al., J. Chem. Theory Comput. 17, 7056
(2021) <https://doi.org/10.1021/acs.jctc.1c00628>`_. Both match Tinker term
by term. Forces, torques and the stress come from autograd.

.. code-block:: python

   from xnn.ffnn.models import Amoeba, Hippo

   water = Amoeba.from_atoms(atoms, "amoeba-water-2003", cutoff=30.0)   # a water cluster
   hippo = Hippo.from_atoms(atoms, "hippo-water-2021", cutoff=30.0)
   peptide = Amoeba.from_atoms(atoms, "amoebapro13", cutoff=12.0)       # a peptide, water, ions
   model, structure = Amoeba.from_tinker_xyz("ubiquitin.xyz", "amoebapro13")
   model = Amoeba.from_atoms(atoms, "amoebapro13", trainable=("torsion", "vdw", "polarizability"))

The AMOEBA energy has the valence terms of the papers (anharmonic bonds and
angles, stretch-bend, Urey-Bradley, out-of-plane bends, torsions, pi-torsions
and torsion-torsion maps), the buffered 14-7 van der Waals term, permanent
multipoles in local frames and Thole-damped mutual induction. HIPPO keeps the
valence terms and multipoles and replaces the intermolecular terms by charge
penetration, density-damped induction, Pauli repulsion, damped dispersion,
charge transfer and charge flux. A library with ``exchpol`` parameters adds the
exchange polarization of `Chung et al., J. Phys. Chem. B 126, 7579 (2022)
<https://doi.org/10.1021/acs.jpcb.2c04237>`_: the overlap with each neighbor
shrinks an atom's polarizability along the pair. Similar to Tinker, a pair takes
part when either of its classes is used, pairs within a molecule carry the
polarization scale factors of the permanent field (not those of the polarization
groups), and in a cell the term is tapered by the switch of the repulsion.

Molecules
---------
For a molecule or cluster, every pair within ``cutoff`` interacts in full.
Choose a cutoff that encloses the whole system. ``polarization`` selects
mutual, direct or no induction. The default is the library's setting.
A batch of molecules of up to a thousand atoms each is solved directly, as
one block-diagonal system. Larger molecules and cells use conjugate
gradients, and ``solver`` overrides the choice. ``preconditioner`` selects
``"diagonal"`` (the default), ``"block"``, ``"neumann"`` or
``"block-neumann"``. The others take fewer iterations but rarely give a
faster step. In molecular dynamics,
``predictor="aspc"`` starts each solve from dipoles extrapolated from the
last steps, which saves a fifth to a third of the iterations. The MDI engine turns it
on by default. Every backward solve starts from the previous one.
``use_fast=True`` compiles the pair terms into fused GPU kernels. It
applies to evaluation on a GPU, such as molecular dynamics, while
training keeps the reference path. The first evaluations pay for the
compilation. The MDI engine sets ``use_fast="auto"``.

For molecular dynamics, ``plan_step(data, skin, capacity)`` records the
decisions of one evaluation, and ``model(data, plan)`` reuses them while
the atoms move. The plan keeps a Verlet pair list built with
``cutoff + skin``, and ``plan.outdated(pos)`` tells when to rebuild it. The
MDI engine runs the steps on a plan on a GPU and replays each step as one
CUDA graph (``--step-graph``).

Periodic cells
--------------
A structure with a cell is evaluated with Ewald sums: the multipoles, the
induction and HIPPO's dispersion split into a real-space part within
``ewald_cutoff`` and a particle-mesh part. The van der Waals, repulsion and
charge-transfer terms are switched off smoothly before their cutoff.

.. code-block:: python

   box = Amoeba.from_atoms(water_box, "amoeba-water-2003", cutoff=7.3, vdw_cutoff=7.0, taper=6.3)
   out = ForceStressOutput(box, compute_stress=True)(graph)

The defaults follow Tinker. The Ewald coefficient comes from the real-space
cutoff (7 Angstrom at most) and the mesh has 1.2 points per Angstrom. As in
Tinker, every mesh size, a requested one too, is rounded up to an even size
with prime factors 2, 3 and 5, at least 16 (``pme_grid=27`` gives 30). The
mesh sums use the least-squares spline moduli, as Tinker does, so the
energies match Tinker's at the same keys even on a coarse mesh. The
options are ``ewald_alpha``, ``pewald_alpha`` and ``dewald_alpha`` for the
three coefficients, ``pme_grid`` and ``pme_order`` for the mesh,
``vdw_cutoff``, ``repulsion_cutoff``, ``chgtrn_cutoff`` and
``dispersion_cutoff`` for the switched terms, and ``taper`` for the start of
the switch. Keep the neighbor cutoff about 0.3 Angstrom above the van der
Waals cutoff, because the hydrogen sites of that term sit inside their atoms.
``polar_eps`` sets the convergence of the induced dipoles in Debye.
``mesh_self=True`` removes the spurious mesh self-interaction of every
multipole (its position average) from the energies, the polarizing field
and the dispersion sum. It brings the dispersion and the permanent
multipoles of a coarse mesh closer to the converged values, at the price of
no longer matching Tinker at the same keys.
``ewald=False`` truncates a cell at the cutoff instead. A slab with two
periodic axes is summed on its cell with the Yeh-Berkowitz correction, so
leave a vacuum gap of at least the slab thickness along the free axis.

Training and files
------------------
``trainable`` names the parameter groups to refit: the valence tables
(``bond``, ``angle``, ``strbnd``, ``urey``, ``opbend``, ``torsion``,
``pitors``, ``tortor``), ``vdw``, the multipoles (``charge``, ``dipole``,
``quadrupole``), ``polarizability``, ``thole``, the HIPPO groups
(``chgpen``, ``chgtrn``, ``repulsion``, ``dispersion``, ``exchpol``,
``cflux``) and
``atom_ref``. While the multipoles train, every molecule keeps its total
charge (``conserve_charge``), components the library leaves zero by the
symmetry of the local frame stay zero (``symmetry_mask``), and the
quadrupoles stay symmetric and traceless; ``export_library`` writes the
multipoles the model uses. ``forward`` reports every energy term, the
monopoles, the induced dipoles and the molecular dipole. Libraries are read
from ``.frc`` and Tinker ``.prm`` files and written back to either. A Tinker
``.xyz`` file builds the topology directly.

Semiempirical methods: PM3, PM6, PM6-ORG, PM7
=============================================
:class:`~xnn.sqmnn.models.PM6` and its siblings are the NDDO semiempirical
methods of Stewart (PM3: `Stewart, J. Comput. Chem. 10, 209 (1989)
<https://doi.org/10.1002/jcc.540100208>`_; PM6:
`Stewart, J. Mol. Model. 13, 1173 (2007)
<https://doi.org/10.1007/s00894-007-0233-4>`_; PM7: `Stewart, J. Mol. Model. 19,
1 (2013) <https://doi.org/10.1007/s00894-012-1667-x>`_; PM6-ORG:
`Stewart and Stewart, J. Mol. Model. 29, 284 (2023)
<https://doi.org/10.1007/s00894-023-05695-1>`_), matching MOPAC. With the
published parameters (the default) they reproduce MOPAC, and
``trainable`` makes parameter groups refittable with exact gradients through
the self-consistent field.

.. code-block:: python

   from xnn.sqmnn.models import PM6, PM7

   model = PM7()                                      # Published parameters
   model = PM6(trainable=["BETAS", "BETAP", "ALPB"])  # Refit by gradient descent

``energy`` is the heat of formation in eV and the outputs also include
``heat_of_formation`` (kcal/mol), ``charges`` and ``dipole``. The methods handle
molecules only. The published parameters are downloaded from the MOPAC sources
on first use.

Add-ons for any model
=====================
These wrappers add additional physical terms or corrections to any model
mentioned above. You can enable them in the ``model`` section of a config file.
They can combine freely with other models and every deployment channel which can
carry them.

Long-range electrostatics (LES)
-------------------------------
:class:`~xnn.common.models.les.LatentEwald` implements the latent Ewald
summation of `Cheng, npj Comput. Mater. 11, 80 (2025)
<https://doi.org/10.1038/s41524-025-01577-7>`_. A small
network maps each atom's features to latent charges, and an Ewald sum over them
adds the long-range energy.

.. code-block:: yaml

   model:
     name: mace                  # Any model without charges of its own
     long_range: {n_channels: 4, sigma: 1.0, dl: 2.0}
     # long_range: {n_channels: 4, constrain_charge: true}   # Charges sum to the net charge
     # long_range: {n_channels: 4, charge_solve: true}       # Global charge equilibration
     # long_range: {n_channels: 4, method: pme}              # Particle-mesh sum for large cells
     # long_range: {n_channels: 1, constrain_charge: true, dipole_origin: com}  # Fit to dipoles
     # long_range: {n_channels: 1, charge_solve: true, dielectric: 78.4}       # Screened by a solvent

The outputs gain ``energy_lr``, ``latent_charges``, ``les_charges`` (in e) and
``les_dipole``. The main options:

- ``constrain_charge`` makes one charge channel sum to the structure's net
  charge.
- ``charge_solve`` equilibrates the charges over the whole structure, and
  ``fragments: true`` keeps each molecule or ion at its own charge.
- ``les_charge_weight`` and ``les_dipole_weight`` in ``optim`` fit the charges
  to ``charges`` and ``dipole`` labels, which sets their scale.
- ``dielectric`` screens the long-range energy by a solvent's dielectric
  constant, and ``learn_dielectric: true`` fits it.

Molecules use the exact real-space sum. Models that predict their own charges
(AIMNet2, PhysNet, SpookyNet, BAMBOO, HDNNP 3G/4G) use their own electrostatics
instead.

Dispersion (D4, D3)
-------------------
:class:`~xnn.common.models.d4.D4Dispersion` and
:class:`~xnn.common.models.d3.D3Dispersion` add the D4 (`Caldeweyher et al., J.
Chem. Phys. 150, 154122 (2019) <https://doi.org/10.1063/1.5090222>`_) and D3
(`Grimme et al., J. Chem. Phys. 132, 154104 (2010)
<https://doi.org/10.1063/1.3382344>`_; BJ damping:
`Grimme et al., J. Comput. Chem. 32, 1456 (2011)
<https://doi.org/10.1002/jcc.21759>`_) dispersion corrections, matching
``dftd4`` and ``simple-dftd3``. Both also work as standalone models (``d4`` and
``d3``).

.. code-block:: yaml

   model:
     name: mace
     dispersion: true                                     # PBE0-D4 defaults
     # dispersion: {s6: 1.0, s8: 1.2, a1: 0.4, a2: 5.0}   # D4 with explicit damping
     # dispersion: {name: d3, damping: bj, s9: 1.0}       # D3(BJ) with the three-body term

For condensed phases, shorten the cutoffs and add switching windows so the MD
run can stay smooth:

.. code-block:: yaml

   dispersion: {cutoff_pair: 12.0, switch_width_pair: 2.0,
                cutoff_triple: 10.0, switch_width_triple: 1.0,
                tail_correction: true}                    # D4: add back the cut pair tail

The net charge comes from the structure. A model trained on labels with the
dispersion removed records it in ``subtracted_dispersion`` (see
:ref:`deployment`).
