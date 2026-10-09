.. _models:

******
Models
******

Every model takes an :class:`~xnn.common.data.atomic_data.AtomicGraph` and
returns ``node_energy`` and ``energy``;
:class:`~xnn.common.models.outputs.ForceStressOutput` adds forces and stress
by autograd. Models are built by name from a config, or constructed
directly with the same keys:

.. code-block:: python

   from xnn.common.models import available_models, build_model, ForceStressOutput

   available_models()          # ['aimnet2', 'allegro', 'ani', 'bamboo', 'cace', 'cnn3d', 'dimenet', 'dimenet++', 'hdnnp', 'mace', 'nequip', 'painn', 'physnet', 'reaxff', 'schnet', 'se3cnn', ...]
   model = build_model(cfg.model)   # dispatches to <Model>.from_config(cfg.model)

   model = ForceStressOutput(build_model(cfg.model), compute_stress=True)
   out = model(graph)      # energy, node_energy, forces, stress

Pre-trained models load with ``from_pretrained()``
(:ref:`howto-pretrained-models`). NequIP, MACE, Allegro and the SE(3)
steerable CNN need the ``gnn`` extra (e3nn); every other model is plain
PyTorch. The snippets below show
the options most runs set; each class documents the full list.

MACE
====
:class:`~xnn.gnn.models.mace.MACE`: higher-order equivariant message passing
(Batatia *et al.*), matching `ACEsuit/mace <https://github.com/ACEsuit/mace>`_.
Every published MACE-MP / MACE-OFF checkpoint loads through the hub.

.. code-block:: yaml

   model:
     name: mace
     cutoff: 5.0
     species: [1, 6, 8]
     num_channels: 64
     num_interactions: 2         # any depth from 0
     max_ell: 3
     correlation: 3
     avg_num_neighbors: 20.0
     atomic_energies: {1: -13.6, 6: -1029.9, 8: -2042.8}
     # foundation: mace-off23-small   # start from a pretrained checkpoint instead

Also ``max_L``, ``hidden_irreps``, ``MLP_irreps``, ``radial_type``,
``distance_transform`` (Agnesi / Soft), ``pair_repulsion`` (ZBL),
``scale`` / ``shift``.

NequIP
======
:class:`~xnn.gnn.models.nequip.NequIP`: E(3)-equivariant message passing
with gated nonlinearities (Batzner *et al.*), matching `mir-group/nequip
<https://github.com/mir-group/nequip>`_; state dicts transplant directly.

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

Also ``invariant_layers`` / ``invariant_neurons``, ``resnet``, ``use_sc``,
``trainable_rbf``, ``atomic_scales``.

Allegro
=======
:class:`~xnn.gnn.models.allegro.Allegro`: strictly local equivariant
many-body potential without message passing (Musaelian *et al.*), matching
`mir-group/allegro <https://github.com/mir-group/allegro>`_ v0.3.0.

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

Also ``parity``, ``env_embed``, ``edge_eng``, ``latent_resnet``,
``atomic_scales``.

CACE
====
:class:`~xnn.gnn.models.cace.CACE`: the Cartesian atomic cluster expansion
(Cheng 2024), body-ordered invariants from Cartesian monomials with optional
message passing; no e3nn. Matches `BingqingCheng/cace
<https://github.com/BingqingCheng/cace>`_.

.. code-block:: yaml

   model:
     name: cace
     cutoff: 5.5
     species: [1, 6, 8]
     n_atom_basis: 3
     max_l: 3
     max_nu: 3
     num_message_passing: 1      # 0 = plain Cartesian ACE
     avg_num_neighbors: 10.0

Also ``message_types`` (M / Ar / Bchi), ``n_rbf``, ``readout_hidden``,
``atomic_energies``.

AIMNet2
=======
:class:`~xnn.gnn.models.aimnet2.AIMNet2`: the atoms-in-molecules network
(Anstine, Zubatyuk and Isayev 2025) for neutral, charged and open-shell
molecules; predicts partial charges exact in the net charge and adds their
Coulomb energy. Matches `isayevlab/aimnetcentral
<https://github.com/isayevlab/aimnetcentral>`_; the six published families
load through the hub. Returns ``charges`` and ``dipole`` as well.

.. code-block:: python

   from xnn.common.models import from_pretrained

   model = from_pretrained("aimnet2")                                   # wb97m-d3, with its D3 term
   model = from_pretrained("aimnet2", model_options={"coulomb": "pme"})  # periodic cells: dsf / ewald / pme

The net charge and spin multiplicity come from the structure
(``total_charge``, ``spin_multiplicity``). Training from scratch uses
``n_features``, ``n_rbf``, ``n_interactions``, ``hidden``, ``aim_size``,
``charge_channels`` (2 for open shells), ``coulomb``, ``lr_cutoff``.

SchNet
======
:class:`~xnn.gnn.models.schnet.SchNet`: continuous-filter convolutions
(Schütt *et al.* 2017), a clean-room build of the paper with the paper's
architecture as defaults.

.. code-block:: yaml

   model:
     name: schnet
     cutoff: 5.0
     cutoff_fn: cosine           # smooth finite cutoff; the paper trains cutoff-free
     n_features: 64
     n_interactions: 3
     n_rbf: 50
     species: [1, 6, 8]
     atomic_energies: {1: -13.6, 6: -1029.9, 8: -2042.8}

Also ``gamma``, ``energy_shift`` / ``energy_scale``.

DimeNet and DimeNet++
=====================
:class:`~xnn.gnn.models.dimenet.DimeNet` and
:class:`~xnn.gnn.models.dimenet.DimeNetPP`: directional message passing
(Gasteiger *et al.* 2020). Messages live on directed edges and are updated
from the distances and angles of neighboring edges through Bessel bases;
DimeNet++ swaps the bilinear interaction for a cheaper Hadamard form.
No e3nn. Matches `gasteigerjo/dimenet <https://github.com/gasteigerjo/dimenet>`_.

.. code-block:: yaml

   model:
     name: dimenet++             # or dimenet
     cutoff: 5.0
     n_features: 128
     n_interactions: 4           # 6 for dimenet
     n_rbf: 6
     n_spherical: 7
     species: [1, 6, 8]
     atomic_energies: {1: -13.6, 6: -1029.9, 8: -2042.8}

Also ``n_bilinear`` (dimenet), ``n_triplet_features`` /
``n_basis_features`` / ``n_output_features`` (dimenet++), ``p``,
``trainable_rbf``. The reference code's key spellings are translated.

PaiNN
=====
:class:`~xnn.gnn.models.painn.PaiNN`: equivariant message passing with
scalar and vector features (Schütt *et al.* 2021). Optional heads predict
dipole moments from latent charges and atomic dipoles and polarizability
tensors from a rank-1 decomposition; train them with ``dipole_weight`` /
``polarizability_weight`` on data with ``dipole`` / ``polarizability``
labels. No e3nn. Matches the reference implementation (``schnetpack``).

.. code-block:: yaml

   model:
     name: painn
     cutoff: 5.0
     n_features: 128
     n_interactions: 3
     n_rbf: 20                   # sin(n pi r / r_cut) / r with a cosine cutoff
     dipole: true                # outputs dipole and latent charges
     polarizability: true
     species: [1, 6, 8]
     atomic_energies: {1: -13.6, 6: -1029.9, 8: -2042.8}

Also ``radial_basis`` (``bessel`` / ``gaussian``), ``shared_filters``,
``atomic_dipoles``, ``correct_charges``, ``n_output_blocks``, the ablations
``scalar_product`` / ``vector_propagation`` / ``vector_features``, and
``energy_shift`` / ``energy_scale``.

SE(3) steerable CNN and 3D CNN
==============================
:class:`~xnn.cnn.models.steerable.SteerableCNN`: the SE(3)-equivariant
convolutional network of Weiler *et al.* (NeurIPS 2018) on voxel grids of
each atom's environment, a clean-room build of the paper. Fields of order
0, 1 and 2 pass through gated steerable convolutions; the pooled scalar
fields are read out to per-atom energies. :class:`~xnn.cnn.models.cnn3d.CNN3D`
is the conventional 3D CNN baseline with the same grids, readout and block
layout. Both use :class:`~xnn.cnn.featurizers.voxel.VoxelGrid`, one density
channel per species.

.. code-block:: yaml

   model:
     name: se3cnn                # or cnn3d
     cutoff: 4.0                 # half the side of the voxel cube
     species: [1, 6, 8]
     grid_size: 17               # voxels per axis
     n_features: 32              # scalar fields (channels) of the last block
     n_interactions: 3           # blocks
     l_max: 2                    # highest field order (se3cnn only)
     kernel_size: 5
     atomic_energies: {1: -13.6, 6: -1029.9, 8: -2042.8}

Also ``fields`` / ``channels`` (explicit block widths), ``strides``,
``activation`` ("ssp"), ``normalization`` ("batch"), ``bandlimit``,
``shell_width``, ``sigma``, ``cutoff_fn``, ``energy_shift`` /
``energy_scale``. The steerable model is exactly invariant under the
rotations of the grid onto itself and invariant to the bandlimit under all
others; mirror images are not constrained.

Spherical CNN
=============
:class:`~xnn.cnn.models.spherical.SphericalCNN` (``s2cnn``): the
rotation-equivariant network of Cohen *et al.* (ICLR 2018) over spherical
signals, a clean-room build of the paper. Each atom carries one potential
channel per species sampled on a sphere around it
(:class:`~xnn.cnn.featurizers.spherical.SphericalGrid`); ResNet blocks of
:math:`S^2` and :math:`SO(3)` correlations, computed by generalized FFTs,
map it to :math:`SO(3)` feature maps whose invariant integral is read out
to per-atom energies. No e3nn. Matches `jonas-koehler/s2cnn
<https://github.com/jonas-koehler/s2cnn>`_.

.. code-block:: yaml

   model:
     name: s2cnn
     cutoff: 10.0                # neighbor radius of the potential sums
     species: [1, 6, 7, 8, 16]
     radius: 0.48                # sphere radius (A)
     bandwidth: 10               # 2b x 2b samples per sphere
     n_features: 160             # channels of the last block
     n_interactions: 5           # ResNet blocks (Table 3 of the paper)
     atomic_energies: {1: -13.6, 6: -1029.9, 7: -1485.3, 8: -2042.8, 16: -10831.3}

Also ``features`` / ``bandwidths`` (explicit per block), ``exponent`` (1,
the paper's potential; 2 in the reference data script), ``set_readout``
(the paper's DeepSet readout, e.g. ``[150, 100, 50]``), ``normalization``
("batch"), ``activation`` ("relu"), ``n_alpha`` / ``n_beta`` / ``n_gamma``
/ ``max_beta`` (filter support), ``cutoff_fn``, ``energy_shift`` /
``energy_scale``. The energy is exactly invariant under rotations about the
polar axis that map the coarsest grid onto itself and invariant to the
discretization error under all others.

ANI
===
:class:`~xnn.dnn.models.ani.ANI`: per-element networks over the atomic
environment vector (Smith *et al.* 2017), matching `aiqm/torchani
<https://github.com/aiqm/torchani>`_. The published parameterisations are
presets, each with its training set in the data hub:

.. code-block:: python

   from xnn.dnn.models import ANI

   ANI.ani1(species=[1, 6, 7, 8])      # ANI-1
   ANI.ani1x(species=[1, 6, 7, 8])     # ANI-1x
   ANI.ani1ccx(species=[1, 6, 7, 8])   # ANI-1ccx, coupled-cluster self energies
   ANI.ani2x()                         # ANI-2x, seven elements (H, C, N, O, S, F, Cl)

.. code-block:: yaml

   model:
     name: ani
     preset: ani-2x              # ani-1 / ani-1x / ani-1ccx / ani-2x

Without a preset: ``species``, ``radial_cutoff``, ``angular_cutoff``,
``hidden``, ``activation``, ``atomic_energies``.

PhysNet
=======
:class:`~xnn.dnn.models.physnet.PhysNet`: message passing with explicit
electrostatics and D3 dispersion (Unke and Meuwly 2019), a pure-PyTorch
translation of `MMunibas/PhysNet <https://github.com/MMunibas/PhysNet>`_.
Embeds elements directly (no species list); returns ``charges`` and
``dipole``.

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
:class:`~xnn.dnn.models.hdnnp.HDNNP`: the Behler-Parrinello potential,
radial symmetry functions and one MLP per element. Under development.
Options: ``species``, ``cutoff``, ``etas``, ``rs``, ``hidden``.

BAMBOO
======
:class:`~xnn.hybrid.models.bamboo.BAMBOO`: a graph equivariant transformer
with a charge-equilibrium electrostatic term (Gong *et al.* 2024), matching
`bytedance/bamboo <https://github.com/bytedance/bamboo>`_. Works in
kcal/mol and Å; returns ``charges``, ``dipole``, ``energy_nn`` and
``energy_elec``.

.. code-block:: yaml

   model:
     name: bamboo
     cutoff: 5.0
     n_features: 64
     n_interactions: 3
     num_heads: 16
     use_electrostatics: true
     use_dispersion: false       # optional D3(CSO)

ReaxFF / ReaxFF-nn
==================
:class:`~xnn.ffnn.models.reaxff.ReaxFF`: the bond-order reactive force
field (van Duin *et al.* 2001) and its neural variant. Bond orders come from
distances, so bonds break and form smoothly, and charges are equilibrated
at every geometry. Any parameter group can be refit by gradient descent.
Returns ``charges`` and the per-term energies.

.. code-block:: python

   from xnn.ffnn.models import ReaxFF

   model = ReaxFF("CHO_cho_2008", cutoff=10.0, trainable=("bond", "angle"))
   model.export_library().save_frc("refit.frc")

The parameters are a shipped ``.frc`` field by name, a ``.frc`` path or a
ReaxFF-nn JSON library (:ref:`howto-forcefield-files`).

OPLS / L-OPLS
=============
:class:`~xnn.ffnn.models.opls.OPLS`: the fixed-topology force field of
Jorgensen *et al.* (1996), matching OpenMM. ``from_atoms`` perceives the
bonds with RDKit and assigns the atom types from the SMARTS templates of the
parameter file, so no type names are needed. Any parameter group is
trainable, and several molecules can share one force field.

.. code-block:: python

   from xnn.ffnn.models import OPLS

   model = OPLS.from_atoms(atoms, "oplsaa", cutoff=10.0)   # or "lopls", "oplsaa-1996", "CL&P"
   model = OPLS.from_atoms(atoms, "oplsaa", trainable=("dihedral_v",))

Also ``switch_width``, ``fudge_lj`` / ``fudge_qq``.

DREIDING / DREIDING-X6
======================
:class:`~xnn.ffnn.models.dreiding.Dreiding`: the rule-generated generic
force field of Mayo *et al.* (1990), matching the LAMMPS DREIDING styles.
Bond, angle and torsion parameters are generated from per-atom generators,
so it covers element combinations nobody tabulated; the generators are what
is trainable. ``from_atoms`` perceives bonds and bond orders.

.. code-block:: python

   from xnn.ffnn.models import Dreiding

   model = Dreiding.from_atoms(atoms, "dreiding", cutoff=10.0, charges="gasteiger")
   model = Dreiding.from_atoms(atoms, "dreiding/X6")      # exponential-6 van der Waals

Also ``bond_style`` (harmonic / morse), ``angle_style``, ``hbond``,
``switch_width``. Set resonance types such as an amide ``C_R``-``N_R``
explicitly; automatic perception does not.

Add-ons for any model
=====================
Wrappers that add physics to any model above, enabled from the ``model``
section or by wrapping a model object. They combine freely and every deploy
channel carries them.

Long-range electrostatics (LES)
-------------------------------
:class:`~xnn.common.models.les.LatentEwald` (Cheng 2025): a small MLP maps
each atom's features to latent charges and an Ewald sum over them adds the
long-range energy.

.. code-block:: yaml

   model:
     name: mace                  # any model
     long_range: {n_channels: 4, sigma: 1.0, dl: 2.0}
     # long_range: {n_channels: 4, constrain_charge: true}   # charges sum to total_charge
     # long_range: {n_channels: 4, charge_solve: true}       # global charge equilibration

Molecules use the exact real-space sum (``dl`` then has no effect).
Outputs gain ``energy_sr``, ``energy_lr`` and ``latent_charges``.

Two options tie the charges to the structure's ``total_charge`` (its
``charge`` label; missing means neutral). ``constrain_charge`` shifts each
structure's charges so one channel sums to the net charge and the others to
zero (``charge_weights: learned`` lets the model choose where the shift
goes). ``charge_solve`` goes further: that channel's head output becomes an
electronegativity, a learned hardness per element (``hardness: features``
for a per-atom one) is added, and the charges minimise
``chi.q + 1/2 J q^2 + E_lr`` under the same constraint, so every charge
responds to every other one through the Ewald kernel; the energy gains
``energy_charge``. Charged structures must be molecules (a cell's ``k = 0``
term is dropped). With ``fragments: true`` each molecule or ion is
constrained on its own (plain charge equilibration lets charge flow between
distant fragments): fragments are the covalently bonded groups, ions in
``ion_charges`` (alkali, alkaline-earth and halide ions by default) always
stand alone, and a fragment's charge comes from the per-atom
``fragment_charges`` label of the structure file when present, else from
that table.

Dispersion (DFT-D4, DFT-D3)
---------------------------
:class:`~xnn.common.models.d4.D4Dispersion` and
:class:`~xnn.common.models.d3.D3Dispersion` reproduce ``dftd4`` and
``simple-dftd3`` to floating-point precision; they also stand alone as the
models ``d4`` and ``d3``.

.. code-block:: yaml

   model:
     name: mace
     dispersion: true                                    # PBE0-D4 defaults
     # dispersion: {s6: 1.0, s8: 1.2, a1: 0.4, a2: 5.0}   # D4, explicit damping
     # dispersion: {name: d3, damping: bj, s9: 1.0}       # D3(BJ) with the three-body term

For condensed-phase work shorten the cutoffs (the defaults reproduce the
reference codes) and add switching windows so MD stays smooth:

.. code-block:: yaml

   dispersion: {cutoff_pair: 12.0, switch_width_pair: 2.0,
                cutoff_triple: 10.0, switch_width_triple: 1.0,
                tail_correction: true}                  # D4: add back the cut pair tail

The net charge comes from ``total_charge``. D4 picks its charge solver by
size (``regime: dense | large | auto``); in MD, ``eeq_reuse`` carries the
solve between steps (:ref:`deployment`). Outputs gain ``energy_disp`` and,
for D4, ``eeq_charges``. A model trained on labels with the dispersion
removed records it in ``subtracted_dispersion`` (:ref:`deployment`).
