.. _models:

******
Models
******

All models subclass :class:`~xnn.common.models.base.InteratomicPotential`:
they take an :class:`~xnn.common.data.atomic_data.AtomicGraph` and return
``node_energy`` (per atom) and ``energy`` (per structure). Forces and stress
are added by :class:`~xnn.common.models.outputs.ForceStressOutput`; no model
implements them itself.

.. code-block:: python

   from xnn.common.models import available_models, build_model, ForceStressOutput

   available_models()          # ['aimnet2', 'allegro', 'ani', 'bamboo', 'cace', 'cnn3d', 'hdnnp', 'mace', 'nequip', 'physnet', 'reaxff', 'schnet', 'se3cnn', ...]
   model = build_model(cfg.model)   # dispatches to <Model>.from_config(cfg.model)

   model = ForceStressOutput(build_model(cfg.model), compute_stress=True)
   out = model(graph)      # energy, node_energy, forces (N, 3), stress (B, 3, 3)

Every model can also be constructed directly; its constructor arguments are
the keys accepted under ``model`` in a config file. Pre-trained models load
with ``from_pretrained()`` (:ref:`howto-pretrained-models`). NequIP, MACE
and Allegro need the ``gnn`` extra (e3nn); every other model, SchNet
included, is plain PyTorch.

MACE (``gnn``)
==============
:class:`~xnn.gnn.models.mace.MACE`: higher-order equivariant message
passing with the learned symmetric contraction (Batatia *et al.*).
Reproduces `ACEsuit/mace <https://github.com/ACEsuit/mace>`_, and loads
every published MACE-MP / MACE-OFF checkpoint (MP-0, 0b/0b2/0b3, MPA-0,
OMAT-0, MATPES, the multi-head MH-0, OFF23) through the hub:
``from_pretrained("mace-mp-0-medium")`` converts the file once, verified to
float64 round-off. In a config, ``foundation: mace-off23-small`` builds the
pretrained model.

Options: ``species``, ``cutoff`` (4.0), ``max_ell`` (3), ``max_L`` (0),
``num_channels`` (32), ``n_rbf`` (8), ``num_interactions`` (2; any T from 0
up), ``correlation`` (3), ``MLP_irreps`` ("16x0e"), ``radial_MLP``,
``interaction`` ("RealAgnosticResidualInteractionBlock", or the
density-normalized blocks), ``gate`` ("silu"), ``avg_num_neighbors`` (1.0),
``hidden_irreps``, ``num_cutoff_basis`` (5), ``radial_type`` ("bessel"),
``distance_transform`` ("None" | "Agnesi" | "Soft"), ``pair_repulsion``
(False, ZBL), ``atomic_energies``, ``scale`` / ``shift`` (1, 0).

NequIP (``gnn``)
================
:class:`~xnn.gnn.models.nequip.NequIP`: E(3)-equivariant message passing
with gated nonlinearities (Batzner *et al.*). Reproduces `mir-group/nequip
<https://github.com/mir-group/nequip>`_ with directly transplantable state
dicts.

Options: ``species``, ``cutoff`` (4.0), ``l_max`` (2), ``parity`` (True),
``n_rbf`` (8), ``n_layers`` (3), ``num_features`` (32), ``invariant_layers``
(2), ``invariant_neurons`` (64), ``avg_num_neighbors``, ``use_sc`` (True),
``resnet`` (False), ``nonlinearity_scalars`` / ``nonlinearity_gates``,
``num_polynomial_cutoff`` (6), ``trainable_rbf`` (True),
``atomic_energies``, ``atomic_scales``.

Allegro (``gnn``)
=================
:class:`~xnn.gnn.models.allegro.Allegro`: strictly local equivariant
many-body potential without message passing (Musaelian *et al.*).
Reproduces `mir-group/allegro <https://github.com/mir-group/allegro>`_
v0.3.0 with directly transplantable state dicts.

Options: ``species``, ``cutoff`` (4.0), ``l_max`` (1), ``parity``
("o3_full"), ``n_rbf``, ``num_layers``, ``num_tensor_features``,
``two_body_latent`` / ``latent`` / ``env_embed`` / ``edge_eng`` (MLP widths),
``initial_scalar_embedding_dim``, ``avg_num_neighbors``, ``latent_resnet``
(True), ``num_polynomial_cutoff`` (6), ``trainable_rbf`` (True),
``atomic_energies``, ``atomic_scales``.

CACE (``gnn``)
==============
:class:`~xnn.gnn.models.cace.CACE`: the Cartesian atomic cluster expansion
(Cheng 2024), body-ordered invariants built from Cartesian monomials with
optional message passing; no spherical harmonics and no e3nn. Reproduces
`BingqingCheng/cace <https://github.com/BingqingCheng/cace>`_.

Options: ``species``, ``cutoff`` (5.5), ``n_atom_basis`` (3), ``n_rbf`` (8),
``n_radial_basis``, ``max_l`` (3), ``max_nu`` (3), ``num_message_passing``
(1; 0 is plain Cartesian ACE), ``message_types`` (["M", "Ar", "Bchi"]),
``embed_receiver_nodes`` (False), ``avg_num_neighbors`` (10.0),
``num_polynomial_cutoff`` (6), ``trainable_rbf`` (True), ``readout_hidden``
([32, 16]), ``atomic_energies``.

AIMNet2 (``gnn``)
=================
:class:`~xnn.gnn.models.aimnet2.AIMNet2`: the atoms-in-molecules network of
Anstine, Zubatyuk and Isayev (2025) for neutral, charged and open-shell
molecules. Per-shell element embeddings, message passes that predict partial
charges made exact in the net charge by neural charge equilibration, and
the Coulomb energy of those charges, all pairs for molecules or a
truncated, Ewald or particle-mesh sum for periodic cells. Reproduces
`isayevlab/aimnetcentral <https://github.com/isayevlab/aimnetcentral>`_,
and loads the six published families (``aimnet2-wb97m-d3``,
``aimnet2-b973c-d3``, ``aimnet2-b973c-2025-d3``, ``aimnet2-nse``,
``aimnet2-pd``, ``aimnet2-rxn``, four ensemble members each):
``from_pretrained("aimnet2")`` serves the first ``wb97m-d3`` member with
the D3 term its reference calculator adds. The net charge and spin
multiplicity come from the graph (``total_charge``, ``spin_multiplicity``).
Returns ``charges``, ``dipole`` and ``energy_coulomb`` in addition.

SchNet (``gnn``)
================
:class:`xnn.gnn.models.schnet.SchNet`: continuous-filter convolutions over
a Gaussian radial basis with shifted-softplus activations (Schütt *et al.*,
NIPS 2017). Faithful to the manuscript (see :ref:`fidelity`): the defaults
are the paper architecture — ``F = 64`` feature maps, ``T = 3`` residual
interaction blocks, RBF centers every 0.1 Å on ``[0, 30]`` with
``gamma = 10`` Å\ :sup:`-2` — plus the DTNN per-atom energy standardization
(``energy_shift``/``energy_scale``, or
:meth:`~xnn.gnn.models.schnet.SchNet.set_energy_scale_shift`).

Key options: ``n_features`` (64), ``n_interactions`` (3), ``n_rbf`` (301),
``cutoff`` (30.0), ``gamma`` (10.0), ``cutoff_fn`` (``None``; set
``"cosine"`` to smooth the filters at a finite cutoff for condensed phases),
``energy_shift`` (0.0), ``energy_scale`` (1.0), ``species`` +
``atomic_energies`` (per-element reference energies loaded into
``atom_ref``).

3D steerable CNN (``cnn``)
==========================
:class:`xnn.cnn.models.steerable.SteerableCNN` (registry name ``se3cnn``):
the SE(3)-equivariant 3D steerable CNN of Weiler *et al.* (NeurIPS 2018)
applied to interatomic potentials. Every atom's environment is voxelized
into one scalar density field per species
(:class:`~xnn.cnn.featurizers.VoxelGrid`); gated blocks of steerable
convolutions (kernels spanned by the analytic basis of Sec. 4.2 of the
paper, Gaussian shells with the radius-dependent bandlimits of Sec. 4.4.1)
map it through stacks of fields of order 0, 1, 2 to scalar fields, which a
global average pool and the atom-wise readout turn into the per-atom energy.
The energy is exactly invariant under the rotations of the grid onto
itself, invariant to the bandlimit under every other rotation, and the
forces co-rotate (see :ref:`fidelity`). Needs ``e3nn``.

Key options: ``species``, ``cutoff`` (4.0, the half side of the cube),
``grid_size`` (17 voxels per axis), ``n_features`` (32 scalar fields of the
last block), ``n_interactions`` (3 gated blocks), ``l_max`` (2), ``fields``
(explicit multiplicities per block, e.g. ``[[8, 4, 2], [16, 8, 4], [32]]``),
``kernel_size`` (5), ``strides`` (2 in every block but the first and last,
after a low-pass filter), ``bandlimit`` (``compromise`` / ``conservative``
/ ``sfcnn`` or a list), ``shell_width`` (0.6 voxels), ``activation``
(``ssp``; the paper uses ``relu``), ``gate_activation`` (``sigmoid``),
``normalization`` (``None`` or ``batch``, the equivariant batch norm),
``sigma`` (width of the atomic Gaussians, half a voxel), ``cutoff_fn``
(``cosine`` envelope of the neighbor densities), ``include_center``,
``energy_shift`` / ``energy_scale`` and ``atomic_energies``.

Grid sizes of the form ``4k + 1`` keep every strided grid centered, which
is what makes the invariance under the cube rotations exact. Like the
paper's networks the model is SE(3)- but not O(3)-equivariant: a reflected
structure is not constrained to the same energy.

3D CNN (``cnn``)
================
:class:`xnn.cnn.models.cnn3d.CNN3D` (registry name ``cnn3d``): the same
voxelized environments through blocks of ordinary ``Conv3d`` kernels, the
low-pass filtered strides, optional batch normalization and a global
average pool. It is the non-equivariant control of the paper (its channel
counts default to the component counts of the steerable fields of each
block): its energy changes under rotations of the structure.

Key options: ``species``, ``cutoff``, ``grid_size``, ``channels`` (or
``n_features`` + ``n_interactions``), ``kernel_size``, ``strides``,
``activation``, ``normalization``, ``smooth_stride``, and the voxel and
readout options of the steerable model.

HDNNP (``dnn``)
===============
:class:`xnn.dnn.models.hdnnp.HDNNP`: Behler–Parrinello high-dimensional
neural network potential: radial (G2) symmetry-function descriptors feeding
one MLP per element.

Key options: ``species``, ``cutoff`` (6.0), ``etas`` (0.05, 0.5, 2.0, 8.0),
``rs`` (0.0,), ``hidden`` (64, 64).

ANI (``dnn``)
=============
:class:`~xnn.dnn.models.ani.ANI`: per-element networks over the atomic
environment vector (Smith *et al.* 2017), matching `aiqm/torchani
<https://github.com/aiqm/torchani>`_ element for element. The four
published parameterisations are presets that fix the AEV grid, the network
shapes and the self energies:

.. code-block:: python

   from xnn.dnn.models import ANI

   ANI.ani1(species=[1, 6, 7, 8])      # ANI-1: 768-length AEV, Gaussian activation
   ANI.ani1x(species=[1, 6, 7, 8])     # ANI-1x: leaner 384-length AEV, CELU, per-element widths
   ANI.ani1ccx(species=[1, 6, 7, 8])   # ANI-1ccx: the ANI-1x architecture with coupled-cluster self energies
   ANI.ani2x()                         # ANI-2x: seven elements (H, C, N, O, S, F, Cl), 1008-length AEV

In a config, ``preset: ani-1x`` (``ani-1``, ``ani-1ccx``, ``ani-2x``) picks
the same. Each preset has its training set in the data hub
(``load_dataset("ani1x")`` and so on), so a published model can be
reproduced end to end; the pretrained torchani weights load as shown in
:ref:`fidelity`. Without a preset the options are ``species``,
``radial_cutoff`` (5.2), ``angular_cutoff`` (3.5), ``hidden`` (128, 128,
64), ``activation`` ("celu"), ``atomic_energies``, ``aev_kwargs``.

PhysNet (``dnn``)
=================
:class:`~xnn.dnn.models.physnet.PhysNet`: message passing with explicit
physics (Unke and Meuwly 2019): attention masks over an exponential-Gaussian
basis, residual blocks, per-module energy and charge heads, shielded
electrostatics of the corrected charges and D3(BJ) dispersion with
learnable coefficients. A pure-PyTorch translation of the TensorFlow
`MMunibas/PhysNet <https://github.com/MMunibas/PhysNet>`_. Elements up to
Z = 94 are embedded directly. Returns ``charges``, ``dipole`` and the
``nh_loss`` regularizer in addition.

Options: ``cutoff`` (10.0), ``lr_cutoff`` (None), ``n_features`` (128),
``n_rbf`` (64), ``n_interactions`` (5), ``num_residual_atomic`` (2),
``num_residual_interaction`` (3), ``num_residual_output`` (1),
``use_electrostatics`` (True), ``use_dispersion`` (True), ``s6`` / ``s8`` /
``a1`` / ``a2`` (None = learnable), ``d3_references`` ("2010"), ``species``
+ ``atomic_energies`` / ``atomic_scales``.

HDNNP (``dnn``)
===============
:class:`~xnn.dnn.models.hdnnp.HDNNP`: the Behler-Parrinello potential,
radial symmetry functions feeding one MLP per element. Under development.
Options: ``species``, ``cutoff`` (6.0), ``etas``, ``rs``, ``hidden`` (64, 64).

BAMBOO (``hybrid``)
===================
:class:`~xnn.hybrid.models.bamboo.BAMBOO`: a graph equivariant transformer
with a physics energy split (Gong *et al.* 2024). Multi-head attention on
the neighbor graph couples a scalar and a Cartesian vector channel (no
e3nn), and the energy splits into a neural term, a charge-equilibrium
electrostatic term summed over all pairs and an optional D3(CSO) term.
Reproduces `bytedance/bamboo <https://github.com/bytedance/bamboo>`_; xnn
returns the full conservative force, equal to upstream's ``forces +
qeq_force``. Works in kcal/mol and Å; returns ``charges``, ``dipole``,
``energy_nn`` and ``energy_elec`` in addition.

Options: ``cutoff`` (5.0), ``n_features`` (64), ``n_interactions`` (3),
``n_rbf`` (32), ``num_heads`` (16), ``charge_ub`` (2.0),
``charge_mlp_layers`` / ``energy_mlp_layers`` (2), ``act_fn`` ("silu"),
``attn_act_fn`` ("gelu"), ``use_electrostatics`` (True),
``coul_damping_beta`` (18.7), ``coul_damping_r0`` (2.2), ``use_dispersion``
(False), ``disp_cutoff`` (10.0), ``d3_references`` ("2010").

ReaxFF / ReaxFF-nn (``ffnn``)
=============================
:class:`xnn.ffnn.models.reaxff.ReaxFF`: the bond-order **reactive force
field** (van Duin *et al.*, *J. Phys. Chem. A* 2001; Nielson *et al.* 2005;
Senftle *et al.*, *npj Comput. Mater.* 2016) and its machine-learned variant
**ReaxFF-nn** (Guo *et al.*, *Comput. Mater. Sci.* 2020; Xue *et al.*, *PCCP*
2021), in one model. Bond orders are computed from interatomic distances
(sigma/pi/double-pi), corrected for over-coordination and residual 1-3
contributions — in nn mode by a per-species message-passing network — and
every valence term (bond, lone pair, over/under-coordination, valence angle,
penalty, three-body conjugation, torsion, four-body conjugation, hydrogen
bond) is written in these bond orders so it vanishes smoothly as bonds break.
Nonbonded terms are the tapered, shielded van der Waals and Coulomb
interactions, with partial charges equilibrated at every geometry by a
differentiable EEM solve (so autograd forces stay conservative). ``forward``
additionally returns ``"charges"`` and the full per-term energy decomposition
(``"e_bond"``, ``"e_angle"``, ``"e_vdw"``, ...).

The model is fully specified by a parameter library — a published field in
the SEAMM ``.frc`` force-field format (a dozen ship with xnn, e.g.
``ReaxFF("CHO_cho_2008")``; see :ref:`howto-forcefield-files`) or a
ReaxFF-nn JSON library (:mod:`xnn.ffnn.models.ffield`) — and the whole
functional form is differentiable, so any parameter group can be refit by
gradient descent (``trainable=...``); in nn mode the network weights are
always trainable. :func:`~xnn.ffnn.models.ffield.template_library` builds a
generic seed library for training from scratch, and
``ReaxFF.export_library()`` writes a trained force field back out — as a
``.frc`` file for a classical field, or as JSON when it carries network
weights. Every energy term is verified against the published equations
(see :ref:`fidelity` for why no third-party comparison is distributed).

Key options (defaults in parentheses): ``ffield`` (required), the parameter
library — a shipped field by name, a ``.frc`` path or a JSON path; ``cutoff`` (10.0), the nonbonded vdW/Coulomb/EEM cutoff and
neighbor-list radius; ``species`` (all in the library); ``nn`` (on iff the
library carries network weights); ``messages`` (the library's value), the
message-passing steps; ``hb_short``/``hb_long`` (6.75/7.5), the
hydrogen-bond distance window; ``trainable`` (none), the classical parameter
groups to refit. ReaxFF is evaluated in eV/Angstrom (libraries store
energies in kcal/mol; conversion is automatic).

OPLS / OPLS-AA / L-OPLS (``ffnn``)
==================================
:class:`xnn.ffnn.models.opls.OPLS`: the **fixed-topology classical force
field** of Jorgensen, Maxwell & Tirado-Rives (*J. Am. Chem. Soc.* 118,
11225, 1996): harmonic bonds and angles, Fourier-series proper dihedrals,
``V2`` improper dihedrals at trigonal centers, and Coulomb plus
Lennard-Jones nonbonded interactions with geometric combining rules and the
OPLS 1,2/1,3 exclusions and 1/2-scaled 1,4 pairs. The united-atom variant
(OPLS-UA) and reparameterizations such as **L-OPLS** for long hydrocarbons
(Siu *et al.*, *JCTC* 2012) share the functional form and differ only in
their parameter libraries, so one model serves all of them. ``forward``
additionally returns ``"charges"`` and the per-term decomposition
(``"e_bond"``, ``"e_angle"``, ``"e_torsion"``, ``"e_improper"``, ``"e_lj"``,
``"e_coulomb"``, ``"e_lj14"``, ``"e_coulomb14"``).

Unlike ReaxFF, OPLS needs a fixed molecular topology: a
:class:`~xnn.ffnn.models.topology.MolecularTopology` holds the per-atom
OPLS type names and the bond list and derives the angles, dihedrals,
exclusions and 1,4 pairs. ``OPLS.from_atoms(atoms, "oplsaa")`` builds all of
it from a structure: bonds are perceived with RDKit and the atom types are
assigned from the **SMARTS templates** the parameter file carries
(:func:`~xnn.ffnn.common.typing.assign_atom_types`), so the force field's
own type names never have to be spelled out. The topology binds to the
model instance, so every structure the model evaluates — a training batch
of conformers, an MD trajectory — is a conformation of that system; bonded
terms use minimum-image displacements, so molecules may wrap across periodic
boundaries. Parameters come from an :class:`~xnn.ffnn.models.oplslib.OPLSLibrary`
read from a SEAMM ``.frc`` force-field file (:ref:`howto-forcefield-files`):
the OPLS-AA distribution ships with xnn (``"oplsaa"``; ``"CL&P"`` and
``"oplsaa+"`` add the ionic-liquid extension, loadable with
``strict=False`` since their tabulated ``PF6-`` angle is not implemented),
together with ``"oplsaa-1996"`` (the paper's original alkane and alcohol
torsions, which reproduce its Table 1) and ``"lopls"``; the native JSON
format round-trips trained parameters. Any parameter group can be refit by
gradient descent (``trainable=("dihedral_v", "charge", ...)``), several
models (different molecules) can share one
:class:`~xnn.ffnn.models.opls.OPLSForceField` to fit transferable
parameters jointly, and ``export_library()`` writes the trained values back
to a ``.frc`` file (``save_frc``) or JSON. The implementation is verified
against OpenMM to ~1e-7 kJ/mol and against Table 1 of the 1996 paper (see
:ref:`fidelity`).

Key options (defaults in parentheses): ``library`` (required), the parameter
source (a shipped variant name, a ``.frc`` path or a JSON path);
``topology`` (required), a topology JSON path or inline ``types`` + ``bonds``
(+ ``impropers``; improper parameters resolve by class pattern, or are placed
automatically at trigonal centers); ``cutoff`` (10.0), the
Lennard-Jones/Coulomb cutoff and neighbor-list radius (1,4 pairs are
cutoff-independent); ``switch_width`` (0.0), a quintic switching window at
the cutoff; ``fudge_lj``/``fudge_qq`` (the library's, 0.5 for OPLS), the 1,4
scalings; ``trainable`` (none), the parameter groups to refit. OPLS is
evaluated in eV/Angstrom (libraries store kcal/mol; conversion is
automatic).

DREIDING / DREIDING-X6 (``ffnn``)
=================================
:class:`xnn.ffnn.models.dreiding.Dreiding`: the **rule-generated generic
force field** of Mayo, Olafson & Goddard III (*J. Phys. Chem.* 94, 8897,
1990). Where OPLS tabulates a parameter per bond, angle and torsion type,
DREIDING generates them all from a handful of per-atom generators by
hybridization rules: bond lengths are sums of atomic radii
(``R0_IJ = R0_I + R0_J - 0.01``, eq 6) with one universal stretch constant
scaled by the bond order (eqs 7-9), the equilibrium angle depends only on
the central atom and every bend shares ``K = 100 kcal/mol/rad^2``
(eqs 10-12), and each torsion's barrier, periodicity and phase follow from
the hybridizations of the two central atoms plus the bond order between
them (eqs 13-23, :func:`~xnn.ffnn.models.dreidinglib.torsion_rule`). That
is what makes DREIDING *generic*: it has parameters for element
combinations nobody tabulated.

The energy adds spectroscopic inversions at planar and stereo centers
(eq 28, all three axis choices averaged), van der Waals interactions in
either the Lennard-Jones 12-6 form (eq 31', the ``"dreiding"`` variant) or
the exponential-6 form (eq 32', ``"dreiding/X6"``), optional Coulomb
interactions with the paper's constant (eq 37; DREIDING prescribes no
charges of its own, and ``Dreiding.from_atoms(..., charges="gasteiger")``
supplies the paper's recommended estimate), and the explicit 12-10
hydrogen-bond term on ``H__HB`` donor triplets (eq 38). Unlike OPLS,
**1,4 pairs count in full** -- only 1,2 and 1,3 pairs are excluded, as the
paper specifies. ``forward`` additionally returns ``"charges"`` and the
per-term decomposition (``"e_bond"``, ``"e_angle"``, ``"e_torsion"``,
``"e_inversion"``, ``"e_vdw"``, ``"e_coulomb"``, ``"e_hbond"``).

Like OPLS, DREIDING binds to a fixed
:class:`~xnn.ffnn.models.topology.MolecularTopology`, but the topology also
carries **bond orders**, since the rules read them.
``Dreiding.from_atoms(atoms, "dreiding")`` builds everything from a
structure: RDKit perceives connectivity *and* bond orders, and the SMARTS
templates in the parameter file assign the DREIDING types. For
resonance-delocalized bonds the automatic perception is not always what
DREIDING intends -- an amide C-N is described as ``C_R``-``N_R`` with bond
order 1.5 (the paper's footnote 8), not as ``C_2``-``N_3`` -- so set the
types and orders explicitly in those cases.

Because nothing bonded is tabulated, what is trainable are the *generators*
themselves: ``radius``, ``theta0``, ``bond_k``, ``bond_d``, ``angle_k``,
``torsion_v`` (one total barrier per rule), ``oop_k``, ``oop_psi0``,
``vdw_r0``, ``vdw_d0``, ``x6_zeta``, ``hbond_d0``, ``hbond_r0``, plus the
model's per-atom ``charge``. Several models can share one
:class:`~xnn.ffnn.models.dreiding.DreidingForceField` to fit transferable
parameters across molecules jointly, and ``export_library()`` writes the
trained values back to a JSON library that ``ffield:`` accepts. Parameters
come from the SEAMM ``dreiding.frc`` shipped with xnn
(:ref:`howto-forcefield-files`), whose two ``#define`` variants select the
nonbond form. The implementation is verified term by term against LAMMPS's
DREIDING styles to ~1e-10 kcal/mol and reproduces Tables XI and XII of the
1990 paper (see :ref:`fidelity`).

Key options (defaults in parentheses): ``ffield`` (``"dreiding"``), the
parameter source (``"dreiding"``, ``"dreiding/X6"``, a ``.frc`` path or a
JSON path); ``topology`` (required), a topology JSON path or inline
``types`` + ``bonds`` (+ ``bond_orders``); ``cutoff`` (10.0), the
nonbonded cutoff and neighbor-list radius; ``switch_width`` (0.0), a
quintic switching window at the cutoff; ``charges`` (none), per-atom
partial charges; ``bond_style`` (``"harmonic"``, or ``"morse"`` for
DREIDING/M); ``angle_style`` (``"cosine"``, the harmonic-cosine form of
eq 10a, or ``"harmonic"`` for eq 11); ``hbond`` (True),
``hbond_cutoff``/``hbond_angle`` (the nonbonded cutoff / 90 degrees);
``trainable`` (none), the generator groups to refit. DREIDING is evaluated
in eV/Angstrom (libraries store kcal/mol; conversion is automatic).

Long-range interactions: Latent Ewald Summation (LES)
======================================================
Short-range models miss electrostatics and dispersion beyond their receptive
field. :class:`~xnn.common.models.les.LatentEwald` (Cheng, *npj Comput
Mater* 2025; the CACE-LR method) fixes this for **any** registered model: a
small MLP maps each atom's invariant features to a latent charge ``q`` and an
Ewald summation over ``q`` (:class:`~xnn.common.models.les.EwaldSummation`)
adds the long-range energy. Enable it from a config --

.. code-block:: yaml

   model:
     name: cace            # or mace | nequip | allegro | schnet | physnet | ...
     extra:
       long_range: {n_channels: 4, sigma: 1.0, dl: 2.0}

-- or wrap directly with ``LatentEwald(model, n_channels=4)``. Every model
exposes the required invariant per-atom features through the
``"node_features"`` output key (and ``node_feature_dim``): CACE's symmetrized
B features, the scalar channels of MACE/NequIP features, Allegro's
environment-aggregated edge latents, SchNet/PhysNet feature vectors, and the
HDNNP/ANI descriptors. Key options: ``n_channels`` (4), ``hidden``
([24, 12] q-MLP), ``sigma`` (1.0 -- Gaussian smearing), ``dl`` (2.0 -- the
k-space cutoff is ``2*pi/dl``), ``exponent`` (1 for electrostatics, 6 for
dispersion), ``remove_self_interaction`` (False), and the optional net-charge
constraint ``constrain_charge`` (False): after the q head, each structure's
charges are shifted so that channel ``charge_channel`` (0) sums to its
``total_charge`` in latent units (``LATENT_CHARGE_PER_E`` = 9.5118 per e,
the unit in which two charges interact with the Coulomb energy in eV and Å)
and every other channel to zero, uniformly over the atoms or with learned
per-atom weights (``charge_weights: uniform | learned``). The sum is then
exact for any system size, which a bias in the q head (``q_bias``) only
approximates, and charged clusters get the monopole energy of their net
charge; charged training structures must carry their ``total_charge``
(``charge``) label, since a missing label means neutral. Non-periodic structures use
the equivalent real-space direct sum, which is exact and needs no k-space
cutoff: on a dataset without cells ``dl`` is therefore **inert**, and
``exponent = 6`` is rejected outright (the real-space branch implements the
``1/r`` kernel only), so a dispersion channel needs periodic training data.
Forces and stress flow through
:class:`~xnn.common.models.outputs.ForceStressOutput` unchanged. The outputs
gain ``"energy_sr"``, ``"energy_lr"`` and ``"latent_charges"``.

London dispersion: DFT-D3 and DFT-D4
====================================
:class:`~xnn.common.models.d4.D4Dispersion` adds the DFT-D4 dispersion
energy of Caldeweyher *et al.* (*J. Chem. Phys.* **150**, 154122, 2019) to
**any** registered model, or stands alone as the model ``"d4"``. It is an
independent PyTorch implementation of the default D4 model (EEQ partial
charges, BJ-damped two-body term, approximate ATM three-body term) that
reproduces the reference ``dftd4`` code to floating-point precision --
energies, forces, virials, coordination numbers, charges, polarizabilities
and C6 coefficients, molecular and periodic (Ewald-summed EEQ). Enable it
from a config --

.. code-block:: yaml

   model:
     name: mace            # or nequip | allegro | cace | schnet | physnet | hdnnp | ani | reaxff | ...
     extra:
       dispersion: {s6: 1.0, s8: 1.20065498, a1: 0.40085597, a2: 5.02928789, s9: 1.0}

-- ``dispersion: true`` selects those PBE0-D4 defaults -- or wrap directly
with ``D4Dispersion(model, **options)``. The wrapper's ``cutoff`` becomes the
larger of the model's and the D4 cutoffs (the :class:`Trainer` and ``xnn
export`` build neighbor lists with it), and the wrapped model only ever sees
the edges within its own radius. Options (:class:`~xnn.common.models.d4.DFTD4`):
the damping parameters ``s6, s8, a1, a2, s9, alp`` (PBE0-D4 values by default;
``s9: 0`` disables the three-body term; ``trainable: true`` makes them
learnable), the model constants ``ga, gc, wf`` (3, 2, 6), the real-space
cutoffs ``cutoff_pair, cutoff_triple, cutoff_cn, cutoff_eeq_cn`` in Angstrom
(upstream's 60 / 40 / 30 / 25 bohr by default, which reproduce ``dftd4``
exactly but are far longer than an MLIP needs -- for condensed-phase training
use 10-15 Angstrom for the pair term and less for the triples), and the
quintic switching windows ``switch_width_pair, switch_width_triple`` (0 by
default; a few Angstrom keeps energy and forces continuous under a finite
cutoff in MD). The total charge of each structure is read from the graph's
``total_charge`` (``atoms.info["charge"]`` in ASE files; neutral when
absent). The outputs gain ``"energy_sr"``, ``"energy_disp"``,
``"energy_2body"``, ``"energy_3body"``, ``"eeq_charges"``,
``"coordination_numbers"``, ``"polarizabilities"`` and
``"dynamic_polarizabilities"`` (pairwise C6 via
:func:`~xnn.common.models.d4.c6_matrix`). D4 and LES combine freely (LES
sees the D4-corrected model's features), and every deploy channel carries
the term: :class:`~xnn.common.models.outputs.ForceStressOutput`, the ASE
calculator, the TorchScript export (both ABIs) and the LAMMPS wrapper.

**Long-range tail.** ``tail_correction: true`` adds, for periodic structures,
the two-body dispersion the pair cutoff and its switching window remove,
assuming a uniform distribution of atoms beyond the window (the analog of
LAMMPS ``pair_modify tail yes``, which does not reach energies returned
through ``fix mdi/qm``). It uses the structure's own charge- and CN-dependent
C6 and the BJ damping, and is differentiable, so its forces and stress are
consistent with the energy; molecules are unaffected, and it is off by
default. On 192 periodic water atoms (pair term only, 2 Angstrom switch) a
12 Angstrom cutoff without it is 0.48 meV/atom and 82 atm short of the
converged pair term; with it, 12, 20 and 30 Angstrom agree to 0.003 meV/atom
and 3 atm. The outputs gain ``"energy_tail"``. The three-body term has no
tail correction: beyond a 10 Angstrom triple cutoff it is worth about 10 atm
in water, and a uniform-fluid ATM tail is not a reliable estimate.

**Scale regimes.** The EEQ charges are a charge-constrained linear system of
size ``N``. ``regime: dense`` (bit-exact ``dftd4`` parity) builds its
``(N, N)`` matrix with every intermediate retained for the backward pass,
which exhausts 80 GB near 1500 periodic or 25000 molecular atoms;
``regime: large`` applies the same system as a matrix-free Ewald operator
(neighbor-list real space, structure-factor reciprocal space), solves it by
LU or conjugate gradients (``eeq_solver: auto | lu | cg``) and
differentiates it implicitly, so memory is linear in the neighbor list and
reciprocal set and gradients cost one operator application. ``regime: auto``
(default) switches above 1500 / 6000 atoms. The two regimes agree to the
reference code's own Ewald tolerance (about 1e-8 in the charges); forces,
stress and force training are exact in both. ``cutoff_eeq`` sets the
real-space range of the split: by default 16 Å, or the largest of the other
cutoffs if that is larger (a longer range shrinks the reciprocal set as
``r^-3`` and makes the EEQ operator about four times cheaper than at 12 Å,
with charges unchanged to 2e-10 e; the neighbor list is then at most 16 Å,
about 2.4 times the edges of a 12 Å list). ``regime: dense`` keeps the other
cutoffs' radius, and an explicit value always wins; crystals need at least
20 bohr. In float32 the EEQ solve of both regimes refines its solution
with float64 residuals, so charges and forces are as accurate as the
float32 matrix allows (dense regime, 1536 atoms: charges 4e-5 to 2e-6 e,
forces 9e-6 to 1.5e-7 eV/Å against float64). The dense path factors the
matrix once and differentiates through the residual rather than through
the factor, so the refined solve is no slower than the plain one, and its
first and second derivatives (forces, force-training gradients, Hessians)
carry the same accuracy. The three-body term runs in
recompute blocks of centers by default (``checkpoint_triplets: true``;
``triplet_chunk`` sets the block size, by default from the free device
memory) and the two-body term in recompute blocks of edges
(``recompute_pairs: true``): memory bounded by one block at every derivative
order (force training included), no ``(N, N)`` C6 matrix and no retained
``(E, 23)`` polarizability products. The three-body blocks form the pair C6
once per edge, visit each triangle of distinct atoms once, and take their
first derivative in closed form; on 5184 water atoms (float32, A100) the
full D4 step with forces and stress went from 3.7 s to 0.7 s at an 8 Å
triple cutoff and from 7.6 s to 1.8 s at 10 Å. On an A100 a 5000-atom water cell with 12 / 8 A
cutoffs takes 5 s for energy, forces and stress and 13 s for a force-loss
training step in 21 GB, where the dense regime runs out of 80 GB at 1500
atoms. Both are eager-only; TorchScript exports pin the dense paths.

For molecular dynamics two more options pay. ``triplet_cache`` (on by
default: a quarter of the free device memory; a number in GB, or 0 to switch
it off) keeps each three-body block's triples from the forward to the
backward pass, so they are enumerated once per step instead of twice; it is
exact and costs 14 bytes per triple (0.5 GB for 5000 water atoms at an 8 Å
triple cutoff). :meth:`~xnn.common.models.d4.DFTD4.enable_eeq_reuse` (``xnn
mdi --eeq-reuse``, ``XNNCalculator(..., eeq_reuse=True)``) carries the
large-regime EEQ solve from one step to the next
(:class:`~xnn.common.models.eeq.EEQReuse`), for a structure evaluated on its
own (batches take the fresh path): a preconditioner formed once
from an earlier step's matrix and warm-started conjugate gradients replace
the assembly and factorization of every step, with results equal to the
fresh solve to its tolerance (1e-9 relative residual in float64, 1e-6 in
float32). On 5001
water atoms (A100, float32, real 0.5 fs MD frames) MACE-LES + D4 at an 8 Å
triple cutoff went from 0.83 s to 0.47 s per step with both and the 16 Å
EEQ range. ``tools/d4_bench.py`` and ``tools/d4_md_step.py`` measure these
settings on a water box or on a trajectory.

The geometry-only predecessor **DFT-D3** (Grimme *et al.*, *J. Chem. Phys.*
**132**, 154104, 2010; BJ damping from Grimme, Ehrlich & Goerigk, *J. Comput.
Chem.* **32**, 1456, 2011) is available the same way as
:class:`~xnn.common.models.d3.D3Dispersion` / the model ``"d3"``, verified
against the reference ``simple-dftd3`` to floating-point precision for every
damping function. Select it with ``dispersion: {name: d3, ...}``; the options
(:class:`~xnn.common.models.d3.DFTD3`) are ``damping`` (``bj`` -- the 2011
default -- ``zero``, ``mzero`` or ``op``), the damping parameters ``s6, s8,
s9, a1, a2, rs6, rs8, alp, bet`` (PBE0 values of the papers by default:
``s8 = 1.2177, a1 = 0.4145, a2 = 4.8593`` for BJ, ``s8 = 0.928, rs6 = 1.287``
for zero damping; ``s9 = 0`` as recommended in the 2010 paper -- set ``s9: 1``
for D3-ATM), the cutoffs ``cutoff_pair, cutoff_triple, cutoff_cn`` (upstream's
60 / 40 / 40 bohr) and the switching widths, ``trainable``, and
``references``: ``"2024"`` (default) follows the current reference code,
which re-parametrized the actinides Fr-Pu in ``simple-dftd3`` 1.1.0 and
added Am-Lr; ``"2010"`` selects Grimme's original reference systems (Z <= 94),
the tables of the original D3 codes. The two sets are identical for Z <= 86.
D3 needs no charges, so the outputs carry ``"coordination_numbers"`` and the
dense ``"c6_matrix"`` instead of the EEQ quantities. Both dispersion models
share their machinery (:mod:`~xnn.common.models.dispersion`) and one data
file; the legacy functional D3 API used inside PhysNet and BAMBOO reads the
2010 set from it by default, so those models stay bit-identical to their
upstream codes.

Forces and stress
=================
Wrap any model to get autograd forces and stress:

.. code-block:: python

   from xnn.ffnn.models import ReaxFF

   model = ReaxFF("CHO_cho_2008", trainable=("bond", "angle"))

Options: ``ffield`` (required), ``cutoff`` (10.0, the nonbonded radius),
``species``, ``nn``, ``messages``, ``hb_short`` / ``hb_long`` (6.75 / 7.5),
``trainable``. Evaluated in eV and Å; libraries store kcal/mol.

OPLS / L-OPLS (``ffnn``)
========================
:class:`~xnn.ffnn.models.opls.OPLS`: the fixed-topology force field of
Jorgensen, Maxwell and Tirado-Rives (1996): harmonic bonds and angles,
Fourier torsions, impropers at trigonal centers, Coulomb plus Lennard-Jones
with OPLS exclusions and scaled 1,4 pairs. The topology (types, bonds,
angles, dihedrals) binds to the model; ``OPLS.from_atoms`` builds it from a
structure with RDKit and the SMARTS templates of the parameter file. Any
parameter group is trainable, several molecules can share one force field
to fit transferable parameters, and ``export_library()`` writes a ``.frc``
file. Matches OpenMM to about 1e-7 kJ/mol. Returns ``charges`` and the
per-term energies.

.. code-block:: python

   from xnn.ffnn.models import OPLS

   model = OPLS.from_atoms(atoms, "oplsaa", cutoff=10.0)    # "lopls", "oplsaa-1996", "CL&P"

Options: ``library`` (required), ``topology`` (required; built by
``from_atoms``), ``cutoff`` (10.0), ``switch_width`` (0.0), ``fudge_lj`` /
``fudge_qq`` (the library's), ``trainable``.

DREIDING / DREIDING-X6 (``ffnn``)
=================================
:class:`~xnn.ffnn.models.dreiding.Dreiding`: the rule-generated generic
force field of Mayo, Olafson and Goddard (1990). Bond, angle and torsion
parameters are generated from per-atom generators by hybridization rules,
which is what gives it parameters for element combinations nobody
tabulated; inversions, Lennard-Jones (``"dreiding"``) or exponential-6
(``"dreiding/X6"``) van der Waals, optional Coulomb and an explicit
hydrogen-bond term complete the energy. Like OPLS it binds to a topology,
which also carries bond orders; ``Dreiding.from_atoms`` perceives both.
The trainable parameters are the generators themselves. Matches the LAMMPS
DREIDING styles to about 1e-10 kcal/mol.

.. code-block:: python

   from xnn.ffnn.models import Dreiding

   model = Dreiding.from_atoms(atoms, "dreiding", cutoff=10.0, charges="gasteiger")

Options: ``ffield`` ("dreiding"), ``topology`` (required), ``cutoff``
(10.0), ``switch_width`` (0.0), ``charges``, ``bond_style`` ("harmonic" |
"morse"), ``angle_style`` ("cosine" | "harmonic"), ``hbond`` (True),
``hbond_cutoff`` / ``hbond_angle``, ``trainable``. For resonance-delocalized
bonds such as an amide C-N, set the types (``C_R``-``N_R``, order 1.5)
explicitly; automatic perception does not.

Add-ons for any model
=====================
Three wrappers add physics to any registered model. They are enabled from
the ``model`` section of a config or by wrapping a model object, every
deploy channel carries them, and they combine freely.

Long-range electrostatics (LES)
-------------------------------
:class:`~xnn.common.models.les.LatentEwald` (Cheng 2025, the CACE-LR
method): a small MLP maps each atom's invariant features to latent charges,
and an Ewald sum over them adds the long-range energy. Every model exposes
the features it needs through the ``node_features`` output.

.. code-block:: yaml

   model:
     name: cace                       # or any other model
     long_range: {n_channels: 4, sigma: 1.0, dl: 2.0}

Options: ``n_channels`` (4), ``hidden`` ([24, 12]), ``sigma`` (1.0),
``dl`` (2.0; the k-space cutoff is ``2*pi/dl``), ``exponent`` (1; 6 for
dispersion), ``remove_self_interaction`` (False). Non-periodic structures
use the exact real-space sum, so ``dl`` is inert on them and ``exponent: 6``
needs periodic data. Adds ``energy_sr``, ``energy_lr`` and
``latent_charges`` to the outputs.

Dispersion: DFT-D4 and DFT-D3
-----------------------------
:class:`~xnn.common.models.d4.D4Dispersion` adds the D4 energy of
Caldeweyher *et al.* (2019): EEQ charges, BJ-damped two-body and ATM
three-body terms, reproducing ``dftd4`` to floating-point precision.
:class:`~xnn.common.models.d3.D3Dispersion` adds the geometry-only D3
energy of Grimme *et al.* (2010, 2011), reproducing ``simple-dftd3`` for
every damping function. Both also stand alone as the models ``"d4"`` and
``"d3"``.

.. code-block:: yaml

   model:
     name: mace
     dispersion: true                                   # PBE0-D4 defaults
     # dispersion: {s6: 1.0, s8: 1.2, a1: 0.4, a2: 5.0}  # D4 with explicit damping
     # dispersion: {name: d3, damping: bj, s9: 1.0}      # D3(BJ) with the three-body term

The wrapper's cutoff is the larger of the model's and the dispersion
cutoffs; the model only sees edges within its own radius. The net charge
comes from the graph's ``total_charge``. Outputs gain ``energy_disp``,
``energy_2body``, ``energy_3body`` and, for D4, ``eeq_charges``,
``coordination_numbers`` and ``polarizabilities``.

Options (:class:`~xnn.common.models.d4.DFTD4` and
:class:`~xnn.common.models.d3.DFTD3`):

- Damping: ``s6, s8, a1, a2, s9, alp`` (PBE0 values; ``s9: 0`` disables the
  three-body term; ``trainable: true`` makes them learnable). D3 adds
  ``damping`` (``bj`` | ``zero`` | ``mzero`` | ``op``) with ``rs6, rs8, bet``,
  and ``references`` (``"2024"`` or Grimme's original ``"2010"`` set).
- Cutoffs in Å: ``cutoff_pair, cutoff_triple, cutoff_cn`` (and
  ``cutoff_eeq_cn``, ``cutoff_eeq`` for D4). The defaults reproduce the
  reference codes but are far longer than an MLIP needs; for condensed-phase
  work use 10 to 15 Å for the pair term and less for the triples, with the
  quintic switching windows ``switch_width_pair, switch_width_triple`` to keep
  energy and forces continuous in MD.
- ``tail_correction: true`` (D4) adds the two-body energy the pair cutoff
  removes in periodic cells, assuming a uniform density beyond it.
- D4 charge regimes: ``regime: dense`` builds the full EEQ matrix (bit-exact
  with ``dftd4``; up to about 1500 periodic atoms on 80 GB), ``large`` applies
  it matrix free with implicit differentiation, ``auto`` (default) switches
  by size. Both give exact forces, stress and training gradients.
- Molecular dynamics: ``enable_eeq_reuse`` (``xnn mdi --eeq-reuse``,
  ``XNNCalculator(..., eeq_reuse=True)``) carries the large-regime charge
  solve between steps, and the fused kernels of :ref:`howto-fast-paths`
  speed up the three-body term at every size.

A model trained on labels with the dispersion removed records that in
``subtracted_dispersion`` (:ref:`deployment`).
