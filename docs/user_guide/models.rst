.. _models:

******
Models
******

All models subclass
:class:`~xnn.common.models.base.InteratomicPotential`: they take an
:class:`~xnn.common.data.atomic_data.AtomicGraph` and return a dictionary
with ``node_energy`` (per atom) and ``energy`` (per structure). Forces and
stress are added uniformly by
:class:`~xnn.common.models.outputs.ForceStressOutput`; no model implements
them itself.

Models are registered by name, so they can be built from any config
frontend:

.. code-block:: python

   from xnn.common.models import available_models, build_model

   available_models()          # ['allegro', 'ani', 'bamboo', 'cace', 'hdnnp', 'mace', 'nequip', 'physnet', 'reaxff', 'schnet']
   model = build_model(cfg.model)   # dispatches to <Model>.from_config(cfg.model)

Each model can also be constructed directly; the constructor arguments below
double as the keys accepted in ``model.extra`` of a config file.

.. note::

   The GNN models (NequIP, MACE, Allegro, CACE) register themselves when
   ``xnn.gnn`` is importable, which requires the ``gnn`` extra (``e3nn``).
   CACE itself works entirely in Cartesian coordinates and does not use
   e3nn. The hybrid model (BAMBOO) and the shared building blocks in
   ``xnn.transformer`` likewise need no e3nn and are always available.

MACE (``gnn``)
==============
:class:`xnn.gnn.models.mace.MACE`: higher-order equivariant message
passing with the learned symmetric contraction of Batatia *et al.*
Faithful to `ACEsuit/mace <https://github.com/ACEsuit/mace>`_ (see
:ref:`fidelity`).

Key options (defaults in parentheses): ``species``, ``cutoff`` (4.0),
``max_ell`` (3), the spherical-harmonic order of the edges, ``max_L`` (0),
the order of the message irreps, ``num_channels`` (32), ``n_rbf`` (8),
``num_interactions`` (2), fully flexible T = 0..N, ``correlation`` (3),
the order of the symmetric contraction, ``MLP_irreps`` ("16x0e"),
``radial_MLP``, ``interaction`` ("RealAgnosticResidualInteractionBlock"),
``interaction_first``, ``gate`` ("silu"), ``avg_num_neighbors`` (1.0),
``hidden_irreps``, ``num_cutoff_basis`` (5), ``radial_type`` ("bessel"),
``distance_transform`` ("None" | "Agnesi" | "Soft", the chemistry-aware
warp of the radial coordinate used by the newer foundation models),
``pair_repulsion`` (False), which adds ZBL core repulsion,
``atomic_energies``, the per-species reference energies (E0s), and
``scale`` / ``shift`` (1, 0), the upstream *ScaleShiftMACE* affine on the
per-atom interaction energy (``E_i = E0_i + scale * E_int,i + shift``).
The density-normalized interaction generation is available as
``RealAgnosticDensity(Residual)InteractionBlock``.

**Pretrained foundation models.** ``MACE.from_foundation()`` loads any of
the published MACE-MP / MACE-OFF checkpoints (MP-0, 0b/0b2/0b3, MPA-0,
OMAT-0, MATPES, the multi-head MH-0, OFF23; see
:data:`xnn.gnn.models.mace_foundation.FOUNDATION_MODELS` for the aliases
and licenses) and converts it weight-for-weight into this implementation,
verified to float64 round-off against ``mace-torch``
(:ref:`fidelity`). Multi-head checkpoints are sliced to a chosen ``head``.
In a config, ``foundation: mace-off23-small`` (with ``cutoff`` set to the
checkpoint's ``r_max``) builds the pretrained model instead of a fresh
one, so the standard :class:`~xnn.common.train.Trainer` fine-tunes it
directly. Loading requires the ``mace-torch`` package to unpickle the
checkpoint; the converted model does not. The one unsupported checkpoint
is ``mace-mh-1`` (a next-generation architecture); the converter raises
``NotImplementedError`` naming the unsupported piece.

NequIP (``gnn``)
================
:class:`xnn.gnn.models.nequip.NequIP`: E(3)-equivariant message passing
with gated nonlinearities (Batzner *et al.*). Faithful to
`mir-group/nequip <https://github.com/mir-group/nequip>`_, with directly
transplantable state dicts.

Key options: ``species``, ``cutoff`` (4.0), ``l_max`` (2), ``parity``
(True), ``n_rbf`` (8), ``n_layers`` (3), ``num_features`` (32),
``invariant_layers`` (2), ``invariant_neurons`` (64),
``avg_num_neighbors``, ``use_sc`` (True), the self-connection, ``resnet``
(False), ``nonlinearity_scalars`` / ``nonlinearity_gates``,
``num_polynomial_cutoff`` (6), ``trainable_rbf`` (True),
``conv_to_output_hidden``, ``atomic_energies``, and ``atomic_scales``, the
per-species energy scale/shift.

Allegro (``gnn``)
=================
:class:`xnn.gnn.models.allegro.Allegro`: strictly local equivariant
many-body potential (Musaelian *et al.*), without message passing between
atoms. Faithful to `mir-group/allegro <https://github.com/mir-group/allegro>`_
v0.3.0 (``uuulin`` mode), with directly transplantable state dicts.

Key options: ``species``, ``cutoff`` (4.0), ``l_max`` (1), ``parity``
("o3_full"), ``n_rbf``, ``num_layers``, ``num_tensor_features``,
``two_body_latent`` / ``latent`` / ``env_embed`` / ``edge_eng``, the MLP
widths, ``initial_scalar_embedding_dim``, ``avg_num_neighbors``,
``latent_resnet`` (True), ``num_polynomial_cutoff`` (6), ``trainable_rbf``
(True), ``atomic_energies``, ``atomic_scales``.

CACE (``gnn``)
==============
:class:`xnn.gnn.models.cace.CACE`: Cartesian atomic cluster expansion
(Cheng, *npj Comput Mater* 2024): body-ordered invariant features built
entirely in Cartesian coordinates (monomial angular basis, multinomial
symmetrization instead of Clebsch–Gordan contraction), with a
low-dimensional element embedding, trainable radial channel coupling and
optional message passing. Faithful to
`BingqingCheng/cace <https://github.com/BingqingCheng/cace>`_, with
transplantable weights (see :ref:`fidelity`); the only GNN model here that
needs no spherical harmonics.

Key options: ``species``, ``cutoff`` (5.5), ``n_atom_basis`` (3), the
element embedding length (edge channels are its square), ``n_rbf`` (8),
``n_radial_basis``, the mixed radial channels (``n_rbf``), ``max_l`` (3),
``max_nu`` (3), the maximum body order of the invariants (1–4),
``num_message_passing`` (1), fully flexible T = 0..N (0 = plain Cartesian
ACE), ``message_types`` (["M", "Ar", "Bchi"]), i.e. node memory /
radial-filter message / recursive edge embedding, ``embed_receiver_nodes``
(False), ``avg_num_neighbors`` (10.0), ``num_polynomial_cutoff`` (6),
``trainable_rbf`` (True), ``readout_hidden`` ([32, 16]),
``atomic_energies``.

SchNet (``cnn``)
================
:class:`xnn.cnn.models.schnet.SchNet`: continuous-filter convolutions over
a Gaussian radial basis with shifted-softplus activations (Schütt *et al.*,
NIPS 2017). Faithful to the manuscript (see :ref:`fidelity`): the defaults
are the paper architecture — ``F = 64`` feature maps, ``T = 3`` residual
interaction blocks, RBF centers every 0.1 Å on ``[0, 30]`` with
``gamma = 10`` Å\ :sup:`-2` — plus the DTNN per-atom energy standardization
(``energy_shift``/``energy_scale``, or
:meth:`~xnn.cnn.models.schnet.SchNet.set_energy_scale_shift`).

Key options: ``n_features`` (64), ``n_interactions`` (3), ``n_rbf`` (301),
``cutoff`` (30.0), ``gamma`` (10.0), ``cutoff_fn`` (``None``; set
``"cosine"`` to smooth the filters at a finite cutoff for condensed phases),
``energy_shift`` (0.0), ``energy_scale`` (1.0), ``species`` +
``atomic_energies`` (per-element reference energies loaded into
``atom_ref``).

HDNNP (``dnn``)
===============
:class:`xnn.dnn.models.hdnnp.HDNNP`: Behler–Parrinello high-dimensional
neural network potential: radial (G2) symmetry-function descriptors feeding
one MLP per element.

Key options: ``species``, ``cutoff`` (6.0), ``etas`` (0.05, 0.5, 2.0, 8.0),
``rs`` (0.0,), ``hidden`` (64, 64).

ANI (``dnn``)
=============
:class:`xnn.dnn.models.ani.ANI`: the ANI potential (Smith *et al.* 2017),
per-element networks over the Atomic Environment Vector (radial + angular
symmetry functions), verified element-for-element against ``aiqm/torchani``.

ANI-1 vs. ANI-1x vs. ANI-1ccx vs. ANI-2x: choosing a preset
-----------------------------------------------------------
There are **four published ANI parameterisations**: three for H/C/N/O and the
seven-element ANI-2x (the only one that also covers S, F, and Cl). In ``xnn``
each one is a *preset*: a classmethod that fills in the AEV grid, the
per-element network shapes, the activation, and the self atomic energies so you
do not have to. Pick the preset, not the individual knobs.

- **ANI-1** (Smith *et al.*, *Chem. Sci.* 2017): the original model, trained on
  the ~20 M-conformation ANI-1 dataset (dense normal-mode sampling of GDB-11
  molecules). Preset: :meth:`~xnn.dnn.models.ani.ANI.ani1`.
- **ANI-1x** (Smith *et al.*, *J. Chem. Phys.* 2018, *"Less is more"*): the
  later model built by **active learning**: it iteratively adds only the
  conformations where an ensemble disagrees, giving a smaller (~5 M) but more
  diverse and more transferable training set. It also uses a *different, leaner*
  AEV grid. Preset: :meth:`~xnn.dnn.models.ani.ANI.ani1x`.
- **ANI-1ccx** (Smith *et al.*, *Nat. Commun.* 2019): the ANI-1x architecture
  retrained by **transfer learning** to ~500 k coupled-cluster (CCSD(T)*/CBS)
  energies, holding 65,280 of the 325,248 network weights fixed (the matrix
  joining each element network's first two hidden layers) to avoid overfitting
  the smaller coupled-cluster set. The descriptor and networks are *identical*
  to ANI-1x; only the training data, self atomic energies, and resulting
  weights differ. Preset: :meth:`~xnn.dnn.models.ani.ANI.ani1ccx`.
- **ANI-2x** (Devereux *et al.*, *J. Chem. Theory Comput.* 2020): the only
  seven-element parameterisation, extending ANI from H/C/N/O to **seven
  elements** by adding S, F, and Cl. It pairs a larger 1008-length AEV with
  wider per-element networks (see the paragraph after the table). Preset:
  :meth:`~xnn.dnn.models.ani.ANI.ani2x`.

ANI-1 and ANI-1x differ in both the descriptor geometry *and* the network body
(ANI-1ccx shares the ``ani1x`` column, with its own coupled-cluster self
energies):

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Setting
     - ``ani1`` (ANI-1)
     - ``ani1x`` (ANI-1x)
   * - Radial cutoff (Å)
     - 4.6
     - 5.2
   * - Angular cutoff (Å)
     - 3.1
     - 3.5
   * - Radial shifts / angular radial shifts
     - 32 / 8
     - 16 / 4
   * - Angular ``zeta``
     - 8
     - 32
   * - AEV length
     - 768
     - 384
   * - Network widths
     - uniform ``768:128:128:64:1``
     - per-element (H ``160:128:96``, C ``144:112:96``, N/O ``128:112:96``)
   * - Activation
     - Gaussian
     - ``CELU``
   * - Self atomic energies
     - none by default
     - torchani ANI-1x SAE (``atomic_energies="torchani"``)

Note that ANI-1x uses the *larger* cutoff but the *shorter* AEV: the
active-learning data lets it do more with a leaner descriptor.

ANI-2x uses a different grid again, sized for its seven elements: a 1008-length
AEV built from a 5.1 Å radial cutoff (16 shifts) and a 3.5 Å angular cutoff
(8 radial x 4 angular shifts), both shift grids starting at 0.8 Å, feeding
wider per-element networks (H ``256:192:160``, C ``224:192:160``, N/O
``192:160:128``, S/F/Cl ``160:128:96``) with the ``CELU`` activation and
wB97X/6-31G* self atomic energies. Its default species set is the seven-element
``[1, 6, 7, 8, 16, 9, 17]`` (H, C, N, O, S, F, Cl, in torchani's order), so
:meth:`~xnn.dnn.models.ani.ANI.ani2x` needs no ``species`` argument.

Select a preset in Python:

.. code-block:: python

   from xnn.dnn.models.ani import ANI

   ani1    = ANI.ani1(species=[1, 6, 7, 8])                        # original ANI-1
   ani1x   = ANI.ani1x(species=[1, 6, 7, 8], atomic_energies="torchani")  # ANI-1x
   ani1ccx = ANI.ani1ccx(species=[1, 6, 7, 8])                     # ANI-1ccx (CC)
   ani2x   = ANI.ani2x()                                           # ANI-2x (7 elem)

...or from a config file with the ``preset`` key, which
:meth:`~xnn.dnn.models.ani.ANI.from_config` routes to the matching classmethod
(accepts ``"ani-1"``/``"ani1"``, ``"ani-1x"``/``"ani1x"``,
``"ani-1ccx"``/``"ani1ccx"``, and ``"ani-2x"``/``"ani2x"``):

.. code-block:: yaml

   model:
     name: ani
     preset: ani-1x        # or ani-1 / ani-1ccx / ani-2x

Any key under ``model`` that is not a core config field (like ``preset``) is
collected into ``model.extra`` and forwarded to
:meth:`~xnn.dnn.models.ani.ANI.from_config`. Omit ``preset`` to build a bare
``ANI`` from the individual keys below instead.

.. note::

   The preset fixes the model *architecture*; the matching training data lives
   in the hub. Each preset has its own dataset builder: the original ANI-1 set
   as ``load_dataset("ani1")``, the active-learning ANI-1x set (with forces) as
   ``load_dataset("ani1x")``, the coupled-cluster ANI-1ccx set (energy-only)
   as ``load_dataset("ani1ccx")``, and the seven-element ANI-2x set (energies
   and forces for H/C/N/O/S/F/Cl) as ``load_dataset("ani2x")``. So ``ani-1`` +
   ``load_dataset("ani1")``, ``ani-1x`` + ``load_dataset("ani1x")``,
   ``ani-1ccx`` + ``load_dataset("ani1ccx")``, and ``ani-2x`` +
   ``load_dataset("ani2x")`` each reproduce a published model end-to-end
   (the published ANI-1ccx was *transfer-learned*: pre-trained on ANI-1x DFT
   data, then fine-tuned on the ANI-1ccx coupled-cluster energies). To load
   torchani's **pretrained** ANI-1x/ANI-1ccx/ANI-2x weights instead of
   training, see :ref:`fidelity`.

Key options (for the bare constructor, when not using a preset): ``species``
([1, 6, 7, 8]), ``radial_cutoff`` (5.2), ``angular_cutoff`` (3.5), ``hidden``
(128, 128, 64), ``activation`` (``"celu"``), ``atomic_energies``, ``aev_kwargs``
(symmetry-function grids, ``radial_prefactor``, ``angular_cos_factor``).

PhysNet (``dnn``)
=================
:class:`xnn.dnn.models.physnet.PhysNet`: message-passing HDNN with
explicit physics (Unke & Meuwly 2019): distance-based attention masks over
an exponential-Gaussian radial basis, pre-activation residual blocks,
per-module output heads predicting atomic energies *and* partial charges,
switched/shielded electrostatics of the corrected charges, and Grimme
D3(BJ) dispersion (tables included, coefficients learnable). A faithful
pure-PyTorch translation of the original TensorFlow
`MMunibas/PhysNet <https://github.com/MMunibas/PhysNet>`_ (see
:ref:`fidelity`); no species list needed, as elements up to Z = 94 are
embedded directly. ``forward`` additionally returns ``"charges"``,
``"dipole"``, and the ``"nh_loss"`` regularizer.

Key options: ``cutoff`` (10.0), the short-range ``sr_cut``, ``lr_cutoff``
(None), the long-range cutoff for electrostatics/dispersion (also the
neighbor-list radius when set), ``n_features`` (128), ``n_rbf`` (64),
``num_blocks`` = ``n_interactions`` (5), ``num_residual_atomic`` (2),
``num_residual_interaction`` (3), ``num_residual_output`` (1),
``use_electrostatics`` (True), ``use_dispersion`` (True),
``s6/s8/a1/a2`` (None = learnable), and ``species`` +
``atomic_energies``/``atomic_scales``, loaded into the per-element
``Eshift``/``Escale`` tables.

BAMBOO (``hybrid``)
===================
:class:`xnn.hybrid.models.bamboo.BAMBOO`: a graph equivariant transformer
with a physics energy split (Gong *et al.* 2024). Each message-passing layer
is a multi-head QKV attention on the neighbour graph that couples a scalar and
a Cartesian **vector** node channel (so equivariance comes from vectors, not
spherical harmonics; no e3nn), and the atomic energy is split into a
semi-local neural-network term, a **charge-equilibrium electrostatic** term
(a per-atom electronegativity/hardness energy plus a damped Coulomb summed over
*all* pairs, so it is genuinely long-range), and an optional D3(CSO) dispersion
term. ``forward`` additionally returns ``"charges"`` (per-atom partial charges,
conserved to the total charge), ``"dipole"``, and the component energies
``"energy_nn"`` / ``"energy_elec"``. Faithful to
`bytedance/bamboo <https://github.com/bytedance/bamboo>`_, with directly
transplantable weights (see :ref:`fidelity`). BAMBOO works in kcal/mol and Å,
and embeds elements directly by atomic number (no species list required).

The shared transformer pieces live in :mod:`xnn.transformer`
(:class:`~xnn.transformer.featurizers.ExpNormalSmearing` radial basis,
:class:`~xnn.transformer.attention.EdgeMultiheadAttention`) so future
attention-based models can reuse them.

Key options (defaults in parentheses): ``cutoff`` (5.0), the semi-local GET
cutoff, ``n_features`` (64), the ``dim`` node width (divisible by
``num_heads``), ``n_interactions`` (3), the GET layers (``n_layers``, ≥ 2),
``n_rbf`` (32), ``num_heads`` (16), ``charge_ub`` (2.0), the ``tanh`` bound on
the partial charge, ``charge_mlp_layers`` / ``energy_mlp_layers`` (2),
``n_elements`` (87), ``act_fn`` ("silu") / ``attn_act_fn`` ("gelu"),
``use_electrostatics`` (True), ``coul_damping_beta`` (18.7) /
``coul_damping_r0`` (2.2), ``use_dispersion`` (False) for the optional
D3(CSO), and ``disp_cutoff`` (10.0).

.. note::

   xnn returns the **full conservative force** ``-dE/dr`` uniformly via
   :class:`~xnn.common.models.outputs.ForceStressOutput`. The original BAMBOO
   instead reports ``nn_forces + coul_forces`` (charges held fixed) and
   regularises the charge–position-derivative ``qeq_force`` toward zero during
   training; the xnn force equals the upstream ``forces + qeq_force`` to
   machine precision (see :ref:`fidelity`).

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
dispersion), ``remove_self_interaction`` (False). Non-periodic structures use
the equivalent real-space direct sum; forces and stress flow through
:class:`~xnn.common.models.outputs.ForceStressOutput` unchanged. The outputs
gain ``"energy_sr"``, ``"energy_lr"`` and ``"latent_charges"``.

Forces and stress
=================
Wrap any model to get autograd forces and stress:

.. code-block:: python

   from xnn.common.models import ForceStressOutput

   model = ForceStressOutput(base_model, compute_forces=True, compute_stress=True)
   out = model(graph)   # adds "forces" (N, 3) and "stress" (B, 3, 3)

The stress is obtained by differentiating with respect to a symmetric
strain, so it is available for any model on periodic data. The
:class:`~xnn.common.train.trainer.Trainer` applies this wrapper
automatically, enabling the force/stress heads when the corresponding loss
weights are nonzero.

TorchScript deployment
======================
SchNet, NequIP, MACE, and Allegro additionally expose a scriptable
``node_energy(atomic_numbers, edge_index, edge_vec)`` core, which makes them
exportable to TorchScript and LAMMPS; see :ref:`deployment`. CACE provides
the same ``node_energy`` tensor core but is not TorchScript-exportable
(neither is the original CACE, which has no LAMMPS interface); it deploys
through the ASE calculator, as does PhysNet (whose original is a TF1 graph
driven through an ASE calculator as well). BAMBOO likewise deploys through the
ASE calculator (its all-pairs electrostatics and charge-equilibrium physics
match the paper's cluster-training setup; the original couples to LAMMPS
through a separate Ewald interface).
