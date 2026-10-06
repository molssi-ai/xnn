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

   available_models()
   # ['aimnet2', 'allegro', 'ani', 'bamboo', 'cace', 'd3', 'd4', 'dreiding',
   #  'hdnnp', 'mace', 'nequip', 'opls', 'physnet', 'reaxff', 'schnet']

   model = ForceStressOutput(build_model(cfg.model), compute_stress=True)
   out = model(graph)      # energy, node_energy, forces (N, 3), stress (B, 3, 3)

Every model can also be constructed directly; its constructor arguments are
the keys accepted under ``model`` in a config file. Pre-trained models load
with ``from_pretrained()`` (:ref:`howto-pretrained-models`). NequIP, MACE
and Allegro need the ``gnn`` extra (e3nn); every other model is plain
PyTorch.

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

Options: ``species``, ``cutoff`` (5.0), ``n_features`` (16), ``n_rbf`` (16),
``n_interactions`` (3), ``rbf_start`` (0.8), ``hidden``,
``n_vector_combinations`` (12), ``aim_size`` (256), ``readout_hidden``
([128, 128]), ``charge_channels`` (1; 2 for open shells), ``coulomb``
("simple" | "dsf" | "ewald" | "pme" | null), ``coulomb_sr_cutoff`` (4.6),
``lr_cutoff`` (15.0), ``dsf_alpha`` (0.2), ``ewald_accuracy`` (1e-6),
``pme_spline_order`` (4), ``atomic_energies``. For a periodic structure pass
``model_options={"coulomb": "dsf"}`` (or ``"ewald"`` / ``"pme"``) to
``from_pretrained``.

SchNet (``cnn``)
================
:class:`~xnn.cnn.models.schnet.SchNet`: continuous-filter convolutions over
a Gaussian basis with shifted-softplus activations (Schütt *et al.* 2017).
A clean-room build of the paper, verified against an independent
implementation of its equations. The defaults are the paper architecture.

Options: ``n_features`` (64), ``n_interactions`` (3), ``n_rbf`` (301),
``cutoff`` (30.0), ``gamma`` (10.0), ``cutoff_fn`` (None; ``"cosine"`` for a
smooth finite cutoff), ``energy_shift`` (0.0), ``energy_scale`` (1.0),
``species`` + ``atomic_energies``.

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
:class:`~xnn.ffnn.models.reaxff.ReaxFF`: the bond-order reactive force
field (van Duin *et al.* 2001) and its neural variant ReaxFF-nn in one
model. Bond orders come from distances, every valence term is written in
them so it vanishes smoothly as bonds break, and charges are equilibrated
at every geometry by a differentiable EEM solve. The whole functional form
is differentiable, so any parameter group can be refit by gradient descent.
Parameters come from a published ``.frc`` field (a dozen ship with xnn) or a
ReaxFF-nn JSON library, and ``export_library()`` writes a trained field back
out. Returns ``charges`` and the per-term energies (``e_bond``,
``e_angle``, ``e_vdw``, ...).

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
