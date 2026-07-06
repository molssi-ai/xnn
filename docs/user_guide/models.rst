.. _models:

******
Models
******

All models subclass
:class:`~xnns.common.models.base.InteratomicPotential`: they take an
:class:`~xnns.common.data.atomic_data.AtomicGraph` and return a dictionary
with ``node_energy`` (per atom) and ``energy`` (per structure). Forces and
stress are added uniformly by
:class:`~xnns.common.models.outputs.ForceStressOutput` — no model implements
them itself.

Models are registered by name, so they can be built from any config
frontend:

.. code-block:: python

   from xnns.common.models import available_models, build_model

   available_models()          # ['schnet', 'hdnnp', 'ani', 'nequip', 'mace', 'allegro']
   model = build_model(cfg.model)   # dispatches to <Model>.from_config(cfg.model)

Each model can also be constructed directly; the constructor arguments below
double as the keys accepted in ``model.extra`` of a config file.

.. note::

   The GNN models (NequIP, MACE, Allegro) require the ``gnn`` extra
   (``e3nn``); they register themselves when ``xnns.gnn`` is importable.

MACE (``gnn``)
==============
:class:`xnns.gnn.models.mace.MACE` — higher-order equivariant message
passing with the learned symmetric contraction of Batatia *et al.*
Faithful to `ACEsuit/mace <https://github.com/ACEsuit/mace>`_ (see
:ref:`fidelity`).

Key options (defaults in parentheses): ``species``, ``cutoff`` (4.0),
``max_ell`` (3) — spherical-harmonic order of the edges, ``max_L`` (0) —
order of the message irreps, ``num_channels`` (32), ``n_rbf`` (8),
``num_interactions`` (2) — fully flexible T = 0..N, ``correlation`` (3) —
order of the symmetric contraction, ``MLP_irreps`` ("16x0e"),
``radial_MLP``, ``interaction`` ("RealAgnosticResidualInteractionBlock"),
``interaction_first``, ``gate`` ("silu"), ``avg_num_neighbors`` (1.0),
``hidden_irreps``, ``num_cutoff_basis`` (5), ``radial_type`` ("bessel"),
``distance_transform``, ``pair_repulsion`` (False) — adds ZBL core
repulsion, ``atomic_energies`` — per-species reference energies (E0s).

NequIP (``gnn``)
================
:class:`xnns.gnn.models.nequip.NequIP` — E(3)-equivariant message passing
with gated nonlinearities (Batzner *et al.*). Faithful to
`mir-group/nequip <https://github.com/mir-group/nequip>`_, with directly
transplantable state dicts.

Key options: ``species``, ``cutoff`` (4.0), ``l_max`` (2), ``parity``
(True), ``n_rbf`` (8), ``n_layers`` (3), ``num_features`` (32),
``invariant_layers`` (2), ``invariant_neurons`` (64),
``avg_num_neighbors``, ``use_sc`` (True) — self-connection, ``resnet``
(False), ``nonlinearity_scalars`` / ``nonlinearity_gates``,
``num_polynomial_cutoff`` (6), ``trainable_rbf`` (True),
``conv_to_output_hidden``, ``atomic_energies``, ``atomic_scales`` —
per-species energy scale/shift.

Allegro (``gnn``)
=================
:class:`xnns.gnn.models.allegro.Allegro` — strictly local equivariant
many-body potential (Musaelian *et al.*), without message passing between
atoms. Faithful to `mir-group/allegro <https://github.com/mir-group/allegro>`_
v0.3.0 (``uuulin`` mode), with directly transplantable state dicts.

Key options: ``species``, ``cutoff`` (4.0), ``l_max`` (1), ``parity``
("o3_full"), ``n_rbf``, ``num_layers``, ``num_tensor_features``,
``two_body_latent`` / ``latent`` / ``env_embed`` / ``edge_eng`` — the MLP
widths, ``initial_scalar_embedding_dim``, ``avg_num_neighbors``,
``latent_resnet`` (True), ``num_polynomial_cutoff`` (6), ``trainable_rbf``
(True), ``atomic_energies``, ``atomic_scales``.

SchNet (``cnn``)
================
:class:`xnns.cnn.models.schnet.SchNet` — continuous-filter convolutions over
a Gaussian radial basis (Schütt *et al.*).

Key options: ``n_features`` (128), ``n_interactions`` (3), ``n_rbf`` (50),
``cutoff`` (5.0).

HDNNP (``dnn``)
===============
:class:`xnns.dnn.models.hdnnp.HDNNP` — Behler–Parrinello high-dimensional
neural network potential: radial (G2) symmetry-function descriptors feeding
one MLP per element.

Key options: ``species``, ``cutoff`` (6.0), ``etas`` (0.05, 0.5, 2.0, 8.0),
``rs`` (0.0,), ``hidden`` (64, 64).

ANI (``dnn``)
=============
:class:`xnns.dnn.models.ani.ANI` — ANI-style potential over atomic
environment vectors (radial + angular AEV) with per-element networks.

Key options: ``species``, ``radial_cutoff`` (5.2), ``angular_cutoff``
(3.5), ``hidden`` (128, 96, 64), ``aev_kwargs``.

Forces and stress
=================
Wrap any model to get autograd forces and stress:

.. code-block:: python

   from xnns.common.models import ForceStressOutput

   model = ForceStressOutput(base_model, compute_forces=True, compute_stress=True)
   out = model(graph)   # adds "forces" (N, 3) and "stress" (B, 3, 3)

The stress is obtained by differentiating with respect to a symmetric
strain, so it is available for any model on periodic data. The
:class:`~xnns.common.train.trainer.Trainer` applies this wrapper
automatically, enabling the force/stress heads when the corresponding loss
weights are nonzero.

TorchScript deployment
======================
SchNet, NequIP, MACE, and Allegro additionally expose a scriptable
``node_energy(atomic_numbers, edge_index, edge_vec)`` core, which makes them
exportable to TorchScript and LAMMPS — see :ref:`deployment`.
