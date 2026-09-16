.. _howto-upstream-configs:

***********************************
Reuse Upstream MACE/NequIP Configs
***********************************

If you already have a YAML config written for the upstream `MACE CLI
<https://github.com/ACEsuit/mace>`_ or `NequIP <https://github.com/mir-group/nequip>`_,
you can copy its model keys into an xnn config verbatim. A per-model
key-translation registry (:mod:`xnn.common.config.translate`) rewrites the
foreign spellings to the xnn canonical names at config-load time.

For example, these MACE-CLI spellings are understood directly:

.. code-block:: yaml

   model:
     name: mace
     r_max: 4.0                 # -> cutoff
     num_radial_basis: 8        # -> n_rbf
     atomic_numbers: [18]       # -> species
     E0s: {18: -0.05}           # -> atomic_energies

NequIP YAML spellings work the same way:

.. code-block:: yaml

   model:
     name: nequip
     r_max: 4.0                 # -> cutoff
     num_layers: 3              # -> n_interactions
     num_features: 32           # -> n_features
     num_basis: 8               # -> n_rbf
     chemical_symbols: [Ar]     # -> species

as do the original CACE constructor spellings
(``BingqingCheng/cace``'s ``Cace(...)`` kwargs):

.. code-block:: yaml

   model:
     name: cace
     zs: [18]                   # -> species
     num_message_passing: 1     # -> n_interactions
     type_message_passing: ["M", "Ar", "Bchi"]  # -> message_types

and the schnetpack SchNet spellings (key names only; see :ref:`fidelity`):

.. code-block:: yaml

   model:
     name: schnet
     n_atom_basis: 64           # -> n_features
     n_gaussians: 25            # -> n_rbf
     atomref: {18: -0.05}       # -> atomic_energies

If both an upstream spelling and the xnn canonical name are given, the xnn
spelling wins. Values are also coerced: ``E0s``-style per-species energies can
be a list, a ``{Z: E0}`` mapping, or a string, and ``species`` accepts the
equivalent forms (see :mod:`xnn.common.config.coerce`).

Extending the translation table
===============================
To support another upstream code's spellings, register a translation table
once at import time:

.. code-block:: python

   from xnn.common.config.translate import register_key_translation

   register_key_translation(
       "mymodel",
       {
           "r_cut": "cutoff",
           "n_bessel": "n_rbf",
       },
   )

The table is applied whenever a config with ``model.name == "mymodel"`` is
loaded through any of the frontends (YAML, argparse, Hydra).
