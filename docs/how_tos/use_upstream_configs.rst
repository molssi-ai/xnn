.. _howto-upstream-configs:

***********************************
Reuse Upstream MACE/NequIP Configs
***********************************

Model keys copied from an upstream YAML work as they are. A per-model
translation table (:mod:`xnn.common.config.translate`) rewrites the foreign
spellings to the xnn names when the config loads:

.. code-block:: yaml

   model:                       # MACE CLI spellings
     name: mace
     r_max: 4.0                 # -> cutoff
     num_radial_basis: 8        # -> n_rbf
     atomic_numbers: [18]       # -> species
     E0s: {18: -0.05}           # -> atomic_energies

.. code-block:: yaml

   model:                       # NequIP spellings
     name: nequip
     r_max: 4.0                 # -> cutoff
     num_layers: 3              # -> n_interactions
     num_features: 32           # -> n_features
     chemical_symbols: [Ar]     # -> species

CACE's constructor keywords (``zs``, ``num_message_passing``,
``type_message_passing``) and schnetpack's SchNet keys (``n_atom_basis``,
``n_gaussians``, ``atomref``) translate the same way. When both spellings
appear, the xnn one wins. Values are coerced too: per-species energies can be
a list, a ``{Z: E0}`` mapping or a string, and ``species`` accepts symbols
or numbers.

Add a table
===========

.. code-block:: python

   from xnn.common.config.translate import register_key_translation

   register_key_translation("mymodel", {"r_cut": "cutoff", "n_bessel": "n_rbf"})

The table applies to every config with ``model.name == "mymodel"``, whatever
the frontend.
