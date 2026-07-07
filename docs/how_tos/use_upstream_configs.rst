.. _howto-upstream-configs:

***********************************
Reuse Upstream MACE/NequIP Configs
***********************************

If you already have a YAML config written for the upstream `MACE CLI
<https://github.com/ACEsuit/mace>`_ or `NequIP <https://github.com/mir-group/nequip>`_,
you can copy its model keys into an xnns config verbatim. A per-model
key-translation registry (:mod:`xnns.common.config.translate`) rewrites the
foreign spellings to the xnns canonical names at config-load time.

For example, these MACE-CLI spellings are understood directly:

.. code-block:: yaml

   model:
     name: mace
     r_max: 4.0                 # -> cutoff
     num_radial_basis: 8        # -> n_rbf
     atomic_numbers: [18]       # -> species
     E0s: {18: -0.05}           # -> atomic_energies

as are NequIP spellings such as ``num_layers`` (→ ``n_layers``) and the
original CACE constructor spellings (``zs`` → ``species``,
``num_message_passing`` → ``n_interactions``, ``type_message_passing`` →
``message_types``). If both an upstream spelling and the xnns canonical name
are given, the xnns spelling wins.

Values are also coerced: ``E0s``-style per-species energies can be a list, a
``{Z: E0}`` mapping, or a string, and ``species`` accepts the equivalent
forms (see :mod:`xnns.common.config.coerce`).

Extending the translation table
===============================
To support another upstream code's spellings, register a translation table
once at import time:

.. code-block:: python

   from xnns.common.config.translate import register_key_translation

   register_key_translation(
       "mymodel",
       {
           "r_cut": "cutoff",
           "n_bessel": "n_rbf",
       },
   )

The table is applied whenever a config with ``model.name == "mymodel"`` is
loaded through any of the frontends (YAML, argparse, Hydra).
