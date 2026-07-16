.. _howto-transplant:

**************************************
Transplant Weights from Upstream Codes
**************************************

Because the xnns NequIP and Allegro implementations use the upstream
parameter names and weight layouts, a state dict trained with the reference
code loads directly into the xnns model (and vice versa). The MACE
implementation reproduces upstream block-by-block, so whole models transplant
as well; the example notebooks do exactly this and reproduce upstream
energies and forces to ~1e-15/1e-16.

The pattern
===========
1. Build the xnns model with the *same architecture hyperparameters* as the
   upstream model (cutoff, ``l_max``/``max_ell``, channels, layers, radial
   basis size, ``avg_num_neighbors``, per-species energies/scales).
2. Map the upstream state dict onto the xnns parameter names.
3. ``load_state_dict`` and verify on a batch.

For NequIP the parameter names already match (the interaction block uses
upstream names), so step 2 is nearly a no-op. For Allegro the strided
channel-mixing linears keep the same flat weight layout as upstream, so
tensors copy over unchanged.

Worked examples
===============
The block-by-block notebooks perform full transplants and check every
intermediate tensor:

- ``examples/gnn/mace/recreate_mace_architecture.ipynb``: rebuilds the
  MACE architecture step by step in both ``mace-torch`` and xnns, transplants
  a whole model, and reproduces its energy and forces to ~1e-15.
- ``examples/fidelity_checks/nequip_verification.ipynb``: ends
  with a whole-model weight transplant (~1e-16 agreement).
- ``examples/fidelity_checks/allegro_verification.ipynb``: the
  same for Allegro (~1e-15).

Parity with upstream given identical weights is also enforced in the test
suite (``tests/test_nequip.py``, ``tests/test_allegro.py``) whenever the
reference packages are installed.
