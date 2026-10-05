.. _howto-transplant:

**************************************
Transplant Weights from Upstream Codes
**************************************

Every xnn model with a published reference implementation loads that code's
weights and reproduces its energies and forces to round-off error. This is how the
implementations are verified (:ref:`fidelity`), and it is also how you bring
a model trained elsewhere into xnn, or take an xnn model back to the
reference code. SchNet is the exception: it is a clean-room build from the
papers, checked against an independent NumPy implementation of the
equations, so there is nothing to transplant.

Published checkpoints need no transplant
========================================
The pretrained MACE and AIMNet2 models convert in one call, through the model
hub or the classes themselves:

.. code-block:: python

   from xnn.common.models import from_pretrained
   from xnn.gnn.models import MACE, AIMNet2

   mace = from_pretrained("mace-off23-small")
   mace = MACE.from_foundation("mace-mp-0-medium")
   aim = AIMNet2.from_foundation("aimnet2")

torchani's pretrained ANI ensembles load through the presets ``ANI.ani1x()``,
``ANI.ani1ccx()`` and ``ANI.ani2x()``. See :ref:`howto-pretrained-models`.

The pattern
===========
For any other upstream model:

1. Build the xnn model with the same hyperparameters as the upstream one
   (cutoff, angular order, channels, layers, radial basis size, average
   number of neighbours, per-species energies and scales).
2. Rename the upstream state dict to the xnn parameter names.
3. Load it and compare on a batch:

.. code-block:: python

   missing, unexpected = model.load_state_dict(mapped_state, strict=False)
   assert not missing and not unexpected
   torch.testing.assert_close(model(batch)["energy"], reference_energy)

The same mapping, applied in reverse, loads xnn-trained weights into the
reference code.

Per-model notes
===============

.. list-table::
   :header-rows: 1
   :widths: 14 56 30

   * - Model
     - Mapping
     - Verified in
   * - MACE
     - Block-by-block match, a plain name map. Agreement about 1e-15.
     - ``fidelity_checks/mace_verification``
   * - NequIP
     - Parameter names already match upstream; the state dict loads nearly
       as is. About 1e-16, including stress.
     - ``fidelity_checks/nequip_verification``
   * - Allegro
     - Names match, and the strided channel-mixing linears keep the upstream
       flat layout. About 1e-15.
     - ``fidelity_checks/allegro_verification``
   * - CACE
     - Plain name map. Upstream initializes some layers lazily, so run one
       forward pass on a sample batch before reading its state dict. About
       1e-16, molecular and periodic.
     - ``fidelity_checks/cace_verification``
   * - PhysNet
     - Crosses frameworks: each TensorFlow 1 variable is copied to or from
       the torch parameter as a NumPy array. About 1e-15 in float64.
     - ``fidelity_checks/physnet_verification``
   * - ANI
     - The presets above take torchani's weights, one network or the full
       ensemble. AEV about 1e-16; ensemble energies about 1e-8 Ha.
     - ``fidelity_checks/ani_verification``
   * - BAMBOO
     - Plain name map. xnn returns the conservative force, equal to upstream
       ``forces + qeq_force``. About 1e-15.
     - ``fidelity_checks/bamboo_verification``
   * - LES
     - Name map of the upstream latent-charge head and Ewald settings into
       ``LatentEwald(CACE)``. Kernels about 1e-16 in float64.
     - ``fidelity_checks/les_verification``
   * - AIMNet2
     - ``from_foundation`` converts all published members; float32
       round-off against the reference package.
     - ``fidelity_checks/aimnet2_verification``

The MD notebook ``dnn/physnet/physnet_argon_density_md`` shows the reverse
direction in use: xnn-trained PhysNet weights go back into the original
TensorFlow graph, and both engines run the same NVE trajectory in lock step.

Classical force fields
======================
ReaxFF, OPLS and DREIDING have no weights to transplant. Their parameters are
SEAMM ``.frc`` force-field files, which the classes read directly and write
back after training with ``export_library()``. See
:ref:`howto-forcefield-files`.

Tests
=====
The test suite checks parity with the upstream code whenever the reference
package is installed: ``test_nequip``, ``test_allegro``, ``test_cace``,
``test_les``, ``test_ani`` (the ``ani`` extra), ``test_physnet`` (TensorFlow
and ``PHYSNET_UPSTREAM_PATH``) and ``test_bamboo`` (``BAMBOO_UPSTREAM_PATH``).
