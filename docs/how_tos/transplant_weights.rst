.. _howto-transplant:

**************************************
Transplant Weights from Upstream Codes
**************************************

Every xnn model with a reference implementation loads that code's weights and
reproduces its energies and forces to round-off. That is how the
implementations are verified (:ref:`fidelity`), and it is how a model trained
elsewhere comes into xnn, or goes back.

Published checkpoints need no transplant
========================================
The MACE and AIMNet2 foundation models convert in one call, and torchani's
ANI ensembles load through presets:

.. code-block:: python

   from xnn.common.models import from_pretrained
   from xnn.dnn.models import ANI

   mace = from_pretrained("mace-off23-small")
   aim = from_pretrained("aimnet2")
   ani = ANI.ani2x()

See :ref:`howto-pretrained-models`. A RuNNer model directory (``input.nn``
with the weights and scaling files of a 2G, 3G or 4G HDNNP) loads directly:

.. code-block:: python

   from xnn.dnn.common.runner import load_runner_model

   model = load_runner_model("runner_model/")

The pattern for anything else
=============================
1. Build the xnn model with the upstream hyperparameters (cutoff, angular
   order, channels, layers, radial basis, average neighbors, per-species
   energies and scales).
2. Rename the upstream state dict to the xnn parameter names.
3. Load it and compare on a batch:

.. code-block:: python

   missing, unexpected = model.load_state_dict(mapped_state, strict=False)
   assert not missing and not unexpected
   torch.testing.assert_close(model(batch)["energy"], reference_energy)

The same map applied in reverse loads xnn-trained weights into the reference
code. Each ``examples/fidelity_checks/<model>_verification`` notebook ends
with the transplant for its model; the notes per model:

- **MACE, BAMBOO, CACE**: a plain name map. CACE initializes some layers
  lazily, so run one upstream forward pass before reading its state dict.
- **NequIP, Allegro**: parameter names already match; the state dict loads
  nearly as is.
- **PhysNet**: crosses frameworks; each TensorFlow variable is copied as a
  NumPy array. ``dnn/physnet/physnet_argon_density_md`` shows the reverse
  direction, xnn weights back into the original graph.
- **DimeNet, DimeNet++**: crosses frameworks like PhysNet. The reference
  code folds basis constants and an angle convention into its weights, so
  the map rescales the basis layers and the model takes
  ``reference_basis=True``; ``tests/dimenet_tf_parity.py`` holds the map and
  applies it to random weights and to the published DimeNet++ QM9 model.
- **LES**: the upstream latent-charge head and Ewald settings map into
  ``LatentEwald(CACE)``.
- **SchNet**: a clean-room build from the paper, verified against an
  independent NumPy implementation; nothing to transplant.
- **ReaxFF, OPLS, DREIDING**: no weights. Their parameters are ``.frc``
  files, read directly and written back with ``export_library()``
  (:ref:`howto-forcefield-files`).

The test suite repeats the parity check whenever the reference package is
installed.
