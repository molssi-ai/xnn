.. _howto-transplant:

**************************************
Transplant Weights from Upstream Codes
**************************************

Every xnn model with a reference implementation can load that code's trained
weights and reproduce its energies and forces to round-off error. This is how
the xnn implementations are verified (:ref:`fidelity`), and how you can bring a
model trained elsewhere into xnn.

.. note::

   **xnn does not ship any pre-trained models or third-party weights.**
   Published models are downloaded from their developers' own release
   locations when you first ask for one, and they stay under their original
   licenses.

Published models are fetched at runtime
=======================================
``from_pretrained()`` downloads a published model from where its developers host
it. It then verifies the file's checksum, converts it to xnn's format on your
machine and caches the result:

.. code-block:: python

   from xnn.common.models import from_pretrained, model_card

   model = from_pretrained("mace-off23-small")    # Downloaded on first use, then cached
   card = model_card("mace-off23-small")
   print(card.url, card.license, card.citation)

Every model comes with a model card that records its provenance: the download
source, the license and the citation. The card is saved in the cache next to
the converted weights (``card.json``), so it travels with the model. Models
under a restrictive license, such as the Academic Software License of some MACE
models, print their terms when they are loaded. Check a model's card before
you use it, and cite the original work. See :ref:`howto-pretrained-models` for
full details on model hub.

Weights you already have
========================
xnn also reads checkpoints that you obtained yourself:

- **RuNNer HDNNPs.** A RuNNer model directory (``input.nn`` with its weights and
  scaling files) loads directly:

  .. code-block:: python

     from xnn.dnn.common.runner import load_runner_model

     model = load_runner_model("runner_model/")

- **SpookyNet.** A saved reference model loads with
  ``SpookyNet.from_reference_checkpoint(path)``.
- **ANI.** ``ANI.ani1x()``, ``ANI.ani1ccx()`` and ``ANI.ani2x()`` build the
  matching torchani architectures. They contain no trained weights. Transplant
  torchani's weights into them as described below.

Transplanting weights manually
==============================
For any other upstream model:

1. Build the xnn model with the upstream hyperparameters (cutoff, angular
   order, channels, layers, radial basis, average number of neighbors,
   per-species energies and scales).
2. Rename the upstream state dict to the xnn parameter names.
3. Load it, and check on a batch that the energies and forces match the
   reference code:

.. code-block:: python

   missing, unexpected = model.load_state_dict(mapped_state, strict=False)
   assert not missing and not unexpected                      # Every parameter was mapped
   torch.testing.assert_close(model(batch)["energy"], reference_energy)

The same mapping in reverse loads xnn-trained weights into the reference code.
Each ``notebooks/fidelity_checks/<family>/<model>/<model>_verification`` notebook
ends with the transplant for its model.
