.. _howto-pretrained-models:

*****************************
Load a Pre-trained Model
*****************************

``from_pretrained()`` loads any pre-trained potential: the MACE and AIMNet2
foundation models, models trained with xnn, Zenodo uploads and local
checkpoints all go through the same call and the same cache.
``examples/models/pretrained_models_tutorial.ipynb`` is the runnable
version of this page.

See what is available
=====================

.. code-block:: python

   from xnn.common.models import list_models, model_card

   list_models()                      # registered and cached models
   list_models(tag="organic")         # ['mace-off23-large', 'mace-off23-medium', 'mace-off23-small']
   model_card("mace-off23-small")     # source, license, citation, cutoff, elements, heads

.. code-block:: bash

   xnn models list --cached
   xnn models info mace-mh-0

Load a model
============

.. code-block:: python

   from xnn.common.models import from_pretrained

   model = from_pretrained("mace-off23-small")                   # registered name
   model = from_pretrained("mace-mh-0", head="omat_pbe")         # one head of a multi-head model
   model = from_pretrained("aimnet2")                            # AIMNet2 with its D3 term
   model = from_pretrained("doi:10.5281/zenodo.18957344",        # Zenodo DOI
                           filename="mace_csfapbbri_al_5_1_stagetwo.model")
   model = from_pretrained("runs/exp/best.pt")                   # trainer checkpoint
   model = from_pretrained("path/to/model_dir")                  # portable model directory

The result is wrapped in :class:`~xnn.common.models.outputs.ForceStressOutput`,
in eval mode and in the checkpoint's dtype. Useful options:

.. code-block:: python

   from_pretrained(name, wrap=False)              # the bare potential
   from_pretrained(name, dtype=torch.float64)     # cast the weights
   from_pretrained(name, device="cuda")           # move it
   from_pretrained(name, compute_stress=True)     # add the stress head
   from_pretrained(name, dispersion="d4")         # add D3 / D4 to a plain model
   from_pretrained(name, use_fast=True)           # fused GPU kernels (see fast paths)

:func:`~xnn.common.models.hub.load_pretrained` returns the model together
with its config, card and cutoff. For ASE:

.. code-block:: python

   from xnn.common.deploy import XNNCalculator

   atoms.calc = XNNCalculator.from_pretrained("mace-mp-0-medium")

The cache
=========
Each model is cached once as a portable directory with ``card.json``,
``config.yaml`` and ``model.pt`` (no pickled objects, no absolute paths).
A foreign format such as a ``mace-torch`` file is converted on the first
load; later loads need neither the network nor the upstream package.

.. code-block:: python

   from_pretrained("mace-off23-small", cache_dir="/scratch/me/xnn-models")

Without ``cache_dir`` the ``XNN_MODELS`` variable applies, then
``$XNN_CACHE/models``, then ``models/`` in a source checkout
(``~/.cache/xnn/models`` for an installed package). ``XNN_OFFLINE=1`` or
``local_files_only=True`` forbids network access; ``force_download=True``
fetches again. Copy a cached directory to another machine and it loads
there by name, offline.

Share your own models
=====================
Write a model in the same layout, then upload the directory (or the zip) to
Zenodo:

.. code-block:: python

   trainer.save_pretrained("my-model", description="...", license="MIT")

.. code-block:: bash

   xnn models pack runs/exp/best.pt my-model --description "..." --license MIT --archive

``from_pretrained("doi:10.5281/zenodo.<id>")`` then works for everyone. A
short name is one registration away:

.. code-block:: python

   from xnn.common.models import register_pretrained

   register_pretrained(name="lab-water", doi="10.5281/zenodo.<id>", license="CC-BY-4.0")

Names work everywhere a checkpoint does: ``xnn export --ckpt``, ``xnn mdi
--ckpt``, the ``checkpoint`` of a benchmark entry, and ``model.pretrained``
in a fine-tuning config.
