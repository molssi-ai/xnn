.. _howto-pretrained-models:

*****************************
Load a Pre-trained Model
*****************************

The ``from_pretrained()`` function loads the model artifacts: the original
foundation models pre-trained by their main developer teams (e.g., MACE and
AIMNet2), models trained with xnn, Zenodo uploads and local checkpoints, all can
be loaded through the same function call and the same cache. The
``notebooks/examples/common/models/pretrained_models_tutorial.ipynb`` notebook provides a
runnable version of this page.

.. note::

   **xnn does not ship any pre-trained models or third-party weights.**
   Published models are downloaded from their developers' own release
   locations when you first ask for one, and they stay under their original
   licenses.

Inspect the model registry
==========================

.. code-block:: python

   from xnn.common.models import list_models, model_card

   # List all registered and cached models
   list_models()

   # ['mace-off23-large', 'mace-off23-medium', 'mace-off23-small']
   list_models(tag="organic")

   # Source, license, citation, cutoff, elements, heads
   model_card("mace-off23-small")

The ``xnn models`` command does the same from a shell:

.. code-block:: bash

   # Models that load without a download
   xnn models list --cached

   # The model card
   xnn models info mace-mh-0

Load a model
============

.. code-block:: python

   from xnn.common.models import from_pretrained

   model = from_pretrained("mace-off23-small")                   # Registered name
   model = from_pretrained("mace-mh-0", head="omat_pbe")         # One head of a multi-head model
   model = from_pretrained("aimnet2")                            # AIMNet2 with its D3 term
   model = from_pretrained("doi:10.5281/zenodo.18957344",        # Zenodo DOI
                           filename="mace_csfapbbri_al_5_1_stagetwo.model")
   model = from_pretrained("runs/exp/best.pt")                   # Trainer checkpoint
   model = from_pretrained("path/to/model_dir")                  # Portable model directory

The ``from_pretrained`` function returns a model object wrapped in the
:class:`~xnn.common.models.outputs.ForceStressOutput` class. This object is
already in the ``eval()`` mode and in the model checkpoint's ``dtype``. 

A checkpoint rebuilds the model it was trained as, even when a later xnn
version changes that model's defaults: its config records every option of the
model, the defaults it used included.

Here are some additional options for customizing the load behavior:

.. code-block:: python

   from_pretrained(name, wrap=False)              # The bare potential
   from_pretrained(name, dtype=torch.float64)     # Cast the weights
   from_pretrained(name, device="cuda")           # Move it
   from_pretrained(name, compute_stress=True)     # Add the stress head
   from_pretrained(name, dispersion="d4")         # Add D3 / D4 to a plain model
   from_pretrained(name, use_fast=True)           # Fused GPU kernels

The :func:`~xnn.common.models.hub.load_pretrained` function behaves similar to
``from_pretrained()`` but returns a
:class:`~xnn.common.models.hub.loader.PretrainedModel` which also provides its
config, model card and other information. For ASE, you can wrap the model in an
:class:`~xnn.common.deploy.XNNCalculator` using:

.. code-block:: python

   from xnn.common.deploy import XNNCalculator

   atoms.calc = XNNCalculator.from_pretrained("mace-mp-0-medium")

The cache
=========
Each model is cached once into a portable directory with ``card.json``,
``config.yaml`` and ``model.pt`` (no pickled objects, no absolute paths). An
external format such as a ``mace-torch`` file is automatically converted to an
xnn-native format on the first load and subsequent loads will be carried out
offline.

.. code-block:: python

   from_pretrained("mace-off23-small", cache_dir="/scratch/me/xnn-models")

Without ``cache_dir`` the ``XNN_MODELS`` environment variable sets the
directory, then ``$XNN_CACHE/models``, and then ``models/`` in a source checkout
(``~/.cache/xnn/models`` for an installed package).

The ``XNN_OFFLINE=1`` or ``local_files_only=True`` variables forbid xnn to
access the network. Setting ``force_download=True`` forces xnn to fetch the
model over the network again.

Cached directories are designed to be portable and self-sufficient. You can copy
them to another machine and they can be loaded there by using the folder name,
in offline mode.

Share your own models
=====================
You can save a model using ``save_pretrained()`` and then upload the resulting
directory (or the compressed version of it) to Zenodo from either the Python API

.. code-block:: python

   trainer.save_pretrained("my-model", description="...", license="MIT")

or the command line:

.. code-block:: bash

   xnn models pack runs/exp/best.pt my-model --description "..." --license MIT --archive

Once the portable directory becomes accessible on Zenodo, the
``from_pretrained("doi:10.5281/zenodo.<id>")`` function call then works for
everyone with a network access. You can give the model a short name to register
it in your model registry for easier access:

.. code-block:: python

   from xnn.common.models import register_pretrained

   register_pretrained(name="lab-water", doi="10.5281/zenodo.<id>", license="CC-BY-4.0")

This short name will work like a shortcut for accessing the model. Registered
names work everywhere a checkpoint could otherwise be normally used. For
example, in command line ``xnn export --ckpt``, and ``xnn mdi --ckpt``, as a
``checkpoint`` in a benchmarking config file entry, and as a
``model.pretrained`` entry in a fine-tuning configuration file.
