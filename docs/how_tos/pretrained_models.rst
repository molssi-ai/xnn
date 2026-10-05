.. _howto-pretrained-models:

*****************************
Load a Pre-trained Model
*****************************

The model hub loads any pre-trained potential with a one-line
``from_pretrained()`` call: the MACE and AIMNet2 foundation models,
models trained with xnn, models published on Zenodo, and your own
checkpoints all go through the same call, the same registry and the same
cache. See ``examples/models/pretrained_models_tutorial.ipynb`` for a
runnable walkthrough.

List what is available
======================

.. code-block:: python

   from xnn.common.models import list_models, model_card

   list_models()                          # every registered and cached model
   list_models(details=True)              # one row per model: format, cutoff, license, cache status
   list_models(tag="organic")             # ['mace-off23-large', 'mace-off23-medium', 'mace-off23-small']
   model_card("mace-off23-small")         # source, license, citation, cutoff, elements, heads

The registry is the ``models.json`` file shipped inside the package
(``src/xnn/common/models/hub/models.json``). ``list_models`` also shows every
model directory it finds in the cache, so models fetched by URL or DOI, and
directories copied in from another machine, are listed by name. From a shell:

.. code-block:: bash

   xnn models list --cached
   xnn models info mace-mh-0

Load a model
============

.. code-block:: python

   from xnn.common.models import from_pretrained

   model = from_pretrained("mace-off23-small")                 # registered name
   model = from_pretrained("mace-mh-0", head="omat_pbe")       # one head of a multi-head model
   model = from_pretrained("aimnet2")                          # alias of aimnet2-wb97m-d3-0, with its D3 term
   model = from_pretrained("doi:10.5281/zenodo.18957344",      # Zenodo DOI (or record link)
                           filename="mace_csfapbbri_al_5_1_stagetwo.model")
   model = from_pretrained("https://host/path/model.pt")       # plain URL
   model = from_pretrained("runs/exp/best.pt")                 # local trainer checkpoint
   model = from_pretrained("path/to/model_dir")                # local model directory

The result is the model wrapped in :class:`~xnn.common.models.ForceStressOutput`
(forces by autograd; pass ``compute_stress=True`` for stress), in eval mode
and in the checkpoint's dtype. ``wrap=False`` returns the bare potential,
``dtype=torch.float64`` changes the precision, ``device="cuda"`` moves it,
and ``dispersion=`` adds a D3/D4 term as for :ref:`deployment <howto-ase>`.
:func:`~xnn.common.models.hub.load_pretrained` returns the same model along
with its config, card and neighbor-list cutoff. For ASE:

.. code-block:: python

   from xnn.common.deploy import XNNCalculator

   atoms.calc = XNNCalculator.from_pretrained("mace-mp-0-medium")

Choose where models are cached
==============================

Every download lands in a cache directory, one subdirectory per model:

.. code-block:: python

   model = from_pretrained("mace-off23-small", cache_dir="/scratch/me/xnn-models")

Without ``cache_dir``, the ``XNN_MODELS`` environment variable applies, then
``$XNN_CACHE/models``, then the repository's ``models/`` directory in a source
checkout (``~/.cache/xnn/models`` for an installed package). Files are
MD5-verified (Zenodo supplies the digests), written atomically, and reused on
the next call. ``local_files_only=True``, or ``XNN_OFFLINE=1``, forbids network
access; ``force_download=True`` fetches again.

A foreign format such as a pickled ``mace-torch`` model is **converted once**:
the cache keeps the converted model, not the upstream file, so later loads
need neither the network nor ``mace-torch``. A multi-head checkpoint keeps its
upstream file in ``raw/`` so the other heads convert offline. Upstream files
already in ``mace-torch``'s own cache (``~/.cache/mace``) are reused.

Move models between machines
============================

A cached model is a portable directory:

.. code-block:: text

   mace-off23-small/
       card.json      the model card, with an MD5 per file
       config.yaml    the configuration that rebuilds the architecture
       model.pt       the weights, readable with torch.load(..., weights_only=True)

It holds no pickled Python objects and no absolute paths. Copy one directory,
or the whole cache, to another system and load it there by name with
``cache_dir`` (or ``XNN_MODELS``) pointing at the copy, offline. A config that
refers to files outside the directory, such as a ReaxFF ``ffield`` path or an
OPLS topology JSON, is the one exception: :func:`~xnn.common.models.hub.save_pretrained`
warns about it, and the fix is a packaged library name or inline data.

Share your own models
=====================

Write a model in the same layout:

.. code-block:: python

   from xnn.common.models import save_pretrained

   trainer.save_pretrained("my-model", description="...", license="MIT")     # after training
   save_pretrained("runs/exp/best.pt", "my-model", description="...")          # an existing checkpoint
   save_pretrained(model, "my-model", config=cfg)                              # a model object

or from a shell, with ``--archive`` for a zip ready to upload:

.. code-block:: bash

   xnn models pack runs/exp/best.pt my-model --description "..." --license MIT --archive

Upload the three files (or the zip) to Zenodo; ``from_pretrained("doi:10.5281/zenodo.<id>")``
then works for everyone. To give it a short name, register it:

.. code-block:: python

   from xnn.common.models import register_pretrained

   register_pretrained(name="lab-water", doi="10.5281/zenodo.<id>",
                       description="...", license="CC-BY-4.0")

or add the same entry to ``models.json`` so it ships with xnn.

Use a name anywhere a checkpoint goes
=====================================

Everything that loads a model resolves names through the hub:
``MDIEngine.from_checkpoint`` and ``xnn mdi --ckpt``, ``xnn export --ckpt``,
the ``checkpoint`` of a benchmark entry, ``Model.from_pretrained`` for the bare
potential of a given class (``MACE.from_pretrained("mace-off23-small")``), and
``MACE.from_foundation`` together with the config form
``model: {name: mace, foundation: mace-off23-small}``.
