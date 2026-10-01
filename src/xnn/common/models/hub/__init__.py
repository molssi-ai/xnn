"""Model hub: one-line ``from_pretrained()`` loading of pre-trained models.

One API for every pre-trained model, whatever its origin::

    from xnn.common.models import from_pretrained, list_models

    list_models()                                       # registry + cache
    model = from_pretrained("mace-off23-small")          # MACE foundation model
    model = from_pretrained("xnn-mace-argon")            # xnn-trained model
    model = from_pretrained("doi:10.5281/zenodo.18957344",
                            filename="mace_csfapbbri_al_5_1_stagetwo.model")
    model = from_pretrained("runs/exp/best.pt")          # local checkpoint

Downloads are MD5-verified and cached under ``cache_dir`` (by default
:func:`~xnn.common.models.hub.cache.default_model_cache_dir`), one
directory per model. Every cached model is stored as a portable directory
(``card.json``, ``config.yaml``, ``model.pt``) with no pickled objects and no
absolute paths, so a cache, or one model of it, can be copied to another
system and loaded there; foreign formats (``mace-torch``) are converted once
on download and need no upstream package afterwards. Write your own models in
the same layout with :func:`save_pretrained`, and add them to the registry
with :func:`register_pretrained` (or an entry in the packaged
``models.json``).
"""
from __future__ import annotations

from .cache import default_model_cache_dir
from .card import ModelCard
from .checkpoint import Checkpoint, build_potential, load_checkpoint, save_pretrained
from .formats import ModelFormat, register_format
from .loader import PretrainedModel, fetch_model, from_pretrained, load_pretrained
from .registry import list_models, model_card, register_pretrained

__all__ = [
    "from_pretrained",
    "load_pretrained",
    "fetch_model",
    "save_pretrained",
    "list_models",
    "model_card",
    "register_pretrained",
    "register_format",
    "load_checkpoint",
    "build_potential",
    "default_model_cache_dir",
    "ModelCard",
    "ModelFormat",
    "Checkpoint",
    "PretrainedModel",
]
