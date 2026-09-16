"""Dataset hub: HuggingFace ``load_dataset()``-style one-liner loading.

Download and preprocess upstream atomistic datasets into xnn' native structure
dicts (optionally ready-to-train :class:`~xnn.common.data.dataset.AtomicDataset`
objects) with a single call::

    from xnn.common.data import load_dataset

    splits = load_dataset("rmd17", molecule="aspirin")          # {"train", "test"}
    train = load_dataset("rmd17", molecule="aspirin",
                         split="train", cutoff=5.0)             # AtomicDataset

Files are cached (and MD5-verified) under ``datasets/<name>/`` in the repo by
default; see :func:`~xnn.common.data.hub._download.default_cache_dir`. Add a new
dataset by subclassing :class:`DatasetBuilder` and calling
:func:`register_dataset`.
"""
from __future__ import annotations

from ._download import default_cache_dir
from .base import (
    DatasetBuilder,
    get_builder,
    list_datasets,
    load_dataset,
    register_dataset,
)

# Import builder modules for their registration side effects.
from . import rmd17  # noqa: F401
from . import lode_dimers  # noqa: F401
from . import ani1  # noqa: F401
from . import ani1x  # noqa: F401
from . import ani1ccx  # noqa: F401
from . import ani2x  # noqa: F401
from . import argon_md  # noqa: F401

__all__ = [
    "load_dataset",
    "list_datasets",
    "register_dataset",
    "get_builder",
    "DatasetBuilder",
    "default_cache_dir",
]
