"""Dataset hub: one-liner ``load_dataset()`` loading.

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

# Import builder modules for their registration side effects.
from . import (
    ani1,  # noqa: F401
    ani1ccx,  # noqa: F401
    ani1x,  # noqa: F401
    ani2x,  # noqa: F401
    argon_md,  # noqa: F401
    ethanol_response,  # noqa: F401
    gmtkn55,  # noqa: F401
    hdnnp4g,  # noqa: F401
    lode_dimers,  # noqa: F401
    noncovalent,  # noqa: F401
    oc20,  # noqa: F401
    qm7,  # noqa: F401
    qm9,  # noqa: F401
    rmd17,  # noqa: F401
    spookynet_states,  # noqa: F401
)
from ._download import default_cache_dir
from .base import (
    DatasetBuilder,
    get_builder,
    list_datasets,
    load_dataset,
    register_dataset,
)

__all__ = [
    "load_dataset",
    "list_datasets",
    "register_dataset",
    "get_builder",
    "DatasetBuilder",
    "default_cache_dir",
]
