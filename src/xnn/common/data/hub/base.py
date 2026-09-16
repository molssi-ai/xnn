"""Dataset registry and the ``load_dataset`` entry point.

A :class:`DatasetBuilder` knows how to download and preprocess one upstream
dataset into xnn' native *structure dicts* (the list-of-dicts format consumed
by :class:`~xnn.common.data.dataset.AtomicDataset`). Builders self-register
under a short name via :func:`register_dataset`; :func:`load_dataset` looks one
up and drives it, giving a HuggingFace ``load_dataset()``-style one-liner.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

from ._download import default_cache_dir

_REGISTRY: dict[str, "DatasetBuilder"] = {}


class DatasetBuilder(ABC):
    """Base class for a downloadable, preprocessable dataset.

    Subclasses set the class attribute :attr:`name` and implement
    :meth:`load`, which downloads (with caching) and converts the upstream data
    into xnn structure dicts. Register an instance with
    :func:`register_dataset` to make it available through :func:`load_dataset`.

    Attributes
    ----------
    name : str
        Short registry key (e.g. ``"rmd17"``).
    description : str
        One-line human-readable summary, shown by :func:`list_datasets`.
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    def load(self, *, split: Optional[str], cache_dir: Path,
             **kwargs: Any) -> Union[dict[str, list[dict]], list[dict]]:
        """Download, preprocess, and return the dataset as structure dicts.

        Parameters
        ----------
        split : str or None
            Which split to return. ``None`` returns every split as a mapping
            ``{split_name: structures}``; a name returns that split's list.
        cache_dir : pathlib.Path
            Base cache directory; the builder stores its files under
            ``cache_dir / self.name``.
        **kwargs
            Builder-specific options.

        Returns
        -------
        dict of {str: list of dict} or list of dict
            All splits (``split is None``) or the requested split, each a list
            of structure dicts in the format accepted by
            :func:`~xnn.common.data.dataset.structure_to_graph`.
        """
        raise NotImplementedError


def register_dataset(builder: DatasetBuilder) -> DatasetBuilder:
    """Register a builder instance under its :attr:`~DatasetBuilder.name`.

    Parameters
    ----------
    builder : DatasetBuilder
        The builder to register.

    Returns
    -------
    DatasetBuilder
        The same ``builder``, so this can be used as a decorator on a
        zero-argument construction.
    """
    _REGISTRY[builder.name] = builder
    return builder


def get_builder(name: str) -> DatasetBuilder:
    """Look up a registered builder by name.

    Parameters
    ----------
    name : str
        Registry key.

    Returns
    -------
    DatasetBuilder
        The registered builder.

    Raises
    ------
    KeyError
        If no dataset is registered under ``name``.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown dataset {name!r}; available: {', '.join(list_datasets())}"
        ) from None


def list_datasets() -> list[str]:
    """Return the names of all registered datasets, sorted.

    Returns
    -------
    list of str
        Registered dataset names.
    """
    return sorted(_REGISTRY)


def load_dataset(name: str, *, split: Optional[str] = None,
                 cutoff: Optional[float] = None,
                 cache_dir: Optional[Union[str, Path]] = None,
                 **kwargs: Any):
    """Download and preprocess an upstream dataset in one call.

    Looks up the builder registered under ``name`` and returns its data as xnn
    structure dicts, optionally wrapped as ready-to-train
    :class:`~xnn.common.data.dataset.AtomicDataset` objects.

    Files are downloaded (and cached, verified by MD5) under
    ``cache_dir / name`` -- by default the repository's ``datasets/`` directory,
    so re-running is instant and offline. See
    :func:`~xnn.common.data.hub._download.default_cache_dir`.

    Parameters
    ----------
    name : str
        Registered dataset name; see :func:`list_datasets` (e.g. ``"rmd17"``).
    split : str, optional
        Which split to return. ``None`` (default) returns a mapping
        ``{split_name: data}``; a name (e.g. ``"train"``) returns just that
        split. Available splits are dataset-specific.
    cutoff : float, optional
        If given, each returned split is wrapped in an ``AtomicDataset`` built
        with this neighbor-list cutoff (ready for a ``DataLoader``). If omitted,
        raw lists of structure dicts are returned.
    cache_dir : str or pathlib.Path, optional
        Base cache directory. Defaults to
        :func:`~xnn.common.data.hub._download.default_cache_dir`.
    **kwargs
        Forwarded to the builder's :meth:`DatasetBuilder.load` (e.g. rMD17's
        ``molecule``, ``fold``, ``units``).

    Returns
    -------
    dict or list or AtomicDataset
        Structure-dict lists (or ``AtomicDataset``\\ s when ``cutoff`` is set),
        as a per-split mapping when ``split is None`` else for the single split.

    Examples
    --------
    >>> from xnn.common.data import load_dataset
    >>> splits = load_dataset("rmd17", molecule="aspirin")      # {"train","test"}
    >>> train = load_dataset("rmd17", molecule="aspirin", split="train", cutoff=5.0)
    """
    builder = get_builder(name)
    base = Path(cache_dir).expanduser() if cache_dir is not None else default_cache_dir()
    result = builder.load(split=split, cache_dir=base, **kwargs)

    if cutoff is None:
        return result

    from ..dataset import AtomicDataset
    if isinstance(result, dict):
        return {k: AtomicDataset(v, cutoff) for k, v in result.items()}
    return AtomicDataset(result, cutoff)
