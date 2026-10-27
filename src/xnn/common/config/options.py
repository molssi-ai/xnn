"""Config options that nothing uses are ignored, with a warning.

A registered model's ``from_config`` reads its options (:attr:`ModelConfig.extra`)
through :func:`model_options`, which records every key read and the default a
missing one falls back to; so do the build steps around it (pretrained
weights, LoRA, heads, dispersion, long-range).
:func:`~xnn.common.models.registry.build_model` then warns about the options
that nothing read, and the config loaders about keys they do not know: a
misspelled option, or one carried over from another code's config (whose keys
the translation tables may not cover), is ignored, and the user is told so.
The resolved options (those given and the defaults used,
:meth:`ModelOptions.resolved`) are what a saved model records, so that it
rebuilds as it was trained whatever later versions choose as defaults.
"""
from __future__ import annotations

import warnings

#: Options read outside the model build: ``fit_scaling`` by the trainer,
#: ``species`` by the exporters (recorded with every model).
CONFIG_KEYS = ("fit_scaling", "species")


class IgnoredOptionWarning(UserWarning):
    """A config key or option that xnn does not use, and ignores."""


def warn_ignored(where: str, keys) -> None:
    """Warn that ``keys`` of ``where`` are ignored (nothing for no keys).

    Parameters
    ----------
    where : str
        What the keys belong to, for the message (``"model 'painn'"``,
        ``"data"``, ...).
    keys : iterable of str
        The ignored keys.
    """
    keys = sorted(keys)
    if keys:
        warnings.warn(f"{where}: ignoring the unrecognized option(s) {keys}",
                      IgnoredOptionWarning, stacklevel=3)


class ModelOptions(dict):
    """A copy of :attr:`ModelConfig.extra` that records which options are read.

    Reading a key (``options[key]``, :meth:`get`, ``key in options``,
    :meth:`pop`, :meth:`setdefault`) adds it to :attr:`read`, and the default
    :meth:`get` returns for a missing key to :attr:`defaults`; both are shared
    with the copies made by :meth:`copy` and :func:`model_options`.

    Parameters
    ----------
    values : mapping, optional
        The options.
    read : set or None, optional
        The set to record reads in; ``None`` starts a new one.
    defaults : dict or None, optional
        The dict to record defaults in; ``None`` starts a new one.
    """

    def __init__(self, values=(), read: set | None = None, defaults: dict | None = None):
        super().__init__(values)
        self.read = set() if read is None else read
        self.defaults = {} if defaults is None else defaults

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.read.add(key)
        if not super().__contains__(key):
            self.defaults.setdefault(key, default)
        return super().get(key, default)

    def __contains__(self, key):
        self.read.add(key)
        return super().__contains__(key)

    def pop(self, key, *default):
        self.read.add(key)
        return super().pop(key, *default)

    def setdefault(self, key, default=None):
        self.read.add(key)
        return super().setdefault(key, default)

    def copy(self) -> "ModelOptions":
        return ModelOptions(self, self.read, self.defaults)

    def resolved(self) -> dict:
        """The options given and the defaults used for the others.

        Only defaults that a config file can hold (``None``, numbers, strings
        and lists or dicts of those) are included; tuples become lists.
        """
        out = {k: _plain(v) for k, v in self.defaults.items() if _is_plain(v)}
        out.update(self)
        return out

    def unread(self) -> list[str]:
        """The options nothing has read, sorted.

        :data:`CONFIG_KEYS` and options set to ``None`` (not set) are excepted.
        """
        return sorted(k for k, v in self.items()
                      if k not in self.read and k not in CONFIG_KEYS and v is not None)


def model_options(cfg) -> ModelOptions:
    """The options of a model config, for a ``from_config`` to read.

    A copy, so that the caller may change it; inside
    :func:`~xnn.common.models.registry.build_model` the reads are recorded, so
    that unused options can be reported.

    Parameters
    ----------
    cfg : ModelConfig
        The model configuration.

    Returns
    -------
    ModelOptions
        ``cfg.extra`` (empty when ``None``).
    """
    extra = cfg.extra or {}
    if isinstance(extra, ModelOptions):
        return ModelOptions(extra, extra.read, extra.defaults)
    return ModelOptions(extra)


def _is_plain(value) -> bool:
    """Whether ``value`` can be written to a config file."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_plain(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, (str, int)) and _is_plain(v) for k, v in value.items())
    return False


def _plain(value):
    """``value`` with tuples as lists."""
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value
