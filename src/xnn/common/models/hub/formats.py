"""Model file formats the hub can read, beyond xnn's own.

xnn's own format (a portable model directory, or a trainer checkpoint) is
built in. Other formats register a :class:`ModelFormat` whose ``convert``
turns a file into an xnn model plus the config that rebuilds it; the hub then
caches the result in the portable format, so a converted model loads without
the upstream package afterwards. The MACE foundation models use this through
the ``"mace-torch"`` format, registered by
:mod:`xnn.gnn.models.mace_foundation`.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

XNN_FORMAT = "xnn"
XNN_SUFFIXES = (".pt", ".pth", ".ckpt")
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")

# formats provided by an optional family: name -> (module that registers it,
# file suffixes), so detection and the error messages work before the import
_PROVIDERS: dict[str, tuple[str, tuple[str, ...]]] = {
    "mace-torch": ("xnn.gnn.models.mace_foundation", (".model",)),
    "aimnet2": ("xnn.gnn.models.aimnet2_foundation", (".safetensors",)),
}


@dataclass(frozen=True)
class ModelFormat:
    """A foreign model format the hub converts into xnn's own.

    Attributes
    ----------
    name : str
        Format key used in model cards (e.g. ``"mace-torch"``).
    suffixes : tuple of str
        File suffixes recognized as this format.
    convert : callable
        ``convert(path, head) -> (model, config)``: the bare xnn potential and
        the :class:`~xnn.common.config.ModelConfig` that
        :func:`~xnn.common.models.build_model` rebuilds it from (the weights
        then load strictly), or a full :class:`~xnn.common.config.Config`
        when the model comes with more than its architecture (a
        ``subtracted_dispersion`` record).
    find_cached : callable, optional
        ``find_cached(card) -> path or None``: a copy of the card's file that
        some other cache already holds, so it is not downloaded again.
    requires : str
        What has to be installed to convert (shown in error messages).
    detect : callable, optional
        ``detect(path) -> bool``: recognizes a file or directory of this
        format when its suffix does not tell (a ``.pt`` artifact, a directory
        of weights and metadata); a path that does not exist is never one.
    """

    name: str
    suffixes: tuple[str, ...]
    convert: Callable
    find_cached: Optional[Callable] = None
    requires: str = ""
    detect: Optional[Callable] = None


_FORMATS: dict[str, ModelFormat] = {}


def register_format(fmt: ModelFormat) -> ModelFormat:
    """Register a foreign model format.

    Parameters
    ----------
    fmt : ModelFormat
        The format.

    Returns
    -------
    ModelFormat
        ``fmt``.
    """
    _FORMATS[fmt.name] = fmt
    return fmt


def get_format(name: str) -> ModelFormat:
    """Look up a foreign format, importing the family that provides it.

    Parameters
    ----------
    name : str
        Format key.

    Returns
    -------
    ModelFormat
        The registered format.

    Raises
    ------
    KeyError
        For an unknown format.
    ImportError
        When the providing family cannot be imported.
    """
    if name not in _FORMATS and name in _PROVIDERS:
        module = _PROVIDERS[name][0]
        try:
            importlib.import_module(module)
        except ImportError as e:
            raise ImportError(
                f"the {name!r} model format is provided by {module}, which "
                f"failed to import ({e}); the GNN family needs e3nn") from e
    try:
        return _FORMATS[name]
    except KeyError:
        known = sorted({XNN_FORMAT, *_FORMATS, *_PROVIDERS})
        raise KeyError(f"unknown model format {name!r}; known: {known}") from None


def suffixes() -> tuple[str, ...]:
    """Every suffix that marks a model file or an archive of one."""
    out = list(XNN_SUFFIXES) + list(ARCHIVE_SUFFIXES)
    for _, sfx in _PROVIDERS.values():
        out += sfx
    for fmt in _FORMATS.values():
        out += fmt.suffixes
    return tuple(dict.fromkeys(out))


def is_archive(path: Path) -> bool:
    """Whether ``path`` has an archive suffix."""
    return str(path).lower().endswith(ARCHIVE_SUFFIXES)


def detect_format(path: Path) -> str:
    """Guess a file's format from its suffix, or a format's own ``detect`` (xnn's own when unknown).

    Parameters
    ----------
    path : pathlib.Path
        A model file or directory.

    Returns
    -------
    str
        Format key.
    """
    path = Path(path)
    for fmt in _FORMATS.values():
        if fmt.detect is not None and fmt.detect(path):
            return fmt.name
    if path.is_dir():
        return XNN_FORMAT
    name = str(path).lower()
    for fmt in _FORMATS.values():
        if name.endswith(fmt.suffixes):
            return fmt.name
    for fmt_name, (_, sfx) in _PROVIDERS.items():
        if name.endswith(sfx):
            return fmt_name
    return XNN_FORMAT
