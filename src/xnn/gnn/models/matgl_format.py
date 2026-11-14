"""The model directories of MatGL (BSD-3-Clause), read by the model hub's ``"matgl"`` format.

A MatGL model is a directory of ``model.json`` (the architecture) and
``state.pt`` (the weights). The architecture's class picks the converter:
TensorNet potentials (:mod:`xnn.gnn.models.tensornet_foundation`), QET
potentials (:mod:`xnn.gnn.models.qet_foundation`) and MEGNet property models
(:mod:`xnn.gnn.models.megnet_foundation`).
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Optional, Union

from xnn.common.models.hub import ModelFormat, register_format

#: MatGL architecture -> the module whose ``convert_directory`` converts it.
CONVERTERS = {
    "TensorNet": "xnn.gnn.models.tensornet_foundation",
    "QET": "xnn.gnn.models.qet_foundation",
    "MEGNet": "xnn.gnn.models.megnet_foundation",
}


def matgl_class(path: Union[str, Path]) -> Optional[str]:
    """The architecture of a MatGL directory (``"TensorNet"``, ``"MEGNet"``, ...), or ``None``.

    Parameters
    ----------
    path : str or pathlib.Path
        The directory, or its ``model.json`` or ``state.pt``.

    Returns
    -------
    str or None
        The class of the wrapped model, ``None`` when ``path`` is not a MatGL directory.
    """
    path = Path(path)
    if path.is_file() and path.name in ("model.json", "state.pt"):
        path = path.parent
    if not ((path / "model.json").is_file() and (path / "state.pt").is_file()):
        return None
    try:
        meta = json.loads((path / "model.json").read_text())
        return str(meta["kwargs"]["model"]["@class"])
    except (ValueError, KeyError, TypeError):
        return None


def is_matgl_dir(path: Union[str, Path]) -> bool:
    """Whether ``path`` is a MatGL directory of an architecture xnn converts."""
    return matgl_class(path) in CONVERTERS


def convert_directory(path: Union[str, Path], dtype=None) -> tuple:
    """Convert a MatGL directory with the converter of its architecture.

    Parameters
    ----------
    path : str or pathlib.Path
        The directory, or its ``model.json`` or ``state.pt``.
    dtype : torch.dtype, optional
        The dtype the model is built in, by default the checkpoint's.

    Returns
    -------
    tuple
        The xnn model and the :class:`~xnn.common.config.Config` that rebuilds it.

    Raises
    ------
    NotImplementedError
        For an architecture without a converter.
    """
    cls = matgl_class(path)
    if cls not in CONVERTERS:
        raise NotImplementedError(f"{path}: MatGL {cls} models are not converted; "
                                  f"converted: {sorted(CONVERTERS)}")
    return importlib.import_module(CONVERTERS[cls]).convert_directory(path, dtype)


def _convert_file(path: Path, head: Optional[str]):
    """The model hub's ``matgl`` converter: directory -> (model, config)."""
    return convert_directory(path)


register_format(ModelFormat(name="matgl", suffixes=(), convert=_convert_file, detect=is_matgl_dir))

__all__ = ["CONVERTERS", "matgl_class", "is_matgl_dir", "convert_directory"]
