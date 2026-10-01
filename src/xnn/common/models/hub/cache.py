"""Where the model hub keeps downloaded and converted models.

Every model lives in its own directory under the cache, named after its
registry key, so a cache (or any one model directory) can be copied to
another system and used there as is::

    <cache>/
        mace-off23-small/             card.json  config.yaml  model.pt
        mace-mh-0/raw/mace-mh-0.model (kept: the other heads convert from it)
        mace-mh-0/heads/omat_pbe/     card.json  config.yaml  model.pt
        zenodo.18957344/record.json
        zenodo.18957344/<file>/       card.json  config.yaml  model.pt
        url/<file>-<hash>/            card.json  config.yaml  model.pt

The directory is chosen per call with ``cache_dir``; otherwise
:func:`default_model_cache_dir` applies.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Iterator, Optional, Union

from .card import CARD_FILE, CONFIG_FILE, WEIGHTS_FILE, ModelCard


def _repo_root() -> Optional[Path]:
    """The source checkout this module lives in, or ``None`` for an install.

    Returns
    -------
    pathlib.Path or None
        ``<repo>`` when ``src/xnn`` and ``pyproject.toml`` sit there.
    """
    root = Path(__file__).resolve().parents[5]
    if (root / "pyproject.toml").is_file() and (root / "src" / "xnn").is_dir():
        return root
    return None


def default_model_cache_dir() -> Path:
    """Return the directory models are cached in when no ``cache_dir`` is given.

    The ``XNN_MODELS`` environment variable wins, then ``XNN_CACHE`` (models go
    to its ``models/`` subfolder, beside the datasets), then the repository's
    ``models/`` directory in a source checkout, mirroring ``datasets/``. An
    installed package falls back to ``~/.cache/xnn/models``.

    Returns
    -------
    pathlib.Path
        The cache directory (not created here).
    """
    v = os.environ.get("XNN_MODELS")
    if v:
        return Path(v).expanduser()
    v = os.environ.get("XNN_CACHE")
    if v:
        return Path(v).expanduser() / "models"
    root = _repo_root()
    if root is not None:
        return root / "models"
    return Path.home() / ".cache" / "xnn" / "models"


def resolve_cache_dir(cache_dir: Optional[Union[str, Path]]) -> Path:
    """The explicit ``cache_dir`` (``~`` expanded) or the default one."""
    if cache_dir is None:
        return default_model_cache_dir()
    return Path(cache_dir).expanduser()


def offline() -> bool:
    """Whether ``XNN_OFFLINE`` forbids network access (``1``/``true``/``yes``)."""
    return os.environ.get("XNN_OFFLINE", "").strip().lower() in ("1", "true", "yes")


def sanitize(name: str) -> str:
    """Make ``name`` safe as one path component (keeps ``[A-Za-z0-9._-]``)."""
    out = re.sub(r"[^A-Za-z0-9._-]", "_", str(name)).strip(".")
    return out or "_"


def url_slot(url: str) -> str:
    """Cache key of a model downloaded from a plain URL."""
    base = sanitize(os.path.basename(url.split("?", 1)[0].rstrip("/")) or "model")
    digest = hashlib.sha1(url.encode()).hexdigest()[:10]
    return f"url/{base}-{digest}"


def is_cache_key(source: str) -> bool:
    """Whether ``source`` can name a directory inside the cache.

    A relative path of safe components, so a key never escapes the cache.
    """
    if not source or "://" in source or source.startswith(("/", "~", "\\")):
        return False
    parts = source.replace("\\", "/").split("/")
    return all(p and p not in (".", "..") and sanitize(p) == p for p in parts)


def head_dir(slot: Path, head: Optional[str]) -> Path:
    """Directory of one head of a multi-head model (the slot itself for none)."""
    return slot if head is None else slot / "heads" / sanitize(head)


def is_ready(path: Path) -> bool:
    """Whether ``path`` holds a complete portable model.

    The card is written last, so its presence means the other files are
    complete.
    """
    return all((path / f).is_file() for f in (CARD_FILE, CONFIG_FILE, WEIGHTS_FILE))


def scan(cache: Path) -> Iterator[tuple[str, Path, ModelCard]]:
    """Yield every complete model directory in the cache.

    Parameters
    ----------
    cache : pathlib.Path
        Cache directory.

    Yields
    ------
    tuple of (str, pathlib.Path, ModelCard)
        The cache key (relative path, with ``heads/<head>`` stripped), the
        directory, and its card.
    """
    if not cache.is_dir():
        return
    for card_path in sorted(cache.rglob(CARD_FILE)):
        d = card_path.parent
        if not is_ready(d):
            continue
        try:
            card = ModelCard.load(card_path)
        except (OSError, ValueError):
            continue
        rel = d.relative_to(cache).as_posix()
        if "/heads/" in rel:
            rel = rel.rsplit("/heads/", 1)[0]
        yield rel, d, card
