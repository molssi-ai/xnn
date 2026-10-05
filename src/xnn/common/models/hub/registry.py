"""The registry of pre-trained models and :func:`list_models`.

The models xnn knows about are listed in ``models.json``, shipped inside the
package: xnn-trained models and the MACE and AIMNet2 foundation models alike, one
:class:`~xnn.common.models.hub.card.ModelCard` each. :func:`register_pretrained`
adds more at run time (a lab's own Zenodo uploads, say). :func:`list_models`
shows the registry together with whatever else sits in the cache, so a model
directory copied in by hand is listed and loadable by its name.
"""
from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any, Optional, Union

from .cache import head_dir, is_cache_key, is_ready, resolve_cache_dir, scan
from .card import ModelCard

_REGISTRY: dict[str, ModelCard] = {}
_ALIASES: dict[str, str] = {}      # alias -> registered name
REGISTRY_FILE = "models.json"


def _load_shipped() -> None:
    """Register the cards of the packaged ``models.json``."""
    text = resources.files(__package__).joinpath(REGISTRY_FILE).read_text()
    for entry in json.loads(text)["models"]:
        register_pretrained(ModelCard.from_dict(entry))


def register_pretrained(card: Union[ModelCard, dict, None] = None, **fields: Any) -> ModelCard:
    """Add a model to the registry (for this process).

    Parameters
    ----------
    card : ModelCard or dict, optional
        The card. Keyword arguments are merged on top, so
        ``register_pretrained(name="my-model", doi="10.5281/zenodo.123")``
        works too.
    **fields
        Card fields.

    Returns
    -------
    ModelCard
        The registered card.

    Raises
    ------
    ValueError
        If the name is not a safe cache key (letters, digits, ``._-``, with
        ``/`` between parts).
    """
    data = card.to_dict() if isinstance(card, ModelCard) else dict(card or {})
    data.update(fields)
    new = ModelCard.from_dict(data)
    if not is_cache_key(new.name):
        raise ValueError(f"invalid model name {new.name!r}: use letters, digits, "
                         f"'.', '_' and '-' (with '/' between parts)")
    for alias in new.aliases:
        if not is_cache_key(alias):
            raise ValueError(f"invalid alias {alias!r} of {new.name}")
        if alias in _REGISTRY and alias != new.name:
            raise ValueError(f"alias {alias!r} of {new.name} is a registered model name")
    _REGISTRY[new.name] = new
    for alias in new.aliases:
        _ALIASES[alias] = new.name
    return new


def registered_cards(format: Optional[str] = None) -> list[ModelCard]:
    """The registered cards, sorted by name, optionally of one ``format``."""
    return [c for _, c in sorted(_REGISTRY.items()) if format is None or c.format == format]


def get_card(name: str) -> Optional[ModelCard]:
    """The registered card for ``name`` (a model name or an alias), or ``None``."""
    name = str(name)
    return _REGISTRY.get(name) or _REGISTRY.get(_ALIASES.get(name, ""))


def model_card(name: str, cache_dir: Optional[Union[str, Path]] = None) -> ModelCard:
    """Return the card of a registered or cached model.

    Parameters
    ----------
    name : str
        Registry name, alias, or cache key.
    cache_dir : str or pathlib.Path, optional
        Cache to look in for models that are not registered.

    Returns
    -------
    ModelCard
        The card (the cached copy's for a model that is only cached).

    Raises
    ------
    KeyError
        If the model is neither registered nor cached.
    """
    card = get_card(name)
    if card is not None:
        return card
    if is_cache_key(str(name)):
        slot = resolve_cache_dir(cache_dir) / str(name)
        if is_ready(slot):
            return ModelCard.load(slot / "card.json")
        for _, d, c in scan(slot):
            return c
    raise KeyError(f"unknown model {name!r}; see list_models()")


def _status(card: ModelCard, cache: Path) -> str:
    """How much of a registered model is on disk already.

    ``"ready"`` (loads offline), ``"ready: <heads>"`` for the converted heads
    of a multi-head model, ``"raw"`` (downloaded, converted on first load), or
    ``""`` (needs a download), or ``"unsupported"``.
    """
    if card.unsupported:
        return "unsupported"
    slot = cache / card.name
    if is_ready(slot):
        return "ready"
    heads = [h for h in card.heads or [] if is_ready(head_dir(slot, h))]
    if heads:
        return "ready: " + ", ".join(heads)
    raw = slot / "raw"
    if raw.is_dir() and any(raw.iterdir()):
        return "raw"
    if card.format != "xnn":
        try:
            from .formats import get_format
            find = get_format(card.format).find_cached
            if find is not None and find(card) is not None:
                return "raw"
        except (ImportError, KeyError):
            pass
    return ""


def _row(card: ModelCard, status: str) -> dict[str, Any]:
    """One :func:`list_models` row."""
    return {
        "name": card.name,
        "architecture": card.architecture or "",
        "format": card.format,
        "cutoff": card.cutoff if card.cutoff is not None else "",
        "elements": len(card.species) if card.species else "",
        "heads": len(card.heads) if card.heads and len(card.heads) > 1 else "",
        "license": card.license or "",
        "cached": status,
        "description": card.description,
    }


def list_models(cache_dir: Optional[Union[str, Path]] = None, *,
                cached_only: bool = False, format: Optional[str] = None,
                architecture: Optional[str] = None, tag: Optional[str] = None,
                details: bool = False) -> Union[list[str], list[dict[str, Any]]]:
    """List the pre-trained models available to load.

    The registry (``models.json`` plus :func:`register_pretrained`) is listed
    together with every complete model directory found in the cache, so
    models fetched by URL or DOI, or copied in from another machine, show up
    by their cache key.

    Parameters
    ----------
    cache_dir : str or pathlib.Path, optional
        Cache to inspect. Defaults to
        :func:`~xnn.common.models.hub.cache.default_model_cache_dir`.
    cached_only : bool, optional
        Only models that load without a download.
    format : str, optional
        Only models published in this format (``"xnn"``, ``"mace-torch"``).
    architecture : str, optional
        Only this xnn architecture (``"mace"``, ``"schnet"``, ...).
    tag : str, optional
        Only models carrying this tag.
    details : bool, optional
        Return one dict per model (name, architecture, format, cutoff, number
        of elements and heads, license, cache status, description) instead of
        the names.

    Returns
    -------
    list of str or list of dict
        Names (sorted), or rows when ``details=True``.

    Examples
    --------
    >>> from xnn.common.models import list_models
    >>> list_models(architecture="mace")[:2]
    ['mace-mh-0', 'mace-mh-1']
    """
    cache = resolve_cache_dir(cache_dir)
    rows: dict[str, dict[str, Any]] = {}
    for name, card in _REGISTRY.items():
        rows[name] = _row(card, _status(card, cache))
    for key, _, card in scan(cache):
        if key in rows or not key:
            continue
        card.name = key
        rows[key] = _row(card, "ready")

    def keep(r: dict[str, Any], card: ModelCard) -> bool:
        if cached_only and not r["cached"].startswith(("ready", "raw")):
            return False
        if format is not None and r["format"] != format:
            return False
        if architecture is not None and r["architecture"] != architecture:
            return False
        if tag is not None and tag not in card.tags:
            return False
        return True

    selected = []
    for name in sorted(rows):
        card = _REGISTRY.get(name) or model_card(name, cache)
        if keep(rows[name], card):
            selected.append(rows[name])
    return selected if details else [r["name"] for r in selected]


_load_shipped()
