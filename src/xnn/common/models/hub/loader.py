"""``from_pretrained``: one call to load any pre-trained model.

:func:`fetch_model` turns a source (registry name, local path, URL or Zenodo
DOI) into a local model directory, downloading, verifying, unpacking and
converting as needed; :func:`load_pretrained` builds the model from it. Both
honor ``cache_dir``, ``force_download`` and ``local_files_only``.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from torch import nn

from ...data.hub._download import download_file, extract_archive
from ..fast import set_use_fast
from . import zenodo
from .cache import (head_dir, is_cache_key, is_ready, offline, resolve_cache_dir,
                    sanitize, url_slot)
from .card import CARD_FILE, CONFIG_FILE, WEIGHTS_FILE, ModelCard
from .checkpoint import build_potential, load_checkpoint, save_pretrained
from .formats import XNN_FORMAT, detect_format, get_format, is_archive, suffixes
from .registry import get_card

logger = logging.getLogger(__name__)

# card fields a cached directory inherits from the card it was fetched for
_INHERITED = ("name", "description", "heads", "units", "license", "citation", "doi",
              "url", "filename", "tags", "notes")
_ASL = ("{name} is distributed under the Academic Software License "
        "(https://github.com/gabor1/ASL); by using it you accept its terms "
        "(no commercial use).")
_noticed: set[str] = set()


@dataclass
class PretrainedModel:
    """A loaded model with everything known about it.

    Attributes
    ----------
    model : ForceStressOutput
        The ready model (forces, optionally stress), in eval mode.
    config : Config
        The configuration it was built from.
    card : ModelCard or None
        Its card (``None`` for a bare checkpoint file).
    cutoff : float
        Neighbor-list cutoff of the built model, in angstrom (a dispersion
        wrapper widens it beyond the core model's radius).
    path : pathlib.Path
        The local directory or file it was loaded from.
    """

    model: nn.Module
    config: Any
    card: Optional[ModelCard]
    cutoff: float
    path: Path


def _pick_head(card: ModelCard, head: Optional[str]) -> Optional[str]:
    """Validate ``head`` against the card; ``None`` for single-head models."""
    heads = card.heads
    if not heads:
        return head
    if len(heads) == 1:
        if head is not None and head != heads[0]:
            raise ValueError(f"{card.name} has the single head {heads[0]!r}, not {head!r}")
        return None
    if head is None:
        raise ValueError(f"{card.name} has {len(heads)} heads; pass head= with one of {heads}")
    if head not in heads:
        raise ValueError(f"unknown head {head!r} of {card.name}; it has {heads}")
    return head


def _card_fields(card: ModelCard, head: Optional[str]) -> dict[str, Any]:
    """Fields a cached directory takes over from the card it was fetched for."""
    d = card.to_dict()
    out = {k: d[k] for k in _INHERITED if k in d}
    out["source"] = (f"https://doi.org/{card.doi}" if card.doi else card.url) or card.source
    if head is not None:
        out["head"] = head
    return out


def _model_dir(root: Path) -> Optional[Path]:
    """The directory under ``root`` that holds a portable model, if any."""
    for config in sorted(root.rglob(CONFIG_FILE)):
        if (config.parent / WEIGHTS_FILE).is_file():
            return config.parent
    return None


def _install(raw: Path, target: Path, card: ModelCard, head: Optional[str],
             fmt: str) -> None:
    """Turn a downloaded file into the portable directory ``target``.

    An archive is unpacked (a portable directory inside is taken as is, a
    single checkpoint inside is installed in turn), an xnn checkpoint is
    repacked, and a foreign format is converted.
    """
    fields = _card_fields(card, head)
    if raw.is_dir() or is_archive(raw):
        work = Path(tempfile.mkdtemp(prefix=".unpack-", dir=target.parent))
        try:
            if raw.is_dir():
                shutil.copytree(raw, work, dirs_exist_ok=True)
            else:
                extract_archive(raw, work)
            found = _model_dir(work)
            if found is not None:
                if (found / CARD_FILE).is_file():
                    # the uploader's card wins; the registry fills in the
                    # rest, and name and source say where this copy is from
                    own = ModelCard.load(found / CARD_FILE).to_dict()
                    fields = {k: v for k, v in fields.items()
                              if k not in own or k in ("name", "source")}
                save_pretrained(found, target, card=fields)
                return
            inner = [p for p in work.rglob("*") if p.is_file()
                     and p.name.lower().endswith(suffixes()) and not is_archive(p)]
            if len(inner) != 1:
                raise FileNotFoundError(
                    f"{raw.name} holds no model directory and "
                    f"{'several' if inner else 'no'} checkpoint files "
                    f"{[p.name for p in inner]}")
            _install(inner[0], target, card, head, detect_format(inner[0]))
        finally:
            shutil.rmtree(work, ignore_errors=True)
        return
    if fmt == XNN_FORMAT:
        save_pretrained(raw, target, card=fields)
    else:
        model, cfg = get_format(fmt).convert(raw, head)
        # the converter builds the model from this very config, so the
        # strict rebuild check of save_pretrained would only cost time
        save_pretrained(model, target, config=cfg, card=fields, verify=False)


def _existing_raw(card: ModelCard, slot: Path, fname: str) -> Optional[Path]:
    """A copy of the card's file that is already on disk."""
    raw = slot / "raw" / sanitize(fname)
    if raw.is_file():
        return raw
    if card.format != XNN_FORMAT:
        find = get_format(card.format).find_cached
        if find is not None:
            return find(card)
    return None


def _fetch_card(card: ModelCard, slot: Path, head: Optional[str], cache: Path, *,
                format: Optional[str], force: bool, no_net: bool, quiet: bool) -> Path:
    """Make sure the model of ``card`` is in ``slot``; return its directory."""
    if card.unsupported:
        raise NotImplementedError(f"{card.name} cannot be loaded: {card.unsupported}")
    head = _pick_head(card, head)
    target = head_dir(slot, head)
    if is_ready(target) and not force:
        return target
    fmt = format or card.format

    # where to get it from
    portable_files = None
    if card.doi:
        ref = zenodo.parse(card.doi)
        if ref is None:
            raise ValueError(f"{card.name}: only Zenodo DOIs can be resolved, got {card.doi}")
        recid = ref[0]
        record = zenodo.fetch_record(recid, cache / f"zenodo.{recid}",
                                     refresh=force, offline=no_net)
        entries = zenodo.files(record)
        if card.filename is None and any(e["key"] == CARD_FILE for e in entries):
            portable_files = [e for e in entries
                              if e["key"] in (CARD_FILE, CONFIG_FILE, WEIGHTS_FILE)]
            fname = card.name
        else:
            entry = zenodo.choose(entries, card.filename, suffixes(), recid)
            url, md5, fname = entry["url"], entry["md5"], entry["key"]
    elif card.url:
        url, md5 = card.url, card.md5
        fname = os.path.basename(urllib.parse.urlparse(card.url).path) or "model"
    else:
        raise FileNotFoundError(
            f"{card.name} is registered but has no download source yet (its card "
            f"names no DOI or URL) and is not cached in {cache}")

    target.parent.mkdir(parents=True, exist_ok=True)
    slot.mkdir(parents=True, exist_ok=True)
    download = slot / ".download"
    try:
        if portable_files is not None:
            if no_net:
                raise FileNotFoundError(f"{card.name} is not cached in {cache} and "
                                        f"network access is disabled")
            for e in portable_files:
                dest = download / e["key"]
                if force:
                    dest.unlink(missing_ok=True)
                download_file(e["url"], dest, md5=e["md5"], quiet=quiet)
            _install(download, target, card, head, XNN_FORMAT)
            return target
        raw = None if force else _existing_raw(card, slot, fname)
        if raw is None:
            if no_net:
                raise FileNotFoundError(f"{card.name} is not cached in {cache} and "
                                        f"network access is disabled "
                                        f"(local_files_only=True or XNN_OFFLINE)")
            # a file to convert goes to raw/, where it outlives a failed
            # conversion (a forgotten head=) and serves the other heads
            dest = (download if fmt == XNN_FORMAT else slot / "raw") / sanitize(fname)
            if force:
                dest.unlink(missing_ok=True)
            raw = download_file(url, dest, md5=md5, quiet=quiet)
        _install(raw, target, card, head, fmt)
        # converted without a head means single-head: nothing else needs it
        if head is None and raw.parent == slot / "raw":
            raw.unlink(missing_ok=True)
            if not any(raw.parent.iterdir()):
                raw.parent.rmdir()
    finally:
        shutil.rmtree(download, ignore_errors=True)
    return target


def fetch_model(source: Union[str, Path], *, cache_dir: Optional[Union[str, Path]] = None,
                filename: Optional[str] = None, head: Optional[str] = None,
                format: Optional[str] = None, force_download: bool = False,
                local_files_only: bool = False, quiet: bool = False) -> Path:
    """Return a local copy of a model, downloading and converting it if needed.

    Sources are tried in this order:

    1. an existing local path (a model directory or a checkpoint file), used
       in place;
    2. a registered name (:func:`list_models`), fetched from the DOI or URL
       in its card into ``<cache_dir>/<name>/``;
    3. a Zenodo DOI or record link, into ``<cache_dir>/zenodo.<id>/<file>/``;
    4. a plain ``http(s)`` URL of a checkpoint or archive, into
       ``<cache_dir>/url/<file>-<hash>/``;
    5. the key of a directory already in the cache (as :func:`list_models`
       shows it), e.g. one copied over from another machine.

    Whatever is fetched is stored as a portable model directory (see
    :func:`~xnn.common.models.hub.save_pretrained`), so the next call needs
    no network and no converter package.

    Parameters
    ----------
    source : str or pathlib.Path
        What to load.
    cache_dir : str or pathlib.Path, optional
        Cache directory. Defaults to
        :func:`~xnn.common.models.hub.cache.default_model_cache_dir`.
    filename : str, optional
        File to take from a Zenodo record that holds several models.
    head : str, optional
        Head to keep from a multi-head model.
    format : str, optional
        Format of the published file when its suffix does not tell
        (``"xnn"``, ``"mace-torch"``).
    force_download : bool, optional
        Download and convert again even when cached.
    local_files_only : bool, optional
        Never use the network (also set by ``XNN_OFFLINE=1``); fail when the
        model is not cached.
    quiet : bool, optional
        Hide download progress bars.

    Returns
    -------
    pathlib.Path
        A model directory, or the local file that was passed in.

    Raises
    ------
    FileNotFoundError
        If the source is not found, or not cached while offline.
    ValueError
        For a DOI that is not on Zenodo, or an ambiguous record or head.
    """
    s = str(source)
    local = Path(s).expanduser()
    if "://" not in s and local.exists():
        return local
    cache = resolve_cache_dir(cache_dir)
    opts = dict(format=format, force=force_download,
                no_net=local_files_only or offline(), quiet=quiet)

    card = get_card(s)
    if card is not None:
        return _fetch_card(card, cache / card.name, head, cache, **opts)

    ref = zenodo.parse(s)
    if ref is not None:
        recid, fname = ref
        fname = fname or filename
        root = cache / f"zenodo.{recid}"
        if fname is not None and not force_download:
            target = head_dir(root / sanitize(fname), head)
            if is_ready(target):
                return target
        record = zenodo.fetch_record(recid, root, refresh=force_download,
                                     offline=opts["no_net"])
        entries = zenodo.files(record)
        doi = record.get("doi") or f"10.5281/zenodo.{recid}"
        common = dict(doi=doi, description=str(record.get("metadata", {}).get("title", "")),
                      license=zenodo.license_id(record), citation=zenodo.citation(record))
        if fname is None and any(e["key"] == CARD_FILE for e in entries):
            card = ModelCard(name=f"zenodo.{recid}", **common)
            slot = root
        else:
            entry = zenodo.choose(entries, fname, suffixes(), recid)
            key = sanitize(entry["key"])
            card = ModelCard(name=f"zenodo.{recid}/{key}", filename=entry["key"],
                             format=format or detect_format(Path(entry["key"])), **common)
            slot = root / key
        return _fetch_card(card, slot, head, cache, **opts)

    if s.startswith(("http://", "https://")):
        if zenodo.is_doi(s):
            raise ValueError(f"only Zenodo DOIs can be resolved to files; for {s} pass "
                             f"the direct download URL of the model file")
        key = url_slot(s)
        path = Path(urllib.parse.urlparse(s).path)
        card = ModelCard(name=key, url=s, format=format or detect_format(path))
        return _fetch_card(card, cache / key, head, cache, **opts)

    if zenodo.is_doi(s):
        raise ValueError(f"only Zenodo DOIs can be resolved to files, got {s}")
    if is_cache_key(s) and not force_download:
        target = head_dir(cache / s, head)
        if is_ready(target):
            return target
    raise FileNotFoundError(
        f"{s!r} is not a registered model (see list_models()), a local path, a URL "
        f"or a Zenodo DOI, and nothing is cached under that name in {cache}")


def _license_notice(card: Optional[ModelCard]) -> None:
    """Log the terms of a restrictively licensed model, once per process."""
    if card is None or not str(card.license or "").upper().startswith("ASL"):
        return
    if card.name not in _noticed:
        _noticed.add(card.name)
        logger.warning(_ASL.format(name=card.name))


def load_pretrained(source: Union[str, Path], *, cache_dir: Optional[Union[str, Path]] = None,
                    filename: Optional[str] = None, head: Optional[str] = None,
                    format: Optional[str] = None, force_download: bool = False,
                    local_files_only: bool = False, quiet: bool = False,
                    device: str = "cpu", dtype=None, dispersion: Any = None,
                    compute_forces: bool = True, compute_stress: bool = False,
                    eeq_reuse: bool = False, use_fast: Union[bool, str] = "auto",
                    model_options: Optional[dict[str, Any]] = None) -> PretrainedModel:
    """Load a pre-trained model together with its config and card.

    The same as :func:`from_pretrained`, returning a :class:`PretrainedModel`
    so callers that need the config, card or cutoff (an ASE calculator, the
    MDI engine, ``xnn export``) get them without reloading. See
    :func:`fetch_model` for the source and cache options and
    :func:`~xnn.common.models.hub.checkpoint.build_potential` for ``dtype``,
    ``dispersion``, ``eeq_reuse`` and ``model_options``. ``use_fast``
    (``"auto"``, ``True`` or ``False``) chooses between the fused GPU
    kernels and the reference implementation of the blocks that have both
    (see :mod:`xnn.common.models.fast`).

    Returns
    -------
    PretrainedModel
        The model and its metadata.

    Raises
    ------
    ValueError
        If the checkpoint embeds no config.
    """
    path = fetch_model(source, cache_dir=cache_dir, filename=filename, head=head,
                       format=format, force_download=force_download,
                       local_files_only=local_files_only, quiet=quiet)
    ck = load_checkpoint(path, format=format, head=head)
    if ck.config is None:
        raise ValueError(f"{path} embeds no config; save it with save_pretrained(model, "
                         f"directory, config=cfg) first")
    card = ck.card or get_card(str(source))
    _license_notice(card)
    model = build_potential(ck.config, ck.state_dict, dtype=dtype, dispersion=dispersion,
                            compute_forces=compute_forces, compute_stress=compute_stress,
                            eeq_reuse=eeq_reuse, model_options=model_options,
                            label=str(source)).to(device)
    set_use_fast(model, use_fast)
    cutoff = float(getattr(model.model, "cutoff", ck.config.model.cutoff))
    return PretrainedModel(model, ck.config, card, cutoff, Path(path))


def from_pretrained(source: Union[str, Path], *, wrap: bool = True, **kwargs: Any) -> nn.Module:
    """Load a pre-trained model in one call.

    MACE foundation models, xnn-trained models, Zenodo uploads and local
    checkpoints all load the same way::

        from xnn.common.models import from_pretrained

        model = from_pretrained("mace-off23-small")             # registry name
        model = from_pretrained("mace-mh-0", head="omat_pbe")   # one head
        model = from_pretrained("doi:10.5281/zenodo.18957344",
                                filename="mace_csfapbbri_al_5_1_stagetwo.model")
        model = from_pretrained("runs/exp/best.pt")             # trainer checkpoint
        model = from_pretrained("mace-off23-small", cache_dir="/scratch/models")

    The first call downloads (and verifies) the file into ``cache_dir`` and,
    for a foreign format such as ``mace-torch``, converts it once; later
    calls load the cached portable directory without network access and
    without the upstream package.

    Parameters
    ----------
    source : str or pathlib.Path
        Registry name, local directory or checkpoint, URL, or Zenodo DOI;
        see :func:`fetch_model`.
    wrap : bool, optional
        Return the :class:`~xnn.common.models.ForceStressOutput` wrapper
        (forces, and stress with ``compute_stress=True``). ``False`` returns
        the bare potential inside it. Defaults to ``True``.
    **kwargs
        ``cache_dir``, ``filename``, ``head``, ``format``,
        ``force_download``, ``local_files_only``, ``quiet``, ``device``,
        ``dtype``, ``dispersion``, ``compute_forces``, ``compute_stress``,
        ``eeq_reuse``, ``use_fast``, ``model_options``; see
        :func:`load_pretrained`.

    Returns
    -------
    torch.nn.Module
        The model in eval mode.
    """
    model = load_pretrained(source, **kwargs).model
    return model if wrap else model.model
