"""Resolve Zenodo DOIs and record links to downloadable files.

Accepted spellings of a Zenodo reference::

    doi:10.5281/zenodo.18957344
    10.5281/zenodo.18957344
    https://doi.org/10.5281/zenodo.18957344
    https://zenodo.org/records/18957344
    https://zenodo.org/records/18957344/files/<file>     (selects that file)
    zenodo:18957344

The record metadata comes from the public records API
(``https://zenodo.org/api/records/<id>``), which also gives each file's MD5,
so every download is verified. A concept DOI (the "all versions" DOI)
resolves to the latest version.
"""
from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

API = "https://zenodo.org/api/records/{recid}"
RECORD_FILE = "record.json"

_DOI = re.compile(
    r"^(?:doi:|https?://(?:dx\.)?doi\.org/|https?://(?:www\.)?zenodo\.org/doi/)?"
    r"10\.5281/zenodo\.(\d+)/?$", re.IGNORECASE)
_RECORD = re.compile(
    r"^https?://(?:www\.)?zenodo\.org/(?:records?|api/records)/(\d+)"
    r"(?:/files/([^/?#]+)(?:/content)?)?/?(?:[?#].*)?$", re.IGNORECASE)
_SHORT = re.compile(r"^zenodo:(\d+)$", re.IGNORECASE)
_ANY_DOI = re.compile(r"^(?:doi:|https?://(?:dx\.)?doi\.org/)?10\.\d{4,9}/\S+$",
                      re.IGNORECASE)


def parse(source: str) -> Optional[tuple[str, Optional[str]]]:
    """Recognize a Zenodo reference.

    Parameters
    ----------
    source : str
        Candidate reference.

    Returns
    -------
    tuple of (str, str or None) or None
        ``(record_id, filename)`` (the file named by a ``/files/`` link, else
        ``None``), or ``None`` when ``source`` is not a Zenodo reference.
    """
    s = str(source).strip()
    for pattern in (_DOI, _SHORT):
        m = pattern.match(s)
        if m:
            return m.group(1), None
    m = _RECORD.match(s)
    if m:
        name = urllib.parse.unquote(m.group(2)) if m.group(2) else None
        return m.group(1), name
    return None


def is_doi(source: str) -> bool:
    """Whether ``source`` looks like a DOI of any registrant."""
    return bool(_ANY_DOI.match(str(source).strip()))


def fetch_record(recid: str, root: Path, *, refresh: bool = False,
                 offline: bool = False) -> dict[str, Any]:
    """Return a record's metadata, caching it as ``<root>/record.json``.

    Parameters
    ----------
    recid : str
        Zenodo record id.
    root : pathlib.Path
        Cache directory of this record.
    refresh : bool, optional
        Ask the API again even when the metadata is cached.
    offline : bool, optional
        Never touch the network; the metadata must be cached already.

    Returns
    -------
    dict
        The record JSON.

    Raises
    ------
    FileNotFoundError
        Offline with no cached metadata.
    """
    path = root / RECORD_FILE
    if path.is_file() and (offline or not refresh):
        return json.loads(path.read_text())
    if offline:
        raise FileNotFoundError(
            f"Zenodo record {recid} is not cached in {root} and network access "
            f"is disabled (local_files_only=True or XNN_OFFLINE)")
    req = urllib.request.Request(API.format(recid=recid),
                                 headers={"User-Agent": "xnn",
                                          "Accept": "application/json"})
    with urllib.request.urlopen(req) as resp:
        record = json.loads(resp.read().decode())
    root.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(record, indent=1))
    tmp.replace(path)
    return record


def files(record: dict[str, Any]) -> list[dict[str, Any]]:
    """The downloadable files of a record.

    Parameters
    ----------
    record : dict
        Record JSON from :func:`fetch_record`.

    Returns
    -------
    list of dict
        One ``{"key", "url", "md5", "size"}`` mapping per file.
    """
    out = []
    for f in record.get("files") or []:
        key = f.get("key") or f.get("filename")
        links = f.get("links") or {}
        url = links.get("self") or links.get("content") or links.get("download")
        checksum = str(f.get("checksum") or "")
        md5 = checksum.split(":", 1)[1] if checksum.startswith("md5:") else (checksum or None)
        if key and url:
            out.append({"key": key, "url": url, "md5": md5, "size": f.get("size")})
    return out


def choose(record_files: list[dict[str, Any]], filename: Optional[str],
           suffixes: tuple[str, ...], recid: str) -> dict[str, Any]:
    """Pick the model file of a record.

    Parameters
    ----------
    record_files : list of dict
        Output of :func:`files`.
    filename : str or None
        Requested file name; ``None`` picks the only file with a model or
        archive suffix.
    suffixes : tuple of str
        Suffixes that mark a model file or an archive.
    recid : str
        Record id, for error messages.

    Returns
    -------
    dict
        The chosen file entry.

    Raises
    ------
    FileNotFoundError
        If ``filename`` is not in the record, or nothing looks like a model.
    ValueError
        If several files could be the model and no ``filename`` was given.
    """
    keys = [f["key"] for f in record_files]
    if filename is not None:
        for f in record_files:
            if f["key"] == filename:
                return f
        raise FileNotFoundError(
            f"Zenodo record {recid} has no file {filename!r}; it has {keys}")
    candidates = [f for f in record_files if f["key"].lower().endswith(suffixes)]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"no model file in Zenodo record {recid} (files: {keys})")
    raise ValueError(
        f"Zenodo record {recid} holds several model files; pass filename= with "
        f"one of {[f['key'] for f in candidates]}")


def citation(record: dict[str, Any]) -> Optional[str]:
    """A one-line citation built from a record's metadata."""
    meta = record.get("metadata") or {}
    names = [c.get("name", "") for c in meta.get("creators") or []]
    if not names:
        return None
    authors = names[0] + (" et al." if len(names) > 2 else
                          f" and {names[1]}" if len(names) == 2 else "")
    year = str(meta.get("publication_date", ""))[:4]
    doi = record.get("doi") or meta.get("doi")
    parts = [f"{authors} ({year})." if year else f"{authors}.",
             f"{meta.get('title', '').strip()}.", "Zenodo."]
    if doi:
        parts.append(f"https://doi.org/{doi}")
    return " ".join(p for p in parts if p.strip(". "))


def license_id(record: dict[str, Any]) -> Optional[str]:
    """The record's license identifier (e.g. ``"cc-by-4.0"``)."""
    lic = (record.get("metadata") or {}).get("license")
    if isinstance(lic, dict):
        return lic.get("id")
    return lic
