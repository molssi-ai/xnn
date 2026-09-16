"""Caching downloader for the dataset hub.

Small stdlib-only helpers (no new dependency) that fetch remote files into a
local cache, verify them by MD5, and unpack archives. Downloads are written to
a temporary file and atomically renamed on success, so an interrupted transfer
never leaves a half-written file that a later run would trust.

The default cache location is the repository's ``datasets/`` directory, so
downloaded and processed data live alongside the project. Override it with the
``XNN_DATASETS`` (or ``XNN_CACHE``) environment variable, or per call with the
``cache_dir`` argument of :func:`~xnn.common.data.hub.load_dataset`.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path
from typing import Optional

from tqdm.auto import tqdm


def _repo_datasets_dir() -> Path:
    """Return the repository's ``datasets/`` directory.

    Resolved relative to this file (``src/xnn/common/data/hub/_download.py``),
    so it points at ``<repo>/datasets`` in a source checkout or editable
    install.

    Returns
    -------
    pathlib.Path
        Absolute path to ``<repo>/datasets``.
    """
    return Path(__file__).resolve().parents[5] / "datasets"


def default_cache_dir() -> Path:
    """Return the base directory under which datasets are cached.

    Honors the ``XNN_DATASETS`` then ``XNN_CACHE`` environment variables (both
    ``~`` expanded); otherwise falls back to the repository's ``datasets/``
    directory (see :func:`_repo_datasets_dir`).

    Returns
    -------
    pathlib.Path
        Base cache directory. Each dataset stores its files in a
        ``<cache_dir>/<dataset_name>/`` subfolder.
    """
    for var in ("XNN_DATASETS", "XNN_CACHE"):
        v = os.environ.get(var)
        if v:
            return Path(v).expanduser()
    return _repo_datasets_dir()


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    """Compute the MD5 hex digest of a file.

    Parameters
    ----------
    path : pathlib.Path
        File to hash.
    chunk : int, optional
        Read block size in bytes. Defaults to 1 MiB.

    Returns
    -------
    str
        Lower-case hexadecimal MD5 digest.
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download_file(url: str, dest: Path, md5: Optional[str] = None,
                  quiet: bool = False, chunk: int = 1 << 20) -> Path:
    """Download ``url`` to ``dest``, caching and verifying by MD5.

    If ``dest`` already exists and (when ``md5`` is given) matches, the download
    is skipped. The transfer streams to a ``.tmp`` sibling and is atomically
    renamed on success; on an MD5 mismatch the temporary file is removed and
    :class:`ValueError` is raised.

    Parameters
    ----------
    url : str
        Source URL.
    dest : pathlib.Path
        Destination path; parent directories are created as needed.
    md5 : str, optional
        Expected MD5 digest. When provided, a cached file is re-used only if it
        matches, and a fresh download is verified against it.
    quiet : bool, optional
        Suppress the progress bar. Defaults to ``False``.
    chunk : int, optional
        Read/write block size in bytes. Defaults to 1 MiB.

    Returns
    -------
    pathlib.Path
        The path to the downloaded (or already-cached) file, ``dest``.

    Raises
    ------
    ValueError
        If the freshly downloaded file's MD5 does not match ``md5``.
    """
    dest = Path(dest)
    if dest.exists() and (md5 is None or md5sum(dest) == md5):
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": "xnn"})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        bar = tqdm(total=total or None, unit="B", unit_scale=True,
                   unit_divisor=1024, desc=dest.name, disable=quiet,
                   leave=False)
        with bar:
            while True:
                block = resp.read(chunk)
                if not block:
                    break
                f.write(block)
                bar.update(len(block))

    if md5 is not None:
        got = md5sum(tmp)
        if got != md5:
            tmp.unlink(missing_ok=True)
            raise ValueError(
                f"MD5 mismatch for {dest.name}: expected {md5}, got {got}")
    tmp.replace(dest)
    return dest


def extract_archive(path: Path, dest_dir: Path) -> Path:
    """Extract a ``.zip`` or ``.tar(.gz/.bz2/.xz)`` archive.

    Parameters
    ----------
    path : pathlib.Path
        Archive file to extract.
    dest_dir : pathlib.Path
        Directory to extract into; created if missing.

    Returns
    -------
    pathlib.Path
        ``dest_dir``.

    Raises
    ------
    ValueError
        If the archive type is not recognized.
    """
    import tarfile
    import zipfile

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            z.extractall(dest_dir)
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as t:
            t.extractall(dest_dir)
    else:
        raise ValueError(f"unrecognized archive format: {path}")
    return dest_dir
