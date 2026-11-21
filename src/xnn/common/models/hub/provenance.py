"""A provenance note next to files of other programs downloaded into the model cache.

xnn does not ship the parameters of other programs. It downloads them on first
use and writes ``LICENSE.txt`` beside them, which names the program, the
release, the source of the files and the license they are distributed under.
The first time, it also warns the user that using the files means accepting that license.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

NOTE = "LICENSE.txt"


class ParameterLicenseWarning(UserWarning):
    """Parameter files of another program were placed in the cache: its license applies to them."""


def write_license_note(folder: Path, program: str, release: str, source: str, license_name: str,
                       license_urls: Sequence[str]) -> Path:
    """Write the provenance note of a cache folder, unless it is already there with the same text;
    the first time, warn that the license of the program applies to the files.

    Parameters
    ----------
    folder : Path
        The cache folder that holds the downloaded files.
    program, release : str
        The program the files belong to and its release.
    source : str
        Where the files were downloaded from.
    license_name : str
        Identifier of the license of the program.
    license_urls : sequence of str
        Where the full license text is found.

    Returns
    -------
    Path
        The note.
    """
    text = "\n".join([
        f"The files in this folder are parameter files of {program} {release}.",
        f"xnn downloaded them from {source}",
        "on first use. They are not part of xnn and are not covered by its license.",
        f"{program} is distributed under {license_name}; the full text is at",
        *[f"  {url}" for url in license_urls],
        ""])
    path = Path(folder) / NOTE
    try:
        first = not path.exists()
        if first or path.read_text() != text:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    except OSError:  # a read-only cache: the files are usable without the note
        return path
    if first:
        warnings.warn(f"Parameter files of {program} {release} were downloaded from {source} into {path.parent}. "
                      f"They are distributed under {license_name}; by using them you accept its terms (see {path}).",
                      ParameterLicenseWarning, stacklevel=3)
    return path
