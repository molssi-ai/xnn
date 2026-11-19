"""The published GFN1-xTB and GFN2-xTB parameters, read from the xtb program's parameter files.

xnn does not ship these values. On first use the parameter files of the xtb
program (github.com/grimme-lab/xtb, release 6.8.0) are downloaded, checked by
MD5 and kept in the model cache under ``sqmnn/xtb-6.8.0``; later loads read the
cached files. A ``LICENSE.txt`` beside them records where they came from and
the license of xtb. With ``XNN_OFFLINE=1`` the files must already be cached. A
parameter file in the same format can be passed instead.

The format has a ``$globpar`` block of global parameters, a ``$pairpar`` block
of element-pair scaling factors and one ``$Z=`` block per element with the
shells (``ao=``), energy levels (``lev=``), Slater exponents (``exp=``) and the
element's scalars.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

RELEASE = "v6.8.0"
_URL = "https://raw.githubusercontent.com/grimme-lab/xtb/{release}/{file}"
_REPOSITORY = "https://github.com/grimme-lab/xtb"
_LICENSES = (_URL.format(release=RELEASE, file="COPYING"), _URL.format(release=RELEASE, file="COPYING.LESSER"))
_FILES = {"gfn0": "param_gfn0-xtb.txt", "gfn1": "param_gfn1-xtb.txt", "gfn2": "param_gfn2-xtb.txt"}
_MD5 = {"param_gfn0-xtb.txt": "96f45ffa716e8b6749566af213585d46",
        "param_gfn1-xtb.txt": "d8e911d0281710021909e3c6f6598190",
        "param_gfn2-xtb.txt": "22128a9d8a46700842afaa73089a47c5"}


@dataclass
class XTBParameters:
    """One parameter file: global values, pair factors and element blocks."""

    name: str
    globals: Dict[str, float] = field(default_factory=dict)
    pairs: Dict[Tuple[int, int], float] = field(default_factory=dict)
    shells: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)   # z -> [(n, l)]
    elements: Dict[int, Dict[str, List[float]]] = field(default_factory=dict)
    source: str = field(default="", repr=False, compare=False)                 # the text the set was parsed from
    _values: Dict[tuple, float] = field(default_factory=dict, repr=False, compare=False)

    def value(self, z: int, key: str, index: int = 0, default: float = 0.0) -> float:
        """One element value (``index`` selects the shell for shell-resolved entries); the values
        are read once, the parsed set is not meant to change."""
        at = (z, key, index, default)
        if at not in self._values:
            v = self.elements.get(z, {}).get(key)
            self._values[at] = float(v[index]) if v is not None and index < len(v) else default
        return self._values[at]


def parse_parameters(text: str, name: str = "") -> XTBParameters:
    """Parse the text of an xtb parameter file."""
    out = XTBParameters(name=name, source=text)
    for block, body in re.findall(r"^\$(globpar|pairpar)\s*\n(.*?)^\$end", text, re.M | re.S):
        for line in body.splitlines():
            t = line.split()
            if block == "globpar" and len(t) >= 2:
                out.globals[t[0].lower()] = float(t[1])
            elif block == "pairpar" and len(t) >= 3:
                i, j = int(t[0]), int(t[1])
                out.pairs[(max(i, j), min(i, j))] = float(t[2])
    lmap = {"s": 0, "p": 1, "d": 2, "f": 3}
    for z, body in re.findall(r"^\$Z=\s*(\d+)[^\n]*\n(.*?)^\$end", text, re.M | re.S):
        z = int(z)
        values: Dict[str, List[float]] = {}
        for key, rest in re.findall(r"^\s*(\w+)=(.*)$", body, re.M):
            if key == "ao":
                out.shells[z] = [(int(n), lmap[l]) for n, l in re.findall(r"(\d)([spdf])", rest)]
            else:
                values[key.upper()] = [float(x) for x in rest.split()]
        out.elements[z] = values
    return out


def write_parameters(par: XTBParameters, values: Dict[Tuple[int, str], List[float]], digits: int = 10,
                     globals: Optional[Dict[str, float]] = None,
                     pairs: Optional[Dict[Tuple[int, int], float]] = None) -> str:
    """The text of a parameter file with entries replaced.

    Parameters
    ----------
    par : XTBParameters
        A parsed set (it keeps the text it came from).
    values : dict
        ``(atomic number, key) -> values`` of the element entries to replace, with the keys of the
        file (``LEV``, ``GAM``, ``KCNS``, ...); entries a block does not have are skipped.
    digits : int
        Decimal places of the written numbers.
    globals : dict, optional
        Global parameters to replace, by name.
    pairs : dict, optional
        The element-pair factors ``(i, j) -> value`` (``i >= j``); they replace the pair block.

    Returns
    -------
    str
        The file text, which the xtb program reads as well (``--vparam``).
    """
    def number(v: float) -> str:
        return f"{v:{digits + 6}.{digits}f}"

    def block(match):
        z, body = int(match.group(2)), match.group(3)
        for (zz, key), new in values.items():
            if zz == z:
                line = " ".join(number(v) for v in new)
                body = re.sub(rf"^(\s*{key}=).*$", lambda m, line=line: m.group(1) + line, body, flags=re.M | re.I)
        return match.group(1) + body + match.group(4)

    def global_block(match):
        body = match.group(2)
        for key, v in (globals or {}).items():
            body = re.sub(rf"^(\s*{re.escape(key)}\s+)\S+", lambda m, v=v: m.group(1) + number(v).strip(), body,
                          flags=re.M | re.I)
        return match.group(1) + body + match.group(3)

    def pair_block(match):
        body = "".join(f"{i:4d}{j:4d} {number(v)}\n" for (i, j), v in sorted(pairs.items()))
        return match.group(1) + body + match.group(3)

    text = re.sub(r"^(\$Z=\s*(\d+)[^\n]*\n)(.*?)(^\$end)", block, par.source, flags=re.M | re.S)
    if globals:
        text = re.sub(r"^(\$globpar[^\n]*\n)(.*?)(^\$end)", global_block, text, flags=re.M | re.S)
    if pairs:
        text = re.sub(r"^(\$pairpar[^\n]*\n)(.*?)(^\$end)", pair_block, text, flags=re.M | re.S)
    return text


def parameter_file(method: str, cache_dir: Optional[Union[str, Path]] = None) -> Path:
    """The xtb parameter file of a method, downloaded into the model cache if needed.

    Parameters
    ----------
    method : str
        ``"gfn1"`` or ``"gfn2"``.
    cache_dir : str or Path, optional
        Model cache; defaults to the xnn model cache.

    Returns
    -------
    Path
        The cached file.

    Raises
    ------
    FileNotFoundError
        When the file is not cached and ``XNN_OFFLINE`` forbids downloading.
    """
    from ...common.data.hub._download import download_file, md5sum
    from ...common.models.hub.cache import offline, resolve_cache_dir
    from .provenance import write_license_note

    file = _FILES[method]
    dest = resolve_cache_dir(cache_dir) / "sqmnn" / f"xtb-{RELEASE.lstrip('v')}" / file
    if dest.exists() and md5sum(dest) == _MD5[file]:
        write_license_note(dest.parent, "xtb", RELEASE, _REPOSITORY, "LGPL-3.0-or-later", _LICENSES)
        return dest
    if offline():
        raise FileNotFoundError(
            f"{dest} is not cached and XNN_OFFLINE is set; load the method once with network "
            "access, or pass a parameter file")
    path = download_file(_URL.format(release=RELEASE, file=file), dest, _MD5[file], quiet=True)
    write_license_note(dest.parent, "xtb", RELEASE, _REPOSITORY, "LGPL-3.0-or-later", _LICENSES)
    return path


def published_parameters(method: str, cache_dir: Optional[Union[str, Path]] = None) -> XTBParameters:
    """The published parameters of ``"gfn0"``, ``"gfn1"`` or ``"gfn2"``, as xtb 6.8.0 uses them."""
    key = method.strip().lower().replace("-xtb", "").replace("_xtb", "")
    return parse_parameters(parameter_file(key, cache_dir).read_text(), key)
