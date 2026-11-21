"""The published PM3, PM6, PM6-ORG and PM7 parameter values, read from MOPAC.

xnn does not ship these values. On first use the data modules of the MOPAC
program that hold them (github.com/openmopac/mopac, release 23.2.5) are
downloaded, checked by MD5 and kept in the model cache under
``sqmnn/mopac-23.2.5`` (see
:func:`~xnn.common.models.hub.cache.default_model_cache_dir`); later loads read
the cached files. A ``LICENSE.txt`` beside them records where they came from
and the license of MOPAC. The values are taken from the ``data`` statements and
the diatomic assignments of those files as text; nothing in them is compiled
or executed.

With ``XNN_OFFLINE=1`` the files must already be in the cache. A parameter
set can always be given as a MOPAC ``EXTERNAL`` file instead (for example one
written by :func:`~xnn.sqmnn.common.parameters.write_parameters` on a machine
with network access).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

RELEASE = "v23.2.5"
_URL = "https://raw.githubusercontent.com/openmopac/mopac/{release}/src/models/{file}"
_MD5 = {
    "parameters_for_PM3_C.F90": "c8f05dbef7128a35761f154dbcbf22ae",
    "alpb_and_xfac_pm3.F90": "0d7b4b8ed4df1ab7a49d6e4f049e739b",
    "parameters_for_PM6_C.F90": "ba2a7a9b86ad725a7d43f09c31efb2f7",
    "parameters_for_PM6_ORG_C.F90": "9d6643a63dd863cd5bc1dc0d80676eda",
    "parameters_for_PM7_C.F90": "f699aaa5336c777b90bbdcb97a16e23b",
}
_NUMBER = r"([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[dDeE][-+]?\d+)?)"
# lanthanides are sparkles in MOPAC (La-Lu in PM3, Ce-Yb otherwise), not NDDO atoms
_LANTHANIDES = frozenset(range(58, 71))


@dataclass(frozen=True)
class _Source:
    """Where one method's values live in MOPAC's data modules."""

    element_file: str
    pair_file: str
    arrays: Tuple[Tuple[str, str], ...]   # (EXTERNAL name, Fortran array)
    gaussians: Tuple[str, str, str]       # amplitude, exponent, center arrays
    globals: Optional[str]
    exclude: frozenset


def _arrays(suffix: str, poc: str) -> Tuple[Tuple[str, str], ...]:
    from .parameters import ELEMENT_PARAMETERS
    return tuple((name, poc if name == "POC" else name.lower() + suffix)
                 for name in ELEMENT_PARAMETERS)


_SOURCES: Dict[str, _Source] = {
    "pm3": _Source(
        "parameters_for_PM3_C.F90", "alpb_and_xfac_pm3.F90",
        (("USS", "usspm3"), ("UPP", "upppm3"), ("ZS", "zspm3"), ("ZP", "zppm3"),
         ("BETAS", "betasp"), ("BETAP", "betapp"), ("ALP", "alppm3"), ("GSS", "gsspm3"),
         ("GSP", "gsppm3"), ("GPP", "gpppm3"), ("GP2", "gp2pm3"), ("HSP", "hsppm3")),
        ("guesp1", "guesp2", "guesp3"), None, _LANTHANIDES | {57, 71}),
    "pm6": _Source("parameters_for_PM6_C.F90", "parameters_for_PM6_C.F90", _arrays("6", "poc_6"),
                   ("gues61", "gues62", "gues63"), "v_par6", _LANTHANIDES),
    "pm7": _Source("parameters_for_PM7_C.F90", "parameters_for_PM7_C.F90", _arrays("7", "poc_7"),
                   ("gues71", "gues72", "gues73"), "v_par7", _LANTHANIDES),
    "pm6_org": _Source("parameters_for_PM6_ORG_C.F90", "parameters_for_PM6_ORG_C.F90",
                       _arrays("_org", "poc__org"),
                       ("gues_org1", "gues_org2", "gues_org3"), "v_par_org", frozenset()),
}
#: The methods whose published parameters can be loaded by name.
PUBLISHED = ("pm3", "pm6", "pm6-org", "pm7")


def _value(text: str) -> float:
    return float(text.replace("d", "e").replace("D", "e"))


def _data(text: str, array: str, rank: int = 1):
    """``(indices, value)`` of every ``data array(i[, j]) / value /`` statement."""
    index = r"\(\s*(\d+)\s*" + r",\s*(\d+)\s*" * (rank - 1) + r"\)"
    pattern = r"data\s+" + re.escape(array) + index + r"\s*/\s*" + _NUMBER + r"\s*/"
    for m in re.finditer(pattern, text, re.IGNORECASE):
        yield tuple(int(g) for g in m.groups()[:rank]), _value(m.group(rank + 1))


def source_file(file: str, cache_dir: Optional[Union[str, Path]] = None) -> Path:
    """One of MOPAC's data modules, downloaded into the cache if needed.

    Parameters
    ----------
    file : str
        The file name (a key of the pinned MD5 table).
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
    from ...common.models.hub.provenance import write_license_note

    dest = resolve_cache_dir(cache_dir) / "sqmnn" / f"mopac-{RELEASE.lstrip('v')}" / file
    note = (dest.parent, "MOPAC", RELEASE, "https://github.com/openmopac/mopac", "Apache-2.0",
            (f"https://raw.githubusercontent.com/openmopac/mopac/{RELEASE}/LICENSE",))
    if dest.exists() and md5sum(dest) == _MD5[file]:
        write_license_note(*note)
        return dest
    if offline():
        raise FileNotFoundError(
            f"{dest} is not cached and XNN_OFFLINE is set; load the method once with network "
            "access, or pass a MOPAC EXTERNAL parameter file")
    path = download_file(_URL.format(release=RELEASE, file=file), dest, _MD5[file], quiet=True)
    write_license_note(*note)
    return path


def published_parameters(method: str, cache_dir: Optional[Union[str, Path]] = None):
    """The published parameter set of a method, as MOPAC 23.2.5 uses it.

    Parameters
    ----------
    method : str
        ``"pm3"``, ``"pm6"``, ``"pm6-org"`` or ``"pm7"``.
    cache_dir : str or Path, optional
        Model cache; defaults to the xnn model cache.

    Returns
    -------
    ParameterSet
    """
    from .parameters import ParameterSet

    key = method.strip().lower().replace("-", "_")
    if key == "pm6org":
        key = "pm6_org"
    src = _SOURCES[key]
    text = source_file(src.element_file, cache_dir).read_text()
    elements: Dict[int, Dict[str, float]] = {}
    for name, array in src.arrays:
        for (z,), v in _data(text, array):
            elements.setdefault(z, {})[name] = v
    keep = {z for z, p in elements.items()
            if z <= 83 and z not in src.exclude
            and (p.get("ZS", 0.0) > 0.0 or p.get("GSS", 0.0) > 0.0)}
    out = ParameterSet(name=key.replace("_", "-"))
    for z in sorted(keep):
        out.elements[z] = {k: v for k, v in elements[z].items() if v != 0.0}
    for i, array in enumerate(src.gaussians, start=1):
        for (z, k), v in _data(text, array, rank=2):
            if z in keep and v != 0.0:
                out.elements[z][f"FN{i}{k}"] = v
    if src.globals:
        for (n,), v in _data(text, src.globals):
            if v != 0.0:
                out.globals[n] = v
    pairs = source_file(src.pair_file, cache_dir).read_text()
    for m in re.finditer(r"\b(alpb|xfac)\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*=\s*" + _NUMBER, pairs,
                         re.IGNORECASE):
        i, j, v = int(m.group(2)), int(m.group(3)), _value(m.group(4))
        if i in keep and j in keep and v != 0.0:
            out.pairs.setdefault((max(i, j), min(i, j)), {})[m.group(1).upper()] = v
    return out
