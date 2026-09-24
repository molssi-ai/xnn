#!/usr/bin/env python
"""Regenerate ``xnn/common/models/d4_reference.npz`` from the dftd4 sources.

The D4 dispersion model needs a set of *element-specific reference data*:
the TD-DFT dynamic polarizabilities of the reference systems, their
coordination numbers and EEQ partial charges, the partitioning factors of
paper eq 5, and the per-element constants (covalent radii, Pauling
electronegativities, effective nuclear charges, chemical hardnesses, the
``<r^4>/<r^2>`` expectation values and the EEQ parameters). These numbers are
distributed as Fortran ``data`` statements inside the
`dftd4 <https://github.com/dftd4/dftd4>`_, `multicharge
<https://github.com/grimme-lab/multicharge>`_ and `mctc-lib
<https://github.com/grimme-lab/mctc-lib>`_ source trees. This script reads
them and stores the *values* (nothing else) in one compressed NumPy archive
that ships with xnn, so the package needs neither a Fortran toolchain nor the
upstream library at run time.

Usage::

    python tools/build_d4_reference.py --dftd4 /path/to/dftd4 \
        --multicharge /path/to/multicharge --mctc /path/to/mctc-lib \
        --out src/xnn/common/models/d4_reference.npz

Every array is stored 0-based and padded to ``Z = 0..118`` (index = atomic
number, row 0 unused) with the reference axis padded to 7 slots. Values are
copied verbatim (as decimal strings -> float64); nothing is rescaled here
except the two derived integer tables ``nref`` and ``ngw`` (the number of
reference systems and of Gaussian weights per reference, paper eq 8),
which upstream derives from the same numbers at start-up.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np

MAX_Z = 118
MAX_REF = 7
N_FREQ = 23
MAX_SEC = 17
MAX_CN_BIN = 19


# Fortran parsing helpers
def _read_joined(path: Path) -> str:
    """File contents with comments dropped and ``&`` continuations joined."""
    out = []
    for raw in path.read_text().splitlines():
        line = raw.split("!", 1)[0] if not raw.lstrip().startswith("!") else ""
        out.append(line)
    text = "\n".join(out)
    # join continuation lines: trailing '&' [newline] optional leading '&'
    text = re.sub(r"&\s*\n\s*&?", " ", text)
    return text


def _numbers(blob: str) -> list[float]:
    blob = blob.replace("_wp", "")
    return [float(tok) for tok in re.split(r"[,\s]+", blob.strip()) if tok]


def parse_data_statements(text: str) -> dict[str, list[tuple[tuple[int, ...], list[float]]]]:
    """All ``data NAME(idx) / values /`` statements, grouped by lower-case NAME.

    The index tuple keeps the Fortran (1-based) integer indices, with ``:``
    slices dropped (they always address the leading frequency axis here).
    """
    pattern = re.compile(
        r"data\s+([A-Za-z0-9_]+)\s*\(([^)]*)\)\s*/([^/]*)/", re.IGNORECASE)
    found: dict[str, list] = {}
    for m in pattern.finditer(text):
        name = m.group(1).lower()
        idx = tuple(int(i) for i in m.group(2).split(",") if i.strip() != ":")
        found.setdefault(name, []).append((idx, _numbers(m.group(3))))
    return found


def parse_parameter_array(text: str, name: str) -> list[float]:
    """The values of ``real(wp), parameter :: NAME(...) = [ ... ]``."""
    m = re.search(
        rf"::\s*{name}\s*\([^)]*\)\s*=\s*(?:[A-Za-z_]+\s*\*\s*)?\[(.*?)\]",
        text, re.IGNORECASE | re.DOTALL)
    if m is None:
        raise KeyError(f"array {name!r} not found")
    return _numbers(m.group(1))


def _pad(values: list[float], n: int, fill: float = 0.0) -> np.ndarray:
    out = np.full(n + 1, fill, dtype=np.float64)   # index 0 unused
    out[1:len(values) + 1] = values
    return out


def _nint(x: float) -> int:
    """Fortran NINT: round half away from zero."""
    return int(math.floor(abs(x) + 0.5)) * (1 if x >= 0 else -1)


# build
def build(dftd4: Path, multicharge: Path, mctc: Path) -> dict[str, np.ndarray]:
    src = dftd4 / "src" / "dftd4"
    ref = parse_data_statements(_read_joined(src / "reference.inc"))

    def table(name: str, dtype=np.float64) -> np.ndarray:
        arr = np.zeros((MAX_REF, MAX_Z + 1), dtype=dtype)
        for (r, z), vals in ref[name]:
            arr[r - 1, z] = vals[0]
        return arr

    nref = np.zeros(MAX_Z + 1, dtype=np.int64)
    for (z,), vals in ref["refn"]:
        nref[z] = int(vals[0])

    alphaiw = np.zeros((N_FREQ, MAX_REF, MAX_Z + 1))
    for (r, z), vals in ref["alphaiw"]:
        assert len(vals) == N_FREQ
        alphaiw[:, r - 1, z] = vals

    secaiw = np.zeros((N_FREQ, MAX_SEC + 1))
    sscale = np.zeros(MAX_SEC + 1)
    for (i,), vals in ref["secaiw"]:
        secaiw[:, i] = vals
    for (i,), vals in ref["sscale"]:
        sscale[i] = vals[0]

    refcn = table("refcn")           # D3-type CN, only used to derive ngw
    refcovcn = table("refcovcn")     # the D4 (EN-weighted) reference CN
    refsys = table("refsys", dtype=np.int64)

    # number of Gaussian weights per reference (paper eq 8, N^s): references
    # whose D3-type CN rounds to the same integer share a bin; the bin
    # for CN = 0 starts at one occupant
    ngw = np.ones((MAX_REF, MAX_Z + 1), dtype=np.int64)
    for z in range(1, MAX_Z + 1):
        counts = [1] + [0] * MAX_CN_BIN
        for r in range(nref[z]):
            counts[min(_nint(refcn[r, z]), MAX_CN_BIN)] += 1
        for r in range(nref[z]):
            n = counts[min(_nint(refcn[r, z]), MAX_CN_BIN)]
            ngw[r, z] = n * (n + 1) // 2

    data_dir = src / "data"
    covrad_aa = parse_parameter_array(_read_joined(data_dir / "covrad.f90"),
                                      "covalent_rad_2009")
    pauling_en = parse_parameter_array(_read_joined(data_dir / "en.f90"),
                                       "pauling_en")
    zeff = parse_parameter_array(_read_joined(data_dir / "zeff.f90"),
                                 "effective_nuclear_charge")
    hardness = parse_parameter_array(_read_joined(data_dir / "hardness.f90"),
                                     "chemical_hardness")
    r4r2_raw = parse_parameter_array(_read_joined(data_dir / "r4r2.f90"),
                                     "r4_over_r2")
    eeq = _read_joined(multicharge / "src" / "multicharge" / "param" / "eeq2019.f90")
    eeq_chi = parse_parameter_array(eeq, "eeq_chi")
    eeq_eta = parse_parameter_array(eeq, "eeq_eta")
    eeq_kcnchi = parse_parameter_array(eeq, "eeq_kcnchi")
    eeq_rad = parse_parameter_array(eeq, "eeq_rad")
    # the EEQ coordination number uses mctc-lib's covalent radii, which are
    # the same 2009 table; keep them separately anyway so the provenance of
    # both tables is explicit
    mctc_covrad_aa = parse_parameter_array(
        _read_joined(mctc / "src" / "mctc" / "data" / "covrad.f90"),
        "covalent_rad_2009")
    assert np.allclose(covrad_aa, mctc_covrad_aa)

    for name, arr in [("covrad", covrad_aa), ("en", pauling_en), ("zeff", zeff),
                      ("hardness", hardness), ("r4r2", r4r2_raw)]:
        assert len(arr) == MAX_Z, (name, len(arr))
    for name, arr in [("chi", eeq_chi), ("eta", eeq_eta),
                      ("kcnchi", eeq_kcnchi), ("rad", eeq_rad)]:
        assert len(arr) == 103, (name, len(arr))

    return {
        "nref": nref,
        "ngw": ngw,
        "refcn": refcovcn,                    # reference CN entering eq 8
        "refcn_d3": refcn,                    # only for provenance / ngw
        "refq": table("clsq"),                # EEQ charge of A in the reference
        "refh": table("clsh"),                # EEQ charge of X in the reference
        "hcount": table("hcount"),            # n / m of eq 5
        "ascale": table("ascale"),            # 1 / m of eq 5
        "refsys": refsys,                     # secondary system (X) index
        "alphaiw": alphaiw,                   # alpha^{AmXn}(i omega), 23 freqs
        "secaiw": secaiw,                     # alpha^{X}(i omega) of X_l
        "sscale": sscale,                     # 1 / l of eq 5
        "covalent_radius_aa": _pad(covrad_aa, MAX_Z),   # Angstrom (Pyykko 2009)
        "pauling_en": _pad(pauling_en, MAX_Z),
        "zeff": _pad(zeff, MAX_Z),
        "hardness": _pad(hardness, MAX_Z),
        "r4r2_raw": _pad(r4r2_raw, MAX_Z),     # <r4>/<r2>, before sqrt(0.5 sqrt(Z) .)
        "eeq_chi": _pad(eeq_chi, MAX_Z, np.nan),
        "eeq_eta": _pad(eeq_eta, MAX_Z, np.nan),
        "eeq_kcnchi": _pad(eeq_kcnchi, MAX_Z, np.nan),
        "eeq_rad": _pad(eeq_rad, MAX_Z, np.nan),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dftd4", required=True, type=Path)
    p.add_argument("--multicharge", required=True, type=Path)
    p.add_argument("--mctc", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    tables = build(args.dftd4, args.multicharge, args.mctc)
    np.savez_compressed(args.out, **tables)
    n_ref = int(tables["nref"].sum())
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} kB): "
          f"{(tables['nref'] > 0).sum()} elements, {n_ref} reference systems")


if __name__ == "__main__":
    main()
