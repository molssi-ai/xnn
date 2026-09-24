#!/usr/bin/env python
"""Regenerate ``xnn/common/models/d3_reference.npz`` from the simple-dftd3 sources.

The D3 dispersion model needs the reference ``C6`` coefficients of Grimme's
2010 parametrization (TD-DFT values for up to seven reference systems per
element, tabulated as pair-wise ``C6(ref_i, ref_j)`` with the coordination
numbers of the reference systems), the pair cutoff radii ``R0^AB`` of the
zero-damping function (paper sec II.D, 4465 pairs, extended to Lr), the
``<r^4>/<r^2>`` expectation values and the covalent radii of the coordination
number. These numbers are distributed as Fortran ``parameter`` arrays inside
the `simple-dftd3 <https://github.com/dftd3/simple-dftd3>`_
and `mctc-lib <https://github.com/grimme-lab/mctc-lib>`_
source trees. This script reads them and stores the *values* in one compressed
NumPy archive that ships with xnn.

simple-dftd3 1.1.0 (July 2024) re-parametrized the actinides Fr-Pu and added
Am-Lr. Passing a checkout of an earlier release (``--sdftd3-original``, e.g.
v1.0.0) additionally stores Grimme's original reference systems for the
elements whose data changed, so both reference sets can be selected at run
time (``references="2010"`` keeps the values of the original D3 codes, e.g.
PhysNet's; ``references="2024"`` follows the current reference code).

Usage::

    python tools/build_d3_reference.py --sdftd3 /path/to/simple-dftd3 \
        --sdftd3-original /path/to/simple-dftd3-1.0.0 \
        --mctc /path/to/mctc-lib --out src/xnn/common/models/d3_reference.npz

All arrays are stored 0-based and padded to ``Z = 0..103`` (index = atomic
number, row 0 unused). ``c6`` has shape ``(7, 7, 104, 104)`` indexed
``[ref_i, ref_j, Z_i, Z_j]`` and is symmetric under the simultaneous swap of
both index pairs; unused reference slots hold zero and are marked by a
negative reference coordination number. The original references are stored
as patches for the ``n`` differing elements ``original_z``: ``original_nref``
``(n,)``, ``original_refcn`` ``(7, n)`` and ``original_c6`` ``(7, 7, n, 104)``
indexed ``[ref_i, ref_j, k, Z_j]`` (``k`` enumerates ``original_z``), plus
``original_max_z``, the last element the earlier release parametrized.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

MAX_Z = 103
MAX_REF = 7


def _numbers(blob: str) -> list[float]:
    blob = re.sub(r"_wp\b", "", blob)
    return [float(x) for x in re.findall(r"[-+]?\d+\.?\d*(?:[eEdD][-+]?\d+)?", blob)]


def _read_code(path: Path) -> str:
    text = path.read_text()
    # drop Fortran comments (``!`` outside of strings; the data files have none)
    lines = [ln.split("!", 1)[0] for ln in text.splitlines()]
    return "\n".join(lines)


def parse_parameter_array(text: str, name: str) -> list[float]:
    m = re.search(rf"::\s*{name}\s*\((?:[^()]|\([^()]*\))*\)\s*=\s*(?:reshape\(\s*)?(?:[A-Za-z_]+\s*\*\s*)?\[(.*?)\]",
                  text, re.IGNORECASE | re.DOTALL)
    if m is None:
        raise KeyError(name)
    return _numbers(m.group(1))


def _parameter_int(text: str, name: str) -> int:
    m = re.search(rf"::\s*{name}\s*=\s*(\d+)", text)
    if m is None:
        raise KeyError(name)
    return int(m.group(1))


def parse_reference_systems(sdftd3: Path) -> dict[str, np.ndarray]:
    """Reference counts, coordination numbers and C6 tables of one release.

    The release's own ``max_elem`` / ``max_ref`` are read from the source;
    the arrays are padded to ``(MAX_REF, MAX_Z + 1)`` and
    ``(MAX_REF, MAX_REF, MAX_Z + 1, MAX_Z + 1)``.
    """
    ref = _read_code(sdftd3 / "src" / "dftd3" / "reference.f90")
    max_elem = _parameter_int(ref, "max_elem")
    max_ref = _parameter_int(ref, "max_ref")
    assert max_elem <= MAX_Z and max_ref <= MAX_REF, (max_elem, max_ref)

    nref_list = parse_parameter_array(ref, "number_of_references")
    assert len(nref_list) == max_elem
    nref = np.zeros(MAX_Z + 1, dtype=np.int64)
    nref[1:max_elem + 1] = np.asarray(nref_list, dtype=int)

    refcn_flat = parse_parameter_array(ref, "reference_cn")
    assert len(refcn_flat) == max_ref * max_elem
    refcn = np.full((MAX_REF, MAX_Z + 1), -1.0)
    refcn[:max_ref, 1:max_elem + 1] = np.asarray(refcn_flat).reshape(max_elem, max_ref).T
    # unused slots hold release-dependent fillers; normalize them to -1
    refcn[np.arange(MAX_REF)[:, None] >= nref[None, :]] = -1.0

    # the C6 reference table is filled chunk-wise into a flat view of the
    # (max_ref, max_ref, npairs) array, column-major
    npairs = max_elem * (max_elem + 1) // 2
    flat = np.zeros(max_ref * max_ref * npairs)
    # chunk bounds are integer literals (>= 1.1.0) or expressions in max_ref
    # such as ``19 * max_ref * max_ref + 1`` (<= 1.0.0)
    def bound(expr: str) -> int:
        expr = expr.replace("max_ref", str(max_ref))
        assert re.fullmatch(r"[\d\s*+]+", expr), expr
        return int(eval(expr))  # noqa: S307 - digits, '*' and '+' only

    for m in re.finditer(r"c6ab_view\(([^:()]+):([^:()]+)\)\s*=\s*\[(.*?)\]", ref, re.DOTALL):
        lo, hi = bound(m.group(1)), bound(m.group(2))
        vals = _numbers(m.group(3))
        assert len(vals) == hi - lo + 1, (lo, hi, len(vals))
        flat[lo - 1:hi] = vals
    assert np.count_nonzero(flat) > 0
    # column-major storage c6ab(ref_a, ref_b, pair):
    # flat index = a + max_ref (b - 1) + max_ref^2 (pair - 1)
    packed = flat.reshape(npairs, max_ref, max_ref).transpose(2, 1, 0)   # (ref_a, ref_b, pair)
    c6 = np.zeros((MAX_REF, MAX_REF, MAX_Z + 1, MAX_Z + 1))
    for za in range(1, max_elem + 1):         # za >= zb, pair index zb + za (za - 1) / 2
        for zb in range(1, za + 1):
            ic = zb + za * (za - 1) // 2 - 1
            block = packed[:, :, ic]          # [ref of za, ref of zb]
            c6[:max_ref, :max_ref, za, zb] = block
            c6[:max_ref, :max_ref, zb, za] = block.T
    return {"nref": nref, "refcn": refcn, "c6": c6}


def build(sdftd3: Path, mctc: Path, sdftd3_original: Path | None = None) -> dict[str, np.ndarray]:
    src = sdftd3 / "src" / "dftd3"
    current = parse_reference_systems(sdftd3)
    nref, refcn, c6 = current["nref"], current["refcn"], current["c6"]
    assert int(nref[1:].min()) > 0, "current release must cover all elements"

    npairs = MAX_Z * (MAX_Z + 1) // 2
    vdw_text = _read_code(src / "data" / "vdwrad.f90")
    vdw_flat = parse_parameter_array(vdw_text, "vdwrad")
    assert len(vdw_flat) == npairs, len(vdw_flat)
    rvdw = np.zeros((MAX_Z + 1, MAX_Z + 1))
    k = 0
    for za in range(1, MAX_Z + 1):            # same packed ordering as C6 pairs
        for zb in range(1, za + 1):
            rvdw[za, zb] = rvdw[zb, za] = vdw_flat[k]
            k += 1

    r4r2_raw = parse_parameter_array(_read_code(src / "data" / "r4r2.f90"), "r4_over_r2")
    covrad = parse_parameter_array(_read_code(mctc / "src" / "mctc" / "data" / "covrad.f90"),
                                   "covalent_rad_2009")

    def pad(values, n=MAX_Z):
        out = np.zeros(n + 1)
        out[1:n + 1] = values[:n]
        return out

    tables = {
        "nref": nref,
        "refcn": refcn,                       # (7, 104), -1 marks an unused slot
        "c6": c6,                             # (7, 7, 104, 104), hartree bohr^6
        "rvdw_aa": rvdw,                      # (104, 104) pair cutoff radii, Angstrom
        "r4r2_raw": pad(r4r2_raw),            # <r4>/<r2>, before sqrt(0.5 sqrt(Z) .)
        "covalent_radius_aa": pad(covrad),    # Angstrom (Pyykko 2009)
    }

    if sdftd3_original is not None:
        orig = parse_reference_systems(sdftd3_original)
        covered = np.nonzero(orig["nref"])[0]         # elements the old release knew
        # an element's reference systems changed if their number, their
        # coordination numbers or the homoatomic C6 block changed
        differ = [z for z in covered
                  if orig["nref"][z] != nref[z]
                  or not np.array_equal(orig["refcn"][:, z], refcn[:, z])
                  or not np.array_equal(orig["c6"][:, :, z, z], c6[:, :, z, z])]
        # the patch must be complete: every covered pair whose C6 block
        # changed involves an element whose reference systems changed
        keep = np.setdiff1d(covered, differ)
        assert np.array_equal(orig["c6"][:, :, keep][:, :, :, keep], c6[:, :, keep][:, :, :, keep])
        zs = np.asarray(differ, dtype=np.int64)
        tables["original_max_z"] = np.int64(covered.max())   # last element the old release knew
        tables["original_z"] = zs
        tables["original_nref"] = orig["nref"][zs]
        tables["original_refcn"] = orig["refcn"][:, zs]
        tables["original_c6"] = orig["c6"][:, :, zs, :]
    return tables


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sdftd3", required=True, type=Path,
                   help="checkout of the current simple-dftd3 release")
    p.add_argument("--sdftd3-original", type=Path, default=None,
                   help="checkout of a simple-dftd3 release <= 1.0.0 (Grimme's original references)")
    p.add_argument("--mctc", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    tables = build(args.sdftd3, args.mctc, args.sdftd3_original)
    np.savez_compressed(args.out, **tables)
    msg = (f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} kB): "
           f"{int(tables['nref'].sum())} reference systems, "
           f"{np.count_nonzero(tables['c6']) // 2} reference C6 pairs")
    if "original_z" in tables:
        msg += f"; original references kept for Z = {tables['original_z'].tolist()}"
    print(msg)


if __name__ == "__main__":
    main()
