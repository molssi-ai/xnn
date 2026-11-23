"""Fidelity harness: xnn's DFTB1, DFTB2 and DFTB3 against the DFTB+ program.

DFTB+ (LGPL) is run as a black box on the same structures, with coordinates in
bohr converted with the model's own Bohr radius, a tight charge tolerance and
Richardson-extrapolated numerical derivatives; energies, Mulliken charges and
forces are compared. Run directly for a table::

    XNN_DFTB=/path/to/dftb+ python tests/sqmnn/sqmnn_dftb_fidelity.py
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from xnn.common.data import collate, structure_to_graph
from xnn.common.data.elements import CHEMICAL_SYMBOLS
from xnn.common.models import ForceStressOutput
from xnn.sqmnn.common.skf import published_folder
from xnn.sqmnn.models import DFTB1, DFTB2, DFTB3

#: Structures (symbols, Angstrom, total charge).
STRUCTURES = {
    "water": ("OHH", [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]], 0),
    "methanol": ("COHHHH", [[-0.0467, 0.6614, 0.0], [-0.0467, -0.7558, 0.0], [-1.0868, 1.0162, 0.0],
                            [0.4410, 1.0778, 0.8923], [0.4410, 1.0778, -0.8923], [0.8714, -1.0807, 0.0]], 0),
    "formamide": ("CONHHH", [[0.0, 0.4192, 0.0], [1.2079, 0.5955, 0.0], [-0.6386, -0.7898, 0.0],
                             [-0.6098, 1.3460, 0.0], [-1.6463, -0.8352, 0.0], [-0.1035, -1.6464, 0.0]], 0),
    "water_dimer": ("OHHOHH", [[-1.551, -0.114, 0.0], [-1.934, 0.762, 0.0], [-0.599, 0.040, 0.0],
                               [1.350, 0.111, 0.0], [1.680, -0.373, -0.763], [1.680, -0.373, 0.763]], 0),
    "hydroxide": ("OH", [[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]], -1),
    "ammonium": ("NHHHH", [[0.0, 0.0, 0.0], [0.59, 0.59, 0.59], [-0.59, -0.59, 0.59], [-0.59, 0.59, -0.59],
                           [0.59, -0.59, -0.59]], 1),
    "methanethiol": ("CSHHHH", [[-0.047, 1.149, 0.0], [-0.047, -0.669, 0.0], [-1.083, 1.495, 0.0],
                                [0.454, 1.536, 0.889], [0.454, 1.536, -0.889], [1.296, -0.764, 0.0]], 0),
    "trihydrogen": ("HHH", [[0.0, 0.0, 0.0], [0.87, 0.0, 0.0], [0.435, 0.75, 0.0]], 1),
}
#: Open-shell structures (symbols, Angstrom, total charge, spin multiplicity).
SPIN_STRUCTURES = {
    "methyl": ("CHHH", [[0, 0, 0], [1.08, 0, 0], [-0.54, 0.94, 0.02], [-0.55, -0.93, 0.0]], 0, 2),
    "amino": ("NHH", [[0, 0, 0], [1.02, 0, 0], [-0.28, 0.98, 0.05]], 0, 2),
    "dioxygen": ("OO", [[0, 0, 0], [0, 0, 1.21]], 0, 3),
    "methylene": ("CHH", [[0, 0, 0], [1.08, 0, 0], [-0.37, 1.01, 0.0]], 0, 3),
    "water_triplet": ("OHH", [[0, 0, 0.1173], [0, 0.7572, -0.4692], [0, -0.7572, -0.4692]], 0, 3),
}
#: Spin constants (hartree): one per atom, and the shell matrices.
ATOM_W = {"H": [-0.0717], "C": [-0.0227], "N": [-0.0255], "O": [-0.0278]}
SHELL_W = {"H": [-0.0717], "C": [-0.0306, -0.0251, -0.0251, -0.0227], "N": [-0.0332, -0.0275, -0.0275, -0.0255],
           "O": [-0.0352, -0.0296, -0.0296, -0.0278]}
#: Onsite constants (same spin, opposite spin; hartree) of the onsite correction.
ONSITE = {"H": {"uu": [0.0], "ud": [0.0]},
          "C": {"uu": [0.0, 0.04973, 0.04973, -0.01203], "ud": [0.0, 0.10512, 0.10512, 0.02643]},
          "N": {"uu": [0.0, 0.06816, 0.06816, -0.00879], "ud": [0.0, 0.12770, 0.12770, 0.03246]},
          "O": {"uu": [0.0, 0.08672, 0.08672, -0.00523], "ud": [0.0, 0.14969, 0.14969, 0.03834]},
          "S": {"uu": [0.0, 0.07501, 0.00398, 0.07501, 0.00310, 0.01100, 0.00398, 0.01100, -0.01792],
                "ud": [0.0, 0.11653, 0.03915, 0.11653, 0.03058, 0.04979, 0.03915, 0.04979, 0.01582]}}
#: Highest shells.
SHELLS = {"H": "s", "C": "p", "N": "p", "O": "p", "S": "d"}
_DFTB3 = "  ThirdOrderFull = Yes\n  HCorrection = Damping { Exponent = 4.00 }\n"
_D3BJ = {"model": "d3", "damping": "bj", "s8": 3.209, "a1": 0.746, "a2": 4.191}
#: Method, Slater-Koster set, input lines of DFTB+, options of the xnn model.
CASES = [("dftb1", "mio-1-1", "", {}), ("dftb2", "mio-1-1", "", {}), ("dftb2", "3ob-3-1", "", {}),
         ("dftb3", "3ob-3-1", _DFTB3, {}),
         ("dftb3", "3ob-3-1", _DFTB3 + "  Dispersion = DftD3 {\n    Damping = BeckeJohnson { a1 = 0.746; a2 = 4.191 }\n"
          "    s6 = 1.0\n    s8 = 3.209\n  }\n", {"dispersion": _D3BJ}),
         ("dftb2", "3ob-3-1", "  HCorrection = H5 {}\n", {"h5": True}),
         ("dftb3", "3ob-3-1", _DFTB3 + "  Dispersion = DftD4 {\n    s6 = 1.0\n    s8 = 0.6635015\n    s9 = 1.0\n"
          "    a1 = 0.5523240\n    a2 = 4.3537076\n  }\n",
          {"dispersion": {"model": "d4", "s8": 0.6635015, "s9": 1.0, "a1": 0.5523240, "a2": 4.3537076}}),
         ("dftb3", "3ob-3-1", "  ThirdOrderFull = Yes\n  HCorrection = H5 {}\n  Dispersion = DftD3 {\n"
          "    Damping = ZeroDamping { sr6 = 1.25; alpha6 = 29.61 }\n    s6 = 1.0\n    s8 = 0.49\n"
          "    HHRepulsion = Yes\n  }\n",
          {"h5": True, "dispersion": {"model": "d3", "damping": "zero", "rs6": 1.25, "alp": 29.61, "s8": 0.49,
                                      "hh_repulsion": True}}),
         ("dftb2", "ob2-1-1", "  Hybrid = LC {\n    Screening = MatrixBased {}\n  }\n", {}),
         ("dftb3", "3ob-3-1", "  ThirdOrderFull = Yes\n  HCorrection = H5 {}\n", {"h5": True})]


def _number(text: str) -> float:
    try:
        return float(text)
    except ValueError:                     # Fortran prints 1.2-309 for 1.2E-309
        return float(re.sub(r"(\d)([+-]\d{3})$", r"\1e\2", text))


def read_results(path: Path) -> dict:
    """The real arrays of a ``results.tag`` file, by label."""
    out, lines, k = {}, path.read_text().splitlines(), 0
    while k < len(lines):
        m = re.match(r"(\S[^:]*?)\s*:(\w+):(\d+):(.*)", lines[k])
        k += 1
        if not m:
            continue
        shape = [int(x) for x in m.group(4).split(",")] if int(m.group(3)) else []
        count = int(np.prod(shape)) if shape else 1
        values = []
        while len(values) < count and k < len(lines):
            values += [_number(x) for x in lines[k].split()]
            k += 1
        if m.group(2) == "real":
            arr = np.array(values[:count])
            out[m.group(1).strip()] = arr.reshape(shape[::-1]) if shape else arr[0]
    return out


def _spin_lines(spin: dict, unpaired: int, types) -> str:
    """Input lines of collinear spin polarization."""
    resolved = any(len(v) > 1 for v in spin.values())
    values = "".join(f"    {t} = {{ {' '.join(str(v) for v in spin[t])} }}\n" for t in types)
    return (f"  SpinPolarisation = Colinear {{\n    UnpairedElectrons = {unpaired}\n  }}\n"
            f"  SpinConstants {{\n    ShellResolvedSpin = {'Yes' if resolved else 'No'}\n{values}  }}\n")


def run_dftbplus(binary: Path, folder: Path, method: str, extra: str, symbols, pos_bohr, charge: int,
                 derivatives: dict, spin: Optional[dict] = None, multiplicity: int = 1) -> dict:
    """Energy (hartree), charges and forces (hartree / bohr) from DFTB+."""
    types = list(dict.fromkeys(symbols))
    coords = "\n".join(f"{types.index(s) + 1} {x:.16e} {y:.16e} {z:.16e}" for s, (x, y, z) in zip(symbols, pos_bohr))
    scc = method != "dftb1"
    if spin:
        extra = extra + _spin_lines(spin, multiplicity - 1, types)
    third = ""
    if "ThirdOrder" in extra:
        third = "  HubbardDerivs {\n" + "".join(f"    {t} = {derivatives[CHEMICAL_SYMBOLS.index(t)]}\n"
                                                for t in types) + "  }\n"
    text = (f'Geometry = {{\n TypeNames = {{ {" ".join(chr(34) + t + chr(34) for t in types)} }}\n'
            f' TypesAndCoordinates [Bohr] = {{\n{coords}\n }}\n}}\n'
            f'Hamiltonian = DFTB {{\n  Scc = {"Yes" if scc else "No"}\n'
            + ("  SccTolerance = 1e-12\n  MaxSccIterations = 1000\n" if scc else "")
            + f'  Charge = {charge}\n  SlaterKosterFiles = Type2FileNames {{\n    Prefix = "{folder}/"\n'
              '    Separator = "-"\n    Suffix = ".skf"\n  }\n  MaxAngularMomentum {\n'
            + "".join(f'    {t} = "{SHELLS[t]}"\n' for t in types) + "  }\n  Differentiation = Richardson {}\n"
            + extra + third + "}\nAnalysis {\n  PrintForces = Yes\n}\nOptions {\n  WriteResultsTag = Yes\n}\n"
              "ParserOptions {\n  ParserVersion = 15\n}\n")
    with tempfile.TemporaryDirectory() as work:
        (Path(work) / "dftb_in.hsd").write_text(text)
        run = subprocess.run([str(binary)], cwd=work, capture_output=True, text=True)
        tag = Path(work) / "results.tag"
        if not tag.exists():
            raise RuntimeError(f"DFTB+ failed:\n{run.stdout[-2000:]}")
        res = read_results(tag)
    return {"energy": float(res["mermin_energy"]), "charges": res["gross_atomic_charges"], "forces": res["forces"]}


def run_xnn(model, symbols, pos, charge: int, multiplicity: Optional[int] = None) -> dict:
    """The same quantities from xnn."""
    z = np.array([CHEMICAL_SYMBOLS.index(s) for s in symbols])
    s = {"pos": torch.tensor(np.asarray(pos, dtype=np.float64)), "atomic_numbers": z, "total_charge": float(charge)}
    if multiplicity is not None:
        s["spin_multiplicity"] = float(multiplicity)
    graph = collate([structure_to_graph(s, 0.1)])
    out = ForceStressOutput(model)(graph)
    return {"energy": float(out["energy"][0]) / model.hartree, "charges": out["charges"].numpy(),
            "forces": out["forces"].detach().numpy() / model.hartree * model.bohr,
            "converged": bool(out["scf_converged"])}


def compare(binary: Path, cases=CASES, structures=STRUCTURES) -> list:
    """One row per method, set and structure: differences in energy, charges and forces."""
    rows = []
    for method, name, extra, options in cases:
        folder = published_folder(name)
        cls = {"dftb1": DFTB1, "dftb2": DFTB2, "dftb3": DFTB3}[method]
        kwargs = {} if method == "dftb1" else {"scc_tolerance": 1e-12, "max_iter": 1000}
        model = cls(name, **kwargs, **options)
        for key, (symbols, pos, charge) in structures.items():
            symbols = list(symbols)
            if any(CHEMICAL_SYMBOLS.index(s) not in model.supported for s in symbols):
                continue
            ref = run_dftbplus(binary, folder, method, extra, symbols, np.asarray(pos) / model.bohr, charge,
                               model.hubbard_derivatives)
            mine = run_xnn(model, symbols, pos, charge)
            disp = options.get("dispersion", {}).get("model", "")
            label = method + (f"-{disp}" if disp else "") + ("-h5" if "h5" in options else "")
            rows.append({"case": f"{label}/{name}/{key}", "reference": ref["energy"],
                         "energy": mine["energy"] - ref["energy"],
                         "charges": float(np.abs(mine["charges"] - ref["charges"]).max()),
                         "forces": float(np.abs(mine["forces"] - ref["forces"]).max()),
                         "converged": mine["converged"]})
    # the set as xnn writes it: DFTB+ gives the same energies as with the published files
    model = DFTB2("mio-1-1", scc_tolerance=1e-12, max_iter=1000)
    with tempfile.TemporaryDirectory() as written:
        model.write_set(written)
        for key in ("water", "methanethiol"):
            symbols, pos, charge = structures[key]
            pos_bohr = np.asarray(pos) / model.bohr
            a = run_dftbplus(binary, published_folder("mio-1-1"), "dftb2", "", list(symbols), pos_bohr, charge, {})
            b = run_dftbplus(binary, Path(written), "dftb2", "", list(symbols), pos_bohr, charge, {})
            rows.append({"case": f"written-set/mio-1-1/{key}", "reference": a["energy"],
                         "energy": b["energy"] - a["energy"],
                         "charges": float(np.abs(b["charges"] - a["charges"]).max()),
                         "forces": float(np.abs(b["forces"] - a["forces"]).max()), "converged": True})
    model = DFTB2("mio-1-1", scc_tolerance=1e-12, max_iter=1000, onsite_correction=ONSITE)
    folder = published_folder("mio-1-1")
    for key, (symbols, pos, charge) in structures.items():
        symbols = list(symbols)
        values = "".join(f"    {t}uu = {{ {' '.join(map(str, ONSITE[t]['uu']))} }}\n"
                         f"    {t}ud = {{ {' '.join(map(str, ONSITE[t]['ud']))} }}\n" for t in dict.fromkeys(symbols))
        ref = run_dftbplus(binary, folder, "dftb2", f"  OnSiteCorrection = {{\n{values}  }}\n", symbols,
                           np.asarray(pos) / model.bohr, charge, {})
        mine = run_xnn(model, symbols, pos, charge)
        rows.append({"case": f"dftb2-onsite/mio-1-1/{key}", "reference": ref["energy"],
                     "energy": mine["energy"] - ref["energy"],
                     "charges": float(np.abs(mine["charges"] - ref["charges"]).max()),
                     "forces": float(np.abs(mine["forces"] - ref["forces"]).max()), "converged": mine["converged"]})
    for functional, label in (("fll", "FLL"), ("psic", "pSIC")):
        model = DFTB2("mio-1-1", scc_tolerance=1e-12, max_iter=1000, spin_constants=ATOM_W,
                      orbital_potential={"functional": functional, "shells": {"O": [{"shells": [1], "uj": 0.2}],
                                                                             "C": [{"shells": [1], "uj": 0.2}]}})
        for key, (symbols, pos, charge, mult) in SPIN_STRUCTURES.items():
            symbols = list(symbols)
            types = list(dict.fromkeys(symbols))
            blocks = "".join(f"    {t} = {{\n      Shells = {{ 2 }}\n      UJ = 0.2\n    }}\n"
                             for t in types if t in "OC")
            extra = f"  OrbitalPotential = {{\n    Functional = {{ {label} }}\n{blocks}  }}\n"
            ref = run_dftbplus(binary, folder, "dftb2", extra, symbols, np.asarray(pos) / model.bohr, charge, {},
                               spin={t: ATOM_W[t] for t in types}, multiplicity=mult)
            mine = run_xnn(model, symbols, pos, charge, mult)
            rows.append({"case": f"dftb2-u-{functional}/mio-1-1/{key}", "reference": ref["energy"],
                         "energy": mine["energy"] - ref["energy"],
                         "charges": float(np.abs(mine["charges"] - ref["charges"]).max()),
                         "forces": float(np.abs(mine["forces"] - ref["forces"]).max()), "converged": mine["converged"]})
    for method, name, extra in (("dftb2", "mio-1-1", ""), ("dftb3", "3ob-3-1", _DFTB3)):
        cls = {"dftb2": DFTB2, "dftb3": DFTB3}[method]
        folder = published_folder(name)
        for label, table in (("atom", ATOM_W), ("shell", SHELL_W)):
            model = cls(name, scc_tolerance=1e-12, max_iter=1000, spin_constants=table)
            for key, (symbols, pos, charge, mult) in SPIN_STRUCTURES.items():
                symbols = list(symbols)
                ref = run_dftbplus(binary, folder, method, extra, symbols, np.asarray(pos) / model.bohr, charge,
                                   model.hubbard_derivatives, spin=table, multiplicity=mult)
                mine = run_xnn(model, symbols, pos, charge, mult)
                rows.append({"case": f"{method}-spin-{label}/{name}/{key}", "reference": ref["energy"],
                             "energy": mine["energy"] - ref["energy"],
                             "charges": float(np.abs(mine["charges"] - ref["charges"]).max()),
                             "forces": float(np.abs(mine["forces"] - ref["forces"]).max()),
                             "converged": mine["converged"]})
    return rows


def run_dftbplus_periodic(binary: Path, folder: Path, extra: str, symbols, pos_bohr, cell_bohr, shells,
                          kpoints: Optional[dict] = None) -> dict:
    """Energy, charges, forces and stress (hartree, bohr) of a periodic structure from DFTB+, at the Gamma point
    or on a Monkhorst-Pack grid ``{"grid": ..., "shift": ...}``."""
    sampling = "  KPointsAndWeights = { 0.0 0.0 0.0 1.0 }\n"
    if kpoints is not None:
        (n1, n2, n3), shift = kpoints["grid"], kpoints.get("shift", (0.0, 0.0, 0.0))
        sampling = (f"  KPointsAndWeights = SupercellFolding {{\n    {n1} 0 0\n    0 {n2} 0\n    0 0 {n3}\n"
                    f"    {shift[0]} {shift[1]} {shift[2]}\n  }}\n")
    types = list(dict.fromkeys(symbols))
    coords = "\n".join(f"{types.index(s) + 1} {x:.16e} {y:.16e} {z:.16e}" for s, (x, y, z) in zip(symbols, pos_bohr))
    lattice = "\n".join(f"{x:.16e} {y:.16e} {z:.16e}" for x, y, z in cell_bohr)
    text = (f'Geometry = {{\n TypeNames = {{ {" ".join(chr(34) + t + chr(34) for t in types)} }}\n'
            f' TypesAndCoordinates [Bohr] = {{\n{coords}\n }}\n Periodic = Yes\n'
            f' LatticeVectors [Bohr] = {{\n{lattice}\n }}\n}}\n'
            'Hamiltonian = DFTB {\n  Scc = Yes\n  SccTolerance = 1e-12\n  MaxSccIterations = 1000\n'
            f'  EwaldTolerance = 1e-14\n  SlaterKosterFiles = Type2FileNames {{\n    Prefix = "{folder}/"\n'
            '    Separator = "-"\n    Suffix = ".skf"\n  }\n  MaxAngularMomentum {\n'
            + "".join(f'    {t} = "{shells[t]}"\n' for t in types)
            + "  }\n" + sampling + "  Differentiation = Richardson {}\n" + extra
            + "}\nAnalysis {\n  PrintForces = Yes\n}\nOptions {\n  WriteResultsTag = Yes\n}\n"
              "ParserOptions {\n  ParserVersion = 15\n}\n")
    with tempfile.TemporaryDirectory() as work:
        (Path(work) / "dftb_in.hsd").write_text(text)
        run = subprocess.run([str(binary)], cwd=work, capture_output=True, text=True)
        tag = Path(work) / "results.tag"
        if not tag.exists():
            raise RuntimeError(f"DFTB+ failed:\n{run.stdout[-2000:]}")
        res = read_results(tag)
    return {"energy": float(res["mermin_energy"]), "charges": res["gross_atomic_charges"], "forces": res["forces"],
            "stress": -res["stress"] if "stress" in res else None}      # DFTB+ reports the opposite sign


def periodic_structures():
    """Periodic test structures (symbols, Angstrom positions, Angstrom cell, set, shells, spin multiplicity)."""
    rng = np.random.default_rng(1)
    frac = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0], [.25, .25, .25], [.25, .75, .75],
                     [.75, .25, .75], [.75, .75, .25]])
    si = (["Si"] * 8, frac * 5.431 + 0.05 * rng.normal(size=(8, 3)), np.eye(3) * 5.431, "pbc-0-3", {"Si": "d"}, 1)
    water = np.array(STRUCTURES["water"][1])
    pos, sym = [], []
    for k in range(8):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        pos.append(water @ q.T + 3.1 * np.array([k % 2, (k // 2) % 2, k // 4]) + 0.1 * rng.normal(size=3))
        sym += ["O", "H", "H"]
    box = (sym, np.concatenate(pos), np.eye(3) * 6.2, "3ob-3-1", {"O": "p", "H": "s"}, 1)
    o2 = (["O", "O"], np.array([[0.3, 0.2, 0.1], [1.508, 0.25, 0.05]]), np.eye(3) * 7.0, "mio-1-1", {"O": "p"}, 3)
    skewed = (sym, np.concatenate(pos), np.array([[6.2, 0.0, 0.0], [3.0, 6.0, 0.0], [0.0, 0.0, 7.5]]), "ob2-1-1",
              {"O": "p", "H": "s"}, 1)
    return {"si8": si, "water_box": box, "o2_box": o2, "water_skewed": skewed}


#: Structure, method, Slater-Koster set (the structure's own when None), input lines of DFTB+, xnn options.
#: D4 is left out: in a cell DFTB+'s D4 charges depend on its Ewald parameter, while xnn equals the dftd4 program.
PERIODIC_CASES = [("si8", "dftb2", None, "", {}), ("water_box", "dftb2", None, "", {}),
                  ("water_box", "dftb3", None, None, {}),
                  ("water_box", "dftb3", None, CASES[4][2], CASES[4][3]),
                  ("water_box", "dftb3", None, CASES[7][2], CASES[7][3]),
                  ("water_box", "dftb2", "mio-1-1", "  ShellResolvedSCC = Yes\n", {"shell_resolved": True}),
                  ("si8", "dftb2", "matsci-0-3", "  ShellResolvedSCC = Yes\n", {"shell_resolved": True}),
                  ("o2_box", "dftb2", None, _spin_lines({"O": ATOM_W["O"]}, 2, ["O"]), {"spin_constants": ATOM_W}),
                  ("o2_box", "dftb2", None, _spin_lines({"O": ATOM_W["O"]}, 2, ["O"])
                   + "  OrbitalPotential = {\n    Functional = { FLL }\n    O = {\n      Shells = { 2 }\n"
                     "      UJ = 0.2\n    }\n  }\n",
                   {"spin_constants": ATOM_W,
                    "orbital_potential": {"functional": "fll", "shells": {"O": [{"shells": [1], "uj": 0.2}]}}}),
                  ("water_box", "dftb2", "mio-1-1", "  OnSiteCorrection = {\n" + "".join(
                      f"    {t}uu = {{ {' '.join(map(str, ONSITE[t]['uu']))} }}\n"
                      f"    {t}ud = {{ {' '.join(map(str, ONSITE[t]['ud']))} }}\n" for t in "OH") + "  }\n",
                   {"onsite_correction": ONSITE}),
                  ("water_box", "dftb2", "ob2-1-1", CASES[8][2], {}), ("water_skewed", "dftb2", None, CASES[8][2], {}),
                  ("si8", "dftb2", None, "", {"kpoints": {"grid": [2, 2, 2], "shift": [0.5, 0.5, 0.5]}}),
                  ("si8", "dftb2", None, "", {"kpoints": {"grid": [2, 2, 2], "shift": [0.0, 0.0, 0.0]}}),
                  ("water_box", "dftb3", None, None, {"kpoints": {"grid": [2, 1, 1], "shift": [0.0, 0.0, 0.0]}}),
                  ("o2_box", "dftb2", None, _spin_lines({"O": ATOM_W["O"]}, 2, ["O"]),
                   {"spin_constants": ATOM_W, "kpoints": {"grid": [2, 2, 2], "shift": [0.5, 0.5, 0.5]}})]


def compare_periodic(binary: Path, cases=PERIODIC_CASES) -> list:
    """Rows of energy, charge, force and stress differences of periodic structures at the Gamma point."""
    rows = []
    structures = periodic_structures()
    for key, method, name, extra, options in cases:
        symbols, pos, cell, own, shells, multiplicity = structures[key]
        name = name or own
        cls = DFTB3 if method == "dftb3" else DFTB2
        model = cls(name, max_angular_momentum=shells, scc_tolerance=1e-12, max_iter=1000, **options)
        if extra is None or (method == "dftb3" and "HubbardDerivs" not in extra):
            extra = (extra or _DFTB3) + _third_lines(model, symbols)
        ref = run_dftbplus_periodic(binary, published_folder(name), extra, symbols, pos / model.bohr,
                                    cell / model.bohr, shells, options.get("kpoints"))
        z = np.array([CHEMICAL_SYMBOLS.index(s) for s in symbols])
        data = {"pos": torch.tensor(pos), "atomic_numbers": z, "total_charge": 0.0, "cell": torch.tensor(cell),
                "pbc": torch.ones(3, dtype=torch.bool)}
        if multiplicity > 1:
            data["spin_multiplicity"] = float(multiplicity)
        graph = collate([structure_to_graph(data, 0.1)])
        out = ForceStressOutput(model, compute_stress=True)(graph)
        tags = "+".join(sorted(options)) + ("-" + options["dispersion"]["model"] if "dispersion" in options else "")
        rows.append({"case": f"{method}-periodic{'-' + tags if tags else ''}/{name}/{key}", "reference": ref["energy"],
                     "energy": float(out["energy"][0]) / model.hartree - ref["energy"],
                     "charges": float(np.abs(out["charges"].numpy() - ref["charges"]).max()),
                     "forces": float(np.abs(out["forces"].detach().numpy() / model.hartree * model.bohr
                                            - ref["forces"]).max()),
                     "stress": 0.0 if ref["stress"] is None else float(np.abs(
                         out["stress"][0].detach().numpy() / model.hartree * model.bohr ** 3 - ref["stress"]).max()),
                     "converged": bool(out["scf_converged"])})
    return rows


def _third_lines(model, symbols) -> str:
    return "  HubbardDerivs {\n" + "".join(f"    {t} = {model.hubbard_derivatives[CHEMICAL_SYMBOLS.index(t)]}\n"
                                           for t in dict.fromkeys(symbols)) + "  }\n"


def main(argv) -> int:
    torch.set_default_dtype(torch.float64)
    binary = Path(os.environ["XNN_DFTB"])
    for row in compare_periodic(binary):
        print(f"{row['case']:32s} E={row['reference']:.12f} dE={row['energy']:+.1e} dq={row['charges']:.1e} "
              f"dF={row['forces']:.1e} dstress={row['stress']:.1e}")
    for row in compare(binary):
        print(f"{row['case']:32s} E={row['reference']:.12f} dE={row['energy']:+.1e} dq={row['charges']:.1e} "
              f"dF={row['forces']:.1e} converged={row['converged']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
