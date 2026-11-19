"""Fidelity of the xnn GFN methods (GFN0-xTB, GFN1-xTB, GFN2-xTB) to the xtb program.

    XNN_XTB=/path/to/xtb python sqmnn_xtb_fidelity.py [gfn0 gfn1 gfn2]

Every structure is evaluated by xtb (``--acc 0.0001``, gradient, JSON output)
and by xnn in float64; total energies, gradients and Mulliken charges are
compared. xtb is an external oracle only: it is run, never linked. Its
parameter files (``share/xtb`` next to the binary, or ``XTBPATH``) are passed
to xnn as well, so both programs use the same values. The last line printed is
the worst deviation of each quantity.

``--no-dispersion`` switches the dispersion term off in both programs
(``s8 = s9 = 0`` and a large ``a2`` in the parameter file), which isolates the
tight-binding part.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from xnn.common.data import collate, structure_to_graph
from xnn.common.data.elements import atomic_number
from xnn.common.models import ForceStressOutput
from xnn.sqmnn.common.xtb_published import parse_parameters
from xnn.sqmnn.models import GFN0, GFN1, GFN2
from xnn.sqmnn.models.xtb import AUTOEV, BOHR

METHODS = {"gfn0": GFN0, "gfn1": GFN1, "gfn2": GFN2}
_TETRA = np.array([[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]]) / np.sqrt(3.0)
# (name, symbols, xyz (Angstrom), charge, unpaired electrons); slightly distorted geometries
STRUCTURES = [
    ("water", ["O", "H", "H"], [[0, 0, 0.12], [0, 0.76, -0.47], [0.05, -0.75, -0.49]], 0, 0),
    ("methane", ["C"] + ["H"] * 4, np.vstack([[0.01, -0.02, 0.0], 1.09 * _TETRA + [[0.02, 0, 0], [0, 0, 0],
                                                                               [0, 0.03, 0], [0, 0, -0.02]]]), 0, 0),
    ("formamide", ["C", "O", "N", "H", "H", "H"],
     [[0, 0.42, 0], [1.2, 0.5, 0.02], [-0.72, -0.73, 0], [-0.55, 1.36, 0.01], [-1.73, -0.69, 0.05],
      [-0.27, -1.63, -0.03]], 0, 0),
    ("water dimer", ["O", "H", "H", "O", "H", "H"],
     [[-1.551, -0.115, 0.0], [-1.934, 0.763, 0.0], [-0.600, 0.041, 0.0],
      [1.351, 0.111, 0.0], [1.680, -0.374, -0.759], [1.680, -0.374, 0.759]], 0, 0),
    ("methyl radical", ["C", "H", "H", "H"], [[0, 0, 0.02], [1.08, 0, 0], [-0.54, 0.94, 0.01],
                                             [-0.55, -0.93, 0]], 0, 1),
    ("ammonium", ["N"] + ["H"] * 4, np.vstack([[0, 0, 0], 1.03 * _TETRA + [[0.01, 0, 0], [0, 0, 0],
                                                                          [0, 0, 0], [0, 0, 0.02]]]), 1, 0),
    ("hydroxide", ["O", "H"], [[0, 0, 0], [0.02, 0, 0.97]], -1, 0),
    ("silanol", ["Si", "O", "H", "H", "H", "H"],
     [[0, 0, 0], [1.65, 0, 0], [2.05, 0.85, 0], [-0.5, 1.3, 0], [-0.5, -0.7, 1.2], [-0.5, -0.7, -1.2]], 0, 0),
    ("sulfur dioxide", ["S", "O", "O"], [[0, 0, 0.36], [0, 1.26, -0.37], [0.02, -1.25, -0.36]], 0, 0),
    ("titanium tetrachloride", ["Ti", "Cl", "Cl", "Cl", "Cl"],
     np.vstack([[0, 0, 0], 2.17 * _TETRA + [[0.02, 0, 0], [0, 0, 0], [0, 0.01, 0], [0, 0, 0]]]), 0, 0),
    ("iron chloride", ["Fe", "Cl"], [[0, 0, 0], [0, 0, 2.1]], 0, 1),
    ("ammonia bromine fluoride", ["N", "H", "H", "H", "Br", "F"],
     [[0, 0, 0], [0.94, 0, -0.33], [-0.47, 0.814, -0.33], [-0.47, -0.814, -0.33], [0.05, 0, 2.9],
      [0.1, 0, 4.66]], 0, 0),
    ("iodomethane", ["C", "I", "H", "H", "H"],
     [[0, 0, 0], [0, 0, 2.14], [1.03, 0, -0.36], [-0.52, 0.89, -0.36], [-0.51, -0.9, -0.35]], 0, 0),
    ("dichromium", ["Cr", "Cr"], [[0, 0, 0], [0, 0, 1.75]], 0, 0),
    ("palladium dihydride", ["Pd", "H", "H"], [[0, 0, 0], [1.55, 0, 0], [-0.5, 1.47, 0]], 0, 0),
    ("gadolinium hydride", ["Gd", "H"], [[0, 0, 0], [0, 0, 2.05]], 0, 0),
]


def xtb_home(binary: Path) -> Path:
    """The ``share/xtb`` folder holding the parameter files of an xtb binary."""
    path = os.environ.get("XTBPATH")
    return Path(path) if path else binary.resolve().parent.parent / "share" / "xtb"


def parameter_text(binary: Path, method: str, dispersion: bool = True) -> str:
    """The parameter file of ``method``, with the dispersion term switched off when asked."""
    text = (xtb_home(binary) / f"param_{method}-xtb.txt").read_text()
    if not dispersion:
        for key, val in (("s8", 0.0), ("s9", 0.0), ("a2", 1000.0)):
            text = re.sub(rf"^{key}\s+\S+", f"{key}          {val:.5f}", text, flags=re.M)
    return text


def run_xtb(binary: Path, method: str, numbers, pos, charge: int, unpaired: int, text: str) -> dict:
    """Total energy (hartree), gradient (hartree/bohr) and charges from one xtb run."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        lines = [str(len(numbers)), ""]
        for z, p in zip(numbers, pos):
            lines.append(f"{z} {p[0]:.12f} {p[1]:.12f} {p[2]:.12f}")
        (work / "mol.xyz").write_text("\n".join(lines) + "\n")
        (work / "param.txt").write_text(text)
        env = dict(os.environ, XTBPATH=str(xtb_home(binary)), OMP_NUM_THREADS="1", OMP_STACKSIZE="1G")
        cmd = [str(binary), "mol.xyz", "--gfn", method[-1], "--chrg", str(charge), "--uhf", str(unpaired),
               "--acc", "0.0001", "--grad", "--json", "--norestart", "--vparam", str(work / "param.txt")]
        out = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, check=True).stdout
        energy = float(re.findall(r"TOTAL ENERGY\s+(-?\d+\.\d+)", out)[-1])
        rows = [ln.split() for ln in (work / "gradient").read_text().splitlines()]
        rows = [r for r in rows if len(r) == 3 and not r[0].startswith("$")][-len(numbers):]
        grad = np.array([[float(x.replace("D", "E")) for x in r] for r in rows])
        charges = np.array(json.loads((work / "xtbout.json").read_text())["partial charges"])
    return {"energy": energy, "gradient": grad, "charges": charges}


def run_xnn(model, numbers, pos, charge: int, unpaired: int) -> dict:
    """The same quantities from xnn (float64)."""
    s = {"pos": torch.tensor(np.asarray(pos, dtype=np.float64)), "atomic_numbers": np.asarray(numbers),
         "total_charge": float(charge), "spin_multiplicity": float(unpaired + 1)}
    graph = collate([structure_to_graph(s, 0.1)])
    out = ForceStressOutput(model)(graph)
    return {"energy": float(out["energy"][0]) / AUTOEV,
            "gradient": -out["forces"].detach().numpy() / AUTOEV * BOHR,
            "charges": out["charges"].numpy(), "converged": bool(out["scf_converged"])}


def compare(binary: Path, method: str, dispersion: bool = True, structures=STRUCTURES) -> dict:
    """Worst deviations of energy (hartree), gradient (hartree/bohr) and charges over ``structures``."""
    text = parameter_text(binary, method, dispersion)
    # converged a decade beyond xtb, so that the deviations are those of xtb's own convergence
    model = METHODS[method](parameters=parse_parameters(text, method), accuracy=1e-5)
    worst = {"energy": 0.0, "gradient": 0.0, "charges": 0.0}
    rows = []
    for name, symbols, pos, charge, unpaired in structures:
        numbers = [atomic_number(s) for s in symbols]
        ref = run_xtb(binary, method, numbers, pos, charge, unpaired, text)
        got = run_xnn(model, numbers, pos, charge, unpaired)
        dev = {k: float(np.abs(np.asarray(got[k]) - np.asarray(ref[k])).max()) for k in worst}
        rows.append((name, dev, got["converged"]))
        for k in worst:
            worst[k] = max(worst[k], dev[k])
    return {"worst": worst, "rows": rows}


def main(argv):
    binary = Path(os.environ["XNN_XTB"])
    dispersion = "--no-dispersion" not in argv
    methods = [a for a in argv if a in METHODS] or list(METHODS)
    for method in methods:
        res = compare(binary, method, dispersion)
        for name, dev, conv in res["rows"]:
            print(f"{method} {name:28s} dE={dev['energy']:.2e} dG={dev['gradient']:.2e} dq={dev['charges']:.2e}"
                  + ("" if conv else " (not converged)"))
        w = res["worst"]
        print(f"{method} worst: energy {w['energy']:.2e} Eh, gradient {w['gradient']:.2e} Eh/bohr, "
              f"charges {w['charges']:.2e}")


if __name__ == "__main__":
    main(sys.argv[1:])
