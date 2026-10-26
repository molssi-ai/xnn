"""Fidelity of the xnn NDDO methods (PM3, PM6, PM6-ORG, PM7) to the MOPAC program.

    XNN_MOPAC=/path/to/mopac python sqmnn_mopac_fidelity.py [PM3 PM6 PM6-ORG PM7]

Every structure is evaluated by MOPAC (``1SCF PRECISE GRADIENTS``, tight
SCF) and by xnn in float64; the heats of formation (kcal/mol), gradients
(kcal/mol/Angstrom), dipole moments (Debye) and atomic charges are compared.
MOPAC's analytic PM6-ORG gradient (and its gradient of systems with a small
HOMO-LUMO gap) is not the derivative of its energy, so there the gradient is
compared with central differences of MOPAC's heats of formation instead. MOPAC is an external oracle only. The last line
printed is the worst deviation of each quantity.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from xnn.common.data import structure_to_graph
from xnn.common.data.elements import atomic_number
from xnn.common.models import ForceStressOutput
from xnn.sqmnn.common.constants import DEBYE_PER_E_ANGSTROM, EV_TO_KCAL
from xnn.sqmnn.models import PM3, PM6, PM6ORG, PM7

METHODS = {"PM3": PM3, "PM6": PM6, "PM6-ORG": PM6ORG, "PM7": PM7}
# small HOMO-LUMO gap: MOPAC's analytic gradient deviates from the derivative of
# its energy by ~1e-2 kcal/mol/A, so the gradient is checked by differences
FD_GRADIENTS = {"iron carbonyl"}

# (name, symbols, xyz (Angstrom), charge, multiplicity); geometries are slightly
# distorted so no symmetry hides an error
_TETRA = np.array([[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]]) / np.sqrt(3.0)
STRUCTURES = [
    ("water", ["O", "H", "H"], [[0, 0, 0.12], [0, 0.76, -0.47], [0.05, -0.75, -0.49]], 0, 1),
    ("methane", ["C"] + ["H"] * 4, np.vstack([[0.01, -0.02, 0.0], 1.09 * _TETRA + [[0.02, 0, 0], [0, 0, 0],
                                                                               [0, 0.03, 0], [0, 0, -0.02]]]), 0, 1),
    ("acetylene", ["C", "C", "H", "H"], [[0, 0, 0.6], [0.02, 0, -0.61], [0, 0.03, 1.66], [0, 0, -1.67]], 0, 1),
    ("formamide", ["C", "O", "N", "H", "H", "H"],
     [[0, 0.42, 0], [1.2, 0.5, 0.02], [-0.72, -0.73, 0], [-0.55, 1.36, 0.01], [-1.73, -0.69, 0.05],
      [-0.27, -1.63, -0.03]], 0, 1),
    ("water dimer", ["O", "H", "H", "O", "H", "H"],
     [[-1.551007, -0.114520, 0.0], [-1.934259, 0.762503, 0.0], [-0.599677, 0.040712, 0.0],
      [1.350625, 0.111469, 0.0], [1.680398, -0.373741, -0.758561], [1.680398, -0.373741, 0.758561]], 0, 1),
    ("methyl radical", ["C", "H", "H", "H"], [[0, 0, 0.02], [1.08, 0, 0], [-0.54, 0.94, 0.01],
                                             [-0.55, -0.93, 0]], 0, 2),
    ("ammonium", ["N"] + ["H"] * 4, np.vstack([[0, 0, 0], 1.03 * _TETRA + [[0.01, 0, 0], [0, 0, 0],
                                                                          [0, 0, 0], [0, 0, 0.02]]]), 1, 1),
    ("hydroxide", ["O", "H"], [[0, 0, 0], [0.02, 0, 0.97]], -1, 1),
    ("silanol", ["Si", "O", "H", "H", "H", "H"],
     [[0, 0, 0], [1.65, 0, 0], [2.05, 0.85, 0], [-0.5, 1.3, 0], [-0.5, -0.7, 1.2], [-0.5, -0.7, -1.2]], 0, 1),
    ("sulfur dioxide", ["S", "O", "O"], [[0, 0, 0.36], [0, 1.26, -0.37], [0.02, -1.25, -0.36]], 0, 1),
    ("phosphorus pentafluoride", ["P", "F", "F", "F", "F", "F"],
     [[0, 0, 0], [0, 0, 1.58], [0.01, 0, -1.58], [1.54, 0, 0], [-0.77, 1.33, 0.02], [-0.77, -1.33, 0]], 0, 1),
    ("titanium tetrachloride", ["Ti", "Cl", "Cl", "Cl", "Cl"],
     np.vstack([[0, 0, 0], 2.17 * _TETRA + [[0.02, 0, 0], [0, 0, 0], [0, 0.01, 0], [0, 0, 0]]]), 0, 1),
    ("zinc chloride", ["Zn", "Cl", "Cl"], [[0, 0, 0], [0, 0.05, 2.07], [0, 0, -2.07]], 0, 1),
    ("copper chloride", ["Cu", "Cl"], [[0, 0, 0], [0, 0, 2.05]], 0, 1),
    ("iron carbonyl", ["Fe", "C", "O", "C", "O", "C", "O", "C", "O", "C", "O"],
     [[0, 0, 0], [0, 0, 1.81], [0, 0, 2.96], [0, 0, -1.81], [0, 0, -2.96], [1.8, 0, 0], [2.95, 0, 0],
      [-0.9, 1.56, 0], [-1.47, 2.55, 0], [-0.9, -1.56, 0.05], [-1.47, -2.55, 0.05]], 0, 1),
]


def run_mopac(symbols, xyz, method, charge=0, multiplicity=1, gradients=True, workdir=None):
    """Heat of formation, gradients, dipole and charges from a MOPAC single point."""
    exe = os.environ["XNN_MOPAC"]
    kw = [method, "1SCF", "PRECISE", "SCFCRT=1.D-12", "AUX(PRECISION=9)", "GEO-OK", "LET",
          f"CHARGE={charge}"]
    if gradients:
        kw.append("GRADIENTS")
    if multiplicity != 1:
        kw += ["UHF", {2: "DOUBLET", 3: "TRIPLET", 4: "QUARTET", 5: "QUINTET", 6: "SEXTET"}[multiplicity]]
    lines = [" ".join(kw), "xnn fidelity", ""]
    lines += [f"{s:2s} {x:20.12f} 1 {y:20.12f} 1 {z:20.12f} 1" for s, (x, y, z) in zip(symbols, xyz)]
    work = Path(workdir or tempfile.mkdtemp())
    work.mkdir(parents=True, exist_ok=True)
    (work / "job.mop").write_text("\n".join(lines) + "\n")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(Path(exe).resolve().parent.parent / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    subprocess.run([exe, "job.mop"], cwd=work, env=env, capture_output=True, timeout=600)
    aux = (work / "job.aux").read_text()

    def num(s):
        return float(s.replace("D", "E"))

    def block(key):
        m = re.search(key + r"\[\d+\]=\s*(.*?)(?=\n\s*[A-Z#])", aux, re.S)
        return np.array([num(v) for v in re.findall(r"[-+]?\d+\.\d+(?:D[-+]\d+)?", m.group(1))]) if m else None

    out = {"hof": num(re.search(r"HEAT_OF_FORMATION:KCAL/MOL=([-+0-9.D]+)", aux).group(1))}
    m = re.search(r"DIP_VEC:DEBYE\[3\]=\s*([-+0-9.D]+)\s+([-+0-9.D]+)\s+([-+0-9.D]+)", aux)
    out["dipole"] = np.array([num(g) for g in m.groups()]) if m else None
    out["charges"] = block("ATOM_CHARGES")
    g = block("GRADIENTS:KCAL/MOL/ANGSTROM")
    out["gradients"] = None if g is None else g.reshape(-1, 3)
    return out


def run_xnn(model, symbols, xyz, charge=0, multiplicity=1):
    """The same quantities from xnn (kcal/mol, kcal/mol/A, Debye, e)."""
    s = {"pos": torch.as_tensor(np.asarray(xyz, dtype=np.float64)),
         "atomic_numbers": np.array([atomic_number(v) for v in symbols]),
         "total_charge": float(charge), "spin_multiplicity": float(multiplicity)}
    out = ForceStressOutput(model)(structure_to_graph(s, model.cutoff))
    return {"hof": float(out["heat_of_formation"][0]),
            "gradients": -out["forces"].detach().numpy() * EV_TO_KCAL,
            "dipole": out["dipole"][0].detach().numpy() * DEBYE_PER_E_ANGSTROM,
            "charges": out["charges"].detach().numpy()}


def compare(method: str, verbose: bool = True) -> dict:
    """Worst deviations of xnn from MOPAC over :data:`STRUCTURES` for one method."""
    model = METHODS[method]()
    worst = {"hof": 0.0, "gradients": 0.0, "dipole": 0.0, "charges": 0.0}
    for name, symbols, xyz, charge, mult in STRUCTURES:
        if any(atomic_number(s) not in model.supported for s in symbols):
            continue
        xyz = np.asarray(xyz, dtype=np.float64)
        ref = run_mopac(symbols, xyz, method, charge, mult)
        if ref["hof"] is None:
            continue
        ours = run_xnn(model, symbols, xyz, charge, mult)
        if method == "PM6-ORG" or name in FD_GRADIENTS:
            # central differences of MOPAC's own energies on the first atom
            h, fd = 1e-4, np.zeros(3)
            for c in range(3):
                e = []
                for sgn in (1, -1):
                    x = xyz.copy()
                    x[0, c] += sgn * h
                    e.append(run_mopac(symbols, x, method, charge, mult, gradients=False)["hof"])
                fd[c] = (e[0] - e[1]) / (2 * h)
            dg = float(np.abs(ours["gradients"][0] - fd).max())
        else:
            dg = float(np.abs(ours["gradients"] - ref["gradients"]).max())
        dev = {"hof": abs(ours["hof"] - ref["hof"]), "gradients": dg,
               "dipole": float(np.abs(ours["dipole"] - ref["dipole"]).max()),
               "charges": float(np.abs(ours["charges"] - ref["charges"]).max())}
        for k in worst:
            worst[k] = max(worst[k], dev[k])
        if verbose:
            print(f"{method:8s} {name:26s} HoF {ours['hof']:12.5f} (MOPAC {ref['hof']:12.5f}) "
                  + " ".join(f"d{k} {v:.1e}" for k, v in dev.items()), flush=True)
    return worst


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    for method in sys.argv[1:] or list(METHODS):
        print(method, "worst", compare(method))
