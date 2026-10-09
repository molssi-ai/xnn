"""Parity of the xnn HDNNP (2G, 3G, 4G) with the RuNNer code.

    XNN_RUNNER=/path/to/RuNNer.x python hdnnp_runner_parity.py [case ...] [--keep DIR]

For every case a RuNNer model is generated with random weights and scaling
(``input.nn``, ``weights_*.data``, ``scaling.data``, ``qeq_scaling.data``)
together with random structures (``input.data``); the RuNNer executable
predicts their energies, forces and charges, and the same files are read by
:func:`xnn.dnn.common.runner.load_runner_model` and evaluated in float64.
The comparison is in RuNNer's atomic units. The case ``au2mgo`` instead runs
the 4G model of RuNNer's own regression tests (``XNN_RUNNER_SRC``: the
unpacked RuNNer source tree). The last printed line is the worst relative
error over all cases (energies relative to the largest ``|E|``, forces and
charges to the largest component). RuNNer is used as an external oracle only.
"""
from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from xnn.common.data import collate, structure_to_graph
from xnn.common.data.elements import CHEMICAL_SYMBOLS
from xnn.common.data.hub.units import BOHR_TO_ANGSTROM, HARTREE_TO_EV
from xnn.common.data.runner_io import read_runner_data, write_runner_data
from xnn.common.models import ForceStressOutput
from xnn.dnn.common.runner import load_runner_model

ACTS = ("t", "s", "p", "st", "sq", "l")


def _cutoff_lines(rc: float) -> list[str]:
    """One cutoff function of every kind (indices 1..5)."""
    return [f"fc_cosine 1 0.8 {rc}", f"fc_hypertangent 2 {rc}", f"fc_hypertangent_approx 3 0.5 {rc}",
            f"fc_polynomial 4 3 {rc}", f"fc_cosine 5 {rc - 1.5}"]


def _symfunctions(elements: list[str], rng, rich: bool) -> list[str]:
    """Radial and angular functions of every type, listed in a shuffled order."""
    lines = []
    for a in elements:
        for b in elements:
            for k, eta in enumerate((0.0, 0.03, 0.2)):
                lines.append(f"{a} 2 {b} {eta + 0.001 * k:.6f} {0.5 * k:.4f} {1 + k % 5}")
        if rich:
            lines.append(f"{a} 1 {elements[0]} 2 5")
        for i, b in enumerate(elements):
            for c in elements[i:]:
                for lam, zeta, cut in ((1.0, 1.0, 1), (-1.0, 2.0, 2), (1.0, 4.0, 4)):
                    lines.append(f"{a} 3 {b} {c} {0.01 + 0.005 * zeta:.4f} {lam} {zeta} {cut}")
                lines.append(f"{a} 9 {b} {c} 0.02 -1.0 1.5 3")
                if rich:
                    lines.append(f"{a} 8 {b} {c} 105.0 0.0005 1")
    order = rng.permutation(len(lines))
    return [lines[i] for i in order]


def _random_weights(n_in: int, nodes: list[int], rng) -> list[float]:
    vals, d = [], n_in
    for h in nodes + [1]:
        vals += list(rng.uniform(-1.0, 1.0, d * h) / np.sqrt(d))
        vals += list(rng.uniform(-0.3, 0.3, h))
        d = h
    return vals


def _features_per_element(lines: list[str], elements: list[str]) -> dict[str, int]:
    count = {e: 0 for e in elements}
    for line in lines:
        tok = line.split()
        count[tok[0]] += int(tok[3]) if tok[1] == "1" and len(tok) == 5 else 1
    return count


def _structures(elements, rng, n_mol: int, n_cell: int, charges=(0,)) -> list[dict]:
    """Random molecules and periodic cells (atomic units), atoms at least 2 bohr apart."""
    out = []
    zs = [CHEMICAL_SYMBOLS.index(e) for e in elements]

    def place(n, box):
        pos = []
        while len(pos) < n:
            p = rng.uniform(0, box, 3)
            if all(np.linalg.norm(p - q) > 2.2 for q in pos):
                pos.append(p)
        return np.array(pos)

    for i in range(n_mol):
        n = int(rng.integers(5, 9))
        out.append({"pos": place(n, 6.5), "atomic_numbers": np.array([zs[k % len(zs)] for k in range(n)]),
                    "total_charge": float(charges[i % len(charges)])})
    for _ in range(n_cell):
        n = int(rng.integers(8, 12))
        box = 10.5
        cell = np.diag([box, box * 1.05, box * 0.97]) + rng.normal(scale=0.3, size=(3, 3))
        frac = place(n, 1.0 * box) / box
        out.append({"pos": frac @ cell, "atomic_numbers": np.array([zs[k % len(zs)] for k in range(n)]),
                    "cell": cell, "pbc": np.ones(3, bool), "total_charge": 0.0})
    for s in out:
        s["energy"] = 0.0
        s["forces"] = np.zeros_like(s["pos"])
        s["charges"] = np.full(len(s["atomic_numbers"]), s["total_charge"] / len(s["atomic_numbers"]))
    return out


CASES = {
    # name: generation, elements, molecules, cells, total charges, extra input.nn lines
    "2g_molecules": (2, ["H", "C", "O", "F"], 5, 0, (0,), ["scale_feature_maps", "center_feature_maps"]),
    "2g_cells": (2, ["H", "O"], 0, 3, (0,), ["scale_feature_maps"]),
    "2g_range": (2, ["H", "C"], 3, 1, (0,), ["scale_feature_maps_range -1.0 1.0"]),
    "3g_molecules": (3, ["H", "O"], 5, 0, (0, 1, -1), ["center_feature_maps", "write_charge_out"]),
    "3g_screened": (3, ["H", "C", "O"], 4, 0, (0, -1), ["scale_feature_maps", "center_feature_maps",
                                                        "screening_function 5", "write_charge_out",
                                                        "fixed_gausswidth H 1.1", "fixed_gausswidth O 1.6"]),
    "3g_cells": (3, ["H", "C", "O"], 0, 3, (0,), ["scale_feature_maps", "center_feature_maps",
                                                  "screening_function 5",
                                                  "fixed_gausswidth H 1.1", "fixed_gausswidth O 1.6"]),
    "3g_point": (3, ["H", "O"], 3, 0, (0, 1), ["fixed_gausswidth H 0.0", "fixed_gausswidth O 0.0",
                                               "write_charge_out"]),
    "3g_point_cells": (3, ["H", "O"], 0, 2, (0,), ["fixed_gausswidth H 0.0", "fixed_gausswidth O 0.0"]),
    "4g_molecules": (4, ["H", "C", "O"], 6, 0, (0, 1, -1), ["scale_feature_maps", "center_feature_maps",
                                                            "model_type_hardness elemental"]),
    "4g_cells": (4, ["Na", "Cl"], 2, 2, (0,), ["model_type_hardness elemental", "screening_function 5"]),
    "4g_hardness_nn": (4, ["H", "O"], 4, 0, (0, -1), ["center_feature_maps"]),
    "4g_hardness_nn_cells": (4, ["H", "O"], 0, 2, (0,), ["center_feature_maps"]),
    # separate descriptors of the short-range and the charge / electronegativity networks
    "3g_separate": (3, ["H", "C", "O"], 4, 0, (0, 1), ["scale_feature_maps", "center_feature_maps",
                                                      "write_charge_out"]),
    "4g_separate": (4, ["H", "C", "O"], 4, 0, (0, -1), ["center_feature_maps",
                                                       "model_type_hardness elemental"]),
}


# quantities where RuNNer itself is inconsistent: with separate charge descriptors its 3G
# analytic forces are not the gradient of its energies (central differences of the RuNNer
# energies agree with the xnn forces)
UNRELIABLE = {"3g_separate": {"forces"}}


def make_case(name: str, directory: Path, seed: int = 0) -> list[dict]:
    """Write the RuNNer model and data of a case into ``directory``."""
    gen, elements, n_mol, n_cell, charges, extra = CASES[name]
    rng = np.random.default_rng(seed + sum(map(ord, name)))
    rich = gen == 2
    rc = 8.0
    sf = _symfunctions(elements, rng, rich)
    nfeat = _features_per_element(sf, elements)
    separate = name.endswith("_separate")
    q_suffix = {3: "charge", 4: "chi"}.get(gen)
    sf_q = [line for line in sf if line.split()[1] == "2"] if separate else sf
    nfeat_q = _features_per_element(sf_q, elements)
    nodes = {e: [int(rng.integers(4, 8)), int(rng.integers(3, 7))] for e in elements}
    acts = {e: [ACTS[int(rng.integers(0, 5))], ACTS[int(rng.integers(0, 5))], "l"] for e in elements}
    lines = ["runner_mode predict", f"nnp_generation {gen}", "elements " + " ".join(elements),
             "calculate_forces", "no_sf_groups", "elec_method ewald", "qeq_method direct",
             "electrostatics_precision 1e-13", "use_old_scaling", "initialization_method read"]
    if not separate:
        lines.append("feature_map_default")
    lines += _cutoff_lines(rc)
    for e in elements:
        lines.append(f"nodes {e} " + " ".join(map(str, nodes[e])))
        lines.append(f"activation_nn {e} " + " ".join(acts[e]))
        lines.append(f"atom_energy {e} {rng.uniform(-50, -1):.10f}")
    lines += extra
    if separate:
        lines += [f"symfunction_short {line}" for line in sf]
        lines += [f"symfunction_{q_suffix} {line}" for line in sf_q]
    else:
        lines += [f"symfunction {line}" for line in sf]
    (directory / "input.nn").write_text("\n".join(lines) + "\n")

    zs = sorted(CHEMICAL_SYMBOLS.index(e) for e in elements)
    by_z = {CHEMICAL_SYMBOLS.index(e): e for e in elements}
    def scaling(counts, path):
        rows = []
        for i, z in enumerate(zs):
            for k in range(counts[by_z[z]]):
                lo = rng.uniform(0.0, 0.5)
                hi = lo + rng.uniform(0.5, 3.0)
                rows.append(f"{i + 1:4d} {k + 1:4d} {lo:.15e} {hi:.15e} {rng.uniform(lo, hi):.15e}")
        path.write_text("\n".join(rows) + "\n")

    if separate:
        scaling(nfeat, directory / "scaling_short.data")
        scaling(nfeat_q, directory / f"scaling_{q_suffix}.data")
    else:
        scaling(nfeat, directory / "scaling.data")

    for z in zs:
        e = by_z[z]
        extra_in = 1 if gen == 4 else 0
        w = _random_weights(nfeat[e] + extra_in, nodes[e], rng)
        (directory / f"weights_short.{z:03d}.data").write_text("\n".join(f"{v:.17e}" for v in w) + "\n")
        if gen >= 3:
            name_q = "weights_charge" if gen == 3 else "weights_chi"
            w = _random_weights(nfeat_q[e], nodes[e], rng)
            if gen == 3:
                w = [0.3 * v for v in w]
            (directory / f"{name_q}.{z:03d}.data").write_text("\n".join(f"{v:.17e}" for v in w) + "\n")
        if gen == 4 and "model_type_hardness elemental" not in extra:
            w = _random_weights(nfeat[e], nodes[e], rng)
            # output layer = the last n_last weights and one bias: J near 0.8 hartree
            n_last = nodes[e][-1]
            w[-1 - n_last:-1] = [0.05 * v for v in w[-1 - n_last:-1]]
            w[-1] = 0.8
            (directory / f"weights_hardness.{z:03d}.data").write_text("\n".join(f"{v:.17e}" for v in w) + "\n")
    if gen == 4:
        if "model_type_hardness elemental" in extra:
            (directory / "weights_hardness.data").write_text(
                "\n".join(f"{v:.17e}" for v in rng.uniform(0.4, 1.0, len(zs))) + "\n")
        (directory / "qeq_scaling.data").write_text("\n".join(
            f"1 {z} 0.5 -0.5 0.0 0.1 10" for z in zs) + "\n")
    structures = _structures(elements, rng, n_mol, n_cell, charges)
    write_runner_data(structures, directory / "input.data", units="atomic")
    return structures


def _table(path: Path) -> list[dict]:
    with open(path) as fh:
        lines = [line for line in fh if line.strip()]
    reader = csv.reader(lines, delimiter="\t")
    rows = list(reader)
    head = [h.strip() for h in rows[0]]
    return [dict(zip(head, [v.strip() for v in r])) for r in rows[1:]]


def run_runner(directory: Path) -> dict:
    """Run the RuNNer executable in ``directory`` and read its predictions (atomic units)."""
    exe = os.environ["XNN_RUNNER"]
    proc = subprocess.run([exe], cwd=directory, capture_output=True, text=True)
    (directory / "runner.out").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0 or not (directory / "energy.out").exists():
        text = proc.stdout + proc.stderr
        errors = [line for line in text.splitlines() if "error" in line.lower() or "warning" in line.lower()]
        raise RuntimeError(f"RuNNer failed in {directory}:\n" + "\n".join(errors[-30:]) + "\n" + text[-1500:])
    energy = [float(r["total_energy_pred[Ha]"]) for r in _table(directory / "energy.out")]
    forces = np.array([[float(r[f"total_forces_{a}_pred[Ha/Bohr]"]) for a in "xyz"]
                       for r in _table(directory / "forces.out")])
    out = {"energy": np.array(energy), "forces": forces}
    if (directory / "charge.out").exists():
        rows = _table(directory / "charge.out")
        key = "constrained_charge_pred[e]" if "constrained_charge_pred[e]" in rows[0] else "charge_pred[e]"
        out["charges"] = np.array([float(r[key]) for r in rows])
    return out


def run_xnn(directory: Path) -> dict:
    """The same predictions with the xnn model read from ``directory`` (float64)."""
    model = load_runner_model(directory)
    calc = ForceStressOutput(model).double()
    structures = read_runner_data(directory / "input.data", charges=False)
    energies, forces, charges = [], [], []
    for s in structures:
        g = structure_to_graph(dict(s, pos=np.asarray(s["pos"], dtype=np.float64)), model.cutoff)
        g = collate([g])
        g.pos = g.pos.double()
        if g.cell is not None:
            g.cell = g.cell.double()
        out = calc(g)
        energies.append(float(out["energy"][0]) / HARTREE_TO_EV)
        forces.append(out["forces"].detach().numpy() * BOHR_TO_ANGSTROM / HARTREE_TO_EV)
        if "charges" in out:
            charges.append(out["charges"].detach().numpy())
    res = {"energy": np.array(energies), "forces": np.concatenate(forces)}
    if charges:
        res["charges"] = np.concatenate(charges)
    return res


def compare(name: str, ref: dict, got: dict) -> float:
    worst = 0.0
    for key in ("energy", "forces", "charges"):
        if key not in ref or key not in got:
            continue
        if key in UNRELIABLE.get(name, ()):
            print(f"{name:16s} {key:8s} not compared (RuNNer's values are inconsistent here)")
            continue
        a, b = np.asarray(ref[key]), np.asarray(got[key])
        if a.shape != b.shape:
            raise RuntimeError(f"{name} {key}: shapes {a.shape} vs {b.shape}")
        scale = max(np.abs(a).max(), 1e-12)
        err = float(np.abs(a - b).max() / scale)
        worst = max(worst, err)
        print(f"{name:16s} {key:8s} max|ref| {scale:.3e}  max|diff| {np.abs(a - b).max():.3e}  rel {err:.3e}")
    return worst


def run_case(name: str, keep: Path | None = None) -> float:
    before = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        return _run_case(name, keep)
    finally:
        torch.set_default_dtype(before)


def _run_case(name: str, keep: Path | None) -> float:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        if name == "au2mgo":
            src = Path(os.environ["XNN_RUNNER_SRC"]) / "tests" / "regression" / "lammps" / "4G"
            for f in src.iterdir():
                if f.is_file():
                    shutil.copy(f, d / f.name)
        else:
            make_case(name, d)
        try:
            ref = run_runner(d)
            got = run_xnn(d)
        finally:
            if keep is not None:
                shutil.copytree(d, keep / name, dirs_exist_ok=True)
        return compare(name, ref, got)


def main(argv: list[str]) -> None:
    keep = None
    if "--keep" in argv:
        i = argv.index("--keep")
        keep = Path(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    names = argv or list(CASES) + (["au2mgo"] if os.environ.get("XNN_RUNNER_SRC") else [])
    worst = 0.0
    for name in names:
        try:
            worst = max(worst, run_case(name, keep))
        except Exception as err:          # report every case, then the failure
            print(f"{name:16s} FAILED: {err}")
            worst = float("inf")
    print(f"worst relative error {worst:.3e}")


if __name__ == "__main__":
    main(sys.argv[1:])
