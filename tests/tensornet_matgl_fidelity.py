"""Fidelity of the xnn TensorNet to MatGL's TensorNet (materialsvirtuallab/matgl, BSD-3-Clause).

    python tensornet_matgl_fidelity.py <MatGL TensorNet directory> [float64|float32]

Loads a published MatGL TensorNet potential (``model.json`` and ``state.pt``)
twice: with MatGL, on its plain PyTorch path, and through
:func:`xnn.gnn.models.tensornet_foundation.convert_directory`. Compares the
energies, forces and stresses (eV/A^3) of random periodic cells of several
elements and of a rattled silicon crystal. The last printed line is the worst
relative error. MatGL is an external oracle only.
"""
import sys

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk

from xnn.common.data import structure_to_graph
from xnn.common.models import ForceStressOutput
from xnn.gnn.models.tensornet_foundation import convert_directory


def random_cell(n, elements, a, rng, min_distance=1.5):
    """``n`` atoms placed at random in a slightly skewed cell, no two closer than ``min_distance``."""
    cell = np.eye(3) * a + rng.uniform(-0.4, 0.4, (3, 3))
    atoms = Atoms(cell=cell, pbc=True)
    while len(atoms) < n:
        trial = atoms + Atoms([elements[len(atoms) % len(elements)]],
                              positions=[rng.uniform(0, 1, 3) @ cell], cell=cell, pbc=True)
        if len(trial) == 1 or trial.get_distances(len(trial) - 1, range(len(trial) - 1), mic=True).min() > min_distance:
            atoms = trial
    return atoms


def structures(seed=0):
    rng = np.random.default_rng(seed)
    out = [random_cell(24, ["Si", "O"], 7.0, rng), random_cell(28, ["Li", "Fe", "P", "O"], 7.5, rng),
           random_cell(16, ["H", "C", "N", "O"], 6.5, rng)]
    si = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
    si.rattle(0.05, seed=seed)
    out.append(si)
    return out


def main():
    path = sys.argv[1]
    dtype = getattr(torch, sys.argv[2] if len(sys.argv) > 2 else "float64")
    import matgl
    from matgl.ext.ase import PESCalculator
    matgl.set_default_dtype("float", 64 if dtype == torch.float64 else 32)
    pot = matgl.load_model(path).to(dtype)
    pot.model._use_warp = False
    calc = PESCalculator(pot, stress_unit="eV/A3")
    torch.set_default_dtype(torch.float64)
    model, _ = convert_directory(path, dtype)
    model = model.eval()
    wrapped = ForceStressOutput(model, compute_stress=True)
    worst = 0.0
    for k, atoms in enumerate(structures()):
        atoms.calc = calc
        ref = {"energy": np.array([atoms.get_potential_energy()]), "forces": atoms.get_forces(),
               "stress": atoms.get_stress(voigt=False)}
        g = structure_to_graph({"pos": atoms.positions, "atomic_numbers": atoms.numbers,
                                "cell": np.asarray(atoms.cell), "pbc": [True] * 3}, model.cutoff)
        out = wrapped(g)
        ours = {"energy": out["energy"].detach().double().numpy(),
                "forces": out["forces"].detach().double().numpy(),
                "stress": out["stress"][0].detach().double().numpy()}
        for key in ("energy", "forces", "stress"):
            err = np.abs(ours[key] - ref[key]).max() / max(np.abs(ref[key]).max(), 1e-12)
            print(f"structure {k} ({atoms.get_chemical_formula()}) {key}: relative error {err:.3e}")
            worst = max(worst, err)
    print(worst)


if __name__ == "__main__":
    main()
