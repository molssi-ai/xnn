"""Fidelity of the xnn QET to MatGL's QET (materialyzeai/matgl, BSD-3-Clause).

    python qet_matgl_fidelity.py [<MatGL QET directory> ...] [float64|float32]

Compares the energies, forces, stresses (eV/A^3) and charges of random periodic
cells of several elements and of a rattled silicon crystal, neutral, charged
and under an applied potential, one structure at a time with MatGL and as one
batch with xnn. The models are every published QET directory named on the
command line and MatGL QET models with random weights in the configurations
those do not cover: MatGL's defaults (Gaussian radial basis, one hardness per
element) and learned charge widths with the environment hardness. Each goes
through :func:`xnn.gnn.models.qet_foundation.convert_directory`. The last
printed line is the worst relative error; the charge errors are relative to the
largest charge or 0.1 e, whichever is larger, since the charges of the silicon
crystal are near zero. MatGL is an external oracle only.

Three MatGL details are set to the mathematics first: its softplus
(``torch.nn.Softplus``) drops the tail ``log(1 + e^-x)`` above ``x = 20``,
about 2e-9 in float64, and its Gaussian centers and covalent radii are rounded
to float32. The harness evaluates all three exactly.
"""
import sys
import tempfile

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk

from xnn.common.data import structure_to_graph
from xnn.common.data.dataset import collate
from xnn.common.models import ForceStressOutput
from xnn.gnn.models.qet_foundation import convert_directory

ELEMENTS = ("H", "Li", "C", "N", "O", "F", "Na", "Si", "P", "S", "Cl", "Ca", "Fe", "Ga", "Ag", "Cs", "Pu")


def random_cell(n, elements, a, rng, min_distance=1.6):
    """``n`` atoms placed at random in a slightly skewed cell, no two closer than ``min_distance``."""
    cell = np.eye(3) * a + rng.uniform(-0.4, 0.4, (3, 3))
    atoms = Atoms(cell=cell, pbc=True)
    while len(atoms) < n:
        trial = atoms + Atoms([elements[len(atoms) % len(elements)]],
                              positions=[rng.uniform(0, 1, 3) @ cell], cell=cell, pbc=True)
        if len(trial) == 1 or trial.get_distances(len(trial) - 1, range(len(trial) - 1), mic=True).min() > min_distance:
            atoms = trial
    return atoms


def cases(seed=0):
    """(atoms, total charge, applied potential or None)."""
    rng = np.random.default_rng(seed)
    out = [(random_cell(16, ["Na", "Cl", "Ca"], 6.5, rng), 0.0, None),
           (random_cell(18, ["Li", "P", "S", "Cl"], 6.5, rng), 1.0, None),
           (random_cell(12, ["Ga", "N", "O", "F"], 5.5, rng), -1.0, None),
           (random_cell(14, ["Ag", "Cs", "H", "C"], 7.0, rng), 0.0, rng.uniform(-0.5, 0.5, 14))]
    si = bulk("Si", "diamond", a=5.43, cubic=True)
    si.rattle(0.05, seed=seed)
    out.append((si, 0.0, None))
    return out


def random_matgl(folder, seed, **options):
    """A MatGL QET potential with random weights, saved to ``folder``."""
    from matgl.apps.pes import Potential
    from matgl.models import QET
    torch.manual_seed(seed)
    model = QET(element_types=ELEMENTS, **options)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in ("hardness_readout", "sigma"):
                p.copy_(torch.empty_like(p).uniform_(0.6, 1.8))
            elif "norm" in name:
                p.add_(0.1 * torch.randn_like(p))
    refs = torch.linspace(-3.0, -0.5, len(ELEMENTS), dtype=torch.float64)
    pot = Potential(model=model, data_std=torch.tensor(1.7), element_refs=refs, calc_stresses=True,
                    calc_charge=True)
    pot.save(folder)
    return folder


def exact_matgl(pot, dtype):
    """MatGL's softplus, Gaussian centers and fixed covalent radii evaluated exactly, in ``dtype``."""
    from ase.data import atomic_numbers, covalent_radii
    from matgl.layers._basis import GaussianExpansion
    model = pot.model
    for m in model.modules():
        if isinstance(m, torch.nn.Softplus):
            m.threshold = 40.0 if dtype == torch.float64 else 20.0
    rbf = model.bond_expansion.rbf
    if isinstance(rbf, GaussianExpansion):
        n = rbf.centers.numel()
        rbf.centers.data = torch.linspace(0.0, model.cutoff + 1.0, n, dtype=torch.float64).to(dtype)
    if not isinstance(model.sigma, torch.nn.Parameter):
        radii = [covalent_radii[atomic_numbers[e]] for e in model.element_types]
        model.sigma = torch.tensor(radii, dtype=torch.float64).to(dtype)
    return pot


def compare(name, path, dtype):
    import matgl
    from matgl.ext.ase import PESCalculator
    pot = exact_matgl(matgl.load_model(path).to(dtype), dtype)
    pot.model._use_warp = False
    model, _ = convert_directory(path, dtype)
    wrapped = ForceStressOutput(model.eval(), compute_stress=True)
    graphs, refs = [], []
    for atoms, charge, phi in cases():
        calc = PESCalculator(pot, stress_unit="eV/A3", total_charge=torch.tensor(charge, dtype=dtype),
                             ext_pot=None if phi is None else torch.tensor(phi, dtype=dtype))
        atoms.calc = calc
        refs.append({"energy": np.array([atoms.get_potential_energy()]), "forces": atoms.get_forces(),
                     "stress": atoms.get_stress(voigt=False), "charges": calc.results["charges"]})
        s = {"pos": atoms.positions, "atomic_numbers": atoms.numbers, "cell": np.asarray(atoms.cell),
             "pbc": [True] * 3, "total_charge": charge}
        if phi is not None:
            s["external_potential"] = phi
        graphs.append(structure_to_graph(s, model.cutoff))
    batch = collate(graphs)
    batch.compute_dtype = dtype
    out = wrapped(batch)
    worst = 0.0
    offsets = np.cumsum([0] + [g.num_nodes for g in graphs])
    for k, ref in enumerate(refs):
        a, b = offsets[k], offsets[k + 1]
        ours = {"energy": out["energy"][k:k + 1].detach().double().numpy(),
                "forces": out["forces"][a:b].detach().double().numpy(),
                "stress": out["stress"][k].detach().double().numpy(),
                "charges": out["charges"][a:b].detach().double().numpy()}
        for key in ("energy", "forces", "stress", "charges"):
            err = np.abs(ours[key] - ref[key]).max() / max(np.abs(ref[key]).max(), 0.1 if key == "charges" else 1e-12)
            print(f"{name} structure {k} {key}: relative error {err:.3e}")
            worst = max(worst, err)
    return worst


def main():
    args = sys.argv[1:]
    dtype = torch.float64
    if args and args[-1] in ("float64", "float32"):
        dtype = getattr(torch, args.pop())
    import matgl
    matgl.set_default_dtype("float", 64 if dtype == torch.float64 else 32)
    torch.set_default_dtype(torch.float64)
    worst = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        default = random_matgl(f"{tmp}/defaults", 0, units=32, nblocks=2)
        learned = random_matgl(f"{tmp}/learned_widths", 1, units=16, nblocks=3, rbf_type="SphericalBessel",
                               use_smooth=True, max_n=5, is_hardness_envs=True, is_sigma_train=True)
        for name, path in [("defaults", default), ("learned_widths", learned)] + [(p, p) for p in args]:
            worst = max(worst, compare(name, path, dtype))
    print(worst)


if __name__ == "__main__":
    main()
