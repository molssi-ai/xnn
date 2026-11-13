"""Fidelity of the xnn MEGNet to MatGL's MEGNet (materialyzeai/matgl, BSD-3-Clause).

    python megnet_matgl_fidelity.py <MatGL MEGNet directory> [...] [float64|float32]

Loads every published MatGL MEGNet model (``model.json``, ``model.pt`` and
``state.pt``) twice: with MatGL and through
:func:`xnn.gnn.models.megnet_foundation.convert_directory`. Compares the
predicted property of random periodic cells of several elements, a rattled
silicon crystal and a molecule, one at a time and as one batch, at every state
label of a multi-fidelity model. The errors are relative to the property or
to 1 (eV, eV/atom), whichever is larger, since band gaps of metals are near
zero. The last printed line is the worst. MatGL is an external oracle only.

Two MatGL details are set to the paper's mathematics first: its activation
(``torch.nn.Softplus`` minus ``ln 2``) drops the tail ``log(1 + e^-x)`` above
``x = 20``, about 2e-9 in float64, and its checkpoints store the Gaussian
centers rounded to float32. The harness evaluates both exactly.
"""
import math
import sys

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk, molecule

from xnn.common.data import structure_to_graph
from xnn.common.data.dataset import collate
from xnn.gnn.models.megnet_foundation import convert_directory


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


def structures(seed=0):
    rng = np.random.default_rng(seed)
    out = [random_cell(12, ["Si", "O"], 6.0, rng), random_cell(16, ["Li", "Fe", "P", "O"], 6.5, rng),
           random_cell(10, ["Ga", "N", "Al"], 5.5, rng)]
    si = bulk("Si", "diamond", a=5.43, cubic=True)
    si.rattle(0.05, seed=seed)
    out.append(si)
    out.append(molecule("CH3CH2OH"))
    return out


def exact_matgl(model, dtype):
    """MatGL's activation and Gaussian centers evaluated exactly, in ``dtype``."""
    from matgl.layers._activations import SoftPlus2

    def forward(self, x):
        return torch.nn.functional.softplus(x, threshold=40.0 if x.dtype == torch.float64 else 20.0) \
            - math.log(2.0)
    SoftPlus2.forward = forward
    inner = model.model
    rbf = inner.bond_expansion.rbf
    n = rbf.centers.numel()
    rbf.centers.data = torch.linspace(float(rbf.centers[0]), inner.cutoff + 1.0, n, dtype=dtype)
    return model


def main():
    args = sys.argv[1:]
    dtype = torch.float64
    if args and args[-1] in ("float64", "float32"):
        dtype = getattr(torch, args.pop())
    import matgl
    from matgl.data.transformer import Normalizer
    from matgl.models._core import structure_to_graph as matgl_graph
    matgl.set_default_dtype("float", 64 if dtype == torch.float64 else 32)
    torch.set_default_dtype(dtype)
    worst = 0.0
    for path in args:
        # MatGL's own loader reads model.pt with weights_only=True, which refuses its Normalizer
        with torch.serialization.safe_globals([Normalizer]):
            upstream = matgl.load_model(path)
        upstream = exact_matgl(upstream.to(dtype), dtype)
        upstream.eval()
        ours, _ = convert_directory(path, dtype)
        ours.eval()
        n_states = ours.n_state_types
        labels = [None] if n_states is None else list(range(n_states))
        cases = structures()
        for label in labels:
            graphs, refs = [], []
            for atoms in cases:
                # MatGL's own graph and default state; the wrapper applies the standardization
                g, default = matgl_graph(upstream.model, atoms)
                state = default if label is None else torch.tensor([label])
                with torch.no_grad():
                    ref = float(upstream(g, state))
                s = {"pos": atoms.positions, "atomic_numbers": atoms.numbers}
                if atoms.pbc.any():
                    s.update(cell=np.asarray(atoms.cell), pbc=[True] * 3)
                if label is not None:
                    s["state"] = label
                g = structure_to_graph(s, ours.cutoff)
                graphs.append(g)
                refs.append(ref)
                mine = float(ours(g)["properties"])
                err = abs(mine - ref) / max(abs(ref), 1.0)
                print(f"{path} state {label} {atoms.get_chemical_formula()}: {mine:.12g} vs {ref:.12g}, "
                      f"relative error {err:.3e}")
                worst = max(worst, err)
            batched = ours(collate(graphs))["properties"].detach().double().numpy()
            err = float(np.max(np.abs(batched - np.array(refs)) / np.maximum(np.abs(refs), 1.0)))
            print(f"{path} state {label} batch of {len(graphs)}: relative error {err:.3e}")
            worst = max(worst, err)
    print(worst)


if __name__ == "__main__":
    main()
