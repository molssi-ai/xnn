"""Fidelity of the xnn (s)GDML to the sGDML package.

    python sgdml_fidelity.py [case ...]

Labelled geometries of an ethanol-like molecule come from a Morse pair potential
(built in code). For every case the sGDML package trains a model, which is loaded
into xnn (:func:`xnn.krnn.common.sgdml_io.load_sgdml_model`) and compared on test
geometries in float64 and float32. The package computes in float64 only (its torch
backend rounds positions to float32 first), so a float32 model is compared with the
package on float32-rounded positions. The fit is checked by refitting with
:func:`xnn.krnn.common.sgdml_fit.fit_sgdml` on the same training geometries and
permutations, in float64 and from float32-rounded labelled geometries, and the
symmetry search by comparing the recovered permutations. ``use_fast`` runs every
xnn prediction on the fused kernels instead of the reference.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch

from xnn.common.data import collate, structure_to_graph
from xnn.common.models import ForceStressOutput
from xnn.krnn.common.permutations import find_permutations
from xnn.krnn.common.sgdml_fit import fit_sgdml
from xnn.krnn.common.sgdml_io import load_sgdml_model

#: an ethanol geometry (Angstrom): C, C, O, H x 6
ETHANOL_Z = np.array([6, 6, 8, 1, 1, 1, 1, 1, 1])
ETHANOL_R = np.array([
    [-0.0012, -0.5585, 0.0000], [1.2280, 0.3374, 0.0000], [-1.1716, 0.2491, 0.0000],
    [0.0164, -1.2085, 0.8796], [0.0164, -1.2085, -0.8796], [1.2203, 0.9922, 0.8822],
    [1.2203, 0.9922, -0.8822], [2.1394, -0.2735, 0.0000], [-1.9472, -0.3229, 0.0000]])

#: case: (use_sym, use_E_cstr, sigma)
CASES = {"sym": (True, False, 4.0), "nosym": (False, False, 4.0), "ecstr": (True, True, 4.0)}


def have_sgdml() -> bool:
    """Whether the sGDML package is installed."""
    try:
        import sgdml  # noqa: F401
    except ImportError:
        return False
    return True


#: covalent radii (Angstrom) of the elements of the test molecule
RADII = {1: 0.31, 6: 0.76, 8: 0.66}
#: atom permutations that exchange equivalent hydrogens of the ethanol geometry (CH2 pair, CH3 rotations)
SYMMETRIES = [[0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 4, 3, 5, 6, 7, 8], [0, 1, 2, 3, 4, 6, 7, 5, 8],
              [0, 1, 2, 3, 4, 7, 5, 6, 8], [0, 1, 2, 4, 3, 6, 7, 5, 8]]


def morse(r: np.ndarray, z: np.ndarray):
    """Energy and forces of a Morse pair potential whose parameters depend on the elements only."""
    r = torch.tensor(r, dtype=torch.float64, requires_grad=True)
    i, j = torch.triu_indices(len(z), len(z), 1)
    rad = torch.tensor([RADII[int(a)] for a in z], dtype=torch.float64)
    zz = torch.tensor(z, dtype=torch.float64)
    r0 = rad[i] + rad[j]
    depth = 0.02 * torch.sqrt(zz[i] * zz[j])
    d = (r[:, i] - r[:, j]).norm(dim=-1)
    e = (depth * (1.0 - torch.exp(-1.5 * (d - r0))) ** 2).sum(-1)
    f = -torch.autograd.grad(e.sum(), r)[0]
    return e.detach().numpy(), f.numpy()


def dataset(n: int, seed: int, amplitude: float = 0.06, permute: bool = True) -> dict:
    """``n`` displaced ethanol geometries with Morse energies and forces (sGDML dataset layout).

    With ``permute`` the hydrogens of a geometry are relabelled by a random
    symmetry, as in a trajectory where the methyl group rotates.
    """
    rng = np.random.default_rng(seed)
    r = []
    for _ in range(n):
        p = SYMMETRIES[rng.integers(len(SYMMETRIES))] if permute else SYMMETRIES[0]
        r.append(ETHANOL_R[p] + amplitude * rng.standard_normal(ETHANOL_R.shape))
    r = np.array(r)
    e, f = morse(r, ETHANOL_Z)
    return {"type": np.array("d"), "name": np.array("morse_ethanol"), "theory": np.array("morse"), "z": ETHANOL_Z,
            "R": r, "E": e[:, None], "F": f, "r_unit": np.array("Ang"), "e_unit": np.array("kcal/mol"),
            "md5": np.array("0" * 32)}


_TRAINER = None


def upstream_model(data: dict, n_train: int, case: str) -> dict:
    """Train the sGDML package's model on ``data`` (fixed seed)."""
    global _TRAINER
    from sgdml.train import GDMLTrain
    use_sym, use_e_cstr, sig = CASES[case]
    np.random.seed(3)
    if _TRAINER is None:
        _TRAINER = GDMLTrain(max_processes=1)  # the package allows one instance per process
    train = _TRAINER
    task = train.create_task(data, n_train, data, 20, sig, lam=1e-10, use_sym=use_sym, use_E=True,
                             use_E_cstr=use_e_cstr)
    return train.train(task)


def upstream_predict(model: dict, r: np.ndarray, use_torch: bool = False):
    """Energies and forces from the package (its numpy backend, or its float32 torch backend)."""
    from sgdml.predict import GDMLPredict
    e, f = GDMLPredict(model, max_processes=1, use_torch=use_torch).predict(r.reshape(len(r), -1))
    return np.ravel(e), f.reshape(r.shape)


def _graphs(r, z, cutoff, dtype=torch.float64, energies=None, forces=None):
    out = []
    for k in range(len(r)):
        s = {"pos": torch.tensor(r[k], dtype=dtype), "atomic_numbers": torch.tensor(z)}
        if energies is not None:
            s["energy"] = float(np.ravel(energies)[k])
            s["forces"] = torch.tensor(forces[k], dtype=dtype)
        out.append(structure_to_graph(s, cutoff))
    return out


def xnn_predict(model, r, z, dtype=torch.float64, use_fast=False):
    model = model.to(dtype)
    model.set_use_fast(use_fast)
    out = ForceStressOutput(model)(collate(_graphs(r, z, model.cutoff, dtype)))
    return out["energy"].detach().double().numpy(), out["forces"].detach().double().numpy().reshape(r.shape)


def _rel(a, b) -> float:
    return float(np.abs(a - b).max() / np.abs(b).max())


def _rounded(a: np.ndarray) -> np.ndarray:
    """Values rounded to float32, kept as float64 (the package's input in float32)."""
    return np.asarray(a).astype(np.float32).astype(np.float64)


def _refit(path, data, idx, case, dtype, use_fast):
    """The model at ``path`` refitted by xnn on the training geometries ``data[idx]`` given in ``dtype``."""
    use_sym, use_e_cstr, sig = CASES[case]
    refit = load_sgdml_model(path)
    refit.train_weights.zero_()
    graphs = _graphs(data["R"][idx], data["z"], refit.cutoff, dtype, energies=data["E"][idx], forces=data["F"][idx])
    perms = np.asarray(np.load(path, allow_pickle=True)["perms"])
    fit_sgdml(refit, graphs, sigma=sig, lam=1e-10, perms=perms, energy_constraint=use_e_cstr)
    refit.set_use_fast(use_fast)
    return refit


def run_case(case: str, n_train: int = 60, use_fast: bool = False) -> dict:
    """Errors of the transplanted model, the refits and the permutations for one case."""
    data = dataset(400, 1)
    model = upstream_model(data, n_train, case)
    test = dataset(25, 2)
    e_ref, f_ref = upstream_predict(model, test["R"])
    rows = {}
    try:
        e_t, f_t = upstream_predict(model, test["R"], use_torch=True)
        rows["upstream_f32"] = {"energy": float(np.abs(e_t - e_ref).max() / np.abs(e_ref).max()),
                                "forces": float(np.abs(f_t - f_ref).max() / np.abs(f_ref).max())}
    except TypeError:
        pass  # the package's torch backend does not take models with energy coefficients
    tmp = tempfile.TemporaryDirectory()
    path = os.path.join(tmp.name, "model.npz")
    np.savez(path, **model)
    # the reference with its training terms in reverse order: the model's own spread under reordering
    fwd = xnn_predict(load_sgdml_model(path), test["R"], test["z"])
    rev = load_sgdml_model(path)
    for name in ("train_desc", "train_weights", "energy_coef"):
        setattr(rev, name, getattr(rev, name).flip(0))
    e_rev, f_rev = xnn_predict(rev, test["R"], test["z"])
    rows["f64_reorder"] = {"energy": _rel(e_rev, fwd[0]), "forces": _rel(f_rev, fwd[1])}
    r32 = _rounded(test["R"])
    e_up32, f_up32 = upstream_predict(model, r32)
    for name, dtype in (("f64", torch.float64), ("f32", torch.float32)):
        xm = load_sgdml_model(path)
        e, f = xnn_predict(xm, test["R"], test["z"], dtype, use_fast)
        rows[name] = {"energy": float(np.abs(e - e_ref).max() / np.abs(e_ref).max()),
                      "forces": float(np.abs(f - f_ref).max() / np.abs(f_ref).max())}
        if name == "f32" and "upstream_f32" in rows:
            # both round the positions to float32 and compute in float64
            rows["f32_vs_upstream_f32"] = {"energy": float(np.abs(e - e_t).max() / np.abs(e_t).max()),
                                           "forces": float(np.abs(f - f_t).max() / np.abs(f_t).max())}
        if name == "f32":
            # the package's float64 arithmetic on the float32 positions: the float32 oracle of every case
            rows["f32_vs_upstream"] = {"energy": _rel(e, e_up32), "forces": _rel(f, f_up32)}
    use_sym = CASES[case][0]
    idx = np.asarray(model["idxs_train"])
    refit = _refit(path, data, idx, case, torch.float64, use_fast)
    e, f = xnn_predict(refit, test["R"], test["z"], torch.float64, use_fast)
    w_ref = np.asarray(model["R_d_desc_alpha"])
    rows["refit"] = {"energy": float(np.abs(e - e_ref).max() / np.abs(e_ref).max()),
                     "forces": float(np.abs(f - f_ref).max() / np.abs(f_ref).max()),
                     "weights": float(np.abs(refit.train_weights.numpy() - w_ref).max() / np.abs(w_ref).max()),
                     "c": abs(float(refit.c) - float(model["c"])), "std": abs(float(refit.std) - float(model["std"]))}
    # float32 training data: the package trains on the rounded labelled geometries, xnn on float32 graphs
    data32 = dict(data, R=_rounded(data["R"]), E=_rounded(data["E"]), F=_rounded(data["F"]))
    model32 = upstream_model(data32, n_train, case)
    path32 = os.path.join(tmp.name, "model32.npz")
    np.savez(path32, **model32)
    idx32 = np.asarray(model32["idxs_train"])
    refit32 = _refit(path32, data32, idx32, case, torch.float32, use_fast)
    e, f = xnn_predict(refit32, test["R"], test["z"], torch.float32, use_fast)
    e_up, f_up = upstream_predict(model32, r32)
    w_up = np.asarray(model32["R_d_desc_alpha"])
    rows["refit_f32"] = {"energy": _rel(e, e_up), "forces": _rel(f, f_up),
                         "weights": float(np.abs(refit32.train_weights.numpy() - w_up).max() / np.abs(w_up).max()),
                         "same_train": float(np.array_equal(idx32, idx))}
    if use_sym:
        mine = find_permutations(data["R"][idx], data["z"])
        rows["perms"] = {"same_group": float(set(map(tuple, mine.tolist()))
                                             == set(map(tuple, np.asarray(model["perms"]).tolist()))),
                         "n_upstream": float(len(model["perms"])), "n_xnn": float(len(mine))}
    tmp.cleanup()
    return rows


def main(argv=None) -> int:
    names = list(sys.argv[1:] if argv is None else argv) or list(CASES)
    worst = worst32 = 0.0
    for name in names:
        rows = run_case(name)
        for mode, errs in rows.items():
            print(f"{name:6s} {mode:6s} " + "  ".join(f"{k} {v:.3e}" for k, v in errs.items()), flush=True)
        worst = max(worst, max(rows["f64"].values()))
        worst32 = max(worst32, max(rows["f32_vs_upstream"].values()))
    print(f"worst f64 relative error {worst:.3e}")
    print(f"worst f32 relative error (rounded positions) {worst32:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
