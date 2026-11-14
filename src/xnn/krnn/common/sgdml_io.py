"""Read sGDML model files (the ``.npz`` files of the sGDML package)."""
from __future__ import annotations

import os

import numpy as np
import torch

from ..models.sgdml import SGDML


def load_sgdml_model(path, cutoff: float = 1.0) -> SGDML:
    """Load a trained (s)GDML model from an sGDML model file.

    Parameters
    ----------
    path : str or os.PathLike
        The ``.npz`` model file.
    cutoff : float, optional
        Neighbour-list radius of the data pipeline, by default 1.0.

    Returns
    -------
    SGDML
        The model in float64, in eval mode.
    """
    with np.load(os.fspath(path), allow_pickle=True) as f:
        m = {k: f[k] for k in f.files}
    perms = np.asarray(m["perms"], dtype=np.int64)
    model = SGDML([int(z) for z in m["z"]], sigma=float(m["sig"]), use_symmetries=perms.shape[0] > 1,
                  cutoff=cutoff)
    model.set_permutations(perms)
    model.train_desc = torch.as_tensor(np.asarray(m["R_desc"]).T.copy(), dtype=torch.float64)
    model.train_weights = torch.as_tensor(np.asarray(m["R_d_desc_alpha"]), dtype=torch.float64)
    if "alphas_E" in m:
        model.energy_coef = torch.as_tensor(np.asarray(m["alphas_E"]).ravel(), dtype=torch.float64)
    model.std = torch.tensor(float(m["std"]) if "std" in m else 1.0, dtype=torch.float64)
    model.c = torch.tensor(float(m["c"]), dtype=torch.float64)
    if "lattice" in m:
        # the file stores the cell vectors as columns
        model.lattice = torch.as_tensor(np.asarray(m["lattice"]).T.copy(), dtype=torch.float64)
    return model.double().eval()
