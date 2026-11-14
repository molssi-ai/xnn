"""The QET foundation potentials of MatGL, converted weight for weight.

MatGL (BSD-3-Clause) publishes QET potentials trained on MatPES as a
directory of ``model.json`` (the architecture) and ``state.pt`` (the weights).
They convert to :class:`~xnn.gnn.models.qet.QET` with the trunk mapped as for
TensorNet (:mod:`xnn.gnn.models.tensornet_foundation`) and the charge head
layer by layer: the electronegativity and hardness networks, the layer norm of
the energy network's input and the per-element tables, re-indexed by atomic
number. The energy is ``data_std`` times the network's sum plus the
per-element references, which map onto ``energy_scale`` and ``atom_ref``.

``QET.from_foundation("qet-matpes-pbe-2025.2")`` downloads a pinned
revision into the model cache and converts it; a local directory loads
through the model hub's ``"matgl"`` format
(``xnn.common.models.from_pretrained(path)``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch

from xnn.common.config import Config

from .qet import COVALENT_RADII, QET
from .tensornet_foundation import (
    build_converted,
    fetch_foundation,
    potential_options,
    read_matgl_directory,
    transplant,
    trunk_options,
)

#: Registered MatGL QET checkpoints: name -> (repository, pinned revision).
FOUNDATION_MODELS = {
    "qet-matpes-pbe-2025.2": ("materialyze/QET-PES-MatPES-PBE-2025.2",
                              "b6e610e509f53c9c475c81a241e42eda97a249eb"),
    "qet-matpes-r2scan-2025.2": ("materialyze/QET-PES-MatPES-r2SCAN-2025.2",
                                 "98bfd77c3ec49c574f3f348213f073e2ba1e7704"),
}


def _dense(state: dict, theirs: str, ours: str, n: int = 3) -> dict:
    """The dense layers of a MatGL MLP (at 0, 2, 4, ... of its stack) under xnn's names."""
    return {f"{ours}.{i}.{p}": state[f"model.{theirs}.{2 * i}.{p}"] for i in range(n) for p in ("weight", "bias")}


def _by_atomic_number(own: torch.Tensor, values: torch.Tensor, species: list) -> torch.Tensor:
    """A per-element table indexed by atomic number with the checkpoint's rows at ``species``."""
    table = own.clone()
    table[torch.tensor(species)] = values.reshape(-1).to(table.dtype)
    return table


def convert_directory(path: Union[str, Path], dtype=None) -> tuple:
    """Convert a MatGL QET directory into xnn's QET.

    Parameters
    ----------
    path : str or pathlib.Path
        The directory of ``model.json`` and ``state.pt`` (or either file in it).
    dtype : torch.dtype, optional
        The dtype the model is built in, by default the checkpoint's.

    Returns
    -------
    tuple
        The :class:`~xnn.gnn.models.qet.QET` and the
        :class:`~xnn.common.config.Config` that rebuilds it.

    Raises
    ------
    NotImplementedError
        For an architecture outside the converted ones (see
        :func:`~xnn.gnn.models.tensornet_foundation.convert_directory`, and a
        magnetic-moment head).
    ValueError
        When the checkpoint's fixed charge widths are not the covalent radii.
    """
    path, kwargs, a, state = read_matgl_directory(path, "QET")
    unsupported = []
    extra, n_rbf = trunk_options(kwargs, a, unsupported)
    if a.get("include_magmom", False):
        unsupported.append("a magnetic-moment head")
    if unsupported:
        raise NotImplementedError(f"{path}: not converted: {', '.join(unsupported)}")
    environment = bool(a.get("is_hardness_envs", False))
    trainable_widths = bool(a.get("is_sigma_train", False))
    extra.update(potential_options(path, a, state), readout="gated",
                 hardness="environment" if environment else "element",
                 trainable_widths=trainable_widths)
    model, model_cfg = build_converted(QET, "qet", a, extra, n_rbf, state, dtype)
    species = extra["species"]
    own = model.state_dict()

    weights = _dense(state, "chi_readout.layers", "chi_readout.layers")
    if environment:
        weights.update(_dense(state, "hardness_readout.layers", "hardness_readout.mlp.layers"))
    else:
        weights["hardness_readout.weight"] = _by_atomic_number(
            own["hardness_readout.weight"], state["model.hardness_readout"], species)
    if trainable_widths:
        weights["sigma"] = _by_atomic_number(own["sigma"], state["model.sigma"], species)
    else:
        # fixed widths are the covalent radii, which xnn keeps exact in float64
        radii = torch.tensor([COVALENT_RADII[z] for z in species], dtype=torch.float64)
        stored = state["model.sigma"].reshape(-1).double()
        if stored.shape != radii.shape or not torch.allclose(stored, radii, rtol=0.0, atol=1e-6):
            raise ValueError(f"{path}: the checkpoint's charge widths are not the covalent radii")
    for p in ("weight", "bias"):
        weights[f"readout_norm.{p}"] = state[f"model.norm.{p}"]
    transplant(model, state, species, int(a["nblocks"]), weights, skip=("sigma", "coulomb_cutoff."))
    return model, Config(model=model_cfg)


def from_foundation(name: str = "qet-matpes-pbe-2025.2", dtype=None,
                    cache_dir: Optional[Union[str, Path]] = None) -> QET:
    """Download (once) and convert a registered MatGL QET potential.

    Parameters
    ----------
    name : str, optional
        A key of :data:`FOUNDATION_MODELS`, by default the MatPES PBE model.
    dtype : torch.dtype or str, optional
        Final dtype; ``None`` keeps the checkpoint's float32.
    cache_dir : str or pathlib.Path, optional
        Model cache directory (the hub's default when ``None``).

    Returns
    -------
    QET
        The converted potential, in evaluation mode.
    """
    folder = fetch_foundation(name, FOUNDATION_MODELS, cache_dir)
    model, _ = convert_directory(folder, getattr(torch, dtype) if isinstance(dtype, str) else dtype)
    return model.eval()


__all__ = ["FOUNDATION_MODELS", "convert_directory", "from_foundation"]
