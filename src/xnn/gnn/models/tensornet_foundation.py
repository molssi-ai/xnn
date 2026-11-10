"""The TensorNet foundation potentials of MatGL, converted weight for weight.

MatGL (BSD-3-Clause) publishes TensorNet potentials trained on MatPES as a
directory of ``model.json`` (the architecture) and ``state.pt`` (the weights).
Their architecture is xnn's :class:`~xnn.gnn.models.tensornet.TensorNet` with
three options set: the smooth spherical Bessel radial basis
(``radial_basis="smooth_bessel"``), a gated energy network
(``readout="gated"``) and no self pairs (``self_pairs=False``). The energy is
``data_std`` times the network's sum plus the per-element references, which
map onto ``energy_scale`` and ``atom_ref``.

``TensorNet.from_foundation("tensornet-matpes-pbe-2025.2")`` downloads a
pinned revision into the model cache and converts it; a local directory
loads through the model hub's ``"matgl"`` format
(``xnn.common.models.from_pretrained(path)``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import torch

from xnn.common.config import Config, ModelConfig
from xnn.common.data.elements import SYMBOL_TO_Z
from xnn.common.models.hub import ModelFormat, register_format

from .tensornet import TensorNet

_HF = "https://huggingface.co/{repo}/resolve/{revision}/{file}"

#: Registered MatGL TensorNet checkpoints: name -> (repository, pinned revision).
FOUNDATION_MODELS = {
    "tensornet-matpes-pbe-2025.2": ("materialyze/TensorNet-PES-MatPES-PBE-2025.2",
                                    "7ac2b90400a87eca7f44c1e7956dca6b1c85e418"),
    "tensornet-matpes-r2scan-2025.2": ("materialyze/TensorNet-PES-MatPES-r2SCAN-2025.2",
                                       "a9934dd4311849373230c253c74c6eab2519f867"),
}

# the gated energy network of MatGL is SiLU whatever the model activation: only SiLU models convert exactly
_ACTIVATIONS = {"swish": "silu", "silu": "silu"}


def _is_matgl_dir(path: Path) -> bool:
    """Whether ``path`` is a MatGL TensorNet directory (``model.json`` and ``state.pt``)."""
    path = Path(path)
    if path.is_file() and path.name in ("model.json", "state.pt"):
        path = path.parent
    if not ((path / "model.json").is_file() and (path / "state.pt").is_file()):
        return False
    try:
        meta = json.loads((path / "model.json").read_text())
        return meta["kwargs"]["model"]["@class"] == "TensorNet"
    except (ValueError, KeyError, TypeError):
        return False


def convert_directory(path: Union[str, Path], dtype=None) -> tuple:
    """Convert a MatGL TensorNet directory into xnn's TensorNet.

    Parameters
    ----------
    path : str or pathlib.Path
        The directory of ``model.json`` and ``state.pt`` (or either file in it).
    dtype : torch.dtype, optional
        The dtype the model is built in, by default the checkpoint's. Building
        in float64 keeps MatGL's float64 element references exact.

    Returns
    -------
    tuple
        The :class:`~xnn.gnn.models.tensornet.TensorNet` and the
        :class:`~xnn.common.config.Config` that rebuilds it.

    Raises
    ------
    NotImplementedError
        For an architecture outside the converted ones (another radial basis,
        an intensive target, state features, a repulsion term, a constant
        energy offset per structure).
    """
    path = Path(path)
    if path.is_file():
        path = path.parent
    meta = json.loads((path / "model.json").read_text())
    kwargs = meta["kwargs"]
    model_meta = kwargs["model"]
    if model_meta.get("@class") != "TensorNet":
        raise NotImplementedError(f"{path} holds a {model_meta.get('@class')}, not a TensorNet")
    a = model_meta["init_args"]
    unsupported = []
    if str(a.get("rbf_type", "")).lower() != "sphericalbessel" or not a.get("use_smooth", False):
        unsupported.append(f"radial basis {a.get('rbf_type')} (smooth={a.get('use_smooth')})")
    if a.get("is_intensive", True):
        unsupported.append("an intensive target")
    if a.get("include_state", False):
        unsupported.append("state features")
    if int(a.get("ntargets", 1)) != 1:
        unsupported.append("several targets")
    if kwargs.get("calc_repuls", False):
        unsupported.append("a repulsion term")
    if str(a.get("activation_type", "swish")) not in _ACTIVATIONS:
        unsupported.append(f"activation {a.get('activation_type')}")
    if unsupported:
        raise NotImplementedError(f"{path}: not converted: {', '.join(unsupported)}")

    state = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if float(state.get("data_mean", torch.tensor(0.0))) != 0.0:
        raise NotImplementedError(f"{path}: a constant energy offset per structure is not converted")
    units, n_layers = int(a["units"]), int(a["nblocks"])
    n_rbf = int(a["max_n"])
    species = [SYMBOL_TO_Z[s] for s in a["element_types"]]
    refs = state.get("element_refs.property_offset")
    extra = {
        "radial_basis": "smooth_bessel", "readout": "gated", "self_pairs": False,
        "activation": _ACTIVATIONS[str(a.get("activation_type", "swish"))],
        "equivariance_group": str(a.get("equivariance_invariance_group", "O(3)")),
        "n_elements": max(128, max(species) + 1),
        "energy_scale": float(state.get("data_std", torch.tensor(1.0))),
        "species": species,
    }
    if refs is not None:
        extra["atomic_energies"] = [float(v) for v in refs.reshape(-1)[:len(species)]]
    model_cfg = ModelConfig(name="tensornet", cutoff=float(a["cutoff"]), n_features=units,
                            n_interactions=n_layers, n_rbf=n_rbf, extra=extra)

    dtype = state["model.linear.weight"].dtype if dtype is None else dtype
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = TensorNet.from_config(model_cfg)
    finally:
        torch.set_default_dtype(previous)
    _transplant(model, state, species, n_layers)
    return model, Config(model=model_cfg)


def _transplant(model: TensorNet, state: dict, species: list, n_layers: int) -> None:
    """Copy MatGL's weights into ``model``: the same layer names, the embedding rows by atomic
    number, the ``3F`` outputs of the last scalar layers reordered from per feature to per part."""
    own = model.state_dict()
    out = {}
    for name in own:
        if name.startswith(("tensor_embedding.", "layers.", "out_norm.", "linear.")):
            key = "model." + name
            if name == "tensor_embedding.emb.weight":
                emb = own[name].clone()
                emb[torch.tensor(species)] = state[key].to(emb.dtype)
                out[name] = emb
            elif key in state:
                out[name] = state[key]
    # the gated energy network: dense layers at 0, 2, 4, ... of MatGL's stacks
    for branch, theirs in (("mlp", "layers"), ("gate", "gates")):
        i = 0
        while f"readout.{branch}.layers.{i}.weight" in own:
            for p in ("weight", "bias"):
                out[f"readout.{branch}.layers.{i}.{p}"] = state[f"model.final_layer.gated.{theirs}.{2 * i}.{p}"]
            i += 1
    # the last scalar layers of the embedding and of the filter networks end in 3F outputs: MatGL
    # lays them out per feature (f, part), xnn per part (part, f)
    units = model.n_features
    for name in ["tensor_embedding.linears_scalar.1"] + [f"layers.{i}.linears_scalar.2" for i in range(n_layers)]:
        for p in ("weight", "bias"):
            t = out[f"{name}.{p}"]
            out[f"{name}.{p}"] = t.reshape(units, 3, *t.shape[1:]).transpose(0, 1).reshape(t.shape)
    missing = [n for n in own if n not in out and not n.startswith(("rbf.", "energy_", "atom_ref.", "cutoff_fn."))]
    if missing:
        raise KeyError(f"MatGL checkpoint lacks weights for {missing}")
    model.load_state_dict({**own, **{k: v.to(own[k].dtype) for k, v in out.items()}})


def from_foundation(name: str = "tensornet-matpes-pbe-2025.2", dtype=None,
                    cache_dir: Optional[Union[str, Path]] = None) -> TensorNet:
    """Download (once) and convert a registered MatGL TensorNet potential.

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
    TensorNet
        The converted potential, in evaluation mode.
    """
    from xnn.common.data.hub._download import download_file
    from xnn.common.models.hub.cache import resolve_cache_dir
    if name not in FOUNDATION_MODELS:
        raise KeyError(f"unknown MatGL TensorNet model {name!r}; known: {sorted(FOUNDATION_MODELS)}")
    repo, revision = FOUNDATION_MODELS[name]
    folder = resolve_cache_dir(cache_dir) / "matgl" / name / revision
    for file in ("model.json", "state.pt"):
        download_file(_HF.format(repo=repo, revision=revision, file=file), folder / file)
    model, _ = convert_directory(folder, getattr(torch, dtype) if isinstance(dtype, str) else dtype)
    return model.eval()


def _convert_file(path: Path, head: Optional[str]):
    """The model hub's ``matgl`` converter: directory -> (model, config)."""
    return convert_directory(path)


register_format(ModelFormat(name="matgl", suffixes=(), convert=_convert_file, detect=_is_matgl_dir))

__all__ = ["FOUNDATION_MODELS", "convert_directory", "from_foundation"]
