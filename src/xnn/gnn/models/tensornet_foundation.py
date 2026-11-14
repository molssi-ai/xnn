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


def read_matgl_directory(path: Union[str, Path], model_class: str) -> tuple:
    """The parts of a MatGL potential directory whose model is ``model_class``.

    Parameters
    ----------
    path : str or pathlib.Path
        The directory of ``model.json`` and ``state.pt`` (or either file in it).
    model_class : str
        The MatGL class of the wrapped model (``"TensorNet"``, ``"QET"``).

    Returns
    -------
    tuple
        The directory, the potential's arguments, the model's ``init_args``
        and the state dict.

    Raises
    ------
    NotImplementedError
        When the directory holds another model.
    """
    path = Path(path)
    if path.is_file():
        path = path.parent
    meta = json.loads((path / "model.json").read_text())
    kwargs = meta["kwargs"]
    model_meta = kwargs["model"]
    if model_meta.get("@class") != model_class:
        raise NotImplementedError(f"{path} holds a {model_meta.get('@class')}, not a {model_class}")
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    return path, kwargs, model_meta["init_args"], state


def trunk_options(kwargs: dict, a: dict, unsupported: list) -> tuple:
    """The radial basis of a MatGL TensorNet trunk as xnn options, and the number of its functions.

    Pieces xnn does not convert (another basis, the ``SO(3)`` group, state
    features, several targets, a repulsion term, an activation other than
    SiLU) are appended to ``unsupported``.
    """
    rbf = str(a.get("rbf_type", "Gaussian")).lower()
    extra, n_rbf = {}, 0
    if rbf == "sphericalbessel" and a.get("use_smooth", False):
        extra, n_rbf = {"radial_basis": "smooth_bessel"}, int(a["max_n"])
    elif rbf == "gaussian":
        # n Gaussians from 0 to the cutoff plus one; a width of None is the inverse center spacing
        n_rbf, span = int(a.get("num_rbf", 32)), float(a["cutoff"]) + 1.0
        width = a.get("width", 0.5)
        extra = {"radial_basis": "gaussian", "rbf_max": span,
                 "gamma": float(width) if width is not None else (n_rbf - 1) / span}
    else:
        unsupported.append(f"radial basis {a.get('rbf_type')} (smooth={a.get('use_smooth')})")
    if str(a.get("equivariance_invariance_group", "O(3)")) != "O(3)":
        # MatGL's edge vectors point from the center to the neighbor, xnn's the other way: the same weights
        # give the mirror-image model, which only an O(3) model does not tell apart
        unsupported.append(f"the {a.get('equivariance_invariance_group')} group")
    if a.get("include_state", False):
        unsupported.append("state features")
    if int(a.get("ntargets", 1)) != 1:
        unsupported.append("several targets")
    if kwargs.get("calc_repuls", False):
        unsupported.append("a repulsion term")
    if str(a.get("activation_type", "swish")) not in _ACTIVATIONS:
        unsupported.append(f"activation {a.get('activation_type')}")
    return extra, n_rbf


def potential_options(path: Path, a: dict, state: dict) -> dict:
    """The options of a MatGL potential's trunk and energy standardization: activation, symmetry
    group, species, the energy scale and the element references."""
    if float(state.get("data_mean", torch.tensor(0.0))) != 0.0:
        raise NotImplementedError(f"{path}: a constant energy offset per structure is not converted")
    species = [SYMBOL_TO_Z[s] for s in a["element_types"]]
    extra = {
        "self_pairs": False,
        "activation": _ACTIVATIONS[str(a.get("activation_type", "swish"))],
        "equivariance_group": str(a.get("equivariance_invariance_group", "O(3)")),
        "n_elements": max(128, max(species) + 1),
        "energy_scale": float(state.get("data_std", torch.tensor(1.0))),
        "species": species,
    }
    refs = state.get("element_refs.property_offset")
    if refs is not None:
        extra["atomic_energies"] = [float(v) for v in refs.reshape(-1)[:len(species)]]
    return extra


def build_converted(cls, name: str, a: dict, extra: dict, n_rbf: int, state: dict, dtype) -> tuple:
    """Build ``cls`` from converted options in ``dtype`` (by default the checkpoint's); returns the
    model and its :class:`~xnn.common.config.ModelConfig`."""
    model_cfg = ModelConfig(name=name, cutoff=float(a["cutoff"]), n_features=int(a["units"]),
                            n_interactions=int(a["nblocks"]), n_rbf=n_rbf, extra=extra)
    dtype = state["model.linear.weight"].dtype if dtype is None else dtype
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = cls.from_config(model_cfg)
    finally:
        torch.set_default_dtype(previous)
    return model, model_cfg


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
        For an architecture outside the converted ones (a radial basis other
        than the Gaussians or the smooth Bessel functions, the ``SO(3)``
        group, an intensive target, state features, a repulsion term, a
        constant energy offset per structure).
    """
    path, kwargs, a, state = read_matgl_directory(path, "TensorNet")
    unsupported = []
    extra, n_rbf = trunk_options(kwargs, a, unsupported)
    if a.get("is_intensive", True):
        unsupported.append("an intensive target")
    if unsupported:
        raise NotImplementedError(f"{path}: not converted: {', '.join(unsupported)}")
    extra.update(potential_options(path, a, state), readout="gated")
    model, model_cfg = build_converted(TensorNet, "tensornet", a, extra, n_rbf, state, dtype)
    transplant(model, state, extra["species"], int(a["nblocks"]))
    return model, Config(model=model_cfg)


def transplant(model: TensorNet, state: dict, species: list, n_layers: int,
               extra: Optional[dict] = None, skip: tuple = ()) -> None:
    """Copy MatGL's weights into ``model``: the same layer names, the embedding rows by atomic
    number, the ``3F`` outputs of the last scalar layers reordered from per feature to per part.

    ``extra`` holds the converted weights of a subclass's own layers; parameters whose names start
    with a prefix in ``skip`` keep their values.
    """
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
    out.update(extra or {})
    kept = ("rbf.", "energy_", "atom_ref.", "cutoff_fn.") + tuple(skip)
    missing = [n for n in own if n not in out and not n.startswith(kept)]
    if missing:
        raise KeyError(f"MatGL checkpoint lacks weights for {missing}")
    model.load_state_dict({**own, **{k: v.to(own[k].dtype) for k, v in out.items()}})


def fetch_foundation(name: str, models: dict, cache_dir: Optional[Union[str, Path]] = None) -> Path:
    """Download (once) a registered MatGL potential ``name`` of ``models`` into the model cache and
    return its directory."""
    from xnn.common.data.hub._download import download_file
    from xnn.common.models.hub.cache import resolve_cache_dir
    if name not in models:
        raise KeyError(f"unknown MatGL model {name!r}; known: {sorted(models)}")
    repo, revision = models[name]
    folder = resolve_cache_dir(cache_dir) / "matgl" / name / revision
    for file in ("model.json", "state.pt"):
        download_file(_HF.format(repo=repo, revision=revision, file=file), folder / file)
    return folder


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
    folder = fetch_foundation(name, FOUNDATION_MODELS, cache_dir)
    model, _ = convert_directory(folder, getattr(torch, dtype) if isinstance(dtype, str) else dtype)
    return model.eval()


__all__ = ["FOUNDATION_MODELS", "convert_directory", "from_foundation", "read_matgl_directory",
           "trunk_options", "potential_options", "build_converted", "transplant", "fetch_foundation"]
