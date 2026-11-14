"""The pretrained MEGNet models of MatGL, converted weight for weight.

MatGL (BSD-3-Clause) publishes two MEGNet property models: the formation
energy per atom of the Materials Project 2018.6.1 crystals (eV/atom) and the
multi-fidelity band gap of the 2019.4.1 crystals (eV, fidelities PBE,
GLLB-SC, HSE and SCAN). Each is a directory of ``model.json`` (the
architecture), ``model.pt`` (the target standardization) and ``state.pt``
(the weights). The architecture is xnn's
:class:`~xnn.gnn.models.megnet.MEGNet` with the same layer layout; the
standardization becomes ``property_scale`` and ``property_shift``.

The formation energy model was trained with the bond direction reversed: its
bond updates average at the atom MatGL now lists second. ``from_foundation``
swaps the two atom blocks of every bond update's first layer to match, which
restores its published accuracy (current MatGL does not). The band gap model
is converted as MatGL runs it.

``MEGNet.from_foundation("megnet-mp-2019.4.1-bandgap-mfi")`` downloads a
pinned revision into the model cache and converts it; a local directory
loads through the model hub's ``"matgl"`` format
(``xnn.common.models.from_pretrained(path)``).
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional, Union

import torch

from xnn.common.config import Config, ModelConfig
from xnn.common.data.elements import SYMBOL_TO_Z

from .megnet import MEGNet

_URL = "https://raw.githubusercontent.com/materialyzeai/matgl/{revision}/pretrained_models/{folder}/{file}"
_REVISION = "a6ebaebd6a0021e14d49d001103508c4980d94da"

#: Registered MatGL MEGNet models: name -> (folder, pinned revision, MD5 of every file, state names, whether
#: the weights were trained with the bond direction reversed).
FOUNDATION_MODELS = {
    "megnet-mp-2018.6.1-eform": (
        "MEGNet-MP-2018.6.1-Eform", _REVISION,
        {"model.json": "b36717f9d18347463f0e0a7f75e324f8", "model.pt": "95706ba997cd939e1a193cca2e4cb45b",
         "state.pt": "b0fd06137b8e406b6f26e2405456fe9d"}, None, True),
    "megnet-mp-2019.4.1-bandgap-mfi": (
        "MEGNet-MP-2019.4.1-BandGap-mfi", _REVISION,
        {"model.json": "614a900ace0007e1d113e31c5a1a5c5d", "model.pt": "d07abbf3864d0c162dee40056b9fbc67",
         "state.pt": "12b76083e5ad4e4d7d82153afc12537b"}, ["PBE", "GLLB-SC", "HSE", "SCAN"], True),
}

# MatGL's default element list: atomic numbers 1-94 without Po, At, Rn, Fr and Ra
_DEFAULT_SPECIES = [z for z in range(1, 95) if not 84 <= z <= 88]

# MatGL activation names -> xnn's
_ACTIVATIONS = {"softplus2": "ssp", "swish": "silu", "silu": "silu", "tanh": "tanh", "sigmoid": "sigmoid"}


class _Normalizer:
    """Stands in for MatGL's ``Normalizer`` while its fields are read from ``model.pt``."""


class _Unpickler(pickle.Unpickler):
    """Reads the arguments of ``model.pt``: plain values, tensors and MatGL's ``Normalizer`` only."""

    _TORCH = {("torch._utils", "_rebuild_tensor_v2"), ("torch", "FloatStorage"),
              ("torch", "DoubleStorage"), ("collections", "OrderedDict")}

    def find_class(self, module, name):
        if (module, name) == ("matgl.data.transformer", "Normalizer"):
            return _Normalizer
        if module == "matgl.data.transformer":
            # another transformer: read as a named stand-in, refused by the converter
            return type(name, (), {})
        if (module, name) in self._TORCH:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"model.pt refers to {module}.{name}, which is not read")


def _restricted_load(file, **kwargs):
    """:func:`pickle.load` through :class:`_Unpickler`."""
    return _Unpickler(file, **kwargs).load()


class _PickleModule:
    """The ``pickle_module`` of :func:`torch.load`: every read goes through :class:`_Unpickler`."""

    Unpickler = _Unpickler
    load = staticmethod(_restricted_load)


def _standardization(path: Path) -> tuple:
    """The ``(scale, shift)`` of MatGL's target transformer, ``(1, 0)`` without one."""
    if not (path / "model.pt").is_file():
        return 1.0, 0.0
    args = torch.load(path / "model.pt", map_location="cpu", weights_only=False,
                      pickle_module=_PickleModule)
    transformer = args.get("target_transformer") if isinstance(args, dict) else None
    if transformer is None:
        return 1.0, 0.0
    if not isinstance(transformer, _Normalizer):
        raise NotImplementedError(f"{path}: target transformer {type(transformer).__name__} is not converted")
    return float(transformer.std), float(transformer.mean)


def convert_directory(path: Union[str, Path], dtype=None, state_names=None) -> tuple:
    """Convert a MatGL MEGNet directory into xnn's MEGNet.

    Parameters
    ----------
    path : str or pathlib.Path
        The directory of ``model.json``, ``model.pt`` and ``state.pt`` (or a
        file in it).
    dtype : torch.dtype, optional
        The dtype the model is built in, by default the checkpoint's.
    state_names : sequence of str, optional
        Names of the state labels (the fidelities of a multi-fidelity model),
        which MatGL's files do not record.

    Returns
    -------
    tuple
        The :class:`~xnn.gnn.models.megnet.MEGNet` and the
        :class:`~xnn.common.config.Config` that rebuilds it.

    Raises
    ------
    NotImplementedError
        For an architecture outside the converted ones (a custom bond
        expansion, graph transformations, a model without a state, a target
        transformer other than a normalizer).
    """
    path = Path(path)
    if path.is_file():
        path = path.parent
    meta = json.loads((path / "model.json").read_text())
    model_meta = meta["kwargs"]["model"] if "model" in meta.get("kwargs", {}) else meta
    if model_meta.get("@class") != "MEGNet":
        raise NotImplementedError(f"{path} holds a {model_meta.get('@class')}, not a MEGNet")
    a = model_meta["init_args"]
    unsupported = []
    if a.get("bond_expansion") is not None:
        unsupported.append("a custom bond expansion")
    if a.get("graph_transformations"):
        unsupported.append("graph transformations")
    if not a.get("include_state", True):
        unsupported.append("a model without a state")
    if str(a.get("activation_type", "softplus2")) not in _ACTIVATIONS:
        unsupported.append(f"activation {a.get('activation_type')}")
    if unsupported:
        raise NotImplementedError(f"{path}: not converted: {', '.join(unsupported)}")
    scale, shift = _standardization(path)

    state = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
    state = {k[len("model."):] if k.startswith("model.") else k: v for k, v in state.items()}
    elements = a.get("element_types")
    species = _DEFAULT_SPECIES if elements is None else [SYMBOL_TO_Z[s] for s in elements]
    n_blocks = int(a.get("nblocks", 3))
    extra = {
        "gamma": float(a.get("gauss_width", 0.5)),
        "state_dim": int(a.get("dim_state_embedding", 2)),
        "n_state_types": None if a.get("ntypes_state") is None else int(a["ntypes_state"]),
        "state_names": None if state_names is None else [str(s) for s in state_names],
        "encoder_sizes": [int(s) for s in a.get("hidden_layer_sizes_input", (64, 32))],
        "conv_sizes": [int(s) for s in a.get("hidden_layer_sizes_conv", (64, 64, 32))],
        "readout_sizes": [int(s) for s in a.get("hidden_layer_sizes_output", (32, 16))],
        "set2set_steps": int(a.get("niters_set2set", 2)),
        "set2set_layers": int(a.get("nlayers_set2set", 1)),
        "activation": _ACTIVATIONS[str(a.get("activation_type", "softplus2"))],
        "dropout": float(a.get("dropout") or 0.0),
        "classification": bool(a.get("is_classification", False)),
        "n_elements": max(95, max(species) + 1),
        "property_scale": scale,
        "property_shift": shift,
    }
    model_cfg = ModelConfig(name="megnet", cutoff=float(a.get("cutoff", 4.0)),
                            n_features=int(a.get("dim_node_embedding", 16)), n_interactions=n_blocks,
                            n_rbf=int(a.get("dim_edge_embedding", 100)), extra=extra)

    dtype = state["edge_encoder.layers.0.weight"].dtype if dtype is None else dtype
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = MEGNet.from_config(model_cfg)
    finally:
        torch.set_default_dtype(previous)
    _transplant(model, state, species, n_blocks)
    return model, Config(model=model_cfg)


def _transplant(model: MEGNet, state: dict, species: list, n_blocks: int) -> None:
    """Copy MatGL's weights into ``model``: the embedding rows by atomic number, the dense layers
    of MatGL's stacks (at 0, 2, 4, ... between the activations) one after another."""
    own = model.state_dict()
    out = {}

    def stack(mine: str, theirs: str) -> None:
        i = 0
        while f"{mine}.layers.{i}.weight" in own:
            for p in ("weight", "bias"):
                out[f"{mine}.layers.{i}.{p}"] = state[f"{theirs}.layers.{2 * i}.{p}"]
            i += 1

    emb = own["node_embedding.weight"].clone()
    emb[torch.tensor(species)] = state["embedding.layer_node_embedding.weight"].to(emb.dtype)
    out["node_embedding.weight"] = emb
    if model.state_embedding is not None:
        out["state_embedding.weight"] = state["embedding.layer_state_embedding.weight"]
    for kind in ("edge", "node", "state"):
        stack(f"{kind}_encoder", f"{kind}_encoder")
    for k in range(n_blocks):
        for kind in ("edge", "node", "state"):
            stack(f"blocks.{k}.{kind}_mlp", f"blocks.{k}.conv.{kind}_func")
            if k > 0:
                stack(f"blocks.{k}.{kind}_dense", f"blocks.{k}.{kind}_func")
    for mine, theirs in (("node_set2set", "node_s2s"), ("edge_set2set", "edge_s2s")):
        for name in own:
            if name.startswith(mine + ".lstm."):
                out[name] = state[theirs + name[len(mine):]]
    stack("readout", "output_proj")
    missing = [n for n in own if n not in out and not n.startswith(("rbf.", "property_"))]
    if missing:
        raise KeyError(f"MatGL checkpoint lacks weights for {missing}")
    model.load_state_dict({**own, **{k: v.to(own[k].dtype) for k, v in out.items()}})


def from_foundation(name: str = "megnet-mp-2018.6.1-eform", dtype=None,
                    cache_dir: Optional[Union[str, Path]] = None, quiet: bool = False) -> MEGNet:
    """Download (once) and convert a registered MatGL MEGNet model.

    Parameters
    ----------
    name : str, optional
        A key of :data:`FOUNDATION_MODELS`, by default the formation energy model.
    dtype : torch.dtype or str, optional
        Final dtype; ``None`` keeps the checkpoint's float32.
    cache_dir : str or pathlib.Path, optional
        Model cache directory (the hub's default when ``None``).
    quiet : bool, optional
        Hide the download progress bars. Default ``False``.

    Returns
    -------
    MEGNet
        The converted model, in evaluation mode; the multi-fidelity model
        knows its fidelities by name (:meth:`MEGNet.state_label`).
    """
    from xnn.common.data.hub._download import download_file
    from xnn.common.models.hub.cache import resolve_cache_dir
    if name not in FOUNDATION_MODELS:
        raise KeyError(f"unknown MatGL MEGNet model {name!r}; known: {sorted(FOUNDATION_MODELS)}")
    folder_name, revision, md5s, state_names, reversed_bonds = FOUNDATION_MODELS[name]
    folder = resolve_cache_dir(cache_dir) / "matgl" / name / revision
    for file, md5 in md5s.items():
        download_file(_URL.format(revision=revision, folder=folder_name, file=file), folder / file, md5=md5,
                      quiet=quiet)
    model, _ = convert_directory(folder, getattr(torch, dtype) if isinstance(dtype, str) else dtype,
                                 state_names)
    if reversed_bonds:
        swap_bond_atoms(model)
    return model.eval()


def swap_bond_atoms(model: MEGNet) -> MEGNet:
    """Swap the receiving and sending atom blocks of the first layer of every bond update, in place.

    For weights trained with the other bond direction (each bond's update averaged at the atom it comes
    from); the model then computes what it was trained to.

    Returns
    -------
    MEGNet
        ``model``.
    """
    with torch.no_grad():
        for block in model.blocks:
            w = block.edge_mlp.layers[0].weight
            n = w.shape[1] // 4
            w[:, :2 * n] = torch.cat([w[:, n:2 * n], w[:, :n]], dim=1)
    return model


__all__ = ["FOUNDATION_MODELS", "convert_directory", "from_foundation", "swap_bond_atoms"]
