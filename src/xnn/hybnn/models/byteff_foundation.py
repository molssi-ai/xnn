"""The released ByteFF and ByteFF-Pol networks in xnn.

The authors publish five independently trained ByteFF networks (ByteFF-26, whose
parameters are averaged) and one ByteFF-Pol network (ByteFF-Pol-25). They are
distributed under the Creative Commons Attribution-NonCommercial 4.0 license
(CC BY-NC 4.0): xnn does not ship them, the model hub downloads them on first use
and states the terms when they load.

* :func:`from_foundation` builds a bound or unbound model from a released name
  (``"byteff-26"``, ``"byteff-26-member-1"``, ..., ``"byteff-pol-25"``);
* the hub format ``"byteff2"`` converts one released checkpoint (a state dict saved
  under ``"model_state_dict"``) into a :class:`~xnn.hybnn.models.byteff.ByteFF` or
  :class:`~xnn.hybnn.models.byteff.ByteFFPol`, so ``from_pretrained`` loads every
  single network;
* :func:`convert_state_dict` and :func:`architecture` map such a state dict onto one
  ensemble member of the xnn model.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Union

import torch

from xnn.common.models.hub import ModelFormat, register_format

from .byteff import ByteFF, ByteFFPol

#: Released ensembles: name -> the hub cards of their members.
FOUNDATION_MODELS = {
    "byteff-26": [f"byteff-26-member-{i}" for i in range(1, 6)],
    "byteff-pol-25": ["byteff-pol-25"],
}

_LAYER = {"lin_key": "key", "lin_query": "query", "lin_value": "value", "lin_proj_edge": "edge_proj",
          "lin_proj": "proj", "lin_gate": "gate", "lin_edge": "edge_bias", "norm1_edge": "edge_norm1",
          "norm2_edge": "edge_norm2", "norm1": "norm1", "norm2": "norm2", "ffn_edge": "edge_ffn", "ffn": "ffn"}
_EMBED = {"atom_embedding": "element", "connectivity_embedding": "degree", "fm_chg_embedding": "formal_charge",
          "ring_con_embedding": "ring_degree", "min_ring_size_embedding": "min_ring",
          "bond_ring_embedding": "bond_ring", "bond_order_embedding": "bond_order"}
_ATOM_HEADS = {"sigma_mlp": "sigma", "epsilon_mlp": "epsilon", "charge_pre_mlp": "charge.pre",
               "charge_out_mlp": "charge.out", "volume_mlp": "volume", "lamb_mlp": "lam", "eps_mlp": "eps",
               "ct_eps_mlp": "ct_eps", "ct_lamb_mlp": "ct_lam"}
_FREE = {"v_free": "v_free", "c6_free": "c6_free", "rvdw_free": "r_free", "alpha_free": "alpha_free"}


def _rename(key: str) -> Optional[str]:
    """The xnn name (relative to one ensemble member) of a released parameter."""
    k = key.replace(".lins.", ".layers.")
    m = re.fullmatch(r"graph_block\.feature_layer\.(\w+)\.(weight|layers\..+)", k)
    if m:
        name = m.group(1)
        if name in _EMBED:
            return f"graph.embedding.{_EMBED[name]}.{m.group(2)}"
        if name in ("node_mlp", "edge_mlp"):
            return f"graph.embedding.{name}.{m.group(2)}"
        return None
    m = re.fullmatch(r"graph_block\.gnn_layer\.(edge_in_lin|edge_out_lin)\.(.+)", k)
    if m:
        return f"graph.{m.group(1)[:-4]}.{m.group(2)}"
    m = re.fullmatch(r"graph_block\.gnn_layer\.gnn\.lin\.(.+)", k)
    if m:
        return f"graph.out.{m.group(1)}"
    m = re.fullmatch(r"graph_block\.gnn_layer\.gnn\.convs\.(\d+)\.(\w+)\.(.+)", k)
    if m and m.group(2) in _LAYER:
        return f"graph.layers.{m.group(1)}.{_LAYER[m.group(2)]}.{m.group(3)}"
    m = re.fullmatch(r"preff_block\.pre_layers\.PreMMBonded(?:Conj)?\.(pre|post|out)_mlp\.(\w+)\.(.+)", k)
    if m:
        return f"bonded.{dict(pre='pool', post='post', out='out')[m.group(1)]}.{m.group(2)}.{m.group(3)}"
    m = re.fullmatch(r"preff_block\.pre_layers\.Pre(?:LJEs|ChargeVolume|Exp6Pol)\.(\w+)\.(.+)", k)
    if m:
        if m.group(1) in _ATOM_HEADS:
            return f"nonbonded.{_ATOM_HEADS[m.group(1)]}.{m.group(2)}"
        if m.group(1) in _FREE and m.group(2) == "weight":
            return f"nonbonded.{_FREE[m.group(1)]}"
    return None


def convert_state_dict(state: dict) -> dict:
    """Map a released state dict onto one ensemble member of the xnn model.

    Parameters
    ----------
    state : dict of str to Tensor
        The released network's ``model_state_dict``.

    Returns
    -------
    dict of str to Tensor
        Weights keyed relative to ``members.<m>`` of the xnn model.

    Raises
    ------
    KeyError
        For a parameter with no counterpart.
    """
    out = {}
    for key, value in state.items():
        new = _rename(key)
        if new is None:
            raise KeyError(f"no xnn counterpart for the parameter {key!r}")
        out[new] = value.reshape(-1) if new.endswith("_free") else value
    return out


def _mlp_shape(state: dict, prefix: str) -> tuple:
    """``(hidden, out, layers)`` of a released MLP."""
    head = prefix + ".lins."
    rows = sorted(int(k[len(head):-len(".weight")]) for k in state if k.startswith(head) and k.endswith(".weight"))
    return int(state[f"{prefix}.lins.0.weight"].shape[0]), int(state[f"{prefix}.lins.{rows[-1]}.weight"].shape[0]), \
        len(rows)


def architecture(state: dict) -> tuple:
    """The xnn class and options of a released state dict, read from its shapes.

    Parameters
    ----------
    state : dict of str to Tensor
        The released network's ``model_state_dict``.

    Returns
    -------
    tuple
        The class (:class:`ByteFF` or :class:`ByteFFPol`) and its keyword
        options (``graph``, ``bonded``, ``nonbonded``).
    """
    fl, gl, pre = "graph_block.feature_layer", "graph_block.gnn_layer.gnn", "preff_block.pre_layers"
    polarizable = any(k.startswith(f"{pre}.PreExp6Pol.") for k in state)
    conj = any(k.startswith(f"{pre}.PreMMBondedConj.") for k in state)
    layers = len({k.split(".")[4] for k in state if k.startswith(f"{gl}.convs.")})
    heads = int(state[f"{gl}.convs.0.lin_gate.weight"].shape[0])
    key = state[f"{gl}.convs.0.lin_key.weight"]
    ffn = _mlp_shape(state, f"{gl}.convs.0.ffn")
    hidden = int(key.shape[1])
    jk, out_dim = None, hidden
    if f"{gl}.lin.weight" in state:
        lin = state[f"{gl}.lin.weight"]
        jk, out_dim = ("cat" if lin.shape[1] == hidden * layers else "last"), int(lin.shape[0])
    nm, em = _mlp_shape(state, f"{fl}.node_mlp"), _mlp_shape(state, f"{fl}.edge_mlp")
    graph = dict(atom_dim=int(state[f"{fl}.atom_embedding.weight"].shape[1]),
                 feature_dim=int(state[f"{fl}.connectivity_embedding.weight"].shape[1]),
                 node_mlp=nm, edge_mlp=em, hidden=hidden, layers=layers, heads=heads,
                 head_dim=int(key.shape[0]) // heads, ffn=(ffn[0], ffn[2]), out_dim=out_dim, jumping_knowledge=jk)
    bp = f"{pre}.PreMMBondedConj" if conj else f"{pre}.PreMMBonded"
    first = "bond_k1" if conj else "bond_k"
    o = _mlp_shape(state, f"{bp}.out_mlp.{first}")
    bonded = dict(pre=_mlp_shape(state, f"{bp}.pre_mlp.bond"), out=(o[0], o[2]),
                  form="two_well" if conj else "harmonic")
    head = f"{pre}.PreChargeVolume" if polarizable else f"{pre}.PreLJEs"
    o = _mlp_shape(state, f"{head}.{'volume_mlp' if polarizable else 'sigma_mlp'}")
    nonbonded = dict(pre=_mlp_shape(state, f"{head}.charge_pre_mlp"), out=(o[0], o[2]))
    return (ByteFFPol if polarizable else ByteFF), dict(graph=graph, bonded=bonded, nonbonded=nonbonded)


def _read(path: Union[str, Path]) -> dict:
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    return state["model_state_dict"] if "model_state_dict" in state else state


def from_byteff2(paths, *, structure=None, dtype=None, charge: int = 0, **options):
    """Build an xnn model from released checkpoint files, one per ensemble member.

    Parameters
    ----------
    paths : str, Path or list of them
        Checkpoint files (``optimal.pt``) of networks of the same architecture.
    structure : optional
        A system to bind (see :meth:`~xnn.hybnn.models.byteff.ByteFF.bind`).
    dtype : torch.dtype, optional
        Floating-point dtype of the model.
    charge : int, optional
        Total charge for bond perception when binding an ``ase.Atoms``.
    **options
        Further keyword options of the model class (``supported_elements``, ...).

    Returns
    -------
    ByteFF or ByteFFPol
        The model in evaluation mode.
    """
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    states = [_read(p) for p in paths]
    cls, arch = architecture(states[0])
    model = cls(members=len(states), **arch, **options)
    full = {f"members.{m}.{k}": v for m, s in enumerate(states) for k, v in convert_state_dict(s).items()}
    model.load_state_dict(full, strict=True)
    if dtype is not None:
        model = model.to(dtype)
    if structure is not None:
        model.bind(structure, charge=charge)
    return model.eval()


def from_foundation(name: str = "byteff-26", *, structure=None, cache_dir=None, dtype=None, charge: int = 0,
                    local_files_only: bool = False):
    """Load a released ByteFF or ByteFF-Pol model through the model hub.

    The networks are downloaded on first use into the hub cache. They are
    distributed under CC BY-NC 4.0: by loading them you accept its terms (no
    commercial use).

    Parameters
    ----------
    name : str
        ``"byteff-26"`` (the five-network ensemble), ``"byteff-26-member-<i>"``
        (one of its networks) or ``"byteff-pol-25"``.
    structure : optional
        A system to bind.
    cache_dir : str or Path, optional
        Model hub cache directory.
    dtype : torch.dtype, optional
        Floating-point dtype of the model.
    charge : int, optional
        Total charge for bond perception when binding an ``ase.Atoms``.
    local_files_only : bool, optional
        Do not download; fail when a network is not cached.

    Returns
    -------
    ByteFF or ByteFFPol
        The model in evaluation mode.
    """
    from xnn.common.data.hub._download import download_file
    from xnn.common.models.hub.cache import resolve_cache_dir
    from xnn.common.models.hub.loader import _license_notice
    from xnn.common.models.hub.registry import get_card

    cards = [get_card(c) for c in FOUNDATION_MODELS.get(name, [name])]
    if any(c is None for c in cards):
        raise ValueError(f"unknown released model {name!r}; choose one of "
                         f"{sorted(FOUNDATION_MODELS) + FOUNDATION_MODELS['byteff-26']}")
    cache = resolve_cache_dir(cache_dir)
    paths = []
    for card in cards:
        _license_notice(card)
        dest = cache / card.name / "raw" / "optimal.pt"
        if not dest.is_file():
            if local_files_only:
                raise FileNotFoundError(f"{card.name} is not cached in {cache}")
            download_file(card.url, dest, md5=card.md5)
        paths.append(dest)
    species = cards[0].species
    options = {"supported_elements": tuple(species)} if species else {}
    return from_byteff2(paths, structure=structure, dtype=dtype, charge=charge, **options)


def _convert_file(path: Path, head: Optional[str]) -> tuple:
    """The model hub's ``byteff2`` converter: one released checkpoint -> (model, config)."""
    from xnn.common.config import ModelConfig
    model = from_byteff2(path)
    cls, arch = architecture(_read(path))
    name = "byteff_pol" if cls is ByteFFPol else "byteff"
    return model, ModelConfig(name=name, cutoff=model.cutoff, extra=arch)


def _detect(path: Path) -> bool:
    try:
        return path.is_file() and any(k.startswith("graph_block.") for k in _read(path))
    except Exception:
        return False


register_format(ModelFormat(name="byteff2", suffixes=(), convert=_convert_file, detect=_detect,
                            requires="rdkit (to bind molecules)"))


__all__ = ["FOUNDATION_MODELS", "from_foundation", "from_byteff2", "convert_state_dict", "architecture"]
