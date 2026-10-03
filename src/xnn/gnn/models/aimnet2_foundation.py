"""Load the published AIMNet2 models into the xnn :class:`~xnn.gnn.models.aimnet2.AIMNet2`.

The AIMNet2 models of Anstine, Zubatyuk and Isayev (*Chem. Sci.* **16**,
10228, 2025; MIT license) are published by `isayevlab/aimnetcentral
<https://github.com/isayevlab/aimnetcentral>`_ as six families of four
ensemble members each:

* ``aimnet2-wb97m-d3``: the general organic model (wB97M-D3/def2-TZVPP; H,
  B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I);
* ``aimnet2-b973c-d3``: the same chemistry at the B97-3c level;
* ``aimnet2-b973c-2025-d3``: the 2025 B97-3c retraining with improved
  intermolecular interactions;
* ``aimnet2-nse``: the open-shell variant (two charge channels, spin
  multiplicity input);
* ``aimnet2-pd``: palladium organometallics (B97-3c, no Coulomb term);
* ``aimnet2-rxn``: reactive CHNO chemistry (wB97M, neutral systems).

Each member is a self-describing artifact: a ``torch.save`` file loadable
with ``weights_only=True`` holding the state dict, the architecture as YAML
and the metadata (cutoff, elements, Coulomb mode, D3 parameters). The same
content is distributed as ``config.json`` + ``ensemble_<k>.safetensors``
directories; both are read here. Since the xnn AIMNet2 follows the reference
architecture block by block, the weights transplant directly:

* the per-element embedding rows, the Gaussian centers, the shell-combination
  weights, the MLPs of every pass and the energy readout copy over;
* the per-element energy shifts go to ``atom_ref`` (float64, as stored);
* the short-range Coulomb cutoff and the Gaussian exponent are read off the
  artifact into the model config, so a model rebuilt from its config
  reproduces the conversion exactly;
* the D3(BJ) dispersion the models are served with (the reference
  calculator adds it post hoc, the labels were fitted without it) becomes
  the config's ``subtracted_dispersion`` record, which the model hub adds
  back through the shared :class:`~xnn.common.models.d3.D3Dispersion`
  (two-body, 15 Angstrom cutoff with a 3 Angstrom switching window, as the
  reference does). ``dispersion=False`` at load time serves the bare network.

The models are entries of the model hub registry in the ``"aimnet2"``
format, registered by this module: :func:`xnn.common.models.from_pretrained`
downloads an artifact, converts it once and caches the converted model as a
portable xnn directory. The conversion needs nothing beyond torch and PyYAML.

Not covered: the legacy TorchScript (``.jpt``) checkpoints of the first
AIMNet2 release (the reference package converts them to the artifact format
read here), and architectures the reference code can express but never
published (shared-shell features, a second Gaussian basis for the vector
part); the converter raises ``NotImplementedError`` naming the piece.
"""
from __future__ import annotations

import json
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional, Union

import torch
from torch import nn

from xnn.common.config import Config, ModelConfig
from xnn.common.models.hub import ModelFormat, register_format
from xnn.common.models.hub.registry import get_card, registered_cards

from .aimnet2 import AIMNet2

#: The published checkpoints: name -> (download URL, license), a view of the
#: "aimnet2" entries of the model hub registry (models.json), which is the
#: single list. Every family is MIT-licensed.
FOUNDATION_MODELS = {
    card.name: (card.url, card.license) for card in registered_cards("aimnet2")
}
#: Short names of the registry: alias -> registered name (member 0 of each
#: family, as the reference package resolves them).
ALIASES = {alias: card.name for card in registered_cards("aimnet2") for alias in card.aliases}

# Settings of the D3(BJ) term the reference calculator adds post hoc: the
# two-body sum to 15 Angstrom, switched off over the last 20% of it, with
# the coordination numbers counted over the same list (no three-body term;
# its cutoff is set to the same radius so the neighbor list stops there).
_D3_SETTINGS = {"damping": "bj", "s9": 0.0, "cutoff_pair": 15.0, "switch_width_pair": 3.0,
                "cutoff_cn": 15.0, "cutoff_triple": 15.0}
# Families whose artifact carries no D3 parameters but whose reference
# calculator applies them anyway (the registry's per-family policy): the
# rxn models are fitted on a dispersion-free wB97M scale.
_POSTHOC_D3 = {"rxn": {"s6": 1.0, "s8": 0.3908, "a1": 0.566, "a2": 3.128}}

_SAFETENSORS_DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
    "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
    "U8": torch.uint8, "BOOL": torch.bool,
}


def read_safetensors(path: Union[str, Path]) -> dict[str, torch.Tensor]:
    """Read a ``.safetensors`` file into a dict of CPU tensors.

    The format is a little-endian 8-byte header length, a JSON header
    mapping tensor names to dtype, shape and byte offsets, and the raw
    buffers; no package beyond torch is needed to read it.

    Parameters
    ----------
    path : str or Path
        The file.

    Returns
    -------
    dict of str to torch.Tensor
        The tensors, in their stored dtypes.
    """
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
        data = f.read()
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, stop = info["data_offsets"]
        dtype = _SAFETENSORS_DTYPES[info["dtype"]]
        chunk = bytearray(data[start:stop])
        if not chunk:
            out[name] = torch.empty(info["shape"], dtype=dtype)
            continue
        out[name] = torch.frombuffer(chunk, dtype=dtype).reshape(info["shape"]).clone()
    return out


def _is_hf_directory(path: Path) -> bool:
    """Whether ``path`` is a ``config.json`` + ``ensemble_<k>.safetensors`` directory."""
    return (path.is_dir() and (path / "config.json").is_file()
            and any(path.glob("ensemble_*.safetensors")))


def _is_artifact_mapping(obj: Any) -> bool:
    """Whether ``obj`` is an AIMNet2 artifact (state dict plus architecture YAML)."""
    return isinstance(obj, Mapping) and "state_dict" in obj and "model_yaml" in obj


def detect_artifact(path: Union[str, Path]) -> bool:
    """Whether ``path`` is an AIMNet2 artifact file or directory.

    A ``.safetensors`` file, a directory of ``config.json`` and
    ``ensemble_<k>.safetensors``, or a ``torch.save`` file whose pickle
    names the ``state_dict`` and ``model_yaml`` keys (read from the zip
    without loading the tensors). Paths that do not exist are not artifacts.

    Parameters
    ----------
    path : str or Path
        The candidate.

    Returns
    -------
    bool
        ``True`` for an AIMNet2 artifact.
    """
    path = Path(path)
    if _is_hf_directory(path):
        return True
    if not path.is_file():
        return False
    if path.suffix.lower() == ".safetensors":
        return True
    # a torch.save zip whose pickle names the artifact keys; the pickle is
    # a few kB, so this costs nothing next to loading the file
    if not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            pickles = [n for n in archive.namelist() if n.rsplit("/", 1)[-1] == "data.pkl"]
            if not pickles:
                return False
            head = archive.read(pickles[0])
    except (OSError, zipfile.BadZipFile):
        return False
    return b"model_yaml" in head and b"state_dict" in head


def load_artifact(source: Union[str, Path], member: int = 0) -> dict[str, Any]:
    """Read an AIMNet2 artifact into one mapping.

    Parameters
    ----------
    source : str or Path
        A reference ``.pt`` artifact, a ``config.json`` +
        ``ensemble_<k>.safetensors`` directory, or one ``.safetensors`` file
        (its ``config.json`` is read from the same directory).
    member : int, optional
        Ensemble member to take from a directory, by default 0.

    Returns
    -------
    dict
        ``state_dict`` (CPU tensors), ``model_yaml`` and the metadata keys
        of the reference format (``cutoff``, ``implemented_species``,
        ``coulomb_mode``, ``d3_params``, ...).

    Raises
    ------
    FileNotFoundError
        If the file, directory or ensemble member does not exist.
    ValueError
        If the file is not an AIMNet2 artifact.
    """
    path = Path(source).expanduser()
    if path.is_dir():
        if not _is_hf_directory(path):
            raise FileNotFoundError(f"{path} holds no config.json with ensemble_<k>.safetensors")
        weights = path / f"ensemble_{int(member)}.safetensors"
        if not weights.is_file():
            raise FileNotFoundError(f"{path} has no ensemble member {member} ({weights.name})")
        config = json.loads((path / "config.json").read_text())
    elif path.suffix.lower() == ".safetensors":
        weights = path
        config_path = path.with_name("config.json")
        if not config_path.is_file():
            raise FileNotFoundError(f"{path} needs its config.json next to it")
        config = json.loads(config_path.read_text())
    elif path.is_file():
        obj = torch.load(path, map_location="cpu", weights_only=True)
        if not _is_artifact_mapping(obj):
            raise ValueError(f"{path} is not an AIMNet2 artifact (no state_dict / model_yaml)")
        return dict(obj)
    else:
        raise FileNotFoundError(f"no AIMNet2 artifact at {path}")
    if "model_yaml" not in config:
        raise ValueError(f"{path}: config.json carries no model_yaml (a family-level "
                         f"config); load the reference .pt artifact instead")
    artifact = dict(config)
    artifact["state_dict"] = read_safetensors(weights)
    return artifact


def _require(condition: bool, what: str) -> None:
    if not condition:
        raise NotImplementedError(f"this AIMNet2 artifact uses {what}, which the xnn "
                                  f"AIMNet2 does not implement")


def _prefixed(sd: Mapping[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """The entries of ``sd`` under ``prefix``, with the prefix stripped."""
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def from_aimnet_artifact(artifact: Mapping[str, Any], dtype=None) -> AIMNet2:
    """Convert a reference AIMNet2 artifact to an xnn :class:`AIMNet2`.

    Parameters
    ----------
    artifact : Mapping
        As returned by :func:`load_artifact`.
    dtype : torch.dtype or str, optional
        Final dtype; ``None`` keeps the artifact's float32.

    Returns
    -------
    AIMNet2
        The bare network (no dispersion term); its D3 record is available
        through :func:`convert_artifact`.
    """
    return convert_artifact(artifact, dtype=dtype)[0]


def convert_artifact(artifact: Mapping[str, Any], dtype=None) -> tuple[AIMNet2, Config]:
    """Convert a reference artifact; return the model and the config that rebuilds it.

    The body of :func:`from_aimnet_artifact`. The model is built from the
    returned :class:`~xnn.common.config.Config` through
    :meth:`AIMNet2.from_config`, so the config plus the transplanted weights
    rebuild exactly this model, which is how the model hub caches it. The
    config's ``subtracted_dispersion`` records the D3(BJ) term the published
    model is served with, when it has one.

    Parameters
    ----------
    artifact : Mapping
        As returned by :func:`load_artifact`.
    dtype : torch.dtype or str, optional
        Final dtype of the model; ``None`` keeps the artifact's float32.

    Returns
    -------
    tuple of (AIMNet2, Config)
        The converted model and its config.

    Raises
    ------
    NotImplementedError
        For an architecture outside the xnn AIMNet2 (see the module
        docstring) or an unexpected tensor in the state dict.
    """
    import yaml

    spec = yaml.safe_load(artifact["model_yaml"])
    _require(str(spec.get("class", "")).rsplit(".", 1)[-1] == "AIMNet2",
             f"the model class {spec.get('class')!r}")
    kw = dict(spec.get("kwargs", {}))
    aev = dict(kw["aev"])
    sd = {k: torch.as_tensor(v) for k, v in artifact["state_dict"].items()}

    # the architecture, off the YAML and the stored tensors
    _require(bool(kw.get("d2features", False)), "shared-shell features (d2features=false)")
    n_rbf = int(aev["nshifts_s"])
    _require(aev.get("rc_v") is None and aev.get("nshifts_v") in (None, n_rbf),
             "a second Gaussian basis for the vector features")
    _require(aev.get("shifts_s") is None or len(aev["shifts_s"]) == n_rbf,
             "a custom set of Gaussian centers")
    cutoff = float(aev["rc_s"])
    hidden = [[int(w) for w in widths] for widths in kw["hidden"]]
    charge_channels = int(kw.get("num_charge_channels", 1))
    outputs = dict(kw.get("outputs", {}))
    known_outputs = {"energy_mlp", "atomic_shift", "atomic_sum", "srcoulomb",
                     "dipole", "quadrupole"}
    _require(set(outputs) <= known_outputs,
             f"the output heads {sorted(set(outputs) - known_outputs)}")
    for head in ("energy_mlp", "atomic_shift", "atomic_sum"):
        _require(head in outputs, f"an energy expression without {head}")
    mlp = dict(outputs["energy_mlp"]["kwargs"]["mlp"])
    _require(str(mlp.get("activation_fn", "torch.nn.GELU")).endswith("GELU")
             and mlp.get("last_linear", True), "a readout other than Linear/GELU/.../Linear")
    coulomb_mode = str(artifact.get("coulomb_mode", "sr_embedded" if "srcoulomb" in outputs
                                    else "none"))
    _require(coulomb_mode in ("sr_embedded", "none"),
             f"the Coulomb mode {coulomb_mode!r} (an embedded long-range term)")
    extra: dict[str, Any] = dict(
        species=[int(z) for z in artifact["implemented_species"]],
        hidden=hidden, aim_size=int(kw["aim_size"]),
        readout_hidden=[int(w) for w in mlp.get("hidden", [])],
        n_vector_combinations=int(kw["ncomb_v"]),
        charge_channels=charge_channels, rbf_start=float(aev.get("rmin", 0.8)),
        gaussian_width=float(sd["aev.eta_s"]),
    )
    if coulomb_mode == "sr_embedded":
        sr = dict(outputs["srcoulomb"]["kwargs"])
        extra.update(coulomb="simple", coulomb_sr_cutoff=float(sd["outputs.srcoulomb.rc"]),
                     coulomb_sr_envelope=str(sr.get("envelope", "exp")))
    else:
        extra["coulomb"] = None
    model_cfg = ModelConfig(name="aimnet2", cutoff=cutoff, n_features=int(kw["nfeature"]),
                            n_interactions=len(hidden), n_rbf=n_rbf, extra=extra)

    # the D3(BJ) term the reference calculator adds post hoc
    d3 = artifact.get("d3_params") or _POSTHOC_D3.get(str(artifact.get("family")))
    record = None
    if d3 is not None:
        record = {"name": "d3", "s6": float(d3.get("s6", 1.0)), "s8": float(d3["s8"]),
                  "a1": float(d3["a1"]), "a2": float(d3["a2"]), **_D3_SETTINGS,
                  "note": "the published AIMNet2 model is served with this D3(BJ) term "
                          "(its training labels carry no dispersion correction)"}
    cfg = Config(model=model_cfg, subtracted_dispersion=record)

    # build the twin in the artifact's dtype and transplant
    up_dtype = sd["afv.weight"].dtype
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(up_dtype)
    try:
        model = AIMNet2.from_config(model_cfg)
    finally:
        torch.set_default_dtype(prev_dtype)
    if abs(float(sd["aev.rc_s"]) - cutoff) > 1e-6:
        raise ValueError(f"the stored cutoff {float(sd['aev.rc_s'])} differs from the "
                         f"architecture's {cutoff}")
    species = torch.tensor(extra["species"])
    expected = {"aev.rc_s", "aev.eta_s", "aev.shifts_s", "aev.rc_v", "aev.eta_v", "aev.shifts_v",
                "afv.weight", "conv_a.agh", "conv_q.agh", "outputs.atomic_shift.shifts.weight",
                "outputs.srcoulomb.rc"}
    with torch.no_grad():
        model.embedding.weight.copy_(sd["afv.weight"][species])
        model.rbf.centers.copy_(sd["aev.shifts_s"])
        model.conv_a.weight.copy_(sd["conv_a.agh"])
        model.conv_q.weight.copy_(sd["conv_q.agh"])
        for i, mlp_i in enumerate(model.passes):
            mlp_i.load_state_dict(_prefixed(sd, f"mlps.{i}."))
            expected |= {f"mlps.{i}.{k}" for k in mlp_i.state_dict()}
        model.readout.load_state_dict(_prefixed(sd, "outputs.energy_mlp.mlp."))
        expected |= {f"outputs.energy_mlp.mlp.{k}" for k in model.readout.state_dict()}
        shifts = sd["outputs.atomic_shift.shifts.weight"].to(torch.float64)
        model.atom_ref.weight[species] = shifts[species]
    # the dipole / quadrupole heads of the rxn models carry an atomic-mass table
    # (unused: they do not center the coordinates); nothing to transplant
    unexpected = sorted(k for k in set(sd) - expected
                        if not k.startswith(("outputs.dipole.", "outputs.quadrupole.")))
    _require(not unexpected, f"the parameters {unexpected}")

    if dtype is not None:
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        model = model.to(dtype)
    return model, cfg


def foundation_to_xnn(source, dtype=None, cache_dir=None, dispersion=None,
                      **model_options) -> nn.Module:
    """Resolve, load and convert a published AIMNet2 model (the body of
    :meth:`AIMNet2.from_foundation`).

    A registered name or alias, URL or DOI goes through the model hub
    (:func:`xnn.common.models.from_pretrained`), which converts the artifact
    once and caches the converted model; a local file or directory is
    converted on the fly, and an already-loaded artifact mapping directly.

    Parameters
    ----------
    source : str, Path or Mapping
        Name / alias / URL / DOI / local path of a checkpoint, or an
        artifact mapping (see :func:`load_artifact`).
    dtype : torch.dtype or str, optional
        Final dtype; ``None`` keeps the checkpoint's.
    cache_dir : str or Path, optional
        Model hub cache directory.
    dispersion : dict, str, bool or None, optional
        ``None`` adds the recorded D3(BJ) term, ``False`` serves the bare
        network; see :func:`~xnn.common.models.hub.load_pretrained`.
    **model_options
        Overrides of the model options (``coulomb``, ``lr_cutoff``,
        ``dsf_alpha``, ``ewald_accuracy``, ``pme_spline_order``), applied
        before the model is built.

    Returns
    -------
    torch.nn.Module
        The potential (bare, or inside its dispersion term), in eval mode.

    Raises
    ------
    TypeError
        If ``source`` names a model that is not an AIMNet2.
    """
    from xnn.common.models.hub import from_pretrained
    from xnn.common.models.hub.checkpoint import build_potential, core_model, wrapped_state_dict
    if isinstance(source, Mapping):
        model, cfg = convert_artifact(source)
        if model_options:
            cfg.model.extra.update(model_options)
        return build_potential(cfg, wrapped_state_dict(model), dtype=dtype,
                               dispersion=dispersion, label="AIMNet2 artifact").model
    s = str(source)
    fmt = None if get_card(s) is not None else "aimnet2"
    model = from_pretrained(source, wrap=False, dtype=dtype, cache_dir=cache_dir,
                            dispersion=dispersion, model_options=model_options or None,
                            format=fmt)
    if not isinstance(core_model(model), AIMNet2):
        raise TypeError(f"{source!r} is a {type(core_model(model)).__name__}, not an AIMNet2")
    return model


def _convert_file(path: Path, head: Optional[str]) -> tuple[AIMNet2, Config]:
    """The model hub's ``aimnet2`` converter: file or directory -> (model, config).

    ``head`` selects the ensemble member of a ``config.json`` +
    ``ensemble_<k>.safetensors`` directory (default 0).
    """
    member = 0 if head is None else int(head)
    return convert_artifact(load_artifact(path, member=member))


register_format(ModelFormat(
    name="aimnet2", suffixes=(".safetensors",), convert=_convert_file, detect=detect_artifact,
    requires="pyyaml"))
