"""Reading, writing and building models from checkpoints.

Three kinds of file hold a model's weights, and :func:`load_checkpoint` reads
all of them into the same :class:`Checkpoint`:

* a **portable model directory** (``card.json``, ``config.yaml``,
  ``model.pt``), written by :func:`save_pretrained`. The config is plain YAML
  and the weights load with ``torch.load(weights_only=True)``, so the
  directory carries no pickled Python objects and no absolute paths: copy it
  anywhere and it loads there;
* an **xnn trainer checkpoint** (``best.pt``: ``{"model": state_dict, "cfg":
  Config}``, written by :meth:`~xnn.common.train.Trainer.save`), or a bare
  state dict;
* a file in a **foreign format** registered in
  :mod:`~xnn.common.models.hub.formats` (e.g. a ``mace-torch`` foundation
  model), converted on the fly.

:func:`build_potential` turns a checkpoint into a ready model. It is the one
loading recipe behind :func:`~xnn.common.models.hub.from_pretrained`, the MDI
engine, ``xnn export`` and the benchmark runner.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import itertools
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import torch
from torch import nn

from .card import CARD_FILE, CONFIG_FILE, WEIGHTS_FILE, ModelCard
from .formats import XNN_FORMAT, detect_format, get_format

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    """A model's weights and the config that rebuilds it.

    Attributes
    ----------
    config : Config or None
        The run configuration (``None`` for a bare state dict).
    state_dict : dict of str to torch.Tensor
        Weights in the :class:`~xnn.common.models.ForceStressOutput` layout
        (keys prefixed ``model.``), as the trainer saves them.
    card : ModelCard or None
        The card of a portable directory.
    path : pathlib.Path
        Where the checkpoint was read from.
    """

    config: Any
    state_dict: dict
    card: Optional[ModelCard]
    path: Path


# config <-> YAML
def _plain(value: Any, where: str) -> Any:
    """Convert a config value to plain YAML types (lists, dicts, scalars)."""
    if isinstance(value, dict):
        return {(k if isinstance(k, (str, int, float, bool)) else str(k)):
                _plain(v, f"{where}.{k}") for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v, f"{where}[{i}]") for i, v in enumerate(value)]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):          # numpy arrays and scalars
        return value.tolist()
    if dataclasses.is_dataclass(value):
        return _dataclass_dict(value, where)
    raise TypeError(f"config value {where} = {value!r} ({type(value).__name__}) "
                    f"cannot be written to YAML")


def _dataclass_dict(obj: Any, where: str = "cfg") -> dict[str, Any]:
    """``dataclasses.asdict`` that tolerates instances pickled by older xnn.

    A field missing from the instance (added to the class after the
    checkpoint was written) takes its default.
    """
    out = {}
    for f in dataclasses.fields(obj):
        if hasattr(obj, f.name):
            v = getattr(obj, f.name)
        elif f.default is not dataclasses.MISSING:
            v = f.default
        elif f.default_factory is not dataclasses.MISSING:
            v = f.default_factory()
        else:
            continue
        out[f.name] = _plain(v, f"{where}.{f.name}")
    return out


def config_to_dict(config: Any) -> dict[str, Any]:
    """Serialize a :class:`Config` (or a bare :class:`ModelConfig`) to a dict.

    Top-level sections and fields equal to the defaults are left out, so a
    converted foundation model's file holds just its ``model`` section while
    a trained model keeps its data and optimizer settings as provenance.

    Parameters
    ----------
    config : Config or ModelConfig
        The configuration.

    Returns
    -------
    dict
        Mapping accepted by :func:`xnn.common.config.from_dict`.
    """
    from ...config import Config, ModelConfig
    if isinstance(config, ModelConfig):
        return {"model": _dataclass_dict(config, "model")}
    d = _dataclass_dict(config)
    default = _dataclass_dict(Config())
    return {k: v for k, v in d.items() if k == "model" or v != default.get(k)}


def file_references(config: Any) -> list[str]:
    """Config values of the model section that name existing files.

    A config that points at a file (a ReaxFF ``ffield`` path, an OPLS or
    DREIDING topology JSON) loads only where that file exists, which breaks
    the portability of a model directory.

    Parameters
    ----------
    config : Config or ModelConfig
        The configuration.

    Returns
    -------
    list of str
        ``"<key>=<value>"`` for each such value.
    """
    found = []

    def walk(v, where):
        if isinstance(v, dict):
            for k, x in v.items():
                walk(x, f"{where}.{k}")
        elif isinstance(v, list):
            for i, x in enumerate(v):
                walk(x, f"{where}[{i}]")
        elif isinstance(v, str) and ("/" in v or "\\" in v or "." in v):
            try:
                if Path(v).expanduser().is_file():
                    found.append(f"{where}={v}")
            except (OSError, ValueError):
                pass

    walk(config_to_dict(config)["model"], "model")
    return found


def write_config(config: Any, path: Union[str, Path]) -> Path:
    """Write a config as YAML, atomically (see :func:`config_to_dict`)."""
    import yaml
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(yaml.safe_dump(config_to_dict(config), sort_keys=False))
    tmp.replace(path)
    return path


def read_config(path: Union[str, Path]):
    """Read a config written by :func:`write_config`."""
    import yaml
    from ...config import from_dict
    return from_dict(yaml.safe_load(Path(path).read_text()) or {})


# weights
def float_dtype(source) -> torch.dtype:
    """Floating-point dtype of a module's parameters or of a state dict."""
    # a module's parameters first, then its buffers: a standalone dispersion
    # model (``name: d4``) has buffers only
    tensors = (source.values() if isinstance(source, dict)
               else itertools.chain(source.parameters(), source.buffers()))
    for t in tensors:
        if torch.is_tensor(t) and t.is_floating_point():
            return t.dtype
    return torch.get_default_dtype()


def has_dispersion(model: nn.Module) -> bool:
    """Whether a model already includes a D3 / D4 wrapper (at any nesting)."""
    from ..dispersion import DispersionCorrection
    from ..les import LatentEwald
    while True:
        if isinstance(model, DispersionCorrection):
            return True
        if isinstance(model, LatentEwald):
            model = model.model
        else:
            return False


def core_model(model: nn.Module) -> nn.Module:
    """The innermost potential under the force, dispersion, LES and multi-head wrappers."""
    from ..registry import core_model as _core
    return _core(model)


def wrapped_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """State dict in the trainer's :class:`ForceStressOutput` layout, on CPU."""
    from ..outputs import ForceStressOutput
    if not isinstance(model, ForceStressOutput):
        model = ForceStressOutput(model)
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    """MD5 hex digest of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_checkpoint(path: Union[str, Path], *, format: Optional[str] = None,
                    head: Optional[str] = None) -> Checkpoint:
    """Read any supported checkpoint into a :class:`Checkpoint`.

    Parameters
    ----------
    path : str or pathlib.Path
        A portable model directory, an xnn trainer checkpoint (or bare state
        dict), or a file in a registered foreign format.
    format : str, optional
        Force the format instead of detecting it from the suffix
        (``"xnn"``, ``"mace-torch"``).
    head : str, optional
        Head to keep when converting a multi-head foreign checkpoint. For an
        xnn checkpoint the heads are kept and chosen at build time
        (:func:`build_potential`).

    Returns
    -------
    Checkpoint
        Config, weights and card.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist, or a directory lacks its config/weights.
    ValueError
        If the file is not a recognizable checkpoint.
    """
    from ...config import Config, from_dict
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint at {path}")
    # a foreign format first: its detect() may claim a directory or a .pt file
    fmt = format or detect_format(path)
    if fmt != XNN_FORMAT:
        model, cfg = get_format(fmt).convert(path, head)
        if not isinstance(cfg, Config):
            cfg = Config(model=cfg)
        return Checkpoint(cfg, wrapped_state_dict(model), None, path)
    if path.is_dir():
        missing = [f for f in (CONFIG_FILE, WEIGHTS_FILE) if not (path / f).is_file()]
        if missing:
            raise FileNotFoundError(f"{path} is not a model directory (missing {missing})")
        state = torch.load(path / WEIGHTS_FILE, map_location="cpu", weights_only=True)
        sd = state["model"] if isinstance(state.get("model"), dict) else state
        card = ModelCard.load(path / CARD_FILE) if (path / CARD_FILE).is_file() else None
        return Checkpoint(read_config(path / CONFIG_FILE), sd, card, path)

    # trainer checkpoints pickle their Config, hence weights_only=False
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, nn.Module):
        raise ValueError(
            f"{path} holds a pickled {type(state).__module__}.{type(state).__name__} "
            f"module, not an xnn checkpoint; for an upstream MACE model pass "
            f"format='mace-torch'")
    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        cfg = state.get("cfg")
        if isinstance(cfg, dict):
            cfg = from_dict(cfg)
        return Checkpoint(cfg, state["model"], None, path)
    if isinstance(state, dict) and state and all(torch.is_tensor(v) for v in state.values()):
        return Checkpoint(None, state, None, path)
    raise ValueError(f"{path} is not a recognizable xnn checkpoint")


def save_pretrained(model: Union[nn.Module, str, Path],
                    save_directory: Union[str, Path], *, config: Any = None,
                    card: Union[ModelCard, dict, None] = None, verify: bool = True,
                    **card_fields: Any) -> Path:
    """Write a model as a portable directory.

    The directory holds ``config.yaml`` (the configuration that rebuilds the
    architecture), ``model.pt`` (the weights, loadable with
    ``weights_only=True``) and ``card.json`` (the :class:`ModelCard`, with an
    MD5 per file). The card is written last, so a directory with a card is
    complete. Nothing in it refers to the machine it was written on.

    Parameters
    ----------
    model : torch.nn.Module or str or pathlib.Path
        A model (bare, or wrapped in :class:`ForceStressOutput`) or the path
        of any checkpoint :func:`load_checkpoint` reads (e.g. a trainer's
        ``best.pt``), which is then repacked.
    save_directory : str or pathlib.Path
        Output directory; created if missing.
    config : Config or ModelConfig, optional
        The configuration the model was built from. Taken from the checkpoint
        when ``model`` is a path; required for a module.
    card : ModelCard or dict, optional
        Card fields (name, description, license, citation, ...). The
        architecture, cutoff, species, dtype, files and xnn version are
        filled in from the model.
    verify : bool, optional
        Rebuild the architecture from ``config`` and load the weights
        strictly before writing, so a mismatched config fails here rather
        than at load time. Defaults to ``True``.
    **card_fields
        More card fields, overriding ``card``.

    Returns
    -------
    pathlib.Path
        The model directory.

    Raises
    ------
    ValueError
        If no config is available, or ``config`` does not rebuild the weights.
    """
    import xnn
    from ...config import ModelConfig
    from ..outputs import ForceStressOutput
    from ..registry import build_model

    out = Path(save_directory).expanduser()
    fields: dict[str, Any] = {}
    module = None
    if isinstance(model, (str, Path)):
        ck = load_checkpoint(model)
        sd, config = ck.state_dict, (config if config is not None else ck.config)
        if ck.card is not None:
            fields.update(ck.card.to_dict())
            fields.pop("files", None)
        fields.setdefault("source", Path(model).name)
    else:
        sd = wrapped_state_dict(model)
        module = model.model if isinstance(model, ForceStressOutput) else model
    if config is None:
        raise ValueError("save_pretrained needs the model's config (pass config=)")
    model_cfg = config if isinstance(config, ModelConfig) else config.model

    if verify or module is None:
        try:
            module = build_model(model_cfg)
            ForceStressOutput(module).load_state_dict(sd)
        except (RuntimeError, KeyError) as e:
            raise ValueError(
                f"the config does not rebuild these weights ({model_cfg.name}): {e}") from e

    if isinstance(card, ModelCard):
        fields.update(card.to_dict())
    elif card:
        fields.update(card)
    fields.update(card_fields)
    fields.pop("files", None)
    from ...finetune.heads import find_multihead
    core = core_model(module)
    species = getattr(core, "species", None) or (model_cfg.extra or {}).get("species")
    fields.update(
        format=XNN_FORMAT,
        architecture=model_cfg.name,
        cutoff=float(getattr(module, "cutoff", model_cfg.cutoff)),
        species=[int(z) for z in species] if species is not None else None,
        heads=(list(find_multihead(module).heads) if find_multihead(module) is not None
               else fields.get("heads")),
        dtype=str(float_dtype(sd)).replace("torch.", ""),
        xnn_version=getattr(xnn, "__version__", None),
    )
    fields.setdefault("name", out.name)
    new_card = ModelCard.from_dict(fields)

    refs = file_references(config)
    if refs:
        logger.warning("the config of %s refers to files outside the model directory (%s); "
                       "it loads only where those paths exist. Use packaged library names "
                       "or inline data to make it portable", out, ", ".join(refs))
    out.mkdir(parents=True, exist_ok=True)
    (out / CARD_FILE).unlink(missing_ok=True)       # incomplete until rewritten
    write_config(config, out / CONFIG_FILE)
    tmp = out / f"{WEIGHTS_FILE}.tmp{os.getpid()}"
    torch.save({"model": {k: v.detach().cpu() for k, v in sd.items()}}, tmp)
    tmp.replace(out / WEIGHTS_FILE)
    new_card.files = {f: _md5(out / f) for f in (CONFIG_FILE, WEIGHTS_FILE)}
    new_card.save(out / CARD_FILE)
    return out


def _as_dtype(dtype) -> Optional[torch.dtype]:
    """Accept ``torch.float64`` or ``"float64"``."""
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    return getattr(torch, str(dtype).replace("torch.", ""))


def build_potential(config: Any, state_dict: dict, *, dtype=None, dispersion: Any = None,
                    compute_forces: bool = True, compute_stress: bool = False,
                    eeq_reuse: bool = False, model_options: Optional[dict[str, Any]] = None,
                    head: Optional[str] = None, label: str = "checkpoint"):
    """Rebuild a model from its config and weights.

    The model is built in float64, so the constant tables of the physics
    terms (D3 / D4 reference data, LES kernels) hold their exact values, and
    cast once afterwards, to ``dtype`` or to the weights' own dtype. Building
    in float32 and upcasting would keep float32-rounded tables, which costs
    about 5e-8 hartree in a D4 energy even when serving in float64.

    A config that records ``subtracted_dispersion`` (what the training labels
    had removed) gets that term added back unless ``dispersion=False``; see
    :func:`~xnn.common.models.registry.resolve_dispersion`.

    A fine-tuned checkpoint is served as a plain potential: of a multi-head
    model (``extra["heads"]``) the head named by ``head`` is kept
    (:meth:`~xnn.common.finetune.MultiHead.select`), and LoRA updates
    (``extra["lora"]``) are folded into the base weights
    (:func:`~xnn.common.finetune.merge_lora`).

    Parameters
    ----------
    config : Config
        The run configuration.
    state_dict : dict
        Weights in the :class:`ForceStressOutput` layout.
    dtype : torch.dtype or str, optional
        Dtype to serve in. Defaults to the dtype of the weights.
    dispersion : dict, str, bool or None, optional
        D3 / D4 correction to add (as in ``extra["dispersion"]``); ``None``
        adds the recorded one, if any, and ``False`` adds nothing.
    compute_forces, compute_stress : bool, optional
        Heads of the returned :class:`ForceStressOutput`.
    eeq_reuse : bool, optional
        Carry the D4 EEQ solve over between steps (molecular dynamics).
    model_options : dict, optional
        Overrides of ``config.model.extra`` applied before the model is
        built: deployment knobs that change no weights, such as AIMNet2's
        Coulomb method (``{"coulomb": "dsf", "lr_cutoff": 15.0}``); the
        weights still load strictly.
    head : str, optional
        The head of a multi-head checkpoint to serve; required when it has
        several, ignored (if it matches) for a single-head one.
    label : str, optional
        Name used in log messages.

    Returns
    -------
    ForceStressOutput
        The model with its weights loaded, in eval mode.

    Raises
    ------
    ValueError
        If ``dispersion`` is given for a model that already includes one, or
        ``head`` is missing or unknown for a multi-head checkpoint.
    """
    from ...finetune.heads import find_multihead
    from ...finetune.lora import has_lora, merge_lora
    from ..outputs import ForceStressOutput
    from ..registry import (add_dispersion, build_model, recorded_dispersion, replace_module,
                            resolve_dispersion)

    recorded = recorded_dispersion(config)
    spec = resolve_dispersion(dispersion, recorded)
    if model_options:
        config = copy.deepcopy(config)
        config.model.extra = {**(config.model.extra or {}), **model_options}
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        base = build_model(config.model)
        # ForceStressOutput has no parameters of its own; loading through
        # it matches the trainer's state-dict layout
        ForceStressOutput(base).load_state_dict(state_dict)
        multi = find_multihead(base)
        if multi is not None:
            if head is None and len(multi.heads) > 1:
                raise ValueError(f"{label} has the heads {multi.heads}; pass head= with one of them")
            chosen = multi.heads[0] if head is None else str(head)
            if chosen not in multi.heads:
                raise ValueError(f"unknown head {chosen!r} of {label}; it has {multi.heads}")
            base = replace_module(base, multi, multi.select(chosen))
            logger.info("serving head %s of %s", chosen, label)
        elif head is not None:
            logger.debug("%s has a single head; head=%r ignored", label, head)
        if has_lora(base):
            merge_lora(base)
            logger.info("LoRA updates of %s folded into the base weights", label)
        if spec is not None:
            if has_dispersion(base):
                raise ValueError(
                    f"{label} already includes a dispersion correction "
                    f"({config.model.name} with extra['dispersion']); adding "
                    f"another one would count dispersion twice")
            base = add_dispersion(base, spec)
            if recorded is None:
                logger.info("added %s dispersion on top of the checkpoint: %s",
                            type(base).__name__, spec)
            elif dispersion is None or dispersion is True:
                logger.info("adding back the dispersion recorded as subtracted "
                            "from the training labels: %s", spec)
            else:
                logger.info("adding back the dispersion recorded as subtracted "
                            "from the training labels, with overrides %s: %s",
                            dispersion, spec)
        elif recorded is not None:
            logger.warning("serving %s WITHOUT the dispersion recorded as subtracted "
                           "from its training labels (%s)", label, recorded)
    finally:
        torch.set_default_dtype(prev_dtype)
    dtype = _as_dtype(dtype) or float_dtype(state_dict)
    model = ForceStressOutput(base, compute_forces=compute_forces,
                              compute_stress=compute_stress).to(dtype)
    if eeq_reuse:
        from ..d4 import enable_eeq_reuse
        n_terms = enable_eeq_reuse(model)
        if n_terms:
            logger.info("EEQ reuse between steps enabled for %d D4 term(s)", n_terms)
        else:
            logger.info("--eeq-reuse has no effect: the model has no D4 term")
    return model.eval()
