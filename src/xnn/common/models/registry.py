"""Name -> class registry so new models are added without touching the core.

For example::

    from xnn.common.models.registry import register_model, build_model

    @register_model("mymodel")
    class MyModel(InteratomicPotential):
        @classmethod
        def from_config(cls, cfg): ...

A model becomes usable from any config (YAML/Hydra/argparse) the moment
it is imported and registered. This is the extension point referenced in the
README.
"""
from __future__ import annotations

from typing import Callable, Type

_MODELS: dict[str, Type] = {}


def register_model(name: str) -> Callable[[Type], Type]:
    """Return a class decorator that registers a model under ``name``.

    Parameters
    ----------
    name : str
        The name under which to register the decorated class. Lookups are
        case-insensitive (the name is lowercased internally).

    Returns
    -------
    Callable[[Type], Type]
        A decorator that registers the class it wraps and returns it unchanged.

    Raises
    ------
    KeyError
        When applied, if ``name`` is already registered to a different class.

    Examples
    --------
    >>> @register_model("mymodel")
    ... class MyModel(InteratomicPotential):
    ...     ...
    """
    def deco(cls: Type) -> Type:
        """Register ``cls`` under ``name`` and return it unchanged.

        Parameters
        ----------
        cls : type
            The model class being decorated.

        Returns
        -------
        type
            The same class, unmodified.

        Raises
        ------
        KeyError
            If ``name`` is already registered to a different class.
        """
        key = name.lower()
        if key in _MODELS and _MODELS[key] is not cls:
            raise KeyError(f"model '{name}' already registered")
        _MODELS[key] = cls
        return cls
    return deco


def build_model(cfg):
    """Instantiate a registered model from a configuration dataclass.

    Parameters
    ----------
    cfg : ModelConfig
        The model configuration. Its ``name`` attribute selects the registered
        model class (case-insensitively); the whole config is passed to that
        class's ``from_config``.

    Returns
    -------
    InteratomicPotential
        The model instance built by the selected class's ``from_config``,
        with the add-ons the config's ``extra`` asks for (see
        :func:`prepare_model`): a ``"pretrained"`` source replaces the fresh
        build by the weights of any hub model; ``"lora"`` adapts the linear
        layers (:func:`~xnn.common.finetune.inject_lora`); ``"heads"`` gives
        the model several readout heads
        (:class:`~xnn.common.finetune.MultiHead`); ``"dispersion"`` (a dict
        with ``name: d4`` (default) or ``name: d3`` plus the options of
        :class:`~xnn.common.models.d4.DFTD4` / :class:`~xnn.common.models.d3.DFTD3`,
        or ``True`` for the PBE0-D4 defaults) wraps the model with the
        dispersion correction (:class:`~xnn.common.models.d4.D4Dispersion` or
        :class:`~xnn.common.models.d3.D3Dispersion`); ``"long_range"`` (a
        dict of :class:`~xnn.common.models.les.LatentEwald` options, or
        ``True`` for the defaults) wraps it with the Latent Ewald Summation
        long-range term. Dispersion is applied before LES, so LES sees the
        D4-corrected model's features (they are passed through unchanged).

    Raises
    ------
    KeyError
        If ``cfg.name`` does not match any registered model.
    """
    return prepare_model(cfg)[0]


#: ``model.extra`` keys that describe how a pretrained model is adapted rather
#: than its architecture.
FINETUNE_KEYS = ("pretrained", "head", "dtype", "heads", "lora")
#: The keys a config with ``pretrained`` may set besides those (everything else
#: comes from the checkpoint).
_PRETRAINED_OVERRIDES = ("heads", "lora", "atomic_energies", "dispersion", "long_range")
_WRAPPER_KEYS = ("dispersion", "long_range")


def prepare_model(cfg):
    """Build a model and the config that rebuilds it.

    The same as :func:`build_model`, also returning the resolved
    :class:`~xnn.common.config.schema.ModelConfig`: with a ``"pretrained"``
    source the architecture keys are taken from the checkpoint and recorded
    in place of the source, so the returned config rebuilds the model from
    its weights alone (what the trainer writes into its checkpoints).

    ``extra["pretrained"]`` names any source
    :func:`~xnn.common.models.from_pretrained` accepts (registry name, local
    checkpoint or model directory, URL, Zenodo DOI); ``extra["head"]`` picks
    the head of a multi-head source; ``extra["dtype"]`` casts the weights.
    ``cfg.cutoff`` must equal the checkpoint's (the data pipeline follows it),
    and the only other keys allowed beside the source are ``heads``,
    ``lora``, ``atomic_energies`` (new reference energies, applied after the
    weights are loaded) and, when the checkpoint has none, ``dispersion`` /
    ``long_range``. A reference-energy marker (``atomic_energies:
    estimated`` / ``average``) needs data and is resolved by the
    :class:`~xnn.common.train.Trainer`; here it is an error.

    Parameters
    ----------
    cfg : ModelConfig
        The model configuration.

    Returns
    -------
    tuple of (InteratomicPotential, ModelConfig)
        The model and the resolved config.

    Raises
    ------
    KeyError
        If the model name is not registered.
    ValueError
        For an inconsistent ``pretrained`` config, or a reference-energy
        marker.
    """
    import copy

    from ..finetune.heads import MultiHead, head_names, head_options
    from ..finetune.lora import inject_lora, lora_options
    from ..finetune.reference import REFERENCE_MARKERS, reference_markers, set_atomic_energies

    markers = reference_markers(cfg)[1]
    if markers:
        raise ValueError(
            f"atomic_energies (MACE E0s) {sorted(set(m for _, m in markers))} must be estimated "
            f"from data: train with the Trainer (which resolves {REFERENCE_MARKERS}) or call "
            "xnn.common.finetune.estimate_atomic_energies / average_atomic_energies")

    extra = dict(cfg.extra or {})
    if extra.get("pretrained") is not None:
        # the checkpoint's own wrappers come loaded; only the ones the config
        # adds on top are built here
        model, cfg, new_wrappers = _from_pretrained(cfg)
        extra = dict(cfg.extra or {})
    else:
        key = cfg.name.lower()
        if key not in _MODELS:
            raise KeyError(f"unknown model '{cfg.name}'. registered: {sorted(_MODELS)}")
        model = _MODELS[key].from_config(cfg)
        cfg = copy.deepcopy(cfg)
        new_wrappers = {k: extra[k] for k in _WRAPPER_KEYS if extra.get(k)}

    core = core_model(model)
    lora = extra.get("lora")
    if lora:
        inject_lora(core, **lora_options(lora))
    heads = extra.get("heads")
    if heads:
        names = head_names(heads)
        multi = MultiHead(core, names)
        for name in names:
            values = head_options(heads, name).get("atomic_energies")
            if values is not None:
                multi.set_atomic_energies(name, _as_reference_mapping(values, core))
        model = replace_module(model, core, multi)

    for key in _WRAPPER_KEYS:
        if not new_wrappers.get(key):
            continue
        if key == "dispersion":
            model = add_dispersion(model, new_wrappers[key])
        else:
            from .les import LatentEwald
            opts = dict(new_wrappers[key]) if isinstance(new_wrappers[key], dict) else {}
            model = LatentEwald(model, **opts)
    return model, cfg


def _as_reference_mapping(values, model) -> dict:
    """``{Z: E0}`` from a mapping (Z or symbol keys) or a per-species sequence."""
    from ..config.coerce import coerce_per_species, coerce_species
    if isinstance(values, dict):
        zs = coerce_species(list(values.keys()))
        return {z: float(v) for z, v in zip(zs, values.values())}
    species = getattr(model, "species", None)
    if species is None:
        raise ValueError("per-species atomic_energies need a model that lists its species; "
                         "use a {Z: E0} mapping")
    return dict(zip([int(z) for z in species],
                    coerce_per_species(values, species, "atomic_energies")))


def _from_pretrained(cfg):
    """Load the ``pretrained`` source of ``cfg``.

    Returns the loaded model (with the wrappers of its own config), the
    resolved model config, and the wrapper entries of ``cfg`` that the
    checkpoint lacks (for :func:`prepare_model` to add on top).
    """
    import copy
    import dataclasses

    from ..config.schema import ModelConfig
    from ..finetune.reference import set_atomic_energies
    from .hub import fetch_model, load_checkpoint
    from .hub.checkpoint import build_potential

    extra = dict(cfg.extra or {})
    source = extra.pop("pretrained")
    head = extra.pop("head", None)
    dtype = extra.pop("dtype", None)
    ck = load_checkpoint(fetch_model(source), head=head)
    if ck.config is None:
        raise ValueError(f"{source} embeds no config; pack it with save_pretrained first")
    base = ck.config.model
    default_name = ModelConfig().name
    if cfg.name.lower() not in (base.name.lower(), default_name):
        raise ValueError(f"model.name={cfg.name!r} but {source} holds a {base.name!r} model")
    if abs(float(cfg.cutoff) - float(base.cutoff)) > 1e-9:
        raise ValueError(f"model.cutoff={cfg.cutoff} does not match the cutoff {base.cutoff} "
                         f"of {source}; set cutoff: {base.cutoff} in the config")
    base_extra = dict(base.extra or {})
    for key, value in extra.items():
        if key in _PRETRAINED_OVERRIDES:
            continue
        if key in base_extra and base_extra[key] == value:
            continue                      # the checkpoint's own value, restated
        raise ValueError(
            f"model.{key} cannot be set together with pretrained={source!r}: the "
            f"architecture comes from the checkpoint (allowed: {_PRETRAINED_OVERRIDES})")
    for key in _WRAPPER_KEYS:
        if extra.get(key) and base_extra.get(key):
            raise ValueError(f"{source} already includes {key}; drop model.{key}")

    loaded = build_potential(ck.config, ck.state_dict, dispersion=False, head=head, dtype=dtype,
                             label=str(source))
    model = loaded.model
    values = extra.get("atomic_energies")
    if values is not None and not extra.get("heads"):
        set_atomic_energies(core_model(model), _as_reference_mapping(values, core_model(model)))

    resolved_extra = {k: v for k, v in base_extra.items() if k not in FINETUNE_KEYS}
    resolved_extra.update({k: v for k, v in extra.items() if k in _PRETRAINED_OVERRIDES})
    resolved = dataclasses.replace(copy.deepcopy(base), extra=resolved_extra)
    new_wrappers = {k: extra[k] for k in _WRAPPER_KEYS if extra.get(k) and not base_extra.get(k)}
    return model, resolved, new_wrappers


def core_model(model):
    """The innermost potential under the force, dispersion, LES and multi-head wrappers."""
    from ..finetune.heads import MultiHead
    from .dispersion import DispersionCorrection
    from .les import LatentEwald
    from .outputs import ForceStressOutput
    while isinstance(model, (ForceStressOutput, DispersionCorrection, LatentEwald, MultiHead)):
        inner = getattr(model, "model", None)
        if inner is None:
            break
        model = inner
    return model


def replace_module(chain, old, new):
    """Replace ``old`` by ``new`` inside a chain of wrappers linked by ``.model``.

    Parameters
    ----------
    chain : torch.nn.Module
        The outermost module (``old`` itself, or a wrapper around it).
    old, new : torch.nn.Module
        The module to replace and its replacement.

    Returns
    -------
    torch.nn.Module
        ``new`` when ``chain is old``, otherwise ``chain`` with the
        replacement made in place.

    Raises
    ------
    ValueError
        If ``old`` is not in the chain.
    """
    if chain is old:
        return new
    parent = chain
    while True:
        inner = getattr(parent, "model", None)
        if inner is None:
            raise ValueError(f"{type(old).__name__} is not wrapped by {type(chain).__name__}")
        if inner is old:
            parent.model = new
            return chain
        parent = inner


def deployment_config(cfg):
    """A copy of ``cfg`` describing the deployed (single-head, LoRA-merged) model.

    :func:`~xnn.common.models.hub.build_potential` serves one head with the
    LoRA updates folded in, so the ``heads`` / ``lora`` / ``head`` /
    ``pretrained`` entries of the model section no longer apply to the model
    it returns.
    """
    import copy
    out = copy.deepcopy(cfg)
    model_cfg = out.model if hasattr(out, "model") else out
    model_cfg.extra = {k: v for k, v in (model_cfg.extra or {}).items() if k not in FINETUNE_KEYS}
    return out

def available_models() -> list[str]:
    """List the names of all currently registered models.

    Returns
    -------
    list of str
        The registered model names, sorted alphabetically.
    """
    return sorted(_MODELS)


def recorded_dispersion(cfg) -> dict | None:
    """The ``subtracted_dispersion`` record of a (possibly older) ``Config``.

    Checkpoints written before the field existed unpickle without it, so
    the attribute is read with a default.
    """
    return getattr(cfg, "subtracted_dispersion", None)


def resolve_dispersion(explicit, recorded):
    """The dispersion a deployment adds to a checkpoint's model.

    ``recorded`` is the checkpoint's ``subtracted_dispersion`` record (what
    the labels had removed, with the settings to add it back); ``explicit``
    is what the caller asked for. The rules: ``explicit=False`` serves the
    model as is; with no record the explicit request is used; with a record
    and no request the record is used; with both, the explicit keys override
    the recorded ones (so ``{"cutoff_triple": 8.0}`` keeps the recorded
    functional parameters), and a different term name is an error.

    Returns
    -------
    dict, str, True or None
        The spec for :func:`add_dispersion`, or ``None`` to add nothing.

    Raises
    ------
    ValueError
        When the explicit spec names another term than the record.
    """
    if explicit is False:
        return None
    if recorded is None:
        return explicit
    record = {"name": recorded} if isinstance(recorded, str) else dict(recorded)
    record.setdefault("name", "d4")
    if explicit is None or explicit is True:
        return record
    override = {"name": explicit} if isinstance(explicit, str) else dict(explicit)
    if str(override.get("name", record["name"])).lower() != str(record["name"]).lower():
        raise ValueError(
            f"the checkpoint records {record['name']} as subtracted from its labels; "
            f"a {override['name']} correction cannot be added instead")
    return {**record, **override}


def add_dispersion(model, spec):
    """Wrap ``model`` in a D3 or D4 dispersion correction.

    The same hook :func:`build_model` applies for ``extra["dispersion"]``,
    exposed so deployment code can add dispersion to a trained model after
    its weights are loaded.

    Parameters
    ----------
    model : InteratomicPotential or None
        The short-range model to correct (``None`` for pure dispersion).
    spec : dict, str or True
        ``{"name": "d4" | "d3", **options}`` with the options of
        :class:`~xnn.common.models.d4.DFTD4` /
        :class:`~xnn.common.models.d3.DFTD3`; a bare name string; or ``True``
        for the PBE0-D4 defaults.

    Returns
    -------
    DispersionCorrection
        :class:`~xnn.common.models.d4.D4Dispersion` or
        :class:`~xnn.common.models.d3.D3Dispersion` around ``model``.

    Raises
    ------
    KeyError
        For a dispersion name other than ``"d3"`` or ``"d4"``.
    """
    from .d3 import D3Dispersion, d3_options_from_extra
    from .d4 import D4Dispersion, d4_options_from_extra
    if isinstance(spec, str):
        spec = {"name": spec}
    opts = dict(spec) if isinstance(spec, dict) else {}
    name = str(opts.pop("name", "d4")).lower()
    if name == "d4":
        return D4Dispersion(model, **d4_options_from_extra(opts))
    if name == "d3":
        return D3Dispersion(model, **d3_options_from_extra(opts))
    raise KeyError(f"unknown dispersion model '{name}' (use 'd3' or 'd4')")
