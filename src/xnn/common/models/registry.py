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
        The model instance built by the selected class's ``from_config``.
        When the config's ``extra`` carries a ``"dispersion"`` entry (a dict
        with ``name: d4`` (default) or ``name: d3`` plus the options of
        :class:`~xnn.common.models.d4.DFTD4` / :class:`~xnn.common.models.d3.DFTD3`,
        or ``True`` for the PBE0-D4 defaults), the model is wrapped with the
        dispersion correction (:class:`~xnn.common.models.d4.D4Dispersion` or
        :class:`~xnn.common.models.d3.D3Dispersion`); when it
        carries a ``"long_range"`` entry (a dict of
        :class:`~xnn.common.models.les.LatentEwald` options, or ``True`` for
        the defaults), the model is wrapped with the Latent Ewald Summation
        long-range term. Both may be combined; dispersion is applied first,
        so LES sees the D4-corrected model's features (they are passed
        through unchanged).

    Raises
    ------
    KeyError
        If ``cfg.name`` does not match any registered model.
    """
    key = cfg.name.lower()
    if key not in _MODELS:
        raise KeyError(
            f"unknown model '{cfg.name}'. registered: {sorted(_MODELS)}"
        )
    model = _MODELS[key].from_config(cfg)
    dispersion = (cfg.extra or {}).get("dispersion")
    if dispersion:
        from .d3 import D3Dispersion, d3_options_from_extra
        from .d4 import D4Dispersion, d4_options_from_extra
        opts = dict(dispersion) if isinstance(dispersion, dict) else {}
        name = str(opts.pop("name", "d4")).lower()
        if name == "d4":
            model = D4Dispersion(model, **d4_options_from_extra(opts))
        elif name == "d3":
            model = D3Dispersion(model, **d3_options_from_extra(opts))
        else:
            raise KeyError(f"unknown dispersion model '{name}' (use 'd3' or 'd4')")
    long_range = (cfg.extra or {}).get("long_range")
    if long_range:
        from .les import LatentEwald
        opts = dict(long_range) if isinstance(long_range, dict) else {}
        model = LatentEwald(model, **opts)
    return model


def available_models() -> list[str]:
    """List the names of all currently registered models.

    Returns
    -------
    list of str
        The registered model names, sorted alphabetically.
    """
    return sorted(_MODELS)
