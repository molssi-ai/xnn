"""Name -> class registry so new models are added without touching the core.

For example::

    from xnns.common.models.registry import register_model, build_model

    @register_model("mymodel")
    class MyModel(InteratomicPotential):
        @classmethod
        def from_config(cls, cfg): ...

A model becomes usable from any config (YAML/TOML/Hydra/argparse) the moment
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
    return _MODELS[key].from_config(cfg)


def available_models() -> list[str]:
    """List the names of all currently registered models.

    Returns
    -------
    list of str
        The registered model names, sorted alphabetically.
    """
    return sorted(_MODELS)
