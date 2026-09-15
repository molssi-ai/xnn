"""Loaders that turn YAML / argparse / Hydra into one `Config`.

The interchangeability you asked for comes from a single rule: every frontend
produces a plain nested dict, and `from_dict` builds the typed `Config` from it.
Dotted-key overrides (``model.cutoff=6.0``) are supported uniformly so the same
override syntax works on the CLI and with Hydra. Upstream key spellings in the
``model`` section (MACE-CLI ``r_max``, NequIP ``num_layers``, ...) are rewritten
to the xnns canonical names by the translation registry (see `translate.py`).
"""
from __future__ import annotations

import dataclasses
from typing import Any

from .schema import Config, ModelConfig, DataConfig, OptimConfig
from .translate import translate_model_keys

_SECTIONS = {"model": ModelConfig, "data": DataConfig, "optim": OptimConfig}


def from_dict(d: dict[str, Any]) -> Config:
    """Build a typed :class:`Config` from a plain nested dict.

    This is the funnel every frontend loader passes through. Recognized
    sections (``model``, ``data``, ``optim``) are dispatched to their dataclass;
    the ``model`` section first has upstream key spellings translated to the
    xnns canonical names (:func:`~xnns.common.config.translate.translate_model_keys`),
    then unknown keys inside it are collected into ``ModelConfig.extra``. Known
    top-level scalars (``device``, ``seed``, ``output_dir``) are applied
    directly. Unrecognized keys are ignored.

    Parameters
    ----------
    d : dict[str, Any]
        Nested configuration mapping (as produced by any frontend). A value of
        ``None`` is treated as an empty dict.

    Returns
    -------
    Config
        The populated configuration object.
    """
    d = dict(d or {})
    kwargs: dict[str, Any] = {}
    for key, klass in _SECTIONS.items():
        sub = d.pop(key, {}) or {}
        if key == "model":
            sub = translate_model_keys(sub)
        valid = {f.name for f in dataclasses.fields(klass)}
        known = {k: v for k, v in sub.items() if k in valid}
        if key == "model":
            known.setdefault("extra", {})
            known["extra"].update({k: v for k, v in sub.items() if k not in valid})
        kwargs[key] = klass(**known)
    # top-level scalars (device, seed, output_dir)
    top_valid = {f.name for f in dataclasses.fields(Config)} - set(_SECTIONS)
    for k in list(d):
        if k in top_valid:
            kwargs[k] = d.pop(k)
    return Config(**kwargs)


def apply_overrides(cfg: Config, overrides: list[str]) -> Config:
    """Apply ``a.b=c`` style dotted overrides in place.

    Shared by the CLI and Hydra paths so the same override syntax works
    everywhere. Each value is parsed with :func:`ast.literal_eval` and falls
    back to the raw string if parsing fails. Entries without ``=`` are skipped.
    :meth:`Config.__post_init__` is re-run afterwards to re-synchronize the
    cutoffs.

    Parameters
    ----------
    cfg : Config
        The configuration to mutate.
    overrides : list[str]
        Override strings of the form ``"model.cutoff=6.0"``.

    Returns
    -------
    Config
        The same ``cfg`` instance, with overrides applied.
    """
    import ast
    for ov in overrides:
        if "=" not in ov:
            continue
        path, raw = ov.split("=", 1)
        try:
            val = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            val = raw
        obj = cfg
        parts = path.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], val)
    cfg.__post_init__()
    return cfg


def from_yaml(path: str) -> Config:
    """Load a YAML config file into a :class:`Config`.

    The file is parsed with :func:`yaml.safe_load` and funneled through
    :func:`from_dict`.

    Parameters
    ----------
    path : str
        Path to the YAML file.

    Returns
    -------
    Config
        The populated configuration object.
    """
    import yaml
    with open(path) as f:
        return from_dict(yaml.safe_load(f))


def from_argparse(argv: list[str] | None = None) -> Config:
    """Build a :class:`Config` from command-line arguments.

    Parses ``--config path.yaml`` plus repeatable ``--set a.b=c`` overrides.
    The config file is loaded via :func:`from_yaml`. If no config file is
    given, defaults are used. Overrides are then applied via
    :func:`apply_overrides`.

    Parameters
    ----------
    argv : list[str] or None, optional
        Argument list to parse. If ``None``, :data:`sys.argv` is used.

    Returns
    -------
    Config
        The populated configuration object.
    """
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None,
                   help="YAML config file")
    p.add_argument("--set", dest="overrides", action="extend", nargs="+",
                   default=[], metavar="KEY=VALUE",
                   help="dotted override(s); repeatable, several per flag")
    args = p.parse_args(argv)

    cfg = Config() if args.config is None else from_yaml(args.config)
    return apply_overrides(cfg, args.overrides)


def from_hydra(dict_config) -> Config:
    """Build a :class:`Config` from a Hydra / OmegaConf ``DictConfig``.

    Converts the ``DictConfig`` to a plain container (resolving interpolations)
    via :func:`omegaconf.OmegaConf.to_container`, falling back to ``dict()`` if
    OmegaConf is not installed, then funnels it through :func:`from_dict`.

    Parameters
    ----------
    dict_config : omegaconf.DictConfig
        The Hydra-provided configuration node.

    Returns
    -------
    Config
        The populated configuration object.

    Examples
    --------
    In your Hydra entrypoint::

        from omegaconf import OmegaConf

        @hydra.main(config_path="configs", config_name="train")
        def main(cfg):
            config = from_hydra(cfg)
    """
    try:
        from omegaconf import OmegaConf
        container = OmegaConf.to_container(dict_config, resolve=True)
    except ModuleNotFoundError:
        container = dict(dict_config)
    return from_dict(container)
