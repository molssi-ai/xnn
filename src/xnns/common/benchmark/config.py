"""Configuration for a multi-model benchmark run.

A benchmark orchestrates several models over one dataset, so its config wraps
the *same* building blocks a single run uses -- :class:`DataConfig`,
:class:`OptimConfig`, :class:`ModelConfig` -- rather than inventing parallel
ones. Each entry in ``models`` becomes a :class:`ModelEntry`, and
:meth:`ModelEntry.to_run_config` folds it together with the shared data/optim
sections into an ordinary :class:`Config`, so training and evaluation reuse the
unchanged :class:`~xnns.common.train.Trainer` and model registry.

The one internal representation is :class:`BenchmarkConfig`; every frontend is
just a loader that produces it (mirroring ``config.loaders``). :func:`from_dict`
is the funnel, and :func:`from_yaml` reads a YAML file through it.

Model entries accept either a bare string (``"mace"``) or a mapping that mixes
architecture keys with benchmark-only keys::

    models:
      - mace                                    # train from defaults
      - name: schnet
        cutoff: 5.0                             # architecture override
        checkpoint: runs/schnet/best.pt         # skip training, load this
      - name: nequip
        config: configs/model/nequip.yaml       # architecture from a file
        optim: {epochs: 50}                     # per-model training override

Upstream key spellings inside a model entry (MACE ``r_max`` ...) are translated
to the canonical names by the shared model-key registry, exactly as in a normal
run.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional

from ..config import Config, DataConfig, OptimConfig
from ..config.loaders import from_dict as _run_from_dict

# Keys inside a model entry that steer the benchmark itself rather than the
# model architecture; stripped out before the rest is read as a ModelConfig.
_ENTRY_KEYS = {"label", "checkpoint", "config", "optim", "output_dir"}


@dataclass
class ModelEntry:
    """One model in a benchmark: its architecture plus how to obtain weights.

    Attributes
    ----------
    label : str
        Human-readable name for this entry in the results table. Defaults to
        the model name; :func:`from_dict` disambiguates duplicates by suffix.
    model : dict[str, Any]
        The model section as a plain dict (already merged with any ``config``
        file and with upstream key spellings translated). Turned into a
        :class:`~xnns.common.config.ModelConfig` by :meth:`to_run_config`.
    checkpoint : Optional[str]
        Path to a pre-trained checkpoint (as written by
        :meth:`~xnns.common.train.Trainer.save`). When set, the benchmark can
        skip training and load these weights directly. Defaults to ``None``.
    optim : Optional[dict[str, Any]]
        Per-model optimizer/loss overrides layered on top of the shared
        ``optim`` section during training. Defaults to ``None``.
    output_dir : Optional[str]
        Directory for this model's training artifacts (checkpoints). Defaults
        to ``<benchmark output_dir>/<label>`` when not given.
    """

    label: str
    model: dict[str, Any]
    checkpoint: Optional[str] = None
    optim: Optional[dict[str, Any]] = None
    output_dir: Optional[str] = None

    def to_run_config(self, bench: "BenchmarkConfig") -> Config:
        """Assemble the ordinary :class:`Config` used to train/evaluate this model.

        Folds this entry's model section and per-model optim overrides together
        with the benchmark's shared ``data`` section, device and seed, routing
        everything through the standard :func:`~xnns.common.config.from_dict`
        funnel so model-key translation and ``extra`` collection behave exactly
        as in a single run.

        Parameters
        ----------
        bench : BenchmarkConfig
            The parent benchmark configuration providing the shared data,
            optim, device, seed and output directory.

        Returns
        -------
        Config
            A fully-populated run configuration for this model, with
            ``output_dir`` set to this entry's :attr:`output_dir` (or a
            per-label subdirectory of the benchmark output directory).
        """
        optim = dataclasses.asdict(bench.optim)
        optim.update(self.optim or {})
        out_dir = self.output_dir or f"{bench.output_dir}/{self.label}"
        return _run_from_dict({
            "model": dict(self.model),
            "data": dataclasses.asdict(bench.data),
            "optim": optim,
            "device": bench.device,
            "seed": bench.seed,
            "output_dir": out_dir,
        })


@dataclass
class OutputConfig:
    """Where and in which formats to write the benchmark results table.

    Attributes
    ----------
    dir : str
        Directory the results (and per-model training artifacts) are written
        to. Defaults to ``"runs/benchmark"``.
    filename : str
        Base filename (without extension) for the results table; each format
        appends its own extension. Defaults to ``"results"``.
    formats : list[str]
        Output formats to write, each a registered writer name
        (``"csv"`` / ``"json"`` / ``"md"`` or a user-registered one; see
        ``benchmark.report``). Defaults to ``["csv", "json"]``.
    """

    dir: str = "runs/benchmark"
    filename: str = "results"
    formats: list[str] = field(default_factory=lambda: ["csv", "json"])


@dataclass
class BenchmarkConfig:
    """Top-level configuration for a multi-model benchmark.

    The single internal representation produced by every loader in this module.

    Attributes
    ----------
    models : list[ModelEntry]
        The models to benchmark, in table order.
    data : DataConfig
        Shared dataset paths, split fractions and target-key names, reused
        unchanged from a normal run.
    optim : OptimConfig
        Shared training settings, used for any model trained in the ``train``
        phase and layered under each entry's per-model ``optim`` overrides.
    phases : list[str]
        Which phases to run, a subset of ``"train"``, ``"evaluate"`` and
        ``"benchmark"`` (see :class:`~xnns.common.benchmark.runner.Benchmark`).
        Defaults to all three.
    metrics : list[str]
        Registered error-metric names applied to every target (e.g.
        ``["mae", "rmse"]``). Defaults to ``["mae", "rmse"]``.
    targets : list[str]
        Quantities to score, a subset of ``"energy"``, ``"forces"`` and
        ``"stress"``. Defaults to ``["energy", "forces"]``.
    custom_metrics : list[dict[str, str]]
        User-defined metrics to import and register before scoring, each a
        ``{"name": ..., "path": "module:function"}`` mapping (see
        :func:`~xnns.common.benchmark.metrics.load_custom_metric`). Defaults to
        an empty list.
    output : OutputConfig
        Results destination and formats.
    device : str
        Compute device (``auto`` / ``cpu`` / ``cuda`` / ``cuda:0`` ...).
        Defaults to ``"auto"``.
    seed : int
        Random seed for reproducible data splits and training. Defaults to
        ``1234``.
    """

    models: list[ModelEntry] = field(default_factory=list)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    phases: list[str] = field(
        default_factory=lambda: ["train", "evaluate", "benchmark"])
    metrics: list[str] = field(default_factory=lambda: ["mae", "rmse"])
    targets: list[str] = field(default_factory=lambda: ["energy", "forces"])
    custom_metrics: list[dict[str, str]] = field(default_factory=list)
    output: OutputConfig = field(default_factory=OutputConfig)
    device: str = "auto"
    seed: int = 1234

    @property
    def output_dir(self) -> str:
        """Shortcut for :attr:`output.dir`, the benchmark output directory."""
        return self.output.dir


def _as_list(x: Any) -> list:
    """Coerce a scalar, ``None`` or list into a list.

    Lets config fields accept ``"mace"``, ``["mace", "schnet"]`` or ``null``
    interchangeably.

    Parameters
    ----------
    x : Any
        A single value, a list/tuple of values, or ``None``.

    Returns
    -------
    list
        ``[]`` for ``None``, ``list(x)`` for a list/tuple, otherwise ``[x]``.
    """
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _model_entry(spec: Any, seen: dict[str, int]) -> ModelEntry:
    """Build one :class:`ModelEntry` from a string or mapping spec.

    A string is shorthand for ``{"name": spec}``. A mapping may carry an
    architecture (the model section, optionally seeded from a ``config`` YAML
    file) alongside the benchmark-only keys in :data:`_ENTRY_KEYS`. Labels are
    made unique by appending ``#2``, ``#3`` ... to repeats.

    Parameters
    ----------
    spec : str or dict
        The model entry as written in the config.
    seen : dict[str, int]
        Running count of labels already used, mutated to keep labels unique.

    Returns
    -------
    ModelEntry
        The parsed entry.

    Raises
    ------
    TypeError
        If ``spec`` is neither a string nor a mapping.
    """
    if isinstance(spec, str):
        spec = {"name": spec}
    elif not isinstance(spec, dict):
        raise TypeError(f"model entry must be a string or mapping, got {spec!r}")

    spec = dict(spec)
    checkpoint = spec.pop("checkpoint", None)
    optim = spec.pop("optim", None)
    output_dir = spec.pop("output_dir", None)
    label = spec.pop("label", None)

    # A model entry may seed its architecture from a standalone model YAML,
    # with inline keys overriding the file.
    config_path = spec.pop("config", None)
    model_section = spec
    if config_path is not None:
        import yaml
        with open(config_path) as f:
            base = yaml.safe_load(f) or {}
        base.update(model_section)
        model_section = base

    label = label or model_section.get("name", "model")
    seen[label] = seen.get(label, 0) + 1
    if seen[label] > 1:
        label = f"{label}#{seen[label]}"

    return ModelEntry(label=label, model=model_section, checkpoint=checkpoint,
                      optim=optim, output_dir=output_dir)


def from_dict(d: dict[str, Any]) -> BenchmarkConfig:
    """Build a :class:`BenchmarkConfig` from a plain nested dict.

    The funnel every frontend loader passes through. The ``models`` list may
    hold strings and/or mappings (see :func:`_model_entry`); ``data`` and
    ``optim`` reuse the dataclasses from a normal run (unknown keys ignored);
    ``metrics``, ``targets`` and ``phases`` accept a scalar or a list; and
    ``output`` populates :class:`OutputConfig`.

    Parameters
    ----------
    d : dict[str, Any]
        Nested benchmark configuration mapping. ``None`` is treated as empty.

    Returns
    -------
    BenchmarkConfig
        The populated benchmark configuration.
    """
    d = dict(d or {})

    seen: dict[str, int] = {}
    models = [_model_entry(s, seen) for s in _as_list(d.get("models"))]

    def _sub(klass, key):
        sub = d.get(key) or {}
        valid = {f.name for f in dataclasses.fields(klass)}
        return klass(**{k: v for k, v in sub.items() if k in valid})

    data = _sub(DataConfig, "data")
    optim = _sub(OptimConfig, "optim")
    output = _sub(OutputConfig, "output")
    if "formats" in (d.get("output") or {}):
        output.formats = _as_list(d["output"]["formats"])

    kwargs: dict[str, Any] = dict(models=models, data=data, optim=optim,
                                  output=output)
    if "phases" in d:
        kwargs["phases"] = _as_list(d["phases"])
    if "metrics" in d:
        kwargs["metrics"] = _as_list(d["metrics"])
    if "targets" in d:
        kwargs["targets"] = _as_list(d["targets"])
    if "custom_metrics" in d:
        kwargs["custom_metrics"] = _as_list(d["custom_metrics"])
    for k in ("device", "seed"):
        if k in d:
            kwargs[k] = d[k]

    return BenchmarkConfig(**kwargs)


def from_yaml(path: str) -> BenchmarkConfig:
    """Load a YAML benchmark config into a :class:`BenchmarkConfig`.

    Parameters
    ----------
    path : str
        Path to the YAML file; parsed with :func:`yaml.safe_load` and funneled
        through :func:`from_dict`.

    Returns
    -------
    BenchmarkConfig
        The populated benchmark configuration.
    """
    import yaml
    with open(path) as f:
        return from_dict(yaml.safe_load(f))
