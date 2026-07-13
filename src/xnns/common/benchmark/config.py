"""Configuration for a multi-model benchmark.

Benchmarking scores *pre-trained* models on one dataset and tabulates their
errors -- it does not train or otherwise produce models (train with
``xnns train`` first). Its config therefore wraps only the building blocks a
scoring pass needs -- :class:`DataConfig` for the dataset and
:class:`ModelConfig` for each model's architecture -- rather than inventing
parallel ones. Each entry in ``models`` becomes a :class:`ModelEntry` that
pairs an architecture with the ``checkpoint`` whose weights are loaded into it,
and :meth:`ModelEntry.to_config` folds it together with the shared ``data``
section into an ordinary :class:`Config` so the model is built exactly as in a
normal run.

The one internal representation is :class:`BenchmarkConfig`; every frontend is
just a loader that produces it (mirroring ``config.loaders``). :func:`from_dict`
is the funnel, and :func:`from_yaml` reads a YAML file through it.

An xnns checkpoint stores the :class:`Config` it was trained with, so an entry
usually needs only its ``checkpoint`` -- the architecture is read from the
checkpoint. A mapping may still carry a ``label`` (the row name), an explicit
architecture (for checkpoints that embed no config), or a ``config`` file::

    models:
      - checkpoint: runs/mace/best.pt           # architecture read from the checkpoint
      - label: nequip
        checkpoint: runs/nequip/best.pt
      - name: schnet                            # explicit architecture (no embedded config)
        config: configs/model/schnet.yaml
        checkpoint: runs/schnet/best.pt

Upstream key spellings inside a model entry (MACE ``r_max`` ...) are translated
to the canonical names by the shared model-key registry, exactly as in a normal
run.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional

from ..config import Config, DataConfig
from ..config.loaders import from_dict as _run_from_dict

# Keys inside a model entry that steer the benchmark itself rather than the
# model architecture; stripped out before the rest is read as a ModelConfig.
_ENTRY_KEYS = {"label", "checkpoint", "config"}


@dataclass
class ModelEntry:
    """One model in a benchmark: its architecture plus the weights to score.

    Attributes
    ----------
    label : str
        Human-readable name for this entry in the results table. Defaults to
        the model name; :func:`from_dict` disambiguates duplicates by suffix.
    model : dict[str, Any]
        The model section as a plain dict (already merged with any ``config``
        file and with upstream key spellings translated). Turned into a
        :class:`~xnns.common.config.ModelConfig` by :meth:`to_config`. Optional
        when the checkpoint was written by :meth:`~xnns.common.train.Trainer.save`
        and carries its own config: that stored architecture is used, so an
        entry can be as small as just a ``checkpoint``. Supply the architecture
        here only for checkpoints that do not embed one.
    checkpoint : Optional[str]
        Path to the pre-trained checkpoint (as written by
        :meth:`~xnns.common.train.Trainer.save`) whose weights are loaded into
        the model before scoring. Required for the model to be benchmarked;
        defaults to ``None``.
    """

    label: str
    model: dict[str, Any]
    checkpoint: Optional[str] = None

    def to_config(self, bench: "BenchmarkConfig",
                  model: Optional[dict[str, Any]] = None) -> Config:
        """Assemble the ordinary :class:`Config` used to build this model.

        Folds a model section together with the benchmark's shared ``data``
        section, device and seed, routing everything through the standard
        :func:`~xnns.common.config.from_dict` funnel so model-key translation
        and ``extra`` collection behave exactly as in a single run. The synced
        ``data.cutoff`` (kept in lockstep with ``model.cutoff`` by
        :class:`Config`) is what the benchmark dataset's neighbor list uses.

        Parameters
        ----------
        bench : BenchmarkConfig
            The parent benchmark configuration providing the shared data,
            device and seed.
        model : dict[str, Any] or None, optional
            The model section to build from. Defaults to this entry's
            :attr:`model`; the runner passes the architecture read from the
            checkpoint here so a checkpoint's own config is used.

        Returns
        -------
        Config
            A run configuration for this model (model + data + device + seed).
        """
        return _run_from_dict({
            "model": dict(self.model if model is None else model),
            "data": dataclasses.asdict(bench.data),
            "device": bench.device,
            "seed": bench.seed,
        })


@dataclass
class OutputConfig:
    """Where and in which formats to write the benchmark results table.

    Attributes
    ----------
    dir : str
        Directory the results table is written to. Defaults to
        ``"runs/benchmark"``.
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
        The pre-trained models to benchmark, in table order. Each must carry a
        ``checkpoint``; the architecture is read from the checkpoint when it
        embeds a config.
    data : DataConfig
        The dataset to score on and its target-key names, reused unchanged from
        a normal run. The benchmark dataset is resolved from ``test_path``,
        falling back to ``val_path`` then ``train_path``.
    metrics : list[str]
        Registered error-metric names applied to every target (e.g.
        ``["mae", "rmse"]``). Defaults to ``["mae", "rmse"]``.
    targets : list[str]
        Quantities to score, a subset of ``"energy"``, ``"forces"`` and
        ``"stress"``. Defaults to ``["energy", "forces"]``.
    atomic_energies : Any
        Per-element reference energies (E0s). When set, energy is scored as the
        atomization / interaction energy (total minus the summed atomic
        references) -- the physically meaningful quantity. Accepts a
        ``{Z: E0}`` / ``{symbol: E0}`` dict, a list aligned with :attr:`species`,
        a single number, the string form of any of these, or ``"average"`` to
        fit E0s from the benchmark dataset by least squares (see
        :func:`~xnns.common.benchmark.energy.build_e0_lookup`). Defaults to
        ``None`` (raw total energy).
    species : Any
        Atomic numbers or chemical symbols the :attr:`atomic_energies` values
        are aligned with, needed only for the list / scalar forms. Defaults to
        ``None``.
    energy_per_atom : bool
        Whether energy metrics are computed per atom (dividing by the atom
        count). Defaults to ``True``.
    units : dict[str, str]
        Physical units to show next to each target's metrics in the printed
        table, keyed by target. xnns is unit-agnostic, so these are labels
        only; the runner fills in defaults for any target not given here --
        ``"eV/atom"`` (or ``"eV"`` when :attr:`energy_per_atom` is off) for
        energy, ``"eV/A"`` for forces, ``"eV/A**3"`` for stress. Set e.g.
        ``{energy: "meV/atom"}`` to match your data. Defaults to an empty dict
        (all defaults).
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
        Random seed (kept for reproducibility of any stochastic metric).
        Defaults to ``1234``.
    """

    models: list[ModelEntry] = field(default_factory=list)
    data: DataConfig = field(default_factory=DataConfig)
    metrics: list[str] = field(default_factory=lambda: ["mae", "rmse"])
    targets: list[str] = field(default_factory=lambda: ["energy", "forces"])
    atomic_energies: Any = None
    species: Any = None
    energy_per_atom: bool = True
    units: dict[str, str] = field(default_factory=dict)
    custom_metrics: list[dict[str, str]] = field(default_factory=list)
    output: OutputConfig = field(default_factory=OutputConfig)
    device: str = "auto"
    seed: int = 1234


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

    return ModelEntry(label=label, model=model_section, checkpoint=checkpoint)


def from_dict(d: dict[str, Any]) -> BenchmarkConfig:
    """Build a :class:`BenchmarkConfig` from a plain nested dict.

    The funnel every frontend loader passes through. The ``models`` list may
    hold strings and/or mappings (see :func:`_model_entry`); ``data`` reuses the
    dataclass from a normal run (unknown keys ignored); ``metrics`` and
    ``targets`` accept a scalar or a list; and ``output`` populates
    :class:`OutputConfig`.

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
    output = _sub(OutputConfig, "output")
    if "formats" in (d.get("output") or {}):
        output.formats = _as_list(d["output"]["formats"])

    kwargs: dict[str, Any] = dict(models=models, data=data, output=output)
    if "metrics" in d:
        kwargs["metrics"] = _as_list(d["metrics"])
    if "targets" in d:
        kwargs["targets"] = _as_list(d["targets"])
    if "custom_metrics" in d:
        kwargs["custom_metrics"] = _as_list(d["custom_metrics"])
    for k in ("atomic_energies", "species", "energy_per_atom", "units",
              "device", "seed"):
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
