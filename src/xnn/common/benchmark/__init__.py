"""Benchmark several pre-trained models on one dataset and tabulate their errors.

Benchmarking does one thing: it scores the listed models -- each built from its
architecture and loaded from its ``checkpoint`` -- on one dataset and writes a
comparison table of error metrics in user-selectable formats. It does not train
or evaluate during training; produce the checkpoints first (e.g. with
``xnn train``). Model building reuses the shared abstractions: the
:class:`~xnn.common.config.Config` tree and model registry, and
:class:`~xnn.common.models.ForceStressOutput`.

Via the CLI::

    xnn benchmark --config configs/benchmark.yaml

or programmatically::

    from xnn.common.benchmark import from_yaml, run_benchmark
    run_benchmark(from_yaml("configs/benchmark.yaml"))

Extension points mirror the model registry: register new error metrics with
:func:`register_metric` and new output formats with :func:`register_writer`.
"""
from __future__ import annotations

from .config import (
    BenchmarkConfig, ModelEntry, OutputConfig, from_dict, from_yaml,
)
from .metrics import (
    register_metric, get_metric, available_metrics, load_custom_metric,
    collect_predictions, score,
)
from .energy import build_e0_lookup, fit_atomic_energies
from .report import (
    register_writer, available_writers, write, write_all, format_table,
    apply_units,
)
from .runner import Benchmark, run_benchmark

__all__ = [
    "BenchmarkConfig", "ModelEntry", "OutputConfig", "from_dict", "from_yaml",
    "register_metric", "get_metric", "available_metrics", "load_custom_metric",
    "collect_predictions", "score", "build_e0_lookup", "fit_atomic_energies",
    "register_writer", "available_writers", "write", "write_all",
    "format_table", "apply_units",
    "Benchmark", "run_benchmark",
]
