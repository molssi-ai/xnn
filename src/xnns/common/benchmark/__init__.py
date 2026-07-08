"""Benchmark several models over one dataset and tabulate their errors.

Engages all registered models (or any subset), driving each through a
configurable combination of train / evaluate / benchmark phases and writing a
comparison table of error metrics in user-selectable formats. Everything is
built on the shared abstractions: the :class:`~xnns.common.config.Config` tree
and model registry, the :class:`~xnns.common.train.Trainer`, and
:class:`~xnns.common.models.ForceStressOutput`.

Programmatic entry point::

    from xnns.common.benchmark import from_yaml, run_benchmark
    run_benchmark(from_yaml("configs/benchmark.yaml"))

or via the CLI::

    xnns benchmark --config configs/benchmark.yaml

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
from .report import (
    register_writer, available_writers, write, write_all, format_table,
)
from .runner import Benchmark, run_benchmark

__all__ = [
    "BenchmarkConfig", "ModelEntry", "OutputConfig", "from_dict", "from_yaml",
    "register_metric", "get_metric", "available_metrics", "load_custom_metric",
    "collect_predictions", "score",
    "register_writer", "available_writers", "write", "write_all",
    "format_table",
    "Benchmark", "run_benchmark",
]
