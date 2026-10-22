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

Several GPUs or nodes: launch the same command through ``torchrun`` to score
each model data-parallel, or split the models over runs (``--models`` /
``--shard``, one Slurm job-array task each, then ``--merge``), or let
``--parallel`` fan the models out over the GPUs of one node.

Extension points mirror the model registry: register new error metrics with
:func:`register_metric` and new output formats with :func:`register_writer`.
"""
from __future__ import annotations

from .config import (
    BenchmarkConfig,
    ModelEntry,
    OutputConfig,
    from_dict,
    from_yaml,
)
from .energy import build_e0_lookup, fit_atomic_energies
from .metrics import (
    available_metrics,
    collect_predictions,
    get_metric,
    load_custom_metric,
    prediction_pairs,
    register_metric,
    score,
)
from .report import (
    apply_units,
    available_writers,
    format_table,
    register_writer,
    write,
    write_all,
)
from .runner import (
    Benchmark,
    merge_parts,
    run_benchmark,
    run_parallel,
    select_entries,
)

__all__ = [
    "BenchmarkConfig", "ModelEntry", "OutputConfig", "from_dict", "from_yaml",
    "register_metric", "get_metric", "available_metrics", "load_custom_metric",
    "collect_predictions", "prediction_pairs", "score", "build_e0_lookup", "fit_atomic_energies",
    "register_writer", "available_writers", "write", "write_all",
    "format_table", "apply_units",
    "Benchmark", "run_benchmark", "select_entries", "merge_parts", "run_parallel",
]
