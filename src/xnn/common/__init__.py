"""Shared abstractions used across all model families (cnn / dnn / gnn)."""
# Before anything else: this applies the torch compatibility shims, and must
# run ahead of any import that pulls in e3nn (see _torch_compat).
from ._torch_compat import allow_e3nn_constants

allow_e3nn_constants()

from . import data, featurizers, config, models, train, deploy, benchmark, cli  # noqa: E402

__all__ = ["data", "featurizers", "config", "models", "train", "deploy",
           "benchmark", "cli"]
