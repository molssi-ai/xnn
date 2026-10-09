"""Shared infrastructure of the ``dnn`` family.

* :mod:`~xnn.dnn.common.runner` -- the RuNNer model files (``input.nn``, the
  weights and scaling files) read into :class:`~xnn.dnn.models.hdnnp.HDNNP`.
"""
from .runner import load_runner_model, parse_input_nn, runner_feature_key

__all__ = ["load_runner_model", "parse_input_nn", "runner_feature_key"]
