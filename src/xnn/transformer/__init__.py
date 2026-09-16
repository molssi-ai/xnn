"""Transformer building blocks shared by attention-based potentials.

This family package holds the pieces that graph-*transformer* potentials reuse
regardless of the rest of their architecture:

* :class:`~xnn.transformer.featurizers.ExpNormalSmearing` -- the TorchMD-Net
  exponential-normal radial basis (also used by ET / TensorNet / BAMBOO);
* :class:`~xnn.transformer.attention.EdgeMultiheadAttention` -- a multi-head
  QKV attention on graph *edges* (the shared attention primitive of the
  BAMBOO graph-equivariant transformer, factored out so future
  transformer-based models can build on it).

The concrete hybrid GNN+transformer model that consumes these lives under
:mod:`xnn.hybrid` (see :class:`xnn.hybrid.models.bamboo.BAMBOO`).
"""
from . import attention, featurizers  # noqa: F401

__all__ = ["attention", "featurizers"]
