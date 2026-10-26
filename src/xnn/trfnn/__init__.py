"""Transformer family (``trfnn``): graph-transformer potentials and their shared attention primitive.

* :class:`~xnn.trfnn.models.et.EquivariantTransformer` (``"et"``), the
  equivariant transformer of TorchMD-Net (Thoelke & De Fabritiis, ICLR 2022,
  with the cutoff fix of TorchMD-Net 2.0);
* :class:`~xnn.trfnn.models.equiformer.Equiformer` (``"equiformer"``), the
  equivariant graph attention transformer (Liao & Smidt, ICLR 2023);
* :class:`~xnn.trfnn.models.equiformer_v2.EquiformerV2` (``"equiformer_v2"``),
  its eSCN-based successor for higher degrees (Liao et al., ICLR 2024);
* :class:`~xnn.trfnn.attention.EdgeMultiheadAttention`, the multi-head
  QKV attention on graph edges that :class:`~xnn.hybnn.models.bamboo.BAMBOO`
  builds on.

The exponential-normal radial basis these models share with BAMBOO and
TensorNet lives in :mod:`xnn.common.featurizers`.
"""
from . import attention, models  # noqa: F401  (registers the transformer models)

__all__ = ["attention", "models"]
