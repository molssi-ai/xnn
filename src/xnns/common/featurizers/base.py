"""The Featurizer abstraction.

A `Featurizer` turns an :class:`~xnns.common.data.AtomicGraph` into model inputs --
either invariant per-atom descriptors (symmetry functions, AEV) or equivariant
edge/node embeddings (spherical-harmonic edge attributes). Featurizers are
plain ``nn.Module``s and are *independently usable*: you can instantiate one and
call it on a graph without any model, e.g. to inspect descriptors or to build
your own model on top.

Two output conventions are used in this package, distinguished by what a model
needs:

* **Invariant featurizers** (``RadialSymmetryFunctions``, ``AngularSymmetryFunctions``,
  ``AEV``) return a single ``(N, output_dim)`` per-atom descriptor tensor.
* **Equivariant featurizers** (``SphericalHarmonicEdgeEmbedding``) return a dict
  of edge tensors (geometry + spherical harmonics + radial embedding) consumed
  by the GNN interaction blocks.

Both subclass :class:`Featurizer` so they share the ``output_dim`` contract and
can be discovered / swapped uniformly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from torch import nn

from ..data import AtomicGraph


class Featurizer(nn.Module, ABC):
    """Abstract base class for all featurizers.

    A featurizer is a plain ``nn.Module`` that maps an
    :class:`~xnns.common.data.AtomicGraph` to model inputs -- either invariant
    per-atom descriptors or equivariant edge/node embeddings. Subclasses share
    the ``output_dim`` contract so they can be discovered and swapped uniformly.

    Notes
    -----
    Featurizers are independently usable: an instance can be called on a graph
    without any surrounding model, e.g. to inspect descriptors.
    """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Size of the per-atom feature vector.

        Returns
        -------
        int
            The per-atom descriptor width for invariant featurizers.
            Equivariant featurizers that return dicts may report the scalar
            (``l=0``) channel width here, or raise if not meaningful.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, data: AtomicGraph):
        """Compute features for an atomic graph.

        Parameters
        ----------
        data : AtomicGraph
            The input atomic graph (positions, neighbor lists, etc.).

        Returns
        -------
        torch.Tensor or dict[str, torch.Tensor]
            A single ``(N, output_dim)`` per-atom descriptor tensor for
            invariant featurizers, or a dict of edge tensors for equivariant
            featurizers.
        """
        raise NotImplementedError
