"""Hybrid GNN + transformer interatomic potentials.

This family holds models that combine graph message passing with a transformer
(attention) update and a physics-based split of the energy. The flagship model
is :class:`~xnn.hybrid.models.bamboo.BAMBOO` (Gong et al. 2024), a graph
equivariant transformer whose energy is the sum of a semi-local neural-network
term, a charge-equilibrium electrostatic term, and an optional D3(CSO)
dispersion term.

BAMBOO reuses the shared transformer primitives in :mod:`xnn.transformer`
(the :class:`~xnn.transformer.featurizers.ExpNormalSmearing` radial basis and
the :class:`~xnn.transformer.attention.EdgeMultiheadAttention` core) and the
common :class:`~xnn.common.models.base.InteratomicPotential` contract, so it
plugs into the same training/deploy pipeline as every other xnn model.
"""
from . import models  # noqa: F401  (registers the hybrid models)

__all__ = ["models"]
