"""Hybrid GNN + transformer interatomic potentials.

This family holds models that combine graph message passing with a transformer
(attention) update and a physics-based split of the energy:

* :class:`~xnn.hybrid.models.bamboo.BAMBOO` (Gong et al. 2024), a graph
  equivariant transformer whose energy is the sum of a semi-local
  neural-network term, a charge-equilibrium electrostatic term, and an
  optional D3(CSO) dispersion term;
* :class:`~xnn.hybrid.models.spookynet.SpookyNet` (Unke et al. 2021), message
  passing with s-, p- and d-orbital-like interactions, charge and spin
  embeddings and self-attention over all atoms, with ZBL repulsion, damped
  electrostatics and D4 dispersion added to the energy.

BAMBOO reuses the shared transformer primitives in :mod:`xnn.transformer`;
SpookyNet's radial basis lives in :mod:`xnn.hybrid.featurizers`. Both follow the
common :class:`~xnn.common.models.base.InteratomicPotential` contract, so they
plug into the same training/deploy pipeline as every other xnn model.
"""
from . import featurizers, models  # noqa: F401  (registers the hybrid models)

__all__ = ["featurizers", "models"]
