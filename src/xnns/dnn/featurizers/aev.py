"""Atomic Environment Vector (ANI) -- radial + angular symmetry functions.

The AEV is the concatenation of radial and angular symmetry functions and is
exactly the per-atom input ANI feeds to its element networks. Usable on its own:

    aev = AEV(species=[1, 6, 8])
    x = aev(graph)          # (N, aev.output_dim)
"""
from __future__ import annotations

import torch
from torch import Tensor

from xnns.common.data import AtomicGraph
from xnns.common.featurizers import Featurizer
from .symmetry_functions import RadialSymmetryFunctions, AngularSymmetryFunctions


class AEV(Featurizer):
    """Atomic Environment Vector: concatenated radial + angular descriptors.

    Composes :class:`RadialSymmetryFunctions` and
    :class:`AngularSymmetryFunctions` and concatenates their per-atom outputs
    to form the invariant descriptor ANI feeds to its per-element networks. The
    two sub-featurizers may use different cutoffs; the model's neighbour list
    must use the larger of the two.

    Parameters
    ----------
    species : list[int]
        Atomic numbers the descriptor resolves.
    radial_cutoff : float, optional
        Cutoff radius for the radial symmetry functions, by default 5.2.
    angular_cutoff : float, optional
        Cutoff radius for the angular symmetry functions, by default 3.5.
    radial_etas : sequence of float, optional
        Width parameters for the radial term, by default
        ``(0.5, 1.0, 2.0, 4.0)``.
    radial_rs : sequence of float, optional
        Radial shifts for the radial term, by default
        ``(0.5, 1.1, 1.7, 2.3, 2.9, 3.5, 4.1, 4.7)``.
    angular_etas : sequence of float, optional
        Width parameters for the angular term, by default ``(0.5,)``.
    angular_zetas : sequence of float, optional
        Angular resolution exponents, by default ``(8.0,)``.
    angular_rs : sequence of float, optional
        Radial shifts for the angular term, by default ``(0.5, 1.5, 2.5)``.
    angular_theta_s : sequence of float, optional
        Angular shifts (in radians), by default
        ``(0.0, 1.5708, 3.1416, 4.7124)``.

    Attributes
    ----------
    species : list[int]
        The resolved species.
    radial : RadialSymmetryFunctions
        Radial symmetry-function sub-featurizer.
    angular : AngularSymmetryFunctions
        Angular symmetry-function sub-featurizer.
    cutoff : float
        Neighbour-list cutoff, ``max(radial_cutoff, angular_cutoff)``.
    """

    def __init__(self, species: list[int], radial_cutoff: float = 5.2,
                 angular_cutoff: float = 3.5,
                 radial_etas=(0.5, 1.0, 2.0, 4.0),
                 radial_rs=(0.5, 1.1, 1.7, 2.3, 2.9, 3.5, 4.1, 4.7),
                 angular_etas=(0.5,), angular_zetas=(8.0,),
                 angular_rs=(0.5, 1.5, 2.5),
                 angular_theta_s=(0.0, 1.5708, 3.1416, 4.7124)):
        super().__init__()
        self.species = list(species)
        self.radial = RadialSymmetryFunctions(
            species, radial_cutoff, etas=radial_etas, rs=radial_rs)
        self.angular = AngularSymmetryFunctions(
            species, angular_cutoff, etas=angular_etas, zetas=angular_zetas,
            rs=angular_rs, theta_s=angular_theta_s)
        # the model's neighbor list must use the larger of the two cutoffs
        self.cutoff = max(radial_cutoff, angular_cutoff)

    @property
    def output_dim(self) -> int:
        """int: Descriptor length, the sum of the radial and angular
        symmetry-function output dimensions."""
        return self.radial.output_dim + self.angular.output_dim

    def forward(self, data: AtomicGraph) -> Tensor:
        """Compute the AEV by concatenating radial and angular descriptors.

        Parameters
        ----------
        data : AtomicGraph
            Atomic graph providing atomic numbers, edge index and edge vectors.

        Returns
        -------
        Tensor
            Per-atom AEV of shape ``(N, output_dim)``, the concatenation of the
            radial and angular symmetry functions along the last dimension.
        """
        return torch.cat([self.radial(data), self.angular(data)], dim=-1)
