"""Atomic Environment Vector (ANI) -- radial + angular symmetry functions.

The AEV is the concatenation of radial and angular symmetry functions and is
exactly the per-atom input ANI feeds to its element networks (Smith et al.,
*Chem. Sci.* **8**, 3192, 2017). Usable on its own::

    aev = AEV.ani1x(species=[1, 6, 7, 8])
    x = aev(graph)          # (N, aev.output_dim)

Two ready-made parameterisations are provided as classmethods:

* :meth:`AEV.ani1x` -- the exact constants shipped by ``torchani`` for the
  ANI-1x model (radial cutoff 5.2 A, angular cutoff 3.5 A; 384-length AEV for
  the four elements H, C, N, O). With the ANI/NeuroChem conventions baked in
  (``0.25`` radial prefactor, ``0.95`` cosine factor) this reproduces
  ``torchani.AEVComputer`` element-for-element -- see the fidelity notebook.
* :meth:`AEV.ani1` -- the parameterisation described for the original ANI-1
  potential in the paper (radial cutoff 4.6 A, angular cutoff 3.1 A; 32 radial
  shifts, 8 x 8 angular shifts -> 768-length AEV for H, C, N, O), built with the
  paper's evenly-spaced-shift recipe (Section 3.4).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from xnn.common.data import AtomicGraph
from xnn.common.featurizers import Featurizer
from .symmetry_functions import RadialSymmetryFunctions, AngularSymmetryFunctions

# ANI-1 elements: H, C, N, O.
ANI_SPECIES = [1, 6, 7, 8]

# ANI-2x element set in torchani's order (H, C, N, O, S, F, Cl). The order
# fixes the AEV species/pair bucketing, so keep it for weight transplants.
ANI2X_SPECIES = [1, 6, 7, 8, 16, 9, 17]


def _even_shifts(cutoff: float, n: int, start: float = 0.9) -> tuple[float, ...]:
    """Evenly spaced radial shifts on ``[start, cutoff)`` (NeuroChem recipe)."""
    step = (cutoff - start) / n
    return tuple(start + step * i for i in range(n))


def _angle_shifts(n: int) -> tuple[float, ...]:
    """``n`` evenly spaced angular shifts on ``[0, pi)`` (NeuroChem recipe)."""
    return tuple(math.pi / (2 * n) * (2 * i + 1) for i in range(n))


class AEV(Featurizer):
    """Atomic Environment Vector: concatenated radial + angular descriptors.

    Composes :class:`RadialSymmetryFunctions` and
    :class:`AngularSymmetryFunctions` and concatenates their per-atom outputs to
    form the invariant descriptor ANI feeds to its per-element networks. The two
    sub-featurizers may use different cutoffs; the model's neighbour list uses
    the larger of the two.

    The defaults are the ANI / NeuroChem conventions (radial prefactor ``0.25``,
    cosine factor ``0.95``), so with matching grids this reproduces
    ``torchani.AEVComputer``. Prefer the :meth:`ani1x` / :meth:`ani1`
    classmethods for the two published grids.

    Parameters
    ----------
    species : list[int]
        Atomic numbers the descriptor resolves.
    radial_cutoff : float, optional
        Cutoff radius for the radial symmetry functions, by default 5.2.
    angular_cutoff : float, optional
        Cutoff radius for the angular symmetry functions, by default 3.5.
    radial_etas : sequence of float, optional
        Width parameters for the radial term, by default ``(16.0,)``.
    radial_rs : sequence of float, optional
        Radial shifts for the radial term. Defaults to the 16 ANI-1x shifts.
    angular_etas : sequence of float, optional
        Width parameters for the angular term, by default ``(8.0,)``.
    angular_zetas : sequence of float, optional
        Angular resolution exponents, by default ``(32.0,)``.
    angular_rs : sequence of float, optional
        Radial shifts for the angular term. Defaults to the 4 ANI-1x shifts.
    angular_theta_s : sequence of float, optional
        Angular shifts (radians). Defaults to the 8 ANI-1x shifts.
    radial_prefactor : float, optional
        Constant multiplying the radial term, by default ``0.25`` (ANI /
        NeuroChem). Use ``1.0`` for the Behler-Parrinello convention.
    angular_cos_factor : float, optional
        Value multiplying ``cos(theta)`` before ``acos`` in the angular term, by
        default ``0.95`` (ANI / NeuroChem). Use ``1.0`` for Behler-Parrinello.

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
                 radial_etas=(16.0,),
                 radial_rs=_even_shifts(5.2, 16),
                 angular_etas=(8.0,), angular_zetas=(32.0,),
                 angular_rs=_even_shifts(3.5, 4),
                 angular_theta_s=_angle_shifts(8),
                 radial_prefactor: float = 0.25,
                 angular_cos_factor: float = 0.95):
        super().__init__()
        self.species = list(species)
        self.radial = RadialSymmetryFunctions(
            species, radial_cutoff, etas=radial_etas, rs=radial_rs,
            prefactor=radial_prefactor)
        self.angular = AngularSymmetryFunctions(
            species, angular_cutoff, etas=angular_etas, zetas=angular_zetas,
            rs=angular_rs, theta_s=angular_theta_s, cos_factor=angular_cos_factor)
        # the model's neighbor list must use the larger of the two cutoffs
        self.cutoff = max(radial_cutoff, angular_cutoff)

    @classmethod
    def ani1x(cls, species: list[int] = ANI_SPECIES) -> "AEV":
        """AEV with the exact ANI-1x constants shipped by ``torchani``.

        Radial cutoff 5.2 A (16 shifts), angular cutoff 3.5 A (4 radial x 8
        angular shifts). For the four ANI elements this is a 384-length AEV that
        reproduces ``torchani.AEVComputer`` element-for-element.

        Parameters
        ----------
        species : list[int], optional
            Atomic numbers, by default ``[1, 6, 7, 8]`` (H, C, N, O).

        Returns
        -------
        AEV
            The ANI-1x-configured featurizer.
        """
        return cls(species, radial_cutoff=5.2, angular_cutoff=3.5,
                   radial_etas=(16.0,), radial_rs=_even_shifts(5.2, 16),
                   angular_etas=(8.0,), angular_zetas=(32.0,),
                   angular_rs=_even_shifts(3.5, 4),
                   angular_theta_s=_angle_shifts(8))

    @classmethod
    def ani2x(cls, species: list[int] = ANI2X_SPECIES) -> "AEV":
        """AEV with the exact ANI-2x constants shipped by ``torchani``.

        Radial cutoff 5.1 A (16 shifts), angular cutoff 3.5 A (8 radial x 4
        angular shifts), with the shift grids starting at 0.8 A and the
        ANI-2x widths (eta 19.7 / 12.5, zeta 14.1). For the seven ANI-2x
        elements (H, C, N, O, S, F, Cl) this is a 1008-length AEV that
        reproduces ``torchani.AEVComputer`` element-for-element.

        Parameters
        ----------
        species : list[int], optional
            Atomic numbers, by default ``[1, 6, 7, 8, 16, 9, 17]``
            (H, C, N, O, S, F, Cl, in torchani's order).

        Returns
        -------
        AEV
            The ANI-2x-configured featurizer.
        """
        return cls(species, radial_cutoff=5.1, angular_cutoff=3.5,
                   radial_etas=(19.7,),
                   radial_rs=_even_shifts(5.1, 16, start=0.8),
                   angular_etas=(12.5,), angular_zetas=(14.1,),
                   angular_rs=_even_shifts(3.5, 8, start=0.8),
                   angular_theta_s=_angle_shifts(4))

    @classmethod
    def ani1(cls, species: list[int] = ANI_SPECIES) -> "AEV":
        """AEV with the original ANI-1 parameterisation (Smith et al. 2017).

        Radial cutoff 4.6 A (32 shifts), angular cutoff 3.1 A (8 radial x 8
        angular shifts). For the four ANI elements this is the 768-length AEV
        described in the paper, built with its evenly-spaced-shift recipe.

        Parameters
        ----------
        species : list[int], optional
            Atomic numbers, by default ``[1, 6, 7, 8]`` (H, C, N, O).

        Returns
        -------
        AEV
            The ANI-1-configured featurizer.
        """
        return cls(species, radial_cutoff=4.6, angular_cutoff=3.1,
                   radial_etas=(16.0,), radial_rs=_even_shifts(4.6, 32),
                   angular_etas=(8.0,), angular_zetas=(8.0,),
                   angular_rs=_even_shifts(3.1, 8),
                   angular_theta_s=_angle_shifts(8))

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
