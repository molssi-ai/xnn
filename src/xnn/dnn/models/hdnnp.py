"""High-Dimensional Neural Network Potential (Behler-Parrinello, 2007).

A thin composition over a featurizer: it pairs a per-atom invariant descriptor
(here ``RadialSymmetryFunctions``) with per-element atomic networks. The shared
body (:class:`DescriptorPotential`) lives in ``dnn/models/base.py``; swap the
featurizer for an :class:`~xnn.dnn.featurizers.AEV` and you have ANI (see
ani.py) -- the model code is identical.
"""
from __future__ import annotations

from xnn.common.models.registry import register_model
from xnn.dnn.featurizers import RadialSymmetryFunctions
from .base import DescriptorPotential


@register_model("hdnnp")
class HDNNP(DescriptorPotential):
    """High-Dimensional Neural Network Potential (Behler-Parrinello, 2007).

    A thin :class:`DescriptorPotential` whose featurizer is
    :class:`RadialSymmetryFunctions`: per-element atomic networks on radial
    symmetry-function descriptors. Swapping the featurizer for an
    :class:`~xnn.dnn.featurizers.AEV` yields ANI (see ``ani.py``).

    Parameters
    ----------
    species : sequence of int
        Atomic numbers to build per-element networks for and to resolve the
        radial symmetry functions into.
    cutoff : float, optional
        Cutoff radius for the radial symmetry functions, by default 6.0.
    etas : sequence of float, optional
        Radial Gaussian width parameters, by default ``(0.05, 0.5, 2.0, 8.0)``.
    rs : sequence of float, optional
        Radial shifts ``Rs``, by default ``(0.0,)``.
    hidden : sequence of int, optional
        Hidden-layer widths of each per-element MLP, by default ``(64, 64)``.
    """

    def __init__(self, species, cutoff=6.0, etas=(0.05, 0.5, 2.0, 8.0),
                 rs=(0.0,), hidden=(64, 64)):
        featurizer = RadialSymmetryFunctions(species, cutoff, etas=etas, rs=rs)
        super().__init__(featurizer, species, hidden)

    @classmethod
    def from_config(cls, cfg):
        """Build an :class:`HDNNP` from a configuration object.

        Parameters
        ----------
        cfg : object
            Configuration exposing ``cutoff`` and an optional ``extra`` mapping
            (with keys ``species``, ``etas``, ``rs`` and ``hidden``); defaults
            are used for any missing entries.

        Returns
        -------
        HDNNP
            Instantiated model.
        """
        extra = cfg.extra or {}
        return cls(
            species=extra.get("species", [1, 6, 8]),
            cutoff=cfg.cutoff,
            etas=extra.get("etas", (0.05, 0.5, 2.0, 8.0)),
            rs=extra.get("rs", (0.0,)),
            hidden=extra.get("hidden", (64, 64)),
        )
