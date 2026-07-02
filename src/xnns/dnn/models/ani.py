"""ANI (Smith et al. 2017): per-element networks on the Atomic Environment Vector.

ANI is exactly an HDNNP whose descriptor is the AEV -- radial *and* angular
symmetry functions. With the angular term implemented in
``xnns.dnn.featurizers.AEV``, this is a faithful ANI in architecture (element
networks over radial+angular AEV); tune the symmetry-function grids in the
config ``extra`` to match a specific ANI parameterization (e.g. ANI-1x).
"""
from __future__ import annotations

from xnns.common.models.registry import register_model
from xnns.dnn.featurizers import AEV
from .base import DescriptorPotential


@register_model("ani")
class ANI(DescriptorPotential):
    """ANI (Smith et al. 2017): per-element networks on the AEV.

    A thin :class:`DescriptorPotential` whose featurizer is the
    :class:`~xnns.dnn.featurizers.AEV` (radial *and* angular symmetry
    functions). Architecturally identical to :class:`HDNNP` but with the AEV as
    descriptor; tune the symmetry-function grids via ``aev_kwargs`` to match a
    specific ANI parameterization.

    Parameters
    ----------
    species : sequence of int
        Atomic numbers to build per-element networks for and to resolve the AEV
        into.
    radial_cutoff : float, optional
        Cutoff radius for the radial part of the AEV, by default 5.2.
    angular_cutoff : float, optional
        Cutoff radius for the angular part of the AEV, by default 3.5.
    hidden : sequence of int, optional
        Hidden-layer widths of each per-element MLP, by default
        ``(128, 96, 64)``.
    aev_kwargs : dict, optional
        Extra keyword arguments forwarded to :class:`AEV` (e.g.
        symmetry-function grids); by default ``None``.
    """

    def __init__(self, species, radial_cutoff=5.2, angular_cutoff=3.5,
                 hidden=(128, 96, 64), aev_kwargs=None):
        aev = AEV(species, radial_cutoff=radial_cutoff,
                  angular_cutoff=angular_cutoff, **(aev_kwargs or {}))
        super().__init__(aev, species, hidden)

    @classmethod
    def from_config(cls, cfg):
        """Build an :class:`ANI` from a configuration object.

        Parameters
        ----------
        cfg : object
            Configuration exposing an optional ``extra`` mapping (with keys
            ``species``, ``radial_cutoff``, ``angular_cutoff``, ``hidden`` and
            ``aev_kwargs``); defaults are used for any missing entries.

        Returns
        -------
        ANI
            Instantiated model.
        """
        extra = cfg.extra or {}
        return cls(
            species=extra.get("species", [1, 6, 7, 8]),
            radial_cutoff=extra.get("radial_cutoff", 5.2),
            angular_cutoff=extra.get("angular_cutoff", 3.5),
            hidden=extra.get("hidden", (128, 96, 64)),
            aev_kwargs=extra.get("aev_kwargs"),
        )
