"""The ANI potential: per-element networks on the Atomic Environment Vector.

ANI (Smith et al., *Chem. Sci.* **8**, 3192, 2017) is an HDNNP whose descriptor
is the AEV -- radial *and* angular symmetry
functions -- feeding one neural network per element, whose scalar outputs are
summed (plus a per-element self energy) into the total energy. The AEV lives in
:class:`xnns.dnn.featurizers.AEV` and reproduces ``torchani.AEVComputer``
element-for-element; the per-element-network body is the shared
:class:`~xnns.dnn.models.base.DescriptorPotential`.

Three published parameterisations are exposed as classmethods:

* :meth:`ANI.ani1` -- the original ANI-1 potential of the paper: radial cutoff
  4.6 A, angular cutoff 3.1 A (768-length AEV for H, C, N, O), pyramidal
  ``768:128:128:64:1`` element networks with a Gaussian activation.
* :meth:`ANI.ani1x` -- the ANI-1x architecture matching ``torchani``: radial
  cutoff 5.2 A, angular cutoff 3.5 A (384-length AEV), per-element network
  widths (H ``160:128:96``, C ``144:112:96``, N/O ``128:112:96``) with the
  ``CELU`` activation. Building this and transplanting ``torchani``'s pretrained
  weights reproduces its energies and forces (see the fidelity notebook).
* :meth:`ANI.ani1ccx` -- the ANI-1ccx potential (Smith et al., *Nat. Commun.*
  **10**, 2903, 2019): the *same* architecture as ANI-1x, retrained by transfer
  learning on CCSD(T)*/CBS coupled-cluster data. Only the self atomic energies
  (and the trained weights) differ, so the preset delegates to :meth:`ANI.ani1x`.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

from xnns.common.config.coerce import coerce_per_species, coerce_species
from xnns.common.models.registry import register_model
from xnns.dnn.featurizers import AEV
from xnns.dnn.featurizers.aev import (
    ANI_SPECIES, _angle_shifts as _ang, _even_shifts as _even)
from .base import DescriptorPotential

# ANI-1x per-element hidden-layer widths (torchani), keyed by atomic number.
ANI1X_HIDDEN: dict[int, tuple[int, ...]] = {
    1: (160, 128, 96),   # H
    6: (144, 112, 96),   # C
    7: (128, 112, 96),   # N
    8: (128, 112, 96),   # O
}

# ANI-1x self atomic energies in Hartree (torchani EnergyShifter), order H C N O.
ANI1X_SELF_ENERGIES: dict[int, float] = {
    1: -0.60095298,
    6: -38.08316124,
    7: -54.70775770,
    8: -75.19446356,
}

# ANI-1ccx self atomic energies in Hartree: the paper's "ANI-1x CCSD(T)*/CBS
# linear fitting parameters" (Smith et al. 2019, SI S1.2.3), identical to
# torchani's ani-1ccx_8x sae_linfit values.
ANI1CCX_SELF_ENERGIES: dict[int, float] = {
    1: -0.5991501324919538,
    6: -38.03750806057356,
    7: -54.67448347695333,
    8: -75.16043537275567,
}


@register_model("ani")
class ANI(DescriptorPotential):
    """ANI (Smith et al. 2017): per-element networks on the AEV.

    A :class:`DescriptorPotential` whose featurizer is the
    :class:`~xnns.dnn.featurizers.AEV` (radial *and* angular symmetry functions).
    Prefer the :meth:`ani1` / :meth:`ani1x` / :meth:`ani1ccx` classmethods for
    the published parameterisations; the raw constructor exposes every knob for
    custom grids and architectures.

    The presets are distinct published models, not tunings of one:
    :meth:`ani1` is the original ANI-1 (Smith et al. 2017; 768-length AEV,
    4.6/3.1 A cutoffs, a uniform ``768:128:128:64:1`` network, Gaussian
    activation) trained on the dense 20 M-conformation ANI-1 dataset, while
    :meth:`ani1x` is the later ANI-1x (Smith et al. 2018) built by active
    learning, with a leaner 384-length AEV (5.2/3.5 A cutoffs), torchani's
    per-element network widths, ``CELU`` activation, and ANI-1x self energies.
    :meth:`ani1ccx` (Smith et al. 2019) keeps the ANI-1x architecture but was
    trained by transfer learning to CCSD(T)*/CBS coupled-cluster data, so only
    its self energies (and trained weights) differ. From a config, the
    ``preset`` key (``"ani-1"`` / ``"ani-1x"`` / ``"ani-1ccx"``) selects among
    them; see :meth:`from_config`.

    Parameters
    ----------
    species : sequence of int, optional
        Atomic numbers to build per-element networks for and to resolve the AEV
        into, by default ``[1, 6, 7, 8]`` (H, C, N, O).
    radial_cutoff : float, optional
        Cutoff radius for the radial part of the AEV, by default 5.2.
    angular_cutoff : float, optional
        Cutoff radius for the angular part of the AEV, by default 3.5.
    hidden : sequence of int or dict[int, sequence of int], optional
        Hidden-layer widths of each per-element MLP (a shared sequence or a
        per-``Z`` dict), by default ``(128, 128, 64)``.
    activation : str or torch.nn.Module, optional
        Hidden-layer activation, by default ``"celu"`` (ANI convention).
    atomic_energies : sequence of float or None, optional
        Per-species self atomic energy added to each atom's contribution
        (aligned with ``species``); ``None`` (default) adds nothing.
    aev_kwargs : dict, optional
        Extra keyword arguments forwarded to :class:`AEV` (symmetry-function
        grids, ``radial_prefactor``, ``angular_cos_factor``); by default
        ``None``.
    """

    def __init__(self, species: Sequence[int] = ANI_SPECIES,
                 radial_cutoff: float = 5.2, angular_cutoff: float = 3.5,
                 hidden: Union[Sequence[int], dict] = (128, 128, 64),
                 activation: Union[str, object] = "celu",
                 atomic_energies: Optional[Sequence[float]] = None,
                 aev_kwargs: Optional[dict] = None):
        aev = AEV(species, radial_cutoff=radial_cutoff,
                  angular_cutoff=angular_cutoff, **(aev_kwargs or {}))
        super().__init__(aev, species, hidden, activation=activation,
                         atomic_energies=atomic_energies)

    @classmethod
    def ani1(cls, species: Sequence[int] = ANI_SPECIES,
             activation: Union[str, object] = "gaussian",
             atomic_energies: Optional[Sequence[float]] = None) -> "ANI":
        """Build the original ANI-1 potential (Smith et al. 2017).

        768-length AEV (radial cutoff 4.6 A, angular cutoff 3.1 A) with the
        paper's pyramidal ``768:128:128:64:1`` element networks and a Gaussian
        hidden activation.

        Parameters
        ----------
        species : sequence of int, optional
            Atomic numbers, by default ``[1, 6, 7, 8]``.
        activation : str or torch.nn.Module, optional
            Hidden activation, by default ``"gaussian"`` (the paper's choice).
        atomic_energies : sequence of float or None, optional
            Per-species self energies aligned with ``species``.

        Returns
        -------
        ANI
            The ANI-1 model.
        """
        model = cls(species, radial_cutoff=4.6, angular_cutoff=3.1,
                    hidden=(128, 128, 64), activation=activation,
                    atomic_energies=atomic_energies,
                    aev_kwargs=dict(
                        radial_etas=(16.0,), radial_rs=_even(4.6, 32),
                        angular_etas=(8.0,), angular_zetas=(8.0,),
                        angular_rs=_even(3.1, 8), angular_theta_s=_ang(8)))
        return model

    @classmethod
    def ani1x(cls, species: Sequence[int] = ANI_SPECIES,
              atomic_energies: Optional[Sequence[float]] = "torchani") -> "ANI":
        """Build the ANI-1x architecture matching ``torchani``.

        384-length AEV (radial cutoff 5.2 A, angular cutoff 3.5 A) with
        torchani's per-element network widths and the ``CELU`` activation. With
        ``torchani``'s pretrained weights transplanted this reproduces its
        energies and forces.

        Parameters
        ----------
        species : sequence of int, optional
            Atomic numbers, by default ``[1, 6, 7, 8]``.
        atomic_energies : sequence of float, str or None, optional
            Per-species self energies. ``"torchani"`` (default) uses torchani's
            ANI-1x self energies in Hartree; a sequence sets them explicitly;
            ``None`` adds nothing.

        Returns
        -------
        ANI
            The ANI-1x model.
        """
        if atomic_energies == "torchani":
            atomic_energies = [ANI1X_SELF_ENERGIES[z] for z in species]
        hidden = {z: ANI1X_HIDDEN[z] for z in species}
        return cls(species, radial_cutoff=5.2, angular_cutoff=3.5,
                   hidden=hidden, activation="celu",
                   atomic_energies=atomic_energies,
                   aev_kwargs=dict(
                       radial_etas=(16.0,), radial_rs=_even(5.2, 16),
                       angular_etas=(8.0,), angular_zetas=(32.0,),
                       angular_rs=_even(3.5, 4), angular_theta_s=_ang(8)))

    @classmethod
    def ani1ccx(cls, species: Sequence[int] = ANI_SPECIES,
                atomic_energies: Optional[Sequence[float]] = "torchani",
                ) -> "ANI":
        """Build the ANI-1ccx potential (Smith et al. 2019, transfer learning).

        Architecturally identical to :meth:`ani1x` (384-length AEV, 5.2/3.5 A
        cutoffs, torchani per-element widths, ``CELU``); the published model was
        retrained by transfer learning on the CCSD(T)*/CBS energies of the
        ANI-1ccx data set (holding 65,280 of the 325,248 network weights fixed
        -- the matrix joining each element network's first two hidden layers),
        so only the self atomic energies (and the trained weights) differ. With
        ``torchani``'s pretrained ANI-1ccx weights transplanted this reproduces
        its energies and forces.

        Parameters
        ----------
        species : sequence of int, optional
            Atomic numbers, by default ``[1, 6, 7, 8]``.
        atomic_energies : sequence of float, str or None, optional
            Per-species self energies. ``"torchani"`` (default) uses torchani's
            ANI-1ccx self energies in Hartree (the CCSD(T)*/CBS linear fit); a
            sequence sets them explicitly; ``None`` adds nothing.

        Returns
        -------
        ANI
            The ANI-1ccx model.
        """
        if atomic_energies == "torchani":
            atomic_energies = [ANI1CCX_SELF_ENERGIES[z] for z in species]
        return cls.ani1x(species, atomic_energies=atomic_energies)

    @classmethod
    def from_config(cls, cfg):
        """Build an :class:`ANI` from a configuration object.

        Recognises a ``preset`` key (``"ani-1"`` / ``"ani-1x"`` /
        ``"ani-1ccx"``) in ``extra`` to
        select a published parameterisation; any other ``extra`` keys override
        the corresponding constructor argument. Upstream torchani / NeuroChem
        key spellings are translated to xnns names by the loader (see
        :mod:`xnns.common.config.translate`).

        Parameters
        ----------
        cfg : object
            Configuration exposing ``cutoff`` and an optional ``extra`` mapping
            (keys ``species``, ``preset``, ``radial_cutoff``,
            ``angular_cutoff``, ``hidden``, ``activation``, ``atomic_energies``,
            ``aev_kwargs``).

        Returns
        -------
        ANI
            Instantiated model.
        """
        extra = dict(cfg.extra or {})
        species = coerce_species(extra.get("species"), default=ANI_SPECIES)
        atomic_energies = coerce_per_species(
            extra.get("atomic_energies"), species, "atomic_energies")
        if atomic_energies is not None:
            atomic_energies = atomic_energies.tolist()

        preset = str(extra.get("preset", "")).lower().replace("_", "-")
        if preset in ("ani-1", "ani1"):
            model = cls.ani1(species, atomic_energies=atomic_energies)
        elif preset in ("ani-1x", "ani1x"):
            ae = atomic_energies if atomic_energies is not None else "torchani"
            model = cls.ani1x(species, atomic_energies=ae)
        elif preset in ("ani-1ccx", "ani1ccx"):
            ae = atomic_energies if atomic_energies is not None else "torchani"
            model = cls.ani1ccx(species, atomic_energies=ae)
        else:
            model = cls(
                species=species,
                radial_cutoff=extra.get("radial_cutoff", 5.2),
                angular_cutoff=extra.get("angular_cutoff", 3.5),
                hidden=extra.get("hidden", (128, 128, 64)),
                activation=extra.get("activation", "celu"),
                atomic_energies=atomic_energies,
                aev_kwargs=extra.get("aev_kwargs"),
            )
        return model
