"""ANI-1ccx dataset builder (coupled-cluster subset of the ANI-1x release).

The ANI-1ccx data set holds ~500 k conformations for H/C/N/O organic molecules
with energies at an approximate **CCSD(T)/CBS** level: the CCSD(T)*/CBS
composite extrapolation (DLPNO-CCSD(T) + MP2/HF basis-set extrapolation) of
Smith *et al.* It is an intelligently selected ~10 % sub-sample of the ANI-1x
set, recomputed with coupled cluster and used to train the ANI-1ccx *model*
(:meth:`xnns.dnn.models.ani.ANI.ani1ccx`) by transfer learning. Upstream ships
it inside the same ``ani1x-release.h5`` file as ANI-1x -- an ANI-1ccx
conformation is simply one whose ``ccsd(t)_cbs.energy`` is not NaN -- so this
builder is a thin restriction of :class:`~xnns.common.data.hub.ani1x.ANI1xBuilder`
to that level, sharing its cached download. CCSD(T)*/CBS is **energy-only**
(no coupled-cluster forces were computed).

Reference
---------
Smith *et al.*, "Approaching coupled cluster accuracy with a general-purpose
neural network potential through transfer learning", *Nat. Commun.* **10**,
2903 (2019) (the ANI-1ccx model), and Smith *et al.*, "The ANI-1ccx and ANI-1x
data sets, coupled-cluster and density functional theory properties for
molecules", *Sci. Data* **7**, 134 (2020) (the release used here).
Data: https://doi.org/10.6084/m9.figshare.10047041
Format / reader spec: https://github.com/aiqm/ANI1x_datasets

Notes
-----
* Energies are in **Hartree** upstream; by default they are converted to eV
  (``units="hartree"`` keeps the raw values). Positions are in angstrom.
* The DFT properties of these same conformations (e.g. for the transfer-learning
  pre-training stage) are available through ``load_dataset("ani1x", ...)``.
* Splits, caps, and caching behave exactly as for ``ani1x``; the one ~5.6 GB
  release file is shared with the ``ani1x`` cache directory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from .ani1x import ANI1xBuilder
from .base import register_dataset


class ANI1ccxBuilder(ANI1xBuilder):
    """Builder for the ANI-1ccx data set (CCSD(T)*/CBS energies, no forces).

    See the module docstring for the dataset description and citation. All
    download, NaN-masking, capping, and splitting machinery is inherited from
    :class:`~xnns.common.data.hub.ani1x.ANI1xBuilder`; this class only pins the
    level of theory to ``"ccsd(t)_cbs"`` and shares the cached release file.
    """

    name = "ani1ccx"
    description = ("ANI-1ccx: ~500k conformations with CCSD(T)*/CBS energies "
                   "for H/C/N/O molecules (coupled-cluster subset of ANI-1x).")

    def load(self, *, split: Optional[str] = None, cache_dir: Path,
             units: str = "eV", max_molecules: Optional[int] = None,
             max_conformations: Optional[int] = None, seed: int = 1234,
             quiet: bool = False) -> Union[dict[str, list[dict]], list[dict]]:
        """Download and preprocess the ANI-1ccx data set.

        Parameters
        ----------
        split : str or None
            ``None`` returns ``{"all": ...}``; ``"train"`` / ``"val"`` /
            ``"test"`` returns a per-conformation 80/10/10 partition (fixed
            ``seed``, disjoint splits).
        cache_dir : pathlib.Path
            Base cache directory; the release file is shared with
            ``cache_dir/"ani1x"``.
        units : str, optional
            ``"eV"`` (default) converts energies to eV; ``"hartree"`` keeps the
            raw upstream values.
        max_molecules : int, optional
            Cap the number of molecule groups read (useful for demos).
        max_conformations : int, optional
            Cap the number of coupled-cluster conformations kept per molecule.
        seed : int, optional
            Seed for the reproducible 80/10/10 split. Defaults to ``1234``.
        quiet : bool, optional
            Suppress progress output. Defaults to ``False``.

        Returns
        -------
        dict of {str: list of dict} or list of dict
            Structure dicts with keys ``pos`` ``(N, 3)``, ``atomic_numbers``
            ``(N,)`` and ``energy`` (scalar). CCSD(T)*/CBS carries no forces.

        Raises
        ------
        ValueError
            If ``units`` or ``split`` is unrecognized.
        ImportError
            If ``h5py`` is not installed.
        """
        return super().load(
            split=split, cache_dir=cache_dir, level="ccsd(t)_cbs",
            forces=False, units=units, max_molecules=max_molecules,
            max_conformations=max_conformations, seed=seed, quiet=quiet)

    def _ensure_file(self, root: Path, quiet: bool) -> Path:
        """Reuse the ``ani1x`` cache: both sets live in one release file."""
        return super()._ensure_file(root.parent / "ani1x", quiet)


register_dataset(ANI1ccxBuilder())
