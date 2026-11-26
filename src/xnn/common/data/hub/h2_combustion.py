"""Hydrogen combustion data set builder.

About 290,000 energies and forces of the 19 reaction channels of hydrogen combustion at the
ωB97X-V/cc-pVTZ level: intrinsic reaction coordinates (``irc``), ab initio molecular dynamics around them
(``aimd``) and normal-mode displacements (``nm``). Structures hold two to six H and O atoms. The data were
made for NewtonNet (Haghighatlari et al., Digital Discovery 1, 333 (2022)), which trains on the 16 channels
that are not diatomic dissociations or recombinations.

Reference
---------
Guan, Das, Stein, Heidar-Zadeh, Bertels, Liu, Haghighatlari, Li, Zhang, Hao, Leven, Head-Gordon and
Head-Gordon, "Hydrogen Combustion using IRC, AIMD and normal modes", figshare (2022),
https://doi.org/10.6084/m9.figshare.19601689 (CC BY 4.0).

Notes
-----
* Upstream units are kcal/mol (energies, relative to the separated atoms) and kcal/mol/angstrom (forces);
  positions are in angstrom. By default these are converted to eV and eV/A (``units``).
* There are no official splits; NewtonNet's protocol draws random training, validation and test sets per
  reaction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from ._download import download_file, extract_archive
from .base import DatasetBuilder, register_dataset
from .units import KCAL_MOL_TO_EV as _KCAL_MOL_TO_EV

_URL = "https://ndownloader.figshare.com/files/36445941"
_MD5 = "1d3fb731deb1ddf245382a2e39356752"
_FOLDER = "H2COMBUSTION_DATA-main-2"

#: The 16 reaction channels NewtonNet trains on (the diatomic ones 05, 06a, 06b and 07 left out).
REACTIONS = ("01", "02", "03", "04", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19")
#: Every channel of the archive; 20 and 21 are molecular dynamics of two further channels.
ALL_REACTIONS = REACTIONS + ("05", "06a", "06b", "07", "20", "21")
SOURCES = ("irc", "aimd", "nm")


def _files(reaction: str, sources: Sequence[str]) -> list[str]:
    """The archive files of one reaction channel for the requested sources."""
    if reaction == "12":                # both branches (12a, 12b) of the channel
        stems = {"irc": ["12a_irc", "12b_irc"], "aimd": ["12_aimd"], "nm": ["12_nm"]}
    else:
        stems = {s: [f"{reaction}_{s}"] for s in SOURCES}
    return [f"{stem}.npz" for s in sources for stem in stems[s]]


class H2CombustionBuilder(DatasetBuilder):
    """Builder for the hydrogen combustion data set (see the module docstring)."""

    name = "h2_combustion"
    description = "Hydrogen combustion: 19 reaction channels, IRC, AIMD and normal modes at wB97X-V."

    def load(self, *, split: Optional[str] = None, cache_dir: Path, reactions: Optional[Sequence[str]] = None,
             sources: Sequence[str] = SOURCES, units: str = "eV",
             quiet: bool = False) -> Union[dict[str, list[dict]], list[dict]]:
        """Download and preprocess the reaction channels.

        Parameters
        ----------
        split : str or None
            ``None`` returns ``{reaction: structures}``; a reaction label (e.g. ``"03"``) returns that
            channel's structures.
        cache_dir : pathlib.Path
            Base cache directory; files are stored under ``cache_dir/"h2_combustion"``.
        reactions : sequence of str, optional
            Reaction labels to load; by default the 16 channels of :data:`REACTIONS`.
        sources : sequence of str, optional
            Any of ``"irc"``, ``"aimd"`` and ``"nm"`` (all by default).
        units : str, optional
            ``"eV"`` (default) or ``"kcal/mol"``.
        quiet : bool, optional
            Suppress download progress output.

        Returns
        -------
        dict of {str: list of dict} or list of dict
            Structure dicts with ``pos``, ``atomic_numbers``, ``energy`` and ``forces``.

        Raises
        ------
        ValueError
            For an unknown reaction, source or unit.
        """
        if units not in ("eV", "kcal/mol", "kcal"):
            raise ValueError(f"units must be 'eV' or 'kcal/mol', got {units!r}")
        bad = [s for s in sources if s not in SOURCES]
        if bad:
            raise ValueError(f"unknown sources {bad}; use {', '.join(SOURCES)}")
        chosen = list(REACTIONS if reactions is None else reactions)
        if split is not None:
            chosen = [split]
        bad = [r for r in chosen if r not in ALL_REACTIONS]
        if bad:
            raise ValueError(f"unknown reactions {bad}; use {', '.join(ALL_REACTIONS)}")
        raw = Path(cache_dir) / self.name / "raw"
        archive = download_file(_URL, raw / f"{_FOLDER}.zip", _MD5, quiet=quiet)
        if not (raw / "01_irc.npz").exists():
            extract_archive(archive, raw, select=lambda m: m.startswith(_FOLDER + "/") and m.endswith(".npz"),
                            strip=1)
        scale = 1.0 if units in ("kcal/mol", "kcal") else _KCAL_MOL_TO_EV
        out: dict[str, list[dict]] = {}
        for reaction in chosen:
            structures = []
            for name in _files(reaction, sources):
                if not (raw / name).exists():          # the diatomic channels have IRC data only
                    continue
                with np.load(raw / name) as data:      # every key access of an npz reads its array again
                    counts = data["N"].reshape(-1).astype(np.int64)
                    R, Z, E, F = data["R"], data["Z"].astype(np.int64), data["E"].reshape(-1), data["F"]
                for i, n in enumerate(counts):
                    structures.append({"pos": R[i, :n], "atomic_numbers": Z[i, :n], "energy": float(E[i]) * scale,
                                       "forces": F[i, :n] * scale})
            out[reaction] = structures
        return out if split is None else out[split]


register_dataset(H2CombustionBuilder())
