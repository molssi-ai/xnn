"""Model cards: the metadata that describes one pre-trained model.

A :class:`ModelCard` is what the registry lists, what :func:`list_models`
shows, and what a portable model directory carries in its ``card.json``. The
same record describes every kind of model the hub manages: an xnn-trained
checkpoint, a converted MACE foundation model, or a model published on
Zenodo. Only plain JSON types are stored, so a card reads the same on every
system the directory is copied to.
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

# the three files of a portable model directory
CARD_FILE = "card.json"
CONFIG_FILE = "config.yaml"
WEIGHTS_FILE = "model.pt"


@dataclass
class ModelCard:
    """Metadata of one pre-trained model.

    Attributes
    ----------
    name : str
        Registry key, or the cache key of a model fetched by URL or DOI
        (e.g. ``"zenodo.18957344/mace_csfapbbri_al_5_1_stagetwo.model"``).
    description : str
        One-line human-readable summary.
    format : str
        On-disk format of the published file: ``"xnn"`` for a portable model
        directory or an xnn trainer checkpoint, ``"mace-torch"`` for a pickled
        upstream MACE model. Converted formats are stored as ``"xnn"`` once
        they are cached.
    architecture : str, optional
        Name of the registered xnn model class (``"mace"``, ``"schnet"``, ...).
    cutoff : float, optional
        Neighbor-list cutoff in angstrom.
    species : list of int, optional
        Atomic numbers the model was trained on.
    heads : list of str, optional
        Heads of a multi-head checkpoint; one must be chosen when loading.
    head : str, optional
        In a cached directory, the head that was kept.
    dtype : str, optional
        Floating-point dtype of the stored weights (``"float64"``, ...).
    units : dict of str to str
        Energy and length units of the model's inputs and outputs.
    license : str, optional
        License identifier. ``"ASL"`` (the Academic Software License) prints
        its terms when the model is loaded.
    citation : str, optional
        Reference to cite when using the model.
    doi : str, optional
        Zenodo DOI the model is downloaded from.
    url : str, optional
        Direct download URL, used when there is no DOI.
    filename : str, optional
        File to take from a Zenodo record that holds several.
    md5 : str, optional
        MD5 digest of the file at ``url``. Zenodo supplies its own.
    tags : list of str
        Free-form keywords for filtering.
    aliases : list of str
        Other names the model loads under (``"aimnet2"`` for
        ``"aimnet2-wb97m-d3-0"``); they resolve to this card and share its
        cache directory.
    notes : str, optional
        Anything else a user should know.
    unsupported : str, optional
        Why the model cannot be loaded yet. Set for registry entries that are
        listed for completeness only.
    source : str, optional
        Where a cached directory came from (a URL, DOI or file name).
    files : dict of str to str
        MD5 digest of each file of a portable directory.
    xnn_version : str, optional
        xnn version that wrote the directory.
    """

    name: str
    description: str = ""
    format: str = "xnn"
    architecture: Optional[str] = None
    cutoff: Optional[float] = None
    species: Optional[list[int]] = None
    heads: Optional[list[str]] = None
    head: Optional[str] = None
    dtype: Optional[str] = None
    units: dict[str, str] = field(
        default_factory=lambda: {"energy": "eV", "length": "angstrom"})
    license: Optional[str] = None
    citation: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    filename: Optional[str] = None
    md5: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    notes: Optional[str] = None
    unsupported: Optional[str] = None
    source: Optional[str] = None
    files: dict[str, str] = field(default_factory=dict)
    xnn_version: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelCard":
        """Build a card from a mapping, ignoring keys this version does not know.

        Unknown keys are dropped rather than rejected, so a card written by a
        newer xnn still loads.

        Parameters
        ----------
        d : dict
            Card fields.

        Returns
        -------
        ModelCard
            The card.
        """
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        """Return the card as a JSON-ready dict, leaving out unset fields.

        Returns
        -------
        dict
            Fields whose value is not ``None`` or empty.
        """
        return {k: v for k, v in dataclasses.asdict(self).items()
                if v is not None and v != [] and v != {}}

    def save(self, path: Union[str, Path]) -> Path:
        """Write the card as JSON, atomically.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file (normally ``<model dir>/card.json``).

        Returns
        -------
        pathlib.Path
            ``path``.
        """
        path = Path(path)
        tmp = path.with_name(path.name + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ModelCard":
        """Read a card from a JSON file.

        Parameters
        ----------
        path : str or pathlib.Path
            A ``card.json`` file.

        Returns
        -------
        ModelCard
            The card.
        """
        return cls.from_dict(json.loads(Path(path).read_text()))

    @property
    def downloadable(self) -> bool:
        """Whether the card names a place to download the model from."""
        return bool(self.doi or self.url)
