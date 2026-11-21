"""The pretrained CGCNN models of the authors, converted weight for weight.

The CGCNN repository (MIT license) publishes eight models trained on Materials
Project data: seven regressions (formation energy and absolute energy per
atom, band gap, Fermi energy, the base-10 logarithms of the bulk and shear moduli and
the Poisson ratio) and a metal / semiconductor classifier. Each checkpoint
holds the network's state dict, the target normalization (a mean and a
standard deviation) and the training arguments. The network maps onto
:class:`~xnn.gnn.models.cgcnn.CGCNN` layer for layer and the normalization
becomes ``property_shift`` and ``property_scale``.

The element feature table of the repository (``atom_init.json``, 92 columns
for atomic numbers 1 to 100) is the default input of every CGCNN. It is
downloaded once into the model cache, like the checkpoints, with a note of
the license beside them.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Union

import torch

from .cgcnn import CGCNN, CGCNN_TABLE_SHAPE, read_atom_features

_REVISION = "f42ab233c4ee0c416879d6bc2d22a264418413ad"
_URL = "https://raw.githubusercontent.com/txie-93/cgcnn/{revision}/{path}"
_REPOSITORY = "https://github.com/txie-93/cgcnn"
_LICENSE_URL = f"{_REPOSITORY}/blob/{_REVISION}/LICENSE"

#: The element feature table: path in the repository and MD5.
ATOM_INIT = ("data/sample-regression/atom_init.json", "29a97daf8b25c28d137a8ebff6c2f812")

#: The pretrained models: name -> (path in the repository, MD5, unit of the property).
FOUNDATION_MODELS = {
    "cgcnn-formation-energy-per-atom": ("pre-trained/formation-energy-per-atom.pth.tar",
                                        "020f8903b6c057aef925ba6ea71dd508", "eV/atom"),
    "cgcnn-final-energy-per-atom": ("pre-trained/final-energy-per-atom.pth.tar",
                                    "01747d6945a992f49e91d4030bee83a0", "eV/atom"),
    "cgcnn-band-gap": ("pre-trained/band-gap.pth.tar", "086221b79a7a24d39151ff93cd1841ad", "eV"),
    "cgcnn-efermi": ("pre-trained/efermi.pth.tar", "cc798c1e1858382a82677ef702f1d5be", "eV"),
    "cgcnn-bulk-moduli": ("pre-trained/bulk-moduli.pth.tar", "b0044f8177075cf52c04812e8acf2a09", "log10(GPa)"),
    "cgcnn-shear-moduli": ("pre-trained/shear-moduli.pth.tar", "f18234e82d1b6edb3abb529b4e478d23", "log10(GPa)"),
    "cgcnn-poisson-ratio": ("pre-trained/poisson-ratio.pth.tar", "7307147501cd5d13bb8a2e9b45810bf2", "1"),
    "cgcnn-semi-metal-classification": ("pre-trained/semi-metal-classification.pth.tar",
                                        "6b5483cc0f7269ad728dee7328638149", "probability of a metal"),
}


def _fetch(path: str, md5: str, cache_dir, quiet: bool) -> Path:
    """Download one file of the pinned revision into the cache (once), with the license note beside it."""
    from xnn.common.data.hub._download import download_file
    from xnn.common.models.hub.cache import resolve_cache_dir
    from xnn.common.models.hub.provenance import write_license_note
    folder = resolve_cache_dir(cache_dir) / "cgcnn" / _REVISION
    dest = download_file(_URL.format(revision=_REVISION, path=path), folder / Path(path).name, md5=md5,
                         quiet=quiet)
    write_license_note(folder, "CGCNN", _REVISION[:7], _REPOSITORY, "the MIT License", [_LICENSE_URL])
    return dest


def atom_init_table(cache_dir: Optional[Union[str, Path]] = None, quiet: bool = True) -> torch.Tensor:
    """The authors' element feature table, ``(101, 92)`` in float64 (row ``Z`` for atomic number ``Z``).

    Parameters
    ----------
    cache_dir : str or pathlib.Path, optional
        Model cache directory (the hub's default when ``None``).
    quiet : bool, optional
        Hide the download progress bar. Default ``True``.

    Returns
    -------
    torch.Tensor
        The table; row 0 is zeros.
    """
    table = read_atom_features(_fetch(*ATOM_INIT, cache_dir, quiet))
    if tuple(table.shape) != CGCNN_TABLE_SHAPE:
        raise ValueError(f"the element table has shape {tuple(table.shape)}, expected {CGCNN_TABLE_SHAPE}")
    return table


class _Unpickler(pickle.Unpickler):
    """Reads a checkpoint as plain data: tensors, containers and the numpy scalars some of them hold."""

    def find_class(self, module, name):
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name == "scalar":
            import numpy as np
            return (getattr(np, "_core", None) or np.core).multiarray.scalar
        if (module, name) == ("numpy", "dtype"):
            import numpy as np
            return np.dtype
        if ((module == "torch._utils" and name.startswith("_rebuild_tensor"))
                or (module == "torch" and name.endswith("Storage"))
                or (module, name) in (("collections", "OrderedDict"), ("_codecs", "encode"))):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"the checkpoint refers to {module}.{name}, which is not read")


class _PickleModule:
    """The ``pickle_module`` of :func:`torch.load`: every read goes through :class:`_Unpickler`."""

    Unpickler = _Unpickler

    @staticmethod
    def load(file, **kwargs):
        return _Unpickler(file, **kwargs).load()


def _load(path) -> dict:
    """The contents of a checkpoint, read through :class:`_Unpickler`."""
    return torch.load(path, map_location="cpu", weights_only=False, pickle_module=_PickleModule)


def convert_checkpoint(path: Union[str, Path], dtype=None, atom_features="cgcnn") -> CGCNN:
    """A :class:`CGCNN` with the weights of one of the authors' checkpoints (``*.pth.tar``).

    Parameters
    ----------
    path : str or pathlib.Path
        The checkpoint written by the authors' training script.
    dtype : torch.dtype, optional
        Final dtype; ``None`` keeps float32.
    atom_features : optional
        The element features the model was trained with (see
        :func:`~xnn.gnn.models.cgcnn.read_atom_features`); by default the
        authors' table.

    Returns
    -------
    CGCNN
        The converted model, in evaluation mode.
    """
    checkpoint = _load(path)
    args, state = checkpoint["args"], checkpoint["state_dict"]
    f = int(args.get("atom_fea_len", 64))
    n_rbf = int(state["convs.0.fc_full.weight"].shape[1]) - 2 * f
    classification = args.get("task", "regression") == "classification"
    norm = checkpoint.get("normalizer", {"mean": 0.0, "std": 1.0})
    table = atom_init_table() if isinstance(atom_features, str) and atom_features == "cgcnn" else atom_features
    n_conv = sum(1 for k in state if k.startswith("convs.") and k.endswith(".fc_full.weight"))
    model = CGCNN(n_features=f, n_interactions=n_conv, n_rbf=n_rbf,
                  readout_sizes=[int(args.get("h_fea_len", 128))] * int(args.get("n_h", 1)),
                  classification=classification, atom_features=table,
                  property_scale=1.0 if classification else float(norm["std"]),
                  property_shift=0.0 if classification else float(norm["mean"]))
    missing, unexpected = model.load_state_dict(state, strict=False)
    own = {"atom_features", "rbf.centers", "property_scale", "property_shift"}
    if unexpected or set(missing) - own:
        raise ValueError(f"{path}: the checkpoint does not map onto CGCNN (missing {sorted(set(missing) - own)}, "
                         f"unexpected {sorted(unexpected)})")
    if dtype is not None:
        model = model.to(getattr(torch, dtype) if isinstance(dtype, str) else dtype)
    if not classification:
        # the normalization in the full precision of the model, not rounded through the default dtype
        model.set_property_scale_shift(float(norm["std"]), float(norm["mean"]))
    return model.eval()


def from_foundation(name: str = "cgcnn-formation-energy-per-atom", dtype=None,
                    cache_dir: Optional[Union[str, Path]] = None, quiet: bool = False) -> CGCNN:
    """Download (once) and convert one of the authors' pretrained models.

    Parameters
    ----------
    name : str, optional
        A key of :data:`FOUNDATION_MODELS`, by default the formation energy model.
    dtype : torch.dtype or str, optional
        Final dtype; ``None`` keeps the checkpoint's float32.
    cache_dir : str or pathlib.Path, optional
        Model cache directory (the hub's default when ``None``).
    quiet : bool, optional
        Hide the download progress bars. Default ``False``.

    Returns
    -------
    CGCNN
        The converted model, in evaluation mode.
    """
    if name not in FOUNDATION_MODELS:
        raise KeyError(f"unknown CGCNN model {name!r}; known: {sorted(FOUNDATION_MODELS)}")
    path, md5, _ = FOUNDATION_MODELS[name]
    checkpoint = _fetch(path, md5, cache_dir, quiet)
    return convert_checkpoint(checkpoint, dtype, atom_init_table(cache_dir, quiet))
