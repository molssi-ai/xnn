"""Load pretrained MACE foundation models into the xnn :class:`MACE`.

The MACE foundation models -- the MACE-MP materials series (MP-0, MP-0b/0b2/
0b3, MPA-0, OMAT-0, MATPES, the multi-head MH series; Batatia *et al.*,
arXiv:2401.00096) and the MACE-OFF23 organic series (Kovacs *et al.*,
arXiv:2312.15211) -- are distributed as pickled ``mace-torch`` modules of the
upstream ``ScaleShiftMACE`` class. Since the xnn MACE reproduces upstream
block by block, those checkpoints convert weight-for-weight into
:class:`~xnn.gnn.models.mace.MACE`:

* the ``ScaleShiftMACE`` energy expression maps onto the model's
  ``scale_shift`` block,
* the Agnesi distance transform, ZBL pair repulsion and the
  density-normalized interaction blocks map onto the same-named xnn options,
* multi-head checkpoints are *sliced to one head*: the per-head rows of the
  atomic energies, readout weights and scale/shift are extracted, so the
  converted model is an ordinary single-head potential (the readout hidden
  channels of head ``h`` occupy one contiguous block, and only the final
  readout linear needs a ``sqrt(1/n_heads)`` fan-in renormalization).

Loading a checkpoint requires the ``mace-torch`` package (the pickle
references its classes); the conversion itself and the converted model do
not. Checkpoints are cached under the xnn dataset cache
(``<cache>/foundations/``); a file already in ``mace-torch``'s own cache
(``~/.cache/mace``) is reused instead of re-downloading.

Not covered: checkpoints trained with ``apply_cutoff=False`` radial
embeddings or interaction blocks outside the xnn registry (currently only
``mace-mh-1``, whose ``RealAgnosticResidualNonLinearInteractionBlock`` is a
different architecture generation) -- the converter raises
``NotImplementedError`` naming the offending piece.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional, Union

import torch
from torch import nn

from e3nn import o3

from xnn.common.data.hub._download import default_cache_dir, download_file

from .mace import MACE, GATES, INTERACTIONS

# Published foundation checkpoints: alias -> (download URL, license).
# URLs are the upstream release assets (github.com/ACEsuit); the materials
# series up to MPA-0 and the MH series are MIT-licensed, while OMAT-0,
# MATPES and MACE-OFF23 are distributed under the Academic Software License
# (ASL, https://github.com/gabor1/ASL): by downloading those you accept its
# terms (no commercial use).
_MP = "https://github.com/ACEsuit/mace-mp/releases/download"
_FOUND = "https://github.com/ACEsuit/mace-foundations/releases/download"
_OFF = "https://raw.githubusercontent.com/ACEsuit/mace-off/main/mace_off23"
FOUNDATION_MODELS = {
    "mace-mp-0-small": (f"{_MP}/mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model", "MIT"),
    "mace-mp-0-medium": (f"{_MP}/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model", "MIT"),
    "mace-mp-0-large": (f"{_MP}/mace_mp_0/MACE_MPtrj_2022.9.model", "MIT"),
    "mace-mp-0b-small": (f"{_MP}/mace_mp_0b/mace_agnesi_small.model", "MIT"),
    "mace-mp-0b-medium": (f"{_MP}/mace_mp_0b/mace_agnesi_medium.model", "MIT"),
    "mace-mp-0b2-small": (f"{_MP}/mace_mp_0b2/mace-small-density-agnesi-stress.model", "MIT"),
    "mace-mp-0b2-medium": (f"{_MP}/mace_mp_0b2/mace-medium-density-agnesi-stress.model", "MIT"),
    "mace-mp-0b2-large": (f"{_MP}/mace_mp_0b2/mace-large-density-agnesi-stress.model", "MIT"),
    "mace-mp-0b3-medium": (f"{_MP}/mace_mp_0b3/mace-mp-0b3-medium.model", "MIT"),
    "mace-mpa-0-medium": (f"{_MP}/mace_mpa_0/mace-mpa-0-medium.model", "MIT"),
    "mace-omat-0-small": (f"{_MP}/mace_omat_0/mace-omat-0-small.model", "ASL"),
    "mace-omat-0-medium": (f"{_MP}/mace_omat_0/mace-omat-0-medium.model", "ASL"),
    "mace-matpes-pbe-0-medium": (f"{_FOUND}/mace_matpes_0/MACE-matpes-pbe-omat-ft.model", "ASL"),
    "mace-matpes-r2scan-0-medium": (f"{_FOUND}/mace_matpes_0/MACE-matpes-r2scan-omat-ft.model", "ASL"),
    "mace-mh-0": (f"{_FOUND}/mace_mh_1/mace-mh-0.model", "MIT"),
    "mace-mh-1": (f"{_FOUND}/mace_mh_1/mace-mh-1.model", "MIT"),
    "mace-off23-small": (f"{_OFF}/MACE-OFF23_small.model", "ASL"),
    "mace-off23-medium": (f"{_OFF}/MACE-OFF23_medium.model", "ASL"),
    "mace-off23-large": (f"{_OFF}/MACE-OFF23_large.model", "ASL"),
}


def _checkpoint_path(source: Union[str, Path]) -> Path:
    """Resolve an alias, URL or local path to a cached checkpoint file.

    Parameters
    ----------
    source : str or Path
        A :data:`FOUNDATION_MODELS` alias, a checkpoint URL, or a local
        path.

    Returns
    -------
    pathlib.Path
        The local checkpoint file.

    Raises
    ------
    FileNotFoundError
        If ``source`` is neither a known alias, a URL, nor an existing file.
    """
    if isinstance(source, str) and source in FOUNDATION_MODELS:
        url, license_ = FOUNDATION_MODELS[source]
        if license_ == "ASL":
            print(f"{source} is distributed under the Academic Software "
                  "License (https://github.com/gabor1/ASL); by using it you "
                  "accept its terms (no commercial use).")
    elif isinstance(source, str) and source.startswith(("http://", "https://")):
        url = source
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(
                f"{source!r} is not a known foundation alias "
                f"({sorted(FOUNDATION_MODELS)}), a URL, or an existing file")
        return path
    fname = "".join(c for c in os.path.basename(url) if c.isalnum() or c in "._-")
    # reuse a file already fetched by mace-torch's own downloader
    mace_cache = Path.home() / ".cache" / "mace" / \
        "".join(c for c in os.path.basename(url) if c.isalnum() or c in "_")
    if mace_cache.is_file():
        return mace_cache
    return download_file(url, default_cache_dir() / "foundations" / fname)


def load_foundation(source: Union[str, Path]) -> nn.Module:
    """Load an upstream foundation checkpoint as a ``mace-torch`` module.

    Parameters
    ----------
    source : str or Path
        Alias, URL, or local path (see :func:`_checkpoint_path`).

    Returns
    -------
    torch.nn.Module
        The unpickled upstream model, on CPU.

    Raises
    ------
    ImportError
        If ``mace-torch`` is not installed (its classes are needed to
        unpickle the checkpoint).
    """
    path = _checkpoint_path(source)
    try:
        import mace  # noqa: F401  (needed by the pickle)
    except ModuleNotFoundError as e:
        raise ImportError(
            "loading a MACE foundation checkpoint needs the mace-torch "
            'package to unpickle it: pip install "xnn[examples]" or '
            "pip install mace-torch") from e
    return torch.load(path, map_location="cpu", weights_only=False)


# ----------------------------------------------------------------------
# introspection helpers
# ----------------------------------------------------------------------
def _resolve_head(upstream: nn.Module, head: Optional[str]) -> tuple[int, int]:
    """Pick the head index of a (possibly multi-head) checkpoint.

    Parameters
    ----------
    upstream : torch.nn.Module
        The upstream model.
    head : str or None
        Requested head name; ``None`` is allowed only for single-head
        checkpoints.

    Returns
    -------
    tuple of int
        ``(head_index, num_heads)``.

    Raises
    ------
    ValueError
        If the head is unknown, or omitted for a multi-head checkpoint.
    """
    heads = [str(h) for h in getattr(upstream, "heads", ["Default"])]
    if head is None:
        if len(heads) == 1:
            return 0, 1
        raise ValueError(f"this checkpoint has {len(heads)} heads; pass "
                         f"head=<name> with one of {heads}")
    if str(head) not in heads:
        raise ValueError(f"unknown head {head!r}; this checkpoint has {heads}")
    return heads.index(str(head)), len(heads)


def _gate_name(fn) -> str:
    """Map an upstream gate callable back to its registry name.

    Parameters
    ----------
    fn : callable
        The activation inside the upstream non-linear readout (possibly the
        e3nn ``normalize2mom`` wrapper, whose raw callable sits in ``.f``).

    Returns
    -------
    str
        A key of the shared ``GATES`` registry.

    Raises
    ------
    NotImplementedError
        If the activation matches no registered gate.
    """
    raw = getattr(fn, "f", fn)
    for name, gate in GATES.items():
        if gate is raw or (getattr(raw, "__name__", "?") ==
                           getattr(gate, "__name__", "!")):
            return name
    raise NotImplementedError(f"unsupported readout gate {raw!r}")


def _scalar_weight(linear: o3.Linear, n_out: int) -> torch.Tensor:
    """View an e3nn linear's flat weight as ``(fan_in, n_out)`` scalars.

    Valid for linears whose only weighted path is scalars-to-scalars (true
    for every MACE readout linear: non-scalar inputs cannot connect to the
    ``0e`` output).

    Parameters
    ----------
    linear : e3nn.o3.Linear
        The linear layer.
    n_out : int
        Multiplicity of the scalar output irrep.

    Returns
    -------
    torch.Tensor
        The weight, viewed as ``(fan_in, n_out)``.
    """
    return linear.weight.detach().reshape(-1, n_out)


# ----------------------------------------------------------------------
# conversion
# ----------------------------------------------------------------------
def _copy_symmetric_contractions(xnn_sc, up_sc, correlation: int) -> None:
    """Transplant upstream symmetric-contraction weights and CG bases.

    Upstream stores the top correlation order as ``weights_max`` and the
    lower orders in *descending* order in ``weights``; xnn stores all orders
    ascending in one list. The checkpoint's ``U_matrix_{nu}`` coupling-basis
    buffers are transplanted too: the weights are expressed in the basis the
    model was *trained* with, and that basis is not identical across the
    e3nn versions the foundation models were built against (the higher-``l``
    ``nu = 3`` couplings of MACE-MP-0b2-large differ from a freshly
    generated basis by O(1)); xnn's persistent ``U`` buffers make the copy
    survive save/load.

    Parameters
    ----------
    xnn_sc, up_sc : torch.nn.Module
        The xnn and upstream ``SymmetricContraction`` modules.
    correlation : int
        The correlation order of the contraction.
    """
    with torch.no_grad():
        for xc, uc in zip(xnn_sc.contractions, up_sc.contractions):
            xc.weights[correlation - 1].copy_(uc.weights_max)
            for nu in range(1, correlation):
                xc.weights[nu - 1].copy_(uc.weights[correlation - 1 - nu])
            _copy_matching_buffers(xc, uc)   # the U_matrix_{nu} bases


def _copy_matching_buffers(xnn_mod: nn.Module, up_mod: nn.Module) -> None:
    """Copy every same-named, same-shaped buffer from ``up_mod``.

    Parameters
    ----------
    xnn_mod, up_mod : torch.nn.Module
        Destination and source modules.
    """
    up = dict(up_mod.named_buffers())
    with torch.no_grad():
        for name, buf in xnn_mod.named_buffers():
            if name in up and up[name].shape == buf.shape:
                buf.copy_(up[name].to(buf.dtype))


def from_mace_torch(upstream: nn.Module, head: Optional[str] = None,
                    dtype=None) -> MACE:
    """Convert a ``mace-torch`` model (plain or ScaleShift) to an xnn MACE.

    Reads every architecture hyper-parameter off the upstream module, builds
    the equivalent :class:`~xnn.gnn.models.mace.MACE`, and copies the
    weights; multi-head checkpoints are sliced to the requested ``head``.
    The converted model reproduces the upstream energies, forces and stress
    to numerical precision (see ``tests/test_mace.py`` and the foundation
    fidelity notebook).

    Parameters
    ----------
    upstream : torch.nn.Module
        A ``mace.modules.models.MACE`` or ``ScaleShiftMACE`` instance.
    head : str, optional
        Head to keep for multi-head checkpoints (see :func:`_resolve_head`).
    dtype : torch.dtype or str, optional
        Final dtype; ``None`` keeps the upstream parameters' dtype.

    Returns
    -------
    MACE
        The converted model.

    Raises
    ------
    NotImplementedError
        For upstream features outside the xnn MACE (un-enveloped radial
        embeddings, unknown interaction blocks / radial bases / transforms,
        readout layouts other than one readout per interaction).
    """
    head_idx, n_heads = _resolve_head(upstream, head)
    up_dtype = next(upstream.parameters()).dtype

    # --- read the architecture off the checkpoint -----------------------
    species = [int(z) for z in upstream.atomic_numbers]
    r_max = float(upstream.r_max)
    T = int(upstream.num_interactions)
    embed = upstream.radial_embedding
    if not getattr(embed, "apply_cutoff", True):
        raise NotImplementedError(
            "this checkpoint uses an un-enveloped radial embedding "
            "(apply_cutoff=False), which the xnn MACE does not implement")
    basis = type(embed.bessel_fn).__name__
    radial_type = {"BesselBasis": "bessel", "GaussianBasis": "gaussian"}.get(basis)
    if radial_type is None:
        raise NotImplementedError(f"unsupported radial basis {basis}")
    n_rbf = int(embed.bessel_fn.bessel_weights.numel())
    cutoff_p = int(float(embed.cutoff_fn.p))
    transform = getattr(embed, "distance_transform", None)
    transform_name = {type(None): "None", }.get(type(transform)) or \
        {"AgnesiTransform": "Agnesi", "SoftTransform": "Soft"}.get(
            type(transform).__name__)
    if transform_name is None:
        raise NotImplementedError(
            f"unsupported distance transform {type(transform).__name__}")

    inter_names = [type(b).__name__ for b in upstream.interactions]
    for name in inter_names:
        if name not in INTERACTIONS:
            raise NotImplementedError(
                f"interaction block {name} is not implemented in the xnn "
                "MACE (this is a newer-generation checkpoint, e.g. mace-mh-1)")
    if T >= 2 and len(set(inter_names[1:])) > 1:
        raise NotImplementedError(
            f"mixed interaction blocks after the first layer: {inter_names}")
    first = upstream.interactions[0]
    node_feats_irreps = o3.Irreps(str(first.node_feats_irreps))
    if o3.Irreps(str(getattr(first, "edge_irreps", node_feats_irreps))) \
            != node_feats_irreps:
        raise NotImplementedError(
            "this checkpoint uses widened edge irreps (edge_irreps != "
            "node_feats_irreps), which the xnn MACE does not implement")
    hidden_irreps = str(o3.Irreps(str(first.hidden_irreps)))
    max_ell = o3.Irreps(str(first.edge_attrs_irreps)).lmax
    radial_MLP = list(first.conv_tp_weights.hs[1:-1])
    correlation = [
        int(p.symmetric_contractions.contractions[0].correlation)
        for p in upstream.products]

    readouts = list(upstream.readouts)
    if len(readouts) != max(T, 1):
        raise NotImplementedError(
            f"{len(readouts)} readouts for {T} interactions is not the "
            "one-readout-per-layer layout the xnn MACE implements")
    last = readouts[-1]
    if not hasattr(last, "linear_1"):
        raise NotImplementedError("the final readout is not the gated "
                                  "non-linear readout")
    up_mlp = o3.Irreps(str(last.hidden_irreps))
    if len(up_mlp) != 1 or up_mlp[0].ir != o3.Irrep("0e") \
            or up_mlp[0].mul % n_heads:
        raise NotImplementedError(f"unsupported readout MLP irreps {up_mlp}")
    mlp_mul = up_mlp[0].mul // n_heads
    gate = _gate_name(last.non_linearity.acts[0])

    scale, shift = 1.0, 0.0
    if hasattr(upstream, "scale_shift"):
        scale = float(torch.atleast_1d(upstream.scale_shift.scale)[
            head_idx if upstream.scale_shift.scale.ndim else 0])
        shift = float(torch.atleast_1d(upstream.scale_shift.shift)[
            head_idx if upstream.scale_shift.shift.ndim else 0])
    pair_repulsion = hasattr(upstream, "pair_repulsion")
    e0s = torch.atleast_2d(
        upstream.atomic_energies_fn.atomic_energies.detach().cpu())[head_idx]

    # --- build the xnn twin under the checkpoint's dtype -----------------
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(up_dtype)
    try:
        model = MACE(
            species=species, cutoff=r_max, max_ell=max_ell,
            n_rbf=n_rbf, num_interactions=T, correlation=correlation,
            MLP_irreps=f"{mlp_mul}x0e", radial_MLP=radial_MLP,
            interaction=inter_names[-1], interaction_first=inter_names[0],
            gate=gate, avg_num_neighbors=float(first.avg_num_neighbors),
            hidden_irreps=hidden_irreps, num_cutoff_basis=cutoff_p,
            radial_type=radial_type, distance_transform=transform_name,
            pair_repulsion=pair_repulsion, atomic_energies=e0s,
            scale=scale, shift=shift,
        )
    finally:
        torch.set_default_dtype(prev_dtype)

    # --- transplant the weights -----------------------------------------
    with torch.no_grad():
        model.node_embedding.load_state_dict(
            upstream.node_embedding.linear.state_dict())
        for i in range(T):
            model.interactions[i].load_state_dict(
                upstream.interactions[i].state_dict())
            model.interactions[i].avg_num_neighbors = float(
                upstream.interactions[i].avg_num_neighbors)
            _copy_symmetric_contractions(
                model.products[i].symmetric_contractions,
                upstream.products[i].symmetric_contractions, correlation[i])
            model.products[i].linear.load_state_dict(
                upstream.products[i].linear.state_dict())
        for i, up_readout in enumerate(readouts[:-1]):
            model.readouts[i].linear.weight.copy_(
                _scalar_weight(up_readout.linear, n_heads)[:, head_idx])
        model.readouts[-1].linear_1.weight.copy_(
            _scalar_weight(last.linear_1, up_mlp[0].mul)
            [:, head_idx * mlp_mul:(head_idx + 1) * mlp_mul].reshape(-1))
        # the sliced final linear sees fan-in mlp_mul instead of the full
        # hidden width; e3nn normalizes by 1/sqrt(fan_in), hence the factor
        model.readouts[-1].linear_2.weight.copy_(
            _scalar_weight(last.linear_2, n_heads)
            [head_idx * mlp_mul:(head_idx + 1) * mlp_mul, head_idx]
            * math.sqrt(1.0 / n_heads))
        if transform is not None:
            _copy_matching_buffers(model.distance_transform, transform)
        if pair_repulsion:
            _copy_matching_buffers(model.pair_repulsion_fn,
                                   upstream.pair_repulsion_fn)
        # the Bessel frequencies and prefactor as stored (a float32
        # checkpoint quantizes them; regenerating exactly would leave ~1e-7
        # residuals). Upstream stores n*pi/r_max, xnn n*pi with the division
        # at evaluation time.
        if radial_type == "bessel":
            model.edge_feat.rbf.freqs.copy_(
                embed.bessel_fn.bessel_weights.detach().reshape(-1)
                .to(model.edge_feat.rbf.freqs.dtype) * r_max)
            model.edge_feat.rbf.norm = float(embed.bessel_fn.prefactor)

    if dtype is not None:
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        model = model.to(dtype)
    return model


def foundation_to_xnn(source, head: Optional[str] = None, dtype=None) -> MACE:
    """Resolve, load and convert a foundation checkpoint (or model).

    The one-call form behind :meth:`MACE.from_foundation`.

    Parameters
    ----------
    source : str, Path or torch.nn.Module
        Alias / URL / local path of a checkpoint, or an already-loaded
        ``mace-torch`` model.
    head : str, optional
        Head to keep for multi-head checkpoints.
    dtype : torch.dtype or str, optional
        Final dtype; ``None`` keeps the checkpoint's.

    Returns
    -------
    MACE
        The converted model.
    """
    if isinstance(source, nn.Module):
        return from_mace_torch(source, head=head, dtype=dtype)
    return from_mace_torch(load_foundation(source), head=head, dtype=dtype)
