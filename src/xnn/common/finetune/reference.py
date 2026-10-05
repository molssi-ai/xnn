"""Per-element reference energies (E0s) for fine-tuning.

A potential predicts energies relative to per-element reference energies
(``atom_ref`` in most xnn models). A fine-tuning set computed at another
level of theory differs from the pretraining labels mostly by such
composition-dependent constants, so the references are re-set before the fit
(Tompa *et al.*, arXiv:2606.12704, Sec. 2.6). Two estimators are provided:

* :func:`average_atomic_energies`: the least-squares fit of the reference
  energies to the compositions (eq. 5), the minimum-norm solution when the
  compositions are degenerate;
* :func:`estimate_atomic_energies`: the model-aware reestimation (eq. 6),
  the per-element corrections that best align the *pretrained model's
  predictions* with the fine-tuning energies, which recovers the target
  references exactly for a model that predicts the interaction energies
  exactly and is the recommended choice.

:func:`get_atomic_energies` / :func:`set_atomic_energies` read and write the
references of any xnn model, whatever it calls them.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn

from ..config.coerce import coerce_species
from ..data import collate, structure_to_graph


def species_of(structures) -> list[int]:
    """The sorted atomic numbers present in ``structures`` (dicts or a dataset)."""
    structures = getattr(structures, "structures", structures)
    zs: set[int] = set()
    for s in structures:
        zs.update(int(z) for z in torch.as_tensor(s["atomic_numbers"]).flatten().tolist())
    return sorted(zs)


def composition_matrix(structures, species: Sequence[int]) -> Tensor:
    """Atom counts per structure and element, float64 ``(n_structures, n_species)``."""
    structures = getattr(structures, "structures", structures)
    index = {int(z): i for i, z in enumerate(species)}
    out = torch.zeros(len(structures), len(species), dtype=torch.float64)
    for i, s in enumerate(structures):
        for z in torch.as_tensor(s["atomic_numbers"]).flatten().tolist():
            if int(z) not in index:
                raise ValueError(f"structure {i} contains Z={int(z)}, not in species {list(species)}")
            out[i, index[int(z)]] += 1.0
    return out


def _min_norm_lstsq(a: Tensor, b: Tensor) -> Tensor:
    """Minimum-norm least-squares solution of ``a x = b`` (the ``lambda -> 0+`` limit)."""
    return torch.linalg.lstsq(a.cpu(), b.cpu()[:, None], driver="gelsd").solution[:, 0]


def _energies(structures) -> Tensor:
    structures = getattr(structures, "structures", structures)
    missing = [i for i, s in enumerate(structures) if s.get("energy") is None]
    if missing:
        raise ValueError(f"{len(missing)} structure(s) carry no energy (first: {missing[0]})")
    return torch.tensor([float(s["energy"]) for s in structures], dtype=torch.float64)


def average_atomic_energies(structures, species: Optional[Sequence[int]] = None) -> dict[int, float]:
    """Per-element reference energies fitted to the energies of ``structures``.

    The least-squares solution of ``E_i = sum_Z n_iZ E0_Z`` (Tompa *et al.*
    eq. 5); when the compositions do not determine every element the
    minimum-norm solution is taken.

    Parameters
    ----------
    structures : iterable of dict or AtomicDataset
        Labelled structures.
    species : sequence of int, optional
        The elements to fit; by default those present.

    Returns
    -------
    dict of int to float
        ``{Z: E0}``.
    """
    species = list(species) if species is not None else species_of(structures)
    e0 = _min_norm_lstsq(composition_matrix(structures, species), _energies(structures))
    return {int(z): float(v) for z, v in zip(species, e0)}


def _bare(model: nn.Module) -> nn.Module:
    """The potential under a ForceStressOutput wrapper, if any."""
    from ..models.outputs import ForceStressOutput
    return model.model if isinstance(model, ForceStressOutput) else model


@torch.no_grad()
def predict_energies(model: nn.Module, structures, batch_size: int = 32,
                     device: Optional[torch.device] = None) -> Tensor:
    """Energies ``model`` predicts for ``structures``, float64 ``(n,)``.

    Parameters
    ----------
    model : torch.nn.Module
        A potential (bare or wrapped in ``ForceStressOutput``); its
        ``cutoff`` builds the neighbor lists.
    structures : iterable of dict or AtomicDataset
        The structures.
    batch_size : int, optional
        Structures per forward pass, by default 32.
    device : torch.device, optional
        Where to evaluate; by default where the model's parameters are.
    """
    model = _bare(model)
    structures = list(getattr(structures, "structures", structures))
    if device is None:
        device = next((p.device for p in model.parameters()), torch.device("cpu"))
    cutoff = float(getattr(model, "cutoff"))
    was_training = model.training
    model.eval()
    out = []
    try:
        for i0 in range(0, len(structures), batch_size):
            graphs = [structure_to_graph({k: v for k, v in s.items()
                                          if k not in ("energy", "forces", "stress")}, cutoff)
                      for s in structures[i0:i0 + batch_size]]
            batch = collate(graphs).to(device)
            out.append(model(batch)["energy"].detach().to(torch.float64).cpu())
    finally:
        model.train(was_training)
    return torch.cat(out) if out else torch.zeros(0, dtype=torch.float64)


def estimate_atomic_energies(model: nn.Module, structures, species: Optional[Sequence[int]] = None,
                             batch_size: int = 32, device: Optional[torch.device] = None
                             ) -> dict[int, float]:
    """Model-aware reestimation of the per-element reference energies.

    Solves ``E_i^ref - E_i^model = sum_Z n_iZ dE0_Z`` for the corrections
    ``dE0`` in the least-squares (minimum-norm) sense (Tompa *et al.* eq. 6)
    and returns the model's current references plus the corrections. A model
    that already predicts the interaction energies correctly keeps them and
    only its baseline moves, which is what fine-tuning across levels of
    theory needs.

    Parameters
    ----------
    model : torch.nn.Module
        The pretrained potential (bare or wrapped). For a
        :class:`~xnn.common.finetune.MultiHead`, call this inside
        :meth:`~xnn.common.finetune.MultiHead.using` on ``model.model``.
    structures : iterable of dict or AtomicDataset
        The fine-tuning structures with energies.
    species : sequence of int, optional
        The elements to correct; by default those present.
    batch_size : int, optional
        Structures per forward pass, by default 32.
    device : torch.device, optional
        Where to evaluate.

    Returns
    -------
    dict of int to float
        ``{Z: E0}``, the new reference energies.
    """
    species = list(species) if species is not None else species_of(structures)
    residual = _energies(structures) - predict_energies(model, structures, batch_size, device)
    delta = _min_norm_lstsq(composition_matrix(structures, species), residual)
    current = get_atomic_energies(_bare(model), species)
    return {int(z): float(current.get(int(z), 0.0) + d) for z, d in zip(species, delta)}


def _reference_table(model: nn.Module) -> tuple[Tensor, str]:
    """The tensor holding the per-element references and how it is indexed."""
    model = _bare(model)
    atom_ref = getattr(model, "atom_ref", None)
    if isinstance(atom_ref, nn.Embedding):
        return atom_ref.weight, "embedding"           # (200, 1), row Z
    by_z = getattr(model, "_self_energies_by_z", None)
    if torch.is_tensor(by_z):
        return by_z, "vector"                          # (max_z + 1,), entry Z
    shift = getattr(model, "Eshift", None)
    if torch.is_tensor(shift):
        return shift, "vector"                         # PhysNet per-element shift
    raise TypeError(f"{type(model).__name__} has no per-element reference energies")


def get_atomic_energies(model: nn.Module, species: Optional[Sequence[int]] = None) -> dict[int, float]:
    """The per-element reference energies of a model, ``{Z: E0}``.

    Parameters
    ----------
    model : torch.nn.Module
        A potential with ``atom_ref`` (the GNNs, SchNet, AIMNet2), per-element
        self energies (HDNNP, ANI) or a per-element shift (PhysNet).
    species : sequence of int, optional
        Which elements to report; by default the model's ``species``, or every
        element with a nonzero reference when it lists none.
    """
    table, kind = _reference_table(model)
    values = table.detach().reshape(-1).to(torch.float64).cpu()
    if species is None:
        species = getattr(_bare(model), "species", None)
    if species is None:
        species = [int(z) for z in values.nonzero().flatten().tolist()]
    return {int(z): float(values[int(z)]) for z in species if int(z) < values.numel()}


@torch.no_grad()
def set_atomic_energies(model: nn.Module, values: Any) -> None:
    """Set the per-element reference energies of a model.

    Parameters
    ----------
    model : torch.nn.Module
        A potential (see :func:`get_atomic_energies` for the supported
        conventions).
    values : mapping or sequence
        ``{Z: E0}`` with atomic numbers or chemical symbols as keys, or one
        value per entry of the model's ``species``.

    Raises
    ------
    ValueError
        If a sequence does not match the model's species, or an element is
        outside the model's reference table.
    """
    table, kind = _reference_table(model)
    if isinstance(values, Mapping):
        zs = coerce_species(list(values.keys()))
        mapping = {z: float(v) for z, v in zip(zs, values.values())}
    else:
        species = getattr(_bare(model), "species", None)
        seq = list(values)
        if species is None or len(seq) != len(species):
            raise ValueError("a sequence of reference energies needs one value per model "
                             f"species ({species}), got {len(seq)} values")
        mapping = {int(z): float(v) for z, v in zip(species, seq)}
    for z, v in mapping.items():
        if z >= table.shape[0]:
            raise ValueError(f"Z={z} is outside the model's reference table ({table.shape[0]} entries)")
        if kind == "embedding":
            table[z, 0] = v
        else:
            table[z] = v


#: Values of ``atomic_energies`` in a config that ask for an estimate from the
#: training data: ``"estimated"`` (model-aware reestimation,
#: :func:`estimate_atomic_energies`) or ``"average"``
#: (:func:`average_atomic_energies`). The trainer resolves them before the
#: fit and records the numbers in the saved config.
REFERENCE_MARKERS = ("estimated", "average")


def reference_markers(model_cfg):
    """Split the reference-energy markers out of a model config.

    Parameters
    ----------
    model_cfg : ModelConfig
        The model section; ``extra["atomic_energies"]`` and the per-head
        ``extra["heads"][name]["atomic_energies"]`` entries may be markers.

    Returns
    -------
    tuple of (ModelConfig, list of tuple)
        A deep copy of the config with the markers removed, and the markers
        as ``(head_name or None, marker)`` pairs.
    """
    import copy
    cfg = copy.deepcopy(model_cfg)
    extra = dict(cfg.extra or {})
    found: list[tuple[Optional[str], str]] = []
    value = extra.get("atomic_energies")
    if isinstance(value, str) and value.lower() in REFERENCE_MARKERS:
        found.append((None, value.lower()))
        extra.pop("atomic_energies")
    heads = extra.get("heads")
    if isinstance(heads, dict):
        heads = {k: (dict(v) if isinstance(v, dict) else v) for k, v in heads.items()}
        for name, opts in heads.items():
            if isinstance(opts, dict):
                value = opts.get("atomic_energies")
                if isinstance(value, str) and value.lower() in REFERENCE_MARKERS:
                    found.append((str(name), value.lower()))
                    opts.pop("atomic_energies")
        extra["heads"] = heads
    cfg.extra = extra
    return cfg, found
