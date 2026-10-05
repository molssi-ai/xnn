"""Replay data for multi-head fine-tuning: selection and pseudolabels.

Multi-head replay trains a second head on structures from the pretraining
distribution so the shared trunk keeps the foundation model's breadth
(Tompa *et al.*, arXiv:2606.12704, Sec. 2.5). The replay set is usually a
subsample of the pretraining data, element-matched to the fine-tuning set,
with its original labels; with *pseudolabels* (the pretrained model's own
predictions) any structurally diverse set serves, and the replay head is
asked to reproduce the pretrained model rather than its reference data.
"""
from __future__ import annotations

import copy
import logging
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import torch
from torch import nn

from ..data import collate, structure_to_graph

logger = logging.getLogger(__name__)

#: Element filters of :func:`select_replay`, relative to the fine-tuning
#: elements: keep every structure (``none``); only those made of fine-tuning
#: elements (``subset``, the default and the paper's protocol); exactly the
#: fine-tuning element set (``exact``); those containing all fine-tuning
#: elements (``superset``).
REPLAY_FILTERS = ("none", "subset", "exact", "superset")


def _elements(s: dict) -> set[int]:
    return {int(z) for z in np.asarray(s["atomic_numbers"]).flatten().tolist()}


def element_filter(structures: Iterable[dict], species: Iterable[int], mode: str = "subset") -> list[dict]:
    """Keep the structures whose elements relate to ``species`` as ``mode`` says.

    Parameters
    ----------
    structures : iterable of dict
        Candidate structures.
    species : iterable of int
        The fine-tuning elements.
    mode : str, optional
        One of :data:`REPLAY_FILTERS`, by default ``"subset"``.
    """
    if mode not in REPLAY_FILTERS:
        raise ValueError(f"unknown replay filter {mode!r}; use one of {REPLAY_FILTERS}")
    target = {int(z) for z in species}
    if mode == "none":
        return list(structures)
    keep = {"subset": lambda e: e <= target,
            "exact": lambda e: e == target,
            "superset": lambda e: target <= e}[mode]
    return [s for s in structures if keep(_elements(s))]


def select_replay(structures: Iterable[dict], species: Optional[Iterable[int]] = None,
                  n: Optional[int] = None, mode: str = "subset", seed: int = 0) -> list[dict]:
    """Assemble a replay set: element filter, then a random subsample.

    Parameters
    ----------
    structures : iterable of dict
        Candidate structures (the pretraining data, or any diverse set).
    species : iterable of int, optional
        The fine-tuning elements; ``None`` applies no element filter.
    n : int, optional
        Number of structures to keep (a seeded random choice); ``None`` keeps
        all that pass the filter.
    mode : str, optional
        The element filter (:data:`REPLAY_FILTERS`), by default ``"subset"``.
    seed : int, optional
        Seed of the subsample.

    Returns
    -------
    list of dict
        The selected structures (the original dicts).
    """
    structures = list(structures)
    n_in = len(structures)
    if species is not None:
        structures = element_filter(structures, species, mode)
    if n is not None and n < len(structures):
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(structures), size=int(n), replace=False))
        structures = [structures[i] for i in idx]
    logger.info("replay set: %d of %d structures kept (filter %s, n=%s)",
                len(structures), n_in, mode, n)
    return structures


@torch.no_grad()
def pseudolabel(model: nn.Module, structures: Iterable[dict], batch_size: int = 32,
                device: Optional[torch.device] = None, stress: Optional[bool] = None,
                head: Union[int, str, None] = None) -> list[dict]:
    """Label structures with a model's own energies, forces (and stress).

    Parameters
    ----------
    model : torch.nn.Module
        The pretrained potential: bare, wrapped in ``ForceStressOutput``, or
        a :class:`~xnn.common.finetune.MultiHead` (then ``head`` picks the
        head that labels; by default the first).
    structures : iterable of dict
        The structures to relabel; other keys (``head``, ``weight``, cell)
        are kept.
    batch_size : int, optional
        Structures per forward pass, by default 32.
    device : torch.device, optional
        Where to evaluate; by default where the model is.
    stress : bool, optional
        Whether to label the stress; by default when any structure has a
        cell.
    head : int or str, optional
        The head of a multi-head model that produces the labels.

    Returns
    -------
    list of dict
        Copies of the structures with ``energy``, ``forces`` (and ``stress``)
        replaced by the predictions.
    """
    from ..models.outputs import ForceStressOutput
    from .heads import MultiHead

    structures = list(structures)
    if not structures:
        return []
    if isinstance(model, ForceStressOutput):
        model = model.model
    if isinstance(model, MultiHead):
        with model.using(0 if head is None else head) as single:
            return pseudolabel(single, structures, batch_size, device, stress)
    if stress is None:
        stress = any(s.get("cell") is not None for s in structures)
    if device is None:
        device = next((p.device for p in model.parameters()), torch.device("cpu"))
    cutoff = float(getattr(model, "cutoff"))
    wrapped = ForceStressOutput(model, compute_forces=True, compute_stress=bool(stress))
    was_training = model.training
    model.eval()
    out: list[dict] = []
    try:
        for i0 in range(0, len(structures), batch_size):
            chunk = structures[i0:i0 + batch_size]
            graphs = [structure_to_graph({k: v for k, v in s.items()
                                          if k not in ("energy", "forces", "stress")}, cutoff)
                      for s in chunk]
            batch = collate(graphs).to(device)
            with torch.enable_grad():
                pred = wrapped(batch)
            energies = pred["energy"].detach().cpu().numpy()
            forces = pred["forces"].detach().cpu().numpy()
            stresses = pred["stress"].detach().cpu().numpy() if "stress" in pred else None
            offsets = np.concatenate([[0], np.cumsum(batch.n_atoms.cpu().numpy())])
            for j, s in enumerate(chunk):
                new = copy.copy(s)
                new["energy"] = float(energies[j])
                new["forces"] = forces[offsets[j]:offsets[j + 1]].copy()
                if stresses is not None and s.get("cell") is not None:
                    new["stress"] = stresses[j].copy()
                else:
                    new.pop("stress", None)
                out.append(new)
    finally:
        model.train(was_training)
    logger.info("pseudolabelled %d structures with %s", len(out), type(model).__name__)
    return out
