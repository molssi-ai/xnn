"""Reaction-energy loss: errors of energy differences between structures.

A reaction energy is ``dE_r = sum_s nu_rs E_s`` over the species ``s`` with signed stoichiometric coefficients
``nu_rs``. :class:`ReactionSet` holds the species, the coefficients, the reference reaction energies and two
per-reaction factors: a ``scale`` that multiplies the error (``scale = 56.84 / <|dE|>`` of each set gives the
weighting of WTMAD-2) and a ``weight`` of the mean. :func:`reaction_energy_loss` is the weighted mean of the
squared (or Huber) scaled errors. :meth:`ReactionSet.batches` draws mini-batches of reactions with the species
they need, and :class:`~xnn.common.train.Trainer` adds the term to its loss (``optim.reaction_weight``).
"""
from __future__ import annotations

from typing import Iterator, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from ..data import AtomicGraph, collate, structure_to_graph
from ..data.hub.units import HARTREE_TO_EV, KCAL_MOL_TO_EV
from .losses import _residual_sq

_UNITS = {"ev": 1.0, "kcal/mol": KCAL_MOL_TO_EV, "hartree": HARTREE_TO_EV}


def reaction_energies(species_energy: Tensor, stoichiometry: Tensor) -> Tensor:
    """Reaction energies ``nu E`` ``(R,)`` from species energies ``(S,)`` and coefficients ``(R, S)``."""
    return stoichiometry.to(species_energy.dtype) @ species_energy


def reaction_energy_loss(species_energy: Tensor, stoichiometry: Tensor, reference: Tensor,
                         weight: Optional[Tensor] = None, scale: Optional[Tensor] = None,
                         huber_delta: float = 0.0) -> Tensor:
    """The weighted mean of the squared errors of reaction energies.

    Parameters
    ----------
    species_energy : Tensor
        Predicted energies of the species ``(S,)``.
    stoichiometry : Tensor
        Signed coefficients ``(R, S)`` of the species in each reaction.
    reference : Tensor
        Reference reaction energies ``(R,)`` in the units of ``species_energy``.
    weight : Tensor, optional
        Weights of the reactions in the mean ``(R,)``; default equal.
    scale : Tensor, optional
        Factors of the errors before squaring ``(R,)``; default 1.
    huber_delta : float, optional
        ``> 0`` makes the residual a Huber function of the scaled error with
        this threshold (as in :func:`~xnn.common.train.weighted_loss`).

    Returns
    -------
    Tensor
        The loss (a scalar), differentiable in ``species_energy``.
    """
    err = reaction_energies(species_energy, stoichiometry) - reference.to(species_energy.dtype)
    if scale is not None:
        err = err * scale.to(err.dtype)
    sq = _residual_sq(err, huber_delta)
    if weight is None:
        return sq.mean()
    w = weight.to(sq.dtype)
    return (w * sq).sum() / w.sum()


class ReactionSet:
    """Reactions over a list of species structures.

    Parameters
    ----------
    structures : sequence of dict
        The species, structure dicts as :func:`~xnn.common.data.structure_to_graph` takes them, each with a
        ``"name"`` (and optionally a ``"subset"``, which together key the species).
    reactions : sequence of dict
        One record per reaction: ``"species"`` (names), ``"stoichiometry"`` (signed coefficients),
        ``"reference"`` (the reaction energy), optionally ``"subset"`` (the species are looked up under it),
        ``"weight"`` and ``"scale"`` (see :func:`reaction_energy_loss`).
    units : str, optional
        Units of the references: ``"eV"`` (default), ``"kcal/mol"`` or ``"hartree"``; they are kept in eV.

    Attributes
    ----------
    structures : list of dict
        The species.
    members : list of tuple
        The species indices and coefficients of every reaction.
    reference, weight, scale : Tensor
        Per-reaction references (eV), weights and error scales ``(R,)``.
    labels : list
        ``(subset, index)`` of every reaction (``index`` its position when the records carry none).
    """

    def __init__(self, structures: Sequence[Mapping], reactions: Sequence[Mapping], units: str = "eV"):
        if units.lower() not in _UNITS:
            raise ValueError(f"units must be one of 'eV', 'kcal/mol', 'hartree', got {units!r}")
        factor = _UNITS[units.lower()]
        self.structures = list(structures)
        where = {(s.get("subset"), s["name"]): k for k, s in enumerate(self.structures)}
        self.members: list[Tuple[list[int], list[float]]] = []
        reference, weight, scale, self.labels = [], [], [], []
        for k, r in enumerate(reactions):
            subset = r.get("subset")
            try:
                idx = [where[(subset, name)] if (subset, name) in where else where[(None, name)]
                       for name in r["species"]]
            except KeyError as exc:
                raise KeyError(f"reaction {k}: no structure for species {exc.args[0]!r}") from None
            if len(idx) != len(r["stoichiometry"]):
                raise ValueError(f"reaction {k}: {len(idx)} species but {len(r['stoichiometry'])} coefficients")
            self.members.append((idx, [float(c) for c in r["stoichiometry"]]))
            reference.append(factor * float(r["reference"]))
            weight.append(float(r.get("weight", 1.0)))
            scale.append(float(r.get("scale", 1.0)))
            self.labels.append((subset, r.get("index", k)))
        self.reference = torch.tensor(reference, dtype=torch.float64)
        self.weight = torch.tensor(weight, dtype=torch.float64)
        self.scale = torch.tensor(scale, dtype=torch.float64)
        self._graphs: dict = {}

    @classmethod
    def from_gmtkn55(cls, data: Mapping, wtmad2: bool = False, scale: float = 56.84) -> "ReactionSet":
        """The reactions of ``load_dataset("gmtkn55")`` (references in kcal/mol).

        Parameters
        ----------
        data : dict
            ``{"structures": [...], "reactions": [...]}`` as the dataset returns them.
        wtmad2 : bool, optional
            Scale the errors of every set by ``scale / <|dE|>`` (its mean absolute reference energy), the
            weighting of WTMAD-2; default ``False``.
        scale : float, optional
            The constant of that weighting in kcal/mol; default 56.84 (the paper's).
        """
        records = [dict(r) for r in data["reactions"]]
        if wtmad2:
            sums: dict = {}
            for r in records:
                n, s = sums.get(r["subset"], (0, 0.0))
                sums[r["subset"]] = (n + 1, s + abs(float(r["reference"])))
            for r in records:
                n, s = sums[r["subset"]]
                r["scale"] = scale / (s / n)
        return cls(data["structures"], records, units="kcal/mol")

    def __len__(self) -> int:
        return len(self.members)

    def subset(self, reactions: Sequence[int]) -> "ReactionSet":
        """The given reactions with the species they need (in the order first met)."""
        out = ReactionSet.__new__(ReactionSet)
        keep: dict[int, int] = {}
        out.members = []
        for r in reactions:
            idx, coef = self.members[int(r)]
            out.members.append(([keep.setdefault(i, len(keep)) for i in idx], list(coef)))
        order = sorted(keep, key=keep.get)
        out.structures = [self.structures[i] for i in order]
        sel = torch.as_tensor([int(r) for r in reactions], dtype=torch.long)
        out.reference, out.weight, out.scale = self.reference[sel], self.weight[sel], self.scale[sel]
        out.labels = [self.labels[int(r)] for r in reactions]
        out._graphs = {key: [graphs[i] for i in order] for key, graphs in self._graphs.items()}
        return out

    def stoichiometry(self) -> Tensor:
        """The coefficients ``(R, S)`` (float64)."""
        out = torch.zeros(len(self.members), len(self.structures), dtype=torch.float64)
        for r, (idx, coef) in enumerate(self.members):
            for i, c in zip(idx, coef):
                out[r, i] += c
        return out

    def graphs(self, cutoff: float) -> list[AtomicGraph]:
        """The species as graphs with neighbor lists of radius ``cutoff`` (built once per radius)."""
        key = float(cutoff)
        if key not in self._graphs:
            self._graphs[key] = [structure_to_graph(dict(s), key) for s in self.structures]
        return self._graphs[key]

    def batches(self, size: int, shuffle: bool = True, seed: int = 0) -> Iterator["ReactionSet"]:
        """Mini-batches of ``size`` reactions covering the set once (shuffled by ``seed``), each a
        :meth:`subset` with the species it needs."""
        order = np.arange(len(self))
        if shuffle:
            order = np.random.default_rng(seed).permutation(len(self))
        for start in range(0, len(order), size):
            yield self.subset(order[start:start + size].tolist())

    def species_energies(self, model: torch.nn.Module, cutoff: float, batch_size: int = 32,
                         device: Optional[torch.device] = None) -> Tensor:
        """The energies ``(S,)`` of the species from ``model(graph)["energy"]``, in batches of structures
        (differentiable in the model's parameters when gradients are enabled)."""
        graphs = self.graphs(cutoff)
        out = []
        for start in range(0, len(graphs), batch_size):
            batch = collate(graphs[start:start + batch_size])
            if device is not None:
                batch = batch.to(device)
            out.append(model(batch)["energy"].reshape(-1))
        return torch.cat(out) if out else torch.zeros(0, dtype=torch.float64)

    def loss(self, species_energy: Tensor, huber_delta: float = 0.0) -> Tensor:
        """:func:`reaction_energy_loss` of these reactions for the species energies ``(S,)``."""
        dev = species_energy.device
        return reaction_energy_loss(species_energy, self.stoichiometry().to(dev), self.reference.to(dev),
                                    self.weight.to(dev), self.scale.to(dev), huber_delta)

    def errors(self, species_energy: Tensor) -> Tensor:
        """Predicted minus reference reaction energies ``(R,)`` in eV (unscaled)."""
        dev = species_energy.device
        return reaction_energies(species_energy, self.stoichiometry().to(dev)) - self.reference.to(dev)
