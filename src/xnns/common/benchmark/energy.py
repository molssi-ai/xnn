"""Per-element reference energies (E0s) for atomization-energy scoring.

The total energy of a structure is dominated by per-atom self-energies that
carry no information about how the atoms interact. Subtracting a per-element
reference energy ``E0[Z]`` for every atom turns it into the *atomization* (a.k.a.
interaction) energy -- the physically meaningful quantity to report:

    E_atomization = E_total - sum_i E0[Z_i]

This module turns a configured ``atomic_energies`` value into a lookup tensor
indexed by atomic number, reusing the same value parser every model uses
(:func:`~xnns.common.config.coerce.coerce_per_species`) so the accepted
spellings match the rest of xnns: a ``{Z: E0}`` / ``{symbol: E0}`` dict, a list
aligned with ``species``, a single number, or the string form of any of these.
The special value ``"average"`` (or ``"mean"``) instead fits the E0s from the
benchmark dataset by least squares -- the same convention MACE uses when no E0s
are given.

Note that MAE / MSE / RMSE between prediction and reference are *unchanged* by
this subtraction (the same per-structure offset cancels out of ``pred - ref``);
its effect is to make the reported energies physically meaningful and to give
relative or otherwise reference-dependent custom metrics a sensible baseline.
"""
from __future__ import annotations

import ast
from typing import Any, Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import Subset

from ..config.coerce import coerce_per_species, coerce_species, _symbol_to_z

# One more than the largest supported atomic number, so a lookup tensor of this
# length can be indexed directly by ``Z`` (index 0 unused, matching CHEMICAL_SYMBOLS).
MAX_Z = 119


def _lookup_from(species: Sequence[int], values: Sequence[float]) -> Tensor:
    """Scatter per-species values into a ``Z``-indexed lookup tensor.

    Parameters
    ----------
    species : sequence of int
        Atomic numbers.
    values : sequence of float
        The E0 for each atomic number in ``species`` (same order).

    Returns
    -------
    torch.Tensor
        A tensor of length :data:`MAX_Z` (default dtype), zero everywhere
        except at the given atomic numbers.
    """
    out = torch.zeros(MAX_Z, dtype=torch.get_default_dtype())
    for z, v in zip(species, values):
        out[int(z)] = float(v)
    return out


def dataset_structures(dataset) -> list[dict]:
    """Return the raw structure dicts backing a dataset or subset.

    Works for both an :class:`~xnns.common.data.AtomicDataset` and a
    ``torch.utils.data.Subset`` of one, so E0s can be fit from the dataset
    without building any graphs.

    Parameters
    ----------
    dataset : AtomicDataset or torch.utils.data.Subset
        The dataset (or subset) to read structures from.

    Returns
    -------
    list of dict
        The underlying structure dicts (each with ``atomic_numbers`` and,
        when present, ``energy``).
    """
    if isinstance(dataset, Subset):
        base = dataset.dataset
        return [base.structures[i] for i in dataset.indices]
    return list(dataset.structures)


def fit_atomic_energies(structures: Sequence[dict]) -> tuple[list[int], list[float]]:
    """Least-squares fit per-element reference energies from total energies.

    Solves ``C @ E0 ~= E_total`` where ``C[i, j]`` is the count of species ``j``
    in structure ``i`` -- the standard way to estimate E0s when they are not
    supplied (MACE's ``average`` scheme). Structures without an energy are
    skipped.

    Parameters
    ----------
    structures : sequence of dict
        Structure dicts with ``atomic_numbers`` and ``energy``.

    Returns
    -------
    tuple of (list of int, list of float)
        The sorted unique atomic numbers and their fitted reference energies.

    Raises
    ------
    ValueError
        If no structure carries an energy.
    """
    rows = [s for s in structures if s.get("energy") is not None]
    if not rows:
        raise ValueError(
            "atomic_energies='average' needs structures with reference "
            "energies to fit E0s from")
    species = sorted({int(z) for s in rows for z in s["atomic_numbers"]})
    col = {z: j for j, z in enumerate(species)}
    counts = torch.zeros(len(rows), len(species), dtype=torch.get_default_dtype())
    energy = torch.zeros(len(rows), dtype=torch.get_default_dtype())
    for i, s in enumerate(rows):
        for z in s["atomic_numbers"]:
            counts[i, col[int(z)]] += 1.0
        energy[i] = float(s["energy"])
    solution = torch.linalg.lstsq(counts, energy.unsqueeze(1)).solution.squeeze(1)
    return species, solution.tolist()


def build_e0_lookup(atomic_energies: Any, species: Any = None,
                    dataset=None) -> Optional[Tensor]:
    """Build a ``Z``-indexed E0 lookup tensor from a configured value.

    Parameters
    ----------
    atomic_energies : Any
        The configured reference energies. ``None`` disables subtraction
        (returns ``None``). ``"average"`` / ``"mean"`` fits E0s from
        ``dataset``. Otherwise a ``{Z: E0}`` / ``{symbol: E0}`` dict, a list
        aligned with ``species``, a single number, or the string form of any of
        these (parsed by
        :func:`~xnns.common.config.coerce.coerce_per_species`).
    species : Any, optional
        Atomic numbers or chemical symbols the values are aligned with, needed
        only for the list / scalar forms (a dict carries its own species).
    dataset : AtomicDataset or torch.utils.data.Subset or None, optional
        Benchmark dataset used to fit E0s when ``atomic_energies`` is
        ``"average"``.

    Returns
    -------
    torch.Tensor or None
        A lookup tensor of length :data:`MAX_Z`, or ``None`` when subtraction
        is disabled.

    Raises
    ------
    ValueError
        If ``"average"`` is requested without a dataset, or a list/scalar
        form is given without ``species``.
    """
    if atomic_energies is None:
        return None

    if isinstance(atomic_energies, str) and atomic_energies.lower() in (
            "average", "mean"):
        if dataset is None:
            raise ValueError(
                "atomic_energies='average' requires a dataset to fit from")
        fitted_species, values = fit_atomic_energies(dataset_structures(dataset))
        return _lookup_from(fitted_species, values)

    value = atomic_energies
    if isinstance(value, str):
        value = ast.literal_eval(value)

    if isinstance(value, dict):
        by_z = {_symbol_to_z(k, "atomic_energies"): float(v)
                for k, v in value.items()}
        return _lookup_from(list(by_z), list(by_z.values()))

    if species is None:
        raise ValueError(
            "atomic_energies given as a list/number needs 'species' to align "
            "the values with; use a {Z: E0} dict to avoid this")
    sp = coerce_species(species)
    tensor = coerce_per_species(value, sp, "atomic_energies (E0s)")
    return _lookup_from(sp, tensor.tolist())
