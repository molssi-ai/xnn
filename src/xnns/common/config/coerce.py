"""Shared coercers for upstream config *value* forms.

The key-translation registry (:mod:`~xnns.common.config.translate`) rewrites
upstream key *names* to the xnns canonical ones; this module handles the value
forms those keys carry, so every ``Model.from_config`` accepts the same
spellings: species as atomic numbers or chemical symbols (also as the string
form of either), and per-species values as a list aligned with ``species``, a
``{Z: value}`` / ``{symbol: value}`` dict, a single number (broadcast), or the
string form of any of these. One implementation, used by every model, so an
upstream MACE / NequIP yaml parses identically under every model name.
"""
from __future__ import annotations

import ast
from typing import Any, Optional, Sequence

import torch

# Z -> symbol for Z = 1..118 (index 0 is a placeholder); avoids requiring ase
# for the chemical-symbol spellings used by upstream NequIP configs.
CHEMICAL_SYMBOLS = (
    "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
    "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb",
    "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm",
    "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta",
    "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At",
    "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk",
    "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt",
    "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)
SYMBOL_TO_Z = {sym: z for z, sym in enumerate(CHEMICAL_SYMBOLS) if z > 0}


def _symbol_to_z(entry: Any, key: str) -> int:
    """Coerce one species entry (atomic number or chemical symbol) to ``Z``.

    Parameters
    ----------
    entry : int or str
        An atomic number or a chemical symbol (e.g. ``"H"``).
    key : str
        Config key name, used in error messages.

    Returns
    -------
    int
        The atomic number.

    Raises
    ------
    ValueError
        If ``entry`` is not a known symbol or a valid atomic number.
    """
    if isinstance(entry, str):
        z = SYMBOL_TO_Z.get(entry)
        if z is None:
            raise ValueError(f"{key}: unknown chemical symbol {entry!r}")
        return z
    return int(entry)


def coerce_species(value: Any, default: Optional[Sequence[int]] = None) -> list[int]:
    """Coerce a configured species list to a list of atomic numbers.

    Accepts atomic numbers, chemical symbols (upstream NequIP
    ``chemical_symbols``), a mix of both, or the string form of such a list.

    Parameters
    ----------
    value : Any
        The raw config value; ``None`` selects ``default``.
    default : sequence of int or None, optional
        Fallback when ``value`` is ``None``.

    Returns
    -------
    list of int
        The atomic numbers, in the configured (channel) order.

    Raises
    ------
    ValueError
        If the resulting list is empty or contains an unknown symbol.
    """
    if value is None:
        value = default
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not value:
        raise ValueError("species must be a non-empty list of atomic numbers "
                         "or chemical symbols")
    return [_symbol_to_z(z, "species") for z in value]


def coerce_per_species(value: Any, species: Sequence[int], key: str,
                       allow_scalar: bool = True) -> Optional[torch.Tensor]:
    """Coerce a per-species config value to a tensor aligned with ``species``.

    Accepts a list aligned with ``species``, a ``{Z: value}`` or
    ``{symbol: value}`` dict, a single number (broadcast, when
    ``allow_scalar``), or the string form of any of these.

    Parameters
    ----------
    value : Any
        The raw config value; ``None`` passes through.
    species : sequence of int
        The atomic numbers the result is aligned with.
    key : str
        Config key name (e.g. ``"atomic_energies (MACE E0s)"``), used in error
        messages.
    allow_scalar : bool, optional
        Whether a single number is broadcast to every species, by default
        ``True``.

    Returns
    -------
    torch.Tensor or None
        One value per species (default dtype), or ``None`` when ``value`` is.

    Raises
    ------
    ValueError
        If the value cannot be parsed, a species is missing from a dict, or a
        list length does not match ``species``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError) as e:
            raise ValueError(
                f"{key} = {value!r} is not supported; give explicit per-species "
                "values (a list aligned with species, a {Z: value} or "
                "{symbol: value} dict, or a single number)"
            ) from e
    if isinstance(value, dict):
        by_z = {_symbol_to_z(k, key): v for k, v in value.items()}
        missing = [z for z in species if z not in by_z]
        if missing:
            raise ValueError(f"{key}: missing values for species {missing}")
        value = [by_z[z] for z in species]
    if isinstance(value, (int, float)):
        if not allow_scalar:
            raise ValueError(f"{key}: a single number is not supported here")
        value = [float(value)] * len(species)
    value = torch.as_tensor(value, dtype=torch.get_default_dtype())
    if value.numel() != len(species):
        raise ValueError(
            f"{key}: got {value.numel()} values for {len(species)} species"
        )
    return value
