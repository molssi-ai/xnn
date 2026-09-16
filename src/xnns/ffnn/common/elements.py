"""Element symbols and atomic numbers shared by the ``ffnn`` family.

Force-field files key species by symbol (``C``, ``Fe``), models by atomic
number; this is the one table both directions go through.
"""
from __future__ import annotations

# Chemical symbols indexed by atomic number (Z = index; index 0 is a dummy).
CHEMICAL_SYMBOLS = (
    "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
    "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb",
    "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm",
    "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta",
    "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At",
    "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu",
)

SYMBOL_TO_Z = {sym: z for z, sym in enumerate(CHEMICAL_SYMBOLS) if z > 0}


def atomic_number(symbol: str) -> int:
    """Atomic number of an element symbol (case-insensitive).

    Raises
    ------
    KeyError
        For an unknown symbol.
    """
    s = str(symbol).strip()
    if s in SYMBOL_TO_Z:
        return SYMBOL_TO_Z[s]
    cap = s[:1].upper() + s[1:].lower()
    if cap in SYMBOL_TO_Z:
        return SYMBOL_TO_Z[cap]
    raise KeyError(f"unknown element symbol {symbol!r}")
