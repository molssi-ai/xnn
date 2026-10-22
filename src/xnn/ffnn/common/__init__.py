"""Shared infrastructure of the ``ffnn`` (classical force field) family.

* :mod:`~xnn.ffnn.common.frc` -- the MolSSI/SEAMM ``.frc`` force-field file
  format: a schema-driven reader, the ``#define`` resolver that composes a
  named force-field variant out of labelled parameter sections, the bonded
  parameter lookup with equivalences and wildcards, a writer, and the
  registry of parameter files shipped with xnn.
* :mod:`~xnn.ffnn.common.typing` -- atom typing from the SMARTS templates a
  force field carries (RDKit).
* the element symbol table, re-exported from :mod:`xnn.common.data.elements`.

The model-specific bridges that turn a resolved force field into the
parameter tables a model consumes live next to the models
(:mod:`xnn.ffnn.models.oplslib`, :mod:`xnn.ffnn.models.ffield`).
"""
from xnn.common.data.elements import CHEMICAL_SYMBOLS, SYMBOL_TO_Z, atomic_number

from .frc import (
    ForceField,
    FrcFile,
    Row,
    Section,
    builtin_data_dir,
    convert_units,
    find_forcefield,
    list_forcefields,
    read_forcefield,
    read_frc,
    register_section_schema,
    write_frc,
)
from .typing import AtomTypingError, assign_atom_types, perceive_bond_orders, perceive_bonds, to_rdkit

__all__ = [
    "ForceField", "FrcFile", "Section", "Row", "read_frc", "write_frc",
    "read_forcefield", "find_forcefield", "list_forcefields", "builtin_data_dir",
    "register_section_schema", "convert_units",
    "assign_atom_types", "to_rdkit", "perceive_bonds",
    "perceive_bond_orders", "AtomTypingError",
    "CHEMICAL_SYMBOLS", "SYMBOL_TO_Z", "atomic_number",
]
