"""Shared infrastructure of the ``ffnn`` (classical force field) family.

* :mod:`~xnns.ffnn.common.frc` -- the MolSSI/SEAMM ``.frc`` force-field file
  format: a schema-driven reader, the ``#define`` resolver that composes a
  named force-field variant out of labelled parameter sections, the bonded
  parameter lookup with equivalences and wildcards, a writer, and the
  registry of parameter files shipped with xnns.
* :mod:`~xnns.ffnn.common.typing` -- atom typing from the SMARTS templates a
  force field carries (RDKit).
* :mod:`~xnns.ffnn.common.elements` -- the element symbol table.

The model-specific bridges that turn a resolved force field into the
parameter tables a model consumes live next to the models
(:mod:`xnns.ffnn.models.oplslib`, :mod:`xnns.ffnn.models.ffield`).
"""
from .frc import (ForceField, FrcFile, Section, Row, read_frc, write_frc,
                  read_forcefield, find_forcefield, list_forcefields,
                  builtin_data_dir, register_section_schema, convert_units)
from .typing import (assign_atom_types, to_rdkit, perceive_bonds,
                     AtomTypingError)
from .elements import CHEMICAL_SYMBOLS, SYMBOL_TO_Z, atomic_number

__all__ = [
    "ForceField", "FrcFile", "Section", "Row", "read_frc", "write_frc",
    "read_forcefield", "find_forcefield", "list_forcefields", "builtin_data_dir",
    "register_section_schema", "convert_units",
    "assign_atom_types", "to_rdkit", "perceive_bonds", "AtomTypingError",
    "CHEMICAL_SYMBOLS", "SYMBOL_TO_Z", "atomic_number",
]
