"""Atom typing from a force field's SMARTS templates.

A ``.frc`` force field carries, in its ``#templates`` section, one or more
SMARTS patterns per atom type; the atom-mapped atoms of a pattern (``[C:1]``)
receive that type wherever the pattern matches. This module applies those
templates to a structure with RDKit, following the procedure of SEAMM's
``FFAssigner``: fragments (whole-molecule patterns with a fixed type list)
are assigned first and are never overwritten; then the templates are applied
in file order, each later match overriding earlier ones, which is why the
files list general patterns before specific ones. An atom left untyped is an
error -- the force field cannot describe the structure.

RDKit is an optional dependency (``pip install "xnn[ffnn]"``); everything
here imports it lazily and raises a clear error without it.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from .frc import ForceField


class AtomTypingError(ValueError):
    """The force field's templates do not cover (part of) the structure."""


def _rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
    except ImportError as err:  # pragma: no cover - environment dependent
        raise ImportError(
            "atom typing from SMARTS templates needs RDKit: "
            "pip install rdkit  (or the 'ffnn' extra of xnn)") from err
    return Chem, rdDetermineBonds


def to_rdkit(structure=None, *, positions=None, atomic_numbers=None,
             bonds: Optional[Sequence[Sequence[int]]] = None,
             charge: int = 0, smiles: Optional[str] = None):
    """Build an RDKit molecule with explicit hydrogens from a structure.

    Accepts, in order of preference: an RDKit ``Mol`` (returned as is, after
    adding explicit hydrogens if it has implicit ones); a SMILES string;
    an ``ase.Atoms``; a dict with ``"pos"`` / ``"atomic_numbers"``; or
    ``positions`` + ``atomic_numbers`` arrays. For geometric input the
    connectivity and bond orders are perceived with RDKit's
    ``DetermineBonds`` (the xyz2mol algorithm); if ``bonds`` are given only
    the bond *orders* are determined, keeping that connectivity.

    Parameters
    ----------
    structure : object, optional
        See above.
    positions : array_like, optional
        Cartesian coordinates ``(N, 3)`` in Angstrom.
    atomic_numbers : array_like, optional
        Atomic numbers ``(N,)``.
    bonds : sequence of (int, int), optional
        Known connectivity; bond orders are then perceived, not connectivity.
    charge : int, optional
        Total charge of the structure (needed to perceive bond orders).
    smiles : str, optional
        A SMILES string (hydrogens are added explicitly).

    Returns
    -------
    rdkit.Chem.Mol
        A sanitized molecule whose atom order matches the input order (for
        SMILES input, the RDKit order with hydrogens appended).
    """
    Chem, det = _rdkit()
    if smiles is not None or isinstance(structure, str):
        mol = Chem.MolFromSmiles(smiles if smiles is not None else structure)
        if mol is None:
            raise ValueError(f"RDKit cannot parse SMILES {smiles or structure!r}")
        return Chem.AddHs(mol)
    if structure is not None and hasattr(structure, "GetNumAtoms"):
        mol = structure
        if any(a.GetNumImplicitHs() for a in mol.GetAtoms()):
            mol = Chem.AddHs(mol)
        return mol
    if structure is not None:
        if hasattr(structure, "get_positions"):          # ase.Atoms
            positions = structure.get_positions()
            atomic_numbers = structure.get_atomic_numbers()
        elif isinstance(structure, dict):
            positions = structure["pos"]
            atomic_numbers = structure["atomic_numbers"]
        elif isinstance(structure, (tuple, list)) and len(structure) == 2:
            positions, atomic_numbers = structure
    if positions is None or atomic_numbers is None:
        raise TypeError("give an RDKit Mol, a SMILES, an ase.Atoms, a dict with "
                        "'pos'/'atomic_numbers', or positions + atomic_numbers")
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    z = [int(x) for x in np.asarray(atomic_numbers).reshape(-1)]
    if len(z) != len(pos):
        raise ValueError(f"{len(z)} atomic numbers for {len(pos)} positions")

    rw = Chem.RWMol()
    for zi in z:
        atom = Chem.Atom(zi)
        atom.SetNoImplicit(True)
        rw.AddAtom(atom)
    conf = Chem.Conformer(len(z))
    for i, p in enumerate(pos):
        conf.SetAtomPosition(i, tuple(float(x) for x in p))
    rw.AddConformer(conf, assignId=True)
    if bonds is not None:
        for a, b in bonds:
            rw.AddBond(int(a), int(b), Chem.BondType.SINGLE)
    mol = rw.GetMol()
    try:
        if bonds is None:
            det.DetermineBonds(mol, charge=int(charge))
        else:
            det.DetermineBondOrders(mol, charge=int(charge))
    except Exception as err:
        raise ValueError(
            "RDKit could not perceive the bonding of the structure "
            f"(total charge {charge}); pass the connectivity with bonds=, "
            "the right total charge, or a SMILES") from err
    Chem.SanitizeMol(mol)
    return mol


def perceive_bonds(mol) -> list[tuple[int, int]]:
    """Sorted ``(i, j)`` bond list (``i < j``) of an RDKit molecule."""
    return sorted((min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                   max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
                  for b in mol.GetBonds())


def perceive_bond_orders(mol) -> list[float]:
    """Bond orders of an RDKit molecule, aligned with :func:`perceive_bonds`.

    Aromatic bonds come back as 1.5 (RDKit's ``GetBondTypeAsDouble``), the
    resonance bond order of rule-generated force fields such as DREIDING.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        The molecule.

    Returns
    -------
    list of float
        One order per bond, in the sorted ``(i, j)`` order of
        :func:`perceive_bonds`.
    """
    pairs = sorted(((min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                     max(b.GetBeginAtomIdx(), b.GetEndAtomIdx())),
                    float(b.GetBondTypeAsDouble())) for b in mol.GetBonds())
    return [order for _, order in pairs]


def _map_list(pattern) -> list[int]:
    """Pattern atom indices carrying atom maps, in map-number order."""
    ind = {}
    for atom in pattern.GetAtoms():
        n = atom.GetAtomMapNum()
        if n:
            ind[n] = atom.GetIdx()
    return [ind[n] for n in sorted(ind)]


def assign_atom_types(structure, forcefield: Union[ForceField, str], *,
                      charge: int = 0,
                      bonds: Optional[Sequence[Sequence[int]]] = None,
                      return_mol: bool = False):
    """Assign the force field's atom types to a structure.

    Parameters
    ----------
    structure : object
        Anything :func:`to_rdkit` accepts.
    forcefield : ForceField, str or object
        A resolved force field, a spec for
        :func:`~xnn.ffnn.common.frc.read_forcefield` (``"oplsaa"``, a
        ``.frc`` path, ...), or any object exposing ``templates`` /
        ``fragments`` dicts in the same shape (an
        :class:`~xnn.ffnn.models.oplslib.OPLSLibrary` read from a ``.frc``
        file does). It must carry templates.
    charge : int, optional
        Total charge, for bond perception from coordinates.
    bonds : sequence of (int, int), optional
        Known connectivity (see :func:`to_rdkit`).
    return_mol : bool, optional
        Also return the RDKit molecule (whose bonds give the topology).

    Returns
    -------
    list of str
        The atom type per atom, in input order.
    rdkit.Chem.Mol
        Only when ``return_mol`` is set.

    Raises
    ------
    AtomTypingError
        If any atom matches no template, or two fragments disagree.
    ValueError
        If the force field has no templates.
    """
    Chem, _ = _rdkit()
    if isinstance(forcefield, str):
        from .frc import read_forcefield
        forcefield = read_forcefield(forcefield)
    templates = getattr(forcefield, "templates", None) or {}
    fragments = getattr(forcefield, "fragments", None) or {}
    if not templates and not fragments:
        raise ValueError(f"force field {forcefield.name!r} has no #templates; "
                         "atom types must be given explicitly")
    mol = to_rdkit(structure, bonds=bonds, charge=charge)
    n = mol.GetNumAtoms()
    types = ["?"] * n
    max_matches = 6 * n + 1

    # fragments first, and they are final
    for name, data in fragments.items():
        smarts = data.get("SMARTS")
        if not smarts:
            continue
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            raise ValueError(f"fragment {name!r}: bad SMARTS {smarts!r}")
        for match in mol.GetSubstructMatches(pattern, maxMatches=max_matches):
            for atype, atom in zip(data.get("atom types", []), match):
                if types[atom] != "?" and types[atom] != atype:
                    raise AtomTypingError(
                        f"atom {atom} matched by two fragments with different "
                        f"types ({types[atom]} vs {atype})")
                types[atom] = atype
    fixed = [t != "?" for t in types]

    # templates in file order; later matches override earlier ones
    for atype, entry in templates.items():
        for smarts in entry.get("smarts", []):
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is None:
                raise ValueError(f"template {atype!r}: bad SMARTS {smarts!r}")
            mapped = _map_list(pattern)
            if not mapped:
                continue
            for match in mol.GetSubstructMatches(pattern, maxMatches=max_matches):
                for x in (match[m] for m in mapped):
                    if not fixed[x]:
                        types[x] = atype

    untyped = [i for i, t in enumerate(types) if t == "?"]
    if untyped:
        desc = ", ".join(f"{i}:{mol.GetAtomWithIdx(i).GetSymbol()}"
                         for i in untyped[:12])
        raise AtomTypingError(
            f"force field {forcefield.name!r} has no atom type for "
            f"{len(untyped)} atom(s): {desc}"
            + (" ..." if len(untyped) > 12 else ""))
    return (types, mol) if return_mol else types
