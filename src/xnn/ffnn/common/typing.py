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
             charge: int = 0, smiles: Optional[str] = None,
             orders: bool = True):
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
    orders : bool, optional
        With ``bonds`` given, also perceive the bond orders and formal
        charges (default). ``False`` keeps the connectivity as single bonds
        without a valence check, which is enough for templates written in
        terms of elements, connectivity and hydrogen counts (the AMBER
        residue templates) and never fails on charged groups or ions.

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
    if bonds is not None and not orders:
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES
                         ^ Chem.SANITIZE_KEKULIZE ^ Chem.SANITIZE_SETAROMATICITY)
        return mol
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
                      return_mol: bool = False,
                      return_details: bool = False,
                      orders: bool = True):
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
    orders : bool, optional
        Perceive bond orders when ``bonds`` are given (see :func:`to_rdkit`).
    return_mol : bool, optional
        Also return the RDKit molecule (whose bonds give the topology).
    return_details : bool, optional
        Also return what the fragments assigned besides the types: a dict
        with ``"charges"`` (per-atom partial charge from the matching
        fragment's ``charges`` list, ``None`` where no fragment matched),
        ``"residues"`` (per atom, ``(fragment match index, position in the
        fragment)`` or ``None``; the position is the atom-map number of the
        pattern atom when the fragment SMARTS carries atom maps, else the
        pattern atom's index), ``"fragment_names"`` (per atom, the fragment
        name or ``None``) and ``"mol"``. Residue-based force fields (AMBER)
        read their charges and improper ordering from these.

    Returns
    -------
    list of str
        The atom type per atom, in input order.
    rdkit.Chem.Mol
        Only when ``return_mol`` is set (and ``return_details`` is not).
    dict
        Only when ``return_details`` is set.

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
    mol = to_rdkit(structure, bonds=bonds, charge=charge, orders=orders)
    n = mol.GetNumAtoms()
    types = ["?"] * n
    max_matches = 6 * n + 1

    # fragments first, and they are final. A fragment's lists ("atom types",
    # optional "charges") follow the pattern's atom order, or, when the
    # SMARTS carries atom maps, the map numbers (position m <-> list index
    # m - 1), which lets a residue keep its template's atom order whatever
    # the string order of the pattern
    charges: list = [None] * n
    residues: list = [None] * n
    frag_names: list = [None] * n
    n_matches = 0
    # larger fragments take precedence: a small template that also matches a
    # group inside a bigger one (an amide cap inside an asparagine side chain)
    # is skipped where the bigger match already covers all of its atoms
    found = []
    for order, (name, data) in enumerate(fragments.items()):
        smarts = data.get("SMARTS")
        if not smarts:
            continue
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            raise ValueError(f"fragment {name!r}: bad SMARTS {smarts!r}")
        maps = [a.GetAtomMapNum() for a in pattern.GetAtoms()]
        positions = [m - 1 for m in maps] if all(maps) else \
            list(range(pattern.GetNumAtoms()))
        for match in mol.GetSubstructMatches(pattern, maxMatches=max_matches):
            found.append((-len(match), order, name, data, positions, match))
    found.sort(key=lambda f: (f[0], f[1]))
    for _, _, name, data, positions, match in found:
        ftypes = list(data.get("atom types", []))
        fcharges = data.get("charges")
        if all(residues[atom] is not None for atom in match):
            continue
        for pos, atom in zip(positions, match):
            atype = ftypes[pos] if pos < len(ftypes) else None
            if atype is None:
                continue
            if types[atom] != "?" and types[atom] != atype:
                raise AtomTypingError(
                    f"atom {atom} matched by two fragments with different "
                    f"types ({types[atom]} vs {atype})")
            types[atom] = atype
            if fcharges is not None and pos < len(fcharges):
                q = float(fcharges[pos])
                if charges[atom] is not None and abs(charges[atom] - q) > 1e-12:
                    raise AtomTypingError(
                        f"atom {atom} matched by two fragments with "
                        f"different charges ({charges[atom]} vs {q})")
                charges[atom] = q
            if residues[atom] is None:
                residues[atom] = (n_matches, pos)
                frag_names[atom] = name
        n_matches += 1
    # residues are numbered in the order their first atom appears in the
    # structure (the chain order of a sequence-built or PDB-ordered input),
    # not in the order the fragment table lists them
    first_atom = {}
    for atom, r in enumerate(residues):
        if r is not None:
            first_atom.setdefault(r[0], atom)
    renumber = {m: k for k, m in enumerate(sorted(first_atom, key=first_atom.get))}
    residues = [None if r is None else (renumber[r[0]], r[1]) for r in residues]
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
            # every mapping, not only one per atom set: a pattern that names
            # two symmetric atoms (both oxygens of a carboxylate) types both
            for match in mol.GetSubstructMatches(pattern, uniquify=False,
                                                 maxMatches=max_matches):
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
    if return_details:
        return types, {"charges": charges, "residues": residues,
                       "fragment_names": frag_names, "mol": mol}
    return (types, mol) if return_mol else types
