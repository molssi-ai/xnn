"""Fixed molecular topology for valence force fields (OPLS and friends).

Classical valence force fields such as OPLS assign every atom a fixed *atom
type* and evaluate bonded terms over a fixed connectivity: bonds, the angles
and proper dihedrals implied by those bonds, and explicitly declared improper
dihedrals at trigonal centers. The same connectivity determines the nonbonded
bookkeeping -- 1,2 and 1,3 pairs are excluded, 1,4 pairs are scaled.

:class:`MolecularTopology` holds exactly that information for one structure:
the per-atom type names, the bond list, and everything derived from it
(angles, dihedrals, exclusions, 1,4 pairs). It is deliberately independent of
any parameter library -- type names are opaque strings resolved by the model
that consumes the topology -- so the same class can serve other fixed-topology
force fields later.

A topology can be built from an explicit bond list
(:meth:`MolecularTopology.from_bonds`), from an ASE ``Atoms`` object with
bonds guessed from covalent radii (:meth:`MolecularTopology.from_ase`), or
loaded from the JSON file written by :meth:`MolecularTopology.save`.
:meth:`MolecularTopology.replicate` tiles a molecule into a multi-molecule
system (e.g. a liquid box built with ``ase.build``).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union


# Covalent radii in Angstrom (Cordero et al., Dalton Trans. 2008, 2832),
# indexed by atomic number; used only by the bond-guessing helper.
COVALENT_RADII = {
    1: 0.31, 2: 0.28, 3: 1.28, 4: 0.96, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66,
    9: 0.57, 10: 0.58, 11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07,
    16: 1.05, 17: 1.02, 18: 1.06, 19: 2.03, 20: 1.76, 35: 1.20, 53: 1.39,
}


def guess_bonds(positions, atomic_numbers, cell=None, scale: float = 1.2
                ) -> list[tuple[int, int]]:
    """Guess covalent bonds from interatomic distances.

    Two atoms are bonded when their (minimum-image) distance is below
    ``scale * (r_cov_a + r_cov_b)`` with the Cordero covalent radii. This is
    the standard structure-to-topology heuristic; inspect the result for
    unusual geometries.

    In a periodic cell every image within the bond cutoff is considered, for
    any (also unreduced) cell: a pair can only be that close when each
    fractional component of its displacement lies within ``cutoff / h_i`` of
    a lattice plane (``h_i`` the spacing of the planes along axis ``i``), so
    the distances of the few pairs passing that test are minimized over
    their candidate images.

    Parameters
    ----------
    positions : array_like
        Cartesian positions of shape ``(N, 3)`` in Angstrom.
    atomic_numbers : array_like
        Atomic numbers of shape ``(N,)``.
    cell : array_like, optional
        Lattice vectors as rows, shape ``(3, 3)``; when given, distances use
        the minimum-image convention.
    scale : float, optional
        Multiplier on the covalent-radius sum, by default 1.2.

    Returns
    -------
    list of tuple of int
        Bonds as ``(i, j)`` index pairs with ``i < j``, sorted.

    Raises
    ------
    KeyError
        If an atomic number has no tabulated covalent radius.
    """
    import itertools

    import numpy as np
    pos = np.asarray(positions, dtype=float)
    z = np.asarray(atomic_numbers, dtype=int)
    radii = np.array([COVALENT_RADII[int(zi)] for zi in z])
    n = len(z)
    if cell is not None:
        cell = np.asarray(cell, dtype=float)
        inv = np.linalg.inv(cell)
        normals = np.cross(np.roll(cell, -1, axis=0), np.roll(cell, -2, axis=0))
        spacing = abs(np.linalg.det(cell)) / np.linalg.norm(normals, axis=1)   # h_i
        eps = scale * 2.0 * radii.max() / spacing                                # (3,)
        reps = np.maximum(1, np.floor(eps + 0.5).astype(int))
        offsets = np.array(list(itertools.product(*[range(-r, r + 1) for r in reps])), dtype=float)
    bonds = []
    block = max(1, int(2e6 // max(n, 1)))        # rows per block: about 2e6 pairs
    for i0 in range(0, n - 1, block):
        i1 = min(n - 1, i0 + block)
        vec = pos[None, :, :] - pos[i0:i1, None, :]                              # (b, n, 3)
        rmax = scale * (radii[i0:i1, None] + radii[None, :])
        upper = np.arange(n)[None, :] > np.arange(i0, i1)[:, None]
        if cell is None:
            d = np.sqrt((vec * vec).sum(-1))
            rows, cols = np.nonzero((d < rmax) & upper)
        else:
            frac = vec @ inv
            frac -= np.round(frac)
            near = ((np.abs(frac) < eps) | (eps > 0.5)).all(-1) & upper
            rows, cols = np.nonzero(near)
            cand = (frac[rows, cols][:, None, :] + offsets) @ cell               # (P, K, 3)
            d = np.sqrt((cand * cand).sum(-1)).min(-1)
            keep = d < rmax[rows, cols]
            rows, cols = rows[keep], cols[keep]
        bonds.extend(zip((rows + i0).tolist(), cols.tolist()))
    return sorted(bonds)


@dataclass
class MolecularTopology:
    """Atom types and fixed valence connectivity for one structure.

    Build instances with :meth:`from_bonds` (which derives angles, proper
    dihedrals, exclusions and 1,4 pairs from the bond list) rather than by
    filling every field manually.

    Parameters and attributes
    -------------------------
    types : list[str]
        Per-atom type names (e.g. ``"opls_135"``), length ``n_atoms``.
    bonds : list[tuple[int, int]]
        Bonds as ``(i, j)`` with ``i < j``.
    bond_orders : list[float]
        Bond order per entry of ``bonds`` (1, 1.5, 2, 3); empty means all
        single bonds. Only rule-generated force fields (DREIDING) consume
        these; OPLS ignores them.
    angles : list[tuple[int, int, int]]
        Angles ``(i, j, k)`` with center ``j`` and ``i < k``.
    dihedrals : list[tuple[int, int, int, int]]
        Proper dihedrals ``(i, j, k, l)`` around the central bond ``j-k``.
    impropers : list[tuple[int, int, int, int]]
        Improper dihedrals, evaluated with the same four-atom dihedral angle
        as propers over the atoms *in the order given*.
    improper_keys : list[str]
        Improper type key per entry of ``impropers`` (e.g. ``"Z-CM-X-Y"``),
        resolved by the consuming force field.
    exclusions : list[tuple[int, int]]
        Nonbonded exclusions: all 1,2 and 1,3 pairs, ``i < j``.
    pairs14 : list[tuple[int, int]]
        1,4 pairs (dihedral end atoms, minus any that are also 1,2 or 1,3
        pairs in rings), ``i < j``; the force field scales these.
    pairs15 : list[tuple[int, int]]
        1,5 pairs (atoms separated by exactly four bonds, minus any closer
        pairs in rings), ``i < j``; the polarizable force fields (AMOEBA)
        scale these.
    charges : list[float]
        Per-atom partial charges in e, for force fields whose charges belong
        to the structure (residue templates, AMBER) rather than to the atom
        types; empty when none were assigned. OPLS and DREIDING ignore
        these.
    residues : list
        Per atom, ``(residue index, position within the residue template)``
        or ``None``; set by template typing and read by force fields whose
        conventions depend on template order (the AMBER improper ordering).
        Empty when unknown.
    residue_names : list[str]
        The template (residue) name of each residue index in ``residues``;
        empty when unknown.
    """

    types: list[str]
    bonds: list[tuple[int, int]]
    bond_orders: list[float] = field(default_factory=list)
    angles: list[tuple[int, int, int]] = field(default_factory=list)
    dihedrals: list[tuple[int, int, int, int]] = field(default_factory=list)
    impropers: list[tuple[int, int, int, int]] = field(default_factory=list)
    improper_keys: list[str] = field(default_factory=list)
    exclusions: list[tuple[int, int]] = field(default_factory=list)
    pairs14: list[tuple[int, int]] = field(default_factory=list)
    pairs15: list[tuple[int, int]] = field(default_factory=list)
    charges: list[float] = field(default_factory=list)
    residues: list = field(default_factory=list)
    residue_names: list = field(default_factory=list)

    @property
    def n_atoms(self) -> int:
        """int : Number of atoms in the topology."""
        return len(self.types)

    # constructors
    @classmethod
    def from_bonds(cls, types: Sequence[str],
                   bonds: Sequence[Sequence[int]],
                   impropers: Sequence[Sequence[int]] = (),
                   improper_keys: Sequence[str] = (),
                   bond_orders: Optional[Sequence[float]] = None,
                   charges: Optional[Sequence[float]] = None,
                   residues: Optional[Sequence] = None,
                   residue_names: Optional[Sequence[str]] = None
                   ) -> "MolecularTopology":
        """Build a topology from atom types and a bond list.

        Angles are every unordered pair of bonded neighbors of a common
        center; proper dihedrals are enumerated once around every bond;
        exclusions are the 1,2 and 1,3 pairs; 1,4 pairs are the dihedral end
        atoms that are not themselves 1,2 or 1,3 pairs (each pair counted
        once, so fused rings are handled correctly).

        Parameters
        ----------
        types : sequence of str
            Per-atom type names.
        bonds : sequence of (int, int)
            Bonded atom index pairs (order within a pair does not matter).
        impropers : sequence of (int, int, int, int), optional
            Improper dihedrals, in evaluation order.
        improper_keys : sequence of str, optional
            Exact improper type key per improper (same length as
            ``impropers``). Leave empty to let the force field resolve each
            improper from the classes of its atoms (center third).
        bond_orders : sequence of float, optional
            Bond order per entry of ``bonds`` (1, 1.5, 2, 3), in the *input*
            bond order; omitted means all single bonds.
        charges : sequence of float, optional
            Per-atom partial charges in e (see the class attributes).
        residues : sequence, optional
            Per-atom ``(residue index, template position)`` or ``None``.
        residue_names : sequence of str, optional
            Template name per residue index.

        Returns
        -------
        MolecularTopology
            The completed topology.

        Raises
        ------
        ValueError
            If a bond index is out of range, a bond is duplicated or joins an
            atom to itself, ``impropers`` and ``improper_keys`` disagree in
            length, or ``bond_orders``, ``charges`` or ``residues`` have the
            wrong length.
        """
        n = len(types)
        if bond_orders is not None and len(bond_orders) != len(bonds):
            raise ValueError(f"got {len(bond_orders)} bond orders for "
                             f"{len(bonds)} bonds")
        if charges is not None and len(charges) != n:
            raise ValueError(f"got {len(charges)} charges for {n} atoms")
        if residues is not None and len(residues) != n:
            raise ValueError(f"got {len(residues)} residue entries for {n} atoms")
        norm_bonds: list[tuple[int, int]] = []
        order_of: dict[tuple[int, int], float] = {}
        seen = set()
        for m, pair in enumerate(bonds):
            i, j = int(pair[0]), int(pair[1])
            if not (0 <= i < n and 0 <= j < n):
                raise ValueError(f"bond ({i}, {j}) is out of range for "
                                 f"{n} atoms")
            if i == j:
                raise ValueError(f"bond ({i}, {j}) joins an atom to itself")
            key = (min(i, j), max(i, j))
            if key in seen:
                raise ValueError(f"duplicate bond {key}")
            seen.add(key)
            norm_bonds.append(key)
            if bond_orders is not None:
                order_of[key] = float(bond_orders[m])
        norm_bonds.sort()
        norm_orders = [order_of[b] for b in norm_bonds] if bond_orders \
            is not None else []

        if improper_keys and len(impropers) != len(improper_keys):
            raise ValueError("impropers and improper_keys must have the same "
                             f"length (got {len(impropers)} and "
                             f"{len(improper_keys)}); leave improper_keys "
                             "empty to resolve impropers by atom class")

        neighbors: list[list[int]] = [[] for _ in range(n)]
        for i, j in norm_bonds:
            neighbors[i].append(j)
            neighbors[j].append(i)
        for nb in neighbors:
            nb.sort()

        angles = []
        for j in range(n):
            nb = neighbors[j]
            for a in range(len(nb)):
                for b in range(a + 1, len(nb)):
                    angles.append((nb[a], j, nb[b]))

        dihedrals = []
        dihedral_seen = set()
        for j, k in norm_bonds:
            for i in neighbors[j]:
                if i == k:
                    continue
                for l in neighbors[k]:
                    if l == j or l == i:
                        continue
                    quad = (i, j, k, l)
                    canon = min(quad, quad[::-1])
                    if canon not in dihedral_seen:
                        dihedral_seen.add(canon)
                        dihedrals.append(quad)

        excl = set(norm_bonds)
        for i, j, k in angles:
            excl.add((min(i, k), max(i, k)))

        pairs14 = set()
        for i, j, k, l in dihedrals:
            pair = (min(i, l), max(i, l))
            if pair not in excl:
                pairs14.add(pair)

        # 1,5 pairs: one bond beyond a dihedral, not already closer
        pairs15 = set()
        for i, j, k, l in dihedrals:
            for m in neighbors[l]:
                if m in (i, j, k):
                    continue
                pair = (min(i, m), max(i, m))
                if pair not in excl and pair not in pairs14:
                    pairs15.add(pair)
            for m in neighbors[i]:
                if m in (j, k, l):
                    continue
                pair = (min(m, l), max(m, l))
                if pair not in excl and pair not in pairs14:
                    pairs15.add(pair)

        return cls(types=list(types), bonds=norm_bonds,
                   bond_orders=norm_orders, angles=angles,
                   dihedrals=dihedrals,
                   impropers=[tuple(int(x) for x in im) for im in impropers],
                   improper_keys=list(improper_keys),
                   exclusions=sorted(excl), pairs14=sorted(pairs14),
                   pairs15=sorted(pairs15),
                   charges=[float(q) for q in charges] if charges is not None else [],
                   residues=[None if r is None else (int(r[0]), int(r[1]))
                             for r in residues] if residues is not None else [],
                   residue_names=[str(x) for x in residue_names]
                   if residue_names is not None else [])

    @classmethod
    def from_ase(cls, atoms, types: Optional[Sequence[str]] = None,
                 bonds: Optional[Sequence[Sequence[int]]] = None,
                 scale: float = 1.2,
                 impropers: Sequence[Sequence[int]] = (),
                 improper_keys: Sequence[str] = (), *,
                 forcefield=None, charge: int = 0,
                 bond_orders: Optional[Sequence[float]] = None,
                 charges: Optional[Sequence[float]] = None,
                 residues: Optional[Sequence] = None,
                 residue_names: Optional[Sequence[str]] = None
                 ) -> "MolecularTopology":
        """Build a topology for an ASE ``Atoms`` object.

        Parameters
        ----------
        atoms : ase.Atoms
            The structure (used for positions, atomic numbers, and the cell
            when bonds are guessed).
        types : sequence of str, optional
            Per-atom type names, in ``atoms`` order. When omitted,
            ``forcefield`` must be given and the types are assigned from its
            SMARTS templates (:func:`~xnn.ffnn.common.typing.assign_atom_types`).
        bonds : sequence of (int, int), optional
            Explicit bond list; when omitted, bonds are perceived by RDKit
            together with the typing when ``forcefield`` is given, else
            guessed from covalent radii with :func:`guess_bonds`.
        scale : float, optional
            Covalent-radius multiplier for bond guessing, by default 1.2.
        impropers, improper_keys : sequence, optional
            Improper dihedrals and their type keys (see :meth:`from_bonds`).
        forcefield : ForceField, OPLSLibrary or str, optional
            Source of typing templates (a resolved force field, a library
            read from a ``.frc`` file, or a spec such as ``"oplsaa"``).
        charge : int, optional
            Total charge, for bond-order perception.
        bond_orders, charges, residues, residue_names : sequence, optional
            Explicit values (see :meth:`from_bonds`); when typing from a
            force field whose fragments carry charges, the charges and the
            residue positions come from the templates unless given here.

        Returns
        -------
        MolecularTopology
            The completed topology.

        Raises
        ------
        ValueError
            If ``types`` does not match the number of atoms, or neither
            ``types`` nor ``forcefield`` is given.
        """
        if types is None:
            if forcefield is None:
                raise ValueError("give per-atom types, or a forcefield whose "
                                 "templates assign them")
            from ..common.typing import (assign_atom_types, perceive_bonds,
                                         perceive_bond_orders)
            types, details = assign_atom_types(atoms, forcefield, charge=charge,
                                               bonds=bonds, return_details=True)
            mol = details["mol"]
            if bonds is None:
                bonds = perceive_bonds(mol)
                if bond_orders is None:
                    bond_orders = perceive_bond_orders(mol)
            if charges is None and any(q is not None for q in details["charges"]):
                charges = [0.0 if q is None else q for q in details["charges"]]
            if residues is None and any(r is not None for r in details["residues"]):
                residues = details["residues"]
                if residue_names is None:
                    residue_names = residue_name_list(details["residues"],
                                                      details["fragment_names"])
        if len(types) != len(atoms):
            raise ValueError(f"got {len(types)} types for {len(atoms)} atoms")
        if bonds is None:
            cell = atoms.get_cell()[:] if atoms.pbc.any() else None
            bonds = guess_bonds(atoms.get_positions(),
                                atoms.get_atomic_numbers(), cell=cell,
                                scale=scale)
        return cls.from_bonds(types, bonds, impropers=impropers,
                              improper_keys=improper_keys,
                              bond_orders=bond_orders, charges=charges,
                              residues=residues, residue_names=residue_names)

    def replicate(self, n_copies: int) -> "MolecularTopology":
        """Tile this topology into ``n_copies`` consecutive molecules.

        Atom ``a`` of copy ``c`` becomes atom ``c * n_atoms + a``; this
        matches the atom ordering of ``ase.Atoms`` multiplication and of
        packing tools that concatenate whole molecules.

        Parameters
        ----------
        n_copies : int
            Number of copies (>= 1).

        Returns
        -------
        MolecularTopology
            The tiled topology.
        """
        n = self.n_atoms

        def shift(items, off):
            """Offset every index tuple in ``items`` by ``off``."""
            return [tuple(x + off for x in item) for item in items]

        out = MolecularTopology(types=list(self.types) * n_copies,
                                bonds=[], bond_orders=[], angles=[],
                                dihedrals=[], impropers=[], improper_keys=[],
                                exclusions=[], pairs14=[], pairs15=[],
                                charges=list(self.charges) * n_copies,
                                residues=[],
                                residue_names=list(self.residue_names) * n_copies)
        n_res = 1 + max((r[0] for r in self.residues if r is not None),
                        default=-1)
        for c in range(n_copies):
            off = c * n
            out.residues += [None if r is None else (r[0] + c * n_res, r[1])
                             for r in self.residues]
            out.bonds += shift(self.bonds, off)
            out.bond_orders += list(self.bond_orders)
            out.angles += shift(self.angles, off)
            out.dihedrals += shift(self.dihedrals, off)
            out.impropers += shift(self.impropers, off)
            out.improper_keys += list(self.improper_keys)
            out.exclusions += shift(self.exclusions, off)
            out.pairs14 += shift(self.pairs14, off)
            out.pairs15 += shift(self.pairs15, off)
        return out

    # persistence
    def save(self, path: Union[str, Path]) -> None:
        """Write the topology as JSON (round-tripped by :func:`read_topology`).

        Parameters
        ----------
        path : str or Path
            Output file path.
        """
        data = {
            "types": self.types,
            "bonds": [list(b) for b in self.bonds],
            "impropers": [list(im) for im in self.impropers],
            "improper_keys": self.improper_keys,
        }
        if self.bond_orders:
            data["bond_orders"] = list(self.bond_orders)
        if self.charges:
            data["charges"] = list(self.charges)
        if self.residues:
            data["residues"] = [None if r is None else list(r)
                                for r in self.residues]
        if self.residue_names:
            data["residue_names"] = list(self.residue_names)
        Path(path).write_text(json.dumps(data, indent=2) + "\n")


def read_topology(path: Union[str, Path]) -> MolecularTopology:
    """Read a topology JSON file written by :meth:`MolecularTopology.save`.

    Only the independent information (types, bonds, impropers, and the
    optional bond orders, charges and residue positions) is stored on disk;
    the derived lists are rebuilt on load.

    Parameters
    ----------
    path : str or Path
        The JSON file.

    Returns
    -------
    MolecularTopology
        The loaded topology.
    """
    data = json.loads(Path(path).read_text())
    return MolecularTopology.from_bonds(
        data["types"], data["bonds"], impropers=data.get("impropers", ()),
        improper_keys=data.get("improper_keys", ()),
        bond_orders=data.get("bond_orders"), charges=data.get("charges"),
        residues=data.get("residues"), residue_names=data.get("residue_names"))


def residue_name_list(residues: Sequence, fragment_names: Sequence) -> list:
    """Template name per residue index from per-atom typing details.

    Parameters
    ----------
    residues : sequence
        Per atom, ``(residue index, position)`` or ``None``.
    fragment_names : sequence
        Per atom, the fragment name or ``None``.

    Returns
    -------
    list of str
        Name of residue ``0, 1, ...`` (``""`` for an index without a name).
    """
    names: dict[int, str] = {}
    for r, name in zip(residues, fragment_names):
        if r is not None and name is not None:
            names.setdefault(int(r[0]), str(name))
    n = 1 + max(names, default=-1)
    return [names.get(i, "") for i in range(n)]
