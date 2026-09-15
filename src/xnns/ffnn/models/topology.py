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


def _min_image(vec, cell):
    """Apply the minimum-image convention to displacement vectors.

    Parameters
    ----------
    vec : numpy.ndarray
        Displacement vectors of shape ``(..., 3)``.
    cell : numpy.ndarray or None
        Lattice vectors as rows, shape ``(3, 3)``, or ``None`` for a
        non-periodic system.

    Returns
    -------
    numpy.ndarray
        Minimum-image displacement vectors.
    """
    import numpy as np
    if cell is None:
        return vec
    frac = vec @ np.linalg.inv(cell)
    return (frac - np.round(frac)) @ cell


def guess_bonds(positions, atomic_numbers, cell=None, scale: float = 1.2
                ) -> list[tuple[int, int]]:
    """Guess covalent bonds from interatomic distances.

    Two atoms are bonded when their (minimum-image) distance is below
    ``scale * (r_cov_a + r_cov_b)`` with the Cordero covalent radii. This is
    the standard structure-to-topology heuristic; inspect the result for
    unusual geometries.

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
    import numpy as np
    pos = np.asarray(positions, dtype=float)
    z = np.asarray(atomic_numbers, dtype=int)
    cell = None if cell is None else np.asarray(cell, dtype=float)
    radii = np.array([COVALENT_RADII[int(zi)] for zi in z])
    bonds = []
    for i in range(len(z) - 1):
        vec = _min_image(pos[i + 1:] - pos[i], cell)
        d = np.sqrt((vec * vec).sum(axis=1))
        rmax = scale * (radii[i] + radii[i + 1:])
        for j in np.nonzero(d < rmax)[0]:
            bonds.append((i, i + 1 + int(j)))
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
    """

    types: list[str]
    bonds: list[tuple[int, int]]
    angles: list[tuple[int, int, int]] = field(default_factory=list)
    dihedrals: list[tuple[int, int, int, int]] = field(default_factory=list)
    impropers: list[tuple[int, int, int, int]] = field(default_factory=list)
    improper_keys: list[str] = field(default_factory=list)
    exclusions: list[tuple[int, int]] = field(default_factory=list)
    pairs14: list[tuple[int, int]] = field(default_factory=list)

    @property
    def n_atoms(self) -> int:
        """int : Number of atoms in the topology."""
        return len(self.types)

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_bonds(cls, types: Sequence[str],
                   bonds: Sequence[Sequence[int]],
                   impropers: Sequence[Sequence[int]] = (),
                   improper_keys: Sequence[str] = ()) -> "MolecularTopology":
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
            Improper type key per improper (same length as ``impropers``).

        Returns
        -------
        MolecularTopology
            The completed topology.

        Raises
        ------
        ValueError
            If a bond index is out of range, a bond is duplicated or joins an
            atom to itself, or ``impropers`` and ``improper_keys`` disagree in
            length.
        """
        n = len(types)
        norm_bonds: list[tuple[int, int]] = []
        seen = set()
        for pair in bonds:
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
        norm_bonds.sort()

        if len(impropers) != len(improper_keys):
            raise ValueError("impropers and improper_keys must have the same "
                             f"length (got {len(impropers)} and "
                             f"{len(improper_keys)})")

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

        return cls(types=list(types), bonds=norm_bonds, angles=angles,
                   dihedrals=dihedrals,
                   impropers=[tuple(int(x) for x in im) for im in impropers],
                   improper_keys=list(improper_keys),
                   exclusions=sorted(excl), pairs14=sorted(pairs14))

    @classmethod
    def from_ase(cls, atoms, types: Sequence[str],
                 bonds: Optional[Sequence[Sequence[int]]] = None,
                 scale: float = 1.2,
                 impropers: Sequence[Sequence[int]] = (),
                 improper_keys: Sequence[str] = ()) -> "MolecularTopology":
        """Build a topology for an ASE ``Atoms`` object.

        Parameters
        ----------
        atoms : ase.Atoms
            The structure (used for positions, atomic numbers, and the cell
            when bonds are guessed).
        types : sequence of str
            Per-atom type names, in ``atoms`` order.
        bonds : sequence of (int, int), optional
            Explicit bond list; when omitted, bonds are guessed with
            :func:`guess_bonds`.
        scale : float, optional
            Covalent-radius multiplier for bond guessing, by default 1.2.
        impropers, improper_keys : sequence, optional
            Improper dihedrals and their type keys (see :meth:`from_bonds`).

        Returns
        -------
        MolecularTopology
            The completed topology.

        Raises
        ------
        ValueError
            If ``types`` does not match the number of atoms.
        """
        if len(types) != len(atoms):
            raise ValueError(f"got {len(types)} types for {len(atoms)} atoms")
        if bonds is None:
            cell = atoms.get_cell()[:] if atoms.pbc.any() else None
            bonds = guess_bonds(atoms.get_positions(),
                                atoms.get_atomic_numbers(), cell=cell,
                                scale=scale)
        return cls.from_bonds(types, bonds, impropers=impropers,
                              improper_keys=improper_keys)

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
                                bonds=[], angles=[], dihedrals=[],
                                impropers=[], improper_keys=[],
                                exclusions=[], pairs14=[])
        for c in range(n_copies):
            off = c * n
            out.bonds += shift(self.bonds, off)
            out.angles += shift(self.angles, off)
            out.dihedrals += shift(self.dihedrals, off)
            out.impropers += shift(self.impropers, off)
            out.improper_keys += list(self.improper_keys)
            out.exclusions += shift(self.exclusions, off)
            out.pairs14 += shift(self.pairs14, off)
        return out

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
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
        Path(path).write_text(json.dumps(data, indent=2) + "\n")


def read_topology(path: Union[str, Path]) -> MolecularTopology:
    """Read a topology JSON file written by :meth:`MolecularTopology.save`.

    Only the independent information (types, bonds, impropers) is stored on
    disk; the derived lists are rebuilt on load.

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
        improper_keys=data.get("improper_keys", ()))
