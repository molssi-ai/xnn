"""Molecular graphs for networks that read the two-dimensional structure (ByteFF, ByteFF-Pol).

A graph-parameterized force field predicts its parameters from the two-dimensional
molecular graph alone. :func:`molecular_graph` turns an RDKit molecule (explicit
hydrogens, atoms in the order of the coordinates) into the integer features and
term lists such a model reads:

* the Lewis structure is standardized first: a few hypervalent groups are written
  in charge-separated form (sulfoxides, trivalent sulfur, deprotonated
  sulfonamides, azides, odd N-oxides) and the resonance structure with the fewest
  formal charges is chosen (:func:`canonical_resonance`), so that equivalent
  inputs give the same graph;
* per atom: element, number of bonded neighbors, formal charge, number of ring
  bonds and the size of the smallest ring through the atom; per bond: whether it
  lies in a ring and its order (single, double, triple, aromatic). Rings are the
  simple cycles of at most :data:`MAX_RING_SIZE` atoms;
* the bonds, angles, proper torsions and impropers (one per carbon or nitrogen
  with three neighbors, center first) with the directed edges along each term;
* the classes of topologically equivalent atoms and bonds (symmetric atoms of the
  graph with every bond made single and every charge removed);
* the pairs separated by one to four bonds, and the torsions through linear
  centers (sp carbon, two-coordinated N+), whose barriers vanish.

RDKit is an optional dependency (``pip install "xnns[ffnn]"``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

#: Largest ring whose bonds count as ring bonds.
MAX_RING_SIZE = 8
#: Bond orders of the bond-order feature, in feature order.
BOND_ORDERS = (1.0, 2.0, 3.0, 1.5)


def _chem():
    try:
        from rdkit import Chem
    except ImportError as err:  # pragma: no cover - environment dependent
        raise ImportError("molecular graphs need RDKit: pip install rdkit "
                          "(or the 'ffnn' extra of xnn)") from err
    return Chem


def sanitized(mol):
    """A sanitized copy of ``mol`` with RDKit's aromaticity model."""
    Chem = _chem()
    m = Chem.Mol(mol)
    Chem.SanitizeMol(m, Chem.SANITIZE_ALL ^ Chem.SANITIZE_SETAROMATICITY)
    Chem.SetAromaticity(m, Chem.AromaticityModel.AROMATICITY_RDKIT)
    return m


def _smiles(mol) -> str:
    """Canonical SMILES of ``mol`` without hydrogens or atom maps."""
    Chem = _chem()
    m = Chem.Mol(mol)
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    m = Chem.RemoveAllHs(m, sanitize=False)
    return Chem.MolToSmiles(m, isomericSmiles=True, canonical=True)


# Charge-separated forms of hypervalent groups: (query, edits). Each edit is
# ("charge", query atom, new formal charge) or ("order", atom, atom, bond type).
_GROUPS = (
    ("[!O][S+0;D3](=[O])[!O]", (("charge", 1, 1), ("charge", 2, -1), ("order", 1, 2, "SINGLE"))),
    ("[!O][S+1;H1;D3](=[O])[!O]", (("charge", 1, 1), ("charge", 2, -1), ("order", 1, 2, "SINGLE"))),
    ("[O]=[S+0;D2]-[#6]", (("charge", 1, 1),)),
    ("[N]=[S;X4]([O-])=[O]", (("charge", 0, -1), ("order", 0, 1, "SINGLE"), ("charge", 2, 0),
                               ("order", 1, 2, "DOUBLE"))),
    ("[C-;D2,D3]-[N+2;D3]-[O-;D1]", (("charge", 0, 0), ("order", 0, 1, "DOUBLE"), ("charge", 1, 1))),
    ("[N-;D2,D3]-[N+2;D3]-[O-;D1]", (("charge", 0, 0), ("order", 0, 1, "DOUBLE"), ("charge", 1, 1))),
    ("[*][N-][N+]#[N]", (("charge", 1, 0), ("order", 1, 2, "DOUBLE"), ("order", 2, 3, "DOUBLE"),
                         ("charge", 3, -1))),
)


def standardize_groups(mol):
    """Write the hypervalent groups of ``mol`` in charge-separated form.

    One match at a time is rewritten until a group no longer matches, group by
    group in a fixed order; the atoms and their order are unchanged.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        A sanitized molecule.

    Returns
    -------
    rdkit.Chem.Mol
        The rewritten molecule (a copy), with its property cache updated.
    """
    Chem = _chem()
    m = Chem.RWMol(mol)
    for smarts, edits in _GROUPS:
        query = Chem.MolFromSmarts(smarts)
        while True:
            m.UpdatePropertyCache(strict=False)
            hits = m.GetSubstructMatches(query, uniquify=False, maxMatches=1)
            if not hits:
                break
            hit = hits[0]
            for edit in edits:
                if edit[0] == "charge":
                    atom = m.GetAtomWithIdx(hit[edit[1]])
                    atom.SetFormalCharge(edit[2])
                    atom.SetNoImplicit(True)
                else:
                    bond = m.GetBondBetweenAtoms(hit[edit[1]], hit[edit[2]])
                    bond.SetBondType(getattr(Chem.BondType, edit[3]))
                    bond.SetIsAromatic(False)
    out = m.GetMol()
    out.UpdatePropertyCache()
    return out


def _standardized(mol):
    Chem = _chem()
    m = standardize_groups(sanitized(mol))
    m.UpdatePropertyCache(True)
    m = sanitized(m)
    Chem.Cleanup(m)
    return m


def canonical_resonance(mol):
    """The resonance structure of ``mol`` with the fewest formal charges.

    The structures of RDKit's resonance enumeration with the molecule's total
    charge are narrowed to those with the fewest charged atoms, then the smallest
    sum of absolute charges. The input is kept when it is among them; otherwise the
    one with the smallest canonical SMILES is taken.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        A sanitized molecule.

    Returns
    -------
    rdkit.Chem.Mol
        The chosen structure (a copy, atom order unchanged).
    """
    Chem = _chem()
    supplier = Chem.ResonanceMolSupplier(mol, 0)
    supplier.SetNumThreads(1)
    total = Chem.GetFormalCharge(mol)
    found = [m for m in supplier if m is not None and Chem.GetFormalCharge(m) == total]
    if not found:
        return Chem.Mol(mol)
    nnz = [sum(a.GetFormalCharge() != 0 for a in m.GetAtoms()) for m in found]
    found = [m for m, k in zip(found, nnz) if k == min(nnz)]
    tot = [sum(abs(a.GetFormalCharge()) for a in m.GetAtoms()) for m in found]
    found = [m for m, k in zip(found, tot) if k == min(tot)]
    by_smiles = {}
    for m in found:
        s = sanitized(m)
        by_smiles[_smiles(s)] = s
    if _smiles(mol) in by_smiles:
        return Chem.Mol(mol)
    return Chem.Mol(by_smiles[min(by_smiles)])


def prepare_molecule(mol):
    """The standardized structure the graph of ``mol`` is built from.

    Stereochemistry and atom maps are dropped, the hypervalent groups are
    written in charge-separated form and the canonical resonance structure is
    chosen (:func:`canonical_resonance`). Hydrogens must be explicit atoms.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        The molecule, one connected component, explicit hydrogens.

    Returns
    -------
    rdkit.Chem.Mol
        The standardized molecule (atom order unchanged).

    Raises
    ------
    ValueError
        If the molecule has implicit hydrogens or several fragments.
    """
    Chem = _chem()
    m = Chem.Mol(mol)
    m.RemoveAllConformers()
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
        a.SetIsotope(0)
    Chem.RemoveStereochemistry(m)
    m.UpdatePropertyCache(strict=False)
    if any(a.GetNumImplicitHs() for a in m.GetAtoms()):
        raise ValueError("hydrogens must be explicit atoms (Chem.AddHs)")
    if len(Chem.GetMolFrags(m)) > 1:
        raise ValueError("one molecule at a time: the structure has several fragments")
    m = _standardized(m)
    m = _standardized(canonical_resonance(m))
    return m


def _distances(adj: list, n: int, limit: int) -> list:
    """Shortest-path distances up to ``limit`` bonds from every atom, as dicts."""
    out = []
    for s in range(n):
        seen, front = {s: 0}, [s]
        for d in range(1, limit + 1):
            nxt = []
            for a in front:
                for b in adj[a]:
                    if b not in seen:
                        seen[b] = d
                        nxt.append(b)
            front = nxt
        out.append(seen)
    return out


def _cycle_through(adj: list, a: int, b: int, limit: int) -> int:
    """Length of the shortest cycle through bond ``(a, b)`` of at most ``limit``
    atoms, 0 if there is none."""
    seen, front = {a}, [a]
    for d in range(1, limit):
        nxt = []
        for x in front:
            for y in adj[x]:
                if x == a and y == b:
                    continue
                if y == b:
                    return d + 1
                if y not in seen:
                    seen.add(y)
                    nxt.append(y)
        front = nxt
    return 0


def _canonical(t: tuple) -> tuple:
    """The orientation of a chain (bond, angle, torsion) whose first differing end
    atom is the smaller."""
    for i in range(len(t) // 2):
        if t[i] != t[-i - 1]:
            return t if t[i] < t[-i - 1] else t[::-1]
    return t


def _pairs(rows) -> np.ndarray:
    return np.asarray(sorted(rows), dtype=np.int64).reshape(-1, 2)


@dataclass
class MolecularGraph:
    """The graph of one molecule as read by a graph-parameterized force field.

    Index arrays are ``int64``; directed edge ``2 k`` runs along bond ``k`` from its
    lower to its higher atom index, edge ``2 k + 1`` back.

    Attributes
    ----------
    atomic_numbers, formal_charges, degree, ring_degree, min_ring : numpy.ndarray
        Per-atom features ``(n,)`` (``min_ring`` 0 outside rings).
    bonds : numpy.ndarray
        ``(m, 2)``, sorted, lower index first.
    bond_in_ring, bond_order : numpy.ndarray
        Per-bond features ``(m,)``; the order as an index into :data:`BOND_ORDERS`.
    angles, propers, impropers : numpy.ndarray
        ``(a, 3)``, ``(p, 4)`` and ``(i, 4)`` (impropers center first, the other
        three sorted).
    angle_edges, proper_edges, improper_edges : numpy.ndarray
        The directed edges along each term: ``(a, 2)``, ``(p, 3)`` and ``(i, 3)``
        (impropers from the center to each neighbor).
    proper_linear : numpy.ndarray
        ``(p,)`` bool: torsions about a linear center.
    hypervalent : numpy.ndarray
        ``(n,)`` bool: phosphorus or sulfur with five or more neighbors.
    atom_class, edge_class : numpy.ndarray
        Equivalence classes ``(n,)`` and ``(2 m,)``: the index of the first member.
    pairs12, pairs13, pairs14, pairs15 : numpy.ndarray
        Atom pairs ``(k, 2)`` (lower index first) whose shortest path has one to
        four bonds.
    smiles : str
        Canonical SMILES of the standardized structure.
    """

    atomic_numbers: np.ndarray
    formal_charges: np.ndarray
    degree: np.ndarray
    ring_degree: np.ndarray
    min_ring: np.ndarray
    bonds: np.ndarray
    bond_in_ring: np.ndarray
    bond_order: np.ndarray
    angles: np.ndarray
    propers: np.ndarray
    impropers: np.ndarray
    angle_edges: np.ndarray
    proper_edges: np.ndarray
    improper_edges: np.ndarray
    proper_linear: np.ndarray
    hypervalent: np.ndarray
    atom_class: np.ndarray
    edge_class: np.ndarray
    pairs12: np.ndarray
    pairs13: np.ndarray
    pairs14: np.ndarray
    pairs15: np.ndarray
    smiles: str = field(default="")

    @property
    def n_atoms(self) -> int:
        """Number of atoms."""
        return int(self.atomic_numbers.shape[0])

    @property
    def total_charge(self) -> int:
        """Net formal charge."""
        return int(self.formal_charges.sum())


def _linear_center(mol, i: int, ends: tuple) -> bool:
    """Whether atom ``i`` is a linear torsion center between atoms ``ends``."""
    a = mol.GetAtomWithIdx(i)
    if a.GetAtomicNum() == 6 and a.GetDegree() == 2:
        return not (a.IsInRingSize(5) or a.IsInRingSize(6))
    if a.GetAtomicNum() == 7 and a.GetDegree() == 2 and a.GetFormalCharge() == 1:
        for j in ends:
            b = mol.GetAtomWithIdx(j)
            if b.GetAtomicNum() == 7 and b.GetTotalNumHs() == 1:
                return False
        return True
    return False


def molecular_graph(mol, *, prepare: bool = True) -> MolecularGraph:
    """Build the :class:`MolecularGraph` of an RDKit molecule.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        One molecule with explicit hydrogens, atoms in the order of the
        coordinates it will be evaluated on.
    prepare : bool, optional
        Standardize the structure first (:func:`prepare_molecule`); default
        ``True``. ``False`` takes ``mol`` as already standardized.

    Returns
    -------
    MolecularGraph
        The features and term lists.

    Raises
    ------
    ValueError
        For implicit hydrogens, several fragments or a bond type other than
        single, double, triple or aromatic.
    """
    Chem = _chem()
    m = prepare_molecule(mol) if prepare else mol
    n = m.GetNumAtoms()
    z = np.array([a.GetAtomicNum() for a in m.GetAtoms()], dtype=np.int64)
    fc = np.array([a.GetFormalCharge() for a in m.GetAtoms()], dtype=np.int64)
    adj = [sorted(nb.GetIdx() for nb in a.GetNeighbors()) for a in m.GetAtoms()]

    bonds, orders = [], {}
    for b in m.GetBonds():
        i, j = sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        order = b.GetBondTypeAsDouble()
        if order not in BOND_ORDERS:
            raise ValueError(f"bond {i}-{j} has unsupported bond type {b.GetBondType()}")
        bonds.append((i, j))
        orders[(i, j)] = BOND_ORDERS.index(order)
    bonds.sort()
    edge = {}
    for k, (i, j) in enumerate(bonds):
        edge[(i, j)], edge[(j, i)] = 2 * k, 2 * k + 1

    cycle = {bd: _cycle_through(adj, bd[0], bd[1], MAX_RING_SIZE) for bd in bonds}
    in_ring = np.array([cycle[bd] > 0 for bd in bonds], dtype=np.int64)
    ring_degree = np.zeros(n, dtype=np.int64)
    min_ring = np.zeros(n, dtype=np.int64)
    for (i, j), size in cycle.items():
        if size:
            for a in (i, j):
                ring_degree[a] += 1
                min_ring[a] = size if min_ring[a] == 0 else min(min_ring[a], size)

    angles = sorted({(a, c, b) if a < b else (b, c, a)
                     for c in range(n) for x, a in enumerate(adj[c]) for b in adj[c][x + 1:]})
    propers = set()
    for b, c in bonds:
        for a in adj[b]:
            for d in adj[c]:
                if a != c and d != b and a != d:
                    propers.add(_canonical((a, b, c, d)))
    propers = sorted(propers)
    impropers = sorted((c, *adj[c]) for c in range(n) if z[c] in (6, 7) and len(adj[c]) == 3)

    linear = np.array([_linear_center(m, p[1], (p[0], p[2])) or _linear_center(m, p[2], (p[3], p[1]))
                       for p in propers], dtype=bool)

    work = Chem.Mol(m)
    for b in work.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetIsAromatic(False)
        b.SetIsConjugated(False)
    for a in work.GetAtoms():
        a.SetFormalCharge(0)
        a.SetIsAromatic(False)
        a.SetIsotope(0)
        a.SetHybridization(Chem.HybridizationType.S)
    rank = list(Chem.CanonicalRankAtoms(work, breakTies=False, includeChirality=False))
    first: dict = {}
    atom_class = np.array([first.setdefault(r, i) for i, r in enumerate(rank)], dtype=np.int64)
    first = {}
    directed = [e for i, j in bonds for e in ((i, j), (j, i))]
    edge_class = np.array([first.setdefault(tuple(sorted((rank[i], rank[j]))), k)
                           for k, (i, j) in enumerate(directed)], dtype=np.int64)

    dist = _distances(adj, n, 4)
    shells = {d: set() for d in (1, 2, 3, 4)}
    for i in range(n):
        for j, d in dist[i].items():
            if i < j and d in shells:
                shells[d].add((i, j))

    def arr(rows, width):
        return np.asarray(rows, dtype=np.int64).reshape(-1, width)

    return MolecularGraph(
        atomic_numbers=z, formal_charges=fc,
        degree=np.array([len(x) for x in adj], dtype=np.int64),
        ring_degree=ring_degree, min_ring=min_ring,
        bonds=arr(bonds, 2), bond_in_ring=in_ring,
        bond_order=np.array([orders[bd] for bd in bonds], dtype=np.int64),
        angles=arr(angles, 3), propers=arr(propers, 4), impropers=arr(impropers, 4),
        angle_edges=arr([(edge[(a, b)], edge[(b, c)]) for a, b, c in angles], 2),
        proper_edges=arr([(edge[(a, b)], edge[(b, c)], edge[(c, d)]) for a, b, c, d in propers], 3),
        improper_edges=arr([(edge[(c, x)], edge[(c, y)], edge[(c, w)]) for c, x, y, w in impropers], 3),
        proper_linear=linear,
        hypervalent=np.array([a.GetAtomicNum() in (15, 16) and a.GetDegree() >= 5 for a in m.GetAtoms()],
                             dtype=bool),
        atom_class=atom_class, edge_class=edge_class,
        pairs12=_pairs(shells[1]), pairs13=_pairs(shells[2]), pairs14=_pairs(shells[3]),
        pairs15=_pairs(shells[4]), smiles=_smiles(m))


def split_molecules(mol) -> list:
    """The connected components of ``mol`` with their atom indices.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        A structure of one or more molecules.

    Returns
    -------
    list of (rdkit.Chem.Mol, list of int)
        Each molecule (atoms in their original relative order) and the indices
        of its atoms in ``mol``.
    """
    Chem = _chem()
    frags = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
    mols = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    out = []
    for atoms, frag in zip(frags, mols):
        frag.UpdatePropertyCache(strict=False)
        out.append((frag, list(atoms)))
    return out


def graph_key(mol) -> Optional[str]:
    """A key identifying the topology of ``mol`` with its atom order (mapped SMILES)."""
    Chem = _chem()
    m = Chem.Mol(mol)
    for a in m.GetAtoms():
        a.SetAtomMapNum(a.GetIdx() + 1)
    return Chem.MolToSmiles(m, canonical=True, allHsExplicit=False)


__all__ = ["MAX_RING_SIZE", "BOND_ORDERS", "MolecularGraph", "molecular_graph", "prepare_molecule",
           "canonical_resonance", "standardize_groups", "sanitized", "split_molecules", "graph_key"]
