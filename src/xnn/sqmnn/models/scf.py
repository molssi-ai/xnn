"""Self-consistent field of the NDDO Hamiltonian, batched over structures.

The basis is orthonormal (zero differential overlap), so the Roothaan-Hall
equations are ordinary eigenvalue problems ``F C = C e``. A batch of
structures is solved together: every structure's matrices are padded to the
largest basis, padding orbitals get a large diagonal so they stay empty, and
the Fock matrix is assembled from atom blocks:

* one-center: ``F_mu nu += sum_la si P_la si [(mu nu|la si) - 1/2 (mu la|nu si)]``
  over the atom's own orbitals, plus the Coulomb term ``sum_B (mu nu|la si)_AB
  P_la si`` of every other atom;
* two-center: ``F_mu la -= 1/2 sum_nu si (mu nu|la si)_AB P_nu si``;

(spin-unrestricted: the Coulomb term from the total density and the
exchange terms, without the factor 1/2, from each spin density).
Convergence is accelerated with Pulay's DIIS on the commutator ``FP - PF``.

:class:`SCFDensity` returns the converged densities as a differentiable
function of the core Hamiltonian and the two-electron integrals: its
backward pass solves the coupled-perturbed (linear-response) equations, so
properties that depend on the density (dipole moments, charges, forces of
force-trained parameters) have exact parameter gradients.
"""
import warnings
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import torch
from torch import Tensor

PAD_ENERGY = 1.0e4   # diagonal of the padding orbitals (eV)
# below this many pairs, all pairs form one group padded to the largest orbital count (fewer, larger operations),
# per device type (measured crossovers; a GPU is bound by kernel launches longer)
GROUP_MIN_PAIRS = {"cpu": 1500, "cuda": 15000}
# below this basis size the iterations diagonalize in full (cheaper than the orbital rotations), per device type
PSEUDO_MIN_BASIS = {"cpu": 110, "cuda": 48}
# below this largest density change the iterations rotate the previous orbitals (on a GPU a full diagonalization
# costs several rotated iterations, so it rotates earlier)
PSEUDO_THRESHOLD = {"cpu": 0.05, "cuda": 0.5}
# whether a warm start (the previous geometry's orbitals given) rotates them from its first iteration instead of
# first diagonalizing the Fock matrix of the previous density
WARM_ORBITALS = {"cpu": False, "cuda": True}
# evaluations of a batch composition launched before its CUDA graphs are captured (a capture costs about as much
# as ten replays save, so compositions met once or twice never pay for one)
CAPTURE_AFTER = 2


def eigh(A: Tensor) -> Tuple[Tensor, Tensor]:
    """``torch.linalg.eigh`` of symmetric matrices; on a GPU those of up to 200 basis functions on the host
    (LAPACK, with the two transfers, is faster there than cusolver's solver of one matrix at a time; it is
    never inside a CUDA graph)."""
    if A.is_cuda and A.shape[-1] <= 200:
        e, v = torch.linalg.eigh(A.cpu())
        return e.to(A.device), v.to(A.device)
    return torch.linalg.eigh(A)


def _per_device(limit, device: torch.device) -> int:
    return limit.get(device.type, limit["cuda"]) if isinstance(limit, dict) else limit


class Layout:
    """Index maps between atom blocks and the padded dense matrices.

    Attributes
    ----------
    n_struct, n_basis : int
        Structures in the batch and the padded basis size.
    k : int
        Orbitals per atom block (1, 4 or 9).
    atom_struct : Tensor
        Structure of every atom ``(N,)``.
    ao : Tensor
        Basis index of every atom orbital ``(N, k)``; ``n_basis`` (the sink
        row) for the orbitals an atom does not have.
    pair_a, pair_b : Tensor
        Atoms of every pair ``(P,)``.
    valid : Tensor
        Real (non-padding) basis functions ``(B, n_basis)``.
    group_min_pairs : int
        Below this many pairs, one padded group of all pairs (:meth:`pair_groups`); by default
        :data:`GROUP_MIN_PAIRS` of the device.
    """

    def __init__(self, n_struct: int, n_basis: int, k: int, atom_struct: Tensor, ao: Tensor, pair_a: Tensor,
                 pair_b: Tensor, valid: Tensor, group_min_pairs: int = -1):
        self.n_struct = n_struct
        self.n_basis = n_basis
        self.k = k
        self.atom_struct = atom_struct
        self.ao = ao
        self.pair_a = pair_a
        self.pair_b = pair_b
        self.valid = valid
        self.group_min_pairs = group_min_pairs
        self.padded_pairs = False
        # caches (typed for TorchScript, which takes attribute types from these values)
        self._groups = torch.jit.annotate(Optional[List[Tuple[int, int, Tensor, Tensor, Tensor]]], None)
        self._group_maps = torch.jit.annotate(Optional[List[Dict[str, Tensor]]], None)
        self._dense_grouped = torch.jit.annotate(Optional[Tensor], None)
        self._dense_padded = torch.jit.annotate(Optional[Tensor], None)
        self._atom_index = torch.jit.annotate(Optional[Tensor], None)

    def flat(self, struct: Tensor, rows: Tensor, cols: Tensor) -> Tensor:
        n = self.n_basis + 1
        return (struct * n + rows) * n + cols

    def gather_atom(self, M: Tensor, with_sink: bool = False) -> Tensor:
        """Diagonal atom blocks ``(N, k, k)`` of padded matrices ``(B, n, n)`` (``(B, n + 1, n + 1)`` with the
        sink row and column of :meth:`_with_sink` when ``with_sink``)."""
        full = (M if with_sink else self._with_sink(M)).reshape(-1)
        idx = self._atom_index
        if idx is None:
            idx = self.flat(self.atom_struct[:, None, None], self.ao[:, :, None], self.ao[:, None, :]).reshape(-1)
            self._atom_index = idx
        return full.index_select(0, idx).reshape(-1, self.k, self.k)

    def gather_pair(self, M: Tensor) -> Tensor:
        """Off-diagonal blocks ``(P, k, k)`` (rows on atom A, columns on atom B)."""
        full = self._with_sink(M).reshape(-1)
        s = self.atom_struct[self.pair_a]
        idx = self.flat(s[:, None, None], self.ao[self.pair_a][:, :, None],
                        self.ao[self.pair_b][:, None, :])
        return full.index_select(0, idx.reshape(-1)).reshape(idx.shape)

    def scatter(self, atom_blocks: Tensor, pair_blocks: Tensor) -> Tensor:
        """Padded symmetric matrices from atom blocks ``(N, k, k)`` and (A, B) pair blocks ``(P, k, k)``."""
        src = torch.cat([atom_blocks.reshape(-1), pair_blocks.reshape(-1), atom_blocks.new_zeros(1)])
        out = src.index_select(0, self._dense_map(False))
        return out.reshape(self.n_struct, self.n_basis, self.n_basis)

    def scatter_groups(self, atom_blocks: Tensor, pair_blocks: List[Tensor]) -> Tensor:
        """:meth:`scatter` with one ``(P_g, ka, kb)`` block per group of :meth:`pair_groups`."""
        parts = [atom_blocks.reshape(-1)]
        for b in pair_blocks:
            parts.append(b.reshape(-1))
        parts.append(atom_blocks.new_zeros(1))
        out = torch.cat(parts).index_select(0, self._dense_map(True))
        return out.reshape(self.n_struct, self.n_basis, self.n_basis)

    def _dest(self, s: Tensor, rows: Tensor, cols: Tensor) -> Tensor:
        # flat index of (s, rows, cols), -1 for the sink of the orbitals an atom does not have
        nb = self.n_basis
        return torch.where((rows < nb) & (cols < nb), (s * nb + rows) * nb + cols, -1).reshape(-1)

    def _dense_map(self, grouped: bool) -> Tensor:
        """For every element of the dense matrices, its entry in the concatenated atom and pair blocks of
        :meth:`scatter` (the trailing zero where no block reaches). Each element has at most one: the atom blocks
        sit on the diagonal, a pair's blocks (A, B) and (B, A) off it."""
        cached = self._dense_grouped if grouped else self._dense_padded
        if cached is not None:
            return cached
        nb, device = self.n_basis, self.ao.device
        blocks: List[Tuple[Tensor, Tensor, int, int]] = []
        if grouped:
            for ka, kb, _, ia, ib in self.pair_groups():
                blocks.append((ia, ib, ka, kb))
        else:
            blocks.append((self.pair_a, self.pair_b, self.k, self.k))
        d_atom = self._dest(self.atom_struct[:, None, None], self.ao[:, :, None], self.ao[:, None, :])
        d_ab, d_ba = [d_atom], [torch.full_like(d_atom, -1)]
        for ia, ib, ka, kb in blocks:
            s = self.atom_struct[ia][:, None, None]
            ra, rb = self.ao[ia, :ka], self.ao[ib, :kb]
            d_ab.append(self._dest(s, ra[:, :, None], rb[:, None, :]))
            d_ba.append(self._dest(s, rb[:, None, :], ra[:, :, None]))
        total = 0
        for d in d_ab:
            total += d.shape[0]
        src = torch.arange(total, device=device)
        m = torch.full((self.n_struct * nb * nb,), total, dtype=torch.long, device=device)
        for d in (torch.cat(d_ab), torch.cat(d_ba)):
            keep = d >= 0
            m[d[keep]] = src[keep]
        if grouped:
            self._dense_grouped = m
        else:
            self._dense_padded = m
        return m

    def _with_sink(self, M: Tensor) -> Tensor:
        return torch.nn.functional.pad(M, [0, 1, 0, 1])

    def pair_groups(self) -> List[Tuple[int, int, Tensor, Tensor, Tensor]]:
        """The pairs by the orbital counts of their atoms: ``(ka, kb, rows, atoms A, atoms B)`` per group; below
        :attr:`group_min_pairs` pairs one group of all, padded to ``k`` (:attr:`padded_pairs`)."""
        groups = self._groups
        if groups is None:
            n_orb = (self.ao < self.n_basis).sum(1)
            key = n_orb[self.pair_a] * 16 + n_orb[self.pair_b]
            codes = torch.jit.annotate(List[int], torch.unique(key).tolist())
            limit = self.group_min_pairs
            if limit < 0:
                limit = 15000 if key.is_cuda else 1500            # GROUP_MIN_PAIRS
            self.padded_pairs = len(codes) > 1 and self.pair_a.shape[0] < limit
            made: List[Tuple[int, int, Tensor, Tensor, Tensor]] = []
            if self.padded_pairs:
                rows = torch.arange(self.pair_a.shape[0], device=key.device)
                made.append((self.k, self.k, rows, self.pair_a, self.pair_b))
            else:
                for code in codes:
                    rows = (key == code).nonzero().squeeze(-1)
                    made.append((code // 16, code % 16, rows, self.pair_a[rows], self.pair_b[rows]))
            self._groups = made
            return made
        return groups

    def pair_group_maps(self) -> List[Dict[str, Tensor]]:
        """Per group of :meth:`pair_groups`, flat indices into the dense matrices of the blocks ``(A, A)``,
        ``(B, B)`` and ``(A, B)`` of its pairs, and into the atom blocks ``(N, k, k)`` of ``(A, A)`` and
        ``(B, B)``. A padded group indexes matrices with the sink row and column (:meth:`_with_sink`)."""
        maps = self._group_maps
        if maps is None:
            groups = self.pair_groups()
            nb, k = self.n_basis + (1 if self.padded_pairs else 0), self.k
            made: List[Dict[str, Tensor]] = []
            for ka, kb, _, ia, ib in groups:
                s = self.atom_struct[ia][:, None, None]
                oa, ob = self.ao[ia, :ka], self.ao[ib, :kb]
                ra, rb = torch.arange(ka, device=s.device), torch.arange(kb, device=s.device)
                m: Dict[str, Tensor] = {
                    "aa": ((s * nb + oa[:, :, None]) * nb + oa[:, None, :]).reshape(-1),
                    "bb": ((s * nb + ob[:, :, None]) * nb + ob[:, None, :]).reshape(-1),
                    "ab": ((s * nb + oa[:, :, None]) * nb + ob[:, None, :]).reshape(-1),
                    "acc_a": ((ia[:, None, None] * k + ra[:, None]) * k + ra[None, :]).reshape(-1),
                    "acc_b": ((ib[:, None, None] * k + rb[:, None]) * k + rb[None, :]).reshape(-1)}
                made.append(m)
            self._group_maps = made
            return made
        return maps


PackedPairs = List[Tuple[int, int, Dict[str, Tensor], Tensor, Tensor]]


def pack_pairs(layout: Layout, W2: Tensor) -> PackedPairs:
    """The two-center integrals ``(P, k, k, k, k)`` of each pair group at its own size, as ``(ka, kb, maps,
    coulomb, exchange)``: the Coulomb form ``(P_g, ka * ka, kb * kb)`` of ``(mu nu | la si)`` and the exchange
    form ``(P_g, ka * kb, ka * kb)`` (rows ``mu la``, columns ``nu si``), with the index maps of
    :meth:`Layout.pair_group_maps`."""
    out: PackedPairs = []
    groups, group_maps = layout.pair_groups(), layout.pair_group_maps()
    for g in range(len(groups)):
        ka, kb, rows, maps = groups[g][0], groups[g][1], groups[g][2], group_maps[g]
        # one group of all pairs in order (the padded form) is a view
        if layout.padded_pairs:
            Wg = W2[:, :ka, :ka, :kb, :kb]
        elif W2.is_cuda:
            # index_select: its backward adds atomically (that of indexing sorts the indices)
            Wg = W2[:, :ka, :ka, :kb, :kb].index_select(0, rows)
        else:
            Wg = W2[rows, :ka, :ka, :kb, :kb]
        out.append((ka, kb, maps, Wg.reshape(-1, ka * ka, kb * kb),
                    Wg.permute(0, 1, 3, 2, 4).reshape(-1, ka * kb, ka * kb)))
    return out


def pack_one_center(W1: Tensor) -> Dict[str, Tensor]:
    """The one-center integrals ``(N, k, k, k, k)`` as the matrices ``(N, k * k, k * k)`` of the Coulomb
    term ``(mu nu | la si)`` (rows ``mu nu``) and of the exchange term ``(mu la | nu si)``."""
    n, k = W1.shape[0], W1.shape[1]
    out: Dict[str, Tensor] = {"coulomb": W1.reshape(n, k * k, k * k),
                              "exchange": W1.permute(0, 1, 3, 2, 4).reshape(n, k * k, k * k)}
    return out


def two_electron(layout: Layout, one: Dict[str, Tensor], groups: PackedPairs, P_total: Tensor, P_spin: Tensor,
                 exchange: float, closed: bool) -> Tensor:
    """Two-electron part ``G`` of the Fock matrix (padded ``(B, n, n)``).

    Parameters
    ----------
    one : dict
        One-center integrals of every atom packed by :func:`pack_one_center` (the closed-shell matrix is kept
        in it as ``"closed"``).
    groups : list
        Two-center integrals of every pair packed by :func:`pack_pairs`.
    P_total : Tensor
        Total density ``(B, n, n)`` (Coulomb term).
    P_spin : Tensor
        Density of the exchange term ``(B, n, n)``.
    exchange : float
        Weight of the exchange term: 1/2 for the closed-shell Fock matrix with
        ``P_spin = P_total``, 1 for one spin of the unrestricted one.
    closed : bool
        Whether ``P_spin`` is ``P_total`` (the closed-shell form).
    """
    n_atoms, kk = one["coulomb"].shape[0], one["coulomb"].shape[1]
    if layout.padded_pairs and layout.pair_a.shape[0] > 0:
        # the pair blocks index the matrices with the sink row and column: pad once for both
        P_total = layout._with_sink(P_total)
        P_spin = P_total if closed else layout._with_sink(P_spin)
    with_sink = layout.padded_pairs and layout.pair_a.shape[0] > 0
    Paa = layout.gather_atom(P_total, with_sink).reshape(n_atoms, kk, 1)
    if closed:
        # closed shell: one matrix of the Coulomb and exchange terms (kept for the calls of an SCF)
        if "closed" in one:
            M = one["closed"]
        else:
            M = one["coulomb"] - exchange * one["exchange"]
            one["closed"] = M
        atom = torch.bmm(M, Paa)
    else:
        Xaa = layout.gather_atom(P_spin, with_sink).reshape(n_atoms, kk, 1)
        atom = torch.bmm(one["coulomb"], Paa) - exchange * torch.bmm(one["exchange"], Xaa)
    atom = atom.reshape(n_atoms, layout.k, layout.k)
    pair: List[Tensor] = []
    if layout.pair_a.shape[0] > 0:
        Pt, Ps = P_total.reshape(-1), P_spin.reshape(-1)
        flat = atom.reshape(-1)
        # each group at the orbital counts of its atoms
        for ka, kb, maps, Wc, Wx in groups:
            coul_a = torch.bmm(Wc, Pt.index_select(0, maps["bb"]).reshape(-1, kb * kb, 1))
            coul_b = torch.bmm(Pt.index_select(0, maps["aa"]).reshape(-1, 1, ka * ka), Wc)
            flat = (flat.index_add(0, maps["acc_a"], coul_a.reshape(-1))
                    .index_add(0, maps["acc_b"], coul_b.reshape(-1)))
            X = Ps.index_select(0, maps["ab"]).reshape(-1, ka * kb, 1)
            pair.append(-exchange * torch.bmm(Wx, X).reshape(-1, ka, kb))
        atom = flat.reshape(atom.shape)
    # the scatter places each off-diagonal block at (A,B) and its transpose at
    # (B,A); the atom blocks are symmetric already
    return layout.scatter_groups(atom, pair)


def _occupations(eps: Tensor, n_occ: Tensor, weight: float) -> Tensor:
    """Aufbau occupation numbers ``(B, n)``: ``weight`` for the lowest ``n_occ``."""
    n = eps.shape[-1]
    rank = torch.arange(n, device=eps.device).expand_as(eps)
    return torch.where(rank < n_occ.unsqueeze(-1), torch.full_like(eps, weight),
                       torch.zeros_like(eps))


def _fermi_occupations(eps: Tensor, n_occ: Tensor, weight: float, kt: float) -> Tensor:
    """Fermi-Dirac occupations at temperature ``kt`` (eV) holding ``n_occ * weight`` electrons."""
    target = n_occ.to(eps.dtype) * weight
    lo = eps.amin(-1) - 50.0 * kt - 1.0
    hi = eps.amax(-1) + 50.0 * kt + 1.0
    for _ in range(100):
        mu = 0.5 * (lo + hi)
        f = weight * torch.sigmoid((mu.unsqueeze(-1) - eps) / kt)
        more = f.sum(-1) > target
        hi = torch.where(more, mu, hi)
        lo = torch.where(more, lo, mu)
    mu = 0.5 * (lo + hi)
    return weight * torch.sigmoid((mu.unsqueeze(-1) - eps) / kt)


def _bmm(a: Tensor, b: Tensor) -> Tensor:
    """``a @ b`` over the same leading dimensions as one batched product: what ``torch.matmul`` computes, without
    its extra operations per call, which the SCF of a small system feels."""
    return torch.bmm(a.flatten(0, -3), b.flatten(0, -3)).unflatten(0, a.shape[:-2])


def density(C: Tensor, occ: Tensor, m: Optional[int] = None) -> Tensor:
    """``P = C diag(occ) C^T`` (any leading dimensions), over the first ``m`` orbitals when only those can be
    occupied."""
    if m is None:
        # only the orbitals some structure occupies (the lowest ones of an aufbau occupation)
        filled = (occ != 0).reshape(-1, occ.shape[-1]).any(0).nonzero()
        m = int(filled.max()) + 1 if filled.numel() > 0 else 0
    C = C[..., :m]
    return _bmm(C * occ[..., :m].unsqueeze(-2), C.transpose(-1, -2))


class DIIS:
    """Pulay extrapolation of the Fock matrices of a batch: a ring of the last ``size`` Fock matrices and errors
    and their overlap matrix. With ``static`` the count stays on the device and the subspace is the whole ring
    with the empty slots decoupled (no host synchronization, fixed shapes: capturable in a CUDA graph; the form
    of every GPU run); otherwise the count is a host number and the subspace the filled slots. Both solve the
    same equations."""

    def __init__(self, nb: int, fock_shape: List[int], n_error: int, size: int, dtype: torch.dtype,
                 device: torch.device, static: bool = False):
        self.size = size
        self.static = static
        self.n = 0
        self.focks = torch.zeros([nb, size] + fock_shape, dtype=dtype, device=device)
        self.errors = torch.zeros([nb, size, n_error], dtype=dtype, device=device)
        self.gram = torch.zeros([nb, size, size], dtype=dtype, device=device)
        self.count = torch.zeros([], dtype=torch.long, device=device)
        self.eye = torch.eye(size, dtype=dtype, device=device)
        self.slots = torch.arange(size, device=device)

    def reset(self) -> None:
        self.n = 0
        if self.static:
            self.count.zero_()
            self.gram.zero_()
            self.focks.zero_()
            self.errors.zero_()

    def extrapolate(self, F: Tensor, err: Tensor) -> Tensor:
        if self.static:
            return self._static(F, err)
        nb = F.shape[0]
        e = err.reshape(nb, -1)
        slot = self.n % self.size
        self.n += 1
        m = min(self.n, self.size)
        self.errors[:, slot] = e
        self.focks[:, slot] = F
        # the overlaps of the new error with the stored ones (the rest of the Gram matrix is kept)
        row = torch.bmm(self.errors[:, :m], e.unsqueeze(-1)).squeeze(-1)       # (B, m)
        self.gram[:, slot, :m] = row
        self.gram[:, :m, slot] = row
        if m < 2:
            return F
        Bm = self.gram[:, :m, :m]
        scale = Bm.diagonal(dim1=1, dim2=2).abs().amax(-1).clamp(min=1e-30)
        Bm = Bm / scale[:, None, None] + 1e-12 * self.eye[:m, :m]
        # min c^T B c with sum(c) = 1: c = B^-1 1 / (1^T B^-1 1), the solution of the bordered system
        y, info = torch.linalg.solve_ex(Bm, Bm.new_ones((nb, m)))
        c = y / y.sum(-1, keepdim=True)
        out = torch.bmm(c.unsqueeze(1), self.focks[:, :m].reshape(nb, m, -1)).reshape(F.shape)
        return torch.where((info == 0).reshape([-1] + [1] * (F.dim() - 1)), out, F)

    def _static(self, F: Tensor, err: Tensor) -> Tensor:
        nb = F.shape[0]
        e = err.reshape(nb, -1)
        slot = (self.count % self.size).reshape(1)
        self.count.add_(1)
        valid = self.slots < self.count                                      # the stored slots
        self.errors.index_copy_(1, slot, e.unsqueeze(1))
        self.focks.index_copy_(1, slot, F.unsqueeze(1))
        # the overlaps of the new error with the stored ones (the rest of the Gram matrix is kept)
        row = torch.where(valid, torch.bmm(self.errors, e.unsqueeze(-1)).squeeze(-1), 0.0)   # (B, size)
        self.gram.index_copy_(1, slot, row.unsqueeze(1))
        self.gram.index_copy_(2, slot, row.unsqueeze(2))
        pair = valid.unsqueeze(-1) & valid.unsqueeze(-2)
        scale = torch.where(valid, self.gram.diagonal(dim1=1, dim2=2).abs(), 0.0).amax(-1).clamp(min=1e-30)
        # min c^T B c with sum(c) = 1: c = B^-1 1 / (1^T B^-1 1), the solution of the bordered system; the slots
        # not yet filled are decoupled identity rows with a zero right-hand side
        Bm = torch.where(pair, self.gram / scale[:, None, None] + 1e-12 * self.eye, self.eye)
        y, info = torch.linalg.solve_ex(Bm, valid.to(F.dtype).expand(nb, self.size))
        c = y / y.sum(-1, keepdim=True)
        out = torch.bmm(c.unsqueeze(1), self.focks.reshape(nb, self.size, -1)).reshape(F.shape)
        use = (self.count >= 2) & (info == 0)
        return torch.where(use.reshape([-1] + [1] * (F.dim() - 1)), out, F)


class SCFResult(NamedTuple):
    """Converged SCF quantities (detached)."""

    densities: Tensor          # (S, B, n, n): one total density (RHF) or alpha, beta (UHF)
    orbitals: Tensor           # (S, B, n, n)
    energies: Tensor           # (S, B, n) orbital energies, eV
    occupations: Tensor        # (S, B, n)
    iterations: int
    converged: Tensor          # (B,) bool
    electronic: Tensor         # (B,) electronic energy, eV


class SCFLoop:
    """The SCF iteration of a batch as stages on one state, without host synchronization or data-dependent
    shapes (so that a stage can be captured in a CUDA graph and replayed on the same buffers):

    * :meth:`fock_step`: Fock matrices, electronic energy, its change, DIIS extrapolation ``Fx``;
    * :meth:`rotate_step`: the orbitals of ``Fx`` by rotation of the previous ones, the density and its change
      (and the flag of a near-degenerate occupied-virtual pair, which needs a diagonalization instead);
    * :meth:`diagonalize` (eigenvectors, outside any graph) and :meth:`occupy`: aufbau occupations, density
      and its change;
    * :meth:`commit`: the new density, orbitals and occupations become the state.

    With ``static`` the stages write into fixed buffers (copies), otherwise they rebind the attributes.
    """

    def __init__(self, layout: Layout, H: Tensor, W1: Dict[str, Tensor], W2: PackedPairs, n_occ: Tensor,
                 weight: float, unrestricted: bool, diis_size: int, filled: int, blocks: List[Optional[int]],
                 static: bool = False):
        self.layout = layout
        self.H = H
        self.W1 = W1
        self.W2 = W2
        self.n_occ = n_occ
        self.weight = weight
        self.unrestricted = unrestricted
        self.filled = filled
        self.blocks = blocks
        self.static = static
        S = 2 if unrestricted else 1
        B, n = H.shape[0], H.shape[-1]
        z = H.new_zeros([S, B, n, n])
        self.P = z
        self.C = z.clone()
        self.Fx = z.clone()
        self.P_new = z.clone()
        self.C_new = z.clone()
        self.occ = H.new_zeros([S, B, n])
        self.occ_new = self.occ.clone()
        self.eps = self.occ.clone()
        self.aufbau = torch.jit.annotate(Optional[Tensor], None)
        self.e_old = H.new_full([B], float("inf"))
        self.dP = H.new_zeros([B])
        self.dE = H.new_zeros([B])
        self.flag = torch.zeros([], dtype=torch.bool, device=H.device)
        # on a GPU the device-side form (that of the graphs), so that replays and launches compute the same numbers
        self.diis = DIIS(B, [S, n, n], S * n * n, diis_size, H.dtype, H.device, static or H.is_cuda)

    def _put(self, buffer: Tensor, value: Tensor) -> Tensor:
        """The new value of a state tensor: copied into its buffer when static (and the buffer returned)."""
        if self.static:
            buffer.copy_(value)
            return buffer
        return value

    def start(self, P: Tensor, C: Optional[Tensor] = None, occ: Optional[Tensor] = None) -> None:
        """Start from the density ``P``, and from the orbitals ``C`` with occupations ``occ`` where given."""
        self.P = self._put(self.P, P)
        if C is not None and occ is not None:
            self.C = self._put(self.C, C)
            self.occ = self._put(self.occ, occ)
        self.e_old.fill_(float("inf"))
        self.diis.reset()

    def fock_step(self) -> None:
        F = _fock(self.layout, self.H, self.W1, self.W2, self.P, self.unrestricted)
        energy = _electronic_energy(self.H, F, self.P, self.unrestricted)
        # F and P are symmetric: P F = (F P)^T
        FP = _bmm(F, self.P)
        err = FP - FP.transpose(-1, -2)
        self.Fx = self._put(self.Fx, self.diis.extrapolate(F.transpose(0, 1), err.transpose(0, 1)).transpose(0, 1))
        self.dE = self._put(self.dE, (energy - self.e_old).abs())
        self.e_old.copy_(energy)

    def _changes(self, C: Tensor, occ: Tensor, m: Optional[int]) -> None:
        P_new = density(C, occ, m)
        self.C_new = self._put(self.C_new, C)
        self.occ_new = self._put(self.occ_new, occ)
        self.P_new = self._put(self.P_new, P_new)
        self.dP = self._put(self.dP, (P_new - self.P).abs().amax(dim=(0, 2, 3)))

    def rotate_step(self) -> None:
        C, flag = _pseudo_diagonalize(self.Fx, self.C, self.occ > 0, self.blocks)
        self.flag.copy_(flag)
        self._changes(C, self.occ, self.filled)

    def diagonalize(self) -> None:
        eps, C = eigh(self.Fx)
        self.eps = self._put(self.eps, eps)
        self.C_new = self._put(self.C_new, C)

    def occupy(self, kt: float = 0.0) -> None:
        if kt > 0.0:
            self._changes(self.C_new, _fermi_occupations(self.eps, self.n_occ, self.weight, kt), None)
            return
        # the eigenvalues come in ascending order, so the aufbau occupations are the same at every iteration
        occ = self.aufbau
        if occ is None:
            occ = _occupations(self.eps, self.n_occ, self.weight)
            self.aufbau = occ
        self._changes(self.C_new, occ, self.filled)

    def commit(self) -> None:
        self.P = self._put(self.P, self.P_new)
        self.C = self._put(self.C, self.C_new)
        self.occ = self._put(self.occ, self.occ_new)


class SCFGraphs:
    """CUDA graphs of the stages of :class:`SCFLoop` for one batch composition (shapes, layout, electrons).

    The loop runs on fixed buffers; :meth:`load` copies a new evaluation's core Hamiltonian, integrals and guess
    into them and :meth:`run` replays a stage instead of launching its kernels, so an iteration costs a few
    replays and one host synchronization. The eigenvalue problem (not capturable) runs between the replays.
    Replays compute exactly what the stages compute when launched.
    """

    STAGES = ("fock_step", "rotate_step", "occupy", "commit")

    def __init__(self, layout: Layout, H: Tensor, W1: dict, W2: PackedPairs, n_occ: Tensor, weight: float,
                 unrestricted: bool, diis_size: int, filled: int, blocks):
        self.H = H.clone()
        self.W1 = {key: v.clone() for key, v in W1.items()}
        self.W2 = [(ka, kb, maps, Wc.clone(), Wx.clone()) for ka, kb, maps, Wc, Wx in W2]
        self.loop = SCFLoop(layout, self.H, self.W1, self.W2, n_occ.clone(), weight, unrestricted, diis_size,
                            filled, blocks, static=True)
        self.graphs: Dict[str, torch.cuda.CUDAGraph] = {}
        pool = torch.cuda.graph_pool_handle()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            # warm-up launches on a side stream before the capture
            self.loop.start(self.loop.P)
            for stage in self.STAGES:
                getattr(self.loop, stage)()
        torch.cuda.current_stream().wait_stream(side)
        for stage in self.STAGES:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                getattr(self.loop, stage)()
            self.graphs[stage] = g

    def load(self, H: Tensor, W1: dict, W2: PackedPairs, P: Tensor, C: Optional[Tensor] = None,
             occ: Optional[Tensor] = None) -> SCFLoop:
        """Copy the inputs of a new evaluation into the buffers and start the loop from the density ``P`` (and
        the orbitals ``C`` with occupations ``occ`` where given)."""
        self.H.copy_(H)
        for key, v in self.W1.items():
            if key in W1:
                v.copy_(W1[key])
            else:
                # the cached closed-shell matrix of two_electron
                v.copy_(W1["coulomb"] - 0.5 * W1["exchange"])
        for (_, _, _, Wc, Wx), (_, _, _, nc, nx) in zip(self.W2, W2):
            Wc.copy_(nc)
            Wx.copy_(nx)
        self.loop.start(P, C, occ)
        return self.loop

    def run(self, stage: str) -> None:
        self.graphs[stage].replay()


class SCFGraphCache:
    """:class:`SCFGraphs` per batch composition, captured after :data:`CAPTURE_AFTER` launched evaluations of a
    composition and kept for the ``capacity`` most recent ones."""

    def __init__(self, capacity: int = 8):
        self.capacity = capacity
        self.entries: Dict[tuple, SCFGraphs] = {}
        self.seen: Dict[tuple, int] = {}

    def get(self, key: tuple, build) -> Optional[SCFGraphs]:
        if key in self.entries:
            self.entries[key] = self.entries.pop(key)        # most recent last
            return self.entries[key]
        # counted among the recent compositions only: one that recurs only after many others (a data set cycled
        # through in training) is never captured, since it would be evicted before its next use
        self.seen[key] = self.seen.pop(key, 0) + 1
        while len(self.seen) > 2 * self.capacity:
            self.seen.pop(next(iter(self.seen)))
        if self.seen[key] <= CAPTURE_AFTER:
            return None
        graphs = build()
        self.entries[key] = graphs
        while len(self.entries) > self.capacity:
            self.entries.pop(next(iter(self.entries)))
        return graphs

    def clear(self) -> None:
        self.entries.clear()
        self.seen.clear()

    # graphs belong to one process and device: copies and pickles of a model start with an empty cache
    def __getstate__(self):
        return {"capacity": self.capacity, "entries": {}, "seen": {}}

    def __deepcopy__(self, memo):
        return SCFGraphCache(self.capacity)


class _GraphedLoop:
    """The stages of an :class:`SCFLoop` replayed from its :class:`SCFGraphs` (the state is the loop's)."""

    def __init__(self, graphs: "SCFGraphs"):
        self.graphs = graphs
        self.loop = graphs.loop

    def __getattr__(self, name):
        return getattr(self.loop, name)

    def fock_step(self) -> None:
        self.graphs.run("fock_step")

    def rotate_step(self) -> None:
        self.graphs.run("rotate_step")

    def diagonalize(self) -> None:
        self.loop.diagonalize()

    def occupy(self, kt: float = 0.0) -> None:
        if kt > 0.0:
            self.loop.occupy(kt)
        else:
            self.graphs.run("occupy")

    def commit(self) -> None:
        self.graphs.run("commit")


def run_scf(layout: Layout, H: Tensor, W1: Tensor, W2: Union[Tensor, PackedPairs], n_alpha: Tensor, n_beta: Tensor,
            unrestricted: bool, guess: Tensor, max_iter: int = 300, tol_density: float = 1e-9,
            tol_energy: float = 1e-10, diis_size: int = 8, smearing: float = 2.0,
            smear_steps: int = 25, spin_guess: Optional[Tensor] = None,
            pseudo_diagonalization: Optional[float] = None, graph_cache: Optional[SCFGraphCache] = None,
            graph_key: tuple = (), orbitals: Optional[Tensor] = None,
            occupations: Optional[Tensor] = None) -> SCFResult:
    """Solve the SCF equations of a batch (no gradient tracking).

    Parameters
    ----------
    layout : Layout
        Index maps of the batch.
    H : Tensor
        Core Hamiltonian ``(B, n, n)`` (padding orbitals on a large diagonal).
    W1, W2 : Tensor
        One- and two-center integrals (``W2`` also packed by :func:`pack_pairs`).
    n_alpha, n_beta : Tensor
        Electrons of each spin ``(B,)``.
    unrestricted : bool
        UHF (two spin densities) instead of closed-shell RHF.
    guess : Tensor
        Initial total density ``(B, n, n)``.
    max_iter, tol_density, tol_energy, diis_size
        Iteration limit, convergence thresholds (largest density change,
        energy change in eV) and DIIS subspace size.
    smearing, smear_steps
        The first ``smear_steps`` iterations use Fermi-Dirac occupations
        with a temperature falling linearly from ``smearing`` (eV) to zero,
        which steers the aufbau occupation to the ground state (open shells,
        near-degenerate frontier orbitals); the converged density has integer
        occupations.
    spin_guess : Tensor, optional
        Initial spin densities ``(2, B, n, n)`` of an unrestricted calculation
        (in place of the split of ``guess``).
    pseudo_diagonalization : float, optional
        Below this largest density change (and with integer occupations) the
        iterations rotate the previous orbitals by first-order occupied-virtual
        mixing in place of a full diagonalization (by default
        :data:`PSEUDO_THRESHOLD` of the device; 0 disables; small bases, below
        :data:`PSEUDO_MIN_BASIS`, always diagonalize). The converged
        density is checked against the aufbau occupation of a full
        diagonalization, and the iterations go on with full ones if they differ.
    graph_cache, graph_key : optional
        CUDA graphs of the iteration stages (see :class:`SCFGraphs`), looked up by ``graph_key`` (the batch
        composition) in ``graph_cache`` for a run on a GPU without smeared iterations; the stages are replayed
        instead of launched, with the same results.
    orbitals, occupations : Tensor, optional
        The converged orbitals ``(S, B, n, n)`` and occupations ``(S, B, n)`` of the previous geometry of a warm
        start (``guess`` its density): where :data:`WARM_ORBITALS` of the device allows, without smeared
        iterations and with pseudo-diagonalization on, the iterations rotate them from the first one.

    Returns
    -------
    SCFResult
    """
    if pseudo_diagonalization is None:
        pseudo_diagonalization = float(_per_device(PSEUDO_THRESHOLD, H.device))
    with torch.no_grad():
        H = H.detach()
        W1p = pack_one_center(W1.detach()) if isinstance(W1, Tensor) else W1
        W2p = pack_pairs(layout, W2.detach()) if isinstance(W2, Tensor) else W2
        P, n_occ, weight, filled, blocks = scf_start(guess, n_alpha, n_beta, unrestricted, spin_guess)
        rotate = pseudo_diagonalization if H.shape[-1] >= _per_device(PSEUDO_MIN_BASIS, H.device) else 0.0
        if not (_per_device(WARM_ORBITALS, H.device) and smear_steps == 0 and rotate > 0.0):
            orbitals = occupations = None
        graphs = None
        if graph_cache is not None and H.is_cuda and smear_steps == 0:
            key = (graph_key, tuple(H.shape), H.dtype, unrestricted, diis_size, filled, tuple(blocks),
                   tuple(tuple(Wc.shape) for _, _, _, Wc, _ in W2p))
            graphs = graph_cache.get(key, lambda: SCFGraphs(layout, H, W1p, W2p, n_occ, weight, unrestricted,
                                                            diis_size, filled, blocks))
        if graphs is None:
            return scf_solve(layout, H, W1p, W2p, n_alpha, n_beta, unrestricted, guess, max_iter, tol_density,
                             tol_energy, diis_size, smearing, smear_steps, spin_guess, pseudo_diagonalization,
                             _per_device(PSEUDO_MIN_BASIS, H.device), orbitals, occupations)
        graphs.load(H, W1p, W2p, P, orbitals, occupations)
        return solve_scf(_GraphedLoop(graphs), layout, H, W1p, W2p, n_occ, weight, unrestricted, diis_size, filled,
                         blocks, max_iter, tol_density, tol_energy, smearing, smear_steps, rotate,
                         orbitals is not None)


def scf_solve(layout: Layout, H: Tensor, W1: Dict[str, Tensor], W2: PackedPairs, n_alpha: Tensor, n_beta: Tensor,
              unrestricted: bool, guess: Tensor, max_iter: int, tol_density: float, tol_energy: float,
              diis_size: int, smearing: float, smear_steps: int, spin_guess: Optional[Tensor],
              pseudo_diagonalization: float, min_basis: int, orbitals: Optional[Tensor] = None,
              occupations: Optional[Tensor] = None) -> SCFResult:
    """:func:`run_scf` on packed integrals without graphs (detached inputs); pseudo-diagonalization from
    ``min_basis`` basis functions on, rotating the given ``orbitals`` (with their ``occupations``) from the
    first iteration where it is on and nothing is smeared."""
    P, n_occ, weight, filled, blocks = scf_start(guess, n_alpha, n_beta, unrestricted, spin_guess)
    rotate = pseudo_diagonalization if H.shape[-1] >= min_basis else 0.0
    if smear_steps != 0 or rotate == 0.0:
        orbitals = occupations = None
    loop = SCFLoop(layout, H, W1, W2, n_occ, weight, unrestricted, diis_size, filled, blocks)
    loop.start(P, orbitals, occupations)
    return solve_scf(loop, layout, H, W1, W2, n_occ, weight, unrestricted, diis_size, filled, blocks, max_iter,
                     tol_density, tol_energy, smearing, smear_steps, rotate,
                     orbitals is not None and occupations is not None)


def scf_with_starts(layout: Layout, H: Tensor, W1: Dict[str, Tensor], W2: PackedPairs, n_alpha: Tensor,
                    n_beta: Tensor, unrestricted: bool, guess: Tensor, max_iter: int, tol_density: float,
                    tol_energy: float, starts: List[Tuple[float, int]], restart_gap: float,
                    pseudo_diagonalization: float, min_basis: int) -> SCFResult:
    """The SCF from each start ``(smearing, smeared iterations)`` where the first may have settled in an
    excited aufbau state (unrestricted, or a frontier gap below ``restart_gap``), per structure the converged
    result of lowest energy (:func:`lowest`); the rule of :class:`~.nddo.NDDO`."""
    results = [scf_solve(layout, H, W1, W2, n_alpha, n_beta, unrestricted, guess, max_iter, tol_density,
                         tol_energy, 8, starts[0][0], starts[0][1], None, pseudo_diagonalization, min_basis)]
    if len(starts) > 1 and (unrestricted or bool((frontier_gap(results[0]) < restart_gap).any())):
        for kt, steps in starts[1:]:
            results.append(scf_solve(layout, H, W1, W2, n_alpha, n_beta, unrestricted, guess, max_iter,
                                     tol_density, tol_energy, 8, kt, steps, None, pseudo_diagonalization,
                                     min_basis))
    return lowest(results)


def scf_start(guess: Tensor, n_alpha: Tensor, n_beta: Tensor, unrestricted: bool,
              spin_guess: Optional[Tensor]) -> Tuple[Tensor, Tensor, float, int, List[Optional[int]]]:
    """The starting densities ``(S, B, n, n)``, the electrons of each spin ``(S, B)``, the occupation weight,
    the highest occupied orbital count and, per spin, the count shared by all structures (``None`` where they
    differ; the aufbau orbitals of a spin are its first ``n_occ``, so a shared count allows the block form of
    the orbital rotations)."""
    if unrestricted:
        na = n_alpha.to(guess.dtype)
        nb = n_beta.to(guess.dtype)
        tot = (na + nb).clamp(min=1e-12)
        P = torch.stack([guess * (na / tot)[:, None, None], guess * (nb / tot)[:, None, None]])
        if spin_guess is not None:
            P = spin_guess.detach().clone()
        n_occ = torch.stack([n_alpha, n_beta])
        weight = 1.0
    else:
        P = guess.unsqueeze(0)
        n_occ = n_alpha.unsqueeze(0)
        weight = 2.0
    counts = torch.jit.annotate(List[List[int]], n_occ.tolist())
    filled = 0
    blocks: List[Optional[int]] = []
    for c in counts:
        same = True
        for x in c:
            filled = max(filled, x)
            same = same and x == c[0]
        if len(c) > 0 and same:
            blocks.append(c[0])
        else:
            blocks.append(None)
    return P, n_occ, weight, filled, blocks


def _iterate(loop: SCFLoop, max_iter: int, smear_steps: int, smearing: float, rotate: float, tol_density: float,
             tol_energy: float, seeded: bool = False) -> Tuple[int, bool]:
    """The SCF iterations on a started :class:`SCFLoop` (or the same replayed from graphs): the iteration count
    and whether the last one rotated the orbitals. A loop ``seeded`` with orbitals rotates them from the first
    iteration."""
    it, have_orbitals, change, de_max, rotated = 0, seeded, 0.0 if seeded else float("inf"), float("inf"), False
    while it < max_iter:
        it += 1
        if it == smear_steps:
            loop.diis.reset()       # forget the fractional-occupation history
        loop.fock_step()
        kt = smearing * (1.0 - it / smear_steps) if it < smear_steps else 0.0
        rotated = False
        if kt == 0.0 and it > smear_steps and change < rotate and have_orbitals:
            loop.rotate_step()
            values = torch.jit.annotate(List[float], torch.stack([loop.dP.max(), loop.dE.max(),
                                                                  loop.flag.to(loop.dP.dtype)]).tolist())
            change, de_max = values[0], values[1]
            rotated = values[2] == 0.0
        if not rotated:
            loop.diagonalize()
            loop.occupy(kt)
            values = torch.jit.annotate(List[float], torch.stack([loop.dP.max(), loop.dE.max()]).tolist())
            change, de_max = values[0], values[1]
        loop.commit()
        have_orbitals = True
        # every structure is converged when the largest changes are below the thresholds
        if change < tol_density and de_max < tol_energy and it >= smear_steps:
            break
    return it, rotated


def solve_scf(loop: SCFLoop, layout: Layout, H: Tensor, W1: Dict[str, Tensor], W2: PackedPairs, n_occ: Tensor,
              weight: float, unrestricted: bool, diis_size: int, filled: int, blocks: List[Optional[int]],
              max_iter: int, tol_density: float, tol_energy: float, smearing: float, smear_steps: int,
              rotate: float, seeded: bool = False) -> SCFResult:
    """The iterations of a started loop and the converged quantities: after rotations the eigenpairs from the
    occupied and virtual blocks, and a check of the density against the aufbau occupation of the converged Fock
    matrix (a rotated solution with another occupation goes on with full diagonalizations)."""
    it, rotated = _iterate(loop, max_iter, smear_steps, smearing, rotate, tol_density, tol_energy, seeded)
    result, P_aufbau = _converged(loop, layout, H, W1, W2, n_occ, weight, unrestricted, filled, blocks, it, rotated,
                                  smear_steps, max_iter, tol_density, tol_energy, rotate)
    if P_aufbau is None:
        return result
    # a rotated solution with another occupation goes on with full diagonalizations
    loop = SCFLoop(layout, H, W1, W2, n_occ, weight, unrestricted, diis_size, filled, blocks)
    loop.start(P_aufbau)
    rest, _ = _iterate(loop, max_iter - it, 0, smearing, 0.0, tol_density, tol_energy)
    result, _ = _converged(loop, layout, H, W1, W2, n_occ, weight, unrestricted, filled, blocks, rest, False, 0,
                           max_iter - it, tol_density, tol_energy, 0.0)
    return SCFResult(result.densities, result.orbitals, result.energies, result.occupations, it + rest,
                     result.converged, result.electronic)


def _converged(loop: SCFLoop, layout: Layout, H: Tensor, W1: Dict[str, Tensor], W2: PackedPairs, n_occ: Tensor,
               weight: float, unrestricted: bool, filled: int, blocks: List[Optional[int]], it: int, rotated: bool,
               smear_steps: int, max_iter: int, tol_density: float, tol_energy: float,
               rotate: float) -> Tuple[SCFResult, Optional[Tensor]]:
    """The converged quantities of the iterations of :func:`_iterate`, and the aufbau density of the converged
    Fock matrix where a rotated solution has another occupation (else ``None``)."""
    converged = (loop.dP < tol_density) & (loop.dE < tol_energy) & (it >= smear_steps)
    P = loop.P.clone()
    F = _fock(layout, H, W1, W2, P, unrestricted)
    e_el = _electronic_energy(H, F, P, unrestricted)
    # after rotations the converged orbitals separate the occupied and virtual spaces of F (to the SCF
    # tolerance): the eigenpairs from the two smaller blocks, a full diagonalization otherwise
    pairs = _block_eigenpairs(F, loop.C, blocks) if rotated and it < max_iter else None
    if pairs is not None:
        eps, C = pairs
    else:
        eps, C = eigh(F)
    occ = _occupations(eps, n_occ, weight)
    restart = torch.jit.annotate(Optional[Tensor], None)
    if pairs is None and rotate > 0.0 and it < max_iter:
        P_aufbau = density(C, occ, filled)
        if float((P_aufbau - P).abs().max()) > max(1e3 * tol_density, 1e-6):
            restart = P_aufbau
    return SCFResult(P, C, eps, occ, it, converged, e_el), restart


def _block_eigenpairs(F: Tensor, C: Tensor, blocks: List[Optional[int]]) -> Optional[Tuple[Tensor, Tensor]]:
    """Eigenvalues and orbitals of the Fock matrices ``F`` ``(S, B, n, n)`` from the occupied (the first ``o`` of
    each spin, ``blocks``) and virtual blocks of the orbitals ``C`` that separate them: ``None`` where the blocks
    are not those of every structure, or where an occupied eigenvalue is not below every virtual one (then the
    aufbau occupation needs a full diagonalization)."""
    n = F.shape[-1]
    sizes: List[int] = []
    for o in blocks:
        if o is None or o == 0 or o == n:
            return None
        sizes.append(o)
    eps: List[Tensor] = []
    orbitals: List[Tensor] = []
    for s in range(len(sizes)):
        o = sizes[s]
        Fs, Cs = F[s], C[s]
        O, V = Cs[..., :o], Cs[..., o:]
        eo, Uo = eigh(torch.matmul(O.transpose(-1, -2), torch.matmul(Fs, O)))
        ev, Uv = eigh(torch.matmul(V.transpose(-1, -2), torch.matmul(Fs, V)))
        eps.append(torch.cat([eo, ev], -1))
        orbitals.append(torch.cat([torch.matmul(O, Uo), torch.matmul(V, Uv)], -1))
        if not bool((eo.amax(-1) < ev.amin(-1)).all()):
            return None
    return torch.stack(eps), torch.stack(orbitals)


def _pseudo_diagonalize(F: Tensor, C: Tensor, occupied: Tensor, blocks: Optional[List[Optional[int]]] = None,
                        min_gap: float = 0.5) -> Tuple[Tensor, Tensor]:
    """Orbitals of ``F`` ``(S, B, n, n)`` from the previous ones ``C`` by first-order occupied-virtual mixing,
    ``C (1 + K - K^T)`` with ``K_ia = F_ia / (F_aa - F_ii)`` in the basis of ``C``, orthonormalized (the occupied
    and the virtual spaces stay apart), and the flag (a 0-d tensor) of an occupied and a virtual orbital closer
    than ``min_gap`` (eV), which needs a full diagonalization instead. ``blocks`` (per spin, optional) gives the
    count of leading occupied orbitals shared by all structures, which allows the block form."""
    if blocks is not None:
        sizes: List[int] = []
        for o in blocks:
            if o is not None:
                sizes.append(o)
        if len(sizes) == len(blocks):
            if len(sizes) == 1 or sizes[0] == sizes[1]:
                return _pseudo_diagonalize_blocks(F, C, sizes[0], min_gap)
            c0, f0 = _pseudo_diagonalize_blocks(F[0:1], C[0:1], sizes[0], min_gap)
            c1, f1 = _pseudo_diagonalize_blocks(F[1:2], C[1:2], sizes[1], min_gap)
            return torch.cat([c0, c1]), f0 | f1
    Fm = torch.matmul(C.transpose(-1, -2), torch.matmul(F, C))
    d = Fm.diagonal(dim1=-2, dim2=-1)
    ov = occupied.unsqueeze(-1) & ~occupied.unsqueeze(-2)              # i occupied (rows), a virtual (columns)
    gap = d.unsqueeze(-2) - d.unsqueeze(-1)                            # F_aa - F_ii
    flag = (ov & (gap < min_gap)).any()
    K = torch.where(ov, Fm / torch.where(ov, gap, torch.ones_like(gap)), torch.zeros_like(Fm))
    C = C + torch.matmul(C, K - K.transpose(-1, -2))
    L = torch.linalg.cholesky_ex(torch.matmul(C.transpose(-1, -2), C))[0]
    return torch.linalg.solve_triangular(L, C.transpose(-1, -2), upper=False).transpose(-1, -2), flag


def _orthonormal(X: Tensor) -> Tensor:
    """``X L^-T`` with ``L L^T = X^T X``."""
    L = torch.linalg.cholesky_ex(torch.matmul(X.transpose(-1, -2), X))[0]
    return torch.linalg.solve_triangular(L.transpose(-1, -2), X, upper=True, left=False)


def _pseudo_diagonalize_blocks(F: Tensor, C: Tensor, o: int, min_gap: float) -> Tuple[Tensor, Tensor]:
    """:func:`_pseudo_diagonalize` where the occupied orbitals are the first ``o`` of every structure: the
    products run on the occupied and virtual blocks ``O``, ``V`` only, ``O - V K^T`` and ``V + O K``."""
    O, V = C[..., :o], C[..., o:]
    FC = torch.matmul(F, C)
    FO, FV = FC[..., :o], FC[..., o:]
    gap = (V * FV).sum(-2).unsqueeze(-2) - (O * FO).sum(-2).unsqueeze(-1)        # F_aa - F_ii, (S, B, o, v)
    flag = (gap < min_gap).any()
    K = torch.matmul(O.transpose(-1, -2), FV) / gap
    O, V = O - torch.matmul(V, K.transpose(-1, -2)), V + torch.matmul(O, K)
    return torch.cat([_orthonormal(O), _orthonormal(V)], dim=-1), flag


def frontier_gap(result: SCFResult) -> Tensor:
    """The HOMO-LUMO gap ``(B,)`` (eV) of each structure, the smaller of the two spins of an unrestricted
    result (a structure without a virtual orbital gets the padding's distance)."""
    occupied = result.occupations > 0.0
    eps = result.energies
    pad = 1.0e4                                                          # PAD_ENERGY
    homo = torch.where(occupied, eps, torch.full_like(eps, -pad)).amax(-1)
    lumo = torch.where(occupied | (eps > 0.5 * pad), torch.full_like(eps, pad), eps).amin(-1)
    return (lumo - homo).amin(0)


def lowest(results: List[SCFResult]) -> SCFResult:
    """Per structure, the converged result of lowest electronic energy (any if none converged)."""
    best = results[0]
    for r in results[1:]:
        e_best = torch.where(best.converged, best.electronic, torch.full_like(best.electronic, float("inf")))
        e_new = torch.where(r.converged, r.electronic, torch.full_like(r.electronic, float("inf")))
        take = (e_new < e_best - 1e-9) | (~best.converged & r.converged)
        if not bool(take.any()):
            continue
        t2 = take[None, :, None, None]
        best = SCFResult(torch.where(t2, r.densities, best.densities), torch.where(t2, r.orbitals, best.orbitals),
                         torch.where(take[None, :, None], r.energies, best.energies),
                         torch.where(take[None, :, None], r.occupations, best.occupations),
                         max(best.iterations, r.iterations), torch.where(take, r.converged, best.converged),
                         torch.where(take, r.electronic, best.electronic))
    return best


def _fock(layout: Layout, H: Tensor, W1: Dict[str, Tensor], W2: PackedPairs, P: Tensor,
          unrestricted: bool) -> Tensor:
    if unrestricted:
        tot = P[0] + P[1]
        return torch.stack([H + two_electron(layout, W1, W2, tot, P[s], 1.0, False) for s in range(2)])
    p = P[0]
    return (H + two_electron(layout, W1, W2, p, p, 0.5, True)).unsqueeze(0)


def _electronic_energy(H: Tensor, F: Tensor, P: Tensor, unrestricted: bool) -> Tensor:
    if unrestricted:
        tot = P[0] + P[1]
        return 0.5 * ((tot * H).sum((-1, -2)) + (P * F).sum((0, -1, -2)))
    return 0.5 * (P[0] * (H + F[0])).sum((-1, -2))


def fock_matrices(layout: Layout, H: Tensor, W1: Tensor, W2: Tensor, P: Tensor,
                  unrestricted: bool) -> Tensor:
    """Fock matrices ``(S, B, n, n)`` of densities ``P`` ``(S, B, n, n)`` (differentiable)."""
    return _fock(layout, H, pack_one_center(W1), pack_pairs(layout, W2), P, unrestricted)


def electronic_energy(H: Tensor, F: Tensor, P: Tensor, unrestricted: bool) -> Tensor:
    """``E_el = 1/2 sum P (H + F)`` per structure ``(B,)`` (differentiable)."""
    return _electronic_energy(H, F, P, unrestricted)


class CPHFLoop:
    """The coupled-perturbed SCF equations of converged orbitals as stages on one state (see :class:`SCFDensity`):
    :meth:`start` (the occupied-virtual masks, denominators and the first residual of the gradient ``gP``) and
    :meth:`step` (one preconditioned conjugate-gradient iteration, with the largest residual of the structures
    still iterating in ``res``). Without host synchronization or data-dependent shapes; with ``static`` the
    stages write into fixed buffers (the form of :class:`CPHFGraphs`)."""

    def __init__(self, layout: Layout, W1: Dict[str, Tensor], W2: PackedPairs, C: Tensor, eps: Tensor, occ: Tensor,
                 gP: Tensor, unrestricted: bool, static: bool = False):
        self.layout, self.W1, self.W2 = layout, W1, W2
        self.C, self.eps, self.occ, self.gP = C, eps, occ, gP
        self.unrestricted, self.static = unrestricted, static
        self.weight = 1.0 if unrestricted else 2.0
        self.ov = torch.zeros(C.shape, dtype=torch.bool, device=C.device)
        self.D = torch.ones_like(C)
        self.u, self.r, self.p = torch.zeros_like(C), torch.zeros_like(C), torch.zeros_like(C)
        self.rz = C.new_zeros([C.shape[1]])
        self.unstable = torch.zeros([C.shape[1]], dtype=torch.bool, device=C.device)
        self.res = C.new_zeros([])

    def _put(self, buffer: Tensor, value: Tensor) -> Tensor:
        if self.static:
            buffer.copy_(value)
            return buffer
        return value

    def _to_mo(self, X: Tensor) -> Tensor:
        return torch.matmul(self.C.transpose(-1, -2), torch.matmul(X, self.C))

    def from_mo(self, U: Tensor) -> Tensor:
        Uov = torch.where(self.ov, U, torch.zeros_like(U))
        return torch.matmul(self.C, torch.matmul(Uov + Uov.transpose(-1, -2), self.C.transpose(-1, -2)))

    def _G(self, Z: Tensor) -> Tensor:
        if self.unrestricted:
            tot = Z[0] + Z[1]
            return torch.stack([two_electron(self.layout, self.W1, self.W2, tot, Z[s], 1.0, False)
                                for s in range(2)])
        z0 = Z[0]
        return two_electron(self.layout, self.W1, self.W2, z0, z0, 0.5, True).unsqueeze(0)

    def _A(self, u: Tensor) -> Tensor:
        # A u = (D / w) u + [G(z(u))]_ov with z(u) = from_mo(u)
        return torch.where(self.ov, self.D / self.weight * u + self._to_mo(self._G(self.from_mo(u))),
                           torch.zeros_like(u))

    def start(self) -> None:
        gP = 0.5 * (self.gP + self.gP.transpose(-1, -2))
        occ_mask = self.occ > 0
        denom = self.eps.unsqueeze(-1) - self.eps.unsqueeze(-2)              # (S,B,n,n) e_i - e_a
        valid = self.layout.valid.unsqueeze(0)
        ov = occ_mask.unsqueeze(-1) & ~occ_mask.unsqueeze(-2) & valid.unsqueeze(-1) & valid.unsqueeze(-2)
        self.ov = self._put(self.ov, ov)
        self.D = self._put(self.D, torch.where(ov, -denom, torch.ones_like(denom)))   # e_a - e_i > 0
        D = self.D
        # solve (I - R G) z = R g by conjugate gradients on the symmetric form A u = -g_ov
        b = -torch.where(ov, self._to_mo(gP), torch.zeros_like(gP))
        u = torch.where(ov, b * self.weight / D, torch.zeros_like(b))
        r = b - self._A(u)
        zp = torch.where(ov, r * self.weight / D, torch.zeros_like(r))
        self.u = self._put(self.u, u)
        self.r = self._put(self.r, r)
        self.p = self._put(self.p, zp)
        self.rz = self._put(self.rz, (r * zp).sum((0, 2, 3)))
        self.unstable = self._put(self.unstable, torch.zeros_like(self.unstable))

    def step(self) -> None:
        p, rz = self.p, self.rz
        Ap = self._A(p)
        pAp = (p * Ap).sum((0, 2, 3))
        # A is positive definite at an SCF minimum; a non-positive curvature marks a saddle-point solution,
        # whose iteration stops
        unstable = self.unstable | ((pAp <= 0) & (rz > 0))
        step = ~unstable & (pAp > 0)
        alpha = torch.where(step, rz / torch.where(step, pAp, torch.ones_like(pAp)), torch.zeros_like(pAp))
        u = self.u + alpha[None, :, None, None] * p
        r = self.r - alpha[None, :, None, None] * Ap
        # the largest residual of the structures still iterating (0 when none is)
        res = torch.where((~unstable)[None, :, None, None], r.abs(), torch.zeros_like(r)).amax()
        zp = torch.where(self.ov, r * self.weight / self.D, torch.zeros_like(r))
        rz_new = (r * zp).sum((0, 2, 3))
        beta = torch.where(rz > 0, rz_new / torch.where(rz > 0, rz, torch.ones_like(rz)), torch.zeros_like(rz))
        self.u = self._put(self.u, u)
        self.r = self._put(self.r, r)
        self.p = self._put(self.p, zp + beta[None, :, None, None] * p)
        self.rz = self._put(self.rz, rz_new)
        self.unstable = self._put(self.unstable, unstable)
        self.res = self._put(self.res, res)


class CPHFGraphs:
    """CUDA graphs of the stages of :class:`CPHFLoop` for one batch composition; :meth:`load` copies a new
    backward's inputs into the buffers and :meth:`run` replays a stage."""

    STAGES = ("start", "step")

    def __init__(self, layout: Layout, W1: Dict[str, Tensor], W2: PackedPairs, C: Tensor, eps: Tensor,
                 occ: Tensor, gP: Tensor, unrestricted: bool):
        self.W1 = {key: v.clone() for key, v in W1.items()}
        self.W2 = [(ka, kb, maps, Wc.clone(), Wx.clone()) for ka, kb, maps, Wc, Wx in W2]
        self.loop = CPHFLoop(layout, self.W1, self.W2, C.clone(), eps.clone(), occ.clone(), gP.clone(),
                             unrestricted, static=True)
        self.graphs: Dict[str, torch.cuda.CUDAGraph] = {}
        pool = torch.cuda.graph_pool_handle()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for stage in self.STAGES:
                getattr(self.loop, stage)()
        torch.cuda.current_stream().wait_stream(side)
        for stage in self.STAGES:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                getattr(self.loop, stage)()
            self.graphs[stage] = g

    def load(self, W1: Dict[str, Tensor], W2: PackedPairs, C: Tensor, eps: Tensor, occ: Tensor,
             gP: Tensor) -> CPHFLoop:
        for key, v in self.W1.items():
            if key in W1:
                v.copy_(W1[key])
            else:
                v.copy_(W1["coulomb"] - 0.5 * W1["exchange"])        # the cached closed-shell matrix
        for (_, _, _, Wc, Wx), (_, _, _, nc, nx) in zip(self.W2, W2):
            Wc.copy_(nc)
            Wx.copy_(nx)
        for buf, x in ((self.loop.C, C), (self.loop.eps, eps), (self.loop.occ, occ), (self.loop.gP, gP)):
            buf.copy_(x)
        return self.loop

    def run(self, stage: str) -> None:
        self.graphs[stage].replay()


class SCFDensity(torch.autograd.Function):
    """Converged densities as a function of ``H``, ``W1`` and ``W2``.

    The forward pass returns the densities of a converged :class:`SCFResult`
    (computed outside, passed through ``ctx``); the backward pass solves the
    coupled-perturbed SCF equations in the occupied-virtual space by
    preconditioned conjugate gradients and returns the gradients with respect
    to the core Hamiltonian and the two-electron integrals.
    """

    @staticmethod
    def forward(ctx, H, W1, W2, result: SCFResult, layout: Layout, unrestricted: bool,
                tol: float, graph_cache: Optional[SCFGraphCache] = None, graph_key: tuple = ()):
        ctx.save_for_backward(H, W1, W2)
        ctx.result, ctx.layout, ctx.unrestricted, ctx.tol = result, layout, unrestricted, tol
        ctx.graphs = None
        C = result.orbitals
        if graph_cache is not None and C.is_cuda:
            # the conjugate-gradient iterations replayed from CUDA graphs of a recurring composition, captured
            # here on the calling thread (a capture in the backward pass could meet autograd's other threads)
            W1p, W2p = pack_one_center(W1.detach()), pack_pairs(layout, W2.detach())
            key = ("cphf", graph_key, tuple(C.shape), C.dtype, unrestricted,
                   tuple(tuple(Wc.shape) for _, _, _, Wc, _ in W2p))
            ctx.graphs = graph_cache.get(key, lambda: CPHFGraphs(layout, W1p, W2p, C, result.energies,
                                                                 result.occupations, torch.zeros_like(C),
                                                                 unrestricted))
        return result.densities.clone()

    @staticmethod
    def backward(ctx, gP):
        H, W1, W2 = ctx.saved_tensors
        res, layout, uhf = ctx.result, ctx.layout, ctx.unrestricted
        C, eps, occ = res.orbitals, res.energies, res.occupations
        with torch.no_grad():
            W1d, W2d = W1.detach(), W2.detach()
            W1p, W2p = pack_one_center(W1d), pack_pairs(layout, W2d)
            gP = gP.detach()
            graphs = ctx.graphs
            if graphs is not None:
                loop = graphs.load(W1p, W2p, C, eps, occ, gP)
                run = graphs.run
            else:
                loop = CPHFLoop(layout, W1p, W2p, C, eps, occ, gP, uhf)

                def run(stage: str) -> None:
                    getattr(loop, stage)()
            run("start")
            for _ in range(200):
                run("step")
                if float(loop.res) < ctx.tol:
                    break
            unstable = loop.unstable
            if bool(unstable.any()):
                warnings.warn(f"the SCF solution of structure(s) {unstable.nonzero().flatten().tolist()} "
                              "is not a minimum; their density response is incomplete", RuntimeWarning,
                              stacklevel=2)
            z = loop.from_mo(loop.u)          # dL/dF (AO), one per spin
        # dL/dH = sum over spins of z; dL/dW from <z, G_W(P)>
        gH = z.sum(0)
        with torch.enable_grad():
            W1g = W1d.clone().requires_grad_(True)
            W2g = W2d.clone().requires_grad_(True)
            P = res.densities
            if uhf:
                tot = P[0] + P[1]
                one, groups = pack_one_center(W1g), pack_pairs(layout, W2g)
                Gs = torch.stack([two_electron(layout, one, groups, tot, P[s], 1.0, False) for s in range(2)])
            else:
                p0 = P[0]
                Gs = two_electron(layout, pack_one_center(W1g), pack_pairs(layout, W2g), p0, p0, 0.5,
                                  True).unsqueeze(0)
            gW1, gW2 = torch.autograd.grad((Gs * z).sum(), (W1g, W2g), allow_unused=True)
        return gH, gW1, gW2, None, None, None, None, None, None
