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
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor

PAD_ENERGY = 1.0e4   # diagonal of the padding orbitals (eV)


@dataclass
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
    """

    n_struct: int
    n_basis: int
    k: int
    atom_struct: Tensor
    ao: Tensor
    pair_a: Tensor
    pair_b: Tensor
    valid: Tensor

    def flat(self, struct: Tensor, rows: Tensor, cols: Tensor) -> Tensor:
        n = self.n_basis + 1
        return (struct * n + rows) * n + cols

    def gather_atom(self, M: Tensor) -> Tensor:
        """Diagonal atom blocks ``(N, k, k)`` of padded matrices ``(B, n, n)``."""
        full = self._with_sink(M).reshape(-1)
        idx = self.flat(self.atom_struct[:, None, None], self.ao[:, :, None], self.ao[:, None, :])
        return full[idx]

    def gather_pair(self, M: Tensor) -> Tensor:
        """Off-diagonal blocks ``(P, k, k)`` (rows on atom A, columns on atom B)."""
        full = self._with_sink(M).reshape(-1)
        s = self.atom_struct[self.pair_a]
        idx = self.flat(s[:, None, None], self.ao[self.pair_a][:, :, None],
                        self.ao[self.pair_b][:, None, :])
        return full[idx]

    def scatter(self, atom_blocks: Tensor, pair_blocks: Tensor) -> Tensor:
        """Padded symmetric matrices from atom blocks and (A, B) pair blocks."""
        n = self.n_basis + 1
        out = atom_blocks.new_zeros(self.n_struct * n * n)
        idx_a = self.flat(self.atom_struct[:, None, None], self.ao[:, :, None], self.ao[:, None, :])
        out = out.index_add(0, idx_a.reshape(-1), atom_blocks.reshape(-1))
        if pair_blocks.shape[0]:
            s = self.atom_struct[self.pair_a][:, None, None]
            ra, rb = self.ao[self.pair_a], self.ao[self.pair_b]
            idx_ab = self.flat(s, ra[:, :, None], rb[:, None, :])
            idx_ba = self.flat(s, rb[:, None, :], ra[:, :, None])
            out = out.index_add(0, idx_ab.reshape(-1), pair_blocks.reshape(-1))
            out = out.index_add(0, idx_ba.reshape(-1), pair_blocks.reshape(-1))
        return out.reshape(self.n_struct, n, n)[:, :-1, :-1]

    def _with_sink(self, M: Tensor) -> Tensor:
        return torch.nn.functional.pad(M, (0, 1, 0, 1))


def two_electron(layout: Layout, W1: Tensor, W2: Tensor, P_total: Tensor, P_spin: Tensor,
                 exchange: float) -> Tensor:
    """Two-electron part ``G`` of the Fock matrix (padded ``(B, n, n)``).

    Parameters
    ----------
    W1 : Tensor
        One-center integrals of every atom ``(N, k, k, k, k)``.
    W2 : Tensor
        Two-center integrals of every pair ``(P, k, k, k, k)``.
    P_total : Tensor
        Total density ``(B, n, n)`` (Coulomb term).
    P_spin : Tensor
        Density of the exchange term ``(B, n, n)``.
    exchange : float
        Weight of the exchange term: 1/2 for the closed-shell Fock matrix with
        ``P_spin = P_total``, 1 for one spin of the unrestricted one.
    """
    Paa = layout.gather_atom(P_total)
    Xaa = layout.gather_atom(P_spin)
    atom = (torch.einsum("aijkl,akl->aij", W1, Paa)
            - exchange * torch.einsum("aikjl,akl->aij", W1, Xaa))
    if W2.shape[0]:
        ia, ib = layout.pair_a, layout.pair_b
        coul_a = torch.einsum("pijkl,pkl->pij", W2, Paa[ib])
        coul_b = torch.einsum("pijkl,pij->pkl", W2, Paa[ia])
        atom = atom.index_add(0, ia, coul_a).index_add(0, ib, coul_b)
        Xab = layout.gather_pair(P_spin)
        pair = -exchange * torch.einsum("pijkl,pjl->pik", W2, Xab)
    else:
        pair = W2.new_zeros((0, layout.k, layout.k))
    # the scatter adds each off-diagonal block twice (A,B) and (B,A); the atom
    # blocks are symmetric already
    return layout.scatter(atom, pair)


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


def density(C: Tensor, occ: Tensor) -> Tensor:
    """``P = C diag(occ) C^T``."""
    return torch.einsum("bik,bk,bjk->bij", C, occ, C)


class DIIS:
    """Pulay extrapolation of the Fock matrices of a batch."""

    def __init__(self, size: int = 8):
        self.size = size
        self.focks: List[Tensor] = []
        self.errors: List[Tensor] = []

    def __call__(self, F: Tensor, err: Tensor) -> Tensor:
        self.focks.append(F)
        self.errors.append(err.reshape(err.shape[0], -1))
        if len(self.focks) > self.size:
            self.focks.pop(0)
            self.errors.pop(0)
        m = len(self.focks)
        if m < 2:
            return F
        E = torch.stack(self.errors, dim=1)                          # (B, m, L)
        Bm = torch.einsum("bil,bjl->bij", E, E)
        scale = Bm.diagonal(dim1=1, dim2=2).abs().amax(-1).clamp(min=1e-30)
        Bm = Bm / scale[:, None, None]
        nb = Bm.shape[0]
        A = Bm.new_zeros((nb, m + 1, m + 1))
        A[:, :m, :m] = Bm + 1e-12 * torch.eye(m, dtype=Bm.dtype, device=Bm.device)
        A[:, :m, m] = -1.0
        A[:, m, :m] = -1.0
        rhs = Bm.new_zeros((nb, m + 1))
        rhs[:, m] = -1.0
        try:
            c = torch.linalg.solve(A, rhs)[:, :m]
        except RuntimeError:
            return F
        Fs = torch.stack(self.focks, dim=1)                          # (B, m, ...)
        return torch.einsum("bm,bm...->b...", c, Fs)


@dataclass
class SCFResult:
    """Converged SCF quantities (detached)."""

    densities: Tensor          # (S, B, n, n): one total density (RHF) or alpha, beta (UHF)
    orbitals: Tensor           # (S, B, n, n)
    energies: Tensor           # (S, B, n) orbital energies, eV
    occupations: Tensor        # (S, B, n)
    iterations: int
    converged: Tensor          # (B,) bool
    electronic: Tensor = None  # (B,) electronic energy, eV


def run_scf(layout: Layout, H: Tensor, W1: Tensor, W2: Tensor, n_alpha: Tensor, n_beta: Tensor,
            unrestricted: bool, guess: Tensor, max_iter: int = 300, tol_density: float = 1e-9,
            tol_energy: float = 1e-10, diis_size: int = 8, smearing: float = 2.0,
            smear_steps: int = 25, spin_guess: Optional[Tensor] = None) -> SCFResult:
    """Solve the SCF equations of a batch (no gradient tracking).

    Parameters
    ----------
    layout : Layout
        Index maps of the batch.
    H : Tensor
        Core Hamiltonian ``(B, n, n)`` (padding orbitals on a large diagonal).
    W1, W2 : Tensor
        One- and two-center integrals.
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

    Returns
    -------
    SCFResult
    """
    with torch.no_grad():
        H, W1, W2 = H.detach(), W1.detach(), W2.detach()
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
        diis = DIIS(diis_size)
        e_old = None
        converged = torch.zeros(H.shape[0], dtype=torch.bool, device=H.device)
        it = 0
        for it in range(1, max_iter + 1):
            if it == smear_steps:
                diis = DIIS(diis_size)      # forget the fractional-occupation history
            F = _fock(layout, H, W1, W2, P, unrestricted)
            energy = _electronic_energy(H, F, P, unrestricted)
            err = torch.einsum("sbij,sbjk->sbik", F, P) - torch.einsum("sbij,sbjk->sbik", P, F)
            Fx = diis(F.transpose(0, 1), err.transpose(0, 1)).transpose(0, 1)
            eps, C = torch.linalg.eigh(Fx)
            kt = smearing * (1.0 - it / smear_steps) if it < smear_steps else 0.0
            if kt > 0.0:
                occ = torch.stack([_fermi_occupations(eps[s], n_occ[s], weight, kt)
                                   for s in range(P.shape[0])])
            else:
                occ = torch.stack([_occupations(eps[s], n_occ[s], weight) for s in range(P.shape[0])])
            P_new = torch.stack([density(C[s], occ[s]) for s in range(P.shape[0])])
            dP = (P_new - P).abs().amax(dim=(0, 2, 3))
            dE = (energy - e_old).abs() if e_old is not None else torch.full_like(energy, 1e10)
            converged = (dP < tol_density) & (dE < tol_energy) & (it >= smear_steps)
            P, e_old = P_new, energy
            if bool(converged.all()):
                break
        F = _fock(layout, H, W1, W2, P, unrestricted)
        e_el = _electronic_energy(H, F, P, unrestricted)
        eps, C = torch.linalg.eigh(F)
        occ = torch.stack([_occupations(eps[s], n_occ[s], weight) for s in range(P.shape[0])])
    return SCFResult(densities=P, orbitals=C, energies=eps, occupations=occ, iterations=it,
                     converged=converged, electronic=e_el)


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
        best = SCFResult(
            densities=torch.where(t2, r.densities, best.densities),
            orbitals=torch.where(t2, r.orbitals, best.orbitals),
            energies=torch.where(take[None, :, None], r.energies, best.energies),
            occupations=torch.where(take[None, :, None], r.occupations, best.occupations),
            iterations=max(best.iterations, r.iterations),
            converged=torch.where(take, r.converged, best.converged),
            electronic=torch.where(take, r.electronic, best.electronic))
    return best


def _fock(layout: Layout, H: Tensor, W1: Tensor, W2: Tensor, P: Tensor, unrestricted: bool) -> Tensor:
    if unrestricted:
        tot = P[0] + P[1]
        return torch.stack([H + two_electron(layout, W1, W2, tot, P[s], 1.0) for s in range(2)])
    return (H + two_electron(layout, W1, W2, P[0], P[0], 0.5)).unsqueeze(0)


def _electronic_energy(H: Tensor, F: Tensor, P: Tensor, unrestricted: bool) -> Tensor:
    if unrestricted:
        tot = P[0] + P[1]
        return 0.5 * ((tot * H).sum((-1, -2)) + (P * F).sum((0, -1, -2)))
    return 0.5 * (P[0] * (H + F[0])).sum((-1, -2))


def fock_matrices(layout: Layout, H: Tensor, W1: Tensor, W2: Tensor, P: Tensor,
                  unrestricted: bool) -> Tensor:
    """Fock matrices ``(S, B, n, n)`` of densities ``P`` ``(S, B, n, n)`` (differentiable)."""
    return _fock(layout, H, W1, W2, P, unrestricted)


def electronic_energy(H: Tensor, F: Tensor, P: Tensor, unrestricted: bool) -> Tensor:
    """``E_el = 1/2 sum P (H + F)`` per structure ``(B,)`` (differentiable)."""
    return _electronic_energy(H, F, P, unrestricted)


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
                tol: float):
        ctx.save_for_backward(H, W1, W2)
        ctx.result, ctx.layout, ctx.unrestricted, ctx.tol = result, layout, unrestricted, tol
        return result.densities.clone()

    @staticmethod
    def backward(ctx, gP):
        H, W1, W2 = ctx.saved_tensors
        res, layout, uhf = ctx.result, ctx.layout, ctx.unrestricted
        C, eps, occ = res.orbitals, res.energies, res.occupations
        weight = 1.0 if uhf else 2.0
        gP = 0.5 * (gP + gP.transpose(-1, -2))
        occ_mask = occ > 0
        # response of each spin density to a symmetric perturbation X of its
        # Fock matrix: R[X] = sum_ia w (C_i C_a^T + C_a C_i^T) X_ia / (e_i - e_a)
        denom = eps.unsqueeze(-1) - eps.unsqueeze(-2)                       # (S,B,n,n) e_i - e_a
        ov = occ_mask.unsqueeze(-1) & ~occ_mask.unsqueeze(-2)              # i occupied, a virtual
        valid = layout.valid.unsqueeze(0)
        ov = ov & valid.unsqueeze(-1) & valid.unsqueeze(-2)
        safe = torch.where(ov, denom, torch.ones_like(denom))

        def to_mo(X):
            return torch.einsum("sbki,sbkl,sblj->sbij", C, X, C)

        def from_mo(U):
            Uov = torch.where(ov, U, torch.zeros_like(U))
            return torch.einsum("sbik,sbkl,sbjl->sbij", C, Uov + Uov.transpose(-1, -2), C)

        def response(X):
            return from_mo(weight * torch.where(ov, to_mo(X), torch.zeros_like(X)) / safe)

        def G(Z):
            if uhf:
                tot = Z[0] + Z[1]
                return torch.stack([two_electron(layout, W1, W2, tot, Z[s], 1.0) for s in range(2)])
            return two_electron(layout, W1, W2, Z[0], Z[0], 0.5).unsqueeze(0)

        with torch.no_grad():
            W1d, W2d = W1.detach(), W2.detach()
            # solve (I - R G) z = R g by conjugate gradients on the symmetric form
            # A u = -g_ov with A u = (D / w) u + [G(z(u))]_ov,  z(u) = from_mo(u)
            D = torch.where(ov, -denom, torch.ones_like(denom))           # e_a - e_i > 0
            b = -torch.where(ov, to_mo(gP), torch.zeros_like(gP))

            def A(u):
                z = from_mo(u)
                return torch.where(ov, D / weight * u + to_mo(G(z)), torch.zeros_like(u))

            u = torch.where(ov, b * weight / D, torch.zeros_like(b))
            r = b - A(u)
            zp = torch.where(ov, r * weight / D, torch.zeros_like(r))
            p = zp
            rz = (r * zp).sum((0, 2, 3))
            unstable = torch.zeros_like(rz, dtype=torch.bool)
            for _ in range(200):
                Ap = A(p)
                pAp = (p * Ap).sum((0, 2, 3))
                # A is positive definite at an SCF minimum; a non-positive curvature
                # marks a saddle-point solution, whose iteration stops
                unstable = unstable | ((pAp <= 0) & (rz > 0))
                step = ~unstable & (pAp > 0)
                alpha = torch.where(step, rz / torch.where(step, pAp, torch.ones_like(pAp)),
                                    torch.zeros_like(pAp))
                u = u + alpha[None, :, None, None] * p
                r = r - alpha[None, :, None, None] * Ap
                active = r[:, ~unstable]
                if active.numel() == 0 or float(active.abs().max()) < ctx.tol:
                    break
                zp = torch.where(ov, r * weight / D, torch.zeros_like(r))
                rz_new = (r * zp).sum((0, 2, 3))
                beta = torch.where(rz > 0, rz_new / torch.where(rz > 0, rz, torch.ones_like(rz)),
                                   torch.zeros_like(rz))
                p = zp + beta[None, :, None, None] * p
                rz = rz_new
            if bool(unstable.any()):
                warnings.warn(f"the SCF solution of structure(s) {unstable.nonzero().flatten().tolist()} "
                              "is not a minimum; their density response is incomplete", RuntimeWarning,
                              stacklevel=2)
            z = from_mo(u)          # dL/dF (AO), one per spin
        # dL/dH = sum over spins of z; dL/dW from <z, G_W(P)>
        gH = z.sum(0)
        with torch.enable_grad():
            W1g = W1d.clone().requires_grad_(True)
            W2g = W2d.clone().requires_grad_(True)
            P = res.densities
            if uhf:
                tot = P[0] + P[1]
                Gs = torch.stack([two_electron(layout, W1g, W2g, tot, P[s], 1.0) for s in range(2)])
            else:
                Gs = two_electron(layout, W1g, W2g, P[0], P[0], 0.5).unsqueeze(0)
            gW1, gW2 = torch.autograd.grad((Gs * z).sum(), (W1g, W2g), allow_unused=True)
        return gH, gW1, gW2, None, None, None, None
