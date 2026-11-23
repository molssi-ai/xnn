"""A bond-order energy term of the NDDO methods: a smooth switch on the Wiberg bond index of atom pairs.

For each pair of atoms ``A``, ``B`` whose elements have an entry,

    E_W = sum_AB c_AB S(t_AB) g(R_AB),    t_AB = (W_AB - W_lo) / (W_hi - W_lo),    W_AB = sum_{A, B} P_mu,nu^2,

(the last sum over the orbitals ``mu`` of ``A`` and ``nu`` of ``B``),

with ``P`` the total density (both spins; the NDDO basis is orthonormal), ``S(t) = 6 t^5 - 15 t^4 + 10 t^3`` on
``[0, 1]`` (0 below, 1 above; first and second derivatives vanish at both ends) and the cutoff
``g(R) = 1 - S((R - R_on) / (R_off - R_on))``. The bond index separates triple bonds (W near 3) from double bonds
(W near 2) far more clearly than the bond length does, so the switch can be smooth and flat on both sides.

The term depends on the density, so it is part of the SCF: its derivative ``dE_W/dP`` adds to the Fock matrices
(of both spins), its second derivative (:func:`kernel`) to the response equations. Forces, Hessians and parameter
gradients are then exact.

:class:`BondOrder` holds the pairs of a batch and their values; the functions here act on padded matrices
``(B, n, n)`` of :class:`~.scf.Layout`.
"""
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

#: eV in kcal/mol (the energies ``c_AB`` are in kcal/mol).
EV_TO_KCAL = 23.06054783061903


class BondOrder:
    """The pairs of a batch that carry the bond-order term and their values.

    Attributes
    ----------
    atom_a, atom_b : Tensor
        Atoms of every pair ``(P,)``.
    struct : Tensor
        Structure of every pair ``(P,)``.
    gather : Tensor
        Flat indices ``(P * k * k,)`` of the pairs' ``(A, B)`` blocks in matrices ``(B, n, n)`` (0 for the orbitals
        an atom does not have, which :attr:`mask` clears).
    put : Tensor
        Flat indices ``(2 P k k,)`` of the ``(A, B)`` blocks, then the ``(B, A)`` blocks, in matrices ``(B, n, n)``
        followed by one spare element (the destination of the orbitals an atom does not have).
    mask : Tensor
        Ones ``(P, k, k)``, zero for the orbitals an atom does not have; applied where :attr:`masked`.
    coef : Tensor
        ``c_AB g(R_AB)`` of every pair in eV ``(P,)``.
    lower, width : Tensor
        The window ``W_lo`` and ``W_hi - W_lo`` of every pair ``(P,)``.
    n_struct, n_basis, k : int
        Structures, padded basis size and orbitals per atom block of the layout.
    """

    def __init__(self, atom_a: Tensor, atom_b: Tensor, struct: Tensor, gather: Tensor, put: Tensor, mask: Tensor,
                 coef: Tensor, lower: Tensor, width: Tensor, n_struct: int, n_basis: int, k: int, masked: bool):
        self.atom_a = atom_a
        self.atom_b = atom_b
        self.struct = struct
        self.gather = gather
        self.put = put
        self.mask = mask
        self.coef = coef
        self.lower = lower
        self.width = width
        self.n_struct = n_struct
        self.n_basis = n_basis
        self.k = k
        self.masked = masked

    def tensors(self) -> List[Tensor]:
        """Every tensor (index tensors, then values), in the order of :func:`load`."""
        return [self.atom_a, self.atom_b, self.struct, self.gather, self.put, self.mask, self.coef, self.lower,
                self.width]


def from_tensors(t: List[Tensor], n_struct: int, n_basis: int, k: int, masked: bool) -> BondOrder:
    """The :class:`BondOrder` of the tensors of :meth:`BondOrder.tensors`."""
    return BondOrder(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], n_struct, n_basis, k, masked)


def with_values(bo: BondOrder, coef: Tensor, lower: Tensor, width: Tensor) -> BondOrder:
    """The pairs of ``bo`` with other values."""
    return BondOrder(bo.atom_a, bo.atom_b, bo.struct, bo.gather, bo.put, bo.mask, coef, lower, width, bo.n_struct,
                     bo.n_basis, bo.k, bo.masked)


def cloned(bo: BondOrder) -> BondOrder:
    """A copy of ``bo`` on buffers of its own (for :func:`load`)."""
    return from_tensors([x.detach().clone() for x in bo.tensors()], bo.n_struct, bo.n_basis, bo.k, bo.masked)


def load(dst: BondOrder, src: BondOrder) -> None:
    """Copy the tensors of ``src`` into the buffers of ``dst`` (of the same shapes)."""
    for a, b in zip(dst.tensors(), src.tensors()):
        a.copy_(b)


def index(atom_struct: Tensor, ao: Tensor, pair_a: Tensor, pair_b: Tensor, n_struct: int, n_basis: int,
          coef: Tensor, lower: Tensor, width: Tensor) -> BondOrder:
    """The :class:`BondOrder` of the pairs ``pair_a``, ``pair_b`` of a layout (its ``atom_struct`` and ``ao``)."""
    k = int(ao.shape[1])
    n = n_basis
    s = atom_struct[pair_a][:, None, None]
    oa, ob = ao[pair_a][:, :, None], ao[pair_b][:, None, :]
    real = (oa < n) & (ob < n)
    ab = torch.where(real, (s * n + oa) * n + ob, torch.zeros_like(oa * ob)).reshape(-1)
    ba = torch.where(real, (s * n + ob) * n + oa, torch.zeros_like(oa * ob)).reshape(-1)
    spare = torch.full_like(ab, n_struct * n * n)
    flat = real.reshape(-1)
    put = torch.cat([torch.where(flat, ab, spare), torch.where(flat, ba, spare)])
    masked = not bool(real.all())
    return BondOrder(pair_a, pair_b, atom_struct[pair_a], ab, put, real.to(coef.dtype), coef, lower, width,
                     n_struct, n_basis, k, masked)


def _step(x: Tensor) -> Tuple[Tensor, Tensor]:
    """``S`` and ``S'`` of the quintic step at ``x`` in ``[0, 1]`` (tensors or NumPy arrays)."""
    x2 = x * x
    y = 1.0 - x
    return x2 * x * (10.0 + x * (6.0 * x - 15.0)), 30.0 * x2 * (y * y)


def smooth_step(t: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """``S(t)``, ``S'(t)`` and ``S''(t)`` of the quintic step, constant outside ``[0, 1]`` (their derivatives by
    autograd are those of the step to any order)."""
    x = t.clamp(0.0, 1.0)
    S, S1 = _step(x)
    return S, S1, 60.0 * x * (1.0 - x) * (1.0 - 2.0 * x)


def cutoff(r: Tensor, r_on: Tensor, r_width: Tensor) -> Tensor:
    """``g(R) = 1 - S((R - R_on) / width)``: 1 below ``R_on``, 0 beyond ``R_on + width``."""
    return 1.0 - _step(((r - r_on) / r_width).clamp(0.0, 1.0))[0]


def pair_values(slots: Tensor, energy: Tensor, window: Tensor, cut: Tensor, za: Tensor, zb: Tensor,
                r: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """The values of :class:`BondOrder` of pairs of atomic numbers ``za``, ``zb`` at distances ``r``.

    ``slots`` ``(Z, Z)`` gives each element pair's entry (``-1``: none, which gets ``coef = 0``), ``energy`` the
    energies ``c`` (kcal/mol), ``window`` the rows ``(W_lo, log width)`` and ``cut`` the rows ``(R_on, log width)``
    of the cutoff, one per entry or a single shared row.
    """
    slot = slots[za, zb]
    on = slot >= 0
    s = slot.clamp(min=0)
    c = torch.where(on, energy.index_select(0, s), torch.zeros_like(r))
    rows = cut.index_select(0, s if cut.shape[0] > 1 else torch.zeros_like(s))
    win = window.index_select(0, s)
    coef = c * cutoff(r, rows[:, 0], rows[:, 1].exp()) / 23.06054783061903          # EV_TO_KCAL
    return coef, win[:, 0], win[:, 1].exp()


def blocks(bo: BondOrder, M: Tensor) -> Tensor:
    """The ``(A, B)`` blocks ``(P, k, k)`` of matrices ``(B, n, n)``."""
    v = M.reshape(-1).index_select(0, bo.gather).reshape(-1, bo.k, bo.k)
    return v * bo.mask if bo.masked else v


def scatter(bo: BondOrder, blk: Tensor) -> Tensor:
    """Symmetric matrices ``(B, n, n)`` with the pair blocks ``blk`` ``(P, k, k)`` at ``(A, B)`` and their
    transposes at ``(B, A)``."""
    flat = blk.reshape(-1)
    n = bo.n_struct * bo.n_basis * bo.n_basis
    out = blk.new_zeros(n + 1).index_add(0, bo.put, torch.cat([flat, flat]))
    return out[:n].reshape(bo.n_struct, bo.n_basis, bo.n_basis)


def switch(bo: BondOrder, P: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """The pair blocks of the total density ``P`` ``(B, n, n)``, and the step and its derivative at each pair's
    bond index."""
    Pab = blocks(bo, P)
    W = (Pab * Pab).sum((-1, -2))
    S, S1 = _step(((W - bo.lower) / bo.width).clamp(0.0, 1.0))
    return Pab, S, S1


def pair_energies(bo: BondOrder, P: Tensor) -> Tensor:
    """``c g S`` of every pair (eV) at the total density ``P``."""
    return bo.coef * switch(bo, P)[1]


def fock(bo: BondOrder, P: Tensor) -> Tuple[Tensor, Tensor]:
    """The term's Fock matrices ``dE_W/dP`` ``(B, n, n)`` and energies ``(B,)`` (eV) at the total density ``P``."""
    Pab, S, S1 = switch(bo, P)
    F = scatter(bo, (bo.coef * S1 / bo.width)[:, None, None] * Pab)
    E = P.new_zeros(bo.n_struct).index_add(0, bo.struct, bo.coef * S)
    return F, E


def linearized(bo: BondOrder, P: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """The pair blocks of ``P`` and the factors ``c g S' / dW`` and ``c g S'' / dW^2`` of :func:`kernel`."""
    Pab = blocks(bo, P)
    W = (Pab * Pab).sum((-1, -2))
    _, S1, S2 = smooth_step((W - bo.lower) / bo.width)
    return Pab, bo.coef * S1 / bo.width, bo.coef * S2 / (bo.width * bo.width)


def kernel_at(bo: BondOrder, Pab: Tensor, a1: Tensor, a2: Tensor, X: Tensor) -> Tensor:
    """:func:`kernel` from the quantities of :func:`linearized`."""
    Xab = blocks(bo, X)
    dW = 2.0 * (Pab * Xab).sum((-1, -2))
    return scatter(bo, a1[:, None, None] * Xab + (a2 * dW)[:, None, None] * Pab)


def kernel(bo: BondOrder, P: Tensor, X: Tensor) -> Tensor:
    """The change of the term's Fock matrices for a change ``X`` ``(B, n, n)`` of the total density ``P``:
    ``c g / dW [S' X_AB + S'' / dW P_AB 2 sum(P_AB X_AB)]`` on the pair blocks."""
    Pab, a1, a2 = linearized(bo, P)
    return kernel_at(bo, Pab, a1, a2, X)


def total_density(P: Tensor) -> Tensor:
    """The total density ``(B, n, n)`` of densities ``(S, B, n, n)`` (one total density, or alpha and beta)."""
    return P[0] if P.shape[0] == 1 else P[0] + P[1]


def atom_energies(bo: BondOrder, P: Tensor, n_atoms: int) -> Tensor:
    """The term's energies (eV) split equally over the two atoms of each pair ``(N,)``."""
    half = 0.5 * pair_energies(bo, P)
    return P.new_zeros(n_atoms).index_add(0, bo.atom_a, half).index_add(0, bo.atom_b, half)


def _host(bo: BondOrder, M: Tensor):
    """NumPy views of the pair blocks of ``M`` and of the term's values (CPU, no gradients)."""
    v = M.numpy().reshape(-1)[bo.gather.numpy()].reshape(-1, bo.k, bo.k)
    if bo.masked:
        v = v * bo.mask.numpy()
    return v, bo.coef.detach().numpy(), bo.lower.detach().numpy(), bo.width.detach().numpy()


def _unit(x: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, 0.0), 1.0)


def _host_scatter(bo: BondOrder, blk: np.ndarray) -> Tensor:
    n = bo.n_struct * bo.n_basis * bo.n_basis
    flat = blk.reshape(-1)
    out = np.zeros(n + 1, dtype=blk.dtype)
    out[bo.put.numpy()] = np.concatenate([flat, flat])        # distinct destinations but the spare element
    return torch.from_numpy(out[:n].reshape(bo.n_struct, bo.n_basis, bo.n_basis))


def fock_host(bo: BondOrder, P: Tensor) -> Tuple[Optional[Tensor], Tensor]:
    """:func:`fock` in NumPy, for the iterations of an SCF on the CPU (no gradients): the same formulas without
    the cost of a PyTorch operation per step; no Fock matrices (``None``) where they vanish."""
    Pab, coef, lower, width = _host(bo, P)
    S, S1 = _step(_unit(((Pab * Pab).sum((1, 2)) - lower) / width))
    E = torch.from_numpy(np.bincount(bo.struct.numpy(), weights=coef * S, minlength=bo.n_struct))
    if not S1.any():
        return None, E                       # every bond index on a plateau of the step
    return _host_scatter(bo, (coef * S1 / width)[:, None, None] * Pab), E


def linearized_host(bo: BondOrder, P: Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:func:`linearized` in NumPy (for the response iterations on the CPU)."""
    Pab, coef, lower, width = _host(bo, P)
    x = _unit(((Pab * Pab).sum((1, 2)) - lower) / width)
    return Pab, coef * 30.0 * x * x * (1.0 - x) ** 2 / width, coef * 60.0 * x * (1.0 - x) * (1.0 - 2.0 * x) / width ** 2


def kernel_host(bo: BondOrder, Pab: np.ndarray, a1: np.ndarray, a2: np.ndarray, X: Tensor) -> Tensor:
    """:func:`kernel_at` in NumPy."""
    Xab = _host(bo, X)[0]
    dW = 2.0 * (Pab * Xab).sum((1, 2))
    return _host_scatter(bo, a1[:, None, None] * Xab + (a2 * dW)[:, None, None] * Pab)
