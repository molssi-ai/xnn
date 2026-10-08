"""Global charge solve for LES: charge equilibration with the latent Ewald kernel.

The latent charges of one channel are not read off the atom features but chosen
to minimise, per structure::

    E(q) = sum_i chi_i q_i + 1/2 sum_i J_i q_i^2 + E_lr(q),   with sum_i q_i = Q,

where ``chi`` (an electronegativity, from the q head) and ``J > 0`` (a hardness,
learned) come from the network and ``E_lr`` is the Ewald energy of
:class:`~xnn.common.models.les.EwaldSummation` for that channel, so each charge
responds to every other charge in the system, not only to its receptive field
(the charge-induced-dipole energy of an ion in a polar liquid is the case this
exists for). Without the Coulomb coupling the solution is the net-charge
constraint of :meth:`~xnn.common.models.les.LatentEwald.constrain` with weights
``1/J``; with ``J`` large it is the local head.

``E_lr`` is an exact quadratic form, ``E_lr = 1/2 q^T gamma q``;
:func:`coulomb_matrix` writes ``gamma`` in closed form from the same kernel
functions. The stationarity conditions ``chi + J q + gamma q = mu`` with the
constraint row form one symmetric augmented system, solved by LU with the
derivatives taken through the residual (:func:`~xnn.common.models.eeq.lu_solve_implicit`):
forces and stress then need no charge response (the energy is the minimised
functional itself), and force training gets its mixed second derivatives at
O(N^2) in the backward.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

from .eeq import lu_solve_implicit
from .ops import cell_volume

_SYMBOLS = ("X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si",
            "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
            "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc",
            "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La",
            "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
            "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn")
#: formal charges of the monatomic ions a fragment solve knows by default: each
#: of these atoms is its own fragment with this charge; every other fragment is
#: neutral unless the structure carries a ``fragment_charges`` label
ION_CHARGES = {"Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1, "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2,
               "Zn": 2, "F": -1, "Cl": -1, "Br": -1, "I": -1}


def ion_charge_table(ions: Optional[dict] = None) -> Tensor:
    """Formal ion charges indexed by atomic number, ``(119,)``; zero for the rest.

    Parameters
    ----------
    ions : dict, optional
        Element symbol or atomic number to charge; :data:`ION_CHARGES` by default.
    """
    table = torch.zeros(119)
    for key, value in (ION_CHARGES if ions is None else ions).items():
        z = key if isinstance(key, int) else _SYMBOLS.index(str(key))
        table[z] = float(value)
    return table


def covalent_radii() -> Tensor:
    """Covalent radii in Angstrom indexed by atomic number, ``(119,)`` (the
    D4 reference table)."""
    from .d4 import _load_reference
    return torch.tensor(_load_reference()["covalent_radius_aa"], dtype=torch.get_default_dtype())


def bonded_fragments(atomic_numbers: Tensor, edge_index: Tensor, edge_vec: Tensor,
                     radii: Tensor, ions: Tensor, bond_factor: float = 1.2) -> Tensor:
    """Fragment index of every atom, ``(N,)``, from covalent connectivity.

    Two atoms are bonded when their distance is below ``bond_factor`` times the
    sum of their covalent radii; an atom with a formal ion charge is never
    bonded and forms its own fragment. The fragments are the connected
    components, numbered in the order of their first atom, found by label
    propagation over the edge list (one pass per bond of the longest chain).

    Parameters
    ----------
    atomic_numbers : Tensor
        ``(N,)``.
    edge_index, edge_vec : Tensor
        The neighbor list ``(2, E)`` and its vectors ``(E, 3)`` (any cutoff
        beyond the bond lengths; periodic images included).
    radii, ions : Tensor
        Tables indexed by atomic number: :func:`covalent_radii` and
        :func:`ion_charge_table`.
    bond_factor : float, optional
        The tolerance on the covalent-radius sum, by default 1.2.
    """
    n = atomic_numbers.shape[0]
    src, dst = edge_index[0], edge_index[1]
    z = atomic_numbers
    bonded = torch.linalg.norm(edge_vec, dim=-1) < bond_factor * (radii[z[src]] + radii[z[dst]])
    is_ion = ions[z] != 0
    bonded = bonded & ~is_ion[src] & ~is_ion[dst]
    src, dst = src[bonded], dst[bonded]
    labels = torch.arange(n, device=z.device)
    for _ in range(n):
        new = labels.scatter_reduce(0, dst, labels[src], reduce="amin", include_self=True)
        if torch.equal(new, labels):
            break
        labels = new
    return torch.unique(labels, return_inverse=True)[1]


def fragment_targets(fragments: Tensor, atomic_numbers: Tensor, ions: Tensor,
                     label: Optional[Tensor] = None,
                     total_charge: Optional[Tensor] = None) -> Tensor:
    """The net charge of every fragment in e, ``(F,)``.

    From the per-atom ``label`` (``fragment_charges``: the charge of the
    fragment each atom belongs to, which must agree within a fragment) when
    given, otherwise from the formal ion charges. The targets must add up to
    ``total_charge`` when that is known.

    Raises
    ------
    ValueError
        When the label disagrees within a fragment, or the fragment charges
        do not add up to the structure's net charge.
    """
    n_frag = int(fragments.max()) + 1
    if label is not None:
        hi = torch.full((n_frag,), -float("inf"), dtype=label.dtype, device=label.device)
        lo = torch.full((n_frag,), float("inf"), dtype=label.dtype, device=label.device)
        hi = hi.scatter_reduce(0, fragments, label, reduce="amax")
        lo = lo.scatter_reduce(0, fragments, label, reduce="amin")
        if float((hi - lo).abs().max()) > 1e-6:
            raise ValueError("fragment_charges differ between atoms of one fragment: the label "
                             "is the charge of the fragment each atom belongs to, with fragments "
                             "found by connectivity (ions always on their own)")
        targets = hi
    else:
        targets = torch.zeros(n_frag, dtype=ions.dtype, device=fragments.device)
        targets = targets.index_add(0, fragments, ions[atomic_numbers])
    if total_charge is not None and abs(float(targets.sum()) - float(total_charge)) > 1e-6:
        raise ValueError(
            f"the fragment charges add up to {float(targets.sum()):g} e but the structure's "
            f"total_charge is {float(total_charge):g} e; label the charged fragments with "
            "fragment_charges, or name the ions in ion_charges")
    return targets


def coulomb_matrix(ewald, pos: Tensor, cell: Optional[Tensor] = None) -> Tensor:
    """``gamma`` with ``E_lr(q) = 1/2 q^T gamma q`` for one structure.

    Differentiable in ``pos`` and ``cell``. For a cluster it is the
    ``erf(r / sqrt(2) sigma) / r`` kernel over ``2 pi`` (twice the pair term of
    :meth:`EwaldSummation.realspace`, which sums every ordered pair over
    ``4 pi``) plus the Gaussian self term on the diagonal; for a cell it is the
    Hessian of :meth:`EwaldSummation.reciprocal`, ``4/V sum_k f(k) cos k.(r_i - r_j)``
    over the half grid, with ``k = 0`` omitted.

    Parameters
    ----------
    ewald : EwaldSummation
        The kernel (``sigma``, ``dl``, ``remove_self_interaction``); ``exponent`` must be 1.
    pos : Tensor
        Positions ``(n, 3)`` in Angstrom.
    cell : Tensor or None
        Row-vector cell ``(3, 3)`` for a periodic structure.

    Returns
    -------
    Tensor
        ``(n, n)``, symmetric positive definite.
    """
    if ewald.exponent != 1:
        raise ValueError("the charge solve needs the Coulomb kernel (exponent=1)")
    n = pos.shape[0]
    eye = torch.eye(n, dtype=pos.dtype, device=pos.device)
    self_term = 2.0 / (ewald.sigma * (2.0 * math.pi) ** 1.5)
    if cell is None:
        # the same expressions as realspace, so 1/2 q^T gamma q is its energy exactly
        dist = torch.linalg.norm(pos[:, None, :] - pos[None, :, :], dim=-1)
        screen = torch.special.erf(dist / (ewald.sigma * math.sqrt(2.0)))
        gamma = screen / (dist + 1e-6) / (2.0 * math.pi)
        if not ewald.remove_self_interaction:
            gamma = gamma + eye * self_term
        return gamma
    _, kpts, k2 = ewald.k_set(cell)
    weights = ewald._kfac(k2)                                  # (M,)
    phase = pos @ kpts.T                                       # (n, M)
    cos, sin = torch.cos(phase), torch.sin(phase)
    gamma = 4.0 / cell_volume(cell) * ((cos * weights) @ cos.T + (sin * weights) @ sin.T)
    if ewald.remove_self_interaction:
        gamma = gamma - eye * self_term
    return gamma


def solve_charges(chi: Tensor, hardness: Tensor, gamma: Tensor, target: Tensor,
                  fragments: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    """The constrained minimum of ``chi.q + 1/2 sum J q^2 + 1/2 q^T gamma q``.

    Parameters
    ----------
    chi : Tensor
        Electronegativities ``(n,)`` (eV per latent charge).
    hardness : Tensor
        ``J > 0``, ``(n,)`` (eV per latent charge squared).
    gamma : Tensor
        The Coulomb matrix ``(n, n)`` of :func:`coulomb_matrix` (zero for the
        coupling-free limit).
    target : Tensor
        The net latent charge of the structure (scalar), or of every fragment
        ``(F,)`` when ``fragments`` is given.
    fragments : Tensor, optional
        Fragment index of every atom ``(n,)``, values ``0 .. F-1``: one
        constraint row per fragment instead of one for the structure.

    Returns
    -------
    (Tensor, Tensor)
        The charges ``q`` ``(n,)`` and the chemical potentials ``mu``, one per
        constraint row (``(1,)`` or ``(F,)``), with ``dE/dq_i = mu`` of its
        row for every atom at the solution. Both carry gradients with respect
        to every input, through the residual of the detached factor (first
        and second derivatives).
    """
    n = chi.shape[0]
    if fragments is None:
        rows = torch.ones((1, n), dtype=chi.dtype, device=chi.device)
    else:
        rows = torch.nn.functional.one_hot(fragments).to(chi.dtype).t()       # (F, n)
    m = rows.shape[0]
    full = gamma + torch.diag(hardness)
    zero = torch.zeros((m, m), dtype=chi.dtype, device=chi.device)
    full = torch.cat([torch.cat([full, rows.t()], dim=1), torch.cat([rows, zero], dim=1)], dim=0)
    rhs = torch.cat([-chi, target.to(chi.dtype).reshape(m)]).unsqueeze(1)
    lu, pivots = torch.linalg.lu_factor(full.detach())
    # one float64-residual refinement round in float32 (the plain factor in float64)
    sol = lu_solve_implicit(lu, pivots, full, rhs, 1 if chi.dtype == torch.float32 else 0)
    # the augmented system carries -mu in its last unknowns
    return sol[:n, 0], -sol[n:, 0]


def charge_energy(chi: Tensor, hardness: Tensor, q: Tensor) -> Tensor:
    """Per-atom ``chi_i q_i + 1/2 J_i q_i^2``, ``(n,)``; the Ewald part is
    :class:`EwaldSummation`'s energy of the same charges."""
    return chi * q + 0.5 * hardness * q * q
