"""Self-consistent-charge tight binding: the machinery shared by the GFN-xTB and DFTB methods.

A method builds, for a batch of molecules, the overlap ``S`` and Hamiltonian
``H0`` of its minimal valence basis and the charge-dependent energy of the
Mulliken shell charges (and, for GFN2-xTB, of atomic multipole moments). The
cycle here then mixes the charges until they are self-consistent:

    F = H0 - (S V + V S) / 2,    V = dE / dq   (shell potentials on the orbitals),

with Fermi occupations of the levels of ``F C = S C eps`` in two spin channels.
The energy is ``sum P H0 + E(q) + E_classical - T S_el``. Its gradient is the
derivative at a fixed density (the energy is stationary in it); its exact
second derivatives come from the response of the density (see
:meth:`TightBinding.energy_gradient`). Derivatives in trained parameters solve
the linear response of the self-consistent equations.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from xnn.common.data import AtomicGraph
from xnn.common.models import InteratomicPotential

#: Shells, atomic orbitals and their structures of a batch (made by :meth:`TightBinding._make_basis`): the
#: cycle reads ``ao_shell``, ``ao_atom``, ``ao_struct``, ``ao_index``, ``n_ao``, ``nb``, ``shell_atom``,
#: ``shell_struct``, ``shell_struct_t`` and ``shell_l``.
Basis = Any


class TightBinding(InteratomicPotential):
    """Base class of the self-consistent-charge tight-binding methods (see the module docstring).

    A method sets ``electronic_temperature``, ``max_iter``, ``mixing``, ``history``, ``supported``,
    ``trainable``, ``scc_terms`` and ``reported_terms`` and implements the hooks: the basis, the
    integrals, the Hamiltonian, the reference occupations, the charge-dependent and classical energies.
    """

    #: Second derivatives of the energy by plain autograd lack the response of the electronic density;
    #: :meth:`energy_gradient` gives a gradient whose derivatives are exact (:func:`xnn.common.models.hessian`
    #: takes it).
    exact_second_derivatives = False
    #: Interactions of atomic dipoles and second moments (their Mulliken-like moments are mixed too).
    multipoles = False
    #: Collinear spin polarization: two spin channels with their own Fock matrices; the shell magnetizations are
    #: mixed with the charges and reach the energy terms as ``ctx["m_shell"]``.
    spin_polarized = False
    #: The Fock matrix depends on the whole density matrix (long-range exchange): the cycle mixes the density
    #: matrices with the charges, :meth:`density_fock` adds their term and the energy terms read ``ctx["P"]``.
    density_mixing = False
    #: A cycle longer than this is run again with half the damping (see :meth:`_safeguarded`).
    long_cycle = 100
    #: Electronic temperature (K) of the warm start of a cycle that did not converge.
    warm_temperature = 3000.0
    #: Relative residual that ends the linear solve of the density response in training.
    response_tolerance = 1e-10
    #: Orbitals with occupations below this are left out of the density matrix.
    occupation_neglect = 1e-20
    #: Level (hartree) of the padding orbitals of a batch.
    padding_level = 1.0e3
    #: Smallest smearing energy ``kT`` (hartree): degenerate partly filled levels then share their electrons.
    minimum_kt = 0.0
    #: Bohr radius (Angstrom), Hartree energy (eV) and Boltzmann constant (hartree per kelvin) of the method.
    bohr = 0.5291772109044924
    hartree = 27.21138505
    boltzmann = 3.166808578545117e-06
    trainable: Tuple[str, ...] = ()
    scc_terms: Tuple[str, ...] = ()
    reported_terms: Tuple[str, ...] = ()

    def _make_basis(self, z: List[int], batch: List[int], n_struct: int, dtype, device) -> Basis:
        """The shells and atomic orbitals of a batch."""
        raise NotImplementedError

    def _integrals(self, basis: Basis, pos: Tensor, n_struct: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Dense overlap matrices ``(B, nb, nb)`` and, with multipoles, the dipole ``(B, 3, nb, nb)`` and
        second-moment ``(B, 6, nb, nb)`` integrals (empty tensors otherwise)."""
        raise NotImplementedError

    def _hamiltonian(self, z: List[int], basis: Basis, pos: Tensor, S: Tensor, cn: Optional[Tensor],
                     n_struct: int) -> Tensor:
        """The charge-independent Hamiltonian ``(B, nb, nb)``."""
        raise NotImplementedError

    def _electron_counts(self, n_el: List[int], multiplicity: Optional[List[int]], dtype, device
                         ) -> Tuple[Tensor, Tensor]:
        """Electrons of the two spin channels per structure: a given multiplicity ``m`` puts ``m - 1``
        more electrons in the first; without one, an odd count puts one more there."""
        uhf = [(m - 1 if multiplicity is not None and m > 0 else n % 2)
               for n, m in zip(n_el, multiplicity or [0] * len(n_el))]
        return (torch.tensor([(n + u) / 2 for n, u in zip(n_el, uhf)], dtype=dtype, device=device),
                torch.tensor([(n - u) / 2 for n, u in zip(n_el, uhf)], dtype=dtype, device=device))

    def _scc_threshold(self) -> float:
        """The value of :meth:`_scc_error` below which the cycle has converged."""
        raise NotImplementedError

    def _scc_error(self, r: Tensor, owner: Tensor, count: Tensor) -> Tensor:
        """Convergence measure per structure of the change ``r`` of the mixed quantities (``owner``: their
        structures, ``count``: how many each structure has): the root-mean-square change."""
        return torch.sqrt(torch.zeros_like(count).index_add(0, owner, r * r) / count)

    def _padded(self, H0: Tensor, basis: Basis) -> Tensor:
        """``H0`` with the padding orbitals of the smaller structures of a batch (unit overlap, no coupling) at
        ``padding_level``, above every real level, so that the occupations never reach them."""
        pad = [b for b, n in enumerate(basis.n_ao) if n < basis.nb]
        if not pad:
            return H0
        level = torch.zeros(H0.shape[0], basis.nb, dtype=H0.dtype, device=H0.device)
        for b in pad:
            level[b, basis.n_ao[b]:] = self.padding_level
        return H0 + torch.diag_embed(level)

    def _integrals_trained(self, basis: Basis) -> bool:
        """Whether the overlap depends on trained parameters (then it is formed again in training)."""
        return False

    def _on_host(self, data: AtomicGraph) -> bool:
        """Whether the batch is evaluated by a copy of the model on the host."""
        return False

    def _initial_charges(self, z: Tensor, pos: Tensor, batch: Tensor, n_struct: int, charge: Tensor,
                         basis: Basis, n0: Tensor) -> Optional[Tensor]:
        """Shell charges that start the cycle; ``None`` starts from neutral atoms."""
        return None

    def reference_occupations(self, z: int) -> List[float]:
        """Electrons per shell of the neutral reference atom."""
        raise NotImplementedError

    def coordination_numbers(self, z: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
        """Coordination numbers of the atoms (pairs within each structure)."""
        raise NotImplementedError

    def classical_energy(self, z: Tensor, pos: Tensor, batch: Tensor, n_struct: int) -> Dict[str, Tensor]:
        """Repulsion and the charge-independent terms, per structure (hartree)."""
        raise NotImplementedError

    def _terms(self, q_shell: Tensor, mu: Tensor, theta: Tensor, ctx: dict) -> Dict[str, Tensor]:
        """Charge- and moment-dependent energies per structure (hartree), by name: those of
        ``scc_terms`` add up to the energy, those of ``reported_terms`` are returned by the model."""
        raise NotImplementedError

    def second_order_energy(self, q_shell: Tensor, mu: Tensor, theta: Tensor, ctx: dict,
                            per_structure: bool = False) -> Tensor:
        """Charge- and moment-dependent energy (hartree), summed over the batch or per structure."""
        terms = self._terms(q_shell, mu, theta, ctx)
        e = sum(terms[k] for k in self.scc_terms)
        return e if per_structure else e.sum()

    # the calculation

    def energy_gradient(self, data: AtomicGraph, create_graph: bool = False) -> Tuple[Tensor, Tensor]:
        """The positions and the gradient of the energy with its graph, whose derivatives are exact.

        The free energy is stationary in the self-consistent density, so its gradient is the derivative
        at a fixed density (with the overlap term of the energy-weighted density). The energy is formed
        with two aliases of the positions, one for the explicit dependences and one for the density
        (a function of the positions through the self-consistent equations, see
        :meth:`_responsive_density`); the gradient is taken in the first. Differentiated once more
        through both, it gives the exact Hessian.

        Parameters
        ----------
        data : AtomicGraph
            The structures.
        create_graph : bool, optional
            The Hessian will be differentiated once more (parameter gradients of curvatures): the
            evaluation takes the reference's functions, whose derivatives go to any order (the fast
            path's kernels stop at the second); default ``False``.

        Returns
        -------
        tuple of Tensor
            The positions (a tensor requiring gradients; ``data.pos`` itself when it does) and the
            gradient ``(N, 3)`` in eV / Angstrom, differentiable in them.
        """
        pos = data.pos if data.pos.requires_grad else data.pos.detach().requires_grad_(True)
        explicit, density = pos.view_as(pos), pos.view_as(pos)
        self.__dict__["_density_positions"] = density
        use_fast = self.use_fast
        if create_graph:
            self.use_fast = False
            self.__dict__["_second_order_density"] = True
        try:
            with torch.enable_grad():
                energy = self(dataclasses.replace(data, pos=explicit))["energy"]
                (grad,) = torch.autograd.grad(energy.sum(), explicit, create_graph=True)
        finally:
            self.__dict__.pop("_density_positions", None)
            self.__dict__.pop("_second_order_density", None)
            self.use_fast = use_fast
        return pos, grad

    def forward(self, data: AtomicGraph) -> Dict[str, Tensor]:
        """Total energies (eV), charges and moments of a batch of molecules."""
        # the density at positions of its own (an alias of the positions, see energy_gradient)
        density_pos = self.__dict__.get("_density_positions") if torch.is_grad_enabled() else None
        if density_pos is None and self._on_host(data):
            twin = self._host_model()
            out = twin(data.to("cpu"))
            self.last_scf = twin.last_scf
            return {k: (v.to(data.pos.device) if isinstance(v, Tensor) else v) for k, v in out.items()}
        dtype = torch.float64
        device = data.pos.device
        pos = data.pos.to(dtype) / self.bohr
        z = data.atomic_numbers.long()
        zl = z.tolist()
        missing = sorted(set(zl) - set(self.supported))
        if missing:
            raise ValueError(f"{type(self).__name__} has no parameters for Z = {missing}")
        batch = data.batch
        n_struct = data.num_graphs
        bl = batch.tolist()
        basis = self._make_basis(zl, bl, n_struct, dtype, device)

        charge = (data.total_charge.to(dtype) if data.total_charge is not None
                  else torch.zeros(n_struct, dtype=dtype, device=device))
        self._begin(z, pos, batch, n_struct, charge)

        # integrals and Hamiltonian (differentiable in the positions)
        S, DA, QA = self._integrals(basis, pos, n_struct)
        cn = self.coordination_numbers(z, pos, batch)
        H0 = self._padded(self._hamiltonian(zl, basis, pos, S, cn, n_struct), basis)
        ctx = self.scc_context(z, pos, basis, cn, batch, n_struct)

        n0 = torch.tensor([o for za in zl for o in self.reference_occupations(za)], dtype=dtype, device=device)
        shell_struct = torch.tensor(basis.shell_struct, dtype=torch.long, device=device)
        n_el = torch.zeros(n_struct, dtype=dtype, device=device).index_add(0, shell_struct, n0) - charge
        n_el_i = [int(round(float(x))) for x in n_el]
        mult = getattr(data, "spin_multiplicity", None)
        n_alpha, n_beta = self._electron_counts(n_el_i, None if mult is None else [int(m) for m in mult], dtype,
                                                device)
        # several k-points: every orbital appears in several copies, which hold as many times the electrons
        copies = self._copies(basis, n_struct, dtype, device)
        n_alpha, n_beta = n_alpha * copies, n_beta * copies

        with torch.no_grad():
            start = self._initial_charges(z, pos.detach(), batch, n_struct, charge, basis, n0)
            inputs = [t.detach() for t in (S, DA, QA, H0)]
            scc_ctx = {k: (v.detach() if isinstance(v, Tensor) else v) for k, v in ctx.items()}
            res = self._scc(basis, *inputs, scc_ctx, n0, n_alpha, n_beta, n_struct, start)
            if start is not None and not bool(res["converged"]):
                # the cycle did not converge from the model charges: start from neutral atoms
                retry = self._scc(basis, *inputs, scc_ctx, n0, n_alpha, n_beta, n_struct, None)
                if bool(retry["converged"]):
                    res = retry
            res = self._safeguarded(res, basis, inputs, scc_ctx, n0, n_alpha, n_beta, n_struct, start)
        self.last_scf = res
        P, W, ts = res.get("P2", res["P"]), res["W"], res["ts"]
        fixed = S.detach()
        if density_pos is not None:
            # the density as a function of the positions (and of trained parameters)
            pos_s = density_pos.to(dtype) / self.bohr
            still = self._integrals(basis, pos_s, n_struct)
            fixed = still[0]
            P, W, ts = self._responsive_density(z, zl, pos_s, batch, basis, *still, n0, n_alpha, n_beta, n_struct,
                                                res["x"])
        elif self.trainable and self.training and torch.is_grad_enabled():
            # training: the density follows the parameters through the self-consistent solution
            still = (fixed, DA.detach(), QA.detach())
            if self._integrals_trained(basis):
                # trained exponents: the integrals at fixed positions, as functions of the exponents
                still = self._integrals(basis, pos.detach(), n_struct)
                fixed = still[0]
            P, W, ts = self._responsive_density(z, zl, pos.detach(), batch, basis, *still, n0, n_alpha, n_beta,
                                                n_struct, res["x"])
        m_shell, P_mixed = None, P
        if self.spin_polarized:
            P, q_shell, m_shell = self._spin_densities(basis, P_mixed, S, n0)
            mu, theta = q_shell.new_zeros(z.shape[0], 3), q_shell.new_zeros(z.shape[0], 6)
            ctx = dict(ctx, m_shell=m_shell)
        else:
            q_shell, mu, theta = self._moments(basis, P, S, DA, QA, n0)
        if self.density_mixing:
            ctx = dict(ctx, P=P_mixed)                  # spin-polarized: the stacked channel densities
        e_h0 = self._per_structure((P * H0).sum((-1, -2)), basis, n_struct) / copies
        ts = self._per_structure(ts, basis, n_struct) / copies
        terms = self._terms(q_shell, mu, theta, ctx)
        e_scc = sum(terms[k] for k in self.scc_terms)
        cls = self.classical_energy(z, pos, batch, n_struct)
        # energy-weighted density: makes the gradient include the overlap derivative (a zero that
        # depends on the positions only through the overlap)
        e_total = (e_h0 + e_scc + cls["total"] - ts
                   - self._per_structure((W * (S - fixed)).sum((-1, -2)), basis, n_struct) / copies)
        energy = e_total * self.hartree
        n_atoms = z.shape[0]
        counts = torch.zeros(n_struct, dtype=dtype, device=device).index_add(0, batch, torch.ones_like(pos[:, 0]))
        node_energy = (energy / counts)[batch]
        q_atom = torch.zeros(n_atoms, dtype=dtype, device=device).index_add(0, basis.shell_atom, q_shell)
        out = {
            "energy": energy,
            "node_energy": node_energy,
            "charges": q_atom.detach(),
            "atomic_dipoles": mu.detach(),
            "atomic_quadrupoles": theta.detach(),
            "electronic_energy": ((e_h0 + e_scc - ts) * self.hartree).detach(),
            "repulsion_energy": (cls["repulsion"] * self.hartree).detach(),
            "scf_converged": res["converged"],
        }
        if m_shell is not None:
            out["magnetizations"] = torch.zeros_like(q_atom).index_add(0, basis.shell_atom, m_shell).detach()
        for k, v in cls.items():
            if k not in ("total", "repulsion"):
                out[f"{k}_energy"] = (v * self.hartree).detach()
        for k in self.reported_terms:
            out[f"{k}_energy"] = (terms[k] * self.hartree).detach()
        return out

    def scc_context(self, z, pos, basis, cn, batch, n_struct) -> dict:
        return {}

    @staticmethod
    def _per_structure(values: Tensor, basis: Basis, n_struct: int) -> Tensor:
        """Sums of per-row values over the rows of each structure (one row per structure unless a method samples
        several k-points)."""
        groups, weights = _row_groups(basis), _row_weights(basis)
        if groups is None:
            return values
        if weights is not None:
            values = values * weights.to(values.dtype)
        return values.new_zeros(n_struct).index_add(0, groups, values)

    @staticmethod
    def _copies(basis: Basis, n_struct: int, dtype, device) -> Tensor:
        """Copies of the orbitals of each structure in its matrices (one unless a method samples several
        k-points, as blocks of a real form of their complex matrices)."""
        copies = getattr(basis, "copies", None)
        if copies is None:
            return torch.ones(n_struct, dtype=dtype, device=device)
        return torch.as_tensor(copies, dtype=dtype, device=device)

    def _begin(self, z: Tensor, pos: Tensor, batch: Tensor, n_struct: int, charge: Tensor) -> None:
        """What a method prepares from the structures and their total charges before the Hamiltonian."""

    def _moments(self, basis: Basis, P: Tensor, S: Tensor, DA: Tensor, QA: Tensor, n0: Tensor,
                 stack: Optional[Tensor] = None):
        """Mulliken shell charges, atomic dipoles and second moments (atomic units); with the integral
        matrices as ``stack[b, nu, c, mu]``, one product per column gives all of them."""
        n_shell = len(basis.shell_l)
        n_atoms = int(basis.ao_atom.max()) + 1 if basis.ao_atom.numel() else 0
        b, i = basis.ao_struct, basis.ao_index
        nb = P.shape[-1]
        flat = b * nb + i
        if stack is not None:
            sums = torch.matmul(stack, P.transpose(1, 2).unsqueeze(-1)).squeeze(-1)      # (B, nb, n_ops)
            sums = sums.reshape(-1, sums.shape[-1]).index_select(0, flat)
            if getattr(basis, "ao_copies", None) is not None:
                sums = sums / basis.ao_copies[:, None]
            q_shell = n0 - torch.zeros(n_shell, dtype=P.dtype, device=P.device).index_add(0, basis.ao_shell, sums[:, 0])
            mu = torch.zeros(n_atoms, 3, dtype=P.dtype, device=P.device)
            theta = torch.zeros(n_atoms, 6, dtype=P.dtype, device=P.device)
            if self.multipoles:
                mu = mu.index_add(0, basis.ao_atom, -sums[:, 1:4])
                theta = theta.index_add(0, basis.ao_atom, -sums[:, 4:10])
            return q_shell, mu, theta
        PS = (P * S).sum(-1)                                         # (B, nb): sum_nu P_mu,nu S_nu,mu
        pop_ao = PS.reshape(-1).index_select(0, flat)
        if getattr(basis, "ao_copies", None) is not None:
            pop_ao = pop_ao / basis.ao_copies                      # several k-points: the mean of the copies
        pop_shell = torch.zeros(n_shell, dtype=P.dtype, device=P.device).index_add(0, basis.ao_shell, pop_ao)
        q_shell = n0 - pop_shell
        if not self.multipoles:
            return (q_shell, torch.zeros(n_atoms, 3, dtype=P.dtype, device=P.device),
                    torch.zeros(n_atoms, 6, dtype=P.dtype, device=P.device))
        # dipole: mu_A = - sum_{k in A} sum_l P_lk DA[l, k]
        pd = (P[:, None] * DA).sum(-2)                               # (B, 3, nb): column sums over l
        pq = (P[:, None] * QA).sum(-2)                               # (B, 6, nb)
        pd = pd.transpose(1, 2).reshape(-1, 3).index_select(0, flat)
        pq = pq.transpose(1, 2).reshape(-1, 6).index_select(0, flat)
        mu = -torch.zeros(n_atoms, 3, dtype=P.dtype, device=P.device).index_add(0, basis.ao_atom, pd)
        theta = -torch.zeros(n_atoms, 6, dtype=P.dtype, device=P.device).index_add(0, basis.ao_atom, pq)
        return q_shell, mu, theta

    def _ao_potentials(self, basis: Basis, v_shell: Tensor, v_mu: Tensor, v_theta: Tensor, n_struct: int) -> Tensor:
        """The potentials of the shell charges and, with multipoles, of the atomic dipoles and second
        moments on the atomic orbitals, ``(B, n_ops, nb)``. The Fock matrix is
        ``H0 - (Z + Z^T) / 2`` with ``Z[mu, nu] = sum_c M_c[mu, nu] V[c, nu]`` over the overlap, dipole
        and second-moment matrices ``M_c``."""
        vals = v_shell[basis.ao_shell][:, None]
        if self.multipoles:
            vals = torch.cat([vals, v_mu[basis.ao_atom], v_theta[basis.ao_atom]], 1)
        V = vals.new_zeros(n_struct, vals.shape[1], basis.nb)
        V[basis.ao_struct, :, basis.ao_index] = vals
        return V

    def potentials(self, q_shell: Tensor, mu: Tensor, theta: Tensor, ctx: dict) -> Tuple[Tensor, Tensor, Tensor]:
        """Derivatives of the charge-dependent energy with respect to the shell charges, atomic
        dipoles and second moments (by automatic differentiation; the methods give closed forms)."""
        inputs = [t.detach().clone().requires_grad_(True) for t in (q_shell, mu, theta)]
        with torch.enable_grad():
            e = self.second_order_energy(*inputs, ctx)
            grads = torch.autograd.grad(e, inputs, allow_unused=True)
        return tuple(g if g is not None else torch.zeros_like(t) for g, t in zip(grads, inputs))

    def density_fock(self, P: Tensor, ctx: dict) -> Tensor:
        """The density-dependent part of the Fock matrices ``(B, nb, nb)`` in the atomic orbitals."""
        return torch.zeros_like(P)

    def reference_density(self, basis: Basis, n0: Tensor, S: Tensor) -> Tensor:
        """The density matrices of the neutral reference atoms (diagonal; the shell occupations spread over the
        shell's orbitals), ``(B, nb, nb)``."""
        per = n0.index_select(0, basis.ao_shell) / (2.0 * basis.shell_l_t.index_select(0, basis.ao_shell) + 1.0)
        diag = S.new_zeros(S.shape[0], S.shape[-1])
        diag[basis.ao_struct, basis.ao_index] = per
        return torch.diag_embed(diag)

    def spin_potentials(self, m_shell: Tensor, ctx: dict) -> Tensor:
        """Derivative of the energy in the shell magnetizations (spin-polarized methods)."""
        return torch.zeros_like(m_shell)

    def _fock(self, q_shell: Tensor, mu: Tensor, theta: Tensor, basis: Basis, ctx: dict, ortho: "_Orthogonalizer",
              left: Tensor, h0: Tensor) -> Tensor:
        """The Fock matrix in the orthogonal basis from the charges and moments: one product (or
        triangular solve) with the transformed integral matrices ``left`` (in the layout of
        :func:`_stacked`) and ``h0``."""
        return self._fock_from(self._ao_potentials(basis, *self.potentials(q_shell, mu, theta, ctx), left.shape[0]),
                               ortho, left, h0)

    @staticmethod
    def _fock_from(V: Tensor, ortho: "_Orthogonalizer", left: Tensor, h0: Tensor) -> Tensor:
        """``h0 - (Z + Z^T) / 2`` in the orthogonal basis for the orbital potentials ``V``."""
        # Z^T[nu, mu] = sum_c left_c[mu, nu] V[c, nu]: one product per column over its operators
        Zt = torch.matmul(left, V.transpose(1, 2).unsqueeze(-1)).squeeze(-1)
        K = ortho.left(Zt)
        return h0 - 0.5 * (K + K.transpose(-1, -2))

    def _spin_fock(self, q_shell: Tensor, m_shell: Tensor, basis: Basis, ctx: dict, ortho: "_Orthogonalizer",
                   left: Tensor, h0: Tensor) -> Tensor:
        """The Fock matrices of the two spin channels, up then down, from the doubled ``left`` and ``h0``: the
        orbital potentials ``v_q -+ v_m`` (``m`` the up minus down populations)."""
        n_struct = h0.shape[0] // 2
        n_atoms = int(basis.ao_atom.max()) + 1
        zm, zt = q_shell.new_zeros(n_atoms, 3), q_shell.new_zeros(n_atoms, 6)
        v_q = self.potentials(q_shell, zm, zt, ctx)[0]
        v_m = self.spin_potentials(m_shell, ctx)
        V = torch.cat([self._ao_potentials(basis, v_q - v_m, zm, zt, n_struct),
                       self._ao_potentials(basis, v_q + v_m, zm, zt, n_struct)])
        return self._fock_from(V, ortho, left, h0)

    def _populations(self, basis: Basis, P: Tensor, S: Tensor) -> Tensor:
        """Mulliken shell populations of densities ``P`` ``(B, nb, nb)``."""
        n_shell = len(basis.shell_l)
        return -self._moments(basis, P, S, S, S, P.new_zeros(n_shell))[0]

    def _spin_densities(self, basis: Basis, P2: Tensor, S: Tensor, n0: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """The total density, shell charges and shell magnetizations of the stacked channel densities."""
        n = P2.shape[0] // 2
        up, down = self._populations(basis, P2[:n], S), self._populations(basis, P2[n:], S)
        return P2[:n] + P2[n:], n0 - up - down, up - down

    def _scc_step(self, x, basis, S, DA, QA, ctx, n0, n_alpha, n_beta, ortho, left, h0, kt, n_shell, n_atoms,
                  stack: Optional[Tensor] = None):
        """One map of the cycle: charges and moments -> Fock matrices -> occupations -> new charges and moments."""
        dtype, device = S.dtype, S.device
        q_shell = x[:n_shell]
        if self.spin_polarized:
            F = self._spin_fock(q_shell, x[n_shell:2 * n_shell], basis, ctx, ortho, left, h0)
            if self.density_mixing:
                P_in = x[2 * n_shell:].view(2 * S.shape[0], *S.shape[1:])
                F = F + ortho.right(ortho.left(self.density_fock(P_in, ctx)))
            eps, Co = torch.linalg.eigh(F)
            groups, weights = _row_groups(basis), _row_weights(basis)
            if groups is not None:
                groups = torch.cat([groups, groups + n_alpha.shape[0]])
                weights = None if weights is None else torch.cat([weights, weights])
            occ, ts = _fermi(eps, torch.cat([n_alpha, n_beta]), torch.zeros_like(torch.cat([n_alpha, n_beta])), kt,
                             basis.n_ao * 2, groups, weights)
            n_occ = max(1, int((occ > self.occupation_neglect).sum(-1).max()))
            eps, occ = eps[:, :n_occ], occ[:, :n_occ]
            C = ortho.back(Co[:, :, :n_occ])
            P2 = (C * occ[:, None, :]) @ C.transpose(-1, -2)
            P, qn, mn = self._spin_densities(basis, P2, S, n0)
            n = eps.shape[0] // 2
            xn = torch.cat([qn, mn, P2.reshape(-1)]) if self.density_mixing else torch.cat([qn, mn])
            return {"P": P, "P2": P2, "C": C, "eps": eps, "occ": occ, "ts": ts[:n] + ts[n:]}, xn
        if self.multipoles:
            mu = x[n_shell:n_shell + 3 * n_atoms].view(n_atoms, 3)
            theta = x[n_shell + 3 * n_atoms:].view(n_atoms, 6)
        else:
            mu = torch.zeros(n_atoms, 3, dtype=dtype, device=device)
            theta = torch.zeros(n_atoms, 6, dtype=dtype, device=device)
        F = self._fock(q_shell, mu, theta, basis, ctx, ortho, left, h0)
        if self.density_mixing:
            P_in = x[n_shell:].view(S.shape)
            F = F + ortho.right(ortho.left(self.density_fock(P_in, ctx)))
        eps, Co = torch.linalg.eigh(F)
        occ, ts = _fermi(eps, n_alpha, n_beta, kt, basis.n_ao, _row_groups(basis), _row_weights(basis))
        # the levels are ascending: orbitals beyond the last occupied one add nothing to the density
        n_occ = max(1, int((occ > self.occupation_neglect).sum(-1).max()))
        eps, occ = eps[:, :n_occ], occ[:, :n_occ]
        C = ortho.back(Co[:, :, :n_occ])
        P = (C * occ[:, None, :]) @ C.transpose(-1, -2)
        qn, mun, thn = self._moments(basis, P, S, DA, QA, n0, stack)
        xn = torch.cat([qn, mun.reshape(-1), thn.reshape(-1)]) if self.multipoles else qn
        if self.density_mixing:
            xn = torch.cat([qn, P.reshape(-1)])
        return {"P": P, "C": C, "eps": eps, "occ": occ, "ts": ts}, xn

    def _responsive_density(self, z: Tensor, zl: List[int], pos: Tensor, batch: Tensor, basis: Basis, S: Tensor,
                            DA: Tensor, QA: Tensor, n0: Tensor, n_alpha: Tensor, n_beta: Tensor, n_struct: int,
                            x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """The converged density, energy-weighted density and ``T S_el`` as differentiable functions
        of the trained parameters (at fixed positions); spin-polarized, the stacked densities of the two
        channels in place of the density.

        With the map ``g`` of one iteration, the converged charges and moments obey ``x = g(x, p)``, so
        a gradient ``v`` in ``x`` reaches the parameters ``p`` as ``y dg/dp`` with
        ``(1 - (dg/dx)^T) y = v``; this linear system is solved by Anderson-mixed fixed-point steps,
        each one backward pass through an iteration (whose density uses first-order perturbation
        theory at the Fermi occupations, :class:`_SpectralDensity`).
        """
        if _row_groups(basis) is not None:
            raise NotImplementedError("second derivatives and training need the Gamma point of a cell")
        cn = self.coordination_numbers(z, pos, batch)
        H0 = self._padded(self._hamiltonian(zl, basis, pos, S, cn, n_struct), basis)
        ctx = self.scc_context(z, pos, basis, cn, batch, n_struct)
        spin = self.spin_polarized
        ortho = _Orthogonalizer(torch.cat([S, S]) if spin else S)
        if ortho.L is None:
            raise RuntimeError("training needs a positive definite overlap matrix")
        M = torch.cat([S[:, None], DA, QA], 1) if self.multipoles else S[:, None]
        left = _stacked(ortho.left(torch.cat([M, M]) if spin else M))
        h0 = ortho.right(ortho.left(torch.cat([H0, H0]) if spin else H0))
        n_shell, n_atoms = n0.shape[0], int(basis.ao_atom.max()) + 1
        kt = max(self.boltzmann * self.electronic_temperature, self.minimum_kt)
        if spin:
            # the channels as structures of their own, each with its electrons in one Fermi channel
            n_alpha, n_beta = torch.cat([n_alpha, n_beta]), torch.zeros_like(torch.cat([n_alpha, n_beta]))

        def fock(xx: Tensor, c: dict) -> Tensor:
            if spin:
                F = self._spin_fock(xx[:n_shell], xx[n_shell:2 * n_shell], basis, c, ortho, left, h0)
                if self.density_mixing:
                    P_in = xx[2 * n_shell:].view(2 * S.shape[0], *S.shape[1:])
                    F = F + ortho.right(ortho.left(self.density_fock(P_in, c)))
                return F
            if self.multipoles:
                mu = xx[n_shell:n_shell + 3 * n_atoms].view(n_atoms, 3)
                theta = xx[n_shell + 3 * n_atoms:].view(n_atoms, 6)
            else:
                mu, theta = xx.new_zeros(n_atoms, 3), xx.new_zeros(n_atoms, 6)
            F = self._fock(xx[:n_shell], mu, theta, basis, c, ortho, left, h0)
            if self.density_mixing:
                F = F + ortho.right(ortho.left(self.density_fock(xx[n_shell:].view(S.shape), c)))
            return F

        with torch.no_grad():
            # on a copy of the context: matrices it caches here would be constants for the density below
            eps, U = torch.linalg.eigh(fock(x, dict(ctx)))
            f, level = _fermi_channels(eps, n_alpha, n_beta, kt, basis.n_ao * (2 if spin else 1))

        def density(xx: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
            Po, Wo, ts = _SpectralDensity.apply(fock(xx, ctx), eps, U, f, level, kt)
            # back to the atomic orbitals: L^-T M L^-1 of the symmetric matrices
            P = ortho.back(ortho.back(Po).transpose(-1, -2)).transpose(-1, -2)
            W = ortho.back(ortho.back(Wo).transpose(-1, -2)).transpose(-1, -2)
            if spin:
                # the stacked channel densities, the total energy-weighted density and both shell vectors
                _, q, mag = self._spin_densities(basis, P, S, n0)
                xs = [q, mag, P.reshape(-1)] if self.density_mixing else [q, mag]
                return P, W[:n_struct] + W[n_struct:], ts[:n_struct] + ts[n_struct:], torch.cat(xs)
            q, m, th = self._moments(basis, P, S, DA, QA, n0)
            if self.density_mixing:
                return P, W, ts, torch.cat([q, P.reshape(-1)])
            return P, W, ts, (torch.cat([q, m.reshape(-1), th.reshape(-1)]) if self.multipoles else q)

        x_in = x.detach().clone().requires_grad_(True)
        x_out = density(x_in)[3]

        if self.__dict__.get("_second_order_density", False):
            # derivatives to second order (parameter gradients of Hessians): with M = (1 - dg/dx)^-1 at the
            # solution, x1 = x with the adjoint M^T (exact first derivatives, linear in the incoming gradient)
            # and the step x2 = x1 + M (g(x1) - x1), whose second derivatives are those of the solution (the
            # residual and its first derivative vanish there, so M enters as a constant)
            eye = torch.eye(x_out.numel(), dtype=x_out.dtype, device=x_out.device)
            rows = [torch.autograd.grad(x_out, x_in, e, retain_graph=True, allow_unused=True)[0] for e in eye]
            J = torch.stack([torch.zeros_like(x_out) if r is None else r for r in rows])
            M = torch.linalg.inv(eye - J)
            x1 = x_out.clone()
            x1.register_hook(lambda v: M.transpose(0, 1) @ v)
            x2 = x1 + M @ (density(x1)[3] - x1)
            return density(x2)[:3]

        def adjoint(v: Tensor) -> Tensor:
            scale = float(v.detach().abs().max())
            if scale == 0.0:
                return v
            y, ys, rs = v, [], []
            for _ in range(self.max_iter):
                back = torch.autograd.grad(x_out, x_in, y, retain_graph=True, allow_unused=True)[0]
                r = v - y if back is None else v + back - y
                if float(r.abs().max()) <= self.response_tolerance * scale:
                    break
                ys, rs = (ys + [y])[-self.history:], (rs + [r])[-self.history:]
                y = _anderson(ys, rs, 1.0)
            return y

        x_sc = x_out.clone()
        x_sc.register_hook(adjoint)
        return density(x_sc)[:3]

    def _safeguarded(self, res: dict, basis: Basis, inputs: list, ctx: dict, n0: Tensor, n_alpha: Tensor,
                     n_beta: Tensor, n_struct: int, start: Optional[Tensor]) -> dict:
        """The result of the cycle after two safeguards. A cycle that did not converge is run again
        from the charges that converge at a higher electronic temperature (``warm_temperature``). A
        cycle longer than ``long_cycle`` iterations may have several solutions: it is run again with
        half the damping, and each structure keeps the solution of lower electronic free energy."""
        def run(**options) -> dict:
            return self._scc(basis, *inputs, ctx, n0, n_alpha, n_beta, n_struct, **options)

        n_shell = n0.shape[0]
        if not bool(res["converged"]):
            hot = max(self.warm_temperature, 10.0 * self.electronic_temperature)
            warm = run(start=start, temperature=hot)
            alt = run(start=warm["x"][:n_shell])
            alt["iterations"] += res["iterations"] + warm["iterations"]
            return alt if bool(alt["converged"]) else res
        if res["iterations"] <= self.long_cycle:
            return res
        alt = run(start=start, mixing=0.5 * self.mixing)
        if not bool(alt["converged"]):
            return res
        n_atoms = int(basis.ao_atom.max()) + 1
        atom_struct = torch.zeros(n_atoms, dtype=torch.long, device=n0.device)
        atom_struct[basis.shell_atom] = basis.shell_struct_t
        owner = basis.shell_struct_t
        if self.multipoles:
            owner = torch.cat([owner, atom_struct.repeat_interleave(3), atom_struct.repeat_interleave(6)])
        if self.density_mixing:
            channels = 2 if self.spin_polarized else 1
            owner = torch.cat([owner.repeat(channels), torch.arange(n_struct, device=n0.device).repeat(channels)
                               .repeat_interleave(inputs[0][0].numel())])

        def free_energy(r: dict) -> Tensor:
            x = r["x"]
            mu = x[n_shell:n_shell + 3 * n_atoms].view(n_atoms, 3) if self.multipoles else x.new_zeros(n_atoms, 3)
            theta = x[n_shell + 3 * n_atoms:].view(n_atoms, 6) if self.multipoles else x.new_zeros(n_atoms, 6)
            c = dict(ctx, m_shell=x[n_shell:2 * n_shell]) if self.spin_polarized else ctx
            if self.density_mixing:
                channels = 2 if self.spin_polarized else 1
                c = dict(c, P=x[channels * n_shell:].view(channels * inputs[0].shape[0], *inputs[0].shape[1:]))
            terms = self._terms(x[:n_shell], mu, theta, c)
            copies = self._copies(basis, n_struct, r["ts"].dtype, r["ts"].device)
            band = self._per_structure((r["P"] * inputs[3]).sum((-1, -2)) - r["ts"], basis, n_struct)
            return band / copies + sum(terms[k] for k in self.scc_terms)

        take = free_energy(alt) < free_energy(res) - 1e-9
        out = dict(res)
        out["iterations"] = res["iterations"] + alt["iterations"]
        if not bool(take.any()):
            return out
        width = max(res["occ"].shape[-1], alt["occ"].shape[-1])
        for key in ("P", "P2", "W", "ts", "C", "eps", "occ"):
            if key not in res:
                continue
            a, b = res[key], alt[key]
            if key in ("C", "eps", "occ"):
                a = torch.nn.functional.pad(a, (0, width - a.shape[-1]))
                b = torch.nn.functional.pad(b, (0, width - b.shape[-1]))
            rows = take if _row_groups(basis) is None else take[_row_groups(basis)]
            pick = rows.repeat(a.shape[0] // rows.shape[0])                     # spin: both channels
            out[key] = torch.where(pick.view(-1, *([1] * (a.dim() - 1))), b, a)
        out["x"] = torch.where(take[owner.repeat(res["x"].shape[0] // owner.shape[0])], alt["x"], res["x"])
        return out

    def _scc(self, basis: Basis, S, DA, QA, H0, ctx, n0, n_alpha, n_beta, n_struct,
             start: Optional[Tensor] = None, mixing: Optional[float] = None,
             temperature: Optional[float] = None) -> Dict[str, Tensor]:
        """Self-consistent charges by Anderson mixing of the shell charges and atomic moments; the mixing
        restarts from the best iterate with half the damping when the residual grows or stalls.
        ``mixing`` and ``temperature`` replace the model's damping and electronic temperature."""
        dtype, device = S.dtype, S.device
        n_shell = n0.shape[0]
        n_atoms = int(basis.ao_atom.max()) + 1
        # the integral matrices and H0 with the orthogonalizing transformation applied (spin: both channels)
        spin = self.spin_polarized
        ortho = _Orthogonalizer(torch.cat([S, S]) if spin else S)
        M = torch.cat([S[:, None], DA, QA], 1) if self.multipoles else S[:, None]
        stack = None if spin else M.permute(0, 3, 1, 2).contiguous()
        left = _stacked(ortho.left(torch.cat([M, M]) if spin else M))
        del M
        h0 = ortho.right(ortho.left(torch.cat([H0, H0]) if spin else H0))
        temperature = self.electronic_temperature if temperature is None else temperature
        kt = max(self.boltzmann * temperature, self.minimum_kt)
        x = torch.zeros(n_shell * (2 if spin else 1) + 9 * n_atoms * int(self.multipoles), dtype=dtype, device=device)
        if start is not None:
            x[:n_shell] = start
        if self.density_mixing:
            if self.multipoles:
                raise NotImplementedError("density mixing works without multipoles")
            ref = self.reference_density(basis, n0, S)
            # spin-polarized: each channel starts from half the reference density
            x = torch.cat([x, torch.cat([0.5 * ref, 0.5 * ref]).reshape(-1) if spin else ref.reshape(-1)])
        # the structure of every mixed quantity, for the convergence test per structure
        atom_struct = torch.zeros(n_atoms, dtype=torch.long, device=device)
        atom_struct[basis.shell_atom] = basis.shell_struct_t
        owner = basis.shell_struct_t.repeat(2) if spin else basis.shell_struct_t
        if self.density_mixing:
            blocks = torch.arange(n_struct, device=device).repeat(2 if spin else 1)
            owner = torch.cat([owner, blocks.repeat_interleave(S[0].numel())])
        if self.multipoles:
            owner = torch.cat([owner, atom_struct.repeat_interleave(3), atom_struct.repeat_interleave(6)])
        count = torch.zeros(n_struct, dtype=dtype, device=device).index_add(0, owner, torch.ones_like(x))
        threshold = self._scc_threshold()
        xs, rs = [], []
        converged = False
        result = None
        iterations = 0
        beta = self.mixing if mixing is None else mixing
        best_err, best_x, worse = float("inf"), x.clone(), 0
        checkpoint = float("inf")
        for _ in range(self.max_iter):
            iterations += 1
            result, xn = self._scc_step(x, basis, S, DA, QA, ctx, n0, n_alpha, n_beta, ortho, left, h0, kt, n_shell,
                                        n_atoms, stack)
            r = xn - x
            err = float(r.abs().max())
            if float(self._scc_error(r, owner, count).max()) < threshold:
                converged = True
                break
            if err < best_err:
                best_err, best_x, worse = err, x.clone(), 0
            else:
                worse += 1
            stalled = iterations % 40 == 0 and best_err > 1e-2 * checkpoint
            if iterations % 40 == 0:
                checkpoint = best_err
            if (worse >= 8 and err > 2.0 * best_err) or stalled:
                # restart from the best iterate with a smaller damping
                beta, worse = 0.5 * beta, 0
                xs, rs = [], []
                x = best_x.clone()
                continue
            xs.append(x.clone())
            rs.append(r.clone())
            if len(xs) > self.history:
                xs.pop(0)
                rs.pop(0)
            x = _anderson(xs, rs, beta)
        W = (result["C"] * (result["occ"] * result["eps"])[:, None, :]) @ result["C"].transpose(-1, -2)
        result["W"] = W[:W.shape[0] // 2] + W[W.shape[0] // 2:] if spin else W
        result["x"] = x
        result["converged"] = torch.tensor(converged, device=device)
        result["iterations"] = iterations
        return result


class _Orthogonalizer:
    """The transformation ``T`` that turns ``F C = S C eps`` into a standard eigenproblem of
    ``T F T^T``: the inverse of the Cholesky factor of the overlap, ``S = L L^T``, or, where the
    factorization fails (a numerically singular overlap), ``T^T = U s^-1/2`` from the eigenvectors
    and eigenvalues of ``S``."""

    def __init__(self, S: Tensor):
        L, info = torch.linalg.cholesky_ex(S)
        self.L: Optional[Tensor] = None if bool(info.any()) else L
        self.X: Optional[Tensor] = None
        if self.L is None:
            es, us = torch.linalg.eigh(S)
            self.X = us * es.clamp_min(1e-14).rsqrt()[:, None, :]

    def left(self, M: Tensor) -> Tensor:
        """``T M`` for ``M`` of shape ``(B, nb, nb)`` or ``(B, k, nb, nb)``."""
        if self.L is None:
            Xt = self.X.transpose(-1, -2)
            return (Xt if M.dim() == 3 else Xt[:, None]) @ M
        return torch.linalg.solve_triangular(self.L if M.dim() == 3 else self.L[:, None], M, upper=False)

    def right(self, M: Tensor) -> Tensor:
        """``M T^T``."""
        if self.L is None:
            return M @ self.X
        return torch.linalg.solve_triangular(self.L.transpose(-1, -2), M, upper=True, left=False)

    def back(self, C: Tensor) -> Tensor:
        """Eigenvectors of the standard problem to those of the generalized one, ``T^T C``."""
        if self.L is None:
            return self.X @ C
        return torch.linalg.solve_triangular(self.L.transpose(-1, -2), C, upper=True)


def _stacked(M: Tensor) -> Tensor:
    """Integral matrices ``(B, n_ops, mu, nu)`` as ``(B, nu, mu, n_ops)``: the operators of an element
    side by side, the elements of a column in a row."""
    return M.permute(0, 3, 2, 1).contiguous()


def _anderson(xs: List[Tensor], rs: List[Tensor], beta: float) -> Tensor:
    """Anderson (Pulay) mixing of the iterates ``xs`` with residuals ``rs``."""
    m = len(xs)
    if m == 1:
        return xs[0] + beta * rs[0]
    R = torch.stack(rs)
    A = R @ R.T
    A = A + torch.eye(m, dtype=A.dtype, device=A.device) * 1e-14 * A.diagonal().max()
    ones = torch.ones(m, dtype=A.dtype, device=A.device)
    try:
        w = torch.linalg.solve(A, ones)
    except RuntimeError:
        return xs[-1] + beta * rs[-1]
    c = w / w.sum()
    X = torch.stack(xs)
    return (c[:, None] * (X + beta * R)).sum(0)


def _row_groups(basis) -> Optional[Tensor]:
    """The structure of every row of the matrices when structures span several rows (k-points), else None."""
    return getattr(basis, "row_struct_t", None)


def _row_weights(basis) -> Optional[Tensor]:
    """The weight of every row (a k-point that stands for itself and its inverse counts twice), else None."""
    return getattr(basis, "row_weight_t", None)


def _fermi(eps: Tensor, n_alpha: Tensor, n_beta: Tensor, kt: float, n_ao: List[int],
           groups: Optional[Tensor] = None, weights: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    """Spin-summed Fermi occupations and ``T S_el`` (hartree) per row. Rows of one group (the k-points of a
    structure) share their Fermi levels, each row's levels counted ``weights`` times; ``n_alpha`` and
    ``n_beta`` are then given per group."""
    if groups is None:
        f, _ = _fermi_channels(eps, n_alpha, n_beta, kt, n_ao)
        return f.sum(1), _electronic_entropy(f, kt)
    # the levels of each group pooled in one row give its Fermi levels, which then occupy every row of it
    owner = groups.tolist()
    members = [[r for r, g in enumerate(owner) if g == k] for k in range(int(max(owner)) + 1)]
    times = [1] * len(owner) if weights is None else [int(round(w)) for w in weights.tolist()]
    counts = [sum(n_ao[r] * times[r] for r in rows) for rows in members]
    pooled = eps.new_full((len(members), max(counts)), float("inf"))
    for k, rows in enumerate(members):
        levels = torch.cat([eps[r, :n_ao[r]].repeat(times[r]) for r in rows])
        pooled[k, :levels.shape[0]] = torch.sort(levels.detach()).values
    _, mu = _fermi_channels(pooled, n_alpha, n_beta, kt, counts)
    valid = torch.arange(eps.shape[-1], device=eps.device)[None, :] < torch.as_tensor(n_ao, device=eps.device)[:, None]
    ee = torch.where(valid, eps, torch.full_like(eps, float("inf")))[:, None, :]
    has = (torch.stack([n_alpha, n_beta], 1) > 0).to(eps.dtype)[groups]
    f = torch.sigmoid(-(ee - mu[groups][:, :, None]) / kt) * has[:, :, None]
    return f.sum(1), _electronic_entropy(f, kt)


def _electronic_entropy(f: Tensor, kt: float) -> Tensor:
    """``T S_el`` (hartree) per structure of the channel occupations ``(B, 2, nb)``."""
    return -kt * (torch.special.xlogy(f, f) + torch.special.xlogy(1.0 - f, 1.0 - f)).sum((-1, -2))


def _fermi_channels(eps: Tensor, n_alpha: Tensor, n_beta: Tensor, kt: float, n_ao: List[int]) -> Tuple[Tensor, Tensor]:
    """Fermi occupations ``(B, 2, nb)`` and levels ``(B, 2)`` of the two spin channels of all
    structures at once; the Fermi levels by Newton steps on the host, bisecting where a step
    leaves the bracket."""
    nb = eps.shape[-1]
    e_host = eps.detach().cpu().numpy()
    n_valid = np.asarray(n_ao)
    valid = np.arange(nb)[None, :] < n_valid[:, None]
    nel = np.stack([n_alpha.detach().cpu().numpy(), n_beta.detach().cpu().numpy()], 1)       # (B, 2)
    if kt <= 0:
        f = ((np.arange(nb)[None, None, :] < nel[:, :, None]) & valid[:, None, :]).astype(np.float64)
        return (torch.as_tensor(f, dtype=eps.dtype, device=eps.device),
                torch.zeros(eps.shape[0], 2, dtype=eps.dtype, device=eps.device))
    e = np.where(valid, e_host, np.inf)[:, None, :]
    lo = np.repeat((np.where(valid, e_host, np.inf).min(-1) - 1.0)[:, None], 2, 1)
    hi = np.repeat((np.where(valid, e_host, -np.inf).max(-1) + 1.0)[:, None], 2, 1)
    # start between the highest occupied and the lowest empty level (the levels are ascending)
    top = np.clip(np.ceil(nel).astype(np.int64), 1, n_valid[:, None])
    rows = np.arange(e_host.shape[0])[:, None]
    below = e_host[rows, top - 1]
    above = e_host[rows, np.minimum(top, n_valid[:, None] - 1)]
    mid = 0.5 * (below + above)
    target = 4.0 * np.finfo(np.float64).eps * np.maximum(nel, 1.0)
    for _ in range(200):
        f = 0.5 * (1.0 - np.tanh(0.5 * (e - mid[:, :, None]) / kt))
        res = f.sum(-1) - nel
        done = (np.abs(res) <= target) | (nel <= 0)
        if done.all():
            break
        over = res > 0
        hi = np.where(over, mid, hi)
        lo = np.where(over, lo, mid)
        slope = (f * (1.0 - f)).sum(-1) / kt
        step = mid - res / np.maximum(slope, 1e-300)
        inside = (step > lo) & (step < hi)
        new = np.where(inside, step, 0.5 * (lo + hi))
        if (done | (new == mid)).all():
            break
        mid = np.where(done, mid, new)
    mu = torch.as_tensor(mid, dtype=eps.dtype, device=eps.device)
    has = torch.as_tensor(nel > 0, device=eps.device).to(eps.dtype)
    ee = torch.where(torch.as_tensor(valid, device=eps.device), eps, torch.full_like(eps, float("inf")))[:, None, :]
    return torch.sigmoid(-(ee - mu[:, :, None]) / kt) * has[:, :, None], mu


#: Levels closer than this (hartree) take the derivative limits of the divided differences.
_NEAR = 1e-8


def _divided1(lam: Tensor, v: Tensor, dv: Tensor) -> Tensor:
    """First divided differences ``[v_i, v_j]`` ``(B, n, n)`` of a function with values ``v`` and derivatives
    ``dv`` at the levels ``lam`` ``(B, n)`` (the mean derivative for close levels)."""
    d = lam[:, :, None] - lam[:, None, :]
    near = d.abs() < _NEAR
    return torch.where(near, 0.5 * (dv[:, :, None] + dv[:, None, :]),
                       (v[:, :, None] - v[:, None, :]) / torch.where(near, torch.ones_like(d), d))


def _divided2(lam: Tensor, v: Tensor, dv: Tensor, d2v: Tensor) -> Tensor:
    """Second divided differences ``[v_i, v_k, v_j]`` ``(B, n, n, n)`` (indices ``i, k, j``), with the limits of
    two and three close levels."""
    d1 = _divided1(lam, v, dv)
    dij = lam[:, :, None, None] - lam[:, None, None, :]
    dik = lam[:, :, None, None] - lam[:, None, :, None]
    far_ij, far_ik = dij.abs() >= _NEAR, dik.abs() >= _NEAR
    general = (d1[:, :, :, None] - d1[:, None, :, :]) / torch.where(far_ij, dij, torch.ones_like(dij))
    pair = (dv[:, :, None, None] - d1[:, :, :, None]) / torch.where(far_ik, dik, torch.ones_like(dik))
    triple = (d2v[:, :, None, None] + d2v[:, None, :, None] + d2v[:, None, None, :]) / 6.0
    return torch.where(far_ij, general, torch.where(far_ik, pair, triple))


def _fermi_derivatives(fs: Tensor, kt: float) -> Tuple[Tensor, Tensor]:
    """The first and second derivatives of Fermi occupations ``fs`` in their level, at a fixed Fermi level."""
    if kt <= 0:
        return torch.zeros_like(fs), torch.zeros_like(fs)
    d1 = -fs * (1.0 - fs) / kt
    return d1, -d1 * (1.0 - 2.0 * fs) / kt


def _sym(M: Tensor) -> Tensor:
    return 0.5 * (M + M.transpose(-1, -2))


def _spectral_gradient(eps: Tensor, U: Tensor, f: Tensor, level: Tensor, kt: float, gP: Optional[Tensor],
                       gW: Optional[Tensor], gts: Optional[Tensor]) -> Tensor:
    """The gradient in ``F`` of ``<gP, P> + <gW, W> + gts T S_el`` (see :class:`_SpectralDensity`)."""
    Ut = U.transpose(-1, -2)
    A = None if gP is None else Ut @ _sym(gP) @ U
    B = None if gW is None else Ut @ _sym(gW) @ U
    gap = eps[:, :, None] - eps[:, None, :]
    near = gap.abs() < _NEAR
    gap = torch.where(near, torch.ones_like(gap), gap)

    def quotient(a: Tensor, slope: Tensor) -> Tensor:
        equal = 0.5 * (slope[:, :, None] + slope[:, None, :])
        return torch.where(near, equal, (a[:, :, None] - a[:, None, :]) / gap)

    M = torch.zeros_like(gap)
    diagonal = torch.zeros_like(eps)
    for s in range(2):
        fs = f[:, s]
        slope = -fs * (1.0 - fs) / kt if kt > 0 else torch.zeros_like(fs)      # df / d eps at a fixed level
        total = slope.sum(-1, keepdim=True)
        share = torch.where(total != 0, slope / torch.where(total != 0, total, torch.ones_like(total)),
                            torch.zeros_like(slope))
        moved = torch.zeros_like(total[:, 0])                                   # gradient in the Fermi level
        if A is not None:
            M = M + quotient(fs, slope) * A
            moved = moved - (A.diagonal(dim1=-2, dim2=-1) * slope).sum(-1)
        if B is not None:
            M = M + quotient(fs * eps, fs + eps * slope) * B
            moved = moved - (B.diagonal(dim1=-2, dim2=-1) * eps * slope).sum(-1)
        if gts is not None:
            entropy = (eps - level[:, s, None]) * slope
            diagonal = diagonal + gts[:, None] * entropy
            moved = moved - gts * entropy.sum(-1)
        diagonal = diagonal + moved[:, None] * share
    return _sym(U @ (M + torch.diag_embed(diagonal)) @ Ut)


def _spectral_second(eps: Tensor, U: Tensor, f: Tensor, level: Tensor, kt: float, gP: Optional[Tensor],
                     gW: Optional[Tensor], gts: Optional[Tensor], gG: Tensor):
    """The derivatives of ``<gG, G>``, ``G`` the gradient of :func:`_spectral_gradient`: in ``F`` (the second
    derivative of ``L = <gP, P> + <gW, W> + gts T S_el`` along ``gG``) and in ``gP``, ``gW``, ``gts`` (the
    first-order changes of ``P``, ``W`` and ``T S_el`` along ``gG``).

    Each spin channel contributes ``L_s(F, mu_s(F))`` with the matrix functions ``a = f(e - mu)``,
    ``b = e f(e - mu)``, the trace of ``t = -kT H(f(e - mu))`` and the Fermi level fixed by
    ``tr f(F - mu) = N``. In the eigenbasis the second derivatives take the second divided differences of
    ``a`` and ``b``, the first of ``t'``, ``df / dmu`` and ``f'``, and the first and second derivatives of
    ``mu``.
    """
    Ut = U.transpose(-1, -2)
    zero = torch.zeros_like(U)
    A = zero if gP is None else Ut @ _sym(gP) @ U
    B = zero if gW is None else Ut @ _sym(gW) @ U
    g = torch.zeros_like(eps[:, 0]) if gts is None else gts.to(eps.dtype)
    E = Ut @ _sym(gG) @ U
    Ad, Bd, Ed = (M.diagonal(dim1=-2, dim2=-1) for M in (A, B, E))
    G = torch.zeros_like(U)
    dP, dW = torch.zeros_like(U), torch.zeros_like(U)
    dT = torch.zeros_like(eps[:, 0])
    for s in range(2):
        fs = f[:, s]
        f1, f2 = _fermi_derivatives(fs, kt)
        x = eps - level[:, s, None]
        tp, tpp = x * f1, f1 + x * f2                                           # t' and t''
        S1 = f1.sum(-1)
        ok = S1 != 0
        inv = torch.where(ok, 1.0 / torch.where(ok, S1, torch.ones_like(S1)), torch.zeros_like(S1))
        share = f1 * inv[:, None]
        dmu = (f1 * Ed).sum(-1) * inv                                           # d mu along gG
        # the first-order changes: the gradients in gP, gW and gts
        a1 = _divided1(eps, fs, f1)
        b1 = _divided1(eps, eps * fs, fs + eps * f1)
        dP = dP + a1 * E - torch.diag_embed(f1 * dmu[:, None])
        dW = dW + b1 * E - torch.diag_embed(eps * f1 * dmu[:, None])
        dT = dT + (tp * (Ed - dmu[:, None])).sum(-1)
        # the second derivative at the fixed Fermi level
        a2 = _divided2(eps, fs, f1, f2)
        b2 = _divided2(eps, eps * fs, fs + eps * f1, 2.0 * f1 + eps * f2)
        X = torch.einsum("bikj,bik,bij->bkj", a2, E, A) + torch.einsum("bikj,bik,bij->bkj", b2, E, B)
        G = G + X + X.transpose(-1, -2) + g[:, None, None] * _divided1(eps, tp, tpp) * E
        # the terms of the moving Fermi level
        K = (_divided1(eps, -f1, -f2) * A + _divided1(eps, -eps * f1, -(f1 + eps * f2)) * B
             - torch.diag_embed(g[:, None] * tpp))
        kE = (K * E).sum((-1, -2))
        L_mu = -(f1 * Ad + eps * f1 * Bd + g[:, None] * tp).sum(-1)
        L_mumu = (f2 * Ad + eps * f2 * Bd + g[:, None] * tpp).sum(-1)
        c_fmu = -(f2 * Ed).sum(-1)
        c_mumu = f2.sum(-1)
        d2mu = (_divided1(eps, f1, f2) * E + torch.diag_embed((c_fmu + c_mumu * dmu)[:, None] * share
                                                               - dmu[:, None] * f2)) * inv[:, None, None]
        moving = (dmu[:, None, None] * K + torch.diag_embed((kE + L_mumu * dmu)[:, None] * share)
                  + L_mu[:, None, None] * d2mu)
        G = G + torch.where(ok[:, None, None], moving, torch.zeros_like(moving))
    rot = lambda M: U @ M @ Ut                                                  # noqa: E731
    return _sym(rot(G)), rot(_sym(dP)), rot(_sym(dW)), dT


class _SpectralDensity(torch.autograd.Function):
    """Density ``U f U^T``, energy-weighted density ``U (f eps) U^T`` and ``T S_el`` of a symmetric
    matrix ``F`` with eigenvalues ``eps``, eigenvectors ``U`` and Fermi occupations ``f`` (two spin
    channels, each with its own level).

    The backward pass is first-order perturbation theory: a spectral function ``a(eps)`` responds to
    ``dF`` with ``U [D * (U^T dF U)] U^T``, ``D_ij = (a_i - a_j) / (eps_i - eps_j)`` (its derivative
    for equal levels), and the Fermi level moves with the levels so that the electron number stays,
    ``d mu = sum_k f'_k d eps_k / sum_k f'_k``. Under ``create_graph`` it is :class:`_SpectralResponse`,
    differentiable once more.
    """

    @staticmethod
    def forward(ctx, F, eps, U, f, level, kt):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(F, eps, U, f, level)
        ctx.kt = kt
        occ = f.sum(1)
        Ut = U.transpose(-1, -2)
        return (U * occ[:, None, :]) @ Ut, (U * (occ * eps)[:, None, :]) @ Ut, _electronic_entropy(f, kt)

    @staticmethod
    def backward(ctx, gP, gW, gts):
        F, eps, U, f, level = ctx.saved_tensors
        if torch.is_grad_enabled():
            g = _SpectralResponse.apply(F, gP, gW, gts, eps, U, f, level, ctx.kt)
        else:
            g = _spectral_gradient(eps, U, f, level, ctx.kt, gP, gW, gts)
        return g, None, None, None, None, None


class _SpectralResponse(torch.autograd.Function):
    """The gradient of :class:`_SpectralDensity` as a function of ``F`` and the incoming gradients, whose
    backward pass is second-order perturbation theory (:func:`_spectral_second`)."""

    @staticmethod
    def forward(ctx, F, gP, gW, gts, eps, U, f, level, kt):
        ctx.save_for_backward(eps, U, f, level, gP, gW, gts)
        ctx.kt = kt
        return _spectral_gradient(eps, U, f, level, kt, gP, gW, gts)

    @staticmethod
    @once_differentiable
    def backward(ctx, gG):
        eps, U, f, level, gP, gW, gts = ctx.saved_tensors
        dF, dP, dW, dT = _spectral_second(eps, U, f, level, ctx.kt, gP, gW, gts, gG)
        need = ctx.needs_input_grad
        return (dF if need[0] else None, dP if need[1] else None, dW if need[2] else None,
                dT if need[3] else None, None, None, None, None, None)


def _all_pairs(batch: Tensor) -> Tuple[Tensor, Tensor]:
    """Atom pairs ``i < j`` within each structure."""
    n = batch.shape[0]
    i, j = torch.triu_indices(n, n, 1, device=batch.device)
    keep = batch[i] == batch[j]
    return i[keep], j[keep]
