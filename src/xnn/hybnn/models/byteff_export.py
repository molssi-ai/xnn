"""The scriptable core of an exported ByteFF-Pol model (see :func:`~xnn.common.deploy.export_torchscript_potential`).

A ByteFF-Pol model predicts the parameters of the system it is bound to once; the export keeps those parameter
rows. For molecules and clusters (every pair, no cell) it evaluates the energy block of the reference
(:class:`~xnn.hybnn.models.byteff._PolEnergy`) with the induced-dipole solve; in periodic cells the terms of the
reference's periodic path from the same functions: the bonded terms and the pairs within 1,5 on minimum-image
vectors, the neighbor pairs, the particle-mesh Ewald sums and the induced dipoles by the reference's
preconditioned conjugate gradients, with the forces of the polarization energy from its variational form at the
solved dipoles (the gradient the reference's adjoint gives).
"""
# no postponed annotations: TorchScript reads the class-level attribute annotations below
import math
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn

from xnn.common.models.ops import cell_volume
from xnn.common.models.pme import spline_moduli
from xnn.ffnn.models.amoeba import _mutual_field_real
from xnn.ffnn.models.amoeba_ewald import (
    _first_stencils,
    _stencil_coefficients,
    _stencil_potential,
    mesh_convolve,
    mesh_influence_of,
    mesh_layout,
    self_field_coefficient,
    vecmat,
)
from xnn.ffnn.models.oplslib import KCAL_TO_EV
from xnn.ffnn.models.terms import solve_dipoles

from .byteff import _fft_size, _pol_local_terms, _pol_real_space, _pol_scaled_terms

#: the parameter rows the energy block takes, in its argument order
_ROWS = ("bond_k", "bond_x0", "angle_k", "angle_x0", "torsion_k", "torsion_sign")
_PAIR_ROWS = ("vdw_scale", "lam", "eps", "r0", "c6", "qq", "ct_eps", "ct_lam", "charge", "alpha", "damp")
_INDEX = ("src", "dst", "b0", "b1", "a1", "t_owner", "p0", "p1")


class ByteFFPolExport(nn.Module):
    """``node_features_energy_charges`` of a bound ByteFF-Pol model: the per-atom energies (eV) of one structure
    of the bound system, its charges, and no features.

    Parameters
    ----------
    model : ByteFFPol
        A bound model that takes every pair (``all_pairs``).

    Raises
    ------
    ValueError
        If the model is not bound, or evaluates periodic cells (a cutoff).
    """

    cutoff: float
    n_atoms: int
    coulomb: float
    kcal_to_ev: float
    sizes: List[int]

    def __init__(self, model: nn.Module):
        super().__init__()
        if getattr(model, "atomic_numbers", None) is None:
            raise ValueError("bind the model to its system before exporting it")
        if not model.all_pairs:
            raise ValueError("a model with a cutoff evaluates periodic cells: export it with ByteFFPolCellExport")
        dtype = next(p.dtype for p in model.parameters() if p.is_floating_point())
        device = model.atomic_numbers.device
        with torch.no_grad():
            R = model._rows(model._parameters_cached(), 1, dtype)
            plan = model._plan(1, device)
        self.cutoff = float(model.cutoff)
        self.n_atoms = int(model.n_atoms)
        self.coulomb = float(model.coulomb_constant)
        self.kcal_to_ev = float(KCAL_TO_EV)
        self.sizes = [int(k) for k in plan["sizes"]]
        self.energy = model._energy
        for name in _ROWS + _PAIR_ROWS:
            self.register_buffer(name, R[name].detach().clone())
        b, a, p = plan["bonds"], plan["angles"], plan["pairs"]
        index = {"src": plan["src"], "dst": plan["dst"], "b0": b[0], "b1": b[1], "a1": a[1],
                 "t_owner": plan["t_owner"], "p0": p[0], "p1": p[1]}
        for name in _INDEX:
            self.register_buffer(name, index[name].clone())
        self.register_buffer("improper", plan["improper"].to(dtype))
        self.register_buffer("field_scale", model._field_scale.to(dtype).clone())
        self.register_buffer("atomic_numbers", model.atomic_numbers.clone())

    def node_features_energy_charges(self, atomic_numbers: Tensor, edge_index: Tensor, edge_vec: Tensor,
                                     pos: Tensor, cell: Tensor, pbc: Tensor, total_charge: float,
                                     spin_multiplicity: float) -> Tuple[Tensor, Tensor, Tensor]:
        """The features ``(N, 0)``, the per-atom energies ``(N,)`` in eV and the charges ``(N,)`` of the bound
        system at ``pos``; the neighbor list is not used (every pair interacts).

        Raises
        ------
        RuntimeError
            If the atoms are not those of the bound system in its order, or the structure is periodic.
        """
        if atomic_numbers.shape[0] != self.n_atoms or not bool(torch.equal(atomic_numbers, self.atomic_numbers)):
            raise RuntimeError("the atoms differ from those of the system the model was bound to")
        if bool(pbc.any()):
            raise RuntimeError("the exported ByteFF-Pol model evaluates molecules and clusters, not cells")
        e, A, field = self.energy._reference(
            pos, self.bond_k, self.bond_x0, self.angle_k, self.angle_x0, self.torsion_k, self.torsion_sign,
            self.improper, self.vdw_scale, self.lam, self.eps, self.r0, self.c6, self.qq, self.ct_eps, self.ct_lam,
            self.charge, self.alpha, self.damp, self.field_scale, self.src, self.dst, self.b0, self.b1, self.a1,
            self.t_owner, self.p0, self.p1, self.sizes, 1, self.n_atoms)
        mu = solve_dipoles(A, field)
        polarization = -0.5 * self.coulomb * (mu * field).sum(-1).reshape(-1)
        node = (e.sum(1) + polarization) * self.kcal_to_ev
        return torch.zeros((self.n_atoms, 0), dtype=pos.dtype, device=pos.device), node, self.charge


#: Debye per e Angstrom, the unit of the induction convergence test (as in the reference)
_DEBYE = 4.80320427


def _minimum_image(vec: Tensor, cell: Tensor) -> Tensor:
    """Minimum-image vectors ``(M, 3)`` in one periodic cell: the rounded fractional shift, then the shortest of
    the 27 images around it (the rule of :func:`~xnn.ffnn.models.geometry.pair_vectors`), with the shifts fixed
    and the gradient flowing to the positions and the cell."""
    inv = torch.linalg.inv(cell.detach())
    shift = -torch.round(vec.detach() @ inv)
    one = torch.arange(-1, 2, device=vec.device, dtype=vec.dtype)
    offsets = torch.cartesian_prod(one, one, one)                                   # (27, 3)
    cand = vec.detach()[:, None, :] + (shift[:, None, :] + offsets[None]) @ cell.detach()
    best = (cand * cand).sum(-1).argmin(1)
    shift = shift + offsets.index_select(0, best)
    return vec + shift @ cell


class ByteFFPolCellExport(nn.Module):
    """``node_features_energy_charges`` of a bound ByteFF-Pol model with a cutoff: the per-atom energies (eV) of
    one periodic cell of the bound system, its charges, and no features. The neighbor list of the caller (within
    the cutoff) gives the pairs; the mesh follows the model's rule for the cell of every call (or its fixed
    ``pme_grid``).

    Parameters
    ----------
    model : ByteFFPol
        A bound model with a cutoff (``all_pairs: false``).

    Raises
    ------
    ValueError
        If the model is not bound, or takes every pair.
    """

    cutoff: float
    n_atoms: int
    coulomb: float
    kcal_to_ev: float
    thole: float
    prefactor: float
    wall: float
    damping: float
    alpha_e: float
    tolerance: float
    order: int
    polar_eps: float
    max_iter: int
    debye: float
    grid: List[int]
    sizes: List[int]
    has_tails: bool

    def __init__(self, model: nn.Module):
        super().__init__()
        if getattr(model, "atomic_numbers", None) is None:
            raise ValueError("bind the model to its system before exporting it")
        if model.all_pairs:
            raise ValueError("a model that takes every pair evaluates molecules and clusters: export it with "
                             "ByteFFPolExport")
        dtype = next(p.dtype for p in model.parameters() if p.is_floating_point())
        device = model.atomic_numbers.device
        with torch.no_grad():
            R = model._rows(model._parameters_cached(), 1, dtype)
            plan = model._plan(1, device)
        rc = float(model.cutoff)
        tol = float(model.ewald_tolerance)
        self.cutoff, self.n_atoms = rc, int(model.n_atoms)
        self.coulomb, self.kcal_to_ev, self.thole = float(model.coulomb_constant), float(KCAL_TO_EV), float(model.thole)
        self.prefactor, self.wall = float(model.repulsion_prefactor), float(model.wall)
        self.damping = float(model.dispersion_damping)
        self.alpha_e = float(model.ewald_alpha) if model.ewald_alpha is not None else \
            math.sqrt(-math.log(2.0 * tol)) / rc
        self.tolerance, self.order = tol, int(model.pme_order)
        self.polar_eps, self.max_iter, self.debye = float(model.polar_eps), int(model.max_iter), _DEBYE
        self.grid = [] if model.pme_grid is None else [int(k) for k in model.pme_grid]
        self.sizes = [int(k) for k in plan["sizes"]]
        b, a, p = plan["bonds"], plan["angles"], plan["pairs"]
        rows = {name: R[name] for name in ("bond_k", "bond_x0", "angle_k", "angle_x0", "torsion_k", "torsion_sign",
                                           "vdw_scale", "eps", "lam", "r0", "c6", "charge", "damp", "alpha")}
        rows["atoms"] = torch.stack([R["charge"], R["damp"]] + [R["atom_" + k] for k in
                                                                ("eps", "lam", "r0", "c6", "ct_eps", "ct_lam")], 1)
        self.has_tails = "tails" in R
        rows["tails"] = R["tails"] if self.has_tails else torch.zeros(3, dtype=dtype, device=device)
        cm, cs = model._corr_scales.to(dtype)
        rows.update(cm=cm, cs=cs, improper=plan["improper"].to(dtype))
        for name, value in rows.items():
            self.register_buffer(name, value.detach().clone())
        n = self.n_atoms
        ex = model._excluded
        index = {"src": plan["src"], "dst": plan["dst"], "b0": b[0], "b1": b[1], "a1": a[1], "t_owner": plan["t_owner"],
                 "p0": p[0], "p1": p[1], "ci": model._corrected[0], "cj": model._corrected[1],
                 "excluded": torch.sort(torch.minimum(ex[0], ex[1]) * n + torch.maximum(ex[0], ex[1])).values,
                 "atomic_numbers": model.atomic_numbers}
        for name, value in index.items():
            self.register_buffer(name, value.clone())

    def _mesh(self, cell: Tensor, dtype: torch.dtype) -> List[int]:
        """The mesh of a cell: the fixed ``pme_grid``, or the model's rule (OpenMM's GPU sizes)."""
        if len(self.grid) == 3:
            return self.grid
        mesh: List[int] = []
        for i in range(3):
            k = int(math.ceil(2.0 * self.alpha_e * float(cell[i, i]) / (3.0 * self.tolerance ** 0.2)))
            mesh.append(_fft_size(max(6, k)))
        return mesh

    def node_features_energy_charges(self, atomic_numbers: Tensor, edge_index: Tensor, edge_vec: Tensor,
                                     pos: Tensor, cell: Tensor, pbc: Tensor, total_charge: float,
                                     spin_multiplicity: float) -> Tuple[Tensor, Tensor, Tensor]:
        """The features ``(N, 0)``, the per-atom energies ``(N,)`` in eV and the charges ``(N,)`` of one cell of
        the bound system, from the neighbor pairs ``edge_index`` with their vectors ``edge_vec``.

        Raises
        ------
        RuntimeError
            If the atoms are not those of the bound system in its order, or the cell is not periodic along all
            three axes or narrower than twice the cutoff.
        """
        N = self.n_atoms
        if atomic_numbers.shape[0] != N or not bool(torch.equal(atomic_numbers, self.atomic_numbers)):
            raise RuntimeError("the atoms differ from those of the system the model was bound to")
        if not bool(pbc.all()):
            raise RuntimeError("the exported ByteFF-Pol model evaluates cells periodic along all three axes")
        dtype = pos.dtype
        C, rc = self.coulomb, self.cutoff
        cross = torch.stack([torch.linalg.cross(cell[1], cell[2]), torch.linalg.cross(cell[2], cell[0]),
                             torch.linalg.cross(cell[0], cell[1])])
        volume = cell_volume(cell[None])
        if bool((volume[0] / torch.linalg.norm(cross, dim=-1) < 2.0 * rc - 1e-9).any()):
            raise RuntimeError("every cell must be at least twice the cutoff wide")

        # bonded terms and the fixed 1,4 and 1,5 pairs on minimum-image vectors
        vec = _minimum_image(pos.index_select(0, self.dst) - pos.index_select(0, self.src), cell)
        local = _pol_local_terms(vec, self.sizes, self.bond_k, self.bond_x0, self.angle_k, self.angle_x0,
                                 self.torsion_k, self.torsion_sign, self.improper, self.vdw_scale, self.eps,
                                 self.lam, self.r0, self.c6, self.b0, self.b1, self.a1, self.t_owner, self.p0, self.p1,
                                 N, self.prefactor, self.wall, self.damping)
        rep, disp = local[:, 4], local[:, 5]

        # the neighbor pairs once each; the short-range terms skip those within 1,5 (every image of them)
        once = (edge_index[0] < edge_index[1]).nonzero()[:, 0]
        src, dst = edge_index[0].index_select(0, once), edge_index[1].index_select(0, once)
        vn = edge_vec.index_select(0, once)
        key = torch.minimum(src, dst) * N + torch.maximum(src, dst)
        if self.excluded.shape[0] > 0:
            at = torch.searchsorted(self.excluded, key).clamp(max=self.excluded.shape[0] - 1)
            beyond = (self.excluded.index_select(0, at) != key).to(dtype)
        else:
            beyond = torch.ones_like(vn[:, 0])
        valid = torch.ones_like(vn[:, 0])
        q, damp = self.charge, self.damp
        pairs, field, D1u, D2u = _pol_real_space(vn, valid, beyond, src, dst, self.atoms, N, rc * rc, self.alpha_e,
                                                 self.thole, C, self.prefactor, self.wall, self.damping)
        rep, disp = rep + pairs[:, 0], disp + pairs[:, 1]
        ct, elec = pairs[:, 2], pairs[:, 3]

        # the pairs within 1,5: charge scale and field scale less one
        vc = _minimum_image(pos.index_select(0, self.cj) - pos.index_select(0, self.ci), cell)
        scaled = _pol_scaled_terms(vc, self.ci, self.cj, self.cm, self.cs, q, damp, N, C, self.thole)
        elec, field = elec + scaled[:, 0], field + scaled[:, 1:]

        # the mesh: reciprocal energy and field of the charges, and the self energy
        mesh = self._mesh(cell, dtype)
        sizes = torch.tensor(mesh, dtype=dtype, device=pos.device)
        inv = torch.linalg.inv_ex(cell[None])[0]
        batch = torch.zeros(N, dtype=torch.long, device=pos.device)
        inv_n = inv.index_select(0, batch)
        jac = inv_n * sizes
        w, dw, d2w, flat, start = mesh_layout(vecmat(pos, inv_n) * sizes, sizes, batch, mesh, self.order)
        stencils = _first_stencils(w, dw)
        moduli = spline_moduli(mesh, self.order, dtype, pos.device, least_squares=True)
        influence = mesh_influence_of(inv, volume, mesh, moduli, self.alpha_e, "coulomb")
        at_q = mesh_convolve(_stencil_coefficients(q, None, None, jac, stencils), flat, influence, 1, mesh)
        phi, dphi, _ = _stencil_potential(at_q, jac, stencils, False)
        elec, field = elec + 0.5 * C * q * phi, field - dphi
        elec = elec - C * self.alpha_e / math.sqrt(math.pi) * q * q

        # the induced dipoles by the reference's conjugate gradients (no gradient), then the polarization
        # energy -C/2 mu.E with the gradient of its variational form C (-mu.E + mu.A mu / 2) at those dipoles
        alpha = self.alpha
        polar = alpha > 0
        coef = float(self_field_coefficient(torch.tensor(self.alpha_e, dtype=torch.float64)))
        ik, ki = torch.cat([src, dst]), torch.cat([dst, src])
        with torch.no_grad():
            mu = self._solve(field.detach(), alpha, polar, vn.detach(), D1u.detach(), D2u.detach(), ik, ki,
                             jac.detach(), {k: v.detach() for k, v in stencils.items()}, flat,
                             influence.detach(), mesh, coef)
        at_d = mesh_convolve(_stencil_coefficients(None, mu, None, jac, stencils), flat, influence, 1, mesh)
        _, dphi_d, _ = _stencil_potential(at_d, jac, stencils, False)
        mutual = _mutual_field_real(mu, vn, D1u, D2u, ik, ki, N) + (-dphi_d + coef * mu)
        inv_alpha = torch.where(polar, 1.0 / alpha.clamp_min(1e-30), torch.zeros_like(alpha))[:, None]
        variational = C * (-(mu * field).sum(1) + 0.5 * (mu * (mu * inv_alpha - mutual)).sum(1))
        reported = -0.5 * C * (mu * field).sum(1)
        polarization = variational + (reported - variational).detach()

        # the tails beyond the cutoff
        if self.has_tails:
            tail = (4.0 * math.pi * N / (N + 1.0)) * self.tails / volume[0] / N
            rep, disp, ct = rep + tail[0], disp + tail[1], ct + tail[2]
        # the terms in the reference's order
        node = local[:, 0] + local[:, 1] + local[:, 2] + local[:, 3] + polarization + rep + disp + elec + ct
        return torch.zeros((N, 0), dtype=dtype, device=pos.device), node * self.kcal_to_ev, q

    def _solve(self, rhs: Tensor, alpha: Tensor, polar: Tensor, r: Tensor, D1: Tensor, D2: Tensor, ik: Tensor,
               ki: Tensor, jac: Tensor, stencils: Dict[str, Tensor], flat: Tensor, influence: Tensor,
               mesh: List[int], coef: float) -> Tensor:
        """``(1/alpha - T) mu = rhs`` on the polarizable atoms by the reference's preconditioned conjugate
        gradients (the polarizabilities as the preconditioner, a zero start), converged when the rms change of the
        dipoles per iteration drops below ``polar_eps`` Debye."""
        N = self.n_atoms
        mask = polar[:, None].to(rhs.dtype)
        n_polar = max(int(polar.sum()), 1)
        inv_alpha = torch.where(polar, 1.0 / alpha.clamp_min(1e-30), torch.zeros_like(alpha))[:, None]
        diag = alpha[:, None] * mask
        b = rhs * mask
        x = torch.zeros_like(b)
        res = b
        z = diag * res
        p = z
        rz = (res * z).sum()
        for _ in range(self.max_iter):
            at = mesh_convolve(_stencil_coefficients(None, p, None, jac, stencils), flat, influence, 1, mesh)
            _, dphi, _ = _stencil_potential(at, jac, stencils, False)
            field = _mutual_field_real(p, r, D1, D2, ik, ki, N) + (-dphi + coef * p)
            ap = (p * inv_alpha - field) * mask
            denom = (p * ap).sum()
            a = torch.where(denom != 0, rz / torch.where(denom != 0, denom, torch.ones_like(denom)),
                            torch.zeros_like(denom))
            step = a * p
            x = x + step
            res = res - a * ap
            z = diag * res
            rz_new = (res * z).sum()
            if math.sqrt(float((step * step).sum()) / n_polar) * self.debye < self.polar_eps:
                break
            p = z + (rz_new / torch.where(rz != 0, rz, torch.ones_like(rz))) * p
            rz = rz_new
        return x
